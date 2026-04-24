import os
import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from environment import TradingEnvironment
from agent import PPOAgentLSTM # Using the new LSTM Brain!

# --- HYPERPARAMETERS ---
TOTAL_EPISODES = 4000
NUM_CORES = 10 
RISK_AVERSION_LAMBDA = 0.05
PPO_EPOCHS = 4           
CLIP_EPSILON = 0.2       
GAMMA = 0.99             
GAE_LAMBDA = 0.95        

def calculate_entropic_utility(raw_pnl, lambda_val):
    scaled_pnl = raw_pnl / 1000.0
    clipped_pnl = np.clip(scaled_pnl, -100.0, 100.0)
    exponent = np.clip(-lambda_val * clipped_pnl, -40.0, 40.0)
    return float((1.0 - np.exp(exponent)) / lambda_val)

def worker_process(worker_id):
    torch.set_num_threads(1) 
    env = TradingEnvironment()
    agent = PPOAgentLSTM(learning_rate=3e-4)
    
    weights_path = f"weights_core_{worker_id}.pth"
    if os.path.exists(weights_path):
        try: agent.policy_network.load_state_dict(torch.load(weights_path))
        except: pass 
            
    history_pnl = []
    
    for episode in range(1, TOTAL_EPISODES + 1):
        state = env.reset()
        
        # THE FIX: Wipe the LSTM's memory clean at the start of every episode
        hidden = agent.get_initial_hidden()
        
        buffer_states, buffer_actions, buffer_logprobs = [], [], []
        buffer_rewards, buffer_state_values = [], []
        
        while True:
            # The AI reads the state AND its own memory
            action, log_prob, _, state_value, hidden = agent.select_action(state, hidden)
            next_state, reward, done = env.step(action)
            
            utility = calculate_entropic_utility(reward, RISK_AVERSION_LAMBDA)
            
            buffer_states.append(state)
            buffer_actions.append(action)
            buffer_logprobs.append(log_prob)
            buffer_state_values.append(state_value)
            buffer_rewards.append(utility)
            
            state = next_state
            
            # We only perform PPO Math at the EXACT end of an episode
            if done:
                _, _, _, next_state_value, _ = agent.select_action(state, hidden)
                
                advantages = []
                gae = 0
                for step in reversed(range(len(buffer_rewards))):
                    if step == len(buffer_rewards) - 1:
                        next_val = next_state_value.item()
                        is_terminal = 1 # End of the simulation
                    else:
                        next_val = buffer_state_values[step + 1].item()
                        is_terminal = 0 # Middle of the sequence
                        
                    delta = buffer_rewards[step] + GAMMA * next_val * (1 - is_terminal) - buffer_state_values[step].item()
                    gae = delta + GAMMA * GAE_LAMBDA * (1 - is_terminal) * gae
                    advantages.insert(0, gae)
                
                # Format exactly for the LSTM: [1 Batch, 1000 Sequence Steps, 3 Features]
                old_states = torch.FloatTensor(np.array(buffer_states)).unsqueeze(0) 
                old_actions = buffer_actions
                old_logprobs = torch.stack(buffer_logprobs).detach()
                old_state_values = torch.stack(buffer_state_values).detach()
                advantages_tensor = torch.FloatTensor(advantages).detach()
                
                returns = advantages_tensor + old_state_values
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
                
                # Backpropagate through time
                initial_hidden_for_update = agent.get_initial_hidden()
                
                for _ in range(PPO_EPOCHS):
                    logprobs, state_values, dist_entropy = agent.evaluate(old_states, old_actions, initial_hidden_for_update)
                    
                    ratios = torch.exp(logprobs - old_logprobs)
                    surr1 = ratios * advantages_tensor
                    surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages_tensor
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(state_values, returns)
                    
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
                    
                    agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.policy_network.parameters(), max_norm=0.5)
                    agent.optimizer.step()
                
                terminal_pnl = env.cash + (env.holdings * env.current_price) - env.starting_cash
                history_pnl.append(terminal_pnl)
                break
                
        if episode % 500 == 0:
            torch.save(agent.policy_network.state_dict(), weights_path)
        if episode % 50 == 0:
            print(f"[Core {worker_id}] Episode {episode:04d} | PnL: ${terminal_pnl:8.2f}")

    torch.save(agent.policy_network.state_dict(), weights_path)
    return worker_id, np.mean(history_pnl[-100:])

if __name__ == '__main__':
    mp.set_start_method('spawn')
    print(f"Launching Asynchronous PPO-LSTM Ensemble across {NUM_CORES} Cores...")
    
    with mp.Pool(processes=NUM_CORES) as pool:
        results = pool.map(worker_process, range(NUM_CORES))
        
    print("\n--- All Cores Finished PPO Training ---")
    best_core, best_score = -1, -float('inf')
    
    for worker_id, score in results:
        print(f"Core {worker_id} Final Avg PnL (Last 100 Eps): ${score:8.2f}")
        if score > best_score:
            best_score = score; best_core = worker_id
            
    print(f"\n[*] WINNER: Core {best_core} (Score: ${best_score:8.2f})")