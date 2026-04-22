import os
import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from environment import TradingEnvironment
from agent import PPOAgent # Ensure you use the new PPOAgent!

# --- HYPERPARAMETERS ---
TOTAL_EPISODES = 4000
NUM_CORES = 10 
RISK_AVERSION_LAMBDA = 0.05

# PPO Specific Hyperparameters
UPDATE_TIMESTEPS = 2000  # Update the brain every 2000 ticks
PPO_EPOCHS = 4           # Study the batch 4 times
CLIP_EPSILON = 0.2       # Maximum 20% brain change per update
GAMMA = 0.99             # Discount factor
GAE_LAMBDA = 0.95        # Smoothing factor for Advantage

def calculate_entropic_utility(raw_pnl, lambda_val):
    scaled_pnl = raw_pnl / 1000.0
    clipped_pnl = np.clip(scaled_pnl, -100.0, 100.0)
    exponent = np.clip(-lambda_val * clipped_pnl, -40.0, 40.0)
    return float((1.0 - np.exp(exponent)) / lambda_val)

def worker_process(worker_id):
    torch.set_num_threads(1) 
    env = TradingEnvironment()
    agent = PPOAgent(learning_rate=3e-4)
    
    weights_path = f"weights_core_{worker_id}.pth"
    if os.path.exists(weights_path):
        try: agent.policy_network.load_state_dict(torch.load(weights_path))
        except: pass 
            
    history_pnl = []
    
    # Rollout Buffer variables
    buffer_states, buffer_actions, buffer_logprobs = [], [], []
    buffer_rewards, buffer_state_values, buffer_dones = [], [], []
    time_step = 0
    
    for episode in range(1, TOTAL_EPISODES + 1):
        state = env.reset()
        
        while True:
            time_step += 1
            
            # 1. Actor picks action, Critic predicts value
            action, log_prob, _, state_value = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Convert raw PnL to Risk-Adjusted Utility
            utility = calculate_entropic_utility(reward, RISK_AVERSION_LAMBDA)
            
            # 2. Store to Rollout Buffer
            buffer_states.append(state)
            buffer_actions.append(action)
            buffer_logprobs.append(log_prob)
            buffer_state_values.append(state_value)
            buffer_rewards.append(utility)
            buffer_dones.append(done)
            
            state = next_state
            
            # 3. PPO Update Phase (Triggered every UPDATE_TIMESTEPS)
            if time_step % UPDATE_TIMESTEPS == 0:
                # Get the value of the final state to cap off the math
                _, _, _, next_state_value = agent.select_action(state)
                
                # --- CALCULATE GAE (Advantage) ---
                advantages = []
                gae = 0
                for step in reversed(range(len(buffer_rewards))):
                    if step == len(buffer_rewards) - 1:
                        next_val = next_state_value.item()
                    else:
                        next_val = buffer_state_values[step + 1].item()
                        
                    delta = buffer_rewards[step] + GAMMA * next_val * (1 - buffer_dones[step]) - buffer_state_values[step].item()
                    gae = delta + GAMMA * GAE_LAMBDA * (1 - buffer_dones[step]) * gae
                    advantages.insert(0, gae)
                
                # Convert buffers to PyTorch Tensors
                old_states = torch.FloatTensor(np.array(buffer_states))
                old_actions = torch.FloatTensor(buffer_actions)
                old_logprobs = torch.stack(buffer_logprobs).detach()
                old_state_values = torch.stack(buffer_state_values).detach()
                advantages_tensor = torch.FloatTensor(advantages).detach()
                
                # Returns = Advantage + Critic's Prediction
                returns = advantages_tensor + old_state_values
                
                # Normalize advantages for mathematical stability
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
                
                # --- PPO EPOCH LOOP ---
                for _ in range(PPO_EPOCHS):
                    # Evaluate old actions using the updated network
                    logprobs, state_values, dist_entropy = agent.evaluate(old_states, old_actions)
                    
                    # Calculate the ratio (pi_theta / pi_theta_old)
                    ratios = torch.exp(logprobs - old_logprobs)
                    
                    # Calculate Surrogate Loss 1 & 2
                    surr1 = ratios * advantages_tensor
                    surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages_tensor
                    
                    # Actor Loss (Maximize advantage)
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Critic Loss (Minimize prediction error)
                    critic_loss = nn.MSELoss()(state_values, returns)
                    
                    # Final Loss = Actor Loss + 0.5 * Critic Loss - 0.01 * Entropy (Exploration)
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
                    
                    # Backpropagation
                    agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.policy_network.parameters(), max_norm=0.5)
                    agent.optimizer.step()
                
                # Clear the buffer after learning
                buffer_states, buffer_actions, buffer_logprobs = [], [], []
                buffer_rewards, buffer_state_values, buffer_dones = [], [], []

            if done:
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
    print(f"Launching PPO Vectorized Ensemble across {NUM_CORES} CPU Cores...")
    
    with mp.Pool(processes=NUM_CORES) as pool:
        results = pool.map(worker_process, range(NUM_CORES))
        
    print("\n--- All Cores Finished PPO Training ---")
    best_core, best_score = -1, -float('inf')
    
    for worker_id, score in results:
        print(f"Core {worker_id} Final Avg PnL (Last 100 Eps): ${score:8.2f}")
        if score > best_score:
            best_score = score; best_core = worker_id
            
    print(f"\n[*] WINNER: Core {best_core} (Score: ${best_score:8.2f})")
    print(f"[*] Rename 'weights_core_{best_core}.pth' to 'deep_hedging_weights.pth' for evaluation.")