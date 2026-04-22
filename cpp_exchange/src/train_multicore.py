import os
import torch
import numpy as np
import multiprocessing as mp
from environment import TradingEnvironment
from agent import RLAgent

# --- HYPERPARAMETERS ---
TOTAL_EPISODES = 2000
RISK_AVERSION_LAMBDA = 0.05
NUM_CORES = 10  # Max out 10 processors, leave 2 for the Windows OS

def calculate_entropic_utility(raw_pnl, lambda_val):
    scaled_pnl = raw_pnl / 1000.0
    clipped_pnl = np.clip(scaled_pnl, -100.0, 100.0)
    exponent = np.clip(-lambda_val * clipped_pnl, -40.0, 40.0)
    return float((1.0 - np.exp(exponent)) / lambda_val)

def worker_process(worker_id):
    """
    This function runs completely isolated on a single CPU core.
    """
    # CRITICAL: Force this specific core to only use 1 PyTorch thread 
    # to prevent the 10 processes from thrashing each other.
    torch.set_num_threads(1) 
    
    env = TradingEnvironment()
    agent = RLAgent()
    
    # Give this core its own isolated memory file
    weights_path = f"weights_core_{worker_id}.pth"
    
    if os.path.exists(weights_path):
        try:
            agent.policy_network.load_state_dict(torch.load(weights_path))
        except Exception:
            pass # Start fresh if no valid weights are found
            
    history_pnl = []
    
    # --- ISOLATED TRAINING LOOP ---
    for episode in range(1, TOTAL_EPISODES + 1):
        state = env.reset()
        episode_log_probs, episode_entropies, episode_rewards = [], [], []
        
        while True:
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            episode_log_probs.append(log_prob)
            episode_entropies.append(entropy)
            episode_rewards.append(reward)
            state = next_state
            
            if done:
                terminal_pnl = env.cash + (env.holdings * env.current_price) - env.starting_cash
                history_pnl.append(terminal_pnl)
                break
                
        # Calculate Discounted Reward-To-Go
        returns = []
        G = 0
        gamma = 0.99
        for r in reversed(episode_rewards):
            u = calculate_entropic_utility(r, RISK_AVERSION_LAMBDA)
            G = u + (gamma * G)
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Backpropagation
        policy_loss = []
        for log_prob, G in zip(episode_log_probs, returns):
            policy_loss.append(-log_prob * G)
        loss = torch.stack(policy_loss).sum()
        
        entropy_bonus = torch.stack(episode_entropies).sum()
        final_loss = loss - (0.01 * entropy_bonus)
        
        agent.optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy_network.parameters(), max_norm=1.0)
        agent.optimizer.step()
        
        # Dynamic Saving per Core
        if episode % 500 == 0:
            torch.save(agent.policy_network.state_dict(), weights_path)
            
        if episode % 100 == 0:
            print(f"[Core {worker_id}] Episode {episode:04d} | PnL: ${terminal_pnl:8.2f}")

    # Final Save
    torch.save(agent.policy_network.state_dict(), weights_path)
    
    # Return the average PnL of the last 100 episodes to the Master to grade the AI
    final_score = np.mean(history_pnl[-100:])
    return worker_id, final_score

if __name__ == '__main__':
    # REQUIRED FOR WINDOWS MULTIPROCESSING
    mp.set_start_method('spawn')
    
    print(f"Launching Vectorized Seed Sweep across {NUM_CORES} CPU Cores...")
    print("Open Task Manager - CPU Utilization is about to skyrocket!\n")
    
    # Launch the parallel architecture
    with mp.Pool(processes=NUM_CORES) as pool:
        # Map the worker function to core IDs 0 through 9
        results = pool.map(worker_process, range(NUM_CORES))
        
    print("\n--- All Cores Finished ---")
    
    # Find the winning AI
    best_core = -1
    best_score = -float('inf')
    
    for worker_id, score in results:
        print(f"Core {worker_id} Final Avg PnL (Last 100 Eps): ${score:8.2f}")
        if score > best_score:
            best_score = score
            best_core = worker_id
            
    print(f"\n[*] WINNER: Core {best_core} produced the smartest AI with an Average PnL of ${best_score:8.2f}")
    print(f"[*] Action Required: Rename 'weights_core_{best_core}.pth' to 'deep_hedging_weights.pth' to use it in your test.py script.")