import torch
import os
import numpy as np
from environment import TradingEnvironment
from agent import RLAgent

torch.set_num_threads(1)
# --- HYPERPARAMETERS ---
TOTAL_EPISODES = 2000         # Start small for Phase 1 (Sandbox Testing)
RISK_AVERSION_LAMBDA = 0.05  # The 'Fear' parameter for Entropic Risk

def calculate_entropic_utility(raw_pnl, lambda_val):
    """
    Transforms raw PnL into Risk-Adjusted Utility.
    Uses the normalized formula to ensure positive rewards for profit 
    and exponential penalties for drawdowns.
    """
    # 1. Scale down the PnL so the exponent doesn't explode
    scaled_pnl = raw_pnl / 1000.0
    
    # 2. Cap the PnL to prevent math overflow
    clipped_pnl = np.clip(scaled_pnl, -100.0, 100.0)
    
    # 3. The Normalized Entropic Formula
    exponent = np.clip(-lambda_val * clipped_pnl, -40.0, 40.0)
    utility = (1.0 - np.exp(exponent)) / lambda_val
    
    return float(utility)

def train_agent():
    print("Initializing the Deep Hedging Training Loop...")
    env = TradingEnvironment()
    agent = RLAgent()
    
    weights_path = "deep_hedging_weights.pth"
    
    # --- NEW: THE MEMORY INJECTION ---
    if os.path.exists(weights_path):
        print(f"[*] Found existing brain at '{weights_path}'")
        try:
            # Load the saved state dictionary into the neural network
            agent.policy_network.load_state_dict(torch.load(weights_path))
            print("[*] Successfully restored previous weights. Resuming training...")
        except Exception as e:
            print(f"[!] Error loading weights: {e}. Starting with a blank brain.")
    else:
        print("[*] No existing weights found. Training a blank brain from scratch.")
        
    history_pnl = []
    
    for episode in range(1, TOTAL_EPISODES + 1):
        state = env.reset()
        
        episode_log_probs = []
        episode_entropies = []
        episode_rewards = [] # NEW: Track every single step's reward
        
        # --- 1. PLAY THE EPISODE ---
        while True:
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            episode_log_probs.append(log_prob)
            episode_entropies.append(entropy)
            episode_rewards.append(reward) # Save the dense tick reward
            
            state = next_state
            
            if done:
                # Calculate terminal PnL just for logging purposes
                terminal_pnl = env.cash + (env.holdings * env.current_price) - env.starting_cash
                break
                
        # --- 2. CALCULATE DISCOUNTED REWARD-TO-GO ---
        returns = []
        G = 0
        gamma = 0.99 # Discount factor
        
        for r in reversed(episode_rewards):
            # Apply the Entropic Risk formula to the individual step
            u = calculate_entropic_utility(r, RISK_AVERSION_LAMBDA)
            G = u + (gamma * G)
            returns.insert(0, G)
            
        returns = torch.tensor(returns)
        
        # Normalize the returns tensor to mathematically stabilize gradients
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # --- 3. REINFORCE BACKPROPAGATION ---
        # --- 3. DYNAMIC SAVING ---
        # Save a checkpoint every 500 episodes just in case the node crashes
        if episode % 500 == 0:
            torch.save(agent.policy_network.state_dict(), weights_path)
            print(f"[*] Checkpoint saved at Episode {episode}")

        policy_loss = []
        for log_prob, G in zip(episode_log_probs, returns):
            policy_loss.append(-log_prob * G)
            
        loss = torch.stack(policy_loss).sum()
        
        # The Entropy Bonus (forces exploration)
        entropy_bonus = torch.stack(episode_entropies).sum()
        entropy_coefficient = 0.01 
        final_loss = loss - (entropy_coefficient * entropy_bonus)
        
        agent.optimizer.zero_grad()  
        final_loss.backward()     
        torch.nn.utils.clip_grad_norm_(agent.policy_network.parameters(), max_norm=1.0)         
        agent.optimizer.step()       
        
        # --- 4. LOGGING ---
        history_pnl.append(terminal_pnl)
        
        if episode % 100 == 0 or episode == 1:
            print(f"Episode {episode:04d} | Terminal PnL: ${terminal_pnl:8.2f} | Loss: {final_loss.item():8.2f}")
        
    print("\n--- Training Complete ---")
    torch.save(agent.policy_network.state_dict(), weights_path)
    print("Training Complete. Final model weights saved.")
if __name__ == "__main__":
    train_agent()