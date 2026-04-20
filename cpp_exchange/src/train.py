import torch
import numpy as np
from environment import TradingEnvironment
from agent import RLAgent

# --- HYPERPARAMETERS ---
TOTAL_EPISODES = 5000         # Start small for Phase 1 (Sandbox Testing)
RISK_AVERSION_LAMBDA = 0.05  # The 'Fear' parameter for Entropic Risk

def calculate_entropic_utility(raw_pnl, lambda_val):
    """
    Transforms raw PnL into Risk-Adjusted Utility with Float Overflow Protection.
    """
    # 1. Cap the PnL artificially to prevent catastrophic math overflow
    clipped_pnl = np.clip(raw_pnl, -1000.0, 1000.0)
    
    # 2. Calculate the exponent and put a hard ceiling on it (max e^40)
    exponent = np.clip(-lambda_val * clipped_pnl, -40.0, 40.0)
    
    utility = -np.exp(exponent)
    return float(utility)

def train_agent():
    print("Initializing the Deep Hedging Training Loop...")
    env = TradingEnvironment()
    agent = RLAgent()
    
    # Track metrics for your thesis graphs
    history_pnl = []
    
    for episode in range(1, TOTAL_EPISODES + 1):
        state = env.reset()
        
        episode_log_probs = []
        episode_pnl = 0.0
        
        # --- 1. PLAY THE EPISODE (The Trajectory) ---
        while True:
            # Brain selects an action
            action, log_prob = agent.select_action(state)
            
            # World executes the action
            next_state, reward, done = env.step(action)
            
            # Memorize the probability of the action taken
            episode_log_probs.append(log_prob)
            
            # Update state
            state = next_state
            
            if done:
                # In our environment, the final step's reward is the Terminal PnL
                episode_pnl = reward 
                break
                
        # --- 2. CALCULATE THE RISK (The Thesis Core) ---
        # Convert the raw money into Entropic Risk Utility
        risk_adjusted_reward = calculate_entropic_utility(episode_pnl, RISK_AVERSION_LAMBDA)
        
        # --- 3. REINFORCE BACKPROPAGATION ---
        policy_loss = []
        
        # Apply the update rule: Loss = -Reward * log(Probability)
        for log_prob in episode_log_probs:
            policy_loss.append(-log_prob * risk_adjusted_reward)
            
        # Sum the loss for the entire 1,000-tick episode
        # We must explicitly set requires_grad=True to keep the Autograd graph alive
        loss = torch.stack(policy_loss).sum()
        
        # PyTorch Magic: Compute the calculus and nudge the weights
        agent.optimizer.zero_grad()  # Clear old math
        loss.backward()     # Calculate the gradients
        # --- NEW LINE: Gradient Clipping ---
        # Acts as a surge protector for the Neural Network weights
        torch.nn.utils.clip_grad_norm_(agent.policy_network.parameters(), max_norm=1.0)         
        agent.optimizer.step()       # Update the Neural Network weights
        
        # --- 4. LOGGING ---
        history_pnl.append(episode_pnl)
        
        # Print an update every episode so you can monitor it in VirtualBox
        print(f"Episode {episode:03d} | Terminal PnL: ${episode_pnl:8.2f} | Risk Utility: {risk_adjusted_reward:8.2f} | Loss: {loss.item():8.2f}")
        
    print("\n--- Training Complete ---")
    print(f"Average PnL over last 10 episodes: ${np.mean(history_pnl[-10:]):.2f}")
    
    # Save the trained brain to your disk
    torch.save(agent.policy_network.state_dict(), "deep_hedging_weights.pth")
    print("Model weights saved to 'deep_hedging_weights.pth'.")

if __name__ == "__main__":
    train_agent()