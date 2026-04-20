import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import TradingEnvironment
from agent import RLAgent

def test_agent():
    print("Booting up the Evaluation Environment...")
    env = TradingEnvironment()
    agent = RLAgent()
    
    # --- 1. LOAD THE TRAINED BRAIN ---
    try:
        agent.policy_network.load_state_dict(torch.load("deep_hedging_weights.pth"))
        agent.policy_network.eval() # Put PyTorch into 'Testing Mode' (disables learning algorithms)
        print("Successfully loaded 'deep_hedging_weights.pth'.")
    except FileNotFoundError:
        print("Error: Could not find weights file. Did you run train.py?")
        return

    # Data lists to feed into Matplotlib
    history_prices = []
    history_holdings = []
    
    state = env.reset()
    episode_pnl = 0.0
    
    print("\nRunning a 30-Day Deterministic Simulation...")
    
    # --- 2. PLAY ONE STRICT EPISODE ---
    while True:
        # Record the current state of the world before the AI acts
        history_prices.append(env.current_price)
        history_holdings.append(env.holdings)
        
        # Turn off PyTorch's memory tracking to speed up the loop
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = agent.policy_network(state_tensor)
            
        # DETERMINISTIC ACTION: Pick the exact highest probability, no random sampling
        action_index = torch.argmax(action_probs).item()
        
        # Map the index to the C++ commands
        action_mapping = [-1, 0, 1]
        best_action = action_mapping[action_index]
        
        # Execute the action
        next_state, reward, done = env.step(best_action)
        state = next_state
        
        if done:
            episode_pnl = reward
            break
            
    print(f"Simulation Complete. Final PnL: ${episode_pnl:.2f}")
    
    # --- 3. GENERATE THE THESIS GRAPHS ---
    print("Generating Matplotlib visuals...")
    
    # Create a visual figure with two stacked charts
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top Graph: The C++ Market Price
    ax1.plot(history_prices, color='blue', linewidth=1.5, label='C++ Market Price')
    ax1.set_title('Baseline Evaluation: Market Price vs. AI Inventory Holdings')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Bottom Graph: The AI's Inventory Holdings
    ax2.plot(history_holdings, color='red', linewidth=1.5, label='AI Inventory (Shares)')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.8) # The Delta-Neutral 'Zero' Line
    ax2.set_xlabel('Time (Simulation Steps)')
    ax2.set_ylabel('Inventory Count')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save the graph to your folder and display it
    plt.savefig("baseline_evaluation.png", dpi=300)
    print("Graph successfully saved as 'baseline_evaluation.png'.")
    plt.show()

if __name__ == "__main__":
    test_agent()