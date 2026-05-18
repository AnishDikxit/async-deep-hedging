import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import TradingEnvironment

# THE FIX: Import the Recurrent Brain
from agent import PPOAgentLSTM 

def test_agent():
    print("Booting up the Asynchronous Evaluation Environment...")
    env = TradingEnvironment()
    
    # Instantiate the new LSTM brain
    agent = PPOAgentLSTM() 
    
    try:
        agent.policy_network.load_state_dict(torch.load("deep_hedging_weights.pth"))
        agent.policy_network.eval() # Lock the neural network parameters
        print("Successfully loaded 'deep_hedging_weights.pth'.")
    except FileNotFoundError:
        print("Error: Could not find weights file. Waiting for training to finish.")
        return

    # Data tracking arrays
    history_prices = []
    history_holdings = []
    history_actions = []
    history_pnl = []
    
    state = env.reset()
    
    # THE FIX: Initialize the LSTM's blank memory
    hidden = agent.get_initial_hidden()
    
    print(f"AI woke up with Forced Exposure: {env.holdings} shares.")
    print("Running a 1,000-Tick Asynchronous Simulation...")
    
    while True:
        history_prices.append(env.current_price)
        history_holdings.append(env.holdings)
        
        with torch.no_grad():
            # THE FIX: Pass the hidden state in, and receive the updated hidden state back
            best_action, _, _, _, hidden = agent.select_action(state, hidden, test_mode=True)
            
        history_actions.append(best_action)
        
        next_state, reward, done = env.step(best_action)
        
        # Calculate Mark-to-Market PnL
        current_mtm_value = env.cash + (env.holdings * env.current_price)
        history_pnl.append(current_mtm_value - env.starting_cash) 
        
        state = next_state
        
        if done:
            print(f"Simulation Complete. Final MTM PnL: ${history_pnl[-1]:.2f}")
            break
            
    # --- GENERATE THE VISUALS ---
    print("Generating Asynchronous Matplotlib visuals...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # --- Panel 1: Market Price & AI Actions ---
    ax1.plot(history_prices, color='black', linewidth=1.2, label='Hawkes Market Price', alpha=0.8)
    
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for t in range(len(history_actions)):
        if history_actions[t] == 1:   
            buy_x.append(t)
            buy_y.append(history_prices[t])
        elif history_actions[t] == -1: 
            sell_x.append(t)
            sell_y.append(history_prices[t])
            
    ax1.scatter(buy_x, buy_y, color='green', marker='^', s=40, label='AI Buy Order Sent (Delayed Fill)', alpha=0.6)
    ax1.scatter(sell_x, sell_y, color='red', marker='v', s=40, label='AI Sell Order Sent (Delayed Fill)', alpha=0.6)
    
    ax1.set_title('Deep Hedging v2.0: Asynchronous Execution in a Hawkes Market')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # --- Panel 2: The Delta-Neutral Dance (Inventory) ---
    ax2.plot(history_holdings, color='blue', linewidth=1.5, label='AI Inventory Exposure')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Delta-Neutral (Zero Risk)')
    
    ax2.set_ylabel('Inventory (Shares)')
    ax2.set_title('Predictive Hedging: Shedding Exposure via 50ms Latency Queue')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # --- Panel 3: Cumulative PnL ---
    ax3.plot(history_pnl, color='purple', linewidth=1.5, label='Mark-to-Market PnL')
    ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    ax3.set_xlabel('Time (Simulation Milliseconds)')
    ax3.set_ylabel('PnL ($)')
    ax3.set_title('Portfolio Mark-to-Market Value (Factoring Slippage)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig("thesis_v2_asynchronous_evaluation.png", dpi=300)
    print("Graph successfully saved as 'thesis_v2_asynchronous_evaluation.png'.")

if __name__ == "__main__":
    test_agent()