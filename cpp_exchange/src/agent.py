import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Import the environment we just built
from environment import TradingEnvironment

class DeepHedgingModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        """
        The mathematical architecture of the Neural Network.
        Input: [Price, Holdings, Time] (Size: 3)
        Output: Probabilities for [Sell, Hold, Buy] (Size: 3)
        """
        super(DeepHedgingModel, self).__init__()
        
        # Layer 1: Ingests the 3 state variables
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Layer 2: Hidden processing
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer 3: Output layer for the 3 possible actions
        self.action_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        """
        Defines how data flows through the network.
        """
        # Pass through hidden layers with ReLU activation
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Output layer gets Softmax to convert raw numbers into probabilities (summing to 1.0)
        action_probs = F.softmax(self.action_head(x), dim=-1)
        
        return action_probs

class RLAgent:
    def __init__(self):
        # Initialize the neural network
        self.policy_network = DeepHedgingModel()
        
        # The Optimizer (Adam) will adjust the weights later during training
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.0001)

    def select_action(self, state_numpy):
        """
        Takes the NumPy array from the environment, feeds it to the Brain, 
        and decides what to do.
        """
        # 1. Convert the C++/NumPy state into a PyTorch Tensor
        state_tensor = torch.FloatTensor(state_numpy).unsqueeze(0)
        
        # 2. Feed it to the Brain (Forward Pass)
        # Returns something like: [0.10, 0.20, 0.70]
        action_probs = self.policy_network(state_tensor)
        
        # 3. Create a probability distribution from the network's output
        m = Categorical(action_probs)
        
        # 4. Sample an action based on those probabilities (0, 1, or 2)
        action_index = m.sample()
        
        # 5. Map the index back to our C++ network commands:
        # Index 0 -> Action -1 (Sell)
        # Index 1 -> Action  0 (Hold)
        # Index 2 -> Action  1 (Buy)
        action_mapping = [-1, 0, 1]
        chosen_action = action_mapping[action_index.item()]
        
        # We also return the 'log probability' of the action. 
        # (David Silver's math requires this for the Policy Gradient update later).
        return chosen_action, m.log_prob(action_index), m.entropy()


# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    import time
    
    print("Booting up the PyTorch Brain...")
    agent = RLAgent()
    
    print("Connecting to the C++ World...")
    env = TradingEnvironment()
    
    state = env.reset()
    
    print("\n--- Running AI Integration Test (10 Steps) ---")
    for step in range(10):
        # 1. The Brain looks at the state and chooses an action
        action, log_prob = agent.select_action(state)
        
        print(f"Step {step+1}:")
        print(f"  State Tensor In: {state}")
        print(f"  AI Chose Action: {action} (LogProb: {log_prob.item():.4f})")
        
        # 2. The World executes the action and returns the results
        next_state, reward, done = env.step(action)
        print(f"  Resulting PnL:   ${reward:.2f}\n")
        
        # 3. Update time
        state = next_state
        time.sleep(0.5)
        
    print("AI successfully mapped states to actions. Tensors are flowing into Redis.")