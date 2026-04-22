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
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        """
        Defines how data flows through the network.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        
        return action_probs

class RLAgent:
    def __init__(self):
        self.policy_network = DeepHedgingModel()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.0001)

    def select_action(self, state_numpy, test_mode=False):
        """
        Takes the NumPy array from the environment, feeds it to the Brain, 
        and decides what to do.
        """
        state_tensor = torch.FloatTensor(state_numpy).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)
        
        if test_mode:
            # OUT-OF-SAMPLE MODE: Strictly pick the highest probability action
            action_index = torch.argmax(action_probs)
            log_prob, entropy = None, None
        else:
            # TRAINING MODE: Sample from the distribution to encourage exploration
            m = Categorical(action_probs)
            action_index = m.sample()
            log_prob = m.log_prob(action_index)
            entropy = m.entropy()
        
        # Map the index back to our C++ network commands:
        action_mapping = [-1, 0, 1]
        chosen_action = action_mapping[action_index.item()]
        
        return chosen_action, log_prob, entropy


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
        action, log_prob, _ = agent.select_action(state)
        
        print(f"Step {step+1}:")
        print(f"  State Tensor In: {state}")
        print(f"  AI Chose Action: {action}")
        
        next_state, reward, done = env.step(action)
        print(f"  Resulting PnL:   ${reward:.2f}\n")
        
        state = next_state
        time.sleep(0.5)
        
    print("AI successfully mapped states to actions.")