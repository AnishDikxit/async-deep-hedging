import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from environment import TradingEnvironment

class ActorCriticModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, action_dim=3):
        """
        The Dual-Headed PPO Network.
        Input: [Price, Holdings, Time]
        Outputs: 
          1. Action Probabilities (The Actor)
          2. State Value / Expected PnL (The Critic)
        """
        super(ActorCriticModel, self).__init__()
        
        # Shared Market Representation Layer
        self.shared_layer = nn.Linear(input_dim, hidden_dim)
        
        # --- THE ACTOR HEAD ---
        self.actor_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        
        # --- THE CRITIC HEAD ---
        self.critic_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1) # Outputs a single number (Expected PnL)

    def forward(self, state):
        # 1. Process the raw market state
        shared_features = F.relu(self.shared_layer(state))
        
        # 2. Actor decides what to do
        actor_x = F.relu(self.actor_hidden(shared_features))
        action_probs = F.softmax(self.actor_out(actor_x), dim=-1)
        
        # 3. Critic predicts the future
        critic_x = F.relu(self.critic_hidden(shared_features))
        state_value = self.critic_out(critic_x)
        
        return action_probs, state_value

class PPOAgent:
    def __init__(self, learning_rate=3e-4):
        self.policy_network = ActorCriticModel()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def select_action(self, state_numpy, test_mode=False):
        """
        Called during the simulation loop to pick an action and record the Critic's prediction.
        """
        state_tensor = torch.FloatTensor(state_numpy).unsqueeze(0)
        
        with torch.no_grad(): # No backprop during the data collection phase!
            action_probs, state_value = self.policy_network(state_tensor)
        
        if test_mode:
            action_index = torch.argmax(action_probs)
            log_prob, entropy = None, None
        else:
            m = Categorical(action_probs)
            action_index = m.sample()
            log_prob = m.log_prob(action_index)
            entropy = m.entropy()
            
        action_mapping = [-1, 0, 1]
        chosen_action = action_mapping[action_index.item()]
        
        # We now return the Critic's value alongside the action
        return chosen_action, log_prob, entropy, state_value.squeeze()

    def evaluate(self, states, actions):
        """
        NEW FOR PPO: Called during the backpropagation update to grade past actions.
        """
        action_probs, state_values = self.policy_network(states)
        dist = Categorical(action_probs)
        
        # Map our [-1, 0, 1] actions back to PyTorch indices [0, 1, 2] for the math
        action_indices = []
        for a in actions:
            if a == -1: action_indices.append(0)
            elif a == 0: action_indices.append(1)
            elif a == 1: action_indices.append(2)
        action_indices_tensor = torch.tensor(action_indices)
        
        action_logprobs = dist.log_prob(action_indices_tensor)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values.squeeze(), dist_entropy

# ==========================================
# SANITY CHECK
# ==========================================
if __name__ == "__main__":
    print("Booting up the Actor-Critic Brain...")
    agent = PPOAgent()
    env = TradingEnvironment()
    state = env.reset()
    
    action, log_prob, _, state_value = agent.select_action(state)
    print(f"Market State: {state}")
    print(f"Actor chose Action: {action}")
    print(f"Critic predicts terminal PnL of: ${state_value.item():.2f}")