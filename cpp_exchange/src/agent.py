import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ActorCriticLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, action_dim=3):
        super(ActorCriticLSTM, self).__init__()
        
        # 1. The Recurrent Memory Core
        # batch_first=True means data goes in as [Batch, Sequence_Length, Features]
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # 2. The Actor Head
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        
        # 3. The Critic Head
        self.critic_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, hidden):
        # Process the sequence through the LSTM memory
        lstm_out, new_hidden = self.lstm(state, hidden)
        
        # The Actor and Critic branch off the memory state
        action_probs = F.softmax(self.actor_out(lstm_out), dim=-1)
        state_value = self.critic_out(lstm_out)
        
        return action_probs, state_value, new_hidden

class PPOAgentLSTM:
    def __init__(self, learning_rate=3e-4, hidden_dim=64):
        self.hidden_dim = hidden_dim
        self.policy_network = ActorCriticLSTM(hidden_dim=hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def get_initial_hidden(self):
        # Creates a blank slate memory (hx, cx) for the start of an episode
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def select_action(self, state_numpy, hidden, test_mode=False):
        # Format for a single tick: [1 Batch, 1 Sequence Step, 3 Features]
        state_tensor = torch.FloatTensor(state_numpy).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, state_value, new_hidden = self.policy_network(state_tensor, hidden)
            
        action_probs = action_probs.squeeze()
        
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
        
        return chosen_action, log_prob, entropy, state_value.squeeze(), new_hidden

    def evaluate(self, states, actions, initial_hidden):
        # NEW: Evaluates the entire 1,000-tick sequence at once during backprop
        action_probs, state_values, _ = self.policy_network(states, initial_hidden)
        
        # Squeeze out the batch dimension to align with actions
        action_probs = action_probs.squeeze(0)
        state_values = state_values.squeeze()
        
        dist = Categorical(action_probs)
        
        action_indices = []
        for a in actions:
            if a == -1: action_indices.append(0)
            elif a == 0: action_indices.append(1)
            elif a == 1: action_indices.append(2)
        action_indices_tensor = torch.tensor(action_indices)
        
        action_logprobs = dist.log_prob(action_indices_tensor)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy