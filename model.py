import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_sizes=[256, 128]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (list of int): Number of nodes in each hidden layer
        """
        assert(len(hidden_sizes) > 0, "hidden_sizes parameter needs to be a list of at least one integer")
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_first = nn.Linear(state_size, hidden_sizes[0])

        self.fc_list = nn.ModuleList()
        if (len(hidden_sizes) > 1):
            self.fc_list.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(0,len(hidden_sizes)-1)])

        self.fc_last = nn.Linear(hidden_sizes[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_first.weight.data.uniform_(*hidden_init(self.fc_first))
        for fc_hidden in self.fc_list:
            fc_hidden.weight.data.uniform_(*hidden_init(fc_hidden))
        self.fc_last.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc_first(states))

        for fc_hidden in self.fc_list:
            x = F.relu(fc_hidden(x))
        return F.tanh(self.fc_last(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_sizes=[256, 128]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_sizes (list of int): Number of nodes in each hidden layer
        """
        super(Critic, self).__init__()
        assert(len(hidden_sizes) > 1, "hidden_sizes parameter needs to be a list of at least two integers")
        self.seed = torch.manual_seed(seed)
        self.fc_first = nn.Linear(state_size, hidden_sizes[0])
        self.fc_second = nn.Linear(hidden_sizes[0]+action_size, hidden_sizes[1])
        
        self.fc_list = nn.ModuleList()
        if (len(hidden_sizes) > 2):
            self.fc_list.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(1,len(hidden_sizes)-1)])
        
        self.fc_last = nn.Linear(hidden_sizes[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_first.weight.data.uniform_(*hidden_init(self.fc_first))
        self.fc_second.weight.data.uniform_(*hidden_init(self.fc_second))
        for fc_hidden in self.fc_list:
            fc_hidden.weight.data.uniform_(*hidden_init(fc_hidden))
        self.fc_last.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fc_first(states))
        x = torch.cat((xs, actions), dim=1)
        x = F.relu(self.fc_second(x))
        for fc_hidden in self.fc_list:
            x = F.relu(fc_hidden(x))
        return self.fc_last(x)
