import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        hidden_sizes = [64, 64]
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_sizes[0])])
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_sizes[-1], action_size)
#         self.dropout = nn.Dropout(p=0.2)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
#             x = self.dropout(x)
        x = self.output(x)
        return x
