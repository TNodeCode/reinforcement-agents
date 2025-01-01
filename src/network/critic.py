import torch
import torch.nn as nn
from typing import List
    

class Critic(nn.Module):
    """Critic network that computes a score for a state-action tuple.
    """
    def __init__(self, dim_state: int, dim_action: int, hidden_dims: List[int] = [64]):
        super(Critic, self).__init__()
        dim_layers = [dim_state + dim_action] + hidden_dims + [1]
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.LeakyReLU()
            )
            for dim_in, dim_out in zip(dim_layers[:-1], dim_layers[1:])
        ])

    def forward(self, x):
        """Compute a score for a state-action tuple.

        Args:
            x (torch.tensor): State

        Returns:
            torch.tensor: Action
        """
        return self.net(x)