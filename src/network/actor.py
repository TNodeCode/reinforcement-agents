import torch
import torch.nn as nn
from typing import List


class Actor(nn.Module):
    """Actor network that maps a state to an action.
    """
    def __init__(self, dim_state: int, dim_action: int, hidden_dims: List[int] = [64]):
        """Constructor.

        Args:
            dim_state (int): dimension of state vectors
            dim_action (int): dimension of action vectors
            hidden_dims (List[int], optional): Dimenions of hidden layers. Defaults to [].
        """
        super(Actor, self).__init__()
        dim_layers = [dim_state] + hidden_dims + [dim_action]
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.LeakyReLU()
            )
            for dim_in, dim_out in zip(dim_layers[:-1], dim_layers[1:])
        ])

    def forward(self, x):
        """Compute action for state input.

        Args:
            x (torch.tensor): State

        Returns:
            torch.tensor: Action
        """
        return self.net(x)