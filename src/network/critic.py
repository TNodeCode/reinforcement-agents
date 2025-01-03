import torch
import torch.nn as nn
from typing import List
    

class Critic(nn.Module):
    """Critic network that computes a score for a state-action tuple.
    """
    def __init__(self, dim_state: int, dim_action: int, hidden_dims: List[int] = [64], activation=nn.LeakyReLU, norm=nn.LayerNorm, seed=42):
        """Constructor.

        Args:
            dim_state (int): dimension of state vectors
            dim_action (int): dimension of action vectors
            hidden_dims (List[int], optional): Dimenions of hidden layers. Defaults to [].
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        dim_layers = [dim_state + dim_action] + hidden_dims + [1]
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                activation(),
                norm(dim_out),
            )
            for dim_in, dim_out in zip(dim_layers[:-1], dim_layers[1:])
        ])

    def forward(self, states, actions):
        """Compute a score for a state-action tuple.

        Args:
            states (torch.tensor): State tensor
            actions (torch.tensor); Action tensor

        Returns:
            torch.tensor: Target value
        """
        return self.net(torch.cat((states, actions), dim=1))