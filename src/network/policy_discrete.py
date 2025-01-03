import torch.nn as nn
from typing import List


class DiscretePolicy(nn.Module):
    """Policy network that maps a state to an action probability distribution.
    """
    def __init__(self, dim_state: int, dim_action: int, hidden_dims: List[int] = [64], activation=nn.LeakyReLU, norm=nn.LayerNorm):
        """Constructor.

        Args:
            dim_state (int): dimension of state vectors
            dim_action (int): dimension of action vectors
            hidden_dims (List[int], optional): Dimenions of hidden layers. Defaults to [].
        """
        super(DiscretePolicy, self).__init__()
        dim_layers = [dim_state] + hidden_dims + [dim_action]
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                activation(),
                norm(dim_out),
            )
            for dim_in, dim_out in zip(dim_layers[:-1], dim_layers[1:])
        ], nn.Softmax(dim=-1))

    def forward(self, states):
        """Compute action probs for state input.

        Args:
            states (torch.tensor): State tensor

        Returns:
            torch.tensor: Action
        """
        return self.net(states)