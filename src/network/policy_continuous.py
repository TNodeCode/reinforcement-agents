import torch
import torch.nn as nn
from typing import List


class ContinuousPolicy(nn.Module):
    """Policy network that maps a state to an action probability distribution.
    """
    def __init__(self, dim_state: int, dim_action: int, hidden_dims: List[int] = [64], activation=nn.LeakyReLU):
        """Constructor.

        Args:
            dim_state (int): dimension of state vectors
            dim_action (int): dimension of action vectors
            hidden_dims (List[int], optional): Dimenions of hidden layers. Defaults to [].
        """
        super(ContinuousPolicy, self).__init__()
        dim_layers = [dim_state] + hidden_dims

        # Feature extraction
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                activation()
            )
            for dim_in, dim_out in zip(dim_layers[:-1], dim_layers[1:])
        ], nn.Softmax())

        # Heads
        self.head_mean = nn.Linear(hidden_dims[-1], dim_action)
        self.head_log_std = nn.Linear(hidden_dims[-1], dim_action)

    def forward(self, state):
        """Compute action probs for state input.

        Args:
            states (torch.tensor): State tensor

        Returns:
            (torch.tensor, torch.tensor): action probabilities and state values
        """
        # compute shared features
        features = self.net(state)

        # Actor: outputs action probabilities
        mean = self.head_mean(features)
        std = torch.exp(self.head_log_std(features))

        return mean, std