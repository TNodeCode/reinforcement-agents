import torch
import torch.nn as nn
from typing import List


class ActorCriticNetwork(nn.Module):
    """Actor-Critic Network.
    """
    def __init__(self, dim_state, dim_action, hidden_dims=[64], activation=nn.LeakyReLU, norm=nn.LayerNorm):
        """Constructor.

        Args:
            dim_state (int): dimension of state vectors
            dim_action (int): dimension of action vectors
            hidden_dims (List[int], optional): Dimenions of hidden layers. Defaults to [].
        """
        super(ActorCriticNetwork, self).__init__()
        dim_layers = [dim_state] + hidden_dims
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                activation,
                norm(dim_out),
            )
            for dim_in, dim_out in zip(dim_layers[:-1], dim_layers[1:])
        ])
        
        # Actor head
        self.actor = nn.Linear(hidden_dims[-1], dim_action)
        
        # Critic head
        self.critic = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state):
        """Compute action probabilities and state values

        Args:
            state (torch.tensor): state

        Returns:
            (torch.tensor, torch.tensor): action probabilities and state values
        """
        # compute shared features
        features = self.net(state)

        # Actor: outputs action probabilities
        action_probabilities = torch.softmax(self.actor(features), dim=-1)
        
        # Critic: outputs state value
        state_value = self.critic(features)
        
        return action_probabilities, state_value