import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from src.agent.simple import SimpleAgent
from src.network.policy_continuous import ContinuousPolicy
from src.network.utils import NetworkUtils


class REINFORCEAgentContinuous(SimpleAgent):
    def __init__(
            self,
            env,
            eps=1.0,
            gamma=0.99,
            lr=5e-4,
            hidden_dims=[64],
            memory_size=None,
            activation=nn.LeakyReLU,
            max_steps=100,
            batch_size=256,
            update_every=1,
            device="cpu"
        ):
        """
        Constructor.
        """
        super().__init__(
            env,
            eps=eps,
            memory_size=memory_size,
            action_space_discrete=True,
            batch_size=batch_size,
            device=device
        )
        self.eps = eps
        self.gamma = torch.tensor(gamma).to(device)
        self.tau = 1e-3
        self.max_steps = max_steps
        self.update_every = update_every
        self.device = device
        # Build policy network
        self.policy = ContinuousPolicy(
            dim_state=self.state.shape[1],
            dim_action=self.brain.vector_action_space_size,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device)
        # optimizer for policy network
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    def choose_action(self):
        """Samples an action from the policy's distribution over actions and returns both the action and its log probability.

        Returns:
            Action and its log probability
        """
        # convert state to torch tensor
        state = torch.from_numpy(self.state).float().to(self.device)
        # policy computes mean and std for a normal distribution that is used for sampling an action
        mean, std = self.policy(state)
        # create a normal distribution based on the policy output and sample from that distribution
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        # compute log probability of chosen action
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        action = action.detach().cpu().numpy()
        action = np.clip(action, a_min=-1, a_max=1)
        return action, action_log_prob
    
    def learn(self, rewards, log_probs):
        """Computes returns from rewards, normalizes them, and calculates the policy loss.
        The policy is updated by taking the gradient of the expected return with respect to the policy parameters.

        Args:
            rewards (_type_): _description_
            log_probs (_type_): _description_
        """
        returns = []
        Gt = 0
        for reward in reversed(rewards):
            Gt = reward + self.gamma * Gt
            returns.insert(0, Gt)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize returns

        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)
        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def play(self):
        total_reward = 0

        for _ in range(self.max_steps):
            actions, log_probs = self.choose_action()
            _, rewards, _, dones = self.do_step(action=actions)

            step_rewards = sum(rewards)/len(rewards)
            total_reward += step_rewards

            if any(dones):
                break

        self.learn(rewards, log_probs)
        return total_reward