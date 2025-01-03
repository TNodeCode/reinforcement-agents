import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from src.agent.simple import SimpleAgent
from src.network.policy_discrete import DiscretePolicy
from src.network.utils import NetworkUtils


class REINFORCEAgentDiscrete(SimpleAgent):
    def __init__(
            self,
            env,
            eps=1.0,
            lr=5e-4,
            hidden_dims=[64],
            memory_size=int(1e5),
            activation=nn.LeakyReLU,
            max_steps=100,
            batch_size=256,
            update_every=10,
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
        self.gamma = torch.tensor(0.99).to(device)
        self.tau = 1e-3
        self.max_steps = max_steps
        self.update_every = update_every
        self.device = device
        # Build policy network
        self.policy = DiscretePolicy(
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
        # use the policy network to compute the action probabilities for each action
        action_probs = self.policy(state)
        # create a categorical distribution with weights probs from policy network and randomly sample from that distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        # return the action and its log probability
        return action.item(), action_dist.log_prob(action)
    
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

        rewards = []
        log_probs = []

        for _ in range(self.max_steps):
            action, log_prob = self.choose_action()
            _, reward, _, done = self.do_step(action=action)

            rewards.append(reward)
            log_probs.append(log_prob)

            total_reward += reward

            if done:
                break

        self.learn(rewards, log_probs)
        return total_reward