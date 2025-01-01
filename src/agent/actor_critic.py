import os
import torch
import torch.nn.functional as F
import random
import numpy as np
from src.agent.simple import SimpleAgent
from src.network.actor import Actor
from src.network.critic import Critic


class ActorCriticAgent(SimpleAgent):
    def __init__(self, env, eps=1.0, lr_actor=5e-4, lr_critic=5e-4, memory_size=int(1e5), batch_size=256, device="cpu"):
        """
        Constructor.
        """
        super().__init__(env, eps=eps, memory_size=memory_size, batch_size=batch_size)
        self.eps = eps
        self.gamma = torch.tensor(0.99).to(device)
        self.tau = 1e-3
        self.update_every = 5
        self.device = device

        # Build a local and a target agent
        self.actor_local = Actor(
            dim_state=len(self.state),
            dim_action=self.brain.vector_action_space_size
        ).to(device)        
        self.actor_target = Actor(
            dim_state=len(self.state),
            dim_action=self.brain.vector_action_space_size
        ).to(device)

        # Build a local and a target critic network
        self.critic_local = Critic(
            dim_state=len(self.state),
            dim_action=self.brain.vector_action_space_size
        ).to(device)        
        self.critic_target = Critic(
            dim_state=len(self.state),
            dim_action=self.brain.vector_action_space_size
        ).to(device)

        # optimizers
        self.optim_actor = torch.optim.AdamW(self.actor_local.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.AdamW(self.critic_local.parameters(), lr=lr_critic)

    def choose_action(self):
        """
        Choose an action based on internal knowledge.

        Returns:
            torch.tensor: best action as continuous vector
        """
        # get a continuous representation of the optimal action
        self.actor_local.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(self.state).float()
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            action = self.actor_local(state_tensor)
        self.actor_local.train()
        # limit range to interval [min, max]
        action = torch.clamp(action, min=-1, max=1)
        # add random noise to action tensor, this will help the model to stabilize predictions
        action += torch.randn(*action.shape)
        return action.detach().cpu().numpy()

    def learn(self):
        """
        Learning step.
        """
        if len(self.memory) % self.update_every == 0:
            # get a random batch from memory
            states, actions, rewards, next_states, dones = self.memory.sample(device=self.device, batch_size=256)
            # actor computes action vector based on state
            predicted_actions = self.actor_local(states)
            target_actions = self.actor_target(next_states)
            # critic computes expected future rewards (values of states)
            predicted_values = self.critic_local(torch.cat((states, actions), dim=1))
            target_values = self.critic_target(torch.cat((next_states, target_actions), dim=1))
            # compute TD estimate
            td_estimate = rewards + self.gamma * target_values * (1 - dones)

            ## --- Critic update --- ##

            # compute difference between expected / target values that local / target critic computed
            loss_critic = F.mse_loss(predicted_values, td_estimate)
            self.optim_critic.zero_grad()
            loss_critic.backward()
            self.optim_critic.step()

            ## --- Actor update --- ##

            loss_actor = -self.critic_local(torch.cat((states, predicted_actions), dim=1)).mean()
            self.optim_actor.zero_grad()
            loss_actor.backward()
            self.optim_actor.step()

            ## --- Target network updates --- ##

            # update weights of target actor
            for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            # update weights of target critic
            for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

