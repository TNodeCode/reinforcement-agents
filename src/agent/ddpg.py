import os
import torch
import torch.nn.functional as F
import random
import numpy as np
from src.agent.simple import SimpleAgent
from src.network.actor import Actor
from src.network.critic import Critic


class DDPGAgent(SimpleAgent):
    def __init__(self, env, eps=1.0, lr_actor=1e-4, lr_critic=1e-3, memory_size=int(1e6), batch_size=64, device="cpu"):
        """
        Constructor.
        """
        super().__init__(env, eps=eps, memory_size=memory_size, batch_size=batch_size)
        self.eps = eps
        self.gamma = torch.tensor(0.99).to(device)
        self.tau = 0.005
        self.update_every = 1
        self.device = device

        # Build a local and a target agent
        self.actor_local = Actor(
            dim_state=len(self.state),
            dim_action=self.brain.vector_action_space_size,
            hidden_dims=[64,64],
        ).to(device)        
        self.actor_target = Actor(
            dim_state=len(self.state),
            dim_action=self.brain.vector_action_space_size,
            hidden_dims=[64,64],
        ).to(device)

        # Build a local and a target critic network
        self.critic_local = Critic(
            dim_state=len(self.state),
            dim_action=self.brain.vector_action_space_size,
            hidden_dims=[64,64],
        ).to(device)        
        self.critic_target = Critic(
            dim_state=len(self.state),
            dim_action=self.brain.vector_action_space_size,
            hidden_dims=[64,64],
        ).to(device)

        # Copy weights from local actor and local critic to target networks
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.critic_target.load_state_dict(self.critic_local.state_dict())

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
            state_tensor = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
            action = self.actor_local(state_tensor).squeeze()
            # limit range to interval [min, max]
            action = torch.clamp(action, min=-1, max=1)
            # add random noise to action tensor, this will help the model to stabilize predictions
            action += 0.01 * torch.randn(*action.shape)
        self.actor_local.train()
        return action.cpu().numpy()

    def learn(self):
        """
        Learning step.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # get a random batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size=self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        #print("states", states.shape, "actions", actions.shape, "rewards", rewards.shape, "next_states", next_states.shape, "dones", dones.shape)

        ## --- Critic update --- ##

        # actor computes action vector based on state
        with torch.no_grad():
            target_actions = self.actor_target(states=next_states)
            target_q_values = self.critic_target(states=next_states, actions=target_actions)
            target_values = rewards + self.gamma * target_q_values * (1 - dones)

        # compute difference between expected / target values that local / target critic computed
        critic_values = self.critic_local(states=states, actions=actions)
        loss_critic = F.mse_loss(critic_values, target_values)
        
        self.optim_critic.zero_grad()
        loss_critic.backward()
        self.optim_critic.step()

        ## --- Actor update --- ##

        loss_actor = -self.critic_local(states=states, actions=self.actor_local(states=states)).mean()
        
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        ## --- Target network updates --- ##

        self.soft_update(self.actor_local, self.actor_target, tau=self.tau)
        self.soft_update(self.critic_local, self.critic_target, tau=self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Perform a soft update of the target network with the weights from the local network.

        Args:
            local_model (nn.Module): Local model
            target_model (nn.Module): Target model
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
