import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from src.agent.simple import SimpleAgent
from src.network.q_network import DeepQNetwork
from src.network.utils import NetworkUtils


class DeepQAgent(SimpleAgent):
    def __init__(
            self,
            env,
            eps=1.0,
            lr=5e-4,
            hidden_dims=[64],
            memory_size=int(1e5),
            activation=nn.LeakyReLU,
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
        self.update_every = update_every
        self.device = device
        # Build two Deep-Q-Networks 
        self.local_q_network = DeepQNetwork(
            dim_state=self.state.shape[1],
            n_actions=self.brain.vector_action_space_size,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device)        
        self.target_q_network = DeepQNetwork(
            dim_state=self.state.shape[1],
            n_actions=self.brain.vector_action_space_size,
            hidden_dims=hidden_dims,
            activation=activation,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.local_q_network.parameters(), lr=lr)
        
    def choose_action(self):
        """
        Choose an action based on internal knowledge.
        """
        # For choosing next step put local_q_network into validation mode
        self.local_q_network.eval()
        # Use local_q_network to compute scores for each action based on state and get best action
        with torch.no_grad():
            # compute scores for each action
            state_tensor = torch.from_numpy(self.state).float()
            state_tensor = state_tensor.to(self.device)
            action_scores = self.local_q_network(state_tensor)
            # get index of best action
            best_action_idx, _ = self.local_q_network.get_best_choice(action_scores)
        # Put local_q_network back into training mode
        self.local_q_network.train()
        # Epsilon greedy strategy
        if random.random() > self.eps:
            return int(best_action_idx)
        else:
            return np.random.randint(self.brain.vector_action_space_size)
        
    def learn(self):
        if len(self.memory) < self.batch_size or len(self.memory) % self.update_every != 0:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        actions = actions.unsqueeze(1)

        # Calculate target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_q_network(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Calculate current Q-values
        current_q_values = self.local_q_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the local Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network with soft updates
        NetworkUtils.soft_update(self.local_q_network, self.target_q_network, tau=self.tau)
        
    def _learn_old(self):
        """
        Learning step.
        """
        if len(self.memory) % self.update_every == 0 and len(self.memory) > self.batch_size:
            # get a random batch from memory
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size=self.batch_size)
            # get best actions and their scores
            action_scores = self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states 
            target = rewards + self.gamma * action_scores * (1-dones)
            # Get expected Q values from local model
            current = self.local_q_network(states).gather(1, actions)
            # compare current action scores to target action scores
            loss = F.mse_loss(current, target)
            # perform backpropagation and update weights of network1
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # update weights of network 2
            for target_param, local_param in zip(self.target_q_network.parameters(), self.local_q_network.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save(self, dir="./output"):
        """
        Save knowledge of agent.
        """
        # save weights of local network
        torch.save(self.local_q_network.state_dict(), os.path.join(dir, 'checkpoint.pth'))

    def load(self, dir="./output"):
        """
        Load agent knowledge.

        Args:
            dir (str): Directory where knowledge (weigh files) is stored.
        """
        self.local_q_network.load_state_dict(torch.load(os.path.join(dir, 'checkpoint.pth')))

    def test(self):
        """
        Play a round with epsilon set to zero.

        Returns:
            score (int): achieved score
        """
        self.agent.eps = 0.0
        score = self.agent.play()
        return score
        