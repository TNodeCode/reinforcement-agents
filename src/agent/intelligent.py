import os
import torch
import torch.nn.functional as F
import random
import numpy as np
from src.agent.simple import SimpleAgent
from src.network.q_network import DeepQNetwork


class IntelligentAgent(SimpleAgent):
    def __init__(self, env, eps=1.0, lr=5e-4, memory_size=int(1e5), batch_size=256, device="cpu"):
        """
        Constructor.
        """
        super().__init__(env, eps=eps, memory_size=memory_size, batch_size=batch_size)
        self.eps = eps
        self.gamma = torch.tensor(0.99).to(device)
        self.tau = 1e-3
        self.update_every = 10
        self.device = device
        # Build two Deep-Q-Networks 
        self.net_local = DeepQNetwork(
            n_states=len(self.state),
            n_actions=self.brain.vector_action_space_size
        ).to(device)        
        self.net_target = DeepQNetwork(
            n_states=len(self.state),
            n_actions=self.brain.vector_action_space_size
        ).to(device)
        self.optim = torch.optim.AdamW(self.net_local.parameters(), lr=lr)
        
    def choose_action(self):
        """
        Choose an action based on internal knowledge.
        """
        # For choosing next step put net_local into validation mode
        self.net_local.eval()
        # Use net_local to compute scores for each action based on state and get best action
        with torch.no_grad():
            # compute scores for each action
            state_tensor = torch.from_numpy(self.state).float()
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            action_scores = self.net_local(state_tensor)
            # get index of best action
            best_action_idx, _ = self.net_local.get_best_choice(action_scores)
        # Put net_local back into training mode
        self.net_local.train()
        # Epsilon greedy strategy
        if random.random() > self.eps:
            return int(best_action_idx)
        else:
            return np.random.randint(self.brain.vector_action_space_size) 
        
    def learn(self):
        """
        Learning step.
        """
        if len(self.memory) % self.update_every == 0:
            # get a random batch from memory
            states, actions, rewards, next_states, dones = self.memory.sample(device=self.device, batch_size=256)
            # get best actions and their scores
            action_scores = self.net_target(next_states).detach().max(1)[0].unsqueeze(1)
            # Compute Q targets for current states 
            target = rewards + self.gamma * action_scores * (1-dones)
            # Get expected Q values from local model
            current = self.net_local(states).gather(1, actions)
            # compare current action scores to target action scores
            loss = F.mse_loss(current, target)
            # perform backpropagation and update weights of network1
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # update weights of network 2
            for target_param, local_param in zip(self.net_target.parameters(), self.net_local.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def save(self, dir="./output"):
        """
        Save knowledge of agent.
        """
        # save weights of local network
        torch.save(self.net_local.state_dict(), os.path.join(dir, 'checkpoint.pth'))

    def load(self, dir="./output"):
        """
        Load agent knowledge.

        Args:
            dir (str): Directory where knowledge (weigh files) is stored.
        """
        self.net_local.load_state_dict(torch.load(os.path.join(dir, 'checkpoint.pth')))

    def test(self):
        """
        Play a round with epsilon set to zero.

        Returns:
            score (int): achieved score
        """
        self.agent.eps = 0.0
        score = self.agent.play()
        return score
        