import torch
import random
import numpy as np
from collections import deque


class Memory:
    def __init__(self, maxlen=1e5):
        """
        Initialize memory of variables needed for reinforcement learning.
        
        Args:
            maxlen (int): Maximum size of memory
        """
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)
        self.done = deque(maxlen=maxlen)
        
    def append(self, state, action, reward, next_state, done):
        """
        Add variables to memory.
        
        Args:
            state: environment state
            action: action that was taken by the agent
            reward: reward that was gained by taking an action
            next_state: state that was reached by taking an action
            done: True if game was over after taking action, False otherwise
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.done.append(done)
        
    def sample(self, batch_size, device="cpu"):
        """
        Get a random batch of memorized variables.
        
        Args:
            batch_size: size of the random batch
            device: cpu or cuda
        
        Returns:
            torch.tensor: random batch
        """
        idx = [random.randint(0, len(self.states) - 1) for x in range(batch_size)]
        return (
            torch.from_numpy(np.vstack(self.states)[idx]).float().to(device),
            torch.from_numpy(np.vstack(self.actions)[idx]).long().to(device),
            torch.from_numpy(np.vstack(self.rewards)[idx]).float().to(device),
            torch.from_numpy(np.vstack(self.next_states)[idx]).float().to(device),
            torch.from_numpy(np.vstack(self.done)[idx].astype(np.uint8)).float().to(device),
        )
    
    def __len__(self):
        """
        Get current size of memory.
        
        Returns:
            int: memory size
        """
        return len(self.states)