import torch
import random
import numpy as np
from collections import deque



class Memory:
    def __init__(self, maxlen):
        """
        Initializes the replay buffer.

        Parameters:
            max_size: The maximum number of experiences the buffer can hold
        """
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)

    def append(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the buffer.

        Parameters:
            state: The initial state
            action: The action taken
            reward: The reward received
            next_state: The state after taking the action
            done: A boolean indicating if the episode ended
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.

        Parameters:
            batch_size: The number of experiences to sample

        Returns:
            A tuple of arrays: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)