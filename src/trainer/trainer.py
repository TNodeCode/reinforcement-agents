import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer:
    """
    The trainer class trains an agent on a given environment.

    Args:
        env : Unity environment
        agent: Agent that plays the game
        device: cuda or cpu
    """
    def __init__(self, env, agent, device="cpu"):
        self.env = env
        self.agent = agent
        self.scores = []
        self.avg_epochs_score = 100

    def reset(self):
        """
        Reset training.
        """
        self.scores = []

    def train(self, n_epochs=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """
        Train the agent.

        Args:
            eps_start: epsilon value for eps-greedy strategy at the beginning of the training
            eps_end: minimum epsilon value for eps-greedy strategy
            eps_decay: decay factor for epsilon value after each epoch
        """
        eps = eps_start
        it = tqdm(range(n_epochs))
        for epoch in it:
            self.agent.eps = eps
            score = self.agent.play()
            self.scores.append(score)
            self.agent.reset()
            eps = max(eps_end, eps_decay*eps)
            it.set_description_str(f"Epoch {epoch} - Avg Score: {sum(self.scores[-self.avg_epochs_score:])/self.avg_epochs_score}")

    def plot_scores(self):
        """
        Plot scores.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        fig.suptitle("Scores over episodes")
        ax.plot(np.arange(len(self.scores)), self.scores)
        ax.set_ylabel('Score')
        ax.set_xlabel('Episode')

