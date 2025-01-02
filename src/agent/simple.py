import numpy as np
from src.memory.memory import Memory
from tqdm import tqdm


class SimpleAgent:
    def __init__(self, env, eps=1.0, memory_size=int(1e5), batch_size=256, action_space_discrete=True, max_steps=100, device="cpu"):
        """
        Constructor.
        
        Params:
            env: Unity environment
        """
        self.env = env
        self.max_steps = max_steps
        self.memory = Memory(maxlen=memory_size, action_space_discrete=action_space_discrete, device=device)
        self.batch_size = batch_size
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = None
        self.state = None
        self.score = 0
        self.score_hist = []
        self.reset()
        
    def reset(self):
        """
        Reset the environment.
        """
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.state = None
        self.score = 0
        self.score_hist = []
        self.update_state()
        
    def update_state(self):
        """
        Get current state from environment and update internal state values.
        
        Returns:
            (int, int): tuple containing state before and after update
        """
        previous_state = self.state
        next_state = self.env_info.vector_observations[0]
        self.state = next_state
        return previous_state, next_state
        
    def update_memory(self, state, action, reward, next_state, done):
        """
        Save step results in memory.
        
        Args:
            state: state before step was taken
            action: action that was taken by the step
            reward: reward that was earned by performing the step
            next_state: state after step was performed
            done: indocator whether game was over after step
        """
        self.memory.append(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
    
    def choose_action(self):
        """
        Choose an action.
        
        Returns:
            int: Action
        """
        return np.random.randint(self.brain.vector_action_space_size)
    
    def do_step(self, action):
        """
        Perform a step based on a chosen action.
        
        Params:
            action (int): The action that was chosen
        """
        # perform action
        self.env_info = self.env.step(action)[self.brain_name]
        # get the next state
        state, next_state = self.update_state()
        # get reward
        reward = self.env_info.rewards[0]
        # update score
        self.score += reward
        self.score_hist.append(self.score)
        # check if game is over
        done = self.env_info.local_done[0]
        return state, reward, next_state, done
    
    def learn(self):
        """
        Method for learning based on experience.
        """
        return # this is a simple agent, it wn't learn anything.
        
    def play(self, max_steps=100):
        """
        Play the game until it is finished.
        
        Args:
            max_steps (int): Maximum number of steps in one epoch
        """
        for step in range(self.max_steps):
            action = self.choose_action()
            state, reward, next_state, done = self.do_step(action=action)
            self.update_memory(state=state, action=action, reward=reward, next_state=next_state, done=done)
            self.learn()
            if done:
                break
        return self.score
    
    def save(self):
        """
        Save status of agent.
        """
        return # As the simple agent doesn't learn anything there is nothing to save here.