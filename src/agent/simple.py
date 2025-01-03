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
        self.memory = Memory(maxlen=memory_size, action_space_discrete=action_space_discrete, device=device) if memory_size else None
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
        next_state = self.env_info.vector_observations
        self.state = next_state
        return previous_state, next_state
        
    def update_memory(self, states, actions, rewards, next_states, dones):
        """
        Save step results in memory.
        
        Args:
            states: states before step was taken
            actions: actions that was taken by the step
            rewards: rewards that was earned by performing the step
            next_states: states after step was performed
            dones: indocator whether game was over after step
        """
        self.memory.append(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones
        )
    
    def choose_action(self):
        """
        Choose an action.
        
        Returns:
            int: Action
        """
        return np.random.randint(self.brain.vector_action_space_size)
    
    def do_step(self, actions):
        """
        Perform a step based on a chosen action.
        
        Params:
            action (int): The action that was chosen
        """
        # perform action
        self.env_info = self.env.step(actions)[self.brain_name]
        # get the next state
        states, next_states = self.update_state()
        # get reward
        rewards = self.env_info.rewards
        # update score
        self.score += sum(rewards) / len(rewards)
        self.score_hist.append(self.score)
        # check if game is over
        dones = self.env_info.local_done
        return states, rewards, next_states, dones
    
    def learn(self):
        """
        Method for learning based on experience.
        """
        return # this is a simple agent, it wn't learn anything.
        
    def play(self):
        """
        Play the game until it is finished.
        
        Args:
            max_steps (int): Maximum number of steps in one epoch
        """
        for step in range(self.max_steps):
            actions = self.choose_action()
            states, rewards, next_states, dones = self.do_step(actions=actions)
            if self.memory is not None:
                self.update_memory(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)
            self.learn()
            if all(dones):
                break
        return self.score
    
    def save(self):
        """
        Save status of agent.
        """
        return # As the simple agent doesn't learn anything there is nothing to save here.