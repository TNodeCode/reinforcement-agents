import torch
import torch.nn.functional as F
from src.agent.simple import SimpleAgent
from src.network.actor_critic_continuous import ActorCriticNetworkContinuous


class A2CAgent(SimpleAgent):
    def __init__(self, env, batch_size=64, lr=1e-4, gamma=0.99, max_steps=1_000, device="cpu"):
        super().__init__(env, eps=1.0, memory_size=int(1e5), batch_size=batch_size, action_space_discrete=False, device=device)
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.device = device
        
        # Initialize actor-critic network
        self.ac_network = ActorCriticNetworkContinuous(
            dim_state=len(self.state),
            dim_action=self.brain.vector_action_space_size
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.ac_network.parameters(), lr=lr)

    def choose_action(self):
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        mean, std, _ = self.ac_network(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze().numpy(), action_log_prob

    def choose_action_discrete(self):
        state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs, _ = self.ac_network(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def learn(self, rewards, log_probs, state_values, dones):
        # Compute returns
        returns = []
        Gt = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            Gt = reward + self.gamma * Gt * (1 - done)
            returns.insert(0, Gt)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        state_values = torch.stack(state_values).squeeze()

        # Calculate advantages
        advantages = returns - state_values.detach()

        # Actor loss
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss
        critic_loss = F.mse_loss(state_values, returns)

        # Total loss
        loss = actor_loss + critic_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def play(self):
        total_reward = 0

        rewards = []
        log_probs = []
        state_values = []
        dones = []

        for _ in range(self.max_steps):
            action, log_prob = self.choose_action()
            state, reward, next_state, done = self.do_step(action=action)

            _, _, state_value = self.ac_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))

            rewards.append(reward)
            log_probs.append(log_prob)
            state_values.append(state_value)
            dones.append(done)

            total_reward += reward
            state = next_state

            if done:
                break

        self.learn(rewards, log_probs, state_values, dones)
        return total_reward