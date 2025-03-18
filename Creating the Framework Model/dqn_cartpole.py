import torch
import torch.nn as nn
import torch.optim as optim
import gym
from framework.base_agent import BaseAgent
from framework.environment import BaseEnvironment
from framework.trainer import Trainer

# DQN Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Gym Environment Wrapper
class GymEnv(BaseEnvironment):
    def __init__(self, env_name="CartPole-v1"):
        self.env = gym.make(env_name)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    @property
    def state_dim(self):
        return self.env.observation_space.shape[0]

    @property
    def action_dim(self):
        return self.env.action_space.n

# DQN Agent
class DQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def learn(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        q_value = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = self.q_network(next_state).max(1)[0].detach()
        target = reward + (1 - done) * 0.99 * next_q_value

        loss = nn.MSELoss()(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

if __name__ == "__main__":
    env = GymEnv("CartPole-v1")
    agent = DQNAgent(env.state_dim, env.action_dim)
    trainer = Trainer(agent, env, max_episodes=500)
    trainer.train()
