from framework.base_agent import BaseAgent
from framework.environment import BaseEnvironment
from framework.trainer import Trainer

class SimpleAgent(BaseAgent):
    """Simple rule-based agent: Push left if pole tilts left, right otherwise"""
    def act(self, state):
        return 0 if state[2] < 0 else 1  # state[2] is pole angle

    def learn(self, state, action, reward, next_state, done):
        pass  # Rule-based agent doesn't learn

class DummyEnv(BaseEnvironment):
    """Simple test environment"""
    def reset(self):
        return [0, 0, 0, 0]

    def step(self, action):
        return [0, 0, action, 0], 1, False, {}

    @property
    def state_dim(self):
        return 4

    @property
    def action_dim(self):
        return 2

if __name__ == "__main__":
    env = DummyEnv()
    agent = SimpleAgent(env.state_dim, env.action_dim)
    trainer = Trainer(agent, env, max_episodes=10)
    trainer.train()
