class Trainer:
    """Generic trainer to run agent-environment interactions"""
    def __init__(self, agent, env, max_episodes=1000, max_steps=500):
        self.agent = agent
        self.env = env
        self.max_episodes = max_episodes
        self.max_steps = max_steps

    def train(self):
        for episode in range(self.max_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            step = 0

            while not done and step < self.max_steps:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step += 1

            print(f"Episode {episode}, Total Reward: {total_reward}")
            if episode % 100 == 0 and hasattr(self.agent, 'save'):
                self.agent.save(f"model_episode_{episode}.pth")
