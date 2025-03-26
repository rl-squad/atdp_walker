from environment import Environment

class QAgent:
    def __init__(self, num_episodes):
        self.env = Environment(num_episodes=num_episodes)
        self.q = {}

    def train(self):
        action = self.choose_action()

        terminated = False
        while not terminated:
            observation, reward, terminated, truncated, info  = self.env.step(action)

            if terminated or truncated:
                observation, _ = self.env.reset()
    
    def choose_action():
        pass

num_episodes = 1000

q = QAgent(num_episodes)
for _ in range(num_episodes):
    q.train()