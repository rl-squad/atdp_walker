from environment import Environment
import time

class AgentTemplate:
    def __init__(self, num_episodes):
        self.env = Environment(num_episodes=num_episodes)

    def play_episode(self):

        # replace initial action
        action = self.env.env.action_space.sample()
        observation, _ = self.env.reset()

        # agent needs initial action and observation to make another action

        while True:

            observation, reward, terminated, truncated, info  = self.env.step(action)

            # update model

            if terminated or truncated:
                break

    def choose_action(self):
        pass

    # def train(self):
    #     while not self.env.done():
    #         time.sleep(1)
    #         self.play_episode()

num_episodes = 5
q = AgentTemplate(num_episodes=num_episodes)
for _ in range(num_episodes):
    time.sleep(1)
    q.play_episode()