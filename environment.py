import numpy as np
import os
from tqdm import tqdm
import gymnasium as gym

class Environment:
    def __init__(self, num_episodes, render_mode = "rgb_array"):
        # no output dir raise value error
        file_name = os.getenv("OUT")
        if file_name is None:
            raise ValueError
        os.makedirs("./out", exist_ok=True)
        self.file_name = file_name
        self.env = gym.make("Walker2d-v5", render_mode = render_mode)
        self.pbar = tqdm(total=num_episodes)
        self.num_episodes = num_episodes
        self.episode = 0
        self.episode_rewards = []
        self.total_episode_reward = 0

    def step(self, action):
        
        if self.done():
            raise IndexError("All episodes have terminated")

        observation, reward, terminated, truncated, info = self.env.step(action)

        self.total_episode_reward += reward

        if terminated or truncated:
            self.episode_rewards.append(self.total_episode_reward)
            self.total_episode_reward = 0
            self.episode += 1
            self.pbar.update()

            if self.episode == self.num_episodes:

                # Write rewards to file
                np.save(f"out/{self.file_name}.npy", self.episode_rewards)

                # Cleanup
                self.env.close()
                self.pbar.close()

        return observation, reward, terminated, truncated, info

    def reset(self):
        
        if self.done():
            raise IndexError("All episodes have terminated")
        
        return self.env.reset()
    
    def done(self):
        return self.episode == self.num_episodes