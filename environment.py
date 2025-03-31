import numpy as np
import os
from tqdm import tqdm
import gymnasium as gym
import torch

class Environment:
    def __init__(self, num_episodes, render_mode = "rgb_array"):
        file_name = os.getenv("OUT")
        
        # no output dir raise value error
        if file_name is None:
            raise ValueError("no output file specified. please specify a file to output results to by setting the OUT environment variable")
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

            if self.done():
                # Write rewards to file
                np.save(f"out/{self.file_name}.npy", self.episode_rewards)

                # Cleanup
                self.env.close()
                self.pbar.close()

        return observation, reward, terminated, truncated, info

    def reset(self):
        return self.env.reset()
    
    def done(self):
        return self.episode == self.num_episodes
    

class TorchEnvironment(Environment):
    def __init__(self, num_episodes, render_mode="rgb_array", device=None):
        super().__init__(num_episodes, render_mode)

        if device is None:
            device = torch.device("cpu")
        
        self.device = device
    
    def step(self, action):
        action = action.detach().cpu().numpy()
        observation, reward, terminated, truncated, info = super().step(action)

        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        terminated = torch.tensor(terminated, dtype=torch.bool).to(self.device)
        truncated = torch.tensor(truncated, dtype=torch.bool).to(self.device)

        return observation, reward, terminated, truncated, info

    def reset(self):
        s, info = super().reset()
        s = torch.tensor(s, dtype=torch.float32).to(self.device)

        return s, info
        