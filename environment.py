import numpy as np
import os
from tqdm import tqdm
import gymnasium as gym
import torch

class Environment:
    def __init__(self, num_episodes, render_mode = "rgb_array", use_torch = False, torch_device=None):
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
        self.use_torch = use_torch

        if use_torch:
            if torch_device is not None:
                self.torch_device = torch_device
            else:
                self.torch_device = torch.device("cpu")

    def step(self, action):
        if self.done():
            raise IndexError("All episodes have terminated")

        if self.use_torch:
            action = action.detach().cpu().numpy()

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

        if self.use_torch:
            observation = torch.tensor(observation, dtype=torch.float32).to(self.torch_device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.torch_device)
            terminated = torch.tensor(terminated, dtype=torch.bool).to(self.torch_device)
            truncated = torch.tensor(truncated, dtype=torch.bool).to(self.torch_device)

        return observation, reward, terminated, truncated, info

    def reset(self):
        s, info = self.env.reset()
        
        if self.use_torch:
            s = torch.tensor(s, dtype=torch.float32).to(self.torch_device)

        return s, info
    
    def done(self):
        return self.episode == self.num_episodes