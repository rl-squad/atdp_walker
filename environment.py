import numpy as np
import os
from tqdm import tqdm
import gymnasium as gym
import torch

# local imports
from algorithms.common import PolicyNetwork, copy_params, DEFAULT_DEVICE

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
    def __init__(self, num_episodes, render_mode="rgb_array", policy=None, benchmark=False, benchmark_every=10000, device=DEFAULT_DEVICE, use_noisy_policy=False):
        super().__init__(num_episodes, render_mode)

        if policy is None and benchmark:
            raise ValueError("Can't run benchmarks without a reference to the policy network")

        self.policy = policy
        self.benchmark = benchmark
        self.benchmark_every = benchmark_every
        self.device = device
        self.use_noisy_policy = use_noisy_policy  #new argument

        self.current_step = 0
        self.benchmark_results = []
    
    def step(self, action):
        action = action.detach().cpu().numpy()
        observation, reward, terminated, truncated, info = super().step(action)

        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        terminated = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device)

        if self.benchmark and self.current_step > 0 and self.current_step % self.benchmark_every == 0:
            self.benchmark_results.append(self._run_benchmark(self._policy_snapshot()))

        if self.done() and self.policy is not None:
            if self.benchmark:
                np.save(f"out/{self.file_name}_bench.npy", self.benchmark_results)

            torch.save(self.policy.state_dict(), f"out/{self.file_name}.pth")

        self.current_step += 1

        return observation, reward, terminated, truncated, info

    def reset(self):
        s, info = super().reset()
        s = torch.tensor(s, dtype=torch.float32, device=self.device)

        return s, info
    
    def _policy_snapshot(self):
        # create a new policy network and clone the current policy
        policy = PolicyNetwork().to(self.device)
        copy_params(policy, self.policy)

        return policy

    def _run_benchmark(self, policy, num_episodes=10):
        env = gym.make("Walker2d-v5", render_mode="rgb_array")

        episode_rewards = np.zeros(num_episodes)
        
        for i in range(num_episodes):
            s, _ = env.reset()
            
            while True:    
                a = policy(torch.tensor(s, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
                s, r, terminated, truncated, _ = env.step(a)

                episode_rewards[i] += r

                if terminated or truncated:
                    break
        
        average_reward = episode_rewards.mean()
        sd_reward = episode_rewards.std()

        return (average_reward, sd_reward)