import numpy as np
import os
from tqdm import tqdm
import gymnasium as gym
import torch

# local imports
from algorithms.common import copy_params, DEFAULT_DEVICE
from algorithms.stochastic_policy_net import StochasticPolicyNetwork
from algorithms.noisy_net import NoisyPolicyNetwork

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

    def reset(self, seed=0):
        return self.env.reset(seed=seed)
    
    def done(self):
        return self.episode == self.num_episodes
    

class TorchEnvironment(Environment):
    def __init__(self, num_episodes, render_mode="rgb_array", policy=None, benchmark=False, benchmark_every=10000, device=DEFAULT_DEVICE):
        super().__init__(num_episodes, render_mode)

        if policy is None and benchmark:
            raise ValueError("Can't run benchmarks without a reference to the policy network")

        self.policy = policy
        self.benchmark = benchmark
        self.benchmark_every = benchmark_every
        self.device = device
        self.current_step = 0
        self.benchmark_results = []
    
    def step(self, action):
        action = action.detach().cpu().numpy()
        observation, reward, terminated, truncated, info = super().step(action)

        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        terminated = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device)

        self.current_step += 1

        if self.benchmark and self.current_step > 0 and self.current_step % self.benchmark_every == 0:
            self.benchmark_results.append(self._run_benchmark(self._policy_snapshot()))

        if self.done() and self.policy is not None:
            if self.benchmark:
                np.save(f"out/{self.file_name}_bench.npy", self.benchmark_results)

            torch.save(self.policy.state_dict(), f"out/{self.file_name}.pth")


        return observation, reward, terminated, truncated, info

    def reset(self, seed=0):
        s, info = super().reset(seed=seed)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)

        return s, info
    
    def _policy_snapshot(self):
        # create a new policy network and clone the current policy
        policy = type(self.policy)().to(self.device)
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
    

class BatchEnvironment:
    def __init__(
        self,
        num_envs=10,
        num_steps=1000000,
        policy=None,
        benchmark=False,
        benchmark_every=10000,
        begin_learning=10000,
        device=DEFAULT_DEVICE,
        log_save_policy_weights=False
    ):

        file_name = os.getenv("OUT")
        
        # no output dir raise value error
        if file_name is None:
            raise ValueError("no output file specified. please specify a file to output results to by setting the OUT environment variable")

        self.file_name = file_name
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.policy = policy
        self.benchmark = benchmark
        self.benchmark_every = max((benchmark_every // num_envs) * num_envs, num_envs)
        self.begin_learning = begin_learning
        self.benchmark_results = []
        self.device = device
        self.current_step = 0
        self.envs = gym.vector.AsyncVectorEnv(
            [lambda: gym.make("Walker2d-v5") for _ in range(num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
        )
        self.pbar = tqdm(total=num_steps)

        # saves policy at log scale step intervals after the first update
        self.log_save_policy_weights = log_save_policy_weights
        self.log_step_intervals = {
            begin_learning + 10: "10",
            begin_learning + 100: "100",
            begin_learning + 1000: "1k",
            begin_learning + 10000: "10k",
            begin_learning + 100000: "100k",
            begin_learning + 1000000: "1m"
        }

        os.makedirs("./out", exist_ok=True)
    
    def step(self, action):
        action = action.detach().cpu().numpy()
        observation, reward, terminated, truncated, info = self.envs.step(action)

        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        terminated = torch.tensor(terminated, dtype=torch.bool, device=self.device)
        truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device)

        # saves the current policy weights at log interval steps after the first update
        if self.log_save_policy_weights and (self.current_step in self.log_step_intervals):
            torch.save(
                self.policy.state_dict(),
                f"out/{self.file_name}_{self.log_step_intervals[self.current_step]}.pth"
            )

        self.current_step += self.num_envs
        self.pbar.update(self.num_envs)

        if self.benchmark and self.current_step > 0 and self.current_step % self.benchmark_every == 0:
            self.benchmark_results.append(self._run_benchmark(self._policy_snapshot()))

        if self.done() and self.policy is not None:
            if self.benchmark:
                np.save(f"out/{self.file_name}_bench.npy", self.benchmark_results)

            torch.save(self.policy.state_dict(), f"out/{self.file_name}_end.pth")

        if self.done():
            self.envs.close()
            self.pbar.close()

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=0):
        observation, info = self.envs.reset(seed=seed)
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device)

        return observation, info

    def get_current_step(self):
        return self.current_step

    def done(self):
        return self.current_step >= self.num_steps

    def _policy_snapshot(self):
        # create a new policy network and clone the current policy
        policy = type(self.policy)().to(self.device) 
        copy_params(policy, self.policy)

        return policy

    def _run_benchmark(self, policy, num_episodes=10):
        env = gym.make("Walker2d-v5", render_mode="rgb_array")

        episode_rewards = np.zeros(num_episodes)
        
        for i in range(num_episodes):
            s, _ = env.reset(seed=i)
            
            while True:
                if isinstance(policy, StochasticPolicyNetwork):
                    a = policy.mean_action(torch.tensor(s, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
                elif isinstance(policy, NoisyPolicyNetwork):
                    a = policy(torch.tensor(s, dtype=torch.float32, device=self.device), noisy=False).detach().cpu().numpy()
                else:
                    a = policy(torch.tensor(s, dtype=torch.float32, device=self.device)).detach().cpu().numpy()

                s, r, terminated, truncated, _ = env.step(a)

                episode_rewards[i] += r

                if terminated or truncated:
                    break
        
        average_reward = episode_rewards.mean()
        sd_reward = episode_rewards.std()

        return (average_reward, sd_reward)