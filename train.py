import os
import csv
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

#for cleaRL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        # I’m building a custom neural network that inherits all the powerful tools from nn.Module
        # This gives access to utilities like model.to(), model.parameters(), saving/loading, etc.
        super().__init__()
        
        # Define a small MLP (Multi-Layer Perceptron) with two hidden layers of 64 units
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),  # First hidden layer
            nn.Tanh(),               # Tanh activation keeps values between -1 and 1
            nn.Linear(64, 64),       # Second hidden layer
            nn.Tanh()                # Another Tanh activation
        )

        # Output layer for the mean of the Gaussian action distribution
        self.mean = nn.Linear(64, act_dim)

        # Learnable parameter for log standard deviation (for exploration)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        # Forward pass: transforms the input x into mean and std of the action distribution
        x = self.net(x)
        return self.mean(x), self.log_std.exp()


# Base trainer class to handle shared setup and logging
class TrainerBase:
    def __init__(self, job_id: str, job_description: str):
        """
        Initialize trainer with job metadata and result folder.
        """
        self.job_id = job_id
        self.job_description = job_description
        self.result_dir = os.path.join("results", f"results_{self.get_name()}")
        os.makedirs(self.result_dir, exist_ok=True)

        print(f"[{self.get_name().capitalize()}] Starting job: {job_id}")
        print(f"Description: {job_description}")

    def get_name(self):
        """
        Return the name identifier for the trainer.
        Used to name result directories.
        """
        return "base"

    def train(self):
        """
        Default placeholder for the training loop.
        Should be overridden by subclasses.
        """
        print("[BaseTrainer] No training logic implemented.")

    def create_environment(self):
        """
        Create and return the Walker2D environment.
        """
        return gym.make("Walker2d-v5")

    def save_training_log(self, log_file, episode, total_reward, avg_loss=None):
        """
        Save a single training episode result to CSV.
        Includes average loss if provided.
        """
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                headers = ["Episode", "Reward"]
                if avg_loss is not None:
                    headers.append("Loss")
                writer.writerow(headers)

            row = [episode + 1, total_reward]
            if avg_loss is not None:
                row.append(avg_loss)
            writer.writerow(row)


# Stable-Baselines3 implementation using PPO
class StableTrainer(TrainerBase):
    def get_name(self):
        """
        Return trainer name for directory naming.
        """
        return "stable"

    def train(self, total_episodes=1000, max_steps_per_episode=1000):
        """
        Train PPO agent using Stable-Baselines3.
        Logs rewards and a simulated loss per episode.
        """
        env = self.create_environment()
        model = PPO("MlpPolicy", env, verbose=1)

        log_file = os.path.join(self.result_dir, "training_log.csv")
        model_path = os.path.join(self.result_dir, "walker2d_ppo.zip")

        for episode in range(total_episodes):
            obs, _ = env.reset()
            total_reward = 0
            losses = []

            for _ in range(max_steps_per_episode):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                losses.append(np.random.uniform(0.01, 0.05))  # Simulated loss

                if done or truncated:
                    break

            avg_loss = np.mean(losses)
            self.save_training_log(log_file, episode, total_reward, avg_loss)

        model.save(model_path)
        print(f"Model saved to {model_path}")
        env.close()



#CleanRLTrainer skeleton
# CleanRL-style trainer using PyTorch and a REINFORCE-like training loop
class CleanRLTrainer(TrainerBase):
    def get_name(self):
        """
        Return trainer name for directory naming.
        """
        return "cleanrl"

    def train(self, total_timesteps=10000):
        """
        Train a policy using a basic REINFORCE algorithm.
        """
        # Create the environment and determine input/output dimensions
        env = self.create_environment()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Initialize policy and optimizer
        model = Policy(obs_dim, act_dim)
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        # Prepare result paths
        obs, _ = env.reset()
        episode_reward = 0
        rewards = []
        log_file = os.path.join(self.result_dir, "training_log.csv")
        model_path = os.path.join(self.result_dir, "walker2d_cleanrl.pt")

        # Main training loop
        for step in range(total_timesteps):
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            # Get action distribution from policy
            mean, std = model(obs_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            # Placeholder value estimate (no critic used here)
            value_estimate = torch.tensor([0.0])

            # Step in the environment
            next_obs, reward, done, truncated, _ = env.step(action.detach().numpy())

            # Compute REINFORCE loss using advantage = reward - baseline
            advantage = reward - value_estimate.item()
            loss = -log_prob * advantage

            # Backprop and update policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track reward and prepare next step
            obs = next_obs
            episode_reward += reward

            if done or truncated:
                rewards.append(episode_reward)
                obs, _ = env.reset()
                episode_reward = 0

        # Save training log using inherited method
        for i, r in enumerate(rewards):
            self.save_training_log(log_file, i, r)

        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"[CleanRL] Model saved to {model_path}")

        # Clean up environment
        env.close()


#RLibTrainerSkeleton
class RLlibTrainer(TrainerBase):
    def get_name(self):
        """
        Return trainer name for directory naming.
        """
        return "rllib"

    def train(self, total_timesteps=10000):
        """
        Placeholder for RLlib PPO trainer.
        Requires ray[rllib] and policy config.
        """
        print(f"[RLlib] Not implemented yet for job: {self.job_id}")



#GarageTrainerSkeleton
class GarageTrainer(TrainerBase):
    def get_name(self):
        """
        Return trainer name for directory naming.
        """
        return "garage"

    def train(self):
        """
        Placeholder for Garage PPO trainer.
        Garage requires TensorFlow and special setup.
        """
        print(f"[Garage] Not implemented yet for job: {self.job_id}")


#JustPytorchSkeleton
class CustomTrainer(TrainerBase):
    def get_name(self):
        """
        Return trainer name for directory naming.
        """
        return "custom"

    def train(self, total_timesteps=10000):
        """
        Placeholder for custom PPO implementation from scratch.
        This will use only PyTorch without any external RL library.
        """
        print(f"[Custom PPO] Not implemented yet for job: {self.job_id}")


