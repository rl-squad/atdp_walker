import os
import csv
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from huggingface_sb3 import package_to_hub



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
class CleanRLTrainer(TrainerBase):
    def get_name(self):
        """
        Return trainer name for directory naming.
        """
        return "cleanrl"

    def train(self, total_timesteps=10000):
        """
        Placeholder for CleanRL-style PPO training logic.
        To be implemented with a minimal PyTorch policy and loop.
        """
        print(f"[CleanRL] Not implemented yet for job: {self.job_id}")



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


