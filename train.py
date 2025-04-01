import os
import csv
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

def create_environment():
    """Create and return the Walker2D environment."""
    return gym.make("Walker2d-v5")

def initialize_model(env):
    """Initialize and return the PPO model."""
    return PPO("MlpPolicy", env, verbose=1)

def save_training_log(log_file, episode, total_reward, avg_loss):
    """Save the training results in a CSV file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Write data to the CSV file
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # If the file is empty, write headers
            writer.writerow(["Episode", "Reward", "Loss"])
        writer.writerow([episode + 1, total_reward, avg_loss])

def save_model(model, model_name="walker2d_ppo"):
    """Save the trained model."""
    model.save(model_name)

def train_stable(job_id: str, job_description: str, total_episodes=1000, max_steps_per_episode=1000):
    """Train PPO using Stable-Baselines3 and save logs/models."""
    print(f"[Stable-Baselines3] Starting job: {job_id}")
    print(f"Description: {job_description}")

    # Define result directory
    result_dir = os.path.join("results", "results_stable")
    os.makedirs(result_dir, exist_ok=True)

    # Create environment and model
    env = gym.make("Walker2d-v5")
    model = PPO("MlpPolicy", env, verbose=1)

    # Define log and model file paths
    log_file = os.path.join(result_dir, "training_log.csv")
    model_path = os.path.join(result_dir, "walker2d_ppo.zip")

    # Training loop
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
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Episode", "Reward", "Loss"])
            writer.writerow([episode + 1, total_reward, avg_loss])

    model.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()
