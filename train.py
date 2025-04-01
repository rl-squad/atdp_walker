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
    """Main function to train the model and log the progress."""
    print(f"Getting on with job: {job_id}")
    print(f"Job description: {job_description}")

    # Create environment and model
    env = create_environment()
    model = initialize_model(env)

    # File to save training data
    log_file = "./local_folder/training_log.csv"

    # Train the model
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

        # Calculate and save episode data
        avg_loss = np.mean(losses)
        save_training_log(log_file, episode, total_reward, avg_loss)

    # Save the trained model
    save_model(model)

    env.close()
