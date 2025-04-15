import time
import gymnasium as gym
import torch
from algorithms.common import PolicyNetwork, DEFAULT_DEVICE

# Load the trained policy from your ddpg checkpoint.
policy = PolicyNetwork().to(DEFAULT_DEVICE)
policy.load_state_dict(torch.load("Calum_Testing/ddpg_batch.pth", map_location=DEFAULT_DEVICE))
policy.eval()

# Initialize the environment.
env = gym.make("Walker2d-v5", render_mode="human")
observation, _ = env.reset()

# Run the agent for a number of steps.
for _ in range(1000):
    # Delay to observe rendering (adjust or remove if not needed).
    time.sleep(0.05)
    
    # Convert observation to tensor and get action from the policy.
    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(DEFAULT_DEVICE)
    with torch.no_grad():
        action = policy(obs_tensor).cpu().numpy()[0]
    
    # Step through the environment using the policy's action.
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render the frame (the environment returns RGB arrays).
    frame = env.render()
    
    # Reset the environment if done.
    if terminated or truncated:
        observation, _ = env.reset()

env.close()
