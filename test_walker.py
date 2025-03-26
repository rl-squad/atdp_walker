import time
import gymnasium as gym
from tqdm import tqdm

env = gym.make("Walker2d-v5", render_mode="rgb_array")
observation, _ = env.reset()

for _ in tqdm(range(1000)):
    time.sleep(1) # Just to test the progress bar works in container
    action = env.action_space.sample()  # Random policy
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, _ = env.reset()
        
env.close()