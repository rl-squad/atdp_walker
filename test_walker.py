import gymnasium as gym

env = gym.make("Walker2d-v5", render_mode="rgb_array")
observation, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random policy
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, _ = env.reset()
        
env.close()