from stable_baselines3 import PPO
import gymnasium as gym

# Load the trained model
model = PPO.load("walker2d_ppo")

# Create the environment with human rendering
env = gym.make("Walker2d-v5", render_mode="human")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)

    if done or truncated:
        obs, _ = env.reset()

env.close()
