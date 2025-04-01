import gymnasium as gym
import time
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

# Load visual environment
env = gym.make("Walker2d-v5", render_mode="rgb_array")

# Load the trained model and connect it to the environment
model = SAC.load("sac_walker2d_v2", env=env)

'''
num_eval_episodes = 12
env = RecordVideo(env, video_folder="Walker2D_SAC", name_prefix="eval",
                  episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)'
'''

# Reset environment
total_reward = 0

# Run the model for multiple episodes
for episode in range(1):
    print(f"\nEpisode {episode + 1}")
    obs, _ = env.reset()
    total_reward = 0

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        #time.sleep(0.01)  # slow down to see what's happening

        if terminated or truncated:
            print(f"Episode ended at step {step} with total reward: {total_reward:.2f}")
            break

env.close()
