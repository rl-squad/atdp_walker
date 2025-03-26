from tqdm import tqdm
import gymnasium as gym


class Environment:
    def __init__(self, num_episodes, render_mode = "rgb_array"):
        self.env = gym.make("Walker2d-v5", render_mode = render_mode)
        self.pbar = tqdm(total=num_episodes)
        self.num_episodes = num_episodes
        self.episode = 0

        self.episode_rewards = []
        self.current_episode_reward = 0


    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # do boilerplate stuff
        self.current_episode_reward += reward
        
        return observation, reward, terminated, truncated, info
    
    def reset(self):
        self.pbar.update()
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
        self.episode += 1
        
        if self.episode == self.num_episodes:
            # Write rewards to file
            pass

        return self.env.reset()
       