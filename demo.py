import gymnasium as gym
import torch

# local imports
from algorithms.common import DEFAULT_DEVICE

class Demo:
    def __init__(self, policy):
        self.policy = policy
        self.env = gym.make("Walker2d-v5", render_mode="human")

    def run(self):
        s, _ = self.env.reset()

        while True:
            s, _, terminated, truncated, _ = self.env.step(self.policy(s))
            
            if terminated or truncated:
                s, _ = self.env.reset()

class TorchDemo(Demo):
    def __init__(self, policy, device=DEFAULT_DEVICE):
        def translated_policy(s):
            s = torch.tensor(s, dtype=torch.float32).to(device)
            return policy(s).detach().cpu().numpy()

        super().__init__(translated_policy)
