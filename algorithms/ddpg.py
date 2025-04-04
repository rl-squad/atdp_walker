import torch
import torch.nn as nn
import torch.optim as optim

# local imports
from environment import TorchEnvironment
from algorithms.common import (
    QNetwork,
    PolicyNetwork,
    ReplayBuffer,
    GaussianSampler,
    copy_params,
    polyak_update,
    ACTION_DIM,
    DEFAULT_DEVICE
)

class DDPG:
    def __init__(
        self,
        buffer_size=1000000,
        batch_size=128,
        start_steps=10000,
        update_after=1000,
        update_every=50, exploration_noise_params=[0, 0.2],
        gamma = 0.99,
        q_lr=1e-4,
        policy_lr=1e-4,
        polyak=0.995,
        device=DEFAULT_DEVICE
    ):
        self.device = device
        self.q = QNetwork().to(device)
        self.q_target = QNetwork().to(device)
        self.policy = PolicyNetwork().to(device)
        self.policy_target = PolicyNetwork().to(device)
        self.buffer = ReplayBuffer(buffer_size, device=device)
        self.exploration_noise = GaussianSampler(mean=exploration_noise_params[0], sigma=exploration_noise_params[1], device=device)
        
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.gamma = gamma
        self.polyak = polyak

        # initially set parameters of the target networks 
        # to those from the actual networks
        copy_params(self.q_target, self.q)
        copy_params(self.policy_target, self.policy)

        # initialize optimizers for the q and policy networks
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)

        # define the loss function
        self.loss = nn.MSELoss()
    
    # returns the policy action at state s
    def policy_action(self, s):
        with torch.no_grad():
            return self.policy(s)

    # samples a policy action with exploratory noise
    def noisy_policy_action(self, s):
        a = self.policy_action(s)
        noise = self.exploration_noise.sample()

        return torch.clamp(a + noise, min=-1, max=1)
    
    # the update is implemented as described here
    # https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    def update(self):
        s, a, r, s_n, d = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            target = r + self.gamma * (1 - d) * self.q_target(s_n, self.policy_target(s_n))

        # do a gradient descent update of the
        # q network to minimize the MSBE loss
        q = self.q(s, a)
        loss = self.loss(q, target)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        # do a gradient ascent update of the policy
        # network to maximize the average state-action value
        policy_loss = -1 * self.q(s, self.policy(s)).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # shift target networks forward
        polyak_update(self.q_target, self.q, self.polyak)
        polyak_update(self.policy_target, self.policy, self.polyak)


    def train(self, num_episodes=5000, benchmark=False):
        env = TorchEnvironment(num_episodes=num_episodes, policy=self.policy, benchmark=benchmark, device=self.device)
        
        steps = 0
        s, _ = env.reset()

        while not env.done():
            if steps < self.start_steps:
                a = 2 * torch.rand((ACTION_DIM,), device=self.device) - 1
            else:
                a = self.noisy_policy_action(s)

            s_n, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated
            
            self.buffer.append(s, a, r, s_n, d.to(torch.float32))
            s = s_n
            
            if d:
                s, _ = env.reset()
            
            steps += 1

            if (steps > self.update_after) and (steps % self.update_every == 0):
                for _ in range(self.update_every):
                    self.update()
    
    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
