import torch
import torch.nn as nn
import torch.optim as optim

# local imports
from environment import TorchEnvironment
from algorithms.common import (
    QNetwork,
    PolicyNetwork,
    ReplayBuffer,
    OrnsteinUhlenbeckSampler,
    copy_params,
    polyak_update,
    ACTION_DIM,
    DEFAULT_DEVICE
)
    
class TD3:
    def __init__(self, buffer_size=1000000, batch_size=128, start_steps=10000, update_after=1000, update_every=50, policy_delay=2, action_noise_params=[0, 0.2], target_noise_params=[0, 0.2, 0.2], gamma=0.99, q_lr=1e-4, policy_lr=1e-4, polyak=0.995, device=DEFAULT_DEVICE):
        self.device = device
        self.q1 = QNetwork().to(device)
        self.q1_target = QNetwork().to(device)
        self.q2 = QNetwork().to(device)
        self.q2_target = QNetwork().to(device)
        self.policy = PolicyNetwork().to(device)
        self.policy_target = PolicyNetwork().to(device)
        self.buffer = ReplayBuffer(buffer_size, device=device)
        self.noise_sampler = OrnsteinUhlenbeckSampler(mean=action_noise_params[0], sigma=action_noise_params[1], device=device)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.policy_delay = policy_delay
        self.action_noise_params = action_noise_params
        self.target_noise_params = target_noise_params
        self.gamma = gamma
        self.polyak = polyak

        copy_params(self.q1_target, self.q1)
        copy_params(self.q2_target, self.q2)
        copy_params(self.policy_target, self.policy)

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
    
        self.loss = nn.MSELoss()

    def policy_action(self, s):
        with torch.no_grad():
            return self.policy(s)
        
    def noisy_policy_action(self, s):
        a = self.policy_action(s)

        noise = self.noise_sampler.sample()

        return torch.clamp(a + noise, min=-1, max=1)
    
    def update(self, skip_policy_update):
        s, a, r, s_n, d = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            a_target = self.policy_target(s_n)
            clipped_noise = torch.clamp(torch.normal(mean=self.target_noise_params[0], std=self.target_noise_params[1], size=a_target.shape), min=-self.target_noise_params[2], max=self.target_noise_params[2])
            a_target_noisy = torch.clamp(a_target + clipped_noise, min=-1, max=1)

            target = r + self.gamma * (1 - d) * torch.min(self.q1_target(s_n, a_target_noisy), self.q2_target(s_n, a_target_noisy))
        
        q1 = self.q1(s, a)
        loss1 = self.loss(q1, target)

        self.q1_optimizer.zero_grad()
        loss1.backward()
        self.q1_optimizer.step()

        q2 = self.q2(s, a)
        loss2 = self.loss(q2, target)

        self.q2_optimizer.zero_grad()
        loss2.backward()
        self.q2_optimizer.step()

        if skip_policy_update:
            return
        
        # do a gradient ascent update of the policy network
        # while keeping the q network constant
        policy_loss = -1 * self.q1(s, self.policy(s)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update target networks whenever policy is updated
        polyak_update(self.q1_target, self.q1, self.polyak)
        polyak_update(self.q2_target, self.q2, self.polyak)
        polyak_update(self.policy_target, self.policy, self.polyak)

    def train(self, num_episodes=5000, benchmark=False):
        env = TorchEnvironment(num_episodes=num_episodes, policy=self.policy, benchmark=benchmark, device=self.device)

        steps = 0
        s, _ = env.reset()

        while not env.done():
            if steps < self.start_steps:
                a = torch.distributions.Uniform(-1, 1).sample((ACTION_DIM,)).to(self.device)
            else:
                a = self.noisy_policy_action(s)
        
            s_n, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated

            self.buffer.append(s, a, r, s_n, d.to(torch.float32))
            s = s_n

            if d:
                s, _ = env.reset()

            steps += 1

            if steps > self.update_after and steps % self.update_every == 0:
                for i in range(self.update_every):
                    self.update(skip_policy_update=i % self.policy_delay != 0)
    
    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))