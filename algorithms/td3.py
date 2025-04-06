import torch
import torch.nn as nn
import torch.optim as optim

# local imports
from environment import TorchEnvironment, BatchEnvironment
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
    
class TD3:
    def __init__(
        self,
        buffer_size=1000000,
        batch_size=128,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        policy_delay=2,
        exploration_noise_params=[0.0, 0.2],
        smoothing_noise_params=[0.0, 0.2, 0.2],
        gamma=0.99,
        q_lr=1e-4,
        policy_lr=1e-4,
        polyak=0.995,
        device=DEFAULT_DEVICE
    ):
        e_mu, e_sigma = exploration_noise_params
        s_mu, s_sigma, s_clip = smoothing_noise_params
        
        self.device = device
        self.q1 = QNetwork().to(device)
        self.q1_target = QNetwork().to(device)
        self.q2 = QNetwork().to(device)
        self.q2_target = QNetwork().to(device)
        self.policy = PolicyNetwork().to(device)
        self.policy_target = PolicyNetwork().to(device)
        self.buffer = ReplayBuffer(buffer_size, device=device)
        self.exploration_noise = GaussianSampler(mean=e_mu, sigma=e_sigma, device=device)
        self.smoothing_noise = GaussianSampler(mean=s_mu, sigma=s_sigma, clip=(-s_clip, s_clip), device=device)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.policy_delay = policy_delay
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
        noise = self.exploration_noise.sample(a.shape)

        return torch.clamp(a + noise, min=-1, max=1)
    
    def update(self, skip_policy_update):
        s, a, r, s_n, d = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            a_target = torch.clamp(self.policy_target(s_n) + self.smoothing_noise.sample((self.batch_size, ACTION_DIM)), min=-1, max=1)
            
            target = r + self.gamma * (1 - d) * torch.min(
                self.q1_target(s_n, a_target),
                self.q2_target(s_n, a_target)
            )
        
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
                a = 2 * torch.rand((ACTION_DIM,), device=self.device) - 1
            else:
                a = self.noisy_policy_action(s)
        
            s_n, r, terminated, truncated, _ = env.step(a)

            self.buffer.append(s, a, r, s_n, terminated.to(torch.float32))
            s = s_n

            if (terminated or truncated):
                s, _ = env.reset()

            steps += 1

            if (steps > self.update_after) and (steps % self.update_every == 0):
                for i in range(self.update_every):
                    self.update(skip_policy_update=(i % self.policy_delay != 0))
    
    def train_batch(self, num_steps=1e6, num_envs=10, benchmark=False):
        if self.update_every < num_envs:
            raise ValueError(f"the value of self.update_every must be greater than num_envs. self.update_every is currently set to {self.update_every}")

        env = BatchEnvironment(num_steps=num_steps, num_envs=num_envs, policy=self.policy, benchmark=benchmark, device=self.device)
        update_every = max((self.update_every // num_envs) * num_envs, num_envs)

        s, _ = env.reset()
        
        while not env.done():
            if env.get_current_step() < self.start_steps:
                a = 2 * torch.rand((num_envs, ACTION_DIM), device=self.device) - 1
            else:
                a = self.noisy_policy_action(s)

            s_n, r, terminated, truncated, info = env.step(a)
            d = torch.logical_or(terminated, truncated)
            
            for i in range(num_envs):
                if d[i]:
                    s_n_actual = torch.tensor(info["final_obs"][i], dtype=torch.float32, device=self.device)
                else:
                    s_n_actual = s_n[i]

                self.buffer.append(s[i], a[i], r[i], s_n_actual, terminated[i].to(torch.float32))
            
            s = s_n

            if env.get_current_step() > self.update_after and env.get_current_step() % update_every == 0:
                for i in range(update_every):
                    self.update(skip_policy_update=(i % self.policy_delay != 0))
    
    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
