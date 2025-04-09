import torch
import torch.nn as nn
import torch.optim as optim

from environment import TorchEnvironment
from algorithms.common import (
    QNetwork,
    ReplayBuffer,
    copy_params,
    polyak_update,
    DEFAULT_DEVICE,
    ACTION_DIM,
    STATE_DIM
)


class StochasticPolicyNetwork(nn.Module):
    def __init__(self, hidden_sizes=[256, 256]):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.mu_layer = nn.Linear(hidden_sizes[1], ACTION_DIM)
        self.log_std_layer = nn.Linear(hidden_sizes[1], ACTION_DIM)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()  # reparameterization trick
        action = torch.tanh(z)

        # log_prob correction for tanh
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def mean_action(self, state):
        mu, _ = self.forward(state)
        return torch.tanh(mu)


class SAC:
    def __init__(
        self,
        buffer_size=1000000,
        batch_size=128,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        gamma=0.99,
        polyak=0.995,
        alpha=0.2,
        lr=3e-4,
        device=DEFAULT_DEVICE,
    ):
        self.device = device

        # Networks
        self.q1 = QNetwork().to(device)
        self.q2 = QNetwork().to(device)
        self.q1_target = QNetwork().to(device)
        self.q2_target = QNetwork().to(device)
        self.policy = StochasticPolicyNetwork().to(device)

        # Copy initial weights
        copy_params(self.q1_target, self.q1)
        copy_params(self.q2_target, self.q2)

        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size, device=device)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.loss_fn = nn.MSELoss()

    def update(self):
        s, a, r, s_n, d = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            a_n, logp_a_n = self.policy.sample(s_n)
            q1_target_val = self.q1_target(s_n, a_n)
            q2_target_val = self.q2_target(s_n, a_n)
            q_target_min = torch.min(q1_target_val, q2_target_val)
            target = r + self.gamma * (1 - d) * (q_target_min - self.alpha * logp_a_n)

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)

        q1_loss = self.loss_fn(q1, target)
        q2_loss = self.loss_fn(q2, target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update policy
        a_sampled, logp = self.policy.sample(s)
        q1_pi = self.q1(s, a_sampled)
        q2_pi = self.q2(s, a_sampled)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * logp - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Polyak averaging
        polyak_update(self.q1_target, self.q1, self.polyak)
        polyak_update(self.q2_target, self.q2, self.polyak)

    def train(self, num_episodes=5000, benchmark=False):
        env = TorchEnvironment(num_episodes=num_episodes, policy=self.policy, benchmark=benchmark, device=self.device)

        steps = 0
        s, _ = env.reset()

        while not env.done():
            if steps < self.start_steps:
                a = 2 * torch.rand((ACTION_DIM,), device=self.device) - 1
            else:
                with torch.no_grad():
                    a, _ = self.policy.sample(s)

            s_n, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated

            self.buffer.append(s, a, r, s_n, d.to(torch.float32))
            s = s_n

            if d:
                s, _ = env.reset()

            steps += 1

            if steps > self.update_after and steps % self.update_every == 0:
                for _ in range(self.update_every):
                    self.update()

    def save_policy(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
