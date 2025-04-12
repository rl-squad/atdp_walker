import torch
import torch.nn as nn
import torch.optim as optim

# local imports
from environment import TorchEnvironment, BatchEnvironment
from algorithms.common import (
    QNetwork,
    PolicyNetwork,
    PrioritisedReplayBuffer,
    GaussianSampler,
    copy_params,
    polyak_update,
    ACTION_DIM,
    DEFAULT_DEVICE
)

class DDPGPER:
    def __init__(
        self,
        buffer_size=1048576, # Closest power of 2 to 1 mill
        batch_size=128,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        exploration_noise_params=[0, 0.2],
        gamma = 0.99,
        q_lr=1e-4,
        policy_lr=1e-4,
        polyak=0.995,
        device=DEFAULT_DEVICE
    ):
        e_mu, e_sigma = exploration_noise_params
        
        self.device = device
        self.q = QNetwork().to(device)
        self.q_target = QNetwork().to(device)
        self.policy = PolicyNetwork().to(device)
        self.policy_target = PolicyNetwork().to(device)
        self.buffer = PrioritisedReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=device)
        self.exploration_noise = GaussianSampler(mean=e_mu, sigma=e_sigma, device=device)

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
        noise = self.exploration_noise.sample(a.shape)

        return torch.clamp(a + noise, min=-1, max=1)

    def calculate_td_error(self, s, a, r, s_n, terminated):
        """
        batch-aware and single-sample td error calculation
        terminated must be passed as a float32
        """
        with torch.no_grad():
            target = r + self.gamma * terminated * self.q_target(s_n, self.policy_target(s_n))
            q = self.q(s, a)
            td_error = target - q
            return td_error

    # the update is implemented as described here
    # https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    def update(self):
        s, a, r, s_n, t, w, buffer_indices = self.buffer.sample_batch()

        with torch.no_grad():
            target = r + self.gamma * (1 - t) * self.q_target(s_n, self.policy_target(s_n))

        # do a gradient descent update of the
        # q network to minimize the MSBE loss
        q = self.q(s, a)
        # Fold normalised importance sampling weights into q-learning update
        loss = self.loss(w * q, w * target)

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

        # After network updates, recalculate priorities of sampled transitions only
        td_errors = self.calculate_td_error(s, a, r, s_n, t)
        self.buffer.recalculate_priorities(buffer_indices, td_errors.squeeze())

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
                # update networks
                for _ in range(self.update_every):
                    self.update()

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

            if (env.get_current_step() > self.update_after) and (env.get_current_step() % update_every == 0):
                # update networks
                for _ in range(self.update_every):
                    self.update()

    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
