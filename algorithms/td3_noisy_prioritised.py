import torch
import torch.nn as nn
import torch.optim as optim

# local imports
from environment import TorchEnvironment, BatchEnvironment
from algorithms.common import (
    QNetwork,
    PolicyNetwork,
    ReplayBuffer,
    PrioritisedReplayBuffer,
    GaussianSampler,
    copy_params,
    polyak_update,
    ACTION_DIM,
    DEFAULT_DEVICE
)

from algorithms.noisy_net import NoisyPolicyNetwork

class NoisyPrioritisedTD3:
    def __init__(
        self,
        batch_size=128,
        begin_learning=10000,
        policy_delay=2,
        exploration_noise_params=[0.0, 0.2],
        smoothing_noise_params=[0.0, 0.2, 0.2],
        gamma=0.99,
        q_lr=1e-4,
        policy_lr=1e-4,
        polyak=0.995,
        device=DEFAULT_DEVICE,
        debug_per=False,
        noisy_net=False,
        prioritised_xp_replay=False,
    ):
        self.noisy_net=noisy_net
        self.prioritised_xp_replay=prioritised_xp_replay

        e_mu, e_sigma = exploration_noise_params
        s_mu, s_sigma, s_clip = smoothing_noise_params
        self.exploration_noise = GaussianSampler(mean=e_mu, sigma=e_sigma, device=device)
        self.smoothing_noise = GaussianSampler(mean=s_mu, sigma=s_sigma, clip=(-s_clip, s_clip), device=device)
        self.device = device
        self.q1 = QNetwork().to(device)
        self.q1_target = QNetwork().to(device)
        self.q2 = QNetwork().to(device)
        self.q2_target = QNetwork().to(device)

        if self.noisy_net:
            self.policy = NoisyPolicyNetwork().to(device)
            self.policy_target = NoisyPolicyNetwork().to(device)
        else:
            self.policy = PolicyNetwork().to(device)
            self.policy_target = PolicyNetwork().to(device)

        self.batch_size = batch_size
        self.begin_learning = begin_learning
        self.policy_delay = policy_delay
        self.gamma = gamma
        self.polyak = polyak
        
        if self.prioritised_xp_replay:
            self.buffer = PrioritisedReplayBuffer(
                buffer_size=2**20, # Closest power of 2 to 1 mill,
                batch_size=batch_size,
                begin_learning=begin_learning,
                device=device
            )
            # PER properties
            self.debug_per = debug_per # for debugging the priorities buffer
            self.log_every = 10000 # same as benchmark_every for comparison
        else: # uniform sampling
            self.buffer = ReplayBuffer(
                buffer_size=1000000,
                batch_size=batch_size,
                device=device
            )
            # weights used to mimic PER bias correction
            self.w = torch.ones((batch_size, 1), device=device)

        copy_params(self.q1_target, self.q1)
        copy_params(self.q2_target, self.q2)
        copy_params(self.policy_target, self.policy)

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)

        # HuberLoss, since gradients magnitudes are larger in prioritised replay
        self.loss = nn.HuberLoss(reduction='none')

    def policy_action(self, s):
        with torch.no_grad():
            return self.policy(s)

    def noisy_policy_action(self, s):
        a = self.policy_action(s)
        noise = self.exploration_noise.sample(a.shape)

        return torch.clamp(a + noise, min=-1, max=1)

    def calculate_td_error(self, s, a, r, s_n, t):
        """
        batch-aware and single-sample td error calculation
        used for recalculating priorities
        """
        with torch.no_grad():
            a_target = torch.clamp(
                self.policy_target(s_n) + self.smoothing_noise.sample((self.batch_size, ACTION_DIM)),
                min=-1,
                max=1
            )
            target = r + self.gamma * (1 - t) * torch.min(
                self.q1_target(s_n, a_target),
                self.q2_target(s_n, a_target)
            )

            # Minimum estimate between (q1, q2) to correct overestimation bias
            # in estimate of TD error for priorities recalculation
            td_error = target - torch.min(
                self.q1(s, a),
                self.q2(s, a)
            )
            return td_error

    def update(self, skip_policy_update):

        if isinstance(self.buffer, PrioritisedReplayBuffer):
            s, a, r, s_n, t, w, buffer_indices = self.buffer.sample()
        else: # uniform sampling
            s, a, r, s_n, t = self.buffer.sample()
            w = self.w

        with torch.no_grad():
            a_target = torch.clamp(
                self.policy_target(s_n) + self.smoothing_noise.sample((self.batch_size, ACTION_DIM)),
                min=-1,
                max=1
            )
            target = r + self.gamma * (1 - t) * torch.min(
                self.q1_target(s_n, a_target),
                self.q2_target(s_n, a_target)
            )
        
        # If prioritised_xp_replay, folds normalised importance sampling weights into q-learning updates
        # Each error is weighted downwards to correct bias so that update distribution
        # is the same as uniform sampling.
        # Otherwise, weights are ones

        q1 = self.q1(s, a)
        loss1 = (w * self.loss(q1, target)).mean()

        self.q1_optimizer.zero_grad()
        loss1.backward()
        self.q1_optimizer.step()

        q2 = self.q2(s, a)
        loss2 = (w * self.loss(q2, target)).mean()

        self.q2_optimizer.zero_grad()
        loss2.backward()
        self.q2_optimizer.step()

        if self.prioritised_xp_replay:
            # Recalculate priorities of sampled transitions only
            td_errors = self.calculate_td_error(s, a, r, s_n, t)
            self.buffer.recalculate_priorities(buffer_indices, td_errors.squeeze())

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

    def train_batch(self, num_steps=1e6, num_envs=10, benchmark=False):

        env = BatchEnvironment(
            num_steps=num_steps,
            num_envs=num_envs,
            policy=self.policy,
            use_noisy_policy=self.noisy_net,
            benchmark=benchmark,
            device=self.device
        )

        if self.prioritised_xp_replay:
            self.buffer.sum_tree.calculate_beta_end(num_steps)
            log_every = max((self.log_every // num_envs) * num_envs, num_envs)

        s, _ = env.reset()
        
        while not env.done():
            if env.get_current_step() < self.begin_learning:
                a = 2 * torch.rand((num_envs, ACTION_DIM), device=self.device) - 1
            else:
                
                 # The noisy policy network provides the exploratory noise
                if self.noisy_net:
                    a = self.policy_action(s)
                else: # exploratory noise from gaussian
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

            if env.get_current_step() > self.begin_learning:

                # update networks
                for i in range(num_envs):
                    self.update(skip_policy_update=(i % self.policy_delay != 0))
                
                if self.prioritised_xp_replay:
                    # beta schedule for annealling bias of PER sampling after updates
                    self.buffer.sum_tree.anneal_beta(steps=num_envs)
            
            # debug priorities of buffer, checking for degeneration and diversity issues
            if self.prioritised_xp_replay and self.debug_per:

                if (env.get_current_step() >= self.begin_learning) and (env.get_current_step() % log_every == 0):
                    self.buffer.log_priorities()

                if env.done():
                    self.buffer.write_priorities_log(env.file_name)

    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
