import torch
import torch.nn as nn
import torch.optim as optim

from environment import TorchEnvironment, BatchEnvironment
from algorithms.common import (
    QNetwork,
    ReplayBuffer,
    PrioritisedReplayBuffer,
    copy_params,
    polyak_update,
    DEFAULT_DEVICE,
    ACTION_DIM,
    STATE_DIM
)

from algorithms.stochastic_policy_net import StochasticPolicyNetwork

class SACPER:
    def __init__(
        self,
        batch_size=128,
        begin_learning=10000,
        gamma=0.99,
        polyak=0.995,
        alpha=0.2,
        q_lr=1e-3,
        policy_lr=3e-4,
        device=DEFAULT_DEVICE,
        seed=0,
        prioritised_xp_replay=False,
        debug_per=False,
    ):
        self.prioritised_xp_replay=prioritised_xp_replay

        torch.manual_seed(seed)
        self.seed = seed
        self.device = device

        # Q networks and targets (Q1, Q2 are twin critics)
        self.q1 = QNetwork().to(device)
        self.q2 = QNetwork().to(device)
        self.q1_target = QNetwork().to(device)
        self.q2_target = QNetwork().to(device)

        # Policy network
        self.policy = StochasticPolicyNetwork().to(device)

        # Initialize target networks to match original Q networks
        copy_params(self.q1_target, self.q1)
        copy_params(self.q2_target, self.q2)

        # Optimizers for Q and policy networks
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)

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

        # Hyperparameters
        self.batch_size = batch_size
        self.begin_learning = begin_learning
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha

        # HuberLoss, since gradients magnitudes are larger in prioritised replay
        self.loss = nn.HuberLoss(reduction='none')

    def policy_action(self, s):
        with torch.no_grad():
            a, _ = self.policy.sample(s)
            return a

    def calculate_td_error(self, s, a, r, s_n, t):
        """
        batch-aware and single-sample td error calculation
        used for recalculating priorities
        """
        with torch.no_grad():
            # Sample next action and log prob from policy
            a_n, logp_a_n = self.policy.sample(s_n)

            # Use clipped double Q trick
            q1_target_val = self.q1_target(s_n, a_n)
            q2_target_val = self.q2_target(s_n, a_n)
            q_target_min = torch.min(q1_target_val, q2_target_val)

            # Compute entropy-augmented Q target
            target = r + self.gamma * (1 - t) * (q_target_min - self.alpha * logp_a_n)

            # Minimum estimate between (q1, q2) to correct overestimation bias
            # in estimate of TD error for priorities recalculation
            td_error = target - torch.min(
                self.q1(s, a),
                self.q2(s, a)
            )
            return td_error

    def update(self):

        # Sample batch from replay buffer
        if isinstance(self.buffer, PrioritisedReplayBuffer):
            s, a, r, s_n, t, w, buffer_indices = self.buffer.sample()
        else: # uniform sampling
            s, a, r, s_n, t = self.buffer.sample()
            w = self.w

        with torch.no_grad():
            # Sample next action and log prob from policy
            a_n, logp_a_n = self.policy.sample(s_n)

            # Use clipped double Q trick
            q1_target_val = self.q1_target(s_n, a_n)
            q2_target_val = self.q2_target(s_n, a_n)
            q_target_min = torch.min(q1_target_val, q2_target_val)

            # Compute entropy-augmented Q target
            target = r + self.gamma * (1 - t) * (q_target_min - self.alpha * logp_a_n)

        # Compute current Q estimates
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)

        # If prioritised_xp_replay, weight error downards to correct for bias
        # in final update distribution. Otherwise, weights have no effect (1s)

        # Compute losses
        q1_loss = (w * self.loss(q1, target)).mean()
        q2_loss = (w * self.loss(q2, target)).mean()

        # Update Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # Update Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        if self.prioritised_xp_replay:
            # Recalculate priorities of sampled transitions only
            td_errors = self.calculate_td_error(s, a, r, s_n, t)
            self.buffer.recalculate_priorities(buffer_indices, td_errors.squeeze())

        # Update policy network
        a_sampled, logp = self.policy.sample(s)
        q1_pi = self.q1(s, a_sampled)
        q2_pi = self.q2(s, a_sampled)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Policy loss encourages high Q and high entropy
        policy_loss = (self.alpha * logp - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update target Q-networks using Polyak averaging
        polyak_update(self.q1_target, self.q1, self.polyak)
        polyak_update(self.q2_target, self.q2, self.polyak)

    def train_batch(self, num_steps=1e6, num_envs=10, benchmark=False):

        env = BatchEnvironment(
            num_steps=num_steps,
            num_envs=num_envs,
            policy=self.policy,
            benchmark=benchmark,
            device=self.device
        )

        if self.prioritised_xp_replay:
            self.buffer.sum_tree.calculate_beta_end(num_steps)
            log_every = max((self.log_every // num_envs) * num_envs, num_envs)

        s, _ = env.reset(seed=self.seed)
        
        while not env.done():
            if env.get_current_step() < self.begin_learning:
                a = 2 * torch.rand((num_envs, ACTION_DIM), device=self.device) - 1
            else:
                a = self.policy_action(s)

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
                for _ in range(num_envs):
                    self.update()

                if self.prioritised_xp_replay:
                    # beta schedule for annealling bias of PER sampling after updates
                    self.buffer.sum_tree.anneal_beta(steps=num_envs)
            
            # debug priorities of buffer, checking for degeneration and diversity issues
            if self.prioritised_xp_replay and self.debug_per:

                if (env.get_current_step() >= self.begin_learning) and (env.get_current_step() % log_every == 0):
                    self.buffer.log_priorities()

                if env.done():
                    self.buffer.write_priorities_log(env.file_name)

    def save_policy(self, path):
        # Save trained policy to file
        torch.save(self.policy.state_dict(), path)

    def load_policy(self, path):
        # Load policy from file
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
