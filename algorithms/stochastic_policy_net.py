import torch
import torch.nn as nn

from algorithms.common import ACTION_DIM, STATE_DIM

# Stochastic policy network that outputs Gaussian distribution (mean and std)
class StochasticPolicyNetwork(nn.Module):
    def __init__(self, hidden_sizes=[256, 256]):
        super().__init__()

        # First fully-connected (dense) layer: input is state vector
        self.fc1 = nn.Linear(STATE_DIM, hidden_sizes[0])

        # Second hidden layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # Final output layers: mean and log standard deviation of action distribution
        self.mu_layer = nn.Linear(hidden_sizes[1], ACTION_DIM)
        self.log_std_layer = nn.Linear(hidden_sizes[1], ACTION_DIM)

    def forward(self, state):
        # Apply ReLU activation to hidden layers
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        # Output mean of the Gaussian
        mu = self.mu_layer(x)

        # Output log standard deviation, then exponentiate to get std
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)  # constrain log_std range
        std = torch.exp(log_std)

        return mu, std

    def sample(self, state):
        # Sample actions from squashed Gaussian using reparameterization trick
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()  # allows gradients to flow

        action = torch.tanh(z)  # squash action to [-1, 1] range

        # Correct the log probability due to tanh squashing
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def mean_action(self, state):
        # Deterministic action used for evaluation (no sampling)
        mu, _ = self.forward(state)
        return torch.tanh(mu)