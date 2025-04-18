import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common import ACTION_DIM, STATE_DIM

# this is the policy network over the state space S
# modelled as a neural network with 2 hidden layers.
# this function maps to a scalar vector of size 6 (representing an action)
# where each scalar is bounded within -1 and 1 by a tanh transform
class NoisyPolicyNetwork(nn.Module):
    def __init__(self, hidden_sizes=[256, 256]):
        super(NoisyPolicyNetwork, self).__init__()
        self.fc1 = NoisyLinear(STATE_DIM, hidden_sizes[0])
        self.fc2 = NoisyLinear(hidden_sizes[0], hidden_sizes[1])
        self.out = NoisyLinear(hidden_sizes[1], ACTION_DIM)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.out(x))
        return action
    
class NoisyLinear(nn.Module):
    def __init__(self, in_dims, out_dims, epsilon_sigma=1, init_weight_sigma=0.017, init_bias_sigma=0.017):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.init_weight_sigma = init_weight_sigma
        self.init_bias_sigma = init_bias_sigma

        self.register_buffer("epsilon_sigma", torch.tensor(epsilon_sigma))

        # define the weight parameters explicitly
        self.weight_mu = nn.Parameter(torch.empty((out_dims, in_dims)))
        self.weight_sigma = nn.Parameter(torch.empty((out_dims, in_dims)))

        # define the bias parameters explicitly 
        self.bias_mu = nn.Parameter(torch.empty((out_dims,)))
        self.bias_sigma = nn.Parameter(torch.empty((out_dims,)))

        self.reset_params()

    def reset_params(self):
        bound = 1/self.in_dims ** 0.5

        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        nn.init.constant_(self.weight_sigma, self.init_weight_sigma)
        nn.init.constant_(self.bias_sigma, self.init_bias_sigma)

    def forward(self, x):
        weight_noise = torch.normal(
            mean = 0,
            std = self.epsilon_sigma,
            size = self.weight_sigma.shape,
            device = self.weight_sigma.device
        )

        bias_noise = torch.normal(
            mean = 0,
            std = self.epsilon_sigma,
            size = self.bias_sigma.shape,
            device = self.bias_sigma.device
        )

        weight = self.weight_mu + self.weight_sigma * weight_noise
        bias = self.bias_mu + self.bias_sigma * bias_noise

        return F.linear(x, weight, bias)