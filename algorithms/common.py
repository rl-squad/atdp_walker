import torch
import torch.nn as nn
import torch.nn.functional as F

# declaring the state and action dimensions as constants
STATE_DIM = 17
ACTION_DIM = 6

# declare a default device
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# this is the action value function over the State x Action space
# modelled as a neural network with 2 hidden layers.
# this function maps to a single scalar value which corresponds
# to the state-action value. please note that the constructor loads
# random initial parameter values by default
class QNetwork(nn.Module):
    def __init__(self, hidden_sizes=[256, 256]):
        super(QNetwork, self).__init__()
        input_dim = STATE_DIM + ACTION_DIM
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.out(x)

# this is the policy network over the state space S
# modelled as a neural network with 2 hidden layers.
# this function maps to a scalar vector of size 6 (representing an action)
# where each scalar is bounded within -1 and 1 by a tanh transform
class PolicyNetwork(nn.Module):
    def __init__(self, hidden_sizes=[256, 256]):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], ACTION_DIM)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.out(x))
        return action

# this is a utility class that constructs a buffer of a specified size
# and begins overwriting from the start once full. this class also
# exposes a sample method which allows us to generate a random sample
# of a specified size from the buffer. this is used to bootstrap samples
# for the Q and Policy network updates
class ReplayBuffer:
    def __init__(self, size, state_dim=STATE_DIM, action_dim=ACTION_DIM, device=DEFAULT_DEVICE):
        self.size = size
        self.index = 0
        self.full = False
        self.device = device

        # Pre-allocate tensors on the specified device
        self.states = torch.zeros((size, state_dim), device=device)
        self.actions = torch.zeros((size, action_dim), device=device)
        self.rewards = torch.zeros((size, 1), device=device)
        self.next_states = torch.zeros((size, state_dim), device=device)
        self.dones = torch.zeros((size, 1), device=device)

    def append(self, state, action, reward, next_state, done):
        """Store a transition in pre-allocated tensor memory"""
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        # Update circular buffer index
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True

    def sample(self, batch_size):
        """Efficiently sample tensors from pre-allocated memory"""
        max_size = self.size if self.full else self.index
        indices = torch.randint(0, max_size, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
    
class GaussianSampler:
    def __init__(self, mean=0.0, sigma=0.2, clip=None, device=DEFAULT_DEVICE):
        self.mean = mean
        self.sigma = sigma
        self.clip = clip
        self.device = device
    
    def sample(self, size):
        sample = torch.normal(mean=self.mean, std=self.sigma, size=size, device=self.device)

        if self.clip is None:
            return sample
        
        return torch.clamp(sample, min=self.clip[0], max=self.clip[1])
    
    # copies params from a source to a target network
def copy_params(target_net, source_net):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(source_param.data)

# performs an update of the target network parameters via Polyak averaging
# where target_params = p * target_params + (1 - p) * source_params
def polyak_update(target_net, source_net, p):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(p * target_param.data + (1 - p) * source_param.data)