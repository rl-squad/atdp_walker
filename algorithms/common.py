import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
    def __init__(
        self,
        buffer_size=1000000,
        batch_size=128,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=DEFAULT_DEVICE,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.index = 0
        self.full = False
        self.device = device

        # Pre-allocate tensors on the specified device
        self.states = torch.zeros((buffer_size, state_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros((buffer_size, 1), device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), device=device)
        self.dones = torch.zeros((buffer_size, 1), device=device)

    def append(self, state, action, reward, next_state, done):
        """Store a transition in pre-allocated tensor memory"""
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        # Update circular buffer index
        self.index = (self.index + 1) % self.buffer_size
        if self.index == 0:
            self.full = True

    def sample(self):
        """Efficiently sample tensors from pre-allocated memory"""
        max_size = self.buffer_size if self.full else self.index
        indices = torch.randint(0, max_size, (self.batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

class SumTree:
    def __init__(
        self,
        buffer_size=1048576,
        batch_size=128,
        device=DEFAULT_DEVICE,
        epsilon=1e-2,
        alpha=0.7,
        beta_zero=0.4, # optimal ß0 for proportional based priority according to Schaul et al
    ):
        self.device = device
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.alpha = alpha
        self.values = torch.zeros(2 * buffer_size - 1, device=device)
        # is_weights has length buffer size, each element is the is_weight of each transition
        self.is_weights = torch.zeros(buffer_size, device=device)
        self.buffer_size = buffer_size
        self.max_is_weight_index = -1

        self.beta_zero = beta_zero
        self.beta = beta_zero
        self.beta_end = 400000 # schedule should finish by 400,000 steps
        self.beta_start = 1000 # start updates at 1000 steps
        self.beta_current_steps = self.beta_start

    def propagate(self, tree_indices, diffs):
        """update the sums at all ancestor nodes node with the differences between old and new values"""
        while True:
            is_root = tree_indices > 0
            if is_root.all():
                break
            self.values[tree_indices] += diffs
            # traverse to immediate parent node
            tree_indices = (tree_indices - 1) // 2
        # finally update root
        self.values[tree_indices] += diffs

    def calculate_is_weights(self, leaf_indices):
        """calculates the importance sampling weights used to correct for sampling bias"""
        transition_probabilities = self.values[leaf_indices] / self.values[0]
        return (self.batch_size * transition_probabilities) ** (-1 * self.beta)

    def add(self, buffer_indices, td_errors):
        """adds a batch of transitions to the sum tree using td errors"""
        leaf_indices = buffer_indices + (self.buffer_size - 1)

        old_priorities = self.values[leaf_indices]
        # proportional based prioritization
        new_priorities = (torch.abs(td_errors) + self.epsilon) ** self.alpha
        diffs = new_priorities - old_priorities
        self.propagate(leaf_indices, diffs)

        is_weights = self.calculate_is_weights(leaf_indices)
        # keep track of all is_weights
        self.is_weights[buffer_indices] = is_weights
        # recompute max is_weight
        current_max = self.is_weights[self.max_is_weight_index]
        if is_weights.max() > current_max:
            self.max_is_weight_index = torch.argmax(self.is_weights)

    # def get_leaf_from_priority(self, tree_index, priority):
    #     """returns the leaf index corresponding to a priority"""
    #     if tree_index >= (self.buffer_size - 1):
    #         return tree_index
    #     left_value = self.values[2 * tree_index + 1]
    #     if left_value >= priority:
    #         return self.get_leaf_from_priority(2 * tree_index + 1, priority)
    #     else:
    #         return self.get_leaf_from_priority(2 * tree_index + 2, priority - left_value)

    def _get_leaves_from_priorities(self, priorities: torch.Tensor) -> torch.Tensor:
        """
        Batched version of `get_leaf_from_priority`.
        Args:
            priorities: Tensor of shape [batch_size]
        Returns:
            leaf_indices: Tensor of shape [batch_size]
        """
        indices = torch.zeros_like(priorities, dtype=torch.int, device=self.device)  # Start at root (index 0)
        while True:
            left_children = 2 * indices + 1
            right_children = left_children + 1
            # Check if the current indices have reached the leaves
            is_leaf = indices >= (self.buffer_size - 1)
            if is_leaf.all():
                break
            # Compare priorities to left/right child values
            left_values = self.values[left_children]
            go_right = priorities > left_values
            # Update indices and priorities
            indices = torch.where(go_right, right_children, left_children)
            priorities = torch.where(go_right, priorities - left_values, priorities)
        return indices

    def anneal_beta(self):
        """increase correction for sampling bias"""
        progress = (self.beta_current_steps - self.beta_start) / \
                    (self.beta_end - self.beta_start)
        
        self.beta = min(self.beta_zero + (1 - self.beta_zero) * progress, 1.0)
        # Note: dependent on how many steps made when _sample is called
        self.beta_current_steps += 1

    # def sample(self):
    #     """
    #     samples a buffer index from the tree based on its priority
    #     returns a 2-tuple:
    #         the buffer index of the transition,
    #         the importance sampling weight of the transition
    #     """
    #     max_priority = self.values[0] # at the root of the tree
    #     # priority uniformly sampled from range [0, max_priority)
    #     sampled_priority = random.random() * max_priority
    #     leaf_index = self.get_leaf_from_priority(0, sampled_priority)
    #     buffer_index = leaf_index - (self.buffer_size - 1)
    #     is_weight = self.is_weights[buffer_index]
    #     normalised_weight = is_weight / self.is_weights[self.max_is_weight_index]
    #     return buffer_index, normalised_weight

    def _sample(self):
        """
        Samples multiple indices and weights in parallel.
        Returns:
            buffer_indices: Tensor of shape [batch_size]
            normalised_weights: Tensor of shape [batch_size]
        """
        max_priority = self.values[0]  # Root node holds the sum of all priorities
        # Sample `batch_size` priorities in parallel
        sampled_priorities = torch.rand(self.batch_size, device=self.device) * max_priority
        # Vectorized tree traversal (see below for implementation)
        leaf_indices = self._get_leaves_from_priorities(sampled_priorities)
        buffer_indices = leaf_indices - (self.buffer_size - 1)
        # Batch-compute IS weights
        is_weights = self.is_weights[buffer_indices]
        normalised_weights = is_weights / self.is_weights[self.max_is_weight_index]

        if self.beta != 1.0:
            self.anneal_beta()
        
        return buffer_indices, normalised_weights

class PrioritisedReplayBuffer:
    def __init__(
        self,
        buffer_size=1048576,
        batch_size=128,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=DEFAULT_DEVICE,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.index = 0
        self.device = device
        self.sum_tree = SumTree(buffer_size=buffer_size, batch_size=batch_size, device=device)

        # Pre-allocate tensors on the specified device
        self.states = torch.zeros((buffer_size, state_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros((buffer_size, 1), device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), device=device)
        self.is_terminal = torch.zeros((buffer_size, 1), device=device)

    def append(self, state, action, reward, next_state, done, td_error):
        """Store a transition in pre-allocated tensor memory"""
        
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.is_terminal[self.index] = done
        self.sum_tree.add(self.index, td_error)
        self.index = (self.index + 1) % self.buffer_size

    def sample(self):
        """prioritised experience sampling"""

        batch_indices = torch.zeros(self.batch_size, dtype=int, device=self.device)
        normalised_is_weights = torch.zeros(self.batch_size, device=self.device)

        # fill up a batch with indices, that correspond to the replay buffer, sampled from the SumTree
        for batch_index in range(self.batch_size):
            sampled_index, normalised_is_weight = self.sum_tree.sample()
            batch_indices[batch_index] = sampled_index
            normalised_is_weights[batch_index] = normalised_is_weight

        return (
            self.states[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.next_states[batch_indices],
            self.is_terminal[batch_indices],
            normalised_is_weights
        )
    
    def _sample(self):
        """batched prioritised experience sampling"""
        buffer_indices, normalised_is_weights = self.sum_tree._sample()
        return (
            self.states[buffer_indices],
            self.actions[buffer_indices],
            self.rewards[buffer_indices],
            self.next_states[buffer_indices],
            self.is_terminal[buffer_indices],
            normalised_is_weights,
            buffer_indices
        )

    def update_priorities(self, buffer_indices, td_errors):
        """updates the priorities of a batch of buffer_indices"""
        self.sum_tree.add(buffer_indices, td_errors)

    def update_all_priorities(self, calculate_td_error):
        """update all priorities in the buffer"""        
        for i in range(self.buffer_size):
            s = self.states[i]
            a = self.actions[i]
            r = self.rewards[i]
            s_n = self.next_states[i]
            terminated = self.is_terminal[i]
            td_error = calculate_td_error(s, a, r, s_n, terminated)
            self.sum_tree.add(i, td_error)

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