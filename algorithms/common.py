import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

        # weights returned to mimic PER buffer sampling
        self.weights = torch.ones(batch_size, 1, device=device)

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
            self.weights, # weights are 1s so that this method is compatible with PER sampling
            None # buffer_indices not used in normal replay buffer
        )

class SumTree:
    def __init__(
        self,
        buffer_size=2**20,
        batch_size=128,
        begin_learning=10000,
        device=DEFAULT_DEVICE,
        epsilon=1e-2, # added to abs tderr to prevent non-zero priorities
        alpha=0.6, # optimal alpha for proportional based priority according to Schaul et al
        beta_zero=0.4, # optimal beta-initial
    ):
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.alpha = alpha
        self.values = torch.zeros(2 * buffer_size - 1, device=device)

        # buffer-indexed
        self.max_priority_index = -1
        self.min_priority_index = 0

        self.beta_zero = beta_zero
        self.beta = beta_zero
        self.beta_end = 1000000 # schedule should finish by 1,000,000 steps
        self.beta_start = begin_learning # learning updates/non-random policy starts at 10,000 steps
        self.beta_current_steps = self.beta_start

    def buffer_to_leaf(self, buffer):
        """converts buffer to leaf index/indices"""
        return buffer + (self.buffer_size - 1)

    def leaf_to_buffer(self, leaf):
        """converts leaf to buffer index/indices"""
        return leaf - (self.buffer_size - 1)

    def calculate_is_weight(self, leaf_indices):
        """
        batch-aware and single-sample is_weight calculation,
        calculates the importance sampling weight used to correct for sampling bias,
        """
        transition_probabilities = self.values[leaf_indices] / self.values[0]
        return (self.batch_size * transition_probabilities) ** (-1 * self.beta)

    def batch_propagate(self, tree_indices, priorities):
        """
        Maintain sum tree property after priorities update at leaves.
        Batch update the sums at all ancestor nodes of each leaf_index
        with the differences between old and new priorities
        """
        # diffs calculated at the leaves, where priorities are stored
        # add new prio + substract old prio
        diffs = priorities - self.values[tree_indices]
        # propagate diffs to all ancestors
        while True:
            is_root = (tree_indices == 0)
            # all leaves have reached the root
            if is_root.all():
                # finally update root
                self.values.scatter_add_(0, tree_indices, diffs)
                return
            # update sums at this level of the tree
            self.values.scatter_add_(0, tree_indices, diffs)
            # traverse to immediate parent nodes
            tree_indices = (tree_indices - 1) // 2

    def batch_update(self, buffer_indices, td_errors, buffer_pointer):
        """batch update the (proportial-based) priorities of the selected transitions"""

        leaf_indices = self.buffer_to_leaf(buffer_indices)
        priorities = (torch.abs(td_errors) + self.epsilon) ** self.alpha
        # update sum tree priorities
        self.batch_propagate(leaf_indices, priorities)

        # update maximums/minimums local to this batch

        overwriting_max_priority_index = (self.max_priority_index in buffer_indices)

        # try to avoid costly search over the whole buffer
        if overwriting_max_priority_index:
            self.max_priority_index = torch.argmax(self.values[self.buffer_to_leaf(0):])
        else:
            global_max_priority = self.values[self.buffer_to_leaf(self.max_priority_index)]

            batch_max_priority, batch_max_priority_index = torch.max(priorities, dim=0)

            if batch_max_priority > global_max_priority:
                self.max_priority_index = buffer_indices[batch_max_priority_index]

        overwriting_min_priority_index = (self.min_priority_index in buffer_indices)

        # try to avoid costly search over the whole buffer
        if overwriting_min_priority_index:
            # Must account for the buffer being initialised to zeros
            self.min_priority_index = torch.argmin(
                # buffer pointer shows where the circular buffer has been filled to so far
                self.values[self.buffer_to_leaf(0):self.buffer_to_leaf(buffer_pointer)]
            )
        else:
            global_min_priority = self.values[self.buffer_to_leaf(self.min_priority_index)]

            batch_min_priority, batch_min_priority_index = torch.min(priorities, dim=0)

            if batch_min_priority < global_min_priority:
                self.min_priority_index = buffer_indices[batch_min_priority_index]

    def propagate(self, tree_index, priority):
        """
        Maintain sum tree property after priority update at leaf.
        Update the sums at all ancestor nodes with the difference between old and new priority
        """
        # diff calculated at the leaf, where priority is stored
        # add new prio + substract old prio
        diff = priority - self.values[tree_index]
        # propagate diff to all ancestors
        while tree_index > 0:
            self.values[tree_index] += diff
            # traverse to immediate parent node
            tree_index = (tree_index - 1) // 2
        # finally update root
        self.values[tree_index] += diff

    def add(self, buffer_index):
        """adds a new transition to the sum tree with max priority"""
        leaf_index = self.buffer_to_leaf(buffer_index)

        # give new transitions the highest priority
        if self.max_priority_index == -1:
            priority = self.epsilon
            self.max_priority_index = buffer_index
        else:
            max_prio_leaf_index = self.buffer_to_leaf(self.max_priority_index)
            priority = self.values[max_prio_leaf_index]

        # update sum tree priorities
        self.propagate(leaf_index, priority)

    def get_leaves_from_priorities(self, priorities):
        """returns leaf_indices corresponding to the priorities batch"""
        # all indices start at root node
        tree_indices = torch.zeros_like(priorities, dtype=torch.int64, device=self.device)
        while True:
            # check if indices have reached the leaves
            is_leaf = tree_indices >= (self.buffer_size - 1)
            if is_leaf.all():
                return tree_indices

            left_children = 2 * tree_indices + 1
            right_children = left_children + 1

            left_values = self.values[left_children]
            right_values = self.values[right_children]

            # added additional explicit (right_values > 0) after receiving leaves with zero priority
            # possible floating point imprecision calculation when subtracting priorities here or in sum tree propagation
            go_right = (priorities > left_values) & (right_values > 0)

            # update indices and priorities
            tree_indices = torch.where(go_right, right_children, left_children)
            priorities = torch.where(go_right, priorities - left_values, priorities)

    def anneal_beta(self, steps=1):
        """increase correction for sampling bias"""
        if self.beta == 1.0:
            return
        progress = (self.beta_current_steps - self.beta_start) / \
                    (self.beta_end - self.beta_start)
        
        self.beta = min(self.beta_zero + (1 - self.beta_zero) * progress, 1.0)
        self.beta_current_steps += steps

    def sample_batch(self):
        """
        batched sampling according to priorities in the sum tree
        returning buffer_indices of transitions sampled and associated normalised_weights
        """
        sum_all_priorities = self.values[0] # root node
        # possible to change to stratified instead of uniform sampling (appendix B.2.1 PER)
        sampled_priorities = torch.rand(self.batch_size, device=self.device) * sum_all_priorities
        # vectorised root-to-leaf traversal
        leaf_indices = self.get_leaves_from_priorities(sampled_priorities)
        buffer_indices = self.leaf_to_buffer(leaf_indices)
        # batch-compute is_weights
        is_weights = self.calculate_is_weight(leaf_indices)
        # compute max_is_weight as needed using min priority for maximum freshness
        # since batch size and the sum of all priorities is the same for all priorities
        numerator = self.values[0] ** self.beta
        min_priority = self.values[self.buffer_to_leaf(self.min_priority_index)]
        denominator = (self.batch_size * min_priority) ** self.beta
        max_is_weight =  numerator / denominator
        # each is_weight normalised by max_is_weight
        normalised_weights = is_weights / max_is_weight

        return buffer_indices, normalised_weights

class PrioritisedReplayBuffer:
    def __init__(
        self,
        buffer_size=2**20,
        batch_size=128,
        begin_learning=10000,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=DEFAULT_DEVICE,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_pointer = 0
        self.full = False
        self.device = device
        self.sum_tree = SumTree(buffer_size=buffer_size, batch_size=batch_size, begin_learning=begin_learning, device=device)

        # Pre-allocate tensors on the specified device
        self.states = torch.zeros((buffer_size, state_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros((buffer_size, 1), device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), device=device)
        self.is_terminal = torch.zeros((buffer_size, 1), device=device)

        # stores relevant priorities metrics for debugging
        self.priorities_log = []

    def append(self, state, action, reward, next_state, terminal):
        """
        store transition in buffer and corresponding priority in sum tree,
        not expecting batch
        """
        if state.dim() > 1:
            raise Exception("buffer does not yet support batch append")
        self.states[self.buffer_pointer] = state
        self.actions[self.buffer_pointer] = action
        self.rewards[self.buffer_pointer] = reward
        self.next_states[self.buffer_pointer] = next_state
        self.is_terminal[self.buffer_pointer] = terminal
        self.sum_tree.add(self.buffer_pointer)
        self.buffer_pointer = (self.buffer_pointer + 1) % self.buffer_size
        if self.buffer_pointer == 0:
            self.full = True

    def sample(self):
        """batched prioritised experience sampling"""
        buffer_indices, normalised_is_weights = self.sum_tree.sample_batch()
        return (
            self.states[buffer_indices],
            self.actions[buffer_indices],
            self.rewards[buffer_indices],
            self.next_states[buffer_indices],
            self.is_terminal[buffer_indices],
            normalised_is_weights,
            buffer_indices
        )

    def recalculate_priorities(self, buffer_indices, td_errors):
        """batch update the priorities at the given buffer_indices"""

        # Must remove duplicate buffer indices before propagating from leaves

        # get unique buffer_indices + inverse_indices (mapping to new unique buffer_indices)
        unique_buffer_indices, inverse_indices = torch.unique(buffer_indices, return_inverse=True)

        deduplicated_td_errors = torch.zeros_like(unique_buffer_indices, dtype=torch.float32)

        deduplicated_td_errors.scatter_reduce_(
            dim=0,
            index=inverse_indices, # group td_errors by duplicate buffer_indices
            src=torch.abs(td_errors),
            reduce="max", # keep abs max val from src in each group
            include_self=False # do not include inital zeros in group
        )

        self.sum_tree.batch_update(
            unique_buffer_indices,
            deduplicated_td_errors,
            self.buffer_size if self.full else self.buffer_pointer
        )

    def log_priorities(self):
        """logs various priority metrics to diagnose/debug degeneration"""
        s = self.sum_tree
        p = self.buffer_size if self.full else self.buffer_pointer
        priorities = s.values[s.buffer_to_leaf(0):p]
        numerator = s.values[0] ** s.beta
        min_priority = s.values[s.buffer_to_leaf(s.min_priority_index)]
        max_priority = s.values[s.buffer_to_leaf(s.max_priority_index)]
        denominator = (self.batch_size * min_priority) ** s.beta
        max_is_weight =  numerator / denominator
        self.priorities_log.append((
            # These are directly used in calculations
            max_priority.detach().cpu().numpy(),
            min_priority.detach().cpu().numpy(),
            max_is_weight.detach().cpu().numpy(),
            # Check whether stored max/mins are equal to actual
            torch.max(priorities).detach().cpu().numpy(),
            torch.min(priorities).detach().cpu().numpy(),
            torch.mean(priorities).detach().cpu().numpy(),
        ))

    def write_priorities_log(self, filename):
        """write logged priorities to file"""
        np.save(f"out/{filename}_priorities.npy", self.priorities_log)

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