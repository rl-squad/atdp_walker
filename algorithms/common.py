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
        buffer_size=2**20,
        batch_size=128,
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
        self.max_is_weight_buffer_index = -1
        self.max_prio_buffer_index = -1
        # importance sampling weights of each transition buffer-indexed
        self.is_weights = torch.zeros(buffer_size, device=device)

        self.beta_zero = beta_zero
        self.beta = beta_zero
        self.beta_end = 400000 # schedule should finish by 400,000 steps
        self.beta_start = 1000 # start updates at 1000 steps
        self.beta_current_steps = self.beta_start

    def buffer_to_leaf(self, buffer):
        """converts buffer to leaf index/indices"""
        return buffer + (self.buffer_size - 1)

    def leaf_to_buffer(self, leaf):
        """converts leaf to buffer index/indices"""
        return leaf - (self.buffer_size - 1)

    def recalculate_max_is_weight_index(self):
        self.max_is_weight_buffer_index = torch.argmax(self.is_weights)

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

            self.values.scatter_add_(0, tree_indices, diffs)
            # traverse to immediate parent nodes
            tree_indices = (tree_indices - 1) // 2

    def batch_update(self, buffer_indices, td_errors):
        """batch update the (proportial-based) priorities of the selected transitions"""

        leaf_indices = self.buffer_to_leaf(buffer_indices)
        priorities = (torch.abs(td_errors) + self.epsilon) ** self.alpha
        # update sum tree priorities
        self.batch_propagate(leaf_indices, priorities)

        # calculate importance sampling weights
        is_weights = self.calculate_is_weight(leaf_indices)

        # update to is_weights of this batch, overwrites previous
        self.is_weights[buffer_indices] = is_weights

        # update maximums local to this batch

        # If the sampled buffer indices contained the max priority index,
        # must recalculate after overwrite to reduce staleness
        if self.max_prio_buffer_index in buffer_indices:
            self.max_prio_buffer_index = torch.argmax(self.values[self.buffer_to_leaf(0):])
        else:
            max_prio = self.values[self.buffer_to_leaf(self.max_prio_buffer_index)]

            batch_max_prio, batch_max_prio_index = torch.max(priorities, dim=0)

            if batch_max_prio > max_prio:
                self.max_prio_buffer_index = buffer_indices[batch_max_prio_index]

        # If the sampled buffer indices contained the max is_weight index,
        # must recalculate after overwrite to reduce staleness
        if self.max_is_weight_buffer_index in buffer_indices:
            self.recalculate_max_is_weight_index()
        else:
            max_is_weight = self.is_weights[self.max_is_weight_buffer_index]
            batch_max_is_weight, batch_max_is_weight_index = torch.max(is_weights, dim=0)
            # slight staleness from not updating all is_weights when sum tree changes
            if batch_max_is_weight > max_is_weight:
                self.max_is_weight_buffer_index = buffer_indices[batch_max_is_weight_index]

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
        if self.max_prio_buffer_index == -1:
            priority = self.epsilon
            self.max_prio_buffer_index = buffer_index
        else:
            max_prio_leaf_index = self.buffer_to_leaf(self.max_prio_buffer_index)
            priority = self.values[max_prio_leaf_index]

        self.propagate(leaf_index, priority)

        is_weight = self.calculate_is_weight(leaf_index)

        # store all is_weights
        self.is_weights[buffer_index] = is_weight

        # Possibly wrapped around the circular buffer and still have the same max is weight index
        overwriting_max = (buffer_index == self.max_is_weight_buffer_index)

        # keep track of the max importance sampling weight
        if is_weight > self.is_weights[self.max_is_weight_buffer_index]:
            self.max_is_weight_buffer_index = buffer_index
        elif overwriting_max:
            # Note: the batch_update could put the max_index in front of the buffer index pointer but
            # this attempts to minimise extra computation
            self.recalculate_max_is_weight_index()

    def get_leaves_from_priorities(self, priorities):
        """returns leaf_indices corresponding to the priorities batch"""
        # all indices start at root node
        tree_indices = torch.zeros_like(priorities, dtype=torch.int64, device=self.device)
        while True:
            left_children = 2 * tree_indices + 1
            right_children = left_children + 1
            # check if indices have reached the leaves
            is_leaf = tree_indices >= (self.buffer_size - 1)
            if is_leaf.all():
                break
            # compare priorities to left/right child values
            left_values = self.values[left_children]
            go_right = priorities > left_values
            # update indices and priorities
            tree_indices = torch.where(go_right, right_children, left_children)
            priorities = torch.where(go_right, priorities - left_values, priorities)
        return tree_indices

    def anneal_beta(self):
        """increase correction for sampling bias"""
        progress = (self.beta_current_steps - self.beta_start) / \
                    (self.beta_end - self.beta_start)
        
        self.beta = min(self.beta_zero + (1 - self.beta_zero) * progress, 1.0)
        # TODO dependent on how many steps made when sample_batch is called
        self.beta_current_steps += 10

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
        # batch-compute weights
        is_weights = self.is_weights[buffer_indices]
        normalised_weights = is_weights / self.is_weights[self.max_is_weight_buffer_index]
        # beta schedule
        if self.beta != 1.0:
            self.anneal_beta()
        return buffer_indices, normalised_weights

class PrioritisedReplayBuffer:
    def __init__(
        self,
        buffer_size=2**20,
        batch_size=128,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=DEFAULT_DEVICE,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_index = 0
        self.full = False
        self.device = device
        self.sum_tree = SumTree(buffer_size=buffer_size, batch_size=batch_size, device=device)

        # Pre-allocate tensors on the specified device
        self.states = torch.zeros((buffer_size, state_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros((buffer_size, 1), device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), device=device)
        self.is_terminal = torch.zeros((buffer_size, 1), device=device)

    def append(self, state, action, reward, next_state, terminal):
        """
        store transition in buffer and corresponding priority in sum tree,
        not expecting batch
        """
        if state.dim() > 1:
            raise Exception("buffer does not yet support batch append")
        self.states[self.buffer_index] = state
        self.actions[self.buffer_index] = action
        self.rewards[self.buffer_index] = reward
        self.next_states[self.buffer_index] = next_state
        self.is_terminal[self.buffer_index] = terminal
        self.sum_tree.add(self.buffer_index)
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        if self.buffer_index == 0:
            self.full = True

    def sample_batch(self):
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
        self.sum_tree.batch_update(buffer_indices, td_errors)

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