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
        buffer_size=1048576,
        batch_size=128,
        device=DEFAULT_DEVICE,
        epsilon=1e-2,
        alpha=0.6, # optimal alpha for proportional based priority according to Schaul et al
        beta_zero=0.4, # optimal beta-initial
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

    def recalculate_max_is_weight_index(self):
        self.max_is_weight_index = torch.argmax(self.is_weights)

    def propagate_batch(self, tree_indices, diffs):
        """batch update the sums at all ancestor nodes with the differences between old and new values"""
        while True:
            is_root = (tree_indices == 0)
            if is_root.all():
                break
            self.values[tree_indices] += diffs
            # traverse to immediate parent nodes
            tree_indices = (tree_indices - 1) // 2
        # finally update root
        # print(self.values[0])
        self.values[tree_indices] += diffs # Does not work as intended
        # print(diffs)
        # print(diffs.sum())
        # print(self.values[0])

    def propagate(self, tree_index, diff):
        """update the sums at all ancestor nodes node with the difference between old and new value"""
        while tree_index > 0:
            self.values[tree_index] += diff[0]
            # traverse to immediate parent node
            tree_index = (tree_index - 1) // 2
        # finally update root
        self.values[tree_index] += diff[0]

    def calculate_is_weight(self, leaf_indices):
        """
        batch-aware and single-sample is_weight calculation,
        calculates the importance sampling weight used to correct for sampling bias,
        """
        transition_probabilities = self.values[leaf_indices] / self.values[0]
        return (self.batch_size * transition_probabilities) ** (-1 * self.beta)

    def add_batch(self, buffer_indices, td_errors):
        """adds a batch of transitions to the sum tree using td errors"""
        leaf_indices = buffer_indices + (self.buffer_size - 1)
        old_priorities = self.values[leaf_indices]
        # proportional based prioritization
        new_priorities = (torch.abs(td_errors) + self.epsilon) ** self.alpha
        diffs = new_priorities - old_priorities
        self.propagate_batch(leaf_indices, diffs)
        # calculate is_weights after full sum_tree update
        is_weights = self.calculate_is_weight(leaf_indices)
        # store all is_weights
        self.is_weights[buffer_indices] = is_weights

    def add(self, buffer_index, td_error):
        """adds new transition to the sum tree based on td error"""
        leaf_index = buffer_index + (self.buffer_size - 1)
        # proportional based prioritization
        new_priority = (torch.abs(td_error) + self.epsilon) ** self.alpha
        old_priority = self.values[leaf_index]
        diff = new_priority - old_priority
        # propagate diff from leaf to root
        self.propagate(leaf_index, diff)

        is_weight = self.calculate_is_weight(leaf_index)

        prev_max_is_weight_index = self.max_is_weight_index

        # keep track of the max importance sampling weight
        if is_weight >= self.is_weights[self.max_is_weight_index]:
            self.max_is_weight_index = buffer_index

        # store all is_weights
        self.is_weights[buffer_index] = is_weight

        # update if previous max index not equal to current max index
        if self.max_is_weight_index != prev_max_is_weight_index:
            self.recalculate_max_is_weight_index()

    def get_leaves_from_priorities(self, priorities):
        """returns leaf_indices corresponding to the priorities batch"""
        # each index starts at root node
        indices = torch.zeros_like(priorities, dtype=torch.int, device=self.device)
        while True:
            left_children = 2 * indices + 1
            right_children = left_children + 1
            # check if the current indices have all reached the leaves
            is_leaf = indices >= (self.buffer_size - 1)
            if is_leaf.all():
                break
            # compare priorities to left/right child values
            left_values = self.values[left_children]
            go_right = priorities > left_values
            # update indices and priorities
            indices = torch.where(go_right, right_children, left_children)
            priorities = torch.where(go_right, priorities - left_values, priorities)
        return indices

    def anneal_beta(self):
        """increase correction for sampling bias"""
        progress = (self.beta_current_steps - self.beta_start) / \
                    (self.beta_end - self.beta_start)
        
        self.beta = min(self.beta_zero + (1 - self.beta_zero) * progress, 1.0)
        # TODO dependent on how many steps made when sample_batch is called
        self.beta_current_steps += 10

    def sample_batch(self):
        """samples buffer_indices and calculates normalised weights in parallel"""
        sum_all_priorities = self.values[0] # root node
        sampled_priorities = torch.rand(self.batch_size, device=self.device) * sum_all_priorities
        # vectorized root-to-leaf traversal
        leaf_indices = self.get_leaves_from_priorities(sampled_priorities)
        buffer_indices = leaf_indices - (self.buffer_size - 1)
        # batch-compute weights
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
        self.full = False
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
        # add priority to sum tree
        self.sum_tree.add(self.index, td_error)
        self.index = (self.index + 1) % self.buffer_size
        if self.index == 0:
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
        """recalculate the priorities at the given buffer_indices"""

        # for i in range(len(buffer_indices)):
        #     self.sum_tree.add(buffer_indices[i], td_errors[i])

        self.sum_tree.add_batch(buffer_indices, td_errors)
        # # recalculate max is_weight
        self.sum_tree.recalculate_max_is_weight_index()

    def recalc_all_prios(self, calculate_td_errors_batch):
        """recalculate all priorities in the buffer, splitting the update into chunks"""
        max_size = self.buffer_size if self.full else self.index
        valid_indices = torch.arange(max_size, device=self.device)
        chunk_size = min(4096, max_size)
        for i in range(0, max_size, chunk_size):
            chunk_indices = valid_indices[i:i + chunk_size]
            s = self.states[chunk_indices].squeeze()
            a = self.actions[chunk_indices].squeeze()
            r = self.rewards[chunk_indices]
            s_n = self.next_states[chunk_indices].squeeze()
            t = self.is_terminal[chunk_indices]
            td_errors = calculate_td_errors_batch(s, a, r, s_n, t)
            self.sum_tree.add_batch(chunk_indices.unsqueeze(1), td_errors)
        # recalculate max is_weight
        self.sum_tree.recalculate_max_is_weight_index()

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