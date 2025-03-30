import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# local imports
from environment import Environment

# declaring the state and action dimensions as constants
STATE_DIM = 17
ACTION_DIM = 6

# dynamically infer device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# this is the action value function over the State x Action space
# modelled as a neural network with 3 hidden layers.
# this function maps to a single scalar value which corresponds
# to the state-action value. please note that the constructor loads
# random initial parameter values by default
class QNetwork(nn.Module):
    def __init__(self, hidden_sizes=[512, 256, 128]):
        super(QNetwork, self).__init__()
        input_dim = STATE_DIM + ACTION_DIM
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.out = nn.Linear(hidden_sizes[2], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return self.out(x)

# this is the policy network over the state space S
# modelled as a neural network with 3 hidden layers.
# this function maps to a scalar vector of size 6 (representing an action)
# where each scalar is bounded within -1 and 1 by a tanh transform
class PolicyNetwork(nn.Module):
    def __init__(self, hidden_sizes=[512, 256, 128]):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.out = nn.Linear(hidden_sizes[2], ACTION_DIM)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.out(x))
        return action

# this is a utility class that constructs a buffer of a specified size
# and begins overwriting from the start once full. this class also
# exposes a sample method which allows us to generate a random sample
# of a specified size from the buffer. this is used to bootstrap samples
# for the Q and Policy network updates
class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.index = 0
        self.full = False

    def append(self, item):
        self.buffer[self.index] = item
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True

    # generate a random sample of batch_size from the buffer
    def sample(self, batch_size):
        import random
        return random.sample(self.buffer if self.full else self.buffer[:self.index], batch_size)


class DDPG:
    def __init__(self, buffer_size=1000000, batch_size=128, warmup=10000, q_update_after=1, policy_update_after=50, action_noise_params=[0, 0.2, 0.999], gamma = 0.99):
        self.q = QNetwork().to(device)
        self.q_target = QNetwork().to(device)
        self.policy = PolicyNetwork().to(device)
        self.policy_target = PolicyNetwork().to(device)
        self.buffer = RingBuffer(buffer_size)
        
        self.batch_size = batch_size
        self.warmup = warmup
        self.q_update_after = q_update_after
        self.policy_update_after = policy_update_after
        self.action_noise_params = action_noise_params
        self.gamma = gamma

        # initially set parameters of the target networks 
        # to those from the actual networks
        copy_params(self.q_target, self.q)
        copy_params(self.policy_target, self.policy)

        # initialize optimizers for the q and policy networks
        self.q_optimizer = optim.Adam(list(self.q.parameters()), lr=1e-3)
        self.policy_optimizer = optim.Adam(list(self.policy.parameters()), lr=1e-4)

        # define the loss function
        self.loss = nn.MSELoss()
    
    # returns the policy action at state s
    def policy_action(self, s):
        with torch.no_grad():
            return self.policy(s)

    # samples a policy action with random Gaussian noise
    def noisy_policy_action(self, s):
        a = self.policy_action(s)  
        noise = torch.normal(mean=self.action_noise_params[0], std=self.action_noise_params[1], size=a.shape).to(device)

        return torch.clamp(a + noise, min=-1, max=1)
    
    def sample_batch(self):
        samples = self.buffer.sample(self.batch_size)
        
        s = torch.cat([sample[0].unsqueeze(0) for sample in samples], dim=0).to(device)
        a = torch.cat([sample[1].unsqueeze(0) for sample in samples], dim=0).to(device)
        r = torch.tensor([sample[2] for sample in samples], dtype=torch.float32).unsqueeze(1).to(device)
        s_n = torch.cat([sample[3].unsqueeze(0) for sample in samples], dim=0).to(device)
        d = torch.tensor([sample[4] for sample in samples], dtype=torch.float32).unsqueeze(1).to(device)

        return s, a, r, s_n, d
    
    # the q update is implemented as described here
    # https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    def update_q(self):
        s, a, r, s_n, d = self.sample_batch()

        target = r + self.gamma * (1 - d) * self.q_target(s_n, self.policy_target(s_n))

        # do a gradient descent update of the
        # q network to minimize the MSBE loss
        current = self.q(s, a)
        loss = self.loss(current, target)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        # shift q target forward
        polyak_update(self.q_target, self.q)

    # the policy update is implemented as described here
    # https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    def update_policy(self):
        s, _, _, _, _ = self.sample_batch()

        # do a gradient ascent update of the policy
        # network to maximize the average state-action value
        mean_q = -1 * torch.mean(self.q(s, self.policy(s)))
        
        self.policy_optimizer.zero_grad()
        mean_q.backward()
        self.policy_optimizer.step()

        # shift policy target forward
        polyak_update(self.policy_target, self.policy)


    def train(self, num_episodes=5000):
        env = Environment(num_episodes=num_episodes, use_torch=True, torch_device=device)
        
        steps = 0
        s, _ = env.reset()

        while not env.done():
            if steps < self.warmup:
                a = torch.distributions.Uniform(-1, 1).sample((ACTION_DIM,)).to(device)
            else:
                a = self.noisy_policy_action(s)

            s_n, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated
            
            self.buffer.append((s, a, r, s_n, d.to(torch.float32)))
            s = s_n
            
            if d:
                s, _ = env.reset()
            
            steps += 1

            if steps > self.warmup:
                if steps % self.q_update_after == 0:
                    self.update_q()
                
                if steps % self.policy_update_after == 0:
                    self.update_policy()

        print(steps)

# copies params from a source to a target network
def copy_params(target_net, source_net):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(source_param.data)

# performs an update of the target network parameters via Polyak averaging
# where target_params = p * target_params + (1 - p) * source_params
def polyak_update(target_net, source_net, p=0.995):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(p * target_param.data + (1 - p) * source_param.data)

ddpg = DDPG(warmup=10000, policy_update_after=1, q_update_after=50, action_noise_params=[0, 0.2, 1])
ddpg.train(num_episodes=10000)