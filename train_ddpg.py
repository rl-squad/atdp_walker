import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# local imports
from environment import Environment

# declaring the state and action dimensions as constants
STATE_DIM = 17
ACTION_DIM = 6

# this is the action value function over the State x Action space
# modelled as a neural network with two hidden layers.
# this function maps to a single scalar value which corresponds
# to the state-action value. please note that the constructor loads
# random initial parameter values by default
class QNetwork(nn.Module):
    def __init__(self, hidden_sizes=[128, 128]):
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
# modelled as a neural network with two hidden layers.
# this function maps to a scalar vector of size 6 (representing an action)
# where each scalar is bounded within -1 and 1 by a tanh transform
class PolicyNetwork(nn.Module):
    def __init__(self, hidden_sizes=[128, 128]):
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
# of a specified size from the buffer. this is used to generate samples
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
    def __init__(self, buffer_size=1000000, batch_size=128, action_noise_params=[0, 0.2], gamma = 0.97):
        self.q = QNetwork()
        self.q_target = QNetwork()
        self.policy = PolicyNetwork()
        self.policy_target = PolicyNetwork()
        self.buffer = RingBuffer(buffer_size)
        self.batch_size = batch_size
        self.action_noise_params = action_noise_params
        self.gamma = gamma

        # initially set parameters of the target networks 
        # to those from the actual networks
        self.q_target.load_state_dict(self.q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

        # initialize optimizers for the q and policy networks
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=1e-3)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)

        # define the loss function
        self.loss = nn.MSELoss()
    
    # returns the policy action at state s
    def policy_action(self, s):
        with torch.no_grad():
            return self.policy(s)

    # samples a policy action with random Gaussian noise
    def noisy_policy_action(self, s):
        a = self.policy_action(s)
        noise = torch.normal(mean=self.action_noise_params[0], std=self.action_noise_params[1], size=a.shape)
        
        # experimental
        # noise = torch.distributions.Chi2(3).sample(a.shape)

        return torch.clamp(a + noise, min=-1, max=1)
    
    # this is where all the magic happens
    # updates are implemented exactly as prescribed here
    # https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    def update(self):
        samples = self.buffer.sample(self.batch_size)
        
        s = torch.cat([sample[0].unsqueeze(0) for sample in samples], dim=0)
        a = torch.cat([sample[1].unsqueeze(0) for sample in samples], dim=0)
        r = torch.tensor([sample[2] for sample in samples], dtype=torch.float32).unsqueeze(1)
        s_n = torch.cat([sample[3].unsqueeze(0) for sample in samples], dim=0)
        d = torch.tensor([sample[4] for sample in samples], dtype=torch.float32).unsqueeze(1)

        target = r + self.gamma * (1 - d) * self.q_target(s_n, self.policy_target(s_n))

        # do a gradient descent update of the
        # q network to minimize the MSBE loss
        current = self.q(s, a)
        loss = self.loss(current, target)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
    
        # do a gradient ascent update of the policy
        # network to maximize the average action-value
        mean_q = -1 * torch.mean(self.q(s, self.policy(s)))
        
        self.policy_optimizer.zero_grad()
        mean_q.backward()
        self.policy_optimizer.step()

        # shift target networks forwards
        polyak_update(self.q_target, self.q)
        polyak_update(self.policy_target, self.policy)



    def train(self, update_after=10000, update_every=50, num_episodes=5000):
        env = Environment(num_episodes=num_episodes, use_torch=True)
        steps = 0

        s, _ = env.reset()

        while not env.done():
            a = self.noisy_policy_action(s)
            s_n, r, terminated, truncated, _ = env.step(a)
            d = terminated or truncated
            
            self.buffer.append((s, a, r, s_n, d.to(torch.float32)))
            s = s_n
            
            if d:
                s, _ = env.reset()
            
            steps += 1
            
            if steps > update_after and steps % update_every == 0:
                self.update()
                
# performs an update of the target network parameters via Polyak averaging
# where target_params = p * target_params + (1 - p) * source_params
def polyak_update(target_net, source_net, p=0.995):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(p * target_param.data + (1 - p) * source_param.data)

ddpg = DDPG()
ddpg.train()