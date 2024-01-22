import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym

import numpy as np

# RND

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class RunningMeanStd():
    def __init__(self,
                 device):
        self.mean = torch.zeros(1).to(device)
        self.var = torch.ones(1).to(device)
        self.count = 0

    def update(self, x):
        batch_mean, batch_std, batch_count = torch.mean(x, dim = 0), torch.std(x, dim = 0), x.shape[0]
        batch_var = batch_std ** 2
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RND_NET(nn.Module):
    def __init__(self, 
                 input_size,
                 output_size,
                 device,
                 hidden_size,
                 lr,
                 env_spec):
        super().__init__()
                        
        self.random_net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                  Swish(),
                                  nn.Linear(hidden_size, hidden_size),
                                  Swish(),
                                  nn.Linear(hidden_size, hidden_size),
                                  Swish(),
                                  nn.Linear(hidden_size, output_size))
        
        self.predictor_net = nn.Sequential(nn.Linear(input_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, output_size))
        
        for m in self.modules():
            if type(m) is nn.Linear:
                nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # Input scaler parameters
        self.mu = nn.Parameter(torch.zeros((1, input_size)), requires_grad = False)
        self.sigma = nn.Parameter(torch.ones((1, input_size)), requires_grad = False)
        
        # Penalty Scaler (Std of cumulative penalty)
        self.penalty_scaler = RunningMeanStd(device)
                        
        self.input_size = input_size
        
        self.device = device
        self.optim = optim.Adam(self.predictor_net.parameters(), lr = lr)
        
        self.env_spec = env_spec
        
        self.to(device)
                        
    def forward(self, comb):
            
        output = self.predictor_net(comb)
                            
        return output
                                
    def update_parameters(self, replay_buffer, n_epochs = 5, batch_size = 32):
        
        # Shape: 0: Obs 1: n_step Index 2: Dim
                
        state_seq, action_seq = [np.array(t) for 
                                      t in zip(*replay_buffer)]
        
        states = state_seq[:, 0, :]
        actions = action_seq[:, 0, :]
        
        n = states.shape[0]
                
        # Shape: 0: Obs Index
        
        train_set_ind = np.random.permutation(n)
        
        n_train = n 
        
        inputs = np.concatenate((self.env_spec.state_preproc(states), actions), axis = 1)
        
        mu = np.mean(inputs, axis = 0, keepdims = True)
        sigma = np.std(inputs, axis = 0, keepdims = True)
        sigma[sigma < 1e-12] = 1.0
        
        self.mu.data = torch.FloatTensor(mu).to(self.device)
        self.sigma.data = torch.FloatTensor(sigma).to(self.device)
                
        for i_epoch in range(n_epochs):
            
            for batch_n in range(int(np.ceil(n_train / batch_size))):
                
                batch_ind = train_set_ind[batch_n * batch_size:(batch_n + 1) * batch_size]
                                                
                states_batch = states[batch_ind, :]
                actions_batch = actions[batch_ind, :]
                
                inputs = np.concatenate((self.env_spec.state_preproc(states_batch), actions_batch), axis = 1)
                inputs = torch.FloatTensor(inputs).to(self.device)
                
                inputs = (inputs - self.mu)/self.sigma
                
                loss = ((self.predictor_net(inputs) - self.random_net(inputs)) ** 2).mean()
                
                self.optim.zero_grad()                                             
                loss.backward()
                self.optim.step()
                
            train_set_ind = np.random.permutation(train_set_ind)
            
    def penalty(self, state_action_seqs):
        
        # Assumes 3-D CUDA tensor input with 0: Trajectory Ind, 1: Horizon Ind, 2: Dim (state-action pairs)
        
        inputs = (state_action_seqs - self.mu)/self.sigma
        
        with torch.no_grad():
            penalty = (self.predictor_net(inputs) - self.random_net(inputs)) ** 2
        penalty = penalty.sum(-1).sum(-1)
        
        self.penalty_scaler.update(penalty)
        
        return penalty#/self.penalty_scaler.var.sqrt()