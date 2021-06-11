import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym

import numpy as np

# DAE network

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class DAE_NET(nn.Module):
    def __init__(self, 
                 input_size, 
                 code_size,
                 device,
                 hidden_size,
                 lr,
                 env_spec,
                 DAE_noise_std):
        super().__init__()
                        
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size),
                                  Swish(),
                                  nn.Linear(hidden_size, hidden_size),
                                  Swish(),
                                  nn.Linear(hidden_size, hidden_size),
                                  Swish(),
                                  # nn.Linear(hidden_size, hidden_size),
                                  # Swish(),
                                  nn.Linear(hidden_size, input_size))
        
        # self.decoder = nn.Sequential(nn.Linear(code_size, hidden_size),
        #                               Swish(),
        #                               nn.Linear(hidden_size, hidden_size),
        #                               Swish(),
        #                               nn.Linear(hidden_size, hidden_size),
        #                               Swish(),
        #                               nn.Linear(hidden_size, input_size))
        for m in self.modules():
            if type(m) is nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Input scaler parameters
        self.mu = nn.Parameter(torch.zeros((1, input_size)), requires_grad = False)
        self.sigma = nn.Parameter(torch.ones((1, input_size)), requires_grad = False)
                        
        self.input_size = input_size
        self.code_size = code_size
        
        self.device = device
        self.optim = optim.Adam(self.encoder.parameters(), lr = lr)
        # self.optim = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()) , lr = lr)
        
        self.env_spec = env_spec
        
        self.DAE_noise_std = DAE_noise_std
        
        self.to(device)
                        
    def forward(self, comb):
            
        encoded = self.encoder(comb)
        
        decoded = encoded #self.decoder(encoded)
                            
        return decoded
                                
    def update_parameters(self, replay_buffer, n_epochs = 5, batch_size = 32):
        
        # Shape: 0: Obs 1: n_step Index 2: Dim
                
        state_seq, action_seq = [np.array(t) for 
                                      t in zip(*replay_buffer)]
        
        # states = state_seq[:, 0, :]
        actions = action_seq[:, 0, :]
        
        states = state_seq[:, 0:2, :]
        
        n = states.shape[0]
                
        # Shape: 0: Obs Index
        
        train_set_ind = np.random.permutation(n)
        
        n_train = n 
        
        inputs = np.concatenate((self.env_spec.state_preproc(states[:, 0, :]), actions, 
                                 self.env_spec.state_preproc(states[:, 1, :])), axis = 1)
        
        mu = np.mean(inputs, axis = 0, keepdims = True)
        sigma = np.std(inputs, axis = 0, keepdims = True)
        sigma[sigma < 1e-12] = 1.0
        
        self.mu.data = torch.FloatTensor(mu).to(self.device)
        self.sigma.data = torch.FloatTensor(sigma).to(self.device)
                
        for i_epoch in range(n_epochs):
            
            for batch_n in range(int(np.ceil(n_train / batch_size))):
                
                batch_ind = train_set_ind[batch_n * batch_size:(batch_n + 1) * batch_size]
                                                
                states_batch = states[batch_ind, :, :]
                actions_batch = actions[batch_ind, :]
                
                inputs = np.concatenate((self.env_spec.state_preproc(states_batch[:, 0, :]), actions_batch,
                                         self.env_spec.state_preproc(states_batch[:, 1, :])), axis = 1)
                inputs = torch.FloatTensor(inputs).to(self.device)
                
                inputs = (inputs - self.mu)/self.sigma
                
                # Corruption and denoising
                
                inputs_corrupted = inputs + torch.randn_like(inputs) * self.DAE_noise_std
                
                reconstruction = self.forward(inputs_corrupted)
                
                loss = ((reconstruction - inputs) ** 2).mean()        
                                             
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
            train_set_ind = np.random.permutation(train_set_ind)
            
    def penalty(self, states, actions):
        
        # Assumes 3-D CUDA tensor input with 0: Trajectory Ind, 1: Step Ind, 2: Dim
        
        inputs = torch.cat((states[:, :-1, :], actions,
                            states[:, 1:, :]), dim = -1)
        
        inputs = (inputs - self.mu)/self.sigma
        
        with torch.no_grad():
            reconstruction = self.forward(inputs)
        
        penalty = ((reconstruction - inputs) ** 2).sum(-1).sum(-1)
        
        return penalty