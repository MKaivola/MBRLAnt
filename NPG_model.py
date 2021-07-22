import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import scipy.stats as stats

from collections import deque

def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]

def create_weight(size, device, bias = False):
    tensor = torch.empty(size, device = device)
    if not bias:
        nn.init.xavier_normal_(tensor)
    else:
        nn.init.zeros_(tensor)
    return nn.Parameter(tensor)

def Swish(x):
    return x * torch.sigmoid(x)

def trunc_initializer(size, std):
    trunc_norm = stats.truncnorm(-2, 2, loc = np.zeros(size), scale = np.ones(size))
    W = trunc_norm.rvs() * std
    return torch.FloatTensor(W) #nn.Parameter(W.to(device))

class NPG_Model(nn.Module):
    def __init__(self, 
                 state_size,
                 action_size,
                 output_size,
                 device,
                 env_spec,
                 ensemble_size = 4,
                 hidden_size = 512,
                 lr = 1e-3,
                 replay_buffer_size = 2500):
        super().__init__()
        
        input_size = state_size + action_size
        
        self.weights = [create_weight((ensemble_size, input_size, hidden_size), device),
                        create_weight((ensemble_size, hidden_size, hidden_size), device),
                        create_weight((ensemble_size, hidden_size, output_size), device)]
        
        # self.weights = [trunc_initializer((ensemble_size, input_size, hidden_size), 1.0 / (2.0 * np.sqrt(input_size))).to(device),
        #         trunc_initializer((ensemble_size, hidden_size, hidden_size), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device),
        #         # trunc_initializer((ensemble_size, hidden_size, hidden_size), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device),
        #         trunc_initializer((ensemble_size, hidden_size, output_size), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device)]
        # for w in self.weights:
        #     w.requires_grad = True
          
        self.biases = [create_weight((ensemble_size, 1, hidden_size), device, bias = True),
                        create_weight((ensemble_size, 1, hidden_size), device, bias = True),
                        create_weight((ensemble_size, 1, output_size), device, bias = True)]
        
        # self.biases = [torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
        #         torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
        #         # torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
        #         torch.zeros(ensemble_size, 1, output_size, requires_grad = True, device = device)]
        
        self.act_funs = [nn.ReLU(), nn.ReLU(), nn.Sequential()]
        
        # self.act_funs = [Swish, Swish, nn.Sequential()]
        
        self.mu = nn.Parameter(torch.zeros((1, input_size), device = device), requires_grad = False)
        self.sigma = nn.Parameter(torch.ones((1, input_size), device = device), requires_grad = False)
        
        self.mu_target = nn.Parameter(torch.zeros((1, output_size), device = device), requires_grad = False)
        self.sigma_target = nn.Parameter(torch.ones((1, output_size), device = device), requires_grad = False)

        self.input_size = input_size
        self.output_size = output_size
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        self.device = device
        self.optim = optim.Adam(list(self.weights) + list(self.biases), lr = lr)
        self.replay_buffer = deque(maxlen = replay_buffer_size)
        self.ensemble_size = ensemble_size
                
        self.env_spec = env_spec
                                
    def forward(self, states, actions):
        
        states_proc = self.env_spec.state_preproc(states)
                        
        comb = torch.cat([states_proc ,actions], -1)
    
        comb = (comb - self.mu)/(self.sigma + 1e-8)
              
        for weight, bias, act_fun in zip(self.weights, self.biases, self.act_funs):
            comb = torch.bmm(comb, weight) + bias
            comb = act_fun(comb)
            
        return comb
    
    def mean_prediction(self, state, act):
        state = torch.from_numpy(state).float().to(self.device).expand(self.ensemble_size, state.shape[0], -1)
        act = torch.from_numpy(act).float().to(self.device).expand(self.ensemble_size, act.shape[0], -1)
        with torch.no_grad():
            state_diffs = self.forward(state, act)
            
        prediction = state_diffs * (self.sigma_target + 1e-8) + self.mu_target
        prediction = prediction * self.mask
        
        return self.env_spec.state_postproc(state[0, :, :], prediction.mean(0)).cpu().numpy()
    
    def ensemble_prediction(self, state, act, model_index):
        state = state.expand(self.ensemble_size, 1, -1)
        act = act.expand(self.ensemble_size, 1, -1)
        with torch.no_grad():
            state_diffs = self.forward(state, act)[model_index, :, :]
            
        prediction = state_diffs * (self.sigma_target + 1e-8) + self.mu_target
        prediction = prediction * self.mask
        
        return self.env_spec.state_postproc(state[0, :, :], prediction).squeeze(0)
    
    def sample_model(self, state, act):
                
        states = self._expand_to_ts_format(state)
        acts = self._expand_to_ts_format(act)
               
        with torch.no_grad():
            state_diffs = self.forward(states, acts)
                    
        state_diffs = self._flatten_to_matrix(state_diffs)
        
        prediction = state_diffs * (self.sigma_target + 1e-8) + self.mu_target
        prediction = prediction * self.mask
        
        return self.env_spec.state_postproc(state, prediction)
    
    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [10, 5] in case of proc_obs
        reshaped = mat.view(-1, self.ensemble_size, self.ensemble_size // self.ensemble_size, dim)
        # After, [2, 5, 1, 5]

        transposed = reshaped.transpose(0, 1)
        # After, [5, 2, 1, 5]

        reshaped = transposed.contiguous().view(self.ensemble_size, -1, dim)
        # After. [5, 2, 5]

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        reshaped = ts_fmt_arr.view(self.ensemble_size, -1, self.ensemble_size // self.ensemble_size, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped
    
    def save_transition(self, state, action, next_state, reward):
        self.replay_buffer.append([state, action, next_state])
        
    def get_state_stats(self):
        return self.mu[:, :self.state_size], self.sigma[:, :self.state_size]
    
    def eval_loss(self, states, actions, next_states):
        
        states = torch.FloatTensor(states).to(self.device).expand(self.ensemble_size, -1, -1)
        actions = torch.FloatTensor(actions).to(self.device).expand(self.ensemble_size, -1, -1)
        next_states = torch.FloatTensor(next_states).to(self.device).expand(self.ensemble_size, -1, -1)
        
        targets = self.env_spec.target_proc(states, next_states)
                
        targets = (targets - self.mu_target)/(self.sigma_target + 1e-8)
        
        with torch.no_grad():
            outputs = self.forward(states, actions)
                
        loss = ((targets - outputs) ** 2).mean(-1).mean(-1)
        
        return (loss.cpu().numpy(), np.zeros(4))
        
    def reset_params(self, input_size, output_size, hidden_size, ensemble_size, device):
        self.weights = [create_weight((ensemble_size, input_size, hidden_size), device),
                        create_weight((ensemble_size, hidden_size, hidden_size), device),
                        create_weight((ensemble_size, hidden_size, output_size), device)]
        self.biases = [create_weight((ensemble_size, 1, hidden_size), device, bias = True),
                        create_weight((ensemble_size, 1, hidden_size), device, bias = True),
                        create_weight((ensemble_size, 1, output_size), device, bias = True)]
    
    # n_epochs seems quite critical to performance  25 10**4  
    def update_parameters(self, n_epochs = 100, batch_size = 200, min_grad_upds = 10**2,
                          max_grad_upds = 10**5):
        
        # Shape: 0: Obs Index 1: Dim
        
        # self.reset_params(self.input_size, self.output_size, self.hidden_size, self.ensemble_size, self.device)
        
        state_all, action_all, next_state_all = [np.array(t) for 
                                      t in zip(*self.replay_buffer)]
        
        
        gen_loss = self.eval_loss(state_all[-1000:, :], action_all[-1000:, :], next_state_all[-1000:, :])
        
        n = state_all.shape[0]
            
        n_train = n 
        
        inputs = np.concatenate((self.env_spec.state_preproc(state_all), action_all), axis = 1)
        targets = self.env_spec.target_proc(state_all, next_state_all)
        
        mu_targets = np.mean(targets, axis = 0, keepdims = True)
        sigma_targets = np.mean(np.abs(targets - mu_targets), axis = 0, keepdims = True)
        
        self.mu_target.data = torch.FloatTensor(mu_targets).to(self.device)
        self.sigma_target.data = torch.FloatTensor(sigma_targets).to(self.device)
        self.mask = self.sigma_target >= 1e-8
        
        mu = np.mean(inputs, axis = 0, keepdims = True)
        sigma = np.mean(np.abs(inputs - mu), axis = 0, keepdims = True)
        
        self.mu.data = torch.FloatTensor(mu).to(self.device)
        self.sigma.data = torch.FloatTensor(sigma).to(self.device)
                
        # Shape: 0: Ensemble 1: Obs Index
                
        bootstrap_inds = np.random.randint(n, size=[self.ensemble_size, n])
        
        epoch_counter = 0
        grad_upd_counter = 0
        
        while True:
                        
            bootstrap_inds = shuffle_rows(bootstrap_inds)
            
            for batch_n in range(int(np.ceil(n_train / batch_size))):
                
                batch_ind = bootstrap_inds[:, batch_n * batch_size:(batch_n + 1) * batch_size]
                                
                state_batch, action_batch, next_state_batch = state_all[batch_ind, :], action_all[batch_ind, :], next_state_all[batch_ind, :]
                                                    
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                
                targets = self.env_spec.target_proc(state_batch, next_state_batch)
                
                targets = (targets - self.mu_target)/(self.sigma_target + 1e-8)
                
                outputs = self.forward(state_batch, action_batch)
                
                loss = ((targets - outputs) ** 2).mean(-1).mean(-1).sum()
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                grad_upd_counter += 1
                # if grad_upd_counter >= max_grad_upds:
                #     break
                
            if grad_upd_counter >= max_grad_upds:
                # print("Model training stopped due to grad max")
                break
            
            epoch_counter += 1
            
            if grad_upd_counter < min_grad_upds:
                continue
            if epoch_counter >= n_epochs:
                # print("Model training stopped due to epoch max")
                break
            
        return gen_loss