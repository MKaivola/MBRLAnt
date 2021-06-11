import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import scipy.stats as stats

import numpy as np
from collections import deque

from timeit import default_timer as timer

# Dynamics model

# Probabilistic Ensemble

# Predicts the next state

def Swish(x):
    return x * torch.sigmoid(x)

def trunc_initializer(size, std):
    trunc_norm = stats.truncnorm(-2, 2, loc = np.zeros(size), scale = np.ones(size))
    W = trunc_norm.rvs() * std
    return torch.FloatTensor(W) #nn.Parameter(W.to(device))

def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]

class ProbNet(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size,
                 device,
                 hidden_size,
                 n_hidden_layers,
                 ensemble_size,
                 lr,
                 replay_buffer_size,
                 holdout_ratio,
                 n_particles,
                 env_spec):
        super().__init__()
        
        # self.weights = []
        # self.weights.append(trunc_initializer((ensemble_size, input_size, hidden_size), 1.0 / (2.0 * np.sqrt(input_size)), device))
        # for _ in range(n_hidden_layers-1):
        #     self.weights.append(trunc_initializer((ensemble_size, hidden_size, hidden_size), 1.0 / (2.0 * np.sqrt(hidden_size)), device))
        # self.weights.append(trunc_initializer((ensemble_size, hidden_size, 2*output_size), 1.0 / (2.0 * np.sqrt(hidden_size)), device))
        self.weights = [trunc_initializer((ensemble_size, input_size, hidden_size), 1.0 / (2.0 * np.sqrt(input_size))).to(device),
                        trunc_initializer((ensemble_size, hidden_size, hidden_size), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device),
                        trunc_initializer((ensemble_size, hidden_size, hidden_size), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device),
                        trunc_initializer((ensemble_size, hidden_size, 2*output_size), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device)]
        for w in self.weights:
            w.requires_grad = True
        # self.biases = []
        # self.act_funs = []
        # for _ in range(n_hidden_layers):
        #     self.biases.append(nn.Parameter(torch.zeros(ensemble_size, 1, hidden_size, device = device)))
        #     self.act_funs.append(Swish)
        # self.biases.append(nn.Parameter(torch.zeros(ensemble_size, 1, 2 * (output_size), device = device)))
        # self.act_funs.append(nn.Sequential())
            
        self.biases = [torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
                        torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
                        torch.zeros(ensemble_size, 1 ,hidden_size, requires_grad = True, device = device),
                        torch.zeros(ensemble_size, 1, 2*(output_size), requires_grad = True, device = device)]
                
        self.act_funs = [Swish , Swish, Swish, nn.Sequential()]
        
        # self.max_log_var = nn.Parameter(torch.full((1, output_size), 0.5,
        #                              device = device))
        # self.min_log_var = nn.Parameter(torch.full((1, output_size), -10.0,
        #                             device = device))
        
        self.max_log_var = torch.full((1, output_size), 0.5,
                                     device = device, requires_grad = True)
        self.min_log_var = torch.full((1, output_size), -10.0,
                                    device = device, requires_grad = True)
        
        # self.mu = nn.Parameter(torch.zeros((1,input_size), device = device), requires_grad = False)
        # self.sigma = nn.Parameter(torch.ones((1,input_size), device = device), requires_grad = False)
        
        self.mu = torch.zeros((input_size), device = device)
        self.sigma = torch.zeros((input_size), device = device)

        self.input_size = input_size
        self.output_size = output_size
        
        self.device = device
        self.optim = optim.Adam(list(self.weights) + list(self.biases) + [self.max_log_var] + [self.min_log_var], lr = lr)
        # self.optim = optim.Adam(self.parameters(), lr = lr)
        self.replay_buffer = deque(maxlen = replay_buffer_size)
        self.holdout_ratio = holdout_ratio
        self.ensemble_size = ensemble_size
        
        self.n_particles = n_particles
        
        self.env_spec = env_spec
        
    def state_scaler(self, state):
        return (state - self.mu[:self.output_size])/self.sigma[:self.output_size]
    
    def state_descaler(self, state, shift = False):
        scaled = state * self.sigma[:self.output_size]
        if shift:
             scaled += self.mu[:self.output_size]
        return scaled
        
    def get_decays(self):
        lin0_decays = 0.0001 * (self.weights[0] ** 2).sum() / 2.0
        lin1_decays = 0.00025 * (self.weights[1] ** 2).sum() / 2.0
        lin2_decays = 0.00025 * (self.weights[2] ** 2).sum() / 2.0
        lin3_decays = 0.0005 * (self.weights[3] ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays
        
    def forward(self, state, act):
        
        state_proc = self.env_spec.state_preproc(state)
                        
        comb = torch.cat([state_proc ,act], 2)
    
        comb = (comb - self.mu)/self.sigma
              
        for weight,bias,act_fun in zip(self.weights, self.biases, self.act_funs):
            comb = torch.bmm(comb,weight) + bias
            comb = act_fun(comb)
            
        mean, log_var = comb[:, :, :self.output_size], comb[:, :, self.output_size:]
                        
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)
                
        return mean, log_var
        
    def sample_model_ts_inf(self, state, act):
        
        state = self._expand_to_ts_format(state)
        act = self._expand_to_ts_format(act)
        
        with torch.no_grad():
            mean, log_var = self.forward(state, act)
            
        dist = Normal(mean,log_var.exp().sqrt())
            
        next_state = dist.sample()
               
        return self._flatten_to_matrix(next_state)
    
    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]

        # Before, [10, 5] in case of proc_obs
        reshaped = mat.view(-1, self.ensemble_size, self.n_particles // self.ensemble_size, dim)
        # After, [2, 5, 1, 5]

        transposed = reshaped.transpose(0, 1)
        # After, [5, 2, 1, 5]

        reshaped = transposed.contiguous().view(self.ensemble_size, -1, dim)
        # After. [5, 2, 5]

        return reshaped

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.shape[-1]

        reshaped = ts_fmt_arr.view(self.ensemble_size, -1, self.n_particles // self.ensemble_size, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped
                        
    # def evaluate_model(self, state, action, next_state, test_set_ind, batch_size = 32):
    #     losses = torch.zeros((self.ensemble_size), device = self.device)
    #     with torch.no_grad():
    #         for batch_n in range(int(np.ceil(test_set_ind.shape[1] / batch_size))):
                
    #             batch_ind = test_set_ind[:, batch_n * batch_size:(batch_n + 1) * batch_size]
                                
    #             state_batch, action_batch, next_state_batch = state[batch_ind, :], action[batch_ind, :], next_state[batch_ind, :]
                
    #             state_batch = torch.FloatTensor(state_batch).to(self.device)
    #             action_batch = torch.FloatTensor(action_batch).to(self.device)
    #             next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                
    #             mean, log_var = self.forward(state_batch, action_batch)
    #             inv_var = torch.exp(-log_var)
                
    #             loss_log = ((mean - next_state_batch) ** 2) * inv_var + log_var
                
    #             loss_log = loss_log.mean(-1).mean(-1)
                
    #             losses += loss_log
                                
    #     losses = losses.cpu().numpy()
    
    #     return losses
    
    def update_parameters(self, n_epochs = 5, batch_size = 32, 
                          max_epochs_since_improv = 20):
        
        # Shape: 0: Obs Index 1: Dim
        
        state_all, action_all, next_state_all = [np.array(t) for 
                                      t in zip(*self.replay_buffer)]
        
        n = state_all.shape[0]
        
        # train_test_perm  = np.random.permutation(n)
        # n_holdout = int(np.ceil(self.holdout_ratio * n))
        
        # Shape: 0: Obs Index
        
        # train_set_ind, test_set_ind = train_test_perm[n_holdout:], train_test_perm[:n_holdout]
        
        n_train = n #3train_set_ind.shape[0]
        
        inputs = np.concatenate((self.env_spec.state_preproc(state_all), action_all), axis = 1)
        
        mu = np.mean(inputs, axis = 0)#, keepdims = True)
        sigma = np.std(inputs, axis = 0)#, keepdims = True)
        sigma[sigma < 1e-12] = 1.0
        
        self.mu.data = torch.FloatTensor(mu).to(self.device)
        self.sigma.data = torch.FloatTensor(sigma).to(self.device)
                
        # Shape: 0: Ensemble 1: Obs Index
        
        # bootstrap_inds = np.stack([np.random.choice(train_set_ind, size = n_train) 
        #                             for _ in range(self.ensemble_size)])
        
        bootstrap_inds = np.random.randint(n, size=[self.ensemble_size, n])
        
        # test_set_inds = np.broadcast_to(test_set_ind, (self.ensemble_size, test_set_ind.shape[0]))
                                                
        # best_holdout = self.evaluate_model(state_all, action_all, next_state_all, test_set_inds)
        # best_weights = list(map(lambda x: x.data, self.weights))
        # best_biases = list(map(lambda x: x.data, self.biases))
        # best_max_log_var = self.max_log_var.data
        # best_min_log_var = self.min_log_var.data
        # epochs_since_improv = 0
        for i_epoch in range(n_epochs):
            # if (epochs_since_improv > max_epochs_since_improv or i_epoch == n_epochs - 1):
            #     for weight, best_w, bias, best_b in zip(self.weights, best_weights ,self.biases, best_biases):
            #         weight.data.copy_(best_w)
            #         bias.data.copy_(best_b)
            #     self.max_log_var.data.copy_(best_max_log_var)
            #     self.min_log_var.data.copy_(best_min_log_var)
            #     #print('Terminated at epoch {}'.format(i_epoch))
            #     break
            
            for batch_n in range(int(np.ceil(n_train / batch_size))):
                
                batch_ind = bootstrap_inds[:, batch_n * batch_size:(batch_n + 1) * batch_size]
                                
                state_batch, action_batch, next_state_batch = state_all[batch_ind, :], action_all[batch_ind, :], next_state_all[batch_ind, :]
                                                    
                state_batch = torch.FloatTensor(state_batch).to(self.device)
                action_batch = torch.FloatTensor(action_batch).to(self.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
                                
                loss = 0.01 * torch.sum(self.max_log_var) - 0.01 * torch.sum(self.min_log_var)
                
                loss += self.get_decays()
                
                mean, log_var = self.forward(state_batch,action_batch)
                inv_var = torch.exp(-log_var)
                
                targets = self.env_spec.target_proc(self.state_scaler(state_batch),
                                                    self.state_scaler(next_state_batch))
                
                loss_log =  ((mean - targets) ** 2) * inv_var + log_var
                
                loss_log = loss_log.mean(-1).mean(-1).sum()
                
                loss += loss_log 
                        
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
            bootstrap_inds = shuffle_rows(bootstrap_inds)
                
            # current_holdout = self.evaluate_model(state_all, action_all, next_state_all, test_set_inds)
            # improv = (best_holdout - current_holdout)/best_holdout
            # updated = False
            # for i in range(self.ensemble_size):
            #     if improv[i] > 0.01:
            #         updated = True
            #         current_weights = list(map(lambda x: x.data, self.weights))
            #         current_biases = list(map(lambda x: x.data, self.biases))
            #         best_max_log_var = self.max_log_var.data
            #         best_min_log_var = self.min_log_var.data
            #         for k in range(len(current_weights)):
            #             best_weights[k][i,:,:] = current_weights[k][i,:,:]
            #             best_biases[k][i,:,:] = current_biases[k][i,:,:]
            #         best_holdout[i] = current_holdout[i]
            # if updated:
            #     epochs_since_improv = 0
            # else:
            #     epochs_since_improv += 1


class PETS():
    def __init__(self,
                 input_size, 
                 output_size,
                 action_size,
                 min_action,
                 max_action,
                 env_spec,
                 device,
                 hidden_size = 500,
                 n_hidden_layers = 3,
                 ensemble_size = 5,
                 lr = 0.001,
                 replay_buffer_size = 1000000,
                 holdout_ratio = 0,
                 horizon = 25,
                 population_size = 400,
                 elite_size = 40,
                 n_cem_iterations = 5,
                 n_particles = 20,
                 min_var_threshold = 0.001,
                 alpha = 0.1,
                 filter_coeff = 0.5):
        
        self.env_spec = env_spec
        
        self.prob_net = ProbNet(input_size, 
                 output_size,
                 device,
                 hidden_size,
                 n_hidden_layers,
                 ensemble_size,
                 lr,
                 replay_buffer_size,
                 holdout_ratio,
                 n_particles,
                 env_spec)
        
        self.ensemble_size = ensemble_size
                
        self.min_action = torch.repeat_interleave(torch.tensor(min_action).view(1, -1),
                                                  horizon, axis = 0).float().to(device)
        self.max_action = torch.repeat_interleave(torch.tensor(max_action).view(1, -1),
                                                  horizon, axis = 0).float().to(device)
        self.action_size = action_size

        self.horizon = horizon
        self.population_size = population_size
        self.elite_size = elite_size
        self.n_cem_iterations = n_cem_iterations
        self.n_particles = n_particles
        self.min_var_threshold = min_var_threshold
        self.alpha = alpha
        self.filter_coeff = filter_coeff
        
        self.device = device
        
        self.init_mean = (self.max_action + self.min_action) / 2
        self.init_mean_eval = (self.max_action + self.min_action) / 2
        self.init_var = (self.max_action - self.min_action) ** 2 / 16
                         
    def reset(self):
        self.init_mean = (self.max_action + self.min_action) / 2
        
    def reset_eval(self):
        self.init_mean_eval = (self.max_action + self.min_action) / 2
        
    def train_models(self, n_epochs_dyn, batch_size_dyn, n_epochs_reg,
                     batch_size_reg):
        self.prob_net.update_parameters(n_epochs_dyn, batch_size_dyn)
        
    def predict(self, state, action, use_mean = True):
        state = torch.from_numpy(state).float().to(self.device).expand(self.ensemble_size, 1, -1)
        action = torch.from_numpy(action).float().to(self.device).expand(self.ensemble_size, 1, -1)
        with torch.no_grad():
            mean, log_var = self.prob_net(state, action)
        if use_mean:
            mean_pred = mean.mean(0)
        else:
            dist = Normal(mean, log_var.exp().sqrt())
            next_state_diff = dist.sample()
            mean_pred = next_state_diff.mean(0)
        return self.env_spec.state_postproc(state[0, :, :], 
                                            self.prob_net.state_descaler(mean_pred)).squeeze(0).cpu().numpy()
        
    def save_transition(self, state, action, next_state, reward):
        self.prob_net.replay_buffer.append([state, action, next_state])
        
    def replay_buff_len(self):
        return len(self.prob_net.replay_buffer)
    
    def save_model(self, path = 'PETS_model.pt'):
        torch.save(self.prob_net.state_dict(), path)
        
    def load_model(self, path = 'PETS_model.pt'):
        
        model = torch.load(path, map_location = self.device)
        
        self.prob_net.load_state_dict(model)
        
    def cem_iteration(self, observation,  mean, var, return_rewards = False):
                
        # Each action sequence has n_particles trajectories
        
        x = observation.unsqueeze(0).expand((self.population_size * self.n_particles, -1))
                
        # Bound the variance
        
        lb_dist, ub_dist = mean - self.min_action, self.max_action - mean
    
        constr_var = torch.min(torch.min((lb_dist / 2) ** 2, (ub_dist / 2) ** 2), var)
                
        ### Time correlated noise
        
        iid_sample = torch.fmod(torch.randn((self.population_size, self.horizon, self.action_size), device = self.device), 2)
        
        actions = torch.zeros((self.population_size, self.horizon, self.action_size), device = self.device)
        
        actions[:, 0, :] = iid_sample[:, 0, :]
        
        for t in range(1, self.horizon):
            actions[:, t, :] = self.filter_coeff * actions[:, t - 1, :] + \
                np.sqrt(1 - self.filter_coeff ** 2) * iid_sample[:, t, :]        
        ###
        
        actions = actions * constr_var.sqrt() + mean
                        
        # Repeat each action sequence n_particles times
                                        
        actions_repeat = torch.repeat_interleave(actions, self.n_particles, dim = 0)
                
        trajectory_returns = torch.zeros((self.population_size * self.n_particles), dtype=torch.float, device = self.device)
        
        # self.env_spec.reset(self.population_size * self.n_particles, self.device)
        
        state = x
        for t in range(self.horizon):
            act_repeat = actions_repeat[:, t, :]
            next_state = self.env_spec.state_postproc(state,
                                                      self.prob_net.state_descaler(
                                                          self.prob_net.sample_model_ts_inf(state, act_repeat)))
            trajectory_returns += self.env_spec.reward_fun(act_repeat, next_state)
            state = next_state
                    
        # Check that the filling order is correct
        
        returns = trajectory_returns.reshape(-1, self.n_particles)
        
        returns = torch.where(torch.isnan(returns), -1e6 * torch.ones_like(returns), returns)
        
        returns = returns.mean(dim = 1)
                                
        best_action_seqs = torch.argsort(returns, descending=True)[:self.elite_size]
        
        best_actions = actions[best_action_seqs,:,:]
        
        new_mean, new_var = best_actions.mean(dim = 0), best_actions.var(dim = 0)
        
        if return_rewards:
            return self.alpha * mean + (1 - self.alpha) * new_mean, self.alpha * var + (1 - self.alpha) * new_var, \
                returns[best_action_seqs]
        else:
            return self.alpha * mean + (1 - self.alpha) * new_mean, self.alpha * var + (1 - self.alpha) * new_var
    
    def cem_test(self, observation, init_action = None, init_var = None):
        observation = torch.from_numpy(observation).float().to(self.device)
        
        mean, var, returns = self.cem_iteration(observation, self.init_mean if init_action is None else init_action,
                                  self.init_var if init_var is None else init_var,
                                  True)
        return mean, var, returns
    
    def act(self, observation, evaluation = False):
        
        # start = timer()
        
        observation = torch.from_numpy(observation).float().to(self.device)
                                   
        mean = self.init_mean_eval if evaluation else self.init_mean
        var = self.init_var
                        
        n_iter = 0
        while n_iter < self.n_cem_iterations and torch.max(var).item() > self.min_var_threshold:
            mean, var = self.cem_iteration(observation, mean, var)
            n_iter += 1
                        
        new_init_mean = torch.cat([mean[1:, :], torch.zeros((1, self.action_size), device = self.device)], dim = 0)
        
        if evaluation:
            self.init_mean_eval = new_init_mean
        else:
            self.init_mean = new_init_mean
            
        # end = timer()
        # print('Selection time {}'.format(end - start))
                
        return mean[0,:].cpu().numpy()
