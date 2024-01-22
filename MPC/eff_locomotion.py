import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

import numpy as np
from collections import deque

from timeit import default_timer as timer

import torch.multiprocessing as mp

from DAE import DAE_NET
from DEEN import DEEN_NET
from RND import RND_NET

# Dynamics model

# Deterministic model

# Predicts the difference between next and current state

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ForwardNet(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size,
                 device,
                 hidden_size,
                 lr,
                 replay_buffer_size,
                 env_spec):
        super().__init__()
                                
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, 2*output_size))
                        
        for m in self.modules():
            if type(m) is nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Input scaler parameters
        self.mu = nn.Parameter(torch.zeros((1, input_size)), requires_grad = False)
        self.sigma = nn.Parameter(torch.ones((1, input_size)), requires_grad = False)
        
        self.mu_target = nn.Parameter(torch.zeros((1, output_size)), requires_grad = False)
        self.sigma_target = nn.Parameter(torch.ones((1, output_size)), requires_grad = False)
        
        self.max_log_var = nn.Parameter(torch.full((1, output_size), 0.5,
                                      device = device))
        self.min_log_var = nn.Parameter(torch.full((1, output_size), -10.0,
                                    device = device))
                
        self.input_size = input_size
        self.output_size = output_size
        
        self.device = device
        self.optim = optim.Adam(list(self.net.parameters()) + [self.max_log_var] + [self.min_log_var], lr = lr)
        self.replay_buffer = deque(maxlen = replay_buffer_size)
        
        self.env_spec = env_spec
        
        self.to(device)
        
    def state_scaler(self, state):
        return (state - self.mu[:self.output_size])/self.sigma[:self.output_size]
    
    def state_descaler(self, state, shift = True):
        scaled = state * (self.sigma_target)
        if shift:
            scaled += self.mu_target
        return scaled
                
    def forward(self, state, act, return_var = False):
        
        state_proc = self.env_spec.state_preproc(state)
                
        comb = torch.cat([state_proc,act], -1)
    
        comb = (comb - self.mu)/(self.sigma)
        
        comb = self.net(comb)
        
        if len(comb.shape) == 3:
            mean, log_var = comb[:, :, :self.output_size], comb[:, :, self.output_size:]
        else:
            mean, log_var = comb[:, :self.output_size], comb[:, self.output_size:]
        
        log_var = self.max_log_var - F.softplus(self.max_log_var - log_var)
        log_var = self.min_log_var + F.softplus(log_var - self.min_log_var)
        
        if return_var:
            return mean, log_var
        else:
            return mean
                                            
    def update_parameters(self, n_epochs = 5, batch_size = 32):
        
        # Shape: 0: Obs 1: Dim
                
        states, actions, next_states = [np.array(t) for 
                                      t in zip(*self.replay_buffer)]
                                        
        n_train = states.shape[0]
        
        inputs = np.concatenate((self.env_spec.state_preproc(states), actions), axis = 1)
        
        targets = self.env_spec.target_proc(states, next_states)
        
        mu_target = np.mean(targets, axis = 0, keepdims = True)
        sigma_target = np.std(targets, axis = 0, keepdims = True)
        # sigma_target[sigma_target < 1e-12] = 1.0
        
        self.mu_target.data = torch.FloatTensor(mu_target).to(self.device)
        self.sigma_target.data = torch.FloatTensor(sigma_target).to(self.device)
        
        mu = np.mean(inputs, axis = 0, keepdims = True)
        sigma = np.std(inputs, axis = 0, keepdims = True)
        # sigma[sigma < 1e-12] = 1.0
        
        self.mu.data = torch.FloatTensor(mu).to(self.device)
        self.sigma.data = torch.FloatTensor(sigma).to(self.device)
        
        train_set_ind = np.random.permutation(n_train)
                                                                                                
        for i_epoch in range(n_epochs):
            
            for batch_n in range(int(np.ceil(n_train / batch_size))):
                
                batch_ind = train_set_ind[batch_n * batch_size:(batch_n + 1) * batch_size]
                                                
                states_batch = torch.FloatTensor(states[batch_ind, :]).to(self.device)
                actions_batch = torch.FloatTensor(actions[batch_ind, :]).to(self.device)
                next_states_batch = torch.FloatTensor(next_states[batch_ind, :]).to(self.device)
                                                        
                targets_batch = self.env_spec.target_proc(states_batch,
                                                    next_states_batch)
                
                targets_batch = (targets_batch - self.mu_target)/(self.sigma_target)
                                                
                loss = 0.01 * torch.sum(self.max_log_var) - 0.01 * torch.sum(self.min_log_var)
                
                mean, log_var = self.forward(states_batch, actions_batch, return_var = True)
                inv_var = torch.exp(-log_var)
                
                loss_log = ((mean - targets_batch) ** 2) * inv_var + log_var
                
                loss_log = loss_log.mean()
                
                loss += loss_log 
                                             
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
            train_set_ind = np.random.permutation(train_set_ind)
                
class DETNET():
    def __init__(self,
                 input_size, 
                 output_size,
                 action_size,
                 min_action,
                 max_action,
                 env_spec,
                 device,
                 hidden_size_dyn = 256,
                 hidden_size_reg = 256,
                 lr = 0.001,
                 replay_buffer_size = 1000000,
                 horizon = 25,
                 population_size = 400,
                 elite_size = 40,
                 n_cem_iterations = 5,
                 min_var_threshold = 0.001,
                 alpha = 0.1,
                 filter_coeff = 0.5,
                 reg = 'None',
                 reg_alpha = 0.045,
                 reg_noise_std = 0.3):
        
        self.env_spec = env_spec
                
        self.prob_net = ForwardNet(input_size, 
                 output_size,
                 device,
                 hidden_size_dyn,
                 lr,
                 replay_buffer_size,
                 env_spec)
        
        self.min_action = torch.repeat_interleave(torch.tensor(min_action).view(1, -1),
                                                  horizon, axis = 0).float().to(device)
        self.max_action = torch.repeat_interleave(torch.tensor(max_action).view(1, -1),
                                                  horizon, axis = 0).float().to(device)
        self.action_size = action_size

        self.horizon = horizon
        self.population_size = population_size
        self.elite_size = elite_size
        self.n_cem_iterations = n_cem_iterations
        self.min_var_threshold = min_var_threshold
        self.alpha = alpha
        
        self.device = device
        
        self.init_mean = (self.max_action + self.min_action) / 2
        self.init_mean_eval = (self.max_action + self.min_action) / 2
        self.init_var = (self.max_action - self.min_action) ** 2 / 16
        
        self.filter_coeff = filter_coeff
        
        self.reg = reg
        self.reg_alpha = reg_alpha
        
        if reg == 'DAE':
            self.regularizer = DAE_NET((input_size - action_size) * 2 + action_size,
                                       device, hidden_size_reg, lr, env_spec, reg_noise_std)
        elif reg == 'DEEN':
            self.regularizer = DEEN_NET((input_size - action_size) * 2 + action_size, device, hidden_size_reg, lr, env_spec, reg_noise_std)
        # elif reg == 'RND':
        #     self.regularizer = RND_NET(input_size, output_size, device, hidden_size_reg, lr, env_spec)
        elif reg == 'None':
            pass
        else:
            raise Exception('Unknown regularization method')
        
        ### These are for decoupled planning and execution
        
        # self.current_state = torch.zeros((input_size - action_size), device = device)

        # self.episode_ended = torch.tensor([False], dtype = torch.bool, device = device)
        # self.wait_for_episode = torch.tensor([True], dtype = torch.bool, device = device)
        # self.wait_for_agent = torch.tensor([True], dtype = torch.bool, device = device)
        
        # self.current_plan = self.init_mean.clone()
        # # Where to read the action from current plan
        # self.current_plan_step = torch.tensor([0], dtype = torch.uint8, device = device)
        # self.plan_not_used = torch.tensor([True], dtype = torch.bool, device = device)
        
        # if 'RealAntMujoco-v0' in env_name:
        #     self.env = gym.make(
        #         'RealAntMujoco-v0',
        #         task=task,
        #         latency=latency,
        #         xyz_noise_std=xyz_noise_std,
        #         rpy_noise_std=rpy_noise_std,
        #         min_obs_stack=min_obs_stack,
        #         )
        # else:
        #         self.env = gym.make(env_name)
                
        # self.env.seed(seed)
        # self.env.action_space.seed(seed)
        
        ###
        
    def train_models(self, n_epochs_dyn, batch_size_dyn, n_epochs_reg,
                     batch_size_reg):
        self.prob_net.update_parameters(n_epochs_dyn, batch_size_dyn)
        if self.reg != 'None':
            self.regularizer.update_parameters(self.prob_net.replay_buffer,
                                            n_epochs_reg, batch_size_reg)
        
    def predict(self, state, action):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        action = torch.from_numpy(action).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.env_spec.state_postproc(state, 
                                                self.prob_net.state_descaler(self.prob_net(state, action))
                                                ).squeeze(0).cpu().numpy()
        
    def save_transition(self, state, action, next_state, reward):
        self.prob_net.replay_buffer.append([state, action, next_state])
                        
    def save_model(self, path = 'DETNET_model.pt'):
        torch.save(self.prob_net.state_dict(), path)
        
    def load_model(self, path = 'DETNET_model.pt'):
        
        model = torch.load(path, map_location = self.device)
        
        self.prob_net.load_state_dict(model)
        
    # def update_plan(self, get_action, new_plan = None, new_current_state = None,
    #                 random_action = False):
    #     # Use lock to prevent data races between planning and execution processes
    #     with mp.Lock():
    #         if get_action:
    #             action = self.env.action_space.sample() if random_action else \
    #                 self.current_plan[self.current_plan_step.item() ,:].cpu().numpy()
    #             next_state, reward, done, _ = self.env.step(action)
    #             self.current_state = torch.from_numpy(next_state).float().to(self.device)
    #             self.current_plan_step += 1
    #             return action, next_state, reward, done
    #             # self.plan_not_used = torch.tensor([False], dtype = torch.bool, device = self.device)
    #         elif new_plan is not None:
    #             self.current_plan = new_plan
    #             self.current_plan_step = torch.tensor([0], dtype = torch.uint8, device = self.device)
    #         #     self.plan_not_used = torch.tensor([True], dtype = torch.bool, device = self.device)
    #         # elif new_current_state is not None:
    #         #     self.current_state = torch.from_numpy(new_current_state).float().to(self.device)
    #         #     if self.plan_not_used.item():
    #         #         pass
    #         #     else:
    #         #         self.current_plan_step += 1
    #         else:
    #             return self.current_state.clone().unsqueeze(0), self.current_plan_step.clone().item()
                    
    # def perform_step(self, random_action):
    #     return self.update_plan(True, random_action = random_action)
    
    # def get_state(self):
    #     return self.current_state.cpu().numpy()
        
    # def reset(self, initial_state):
    def reset(self):
        self.init_mean = (self.max_action + self.min_action) / 2
        
        # self.current_plan = self.init_mean.clone()
        # self.current_plan_step = torch.tensor([0], dtype = torch.uint8, device = self.device)
                
        # self.episode_ended = torch.tensor([False], dtype = torch.bool, device = self.device)
        # self.wait_for_episode = torch.tensor([True], dtype = torch.bool, device = self.device)
        # self.wait_for_agent = torch.tensor([True], dtype = torch.bool, device = self.device)
        
        # self.current_state = torch.from_numpy(self.env.reset()).float().to(self.device)
                
    def reset_eval(self):
        self.init_mean_eval = (self.max_action + self.min_action) / 2
        
    # def plan(self, observation, n_step_predict, plan_update_time, time_start = None):
    #     # start = timer()                                
    #     mean = self.init_mean
    #     var = self.init_var
                        
    #     n_iter = 0
    #     while n_iter < self.n_cem_iterations and torch.max(var).item() > self.min_var_threshold:
    #         mean, var = self.cem_iteration(observation, mean, var)
    #         n_iter += 1
        
    #     # Wait until enough time has passed
    #     while time_start is not None and timer() - time_start < plan_update_time:
    #         pass
                
    #     self.update_plan(False, new_plan = mean)
    #     # print('Plan Updated')
                
    #     # Improve mean initialization
    #     self.init_mean = torch.cat([mean[n_step_predict:, :], 
    #                                 torch.zeros((n_step_predict, self.action_size), device = self.device)], dim = 0)
                
    #     # end = timer()
    #     # print('Planning time {}'.format(end - start))
        
    # def plan_timer(self, n_step_predict = 4, plan_update_time = 0.025, replan_time = 0.050):
        
    #     # Plan at the start of an episode
    #     self.plan(self.current_state, n_step_predict, plan_update_time)
        
    #     # Tell the execution process that episode can begin
    #     self.wait_for_agent = torch.tensor([False], dtype = torch.bool, device = self.device)
        
    #     # Wait until episode starts proper
    #     while self.wait_for_episode.item():
    #         pass
                    
    #     while True:
    #         # Stop planning when episode ends
    #         if self.episode_ended.item():
    #             break
    #         # Predict the future state here
    #         start = timer()
    #         current_state, current_step = self.update_plan(False)
    #         with torch.no_grad():
    #             for _ in range(n_step_predict):
    #                 current_state = self.env_spec.state_postproc(current_state, 
    #                                                              self.prob_net(current_state, 
    #                                                                            self.current_plan[current_step, :].unsqueeze(0)))
    #                 current_step += 1
    #         # print('Planning')
    #         self.plan(current_state.squeeze(0), n_step_predict, plan_update_time, start)
    #         while timer() - start < replan_time:
    #             pass
    #         # end = timer()
    #         # print(end - start)
                
    # def start_planning(self):
    #     self.wait_for_episode = torch.tensor([False], dtype = torch.bool, device = self.device)
        
    # def stop_planning(self):
    #     self.episode_ended = torch.tensor([True], dtype = torch.bool, device = self.device)
        
    # def plan_not_ready(self):
    #     return self.wait_for_agent.item()
    
    # def act_paral(self):
    #     action = self.update_plan(True)
            
    #     return action
    
    def cem_iteration(self, observation, mean, var, return_rewards = False):
                        
        x = observation.unsqueeze(0).expand((self.population_size, -1))
                
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
                                        
        returns = torch.zeros((self.population_size), dtype=torch.float, device = self.device)
        
        states = []
                
        with torch.no_grad():
            state = x
            states.append(self.env_spec.state_preproc(state))
            for t in range(self.horizon):
                action = actions[:, t, :]
                next_state = self.env_spec.state_postproc(state, 
                                                          self.prob_net.state_descaler(self.prob_net(state, action)))
                returns += self.env_spec.reward_fun(action, next_state, self.device)
                state = next_state
                states.append(self.env_spec.state_preproc(state))
                                
        states = torch.stack(states, dim = 1)
        
        penalty = self.regularizer.penalty(states, actions) if self.reg != 'None' else 0
        
        # print([returns.mean(),penalty.mean() * self.reg_alpha])
                    
        returns = torch.where(torch.isnan(returns), -1e6 * torch.ones_like(returns), returns) - self.reg_alpha * penalty  
        
        best_action_seqs = torch.argsort(returns, descending = True)[:self.elite_size]
        
        best_actions = actions[best_action_seqs, :, :]
        
        new_mean, new_var = best_actions.mean(dim = 0), best_actions.var(dim = 0)
        
        if return_rewards:
            return self.alpha * mean + (1 - self.alpha) * new_mean, self.alpha * var + (1 - self.alpha) * new_var,\
                returns[best_action_seqs]
        else:
            return self.alpha * mean + (1 - self.alpha) * new_mean, self.alpha * var + (1 - self.alpha) * new_var
    
    def cem_test(self, observation, init_action = None, init_var = None):
        observation = torch.from_numpy(observation).float().to(self.device)
        
        if init_action is not None:
            init_action = torch.from_numpy(init_action).float().to(self.device)
        
        mean, var = self.cem_iteration(observation, self.init_mean if init_action is None else init_action,
                                  self.init_var if init_var is None else init_var,
                                  False)
        return mean.cpu().numpy(), var.cpu().numpy()
    
    def act(self, observation, evaluation = False, noise_std = None):
        
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
        
        current_action = mean[0, :]
        if noise_std is not None:
            current_action += torch.randn_like(current_action) * noise_std
            
        current_action = torch.max(torch.min(current_action, self.max_action[0, :]), self.min_action[0, :])
                
        return current_action.cpu().numpy()