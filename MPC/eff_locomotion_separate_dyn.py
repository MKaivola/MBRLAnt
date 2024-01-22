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
        
        self.n_leg_dims = 16
        self.n_pose_dims = 13
        
        self.n_leg_input = 6
        self.n_leg_output = 4
        self.n_pose_input = self.n_leg_dims + self.n_pose_dims
        self.n_pose_output = self.n_pose_dims
        
        
        self.net_leg = nn.Sequential(nn.Linear(self.n_leg_input, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, 2*self.n_leg_output))
        
        self.net_pose = nn.Sequential(nn.Linear(self.n_pose_input, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, hidden_size),
                          Swish(),
                          nn.Linear(hidden_size, 2*self.n_pose_output))
                        
        for m in self.modules():
            if type(m) is nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Input scaler parameters
        
        self.mu_leg = nn.Parameter(torch.zeros((1, self.n_leg_input), device = device), requires_grad = False)
        self.sigma_leg = nn.Parameter(torch.ones((1, self.n_leg_input), device = device), requires_grad = False)
        
        self.mu_pose = nn.Parameter(torch.zeros((1, self.n_pose_input), device = device), requires_grad = False)
        self.sigma_pose = nn.Parameter(torch.ones((1, self.n_pose_input), device = device), requires_grad = False)
        
        self.mu_target_leg = nn.Parameter(torch.zeros((1, self.n_leg_output), device = device), requires_grad = False)
        self.sigma_target_leg = nn.Parameter(torch.ones((1, self.n_leg_output), device = device), requires_grad = False)
        
        self.mu_target_pose = nn.Parameter(torch.zeros((1, self.n_pose_output), device = device), requires_grad = False)
        self.sigma_target_pose = nn.Parameter(torch.ones((1, self.n_pose_output), device = device), requires_grad = False)
        
        self.max_log_var_leg = nn.Parameter(torch.full((1, self.n_leg_output), 0.5,
                                      device = device))
        self.min_log_var_leg = nn.Parameter(torch.full((1, self.n_leg_output), -10.0,
                                    device = device))
        
        self.max_log_var_pose = nn.Parameter(torch.full((1, self.n_pose_output), 0.5,
                                      device = device))
        self.min_log_var_pose = nn.Parameter(torch.full((1, self.n_pose_output), -10.0,
                                    device = device))
                
        self.input_size = input_size
        self.output_size = output_size
        
        self.device = device
        self.optim_leg = optim.Adam(list(self.net_leg.parameters()) + [self.max_log_var_leg] + [self.min_log_var_leg], lr = lr)
        self.optim_pose = optim.Adam(list(self.net_pose.parameters()) + [self.max_log_var_pose] + [self.min_log_var_pose], lr = lr)
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
    
    def get_max_min_log_var(self, predict_leg):
        if predict_leg:
            return self.max_log_var_leg, self.min_log_var_leg
        else:
            return self.max_log_var_pose, self.min_log_var_pose
    
    def leg_net_processor(self, states, actions = None):
        # Formats the leg network inputs
        
        leg_states = states[:, -self.n_leg_dims:]
        # leg_angles = states[:, :, 8:16]
        # leg_angle_vels = states[:, :, -8:]
        # leg_states = torch.cat([leg_angles, leg_angle_vels], dim = -1)
        inputs = []
        if actions is None:
            for i in np.arange(0, 8, 2):
                inputs.append(torch.stack([leg_states[:, i], leg_states[:, 8 + i], 
                                                 leg_states[:, i + 1], leg_states[:, 8 + i + 1]], dim = -1))
        else:
            for i in np.arange(0, 8, 2):
                inputs.append(torch.stack([leg_states[:, i], leg_states[:, 8 + i], 
                                                 leg_states[:, i + 1], leg_states[:, 8 + i + 1],
                                                 actions[:, i], actions[:, i + 1]],  dim = -1))
        inputs = torch.cat(inputs, dim = 0)
        return inputs
            
    def leg_predict(self, legs_diff, legs_curr = None):
        # Formats the leg network prediction to original state representation
        
        next_legs = legs_diff
        if legs_curr is not None:
            next_legs = legs_curr + legs_diff
        n_states = legs_diff.shape[0] // 4
        legs_formatted = torch.zeros((n_states, self.input_size - 8)).to(self.device)
        for state_i in np.arange(n_states):
            for i in range(4):
                # Hip angle
                legs_formatted[state_i, (self.n_pose_dims + 2*i)] = next_legs[(n_states) * i + state_i, 0]
                # legs_formatted[:, state_i, (8 + 2*i)] = next_legs[:, (n_states) * i + state_i, 0]
                # Knee angle
                legs_formatted[state_i, (self.n_pose_dims + 1 + 2*i)] = next_legs[(n_states) * i + state_i, 2]
                # legs_formatted[:, state_i, (8 + 1 + 2*i)] = next_legs[:, (n_states) * i + state_i, 2]
                # Hip velocity
                legs_formatted[state_i, (self.n_pose_dims + 8 + 2*i)] = next_legs[(n_states) * i + state_i, 1]
                # legs_formatted[:, state_i, (29 - 7 + 2*i)] = next_legs[:, (n_states) * i + state_i, 1]
                # Knee velocity
                legs_formatted[state_i, (self.n_pose_dims + 8 + 1 + 2*i)] = next_legs[(n_states) * i + state_i, 3]
                # legs_formatted[:, state_i, (29 - 7 + 1 + 2*i)] = next_legs[:, (n_states) * i + state_i, 3]

        # print(f"This {next_legs[:, 1, 3]}")
        # print(f"Should agree with this {legs_formatted[:, 0, -5]} \n")
        return legs_formatted
    
    def forward(self, inputs, predict_leg):
            
        if predict_leg:
                    
            leg_net_input = (inputs - self.mu_leg)/(self.sigma_leg)
            
            leg_net_input = self.net_leg(leg_net_input)
                
            mean, log_var = leg_net_input[:, :self.n_leg_output], leg_net_input[:, self.n_leg_output:]
        
            log_var = self.max_log_var_leg - F.softplus(self.max_log_var_leg - log_var)
            log_var = self.min_log_var_leg + F.softplus(log_var - self.min_log_var_leg)
                
            return mean, log_var
        
        else:
            
            pose_net_input = (inputs - self.mu_pose)/(self.sigma_pose)
            
            pose_net_input = self.net_pose(pose_net_input)
                
            mean, log_var = pose_net_input[:, :self.n_pose_output], pose_net_input[:, self.n_pose_output:]
        
            log_var = self.max_log_var_pose - F.softplus(self.max_log_var_pose - log_var)
            log_var = self.min_log_var_pose + F.softplus(log_var - self.min_log_var_pose)
                
            return mean, log_var
        
    def fit_model(self, inputs, targets, n_epochs, batch_size, predict_leg, optim):
        n = inputs.shape[0]
        
        train_set_ind = np.random.permutation(n)
                
        for i_epoch in range(n_epochs):
            
            for batch_n in range(int(np.ceil(n / batch_size))):
                
                batch_ind = train_set_ind[batch_n * batch_size:(batch_n + 1) * batch_size]
                                
                input_batch, target_batch = inputs[batch_ind, :], targets[batch_ind, :]
                                                    
                input_batch = torch.FloatTensor(input_batch).to(self.device)
                
                target_batch = torch.FloatTensor(target_batch).to(self.device)
                                
                mean, log_var = self.forward(input_batch, predict_leg)
                
                max_log_var, min_log_var = self.get_max_min_log_var(predict_leg)
                
                loss = 0.01 * torch.sum(max_log_var) - 0.01 * torch.sum(min_log_var)
                
                inv_var = torch.exp(-log_var)
                
                loss_log = ((mean - target_batch) ** 2) * inv_var + log_var
                
                loss_log = loss_log.mean()
                
                loss += loss_log 
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
            train_set_ind = np.random.permutation(train_set_ind)
                                            
    def update_parameters(self, n_epochs = 10, batch_size = 200):
        
        # Shape: 0: Obs Index 1: Dim
        
        state_all, action_all, next_state_all = [np.array(t) for 
                                      t in zip(*self.replay_buffer)]
                
        leg_states = state_all[:, -self.n_leg_dims:]
        leg_next_states = next_state_all[:, -self.n_leg_dims:]
        
        # leg_angles = state_all[:, 8:16]
        # leg_angle_vels = state_all[:, -8:]
        
        # leg_states = np.concatenate([leg_angles, leg_angle_vels], axis = -1)
        
        # next_leg_angles = next_state_all[:, 8:16]
        # next_leg_angle_vels = next_state_all[:, -8:]
        
        # leg_next_states = np.concatenate([next_leg_angles, next_leg_angle_vels], axis = -1)
        
        leg_net_input = []
        leg_net_output = []
        
        for i in np.arange(0, 8, 2):
                leg_net_input.append(np.stack([leg_states[:, i], leg_states[:, 8 + i], 
                                                 leg_states[:, i + 1], leg_states[:, 8 + i + 1],
                                                 action_all[:, i], action_all[:, i + 1]], axis = 1))
                leg_net_output.append(np.stack([leg_next_states[:, i], leg_next_states[:, 8 + i], 
                                                 leg_next_states[:, i + 1], leg_next_states[:, 8 + i + 1]], axis = 1))
        
        leg_net_input = np.concatenate(leg_net_input, axis = 0)
        leg_net_output = np.concatenate(leg_net_output, axis = 0)
        
        mu_leg = np.mean(leg_net_input, axis = 0, keepdims = True)
        sigma_leg = np.std(leg_net_input, axis = 0, keepdims = True)
        
        self.mu_leg.data = torch.FloatTensor(mu_leg).to(self.device)
        self.sigma_leg.data = torch.FloatTensor(sigma_leg).to(self.device)
        
        targets_leg = self.env_spec.target_proc(leg_net_input[:, :-2], leg_net_output)
        
        mu_targets_leg = np.mean(targets_leg, axis = 0, keepdims = True)
        sigma_targets_leg = np.std(targets_leg, axis = 0, keepdims = True)
        
        targets_leg = (targets_leg - mu_targets_leg)/(sigma_targets_leg)
                
        self.mu_target_leg.data = torch.FloatTensor(mu_targets_leg).to(self.device)
        self.sigma_target_leg.data = torch.FloatTensor(sigma_targets_leg).to(self.device)
        
        pose_net_input = np.concatenate([state_all[:, :self.n_pose_dims], leg_next_states - leg_states], axis = -1)
        # pose_net_input = leg_states # leg_next_states - leg_states
        
        # pose_net_input = np.concatenate([state_all[:, :8], state_all[:, 16:22],  leg_next_states - leg_states]
        #                                                     , axis = -1)
        
        mu_pose = np.mean(pose_net_input, axis = 0, keepdims = True)
        sigma_pose = np.std(pose_net_input, axis = 0, keepdims = True)
        
        self.mu_pose.data = torch.FloatTensor(mu_pose).to(self.device)
        self.sigma_pose.data = torch.FloatTensor(sigma_pose).to(self.device)
        
        targets_pose = self.env_spec.target_proc(state_all[:, :self.n_pose_dims], next_state_all[:, :self.n_pose_dims]) # state_all[:, :self.n_pose_dims]
        
        # targets_pose = self.env_spec.target_proc(np.concatenate([state_all[:, :8], state_all[:, 16:22]], axis = -1), 
        #                                           np.concatenate([next_state_all[:, :8], next_state_all[:, 16:22]], axis = -1))
        
        mu_targets_pose = np.mean(targets_pose, axis = 0, keepdims= True)
        sigma_targets_pose = np.std(targets_pose, axis = 0, keepdims = True)
        
        targets_pose = (targets_pose - mu_targets_pose)/(sigma_targets_pose)
                
        self.mu_target_pose.data = torch.FloatTensor(mu_targets_pose).to(self.device)
        self.sigma_target_pose.data = torch.FloatTensor(sigma_targets_pose).to(self.device)
        
        self.fit_model(leg_net_input, targets_leg, n_epochs, batch_size, True, self.optim_leg)
        self.fit_model(pose_net_input, targets_pose, n_epochs, batch_size, False, self.optim_pose)
                
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
        
    def predict(self, state, act):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        act = torch.from_numpy(act).float().to(self.device).unsqueeze(0)
        inputs_leg = self.prob_net.leg_net_processor(state, act)
        with torch.no_grad():
            legs_diff, _ = self.prob_net.forward(inputs_leg, True)
            legs_diff = legs_diff * (self.prob_net.sigma_target_leg) + self.prob_net.mu_target_leg
            next_state = self.prob_net.leg_predict(legs_diff, self.prob_net.leg_net_processor(state))
            
            legs_diff_formatted = self.prob_net.leg_predict(legs_diff)[:, -self.prob_net.n_leg_dims:]
            # legs_diff_formatted = self.leg_predict(legs_diff)
            # legs_diff_formatted = torch.cat([legs_diff_formatted[:, :, 8:16], legs_diff_formatted[:, :, -8:]], dim = -1)
            pose_formatted = state[:, :self.prob_net.n_pose_dims]
            # pose_formatted = torch.cat([state[:, :, :8], state[:, :, 16:22]], dim = -1)
            inputs_pose = torch.cat([pose_formatted, legs_diff_formatted], dim = -1)
            # inputs_pose = self.leg_predict(legs_diff)[:, :, -self.n_leg_dims:]
            # inputs_pose = next_state[:, :, -self.n_leg_dims:]
            pose_predict, _ = self.prob_net.forward(inputs_pose, False)    
            pose_predict = pose_predict * (self.prob_net.sigma_target_pose) + self.prob_net.mu_target_pose
        
        next_state[:, :self.prob_net.n_pose_dims] = pose_predict + state[:, :self.prob_net.n_pose_dims]
        # next_state[:, :, :8] = pose_predict[:, :, :8] + state[:, :, :8]
        # next_state[:, :, 16:22] = pose_predict[:, :, 8:] + state[:, :, 16:22]
        
        mean_pred = next_state
        
        return mean_pred.squeeze(0).cpu().numpy()
        
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
                returns += self.env_spec.reward_fun(action, next_state)
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
            # mean, var = self.cem_iteration(observation, mean, var)
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