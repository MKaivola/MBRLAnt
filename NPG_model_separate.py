import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import scipy.stats as stats

from collections import deque

import matplotlib.pyplot as plt

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
                 replay_buffer_size = 100000):
        super().__init__()
        
        input_size = state_size + action_size
        
        self.n_leg_dims = 16
        self.n_pose_dims = 13 # 14
        
        self.n_leg_input = 6
        self.n_leg_output = 4
        self.n_pose_input = self.n_leg_dims + self.n_pose_dims
        self.n_pose_output = self.n_pose_dims
        
        # self.weights = [create_weight((ensemble_size, input_size, hidden_size), device),
        #                 create_weight((ensemble_size, hidden_size, hidden_size), device),
        #                 create_weight((ensemble_size, hidden_size, output_size), device)]
        
        self.weights_leg = [trunc_initializer((ensemble_size, self.n_leg_input, hidden_size), 1.0 / (2.0 * np.sqrt(input_size))).to(device),
                trunc_initializer((ensemble_size, hidden_size, hidden_size), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device),
                # trunc_initializer((ensemble_size, hidden_size, hidden_size), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device),
                trunc_initializer((ensemble_size, hidden_size, 2 * self.n_leg_output), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device)]
        for w in self.weights_leg:
            w.requires_grad = True
            
        self.weights_pose = [trunc_initializer((ensemble_size, self.n_pose_input, hidden_size), 1.0 / (2.0 * np.sqrt(input_size))).to(device),
            trunc_initializer((ensemble_size, hidden_size, hidden_size), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device),
            # trunc_initializer((ensemble_size, hidden_size, hidden_size), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device),
            trunc_initializer((ensemble_size, hidden_size, 2 * self.n_pose_output), 1.0 / (2.0 * np.sqrt(hidden_size))).to(device)]
        for w in self.weights_pose:
            w.requires_grad = True
          
        # self.biases = [create_weight((ensemble_size, 1, hidden_size), device, bias = True),
        #                 create_weight((ensemble_size, 1, hidden_size), device, bias = True),
        #                 create_weight((ensemble_size, 1, output_size), device, bias = True)]
        
        self.biases_leg = [torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
            torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
            # torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
            torch.zeros(ensemble_size, 1, 2 * self.n_leg_output, requires_grad = True, device = device)]
        
        self.biases_pose = [torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
            torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
            # torch.zeros(ensemble_size, 1, hidden_size, requires_grad = True, device = device),
            torch.zeros(ensemble_size, 1, 2 * self.n_pose_output, requires_grad = True, device = device)]
        
        self.act_funs = [nn.ReLU(), nn.ReLU(), nn.Sequential()]
        
        # self.act_funs = [Swish, Swish, nn.Sequential()]
        
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
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        self.device = device
        self.optim_leg = optim.Adam(list(self.weights_leg) + list(self.biases_leg), lr = lr)
        self.optim_pose = optim.Adam(list(self.weights_pose) + list(self.biases_pose), lr = lr)
        self.replay_buffer = deque(maxlen = replay_buffer_size)
        self.ensemble_size = ensemble_size
                
        self.env_spec = env_spec
        
    def get_max_min_log_var(self, predict_leg):
        if predict_leg:
            return self.max_log_var_leg, self.min_log_var_leg
        else:
            return self.max_log_var_pose, self.min_log_var_pose
        
    def leg_net_processor(self, states, actions = None):
        # Formats the leg network inputs
        
        leg_states = states[:, :, -self.n_leg_dims:]
        # leg_angles = states[:, :, 8:16]
        # leg_angle_vels = states[:, :, -8:]
        # leg_states = torch.cat([leg_angles, leg_angle_vels], dim = -1)
        inputs = []
        if actions is None:
            for i in np.arange(0, 8, 2):
                inputs.append(torch.stack([leg_states[:, :, i], leg_states[:, :, 8 + i], 
                                                 leg_states[:, :, i + 1], leg_states[:, :, 8 + i + 1]], dim = 2))
        else:
            for i in np.arange(0, 8, 2):
                inputs.append(torch.stack([leg_states[:, :, i], leg_states[:, :, 8 + i], 
                                                 leg_states[:, :, i + 1], leg_states[:, :, 8 + i + 1],
                                                 actions[:, :, i], actions[:, :, i + 1]],  dim = 2))
        inputs = torch.cat(inputs, dim = 1)
        return inputs
            
    def leg_predict(self, legs_diff, legs_curr = None):
        # Formats the prediction to original state representation
        
        next_legs = legs_diff
        if legs_curr is not None:
            next_legs = legs_curr + legs_diff
        n_states = legs_diff.shape[1] // 4
        legs_formatted = torch.zeros((legs_diff.shape[0], n_states, self.state_size)).to(self.device)
        for state_i in np.arange(n_states):
            for i in range(4):
                # Hip angle
                legs_formatted[:, state_i, (self.n_pose_dims + 2*i)] = next_legs[:, (n_states) * i + state_i, 0]
                # legs_formatted[:, state_i, (8 + 2*i)] = next_legs[:, (n_states) * i + state_i, 0]
                # Knee angle
                legs_formatted[:, state_i, (self.n_pose_dims + 1 + 2*i)] = next_legs[:, (n_states) * i + state_i, 2]
                # legs_formatted[:, state_i, (8 + 1 + 2*i)] = next_legs[:, (n_states) * i + state_i, 2]
                # Hip velocity
                legs_formatted[:, state_i, (self.n_pose_dims + 8 + 2*i)] = next_legs[:, (n_states) * i + state_i, 1]
                # legs_formatted[:, state_i, (29 - 7 + 2*i)] = next_legs[:, (n_states) * i + state_i, 1]
                # Knee velocity
                legs_formatted[:, state_i, (self.n_pose_dims + 8 + 1 + 2*i)] = next_legs[:, (n_states) * i + state_i, 3]
                # legs_formatted[:, state_i, (29 - 7 + 1 + 2*i)] = next_legs[:, (n_states) * i + state_i, 3]

        # print(f"This {next_legs[:, 1, 3]}")
        # print(f"Should agree with this {legs_formatted[:, 0, -5]} \n")
        return legs_formatted
                        
    def forward(self, inputs, predict_leg):
                
        if predict_leg:
                    
            leg_net_input = (inputs - self.mu_leg)/(self.sigma_leg + 1e-8)
            
            for weight, bias, act_fun in zip(self.weights_leg, self.biases_leg, self.act_funs):
                leg_net_input = torch.bmm(leg_net_input, weight) + bias
                leg_net_input = act_fun(leg_net_input)
                
            mean, log_var = leg_net_input[:, :, :self.n_leg_output], leg_net_input[:, :, self.n_leg_output:]
        
            log_var = self.max_log_var_leg - F.softplus(self.max_log_var_leg - log_var)
            log_var = self.min_log_var_leg + F.softplus(log_var - self.min_log_var_leg)
                
            return mean, log_var
        
        else:
            
            pose_net_input = (inputs - self.mu_pose)/(self.sigma_pose + 1e-8)
            
            for weight, bias, act_fun in zip(self.weights_pose, self.biases_pose, self.act_funs):
                pose_net_input = torch.bmm(pose_net_input, weight) + bias
                pose_net_input = act_fun(pose_net_input)
                
            mean, log_var = pose_net_input[:, :, :self.n_pose_output], pose_net_input[:, :, self.n_pose_output:]
        
            log_var = self.max_log_var_pose - F.softplus(self.max_log_var_pose - log_var)
            log_var = self.min_log_var_pose + F.softplus(log_var - self.min_log_var_pose)
                
            return mean, log_var
            
    def mean_prediction(self, state, act):
        state = torch.from_numpy(state).float().to(self.device).expand(self.ensemble_size, state.shape[0], -1)
        act = torch.from_numpy(act).float().to(self.device).expand(self.ensemble_size, act.shape[0], -1)
        inputs_leg = self.leg_net_processor(state, act)
        with torch.no_grad():
            legs_diff, _ = self.forward(inputs_leg, True)
            legs_diff = legs_diff * (self.sigma_target_leg + 1e-8) + self.mu_target_leg
            next_state = self.leg_predict(legs_diff, self.leg_net_processor(state))
            
            legs_diff_formatted = self.leg_predict(legs_diff)[:, :, -self.n_leg_dims:]
            # legs_diff_formatted = self.leg_predict(legs_diff)
            # legs_diff_formatted = torch.cat([legs_diff_formatted[:, :, 8:16], legs_diff_formatted[:, :, -8:]], dim = -1)
            pose_formatted = state[:, :, :self.n_pose_dims]
            # pose_formatted = torch.cat([state[:, :, :8], state[:, :, 16:22]], dim = -1)
            inputs_pose = torch.cat([pose_formatted, legs_diff_formatted], dim = -1)
            # inputs_pose = self.leg_predict(legs_diff)[:, :, -self.n_leg_dims:]
            # inputs_pose = next_state[:, :, -self.n_leg_dims:]
            pose_predict, _ = self.forward(inputs_pose, False)    
            pose_predict = pose_predict * (self.sigma_target_pose + 1e-8) + self.mu_target_pose
        
        next_state[:, :, :self.n_pose_dims] = pose_predict + state[:, :, :self.n_pose_dims]
        # next_state[:, :, :8] = pose_predict[:, :, :8] + state[:, :, :8]
        # next_state[:, :, 16:22] = pose_predict[:, :, 8:] + state[:, :, 16:22]
        
        mean_pred = next_state.mean(0)
        
        return mean_pred.cpu().numpy()
    
    # def ensemble_prediction(self, state, act, model_index):
    #     state = state.expand(self.ensemble_size, 1, -1)
    #     act = act.expand(self.ensemble_size, 1, -1)
    #     with torch.no_grad():
    #         state_diffs = self.forward(state, act)[model_index, :, :]
            
    #     prediction = state_diffs * (self.sigma_target + 1e-8) + self.mu_target
        
    #     return self.env_spec.state_postproc(state[0, :, :], prediction).squeeze(0)
    
    def sample_model(self, state, act):
                
        states = self._expand_to_ts_format(state)
        acts = self._expand_to_ts_format(act)
        inputs_leg = self.leg_net_processor(states, acts)
               
        with torch.no_grad():
            legs_diff, _ = self.forward(inputs_leg, True)
            legs_diff = legs_diff * (self.sigma_target_leg + 1e-8) + self.mu_target_leg
            next_state = self.leg_predict(legs_diff, self.leg_net_processor(states))
            
            # inputs_pose = torch.cat([states[:, :, :self.n_pose_dims], self.leg_predict(legs_diff)[:, :, -self.n_leg_dims:]], dim = -1)
            inputs_pose = self.leg_predict(legs_diff)[:, :, -self.n_leg_dims:]
            pose_predict, _ = self.forward(inputs_pose, False)    
            pose_predict = pose_predict * (self.sigma_target_pose + 1e-8) + self.mu_target_pose
                    
        next_state[:, :, :self.n_pose_dims] = pose_predict + states[:, :, :self.n_pose_dims]
        
        prediction = self._flatten_to_matrix(next_state)
        
        return prediction
    
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
        
        leg_states = states[:, -self.n_leg_dims:]
        
        # leg_angles = states[:, 8:16]
        # leg_angle_vels = states[:, -8:]
        
        # leg_states = np.concatenate([leg_angles, leg_angle_vels], axis = -1)
        
        leg_next_states = next_states[:, -self.n_leg_dims:]
        
        # next_leg_angles = next_states[:, 8:16]
        # next_leg_angle_vels = next_states[:, -8:]
        
        # leg_next_states = np.concatenate([next_leg_angles, next_leg_angle_vels], axis = -1)
        
        leg_net_input = []
        leg_net_output = []
        
        for i in np.arange(0, 8, 2):
                leg_net_input.append(np.stack([leg_states[:, i], leg_states[:, 8 + i], 
                                                 leg_states[:, i + 1], leg_states[:, 8 + i + 1],
                                                 actions[:, i], actions[:, i + 1]], axis = 1))
                leg_net_output.append(np.stack([leg_next_states[:, i], leg_next_states[:, 8 + i], 
                                                 leg_next_states[:, i + 1], leg_next_states[:, 8 + i + 1]], axis = 1))
        
        leg_net_input = np.concatenate(leg_net_input, axis = 0)
        leg_net_output = np.concatenate(leg_net_output, axis = 0)
        
        leg_net_input = torch.FloatTensor(leg_net_input).to(self.device).expand(self.ensemble_size, -1, -1)
        leg_net_output = torch.FloatTensor(leg_net_output).to(self.device).expand(self.ensemble_size, -1, -1)
        
        targets_leg = self.env_spec.target_proc(leg_net_input[:, :, :-2], leg_net_output)
        
        targets_leg = (targets_leg - self.mu_target_leg)/(self.sigma_target_leg + 1e-8)
        
        with torch.no_grad():
            leg_pred, _ = self.forward(leg_net_input, True)
            
        loss_leg = ((leg_pred - targets_leg) ** 2).mean(-1).mean(-1)
            
        pose_net_input = torch.FloatTensor(np.concatenate([states[:, :self.n_pose_dims], leg_next_states - leg_states]
                                                            , axis = -1)).to(self.device).expand(self.ensemble_size, -1, -1)
        # pose_net_input = torch.FloatTensor(np.concatenate([states[:, :8], states[:, 16:22],  leg_next_states - leg_states]
        #                                                     , axis = -1)).to(self.device).expand(self.ensemble_size, -1, -1)
        
        # pose_net_input = torch.FloatTensor(leg_next_states - leg_states).to(self.device).expand(self.ensemble_size, -1, -1)
        # pose_net_input = torch.FloatTensor(leg_states).to(self.device).expand(self.ensemble_size, -1, -1)
        
        targets_pose = self.env_spec.target_proc(states[:, :self.n_pose_dims], next_states[:, :self.n_pose_dims]) # states[:, :self.n_pose_dims]
        
        # targets_pose = self.env_spec.target_proc(np.concatenate([states[:, :8], states[:, 16:22]], axis = -1), 
        #                                           np.concatenate([next_states[:, :8], next_states[:, 16:22]], axis = -1))
        
        targets_pose = torch.FloatTensor(targets_pose).to(self.device).expand(self.ensemble_size, -1, -1)
        
        targets_pose = (targets_pose - self.mu_target_pose)/(self.sigma_target_pose + 1e-8)
        
        with torch.no_grad():
            pose_pred, _ = self.forward(pose_net_input, False)
            
        loss_pose = ((pose_pred - targets_pose) ** 2).mean(-1).mean(-1)
        
        return (loss_leg.cpu().numpy(), loss_pose.cpu().numpy())
        
    
    def fit_model(self, inputs, targets, n_epochs, batch_size, min_grad_upds, max_grad_upds, predict_leg, optim):
        n = inputs.shape[0]
        
        bootstrap_inds = np.random.randint(n, size=[self.ensemble_size, n])
        
        epoch_counter = 0
        grad_upd_counter = 0
        
        losses = []
                
        while True:
                        
            bootstrap_inds = shuffle_rows(bootstrap_inds)
            
            epoch_loss = 0
            
            for batch_n in range(int(np.ceil(n / batch_size))):
                
                batch_ind = bootstrap_inds[:, batch_n * batch_size:(batch_n + 1) * batch_size]
                                
                input_batch, target_batch = inputs[batch_ind, :], targets[batch_ind, :]
                                                    
                input_batch = torch.FloatTensor(input_batch).to(self.device)
                
                target_batch = torch.FloatTensor(target_batch).to(self.device)
                                
                mean, log_var = self.forward(input_batch, predict_leg)
                
                max_log_var, min_log_var = self.get_max_min_log_var(predict_leg)
                
                loss = 0.01 * torch.sum(max_log_var) - 0.01 * torch.sum(min_log_var)
                
                inv_var = torch.exp(-log_var)
                
                loss_log = ((mean - target_batch) ** 2) * inv_var + log_var
                
                loss_log = loss_log.mean(-1).mean(-1).sum()
                
                loss += loss_log 
                
                # loss = ((target_batch - outputs) ** 2).mean(-1).mean(-1).sum()
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                epoch_loss += loss.item()
                
                grad_upd_counter += 1
                # if grad_upd_counter >= max_grad_upds:
                #     break
                
            # if grad_upd_counter >= max_grad_upds:
            #     # print("Model training stopped due to grad max")
            #     break
            
            epoch_counter += 1
            
            losses.append(epoch_loss/n)
            
            # if grad_upd_counter < min_grad_upds:
            #     continue
            if epoch_counter >= n_epochs:
                # print("Model training stopped due to epoch max")
                return losses
        
    
    # n_epochs seems quite critical to performance  25 10**4  
    def update_parameters(self, n_epochs = 200, batch_size = 200, min_grad_upds = 10**2,
                          max_grad_upds = 10**5):
        
        # Shape: 0: Obs Index 1: Dim
        
        state_all, action_all, next_state_all = [np.array(t) for 
                                      t in zip(*self.replay_buffer)]
        
        # gen_loss = self.eval_loss(state_all[-1000:, :], action_all[-1000:, :], next_state_all[-1000:, :])
                
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
        sigma_leg = np.mean(np.abs(leg_net_input - mu_leg), axis = 0, keepdims = True)
        
        self.mu_leg.data = torch.FloatTensor(mu_leg).to(self.device)
        self.sigma_leg.data = torch.FloatTensor(sigma_leg).to(self.device)
        
        targets_leg = self.env_spec.target_proc(leg_net_input[:, :-2], leg_net_output)
        
        mu_targets_leg = np.mean(targets_leg, axis = 0, keepdims = True)
        sigma_targets_leg = np.mean(np.abs(targets_leg - mu_targets_leg), axis = 0, keepdims = True)
        
        targets_leg = (targets_leg - mu_targets_leg)/(sigma_targets_leg + 1e-8)
                
        self.mu_target_leg.data = torch.FloatTensor(mu_targets_leg).to(self.device)
        self.sigma_target_leg.data = torch.FloatTensor(sigma_targets_leg).to(self.device)
        
        pose_net_input = np.concatenate([state_all[:, :self.n_pose_dims], leg_next_states - leg_states], axis = -1)
        # pose_net_input = leg_states # leg_next_states - leg_states
        
        # pose_net_input = np.concatenate([state_all[:, :8], state_all[:, 16:22],  leg_next_states - leg_states]
        #                                                     , axis = -1)
        
        mu_pose = np.mean(pose_net_input, axis = 0, keepdims = True)
        sigma_pose = np.mean(np.abs(pose_net_input - mu_pose), axis = 0, keepdims = True)
        
        self.mu_pose.data = torch.FloatTensor(mu_pose).to(self.device)
        self.sigma_pose.data = torch.FloatTensor(sigma_pose).to(self.device)
        
        targets_pose = self.env_spec.target_proc(state_all[:, :self.n_pose_dims], next_state_all[:, :self.n_pose_dims]) # state_all[:, :self.n_pose_dims]
        
        # targets_pose = self.env_spec.target_proc(np.concatenate([state_all[:, :8], state_all[:, 16:22]], axis = -1), 
        #                                           np.concatenate([next_state_all[:, :8], next_state_all[:, 16:22]], axis = -1))
        
        mu_targets_pose = np.mean(targets_pose, axis = 0, keepdims= True)
        sigma_targets_pose = np.mean(np.abs(targets_pose - mu_targets_pose), axis = 0, keepdims = True)
        
        targets_pose = (targets_pose - mu_targets_pose)/(sigma_targets_pose + 1e-8)
                
        self.mu_target_pose.data = torch.FloatTensor(mu_targets_pose).to(self.device)
        self.sigma_target_pose.data = torch.FloatTensor(sigma_targets_pose).to(self.device)
        
        losses_leg = self.fit_model(leg_net_input, targets_leg, n_epochs, batch_size, min_grad_upds, max_grad_upds, True, self.optim_leg)
        losses_pose = self.fit_model(pose_net_input, targets_pose, n_epochs, batch_size, min_grad_upds, max_grad_upds, False, self.optim_pose)
        
        return losses_leg