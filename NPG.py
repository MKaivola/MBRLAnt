import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F

import numpy as np
from utils import *

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class NPG_NETS(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 device,
                 policy_hidden,
                 value_hidden,
                 lr_value,
                 reg_value):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(state_size, policy_hidden),
                                   nn.Tanh(),
                                   nn.Linear(policy_hidden, policy_hidden),
                                   nn.Tanh(),
                                    # nn.Linear(policy_hidden, policy_hidden),
                                    # nn.ReLU(),
                                   nn.Linear(policy_hidden, action_size))
        
        self.min_log_std = torch.ones(action_size, device = device) * -2.5
        
        self.policy_log_std = nn.Parameter(torch.ones(action_size, device = device) * 0.0)
                
        self.policy_params = [self.policy_log_std] + list(self.policy.parameters()) 
        
        # self.param_shapes = [p.data.numpy().shape for p in self.policy_params]
        # self.param_sizes = [p.data.numpy().size for p in self.policy_params]
        
        self.value = nn.Sequential(nn.Linear(state_size, value_hidden),
                                   nn.ReLU(),
                                   nn.Linear(value_hidden, value_hidden),
                                   nn.ReLU(),
                                    # nn.Linear(value_hidden, value_hidden),
                                    # nn.ReLU(),
                                   nn.Linear(value_hidden, 1))
        
        self.optim_value = optim.Adam(self.value.parameters(), weight_decay = reg_value)
        
        # for m in self.modules():
        #     if type(m) is nn.Linear:
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.zeros_(m.bias)
        
        for param in list(self.policy.parameters())[-2:]:
            param.data = 1e-2 * param.data
                                        
        self.to(device)

class NPG_Agent():
    def __init__(self,
                 state_size,
                 action_size,
                 max_action,
                 min_action,
                 device,
                 policy_hidden = 64,
                 value_hidden = 128,
                 lr_value = 1e-3,
                 reg_value = 0, #1e-3,
                 gamma = 0.995,
                 GAE_lambda = 0.97,
                 norm_step_size = 0.05):
        
        self.nets = NPG_NETS(state_size, 
                        action_size, 
                        device, 
                        policy_hidden, 
                        value_hidden, 
                        lr_value,
                        reg_value)
        
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = torch.from_numpy(max_action).float().to(device)
        self.min_action = torch.from_numpy(min_action).float().to(device)
        self.gamma = gamma
        self.GAE_lambda = GAE_lambda
        self.norm_step_size = norm_step_size
        self.device = device
        
        # self.reward_scale = 5.0
        # self.target_entropy = -action_size
        # self.log_alpha = torch.zeros(1, requires_grad = True, device = device)
        # self.optim_alpha = optim.Adam([self.log_alpha], lr = 3e-4)
        
        self.mu_state = nn.Parameter(torch.zeros((1, state_size), device = device), requires_grad = False)
        self.sigma_state = nn.Parameter(torch.ones((1, state_size), device = device), requires_grad = False)
        
        self.mu_state_sim = nn.Parameter(torch.zeros((1, state_size), device = device), requires_grad = False)
        self.sigma_state_sim = nn.Parameter(torch.ones((1, state_size), device = device), requires_grad = False)
        
    def sample_actions(self, states):
        
        # State standardization might be the cause of the observed performance decay (?)
        # Probably not
        states = (states - self.mu_state_sim)/(self.sigma_state_sim + 1e-8)
        
        # states = torch.clamp(states, min = -10.0, max = 10.0)/10.0
        
        mean = self.nets.policy(states)
                
        log_std = self.nets.policy_log_std
            
        dist = Normal(mean, log_std.exp())
        actions_raw = dist.sample()
        
        # Transform actions to correct scale (tanh might cause issues, more likely to improve performance in MBRLAnt)
        actions = torch.tanh(actions_raw)
        
        log_probs = dist.log_prob(actions_raw) #- torch.log(1 - actions**2 + 1e-6)
        log_probs = log_probs.sum(-1)
        entropy = dist.entropy()
        entropy = entropy.sum(-1)
        
        # Clamping does not seem to work as well as Tanh
        # actions = actions_raw
        # actions = torch.max(torch.min(self.max_action, actions), self.min_action)
        # log_probs = dist.log_prob(actions).sum(-1)

        return actions, log_probs, entropy
        
    def sample_action(self, state, evaluate):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        
        state = (state - self.mu_state)/(self.sigma_state + 1e-8)
        
        # state = torch.clamp(state, min = -10.0, max = 10.0)/10.0
        
        with torch.no_grad():
            mean = self.nets.policy(state)
                        
        log_std = self.nets.policy_log_std.detach().clone()
                    
        if evaluate:
            # action = torch.max(torch.min(self.max_action, mean), self.min_action)
            return torch.tanh(mean).squeeze(0).cpu().numpy() # action.squeeze(0).cpu().numpy()
        else:
            action = mean + torch.randn_like(mean) * log_std.exp()
            # Constant exploration noise level seems to not work
            # action = mean + torch.randn_like(mean) * 0.1
            # action = torch.max(torch.min(self.max_action, action), self.min_action)
            return torch.tanh(action).squeeze(0).cpu().numpy() # action.squeeze(0).cpu().numpy()
        
    def update_state_stats(self, new_mu, new_sigma, sim):
        if not sim:
            self.mu_state.data = new_mu.data
            self.sigma_state.data = new_sigma.data
        else:
            self.mu_state_sim.data = new_mu.data
            self.sigma_state_sim.data = new_sigma.data
        
    def get_policy_std(self):
        print(f"Policy std: {self.nets.policy_log_std.exp()}")
        return self.nets.policy_log_std.exp().detach().clone().cpu().numpy()
        
    def GAE(self, states, rewards, log_probs, terminated):
        
        # Inputs are 3-D tensors with 0: Traj ind, 1: Step ind, 2: Dim
        
        TD_resids = []
        
        for state, reward, log_prob, term in zip(states, rewards, log_probs, terminated):
            with torch.no_grad():
                state = torch.clamp(state, min = -10.0, max = 10.0)/10.0
                state_values = self.nets.value(state).squeeze(-1)
                state_values = torch.cat([state_values[:-1], 
                                          torch.tensor([0.0]).to(self.device) if term else state_values[-1, None]])
                TD_resid = reward + self.gamma * state_values[1:] - state_values[:-1]
                # Entropy regularization
                # TD_resid = reward * self.reward_scale - log_prob.detach() + self.gamma * state_values[1:] - state_values[:-1]
                TD_resids.append(TD_resid)
        
        advantages = []
        for TD_resid in TD_resids:
            advantage = torch.zeros_like(TD_resid)
            run_sum = 0.0
            for t in reversed(range(0, TD_resid.shape[0])):
                run_sum = TD_resid[t] + (self.gamma * self.GAE_lambda) * run_sum
                advantage[t] = run_sum
            advantages.append(advantage)
        
        # with torch.no_grad():
        #     state_values = self.nets.value(states).squeeze(-1) #* not_dones
        #     TD_resid = rewards + state_values[:, 1:] - state_values[:, :-1]
            
        # advantages = torch.zeros(TD_resid.shape[:2], device = self.device)
        # run_sum = 0.0
        # for t in reversed(range(0, TD_resid.shape[1])):
        #     run_sum = TD_resid[:, t] + (self.gamma * self.GAE_lambda) * run_sum
        #     advantages[:, t] = run_sum
        
        # Whitening (seems to contribute to performance decay (?))
        advantages = torch.cat(advantages, dim = 0)
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-6)
                                    
        return advantages
    
    def update_policy(self, states, actions,
                      log_probs, entropies, rewards, terminated):
        
        # Compute advantages
        advantages = self.GAE(states, rewards, log_probs, terminated)
        
        # Compute vanilla PG objective (surrogate CPI)
        # old_log_probs = log_probs.detach().clone() 
        
        # imp_ratio = (log_probs - old_log_probs).exp()
        # obj_fun = imp_ratio * advantages
        
        log_probs = torch.cat(log_probs, dim = 0)
        entropies = torch.cat(entropies, dim = 0)
        # log_probs_old = log_probs.detach().clone()
        # imp_ratios = (log_probs - log_probs_old).exp()
        
        # 0.001
        obj_fun = log_probs * advantages + 0.001 * entropies
                
        surr_CPI = obj_fun.mean()
                
        # Compute PG
        vpg = torch.autograd.grad(surr_CPI, self.nets.policy_params)
        vpg = flat_grad(vpg)
        
        print(f'VPG: {vpg} nan? : {torch.isnan(vpg).any()} inf? : {torch.isinf(vpg).any()}')
        
        states = torch.cat(states, dim = 0)
        states_cg = (states - self.mu_state_sim)/(self.sigma_state_sim + 1e-8)
        
        # states_cg = torch.clamp(states_cg, min = -10.0, max = 10.0)/10.0
                
        # Compute Natural PG
        npg = conjugate_gradient((self.nets.policy, self.nets.policy_params, self.nets.policy_log_std), states_cg, vpg, vpg.clone(),
                                 action_size = self.action_size, device = self.device)
        print(f'NPG: {npg}')
        
        # Update policy parameters
        current_params = flat_params(self.nets.policy_params)
        new_params = current_params + (torch.abs(self.norm_step_size / (torch.dot(vpg, npg) + 1e-10))).sqrt() * npg
        
        # set_param_values(self.nets.policy_params, new_params.clone(),
        #                  self.nets.param_sizes, self.nets.param_shapes, self.nets.min_log_std, self.device)
        # self.nets.policy_params = [self.nets.policy_log_std] + list(self.nets.policy.parameters()) 
        
        update_model(self.nets.policy_params, new_params)
        self.nets.policy_log_std.data.copy_(torch.max(self.nets.policy_log_std.data, self.nets.min_log_std))
        
        # Update simulated state stats
        # self.mu_state_sim.data = (states).mean(0).view(1, -1)
        # self.sigma_state_sim.data = torch.mean(torch.abs(states - (states).mean(0)), dim = 0).view(1, -1)
        
    # def update_temperature(self, log_probs):
    #     log_probs = torch.cat(log_probs, dim = 0)
    #     loss_alpha = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        
    #     self.optim_alpha.zero_grad()
    #     loss_alpha.backward()
    #     self.optim_alpha.step()
        
    def update_value(self, states, rewards, log_probs, terminated, epochs = 1, batch_size = 128):
                
        returns = []
        processed_states = []
        for state, reward, log_prob, term in zip(states, rewards, log_probs, terminated):
            returns_traj = []
            run_sum = 0.0
            for t in reversed(range(0, reward.shape[0])):
                run_sum = reward[t, None] + self.gamma * run_sum
                # run_sum = reward[t, None] * self.reward_scale - log_prob[t, None].detach() + self.gamma * run_sum
                returns_traj.append(run_sum)
            returns_traj.reverse()
            if term:
                returns_traj.append(torch.FloatTensor([0]).to(self.device))
                processed_states.append(state)
            else:
                processed_states.append(state[:-1, :])
                
            returns_traj = torch.cat(returns_traj, dim = 0)
            returns.append(returns_traj)
            
        returns = torch.cat(returns, dim = 0)
        
        states_target = torch.cat(processed_states, dim = 0)
        
        states_target = torch.clamp(states_target, min = -10.0, max = 10.0)/10.0
        
        n = states_target.shape[0]
        
        for epoch in range(epochs):

            indices = np.random.permutation(n)
            for batch_num in range(int(np.ceil(n / batch_size))):
                batch_inds = indices[batch_num * batch_size:(batch_num + 1) * batch_size]
                
                states_batch = states_target[batch_inds, :]
                returns_batch = returns[batch_inds]
                                
                value_est = self.nets.value(states_batch)
                loss = ((value_est - returns_batch) ** 2).mean()

                
                self.nets.optim_value.zero_grad()
                loss.backward()
                self.nets.optim_value.step()
