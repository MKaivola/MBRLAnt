import numpy as np
import torch

class MBRLHalfCheetah:
            
    def reward_fun(self, actions, next_states):
        
        reward_state = next_states[:, 0]
        reward_act = -0.1 * (actions ** 2).sum(dim = 1) 
        
        return reward_state + reward_act       
        
    def state_preproc(self, state):
        if isinstance(state, np.ndarray):
            return np.concatenate([state[:, 1:2], np.sin(state[:, 2:3]), np.cos(state[:, 2:3]), state[:, 3:]], axis = 1)
        elif isinstance(state, torch.Tensor):
            return torch.cat([
                state[:, :, 1:2],
                state[:, :, 2:3].sin(),
                state[:, :, 2:3].cos(),
                state[:, :, 3:]
            ], dim = 2)
    
    def target_proc(self, state, next_state):
        return torch.cat([
                next_state[:, :, :1],
                next_state[:, :, 1:] - state[:, :, 1:]
            ], dim = 2)
    
    def state_postproc(self, state, pred):
        return torch.cat([
            pred[:, :1],
            state[:, 1:] + pred[:, 1:]
        ], dim = 1)

    
class MBRLCartpole:
    
    def not_done(self, states, device):
        not_dones = torch.ones(states.shape[:2], device = device)
        
        return not_dones
            
    def reward_fun(self, actions, next_states):
        
        x0, sin_theta, cos_theta  = next_states[:, :1], next_states[:, 1:2], next_states[:, 2:3]
        ee_pos = torch.cat([x0 - 0.6 * sin_theta, -0.6 * cos_theta], 1)
        state_reward = torch.exp(-torch.sum(
            torch.pow(ee_pos - torch.FloatTensor([0.0, 0.6]).to(next_states.device), 2), 1
        ) / (0.6 ** 2))
        
        action_reward = -0.01 * (actions ** 2).sum(-1)
        
        return state_reward + action_reward
        
    def state_preproc(self, state):
        return state
    
    def target_proc(self, state, next_state):
        return next_state - state
    
    def state_postproc(self, state, pred):
        return state + pred
    
class RealAntMujoco:
    
    def __init__(self, task, latency, min_obs_stack):
        self.task = task

        self.n_delay_steps = latency # 1 step = 50 ms
        self.n_past_obs = self.n_delay_steps + min_obs_stack
        
    def not_done(self, states, device):
        not_dones = torch.ones(states.shape[:2], device = device)
        
        # not_dones[torch.logical_or(states[:, :, 3] < 0.063, states[:, :, 3] > 0.31)] = 0.0
        
        # not_dones = torch.cumprod(not_dones, dim = 1)
        
        return not_dones
        
    def reward_fun(self, actions, next_states, device):
        
        if self.task == 'walk':
            reward = next_states[:, 0]
        elif self.task == 'sleep':
            reward = -(next_states[:, 3]) ** 2
        elif self.task == 'turn':
            goal = torch.zeros((3), device = device)
            body_rpy = torch.atan2(next_states[:, 7:10], next_states[:, 10:13])
            reward = -torch.square(goal[0]-body_rpy[:,0])
        else:
            raise Exception('Unknown task')
        
        return reward
        
    def state_preproc(self, state):
        return state
    
    def target_proc(self, state, next_state):
        # return next_state[:, :, -29:] - state[:, :, -29:]
        return next_state - state
    
    def state_postproc(self, state, pred):
        # return torch.cat([state[:, 29:], pred + state[:, -29:]], dim = 1)
        return state + pred
    
class MBRLAnt:
    def reward_fun(self, actions, next_states):
        
        forward_reward = next_states[:, 0]
        ctrl_cost = .5 * (actions ** 2).sum(-1)
        
        # Last 84 dimensions
        
        # contact_cost = 0.5 * 1e-3 * torch.sum(
        #     torch.clamp(next_states[:, -84:], min = -1, max = 1) ** 2)
        survive_reward = 1.0
        reward = forward_reward + survive_reward - ctrl_cost
        
        return reward
        
        # return next_states[:, 0]
    
    def not_done(self, states, device):
        # Maybe third index?
        not_dones = torch.ones(states.shape[:2], device = device)
        not_dones[torch.logical_or(states[:, :, 3] < 0.2, states[:, :, 3] > 1.0)] = 0.0
        
        not_dones = torch.cumprod(not_dones, dim = 1)
        
        return not_dones
    
    def state_preproc(self, state):
        return state
    
    def target_proc(self, state, next_state):
        # if isinstance(state, np.ndarray):
        #     return np.concatenate([next_state[:, :3], next_state[:, 3:] - state[:, 3:]], -1)
        # else:
        #     return torch.cat([next_state[:, :, :3], next_state[:, :, 3:] - state[:, :, 3:]], -1)
        return next_state - state
    
    def state_postproc(self, state, pred):
        # return torch.cat([pred[:, :3], state[:, 3:] + pred[:, 3:]], -1)
        return state + pred

# class Inverted_Pendulum:
    
#     def reset(self, n_trajectories, device):
#         self.not_dones = torch.ones((n_trajectories), device = device, dtype = torch.bool)
#         self.reward = torch.ones((n_trajectories), device = device)
#         self.penalty = torch.zeros((n_trajectories), device = device)
        
#     def reward_fun(self, actions, next_states):
        
#         # x0, sin_theta, cos_theta = next_states[:, :1], next_states[:, 1:2], next_states[:, 2:3]
#         # ee_pos = torch.cat([x0 - 0.6 * sin_theta, -0.6 * cos_theta], 1)
#         # state_reward = torch.exp(-torch.sum(
#         #     torch.pow(ee_pos - torch.FloatTensor([0.0, 0.6]).to(next_states.device), 2), 1
#         # ) / (0.6 ** 2))
        
#         # action_reward = - 0.01 * (actions ** 2).sum(-1)
        
#         # return state_reward + action_reward
        
#         reward = torch.where(self.not_dones, self.reward, self.penalty)
        
#         not_done = torch.abs(next_states[:, 1]) <= .2
              
#         self.not_dones = torch.logical_and(self.not_dones, not_done)
        
#         return reward #- 0.01 * (actions ** 2).sum(-1)
            
#     def next_state_preproc(self, action, next_state, reward):
#         # return np.concatenate([next_state[:1], np.sin(next_state[1:2]), np.cos(next_state[1:2]), next_state[2:]])
#         return next_state
    
#     def state_preproc(self, state):
#         # return np.concatenate([state[:1], np.sin(state[1:2]), np.cos(state[1:2]), state[2:]])
#         return state
        
# class Half_Cheetah:
#     def reward_fun(self, actions, next_states):
        
#         reward_state = next_states[:,0]
#         reward_act = -0.1 * (actions ** 2).sum(dim = 1) 
        
#         return reward_state + reward_act
    
#     def next_state_preproc(self, action, next_state, reward):
        
#         reward_state = reward - (-0.1 * np.square(action).sum())
#         return np.concatenate([[reward_state], next_state])
    
#     def obs_preproc(self, state):
#         return state