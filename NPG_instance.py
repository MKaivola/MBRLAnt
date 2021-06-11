import gym
from gym import wrappers
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os

import env as env_specs
from NPG import NPG_Agent
from SAC_MB import SAC
from NPG_model import NPG_Model

from collections import deque

def enforce_tensor_bounds(torch_tensor, min_val=None, max_val=None, 
                          large_value=float(1e2), device=None):
    """
        Clamp the torch_tensor to Box[min_val, max_val]
        torch_tensor should have shape (A, B)
        min_val and max_val can either be scalars or tensors of shape (B,)
        If min_val and max_val are not given, they are treated as large_value
    """
    # compute bounds
    if min_val is None: min_val = - large_value
    if max_val is None: max_val = large_value
    if device is None:  device = torch_tensor.data.device

    assert type(min_val) == float or type(min_val) == torch.Tensor
    assert type(max_val) == float or type(max_val) == torch.Tensor
    
    if type(min_val) == torch.Tensor:
        if len(min_val.shape) > 0: assert min_val.shape[-1] == torch_tensor.shape[-1]
    else:
        min_val = torch.tensor(min_val)
    
    if type(max_val) == torch.Tensor:
        if len(max_val.shape) > 0: assert max_val.shape[-1] == torch_tensor.shape[-1]
    else:
        max_val = torch.tensor(max_val)
    
    min_val = min_val.to(device)
    max_val = max_val.to(device)

    return torch.max(torch.min(torch_tensor, max_val), min_val)


class Train_Instance():
    
    def __init__(self, fargs):
        seed, args = fargs
        if 'RealAntMujoco-v0' in args.env_name:
            self.env = gym.make(
                'RealAntMujoco-v0',
                task = args.task,
                latency = args.latency,
                xyz_noise_std = args.xyz_noise_std,
                rpy_noise_std = args.rpy_noise_std,
                min_obs_stack = args.min_obs_stack
            )
            self.env_eval = gym.make(
                'RealAntMujoco-v0',
                task = args.task,
                latency = args.latency,
                xyz_noise_std = args.xyz_noise_std,
                rpy_noise_std = args.rpy_noise_std,
                min_obs_stack = args.min_obs_stack)
            
        else:
            self.env = gym.make(args.env_name)
            self.env_eval = gym.make(args.env_name)
            
        # self.env_eval = wrappers.Monitor(self.env_eval,'./videos/MPC_{}/seed_{}/'.format(args.model, seed),
        #                         video_callable=lambda episode_id: True, force = True)
        
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env_eval.seed(seed)
        self.env_eval.action_space.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        action_size, state_size = self.env.action_space.shape[0], self.env.observation_space.shape[0]
        
        max_action = self.env.action_space.high
        min_action = self.env.action_space.low
        
        self.output_dir = args.env_name + f'_{args.leader}_{args.policy_alg}'
        
        self.output_dir += f'/seed_{seed}'
        
        os.makedirs(self.output_dir, mode = 0o755, exist_ok = True)
        
        if 'NPG' in args.policy_alg:
            self.agent = NPG_Agent(state_size, action_size, max_action, min_action, self.device)
        elif 'SAC' in args.policy_alg:
            self.agent = SAC(state_size, action_size, max_action, self.device)
        else:
            Exception("No such alg exists")
                    
        if 'MBRLAnt' in args.env_name:
            self.env_spec = env_specs.env_funcs.MBRLAnt()
        elif 'MBRLCartpole' in args.env_name:
            self.env_spec = env_specs.env_funcs.MBRLCartpole()
        elif 'RealAntMujoco' in args.env_name:
            self.env_spec = env_specs.env_funcs.RealAntMujoco(args.task, args.latency, args.min_obs_stack)
        else:
            Exception("No such env spec exists")
                
        self.model = NPG_Model(state_size, action_size, state_size, self.device,
                               self.env_spec, ensemble_size = args.ensemble_size)
        
        self.args = args
        
        self.initial_state_buffer = deque(maxlen = 10**6)
        self.interm_state_buffer = deque(maxlen = 2500)
        
        self.seed = seed
        
    def evaluate_policy(self, n_samples_collected, n_episodes = 3):
        returns = []
        for ep in range(n_episodes):
            state = self.env_eval.reset()
            return_ep = 0
            done = False
            for _ in range(self.args.ep_len):
                action = self.agent.sample_action(state, evaluate = True)
                state, reward, done, _ = self.env_eval.step(action)
                return_ep += reward
                if done:
                    break
                
            returns.append(return_ep)
            
        mean_return = np.mean(returns)
        print(f"Mean performance after {n_samples_collected} samples: {mean_return}")
        
        return mean_return
    
    def evaluate_dynamics(self, n_samples_collected):
        state = self.env_eval.reset()
        pred_state = state.copy()
        states = [state]
        pred_states = [pred_state]
        actions = []
        
        horizon = 1
        step = 0
                
        for _ in range(self.args.ep_len):
            action = self.agent.sample_action(state, evaluate = True)
            actions.append(action)
            
            next_state, reward, done, _ = self.env_eval.step(action)
            
            states.append(next_state)
            
            pred_next_state = self.model.mean_prediction(pred_state, action)
            
            pred_states.append(pred_next_state)
            
            state = next_state
            
            if done:
                break
            
            step += 1
            
            if step % horizon == 0:
                pred_state = state
            else:
                pred_state = pred_next_state
        
        states = np.stack(states)
        pred_states = np.stack(pred_states)
        actions = np.stack(actions)
        
        with PdfPages(os.path.join(self.output_dir, f'Model_predictions_{n_samples_collected}.pdf')) as pdf:
            for dim in range(state.shape[0]):
                plt.figure(figsize = (10, 10))
                plt.plot(states[:, dim])
                plt.plot(pred_states[:, dim])
                plt.xlabel('Env step')
                plt.ylabel(f'Dimension {dim}')
                plt.title('Dynamics model performance ({} samples)'.format(n_samples_collected))
                plt.legend(['Ground Truth', 'Predicted'])
                pdf.savefig()
                plt.close()
                
        with PdfPages(os.path.join(self.output_dir, f'Action_trajectories_{n_samples_collected}.pdf')) as pdf:
            for dim in range(action.shape[0]):
                plt.figure(figsize = (10, 10))
                plt.plot(actions[:, dim])
                plt.xlabel('Env step')
                plt.ylabel(f'Action Dimension {dim}')
                plt.title('Action Trajectory ({} samples)'.format(n_samples_collected))
                pdf.savefig()
                plt.close()
    
    def collect_data(self, n_samples):
        sample_counter = 0
        # self.interm_state_buffer = deque(maxlen = n_samples)
        
        if 'RealAntMujoco-v0' in self.args.env_name:
            env = gym.make(
                'RealAntMujoco-v0',
                task = self.args.task,
                latency = self.args.latency,
                xyz_noise_std = self.args.xyz_noise_std,
                rpy_noise_std = self.args.rpy_noise_std,
                min_obs_stack = self.args.min_obs_stack
            )
        else:
            env = gym.make(self.args.env_name)
            
        env.seed(self.seed)
        
        while True:
            state = env.reset()
            self.initial_state_buffer.append(state)
            done = False
            for _ in range(self.args.ep_len):
                self.interm_state_buffer.append(state)
                action = self.agent.sample_action(state, evaluate = False)
                next_state, reward, done, _ = env.step(action)
                
                self.model.save_transition(state, action, next_state, reward)
                sample_counter += 1
                
                state = next_state
                if sample_counter >= n_samples or done:
                    break
            if sample_counter >= n_samples:
                break
            
        self.seed += 123
            
    def generate_real_rollouts(self, num_trajectories = 100):
        state_traj, action_traj, log_prob_traj, entropy_traj, reward_traj, terminated = [], [], [], [], [], []
        
        for _ in range(num_trajectories):
            states, actions, log_probs, entropies, rewards = [], [], [], [], []
            state = self.env.reset()
            state_torch = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            states.append(state_torch)
            done = False
            for _ in range(self.args.ep_len):
                action, log_prob, entropy = self.agent.sample_actions(state_torch)
                actions.append(action)
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                next_state, reward, done, _  = self.env.step(action.squeeze(0).cpu().numpy())
                rewards.append(torch.FloatTensor([reward]).to(self.device))
                state_torch = torch.from_numpy(next_state).float().to(self.device).unsqueeze(0)
                states.append(state_torch)
                
                if done:
                    break
            if done:
                terminated.append(True)
            else:
                terminated.append(False)
            state_traj.append(torch.cat(states, dim = 0))
            action_traj.append(torch.cat(actions, dim = 0))
            log_prob_traj.append(torch.cat(log_probs, dim = 0))
            entropy_traj.append(torch.cat(entropies, dim = 0))
            reward_traj.append(torch.cat(rewards, dim = 0))
            
        return state_traj, action_traj, log_prob_traj, entropy_traj, reward_traj, terminated
        
            
    def sim_rollout_gen(self):
        states, actions, log_probs, entropies, rewards= [], [], [], [], []
        
        # Sample initial states from the buffer (might need to sample intermediate states as well)
        # Since the actions pretty much stop changing after approx. 200 steps
        
        if 'RealAntMujoco-v0' in self.args.env_name:
            env = gym.make(
                'RealAntMujoco-v0',
                task = self.args.task,
                latency = self.args.latency,
                xyz_noise_std = self.args.xyz_noise_std,
                rpy_noise_std = self.args.rpy_noise_std,
                min_obs_stack = self.args.min_obs_stack
            )
        else:
            env = gym.make(self.args.env_name)
        
        # init_states = random.choices(self.initial_state_buffer, k = int(self.args.n_rollouts * 0.5))
        
        init_states = np.array([env.reset() for _ in range(self.args.n_rollouts // 2)])
        # init_states = np.array(random.choices(self.initial_state_buffer, k = self.args.n_rollouts // 2))
        interm_states = np.array(random.choices(self.interm_state_buffer, k = self.args.n_rollouts // 2))
        
        state = np.concatenate([init_states, interm_states], axis = 0)
        state = torch.FloatTensor(state).to(self.device)
        
        # Repeat each state for each ensemble model
        state = torch.repeat_interleave(state, self.args.ensemble_size, dim = 0)
        
        states.append(state)
        
        # H step rollout
        for _ in range(self.args.rollout_len):
            if 'NPG' in self.args.policy_alg:
                action, log_prob, entropy = self.agent.sample_actions(state)
                log_probs.append(log_prob)
                entropies.append(entropy)
            elif 'SAC' in self.args.policy_alg:
                action = self.agent.sample_actions(state)
            actions.append(action)
            
            # test_state = state[4, :]
            # test_action = action[4, :]
            
            # test_next_state = self.model.ensemble_prediction(test_state, test_action, 0)
            # print(f'Test prediction: {test_next_state}')
            
            # The bug is here with the sample_model method, the predictions do not agree
            # Should work now
            next_state = self.model.sample_model(state, action)
            next_state = enforce_tensor_bounds(next_state)
            # print(f'Actual prediction: {next_state[4, :]} \n')
            states.append(next_state)
            
            reward = self.env_spec.reward_fun(action, next_state)
            rewards.append(reward)
            
            state = next_state
            
        states = torch.stack(states, dim = 1)
        actions = torch.stack(actions, dim = 1)
        if 'NPG' in self.args.policy_alg:
            log_probs = torch.stack(log_probs, dim = 1)
            entropies = torch.stack(entropies, dim = 1)
        rewards = torch.stack(rewards, dim = 1)
        
        not_dones = self.env_spec.not_done(states, self.device)
        # avg_traj_len = not_dones.sum(-1).mean()
        # print(f"Mean trajectory len (sim): {avg_traj_len}")
        
        state_traj_trunc = []
        action_traj_trunc = []
        log_prob_traj_trunc = []
        entropy_traj_trunc = []
        reward_traj_trunc = []
        terminated_traj = []
        returns = []
        for traj_ind in range(states.shape[0]):
            
            terminal_state = (not_dones[traj_ind, :] == 0.0).nonzero()
            
            terminal_index = states.shape[1]
            
            if terminal_state.nelement() != 0:
                terminal_index = terminal_state[0, 0]
                terminated_traj.append(True)
            else:
                terminated_traj.append(False)
            state_traj_trunc.append(states[traj_ind, :(terminal_index + 1), :])
            action_traj_trunc.append(actions[traj_ind, :terminal_index, :])
            if 'NPG' in self.args.policy_alg:
                log_prob_traj_trunc.append(log_probs[traj_ind, :terminal_index])
                entropy_traj_trunc.append(entropies[traj_ind, :terminal_index])
            reward_traj_trunc.append(rewards[traj_ind, :terminal_index])
            returns.append(rewards[traj_ind, :terminal_index].sum().item())
                
        print(f'Mean return (sim): {np.mean(returns)}')
            
        return state_traj_trunc, action_traj_trunc, log_prob_traj_trunc, entropy_traj_trunc,  \
                reward_traj_trunc, terminated_traj
                    
    def run_training(self, n_total_samples, n_initial_samples = 2500):
        
        sample_counter = 0
        policy_perf = []
        policy_std = []
        
        policy_perf.append(self.evaluate_policy(sample_counter))
        policy_std.append(self.agent.get_policy_std())
        
        # Collect initial data
        print('Collecting initial data...')
        self.collect_data(n_initial_samples)

        sample_counter += n_initial_samples
        
        # for _ in range(50):
        #     states, actions, log_probs, entropies, rewards, terminated = self.generate_real_rollouts()
            
        #     self.agent.update_policy(states, actions,
        #                     log_probs, entropies, rewards, terminated)
        #     self.agent.update_value(states, rewards, log_probs, terminated)
            
        #     policy_perf.append(self.evaluate_policy(sample_counter))
        #     policy_std.append(self.agent.get_policy_std())
        
        while sample_counter <= n_total_samples:
            
            # Train models
            print('Training model...')
            self.model.update_parameters()
            # mu_state, sigma_state = self.model.get_state_stats()
            # self.agent.update_state_stats(mu_state, sigma_state, sim = False)
            
            # self.evaluate_dynamics(sample_counter)
            
            # NPG updates
            print('Starting policy updates...')
            for _ in range(self.args.n_NPG_updates):
                # Generate synthetic trajectories
                states, actions, log_probs, entropies, rewards, terminated = self.sim_rollout_gen()
                print('Rollouts done.')
                # Update policy
                self.agent.update_policy(states, actions,
                          log_probs, entropies, rewards, terminated)
                print('Policy Updated.')
                # Update value function
                if 'NPG' in self.args.policy_alg:
                    self.agent.update_value(states, rewards, log_probs, terminated)
                print('Value function updated.')
                
            self.evaluate_dynamics(sample_counter)
                        
            # Evaluate new policy
            policy_perf.append(self.evaluate_policy(sample_counter))
            policy_std.append(self.agent.get_policy_std())
            
            # Collect data under new policy
            print('Collecting data...')
            self.collect_data(self.args.n_iter_samples)
            
            sample_counter += self.args.n_iter_samples
        
        np.save(os.path.join(self.output_dir, 'Perf_data'), np.array(policy_perf))
        policy_std = np.stack(policy_std, axis = 0)
        with PdfPages(os.path.join(self.output_dir, 'std_trajectories.pdf')) as pdf:
            for dim in range(policy_std.shape[1]):
                plt.figure(figsize = (10, 10))
                plt.plot(policy_std[:, dim])
                plt.xlabel('Episode')
                plt.ylabel(f'Action dimension {dim}')
                plt.title('Policy standard deviation trajectory')
                pdf.savefig()
                plt.close()
        