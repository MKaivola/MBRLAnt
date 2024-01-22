import gymnasium as gym
from gymnasium import wrappers
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os

from timeit import default_timer as timer
from time import sleep

import env as env_specs

from td3 import TD3
from eff_locomotion import DETNET
from pets import PETS

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
        
        #self.env.seed(seed)
        self.env.action_space.seed(seed)
        # self.env_eval.seed(seed)
        self.env_eval.action_space.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        action_size, state_size = self.env.action_space.shape[0], self.env.observation_space.shape[0]
        
        max_action = self.env.action_space.high
        min_action = self.env.action_space.low
        
        if args.regularization == 'None':
            prepend_str = ''
        else:
            prepend_str = '_' + args.regularization
        
        self.output_dir = args.env_name + '_{}_horizon_{}_act_smooth_{}_dyn_epochs_{}_epoch_decay_{}'.format(args.model, args.horizon,
                                                                         args.action_smooth, 
                                                                         args.n_epochs_dyn, 
                                                                         args.n_epoch_decay_dyn) + prepend_str
        if args.TD3_init:
            self.output_dir += '_TD3_init'
            
        if args.use_TD3:
            self.output_dir += '_use_TD3'
        
        if args.regularization != 'None':
            self.output_dir += f'/reg_noise_std_{args.reg_noise_std}_reg_alpha_' + \
            f'{args.reg_alpha}_reg_epochs_{args.n_epochs_reg}_epoch_decay_{args.n_epoch_decay_reg}'
        
        self.output_dir += f'/seed_{seed}'
        
        os.makedirs(self.output_dir, mode = 0o755, exist_ok = True)
        
        if "RealAntMujoco-v0" in args.env_name:
            env_spec = env_specs.env_funcs.RealAntMujoco(args.task, args.latency, args.min_obs_stack)
        else:
            Exception("No such env defined")
        
        if args.model == 'Determ':
            self.model = DETNET(state_size + action_size, state_size, action_size,
                              min_action, max_action,
                              env_spec,
                              self.device,
                              horizon = args.horizon,
                              population_size = 400,
                              elite_size = 40,
                              hidden_size_dyn = 400,
                              hidden_size_reg = 400,
                              filter_coeff = args.action_smooth,
                              lr = 0.001,
                              n_cem_iterations = 7,
                              reg = args.regularization,
                              reg_alpha = args.reg_alpha,
                              reg_noise_std = args.reg_noise_std)
        elif args.model == 'PETS':
            self.model = PETS(state_size + action_size, state_size, action_size,
                             min_action, max_action,
                             env_spec,
                             self.device,
                             horizon = args.horizon,
                             population_size = 400,
                             elite_size = 40,
                             hidden_size = 256,
                             filter_coeff = args.action_smooth,
                             lr = 0.001,
                             holdout_ratio = 0,
                             n_cem_iterations = 5)
        else:
            Exception("No such model defined")
        
        self.agent = TD3(self.device, state_size, action_size)
        self.agent.load_model('td3_model_599.pt')
        
        self.args = args
        self.seed = seed
        
        body_dim_names = np.array(["x Vel", "y Vel", "z Vel", "z Pos", "Roll Vel", "Pitch Vel", "Yaw Vel", "Sin Roll", 
                          "Sin Pitch", "Sin Yaw", "Cos Roll", "Cos Pitch", "Cos Yaw"])
        
        leg_dim_names = np.concatenate((np.tile(np.array(["Hip Angle", "Knee Angle"]),4),
                                       np.tile(np.array(["Hip Vel", "Knee Vel"]),4)))
        
        self.dim_names = np.concatenate((body_dim_names,leg_dim_names))

    # def execute_action_seq(self, env, action_seq, n_executions = 10):
    #     ep_returns = []
    #     for _ in range(n_executions):
    #         env.reset()
    #         ep_return = 0
    #         for i in range(action_seq.shape[0]):
    #             _, reward, _, _ = env.step(action_seq[i, :])
    #             ep_return += reward
    #         ep_returns.append(ep_return)
    #     return np.mean(ep_returns), np.std(ep_returns, ddof=1)
    
    def evaluate_CEM(self): 
            state, info = self.env.reset()
            self.model.reset()
            
            for _ in range(25): 
                # action = self.agent.act(state, train = False, noise = 0.15)
                action = self.model.act(state, evaluation = False)
                
                next_state, reward, done, trunc, _ = self.env.step(action)
                state = next_state
                
            
            self.model.reset()
            
            pred_returns = []
            real_returns = []
            
            mean_action = None
            
            for _ in range(50):
                state = self.env_eval.copy_state(self.env)
                pred_state = state.copy()
                mean_action, _ = self.model.cem_test(pred_state, mean_action)
                pred_return = 0
                real_return = 0
                for i in range(self.args.horizon):
                    action = mean_action[i,:]
                    
                    pred_next_state = self.model.predict(pred_state, action)
                    pred_return += pred_next_state[0]
                    pred_state = pred_next_state
                    
                    next_state, reward, _, _ , _ = self.env_eval.step(action)
                    real_return += reward
                    
                pred_returns.append(pred_return)
                real_returns.append(real_return)
                
                
            f, ax = plt.subplots(figsize = (10.8, 19.2))
            
            cem_iter_count = np.arange(start = 1, stop = 51, step = 1)
            
            ax.plot(cem_iter_count, np.array(pred_returns), label = "Predicted Return")
            ax.plot(cem_iter_count, np.array(real_returns), label = "Real Return")
            
            ax.set_ylabel('Return', fontsize = 22)
            ax.set_xlabel('CEM iteration', fontsize = 22)
            ax.grid(visible = True)
            f.tight_layout()
            plt.legend(fontsize = 22)
            # plt.title(f"Regularization: {self.args.regularization}")
            plt.savefig(os.path.join(self.output_dir, 'CEM_Planning_Performance.pdf'))
    
        
    def evaluate_MPC(self, ep, horizon, n_episodes = 5):
        returns = []
        for ep_ind in range(n_episodes):
            state, info = self.env_eval.reset()
            done = False
            self.model.reset_eval()
            
            pred_state = state.copy()
            states = [state]
            pred_states = [pred_state]
        
            ep_return = 0
            step = 0
            
            for _ in range(self.args.ep_len):
                action = self.model.act(state, evaluation = True)
                
                next_state, reward, done, trunc, _ = self.env_eval.step(action)
                ep_return += reward
                
                states.append(next_state)
            
                pred_next_state = self.model.predict(pred_state, action)
                
                pred_states.append(pred_next_state)
                            
                state = next_state
                
                step += 1
            
                if step % horizon == 0:
                    pred_state = state
                else:
                    pred_state = pred_next_state
        
                if (done or trunc):
                    break
            returns.append(ep_return)
            
            states = np.stack(states)
            pred_states = np.stack(pred_states)
            
            if ep_ind == n_episodes - 1:
                with PdfPages(os.path.join(self.output_dir, f'MPC_predictions_ep_{ep}.pdf')) as pdf:
                    for dim in range(state.shape[0]):
                        plt.figure(figsize = (10, 10))
                        plt.plot(states[:, dim])
                        plt.plot(pred_states[:, dim])
                        plt.xlabel('Env step')
                        plt.ylabel(f'Dimension {dim}')
                        plt.title('Dynamics model performance ({} episodes)'.format(ep))
                        plt.legend(['Ground Truth', 'Predicted'])
                        pdf.savefig()
                        plt.close()
        
        print(f"Mean return after {ep} episodes: {np.mean(returns)} ")
        
        return np.mean(returns)
    
    def evaluate_dynamics(self, ep, horizons, switch_step):
        states = np.zeros((self.args.ep_len, self.env.observation_space.shape[0]))
        actions = np.zeros((self.args.ep_len - 1, self.env.action_space.shape[0]))
        pred_states = np.zeros((len(horizons), self.args.ep_len, self.env.observation_space.shape[0]))
        
        
        state, info = self.env_eval.reset()
        self.model.reset_eval()
        # pred_state = state.copy()
        states[0, :] = state
        # pred_states[horizon_ind, 0, :] = pred_state
            
        step = 0
        ep_return = 0
        for step_ind in range(self.args.ep_len - 1):
            
            if step < switch_step:
                action = self.agent.act(state, train = True, noise = 0.15)
            else:
                action = self.model.act(state, evaluation = True)
            
            next_state, reward, done, trunc, _ = self.env_eval.step(action)
            ep_return += reward
            
            states[step_ind + 1, :] = next_state
            actions[step_ind, :] = action
            
            # pred_next_state = self.model.predict(pred_state, action)
            
            # pred_states[horizon_ind, step_ind + 1, :] = pred_next_state
            
            state = next_state
            
            step += 1
            
            # if step % horizons[horizon_ind] == 0:
            #     pred_state = state
            # else:
            #     pred_state = pred_next_state
        
        print(ep_return)
        
        for horizon_ind in range(len(horizons)):
            pred_state = states[0, :]
            pred_states[horizon_ind, 0, :] = pred_state
            step = 0
            
            for step_ind in range(self.args.ep_len - 1):
                action = actions[step_ind, :]
                
                pred_next_state = self.model.predict(pred_state, action)
                pred_states[horizon_ind, step_ind + 1, :] = pred_next_state
                
                step += 1
                
                if step % horizons[horizon_ind] == 0:
                    pred_state = states[step_ind + 1, :]
                else:
                    pred_state = pred_next_state
            
        
        
        with PdfPages(os.path.join(self.output_dir, f'TD3_predictions_ep_{ep}.pdf')) as pdf:
            for dim in range(state.shape[0]):
                plt.figure(figsize = (19.2, 10.8))
                plt.plot(states[:, dim], label = "Ground Truth")
                for horizon_ind in range(len(horizons)):
                    plt.plot(pred_states[horizon_ind, :, dim], label = f"{horizons[horizon_ind]}-step prediction")
                
                plt.xlabel('Env step', fontsize = 22)
                plt.ylabel(self.dim_names[dim], fontsize = 22)
                plt.title('TD3 trajectory predictions')
                plt.legend(fontsize = 22)
                pdf.savefig()
                plt.close()
    
    def run_training(self, n_episodes, n_random_eps = 10):

        returns_MPC = []
        n_epochs_dyn = self.args.n_epochs_dyn
        n_epochs_reg = self.args.n_epochs_reg
        
        for ep in range(n_episodes):
            
            plan_times = []
        
            if ep >= n_random_eps:
                self.model.train_models(n_epochs_dyn = n_epochs_dyn, batch_size_dyn = 256,
                                        n_epochs_reg = n_epochs_reg, batch_size_reg = 256)
                n_epochs_dyn = int(max(5, n_epochs_dyn * self.args.n_epoch_decay_dyn))
                n_epochs_reg = int(max(5, n_epochs_reg * self.args.n_epoch_decay_reg))
                returns_MPC.append(self.evaluate_MPC(ep, 10))
                # self.evaluate_dynamics(ep, [1,15], 300)
                
            state, info = self.env.reset(seed = self.seed)
            done = False    
            self.model.reset()
            for _ in range(self.args.ep_len):
                
                if ep >= n_random_eps:
                    if self.args.use_TD3:
                        action = self.agent.act(state, train = True)
                    else:
                        plan_start = timer()
                        action = self.model.act(state, evaluation = False)
                        time_to_plan = timer() - plan_start
                        plan_times.append(time_to_plan)
                        # print(time_to_plan)
                else:
                    action = self.agent.act(state, train = True) if self.args.TD3_init else self.env.action_space.sample()
                
                next_state, reward, done, trunc, _ = self.env.step(action)
                self.model.save_transition(state, action, next_state, reward)
                
                state = next_state
                
                if (done or trunc):
                    break
            # print(f'Mean planning time: {np.array(plan_times).mean()}')
            
        # self.evaluate_CEM()
        np.save(os.path.join(self.output_dir, 'MPC_Performance_data'), np.array(returns_MPC))

# for i in range(args.n_train_epochs):
#     if 'Determ' in args.model:
#         model.train_models(n_epochs_dyn = 1, n_epochs_DAE = 1)
#     else:
#         model.train_dynamics_model(n_epochs = 1)
    # if (i + 1) & 3 == 0: 
        # evaluate_dynamics(agent, model, i, args.horizon)
        # returns_MPC.append(evaluate_MPC(model, i, args.horizon))

# mean_pred_return = []
# std_pred_return = []

# mean_real_return = []
# std_real_return = []

# TD3_actions = []

# state = env.reset()
# model.reset()

# TD3_return = 0

# for _ in range(args.horizon):
#     action = agent.act(state, train = False)
#     TD3_actions.append(action)
#     state, reward, _, _ = env.step(action)
#     TD3_return += reward

# state = env.reset()
# TD3_actions = torch.from_numpy(np.stack(TD3_actions)).float().to(device)

# action_mean, _, returns = model.cem_test(state, TD3_actions, torch.zeros((args.horizon, action_size)).to(device))

# mean_pred_return.append(returns.mean().cpu().numpy())
# std_pred_return.append(returns.std().cpu().numpy())

# mean_real, std_real = execute_action_seq(env, action_mean.cpu().numpy())
# mean_real_return.append(mean_real)
# std_real_return.append(std_real)

# state = env.reset()

# action_mean, action_var, returns = model.cem_test(state, TD3_actions)

# mean_pred_return.append(returns.mean().cpu().numpy())
# std_pred_return.append(returns.std().cpu().numpy())

# mean_real, std_real = execute_action_seq(env, action_mean.cpu().numpy())
# mean_real_return.append(mean_real)
# std_real_return.append(std_real)

# for _ in range(49):
#     state = env.reset()
    
#     action_mean, action_var, returns = model.cem_test(state, action_mean, action_var)
    
#     mean_pred_return.append(returns.mean().cpu().numpy())
#     std_pred_return.append(returns.std().cpu().numpy())
    
#     mean_real, std_real = execute_action_seq(env, action_mean.cpu().numpy())
#     mean_real_return.append(mean_real)
#     std_real_return.append(std_real)

# mean_pred_return = np.array(mean_pred_return)
# std_pred_return = np.array(std_pred_return)

# mean_real_return = np.array(mean_real_return)
# std_real_return = np.array(std_real_return)
    
# plt.figure(figsize = (19.2, 10.8))
# plt.plot(mean_pred_return, 'r')
# plt.fill_between(np.arange(51), mean_pred_return + std_pred_return, mean_pred_return - std_pred_return, alpha=0.2,
#                  color = 'r')
# plt.plot(mean_real_return, 'k')
# # plt.fill_between(np.arange(50), mean_real_return + std_real_return, mean_real_return - std_real_return, alpha=0.2)
# plt.hlines(TD3_return, 0, 50, colors = 'k', linestyles='dashed')
# plt.xlabel('CEM iteration')
# plt.ylabel('Return')
# plt.legend(['Predicted return', 'Real return', 'Std', 'TD3 return'])
# plt.savefig(os.path.join(output_dir, 'planner_returns.pdf'))

# def execute_action_seq_full_traj(env, action_seq):
#     traj_return = []
#     for i in range(action_seq.shape[0]):
#         state, reward, _, _ = env.step(action_seq[i, :])
#         traj_return.append(reward)
#     return state, traj_return

# def execute_policy(env, state, policy, n_steps):
#     traj_return = []
#     actions = []
#     for _ in range(n_steps):
#         action = policy.act(state, train = False)
#         actions.append(action)
#         state, reward, _, _ = env.step(action)
#         traj_return.append(reward)
#     return state, actions, traj_return

# state_TD3 = env.reset()
# state_MPC = env_eval.reset()

# TD3_returns = []
# MPC_returns = []

# for _ in range(args.ep_len // args.horizon):
#     state_TD3, TD3_actions, TD3_return = execute_policy(env, state_TD3, agent, args.horizon)
#     action_mean, action_var, _ = model.cem_test(state_MPC, torch.from_numpy(np.stack(TD3_actions)).float().to(device))
#     for _ in range(29):
#         action_mean, action_var, _ = model.cem_test(state_MPC, action_mean, action_var)
#     state_MPC, MPC_return = execute_action_seq_full_traj(env_eval, action_mean.cpu().numpy())
#     state_MPC = env_eval.copy_state(env)
#     TD3_returns.extend(TD3_return)
#     MPC_returns.extend(MPC_return)

# state_TD3, TD3_actions, TD3_return = execute_policy(env, state_TD3, agent, args.ep_len) #args.horizon)
# TD3_returns.extend(TD3_return)
# init_action = torch.from_numpy(np.stack(TD3_actions)).float().to(device)

# model.reset()

# for i in range(args.ep_len):
#     if i == 0:
#         action_mean, action_var, _ = model.cem_test(state_MPC)#, init_action)
#     else:
#         action_mean, action_var, _ = model.cem_test(state_MPC, init_action)
#     for _ in range(4):
#         action_mean, action_var, _ = model.cem_test(state_MPC, action_mean, action_var)
#     state_MPC, MPC_return = execute_action_seq_full_traj(env_eval, action_mean[0, :].cpu().numpy()[np.newaxis, :])
#     MPC_returns.extend(MPC_return)
#     if True:# i >= args.ep_len - args.horizon:
#         init_action = torch.cat([action_mean[1:, :], torch.zeros((1, action_size), device = device)], dim = 0)
#     else:
#         TD3_actions_prev = TD3_actions[1:]
#         state_TD3, TD3_actions, TD3_return = execute_policy(env, state_TD3, agent, 1)
#         TD3_returns.extend(TD3_return)
#         TD3_actions_prev.extend(TD3_actions)
#         init_action = torch.from_numpy(np.stack(TD3_actions_prev)).float().to(device)

# plt.figure(figsize = (19.2, 10.8))
# plt.plot(MPC_returns, 'r')
# plt.plot(TD3_returns, 'k')
# plt.xlabel('Action sequence')
# plt.ylabel('Return')
# plt.legend(['MPC return', 'TD3 return'])
# plt.title('MPC return {}, TD3 return {}'.format(np.round(np.sum(MPC_returns), 2), np.round(np.sum(TD3_returns), 2)))
# plt.savefig(os.path.join(output_dir, 'planner_TD3_episode_returns.pdf'))

# plt.figure(figsize = (19.2, 10.8))
# plt.plot(returns_MPC)
# plt.xlabel('Episode')
# plt.ylabel('Mean Return')
# plt.savefig(os.path.join(output_dir, 'MPC_return.pdf'))
