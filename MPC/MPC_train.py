import argparse
from MPC_instance import Train_Instance
import torch.multiprocessing as mp
import itertools

import numpy as np
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()

# Environment parameters
parser.add_argument('--ep_len', type=int, default=200,
                    help='Length of an episode')
parser.add_argument('--env_name', type=str, required=True,
                    help='Name of the environment')


# Dynamics hyperparameters
parser.add_argument('--model', type=str, default='Determ', choices=['Determ', 'PETS'],
                    help='Dynamics model used')
parser.add_argument('--n_epochs_dyn', type=int, default=200,
                    help='Number of training epochs per episode for dynamics')
parser.add_argument('--n_epoch_decay_dyn', type=float, default=1.0,
                    help='Decay factor for number of training epochs')


# Planning hyperparameters
parser.add_argument('--horizon', type=int, default=10,
                        help='CEM planning horizon')
parser.add_argument('--action_smooth', type=float, default=0.0,
                        help='Action smoothing coefficient')
parser.add_argument('--regularization', type=str, default='None', choices=['None', 'DAE', 'DEEN', 'RND'],
                    help='Regularization method in planning')
parser.add_argument('--n_epochs_reg', type=int, default=200,
                    help='Number of training epochs per episode for regularizer')
parser.add_argument('--reg_noise_std', type=float, default=0.3,
                    help='Std of Gaussian noise added to inputs in regularization')
parser.add_argument('--reg_alpha', type=float, default=0.001,
                    help='Penalty scaling term for regularization')
parser.add_argument('--n_epoch_decay_reg', type=float, default=1.0,
                    help='Decay factor for number of training epochs')

# Training loop hyperparameters
parser.add_argument('--n_data_episodes', type=int, default=30,
                    help='Number of data collection episodes')
parser.add_argument('--n_random_episodes', type=int, default=10,
                    help='Number of random initial episodes')
parser.add_argument('--TD3_init', action='store_true', 
                    help='Use TD3 trajectories as initial data')
parser.add_argument('--use_TD3', action='store_true', 
                    help='Use TD3 as the behavior policy')

# RealAnt parameters
parser.add_argument("--task", default="walk")
parser.add_argument("--latency", default=0, type=int)
parser.add_argument("--xyz_noise_std", default=0.00, type=float)
parser.add_argument("--rpy_noise_std", default=0.00, type=float)
parser.add_argument("--min_obs_stack", default=1, type=int)

def start_training(fargs):
    train_instance, n_episodes, n_random_episodes = fargs
    train_instance.run_training(n_episodes, n_random_episodes)
    
if __name__ == '__main__':
    args = parser.parse_args()
    seeds = [0, 2, 51, 754, 6745]
    # seeds = [3]
    
    mp.set_start_method('spawn')
    
    pool = mp.Pool(processes=mp.cpu_count())
    
    train_instances = [Train_Instance((seed, args)) for seed in seeds]
    
    pool.map(start_training, zip(train_instances, 
                                  itertools.repeat(args.n_data_episodes, len(train_instances)),
                                  itertools.repeat(args.n_random_episodes, len(train_instances))))
    
    if args.regularization == 'None':
        prepend_str = ''
    else:
        prepend_str = '_' + args.regularization
        
    output_dir = args.env_name + '_{}_horizon_{}_act_smooth_{}_dyn_epochs_{}_epoch_decay_{}'.format(args.model, args.horizon,
                                                                         args.action_smooth, args.n_epochs_dyn, 
                                                                         args.n_epoch_decay_dyn) + prepend_str
    if args.TD3_init:
        output_dir += '_TD3_init'
        
    if args.use_TD3:
        output_dir += '_use_TD3'
        
    if args.regularization != 'None':
        output_dir += f'/reg_noise_std_{args.reg_noise_std}_reg_alpha_{args.reg_alpha}_' + \
            f'reg_epochs_{args.n_epochs_reg}_epoch_decay_{args.n_epoch_decay_reg}'
                
    returns = []
    f, ax = plt.subplots(figsize = (19.2, 10.8))

    for seed in seeds:
        data = np.load(output_dir + f'/seed_{seed}/MPC_Performance_data.npy')
        returns.append(data)
    
    returns = np.stack(returns, axis = 1)
    episode_steps = np.arange(start = 0, stop = (returns.shape[0]) * 1, step = 1)
    
    mean_return = np.mean(returns, axis = 1)
    ax.plot(episode_steps, mean_return, label = f'{args.model} ({args.regularization})')
    
    std_return = np.std(returns, axis = 1, ddof = 1)
    
    ax.fill_between(episode_steps, mean_return + std_return, mean_return - std_return, alpha = 0.2)
    
    ax.set_xlabel('Episode', fontsize = 22)
    ax.set_ylabel('Return', fontsize = 22)
    ax.grid(visible = True)
    f.tight_layout()
    plt.legend(fontsize = 22)
    plt.savefig(os.path.join(output_dir, 'Performance.pdf'))