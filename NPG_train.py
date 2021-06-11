import argparse
from NPG_instance import Train_Instance
import torch.multiprocessing as mp
import itertools

import numpy as np
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()

# Environment parameters
parser.add_argument('--env_name', type=str, required=True,
                    help='Name of the environment')
parser.add_argument('--n_total_samples', type=int, default=10000,
                    help='Number of samples collected in total')
parser.add_argument('--ep_len', type=int, default = 1000,
                    help='Length of an episode')

parser.add_argument('--policy_alg', type=str, default = 'NPG', choices=['NPG', 'SAC'],
                    help='Policy optimization algorithm')
parser.add_argument('--leader', type=str, default = 'PAL',
                    help='Algorithm family')
parser.add_argument('--n_rollouts', type=int, default = 200,
                    help='Number of rollouts per iteration')
parser.add_argument('--rollout_len', type=int, default = 500,
                    help='Rollout length')
parser.add_argument('--ensemble_size', type=int, default = 4,
                    help='Number of dynamics models in ensemble')
parser.add_argument('--n_iter_samples', type=int, default = 1000,
                    help='Number of samples collected in each iteration')
parser.add_argument('--n_NPG_updates', type=int, default = 4,
                    help='Number of NPG updates per iteration')


# RealAnt parameters
parser.add_argument("--task", default="walk")
parser.add_argument("--latency", default=0, type=int)
parser.add_argument("--xyz_noise_std", default=0.00, type=float)
parser.add_argument("--rpy_noise_std", default=0.00, type=float)
parser.add_argument("--min_obs_stack", default=1, type=int)

def start_training(fargs):
    train_instance, n_total_samples = fargs
    train_instance.run_training(n_total_samples)
    
if __name__ == '__main__':
    args = parser.parse_args()
    # seeds = [0, 2, 51]#, 754]#, 6745]
    seeds = [3]
    
    # mp.set_start_method('spawn')
    
    # pool = mp.Pool(processes=mp.cpu_count())
    
    # train_instances = [Train_Instance((seed, args)) for seed in seeds]
    
    # pool.map(start_training, zip(train_instances, 
    #                               itertools.repeat(args.n_total_samples, len(train_instances))))
    
    train_instance = Train_Instance((seeds[0], args))
    train_instance.run_training(args.n_total_samples)
    
    output_dir = args.env_name + f'_{args.leader}_{args.policy_alg}'
            
    returns = []
    plt.figure(figsize = (19.2, 10.8))

    for seed in seeds:
        data = np.load(output_dir + f'/seed_{seed}/Perf_data.npy')
        returns.append(data)
    
    returns = np.stack(returns, axis = 1)
    plt.plot(returns)
    plt.plot(np.mean(returns, axis = 1), linestyle='dashed')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.savefig(os.path.join(output_dir, 'performance.pdf'))
