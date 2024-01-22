import argparse
import random
import numpy as np
import torch

import gymnasium as gym
import realant_sim

from td3 import TD3
from sac import SAC

import torch.multiprocessing as mp
import itertools

import os
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

def rollout(agent, env, train=False, random=False):
    state, info = env.reset()
    episode_step, episode_return = 0, 0
    done = False
    trunc = False
    while not (done or trunc) :
        if random:
            action = env.action_space.sample()
        else:
            action = agent.act(state, train=train)

        next_state, reward, done, trunc, info = env.step(action)
        episode_return += reward

        if train:
            not_done = 1.0 if (episode_step+1) == env._max_episode_steps else float(not done)
            agent.replay_buffer.append([state, action, [reward], next_state, [not_done]])
            agent._timestep += 1

        state = next_state
        episode_step += 1

    if train and not random:
        for _ in range(episode_step):
            agent.update_parameters()

    return episode_return

def evaluate(agent, env, n_episodes=10):
    returns = [rollout(agent, env, train=False, random=False) for _ in range(n_episodes)]
    return np.mean(returns)

def train(agent, env, seed, output_dir, args, n_episodes=1000, n_random_episodes=10):
    returns = []
    eval_return = evaluate(agent, env)
    returns.append(eval_return)
    for episode in range(n_episodes):
        train_return = rollout(agent, env, train=True, random=episode<n_random_episodes)
        print(f'Episode {episode}. Return {train_return}')

        if (episode+1) % 10 == 0:
            eval_return = evaluate(agent, env)
            returns.append(eval_return)
            print(f'Eval Reward ({seed}) {eval_return}')
        
        if (episode+1) % 100 == 0:
            agent.save_model(os.path.join(output_dir, '{}_model_{}.pt'.format(args.agent, episode)))
            
    np.save(os.path.join(output_dir, 'Perf_data'), np.array(returns))
            
def run_training(fargs):
    
    seed, args = fargs

    if args.env == 'mujoco':
        env =  gym.make(
            'RealAntMujoco-v0',
            task=args.task,
            latency=args.latency,
            xyz_noise_std=args.xyz_noise_std,
            rpy_noise_std=args.rpy_noise_std,
            min_obs_stack=args.min_obs_stack,
        )
    elif args.env == 'pybullet':
        env = gym.make('RealAntBullet-v0', task=args.task)
    else:
        raise Exception('Unknown env')
    
    #env =  gym.make(
            #'MBRLAnt-v0'#,
            # task=args.task,
            # latency=args.latency,
            # xyz_noise_std=args.xyz_noise_std,
            # rpy_noise_std=args.rpy_noise_std,
            # min_obs_stack=args.min_obs_stack,
            #)

    obs_size, act_size = env.observation_space.shape[0], env.action_space.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = f"RealAntMujoco-v0_{args.agent}/seed_{seed}"
    os.makedirs(output_dir, mode = 0o755, exist_ok = True)

    # env.seed(seed)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.agent == 'td3':
        agent = TD3(device, obs_size, act_size)
    elif args.agent == 'sac':
        agent = SAC(device, obs_size, act_size)
    else:
        raise Exception('Unknown agent')

    train(agent, env, seed, output_dir, args, n_episodes=400, n_random_episodes=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="td3")  # td3 or sac
    parser.add_argument("--env", default="mujoco") # mujoco or pybullet
    parser.add_argument("--task", default="walk")  # sleep or turn or walk

    # parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--latency", default=2, type=int)
    parser.add_argument("--xyz_noise_std", default=0.01, type=float)
    parser.add_argument("--rpy_noise_std", default=0.01, type=float)
    parser.add_argument("--min_obs_stack", default=4, type=int)
    
    args = parser.parse_args()

    mp.set_start_method('spawn')
    
    pool = mp.Pool(processes=mp.cpu_count())
    
    seeds = [123, 643, 91, 213, 765]
    
    pool.map(run_training, zip(seeds, 
                                itertools.repeat(args, len(seeds))))
    
    f, ax = plt.subplots(figsize = (19.2, 10.8))
    
    for agent_name in ['sac', 'td3']:
        
        name_dict = {
            'sac':'SAC',
            'td3':'TD3'}
    
        output_dir = f"RealAntMujoco-v0_{agent_name}"
                
        returns = []
    
        for seed in seeds:
            data = np.load(output_dir + f'/seed_{seed}/Perf_data.npy')
            returns.append(data)
        
        returns = np.stack(returns, axis = 1)
        mean_return = np.mean(returns, axis = 1)
        std_return = np.std(returns, axis = 1, ddof = 1)
        
        episode_steps = np.arange(start = 0, stop = (mean_return.shape[0]) * 10, step = 10)
            
        ax.plot(episode_steps, mean_return, label = name_dict[agent_name])
        ax.fill_between(episode_steps, mean_return + std_return, mean_return - std_return, alpha = 0.2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
        
    # ax.hlines(y, xmin = episode_steps[0], xmax = episode_steps[-1])
    
    ax.grid(visible = True)
    f.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Performance.pdf'))

