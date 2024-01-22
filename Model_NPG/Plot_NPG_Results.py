import numpy as np
import matplotlib.pyplot as plt
import os
        
f, ax = plt.subplots(figsize = (19.2, 10.8))

seeds = [0, 2, 754, 6745]
for model, ep_step in zip(["PAL", 'MAL'],[5,15]):
    output_dir = f'RealAntMujoco-v0_{model}_NPG'
    
    returns = []
    
    for seed in seeds:
        data = np.load(output_dir + f'/seed_{seed}/Perf_data.npy')
        returns.append(data)
    
    
    returns = np.stack(returns, axis = 1)
    
    mean_return = np.mean(returns, axis = 1)
    std_return = np.std(returns, axis = 1, ddof = 1)
    episode_steps = np.arange(start = 0, stop = (returns.shape[0]) * ep_step, step = ep_step)
    
    ax.plot(episode_steps, mean_return, label = model)
    ax.fill_between(episode_steps, mean_return + std_return, mean_return - std_return, alpha = 0.2)

ax.set_xlabel('Episode', fontsize = 22)
ax.set_ylabel('Return', fontsize = 22)
ax.grid(visible = True)
f.tight_layout()
plt.legend(fontsize = 22)
plt.savefig('NPG_Performance.pdf')