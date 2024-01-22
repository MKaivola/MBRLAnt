import numpy as np
import matplotlib.pyplot as plt
import os

seeds = [123, 643, 91, 213, 765]

f, ax = plt.subplots(figsize = (19.2, 10.8))

for agent_name in ['sac', 'td3']:
    
    name_dict = {
        'sac':'SAC',
        'td3':'TD3'}

    output_dir = f"RealAntMujoco-v0_{agent_name}_Walk"
            
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
ax.set_xlabel('Episode', fontsize = 22)
ax.set_ylabel('Return', fontsize = 22)
    
# ax.hlines(y, xmin = episode_steps[0], xmax = episode_steps[-1])

ax.grid(visible = True)
f.tight_layout()
plt.legend(fontsize = 22)
plt.savefig(os.path.join(output_dir, 'Performance.pdf'))