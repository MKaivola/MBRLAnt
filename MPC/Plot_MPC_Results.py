import numpy as np
import matplotlib.pyplot as plt
import os


seeds = [0, 2, 51, 754, 6745]


 
filepaths = ["RealAntMujoco-v0_Determ_horizon_35_act_smooth_0.0_dyn_epochs_400_epoch_decay_1.0_DAE_TD3_init/"
             "reg_noise_std_0.3_reg_alpha_0.00045_reg_epochs_400_epoch_decay_1.0",
             "RealAntMujoco-v0_Determ_horizon_35_act_smooth_0.0_dyn_epochs_600_epoch_decay_0.6_DEEN_TD3_init/"
             "reg_noise_std_0.9_reg_alpha_0.00045_reg_epochs_800_epoch_decay_0.7"]
             # ,"RealAntMujoco-v0_PETS_horizon_35_act_smooth_0.0_dyn_epochs_5_epoch_decay_1.0"]

models = ["BNN", "BNN"]
regularizations = ['DAE', 'DEEN']

f, ax = plt.subplots(figsize = (19.2, 10.8))

for model, regularization, output_dir in zip(models, regularizations, filepaths):

    returns = []
    
    for seed in seeds:
        data = np.load(output_dir + f'/seed_{seed}/MPC_Performance_data.npy')
        returns.append(data)
    
    returns = np.stack(returns, axis = 1)
    episode_steps = np.arange(start = 0, stop = (returns.shape[0]) * 1, step = 1)
    
    mean_return = np.mean(returns, axis = 1)
    ax.plot(episode_steps, mean_return, label = f'{model} ({regularization})')
    
    std_return = np.std(returns, axis = 1, ddof = 1)
    
    ax.fill_between(episode_steps, mean_return + std_return, mean_return - std_return, alpha = 0.2)
    
ax.set_xlabel('Episode', fontsize = 22)
ax.set_ylabel('Return', fontsize = 22)
ax.grid(visible = True)
f.tight_layout()
plt.legend(fontsize = 22)
plt.savefig("MPC_Performance_TD3_init.pdf")