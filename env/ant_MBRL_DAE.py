import os

import numpy as np
import torch
from gym import utils
from gym.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_pos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_pos = np.copy(self.get_body_com("torso"))
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        #reward_ctrl = -0.5 * np.square(action).sum()
        reward_ctrl = -0.0 * np.square(action).sum()
        reward_run = ob[0]
        reward = reward_run + reward_ctrl

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone

        return ob, reward, done, {}

    def _get_obs(self):
        pos = self.get_body_com("torso")
        obs = np.concatenate([
            (pos - self.prev_pos) / self.dt,
            #pos,
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            #np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.prev_pos = np.copy(self.get_body_com("torso"))
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.sim.stat.extent * 0.5
        
    # @staticmethod
    # def obs_preproc(obs):
    #     return obs

    # @staticmethod
    # def obs_postproc(obs, pred):
    #     "Defined only for BNN model"
    #     return torch.cat([pred[:, :3], obs[:, 3:] + pred[:, 3:]], 1)

    # @staticmethod
    # def targ_proc(obs, next_obs):
    #     "Defined only for BNN model"
    #     return torch.cat([next_obs[:, :3], next_obs[:, 3:] - obs[:, 3:]], 1)

    # @staticmethod
    # def cost_fn(obs):
    #     return -obs[:, :, 0]