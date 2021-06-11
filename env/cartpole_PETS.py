from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    PENDULUM_LENGTH = 0.6

    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/cartpole.xml' % dir_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = CartpoleEnv.PENDULUM_LENGTH
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, CartpoleEnv.PENDULUM_LENGTH]))) / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
        return np.concatenate([obs[:1], np.sin(obs[1:2]), np.cos(obs[1:2]), obs[2:]])

    @staticmethod
    def _get_ee_pos(x):
        x0, sin_theta, cos_theta = x[0], x[1], x[2]
        return np.array([
            x0 - CartpoleEnv.PENDULUM_LENGTH * sin_theta,
            -CartpoleEnv.PENDULUM_LENGTH * cos_theta
        ])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent