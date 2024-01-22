from gymnasium.envs.registration import register

from . import env_funcs

register(
    id='RealAntBullet-v0',
    entry_point='env.pybullet:AntBulletEnv',
    max_episode_steps=600,
)

register(
    id='RealAntMujoco-v0',
    entry_point='env.mujoco:AntEnv',
    max_episode_steps=200,
)
