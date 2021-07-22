from gym.envs.registration import register

from . import env_funcs

register(
        id='MBRLCartpole-v0',
        entry_point='env.cartpole_PETS:CartpoleEnv'
)

register(
        id='MBRLHalfCheetah-v0',
        entry_point='env.cheetah_PETS:HalfCheetahEnv'
)

register(
    id='RealAntMujoco-v0',
    entry_point='env.RealAnt_Mujoco:AntEnv',
    max_episode_steps=200,
)

register(
    id='MBRLAnt-v0',
    entry_point='env.ant_MBRL:AntEnv',
    max_episode_steps=200)