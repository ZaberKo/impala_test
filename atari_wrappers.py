import gym
from gym.wrappers import TimeLimit
from ray.rllib.env.wrappers.atari_wrappers import get_wrapper_by_cls
from ray.tune.registry import register_env

def register_timelimit_env(env_name, max_episode_steps=3600*5):

    def env_creator(env_config):
        env=gym.make(env_name,**env_config)
        timelimit_wrapper=get_wrapper_by_cls(env, TimeLimit)
        new_env=TimeLimit(timelimit_wrapper.env, max_episode_steps=max_episode_steps)
        return new_env

    new_env_name=f"{env_name}-TimeLimit{max_episode_steps}"
    register_env(new_env_name,env_creator)

    return new_env_name