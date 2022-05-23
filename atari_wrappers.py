import gym
from gym.wrappers import TimeLimit, AtariPreprocessing
from ray.rllib.env.wrappers.atari_wrappers import get_wrapper_by_cls, FrameStack
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.atari_wrappers import MonitorEnv


def register_my_env(env_name, max_episode_steps: int = None):

    def env_creator(env_config):
        env = gym.make(env_name, **env_config)
        if max_episode_steps is not None:
            timelimit_wrapper = get_wrapper_by_cls(env, TimeLimit)

            env = TimeLimit(timelimit_wrapper.env,
                            max_episode_steps=max_episode_steps)

        env = MyMonitorEnv(env)
        env = AtariPreprocessing(env,
                                 noop_max=30,
                                 frame_skip=4,
                                 screen_size=84,
                                 terminal_on_life_loss=False,
                                 grayscale_obs=True,
                                 grayscale_newaxis=True,
                                 scale_obs=False,
                                 )

        env = FrameStack(env, 4)
        return env

    new_env_name = f"{env_name}-TimeLimit{max_episode_steps}"
    register_env(new_env_name, env_creator)

    return new_env_name


class MyMonitorEnv(MonitorEnv):
    def __init__(self, env=None):
        super().__init__(env)
        self._total_steps = 0
        self._done = True

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self._current_reward = 0
        self._num_steps = 0

        # handle horizon case
        if not self._done:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1
            self._done = True

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1

        self._done = done

        if done:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._num_episodes += 1

        return (obs, rew, done, info)
