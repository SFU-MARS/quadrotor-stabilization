"""Hanyang
Store some useful functions.
"""

import gym
import numpy as np
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

class NormalizeActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Store both the high and low arrays in their original forms
        self.action_space_low = self.action_space.low
        self.action_space_high = self.action_space.high

        # We normalize action space to a range [-1, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.action_space.shape, dtype=np.float32)

    def action(self, action):
        # convert action from [-1,1] to original range
        action = self.denormalize_action(action)
        return action

    def reverse_action(self, action):
        # convert action from original range to [-1,1]
        action = self.normalize_action(action)
        return action

    def normalize_action(self, action):
        action = 2 * ((action - self.action_space_low) / (self.action_space_high - self.action_space_low)) - 1
        return action

    def denormalize_action(self, action):
        action = (action + 1) / 2 * (self.action_space_high - self.action_space_low) + self.action_space_low
        return action


def init_env(n_envs=-1, env_id=None, seed=None):
    """
    n_envs == -1: only make 1 default environment without any wrappers
    n_envs == 1: make 1 default environment wrapped with DummyVecEnv
    n_env == c: make c default environment wrapped with SubprocVecEnv
    """

    def make_env():
        def _make_env():
            env = gym.make(env_id)
            return NormalizeActionSpaceWrapper(env)
        
        if n_envs == -1:
            return _make_env()
        else:
            return _make_env()
            # return _make_env()

    if n_envs == -1:
        return make_env()
    if n_envs == 1:
        return DummyVecEnv([make_env for _ in range(n_envs)])
    else:
        return SubprocVecEnv([make_env for _ in range(n_envs)])