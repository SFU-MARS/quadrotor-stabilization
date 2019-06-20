#!/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python

import numpy as np
import os, sys
import gym
from gym_foo import gym_foo

if __name__ == "__main__":
    
    env_name = 'QuadFallingDownEnv-v0'
    env = gym.make(env_name)
    ob = env.reset()
    re = env.step([0.1,0.1,0.1,0.1])

