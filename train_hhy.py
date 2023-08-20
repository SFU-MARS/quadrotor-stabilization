import os, sys
import numpy as np
import abc
import pybullet as pb
from pybullet_utils import bullet_client
import pybullet_data
from typing import Tuple
import gym
import time
from datetime import datetime
from gym.envs.registration import register
import torch
from phoenix_drone_simulation.algs.model import Model
from adversaryhover_phoenix import DroneHoverBulletEnvWithAdversary

# 1) set up algorithm, environment id
algo = 'ppo'
env_id = "DroneHoverBulletEnvWithAdversary-v0"
default_log_dir = f"./runs/phoenix"

if "Adversary" in env_id:
    assert env_id == 'DroneHoverBulletEnvWithAdversary-v0'
    register(id=env_id, entry_point="{}:{}".format(
        DroneHoverBulletEnvWithAdversary.__module__, 
        DroneHoverBulletEnvWithAdversary.__name__), 
        max_episode_steps=500,)

# 2) use 