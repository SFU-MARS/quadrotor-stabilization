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

# output_dir = data_dir/env-id/algo/YY-MM-DD_HH-MM-SS/seed[seed]
# fpath = 'torch_save'
# fpath = osp.join(self.log_dir, fpath)
# fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
# fname = osp.join(fpath, fname)
# relative path: runs/phoenix/DroneHoverBulletEnvWithAdversary-v0/ppo/2023-07-17__20-39-02/seed_02390/torch_save/model.pt
# 1) set up algorithm, environment id and the trained model
algo = 'ppo'
env_id = "DroneHoverBulletEnvWithAdversary-v0"
default_log_dir = f"./runs/phoenix"

if "Adversary" in env_id:
    assert env_id == 'DroneHoverBulletEnvWithAdversary-v0'
    register(id=env_id, entry_point="{}:{}".format(
        DroneHoverBulletEnvWithAdversary.__module__, 
        DroneHoverBulletEnvWithAdversary.__name__), 
        max_episode_steps=500,)
    
model = Model(
    alg=algo,  # choose between: trpo, ppo
    env_id=env_id,
    log_dir=default_log_dir,
    init_seed=2390,
)
model = torch.load("runs/phoenix/DroneHoverBulletEnvWithAdversary-v0/ppo/2023-07-17__20-39-02/seed_02390/torch_save/model.pt")

# 4) visualize trained PPO model
env = gym.make(env_id)
# Important note: PyBullet necessitates to call env.render()
# before env.reset() to display the GUI!
env.render() 
while True:
    obs = env.reset()
    done = False
    while not done:
        obs = torch.as_tensor(obs, dtype=torch.float32)
        action, value, *_ = model.actor_critic(obs)
        obs, reward, done, info = env.step(action)

        time.sleep(0.05)
        if done:
            obs = env.reset()