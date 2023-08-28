import gym
import torch
from gym.wrappers import Monitor
import time
import phoenix_drone_simulation
from gym.envs.registration import register
# from phoenix-drone-simulation/phoenix_drone_simulation/envs/hover.py import DroneHoverBulletEnvWithAdversary
from stable_baselines3.common.env_checker import check_env
from datetime import datetime
import numpy as np


env_id='DroneHoverBulletEnvWithAdversary-v0'
env = gym.make(env_id)

video_dir = '/localhome/hha160/projects/quadrotor-stabilization/test_videos'

env = Monitor(env, video_dir, force=True)
obs = env.reset()
# while True:
#         obs = env.reset()
#         done = False
#         while not done:
#             obs = torch.as_tensor(obs, dtype=torch.float32)
#             action, _ = model.predict(obs)
#             obs, reward, done, info = env.step(action)
#             # env.render(mode="rgb_array")

#             time.sleep(0.05)
#             if done:
#                 obs = env.reset()