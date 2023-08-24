import gym
import time
import phoenix_drone_simulation
from gym.envs.registration import register
from adversaryhover_phoenix import DroneHoverBulletEnvWithAdversary
from stable_baselines3.common.env_checker import check_env

# env = gym.make('DroneHoverBulletEnv-v0')
# print(env.action_space)


env_id = 'DroneHoverBulletEnvWithAdversary-v0'
# register(id=env_id, entry_point="{}:{}".format(DroneHoverBulletEnvWithAdversary.__module__, DroneHoverBulletEnvWithAdversary.__name__), max_episode_steps=500)
register(id=env_id, entry_point="{}:{}".format(
        DroneHoverBulletEnvWithAdversary.__module__, 
        DroneHoverBulletEnvWithAdversary.__name__), max_episode_steps=500)
env = gym.make('DroneHoverBulletEnvWithAdversary-v0')
# env.seed(seed=2022)
# print(50*"=")
check_env(env)

# print(env.observation_space)
# print(env.action_space)
# print((env.drone.rpy))
# done = False
# for i in range(10):
#     # done = False
#     env.render()  # make GUI of PyBullet appear
#     x = env.reset()
#     while not done:
#         random_action = env.action_space.sample()
#         x, reward, done, info = env.step(random_action)
#         print(reward)
#         time.sleep(0.05)