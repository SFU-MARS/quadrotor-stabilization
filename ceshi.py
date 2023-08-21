import gym
import time
import phoenix_drone_simulation
from gym.envs.registration import register
from adversaryhover_phoenix import DroneHoverBulletEnvWithAdversary


env = gym.make('DroneHoverBulletEnv-v0')
print(env.action_space)


env_id = 'DroneHoverBulletEnvWithAdversary-v0'
# register(id=env_id, entry_point="{}:{}".format(DroneHoverBulletEnvWithAdversary.__module__, DroneHoverBulletEnvWithAdversary.__name__), max_episode_steps=500,)
# while True:
#     done = False
#     env.render()  # make GUI of PyBullet appear
#     x = env.reset()
#     while not done:
#         random_action = env.action_space.sample()
#         x, reward, done, info = env.step(random_action)
#         time.sleep(0.05)