import gym
import time
import phoenix_drone_simulation

env = gym.make('DroneHoverBulletEnv-v0')
print(env.action_space)

# while True:
#     done = False
#     env.render()  # make GUI of PyBullet appear
#     x = env.reset()
#     while not done:
#         random_action = env.action_space.sample()
#         x, reward, done, info = env.step(random_action)
#         time.sleep(0.05)