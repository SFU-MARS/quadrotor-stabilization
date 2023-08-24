import gym
import time
import torch
from gym.envs.registration import register
from stable_baselines3 import PPO, SAC 
from adversaryhover_phoenix import DroneHoverBulletEnvWithAdversary


def test_with_sb3(env_id='DroneHoverBulletEnvWithAdversary-v0', alg='PPO', timesteps=100000):
    # register and initilaize the environment
    assert env_id == 'DroneHoverBulletEnvWithAdversary-v0'
    register(id=env_id, entry_point="{}:{}".format(
            DroneHoverBulletEnvWithAdversary.__module__, 
            DroneHoverBulletEnvWithAdversary.__name__), max_episode_steps=500)
    env = gym.make(env_id)

    # Setup sb3 algorithm model and load the trained model
    # TODO: future could tune more hyperparameters
    model_path = f"sb3_models/{env_id}/{alg}_{timesteps}"
    if alg == 'PPO':
        model = PPO(policy='MlpPolicy', env=env, verbose=1, tensorboard_log=f"sb3_tensorboard/{env_id}/{alg}")
        model = PPO.load(model_path)
    elif alg == 'SAC':
        model = SAC(policy='MlpPolicy', env=env, verbose=1, tensorboard_log=f"sb3_tensorboard/{env_id}/{alg}")
        model = SAC.load(model_path)
    else:
        print("Please check out the algorithm name again.")
    print("The model has been loaded successfully!")
    
    # test
    env.render() 
    while True:
        obs = env.reset()
        done = False
        while not done:
            obs = torch.as_tensor(obs, dtype=torch.float32)
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            time.sleep(0.05)
            if done:
                obs = env.reset()


if __name__ == "__main__":
    test_with_sb3(alg='SAC', timesteps=5000000)
