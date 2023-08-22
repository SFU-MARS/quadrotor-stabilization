import time
from gym.envs.registration import register

from stable_baselines3 import PPO, SAC 
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from adversaryhover_phoenix import DroneHoverBulletEnvWithAdversary
from utility import init_env



def train_with_sb3(env_id='DroneHoverBulletEnvWithAdversary-v0', 
                   alg='PPO', 
                   n_envs=-1,
                   timesteps=100000, 
                   episode=100, 
                   max_steps=500, 
                   save_freq=20000, 
                   eval_freq=1000, 
                   seed=2023):
    """Hanyang

    """
    # Choose and register the environment 
    assert env_id == 'DroneHoverBulletEnvWithAdversary-v0'
    register(id=env_id, entry_point="{}:{}".format(
        DroneHoverBulletEnvWithAdversary.__module__, 
        DroneHoverBulletEnvWithAdversary.__name__), max_episode_steps=max_steps)
    print(f"The {env_id} environment has been registered!")

    # Setup checkpoints and log directory
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, 
                                             save_path=f'sb3_train_logs/{env_id}/', 
                                             name_prefix=alg, 
                                             save_replay_buffer=False, 
                                             save_vecnormalize=False)
    train_callback = CallbackList([checkpoint_callback])

    # initilize environments, choose the number of training env 
    env = init_env(n_envs=n_envs, env_id=env_id)

    # Setup sb3 algorithm model
    # TODO: future could tune more hyperparameters
    if alg == 'PPO':
        model = PPO(policy='MlpPolicy', env=env, verbose=1, tensorboard_log=f"sb3_tensorboard/{env_id}/{alg}")
    elif alg == 'SAC':
        model = SAC(policy='MlpPolicy', env=env, verbose=1, tensorboard_log=f"sb3_tensorboard/{env_id}/{alg}")
    else:
        print("Please check out the algorithm name again.")
    start_time = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=train_callback)
    model.save(f"sb3_models/{env_id}/{alg}_{timesteps}")
    duration = time.perf_counter() - start_time
    print(100*"=")
    print(f"The training is finished and it costs {duration} seconds.")


if __name__ == "__main__":
    train_with_sb3(alg='SAC', timesteps=500000)