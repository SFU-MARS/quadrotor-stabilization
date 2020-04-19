import sys
import gym
from gym_foo import gym_foo
from gym import wrappers
from time import *

from ppo1 import ppo
from utils.plotting_performance import *
from baselines import logger

import argparse
from utils.utils import *
import json
import pickle
import copy

import tensorflow as tf


def run(env, algorithm, args, params=None, load=False, loadpath=None, loaditer=None, save_obs=False):

    assert algorithm == ppo
    assert args['gym_env'] in ["QuadTakeOffHoverEnv-v0"]

    # Initialize policy
    ppo.create_session()
    init_policy = ppo.create_policy('pi', env)
    ppo.initialize()

    # load trained policy
    if load and loadpath is not None and loaditer is not None:
        pi = init_policy
        pi.load_model(loadpath, iteration=loaditer)
        pi.save_model(args['MODEL_DIR'], iteration=0)
    else:
        # init policy
        pi = init_policy
        pi.save_model(args['MODEL_DIR'], iteration=0)

    # init params
    with open(params) as params_file:
        d = json.load(params_file)
        num_ppo_iters = d.get('num_ppo_iters')
        timesteps_per_actorbatch = d.get('timesteps_per_actorbatch')
        clip_param = d.get('clip_param')
        entcoeff = d.get('entcoeff')
        optim_epochs = d.get('optim_epochs')
        optim_stepsize = d.get('optim_stepsize')
        optim_batchsize = d.get('optim_batchsize')
        gamma = d.get('gamma')
        lam = d.get('lam')
        max_iters = num_ppo_iters

    if not load:
        pi = algorithm.ppo_learn(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,
                                 clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
                                 optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                                 gamma=gamma, lam=lam,  args=args, max_iters=max_iters, schedule='constant', save_obs=save_obs)
    else:
        if args['eval'] == 'yes':
            pi = algorithm.ppo_eval(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,max_iters=5)
        else:
            logger.log("we continue to train from a loaded model ...")
            pi = algorithm.ppo_learn(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,
                                     clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
                                     optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                                     gamma=gamma, lam=lam, args=args, max_iters=max_iters, schedule='constant',
                                     save_obs=save_obs)

    env.close()
    return pi


if __name__ == "__main__":
    with tf.device('/gpu:1'):
        # --- path setting ---
        parser = argparse.ArgumentParser()
        parser.add_argument("--gym_env", help="which gym environment to use.", type=str, default='QuadTakeOffHoverEnv-v0')
        parser.add_argument("--algo", help="which algorithm to use.", type=str,
                            default='ppo')
        parser.add_argument("--rew_type", help="which reward to use.", type=str,
                            default='ttr')
        parser.add_argument("--eval", help="To do policy evaluation or not", type=str, default='no')
        args = parser.parse_args()
        args = vars(args)

        # --- logger initialize and configuration ---
        RUN_DIR = MODEL_DIR = FIGURE_DIR = RESULT_DIR = None
        if args['algo'] == "ppo":
            RUN_DIR = os.path.join(os.getcwd(), 'runs_log_tests', strftime('%d-%b-%Y_%H-%M-%S') + args['gym_env'] + '_' + args['algo'])
            MODEL_DIR = os.path.join(RUN_DIR, 'model')
            FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
            RESULT_DIR = os.path.join(RUN_DIR, 'result')
        else:
            raise ValueError("unknown algorithm!!")
        args['RUN_DIR'] = RUN_DIR
        args['MODEL_DIR'] = MODEL_DIR
        args['FIGURE_DIR'] = FIGURE_DIR
        args['RESULT_DIR'] = RESULT_DIR

        logger.configure(dir=args['RUN_DIR'])
        logger.record_tabular("algo", args['algo'])
        logger.record_tabular("env", args['gym_env'])
        logger.dump_tabular()

        # --- start to train RL agent ---
        if args['algo'] == "ppo":
            # Make necessary directories
            maybe_mkdir(args['RUN_DIR'])
            maybe_mkdir(args['MODEL_DIR'])
            maybe_mkdir(args['FIGURE_DIR'])
            maybe_mkdir(args['RESULT_DIR'])
            ppo_params_json = './ppo1/ppo_params.json'

            # --- Start to train the policy from scratch ---
            # env = gym.make(args['gym_env'], rew=args['rew_type'])
            # trained_policy = run(env=env, algorithm=ppo, params=ppo_params_json, args=args)
            # trained_policy.save_model(args['MODEL_DIR'])

            # --- Load pre-trained model for evaluation ---
            # LOAD_DIR = os.environ['PROJ_HOME_3'] + '/runs_log_tests/grad_norm_0.5_kl_0.015_std_0.5_baseline/27-Jan-2020_01-44-06DubinsCarEnv-v0_hand_craft_ppo/model'
            # LOAD_DIR =  os.environ['PROJ_HOME_3'] + '/runs_log_tests/grad_norm_0.5_kl_0.015_std_0.5_fixed_value_vi/23-Jan-2020_00-13-24DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/model'
            # eval_policy = run(env=env, algorithm=ppo, params=ppo_params_json, load=True, loadpath=LOAD_DIR, loaditer=180, args=args)

            # --- Load pre-trained model and continue training ---
            env = gym.make(args['gym_env'], rew=args['rew_type'])
            LOAD_DIR = '/local-scratch/xlv/quad_stabilization' + '/runs_log_tests/18-Apr-2020_00-22-43QuadTakeOffHoverEnv-v0_ppo/model'
            trained_policy = run(env=env, algorithm=ppo, params=ppo_params_json, load=True, loadpath=LOAD_DIR,
                              loaditer=420, args=args)
            trained_policy.save_model(args['MODEL_DIR'])
        else:
            raise ValueError("arg algorithm is invalid!")















