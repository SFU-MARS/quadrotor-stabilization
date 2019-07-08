#!/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python

import numpy as np
import os, sys
import gym
from gym_foo import gym_foo

import ppo

import json
from time import *


# import spinup
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='QuadFallingDownEnv-v0')
    parser.add_argument('--reward_type', type=str, default='ttr')
    parser.add_argument('--algo', type=str, default='ppo')

    parser.add_argument('--load_path', type=str, default='None')
    parser.add_argument('--load_iter', type=int, default=0)
    args = parser.parse_args()
    
    SAVE_PATH = os.path.join(os.environ['PROJ_HOME_2'] + '/log/' + args.env + '_' + args.reward_type + '_' + args.algo + '_' + strftime('%d-%b-%Y_%H-%M-%S'), 'model')
    print("saving path:", SAVE_PATH)

    kwargs = {'reward_type':args.reward_type, 'algo':args.algo, 'env':args.env}
    print(kwargs)
    
    env = gym.make(kwargs['env'], **kwargs)
    params = os.environ['PROJ_HOME_2']+'/ppo_params.json'
    if kwargs['algo'] == 'ppo':
        ppo.create_session()
        init_policy = ppo.create_policy('pi', env)
        ppo.initialize()

        if args.load_path != 'None' and args.load_iter != 0:
            pi = init_policy
            if os.path.exists(args.load_path):
                pi.load_model(args.load_path, iteration=args.load_iter)
                print("loading from %s" %args.load_path)
                print("loading iter %d" %args.load_iter)
            else:
                raise ValueError('Model loading path do not exist!!')
        else:
            pi = init_policy

        with open(params) as params_file:
            d = json.load(params_file)
            num_iters = d.get('num_iters')
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
            
        i = 0
        while i < num_iters:
            print('overall training iteration %d' %i)
            pi, ep_mean_length, ep_mean_reward, suc_percent = ppo.ppo_learn(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,
                                                                                    clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
                                                                                    optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                                                                                    gamma=gamma, lam=lam, max_iters=max_iters, schedule='constant')
           
            pi.save_model(SAVE_PATH, iteration=i)
            i += 1
            
        env.close()
        
    else:
        raise ValueError("no such algorithm supported!!")

       
    
    # spinup.ppo(lambda:gym.make(args.env, **kwargs))
    # ob = env.reset()
    # re = env.step([0.1,0.1,0.1,0.1])

