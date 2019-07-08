#!/usr/bin/env bash
BASEDIR=$(dirname "$0")

#python3.5 $BASEDIR/train.py --reward_type=ttr --load_path=/local-scratch/xlv/pre_maml/log/QuadFallingDownEnv-v0_ttr_ppo_27-Jun-2019_23-54-19/model  --load_iter=6

python3.5 $BASEDIR/train.py --reward_type=ttr