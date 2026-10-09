#!/bin/bash
# Isaac Lab version of scripts/nutbolt_teacher.sh (same arguments)
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

PYTHON=${PYTHON:-$HOME/env_isaaclab6/bin/python}

CUDA_VISIBLE_DEVICES=${GPUS} \
${PYTHON} train_lab.py task=XHandHoraNutBolt headless=True seed=${SEED} \
task.env.object.type=screw_trinut \
experiment=rl \
train.algo=PPO \
task.env.reset_dist_threshold=0.07 \
wandb_activate=False \
train.ppo.output_name=XHandHoraNutBolt_teacher/${CACHE} \
${EXTRA_ARGS}
