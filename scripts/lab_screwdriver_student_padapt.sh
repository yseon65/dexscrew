#!/bin/bash
# Isaac Lab version of scripts/screwdriver_student_padapt.sh (same arguments).
# Pass the teacher checkpoint as an extra arg, e.g.
#   scripts/lab_screwdriver_student_padapt.sh 0 42 run1 checkpoint=outputs/XHandHoraScrewDriver_teacher/run1/stage1_nn/last.pth
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

PYTHON=${PYTHON:-$HOME/env_isaaclab6/bin/python}

CUDA_VISIBLE_DEVICES=${GPUS} \
${PYTHON} train_lab.py task=XHandHoraScrewDriver headless=True seed=${SEED} \
train.algo=ProprioAdapt \
train.ppo.proprio_adapt=True train.ppo.output_name=XHandHoraScrewDriver_student_padapt/${CACHE} \
experiment=student_sim \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=48 \
wandb_activate=False \
task.env.reset_dist_threshold=0.15 \
checkpoint=last.pth \
${EXTRA_ARGS}
