#!/bin/bash
#SBATCH -J distillation/mock
#SBATCH -p mit_normal_gpu
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G h200:1
#SBATCH --chdir=/home/akshatat/distillation/mock
#SBATCH -o /home/akshatat/distillation/mock/logs/%x_%j.out
#SBATCH -e /home/akshatat/distillation/mock/logs/%x_%j.err

mkdir -p /home/akshatat/distillation/mock/logs

# source activate distill

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

python exclusion_experiment.py

echo "End time: $(date)"
