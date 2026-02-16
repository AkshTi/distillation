#!/bin/bash
#SBATCH --job-name=distill_kd
#SBATCH --output=results/slurm_%j_distill.out
#SBATCH --error=results/slurm_%j_distill.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/orcd/home/002/akshatat/VisionResearch

mkdir -p results models

# Aliases to check latest output/error:
#   alias slout='cat $(ls -t results/slurm_*_distill.out | head -1)'
#   alias slerr='cat $(ls -t results/slurm_*_distill.err | head -1)'
#   alias sltail='tail -f $(ls -t results/slurm_*_distill.out | head -1)'

python distill.py --epochs 20 --batch-size 128 --num-workers 8
