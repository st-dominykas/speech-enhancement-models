#!/bin/bash
#SBATCH --job-name=dtln
#SBATCH --output=slurm-out_%j.out
#SBATCH --error=slurm-err_%j.err
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --time=60:0:0
singularity exec --nv python.sif python3 "train.py"
