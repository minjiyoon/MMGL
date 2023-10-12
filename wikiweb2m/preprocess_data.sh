#!/usr/bin/env bash
#SBATCH --partition=russ_reserved
#SBATCH --job-name=MMHG
#SBATCH --output=slurm_logs/train-mmhg-%j.out
#SBATCH --error=slurm_logs/train-mmhg-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --mem=250gb
#SBATCH --exclude matrix-1-20,matrix-1-18,matrix-0-34 

ulimit -c unlimited
module load cuda-11.1.1

export PYTHONPATH=.

python wikiweb2m/preprocess_data.py
