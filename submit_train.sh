#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH -t 02-00:00:00
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --job-name=stardist_train

# Load any necessary modules
module load anaconda

# Activate virtual environment if needed
conda activate stardist-env

# Run the training script
python train.py