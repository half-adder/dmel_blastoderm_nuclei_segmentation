#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH -t 02-00:00:00
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --job-name=jupyter

module load anaconda
conda activate stardist-env

jupyter lab --no-browser --ip='0.0.0.0' --port=8888