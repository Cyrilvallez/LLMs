#!/bin/bash

#SBATCH --job-name=cybersec-evaluate
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:0
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

# python3 -u cyber_sec_evaluate_wrapper.py "$@"
python3 -u cyber_sec_evaluate.py "$@"

conda deactivate
