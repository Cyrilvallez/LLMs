#!/bin/bash

#SBATCH --job-name=webapp
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/vacy/LLMs

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

# Launch frpc server - note that the '&' is essential to run the command in a non-blocking way
# srun --exclusive --exact --ntasks=1 --cpus-per-task=1 --mem=1G ../frp_server/frp_0.54.0_linux_amd64/frpc -c ../frp_server/frpc/frpc_experiment.toml &
../frp_server/frp_0.54.0_linux_amd64/frpc -c ../frp_server/frpc/frpc_experiment.toml &
# Launch app
python3 -u webapp_chat.py "$@"

conda deactivate
