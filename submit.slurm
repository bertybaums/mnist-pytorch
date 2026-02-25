#!/usr/bin/env bash
#SBATCH --job-name=mnist-cnn
#SBATCH --partition=gpu-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%j_mnist.out
#SBATCH --error=logs/%j_mnist.err

source /etc/profile
set -e

module load python/3.11.11
module load cuda/12.8

source "$HOME/venvs/mnist/bin/activate"

mkdir -p logs

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started:  $(date)"
echo

cd "$SLURM_SUBMIT_DIR"
python3 train.py

echo
echo "Finished: $(date)"
