#!/usr/bin/env bash
# Run once to create the virtual environment and install dependencies.
# Usage: bash setup.sh

set -e

module load python/3.11.11
module load cuda/12.8

VENV_DIR="$HOME/venvs/mnist"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing PyTorch (CUDA 12.8) ..."
pip install --upgrade pip --quiet
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet

echo "Done. Virtual environment ready at $VENV_DIR"
