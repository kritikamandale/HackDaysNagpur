#!/bin/bash
# setup_env.sh — Mac / Linux environment setup
# Usage: bash setup_env.sh

echo "Creating conda environment 'seg'..."
conda create -n seg python=3.10 -y
conda activate seg

echo "Installing PyTorch (CUDA 11.8 — adjust for your CUDA version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Done! Activate with: conda activate seg"
echo "  Then run: python explore_data.py"
