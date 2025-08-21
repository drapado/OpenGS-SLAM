#!/bin/bash

# Run OpenGS-SLAM on Agricultural Dataset (cam_2 only)
# Make sure to activate the conda environment first

echo "Running OpenGS-SLAM on Agricultural Dataset (cam_2)"
echo "=============================================="

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "opengs-slam" ]; then
    echo "Activating conda environment..."
    source miniconda3/bin/activate opengs-slam
fi

# Set CUDA device (modify if you have multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

# Set library path for CUDA libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspaces/OpenGS-SLAM/miniconda3/envs/opengs-slam/lib/python3.11/site-packages/torch/lib/

echo "Starting SLAM on training data..."
python slam.py --config configs/mono/agri/train_val.yaml --validation_start_frame 130

echo "SLAM completed!"
