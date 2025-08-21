#!/usr/bin/env python3
"""
Test script to demonstrate the validation mode feature in OpenGS-SLAM.

This script shows how to use the validation_start_frame parameter to:
1. Train the Gaussian Splatting model up to a certain frame
2. Then run validation-only mode (no GS training) from that frame onwards
3. Evaluate metrics only on the validation frames

Usage:
    python test_validation_mode.py --config configs/mono/agri/base_config.yaml --validation_start_frame 100
"""

import subprocess
import sys
import argparse
import yaml
import os

def run_slam_with_validation(config_path, validation_start_frame):
    """Run SLAM with validation start frame."""
    cmd = [
        sys.executable, "slam.py",
        "--config", config_path,
        "--eval",
        "--validation_start_frame", str(validation_start_frame)
    ]
    
    print(f"Running SLAM with validation starting at frame {validation_start_frame}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SLAM completed successfully!")
        print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"SLAM failed with return code {e.returncode}")
        print("STDERR:", e.stderr[-1000:])  # Last 1000 chars
        return False

def check_config(config_path):
    """Check if config file exists and is valid."""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Config loaded successfully from {config_path}")
        return True
    except Exception as e:
        print(f"Error loading config: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test validation mode feature")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to config file")
    parser.add_argument("--validation_start_frame", type=int, default=50,
                      help="Frame number to start validation mode")
    
    args = parser.parse_args()
    
    print("OpenGS-SLAM Validation Mode Test")
    print("=" * 40)
    print(f"Config: {args.config}")
    print(f"Validation start frame: {args.validation_start_frame}")
    print("=" * 40)
    
    # Check config file
    if not check_config(args.config):
        sys.exit(1)
    
    # Run SLAM with validation mode
    success = run_slam_with_validation(args.config, args.validation_start_frame)
    
    if success:
        print("\nTest completed successfully!")
        print(f"The system trained GS model up to frame {args.validation_start_frame}")
        print(f"Then ran validation-only mode from frame {args.validation_start_frame} onwards")
        print("Check the output logs for metrics calculated only on validation frames.")
    else:
        print("\nTest failed!")
        sys.exit(1)
