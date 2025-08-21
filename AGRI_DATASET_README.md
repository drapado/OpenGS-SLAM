# Agricultural SLAM Dataset Setup

This document describes how to run OpenGS-SLAM on your agricultural dataset using only cam_2.

## Dataset Structure

Your agricultural dataset is organized as follows:
```
agri-slam-data/
├── train/
│   ├── groundtruth_cam_2.csv      # Camera poses for cam_2
│   └── zed_multi/
│       └── cam_2/
│           └── rgb/               # RGB images from cam_2
└── val/
    ├── groundtruth_cam_2.csv      # Camera poses for cam_2
    └── zed_multi/
        └── cam_2/
            └── rgb/               # RGB images from cam_2
```

The system creates a symlink structure for compatibility:
```
datasets/
└── agri/
    └── train/
        └── cam_2/ -> /workspaces/OpenGS-SLAM/agri-slam-data/train
```

## Configuration Files

The following configuration files have been created:

- `configs/mono/agri/base_config.yaml` - Base configuration for agricultural dataset
- `configs/mono/agri/train.yaml` - Training dataset configuration
- `configs/mono/agri/val.yaml` - Validation dataset configuration

## Camera Parameters

The current configuration uses estimated camera parameters for a ZED camera at 1920x1200 resolution:

```yaml
fx: 1080.0  # Focal length in pixels
fy: 1080.0  # Focal length in pixels  
cx: 960.0   # Principal point x (image center)
cy: 600.0   # Principal point y (image center)
width: 1920
height: 1200
distorted: False
```

**Important**: These are estimated values. For better accuracy, you should calibrate your camera and update these parameters in the configuration files.

## Dataset Features

- **Images**: 132 RGB images from cam_2 in training set
- **Format**: 1920x1200 JPG images
- **Poses**: Ground truth poses from `groundtruth_cam_2.csv`
- **Coordinate System**: Real-world coordinates automatically normalized to SLAM-friendly scale
- **Temporal Matching**: Images and poses are matched using timestamps with 100ms tolerance
- **Path Structure**: Uses symlinks for compatibility with OpenGS-SLAM path expectations
- **Sky Removal**: Optional sky masking using depth_anything masks (depth value 0 = sky pixels)

## Running SLAM

### Option 1: Using the run script
```bash
./run_agri.sh
```

### Option 2: Manual execution
```bash
# Activate environment
source miniconda3/bin/activate opengs-slam

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspaces/OpenGS-SLAM/miniconda3/envs/opengs-slam/lib/python3.11/site-packages/torch/lib/

# Run SLAM
python slam.py --config configs/mono/agri/train.yaml
```

## Results

Results will be saved to:
- `results/` directory with SLAM outputs
- Trajectory files for evaluation
- Rendered images if `eval_rendering: True`

## Customization

### Camera Calibration
To improve accuracy, calibrate your camera and update the parameters in `configs/mono/agri/train.yaml`:

```yaml
Dataset:
  Calibration:
    fx: YOUR_FX
    fy: YOUR_FY
    cx: YOUR_CX
    cy: YOUR_CY
    k1: YOUR_K1  # Distortion coefficients
    k2: YOUR_K2
    p1: YOUR_P1
    p2: YOUR_P2
    k3: YOUR_K3
    distorted: True  # Set to True if you have distortion
```

### Training Parameters
You can adjust SLAM parameters in the configuration files:

- `init_itr_num`: Number of initialization iterations
- `tracking_itr_num`: Tracking iterations per frame
- `mapping_itr_num`: Mapping iterations per frame
- `kf_interval`: Keyframe interval
- `gaussian_th`: Gaussian threshold for updates

### Sky Removal
Sky removal is enabled by default and uses depth_anything masks:

```yaml
Dataset:
  use_sky_removal: True  # Enable sky removal
  sky_depth_threshold: 0  # Pixels with depth value 0 are sky pixels
```

The system automatically finds corresponding depth masks in the `depth_anything` folder and masks out sky regions (depth value 0) by setting them to black pixels.

### GUI Visualization
To enable real-time visualization:
```yaml
Results:
  use_gui: True
```

## Troubleshooting

1. **Memory Issues**: Reduce `pcd_downsample` values in configuration
2. **Poor Results**: Check camera calibration parameters
3. **Timestamp Mismatch**: Verify timestamp format in CSV and image filenames
4. **CUDA Errors**: Ensure proper CUDA environment setup

## File Modifications

The following files were created/modified for agricultural dataset support:

1. `utils/agri_parser.py` - Agricultural dataset parser
2. `utils/dataset.py` - Added AgriDataset class
3. `configs/mono/agri/` - Configuration files
4. `run_agri.sh` - Execution script
