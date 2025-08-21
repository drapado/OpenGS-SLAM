# Validation Mode Feature

This document describes the validation mode feature in OpenGS-SLAM, which allows you to specify a frame number from which the Gaussian Splatting model stops training and enters validation-only mode.

## Overview

The validation mode feature enables you to:

1. **Train the GS model** from the first frame up to a specified frame number
2. **Stop training** at the specified frame (validation_start_frame)
3. **Continue SLAM** from that frame onwards without updating the GS model
4. **Evaluate metrics** only on the validation frames (from validation_start_frame to the end)

This is useful for:
- Creating train/validation splits in long sequences
- Evaluating generalization performance of the learned GS model
- Testing robustness of the learned representation

## Configuration

### Method 1: Configuration File

Add the following parameter to your config YAML file under the `Training` section:

```yaml
Training:
  # ... other parameters ...
  validation_start_frame: 100  # Frame number to start validation mode (-1 to disable)
```

### Method 2: Command Line Argument

Use the `--validation_start_frame` argument when running SLAM:

```bash
python slam.py --config configs/mono/agri/base_config.yaml --eval --validation_start_frame 100
```

## How It Works

### Frontend Modifications

1. **Detection**: The frontend monitors the current frame index and detects when `validation_start_frame` is reached
2. **Mode Switch**: Sets `validation_mode = True` when the threshold is reached
3. **Message Passing**: Includes validation mode flag in messages sent to the backend

### Backend Modifications  

1. **Gaussian Addition**: Skips adding new Gaussians to the scene when in validation mode
2. **Mapping Optimization**: Skips the mapping optimization iterations when in validation mode
3. **Visibility Tracking**: Still tracks frame visibility for consistency

### Evaluation Modifications

1. **ATE Evaluation**: Only evaluates trajectory error on keyframes from `validation_start_frame` onwards
2. **Rendering Evaluation**: Only evaluates PSNR, SSIM, LPIPS on frames from `validation_start_frame` onwards
3. **Logging**: Clearly indicates which frames are used for evaluation

## Example Usage

### Basic Usage

```bash
# Train GS model up to frame 50, then validation-only from frame 50 onwards
python slam.py --config configs/mono/agri/base_config.yaml --eval --validation_start_frame 50
```

### Advanced Usage

```bash
# Use with color refinement and custom validation split
python slam.py --config configs/mono/agri/base_config.yaml --eval --color --validation_start_frame 75
```

### Test Script

Use the provided test script to validate the feature:

```bash
python test_validation_mode.py --config configs/mono/agri/base_config.yaml --validation_start_frame 50
```

## Expected Output

When validation mode is activated, you should see log messages like:

```
Entering validation mode at frame 100
Validation mode: Skipping Gaussian addition for frame 100
Validation mode: Skipping mapping optimization for frame 100
ATE evaluation using 25 keyframes from frame 100 onwards
mean psnr: 28.45, ssim: 0.89, lpips: 0.12 (evaluated from frame 100)
```

## Parameter Guidelines

- **Small sequences (< 200 frames)**: Use 60-80% for training (e.g., validation_start_frame = 120 for 200 frames)
- **Medium sequences (200-500 frames)**: Use 70-80% for training  
- **Large sequences (> 500 frames)**: Use 80-90% for training
- **Disable feature**: Set `validation_start_frame = -1`

## Technical Details

### Modified Files

1. **configs/mono/agri/base_config.yaml**: Added `validation_start_frame` parameter
2. **slam.py**: Added command line argument and parameter passing
3. **utils/slam_frontend.py**: Added validation mode detection and message passing
4. **utils/slam_backend.py**: Added validation mode handling in keyframe processing
5. **utils/eval_utils.py**: Modified evaluation functions to support validation frame filtering

### Message Protocol Changes

- **Keyframe messages**: Now include validation mode flag as 11th element
- **Init messages**: Now include validation mode flag as 9th element

### Performance Impact

- **Training phase**: No performance impact
- **Validation phase**: Slight performance improvement due to skipped optimization iterations
- **Memory usage**: Constant after validation_start_frame (no new Gaussians added)

## Troubleshooting

### Common Issues

1. **Invalid frame number**: Ensure `validation_start_frame` is less than total sequence length
2. **Too early validation**: Setting validation_start_frame too low may result in poor GS model
3. **Config conflicts**: Command line argument overrides config file parameter

### Debugging

Enable verbose logging to see validation mode transitions:

```bash
python slam.py --config configs/mono/agri/base_config.yaml --eval --validation_start_frame 50 --verbose
```
