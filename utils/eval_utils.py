import json
import os
import time
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
import cv2
import evo
import numpy as np
import torch
from PIL import Image
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
matplotlib.use('Agg')  # 使用无头模式
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log

# Use evo to evaluate the alignment between the estimated trajectory and the Ground Truth trajectory, calculate ATE, and plot the trajectory
def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    # Check if we have enough poses for alignment
    if len(poses_gt) < 3 or len(poses_est) < 3:
        Log(f"Warning: Insufficient poses for trajectory evaluation. GT: {len(poses_gt)}, EST: {len(poses_est)}")
        Log("Need at least 3 poses for trajectory alignment. Returning dummy ATE value.")
        return 0.0
        
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )
    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    #data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    # plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))))
    plt.close(fig)  

    return ape_stat

# Evaluate the ATE of keyframes, and when evaluating, invert T to transform back to C2W
def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False, BA=False, validation_start_frame=-1):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    #latest_frame_idx = len(frames) if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    # Filter keyframes to only include those from validation_start_frame onwards
    eval_kf_ids = kf_ids
    if validation_start_frame >= 0:
        eval_kf_ids = [kf_id for kf_id in kf_ids if kf_id >= validation_start_frame]
        Log(f"ATE evaluation using {len(eval_kf_ids)} keyframes from frame {validation_start_frame} onwards")
        
        # Check if we have enough keyframes for meaningful evaluation
        if len(eval_kf_ids) < 3:
            Log(f"Warning: Only {len(eval_kf_ids)} keyframes available for validation. Need at least 3 for trajectory alignment.")
            Log("Returning dummy ATE value. Consider lowering validation_start_frame.")
            return 0.0

    for kf_id in eval_kf_ids:
    #for kf_id in range(latest_frame_idx):
        #print(kf_id)
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    if BA:
        label_evo = "after BA"
    elif final:
        label_evo = "final"
    else:
        label_evo = "{:04}".format(iterations)
    #label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate

# Evaluate the quality of rendered images, including PSNR, SSIM, and LPIPS, without evaluating keyframes
def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
    validation_start_frame=-1,  # New parameter to specify validation start frame
):
    interval = 1
    img_pred, img_gt, saved_frame_idx, img_residual = [], [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    
    # Determine the start index for evaluation
    start_idx = max(0, validation_start_frame) if validation_start_frame >= 0 else 0
    
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    # Directory for saving images
    viz_dir = os.path.join(save_dir, "viz")
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    #render_dir = os.path.join(save_dir, "render")
    #if not os.path.exists(render_dir):
    #    os.makedirs(render_dir)

    N = 0
    start_time1 = time.time()
    for idx in range(start_idx, end_idx, interval):
        if idx in kf_indices:
            continue
        N = N+1
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)
        #gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        #pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        # Compute error metrics
        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())
        
        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        residual = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
        residual = np.clip(residual, 0, 255).astype(np.uint8)
        img_pred.append(pred)
        img_gt.append(gt)
        img_residual.append(residual)
        # plot
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(gt)
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred)
        plt.title('Predicted')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(residual)
        plt.title('Residual')
        plt.axis('off')
        plt.figtext(0.5, 0.01, f"PSNR: {psnr_score.item():.2f}", ha="center", fontsize=12)
        save_path = os.path.join(viz_dir, f"{idx}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close() 

        # Save the rendered images
        #pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
        #    np.uint8
        #)
        #pred_image = Image.fromarray(pred)
        #save_path = os.path.join(render_dir, f"{idx}_pred.png")
        #pred_image.save(save_path, dpi=(300, 300))
        
    end_time1 = time.time()
    render_time = end_time1 - start_time1
    avg_render_time = render_time / N
    print("average render time is:", avg_render_time)
        
    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]} (evaluated from frame {start_idx})',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def save_trajectory_txt(frames, dataset, save_dir, filename="trajectory_all.txt"):
    """
    Save trajectory for all frames in the requested format:
    TIMESTAMP T00 T01 T02 T03 T10 T11 T12 T13 T20 T21 T22 T23 (row first for the matrix)
    
    Args:
        frames: Dictionary of frame indices to camera objects
        dataset: Dataset object to access color paths for timestamp extraction
        save_dir: Directory to save the trajectory file
        filename: Name of the output file
    """
    if save_dir is None:
        return
        
    mkdir_p(save_dir)
    output_path = os.path.join(save_dir, filename)
    
    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose
    
    def extract_timestamp_from_filename(filepath):
        """Extract timestamp from agricultural dataset filename"""
        import os
        basename = os.path.basename(filepath)
        # Remove extension and extract timestamp part like "1744202088-180835000"
        timestamp_str = basename.split('.')[0]
        
        try:
            if '-' in timestamp_str:
                parts = timestamp_str.split('-')
                seconds = int(parts[0])
                nanoseconds = int(parts[1])
                # Convert to decimal seconds with nanosecond precision
                return seconds + nanoseconds / 1e9
            else:
                return float(timestamp_str)
        except Exception:
            return None
    
    def denormalize_pose(pose, scene_center):
        """Convert normalized pose back to original scale"""
        denorm_pose = pose.copy()
        # Reverse the normalization: divide by scale factor and add back scene center
        denorm_pose[:3, 3] = pose[:3, 3] + scene_center
        return denorm_pose
    
    # Get normalization parameters from dataset if available
    scene_center = np.array([0, 0, 0])
    scale_factor = 1.0
    if dataset is not None and hasattr(dataset, 'scene_center') and hasattr(dataset, 'scale_factor'):
        scene_center = dataset.scene_center
        scale_factor = dataset.scale_factor
        Log(f"Using dataset normalization - center: {scene_center}, scale: {scale_factor}")
    else:
        Log("No normalization parameters found - saving poses as-is")
    
    # Sort frames by index for sequential output
    sorted_frame_indices = sorted(frames.keys())
    
    with open(output_path, 'w') as f:
        for frame_idx in sorted_frame_indices:
            frame = frames[frame_idx]
            
            # Use estimated pose (convert from W2C to C2W)
            pose_est = np.linalg.inv(gen_pose_matrix(frame.R, frame.T))
            
            # Denormalize the pose to original scale
            pose_est = denormalize_pose(pose_est, dataset.scene_center)
            
            # Extract timestamp from filename if available in the dataset
            timestamp = frame_idx  # Default fallback
            if hasattr(dataset, 'color_paths') and frame_idx < len(dataset.color_paths):
                filepath = dataset.color_paths[frame_idx]
                extracted_timestamp = extract_timestamp_from_filename(filepath)
                if extracted_timestamp is not None:
                    timestamp = extracted_timestamp
            
            # Extract transformation matrix components (row-wise)
            T = pose_est
            line = f"{timestamp:.6f} {T[0,0]:.6f} {T[0,1]:.6f} {T[0,2]:.6f} {T[0,3]:.6f} {T[1,0]:.6f} {T[1,1]:.6f} {T[1,2]:.6f} {T[1,3]:.6f} {T[2,0]:.6f} {T[2,1]:.6f} {T[2,2]:.6f} {T[2,3]:.6f}\n"
            f.write(line)
    
    Log(f"Trajectory saved to {output_path} with {len(sorted_frame_indices)} frames")

def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
