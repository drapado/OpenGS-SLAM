import csv
import glob
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


class AgriParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        # Load images from cam_2 - adjust path structure
        # input_folder should be like "datasets/agri/train/cam_2" 
        # and we want to load from "datasets/agri/train/cam_2/zed_multi/cam_2/rgb"
        rgb_path = f"{self.input_folder}/zed_multi/cam_2/rgb/*.jpg"
        self.color_paths = sorted(glob.glob(rgb_path))
        self.n_img = len(self.color_paths)
        print(f"Found {self.n_img} images in {rgb_path}")
        
        # Load poses from groundtruth_cam_2.csv in the same directory
        csv_path = f"{self.input_folder}/groundtruth_cam_2.csv"
        self.load_poses(csv_path)

    def timestamp_from_filename(self, filename):
        """Extract timestamp from filename like '1744202088-180835000.jpg'"""
        basename = os.path.basename(filename)
        timestamp_str = basename.split('.')[0]  # Remove .jpg extension
        return timestamp_str

    def timestamp_to_ns(self, timestamp_str):
        """Convert timestamp string to nanoseconds integer"""
        # Handle both formats: "1744202088-180835000" and "1744202088-000000000"
        parts = timestamp_str.split('-')
        if len(parts) == 2:
            # Convert to single integer: seconds * 1e9 + nanoseconds
            seconds = int(parts[0])
            nanoseconds = int(parts[1])
            return seconds * 1000000000 + nanoseconds
        else:
            return int(timestamp_str)

    def load_poses(self, csv_path):
        """Load poses from groundtruth CSV file"""
        print(f"Loading poses from {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} pose entries from CSV")
        
        self.poses = []
        self.frames = []
        
        # Create a dictionary for fast timestamp lookup
        pose_dict = {}
        for _, row in df.iterrows():
            timestamp = str(row['timestamp'])
            # Create pose matrix from translation and quaternion
            tx, ty, tz = row['tx'], row['ty'], row['tz']
            qx, qy, qz, qw = row['qx'], row['qy'], row['qz'], row['qw']
            
            # Create rotation matrix from quaternion (x, y, z, w)
            rotation = R.from_quat([qx, qy, qz, qw])
            rotation_matrix = rotation.as_matrix()
            
            # Create 4x4 transformation matrix
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = [tx, ty, tz]
            
            pose_dict[timestamp] = pose

        # Match images with poses
        matched_pairs = []
        for color_path in self.color_paths:
            img_timestamp_str = self.timestamp_from_filename(color_path)
            img_timestamp_ns = self.timestamp_to_ns(img_timestamp_str)
            
            # Find the closest timestamp in pose data
            closest_timestamp = min(pose_dict.keys(), 
                                  key=lambda x: abs(self.timestamp_to_ns(x) - img_timestamp_ns))
            
            # Check if the match is reasonable (within 100ms = 100,000,000 ns)
            time_diff = abs(self.timestamp_to_ns(closest_timestamp) - img_timestamp_ns)
            if time_diff < 100000000:  # 100ms tolerance
                matched_pairs.append((color_path, pose_dict[closest_timestamp]))
            else:
                print(f"Warning: No close pose found for image {color_path}, time diff: {time_diff}ns")

        print(f"Successfully matched {len(matched_pairs)} image-pose pairs")

        # Store matched data with normalization
        if matched_pairs:
            # Calculate scene center and scale for normalization
            all_translations = np.array([pair[1][:3, 3] for pair in matched_pairs])
            scene_center = np.mean(all_translations, axis=0)
            
            # Calculate scale - use max distance from center, but cap it at reasonable value
            distances = np.linalg.norm(all_translations - scene_center, axis=1)
            max_distance = np.max(distances)
            
            # Normalize to fit within a sphere of radius 5 (typical for NeRF/GS)
            target_radius = 5.0
            scale_factor = target_radius / max_distance if max_distance > 0 else 1.0
            
            print("Scene normalization:")
            print(f"  Original center: [{scene_center[0]:.2f}, {scene_center[1]:.2f}, {scene_center[2]:.2f}]")
            print(f"  Max distance: {max_distance:.2f}")
            print(f"  Scale factor: {scale_factor:.6f}")
            
            # Store normalization parameters
            self.scene_center = scene_center
            self.scale_factor = scale_factor
        
        for color_path, pose in matched_pairs:
            # Normalize the pose
            normalized_pose = pose.copy()
            # Translate to center, then scale
            normalized_pose[:3, 3] = (pose[:3, 3] - scene_center) * scale_factor
            
            # Convert to camera coordinate system (invert the pose)
            inv_pose = np.linalg.inv(normalized_pose)
            self.poses.append(inv_pose)
            
            frame = {
                "file_path": color_path,
                "depth_path": None,  # No depth for RGB-only
                "transform_matrix": normalized_pose.tolist(),
            }
            self.frames.append(frame)
        
        # Update number of images to matched pairs
        self.n_img = len(matched_pairs)
        self.color_paths = [pair[0] for pair in matched_pairs]
        print(f"Final dataset size: {self.n_img} images")
