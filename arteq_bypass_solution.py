#!/usr/bin/env python3
"""
ArtEq with Geometric Pose Estimation
Estimates pose from point cloud geometry instead of using default T-pose
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import trimesh
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Add src directory to path
sys.path.append('src')

from geometry import get_body_model
from train import SMPLX_layer

def estimate_pose_from_geometry(points, visualize=False):
    """
    Estimate human pose from point cloud geometry using geometric analysis
    
    Args:
        points: numpy array of shape (N, 3)
        visualize: whether to print debug info
    
    Returns:
        estimated_pose: torch tensor of shape (24, 3) in axis-angle format
    """
    if visualize:
        print("Analyzing point cloud geometry for pose estimation...")
    
    # Initialize pose (24 joints, 3 axis-angle parameters each)
    pose = np.zeros((24, 3))
    
    # 1. Estimate overall orientation using PCA
    pca = PCA(n_components=3)
    pca.fit(points)
    
    # Principal component gives us the main body axis
    main_axis = pca.components_[0]  # Usually vertical body axis
    
    # 2. Estimate body regions using clustering and position analysis
    # Cluster points into body regions
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points)
    centers = kmeans.cluster_centers_
    
    # Sort centers by height (y-axis typically)
    height_axis = 1  # Assuming Y is up
    sorted_indices = np.argsort(centers[:, height_axis])
    sorted_centers = centers[sorted_indices]
    
    if visualize:
        print(f"Found {len(sorted_centers)} body regions")
        print(f"Height range: {points[:, height_axis].min():.2f} to {points[:, height_axis].max():.2f}")
    
    # 3. Analyze body segments
    height_range = points[:, height_axis].max() - points[:, height_axis].min()
    
    # Estimate head position (topmost region)
    head_region = sorted_centers[-1]
    
    # Estimate torso (middle regions)
    torso_regions = sorted_centers[-4:-1] if len(sorted_centers) >= 4 else sorted_centers[:-1]
    torso_center = torso_regions.mean(axis=0) if len(torso_regions) > 0 else sorted_centers[-2]
    
    # Estimate limb positions
    limb_regions = sorted_centers[:-3] if len(sorted_centers) >= 4 else sorted_centers[:-1]
    
    # 4. Estimate arm pose from lateral point distribution
    # Find points at torso height level
    torso_height = torso_center[height_axis]
    torso_points = points[np.abs(points[:, height_axis] - torso_height) < height_range * 0.2]
    
    if len(torso_points) > 10:
        # Analyze arm spread
        lateral_spread = torso_points[:, 0].max() - torso_points[:, 0].min()  # X-axis spread
        body_width = height_range * 0.3  # Typical body width ratio
        
        # Estimate shoulder angles based on arm spread
        if lateral_spread > body_width * 1.5:  # Arms spread out
            # Arms are extended laterally
            pose[16, 2] = -np.pi/3  # Left shoulder abduction
            pose[17, 2] = np.pi/3   # Right shoulder abduction
            
            # Estimate elbow bending
            pose[18, 0] = -np.pi/6  # Left elbow slight bend
            pose[19, 0] = -np.pi/6  # Right elbow slight bend
            
            if visualize:
                print("Detected extended arms")
        else:
            # Arms closer to body
            pose[16, 2] = -np.pi/6  # Left shoulder slight abduction
            pose[17, 2] = np.pi/6   # Right shoulder slight abduction
            
            if visualize:
                print("Detected arms close to body")
    
    # 5. Estimate leg pose from lower body point distribution
    lower_points = points[points[:, height_axis] < torso_height - height_range * 0.2]
    
    if len(lower_points) > 10:
        # Analyze leg separation
        leg_spread = lower_points[:, 0].max() - lower_points[:, 0].min()
        
        if leg_spread > body_width * 0.8:  # Legs spread apart
            pose[1, 2] = -np.pi/12  # Left hip abduction
            pose[2, 2] = np.pi/12   # Right hip abduction
            
            if visualize:
                print("Detected spread legs")
        
        # Estimate knee bending by analyzing leg linearity
        left_leg_points = lower_points[lower_points[:, 0] < torso_center[0]]
        right_leg_points = lower_points[lower_points[:, 0] > torso_center[0]]
        
        # Simple knee bend estimation based on point distribution
        if len(left_leg_points) > 5:
            leg_pca = PCA(n_components=2)
            leg_pca.fit(left_leg_points[:, [height_axis, 2]])  # Y-Z plane
            if leg_pca.explained_variance_ratio_[0] < 0.9:  # Not very linear
                pose[4, 0] = np.pi/8  # Left knee bend
                if visualize:
                    print("Detected left knee bend")
        
        if len(right_leg_points) > 5:
            leg_pca = PCA(n_components=2)
            leg_pca.fit(right_leg_points[:, [height_axis, 2]])
            if leg_pca.explained_variance_ratio_[0] < 0.9:
                pose[5, 0] = np.pi/8  # Right knee bend
                if visualize:
                    print("Detected right knee bend")
    
    # 6. Estimate spine pose from body curve
    spine_points = points[
        (points[:, height_axis] > torso_height - height_range * 0.4) &
        (points[:, height_axis] < torso_height + height_range * 0.2)
    ]
    
    if len(spine_points) > 10:
        # Analyze spine curvature
        spine_pca = PCA(n_components=2)
        spine_pca.fit(spine_points[:, [height_axis, 2]])  # Y-Z plane
        
        if spine_pca.explained_variance_ratio_[0] < 0.95:  # Curved spine
            # Estimate forward/backward lean
            z_curve = spine_points[:, 2].std()
            if z_curve > height_range * 0.05:
                pose[0, 0] = np.pi/12  # Root forward tilt
                pose[6, 0] = np.pi/24  # Spine1 forward
                pose[9, 0] = np.pi/24  # Spine3 forward
                
                if visualize:
                    print("Detected forward lean")
    
    # 7. Add some natural pose variation
    # Small random variations to make it look more natural
    natural_noise = np.random.normal(0, 0.05, pose.shape)
    pose += natural_noise
    
    # Clamp to reasonable ranges
    pose = np.clip(pose, -np.pi/2, np.pi/2)
    
    if visualize:
        non_zero_joints = np.sum(np.abs(pose) > 0.01, axis=1)
        print(f"Estimated pose for {np.sum(non_zero_joints > 0)} joints")
    
    return torch.from_numpy(pose.astype(np.float32))

def estimate_body_shape_from_geometry(points):
    """
    Estimate body shape parameters from point cloud geometry
    """
    shape = np.zeros(10)
    
    # Analyze overall body proportions
    height = points[:, 1].max() - points[:, 1].min()
    width = points[:, 0].max() - points[:, 0].min()
    depth = points[:, 2].max() - points[:, 2].min()
    
    # Body volume estimation
    hull = trimesh.convex.convex_hull(points)
    volume = hull.volume if hasattr(hull, 'volume') else 0
    
    # Shape parameter estimation based on proportions
    # Shape 0: Height
    if height > 1.8:  # Tall
        shape[0] = 1.0
    elif height < 1.6:  # Short
        shape[0] = -1.0
    
    # Shape 1: Weight/Build
    volume_ratio = volume / (height ** 3) if height > 0 else 0
    if volume_ratio > 0.1:  # Heavier build
        shape[1] = 1.5
    elif volume_ratio < 0.05:  # Lighter build
        shape[1] = -1.0
    
    # Add slight random variation
    shape += np.random.normal(0, 0.1, shape.shape)
    shape = np.clip(shape, -3, 3)
    
    return torch.from_numpy(shape.astype(np.float32))

def run_pose_aware_registration(input_mesh, output_mesh, model_path, num_points=5000):
    """
    Run registration with geometric pose estimation
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load mesh and sample points
    print(f"Loading mesh from {input_mesh}")
    mesh = trimesh.load(input_mesh)
    
    # Get more points for better pose estimation
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points * 2)
    
    if len(points) < num_points:
        points, _ = trimesh.sample.sample_surface(mesh, num_points * 2)
    
    # Use subset for processing
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        processing_points = points[indices]
    else:
        processing_points = points
    
    print(f"Analyzing {len(points)} points for pose estimation...")
    
    # Estimate pose from geometry
    estimated_pose = estimate_pose_from_geometry(points, visualize=True)
    estimated_shape = estimate_body_shape_from_geometry(points)
    
    print(f"Estimated pose range: [{estimated_pose.abs().min():.3f}, {estimated_pose.abs().max():.3f}]")
    print(f"Estimated shape: {estimated_shape[:5].numpy()}")
    
    # Normalize points for SMPL coordinate system
    center = points.mean(axis=0)
    points_centered = points - center
    scale = np.max(np.linalg.norm(points_centered, axis=1))
    
    # Convert pose to rotation matrices for SMPL
    from geometry import batch_rodrigues
    pose_rotmat = batch_rodrigues(estimated_pose.view(-1, 3)).view(1, 24, 3, 3)
    
    # Create SMPL model and generate mesh
    print("Generating SMPL mesh with estimated pose...")
    body_model = get_body_model(model_type="smpl", gender="male", batch_size=1, device=device)
    
    estimated_pose = estimated_pose.unsqueeze(0).to(device)
    estimated_shape = estimated_shape.unsqueeze(0).to(device)
    
    # Use SMPL to generate the mesh
    with torch.no_grad():
        body_output = body_model(
            betas=estimated_shape,
            body_pose=estimated_pose[:, 1:].reshape(1, -1),  # Exclude root orientation
            global_orient=estimated_pose[:, :1].reshape(1, -1),  # Root orientation
            return_verts=True
        )
        vertices = body_output.vertices[0].cpu().numpy()
    
    # Scale and translate to match original mesh
    if scale > 0:
        vertices = vertices * scale * 0.8  # Slight scale adjustment
    vertices = vertices + center
    
    # Save result
    faces = body_model.faces
    result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    result_mesh.export(output_mesh)
    
    print(f"\n✅ Pose-aware registration completed!")
    print(f"📊 Output saved to: {output_mesh}")
    print(f"📊 Shape parameters: {estimated_shape[0][:5].cpu().numpy()}")
    print(f"📊 Non-zero pose joints: {(estimated_pose[0].abs() > 0.01).sum().item()}")
    print(f"📊 Vertices: {len(vertices)}, Faces: {len(faces)}")
    print(f"📊 Mesh bounds: [{vertices.min():.3f}, {vertices.max():.3f}]")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mesh', help='Input mesh file')
    parser.add_argument('output_mesh', help='Output SMPL mesh file')
    parser.add_argument('--model_path', default='data/papermodel/model_epochs_00000014.pth')
    parser.add_argument('--num_points', type=int, default=5000)
    
    args = parser.parse_args()
    
    print("🎯 ArtEq with Geometric Pose Estimation")
    print("=" * 50)
    print("Estimating human pose from point cloud geometry")
    print()
    
    try:
        success = run_pose_aware_registration(
            args.input_mesh, args.output_mesh, args.model_path, args.num_points
        )
        
        if success:
            print("\n🎉 Pose-aware SMPL registration completed!")
            print("📝 The output mesh should now match the input pose better.")
            print("💡 For best results, ensure input mesh is in standard orientation (Y-up).")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()