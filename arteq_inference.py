#!/usr/bin/env python3
"""
Modified ArtEq eval.py for custom mesh inference
Based on the original eval.py but adapted for single mesh processing
"""

import argparse
import os
import numpy as np
import torch
import trimesh
import sys
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from geometry import get_body_model
from models_pointcloud import PointCloud_network_equiv
from train import SMPLX_layer, get_nc_and_view_channel, kinematic_layer_SO3_v2

def sample_points_from_mesh(mesh, num_points=5000):
    """
    Sample points from mesh surface (same as your original function)
    """
    print(f"Sampling {num_points} points from mesh...")
    
    # Sample points uniformly from mesh surface
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    
    # If we don't get enough points, use random sampling
    if len(points) < num_points:
        print(f"Even sampling gave {len(points)} points, using random sampling...")
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    # Ensure we have exactly num_points
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        # Pad with random points from existing ones
        extra_needed = num_points - len(points)
        extra_indices = np.random.choice(len(points), extra_needed, replace=True)
        extra_points = points[extra_indices]
        points = np.vstack([points, extra_points])
    
    return points.astype(np.float32)

def load_mesh(filepath):
    """
    Load mesh from various formats (same as your original function)
    """
    print(f"Loading mesh from {filepath}...")
    
    try:
        mesh = trimesh.load(filepath)
        
        # Handle multi-mesh files (take the largest mesh)
        if isinstance(mesh, trimesh.Scene):
            print("Multiple meshes found, taking the largest one...")
            mesh = max(mesh.geometry.values(), key=lambda x: len(x.vertices))
        
        print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Basic mesh validation
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh has no vertices")
        if len(mesh.faces) == 0:
            raise ValueError("Mesh has no faces")
            
        return mesh
        
    except Exception as e:
        print(f"Error loading mesh: {e}")
        print("Supported formats: PLY, OBJ, STL, OFF, 3MF, GLTF, etc.")
        sys.exit(1)

def preprocess_pointcloud(points):
    """
    Preprocess point cloud for ArtEq (simplified version)
    """
    # Convert to tensor and add batch dimension
    points = torch.from_numpy(points).float()
    points = points.unsqueeze(0)  # Shape: (1, N, 3)
    
    # Center the point cloud
    center = points.mean(dim=1, keepdim=True)
    points = points - center
    
    # Normalize to unit sphere
    max_dist = torch.norm(points, dim=2).max()
    if max_dist > 0:
        points = points / max_dist
    
    print(f"Preprocessed point cloud: center={center.squeeze().numpy()}, scale={max_dist.item():.3f}")
    
    return points

def save_mesh(vertices, faces, filepath, joints=None):
    """
    Save mesh to file (same as your original function)
    """
    print(f"Saving mesh to {filepath}...")
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Add joint positions as sphere markers (optional)
    if joints is not None:
        joint_spheres = []
        for i, joint in enumerate(joints):
            sphere = trimesh.creation.icosphere(radius=0.02)
            sphere.vertices += joint
            joint_spheres.append(sphere)
        
        # Create scene with mesh and joints
        scene = trimesh.Scene([mesh] + joint_spheres)
        scene.export(filepath)
    else:
        mesh.export(filepath)
    
    print(f"Mesh saved successfully!")

def run_custom_inference(input_mesh_path, output_mesh_path, args):
    """
    Run inference on a custom mesh using the eval.py model loading approach
    """
    print(f"Processing: {input_mesh_path} -> {output_mesh_path}")
    
    # Set up device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load input mesh
    input_mesh = load_mesh(input_mesh_path)
    
    # Sample points from mesh
    points = sample_points_from_mesh(input_mesh, args.num_point)
    
    # Preprocess point cloud
    pcl_data = preprocess_pointcloud(points).to(args.device)
    
    # Load body model (same as eval.py)
    body_model = get_body_model(model_type="smpl", gender="male", batch_size=1, device=args.device)
    parents = body_model.parents[:22]
    
    # Get network configuration (same as eval.py)
    nc, _ = get_nc_and_view_channel(args)
    
    # Create model (same as eval.py)
    model = PointCloud_network_equiv(
        option=args, 
        z_dim=args.latent_num, 
        nc=nc, 
        part_num=args.part_num
    ).to(args.device)
    
    # Determine model path
    if args.paper_model:
        model_path = "./data/papermodel/model_epochs_00000014.pth"
        print("Using paper model")
    else:
        # Use the same path construction as eval.py
        exps_folder = "gt_part_seg_{}_EPN_layer_{}_radius_{}_aug_{}_kc_{}".format(
            args.gt_part_seg,
            args.EPN_layer_num,
            args.EPN_input_radius,
            args.aug_type,
            args.kinematic_cond,
        )
        if args.num_point != 5000:
            exps_folder = exps_folder + f"_num_point_{args.num_point}"
        if args.i is not None:
            exps_folder = exps_folder + f"_{args.i}"
        
        output_folder = os.path.sep.join(["./experiments", exps_folder])
        model_path = os.path.join(output_folder, f"model_epochs_{args.epoch-1:08d}.pth")
    
    print(f"Loading model from: {model_path}")
    
    # Load model weights (same as eval.py)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    # Run inference (same as eval.py)
    with torch.inference_mode():
        pred_joint, pred_pose, pred_shape, trans_feat = model(
            pcl_data, None, None, None, 
            is_optimal_trans=False, parents=parents
        )
        
        pred_joint_pose = kinematic_layer_SO3_v2(pred_pose, parents)
        
        trans_feat = torch.zeros((1, 3)).to(args.device)
        
        pred_joints_pos, pred_vertices = SMPLX_layer(
            body_model, pred_shape, trans_feat, pred_joint_pose, rep="rotmat"
        )
    
    # Extract results
    pred_vertices = pred_vertices[0].cpu().numpy()
    pred_joints_pos = pred_joints_pos[0].cpu().numpy()
    
    print("✅ Inference completed!")
    
    # Save output mesh
    save_mesh(pred_vertices, body_model.faces, output_mesh_path, 
              joints=pred_joints_pos if args.save_joints else None)
    
    print(f"📊 Input: {len(input_mesh.vertices)} vertices → Output: {len(pred_vertices)} vertices")
    return pred_vertices, pred_joints_pos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArtEq Custom Mesh Inference (based on eval.py)')
    parser.add_argument('input_mesh', help='Input mesh file (PLY, OBJ, STL, etc.)')
    parser.add_argument('output_mesh', help='Output SMPL mesh file')
    
    # Model configuration (same defaults as eval.py)
    parser.add_argument('--latent_num', type=int, default=128, help='Latent dimension')
    parser.add_argument('--epoch', type=int, default=15, help='Which model epoch to use')
    parser.add_argument('--rep_type', type=str, default="6d", help='Rotation representation')
    parser.add_argument('--part_num', type=int, default=22, help='Number of body parts')
    parser.add_argument('--num_point', type=int, default=5000, help='Number of points to sample')
    parser.add_argument('--aug_type', type=str, default="so3", help='Augmentation type')
    parser.add_argument('--gt_part_seg', type=str, default="auto", help='Part segmentation method')
    parser.add_argument('--EPN_input_radius', type=float, default=0.4, help='EPN input radius')
    parser.add_argument('--EPN_layer_num', type=int, default=2, help='Number of EPN layers')
    parser.add_argument('--kinematic_cond', type=str, default="yes", help='Use kinematic conditioning')
    parser.add_argument('--i', type=int, default=None, help='Experiment ID')
    parser.add_argument('--paper_model', action='store_true', help='Use the paper model')
    parser.add_argument('--save_joints', action='store_true', help='Save joint positions as spheres')
    
    args = parser.parse_args()
    
    try:
        # Run inference
        pred_vertices, pred_joints = run_custom_inference(
            args.input_mesh, args.output_mesh, args
        )
        
        print(f"\n✅ Success! Registered SMPL mesh saved to {args.output_mesh}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)