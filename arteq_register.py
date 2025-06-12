#!/usr/bin/env python3
"""
Simple ArtEq Registration Script - Uses eval.py approach directly
"""

import sys
import os
import numpy as np
import torch
import trimesh
from pathlib import Path

# Add src directory to path
sys.path.append('src')

# Import from the existing eval.py and train.py
from geometry import get_body_model
from models_pointcloud import PointCloud_network_equiv
from train import SMPLX_layer, get_nc_and_view_channel, kinematic_layer_SO3_v2

def preprocess_point_cloud(mesh_path, num_points=5000):
    """
    Load and preprocess point cloud to match ArtEq training format
    """
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Sample points from surface
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    
    # If not enough points, use random sampling
    if len(points) < num_points:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    # Ensure exact number of points
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        extra_needed = num_points - len(points)
        extra_indices = np.random.choice(len(points), extra_needed, replace=True)
        extra_points = points[extra_indices]
        points = np.vstack([points, extra_points])
    
    # Convert to tensor and add batch dimension
    points = torch.from_numpy(points.astype(np.float32))
    
    # Add small random noise as done in training (from train.py)
    points_sigma = 0.001
    points += points_sigma * torch.randn(points.shape[0], 3)
    
    # Add batch dimension
    points = points.unsqueeze(0)  # Shape: (1, num_points, 3)
    
    return points

def create_model_from_eval():
    """
    Create model using exact same configuration as eval.py
    """
    # Exact configuration from eval.py
    class Args:
        batch_size = 1
        latent_num = 128
        epoch = 15
        rep_type = "6d"
        part_num = 22
        num_point = 5000
        aug_type = "so3"
        gt_part_seg = "auto"
        EPN_input_radius = 0.4
        EPN_layer_num = 2
        kinematic_cond = "yes"
        i = None
        paper_model = True  # Use paper model settings
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    args = Args()
    
    # Override for paper model (from eval.py lines 50-55)
    if args.paper_model:
        args.EPN_layer_num = 2
        args.EPN_input_radius = 0.4
        args.epoch = 15
        args.aug_type = "so3"
        args.kinematic_cond = "yes"
    
    # Get network configuration
    nc, _ = get_nc_and_view_channel(args)
    
    # Create model exactly as in eval.py
    model = PointCloud_network_equiv(
        option=args,
        z_dim=args.latent_num,
        nc=nc,
        part_num=args.part_num
    ).to(args.device)
    
    return model, args

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mesh', help='Input mesh file (.ply, .obj, etc.)')
    parser.add_argument('output_mesh', help='Output SMPL mesh file')
    parser.add_argument('--model_path', default='data/papermodel/model_epochs_00000014.pth')
    parser.add_argument('--num_points', type=int, default=5000)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Step 1: Load and preprocess point cloud
        print(f"Loading point cloud from {args.input_mesh}")
        pcl_data = preprocess_point_cloud(args.input_mesh, args.num_points)
        pcl_data = pcl_data.to(device)
        print(f"Point cloud shape: {pcl_data.shape}")
        
        # Step 2: Create model with exact eval.py configuration
        print("Creating model...")
        model, model_args = create_model_from_eval()
        
        # Step 3: Load pretrained weights
        print(f"Loading model weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Step 4: Create SMPL body model
        body_model = get_body_model(model_type="smpl", gender="male", batch_size=1, device=device)
        parents = body_model.parents[:22]
        
        # Step 5: Run inference (exactly as in eval.py)
        print("Running inference...")
        with torch.inference_mode():
            pred_joint, pred_pose, pred_shape, trans_feat = model(
                pcl_data, None, None, None, 
                is_optimal_trans=False, parents=parents
            )
            
            pred_joint_pose = kinematic_layer_SO3_v2(pred_pose, parents)
            
            trans_feat = torch.zeros((1, 3)).to(device)
            
            pred_joints_pos, pred_vertices = SMPLX_layer(
                body_model,
                pred_shape,
                trans_feat,
                pred_joint_pose,
                rep="rotmat",
            )
        
        # Step 6: Save result
        vertices = pred_vertices[0].cpu().numpy()
        faces = body_model.faces
        
        # Create and save mesh
        result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        result_mesh.export(args.output_mesh)
        
        print(f"\n✅ Success! SMPL mesh saved to {args.output_mesh}")
        print(f"📊 Shape parameters (first 5): {pred_shape[0][:5].cpu().numpy()}")
        print(f"📊 Vertices: {len(vertices)}, Faces: {len(faces)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()