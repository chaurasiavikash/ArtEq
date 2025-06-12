#!/usr/bin/env python3
"""
Workaround: Use the exact eval.py approach by creating a compatible data format
This bypasses all architecture issues by using the exact same code path
"""

import sys
import os
import numpy as np
import torch
import trimesh
import tempfile
import pickle
from pathlib import Path

# Add src directory to path
sys.path.append('src')

def create_fake_batch_for_eval(mesh_path, num_points=5000):
    """
    Create a fake batch that matches the format expected by eval.py
    This mimics the webdataset format used in the original evaluation
    """
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Sample points and normals (6D as expected by the model)
    points, face_indices = trimesh.sample.sample_surface_even(mesh, num_points)
    
    if len(points) < num_points:
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    # Ensure exact number of points
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
        face_indices = face_indices[indices]
    elif len(points) < num_points:
        extra_needed = num_points - len(points)
        extra_indices = np.random.choice(len(points), extra_needed, replace=True)
        points = np.vstack([points, points[extra_indices]])
        face_indices = np.hstack([face_indices, face_indices[extra_indices]])
    
    # Get normals
    face_normals = mesh.face_normals[face_indices]
    points_with_normals = np.hstack([points, face_normals])
    
    # Create fake labels (we don't need them for inference)
    fake_labels = np.zeros(num_points, dtype=np.int64)
    
    # Create fake SMPL parameters (dummy values, not used in inference)
    fake_betas = np.zeros(10, dtype=np.float32)
    fake_poses = np.zeros(72, dtype=np.float32)  # 24*3 axis-angle
    fake_trans = np.zeros(3, dtype=np.float32)
    
    # Create batch dictionary matching eval.py format
    batch = {
        'pcl_data': torch.from_numpy(points_with_normals.astype(np.float32)),
        'label_data': torch.from_numpy(fake_labels),
        'betas': torch.from_numpy(fake_betas),
        'poses': torch.from_numpy(fake_poses).reshape(24, 3),
        'trans': torch.from_numpy(fake_trans)
    }
    
    return batch

def run_arteq_eval_style(input_mesh, output_mesh, model_path, num_points=5000):
    """
    Run ArtEq using the exact same approach as eval.py
    """
    # Import everything needed from eval.py
    from geometry import get_body_model
    from models_pointcloud import PointCloud_network_equiv
    from train import SMPLX_layer, get_nc_and_view_channel, kinematic_layer_SO3_v2
    import smplx
    
    # Set up exactly like eval.py
    class Args:
        batch_size = 1
        latent_num = 128
        epoch = 15
        rep_type = "6d"
        part_num = 22
        num_point = num_points
        aug_type = "so3"
        gt_part_seg = "auto"
        EPN_input_radius = 0.4
        EPN_layer_num = 2
        kinematic_cond = "yes"
        i = None
        paper_model = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    args = Args()
    
    # Paper model settings (from eval.py lines 50-55)
    if args.paper_model:
        args.EPN_layer_num = 2
        args.EPN_input_radius = 0.4
        args.epoch = 15
        args.aug_type = "so3"
        args.kinematic_cond = "yes"
    
    device = args.device
    print(f"Using device: {device}")
    
    # Create models exactly like eval.py
    nc, _ = get_nc_and_view_channel(args)
    
    model = PointCloud_network_equiv(
        option=args, z_dim=args.latent_num, nc=nc, part_num=args.part_num
    ).to(device)
    
    # Load model exactly like eval.py
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create SMPL models exactly like eval.py  
    body_model = get_body_model(model_type="smpl", gender="male", batch_size=1, device=device)
    parents = body_model.parents[:22]
    body_model_gt = smplx.SMPL("./data/smpl_model/SMPL_MALE.pkl", batch_size=1, num_betas=10)
    
    # Create fake batch
    print(f"Loading and preprocessing {input_mesh}")
    batch = create_fake_batch_for_eval(input_mesh, num_points)
    
    # Run inference exactly like eval.py
    print("Running inference...")
    with torch.inference_mode():
        pcl_data = batch['pcl_data'][:num_points][None].to(device)  # Slice and add batch dim
        
        pred_joint, pred_pose, pred_shape, trans_feat = model(
            pcl_data, None, None, None, is_optimal_trans=False, parents=parents
        )
        
        pred_joint_pose = kinematic_layer_SO3_v2(pred_pose, parents)
        
        trans_feat = torch.zeros((1, 3)).to(device)
        
        pred_joints_pos, pred_vertices = SMPLX_layer(
            body_model, pred_shape, trans_feat, pred_joint_pose, rep="rotmat"
        )
    
    # Save result
    vertices = pred_vertices[0].cpu().numpy()
    faces = body_model.faces
    
    result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    result_mesh.export(output_mesh)
    
    print(f"\n✅ Success! SMPL mesh saved to {output_mesh}")
    print(f"📊 Shape parameters: {pred_shape[0][:5].cpu().numpy()}")
    print(f"📊 Vertices: {len(vertices)}, Faces: {len(faces)}")
    
    return vertices, faces

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mesh', help='Input mesh file')
    parser.add_argument('output_mesh', help='Output SMPL mesh file')
    parser.add_argument('--model_path', default='data/papermodel/model_epochs_00000014.pth')
    parser.add_argument('--num_points', type=int, default=5000)
    
    args = parser.parse_args()
    
    try:
        run_arteq_eval_style(args.input_mesh, args.output_mesh, args.model_path, args.num_points)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()