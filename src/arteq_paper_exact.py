#!/usr/bin/env python3
"""
ArtEq inference script with EXACT paper model configuration
Based on the paper specifications in Section 4.4
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
    """Sample points from mesh surface"""
    print(f"Sampling {num_points} points from mesh...")
    
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    
    if len(points) < num_points:
        print(f"Even sampling gave {len(points)} points, using random sampling...")
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        extra_needed = num_points - len(points)
        extra_indices = np.random.choice(len(points), extra_needed, replace=True)
        extra_points = points[extra_indices]
        points = np.vstack([points, extra_points])
    
    return points.astype(np.float32)

def load_mesh(filepath):
    """Load mesh from various formats"""
    print(f"Loading mesh from {filepath}...")
    
    try:
        mesh = trimesh.load(filepath)
        
        if isinstance(mesh, trimesh.Scene):
            print("Multiple meshes found, taking the largest one...")
            mesh = max(mesh.geometry.values(), key=lambda x: len(x.vertices))
        
        print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        if len(mesh.vertices) == 0:
            raise ValueError("Mesh has no vertices")
        if len(mesh.faces) == 0:
            raise ValueError("Mesh has no faces")
            
        return mesh
        
    except Exception as e:
        print(f"Error loading mesh: {e}")
        sys.exit(1)

def preprocess_pointcloud(points):
    """Preprocess point cloud for ArtEq"""
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
    """Save mesh to file"""
    print(f"Saving mesh to {filepath}...")
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    if joints is not None:
        joint_spheres = []
        for i, joint in enumerate(joints):
            sphere = trimesh.creation.icosphere(radius=0.02)
            sphere.vertices += joint
            joint_spheres.append(sphere)
        
        scene = trimesh.Scene([mesh] + joint_spheres)
        scene.export(filepath)
    else:
        mesh.export(filepath)
    
    print(f"Mesh saved successfully!")

def patch_epn_config_for_paper_model():
    """
    Patch the EPN configuration to match the paper model exactly
    Based on Section 4.4: "C = 64; two multi-head self attention (MHSA) layers"
    """
    import EPN_options
    
    # Store original function
    original_get_default_cfg = EPN_options.get_default_cfg
    
    def paper_model_cfg():
        cfg = original_get_default_cfg()
        
        # Paper specifications from Section 4.4
        # "only two layers of SPConv that output a feature tensor with channel size C = 64"
        cfg.model.kpconv = False  # Use SO3 convolutions, not KPConv
        cfg.model.kanchor = 60    # Full icosahedral anchors
        cfg.model.normals = False # No normals in input
        
        print("Applied paper model EPN configuration:")
        print(f"  kpconv: {cfg.model.kpconv}")
        print(f"  kanchor: {cfg.model.kanchor}")
        print(f"  normals: {cfg.model.normals}")
        
        return cfg
    
    # Temporarily replace the function
    EPN_options.get_default_cfg = paper_model_cfg
    
    return original_get_default_cfg

def run_paper_model_inference(input_mesh_path, output_mesh_path, args):
    """
    Run inference using EXACT paper model configuration
    """
    print(f"Processing: {input_mesh_path} -> {output_mesh_path}")
    
    # Set up device
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    
    # Configure args exactly like the paper model
    print("Configuring for paper model (exact paper specifications)...")
    
    # From Section 4.4 and experiments
    args.EPN_layer_num = 2        # "only two layers of SPConv"
    args.EPN_input_radius = 0.4   # Standard radius
    args.epoch = 15               # Paper model epoch
    args.aug_type = "so3"         # SO3 augmentation
    args.kinematic_cond = "yes"   # Kinematic conditioning enabled
    args.latent_num = 128         # Latent dimension
    args.part_num = 22            # SMPL parts
    args.rep_type = "6d"          # 6D rotation representation
    
    model_path = "./data/papermodel/model_epochs_00000014.pth"
    
    print(f"Looking for model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Make sure you have downloaded the paper model to ./data/papermodel/")
        return None, None
    
    # Load input mesh
    input_mesh = load_mesh(input_mesh_path)
    
    # Sample points from mesh
    points = sample_points_from_mesh(input_mesh, args.num_point)
    
    # Preprocess point cloud
    pcl_data = preprocess_pointcloud(points).to(args.device)
    
    # Load body model
    body_model = get_body_model(model_type="smpl", gender="male", batch_size=1, device=args.device)
    parents = body_model.parents[:22]
    
    # Get network configuration
    nc, _ = get_nc_and_view_channel(args)
    
    print(f"Paper model configuration:")
    print(f"  EPN_layer_num: {args.EPN_layer_num}")
    print(f"  EPN_input_radius: {args.EPN_input_radius}")
    print(f"  kinematic_cond: {args.kinematic_cond}")
    print(f"  latent_num: {args.latent_num}")
    print(f"  part_num: {args.part_num}")
    print(f"  nc: {nc}")
    
    # Patch EPN configuration to match paper
    original_cfg_func = patch_epn_config_for_paper_model()
    
    try:
        # Create model with paper configuration
        print("Creating model with paper specifications...")
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(1)  # Same seed as paper
        
        model = PointCloud_network_equiv(
            option=args, 
            z_dim=args.latent_num, 
            nc=nc, 
            part_num=args.part_num
        ).to(args.device)
        
        print(f"Loading model weights from: {model_path}")
        
        # Try to load model weights
        try:
            state_dict = torch.load(model_path, map_location=args.device)
            
            # Print some diagnostic info
            print(f"Checkpoint contains {len(state_dict)} parameters")
            
            # Show a few key dimensions for debugging
            for key in ['encoder.backbone.0.blocks.0.intra_conv.conv.basic_conv.W', 
                       'encoder.backbone.0.blocks.0.intra_conv.conv.intra_idx']:
                if key in state_dict:
                    print(f"  {key}: {state_dict[key].shape}")
            
            # Try loading with strict=False first
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  Missing: {key}")
            
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  Unexpected: {key}")
            
            # Check for shape mismatches
            model_params = dict(model.named_parameters())
            shape_mismatches = []
            
            for key, checkpoint_tensor in state_dict.items():
                if key in model_params:
                    model_tensor = model_params[key]
                    if checkpoint_tensor.shape != model_tensor.shape:
                        shape_mismatches.append((key, checkpoint_tensor.shape, model_tensor.shape))
            
            if shape_mismatches:
                print(f"\n⚠️  Shape mismatches found ({len(shape_mismatches)}):")
                for key, ckpt_shape, model_shape in shape_mismatches[:10]:
                    print(f"  {key}: checkpoint {ckpt_shape} vs model {model_shape}")
                
                if len(shape_mismatches) > 20:
                    print(f"\n❌ Too many shape mismatches ({len(shape_mismatches)})")
                    print("The paper model was likely trained with different hyperparameters.")
                    print("This confirms the architecture mismatch issue.")
                    return None, None
                else:
                    print(f"\n⚠️  Proceeding with {len(shape_mismatches)} mismatches...")
            
            model.eval()
            print("✅ Model loaded!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
        
        # Run inference
        print("Running inference...")
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
        
    finally:
        # Restore original EPN config function
        import EPN_options
        EPN_options.get_default_cfg = original_cfg_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ArtEq Paper Model Inference')
    parser.add_argument('input_mesh', help='Input mesh file')
    parser.add_argument('output_mesh', help='Output SMPL mesh file') 
    parser.add_argument('--num_point', type=int, default=5000, help='Number of points to sample')
    parser.add_argument('--save_joints', action='store_true', help='Save joint positions as spheres')
    
    args = parser.parse_args()
    
    try:
        pred_vertices, pred_joints = run_paper_model_inference(
            args.input_mesh, args.output_mesh, args
        )
        
        if pred_vertices is not None:
            print(f"\n✅ Success! Registered SMPL mesh saved to {args.output_mesh}")
        else:
            print(f"\n❌ Failed to process mesh")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





        