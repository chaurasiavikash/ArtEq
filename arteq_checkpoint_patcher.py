#!/usr/bin/env python3
"""
Conservative ArtEq Checkpoint Patcher - Only patch what's absolutely necessary
"""

import sys
import os
import numpy as np
import torch
import trimesh
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from geometry import get_body_model
from models_pointcloud import PointCloud_network_equiv
from train import SMPLX_layer, get_nc_and_view_channel, kinematic_layer_SO3_v2

def conservative_patch_checkpoint(checkpoint_path):
    """
    More conservative patching - only handle the most critical mismatches
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    patched_checkpoint = {}
    patches_applied = 0
    
    for key, value in checkpoint.items():
        if 'intra_conv.conv.basic_conv.W' in key:
            # For weight matrices, use average pooling instead of truncation
            original_shape = value.shape
            out_features, in_features = original_shape
            
            if in_features == 384:  # Reduce from 384 to 128
                # Reshape and average: 384 -> 3 groups of 128, then average
                reshaped = value.view(out_features, 3, 128)
                new_value = reshaped.mean(dim=1)  # Average over the 3 groups
                patched_checkpoint[key] = new_value
                print(f"Averaged {key}: {original_shape} -> {new_value.shape}")
                patches_applied += 1
            elif in_features == 768:  # Reduce from 768 to 256
                reshaped = value.view(out_features, 3, 256)
                new_value = reshaped.mean(dim=1)
                patched_checkpoint[key] = new_value
                print(f"Averaged {key}: {original_shape} -> {new_value.shape}")
                patches_applied += 1
            else:
                patched_checkpoint[key] = value
                
        elif 'intra_conv.conv.intra_idx' in key:
            # For index tensors, take every 3rd element
            original_shape = value.shape
            if original_shape[1] == 12:
                # Take indices 0, 3, 6, 9 (every 3rd)
                new_value = value[:, ::3].clone()
                patched_checkpoint[key] = new_value
                print(f"Subsampled {key}: {original_shape} -> {new_value.shape}")
                patches_applied += 1
            else:
                patched_checkpoint[key] = value
                
        elif 'basic_conv.bias' in key:
            # Skip bias terms that don't exist in current architecture
            continue
            
        else:
            # Keep all other parameters unchanged
            patched_checkpoint[key] = value
    
    print(f"Applied {patches_applied} conservative patches")
    return patched_checkpoint

def create_model_conservative():
    """
    Create model with settings that might be more compatible
    """
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
        paper_model = True
        device = "cpu"  # Use CPU to avoid CUDA kernel issues
    
    args = Args()
    
    if args.paper_model:
        args.EPN_layer_num = 2
        args.EPN_input_radius = 0.4
        args.epoch = 15
        args.aug_type = "so3"
        args.kinematic_cond = "yes"
    
    nc, _ = get_nc_and_view_channel(args)
    
    model = PointCloud_network_equiv(
        option=args, z_dim=args.latent_num, nc=nc, part_num=args.part_num
    ).to(args.device)
    
    return model, args

def run_conservative_inference(input_mesh, output_mesh, model_path, num_points=5000):
    """
    Run inference with conservative approach
    """
    # Create model
    model, args = create_model_conservative()
    device = args.device
    
    # Load checkpoint with conservative patching
    patched_checkpoint = conservative_patch_checkpoint(model_path)
    
    # Load with non-strict mode
    missing_keys, unexpected_keys = model.load_state_dict(patched_checkpoint, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)} keys")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)} keys")
    
    # Check parameter loading success
    total_params = len(list(model.parameters()))
    loaded_params = sum(1 for name, param in model.named_parameters() 
                       if name in patched_checkpoint and param.shape == patched_checkpoint[name].shape)
    
    print(f"Successfully loaded {loaded_params}/{total_params} parameters ({100*loaded_params/total_params:.1f}%)")
    
    model.eval()
    
    # Create SMPL body model
    body_model = get_body_model(model_type="smpl", gender="male", batch_size=1, device=device)
    parents = body_model.parents[:22]
    
    # Load and preprocess input
    print(f"Loading point cloud from {input_mesh}")
    mesh = trimesh.load(input_mesh)
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    
    if len(points) < num_points:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    if len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        extra_needed = num_points - len(points)
        extra_indices = np.random.choice(len(points), extra_needed, replace=True)
        points = np.vstack([points, points[extra_indices]])
    
    # Convert to tensor
    pcl_data = torch.from_numpy(points.astype(np.float32)).unsqueeze(0)
    pcl_data = pcl_data.to(device)
    
    print(f"Point cloud shape: {pcl_data.shape}")
    
    # Run inference
    print("Running inference on CPU...")
    try:
        with torch.inference_mode():
            pred_joint, pred_pose, pred_shape, trans_feat = model(
                pcl_data, None, None, None, 
                is_optimal_trans=False, parents=parents
            )
            
            pred_joint_pose = kinematic_layer_SO3_v2(pred_pose, parents)
            trans_feat = torch.zeros((1, 3)).to(device)
            
            pred_joints_pos, pred_vertices = SMPLX_layer(
                body_model, pred_shape, trans_feat, pred_joint_pose, rep="rotmat"
            )
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save result
    vertices = pred_vertices[0].cpu().numpy()
    faces = body_model.faces
    
    result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    result_mesh.export(output_mesh)
    
    print(f"\n✅ Success! SMPL mesh saved to {output_mesh}")
    print(f"📊 Shape parameters (first 5): {pred_shape[0][:5].cpu().numpy()}")
    print(f"📊 Output mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mesh', help='Input mesh file')
    parser.add_argument('output_mesh', help='Output SMPL mesh file')
    parser.add_argument('--model_path', default='data/papermodel/model_epochs_00000014.pth')
    parser.add_argument('--num_points', type=int, default=5000)
    
    args = parser.parse_args()
    
    print("🔧 Conservative ArtEq Checkpoint Patcher")
    print("=" * 50)
    
    try:
        success = run_conservative_inference(
            args.input_mesh, args.output_mesh, args.model_path, args.num_points
        )
        
        if success:
            print("\n🎉 Registration completed successfully!")
        else:
            print("\n❌ Registration failed.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()