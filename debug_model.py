#!/usr/bin/env python3
"""
Debug script to examine model architecture and checkpoint
"""

import sys
import torch
sys.path.append('src')

from models_pointcloud import PointCloud_network_equiv
from train import get_nc_and_view_channel

def debug_model_checkpoint(model_path):
    """Debug the saved model checkpoint"""
    print(f"Examining checkpoint: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"\nCheckpoint keys:")
    for key in sorted(checkpoint.keys()):
        print(f"  {key}: {checkpoint[key].shape}")
    
    # Look for encoder backbone structure
    encoder_keys = [k for k in checkpoint.keys() if 'encoder.backbone' in k]
    print(f"\nEncoder backbone keys ({len(encoder_keys)}):")
    for key in sorted(encoder_keys)[:10]:  # First 10
        print(f"  {key}: {checkpoint[key].shape}")
    
    return checkpoint

def create_debug_model():
    """Create model and examine its structure"""
    
    # Try different configurations
    configs = [
        # Original config
        {
            'EPN_input_radius': 0.4,
            'EPN_layer_num': 2, 
            'kinematic_cond': "yes",
            'latent_num': 128,
            'part_num': 22,
            'rep_type': "6d"
        },
        # Alternative config that might match the checkpoint
        {
            'EPN_input_radius': 0.4,
            'EPN_layer_num': 2,
            'kinematic_cond': "yes", 
            'latent_num': 128,
            'part_num': 22,
            'rep_type': "6d",
            'batch_size': 1,
            'num_point': 5000,
            'aug_type': "so3",
            'gt_part_seg': "auto"
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i+1} ---")
        
        class Args:
            pass
        
        args = Args()
        for k, v in config.items():
            setattr(args, k, v)
        
        nc, _ = get_nc_and_view_channel(args)
        
        model = PointCloud_network_equiv(
            option=args,
            z_dim=args.latent_num,
            nc=nc,
            part_num=args.part_num
        )
        
        print(f"Model created with nc={nc}")
        
        # Examine encoder structure
        encoder_params = [(name, param.shape) for name, param in model.named_parameters() 
                         if 'encoder.backbone' in name]
        print(f"Encoder parameters ({len(encoder_params)}):")
        for name, shape in encoder_params[:10]:  # First 10
            print(f"  {name}: {shape}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='data/papermodel/model_epochs_00000014.pth')
    args = parser.parse_args()
    
    print("=== Debugging Model Architecture ===")
    
    # Debug checkpoint
    try:
        checkpoint = debug_model_checkpoint(args.model_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Create debug models
    create_debug_model()

if __name__ == "__main__":
    main()