#!/usr/bin/env python3
"""
VGTK Dimension Analyzer - Understand what the vgtk library expects
and fix the dimension compatibility issues systematically
"""

import sys
import os
import numpy as np
import torch
import inspect

# Add src directory to path
sys.path.append('src')
sys.path.append('external/vgtk')

# Import vgtk components
try:
    import vgtk
    import vgtk.so3conv as sptk
    from so3conv import M
    print("✅ Successfully imported vgtk")
except Exception as e:
    print(f"❌ Error importing vgtk: {e}")

def analyze_vgtk_so3conv_requirements():
    """
    Analyze what the SO3Conv layers expect in terms of dimensions
    """
    print("\n🔍 Analyzing VGTK SO3Conv Requirements")
    print("=" * 50)
    
    try:
        # Test different input configurations to understand requirements
        batch_size = 1
        num_points = 1000
        
        # Test 1: Basic 3D input
        print("\n1. Testing basic 3D input...")
        test_input_3d = torch.randn(batch_size, num_points, 3).cuda()
        
        try:
            # Try to create a basic SphericalPointCloud
            xyz = test_input_3d.permute(0, 2, 1).contiguous()
            
            # Test different feature dimensions
            for feat_dim in [1, 3, 6, 12]:
                print(f"  Testing feature dimension: {feat_dim}")
                try:
                    feat = torch.randn(batch_size, feat_dim, num_points).cuda()
                    spc = sptk.SphericalPointCloud(xyz, feat, None)
                    print(f"    ✅ Success with feat_dim={feat_dim}")
                    
                    # Test with different na (anchor) values
                    for na in [1, 4, 12, 60]:
                        try:
                            occupancy_feat = sptk.get_occupancy_features(test_input_3d, na, True)
                            print(f"    ✅ Occupancy features work with na={na}, shape: {occupancy_feat.shape}")
                        except Exception as e:
                            print(f"    ❌ Occupancy features failed with na={na}: {e}")
                            
                except Exception as e:
                    print(f"    ❌ Failed with feat_dim={feat_dim}: {e}")
                    
        except Exception as e:
            print(f"❌ Basic 3D test failed: {e}")
    
    except Exception as e:
        print(f"❌ VGTK analysis failed: {e}")

def analyze_preprocess_input_function():
    """
    Analyze the preprocess_input function to understand the expected format
    """
    print("\n🔍 Analyzing preprocess_input Function")
    print("=" * 50)
    
    try:
        # Look at the preprocess_input function
        from so3conv import preprocess_input
        
        print("preprocess_input signature:")
        sig = inspect.signature(preprocess_input)
        print(f"  {sig}")
        
        # Test different inputs
        batch_size = 1
        num_points = 1000
        
        # Test different input formats
        test_cases = [
            (3, "3D coordinates only"),
            (6, "3D coordinates + 3D normals"),
            (9, "3D coordinates + 6D features"),
            (12, "3D coordinates + 9D features")
        ]
        
        for input_dim, description in test_cases:
            print(f"\nTesting: {description} (dim={input_dim})")
            try:
                test_input = torch.randn(batch_size, num_points, input_dim).cuda()
                
                # Test different na values
                for na in [1, 4, 12, 60]:
                    try:
                        result = preprocess_input(test_input, na, True)
                        print(f"  ✅ na={na}: xyz={result.xyz.shape}, feat={result.feats.shape}")
                    except Exception as e:
                        print(f"  ❌ na={na}: {e}")
                        
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    except Exception as e:
        print(f"❌ preprocess_input analysis failed: {e}")

def analyze_checkpoint_vs_model_mismatch():
    """
    Analyze the specific mismatch between checkpoint and current model
    """
    print("\n🔍 Analyzing Checkpoint vs Model Mismatch")
    print("=" * 50)
    
    # Load checkpoint to analyze expected dimensions
    checkpoint_path = "data/papermodel/model_epochs_00000014.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("Key dimension mismatches found:")
        
        # Analyze the problematic layers
        problematic_keys = [
            'encoder.backbone.0.blocks.0.intra_conv.conv.intra_idx',
            'encoder.backbone.0.blocks.0.intra_conv.conv.basic_conv.W'
        ]
        
        for key in problematic_keys:
            if key in checkpoint:
                shape = checkpoint[key].shape
                print(f"  {key}: {shape}")
                
                if 'intra_idx' in key and shape[1] == 12:
                    print(f"    → intra_idx expects 12 features, model creates 4")
                    print(f"    → Ratio: 12/4 = 3 (the 3x factor!)")
                    
                elif 'basic_conv.W' in key:
                    in_feat, out_feat = shape[1], shape[0]
                    expected_in = in_feat // 3
                    print(f"    → Weight matrix expects {in_feat} input features")
                    print(f"    → Model likely creates {expected_in} features")
                    print(f"    → Difference factor: {in_feat // expected_in}")
    
    else:
        print("❌ Checkpoint not found")

def propose_dimension_fix():
    """
    Propose a systematic fix for the dimension mismatch
    """
    print("\n🔧 Proposed Dimension Fix Strategy")
    print("=" * 50)
    
    print("Based on analysis, the issue is:")
    print("1. Checkpoint expects 12-channel features per group")
    print("2. Current model creates 4-channel features per group") 
    print("3. This creates the 3x dimension mismatch (12/4 = 3)")
    print()
    
    print("Possible solutions:")
    print("A. Modify model to output 12 channels instead of 4")
    print("B. Create a feature expansion layer (4→12 channels)")
    print("C. Modify vgtk library to accept 4 channels")
    print("D. Create a compatibility wrapper")
    print()
    
    print("Recommended approach: Feature expansion wrapper")
    print("- Intercept model inputs/outputs")
    print("- Expand 4-channel features to 12-channel")
    print("- Map 12-channel checkpoint weights to 4-channel model")

def main():
    print("🔬 VGTK Dimension Compatibility Analyzer")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - some tests may fail")
    
    try:
        # Run analyses
        analyze_vgtk_so3conv_requirements()
        analyze_preprocess_input_function()
        analyze_checkpoint_vs_model_mismatch()
        propose_dimension_fix()
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()