#!/usr/bin/env python3
"""
Test script for Qwen2.5 conversion to verify everything works correctly.
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

def test_qwen25_conversion():
    """Test the Qwen2.5 conversion pipeline."""
    print("Testing Qwen2.5 conversion pipeline...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        # Test kernel generation
        print("\n1. Testing kernel generation...")
        try:
            # Run the kernel generation script directly
            utils_dir = Path(__file__).parent
            project_root = utils_dir.parent
            
            cmd = [
                sys.executable,
                str(utils_dir / "generate_qwen25_kernels.py"),
                "7B",
                "--output", temp_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                # Check if files were created
                kernel_files = list(Path(temp_dir).glob("*.ini"))
                if kernel_files:
                    print(f"✅ Kernel generation successful: {len(kernel_files)} files created")
                    for f in kernel_files:
                        print(f"   - {f.name}")
                else:
                    print("❌ Kernel generation failed: no files created")
                    return False
            else:
                print(f"❌ Kernel generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Kernel generation failed: {e}")
            return False
        
        # Test conversion script import
        print("\n2. Testing conversion script import...")
        try:
            # Add utils to path and import
            sys.path.insert(0, str(utils_dir))
            from convert_qwen25_to_gguf import Qwen25Converter
            print("✅ Conversion script import successful")
        except Exception as e:
            print(f"❌ Conversion script import failed: {e}")
            return False
        
        # Test tensor mapping
        print("\n3. Testing tensor mapping...")
        try:
            converter = Qwen25Converter("dummy_path", "dummy_output")
            tensor_mapping = converter.get_tensor_mapping()
            
            expected_keys = [
                "model.embed_tokens.weight",
                "model.layers.{}.self_attn.q_proj.weight",
                "model.layers.{}.mlp.gate_proj.weight"
            ]
            
            for key in expected_keys:
                if key in tensor_mapping:
                    print(f"✅ Found mapping for {key}")
                else:
                    print(f"❌ Missing mapping for {key}")
                    return False
            
            print("✅ Tensor mapping test successful")
        except Exception as e:
            print(f"❌ Tensor mapping test failed: {e}")
            return False
        
        # Test quantization function
        print("\n4. Testing quantization function...")
        try:
            import torch
            test_tensor = torch.randn(128, 256)
            quantized, scales = converter.quantize_weights(test_tensor, bits=2)
            
            if quantized is not None and scales is not None:
                print(f"✅ Quantization successful: {quantized.shape}, {scales.shape}")
            else:
                print("❌ Quantization failed")
                return False
        except Exception as e:
            print(f"❌ Quantization test failed: {e}")
            return False
    
    print("\n🎉 All tests passed!")
    return True

def main():
    """Run the test suite."""
    print("=" * 50)
    print("Qwen2.5 Conversion Test Suite")
    print("=" * 50)
    
    if test_qwen25_conversion():
        print("\n✅ All tests passed! Qwen2.5 conversion is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
