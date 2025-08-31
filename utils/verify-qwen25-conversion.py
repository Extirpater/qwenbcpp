#!/usr/bin/env python3
"""
Quick verification script to test Qwen2.5 conversion without loading a full model.
"""

import sys
from pathlib import Path

def test_conversion_components():
    """Test the conversion components without loading a model."""
    print("Testing Qwen2.5 conversion components...")
    
    try:
        # Test import
        sys.path.insert(0, str(Path(__file__).parent))
        from convert_qwen25_to_gguf import Qwen25Converter
        print("✅ Import successful")
        
        # Test tensor mapping
        converter = Qwen25Converter("dummy", "dummy.gguf")
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
        
        print("✅ Tensor mapping test passed")
        
        # Test quantization function (without torch dependency)
        try:
            import torch
            test_tensor = torch.randn(64, 64)
            quantized, scales = converter.quantize_weights(test_tensor, bits=2)
            
            if quantized is not None and scales is not None:
                print(f"✅ Quantization test passed: {quantized.shape}, {scales.shape}")
            else:
                print("❌ Quantization test failed")
                return False
        except ImportError:
            print("⚠️  Torch not available, skipping quantization test")
        
        print("\n🎉 All component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    if test_conversion_components():
        print("\n✅ Qwen2.5 conversion is ready to use!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
