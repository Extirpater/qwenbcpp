#!/usr/bin/env python3
"""
Example script demonstrating how to use Qwen2.5 models with BitNet.cpp.
This script shows the complete workflow from model download to inference.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 BitNet.cpp Example")
    parser.add_argument("--model-size", default="7B", choices=["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"],
                       help="Qwen2.5 model size to use")
    parser.add_argument("--quant-type", default="tl1", choices=["tl1", "tl2", "i2_s"],
                       help="Quantization type for BitNet kernels")
    parser.add_argument("--skip-setup", action="store_true",
                       help="Skip setup if model is already prepared")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Qwen2.5 BitNet.cpp Example")
    print("=" * 60)
    print(f"Model Size: {args.model_size}")
    print(f"Quantization: {args.quant_type}")
    print("=" * 60)
    
    # Get project paths
    project_root = Path(__file__).parent.parent
    utils_dir = project_root / "utils"
    
    if not args.skip_setup:
        print("\n1. Setting up Qwen2.5 model...")
        print("   This will download the model and convert it to GGUF format.")
        
        # Run setup script
        setup_cmd = [
            sys.executable,
            str(utils_dir / "setup-qwen25.py"),
            "--model", f"Qwen/Qwen2.5-{args.model_size}-Instruct",
            "--quant-type", args.quant_type
        ]
        
        print(f"   Running: {' '.join(setup_cmd)}")
        print("   Note: This may take a while depending on your internet connection.")
        
        # In a real scenario, you would run this:
        # subprocess.run(setup_cmd, check=True)
        
        print("   ✅ Setup completed (simulated)")
    else:
        print("\n1. Skipping setup (model already prepared)")
    
    # Model paths
    model_name = f"qwen2.5-{args.model_size.lower()}"
    model_dir = project_root / "models" / model_name
    gguf_model = model_dir / f"ggml-model-{args.quant_type}-qwen25.gguf"
    
    print(f"\n2. Model information:")
    print(f"   Model directory: {model_dir}")
    print(f"   GGUF model: {gguf_model}")
    
    if not gguf_model.exists():
        print(f"   ❌ GGUF model not found. Please run setup first.")
        return
    
    print(f"   ✅ GGUF model found")
    
    print(f"\n3. Kernel configuration:")
    kernel_dir = project_root / "preset_kernels" / model_name
    if kernel_dir.exists():
        print(f"   Kernel directory: {kernel_dir}")
        kernel_files = list(kernel_dir.glob("*.ini"))
        for kernel_file in kernel_files:
            print(f"   print(f"   - {kernel_file.name}")
    else:
        print(f"   ❌ Kernel directory not found")
    
    print(f"\n4. Running inference:")
    print(f"   To run inference with this model, use:")
    print(f"   python run_inference.py -m {gguf_model} -p 'Hello, how are you?' -cnv")
    
    print(f"\n5. Performance optimization:")
    if args.quant_type == "tl1":
        print(f"   Build for ARM: cmake -DBITNET_ARM_TL1=ON -B build")
    elif args.quant_type == "tl2":
        print(f"   Build for x86: cmake -DBITNET_X86_TL2=ON -B build")
    
    print(f"\n6. Benchmarking:")
    print(f"   To benchmark performance:")
    print(f"   python utils/e2e_benchmark.py -m {gguf_model} -p 512 -n 128")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
