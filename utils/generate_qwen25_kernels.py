#!/usr/bin/env python3
"""
Generate Qwen2.5-specific LUT kernels for BitNet.cpp.
This script creates optimized kernel configurations for Qwen2.5 models.
"""

import argparse
import os
import sys
from pathlib import Path

def generate_qwen25_kernels(model_size: str, output_dir: str):
    """Generate Qwen2.5 kernel configurations."""
    
    # Qwen2.5 model configurations
    qwen25_configs = {
        "0.5B": {
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_layers": 24,
            "num_heads": 16,
            "num_kv_heads": 16
        },
        "1.5B": {
            "hidden_size": 1536,
            "intermediate_size": 4096,
            "num_layers": 24,
            "num_heads": 24,
            "num_kv_heads": 24
        },
        "3B": {
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_layers": 24,
            "num_heads": 32,
            "num_kv_heads": 32
        },
        "7B": {
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_layers": 32,
            "num_heads": 32,
            "num_kv_heads": 32
        },
        "14B": {
            "hidden_size": 5120,
            "intermediate_size": 13696,
            "num_layers": 40,
            "num_heads": 40,
            "num_kv_heads": 40
        },
        "32B": {
            "hidden_size": 6656,
            "intermediate_size": 17920,
            "num_layers": 48,
            "num_heads": 52,
            "num_kv_heads": 52
        },
        "72B": {
            "hidden_size": 8192,
            "intermediate_size": 22016,
            "num_layers": 80,
            "num_heads": 64,
            "num_kv_heads": 64
        }
    }
    
    if model_size not in qwen25_configs:
        print(f"Error: Unsupported model size '{model_size}'")
        print(f"Supported sizes: {', '.join(qwen25_configs.keys())}")
        sys.exit(1)
    
    config = qwen25_configs[model_size]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate TL1 kernel configuration
    tl1_config = f"""[Kernels_0]
m = {config['hidden_size']}
k = {config['intermediate_size']}
bm = {min(128, config['hidden_size'])}
bk = 64
bmm = 32

[Kernels_1]
m = {config['hidden_size']}
k = {config['hidden_size']}
bm = {min(256, config['hidden_size'])}
bk = 128
bmm = 64

[Kernels_2]
m = {config['intermediate_size']}
k = {config['hidden_size']}
bm = {min(256, config['intermediate_size'])}
bk = 64
bmm = 32
"""
    
    # Write TL1 config
    with open(output_path / "kernel_config_tl1.ini", "w") as f:
        f.write(tl1_config)
    
    # Generate TL2 kernel configuration (for x86)
    tl2_config = f"""[Kernels_0]
m = {config['hidden_size']}
k = {config['intermediate_size']}
bm = {min(160, config['hidden_size'])}
bk = 64
bmm = 32

[Kernels_1]
m = {config['hidden_size']}
k = {config['hidden_size']}
bm = {min(320, config['hidden_size'])}
bk = 128
bmm = 64

[Kernels_2]
m = {config['intermediate_size']}
k = {config['hidden_size']}
bm = {min(320, config['intermediate_size'])}
bk = 64
bmm = 32
"""
    
    # Write TL2 config
    with open(output_path / "kernel_config_tl2.ini", "w") as f:
        f.write(tl2_config)
    
    print(f"Generated Qwen2.5 {model_size} kernel configurations in {output_path}")
    print(f"Hidden size: {config['hidden_size']}")
    print(f"Intermediate size: {config['intermediate_size']}")
    print(f"Number of layers: {config['num_layers']}")
    print(f"Number of heads: {config['num_heads']}")

def main():
    parser = argparse.ArgumentParser(description="Generate Qwen2.5 LUT kernel configurations")
    parser.add_argument("model_size", choices=["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"],
                       help="Qwen2.5 model size")
    parser.add_argument("--output", "-o", default="preset_kernels/qwen25",
                       help="Output directory for kernel configurations")
    
    args = parser.parse_args()
    
    generate_qwen25_kernels(args.model_size, args.output)

if __name__ == "__main__":
    main()
