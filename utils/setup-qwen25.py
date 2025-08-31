#!/usr/bin/env python3
"""
Setup script for Qwen2.5 models with BitNet.cpp.
This script automates the entire setup process for running Qwen2.5 models.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_command(command_list, cwd=None, check=True):
    """Run a command and handle errors."""
    print(f"Executing: {' '.join(map(str, command_list))}")
    try:
        process = subprocess.run(command_list, cwd=cwd, check=check, capture_output=False, text=True)
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(map(str, e.cmd))}")
        print(f"Return code: {e.returncode}")
        raise

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        "torch", "transformers", "safetensors", 
        "numpy", "sentencepiece", "accelerate"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them using:")
        print(f"pip install -r requirements-qwen25.txt")
        return False
    
    print("All dependencies are installed!")
    return True

def download_model(model_name, output_dir):
    """Download a Qwen2.5 model from Hugging Face."""
    print(f"Downloading {model_name} to {output_dir}...")
    
    try:
        cmd = [
            "huggingface-cli", "download", model_name,
            "--local-dir", output_dir
        ]
        run_command(cmd)
        print(f"✅ Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

def setup_environment(model_dir, quant_type="tl1"):
    """Setup the environment for a specific model."""
    print(f"Setting up environment for {model_dir}...")
    
    utils_dir = Path(__file__).parent
    project_root = utils_dir.parent
    
    # Check if model directory exists
    if not Path(model_dir).exists():
        print(f"❌ Model directory {model_dir} not found!")
        return False
    
    # Generate kernels if they don't exist
    kernel_dir = project_root / "preset_kernels" / Path(model_dir).name
    if not kernel_dir.exists():
        print(f"Generating kernels for {Path(model_dir).name}...")
        try:
            # Extract model size from directory name
            model_size = Path(model_dir).name.split("-")[-1]  # e.g., "7b" from "qwen2.5-7b"
            if model_size.lower().endswith('b'):
                model_size = model_size.upper()
            
            cmd = [
                sys.executable,
                str(utils_dir / "generate_qwen25_kernels.py"),
                model_size,
                "--output", str(kernel_dir)
            ]
            run_command(cmd)
        except Exception as e:
            print(f"❌ Failed to generate kernels: {e}")
            return False
    
    # Convert model to GGUF
    print("Converting model to GGUF format...")
    try:
        cmd = [
            sys.executable,
            str(utils_dir / "convert-helper-qwen25.py"),
            model_dir,
            "--quant-type", quant_type
        ]
        run_command(cmd)
        print(f"✅ Model converted successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to convert model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup Qwen2.5 models for BitNet.cpp")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-7B-Instruct",
                       help="Hugging Face model name or local path")
    parser.add_argument("--output-dir", "-o", default="./models",
                       help="Output directory for models")
    parser.add_argument("--quant-type", "-q", default="tl1", 
                       choices=["tl1", "tl2", "i2_s"],
                       help="Quantization type")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip downloading if model already exists")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Determine if this is a local path or Hugging Face model
    if os.path.exists(args.model):
        # Local path
        model_dir = args.model
        print(f"Using local model: {model_dir}")
    else:
        # Hugging Face model
        model_name = args.model
        model_dir = os.path.join(args.output_dir, model_name.split("/")[-1])
        
        if not args.skip_download and not os.path.exists(model_dir):
            if not download_model(model_name, model_dir):
                sys.exit(1)
        elif os.path.exists(model_dir):
            print(f"Model already exists at {model_dir}")
        else:
            print(f"❌ Model not found at {model_dir}")
            sys.exit(1)
    
    # Setup environment
    if setup_environment(model_dir, args.quant_type):
        print(f"\n🎉 Setup completed successfully!")
        print(f"Model: {model_dir}")
        print(f"Quantization: {args.quant_type}")
        print(f"Kernels: preset_kernels/{Path(model_dir).name}")
        print(f"\nYou can now run inference with:")
        print(f"python run_inference.py -m {model_dir}/ggml-model-{args.quant_type}-qwen25.gguf -p 'Hello!' -cnv")
    else:
        print(f"\n❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
