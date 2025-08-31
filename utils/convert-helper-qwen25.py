#!/usr/bin/env python3
"""
Helper script to convert Qwen2.5 models to GGUF format for BitNet LUT inference.
This script automates the entire conversion pipeline for Qwen2.5 models.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

def run_command(command_list, cwd=None, check=True):
    print(f"Executing: {' '.join(map(str, command_list))}")
    try:
        process = subprocess.run(command_list, cwd=cwd, check=check, capture_output=False, text=True)
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(map(str, e.cmd))}")
        print(f"Return code: {e.returncode}")
        raise

def main():
    if len(sys.argv) < 2:
        script_name = Path(sys.argv[0]).name
        print(f"Usage: python {script_name} <model-directory> [--quant-type {i2_s,tl1,tl2}]")
        sys.exit(1)

    model_dir_arg = sys.argv[1]
    model_dir = Path(model_dir_arg).resolve()

    if not model_dir.is_dir():
        print(f"Error: Model directory '{model_dir}' not found or is not a directory.")
        sys.exit(1)

    # Parse quantization type
    quant_type = "i2_s"  # default
    if len(sys.argv) > 2 and sys.argv[2] == "--quant-type":
        if len(sys.argv) > 3:
            quant_type = sys.argv[3]
        else:
            print("Error: --quant-type requires a value")
            sys.exit(1)

    utils_dir = Path(__file__).parent.resolve()
    project_root_dir = utils_dir.parent

    convert_script = utils_dir / "convert_qwen25_to_gguf.py"
    llama_quantize_binary = project_root_dir / "build" / "bin" / "llama-quantize"
    bitnet_convert_script = utils_dir / "convert-hf-to-gguf-bitnet.py"

    # Check if required files exist
    if not convert_script.is_file():
        print(f"Error: Convert script not found at '{convert_script}'")
        sys.exit(1)
    if not llama_quantize_binary.is_file():
        print(f"Error: llama-quantize binary not found at '{llama_quantize_binary}'")
        sys.exit(1)
    if not bitnet_convert_script.is_file():
        print(f"Error: BitNet conversion script not found at '{bitnet_convert_script}'")
        sys.exit(1)

    # Output file names
    gguf_f16_output = model_dir / "ggml-model-f16-qwen25.gguf"
    gguf_quantized_output = model_dir / f"ggml-model-{quant_type}-qwen25.gguf"

    try:
        if quant_type in ["tl1", "tl2"]:
            # Use our enhanced Qwen2.5 script for TL1/TL2 quantization
            print(f"Converting Qwen2.5 model directly to {quant_type.upper()} using enhanced conversion...")
            cmd_convert = [
                sys.executable,
                str(convert_script),
                str(model_dir),
                "--output", str(gguf_quantized_output),
                "--quant-type", quant_type
            ]
            run_command(cmd_convert)
        else:
            # For other quantization types, first convert to F16, then quantize
            print("Converting Qwen2.5 model to GGUF (f16)...")
            cmd_convert = [
                sys.executable,
                str(convert_script),
                str(model_dir),
                "--output", str(gguf_f16_output)
            ]
            run_command(cmd_convert)

            print(f"Quantizing model to {quant_type.upper()}...")
            cmd_quantize = [
                str(llama_quantize_binary),
                str(gguf_f16_output),
                str(gguf_quantized_output),
                quant_type.upper(),
                "1"
            ]
            run_command(cmd_quantize)

        print("Conversion completed successfully!")
        if quant_type in ["tl1", "tl2"]:
            print(f"TL{quant_type[-1]} quantized model: {gguf_quantized_output}")
        else:
            print(f"F16 model: {gguf_f16_output}")
            print(f"Quantized model: {gguf_quantized_output}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        print("Cleaning up intermediate files...")
        if quant_type not in ["tl1", "tl2"] and gguf_f16_output.exists():
            print(f"Removing f16 GGUF: {gguf_f16_output}")
            try:
                gguf_f16_output.unlink()
            except OSError as e:
                print(f"Warning: Could not remove {gguf_f16_output}: {e}")

if __name__ == "__main__":
    main()
