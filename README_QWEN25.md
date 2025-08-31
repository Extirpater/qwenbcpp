# Qwen2.5 Support for BitNet.cpp

This document explains how to use Qwen2.5 models with the BitNet.cpp framework for efficient CPU inference using Look-Up Table (LUT) kernels.

## Overview

BitNet.cpp now supports Qwen2.5 models through specialized LUT kernels that provide:
- **2-bit quantization** for memory efficiency
- **Optimized CPU inference** using ARM NEON and x86 AVX2 instructions
- **Compatible with existing BitNet.cpp infrastructure**

## Supported Qwen2.5 Models

The following Qwen2.5 model sizes are supported:

| Model Size | Hidden Size | Intermediate Size | Layers | Heads | Status |
|------------|-------------|-------------------|---------|-------|---------|
| 0.5B       | 1024        | 2816              | 24     | 16    | ✅      |
| 1.5B       | 1536        | 4096              | 24     | 24    | ✅      |
| 3B         | 2048        | 5632              | 24     | 32    | ✅      |
| 7B         | 4096        | 11008             | 32     | 32    | ✅      |
| 14B        | 5120        | 13696             | 40     | 40    | ✅      |
| 32B        | 6656        | 17920             | 48     | 52    | ✅      |
| 72B        | 8192        | 22016             | 80     | 64    | ✅      |

## Quick Start

### 1. Download a Qwen2.5 Model

```bash
# Download from Hugging Face
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/qwen2.5-7b
```

### 2. Convert to GGUF Format

```bash
# Convert using the Qwen2.5-specific converter
python utils/convert-helper-qwen25.py ./models/qwen2.5-7b --quant-type tl1
```

### 3. Run Inference

```bash
# Run with the converted model
python run_inference.py -m ./models/qwen2.5-7b/ggml-model-tl1-qwen25.gguf -p "Hello, how are you?" -cnv
```

## Conversion Process

### Automatic Conversion

The `convert-helper-qwen25.py` script automates the entire conversion pipeline:

1. **Load Qwen2.5 model** from Hugging Face format
2. **Apply BitNet quantization** to weights (2-bit precision)
3. **Convert to GGUF format** with proper tensor mapping
4. **Generate quantized model** ready for BitNet.cpp inference

### Manual Conversion

For custom conversion needs:

```bash
# Direct conversion to GGUF
python utils/convert-qwen25-to-gguf.py ./models/qwen2.5-7b --output qwen25-bitnet.gguf

# Quantize using llama-quantize
./build/bin/llama-quantize qwen25-bitnet.gguf qwen25-tl1.gguf TL1 1
```

## Kernel Configuration

### Generated Kernels

Kernel configurations are automatically generated for each model size:

```bash
# Generate kernels for specific model size
python utils/generate-qwen25-kernels.py 7B --output preset_kernels/qwen25-7b
```

### Kernel Types

- **TL1**: Optimized for ARM processors with NEON instructions
- **TL2**: Optimized for x86 processors with AVX2 instructions

### Custom Kernel Tuning

You can manually adjust kernel parameters in the generated `.ini` files:

```ini
[Kernels_0]
m = 4096          # Hidden size
k = 11008         # Intermediate size
bm = 128          # Block size M
bk = 64           # Block size K
bmm = 32          # SIMD block size
```

## Performance Optimization

### Platform-Specific Builds

```bash
# ARM (Apple Silicon, ARM64)
cmake -DBITNET_ARM_TL1=ON -B build
make -C build -j

# x86 (Intel/AMD)
cmake -DBITNET_X86_TL2=ON -B build
make -C build -j
```

### Memory Usage

- **F32 model**: ~28GB for Qwen2.5-7B
- **TL1 quantized**: ~7GB for Qwen2.5-7B
- **TL2 quantized**: ~7GB for Qwen2.5-7B

## Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure the model is in Hugging Face format
2. **Conversion failures**: Check Python dependencies (transformers, torch, safetensors)
3. **Inference errors**: Verify kernel configuration matches model architecture

### Debug Mode

Enable verbose logging during conversion:

```bash
python utils/convert-qwen25-to-gguf.py ./models/qwen2.5-7b --output qwen25-bitnet.gguf --debug
```

## Advanced Usage

### Custom Quantization

Modify the quantization parameters in `convert-qwen25-to-gguf.py`:

```python
def quantize_weights(self, weights: torch.Tensor, bits: int = 2):
    # Customize block size, scaling method, etc.
    block_size = 128  # Instead of default 64
    # ... custom quantization logic
```

### Multi-GPU Support

For large models that don't fit in single GPU memory:

```python
# In convert-qwen25-to-gguf.py
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_path,
    torch_dtype=torch.float16,
    device_map="auto",  # Automatic device mapping
    trust_remote_code=True
)
```

## Contributing

To add support for new Qwen2.5 variants:

1. **Update model configurations** in `generate-qwen25-kernels.py`
2. **Test conversion pipeline** with the new model
3. **Validate inference performance** using the generated kernels
4. **Update documentation** with new model specifications

## References

- [BitNet.cpp Documentation](README.md)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [BitNet LUT Kernels](preset_kernels/)
