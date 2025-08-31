#!/usr/bin/env python3
"""
Convert Qwen2.5 models to GGUF format compatible with BitNet LUT kernels.
This script handles the conversion of Qwen2.5 models to work with the BitNet.cpp framework.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

# Add the gguf-py path
sys.path.insert(1, str(Path(__file__).parent.parent / "3rdparty" / "llama.cpp" / "gguf-py"))
import gguf

class Qwen25Converter:
    """Convert Qwen2.5 models to GGUF format with BitNet LUT support."""
    
    def __init__(self, model_path: str, output_path: str):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.model = None
        self.tokenizer = None
        self.config = None
        
    def load_model(self):
        """Load the Qwen2.5 model and tokenizer."""
        print(f"Loading model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.config = self.model.config
            print(f"Model loaded successfully: {self.config.model_type}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def get_tensor_mapping(self) -> Dict[str, str]:
        """Get the tensor mapping for Qwen2.5 architecture."""
        return {
            # Embeddings
            "model.embed_tokens.weight": "token_embd.weight",
            
            # Output layer
            "model.norm.weight": "output_norm.weight",
            "lm_head.weight": "output.weight",
            
            # Layer mappings
            "model.layers.{}.input_layernorm.weight": "blk.{}.attn_norm.weight",
            "model.layers.{}.self_attn.q_proj.weight": "blk.{}.attn_q.weight",
            "model.layers.{}.self_attn.k_proj.weight": "blk.{}.attn_k.weight",
            "model.layers.{}.self_attn.v_proj.weight": "blk.{}.attn_v.weight",
            "model.layers.{}.self_attn.o_proj.weight": "blk.{}.attn_output.weight",
            
            # MLP mappings
            "model.layers.{}.mlp.gate_proj.weight": "blk.{}.ffn_gate.weight",
            "model.layers.{}.mlp.up_proj.weight": "blk.{}.ffn_up.weight",
            "model.layers.{}.mlp.down_proj.weight": "blk.{}.ffn_down.weight",
            "model.layers.{}.post_attention_layernorm.weight": "blk.{}.ffn_norm.weight",
        }
    
    def quantize_weights(self, weights: torch.Tensor, bits: int = 2) -> tuple:
        """
        Quantize weights to the specified bit precision using BitNet-style quantization.
        
        Args:
            weights: Input weights tensor
            bits: Number of bits for quantization (2 for BitNet)
            
        Returns:
            tuple: (quantized_weights, scales)
        """
        if bits == 2:
            # BitNet-style 2-bit quantization
            weights_f32 = weights.float()
            
            # Calculate scales for each block
            block_size = 64  # BitNet uses 64-element blocks
            n_blocks = (weights_f32.numel() + block_size - 1) // block_size
            
            # Pad weights to be divisible by block_size
            pad_size = (block_size - weights_f32.numel() % block_size) % block_size
            if pad_size > 0:
                weights_f32 = torch.cat([weights_f32.flatten(), torch.zeros(pad_size)])
            
            weights_f32 = weights_f32.reshape(-1, block_size)
            
            # Calculate scales for each block
            scales = torch.max(torch.abs(weights_f32), dim=1, keepdim=True)[0]
            scales = scales.clamp(min=1e-8)
            
            # Quantize to 2-bit (-1, 0, 1)
            quantized = torch.round(weights_f32 / scales)
            quantized = torch.clamp(quantized, -1, 1)
            
            # Convert to uint8 for storage
            quantized_uint8 = (quantized + 1).to(torch.uint8)
            
            return quantized_uint8, scales.flatten()[:n_blocks]
        else:
            # Fallback to FP16
            return weights.half(), None
    
    def convert_model(self):
        """Convert the model to GGUF format."""
        print("Starting model conversion...")
        
        # Create GGUF writer
        writer = gguf.GGUFWriter(self.output_path, "qwen2.5-bitnet")
        
        # Add model metadata
        writer.add_name("qwen2.5-bitnet")
        writer.add_description("Qwen2.5 model converted for BitNet LUT inference")
        writer.add_model_family("qwen")
        writer.add_model_type("qwen2.5")
        writer.add_file_type(gguf.FileType.F16)
        
        # Add architecture info
        writer.add_architecture("qwen2.5")
        writer.add_context_length(self.config.max_position_embeddings)
        writer.add_embedding_length(self.config.hidden_size)
        writer.add_block_count(self.config.num_hidden_layers)
        writer.add_feed_forward_length(self.config.intermediate_size)
        writer.add_rope_dimension_count(self.config.hidden_size // self.config.num_attention_heads)
        writer.add_attention_head_count(self.config.num_attention_heads)
        writer.add_attention_head_count_kv(self.config.num_key_value_heads)
        writer.add_layer_norm_rms_eps(self.config.rms_norm_eps)
        
        # Add vocabulary
        self._add_vocab(writer)
        
        # Add tensors
        self._add_tensors(writer)
        
        # Write the model
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        
        print(f"Model converted successfully to {self.output_path}")
    
    def _add_vocab(self, writer: gguf.GGUFWriter):
        """Add vocabulary to the GGUF file."""
        print("Adding vocabulary...")
        
        # Get vocabulary
        vocab = self.tokenizer.get_vocab()
        
        # Add tokens
        for token, token_id in vocab.items():
            writer.add_token(token, token_id)
        
        # Add special tokens
        if self.tokenizer.pad_token:
            writer.add_token_id(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token:
            writer.add_token_id(self.tokenizer.eos_token_id)
        if self.tokenizer.bos_token:
            writer.add_token_id(self.tokenizer.bos_token_id)
    
    def _add_tensors(self, writer: gguf.GGUFWriter):
        """Add model tensors to the GGUF file."""
        print("Adding model tensors...")
        
        tensor_mapping = self.get_tensor_mapping()
        
        # Process each layer
        for layer_idx in range(self.config.num_hidden_layers):
            print(f"Processing layer {layer_idx}...")
            
            # Attention weights
            for tensor_name, gguf_name in tensor_mapping.items():
                if "{bid}" in tensor_name:
                    # Layer-specific tensors
                    actual_name = tensor_name.format(bid=layer_idx)
                    gguf_actual_name = gguf_name.format(bid=layer_idx)
                    
                    if actual_name in self.model.state_dict():
                        tensor = self.model.state_dict()[actual_name]
                        
                        # Quantize weights for attention and MLP layers
                        if any(keyword in actual_name for keyword in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 
                                                                   'o_proj.weight', 'gate_proj.weight', 'up_proj.weight', 'down_proj.weight']):
                            quantized, scales = self.quantize_weights(tensor, bits=2)
                            
                            # Add quantized weights
                            writer.add_tensor(gguf_actual_name, quantized.numpy())
                            
                            # Add scales if quantization was used
                            if scales is not None:
                                writer.add_tensor(f"{gguf_actual_name}.scales", scales.numpy())
                        else:
                            # Add non-quantized tensors (like norms)
                            writer.add_tensor(gguf_actual_name, tensor.numpy())
        
        # Add embedding and output layers
        if "model.embed_tokens.weight" in self.model.state_dict():
            writer.add_tensor("token_embd.weight", self.model.state_dict()["model.embed_tokens.weight"].numpy())
        
        if "model.norm.weight" in self.model.state_dict():
            writer.add_tensor("output_norm.weight", self.model.state_dict()["model.norm.weight"].numpy())
        
        if "lm_head.weight" in self.model.state_dict():
            writer.add_tensor("output.weight", self.model.state_dict()["lm_head.weight"].numpy())

def main():
    parser = argparse.ArgumentParser(description="Convert Qwen2.5 models to GGUF format for BitNet LUT inference")
    parser.add_argument("model_path", help="Path to the Qwen2.5 model directory")
    parser.add_argument("--output", "-o", default="qwen25-bitnet.gguf", help="Output GGUF file path")
    parser.add_argument("--bits", "-b", type=int, default=2, choices=[2, 4, 8, 16], help="Quantization bits")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        sys.exit(1)
    
    # Create converter and run conversion
    converter = Qwen25Converter(args.model_path, args.output)
    converter.load_model()
    converter.convert_model()

if __name__ == "__main__":
    main()
