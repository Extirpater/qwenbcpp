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
    

    
    def convert_model(self):
        """Convert the model to GGUF format."""
        print("Starting model conversion...")
        
        # Create GGUF writer
        writer = gguf.GGUFWriter(self.output_path, "qwen2.5-bitnet")
        
        # Add model metadata using correct GGUF API methods
        writer.add_name("qwen2.5-bitnet")
        
        # Add architecture info
        writer.add_context_length(self.config.max_position_embeddings)
        writer.add_embedding_length(self.config.hidden_size)
        writer.add_block_count(self.config.num_hidden_layers)
        writer.add_feed_forward_length(self.config.intermediate_size)
        writer.add_head_count(self.config.num_attention_heads)
        writer.add_head_count_kv(self.config.num_key_value_heads)
        writer.add_layer_norm_rms_eps(self.config.rms_norm_eps)
        
        # Add file type
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)
        
        # Add vocabulary
        self._add_vocab(writer)
        
        # Add tensors
        self._add_tensors(writer)
        
        # Write the model
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        
        print(f"Model converted successfully to {self.output_path}")
    
    def _add_vocab(self, writer: gguf.GGUFWriter):
        """Add vocabulary to the GGUF file."""
        print("Adding vocabulary...")
        
        # Get vocabulary
        vocab = self.tokenizer.get_vocab()
        
        # Add vocabulary size
        writer.add_vocab_size(len(vocab))
        
        # Prepare tokens, scores, and types
        tokens = []
        scores = []
        toktypes = []
        
        # Sort vocabulary by token ID to ensure correct ordering
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        
        for token, token_id in sorted_vocab:
            tokens.append(token)
            # Use default score of 0.0 for all tokens
            scores.append(0.0)
            # Use normal token type for all tokens
            toktypes.append(1)  # 1 = NORMAL token type
        
        # Add tokens, scores, and types
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(toktypes)
        
        # Add special tokens
        if self.tokenizer.pad_token:
            writer.add_pad_token_id(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token:
            writer.add_eos_token_id(self.tokenizer.eos_token_id)
        if self.tokenizer.bos_token:
            writer.add_bos_token_id(self.tokenizer.bos_token_id)
        if self.tokenizer.unk_token:
            writer.add_unk_token_id(self.tokenizer.unk_token_id)
    
    def _add_tensors(self, writer: gguf.GGUFWriter):
        """Add model tensors to the GGUF file."""
        print("Adding model tensors...")
        
        tensor_mapping = self.get_tensor_mapping()
        
        # Process each layer
        for layer_idx in range(self.config.num_hidden_layers):
            print(f"Processing layer {layer_idx}...")
            
            # Attention weights
            for tensor_name, gguf_name in tensor_mapping.items():
                if "{}" in tensor_name:
                    # Layer-specific tensors
                    actual_name = tensor_name.format(layer_idx)
                    gguf_actual_name = gguf_name.format(layer_idx)
                    
                    if actual_name in self.model.state_dict():
                        tensor = self.model.state_dict()[actual_name]
                        print(f"  Adding tensor: {actual_name} -> {gguf_actual_name}")
                        
                        # Add weights in F32 format (quantization will be handled by llama-quantize)
                        writer.add_tensor(gguf_actual_name, tensor.numpy())
                    else:
                        print(f"  Warning: Tensor {actual_name} not found in model state dict")
        
        # Add embedding and output layers
        print("Adding embedding and output layers...")
        
        if "model.embed_tokens.weight" in self.model.state_dict():
            print("  Adding token_embd.weight")
            writer.add_tensor("token_embd.weight", self.model.state_dict()["model.embed_tokens.weight"].numpy())
        else:
            print("  Warning: model.embed_tokens.weight not found")
        
        if "model.norm.weight" in self.model.state_dict():
            print("  Adding output_norm.weight")
            writer.add_tensor("output_norm.weight", self.model.state_dict()["model.norm.weight"].numpy())
        else:
            print("  Warning: model.norm.weight not found")
        
        if "lm_head.weight" in self.model.state_dict():
            print("  Adding output.weight")
            writer.add_tensor("output.weight", self.model.state_dict()["lm_head.weight"].numpy())
        else:
            print("  Warning: lm_head.weight not found")

def main():
    parser = argparse.ArgumentParser(description="Convert Qwen2.5 models to GGUF format for BitNet LUT inference")
    parser.add_argument("model_path", help="Path to the Qwen2.5 model directory")
    parser.add_argument("--output", "-o", default="qwen25-bitnet.gguf", help="Output GGUF file path")
    
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
