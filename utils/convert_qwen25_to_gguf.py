#!/usr/bin/env python3
"""
Convert Qwen2.5 models to GGUF format compatible with BitNet LUT kernels.
This script follows the exact same pattern as the working BitNet conversion scripts.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

# Add the gguf-py path
sys.path.insert(1, str(Path(__file__).parent.parent / "3rdparty" / "llama.cpp" / "gguf-py"))
import gguf

class Qwen25Converter:
    """Convert Qwen2.5 models to GGUF format following the same pattern as working BitNet scripts."""
    
    def __init__(self, model_path: str, output_path: str, quant_type: str = "f16"):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.quant_type = quant_type
        self.model = None
        self.tokenizer = None
        self.config = None
        self.hparams = None
        
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
            
            # Load hparams like the working scripts do
            self.hparams = self.load_hparams()
            
            print(f"Model loaded successfully: {self.config.model_type}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def load_hparams(self) -> Dict[str, Any]:
        """Load model hyperparameters from config.json."""
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add architecture info like the working scripts
        config["architectures"] = ["Qwen2ForCausalLM"]
        
        return config
    
    def add_tokenizer_merges(self, writer: gguf.GGUFWriter):
        """Add tokenizer merges for BPE tokenizers like Qwen."""
        try:
            # Check if tokenizer has merges
            if hasattr(self.tokenizer, 'merges') and self.tokenizer.merges:
                print(f"Adding {len(self.tokenizer.merges)} tokenizer merges...")
                writer.add_token_merges(self.tokenizer.merges)
            else:
                # Try to load from merges.txt file
                merges_path = self.model_path / "merges.txt"
                if merges_path.exists():
                    with open(merges_path, 'r', encoding='utf-8') as f:
                        merges = []
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                parts = line.split()
                                if len(parts) == 2:
                                    merges.append(f'{parts[0]} {parts[1]}')
                                else:
                                    print(f"Warning: Invalid merge format at line {line_num}: {line}")
                        if merges:
                            print(f"Adding {len(merges)} merges from merges.txt...")
                            writer.add_token_merges(merges)
                        else:
                            print("Warning: No valid merges found in merges.txt")
                else:
                    print("Warning: No tokenizer merges found, model may not work properly")
        except Exception as e:
            print(f"Warning: Could not add tokenizer merges: {e}")
    
    def get_actual_vocab_size(self) -> int:
        """Get the actual vocabulary size from the embedding tensor dimensions."""
        try:
            # Look for the embedding tensor in the model
            for tensor_name, data_torch in self.get_tensors():
                if tensor_name == "model.embed_tokens.weight":
                    # The first dimension should be the vocabulary size
                    vocab_size = data_torch.shape[0]
                    print(f"Found embedding tensor with shape: {data_torch.shape}, vocab_size: {vocab_size}")
                    return vocab_size
            return None
        except Exception as e:
            print(f"Warning: Could not get actual vocabulary size: {e}")
            return None
    
    def transform_to_tl1(self, x: np.ndarray):
        """Transform weights to TL1 format (2-bit quantization)."""
        scale = np.max(np.abs(x))
        if scale == 0:
            scale = 1.0
        # Quantize to 2-bit (-1, 0, 1) and scale
        res = np.round(x / scale).astype(np.int8)
        res = np.clip(res, -1, 1)
        return res, scale.astype(np.float32)
    
    def transform_to_tl2(self, x: np.ndarray):
        """Transform weights to TL2 format (2-bit quantization)."""
        scale = np.max(np.abs(x))
        if scale == 0:
            scale = 1.0
        # Quantize to 2-bit (-1, 0, 1) and scale
        res = np.round(x / scale).astype(np.int8)
        res = np.clip(res, -1, 1)
        return res, scale.astype(np.float32)
    
    def convert_model(self):
        """Convert the model to GGUF format."""
        print("Starting model conversion...")
        
        # Create GGUF writer using the same pattern as working scripts
        writer = gguf.GGUFWriter(self.output_path, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.QWEN2])
        
        # Set GGUF parameters like the working scripts
        self.set_gguf_parameters(writer)
        
        # Get actual vocabulary size from embedding tensor dimensions FIRST
        actual_vocab_size = self.get_actual_vocab_size()
        if actual_vocab_size:
            print(f"Model embedding tensor indicates vocabulary size: {actual_vocab_size}")
            print(f"Using model vocabulary size ({actual_vocab_size}) to match tensor dimensions")
            # Use the actual model vocabulary size to match tensor dimensions
            writer.add_vocab_size(actual_vocab_size)
        else:
            # Use the tokenizer's actual vocabulary size instead of hardcoded value
            tokenizer_vocab_size = len(self.tokenizer.get_vocab()) if hasattr(self.tokenizer, 'get_vocab') else self.tokenizer.vocab_size
            print(f"Warning: Could not determine actual vocabulary size, using tokenizer size ({tokenizer_vocab_size})")
            writer.add_vocab_size(tokenizer_vocab_size)
        

        
        # Add tokenizer model type (required for GGUF)
        writer.add_tokenizer_model("gpt2")
        
        # Add tokenizer pre-tokenizer type (required for GGUF)
        writer.add_tokenizer_pre("default")
        
        # Add tokenizer merges for BPE tokenizers (required for Qwen)
        self.add_tokenizer_merges(writer)
        
        # Set vocabulary AFTER setting vocabulary size and tokenizer info
        self.set_vocab(writer)
            

        
        # Write tensors like the working scripts
        self.write_tensors(writer)
        
        # Write the model
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        
        print(f"Model converted successfully to {self.output_path}")
    
    def set_gguf_parameters(self, writer: gguf.GGUFWriter):
        """Set GGUF parameters following the same pattern as working scripts."""
        writer.add_name(self.model_path.name)
        writer.add_block_count(self.config.num_hidden_layers)
        
        if hasattr(self.config, 'max_position_embeddings'):
            writer.add_context_length(self.config.max_position_embeddings)
            print(f"gguf: context length = {self.config.max_position_embeddings}")
        
        writer.add_embedding_length(self.config.hidden_size)
        print(f"gguf: embedding length = {self.config.hidden_size}")
        
        if hasattr(self.config, 'intermediate_size'):
            writer.add_feed_forward_length(self.config.intermediate_size)
            print(f"gguf: feed forward length = {self.config.intermediate_size}")
        
        writer.add_head_count(self.config.num_attention_heads)
        print(f"gguf: head count = {self.config.num_attention_heads}")
        
        if hasattr(self.config, 'num_key_value_heads'):
            writer.add_head_count_kv(self.config.num_key_value_heads)
            print(f"gguf: key-value head count = {self.config.num_key_value_heads}")
        
        if hasattr(self.config, 'rope_theta'):
            writer.add_rope_freq_base(self.config.rope_theta)
            print(f"gguf: rope theta = {self.config.rope_theta}")
        
        if hasattr(self.config, 'rms_norm_eps'):
            writer.add_layer_norm_rms_eps(self.config.rms_norm_eps)
            print(f"gguf: rms norm epsilon = {self.config.rms_norm_eps}")
        
        # Set file type to F16 like the working scripts
        writer.add_file_type(gguf.GGMLQuantizationType.F16)
        print(f"gguf: file type = F16")
    
    def set_vocab(self, writer: gguf.GGUFWriter):
        """Set vocabulary following the same pattern as working scripts."""
        print("Setting vocabulary...")
        
        try:
            # Try sentencepiece first like the working scripts
            self._set_vocab_sentencepiece(writer)
        except FileNotFoundError:
            try:
                # Fall back to GPT2-style like the working scripts
                self._set_vocab_gpt2(writer)
            except Exception as e:
                print(f"Error setting vocabulary: {e}")
                sys.exit(1)
    
    def _set_vocab_sentencepiece(self, writer: gguf.GGUFWriter):
        """Set vocabulary using sentencepiece like the working scripts."""
        vocab_path = self.model_path / "tokenizer.model"
        if not vocab_path.exists():
            raise FileNotFoundError(f"tokenizer.model not found at {vocab_path}")
        
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(vocab_path))
        
        # Get the expected vocabulary size from the model
        expected_vocab_size = 152064  # Qwen2.5-7B expected vocab size
        print(f"Expected model vocabulary size: {expected_vocab_size}")
        print(f"Tokenizers vocabulary size: {sp.vocab_size()}")
        
        # Pre-allocate the full vocabulary size with PAD tokens (like working scripts)
        tokens = [f"[PAD{i}]" for i in range(expected_vocab_size)]
        scores = [-10000.0] * expected_vocab_size
        toktypes = [1] * expected_vocab_size  # NORMAL token type
        
        # Fill in the real tokens from the tokenizer
        for i in range(sp.vocab_size()):
            token = sp.id_to_piece(i)
            score = sp.get_score(i)
            toktype = 1  # NORMAL token type
            
            tokens[i] = token
            scores[i] = score
            toktypes[i] = toktype
        
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(toktypes)
        
        # Use GGUF SpecialVocab for proper special token handling (like working scripts)
        special_vocab = gguf.SpecialVocab(self.model_path, n_vocab=len(tokens))
        special_vocab.add_to_gguf(writer)
    
    def _set_vocab_gpt2(self, writer: gguf.GGUFWriter):
        """Set vocabulary using GPT2-style like the working scripts."""
        vocab_path = self.model_path / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"vocab.json not found at {vocab_path}")
        
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        # Get the expected vocabulary size from the model
        expected_vocab_size = 152064  # Qwen2.5-7B expected vocab size
        print(f"Expected model vocabulary size: {expected_vocab_size}")
        print(f"Tokenizers vocabulary size: {len(vocab)}")
        
        # Pre-allocate the full vocabulary size with PAD tokens (like working scripts)
        tokens = [f"[PAD{i}]" for i in range(expected_vocab_size)]
        scores = [-10000.0] * expected_vocab_size
        toktypes = [1] * expected_vocab_size  # NORMAL token type
        
        # Fill in the real tokens from the tokenizer
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, token_id in sorted_vocab:
            if token_id < expected_vocab_size:
                tokens[token_id] = token
                scores[token_id] = 0.0  # Default score
                toktypes[token_id] = 1  # NORMAL token type
        
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(toktypes)
        
        # Use GGUF SpecialVocab for proper special token handling (like working scripts)
        special_vocab = gguf.SpecialVocab(self.model_path, n_vocab=len(tokens))
        special_vocab.add_to_gguf(writer)
    
    def write_tensors(self, writer: gguf.GGUFWriter):
        """Write tensors following the same pattern as working scripts."""
        print("Writing tensors...")
        
        # Get tensor mapping using the same method as working scripts
        tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN2, self.config.num_hidden_layers)
        
        # Process each layer like the working scripts
        for layer_idx in range(self.config.num_hidden_layers):
            print(f"Processing layer {layer_idx}...")
            
            # Get all tensors for this layer
            for tensor_name, data_torch in self.get_tensors():
                if f"layers.{layer_idx}." in tensor_name:
                    # Map tensor name using the same method as working scripts
                    new_name = tensor_map.get_name(tensor_name, try_suffixes=(".weight", ".bias"))
                    if new_name is None:
                        print(f"  Warning: Could not map tensor {tensor_name}")
                        continue
                    
                    # Format the bid placeholder like the working scripts
                    if "{bid}" in new_name:
                        new_name = new_name.format(bid=layer_idx)
                    
                    print(f"  Adding tensor: {tensor_name} -> {new_name}")
                    print(f"    Original shape: {data_torch.shape}")
                    
                    # Convert data type and to numpy like the working scripts
                    old_dtype = data_torch.dtype
                    
                    # Convert any unsupported data types to float32 like the working scripts
                    if data_torch.dtype not in (torch.float16, torch.float32):
                        data_torch = data_torch.to(torch.float32)
                    
                    # Convert to numpy and add like the working scripts
                    # Be careful about squeezing - only remove trailing 1s for certain tensors
                    if tensor_name.endswith((".weight", ".bias")):
                        # For weight/bias tensors, remove trailing 1s but preserve the main dimensions
                        while len(data_torch.shape) > 2 and data_torch.shape[-1] == 1:
                            data_torch = data_torch.squeeze(-1)
                        while len(data_torch.shape) > 2 and data_torch.shape[-2] == 1:
                            data_torch = data_torch.squeeze(-2)
                        
                        # Special handling for embedding tensor to ensure correct vocabulary size
                        if tensor_name == "model.embed_tokens.weight":
                            # The embedding tensor should have shape (vocab_size, hidden_size)
                            # Always ensure it's 2D regardless of original shape
                            if len(data_torch.shape) != 2:
                                print(f"    Reshaping embedding tensor from {data_torch.shape} to ({data_torch.shape[0]}, {data_torch.shape[1]})")
                                data_torch = data_torch.view(data_torch.shape[0], data_torch.shape[1])
                            print(f"    Final embedding tensor shape: {data_torch.shape}")
                    else:
                        # For other tensors, use normal squeeze
                        data_torch = data_torch.squeeze()
                    
                    data = data_torch.numpy()
                    
                    # Handle data type conversion like the working scripts
                    if data.dtype == np.float16:
                        # Convert float16 to float32 for compatibility
                        data = data.astype(np.float32)
                    
                    # Apply TL1/TL2 quantization if requested
                    if self.quant_type in ["tl1", "tl2"] and tensor_name.endswith(".weight") and len(data.shape) >= 2:
                        # Skip certain tensors that shouldn't be quantized
                        if not any(tensor_name.endswith(skip) for skip in ["norm.weight", "embed_tokens.weight", "lm_head.weight"]):
                            if self.quant_type == "tl1":
                                data, scale = self.transform_to_tl1(data)
                                print(f"    TL1 quantized: {old_dtype} -> {data.dtype}, scale: {scale}, shape: {data.shape}")
                                # Add scale tensor
                                scale_name = new_name + "_scale"
                                writer.add_tensor(scale_name, scale)
                                # Use proper GGUF quantization type
                                writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.TL1)
                                continue  # Skip the default add_tensor below
                            elif self.quant_type == "tl2":
                                data, scale = self.transform_to_tl2(data)
                                print(f"    TL2 quantized: {old_dtype} -> {data.dtype}, scale: {scale}, shape: {data.shape}")
                                # Add scale tensor
                                scale_name = new_name + "_scale"
                                writer.add_tensor(scale_name, scale)
                                # Use proper GGUF quantization type
                                writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.TL2)
                                continue  # Skip the default add_tensor below
                        else:
                            print(f"    Skipping quantization for {new_name} (norm/embed/lm_head)")
                    
                    print(f"    Data type: {old_dtype} -> {data.dtype}, shape: {data.shape}")
                    print(f"    Final tensor shape for {new_name}: {data.shape}")
                    writer.add_tensor(new_name, data)
        
        # Add embedding and output layers
        print("Adding embedding and output layers...")
        
        for tensor_name, data_torch in self.get_tensors():
            if "layers." not in tensor_name:
                # Map tensor name using the same method as working scripts
                new_name = tensor_map.get_name(tensor_name, try_suffixes=(".weight", ".bias"))
                if new_name is None:
                    print(f"  Warning: Could not map tensor {tensor_name}")
                    continue
                
                print(f"  Adding tensor: {tensor_name} -> {new_name}")
                print(f"    Original shape: {data_torch.shape}")
                
                # Convert data type and to numpy like the working scripts
                old_dtype = data_torch.dtype
                
                # Convert any unsupported data types to float32 like the working scripts
                if data_torch.dtype not in (torch.float16, torch.float32):
                    data_torch = data_torch.to(torch.float32)
                
                # Convert to numpy and add like the working scripts
                # Be careful about squeezing - only remove trailing 1s for certain tensors
                if tensor_name.endswith((".weight", ".bias")):
                    # For weight/bias tensors, remove trailing 1s but preserve the main dimensions
                    while len(data_torch.shape) > 2 and data_torch.shape[-1] == 1:
                        data_torch = data_torch.squeeze(-1)
                    while len(data_torch.shape) > 2 and data_torch.shape[-2] == 1:
                        data_torch = data_torch.squeeze(-2)
                    
                    # Special handling for embedding tensor to ensure correct vocabulary size
                    if tensor_name == "model.embed_tokens.weight":
                        # The embedding tensor should have shape (vocab_size, hidden_size)
                        # Always ensure it's 2D regardless of original shape
                        if len(data_torch.shape) != 2:
                            print(f"    Reshaping embedding tensor from {data_torch.shape} to ({data_torch.shape[0]}, {data_torch.shape[1]})")
                            data_torch = data_torch.view(data_torch.shape[0], data_torch.shape[1])
                        print(f"    Final embedding tensor shape: {data_torch.shape}")
                else:
                    # For other tensors, use normal squeeze
                    data_torch = data_torch.squeeze()
                
                data = data_torch.numpy()
                
                # Handle data type conversion like the working scripts
                if data.dtype == np.float16:
                    # Convert float16 to float32 for compatibility
                    data = data.astype(np.float32)
                
                # Apply TL1/TL2 quantization if requested
                if self.quant_type in ["tl1", "tl2"] and tensor_name.endswith(".weight") and len(data.shape) >= 2:
                    # Skip certain tensors that shouldn't be quantized
                    if not any(tensor_name.endswith(skip) for skip in ["norm.weight", "embed_tokens.weight", "lm_head.weight"]):
                        if self.quant_type == "tl1":
                            data, scale = self.transform_to_tl1(data)
                            print(f"    TL1 quantized: {old_dtype} -> {data.dtype}, scale: {scale}, shape: {data.shape}")
                            # Add scale tensor
                            scale_name = new_name + "_scale"
                            writer.add_tensor(scale_name, scale)
                            # Use proper GGUF quantization type
                            writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.TL1)
                            continue  # Skip the default add_tensor below
                        elif self.quant_type == "tl2":
                            data, scale = self.transform_to_tl2(data)
                            print(f"    TL2 quantized: {old_dtype} -> {data.dtype}, scale: {scale}, shape: {data.shape}")
                            # Add scale tensor
                            scale_name = new_name + "_scale"
                            writer.add_tensor(scale_name, scale)
                            # Use proper GGUF quantization type
                            writer.add_tensor(new_name, data, raw_dtype=gguf.GGMLQuantizationType.TL2)
                            continue  # Skip the default add_tensor below
                    else:
                        print(f"    Skipping quantization for {new_name} (norm/embed/lm_head)")
                
                print(f"    Data type: {old_dtype} -> {data.dtype}, shape: {data.shape}")
                writer.add_tensor(new_name, data)
    
    def get_tensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Get model tensors like the working scripts."""
        # Check if using safetensors
        is_safetensors = any(f.suffix == '.safetensors' for f in self.model_path.glob('*.safetensors'))
        
        if is_safetensors:
            # Use safetensors like the working scripts
            for f in self.model_path.glob('*.safetensors'):
                with safe_open(f, framework="pt", device="cpu") as model_part:
                    for name in model_part.keys():
                        data = model_part.get_tensor(name)
                        yield name, data
        else:
            # Use regular torch loading like the working scripts
            for f in self.model_path.glob('*.bin'):
                model_part = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
                for name, data in model_part.items():
                    yield name, data

def main():
    parser = argparse.ArgumentParser(description="Convert Qwen2.5 models to GGUF format for BitNet LUT inference")
    parser.add_argument("model_path", help="Path to the Qwen2.5 model directory")
    parser.add_argument("--output", "-o", default="qwen25-bitnet.gguf", help="Output GGUF file path")
    parser.add_argument("--quant-type", default="f16", choices=["f16", "tl1", "tl2"], help="Quantization type")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        sys.exit(1)
    
    # Create converter and run conversion
    converter = Qwen25Converter(args.model_path, args.output, args.quant_type)
    converter.load_model()
    converter.convert_model()

if __name__ == "__main__":
    main()
