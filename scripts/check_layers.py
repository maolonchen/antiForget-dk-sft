#!/usr/bin/env python
"""
Script to check layers a model has.
"""

import argparse
from transformers import AutoConfig, AutoModelForCausalLM


def check_model_layers(model_path):
    """
    Check how many layers a model has.
    
    Args:
        model_path: Path to the model directory or Hugging Face model identifier
    """
    print(f"Loading model configuration from: {model_path}")
    
    try:
        # Load model configuration
        config = AutoConfig.from_pretrained(model_path)
        
        # Get the number of hidden layers
        num_layers = getattr(config, 'num_hidden_layers', None)
        if num_layers is None:
            # Some models might use different attribute names
            num_layers = getattr(config, 'n_layers', None)
            if num_layers is None:
                num_layers = getattr(config, 'num_layers', None)
        
        # Get other relevant info
        hidden_size = getattr(config, 'hidden_size', None)
        num_attention_heads = getattr(config, 'num_attention_heads', None)
        vocab_size = getattr(config, 'vocab_size', None)
        model_type = getattr(config, 'model_type', 'Unknown')
        
        print(f"Model Type: {model_type}")
        print(f"Number of Layers: {num_layers}")
        print(f"Hidden Size: {hidden_size}")
        print(f"Number of Attention Heads: {num_attention_heads}")
        print(f"Vocabulary Size: {vocab_size}")
        
        # Also load the actual model to double-check
        print("\nLoading full model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype='auto',
            trust_remote_code=True
        )
        
        actual_num_layers = model.config.num_hidden_layers
        print(f"Actual number of layers in loaded model: {actual_num_layers}")
        
        # Show model structure briefly
        print(f"\nModel structure:")
        print(f"- Model type: {type(model).__name__}")
        if hasattr(model, 'transformer'):
            # For GPT-like models
            if hasattr(model.transformer, 'h'):
                print(f"- Transformer blocks: {len(model.transformer.h)}")
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # For models like Llama, Qwen
            if hasattr(model.model, 'layers'):
                print(f"- Decoder layers: {len(model.model.layers)}")
        elif hasattr(model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            # For encoder-decoder models
            print(f"- Decoder layers: {len(model.model.decoder.layers)}")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check how many layers a model has")
    parser.add_argument(
        "model_path", 
        type=str, 
        help="Path to the model directory or Hugging Face model identifier"
    )
    
    args = parser.parse_args()
    
    check_model_layers(args.model_path)
