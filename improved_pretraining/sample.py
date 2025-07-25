"""Sample from a trained model checkpoint"""
"""python sample.py checkpoints/best_model.pt --prompt "The king said" --temperature 0.5 --top-k 30"""

import torch
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from baseline_pretraining.model import TinyTransformer
from tokenizer import load_tokenizer


def sample_from_model(
    checkpoint_path: str,
    tokenizer_path: str,
    prompt: str = "The quick brown fox",
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = 'cuda'
):
    """Generate text from a model checkpoint"""
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config (handle both old and new checkpoint formats)
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config if not in checkpoint
        config = {
            'vocab_size': 8192,
            'd_model': 192,
            'n_layers': 9,
            'n_heads': 3,
            'd_ff': 384,
            'max_seq_len': 1024,
            'dropout': 0.1
        }
    
    # Create model
    model = TinyTransformer(
        vocab_size=config.get('vocab_size', 8192),
        d_model=config.get('d_model', 192),
        n_layers=config.get('n_layers', 9),
        n_heads=config.get('n_heads', 3),
        d_ff=config.get('d_ff', 384),
        max_seq_len=config.get('max_seq_len', 1024),
        dropout=0.0  # No dropout during inference
    ).to(device)
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    # Handle DataParallel wrapped models
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"\nModel loaded successfully!")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt.lower())  # Model was trained on lowercase
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating with temperature={temperature}, top_k={top_k}")
    print("-" * 50)
    
    # Generate (model doesn't support top_p yet)
    generated = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k
    )
    
    # Decode
    output_text = tokenizer.decode(generated[0].tolist())
    print(output_text)
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description='Sample from trained model')
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('--tokenizer', default='../gutenberg_tokenizer_8192', 
                        help='Path to tokenizer (default: ../gutenberg_tokenizer_8192)')
    parser.add_argument('--prompt', '-p', default='The quick brown fox', 
                        help='Prompt text')
    parser.add_argument('--max-length', '-m', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                        help='Sampling temperature (0.0-2.0)')
    parser.add_argument('--top-k', '-k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling')
    parser.add_argument('--device', default='cuda', 
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    sample_from_model(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device
    )


if __name__ == "__main__":
    main()