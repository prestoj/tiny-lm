#!/usr/bin/env python3
"""Train a BPE tokenizer with proper space handling (GPT-2 style)"""

import os
import sys
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from tokenizers.normalizers import Lowercase
from transformers import PreTrainedTokenizerFast
from clean_text import clean_gutenberg_text, load_and_clean_books


def train_tokenizer_from_memory(
    texts,
    vocab_size: int = 8192,
    save_path: str = None
):
    """Train tokenizer with GPT-2 style space handling"""
    
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    
    # Normalize to lowercase
    tokenizer.normalizer = Lowercase()
    
    # Use ByteLevel pre-tokenizer (same as GPT-2)
    # This adds Ġ prefix to tokens that start with spaces
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # ByteLevel decoder to properly reconstruct spaces
    tokenizer.decoder = decoders.ByteLevel()
    
    # Configure trainer with all special tokens
    special_tokens = [
        "<|endoftext|>",  # EOS/BOS token (GPT-2 style)
        "<|padding|>",    # Padding token
        "<|unk|>",        # Unknown token
        "<|user|>",       # User message marker
        "<|ai|>",         # AI message marker
    ]
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Train tokenizer
    print(f"\nTraining BPE tokenizer with vocab_size={vocab_size}...")
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # Post-processor to add special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Convert to HuggingFace format
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unk|>",
        pad_token="<|padding|>",
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",  # GPT-2 uses same token for BOS/EOS
    )
    
    # Add the additional special tokens
    hf_tokenizer.add_special_tokens({
        'additional_special_tokens': ['<|user|>', '<|ai|>']
    })
    
    if save_path:
        hf_tokenizer.save_pretrained(save_path)
        print(f"Tokenizer saved to {save_path}")
    
    return hf_tokenizer


def test_tokenizer(tokenizer):
    """Test the tokenizer with various inputs"""
    print("\n" + "="*60)
    print("Testing tokenizer...")
    print("="*60)
    
    test_cases = [
        "Hello world",
        "The quick brown fox",
        "blood coagulation process",
        "The coagulation process",
        "Multiple   spaces   test",
        "New\nlines\ntest",
        "\n\nParagraph breaks\n\nTest",
        "<|user|>Hello<|ai|>Hi there!",  # Test special tokens
    ]
    
    for text in test_cases:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        # Get individual tokens
        token_strs = []
        for tid in tokens:
            token_str = tokenizer.decode([tid])
            token_strs.append(repr(token_str))
        
        print(f"\nOriginal: {repr(text)}")
        print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"Token strings: {' '.join(token_strs[:10])}{'...' if len(token_strs) > 10 else ''}")
        print(f"Decoded: {repr(decoded)}")
        print(f"Exact match? {text == decoded}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train tokenizer with proper space handling')
    parser.add_argument(
        '--books-dir',
        default='/media/preston/one tiny tb/data/gutenberg/books',
        help='Directory containing Gutenberg books'
    )
    parser.add_argument(
        '--metadata',
        default='/media/preston/one tiny tb/data/gutenberg/gutenberg_metadata.csv',
        help='Path to Gutenberg metadata CSV'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=8192,
        help='Vocabulary size (default: 8192)'
    )
    parser.add_argument(
        '--output',
        default='gutenberg_tokenizer_fixed',
        help='Output path for tokenizer'
    )
    parser.add_argument(
        '--max-books',
        type=int,
        default=None,
        help='Maximum number of books to load (default: all)'
    )
    
    args = parser.parse_args()
    
    # Load books
    print("Step 1: Loading books into memory")
    texts = load_and_clean_books(
        books_dir=args.books_dir,
        metadata_path=args.metadata,
        max_books=args.max_books
    )
    
    # Train tokenizer
    print("\nStep 2: Training tokenizer")
    tokenizer = train_tokenizer_from_memory(
        texts=texts,
        vocab_size=args.vocab_size,
        save_path=args.output
    )
    
    # Test it
    test_tokenizer(tokenizer)
    
    print(f"\n✅ Tokenizer training complete!")
    print(f"Saved to: {args.output}")
    
    # Print special token info
    print(f"\nSpecial tokens:")
    print(f"  PAD: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"  UNK: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
    print(f"  BOS: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    print(f"  EOS: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"  Additional: {tokenizer.additional_special_tokens}")
    
    # Clear memory
    del texts
    print("Cleared texts from memory")


if __name__ == "__main__":
    main()