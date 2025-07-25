#!/usr/bin/env python3
"""Inspect a trained tokenizer - show how text is tokenized"""

import argparse
from tokenizer import load_tokenizer


def inspect_tokenization(tokenizer_path: str, text: str = None):
    """Show detailed tokenization of text"""
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Loaded tokenizer from: {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Default texts if none provided
    if not text:
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello, world! How are you today?",
            "Machine learning is transforming the world.",
        ]
    else:
        test_texts = [text]
    
    for text in test_texts:
        print(f"\n{'='*60}")
        print(f"Text: {text}")
        print(f"{'='*60}")
        
        # Encode
        token_ids = tokenizer.encode(text)
        
        # Get token strings
        tokens = []
        for tid in token_ids:
            # Decode each token individually
            token_str = tokenizer.tokenizer.decode([tid])
            tokens.append(token_str)
        
        # Display results
        print(f"\nNumber of tokens: {len(token_ids)}")
        print(f"\nToken IDs: {token_ids}")
        
        print(f"\nTokenization:")
        for i, (tid, token) in enumerate(zip(token_ids, tokens)):
            # Show special tokens differently
            if token in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']:
                print(f"  [{i:3d}] {tid:5d} => {token}")
            else:
                # Show regular tokens with visible spaces
                print(f"  [{i:3d}] {tid:5d} => '{token}'")
        
        # Show concatenated result
        print(f"\nConcatenated: {''.join(tokens)}")
        
        # Decode back
        decoded = tokenizer.decode(token_ids)
        print(f"\nDecoded: {decoded}")


def show_vocabulary_sample(tokenizer_path: str, n: int = 100):
    """Show a sample of the vocabulary"""
    
    tokenizer = load_tokenizer(tokenizer_path)
    
    print(f"\n{'='*60}")
    print(f"Vocabulary Sample (first {n} non-special tokens)")
    print(f"{'='*60}")
    
    # Get vocabulary
    vocab = tokenizer.tokenizer.get_vocab()
    
    # Sort by token ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # Skip special tokens and show first n
    count = 0
    for token, tid in sorted_vocab:
        if token not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
            print(f"{tid:5d}: '{token}'")
            count += 1
            if count >= n:
                break


def main():
    parser = argparse.ArgumentParser(description='Inspect tokenizer')
    parser.add_argument('tokenizer', help='Path to tokenizer')
    parser.add_argument('--text', '-t', help='Text to tokenize')
    parser.add_argument('--vocab', '-v', action='store_true', help='Show vocabulary sample')
    parser.add_argument('--vocab-size', '-n', type=int, default=100, help='Number of vocab items to show')
    
    args = parser.parse_args()
    
    # Inspect tokenization
    inspect_tokenization(args.tokenizer, args.text)
    
    # Show vocabulary if requested
    if args.vocab:
        show_vocabulary_sample(args.tokenizer, args.vocab_size)


if __name__ == "__main__":
    main()