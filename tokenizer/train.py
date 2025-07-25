"""Script to train a tokenizer on text data"""

import argparse
from pathlib import Path

from .tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (text file or directory)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Vocabulary size (default: 8192)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer",
        help="Output path for tokenizer (default: 'tokenizer')",
    )
    
    args = parser.parse_args()
    
    # Handle directory input
    data_path = Path(args.data)
    if data_path.is_dir():
        # Get all text files in directory
        text_files = list(data_path.glob("*.txt"))
        if not text_files:
            print(f"No .txt files found in {args.data}")
            return
        print(f"Found {len(text_files)} text files for training")
        training_files = [str(f) for f in text_files]
    else:
        # Single file
        training_files = args.data
    
    # Train tokenizer
    tokenizer = train_tokenizer(
        texts_file=training_files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_path=args.output,
    )
    
    print(f"\nTokenizer training complete!")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Saved to: {args.output}")
    
    # Test encoding/decoding
    test_text = "This is a test sentence to verify the tokenizer works correctly."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nTest encoding:")
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens[:20]}..." if len(tokens) > 20 else f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")


if __name__ == "__main__":
    main()