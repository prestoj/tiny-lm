#!/usr/bin/env python3
"""Load and inspect tokenized numpy arrays"""

import numpy as np
import sys
from pathlib import Path

def load_tokens(file_path):
    """Load tokenized data from numpy file"""
    tokens = np.load(file_path)
    
    print(f"File: {file_path}")
    print(f"Shape: {tokens.shape}")
    print(f"Data type: {tokens.dtype}")
    print(f"Total tokens: {len(tokens):,}")
    print(f"Min token ID: {tokens.min()}")
    print(f"Max token ID: {tokens.max()}")
    print(f"First 100 tokens: {tokens[:100]}")
    
    return tokens

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to first file in tokenized directory
        tokenized_dir = Path("tokenized_gutenberg_8k_clean")
        if tokenized_dir.exists():
            npy_files = list(tokenized_dir.glob("*.npy"))
            if npy_files:
                file_path = np.random.choice(npy_files)
                print(f"Using first file found: {file_path}\n")
            else:
                print("No .npy files found in tokenized_gutenberg_8k_clean/")
                sys.exit(1)
        else:
            print("Usage: python load_tokens.py <path_to_tokens.npy>")
            sys.exit(1)
    
    tokens = load_tokens(file_path)