"""Dataset tokenization utilities"""

import json
import multiprocessing as mp
import numpy as np
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from tokenizer import load_tokenizer
from clean_text import load_and_clean_books


def process_text(args):
    """Process a single text"""
    text_idx, text, tokenizer_path, output_path, max_length, output_format = args
    
    # Load tokenizer for this worker
    tokenizer = load_tokenizer(tokenizer_path)
    
    try:
        # Split into chunks if text is very large
        chunk_size = 1000000  # characters per chunk
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        all_tokens = []
        for chunk in chunks:
            # Tokenize chunk
            tokens = tokenizer.encode(chunk)
            
            # Apply max_length if specified
            if max_length:
                tokens = tokens[:max_length]
            
            all_tokens.append(tokens)
        
        # Save tokenized data
        if output_format == "numpy":
            # Save as numpy array (more efficient)
            output_file = output_path / f"book_{text_idx:05d}_tokens.npy"
            # Flatten all tokens into one array
            flat_tokens = [token for chunk_tokens in all_tokens for token in chunk_tokens]
            np.save(output_file, np.array(flat_tokens, dtype=np.uint16))
        else:
            # Save as JSON (more readable)
            output_file = output_path / f"book_{text_idx:05d}_tokens.json"
            with open(output_file, 'w') as f:
                json.dump(all_tokens, f)
        
        return sum(len(tokens) for tokens in all_tokens)
        
    except Exception as e:
        print(f"Error processing text {text_idx}: {e}")
        return 0


def tokenize_dataset(
    tokenizer_path: str,
    input_dir: str,
    output_dir: str,
    metadata_path: str,
    max_length: Optional[int] = None,
    num_workers: Optional[int] = None,
    output_format: str = "numpy"
) -> None:
    """
    Tokenize cleaned English books from Project Gutenberg
    
    Args:
        tokenizer_path: Path to trained tokenizer
        input_dir: Directory containing Gutenberg text files
        output_dir: Directory to save tokenized files
        metadata_path: Path to Gutenberg metadata CSV file
        max_length: Optional maximum sequence length
        num_workers: Number of parallel workers (default: CPU count)
        output_format: Output format - "numpy" (efficient) or "json" (readable)
    """
    # Setup paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and clean English books only
    texts = load_and_clean_books(
        books_dir=input_path,
        metadata_path=metadata_path
    )
    
    print(f"Found {len(texts)} texts to tokenize")
    
    # Process texts in parallel
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Prepare arguments for each text
    args_list = [(idx, text, tokenizer_path, output_path, max_length, output_format) 
                 for idx, text in enumerate(texts)]
    
    with mp.Pool(num_workers) as pool:
        total_tokens = sum(tqdm(
            pool.imap(process_text, args_list),
            total=len(texts),
            desc="Tokenizing texts"
        ))
    
    print(f"\nTokenization complete!")
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Output saved to: {output_path}")


def tokenize_text(tokenizer_path: str, text: str) -> List[int]:
    """Simple function to tokenize a single text"""
    tokenizer = load_tokenizer(tokenizer_path)
    return tokenizer.encode(text)

if __name__ == "__main__":
    tokenize_dataset(
        tokenizer_path='../gutenberg_tokenizer_1024',
        input_dir='/media/preston/one tiny tb/data/gutenberg/books',
        output_dir='../tokenized_gutenberg_1k_clean',
        metadata_path='/media/preston/one tiny tb/data/gutenberg/gutenberg_metadata.csv',
        num_workers=8,
        output_format='numpy'
    )