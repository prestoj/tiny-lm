"""BPE Tokenizer using HuggingFace tokenizers library"""

import json
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from tokenizers.normalizers import Lowercase
from transformers import PreTrainedTokenizerFast


class BPETokenizer:
    """BPE tokenizer wrapper using HuggingFace tokenizers"""
    
    def __init__(self, tokenizer_path: Optional[str] = None):
        """Initialize tokenizer, optionally loading from path"""
        if tokenizer_path:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = None
            
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Train or load a tokenizer first.")
        return self.tokenizer.encode(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Train or load a tokenizer first.")
        return self.tokenizer.decode(ids)
    
    def encode_batch(self, texts: List[str], **kwargs) -> dict:
        """Encode multiple texts at once"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Train or load a tokenizer first.")
        return self.tokenizer(texts, **kwargs)
    
    def save(self, path: str):
        """Save tokenizer to directory"""
        if not self.tokenizer:
            raise ValueError("No tokenizer to save")
        self.tokenizer.save_pretrained(path)
        
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        if not self.tokenizer:
            return 0
        return len(self.tokenizer)


def train_tokenizer(
    texts_file: Union[str, List[str]],
    vocab_size: int = 8192,
    min_frequency: int = 2,
    save_path: Optional[str] = None
) -> BPETokenizer:
    """
    Train a BPE tokenizer on text data
    
    Args:
        texts_file: Path to text file or list of paths for training
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token to be included
        save_path: Optional path to save the trained tokenizer
        
    Returns:
        Trained BPETokenizer instance
    """
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    
    # Normalize to lowercase
    tokenizer.normalizer = Lowercase()
    
    # Use ByteLevel pre-tokenizer (same as GPT-2)
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
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Handle single file or list of files
    if isinstance(texts_file, str):
        files = [texts_file]
    else:
        files = texts_file
    
    # Train the tokenizer
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    tokenizer.train(files=files, trainer=trainer)
    
    # Post-processor for ByteLevel
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Convert to HuggingFace format
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unk|>",
        pad_token="<|padding|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )
    
    # Add the additional special tokens
    hf_tokenizer.add_special_tokens({
        'additional_special_tokens': ['<|user|>', '<|ai|>']
    })
    
    # Create BPETokenizer instance
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.tokenizer = hf_tokenizer
    
    # Save if path provided
    if save_path:
        bpe_tokenizer.save(save_path)
        print(f"Tokenizer saved to {save_path}")
    
    return bpe_tokenizer


def load_tokenizer(path: str) -> BPETokenizer:
    """Load a trained tokenizer from path"""
    return BPETokenizer(tokenizer_path=path)