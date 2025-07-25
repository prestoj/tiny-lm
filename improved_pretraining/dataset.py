"""
Dataset for loading tokenized Gutenberg books
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random


class TokenizedGutenbergDataset(Dataset):
    """Dataset for pre-tokenized Gutenberg books stored as numpy arrays"""
    
    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 2048,
        split: str = 'train',
        train_split: float = 0.95
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.split = split
        
        # Get all token files
        self.token_files = sorted(list(self.data_dir.glob('*.npy')))
        
        # Split into train/val
        n_files = len(self.token_files)
        n_train = int(n_files * train_split)
        
        if split == 'train':
            self.token_files = self.token_files[:n_train]
        else:
            self.token_files = self.token_files[n_train:]
        
        print(f"Loaded {len(self.token_files)} files for {split} split")
        
        # Load all tokens into memory for efficiency
        print(f"Loading tokens into memory...")
        self.all_tokens = []
        for file_path in self.token_files[:100]:  # Start with first 100 files
            tokens = np.load(file_path)
            self.all_tokens.extend(tokens.tolist())
        
        self.all_tokens = np.array(self.all_tokens, dtype=np.int32)
        print(f"Total tokens loaded: {len(self.all_tokens):,}")
        
        # Calculate number of sequences
        self.n_sequences = len(self.all_tokens) // max_seq_len
        
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        # Get sequence from concatenated tokens
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len
        
        tokens = self.all_tokens[start_idx:end_idx]
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # For language modeling, labels are the same as inputs
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class StreamingTokenizedDataset(Dataset):
    """Memory-efficient dataset that loads files on demand"""
    
    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 2048,
        split: str = 'train',
        train_split: float = 0.95,
        buffer_size: int = 10
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.split = split
        self.buffer_size = buffer_size
        
        # Get all token files
        self.token_files = sorted(list(self.data_dir.glob('*.npy')))
        
        # Split into train/val
        n_files = len(self.token_files)
        n_train = int(n_files * train_split)
        
        if split == 'train':
            self.token_files = self.token_files[:n_train]
        else:
            self.token_files = self.token_files[n_train:]
        
        print(f"Found {len(self.token_files)} files for {split} split")
        
        # Calculate sequences per file (approximate)
        self.sequences_per_file = []
        self.cumulative_sequences = [0]
        total_sequences = 0
        
        print("Calculating dataset size...")
        for file_path in self.token_files[:100]:  # Sample first 100 files
            tokens = np.load(file_path)
            n_seq = len(tokens) // max_seq_len
            self.sequences_per_file.append(n_seq)
            total_sequences += n_seq
            self.cumulative_sequences.append(total_sequences)
        
        # Estimate total sequences
        avg_sequences = np.mean(self.sequences_per_file)
        self.total_sequences = int(avg_sequences * len(self.token_files))
        
        # Buffer for loaded files
        self.buffer = {}
        self.buffer_order = []
        
    def __len__(self):
        return self.total_sequences
    
    def _load_file(self, file_idx):
        """Load a file into buffer"""
        if file_idx in self.buffer:
            return self.buffer[file_idx]
        
        # Load file
        file_path = self.token_files[file_idx]
        tokens = np.load(file_path)
        
        # Add to buffer
        self.buffer[file_idx] = tokens
        self.buffer_order.append(file_idx)
        
        # Remove oldest if buffer is full
        if len(self.buffer) > self.buffer_size:
            oldest = self.buffer_order.pop(0)
            del self.buffer[oldest]
        
        return tokens
    
    def __getitem__(self, idx):
        # Random access - find which file contains this sequence
        file_idx = idx % len(self.token_files)
        
        # Load file
        tokens = self._load_file(file_idx)
        
        # Get random sequence from file
        if len(tokens) >= self.max_seq_len:
            start_idx = random.randint(0, len(tokens) - self.max_seq_len)
            sequence = tokens[start_idx:start_idx + self.max_seq_len]
        else:
            # Pad if needed
            sequence = np.pad(tokens, (0, self.max_seq_len - len(tokens)))
        
        # Convert to tensor
        input_ids = torch.tensor(sequence, dtype=torch.long)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    max_seq_len: int = 2048,
    num_workers: int = 4,
    streaming: bool = True,
    rank: int = 0,
    world_size: int = 1
):
    """Create train and validation dataloaders with optional DistributedSampler support"""
    
    DatasetClass = StreamingTokenizedDataset if streaming else TokenizedGutenbergDataset
    
    train_dataset = DatasetClass(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        split='train'
    )
    
    val_dataset = DatasetClass(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        split='val'
    )
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using DistributedSampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    dataset = StreamingTokenizedDataset(
        data_dir="/home/preston/git/tiny-bot/tokenized_gutenberg_8k",
        max_seq_len=2048,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading
    sample = dataset[0]
    print(f"Sample input shape: {sample['input_ids'].shape}")
    print(f"Sample labels shape: {sample['labels'].shape}")