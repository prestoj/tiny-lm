"""
Complete example showing how to train the 10M parameter modern transformer.
This includes a simple BPE tokenizer implementation and a full training loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
import re
from tqdm import tqdm
import json
import os


class SimpleBPETokenizer:
    """Simple Byte-Pair Encoding tokenizer for demonstration"""
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.word_tokenizer = re.compile(r'\b\w+\b|[^\w\s]')
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        self.token_to_id = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def train(self, texts: list):
        """Train BPE on texts (simplified version)"""
        # For demonstration, we'll use a character-level tokenizer
        # In practice, you'd use a proper BPE implementation
        
        # Collect all characters
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Build vocabulary
        for i, char in enumerate(sorted(chars)):
            if len(self.token_to_id) < self.vocab_size:
                self.token_to_id[char] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = char

    def encode(self, text: str, max_length: Optional[int] = None, truncation: bool = True) -> list:
        """Encode text to token ids"""
        tokens = [self.bos_token_id]
        
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.unk_token_id)
        
        tokens.append(self.eos_token_id)
        
        if max_length and truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.eos_token_id]
        
        return tokens

    def decode(self, token_ids: list) -> str:
        """Decode token ids to text"""
        text = ''
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                    text += token
        return text


def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    # In practice, you'd load a real dataset
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Transformers have revolutionized natural language processing.",
        "Attention is all you need for sequence modeling.",
        "Deep learning models continue to improve rapidly.",
    ] * 200  # Repeat to create a larger dataset
    
    return texts


def train_model():
    """Complete training example"""
    # Import our modules (assuming they're in the same directory)
    from modern_transformer import ModernTransformer
    from transformer_training_utils import (
        TransformerTrainer, CosineWarmupScheduler, 
        LabelSmoothingCrossEntropy, TextDataset, create_training_config
    )
    
    # Configuration
    config = create_training_config()
    config['batch_size'] = 8  # Smaller for demonstration
    config['max_steps'] = 100  # Fewer steps for demonstration
    config['eval_interval'] = 20
    config['warmup_steps'] = 10
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create tokenizer and dataset
    print("Preparing dataset...")
    texts = create_sample_dataset()
    tokenizer = SimpleBPETokenizer(vocab_size=8192)
    tokenizer.train(texts)
    
    # Create datasets
    train_texts = texts[:int(0.9 * len(texts))]
    val_texts = texts[int(0.9 * len(texts)):]
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=128)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=128)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    # Create model
    print("Creating model...")
    model = ModernTransformer(
        vocab_size=8192,
        d_model=320,
        n_layers=4,
        n_heads=8,
        d_ff=1280,
        n_kv_heads=4,
        max_seq_len=2048,
        dropout=0.1,
        tie_embeddings=True
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create trainer
    trainer = TransformerTrainer(
        model, 
        device=device,
        mixed_precision=config['mixed_precision'],
        compile_model=False  # Disable for demonstration
    )
    
    # Create optimizer and scheduler
    optimizer = trainer.create_optimizer(
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=config['warmup_steps'],
        max_steps=config['max_steps'],
        base_lr=config['learning_rate']
    )
    
    # Loss function
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    
    # Training loop
    print("Starting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(10):  # You'd typically use more epochs
        print(f"\nEpoch {epoch + 1}")
        
        # Training
        model.train()
        train_losses = []
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Training step
            metrics = trainer.train_step(
                input_ids, labels, optimizer, criterion, 
                max_grad_norm=config['max_grad_norm']
            )
            
            train_losses.append(metrics['loss'])
            scheduler.step()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'ppl': f"{metrics['perplexity']:.2f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Evaluation
            if global_step % config['eval_interval'] == 0:
                val_metrics = trainer.evaluate(val_loader, criterion)
                print(f"\nValidation - Loss: {val_metrics['loss']:.4f}, "
                      f"Perplexity: {val_metrics['perplexity']:.2f}")
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'val_loss': best_val_loss,
                        'config': config
                    }, 'best_model.pt')
                    print("Saved best model!")
            
            if global_step >= config['max_steps']:
                break
        
        if global_step >= config['max_steps']:
            break
        
        # Epoch summary
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch + 1} - Avg train loss: {avg_train_loss:.4f}")
    
    print("\nTraining completed!")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    
    # Generate some text
    prompt = "The future of AI"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    generated_ids = trainer.generate(
        prompt_ids,
        max_length=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    return model, tokenizer


def benchmark_model():
    """Benchmark the model's inference speed"""
    from modern_transformer import ModernTransformer
    import time
    
    model = ModernTransformer()
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Warmup
    for _ in range(10):
        input_ids = torch.randint(0, 8192, (1, 128), device=device)
        with torch.no_grad():
            _ = model(input_ids)
    
    # Benchmark different sequence lengths
    seq_lengths = [128, 256, 512, 1024]
    batch_sizes = [1, 8, 16, 32]
    
    print("\nInference Speed Benchmark:")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'Seq Length':<12} {'Tokens/sec':<15} {'Latency (ms)':<15}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            input_ids = torch.randint(0, 8192, (batch_size, seq_len), device=device)
            
            # Time multiple runs
            num_runs = 50
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(input_ids)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            time_per_run = total_time / num_runs
            tokens_per_sec = (batch_size * seq_len) / time_per_run
            latency_ms = time_per_run * 1000
            
            print(f"{batch_size:<12} {seq_len:<12} {tokens_per_sec:<15.1f} {latency_ms:<15.2f}")


if __name__ == "__main__":
    # Run training
    model, tokenizer = train_model()
    
    # Run benchmarks
    print("\n" + "="*60)
    benchmark_model()