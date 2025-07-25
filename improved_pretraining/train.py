"""
Training script for baseline transformer model with DistributedDataParallel
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TinyTransformer
from dataset import create_dataloaders
from muon import Muon, SingleDeviceMuonWithAuxAdam


def setup_ddp():
    """Initialize DDP"""
    # Get local rank from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Set device based on local rank
    torch.cuda.set_device(local_rank)
    
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    """Clean up DDP"""
    dist.destroy_process_group()


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup, then constant LR (no decay as requested)"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    # Create a scheduler that applies to all parameter groups
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, config, 
                global_step, output_dir, val_loader, rank):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    accumulated_loss = 0
    
    # Only show progress bar on rank 0
    if rank == 0:
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        progress_bar = train_loader
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Mixed precision training
        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(input_ids, labels)
        
        # Scale loss by accumulation steps
        loss = loss / config.gradient_accumulation_steps
        accumulated_loss += loss.item()
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Optimizer step every gradient_accumulation_steps
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping - unscale for all optimizers
            scaler.unscale_(optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            
            scaler.update()
            
            # Zero gradients for all optimizers
            optimizer.zero_grad()
            
            # Step scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Momentum warmup for Muon (if using Muon)
            if global_step < config.muon_momentum_warmup:
                frac = min(global_step / config.muon_momentum_warmup, 1.0)
            
            # Update window sizes every 10000 steps
            if global_step > 0 and global_step % 10000 == 0:
                # Calculate new window size (grow by 128 each time)
                steps_per_increase = 10000
                increases = global_step // steps_per_increase
                new_window_size = 128 + (increases * 128)
                new_window_size = min(new_window_size, config.max_seq_len)
                
                # Update model window sizes
                model.module.update_window_sizes(new_window_size)
                
                if rank == 0:
                    print(f"\nStep {global_step}: Updated global attention window size to {new_window_size}")
            
            # Reset accumulated loss
            accumulated_loss = 0
        
        # Update metrics (using unscaled loss)
        actual_loss = loss.item() * config.gradient_accumulation_steps
        total_loss += actual_loss * input_ids.size(0)
        total_tokens += input_ids.numel()
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            avg_loss = total_loss / ((step + 1) * input_ids.size(0))
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'ppl': f"{np.exp(avg_loss):.2f}",
                'lr': f"{current_lr:.2e}",
                'acc_step': f"{(step + 1) % config.gradient_accumulation_steps}/{config.gradient_accumulation_steps}"
            })
        
        # Log to wandb (only on rank 0)
        if rank == 0 and step % config.log_interval == 0:
            wandb.log({
                'train/loss': actual_loss,
                'train/perplexity': np.exp(actual_loss),
                'train/learning_rate': current_lr,
                'train/tokens': total_tokens,
                'global_step': global_step,
            })
        
        # Increment global step
        global_step += 1
        
        # Save checkpoint at intervals (only on rank 0)
        if rank == 0 and global_step % config.save_interval == 0:
            print(f"\nSaving checkpoint at step {global_step}")
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'config': vars(config)
            }
            torch.save(checkpoint, output_dir / f'checkpoint_step_{global_step}.pt')
    
    return total_loss / len(train_loader), global_step


def main():
    parser = argparse.ArgumentParser(description='Train baseline transformer model with DDP')
    
    # Model arguments
    parser.add_argument('--vocab-size', type=int, default=8192, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=384, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--n-kv-heads', type=int, default=None, help='Number of KV heads for GQA (defaults to n_heads)')
    parser.add_argument('--d-ff', type=int, default=1536, help='Feed-forward dimension')
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with tokenized data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Base learning rate')
    parser.add_argument('--embedding-lr', type=float, default=None, help='Learning rate for embeddings (defaults to base LR)')
    parser.add_argument('--head-lr', type=float, default=None, help='Learning rate for lm_head (defaults to base LR)')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N steps')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Evaluate every N steps')
    parser.add_argument('--save-interval', type=int, default=5000, help='Save checkpoint every N steps')
    
    # Muon optimizer arguments
    parser.add_argument('--muon-lr-scale', type=float, default=0.1, help='LR scale for Muon vs base LR')
    parser.add_argument('--muon-momentum', type=float, default=0.95, help='Momentum for Muon optimizer')
    parser.add_argument('--muon-momentum-warmup', type=int, default=300, help='Steps for Muon momentum warmup')
    parser.add_argument('--muon-ns-steps', type=int, default=5, help='Newton-Schulz iterations for Muon')
    
    # Other arguments
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--wandb-project', type=str, default='tiny-bot-baseline', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Setup DDP
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    # Set random seeds (different for each process)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # Only initialize wandb on rank 0
    if rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        config = wandb.config
    else:
        config = argparse.Namespace(**vars(args))
    
    # Create output directory (only on rank 0)
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir)
    
    # Create model
    model = TinyTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        pad_token_id=0
    ).to(device)
    
    # Print model param count (only on rank 0)
    if rank == 0:
        print(f"Model parameters: {model.num_parameters():,}")
        print(f"Using DDP with {world_size} GPUs")
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Compile model for better performance
    if rank == 0:
        print("Compiling model with torch.compile for better performance...")
    model = torch.compile(model, mode='default')
    if rank == 0:
        print("Model compilation complete!")
    
    # Create dataloaders with DistributedSampler
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        streaming=True,
        rank=rank,
        world_size=world_size
    )
    
    # Get parameters from lm_head and token_embedding
    nonhidden_params = [*model.module.lm_head.parameters(), *model.module.token_embedding.parameters()]
    nonhidden_param_ids = {id(p) for p in nonhidden_params}
    
    # Separate other parameters, excluding nonhidden ones
    hidden_weights = [p for p in model.parameters() if p.ndim >= 2 and id(p) not in nonhidden_param_ids]
    hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2 and id(p) not in nonhidden_param_ids]
    
    param_groups = [
        dict(params=hidden_weights, use_muon=True),
        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False),
    ]

    optimizer = SingleDeviceMuonWithAuxAdam(
        param_groups,
    )
    
    # Calculate total training steps
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    
    # Create scheduler for linear warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda')
    
    # Log model to wandb (only on rank 0)
    if rank == 0:
        wandb.watch(model, log_freq=100)
    
    # Training loop
    if rank == 0:
        print(f"\nStarting training for {args.epochs} epochs")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {args.warmup_steps}")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Set epoch for DistributedSampler
        train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, config, global_step, output_dir, val_loader, rank
        )
    
    if rank == 0:
        print("\nTraining complete!")
        wandb.finish()
    
    # Cleanup DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()