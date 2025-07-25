"""
Training script for baseline transformer model
"""

import os
import sys
import torch
import torch.nn as nn
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


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup, then constant LR (no decay as requested)"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, config, 
                global_step, output_dir, val_loader):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    accumulated_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Mixed precision training
        with autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(input_ids, labels)
            
        # Handle DataParallel returning multiple losses
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()  # Average losses from all GPUs
        
        # Scale loss by accumulation steps
        loss = loss / config.gradient_accumulation_steps
        accumulated_loss += loss.item()
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Optimizer step every gradient_accumulation_steps
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            # Handle DataParallel
            if hasattr(model, 'module'):
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), config.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # LR scheduler step
            scheduler.step()
            
            # Reset accumulated loss
            accumulated_loss = 0
        
        # Update metrics (using unscaled loss)
        actual_loss = loss.item() * config.gradient_accumulation_steps
        total_loss += actual_loss * input_ids.size(0)
        total_tokens += input_ids.numel()
        
        # Update progress bar
        avg_loss = total_loss / ((step + 1) * input_ids.size(0))
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({
            'loss': f"{avg_loss:.4f}",
            'ppl': f"{np.exp(avg_loss):.2f}",
            'lr': f"{current_lr:.2e}",
            'acc_step': f"{(step + 1) % config.gradient_accumulation_steps}/{config.gradient_accumulation_steps}"
        })
        
        # Log to wandb
        if step % config.log_interval == 0:
            wandb.log({
                'train/loss': actual_loss,
                'train/perplexity': np.exp(actual_loss),
                'train/learning_rate': current_lr,
                'train/tokens': total_tokens,
                'global_step': global_step,
            })
        
        # Increment global step
        global_step += 1
        
        # Save checkpoint at intervals
        if global_step % config.save_interval == 0:
            print(f"\nSaving checkpoint at step {global_step}")
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': vars(config)
            }
            torch.save(checkpoint, output_dir / f'checkpoint_step_{global_step}.pt')
            
        # # Evaluate at intervals
        # if global_step % config.eval_interval == 0:
        #     model.eval()
        #     val_loss, val_ppl = evaluate(model, val_loader, device)
        #     model.train()
        #     print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        #     wandb.log({
        #         'val/loss': val_loss,
        #         'val/perplexity': val_ppl,
        #         'global_step': global_step,
        #     })
    
    return total_loss / len(train_loader), global_step


# @torch.no_grad()
# def evaluate(model, val_loader, device):
#     """Evaluate on validation set"""
#     model.eval()
#     total_loss = 0
#     total_samples = 0
    
#     for batch in tqdm(val_loader, desc="Evaluating"):
#         input_ids = batch['input_ids'].to(device)
#         labels = batch['labels'].to(device)
        
#         logits, loss = model(input_ids, labels)
        
#         total_loss += loss.item() * input_ids.size(0)
#         total_samples += input_ids.size(0)
    
#     avg_loss = total_loss / total_samples
#     perplexity = np.exp(avg_loss)
    
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description='Train baseline transformer model')
    
    # Model arguments
    parser.add_argument('--vocab-size', type=int, default=8192, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=384, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=1536, help='Feed-forward dimension')
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with tokenized data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N steps')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Evaluate every N steps')
    parser.add_argument('--save-interval', type=int, default=5000, help='Save checkpoint every N steps')
    
    # Other arguments
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--wandb-project', type=str, default='tiny-bot-baseline', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPU(s)")
        if n_gpus > 1:
            print(f"Using DataParallel across {n_gpus} GPUs")
            use_data_parallel = True
        else:
            print(f"Using single GPU")
            use_data_parallel = False
    else:
        device = torch.device('cpu')
        print(f"Using CPU")
        use_data_parallel = False
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )
    config = wandb.config
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model = TinyTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        pad_token_id=0
    )
    
    # Apply DataParallel if using multiple GPUs
    if use_data_parallel:
        model = nn.DataParallel(model)
        model = model.to(device)
        print(f"Model created with {model.module.num_parameters():,} parameters")
    else:
        model = model.to(device)
        print(f"Model created with {model.num_parameters():,} parameters")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        streaming=True
    )
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Calculate total training steps
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    # Create scheduler (linear warmup, no decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda')
    
    # Log model to wandb
    wandb.watch(model, log_freq=100)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, config, global_step, output_dir, val_loader
        )

        # TODO FIX eval code so i can run multiple epochs if i want to but to be hoenst i odn't want to.
        
        # # Evaluate
        # val_loss, val_perplexity = evaluate(model, val_loader, device)
        
        # print(f"\nEpoch {epoch}/{args.epochs}")
        # print(f"Train Loss: {train_loss:.4f}, Train PPL: {np.exp(train_loss):.2f}")
        # print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_perplexity:.2f}")
        
        # # Log epoch metrics
        # wandb.log({
        #     'epoch': epoch,
        #     'train/epoch_loss': train_loss,
        #     'train/epoch_perplexity': np.exp(train_loss),
        #     'val/loss': val_loss,
        #     'val/perplexity': val_perplexity,
        # })
        
        # # Save checkpoint
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'val_loss': val_loss,
        #         'config': vars(args)
        #     }
        #     torch.save(checkpoint, output_dir / 'best_model.pt')
        #     print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # # Regular checkpoint
        # if epoch % 5 == 0:
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'val_loss': val_loss,
        #         'config': vars(args)
        #     }
        #     torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    print("\nTraining complete!")
    wandb.finish()


if __name__ == "__main__":
    main()