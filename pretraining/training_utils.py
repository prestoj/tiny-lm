import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
from typing import Optional, Dict, Any


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with linear warmup"""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        base_lr: float,
        final_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.final_lr + (self.base_lr - self.final_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
        return lr


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Standard cross entropy
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        
        # Label smoothing
        smooth_loss = -log_probs.mean(dim=-1)
        
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class TransformerTrainer:
    """Training utilities for the modern transformer"""
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        mixed_precision: bool = True,
        gradient_checkpointing: bool = False,
        compile_model: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.mixed_precision = mixed_precision
        
        # Compile model with torch.compile for faster training (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Mixed precision training
        self.scaler = GradScaler() if mixed_precision else None
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def create_optimizer(
        self,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-9
    ) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay fix"""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if 'norm' in name or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            fused=True if self.device == 'cuda' else False  # Fused optimizer for speed
        )

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        max_grad_norm: float = 1.0
    ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        optimizer.zero_grad()
        
        if self.mixed_precision:
            with autocast():
                logits = self.model(input_ids)
                # Shift for next-token prediction
                logits = logits[:, :-1].contiguous()
                labels = labels[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            logits = self.model(input_ids)
            logits = logits[:, :-1].contiguous()
            labels = labels[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            optimizer.step()
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'perplexity': math.exp(loss.item())
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(input_ids)
            logits = logits[:, :-1].contiguous()
            labels = labels[:, 1:].contiguous()
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
        
        avg_loss = total_loss / total_tokens
        return {
            'loss': avg_loss,
            'perplexity': math.exp(avg_loss)
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0
    ) -> torch.Tensor:
        """Generate text using the model"""
        self.model.eval()
        generated = prompt_ids.to(self.device)
        past_kvs = None
        
        for _ in range(max_length - len(prompt_ids[0])):
            if past_kvs is None:
                # First forward pass
                logits, past_kvs = self.model(generated, use_cache=True)
                logits = logits[:, -1, :]
            else:
                # Subsequent passes with KV cache
                logits, past_kvs = self.model(
                    generated[:, -1:], 
                    use_cache=True, 
                    past_kvs=past_kvs
                )
                logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[:, token_id] /= repetition_penalty
            
            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS token (assuming 0 is EOS)
            if next_token.item() == 0:
                break
        
        return generated


# Example training configuration
def create_training_config() -> Dict[str, Any]:
    """Create a default training configuration"""
    return {
        'batch_size': 32,
        'gradient_accumulation_steps': 4,
        'learning_rate': 3e-4,
        'warmup_steps': 1000,
        'max_steps': 50000,
        'eval_interval': 1000,
        'save_interval': 5000,
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'label_smoothing': 0.1,
        'mixed_precision': True,
        'compile_model': True
    }


# Data loading utilities
class TextDataset(torch.utils.data.Dataset):
    """Simple text dataset for language modeling"""
    def __init__(self, texts: list, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
        
        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # For language modeling, labels are the same as input_ids
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()
        }


if __name__ == "__main__":
    from modern_transformer import ModernTransformer
    
    # Create model
    model = ModernTransformer()
    
    # Create trainer
    trainer = TransformerTrainer(model)
    
    # Create optimizer and scheduler
    optimizer = trainer.create_optimizer()
    config = create_training_config()
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=config['warmup_steps'],
        max_steps=config['max_steps'],
        base_lr=config['learning_rate']
    )
    
    # Create loss function
    criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    
    print("Training configuration created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")