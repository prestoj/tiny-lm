"""
Simple transformer model for baseline pretraining
Target: ~30M parameters with 65k vocab size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TinyTransformer(nn.Module):
    """
    Tiny transformer model for baseline experiments
    
    With vocab_size=65536:
    - d_model=512, n_layers=8, n_heads=8 → ~33M params
    - d_model=384, n_layers=12, n_heads=6 → ~31M params
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 384,
        n_layers: int = 12,
        n_heads: int = 6,
        d_ff: int = 1536,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 1  # <|padding|> is typically token 1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (share embeddings with output projection)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(input_ids.device)
        
        # Embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
        
        return logits, loss
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        """Simple generation for testing"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                logits, _ = self(input_ids)
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    values, indices = logits.topk(top_k)
                    logits = torch.full_like(logits, -float('inf'))
                    logits.scatter_(1, indices, values)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
        return input_ids
    
    def num_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time

    device = "cuda"

    # Test model size
    model = TinyTransformer(
        vocab_size=8192,
        d_model=384,
        n_layers=12,
        n_heads=6,
        d_ff=1536
    ).to(device)
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 8192, (batch_size, seq_len)).to(device)
    logits, loss = model(input_ids, labels=input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item() if loss is not None else 'N/A'}")