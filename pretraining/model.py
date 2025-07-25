import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the frequency tensor
        inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position embeddings
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len:
            self._extend_cache(seq_len)
        
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device)
        )

    def _extend_cache(self, seq_len):
        if seq_len <= self.max_seq_len:
            return
        
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
        self.max_seq_len = seq_len


def rotate_half(x):
    """Rotates half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to q and k"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with Grouped Query Attention option"""
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Grouped Query Attention projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        B, T, C = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle KV cache for inference
        if use_cache:
            if past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            present_kv = (k, v)
        else:
            present_kv = None
        
        # Repeat KV heads if using MQA/GQA
        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)
        
        if use_cache:
            return out, present_kv
        return out


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization"""
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, n_kv_heads, dropout)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # Pre-norm architecture
        normed = self.norm1(x)
        if use_cache:
            attn_out, present_kv = self.attn(normed, mask, use_cache, past_kv)
        else:
            attn_out = self.attn(normed, mask)
            present_kv = None
        x = x + self.dropout(attn_out)
        
        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        if use_cache:
            return x, present_kv
        return x


class ModernTransformer(nn.Module):
    """
    Modern ~10M parameter Transformer with latest improvements:
    - RMSNorm instead of LayerNorm
    - SwiGLU activation in FFN
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention option
    - Pre-normalization
    - KV-caching for efficient inference
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 320,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1280,
        n_kv_heads: Optional[int] = 4,  # For GQA
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        tie_embeddings: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, n_kv_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(d_model)
        
        # Output projection
        if tie_embeddings:
            self.output_projection = None  # Use embedding weights
        else:
            self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Scale embeddings
        self.token_embedding.weight.data *= math.sqrt(d_model)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kvs: Optional[list] = None
    ):
        B, T = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Create causal mask if needed
        if mask is None and not use_cache:
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        # Process through transformer blocks
        present_kvs = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            
            if use_cache:
                x, present_kv = block(x, mask, use_cache, past_kv)
                present_kvs.append(present_kv)
            else:
                x = block(x, mask)
        
        # Final norm
        x = self.norm(x)
        
        # Output projection
        if self.output_projection is None:
            logits = F.linear(x, self.token_embedding.weight)
        else:
            logits = self.output_projection(x)
        
        if use_cache:
            return logits, present_kvs
        return logits

    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage and parameter count verification
if __name__ == "__main__":
    model = ModernTransformer(
        vocab_size=8192,
        d_model=320,
        n_layers=4,
        n_heads=8,
        d_ff=1280,
        n_kv_heads=4,  # Using GQA with 4 KV heads
        max_seq_len=2048,
        dropout=0.1,
        tie_embeddings=True
    )
    
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Model size: {model.count_parameters() / 1e6:.2f}M parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 8192, (batch_size, seq_len))
    
    # Standard forward pass
    logits = model(input_ids)
    print(f"Output shape: {logits.shape}")
    
    # Test with KV caching
    logits, past_kvs = model(input_ids[:, :10], use_cache=True)
    print(f"With KV cache - Output shape: {logits.shape}")
    print(f"Number of cached KV pairs: {len(past_kvs)}")