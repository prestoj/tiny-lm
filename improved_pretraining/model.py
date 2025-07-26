"""
Simple transformer model for baseline pretraining
Target: ~30M parameters with 65k vocab size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)
        
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
        
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached, seq_len)
    
    @staticmethod
    def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_len: int) -> torch.Tensor:
        x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
        
        cos = cos[:seq_len, :x1.shape[-1]].unsqueeze(1)
        sin = sin[:seq_len, :x1.shape[-1]].unsqueeze(1)
        
        out = torch.empty_like(x_reshape)
        out[..., 0] = x1 * cos - x2 * sin
        out[..., 1] = x1 * sin + x2 * cos
        
        return out.reshape(*x.shape)


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads  # GQA: fewer KV heads
        self.n_rep = self.n_heads // self.n_kv_heads  # Repeat factor for KV heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k)
        self.out_proj = nn.Linear(d_model, d_model)
        
        nn.init.zeros_(self.out_proj.weight)
        
        self.q_norm = RMSNorm(self.d_k)
        self.k_norm = RMSNorm(self.d_k)
        
        # Scale factor for attention scores (not learnable)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        self.rope = RotaryPositionalEmbedding(self.d_k)
    
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        if hasattr(F, 'scaled_dot_product_attention') and mask is None:
            # Use efficient implementation when no custom mask is needed
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scale
            )
        else:
            # Manual attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                causal_mask = mask
            else:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
                
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, -1e4)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        
        return output, None



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
        nn.init.zeros_(self.down_proj.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gate = F.silu(gate, inplace=True)
        x = gate * up
        x = self.down_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_kv_heads: int = None):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, n_kv_heads=n_kv_heads)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        self.feed_forward = SwiGLU(d_model, d_ff)
        
    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class GatedParallelTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_parallel_blocks: int = 3, n_kv_heads: int = None):
        super().__init__()
        
        self.n_parallel_blocks = n_parallel_blocks
        
        # Single shared transformer block - all parallel blocks are identical
        self.transformer_block = TransformerBlock(d_model, n_heads, d_ff, n_kv_heads=n_kv_heads)
        
        # Gating mechanism: residual stream -> weights for each parallel block
        self.gate = nn.Linear(d_model, n_parallel_blocks)
        
        # Initialize gate to slightly favor uniform distribution
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        nn.init.constant_(self.gate.bias, 0.0)
        
    def forward(self, x, mask=None):
        # Compute gate weights from residual stream
        gate_weights = F.softmax(self.gate(x), dim=-1)  # [batch, seq_len, n_parallel_blocks]
        
        # Process through transformer block n_parallel_blocks times
        outputs = []
        for i in range(self.n_parallel_blocks):
            block_output = self.transformer_block(x, mask)
            outputs.append(block_output)
        
        # Stack outputs: [batch, seq_len, d_model, n_parallel_blocks]
        outputs = torch.stack(outputs, dim=-1)
        
        # Apply gating weights and sum
        gate_weights = gate_weights.unsqueeze(-2)  # [batch, seq_len, 1, n_parallel_blocks]
        weighted_outputs = outputs * gate_weights
        output = weighted_outputs.sum(dim=-1)  # [batch, seq_len, d_model]
        
        return output


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 384,
        n_layers: int = 12,
        n_heads: int = 6,
        n_kv_heads: int = None,
        d_ff: int = 1024,
        max_seq_len: int = 2048,
        pad_token_id: int = 1,
        n_parallel_blocks: int = 3,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.n_layers = n_layers
        self.n_parallel_blocks = n_parallel_blocks
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.token_embedding = nn.Embedding(vocab_size, d_model // 2)
        self.token_to_res = nn.Linear(d_model // 2, d_model, bias=False)
        
        # Shared transformer blocks - used by all layers
        self.shared_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff,
                n_kv_heads=n_kv_heads
            )
            for _ in range(n_parallel_blocks)
        ])
        
        # Single shared gating network - used by all layers
        self.shared_gate = nn.Linear(d_model, n_parallel_blocks)
        
        # Initialize gate to slightly favor uniform distribution
        nn.init.xavier_uniform_(self.shared_gate.weight, gain=0.1)
        nn.init.constant_(self.shared_gate.bias, 0.0)
        
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying disabled - incompatible with factorized embeddings
        # self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Modified initialization: 0.5 * (1/sqrt(in_features))
            # Better for small models according to the reference
            std = 0.5 * (module.in_features ** -0.5)
            bound = (3 ** 0.5) * std  # sqrt(3) * std for uniform distribution
            torch.nn.init.uniform_(module.weight, -bound, bound)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Keep embedding initialization slightly larger for sparse updates
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask efficiently
        mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device), diagonal=1)
        
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.token_to_res(x)
        
        # Apply shared transformer blocks multiple times
        for layer_idx in range(self.n_layers):
            # Compute gate weights from current residual stream
            gate_logits = self.shared_gate(x)
            gate_weights = F.softmax(gate_logits, dim=-1)  # [batch, seq_len, n_parallel_blocks]
            
            # Process through all shared blocks in parallel
            block_outputs = []
            for block in self.shared_blocks:
                if self.use_gradient_checkpointing and self.training:
                    # Use gradient checkpointing to save memory
                    block_output = checkpoint(block, x, mask, use_reentrant=False)
                else:
                    block_output = block(x, mask)
                block_outputs.append(block_output)
            
            # Stack outputs: [batch, seq_len, d_model, n_parallel_blocks]
            block_outputs = torch.stack(block_outputs, dim=-1)
            
            # Apply gating weights and sum
            gate_weights = gate_weights.unsqueeze(-2)  # [batch, seq_len, 1, n_parallel_blocks]
            weighted_outputs = block_outputs * gate_weights
            x = weighted_outputs.sum(dim=-1)  # [batch, seq_len, d_model]
        
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test wide transformer with parallel gated blocks
    print("Testing Wide TinyTransformer with Parallel Gated Blocks:")
    model = TinyTransformer(
        vocab_size=8192,
        d_model=384,
        n_layers=6,  # Fewer layers since each is more powerful
        n_heads=6,
        n_kv_heads=2,  # GQA: 6 query heads, 2 key-value heads
        d_ff=1024,
        n_parallel_blocks=3  # 3 parallel blocks per layer
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
    
    # Test generation
    print("\nTesting generation:")
    test_input = torch.tensor([[1, 2, 3, 4, 5]]).to(device)  # Start with a few tokens
    generated = model.generate(test_input, max_length=20, temperature=0.8)
    print(f"Generated sequence shape: {generated.shape}")