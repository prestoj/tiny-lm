# Small Embed Pretraining - Claude Assistant Notes

## Project Overview
Preston is training a tiny (~24M parameter) transformer model on Project Gutenberg data. The goal is to create a HIGH-QUALITY small language model, not just train quickly. Setup:
- 2x RTX 3060 GPUs (12GB each)
- DataParallel training (not DDP)
- 8192 vocab size with custom tokenizer
- Currently achieving good results with optimized hyperparameters
- Using "tinier" config: 192 d_model, 9 layers, 3 heads

## Key Implementations Done

### 1. Muon Optimizer
- Implemented in `muon.py` based on modded-nanogpt
- Uses Newton-Schulz orthogonalization for 2D weight matrices
- Separate from AdamW which handles embeddings/scalars
- Key hyperparams: momentum warmup (0.85→0.95), 5 NS iterations
- Fixed to match reference: proper spectral norm, correct momentum interpolation, multiplicative weight decay

### 2. Model Architecture (model.py)
- SwiGLU activation in FFN blocks
- QK normalization with LayerNorm
- Alternating local (256 window) and global attention by layer
- RoPE positional embeddings
- Weight tying between input/output embeddings

### 3. Differentiated Learning Rates
- Embeddings: 5e-3 (10x base) - sparse updates need higher LR
- LM Head: 1e-3 (2x base)
- Scalars/Norms: 5e-4 (base)
- Muon matrices: 1e-5 (via 0.02 scale)

### 4. Training Optimizations
- Gradient accumulation: 16 (effective batch = 256 tokens)
- Mixed precision fp16 training
- torch.compile disabled for DataParallel compatibility
- Momentum warmup for Muon optimizer

## Recent Experiments Tried

### What Helped:
1. **Removing dropout entirely** - Significant improvement! Small models need all their capacity
2. **Zero-init output projections** - Already implemented, helps training stability

### What Didn't Help:
1. **Learnable attention scales** - Minimal impact on loss
2. **Attention sinks** - No improvement for this size model
3. **ReLU² activation** - Performed about the same as SwiGLU

### Key Insight:
For tiny models, simpler is often better. Removing regularization (dropout) and letting the model use full capacity was the biggest win.

## Current Hyperparameters (run_baseline.sh)
```bash
# Tinier model config
D_MODEL=192
N_LAYERS=9
N_HEADS=3
D_FF=384  # For SwiGLU: (2/3) * 4 * 192 ≈ 256, but using 384

BATCH_SIZE=16
GRADIENT_ACCUMULATION=1  # Effective = 32 tokens (16 * 2 GPUs)
LR=5e-4
EMBEDDING_LR=5e-3
HEAD_LR=1e-3
MUON_LR_SCALE=0.02
WARMUP_STEPS=100
```

## Training Tips
- With tiny batches (64-256 tokens), embeddings need much higher LR due to sparsity
- Quality > Speed: Preston wants good models, not just fast training
- Muon needs conservative LR (1e-5 range) for stability
- Gradient accumulation is crucial for small batch training

## Common Issues
- High embedding LR can cause instability - monitor and reduce if loss bounces
- Small models are fragile - prefer stable training over aggressive optimization

## What's Working Well
- Muon optimizer for 2D matrices
- Differentiated learning rates by parameter type  
- Higher gradient accumulation for cleaner gradients
- Conservative but effective hyperparameter tuning

## Ideas: Trading Computation/Memory for Parameters

### High Impact - Easy to Implement
1. **Layer Recycling with Adapters**
   - Use only 3 base transformer blocks, recycle each 3x (9 effective layers)
   - Add tiny layer-specific adapters (linear projections) for specialization
   - Saves ~70% of layer parameters
   - Example: base_blocks[i % 3] + 0.1 * adapter[i](x)

2. **Low-Rank Factorized Projections**
   - Replace d_model→d_model with d_model→rank→d_model (rank=32)
   - Apply to all QKV projections and FFN weights
   - Saves ~80-90% of projection parameters
   - Still allows full-rank transformations through composition

3. **Cross-Layer Weight Tying**
   - Share QKV weights across groups of layers (e.g., layers 0-2, 3-5, 6-8)
   - Add per-layer scaling factors or small residual adapters
   - Effective weights: W_shared * layer_scale + layer_bias


### Combined Architecture Proposal
For ~8-10M params with 24M param performance:
- 3 base transformer blocks, each used 3x
- All projections use rank-32 factorization
- 4-expert shared FFN pool
- Factorized embeddings (3×64→192)
- Result: 3x more compute, 3x fewer parameters, potentially better performance

Key insight: Small models benefit more from parameter reuse than unique parameters!

Preston really appreciates the help! The goal is a great tiny English chatbot. 🤖