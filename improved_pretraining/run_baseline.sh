#!/bin/bash
# Script to run baseline pretraining with DistributedDataParallel

# Set your paths here
DATA_DIR="/home/preston/git/tiny-bot/tokenized_gutenberg_8k_clean"
OUTPUT_DIR="./checkpoints_ddp"

# # Model configuration tiny
# VOCAB_SIZE=8192
# D_MODEL=384
# N_LAYERS=12
# N_HEADS=6
# N_KV_HEADS=6
# D_FF=768
# WANDB_PROJECT="tiny-bot-baseline"

# Model configuration - wide architecture
VOCAB_SIZE=8192
D_MODEL=192
N_LAYERS=9
N_HEADS=3
N_KV_HEADS=3
D_FF=384
N_PARALLEL_BLOCKS=9
WANDB_PROJECT="tiny-bot-baseline-tinier-2"

# Training configuration
# did 16 and 16 for last most successful run
BATCH_SIZE=16   # Per GPU batch size (total will be 16 * num_gpus)
GRADIENT_ACCUMULATION=1  # Effective batch size = 16 * num_gpus * 1

# Conservative learning rates for stable, high-quality training
LR=1e-4             # Conservative base LR for scalars/norms
EMBEDDING_LR=5e-3   # 10x base - reasonable boost for sparse embeddings
HEAD_LR=1e-3        # 2x base - output layer can handle slightly higher

WARMUP_STEPS=100
EPOCHS=1            # 7B tokens is substantial
MAX_SEQ_LEN=1024    # Reduced from 2048 to fit in GPU memory

# Muon optimizer configuration - extra conservative
MUON_LR_SCALE=0.02    # Muon LR = 5e-4 * 0.02 = 1e-5 (very stable)
MUON_MOMENTUM=0.95    # Final momentum for Muon
MUON_WARMUP=300       # Steps to warmup momentum from 0.85 to 0.95
MUON_NS_STEPS=5       # Newton-Schulz iterations for orthogonalization

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable gradient checkpointing for memory efficiency
# This trades ~20-30% compute for significant memory savings
GRADIENT_CHECKPOINTING=true

# DDP settings
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1

# Number of GPUs
NUM_GPUS=2

# Launch with torchrun for DDP
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train.py \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --wandb-project $WANDB_PROJECT \
    --vocab-size $VOCAB_SIZE \
    --d-model $D_MODEL \
    --n-layers $N_LAYERS \
    --n-heads $N_HEADS \
    --n-kv-heads $N_KV_HEADS \
    --d-ff $D_FF \
    --max-seq-len $MAX_SEQ_LEN \
    --n-parallel-blocks $N_PARALLEL_BLOCKS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --learning-rate $LR \
    --embedding-lr $EMBEDDING_LR \
    --head-lr $HEAD_LR \
    --warmup-steps $WARMUP_STEPS \
    --epochs $EPOCHS \
    --log-interval 10 \
    --eval-interval 1000 \
    --save-interval 10000 \
    --muon-lr-scale $MUON_LR_SCALE \
    --muon-momentum $MUON_MOMENTUM \
    --muon-momentum-warmup $MUON_WARMUP \
    --muon-ns-steps $MUON_NS_STEPS \
    ${GRADIENT_CHECKPOINTING:+--gradient-checkpointing}