#!/bin/bash
# Script to run baseline pretraining

# Set your paths here
DATA_DIR="/home/preston/git/tiny-bot/tokenized_gutenberg_8k_clean"
OUTPUT_DIR="./checkpoints"
WANDB_PROJECT="tiny-bot-baseline"

# Model configuration
VOCAB_SIZE=8192
D_MODEL=384
N_LAYERS=12
N_HEADS=6
D_FF=1536

# Training configuration
BATCH_SIZE=16   # Reduced to 16 total (8 per GPU) to fit in memory
GRADIENT_ACCUMULATION=4  # Effective batch size = 16 * 4 = 64
LR=3e-4
WARMUP_STEPS=1000
EPOCHS=1
MAX_SEQ_LEN=1024  # Reduced from 2048 to fit in GPU memory

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set CUDA devices (use both GPUs)
export CUDA_VISIBLE_DEVICES=0,1

# Run training
python train.py \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --wandb-project $WANDB_PROJECT \
    --vocab-size $VOCAB_SIZE \
    --d-model $D_MODEL \
    --n-layers $N_LAYERS \
    --n-heads $N_HEADS \
    --d-ff $D_FF \
    --max-seq-len $MAX_SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --learning-rate $LR \
    --warmup-steps $WARMUP_STEPS \
    --epochs $EPOCHS \
    --log-interval 100 \
    --eval-interval 1000 \
    --save-interval 5000