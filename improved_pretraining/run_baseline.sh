#!/bin/bash
# Script to run baseline pretraining with DistributedDataParallel

# Set your paths here
# DATA_DIR will be automatically set based on VOCAB_SIZE
OUTPUT_DIR="./checkpoints"

# # Model configuration tinier
# VOCAB_SIZE=1024
# D_MODEL=264
# N_LAYERS=1
# N_HEADS=4
# N_KV_HEADS=4
# D_TOKEN=32
# D_FF=512
# LOG_INTERVAL=10
# NUM_RECURRENCES=8  # Number of times to recycle layers (1 = no recycling)
# USE_CHECKPOINT=false  # Use gradient checkpointing to save memory (true/false)
# WANDB_PROJECT="tlm-quark"

# Model configuration tinier
VOCAB_SIZE=8192
D_MODEL=400
N_LAYERS=1
N_HEADS=8
N_KV_HEADS=8
D_TOKEN=16
D_FF=768
LOG_INTERVAL=10
NUM_RECURRENCES=8  # Number of times to recycle layers (1 = no recycling)
USE_CHECKPOINT=true  # Use gradient checkpointing to save memory (true/false)
WANDB_PROJECT="tlm-atom"

# Set paths based on vocab size
TOKENIZER_PATH="../gutenberg_tokenizer_${VOCAB_SIZE}"  # Automatically uses vocab size
# Map vocab size to data directory suffix (1024->1k, 8192->8k, 65536->65k)
if [ $VOCAB_SIZE -eq 1024 ]; then
    DATA_SUFFIX="1k"
elif [ $VOCAB_SIZE -eq 2048 ]; then
    DATA_SUFFIX="2k"
elif [ $VOCAB_SIZE -eq 4096 ]; then
    DATA_SUFFIX="4k"
elif [ $VOCAB_SIZE -eq 8192 ]; then
    DATA_SUFFIX="8k"
elif [ $VOCAB_SIZE -eq 16384 ]; then
    DATA_SUFFIX="16k"
elif [ $VOCAB_SIZE -eq 32768 ]; then
    DATA_SUFFIX="32k"
elif [ $VOCAB_SIZE -eq 65536 ]; then
    DATA_SUFFIX="65k"
else
    echo "Warning: Unknown vocab size $VOCAB_SIZE, please set DATA_DIR manually"
    exit 1
fi
DATA_DIR="../tokenized_gutenberg_${DATA_SUFFIX}_clean"

# Training configuration
# did 16 and 16 for last most successful run
BATCH_SIZE=16   # Per GPU batch size (total will be 16 * num_gpus)
GRADIENT_ACCUMULATION=16  # Effective batch size = 16 * num_gpus * 1

# Conservative learning rates for stable, high-quality training
LR=5e-4             # Conservative base LR for scalars/norms
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
    --d-token $D_TOKEN \
    --d-ff $D_FF \
    --max-seq-len $MAX_SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --learning-rate $LR \
    --embedding-lr $EMBEDDING_LR \
    --head-lr $HEAD_LR \
    --warmup-steps $WARMUP_STEPS \
    --epochs $EPOCHS \
    --log-interval $LOG_INTERVAL \
    --eval-interval 1000 \
    --save-interval 10000 \
    --muon-lr-scale $MUON_LR_SCALE \
    --muon-momentum $MUON_MOMENTUM \
    --muon-momentum-warmup $MUON_WARMUP \
    --muon-ns-steps $MUON_NS_STEPS \
    --num-recurrences $NUM_RECURRENCES \
    --tokenizer-path $TOKENIZER_PATH \
    $([ "$USE_CHECKPOINT" = "true" ] && echo "--use-checkpoint")