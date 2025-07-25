# Baseline Pretraining

Simple transformer pretraining setup for baseline experiments.

## Model Architecture

- **Parameters**: ~30M (with 65k vocab size)
- **Architecture**: Standard transformer
  - Hidden size: 384
  - Layers: 12
  - Attention heads: 6
  - FFN size: 1536
  - Max sequence length: 2048

## Features

- Mixed precision training (FP16)
- Gradient clipping
- Linear learning rate warmup (no decay)
- WandB logging
- Streaming dataset (memory efficient)
- Weight tying (embeddings shared with output layer)

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Update paths in `run_baseline.sh`:
```bash
DATA_DIR="/path/to/your/tokenized/data"
```

3. Run training:
```bash
./run_baseline.sh
```

## Training Configuration

- Batch size: 32
- Learning rate: 3e-4
- Warmup steps: 1000
- Optimizer: AdamW
- Weight decay: 0.01
- Gradient clipping: 1.0

## Monitoring

Training is logged to Weights & Biases. Metrics include:
- Training/validation loss
- Perplexity
- Learning rate
- Token count

## Checkpoints

Checkpoints are saved to `./checkpoints/`:
- `best_model.pt`: Best validation loss
- `checkpoint_epoch_N.pt`: Regular epoch checkpoints