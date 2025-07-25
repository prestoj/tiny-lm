# BPE Tokenizer

A BPE (Byte Pair Encoding) tokenizer implementation using HuggingFace's tokenizers library for optimal performance.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Tokenizer

```python
from tokenizer import train_tokenizer

# Train on a text file
tokenizer = train_tokenizer(
    texts_file="data/corpus.txt",
    vocab_size=8192,
    min_frequency=2,
    save_path="my_tokenizer"
)
```

### Using a Tokenizer

```python
from tokenizer import load_tokenizer

# Load trained tokenizer
tokenizer = load_tokenizer("my_tokenizer")

# Encode text
tokens = tokenizer.encode("Hello, world!")

# Decode tokens
text = tokenizer.decode(tokens)

# Batch encoding
batch = tokenizer.encode_batch(
    ["First text", "Second text"],
    padding=True,
    max_length=512
)
```

### Command Line Usage

Train a tokenizer:
```bash
python -m tokenizer.train --data corpus.txt --vocab-size 8192 --output my_tokenizer
```

Tokenize a dataset:
```bash
python -m tokenizer.tokenize_dataset \
    --tokenizer my_tokenizer \
    --input-dir raw_texts/ \
    --output-dir tokenized/
```

## API Reference

### `train_tokenizer(texts_file, vocab_size, min_frequency, save_path)`
Trains a new BPE tokenizer on the provided text data.

### `load_tokenizer(path)`
Loads a previously trained tokenizer.

### `BPETokenizer`
Main tokenizer class with methods:
- `encode(text)`: Convert text to token IDs
- `decode(ids)`: Convert token IDs back to text
- `encode_batch(texts, **kwargs)`: Batch encode multiple texts
- `save(path)`: Save tokenizer to disk

## Performance

This implementation uses HuggingFace's Rust-based tokenizers library, providing:
- 100-1000x faster training than pure Python implementations
- Efficient batch processing
- Built-in parallelization
- Memory-efficient token storage