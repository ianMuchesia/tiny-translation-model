# Tiny Translation Model
======================

A minimal seq2seq Transformer for English->Swahili translation.

## Features

- **Encoder/Decoder Blocks** (`src/encoder.py`, `src/decoder.py`): Multi-head attention and feed-forward layers
- **Positional Encoding** (`src/positional_encoding.py`): Sinusoidal position signals
- **Tokenizer and Vocab** (`src/tokenizer.py`): Word-level tokenization with special tokens
- **Dataset Loader** (`src/dataset.py`): Parses "english => swahili" pairs from `data/toy_translations.txt`
- **Seq2Seq Wrapper** (`src/encoder_decoder.py`): End-to-end encoder-decoder model
- **Training Entry Point** (`src/train.py`): Placeholder for the training loop

## Quick Start

```bash
python -m src.train
```

## Project Structure

- `src/` — Core implementation
- `data/` — Translation pairs
- `notebooks/` — Experiments and demos
- `experiments/` — Saved outputs
- `math-notes/` — Derivations and notes
