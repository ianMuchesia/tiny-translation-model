Tiny Translation Model
======================

A compact seq2seq Transformer for English->Swahili translation in PyTorch.

## Features

- **Encoder/Decoder Blocks** (`src/encoder.py`, `src/decoder.py`): Multi-head attention and feed-forward layers
- **Positional Encoding** (`src/positional_encoding.py`): Sinusoidal position signals
- **Tokenizer and Vocab** (`src/tokenizer.py`): Word-level tokenization with special tokens
- **Dataset Loader** (`src/dataset.py`): Parses "english => swahili" pairs from `data/toy_translations.txt`
- **Seq2Seq Wrapper** (`src/encoder_decoder.py`): End-to-end encoder-decoder model
- **Training Entry Point** (`src/train.py`): Training script (in progress)
- **Training Notebook** (`notebooks/transation_examples.ipynb`): End-to-end setup and training loop
- **Attention Debugging**: Heatmaps and shape checks for masking and decoder behavior
- **Inference Module** (`src/inference.py`): Decoding and translation utilities
- **BLEU Evaluation** (`src/evaluate_bleu.py`): Automatic translation quality assessment
- **Greedy Decoding** (`src/greedy_decode.py`): Fast single-path inference strategy
- **Test Dataset** (`data/toy_translations_test.txt`): Holdout evaluation set

## Quick Start

```bash
python -m src.train
```

## Status

- Training, evaluation, and inference pipelines are complete and working.
- Masking, accuracy calculations, and attention heatmap analysis validated.
- BLEU scores and sample predictions generated and saved.

## Project Structure

- `src/` — Core implementation
- `data/` — Translation pairs
- `notebooks/` — Experiments and demos
- `experiments/` — Saved outputs
- `math-notes/` — Derivations and notes
