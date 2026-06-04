Notes
=====

These notes capture design intent and the reasoning behind implementation choices so the project is easy to re-enter months later.

## Data Loading and Batching

**Loop Cost**: With $N=32$, a Python loop over the batch is effectively negligible in runtime.

**Memory Fragmentation**: Returning tensors directly from `__getitem__` creates many small allocations per batch. Over many batches, this can fragment memory and add overhead.

**Practical Choice**: Keep `__getitem__` lightweight and batch in the collate step when possible.

## Masking Intuition

**Padding Mask**: Tells the encoder and decoder to ignore tokens with ID `0` (padding).

**Causal Mask**: Used only in the decoder to prevent looking ahead to future tokens.

**Dynamic Shapes**: Mask size depends on the longest sequence in the batch. If the max length is 9, the mask is $9 \times 9$; if 12, it becomes $12 \times 12$.

## Tensor Layout and Reshaping

**Contiguous Memory**: Tensors are stored as a flat, contiguous buffer with stride metadata that makes them appear multi-dimensional.

**`.view()`**: Fast metadata-only reshape. It requires contiguous memory, so it can throw errors when the underlying buffer is not contiguous.

**`.reshape()`**: Tries a view first; if not possible, it allocates a new contiguous buffer and copies data. This is safer but can be slower.

**Why It Matters in Attention**: Multi-head attention often permutes dimensions, which can break contiguity. Use `.reshape()` when you cannot guarantee contiguity.

## Vocabulary vs. Embeddings

**Tokenizer (Vocabulary)**: Maps strings to integer IDs and defines `vocab_size`.

**Embedding Layer**: A lookup table of vectors; row `i` corresponds to token ID `i`.

**Implication**: Model embedding size must match the tokenizer vocabulary size, not the number of unique tokens seen in a batch.

## Dimension Debugging

**`unsqueeze(dim)`** inserts a size-1 dimension at the given index. Example: `[32, 4] -> unsqueeze(1) -> [32, 1, 4]`.

**Chaining** inserts multiple dimensions in sequence: `[32, 7] -> unsqueeze(1) -> [32, 1, 7] -> unsqueeze(2) -> [32, 1, 1, 7]`.

## Current Status

- Working through masking and tensor shape validation.
- Next step: verify attention weight shapes across encoder and decoder passes.
