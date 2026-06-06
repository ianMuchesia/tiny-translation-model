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

## Training Step Notes

**Masks**:
- `src_mask = (encoder_input == 0).unsqueeze(1).unsqueeze(2)` handles padding in the encoder.
- `decoder_padding_mask = (decoder_input == 0).unsqueeze(1).unsqueeze(2)` handles padding in the decoder.
- `causal_mask = (torch.tril(torch.ones(seq_len, seq_len)) == 0)` prevents future look-ahead.
- `tgt_mask = decoder_padding_mask | causal_mask` combines padding and causal masking.

**Flattening for Loss**:
- `decoder_out` is reshaped to `[batch * seq_len, vocab]` before `CrossEntropyLoss`.
- `expected_labels` is reshaped to `[batch * seq_len]` to match.

**Accuracy With Padding**:
- Compute correctness against `expected_labels`.
- Use `valid_tokens_mask = (expected_labels != 0)` so padding does not inflate accuracy.
- Only count valid tokens in `total`.

**Debugging Shapes**:
- Print prediction and label shapes when loss or accuracy behaves unexpectedly.

## Inference Bug: Premature EOS

**Symptom**: `predict()` returns `[<SOS>, <EOS>]` immediately, even for sentences seen in training.

**Observed Shapes**:
- `tensor_input`: `[1, 6]`
- `prediction`: `[1, 1, 448]`
- `latest_prediction`: `[1, 448]`
- `next_word_id`: `[1]` then `[1, 1]` after `unsqueeze`
- `decoder_input` grows to `[1, 2]`, then `<EOS>` is predicted.

**What is NOT wrong**:
- Autoregressive loop shape logic is correct.
- `model.eval()` was called before inference.
- Encoder input already includes `<SOS>` and `<EOS>` tokens.

**Likely Causes**:
- Incorrect mask logic during training/inference (using `mask == 0` instead of masking padding tokens).

**Checks**:
- Verify padding masks are built from `(input == 0)` and that causal masking is applied only to decoder self-attention.
- Confirm mask shapes align with `[batch, heads, seq_len, seq_len]` expectations.

**Resolution**:
- Fixed masking to use padding checks (`input == 0`) and correct mask composition for the decoder.
- Inference now generates full sequences instead of immediate `<EOS>` (e.g., `[<SOS>, 45, 48, <EOS>]`).

## Current Status

- Training loop uses combined padding + causal masks and reports accuracy on non-padding tokens.
- Predictions are working with deterministic token IDs.
- Next step: verify attention weight shapes across encoder and decoder passes.

## Attention Heatmaps

**Interpretation**: The heatmap suggests the model has memorized the training example rather than learning a broad translation strategy.

**Observed Pattern**: The Swahili `<SOS>` row focuses on the English word "they" when predicting the first real output token.

**Why That Happens**: During autoregressive decoding, the model starts from `<SOS>` and uses the strongest source-word anchor it learned for the memorized sentence.

**Special Tokens**: `<SOS>` and `<EOS>` are structural markers, so they often appear darker or less informative than content words in the heatmap.

**End-of-Sequence Signal**: The model can rely on a strong content token like "home" to decide that the translation is complete, instead of attending to the source `<EOS>` token.