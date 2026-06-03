Notes
=====

## Data Loading and Batching

**Loop Cost**: With $N=32$, a Python loop over the batch is effectively negligible in runtime.

**Memory Fragmentation**: Returning tensors directly from `__getitem__` creates many small allocations per batch. Over many batches, this can fragment memory and add overhead.

## Masking Intuition

**Padding Mask**: Tells the encoder and decoder to ignore tokens with ID `0` (padding).

**Causal Mask**: Used only in the decoder to prevent looking ahead to future tokens.

## Dynamic Mask Shapes

Mask size depends on the longest sequence in the batch. If the max length is 9, the mask is $9 \times 9$; if 12, it becomes $12 \times 12$.

## Current Status

Stopped at masking.