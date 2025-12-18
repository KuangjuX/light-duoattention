# Light-DuoAttention

A lightweight CuTe-based CUDA kernel for DuoAttention, optimized for large language model inference.

## Overview

`Light-DuoAttention` provides a lightweight, fused CUDA kernel for **DuoAttention**. This attention mechanism is a hybrid approach designed for efficient long-context processing. 
Unlike standard attention where all heads behave identically, DuoAttention employs two distinct types of heads:

1.  **Retrieval Heads**: These heads perform **Full Attention** over the entire sequence (or a large chunk), allowing them to capture rich, long-range dependencies and act as a powerful information retrieval component.
2.  **Streaming Heads**: These heads use **Streaming Attention**, processing tokens in a sliding-window or fixed-cache manner. They are highly efficient and ideal for handling very long sequences while maintaining local context.

This hybrid approach allows a model to balance expressive power with computational efficiency. 
This kernel is implemented using CuTeDSL, the core of CUTLASS 4.x, enabling a clean and optimized implementation of this complex logic.

## Installation

```bash
pip install -e .
```

**Note**: This kernel requires NVIDIA Hopper (SM 9.0) or newer GPUs.

## Quick Start

```python
import torch
import math
from light_duo_attn.kernels import streaming_sparse_attn_func

device = torch.device("cuda")
batch_size, seqlen, num_heads, head_dim = 1, 2048, 32, 128
dtype = torch.bfloat16

# Create input tensors
q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)

# Run streaming attention
output, lse = streaming_sparse_attn_func(
    q, k, v,
    softmax_scale=1.0 / math.sqrt(head_dim),
    causal=True,
    window_size=(31, 0),  # local window size
    sink_size=4,           # number of sink tokens
    enable_streaming=True,
)

print(f"Output shape: {output.shape}")  # [1, 2048, 32, 128]
```

