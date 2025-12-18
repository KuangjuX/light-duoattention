# Light-DuoAttention

A lightweight CuTe-based CUDA kernel for DuoAttention, optimized for large language model inference.

## Overview

`Light-DuoAttention` provides a lightweight, fused CUDA kernel for **DuoAttention**. This attention mechanism is a hybrid approach designed for efficient long-context processing. 
Unlike standard attention where all heads behave identically, DuoAttention employs two distinct types of heads:

1.  **Retrieval Heads**: These heads perform **Full Attention** over the entire sequence (or a large chunk), allowing them to capture rich, long-range dependencies and act as a powerful information retrieval component.
2.  **Streaming Heads**: These heads use **Streaming Attention**, processing tokens in a sliding-window or fixed-cache manner. They are highly efficient and ideal for handling very long sequences while maintaining local context.

This hybrid approach allows a model to balance expressive power with computational efficiency. 
This kernel is implemented using CuTeDSL, the core of CUTLASS 4.x, enabling a clean and optimized implementation of this complex logic.

