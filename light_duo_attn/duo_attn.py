import torch
from typing import Any, Tuple, List, Dict, Optional

from light_duo_attn.kernels import streaming_sparse_attn_func
from flash_attn import flash_attn_with_kvcache

def duo_attention_with_kvcache(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    page_table: torch.Tensor,
    softmax_scale: float,
    logit_cap: float,
    window_size: Tuple[int, int],
    gqa_group_size: int,


    # DuoAttention specific parameters
    num_retrieval_heads: int,
    num_streaming_heads: int,
    recent_size: int,
    sink_size: int,
):

    page_size = page_table.shape[1]
    if num_retrieval_heads == 0:
        result, _ = streaming_sparse_attn_func(
            q=q,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,

            seqused_k=cache_seqlens,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(recent_size - 1, 0),
            learnable_sink=None,
            sink_size=sink_size,
            enable_streaming=True,
            softcap=logit_cap,
            pack_gqa=True if gqa_group_size > 1 else False,
            groupwise=False,
            position_ids=None,
            m_block_size=128,
            n_block_size=page_size,
        )
    elif num_streaming_heads == 0:
        result = flash_attn_with_kvcache(
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            softcap=logit_cap,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=None,
        )
    else:
        q_retrieval = q[:, 0:num_retrieval_heads - 1, :]
        q_streaming = q[:, num_retrieval_heads:num_retrieval_heads + num_streaming_heads - 1, :]

        if gqa_group_size == 1:
            kv_retrieval_start = 0
            kv_retrieval_stop = num_retrieval_heads - 1
            kv_streaming_start = num_retrieval_heads
            kv_streaming_stop = num_retrieval_heads + num_streaming_heads - 1
        else:
            kv_retrieval_start = 0
            kv_retrieval_stop = (num_retrieval_heads - 1) // gqa_group_size
            kv_streaming_start = num_retrieval_heads // gqa_group_size
            kv_streaming_stop = (num_retrieval_heads + num_streaming_heads - 1) // gqa_group_size

        k_cache_ret = key_cache[:, :, kv_retrieval_start:kv_retrieval_stop, :]
        v_cache_ret = value_cache[:, :, kv_retrieval_start:kv_retrieval_stop, :]

        k_cache_streaming = key_cache[:, :, kv_streaming_start:kv_streaming_stop, :]
        v_cache_streaming = value_cache[:, :, kv_streaming_start:kv_streaming_stop, :]


        o_retrieval = flash_attn_with_kvcache(
            q=q_retrieval,
            k_cache=k_cache_ret,
            v_cache=v_cache_ret,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            softcap=logit_cap,
            k_descale=None,
            v_descale=None,
            return_softmax_lse=None,
        )

        # For paged KV cache, n_block_size must match the page size in the cache layout
        o_streaming, _ = streaming_sparse_attn_func(
            q=q_streaming,
            k=k_cache_streaming,
            v=v_cache_streaming,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=cache_seqlens,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(recent_size - 1, 0),
            learnable_sink=None,
            sink_size=sink_size,
            enable_streaming=True,
            softcap=logit_cap,
            pack_gqa=True if gqa_group_size > 1 else False,
            groupwise=False,
            position_ids=None,
            m_block_size=128,
            n_block_size=page_size,
        )

        o = torch.empty_like(q)
        o[:, 0:num_retrieval_heads - 1, :] = o_retrieval
        o[:, num_retrieval_heads:num_retrieval_heads + num_streaming_heads - 1, :] = o_streaming
        result = o

    return result


def reorder_weights_for_duo_attn(
    model: Any,
    full_attention_heads: List[List[int]],
    sink_size: int,
    recent_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    hidden_size: int,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Reorder model weights to group retrieval (full attention) and streaming heads together.
    
    This function modifies the attention layer weights in-place to ensure that retrieval heads
    and streaming heads are contiguous in memory, which is required for efficient DuoAttention
    computation. The reordering affects Q, K, V projection weights and the output projection.
    
    Args:
        model: The model object containing layers with attention modules.
               Should have structure: model.model.layers[i].self_attn
        full_attention_heads: A list of lists where each inner list contains binary indicators
                             (1 for full attention, 0 for streaming) for each KV head in that layer.
                             Length must equal number of layers.
        sink_size: Number of initial tokens to always attend to in streaming heads.
        recent_size: Size of the recent token window for streaming heads.
        num_attention_heads: Total number of query heads.
        num_key_value_heads: Total number of key/value heads (for GQA).
        hidden_size: Hidden dimension of the model.
        device: Device to place tensors on. If None, uses the device of the first layer's weights.
    
    Returns:
        A dictionary containing DuoAttention configuration:
        {
            "retrieval_idx": List[slice],  # Slice objects for retrieval heads per layer
            "streaming_idx": List[slice],  # Slice objects for streaming heads per layer
            "sink_size": int,
            "recent_size": int,
            "enable_duo_attention": bool,
        }
    
    Example:
        >>> # For a model with 32 Q heads and 8 KV heads (GQA with group size 4)
        >>> # Mark first 2 KV heads as full attention, rest as streaming
        >>> full_attention_heads = [[1, 1, 0, 0, 0, 0, 0, 0]] * num_layers
        >>> config = reorder_weights_for_duo_attn(
        ...     model=model,
        ...     full_attention_heads=full_attention_heads,
        ...     sink_size=4,
        ...     recent_size=256,
        ...     num_attention_heads=32,
        ...     num_key_value_heads=8,
        ...     hidden_size=4096,
        ... )
    """
    # Initialize configuration
    duo_attn_config = {
        "retrieval_idx": [],
        "streaming_idx": [],
        "sink_size": sink_size,
        "recent_size": recent_size,
        "enable_duo_attention": True,
    }
    
    # Calculate derived parameters
    head_dim = hidden_size // num_attention_heads
    gqa_group_size = num_attention_heads // num_key_value_heads
    
    # Get layers
    layers = model.model.layers
    
    assert len(full_attention_heads) == len(layers), (
        f"full_attention_heads length ({len(full_attention_heads)}) must match "
        f"number of layers ({len(layers)})"
    )
    
    # If device not specified, use the device of the first layer's weights
    if device is None:
        attn = layers[0].self_attn
        if hasattr(attn, 'qkv_proj'):
            device = attn.qkv_proj.weight.device
        elif hasattr(attn, 'q_proj'):
            device = attn.q_proj.weight.device
        else:
            raise ValueError("Cannot determine device from attention layer weights")
    
    # Process each layer
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        
        # --- 1. Define Head Patterns and Indices ---
        # kv_pattern defines which KV head group is full (1) or streaming (0)
        kv_pattern = torch.tensor(
            full_attention_heads[layer_idx], 
            device=device, 
            dtype=torch.int
        )
        assert len(kv_pattern) == num_key_value_heads, (
            f"Layer {layer_idx}: Pattern length ({len(kv_pattern)}) must match "
            f"num_key_value_heads ({num_key_value_heads})"
        )
        
        # Get the indices for full and streaming KV heads
        kv_full_indices = torch.where(kv_pattern == 1)[0]
        kv_stream_indices = torch.where(kv_pattern == 0)[0]
        
        # Expand the KV pattern to the Q heads (for GQA)
        q_head_pattern = torch.repeat_interleave(kv_pattern, repeats=gqa_group_size)
        q_full_indices = torch.where(q_head_pattern == 1)[0]
        q_stream_indices = torch.where(q_head_pattern == 0)[0]
        
        # --- 2. Reorder Q, K, V Projection Weights ---
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        
        if hasattr(attn, 'qkv_proj'):
            # Combined QKV projection (e.g., some Llama implementations)
            _reorder_qkv_proj(
                attn.qkv_proj, 
                q_full_indices, q_stream_indices,
                kv_full_indices, kv_stream_indices,
                q_size, k_size, v_size, 
                num_attention_heads, num_key_value_heads,
                head_dim, hidden_size
            )
        elif hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
            # Separate Q, K, V projections (e.g., standard Llama, Mistral)
            _reorder_separate_qkv_proj(
                attn.q_proj, attn.k_proj, attn.v_proj,
                q_full_indices, q_stream_indices,
                kv_full_indices, kv_stream_indices,
                num_attention_heads, num_key_value_heads,
                head_dim, hidden_size
            )
        else:
            raise ValueError(
                f"Layer {layer_idx}: Unsupported attention architecture. "
                "Expected 'qkv_proj' or 'q_proj'/'k_proj'/'v_proj'"
            )
        
        # --- 3. Reorder O Projection Weights ---
        o_weight = attn.o_proj.weight.data
        # Input to o_proj is concatenation of head outputs. Reorder the columns.
        o_weight_reshaped = o_weight.view(hidden_size, num_attention_heads, head_dim)
        o_weight_reordered = torch.cat([
            o_weight_reshaped[:, q_full_indices, :],
            o_weight_reshaped[:, q_stream_indices, :]
        ], dim=1).view(hidden_size, q_size)
        attn.o_proj.weight.data = o_weight_reordered
        
        # --- 4. Store the new head indices for the attention kernel ---
        num_full_heads = len(q_full_indices)
        duo_attn_config["retrieval_idx"].append(slice(0, num_full_heads))
        duo_attn_config["streaming_idx"].append(slice(num_full_heads, num_attention_heads))
    
    return duo_attn_config


def _reorder_qkv_proj(
    qkv_proj: Any,
    q_full_indices: torch.Tensor,
    q_stream_indices: torch.Tensor,
    kv_full_indices: torch.Tensor,
    kv_stream_indices: torch.Tensor,
    q_size: int,
    k_size: int,
    v_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
) -> None:
    """Reorder weights for combined QKV projection."""
    qkv_weight = qkv_proj.weight.data
    
    # --- Reorder Q weights ---
    w_q = qkv_weight[:q_size, :]
    w_q_reshaped = w_q.view(num_heads, head_dim, hidden_size)
    w_q_reordered = torch.cat([
        w_q_reshaped[q_full_indices],
        w_q_reshaped[q_stream_indices]
    ], dim=0).view(q_size, hidden_size)
    
    # --- Reorder K weights ---
    w_k = qkv_weight[q_size : q_size + k_size, :]
    w_k_reshaped = w_k.view(num_kv_heads, head_dim, hidden_size)
    w_k_reordered = torch.cat([
        w_k_reshaped[kv_full_indices],
        w_k_reshaped[kv_stream_indices]
    ], dim=0).view(k_size, hidden_size)
    
    # --- Reorder V weights ---
    w_v = qkv_weight[q_size + k_size :, :]
    w_v_reshaped = w_v.view(num_kv_heads, head_dim, hidden_size)
    w_v_reordered = torch.cat([
        w_v_reshaped[kv_full_indices],
        w_v_reshaped[kv_stream_indices]
    ], dim=0).view(v_size, hidden_size)
    
    # Combine back into a single QKV weight
    new_qkv_weight = torch.cat([w_q_reordered, w_k_reordered, w_v_reordered], dim=0)
    qkv_proj.weight.data = new_qkv_weight


def _reorder_separate_qkv_proj(
    q_proj: Any,
    k_proj: Any,
    v_proj: Any,
    q_full_indices: torch.Tensor,
    q_stream_indices: torch.Tensor,
    kv_full_indices: torch.Tensor,
    kv_stream_indices: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
) -> None:
    """Reorder weights for separate Q, K, V projections."""
    # --- Reorder Q weights ---
    w_q = q_proj.weight.data
    w_q_reshaped = w_q.view(num_heads, head_dim, hidden_size)
    w_q_reordered = torch.cat([
        w_q_reshaped[q_full_indices],
        w_q_reshaped[q_stream_indices]
    ], dim=0).view(num_heads * head_dim, hidden_size)
    q_proj.weight.data = w_q_reordered
    
    # --- Reorder K weights ---
    w_k = k_proj.weight.data
    w_k_reshaped = w_k.view(num_kv_heads, head_dim, hidden_size)
    w_k_reordered = torch.cat([
        w_k_reshaped[kv_full_indices],
        w_k_reshaped[kv_stream_indices]
    ], dim=0).view(num_kv_heads * head_dim, hidden_size)
    k_proj.weight.data = w_k_reordered
    
    # --- Reorder V weights ---
    w_v = v_proj.weight.data
    w_v_reshaped = w_v.view(num_kv_heads, head_dim, hidden_size)
    w_v_reordered = torch.cat([
        w_v_reshaped[kv_full_indices],
        w_v_reshaped[kv_stream_indices]
    ], dim=0).view(num_kv_heads * head_dim, hidden_size)
    v_proj.weight.data = w_v_reordered
