import torch
from typing import Tuple

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
            **kwargs,
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
            **kwargs,
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
