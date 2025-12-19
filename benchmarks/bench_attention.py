"""
Benchmark script to compare Streaming Attention vs. FlashAttention performance.

This script measures throughput, latency, and memory usage for different configurations.
"""

import argparse
import math
import time
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn.functional as F

# Import custom streaming attention
from light_duo_attn.kernels import streaming_sparse_attn_func

# Try to import FlashAttention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash-attn not installed. Install with: pip install flash-attn")


def benchmark_function(
    func,
    *args,
    num_warmup: int = 10,
    num_iters: int = 100,
    **kwargs
) -> Tuple[float, float]:
    """
    Benchmark a function and return average time and throughput.
    
    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (avg_time_ms, throughput_gflops)
    """
    # Warmup
    for _ in range(num_warmup):
        _ = func(*args, **kwargs)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iters):
        _ = func(*args, **kwargs)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_iters
    
    return avg_time_ms


def compute_attention_flops(
    batch_size: int,
    seqlen: int,
    num_heads: int,
    head_dim: int,
    causal: bool = False
) -> float:
    """
    Compute approximate FLOPs for attention operation.
    
    Args:
        batch_size: Batch size
        seqlen: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        causal: Whether attention is causal
        
    Returns:
        Number of FLOPs
    """
    # QK^T: batch_size * num_heads * seqlen * seqlen * head_dim
    qk_flops = batch_size * num_heads * seqlen * seqlen * head_dim
    
    # Softmax (approximate)
    softmax_flops = batch_size * num_heads * seqlen * seqlen * 5  # exp, sum, div
    
    # Attention @ V: batch_size * num_heads * seqlen * seqlen * head_dim
    av_flops = batch_size * num_heads * seqlen * seqlen * head_dim
    
    total_flops = qk_flops + softmax_flops + av_flops
    
    # For causal attention, roughly half the operations
    if causal:
        total_flops = total_flops * 0.5
    
    return total_flops


def run_streaming_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    recent_size: int,
    sink_size: int,
    enable_streaming: bool = True
) -> torch.Tensor:
    """Run streaming attention."""
    output, _ = streaming_sparse_attn_func(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(recent_size - 1, 0),
        sink_size=sink_size,
        enable_streaming=enable_streaming,
    )
    return output


def run_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool
) -> torch.Tensor:
    """Run FlashAttention."""
    if not HAS_FLASH_ATTN:
        raise RuntimeError("flash-attn is not installed")
    
    output = flash_attn_func(
        q, k, v,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return output


def benchmark_config(
    batch_size: int,
    seqlen: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    causal: bool,
    recent_size: int,
    sink_size: int,
    num_warmup: int = 10,
    num_iters: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Benchmark a specific configuration.
    
    Returns:
        Dictionary with benchmark results
    """
    # Create input tensors
    q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=dtype, device=device)
    
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    results = {
        "batch_size": batch_size,
        "seqlen": seqlen,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "dtype": str(dtype),
    }
    
    # Compute FLOPs
    flops = compute_attention_flops(batch_size, seqlen, num_heads, head_dim, causal)
    results["flops"] = flops
    
    # Benchmark Streaming Attention
    try:
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        avg_time_ms = benchmark_function(
            run_streaming_attention,
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            recent_size=recent_size,
            sink_size=sink_size,
            enable_streaming=True,
            num_warmup=num_warmup,
            num_iters=num_iters
        )
        
        peak_mem = torch.cuda.max_memory_allocated()
        mem_used_mb = (peak_mem - start_mem) / 1024 / 1024
        
        results["streaming_time_ms"] = avg_time_ms
        results["streaming_tflops"] = (flops / (avg_time_ms / 1000)) / 1e12
        results["streaming_memory_mb"] = mem_used_mb
        
    except Exception as e:
        print(f"Streaming Attention failed: {e}")
        results["streaming_time_ms"] = None
        results["streaming_tflops"] = None
        results["streaming_memory_mb"] = None
    
    # Benchmark FlashAttention
    if HAS_FLASH_ATTN:
        try:
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            
            avg_time_ms = benchmark_function(
                run_flash_attention,
                q, k, v,
                softmax_scale=softmax_scale,
                causal=causal,
                num_warmup=num_warmup,
                num_iters=num_iters
            )
            
            peak_mem = torch.cuda.max_memory_allocated()
            mem_used_mb = (peak_mem - start_mem) / 1024 / 1024
            
            results["flash_time_ms"] = avg_time_ms
            results["flash_tflops"] = (flops / (avg_time_ms / 1000)) / 1e12
            results["flash_memory_mb"] = mem_used_mb
            
        except Exception as e:
            print(f"FlashAttention failed: {e}")
            results["flash_time_ms"] = None
            results["flash_tflops"] = None
            results["flash_memory_mb"] = None
    else:
        results["flash_time_ms"] = None
        results["flash_tflops"] = None
        results["flash_memory_mb"] = None
    
    # Compute speedup
    if results["streaming_time_ms"] and results["flash_time_ms"]:
        results["speedup"] = results["flash_time_ms"] / results["streaming_time_ms"]
    else:
        results["speedup"] = None
    
    return results


def print_results(results: List[Dict[str, float]]):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*120)
    print("Benchmark Results: Streaming Attention vs. FlashAttention")
    print("="*120)
    
    # Header
    header = f"{'Config':<30} | {'Streaming':<35} | {'FlashAttn':<35} | {'Speedup':<10}"
    print(header)
    print("-"*120)
    
    for result in results:
        config_str = f"B={result['batch_size']}, L={result['seqlen']}, H={result['num_heads']}, D={result['head_dim']}"
        
        streaming_str = ""
        if result['streaming_time_ms']:
            streaming_str = f"{result['streaming_time_ms']:.3f}ms, {result['streaming_tflops']:.2f}TF, {result['streaming_memory_mb']:.1f}MB"
        else:
            streaming_str = "Failed"
        
        flash_str = ""
        if result['flash_time_ms']:
            flash_str = f"{result['flash_time_ms']:.3f}ms, {result['flash_tflops']:.2f}TF, {result['flash_memory_mb']:.1f}MB"
        else:
            flash_str = "N/A"
        
        speedup_str = ""
        if result['speedup']:
            speedup_str = f"{result['speedup']:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{config_str:<30} | {streaming_str:<35} | {flash_str:<35} | {speedup_str:<10}")
    
    print("="*120)
    print("\nLegend: B=Batch Size, L=Sequence Length, H=Num Heads, D=Head Dim, TF=TFLOPS, MB=Memory (MB)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Streaming Attention vs. FlashAttention"
    )
    
    # Configuration options
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1],
                        help="Batch sizes to test")
    parser.add_argument("--seqlens", type=int, nargs="+", default=[4096, 8192, 16384],
                        help="Sequence lengths to test")
    parser.add_argument("--num-heads", type=int, nargs="+", default=[32],
                        help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Head dimension")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Data type")
    parser.add_argument("--causal", action="store_true",
                        help="Use causal attention")
    parser.add_argument("--recent-size", type=int, default=256,
                        help="Recent size for streaming attention")
    parser.add_argument("--sink-size", type=int, default=128,
                        help="Number of sink tokens for streaming attention")
    
    # Benchmark options
    parser.add_argument("--num-warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--num-iters", type=int, default=50,
                        help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    
    args = parser.parse_args()
    
    # Convert dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"Flash-Attn Available: {HAS_FLASH_ATTN}")
    print()
    
    # Run benchmarks
    all_results = []
    
    total_configs = len(args.batch_sizes) * len(args.seqlens) * len(args.num_heads)
    current_config = 0
    
    for batch_size in args.batch_sizes:
        for seqlen in args.seqlens:
            for num_heads in args.num_heads:
                current_config += 1
                print(f"Running config {current_config}/{total_configs}: "
                      f"B={batch_size}, L={seqlen}, H={num_heads}, D={args.head_dim}")
                
                results = benchmark_config(
                    batch_size=batch_size,
                    seqlen=seqlen,
                    num_heads=num_heads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    causal=args.causal,
                    recent_size=args.recent_size,
                    sink_size=args.sink_size,
                    num_warmup=args.num_warmup,
                    num_iters=args.num_iters,
                    device=args.device
                )
                
                all_results.append(results)
    
    # Print results
    print_results(all_results)
    
    # # Save results to CSV
    # try:
    #     import csv
    #     import datetime
        
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"benchmark_results_{timestamp}.csv"
        
    #     with open(filename, 'w', newline='') as csvfile:
    #         fieldnames = list(all_results[0].keys())
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
    #         writer.writeheader()
    #         for result in all_results:
    #             writer.writerow(result)
        
    #     print(f"Results saved to {filename}")
    # except Exception as e:
    #     print(f"Failed to save CSV: {e}")


if __name__ == "__main__":
    main()

