# Attention Mechanism Optimization Suite

Comprehensive benchmarking framework for evaluating and optimizing transformer attention implementations on real pretrained weights. Compares vanilla PyTorch, SDPA, FlashAttention-2, xFormers, ONNX Runtime, and TensorRT across different sequence lengths and batch sizes.

## Key Results

| Method | Latency (ms) | Throughput | Speedup |
|--------|------------|-----------|---------|
| Vanilla PyTorch | 7.00 | 0.57M tok/s | 1.0x |
| SDPA | 0.58 | 7.06M tok/s | **12.1x** |
| FlashAttention-2 | 0.68 | 6.03M tok/s | **10.5x** |
| xFormers | 0.75 | 5.50M tok/s | **9.6x** |
| ONNX Runtime (FP16) | 0.85 | 4.82M tok/s | **8.2x** |
| TensorRT (FP16) | 3.50 | 1.17M tok/s | **2.0x** |

**Configuration:** batch_size=4, seq_len=1024, hidden_dim=1024, FP16 precision, CUDA

## Findings

- **SDPA achieves 12.1x speedup** through IO-aware memory access patterns and kernel fusion
- **Algorithm-level optimizations outperform hardware-level** by 6x for attention operations  
- **SDPA provides best balance** of performance and ease of integration into PyTorch workflows
- **FlashAttention-2 achieves 10.5x improvement** with minimal code changes
- **JAX implementation** available for cross-framework comparison

## Model Details

- **Base Model:** meta-llama/Llama-3.2-1B
- **Precision:** FP16 pretrained weights
- **Extraction Method:** Forward hooks on real forward pass
- **No random initialization** — uses actual pretrained weights

## Files

- `attention_optimization_benchmark.py` — Runnable benchmark script
- `attention_optimization_benchmak.ipynb` — Interactive Jupyter notebook with plots
- `attention-jax-implementation.py` — JAX/XLA implementation for reference
- `attention_benchmark_results.png` — Benchmark results visualization

## Quick Start

```python
python attention_optimization_benchmark.py
```

## Requirements

- PyTorch 2.0+
- CUDA 11.8+
- transformers (for Llama model)
- FlashAttention-2
- xFormers
- onnxruntime-gpu
- tensorrt (optional)

## Methodology

1. Extract attention operations from Llama-3.2-1B forward pass via hooks
2. Benchmark each implementation independently with same input
3. Measure latency, throughput, and memory consumption
4. Analyze performance across batch sizes (1-32) and sequence lengths (256-4096)
5. Export optimized implementations to ONNX/TensorRT for production

## Key Takeaway

Use SDPA for best PyTorch performance; use ONNX Runtime or TensorRT for cross-platform deployment and further optimization headroom.
