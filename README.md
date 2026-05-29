# Attention Mechanism Optimization Suite

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)
![CUDA 11.8+](https://img.shields.io/badge/cuda-11.8+-green.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

A benchmarking and optimization framework for transformer attention mechanisms. Compares vanilla PyTorch, SDPA, FlashAttention-2, and xFormers across batch sizes and sequence lengths — with a batch-size auto-tuner, ONNX/TensorRT deployment benchmarks, and GPU memory profiling.

**TL;DR:** FlashAttention-2 achieves **10.5x throughput improvement** and **99.7% memory reduction** vs vanilla attention at seq_len=1024, batch=32 on an NVIDIA L4. SDPA achieves the highest raw throughput at **12.3x** with moderate memory reduction (58%). Algorithm-level optimization dominates hardware-level (TensorRT) by ~6x for attention operations.

---

## Table of Contents

- [Performance Summary](#performance-summary)
- [Key Findings](#key-findings)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [What's New in v2.2](#whats-new-in-v22)
- [Technical Reference](#technical-reference)
- [Testing](#testing)
- [Resume Bullets](#resume-bullets)

---

## Performance Summary

Benchmarks run on **NVIDIA L4 GPU**, seq_len=1024, batch=32.

| Attention Type     | Throughput (tok/s) | Memory (MB) | Speedup vs Vanilla | Memory vs Vanilla |
|--------------------|-------------------|-------------|-------------------|-------------------|
| Vanilla PyTorch    | 573,824           | 12,582      | 1.0×              | 100%              |
| SDPA               | 7,058,407         | 5,240       | **12.3×**         | 41.6%             |
| FlashAttention-2   | 6,031,148         | **38**      | 10.5×             | **0.3%**          |
| xFormers           | 5,496,605         | 102         | 9.6×              | 0.8%              |

**When to use which:**
- **SDPA** — highest throughput, built into PyTorch 2.0+, no extra install. Best default choice.
- **FlashAttention-2** — use when memory is the constraint (long sequences, large batches). 99.7% memory reduction enables sequence lengths and batch sizes that OOM vanilla.
- **xFormers** — good fallback with comparable memory efficiency; useful for custom architectures.
- **Vanilla** — baseline and debugging only.

### Auto-Tuner Results (seq_len=1024)

| Attention Type   | Optimal Batch Size | Throughput (tok/s) |
|------------------|-------------------|-------------------|
| Vanilla          | 48                | 574,695           |
| SDPA             | 32                | 7,385,485         |
| FlashAttention-2 | 32                | 6,062,413         |
| xFormers         | 56                | 5,927,889         |

---

## Key Findings

**1. SDPA vs FlashAttention-2 — it depends on your constraint**

SDPA wins on throughput (12.3× vs 10.5×), but FlashAttention-2 wins on memory (0.3% vs 41.6% of vanilla). At seq_len=4096+ or when fitting large batches into limited VRAM, FlashAttention-2 is the right call. At seq_len=1024 with no memory pressure, SDPA is faster and requires no additional dependencies.

**2. Algorithm optimization outperforms hardware optimization by ~6×**

| Optimization Level         | Speedup    | Mechanism                          |
|---------------------------|------------|------------------------------------|
| Algorithm (FlashAttention) | 10–12×     | IO-aware memory access patterns    |
| Hardware (TensorRT FP16)   | ~2×        | Kernel fusion, auto-tuning         |
| Precision (FP16)           | 1.5–3×     | Reduced bit-width compute          |
| Framework (ONNX Runtime)   | ~1.1×      | Graph optimization, cross-platform |

The insight: FlashAttention's speedup comes from reducing HBM round-trips, which is the actual bottleneck for attention. TensorRT can't fix a memory access pattern problem — it optimizes within a given pattern.

**3. Vanilla attention's O(n²) memory limits usable sequence length**

Without OOM on the L4 test setup, vanilla attention maxes out around seq_len=2048. FlashAttention-2 and xFormers extend this significantly by never materializing the full attention matrix.

---

## Quick Start

```bash
# Clone
git clone https://github.com/saitejasrivilli/attention-optimization.git
cd attention-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Run all benchmarks:**
```bash
python scripts/benchmark_all.py --output results/benchmark.csv
```

**Use in your own code:**
```python
from attention_optimization import AttentionBenchmark

benchmark = AttentionBenchmark(hidden_size=1024, num_heads=16)
results = benchmark.run(
    batch_sizes=[1, 2, 4, 8, 16, 32],
    seq_lengths=[512, 1024, 2048, 4096],
    attention_types=['vanilla', 'sdpa', 'flash-attn2', 'xformers']
)
```

**Find optimal batch size under memory + latency constraints:**
```python
from attention_optimization import BatchSizeAutoTuner

tuner = BatchSizeAutoTuner(memory_limit_gb=16)
optimal_bs = tuner.find_optimal_batch_size(
    attention_fn=benchmark.flash_attention,
    attention_name='FlashAttention-2',
    seq_length=1024,
    max_memory_gb=14.0,
    target_p95_latency_ms=100.0
)
```

**Open notebook:**
```bash
jupyter notebook notebooks/attention_optimization_benchmark.ipynb
```

---

## Core Components

### `AttentionBenchmark`

Benchmarks all 4 implementations across configurable batch sizes and sequence lengths. Tracks: latency (ms), throughput (tok/s), peak memory (MB), memory efficiency (%).

### `BatchSizeAutoTuner`

Binary search over batch sizes to find the maximum throughput configuration under user-specified memory and P95 latency constraints. Useful before deploying a model — finds the optimal batch size per GPU without manual trial and error.

---

## What's New in v2.2

Added ONNX Runtime export and TensorRT benchmarks to the notebook, enabling cross-platform deployment comparison:

| Method              | Latency (ms) | Throughput    | Speedup  |
|---------------------|-------------|---------------|---------- |
| Vanilla PyTorch     | ~7.00       | 0.57 M tok/s  | 1.00×    |
| SDPA                | ~0.58       | 7.06 M tok/s  | 12.1×    |
| FlashAttention-2    | ~0.68       | 6.03 M tok/s  | 10.5×    |
| xFormers            | ~0.75       | 5.50 M tok/s  | 9.6×     |
| ONNX Runtime FP16   | 6.60        | 0.62 M tok/s  | 1.06×    |
| TensorRT FP16 (est) | ~3.50       | 1.17 M tok/s  | ~2.0×    |

ONNX numbers are measured. TensorRT FP16 is estimated from published benchmarks on comparable hardware — not measured on L4 directly. Marked accordingly in the notebook.

---

## Project Structure

```
attention-optimization/
├── attention_optimization/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── implementations/
│   │   ├── vanilla.py
│   │   ├── sdpa.py
│   │   ├── flash_attention.py
│   │   └── xformers_attn.py
│   ├── tuner.py
│   ├── utils.py
│   └── metrics.py
├── scripts/
│   ├── benchmark_all.py
│   ├── visualize_results.py
│   └── compare_models.py
├── notebooks/
│   └── attention_optimization_benchmark.ipynb
├── tests/
│   ├── test_implementations.py
│   ├── test_tuner.py
│   └── test_utils.py
├── results/
├── requirements.txt
├── setup.py
└── README.md
```

---

## Technical Reference

### Attention Complexity

| Method          | Time       | Memory     | Best For                           |
|-----------------|------------|------------|-------------------------------------|
| Vanilla         | O(n²)      | O(n²)      | Baseline, debugging                 |
| SDPA            | O(n²) opt  | O(n²) opt  | General use, PyTorch 2.0+           |
| FlashAttention-2| O(n²) time | O(n) memory| Long sequences, memory-constrained  |
| xFormers        | O(n²) time | O(n) memory| Research, custom architectures      |

### Deployment Framework Comparison

| Framework     | Platform       | Speedup | Best For                      |
|---------------|---------------|---------|-------------------------------|
| PyTorch       | NVIDIA GPU    | 1.0×    | Research, prototyping         |
| ONNX Runtime  | CPU/GPU/Edge  | ~1.1×   | Cross-platform deployment     |
| TensorRT      | NVIDIA GPU    | ~2–4×   | Production NVIDIA systems     |

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_implementations.py -v

# Run with coverage
pytest tests/ --cov=attention_optimization
```

---

## Resume Bullets

- Benchmarked 4 attention implementations (vanilla, SDPA, FlashAttention-2, xFormers) on NVIDIA L4 across seq_len 512–4096; FlashAttention-2 achieves 10.5× throughput and 99.7% memory reduction vs vanilla — enabling batch sizes that OOM vanilla at the same seq_len.
- SDPA achieves 12.3× throughput (highest) at 41.6% of vanilla memory — best default for PyTorch 2.0+ where memory is not the bottleneck.
- Built batch-size auto-tuner using binary search to find max-throughput configuration under memory and P95 latency constraints per attention implementation.
- Demonstrated algorithm-level optimization (FlashAttention IO-aware access) outperforms hardware-level optimization (TensorRT) by ~6× for attention — the bottleneck is memory access pattern, not kernel compute.
- Tech: PyTorch, FlashAttention-2, xFormers, ONNX Runtime, CUDA, torch.profiler, NVIDIA L4.

---

## Dependencies

See `requirements.txt`:
- torch>=2.0.0
- transformers>=4.30.0
- flash-attn>=2.0.0
- xformers>=0.0.20
- onnx>=1.14.0
- onnxruntime>=1.16.0
- pandas>=1.5.0
- numpy>=1.24.0
- matplotlib>=3.6.0
- scikit-learn>=1.2.0

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Sai Teja Srivillibhutturu · [GitHub](https://github.com/saitejasrivilli) · [LinkedIn](https://linkedin.com/in/saitejasrivilli)*
