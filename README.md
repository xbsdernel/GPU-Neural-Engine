# GPU Neural Engine

**Optimized GPU Kernels for Deep Learning**

---

## Description
**GPU Neural Engine** is a GPU kernel library implementing core deep learning operations with advanced performance optimizations. The project focuses on:

- **Matrix Multiplication (GEMM), Convolutions, Reductions, and Activations**
- **Warp-level optimization:** fast intra-warp communication
- **Register blocking:** store partial results in registers
- **Loop unrolling:** maximize inner loop throughput
- **Mixed precision / Tensor Cores support:** FP16, FP32, INT8

---

## Goal
Learn and apply industrial GPU optimization techniques similar to **cuBLAS** and **cuDNN**, while building highly efficient kernels from scratch.
