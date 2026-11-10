# T-Head Zhenwu (真武) PPU Backend

## Overview

This backend provides support for Alibaba Cloud's T-Head Zhenwu PPU (Processing Processing Unit) accelerators through the FlagGems framework.

## Hardware Information

**Product**: Zhenwu 810E PPU
**Architecture**: Proprietary T-Head AI accelerator
**SDK**: PPU SDK v2.0.0+
**Device Management**: `ppu-smi` (similar to nvidia-smi)

### Key Features

- **CUDA Compatibility**: Full compatibility with CUDA Runtime and Driver APIs
- **Tensor Core**: Extended PTX instructions for AIU (AI Unit)
- **Memory**: High-bandwidth memory with optimized access patterns
- **Multi-card**: ICN interconnect support (up to 16 cards)
- **Virtualization**: MIG (up to 8 instances) and SRIOV support
- **Triton Support**: Versions 2.3.x, 3.0.x - 3.4.x with AIU extensions

## Software Stack

### Core Components

| Component | Description |
|-----------|-------------|
| Firmware | PPU firmware with dynamic power management |
| KMD | Kernel mode driver |
| UMD/HGGC | User mode driver and runtime |
| Compiler | Clang/LLVM-based compiler with CUDA C/C++ compatibility |
| Acompute | Computing acceleration libraries (acdnn, acblas, acfft, etc.) |
| PCCL | Communication acceleration library |
| PPU-SMI | Device management tool |
| Asight | Performance analysis tools |

### Acceleration Libraries

- **acdnn**: Deep neural network operators (Conv, BatchNorm, Pooling, Softmax, etc.)
- **acblas**: BLAS operations (GEMM, GEMV, Matrix transformations)
- **acfft**: FFT transformations (R2C, C2R, C2C, D2Z, Z2Z)
- **acsolver**: Linear algebra solvers (LU, Cholesky, QR, SVD)
- **acrand**: Random number generation (XORWOW, MRG32K3A, PHILOX4_32_10)
- **acsparse**: Sparse matrix operations

## Configuration

### Environment Variables

```bash
# Specify T-Head PPU backend
export GEMS_VENDOR=thead

# Optional: Configure PPU-specific settings
export HGGC_EXCLUSIVE_STREAMS=1  # Map streams to different hardware queues

```

### FP32 fused gating

The PPU backend exposes `flag_gems.sigmoid_mul(gate, value)`,
`flag_gems.sigmoid_mul_add(gate, value, shift)` and
`flag_gems.silu_mul(gate, value)`. These APIs accept FP32 tensors on the same
PPU, with compatible broadcasting and strided layouts. They preserve the
rounding of the separate PPU FlagGems operations, including separate rounded
multiplication and addition. The matrix-multiplication precision setting does
not change these FP32 elementwise operations.

The APIs support first-order autograd. Backward preserves Add and Mul broadcast
reductions before applying the existing activation derivative. Training stages
the original activation where required to retain that ordering. Higher-order
backward is rejected by `once_differentiable`.

Use these APIs inside the normal `flag_gems.use_gems()` context. The Protenix
integration uses them for adaptive normalization and conditioned transitions;
the dispatch depends on dtype, layout and element count, without model or input
identifiers.
