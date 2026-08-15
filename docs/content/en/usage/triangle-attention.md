---
title: Triangle Attention
weight: 85
---

# Triangle Attention

`flag_gems.ops.triangle_attention` is a standalone streaming Triangle Attention operator for structure-prediction workloads. It applies an additive validity mask and pair bias, performs online softmax, and does not materialize the full
attention probability tensor. It has no dependency on an application framework.

```python
from flag_gems.ops import triangle_attention

output = triangle_attention(
    q,
    k,
    v,
    bias1,
    bias2,
    precision="ieee",
    layout="BNSHD",
)
```

For `BNSHD`, Q, K and V use `(B, N, S, H, D)`. `BNHSD` is also supported and uses `(B, N, H, S, D)`. The additive mask `bias1` uses `(B, N, 1, 1, S)` and must not require gradients. The pair bias `bias2` uses `(B, 1, H, S, S)`.

Version 0.1.0 supports FP32, BF16 and FP16 inputs, head dimensions 16, 32 and 64, non-contiguous inputs, inference, and first-order gradients for Q, K, V and bias2. Bias tensors may use the input dtype or FP32. Higher-order gradients are not supported.

The default `ieee` mode is the precision reference. FP32 users may explicitly select `tf32`, `qk_tf32`, `pv_tf32`, `tf32x3`, `qk_tf32x3`, or `pv_tf32x3`. These modes are never selected implicitly; applications must validate their end-to-end accuracy before enabling them.
