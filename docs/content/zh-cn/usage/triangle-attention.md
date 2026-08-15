---
title: Triangle Attention
weight: 85
---

# Triangle Attention

`flag_gems.ops.triangle_attention` 是面向结构预测任务的独立流式 Triangle Attention 算子。它融合加法掩码与 pair bias，使用在线 softmax，不生成完整的注意力概率张量；实现不依赖具体应用框架。

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

`BNSHD` 布局下 Q、K、V 的形状是 `(B, N, S, H, D)`；也支持形状为 `(B, N, H, S, D)` 的 `BNHSD` 布局。加法掩码 `bias1` 的形状是 `(B, N, 1, 1, S)`，不可参与梯度计算；`bias2` 的形状是 `(B, 1, H, S, S)`。

0.1.0 版本支持 FP32、BF16、FP16，head dimension 16/32/64，支持非连续输入、推理，以及 Q、K、V、bias2 的一阶反向传播。bias 可与输入同类型，也可为 FP32；暂不支持高阶梯度。

默认 `ieee` 是精度基准。FP32 输入还可显式选择 `tf32`、`qk_tf32`、`pv_tf32`、`tf32x3`、`qk_tf32x3`、`pv_tf32x3`。算子不会自动开启这些模式，应用应在完成端到端精度验证后再选择。
