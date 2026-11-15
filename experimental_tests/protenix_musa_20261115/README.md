# Protenix MUSA · 2026-11-15

- [张石健 2026-11-15 实验报告](张石健20261115实验报告.md)
- [对照实验和验证记录](data/对照与验证.md)
- [逐次预测时间和显存](data/predict_timings.json) · [三种子 MFU 和坐标汇总](data/mfu_and_accuracy.json) · [实验协议](data/protocol.json)
- [Linear 权重重排](code/linear_rhs_layout.py) · [扩散分组内核](code/token_linear_kernel.py) · [阶段探针](code/stage_probe.py)
- [TriAttention 的 MUSA 适配](code/lai_triangle_musa.py) · [赖冠熹的原始实现](https://github.com/Xi-shiqing/FlagGems/commit/1ed2f470c5105a3ac8bb1cf875fd16f31744f123)

运行数据来自严格 FP32 的 7bca 推理。算子修改由 Protenix 推理驱动接入，适配代码和核心内核存档在这里。对照使用同精度的 H100 原生推理。数据文件提供计时、显存、阶段区间、精度汇总和算子实际调用，合成的单算子检查使用 CPU FP64 参考。
