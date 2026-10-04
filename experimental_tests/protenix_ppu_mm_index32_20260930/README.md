# PPU TF32 MM：安全的 32 位索引候选

该实验保留原来的 tile、Split-K、RNE 舍入、FP32 累加与算子分派，仅在 TF32 直接 MM 的逻辑维度及掩码覆盖地址均能用有符号 32 位整数表示时缩小索引位宽。其他情况和严格 FP32 继续使用 64 位索引。范围检查包含 Split-K 最后一块的填充区域，并复用一次读取的 stride 和 TF32 开关。

实验基线为 native RNE + fast fill 的隔离 FlagGems 包。基线 `ops/mm.py` SHA256 必须是 `5288e11181d297200306812fed621b1f54bc0bf8aa5d43fcec84dc14f7e7e43b`。最终候选 SHA256 是 `fe8686e66bab01824f5bfe28eba019ba9628a436cae5406ed6a65c3f987e7205`。

```bash
python build_candidate.py \
  --baseline /path/to/native_rne_fast_fill_overlay \
  --candidate /path/to/index32_overlay \
  --patch /path/to/mm_index32.patch \
  --cache-call-arguments --split-k-padding
```

`mm_index32.patch` 是最终补丁；`validation.json` 记录算子及训练探针的输入/输出哈希审计。GPU 计时、端到端输入审计与精度数据保存在 Protenix 仓库的 `comparison/ppu_tf32_precision_20260923/vendor_gemm_20260930/`。这是独立实验候选，是否采用需要由跨输入的端到端结果决定。

验证覆盖 25 组真实激活算子、50 个前向结果哈希及 62 个训练前向/梯度/更新哈希，均与控制实现逐位一致。三组大 MM 配对计时降低约 16.0%、17.1%、23.7%；小矩阵变化为 -0.2% 到 +2.3%，没有承诺小矩阵普遍提速。完整 FlagGems 模型配对中位数降低约 2.5%，但 95% 配对自助法区间包含零，不能宣称稳定的端到端加速。`mm_paired_matrix.csv` 和 `paired_model_statistics.json` 记录对应数据。
