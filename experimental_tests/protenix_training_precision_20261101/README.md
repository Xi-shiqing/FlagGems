# Protenix 训练算子修补与实验归档

日期：2026-11-01。主源码位于本仓库 `src/flag_gems/`，完整训练数据及模型输出比较见 [Protenix 实验目录](https://github.com/Xi-shiqing/Protenix-FlagOS/tree/master/comparison)。

| 工作 | 保存位置 | 验证结果与使用状态 |
|---|---|---|
| 重复索引累加、unfold 反向、Softmax 反向求和顺序 | 主源码及对应 `tests/` | 固定重复执行时的累加顺序，补齐非连续输入、重复索引及空输入检查 |
| Softmax 反向融合 V2 | `patches/fused_softmax_backward_v2_flaggems_20261024.patch` | 实际 MSA 反向接口从约 195 微秒降到 122 微秒，FP32/TF32 各两步训练的完整梯度、Adam、更新及 loss 与融合前一致；以独立补丁保存 |
| 直接读取非连续 Softmax 输入 V3/V4 | `patches/`、`reports/STRIDED_SOFTMAX_BACKWARD_V3_FINDINGS_20261024.md` | 完整反向数值检查失败，停止采用，保存实现与原因 |
| 长求和矩阵乘法分段 | `validation/prepare_long_reduction_splitk_candidate_cpu_v2.py`、`reports/GENERIC_LONG_REDUCTION_MM_FINDINGS.md` | 真实小输出矩阵调用加速约 32 倍；完整模块部分梯度误差下降。两步训练的 TF32 全局梯度误差略增，保留为实验实现 |
| TF32 多次乘积、平衡拆分及其他长求和方案 | `patches/` | 保留独立方案及完整训练结果链接，具体结论见对应报告 |
| 早期开发工作区 | `historical_worktrees/` | 按各自基准提交保存完整差异；恢复时使用 `MANIFEST.json` 指定的基准提交 |

`MANIFEST.json` 记录补丁、验证程序和历史差异的来源与 SHA-256。实验补丁具有各自的前置版本，V3 相对 V2、V4 相对 V3；根据相应构建程序恢复实验源码。主源码提交与实验归档分别保存，便于复核实际启用的实现。

## 数据入口

- [Softmax 融合验证](reports/FUSED_SOFTMAX_BACKWARD_V2_20261024.md)
- [长求和矩阵乘法计时、数值和两步训练](reports/GENERIC_LONG_REDUCTION_MM_FINDINGS.md)
- [长求和矩阵乘法完整两步训练比较](reports/REAL_TWO_STEP_GENERIC_LONG_REDUCTION_TRAINING.md)
- [已有百步实验结果](reports/TRUNK_LONGK_FINDINGS_20261025_FINAL.md)
- [H100 真实反向调用与舍入对照](https://github.com/Xi-shiqing/Protenix-FlagOS/tree/master/comparison/h100_gradient_comparison_20261101)
- [两周工作汇报](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/docs/05_meetings/2026-11-01.md)
