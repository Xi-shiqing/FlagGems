# FlagGems 上游合并实验（2026-11-06）

这份目录归档 FlagGems 上游合并的实现比较与验收。部署控制程序依赖 Protenix 项目已有的连接脚本，从 [Protenix 对应目录](https://github.com/Xi-shiqing/Protenix-FlagOS/tree/master/comparison/upstream_merge_20261106) 运行。

- [实现选择与结果](REPORT.md)
- [机器可读汇总](SUMMARY.json)
- [源码比较的全部逐项数据](conflict_source_cases.csv)
- [实验复核](EXPERIMENT_AUDIT.md)

## 固定版本与环境

项目起点 `9583da7ab090e0f54498a23b2fe2e9a9e35abe7d`，上游 `45339c4ad01d237d4fe66db197afb817eb9c6c60`，共同祖先 `b550ddb996939ee1a30ba4a4b8f8b2d0556703a3`。原 PPU-ZW810E 容器，PyTorch 2.10.0、Triton 3.5.0；镜像 `sha256:307ab15433b3f6775d8f204fc9a469e1e8b9989124dd8aea46ea60e9aa6429ec`。使用已有环境和空闲卡准入程序。

## 程序与证据

| 工作 | 程序 | 结果 |
|---|---|---|
| 冲突源码的逐项比较 | `compare_conflicts.py`、`compare_merged.py` | `conflict_comparison.json`、`merged_source_comparison.json` |
| Linear 调优缓存的精度区分 | `compare_linear_v2.py` | `linear_precision_comparison.json` |
| 后端索引反向覆盖检查 | `check_index_override.py` | `index_override_comparison.json` |
| 实际注册的前向和梯度精度 | `check_precision_routes_v2.py` | `precision_routes_existing_control.json`、`precision_routes_restored.json` |
| 非连续与空维度 Linear | 仓库 `tests/test_linear.py`，`queue_restored_validation.py` | `targeted_regression_results.json` |
| 最终包精选回归 | `run_validation_suite_restored.py` | `validation_restored_results.json` |
| 两步整网训练与现有 Linear 组合对照 | `queue_training_restored.py`、`queue_training_native_linear.py` | 相应训练比较 JSON |
| 完整梯度、实际更新及 step0 Adam 状态 | `analyze_upstream_merge_training_restored.py`、`analyze_upstream_merge_training_native_linear.py` | 相应训练比较 JSON |

`variants/` 保存固定的冲突实现。`merged_*_all_source_hashes.json`、部署回执和训练入口哈希记录每个阶段的源码；各阶段不可覆盖。`retained_training_operator_provenance.json` 记录当前项目的矩阵乘法、融合 Softmax 及辅助程序来源。

脚本使用原服务器的已有模型、输入及统一随机数程序。原始张量和 checkpoint 保留在计算服务器；本目录保存源码、标量统计、哈希及运行回执。首次失败、编译中止、上传失败和后续修复分别保留。复现时选用新的输出目录，保留已执行程序及旧结果。

`*_tf32` 的早期冲突源码用例使用旧快速开关；实际 RNE 入口的数值、梯度及计时使用 `precision_routes_restored.json`。具体环境开关随每个用例记录在 CSV 和 JSON 中。

最终精选回归为 21 个文件、161 通过、25 跳过；定向回归 26 项，实际自动求导 22 项，精度入口 10 项。整网数据和数值变化见报告的完整四路表格。发布源码与实测源码仅一段注释不同，可执行 AST 相同，记录在 `publication_source_manifest.json` 与 `policy_comment_provenance.json`。
