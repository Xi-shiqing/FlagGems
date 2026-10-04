# Protenix PPU / FlagGems 实验索引

本索引归档截至 2026-10-04 的 TF32 实现、失败对照、精度诊断和验证数据。最新结果从 [host pipeline](protenix_ppu_host_pipeline_20261004/README.md) 开始查看。各阶段的目录和 manifest 对应当时隔离的服务器环境，不能按目录日期依次覆盖安装。

| 记录 | 内容与用途 |
| --- | --- |
| [2026-09-20 部署快照](protenix_ppu_20260920/README.md) | 保留严格 FP32 MM 与 mean 修复成果及旧工作区补丁 |
| [TF32 RNE](protenix_ppu_tf32_rne_20260923/README.md) | MM/addmm/BMM 与 TriAttention 真实入口的普通 TF32 舍入修正 |
| [native RNE](protenix_ppu_tf32_native_rne_20260926/README.md) | 公共舍入辅助函数使用 native 转换，保留 NaN/Inf 语义 |
| [fill launch](protenix_ppu_fill_launch_20260926/README.md) | 连续 FP32 fill 的分派和启动开销实验 |
| [BMM prepare](protenix_ppu_bmm_prepare_20260926/README.md) | 分配、布局、内联舍入与启动缓存的独立对照 |
| [MM index32](protenix_ppu_mm_index32_20260930/README.md) | 含 Split-K padding 的安全地址位宽检查及验证 |
| [matched TF32](protenix_ppu_tf32_matched_20261001/complete_comparison.json) | 配对四模式测量和 profiler 归因，保存原精度状态 |
| [Jittor reference](protenix_ppu_tf32_jittor_reference_20261001/final_summary.json) | 参考实现分析、pipeline 候选和被拒绝方案 |
| [rank one](protenix_ppu_rank_one_20261002/manifest.json) | K=1 的 IEEE 外积、out/布局、训练探针与模型对照 |
| [precision route](protenix_ppu_tf32_precision_route_20261002/complete_comparison.json) | MM/BMM/TriAttention 的实际精度路径审计 |
| [GEMV / QK](protenix_ppu_gemv_qk_20261002/manifest.json) | N=1 列 MM 候选；保留未通过的行 MV/BMM 政策 |
| [query round once](protenix_ppu_query_round_once_20261002/complete_comparison.json) | TriAttention Q 舍入复用与真实模型对照 |
| [module boundary](protenix_ppu_module_boundary_20261002/complete_comparison.json) | 模块精度消融及粒度更细的误差定位 |
| [backward dataflow](protenix_ppu_backward_dataflow_20261003/complete_comparison.json) | backward 路由、梯度/更新误差及历史审计证据 |
| [closure](protenix_ppu_closure_20261003/summary.json) | 九模型矩阵、训练确认、选用实现与未通过项 |
| [native reference](protenix_ppu_native_reference_e2e_20261004/report.md) | 冻结 H100 原生基线；自定义 triangle 与 IEEE 消融单独标记 |
| [confidence precision](protenix_ppu_confidence_precision_20261004/summary.json) | 五项置信度、ConfidenceHead 消融和通用几何候选 |
| [host pipeline](protenix_ppu_host_pipeline_20261004/README.md) | 完整写入输出减少预填充、保持布局的 RHS 单次 RNE；最新四基线 + 候选表 |

性能比较采用 PPU+FlagGems FP32/TF32 与 H100 原生 FP32/TF32；额外候选和混合精度诊断另列，不替代基线。报告中的模型、输入、循环/采样步数和重复次数决定结论适用范围。坐标 RMSD 的区间和未通过状态按记录保留，不能把 UNRESOLVED 计为 PASS。

`src/flag_gems/ops/` 中保留 K=1 外积及 N=1 列 MM 的较新通用源码集成；最新 host-pipeline 改动仍在实验目录中。服务器的实际部署源码以各阶段快照和 hash 为准，较新通用源码不能直接继承这些性能数字。

归档内容为源码、补丁、标量测量及 hash。原始输入张量、激活、坐标、模型权重和完整 trace 留在原服务器。旧 manifest 中的提交状态描述的是实验完成时的状态。
