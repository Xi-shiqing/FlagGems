# PPU FlagGems 通用推理优化（2026-10-04）

本轮保留四个基线：**H100 原生 FP32、H100 原生 TF32、PPU+FlagGems FP32、PPU+FlagGems 原 TF32**，另列 PPU TF32 实验候选。耗时是一次完整 `InferenceRunner.predict` 的重复运行中位数，单位秒。

## 实现与启用范围

1. `full_write_allocation.py`、`build_candidate.py`：在隔离的 FlagGems 包里修改八处已验证会写满的输出分配，覆盖 pointwise、softmax、sigmoid、TriAttention 的 forward 输出和 cat。候选推理期间省去确定性输出预填充；保持 dense transpose 的物理 stride。首个完整模型 warmup 使用 NaN/sentinel 污染输出，检验算子是否真正写满。梯度开启时回退，确定性全局开关保持原值。
2. `mm_preserve_layout.py`：足够大的 MM 将 RHS 按原物理布局做一次既有 RNE 舍入，供所有 M tile 复用。继续调用原 `mm_direct_fused`，保持 64×128×32 tile、4 warps、stage=1、原 Split-K、地址位宽规则及累加顺序。没有跨调用权重缓存。

RHS 准备只在 `ceil(M/64) >= 2048`、dense RHS 不超过 64 MiB、普通 TF32 推理且原直接 MM 路径适用时启用；小 M 在读取 precision/stride 前回退。规则对所有模型和输入相同，已验证覆盖仅限本轮记录的输入和真实激活。

`matrix_entry.py` 只在候选 prediction 的作用域内启用上述改动，并在退出时恢复。FP32/原 TF32 对照直接使用 Torch 分配别名，避免辅助分配函数增加基线的 Python 分派成本。**本目录是实验候选，没有把本轮分配和 RHS 准备优化替换为仓库默认实现。**

## 结果与精度

完整表格和审计记录：

- [timings.csv](results_matrix/timings.csv)：九模型 × 两输入的四基线与候选耗时、收益、配对 bootstrap 区间和实际采样来源。
- [confidence.csv](results_matrix/confidence.csv)：18 个组合 × 五项置信度，四基线、候选及差异，共 90 行。
- [accuracy.csv](results_matrix/accuracy.csv)：跨设备 RMSD 区间、正式精度状态及候选相对原 PPU TF32 的差异。
- [summary.json](results_matrix/summary.json)、[route_validation.json](route_validation.json)：汇总、完整输入/RNG 审计和真实路由证据。
- [method.json](method.json)、[profile_evidence.json](profile_evidence.json)：实现原理和 profiler 诊断；profiler 耗时不作为验收耗时。

693-token 的 G01：九个模型耗时降低 4.94%–5.70%，九个配对 95% 区间均为正。245-token 的 G02：九个中位数降低 0.87%–3.20%，只有一个区间为正，多数短输入收益尚不能与测量波动区分。18 个组合的耗时下降几何平均为 3.57%。

新旧 PPU TF32 的坐标和五项置信度逐位一致；候选相对原 PPU TF32 的坐标 RMSD 为 0。完整 feature、atom 顺序、checkpoint、外部资产、规范输入和有序 RNG 审计通过。

正式跨设备精度状态保持原样：FP32 18 PASS；TF32 3 PASS、11 FAIL、4 UNRESOLVED。这里的 RMSD 区间来自之前冻结的正式测量，只有在四基线完整坐标/置信度 hash 一致后才沿用；不是本轮重新传输坐标进行比较。H100 自定义三角注意力诊断不替代原生 H100 基线。

## 测量协议与复现

原服务器物理 GPU 7 串行运行；seed=101、统一随机数程序、dtype=FP32、cycle=1、diffusion step=1、sample=1。TF32 模式改变计算许可。初始化、编译、审计插桩、污染 warmup 和保存输出均排除在干净重复计时之外；MM 准备和分配成本包括在候选计时内。

PPU 初测每模式 1 warmup/5 repeats；G02 三个模型复测为 3/15。H100 最终全部 18 个组合使用 3/7。原始初测保存在 `derived_matrix.json`，复测分别保存于 `derived_recheck.json`、`derived_steady.json`。报告先校验所有非耗时结果一致，再采用预定复测记录；不挑选更快的一次。

本地只依赖 Python 标准库，可重新生成正式表格及运行 profiler 区间自检：

```bash
python3 experimental_tests/protenix_ppu_host_pipeline_20261004/report_results.py --stage matrix --expected 18
python3 experimental_tests/protenix_ppu_host_pipeline_20261004/summarize_trace.py --check
```

归档版 `report_results.py` 额外接受本仓库带 `protenix_ppu_` 前缀的正式参考目录；数字计算和服务器执行代码保持原样。`artifact_manifest.json` 固定本目录源码、复现脚本、标量和 hash 证据；`execution_sources/` 保存各设备实际执行的入口版本。

设备复现继续使用原服务器环境和对应 `run_*_ppu.sh`/`run_*_h100.sh`，不能把本目录覆盖安装到任意 FlagGems 版本。`baseline_reference/` 是构建候选所涉及的原部署文件，不是完整可独立安装的包。`build_candidate.py --extended` 应指向记录的完整基线包；`verify_full_write.py` 和 `verify_mm_layout.py` 留有设备检查入口。

验证包括 28 个完整写入算子案例、13 组真实 MM 激活、4 个合成 MM 控制和 18 个完整模型组合。原始张量、坐标、权重和完整 profiler trace 均留在原服务器；这里只归档代码、标量和 hash。

尚未完成默认多步 diffusion 或九模型训练重新验收。H100/PPU 耗时比不是 MFU；本轮也不代表达到 80% 目标。
