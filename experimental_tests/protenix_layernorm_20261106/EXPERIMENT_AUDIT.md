# Experiment Audit Report

**Date**: 2026-11-06  ·  **Auditor**: GPT-6-Astra ultra（fresh same-family reviewer）
**Review independence**: `same-family`  ·  **Acceptance status**: `provisional`
**Overall verdict**: **WARN**  ·  **Integrity status**: **warn**

最终报告的算子和训练数值均与保存标量匹配；没有发现预测值自归一化、未披露的伪真值或把失败结果改写为成功。算子收益有明确限定。**训练候选未通过采用条件：FP32 与 TF32 的完整梯度均更远离历史同精度 H100，虽然实际更新距离减小。** 当前报告已正确拒绝采用。

本次仅执行本地源码、标量与回执审计；原始张量、数据资产和远端二进制没有重新读取，没有运行模型或 GPU 内核。数值复算可重复；语义与执行归因维持 same-family/provisional。训练来源分工由只读子审计协助，最终标量与门槛由主审计独立复算。

## Checks

### A. 参考与训练目标来源: PASS

CPU64 解析式是固定实际操作数的数值参考；保存统计量和重算统计量分开。H100 是历史实现输出对照，未冒充结构真值。训练监督来自原实验结构资产，未改为预测标签。原始资产/张量未在本次本地审计重新读取。

证据：[analyze_ln_abi_20261106.py:38](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/analyze_ln_abi_20261106.py:38)；[analyze_ln_abi_20261106.py:73](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/analyze_ln_abi_20261106.py:73)；[operator_replay_helpers_20261031AB.py:43](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/operator_replay_helpers_20261031AB.py:43)；[training_experimental.py:225](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/training_experimental.py:225)；[training_experimental.py:327](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/training_experimental.py:327)；[analyze_layernorm_training_20261106.py:27](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/analyze_layernorm_training_20261106.py:27)。

### B. 归一化、单位与汇总: PASS

误差分母为对应参考平方范数。算子相对误差存百分比；训练 relative_l2 存比例，报告乘 100。正确池化平方和；零范数保留 null，近零偏置保留绝对误差。64 个七样本计时中位数、32 个 mean-of-two-medians 汇总、664 个算子及 448 个训练标量对象、64 行 CSV 和报告 82 个数值单元全部复算匹配。

证据：[operator_replay_helpers_20261031AB.py:59](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/operator_replay_helpers_20261031AB.py:59)；[summarize_results.py:10](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/summarize_results.py:10)；[summarize_results.py:43](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/summarize_results.py:43)；[analyze_linear_forward_training_20261023.py:24](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/analyze_linear_forward_training_20261023.py:24)；[REPORT.md:17](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:17)。

### C. 结果存在、失败保留与声明对应: PASS

最终两精度捕获、v2/v3 benchmark、真实注册检查和两份训练分析均有完成记录。源码/候选哈希匹配；历史编译与注册失败未改写成成功。最终报告和 SUMMARY 明确未通过训练采用条件。完整原始执行独立性仍受本地 scalar/hash 回执范围限制。

证据：[capture_v2_complete.json:3](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/capture_v2_complete.json:3)；[benchmark_v3_complete.json:2](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_v3_complete.json:2)；[registered_v3_status.json:21](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/registered_v3_status.json:21)；[training_final_fp32_analysis.json:2](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/training_final_fp32_analysis.json:2)；[training_final_tf32_analysis.json:2](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/training_final_tf32_analysis.json:2)；[SUMMARY.json:132](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/SUMMARY.json:132)；[REPORT.md:82](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:82)。

### D. 实际调用、注册路径与静态安全: PASS

参考/度量函数确实在 benchmark 中调用；注册检查使用 autograd 并观察到候选 helper。每精度训练 FlagGems 路由、ATen 入口和 fused helper 返回计数按 M/N 完全一致（1294 次、7 组），另外 480 次保留原接口。旧维度错误由静态索引证明，没有运行可能越界的旧三维输入。training_routes PASS 只代表跳过 profiler 后的告警检查，不能扩大成全算子路由审计。

证据：[benchmark_layernorm_v3_20261106.py:28](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_layernorm_v3_20261106.py:28)；[benchmark_layernorm_v3_20261106.py:98](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_layernorm_v3_20261106.py:98)；[check_registered_layernorm_v3_20261106.py:25](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/check_registered_layernorm_v3_20261106.py:25)；[layernorm_training_route_20261106.py:23](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/layernorm_training_route_20261106.py:23)；[training_final_fp32_analysis.json:267](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/training_final_fp32_analysis.json:267)；[layernorm_baseline.py:416](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/layernorm_baseline.py:416)；[train_route_entry.py:23](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/train_route_entry.py:23)。

### E. 范围与采用边界: WARN

128 记录仅为 112 个不同 shape/layout/affine/mask 组合，含 40 次全关闭输出；8 个形状/布局、4 种仿射组合，不是 128 个独立数据。16 份实际 ABI 均为 both-affine，8 份非零、8 份全零；更广的训练路由没有逐组性能测量。仅一个模型/配置/种子、每精度两步且一次正学习率更新。两种精度完整梯度对 H100 均退步，因此拒绝默认训练采用。没有新 H100、长训练、质量或整网吞吐证据。

证据：[benchmark_layernorm_v3_20261106.py:33](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_layernorm_v3_20261106.py:33)；[benchmark_layernorm_v3_20261106.py:48](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_layernorm_v3_20261106.py:48)；[benchmark_v3_complete.json:4697](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_v3_complete.json:4697)；[linear_x3_environment_20261023.sh:43](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/linear_x3_environment_20261023.sh:43)；[training_experimental.py:196](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/training_experimental.py:196)；[PLAN.md:20](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/PLAN.md:20)；[REPORT.md:71](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:71)。

### F. 评价类型分类: PASS

算子公式检查、固定输入 ABI 数值一致性和 H100 梯度/更新一致性为 synthetic_proxy（数值/实现参考，非生物学预测真值）；原数据监督 loss 执行为 real_gt（回执限定），不是独立质量 benchmark。GPU 事件计时为经验性能测量，不属于 GT 分类。

证据：[REPORT.md:42](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:42)；[REPORT.md:84](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:84)；[analyze_layernorm_training_20261106.py:54](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/analyze_layernorm_training_20261106.py:54)；[training_experimental.py:327](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/training_experimental.py:327)。

## 数学与参考来源核查

保存统计量参考使用捕获的 dY、X、mean/rstd 和仿射参数，以 CPU FP64 执行标准 LayerNorm VJP 公式；重算统计量参考另列。它回答固定 PPU 操作数上的数值问题，不是 H100 内部真值，也不是一次全网 FP64 训练。[analyze_ln_abi_20261106.py:38](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/analyze_ln_abi_20261106.py:38)、[operator_replay_helpers_20261031AB.py:43](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/operator_replay_helpers_20261031AB.py:43)。

12 组小例覆盖三种归一化形状与四种仿射配置，解析式与 CPU64 autograd 比较的保存最大绝对差为 `5.684341886080802e-14`，低于代码的 `2e-12` 门槛。审计读取自检与调用代码并复算保存标量，没有宣称自行重跑 autograd。[analyze_ln_abi_20261106.py:49](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/analyze_ln_abi_20261106.py:49)、[benchmark_v3_complete.json:9](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_v3_complete.json:9)。

训练监督沿用冻结的 CIF/bioassembly 结构资产和原 native loss。候选实际 batch 数值审计及资产哈希与本地保存的 H100 report 一致；与旧 PPU 的完整一致性由分析器断言与旧历史回执共同支持。该数据仅支持本次微调/短程一致性检查；不能证明训练前预训练数据无重叠。[training_experimental.py:260](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/training_experimental.py:260)、[training_experimental.py:327](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/training_experimental.py:327)、[training_report.json:65](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/experiment_integrity_aten_backend_training_20261027K/extracted_references/fp32/training_report.json:65)。

## 独立标量复算

| 检查 | 数量 | 结果 |
|---|---:|---|
| 算子平方和/相对误差/绝对范数对象 | 664 | PASS |
| 完整训练、分模块、top 参数和 Adam 状态对象 | 448 | PASS |
| 七个原始 GPU-event 样本的中位数 | 64 | PASS |
| 两步相应调用中位数的算术平均 | 32 | PASS |
| operator_cases.csv 完整行 | 64 | PASS |
| REPORT.md 数值单元 | 82 | PASS |

误差聚合为 `100 * sqrt(sum(difference_squared) / sum(reference_squared))`；没有平均每个参数的相对误差，也没有用候选范数作为分母。训练文件 `relative_l2` 是比例，报告乘 100 后为百分比。训练 helper 为零参考使用 `1e-300` 数值保护；本次完整训练比较的参考范数均为正，故该保护不影响门槛。[summarize_results.py:10](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/summarize_results.py:10)、[analyze_linear_forward_training_20261023.py:24](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/analyze_linear_forward_training_20261023.py:24)。

8 份全零 ABI 的全部实现和两种参考均保留 `difference_squared=reference_squared=0` 与 `relative_L2_percent=null`。非零大输入偏置梯度的池化参考范数为 FP32 `1.4585887205801459e-08`、TF32 `1.2170278622236233e-05`。FP32 C++/融合版偏置相对误差为 `85.5154159663% / 32.0619677514%`，绝对差范数仅 `1.2473182116e-08 / 4.6765224522e-09`；不能把高百分比单独解读为大规模失效。完整表保留了旧 FlagGems `333.871497854%`，无删选高误差。[REPORT.md:54](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:54)、[REPORT.md:67](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:67)。

最终单位修正已复核：偏置绝对 L2 从百分比误差表移入独立绝对范数表，新增同组 CPU64 参考 L2。两种精度各两个非零 256 通道调用的参考平方和开方为 FP32 `1.4585887205801459e-08`、TF32 `1.2170278622236233e-05`；四种实现的参考范数完全一致。新增两格及受影响的八格绝对差、八格相对误差共 18 格均匹配保存标量。发布目录的 REPORT 与当前报告字节一致。此次只调整展示单位和补充参考尺度，原始结果及拒绝采用结论未变。[REPORT.md:59](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:59)、[生成器:59](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/summarize_results.py:59)。

FP32 输入梯度误差由 C++ 的 `7.6548468676e-06%` 变为融合版 `7.6634462801e-06%`，略增；所以“全部梯度数学精度同时改善”不受支持。参数梯度改善和吞吐测量可以分别报告。[REPORT.md:46](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:46)、[REPORT.md:49](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:49)。

## 代码路径、静态安全与覆盖

旧 baseline 对 `[2,3,7]` / normalized `[7]` 取 `M=2,N=21`；仿射数组实际长度为 7，但 weight load 与 dW/dB store 的上界按 N 判断，会允许索引 7…20。修正为 `N=prod(normalized_shape), M=numel/N`，空输入提前返回。这是静态可复核错误，审计没有运行危险旧核；性能 baseline 明确为 axes-fixed 版本。[layernorm_baseline.py:416](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/layernorm_baseline.py:416)、[layernorm_baseline.py:265](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/layernorm_baseline.py:265)、[layernorm_baseline.py:325](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/layernorm_baseline.py:325)、[layernorm_candidate_v3.py:418](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/layernorm_candidate_v3.py:418)。

新 v3 与原文件相同的 forward 和旧内核 AST 未发生变化，增加的融合/分段求和路径使用行数、归一化宽度、FP32、仿射 mask 和资源上限。融合触发范围为 M≥4096、N≤2048，分块最多 256，部分和缓冲上限 64 MiB，tile 元素数≤2048；没有模型名、层名、精确 shape 列表或 dY 内容筛选。实际 capture 的零/非零分层仅影响诊断取样，不控制候选计算。[layernorm_candidate_v3.py:499](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/layernorm_candidate_v3.py:499)、[layernorm_candidate_v3.py:559](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/layernorm_candidate_v3.py:559)、[ln_abi_observer_v2_20261106.py:38](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/ln_abi_observer_v2_20261106.py:38)。

128 generic 记录是 112 个不同的 shape/layout/affine/mask 元组；40 条 mask 全关闭，88 条产生至少一个输出，共 128 个非空梯度误差向量。它没有覆盖每个 affine 组合的所有合法梯度 mask，更没有覆盖所有 dtype/architecture。16 个实际算子包均为 both-affine，8 个非零；真实训练额外执行了 weight-only 分支，但本轮没有为全部训练 M/N 逐组采集性能。[benchmark_layernorm_v3_20261106.py:33](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_layernorm_v3_20261106.py:33)、[benchmark_layernorm_v3_20261106.py:48](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_layernorm_v3_20261106.py:48)、[training_final_fp32_analysis.json:267](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/training_final_fp32_analysis.json:267)。

注册检查的六个正尺寸案例中，三组大 M 的 helper 返回融合结果；两组空输入与 CPU native autograd 对照。每种训练精度中，FlagGems adapter、实际 ATen 入口与 fused-helper-return 按七个 M/N 组合精确对齐，均为 1294 次；480 次继续用原接口。该证据支持实际 LN 路径执行。`training_routes.json` 的 PASS 在 `--skip-route-audit` 下仅为告警检查，不能当作完整模型 profiler 证据。[registered_v3_status.json:229](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/registered_v3_status.json:229)、[check_registered_layernorm_v3_20261106.py:42](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/check_registered_layernorm_v3_20261106.py:42)、[training_final_fp32_analysis.json:388](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/training_final_fp32_analysis.json:388)、[train_route_entry.py:23](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/train_route_entry.py:23)。

## 训练与采用门槛

| 精度 | 指标（对旧同精度 H100，相对 L2 %） | 原路径 | 新路径 | 不退步 |
|---|---|---:|---:|---|
| FP32 | gradients | 0.00154035996337 | 0.00172964112247 | FAIL |
| FP32 | updates | 0.0429411421634 | 0.042364646676 | PASS |
| TF32 | gradients | 1.28978806256 | 1.30600537731 | FAIL |
| TF32 | updates | 6.43391728423 | 6.36952252264 | PASS |

训练门槛是两种精度的完整 gradients 和 updates **四项均不增加**；两项 gradients 失败，因此 `training_promotion_passed=false` 正确。这里的 FAIL 是候选采用门槛失败，不是审计发现欺诈。报告保留更新改善和梯度退步，没有调低门槛或用 loss 相同替代梯度验收。[PLAN.md:20](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/PLAN.md:20)、[summarize_results.py:82](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/summarize_results.py:82)、[REPORT.md:82](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:82)。

比较对象是第 1 步完整 preclip 梯度（4170 张量、464441841 元素）和实际参数更新（4174 张量、464442431 元素）；不是两步各自都有 H100 全向量。第 0 步 LR=0 仍更新 Adam；其一、二阶矩另外与旧 PPU 比较，不能当作 H100 step0 Adam 验证。六份候选/旧 PPU/H100 step1 LR 回执均为 `1.8e-5`。两种精度新旧 PPU 的 loss 全部分项相同，但梯度与 Adam 状态有差异。[train_entry_nonzero.py:85](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/train_entry_nonzero.py:85)、[analyze_layernorm_training_20261106.py:44](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/analyze_layernorm_training_20261106.py:44)、[analyze_layernorm_training_20261106.py:59](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/analyze_layernorm_training_20261106.py:59)、[training_final_receipts.json:6](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/training_final_receipts.json:6)。

数据、权重、RNG 检查包括两个完整 batch audit、资产 hashes、4174 个初始参数 hashes 和 3338 项有序随机记录（semantic SHA-256 `2fa07a2f666b52a6bc66821ddb18235795492a7d40ac4708533bac59a7107521`）。候选和既有 PPU 的八项 common-geometry/既有算子 hashes 相同；H100 snapshot hashes 与已有 manifest 匹配。训练 adapter 显式声明 backward replacement，原 forward/loss/优化器路径保留。[analyze_layernorm_training_20261106.py:30](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/analyze_layernorm_training_20261106.py:30)、[layernorm_training_route_20261106.py:64](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/layernorm_training_route_20261106.py:64)、[train_softmax_forward_fidelity_20261023.py:24](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/train_softmax_forward_fidelity_20261023.py:24)。

报告新增的 dX 逐字节保持路线属于后续设计约束：本轮只确认完整梯度退步与若干局部数值变化同时存在，没有单独消融证明 dX 舍入是整网退步的唯一原因，也未证明 dX 一致就足以通过训练验收。[REPORT.md:86](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/REPORT.md:86)。

本轮单模型 `protenix-v2`、种子 `20260825`、每精度两步，只有一次正 LR 更新。沿用单 recycle、单 diffusion sample/step，关闭蛋白/RNA MSA 和模板。没有候选独立重复、长程收敛、九模型/多输入/结构质量或新 H100 同操作数验证。训练 execution 记录 `timing_valid=false`，所以不能从这两次含编译运行推出正式端到端速度或 MFU。[linear_x3_environment_20261023.sh:43](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/linear_x3_environment_20261023.sh:43)、[training_experimental.py:196](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/training_experimental.py:196)、[train_softmax_forward_fidelity_20261023.py:110](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/train_softmax_forward_fidelity_20261023.py:110)。

## 来源哈希、负结果与剩余义务

候选 v3 hash 为 `9ef996d0835e8d53d623cfdf117a8a4f6bd071a26ea025406ecb6eb46d18742f`，adapter 为 `3e1d6c9e70054153bf3c2a65bd3f7fa375d928c4294e1a18497607c488be2e47`，benchmark、注册、训练三阶段一致。原接口回执中的 `.py` 文件是控制 ABI loader，hash `c6aba5e0de3553b22550653c4036e659acaf7ba05e64125b01ddc843dabec619` 与本地保存 loader 相同；该 loader 对真正 C++ extension 及其来源实施 hash 校验，回执中这个 `.py` hash 不应误称为 `.so` hash。[training_final_fp32_analysis.json:469](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/training_final_fp32_analysis.json:469)、[fast_layer_norm_cuda_v2.py:7](/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/checkpoint_current_gems_20261018_224r2/norm_abi/fast_layer_norm_cuda_v2.py:7)。

首个 candidate 的未定义 `ext` 编译失败和首个注册检查的 include/调用计数失败仍保留。v2 仅修正三个 program-id 别名；v3 使用新目录。数值容差仍为 `2e-5*reference_L2+2e-10`，公式校验仍为 `2e-12`。v1 capture 全零大形状保留；v2 按每个 step/norm/affine/zero-status 选择首次真实调用，不替换上游。[benchmark_v1_status.json:2](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_v1_status.json:2)、[layernorm_v2_correction.json:2](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/layernorm_v2_correction.json:2)、[benchmark_layernorm_v3_20261106.py:55](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/benchmark_layernorm_v3_20261106.py:55)、[nonzero_capture_correction.json:2](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/nonzero_capture_correction.json:2)。

审计期间已补齐最终 v2 capture 回执并澄清计时平均口径，问题已解决。最终 receipts 中的三个分析 helper hashes 与本地代码一致，但属于**运行后**验证；mtime 或当前 hash 不能严格证明执行起点代码不可变。原始 tensor/dataset/extension 未由本审计重新认证，应持续使用回执限定的执行归因。[training_final_receipts.json:3094](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/training_final_receipts.json:3094)。

继续使用本结果时必须保留：未通过训练采用门槛、精度参考类型、零/近零梯度绝对误差、有限形状和单配置两步范围。任何更广架构或端到端收益需要独立证据。本轮审计不要求为拒绝的候选另启实验。

## 核心发布补丁补充审计：PASS

拟发布 `FlagGems/src/flag_gems/ops/layernorm.py` 与 `layernorm_core_axes_fixed.py` 字节完全相同，SHA-256 为 `c78e555ea8caf95a3b0d23c2b457d6ce64e6bc73cd0486df0dbdf4fb33284c68`。与已测 axes-fixed 源码比较，仅有许可证头及 `tle`/`ext` 等价别名差别，归一化后的 AST 完全相同。相对发布分支基线的 diff 只改变 M/N 与空输入返回；没有融合、分段求和或训练 adapter。因此本补丁是语义修正，**实验 v3 的加速数字不属于这个核心补丁**。

证据：[publication layernorm.py:431](/home/lgx1429541889/publication-20261105/FlagGems/src/flag_gems/ops/layernorm.py:431)、[核心检查结果:16](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/core_check_status.json:16)。

新增 `test_layer_norm_backward_normalized_suffix` 的 function-body AST 与独立执行脚本相同；CPU native autograd 提供输出梯度及保存的 mean/rstd，待测入口为注册 ATen backward。五个案例覆盖三个正尺寸后缀/多维/一维输入及两个空输入；仅 FP32、both-affine、三个输出全部开启。回执退出码为 0，实际加载源 hash 与发布源码一致。容差为 `rtol=1e-4, atol=1e-5`，这是新增核心回归检查自身的容差，不改变之前算子或训练门槛。没有执行 pytest decorator collection 或完整测试套件；不得将该结果标为完整 pytest 通过。

证据：[新增测试:162](/home/lgx1429541889/publication-20261105/FlagGems/tests/test_layer_norm.py:162)、[精确测试体执行脚本:5](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/check_core_layernorm_axes_20261106.py:5)、[核心结果范围:69](/home/lgx1429541889/Protenix/comparison/training_gradient_20261106/core_check_status.json:69)。

## Claim impact

| Claim | 结论 | 限定 |
|---|---|---|
| C1：归一化外部行划分错误及空输入修正 | supported | 静态证明旧代码错误；修正版通过已列出的 synthetic/native-autograd 检查，不是原 C++ 模型路径故障。 |
| C2：已测大输入完整 LayerNorm 反向更快 | supported_with_qualifier | PPU 同卡 GPU 事件；四组、两个步的各七样本中位数的均值；小输入仍较慢；不推出所有训练 shape 或整网收益。 |
| C3：数学精度改善 | needs_qualifier | 非零大输入参数梯度相对 C++ 改善，但 FP32 dX 略退步；近零 dBeta 必须带绝对范数，零向量不计改善。 |
| C4：实际注册及训练融合路径被执行 | supported | 每精度 1294 次，7 个 M/N 组合；source/adapter hashes 与 tested v3 一致；不是全模型 profiler 审计。 |
| C5：通过两步训练的默认路径提名条件 | unsupported_rejected | FP32/TF32 gradients 均增加对 H100 的相对 L2；updates 改善不能覆盖 conjunctive gate 失败。最终报告已披露拒绝。 |
| C6：H100 同操作数性能、长训练收敛、结构质量或整网速度/MFU改善 | not_evaluated | 本轮无相应证据，最终报告没有提出这些收益。 |
| C7：核心发布后缀维度/空输入修正 | supported | 五个精确 regression function-body 案例通过；未运行 pytest collection/full suite，融合性能不属于核心补丁。 |

全部 93 份已审阅输入的 SHA-256 保存在 `EXPERIMENT_AUDIT.json`。审计仅写入本 Markdown 与 JSON；reviewer 调用/返回 trace 由父执行者保存于 `audit_trace/`；本次已确认 `REQUEST.md` 和 `RESPONSE.md` 存在，未重新审计 trace 正文。
