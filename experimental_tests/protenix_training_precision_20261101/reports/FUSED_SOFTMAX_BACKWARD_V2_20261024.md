# Softmax 反向融合：保持数值结果的 FlagGems 优化（2026-10-24）

将原先单独计算并存储的 g×p 乘积合入 Softmax 反向归约内核，消去一次逐元素乘法调用和同大小临时张量。保留原乘积 FP32 舍入、归约分组及最终算术；矩阵乘精度和全部前向不变。对 PPU 普通 FP32 反向统一生效，FP16/BF16 和记录高阶导的情况保留原路径。没有模型、输入、精确 shape、训练步或梯度数值白名单。

独立私有 FlagGems 包 1371 个 Python 文件中仅 softmax.py 改动；实际注册 ATen/autograd 入口通过，默认包未替换。该算子本身使用 FP32；下表的 FP32/TF32 指模型矩阵乘标志，不能把本项收益说成 TF32 乘法加速。

## 已通过的数值检查

同一份捕获的 MSA 激活在两个矩阵乘精度标志下的完整 VJP、33 个 dtype/轴/布局检查（11个FP32融合、22个FP16/BF16原路径回退）、20 个极小梯度与正负零检查，均与原未融合固定规则的字节表示相同。三种 dtype 的 CPU64 数学参考容差也通过。真实输入 600×600×8，共 2,880,000 个输出梯度元素；直接调用反向函数时 Python torch.mul 由每次 1 个变成 0 个。注册autograd候选归因以包入口设置和源SHA为依据，原检查没有单独记录已注册callable，后续补充证明另列。

首版使用 libdevice.mul_rn，常规检查通过但极小梯度失败，失败原样保留。独立乘法诊断确认该运行时的 libdevice RN 路线会将极小乘积变零，普通乘法会保留；编译 IR 含 nvvm-reflect-ftz=1 及 FTZ intrinsic。v2 使用不带 FTZ 修饰、阻止乘加合并的显式 mul.rn.f32，通过完整复检。这是新融合候选自身的缺陷修复，不能据此解释原有跨 H100 的全部训练误差。

## 完整反向算子调用耗时

以下单位均为微秒/次。固定同一份输入和概率，不计前向；计入梯度 contiguous、乘积物化（原版）、结果分配及内核启动。每配置预热后七个交替次序的配对轮次，每轮十次；表为各轮均值的中位数。设备事件区间也可能含主机提交间隙，不能解释为纯内核执行时间。墙钟包含最后同步；原始七轮全部保留，包括离群值。

|平台／模型矩阵精度标志|输入／归约轴|未融合：事件 / 墙钟|融合：事件 / 墙钟|完整API事件加速比|
|---|---|---:|---:|---:|
|H100 原生 FP32|同一完整API口径|无可用旧记录|未执行|不可计算|
|H100 原生 TF32|同一完整API口径|无可用旧记录|未执行|不可计算|
|PPU FlagGems FP32|[600, 600, 8] / -2|194.752 / 196.229|121.740 / 123.252|1.600×|
|PPU FlagGems TF32|[600, 600, 8] / -2|198.376 / 199.970|122.468 / 123.986|1.620×|
|PPU FlagGems FP32|[12416, 8] / -1|117.472 / 118.858|44.664 / 46.112|2.630×|
|PPU FlagGems TF32|[12416, 8] / -1|115.064 / 116.429|42.184 / 43.579|2.728×|
|PPU FlagGems FP32|[257, 129] / -1|113.776 / 116.164|43.612 / 45.029|2.609×|
|PPU FlagGems TF32|[257, 129] / -1|112.824 / 114.197|40.312 / 41.798|2.799×|
|PPU FlagGems FP32|[3, 2049] / -1|114.744 / 116.122|42.184 / 43.865|2.720×|
|PPU FlagGems TF32|[3, 2049] / -1|113.264 / 114.707|42.140 / 43.548|2.688×|

实际卡为 PPU-ZW810E GPU0，UUID GPU-019e1215-4190-0010-0000-0000e085f370；入口检查空闲卡并持有合作式独占锁。原wrapper的timing_valid=false保留，admission同时记录timing_allowed=true；目前只报告局部原始API计时，不将该字段补写为正式性能验收通过，也未持续证明整个测量期间无外部任务干扰。该结果没有新的 H100 同口径测量、整模型加速比或 MFU 验收。

## 真实训练验证

FP32、TF32 各两步真实训练均 exit0，完整 CPU 审核 exit0。对照是本轮已完成的未融合私有包训练，未重跑对照。全部实际 batch、资产、4174 初始参数、3338 条统一 RNG 和 LR0/1.8e-5 通过。第0步完整梯度、实际 Adam 两阶矩、第1步完整梯度和实际参数更新，除数值比较外还逐张量比较字节表示，全部相同。所有保存 loss 分项也相同。

|平台／训练精度／版本|第0步loss|第1步loss|对同精度旧H100梯度相对L2差%|对同精度旧H100实际更新相对L2差%|
|---|---:|---:|---:|---:|
|H100原生 FP32（旧参考）|1.879856706|4.581809044|参考|参考|
|PPU FlagGems FP32 当前|1.879857898|4.581814766|0.001540359963|0.042941142163|
|PPU FlagGems FP32 固定未融合|1.879857898|4.581814766|0.001540359963|0.042941142163|
|PPU FlagGems FP32 新融合|1.879857898|4.581814766|0.001540359963|0.042941142163|
|H100原生 TF32（旧参考）|1.882853270|4.579020500|参考|参考|
|PPU FlagGems TF32 当前|1.883392215|4.578239441|1.289788062557|6.433917284226|
|PPU FlagGems TF32 固定未融合|1.883392215|4.578239441|1.289788062557|6.433917284226|
|PPU FlagGems TF32 新融合|1.883392215|4.578239441|1.289788062557|6.433917284226|

新融合与未融合完整训练向量相同，因此沿用同一旧 H100 误差；没有新增 H100 执行，也没有声称误差下降。这里验证两个步骤的轨迹不变，不外推为百步或九模型 checkpoint 质量通过。

- FP32：原生ATen观察到 218 次输入满足融合条件；其他dtype/记录高阶导入口 0 次。只读观察器不替换内核或梯度；实际包来源及 SHA 已校验。
- TF32：原生ATen观察到 218 次输入满足融合条件；其他dtype/记录高阶导入口 0 次。只读观察器不替换内核或梯度；实际包来源及 SHA 已校验。

## 边界与证据

非末轴固定配置修补另完成16个真实步骤，解决计时选配置可能改变反向的问题，对H100误差未下降。本项融合另计4个实际步骤（仅在CPU最终审核完成后计数）。旧空间前向候选四输入同权重仍12/16通过，百步新checkpoint主验收仍0/8；原5%置信度和0.1 Å同索引全原子RMSD门槛保持。没有特殊化输入、删小梯度、改监督、混H权重、换参考或改门槛。

[融合源码补丁](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/fused_softmax_backward_v2_flaggems_20261024.patch)；[单文件包构建证明](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/fused_softmax_backward_v2_build_receipt_20261024.json)；[全部算子原始标量与七轮计时](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/fused_softmax_backward_v2_checks_complete_or_progress_20261024.json)；[真实训练与CPU审核](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/fused_backward_v2_training_final_results_20261024.json)；[固定配置修补](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/NONINNER_SOFTMAX_BACKWARD_REPAIR_20261024.md)。

## 审计与补充执行证据

新鲜GPT-6-Astra审计结论WARN（same-family/provisional），只覆盖冻结的23文件算子快照。数字、保存轮次和算术自洽；不等于独立复现远端raw，也不覆盖随后完成的4步训练、附加布局计时或非连续直接读候选。原判定保留。

[原审计全文](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/experiment_integrity_softmax_fusion_20261024/EXPERIMENT_AUDIT.md)；[字段语义和评价类型补充](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/fused_backward_measurement_scope_and_equality_schema_20261024.json)；[5种实际布局的合成数值补测](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/OBSERVED_SOFTMAX_LAYOUTS_SYNTHETIC_API_20261024.md)；[实际注册函数与新进程autograd编译缓存证明](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/fused_backward_registered_v2_final_and_v3_reduction_IR_20261024.json)。

非连续直接读取v3及布局传播v4均在真实完整VJP逐位检查失败，未训练、未计时，保留已验证的融合v2。独立诊断中读取和乘积相同，但编译器改变归约线程布局，末位不同；不会降低门槛或将其包装成成果。

[失败原因与IR](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/STRIDED_SOFTMAX_BACKWARD_V3_FINDINGS_20261024.md)；[本轮20实际步骤和最终选择](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/softmax_backward_round_complete_20261024.json)；[审计之后执行者补充](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/experiment_integrity_softmax_fusion_20261024/EXECUTOR_FOLLOWUP.md)。
