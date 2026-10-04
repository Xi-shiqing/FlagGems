# TF32 通用反向缓存与四模式对照（2026-10-03）

最终保留的实验接入方案为 backward_only：只在 PPU TF32 反向使用通用分块缓存，保留原有公开前向及 IEEE 反向。固定 64 MiB 临时缓存按实际字节数分块，直接写入最终梯度；没有新增模型或 shape 白名单。已经通过真实训练入口验证，尚未修改正式算子包的默认入口。

245-token 真实训练单独复测：当前 PPU TF32 27.1329 秒，最终方案 26.3254 秒，耗时减少 2.98%。只有 3 个正式步骤，不能据此保证长期稳定收益。

合并前后向的 combined 候选仅作为实验记录保留：18 组串行推理耗时变化中位数为 +0.08%，没有稳定推理收益，故其前向改动不纳入最终方案。以下 combined 列均为未采用候选。最终方案前向函数身份与原实现相同，不复制基线耗时冒充新测量。

算子耗时单位 ms。输入是 CPU 统一随机数生成的合成张量（B=1、N=S、H=4、D=32），不是实测模型激活。训练包含前向、反向、一次 SGD 更新；H100 原生 CUE 0.8 的严格 FP32 反向不支持，明确标为 N/A。实际完整模型训练使用官方 Protenix 训练 TriAttention（支持 IEEE），与推理 CUE 入口分别记录。

| S | scope | H100_FP32_ms | H100_TF32_ms | PPU_FP32_ms | PPU_TF32_ms | PPU_combined_TF32_ms | new_vs_old_TF32_speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 131 | inference | 0.2197 | 0.1919 | 0.3133 | 0.2703 | 0.2866 | 0.9430 |
| 131 | training | N/A | 0.8492 | 3.3656 | 1.8997 | 1.3723 | 1.3844 |
| 245 | inference | 0.5425 | 0.3506 | 0.9396 | 0.8134 | 0.7211 | 1.1279 |
| 245 | training | N/A | 1.5830 | 9.1948 | 6.2653 | 3.9711 | 1.5777 |
| 384 | inference | 1.6584 | 0.9304 | 2.5789 | 2.1738 | 1.7793 | 1.2218 |
| 384 | training | N/A | 4.1282 | 18.6771 | 19.0227 | 12.0648 | 1.5767 |

真实训练：Protenix-v2，48 Pairformer + 4 MSA blocks，131/245 tokens，4 个训练步骤，统计后 3 步。并行首轮耗时仅留档，不作为加速结论；245 tokens 在自有推理工作结束后单独复测。训练沿用既有入口，明确接入并计数的 FlagGems 算子是 TriAttention；全局 enable_flaggems=false，triangle_multiplicative=torch，不能把这组训练称为所有算子都走 FlagGems。两服务器统一随机数程序 SHA、增强后全部输入、初始参数和有序随机事件均一致。

| tokens | H100_FP32_ms | H100_TF32_ms | PPU_FP32_ms | PPU_TF32_ms | PPU_combined_TF32_ms | new_vs_old_TF32_change_pct | PPU_final_backward_only_TF32_ms | final_vs_current_TF32_change_pct | final_PPU_TF32_vs_H100_TF32_time_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 245 | 6658.1826 | 4832.8413 | 33779.8906 | 27132.8926 | 26026.5664 | -4.0774 | 26325.4258 | -2.9760 | 5.4472 |

九模型真实推理：G01=693 tokens，G02=245 tokens；1 warmup + 3 次交错重复，cycle/step/sample=1。PPU 正式性能结果使用无自有训练并行的串行复测，预测和置信度必须与首轮完全相同。这里只代表这个测试协议，不能作为完整扩散采样的性能结论。

| input | tokens | model | H100_FP32_s | H100_TF32_s | PPU_FP32_s | PPU_TF32_s | PPU_combined_TF32_s | new_vs_old_TF32_change_pct | PPU_TF32_vs_H100_TF32_time_ratio | timing_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | 693 | protenix-v2 | 7.7965 | 4.3154 | 21.7254 | 17.1386 | 17.1261 | -0.0733 | 3.9715 | serial |
| G01 | 693 | protenix_base_20250630_v1.0.0 | 3.8565 | 2.4736 | 11.4433 | 8.6506 | 8.7417 | 1.0528 | 3.4971 | serial |
| G01 | 693 | protenix_base_constraint_v0.5.0 | 3.5048 | 2.2550 | 10.4407 | 7.8358 | 7.8412 | 0.0686 | 3.4749 | serial |
| G01 | 693 | protenix_base_default_v0.5.0 | 3.5053 | 2.2530 | 10.4102 | 7.8249 | 7.8368 | 0.1523 | 3.4731 | serial |
| G01 | 693 | protenix_base_default_v1.0.0 | 3.8354 | 2.4651 | 11.4328 | 8.6156 | 8.6148 | -0.0095 | 3.4951 | serial |
| G01 | 693 | protenix_mini_default_v0.5.0 | 1.3926 | 0.9415 | 4.0171 | 3.0625 | 3.0654 | 0.0934 | 3.2528 | serial |
| G01 | 693 | protenix_mini_esm_v0.5.0 | 1.4007 | 0.9368 | 4.0011 | 3.0620 | 3.0520 | -0.3294 | 3.2686 | serial |
| G01 | 693 | protenix_mini_ism_v0.5.0 | 1.3942 | 0.9429 | 3.9941 | 3.0555 | 3.0491 | -0.2075 | 3.2406 | serial |
| G01 | 693 | protenix_tiny_default_v0.5.0 | 0.9545 | 0.6587 | 2.6750 | 2.0718 | 2.0626 | -0.4415 | 3.1454 | serial |
| G02 | 245 | protenix-v2 | 0.9544 | 0.5959 | 2.6389 | 2.3242 | 2.3384 | 0.6112 | 3.9006 | serial |
| G02 | 245 | protenix_base_20250630_v1.0.0 | 0.4898 | 0.4766 | 1.8244 | 1.8342 | 1.8543 | 1.0949 | 3.8488 | serial |
| G02 | 245 | protenix_base_constraint_v0.5.0 | 0.4514 | 0.4438 | 1.6338 | 1.5559 | 1.5462 | -0.6247 | 3.5059 | serial |
| G02 | 245 | protenix_base_default_v0.5.0 | 0.4558 | 0.4517 | 1.6470 | 1.6611 | 1.6411 | -1.2052 | 3.6777 | serial |
| G02 | 245 | protenix_base_default_v1.0.0 | 0.4936 | 0.4894 | 1.8516 | 1.7268 | 1.8184 | 5.3020 | 3.5284 | serial |
| G02 | 245 | protenix_mini_default_v0.5.0 | 0.2021 | 0.1929 | 0.7910 | 0.7475 | 0.7840 | 4.8954 | 3.8756 | serial |
| G02 | 245 | protenix_mini_esm_v0.5.0 | 0.2383 | 0.1985 | 0.7797 | 0.7633 | 0.7478 | -2.0301 | 3.8445 | serial |
| G02 | 245 | protenix_mini_ism_v0.5.0 | 0.2086 | 0.1995 | 0.7873 | 0.7520 | 0.7590 | 0.9385 | 3.7690 | serial |
| G02 | 245 | protenix_tiny_default_v0.5.0 | 0.1557 | 0.1529 | 0.6352 | 0.6206 | 0.6370 | 2.6540 | 4.0588 | serial |

RMSD 门槛为 0.1 Å，五项置信度逐项相对误差门槛为 5%。跨设备坐标不导出，本轮使用已冻结的直接跨设备 RMSD 和各服务器实测的新旧预测漂移，计算严格区间；另加入 22 个已有混合精度锚点收紧区间，仍跨门槛时必须记为 UNRESOLVED。候选与当前 PPU TF32 的坐标和五项置信度全部完全相同。

| input | model | PPU_FP32_vs_H100_FP32_lower_A | PPU_FP32_vs_H100_FP32_upper_A | FP32_gate | PPU_TF32_vs_H100_TF32_lower_A | PPU_TF32_vs_H100_TF32_upper_A | TF32_gate | candidate_vs_current_PPU_TF32_RMSD_A | PPU_TF32_vs_local_FP32_RMSD_A | H100_TF32_vs_local_FP32_RMSD_A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | protenix-v2 | 4.928e-05 | 0.0198 | PASS | 0.9652 | 1.0507 | FAIL | 0 | 0.0615 | 1.0142 |
| G01 | protenix_base_20250630_v1.0.0 | 2.039e-05 | 0.0168 | PASS | 1.1234 | 1.2150 | FAIL | 0 | 0.1783 | 1.0956 |
| G01 | protenix_base_constraint_v0.5.0 | 3.521e-04 | 0.0076 | PASS | 4.6318 | 4.6588 | FAIL | 0 | 0.0470 | 4.6791 |
| G01 | protenix_base_default_v0.5.0 | 8.109e-06 | 0.0066 | PASS | 0.9808 | 1.0214 | FAIL | 0 | 0.0263 | 0.9989 |
| G01 | protenix_base_default_v1.0.0 | 1.208e-04 | 0.0236 | PASS | 2.1430 | 2.2029 | FAIL | 0 | 0.0761 | 2.2066 |
| G01 | protenix_mini_default_v0.5.0 | 2.251e-05 | 0.0057 | PASS | 0.6386 | 0.6888 | FAIL | 0 | 0.0644 | 0.6800 |
| G01 | protenix_mini_esm_v0.5.0 | 2.062e-05 | 0.0081 | PASS | 0.2605 | 0.2913 | FAIL | 0 | 0.0427 | 0.2731 |
| G01 | protenix_mini_ism_v0.5.0 | 1.371e-05 | 0.0053 | PASS | 0.2310 | 0.2632 | FAIL | 0 | 0.0450 | 0.2555 |
| G01 | protenix_tiny_default_v0.5.0 | 1.372e-05 | 0.0064 | PASS | 0.4567 | 0.5061 | FAIL | 0 | 0.0692 | 0.4493 |
| G02 | protenix-v2 | 1.097e-04 | 0.0114 | PASS | 0.0541 | 0.1226 | UNRESOLVED | 0 | 0.0626 | 0.1166 |
| G02 | protenix_base_20250630_v1.0.0 | 2.129e-04 | 0.0410 | PASS | 0.0649 | 0.1870 | UNRESOLVED | 0 | 0.1593 | 0.1188 |
| G02 | protenix_base_constraint_v0.5.0 | 3.425e-05 | 0.0171 | PASS | 0.0358 | 0.0692 | PASS | 0 | 0.0431 | 0.0306 |
| G02 | protenix_base_default_v0.5.0 | 5.610e-05 | 0.0094 | PASS | 0.0712 | 0.0934 | FAIL | 0 | 0.0383 | 0.0552 |
| G02 | protenix_base_default_v1.0.0 | 2.374e-05 | 0.0136 | PASS | 0.0873 | 0.2009 | UNRESOLVED | 0 | 0.0587 | 0.1459 |
| G02 | protenix_mini_default_v0.5.0 | 2.850e-06 | 0.0018 | PASS | 0.0114 | 0.0374 | PASS | 0 | 0.0161 | 0.0254 |
| G02 | protenix_mini_esm_v0.5.0 | 1.993e-05 | 0.0137 | PASS | 0.0807 | 0.1641 | UNRESOLVED | 0 | 0.0499 | 0.1201 |
| G02 | protenix_mini_ism_v0.5.0 | 2.359e-05 | 0.0084 | PASS | 0.0492 | 0.0951 | PASS | 0 | 0.0494 | 0.0844 |
| G02 | protenix_tiny_default_v0.5.0 | 5.045e-06 | 0.0077 | PASS | 0.2802 | 0.3040 | FAIL | 0 | 0.0368 | 0.2997 |

| tokens | device | precision | reference | gradient_relative_l2 | update_relative_l2 | update_max_abs | audit_step | learning_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 131 | ppu | fp32 | same-device strict FP32 | 0 | 0 | 0 | 1 | 1.800e-04 |
| 131 | ppu | tf32 | same-device strict FP32 | 3.229e-04 | 0.0150 | 2.932e-04 | 1 | 1.800e-04 |
| 131 | ppu | combined_tf32 | same-device current TF32 | 4.630e-06 | 0.0036 | 2.406e-04 | 1 | 1.800e-04 |
| 131 | h100 | fp32 | same-device strict FP32 | 0 | 0 | 0 | 1 | 1.800e-04 |
| 131 | h100 | tf32 | same-device strict FP32 | 8.446e-04 | 0.0132 | 2.840e-04 | 1 | 1.800e-04 |
| 245 | ppu | fp32 | same-device strict FP32 | 0 | 0 | 0 | 1 | 1.800e-04 |
| 245 | ppu | tf32 | same-device strict FP32 | 3.426e-04 | 0.0110 | 3.106e-04 | 1 | 1.800e-04 |
| 245 | ppu | combined_tf32 | same-device current TF32 | 4.023e-06 | 0.0012 | 2.572e-04 | 1 | 1.800e-04 |
| 245 | h100 | fp32 | same-device strict FP32 | 0 | 0 | 0 | 1 | 1.800e-04 |
| 245 | h100 | tf32 | same-device strict FP32 | 0.0011 | 0.0129 | 3.068e-04 | 1 | 1.800e-04 |
| 131 | ppu | final_backward_only_tf32 | same-device current TF32 | 4.279e-06 | 0.0034 | 2.900e-04 | 1 | 1.800e-04 |
| 245 | ppu | final_backward_only_tf32 | same-device current TF32 | 3.904e-06 | 0.0011 | 2.576e-04 | 1 | 1.800e-04 |

五项置信度的完整四模式数值见 nine_model_confidence.csv；算子输出/梯度/更新误差见 operator_accuracy.csv。第 0 步学习率为零，其零更新不能作为精度证据；已另外重跑第 1 步的非零 Adam 更新（学习率 0.00018），包括最终 backward_only，覆盖全部 4170 个有梯度参数及 4174 个参数更新。最终方案相对当前 PPU TF32 的梯度相对 L2 为 3.90e-6～4.28e-6，Adam 更新相对 L2 为 0.108%～0.342%，最大绝对更新差约 2.6e-4～2.9e-4；不能把它描述为更新逐位一致或已有业务阈值全部通过。完整训练的跨设备全参数梯度 L2 尚未直接计算；真实训练只覆盖 Protenix-v2 的两个裁剪长度，并非九模型训练验收。

筛选门槛：FP32 {'PASS': 18}；TF32 {'FAIL': 11, 'UNRESOLVED': 4, 'PASS': 3}。TF32 未整体过关；未决 RMSD 保留未决标记，未作为通过处理。

PPU 首轮两组推理和训练都被 env_ppu.sh 重置到 GPU 7，产生同卡竞争，首轮并行耗时作废；GPU 选择已放在环境脚本之后，正式串行推理与训练复测使用 GPU 7、无本实验的并发任务。H100 最初训练记录因加载旧随机数程序而作废，保存于 training_failed_rng；精度开关被模型构造函数重置而未进入训练的运行保存于 training_failed_control。有效训练显式设置 configs.enable_tf32，并设置官方训练 TriAttention 的 deterministic 参数，记录真实调用数。训练性能复测的损失与数值审计运行存在少量重复差异，具体差值记录于 summary.json；不声称训练逐位确定。

所有原始模型参数、激活、坐标和梯度快照保留在原服务器；本目录只含自有代码、标量和哈希。3 次重复的微小耗时差异只作筛选证据，不宣称稳定的端到端提升。
