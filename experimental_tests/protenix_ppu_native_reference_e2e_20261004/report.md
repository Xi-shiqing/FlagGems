# TF32 原生参考的模型级消融（2026-10-04）

本轮没有新增 PPU 提速。主要新证据是模型级精度定位：替换 H100 TriAttention 后，相对自身严格 FP32 的结构漂移减少比例中位数为 82.43%；RMSD≤0.1 Å 的组合从原生 TF32 的 4/18 增至自研诊断的 16/18，PPU 当前 TF32 也是 16/18。这些是各设备相对自身 FP32 的检查，不是跨设备验收通过率。

置信度头补测消除了 base v0.5 两输入的 pLDDT 失败，六组合五项指标全部过 5% 差异门；结合 RMSD 区间，六组合为 3 PASS / 3 UNRESOLVED。正式冻结原生 TF32 参考仍为 3 PASS / 11 FAIL / 4 UNRESOLVED；项目整体精度和 80% 效率目标尚未通过。

本轮完成九模型 × G01/G02 的四模式复测，以及 H100 的两条 TriAttention 诊断路径。所有四模式基线的预测哈希和五项置信度与上一轮完全相同；检查点、完整输入特征、原子顺序、外部资产和有序随机事件审计通过。原始张量和坐标保留在各自服务器。

**原生参考保留不变。** H100_mixed_triangle_IEEE 仅在原生 TriAttention 调用内关闭 TF32，其余矩阵仍为 TF32；它是混合精度消融。H100_custom_triangle_RNE 使用已有、通过算子精度检查的 CUDA 一次舍入内核；它改变整个 TriAttention 实现，是自研诊断路径，不能称为 H100 原生 TF32，也不是只修 Q 舍入的实验。PPU 本轮没有修改算子。

输入：G01=693 tokens、G02=245 tokens。cycle/step/sample=1，每模式 1 次预热、3 次测量。两个独立 worker 分别使用各设备的 GPU 0/7，同一组合的所有模式位于同一 GPU。本轮没有自己的训练任务并行；小百分比耗时差异不作为稳定收益。单位：秒。

| input | tokens | model | H100_native_FP32_s | H100_native_TF32_s | PPU_FlagGems_FP32_s | PPU_FlagGems_TF32_s | H100_mixed_triangle_IEEE_s | H100_custom_triangle_RNE_s | PPU_TF32_vs_native_H100_TF32_time_ratio | PPU_gpu | H100_gpu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | 693 | protenix-v2 | 7.7531 | 4.2938 | 21.8002 | 17.2020 | 5.3626 | 4.6704 | 4.0062 | 0 | 0 |
| G01 | 693 | protenix_base_20250630_v1.0.0 | 3.8280 | 2.4490 | 11.4258 | 8.6458 | 3.0311 | 2.6433 | 3.5304 | 7 | 7 |
| G01 | 693 | protenix_base_constraint_v0.5.0 | 3.4984 | 2.2316 | 10.4777 | 7.8369 | 2.7443 | 2.4283 | 3.5117 | 0 | 0 |
| G01 | 693 | protenix_base_default_v0.5.0 | 3.4872 | 2.2408 | 10.4614 | 8.0062 | 2.7451 | 2.4210 | 3.5729 | 7 | 7 |
| G01 | 693 | protenix_base_default_v1.0.0 | 3.8359 | 2.4546 | 11.4794 | 8.6537 | 3.0445 | 2.6423 | 3.5255 | 0 | 0 |
| G01 | 693 | protenix_mini_default_v0.5.0 | 1.3843 | 0.9207 | 4.0350 | 3.0779 | 1.1153 | 0.9953 | 3.3428 | 7 | 7 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | 1.3878 | 0.9282 | 4.0029 | 3.0600 | 1.1164 | 0.9949 | 3.2966 | 0 | 0 |
| G01 | 693 | protenix_mini_ism_v0.5.0 | 1.3983 | 0.9221 | 4.0015 | 3.0578 | 1.1155 | 0.9913 | 3.3162 | 7 | 7 |
| G01 | 693 | protenix_tiny_default_v0.5.0 | 0.9413 | 0.6457 | 2.6451 | 2.0594 | 0.7644 | 0.6824 | 3.1895 | 0 | 0 |
| G02 | 245 | protenix-v2 | 0.9444 | 0.5931 | 2.5969 | 2.3327 | 0.6364 | 0.6029 | 3.9328 | 0 | 0 |
| G02 | 245 | protenix_base_20250630_v1.0.0 | 0.4896 | 0.4896 | 2.0161 | 1.9492 | 0.4850 | 0.4561 | 3.9812 | 7 | 7 |
| G02 | 245 | protenix_base_constraint_v0.5.0 | 0.4518 | 0.4440 | 1.6399 | 1.5608 | 0.4570 | 0.4332 | 3.5154 | 0 | 0 |
| G02 | 245 | protenix_base_default_v0.5.0 | 0.4483 | 0.4188 | 1.6337 | 1.5549 | 0.4137 | 0.4123 | 3.7131 | 7 | 7 |
| G02 | 245 | protenix_base_default_v1.0.0 | 0.4957 | 0.4905 | 1.7770 | 1.6937 | 0.4734 | 0.4422 | 3.4533 | 0 | 0 |
| G02 | 245 | protenix_mini_default_v0.5.0 | 0.1983 | 0.1870 | 0.8470 | 0.8564 | 0.1868 | 0.1985 | 4.5800 | 7 | 7 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | 0.2005 | 0.1900 | 0.7903 | 0.7924 | 0.1900 | 0.1799 | 4.1697 | 0 | 0 |
| G02 | 245 | protenix_mini_ism_v0.5.0 | 0.2207 | 0.2414 | 0.8172 | 0.8117 | 0.2152 | 0.1851 | 3.3622 | 7 | 7 |
| G02 | 245 | protenix_tiny_default_v0.5.0 | 0.1479 | 0.1387 | 0.6263 | 0.6161 | 0.1404 | 0.1424 | 4.4416 | 0 | 0 |

下面是各设备 TF32 相对自身严格 FP32 的**直接实测** Kabsch RMSD，单位 Å。减少百分比描述替换 TriAttention 后的整体结构漂移变化，不能解释为单个舍入错误的独立贡献比例。

| input | tokens | model | H100_native_FP32_vs_own_FP32_A | H100_native_TF32_vs_own_FP32_A | PPU_FP32_vs_own_FP32_A | PPU_TF32_vs_own_FP32_A | H100_mixed_triangle_IEEE_vs_own_FP32_A | H100_custom_triangle_RNE_vs_own_FP32_A | H100_custom_triangle_RNE_vs_native_TF32_A | H100_drift_reduction_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | 693 | protenix-v2 | 0 | 1.0142 | 0 | 0.0615 | 0.0659 | 0.0818 | 1.0080 | 91.9389 |
| G01 | 693 | protenix_base_20250630_v1.0.0 | 0 | 1.0956 | 0 | 0.1783 | 0.2322 | 0.1955 | 1.1698 | 82.1542 |
| G01 | 693 | protenix_base_constraint_v0.5.0 | 0 | 4.6791 | 0 | 0.0470 | 0.0631 | 0.0619 | 4.6346 | 98.6771 |
| G01 | 693 | protenix_base_default_v0.5.0 | 0 | 0.9989 | 0 | 0.0263 | 0.0288 | 0.0333 | 0.9853 | 96.6679 |
| G01 | 693 | protenix_base_default_v1.0.0 | 0 | 2.2066 | 0 | 0.0761 | 0.0931 | 0.0675 | 2.1738 | 96.9406 |
| G01 | 693 | protenix_mini_default_v0.5.0 | 0 | 0.6800 | 0 | 0.0644 | 0.0645 | 0.0580 | 0.6595 | 91.4730 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | 0 | 0.2731 | 0 | 0.0427 | 0.0409 | 0.0426 | 0.2860 | 84.4106 |
| G01 | 693 | protenix_mini_ism_v0.5.0 | 0 | 0.2555 | 0 | 0.0450 | 0.0436 | 0.0442 | 0.2480 | 82.6973 |
| G01 | 693 | protenix_tiny_default_v0.5.0 | 0 | 0.4493 | 0 | 0.0692 | 0.0736 | 0.0591 | 0.4785 | 86.8406 |
| G02 | 245 | protenix-v2 | 0 | 0.1166 | 0 | 0.0626 | 0.0677 | 0.0765 | 0.0762 | 34.4312 |
| G02 | 245 | protenix_base_20250630_v1.0.0 | 0 | 0.1188 | 0 | 0.1593 | 0.1485 | 0.1692 | 0.1611 | -42.4454 |
| G02 | 245 | protenix_base_constraint_v0.5.0 | 0 | 0.0306 | 0 | 0.0431 | 0.0261 | 0.0311 | 0.0531 | -1.6974 |
| G02 | 245 | protenix_base_default_v0.5.0 | 0 | 0.0552 | 0 | 0.0383 | 0.0423 | 0.0408 | 0.0614 | 26.1545 |
| G02 | 245 | protenix_base_default_v1.0.0 | 0 | 0.1459 | 0 | 0.0587 | 0.0908 | 0.0892 | 0.1762 | 38.8528 |
| G02 | 245 | protenix_mini_default_v0.5.0 | 0 | 0.0254 | 0 | 0.0161 | 0.0159 | 0.0162 | 0.0291 | 36.1944 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | 0 | 0.1201 | 0 | 0.0499 | 0.0444 | 0.0513 | 0.1389 | 57.2942 |
| G02 | 245 | protenix_mini_ism_v0.5.0 | 0 | 0.0844 | 0 | 0.0494 | 0.0544 | 0.0556 | 0.0733 | 34.0492 |
| G02 | 245 | protenix_tiny_default_v0.5.0 | 0 | 0.2997 | 0 | 0.0368 | 0.0396 | 0.0411 | 0.2963 | 86.2765 |

跨设备筛选：冻结原生参考 {'fp32': {'PASS': 18}, 'tf32': {'FAIL': 11, 'UNRESOLVED': 4, 'PASS': 3}}；诊断参考 {'tf32_triangle_ieee': {'UNRESOLVED': 11, 'FAIL': 2, 'PASS': 5}, 'tf32_triangle_corrected': {'UNRESOLVED': 11, 'FAIL': 2, 'PASS': 5}}。诊断门通过不能替代项目原生参考验收。

跨设备 RMSD 仍使用已冻结的直接跨设备锚点及本轮各服务器直接实测的新旧预测距离，取三角不等式严格区间。门槛为 0.1 Å；五项置信度逐项相对差 ≤5%。区间跨越门槛时记 UNRESOLVED，不冒充精确距离。

| input | tokens | model | native_fp32_lower_A | native_fp32_upper_A | native_fp32_gate | native_tf32_lower_A | native_tf32_upper_A | native_tf32_gate | mixed_triangle_ieee_lower_A | mixed_triangle_ieee_upper_A | mixed_triangle_ieee_gate | custom_triangle_rne_lower_A | custom_triangle_rne_upper_A | custom_triangle_rne_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | 693 | protenix-v2 | 4.928e-05 | 0.0198 | PASS | 0.9652 | 1.0507 | FAIL | 0 | 0.1378 | UNRESOLVED | 0.0077 | 0.1506 | UNRESOLVED |
| G01 | 693 | protenix_base_20250630_v1.0.0 | 2.039e-05 | 0.0168 | PASS | 1.0017 | 1.2264 | FAIL | 0.0502 | 0.4237 | UNRESOLVED | 0.0128 | 0.3863 | UNRESOLVED |
| G01 | 693 | protenix_base_constraint_v0.5.0 | 3.521e-04 | 0.0076 | PASS | 4.6318 | 4.6601 | FAIL | 0.0679 | 0.1133 | UNRESOLVED | 0.0116 | 0.1132 | UNRESOLVED |
| G01 | 693 | protenix_base_default_v0.5.0 | 8.109e-06 | 0.0066 | PASS | 0.9808 | 1.0214 | FAIL | 0 | 0.0588 | FAIL | 0.0043 | 0.0636 | FAIL |
| G01 | 693 | protenix_base_default_v1.0.0 | 1.208e-04 | 0.0236 | PASS | 2.1306 | 2.2029 | FAIL | 0.0142 | 0.1900 | UNRESOLVED | 0 | 0.1622 | UNRESOLVED |
| G01 | 693 | protenix_mini_default_v0.5.0 | 2.251e-05 | 0.0057 | PASS | 0.6381 | 0.6888 | FAIL | 0 | 0.1320 | UNRESOLVED | 0.0037 | 0.1251 | UNRESOLVED |
| G01 | 693 | protenix_mini_esm_v0.5.0 | 2.062e-05 | 0.0081 | PASS | 0.2605 | 0.2913 | FAIL | 0 | 0.0881 | PASS | 0 | 0.0893 | PASS |
| G01 | 693 | protenix_mini_ism_v0.5.0 | 1.371e-05 | 0.0053 | PASS | 0.2310 | 0.2632 | FAIL | 0 | 0.0905 | PASS | 0 | 0.0912 | PASS |
| G01 | 693 | protenix_tiny_default_v0.5.0 | 1.372e-05 | 0.0064 | PASS | 0.4567 | 0.5061 | FAIL | 0.0022 | 0.1470 | UNRESOLVED | 0.0060 | 0.1324 | UNRESOLVED |
| G02 | 245 | protenix-v2 | 1.097e-04 | 0.0114 | PASS | 0.0541 | 0.1226 | UNRESOLVED | 0 | 0.1336 | UNRESOLVED | 0.0063 | 0.1430 | UNRESOLVED |
| G02 | 245 | protenix_base_20250630_v1.0.0 | 2.129e-04 | 0.0410 | PASS | 0.0649 | 0.1870 | UNRESOLVED | 0 | 0.3336 | UNRESOLVED | 0 | 0.3481 | UNRESOLVED |
| G02 | 245 | protenix_base_constraint_v0.5.0 | 3.425e-05 | 0.0171 | PASS | 0.0358 | 0.0726 | PASS | 0.0067 | 0.0795 | PASS | 0.0018 | 0.0844 | PASS |
| G02 | 245 | protenix_base_default_v0.5.0 | 5.610e-05 | 0.0094 | PASS | 0.0712 | 0.0934 | FAIL | 0.0077 | 0.0891 | FAIL | 0.0097 | 0.0870 | FAIL |
| G02 | 245 | protenix_base_default_v1.0.0 | 2.374e-05 | 0.0136 | PASS | 0.0873 | 0.2046 | UNRESOLVED | 0.0280 | 0.1589 | UNRESOLVED | 0.0247 | 0.1556 | UNRESOLVED |
| G02 | 245 | protenix_mini_default_v0.5.0 | 2.850e-06 | 0.0018 | PASS | 0.0109 | 0.0374 | PASS | 0 | 0.0330 | PASS | 0 | 0.0333 | PASS |
| G02 | 245 | protenix_mini_esm_v0.5.0 | 1.993e-05 | 0.0137 | PASS | 0.0807 | 0.1641 | UNRESOLVED | 0 | 0.1001 | UNRESOLVED | 0 | 0.1067 | UNRESOLVED |
| G02 | 245 | protenix_mini_ism_v0.5.0 | 2.359e-05 | 0.0084 | PASS | 0.0492 | 0.0951 | PASS | 0.0011 | 0.1084 | UNRESOLVED | 0.0026 | 0.1098 | UNRESOLVED |
| G02 | 245 | protenix_tiny_default_v0.5.0 | 5.045e-06 | 0.0077 | PASS | 0.2629 | 0.3040 | FAIL | 0 | 0.0805 | PASS | 0.0014 | 0.0826 | PASS |

全部五项置信度六列对照见 four_mode_and_diagnostic_confidence.csv；调用与精度入口记录见 ppu/derived.json、h100/derived.json。模型级测试没有重新测训练；前一轮通用反向缓存的训练结果应单独引用。

置信度头补测覆盖 base v0.5、v2、mini_esm 各 G01/G02，共六组合。以 ConfidenceHead 类为通用边界，仅在其 forward 内关闭矩阵 TF32；自研 TriAttention 的显式 TF32 保持不变。六组合的预测坐标哈希均完全不变，检查点、完整输入和有序随机事件均通过。这里的 H100_custom_RNE_confidence_matmul_IEEE 是额外的混合精度诊断列；不能当作 H100 原生 TF32。

| input | tokens | model | H100_native_FP32_s | H100_native_TF32_s | PPU_FlagGems_FP32_s | PPU_FlagGems_TF32_s | H100_mixed_triangle_IEEE_s | H100_custom_triangle_RNE_s | PPU_TF32_vs_native_H100_TF32_time_ratio | PPU_gpu | H100_gpu | H100_custom_RNE_confidence_matmul_IEEE_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | 693 | protenix_base_default_v0.5.0 | 3.4872 | 2.2408 | 10.4614 | 8.0062 | 2.7451 | 2.4210 | 3.5729 | 7 | 7 | 2.4460 |
| G02 | 245 | protenix_base_default_v0.5.0 | 0.4483 | 0.4188 | 1.6337 | 1.5549 | 0.4137 | 0.4123 | 3.7131 | 7 | 7 | 0.4264 |
| G01 | 693 | protenix-v2 | 7.7531 | 4.2938 | 21.8002 | 17.2020 | 5.3626 | 4.6704 | 4.0062 | 0 | 0 | 4.8279 |
| G02 | 245 | protenix-v2 | 0.9444 | 0.5931 | 2.5969 | 2.3327 | 0.6364 | 0.6029 | 3.9328 | 0 | 0 | 0.6114 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | 1.3878 | 0.9282 | 4.0029 | 3.0600 | 1.1164 | 0.9949 | 3.2966 | 0 | 0 | 1.0369 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | 0.2005 | 0.1900 | 0.7903 | 0.7924 | 0.1900 | 0.1799 | 4.1697 | 0 | 0 | 0.1885 |

| input | tokens | model | metric | H100_native_FP32 | H100_native_TF32 | PPU_FlagGems_FP32 | PPU_FlagGems_TF32 | H100_mixed_triangle_IEEE | H100_custom_triangle_RNE | H100_custom_RNE_confidence_matmul_IEEE | PPU_TF32_relative_error_before_pct | PPU_TF32_relative_error_after_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | 693 | protenix_base_default_v0.5.0 | plddt | 10.8115 | 10.8954 | 10.8109 | 10.9550 | 12.7629 | 12.7620 | 10.9332 | 14.1593 | 0.1994 |
| G01 | 693 | protenix_base_default_v0.5.0 | gpde | 0.2573 | 0.2572 | 0.2573 | 0.2573 | 0.2572 | 0.2572 | 0.2573 | 0.0335 | 0.0011 |
| G01 | 693 | protenix_base_default_v0.5.0 | ptm | 0.8860 | 0.8861 | 0.8860 | 0.8860 | 0.8861 | 0.8861 | 0.8860 | 0.0123 | 2.960e-04 |
| G01 | 693 | protenix_base_default_v0.5.0 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| G01 | 693 | protenix_base_default_v0.5.0 | ranking_score | 0.1772 | 0.1772 | 0.1772 | 0.1772 | 0.1772 | 0.1772 | 0.1772 | 0.0123 | 2.943e-04 |
| G02 | 245 | protenix_base_default_v0.5.0 | plddt | 8.4922 | 9.1108 | 8.4926 | 8.5016 | 9.0981 | 9.0981 | 8.4985 | 6.5565 | 0.0360 |
| G02 | 245 | protenix_base_default_v0.5.0 | gpde | 0.2580 | 0.2578 | 0.2580 | 0.2580 | 0.2578 | 0.2578 | 0.2580 | 0.0905 | 0.0015 |
| G02 | 245 | protenix_base_default_v0.5.0 | ptm | 0.8445 | 0.8446 | 0.8445 | 0.8445 | 0.8446 | 0.8446 | 0.8445 | 0.0086 | 0.0029 |
| G02 | 245 | protenix_base_default_v0.5.0 | iptm | 0.8435 | 0.8437 | 0.8435 | 0.8435 | 0.8437 | 0.8437 | 0.8435 | 0.0229 | 0.0026 |
| G02 | 245 | protenix_base_default_v0.5.0 | ranking_score | 0.8437 | 0.8439 | 0.8437 | 0.8437 | 0.8439 | 0.8439 | 0.8437 | 0.0201 | 0.0027 |
| G01 | 693 | protenix-v2 | plddt | 84.9767 | 84.9957 | 84.9766 | 84.9547 | 84.9733 | 84.9707 | 84.9672 | 0.0188 | 0.0147 |
| G01 | 693 | protenix-v2 | gpde | 1.0505 | 1.0492 | 1.0505 | 1.0509 | 1.0493 | 1.0493 | 1.0505 | 0.1572 | 0.0408 |
| G01 | 693 | protenix-v2 | ptm | 0.4568 | 0.4577 | 0.4568 | 0.4568 | 0.4577 | 0.4577 | 0.4568 | 0.2064 | 0.0204 |
| G01 | 693 | protenix-v2 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| G01 | 693 | protenix-v2 | ranking_score | 0.0914 | 0.0915 | 0.0914 | 0.0914 | 0.0915 | 0.0915 | 0.0914 | 0.2064 | 0.0204 |
| G02 | 245 | protenix-v2 | plddt | 87.3800 | 87.1814 | 87.3816 | 87.3728 | 87.1822 | 87.1747 | 87.3815 | 0.2273 | 0.0100 |
| G02 | 245 | protenix-v2 | gpde | 1.0829 | 1.0804 | 1.0829 | 1.0834 | 1.0804 | 1.0804 | 1.0829 | 0.2751 | 0.0394 |
| G02 | 245 | protenix-v2 | ptm | 0.3445 | 0.3498 | 0.3445 | 0.3444 | 0.3496 | 0.3496 | 0.3445 | 1.4816 | 0.0194 |
| G02 | 245 | protenix-v2 | iptm | 0.3428 | 0.3475 | 0.3428 | 0.3427 | 0.3473 | 0.3473 | 0.3428 | 1.3329 | 0.0226 |
| G02 | 245 | protenix-v2 | ranking_score | 0.3431 | 0.3480 | 0.3431 | 0.3430 | 0.3478 | 0.3478 | 0.3431 | 1.3628 | 0.0220 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | plddt | 38.1579 | 38.0471 | 38.1578 | 38.1513 | 38.1489 | 38.1580 | 38.1524 | 0.0175 | 0.0028 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | gpde | 8.7820 | 8.7969 | 8.7820 | 8.7779 | 8.7776 | 8.7771 | 8.7770 | 0.0094 | 0.0102 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | ptm | 0.2007 | 0.1999 | 0.2007 | 0.2006 | 0.2006 | 0.2006 | 0.2006 | 0.0078 | 0.0035 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | ranking_score | 0.0401 | 0.0400 | 0.0401 | 0.0401 | 0.0401 | 0.0401 | 0.0401 | 0.0078 | 0.0035 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | plddt | 61.1138 | 60.9744 | 61.1138 | 61.1000 | 61.1350 | 61.1337 | 61.1353 | 0.0551 | 0.0578 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | gpde | 2.1247 | 2.1365 | 2.1247 | 2.1246 | 2.1195 | 2.1239 | 2.1223 | 0.0318 | 0.1074 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | ptm | 0.3609 | 0.3602 | 0.3609 | 0.3604 | 0.3610 | 0.3609 | 0.3608 | 0.1359 | 0.1128 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | iptm | 0.7234 | 0.7215 | 0.7234 | 0.7218 | 0.7237 | 0.7233 | 0.7231 | 0.1943 | 0.1726 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | ranking_score | 0.6509 | 0.6492 | 0.6509 | 0.6496 | 0.6512 | 0.6508 | 0.6506 | 0.1878 | 0.1660 |

六组合补测的诊断门计数：{'PASS': 3, 'UNRESOLVED': 3}。两组 base v0.5 的 pLDDT 失败均被置信度头矩阵精度恢复消除；其他四项置信度也逐项检查。坐标不变，因此沿用同一 RMSD 严格区间；其余组合区间跨门槛仍记 UNRESOLVED。这是模块级消融，尚未隔离到头内单个 MM/BMM，也未推广为 PPU 正式策略或覆盖其余十二组合。

全部十八组合的五项置信度对照：

| input | tokens | model | metric | H100_native_FP32 | H100_native_TF32 | PPU_FlagGems_FP32 | PPU_FlagGems_TF32 | H100_mixed_triangle_IEEE | H100_custom_triangle_RNE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G01 | 693 | protenix-v2 | plddt | 84.9767 | 84.9957 | 84.9766 | 84.9547 | 84.9733 | 84.9707 |
| G01 | 693 | protenix-v2 | gpde | 1.0505 | 1.0492 | 1.0505 | 1.0509 | 1.0493 | 1.0493 |
| G01 | 693 | protenix-v2 | ptm | 0.4568 | 0.4577 | 0.4568 | 0.4568 | 0.4577 | 0.4577 |
| G01 | 693 | protenix-v2 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| G01 | 693 | protenix-v2 | ranking_score | 0.0914 | 0.0915 | 0.0914 | 0.0914 | 0.0915 | 0.0915 |
| G01 | 693 | protenix_base_20250630_v1.0.0 | plddt | 84.0807 | 84.1048 | 84.0807 | 84.0884 | 84.1111 | 84.1085 |
| G01 | 693 | protenix_base_20250630_v1.0.0 | gpde | 5.6216 | 5.6232 | 5.6216 | 5.6232 | 5.6237 | 5.6237 |
| G01 | 693 | protenix_base_20250630_v1.0.0 | ptm | 0.1664 | 0.1664 | 0.1664 | 0.1664 | 0.1664 | 0.1664 |
| G01 | 693 | protenix_base_20250630_v1.0.0 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| G01 | 693 | protenix_base_20250630_v1.0.0 | ranking_score | 0.0333 | 0.0333 | 0.0333 | 0.0333 | 0.0333 | 0.0333 |
| G01 | 693 | protenix_base_constraint_v0.5.0 | plddt | 85.3228 | 85.3237 | 85.3228 | 85.3285 | 85.3297 | 85.3307 |
| G01 | 693 | protenix_base_constraint_v0.5.0 | gpde | 0.2562 | 0.2562 | 0.2562 | 0.2562 | 0.2562 | 0.2562 |
| G01 | 693 | protenix_base_constraint_v0.5.0 | ptm | 0.9625 | 0.9625 | 0.9625 | 0.9624 | 0.9625 | 0.9625 |
| G01 | 693 | protenix_base_constraint_v0.5.0 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| G01 | 693 | protenix_base_constraint_v0.5.0 | ranking_score | 0.1925 | 0.1925 | 0.1925 | 0.1925 | 0.1925 | 0.1925 |
| G01 | 693 | protenix_base_default_v0.5.0 | plddt | 10.8115 | 10.8954 | 10.8109 | 10.9550 | 12.7629 | 12.7620 |
| G01 | 693 | protenix_base_default_v0.5.0 | gpde | 0.2573 | 0.2572 | 0.2573 | 0.2573 | 0.2572 | 0.2572 |
| G01 | 693 | protenix_base_default_v0.5.0 | ptm | 0.8860 | 0.8861 | 0.8860 | 0.8860 | 0.8861 | 0.8861 |
| G01 | 693 | protenix_base_default_v0.5.0 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| G01 | 693 | protenix_base_default_v0.5.0 | ranking_score | 0.1772 | 0.1772 | 0.1772 | 0.1772 | 0.1772 | 0.1772 |
| G01 | 693 | protenix_base_default_v1.0.0 | plddt | 80.0260 | 80.0708 | 80.0258 | 80.0198 | 80.0612 | 80.0620 |
| G01 | 693 | protenix_base_default_v1.0.0 | gpde | 7.3113 | 7.3124 | 7.3113 | 7.3125 | 7.3135 | 7.3135 |
| G01 | 693 | protenix_base_default_v1.0.0 | ptm | 0.2725 | 0.2724 | 0.2725 | 0.2724 | 0.2724 | 0.2724 |
| G01 | 693 | protenix_base_default_v1.0.0 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| G01 | 693 | protenix_base_default_v1.0.0 | ranking_score | 0.0545 | 0.0545 | 0.0545 | 0.0545 | 0.0545 | 0.0545 |
| G01 | 693 | protenix_mini_default_v0.5.0 | plddt | 60.5849 | 60.4863 | 60.5849 | 60.5840 | 60.5948 | 60.5936 |
| G01 | 693 | protenix_mini_default_v0.5.0 | gpde | 2.9087 | 2.9042 | 2.9087 | 2.9068 | 2.9050 | 2.9050 |
| G01 | 693 | protenix_mini_default_v0.5.0 | ptm | 0.1690 | 0.1692 | 0.1690 | 0.1690 | 0.1690 | 0.1690 |
| G01 | 693 | protenix_mini_default_v0.5.0 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| G01 | 693 | protenix_mini_default_v0.5.0 | ranking_score | 0.0338 | 0.0338 | 0.0338 | 0.0338 | 0.0338 | 0.0338 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | plddt | 38.1579 | 38.0471 | 38.1578 | 38.1513 | 38.1489 | 38.1580 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | gpde | 8.7820 | 8.7969 | 8.7820 | 8.7779 | 8.7776 | 8.7771 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | ptm | 0.2007 | 0.1999 | 0.2007 | 0.2006 | 0.2006 | 0.2006 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| G01 | 693 | protenix_mini_esm_v0.5.0 | ranking_score | 0.0401 | 0.0400 | 0.0401 | 0.0401 | 0.0401 | 0.0401 |
| G01 | 693 | protenix_mini_ism_v0.5.0 | plddt | 44.9136 | 44.9614 | 44.9136 | 44.9111 | 44.9127 | 44.9086 |
| G01 | 693 | protenix_mini_ism_v0.5.0 | gpde | 8.3106 | 8.3219 | 8.3106 | 8.3239 | 8.3226 | 8.3236 |
| G01 | 693 | protenix_mini_ism_v0.5.0 | ptm | 0.1695 | 0.1696 | 0.1695 | 0.1695 | 0.1695 | 0.1695 |
| G01 | 693 | protenix_mini_ism_v0.5.0 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| G01 | 693 | protenix_mini_ism_v0.5.0 | ranking_score | 0.0339 | 0.0339 | 0.0339 | 0.0339 | 0.0339 | 0.0339 |
| G01 | 693 | protenix_tiny_default_v0.5.0 | plddt | 32.4808 | 32.4382 | 32.4809 | 32.4688 | 32.4595 | 32.4693 |
| G01 | 693 | protenix_tiny_default_v0.5.0 | gpde | 9.0427 | 9.0645 | 9.0426 | 9.0572 | 9.0580 | 9.0561 |
| G01 | 693 | protenix_tiny_default_v0.5.0 | ptm | 0.1686 | 0.1684 | 0.1686 | 0.1686 | 0.1686 | 0.1686 |
| G01 | 693 | protenix_tiny_default_v0.5.0 | iptm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| G01 | 693 | protenix_tiny_default_v0.5.0 | ranking_score | 0.0337 | 0.0337 | 0.0337 | 0.0337 | 0.0337 | 0.0337 |
| G02 | 245 | protenix-v2 | plddt | 87.3800 | 87.1814 | 87.3816 | 87.3728 | 87.1822 | 87.1747 |
| G02 | 245 | protenix-v2 | gpde | 1.0829 | 1.0804 | 1.0829 | 1.0834 | 1.0804 | 1.0804 |
| G02 | 245 | protenix-v2 | ptm | 0.3445 | 0.3498 | 0.3445 | 0.3444 | 0.3496 | 0.3496 |
| G02 | 245 | protenix-v2 | iptm | 0.3428 | 0.3475 | 0.3428 | 0.3427 | 0.3473 | 0.3473 |
| G02 | 245 | protenix-v2 | ranking_score | 0.3431 | 0.3480 | 0.3431 | 0.3430 | 0.3478 | 0.3478 |
| G02 | 245 | protenix_base_20250630_v1.0.0 | plddt | 77.0690 | 77.3134 | 77.0685 | 77.0805 | 77.3352 | 77.3332 |
| G02 | 245 | protenix_base_20250630_v1.0.0 | gpde | 5.6629 | 5.6653 | 5.6629 | 5.6644 | 5.6658 | 5.6662 |
| G02 | 245 | protenix_base_20250630_v1.0.0 | ptm | 0.1181 | 0.1182 | 0.1181 | 0.1182 | 0.1182 | 0.1182 |
| G02 | 245 | protenix_base_20250630_v1.0.0 | iptm | 0.1147 | 0.1147 | 0.1147 | 0.1147 | 0.1147 | 0.1147 |
| G02 | 245 | protenix_base_20250630_v1.0.0 | ranking_score | 0.1154 | 0.1154 | 0.1154 | 0.1154 | 0.1154 | 0.1154 |
| G02 | 245 | protenix_base_constraint_v0.5.0 | plddt | 85.7434 | 85.7443 | 85.7434 | 85.7452 | 85.7442 | 85.7434 |
| G02 | 245 | protenix_base_constraint_v0.5.0 | gpde | 0.2568 | 0.2566 | 0.2568 | 0.2568 | 0.2566 | 0.2566 |
| G02 | 245 | protenix_base_constraint_v0.5.0 | ptm | 0.9315 | 0.9317 | 0.9315 | 0.9315 | 0.9317 | 0.9317 |
| G02 | 245 | protenix_base_constraint_v0.5.0 | iptm | 0.9310 | 0.9312 | 0.9310 | 0.9310 | 0.9312 | 0.9312 |
| G02 | 245 | protenix_base_constraint_v0.5.0 | ranking_score | 0.9311 | 0.9313 | 0.9311 | 0.9311 | 0.9313 | 0.9313 |
| G02 | 245 | protenix_base_default_v0.5.0 | plddt | 8.4922 | 9.1108 | 8.4926 | 8.5016 | 9.0981 | 9.0981 |
| G02 | 245 | protenix_base_default_v0.5.0 | gpde | 0.2580 | 0.2578 | 0.2580 | 0.2580 | 0.2578 | 0.2578 |
| G02 | 245 | protenix_base_default_v0.5.0 | ptm | 0.8445 | 0.8446 | 0.8445 | 0.8445 | 0.8446 | 0.8446 |
| G02 | 245 | protenix_base_default_v0.5.0 | iptm | 0.8435 | 0.8437 | 0.8435 | 0.8435 | 0.8437 | 0.8437 |
| G02 | 245 | protenix_base_default_v0.5.0 | ranking_score | 0.8437 | 0.8439 | 0.8437 | 0.8437 | 0.8439 | 0.8439 |
| G02 | 245 | protenix_base_default_v1.0.0 | plddt | 80.2621 | 80.5193 | 80.2612 | 80.2439 | 80.4462 | 80.4808 |
| G02 | 245 | protenix_base_default_v1.0.0 | gpde | 7.2691 | 7.2705 | 7.2691 | 7.2703 | 7.2707 | 7.2709 |
| G02 | 245 | protenix_base_default_v1.0.0 | ptm | 0.2111 | 0.2109 | 0.2111 | 0.2109 | 0.2109 | 0.2109 |
| G02 | 245 | protenix_base_default_v1.0.0 | iptm | 0.2074 | 0.2074 | 0.2074 | 0.2073 | 0.2074 | 0.2074 |
| G02 | 245 | protenix_base_default_v1.0.0 | ranking_score | 0.2082 | 0.2081 | 0.2082 | 0.2080 | 0.2081 | 0.2081 |
| G02 | 245 | protenix_mini_default_v0.5.0 | plddt | 77.4927 | 77.4227 | 77.4927 | 77.4971 | 77.4935 | 77.4941 |
| G02 | 245 | protenix_mini_default_v0.5.0 | gpde | 0.8870 | 0.8871 | 0.8870 | 0.8862 | 0.8864 | 0.8861 |
| G02 | 245 | protenix_mini_default_v0.5.0 | ptm | 0.7409 | 0.7403 | 0.7409 | 0.7408 | 0.7408 | 0.7408 |
| G02 | 245 | protenix_mini_default_v0.5.0 | iptm | 0.6915 | 0.6901 | 0.6915 | 0.6915 | 0.6916 | 0.6915 |
| G02 | 245 | protenix_mini_default_v0.5.0 | ranking_score | 0.7014 | 0.7002 | 0.7014 | 0.7013 | 0.7015 | 0.7013 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | plddt | 61.1138 | 60.9744 | 61.1138 | 61.1000 | 61.1350 | 61.1337 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | gpde | 2.1247 | 2.1365 | 2.1247 | 2.1246 | 2.1195 | 2.1239 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | ptm | 0.3609 | 0.3602 | 0.3609 | 0.3604 | 0.3610 | 0.3609 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | iptm | 0.7234 | 0.7215 | 0.7234 | 0.7218 | 0.7237 | 0.7233 |
| G02 | 245 | protenix_mini_esm_v0.5.0 | ranking_score | 0.6509 | 0.6492 | 0.6509 | 0.6496 | 0.6512 | 0.6508 |
| G02 | 245 | protenix_mini_ism_v0.5.0 | plddt | 59.1991 | 59.1883 | 59.1991 | 59.2107 | 59.2134 | 59.2115 |
| G02 | 245 | protenix_mini_ism_v0.5.0 | gpde | 2.7673 | 2.7664 | 2.7673 | 2.7681 | 2.7677 | 2.7678 |
| G02 | 245 | protenix_mini_ism_v0.5.0 | ptm | 0.3719 | 0.3726 | 0.3719 | 0.3720 | 0.3721 | 0.3720 |
| G02 | 245 | protenix_mini_ism_v0.5.0 | iptm | 0.6902 | 0.6907 | 0.6902 | 0.6905 | 0.6907 | 0.6905 |
| G02 | 245 | protenix_mini_ism_v0.5.0 | ranking_score | 0.6266 | 0.6271 | 0.6266 | 0.6268 | 0.6270 | 0.6268 |
| G02 | 245 | protenix_tiny_default_v0.5.0 | plddt | 50.9682 | 50.8659 | 50.9682 | 50.9759 | 50.9797 | 50.9780 |
| G02 | 245 | protenix_tiny_default_v0.5.0 | gpde | 1.8140 | 1.8153 | 1.8140 | 1.8137 | 1.8133 | 1.8138 |
| G02 | 245 | protenix_tiny_default_v0.5.0 | ptm | 0.5054 | 0.5033 | 0.5054 | 0.5055 | 0.5055 | 0.5054 |
| G02 | 245 | protenix_tiny_default_v0.5.0 | iptm | 0.1672 | 0.1669 | 0.1672 | 0.1670 | 0.1671 | 0.1673 |
| G02 | 245 | protenix_tiny_default_v0.5.0 | ranking_score | 0.2349 | 0.2342 | 0.2349 | 0.2347 | 0.2348 | 0.2349 |

适配器首轮错误地去掉了 native low-level CUE 保留的 batch 轴，导致真实调用方 [0] 取走第一个序列组。虽然合成测试按 reshape 比较数值能通过，实际模型输出严重偏差。已修复输出 rank 契约，增加与原生返回 shape 的直接断言，并重跑全部 18 组自研诊断。首轮错误结果作废、仅留在服务器审计目录，不进入上述表格。

此前原生算子位置控制已在冻结 CUE 0.8 和隔离 CUE 0.12 上复现：单个有效 key 从位置 0 移至 244，数学 score 应恒为 1，原生 TF32 却到 1.0078125；一次舍入的 PPU/CUDA 控制无该漂移。对应标量证据与来源哈希保存在 native_reference_prior_evidence.json。本轮用整个 TriAttention 的消融量化模型行为，不能把整内核差异全部归给 Q 漂移。

本轮另提供不依赖模型或检查点的 reproduce_native_query_drift.py；H100 原生 CUE 0.8 实跑记录见 h100/native_query_reproducer.json。十六个 key 位置中，FP32 的 score/LSE 恒为 1，TF32 在末位置为 1.0078125。V 为常数时输出仍正确，说明只比较 attention 输出会漏检这一内部错误。

原生 API 的 dtype、布局、mask、scale 与 TF32 全局开关契约参见 [NVIDIA Triangle Attention 文档](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.triangle_attention.html)。本轮默认软件包、项目正式入口和冻结基线均未修改。
