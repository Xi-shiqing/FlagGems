# PPU BMM 准备开销实验（2026-09-26）

基线是服务器 `generic_pipeline_20260926/candidate`：已包含 native RNE、mean 修复和连续 FP32 fill 优化。这里保留精确的 BMM 文件及独立补丁，不覆盖共享安装。

| 文件 | 相对基线的改动 |
| --- | --- |
| `baseline_bmm.py` | 原始基线，双舍入缓冲区、独立转换 kernel |
| `cached_bmm.py` | 两个辅助 kernel 加 `@libentry()` |
| `packed_bmm.py` | 启动缓存 + 将 A/B 舍入数据放入一次分配 |
| `packed_only_bmm.py` | 仅合并分配，B 从同一存储中 A 之后的位置读取 |
| `fused_bmm.py` | 保持主计算 tile，直接加载原始布局并在 dot 前做 native RNE，移除两块 scratch 和转换 kernel |
| `layout_inline_bmm.py` | 保留连续布局拷贝，在 dot 前做 RNE，用于分离布局和舍入位置的影响 |

所有版本保留现有 dtype、工作量门槛、窄输出回退与 tile 规则；新增 shape 规则为 0。`bmm` 和 `bmm_out` 共用修改后的路径。确定性算法和未初始化内存填充保持开启。不存在跨调用数据缓存。

源码 SHA256 在 [manifest.json](manifest.json)。每个补丁独立应用于 `baseline_bmm.py`。真实激活、布局/out 检查、模型、训练和画像脚本及结论统一放在 [Protenix 实验目录](../../../Protenix/comparison/ppu_tf32_precision_20260923/bmm_prepare_20260926/)。

实验完成，保留上一轮基线。单缓冲区在七个真实 BMM 中有一项降低 18.33%，G02/base 与 G02/v2 的整模没有可靠收益；全融合两项降低约 37%–38%，也出现约 3.93 倍退化，未扩大整模验证。六个版本的真实激活和布局前向检查均逐位一致；单缓冲区的两模型及四组训练探针也与同精度基线一致。完整结果见 [实验报告](../../../Protenix/comparison/ppu_tf32_precision_20260923/bmm_prepare_20260926/BMM_PREPARE_20260926.md)。
