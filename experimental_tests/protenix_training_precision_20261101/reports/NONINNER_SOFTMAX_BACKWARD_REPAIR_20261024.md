# 非末轴Softmax反向：确定性修补与完整训练验证（2026-10-24）

这项修补补齐一个配置选择缺口：先前只固定末轴反向，非末轴反向仍按计时挑选归约配置。单个真实MSA切片已观察到原调优对5个合法配置计时，配置变化会改变梯度。两次自然观测都选32；本轮没有观察到自然训练切到另一个配置，不把强制配置对照冒充自然漂移，也不将它解释为全部PPU/H100残差的根因。

候选复用原内核、公式和原TILE_N/ONE_TILE启发式，只将PPU非末轴反向的TILE_K设为min(1024,max(32,next_power_of_two(inner_extent)))，固定4 warps/2 stages，绕过计时选择。该规则按轴布局与资源预算统一生效，没有模型、输入、精确shape或训练步名单。矩阵乘精度与整个前向保持原样。

## 真正FlagGems源码与实际入口

隔离包1371个Python文件仅flag_gems/ops/softmax.py不同；所有前向与内核函数AST不变，只有反向配置调度函数变化和新增无调优入口。默认包未替换。真实注册ATen/autograd检查已完成：FP32/TF32模型标志下两个真实完整VJP与原型逐位相同；另21个dtype/轴/布局检查覆盖float32、float16和bfloat16，原前向逐位相同，CPU64容差通过，重复逐位相同，原计时调优入口调用0。

首次私有包数值检查末尾因缓存元数据为tuple而导出失败，失败保留；v2只修导出并先保存数值结果，完整exit0。两个完整模型的第一版私有包启动因实际shell仍选旧包被源代码守卫在模型构造前拦住，0训练步，未作为通过；v2独立脚本显式切包并保存实际shell，FP32/TF32各两步集成已完整通过。

## 六组真实训练、十二个步骤已完成审核

使用原共同几何两步配方，一个模型、一种子、两个实际batch；全部资产、4174初始参数、3338条有序统一RNG、LR0及1.8e-5通过。每组保存完整第0步裁剪前梯度和实际Adam状态、第1步完整梯度和实际更新。此处不与原配方百步末态混比。H100全部为旧缓存，没有新增H100训练或推理。

误差分别对同训练精度旧H100，完整4170参数梯度／4174参数实际更新，不剔除小梯度。

|平台／训练精度／规则|第0步loss|第1步loss|完整梯度相对L2差%|实际更新相对L2差%|
|---|---:|---:|---:|---:|
|H100原生 FP32 旧参考|1.879856706|4.581809044|参考|参考|
|PPU当前FlagGems FP32|1.879857898|4.581814766|0.001540359963|0.042941142163|
|PPU固定非末轴反向 FP32|1.879857898|4.581814766|0.001540359963|0.042941142163|
|H100原生 TF32 旧参考|1.882853270|4.579020500|参考|参考|
|PPU当前FlagGems TF32|1.883392215|4.578239441|1.289788062557|6.433917284226|
|PPU固定非末轴反向 TF32|1.883392215|4.578239441|1.289788062557|6.433917284226|
|PPU TF32 强制合法配置64（诊断）|1.883392215|4.578239441|1.289849843704|6.434035832172|

固定规则在这组真实轨迹中保持当前结果：两种精度的第0步完整梯度/Adam两阶矩不变，TF32两个独立重复的第0步状态、第1步完整梯度和更新逐位一致。对H100的误差没有降低。此项成果是消除计时选择可能引入的额外变化，不能包装成跨设备训练精度提升。

## 为什么loss相同还不能说明训练相同

强制合法配置64是预先标记的数值干预，不是候选，也不是自然漂移。两个总loss与原路径逐位相同，所有实际输入/RNG/初始权重/LR一致，但完整反向和真实更新已改变：

|强制配置64／原PPU路径|相对L2差%|不同元素数|最大绝对差|
|---|---:|---:|---:|
|第0步裁剪前梯度|0.000518775059|4422737|4.57698479295e-06|
|第0步Adam一阶矩|0.000518780259|4322615|4.57745045424e-07|
|第0步Adam二阶矩|0.000017401222|4093977|3.23780113831e-09|
|第1步完整梯度|0.001861077754|4448419|0.000254657119513|
|实际参数更新|0.055118264268|971427|2.48923897743e-05|

这是独立反向数值差，经链式法则和Adam历史传播到实际更新的受控证据；不需要前向总loss先出现差异。表中各项不是独立因果份额，不能相加，也不从对H100的主误差中扣除。

## 私有源码包的整模型集成

另有FP32/TF32各两步，直接导入隔离FlagGems包，用只读ATen观察器记录；没有进程内替换内核。须逐位匹配上述固定规则的完整g0/Adam0/g1/update，才计为源码包整模型验证通过。FP32/TF32两组均已exit0，CPU完整恒等审核exit0：g0、Adam0两阶矩、g1与实际更新均完整相同，总loss及所有loss分项记录也相同。本轮共8组、16个实际步骤；初版两组入口失败为0步，不计完成数。

## 原前向候选与主验收

前一轮四输入同H100权重部署已全部收齐：旧路径与新空间前向均12/16通过，G02的TF32 checkpoint/TF32部署RMSD反而0.896882→1.316113 Å。旧前向候选不推广。本反向修补没有叠加旧前向候选；原百步新checkpoint主验收0/8结论保留，5%置信度与0.1 Å同索引全原子RMSD门槛不变。没有九模型、百步验收、速度或MFU新通过声明。

[源码补丁](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/noninner_backward_canonical_flaggems_20261024.patch)；[单文件差异清单](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/noninner_backward_isolated_gems_build_receipt_20261024.json)；[实际注册入口检查](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/noninner_backward_private_gems_package_v2_result_20261024.json)；[六组完整训练标量](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/noninner_backward_six_control_and_private_v2_results_20261024.json)；[旧前向四输入完整表](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/FIXED_CHECKPOINT_SOFTMAX_FOUR_INPUTS_20261024.md)。

补充闭环：[16实际步骤与源码包完整验收](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/noninner_backward_repair_complete_acceptance_20261024.json)。只补齐重复性入口，对H100的主误差没有下降。
