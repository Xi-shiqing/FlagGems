# 非连续梯度直接读取：未接入的候选与原因（2026-10-24）

融合v2已通过完整训练恒等；本候选v3进一步尝试省掉上游梯度contiguous复制，使用任意rank的shape/stride通用地址计算，不设形状或输入白名单。真实MSA主检查失败，未启动模型训练，也未测报性能。

真实上游梯度shape=[600,600,8]，stride=[600,1,360000]。独立地址读取和舍入乘积逐字节相同，连续梯度分支的完整反向也逐字节相同；直接非连续分支的完整反向没有保持。

|检查|不同存储元素|最大绝对差|相对L2（比值）|
|---|---:|---:|---:|
|gathered_input|0|0|0|
|rounded_product|0|0|0|
|contiguous_backward|0|0|0|
|strided_backward|915566|3.63797880709e-12|3.82674769355e-08|

编译IR显示：连续与原版采用threadsPerWarp=[1,32]、warpsPerCTA=[4,1]、order=[1,0]的归约布局；非连续版本新增threadsPerWarp=[32,1]、warpsPerCTA=[1,4]、order=[0,1]，乘积和后续累积使用新增布局。结合相同读取/乘积与连续分支阳性控制，支持地址变化引发编译归约布局及浮点加法分组改变的解释。这里只是单算子因果诊断，不是对全部H100训练残差的唯一解释。

完整反向有915566/2880000个存储元素不同，最大3.63798e-12；不能因差异小而把既定逐位门槛改成容差。该候选不作为训练或性能改进成果。

诊断初版在数字计算后导出IR时误传新kernel参数给原版kernel而失败；第一版包装器又因JIT源行定位失败，均保留。最终独立源码诊断exit0，数字与IR完整保存。没有重跑任何模型训练、推理或H100。

[原主检查失败](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/fused_softmax_backward_v3_checks_results_20261024.json)；[完整标量及IR哈希](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/strided_v3_and_registered_v2_complete_or_progress_20261024.json)；[归约IR关键行](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/fused_backward_registered_v2_final_and_v3_reduction_IR_20261024.json)。

补充：v4只改变inline-asm输入的布局传播顺序，指令操作数顺序仍保持g*p；同一真实VJP再次失败，未计时、未训练，已否决该写法。不能靠简单交换输入顺序承诺固定编译归约布局。后续若继续去复制，需要显式控制加载转换与归约布局并重新验证，不能降格逐位门槛。

[该补充主检查失败](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/fused_softmax_backward_v4_checks_complete_or_progress_20261024.json)。
