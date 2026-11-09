---
title: "减少推理输出分配开销"
weight: 40
---

逐元素运算和矩阵乘法通常会完整写入输出。在这些调用中，输出使用前的额外初始化会增加内核调用和内存写入。可选的 C++ 分配入口直接使用 PyTorch 原有的 CUDA 内存分配器，减少这部分开销，同时保留原来的计算内核。

当前接入范围包括逐元素运算、PPU MM/BMM/addmm，以及乘法、clone、cumsum、to_copy、sum、mean、any 和 new_ones 的输出分配。PPU MM/addmm 的分段部分和、MM 的右矩阵预处理，以及数学注意力路径的 Q/K/V 和概率数组也使用同样的分配方式。选择条件使用布局、类型、输出是否完整写入和自动求导状态；不依赖模型名称或输入编号。自动求导开启时使用原有分配方式。

## 构建和启用

在运行模型的同一个 PyTorch 环境中构建：

```bash
python tools/build_full_write_allocator.py --build-dir /path/to/writable/build
export FLAG_GEMS_NATIVE_FULL_WRITE=1
```

构建只使用现有 C++ 编译器和 PyTorch 头文件、库文件，不需要编译 GPU 内核。构建结果默认写入 `src/flag_gems/utils/`；可以用 `--output-dir` 指定安装位置。更换 PyTorch 版本后重新构建。默认不开启此选项。

## 接入约束

调用方必须保证输出的每个逻辑元素都由后续内核写入，包括边界块。需要读旧输出、依赖零初始化、使用原子累加到未初始化输出的计算，继续使用原有分配方式。

分配器保持每次输出独立，沿用 PyTorch 的设备、执行流和生命周期管理。稠密转置输出保留原步长；其他布局使用原有 `empty_like` 分配。全局确定性和未初始化内存检查开关不变。

clone 在进入自动求导包装之前检查调用方的梯度状态，保持训练时的原分配方式。cumsum 保留原来的逐段扫描与累加顺序，仅替换新建输出的分配；传入 `out` 时继续使用指定的输出。

`empty_for_overwrite` 用于已经核对完整写入的临时数组和输出。MM/addmm 的每个分段会写入自己的完整部分和，后续合并保留原来的顺序。注意力的打包内核完整写入 Q/K/V，softmax 完整写入概率数组。上述计算内核保持原实现。

MM 对空矩阵单独处理：输出没有元素时直接返回；求和维度为零时写入零，避免通用内核预取不存在的输入。已有 `out` 保留原存储和步长。

to_copy 保留 dtype、步长、独立存储及复杂数和设备转换的原有回退。转为 bool 时显式比较输入是否非零，保证非零小数、负值、大整数和 NaN 的转换符合原生语义。

## 求和、均值和逻辑归约

sum、mean 将输出和部分和数组接入分配器，保留原有分块、累加顺序、临时类型和除法。空输入的 sum 仍写入零，mean 仍写入 NaN。any 的多维原子合并路径继续使用明确置零的数组。new_ones 的输出由原有填充内核完整写入。

全张量 any 在 `next_power_of_2(numel) * max(element_size, 4) <= 4096` 时，直接使用第一阶段归约内核写出最终标量，省去部分和数组及第二次内核调用。预算为 4 KiB，并按至少 32 位寄存器计算较小的数据类型；较大输入保留两阶段。布尔 OR 精确且满足结合律，允许改变归约组织。该规则适用于符合资源预算的任意输入。

按维度 any 由归约内核直接读取原类型并判断是否非零，省去前置布尔转换和中间数组。全张量 any 与标量 sum.out 对非连续输入先整理布局；mean 显式传入 `dim=None` 时按全体元素求均值。

## lift_fresh 的别名语义

`lift_fresh` 接收的是构造过程中已经分配的张量，直接返回该输入，保留对象、存储和布局。它不需要再分配输出或启动复制内核。需要独立副本的接口是 `lift_fresh_copy`。

实现与 [PyTorch 2.10 的原生接口](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/TensorShape.cpp#L4616-L4627) 一致，适用于各种尺寸、步长和数据类型，不依赖输出分配开关。

## 验证

```bash
FLAG_GEMS_NATIVE_FULL_WRITE=1 pytest tests/test_full_write_allocation.py tests/test_extra_full_write_allocation.py tests/test_ppu_full_write_buffers.py -q
FLAG_GEMS_NATIVE_FULL_WRITE=1 pytest tests/test_ppu_copy_outputs.py tests/test_lift_fresh.py --ref=cpu -q
FLAG_GEMS_NATIVE_FULL_WRITE=1 pytest tests/test_full_write_reductions.py -q
```

测试覆盖完整写入、非连续输入、广播、空张量、标量、多种类型、独立存储、非默认执行流和自动求导回退。完整写入检查会先把输出填入 NaN 或整数标记，再与 CPU 参考和原分配路径比较。

归约检查另覆盖单线程块规则两侧的尺寸、NaN、无穷值、次正规值、非连续 GPU 视图和原有接口问题。分配切换要求逐位一致，逻辑归约与原生 CPU 结果精确比较。

分配优化前后要求逐位一致。MM 的 CPU 参考使用 FP64 累加；TF32 先在 CPU 上独立执行 RNE 输入舍入。CPU 数值验收沿用现有 MM 测试中按求和长度计算的容差，并另外保存相对严格 FP32 的误差和固定容差诊断结果。

性能验收使用完整模型调用，保持输入、随机数、计算精度、循环次数、扩散步数和输出样本数相同；在同卡上交换运行顺序，并比较结构坐标与置信度。
