---
title: "减少推理输出分配开销"
weight: 40
---

逐元素运算和矩阵乘法通常会完整写入输出。在这些调用中，输出使用前的额外初始化会增加内核调用和内存写入。可选的 C++ 分配入口直接使用 PyTorch 原有的 CUDA 内存分配器，减少这部分开销，同时保留原来的计算内核。

当前接入范围包括逐元素运算、直接执行的 PPU MM、PPU BMM，以及乘法、clone 和 cumsum 的输出分配。选择条件使用布局、类型、输出是否完整写入和自动求导状态；不依赖模型名称或输入编号。自动求导开启时使用原有分配方式。

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

## 验证

```bash
FLAG_GEMS_NATIVE_FULL_WRITE=1 pytest tests/test_full_write_allocation.py tests/test_extra_full_write_allocation.py -q
```

测试覆盖完整写入、非连续输入、广播、空张量、标量、多种类型、独立存储、非默认执行流和自动求导回退。完整写入检查会先把输出填入 NaN 或整数标记，再与 CPU 参考和原分配路径比较。

性能验收使用完整模型调用，保持输入、随机数、计算精度、循环次数、扩散步数和输出样本数相同；在同卡上交换运行顺序，并比较结构坐标与置信度。
