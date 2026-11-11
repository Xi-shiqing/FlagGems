# PPU 短整数操作与布尔索引

本次改动通过 PPU 后端注册到现有 `sort`、`nonzero`、`index.Tensor` 入口。`unique` 使用的排序入口也会调用新的整数排序实现。

## 实现与选择条件

| 操作 | 实现 | 选择条件 |
| --- | --- | --- |
| 短整数排序 | 一个线程块完成稳定排序；相同数值按原下标排序 | bool、uint8、int8/16/32/64；最多 8 维；每条排序维度最多 4096 个元素 |
| nonzero | 同一个内核完成非零判断、整数前缀和及坐标生成 | 连续的 1–4 维实数、整数或布尔输入；总元素数最多 8192 |
| 布尔索引 | 前缀和直接确定输出位置，并在同一次内核调用中复制数据 | 1–4 维实数、整数或布尔输入；一个连续布尔掩码；掩码最多 4096 个元素；预分配输出空间最多 2²⁰ 个元素 |

布尔掩码可以覆盖连续的多个输入维度。输入支持转置、切片、广播步长和存储偏移，输出保持连续布局。其余条件使用原实现。选择规则使用张量元数据。

排序比较始终使用整数，覆盖 int64 的最大值、最小值和大于 2⁵³ 的数值。布尔索引通过同位宽整数指针复制数据位，保留 NaN、正负零和非规格化数的原始位。

`nonzero` 和布尔索引各保留一次读取输出数量的同步。布尔索引无需先生成一份下标数组再调用读取内核。选中的值写入紧凑输出前缀，剩余预分配空间写零；两个写入范围互不重叠。

新内核写入全部分配空间，复用现有的完整写入分配器。梯度模式下使用标准分配路径。

## 验证与复现

完整正确性检查包含 647 项测试，覆盖直接和注册接口、整数极值、重复值、空输入、多种布局、多维掩码、浮点特殊值及反向。

```bash
python -m pytest tests/test_ppu_short_integer_sort.py tests/test_ppu_small_nonzero.py tests/test_ppu_boolean_index.py -q -o addopts= -p no:cacheprovider
```

接口计时包含输出分配与调用开销，同时保留单次同步和 16 次批量调用的结果。计时使用预热后的四阶段对照，全部测量样本保留。

```bash
python benchmark/bench_ppu_short_integer_ops.py --output-root ./short_integer_output --label candidate --operators sort unique nonzero
python benchmark/bench_ppu_boolean_index.py --output-root ./boolean_index_output --label candidate
```

首次编译的耗时单独保存在实验记录中。大整数排序网络需要较多编译时间。

模型耗时、MFU、完整训练状态比较及精确运行配置保存在 [Protenix 实验目录](https://github.com/Xi-shiqing/Protenix-FlagOS/tree/master/comparison/small_integer_sort_20261111)。
