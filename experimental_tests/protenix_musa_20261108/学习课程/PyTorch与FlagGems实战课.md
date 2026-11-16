# PyTorch 与 FlagGems 实战课

用 7bca 做案例，学会从一行 PyTorch 调用追到真正执行的算子，然后把一个优化变成能复现、能核验的实验。建议用 60–90 分钟，先在 CPU 跑通练习，再到已经配置好的实验 GPU 上练习。全文只讲实现和调试。

## 1. 在十分钟内看见分派

在装有 PyTorch 的环境里运行 [配套练习脚本](pytorch_flaggems_lab.py)。

```bash
python -B pytorch_flaggems_lab.py --device cpu --repeats 4
```

先读输出中的 `dispatch`。脚本用 `torch.library.Library` 声明 `protenix_lesson::shift_relu` 的参数和返回值，再注册一个 CPU 实现。调用 `torch.ops.protenix_lesson.shift_relu(x, 1.0)` 时，PyTorch 按输入设备等信息选实现。你应该看到调用计数为 1，结果为 `[0.0, 1.0, 4.0]`，Profiler 同时能看见外层自定义算子以及里面调用的 ATen 算子。

打开代码，找到这三处。

| 代码 | 职责 | 实战中容易踩的坑 |
|---|---|---|
| `Library(..., "DEF")` 和 `define(...)` | 定义命名空间和调用契约 | 同一命名空间只应有一个 DEF 所有者。就地修改和别名关系也属于契约 |
| `impl(..., "CPU")` | 将一个实现注册到对应 dispatch key | CPU 注册只覆盖 CPU。MUSA、CUDA、autograd 都需要对应的支持 |
| `torch.ops...` | 通过 PyTorch 分派调用 | 直接调用 Python 函数会绕开外层自定义算子的分派 |

练习里的注册对象要活到调用结束。把注册写在函数里却不保留 `Library`，可能在对象释放后丢失注册。这里的 CPU 小算子只用于前向教学。要把自定义算子交给训练或 `torch.compile`，还要按用途补反向和 FakeTensor/形状推导支持。 [PyTorch torch.library 文档](https://docs.pytorch.org/docs/2.15/library.html)

**动手**。把 CPU 实现里的 `calls += 1` 改为记录输入设备。在另一个终端分别运行“调用 `torch.ops`”和“直接调用 Python 实现”的版本，观察结果虽然相同，Profiler 的外层算子名不同。

项目里还有一个就地算子 `floor_counts_(Tensor(a!) count) -> Tensor(a!)`。输入和输出标成同一个 `a`，感叹号声明输入会被改写。读实现，找出为什么把 `count[count < 1] = 1` 改成 `clamp_min_` 仍然保留了计数下限，却省掉了需要构造变长索引的路径。这个例子适合练习把 PyTorch 语义与内核执行方式分开看。 [本轮计数下限实现](../code/scatter_floor.py)

## 2. 从模型行号查到 FlagGems 实现

```text
模型里的 nn.Linear / torch.bmm
        ↓
具体 ATen 算子与 overload
        ↓
PyTorch 按设备与执行上下文分派
        ↓
FlagGems 已注册的实现
        ↓
实际启动的 Triton kernel
```

读模型先查调用处。比如 `nn.Linear.forward` 会走线性层接口，最终命中哪条 ATen 路径要看实际输入和注册。看到 `torch.matmul` 也先查 trace。批维广播、维度或 layout 可能让它走不同分解。遇到 `mul.Tensor`、`mul_.Tensor` 和 `mul.Scalar`，逐一核对它们的签名和语义，尤其注意就地更新。

FlagGems 的注册范围可以限制到具体算子。先在进程里打印版本和路径，确认 Python 导入的是你改过的 checkout。这个检查能发现最常见的误判之一：跑到了 site-packages 里的另一份 FlagGems。

```python
import inspect
import torch
import flag_gems

print(torch.__version__, inspect.getfile(flag_gems))
print(inspect.signature(flag_gems.use_gems))

with flag_gems.use_gems(include=["bmm"]):
    output = torch.bmm(left, right)  # left、right 已在已核对的空闲设备上
```

项目脚本分两层记证据。注册记录里有键和函数来源，执行记录里有完成次数、对应输入的 shape/stride/dtype，以及底层 LibEntry 的调用。7bca 的 110 组合在一次完整追踪中实际完成 **5,092 次 BMM** 和 **7,567 次 SiLU×Mul**。这是可以沿代码复核的证据。只看到 `with use_gems()` 或安装成功，还缺实际执行证据。 [FlagGems 上游](https://github.com/flagos-ai/FlagGems) · [本项目 110 配方](../data/前期组合对照.md)

**动手**。在原始输出的 `execution_audit.json` 里分别找 `completed_function_calls` 和底层启动记录。再在 `repository_integration.json` 查 BMM 的代码文件。在这两个文件之间追一遍，比只看一张 profiler 汇总表更容易判断有没有跑到目标实现。

## 3. 将正确性检查写在计时前面

练习里的 BMM 有两份值相同的右操作数。一份是转置视图，一份是连续副本。输入特意用了可精确表示的小整数和非整块尺寸，所以可以用 CPU FP64 算参考，再以零容差比较这个夹具。真实模型的浮点输入应按对应算子和精度的验收规则比较，教学里的零容差只适用于这个夹具。

新算子先挑一个**真实出现的输入形状和 stride**，再加非连续、尾块、空输入和广播。对有 mask 的注意力另查全 mask。遇到以下三类变化，把它们当作独立实验。

- 改 `view`、`reshape`、`contiguous`。前两者能否共享存储取决于 stride，`contiguous()` 在布局已经满足时可能直接返回原对象。仅凭源码一行不能断定发生了多少次拷贝。
- 把 FP32 改成 TF32、BF16 或 TF32×3。记录实际内核路径和精度，按对应精度列重新计算 MFU，并独立验收。
- 将两个 Linear 合并成一个。先核对两路是否读取同一个输入，权重、bias、输出分段是否一致。并排输出的 stride 也可能改变下游内核选路，必须检查完整模块。

整模型再做三层核对。第一层固定权重、输入特征和推理预算。第二层固定采样真正用到的随机流，包括 MSA 抽样和逐步扩散噪声。第三层比较每个对应候选的全复合物坐标和五项置信度。只看 rank 0 容易漏掉异常。项目使用的结构门槛是 Kabsch RMSD 0.1 Å，五项置信度还要核对原作者的容差。 [本次精度检查](../张石健20261108实验报告.md#精度)

## 4. 把“缓存”写成一个可审计的改动

扩散每一步都会更新坐标和噪声等级。一份已经算好的 pair 表征或静态条件特征则可能被各步重复读取。想复用它，先沿调用链把依赖找全。

| 要检查 | 7bca 中怎么查 |
|---|---|
| 数据从哪里来 | 从 `DiffusionModule.f_forward` 跟到 `DiffusionConditioning` 和 `AtomAttentionEncoder` |
| 哪些值逐步变化 | 把 noisy positions 和 timestep 留在每步计算 |
| 谁会原地写 | 沿消费者查 `+=`、`copy_`、切片赋值等。对要复用的输出核对版本计数 |
| 缓存何时清空 | 每次 `predict` 建立新作用域，结束或抛异常时都清空 |
| 缓存有没有命中 | 按次记录 hit/miss，关联注册的 pair bias 算子和预测结果 |

不能把“地址相同”单独当作缓存依据。分配器可能复用已经释放的地址，两个 view 可能共享 storage 却读不同区域。同一个对象也可能被原地改写。本项目候选把地址、shape、stride、设备、dtype 和版本一起核对，并持有输入引用，限制在单次预测的只读条件路径。跨批次复用还需要重新证明输入和权重的生命周期。

**动手**。读 [扩散复用候选](../code/diffusion_reuse.py) 的 `prediction` 和 `memo`，指出哪一行防止上一次预测的值泄漏到下一次。再画出“缓存输入 → pair bias → 注意力”的数据流，并圈出不能随缓存跳过的坐标更新。

## 5. GPU 计时的三种视角

| 读数 | 适合回答什么 |
|---|---|
| 两端同步的 wall time | 用户等到一次预测完成需要多久 |
| 同 stream 的两个 GPU Event | 一段提交给该 stream 的工作的经过时间 |
| Profiler trace | 实际发起了什么、哪里出现了拷贝或同步、阶段是否重叠 |

GPU 执行是异步的。Python 函数返回时工作可能还在队列里，所以 wall time 应明确两侧的同步边界。Profiler 要单独跑，因为记录事件、shape 或 stack 会改变开销。嵌套区域的时间会重叠，例如 diffusion 大区域已经包含 atom encoder，不能把这两项直接相加。 [GPU Event 文档](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.Event.html) · [PyTorch Profiler 文档](https://docs.pytorch.org/docs/2.15/profiler.html)

配套脚本在已配置 FlagGems 的实验 GPU 上也能运行。先确认设备空闲，并用平台的可见设备环境变量只暴露目标卡。

```bash
python -B pytorch_flaggems_lab.py --device musa --warmup 5 --repeats 12
# 在独立诊断调用中增加 FlagGems 日志
python -B pytorch_flaggems_lab.py --device musa --warmup 5 --repeats 12 --gems-log bmm_dispatch.log
```

观察两种 stride 的原始重复值。脚本把拷贝放在计时外，这里回答的是“输入已经准备好，哪种布局的 BMM 更快”。如果真实路径每一步都要做 `contiguous()`，做第二版实验，把拷贝挪到计时内，并保留两版结果。把预热和编译从稳态样本中分开，同一卡交替 A/B，保存每次样本、中位数和波动，单次最快值容易被环境噪声影响。

**动手**。将 `cases` 中的连续副本改成每次运行前生成，再把这一步分别放到计时区间的内外。解释为什么得到的两个结论可以同时成立。

一次实际排障也能说明计时应该看什么。7bca 的广播乘法每次为相同布局构造一份很小的 shape/stride 张量，然后从主机上传到设备。图捕获在这个拷贝上报错。把这份只读元数据按设备、流和布局缓存后，计算仍走同一个 FlagGems 内核，seed 101 的完整热态预测从 119.49 秒降到 112.03 秒。读 trace 时，除了看大 GEMM，也要检查反复出现的拷贝、标量读取和同步。 [本轮广播乘法元数据缓存代码](../code/mul_metadata.py)

## 6. 从“算子快了”到“模型值得改”

这次 7bca 的旧 110 配方平均约 122.45 秒。更新后的 H100 原生平均约 19.64 秒。严格 FP32 采用项目给定峰值 H100 67、S5000 29 TFLOPS。

```text
原始吞吐比 = H100 原生耗时 / MUSA FlagGems 耗时
相对 MFU = 原始吞吐比 × 67 / 29
MUSA 绝对 MFU = 标准有效模型 FLOPs / (MUSA 耗时 × 29 TFLOPS)
```

用这组参考，50% 相对 MFU 对应约 **90.74 秒**。若一个模块加速 2 倍、其余耗时完全相同，模块必须占旧整模型约 52% 才有机会填平差距。这个计算能帮助你先筛掉总占比很低的优化，也能解释为什么一个小 benchmark 提速明显，整模型曲线却几乎不动。

本项目 MFU 使用共同的有效模型计算量。缓存减少了设备实际重复执行的工作，所以另记命中次数和实际执行的 FLOPs（若有对应采集），以免把模型 MFU 和硬件指令吞吐混起来。GPU-Util 反映采样窗口里的忙碌程度，也应单独报告。 [7bca 实验数据](../data/7bca冲刺实测.json)

**最后一个练习**。写一份五行实验记录：改动和源码版本、设备和实际命中、对应输入与精度、计时边界及全部样本、可支持的结论。试着解释为什么“算子首轮耗时下降”“7bca 热态变快”“相对 MFU 达标”各自需要不同证据。
