# 主干真实矩阵与通用长归约修补：完成结果（2026-10-25）

本轮补充了主PairFormer/MSA的57份代表性完整乘积参考，并实现一项隔离的FlagGems MM调度改动。它改善了长归约的数学精度及独立API耗时；实际训练FP32小幅改善，TF32对旧H100的完整梯度和更新差异均略增。**不把局部收益写成TF32训练修复；当前默认不变，TF32不扩大百步。**

## 真实训练：四路径、新旧完整列出

全部为同一共同几何两步诊断，第0步lr=0但Adam历史照常更新，第1步更新前lr=1.8e-5。误差是全部4170份裁剪前g1、4174份实际参数更新，对同精度旧H100完整向量的相对L2百分比。不是逐参数误差均值，不裁小梯度。H100仅复用原PPU驻留缓存，没有新H100执行。

|平台／精度／版本|第0步loss|第1步loss|完整g1差异 %|实际更新差异 %|
|---|---:|---:|---:|---:|
|H100 原生 FP32，旧缓存|1.879856706|4.581809044|同精度参考|同精度参考|
|PPU FlagGems 当前 FP32|1.879857898|4.581814766|0.001540360|0.042941142|
|PPU FlagGems 长归约候选 FP32|1.879857898|4.581814766|0.001507257|0.042883171|
|H100 原生 TF32，旧缓存|1.882853270|4.579020500|同精度参考|同精度参考|
|PPU FlagGems 当前 TF32|1.883392215|4.578239441|1.289788063|6.433917284|
|PPU FlagGems 长归约候选 TF32|1.883392215|4.578239441|1.293299895|6.439156441|

FP32的g1差异相对下降2.1490%，更新差异相对下降0.1350%；TF32分别相对增加0.2723%、0.0814%。这是两个实际batch、一个seed的pilot，不是跨输入鲁棒性或checkpoint质量结论。没有新增五置信度、RMSD验收通过或训练MFU结论。

## 改了什么，为什么值得试

只修改私有FlagGems包flag_gems/ops/mm.py中的_ppu_h100_splitk_parts。1371份Python文件中只这一份改变；所有Triton内核AST、输入RNE舍入、乘积精度和短K策略均不变。复用现有split-K内核，没有另写一个新内核，没有使用TF32×3或IEEE替代普通TF32。BMM、TriAttention及既有IEEE回退保持原实现。

原调度对K>8192返回None，长归约走较长的累加链。新规则对K>8192采用min(ceil(K/8192),128,128MiB/(M×N×4))份子乘积；workspace不足两份时保留原路线。8192沿用原短归约界限，不按模型、输入、层、步骤或精确shape名单选择。预算和份数上限可能使子归约仍长于8192，因此不宣称所有输入都有固定误差上界。792项CPU策略检查核验短K不变及workspace上限。

该改动缩短子乘积的累加链，并增加可以并行执行的工作；代价是部分和workspace、额外写读和归并启动。它有数值和占用率动机，但不能保证更接近H100自身的有限精度结果。局部残差包含MMA/累加方式及FP32输出舍入，不能全部简单称为一个实现bug。

## 数学参考和单次API耗时：全部非零受影响案例

下表都是同一份实际操作数。FP32误差对未量化CPU64；TF32误差对相同输入RNE量化后的CPU64，衡量量化之后的残差。两种CPU64都是完整矩阵，不抽样，不是真实数域的精确oracle。时间是一个完整Gems MM API调用的三次warm device-event中位数，含输出/partial分配；不是裸内核时间、原训练中逐次耗时、train_step或模型加速。H100的这些主干局部输入没有旧切点，两精度误差/耗时均未测，不能补造。

|实际案例|PPU FP32数学误差旧→新 %|PPU TF32量化后残差旧→新 %|PPU FP32 API旧→新 ms|PPU TF32 API旧→新 ms|H100 FP32／TF32局部|
|---|---:|---:|---:|---:|---|
|主PairFormer，step0，256×360000×1024|0.000644819 → 0.000113835|0.061322585 → 0.002399520|22.038 → 8.236|18.001 → 9.623|未测／未测|
|MSA，step0，256×360000×1024|0.000961686 → 0.000100737|0.116636849 → 0.002932812|22.035 → 8.202|18.009 → 9.435|未测／未测|
|主PairFormer，step1，256×360000×1024|0.000628918 → 0.000153704|0.039598533 → 0.002910033|22.041 → 8.194|18.011 → 9.528|未测／未测|
|MSA，step1，256×360000×1024|0.000932623 → 0.000123791|0.098455516 → 0.002723476|22.062 → 8.188|18.045 → 9.453|未测／未测|

另两份K=1228800的受影响MSA矩阵，两精度参考与输出均为零，旧→新误差都是0，不计作改善。51份非目标矩阵原精度输出逐位恢复；57份矩阵×两种PPU精度×两种实现的完整误差和耗时均保留在CSV/JSON。57份选择中有49份TF32_RNE和8份UNCLASSIFIED_ADDMM，没有选到IEEE_FP32回退案例。代码支持已有fallback分类，但本次集合没有验证其实际执行；ADDMM保留两种CPU64参考，不把它的残差归因为已确认的TF32累加误差。所有矩阵来自同一模型的两个实际步骤，不能当57个独立模型或输入。

## TF32为什么没有传到整网H100一致性

TF32候选实际两步的完整diffusion输入、输出、独立输出上游、布局和CPU/CUDA/NumPy/Python/统一wrapper RNG状态逐位等于当前基线，17项loss记录相同。当前捕获边界中的s_inputs/s_trunk/z_trunk也全部包含；没有重演上轮mini匹配翻转。FP32也核验全部loss/raw/初始化/RNG/LR；没有旧FP32完整边界，所以不声称其所有内部张量都逐位不变。

本轮改变的是给定实际训练状态下的部分反向累加结果；P更接近该操作数的CPU64，不等于P更接近使用不同内部状态/算术的H。用原全向量平方和记e=P−H、d=Pnew−P，有||e+d||²=||e||²+||d||²+2<e,d>。下表由保存的全部向量标量作代数计算；不构造混合梯度，不按H距离选层或调插值系数。

|精度／对象|改动范数／原跨H差异范数|改动朝向H修正方向余弦|
|---|---:|---:|
|FP32／g1|0.212378|0.206291|
|FP32／实际更新|0.094146|0.061403|
|TF32／g1|0.019559|-0.129616|
|TF32／实际更新|0.024605|-0.020807|

TF32这次修改对完整梯度的变化量只在原跨H差异约2%的量级，且朝H的投影为负；不能指望这次修补消除主缺口。这只说明本次干预的大小与方向，不是对所有长归约方案的普遍上界，也没有把全部剩余误差归到某个组件。H100主干局部输入/第0步历史缺失，不能凭本结果认定H或PPU哪一侧整体错误。

## 组件结果与后续决定

|精度／模块|g1对同精度旧H误差旧→新 %|实际更新对同精度旧H误差旧→新 %|
|---|---:|---:|
|FP32／msa_module|0.001565971 → 0.001455601|0.056994394 → 0.056923903|
|FP32／pairformer_stack|0.001207524 → 0.001087684|0.065634134 → 0.065537155|
|FP32／diffusion_module|0.001643226 → 0.001634566|0.012489692 → 0.012488344|
|FP32／confidence_head|0.020512276 → 0.002253224|0.274088780 → 0.274083798|
|TF32／msa_module|3.831748033 → 3.835572371|11.902913069 → 11.908740581|
|TF32／pairformer_stack|1.742381057 → 1.751023521|8.779174372 → 8.788905684|
|TF32／diffusion_module|0.733751518 → 0.734062922|3.857302517 → 3.857362108|
|TF32／confidence_head|1.403622080 → 1.399140170|8.197565233 → 8.197409732|

1. FP32保留为有数学、单API和短训正证据的隔离候选，当前小幅整网收益还不足以声称通过多输入训练质量或正式性能验收。后续晋级需独立重复、原配方checkpoint及多输入质量，不能只看局部加速。
2. TF32保留本次数学/性能事实与负结果，不按模块挑选H距离较好的部分拼接，也不扩大百步或更换默认。继续优化前要先解释主干真实前向条件及量化误差传播；矩阵诊断不覆盖融合TriAttention内部dot，缺少H主干切点仍是边界。
3. checkpoint质量仍按原五置信度各5%、同索引五样本全原子proper-Kabsch RMSD≤0.1Å验收；没有改变门槛、监督、匹配、Adam epsilon或小梯度处理。FP32／TF32及PPU／H100四路径继续分开报告。

## 完成范围、失败与文件

原冻结报告的6步之外，另补FP32独立重复2步，本轮最终共8个实际训练步：当前TF32观察控制2步＋候选FP32两组各2步＋候选TF32两步。16项合成公共API检查是8组未见ragged/strided布局×两精度，覆盖K=8192/8193边界、M=1、N>512、非连续输入、strided out guard、完整重复位一致；不算实际模型步。固定矩阵重放和CPU64参考也不算训练步。默认、原包和原结果全部保留，无提交/push。

初版独立重放在provider来源断言处失败、0矩阵结果，旧依赖join也exit1，均保留。v2按既有的editable-finder移除/私有路径导入方式恢复，不重跑模型或CPU64；57份原模式输出全部逐位恢复后才解释比较。raw操作数、参考输出、梯度、坐标、权重仍仅在原PPU；本地只有own源码、标量、计数及SHA。

- [全57矩阵CSV](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/longk_bounded_all57_matrix_results_20261025.csv)／原矩阵完整标量（本地记录）／[候选矩阵完整标量](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/longk_bounded_fixed_matrix_results_20261025.json)
- [两精度完整训练/边界/16项检查](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/longk_bounded_actual_training_results_20261025.json)／[全向量代数解释及来源](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/longk_bounded_result_interpretation_20261025.json)
- [通用选取规则源码](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/longk_bounded_split_policy_20261025.py)／[单文件、单函数构建证明](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/longk_bounded_build_manifest_20261025.json)／[分析前协议](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/trunk_matrix_experiment_plan_20261025.json)

## 未见布局测试的全部数值退步

更短的累加链并不保证每个有限输入都更接近参考。16项公共API检查中以下5项误差上升，仍低于原单位检查阈值（FP32 0.002%、TF32 0.005%）；不计作精度改善，也不构成全输入鲁棒性通过。

|M×N×K|输入非连续|模式|旧误差 %|新误差 %|
|---|---|---|---:|---:|
|17×33×8193|False|FP32|0.000204100546|0.000225367983|
|17×33×8193|True|FP32|0.000420504571|0.000431889692|
|33×65×16385|True|FP32|0.000285159657|0.000289169156|
|1×65×32769|False|FP32|0.000391100069|0.000412649284|
|1×65×32769|False|TF32|0.001457820073|0.003121606333|

## 文件审查与执行边界

[新鲜文件审查](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/experiment_integrity_trunk_longk_20261025/EXPERIMENT_AUDIT.md)结论WARN、same-family/provisional。60份冻结输入及8,411项本地核对一致，包括114行CSV／2,166单元、92项报告数字、792策略检查、16项合成检查和96个选择槽位。核对数不是独立实验次数。原冻结报告的IEEE fallback覆盖措辞已在本FINAL纠正；原文件和审查输入不改。

原始张量、完整远端依赖和1371文件构建范围未由本地审查者独立重算/认证。内部旧训练收据有COMPLETE先于末尾断言的情况，本轮以最终exit和嵌套边界/分析收据核验，不用单个COMPLETE字段判科学通过。同卡GPU0的旧/新API元数据及FP32 subnormal→FP64默认提升见[补充元数据](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/trunk_longk_runtime_supplement_20261025.json)，它是执行者postrun证明，未提高审查独立等级。

## FP32独立进程、另一张同型号卡的完整重复

FP32候选r0在GPU1，新增r1在GPU0；均按同种子与同两份raw batch的原两步配方。全部g0、Adam0 step/m/v、g1、4174实际更新及两步完整diffusion边界逐位一致；边界128份张量、479010742元素纳入，无抽样。三份原始快照／状态／边界文件SHA也分别相同。它验证同配方可重复，不代表跨seed、跨输入或跨架构鲁棒性。FP32相对当前旧基线仍只有完整loss/raw/init/RNG审核，没有旧FP32完整边界，重复一致不能补造这一缺失。

[重复完整收据](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/longk_fp32_repeat_final_20261025.json)／[逐位核验程序](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/ppu_full_acceptance_20261005/precision_residual_20261019/verify_longk_fp32_training_repeat_20261025.py)。原60输入及主审查保持冻结；本补充另行核对，不替代原五置信度/RMSD或正式性能验收。
