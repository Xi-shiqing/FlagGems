"""Build the merge report from completed, immutable scalar records."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
read = lambda name: json.loads((HERE / name).read_text())
train = {mode: read(f"training_comparison_{mode}.json") for mode in ("fp32", "tf32")}
suite = read("validation_restored_results.json")
precision = read("precision_routes_restored.json")
autograd = read("autograd_registration_restored.json")
publication = read("publication_source_manifest.json")
assert suite["status"] == "PASS" and suite["suite_files"] == 21
assert precision["status"] == "PASS_ALL_CURRENT_PRECISION_ROUTES"
assert len(autograd["rows"]) == 22 and all(row["status"] == "PASS" for row in autograd["rows"])

def error(mode, kind, comparison):
    return train[mode]["metrics"][kind][comparison]["total"]["relative_l2"] * 100

for mode, data in train.items():
    assert data["status"] == "COMPLETE_FULL_TRAINING_COMPARISON"
    assert data["uniform_RNG"]["count"] == 3338
    assert data["actual_learning_rate_step1"] == 1.8e-5
    assert data["metrics"]["gradients"]["new_vs_H100"]["total"]["elements"] == 464441841

summary = {
    "date": "2026-11-06",
    "status": "MERGE_VALIDATED_WITH_REPORTED_NUMERICAL_DIFFERENCES",
    "ours": "9583da7ab090e0f54498a23b2fe2e9a9e35abe7d",
    "upstream": "45339c4ad01d237d4fe66db197afb817eb9c6c60",
    "merge_base": "b550ddb996939ee1a30ba4a4b8f8b2d0556703a3",
    "upstream_commits": 1062,
    "upstream_changed_paths": 3816,
    "text_conflicts_resolved": 15,
    "source_conflict_diagnostic_cases": 50,
    "selected_suite": suite["counts"],
    "selected_suite_files": 21,
    "targeted_regression_cases": 26,
    "registered_autograd_cases": 22,
    "registered_precision_cases": 10,
    "tested_source_tree_sha256": publication["tested_tree_sha256"],
    "publication_source_tree_sha256": publication["tree_sha256"],
    "publication_diff": "One policy comment; executable AST unchanged.",
    "default_linear": "Native decomposition into FlagGems MM/addmm/BMM; whole-layer implementation retained for explicit use.",
    "cdist": "Restore deployed kernels with scoped AutogradCUDA and autograd.Function; native CPU autograd retained.",
    "new_H100_execution": False,
    "training_steps_per_precision": 2,
    "training_inputs_per_precision": 2,
    "training_seeds": [20260825],
    "training_timing_valid": False,
    "numerical_acceptance": "Exact old/new errors are reported; completion is not a numeric threshold and no global training-accuracy PASS is asserted.",
    "training": {},
}
for mode in train:
    summary["training"][mode] = {
        "losses": {key: train[mode][key] for key in ("new_losses", "old_PPU_losses", "H100_losses")},
        "gradient_relative_percent": {key: error(mode, "gradients", key) for key in ("old_vs_H100", "new_vs_H100", "new_vs_old_PPU")},
        "update_relative_percent": {key: error(mode, "updates", key) for key in ("old_vs_H100", "new_vs_H100", "new_vs_old_PPU")},
        "all_loss_components_equal_old_PPU": train[mode]["all_loss_components_equal_old_PPU"],
        "actual_batches_assets_initial_weights_learning_rate": train[mode]["actual_batches_assets_initial_weights_learning_rate"],
        "uniform_RNG": train[mode]["uniform_RNG"],
    }
(HERE / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

headers = ["指标", "PPU + FlagGems FP32", "H100 原生 FP32", "PPU + FlagGems TF32", "H100 原生 TF32"]
training_rows = []
for step in (0, 1):
    for label, key in (("合并前", "old_PPU_losses"), ("合并后", "new_losses")):
        training_rows.append([
            f"第 {step} 步 loss，{label}",
            f"{train['fp32'][key][step]['loss']:.9f}",
            f"{train['fp32']['H100_losses'][step]['loss']:.9f}",
            f"{train['tf32'][key][step]['loss']:.9f}",
            f"{train['tf32']['H100_losses'][step]['loss']:.9f}",
        ])
for label, kind in (("完整梯度差异", "gradients"), ("实际参数更新差异", "updates")):
    for stage, key in (("合并前", "old_vs_H100"), ("合并后", "new_vs_H100")):
        training_rows.append([f"{label}，{stage}", f"{error('fp32', kind, key):.6f}%", "0%（参考）", f"{error('tf32', kind, key):.6f}%", "0%（参考）"])
with (HERE / "training_comparison.csv").open("w") as stream:
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(training_rows)

old_precision = read("precision_routes_existing_control.json")
timing_map = lambda data: {(row["operator"], row["precision"]): row for row in data["cases"]}
old_times, new_times = timing_map(old_precision), timing_map(precision)
timing_rows = []
for name in ("mm", "addmm", "bmm", "linear", "softmax"):
    values = []
    for mode in ("fp32", "tf32"):
        before = old_times[name, mode]["median_ms"] * 1000
        after = new_times[name, mode]["median_ms"] * 1000
        values.extend([f"{before:.3f} → {after:.3f}", "—"])
    timing_rows.append([name, *values])
with (HERE / "registered_api_timings.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=["operator", "precision", "package", "shapes", "median_us", "samples_ms", "actual_ATen_calls"])
    writer.writeheader()
    for label, data in (("existing", old_precision), ("merged_restored", precision)):
        for row in data["cases"]:
            writer.writerow({"operator": row["operator"], "precision": row["precision"], "package": label,
                             "shapes": json.dumps(row["shapes"]), "median_us": row["median_ms"] * 1000,
                             "samples_ms": json.dumps(row["device_ms"]), "actual_ATen_calls": json.dumps(row.get("actual_ATen_calls", {}))})

def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *["| " + " | ".join(row) + " |" for row in rows]])

tf_grad = 100 * (1 - error("tf32", "gradients", "new_vs_H100") / error("tf32", "gradients", "old_vs_H100"))
tf_update = 100 * (1 - error("tf32", "updates", "new_vs_H100") / error("tf32", "updates", "old_vs_H100"))
report = f"""# FlagGems 上游合并与训练回归（2026-11-06）

上游固定到 `45339c4a`，本地起点为 `9583da7a`。合入共同祖先之后的 1,062 个上游提交、3,816 个变更路径，解决 15 个文本冲突。工作在独立 worktree 完成，原工作区保留。

## 实现如何取舍

| 算子 | 最终选择 | 依据 |
|---|---|---|
| Copy | 合入官方连续复制与类型转换，保留我们的可选跨步复制 | 官方修复 bool 转换；连续和转置布局比较通过 |
| Mean | 采用官方更新 | 修复空外层维度及非法 dtype 处理 |
| Div | 合并双方修复 | 保留 FP32 舍入、原地自动求导；加入官方标量截断修复 |
| Pad | 保留我们的自动求导 | 官方单独实现丢失反向；我们的前向与梯度检查通过 |
| Unfold 反向 | 保留我们的确定性实现 | FP64 正确，重复输出一致；重叠窗口计算更快 |
| LogSoftmax | 保留现有数值计算方式 | 六个前向／反向例子的参考误差更小；速度各有快慢 |
| Sigmoid、SiLU | 保留可选快速实现，合入官方接口整理 | 连续、非连续布局及前反向比较通过 |
| MM、addmm、BMM、Softmax | PPU FP32／RNE 使用现有已验证内核 | 与当前服务器实现核对来源，保留矩阵舍入和融合反向成果 |
| Linear | 默认沿用原生分解，底层调用 FlagGems MM/addmm/BMM | 保持已有模型计算组织；官方整层实现合入源码，修正布局、空维度及精度缓存问题，保留显式调用接口 |
| cdist 距离计算 | 恢复服务器已有的完整反向实现 | 保留五个内核的运算顺序；注册跟随 use_gems 上下文，仅接管 CUDA 键，CPU 保留原生反向 |

冲突源码比较包含 50 个用例，合并阶段实现通过各自检查。三版本使用同一运行时、相同输入，交替测量七轮。全部误差、环境开关和计时见 [逐项 CSV](conflict_source_cases.csv)。例如重叠 Unfold 反向：现有实现 0.3846 ms、官方 3.6551 ms、合并实现 0.3831 ms；这项复核保留了项目原有优势。

早期源码比较中名为 `*_tf32` 的 Linear 用例采用旧快速开关，官方分支仍为严格 FP32。最终 RNE 检查使用实际注册接口和独立 CPU 舍入参考，记录在 [实际入口数据](precision_routes_restored.json)；这些记录分别保存。

本次新增选择依据精度、数据类型、布局和工作量，未新增按模型、输入名称或精确 shape 选择的白名单。

## 合并验收中修复的问题

上游 PPU `index_select_backward` 的小工作量求和分支重复推进地址，32 索引用例结果错误；取消该覆盖，保留已有确定性实现。Square 的 CUDA 接管丢失自动求导，恢复原生分解，并修正 FP32 `pow(x,1)` 的舍入。clone、max_pool3d 的注册声明及 unbind_copy 的重复注册也已整理，实际反向检查通过。

Linear 前反向对非连续输入先使用 `view`，整网训练会中断。现先整理布局，再按显式行数展平；补齐零宽度、零批次和零输出维度。MM 的空求和与空输出也有独立处理及原地输出检查。

服务器部署版包含仓库缺失的 cdist 反向修复。最初合并包保留了仓库旧文件，坐标损失的一部分梯度因此丢失：FP32 完整梯度差异约 30.8%，而 loss 几乎没变。独立随机上游梯度测试定位到 cdist；恢复后完成下表整网复测。clone 的反向测试正常。服务器训练包 1,406 个源码／配置文件的对照清单见 [部署清单](deployed_full_source_manifest.json)。

## 两步整网数值对照

固定种子 20260825、相同两个真实输入、初始权重和优化器配置，分别运行 FP32、TF32 两步。第 1 步实际更新前的学习率为 `1.8e-5`。PPU 新旧结果与既有 H100 缓存核对：输入与资产、4,174 个初始参数哈希及 3,338 条有序随机数记录均一致。比较第 1 步全部 4,170 份梯度、464,441,841 个元素，以及 4,174 份实际参数更新。

本节比较同配置下的数值一致性。梯度和更新的百分比均为 `100 × ||PPU − H100||₂ / ||H100||₂`，对全部元素组成的向量进行 CPU FP64 统计。

{table(headers, training_rows)}

TF32 的完整梯度和实际更新相对差异分别减少 {tf_grad:.2f}%、{tf_update:.2f}%；第 1 步 loss 与 H100 的绝对差异增大。FP32 的梯度与更新差异分别从 0.001540%／0.042941% 变为 0.001966%／0.056261%，均有增加。

完整分模块统计、前十个参数差异、快照哈希和 step0 Adam 对照见 [FP32](training_comparison_fp32.json)、[TF32](training_comparison_tf32.json)。

## 单算子接口计时

单位为微秒，PPU 单元格为现有部署版 → 合并版。MM/addmm/Linear 的主体输入为 64×128 与 128×64，BMM 另有批次 2；Softmax 输入为 [600,600,8]、沿维度 1 计算。每次测量包含完整前向接口中的分配和必要整理，采用七轮、每轮三次的中位数。建立注册上下文的开销在计时前完成。

{table(headers, timing_rows)}

“—” 表示缺少这组相同输入的 H100 单算子计时。全部逐轮记录见 [计时 CSV](registered_api_timings.csv)。最终 F.linear 通过原生分解调用矩阵算子，实际 ATen 调用保留在入口数据中；导出的 Python Linear 函数另有源码记录。

## 验证与复现

- 最终包精选回归：21 个文件，{suite['counts']['passed']} 通过、{suite['counts']['skipped']} 跳过、0 失败。使用 `--quick --ref cpu --first-parameter-only`，逐文件返回码和 JUnit 计数见 [回归结果](validation_restored_results.json)。
- 定向回归：13 个 Linear 布局／空维度、7 个距离反向、6 个空矩阵前向／反向／out 用例，共 26 项通过。
- 实际自动求导：22 项通过，采用相同的非恒定上游梯度；距离测试还覆盖共享输入、双输入、广播及重复上下文，CPU 梯度保持原生结果。
- 实际 FP32／TF32 入口：MM、addmm、BMM、Linear、Softmax 共 10 项前向与梯度检查通过，单算子相对 L2 检查阈值为 `2e-5`。
- Python 语法和 25 份导出声明检查通过。当前环境未安装 pre-commit，完整格式钩子未执行。`git diff --check` 返回 2，提示上游文件的 CRLF 和保留的辅助源码末尾空行；原始回执保留在 `final_static_validation.json`。

测试源码树为 `{publication['tested_tree_sha256']}`；发布源码树为 `{publication['tree_sha256']}`。两者仅一段已过时的 Linear 完成状态注释不同，可执行 AST 相同，见 [注释修订记录](policy_comment_provenance.json)。测试文件的一次空行整理另有 [记录](test_formatting_provenance.json)。

首次失败、旧候选的整网数据、修复源码、各轮源码哈希和部署回执完整保留。旧整层 Linear 的 FP32 任务后来完成；保留其数值记录，cdist 回退同时影响该任务与组合入口的对照。整网训练包含审计和保存开销，计时标为 `timing_valid=false`。所有原始张量和 checkpoint 留在计算服务器。
"""
(HERE / "REPORT.md").write_text(report)
print(json.dumps({"TF32_gradient_reduction_percent": tf_grad, "TF32_update_reduction_percent": tf_update, "selected_suite": suite["counts"]}))
