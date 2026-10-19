# Protenix MUSA attention fusion experiments

**Platform:** Moore Threads MTT S5000 · **Reporting period:** 14–18 October 2026 · **Contributor:** wellingtonduke

The reporting period follows the stage report; supplementary measurements and quality review continued on 19 October, as preserved in the original records.

This directory archives a measured Protenix inference experiment: a process-local AMP compatibility adapter, bias-add/softmax fusion, and a BF16 separate-output / FP32 in-place storage policy. It contains the tested overlay sources and scalar evidence. It does not change FlagGems operator registration or default dispatch.

## Results

| Experiment | Measured result |
|---|---|
| Final paired comparison | 21 PDB tasks × 2 variants × 6 calls = 252 measured full-budget predictions |
| Generation budget | 10 recycles, 200 diffusion steps, 5 candidates per call, MSA enabled |
| Sum of per-task median prediction times | 6842.80 s → 4571.26 s, **33.20% less time** |
| Quality-screened subset | 20 tasks, **33.20% less time** under the same aggregation |
| Interleaved validation | 5.26%, 22.04%, and 20.86% less prediction time on three tasks |
| Four-path 840-token experiment | Hybrid: 20.79% less time and 32.75% less peak active tensor memory |
| Quality exception | `7bb8`: primary seed exceeds the lDDT screen; three additional seeds pass, but the original exception is retained |
| Supporting work | 647 counted MUSA measured calls, 160 local ablation scenes, 843 quality-tool records |

Prediction timing excludes weight loading, featurization, export, and evaluation. There are 21 tasks but 20 exact frozen inputs: `7amf` and `7amb` are identical inputs. No homology clustering was performed. GPU hours and repeated calls are not independent proteins or human work hours.

The H100 reference contains 21 previously measured tasks at the same generation budget. It uses platform-native kernels and was not regenerated from exactly the same serialized feature tensors. The strict optimization comparison is within MUSA, using the same frozen input and GPU for each pair. The optimized MUSA result remains slower than H100.

## Read first

- [正式实验报告](正式实验报告.md): Chinese research report with methods, full task table, ablations and limitations.
- [汇报摘要](汇报摘要.md): plain-language explanation of the work.
- [复现实验说明](复现实验说明.md): public reproduction instructions and prerequisites.
- [逐输入正式对照](逐输入正式对照.csv): timing, memory, frozen-input matching and quality screen per task.
- [7bb8质量复核](7bb8质量复核.csv): original exception and all additional seeds.

## Audit the published numbers

Only the Python standard library is needed:

```bash
python3 verify_results.py
```

The audit recomputes primary and multi-seed medians, peak memory, lDDT means, interleaved and storage comparisons, aggregate savings and call counts from `measurements.json` and `quality.json`. It also checks H100 reference budgets, the 160-scene numerical gate and original source hashes. This is an arithmetic/archive audit; it does not rerun inference or independently recompute structural quality from coordinates.

`measurements.json` contains 147 scalar measurement groups, including labelled warmups, exploratory groups and older precision paths. `primary` identifies the 42 final groups; `counted_in_647` identifies groups contributing to the reported MUSA call count. Only `final_native` and `final_hybrid` determine the 33.20% headline. `quality.json` includes all 843 labelled tool records, including historical evaluations. Tool `PASS` means evaluation succeeded, not that an optimized prediction passed the quality threshold.

## Implementation and evidence

The measured fusion/AMP/benchmark Python snapshots are preserved with provenance in `source_provenance.json`. Figure files are flattened and report image links adjusted for this archive. `run_worker.sh`, `verify_results.py` and the public guide are packaging utilities; the public launcher was syntax-checked, not GPU re-benchmarked. No kernel algorithm was changed during packaging.

The archive retains the unsuccessful all-in-place strategy and rejected projection screening. Not every task benefits in memory, and the small fallback task has no confirmed fusion acceleration. The original `7bb8` quality exception remains visible despite favourable additional seeds. Training speed and biological non-inferiority have not been established.

Weights, raw MSA/features, coordinates, structures, connection details and authentication material are not part of this archive. The complete experiment remains available in the original authorised environment; exact GPU replay requires those inputs. Public scalar records support numerical auditing, not reconstruction of omitted raw inputs.

This contribution was prepared with AI assistance. The repository-level licence applies; the external Protenix runtime, checkpoint and PXMeter remain separate dependencies.
