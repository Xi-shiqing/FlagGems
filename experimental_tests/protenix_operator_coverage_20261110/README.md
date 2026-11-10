# Protenix PPU operator improvements

This change extends complete-write output allocation to twelve API families, fixes generic negative indices and generated-source disk-cache separation, and packs large batch-interleaved FP32 BMM operands into contiguous storage. BMM uses its existing dot kernel, tile configuration and accumulation schedule. Selection uses dtype/device, dimensions, dense strides, workload size and a 1 GiB per-call input-copy budget; overlapping outputs and allocation exhaustion retain the original route.

The allocation helper accepts grad-disabled CUDA execution, including custom autograd forward methods. Global deterministic settings remain enabled. Index outputs keep ordinary allocation.

On the original PPU runtime, four saved real training operand pairs gave FP32 complete registered BMM timings of 16.36 → 3.68 ms, 16.35 → 3.67 ms, 66.17 → 21.92 ms and 66.73 → 21.88 ms. Timings include input copies, allocations and device synchronization; each path has 40 samples. Complete outputs match byte-for-byte. CPU FP64 checks cover 512 output elements per case. The TF32 rows retain the existing path.

The BMM-only two-step training ablation keeps the same allocation and native LayerNorm changes on both sides. The second step is timed in four restart runs, in control/candidate/candidate/control order; the existing common training observers remain active.

| Training step / seconds | PPU FlagGems FP32 | H100 native FP32 | PPU FlagGems TF32 | H100 native TF32 |
|---|---:|---:|---:|---:|
| Packing disabled | 114.309 | 36.276 | 71.671 | 21.053 |
| Packing enabled | 93.628 | 36.276 | 71.345 | 21.053 |

FP32 step time decreases by 18.09%. For both precisions, the second accuracy step (native audit_step=1) retains all 4,170 gradient tensors and 4,174 actual parameter-update tensors byte-for-byte. Both accuracy steps match losses, gradient norms, inputs and ordered RNG, with matching parameter snapshots at the start of native step 1. Two observed FP32 accuracy steps select packing 1,024 times; TF32 selects it zero times. H100 numbers reuse the existing audited input/step record. Both runs retain activation recomputation with blocks_per_ckpt=1. The stored H100 run enables checkpoint file saving and disables global deterministic algorithms; the PPU run disables checkpoint file saving and enables global deterministic algorithms.

Validation: 280 package checks and two additional producer/BMM/consumer checks on a nondefault stream passed, with no skipped tests; nine-model inference and three extra input cases passed complete-coordinate and five-confidence regression. All four repository metadata/export checks passed. Python pre-commit is unavailable in the local environment; its failed invocation is retained in the companion experiment record.

`results.json` contains scalar metrics, shapes/strides and source hashes. Raw tensors, native extension bodies/binaries and GPU traces remain on the original PPU. Native LayerNorm changes and final inference timing tables are documented in [the companion Protenix report](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/operator_coverage_20261110/REPORT.md).
