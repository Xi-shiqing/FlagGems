# PPU long-reduction MM: grouped accumulation

Date: 2026-11-05. This directory archives an experimental FlagGems MM kernel, numerical comparisons and measured API timings. The default operator implementation is unchanged.

The candidate groups every 1024 products within each parallel task before adding the group sum to a second accumulator. Parallel task selection uses output tile count, SM count and ceil(K/8192), with at most 64 partial outputs and a 64 MiB partial-output budget. TF32 uses 64×64 output tiles when M,N>=64; otherwise 32×32 tiles. The policy has no model, input, layer, training-step or exact-shape whitelist. It retains the existing FP32 and ordinary RNE TF32 dot modes.

`operator_sources/grouped_work_mm.py` is the final source, SHA-256 `a82e7d6b1c1d225e61d1f709db5a5626ad164798c4e87cc30a7485e6ec0c1a6c`. The full patch against the measured baseline is `operator_sources/grouped_work_mm_from_baseline.patch`. Source files retain their existing license headers. `operator_source_manifest.json` lists baseline, previous variants and hashes.

## Fixed actual weight-gradient inputs

Four saved inputs from two original FP32 and two original TF32 training steps were each executed in both precision modes through six implementations. The shape is 64×360000 @ 360000×256. For this table, FP32 uses FP32-source inputs and TF32 uses TF32-source inputs; every PPU/H100 comparison uses identical operands. All eight cross-precision evaluations and all 48 implementation comparisons are in the JSON/CSV files.

Relative L2 errors use the CPU float64 product of original operands as the denominator. Errors are pooled over the two steps by squared norms. Timing records device-event elapsed time around the allocating MM API, using the mean of each step's median from five interleaved samples after warmup. Host API wall time was not measured separately. H100 outputs are existing native records; those records did not time the individual MM calls.

| Measurement | PPU FlagGems FP32 | H100 native FP32 | PPU FlagGems TF32 | H100 native TF32 |
|---|---:|---:|---:|---:|
| Original baseline, CPU64 error / % | 0.00106401744 | 0.0000261605639 | 0.0595370943 | 0.00147221759 |
| Previous bounded/tree variant, CPU64 error / % | 0.0000183103367 | 0.0000261605639 | 0.00172132462 | 0.00147221759 |
| Grouped candidate, CPU64 error / % | 0.0000122274517 | 0.0000261605639 | 0.00147862506 | 0.00147221759 |
| Original baseline, MM API / ms | 16.953840 | unavailable | 11.837320 | unavailable |
| Previous bounded/tree variant, MM API / ms | 0.916720 | unavailable | 1.137860 | unavailable |
| Grouped candidate, MM API / ms | 0.999500 | unavailable | 0.823260 | unavailable |
| Previous bounded/tree variant, difference from native H100 / % | 0.0000253390631 | 0 (reference) | 0.000638215391 | 0 (reference) |
| Grouped candidate, difference from native H100 / % | 0.0000297410369 | 0 (reference) | 0.000174586112 | 0 (reference) |

The TF32 candidate takes 27.6% less time than the measured bounded/tree variant, while reducing its difference from native H100 and partial workspace from 8 MiB to 2.75 MiB. FP32 trades lower mathematical error for higher latency and slightly higher difference from native H100.

## Larger outputs and actual training

All six saved trunk MM calls with K>8192 and output elements>16384 were included. Four nonzero calls are 256×360000 @ 360000×1024; two zero calls are 128×1228800 @ 1228800×512. `large_final_complete.json` includes metrics for five implementations and every timing sample; `large_input_plan.json` records the original strides restored by the replay. Zero outputs are treated as exact-zero checks, with no relative-error improvement attributed to them.

For the four nonzero TF32 calls, the earlier fast 64×64 split variant takes 9.43–9.52 ms, while grouped accumulation takes 9.00–9.17 ms. Error relative to the RNE-quantized-operand CPU64 product falls from 0.00240%–0.00293% to 0.000382%–0.000432%. FP32 slows from 8.23–8.26 ms to 13.87–13.88 ms. Corresponding H100 single-operator records are unavailable for these six inputs.

The final candidate also completed two real Protenix TF32 steps in the original training container. Inputs, 4174 initial parameter hashes and all 3338 ordered RNG records matched the original controls. Step 0 kept the original zero learning rate and updated Adam state; step 1 used 1.8e-5. All 4170 preclip gradients (464441841 elements) and 4174 actual parameter updates were included. Loss records remained equal to the original PPU control. Gradient difference from cached H100 increased from 1.289788% to 1.293408%, and actual-update difference increased from 6.433917% to 6.439163%. FP32 full-model training was not run for this candidate. These results motivate retaining the candidate in this experimental directory.

## Reproduction records

- `replay_mm.py`, `input_plan.json`, `status_latest.json`: small-output actual-input replay plus 20 generic shape/stride/precision checks, including tails, mm_out and sentinel guards.
- `check_large_work_matrices.py`, `large_input_plan.json`, `large_final_complete.json`: all six larger-output calls.
- `compare_h100.py`, `h100_existing_manifest.json`, `comparison_current_cross_device.json`: eight explicitly authorized existing H100 outputs copied directly between the original servers, file hashes verified, CPU-only full-vector comparisons.
- `full_training_final_analysis.json`: complete-vector training comparisons and audits.
- `four_path_math_and_timing.csv`, `large_comparison.csv`, `SUMMARY.json`: scalar tables.
- `prepare_grouped_mm.py`, `prepare_grouped64_mm.py`, `prepare_work_bound_mm.py`: source construction and scope checks, with intermediate variants retained in `operator_sources/`.

The operator replay ran on the PPU host with PyTorch 2.9.0 and Triton 3.6.0; the actual training ran in the existing container with PyTorch 2.10.0 and Triton 3.5.0. Scripts record the original server paths and require those existing data/workspaces. Raw operands, trained weights and tensor outputs stay on the original servers.

The complete report, training scripts, execution receipts, environmental validation and negative intermediate results are in [the companion Protenix experiment](https://github.com/Xi-shiqing/Protenix-FlagOS/tree/master/comparison/mm_accumulation_20261105).
