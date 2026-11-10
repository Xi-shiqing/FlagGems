# PPU FP32 MM data reuse and dense BMM copies

The implementation commit is `ea9c6d6df9f6bad48576a04b75cf208711e74bdc`.

General strict-FP32 MM can use a 64×128 output tile while preserving its K32 loop, FP32 accumulation, tail masks and 64-bit indexing. Selection uses dtype, effective precision, output overlap, dimensions and the operand device's processing-unit count. It requires at least one full wave and at least 75% active task slots over all waves. Resource and plan caches store metadata only. Ordinary TF32 and insufficient parallel work keep 32×32 tiles.

The FP32 BMM preparation policy keeps its existing layout, work and 1 GiB workspace guards. Its new dense-permutation copy loads in physical-contiguous order, exchanges thread layouts in shared memory, and stores contiguously. It preserves bits, uses the operand device/current stream, and has an identity autograd backward. Unsupported layouts retain clone; allocation exhaustion at either input falls back to the existing BMM path.

Registered-interface measurements include allocation, copies and device synchronization, with 40 samples per path in ABBA order. Four eligible synthetic MM shape families improve by about 1.93–2.71×. Four saved real BMM operand pairs improve FP32 from 3.7345→2.1108 ms, 3.6826→2.0893 ms, 21.9283→8.0420 ms and 21.9107→8.0468 ms. Small/fallback and TF32 measurements are retained in REGISTERED_BENCH_v4.json.

All complete operator outputs match byte-for-byte. Independent CPU FP64 slices cover 256 output elements per MM and 512 per BMM. One underflow-scale BMM packet has relative L2=1 and maximum absolute error≈1.24e−46, identically in both paths. Special FP32 bits, supported permutations, offsets, output aliases, poisoned destination writes, gradients and nondefault streams are covered by the targeted checks.

The frozen package passed 312 checks, including the 42 targeted cases in UNIT_CHECKS_v2.json. Four repository static checks passed. Source hashes match the executed package. The local environment has no pre-commit/Black; invocation and environment records are retained in the companion experiment.

[The companion Protenix report](https://github.com/Xi-shiqing/Protenix-FlagOS/blob/master/comparison/fp32_mfu_20261110/REPORT.md) contains four-path inference/training timings, complete-state regression, profiles and unchanged-cohort MFU calculations. All raw tensors, coordinates and traces stay on the original PPU.

## Complete-model results

The fixed aggregate is nine G02 inferences plus one actual600-token training step.

| Metric | PPU FlagGems FP32 | H100 native FP32 | PPU FlagGems TF32 | H100 native TF32 |
|---|---:|---:|---:|---:|
| Previous seconds | 405.5335 | 114.1847 | 336.1224 | 94.9345 |
| Current seconds | 355.2228 | 114.1847 | 338.8835 | 94.9345 |
| Absolute MFU | 15.7207% | 18.2486% | 6.6987% | 2.9709% |
| Relative MFU | 86.1474% | 100.0000% | 225.4777% | 100.0000% |
| Direct efficiency | 32.1445% | 100.0000% | 28.0139% | 100.0000% |

The FP32 training ablation is 93.6225→87.2793 seconds. All twelve inference cases retain complete coordinates and all five confidence metrics byte-for-byte against the retained PPU implementation. Both precision checks retain 4,170 gradients and 4,174 actual updates at native training step 1 (second step), plus both-step loss/input/ordered-RNG checks. Profiles and successful launch-return tile/copy records pass.
