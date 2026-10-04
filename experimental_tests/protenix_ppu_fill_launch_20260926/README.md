# PPU contiguous FP32 fill launch experiment, 2026-09-26

This experiment shortens the scalar `fill_` dispatch path used heavily by
Protenix's deterministic tensor allocation. It changes neither TF32 rounding
nor matrix multiplication. The same improvement applies to FP32 and TF32 model
runs, because both keep tensors in FP32 storage.

`fill.py` adds one flat store kernel for contiguous FP32 tensors. The decision
uses dtype, contiguity and scalar representability, without model names,
input names, size thresholds or shape tables. Noncontiguous tensors, other
dtypes and scalars outside this conversion contract retain the existing path.
Empty tensors return without launching; views retain their data pointer offset.
The kernel uses 64-bit offsets and does not specialize scalar values or sizes.
The previous pointwise kernel already avoided scalar specialization; the
claimed improvement is reduced wrapper/broadcast/layout work, not a new NaN
cache fix.

Deterministic algorithms and `torch.utils.deterministic.fill_uninitialized_memory`
remain enabled. The implementation still writes NaNs to uninitialized floating
storage. PyTorch documents this behavior and its potential performance cost in
the [deterministic utilities documentation](https://docs.pytorch.org/docs/main/deterministic.html).

The isolated server candidate is
`/root/xishiqing/ppu_tf32_rne_20260923/generic_pipeline_20260926/candidate`.
It copies the preceding native-RNE overlay and replaces only
`flag_gems/ops/fill.py`; the shared environment is unchanged. `baseline_fill.py`
is the exact server baseline, not the current repository's main `src` version.
`fill.patch` applies to this recorded baseline and reproduces `fill.py` exactly;
SHA256 values are recorded in [manifest.json](manifest.json).

The actual `torch.Tensor.fill_` dispatch passed 108 combinations of values and
shapes, plus eight view/guard/dtype fallback checks. Coverage includes empty
tensors, ranks zero through five, offset contiguous views, strided views,
signed zero, subnormal/maximal FP32, infinities and NaN. Both versions match
the CPU reference byte for byte. Launch-inclusive per-fill wall time falls
from approximately 55–58 microseconds to 21–23 microseconds across seven sizes
from one through 8,388,608 elements; this is not a TF32 compute-throughput claim.

End-to-end A/B, all five confidence metrics, coordinates, input/RNG audits and
four forward/backward/SGD probes are recorded in the companion
[Protenix report](../../../Protenix/comparison/ppu_tf32_precision_20260923/generic_pipeline_20260926/GENERIC_PIPELINE_20260926.md).
The model experiment covers four model/input pairs with one warmup and five
timed repetitions on one PPU. Profiling is separate from performance timing.
H100 comparisons reuse audited native FP32/TF32 references; they are historical
single-repeat measurements. This is not nine-model or full-training acceptance.

All 16 model processes completed. Same-precision old/new coordinates and all
five confidence metrics are exactly equal in all four pairs; feature, atom,
input/MSA, checkpoint and RNG audits pass. All four operator probes also retain
identical FP32/TF32 outputs, gradients and SGD updates. A final source audit
compares 1,367 Python files and finds only `ops/fill.py` changed.

| Model/input | Old FP32, s | New FP32, s | Old TF32, s | New TF32, s | TF32 latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| base v1 / G02, 245 tokens | 2.1796 | 1.7779 | 2.2453 | 1.8212 | 18.89% |
| v2 / G02, 245 tokens | 2.7815 | 2.6098 | 2.7469 | 2.5802 | 6.07% |
| base v1 / G01, 693 tokens | 11.3271 | 11.1819 | 9.7491 | 9.6005 | 1.52% |
| base v1 / G03, 1218 tokens | 46.5447 | 46.5161 | 34.3055 | 34.1744 | 0.38%, effectively flat |

G02 profiling retains 15,283 fill calls while their self CPU time falls from
1.0633 s to 0.6216 s. TF32 still has 545 more allocations/fills and 816 more GPU
kernel launches than FP32 in this case. Its cumulative GPU kernel duration is
lower (0.9632 vs 1.2445 s), yet unprofiled wall time is slightly higher. This
supports further work on common dispatch and TF32 preparation costs. Profiler
CPU/device totals overlap and must not be added or substituted for wall time.

The existing TF32 structure failures remain: compared with native H100 FP32,
G01/base RMSD is 0.112882 Å and G03/base is 0.330235 Å, above the 0.1 Å gate.
FP32 passes that gate in 4/4 pairs; all five confidence metrics pass in both
precisions. This fill optimization preserves TF32 accuracy rather than fixing
its remaining structural error. The companion
[audited summary](../../../Protenix/comparison/ppu_tf32_precision_20260923/generic_pipeline_20260926/pipeline_summary.json)
contains the complete four-mode comparison.
