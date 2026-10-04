# PPU TF32 RNE candidate (2026-09-23)

This is an isolated FlagGems overlay for correcting ordinary TF32 input
rounding on PPU. It keeps the previously validated FP32 MM/mean candidates and
uses the exact public Triangle Attention source currently mounted on PPU.
Nothing in the shared FlagGems installation is changed by this snapshot.

The TF32 path explicitly rounds FP32 operands to the TF32 mantissa with
round-to-nearest, ties-to-even (RNE), then keeps the existing TF32 dot and FP32
accumulator. With `torch.backends.cuda.matmul.allow_tf32=False`, the rounding
helper is a no-op. The `tf32`, `qk_tf32`, and `pv_tf32` Triangle Attention modes
use RNE; the `*_tf32x3_approx` modes remain separate performance experiments.

The candidate covers all current FP32 TF32 contractions in MM, addmm and BMM,
including MM/addmm split-K and the symmetric MM path. Triangle Attention uses
RNE for QK and PV in forward and for the matching QK/PV contractions in
backward. The Protenix public adapter continues to enter through
`flag_gems.ops.triangle_attention`; the PPU experiment must set
`PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION=tf32` to select the RNE path.

The current performance candidate rounds large BMM inputs once into contiguous
FP32 scratch buffers before the TF32 dot. A measured narrow-output case
`[16, 511, 511] x [16, 511, 24]` did not amortize this conversion, so the
candidate keeps strict FP32 for FP32 BMMs with `M > 1, N <= 32`; the
`M == 1, N <= 32` Protenix case remains eligible because its measured TF32
path is faster. This dispatch only changes performance/accuracy policy when
TF32 is requested; it does not alter the global setting or the strict path.

Run the semantic and true-entry smoke probe inside the existing PPU FlagGems
environment with this candidate package first on `PYTHONPATH`:

```bash
python test_tf32_rne.py --out tf32_rne_probe.json
```

It checks ties-to-even and special values bitwise, invokes dispatched
`torch.mm`/`torch.addmm`/`torch.bmm` while FlagGems is active, compares those
outputs with an FP64 product of independently RNE-quantized operands, and
executes Protenix's public Triangle Attention adapter in combined TF32, QK-only,
PV-only and strict FP32 modes through backward. The first device run passed on
PPU-ZW810E; `results/ppu_probe.json` records the output. This probe only checks
semantics. Real-input operator, training and G01_prefix511 E2E results are
summarized in the manifest and in the companion [Protenix audit](../../../Protenix/comparison/ppu_tf32_precision_20260923/TF32_ALL_OPS_AUDIT_20260924.md).
The current candidate improves on the previous RNE path but remains 18.6%
slower than strict PPU FP32 on that model/input, so keep it experimental until
the nine-model acceptance passes. The manifest records the exact sources and
hashes used to build this snapshot.

## RHS prepack cache experiment (2026-09-26)

The TF32-RNE direct MM path can optionally keep the rounded RHS buffers in a
bounded LRU cache. It is disabled by default; set
`FLAG_GEMS_PPU_TF32_RNE_RHS_CACHE_MB` to a positive MiB limit to enable it. The
key includes storage identity, layout, tensor version, device and current
stream, and the cache retains the source tensor while its packed copy is live.

On PPU GPU 7, a no-profiler/no-shape-trace five-repeat A/B with a 160 MiB cap
kept prediction hashes, RNG traces and all five confidence metrics identical.
G01 was unchanged (9.697 s cache off, 9.694 s on); G02 improved 2.8% (2.431 s
to 2.364 s) while retaining about 129–130 MiB. The G02 result remains about
5.2% slower than the prior direct-only median, so this cache stays experimental
and disabled by default. Full data is in the [Protenix route diagnostic](../../../Protenix/comparison/ppu_tf32_precision_20260923/route_diag_20260924/TF32_ROUTE_DIAG_20260924.md)
and [cache A/B summary](../../../Protenix/comparison/ppu_tf32_precision_20260923/route_diag_20260924/direct_only_control_20260925/route_profile_20260925/rhs_cache_experiment_summary_20260926.json).
