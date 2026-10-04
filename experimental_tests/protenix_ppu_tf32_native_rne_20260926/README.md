# PPU common TF32-RNE conversion experiment, 2026-09-26

This candidate changes the shared rounding helper, without adding shape rules,
changing GEMM tiles, enabling a weight cache, or changing accumulation order.
It is scoped to the existing PPU-ZW810E experimental overlay.

`native_rne.py` replaces the integer mantissa-rounding expression with
`cvt.rn.tf32.f32`. The existing exponent test and bitwise select preserve infinity
and NaN sign/payload. Disabled rounding and non-FP32 inputs retain the original
identity branch. Bare `cvt.rn` canonicalizes NaNs, and `cvt.rna` also changes
halfway rounding; neither bare variant satisfies the existing contract.

The independent probe covers 71,168 FP32 bit patterns, including both signs,
every exponent, rounding ties, subnormals, overflow, zeros, infinities and NaNs.
The candidate matches the integer reference exactly. Actual activation replays
from three model inputs verify 25 MM/addmm/BMM/TriAttention pairs; strict FP32
and TF32 output hashes match the original helper in all pairs.

The server experiment copies the existing overlay and replaces only
`flag_gems/ops/tf32_precision.py`. All other operator hashes, including the
mean fix, are checked unchanged. The original shared candidate stays intact.
`native_rne.patch` provides the exact replacement against the recorded baseline.

End-to-end and backward/update checks are recorded in the companion Protenix
experiment at `comparison/ppu_tf32_precision_20260923/generic_rne_20260926`.
This is an experimental candidate, not a completed nine-model acceptance.

The four direct-path model/input pairs use one warmup and five repeated
predictions on the same PPU, with the RHS cache and diagnostic profiling off.
Old/new RNE coordinates and all five confidence metrics are exactly equal.

| Model/input | Software RNE, s | Native RNE, s | Latency reduction |
| --- | ---: | ---: | ---: |
| base-default v1 / G01 | 10.4275 | 9.7520 | 6.48% |
| base-default v1 / prefix511 | 5.2907 | 4.9599 | 6.25% |
| v2 / G02 | 3.0056 | 2.8178 | 6.25% |
| base-default v1 / G02 | 2.2684 | 2.2659 | 0.11%, effectively flat |

The existing prepack route gains another 2.82% on G01 but loses about 1.02%
on G02; it is not enabled globally. Four operator forward/backward/SGD probes
have identical old/new hashes in FP32 and TF32. MM training latency falls 8.68%.
addmm initially loses 2.19%; a bounded reverse-order recheck loses about 0.69%
by pooled median, so no addmm training gain is claimed.

Matched checkpoints, normalized input assets, every feature tensor, ordered
atom identities and RNG records allow comparison with frozen H100 native
references. PPU FP32 versus H100 FP32 passes the 0.1 Å RMSD gate in 4/4 cases;
PPU TF32 versus H100 FP32 passes 3/4 (G01/base remains at 0.112882 Å).
This change preserves the existing RNE result and does not solve its remaining
TF32 structural error. H100 timings are historical, not fresh paired repeats.
The v2 candidate's first warmup took 562.90 s including JIT compilation; the
steady timings above exclude it, and cold-cache compilation was not compared.

See the [complete report](../../../Protenix/comparison/ppu_tf32_precision_20260923/generic_rne_20260926/GENERIC_RNE_20260926.md)
and [audited summary](../../../Protenix/comparison/ppu_tf32_precision_20260923/generic_rne_20260926/model_ab_summary.json).
`native_rne.patch` was independently applied to the recorded baseline; its
output SHA256 matches the server-tested candidate in `manifest.json`.
