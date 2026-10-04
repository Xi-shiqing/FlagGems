# Protenix operator optimizations

This branch collects the FlagGems changes made while adapting Protenix to the
PPU backend. It is based on `xishiqing/triattention-v0.1.0`, so the existing
triangle-attention implementation and its tuning history remain available in
the same branch.

## Scope

The changes are grouped by the role they play in the Protenix training graph:

- **Attention:** stride-aware SDPA math forward, its autograd path, flash
  attention backward support, and triangle-attention forward/backward schedule
  and accumulation fixes.
- **Matrix operations:** PPU-oriented paths for `mm`, `bmm`, and `linear`,
  including large FP32 layout handling and the corresponding backward fixes.
- **Pointwise and reduction hot spots:** `add`, `mul`, `exp`, `sigmoid`,
  `silu`, `softmax`, `log_softmax`, `mean`, and `rsqrt`. Large contiguous PPU paths are narrowly
  guarded; experimental paths remain opt-in when model-level measurements did
  not establish a stable gain.
- **Tensor movement and indexing:** `copy`, `fill`, `pad`, `index`,
  `index_put`, `index_select`, `index_select_backward`, `scatter_add`, `slice_backward`, and
  `unfold_backward`. These changes cover broadcasting, non-contiguous layouts,
  64-bit address calculations, and autograd-compatible view behavior used by
  Protenix.
- **Geometry and scalar support:** `cdist`, `_euclidean_dist`, `vector_norm`,
  `arcsinh`, `max`, `min`, `pow`, `reciprocal`, `sqrt`, `div`, and
  `randperm`, together with the shape and dtype cases exercised by the model.
- **Runtime support:** The Thead heuristic configuration and LibEntry changes
  needed to dispatch and benchmark the PPU paths consistently.

## Validation boundary

The branch includes focused regression tests for the changed behavior,
including non-contiguous inputs, broadcasting, SDPA forward/backward, padding,
slice and unfold backward, `index_put`, `any`, true division, and LibEntry
dispatch. The source tree also passes Python bytecode compilation and
`git diff --check`.

The performance paths are intentionally separated by their confidence level:

- Linear layout handling produced the stable model-level improvement in the
  Protenix A/B runs.
- SDPA stride-aware execution produced a stable hot-shape improvement and
  passed the real `AttentionPairBias` forward/backward comparison.
- BMM, fused add/mul paths, and the pointwise schedules remain shape- and
  layout-specific. Their opt-in switches should be enabled only after a full
  Protenix graph measurement on the target device.

Numerical correctness remains the first gate: optimization switches must not
silently bypass autograd or change view and alias semantics.

## Repository boundary

This branch is the operator-side source of truth for the Protenix adaptation.
The companion `Protenix-FlagOS` repository contains the model integration,
device launch scripts, end-to-end tests, and model-level reports. It should
pin this branch or a specific commit instead of copying the FlagGems source
tree into the Protenix repository.

The current delivery line is the combination of:

- `7fe4d8e`: the Protenix operator implementation and focused tests;
- `2bae97d`: the inherited correctness fixes and their regression tests.

Triangle Attention tuning is already part of this branch's history. New
operator changes should continue on this branch when they are required by the
Protenix integration; a separate branch is only needed for an independent
experiment with a different compatibility contract.

## TF32 experiment archive through 2026-10-04

The [experiment index](../experimental_tests/protenix_ppu_experiments.md) records
the PPU TF32 work since the 2026-09-20 deployment snapshot, including accepted,
rejected, and diagnostic candidates. Source snapshots pin the actual isolated
server packages; they are not interchangeable with the newer generic source
tree in this branch.

The source integration now includes the previously validated PPU TF32-enabled
`K=1` IEEE outer-product route for MM/addmm/BMM and the `N=1` column-MM route
using the existing IEEE MV implementation. Their helper sources match the
archived tested candidates. The newer generic integration has been checked for
syntax and entry-point placement; the isolated server measurements do not
constitute device validation of the entire upstream-merged source tree.

The latest [host-pipeline experiment](../experimental_tests/protenix_ppu_host_pipeline_20261004/README.md)
removes redundant fill operations at reviewed complete-writing allocation
sites and prepares TF32 RHS operands once while retaining their physical
layout and the original MM kernel. Its nine-model, two-input report includes
all four native/FlagGems FP32/TF32 baselines and the candidate. This optimization
remains scoped to the experimental inference wrapper; it does not enable the
new allocation or RHS preparation paths by default.

The archive retains formal precision failures, unresolved RMSD intervals,
original measurements, and sampling limits. It does not claim full default
diffusion acceptance, a new nine-model training qualification, MFU, or an 80%
performance target. Credentials, model weights, raw activations, and raw
coordinates are not part of this archive.
