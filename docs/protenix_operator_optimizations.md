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
