Read-only review complete. No source was changed and no device/tensor execution was performed. The exact task texts were returned separately for the trace.

The earlier clone response below is a current complete restatement, not a claimed verbatim reconstruction of prose lost during context compaction.

- In pinned upstream `45339c4`, the clone declaration (`src/flag_gems/__init__.py:931-936`) and max_pool3d declaration (`:1514-1519`) pass `(AUTOGRAD_DISPATCH_KEY, torch.library.fallthrough_kernel)` without an enclosing tuple.
- `runtime/op_registrar.py:210-229` interprets those as two separate specifications. `:257-260` registers them sequentially; the exception handler at `:264-267` does not roll back successful registrations.
- Consequently, the old declaration first installs the device implementation, then installs the custom function at `Autograd`, then fails on the function-valued dispatch key. The warning does **not** establish that registration had no effect.
- Changing it to `((AUTOGRAD_DISPATCH_KEY, torch.library.fallthrough_kernel),)` changes the successfully registered Autograd implementation. To preserve the old successful registrations while fixing the malformed syntax, use `(AUTOGRAD_DISPATCH_KEY,)`, or an explicit nested pair such as `((AUTOGRAD_DISPATCH_KEY, clone),)`. Apply the equivalent declaration for max_pool3d.
- Both implementations use custom `autograd.Function.apply`; direct function tests do not establish the behavior of registered `torch.clone`/pool calls. Relevant implementation locations from the reviewed snapshot: `ops/clone.py:38-46,158`; `ops/max_pool3d.py:84-108,128`.
- The CPU AST-extracted registrar experiment used a fake library, not Torch tensors. Its exact output was:

```text
old malformed registered= [('clone', 'clone', 'CUDA'), ('clone', 'clone', 'Autograd')] error= invalid dispatch key type
current nested fallthrough registered= [('clone', 'clone', 'CUDA'), ('clone', 'fallthrough', 'Autograd')] error= None
custom-autograd registration registered= [('clone', 'clone', 'CUDA'), ('clone', 'clone', 'Autograd')] error= None
explicit custom-autograd registration registered= [('clone', 'clone', 'CUDA'), ('clone', 'clone', 'Autograd')] error= None
```

This establishes parser and partial-registration semantics, **not** the magnitude or cause of any training-gradient discrepancy. I found no obvious newly introduced proportional-gradient formula defect in the inspected mul/div/copy source; that negative source finding is not runtime validation.

For the current `cdist` investigation, let `M=/home/lgx1429541889/Protenix/Protenix-main`, `B=/home/lgx1429541889/Protenix/comparison/ppu_full_acceptance_20261005/precision_residual_20261019`, and `P=/home/lgx1429541889/Protenix/comparison/upstream_merge_20261106`.

`B/model_ieee_matching` is absent locally. However, the local mirrored files match both platform entries in `B/matching_candidate_manifests.json:5-7,18-20`, and the recorded imports in `P/training_failed_layout.json:25-33,272-280`:

| Mirrored file | Verified SHA256 |
|---|---|
| `protenix/model/loss.py` | `52139a09a83030cdb0c9931dcc7204592b8007273f87f2a900e3f70a4c444377` |
| `protenix/utils/torch_utils.py` | `de4201459cf1d6ac2c4accc5fb762adb659639a6e0d0cc44e70779e263f86fe2` |
| `protenix/utils/permutation/permutation.py` | `ee5cb4315dc2552b6d1138c66cf909afd5768b83e51c521f325697b151403260` |

All calls in `M/protenix/model/loss.py` import the wrapper at line 29. That wrapper, [`torch_utils.py:70-78`](/home/lgx1429541889/Protenix/Protenix-main/protenix/utils/torch_utils.py:70), calls public `torch.cdist` with default `p=2` and explicit `compute_mode="donot_use_mm_for_euclid_dist"`.

| Loss call site | Shapes and gradient role |
|---|---|
| `loss.py:259-284`, SmoothLDDT dense path | Predicted coordinates `[..., N_sample, N_atom, 3]`; square atom-distance outputs. Prediction distances participate in loss backward. Both arguments alias the same tensor, or are two slices of the same base in the chunked path. |
| `loss.py:582`, distogram labels | Representative coordinates `[..., N_token, 3]`; square token-distance output. Called under `no_grad` at `:618-623`. |
| `loss.py:705-710`, PDE labels | True square token distances and predicted `[..., N_sample, N_token, N_token]` distances. Label generation is under `no_grad` at `:755-761`; labels detach at `:726`; actual training passes detached prediction coordinates at `:1781`. |
| `loss.py:1246-1251`, bespoke lDDT labels | Rectangular all-atom versus representative-polymer subset distances: `[..., N_sample, N_atom, N_atom(m)]`. Actual training invokes this under `no_grad` with detached predictions at `:1758-1771`. |
| `loss.py:1516`, AF3 label preparation | Ground-truth square atom distances, then mask and dtype cast. Called under `no_grad` at `:1829-1830`. |
| `loss.py:1551-1555`, additional prediction distances | Differentiable square predicted distances, but only when `loss_metrics_sparse_enable=False`. |

The important candidate for a registered backward test is therefore [`loss.py:273-284`](/home/lgx1429541889/Protenix/Protenix-main/protenix/model/loss.py:273), not merely the presence of any `cdist` call. The dense branch is selected at `loss.py:1657-1666`. Local defaults specify dense SmoothLDDT enabled and its sample chunk size 1 (`configs/configs_base.py:394,398`), sparse metrics enabled (`:136`), and `train_confidence_only=False` (`:45`). The inspected training configuration builder does not override those loss settings (`comparison/ppu_full_acceptance_20261005/training_experimental.py:165-204`). Final resolved runtime configuration remains the strongest evidence of the branch actually executed.

Shape caveat: the dense method documents leading batch dimensions but its chunk slicing addresses dimension 0 directly (`loss.py:273-283`); the training loss explicitly assumes batch size 1 (`:1738`). Tests should reproduce the actual unbatched/sample-first training layout before generalizing.

There is a separate direct default-mode call at [`confidence.py:279-281`](/home/lgx1429541889/Protenix/Protenix-main/protenix/model/modules/confidence.py:279). It casts representative coordinates to FP32 and computes square token distances, with both `p` and `compute_mode` omitted. In the inspected training model, its `coordinate_mini` input is produced under `no_grad`, detached, and passed to the confidence head (`protenix/model/protenix.py:721-755,773-779`). Thus coordinate backward through this call is not required by that training path, although its forward distances affect confidence features. These additional model files were inspected locally; I did not establish their deployment hashes in this narrow task.

I found no cdist exclusion or monkeypatch in the inspected active training chain:

- `training_experimental.py:520-538` constructs `use_gems` from environment include/exclude lists.
- `B/linear_x3_environment_20261023.sh:21,23` leaves exclusions empty and unsets inclusion filtering.
- `P/queue_training_layout.py:19-25` inherits that environment while replacing the package, entry script, and import path.
- `geometry_ieee` only temporarily disables the CUDA matmul TF32 flag and restores it (`torch_utils.py:25-35`); it does not replace or exclude cdist.
- `P/upstream_training_observer_20261106.py:18-24` forwards the original operation unchanged.
- Historical confidence experiments elsewhere in the comparison tree do monkeypatch cdist, but the inspected current wrapper does not import them.

The remote startup shim is referenced through `PYTHONPATH` but not fully available in this local mirror, so this is a source-supported finding about the inspected chain, not proof that no unseen startup mutation exists.

Candidate registered-operation checks:

1. Call **public** `torch.cdist` inside the complete package’s `use_gems` registration context, using explicit `p=2` and `donot_use_mm_for_euclid_dist`. Compare forward and VJP with native execution outside the context.
2. Reproduce `[1, N_atom, 3]`, chunk views, and the **same-base two-input aliasing**. Verify accumulated base gradients equal the sum of gradients from an equivalent independent-input reference.
3. Use nonsymmetric upstream weights, repeated/coincident points, and self-distance diagonals. These expose a missing input-gradient contribution or unsafe zero-distance handling that a symmetric scalar loss might conceal.
4. Cover representative real atom counts plus tail sizes; test contiguous and actual sliced layouts. Run both global precision policies without relaxing existing tolerances.
5. For no-grad label paths, test rectangular subset distances and values near lDDT/bin thresholds; coordinate-gradient parity is not their training requirement.
6. Test the confidence head’s default-mode, FP32 token-distance call separately.

No evidence from this investigation establishes that `cdist` caused the observed training discrepancy.
