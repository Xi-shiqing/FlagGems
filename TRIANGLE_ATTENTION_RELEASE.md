# Triangle Attention v0.1.0 release candidate

This release candidate adds an independent streaming Triangle Attention operator to `flag_gems.ops`.

## Post-v0.1.0 backward optimization candidate

The current branch contains a follow-up PPU optimization for the measured production shape `(B,N,S,D)=(1,693,693,32)`:

- compute `delta=sum(output*dOutput)` inside the dQ kernel, removing one preprocessing launch;
- save the score gradient `dS` produced by dQ and reduce it directly for `dBias2`, instead of rebuilding QK, softmax, dP and dS in the bias kernel;
- for the measured H8 path, compute `dK=dS^T Q` from the saved scratch and run a separate dV kernel that only rebuilds the probability matrix;
- use FP16 storage for the H8 dS scratch while retaining FP32 matrix and reduction accumulators; H2/H4 keep FP32 scratch because FP16 did not improve the measured H2 path;
- use explicit 64-bit scratch offsets so tensors larger than 4 GiB are addressed correctly.

On PPU-ZW810E with strict IEEE FP32 and `S=693,H=8,D=32`, the measured dK/dV stage decreased from `66.210 ms` to `39.087 ms` (`1.694x`). The public forward-plus-backward path decreased from `153.460 ms` to `126.380 ms` in the same experiment. These are production-shape PPU results, not cross-device or whole-model claims.

An unvalidated forward-score cache experiment is intentionally excluded from this branch update. Before assigning a new release tag, rerun the existing 21 public tests on both H100 and PPU and repeat the H8 production-shape gate.

## Public surface

- `triangle_attention(...)`
- `TriangleAttention(precision=..., layout=...)`
- `triangle_attention_support_error(...)`

## Contract

- layouts: `BNSHD`, `BNHSD`
- dtypes: FP32, BF16, FP16
- head dimensions: 16, 32, 64
- gradients: first-order Q/K/V/bias2; bias1 is a non-learnable mask
- execution: strict Triton implementation with no fallback
- precision: explicit API selection; `ieee` by default

## Release gates

- [x] syntax and clean-diff checks
- [x] FP64-oracle forward/backward suite on NVIDIA H100
- [x] FP64-oracle forward/backward suite on PPU-ZW810E
- [x] real Adam optimizer step on both devices
- [x] wheel and source distribution build
- [x] isolated wheel installation and public-import regression on H100
- [x] production-shape performance evidence recorded

## Validation record

The final test suite passed on both release targets:

| Target | Environment | Result |
| --- | --- | ---: |
| NVIDIA H100 | Python 3.10.20, PyTorch 2.7.1, Triton 3.3.1 | 21 passed |
| PPU-ZW810E (asset 224) | FlagOS, Python 3.12.3, PyTorch 2.10.0, Triton 3.5.0 | 21 passed |

The H100 Python 3.10 wheel was then installed into an isolated virtual environment. Public API import reported operator version `0.1.0`, and the same suite passed from the installed wheel (`21 passed`).

The PPU validation used the release source directly. A target wheel build was also attempted on the asset-224 CPFS mount, but FlagGems' all-backend package assembly failed while creating the archive (`OSError: [Errno 22]`) and the remote checkout used for that packaging attempt was not the exact release base. That artifact is excluded from this release record; it is not treated as an operator failure or as a successful wheel validation.

On PPU-ZW810E, strict IEEE production-shape training with `(B,N,S,H,D)=(1,693,693,H,32)` achieved 1.33x total forward/backward speedup at H=4 and 1.35x at H=8 versus the application Triton baseline. On H100, the high-precision training path remains slower than the vendor library; this is a documented optimization target and is not misrepresented as a v0.1.0 win.

## Artifact policy

FlagGems wheels are tied to the Python ABI and target software stack. The source distribution is the portable release artifact; wheels are built and validated per target environment. The release does not bundle PyTorch or Triton.

## Known v0.1.0 limits

- no gradient for `bias1`, which is defined as a non-learnable additive mask
- first-order gradients only
- head dimension is limited to 16, 32, or 64
- TF32-family modes require FP32 inputs and explicit user selection
- no implicit fallback for unsupported inputs
