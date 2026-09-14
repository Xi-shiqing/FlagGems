# TriAttention Ascend 910C Candidate

This directory contains the AscendC/CANN implementation selected from the
910C optimization experiments. It is an opt-in, standalone custom operator
package. It is not wired into the FlagGems Python dispatcher or the top-level
FlagGems CMake build yet.

## Operator

The primary entry point is `tri_att_fa_official_bias_probe`, registered as
`TriAttFaOfficialBiasProbe`. It implements the Protenix TND attention path with
Q/K/V, two bias inputs, and cumulative sequence lengths.

The 910C schedule currently uses:

- KV stack size: 384
- query tile: 128 heads for H1/H2, 192 for H4, and 256 for H8
- dynamic block count: `min(task_count, 24)`
- compact tail producer for the physical P layout
- CANN/AscendC matrix multiplication, online softmax, and PV accumulation

The `official_fa` subtree is based on CANN Open Software FlashAttention
templates. The original license notices in those files are retained. The
package requires a compatible CANN installation at build time.

## Build on 910C

```bash
source /usr/local/Ascend/cann/set_env.bash
cmake --preset default
cmake --build build_out -j8
```

The preset targets `ascend910_93` and CANN 8.5.0. Override
`ASCEND_CANN_PACKAGE_PATH` when the installation is in another location.

## 910C evidence

The clean source tree was rebuilt in the 910C container with CANN 8.5.0.
For `B=1, S=693, D=32`, maximum absolute error against the reference path was
`7.22e-06` (H1/H2), `7.63e-06` (H4), and `6.20e-06` (H8).

Measured device-event latency for the custom path was:

| heads | custom (ms) | CANN reference (ms) | custom / CANN |
| ---: | ---: | ---: | ---: |
| 1 | 0.06080 | 0.04900 | 1.24x |
| 2 | 0.07808 | 0.04354 | 1.79x |
| 4 | 0.09391 | 0.04727 | 1.99x |
| 8 | 0.12544 | 0.05633 | 2.23x |

The CANN comparison uses a different fused bias representation and precision
contract, so these numbers are an engineering baseline, not an equal-precision
claim. The current candidate is correct on the tested shapes but is still
slower than the CANN vendor path.

This package is validated on Ascend 910C only. No support claim is made for
910B, other Ascend revisions, NVIDIA GPUs, or other sequence lengths without a
separate correctness and performance check.
