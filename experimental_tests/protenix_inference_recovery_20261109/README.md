# PPU inference performance restoration

This commit collects the tested PPU MM/BMM layout and long-reduction implementations, native-compatible activation/division and attention dispatch fixes, host launch overhead reductions, and optional full-write output allocation.

The 111-case comparison uses the frozen `restore_attention` package recorded in REPORT.md. It reaches FP32 relative MFU 68.815671% versus historical 66.127007%, and TF32 relative MFU 207.741565%. All 222 input/RNG/source audits pass. Numerical threshold flags remain in the report and CSV.

The optional allocator defaults off (`FLAG_GEMS_NATIVE_FULL_WRITE=1` enables it). It was separately tested across nine models, plus longer inputs. The repository integration retains native scalar-reciprocal division semantics; the frozen full111 inference package used legacy scalar division. The four-path full111 table and repository allocation comparisons therefore have distinct source manifests.

Validation already completed on the original PPU container: 77 allocator tests, 66 MM/BMM poisoned-output checks, 76 clone/cumsum tests, 104 supplemental full-prediction audits, and four static CI checks. The earlier operator/kernel validation records are in the Protenix comparison directories referenced by the report. Local Python syntax and git diff whitespace checks pass. `pre-commit run --all-files` was attempted again before commit but pre-commit is not installed.

Raw coordinates, weights, profiler traces and generated binary extensions stay on the original server. No new H100 jobs were used for this restoration.
