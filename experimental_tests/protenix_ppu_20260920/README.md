# Validated PPU deployment snapshot (2026-09-20)

This is the exact deployment overlay used with FlagGems 5.3.0rc2 on PPU,
not a replacement for the current upstream-merged generic operators.
`manifest.json` pins every source file. Preserve this distinction when reproducing results.

- `ops/mm.py`: revision 3 no-grad FP32 layout/split-K implementation. Its
  fast path deliberately excludes TF32 and training; no training speedup is claimed.
- `ops/mean.py`: repaired 256-lane coverage with aligned head/tail handling.
  215 cases matched H100 native bitwise; four model/input regressions completed.
- `ops/addmm.py`, `ops/bmm.py`: deployed runtime TF32 switch implementations.
- `test_mean.py`, `compare_unit.py`: standalone cross-device validation tools.
- `legacy_worktrees/`: lossless diffs against recorded base commits from the two
  older local worktrees. They preserve historical work without overwriting newer
  upstream or operator integration changes. They are not alternative releases.

Deployment: in an isolated copy of the recorded PPU FlagGems environment, copy
`ops/*.py` into `flag_gems/ops/`, then run the paired validation tools. Do not
install this historical overlay over arbitrary FlagGems versions. The current
upstream-merged branch has not been device-tested with this overlay.

Model reports and full reproduction entry points are in
[Protenix-FlagOS](https://github.com/Xi-shiqing/Protenix-FlagOS),
`comparison/ppu_mean_fix_20260920` and `comparison/ppu_mm_crossdevice_20260920`.
Strict cross-device MM accuracy at absolute 1e-7 is not universally passed.
