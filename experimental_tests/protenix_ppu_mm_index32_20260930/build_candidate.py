#!/usr/bin/env python3
"""Build an isolated overlay with one MM change; refuse an unexpected baseline."""
import argparse
import difflib
import hashlib
from pathlib import Path
import shutil


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--cache-call-arguments", action="store_true")
    parser.add_argument("--split-k-padding", action="store_true")
    args = parser.parse_args()
    path = args.baseline / "flag_gems/ops/mm.py"
    original = path.read_text()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "5288e11181d297200306812fed621b1f54bc0bf8aa5d43fcec84dc14f7e7e43b"
    modified = replace_once(original,
        "                    B_RNE_PREPACKED: tl.constexpr):\n    pid = tle.program_id(0)",
        "                    B_RNE_PREPACKED: tl.constexpr,\n"
        "                    INDEX_64: tl.constexpr = True):\n"
        "    index_dtype: tl.constexpr = tl.int64 if INDEX_64 else tl.int32\n"
        "    pid = tle.program_id(0).to(index_dtype)")
    start = modified.index("def mm_direct_fused(")
    end = modified.index("# PPU inference: copy", start)
    kernel = modified[start:end]
    assert kernel.count(".to(tl.int64)") == 3
    modified = modified[:start] + kernel.replace(".to(tl.int64)", ".to(index_dtype)") + modified[end:]
    guard = '''def _direct_offsets_fit_i32(a, b, c, M, N, K):
    # Include masked edge tiles; keep large/negative offsets on the original path.
    am, ak = a.stride()
    bk, bn = b.stride()
    cm, cn = c.stride()
    return (
        max(M + 64, N + 128, K + 32) < 2**31
        and min(am, ak, bk, bn, cm, cn) >= 0
        and (M + 64) * am + (K + 32) * ak < 2**31
        and (K + 32) * bk + (N + 128) * bn < 2**31
        and (M + 64) * cm + (N + 128) * cn < 2**31
    )


'''
    modified = replace_once(modified, "def _direct_mm(a, b, c, M, N, K):", guard + "def _direct_mm(a, b, c, M, N, K):")
    modified = replace_once(modified,
        "            B_RNE_PREPACKED=prepack_rhs,\n            num_warps=4, num_stages=1,",
        "            B_RNE_PREPACKED=prepack_rhs,\n"
        "            INDEX_64=not (_allow_tf32() and _direct_offsets_fit_i32(a, b, c, M, N, K)),\n"
        "            num_warps=4, num_stages=1,")
    if args.cache_call_arguments:
        cached_guard = guard.replace("(a, b, c, M, N, K)", "(M, N, K, am, ak, bk, bn, cm, cn)")
        cached_guard = cached_guard.replace("    am, ak = a.stride()\n    bk, bn = b.stride()\n    cm, cn = c.stride()\n", "")
        modified = replace_once(modified, guard, cached_guard)
        modified = replace_once(modified,
            "    with torch_device_fn.device(a.device):\n        mm_direct_fused[",
            "    am, ak = a.stride()\n    bk, bn = b.stride()\n    cm, cn = c.stride()\n"
            "    allow_tf32 = _allow_tf32()\n"
            "    with torch_device_fn.device(a.device):\n        mm_direct_fused[")
        start = modified.index("def _direct_mm(")
        end = modified.index("def _direct_out_safe(", start)
        caller = modified[start:end]
        caller = replace_once(caller,
            "            a.stride(0), a.stride(1), b.stride(0), b.stride(1),\n            c.stride(0), c.stride(1),",
            "            am, ak, bk, bn, cm, cn,")
        caller = caller.replace("ALLOW_TF32=_allow_tf32(),", "ALLOW_TF32=allow_tf32,")
        caller = caller.replace("_allow_tf32() and _direct_offsets_fit_i32(a, b, c, M, N, K)",
                                "allow_tf32 and _direct_offsets_fit_i32(M, N, K, am, ak, bk, bn, cm, cn)")
        modified = modified[:start] + caller + modified[end:]
    if args.split_k_padding:
        assert args.cache_call_arguments
        start = modified.index("def _direct_offsets_fit_i32(")
        end = modified.index("def _direct_mm(", start)
        guard = modified[start:end]
        guard = replace_once(guard,
            "M, N, K, am, ak, bk, bn, cm, cn):",
            "M, N, K, am, ak, bk, bn, cm, cn, split_k):")
        guard = guard.replace("K + 32", "K + 32 + split_k")
        modified = modified[:start] + guard + modified[end:]
        modified = replace_once(modified,
            "_direct_offsets_fit_i32(M, N, K, am, ak, bk, bn, cm, cn)",
            "_direct_offsets_fit_i32(M, N, K, am, ak, bk, bn, cm, cn, split_k)")
    if args.candidate.exists():
        assert (args.candidate / "flag_gems/ops/mm.py").read_text() == modified
    else:
        args.candidate.mkdir(parents=True)
        shutil.copytree(args.baseline / "flag_gems", args.candidate / "flag_gems",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (args.candidate / "flag_gems/ops/mm.py").write_text(modified)
    patch = "".join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                        fromfile="a/flag_gems/ops/mm.py", tofile="b/flag_gems/ops/mm.py"))
    args.patch.write_text(patch)
    print("CANDIDATE_MM_SHA256", hashlib.sha256(modified.encode()).hexdigest())


if __name__ == "__main__":
    main()
