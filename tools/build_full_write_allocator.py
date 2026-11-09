"""Build the optional host-only CUDA allocation helper against the active PyTorch.

Run before importing FlagGems with FLAG_GEMS_NATIVE_FULL_WRITE=1. No GPU compiler,
package installation, or GPU computation is needed. Rebuild when PyTorch changes.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "src/flag_gems/utils"
    )
    args = parser.parse_args()
    os.environ.setdefault("MAX_JOBS", "1")
    import torch
    from torch.utils.cpp_extension import load

    source = root / "cpp/csrc/full_write_allocator.cpp"
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    module = load(
        name="_full_write_allocator",
        sources=[str(source)],
        build_directory=str(build),
        with_cuda=False,
        extra_cflags=["-O2"],
        extra_ldflags=["-ltorch_cuda"],
        verbose=True,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / Path(module.__file__).name
    if destination != Path(module.__file__):
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(module.__file__, temporary)
        temporary.replace(destination)
    receipt = {
        "torch": torch.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "binary": str(destination),
    }
    (build / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
