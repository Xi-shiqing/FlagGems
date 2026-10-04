"""Small precision helpers shared by the PPU TF32 experiment overlay."""

import hashlib
import fnmatch
import json
import os
from pathlib import Path
import threading

import torch
import triton
import triton.language as tl


_CAPTURE_STATE = threading.local()
_CAPTURE_BEST = {}
_CAPTURE_COUNTS = {}


def _capture_mm_families():
    value = os.getenv("PROTENIX_TF32_CAPTURE_MM_FAMILIES", "").strip()
    if not value:
        return None
    families = set()
    for item in value.split(","):
        dimensions = item.lower().split("x")
        if len(dimensions) != 2:
            raise ValueError(f"invalid MM capture family: {item!r}")
        families.add(tuple(int(dimension) for dimension in dimensions))
    return families


def push_active_module(name):
    stack = getattr(_CAPTURE_STATE, "modules", None)
    if stack is None:
        stack = _CAPTURE_STATE.modules = []
    stack.append(name)


def pop_active_module(name):
    stack = getattr(_CAPTURE_STATE, "modules", [])
    if stack and stack[-1] == name:
        stack.pop()
    elif name in stack:
        stack.remove(name)


def active_module():
    stack = getattr(_CAPTURE_STATE, "modules", [])
    return stack[-1] if stack else None


def active_module_restored():
    """Return whether the active Protenix module should use strict FP32."""
    module = active_module()
    if module is None:
        return False
    patterns = (
        pattern.strip()
        for pattern in os.getenv("PROTENIX_TF32_RESTORE_MODULES", "").split(",")
    )
    return any(
        pattern and fnmatch.fnmatchcase(module, pattern)
        for pattern in patterns
    )


def tf32_enabled_for_active_module():
    return bool(torch.backends.cuda.matmul.allow_tf32) and not active_module_restored()


def capture_real_operands(
    kind, tensors, *, flops, output_elements, alpha=1.0, beta=1.0
):
    """Capture the largest live tiny, small, and large call of each kind."""
    root_value = os.getenv("PROTENIX_TF32_CAPTURE_DIR")
    if not root_value:
        return
    mm_families = _capture_mm_families()
    if mm_families is not None:
        if kind != "mm" or len(tensors) < 2:
            return
        family = (int(tensors[0].shape[-1]), int(tensors[1].shape[-1]))
        if family not in mm_families:
            return
    count = _CAPTURE_COUNTS.get(kind, 0) + 1
    _CAPTURE_COUNTS[kind] = count
    bucket = (
        "tiny" if output_elements <= 4096
        else "small" if output_elements <= 262144
        else "large"
    )
    slot = (
        f"{kind}_k{family[0]}_n{family[1]}_{bucket}"
        if mm_families is not None
        else f"{kind}_{bucket}"
    )
    nbytes = sum(t.numel() * t.element_size() for t in tensors)
    max_bytes = int(os.getenv("PROTENIX_TF32_CAPTURE_MAX_BYTES", str(64 << 20)))
    best = _CAPTURE_BEST.get(slot)
    if nbytes > max_bytes or (best is not None and flops <= best["flops"]):
        return

    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    tensor_records = []
    for index, tensor in enumerate(tensors):
        cpu_tensor = tensor.detach().to(
            device="cpu", copy=True, memory_format=torch.preserve_format
        )
        raw = cpu_tensor.contiguous().view(torch.uint8).numpy().tobytes()
        filename = f"{slot}_{index}.pt"
        torch.save(cpu_tensor, root / filename)
        tensor_records.append({
            "file": filename,
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    _CAPTURE_BEST[slot] = {
        "kind": kind,
        "bucket": bucket,
        "flops": int(flops),
        "operand_bytes": int(nbytes),
        "input_shapes": [list(t.shape) for t in tensors],
        "input_strides": [list(t.stride()) for t in tensors],
        "input_dtypes": [str(t.dtype) for t in tensors],
        "alpha": float(alpha),
        "beta": float(beta),
        "linear_module": active_module(),
        "tensors": tensor_records,
    }
    payload = {
        "status": "PASS",
        "calls": _CAPTURE_COUNTS,
        "selected": _CAPTURE_BEST,
    }
    temporary = root / ".activation_capture.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(root / "activation_capture.json")


@triton.jit
def round_tf32_rne(x, enabled: tl.constexpr):
    if enabled and x.dtype == tl.float32:
        bits = x.to(tl.uint32, bitcast=True)
        exponent_mask = tl.full((), 0x7F800000, tl.uint32)
        finite = (bits & exponent_mask) != exponent_mask
        rounded = tl.inline_asm_elementwise(
            "cvt.rn.tf32.f32 $0, $1;",
            constraints="=r,f",
            args=[x],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        # Preserve infinity and NaN bits, including sign and payload.
        return tl.where(finite, rounded.to(tl.uint32, bitcast=True), bits).to(
            tl.float32, bitcast=True
        )
    return x

def empty_full_write(shape, *, dtype, device):
    """Allocate fresh contiguous storage for a kernel that writes every element.

    No global deterministic flag is changed. Callers must never read the buffer
    before the complete-writing kernel, including masked logical edge elements.
    """
    import math
    out = torch.empty(0, dtype=dtype, device=device)
    storage = torch.UntypedStorage(math.prod(shape) * out.element_size(), device=out.device)
    return out.set_(storage, 0, shape)



@triton.jit
def round_tf32_rne_bits(x, enabled: tl.constexpr):
    if enabled and x.dtype == tl.float32:
        bits = x.to(tl.uint32, bitcast=True)
        finite = (bits & 0x7f800000) != 0x7f800000
        rounded = (bits + 0xfff + ((bits >> 13) & 1)) & 0xffffe000
        return tl.where(finite, rounded, bits).to(tl.float32, bitcast=True)
    return x


@triton.jit
def round_tf32_rna_bits(x, enabled: tl.constexpr):
    if enabled and x.dtype == tl.float32:
        bits = x.to(tl.uint32, bitcast=True)
        finite = (bits & 0x7f800000) != 0x7f800000
        rounded = (bits + 0x1000) & 0xffffe000
        return tl.where(finite, rounded, bits).to(tl.float32, bitcast=True)
    return x
