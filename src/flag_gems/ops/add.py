# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.pointwise_dynamic import ComplexMode

logger = logging.getLogger(__name__)


@triton.jit
def _add_ppu_flat_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """Flat contiguous FP32 add for large dense Protenix tensors."""
    pid = tl.program_id(0).to(tl.int64)
    offsets = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


def _try_ppu_flat_add(A: torch.Tensor, B: torch.Tensor, output, alpha):
    """Use a flat address calculation for equal-shaped dense FP32 add.

    ``output`` may be ``None`` for the functional overload; allocation is
    deferred until all fast-path guards pass so ordinary small/broadcast adds
    pay no extra allocation cost.
    """
    if (
        os.getenv(
            "FLAG_GEMS_PPU_ADD_FLAT_FAST",
            # The flat kernel is faster in isolation, but its same-input
            # Protenix end-to-end A/B is statistically neutral (0.02%).
            # Keep it available for targeted workloads without imposing an
            # unproven dispatch change on the default path.
            "0",
        )
        != "1"
        or runtime.device.vendor_name != "thead"
        or not isinstance(alpha, (int, float))
        or float(alpha) != 1.0
        or A.device.type != "cuda"
        or A.dtype != torch.float32
        or B.dtype != torch.float32
        or A.shape != B.shape
        or not A.is_contiguous()
        or not B.is_contiguous()
        or A.numel() < (1 << 20)
        or (torch.is_grad_enabled() and (A.requires_grad or B.requires_grad))
    ):
        return None
    if output is None:
        output = torch.empty_like(A)
    if (
        output.dtype != torch.float32
        or output.shape != A.shape
        or not output.is_contiguous()
    ):
        return None
    n_elements = int(output.numel())
    block = 4096 if n_elements >= 1 << 24 else 2048
    warps = 8 if n_elements >= 1 << 24 else 4
    with torch_device_fn.device(output.device):
        _add_ppu_flat_kernel[(triton.cdiv(n_elements, block),)](
            A,
            B,
            output,
            n_elements,
            BLOCK=block,
            num_warps=warps,
            num_stages=2,
        )
    return output


@triton.jit
def _add_broadcast0_ppu_kernel(
    a_ptr, b_ptr, inner, BLOCK: tl.constexpr
):
    """A[d0,...] += B[1,...] for dense trailing storage.

    The profile shape has more than 2^31 elements.  Explicit int64 address
    arithmetic is therefore required on PPU; leaving ``pid0 * inner`` as the
    compiler's default integer width silently wraps the row offset.
    """
    pid0 = tl.program_id(0).to(tl.int64)
    pid1 = tl.program_id(1)
    inner = inner.to(tl.int64)
    offsets = (pid1 * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offsets < inner
    base = pid0 * inner
    a = tl.load(a_ptr + base + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(a_ptr + base + offsets, a + b, mask=mask)


@triton.jit
def _add_broadcast0_strided_ppu_kernel(
    a_ptr,
    b_ptr,
    d2,
    d3,
    b_s1,
    b_s2,
    b_s3,
    inner,
    BLOCK: tl.constexpr,
):
    """A[d0,d1,d2,d3] += B[1,d1,d2,d3] for the traced transposed B.

    The hot Protenix layout has B strides ``[8, 1, 5544, 8]``: its last
    dimension is interleaved, so the contiguous-B kernel cannot be reused.
    Flattening A keeps its writes coalesced; the three integer transforms map
    each flattened element back to B without materialising a transpose.
    """
    pid0 = tl.program_id(0).to(tl.int64)
    pid1 = tl.program_id(1)
    offsets = (pid1 * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offsets < inner
    plane = d2 * d3
    d1_index = offsets // plane
    rem = offsets % plane
    d2_index = rem // d3
    d3_index = rem % d3
    b_offset = d1_index * b_s1 + d2_index * b_s2 + d3_index * b_s3
    base = pid0 * inner
    a = tl.load(a_ptr + base + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + b_offset, mask=mask, other=0.0)
    tl.store(a_ptr + base + offsets, a + b, mask=mask)


@triton.jit
def _add_broadcast12_ppu_kernel(
    a_ptr, b_ptr, d3, inner, BLOCK: tl.constexpr
):
    """A[d0,d1,d2,d3] += B[d0,1,1,d3] for dense storage."""
    pid0 = tl.program_id(0).to(tl.int64)
    pid1 = tl.program_id(1)
    d3 = d3.to(tl.int64)
    inner = inner.to(tl.int64)
    offsets = (pid1 * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offsets < inner
    base = pid0 * inner
    a = tl.load(a_ptr + base + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + pid0 * d3 + offsets % d3, mask=mask, other=0.0)
    tl.store(a_ptr + base + offsets, a + b, mask=mask)


def _try_ppu_broadcast_hotpath(A: torch.Tensor, B: torch.Tensor, alpha) -> bool:
    """Run measured large broadcast add_ schedules on the PPU.

    ``add_`` also supports arbitrary broadcasting, dtype promotion, complex
    values, and scalar alpha.  The fast path is intentionally limited to the
    exact FP32 layouts observed in the Protenix trace.  It is enabled by
    default on PPU after the same-input end-to-end A/B; ``=0`` remains an
    explicit rollback switch.
    """
    if (
        os.getenv(
            "FLAG_GEMS_PPU_ADD_BROADCAST_FAST",
            "1" if runtime.device.vendor_name == "thead" else "0",
        ) != "1"
        or runtime.device.vendor_name != "thead"
        or not isinstance(alpha, (int, float))
        or float(alpha) != 1.0
        or A.device.type != "cuda"
        or A.dtype != torch.float32
        or B.dtype != torch.float32
        or A.ndim != 4
        or not A.is_contiguous()
        or B.shape == A.shape
    ):
        return False
    # The specialized kernels write ``A`` through Triton and therefore do not
    # create the autograd view/in-place versioning metadata that PyTorch's
    # regular ``add_`` path creates.  They are valid for inference (or an
    # explicitly disabled grad context), but using them on a tensor that
    # participates in a training graph silently drops the add from backward.
    # Keep the measured fast path while forcing training through the normal
    # FlagGems implementation until an autograd-aware fused kernel exists.
    if torch.is_grad_enabled() and (A.requires_grad or B.requires_grad):
        return False
    d0, d1, d2, d3 = (int(value) for value in A.shape)
    inner = d1 * d2 * d3
    if B.shape == (1, d1, d2, d3):
        if B.is_contiguous():
            block = 4096 if inner >= 1 << 20 else 1024
            warps = 8 if inner >= 1 << 20 else 4
            with torch_device_fn.device(A.device):
                _add_broadcast0_ppu_kernel[(d0, triton.cdiv(inner, block))](
                    A,
                    B,
                    inner,
                    BLOCK=block,
                    num_warps=warps,
                    num_stages=2,
                )
        elif (
            d1 in (2, 8)
            and d2 == 693
            and d3 == 693
            and tuple(int(value) for value in B.stride()[1:])
            == (1, d3 * d1, d1)
        ):
            # This covers the high-cost [693,{2,8},693,693] profile layouts.
            # The copy-then-dense schedule is the normal path for this exact
            # transposed layout.  It pays one contiguous materialisation, but
            # the dense kernel is faster end-to-end on the traced shape.  Set
            # the variable to ``0`` for an explicit rollback to strided loads.
            if os.getenv("FLAG_GEMS_PPU_ADD_BROADCAST_COPY", "1") == "1":
                dense = B.contiguous()
                # Keep the measured default stable while allowing isolated
                # shape experiments to override the launch geometry.  The
                # override is intentionally opt-in and is not used by the
                # production launcher unless explicitly requested.
                block = int(os.getenv("FLAG_GEMS_PPU_ADD_BROADCAST_COPY_BLOCK", "256"))
                warps = int(os.getenv("FLAG_GEMS_PPU_ADD_BROADCAST_COPY_WARPS", "4"))
                with torch_device_fn.device(A.device):
                    _add_broadcast0_ppu_kernel[
                        (d0, triton.cdiv(inner, block))
                    ](
                        A,
                        dense,
                        inner,
                        BLOCK=block,
                        num_warps=warps,
                        num_stages=2,
                    )
            else:
                block, warps = 512, 4
                with torch_device_fn.device(A.device):
                    _add_broadcast0_strided_ppu_kernel[
                        (d0, triton.cdiv(inner, block))
                    ](
                        A,
                        B,
                        d2,
                        d3,
                        int(B.stride(1)),
                        int(B.stride(2)),
                        int(B.stride(3)),
                        inner,
                        BLOCK=block,
                        num_warps=warps,
                        num_stages=2,
                    )
        else:
            return False
        return True
    if B.shape == (d0, 1, 1, d3) and B.is_contiguous():
        block = 4096 if inner >= 1 << 20 else 1024
        warps = 8 if inner >= 1 << 20 else 4
        with torch_device_fn.device(A.device):
            _add_broadcast12_ppu_kernel[(d0, triton.cdiv(inner, block))](
                A,
                B,
                d3,
                inner,
                BLOCK=block,
                num_warps=warps,
                num_stages=2,
            )
        return True
    return False


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


# Register complex support (elementwise)
add_func.register_complex(mode=ComplexMode.ELEMENTWISE)
add_func_tensor_scalar.register_complex(
    mode=ComplexMode.ELEMENTWISE, tensorize_scalars=True, fallback_target=add_func
)
add_func_scalar_tensor.register_complex(
    mode=ComplexMode.ELEMENTWISE, tensorize_scalars=True, fallback_target=add_func
)


def add(A, B, *, alpha=1):
    logger.debug("GEMS ADD")
    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        if A_is_complex and B_is_complex:
            Ar = torch.view_as_real(A)
            Br = torch.view_as_real(B)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = add_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
        elif A_is_complex and not B_is_complex:
            Ar = torch.view_as_real(A)
            if isinstance(B, torch.Tensor):
                Br = torch.view_as_real(B.to(A.dtype))
            else:
                # B is a real scalar; construct its real view directly to avoid
                # creating a complex-dtype tensor (unsupported by Triton kernels)
                scalar_real = torch.tensor(
                    [float(B), 0.0], dtype=Ar.dtype, device=Ar.device
                )
                Br = scalar_real.broadcast_to(Ar.shape)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = add_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
        else:
            Br = torch.view_as_real(B)
            if isinstance(A, torch.Tensor):
                Ar = torch.view_as_real(A.to(B.dtype))
            else:
                # A is a real scalar; construct its real view directly to avoid
                # creating a complex-dtype tensor (unsupported by Triton kernels)
                scalar_real = torch.tensor(
                    [float(A), 0.0], dtype=Br.dtype, device=Br.device
                )
                Ar = scalar_real.broadcast_to(Br.shape)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = add_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        if A.shape == B.shape:
            output = _try_ppu_flat_add(A, B, None, alpha)
            if output is not None:
                return output
        return add_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A + B * alpha)


def add_(A, B, *, alpha=1):
    logger.debug("GEMS ADD_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        if _try_ppu_flat_add(A, B, A, alpha) is not None:
            return A
        if _try_ppu_broadcast_hotpath(A, B, alpha):
            return A
        return add_func(A, B, alpha, out0=A)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha, out0=A)
    # elif isinstance(B, torch.Tensor):
    #     return add_func_scalar_tensor(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
