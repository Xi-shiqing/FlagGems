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
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_current_device = getattr(torch_device_fn, "current_device", None)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)

# 4096 x int32 = 16 KB per program at 4 warps.
_FLAT_COPY_BLOCK = 4096
# Offsets are int32, so the last masked tile must still fit in an int32.
_FLAT_COPY_MAX_ELEMS = 2**31 - 1 - _FLAT_COPY_BLOCK

_NARROW_INT_DTYPES = (torch.int8, torch.uint8, torch.int16)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_kernel(src):
    return src


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_to_bool_kernel(src):
    return src != 0


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def _copy_to_narrow_int_kernel(src):
    # PyTorch narrows from int32 (32768 -> int16 is -32768, -1.0 -> uint8 is
    # 255); a direct float store saturates at the narrow width instead.
    return src.to(tl.int32)


@triton.jit
def _flat_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """Bounded-tile flat copy for contiguous same-dtype tensors.

    ``_copy_kernel`` tiles by shape with a fixed 512-element tile, which leaves
    only 8 B (fp16) or 4 B (bool/int8) per thread and loses bandwidth; a flat
    tile keeps the accesses contiguous and 128-bit wide. ``NEED_MASK`` splits
    the exact-multiple case (masked accesses are slower) from the tail.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offsets < n_elements
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets, mask=mask), mask=mask)
    else:
        tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets))


def _int32_viewable(tensor: torch.Tensor) -> bool:
    """Whether ``tensor.reshape(-1).view(torch.int32)`` is expressible.

    A different element size is only allowed when the byte offset is a multiple
    of 4, which a contiguous slice such as ``t[1:]`` does not satisfy.
    """
    esize = tensor.element_size()
    if (tensor.numel() * esize) % 4 != 0:
        return False
    return (tensor.storage_offset() * esize) % 4 == 0


def _flat_copy_plan(dst: torch.Tensor, src: torch.Tensor):
    """Return ``(as_int32, n_elements)`` for ``_flat_copy_kernel``, or None.

    Reinterpreting as int32 keeps accesses 128-bit wide regardless of element
    size and halves the element count for 8-byte dtypes. Both operands must be
    viewable or neither.
    """
    if src.shape != dst.shape or src.dtype != dst.dtype:
        return None
    if not src.is_contiguous() or not dst.is_contiguous():
        return None

    as_int32 = _int32_viewable(src) and _int32_viewable(dst)
    n_elements = src.numel()
    if as_int32:
        n_elements = n_elements * src.element_size() // 4
    if n_elements > _FLAT_COPY_MAX_ELEMS:
        return None
    return as_int32, n_elements


def _flat_copy(
    dst: torch.Tensor, src: torch.Tensor, as_int32: bool, n_elements: int
) -> None:
    view = src.reshape(-1)
    out = dst.reshape(-1)
    if as_int32:
        view = view.view(torch.int32)
        out = out.view(torch.int32)

    grid = ((n_elements + _FLAT_COPY_BLOCK - 1) // _FLAT_COPY_BLOCK,)
    need_mask = n_elements % _FLAT_COPY_BLOCK != 0
    device_index = view.device.index
    if (
        _current_device is None
        or device_index is None
        or _current_device() == device_index
    ):
        _flat_copy_kernel[grid](view, out, n_elements, _FLAT_COPY_BLOCK, need_mask)
    else:
        with torch_device_fn.device(device_index):
            _flat_copy_kernel[grid](view, out, n_elements, _FLAT_COPY_BLOCK, need_mask)


@triton.jit
def _copy_strided_3d_kernel(
    dst_ptr,
    src_ptr,
    d0,
    d1,
    d2,
    ds0,
    ds1,
    ds2,
    ss0,
    ss1,
    ss2,
    n_elements,
    BLOCK: tl.constexpr,
):
    """Copy a 3-D permuted view into dense storage with affine addressing."""
    pid = tl.program_id(0)
    linear = pid * BLOCK + tl.arange(0, BLOCK)
    mask = linear < n_elements
    i0 = linear // (d1 * d2)
    rem = linear - i0 * (d1 * d2)
    i1 = rem // d2
    i2 = rem - i1 * d2
    src_offset = i0 * ss0 + i1 * ss1 + i2 * ss2
    dst_offset = i0 * ds0 + i1 * ds1 + i2 * ds2
    value = tl.load(src_ptr + src_offset, mask=mask, other=0)
    tl.store(dst_ptr + dst_offset, value, mask=mask)


def _strided_3d_schedule(shape: torch.Size) -> tuple[int, int]:
    """Choose the measured PPU schedule for profile-shaped 3-D copies."""
    d0, d1, d2 = (int(value) for value in shape)
    if d0 <= 64:
        return 2048, 8
    if d2 <= 192:
        return 1024, 4
    if d1 >= 512:
        return 512, 4
    return 4096, 8


def _try_strided_3d_copy(dst: torch.Tensor, src: torch.Tensor) -> bool:
    """Run the opt-in affine path for dense-destination 3-D views."""
    if os.getenv("FLAG_GEMS_PPU_COPY_STRIDED", "0") != "1":
        return False
    if dst.ndim != 3 or dst.shape != src.shape or dst.dtype != src.dtype:
        return False
    if not dst.is_contiguous() or src.is_contiguous():
        return False
    if any(int(stride) < 0 for stride in src.stride()):
        return False

    shape = dst.shape
    src_stride = src.stride()
    dst_stride = dst.stride()
    block, warps = _strided_3d_schedule(shape)
    n_elements = int(dst.numel())
    grid = (triton.cdiv(n_elements, block),)
    _copy_strided_3d_kernel[grid](
        dst,
        src,
        int(shape[0]),
        int(shape[1]),
        int(shape[2]),
        int(dst_stride[0]),
        int(dst_stride[1]),
        int(dst_stride[2]),
        int(src_stride[0]),
        int(src_stride[1]),
        int(src_stride[2]),
        n_elements,
        BLOCK=block,
        num_warps=warps,
        num_stages=2,
    )
    return True



def _can_use_triton(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst.layout != torch.strided or src.layout != torch.strided:
        return False
    if dst.device != src.device:
        return False
    if src.is_neg() or dst.is_neg():
        return False
    if dst.is_quantized or src.is_quantized:
        return False
    if src.is_complex() or dst.is_complex():
        # Preserve PyTorch's behaviour of warning when casting complex to real
        # by forcing the redispatch path, which issues the warning internally.
        return False
    if _FLOAT8_E8M0FNU is not None and (
        src.dtype == _FLOAT8_E8M0FNU or dst.dtype == _FLOAT8_E8M0FNU
    ):
        # Triton does not support float8 yet, so defer to PyTorch which has a reference implementation.
        return False
    return True


def _expand_like(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if src.shape == target_shape:
        return src
    return src.expand(target_shape)


def copy(
    template: torch.Tensor, src: torch.Tensor, *, non_blocking: Optional[bool] = False
):
    logger.debug("GEMS COPY (functional)")
    out = torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )
    copy_(out, src, non_blocking=bool(non_blocking))
    return out


def copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False):
    if isinstance(src, (int, float, bool)):
        src = torch.tensor(src, device=dst.device)
    elif not isinstance(src, torch.Tensor):
        raise TypeError("unsupport src type for copy_: ", type(src))

    # this is the same as PyTorch's check
    if dst._is_zerotensor():
        raise RuntimeError("ZeroTensors are immutable. Call clone() before copy_.")
    if src._is_zerotensor():
        return dst.zero_()

    if torch._C._is_alias_of(dst, src):
        # Align with PyTorch: if metadata fully matches, this is a no-op.
        if (
            dst.storage_offset() == src.storage_offset()
            and dst.stride() == src.stride()
            and dst.size() == src.size()
            and dst.dtype == src.dtype
            and dst.device == src.device
            and dst.is_conj() == src.is_conj()
            and dst.is_neg() == src.is_neg()
        ):
            return dst
        # Otherwise defer to PyTorch for well-defined semantics on overlapping writes.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if _FLOAT8_E8M0FNU is not None and (
        src.dtype == _FLOAT8_E8M0FNU or dst.dtype == _FLOAT8_E8M0FNU
    ):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if src.numel() > 2**31 - 1 or dst.numel() > 2**31 - 1:
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if not _can_use_triton(dst, src):
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    if dst.numel() == 0:
        # Respect PyTorch behaviour: empty tensors should still validate broadcast.
        return torch.ops.aten.copy_.default.redispatch(
            _FALLBACK_KEYSET, dst, src, non_blocking
        )

    logger.debug("GEMS COPY_")

    plan = _flat_copy_plan(dst, src)
    if plan is not None:
        _flat_copy(dst, src, *plan)
        return dst

    if src.shape != dst.shape:
        # Equal shapes always broadcast to themselves, so only a mismatch can fail.
        try:
            broadcast_shape = torch.broadcast_shapes(dst.shape, src.shape)
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc

        if torch.Size(broadcast_shape) != dst.shape:
            raise RuntimeError(
                f"The broadcast shape {broadcast_shape} does not match destination shape {tuple(dst.shape)}"
            )

    expanded_src = _expand_like(src, dst.shape)

    if _try_strided_3d_copy(dst, expanded_src):
        return dst

    if dst.dtype == torch.bool and src.dtype != torch.bool:
        kernel = _copy_to_bool_kernel
    elif src.dtype.is_floating_point and dst.dtype in _NARROW_INT_DTYPES:
        kernel = _copy_to_narrow_int_kernel
    else:
        kernel = _copy_kernel
    overload = kernel.instantiate(expanded_src.ndim)
    overload(expanded_src, out0=dst)
    return dst
