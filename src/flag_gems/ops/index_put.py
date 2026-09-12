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

import importlib
import logging
import math
import os
from typing import Any, Callable, List, Mapping, Tuple

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.code_cache import code_cache_dir
from flag_gems.utils.code_utils import IndentedBuffer, write_atomic

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _index_put_sorted_dim0_deterministic_kernel(
    input_ptr,
    index_ptr,
    values_ptr,
    n_rows,
    out_rows,
    n_cols,
    SEARCH_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Accumulate one sorted index segment per output in a fixed order."""
    group_out_row = tl.program_id(0)
    group = group_out_row // out_rows
    out_row = group_out_row % out_rows
    out_col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = out_col < n_cols
    out_offsets = (group * out_rows + out_row) * n_cols + out_col
    values_base = group * n_rows * n_cols

    lo = 0
    hi = n_rows
    for _ in tl.static_range(0, SEARCH_STEPS):
        active = lo < hi
        mid = (lo + hi) // 2
        index_value = tl.load(index_ptr + mid, mask=active, other=0)
        move_right = active & (index_value < out_row)
        lo = tl.where(move_right, mid + 1, lo)
        hi = tl.where(active & ~move_right, mid, hi)
    first = lo

    lo = first
    hi = n_rows
    target = out_row + 1
    for _ in tl.static_range(0, SEARCH_STEPS):
        active = lo < hi
        mid = (lo + hi) // 2
        index_value = tl.load(index_ptr + mid, mask=active, other=0)
        move_right = active & (index_value < target)
        lo = tl.where(move_right, mid + 1, lo)
        hi = tl.where(active & ~move_right, mid, hi)
    last = lo

    acc = tl.load(input_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
    pos = first
    while pos < last:
        value = tl.load(
            values_ptr + values_base + pos * n_cols + out_col,
            mask=mask,
            other=0.0,
        )
        acc += value.to(tl.float32)
        pos += 1
    tl.store(input_ptr + out_offsets, acc, mask=mask)


def _index_put_sorted_dim0_deterministic(inp, indices, values, accumulate):
    """Deterministic fast path for Protenix token-to-atom backward."""
    if not torch.are_deterministic_algorithms_enabled() or not accumulate:
        return False
    if len(indices) != 1 or indices[0].ndim != 1 or inp.ndim == 0:
        return False
    if not inp.is_contiguous() or not values.is_contiguous():
        return False
    index = indices[0].to(torch.int64).contiguous()
    n_rows = int(index.numel())
    if tuple(values.shape) != (n_rows, *tuple(inp.shape[1:])):
        return False
    if n_rows and not bool(torch.all(index[1:] >= index[:-1]).item()):
        return False
    if inp.numel() == 0:
        return True

    out_rows = int(inp.shape[0])
    n_cols = int(inp.numel()) // out_rows
    search_steps = max(1, n_rows.bit_length() + 1)
    block = 128
    _index_put_sorted_dim0_deterministic_kernel[
        (out_rows, triton.cdiv(n_cols, block))
    ](
        inp,
        index,
        values,
        n_rows,
        out_rows,
        n_cols,
        SEARCH_STEPS=search_steps,
        BLOCK=block,
    )
    return True


def _index_put_sorted_single_index_deterministic(
    inp, indices, values, accumulate
):
    """Fixed-order accumulate for one sorted advanced-index dimension."""
    if not torch.are_deterministic_algorithms_enabled() or not accumulate:
        return False
    if not inp.is_contiguous():
        return False
    tensor_pos = [dim for dim, index in enumerate(indices) if index is not None]
    if len(tensor_pos) != 1:
        return False
    dim = tensor_pos[0]
    index = indices[dim]
    if index.ndim != 1:
        return False
    index = index.to(device=inp.device, dtype=torch.int64).contiguous()
    n_rows = int(index.numel())
    if n_rows and not bool(torch.all(index[1:] >= index[:-1]).item()):
        return False

    value_shape = tuple(inp.shape[:dim]) + (n_rows,) + tuple(inp.shape[dim + 1 :])
    values = values.to(inp.device).broadcast_to(value_shape).contiguous()
    if inp.numel() == 0:
        return True

    n_groups = math.prod(inp.shape[:dim])
    out_rows = int(inp.shape[dim])
    n_cols = math.prod(inp.shape[dim + 1 :])
    search_steps = max(1, n_rows.bit_length() + 1)
    block = 128
    _index_put_sorted_dim0_deterministic_kernel[
        (n_groups * out_rows, triton.cdiv(n_cols, block))
    ](
        inp,
        index,
        values,
        n_rows,
        out_rows,
        n_cols,
        SEARCH_STEPS=search_steps,
        BLOCK=block,
    )
    return True


def _index_put_stable_keys_deterministic(inp, indices, values, accumulate):
    """Deterministically accumulate arbitrary leading advanced indices."""
    if (
        not torch.are_deterministic_algorithms_enabled()
        or not accumulate
        or not indices
        or len(indices) > inp.ndim
        or not inp.is_contiguous()
    ):
        return False

    index_shape = tuple(indices[0].shape)
    if any(tuple(index.shape) != index_shape for index in indices):
        return False
    n_rows = int(indices[0].numel())
    if any(int(index.numel()) != n_rows for index in indices):
        return False

    flat_indices = [index.to(torch.int64).contiguous().reshape(-1) for index in indices]
    for dim, index in enumerate(flat_indices):
        if index.numel() and not bool(
            torch.all((index >= 0) & (index < inp.shape[dim])).item()
        ):
            return False

    linear_index = flat_indices[0]
    for dim, index in enumerate(flat_indices[1:], start=1):
        linear_index = linear_index * int(inp.shape[dim]) + index

    out_rows = math.prod(inp.shape[: len(indices)])
    n_cols = math.prod(inp.shape[len(indices) :])
    values = values.to(inp.device).broadcast_to(index_shape + inp.shape[len(indices) :])
    values = values.contiguous().reshape(n_rows, n_cols)
    if n_rows:
        linear_index, order = torch.sort(linear_index, stable=True)
        values = torch.index_select(values, 0, order).contiguous()
    if inp.numel() == 0:
        return True

    search_steps = max(1, n_rows.bit_length() + 1)
    block = 128
    _index_put_sorted_dim0_deterministic_kernel[
        (out_rows, triton.cdiv(n_cols, block))
    ](
        inp.reshape(out_rows, n_cols),
        linear_index,
        values,
        n_rows,
        out_rows,
        n_cols,
        SEARCH_STEPS=search_steps,
        BLOCK=block,
    )
    return True


def get_max_rank_shape(indices: List[torch.Tensor]) -> List[int]:
    # Filter out None values (basic indexing markers)
    tensor_indices = [idx for idx in indices if idx is not None]
    if len(tensor_indices) == 0:
        return []
    max_rank = max([len(index.shape) for index in tensor_indices])
    shape = [0 for _ in range(max_rank)]
    for i in range(max_rank):
        max_num = 0
        for index in tensor_indices:
            axis = len(index.shape) - 1 - i
            if axis >= 0:
                max_num = max(max_num, index.shape[axis])
        shape[max_rank - 1 - i] = max_num
    return shape


def broadcast_indices(indices, target_shape):
    for i, index in enumerate(indices):
        if index is not None and tuple(index.shape) != tuple(target_shape):
            indices[i] = torch.broadcast_to(index, target_shape)


def generate_imports(code: IndentedBuffer) -> IndentedBuffer:
    code.writeline("import triton")
    code.writeline("import triton.language as tl")
    code.newline()
    code.writeline("from flag_gems.utils import libentry")
    code.writeline("from flag_gems import runtime")
    code.writeline("from flag_gems.utils.shape_utils import volume")
    code.writeline("from flag_gems.utils import triton_lang_extension as ext")

    code.newline()
    code.newline()
    return code


def generate_index_put_kernel(
    inp_rank, indices_len, index_rank, kernel_name: str, code: IndentedBuffer
):
    code.writeline("@libentry()")
    code.writeline("@triton.jit")
    code.writeline(f"def {kernel_name}(")
    with code.indent():
        args = ["input_ptr,"]
        args += [f"indices{i}_ptr," for i in range(indices_len)]
        args += ["values_ptr,"]
        args += [f"input_shape{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_shape{j}," for j in range(index_rank)]
        args += [f"input_stride{i}," for i in range(inp_rank)]
        for i in range(indices_len):
            args += [f"indices{i}_stride{j}," for j in range(index_rank)]
        args += [
            f"values_stride{i}," for i in range(index_rank + inp_rank - indices_len)
        ]
        args += [
            "M,",
            "N,",
            "IS_ACCUMULATE: tl.constexpr,",
            "BLOCK_SIZE0: tl.constexpr = 2,",
            "BLOCK_SIZE1: tl.constexpr = 2048,",
        ]
        code.writelines(args)
    code.writeline("):")

    with code.indent():
        code.writeline("pid0 = ext.program_id(axis=0)")
        code.writeline("pid1 = ext.program_id(axis=1)")
        code.writeline(
            "offset0 = pid0 * BLOCK_SIZE0 + tl.arange(0, BLOCK_SIZE0)[:, None]"
        )
        if inp_rank == indices_len:
            code.writeline("offset1 = pid1 * 1 + tl.arange(0, 1)[None, :]")
        else:
            code.writeline(
                "offset1 = pid1 * BLOCK_SIZE1 + tl.arange(0, BLOCK_SIZE1)[None, :]"
            )
        code.newline()
        code.writeline("cur_idx = offset0")
        for i in range(index_rank - 1, -1, -1):
            code.writeline(f"indices_idx{i} = cur_idx % indices0_shape{i}")
            code.writeline(f"cur_idx = cur_idx // indices0_shape{i}")
        code.newline()
        code.writeline("cur_idx = offset1")
        for i in range(inp_rank - 1, indices_len - 1, -1):
            code.writeline(f"input_idx{i} = cur_idx % input_shape{i}")
            code.writeline(f"cur_idx = cur_idx // input_shape{i}")
        code.newline()
        code.writeline("mask0 = offset0 < M")
        for i in range(indices_len):
            comp = [f"indices_idx{j} * indices{i}_stride{j}" for j in range(index_rank)]
            code.writeline(
                f"cur_index{i} = tl.load(indices{i}_ptr + {' + '.join(comp)}, mask=mask0, other=0)"
            )
        code.newline()
        index_mask = [
            f"(cur_index{i} >= 0) & (cur_index{i} < input_shape{i})"
            for i in range(indices_len)
        ]
        code.writeline(f"index_mask = {' & '.join(index_mask)}")
        code.writeline("mask1 = offset1 < N")
        code.writeline("mask = index_mask & mask0 & mask1")
        code.newline()
        comp = [f"cur_index{i} * input_stride{i}" for i in range(indices_len)]
        comp += [
            f"input_idx{i} * input_stride{i}" for i in range(indices_len, inp_rank)
        ]
        code.writeline(f"input_offset = {' + '.join(comp)}")
        comp = [f"indices_idx{i} * values_stride{i}" for i in range(index_rank)]
        comp += [
            f"input_idx{indices_len + i} * values_stride{index_rank + i}"
            for i in range(inp_rank - indices_len)
        ]
        code.writeline(f"values_offset = {' + '.join(comp)}")
        code.newline()
        code.writeline("cur_value = tl.load(values_ptr + values_offset, mask=mask)")
        code.writeline("if IS_ACCUMULATE:")
        with code.indent():
            code.writeline(
                "tl.atomic_add(input_ptr + input_offset, cur_value, mask=mask)"
            )
        code.writeline("else:")
        with code.indent():
            code.writeline("tl.store(input_ptr + input_offset, cur_value, mask=mask)")

    code.newline()
    code.newline()
    return code


def generate_index_put_wrapper(
    inp_rank,
    indices_len,
    index_rank,
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    code.writeline(f"def {wrapper_name}(input, indices, values, accumulate):")
    with code.indent():
        code.writeline("input_shape = input.shape")
        code.writeline("input_stride = input.stride()")
        for i in range(indices_len):
            code.writeline(f"indices{i}_shape = indices[{i}].shape")
            code.writeline(f"indices{i}_stride = indices[{i}].stride()")
        code.writeline("values_shape = values.shape")
        code.writeline("values_stride = values.stride()")
        code.writeline("M = indices[0].numel()")
        code.writeline(f"N = volume(input_shape[{indices_len}: ])")
        code.newline()
        code.writeline("grid = lambda meta: (")
        with code.indent():
            code.writeline("triton.cdiv(M, meta['BLOCK_SIZE0']), ")
            code.writeline("triton.cdiv(N, meta['BLOCK_SIZE1']), ")
        code.writeline(")")
        code.newline()
        code.writeline(f"{kernel_name}[grid](")
        with code.indent():
            args = ["input,"]
            args += [f"indices[{i}]," for i in range(indices_len)]
            args += ["values,"]
            args += [f"input_shape[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_shape[{j}]," for j in range(index_rank)]
            args += [f"input_stride[{i}]," for i in range(inp_rank)]
            for i in range(indices_len):
                args += [f"indices{i}_stride[{j}]," for j in range(index_rank)]
            args += [
                f"values_stride[{i}],"
                for i in range(index_rank + inp_rank - indices_len)
            ]
            args += ["M,", "N,", "accumulate==True,"]
            code.writelines(args)
        code.writeline(")")
        code.writeline("return input")
    code.newline()
    code.newline()
    return code


def generate_code(
    inputs: Tuple[Any],
    wrapper_name: str,
    kernel_name: str,
    code: IndentedBuffer,
):
    inp_rank = inputs[0].ndim
    # Filter out None values to get actual tensor indices
    tensor_indices = [idx for idx in inputs[1] if idx is not None]
    indices_len = len(tensor_indices)
    if indices_len == 0:
        raise ValueError("At least one non-None index tensor is required")
    index_rank = tensor_indices[0].ndim
    code = generate_imports(code)
    generate_index_put_kernel(inp_rank, indices_len, index_rank, kernel_name, code)
    generate_index_put_wrapper(
        inp_rank, indices_len, index_rank, wrapper_name, kernel_name, code
    )
    return code


class IndexPutFunction:
    def __init__(self):
        self.pid = os.getpid()
        self.overloads: Mapping[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        inp, tensor_indices, values, accumulate = args
        if _index_put_stable_keys_deterministic(
            inp, tensor_indices, values, accumulate
        ):
            return inp
        if _index_put_sorted_dim0_deterministic(
            inp, tensor_indices, values, accumulate
        ):
            return inp
        full_args = (inp, tensor_indices, values)

        key = self.arg_key(*full_args)
        if key in self.overloads:
            overload = self.overloads[key]
        else:
            code = IndentedBuffer()
            code = generate_code(
                full_args,
                "_index_put_wrapper",
                "_index_put_jit_function",
                code,
            )
            file_name = f"index_put_{key}.py"
            file_path = code_cache_dir() / file_name
            write_atomic(file_path, code.getvalue())

            spec = importlib.util.spec_from_file_location(
                f"_gen_module_rank_{key}",
                file_path,
            )

            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            overload = getattr(m, "_index_put_wrapper")
            self.overloads[key] = overload

        return overload(*args)

    def arg_key(self, *args, **kwargs):
        inp, tensor_indices, _ = args[0], args[1], args[2]
        inp_rank = inp.ndim
        indices_len = len(tensor_indices)
        if indices_len == 0:
            index_rank = 0
        else:
            index_rank = tensor_indices[0].ndim
        return f"inp_rank_{inp_rank}_indices_len_{indices_len}_index_rank_{index_rank}"


_index_put_func = IndexPutFunction()


def index_put(inp, indices, values, accumulate=False):
    logger.debug("GEMS INDEX PUT")

    out = inp.clone()
    return index_put_(out, indices, values, accumulate)


def index_put_(inp, indices, values, accumulate=False):
    logger.debug("GEMS INDEX PUT_")

    indices = list(indices)

    if not indices:
        raise ValueError("At least one index tensor is required")

    indices = [
        (
            index.to(inp.device)
            if index is not None and index.device != inp.device
            else index
        )
        for index in indices
    ]
    # step 1: index preprocessing
    processed_indices = []
    for idx in indices:
        if idx is None:
            processed_indices.append(None)
        elif idx.dtype in (torch.bool, torch.int8):
            # Expand bool masks into explicit integer indices
            processed_indices.extend(idx.nonzero(as_tuple=True))
        elif torch.is_tensor(idx):
            processed_indices.append(idx)
        else:
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )

    indices = processed_indices
    # Pad missing None indices to match input dimension
    if len(indices) < inp.ndim:
        indices.extend([None] * (inp.ndim - len(indices)))

    if len(indices) > inp.ndim:
        raise IndexError("too many indices for tensor of dimension {}".format(inp.ndim))

    if _index_put_sorted_single_index_deterministic(
        inp, indices, values, accumulate
    ):
        return inp

    # Step 2: Broadcast tensor indices
    tensor_pos = [i for i, x in enumerate(indices) if x is not None]
    if not tensor_pos:
        raise ValueError("At least one non-None index tensor is required")

    tensor_indices = [indices[i] for i in tensor_pos]
    if len(tensor_indices) > 1:
        broadcasted = torch.broadcast_tensors(*tensor_indices)
        for i, pos in enumerate(tensor_pos):
            indices[pos] = broadcasted[i]

    # Step 3: Transpose
    is_contiguous = (tensor_pos[-1] - tensor_pos[0] + 1) == len(tensor_pos)
    starts_with_none = indices[0] is None
    need_transpose = not is_contiguous or starts_with_none

    if need_transpose:
        perm_order = tensor_pos + [i for i, x in enumerate(indices) if x is None]
        inp_view = inp.permute(perm_order)
        final_indices = [indices[i] for i in tensor_pos] + [None] * (
            len(indices) - len(tensor_pos)
        )
    else:
        inp_view = inp
        final_indices = indices

    # Step 4: Handle Values shape and broadcasting
    tensors = [x for x in final_indices if x is not None]
    broadcast_shape = list(tensors[0].shape)
    slice_shape = [inp_view.shape[i] for i, x in enumerate(final_indices) if x is None]

    target_shape = broadcast_shape + slice_shape
    values = values.to(inp.device)
    if need_transpose and is_contiguous:
        num_before = tensor_pos[0]

        # 1. Broadcast to PyTorch natural shape
        before_dims = slice_shape[:num_before]
        after_dims = slice_shape[num_before:]
        natural_shape = before_dims + broadcast_shape + after_dims
        values = values.broadcast_to(natural_shape)

        # 2. Permute to Kernel expectation
        B, T = len(before_dims), len(broadcast_shape)
        val_perm = (
            list(range(B, B + T)) + list(range(0, B)) + list(range(B + T, values.ndim))
        )
        values = values.permute(val_perm)
    else:
        # direct broadcast
        values = values.broadcast_to(target_shape)

    _index_put_func(inp_view, tensors, values, accumulate)

    return inp


def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    logger.debug("GEMS _INDEX_PUT_IMPL_")

    # Autograd may preserve basic-index dimensions as ``None`` entries.  The
    # full normalizer below handles their transpose and broadcasting semantics.
    if any(index is None for index in indices):
        return index_put_(inp, indices, values, accumulate)

    # The `unsafe` parameter is a hint to PyTorch for bounds checking.
    # Our implementation always performs bounds checking, so we ignore this parameter.
    # This is consistent with how PyTorch handles it internally.

    indices = list(indices)
    if len(indices) == 1 and indices[0].dtype == torch.bool:
        mask = indices[0]

        if mask.device != inp.device:
            mask = mask.to(inp.device)

        indices = list(torch.where(mask))

        K = indices[0].numel()
        target_shape = (K,) + inp.shape[len(indices) :]

        if values.numel() == 1:
            values = torch.full(
                target_shape, values.item(), dtype=inp.dtype, device=inp.device
            )
        elif values.numel() == K:
            values = values.reshape((K,)).expand(target_shape)

    indices = [
        (
            index.to(inp.device)
            if index is not None and index.device != inp.device
            else index
        )
        for index in indices
    ]

    target_shape = get_max_rank_shape(indices)
    broadcast_indices(indices, target_shape)
    target_shape += inp.shape[len(indices) :]
    # Filter out None values for kernel call (only tensor indices)
    # Must be done AFTER broadcast_indices, as broadcast may create new tensors
    tensor_indices = [idx for idx in indices if idx is not None]
    if not tensor_indices:
        raise ValueError("At least one non-None index tensor is required")

    if values.device != inp.device:
        values = values.to(inp.device)
    values = torch.broadcast_to(values, target_shape)

    _index_put_func(inp, tensor_indices, values, accumulate)
    return inp
