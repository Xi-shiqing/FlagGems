"""Bit-exact boolean compaction/gather, layouts, boundaries and backward."""
import inspect
from pathlib import Path

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="PPU boolean index")
DTYPES = [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32,
          torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64]


def bits(value):
    return value.contiguous().reshape(-1).view(torch.uint8)


def values(shape, dtype):
    value = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int64).reshape(shape)
    if dtype == torch.bool:
        return value % 3 == 0
    result = (value % 13 - 5).to(dtype)
    if result.numel():
        if dtype.is_floating_point:
            special = [-0.0, float("nan"), float("inf"), -float("inf"), torch.finfo(dtype).tiny / 2]
        else:
            special = [torch.iinfo(dtype).min, torch.iinfo(dtype).max]
        for i, item in enumerate(special[:result.numel()]):
            result.reshape(-1)[i] = item
    return result


def mask(shape, pattern):
    value = torch.arange(torch.tensor(shape).prod().item()).reshape(shape) % 3 != 0
    if pattern == "zero":
        value.zero_()
    elif pattern == "one":
        value.fill_(True)
    return value


def check(cpu, axis, selected):
    source = Path(flag_gems.__file__).resolve().parent / "runtime/backend/_thead/ops/index.py"
    assert Path(inspect.getsourcefile(flag_gems.index)).resolve() == source
    # Expanded strides need a view on a dense allocation instead of overlapping copy_.
    if any(step == 0 for step in cpu.stride()):
        dense_shape = tuple(1 if step == 0 else size for size, step in zip(cpu.shape, cpu.stride()))
        dense = cpu[tuple(slice(0, 1) if step == 0 else slice(None) for step in cpu.stride())]
        gpu = dense.contiguous().to(flag_gems.device).reshape(dense_shape).expand(cpu.shape)
    else:
        offset = cpu.storage_offset()
        span = 0 if not cpu.numel() else 1 + sum((n - 1) * step for n, step in zip(cpu.shape, cpu.stride()))
        storage = torch.empty(span + offset, dtype=cpu.dtype, device=flag_gems.device)
        gpu = storage.as_strided(cpu.shape, cpu.stride(), offset)
        gpu.copy_(cpu)
    indices_cpu = [None] * axis + [selected]
    selected_gpu = torch.empty_strided(selected.shape, selected.stride(), dtype=torch.bool, device=flag_gems.device)
    selected_gpu.copy_(selected)
    indices_gpu = [None] * axis + [selected_gpu]
    expected = torch.ops.aten.index.Tensor(cpu, indices_cpu)
    with torch.no_grad():
        direct = flag_gems.index(gpu, indices_gpu)
        with flag_gems.use_gems():
            actual = torch.ops.aten.index.Tensor(gpu, indices_gpu)
    for result in [direct, actual]:
        assert result.dtype == expected.dtype and result.shape == expected.shape
        assert result.is_contiguous() and torch.equal(bits(result.cpu()), bits(expected))
    assert torch.equal(bits(gpu.cpu()), bits(cpu))


@pytest.mark.index
@pytest.mark.parametrize("n", [0, 1, 7, 31, 129, 245, 693, 2529, 4096, 4097])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("pattern", ["zero", "one", "mixed"])
def test_ppu_boolean_index_length(n, dtype, pattern):
    check(values((n, 3), dtype), 0, mask((n,), pattern))


@pytest.mark.index
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", ["axis0", "axis1", "span"])
@pytest.mark.parametrize("pattern", ["zero", "one", "mixed"])
def test_ppu_boolean_index_dtype(dtype, layout, pattern):
    cpu = values((3, 7, 5), dtype)
    axis, shape = (0, (3,)) if layout == "axis0" else (1, (7,)) if layout == "axis1" else (1, (7, 5))
    check(cpu, axis, mask(shape, pattern))


@pytest.mark.index
@pytest.mark.parametrize("layout", ["rank1", "rank4", "transpose", "slice", "expand", "offset", "mask_stride", "rank5"])
@pytest.mark.parametrize("pattern", ["zero", "one", "mixed"])
def test_ppu_boolean_index_layout(layout, pattern):
    cpu = values((2, 3, 7, 5), torch.float32)
    axis, shape = 1, (3,)
    if layout == "rank1":
        cpu = cpu.reshape(-1); axis, shape = 0, (cpu.numel(),)
    elif layout == "transpose":
        cpu = cpu.transpose(1, 3); axis, shape = 1, (5,)
    elif layout == "slice":
        cpu = cpu[..., ::2]; axis, shape = 2, (7,)
    elif layout == "expand":
        cpu = cpu[:1].expand(4, 3, 7, 5)
    elif layout == "offset":
        cpu = cpu.reshape(-1)[5:205].reshape(4, 10, 5); axis, shape = 1, (10,)
    elif layout == "rank5":
        cpu = cpu.reshape(1, 2, 3, 7, 5); axis, shape = 2, (3,)
    selected = mask(shape, pattern)
    if layout == "mask_stride":
        selected = mask((shape[0] * 2,), pattern)[::2]
    check(cpu, axis, selected)


@pytest.mark.index
@pytest.mark.parametrize("tf32", [False, True])
def test_ppu_boolean_index_backward(tf32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    cpu = torch.arange(51, dtype=torch.float32).reshape(3, 17).requires_grad_()
    gpu = cpu.detach().to(flag_gems.device).requires_grad_()
    selected = torch.arange(17) % 3 != 0
    reference = torch.ops.aten.index.Tensor(cpu, [None, selected])
    reference.sum().backward()
    with flag_gems.use_gems():
        actual = torch.ops.aten.index.Tensor(gpu, [None, selected.to(flag_gems.device)])
        actual.sum().backward()
    assert torch.equal(bits(actual.detach().cpu()), bits(reference.detach()))
    assert torch.equal(bits(gpu.grad.cpu()), bits(cpu.grad))
