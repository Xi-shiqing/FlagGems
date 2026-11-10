"""Check the public PPU gating APIs against separate Gems and CPU FP64."""
import itertools
import inspect
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

import flag_gems

pytestmark = pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="PPU FP32 fused activation API")
CASES = [
    ("matrix", (3, 7), (3, 7), (3, 7), "dense"),
    ("broadcast", (1, 3, 1), (2, 3, 7), (7,), "dense"),
    ("shift_expansion", (1, 3, 1), (1, 3, 7), (2, 1, 7), "dense"),
    ("scalar", (), (2, 3, 7), (), "dense"),
    ("all_scalar", (), (), (), "dense"),
    ("transpose", (3, 7), (3, 7), (3, 7), "transpose"),
    ("expanded", (2, 3, 7), (2, 3, 7), (2, 3, 7), "expand"),
    ("offset", (3, 7), (3, 7), (3, 7), "offset"),
    ("empty", (1, 7), (0, 7), (7,), "dense"),
    ("dense_empty", (0,), (0,), (0,), "dense"),
    ("tail_513", (513,), (513,), (513,), "dense"),
    ("tail_1023", (1023,), (1023,), (1023,), "dense"),
    ("large_tail", (2049, 1024), (2049, 1024), (2049, 1024), "dense"),
    ("long_sum", (1, 1, 7), (3, 257, 7), (7,), "dense"),
]
PARAMS = []
for index, case in enumerate(CASES):
    for kind in ["sigmoid_mul", "sigmoid_mul_add", "silu_mul"]:
        count = 3 if kind == "sigmoid_mul_add" else 2
        masks = list(itertools.product([False, True], repeat=count)) if index == 0 else [
            (True,) * count, tuple(i == index % count for i in range(count))]
        PARAMS.extend((kind, tf32, case, mask) for tf32 in [False, True] for mask in masks)


@pytest.fixture(autouse=True)
def deterministic_gating():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def separate(kind, values):
    activation = F.silu(values[0]) if kind == "silu_mul" else torch.sigmoid(values[0])
    output = activation * values[1]
    return output + values[2] if kind == "sigmoid_mul_add" else output


def same_bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


def sample(shape, layout, generator):
    def random(size):
        return torch.randn(size, generator=generator, device="cpu", dtype=torch.float32).to(flag_gems.device)
    if not shape:
        return random((1,)).reshape(())
    if layout == "expand":
        return random(shape[:-1] + (1,)).expand(shape)
    if layout == "offset":
        return random(shape[:-1] + (2 * shape[-1] + 1,))[..., 1::2]
    if layout == "transpose" and len(shape) >= 2:
        return random(shape[:-2] + (shape[-1], shape[-2])).transpose(-1, -2)
    return random(shape)


@pytest.mark.sigmoid_mul
@pytest.mark.sigmoid_mul_add
@pytest.mark.silu_mul
@pytest.mark.parametrize("kind,tf32,case,mask", PARAMS)
def test_ppu_gated_activation(kind, tf32, case, mask):
    function = getattr(flag_gems, kind)
    expected_source = Path(flag_gems.__file__).resolve().parent / "runtime/backend/_thead/fused/gated_activation.py"
    assert Path(inspect.getsourcefile(function)).resolve() == expected_source
    assert function is getattr(sys.modules[function.__module__], kind)
    name, gate_shape, value_shape, shift_shape, layout = case
    generator = torch.Generator(device="cpu").manual_seed(101)
    shapes = [gate_shape, value_shape] + ([shift_shape] if len(mask) == 3 else [])
    inputs = [sample(shape, layout, generator) for shape in shapes]
    gradient = sample(tuple(torch.broadcast_shapes(*(x.shape for x in inputs))), layout, generator)
    before = [x.detach().cpu().contiguous().clone() for x in inputs]
    groups = [[x.detach().requires_grad_(needed) for x, needed in zip(inputs, mask)] for _ in range(2)]
    native = [x.detach().cpu().double().requires_grad_(needed) for x, needed in zip(inputs, mask)]
    previous = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        with flag_gems.use_gems():
            with torch.no_grad():
                same_bits(separate(kind, groups[0]), function(*groups[1]))
            expected, actual = separate(kind, groups[0]), function(*groups[1])
            same_bits(expected, actual)
            required = [i for i, needed in enumerate(mask) if needed]
            assert expected.requires_grad == actual.requires_grad == bool(required)
            gradients = [] if not required else [torch.autograd.grad(output, [values[i] for i in required], gradient)
                                                for output, values in [(expected, groups[0]), (actual, groups[1])]]
            if required:
                for a, b in zip(*gradients):
                    same_bits(a, b)
        native_output = separate(kind, native)
        torch.testing.assert_close(actual.detach().cpu().double(), native_output.detach(), atol=5e-6, rtol=5e-5)
        if required:
            native_gradients = torch.autograd.grad(native_output, [native[i] for i in required], gradient.cpu().double())
            for a, b in zip(gradients[1], native_gradients):
                torch.testing.assert_close(a.detach().cpu().double(), b, atol=5e-6, rtol=5e-5)
        for value, original in zip(inputs, before):
            same_bits(value, original)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
        torch.backends.cudnn.allow_tf32 = previous_cudnn
