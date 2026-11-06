"""Compare pinned operator sources in one unchanged PPU runtime; no model data export."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
here = Path(__file__).resolve().parent
out = Path(args.output)
out.mkdir(exist_ok=False)
# Editable installation redirection is removed only from this test process.
sys.meta_path[:] = [x for x in sys.meta_path if '_flag_gems_editable' not in getattr(x, '__module__', '')]
import torch
import torch.nn.functional as F
import triton
import flag_gems

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
gen = torch.Generator().manual_seed(20261106)
variants = ['ours', 'upstream', 'merged']
modules = {}
for variant in variants:
    modules[variant] = {}
    for name in ['copy', 'div', 'mean', 'linear', 'pad', 'sigmoid', 'silu', 'log_softmax', 'unfold_backward']:
        path = here / 'variants' / variant / (name + '.py')
        key = 'merge_compare_' + variant + '_' + name
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
        modules[variant][name] = module

report = {
    'torch': torch.__version__, 'triton': triton.__version__,
    'device': torch.cuda.get_device_name(), 'vendor': flag_gems.vendor_name,
    'runtime_package': flag_gems.__file__, 'seed': 20261106,
    'comparison': 'Pinned operator modules, identical existing FlagGems runtime utilities',
    'source_hashes': json.loads((here / 'variant_source_hashes_merged.json').read_text()),
    'cases': [],
}


def save():
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


def rand(shape):
    return torch.randn(shape, generator=gen)


def digest(tensor):
    x = tensor.detach().cpu().contiguous()
    return hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def error(actual, expected):
    actual = actual.detach().cpu()
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    a, b = actual.double(), expected.double()
    assert torch.equal(torch.isnan(a), torch.isnan(b)), 'NaN positions'
    assert torch.equal(torch.isposinf(a), torch.isposinf(b)), '+inf positions'
    assert torch.equal(torch.isneginf(a), torch.isneginf(b)), '-inf positions'
    finite = torch.isfinite(b)
    diff = a[finite] - b[finite]
    ref = b[finite]
    return {'difference_l2': diff.norm().item(), 'reference_l2': ref.norm().item(),
            'relative_l2': (diff.norm() / ref.norm().clamp_min(1e-300)).item(),
            'max_abs': diff.abs().max().item() if diff.numel() else 0.,
            'output_sha256': digest(actual)}


def compare(name, operation, reference, *, rtol=1e-4, atol=1e-5, exact=False, inputs=(), env=None):
    os.environ.update(env or {})
    row = {'name': name, 'inputs': [{'shape': list(x.shape), 'dtype': str(x.dtype),
                                   'stride': list(x.stride()), 'sha256': digest(x)} for x in inputs],
           'env': env or {}, 'variants': {}}
    ready = {}
    for variant in variants:
        record = row['variants'][variant] = {}
        try:
            fn = operation(modules[variant])
            with torch.no_grad():
                value = fn()
                torch.cuda.synchronize()
                expected = reference() if callable(reference) else reference
                record.update(error(value, expected))
                torch.testing.assert_close(value.detach().cpu(), expected, rtol=0 if exact else rtol,
                                           atol=0 if exact else atol, equal_nan=True)
                repeat = fn()
                torch.cuda.synchronize()
                record['repeat_equal'] = digest(repeat) == record['output_sha256']
                record['status'] = 'PASS'
                # Keep timing even if a path is numerically nondeterministic.
                ready[variant] = fn
        except Exception as exc:
            record.update(status='FAIL', exception=str(exc)[:1400], traceback=traceback.format_exc()[-4000:])
    samples = {v: [] for v in ready}
    wall = {v: [] for v in ready}
    for round_id in range(7):
        order = list(ready) if round_id % 2 == 0 else list(reversed(ready))
        for variant in order:
            fn = ready[variant]
            with torch.no_grad():
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                tick = time.perf_counter()
                start.record()
                for _ in range(3):
                    value = fn()
                end.record()
                end.synchronize()
                wall[variant].append((time.perf_counter() - tick) * 1000 / 3)
                samples[variant].append(start.elapsed_time(end) / 3)
    for variant in ready:
        row['variants'][variant].update(device_ms=samples[variant], median_ms=statistics.median(samples[variant]),
                                        wall_ms=wall[variant], wall_median_ms=statistics.median(wall[variant]))
    report['cases'].append(row)
    save()
    print('CASE', name, {v: (x['status'], x.get('median_ms')) for v, x in row['variants'].items()}, flush=True)


def semantics(name, action):
    row = {'name': name, 'variants': {}}
    state = gen.get_state()
    for v in variants:
        try:
            gen.set_state(state)
            action(modules[v])
            row['variants'][v] = {'status': 'PASS'}
        except Exception as exc:
            row['variants'][v] = {'status': 'FAIL', 'exception': str(exc)[:1800]}
    report['cases'].append(row)
    save()
    print('SEMANTICS', name, row['variants'], flush=True)


for shape, layout in [((4096, 256), 'dense'), ((480249, 64), 'dense'),
                      ((64, 127, 128), 'transpose'), ((600, 600, 8), 'transpose')]:
    cpu = rand(shape)
    if layout == 'transpose':
        cpu = cpu.transpose(0, 1)
    src = cpu.to('cuda'); dst = torch.empty(cpu.shape, device='cuda')
    compare('copy_' + str(shape) + '_' + layout,
            lambda mods, src=src, dst=dst: lambda: mods['copy'].copy_(dst, src), cpu.contiguous(),
            exact=True, inputs=(cpu,), env={'FLAG_GEMS_PPU_COPY_FLAT': '1', 'FLAG_GEMS_PPU_COPY_STRIDED': '1'})
    del src, dst, cpu

def copy_conversions(mods):
    src = torch.tensor([-32769., -129., -1., 0., 0.25, 127., 128., 255., 256., 32768.])
    for dtype in [torch.bool, torch.int8, torch.uint8, torch.int16]:
        dst = torch.empty(src.shape, dtype=dtype, device='cuda')
        result = mods['copy'].copy_(dst, src.to('cuda')).cpu()
        torch.testing.assert_close(result, src.to(dtype), rtol=0, atol=0)
semantics('copy_dtype_conversion', copy_conversions)

for n in [4093, 1048576]:
    a, b = rand((n,)), rand((n,)) + .1
    da, db = a.to('cuda'), b.to('cuda')
    compare('div_tensor_' + str(n), lambda m: lambda: m['div'].true_divide(da, db),
            (a.double() / b.double()).float(), inputs=(a, b), rtol=1e-6, atol=1e-7)
    compare('div_scalar_' + str(n), lambda m: lambda: m['div'].true_divide(da, 7.3),
            (a.double() / 7.3).float(), inputs=(a,), rtol=1e-6, atol=1e-7)

def scalar_trunc(mods):
    for a, b in [(7, 3), (-7, 3), (7, -3), (-7, -3), (7.5, 3.)]:
        got = mods['div'].trunc_divide(a, b)
        torch.testing.assert_close(got, torch.div(a, b, rounding_mode='trunc'), rtol=0, atol=0)
semantics('div_scalar_trunc', scalar_trunc)

for shape, dim in [((17, 31, 13), 1), ((64, 600, 8), 1), ((7, 13, 11), (0, 2)),
                   ((2, 0, 4), 1), ((0, 5), 1), ((0, 5), 0)]:
    a = rand(shape); da = a.to('cuda')
    compare('mean_' + str(shape) + '_' + str(dim), lambda m: lambda: m['mean'].mean_dim(da, dim),
            a.mean(dim), inputs=(a,), rtol=2e-5, atol=1e-6)

def mean_dtype(mods):
    for dtype in [torch.int64, torch.bool]:
        try:
            mods['mean'].mean(torch.ones((3,), dtype=dtype, device='cuda'))
        except RuntimeError:
            pass
        else:
            raise AssertionError('integer/bool mean must reject missing floating dtype')
semantics('mean_invalid_dtype', mean_dtype)

for shape, padding in [((17, 31), (2, 3)), ((2, 7, 13), (-2, 1, 3, 0)), ((600, 600, 8), (0, 8))]:
    a = rand(shape); da = a.to('cuda')
    compare('pad_' + str(shape), lambda m: lambda: m['pad'].pad(da, padding, value=.5),
            F.pad(a, padding, value=.5), inputs=(a,), exact=True)

def pad_grad(mods):
    with torch.enable_grad():
        x = rand((3, 7)).to('cuda').requires_grad_()
        y = mods['pad'].constant_pad_nd(x, (2, 1), 0.)
        assert y.requires_grad, 'forward lost autograd'
        got, = torch.autograd.grad(y, x, torch.ones_like(y))
        torch.testing.assert_close(got.cpu(), torch.ones((3, 7)), rtol=0, atol=0)
semantics('pad_autograd', pad_grad)

for opname in ['sigmoid', 'silu']:
    for shape, layout in [((127, 129), 'dense'), ((127, 129), 'transpose'), ((1048576,), 'dense')]:
        a = rand(shape) * 4
        if layout == 'transpose': a = a.T
        dy = rand(a.shape)
        da, ddy = a.to('cuda'), dy.to('cuda')
        ref = (torch.sigmoid(a.double()) if opname == 'sigmoid' else F.silu(a.double())).float()
        dref = ref.to('cuda')
        env = {'FLAG_GEMS_PPU_SIGMOID_FAST': '1', 'FLAG_GEMS_PPU_SILU_FAST': '1'}
        compare(opname + '_' + str(shape) + '_' + layout,
                lambda m, op=opname: lambda: getattr(m[op], op)(da), ref,
                inputs=(a,), rtol=2e-6, atol=1e-6, env=env)
        y = ref.double() if opname == 'sigmoid' else torch.sigmoid(a.double())
        backward = (dy.double() * y * (1-y) if opname == 'sigmoid'
                    else dy.double() * y * (1 + a.double() * (1-y))).float()
        compare(opname + '_backward_' + str(shape) + '_' + layout,
                lambda m, op=opname: lambda: (m[op].sigmoid_backward(ddy, dref) if op == 'sigmoid'
                                             else m[op].silu_backward(ddy, da)),
                backward, inputs=(a, dy), rtol=2e-4, atol=2e-6, env=env)

for shape, dim in [((71, 437), -1), ((17, 1025), -1), ((7, 13, 11), 1)]:
    a, dy = rand(shape), rand(shape); da, ddy = a.to('cuda'), dy.to('cuda')
    ref = torch.log_softmax(a.double(), dim).float(); do = ref.to('cuda')
    compare('log_softmax_' + str(shape), lambda m: lambda: m['log_softmax'].log_softmax(da, dim),
            ref, inputs=(a,), rtol=2e-6, atol=2e-6)
    backward = (dy.double() - torch.exp(ref.double()) * dy.double().sum(dim, keepdim=True)).float()
    compare('log_softmax_backward_' + str(shape),
            lambda m: lambda: m['log_softmax'].log_softmax_backward(ddy, do, dim, torch.float32),
            backward, inputs=(dy, ref), rtol=2e-5, atol=2e-6)

for shape, dim, size, step, dtype in [((2, 17, 7), 1, 5, 2, torch.float32),
                                    ((64, 256, 128), 1, 16, 1, torch.float32),
                                    ((2, 17, 7), 1, 5, 2, torch.float64),
                                    ((3, 31), 1, 7, 3, torch.float16)]:
    x = torch.zeros(shape, dtype=dtype, requires_grad=True)
    y = x.unfold(dim, size, step)
    dy = rand(y.shape).to(dtype); ddy = dy.to('cuda')
    ref, = torch.autograd.grad(y, x, dy)
    compare('unfold_backward_' + str(shape) + '_' + str(dtype),
            lambda m: lambda: m['unfold_backward'].unfold_backward(ddy, shape, dim, size, step),
            ref, inputs=(dy,), rtol=1e-12 if dtype == torch.float64 else 1e-3,
            atol=1e-12 if dtype == torch.float64 else 2e-3)

for m, k, n in [(1, 256, 384), (5, 1024, 384), (7, 17, 13), (32768, 256, 256), (360000, 256, 256)]:
    a, w, b = rand((m, k)), rand((n, k)), rand((n,))
    da, dw, db = a.to('cuda'), w.to('cuda'), b.to('cuda')
    ref = (a.double() @ w.double().T + b.double()).float()
    for precision in ['fp32', 'tf32'] if m >= 32768 else ['fp32']:
        compare('linear_' + str((m, k, n)) + '_' + precision,
                lambda mods: lambda: mods['linear'].linear(da, dw, db), ref,
                inputs=(a, w, b), rtol=2e-3 if precision == 'tf32' else 1e-4,
                atol=.06 if precision == 'tf32' else 5e-5,
                env={'FLAG_GEMS_PPU_FAST_FP32': '1' if precision == 'tf32' else '0',
                     'FLAG_GEMS_PPU_PACKED_LINEAR': '1'})
    del a, w, b, da, dw, db, ref

report['status'] = 'COMPLETE_WITH_PER_CASE_VERDICTS'
save()
print('COMPLETE_CONFLICT_SOURCE_COMPARISON', len(report['cases']), flush=True)
