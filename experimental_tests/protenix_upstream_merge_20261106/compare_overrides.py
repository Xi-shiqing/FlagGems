"""Check semantic overrides that do not produce textual Git conflicts."""
from pathlib import Path

source = Path(__file__).with_name('compare_conflicts.py').read_text()
prefix = source.split('for shape, layout in ')[0]
exec(compile(prefix, __file__, 'exec'), globals())

import importlib
import types

variants = ['ours_generic', 'ours_public', 'upstream_ppu']
modules = {v: {} for v in variants}
new_backend = here/'packages/upstream/flag_gems/runtime/backend/_thead/ops'
package = types.ModuleType('merge_upstream_ppu')
package.__path__ = [str(new_backend)]
sys.modules[package.__name__] = package
report['actual_function_sources'] = {}
for name in ['mm', 'bmm', 'addmm', 'linear', 'index_select_backward']:
    modules['ours_generic'][name] = importlib.import_module('flag_gems.ops.'+name)
    public = getattr(flag_gems, name)
    modules['ours_public'][name] = types.SimpleNamespace(**{name: public})
    modules['upstream_ppu'][name] = importlib.import_module(package.__name__+'.'+name)
    report['actual_function_sources'][name] = {
        v: str(Path(getattr(modules[v][name], name).__code__.co_filename)) for v in variants}
report['scope'] = 'Actual published public entries, generic entries, and upstream PPU overrides with merged generic fallbacks'
save()

for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    a, b = rand((33, 128)).to(dtype), rand((128, 64)).to(dtype)
    da, db = a.to('cuda'), b.to('cuda')
    compare('mm_override_' + str(dtype), lambda mods: lambda: mods['mm'].mm(da, db),
            (a.double() @ b.double()).to(dtype), inputs=(a, b), rtol=1e-4 if dtype == torch.float32 else .02,
            atol=3e-5 if dtype == torch.float32 else .03)
    a, b = rand((2, 33, 128)).to(dtype), rand((2, 128, 64)).to(dtype)
    da, db = a.to('cuda'), b.to('cuda')
    compare('bmm_override_' + str(dtype), lambda mods: lambda: mods['bmm'].bmm(da, db),
            (a.double() @ b.double()).to(dtype), inputs=(a, b), rtol=1e-4 if dtype == torch.float32 else .02,
            atol=3e-5 if dtype == torch.float32 else .03)

for m, k, n in [(1, 256, 384), (5, 1024, 384), (32, 256, 256), (32768, 256, 256)]:
    a, b, bias = rand((m, k)), rand((k, n)), rand((n,))
    da, db, dc = a.to('cuda'), b.to('cuda'), bias.to('cuda')
    compare('addmm_override_' + str((m, k, n)),
            lambda mods: lambda: mods['addmm'].addmm(dc, da, db),
            (a.double() @ b.double() + bias.double()).float(), inputs=(a, b, bias), rtol=1e-4, atol=5e-5)
    weight = b.T.contiguous(); dw = weight.to('cuda')
    compare('linear_override_' + str((m, k, n)),
            lambda mods: lambda: mods['linear'].linear(da, dw, dc),
            (a.double() @ weight.double().T + bias.double()).float(), inputs=(a, weight, bias),
            rtol=1e-4, atol=5e-5,
            env={'FLAG_GEMS_PPU_FAST_FP32': '0', 'FLAG_GEMS_PPU_PACKED_LINEAR': '1'})

torch.use_deterministic_algorithms(True)
for count in [8, 32]:
    # The trailing storage is a guard: incorrect loads beyond the logical input
    # remain within an allocated buffer and become observable numerical errors.
    storage = torch.full((count*2, 8), 777., dtype=torch.float32)
    storage[:count] = rand((count, 8))
    grad = storage[:count]
    index = torch.arange(count).remainder(7).sort().values
    device_storage = storage.to('cuda'); dg = device_storage[:count]; di = index.to('cuda')
    x = torch.zeros((7, 8), requires_grad=True)
    ref, = torch.autograd.grad(x.index_select(0, index), x, grad)
    compare('index_select_backward_override_' + str(count),
            lambda mods: lambda: mods['index_select_backward'].index_select_backward(dg, (7, 8), 0, di),
            ref, inputs=(grad, index), rtol=1e-5, atol=1e-6)

report['status'] = 'COMPLETE_WITH_PER_CASE_VERDICTS'
save()
print('COMPLETE_BACKEND_OVERRIDE_COMPARISON', len(report['cases']), flush=True)
