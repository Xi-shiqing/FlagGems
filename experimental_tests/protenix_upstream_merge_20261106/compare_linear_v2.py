"""Separate FP32/TF32 tuning and compare complete Linear calls after merging."""
from pathlib import Path

source = Path(__file__).with_name('compare_merged.py').read_text()
exec(compile(source.split('for shape, layout in ')[0], __file__, 'exec'), globals())

variants = ['ours', 'upstream', 'upstream_ppu', 'merged_v2']
for variant, path in [('upstream_ppu', here/'packages/upstream/flag_gems/runtime/backend/_thead/ops/linear.py'),
                      ('merged_v2', here/'linear_merged_v2.py')]:
    key = 'merge_linear_precision_' + variant
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    modules[variant] = {'linear': module}
    report['source_hashes'][variant] = {'linear': hashlib.sha256(path.read_bytes()).hexdigest()}

base_compare = compare
def compare(*args, **kwargs):
    base_compare(*args, **kwargs)
    for variant in variants:
        tuner = modules[variant]['linear'].linear_kernel.fn
        best = getattr(tuner, 'best_config', None)
        if best is not None:
            report['cases'][-1]['variants'][variant]['last_best_config'] = best.all_kwargs()
    save()

exec(compile(source[source.index('for m, k, n in '):], __file__, 'exec'), globals())
