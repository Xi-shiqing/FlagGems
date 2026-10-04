"""Resolve output-hash differences using reproducible CPU arithmetic only."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent
tree = ast.parse((root/'bench_tf32_training.py').read_text())
namespace = {'np': np, 'hashlib': hashlib}
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef)
                             and x.name in ('_array', '_inputs', '_sha')], type_ignores=[]),
             'shared_operand_generators', 'exec'), namespace)
results = json.loads((root/'operators.json').read_text())
rows = []
for case in results['cases'][:5]:
    arrays = namespace['_inputs'](case['kind'], case['shape'], 101)
    a, b = arrays['a'], arrays['weight']
    value = a*b if case['kind'] == 'bmm' else a*b.T
    if case['kind'] == 'addmm':
        value = value+arrays['bias']
    direct = namespace['_sha'](value)
    negative_zeros = int(np.count_nonzero((value == 0) & np.signbit(value)))
    value[value == 0] = 0.
    positive_zero = namespace['_sha'](value)
    mode = 'candidate_tf32' if results['server'] == 'ppu' else 'tf32'
    actual = case['modes'][mode]['errors_vs_cpu_fp64']['output']['sha256']
    assert actual in (direct, positive_zero), case['name']
    rows.append({'name': case['name'], 'actual_sha256': actual,
                 'numpy_ieee_product_sha256': direct, 'canonical_zero_product_sha256': positive_zero,
                 'numpy_product_negative_zero_count': negative_zeros,
                 'matches_ieee_product': actual == direct,
                 'matches_positive_zero_product': actual == positive_zero})
(root/'signed_zero.json').write_text(json.dumps({'status': 'PASS', 'server': results['server'],
    'scope': 'Only CPU synthetic benchmark operands reconstructed; no inference or GPU timing.',
    'cases': rows}, indent=2)+'\n')
print('SIGNED_ZERO_CHECK_PASS', results['server'], rows)
