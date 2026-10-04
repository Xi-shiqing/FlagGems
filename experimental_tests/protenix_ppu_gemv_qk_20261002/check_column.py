"""Real MM/out entries: vector orientation, strides, broadcasts and FP32 guard."""
import importlib
import json
import os
from pathlib import Path
import sys

if os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES'):
    sys.path.append(os.environ['PROTENIX_EXTRA_SITE_PACKAGES'])
import torch
import flag_gems
from controls import controls, switch

root = Path(__file__).resolve().parent
baseline = root.parent/'rank_one_20261002/candidate'
entries = controls(baseline)
module = importlib.import_module('flag_gems.ops.mm')
original, count = module.mm_gemv, [0]
def traced(*a, **kw):
    value = original(*a, **kw)
    if value is not None:
        count[0] += 1
    return value
module.mm_gemv = traced
torch.use_deterministic_algorithms(True)
rows = []
with flag_gems.use_gems(exclude=[]):
    for shape in ((1, 3, 1), (7, 31, 1), (1, 127, 1), (33, 257, 1), (1, 1025, 1)):
        m, k, n = shape
        for layout in ('dense', 'strided', 'transpose', 'broadcast'):
            a = torch.arange(m*k, dtype=torch.float32).reshape(m, k).remainder(53).sub(26).div(31).cuda()
            b = torch.arange(k*n, dtype=torch.float32).reshape(k, n).remainder(41).sub(20).div(37).cuda()
            if layout == 'strided':
                a = torch.empty_strided(a.shape, (k*3+1, 3), device='cuda').copy_(a)
                b = torch.empty_strided(b.shape, (n*3+1, 3), device='cuda').copy_(b)
            elif layout == 'transpose':
                a, b = a.T.contiguous().T, b.T.contiguous().T
            elif layout == 'broadcast':
                a, b = a[:, :1].expand(m, k), b[:1, :].expand(k, n)
            gold = a.cpu().double() @ b.cpu().double()
            torch.backends.cuda.matmul.allow_tf32 = True
            switch(entries, True)
            for use_out in (False, True):
                out = torch.full((max(m, n)*3,), float('nan'), device='cuda')[::3].reshape(m, n)
                actual = torch.mm(a, b, out=out) if use_out else torch.mm(a, b)
                value = actual.cpu().double()
                assert torch.isfinite(value).all()
                assert torch.allclose(value, gold, rtol=2e-5, atol=2e-6), (shape, layout)
                if use_out:
                    assert actual.data_ptr() == out.data_ptr()
                rows.append({'shape':list(shape), 'layout':layout, 'out':use_out,
                             'max_abs_vs_cpu_fp64':float((value-gold).abs().max())})
            torch.backends.cuda.matmul.allow_tf32 = False
            switch(entries, False)
            expected = torch.mm(a, b)
            switch(entries, True)
            before = count[0]
            actual = torch.mm(a, b)
            assert count[0] == before and torch.equal(actual.cpu(), expected.cpu())
    assert count[0] == 40, count
    assert original(a.half(), b.half(), enabled=True) is None
    assert original(a.cpu(), b.cpu(), enabled=True) is None
    assert original(a, b, enabled=False) is None
    assert original(torch.ones((1,7),device="cuda"),torch.ones((7,13),device="cuda"),enabled=True) is None
    assert original(a, b, a, enabled=True) is None
(root/'check.json').write_text(json.dumps({'status':'PASS','cases':rows,'gemv_entry_calls':count[0],
    'fp32_bitwise_unchanged':True, 'scope':'Column-vector MM results including single dot, general strides and zero-stride broadcasts; MM/out; NaN-poison output; unsupported inputs fall back.'}, indent=2)+'\n')
print('GEMV_CHECK_PASS',len(rows),count[0])
