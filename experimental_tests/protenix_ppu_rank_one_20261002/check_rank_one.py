"""Check real registered entries, broadcast/strided outputs and beta=0 poison."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    extra = os.environ.get('PROTENIX_EXTRA_SITE_PACKAGES')
    if extra:
        sys.path.append(extra)
    import torch
    import flag_gems
    import flag_gems.ops.rank_one as helper
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    calls, rows = {}, []
    original = helper.rank_one
    active = [None]
    for name in ('mm', 'addmm', 'bmm'):
        module = importlib.import_module('flag_gems.ops.'+name)
        def traced(*a, _name=name, **kw):
            value = original(*a, **kw)
            if value is not None:
                key = active[0]+'_'+_name
                calls[key] = calls.get(key, 0)+1
            return value
        module.rank_one = traced
    for m, n in ((1, 1), (7, 13), (31, 33), (129, 17), (257, 128)):
        for layout in ('dense', 'stride', 'broadcast'):
            a = torch.arange(m, dtype=torch.float32).div(17).sub(.7).reshape(m, 1)
            b = torch.arange(n, dtype=torch.float32).div(19).sub(.9).reshape(1, n)
            if layout == 'stride':
                aa, bb = torch.empty((m, 3)), torch.empty((1, n*3))
                aa[:, :1], bb[:, ::3] = a, b
                a, b = aa[:, :1], bb[:, ::3]
            elif layout == 'broadcast':
                a, b = a[:1].expand(m, 1), b[:, :1].expand(1, n)
            a, b = a.cuda(), b.cuda()
            if layout == 'stride':
                a = torch.empty_strided(a.shape, (3, 1), device='cuda').copy_(a)
                b = torch.empty_strided(b.shape, (n*3, 3), device='cuda').copy_(b)
            with flag_gems.use_gems(exclude=[]):
                for kind in ('mm', 'addmm', 'bmm'):
                    active[0] = kind
                    bias = torch.arange(n, dtype=torch.float32, device='cuda').div(23)
                    aa, bb = (a[None].expand(3, -1, -1), b[None].expand(3, -1, -1)) if kind == 'bmm' else (a, b)
                    shape = (3, m, n) if kind == 'bmm' else (m, n)
                    gold = aa.cpu().double()*bb.cpu().double()
                    if kind == 'addmm':
                        gold = gold*(-.75)+bias.cpu().double()*.5
                    # Column-major views exercise full writes and output strides.
                    out = torch.full((*shape[:-2], n, m), float('nan'), device='cuda').transpose(-1, -2)
                    for use_out in (False, True):
                        kwargs = {'out': out} if use_out else {}
                        if kind == 'addmm':
                            value = torch.addmm(bias, aa, bb, alpha=-.75, beta=.5, **kwargs)
                        else:
                            value = getattr(torch, kind)(aa, bb, **kwargs)
                        actual = value.detach().cpu().double()
                        assert torch.isfinite(actual).all(), (kind, layout, m, n)
                        if kind == 'addmm':
                            assert torch.allclose(actual, gold, rtol=3e-6, atol=3e-7)
                        else:
                            assert torch.equal(actual.float(), gold.float()), (kind, layout)
                        if use_out:
                            assert value.data_ptr() == out.data_ptr()
                        rows.append({'kind': kind, 'shape': list(shape), 'layout': layout, 'out': use_out,
                                     'max_abs_vs_fp64': float((actual-gold).abs().max())})
                active[0] = 'addmm_beta0'
                for bias_shape in ((), (1,), (n,), (m, 1), (m, n)):
                    bias = torch.full(bias_shape, float('nan'), device='cuda')
                    value = torch.addmm(bias, a, b, beta=0)
                    assert torch.equal(value.cpu(), (a.cpu().double()*b.cpu().double()).float())
    assert all(calls.get(kind+'_'+kind, 0)>0 for kind in ('mm', 'addmm', 'bmm')), calls
    # The existing FP32 entry must never call the helper.
    before = dict(calls)
    torch.backends.cuda.matmul.allow_tf32 = False
    with flag_gems.use_gems(exclude=[]):
        torch.mm(a, b)
        torch.addmm(torch.zeros(n, device='cuda'), a, b)
        torch.bmm(a[None], b[None])
    assert calls == before
    torch.backends.cuda.matmul.allow_tf32 = True
    assert original(a.cpu(), b.cpu(), enabled=True) is None
    assert original(a.half(), b.half(), enabled=True) is None
    assert original(a, b, enabled=False) is None
    aliased_a = torch.ones((7, 1), device='cuda')
    assert original(aliased_a, torch.ones((1, 1), device='cuda'), aliased_a, enabled=True) is None
    overlapping_out = torch.zeros(1, device='cuda').expand(7, 13)
    assert original(torch.ones((7, 1), device='cuda'), torch.ones((1, 13), device='cuda'),
                    overlapping_out, enabled=True) is None
    # Force the int64 indexing branch on a small allocation, without allocating
    # a multi-gigabyte tensor solely to check address arithmetic.
    a64 = torch.arange(7, device='cuda', dtype=torch.float32).reshape(7, 1)
    b64 = torch.arange(13, device='cuda', dtype=torch.float32).reshape(1, 13)
    out64 = torch.empty((7, 13), device='cuda')
    helper._outer_product[(1,)](a64, b64, out64, None, 1, 1, 91, 7, 13,
                              0, 1, 0, 1, 0, 13, 1, 0, 0, False, True, 512,
                              num_warps=4, enable_fp_fusion=False)
    assert torch.equal(out64.cpu(), a64.cpu()*b64.cpu())
    bias64 = torch.arange(13, device='cuda', dtype=torch.float32)
    helper._outer_product[(1,)](a64, b64, out64, bias64, -.75, .5, 91, 7, 13,
                              0, 1, 0, 1, 0, 13, 1, 0, 1, True, True, 512,
                              num_warps=4, enable_fp_fusion=False)
    assert torch.equal(out64.cpu(), a64.cpu()*b64.cpu()*(-.75)+bias64.cpu()*.5)
    args.out.write_text(json.dumps({'status': 'PASS', 'entries': calls, 'cases': rows,
        'beta0_nan_bias_shapes_tested': 5*15, 'fp32_helper_calls': 0,
        'forced_int64_indexing': 'PASS',
        'scope': 'Torch mm/addmm/bmm and out variants; dense, strided, batch-broadcast operands; NaN output poison; beta0 NaN bias.'}, indent=2)+'\n')
    print('RANK_ONE_CHECK_PASS', len(rows), calls)


if __name__ == '__main__':
    main()
