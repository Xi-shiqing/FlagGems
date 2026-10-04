"""Poison allocation outputs, check layouts, broadcasting and inference guards."""
from pathlib import Path
import json
import numpy as np
import torch
import flag_gems
from flag_gems.utils import full_write_allocation as allocator


def main():
    torch.use_deterministic_algorithms(True)
    assert torch.utils.deterministic.fill_uninitialized_memory
    rng = np.random.default_rng(101)
    def tensor(shape, dtype=torch.float32):
        values = rng.integers(-32, 33, size=shape).astype(np.float32)/32
        return torch.from_numpy(values).to(device='cuda', dtype=dtype)
    def flat(value):
        return value if isinstance(value, tuple) else (value,)
    rows = []
    def check(label, operation):
        outputs = []
        for enabled in (False, True):
            allocator.ENABLED = enabled
            allocator.POISON = enabled
            with torch.inference_mode(), flag_gems.use_gems(exclude=[]):
                outputs.append(flat(operation()))
            allocator.ENABLED = allocator.POISON = False
        assert len(outputs[0]) == len(outputs[1])
        for before, after in zip(*outputs):
            assert torch.equal(before, after), label
            assert before.stride() == after.stride(), (label, before.stride(), after.stride())
            assert before.dtype == after.dtype and before.device == after.device
        rows.append({'case': label, 'bitwise_equal': True, 'layout_equal': True})
    allocator.AUDIT = True
    allocator.CALLS.clear()
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = tensor((17, 37), dtype)
        for name, value in [('contiguous', x), ('transpose', x.T), ('slice', x[:, ::2])]:
            check('mul_'+name+'_'+str(dtype), lambda v=value: torch.mul(v, v))
        a, b = tensor((17, 1, 37), dtype), tensor((1, 5, 1), dtype)
        check('broadcast_'+str(dtype), lambda: torch.add(a, b))
        check('compare_'+str(dtype), lambda: torch.gt(a, b))
    a, b = tensor((3, 17, 37)), tensor((1, 17, 1))
    check('where_broadcast', lambda: torch.where(a > 0, a, b))
    check('scalar', lambda: torch.mul(torch.tensor(3.0, device='cuda'), 2.0))
    check('empty', lambda: torch.add(torch.empty((0, 17), device='cuda'), 1))
    check('integer', lambda: torch.add(torch.arange(131, device='cuda'), 1))
    for shape, dim in [((17, 37), -1), ((3, 17, 37), 1), ((2, 4097), -1), ((0, 17), -1)]:
        value = tensor(shape)
        check('softmax_'+str(shape)+'_'+str(dim), lambda: torch.softmax(value, dim))
    value = tensor((262147,))
    check('sigmoid_flat_odd', lambda: torch.sigmoid(value))
    parts = [tensor((3, width, 17)) for width in (5, 7, 3, 11, 1)]
    check('cat_five_odd', lambda: torch.cat(parts, dim=1))
    parts = [p.transpose(0, 2) for p in parts]
    check('cat_five_transpose', lambda: torch.cat(parts, dim=1))
    import importlib
    triangle = importlib.import_module('flag_gems.ops.triangle_attention')
    for s in (37, 131):
        q, k, v = [tensor((1, 3, s, 4, 32)) for _ in range(3)]
        b1, b2 = tensor((1, 3, 1, 1, s)), tensor((1, 1, 4, s, s))
        check('triangle_'+str(s), lambda: triangle._streaming_triangle_attention_forward_impl(
            q,k,v,b1,b2,precision='tf32',save_lse=False)[0])
    assert allocator.CALLS['allocations'] > 0
    count = dict(allocator.CALLS)
    allocator.CALLS.clear()
    allocator.ENABLED = True
    with torch.enable_grad():
        output = allocator.empty((17, 37), device='cuda', dtype=torch.float32)
        assert not allocator.CALLS, 'Training allocation guard failed'
    allocator.ENABLED = allocator.AUDIT = False
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.utils.deterministic.fill_uninitialized_memory
    report = {'status':'PASS', 'rows': rows, 'poisoned': count,
              'training_allocator_fallback': True, 'global_deterministic_flags_unchanged': True}
    Path(__file__).with_name('operator_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print('FULL_WRITE_CHECK_PASS', len(rows), count, flush=True)


if __name__ == '__main__':
    main()
