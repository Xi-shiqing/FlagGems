"""Check ordinary TF32 bitwise equality; benchmark original-server activations."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captures', nargs='*', type=Path, default=[])
    args = parser.parse_args()
    mm = importlib.import_module('flag_gems.ops.mm')
    layout = importlib.import_module('flag_gems.ops.mm_preserve_layout')
    old = mm._direct_mm
    new = layout.make_direct(mm)
    rng = np.random.default_rng(101)
    def tensor(shape):
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).cuda()
    rows = []
    with torch.inference_mode():
        torch.backends.cuda.matmul.allow_tf32 = True
        for m,k,n in [(17,97,129),(2049,97,129),(131073,128,256),(4097,384,512)]:
            a,b = tensor((m,k)),tensor((n,k)).T
            c = torch.empty((m,n), device='cuda')
            old(a,b,c,m,n,k)
            reference = c.clone()
            layout.AUDIT = True
            layout.CALLS.clear()
            new(a,b,c,m,n,k)
            assert torch.equal(reference,c), (m,k,n)
            rows.append({'shape':[m,k,n],'bitwise_equal':True,'prepared':bool(layout.CALLS)})
        assert any(r['prepared'] for r in rows)
        real = []
        seen = set()
        for capture in args.captures:
            payload = json.loads((capture/'activation_capture.json').read_text())
            for slot,meta in payload['selected'].items():
                if meta['kind'] != 'mm' or meta['input_dtypes'] != ['torch.float32']*2:
                    continue
                tensors=[]
                for record in meta['tensors']:
                    value=torch.load(capture/record['file'], map_location='cpu', weights_only=True)
                    assert hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==record['sha256']
                    tensors.append(value.cuda())
                a,b=tensors
                if not mm._should_direct(a,b) or a.shape[0] < 2048:
                    continue
                signature=(tuple(a.shape),tuple(b.shape),tuple(a.stride()),tuple(b.stride()))
                if signature in seen:
                    continue
                seen.add(signature)
                outputs, times = {}, {}
                for mode in ('fp32','tf32','tf32_layout'):
                    torch.backends.cuda.matmul.allow_tf32 = mode != 'fp32'
                    mm._direct_mm = new if mode=='tf32_layout' else old
                    layout.AUDIT = False
                    result=mm.mm(a,b)
                    outputs[mode]=result
                    for _ in range(3):mm.mm(a,b)
                    samples=[]
                    for _ in range(15):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record();mm.mm(a,b);end.record();end.synchronize()
                        samples.append(begin.elapsed_time(end))
                    times[mode]={'median_ms':statistics.median(samples),'samples_ms':samples}
                assert torch.equal(outputs['tf32'],outputs['tf32_layout']), (capture,slot)
                real.append({'capture':str(capture),'slot':slot,'MKN':[a.shape[0],a.shape[1],b.shape[1]],
                             'input_sha256':[r['sha256'] for r in meta['tensors']],
                             'times':times,'bitwise_equal_current_tf32':True})
                mm._direct_mm=old
                torch.backends.cuda.matmul.allow_tf32=True
        assert real, 'No eligible real model activations were exercised'
    layout.AUDIT=False
    mm._direct_mm=old
    Path(__file__).with_name('mm_layout_checks.json').write_text(json.dumps(
        {'status':'PASS','controls':rows,'real_activations':real,
         'scope':'PPU-only real activations retained on PPU. Warmed CUDA event times include public MM allocation and RHS preparation.'},indent=2)+'\n')
    print('MM_LAYOUT_CHECK_PASS',len(rows),len(real),[(r['MKN'],r['times']['tf32']['median_ms'],r['times']['tf32_layout']['median_ms']) for r in real],flush=True)


if __name__ == '__main__':
    main()
