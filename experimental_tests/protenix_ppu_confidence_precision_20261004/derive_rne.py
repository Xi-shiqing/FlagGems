"""Full-model fused geometry audit; no prediction tensors leave the server."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--server',choices=('ppu','h100'),required=True)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root.parent/'closure_20261003'))
    from derive_matrix import segments
    baseline=read(root/'derived.json')
    assert baseline['status']=='PASS'
    result={'status':'PARTIAL' if args.partial else 'PASS','server':args.server,'cases':[],
        'scope':'Generic fused geometric distance in ConfidenceHead only. All other model matmuls remain TF32; H100 triangle is custom RNE diagnostic. No native reference changed.'}
    for previous in baseline['cases']:
        run=root/'rne_matrix'/previous['case']/previous['model']
        if args.partial and (not (run/'exit_code').exists() or (run/'exit_code').read_text().strip()!='0'):
            continue
        assert (run/'exit_code').read_text().strip()=='0'
        pair=read(run/'paired.json')
        assert pair['status']=='PASS' and pair['physical_gpu']==previous['physical_gpu']
        assert list(pair['checkpoint_sha256'].values())==previous['checkpoint_sha256']
        summary=pair['summary']['tf32_confidence_rne']
        ref='tf32' if args.server=='ppu' else 'tf32_triangle_corrected'
        assert summary['prediction_digest']==previous['summary'][ref]['prediction_digest']
        feature,atoms,assets=[read(run/name) for name in ('feature_summary.json','atom_identity.json','input_assets.json')]
        assert digest(feature['tensors'])==previous['feature_tensor_manifest_sha256']
        assert atoms['sha256']==previous['atom_identity_sha256']
        assert assets['canonical_input_sha256']==previous['canonical_input_sha256']
        assert sorted(x['file_sha256'] for x in assets['assets'].values() if 'file_sha256' in x)==previous['asset_hashes']
        events=segments(run/'rng_trace.jsonl')
        assert digest(events[0])==previous['rng']['initialization_semantic_sha256']
        assert len(events)==5 and all(digest(e)==previous['rng']['prediction_semantic_sha256'] for e in events[1:])
        audit=read(run/'confidence_precision_audit.json')
        assert sum(audit['fused_distance_calls'].values())==4
        if args.server=='ppu':
            gems=read(run/'gems_entry_audit.json')
            assert gems['status']=='PASS' and gems['full_flag_gems_requested'] and gems['calls']['direct_mm']>0
        result['cases'].append({'case':previous['case'],'model':previous['model'],'N_token':previous['N_token'],
            'summary':summary,'physical_gpu':pair['physical_gpu'],'input_and_rng_audit':'PASS','coordinate_hash_unchanged':True,
            'confidence_precision_audit':audit,'triangle_calls_first_warmup':pair['triangle_calls_first_warmup']})
    if not args.partial:
        assert len(result['cases'])==18
        assert set((root/'rne_matrix.exit').read_text().split())=={'0'}
        result['worker_exit_markers']=(root/'rne_matrix.exit').read_text().strip()
    (root/('derived_rne_partial.json' if args.partial else 'derived_rne.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('FUSED_CONFIDENCE_AUDITS_COMPLETE',args.server,len(result['cases']))


if __name__=='__main__':main()
