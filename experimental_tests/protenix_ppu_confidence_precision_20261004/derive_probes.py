"""Validate probe operands were captured from fully audited frozen model runs."""
import hashlib
import json
from pathlib import Path
import sys


def read(path):return json.loads(path.read_text())


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root.parent/'closure_20261003'))
    from derive_matrix import segments
    baseline=read(root/'derived.json')
    by_case={c['case']:c for c in baseline['cases'] if c['model']=='protenix_base_default_v0.5.0'}
    result={'status':'PASS','cases':[]}
    for case,previous in by_case.items():
        operand_hashes=[]
        for stage in ('distance_probe','fused_probe','rne_probe'):
            run=root/stage/case/'protenix_base_default_v0.5.0'
            assert (run/'exit_code').read_text().strip()=='0'
            pair=read(run/'paired.json')
            assert pair['status']=='PASS' and len(pair['summary'])==1
            mode,summary=next(iter(pair['summary'].items()))
            for field in ('prediction_digest','five_confidence_metrics'):
                assert summary[field]==previous['summary'][mode][field]
            assert list(pair['checkpoint_sha256'].values())==previous['checkpoint_sha256']
            feature,atoms,assets=[read(run/name) for name in ('feature_summary.json','atom_identity.json','input_assets.json')]
            assert digest(feature['tensors'])==previous['feature_tensor_manifest_sha256']
            assert atoms['sha256']==previous['atom_identity_sha256']
            assert assets['canonical_input_sha256']==previous['canonical_input_sha256']
            assert sorted(x['file_sha256'] for x in assets['assets'].values() if 'file_sha256' in x)==previous['asset_hashes']
            events=segments(run/'rng_trace.jsonl')
            assert len(events)==5 and digest(events[0])==previous['rng']['initialization_semantic_sha256']
            assert all(digest(e)==previous['rng']['prediction_semantic_sha256'] for e in events[1:])
            file={'distance_probe':'distance_probe.json','fused_probe':'fused_probe.json','rne_probe':'rne_probe.json'}[stage]
            records=read(run/file)['records']
            assert len(records)==1
            operand_hashes.append(records[0]['input_sha256'])
            if stage=='rne_probe':
                fused=records[0]['modes']['fused_fp32']
                assert fused['changed_pair_bins_vs_fp64']==0 and fused['max_self_distance_A']==0
                assert fused['max_abs_error_A']<.001
            result['cases'].append({'case':case,'stage':stage,'input_and_rng_audit':'PASS',
                'actual_operand_sha256':records[0]['input_sha256']})
        assert len(set(operand_hashes))==1
    (root/'probe_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('REAL_DISTANCE_PROBE_AUDITS_PASS',len(result['cases']))


if __name__=='__main__':main()
