"""Validate confidence-only ablation, including exact retained coordinates."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'closure_20261003'))
from derive_matrix import segments


def read(path):
    return json.loads(path.read_text())


def main():
    result = {'status':'PASS','cases':[], 'scope':'Six-case class-based ConfidenceHead matmul precision diagnostic on H100 corrected TF32; not a promoted PPU policy.'}
    for model in ('protenix_base_default_v0.5.0','protenix-v2','protenix_mini_esm_v0.5.0'):
        for case in ('G01','G02'):
            run, ref = [ROOT / part / case / model for part in ('confidence_matrix','corrected_matrix')]
            assert (run/'exit_code').read_text().strip()=='0'
            pair, baseline = read(run/'paired.json'),read(ref/'paired.json')
            assert pair['checkpoint_sha256']==baseline['checkpoint_sha256']
            current, previous = [x['summary']['tf32_triangle_corrected'] for x in (pair,baseline)]
            assert current['prediction_digest']==previous['prediction_digest'], 'confidence scope changed coordinates'
            for name in ('feature_summary.json','atom_identity.json','input_assets.json'):
                assert read(run/name)==read(ref/name)
            a,b=segments(run/'rng_trace.jsonl'),segments(ref/'rng_trace.jsonl')
            assert a==b
            audit=read(run/'confidence_precision_audit.json')
            assert audit['status']=='PASS' and all(v>0 for v in audit['actual_calls'].values())
            result['cases'].append({'case':case,'model':model,'input_and_rng_audit':'PASS',
                'ordered_rng_sha256':hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest(),
                'coordinate_hash_unchanged':True,'summary':current,'previous_corrected':previous,'precision_audit':audit})
    (ROOT/'derived_confidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CONFIDENCE_SCOPE_DERIVED_PASS',len(result['cases']))


if __name__=='__main__':main()
