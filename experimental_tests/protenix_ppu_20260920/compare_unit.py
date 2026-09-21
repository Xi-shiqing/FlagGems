from pathlib import Path
import json,numpy as np
r=Path(__file__).resolve().parent;p=json.loads((r/'unit_ppu.json').read_text());h=json.loads((r/'unit_h100.json').read_text());diff=[];bits=0
assert p['cases']==h['cases']==215
for a,b in zip(p['rows'],h['rows']):
 assert {k:v for k,v in a.items() if k in ['M','N','offset','kind','special','input_sha256','gradient_sha256']}=={k:v for k,v in b.items() if k in ['M','N','offset','kind','special','input_sha256','gradient_sha256']}
 x=np.asarray(a['values'],np.float32);y=np.asarray(b['values'],np.float32);err=float(np.max(np.abs(x.astype(float)-y.astype(float))));same=np.array_equal(x.view('uint32'),y.view('uint32'));bits+=int(same)
 fe=abs(a['flat']-b['flat']) if 'flat' in a else 0
 if 'flat' in a:assert np.array_equal(np.asarray([a['flat']],np.float32).view('uint32'),np.asarray([b['flat']],np.float32).view('uint32'))
 diff.append(dict(case={k:a[k] for k in ['M','N','offset','kind','special'] if k in a},max_abs=err,flat_abs=fe,bitwise=bool(same)))
summary={'cases':215,'max_abs':max(x['max_abs'] for x in diff),'max_flat_abs':max(x['flat_abs'] for x in diff),'bitwise':bits,'strict_1e7':sum(max(x['max_abs'],x['flat_abs'])<=1e-7 for x in diff),'unaffected_bitwise':all(x['old_new_bitwise'] for x in p['rows'] if 'old_new_bitwise' in x and x['block_x']!=256),'rows':diff}
(r/'unit_comparison.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='rows'})
for x in p['rows']:
 if x.get('M')==1 and x.get('N')==1973 and x.get('kind')=='ones' and x.get('offset')==0:print('FIXED_ONES',x)
print('PPU_TIMING',[(x['M'],x['N'],x['mode'],x['median_us']) for x in p['timings']])
assert summary['strict_1e7']==215 and summary['unaffected_bitwise']
