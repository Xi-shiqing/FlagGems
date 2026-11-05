"""Reuse the existing 8192-product work bound independently of inner grouping."""
import ast,difflib,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=(HERE/'grouped64_mm.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='80708fa04ebb82becd7f2dd8879b626a1d8f0cd54c6871daf493de74da3efe06'
old='    desired = max(1, triton.cdiv(2 * get_sm_count(), output_blocks))'
new='    desired = max(1, triton.cdiv(2 * get_sm_count(), output_blocks), triton.cdiv(K, 8192))'
assert source.count(old)==1
modified=source.replace(old,new)
old='    return min(triton.next_power_of_2(desired), 1 << (maximum.bit_length() - 1))'
assert modified.count(old)==1;modified=modified.replace(old,'    return min(desired, maximum)')
a={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)};b={n.name:n for n in ast.parse(modified).body if isinstance(n,ast.FunctionDef)}
changed=[name for name,node in a.items() if ast.dump(node,include_attributes=False)!=ast.dump(b[name],include_attributes=False)]
assert changed==['_long_reduction_splitk_parts']
p=HERE/'grouped_work_mm.py'
if p.exists():assert p.read_text()==modified
else:p.write_text(modified)
(HERE/'grouped_work_mm.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile='a/src/flag_gems/ops/mm.py',tofile='b/src/flag_gems/ops/mm.py')))
result={'status':'PREPARED_WORK_BOUNDED_GROUPED_MM','candidate_sha256':hashlib.sha256(modified.encode()).hexdigest(),'changed_functions':changed,'all_kernel_AST_unchanged':True,'rule':'max(2 SM waves / output tiles, ceil(K/8192)), limited by 64 partials, minimum work and 64 MiB workspace','inner_group_products':1024,'production_source_changed':False}
(HERE/'grouped_work_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
