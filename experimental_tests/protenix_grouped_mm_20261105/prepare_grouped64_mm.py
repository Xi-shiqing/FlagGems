"""Increase data reuse for TF32 while preserving the 1024-product grouping."""
import ast,difflib,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=(HERE/'grouped_mm.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='3e88d3e11dda49988d81a955c91d3cd3b85dbe6a7dd7cdd8ca372292f3fab4e7'
modified=source
for name in ('_long_reduction_splitk_parts','_long_reduction_splitk_mm'):
    node=next(n for n in ast.parse(modified).body if isinstance(n,ast.FunctionDef) and n.name==name)
    old=ast.get_source_segment(modified,node);new=old
    if name=='_long_reduction_splitk_parts':
        anchor='    output_blocks = triton.cdiv(M, 32) * triton.cdiv(N, 32)'
        assert new.count(anchor)==1
        new=new.replace(anchor,'    tile = _long_reduction_tile(M, N)\n    output_blocks = triton.cdiv(M, tile) * triton.cdiv(N, tile)')
    else:
        anchor='    with torch_device_fn.device(a.device):'
        assert new.count(anchor)==1;new=new.replace(anchor,'    tile = _long_reduction_tile(M, N)\n'+anchor)
        new=new.replace('triton.cdiv(M, 32) * triton.cdiv(N, 32)','triton.cdiv(M, tile) * triton.cdiv(N, tile)')
        new=new.replace('BLOCK_M=32, BLOCK_N=32, BLOCK_K=32','BLOCK_M=tile, BLOCK_N=tile, BLOCK_K=32')
    modified=modified.replace(old,new,1)
modified+='\n\ndef _long_reduction_tile(M, N):\n    return 64 if _allow_tf32() and M >= 64 and N >= 64 else 32\n'
a={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
b={n.name:n for n in ast.parse(modified).body if isinstance(n,ast.FunctionDef)}
changed=[name for name,node in a.items() if ast.dump(node,include_attributes=False)!=ast.dump(b[name],include_attributes=False)]
assert changed==['_long_reduction_splitk_parts','_long_reduction_splitk_mm']
assert set(b)-set(a)=={'_long_reduction_tile'}
target=HERE/'grouped64_mm.py'
if target.exists():assert target.read_text()==modified
else:target.write_text(modified)
(HERE/'grouped64_mm.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile='a/src/flag_gems/ops/mm.py',tofile='b/src/flag_gems/ops/mm.py')))
manifest={'status':'PREPARED_TF32_DATA_REUSE_CANDIDATE','base_sha256':hashlib.sha256(source.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(modified.encode()).hexdigest(),'changed_functions':changed,'all_kernel_AST_unchanged':True,'rule':'64x64 output tile for TF32 M>=64 and N>=64; otherwise32x32; long-K guards and resource caps unchanged','production_source_changed':False}
(HERE/'grouped64_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
