"""Keep few split-K partials; shorten accumulation inside each work item."""
import ast,difflib,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
base=Path('/mnt/cpfs/xishiqing/ppu_full_acceptance_20261005/precision_residual_20261019/gems_long_reduction_splitk_v2/flag_gems/ops/mm.py')
original=base.read_text()
assert hashlib.sha256(original.encode()).hexdigest()=='6c3c087f6760a921759e0721144e1a2312c4b04de8e44797ccbf6bb6618fa1b9'
node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='mm_kernel_h100_fp32_splitk')
body=ast.get_source_segment(original,node)
start=body.index('    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n')
end=body.index('    offset = ',start)
inner=body[start:end]
lines=inner.splitlines(True)
assert lines[1]=='    for offset_k in range(0, chunk_k, BLOCK_K):\n'
replacement='''    total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for group_start in range(0, chunk_k, 1024):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        group_end = tl.minimum(group_start + 1024, chunk_k)
        for offset_k in range(group_start, group_end, BLOCK_K):
'''+''.join('    '+line for line in lines[2:])+'''        total += acc
    acc = total
'''
changed_body=(body[:start]+replacement+body[end:]).replace('def mm_kernel_h100_fp32_splitk(', 'def mm_kernel_grouped_splitk(', 1)
# Keep every existing kernel intact, including the one used by short-K routes.
modified=original+'\n\n@libentry()\n@triton.jit\n'+changed_body+'\n'
route=next(n for n in ast.parse(modified).body if isinstance(n,ast.FunctionDef) and n.name=='_long_reduction_splitk_mm')
route_body=ast.get_source_segment(modified,route)
assert route_body.count('mm_kernel_h100_fp32_splitk[')==1
modified=modified.replace(route_body,route_body.replace('mm_kernel_h100_fp32_splitk[','mm_kernel_grouped_splitk[',1),1)
# Parallelism determines workspace; inner grouping controls summation depth.
modified=modified.replace('    desired = triton.cdiv(2 * get_sm_count(), output_blocks)',
                          '    desired = max(1, triton.cdiv(2 * get_sm_count(), output_blocks))')
modified=modified.replace('    if desired <= 1 or maximum < 2:', '    if maximum < 1:')
original_functions={n.name:n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
new_functions={n.name:n for n in ast.parse(modified).body if isinstance(n,ast.FunctionDef)}
changed=[name for name,node in original_functions.items() if ast.dump(node,include_attributes=False)!=ast.dump(new_functions[name],include_attributes=False)]
assert changed==['_long_reduction_splitk_parts','_long_reduction_splitk_mm'],changed
assert set(new_functions)-set(original_functions)=={'mm_kernel_grouped_splitk'}
out=HERE/'grouped_mm.py'
if out.exists():assert out.read_text()==modified
else:out.write_text(modified)
patch=''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/src/flag_gems/ops/mm.py',tofile='b/src/flag_gems/ops/mm.py'))
(HERE/'grouped_mm.patch').write_text(patch)
manifest={'status':'PREPARED_GROUPED_MM','base_sha256':hashlib.sha256(original.encode()).hexdigest(),
          'candidate_sha256':hashlib.sha256(modified.encode()).hexdigest(),
          'changed_functions':changed,'added_functions':['mm_kernel_grouped_splitk'],'all_existing_kernel_AST_unchanged':True,
          'inner_group_products':1024,'maximum_partial_workspace_bytes':64*1024*1024,
          'input_rounding_and_dot_precision':'Unchanged FP32 or RNE TF32',
          'selection':'K>8192 float32 inputs, index and workspace guards; parallelism selects partials; one partial allowed',
          'production_source_changed':False}
(HERE/'grouped_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
