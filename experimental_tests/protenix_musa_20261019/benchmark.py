"""Repeat full-budget prediction; profiling is a separate mode and dataset."""
import copy
import hashlib
import gzip
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch_musa

AMP_ALIGNED=os.environ.get('MUSA_AMP_ALIGNMENT','0')=='1'
if AMP_ALIGNED:
    from amp_compat import install
    install()
from runner import inference
if os.environ.get("SAFE_TRI_MUL", "0") == "1":
    from triangle_safe import install as install_safe_triangle
    install_safe_triangle()
import faulthandler
faulthandler.enable()
if os.environ.get("ATTENTION_COMPAT", "none") != "none":
    from attention_compat import install as install_attention_compat
    install_attention_compat()

ROOT = Path(os.environ['EXPERIMENT_ROOT'])
MODE = os.environ.get('EXPERIMENT_MODE', 'native')
GPU = os.environ['MUSA_VISIBLE_DEVICES']
# Runtime visibility order differs from host mthreads-gmi order; verified PCI IDs.
HOST_GPU = {0:0, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7}[int(GPU)]
REPEATS = int(os.environ.get('BENCH_REPEATS', '6'))
out = ROOT / 'results' / MODE / ('gpu_' + GPU)
out.mkdir(parents=True, exist_ok=True)
assert torch.musa.is_available() and torch.musa.device_count() == 1
free, total = torch.musa.mem_get_info()
if total-free > 1024**3:
    raise RuntimeError('Selected GPU already occupied; refusing launch')
torch.backends.mudnn.allow_tf32 = os.environ.get('MUSA_TF32','false').lower()=='true'
torch.backends.cuda.matmul.allow_tf32 = torch.backends.mudnn.allow_tf32


def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def feature_manifest(obj, path=''):
    rows = []
    if isinstance(obj, torch.Tensor):
        a = obj.detach().cpu().contiguous()
        raw = a.reshape(-1).view(torch.uint8).numpy().tobytes()
        rows.append(dict(path=path, shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(raw).hexdigest()))
    elif isinstance(obj, dict):
        for k, v in sorted(obj.items()):
            rows.extend(feature_manifest(v, path + '/' + str(k)))
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            rows.extend(feature_manifest(v, path + '/' + str(i)))
    return rows


original = inference.InferenceRunner.predict


def benchmark(self, data):
    name = data['sample_name']
    folder = out / name
    folder.mkdir(exist_ok=True)
    audit={};audit_handles=[]
    sync_handles=[];sync_count=[0]
    sync_boundary=os.environ.get('MUSA_SYNC_BOUNDARY','none')
    if sync_boundary=='cycle' and int(data['N_token'])>1024:
        def synchronize_cycle(module,args,result):
            torch.musa.synchronize();sync_count[0]+=1
        sync_handles.append(self.model.pairformer_stack.register_forward_hook(synchronize_cycle))
    if os.environ.get('PRECISION_AUDIT','0')=='1':
        for label,prefix in [('trunk','pairformer_stack.'),('diffusion','diffusion_module.'),('confidence','confidence_head.')]:
            for module_name,module in self.model.named_modules():
                if module_name.startswith(prefix) and isinstance(module,torch.nn.Linear) and (label!='trunk' or module_name.endswith('pair_transition.linear_no_bias_a')):
                    def observe(mod,args,result,_label=label,_name=module_name):
                        if _label not in audit:
                            audit[_label]=dict(module=_name,musa_autocast_enabled=torch.is_autocast_enabled('musa'),
                                input_dtype=str(args[0].dtype),output_dtype=str(result.dtype))
                    audit_handles.append(module.register_forward_hook(observe));break
    frozen_mode = os.environ.get('FROZEN_MODE')
    if frozen_mode:
        seed = int(self.configs.seeds[0])
        frozen = ROOT/'frozen_features'/f'seed_{seed}'/(name+'.pt.gz')
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen_mode == 'read' and not frozen.exists() and MODE == 'aligned_native':
            frozen_mode = 'write'  # First raw-feature preparation for a previously unvisited input.
        if frozen_mode == 'write' and not frozen.exists():
            temp = frozen.with_name(frozen.name+f'.{os.getpid()}.tmp')
            with gzip.open(temp, 'wb', compresslevel=1) as stream:
                torch.save(data, stream)
            try:os.link(temp,frozen)
            except FileExistsError:pass
            finally:temp.unlink()
        if not frozen.exists():
            raise FileNotFoundError('Frozen features not prepared: '+str(frozen))
        with gzip.open(frozen, 'rb') as stream:
            common = torch.load(stream, map_location='cpu', weights_only=False)
        assert common['sample_name'] == name
        for field in ('N_token', 'N_atom', 'N_msa'):
            assert torch.equal(common[field], data[field]), field
        # Ensure frozen atom metadata still matches the dumper's atom order.
        for field in ('atom_to_token_idx', 'ref_element', 'ref_atom_name_chars', 'asym_id', 'residue_index'):
            if field in common['input_feature_dict'] and field in data['input_feature_dict']:
                assert torch.equal(common['input_feature_dict'][field], data['input_feature_dict'][field]), field
        data = common
        inference.seed_everything(seed=seed, deterministic=self.configs.deterministic)
    fusion_policy=os.environ.get('FUSION_POLICY','inplace')
    if int(data['N_token'])>=200 and ('fused' in MODE or fusion_policy in ('hybrid','separate')) and not getattr(self, '_bias_fusion_installed', False):
        if fusion_policy=='inplace':
            from bias_softmax import install
            install(self.model)
        else:
            from bias_softmax_storage import hybrid,separate
            from attention_compat import wrap
            from protenix.model.triangular import layers
            layers._attention=wrap(hybrid if fusion_policy=='hybrid' else separate)
        self._bias_fusion_installed = True
    write(folder/'config.json', self.configs.to_dict())
    write(folder/'features.json', feature_manifest(data['input_feature_dict']))
    saved = copy.deepcopy(data)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.musa.get_rng_state_all())
    rows = []
    ref = None
    pred = None
    stages = defaultdict(list)
    originals = {}
    uninstall = None
    if MODE.startswith('profile'):
        from profiling import install
        uninstall = install(folder)
        uninstall.register(self.model)
        # The wrapped methods are disjoint in the model inference path.
        for method in ('get_pairformer_output', 'sample_diffusion', 'run_confidence_head'):
            func = getattr(self.model, method)
            originals[method] = func
            def measured(*args, _func=func, _name=method, **kwargs):
                torch.musa.synchronize()
                started = time.perf_counter()
                result = _func(*args, **kwargs)
                torch.musa.synchronize()
                stages[_name].append(time.perf_counter()-started)
                return result
            setattr(self.model, method, measured)
    try:
        for i in range(REPEATS+1):
            pred = None
            batch = copy.deepcopy(saved)
            random.setstate(rng[0]); np.random.set_state(rng[1])
            torch.set_rng_state(rng[2]); torch.musa.set_rng_state_all(rng[3])
            stages.clear()
            torch.musa.synchronize(); torch.musa.reset_peak_memory_stats()
            started = time.perf_counter()
            pred = original(self, batch)
            torch.musa.synchronize()
            seconds = time.perf_counter()-started
            row = dict(pdb=name, iteration=i, phase='warmup' if i == 0 else 'measured',
                       predict_seconds=seconds, physical_gpu=HOST_GPU, visibility_selection_index=GPU, mode=MODE,
                       n_token=int(data['N_token'].item()), n_atom=int(data['N_atom'].item()),
                       n_msa=int(data['N_msa'].item()),
                       peak_allocated_bytes=torch.musa.max_memory_allocated(),
                       peak_reserved_bytes=torch.musa.max_memory_reserved())
            a = pred['coordinate'].detach().float().cpu().numpy()
            row.update(coordinates_finite=bool(np.isfinite(a).all()), coordinate_shape=list(a.shape),
                       coordinate_sha256=hashlib.sha256(a.tobytes()).hexdigest())
            if 'fused' in MODE or fusion_policy in ('hybrid','separate'):
                from bias_softmax import hits
                row['fusion_hits_cumulative'] = dict(hits)
            row['frozen_features'] = bool(frozen_mode)
            row['fusion_policy']=fusion_policy
            row['fusion_active']=bool(getattr(self,'_bias_fusion_installed',False))
            if fusion_policy in ('hybrid','separate'):
                from bias_softmax_storage import hits as storage_hits
                row['storage_hits_cumulative']=dict(storage_hits)
            row['triangle_execution_policy']='out_of_place_large' if os.environ.get('SAFE_TRI_MUL','0')=='1' else 'original'
            row['sync_boundary']=sync_boundary
            row['sync_points_cumulative']=sync_count[0]
            row['attention_compat']=os.environ.get('ATTENTION_COMPAT','none')
            row['attention_batch_cap']=int(os.environ.get('ATTENTION_BATCH_CAP','64'))
            row['amp_alignment']=AMP_ALIGNED
            row['mudnn_tf32']=torch.backends.mudnn.allow_tf32
            if i==0 and audit_handles:
                write(folder/'precision_audit.json',dict(observed=audit,amp_alignment=AMP_ALIGNED,
                    skip_amp=self.configs.skip_amp.to_dict(),mudnn_tf32=torch.backends.mudnn.allow_tf32))
                for handle in audit_handles:handle.remove()
                audit_handles=[]
                if AMP_ALIGNED:
                    for label,config_key in [('diffusion','sample_diffusion'),('confidence','confidence_head')]:
                        if getattr(self.configs.skip_amp,config_key):
                            assert label in audit and audit[label]['output_dtype']=='torch.float32' and not audit[label]['musa_autocast_enabled'], label
            if ref is None:
                ref = a.copy()
            row['max_abs_coordinate_difference_from_warmup'] = float(np.max(np.abs(a-ref)))
            if MODE.startswith('profile'):
                row['stages_seconds'] = {k: sum(v) for k, v in stages.items()}
                row['other_seconds'] = seconds-sum(row['stages_seconds'].values())
            if i > 0:
                np.savez_compressed(folder/f'coordinates_{i}.npz', coordinate=a)
            rows.append(row)
            write(folder/'measurements.json', dict(scope='synchronized predict including H2D; excludes featurization, loading, copies, RNG reset, checks and file writing',
                profile_instrumented=MODE.startswith('profile'), torch=torch.__version__, torch_musa=torch_musa.__version__,
                gpu=torch.musa.get_device_name(), mudnn_tf32=torch.backends.mudnn.allow_tf32,
                amp_alignment=AMP_ALIGNED,requested_repeats=REPEATS,attention_compat=os.environ.get('ATTENTION_COMPAT','none'),sync_boundary=sync_boundary,fusion_policy=fusion_policy,triangle_execution_policy='out_of_place_large' if os.environ.get('SAFE_TRI_MUL','0')=='1' else 'original',rows=rows))
            print('BENCH '+json.dumps(row), flush=True)
            if not row['coordinates_finite']:
                raise RuntimeError('Nonfinite coordinates')
            del batch
    finally:
        for handle in sync_handles:handle.remove()
        for handle in audit_handles:handle.remove()
        if uninstall:
            uninstall()
        for method, func in originals.items():
            setattr(self.model, method, func)
    return pred


if __name__ == '__main__':
    inference.InferenceRunner.predict = benchmark
    inference.run()
