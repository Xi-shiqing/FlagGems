"""Interleaved ABBA/BAAB full-budget inference with one loaded model."""
import copy
import gzip
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import benchmark as shared
from protenix.model.triangular import layers
from attention_compat import wrap
from bias_softmax import fused_attention

native_attention = layers._attention
optimized_attention = wrap(fused_attention)
order = ['native','fused','fused','native','fused','native','native','fused'] * 2
variants={'native':native_attention,'fused':optimized_attention}
experiment_description='Interleaved ABBA/BAAB; 8 measured calls per variant, same loaded model, seed and frozen input'


def crossover(self, data):
    name=data['sample_name'];seed=int(self.configs.seeds[0])
    folder=shared.out/name;folder.mkdir(exist_ok=True)
    frozen=shared.ROOT/'frozen_features'/f'seed_{seed}'/(name+'.pt.gz')
    with gzip.open(frozen,'rb') as f:
        common=torch.load(f,map_location='cpu',weights_only=False)
    for field in ('N_token','N_atom','N_msa'):
        assert torch.equal(data[field],common[field]),field
    for field in ('atom_to_token_idx','ref_element','ref_atom_name_chars','asym_id','residue_index'):
        if field in common['input_feature_dict']:
            assert torch.equal(data['input_feature_dict'][field],common['input_feature_dict'][field]),field
    shared.inference.seed_everything(seed=seed,deterministic=self.configs.deterministic)
    shared.write(folder/'config.json',self.configs.to_dict())
    shared.write(folder/'features.json',shared.feature_manifest(common['input_feature_dict']))
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.musa.get_rng_state_all())
    sync_handle=None
    if int(common['N_token'])>1024 and __import__('os').environ.get('MUSA_SYNC_BOUNDARY')=='cycle':
        sync_handle=self.model.pairformer_stack.register_forward_hook(lambda *args:torch.musa.synchronize())
    rows=[];index=0;pred=None
    for phase,variant in [('warmup',v) for v in variants]+[('measured',v) for v in order]:
        layers._attention=variants[variant]
        pred=None;batch=copy.deepcopy(common)
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.musa.set_rng_state_all(rng[3])
        torch.musa.synchronize();torch.musa.reset_peak_memory_stats();started=time.perf_counter()
        pred=shared.original(self,batch)
        torch.musa.synchronize();seconds=time.perf_counter()-started
        a=pred['coordinate'].detach().float().cpu().numpy();assert np.isfinite(a).all()
        if phase=='measured':
            index+=1;np.savez_compressed(folder/f'coordinates_{index}.npz',coordinate=a)
        rows.append(dict(pdb=name,phase=phase,variant=variant,iteration=index if phase=='measured' else 0,
            predict_seconds=seconds,n_token=int(common['N_token']),n_atom=int(common['N_atom']),n_msa=int(common['N_msa']),
            coordinates_finite=True,frozen_features=True,amp_alignment=True,mudnn_tf32=True,
            peak_allocated_bytes=torch.musa.max_memory_allocated(),peak_reserved_bytes=torch.musa.max_memory_reserved(),
            coordinate_sha256=hashlib.sha256(a.tobytes()).hexdigest(),physical_gpu=shared.HOST_GPU))
        shared.write(folder/'measurements.json',dict(requested_repeats=len(order),amp_alignment=True,mudnn_tf32=True,
            experiment=experiment_description,rows=rows))
        print('CROSSOVER',json.dumps(rows[-1]),flush=True);del batch
    layers._attention=native_attention
    if sync_handle:sync_handle.remove()
    return pred


if __name__=='__main__':
    shared.inference.InferenceRunner.predict=crossover
    shared.inference.run()
