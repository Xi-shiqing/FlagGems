"""Route large FP32 fused-LN backward calls into the tested FlagGems operator."""
from collections import Counter
import hashlib
import importlib
import math
from pathlib import Path

import torch
from fused_backward_v2_observer_20261024 import PrivatePackageObserver as FusedObserver


class PrivatePackageObserver(FusedObserver):
    def __init__(self,kind):
        super().__init__(kind)
        self.routes=Counter();self.norm_entries=Counter();self.fused_entries=Counter()

    def __enter__(self):
        super().__enter__()
        from protenix.model.layer_norm.layer_norm import fast_layer_norm_cuda_v2
        self.backend=fast_layer_norm_cuda_v2
        self.ln=importlib.import_module('flag_gems.ops.layernorm')
        self.originals={}
        def wrap(original,label):
            def routed(*args):
                dy,mean,rstd,x,shape=args[:5]
                weight=args[5] if label in ('backward_with_weight_affine','backward_with_both_affine') else None
                bias=args[-2] if label in ('backward_with_bias_affine','backward_with_both_affine') else None
                n=math.prod(shape);m=x.numel()//n if n else 0
                eligible=(m>=4096 and 0<n<=2048 and not torch.is_grad_enabled()
                          and (weight is not None or bias is not None)
                          and all(t is None or (t.is_cuda and t.dtype==torch.float32) for t in (dy,x,weight,bias)))
                self.routes[(m,n,label,'FlagGems' if eligible else 'original_CUDA')]+=1
                if not eligible:return original(*args)
                stats_shape=tuple(x.shape[:-len(shape)])+(1,)*len(shape)
                return torch.ops.aten.native_layer_norm_backward.default(
                    dy,x,shape,mean.view(stats_shape),rstd.view(stats_shape),weight,bias,
                    (True,weight is not None,bias is not None))
            return routed
        for label in ('backward_none_affine','backward_with_bias_affine','backward_with_weight_affine','backward_with_both_affine'):
            self.originals[label]=getattr(self.backend,label)
            setattr(self.backend,label,wrap(self.originals[label],label))
        self.original_fused=self.ln._fused_large_row_backward
        def submitted(*args):
            result=self.original_fused(*args)
            self.fused_entries[(args[-2],args[-1],result is not None)]+=1
            return result
        self.ln._fused_large_row_backward=submitted
        return self

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if func is torch.ops.aten.native_layer_norm_backward.default and args[0].is_cuda:
            n=math.prod(args[2]);self.norm_entries[(args[1].numel()//n,n)]+=1
        return super().__torch_dispatch__(func,types,args,kwargs)

    def __exit__(self,*args):
        try:
            for label,original in self.originals.items():setattr(self.backend,label,original)
            self.ln._fused_large_row_backward=self.original_fused
        finally:result=super().__exit__(*args)
        return result

    def evidence(self):
        report=super().evidence()
        report.update(gradient_or_kernel_replacement_by_observer=True,
                      scope='Explicit integration adapter: original forward, generic large-row backward uses registered FlagGems; other calls retain original CUDA. No loss, parameter, optimizer or gradient-value filtering.',
                      route_counts=[dict(M=k[0],N=k[1],affine=k[2],route=k[3],count=v) for k,v in self.routes.items()],
                      actual_registered_norm_entries=[dict(M=k[0],N=k[1],count=v) for k,v in self.norm_entries.items()],
                      fused_helper_return_counts=[dict(M=k[0],N=k[1],submitted=k[2],count=v) for k,v in self.fused_entries.items()])
        report['layernorm_source']={'path':self.ln.__file__,'sha256':hashlib.sha256(Path(self.ln.__file__).read_bytes()).hexdigest()}
        report['adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report['original_CUDA_extension']={'path':self.backend.__file__,'sha256':hashlib.sha256(Path(self.backend.__file__).read_bytes()).hexdigest()}
        return report
