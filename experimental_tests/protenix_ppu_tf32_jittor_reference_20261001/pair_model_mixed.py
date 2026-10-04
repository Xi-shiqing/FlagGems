#!/usr/bin/env python3
"""Apply the frozen diffusion/triangle precision policy to allocation controls."""
import fnmatch
from functools import wraps
import os
from pathlib import Path
import runpy


def main():
    import torch
    from runner import batch_inference
    original_init = batch_inference.get_default_runner

    def strict(call):
        @wraps(call)
        def run(*a, **kw):
            previous = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                return call(*a, **kw)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
        return run

    def initialize(*a, **kw):
        runner = original_init(*a, **kw)
        diffusion = runner.model.diffusion_module
        diffusion.forward = strict(diffusion.forward)
        for name in ('diffusion_conditioning', 'atom_attention_encoder'):
            module = getattr(diffusion, name)
            module.prepare_cache = strict(module.prepare_cache)
        for name, module in runner.model.named_modules():
            if fnmatch.fnmatchcase(name, '*tri_att*.mha'):
                module.forward = strict(module.forward)
        original_predict = runner.predict
        def predict(*a, **kw):
            previous = os.environ.get('PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION')
            os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION'] = 'ieee'
            try:
                return original_predict(*a, **kw)
            finally:
                if previous is None:
                    os.environ.pop('PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION', None)
                else:
                    os.environ['PROTENIX_TRIATTENTION_FLAGGEMS_PRECISION'] = previous
        runner.predict = predict
        return runner

    batch_inference.get_default_runner = initialize
    runpy.run_path(str(Path(__file__).with_name('pair_model_outputs.py')), run_name='__main__')


if __name__ == '__main__':
    main()
