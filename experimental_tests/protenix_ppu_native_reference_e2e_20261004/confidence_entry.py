"""Diagnostic IEEE matmul scope by ConfidenceHead class, without shape rules."""
import inspect
import json
from pathlib import Path
import runpy
import sys
import torch
from runner import batch_inference

output = Path(sys.argv[sys.argv.index('--paired-output')+1]).parent
original = batch_inference.get_default_runner
calls = {}
sources = {}


def initialize(*a, **kw):
    runner = original(*a, **kw)
    for name, module in runner.model.named_modules():
        if type(module).__name__ != 'ConfidenceHead':
            continue
        call = module.forward
        sources[name] = inspect.getfile(call)
        def ieee(*a, _call=call, _name=name, **kw):
            previous = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                calls[_name] = calls.get(_name, 0)+1
                return _call(*a, **kw)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous
        module.forward = ieee
    assert sources, 'No semantic confidence module matched'
    return runner


batch_inference.get_default_runner = initialize
try:
    entry = Path(__file__).resolve().parent / 'precision_entry.py'
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry),run_name='__main__')
finally:
    output.joinpath('confidence_precision_audit.json').write_text(json.dumps({
        'status':'PASS' if calls else 'FAIL', 'actual_calls':calls, 'matched_source_files':sources,
        'scope':'Global matmul TF32 disabled only during ConfidenceHead forward. Custom corrected TriAttention retains explicit TF32 RNE. No model or shape whitelist.'},indent=2)+'\n')
