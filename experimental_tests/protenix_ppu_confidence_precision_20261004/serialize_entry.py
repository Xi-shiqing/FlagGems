"""Serialize shared ESM preprocessing through first CPU feature batch only."""
import argparse
import fcntl
from pathlib import Path
import runpy
import sys

from runner import batch_inference
from protenix.data.inference import infer_dataloader

parser=argparse.ArgumentParser()
parser.add_argument('--serialized-entry',type=Path,required=True)
args, remaining=parser.parse_known_args()
lock=Path(__file__).with_name('input_preprocess.lock').open('a')
original_preprocess=batch_inference.preprocess_input
original_loader=infer_dataloader.get_inference_dataloader


def preprocess(*a, **kw):
    fcntl.flock(lock,fcntl.LOCK_EX)
    return original_preprocess(*a, **kw)


def dataloader(*a, **kw):
    loader=original_loader(*a, **kw)
    class FirstBatch:
        def __getattr__(self,name):return getattr(loader,name)
        def __iter__(self):
            for batch in loader:
                fcntl.flock(lock,fcntl.LOCK_UN)
                yield batch
    return FirstBatch()


batch_inference.preprocess_input=preprocess
infer_dataloader.get_inference_dataloader=dataloader
sys.argv=[str(args.serialized_entry),*remaining]
try:
    runpy.run_path(str(args.serialized_entry),run_name='__main__')
finally:
    fcntl.flock(lock,fcntl.LOCK_UN)
    lock.close()
