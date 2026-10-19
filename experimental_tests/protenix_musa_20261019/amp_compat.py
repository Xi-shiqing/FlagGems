"""Process-local CUDA AMP context adaptation for this MUSA-only inference run."""
import functools
from collections import Counter
import torch

stats=Counter()
_installed=False


def install():
    global _installed
    if _installed:return
    assert hasattr(torch,'musa') and torch.musa.is_available()
    assert not torch.cuda.is_available(), 'Do not redirect AMP in a mixed CUDA/MUSA process'
    original=torch.amp.autocast.__init__
    @functools.wraps(original)
    def initialize(self,device_type,dtype=None,enabled=True,cache_enabled=None):
        if device_type=='cuda':
            stats['cuda_amp_contexts_redirected_to_musa']+=1
            device_type='musa'
        return original(self,device_type,dtype=dtype,enabled=enabled,cache_enabled=cache_enabled)
    # Preserve the class identity and inherited CUDA AMP aliases. This also
    # handles references imported before installation; no shared source edits.
    torch.amp.autocast.__init__=initialize
    _installed=True
