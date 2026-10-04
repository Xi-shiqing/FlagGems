"""Verify exact error and linear aggregate statistics against a known perturbation."""
import numpy as np
from activation_boundaries import summaries,error

x=np.arange(8192,dtype=np.float32)/1024
z=x.copy();z[1777]+=3.25
s,t=summaries(x),summaries(z)
a=np.array(list(s['aggregate_checksums'].values()))
b=np.array(list(t['aggregate_checksums'].values()))
assert np.linalg.norm(a-b)==3.25
v=error(z,x)
assert v['max_abs']==3.25
assert abs(v['relative_l2']-3.25/np.linalg.norm(x.astype(np.float64)))<1e-14
assert summaries(x[:8])['aggregate_checksums'] is None
assert error(x,x)['relative_l2']==0
print('DERIVED_CHECKSUM_LINEARITY_AND_FULL_ERROR_PASS')
