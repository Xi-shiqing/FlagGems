"""Experimental stable Euclidean distance: subtract coordinates before squaring."""
import torch
import triton
import triton.language as tl
try:
    from flag_gems.utils import tl_extra_shim as libdevice
except ImportError:
    import triton.language.extra.libdevice as libdevice

NATIVE = torch.cdist


@triton.jit
def _distance(X, Y, O, NX: tl.constexpr, NY: tl.constexpr, D: tl.constexpr,
              POINTS: tl.constexpr, CHANNELS: tl.constexpr):
    pairs = tl.program_id(0)*POINTS + tl.arange(0, POINTS)
    channel = tl.arange(0, CHANNELS)
    batch = tl.program_id(1)
    valid = pairs < NX*NY
    ix, iy = pairs // NY, pairs % NY
    x = tl.load(X + batch*NX*D + ix[:, None]*D + channel[None, :],
                valid[:, None] & (channel[None, :] < D), other=0)
    y = tl.load(Y + batch*NY*D + iy[:, None]*D + channel[None, :],
                valid[:, None] & (channel[None, :] < D), other=0)
    delta = x-y
    value = libdevice.sqrt_rn(tl.sum(delta*delta, axis=1))
    tl.store(O + batch*NX*NY + pairs, value, valid)


def euclidean_distance(x, y):
    if (not x.is_cuda or x.dtype != torch.float32 or y.dtype != torch.float32 or
            (torch.is_grad_enabled() and (x.requires_grad or y.requires_grad))):
        return NATIVE(x, y, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    assert x.ndim >= 2 and y.ndim >= 2 and x.shape[-1] == y.shape[-1]
    assert x.device == y.device
    batch = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    nx, ny, d = x.shape[-2], y.shape[-2], x.shape[-1]
    out = x.new_empty((*batch, nx, ny))
    if not nx or not ny or not out.numel():
        return out
    if not d:
        return out.zero_()
    x = x.expand(*batch, nx, d).contiguous()
    y = y.expand(*batch, ny, d).contiguous()
    channels = triton.next_power_of_2(d)
    points = max(1, min(128, 4096//channels))
    _distance[(triton.cdiv(nx*ny, points), out.numel()//(nx*ny))](
        x, y, out, nx, ny, d, points, channels, num_warps=4)
    return out
