"""Use the existing IEEE MV kernel for TF32-enabled vector MM results."""
import torch

from flag_gems.ops.mv import mv


def mm_gemv(a, b, out=None, *, enabled=False):
    if not enabled or a.ndim != 2 or b.ndim != 2:
        return None
    m, k = a.shape
    if b.shape[0] != k or not m or not k or not b.shape[1] or (m != 1 and b.shape[1] != 1):
        return None
    if (a.dtype != torch.float32 or b.dtype != torch.float32
            or a.device.type != 'cuda' or b.device != a.device):
        return None
    if any(sum((n-1)*s for n, s in zip(x.shape, x.stride())) >= 2**31 for x in (a, b)):
        return None
    shape = (m, b.shape[1])
    if out is not None:
        if out.dtype != a.dtype or out.device != a.device or out.shape != shape:
            return None
        if (any(s == 0 and n > 1 for n, s in zip(out.shape, out.stride())) or
                any(out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in (a, b))):
            return None
    value = mv(a, b[:, 0]) if b.shape[1] == 1 else mv(b.T, a[0])
    value = value.reshape(shape)
    if out is not None:
        out.copy_(value)
        return out
    return value
