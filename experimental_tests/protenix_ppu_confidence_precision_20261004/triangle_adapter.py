"""Isolated corrected TF32 diagnostic; never labeled as native CUE."""
import torch
import cuda_fixed_rne_once as corrected


def triangle_attention(q, k, v, bias, mask=None, scale=None, return_aux=False):
    assert not return_aux, 'Model diagnostic uses output only; no fabricated native auxiliaries.'
    assert q.dtype == k.dtype == v.dtype == torch.float32
    squeeze = q.ndim == 4
    if squeeze:
        q, k, v = [x.unsqueeze(0) for x in (q, k, v)]
    assert q.ndim == 5 and q.shape == k.shape == v.shape
    b, n, h, s, d = q.shape
    if bias.ndim == 4:
        bias = bias.unsqueeze(0)
    assert bias.shape == (b, 1, h, s, s)
    if mask is None:
        bias1 = torch.zeros((b, n, 1, 1, s), device=q.device, dtype=q.dtype)
    else:
        if mask.ndim == 4:
            mask = mask.unsqueeze(0)
        assert mask.shape == (b, n, 1, 1, s) and mask.dtype == torch.bool
        bias1 = (~mask).to(torch.float32) * -1e9
    scale = d**-.5 if scale is None else scale
    q = q * scale
    q, k, v = [x.permute(0, 1, 3, 2, 4).contiguous() for x in (q, k, v)]
    out = corrected.triangle_attention(q, k, v, bias1, bias.float().contiguous(), precision='tf32', layout='BNSHD')
    out = out.permute(0, 1, 3, 2, 4)
    # Low-level CUE normalizes to five dimensions and retains the batch axis.
    # The real Protenix caller removes that axis with [0].
    return out
