"""Diagnostic decomposition of repeated, unmasked query rounding in native CUE.

This reproduces a measured native behavior, not an accepted attention algorithm.
All arithmetic and tensors stay on the original GPU.
"""
from dense_triangle import quantize


def references(bound, parameters):
    import torch
    q, k, v = [bound.arguments[name] for name in parameters]
    if q.ndim == 4:
        q, k, v = [x.unsqueeze(0) for x in (q, k, v)]
    bias = bound.arguments['bias']
    if bias.ndim == 4:
        bias = bias.unsqueeze(0)
    mask = bound.arguments.get('mask')
    if mask is not None and mask.ndim == 4:
        mask = mask.unsqueeze(0)
    scale = bound.arguments.get('scale')
    if scale is None:
        scale = q.shape[-1]**-.5
    k, v = quantize(k, 'rna'), quantize(v, 'rna')
    state = q.contiguous().view(torch.int32).clone()
    scores = []
    # Native TF32 kernel has a 32-key loop with two 16-key MMA sections.
    # Each section adds 0x1000 again to the persistent query registers.
    for start in range(0, k.shape[-2], 16):
        state = state + 0x1000
        current_q = (state & -8192).view(torch.float32)
        stop = start + 16
        part = torch.matmul(current_q, k[..., start:stop, :].transpose(-1, -2)) * scale
        part = part + bias[..., start:stop]
        if mask is not None:
            part = torch.where(mask[..., start:stop], part, -1e9)
        scores.append(part)
    scores = torch.cat(scores, -1)
    maximum = scores.max(-1).values
    lse = torch.logsumexp(scores, -1)
    probabilities = torch.softmax(scores, -1)
    output = torch.matmul(quantize(probabilities, 'rna'), v)
    if len(bound.arguments[parameters[0]].shape) == 4:
        output = output.squeeze(0)
    yield 'leaky_query_normalized_p_rna', output, lse, maximum
    del probabilities, output
    running_max = torch.full_like(maximum, -torch.inf)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(q)
    for start in range(0, k.shape[-2], 32):
        stop = start + 32
        part = scores[..., start:stop]
        next_max = torch.maximum(running_max, part.max(-1).values)
        correction = torch.exp(running_max - next_max)
        exponentials = torch.exp(part - next_max.unsqueeze(-1))
        denominator = denominator * correction + exponentials.sum(-1)
        numerator = numerator * correction.unsqueeze(-1) + torch.matmul(
            quantize(exponentials, 'rna'), v[..., start:stop, :])
        running_max = next_max
    output = numerator / denominator.unsqueeze(-1)
    if len(bound.arguments[parameters[0]].shape) == 4:
        output = output.squeeze(0)
    yield 'leaky_query_online_p_rna', output, running_max + denominator.log(), running_max
