"""Full live attention references on the original GPU; no tensor persistence."""


def quantize(value, mode):
    import torch
    if mode == 'fp32':
        return value
    if mode in ('bf16', 'fp16'):
        return value.to(torch.bfloat16 if mode == 'bf16' else torch.float16).float()
    if mode == 'bf16_rtz':
        return (value.contiguous().view(torch.int32) & -65536).view(torch.float32)
    assert value.dtype == torch.float32 and mode in ('rne', 'rna', 'rtz')
    bits = value.contiguous().view(torch.int32)
    if mode == 'rne':
        bits = bits + 0xfff + ((bits >> 13) & 1)
    elif mode == 'rna':
        bits = bits + 0x1000
    return (bits & -8192).view(torch.float32)


def references(bound, parameters, ppu):
    import torch
    q, k, v = [bound.arguments[name] for name in parameters]
    if ppu:
        q, k, v = [value.permute(0, 1, 3, 2, 4) for value in (q, k, v)]
        bias1, bias2 = [bound.arguments[name] for name in ('bias1', 'bias2')]
        scale = 1.0  # Public FlagGems adapter receives the already scaled Q.
    else:
        if q.ndim == 4:
            q, k, v = [value.unsqueeze(0) for value in (q, k, v)]
        bias2 = bound.arguments['bias']
        if bias2.ndim == 4:
            bias2 = bias2.unsqueeze(0)
        mask = bound.arguments.get('mask')
        if mask is None:
            bias1 = 0.0
        else:
            if mask.ndim == 4:
                mask = mask.unsqueeze(0)
            bias1 = -1e9 * (~mask).float()
        scale = bound.arguments.get('scale')
        if scale is None:
            scale = q.shape[-1]**-.5
    modes = [('fp32', 'scaled', 'fp32'), ('rne', 'scaled', 'rne'), ('rne', 'scaled', 'rtz'),
             ('rna', 'scaled', 'rna'), ('rna', 'scaled', 'rtz')]
    if not ppu:
        modes += [('rna', 'raw', 'rna'), ('rna', 'raw', 'rtz')]
    modes += [('q_rtz_kv_rna', 'scaled', 'rna'), ('q_rtz_kv_rna', 'scaled', 'rtz')]
    modes += [('rne_bias_rna', 'scaled', 'rne'), ('rna_bias_rna', 'scaled', 'rna')]
    modes += [('rna_logits_rna', 'scaled', 'rna'), ('rna_logits_rtz', 'scaled', 'rna'),
              ('rna_log2_logits_rna', 'scaled', 'rna')]
    modes += [('q_bf16_kv_rna', 'scaled', 'rna'), ('q_bf16_rtz_kv_rna', 'scaled', 'rna'),
              ('q_fp16_kv_rna', 'scaled', 'rna')]
    if not ppu:
        modes += [('q_bf16_kv_rna', 'raw', 'rna')]
    for rounding, position, pv in modes:
        name = 'dense_' + rounding + '_q' + position + '_p_' + pv
        if rounding.startswith('q_') and rounding.endswith('_kv_rna'):
            q_rounding, kv_rounding = rounding[2:-7], 'rna'
        else:
            q_rounding = kv_rounding = rounding.split('_')[0]
        a = quantize(q * scale if position == 'scaled' else q, q_rounding)
        scores = torch.matmul(a, quantize(k, kv_rounding).transpose(-1, -2))
        if position == 'raw':
            scores = scores * scale
        scores = scores + bias1 + quantize(bias2, 'rna' if '_bias_rna' in rounding else 'fp32')
        if 'logits' in rounding:
            log2 = 'log2' in rounding
            if log2:
                scores = scores * 1.4426950408889634
            scores = quantize(scores, 'rtz' if rounding.endswith('_rtz') else 'rna')
            if log2:
                scores = scores / 1.4426950408889634
        probabilities = torch.softmax(scores, -1)
        result = torch.matmul(quantize(probabilities, pv), quantize(v, kv_rounding))
        del scores, probabilities
        if ppu:
            result = result.permute(0, 1, 3, 2, 4).contiguous()
        elif len(bound.arguments[parameters[0]].shape) == 4:
            result = result.squeeze(0)
        yield name, result
