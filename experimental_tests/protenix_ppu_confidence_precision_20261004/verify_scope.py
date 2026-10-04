"""CPU check: each boundary changes only intended flags and restores on errors."""
import torch
import confidence_precision as policy

events = []
real_einsum, real_cdist = torch.einsum, torch.cdist


def einsum(*a, **kw):
    events.append(('atom_output', torch.backends.cuda.matmul.allow_tf32))
    return real_einsum(*a, **kw)


def cdist(*a, **kw):
    events.append(('distance', torch.backends.cuda.matmul.allow_tf32))
    return real_cdist(*a, **kw)


class Projection(torch.nn.Module):
    def forward(self, x):
        events.append(('pair_output', torch.backends.cuda.matmul.allow_tf32))
        return x


class ConfidenceHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_no_bias_pae = Projection()
        self.linear_no_bias_pde = Projection()

    def forward(self, fail=False):
        return self.memory_efficient_forward(fail)

    def memory_efficient_forward(self, fail):
        x, w = torch.ones(2, 3), torch.ones(2, 3, 4)
        torch.cdist(x, x)
        self.linear_no_bias_pae(x)
        self.linear_no_bias_pde(x)
        torch.einsum('...nc,ncb->...nb', x, w)
        torch.einsum('...nc,ncb->...nb', x, w)
        if fail:
            raise RuntimeError('intentional restoration control')


def main():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.einsum, torch.cdist = einsum, cdist
    try:
        model = ConfidenceHead()
        policy.install(model)
        for mode in ('fp32', 'tf32', 'tf32_confidence_outputs', 'tf32_confidence_distance', 'tf32_confidence_all'):
            policy.ACTIVE[0] = mode
            torch.backends.cuda.matmul.allow_tf32 = mode != 'fp32'
            for fail in (False, True):
                events.clear()
                try:
                    model(fail=fail)
                except RuntimeError:
                    assert fail
                assert len(events) == 5
                for op, flag in events:
                    expected = mode not in ('fp32', 'tf32_confidence_all')
                    if mode == 'tf32_confidence_outputs' and op != 'distance':
                        expected = False
                    if mode == 'tf32_confidence_distance' and op == 'distance':
                        expected = False
                    assert flag == expected, (mode, op, flag, expected)
                assert torch.backends.cuda.matmul.allow_tf32 == (mode != 'fp32')
                assert torch.einsum is einsum and torch.cdist is cdist
        print('CONFIDENCE_PRECISION_SCOPE_CONTROL_PASS', 'five policies, normal and exception paths')
    finally:
        torch.einsum, torch.cdist = real_einsum, real_cdist
        torch.backends.cuda.matmul.allow_tf32 = previous


if __name__ == '__main__':
    main()
