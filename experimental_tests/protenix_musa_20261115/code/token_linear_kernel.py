"""Experimental grouped FlagGems Linear on aligned, packed FP32 inputs."""
import triton
import triton.language as tl
from flag_gems.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=[
        triton.Config({"BM": 32, "BN": 64, "BK": 32}, num_warps=2, num_stages=2),
        triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 32, "BN": 128, "BK": 32}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def token_linear_kernel(X, W, B, Y, M, N, K, BIAS: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    group_start = pid // (4 * tiles_n) * 4
    group_rows = tl.minimum(tiles_m - group_start, 4)
    within = pid % (4 * tiles_n)
    rows = (group_start + within % group_rows) * BM + tl.arange(0, BM)
    cols = (within // group_rows) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    x = X + rows[:, None] * K + reduction[None, :]
    w = W + reduction[:, None] * N + cols[None, :]
    total = tl.zeros((BM, BN), tl.float32)
    # Only multiples of BK are dispatched, so the reduction tiles need no masks.
    for k in range(0, K // BK):
        lhs = tl.load(x + k * BK, mask=rows[:, None] < M, other=0.0)
        rhs = tl.load(w + k * BK * N, mask=cols[None, :] < N, other=0.0)
        total += tl.dot(lhs, rhs, allow_tf32=False)
    if BIAS:
        total += tl.load(B + cols, mask=cols < N, other=0.0)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], total,
             mask=(rows[:, None] < M) & (cols[None, :] < N))
