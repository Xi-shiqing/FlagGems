"""PPU experiment: native RNE conversion with the existing non-finite contract."""
import triton
import triton.language as tl


@triton.jit
def round_tf32_rne(x, enabled: tl.constexpr):
    if enabled and x.dtype == tl.float32:
        bits = x.to(tl.uint32, bitcast=True)
        exponent_mask = tl.full((), 0x7F800000, tl.uint32)
        finite = (bits & exponent_mask) != exponent_mask
        rounded = tl.inline_asm_elementwise(
            "cvt.rn.tf32.f32 $0, $1;",
            constraints="=f,f",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        # Preserve infinity and NaN bits, including sign and payload.
        return tl.where(finite, rounded.to(tl.uint32, bitcast=True), bits).to(
            tl.float32, bitcast=True
        )
    return x
