# PPU full-write buffer allocation

Extends the optional allocator to MM/addmm partial sums and outputs, RHS preparation, and math-attention temporary buffers. All 12 existing GPU computation kernels are unchanged. Empty MM dimensions are handled before entering the general kernel; zero-K writes zeros, including strided provided outputs.

Validation: 82 operator checks, 14 complete-interface timing comparisons, and 200 fully audited predictions across Tiny and Mini ISM, 245 and 693 token inputs. Short inputs were repeated in two independent processes. Allocation-on/off outputs and all five confidence metrics are bitwise equal. Pooled end-to-end time reductions: FP32 0.65–1.46%, TF32 0.92–4.10%.

The existing long-K TF32 fixed-tolerance diagnostic failures are preserved in the report; no arithmetic or model quality threshold was changed by the allocation optimization. CPU acceptance uses the existing MM suite criterion scaled by reduction length. A comment describing the zero-K negative peeled tile was corrected after validation; source_equivalence.json confirms identical executable ASTs.

Complete executed scripts, frozen source versions, failed-attempt records and scalar data are under Protenix/comparison/operator_allocation_20261109. Raw predictions and logs remain on the original PPU. The optional allocator defaults off; build/enable instructions are in docs/content/zh-cn/optimization/full-write-allocation.md.
