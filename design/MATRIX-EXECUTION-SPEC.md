# Matrix execution specification

This specification controls `xonotic/solver/strat/matmul.py` and the matrix products
composed from it by the policy. It refines the policy data flow in
[`SPECIFICATION.md`](SPECIFICATION.md), [`rl-training-spec.md`](rl-training-spec.md),
and [`ALGORITHM-CONTRACTS.md`](ALGORITHM-CONTRACTS.md).

## Interface boundary

The producer supplies two finite rank-two FP32 tensor views. The matrix execution
boundary supplies three operations: `A B`, `Aᵀ B`, and `A Bᵀ`. Inner dimensions must
agree. The result is one row-major FP32 tensor with the corresponding mathematical
shape. Transposition is an addressing mode and does not manufacture a transposed
application payload.

The implementation owns the Metal kernels, output geometry, hardware-tile schedule,
and reverse products. MLX supplies array storage, command submission, and transformation
plumbing; neither `mlx.linalg` nor an MLX matrix-multiplication operator supplies the
operation. Shapes determine dispatch extent. No player, team, cart, instrument, or row
count is a kernel limit.

The large-output kernels map FP32 operands into 16-by-32 Metal Performance Primitive
products, accumulate in FP32 cooperative tensors, and cover the output with 32-by-32 or
64-by-32 tiles according to row extent. The small-output kernel covers the same operation
with 16-by-16 threadgroup tiles.
Both paths compute the complete requested product. Their selection is an execution-shape
schedule, not a capability gate.

Reverse execution is part of the interface:

- for `Y = A B`, `dA = dY Bᵀ` and `dB = Aᵀ dY`;
- for `Y = Aᵀ B`, `dA = B dYᵀ` and `dB = A dY`;
- for `Y = A Bᵀ`, `dA = dY B` and `dB = dYᵀ A`.

Every reverse product re-enters the same owned kernel boundary. Training cannot silently
substitute a host algebra routine for the forward GPU operation.

Expert products accept sorted `(row, expert)` assignments, one feature row per
assignment, and one matrix per expert. The forward kernel multiplies every row by its
literal selected matrix. The reverse row product uses the transposed selected matrix;
the reverse weight product finds each expert's complete contiguous assignment interval
and reduces every outer product in that interval. Expert identities schedule products
but never enter or alter tensor payload values.

## DPP composition

The policy producer supplies one nonnegative quality coordinate and one feature row per
instrument. The DPP operator RMS-normalizes feature rows, constructs
`B = diag(quality) features`, forms the feature-side covariance `S = I + BᵀB`, and
applies dimension-counted conjugate-gradient iterations to `S X = Bᵀ`. It emits
`p_i = b_iᵀ x_i`, clipped to the probability interval, one coordinate per instrument.
Each denominator carries a dtype-epsilon multiple of the initial residual measure and
the covariance diagonal scale. Once relaxed FP32 products reach their representable
floor, subsequent dimension-counted updates therefore tend continuously to zero instead
of forming an underflowing `0/0` direction.

The feature dimension determines the iteration count. There is no method selector,
host fallback, eigendecomposition, instrument-by-instrument inverse, convergence
rejection, or fixed instrument-count envelope. `strategy.py` consumes every emitted
coordinate by multiplying the corresponding instrument contribution to the mixed IR.

## Measures

`work_estimate.py` counts covariance products, every conjugate-gradient product, and the
final marginal contraction from realized instrument and feature dimensions. Runtime
measurements must retain those dimensions, elapsed GPU time, finite output mass, and
forward/backward finiteness. Numerical comparison is a measure against an FP64 dense
reference, never a runtime acceptance gate.

`measure.py matrix` receives unrestricted row, inner, column, and sample coordinates. It
reports the selected threadgroup, MPP, or wide-MPP schedule for each forward and reverse
product, finite coordinate mass, absolute and relative residual measures against the FP64
reference, elapsed-time distributions, and the corresponding DPP marginal measure. The
residual records the MPP descriptor's relaxed-precision execution; it does not select a
different runtime implementation.
