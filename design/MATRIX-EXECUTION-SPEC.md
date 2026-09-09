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

One shared Metal Performance Primitive implementation supplies dense products and
homogeneous routed tiles. It constructs physical tensor views and selects transposition
in the operation descriptor, covering dense output with 32-by-32 or 64-by-32 tiles
according to row extent. The small-output kernel covers the same operation
with 16-by-16 threadgroup tiles.
Both paths compute the complete requested product. Their selection is an execution-shape
schedule, not a capability gate.

The implementation imports MPP's tensor layout, loading and storage instead of spelling
out a lane-to-coordinate map. That assignment is hardware-dependent, as documented in
[Apple's MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf),
section 3.1. The previous assumed layout produced order-one errors on the Mini despite
working on the MacBook. The MPP interface accepts mutable tensor-view element types;
its source views are read-only operands and the destination is a separately allocated
output. No source operand is written through those views.

The rectangular `matrix_multiply_transpose_left(R, V)` cross factor and its remote pullback are specified in the
[single-policy manifest](POLICY-STATE-CONTRACT.md). Its local and remote callers import
the same implementation. There is no remote model or independently trained "expert".

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

## Policy composition and measures

The live policy uses learned global and team row-Gram actions and a sparse local
neighborhood Gram. The global action is reassociated exactly as `R(RᵀV)`.
The cross factor `RᵀV` may execute remotely with both operand derivatives.
The live policy has no DPP allocator, instrument list, feature-Gram probe or
one-vector broadcast gate. The standalone `dpp.py` numerical utility is not in
the policy call graph.

`work_estimate.py` counts actual raw projections, neighborhood contractions,
global/team Gram products, routed SwiGLU products and state-rate/value readouts.
The row dimensions distinguish all observations, per-owner state pages, carts,
teams and unique event/navigation sources. Routed SwiGLU uses three expert
matrices. These are logical operation estimates, not hardware counter readings;
training and byte envelopes do not establish measured utilization.

`measure.py matrix` receives unrestricted row, inner, column, and sample coordinates. It
reports the selected threadgroup, MPP, or wide-MPP schedule for each forward and reverse
product, finite coordinate mass, absolute and relative residual measures against the FP64
reference, elapsed-time distributions, and the corresponding DPP marginal measure. The
residual records the MPP descriptor's relaxed-precision execution; it does not select a
different runtime implementation.
