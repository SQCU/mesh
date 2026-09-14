# Direct CPU contraction with half operands

September 13, 2026. This is the source and compilation record for the CPU step of
[Pallas-style lowering](pallas-lowering-plan.md#4-lower-to-existing-efficient-backend-functions).
Operational performance evidence remains to be collected through the existing
streaming-algebra workflow.

## Mechanism

`rdma/mesh-algebra.m:cpu_part` retains Accelerate SGEMM for the existing all-FP32
case. Other contraction combinations use a 4 by 4 outer-product microkernel.
Each K iteration loads four left rows and four right columns, converts half
values to FP32, and updates four FP32 vectors using lane-broadcast fused
multiply-add. This reuses each loaded value across four products. Alpha applies
after the complete bound K region; output conversion occurs only on store.

Arm's [Advanced SIMD intrinsic reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html)
defines `vcvt_f32_f16` and `vfmaq_laneq_f32`. These ordinary conversion and FP32
FMA instructions avoid depending on optional half-accumulating arithmetic.
The numerical decomposition is the register-blocked outer product described by
Goto and van de Geijn, [Anatomy of High-Performance Matrix Multiplication](https://doi.org/10.1145/1356052.1356053).
This implementation adopts register blocking; it does not adopt packed operand
copies or claim the performance of their complete GEMM implementation.

Setup selects the half/float and contiguous/strided load functions, output store
function, pointers and rectangle geometry. Numerical invocation neither chooses a
backend nor converts an entire operand. Full vectors load directly from canonical
addresses; strided vectors gather four scalar addresses. Ragged vectors load only
valid lanes and fill unused lanes with zero. The loop stores only valid output
coordinates through the retained row and column strides. Stack vectors are bounded
microkernel scratch, not a dense converted input or a separately owned tensor.
Compiler register spills are possible and are not counted as tensor staging.

The existing publication-boundary rectangles remain unchanged. Transposed output
rectangles use the same physical-to-logical rectangle mapping as bind_part.
Each microkernel covers only its configured rectangle, and the numerical function
publishes through complete_part when all of its own rectangles finish. No larger
operand or unrelated output becomes a dependency. Separate K-region FP32 partials
and their final casts remain owned by the existing contraction composition.

## Why this path

The installed Accelerate SDK exposes s/d/c/z GEMM, with SGEMM requiring float
pointers. It does not expose a half-input, float-output BLAS signature.
BNNSMatMul permits supplied workspace, but its declaration does not establish
half-input FP32 accumulation or absence of internal dense conversion.
BNNSGraph can express casts before contraction and accepts caller-owned
arguments/workspace; its public contract permits internal tensor layouts and
weight repacking. Apple's [BNNS presentation](https://developer.apple.com/videos/play/wwdc2024/10211/)
also discusses 16-bit accumulator overflow. Allocation-free execution and direct
external pointers do not prove copy-free internal contraction.

The former metal-microbench BNNS projection, removed by commit 35ccf7d, made
weights a graph constant and cast both operands to Float before matmul. Its
retained [backend report](../../metal-microbench/docs/soc_compute_backends.md#precision-is-part-of-a-backends-contract)
records precision and throughput limitations; it is not an existing direct mixed
GEMM to reuse. The new path remains in the already-selected CPU backend.

## Evidence and limits

`make -C rdma libmesh.dylib libmesh-algebra.dylib` succeeds. Native disassembly
contains FCVTL vector conversion and four lane FMLA instructions in the K loop.
Source inspection establishes direct pointer flow, setup-owned specialization,
bounded scratch, finite tail accesses and independently bounded output work.
No workload was executed for this source increment.

The 4 by 4 shape is an initial ordinary register block, not a measured optimum.
Indirect loader calls, strided gathers, small rectangles and output stores remain
costs. FMA changes rounding relative to separately rounded multiplication/addition;
existing FP32 accumulation and cancellation/overflow requirements still apply.
Operational checks must include mixed operand types, strided/transposed operands,
ragged sizes, alpha, output precision, and the distributed gold chain before a
performance or complete numerical-validation claim is made. Preserve the fastest
validated local baseline when comparing subsequent CPU or distributed gains.
