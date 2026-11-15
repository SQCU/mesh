# Change request: lower tensor functions through source and typed IR

### 1. Every function lowers from source

The lowerer covers every linear-algebra/tensor program in AOT A1's domain by composing the source operations'
lowering rules (MLIR full conversion; operator's program-domain instruction). Items 5/6 specify particular
numerical mechanisms within that coverage.

PyTorch export and FX produce typed numerical operations with shapes, layouts, alias relations and explicit
operands (Reed et al., *torch.fx*, MLSys 2022; Ansel et al., *PyTorch 2*, ASPLOS 2024). Operator lowering produces
the vertices and accesses of the joint task graph (Sinnen, *Task Scheduling for Parallel Systems*, 2007).
Placement supplies each function's CPU, GPU or ANE target. The target compiler composes and fuses those operations
through TVM's operator fusion (Chen et al., OSDI 2018, §3) and Halide's algorithm/schedule lowering.
LiteRT's typed delegate bodies enter through the composition request's retained compiler interface.
A custom numerical lowerer's position and operands are those of the composition request's
[numerical lowerer boundary](compilation-composition-change-request.md#numerical-lowerer-boundary).

### 2. The capture is a static dataflow graph

`torch.export.export` captures the assigned known-shape tensor program; FX represents its function and call graphs.
The compiler consumes that representation with the operator's supplied placement and cost operands (AOT A1–A3).
Conditions and iterations inside the algorithms of items 5/6 are lowered with those algorithms. Their compiled
bodies preserve the source condition, selected work and returned values. This is the operator's AOT scope in
[the 14 November turns](documentary-review-2026-11-13.md#positive-mechanism-form-2026-11-14).

### 3. Dependences lower to target orderings

The schedule consumes the placed task graph and execution costs, and returns ordered operations and retained
synchronization causes (plan-to-bridge R5; Bhattacharyya, Sriram and Lee 1997; Sriram, PhD thesis, 1995, §5.4/§5.8).
Metal emission realizes those relations through Metal 4 queue, encoder, fence and event operations
(Apple SDK `MTL4CommandQueue.h`, `MTL4CommandEncoder.h`, `MTLEvent.h`; Metal Shading Language 4.1 §6.10.1).
Source operations and required ordering remain connected through the retained relation consumed by the signature.

### 5. Numerical drivers implement their stated domains

The CPU lowering implements `svd`, `eigh`, `eig`, `eigvals` and least-squares drivers through Accelerate's LAPACK
interfaces (`clapack.h`: `dgesdd_`, `dsyevd_`, `zheevd_`, `dgeev_`, `dgelsd_`, `dgelss_`).
PyTorch's `torch.linalg.lstsq` specifies the `gels`, `gelsy`, `gelsd` and `gelss` driver domains and outputs;
`torch.linalg.pinv` specifies its SVD/eigh composition. The compiler preserves each routine's input/output shapes,
precision, rank/tolerance rules and `INFO` result through the ordinary result and error boundary.
The driver's iterations remain inside the compiled numerical call (LAPACK driver implementations and PyTorch
`torch.linalg` reference APIs).

### 6. Matrix and QR formulations lower on the placed processor

Operator lowering produces the numerical methods below as typed operations and conditional regions. Placement
consumes their explicit accesses and assigns CPU, GPU or ANE execution. This target coverage is the operator's
19:55 turn of 9 November, retained in this file's history at `cdd44406`.

| Operation | Mechanism and source | Result consumed |
|---|---|---|
| Polar decomposition | JAX `jax/_src/tpu/linalg/qdwh.py`, `_qdwh`; Nakatsukasa, Bai and Gygi, SIAM J. Matrix Anal. Appl. 31(5), 2010 | Polar factors, iteration count and convergence result |
| Hermitian eigendecomposition | QDWH-eig, Nakatsukasa and Higham, SIAM J. Sci. Comput. 35(3), 2013, §3.1; JAX `tpu/linalg/eigh.py` | Eigenvalues and eigenvectors |
| SVD | Nakatsukasa and Higham, Algorithm 2: polar decomposition, Hermitian eigendecomposition and matrix product | Singular values and vectors |
| Pseudoinverse and least squares | PyTorch `torch.linalg.pinv` and `lstsq`; the stated LAPACK driver threshold/rank semantics | Pseudoinverse, solution, residuals, rank and singular values |
| Matrix exponential | JAX `jax/_src/scipy/linalg.py`, `_calc_P_Q`, `_squaring`, `expm` | Scaling-and-squaring result and required error result |

QDWH computes its coefficient schedule from precision and `max_iterations`, then executes the source's QR or
Cholesky steps, bounded Halley continuation, refinement and Hermitian factor. The continuation uses
`logical_and(is_not_converged, k < max_iterations)`; the result carries the actual convergence information.
QDWH-eig uses the source's median split, polar subspace and work agenda, with sizes computed from the matrix values
(JAX `_eigh_work` and `split_spectrum`).

Matrix exponential selects Padé degree from the L1 norm, using degrees 3/5/7 for float32/complex64 and
3/5/7/9/13 for float64/complex128. It computes `n_squarings` with the source's constants and performs the
`max_squarings` scan, selecting the square for `i < n_squarings` and the carried matrix otherwise. JAX's
conditional batching rule supplies `cond` to `select` under `vmap` over predicates. The scalar case retains
its conditional execution. Exceeding `max_squarings` yields NaNs and the operator's requested runtime error
(operator turn 2026-11-09 20:01, history at `cdd44406`; JAX `expm`).

Typed binary64 operations use the SoftFloat/FDLIBM mechanisms in composition. Every target lowering consumes the
same declared scalar and complex semantics, storage and source correspondence.
