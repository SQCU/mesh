# SVD least-squares source checkpoint

Non-CPU gelss/gelsd now expand through one source composition before the existing SVD and EIGH outline. Two fake-only source targets connect the composition to the existing structured range scaling and cancellation-aware solution unscale; they have actual lowerings and no eager kernels. The numerical SVD is the existing QDWH implementation. CPU LAPACK and the gels/gelsy structured QR routes retain their current paths.

The composition preserves PyTorch's default rcond, separate A/broadcast-RHS batches, vector result shape, rank and singular-value outputs, and the two LAPACK drivers' distinct threshold rules (including GELSD's one-value case and E versus P machine epsilon). It makes rejected singular denominators safe before division. The existing fake-metadata rejection for m>n>0 and these drivers remains: the residual result's shape depends on rank. Empty source shapes retain their captured outputs.

AST and the actual capture/place/outline/device observation passed for A2×3/B2×2 with gelsd: prefix2→21, three calls of the shared2×2 Step6→4, suffix16→3; outer shapes solution3×2/residual0/rank scalar/singular2, rank int64. Both scaling targets and the generated SVD/EIGH retain the original lstsq source site. Prefix/suffix original-source counts316/38 survive device mapping. Numerical invocation is still pending at this checkpoint.

Primary sources: Reference-LAPACK SRC/dgelss.f, SRC/dgelsd.f, SRC/dlalsd.f and INSTALL/dlamch.f at https://github.com/Reference-LAPACK/lapack ; PyTorch v2.14.0 aten/src/ATen/native/BatchLinearAlgebra.cpp linalg_lstsq_out_info and default rcond at https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/BatchLinearAlgebra.cpp . Exact downloaded reference hashes accompany the source hashes. No C/ABI change or original malformed-path edit.

This does not establish complex/ANE coverage, nonfinite LAPACK INFO equivalence, general degeneracy or extreme scaling accuracy. Those remain separate obligations; no invented error policy or convergence guard was added.
