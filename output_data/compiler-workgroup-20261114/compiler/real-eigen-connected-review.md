# Connected real GEEV source

`mesh::geev` now owns the common real/complex result interface. Decomposition selects it for non-CPU eig/eigvals; CPU keeps the existing LAPACK path. Real input outputs the corresponding complex dtype and column-major eigenvectors.

One GEEV driver shares GEBAL, GEHD2/Q, GEBAK, scaling and normalization. It selects DLAHQR plus the new real quasi-triangular DTREVC recovery, or the existing complex ZLAHQR/ZTREVC. The new DLALN2 small scaled solve serves both real and conjugate-pair recovery. Packed real Schur vectors become ordered complex conjugate columns through explicit components. No second balancing/Hessenberg stack or evaluator was added.

The source uses the existing stable complex division, reduction order for the backtransform, and complex phase normalization instead of DLARTG's exact rounding. The common reflector's rounding distinction is recorded in real-eigen-dlahqr-review.md. No bit-identity claim is made.

Supervisor AST and diff checks pass. FakeTensor→actual GEEV record construction passes n3 float32, float64 and complex64, preserving complex output dtypes/layouts (real-eigen-connected-construction.json). This is construction evidence only. The public native numerical result remains pending; source is held through root/frontend finite import windows before the first dense real case.
