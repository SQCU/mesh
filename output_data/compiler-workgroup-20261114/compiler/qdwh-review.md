# QDWH continuation checkpoint

The `_polar` source helper now performs JAX `_qdwh`'s remaining bounded Halley steps, preserving per-batch convergence selection and its final refinement. Codex reviewed the coefficient `(3-1/3,3,1/3)`, counter bound, unchanged fixed steps, and convergence predicate against the primary JAX source. This helper has no production caller yet. The actual placed-graph custom decomposition stage is the next connected consumer, owned by the same numerical session with the frontend driver.

Actual source checks are in numerical.status.md: AST/diff clean; direct extracted current function observations on real/complex and batched matrices. One ill-conditioned f64 case improves orthogonality residual from 4.6e-3 to 5.6e-16; unchanged converged cases remain identical. These are helper arithmetic checks on CPU, not GPU/ANE compilation or execution evidence.

The same source checkpoint includes root's removal of three stale ctypes lines for deleted MetalProgramSignature, paired with Metal4c2fd3d7. No other source hunk is included. Remaining Hermitian recursion and nonsymmetric driver conversion are explicitly unimplemented here. Reference malformed orchestration is unchanged.
