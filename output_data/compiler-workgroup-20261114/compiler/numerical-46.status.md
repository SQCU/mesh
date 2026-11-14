# Bounded polar continuation: source HOLD
- operations._bounded_polar now emits one existing _Repeat body instead of max_iterations-k0 masked Halley copies; scaling, static QR/Cholesky schedule and final Newton-Schulz refinement stay in their existing owner.
- Retained jax-qdwh.reference.py203-222: k0=len(qr_coefs)+len(chol_coefs), condition is_not_converged AND count<max_iterations, count increments once per pass. No added extent predicate or native API.
- The shared step uses _bounded_halley, _squared_distance and _assigned at actual extent e. Current, unconverged, count, active and e retain their original carried-root correspondence.
- JAX185/231: norm>tol produces is_not_converged; the returned convergence is its negation, including the source NaN comparison behavior. No Cholesky coefficients starts unconverged=True.
- metal.py Builder.repeat emits the body once then registers metal_ir_repeat(first,end,condition,carried); execution uses the existing compiled native repeat.
- operations.py +34/-19. Supervisor AST/diff PASS; pyflakes reports only the same two preexisting unused locals. No build or target in this source unit.
- Scalar Padé selection and Torch _polar continuation remain separate missing source-control connections. Existing unchanged complex Hermitian public caller is queued for native observation.
