# Bounded polar corrective source: HOLD
- c5df7c6f's Repeat added host completion/shared-event waits (metal_lower.mm1837-1857, metal_program.h239-250) absent from the prior queued GPU predication. That Repeat and its count/condition buffers are removed.
- One Halley helper emits the static Cholesky schedule and bounded continuation copies. Continuation extent is e while unconverged and zero afterward; inactive copies retain their iterate and false unconverged flag.
- The exact JAX norm>tol predicate and its returned negation remain, including NaN comparison semantics; static schedule, max_iterations, active extent, final refinement unchanged.
- Source +26/-24 relative to c5df7c6f; supervisor AST/diff PASS, pyflakes only the same two preexisting unused locals. No import/build/target in this corrective unit.
- Zero-extent dispatches remain. Full device-side control specialization is unresolved; no faster-execution claim or fallback mode was introduced. The rejected loop-form observation was cancelled.
