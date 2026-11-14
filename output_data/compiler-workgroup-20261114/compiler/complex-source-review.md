# Complex Hermitian QDWH source checkpoint

The existing outlined QDWH owner now carries complex matrix state, real norm/trace/shift state, true adjoints, and the existing complex QR/Householder operations. Complex real-pair operand views are retained on each numerical record; dynamic offsets and Work axes consume those actual view occurrences. Shape-only source-axis correspondence is separate. Haar restart draws use the same source site and retry stream with independent real/imaginary elements of variance one half.

The prefix reflects UPLO’s triangle and discards imaginary diagonal entries. The same complex QR also reaches the existing SVD correction, but no complex SVD invocation is claimed. CPU LAPACK selection is unchanged.

Primary formulations: JAX `_eigh_qdwh_impl` and QDWH sources, Reference-LAPACK complex QR through existing owners, and Mezzadri, arXiv:math-ph/0609050 §5 equation 5.12. Existing QDWH-eig split/acceptance and retry semantics remain those of the cited source; this change does not establish its open finite-precision degeneracy premise.

Supervisor review removed an unsafe global view memo, kept real-pair correspondence separate from shape provenance, renamed new helpers that collided with FFT and another lowering, and used the existing scaled complex modulus for polar normalization. AST and duplicate-definition checks pass. The actual 2×2 complex64 capture→placement→outline→device stage passes with a prefix, three calls of the shared Step, and a suffix; source site 1 is retained, outputs are float32 `[2]` and complex64 `[2,2]`. No numerical evaluator was used.

The one direct native case then passed: complex64 2×2, UPLO=L, deliberately inconsistent upper triangle and nonzero imaginary diagonal. Eigenvalue error 4.37366e-8, residual 5.94645e-8, orthogonality 7.79241e-8, finite. Compilation, native archive reload (17 rows, no relation solve), and invocation completed with exit 0 and empty stderr. `complex-native-result.json` records metrics and raw log hashes; all source and binary hashes stayed unchanged. This establishes this case, not complex SVD/pinv/lstsq, complex128, ANE symbolic maps, degenerate retries, or performance.
