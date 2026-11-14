# Real Schur iteration source checkpoint

The new `_dlahqr` consumes one real Hessenberg matrix, optional real Schur vectors, separate real/imaginary eigenvalue arrays, the original ILO/IHI, and INFO. It uses the existing typed records, dynamic extents and Repeat owner. Source: Reference-LAPACK DLAHQR, with the existing DLANV2 pair conversion.

Supervisor review removed a second reflector implementation: each active two/three-element bulge vector now goes through `_linalg_reflector`. That common owner uses maximum-modulus normalization rather than LAPACK's repeated SAFMIN scaling; rounding and signed-zero reflector choices can differ. No duplicate scalar implementation remains. Pair deflation alone enters the DLANV2 once-gate. NR2 and NR3 update records have separate actual extents; column K−1 stores follow the original branch conditions.

Current source parses. Actual FakeTensor construction for n=3 succeeds for float32 and float64: 78 leaf records and two Repeat regions each (real-eigen-dlahqr-construction.json). This validates construction, not numerical behavior. DLANV2's bounded scalar rescaling expansion remains explicit; no early-exit or optimality claim is made.

The public real eig/eigvals consumer is still incomplete. The next unit is DLALN2/DTREVC recovery, shared balancing/Hessenberg/backtransform and driver selection, then the existing public native caller. No extra evaluator or target run was used for this checkpoint.
