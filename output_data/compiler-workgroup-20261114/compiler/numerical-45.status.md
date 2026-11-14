# Numerical source-branch audit: no source edit, HOLD
- Finite candidates reviewed in operations.py; this is not an exhaustive claim about every compiler branch.
- LAPACK dgels.f 498-514, dgelsy.f 463-479, dgelss.f 743-755 and dgelsd.f 621-633 contain the same IASCL/IBSCL common-scale choice as _linalg_least_unscale; the range and zero-matrix cases remain source cases.
- LAPACK gelss/dlalsd thresholding, dlapy2/dlapy3 zero/NaN cases, gebal iteration/NaN exit, and getf2 zero-pivot handling justify the inspected conditions.
- Retained JAX qdwh/svd sources justify Newton-Schulz refinement, singular-value clipping, nonfinite results, and static shape/job dispatch.
- Supervisor read pub13 sections 5.1-5.4 directly from the retained primary PDF text: Algorithm 3 checks backward error, recognizes stagnation, and supplies Haar restart and polar refinement or a changed shift. These are not arbitrary retry guards. Local mean choice is an implementation of the permitted changed shift, not a literal paper formula.
- Missing source control remains separate: scalar Padé switch, overflow skip before expm work, and polar continuation early exit. No new control representation was introduced here.
- operations.py/decompose.py remain ee6f0294; no numerical source edits, imports, builds, or targets in this audit.
