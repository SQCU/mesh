# numerical-37: DLANV2 real 2×2 Schur kernel (SAME64), intermediate checkpoint, HOLD

**API** (`compiler/mesh_compiler/operations.py`, uncommitted): a region-level callable that DLAHQR's scalar records call
directly. It is lossless: region values in, region values out.
- `_dlanv2(r, t, a, b, c, d, limits) -> (A, B, C, D, RT1R, RT1I, RT2R, RT2I, CS, SN)`, where t is F32 or F64 and
  `limits` is the real dtype's finfo.
- `_dlapy2(r, t, x, y, huge)`: dlapy2.f with its NaN pass-through and the Z = 0 or W > HUGEVAL guard.
- `_fortran_sign(r, t, a, b)`: SIGN(A, B), where a zero B takes the sign of 1/B, so −0 counts as negative.

**Method:** dlanv2.f 159–306, each branch computed and then selected in one region:
- **C = 0:** CS = 1, SN = 0.
- **B = 0:** the swap (CS = 0, SN = 1, A↔D, B = −C, C = 0).
- **A−D = 0 with SIGN(1,B) ≠ SIGN(1,C):** identity.
- **General:** P, BCMAX, BCMIS, SCALE and Z (187–192).
  - **Z ≥ MULTPL·EPS, real eigenvalues:** Z = P + SIGN(√SCALE·√Z, P), A, D, TAU = DLAPY2(C, Z), CS, SN, B − C, C = 0.
  - **Otherwise:** SIGMA = B + C, then the label 10 SAFMN2/SAFMX2 rescaling, unrolled to its own bound (COUNT ≤ 20,
    so at most 21 passes). Then P, TAU = DLAPY2(SIGMA, TEMP), CS and SN, and the AA/BB/CC/DD and A/B/C/D products.
    The issue-1031 parenthesized B and C are kept as separate products.
  - **After that:** A = D = HALF·(A + D). Where C ≠ 0, either the same-sign real-pair split (SAB, SAC, P, TAU,
    CS1/SN1 and the rotation product), or the B = 0 turn (B = −C, C = 0, CS = −SN, SN = CS).
- **Results:** RT1R = A, RT2R = D, and RT1I = √|B|·√|C| = −RT2I unless C = 0.
- **Constants:** SAFMN2 = 2^INT(LOG(SAFMIN/EPS)/LOG(2)/2) uses the exact power-of-two exponent of the F32/F64 machine constants; no tensor execution or scalar extraction occurs during fake construction.
- **No contraction:** the caller's record must have no combiner, so `_record` keeps precision 0 (no
  REASSOCIATE/CONTRACT).

**Notes:**
- Supervisor downloaded/read canonical dlapy2.f and verified every branch/NaN ordering; its hash is retained in real-eigen-references.json.
- `_fortran_sign` follows gfortran's default −0 sign, assuming the native compile keeps IEEE zero/infinity semantics.

**Supervisor review:** AST and source branch comparison pass; pure machine-constant computation replaces the worker’s fake-incompatible torch.log(...).item(). Actual F32/F64 FakeTensor→one structured record construction passes:10 outputs, precision0,927 region instructions each (real-eigen-schur2-construction.json). No native numerical result is claimed at this intermediate checkpoint.

**Remaining (next SAME turns):**
- DLAHQR, which calls `_dlanv2` in its 2×2 deflation;
- the real driver: DGEBAL/DGEHD2/DORGHR, with the existing complex owners shared only where the algorithms coincide;
- DTREVC/DLALN2 conjugate-pair recovery, DGEBAK, and DGEEV normalization;
- the decompose caller for real non-CPU eig.

Public real eig remains unfinished until the actual native output passes.
