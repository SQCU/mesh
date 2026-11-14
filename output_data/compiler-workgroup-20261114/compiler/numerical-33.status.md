# numerical-33: ZTREVC('R','B') + ZLATRS('U','N','N','Y') source routine (SAME64), intermediate checkpoint, HOLD

**Added** to `compiler/mesh_compiler/operations.py` (uncommitted): `_ztrevc(h, z, vr, ops)`.
- **Operands:** h is the upper triangular Schur form (read only; its diagonal stays the source's original T), z the
  Schur vectors, and vr receives the eigenvectors. There is no `_Repeat`, so no host wait per row.
- **Per-KI state:** vectors over KI hold T(KI,KI), SMIN = max(ULP·CABS1, UNFL·N/ULP), TSCAL, GROW, XBND, SCALE and XMAX.
  X (n×n) holds each KI's right-hand side and solution in column KI. The shifted, SMIN-raised diagonal
  T(J,J) − T(KI,KI) is formed on the fly in every record that needs it; T is never mutated.
- **CNORM:** each column's DZASUM, then TMAX (and TMAX2 from the max(|Re|,|Im|) column maxima) over each KI's leading
  block. TSCAL takes zlatrs' cases: 1; HALF/(SMLNUM·TMAX); 1/(SMLNUM·TMAX2); or the ZTRSV return.
  - Case 2 recomputes CNORM as Σ (2·TSCAL)·CABS2(A(I,J)), each summand scaled before the sum. That record runs only
    when some KI is in case 2 (extent `any2`). The per-(J, KI) CNORM is chosen from these.
- **GROW/XBND:** zlatrs 442–504 as static rounds J = n−1…1 over columns KI > J. The GO TO 60 exit keeps GROW, a completed
  loop takes GROW = XBND, and TSCAL ≠ 1 gives GROW = 0. ZTRSV is chosen where GROW·TSCAL > SMLNUM, or on case 2's
  return.
- **Careful-solve init:** SCALE = (BIGNUM/2)/XMAX with XMAX = BIGNUM, or XMAX = 2·XMAX. X = −T(1:KI−1, KI) scaled by
  that SCALE.
- **Solve rounds:** J = n−1…1, each over the actual columns KI = J+1..n. Each round runs:
  - **Pivot:** a per-column record does the ZTRSV division, or the careful REC₁ (both TJJ branches with the CNORM > 1
    division), ZLADIV, the A(j,j) = 0 branch (X = e_J, SCALE = XMAX = 0), and REC₂ (1/(2·XJ) or HALF), with SCALE and
    XMAX updated as the source does;
  - **Below:** rows 1..J−1 take their REC₁·REC₂ scaling (applied in sequence) and the ZAXPY with −X(J)·TSCAL;
  - **Solved rows:** rows J+1..KI−1 are rescaled only when some column scaled (extent `need`);
  - **XMAX:** the largest CABS1 of X(1:J−1).
- **Back-transform:** SCALE/TSCAL, ZTRSV's 1, or 1 for KI = 1 becomes X(KI,KI). Then VR = Z·X (`_gemm`), and each column
  is scaled by 1/(its largest CABS1).

**Known differences:**
- **Summation order:** the DZASUM and case-2 sums and ZGEMV's accumulation use reduction order.
- **Extra work:** the strict-triangle masks double the one-time n² records. The GEMM also runs over X's zero lower
  part; it is exact because those entries are zero.
- **CNORM restore:** the source's in-place CNORM·TSCAL·(1/TSCAL) restore between KI calls is not reproduced, because
  each KI reads its own CNORM.

**Unverified:** grep confirms `_ztrevc` is defined once. I read it through for syntax and indexing. Nothing was run,
imported or built, and I did not run AST.

**Supervisor notes:** cba469af flags `_larfg2` as a second producer next to `_linalg_reflector`, and comments as more
than citations. Unchanged.

**Not done:** ZGEEV scaling and undo, ZHSEQR's isolated W, ZGEBAK, Euclidean/phase normalization, the target, the
caller and the Check. Public eig remains unfinished.

Supervisor checks: AST, no duplicate top-level definitions and diff whitespace pass. Actual FakeTensor construction for a 3×3 complex64 Schur form emits 26 records and no Repeat. Source triangular scaling/order reviewed against the retained ZLATRS reference; this is construction evidence only, not numerical execution. Public non-CPU complex eig remains unconnected until the GEEV caller lands.
