# numerical-32: ZLAHQR source routine (SAME64), intermediate checkpoint, HOLD

**Added** to `compiler/mesh_compiler/operations.py` (uncommitted): `_zlahqr(h, z, w, ilo, ihi, info, ops, held=())`,
following zlahqr.f 250–573, plus two region helpers.
- **Setup:** WANTT = WANTZ = (z given), with ILOZ = ILO and IHIZ = IHI. INFO = 0, and n = 0 returns.
- **Prologue:** the trash, H(J+2,J) for J = ILO..IHI−2 and H(J+3,J) for J = ILO..IHI−3, is zeroed by offset records.
  Then the subdiagonals I = ILO+1..IHI are made real in order: SC = conj(H/CABS1)/|·|, H(I,I−1) = |H|, row I over I..JHI
  by SC, and column I over JLO..min(JHI, I+1) and Z by conj(SC).
- **One `_Repeat`, one DO 130 iteration per pass:** labels 30 and 140 are flattened. Deflation sets W(I) = H(I,I) by
  scatter, KDEFL = 0, I = L−1, then L = ILO and ITS = 0. ITS therefore runs over 0..ITMAX for each I, and INFO = I
  past ITMAX = 30·max(10, NH), with NH from ILO and IHI. Each pass runs:
  - **K search:** over L+1..I, SMLNUM = SAFMIN·(NH/ULP), with Ahues–Tisseur and its literal ILO/IHI guards. The first
    hit from the top is taken as the largest, so L = max(found, L), and H(L, L−1) = 0 is a scatter.
  - **Shift:** exceptional when mod(KDEFL, 20) = 0 or mod(KDEFL, 10) = 0 (DAT1 = 0.75); otherwise Wilkinson, with the
    413–418 sign rule and Smith's division for ZLADIV. I1/I2 are set when ¬WANTT.
  - **M search:** over L+1..I−1, giving V = (H11S/S, H21/S) at M.
  - **Bulge positions K = 1..n−1:** static and unrolled, with no per-reflector host wait. One vector record gives each
    position's actual extents. Each position runs:
    - ZLARFG(2), in `_larfg2`, transcribed literally in one region (DLAPY3, the KNT loop of up to 20 passes, the
      HUGE/2 branch and the undo), with T2 = Re(T1·V2);
    - writes of H(K,K−1) = beta and H(K+1,K−1) = 0 when K > M;
    - rows K, K+1 over K..I2, columns K, K+1 over I1..min(K+2, I), and Z over ILO..IHI;
    - at K = M > L, the 510–534 phase scaling as one in-place region over rows I1..I × columns M..I2, in the source's
      per-entry order, plus Z's columns M..I with J ≠ M+1.
  - **Finish:** make H(I, I−1) real (a scatter of RTEMP, then row I by conj(TEMP), and column I and Z by TEMP), then
    ITS/INFO.
- **ILO = IHI:** the source returns early; here the same result comes from the general path (the empty prologue ranges,
  then the first pass's exit).
- **Helpers:** `_dlapy3` (dlapy3.f is not among the supplied sources; it follows the standard formula with the HUGEVAL
  guard) and `_larfg2`.

**Supervisor checks:** AST/no duplicate top-level definitions/diff whitespace pass. Actual3×3 complex64 FakeTensor
construction produces12 outer records and one numerical Repeat with25 body records. The source phase updates and
ITS/INFO domain were reviewed against ZLAHQR. This is source construction only, not numerical evaluation or native
execution. DLAPY3 source is now retained by URL/hash in the manifest; its largest-magnitude/overflow formula matches.

**Supervisor notes:** the 646dda95 measurement flags `_reflected` as a second producer next to `_linalg_reflect`, and
comments that are more than citations. Both are left as they are.

**Not done:** ZHSEQR's isolated W and its ZLAQR0 path, GEEV scaling, ZTREVC/ZLATRS, ZGEBAK, normalization, the target,
the caller and the Check. Public eig remains unfinished.
