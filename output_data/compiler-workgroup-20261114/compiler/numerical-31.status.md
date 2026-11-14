# numerical-31: active Hessenberg reduction and Q (SAME64), intermediate checkpoint, HOLD

**Added** to `compiler/mesh_compiler/operations.py` (uncommitted):
- **`_zgehd2(h, z, ilo, ihi, ops) -> tau`:** follows zgehrd.f 249–254 and 295–299, zgehd2.f 190–207, and (when z is
  given) zunghr.f 198–232 with zung2r's reflector sequence. It emits no `_Repeat`, so it takes no `held`.
  - **Column sequence:** the static columns I = 1..n−1 are unrolled. Activity and ranges come from one vector record over
    ILO and IHI: the reflector length IHI−I, its tail IHI−I−1, the right-apply rows IHI and the left-apply columns
    N−I. An inactive column has every extent 0.
  - **Reflector:** ZLARFG is the existing `_linalg_reflector` on h[I+1:, I]. Its `sizes` clip is keyed by its only two
    bounds, the column and its tail, which are always distinct. Its complex records have element loops only (`where`
    per component, the rest through `_complex`), so no component loop is clipped.
  - **Applications:** H(I) from the right to A(1:IHI, I+1:IHI), then H(I)ᴴ (conj τ) from the left to A(I+1:IHI,
    I+1:N). Each takes explicit per-loop-position extents, so the square trailing block's IHI-limited rows and its
    N−I columns keep their own ranges.
  - **Stored state:** TAU(I) is τ where active, else 0 (ZGEHRD). Beta (real) goes into A(I+1, I), and v(2:) into
    A(I+2:IHI, I) clipped.
  - **Q:** z = I, then H(I) in reverse order from the packed h column and TAU to z rows and columns I+1..IHI (ZUNG2R's
    active block).
- **`_reflected(block, v, tau, rows, cols, ops, left, adjoint)`:** ZLARF1F with v(1) = 1 held explicitly. It is two
  `_record`s with explicit loop extents, the same arithmetic association as `_linalg_reflect`.
- **`_linalg_reflector` correction:** the XNORM = 0 test is now `tail <= 0`. A static tail is nonnegative, so it is
  unchanged; a clipped empty tail's −inf amax identity now gives H = I, and the complex case still requires
  Im α = 0.

**Not done:** ZGEHRD's blocked ZLAHR2/ZLARFB path, the large-n efficiency boundary, is not transcribed.

**Supervisor checks:** AST, duplicate top-level names, and diff whitespace pass. The actual typed producer constructs
81 records, no repeats, and tau complex64[2] for a 3×3 complex64 FakeTensor member with optional Q. This is source
construction only, not a numerical evaluation or native target run. Source review checked the right/left update ranges
and conjugated tau against ZGEHD2, and the reversed Q reflector sequence.

**Open review note for the supervisor:** the 983afe62 measurement flags `_clipped_norm`/`_record_extreme` as a second
producer (of `_linalg_lengths`/`_reduction`) and flags their comments as more than citations. I left them unchanged
this turn.

**Not done (later SAME turns):** ZGEEV scaling, ZLAHQR, ZTREVC/ZLATRS, ZGEBAK, normalization, the fake target, the
lowering hook and the decompose caller. `_zgebal` and `_zgehd2` have no caller yet, and public non-CPU complex eig
remains unfinished.
