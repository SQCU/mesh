> Supervisor review: the old-record method result below is transitional and not accepted as the final boundary. SAME211 numerical67b is replacing retained `_Region/_Generic` compiler machinery with actual upstream method bodies while preserving cited algorithms. Producer/consumer must not treat FallbackKernel registration plus old generic records as completed replacement.

# numerical67(b): custom-method bodies and operands (operations.py, decompose.py)

Session 211a403e, turn numerical67b (consolidated brief), on top of supervisor checkpoint 8cbc2ab9. The supervisor
has since preserved five WIP checkpoints, none of which accepts the method:
- 43aa1f2e: DLADIV;
- c4bf33fe: the LAQR1/LAQR5/LARFG group;
- e908087a: the LAQR5 stage order and the LAHQR argument lists;
- 4d5d3436: the xLAQR3 dependency group (`_larf1f`, `_gehrd`'s ZGEHD2, `_zlartg`, `_rotated`, `_ztrexc`, `_unm2r`,
  `_unmhr`);
- 53af5eda: the complex xLAQR3/xLAQR0/ZHSEQR group below and GEEV's complex ZHSEQR call. Its xLAQR3 DO 60/70/80 and
  xLAQR5 DO 150/160/170 were whole-slab products with a synthesized last-chunk workspace residue (`_times_right`,
  `_times_left`, `_last_chunk`). The supervisor rejected that as a manual rewrite of the source's chunk sequence.

The operations.py delta on top of 53af5eda is uncommitted:
- It replaces those three helpers with the source's per-chunk passes (`_rows_chunk`, `_columns_chunk`), used by
  `_laqr3` and `_laqr5_columns`. It also drops the two functions' unused locals and rewrites their comments.
- `_zlartg`'s S now follows v3.12.1 zlartg.f90 206-208, 226, 262-264 and 282: S = CONJG(G) (F / SQRT(F2 H2)), else
  CONJG(G) (R / H2), and S = CONJG(G) (F / D). The checkpointed (F / SQRT(F2)) (CONJG(G) / SQRT(H2)) was master's.
- `_larf1f` no longer has the M ≤ 0 or N ≤ 0 return. That return is master's; v3.12.1 zlarf1f.f 196-299 and
  dlarf1f.f 193-288 have none. Every caller passes M, N ≥ 1 at its source call site.

No native import, build or workload has been run.

Source provenance found during this delta:
- The /tmp/n67c and /tmp/n67e batches were fetched from Reference-LAPACK `master`, not v3.12.1.
- The v3.12.1 copies are now in /tmp/n67i/lapack (n67c's names, plus ztrevc3.f and dtrevc3.f) and /tmp/n67j/lapack
  (n67e's names).
- After normalizing spelling (`IMPLICIT NONE`, DCONJG/CONJG, DIMAG/AIMAG, DCMPLX/CMPLX KIND), the executable
  differences are:
  - zlartg.f90's S (corrected above);
  - xLARF1F's M,N ≤ 0 return (corrected above) and INCV < 0 indexing (INCV = 1 here);
  - xLARF's INCV indexing;
  - xUNMQR/xORMQR's LWKOPT and reduced NB: v3.12.1 has NW·NB + TSIZE and (LWORK − TSIZE)/LDWORK;
  - ilaenv.f's master-only entries;
  - EXTERNAL lists.
- The repo's `*.reference.f` files that match these batches are also master copies.
- zgeev.reference.f's `INFO = -4` NaN/overflow return is master's; v3.12.1 zgeev.f has none (OPEN below).
- The line citations written from the master batches are mostly 1-29 lines off v3.12.1. Remapping them is OPEN; only
  this delta's own new citations are v3.12.1 numbers.

Checks were static only:
- `py_compile`;
- undefined-name and unused-import checks;
- duplicate top-level definitions;
- call arity against definitions;
- the reachability cut from every name other modules import.

No body has been traced or executed. The read-only review delegates were stopped. The supervisor's own source review
covered DLADIV (43aa1f2e) and the LAQR5 stage order (the c4bf33fe review, applied below). The semantic owner is
Reference-LAPACK v3.12.1 SRC, held read-only and never executed. As recorded above, the n67c/n67e batches compared
against before this delta were master copies.

## Removed this turn: the record compiler and its method path

`operations.py` no longer contains any `_Region`/`_Generic` record machinery, and no method goes through records.

- **eig/eigvals record method:** `mesh::geev` (define and fake `_geev_shapes`), `GEEV_EIGEN`, `_geev` (with its
  NotImplementedError dtype rejection), and its record routines:
  - `_gebal`, `_reflected`, `_gehd2`, `_dlanv2`, `_dlaln2`, `_larfg2`;
  - `_zlahqr`, `_dlahqr`, `_dtrevc`, `_ztrevc`, `_gebak`, `_geev_normalized`.
- **Method dispatch:** `Method`, `METHODS`, `method`, `lowered`, `_placed`. `methods(graph_module, lookup=routine)`
  remains, finding typed LAPACK operator nodes.
- **Generator's record advance:** `Generator.stepped()` and `_stepped`, a `_Generic` add on the counter word.
- **Record compiler:**
  - `_Region`, `_Generic`, `_Repeat`, `_Check`;
  - `_record`, `_record_extreme`, `_emit`, `_scalars`, `_gemm`, `_copy`, `_fill`, `_temporary`;
  - `_typed`, `_compute`, `_convert`, `_load`, `_store`, `_on_block`, `_rows`, `_same`, `_aligned`;
  - `_coordinates`, `_linear`, `_bounded_copy`, and the DSL `_at`.
- **Scalar, arithmetic and complex machinery:**
  - `_apply`, `_elementwise`, `_power`, `_powi`, `_erf`, `_signed_zero`, `_floordiv`, `_truncdiv`;
  - `_complex`, `_complex_value`, `_pair`, `_component`, `_part`, `_complex_parts`, `_valued`;
  - `_plus`, `_times`, `_conjugate`, `_negated`, `_square`, `_chosen`;
  - the record `_cabs1`, `_modulus`, `_real_scaled`, `_nonzero`.
- **Reduction machinery:** `_reduction`, `_reducing`, `_reduced`, `_dims`, `_identity` (the record one),
  `_identities`, `_joined`, `_extremum`, `_epilogue`.
- **Record linear-algebra helpers:** `_linalg_reflector`, `_linalg_lengths`, `_linalg_identity`, `_clipped_norm`,
  `_first_largest`, `_swapped`, `_offsets`, `_entry` (the record one), `_divided_by`, `_bounded`, `_all`.
- **Type and opcode tables:**
  - `_ints`, `KINDS`, `CLASS`, `STORED`, `REAL`, `FLOATING`, `INTEGRAL`;
  - `PARALLEL`/`REDUCTION`, `ASSIGN`/`SUM`/`REGION`, `IDENTITY`/`ACCUMULATE`, `REASSOCIATE`/`CONTRACT`;
  - `BINARY`, `COMPARE`, `BOOLEAN`, `COMPLEMENT`/`NEGATION`/`NOT`, `BUILTINS`, `SQRT1_2`, `TWO_SQRTPI`;
  - `_class`, `_floating`, `_bits`.
- **Imports:** `itertools`, which only the per-member record loop used.

`SCALAR` (metal_ir.h element kinds by dtype) and the placement codes `GPU`/`CPU`/`ANE` remain as representation
tables. metal.py, lower_metal.py and schedule.py read them.

## Method bodies in the upstream body (connected)

`decompose.decompose` and `decompose.outline` trace each method into the placed graph: `make_fx` over the node's fake
values, then `graph_copy`. HOP subgraphs become `get_attr` GraphModules. Inductor lowers the result like any other
body.

| method | body (operations.py) | cited algorithm kept |
|---|---|---|
| eig, eigvals off the CPU | `_eigen_geev` → `_eigen(self, vectors, _geev_routine, _geev_routine)` | Reference-LAPACK zgeev.f / dgeev.f, below |
| eig, eigvals on the CPU | `_lapack_eigen` → `_eigen(self, vectors, LAPACK_ZGEEV, LAPACK_DGEEV)` | Accelerate's routine. `_eigen` (INFO check, complex assembly) is shared, so both routes consume routine-shaped results |
| QDWH eigh step (outlined; eigh and QDWH-SVD off the CPU) | `_qdwh_eigh_pop`, `_qdwh_eigh_state`, `_qdwh_eigh_result`; Haar draws `_haar_drawn` with Philox words read through `GENERATOR` | JAX `_eigh_work`, Salmon et al. Philox4x32-10 |
| SVD rank correction | `_rank_corrected`: `while_loop_op` over `(u_out, do_correction)` | JAX tpu/linalg/svd.py 104-118 |
| matrix_exp squaring | `_squaring(r, count, max_squarings)`: static scan over the compile-time `max_squarings`. Each step `cond_op(count > i, _squaring_precise, _squaring_kept)` for one matrix, `where` for a batch | JAX scipy/linalg.py `_squaring` (scan + cond); no runtime fixed-counter loop |
| lstsq range scaling and undo | `_range_scaled`, `_range_unscaled` | dgelss/dgelsd ANRM/BNRM |
| gels / gelsy off the CPU | `_least_squares_qr`: `_geqr2`, `_householder_applied`, `solve_triangular`, `_gelsy_rank`, `_gelsy_solution` | dgeqr2/dgeqp3, dorm2r, dtrtrs, dlaic1, dtzrzf |
| gelsd / gelss off the CPU | `_least_squares_svd` | the SVD-based solution with rcond |
| fragment decode | `_fragment_decoded` | docs/provider.md fragment coding |

### GEEV body (`_geev_routine(a, jobvr)` → LAPACK_ZGEEV's `(w, vr, info)` / LAPACK_DGEEV's `(wr, wi, vr, info)`)

It is ATen plus `while_loop_op`. Each DO WHILE / GO TO loop is a `while_loop_op` that runs while any member's
predicate holds and selects each member's step by its own predicate (JAX `_while_loop_batching_rule`). Static DO
loops are unrolled with per-member masks. There is no dtype subset and no rejection.

- **zgeev/dgeev:**
  - ANRM = ZLANGE('M'), with NaN kept. INFO = -4 when ANRM is NaN or above overflow.
  - SCALEA/CSCALE, then `_lascl`: ZLASCL/DLASCL 'G' with the full DONE loop.
  - With vectors: ZUNGHR (`_unghr`) and ZHSEQR('S','V'); without: ZHSEQR('E','N').
  - Only where INFO = 0: TREVC, ZGEBAK, then normalization. Elsewhere VR is ZHSEQR's Z.
  - Label 50: undo scaling of W(INFO+1:N), and of W(1:ILO-1) when INFO > 0.
- **`_gebal`:**
  - ZGEBAL('B'): the row and column isolation loops and the norm balancing with its two DO WHILE loops.
  - The NaN exit (INFO = -3, ignored by zgeev).
- **`_gehrd(order, ilo, ihi, h)`:** ZGEHRD/DGEHRD's unblocked path, which is zgehd2.f/dgehd2.f 3.12.1 on the leading
  ORDER block. TAU is zero outside ILO..IHI-1. Static I = 1..n-1 is active where ILO ≤ I ≤ IHI-1. Each step:
  - ZLARFG(IHI-I, A(I+1,I), A(MIN(I+2,N),I)) through `_larfg_vector`;
  - ZLARF1F('Right', IHI, IHI-I, …, TAU(I), A(1,I+1));
  - ZLARF1F('Left', IHI-I, N-I, …, CONJG(TAU(I)), A(I+1,I+1)), through `_larf1f`.

  The packed layout is the one `_unghr` reads. GEEV calls it with ORDER = n, and xLAQR3's ZGEHRD(JW, 1, NS, T) is
  its other caller. The blocked DO 40 (NH ≥ 130) is OPEN.
- **ZHSEQR:**
  - W outside ILO..IHI comes from the diagonal (zhseqr.f 415-419). The ILO = IHI return (428-431) gives
    W(ILO) = H(ILO,ILO).
  - Complex: `_zhseqr(WANTT = WANTZ = vectors, ILO, IHI, H, W, Z, proceed)`, which is zhseqr.f 435-487 (see the
    complex xLAQR3/xLAQR0/ZHSEQR group below).
  - Real: DLAHQR alone, `_dlahqr(WANTT = WANTZ = vectors, N, ILO, IHI, H, WR, WI, ILOZ = ILO, IHIZ = IHI, Z,
    proceed)`. dhseqr.f's DLAQR0 selection and retry are OPEN (see coverage).
  - Then H's r − c ≥ 2 part is zeroed when WANTT or INFO ≠ 0, with N > 2 (491-492).
- **`_zlahqr(wantt, wantz, order, ilo, ihi, h, w, iloz, ihiz, z, proceed)`:**
  - zlahqr.f's own argument list. N (`order`) is a runtime batch tensor, so xLAQR3/xLAQR0 can call it on a
    JW-, NS- or HL-sized leading block.
  - The ILO = IHI return.
  - DO 10 trash clearing and DO 20 subdiagonal scaling over JLO..JHI (1..N with WANTT, else ILO..IHI).
  - The iteration loop: deflation search, exceptional shifts (KDEFL), Wilkinson shift, M search.
  - The bulge chase using `_larfg`, with the K = M > L TEMP scalings. Rows I1..I2 (1..N with WANTT, else L..I), and
    Z rows ILOZ..IHIZ only with WANTZ.
  - SMLNUM = SAFMIN·(NH/ULP), ITMAX = 30·max(10, NH).
- **`_dlahqr(wantt, wantz, order, ilo, ihi, h, wr, wi, iloz, ihiz, z, proceed)`:**
  - The same argument list and loop, with the ILO = IHI return (WR = H(ILO,ILO), WI = 0).
  - `_dlanv2` 2×2 standardization, with its DROTs over the WANTT columns and the WANTZ Z rows.
  - The DAT1/DAT2 shifts.
  - The NR = 3/2 reflectors (`_larfg`, `_reflected3`).
- **`_larfg` / `_larfg_vector`:** zlarfg.f / dlarfg.f v3.12.1 as written in that source:
  - N ≤ 0 (complex) or N ≤ 1 (real) gives TAU = 0. XNORM = 0 (and ALPHI = 0) gives TAU = 0.
  - BETA = −SIGN(DLAPY3 | DLAPY2, ALPHR).
  - The KNT ≤ 20 SAFMIN rescaling loop, recomputing XNORM and BETA. TAU, then ZLADIV(1, ALPHA − BETA) through
    `_divided`, or 1/(ALPHA − BETA). BETA is unscaled KNT times.
  - 3.12.1 has no BETA > HUGE/2 branch, so none is written.
  - Real and complex share one body, in the input's own precision.

### xLAQR0 sweep dependencies (complex: called by `_laqr0_pass`; real: not yet called)

These go between `_dgeev_normalized` and `_geev_routine`. Each is cited to zlaqr1.f/zlaqr5.f and dlaqr1.f/dlaqr5.f
v3.12.1. The complex branch is reached through `_laqr0_pass`. The real branch waits for DLAQR0, which is OPEN.

- **`_laqr1(e, order, first, second)`:** ZLAQR1/DLAQR1 for N = 2 and N = 3. The real form keeps the source's
  association.
- **`_laqr5(wantt, wantz, accum, order, ktop, kbot, nshfts, shifts, h, iloz, ihiz, z, ku, nv, kwv, nh, kwh, proceed)`:**
  - xLAQR5 with KACC22 = `accum`, a runtime N, and the source's U, WV and WH regions emulated in place in H at
    xLAQR0's KU/KWV/KWH offsets.
  - Real: the DO 10 shift pairing. Complex: CABS1 and conjugates.
  - DO 180 INCOL is a while_loop (`_laqr5_columns`). Inside it, DO 145 KRCOL is a nested while_loop
    (`_laqr5_step`).
  - Each KRCOL step keeps the reference's stage order:
    - the BMP22 2×2 bulge (`_laqr5_pair`);
    - DO 80 (`_laqr5_bulge`);
    - DO 100/90, rows of the far columns MAX(KTOP, KRCOL + 2M)..JBOT (`_laqr5_rows`);
    - DO 120 into U (with ACCUM), or DO 140 into Z (`_laqr5_products`).
  - Each stage is its own while_loop over M. They are not fused (supervisor's c4bf33fe review).
  - xLAQR5 reaches H(KU,1) only through its argument U. U is held as its own array through ZLASET and DO 145, then
    put into H(KU,1) before DO 150.
  - zlaqr5.f 765-810 (where ACCUM) sets JTOP/JBOT, K1 and NU. It then runs DO 150, DO 160 and DO 170 (WANTZ) as
    while_loops, one source chunk per pass, in the source's order:
    - DO 150 (`_columns_chunk`): ZGEMM('C', 'N', NU, JLEN, NU) of U(K1,K1) and H(INCOL+K1,JCOL) into H's own WH
      positions, then ZLACPY back to H(INCOL+K1,JCOL);
    - DO 160 and DO 170 (`_rows_chunk`): ZGEMM('N', 'N', JLEN, NU, NU) of H(JROW,INCOL+K1) or Z(JROW,INCOL+K1) by
      U(K1,K1) into H's own WV positions, then ZLACPY back.
  - `_vigilant` is the source's vigilant deflation check.

### xLAQR3 dependency group (checkpoint 4d5d3436; called by `_gehrd` and `_laqr3`)

- **`_larf1f(left, m, n, v, tau, x, r0, c0, when)`:** v3.12.1 zlarf1f.f 196-299, and dlarf1f.f 193-288 in the same
  body. There is no M ≤ 0 or N ≤ 0 return (master's):
  - LASTV, scanning down past V(1) while V(LASTV) = 0, where TAU ≠ 0;
  - LASTC = ILAZLC(LASTV, N, C) or ILAZLR(M, LASTV, C), where a NaN counts as nonzero; the LASTC = 0 return;
  - with LASTV = 1, the ZSCAL by ONE - TAU;
  - else the ZGEMV, the first-row (left) or first-column (right) correction (zlarf1f's loops or ZAXPYs, dlarf1f's
    DAXPYs), and ZGERC/DGER skipping a zero WORK(J) (left) or a zero v(J) (right).
- **`_zlartg`:** v3.12.1 zlartg.f90 147-288, branch by branch, with S as at 206-208, 226, 262-264 and 282.
- **`_rotated`:** zrot.f DO 30 (drot.f alike), with the real C's products taken by parts.
- **`_ztrexc(wantq, order, t, q, ifst, ilst, when)`:** ztrexc.f 182-224.
  - The N ≤ 1 / IFST = ILST return.
  - DO 10 as one while_loop (`_ztrexc_pass`), forward or back per member. Each step: ZLARTG(T(K,K+1), T22 - T11),
    the row and column ZROTs, the swapped diagonal, and Q's ZROT with WANTQ.
  - dtrexc.f (DLAEXC, DLASY2, DLARFX, DLANV2) is OPEN, so this group's real branch has no xTREXC yet.
- **`_unm2r`:** zunm2r.f 222-271 and dorm2r.f. The loop direction follows LEFT and NOTRAN; MI/IC or NI/JC, TAUI, then
  ZLARF1F on C(IC,JC).
- **`_unmhr`:** zunmhr.f 212 and 262-284 (dormhr.f alike).
  - NH = IHI - ILO and its return.
  - ZUNMQR(MI, NI, NH, A(ILO+1,ILO), TAU(ILO), C(I1,I2)), reaching zunmqr.f 274-279's ZUNM2R where NB = 32 ≥ K.
  - The blocked ZLARFT/ZLARFB path (K > 32) is OPEN.

### Complex xLAQR3 / xLAQR0 / ZHSEQR group (checkpoint 53af5eda, chunk passes uncommitted; called by `_geev_routine` for complex)

These go after `_unmhr` and before `_geev_routine`. Each comment cites its zlaqr3.f, zlaqr0.f, zhseqr.f and iparmq.f
3.12.1 line ranges.
- **`_laqr3(wantt, wantz, nested, size, order, ktop, kbot, nw, h, iloz, ihiz, z, w, kv, kt, nho, kwv, nve, proceed)`:**
  zlaqr3.f 361-577 in source order. It returns H, Z, W (SH), NS and ND.
  - `nested` False is zlaqr2.f. Checked by diff of the executable statements, zlaqr2.f differs only at zlaqr3.f
    416-425 (ZLAHQR for every JW).
  - V = H(KV,1), T = H(KV,KT) and WV = H(KWV,1) are zlaqr0.f 516-520's regions. Each is SIZE × SIZE, with SIZE =
    (n − 1)/3 ≥ NWMAX ≥ NW ≥ JW.
  - ZLAQR3 reaches H(KV,KT) and H(KV,1) only through its arguments T and V. They are taken from those positions at
    entry and put back there after ZUNMHR, their last write before DO 60.
  - zlaqr3.f 527-563 then runs DO 60, DO 70 (WANTT) and DO 80 (WANTZ) as while_loops, one source chunk per pass:
    - DO 60 and DO 80 (`_rows_chunk`): ZGEMM('N', 'N', KLN, JW, JW) of H(KROW,KWTOP) or Z(KROW,KWTOP) by V into H's
      own WV positions, then ZLACPY back;
    - DO 70 (`_columns_chunk`): ZGEMM('C', 'N', JW, KLN, JW) of V and H(KWTOP,KCOL) into H's own T positions, then
      ZLACPY back.
  - In order:
    - the KTOP > KBOT and NW < 1 returns;
    - SMLNUM, JW, KWTOP and S, then the KBOT = KWTOP window;
    - ZLACPY('U') and ZCOPY into T, and V = I;
    - ZLAQR4 where JW > NMIN = 75 and `nested`, else ZLAHQR;
    - DO 10 (`_laqr3_deflated`: the spike-tip test, else ZTREXC(NS → ILST));
    - S = 0 where NS = 0;
    - DO 30 (`_laqr3_sorted`: DO 20's first strict maximum, `_first_largest`, then ZTREXC);
    - DO 40;
    - the NS < JW or S = 0 block: the spike ZLARFG, ZLASET('L'), the three ZLARF1Fs, ZGEHRD, H(KWTOP,KWTOP−1), T
      back into the window, ZUNMHR, then DO 60, DO 70 (WANTT) and DO 80 (WANTZ);
    - ND, and NS − INFQR.
- **`_laqr0(wantt, wantz, four, order, ilo, ihi, h, w, iloz, ihiz, z, proceed)`:** zlaqr0.f 323-693.
  - `four` True is zlaqr4.f. By the same diff, it differs only at zlaqr0.f 381 and 516 (ZLAQR2 for ZLAQR3) and
    572-580 (ZLAHQR for every NS).
  - The N = 0 return. N ≤ NTINY = 15 gives ZLAHQR (alone where the static n ≤ 15).
  - From iparmq.f (`_iparmq_shifts`, iparmq.f 259-279, with single-precision NINT(LOG/LOG(TWO))):
    - NWR (ISPEC 13: NS when NH ≤ 500, else 3NS/2) and NSR (ISPEC 15);
    - NMIN = 75 and NIBBLE = 14;
    - KACC22 (ISPEC 16, the 'LAQR' branch: 2 when NS ≥ 14, else 0), so ACCUM is NS ≥ 14.
  - ITMAX = MAX(30, 2·KEXSH)·MAX(10, NH), and KBOT = IHI.
  - DO 70 is one while_loop (`_laqr0_pass`, zlaqr0.f 439-690). Each pass:
    - finds KTOP, then NW with NDEC, then KV/KT/NHO/KWV/NVE;
    - calls ZLAQR3 or ZLAQR2, then sets KBOT − LD and KS;
    - tests whether to sweep, then sets NS;
    - takes the exceptional shifts (DO 30, WILK1 = 0.75), or the computed ones:
      - the NS × NS ZLACPY to H(KT,1) through ZLAQR4 or ZLAHQR (static NSB = (n − 3)/6 made even ≥ NSMAX);
      - the 2 × 2 fallback shifts;
      - DO 50/DO 40's bubble sort (`_laqr0_sorted`, `_laqr0_swapped`);
    - applies the two-shift rule, then NS, KS, KDU, KU, KWH, NHO, KWV and NVE;
    - calls `_laqr5`, then updates NDFL.
  - INFO = KBOT once IT passes ITMAX. This includes the source's case where that last pass left KBOT < ILO.
  - NWMAX = (N − 1)/3 and NSMAX = (N − 3)/6. The LWORK/2 and 2·LWORK/3 terms are the larger ones, from the call
    graph:
    - ATen's BatchLinearAlgebraKernel.cpp 252-258 (torch 2.14.0, git 08187d9e) passes lapack_work_to_int of the
      query's WORK(1) (204-212);
    - zgeev.f 269 and 298-300: MAXWRK ≥ N + N·ILAENV(1,'ZGEHRD') = 33N (ilaenv.f 349-354);
    - ZHSEQR receives all of it (zgeev.f 352/377/410/420: IWRK = ITAU = 1);
    - it is passed on by zhseqr.f 441-444 and 463-464, zlaqr0.f 516-520 and 573-575, and zlaqr3.f 418-420;
    - zhseqr.f 480-482's HL call passes NL = 49, so 16 < 24 and 7 < 32.
    - The ATen source was fetched read-only to /tmp/n67h/aten.
- **`_zhseqr(wantt, wantz, ilo, ihi, h, w, z, proceed)`:** zhseqr.f 435-487.
  - Static N = n > NMIN = 75: ZLAQR0.
  - Else ZLAHQR. Where INFO > 0, KBOT = INFO and ZLAQR0(…, ILO, KBOT, …, ILOZ = ILO, IHIZ = IHI):
    - on H when N ≥ NL = 49;
    - else on HL, the 49 × 49 copy: H's leading block, HL(N+1,N) = 0, and columns N+1..49 zero.
  - HL's other rows N+1..49 left of column N+1 are uninitialized locals in the source. ZLAQR0/3/5 reach them only as
    workspace that is written before it is read. They are zero here.
  - W and Z are padded to 49 for the static extent. ZLAQR0 reaches W(1:KBOT) and Z rows ILOZ..IHIZ × columns
    1..KBOT, with KBOT ≤ N.
  - HL is copied back where WANTT or INFO ≠ 0.
- **Static branch presence:** ZLAQR4 inside `_laqr3` is present only where SIZE > 75, and inside `_laqr0_pass` only
  where NSB > 75 and not `four`. `_laqr0`'s main path is present only where n > 15, and `_zhseqr`'s three paths
  follow the static n. In each case the omitted source branch's runtime condition cannot hold within the static
  extent.
- **`_ztrevc` (ztrevc.f with zlatrs.f, ZTREVC3's blocked back-transform VR = Z X):**
  - Shifted diagonal raised to SMIN.
  - ZLATRS's in-out CNORM: each KI's call scales CNORM by TSCAL and returns it ×1/TSCAL (except case 2's ZTRSV
    return). A static scan KI = N..2 carries the drifted norms the next KI reads.
  - TSCAL cases 1/2/3 and GROW/XBND.
  - The ZTRSV-or-careful solve with its REC rules and the A(J,J) = 0 null vector. ZAXPY's ZA = 0 return.
  - XMAX recomputation, SCALE/TSCAL, X(KI,KI).
  - Columns normalized by 1/max CABS1.
- **`_dtrevc` (dtrevc.f solve, DTREVC3's back-transform):**
  - Per-column kind: real, or first/second of a pair.
  - The pair's initial 2×2 solve.
  - Static block rounds J = N-1..1, with `_dlaln2` (dlaln2.f: complete pivoting, SMINI, every overflow test).
  - BIGNUM/XNORM rescaling, DSCAL of WORK(1:KI), DAXPYs, then Z X and EMAX normalization.
- **`_gebak`:** ZGEBAK/DGEBAK('B','R'): scaling, then the sequential row interchanges.
- **Normalization:**
  - `_zgeev_normalized`: zgeev 492-503 (1/DZNRM2, then TMP = conj(v_K)/sqrt(RWORK_K), with v_K real).
  - `_dgeev_normalized`: dgeev 505-524 (1/DNRM2; for a pair, 1/DLAPY2 of the two DNRM2s, then IDAMAX of
    VR(K,I)² + VR(K,I+1)², DLARTG/DROT, VR(K,I+1) = 0). The earlier record body used zgeev's phase for pairs, which
    forces VR(K,I) ≥ 0. DLARTG keeps its sign.
  - `_dlartg`: LAPACK 3.10+ dlartg.f90.
- **INFO:** `_eigen` passes INFO to `aten._linalg_check_errors(info, 'torch.linalg.eig' | 'torch.linalg.eigvals',
  is_matrix)` after the results are computed.

## RNG state (connected; the consumer applies the advance)

- `GENERATOR = mesh::generator(site)` has a fake only. A body's Philox draws read the program's Generator words
  through it. It is the explicit operand, never a fresh allocation and never Inductor's RandomSeeds.
- `Generator(state)`:
  - `.site(identity)` records a draw: the source draw identity, else a construction-order number.
  - `.counter` is word 2 of `state`, a view of the same storage.
  - `.advanced()` returns `Advance(counter, increment=1)` when the program draws, else None. It replaces
    `stepped()`'s record, and is the consumer's typed operand for the once-per-execution advance after the draws.
- **Producer obligation (lower_inductor):** bind each `mesh::generator` node to the body's `Generator.state` root.
- **Consumer obligation (lower_metal `ConstructedMachine.generated`, schedule `analyse`):** apply `Advance` to that
  same root after the draws.

## CPU LAPACK: typed routine operators (connected, unchanged this turn)

`outline` expands `_linalg_svd`, `_linalg_eigh`, `linalg_eig`/`linalg_eigvals`/`_linalg_eigvals` and
`linalg_lstsq`/`mesh.linalg_lstsq_fixed` on the CPU into ATen glue plus one typed operator: `mesh::lapack_gesdd`,
`_heevd`, `_dgeev`, `_zgeev`, `_gels`, `_gelsy`, `_gelsd`, `_gelss` (`operations.LAPACK[packet] = Routine(symbols,
arguments, error)`, `routine(target)`, `LAPACK_LIBRARY` Accelerate).

- Each has a fake only. The producer lowers it to one `Routine` call per batch member, never Torch's kernel.
- Workspace query, ILP64 and ABI are the native owner's existing implementation. `Routine.arguments` lists operands in
  clapack.h order; it is not a second workspace or ABI implementation.
- Errors go to `aten._linalg_check_errors(info, Routine.error, is_matrix)`.
- GELS/GELSY now have native selected-call codes 5/6, workspace queries and original result-width conversion in
  the executor's canonical LAPACK owner (9404b7b2/633068d4). Their Python selected-call projection is paired in
  lower_metal; this source record does not claim execution of those methods.
- The `mesh::linalg_lstsq_fixed` fake and its fixed-product autograd (`_lstsq_fixed_backward`,
  `_lstsq_fixed_context`) remain source interface responsibilities and are kept.

## Consumer and producer impact (not my files)

- **consumer476 (metal.py, lower_metal.py, driver.py):**
  - Any remaining read of `operations.lowered`, `method`, `METHODS`, `_placed`, `_copy`, `_Check`, `_Repeat`,
    `_Region`, `_Generic` or `Generator.stepped` now fails.
  - In the current working tree, lower_metal.py reads only `routine`, the `LAPACK_*` operators and
    `SCALAR`/`CPU`/`GPU`/`ANE`. driver.py imports `Generator` and `seeded`. All of these are kept.
  - The advance is `Generator.advanced()` → `Advance`.
- **frontend103 (schedule.py, archive.py, native.Target.composite):** schedule.py imports `SCALAR` and native.py
  imports `_metadata`; both are kept. Nothing they import is a record.
- **producer68 (lower_inductor):**
  - No `mesh::geev` FallbackKernel exists any more. eig/eigvals off the CPU arrive as ATen and `while_loop` subgraphs.
  - Their carries include complex, int64 and bool batch tensors, plus a 0-dim placeholder for Z without vectors.
- **export.py (not mine):** its line 31 comment still names `operations.QDWH_EIGH_STEP`, which was already gone at
  HEAD.

## Coverage this source does not establish

- **Execution:** no body has been traced or executed. HOP carry metadata (contiguous carries, no aliasing returns) is
  arranged statically, not observed.
- **Complex ZHSEQR:** the source is now written (see the group above), but none of it has been traced or executed.
  Still OPEN within it are the blocked routines that its own calls reach:
  - The blocked ZUNMQR (zunmqr.f with zlarft.f/zlarfb.f) for xLAQR3's ZUNMHR. It needs ILAENV's NB = 32 < K =
    NS − 1, so NS ≥ 34, which needs (N − 1)/3 ≥ 34, i.e. N ≥ 103. Its NB also depends on the exact LWORK, against
    v3.12.1's LWKOPT = NW·NB + TSIZE (not master's): ZUNMHR gets LWORK − JW.
    - ATen passes lwork = ceil(nextafter(WORK(1))), i.e. zgeev's MAXWRK + 1.
    - MAXWRK takes the maximum over the ZGEHRD/ZUNGHR terms, N + ZTREVC3's query with vectors, and ZHSEQR's query.
    - This query chain is not yet transcribed; the earlier entries use only its lower bound 33N.
  - The blocked ZGEHRD (DO 40 with zlahr2.f/zlarfb.f) for xLAQR3's ZGEHRD(JW, 1, NS), reached where NS ≥ 130, and for
    GEEV's own ZGEHRD where NH ≥ 130.
  - Their sources are fetched, read-only, and not yet transcribed. Until they are, those members run the unblocked
    ZUNM2R/ZGEHD2.
- **Real DHSEQR's DLAQR0 (OPEN):** DLAQR0 selection (N > NMIN = 75) and the retry (INFO > 0 from DLAHQR; N ≥ NL = 49
  on H, else HL). Every real N reaches DLAHQR alone. That is the source's path only for N ≤ 75 with INFO = 0.
  Unwritten: dlaqr0.f/dlaqr4.f (real shift pairs), dlaqr3.f/dlaqr2.f (2 × 2 blocks with DLANV2), dtrexc.f with
  dlaexc.f/dlasy2.f/dlarfx.f, and dormhr.f's blocked DORMQR. `_laqr5`'s real branch (dlaqr5.f) and `_laqr1` are
  written but not yet called.
- **GEEV's INFO = −4 (OPEN):** `_geev_routine` returns INFO = −4 where ANRM is NaN or above overflow. That return is
  master zgeev.f's (the repo's zgeev.reference.f). v3.12.1 zgeev.f has none: it scales and proceeds. dgeev.f has the
  same split.
- **Line citations (OPEN):** the comments written from the master batches are 1-29 lines off v3.12.1. These cover
  zlahqr/dlahqr, dlanv2, zgehrd/zgehd2, ztrexc, zunm2r, zunmhr, zunmqr, iparmq, ilaenv, and zlaqr0/3/4 with
  zhseqr. Each needs remapping against /tmp/n67i and /tmp/n67j.
- **`_unghr` exactness (OPEN):** ZUNGHR still applies its reflectors with the generic `_dlarf`, not with zung2r.f's own
  reflector code. GEEV's ZGEHD2 now uses `_larfg_vector` and `_larf1f`.
- **Rounding differences from the reference:**
  - The blocked ZGEHRD (NH ≥ 130, and xLAQR3's NS ≥ 130), the blocked ZUNMQR (xLAQR3's NS ≥ 34) and the blocked
    ZUNGHR/ZUNGQR are the unblocked forms here.
  - DLADIV/ZLADIV are transcribed (`_divided`, `_dladiv1`, `_dladiv2`, `_cdiv`; checkpoint 43aa1f2e).
  - The scaled largest-modulus norm stands in for dznrm2.f90/dnrm2.f90.
  - Backend summation order and FMA differ, as does the TREVC3 GEMM back-transform order.
- **DLARTG:** transcribed from dlartg.f90 (3.10+) without a local copy. Accelerate's LAPACK (3.9.1-era) uses the older
  dlartg.f, whose sign rule differs when |F| ≤ |G| and F < 0, so the CPU and non-CPU routes can differ there by the
  pair's sign.
- **Non-finite input message:** INFO = -4 reaches `_linalg_check_errors`, as on Mesh's CPU route. PyTorch eager's
  linalg_eig instead checks `isfinite` first and gives a different message.
- **Upstream lowering coverage:** Inductor coverage of complex while_loop carries, int64 index arithmetic, and float16
  (complex32 results) and bfloat16 inputs is the upstream's to establish. These dtypes run the same body in their own
  precision.
- **GPU/ANE placement:** compiled GPU/ANE coverage of the bodies' ATen ops (`linalg_qr`, `solve_triangular`, `sort`,
  `gather`, complex `matmul`) is unestablished. An installed MPS kernel is not evidence for it.
- **Graph size:** O(n) static positions per while body. ZTREVC/ZLATRS takes O(n) rounds of O(n²) work, and the CNORM
  scan's case-2 recomputation is O(n²) per KI.
- **lower_coreml.py:** the operator's intentionally disabled original, kept as didactic reference. It is not a target
  owner and not a defect.
