> Supervisor review: the old-record method result below is transitional and not accepted as the final boundary. SAME211 numerical67b is replacing retained `_Region/_Generic` compiler machinery with actual upstream method bodies while preserving cited algorithms. Producer/consumer must not treat FallbackKernel registration plus old generic records as completed replacement.

# numerical67(b): custom-method bodies and operands (operations.py, decompose.py)

Session 211a403e, turn numerical67b (consolidated brief), on top of supervisor checkpoint 8cbc2ab9. The supervisor
has since preserved three WIP checkpoints, none of which accepts the method:
- 43aa1f2e: DLADIV;
- c4bf33fe: the LAQR1/LAQR5/LARFG group;
- e908087a: the LAQR5 stage order and the LAHQR argument lists.

The operations.py delta on top of e908087a (the xLAQR3 dependency group below) is uncommitted. No native import, build
or workload has been run.

Checks were static only:
- `py_compile`;
- undefined-name and unused-import checks;
- duplicate top-level definitions;
- call arity against definitions;
- the reachability cut from every name other modules import.

No body has been traced or executed. The read-only review delegates were stopped. The supervisor's own source review
covered DLADIV (43aa1f2e) and the LAQR5 stage order (the c4bf33fe review, applied below). The source compared against
is Reference-LAPACK v3.12.1 SRC, held read-only and never executed.

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
  - W outside ILO..IHI comes from the diagonal. The ILO = IHI return gives W(ILO) = H(ILO,ILO).
  - Otherwise ZLAHQR is called with ZHSEQR's own arguments: `(_zlahqr | _dlahqr)(WANTT = WANTZ = vectors, N, ILO,
    IHI, H, W, ILOZ = ILO, IHIZ = IHI, Z, proceed)`.
  - Then H's r − c ≥ 2 part is zeroed when WANTT or INFO ≠ 0, with N > 2.
  - ZHSEQR's xLAQR0 selection and retry are OPEN (see coverage).
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

### xLAQR0 sweep dependencies (source present, not yet called)

These go between `_dgeev_normalized` and `_geev_routine`. Each is cited to zlaqr1.f/zlaqr5.f and dlaqr1.f/dlaqr5.f
v3.12.1. They are written for the xLAQR0 path, which is still unwired, so no method reaches them yet.

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
  - After DO 145, the far-from-diagonal updates go through U, chunked into WH/WV as the source chunks them. The source
    rows of the residue regions are rewritten to their last chunk.
  - `_vigilant` is the source's vigilant deflation check.

### xLAQR3 dependency group (source present; `_larf1f` live through `_gehrd`, the rest called once `_laqr3` lands)

- **`_larf1f(left, m, n, v, tau, x, r0, c0, when)`:** zlarf1f.f 196-308, and dlarf1f.f in the same body:
  - LASTV, scanning down past V(1) while V(LASTV) = 0, where TAU ≠ 0;
  - LASTC = ILAZLC(LASTV, N, C) or ILAZLR(M, LASTV, C), where a NaN counts as nonzero; the LASTC = 0 return;
  - with LASTV = 1, the ZSCAL by ONE - TAU;
  - else the ZGEMV, the first-row (left) or first-column (right) correction (zlarf1f's loops or ZAXPYs, dlarf1f's
    DAXPYs), and ZGERC/DGER skipping a zero WORK(J) (left) or a zero v(J) (right).
- **`_zlartg`:** zlartg.f90 3.12.1, branch by branch.
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
- **ZHSEQR's xLAQR0 (OPEN):** this covers N > NMIN = 75, and the retry where xLAHQR returns INFO > 0: KBOT = INFO,
  on H itself when N ≥ NL = 49, else on the 49 × 49 HL copy. Every N still reaches xLAHQR alone. That is the source's
  path only for N ≤ 75 with INFO = 0. Unwritten, for both real and complex:
  - xLAQR0 itself: NTINY, IPARMQ NWR/NSR/KACC22, NWMAX/NSMAX, the DO 70 window/NDEC/NDFL logic, exceptional and
    computed shifts, the bubble sort, and the NS × NS shift copy.
    - The LWORK terms never bind: LWORK ≥ 32N from zgeev's MAXWRK, and LWORK = 49 on the HL path.
  - xLAQR3/xLAQR2 itself: the deflation loop, the sorting, the spike reflector and the H/T/V/WV updates. Their
    complex dependencies are written (above). The real DTREXC with DLAEXC/DLASY2/DLARFX is not written.
  - xLAQR4.
  - The blocked ZUNMQR (K > 32, so NS ≥ 34) and blocked ZGEHRD (NH ≥ 130), through zlarft/zlarfb/zlahr2. Their
    sources are fetched, read-only, and not yet transcribed.
  - The HSEQR selection and retry in `_geev_routine`.
- **`_unghr` exactness (OPEN):** ZUNGHR still applies its reflectors with the generic `_dlarf`, not with zung2r.f's own
  reflector code. GEEV's ZGEHD2 now uses `_larfg_vector` and `_larf1f`.
- **Rounding differences from the reference:**
  - The blocked ZGEHRD (NH ≥ 130) and blocked ZUNGHR/ZUNGQR are the unblocked forms here.
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
