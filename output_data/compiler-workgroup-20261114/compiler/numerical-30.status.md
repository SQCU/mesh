# numerical-30: ZGEBAL source routine (SAME64), intermediate checkpoint, HOLD

**Added** to `compiler/mesh_compiler/operations.py` (uncommitted): `_zgebal(h, scale, perm, ilo, ihi, ops, held=())`,
following Reference-LAPACK zgebal.f with JOB = 'B', lines 225–418. It works in place on one complex square member.
- **Records used:** existing `_record`, `_scalars`, `_emit` (runtime extents and offsets through `component_views`),
  `_Repeat` and gathered extraction slots.
- **Row search (242–286):** a `_Repeat` while NOCONV ∧ ¬done over static positions I = n…1, with the DO count fixed at
  entry (L₀).
  - A clipped SUM counts row I's nonzeros over 1..L with J ≠ I.
  - A scalar record scatters SCALE(L) = I into `perm`, sets NOCONV, and does the L = 1 return (ILO = IHI = 1) and
    L −= 1.
  - Two ZSWAP records touch only columns I and L (rows 1..L) and rows I and L (columns K..N).
- **Column search (288–317):** the same over J = K₀…L, rows K..L, I ≠ J, with K += 1.
- **Scale init (321–325):** SCALE(K..L) = 1 through an offset record.
- **Scaling (344–413):** a `_Repeat` while NOCONV over static positions I, active when K ≤ I ≤ L.
  - C and R are DZNRM2, as max-scaled sums. CA and RA are |a| at IZAMAX's first maximal CABS1.
  - CYCLE when C = 0 or R = 0.
  - Each DO WHILE ×/÷ SCLFAC is its own `_Repeat` of one scalar pass. SFMIN1/SFMAX1/SFMIN2/SFMAX2 come from the real
    dtype's finfo.
  - The FACTOR = 0.95 and SCALE-bound tests decide; then ZDSCAL of row I (K..N) by G = 1/F and of column I (1..L) by F.
  - The 363–367 NaN exit (INFO = −3) ends the routine, leaving ILO/IHI unassigned.
- **Small helpers:** `_record_extreme` (a REGION max/min over a record value), `_cabs1`, `_modulus`, `_real_scaled`,
  `_nonzero`, `_offsets`, `_entry`, `_clipped_norm`, `_first_largest` and `_swapped`. I renamed my first `_scaled` after
  finding the existing `_scaled` at line 996.
- **One deviation:** DZNRM2 uses the `_linalg_lengths` max-scaled formula, not Blue's algorithm, so a value at an exact
  test threshold can round differently.

**Supervisor checks:** AST, duplicate top-level definition scan and diff whitespace check pass. The existing typed
producer constructs its records for one complex64 3×3 FakeTensor member (10 outer records, 3 outer repetitions).
This checks source construction only: no numerical evaluation, native build, or target invocation. The supervisor
reviewed the row/column exchange bounds and norm-scaling decisions against the retained zgebal reference.

**Not done (later SAME turns):** ZGEEV scaling, ZGEHD2/ZUNGHR, ZLAHQR, ZTREVC/ZLATRS, ZGEBAK, normalization, the fake
`mesh::zgeev` target, the `_lowering` hook and the decompose `EIG` caller. `_zgebal` has no caller yet, and public
non-CPU complex eig remains unfinished. The plan stays in `numerical-28.status.md`; the `_linalg_reflector` `le` change
is not yet made.
