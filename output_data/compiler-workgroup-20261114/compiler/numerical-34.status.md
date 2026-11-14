# numerical-34: connected non-CPU complex eig/eigvals through ZGEEV (SAME64), source coherent, HOLD

**Target:** `mesh::zgeev(Tensor a, bool vectors) -> (Tensor, Tensor)` in `operations.py`, a fake-only `ZGEEV`.
- **Shape rule:** values `(*batch, n)` contiguous; vectors `(*batch, n, n)`, `new_empty(...).mT`, which is column-major
  like the non-CUDA meta `linalg_eig`; or `(0,)` for values only.
- **Hook:** `_lowering` routes `local.target is ZGEEV` to `_zgeev(outs, a, vectors)`.
- **Formula:** `_eigen_geev(self, vectors)` returns (values, vectors), or values alone.

**Caller:** `decompose.outline`'s pre-scan now includes `EIG` (every overload of `linalg_eig`, `linalg_eigvals` and
`_linalg_eigvals`). The existing `_expanded`, through its source map, expands only nodes on a non-CPU processor with
complex input, before the LSTSQ/PINV/SVD expansions and the re-stage. CPU GEEV and real inputs keep `_eigen`.

**Lowering, per batch member** (n = 0 or an empty batch emits nothing):
- **Scaling (zgeev 332–350):** ANRM is ZLANGE 'M', the NaN-propagating largest modulus. INFO = −4 on NaN or
  ANRM > max, and nothing else runs. SMLNUM = √SAFMIN/EPS and BIGNUM = 1/SMLNUM; A ·= CSCALE/ANRM (ZLASCL's first pass).
- **Once-gate:** a `_Repeat` on INFO = 0 runs, in order:
  - `_zgebal`, then `_zgehd2` (Q when vectors);
  - ZHSEQR's W(1:ILO−1) and W(IHI+1:N) from H's diagonal;
  - `_zlahqr` with this member's INFO;
  - with vectors, a nested once-gate on INFO = 0: `_ztrevc`, `_zgebak`, `_zgeev_normalized`, then a copy into the
    vectors output;
  - W ·= ANRM/CSCALE over W(INFO+1:N), and also W(1:ILO−1) when INFO > 0, where SCALEA;
  - the copy of W into the values output.
- **Check:** then the existing `_Check(info, 3, 'linalg.eig')`.

**New routines:**
- **`_zgebak` (zgebak 205–261):** rows ILO..IHI are scaled when ILO ≠ IHI. The II = 1..N interchanges come from
  `perm` (I = ILO−II below ILO), each a two-row ZSWAP in source order.
- **`_zgeev_normalized` (zgeev 492–503):** 1/DZNRM2 by `_linalg_lengths`, then TMP = conj(v_K)/√RW_K at the first
  largest RW (IDAMAX), and Im v_K set to 0.
- **Helper:** `_divided_by`.

**Unverified:** grep confirms each new name is defined once. I read it through for syntax. Nothing was run, imported
or built, and I did not run AST.

**Limitations:**
- **Shortcuts:** ZLAHQR is used for every n (no ZHSEQR ZLAQR0 multishift/AED and no rescue on failure), and the
  unblocked ZGEHD2/ZTREVC are used (no ZGEHRD/ZTREVC3 blocking).
- **Other differences:** reduction-order sums, Smith division for ZLADIV, and DZNRM2 as the max-scaled sum.
- **Not verified or not covered:** no numerical output yet; nothing on real input or left eigenvectors.

Supervisor: AST, no duplicate definitions, diff whitespace and actual 3×3 complex64 FakeTensor construction pass. Values stride (1,), vectors stride (1,3); four outer records retain the actual nested numerical repetitions. Exact counts in nonsymmetric-source-observation.json. No native numerical result at this source checkpoint.
