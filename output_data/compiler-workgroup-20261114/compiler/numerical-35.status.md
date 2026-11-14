# numerical-35: complex scatter-only `_record` output correction (SAME64), HOLD

**Fix** in `operations._record` (uncommitted; no other source changed):
- **Before:** a complex output of picks None (a whole-tensor scatter destination) emitted two real-pair outputs, both
  with no ordinary map. The scatter wrote only output 0, so output 1 made an ordinary store of the yielded imaginary
  part at element [0,0].
- **Now:** such an output emits ONE output (its `view_as_real` pairs, rows None), still declared in
  `component_views`. Its yield is the value's real part, or the value itself, and is not stored.
- **Unchanged:**
  - mapped complex outputs (two component maps, constant component rows);
  - real outputs;
  - all inputs, including gathered complex inputs as one full real-pair slot.
- **Comment:** the docstring now states that a scatter's output ordinal is the emitted one: a mapped complex output
  emits two, any other one.

**Audit of every `_record` output with picks None.** Each one is a scatter destination and is logical output 0, so
emitted ordinal 0 still names it.

| Call | Destination | Scatter | Emitted outputs before → after |
|---|---|---|---|
| `_qdwh_member` push (2949) | out_agenda, real | scatter(0) | unchanged |
| `_zgebal` row and column decisions (3556, 3596) | perm, real | scatter(0) | unchanged |
| `_zlahqr` stepped (4029) | w, complex | scatter(0, [I−1, part]) | 14 → 13: the stray w duplicate removed, then lo, i, kdefl, its, go, cut, i1, i2, i1z, mspan, shift.re, shift.im |
| `_zlahqr` severed (4039) | h, complex | scatter(0, [L−1, L−2, part]) | 2 → 1 |
| `_zlahqr` settled (4193) | h, complex | scatter(0, [I−1, I−2, 0 or 1]) | 2 → 1 |

Every scatter coordinate already carries the component axis. No source formula, C, native or ABI code changed, and
nothing was rerun.

**Unverified:** I did not run AST (python3 needs approval here); I read the edit through by eye.

**Supervisor notes:** the bc2cb879 measurement flags `_zgeev_shapes`, decompose's `while pending:`, and comments as
more than citations. Unchanged.

Supervisor checks: AST/no duplicate definitions/diff whitespace PASS. Full 3×3 GEEV FakeTensor construction confirms every output with no ordinary map is named by its actual scatter ordinal; stepped has 13 outputs and each H scatter one. Exact records in nonsymmetric-scatter-source.json. No post-fix numerical result at this checkpoint.
