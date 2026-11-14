# Prefix and first-Step emitted operand review (eigh-emitted-review.md) — HOLD

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5. Read-only.
- No source edit, import, run or build.
- I read the emitted text as evidence against the typed binding records (`eigh-source-capture.json`
  `bindings_records`) and the source records (`compiler/mesh_compiler/operations.py` `_qdwh_member`, lines 2587–2622).

**Result: no store, pop or symbol mismatch in the reviewed path.**

## Prefix: row 0 kernel 0 (`eigh-source-row0-kernel0.txt`)

| Member | Emitted | Typed binding | Matches source |
|---|---|---|---|
| 1 | `id(1)[0] = 1` | slot 1, v8, top, @400 | top = 1 |
| 2–5 | threadgroup `sym8 = 4`, `sym9 = 0`, `sym10[c] = (c == 1)`, `sym11[r] = r` | — | — |
| 6 (after the barrier) | `id(2)[r*2+c] = select(sym9, sym8, (sym11[r] == 1) & sym10[c])` | slot 2, v6, agenda (5×2 int), @128 | `agenda[1,1] = 4`, every other element 0 |
| 9 | `id(3)[r,c] = float(r == c)` | slot 3, v4, vectors, @3584 | eye(4) |
| 12 | `id(5)[r,c] = (c − r ≤ 0) ? a[r,c] : a[c,r]` | slot 5, v2, blocks, @64; `a` from slot 4 (v0, operand) | lower-triangle symmetrization |
| 13–14 (after device barriers) | `buffer(6) = abs(h)`, then thread 0 `id(7)[0] = sqrt(Σ abs²)` | slot 6, v57, temp @0; slot 7, v10, norm, @6928 | Frobenius norm |

Member 0 stores `id(0)[0] = 1`: slot 0, v12, @6936. Step kernel 0 does not read it.

## First Step: row 1 kernel 0 (`eigh-source-row1-kernel0.txt`)

**Typed inputs:** slot 0 = v0 blocks @64; slot 2 = v2 vectors @3584; slot 4 = v4 agenda @128; slot 6 = v6 top @400;
slot 11 = v8 norm @6928. Each is exactly the prefix output's arena range.

| Member | Emitted | Matches source |
|---|---|---|
| 0–2 | copies: `id0 → id1` (blocks), `id2 → id3` (vectors), `id4 → id5` (agenda) | `_copy` ×3 |
| 3 (pop, thread 0) | `t = id(6)[0]`; `offset = agenda[t*2 + 0]` to `buffer(7)` (v22 @6440); `size = agenda[t*2 + 1]` to threadgroup; `below = t < 1 ? 0 : t − 1` to `buffer(8)` (v26 @6448) | `_scalars([offset, size, below], [top], …, [agenda])`, reading the input agenda. With the prefix's store: offset 0, size 4, below 0 |
| 4 (after device and threadgroup barriers) | `h[r,c] = blocks_in[(r + offset)*4 + c]` for r, c < `min(size, 4)`, to `buffer(9)` (v28 @512) | `_copy(h, blocks)` with `loops=(sb, sb)`, `offsets=[(blocks, {0: so})]`. The offset is on dim 0 only, as the source's row-offset block storage requires |
| 5 | `Σ h²` and `Σ_{r≠c} h²` over `min(size, 4)`² | the squares/off-diagonal `_record` |
| 6 | `frobenius = sqrt(total)`; `nearly = sqrt(off) ≤ 5·eps·frobenius`; `tiny = frobenius < eps·norm`; `stop = nearly ∨ tiny ∨ b ≤ 1`; `e = stop ? 0 : b`; `d = stop ? b : 0` | `terminal()`. MSL `select(a, b, c)` is `c ? b : a` |
| 7 | `out_blocks[(i + offset)*4 + 0] = h[i,i]`, i < `min(d, 4)` | the `first` column store with `offsets=[(first, {0: so})]` |
| 8 | `diagonal[i] = h[i,i]`, i < `min(e, 4)`, to `buffer(13)` (v39 @0) | the diagonal `_record` |

**Barriers:** each thread-0 scalar producer is followed by the needed `mem_device`/`mem_threadgroup` barrier before
its readers.

**Arena:** the temporaries' offsets are disjoint from the Step's inputs and outputs (out_blocks @3648, out_vectors
@3712, out_agenda @3840).

## Next boundary

These two kernels' stores, pop extraction, symbol loads and extents agree with the source records and the typed
bindings. This is a bounded emitted-expression and operand-correspondence finding, not an observation of those
kernels executing. It does not rule out a runtime binding, execution-order, or lifetime failure in this same slice.

The next observation must distinguish that runtime boundary from later numerical operations. Prefix and Step
intermediates cannot be read after execution unless their actual retained lifetimes still cover that read. A diagnostic
which exposes them as original call outputs changes root ownership/lifetimes; any change in the original numerical
error under that diagnostic is ownership evidence, not proof that the algorithm was correct.

No production correction is justified by this source evidence. HOLD.
