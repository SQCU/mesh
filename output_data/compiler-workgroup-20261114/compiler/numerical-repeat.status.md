# QDWH-eig subspace iteration: the current source contract (numerical, SAME64, numerical-18; uncommitted, HOLD)

Source: pub13 §5.1, Algorithm 3, and its finite-precision remedies (`/tmp/mesh-pub13.txt`, lines 721–780); §5.2 and
§5.4 for the split. Lines are `compiler/mesh_compiler/operations.py` unless another file is named. This file replaces
every earlier proposal; nothing here describes unimplemented design except §6.

## 1. Record

`_Repeat(records, condition, carried)` (2349) is a source numerical while loop.
- **`records`:** one pass, in order. These are structured records, and may include a nested `_Repeat`.
- **`condition`:** a one-element int32 TEMPORARY, written before the record and by each pass. A pass runs while it is
  nonzero, tested before every pass including the first.
- **`carried`:** the TEMPORARY tensors that cross passes or are read after the loop.
- **Placement:** `place` and `node`. `_placed` (5046) recurses into the children.
- **No bound:** there is no iteration-bound field.

The encoding is 047's (mesh `6a444b84`):
- `first = metal_ir_operations`, then the children are encoded, then `end`;
- then `metal_ir_repeat(p, first, end, value(condition), len(carried), [value(t) for t in carried])`.

## 2. Where it sits

It lives inside one `QDWH_EIGH_STEP` lowering, for each batch member (`_qdwh_eigh_step` 2814 → `_qdwh_member` 2587):
- one agenda pop, in the same invocation;
- between the partition and projector records and `turned` (2785).

2n−1 Step calls bound only the positive-child agenda pops; no numerical pass is a Step call.

## 3. Control as emitted (2724–2784)

- **Tolerance:** `√error > 10u·√squares`, with u = `finfo.eps / 2` (2733). `squares` is ‖H‖²_F.
- **`procedure(made)`:** this one helper is emitted at two source occurrences.
  1. A pass: `_subspace(x, q, rotated, h, e, rank)` (2567), which does the QR into `q` (V1 | V2) and
     `rotated` = QᵀHQ, then `error = ‖rotated[rank:b, :rank]‖²_F` (loops `(rest, rank)`, row offset `rank`).
  2. `again := error above tolerance`, and `previous := error`, the actual initial error.
  3. Inner `_Repeat(repeated, again, (x, q, rotated, error, previous, again))`. Its pass is `x := C·q[:, :rank]`
     (loops `(e, rank, e)`), then a pass, then `again := above tolerance ∧ error < previous`, then
     `previous := error`.
  4. `unaccepted := error above tolerance`.
- **First occurrence:** `procedure(ops)` (2758) runs once, on X0 (the k largest-norm rows of C). On the accepted
  path, nothing else runs.
- **Outer `_Repeat(remedied, unaccepted, carried)`** (2783) is entered only while unaccepted. Carried: `x, q, rotated,
  error, previous, again, unaccepted, u, projector, minus, plus, rank, rest, at_minus, at_plus, offset_plus, swap,
  stream`.

  Its pass:
  1. **(ii)** `_bounded_polar(u, refined, e)` (2500). This is QDWH's owner and `max_iterations`; its convergence
     result is returned and not used as a classifier. Then `u := refined`.
  2. ‖U‖²_F and trace U, then the partition state rebuilt from them by the same §5.4/§5.2 side rule, used directly.
  3. C rebuilt.
  4. **(i)** `stream += 1`, then a standard normal G of `b × rank` elements, then the bounded QR of G, then Q's first
     `rank` columns, then each column signed by `sign(R_kk)` (Haar), then `x := C·W1` (loops `(e, rank, e)`).
  5. `procedure(remedied)`, the second occurrence.
- **Accepted output:** `turned` (V0·q), the H_minus/H_plus and V child writes, and the agenda push (2785–2812) run only
  after the outer loop terminates, which requires `unaccepted = 0`.

## 4. Randomness

- **`_draws` (4778):** its counter is (element, generator execution, site, stream). The site and stream may be
  one-element int32 tensors; existing callers pass a constant site and stream 0.
- **`_normal` (4795):** Box–Muller, shared with `_random`.
- **Haar draws:** element = `member·n² + i·n + j`, site = the eigh's site tensor, stream = the restart count
  (numerical state, never protocol t).
- **Generator:** `_lowering` passes the construction's `Generator` into the Step, which calls `generator.site(site)`
  once per member. No independent seed is sampled.

## 5. Site and Step interface

- **`export.capture`** (`export.py:315`) numbers `DRAWING = (torch.ops.aten._linalg_eigh.default,)`
  (`export.py:24`) in the existing seeded-site namespace, before placement. This is the frontend grant; other seeded
  targets are unchanged.
- **`decompose._outlined`** (`decompose.py:326`) passes `eigh.meta["site"]` to `_qdwh_eigh_state(…, site=)` (2305).
  The prefix's sixth state output is a one-element int32 tensor.
- **`STEP_INPUTS`** (`decompose.py:164`) = `blocks, vectors, agenda, top, norm, site`. Every Step call takes the
  prefix's `norm` and `site`.
- **The step operator** (2332–2341): `mesh::qdwh_eigh_step(blocks, vectors, agenda, top, norm, site,
  max_iterations=10) -> (blocks, vectors, agenda, top)`, with a fake-only shape rule.
- **Sharing:** one shape-specialized Step serves every eigh source; the site arrives as data, not in the key.
- **Split query** (`decompose._outlined`): this is the first outline query over the source graph.
  - It builds one `mesh_graph.Graph.adjacency` over the placed FX graph's nodes (its output node apart). The edges are
    each node's direct non-output users, plus a SEND's retained arrival (`placed.meta["transfers"]`).
  - It runs one `Graph.reach` from the eigh's getitem users (non-output).
  - The `Reached` membership decides the early/late slices directly, together with the getitems.
- **The `outline` record:** `(source placed graph, eigh, relation, Reached)` is set on the prefix's and suffix's
  `GraphModule.meta["outline"]`.
  - `_staged` carries it onto the device graph without the deep copy `devices.on_devices` makes.
  - The shared Step has no such record.
- **Unchanged by the query:** `input_parts` triples and `work_axes` references.

## 6. Remaining limitations

- **Encoder and leaves (047, mesh `6a444b84`):** source-ready. `lower_metal` `associate(node, leaves, made)` already
  consumes the actual nested Emitted leaves, `_Repeat` children included.
- **Lifetime projection (placement17):** the direct `p.order`-span/3V lifetime projection is source-ready and strict
  checked.
- **Native (e48):** `metal_ir_repeat` control, emission, Actors and publication are still in progress. Nothing has been
  built, imported or run, so there is no runtime and no accuracy observation.
- **Unverified source premise:** the remedy's rebuilt partition is used directly. If the refined U gives neither side
  positive dimension, the rule yields a non-positive child, and §5.2's premise is violated at that step. There is no
  fallback or guard. If it occurs, the exact operand and step are to be reported after the accuracy observation.
- **No accuracy claim:** that waits for the complete native consumer and the original client's comparison with
  upstream eigh.
- **Checks:** AST and pyflakes are clean on `operations.py` (only the pre-existing unused `real` at 1062),
  `decompose.py` and `export.py`.
