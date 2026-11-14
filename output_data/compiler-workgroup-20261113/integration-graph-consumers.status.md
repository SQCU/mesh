# Graph consumers: status

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

2026-11-13, integration worker (Claude Opus 5.5, `claude-opus-5-5`, session
`59368744-da56-491a-871a-1be809c0cc93`).

**READY.** The original-geometry owner correction (section 4) is complete, with its narrow syntax check.
`metal_ir.h` and `metal_ir.mm` are now held for the compiler supervisor's canonical library and executable build,
until root releases the window. The native `Depends` checkpoint is committed (metal-microbench `09a604f1`). The
memory-result corrections in sections 3 and 4 are uncommitted. Evidence is syntax only: nothing has been built,
linked or run by me.

## 1. Native consumer: `metal_native.mm` `Depends`/`Recognize`

- **Removed:**
  - The recursive `Depends(const ir::Value *, std::vector<bool> &)`, which walked operand edges with no visited
    state.
  - The `vector<bool>` adapter delivered first and rejected in review.
- **Producer (consumer-side extraction).** `OperandGraph(ir::Function *)` builds the region's use-def graph.
  - Each parameter is vertex = `FunctionParam::Index()`, so parameter k is vertex k.
  - Each instruction result of the function is a vertex. Tint's `ir::Traverse` enumerates them, including nested
    blocks.
  - Each result's row lists its instruction's operand vertices. A constant, a block parameter or a module-scope
    value has no vertex, as in the old recursion, which stopped there.
  - Rows are CSR (`first`, `head`), read through `mesh_graph_row_head`.
- **Search (mesh-graph owner).** `Depends(graph, v, Search &)` runs one `mesh_graph_search` (depth-first, root =
  v's vertex) and returns its status. The result stays in a `std::unique_ptr` with `mesh_graph_search_free`.
- **Consumer.** `Recognize` builds the graph once for the analyzed region, before its two side queries. Each side
  owns its search result for that iteration and queries `place[k] != UINT32_MAX`, both for the weight parameter
  and for the n-symbol parameter, which are the old two `used[...]` tests.
- **Failure.** A failed search's status goes to `Program::error` ("the region parameter search failed: ...") and
  `Recognize` returns false. No dependence is fabricated.
- **Preserved.** The intrinsic choices are the same tests on the same reachability. The edges are exactly the old
  recursion's edges (result → its instruction's operands), and the leaves are the same.

## 2. Memory results: `Program` access relations on `mesh_graph_relation`

- **Before.** `viewConflicts` kept only kind + 1, and `Decided` freed the solver's records.
- **Now:**
  - Each `Decided` batch (from-accesses sharing one undecided partner set) moves its `mesh_graph_view` array and
    the solver's original `mesh_graph_conflict` array into a `Decision` owner.
  - `mesh_graph_relation_create(2, views, ..., conflicts, conflict, Undecided, owner, &relation)` adopts that
    owner, and `Undecided` frees both arrays with the last reference.
  - `Program::viewRelations` holds `{std::shared_ptr<mesh_graph_relation>, pool}`, where `pool` is each view's pool
    identity (the pool/view correspondence).
  - `viewConflicts[Ordered(x, y)] = ViewWitness{relation, record}` indexes the relation's original record, read
    through `mesh_graph_relation_data(...)->conflict`. A decided pair with no hazard has `record = SIZE_MAX` (section 3).
- **Projection.** `int OperationConflicts(p, a, b, std::vector<EffectConflict> &)` appends
  `{mine, theirs, relation, const mesh_graph_conflict *record}`. The record is borrowed from the retained relation,
  not rebuilt.
- **Fusion consumers.** `Conflict(p, a, b, mine, theirs)` keeps the same answers.
- **`viewPool`** is the shared geometry owner `std::shared_ptr<ViewPool>` (section 4).
- **`metal_ir_specialize`** copies `viewRelations`; the `shared_ptr`s share the relations. The source program, whose
  pool the relations borrow, is retained by the copy.
- **Unchanged:**
  - Selective invalidation: `Changed` is not edited. Relations and witnesses are immutable access facts.
  - Exact undecided-pair batching: the same grouping.
  - No decided pair is solved again.

## Exact source edits (mine; stage only these)

- `metal_native.mm`: committed in `09a604f1`. Root's review hook flags the `Operands` and `Depends` comments as
  more than citations; that file was left unchanged during its staging.
- `metal_ir.h`:
  - `ViewRelation`, `ViewWitness` (size_t record) and `EffectConflict` before `Program`.
  - `ViewPool` after `ViewRelation`; the `Program::viewPool` comment and its type `std::shared_ptr<ViewPool>` (the
    `viewIds` field is gone, moved into `ViewPool::ids`).
  - `viewRelations` and the `viewConflicts` type (`@@ -773 +797,2`).
  - The `OperationConflicts` declaration, then the `Witness` and public `Pooled` declarations after it.
  - Not mine: `operationEdges`, and `mesh_graph_relation *relation` (`@@ -762,0 +785,2`).
- `metal_ir.mm`:
  - `Access` and the public `Pooled(const Program &, const mesh_graph_view &)`, before `Accesses`; in `Accesses`,
    `Pooled(p, Record(v))`.
  - `Decision` (with `geometry`)/`Undecided`, `Decided`, `Hazards`, `Witness`, `OperationConflicts` and `Conflict`
    (the hunks between `bool Conflict(..vector..)` and `Renumber`).
  - The `metal_ir_specialize` copy lines `p->viewPool = s->viewPool; p->operationViewIds = s->operationViewIds;` and
    `p->viewRelations = s->viewRelations; p->viewConflicts = s->viewConflicts;`.
  - Every other hunk in `metal_ir.mm` belongs to other owners.

## Checks

After the corrections in section 3, `metal_ir.mm` exits 0 with no diagnostics under the same flags. As root asked,
no broader sweep was run for the index-type change.

The `.build/libmetal_ir.dylib` rule's flags with `-fsyntax-only` (logs in `/tmp/integration-evidence/`):
`metal_ir.mm`, `metal_native.mm`, `metal_transform.mm`, `metal_lower.mm`, `metal_clone.mm`, `metal_archive.mm`,
`metal_transport.mm` and `metal_neural.mm` each exit 0 with no diagnostics. Root reports that the canonical libmesh
rebuild passes with the search and relation symbols. I have not linked against it.

## 3. Corrections after review (uncommitted)

- **`ViewWitness`** is now `{uint32_t relation = UINT32_MAX; size_t record = SIZE_MAX;}`. The record index keeps
  the canonical `size_t` with a `SIZE_MAX` sentinel, so `ViewWitness{at, e}` no longer truncates. Pool identities
  and relation indexes stay `uint32_t`.
- **Lifetime.** The copy of each batch's accesses made here was rejected in review; section 4 replaces it.
- **Key.** `Access(key, offset, element, write, pairs, count)` builds the pool key, the exact access operands. Both
  `Pooled` and `Witness` use it, so they cannot diverge.

## 4. Original geometry under one shared owner (uncommitted)

- **Owner.** `struct ViewPool { std::deque<View> views; std::map<std::vector<uint64_t>, uint32_t> ids; }` in
  `metal_ir.h`. It holds each distinct access's original geometry once, its identity its index.
  - It is only appended to, and an entry never changes or moves (a deque), so its `pairs` stay where retained
    relations borrow them.
  - `Program::viewPool` is `std::shared_ptr<ViewPool>`. The pool holds no cache and no relation, so there is no
    owner cycle with `viewRelations` or `viewConflicts`.
- **Batches.** `Decision` (a relation's owner) holds `std::shared_ptr<const ViewPool> geometry`, its
  `mesh_graph_view` array (`Record(geometry->views[pool[v]])`, borrowing the pool's pairs), and the solver's original
  records.
  - The view/pool correspondence is `ViewRelation::pool`.
  - No access payload is copied for a batch. A relation retained past its program keeps the pool alive through
    `geometry`.
- **Specialization.** `metal_ir_specialize` shares the pool (`p->viewPool = s->viewPool`), with no deep copy of the
  pool or of any batch's subset.
  - A specialization's new accesses join the same pool with the same identity rule.
  - Fact: the pool is shared mutable state across a program and its specializations, so pooling into two of them
    concurrently from different threads is unsynchronized.
- **The one pool entry.** `uint32_t Pooled(const Program &, const mesh_graph_view &)` (public, `metal_ir.h`)
  returns an exact access's identity, adding it once where new. That is the single copy of its geometry, owned by
  the pool. `Accesses` pools each effect through it.
- **Check.** `metal_ir.mm` and `metal_transform.mm` exit 0 with no diagnostics under the canonical flags with
  `-fsyntax-only`. Only `metal_ir.mm` names the pool.

## Source-witness import: the API to connect to this pool (for the compiler's contract)

The contract has not arrived yet. These are the existing entry points it should use, so that it shares this pool
and adds no second copy of the views:

- `Pooled(p, view)` names an imported access by the same identity the fusion queries and `Witness` use. The
  identity key is `{key, offset, element, write, pairs}`, and `vertex` is not part of it.
- `p.viewPool->views[id]` is that access's original geometry, borrowed while any holder retains the pool. An
  importer that keeps it longer holds the `shared_ptr`.
- `Witness(p, from, to)` and `viewRelations[...]` serve the decided memory witnesses, as in the next section.
- Not yet provided: inserting decided facts produced outside metal_ir (for example, a frontend memory witness
  imported into `viewConflicts`). The contract would have to state which relation owns those records. I have not
  added an import path ahead of it.

## Review of the `ImportConflicts` seed contract (to implement when the window is released)

The contract is
`ImportConflicts(Program &, const std::shared_ptr<mesh_graph_relation> &original, const std::vector<uint32_t> &pool)`,
where `pool[a]` is the Program identity (`Pooled`) of `original` view a, or `UINT32_MAX` for an access with no
Program counterpart.

It fits the present owner:

- **Retain without copying.** Append `{original, pool}` to `viewRelations`. `ViewRelation` already holds a
  `std::shared_ptr<mesh_graph_relation>` and the view/pool correspondence, so the original relation is retained,
  not copied.
- **Positive witnesses.** For each original record r with both `pool[r.from_view]` and `pool[r.to_view]`
  represented, `viewConflicts[Ordered(pool[from], pool[to])] = ViewWitness{relation, r}`. That is r's own identity,
  with no RAW/WAR/WAW re-derived.
- **Coverage (no-hazard facts).** Only for the pairs the original query actually decided, which depends on its mode
  (`dependences_views` or `conflicts_views`) and on its vertex assignment. My batches decide exactly their
  from × partner cross pairs. Recording an imported pair as "decided, no hazard" needs the graph owner's exact
  coverage of `original`, and the numerical supervisor is coordinating that addition. Until it exists, an import
  registers positive witnesses only, and other pairs stay undecided, to be solved once by `Decided` if they are
  ever asked.
- **Orientation.** `viewConflicts` is per ordered pair. A query in the opposite order of an imported positive
  witness needs the graph owner's directional cause projection, which is also pending. It must not be re-derived
  here, nor re-solved for a known geometry pair.
- **Final dispatch relation.** Its records' accesses go through the same `Pooled`, so known pairs resolve to these
  witnesses; only unknown geometry reaches the solver.
- **Unchanged.** `Changed` invalidation (operation-to-view association only) and the `Decided` batching.

## Exact memory-only access for the finalized `Program` relation (compiler supervisor)

`const ViewWitness *Witness(const Program &p, const mesh_graph_view &from, const mesh_graph_view &to)`
(`metal_ir.h`, after `OperationConflicts`):

- **Lookup.** The exact access key of each view is `{key, view.offset, view.element, write, view.pairs[0 .. 2 *
  view.dims)}`. `vertex` is not part of it.
  - It returns `nullptr` where either access was never pooled, or the ordered pair `(from, to)` is undecided.
  - Otherwise it returns the witness. `record == SIZE_MAX` means decided with no hazard: disjoint, both reads,
    both commutative, or an empty view, all by the canonical relation.
- **The record.** `const struct mesh_graph_relation_data *d =
  mesh_graph_relation_data(p.viewRelations[w->relation].relation.get());` gives `d->conflict[w->record]`, the
  canonical solver's original record.
  - Its `from_view` and `to_view` index `d->view`.
  - `p.viewRelations[w->relation].pool[...]` gives their pool identities.
  - Its `from` is vertex 0, the `from` access. Its `kind` and `key` are the solver's.
- **Orientation.** Facts are per ordered pair, because the kind depends on order. A finalized record oriented
  `(to, from)` asks `Witness(p, to, from)`, which may be undecided even when `(from, to)` is decided.
- **Coverage.** Facts are decided lazily, for the pairs that fusion queries asked. The finalized relation reuses a
  witness only where its view pair has the identical exact accesses (same flags) in the same order. It solves the
  rest itself, without re-solving these, and without creating a second Program-wide relation.
- **Lifetime.** A record is borrowed while the `Program` holds the relation (`viewRelations`). A consumer that
  keeps it longer copies the `std::shared_ptr` (`p.viewRelations[w->relation].relation`) or calls
  `mesh_graph_relation_retain`, then later `mesh_graph_relation_free`. Its records and views are its own, and its views' pairs are the shared pool's, which its owner retains (section 4).

## Unresolved interfaces

- The numerical worker's operation relation (`Program::relation`, with source edges, composition and fusion
  legality) is separate from these memory-only access relations. If its legality needs memory conflicts between
  operations, `EffectConflict` gives each pair's original record, its relation, and the two effect indexes. The
  operation endpoints are the caller's a and b. Whether it wants these, or a quotient of the access relations onto
  operations, is for root to settle with that worker.
- No build or client run has been done. After the hold, compile an existing E2B phase program through the canonical
  path and compare the lowering result and compile time against `9a548ba3`.
