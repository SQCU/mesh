# Graph consumers: status

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

2026-11-13, integration worker (Claude Opus 5.5, `claude-opus-5-5`, session
`59368744-da56-491a-871a-1be809c0cc93`).

Final checkpoint, 14 November 2026: metal `79d7e7a6` connects retained source witnesses and function analysis;
`32f4960a` finishes the one registration function and removes the unused copying `ImportConflicts` wrapper.
The matched native build and actual CPU/GPU plus outer-archive observations are recorded in
[function-binding-review.md](executor/function-binding-review.md). Exact source carrier review is in
[source-given-review.md](compiler/source-given-review.md).

The sections below preserve the worker's intermediate source-review observations. Their build holds,
uncommitted labels, draft interfaces and proposed checks are historical, superseded by the commits and
observations above; they are neither current assignments nor additional requirements.

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

Superseded where it describes `viewConflicts`: section 5 removes that map, and witnesses come from the retained
relations' own queries.

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
  - `ViewOriginal`, the `ViewWitness` (relation, `mesh_graph_witness`) and `EffectConflict` (with `witness`) types.
  - `Program::viewOriginals`, which replaces `viewConflicts`, with its comment.
  - The `OperationConflicts`, `Witness` (int, out), `ImportConflicts` and public `Pooled` declarations.
  - Not mine: `operationEdges`, and `mesh_graph_relation *relation` (`@@ -762,0 +785,2`).
- `metal_ir.mm`:
  - `Access` and the public `Pooled(const Program &, const mesh_graph_view &)`, before `Accesses`; in `Accesses`,
    `Pooled(p, Record(v))`.
  - `Decision` (geometry and views)/`Undecided`, `Register`, `Answer`, `Failed`, `Decided`, `Hazards`, `Witness`,
    `ImportConflicts`, `OperationConflicts` and `Conflict` (the hunks between `bool Conflict(..vector..)` and
    `Renumber`); `Ordered` is removed.
  - The `metal_ir_specialize` copy lines `p->viewPool = s->viewPool; p->operationViewIds = s->operationViewIds;` and
    `p->viewRelations = s->viewRelations; p->viewOriginals = s->viewOriginals;`.
  - Every other hunk in `metal_ir.mm` belongs to other owners.

## Checks

After the corrections in section 3, `metal_ir.mm` exits 0 with no diagnostics under the same flags. As root asked,
no broader sweep was run for the index-type change.

The `.build/libmetal_ir.dylib` rule's flags with `-fsyntax-only` (logs in `/tmp/integration-evidence/`):
`metal_ir.mm`, `metal_native.mm`, `metal_transform.mm`, `metal_lower.mm`, `metal_clone.mm`, `metal_archive.mm`,
`metal_transport.mm` and `metal_neural.mm` each exit 0 with no diagnostics. Root reports that the canonical libmesh
rebuild passes with the search and relation symbols. I have not linked against it.

## 3. Corrections after review (committed in `4db95f8d`)

- **`ViewWitness`** is now `{uint32_t relation = UINT32_MAX; size_t record = SIZE_MAX;}`. The record index keeps
  the canonical `size_t` with a `SIZE_MAX` sentinel, so `ViewWitness{at, e}` no longer truncates. Pool identities
  and relation indexes stay `uint32_t`.
- **Lifetime.** The copy of each batch's accesses made here was rejected in review; section 4 replaces it.
- **Key.** `Access(key, offset, element, write, pairs, count)` builds the pool key, the exact access operands. Both
  `Pooled` and `Witness` use it, so they cannot diverge.

## 4. Original geometry under one shared owner (committed in `4db95f8d`)

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

## 5. `ImportConflicts` and the original query-view consumer (uncommitted)

This is written against the published `mesh-graph.h` query API: `mesh_graph_relation_views`, `mesh_graph_witness`,
`mesh_graph_relation_witness` and `mesh_graph_relation_cause`. Their definitions are in the working-tree
`mesh-graph.c`.

- **The reduced cache is gone.** `Program::viewConflicts`, a map from an ordered pool pair to a seeded fact, is
  deleted along with its pair seeding. Nothing enumerates pairs to seed negatives.
- **Correspondence.** `Program::viewOriginals[x]` lists `ViewOriginal{relation, view}`: every original view of a
  retained relation whose pool identity is x. `Register(p, at)` fills it from `viewRelations[at].pool`, skipping
  `UINT32_MAX`. Original view indexes are unchanged.
- **The query-view consumer (`Answer`).** An ordered pair (x, y) is answered by any retained relation holding views of
  both, through `mesh_graph_relation_witness(relation, from.view, to.view, &w)`. No solver or ISL is involved. The
  first known witness is the answer.
  - Known hazards come back as the original record.
  - Known absences come back with `record = SIZE_MAX, known = 1`, and are kept.
  - A reversed pair comes back with `reverse = 1`. It reuses the symmetric fact, and its direction is
    `mesh_graph_relation_cause`'s.
  - Unknown is `known = 0`: no retained query decided the pair, for example two views on one vertex under DIRECTED.
- **Decided's producer.** Undecided pairs are only those whose `Answer` is unknown. Each group of from-accesses that
  shares a partner set makes one `mesh_graph_relation_views(2, views, ..., 0, nullptr, nullptr, self 0, Undecided,
  made, &relation)` call: the one DIRECTED query, whose original records the relation owns.
  - `Decision` now holds only the shared geometry and the views.
  - The relation is retained in `viewRelations`, and its views are registered.
  - Every pair the group decided, including absences, is then answered by that relation's own query.
- **Consumers.** `Hazards` visits each same-storage pair whose witness is a known hazard.
  - `Conflict(p, a, b, mine, theirs)` answers as before: hazard existence is symmetric, so a reversed witness
    answers it.
  - `OperationConflicts` appends `EffectConflict{mine, theirs, relation, witness, record}`, where `record` is the
    root's original record. Its direction in that orientation is `mesh_graph_relation_cause(relation, &witness)`. No
    Program code derives RAW/WAR/WAW.
- **`ImportConflicts(Program &, const std::shared_ptr<mesh_graph_relation> &original, const std::vector<uint32_t>
  &pool)`** appends `{original, pool}`, retaining the relation without copying it, and registers its views. From
  then on, its own query answers the pairs it decided: positive witnesses, absences, and reversed pairs.
  Undecided pairs, such as same-vertex pairs or views it does not hold, are queried once by `Decided` when asked.
- **`int Witness(const Program &, const mesh_graph_view &from, const mesh_graph_view &to, ViewWitness &out)`** (the
  finalized relation's lookup) finds both exact accesses in the pool, without inserting them, and answers through
  `Answer`. It returns `known = 0` where either access is not pooled or no retained query decided the pair.
- **Unchanged:**
  - The shared `ViewPool` geometry owner and `Pooled`.
  - `Changed`, which invalidates the operation-to-view association only, because the facts are immutable.
  - The batching of undecided pairs.
  - `metal_ir_specialize` now shares `viewRelations` and copies the `viewOriginals` index.
- **Checks.** `metal_ir.mm` and `metal_transform.mm` exit 0 with no diagnostics under the canonical flags with
  `-fsyntax-only`. No other source names the changed types.

### Interface for the finalized `Program` relation and the source importer

- **Import:** the numerical producer forms `pool` from exact source accesses and calls `ImportConflicts(p, original,
  pool)` with the source relation made by `mesh_graph_relation_views`. A `mesh_graph_relation_create` root has query
  NONE, so it answers no pair.
- **Final records:** each final view goes through `Pooled` (the same interning), and a pair is answered by
  `Witness`. A known witness carries `relation` (an index into `p.viewRelations`) and the graph-owned witness; the
  record is `mesh_graph_relation_data(...)->conflict[witness.record]` and the direction is
  `mesh_graph_relation_cause`. An unknown pair is new geometry, queried once by the canonical solver.
- **Lifetime:** records are borrowed while the program holds the relation. Keep the `shared_ptr` to hold one longer.
  The pool's geometry is held by every Decided relation's owner. An imported relation's owner is its producer's.

## 6. Known-answer memo (uncommitted)

- **Field.** `Program::viewAnswers`, an `std::unordered_map<uint64_t, ViewWitness>` keyed by the ordered pool pair
  `x << 32 | y`. It holds only answers already asked and known: `{relation, original mesh_graph_witness}`, which
  covers a hazard record or a known absence (`record = SIZE_MAX, known = 1`) and its `reverse` orientation, exactly as
  the retained relation's own query answered.
- **`Answer`** returns the memo entry for a pair asked before, without joining `viewOriginals[x] × viewOriginals[y]`
  again. Otherwise it asks the retained relations, and records the answer only when it is known.
  - An unknown answer is never held, so a later `Decided` query or `ImportConflicts` can still decide that pair.
  - No pair is seeded: entries exist only for pairs some consumer actually asked.
- **Specialization.** `metal_ir_specialize` copies `viewAnswers` with `viewRelations` and `viewOriginals`. Each
  entry's `relation` names the same shared relation in the copy.
- **Not touched:** `Changed` (known answers are immutable access facts, so nothing invalidates them), the graph C
  owner, the source setter, the final `Relation`, and the work rows.
- **Check.** `metal_ir.mm` exits 0 with no diagnostics under the canonical flags with `-fsyntax-only`.
- **Hunks:** in `metal_ir.h`, the `viewAnswers` field and its comment line; in `metal_ir.mm`, `Answer`'s memo lookup
  and `emplace`, and the specialize copy line.

## 7. Paired source setter (implemented, uncommitted; HOLD for the supervisor's build or commit window)

Published in metal-microbench `metal_ir.h`, working tree, uncommitted, exactly as root approved:

- **C** (beside `metal_ir_operation_edges`):
  - `enum { METAL_IR_SOURCE_VALUE, METAL_IR_SOURCE_LITERAL, METAL_IR_SOURCE_SENT, METAL_IR_SOURCE_RECEIVED,
    METAL_IR_SOURCE_EXTERNAL };`
  - `struct metal_ir_source_operand { uint32_t kind, value; };`
  - `int metal_ir_source_relation(struct metal_ir_program *, struct mesh_graph_relation *original, const size_t
    *operation_first, const uint32_t *operations, const uint32_t *view_values, size_t givens, const struct
    metal_ir_source_operand *given, void (*release)(void *), void *owner, char *error, size_t room);`
- **C++** (beside `ViewRelation`):
  - `struct SourceRelation { std::shared_ptr<mesh_graph_relation> original; const size_t *operationFirst; const
    uint32_t *operations; const uint32_t *viewValues; size_t givens; const metal_ir_source_operand *given; Owner
    metadataOwner; }`. It cannot be copied, and its destructor releases the adopted metadata owner once.
  - `struct SourceBinding { std::shared_ptr<const SourceRelation> source; uint32_t operationOffset = 0;
    std::shared_ptr<const std::vector<uint32_t>> values = nullptr; }`. A null `values` is the identity; otherwise it
    maps a source declaration ValueID to this program's ValueID.
  - `Program::sourceRelations`, a `std::vector<SourceBinding>`.
- **Reading a binding** (numerical consumer): original vertex v is
  `source->operations[source->operationFirst[v] .. source->operationFirst[v + 1])`, each plus `operationOffset`.
  Original view w is `values ? (*values)[source->viewValues[w]] : source->viewValues[w]`, where `UINT32_MAX` is
  external. GIVEN record g's operand is `source->given[g]`, whose `value` maps the same way as a view's.

- **Setter.** `metal_ir_source_relation` (beside `metal_ir_operation_edges`):
  - It reports EINVAL through `error`, adopting nothing and retaining nothing, only for an actual out-of-range memory
    operand: an operation ID ≥ the program's operations, a view or VALUE/SENT/RECEIVED given ValueID ≥ its values.
  - Otherwise it retains `original` (`mesh_graph_relation_retain`, held by a `shared_ptr` with
    `mesh_graph_relation_free`), borrows the arrays, and adopts `{owner, release}` as `metadataOwner`. The record's
    destructor releases it once.
  - It then appends `SourceBinding{record, 0, nullptr}` and calls `ImportConflicts(p, original, Geometry(p,
    binding))`.
  - The original's views, records and GIVEN operands are unchanged and authoritative. No source edge or operation
    product is made.
- **Geometry correspondence (`Geometry`, before `metal_ir_append`).** Original view w gets a pool identity only where
  the program accesses its exact geometry:
  - its typed value (`BoundValue`: through `values`, `UINT32_MAX` external), taken whole (`Whole`), is a single byte
    view with w's offset, element and pairs, and with w's write;
  - its key is that value's storage;
  - no other source storage key among the original's views maps to that storage.
  - Distinct source storages coalesced onto one typed storage, and changed coordinates, get `UINT32_MAX`. Their
    absences are therefore never imported as facts. Their correspondence stays in the binding, and their pairs are
    decided anew, once, where asked.
- **Append.** After the existing operation-edge remap, each source binding becomes
  `{b.source, b.operationOffset + operations, values = out ∘ b.values}`; the identity becomes `out`, and external
  stays `UINT32_MAX`. `ImportConflicts` then runs through the destination's own geometry. The `SourceRelation` and
  its original record IDs are shared, not copied.
- **Specialization.** `p->sourceRelations = s->sourceRelations`, sharing the owners, beside the view-relation copies.
- **Checks.**
  - `metal_ir.mm` and `metal_transform.mm` exit 0 under the canonical C++ flags with `-fsyntax-only`.
  - `tools/e2b_replay.m`, a C includer of the new C declarations, exits 0 under its rule with `-fsyntax-only`.
  - Nothing has been built, linked or run.
- **Hunks:**
  - `metal_ir.h`: the C enum, struct and setter declarations after `metal_ir_operation_edge_records`;
    `SourceRelation` and `SourceBinding` after `ViewRelation`; and `Program::sourceRelations` after `viewAnswers`.
  - `metal_ir.mm`: `BoundValue` and `Geometry` before `metal_ir_append`; the source-binding loop after its
    operation-edge remap; the specialize copy line; and `metal_ir_source_relation` after
    `metal_ir_operation_edge_records`.
- **Not mine, untouched:** the downstream source discharge, fusion legality, final `Relation`, the graph C owner,
  `Work`/`Changed`, and the Python callers.

## 8. Setter review correction: pairwise storage-key equivalence (uncommitted; READY, HOLD)

- **`Geometry`.** The root-wide `keyed` map and its `keyed.second` exclusion are deleted. Every individually exact
  view-to-pool mapping is kept: typed value whole, a single view with w's offset, element, pairs and write, keyed by the
  value's storage.
- **`Answer`.** For each candidate retained relation holding both accesses' original views, and before its witness
  is asked or accepted, it compares the original keys' equality, `original[from.view].key ==
  original[to.view].key` (the root's views), with the typed keys' equality,
  `p.viewPool->views[x].key == p.viewPool->views[y].key`. A candidate is used only where the two agree.
  - Distinct original storages merged into one typed storage give no absence.
  - One original storage split into two typed storages gives its hazard to neither.
  - Same-storage pairs with preserved coordinates stay reusable, even where another root was merged.
  - Distinct-storage pairs stay distinct.
  - Other candidates are still searched. Only an accepted known answer is memoized.
  - Decided's own relations always agree, because their views are the pool's.
- **Orientation** remains the graph owner's (`mesh_graph_relation_witness` and `_cause`).
- **Comment.** The `ImportConflicts` declaration comment states the pairwise condition.
- **Check.** `metal_ir.mm` exits 0 with no diagnostics under the canonical flags with `-fsyntax-only`.
- **Hunks:** in `metal_ir.mm`, `Geometry` (the filter removed, its comment), and `Answer`'s comment and key check; in
  `metal_ir.h`, the `ImportConflicts` comment line.
- **Unchanged:** the setter, append and source record (accepted), the downstream numerical and source-effect code,
  and the graph owner.

**READY.** Holding `metal_ir.h` and `metal_ir.mm` for the supervisor's build or commit window.

## 9. Shared `ViewRelation.pool` owner (uncommitted; READY for the joint build)

- **Exact type.** `struct ViewRelation { std::shared_ptr<mesh_graph_relation> relation;
  std::shared_ptr<const std::vector<uint32_t>> pool; };`. `pool[v]` is the Program pool identity of the relation's
  original view v, `UINT32_MAX` for none. It is immutable, made once where the relation is registered, and shared,
  never copied, by specialization (`viewRelations` is copied as `shared_ptr`s) and by any result that retains it.
- **Producers.**
  - `Decided` moves its derived map into the owner (`make_shared<const vector>(std::move(pool))`).
  - `metal_ir_source_relation` and the append's source bindings move `Geometry`'s freshly derived map into the owner
    through the internal `Retained`, so no second copy is made.
  - The public `ImportConflicts(Program &, const std::shared_ptr<mesh_graph_relation> &, const std::vector<uint32_t> &)`
    is unchanged. It makes the owner once from its argument, the caller's own vector.
- **Reads.** `Register` reads `*pool`. No other read existed. The pairwise alias-equivalence check in `Answer` is
  unchanged.
- **Final decider (compiler-owned; documented fields, not edited by me).** Under the working-tree `mesh-graph.h`,
  `mesh_graph_decided` is `{struct mesh_graph_relation *relation; const uint32_t *map; struct mesh_graph_witness
  witness;}`, and `mesh_graph_relation_decided` takes a `const uint32_t *map`. For a known answer `w` from
  `Answer`, the decider writes:
  - `relation = p.viewRelations[w.relation].relation.get()`;
  - `map = p.viewRelations[w.relation].pool->data()`;
  - `witness = w.witness`.

  To keep the map alive past the program, it retains `p.viewRelations[w.relation].pool` (the
  `shared_ptr<const std::vector<uint32_t>>`) alongside the relation in its result owner. It does not copy the map or
  borrow Program storage.
- **Checks** (canonical `.build/libmetal_ir.dylib` flags with `-fsyntax-only`):
  - `metal_transform.mm` exits 0 with no diagnostics.
  - `metal_ir.mm` has exactly 3 errors, all in the compiler's in-progress `FinalDecider` against the working-tree
    header's new `map` field and parameter:
    - line 1868, `*out = {relation, witness}`, no viable `=` for the 3-field `mesh_graph_decided`;
    - lines 2112 and 2223, `mesh_graph_relation_decided` calls without `map`.
  - None of the errors is in my hunks. They clear when that consumer adopts the fields above.
- **Hunks:**
  - `metal_ir.h`: the `ViewRelation` definition and its comment.
  - `metal_ir.mm`: `Register`'s read; `Decided`'s `push_back`; `Retained` plus `ImportConflicts`; and the two
    `Retained` calls in `metal_ir_append` and `metal_ir_source_relation`.
- **Not touched:** the numerical operation relation, the final decider, archive ownership, and the graph owner.

**READY** for the joint build. `metal_ir.h` and `metal_ir.mm` are held.

## Interfaces awaiting review at the intermediate checkpoint

- The numerical worker's operation relation (`Program::relation`, with source edges, composition and fusion
  legality) is separate from these memory-only access relations. If its legality needs memory conflicts between
  operations, `EffectConflict` gives each pair's original record, its relation, and the two effect indexes. The
  operation endpoints are the caller's a and b. Whether it wants these, or a quotient of the access relations onto
  operations, is for root to settle with that worker.
- No build or client run has been done. After the hold, compile an existing E2B phase program through the canonical
  path and compare the lowering result and compile time against `9a548ba3`.
