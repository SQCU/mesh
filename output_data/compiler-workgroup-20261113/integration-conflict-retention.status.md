# Conflict retention

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

2026-11-13, integration worker (Claude Opus 5.5, session `59368744-da56-491a-871a-1be809c0cc93`). The contract
follows the compiler supervisor's view-pool proposal and the root corrections in the matching `.inbox.md`. It is
implemented. Evidence is syntax only: no library build and no run.

## The result that was lost

`metal_ir.mm` `Conflict(p, a, b, aWrites, bWrites)` ran `mesh_graph_dependences_views`, kept only `edges > 0`, and
freed the records (kind, key and witness views). Eleven fusion queries in `metal_transform.mm` asked two or three
restricted questions of one operation pair, each a new exact solve. `Inline`, then `Epilogue`, then `Proven`
asked the same pairs again while neither operation's effects had changed.

## Implemented connection

- **Producer.** `mesh_graph_dependences_views`, the canonical relation, with the operations' actual access flags.
  Hazard existence and kind are its outputs. No access-flag or cause rule is restated in metal_ir.
- **Result and owner (`Program`, `metal_ir.h`), all memory-only, beside `operationEffects`:**
  - `viewPool` and `viewIds`: each distinct access once. The exact key is `std::vector<uint64_t>{key, offset,
    element, write, pairs...}`, so every operand is kept bit for bit and nothing is hashed.
  - `operationViewIds[op]`: the pool identities of `Accesses(p, op)`, made where `Accesses` makes
    `operationEffects`.
  - `viewConflicts`: for an ordered pair of pool identities (`from << 32 | to`), the relation's hazard kind + 1,
    or 0 for none. This is an immutable fact.
- **Decision (`Decided`, `metal_ir.mm`).** For a query, the undecided same-storage ordered pairs are grouped by
  their from-access x. Each x gets one call over x (vertex 0) and only its undecided partners (vertex 1), so a
  decided pair never re-enters the solver. Reported partners take the call's kind, and the rest take 0. A failed
  call records no fact, returns its status, and sets `Program::error`, as before.
- **Projection.** `int OperationConflicts(const Program &, uint32_t a, uint32_t b,
  std::vector<mesh_graph_conflict> &into)` appends one record for each hazard pair of a's and b's views:
  `from = a`, `to = b`, the relation's `kind`, `key`, and `from_view`/`to_view` = the pool identities (witnesses).
  Orientation is the caller's: facts are stored per ordered pair, so a reversed query (b, a) is its own decided
  fact. Following the compiler agreement, it carries no source edges; the numerical worker combines its source
  givens.
- **Consumers.** `bool Conflict(const Program &, uint32_t a, uint32_t b, mine, theirs)` answers whether some
  hazard pair has a's view satisfying `mine` and b's satisfying `theirs`, or the solve failed. The 9 call sites
  (11 queries) now use it:

  | Site (`metal_transform.mm`) | a, b | Predicates | Old filter |
  |---|---|---|---|
  | `Inline` 478, 479 | producer, o | written (`key == S && write == 1`), then `Writes` or `Any` | `Of(..., S, true)`; `bWrites` true/false |
  | `Inline` 501 | producer, o | `Reads` (`!write`), `Writes` (`write != 0`) | `read`; `bWrites` true |
  | `Epilogue` 549, 551 | producer, o | written of y's storage, `Any` / `Writes` | `Of(..., S, true)` |
  | code motion 585, 588 | reader, producer or o between | reads/writes of reader not on S, `Writes` / `Any` | `x.key != S` split by `x.write` |
  | `Proven` 1105, 1111 | member, o (possibly the member) | `writes` (`key == V.storage && write == 1`), `Any` / `Writes` | `Of(..., V.storage, true)` |

  Each answer equals the old one. The old solve tested the same pairs, with the same flags, over subsets of the
  same effects. A pair's decision depends only on the pair, and the relation's hull sweep enumerates every
  overlapping pair.
- **Invalidation.** `Changed(p, op)` already resets `operationEffectsKnown[op]`, and the next `Accesses`
  remakes `operationViewIds[op]`. Only the operation-to-view association changes. Pool entries and
  `viewConflicts` stay valid, because they describe accesses, not operations. `Changed` is not edited.
- **Copy.** The program copy (`metal_ir.mm`, beside `operationEffects`) copies `viewPool`, `viewIds`,
  `operationViewIds` and `viewConflicts`.

## Unchanged, with the reason

- **`metal_lower.mm` 1186, member-barrier `Before`.** `view_conflicts` skips same-vertex pairs before any exact
  test, so each (pending id, new id) pair is tested once, and the barrier consumes only the yes/no. Root confirmed
  this is legitimate scratch disposal. The vector `Conflict` stays for it.
- **`p.conflicts`.** The dispatch relation is solved once after `Lower` and is already retained. `Changed` clearing
  it during lowering is changed-input invalidation.
- **`workCache`.** Clearing it in `Changed` does lose unrelated functions' `Work` results between fusions,
  through `Time` → `Actor` → `OperationFacts`. But `Work` recurses through `UserCall` targets, and retained
  LiteRT bodies have cloned callees (`metal_clone.mm` 113 `Callee`). Erasing only a changed operation's own
  functions could leave a caller's stale entry if a shared callee is mutated. Exact invalidation needs the
  function mutation and sharing relation. Root required that, and I have not built it, so clear-all stays.

## Exact source edits (mine only)

These files also carry other workers' uncommitted hunks: `metal_ir.h` `metal_ir_storage` alignment, and
`metal_ir.mm` `Storages`. Stage only the hunks below.

- `metal_ir.h`:
  - `#include <map>`.
  - The four `Program` fields after `operationEffectsKnown`, with their comment.
  - The `OperationConflicts` declaration and the `Conflict(p, a, b, mine, theirs)` declaration after the existing
    vector `Conflict`.
- `metal_ir.mm`:
  - `Pooled`, before `Accesses`.
  - `Accesses`: the `operationViewIds` resize, then its fill.
  - After the vector `Conflict`: `Ordered`, `Decided`, `Hazards`, `OperationConflicts`, and the restricted
    `Conflict`.
  - The program copy: two lines after `operationEffects`.
- `metal_transform.mm`:
  - `Any`, `Writes` and `Reads` after `Of`.
  - The 9 call sites in the table. `Inline` and `Epilogue` no longer copy `mine`, `written`, `read` or `theirs`
    vectors used only for those queries. `Proven` keeps its `written` vector for the covering check.

## Checks

The `.build/libmetal_ir.dylib` rule's compile flags, with `-fsyntax-only` replacing `-dynamiclib` and linking,
were run per file against the working tree:

```
clang++ -x objective-c++ -std=c++20 -fno-rtti -isystem .build/provider/dawn ... -fobjc-arc -O2 -Wall -Wextra -Werror \
  -I . -iquote .build -I ../dox/mesh/rdma -fsyntax-only <file>
```

`metal_ir.mm`, `metal_transform.mm`, `metal_lower.mm` and `metal_clone.mm` each exit 0 with no diagnostics. Logs
are in `/tmp/integration-evidence/syntax-*.log`.

## Limitations

- No library was built and no numerical client was run. Root holds the shared build window, and the existing E2B
  client needs `/mesh0`. Syntax evidence says nothing about behavior or time.
- A pair's first decision now makes one solver call for each from-access with undecided partners. A single old
  restricted query made one call. Later questions about the pair, from any function and in either restriction,
  make no solver call. The net count depends on how often pairs are re-asked; it was not measured.
