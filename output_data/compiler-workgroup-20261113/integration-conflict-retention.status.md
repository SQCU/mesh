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
- **`workCache`.** Superseded by the work-retention section below.

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
- First decisions are now batched (work-retention section): one solver call for each distinct set of undecided
  partners, not one for each from-access. The net call count against the old per-query calls has not been
  measured.

## Work retention (`integration-work-retention.md`)

Committed so far: the conflict checkpoint, metal-microbench `2b2cfb38`. Uncommitted and mine: two hunks in
`metal_ir.mm`, `Changed` and `Decided`. Evidence is syntax only.

### The mutation owners, from the source

`Work(p, f, gpu)` is cached per region-module `ir::Function*` and recurses into `UserCall` targets. These are every
in-place rewrite of a function body that it could observe:

| Owner | Function rewritten in place | Followed by |
|---|---|---|
| `Transform` start, `metal_transform.mm` `Simplify` (`Fold`, `Prune`) | each operation's `region`, `combine`, `finalize` | No `Work` query has run yet in this compilation. `Work` is reached only through `Time`, the dispatch `Actor`, and `metal_lower.mm` lowering, all after this point. |
| `Fuse(..., true)` | consumer C's `region` (`SetParams`, spliced body); a new `prologue[slot]` | `Changed(p, consumer)` |
| `Epilogue` | producer P's `region` (`SetParams`, body); `epilogue[j]` replaced, the prior one discarded | `Changed(p, producer)` |
| `Inline` | consumers through `Fuse`; P is then erased and its functions discarded | `Changed(p, producer)` before `Discard` |
| new `pre`/`post` functions | `Simplify` before they are installed | not yet reachable when rewritten |
| `metal_ir.mm` `Binary64(Kernel &)` | a kernel's emitted module, not the region module | not a `workCache` key |

Not owners:
- `Fuse(..., false)` returns before any edit.
- `ZeroPoints` edits no function.
- `Discard` destroys only the functions of an erased operation, which no live function calls.

**Sharing.** Region-module `UserCall`s come from three places, and none of their targets is ever rewritten:
- `metal_clone.mm` `Callee`, which clones a source callee once, complete.
- `Splice` within one module (`x.same`), which keeps the original callee.
- `LogicalRead`, which calls a logical-load function built once, complete, at creation.

An operation's own functions are never `UserCall` targets: a producer's body is spliced into its consumer, not
called. So when an operation changes, the only invalid `Work` entries are those of its own functions. Their callers
have no entries, because none exist.

**Replaced versus in place.**
- A function rewritten in place has its entry erased by its operation's `Changed`.
- A replaced or discarded function keeps an entry that is never looked up again: no live operation or caller
  references it, and Tint does not reuse function objects in a module. Examples are Epilogue's prior epilogue and
  the erased reader's region, which is nulled before `Changed(p, reader)`.
- `metal_ir_specialize` shares the region module and its function pointers, so its copied `workCache` keys name
  the same unchanged functions.

### Edit

- `Changed(p, op)` erases from both `workCache` maps the entries of `op.region`, `op.combine`, `op.finalize`,
  `op.prologue[*]` and `op.epilogue[*]`, instead of clearing them. Every other completed `Work` result is
  retained.
- `Decided`: undecided partners are collected exactly for each from-access (sorted, unique). From-accesses with
  the same partner set are then grouped into one `mesh_graph_dependences_views` call. Its cross pairs are exactly
  those undecided pairs, so no decided pair re-enters the solver. Record mapping: `from_view` indexes the group's
  from-accesses, and `to_view - froms.size()` indexes its partners.

### Checks

The canonical `.build/libmetal_ir.dylib` compile flags with `-fsyntax-only`: `metal_ir.mm` and `metal_transform.mm`
exit 0 with no diagnostics (logs in `/tmp/integration-evidence/`).

### Remaining

- Build and client comparison. The inbox has not announced a coordinated build window, so neither
  `libmetal_ir.dylib` nor any client has been built or run. Once it opens, compile an existing E2B phase program
  (`.build/e2b_replay` on a prepared directory, from its existing inputs) before and after `2b2cfb38` plus these
  two hunks. Compare the lowering result for equality and the compile time. No counter or harness has been added.
- Staging. Next to my `Decided` hunk, the working tree also carries another worker's move of `Renumber`, and
  `metal_transform.mm` carries 34 lines that are not mine. Stage only the `Changed` and `Decided` hunks above.

### Root review and subsequent build

Root reviewed the actual function mutation/sharing argument and the retained canonical hazard outputs. Conflict
retention is checkpointed in metal `2b2cfb38`, with root's selected-hunk index placement error corrected in
`49b10ee4`; the working source was never changed by that staging correction. Unaffected Work retention and
unknown-pair batching are the subsequent connected source unit. Compiler supervisor's canonical
`make .build/e2b_replay .build/native_bench` completed successfully with the source present, followed by fresh
actual imports and Program create/storage/free (Local 136 bytes, Storage 24 bytes). Evidence logs are
`compiler/local-construction-metal-build.log` and `compiler/local-construction-abi.log` in this workgroup.
No numerical or performance comparison is claimed by that build/import evidence.

The operator's subsequent single graph-module/result-view instruction adds an immediate ownership review of
this representation too: canonical graph results must stay owned and accessible through their consumers. A
Program-level projection of hazard kinds is not an excuse to introduce a second graph-result owner or discard
its original relation/correspondence. That consolidation is coordinated with the native graph-owner migration.
