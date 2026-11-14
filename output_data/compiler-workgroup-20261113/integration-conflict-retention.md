# Retain exact conflict results through their actual consumers

This is an implementation assignment under the operator's current turn:

> try reviewing the entire codebase and identifying all mechanisms by which a compilation IR/optimized solution can be 'discarded', then removing all of those 'discard mechanisms' all at once by refactoring (of the entire data flow)

Use the existing compilation-composition request, items 2, 3 and 6, with their cited roles. The runnable
`~/schedule-emit` demonstrates stage-result ownership and composition. It is not a file-layout template.
Use the actual exact-view relation in `rdma/mesh-graph.{h,c}`; Bernstein's dependence conditions are already
cited at its consumers. No second intersection algorithm or natural-language reinterpretation of the graph.

The concrete current loss is in `~/metal-microbench/metal_ir.mm::Conflict`: it asks the canonical owner for a
conflict vector, reduces it to a Boolean and frees it. Inline, Epilogue, Locals and member-barrier construction
ask related questions repeatedly over unchanged operation effects. `Changed` also clears all work caches when
one operation changes. Actual changed operations, tile geometry and access restrictions are different operands;
they cannot borrow stale results. Final emitted code need not retain compile-time analysis.

Review these exact producers and consumers, then implement the smallest connected ownership change that retains
the computed facts for their actual consumers. Reuse existing effect identities, operation provenance and
mutation owners. Do not add a generic string-keyed cache, a guessed cache key, an independent graph analysis,
extra runtime machinery or a full eager pairwise analysis that increases work merely to save repeated queries.
Restricted read/write questions must consume the appropriate retained causes. Distinguish legitimate changed-input
analysis and scratch disposal from repeated work on the same operands. Check whether `workCache` invalidation
actually loses unrelated function results before changing it.

Source ownership is limited to conflict/effect-result ownership and its existing call sites in
`metal_ir.{h,mm}`, `metal_transform.mm` and `metal_lower.mm`. Compiler supervisor concurrently owns specialization
and emission in those same files. Do not stage or commit another worker's hunks. Do not change runtime, transport,
archive, target or numerical semantics for this unit. Root will coordinate the shared commit/build window.

First write a concise concrete producer/result/consumer and invalidation proposal to
`output_data/compiler-workgroup-20261113/integration-conflict-retention.status.md` in mesh. Name actual fields and
call sites, and the result being lost. Then implement that proposal unless the root feedback in the matching
`.inbox.md` changes it. Check the inbox between tool calls. Do not spend the turn surveying unrelated systems.
Use the same current Opus 5.5 session; do not spawn agents or change model.

Use source review, scoped compilation and existing numerical clients. No new repository tests, mock framework,
benchmark framework, source backups, resets, cleans, stashes, services or remote jobs. Record exact source edits,
checks and limitations in the status file. A passing diagnostic is evidence about its input, not a specification.
