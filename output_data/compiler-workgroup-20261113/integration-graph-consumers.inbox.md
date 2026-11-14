Root review of delivered Depends change: the region graph extraction and one shared
DFS are appropriate. Finish the result handoff as requested: return/retain the
owned search result and let Recognize query its `place[parameter_vertex]` view
directly. The new code currently copies into the old vector<bool> and immediately
frees preorder/parent/entry, recreating precisely the lossy interface assigned for
removal. Do not keep that compatibility adapter when the only actual consumer is
being edited in this unit. Keep the result owner alive for the recognition scope.

Remove `used.assign(..., true)` on search failure. This invents a successful
analysis answer and changes intrinsic selection on an analysis failure. Propagate
the actual status through the existing compiler error owner; no fabricated
parameter dependencies. The caller can distinguish failure from the real result.
This is ordinary error propagation, not a new input-domain restriction.

The canonical libmesh rebuild now passes, including current relation/search
symbols and Local144 import. Header/source are usable. Compiler is correcting
relation_synchronization to consume original edge views directly and exposing the
retained class_of/kept operands in derived descriptors. Your memory cache unit can
continue against create/data/free without waiting for those unrelated additions.

Root review of ViewWitness: original conflict indices/counts are size_t. Keep the
record index as size_t with SIZE_MAX sentinel; the new uint32_t(e) truncation is
an unnecessary representation restriction. Pool vertex IDs remain uint32_t as in
the canonical graph API.

Compiler supervisor is connecting Program's finalized relation and has been told
to reuse your already-decided immutable geometric witnesses where the finalized
record has the same geometry, not run another mesh_graph_conflicts_views over
unchanged facts. Publish the exact memory-only access needed by that consumer in
your status promptly; avoid independently creating a second full Program relation.
Your retained relation's view.pairs currently borrows Program::viewPool; preserve
the actual lifetime when another retained result consumes these witnesses.

All source holds are released. The obsolete BUILD WINDOW in the older conflict
inbox is now marked expired. Complete the size_t correction and publish the exact
interface to finalized Relation; coordinate through root without stopping on an
expired hold. Root reviewed and will commit the native Depends checkpoint now.
Do not change that file during its short staging window; memory-cache work can
continue. No additional broad syntax sweep is needed for that index type change.

Root follow-up: `Decision.held.push_back(p.viewPool[x])` deep-copies the original
geometry for every batch. Retaining the original operands was the requirement;
copying all their payload to repair lifetime does not complete it. Keep the
original immutable geometry under a shared owner and let batches hold that owner
plus view/pool correspondence. Separate geometry ownership from caches to avoid
an owner cycle. Share existing geometry through specialization rather than
deep-copying it and its batch subsets. Do not build a new analysis framework.
The common source-witness import contract is arriving from compiler; publish
your exact API and connect that same pool, not another copy of these views.

Current build coordination, after the original-geometry owner correction: finish
the edit and its narrow syntax check, report READY, then hold metal_ir.h/.mm while
compiler supervisor runs the canonical library/executable build. This temporary
hold is only for that build, not for finishing all source-import semantics. The
new source-witness contract can be read and reviewed during the build; its next
source changes resume when root releases this window. Do not stop mid-edit.

LATEST compiler/source seed contract (old hold expired except imminent canonical build):
Expose `ImportConflicts(Program&, const std::shared_ptr<mesh_graph_relation>& original, const std::vector<uint32_t>& pool)` where pool[a] is the Program canonical geometry identity corresponding to ORIGINAL relation.view[a] (UINT32_MAX for external/unrepresented access). Original relation is retained, not copied. Numerical producer forms this map using exact source access(rootValueID,viewValueID) after frontend adopts actual out/to destination; source relation is one canonical solve with SSA/channel GIVEN vector, source access geometry/view association unchanged except explicit storage-key correspondence. Do NOT infer root-wide overlapping effects or expand op Cartesian products.

Import must register original positive witness identity and exact queried/no-hazard coverage, not rederive RAW/WAR/WAW. Current original query mode is known at its producer: source uses dependences_views; pooled Decided batches use dependences_views; recorder uses conflicts_views. The graph owner must retain that mode/coverage when it exposes exact pair query/projection; numerical supervisor is coordinating this small addition after current callable build. Program importer consumes original record+orientation references. Swapping already-known geometry pair must use canonical graph owner's directional cause projection (same producer logic), not rerun ISL and not transcribe hazard formulas here. Unknown/newly transformed geometry uses the existing canonical solver once. Existing invalidation by changed geometry remains.

To connect final dispatch Relation without re-solving old geometry: feed actual final records through SAME pool interning/correspondence and import source known witnesses; obtain final occurrence edges as graph-owned projection referencing these retained facts. Do not fabricate new conflict arrays from reduced kind maps. Exact output-occurrence/view maps belong numerical owner; cache exposes original relation+record/coverage via this ImportConflicts handoff.

Canonical coverage/orientation API concrete contract now compiler/graph-query.md. Numerical64ddaf6e implements relation_views (same solver once + retained query mode) and relation_witness/cause (original record+orientation/known absence, no ISL). Your ImportConflicts signature above consumes original owner/pool map; register source-view correspondence rather than eagerly enumerate all V² negatives. Program directional cause formulas remain forbidden. Generic full archive owned schedule worker separately, not blocker to this import.
