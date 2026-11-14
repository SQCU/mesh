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

EXPIRED build window: the canonical E2B/native-bench build passed and the original
geometry owner is committed in metal `4db95f8d`. Resume the exact source-witness
import and original query handoff below. No root source hold remains.

LATEST compiler/source seed contract (old hold expired except imminent canonical build):
Expose `ImportConflicts(Program&, const std::shared_ptr<mesh_graph_relation>& original, const std::vector<uint32_t>& pool)` where pool[a] is the Program canonical geometry identity corresponding to ORIGINAL relation.view[a] (UINT32_MAX for external/unrepresented access). Original relation is retained, not copied. Numerical producer forms this map using exact source access(rootValueID,viewValueID) after frontend adopts actual out/to destination; source relation is one canonical solve with SSA/channel GIVEN vector, source access geometry/view association unchanged except explicit storage-key correspondence. Do NOT infer root-wide overlapping effects or expand op Cartesian products.

Import must register original positive witness identity and exact queried/no-hazard coverage, not rederive RAW/WAR/WAW. Current original query mode is known at its producer: source uses dependences_views; pooled Decided batches use dependences_views; recorder uses conflicts_views. The graph owner must retain that mode/coverage when it exposes exact pair query/projection; numerical supervisor is coordinating this small addition after current callable build. Program importer consumes original record+orientation references. Swapping already-known geometry pair must use canonical graph owner's directional cause projection (same producer logic), not rerun ISL and not transcribe hazard formulas here. Unknown/newly transformed geometry uses the existing canonical solver once. Existing invalidation by changed geometry remains.

To connect final dispatch Relation without re-solving old geometry: feed actual final records through SAME pool interning/correspondence and import source known witnesses; obtain final occurrence edges as graph-owned projection referencing these retained facts. Do not fabricate new conflict arrays from reduced kind maps. Exact output-occurrence/view maps belong numerical owner; cache exposes original relation+record/coverage via this ImportConflicts handoff.

Root review of current Answer: keep already-answered pair lookups as retained
ViewWitness references too. Removing the kind-only map should not make each
repeated query join viewOriginals[x] × viewOriginals[y] again. A lazy memo for
actually queried known answers can retain {relation, original graph witness},
including known absence and orientation, without dropping the original result
or seeding all negative pairs. Never memoize unknown as absent. This preserves
the earlier repeated-query performance property while completing the full-result
handoff. Compiler's placement worker now owns only adjacent Changed function-row
invalidation plus callgraph fields; your import/cache hunks remain disjoint.

Canonical coverage/orientation API concrete contract now compiler/graph-query.md. Numerical64ddaf6e implements relation_views (same solver once + retained query mode) and relation_witness/cause (original record+orientation/known absence, no ISL). Your ImportConflicts signature above consumes original owner/pool map; register source-view correspondence rather than eagerly enumerate all V² negatives. Program directional cause formulas remain forbidden. Generic full archive owned schedule worker separately, not blocker to this import.

LATEST ROOT-APPROVED PAIRED SOURCE SETTER (all prior source-setter holds EXPIRED)
Root593 owns public setter + retained operands + exact ImportConflicts call now; numerical64dd owns its actual semantic discharge/fusion/final dispatch consumer. Frontend SAME Torch owns actual constructor invocation. Use these exact shared names/fields, no bare-edge fallback:

C:
enum { METAL_IR_SOURCE_VALUE, METAL_IR_SOURCE_LITERAL, METAL_IR_SOURCE_SENT, METAL_IR_SOURCE_RECEIVED, METAL_IR_SOURCE_EXTERNAL };
struct metal_ir_source_operand { uint32_t kind, value; };
int metal_ir_source_relation(struct metal_ir_program *, struct mesh_graph_relation *original,
 const size_t *operation_first, const uint32_t *operations, const uint32_t *view_values,
 size_t givens, const struct metal_ir_source_operand *given,
 void (*release)(void *), void *owner, char *error, size_t room);

Original relation domain gives vertices/views. operation_first[V+1] and operations[] map each ORIGINAL vertex to zero/many actual operation IDs. view_values[original.views] are actual typed logical ValueIDs, UINT32_MAX external. Given array uses ORIGINAL GIVEN.given namespace; VALUE literal scalar already embodied in typed region uses LITERAL/UINT32_MAX, actual typed VALUE and SENT/RECEIVED use valueID (root derives Value.storage); EXTERNAL absent. Original original relation/causes/view operands remain authoritative and retained, no root-wide/cartesian op expansion.

C++ shared record contract:
`SourceRelation` retains shared_ptr<mesh_graph_relation> original, borrowed `const size_t *operationFirst`, `const uint32_t *operations`, `const uint32_t *viewValues`, `size_t givens`, `const metal_ir_source_operand *given`, and one Owner metadataOwner; destructor releases adopted metadata owner. original retain independent; callback adopted only success. NULL callback means actual caller Constructed owns arrays+original source through Target's construction lifetime (frontend confirms directchain; no global registry).
`SourceBinding` retains shared_ptr<const SourceRelation> source, uint32_t operationOffset=0, and shared_ptr<const vector<uint32_t>> values=nullptr (identity); nonnull maps source declaration ValueID→this Program ValueID. Program vector<SourceBinding> sourceRelations. These are immutable producer annotations. Specialize copies only bindings/shared owners, no original arrays/results cloned. Append preserves original SourceRelation, adds actual dst operation base, and composes existing append `out[]` value correspondence for source bindings; necessary mapped array is a derived value-ID correspondence, not source evidence reconstruction. Preserve original record IDs/provenance.

Setter owns source construction handoff, calls ImportConflicts using actual typed geometry-to-pool correspondence. Do not seed negatives where source→typed alias/storage or coordinates changed: original distinct source storage keys coalescing onto one new typed storage is genuinely changed alias geometry, not proof of no hazard. Preserve source mapping even when it cannot directly seed a geometry fact; numerical canonical owner handles new geometry once. Exact source views are mapped from frontend adopted out/to destination values, not guessed by root names. No reject policy or numerical copies. No source all-to-all edges.

Publish declaration/fields promptly; numerical needs them now. Root owns this exact setter/record/append/shared specialization hunk, not numerical final Relation/transform legality. Frontend owns actual caller. Private syntax and HOLD; no shared build or commit without supervisor window.

Setter review, agreed with compiler: Geometry's current root-wide `keyed.second`
exclusion is too broad. Preserve every individually exact view-to-pool mapping.
At Answer's actual pair, compare original storage-key equality with actual typed
storage-key equality before reusing its witness. A distinct-original-root pair
merged into one backing supplies no negative proof; same-original-root pairs
with preserved coordinates remain reusable even if an unrelated root was merged
elsewhere. A source alias split into distinct actual roots likewise does not
transfer its positive memory hazard to those new roots. This is a pairwise
mapping condition, not permission to throw away an entire root's solved facts.
The canonical graph still owns hazard orientation. Delete the broad root map
filter rather than accumulating a second policy beside it.

Final paired owner request from compiler: the actual field is ViewRelation.pool
(currently vector<uint32_t>), not Decision.geometry. Make that existing immutable
correspondence a shared_ptr<const vector<uint32_t>> owner, produced once at
Register/Import construction and shared by specialization and final result
consumers. Adapt your Register/Decided/Import reads and publish its exact type.
Numerical will retain the same owner alongside original relation when recording
which earlier queries supplied negative answers; it must not copy this map or
borrow Program-owned storage that dies first. No source/result payload copies,
no new graph algorithm, and no V-squared negative table. Your API otherwise stays
the same; compiler owns the final decider and generic owner/archive extension.
