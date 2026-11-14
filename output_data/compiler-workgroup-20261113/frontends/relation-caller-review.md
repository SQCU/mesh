# Retained source relation caller checkpoint

2026-11-13. Codex frontend review of SAME Torch9740398d (turn19) and LiteRT41f48493 (turn13), both explicit/reported Claude Opus5.5. Raw logs/work orders remain under metal-microbench output_data/compiler-workgroup-20261113/frontends.

The source dependence owner now submits the original memory views and supplied SSA/channel causes in one native call and retains their original arrays in SourceRelation. Graph reads the canonical relation descriptor; schedule uses that relation's native order, closure and contracted relation. Edge/member/order results are borrowed reusable views with direct Python owner composition. ADOPTED registry/callback indirection and the second closure Graph are removed. Original source-to-copied-node, source access and supplied-cause correspondences reach ConstructedMachine; the native cause import/discharge consumer remains open.

Typed numerical construction now supplies selected out/to destination views to the existing shared lowering owner, retaining CAT destination storage and exact aliases rather than allocating independent producer outputs. Mutation output kinds/original destination roots remain explicit. This checkpoint does not close nested destination wire geometry or sharded mutation placement; those exact producer-consumer corrections remain assigned. No root-wide overlap inference or new memory solver is accepted.

Validation: AST of seven core Python files and git diff --check pass. Fresh imports against canonical Local144/RelationData144 build pass. Existing native Graph order/closure and repeated contracted-edge view access pass. The existing CPU Engines client reaches construction after schedule; its first attempt fails locating libmetal_recording.dylib, and executor is correcting the existing loader environment. No new numerical/performance claim.

Immediate continuation: closure consumes explicit retained native order/stride rather than numeric vertex IDs; skip contraction when no vertices are eliminated; remove obsolete after/preds reconstruction. The first two general graph algorithms remain in the existing C owner. Remaining Python graph algorithm migrations and source-cause C consumption are not closed by this checkpoint. Exact C working checkpoint26861434 is separate.

Continuation source-ready: the same search's borrowed reverse order now feeds closure as original vertex IDs plus
pointer stride. Numeric-ID topological assumptions are removed. Identity contraction is skipped and after/preds
reconstruction is deleted. The source producer directly consumes mesh_graph_relation_views, retaining complete
original query coverage and native conflict ownership; erased local channel causes carry their ordinary typed
operand proof. C strict syntax and Python AST/diff pass. Matching descriptor152/closure native rebuild and actual
API observation remain pending at this entry.

Matching graph9388eb3e gate: fresh actual RelationData152/source relation_views query DIRECTED, source causes,
closure over nonnumeric original vertex order, and reusable views after caller Graph reference deletion all pass.
Executor's existing CPU FFN then passes typed machine/local planning and native artifact reload in Target.prepare,
error0.0,
output[2,32],55values,one native entry. Its earlier failure was post-lower STEP value creation, corrected by the
actual existing compiler word query. No outer archive.save/load filesystem roundtrip was exercised. This is CPU client evidence; source-cause completeness, GPU, transport and
performance remain separate obligations.

2026-11-14 UTC continuation: SAME Torch turns22–27 and LiteRT16, explicit/reported Opus5.5. Source setter now consumes
the original retained relation, original vertex-to-operation CSR, original access-to-logical-ValueID mapping and GIVEN
operand roles before lowering. ConstructedMachine retains the borrowed arrays and original owner. Native cause
consumption/final-fact reuse is a paired compiler/root unit still under review; no endpoint-only fallback is used.
Tuple-getitem producer identity is projected once in source order and retained on SourceRelation for all three consumers.

Weighted longest-path computation moved from the Python helper to mesh_graph_longest over the same relation, original
edge weights and borrowed traversal order; shares.stages reads its native output view. Descriptor alignment now includes
Witness16, Origin24 and RelationData160. C strict syntax and Python AST/diff checks pass; matching shared rebuild and
fresh actual longest/source-setter checks are pending, so these are source-readiness statements.

The production driver and archive use the typed native Target and its emitted signature. Tracked caller inspection plus
executor ownership review found no use of the former AOT emitter/Target/restore or Python signature reconstruction.
Those bodies and their only bare-pair contract/quotient/unrolled wrappers are deleted; the canonical C algorithms remain.
Copies/copies stays the live source grouping owner. Its overlap solve and copied overlap table had only the deleted AOT
consumer and are removed; actual typed copy operations supply the native compiler's memory effects.

Seeded source operations now receive stable site IDs from the original export order before placement. Local carries that
ID explicitly into ordinary random, RReLU, randperm and attention draw owners. Generator's execution advance and existing
element/shard indexing are unchanged. A legal graph order change therefore does not assign a different source draw site.
Installed Torch2.14.0 directly reports nondeterministic_seeded=True and pointwise=False for rand, randn, native_dropout,
randperm, rrelu_with_noise_functional, scaled_dot_product_attention and prims.normal. The earlier worker note that
prims.normal lacked the tag was incorrect. No exception list was added. Actual RNG construction/numerical evidence is
pending the coordinated native ABI release; no distributed stochastic or performance claim is made.

Remaining live generic ownership includes placement replication/view-chain queries, rdma fixed-point sharding and
schedule transfer-event precedence. Domain transfer application over an already produced order is distinct from a
second traversal algorithm. No dead helper was ported merely to preserve its old API.

Transfer-event continuation (SAME Torch28–29): schedule retains one event relation while consuming its native redundant
mask and class-local source/sink flags. Event edge construction uses the existing source closure; the generic reduction
and endpoint algorithms stay in mesh-graph.c. The caller projects those endpoints into the existing transfer IDs, delays
and endpoint-role flags only after their graph consumers complete. Delay-1 isolated self edges remain; cross-thread edges
are excluded only from each thread's boundary query, as before. Old Python precedence/reduction/extremes and their unused
direct ctypes registrations are deleted. Source graph arrays are not copied into a second reduction/DFS input. Pair
enumeration becomes canonical edge order; the declared pair/lag/role set is preserved by source review. Current whole
mesh-graph.c strict C syntax plus the two Python AST/diff checks pass. Matching actual boundaries/longest API and caller
observations remain pending the coordinator's library release.

2026-11-14 UTC event-owner checkpoint (SAME Torch30–31, explicit/reported Opus5.5): schedule returns one native
Dependence array retaining its event relation, source closure, redundancy mask, boundary views and record-to-event/edge
correspondence. mesh.function borrows that exact array and retains its owner; Declarations/Plan keep it through final
plan and signature construction. Archive uses shallow Plan projection rather than deepcopy of native owners, retaining
the existing persisted dependence records. Loaded final plans do not query the source event proof again: the canonical
plan/template and signature are its final compiled consumers, so no second JSON graph is introduced.

Actual canonical-library observation: a three-event source relation produced three records; Function.dependence had
the identical native address, its retained owner was the original array, and mask/records remained valid after source
references were deleted and GC ran. Empty events retained an actual zero-length array/count. AST and diff checks pass.
This is ownership/API evidence, not live transport execution.

Matching graph observations also passed longest paths with nonnumeric vertex order and borrowed results after Graph
reference deletion, and native event boundaries. An actual Torch export→placement→devices→schedule construction
reversed two independent random nodes while preserving original source site IDs 1 and 2; draw consumers saw 2 then 1,
not reassigned traversal IDs. No statistical or distributed RNG claim.
