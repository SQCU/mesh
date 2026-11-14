# Owned exact-result consumer — active implementation

Session: e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5.

Read design/compiler-workgroup-2026-11-13.md, compilation-composition-change-request.md items 2, 3, 6, 7, and current source before editing. Codex settled this finite interface with the compiler supervisor. This is implementation, not an audit or a redesign.

## Exact loss and replacement

mesh_graph_dependences_views currently solves and discards an owned vector to fill caller capacity. MetalProgramPrepare retries after ERANGE; MetalRecorderActors performs quotient during both capacity and fill calls. The graph producer owner is changing the existing C APIs to return their allocated result once. Consume that exact result without a second query/solve. Do not add a wrapper that performs both calls.

Graph producer (compiler sibling, do not edit): mesh_graph_dependences_views(..., struct mesh_graph_conflict **result, size_t *count); mesh_graph_contract, contract_unrolled and quotient return struct mesh_graph_dependence **result,size_t *count instead of room/output arrays. Caller free(). Existing conflict record field definitions remain the authority. Python graph wrapper return shapes stay unchanged. Read the new header as it lands; coordinate missing exact fields through Codex rather than inventing APIs.

## Owned files/hunks

metal-microbench: metal_ir.mm Conflict only; metal_recording.m exact graph callers (actors quotient, plan-order contraction, prepared dependence result); metal_recording.h actor result declaration; tools/native_bench.m and tools/e2b_replay.m actor-result callers.
mesh: rdma/mesh_compile.py actor ctypes/call hunk near _actors only. Do not touch lower_inductor, schedule, signature, provider or graph implementation. No broad formatting or comment cleanup.

MetalRecorderActors must itself return ONE completed owned result: parts, host/work cost arrays, edge count and dependence array. Use a small C result struct and a paired freeing function; creation fills a zero-initialized record. ENODATA retains usable populated costs/edges as before. Actual errors free partial allocations. Consumers build their existing mesh_function/ctypes operands from that result and retain/copy only what their existing owner needs before free. Delete probe/fill callers, not merely move them.

Conflict must preserve exact conservative error behavior and read/write filtering. Prepared recorder must preserve all causes used to classify GPU-to-CPU obligations before quotienting duplicate pairs; new conflict result must not drop kind/given/key/view indexes. Contracted delayed edges preserve delay and original IDs. No added graph algorithm, ordering policy, runtime instrumentation, or serialization changes.

## Evidence and delivery

Inspect every actual caller using rg. Run syntax/private strict compilation with existing Makefile commands when graph declarations are coherent; no shared library build. Coordinate C ABI source readiness through Codex. Existing target checks can follow the coherent shared build; do not create a parallel evaluator. Record exact source checks and remaining runtime evidence scope in executor/owned-relations-status.md. Commit nothing yourself; report named changed paths for coordinated supervisor commit. Do not reset/clean/stash or make source backups.

This unit closes result ownership/probe recomputation only. Do not claim the broad compilation dataflow audit complete.
