# Owned exact-result consumer — status

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5, 2026-11-13. Unit: owned-relations-assignment.md.
Bases: metal-microbench f132ccf2, mesh 4fda8f54 (working trees). Nothing committed by this worker.

## Changed paths (for coordinated supervisor commit)

metal-microbench:
- metal_recording.h: `struct MetalActors { uint32_t parts; double *host, *work; size_t edges; struct mesh_graph_dependence *edge; }`;
  `int MetalRecorderActors(program, gpu, cpu, struct MetalActors *actors)`; `void MetalRecorderActorsFree(struct MetalActors *)`.
- metal_recording.m:
  - Actors: MetalRecorderActors zero-initializes `*actors` and calls the static fill once. mesh_graph_quotient runs once, into
    `result->edge/edges`. host/work are calloc'd P each, and the existing cost accumulation is unchanged. ENODATA keeps the populated result;
    any other error frees the partial allocations through MetalRecorderActorsFree, which also re-zeroes the record. The `room`/ERANGE
    probe path is deleted.
  - Plan-order contraction (MetalPositionAccesses tail): the K·K capacity count is deleted. It now makes one
    mesh_graph_contract_unrolled(..., lag, &edge, &made) call, and the existing loop reads from/to/delay unchanged (unrolled vertex ids
    `copy·V + v`, delay as the producer returns it), with free in @finally.
  - Prepared dependence result (MetalProgramPrepare): the probe and ERANGE retry are deleted. It now makes one
    mesh_graph_dependences_views(..., &cause, &all) call. Every conflict record (kind/given/key/from_view/to_view) is consumed for the
    GPU→CPU classification (`from_view != SIZE_MAX`, side[from_view] GPU, side[to_view] CPU) before adjacent duplicate (from, to) pairs are
    quotiented, the same as before. It then frees, and mesh_graph_synchronization and its successors are unchanged.
- metal_ir.mm Conflict: the read/write view filtering is unchanged. One owned dependences_views call, then free. The error text is
  recorded for any nonzero status (ERANGE was a capacity signal only and no longer exists). The return is still
  `status != 0 || edges > 0` (conservative).
- tools/native_bench.m Plan, tools/e2b_replay.m program_actors: one MetalRecorderActors call. Each copies only what its existing owner
  holds (native_bench: three NSData; e2b_replay: the held host/work/edge NSMutableData that mesh_function points at), then calls
  MetalRecorderActorsFree. The ENODATA message and continuation are as before.

mesh:
- rdma/mesh_compile.py `_actors` hunk only: a ctypes `_Actors` Structure. argtypes/restype are set once per call of `_actors`. One call per
  row, memmove into the existing ctypes arrays, then MetalRecorderActorsFree. The `made[k]` tuple shape is unchanged
  `(host, work, edges, parts, count, complete)`.

## Source checks run (private; no shared library build)

- `rg` over metal-microbench, mesh rdma/ and compiler/: every caller of MetalRecorderActors, mesh_graph_dependences_views,
  mesh_graph_quotient and mesh_graph_contract(_unrolled) uses the owned form. The remaining entries are comments and the producer's own
  mesh_graph.py.
- Existing Makefile compile lines with `-fsyntax-only` (flags taken from the Makefile variables, so `-Wall -Wextra -Werror` is kept):
  metal_ir.mm (clang++ -x objective-c++ $(TINT_CFLAGS) -fobjc-arc …) rc=0; metal_recording.m rc=0; tools/native_bench.m rc=0;
  tools/e2b_replay.m ($(PROVIDER_CFLAGS)) rc=0. The first metal_recording.m run failed: the new parameter `actors` shadowed the body's
  existing `NSMutableData *actors`. The static fill's parameter is now named `result`, and the re-run passes.
- `/Users/mdot/.venv-mesh-uv/bin/python -m py_compile rdma/mesh_compile.py` PASS.
- Layout check: the C `struct MetalActors` (cc, /tmp) and the ctypes `_Actors` both give sizeof 40 with offsets 0/8/16/24/32.

## Producer interface dependency

These are consumed as they stand in the working tree, uncommitted, owned by the graph producer:
- rdma/mesh-graph.h (sha256 4eed61ac…4a87): owned-result signatures for dependences_views, contract, contract_unrolled and quotient.
  `struct mesh_graph_conflict` is 40 bytes, with from_view/to_view at 24/32.
- rdma/mesh-graph.c (d62d6ea6…7b0517): implements them.
- rdma/mesh_graph.py (aa5a15c2…6595): wrapper `_relation` uses the owned form.

Source readiness is coherent with those three. The consumer commit must land together with (or after) the producer's: these sources do not
compile against the committed 4fda8f54 header.

## Runtime evidence scope (not run here)

Not run: link, dylib build, or any execution. The existing target checks (libmetal_ir, libmetal_recording, native_bench, e2b_replay, the
mesh_compile `_actors` path) follow the coherent shared build of libmesh from the producer's mesh-graph.c. They are the root's, per the
assignment. This unit closes result ownership and probe recomputation only. It makes no claim about the broad compilation dataflow audit.

## Supervisor checkpoint

Codex reviewed source/ownership and committed Metal consumer `ee02e6c4`, paired with graph producer `5a43e4b6`. Native worker remains the same verified Opus5.5 session. Shared ABI build/runtime check follows the coordinated extent checkpoint; no runtime claim is added.
