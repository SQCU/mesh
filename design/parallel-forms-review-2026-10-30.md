# Parallel forms review, 2026-10-30

The library and its frontends implement several jobs twice or more. This review maps each form: where it forks, the
lines only it owns, who reaches it, what the repository's documents call canonical, and what deleting it requires.
It was made by reading the source at mesh `17e3d812` and metal-microbench `0de28182`; nothing was built or run.
Line numbers are those commits'. Counts are function boundaries plus counted branch lines, comments included.
The claims marked verified were checked against the source a second time.

A clone scan of rdma/nccl-mesh.c (identifiers and numbers normalized, four-statement windows) finds 99 repeated code
lines in 5,516. The file is not copy-pasted: what it holds twice are whole alternative forms of one job.

## Delete now: no caller

| What | Lines | Evidence |
|---|---|---|
| `python/mesh/` (`__init__.py`, `_native.py`, `collective.py`) | ≈1,000 | Deleted 2026-10-30. Nothing imported it, and `import mesh` gets rdma/mesh.py (mesh.pth adds `rdma/`). It was a Pallas-spelled block DSL whose `_native.py` called torch_mesh's `backend.Call`, `prepare_writes`, `prepare_fragments` and `plan`: a wrapper over the PyTorch deliverable, not one of the closing set (typed MPS programs, FX/torch.export PyTorch, LiteRT on Metal, a non-transformer demo). Calling its deletion the operator's decision was wrong. |
| `mesh_mpi.Write_pages_prepare` | ≈25 py | No caller in either repo, xonotic included (verified). The library entry it wraps, `ncclMeshWritePagesPrepare`, stays: it is the page-mask half of nccl.h's page-table + page-mask WRITE contract, the integration's named interface, and it already lowers through the layout preparer's `write_prepare`. Deleted 2026-10-30 (11 lines). |
| External `ncclMeshComplete`, `ncclMeshStream.issue` | (see B') | No frontend sets `issue` or calls `ncclMeshComplete` (verified). Deleted with B', 014471ee. |
| rdma/mesh_simulate.py | 620 | Deleted, 17e3d812. |
| `ncclMeshStream.workspace` and `.context`, torch_mesh's `workspace`, `scratch`, `holders`, `retire`, `ncclMeshRetired` | 36 | Nothing set the workspace; nothing called `retire`. Deleted, 8cf035d4. |

## The library's forms (rdma/nccl-mesh.c, nccl-mesh-metal.m)

| | Forms | Fork | Lines one side owns | Reached by | Canon | Deleting the other requires |
|---|---|---|---|---|---|---|
| A | Eager re-binding at issue vs declared bindings | `prepared` 5329-5330 (skips WRITEs, undeclared, float64); `issue_bind` 3742 `issue_same` | ≈160 eager-only (`issue_same`, `bound_of`, `binding_contains`, view and segment copies, `operands_check`, `room_of`, `bind_views`) | WRITEs; declared slot instances other than 0; view calls; a third concurrent issue | Declared: pointer-lowering-2026-10-28.md:3-4; eager-execution.md:171-176 lists `issue_bind`'s re-place as what goes | `prepared` lays every slot instance (the period `packets_prepare` already computes) and WRITEs; issues index the prepared generation by holder and invocation; ncclMeshCall gains views |
| B | Host path vs stream | `launch_one` 4869 `list[0].stream`; `ncclGroupEnd` 4931 | ≈115-125 (`host_commit`, `host_finish`, `gpu_finished`, `comm_issued`, host group loop, device-scalar PreMulSum, host-address WRITE plans) | rdma/mesh_mpi.py alone (every call `ffi.NULL`), so xonotic | NCCL.md:246 "functional, not the measured path"; eager-execution.md:182 keeps the executors | mesh_mpi's calls on a library-owned Metal 4 queue; ncclScalarDevice and the synchronous wait move with it |
| B' | Issue-then-complete and multi-communicator groups vs one communicator completed at once | 4749 `issue && !g->fused`; `several` 4918 | ≈76 | No external caller (verified); the multi-communicator path is its only user | — | Confirm no caller groups calls of two communicators (UNSURE) |
| C | Classic Metal vs Metal 4 program | 4617 `metal4 = list[0].table != NULL`; .m `encoder4` | ≈60 classic (`metal_begin`, `close_encoder`, classic `dispatch`, `metal_uses`, classic tracks) | B's command buffers and B''s `on` stream only: every frontend's ncclMeshStream has an argument table (verified) | Neither (nccl.h 144-155 and NCCL.md:271-282 describe both) | B moved to a library Metal 4 queue; command-buffer status and event signal (`metal_ended`, ncclCommDestroy 1577) rebuilt on Metal 4 commit feedback |
| D | Fused crossings vs separate publish and consume | 4637 `fused_kind` | ≈175 fused | fused: `_stream.mm` 1200/1851, mesh_rank.m 912/999, dnn.py, partition.py, chain.py; non-fused: every combine (`metal_land`) | Both (whole-function.md:130-131) | Neither can go: only non-fused combines; fused carries every caller-kernel crossing |
| E | One-track serial completion vs tracks | 4268-4269 `one_track`, `lone` 4684 | ≈62 one-track (`lone_of`, `one_track`, `complete_track`); ≈300 tracks | schedules | Tracks: the operator's 10-20 choice of per-channel progress (NCCL.md:360) | One-track folds into tracks with T = 1; the stream-order fallback and its cycle failure (4167-4169) go |
| F | WRITE preparers and definition builders | Layout 6321 and Pages 6314 (nccl.h's two contract forms; one `write_prepare` beneath); `ncclMeshCommPlan` 5456-5468 vs `write_definitions` 5880-5912 | ≈48 (`write_definitions`, `cache_drop`) | `group_at` returns a group whenever a plan is installed (verified), so a plan's own definitions serve only a communicator with no plan | One lowering, pointer-lowering-2026-10-28.md:52-62; duplication-review #14 | One `definition_prepare` for both builders |
| G | Per-rank view calls (`*Buffers`, `send_views`/`receive_views`) vs contiguous calls | `views != NULL` | ≈80 | `_stream.mm`, one call each | — | Eager by construction: ncclMeshCall has no views field; goes with A |
| H | Read in place (`plan_direct`, `complement_copy`) vs operand copy or premultiply | 4401 | ≈110 in place | both documented (nccl.h 232-236) | Both | Neither: copy is required for premultiplied, merge, fused and view calls |
| I | Two template flatteners: `plan_declaration` and `CommPlanDeclaration`/`Declared` vs the WRITE plan's | — | ≈100 + ≈105 | `_stream.mm`, metal-recording.m | replay-templates-2026-10-29.md R5 names both | One flattener over one declared-call record |

What the library can lose by form: A + G ≈ 240 (with a prepared-generation index), B + B' + C ≈ 260 (with a library
Metal 4 queue for mesh_mpi), E's one-track ≈ 62, F ≈ 48, I ≈ 100. That is about 710 of
nccl-mesh.c's 6,710 lines (11%). D and H are not alternatives: each side does work the other cannot.

### Classic Metal and Metal 4

Both are Apple's Metal. The classic model (MTLCommandQueue, MTLCommandBuffer, per-dispatch setBuffer and setBytes,
automatic hazard tracking and residency) is not deprecated; MPS, MPSGraph and PyTorch's MPS backend encode only onto
it. Metal 4 (macOS 26, Apple silicon: MTL4CommandQueue, reusable MTL4CommandBuffer with explicit allocators,
argument tables, residency sets, explicit barriers and events, batched commits) runs the same kernels at the same GPU
speed and costs less CPU to encode; no measured A/B of the two forms is in either repository. Every binary here is
built with one floor, macOS 26.0 (etc/compute/metal.mk), so the classic form reaches no host the Metal 4 form does not.
The library began on classic (bd57f73b, 09-28) and added Metal 4 for recorded replays (2bf5a50d, 10-15). In the
recorder, the classic Serial form refuses crossings (metal-recording.m:1971); in the library, classic still runs
every host-path collective (B). What classic is needed for is the boundary with MPS and torch's MPS stream.

## The frontends

Five frontends lower into libnccl-mesh: torch_mesh (`_stream.mm`, backend.py, partition.py, fused.py, chain.py,
dnn.py), rdma/mesh_mpi.py (NumPy; xonotic's planner and solver import it), metal-microbench's engine (mesh_rank.m, its
Swift through mesh_rank.h), the recorder (metal-recording.m) and mesh-tensor-layout.m. torch_mesh and the engine both
declare plans (ncclMeshCommPlan) and issue the per-call NCCL entry points inside a Metal 4 crossing; mesh_mpi alone
runs on the host path.

The same job, implemented in two or more of them (S = torch_mesh/_stream.mm):

| | Job | Copies | ≈Lines |
|---|---|---|---|
| D1 | Collective kind → nccl entry point | S:238-326; mesh_rank.m:343-374; mesh_mpi:306-366 | 180 |
| D2-D5 | ncclMeshCall/PlanRun marshalling, producer times, plan demands and route units, route weights | backend.py:181-273; assignment.py:8-60; routing.py:159-181; mesh_rank.m:785-1044 | 340 |
| D6-D9 | The pair loop: own, landing or relay; relay sizing after WriteForwards; page table, ncclMeshLayout and LayoutPrepare; GroupStart, WriteExecute, GroupEnd | S four times (1351-1377, 1489-1516, 1688-1702, 1925-1949); mesh_rank.m:842-916; backend.py:343-358; partition.py:831-883 | 300 |
| D10-D11 | Roles → logical domains; the cut union across ranks and traversal fragmenting | collective_views.py:14-133; mesh_rank.m:377-393, 676-731; exchanged by torch's store (backend.py:293-309) and by ncclMeshCommConfigGather (mesh_rank.m:929-941) | 180 |
| D12-D15 | Communicator lifecycle; dtype → ncclDataType_t; reduce-op table; tensor or array → address | backend.py, mesh_mpi.py, S, mesh_rank.m, engines.py | 200 |
| D16 | Channel count | mesh_rank.m:446-453 counts cables by hand; S asks ncclMeshCommChannels | 8 |
| D17 | FFI to libnccl-mesh | cffi `_mesh_c` from the headers (it binds 155 functions, 63 named in Python); hand ctypes prototypes in mesh_relation.py; pybind11 in S; the Swift bridging header | — |
| D18-D19 | Template declarations; replay dispatch per form | S:331-385, 1138-1187; metal-recording.m:1318-1370 (classic, Metal 4 and indirect dispatch copies) | 150 |

The two Metal frontends hold one lowering twice, about 1,000 lines in all. It belongs once, beneath both: in the
library (a pair loop over a declared WRITE group, relays the library's, the layout given as ncclMeshLayout), or in one
Objective-C unit both link.

## Stale statements found

- nccl-mesh.c 4746-4748 says a fused crossing completes through `complete_track`; it completes in `complete` (4215).
- nccl-mesh.c 3007 names `plan_calls` for fused records; it has no fused code.
- nccl.h 1102's "first-compiled definition" predates declarations.
- NCCL.md:313-316 cites a classic fused path through torch_mesh/fused.py; fused.py goes through `_stream.mm`'s Metal 4.

## Since: Metal 4 alone (2026-10-30)

The operator, 10-30: "no more non-metal-4 apis ever again. anything that called them is ignored or migrated
immediately." Form C is resolved, and with it the recorder's capture:
- libnccl-mesh has one program form (1c736159). Its host path (B) is a submission of the library's own on an
  MTL4CommandQueue, so mesh_mpi.py and xonotic keep their calls.
- torch_mesh's dispatches run on its Metal 4 queue, and torch's queue is ordered through torch's own events (fba41159).
- The recorder's classic Serial form, MetalScheduleCommand, MetalReplayEncode and the recording-device capture are
  deleted (metal-recording.m 2,741 to 2,279 lines). A replay is built from descriptions alone (MetalReplayDescribed,
  2cdae155), and metal-microbench's engine and rank engine append to Metal 4 schedules (73f39fd5).
- Ignored: the standalone classic benchmarks and tools (c446ba94). LiteRT provider stages, which LiteRT's Metal
  accelerator encodes itself and which were captured only through the classic proxies, are refused at preparation
  until they are lowered to native stages (T8, LiteRT's operator tables).

## Since: the relays, B' and two lifetimes (2026-10-30)

- D6-D8, the relay half of the pair loop, is the library's (mesh 734a76e3, metal-microbench 63eccb19). On a rank
  that ends neither side, ncclMeshWriteLayoutPrepare over an empty table leases the relay where a route crosses the
  rank, and the plan holds it. torch_mesh's four relay leases, mb's columns_prepare relay, ncclMeshCommWriteForwards,
  and partition.py's relay on every bystander, forwarding or not, are gone. D9, GroupStart, WriteExecute and GroupEnd
  around a crossing's plans, is still a few lines at each site.
- D16: MeshRankChannels is ncclMeshCommChannels (63eccb19).
- B' is deleted (014471ee): ncclMeshIssue, ncclMeshComplete, the loss records, the several-communicator parts. A
  group names one communicator and completes at once.
- Two defects found on the way, both lifetimes. ScheduleBinding::writes destroyed its plans as soon as their crossing
  was encoded, while the schedule's entries name the plans' span tables. The plans are now held with the schedule.
  MeshRankDetach destroyed WRITE plans after ncclCommDestroy, and a plan's drop reads its communicator; the plans now
  go first.
- F's other side: partition.follow's exact-byte migrations (_follow_spans, _follow_bytes) ran WRITE groups with
  nothing declared. definition_of ends the program on any group but a lone unfused WRITE. Both now declare their
  groups for the transition, as _scatter_spans does (70b5666e). write_definitions still serves a lone unfused WRITE on
  a communicator with no plan.
- E is done (2026-10-30). The tracks' cost was their width, not their pick. One thread group a track ran every step
  of that channel on one GPU core, where a dispatch of the same step spreads over the whole grid. The tracks are now
  one pool of workers that claim chunks of any track's ready task (rdma/NCCL.md, Per-channel progress): one form for
  any number of channels, a single channel at a dispatch's width with no device barrier between its steps, and no
  worker waiting inside a step. `complete_track`, `one_track`, `lone_of`, `publishable`, the steps' `ready` and their
  `done` flags are deleted. Keeping E, as an earlier version of this entry did, ranked the slow form above the fast one.
- I is two of four flatten-and-parse pairs of one shape: `plan_declaration` and `ncclMeshCommPlanDeclared`, the WRITE
  plan's `write_declare` and `ncclMeshWritePlanDeclared`, and `_stream.mm`'s `declared_calls` and `declared_plans` with
  their writers. Each is a head, arrays at aligned offsets and typed places for the recorder to relocate. One record
  codec, a field table a kind, replaces them. D10-D11: cover and traversal are shared already (mesh-tensor-layout's
  `MetalTensorCover` and `MetalTensorTraversal`, which collective_views.py reaches through `_stream`, cc1ece4b). What is
  left twice is each collective kind's roles as intervals, about 20 lines each (collective_views.py `_domains`,
  mesh_rank.m `cross_domains`). `ncclMeshCommForms` had no caller, though
  440ac872 restored it as the configuration's way to name collective forms (rubric C3: the form an operand). It is
  wired now: a configuration's `forms` reach it in torch_mesh's `configuration.bind` and mb's `install_forms`
  (rdma/NCCL.md, Forms as configuration). Calling that the operator's call was wrong; nccl.h already said the
  configuration document decides a form.

## Since: F, and what the frontends still hold (2026-10-30)

- F (mesh f48ded9b). `definition_prepare` builds both a declared group's definition and a WRITE plan's own:
  `compile_group`, a slot each generation where the group crosses, then `prepared`.
- D, what is left. D6-D8 and D16 are done (above). D9 is each site's `ncclGroupStart`, `ncclMeshWriteExecute` and
  `ncclGroupEnd`, three lines. A helper over the page table, `ncclMeshLayout` and `ncclMeshWriteLayoutPrepare` would
  only wrap that call. D1, D2-D5, D12-D15 and D17 are each frontend's own binding of the NCCL API in its own language
  (C1 keeps NCCL's API as the interface). D10-D11: above.

## Since: A and G (2026-10-30)

- A, for collectives (mesh 4ae7ed5f, metal-microbench 2cb4bff4). `prepared` lays and places one issue for each
  generation and each slot of a declared group's bindings, the least common multiple of their slot counts, and
  `packets_prepare` joins every issue's cuts. An issue takes the prepared issue of its generation and its
  invocation's slot (`ncclMeshStream.invocation`, new). Its calls keep what they were laid for; the issue gives only
  its stream's context, its invocation and a premultiplier (a lone WRITE's prepared calls hold its plan already). The buffers an issue names are not read,
  and torch_mesh's `encoded` and mesh_rank's `issue` name none. A WRITE plan's own definitions are prepared the same
  way, as the plan is prepared.
- Deleted: `issue_same`, `bound_of`, `binding_contains`, `struct bound`, the issue's segment copies, `operands_check`
  and `sent_bytes`. nccl-mesh.c is 6,418 lines (-141 net).
- G is deleted rather than declared. torch_mesh's only `Collectives` come from `prepare_fragments`, one tensor a side,
  so its `*Buffers` branches were never reached, and no other caller exists. Gone: `ncclMeshAllGatherBuffers`,
  `ncclMeshReduceScatterBuffers`, `ncclMeshAlltoAllBuffers`, `bind_views`, the view rooms, and `send_views` and
  `receive_views` with every branch on them. A per-rank view form returns, when a caller needs one, as
  `ncclMeshCall` views prepared like bindings.
- A declared group of both WRITEs and collectives is now refused at `ncclMeshCommPlan`, and a declared group of
  collectives that binds nothing covers no issue (`definition_of`).
- An issue is never held past its launch (mesh 9171d8f1). `launch_metal` holds `group_lock` from `issue_bind` to its
  return. In that span a group either completes (`complete`, `retire`) or is lost with its communicator
  (`walk_failed` → `inflight_lose`), and every drop of a definition takes the same lock. `held`, `dropped`,
  `issue_let`, `issue_drop` and the inflight record's issue are gone (-25 net; nccl-mesh.c is 6,393 lines).
- Declared groups of WRITEs stay laid at their issue, for the plans the issue names. That issue happens at record
  time: every one is a `ncclMeshWriteExecute` inside a `MetalScheduleCross` block, in torch_mesh's
  `prepare_fused_writes`, its `views` and `writes` template crossings, and `ViewWrites::encode`, and in mb's
  crossing. So laying there adds no work to an execution. Preparing these groups at declaration would need the
  declaration to name each plan, and a template blocks that: it replays the communicator's declaration (its `plan`
  call) before the crossing whose plans `declared_plans` makes again. A declaration that named the plans would
  have to carry and remake them itself, which gives each plan a second owner beside its crossing. That is a new
  mechanism, and it would buy the deletion of one branch of `issue_bind`.

## Order

1. The no-caller set above.
2. The frontends' one lowering (D6-D11), the largest duplicate, beneath both Metal frontends.
3. B, B' and C together: mesh_mpi onto a library-owned Metal 4 queue, then the host path, the issue-then-complete
   path and the classic program form go; MPS remains an ordering point between queues.
4. A and G: prepared generations for every slot instance and WRITE, then issue-time re-binding goes. Done for
   collectives and a WRITE plan's own definitions, and G deleted (above); declared groups of WRITEs stay laid at
   their record-time issue, for the reason above.
5. E's one-track form into tracks; F's and I's single builders.

Steps 1-5 remove roughly 2,500 to 3,000 lines across the two repositories: about 1,025 without callers (most of it python/mesh, deleted 2026-10-30), 710 of the
library's alternative forms, and the frontends' second copy of one lowering. That is the reduction this evidence
supports.
