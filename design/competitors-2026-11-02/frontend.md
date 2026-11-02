<!-- design/page-maps-2026-11-02.md#the-runs-solve-each-choice-and-its-ground: a read-only inventory of a form the run's solve replaces, read by a subagent of the 2026-11-02 session at mesh-worktrees/joint 815b3af9 plus its uncommitted backend.py, collective_views.py and operand.py. It lists call sites to rewrite after the solve exists. It is a claim about the old code, not a requirement, and it decides nothing (design/rubric-actionable.md, procedure 5). -->

I've built a read-only inventory of the frontend, the three example programs and `rdma/assignment.py`. I edited, built and ran nothing.

The joint worktree has uncommitted changes in `backend.py`, `collective_views.py` and `operand.py`; line numbers below are for the working tree as it is now. Paths are relative to `/Users/mdot/dox/mesh-worktrees/joint/` (`rdma/torch-mesh/torch_mesh/` is shortened to `tm/`).

**Most important findings**

1. **Five allocators/schedulers compete with the page assignment.** They are:
   - placement's fixed bindings (G4: a family touching a caller tensor takes that tensor's pages at every invocation);
   - the binding-slot dict plus `step['invocation']`, used only by `tm/chain.py` (Z6–Z8, C2);
   - `tm/chain.py`'s two generations, its single X/Y scratch, and its products cache (Z2–Z4, Z10);
   - Inductor's `can_reuse` (L4);
   - per-site region leases: `layout.allocate`, `placement.output`, `native.empty_strided`, command temporaries, CPU workspaces (H1, G1, L1, Q1, O3).
2. **`assign_pages` does not yet see transport.** `Plan.append` drops transfer steps from `programs` (page_effects.py:501–502), so lifetimes and `frontier_edges` cover only numerical accesses. Lease-backed spaces are skipped (:80), and the pack is called with capacity `max(1, sum(sizes))` and no pins (_stream.mm:1275).
3. **Gaps in the in-progress change:**
   - before/after copy programs of a transfer always get `run_invocation` 0 (D2);
   - `lowering.bind_spaces` is referenced in `backend.py` but does not exist; nothing in the frontend calls `ncclMeshSpaceBind` (C3);
   - `Occurrence` takes `std::vector<at::Tensor>`, so a `TensorValue` collective operand may not bind (unsure, D3);
   - only the first realized schedule imports torch's progress event (J6).
4. **No callers anywhere in the worktree:** `partition._relay` (so `follow` is unreachable), `engines.model_operand`, `_stream.commit`, `_stream.publish_boundary`. `assignment.declare` is called only from `xonotic/`.

Tags: **[RUN]** storage the run reads or writes. **[CONFIG]** configuration-time eager storage that the run later reads. **[TRACED]** a value made under FakeTensor capture; its run storage is decided later by G4/H2/D1. **[COMPILE]** never read by the run.

---

## A. Region allocator (`tm/allocator.mm`, `tm/__init__.py`)

**A1. allocator.mm:76–97 `lease`** — allocator [RUN]
- Decides: every MPS byte request becomes `ncclMemAlloc` + `ncclMeshMemBuffer` + the lease's page runs, stored in `leases_`.
- Values: every MPS tensor in the process, including torch's own temporaries.
- Safety: none at allocation; reuse is the library free set's.
- Lifetime: created at torch's allocate, ends at A2.
- Consumers: every tensor op; `lease_pages` (:240–244) → `_stream.tensor_value` → `TensorValue.of_lease`; `region_allocations` (:229–235) → `torch_mesh.records`.

**A2. allocator.mm:57–74 `release`** — reuse rule [RUN]
- Decides: when a lease's pages return. An `MPSEvent` is recorded on torch's current stream at release, and its notify calls `ncclMemFree`; the stream is committed if off-queue.
- Safety: torch's stream order plus `ncclMemFree` ("pages kept until every DMA read from them ends"). Prepared schedules run on a separate `MTL4CommandQueue`, so leases they use are kept alive only by B6/B4/J6 references.
- Lifetime: last torch-stream use → event → free.
- Consumer: the library free set.

**A3. allocator.mm:108–124 `RegionHostAllocator`** — allocator [RUN]
- Decides: every CPU tensor is a lease, freed synchronously in `release_host` (:99–106).
- Values: CPU weights (O1), CPU workspaces (O3), host scratch (EX7, EX14).
- Safety: none (no event).
- Consumers: CPU and Accelerate kernels, the bridge.

**A4. allocator.mm:167–171 `allocScalarBufferWithValue`** — allocator [RUN]
- One lease per torch scalar buffer, filled by memcpy.

**A5. allocator.mm:185–207 `recordEvents` / `waitForEvents`** — wait [RUN]
- Decides: a host read of a lease waits on an event recorded on torch's stream after the GPU's last use (`(*event)->synchronize()`).
- Consumer: torch's host-alias reads.

**A6. `__init__.py:177–194` `empty` → `_stream.mm:820–826` `region_empty`** — allocator front [RUN; TRACED under fake mode]
- Decides: at least 1 byte, through A1/A3. Under a fake mode it returns a fake `torch.empty`.
- Consumers: everything tagged "lease" below.

## B. `tm/_stream.mm`

**B1. :147–193 `MeshStream`, `stream_of`, `streams_`** — order/wait [RUN]
- Decides: each torch stream gets a `progress` shared event and a counter `committed`. `signalled(wait)` records a torch event whose notify raises `progress` to `++committed`, then does COMMIT or COMMIT_AND_WAIT.
- Consumers: B3; B2 (B2 has no Python caller).

**B2. :206–228 `host_wait`, `commit`, `publish_boundary`** — wait
- `dispatch_sync` onto torch's queue, optionally COMMIT_AND_WAIT.
- No caller in the worktree.

**B3. :594–615 `schedule_input_reaches` / `schedule_inputs`** — wait [RUN]
- Decides: framework-produced input pages become `MetalScheduleInput(…, progress, committed)`. The recorder makes their intersecting consumers wait on torch's event at the value captured at preparation.
- Values: `Execution.inputs`.
- Consumer: the native recorder. Called only on the first `Execution.append` (J6).

**B4. :659–744 `Commands`** — allocator + order [RUN]
- Allocation: all constant words of one Program are packed with `MESH_COMMAND_DATA_ALIGNMENT` into one `region_empty` CPU lease per `Commands` (:673–685).
- Order: replays are cut at host positions ("host positions must follow declared command order", :724). `append` interleaves replay i, then host i (:735–743).
- Lifetime: held in `Commands.operands` until the object dies.
- Consumer: `MetalScheduleAppend`.

**B5. :748–777 `Occurrence`** — invocation numbering / phase key [COMPILE → RUN]
- `lower(invocation)` → `ncclMeshCommLower(comm, invocation, polls…)`. On the C side this takes the plan's next group (`definition_at(comm, comm->groups+1)`; the cursor advances in `lower_group`, nccl-mesh.c:5799) and sets `calls[i].invocation = invocation`.
- `schema()` returns `{(uint64_t)(uintptr_t)comm, source->definition, bindings…}`. The phase key therefore holds a pointer plus the binding phase (`u->instance` and the coverage slot indices, nccl-mesh.c:5850–5855).
- Consumer: D3/D4.

**B6. :870–922 `ScheduleBinding`** — pool (lifetime holder) [RUN]
- `operands` (tensors) and `prepared` (Occurrences) are retained until the `MetalScheduleFinish` completion block.
- `copy` / `compute` append commands in call order.

**B7. :925–958 `PageEffectPages::page/storage`** — cache [RUN]
- CPU `from_blob` views over native word buffers, deduplicated by containing range in a per-call `storages` vector.
- Values: publication and result words (:1004, :1177).
- Consumer: `schedule.results` (D7).

**B8. :1033–1181 `PageEffectsPlan::bind`** — order + cache [COMPILE → RUN]
- Decides command order. Commands go into `std::map<NSUInteger, Block> blocks` keyed by description `order` (:1070).
- An occurrence's commands and transfers go to `frames.at(frame)` (:1113, :1119).
- `MetalPageEffectsOrder(…, &next)` orders within each block, blocks in key order (:1123–1127).
- The first call solves and keeps `declaration`; later calls `Rebind` (:1129–1132).
- Consumers: `ncclMeshCommSourceInputs`/`Specialize` (:1170–1173); D7.

**B9. :1185–1204 `Executions`** — order/schedule [RUN]
- FIFO of realized schedules. `add` pops ended ones by polling `MetalScheduleStatus`.
- `run()`: `latest=schedules[next++]` — each schedule runs once. There is no bounds check (unsure what an extra run does).
- Consumers: `Execution.forward` (J6), lstm-service.

**B10. :1217–1231 `completion_observe`** — wait-poll [RUN]
- Non-blocking `MetalScheduleEnd` status.
- Consumer: `mesh_statistics` (`rdma/mesh_statistics.py:252`), which the examples poll (EX13).

**B11. :1269–1282 `interval_pack`** — the target allocator's binding [COMPILE]
- Calls `mesh_graph_interval_pack(n, sizes, begins, ends, nullptr /*pins*/, capacity, place, settled)`.
- Consumer: D1.

**B12. :300–433 `pipelines_`, `libraries_`, `batches_`, `state_of`** — cache + wait [COMPILE]
- Pipeline caches. `state_of` waits on a semaphore up to `MESH_PREPARATION_NS`. Not storage.

## C. `tm/backend.py`

**C1. :109–115 `_binding`** — produces `(MTLBuffer, storage_offset·size)` or `(NULL, data_ptr)` [COMPILE]
- Consumers: C2, `'source_ready'` (:346).

**C2. :133–143 `_slots`** — slot construct (producer of `instances`, `stride`, `invocation`) [COMPILE]
- A `TensorValue` gives `{'buffer':(NULL, first_byte), 'instances':1, 'stride':0, 'invocation':0, 'space':handle}`.
- Otherwise it copies `record.get('instances',1)`, `('stride',0)` and `('invocation',0)`.
- Consumer: `ncclMeshBufferSlots[2]` (:333), coverage `active`/`expected` (:340).

**C3. :118–130 `_SPACES` / `space_handle`** — cache + capacity [COMPILE]
- One `ncclMeshSpaceCreate(serial, pages*page_bytes)` per ValueSpace serial, cached forever.
- The comment names `lowering.bind_spaces`, which does not exist. No frontend caller of `ncclMeshSpaceBind`.

**C4. :258–361 `ProcessGroupMesh.plan`** — order/schedule + timing [COMPILE]
- Default order is one `ncclMeshPlanRun{first:g, count:1, repeat}` per group (:272–273).
- `producer_groups` counts producer repeats (:276–280). `ncclMeshProductionTimes(production, n, seconds/producer_groups)` (:311–313) — "Predictions order physical descriptors".
- Views merge `{'tensor': storage, **slot dict}` (:325) before C2.
- Consumer: `ncclMeshCommPlan`, which the C side uses as issue order.

**C5. :372–385 `configuration`** — order
- Temporarily replaces the plan with configuration groups, then re-plans the previous one.
- Consumers: `count_matrices`, partition N4/N7.

**C6. :400–421 `prepare_writes`** — slot/stride [COMPILE]
- Per-call `frame_stride` (scalar units) is appended to the view geometry.
- Consumers: C4 (`frame_stride`), D3 (`view_write_operands`).

**C7. :431–437 `union_cuts`** — order/fragmentation [CONFIG]
- Exchanges and unions view cuts across ranks.
- Consumer: I2.

**C8. :440–461 `prepare_fragments`** — order + in-place binding [COMPILE]
- The transfer step carries `before`/`after` copy commands and `bindings=(source, destination)`. Destinations are the caller's own value or view (landing in place).
- Consumers: D2/D3.

## D. `tm/page_effects.py`

**D1. :55–107 `assign_pages`** — the target allocator [COMPILE → RUN]
- Commands are sorted by `(order, i)`. Edges come from `frontier_edges` over accesses that have a root and spans.
- Items:
  - `('space', root, invocation)`;
  - `('state', state, inv or inv+1)` for state versions;
  - (state, min) and (state, max) are pinned to the caller lease (:92–97);
  - lease spaces are skipped (:80).
- Reuse: only when `last in ancestors(first)` (`settled`).
- Gaps:
  - transfer steps are absent from `descriptions` (programs exclude `'transfer'`, :501–502), so landings and SEND reads are not lifetimes;
  - capacity `max(1, sum(sizes))` means it never runs out;
  - pins are not passed to the pack.
- Consumer: E1.

**D2. :459–479 `Plan.__init__`** — order [COMPILE]
- A transfer step becomes before / step / after declarations.
- Order key = 2·index. Frame slots are odd (`origins[...]+1`).
- The before/after dicts (:466–470) carry no `run_invocation`, so `invocation_of` gives them 0.

**D3. :485–494 `Plan.append` transfer loop** — invocation consumer [COMPILE]
- `bound.lower(step.get('invocation', 0))` consumes C-side plan positions in declaration order.
- Frames map to `origins[...]+1`. `phase` is built from `bound.schema()` (B5).
- `_stream.Occurrence(handle, step['operands'])` takes tensors; a `TensorValue` operand may fail here (unsure).

**D4. :495–520 `self.variants`; `emitted` keyed by `id(step[...])`; `describe` cache (:347–351)** — cache [COMPILE]
- One emission and one `PageEffectsPlan` per binding phase. Repeated identical step objects share one emission.

**D5. :522–533 `self.assignments`, `self.arenas`** — allocator (target) [RUN]
- A new `assign_pages` and a new arena lease (E1) per `append`. Every arena is retained in `Plan.arenas`, never freed while the Plan lives.

**D6. :535–542** — order [RUN]
- `sorted(prepared, key=order)` → `command.append(schedule[, selected])` is the schedule's command order.

**D7. :408–435 `bind_effects`** — storage + wait [RUN]
- Appends the native-allocated metadata page view as an operand. `schedule.results(solved['results'])` gives words with thresholds (completion waits in the recorder).

## E. `tm/lowering.py` (target; decides nothing)

**E1. :54–65 `lower_programs`** — allocator (target) [RUN]
- One arena lease, `max(1, arena)*page_bytes`, per append. Operands are specialized at `invocation_of(order)`.
- Values: every value-space operand.

**E2. :13–51 `_LEASES` / `_PINS`, `lower_value`** — reuse rule [RUN]
- Lease-backed spaces and pinned state versions always resolve to the caller's whole storage at offset 0 plus layout, i.e. the same pages at every invocation.
- Registered by F2 and G5.

## F. `tm/operand.py`

**F1. :13–36 `ValueSpace` serial (`itertools.count(1)`)** — identity numbering [COMPILE]
- Consumers: D1 roots, C3, E2.

**F2. :50–68 `of_lease` / `virtual`** — capacity [COMPILE]
- `pages = ceil(extent/page_bytes)`. A lease space is a constant map, registered with E2.

**F3. :602–631 `signature`** — reuse rule [TRACED]
- Outputs aliasing one storage become one result handle, the storage root.
- Consumers: matrix/host/kernel custom ops.

## G. `tm/placement.py`

**G1. :6–13 `output`** — allocator [RUN]
- Region lease when `out is None`.
- Callers: `native.linears` :160, `kernel.prepare` :77, argmax :152/:165, selection :191, scan :89, structured :263.

**G2. :42–55 `allocate_value`** — capacity [COMPILE]
- Extent rounding and a page-alignment check give a new virtual space.
- Consumers: G4, H2.

**G3. :86–141 `Placements.declare`** — reuse rule [COMPILE]
- `unite`/`identify` merges equated uses into one family, so equated values share one space.
- Equalities come from host (U1), field_storage, collective equalities, page_equalities (M2) and states.

**G4. :148–253 `Placements.solve`** — allocator/placement (competitor) [COMPILE → RUN]
- A family without fixed operands gets G2 and a new space.
- A family with a fixed caller operand gets `set_` over the caller's whole storage → `TensorValue.of_lease` (:246–248): a constant map at every invocation. This is the in-place / fixed-binding path.
- Consumers: J10, L (emit).

**G5. :321–466 `selected`** — reuse rule + order [COMPILE]
- Fake roots (:377–379, compile-only).
- State-tagged operands get `space.state/pins`; `register_pins` (:440–448).
- `resolve` (H2) adds `before` copies for reads and `after` for writes or readwrite (:451–458). `owners` keeps the copies' storage.

## H. `tm/layout.py`

**H1. :178–193 `allocate`** — allocator (competitor) [RUN]
- `empty((reserved,))` padded so that `base + offset + bound + origin` is aligned (reads the lease's address through `page_backing`).
- Callers: H2, command_declarations :125, engines :244/:597.

**H2. :196–243 `resolve`** — reuse rule [COMPILE]
- If the operand fits it is used in place. Otherwise a new value (G2) or lease (H1) plus copy edges (before for reads, after for writes).

**H3. :131–175 `copy_declarations`** — capacity + reuse
- Copies are tiled by `LAYOUT_COPY_TILE_BYTES` and elided when `same_view`.

## I. `tm/collective_views.py`

**I1. :158–191 `_binding`** — reuse rule [COMPILE]
- The collective binds directly in the caller's value or view: in-place landing.
- Record rows split by storage breaks go through H2 with before/after copies.

**I2. :202–272 `lower`** — order/fragmentation [COMPILE]
- Fragments follow call order, then agreed cuts (C7). Frame ends are cuts.
- `before` copies precede and `after` copies follow; coverage `before` is remapped.

## J. `tm/declarations.py`

**J1. :15–246 `SourceCapture`; :257/:269; :319/:326** — fakes [COMPILE/TRACED]

**J2. :395–412 `spans`** — order [COMPILE]
- `mesh_graph.needed` eliminates dead nodes in FX order.

**J3. :623–666 `lower(graph)`** — order [COMPILE]
- Repeat trains merge consecutive identical WRITE calls into `(calls, count)`.

**J4. :854–857 `boundary_projection`** — reuse rule [COMPILE]
- An `out=` collective into the section's own operand "is an effect on that operand: its pages are the result's". This is the in-place h.

**J5. :967–1017 `Execution`** — order + numbering [RUN]
- `run_invocation = enumerate(self.steps)` (:993–1002).
- `schedule.inputs` only while `not _initialized` (:1005–1006), so later schedules import no torch events.
- `realize(n)` makes n schedules (:1010–1016). `forward = executions.run`.
- `pages` hold lease lifetimes.

**J6. :1040–1069 `normalize_edges`** — order
- Deletes `wait_tensor` nodes; ordering is left to native binding.

**J7. :1078–1124 `sections`** — order
- Section order is FX node order, cut at bound nodes.

**J8. :1255–1290 `LoweredGraph.__init__`** — numbering + capacity
- Occurrence relocation `offset += first` (:1267–1281). `reserve_bindings(int(have_transfers))` (:1305–1309) feeds L3.

**J9. :1352–1481 `bind_section`** — placement (competitor) [COMPILE]
- Inputs with held pages become `fixed` to the caller's tensor (:1414–1416).
- Outputs go to `final_targets` when relocatable (:1392–1406).
- An input aliasing an output's final target becomes a state (:1371–1382, :1407–1420).
- Consumer: G5.

**J10. :1483–1565** — order + allocator [RUN]
- Steps are appended in section order; host steps as before / made / after (:1516–1522).
- Zero-size `native.empty_strided((0,))` source (:1531).
- Final `copy_declarations` into destinations appended last (:1556–1564).

## K. `tm/compilation.py`

**K1. :413–455 `repeat_source`** — numbering + reuse [TRACED]
- Unrolls `count` occurrences (`offset + first + at*extent`). "Input and captured-state identities remain shared."

**K2. :483–548 `functionalize_source`** — reuse rule
- `propagate_input_mutations=meta.get('mesh_input_writeback', True)` makes state writes final writebacks into the caller input.

**K3. :241–283 `lift_state`; :102–108 `fold_occurrence` ordinal; :44–89 `bind`** — compiler operands (`BUFFER_BINDINGS`, `LAYOUT_COPY_TILE_BYTES`, …).

## L. `tm/native.py`

**L1. :34–38 `empty_strided`** — allocator [RUN]
- Lease of the view's extent. Callers: `kernel.prepare` :76, declarations :1531.

**L2. :681–698 `reduction_split_factor`; :1869–1954 `split_reduction`** — capacity [COMPILE → RUN]
- Split stages (cores × `REDUCTION_GROUPS_PER_CORE`, `REDUCTION_TILE_ELEMENTS`) create intermediate stage buffers. They are added as `'readwrite'` uses; G5 places them.

**L3. :1497–1500, :2394–2401** — capacity bound
- `max_fusion_unique_io_buffers = BUFFER_BINDINGS - binding_reserve`.

**L4. :1038–1049 `make_allocation` / `can_reuse`; :2415 `never_reuse_buffers`** — reuse rule (competitor) [COMPILE → RUN]
- Inductor's buffer reuse stays enabled except for bound or never-reuse names. An unbound output can only get storage by reusing a dead input's buffer.

**L5. :1525–1529 `storage_aliases`** — reuse rule
- Mutation outputs alias their source storage.

**L6. :1987–1989, :2160–2174 `virtual_values`** — decides no storage for scalar-read values.

**L7. :2176–2197, :2262–2286 `page_aliases` → `removed_ops`** — reuse rule
- Identity copies are dropped when the placed pages are the same view.

**L8. :1256–1300 `can_fuse` / `fuse`** — order
- "A fused cell runs its left producer before its right consumer"; refused on read/write overlaps.

**L9. :263–275 `Program.prepare`** — order
- Host commands are spliced at `len(recorded)`; `owners` keep host programs alive.

## M. `tm/ownership.py`

**M1. :342–443 `prepare_expression`** — order + cache
- Cuts at bound and host nodes and materializations; result cached in `expression['prepared']`.

**M2. :506–527 `page_equalities`** — reuse rule
- A fresh received value is co-placed with its sole identity consumer's unfixed result.

**M3. :923–1075 `compile_assignments`** — order
- "in producer order"; deliveries are routed per section as appended.
- Fakes at :212 and :840 are compile-only.

## N. `tm/partition.py`

**N1. :49–65 `_Operand.cache`; :1537–1544 `_positions`** — cache [RUN]
- Real index tensors (`arange(...).to(device)`, leases), cached per key forever.

**N2. :83–87 `_empty`** — lease helper.

**N3. :96–105 `allgatherv_out` (`mutates_args=('out',)`)** — reuse rule
- Declares an in-place landing.

**N4. :204–238 `_shard_tensor`** — allocator [CONFIG → RUN]
- `out` and `staged` leases, then eager `dist.scatter`. `staged` is a staging copy.

**N5. :304–335 `_scatter_spans`; N6. :338–389 `_follow_spans`** — allocator + order [CONFIG]
- Result lease, or a narrow view in place.
- Each builds its own `_stream.Schedule()`, then `finish` and `enqueue`. No wait before torch's next reader is visible (unsure how order holds, e.g. for `lstm.Tick.held`).

**N7. :393–441 `follow`** — staging x/y + new lease, eager `all_to_all_single` [CONFIG]
- Reachable only through `_relay`, which has no caller.

**N8. :447–483 `_SPLIT` / `_relay`** — reuse rule
- Re-lays registered tensors in place. No caller of `_relay`.

**N9. :464–473 `empty(mesh, …)`** — lease of this rank's share, registered (N8).

**N10. :494–501 `plan`** — order passthrough to C4.

**N11. :1615–1634 `_block` (`SCORES = 1<<26`)** — capacity bound [TRACED]
- Bounds the score tile.

**N12. :1053–1129 `_transition_sources`** — order + renaming
- Sorts transitions by (consumer order, side, order); makes distinct version ports.
- Traced or fake allocations elsewhere in this file: :253–259, :573–577, :1002.

## O. `tm/engines.py`

**O1. :203–212 `Cpu.prepare_linear` / `prepare_ffn`** — allocator [RUN]
- fp32 CPU weight copies (leases).

**O2. :234–244 `declare_views`** — allocator [RUN]
- `layout.allocate` scratch for non-role recipe bindings.

**O3. :455–467 `cpu_engine.workspace`** — allocator [RUN]
- One lease per Accelerate workspace, declared as write reaches.

**O4. :589–598 `model_operand`** — `layout.allocate`; no caller in the worktree.

**O5. :694–790 `page_state_program`** — reuse rule
- Exported state-mutation output/input pairs denote the same pages.
- Traced allocations: :325, :329, :890, :1103, :1115.

## P–W. Other modules

**P1. matrix.py :159–193 `products`; :95–104 `product`** — capacity + order
- Grouping is bounded by `BUFFER_BINDINGS` and `LINEAR_ROW_CROSSOVER`. Order is before / bind / after.

**Q1. command_declarations.py :54–56 `temporary`, :113–151 `bind`** — allocator + reuse [RUN]
- Each emission leases its temporaries through H1; `initializer` is filled once at preparation by host `fill_` (:126–129).
- Users:
  - argmax.py :116–117: `draw/parts`, `draw/arrivals` with initializer 0. Whether it is reset between replays is unsure.
  - scan.py :40–72: `totals`/`partial`/`prefix`. At :43–47 `partial` reuses the target's pages.
  - selection.py :140: `topk/scratch`.

**R1. kernel.py :66–90 `prepare`** — L1 output plus H2 copies (inputs before, outputs after).

**S1. streamed.py :644–645 `_half`, :1030/:1046 `partials[shares,…]`** — capacity [TRACED]
- The split-K shares are chosen by the fit.

**S2. streamed.py :958–972 `coded_input.emit`** — capacity + reuse
- Batches by binding capacity and pads unused output slots by aliasing `outs[0]`.

**S3. streamed.py :1757–1765 `prepare_read`** — its own Schedule with `repetitions` computes (used by examples/memory-read.py).

**T1. reduction.py :70–131** — order
- `reassociate`/`reorder` flags and `fold_topology` choose the combine tree, cached in `fold_choices`.

**U1. host.py :39–52 `assemble`, :71–83 `equalities`** — reuse rule
- Output aliases share bytes; feeds G3.

**V1. dnn.py :15–72** — leases for input, output and weights, `ag`/`bg` copies per declare. One declaration per tile shape, reused through `prepared`, with `occurrence=(first//tile_rows, n)`.

**W1. sparse.py :95–101** — `product_schedule` tile order as CPU lease tables read by the kernel.

## X–Y. `tm/configuration.py`, `rdma/assignment.py`

**X1. configuration.py :19–86 `placement`** — chooses `steps_per_replay`, `input_frames`, `resident_tiles` and parts through `assignment.assign`; rank 0's result is broadcast; `partition._attach` publishes the parts.

**X2. configuration.py :89–128 `bind`** — `producer_seconds` → `partition.plan(order=…)`.

**Y1. assignment.py :9–46 `declare`** — producer of slot field `'instances': 1` (stride and invocation are cffi zero) plus a default order. Called only from `xonotic/`.

**Y2. assignment.py :346–348 `window`** — observation capacity bound. **Y3. :351–579 `assign`** — capacity operands from observations. **Y4. :582–593 `producer_seconds`** — timing → C4.

## Z. `tm/chain.py`

- **Z1. :125–132 `weight_pages`** — leases (zeroed, padded, tiled copy) [RUN].
- **Z2. :138 `X`, `Y`** — one scratch lease each, rewritten every step.
  - Safety: recorded command order with `coherent_accesses=True` hazards.
  - Lifetime: phase 0 writes X → phase 1 reads X, writes Y → phase 2 reads Y; the next step overwrites.
- **Z3. :142 `generations = empty(2, batch, size, R, K)`** — a two-slot ring.
  - Window w uses `len(windows)%2`. The gather lands frame j in `generations[g, j]`, read by the next step as `previous`.
  - Safety: two fixed generations; the comment defers to "the common compiler handles any physical alias hazards" (no proof).
- **Z4. :144–155 `windows`, `recurrent`, `input_starts`** — capacity/schedule.
  - `count = min(batch, steps-first, input_frames - first%input_frames)`.
- **Z6. :156, :184–193 `invocations` and `order`** — invocation numbering + order.
  - A per-definition counter becomes `command=dict(source, invocation=…)`. `order.append({first:index, count:1, repeat:frames})`.
  - Consumers: D3, C4.
- **Z7. :195–203 input slots `dict(instances=0, stride=input_frames*rows*width*elem)`** — a linear slot window.
  - Invocation k lands at `inputs[starts[0] + k*input_frames]`. Broadcast is source == destination (in place).
- **Z8. :213–217 recurrent slots `dict(instances=1, stride=0)`** — one slot; `definitions` cache keyed `(count, generation)`.
- **Z9. :219–237 `frame_orders`** — order: frame j's transfer goes after step j's last command (consumed at D3 and B8).
- **Z10. :226–232 `products[(generation, j)]`, `_phase` `sources`** — cache/reuse.
  - The same command object is replayed for every window of a generation (shares D4's emission).
- **Z11. :240–242 `Execution(pages=…, inputs=…)`** — lifetimes; inputs pass through B3.
  - chain.py uses no `TensorValue`, so D1 assigns nothing for it.

## EX. Examples

- **EX1. lstm.py :24–26** — `held`: W/b lease copies of the shard (a second copy after N4/N5).
- **EX2. lstm.py :28, :35–37** — `c`, a single `h` for all layers and ticks, and `mine = h.narrow(2, …)`, a view into h.
- **EX3. lstm.py :41–44 `reset`** — eager `zero_` of c and h.
- **EX4. lstm.py :50–52** — `cat`/`stack` traced values; `cell` writes c and mine in place through `copy_`.
- **EX5. lstm.py :57** — `allgatherv_out(self.mine, self.h, …)`: in-place landing whose source aliases the destination. Safety today: J4, G4, the recorder's position.
- **EX6. lstm.py :66–77 `stepper`** — the device counter `t`:
  - `X.index_select(0, t…)`, `O.index_copy_(0, t-(L-1), …)`, `t.add_(1)`;
  - a run-time counter, not a page map, selects each step's input and output rows.
- **EX7. lstm.py :108–121** — `chain.region(CHUNK)` host scratch reused across draws [COMPILE/setup].
- **EX8. lstm-service.py :27–36, :67–68** — weight and bias leases; one-element `expand` placeholders on non-root ranks.
- **EX9. lstm-service.py :71–73** — X, O and t leases; `t.fill_(L-1)`.
- **EX10. lstm-service.py :75–91** — `steps_per_replay` chunks.
  - One declaration per chunk length (`functions` dict), the same Function object replayed for every chunk; each chunk gets a distinct `run_invocation` (J5).
- **EX11. lstm-service.py :100–108** — consecutive single-group functions merge into `(calls, n)` for C4.
- **EX12. lstm-service.py :113–120** — broadcast of X into X (in place); `execution_sequence([inputs, *ordered], O)` fixes run order and invocation numbers.
- **EX13. lstm-service.py :121–144** — `realize(repeats)`, then per repeat `t.fill_`, `tick.reset()`, `torch.mps.synchronize()` (host wait), `forward()`, and `while not …close(): time.sleep(0)` (poll).
- **EX14. examples/chain.py :11–16, :20–31** — expand placeholders; a fixed `region(CHUNK)` scratch window in `_drawn` [setup]. The runner chain-service.py :70–87 leases E and O, passes `k` and `input_frames` to Z, and uses the same poll.

---

## Producers and readers of the binding-slot fields and `step['invocation']`

**`instances`, `stride`, `invocation`**

| Role | Site |
|---|---|
| Producer | backend.py:139 (`TensorValue`: 1 / 0 / 0) |
| Producer | backend.py:142–143 (dict defaults 1 / 0 / 0) |
| Producer | chain.py:201 (`instances=0`, `stride=…`) |
| Producer | chain.py:217 (`instances=1`, `stride=0`) |
| Producer | assignment.py:31–32 (`instances: 1`) |
| Carrier | chain.py:173 `bindings=(slots, slots)` → backend.py:325 merge |
| Python reader | backend.py:142–143 only |

- No in-scope code builds coverage `active`/`expected` rows; backend.py:340 only reads them.
- C readers: nccl-mesh.c :5854 (source binding phase), :5970–5971 (`coverage_slot`), :6435–6436, :6776–6777 (`zoblimp` shift by `index*stride`), :6802 (`prepared` period), :7717–7730 (write layout); cffi `_mesh_c.c:1667–1668`.

**`step['invocation']`**
- Producer: chain.py:189–191 only.
- Reader: page_effects.py:491 → `Occurrence::lower` → nccl-mesh.c:5917 (`calls[i].invocation`) → :4982 (`instance = invocation % x->period`) → `zoblimp`.
- It is separate from `run_invocation` (declarations.py:989–1002 → page_effects.py:526 → D1/E1).

## Entries that share state

1. **One region lease pool** (A1/A3 `leases_`, `hosts_`): A6, B4, D5/E1, G1, H1, L1, N2, N4–N9, O1–O3, Q1, Z1–Z3, and EX1, 2, 7, 8, 9, 14. Freed by A2; waits A5.
2. **The slot/generation mechanism:** chain Z3/Z6/Z7/Z8 → C4 merge → C2 → C side, and Z6 → D3 → B5. Its binding phase keys D4 and selects the B8 Rebind.
3. **Placement families:** G3 ← U1, M2, L5, L7, field_storage, J4, J9 states → G4 (lease spaces, E2) and G5 pins → D1 pins and E2 `_PINS`. C3 caches per serial (F1).
4. **Plan order versus issue cursor:** C4 order (from J3, EX11, Z6, X2) must equal the order in which D3 lowers (`comm->groups` cursor), which follows J5/J10 step order.
5. **Plan state:** D4 variants, D5 assignments/arenas, J5 `run_invocation`, D2 frame slots, B8 blocks.
6. **Binding capacity `BUFFER_BINDINGS`:** L3 (with J8 reserve), P1, S2.
7. **Assignment operands:** Y3 → X1 → Z4 (`batch`, `input_frames`), Z/`_form` (`resident_tiles`), EX10 (`steps_per_replay`). Parts go to N1/N9 through `partition._attach`.
8. **Waits and polls:** A5, B1, B3, B9, B10, EX13; the poll loop is shared by lstm-service and chain-service.