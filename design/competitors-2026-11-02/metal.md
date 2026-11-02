<!-- design/page-maps-2026-11-02.md#the-runs-solve-each-choice-and-its-ground: a read-only inventory of a form the run's solve replaces, read by a subagent of the 2026-11-02 session at mesh-worktrees/joint 815b3af9 plus its uncommitted backend.py, collective_views.py and operand.py. It lists call sites to rewrite after the solve exists. It is a claim about the old code, not a requirement, and it decides nothing (design/rubric-actionable.md, procedure 5). -->

## Metal-side inventory of allocators, schedulers, slots and waits

All paths are under `/Users/mdot/dox/mesh-worktrees/joint/rdma/`. Where code outside the worktree reads one of these results (in `/Users/mdot/metal-microbench`), the entry says so. Field order in every entry: kind, what it decides, values, how safety holds today, lifetime, consumers.

### metal-recording.h / metal-recording.m / metal-scope-plan.inc

**R1. `metal-recording.m:840-852` `MetalLeaseBuffer` (declared at `.h:32-33`)**
- Kind: allocator.
- Decides: a fresh region lease for each call, rounded up to whole pages (`(MAX(length,1)+page-1)/page*page`, `getpagesize`). It calls `ncclMemAlloc`, wraps the memory with `newBufferWithBytesNoCopy`, and installs a deallocator that calls `ncclMemFree`.
- Values: address tables (R8), command-data pages (R17), page-effect state, threshold, record and index pages (P1), and effect-field tables (P3).
- Safety: there is no reuse. The lease is freed only when the MTLBuffer is deallocated, after its last retain is gone (schedule `owners`, replay owner, residency set, `MetalPageSource` owners).
- Lifetime: allocated at preparation → filled by the host with `memcpy` before submission → read on every execution of the owning function → freed when the last owner releases it (`MetalScheduleFree` 2415-2416, `MetalReplayFree`).
- Consumers: R8, R17, P1, P3. These are its only callers in the tree.

**R2. `.h:27-36` `struct MetalOperand {stride, instances}` and `MetalInvocationOffset`**
- Kind: slot/ring/modulo.
- Decides: the slot an operand's invocation uses: `offset + ncclMeshInvocationIndex(invocation, displacement, instances) * stride`. `ncclMeshInvocationIndex` (`nccl.h:773-780`) is `(t+displacement) % instances`, or linear when `instances` is 0. The header puts it this way: "invocation t binds slot t % instances".
- Values: bindings, embedded address targets, uniforms, and indirect grids of a described program.
- Safety: the modulo index alone. Reuse of slot k at t and at t+instances holds only if the frontier (R4) sees the same bytes.
- Lifetime: the slots are carved out of caller-owned buffers.
- Consumers: R3, R8, R9, R12, R16, R19 and `MetalAccesses`. Outside the worktree: metal-microbench `engine_recording.swift:69`, `litert_program.m`, `mesh_rank.m`.

**R3. `.m:1104-1150` `MetalCaptureOperands` (with `MetalOperandOf` 1095-1098)**
- Kind: slot/modulo.
- Decides: copies each operand's stride and instances onto every binding (1146-1147), onto each address target and its uniform (1114-1120), and onto the indirect grid (1109-1110). When a buffer is declared as more than one operand, the last declaration wins (1096). A reach into a multi-slot operand that is not a binding is refused (1125-1130).
- Values: `strides[]`, `instances[]` and `invocations[]` of each recorded dispatch and address record.
- Safety: refusal at preparation.
- Lifetime: not applicable.
- Consumers: the R2 readers. It is called with `(NULL,0)` for collective declarations (2037), so those strides are all 0.

**R4. `.m:619-719` byte-span frontier (`MetalSpanAdd/Erase`, `MetalViewPrior/Define/Step`; buckets of 2^`MESH_FRONTIER_BITS`=2^20 bytes, line 624)**
- Kind: order/fence/schedule. This is the recorder's own frontier, the one to be replaced.
- Decides: dependency edges by byte intersection: RAW, WAR and WAW. Atomic contributions are not ordered among themselves but are ordered against ordinary access. Observe adds no anti-dependency. A Next read binds to the next later writer (692-719).
- Values: every node's reaches (bindings, address tables, payload views, progress words, host views).
- Safety: the edges become fences, barriers and events downstream (R27-R33).
- Lifetime: a span lives from its define until a covering non-atomic write erases it (680-682).
- Consumers: R24 and R6.

**R5. `.m:546-556, 592-617` `MetalAllocationKey/Base`, `MetalPageAccesses`, `MetalViewAccesses`**
- Kind: order (it supplies identity to the frontier).
- Decides: what counts as "the same storage":
  - bridge memory → one key (`&MetalRegionPageIdentity`) plus region offsets from `ncclMeshRegionRuns` (a 16-run stack buffer that grows);
  - shared non-bridge buffers → a NULL key plus the virtual address `contents+offset`;
  - private buffers → the whole heap (all its buffers alias) or the buffer itself.
- Values: all reaches.
- Safety and lifetime: not applicable.
- Consumers: R4, R21, R24, `MetalScheduleReadVersions`.

**R6. `.m:759-771` `MetalViewDependencies` and `.m:1692-1723` `MetalViewDependencyDeclaration`**
- Kind: order.
- Decides: dependency sets for any list of access sets. The second function emits a fused producer→consumer adjacency (`offsets`, `targets`, `holders` counts) as constant tables in Metal source.
- Values: fused tile views.
- Safety: unsure. The holders counts look like run-time arrival counts in the fused kernel.
- Consumers: R7. The declaration function is called by metal-microbench `matrix_operations.swift`.

**R7. `.m:800-821` `MetalDataflow`**
- Kind: order (census only).
- Decides: each command's `level`, its longest dependency depth, at invocation 0.
- Consumers: only the stdout census (1229-1234, `MetalCommandLine`). Nothing reads `level` for a decision.

**R8. `.m:856-891` `MetalCaptureAddresses`**
- Kind: allocator plus slot/repetition.
- Decides:
  - For each binding that holds embedded pointers, it allocates `MetalLeaseBuffer(stride*repetitions)` (866), with stride the uniform length rounded to 8.
  - For each repetition t it copies uniform slot t (869) and writes each pointer word `gpuAddress + MetalInvocationOffset(origin, stride, instances, t, invocation)` (874-876).
  - It then rebinds the slot to that table with `strides=stride`, `instances=repetitions`, `invocations=0` (886-887), and moves that slot's reaches onto the table (880-885).
- Values: argument records that contain address words.
- Safety: one row per repetition, written by the host before any submission, never reused.
- Lifetime: replay build → read by invocation t's dispatch → freed with the replay owner (888, 900, 903).
- Consumers: R9, R12, `MetalAccesses`. The capacity is the `invocations` argument.

**R9. `.m:896-975` `MetalReplayBuild` (struct from `posix_memalign` in `MetalReplayMade` 1154-1163)**
- Kind: allocator, capacity bound and repetition.
- Decides:
  - Indirect command buffer (ICB) segments of `MIN(MESH_INDIRECT_COMMANDS=4194304, …)` commands each, for `count*repetitions` in total (928-938). The device allocates them outside the region (comment at 930-931).
  - Repetition t's command i is encoded at row `t*count+i`, with slot offsets (942-963).
  - A residency set (909-915).
  - The associated `records` object keeps `repetitions` (907).
- Values: ICB rows and residency.
- Safety: rows are per repetition. Nothing is ordered inside the ICB (967-970: concurrent dispatch, "never fences unrelated commands"); order comes from the schedule graph.
- Lifetime: until `MetalReplayFree` and the owners are released.
- Consumers: R10, R11, R19, and the bound check at 2108.

**R10. `.m:1002-1012` `MetalReplaySegments`**
- Kind: slot/modulo over ICB storage.
- Decides: segment `at/size`, local row `at%size` (1008). Rows are placed by the same rule at 950-951.
- Consumers: R11, R19.

**R11. `.m:1167-1189` `MetalReplayOptimize`**
- Kind: wait/event.
- Decides: runs Apple's ICB optimizer on a new MTL4 queue it creates itself (1170), with its own allocator, command buffer and shared event, then blocks the host with `waitUntilSignaledValue:1 timeoutMS:UINT64_MAX` (1186). That is an unbounded host wait plus an extra queue.
- Values: ICB segments.
- Lifetime: preparation only.
- Consumers: `MetalReplayDescribed` 1242.

**R12. `.m:1015-1030` `MetalReplayDispatch4` and `.m:1459-1476` `MetalScheduleComputeNow`**
- Kind: slot reader plus reuse rule.
- Decides: a direct dispatch writes `setAddress` into the part's single `MTL4ArgumentTable` and then calls `setArgumentTable` (1017-1021). Addresses use `MetalInvocationOffset(…,t,…)` (1020, 1025). `ComputeNow` always passes t=0.
- Safety: unsure. Reusing one table across dispatches assumes the encoder snapshots it at each dispatch; the code does not state this.
- Consumers: R19, `MetalScheduleCompute`.

**R13. `.m:1034-1087` `MetalCaptureAppend`**
- Kind: capacity bound plus retention.
- Decides: binding index < `MESH_BUFFER_BINDINGS`=31 (1044) and memory index < `MESH_MEMORY_BINDINGS`=31 (1065). Each binding reach sets an `extents` bit (1082). Every named buffer is added to `resources`; pipelines and uniforms to `owners`.
- Values: dispatch records, reaches, address records.
- Lifetime: the replay owner and residency.
- Consumers: R3, R8, R9.

**R14. `.m:1213-1225` (`MetalReplayDescribed` host bindings) and `.m:2121` (`MetalScheduleAppend`)**
- Kind: slot/modulo.
- Decides: a host operation keeps `instances` prepared bindings. Invocation t runs `bindings[ncclMeshInvocationIndex(t, invocation, bindings.count)]`.
- Values: host views and native tasks.
- Safety: the modulo alone, plus R4 on the declared views.
- Consumers: `MetalScheduleHostWork` → R24.

**R15. `.m:2104-2138` `MetalScheduleAppend`**
- Kind: capacity bound plus order.
- Decides: refuses invocations past the recorded repetitions (`NSMaxRange(invocations)>held[4]`, 2108). It splits each t's command range at crossing marks and host positions into Range nodes (2111-2135), and builds crossing views for each (t, mark) (2126).
- Consumers: R24. Called from `torch_mesh/_stream.mm:738` with `NSMakeRange(0,1)`.

**R16. `.m:1629-1639, 1665-1680` `MetalScheduleRange/Replay/Expand`**
- Kind: order.
- Decides: a replay range is one node. At flush it expands into one node per command, with accesses `MetalCommandReaches(records+c, reaches, invocation)` (1673). This is where the slot formula's output enters the frontier.
- Consumers: R24.

**R17. `.m:1447-1457` `MetalScheduleData`**
- Kind: allocator (bump).
- Decides: carves 16-byte units (`MESH_COMMAND_DATA_ALIGNMENT`) out of the current 64 KiB `MetalLeaseBuffer` page (`MESH_COMMAND_DATA_BYTES`, larger if one record is larger). A full page is replaced (1450-1452); the old page stays in `owners` and residency. The binding is a Read at `s->used`, invocation 0.
- Values: kernel uniform words (`_stream.mm:796`).
- Safety: never reused; the host writes it once at preparation.
- Lifetime: until the part is freed (2415).
- Consumers: `MetalScheduleCompute` → R4 and the dispatch.

**R18. `.m:1393-1433` `MetalScheduleCreatePart` and `.m:1682-1690` `MetalScheduleWritable`**
- Kind: allocator (Metal objects).
- Decides: each part gets its own command allocator, command buffer, argument table (31 entries) and residency set. A new part is made when the tail part is finished. The first part attaches the queue's `MetalQueueAccess` (1417-1426).
- Lifetime: until `MetalScheduleFree`.

**R19. `.m:1593-1627` `MetalScheduleRangeNow`**
- Kind: slot reader plus residency.
- Decides: encodes rows `c ∈ [t*n+from, t*n+to)` (1614), using `records+c%n` (1615). It dispatches directly with t (1619) or runs ICB row c (1621-1623). Each replay's residency is merged once into the part's single set (1604-1612).
- Consumers: R29.

**R20. `.m:1506-1510` `MetalScheduleInput`**
- Kind: wait/event.
- Decides: makes an external producer node with (event, value) over the declared views. `hostProducer=YES`, so GPU consumers get it as a host dependency, which becomes a queue `waitForEvent` (R32, R33).
- Values: pages produced outside the function (`_stream.mm:602`).
- Lifetime: the event is retained by the node.

**R21. `.m:1495-1504, 726-757` `MetalScheduleReadVersions` and `MetalViewQualify` (applied at 1779-1783)**
- Kind: order.
- Decides: reads inside the callback get the Next qualifier for bytes not yet overwritten; a local write ends that version (753-755).
- Consumers: R4.

**R22. `.m:1512-1518, 2034-2046` `MetalEncoderCompute` and the flush of declarations**
- Kind: allocator/replay.
- Decides: all collective leaf commands are collected into one capture, built as one replay with 1 invocation and no operands.
- Consumers: R16, R24.

**R23. `.m:1541-1554` `MetalScheduleCross/CrossWork`**
- Kind: order.
- Decides: runs the crossing callback at once on the tail part's encoder, table and residency. The declared views and group are thrown away: `(void)accesses;(void)group;` (1542). A crossing is ordered only by the inner `MetalEncoderCompute` reaches.

**R24. `.m:1771-1863` `MetalWorkDependencies`**
- Kind: order/schedule plus a reuse rule.
- Decides:
  - (a) R21, then R4, over `root->frontier*` in declaration order (1797-1800).
  - (b) Host-resource ring: use i of a `MeshHostResource{identity,capacity}` depends on use i−capacity: `if(uses.count>=resource.capacity)[w->dependencies addIndex:uses[uses.count-capacity]]` (1784-1796). Capacities come from `mesh-cpu-source.m:150` and `mesh-coreml.m:220`.
  - (c) Pending Next/Observe reads are matched to declared inputs (1806-1830); an unmatched one is refused (1831).
  - (d) Topological sort that tags each node's host and GPU predecessors (1843-1860); a cycle is refused (1861).
- Consumers: R25-R32.

**R25. `.m:2031-2101` `MetalScheduleFlush`**
- Kind: schedule (list scheduler).
- Decides: which ready node goes next: a host operation first; else a GPU node whose host dependencies are already selected; else the lowest index (2073-2080). GPU nodes with equal host dependencies are batched. It creates the listener (2056-2059) and one shared event per host node (2060-2064).

**R26. `.m:1985-1998` `MetalScheduleCohorts`**
- Kind: schedule.
- Decides: nodes with equal (deps, successors, host deps, stages) that are not exported and not consumed by the host form one cohort, sharing one fence.

**R27. `metal-scope-plan.inc:88-156` `MetalScheduleScopes` (helpers 4-83)**
- Kind: order/fence.
- Decides: where encoder scopes are cut, the stage barriers inside a scope (`barrierAfter/Before`), and which producers need a fence (`needsFence`: cross-scope or exported). The `generation` counter (21-24) is only search bookkeeping. Exported nodes come from `MetalScheduleFrontierNodes` (`.m:1943-1951, 2053-2055`).

**R28. `.m:1953-1981` `MetalScheduleBeginCommand/EndEncoder`**
- Kind: fence/barrier.
- Decides: one encoder per scope; `barrierAfterEncoderStages:…visibilityOptions:Device|ResourceAlias` (1970-1971); `waitForFence:beforeEncoderStages:` for every producer that is in another scope or external (1972-1980). Waits are deduplicated per encoder in `fenceStages`.

**R29. `.m:2000-2029` `MetalScheduleEmit`**
- Kind: fence/event.
- Decides: one `newFence` per exported cohort, with `updateFence` after its stages (2020-2026). A host-consumed cohort calls `MetalSchedulePublication` (2027). Host nodes are bound through R31.

**R30. `.m:1865-1873` `MetalSchedulePublication` and `.m:1917-1937` `MetalSchedulePart`**
- Kind: event plus part cut.
- Decides: a part gets a shared event, signalled to 1 after its commit (2397); this is the GPU→host edge. A new part is started when the host dependencies differ (1927-1933).

**R31. `.m:1875-1912` `MetalScheduleBindHost` and `.m:1725-1733` `MetalHostFinished`**
- Kind: wait/event.
- Decides: a host task runs when all its dependency events reach their values (listener `notify`, atomic `remaining` countdown 1895, 1908-1909). A task with no internal predecessor also waits on `root->start` (1890-1894). On completion it signals its own event to `eventValue` (1731).

**R32. `.m:2221-2254` `MetalScheduleOrder`**
- Kind: schedule plus wait.
- Decides: a flat commit order of parts, picking ready parts with the fewest new host waits first, and each part's wait list of (event, value) after removing waits already implied (2240-2247).

**R33. `.m:2384-2405` `MetalScheduleEnqueue`**
- Kind: wait/event.
- Decides: for each part, `queue waitForEvent` per wait (2395), `commit` (2396), `signalEvent:s->event value:1` (2397); then `root->start.signaledValue=1` (2400).

**R34. `.m:2198-2216, 2190-2196` `MetalScheduleFinishPart/Completed`**
- Kind: event and retire.
- Decides: commits residency and installs a feedback handler that records status and end time, decrements `pending`, and calls `MetalScheduleFree(part)`. The values a part holds retire at this point.

**R35. `.m:782-798, 2259-2288, 2290-2323` `MetalQueueAccess`, `MetalSchedulePublishFrontier`, the import in `MetalScheduleFinish`**
- Kind: order across functions plus a retire read.
- Decides:
  - Publish replaces producers with stubs that hold (event, fence) (2267-2274), and keeps each host resource's last `capacity` uses (2281-2287).
  - Import drops a producer whose `signaledValue>=eventValue` (2299-2300). That is a read of run-time completion during preparation.
  - Spans are renumbered (2303-2311). The whole step runs under `@synchronized(queueAccesses)`.
- Safety: later functions wait on the stub's fence (1922, 1974) or event.

**R36. `.m:2147-2152, 2338-2382` `MetalScheduleResults/Status/End`**
- Kind: wait/event (non-blocking observation).
- Decides: a function is complete when `pending==0`, all part and host statuses are done, no communicator has an error, and every result word ≥ its threshold (2360-2367). `UINT64_MAX` means the link was lost.
- Consumers: `_stream.mm` `Executions::add` (1192-1198).

**R37. `.m:2407-2424` `MetalScheduleFree`, with retention at 1487, 1491-1493, 1631, 1645-1646**
- Kind: reuse/retire rule.
- Decides: when the last reference drops (the creator, the submission, completion), it releases `owners`, which runs the `MetalLeaseBuffer` deallocators; parts are freed through `next`. The proof that values are retired is Metal's commit feedback plus the root's completion in R36. Each schedule is enqueued once (`_stream.mm:1202`).

**R38. `.m:1577-1591` `MetalStamp`**
- Kind: capacity bound plus an ordering side effect.
- Decides: 4096 timestamps per part (1581). Precise timestamps order the dispatches they bracket, only when `MESH_PREPARE_RECORD` is set.

### nccl-mesh-metal.m / nccl-mesh-metal.h

**N1. `nccl-mesh-metal.m:43-59, 371-395` `MeshArgumentPages` and `argument_pages` (with `metal_piece_hold`, `nccl-mesh.c:4137-4148`)**
- Kind: allocator (bump). It is a second region allocator, separate from R1.
- Decides: pieces rounded up to `RECORD`=128 bytes (`MESH_METAL_ARGUMENT_BYTES`, `etc/bridge/layout.h:35`), cut from the current piece. A new piece comes from `ncclMemAlloc(max(minimum, 64 KiB))` plus `lease_buffer`. The `held` array doubles as it grows (381-386).
- Values: kernel argument records, fold pointer tables, `mask_spans` lengths, premultiply constants, zelloth address tables, CPU pointer arrays.
- Safety: "never recycled while that function can execute" (43-45). `dealloc` → `metal_piece_give` → `ncclMemFree`.
- Lifetime: allocated at lowering, filled by the host, read on each execution. The owner is attached as an associated object to the crossing's residency set (930-932), which is the tail part's set (`.m:1546-1547`, `_stream.mm:769-776`); it is released at R37.
- Consumers: N2 → N13.

**N2. `nccl-mesh-metal.m:487-505` `record4`**
- Kind: allocator use.
- Decides: puts `struct args` into an argument page. Slots 0-5 are bound as d, s, record, a, v, record, all with invocation 0; only slot 2 carries an access (Read). Sets `p.sequence=0`, and `polls` only when failure words are present.

**N3. `nccl-mesh-metal.m:401-486` `fold_coverage`**
- Kind: allocator (address table).
- Decides: a table of 1+K×Q words. Word 0 is the constant 1; each (cell, input) entry is a pointer to that input's `active` word, or to the constant (418-443). It rebinds slot 3 (468-471) or the CPU arguments, and emits `MESH_FOLD_ACTIVE`.
- Consumers: the page-effects solve (the reaches it adds) and the kernel.

**N4. `nccl-mesh-metal.m:761-809, 813-871` `native_zelloth` and `metal_zelloth`**
- Kind: allocator (address table).
- Decides: table `z` holds K, the GPU addresses of `own`, `out` and each input, and the block ranges (841-846, 855). The CPU form builds a pointer array (775-778). Blocks come from `mesh_landing_cells`.

**N5. `nccl-mesh-metal.m:716-747` `fold_relations`**
- Kind: order/wait thresholds plus a reuse rule.
- Decides: for each run of equal thresholds, an Observe transfer on the input's landing word and a Publish transfer on `summed` (727-733). Each `sum->waits` entry becomes an Observe with `observeAccess=MetalReachWritten` (745). That last one is the WAR guard ("ranges which overwrite an outstanding SEND source", `.h:185-187`), i.e. the wait on `sent` before reuse.
- Consumers: P4.

**N6. `nccl-mesh-metal.m:136-152` `metal_page_selection/effect`**
- Kind: order.
- Decides: a transfer's place in the command order: `commands.count + (kind==Publish)` (145). The root is the space serial or the region.

**N7. `nccl-mesh-metal.m:167-208` `metal_coverage`**
- Kind: order thresholds.
- Decides: splits transfers along the coverage ranges and sets `threshold = base + before` by binary search over `ends` (190-193).

**N8. `nccl-mesh-metal.m:221-257` `metal_place_sources`**
- Kind: order.
- Decides: reinterleaves commands at boundary positions and relocates the transfers' orders to match.

**N9. `nccl-mesh-metal.m:688-690, 703-706, 873-876` `release_args`, `metal_publish`, `metal_open_gate` (kernel at `nccl-mesh-kernels.h:57`)**
- Kind: wait/event (publication).
- Decides: a single-threadgroup store of `p.scalar + j*p.stride` for repeat j. This is a `stride*slot` formula on progress counts. The access is Written|Atomic.

**N10. `nccl-mesh-metal.h:83-103, 164-171` `struct metal_wait/metal_release` and the poll bound**
- Kind: wait/event.
- Decides: a wait on a word until it reaches `target`, bounded by `polls` (≤ `NCCL_MESH_POLLS`). `metal_wait.stride` is declared, but no code in my scope reads it (`fold_relations` reads only `target`).

**N11. `nccl-mesh-metal.m:108-116, 546-560` `element_access` and `dispatch4`**
- Kind: capacity bound plus modulo.
- Decides: grid = `MIN(units, MESH_TRANSPORT_GRID_THREADS=1048576)`; element k goes to group `floor((k mod grid)/threads)`.

**N12. `nccl-mesh-metal.m:509-544` `command_emit`**
- Kind: order input.
- Decides: bytes that no element or cell relation covers get a relation spanning all groups (515-528). The metadata slot is fixed at 6 (535), and the frame is recorded (537).

**N13. `nccl-mesh-metal.m:925-999` `metal_record`**
- Kind: retention/residency.
- Decides: attaches the argument pages (930-932); compiles a specialized pipeline (972-977) or uses a cached one; CPU leaves go to `MetalEncoderHost`, GPU leaves to `MetalEncoderCompute`; adds every resource and commits residency (996-998).

**N14. `nccl-mesh-metal.m:259-260, 288-329` `kernels_`**
- Kind: pool/cache.
- Decides: pipelines cached per device, never evicted. This involves no memory placement.

**N15. `nccl-mesh-metal.m:338-346` `metal_wrap`, `metal_event`, `metal_wait`**
- Kind: buffer view and wait.
- Decides: `metal_wrap` wraps a caller address with no deallocator. `metal_wait` is a host `waitUntilSignaledValue` with a timeout of ns/1e6 ms.

**N16. `nccl-mesh-metal.m:154-161` `coverage_value`**
- Kind: pool/cache (preparation).
- Decides: interns coverage values keyed by the buffer pointers and offsets.

### metal-page-effects.m

**P1. `metal-page-effects.m:811-814` `PageData`**
- Kind: allocator.
- Decides: one `MetalLeaseBuffer` per blob, filled with `memcpy`. Blobs: the domain pages (P2), projected offsets (965), cut tables (1011), announcement tables (1046), and observation records `[polls, channel, entry, thresholds…]` (1382, 1422).
- Safety: none needed; written once.
- Lifetime: retained by `domain->pages` and the field owners → `MetalPageSource` → command → schedule.

**P2. `metal-page-effects.m:815-827` `PageDomainPages`**
- Kind: pool/cache plus progress state.
- Decides: four pages per publication domain, made once under `@synchronized` and shared by every command of that domain: thresholds, `completed` (read/write atomic, initialised to 5 or 7 per cut at 691; the meaning of the bits is unsure), expected counts, and `arrivals` (read/write atomic, zeroed).
- Safety: never reset between executions. `Rebind` builds new domains without `pages` (786-806), so each occurrence gets fresh pages. Whether one source can execute twice is unsure.
- Consumers: the `mesh_graph_coverage/settle` code emitted by P10.

**P3. `metal-page-effects.m:832-879, 892-909` `MetalPageFields add:/composeFor:` and `PageReplaceFields`**
- Kind: allocator (address table).
- Decides: fields are deduplicated by a text key `"%p offset bytes"` (852-854), i.e. pointer identity. A table `MetalLeaseBuffer(fields×8)` holds each field's GPU or host address (865-871), and offset tables are rewritten to absolute addresses (872-876). Kernels load `((device uchar * const device *)mesh_effects)[i]`.
- Consumers: `metal_record` (978-981), `_stream.mm:1137`.

**P4. `metal-page-effects.m:612-760` `MetalPageEffectsSolve`**
- Kind: order/wait thresholds.
- Decides:
  - the order of arrivals (640-642);
  - for each publication word and cut, the contributing groups and their expected count (664-692);
  - initial-input spans and the earlier arrivals they wait on (693-713);
  - for each observer and command group, the greatest threshold row (720-749);
  - terminal results: arrivals whose threshold is above anything consumed (750-756).
- Consumers: P9-P11, and callers of `MetalScheduleResults`.

**P5. `metal-page-effects.m:536-562` `PageObservationGroups`**
- Kind: order plus reuse rule.
- Decides: a reader waits on an arrival until its pages are overwritten or a later arrival supersedes it (543-547). With `observeAccess=Written` it selects the later writers instead: the gate on overwriting.

**P6. `metal-page-effects.m:400-496, 580-608` `PageCoverageAccesses`, `PageCoverageAnnouncements`, `PageObservationMasks`**
- Kind: order.
- Decides: the closure of activity reads, and which cuts are announced or active in each group.

**P7. `metal-page-effects.m:374-380, 508-520` `PageWord` and `MetalPagePublicationDomains`**
- Kind: order grouping.
- Decides: publications grouped by the word's region offset (the word must be a single region run), sorted by threshold.

**P8. `metal-page-effects.m:498-506` `MetalPageEffectsOrder`**
- Kind: order.
- Decides: renumbers commands and transfers into global positions; equal positions stay equal.

**P9. `metal-page-effects.m:762-808` `MetalPageEffectsRebind`**
- Kind: reuse of solved incidence.
- Decides: copies the solved incidence to a new occurrence and recomputes thresholds from the new transfers (797-803).

**P10. `metal-page-effects.m:914-1066` `PageCoverageSource`**
- Kind: publication plus allocator.
- Decides: emits `mesh_graph_coverage` atomics on `state`/`arrivals` and `mesh_graph_settle`, which publishes the prefix `cuts[p-1]` to the word (971-972, 1062-1065). It allocates the cut and announcement tables.

**P11. `metal-page-effects.m:1353-1392, 1393-1476` `PageObservations` and `MetalPageEffectsSource`**
- Kind: wait.
- Decides: emits a `mesh_step_observe(word, threshold, violation, polls, channel, entry)` per domain, merged per word at the greatest threshold (1364-1390) and guarded by the `active` predicate (1428-1437). These are GPU spin waits bounded by `polls`.

**P12. `metal-page-effects.m:1192-1259` `PageSites` and `MetalPageEffectsCompose`**
- Kind: fence/barrier (sync).
- Decides: threadgroup or simdgroup barriers plus a system-scope `atomic_thread_fence` around enter, leave and join; lane 0 performs the effects.

**P13. `metal-page-effects.m:1264-1345` `PageObservationOrder`, `PageActivityInputs`, `PageSourceIncidence`**
- Kind: order.
- Decides: a topological order of the observation predicates inside one command.

**P14. `metal-page-effects.m:1112-1125` `MetalPageMetadataSlot`**
- Kind: allocator (binding slot) plus capacity.
- Decides: the first buffer index below `capacity` that the entry's parameters do not use; `NSNotFound` when all are taken.
- Consumers: metal-microbench `common.swift`, `litert_program.m`. The collective path instead fixes slot 6 for GPU and slot 0 for CPU.

**P15. `metal-page-effects.m:12-33, 71-134` `PageBuffer`, `PageOperandIdentity`, `MetalPageRegionRelation/Selection/Inverse`**
- Kind: not a decider; it reads the bridge's placement.
- Decides: turns `buffer.contents+first` into region offsets via `ncclMeshRegionRuns`, so the chooser handles pointers. Coverage equality is by page identity (43-44).

### mesh-tensor-layout.m

**L1. `mesh-tensor-layout.m:1007-1181` `MetalTensorPlace` and `TensorPlacedOnTiles` (→ `mesh_graph_tile_slots`)**
- Kind: allocator (placement inside one storage domain).
- Decides: fast paths for the identity (1043-1049) and for tiling (1050-1051). Otherwise it joins equations into components and places each root at the first aligned base at or after the previous end, a cursor recurrence (1137-1160). It reserves footprints (1098-1107), enforces alignment (1110-1131), refuses overlap (1162-1172) and checks injectivity (1161).
- Values: element positions only; no lifetimes are involved.
- Consumers: `_stream.mm:1315`.

**L2. `mesh-tensor-layout.m:1183-1247` `TensorPlaceDemands` and `MetalTensorPlacePreferred`**
- Kind: capacity bound plus chooser.
- Decides: the extent (largest footprint end) and alignment (lcm) of the storage (1202-1215), and admits preference groups in descending weight (1229-1243).
- Consumers: `_stream.mm:1343`.

**L3. `mesh-tensor-layout.m:2359-2406` `MetalTensorPages`**
- Kind: not a decider; it reads placement.
- Decides: wraps the bridge page-table calls. `bind:` shifts a layout by `_origin/bytes`, an offset derived from a pointer.
- Consumers: `nccl-mesh.c`, `_stream.mm`, metal-microbench `matrix_operations.swift`.

### Readers of slot-index formulas (in scope)

- `metal-recording.h:34-36` `MetalInvocationOffset`, which calls `ncclMeshInvocationIndex`.
- `metal-recording.m:564, 570, 579, 581` (`MetalAccesses`: indirect grid, bindings, address reaches, binding reaches).
- `metal-recording.m:869, 875` (`MetalCaptureAddresses`: uniform slot, pointer target slot).
- `metal-recording.m:963` (ICB `setKernelBuffer`) and `:1020, 1025` (direct dispatch).
- `metal-recording.m:2121` `bindings[ncclMeshInvocationIndex(t,[host[@"invocation"] longLongValue],bindings.count)]`.
- Invocation × count rows: `metal-recording.m:950-951` (`t*count+i`, `/ceiling`, `%ceiling`), `:1008` (`at/size`, `at%size`), `:1614-1615` (`c=t*n+from`, `records+c%n`), `:2108` (invocations ≤ repetitions).
- Fields that carry the phase: `MetalCommandBinding.invocation` (`.h:179`), `MetalCommandAddress.invocation` (`.h:193`), `indirectInvocation` (`.h:213`), `MetalHostDescription.instances/invocation` (`.h:239-244`).
- Count strides: `nccl-mesh-metal.m:689` (`.stride` → `p.scalar + j*p.stride`), `nccl-mesh-metal.h:168` (`metal_wait.stride`, not read here).
- `metal-page-effects.m:91, 194-200` `MetalPageAccessLayout.invocation` and `resolve:displacement:` (shifts the bytes by the displacement). Its only reader in scope passes 0 (261); metal-microbench `litert_program.m` and `matrix_operations.swift` also use the class.

### Entries that share state

- **Slot family:** R2, R3, R8, R9, R12, R14, R16, R19 and `MetalAccesses` share the `strides/instances/invocations` arrays of `MetalRecordedDispatch` and `MetalRecordedAddress`. R8 rewrites them, and R16 feeds their output into R4.
- **Frontier:** R4, R5, R21, R24, R27 (`exported`) and R35 share `root->frontierRead/Written/Atomic` and the per-queue `MetalQueueAccess`. The R24(b) host-resource ring and R35 share `hostResources` / `queue->resources`.
- **Work-node fields:** R24-R34 share `MetalWork`'s `dependencies/hostDependencies/gpuDependencies/part/scope/fence/event` and the parts' `waits/partDependencies`.
- **Two region allocators with the same retire rule:** R1 backs R8, R17, P1 and P3; N1 is a separate bump allocator. Both are freed by object ownership at R37, not by a page assignment.
- **Order and thresholds:** N5-N8 produce `order` and `threshold`, which P4-P11 consume. All commands of a domain share P2's state pages. P3's table and `MetalPageSource->reaches` go back into R13 and R4 through N13 and `_stream.mm`.
- **Pointer-to-region readers:** R5, P7, P15 and L3 all turn pointers into region offsets (`ncclMeshRegionRuns` or page tables).
- **ICB storage:** R9, R10, R11 and R19 share the segments.