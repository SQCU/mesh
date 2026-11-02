<!-- design/page-maps-2026-11-02.md#the-runs-solve-each-choice-and-its-ground: a read-only inventory of a form the run's solve replaces, read by a subagent of the 2026-11-02 session at mesh-worktrees/joint 815b3af9 plus its uncommitted backend.py, collective_views.py and operand.py. It lists call sites to rewrite after the solve exists. It is a claim about the old code, not a requirement, and it decides nothing (design/rubric-actionable.md, procedure 5). -->

The inventory covers 54 entries (E1–E54); everything was read in place, nothing edited or run. One file in scope doesn't exist: `rdma/mesh-lease-retire.c`. The ledger is header-only (`mesh-lease-retire.h`). `schedule_units`, `issue_lay` and `schedule_laid` sit just past line 4000 (4056–4116); I covered them because you named them. `slot_laid`, `zappo`, `place_data`, `landing_progress` and `metal_piece_hold` (4137–4289) appear only as consumers.

Paths: `R` = `/Users/mdot/dox/mesh-worktrees/joint/rdma/`; `c:` = `R/nccl-mesh.c`, `df:` = `mesh-dataflow.c`, `lr:` = `mesh-lease-retire.h`, `mc:` = `mesh-call.h`, `dg:` = `mesh-diagnostics.c`, `nh:` = `nccl.h`. In each entry: **D** = what it decides, **V** = the values it covers, **S** = how reuse or safety holds today, **L** = lifetime events, **C** = who consumes the result.

## A. The region's page allocator and leases

**E1** df:397-412 `mesh_allocate`, with `mesh_scan` 340-356, `mesh_scan_down` 360-385, `mesh_claim` 326-338 and `MESH_LARGE_RUN` 391 — allocator
- D: which run of rows or arena pages a request takes.
  - Runs of 1024 pages or more go top-down, to the highest fit.
  - Smaller runs are first-fit from a per-client resume cursor (`*cursor=first+count`). On a miss it rescans from `begin`, then calls `mesh_reclaim` and scans once more.
- V: every lease of every kind.
- S: compare-and-swap claims on the `own` bitmap; the `hot` bitmap excludes pages released but not yet reclaimed. It knows no lifetimes, and where a run lands depends on the cursor's history.
- L: from section creation to `mesh_lease_return` or reclaim.
- C: `mesh_rows_alloc`, `mesh_arena_alloc`.

**E2** df:439-444 `mesh_arena_alloc` — allocator plus capacity bound
- D: physical pages. It first tries with `extent=MESH_DMA_EXTENT/pgsz`, so a run stays inside one registration extent, and retries with no extent on ENOMEM.
- V: arena pages. S: as E1.
- C: `mesh_section_create` (mesh-call.c:118).

**E3** df:414-426 `mesh_rows_alloc` — allocator
- D: virtual rows, from `row_cursor`. Sets each row's mapping to ABSENT and its owner to the client.
- C: `mesh_section_create`.

**E4** mc:4-12 `struct mesh_section {first,pages,bytes,count,stride}` and `mesh_section_create`; implementation mesh-call.c:95-150 — slot/stride construct
- D: `count` slots on `stride` rows. `mesh_section_row` is `first+index*stride` (mesh-call.c:11).
  - `MESH_BACKING_PAGES` takes one block per row.
  - `MESH_BACKING_LINEAR` takes the whole section's pages as one run.
- Capacity: `span>capacity/count` returns ENOMEM.
- C: `mem_alloc` (always `count=1`, c:1721); the quiet probe (`count=rounds+1`, dg:97).

**E5** c:1707-1735 `mem_alloc` and c:1737 `ncclMemAlloc` — allocator (process level)
- D: reaps returned leases (c:1715), creates a section, records `struct lease{address,row,bytes}` (c:714-722) and marks the `lease_of` rows (c:1662-1667).
- V: every library lease:
  - the failure-word page, c:1360
  - a group's own pages, LINEAR (`zappo` 4180)
  - progress words, LINEAR (`landing_progress` 4262)
  - span tables (`place_group` 4309)
  - relay leases (`write_binding_prepare` 7733)
  - `input_pages` and `input_ready` (`source_generation` 5487/5491)
  - command data pieces (`metal_piece_hold` 4142)
  - plus 6133 and 7251
  - plus callers' leases.
- S: none at allocation; all safety is at return (E7).
- L: no happens-before lifetime is attached.
- C: the callers listed above.

**E6** c:1788-1800 `ncclMemFree` — reuse/retire rule
- D: a logical free. Sets `returned`, appends the lease to `returning`, then calls `leases_reap`. Never waits.
- C: `lease_give` through the reap.

**E7** c:1670-1689 `lease_give`, c:1633 `lease_returnable`, c:1696-1703 `lease_take`/`leases_reap` (over `mesh_graph_settle`, mesh-graph.c:815) — reuse/retire rule
- D: a returned lease goes back to the region only if `!l->pins && mesh_lease_retire_detachable(...)`; otherwise it stays in `returning`.
  - Retried at every `mem_alloc`, `ncclMemFree` and `comm_let_go` (c:902).
- S: a read of the retire ledger, or the detachment fallback (E8).
- L: SEND completion (`sent`), landing (`landed`), the bridge leaving tables (`taken[0/1]`).
- C: `mesh_lease_return` (E10).
- Unsure/dead: `pins` is only ever decremented (c:1676). The composed-view bookkeeping is dead per c:7556-7560.

**E8** c:1624-1632 `lease_detachable` and df:504-515 `mesh_holding_retained` — reuse rule (detachment)
- D: a dependency is released when both hold:
  - no hold of this attach still uses that stream with that holding generation;
  - the bridge holds none of the client's pages on that stream. That means no port child; or the owner word is no longer this client's and either `unfinished!=client` or every lane has `done>=posted`.
- C: `lease_returnable`, `mesh_retired_release`.

**E9** c:1605-1618 `lease_read_sample` — the ledger's query
- D: returns `{wire, holding=h->generation, sent, landed, left=taken[0..1]}` from a live hold. Once the hold is vacated it returns ENOENT, so dependencies still open can then complete only through E8.
- `comm_let_go` (c:893-894) latches every lease's `ready` before giving the holds back.

**E10** df:541-551 `mesh_lease_return`, with `mesh_buffer_reclaim` 465-477, `mesh_reclaim` 480-488, `mesh_retired_release` 519-534, `mesh_retire` 263-276 — reuse rule (bridge side)
- D: `mesh_lease_return` disowns the non-head rows, clears ROW_OWN and reclaims the head at once (arena bits and ROW_HOT cleared).
  - Discrepancy: the comment at 536-540 says the head is closed instead while `mesh_holding_retained`. The code makes no such check. Retention rests wholly on the caller's E7/E8.
- `mesh_retired_release` frees a retired client's closed rows unless they are retained.

**E11** df:278-293 `mesh_slot_take` — pool plus generation
- D: picks the first vacant of `MESH_CLIENTS` client slots by CAS. The owner word is generation (from `m->serial`) | slot | pid.
- V: the client identity that owns every row.
- L: until `mesh_detach`/`mesh_retire`.
- C: `mesh_attach` → `attach_current` (c:819).

**E12** c:808-830 `attach_settle`/`attach_current` — reuse rule for the whole attach
- D: the attach is let go only if expired, with no leases, no communicators and `!wrapped`.
- `wrapped` (c:742) is set by any Metal wrap: c:1372, 1940, 1946. A wrapped attach is never let go; the process must exit.

**E13** c:1926-1952 `stream_page`/`lease_buffer` — allocator (Metal wrapper objects, not pages)
- D: one Metal buffer per lease, made at first use and released in `lease_give` (c:1679); one per attach over the stream pages `[stream_off, client_off)`.
- C: `ncclMeshMemBuffer`, `fabric_realize`, `metal_piece_hold`, `zappo`, `landing_progress`, `place_group`, `source_generation`, `source_group`, `as_metal`, `zoblimp`.

**E14** Readers of allocation results, not decisions (listed because pinned items enter here)
- c:1753-1774 `ncclMeshLeasePages` → torch_mesh/allocator.mm:90-92.
- c:1637-1653 `lease_runs`/`lease_page_offset`.
- c:2445-2460 `binding_backing` → `arithmetic_readers` 5177, `message_backing` 6510, `packets_prepare` 6689.
- c:1843-1851 `lease_offset`, pointer → region offset → `landing_progress`, `place_group`, `present_group`, `ncclMeshCommSourceInputs`.

**E15** c:2464-2473 `binding_meets` — reuse/hazard rule
- D: physical-page intersection of two bindings' runs. Spatial only, no lifetimes.
- C: `arithmetic_readers` 5198.

## B. The retire ledger

**E16** lr:21-30, 56-84 `mark_use`/`mark` — reuse/retire rule
- D: one dependency per (stream, wire, holding, direction, metadata), with `end` the completed-unit prefix needed; a repeated mark raises `end` and clears `done`.
  - Directions: SEND→`sent`, LAND→`landed`, LEFT_SEND/LEFT_LAND→`taken[0/1]` (the bridge has left the tables).
- V: payload sources, landing targets, span and zwoggle tables, source-readiness tables.
- C: `owned_end` (c:1910) only.

**E17** lr:45-54 `begin`/`end` — registration scope
- D: while `registering>0`, `ready` returns 0, so the lease is never reused.
- C: `owned_begin` c:1891, `owned_end` c:1912, rollback c:1894.

**E18** lr:89-117, 124-137 `ready`/`detachable`/`drained`/`destroy`
- `ready`: a non-blocking snapshot that latches `done`. A wire or holding mismatch, or a ~0 prefix, sets `invalid`, and an invalid dependency never completes.
- `drained`: has no caller anywhere in the repo.
- C: `comm_let_go` c:894, `lease_returnable` c:1634, `lease_give` c:1682.

**E19** c:1858-1917 `struct owned`, `owned_add`/`owned_uses`/`owned_begin`/`owned_end` — reuse rule (an owner's hold on leases)
- D: `owned_uses` makes 4 dependencies per channel (`2h+d` sent/landed; `2count+2h+d` left), with `holding=hold->generation`.
  - The `end` values are advanced outside scope: `present_group` 4759-4760 (`=m->end.n`) and 4815 (`=ch->linked`).
  - `owned_end` gives OWNED_SENDS leases SEND dependencies only, and gives LEFT dependencies only to OWNED_TABLE leases. On error the scope stays open and the lease is never reused.
- Owners: an issue (c:3960) or a WRITE plan (c:2159). `place_group` 4300 ends the previous placement's scope first.
- Callers of `owned_add`: `place_group` 4343/4403/4404, `source_coverage` 6054-6055, `ncclMeshCommSourceInputs` 6155/6167, `write_prepare_spans` 7280-7281, `write_ready` 7693.
- Callers of `owned_uses`: 4406, 7283.
- Callers of `owned_begin`: 4407, 6062, 6170, 7284, 7694.
- Callers of `owned_end`: `write_binding_drop` 2260, 4300, `issue_free` 5058, 7285.

## C. Communicator setup: holds, channels, failure words

**E20** c:1316-1336 `fabric_acquire` (with c:726 `struct hold`, c:754 `holds[MESH_STREAMS]`, c:798 `holding_generation`) — pool plus generation
- D: the first hold with `users==0` (ENOSPC if none), `mesh_stream_take`, then `generation=++holding_generation`, process-global.
- S: the generation keeps a reused stream's prefixes from retiring older sources (c:1603).
- L: from init to `comm_let_go`'s `mesh_stream_give` (c:898).
- C: `fabric_realize`, `lease_read_sample`, `lease_detachable`, `owned_uses`, `comm_failure`, `watch_turn`.

**E21** c:134-148 `struct channel`, c:1340-1350 `channel_buffer`, c:1363-1375 — order/schedule (positional counters)
- D: wraps the bridge's `ready`/`landed`/`sent` words, with their offsets. `link`/`link_at` are the zwoggle root.
  - `sends`/`lands` are list positions; `out`/`in` are page prefixes; `linked` counts calls linked.
- S: stream order only; each group is laid after everything already scheduled on the channel.
- C: `schedule_laid` 4103-4108, `commit_group` 4826-4827, `present_group` 4812-4815, `plan_record` 4720, `issue_make` 5033, `zabbadoo_plan` 5138, `ncclMeshCommRealized` 6279; the `landed` waits at 5145/5318/5367 (`threshold=end-pages+packet+1`).
- `ready_at` is written (c:1348) and read nowhere.

**E22** Failure-word page: c:1360-1361 `fabric_realize`, with `comm_lost` 763-768, `library_stop` 201-209, `comm_stop` 1499-1506, `comm_failure` 2088-2102, `bound_reached` 234-262 — allocator (one page) plus timing
- D: one `ncclMemAlloc(getpagesize())` per communicator, with its Metal buffer `violation`. Words: [0] cause, [1] claim (atomic_exchange), [2] entry+1, [3] channel, [4] polls, [5] last read, [6] stop.
- L: from init to `comm_free`'s `ncclMemFree` (c:338). It belongs to no `owned`, so it is returned at once.
- C: kernels through `violation`: 5191, 5318, 5367, 5388, 5643, 5813, 5826, 6123-6145. Also `watch_turn`, `comm_failure`.

**E23** c:328-329 `comm_storage_free` is a no-op: the gate page is deleted. nh:104 still describes it.

## D. WRITE binding slots

**E24** c:2151-2187 `write_binding {occurrences, packets, span_first, step_units}` — slot/modulo
- D: `write_span_index` (c:2183) is `step*occurrences + invocation%occurrences`; `write_packets` (c:2186) is `packets+invocation%occurrences`.
- V: DMA span tables (the `spans` lease), unit counts (these set entry ends and prefixes), packet cuts.
- S: the modulo index, immutability ("issued calls borrow them", c:2148), and `owned` on `spans`.
- C: `schedule_group` 3772, `issue_lay` 4087, `write_steps` 2590, `place_group` 4322, `message_backing` 6499, `write_prepare_spans` 7243-7259.

**E25** c:2192-2207 `write_binding_slot` (`ncclMeshInvocationIndex(invocation, layout.invocation, layout.instances)`) and `write_slot_extent` — slot/modulo plus capacity bound
- D: the cyclic instance of a payload or relay window, used as `instance*instance_stride` (c:4203). `instances==0` means a distinct instance per invocation over a finite domain.
- V: relay landings and WRITE payload windows.
- S: the modulo alone, a ring of N=instances.
- C: `place_data` 4200, `selected_effect` 5267, `write_source_copy` 5652, `ncclMeshCommSourceInputs` 6187, `write_prepare_spans` 7266. The extent goes to `write_prepare` 7345 and `write_binding_prepare` 7731, which sizes the relay lease as `last*instance_stride+frame_bytes`.

**E26** c:2214-2238 `tile_offset` (`(invocation % instances)*instance_stride + …`) and `tile_extent` — slot/modulo plus capacity bound
- V: source-readiness and producer words.
- C: `source_generation_effects` 5641, `write_ready` 7678/7696.

**E27** c:2208-2213 `write_tiles`/`write_tile` and c:3679-3681 `copied_word` (`(frame*tiles+tile)*8`) — slot formula
- V: copied-readiness words, from a lease sized `frames*tiles*8` in `source_generation` 5491.
- C: `present_group` 4789, `write_source_copy` 5666.

**E28** c:2255-2269 `write_binding_drop` — retire rule
- D: `owned_end(...,1)`, then `ncclMemFree` of `spans` and `relay`.
- C: `calls_free` 5012, 7459, 7748.

## E. Steps, scratch and partials

**E29** c:2283-2317 `struct steps {buffers, initialized, extent, buffer}` — capacity bound
- D: how many scratch buffers a call has and their byte sizes. Set at:
  - c:2644-2663 (`rank->extent`/`buffers` from `mesh_routed_compile`)
  - c:2684 (`initialized=1+program->scratch`)
  - c:2700 (`=2`)
  - c:2577 (`buffers=1`)
- V: tree scratch and relay generations (c:2300-2301).
- C: `slot_laid` 4162-4166, `zappo`, `place_data` 4205.

**E30** c:2619-2622 `generation_buffer` — allocator decision
- D: which of a call's buffers are the library's own pages: every b≥1, a reduce-scatter's operand, a non-recipient's operand; for a counted call, b≥2.
- C: `slot_laid` 4164.

**E31** c:2744-2754 `partial_split` — allocator/capacity
- D: one more buffer of `elements*SIZE` for the other segments' partials; steps outside the rank's own segment move to it.
- C: `plan_collective` 3621, `slot_laid`.

**E32** c:2493-2536 `steps_copy`/`steps_reset`/`steps_make` — host-only reuse
- Compiler tables are shared by reference count; per-issue arrays are reset each lay.
- C: `issue_make` 5036, `issue_layout` 4880, `zorvek` 4998.

## F. Routing (allocation of link capacity)

**E33** c:934-992 `fabric_discover`
- D: the lowest available wire per bundle at the smaller end's bandwidth; the cables sorted; default and composed write routes built.

**E34** c:386-527, 687-694 `compile_entry`/`compile_form`/`comm_routes`/`program_of`
- D: tree packing with per-tree shares. A call takes the first entry with `bytes < below`.
- C: `routing_prepare` 3391, `plan_collective` 3607.

**E35** c:2764-2914, 3100-3546 `flow_solve` (EPS 0.1), `demands_of`, `solved_steps`, `writes_weigh`, `routing_prepare`/`routing_prior`/`routing_paths`, `route_path_counts`/`route_tree_counts`
- D: multicommodity flow giving each tree's and path's byte share. Equal demands are merged (c:3442).
- Prior catalogs are taken as authoritative (c:3435): placement drawn from earlier routings.
- C: `plan_collective`, and `bind_routed_steps`/`write_transfers` through `call_paths` (c:2948). The counts go to the plan's view (`plan_routes` 6992).

**E36** c:3212-3362 `fold_basis_*`/`ncclMeshFoldPrepare` — order (reduction association)
- D: a balanced pairing where reassociation is allowed, else a chain (c:3242-3251); cells cut by the page quantum.
- C: torch_mesh/reduction.py:41.

**E37** Unconsumed bounds and dead code
- c:655-682 `ncclMeshCommWriteEntries` (most entries per directed cable): no consumer in the repo.
- c:623 `write_forwards`: no callers.
- c:1456 `(void)small`: the init's `small` argument is ignored.

## G. Scheduling

**E38** c:1357 `comm->page=block*pgsz`, with `pages_of` (3773) and `mesh_page_quantum` (3356/3388/3500/3516) — capacity bound (the transfer unit)
- D: the unit count of each entry, and so every prefix `end`.

**E39** c:3744-3801 `schedule_group` (with `group_of` 3716, `meet` 3722) — order/schedule
- D: the send and landing lists of each channel, in `mesh_group_lay`'s common order, filtered by cable, with cumulative `end` prefixes.
- Also `prior`: earlier non-SEND steps that `meet` on the same buffer index and element range. This is aliasing by buffer number, not by page. Relays get their `prior` in `write_steps` 2598-2610.
- C: `compile_group` 3907. `prior` is read by `place_group` 4382, `send_dependencies` 4435, `send_requirements` 4484-4493, `zabbadoo` 5298.

**E40** c:3823-3897 `order_transfers`, c:3813-3821 `write_order` — order plus timing/prediction plus allocator
- D: `mesh_transfer_order`, given declared production times, returns publication and consumption orders, a predicted finish in `seconds`, and `separate[]` (an alias gets its own generation). If any `separate` is set it returns early (c:3873).
  - Writes `consume_order`. Sorts each list by (frame, priority, round, `a->c < b->c`: the address of the `struct call`, offset). Then entry and end are recomputed.
- C: `compile_group` (aliases NULL), `issue_order` 6604. `seconds` flows to `x->seconds` and then the plan's `view.seconds` (6829, 6953, 6962).

**E41** c:3685-3712 `schedule_consumption`, c:3682 `consumption_at` — order/schedule
- D: consumption order sorted by (`k/each`, consume_order, call address, wire index).
- C: `compile_group` 3913, `issue_order_finish` 6589, `schedule_laid` 4112.

**E42** c:4056-4068 `schedule_units` — order
- D: recomputes each `end`, stores `listed`/`laid`; `each[]` is the prefix at `count/repeats`.
- C: `issue_lay` 4090, `place_group` 4397, `issue_order_finish` 6587.

**E43** c:3992-4000 `repeat_of`, c:4074-4093 `issue_lay` — slot/modulo
- D: message `k%n`, frame `k/n`, `offset += frame*stride`, entry and end shifted by the repeat. The consume index is `k/ins*ins + consume[k%ins]`.
- C: `issue_layout` 4881.

**E44** c:4099-4116 `schedule_laid` — positional order
- D: each entry gets `listed+ch->sends/lands`, `end = ch->out/in + laid`, and its step's channel, entry and consumption. This is the "link at the occurrence's own position" in the design note.
- C: `schedule` 4834 → `present_group` (span `through`, zwoggle entries), `commit_group`.

**E45** c:3983-3989 `group_at`, c:4019-4054 `definition_at`/`definition_of`, `struct compiled.period/laid` (c:3962-3963), `struct plan` (c:3973) — slot/modulo
- D: `(holder-first)%issues`, a binary search in the run ends, then `%run.count`. `period==0` ends the program (`library_end`).
- C: `ncclMeshCommLower` 5910, `occurrence_at` 5929, `ncclMeshCommSources` 5940. Related, outside scope: `zorvek` 4982 takes `instance=invocation%x->period` from a cyclic pool of laid issues.

**E46** c:3924-3960 `struct issue`
- Fields record decisions made elsewhere: `base`/`area`/`zeeb`/`bytes` (from `zappo`), `tables`, `progress`, `linking.left`, `afters`/`readers`/`requirements` (waits on other issues' SENDs), `owned`, `instance`/`used`.

## H. Timing

**E47** c:772-797 `watch_turn`/`watch_start` — timing
- D: a resident dispatch timer every `MESH_PLATFORM_NS/20` checks `mesh_stream_held` and calls `comm_lost`, then `bound_reached`, then `library_end`.

**E48** c:1113-1265 bootstrap — timing (control plane only)
- `MESH_PREPARATION_NS` per phase and `MESH_BOOTSTRAP_SLICE_MS` slices, in `setup_left`/`bootstrap_slice`/`whole`/`bootstrap_open`/`exchange`.

**E49** c:1524-1543 `ncclCommDestroy` — timing
- `usleep(20)` loops on `async` and on `runs`, each under a `mesh_deadline`.

**E50** Poll budget: `call.polls` (c:2357), `issue.polls` (c:3930, set at 5764-5781 with the `NCCL_MESH_POLLS` cap) and words[4] — timing
- C: `destination_effects`, `page_effects`, `bound_reached`.

**E51** df:15-19, 82-104, 198-257 `left_of`, `mesh_records_read`, `mesh_nodes_health`, `mesh_map_wait` — timing
- Seqlock waits in `MESH_PAUSE_NS` slices under a `MESH_PLATFORM_NS` deadline.

## I. Diagnostics

**E52** dg:16-146 quiet probe — allocator, slot and timing
- `rounds+1` slots, at `data.first+(k+1)*data.stride` (dg:41); spans with `.through=k+MESH_STREAM_LANES`; `ready=k+1` raised each round; a spin on `landed`.
- `release_probe` (dg:64-66) spins on the bridge's owner word moving, i.e. the bridge leaving the tables, before `mesh_lease_return`.

**E53** dg:169-203 `mesh_diagnostic_retire` — timing measurement
- Polls the owner word for up to 10 s; random sleeps between rounds.

## J. nccl.h:102-146, claims against the code

**E54** What the header describes, and what the code does
- (a) A gate page (nh:104): deleted (E23).
- (b) Two slots per definition, "its groups taking turns at them" (nh:105-109): no such code in scope.
- (c) "eight sizes an octave" and "ticket" (nh:111-114): no code anywhere in rdma. c:4966 says the tickets are deleted.
- (d) "pieces of the region" (nh:114-117): `metal_piece_hold` c:4137-4148, one `ncclMemAlloc` of at least `MESH_COMMAND_DATA_BYTES` per piece, given back at Obj-C dealloc (nccl-mesh-metal.m:58, 387).
- (e) "each generation and each slot (`ncclMeshBufferSlots`)" (nh:119-121): E24-E26 and `call.bindings[2]` (c:2336).
- (f) "writes a slot only once … every landing awaited" (nh:126-127): E44 plus `landing_afters`/`readers`.
- (g) SEND sources reused after landing or DMA-read completion (nh:127-128): E16-E19.
- (h) The logical free (nh:141-145): E6/E7.

Host-only pools, not region memory: `op_make` c:1971-1984; `grow` c:4122-4131; the thread-local `calls` c:2375-2376 (grown at 5913).

## Readers of slot-index formulas

**Inside scope:**
- c:2183 `write_span_index` (`invocation%p->occurrences`), read by c:3772 `schedule_group` and c:4087 `issue_lay`.
- c:2186 `write_packets` (`invocation%p->occurrences`), read by c:2590 `write_steps`.
- c:2193 `write_binding_slot` (`ncclMeshInvocationIndex`).
- c:2197, 2203-2205 `write_slot_extent` (`ncclMeshInvocationIndex` twice, `instance_stride`).
- c:2215 `tile_offset` (`invocation % layout.instances`, `instance_stride`).
- c:2222, 2236-2237 `tile_extent` (`instance_stride`, `(instances-1)*instance_stride`).
- c:3680 `copied_word` (`frame*tiles+tile`).
- c:3705-3706 `schedule_consumption` (`k/each`, `k%each`).
- c:3984, 3988 `group_at` (`%view.issues`, `%run.count`).
- c:3993-3997 `repeat_of` (`k%n`, `k/n*stride`).
- c:4065 `schedule_units` (`count/repeats`).
- c:4091 `issue_lay` (`k/ins*ins + consume[k%ins]`).
- mc:7 `mesh_section.stride` (mesh-call.c:11 `first+index*stride`), read at dg:41 and dg:69.
- c:2336 `call.bindings[2]`, of type `ncclMeshBufferSlots`.

**Outside scope, consuming these:**
- `write_binding_slot`: 4200, 5267, 5652, 6187, 7266.
- `write_span_index`: 4322, 6499, 7243, 7257. `write_packets`: 7244, 7259.
- Direct `ncclMeshInvocationIndex`: 5854, 5971, 6776-6777, 7197, 7430.
- `instance_stride`: 4203, 7197, 7310, 7347-7348, 7431, 7718, 7727, 7731.
- `tile_offset`: 5641, 7696.
- `% x->period`: 4982 (`zorvek`).

## Entries that share state

- **Region bitmaps** (own/hot arena bits; ROW_OWN, ROW_HOT and FREE planes): E1-E4, E10, E11 and every caller in E5.
- **`struct attach`** (leases, `lease_of`, `returning`, `holds[]`, `wrapped`, `stream_buffer`): E5-E9, E12-E14, E19, E20.
- **Hold generation and wire**: E8, E9, E19, E20, `comm_failure`.
- **`lease.reads`** (the ledger): E7, E16-E19, `comm_let_go`.
- **Channel counters** (`sends`/`lands`/`out`/`in`/`linked`/`link`): E21, E44, `commit_group`, `present_group`. `owned.uses[].end` comes from them (E19).
- **`comm->words`**: E22, E47, E50.
- **`write_binding`**: E24-E28. `step_units` feeds `message.pages`, then the prefixes in E39 and E42-E44. The relay lease size comes from E25.
- **`struct steps`**: E29-E31, E39-E41, E44.
- **`struct schedule`**: E39-E44.
- **Routing**: E34-E36 feed E29 (through `call_paths`), then `write_transfers`, then E39/E40.
- **`compiled`/`plan`** (`period`, `laid`, `seconds`): E40, E45, E46.
- **`comm->page`**: E38 feeds E29, E35, E39 and `slot_laid`.

Findings along the way, beyond the inventory itself:
- Two orderings depend on where records sit in memory: c:3819 compares `struct call` pointers, and so does c:3690.
- `mesh_lease_return` doesn't do the retention check its own comment promises (E10).
- Nothing calls `mesh_lease_retire_drained`. Once a dependency is marked invalid, only detachment can release its lease (E18).