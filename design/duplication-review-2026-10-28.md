# Duplication review, 2026-10-28

Two read-only maps of duplicated patterns, made at mesh origin/main 74885bcc–4145d02f. Line numbers are as of those
commits and drift with later ones. Each finding gives its instances, what they share, the merged form, the net lines,
the risk and the rubric trait (design/rubric-actionable.md).

What landed from them, on mesh main:
- one configuration exchange; split and grow carry no collective (a0f63e7e);
- one presentation form (25276460);
- a landing's need found at lay time (328e0dea);
- mesh_compile_view (f143cd49, 29707d01);
- host parts committed together (2abfe74c);
- one extent cutter (39fd0f24);
- one page quantum (51582d01);
- one program compile (f891a3e9);
- a WRITE compared by its bytes (368f60bd).

Later the same night:
- a WRITE has no type (8f1fa367);
- the pointer lowering (design/pointer-lowering-2026-10-28.md);
- the mesh backend registered on import (04dac936);
- one placement proposal on every member (c3451281);
- device-copied attribute import (bb4c7b34).
- Across-files 1, in part (33504a46): `_stream.mm`'s `_c10d_functional` MPS kernels, a run-time copy of
  declarations.py's op mapping, are deleted. A sweep found no hourly workload reaching them on real tensors.
  `ProcessGroupMesh`'s per-call methods, `_issue` and `submit` (with R17's drain) stay, because they carry
  partition's declared configuration transfers (`_configuration_exchange`: `group.configuration` declares the call,
  then `dist.scatter`/`dist.all_to_all_single` issues it). They go when those transfers become prepared plans.

On 2026-10-29:
- Within 4, two forms of lease retention, landed. An issue holds its leases from its placement (c05241a8), and every
  prepared owner uses one form, struct owned (92661101). A present registers nothing.
- A byte's lease is its arena page's entry, and a freed lease is given back without a scan (517ad9e0).
- Across 4, relay landings, is next. The plan:
  - a rank that is neither endpoint passes an empty page table, its communicator's, meaning the library chooses;
  - write_prepare decides from the installed paths (ncclMeshCommWritePaths: weight > 0 through this rank), before the
    binding exchange, whether this rank forwards anything;
  - if it does, it leases the compact relay itself (bytes x max(repeats, instances), pitch row_bytes) and reports that
    relay's phase;
  - the plan holds the relay with its owned form, and write_plan_drop frees it after owned_end;
  - the five frontends listed under across 4 stop allocating relays.
  - Two constraints the code must meet:
    - Phase before the exchange. Every rank's packet cuts come from the phases reported in write_prepare's binding
      exchange (write_view_packets), and they must be identical on every rank. So a library-chosen relay's phase must
      be reported in that exchange: either its lease is taken from local information before it, or it is taken at an
      extent boundary, whose phase is 0 by construction.
    - Its period follows reuse. The frontends size relays by how soon a slot may be reused: mb's columns_prepare one
      per invocation, the fused ring two generations of repeats, the chain count x numel. The library's sizing must
      follow the plan's own reuse rule (its landing gate and generations), not a fixed bytes x max(repeats, instances).
  - ncclMeshPageTableCreate(NULL, comm, ...) can name the library's relay: a table of no pages bound to comm, which
    write_prepare's existing table->comm already requires.

Considered and not landed: within 7 (one WRITE compile: it saved no compiles), within 8's operands_check (it catches
a returned lease at issue), within 12's write_call (definition and issue differ on tile_bytes), within 17–19 (code
shape only).

The largest duplicate is missing from both maps: frontends lowering layout to addresses by hand, and the library
recovering pages from those addresses. design/pointer-lowering-2026-10-28.md inventories it.

## Within nccl-mesh.c


Source: `git -C ~/dox/mesh archive origin/main rdma design docs`, fetched 2026-10-28. Line numbers are those of
D/rdma/nccl-mesh.c (6070 lines). Every file touched is of kind **Collective (C)**. So each ranking below uses the C
dimensions of design/rubric-actionable.md, plus the eager-execution terms and R09/R10/R17 where they apply.
Net line counts are estimates of the whole change, helpers included.

Ranked by value: lines × mandate or performance gain ÷ risk.

| # | Finding | Net lines | Risk | Gain |
|---|---|---|---|---|
| 1 | One configuration all-gather; split/grow off eager collectives; `aside` goes | ≈ −45 | low–med (fixes a fatal path) | mandate (eager), C1, C9 |
| 2 | Dead host presentation and its room guard | ≈ −40 | none (unreachable) | mandate (guard), R17 |
| 3 | `need` re-searched every issue; same walk as `bind_retirements` | ≈ −6 | exact | R09, hot path O(n²)→O(n) |
| 4 | Two forms of lease retention (WRITE scope vs per-message per-issue marks) | ≈ −10 | medium | hot path, C9 |
| 5 | `ncclGroupEnd` threads host parts because `launch_host` blocks | ≈ −10 | low–med | R17, C4, no thread per group |
| 6 | Registration-extent cutting written three times | ≈ −20 | exact (one cap lifted) | C2 |
| 7 | `mesh_write_compile` sized-then-filled ×3, and compiled twice per rank | ≈ −20 | exact | C9, prepare cost |
| 8 | Issue binding: three walks over `struct bound` plus per-issue re-derivations | ≈ −20 | exact / low | R09 |
| 9 | Schedule lists as `list[2]`; fuse the per-issue channel walks | ≈ −12 | exact | hot path |
| 10 | Per-kind ladders → one kind table and one own-segment helper | ≈ −15 | exact | C2 |
| 11 | `present_group` rebuilds every SEND entry per issue | ≈ −5 | low–med | R09, hot path |
| 12 | Four `struct call` builders | ≈ −15 | exact | C2 |
| 13 | `metal_uses` once per entry, in two forms | ≈ −3 | exact | hot path |
| 14 | Two definition builders; WRITE definitions compiled and slotted at issue | ≈ 0 | medium | R09, R10 |
| 15 | Operand-making decisions re-derived every issue | ≈ −10 | low–med | R09 |
| 16 | Entry→group lookups per entry; one-track order simulated per completion | ≈ −5 | medium | R09, hot path |
| 17 | Stream fields round trip and the program bracket | ≈ −12 | exact | none (shape) |
| 18 | Growable-array, copy and program-allocation helpers | ≈ −33 | exact | none (shape) |
| 19 | Failure chains and repeated checks | ≈ −20 | exact | none (shape) |

The total is about −300 lines. About 235 of them come from findings 1–16, which carry a mandate or performance gain.

---

### 1. One configuration all-gather; ncclCommSplit/ncclCommGrow off eager collectives

**Instances**
- `gather` 1174–1189: the init's bootstrap all-gather of `struct report`.
- `prepare_exchange_locked` 4763–4777 and `prepare_exchange` 4778–4786: the configuration all-gather.
- `write_prepare` 5495–5524: inlines `prepare_exchange`'s lock, `runs` and deadline around two `prepare_exchange_locked`
  calls.
- `split` 5938–5949 and 5968–5973: two all-gathers of rows, run as `ncclAllGather` on rows of a leased page.
- `ncclCommGrow` 6059–6068: `ncclBroadcast` of an id from a lease.
- `aside` 5993–6008, called at 6013/6015/6060/6068: exists only so those two collectives run outside the caller's
  group.

**Shared.** Each is the same root-star all-gather. Non-root ranks send their row to rank 0 and receive the table back.
Rank 0 receives each row and sends the table to every rank. `gather` and `prepare_exchange_locked` are line-for-line
the same algorithm.

**Differs.**
- `gather` can skip the first leg when the root already holds the rows (`sent`). It also checks
  `reports[r].rank == r`.
- `prepare_exchange_locked` checks `peers`.
- `split` and `grow` do the same job through the RDMA collective path. That path is an **undeclared (eager) call**.
  `launch_host` → `launch_metal` → `definition_of` finds no declaration (`comm->plan` NULL, or a declared group that is
  not this all-gather), so it calls `library_end`. **ncclCommSplit and ncclCommGrow therefore end the program on every
  call today.**

**Merged form**
```c
/* rows[r * bytes] made every rank's over comm's bootstrap sockets within comm->until, this rank's already in place
   (`heard`: the root already holds every row; `ranked`: the root checks each row's leading int32 is its rank) */
static int exchange(struct ncclComm *comm, void *rows, size_t bytes, int heard, int ranked);
```
- `gather` becomes `exchange(comm, reports, sizeof *reports, sent, 1)`.
- `prepare_exchange_locked` becomes a memcpy plus `exchange(..., 0, 0)`.
- `split` makes two calls:
  - `prepare_exchange(comm, &mine, sizeof mine, every)`
  - `prepare_exchange(comm, &bound, sizeof bound, binds)`

  This drops the `rows` lease, the page stride and the copy loops.
- `ncclCommGrow` gathers the ids and takes `reports[0]`. This is NCCL's own bootstrapSplit, which exchanges over the
  parent's bootstrap.
- With no group call left in `split` and `grow`, `aside` and its four calls are deleted.
- Optional: factor `prepare_hold`/`prepare_let` (lock, `runs`, deadline) so `write_prepare` stops inlining them.

**Net.** About −45: −9 from the gather merge, −14 split, −5 grow, −20 `aside`, +3 helper.

**Risk.** Low–medium.
- Split and grow change from ending the program to working over the bootstrap sockets, under `comm->prepare` and
  MESH_PREPARATION_NS.
- A shrunk communicator has no peers, so its split fails as a value (ENOTCONN).
- `gather`'s rank check must stay inside the receive loop. Checking after the exchange would let the root send the
  table before failing.

**Rubric (C)**
- eager-execution.md "What follows": `split`'s `ncclAllGather` ≺ `prepare_exchange`, since the run-time path that
  serves an undeclared call goes.
- C1: NCCL's bootstrapSplit form.
- C9: one exchange primitive instead of three.

### 2. Dead host presentation and its room guard

**Instances**
- `room_for` 3199–3204 and `room_wait` 3205–3230.
- The `!program` form inside `present_group`:
  - the room loop 3241–3253 (keep the `need > MESH_STREAM_ENTRIES` refusal);
  - `mesh_stream_present` branches at 3317 and 3322;
  - the `program &&` / `if (program)` tests at 3288, 3314, 3316, 3321 and 3324.
- `schedule`'s `if (program)` at 3361.
- Comments that describe the host form: 3189–3191 and 3231–3238.

**Shared.** `present_group` presents one group's entries in two forms: records for the GPU's `present` kernel, and
direct `mesh_stream_present` calls behind a host wait for room.

**Differs.** None of the host form is reachable:
- `present_group`'s only caller is `schedule` (3362).
- `schedule`'s only caller is `launch_metal` (4396).
- `launch_metal` returns at 4365 when `metal_begin`/`metal_begin4` gives no program.
- `room_wait`'s only caller is the dead branch at 3250.

**Merged form.** Delete the host form, so `present_group` has one form. `program` becomes a required argument.

**Net.** About −40. **Risk.** None: the code is unreachable.

**Rubric**
- eager-execution.md Terms: "A guard is a wait inside a call on something other than the call's own input data: room,
  …". `room_wait` is exactly that guard, kept in source after 3e19b5b8 deleted its executor (R17).
- C4/C6: dead form ≺ none.
- For context, and outside this file: the live form of the same room wait is the GPU `present` kernel's wait on
  `taken` (nccl-mesh-kernels.h:204–206). The bridge's ring capacity is R10's retained "hardware queue credit
  accounting".

### 3. `need` re-searched every issue; it is the same walk as `bind_retirements`

**Instances**
- `schedule_laid` 2907–2916. It runs on every issue. For each REDUCE landing it scans every SEND on the same channel
  for an earlier step of the same call that meets it, and takes `need = max(o->entry + 1)`. That is O(ins × outs) per
  channel per issue.
- `bind_retirements` 3143–3166. It runs only when an issue is laid and placed. It walks every landing × every earlier
  SEND of the same call (all channels) to require a witness for overlapping bytes.

**Shared.** The same pair filter: `o->c == m->c && o->step < m->step`, followed by an overlap test.

**Differs.**
- `need` restricts to `q == h`, REDUCE steps, `!(st->read && named(st->read[o->step]))`, and step-range `meet`.
- `bind_retirements` tests byte addresses.
- `need`'s inputs are the definition's steps and the binding's `read[]`. Both are fixed while `issue_same` holds,
  because `steps_rebind` keeps `read`/`fresh`.
- `o->entry = o->listed + p->first[0]`, so `need − first[0]` is fixed per laid issue.

**Merged form**
- Add `uint64_t needs;` to `struct message`: the need relative to the channel's first send entry, 0 for none.
- Set it inside `bind_retirements`' inner loop (plans made non-const):
  `if (q == h && step->op == MESH_STEP_REDUCE && !(st->read && named(st->read[o->step])) && meet(st->steps + o->step, step)) m->needs = max(m->needs, o->listed + 1);`
  Reset it before the `write_plan` `continue`.
- `schedule_laid` keeps only `m->need = m->needs ? m->needs + p->first[0] : 0;`.

**Net.** About −6.

**Risk.** Exactly behavior-preserving. The values are identical. `bind_retirements` runs whenever `schedule` can run
(`remote`).

**Performance**
- Per issue, O(Σ ins × outs) `meet` checks become O(ins).
- Fused groups lay `repeats ×` entries, so they pay the search R² times over. They never read `need`: a fused group is
  completed at its launch, and `complete`/`complete_track` read `need` only for non-fused groups.

**Rubric.** R09 (execution-time construction): the per-issue search ≺ placement-time. C4: less work between issue and
wait.

### 4. Two forms of lease retention

**Instances**
- WRITE form:
  - `write_prepare_spans` 5373–5396: `mesh_lease_retire_begin` on the payload and span leases, `uses[2h+d]` deps built
    once.
  - `ncclMeshWriteSourceReady` 5732–5741: `begin` on the ready table.
  - `present_group` 3267–3269: `uses[].end = m->end.n`.
  - `write_plan_drop` 1883–1904: `mark_use` per lease × use, then `end`.
- Collective form: `present_group` 3257–3281, on **every issue**. It takes `attach_lock` and, per message:
  - `lease_at(address_of(m->data))` and `lease_at(m->spans)`, two linear scans over the process's leases;
  - two `mesh_lease_retire_mark_use` calls, each a linear scan over that lease's deps.

**Shared.** Both register the same thing before presenting: SEND/LAND ends on (stream, wire, holding) for a payload
lease and a descriptor-table lease.

**Differs.**
- The WRITE form resolves leases and opens a registration scope once, then keeps ends.
- The collective form re-resolves and re-marks everything per message per issue. It does so although the leases
  (binding, slot and `u->tables`) are fixed from placement. `place_group` pass 0 already does a `lease_at` per message
  (3114–3117).

**Merged form**
```c
struct retained { void **at; unsigned char *metadata; uint32_t n, channels; struct mesh_lease_retire_dep *uses; };
static int retained_begin(struct ncclComm *comm, struct retained *r); /* lease_at each, begin; uses[] from channels' holds (5387–5392) */
static int retained_end(struct retained *r);                          /* under attach_lock: mark_use × uses, end (1890–1901) */
```
- The WRITE plan holds one `struct retained` (payload, spans, ready). Each issue holds one, made in `place_group` from
  the distinct leases it already looks up.
- `present_group` only stores the last end of each channel list into `uses[]`. That is O(channels).
- `write_plan_drop`, `issue_free`, and a re-place before `ncclMemFree(u->tables)` call `retained_end`.

**Net.** About −10.

**Risk.** Medium.
- A collective's leases are then retained from placement until the issue is freed, instead of until the bridge's
  prefix passes each present's end. This is more conservative: pages the caller frees return when the declaring plan
  is replaced. That is the WRITE plan's semantics already.
- An ESTALE dep then surfaces at release instead of as the present's `comm_fail`.

**Performance.** It removes a global-lock section with O(entries × leases + entries × deps) work from every issue.

**Rubric**
- C9: one retention form instead of two.
- R10: no ownership search at run time. 5884–5885 already states it for WRITEs: "No execution performs an ownership
  search or reuse query."

### 5. `ncclGroupEnd` threads host parts because `launch_host` blocks

**Instances**
- `launch_host` 4538–4568: convert, `launch_metal`, `metal_commit`, then a blocking `gpu_finished`.
- `launch_one` 4573–4579 and `part_run` 4584–4589.
- `ncclGroupEnd` 4684–4699: one `pthread_create` per host part per group, then join.

**Shared.** Each host part of a group on several communicators is the same "commit, then wait" sequence. It runs on a
thread only so the waits overlap.

**Merged form**
- Split `launch_host` at its commit: `host_commit(list, n, void **command)` covers 4539–4561 plus `metal_commit`, and
  `host_finish(comm, command)` covers 4562–4566.
- `ncclGroupEnd` commits every host part in order, then finishes each one.
- `launch_one` becomes `host_commit` followed by `host_finish`.
- `part_run`, `thread` and `threaded`, and the `pthread_create` failure branch go.

**Net.** About −10.

**Risk.** Low–medium.
- No host wait can block between commits. `room_wait` is dead (finding 2). `gpu_finished` at the start of
  `launch_host` waits only on that communicator's own, already-committed program.
- The parts still progress together, because every command buffer is committed before any wait.
- The first failure is still reported in part order.

**Rubric**
- R17 ("the host adapter blocks until that execution finishes"): one wait after all commits ≺ a blocking wait per
  part.
- C4: issue earlier, wait later.
- It removes thread creation from every multi-communicator host group.

### 6. Registration-extent cutting written three times

**Instances**
- `extent_cuts` 2287–2307: absolute element cuts for uniform and V collectives. It has an insertion sort and a unique
  pass.
- `counted_steps` 2357–2393: the same boundary walk (`b = MESH_DMA_EXTENT − phase; b += MESH_DMA_EXTENT`, skip
  non-element offsets), relative to each transfer. It has its own insertion sort and its own "split step at cuts" loop.
- `routed_steps` 2325–2338: the same "split step at sorted cuts" loop.
- `write_view_packets` (5257) is the WRITE form of the same idea, via `mesh_span_residues`. It is not merged here.

**Merged form**
```c
static size_t extent_bounds(uint64_t phase, uint64_t z, uint64_t lo, uint64_t hi, uint64_t *out, size_t n, size_t room); /* element x in (lo, hi), x·z ≡ −phase mod EXTENT */
static size_t cuts_order(uint64_t *cuts, size_t n);                    /* sort + unique */
static uint32_t step_cut(struct mesh_step s, const uint64_t *cuts, size_t n, struct mesh_step *out); /* s's pieces at absolute cuts */
```
After the merge, `routed_steps` and `counted_steps` share their skeleton: allocate first/initialized/logical, fill per
rank, cut, then `bind_routed_steps`.

**Net.** About −20.

**Risk.** Exact, except one cap. `counted_steps` truncates silently at 64 cuts per transfer (`uint64_t cuts[64]`).
Sizing the room as `extent_cuts` does lifts that cap. That differs only past 32 extents per transfer end.

**Rubric.** C2: one cutting rule for even, uneven and counted calls ≺ two. Prepare-time only, so no performance gain.

### 7. `mesh_write_compile` sized then filled ×3, and compiled again per rank

**Instances**
- `write_steps` 2197–2211.
- `order_writes`:
  - 2657–2665 compiles every (call, rank) only to count;
  - 2675–2680 compiles every (call, rank) again to fill.
- `write_prepare_spans` 5320–5334.

**Shared.** Each instance does the paths lookup and builds the `struct mesh_operand`, compiles with `NULL, 0`,
allocates, compiles again, and checks errors.

**Differs.** Only `rank`. For a lone WRITE, this rank's transfers are compiled 6 times in one preparation: twice each in
`write_prepare_spans`, `write_steps` and `order_writes`. Every other rank's are compiled twice.

**Merged form**
```c
static int write_transfers(const struct ncclComm *comm, uint32_t r, uint32_t from, uint32_t to, ncclDataType_t type,
                           size_t count, size_t tile_bytes, struct mesh_transfer **out); /* count, or −errno; *out count+1 */
```
`order_writes` compiles each (call, rank) once and keeps the arrays: size from their counts, then fill.

**Net.** About −20.

**Risk.** Exact. `write_steps`' NULL check on `comm->writes` becomes common. That is harmless:
- `order_writes` runs only after `write_steps` succeeded;
- `fabric_discover` sets the default write routes.

**Rubric.** C9 (one compile entry point); about half the compiles at preparation. It is execution-time only when
`definition_of` recompiles lazily (finding 14).

### 8. Issue binding: three walks over `struct bound` plus per-issue re-derivations

**Instances**
- Three walks over the same binding fields (send, recv, `views[2]`, fused, occurrence/instance, sum, direct,
  complements):
  - `issue_same` 3417–3432 compares them;
  - `issue_bind` 3461–3471 restores them;
  - `issue_bind` 3480–3495 saves them.
- `steps_copy` 2145–2164, `steps_reset` 2167–2173 and `steps_rebind` 2177–2182 are the same reset three times;
  `reset` is `rebind` plus the buffer/read zeroing.
- Per-issue work the declaration already fixes:
  - `launch_metal` 4328–4330 calls `reduction()` for each reducing call. `declared` already computed it (4947), and
    `prepared` computes it again (4970).
  - `operands_check` runs at 4352 on every issue, including when `issue_same` reuses a placement that already
    validated the same buffers.
  - `launch_host` 4549–4557 runs `as_metal` on 2 + 2N wheres per call on every issue. Each is an `attach_lock` plus a
    linear `lease_at`, although the prepared generation holds their Metal form.

**Merged form**
- `static struct bound bound_of(const struct call *c)` for the scalars, plus
  `static int views_kept(struct where **kept, const struct where *v, int n)`. `issue_same`, restore and save each
  become a few lines.
- `steps_reset(s, keep_binding)` replaces the three resets.
- `call_bind_roles` copies the declaration's operator: `combine`, `post`, `unit`, `premultiplied`, `pre` and
  `scalar_at`.
- The issued-path `reduction()` goes. The device-scalar refusal stays as one line in `definition_of`, where the stream
  is known.
- `operands_check` and `as_metal` move to the re-place branch of `issue_bind`. `issue_same` then compares the issued
  form of each where.

**Net.** About −20.

**Risk.** The walk merges are exact.
- Copying the operator from the declaration differs if a `ncclRedOp_t` is destroyed and re-created under the same
  handle with another scalar between the declaration and the issue. `declared_as` compares only the handle.

**Rubric.** R09 ("per-launch metadata"): per-issue recomputation ≺ placement-time. eager-execution.md: a refusal check
at run time is no remedy (C6), and `operands_check` repeats `place_group`'s check on every issue.

### 9. Schedule lists as `list[2]`; fuse the per-issue channel walks

**Instances**
- About 90 references to `out/in/outs/ins`, and 12 `send ? … : …` selections. They are at 2586, 2708–2709, 2871–2872,
  2896–2899, 3091–3092, 3244 and 3263–3264. Meanwhile `first[2]` and `each[2]` are already indexed 0 for sends and 1
  for landings.
- Walks over channel × list × entry:
  - schedule_group 2569–2591, order_writes 2707–2721, issue_lay 2863–2883, slot_laid 3049–3051,
    place_group 3089–3134, bind_retirements 3144–3166;
  - per issue: schedule_laid 2891–2917, gate_of 3176–3186, present_group 3241–3253, 3259–3279 and 3282–3342,
    commit_group 3347–3352, and launch_metal 4450–4451 and 4455–4458.

**Merged form**
- `struct schedule { struct message *list[2]; size_t count[2]; uint64_t first[2], word, base; mesh_pages each[2]; size_t *consume; }`.
- Per issue, two passes replace five:
  - lay, gate and retain per channel;
  - emit records and commit per channel.

  Every channel stays laid before any record is emitted, because dependencies read `s->channel`/`s->entry` across
  channels. Retention stays before any presentation.

**Net.** About −12. **Risk.** Exact: the order of every walk is preserved, sends before landings.

**Rubric.** Hot path: fewer passes over the group's entries per issue (C1, latency).

### 10. Per-kind ladders → one kind table and one own-segment helper

**Instances.** About 25 kind tests:
- `has_root` 398 and `program_of` 630: rooted.
- `reducing` 2025.
- `elements` 2055, `call_segments` 2809, `declared` 4894 (which re-derives `call_segments`), `plan_calls` 2467 and
  `declared_members` 4871: partitioned.
- `call_name` 2076–2079.
- `admit` 4731: rooted.
- "The binding that holds only the rank's own segment" (all-gather send, reduce-scatter receive): `sent_bytes` 2100,
  `extent_cuts` 2295–2298, `partial_split` 2442, `operand_at` 3387, `result_of` 3396–3397, `plan_direct` 4120 and
  4156, `epilogue` 4206, `launch_metal` 4428, `declared_members` 4872.

**Merged form**
```c
static const struct kind { const char *name; unsigned char rooted, reduces, own /*1 send, 2 receive*/, counts /*a rank*/; } KIND[] = {
  [MESH_ALLREDUCE] = {"ncclAllReduce", 0, 1, 0, 0}, [MESH_BROADCAST] = {"ncclBroadcast", 1, 0, 0, 0},
  [MESH_REDUCE] = {"ncclReduce", 1, 1, 0, 0}, [MESH_REDUCE_SCATTER] = {"ncclReduceScatter", 0, 1, 2, 1},
  [MESH_ALLGATHER] = {"ncclAllGather", 0, 0, 1, 1}, [WHAT_COUNTED] = {"ncclMeshAlltoAllV", 0, 0, 0, 2}};
/* the operand elements binding `receive` of c holds: its own segment where it holds that alone, else all */
static uint64_t binding_range(const struct call *c, int receive, uint64_t *length);
```

**Net.** About −15. **Risk.** Exact. **Rubric.** C2 (invariance): fewer branches per kind.

### 11. `present_group` rebuilds every SEND entry per issue

**Instances.** `present_group` 3292–3315 runs on every issue. For each SEND it:
- runs `forwards()` (2545–2551), which scans priors, or reads `s->prior[...]`;
- finds the dependency's message (`received->in + s->entry[dep] − first[1] − (repeats−1−repeat)·ins/repeats`);
- computes `units` by `span_units`, a binary search over packet cuts, or by `pages_of`;
- assembles `{word, base}`.

**Shared.** Everything except two absolute terms is fixed by the laid-and-placed issue:
- the incoming channel's `in` before this group;
- the channel bases.

`base = word(channels[dep_h].in + from->laid − from->pages + units)`.

**Merged form**
- At placement, store per SEND message `int32_t dep_h; uint64_t dep_rel;`, with `dep_rel = from->laid − from->pages + units`.
- `present_group` emits `entry.word = landed_word[dep_h]` and `entry.base = word(channels[dep_h].in) + dep_rel`.
- Keep the WRITE producer word (`tile_offset` with `ready_layout`) per issue. `ready_layout.instances` need not divide
  `occurrences`, so `issue_same` does not fix it.

**Net.** About −5. **Risk.** Low–medium: the repeat arithmetic must be reproduced exactly.

**Rubric.** R09 ("`place_group` descriptor construction" applies equally to `present_group`'s). Hot path.

### 12. Four `struct call` builders

**Instances**
- WRITE-as-broadcast encoding: `write_definitions` 5416–5417, `ncclMeshWriteExecute` 5748–5752 and `declared` 4899.
- Counts → segments/total/count:
  - `declared` 4932–4935: `count = total − seg[me] − seg[n+me]` for counted calls;
  - `enqueue_counts` 5801–5804;
  - `routed` 5853–5857: Σ_{q≠me}, the same value.
- `declared` 4893–4894 re-derives `call_segments` (2807–2810).
- `ncclMeshAllGatherBuffers` 5819–5826 and `ncclMeshReduceScatterBuffers` 5834–5840 are the same body.

**Merged form**
- `static struct call write_call(const struct ncclMeshWritePlan *p, size_t invocation)`.
- `static void call_counted(struct call *c, const size_t *counts, uint64_t *into)`, using `call_segments(c)` for the
  number of counts.
- One `static ncclResult_t buffers_call(int what, …, void *const *views, int receive)` under both public Buffers
  entry points. Those entry points stay.

**Net.** About −15. **Risk.** Exact; `enqueue_counts`' equal-count collapse stays. **Rubric.** C2.

### 13. `metal_uses` once per entry, in two forms

**Instances**
- `launch_metal` 4450–4451: one `metal_uses` call, which opens an autorelease pool and makes a `useResource`, per
  landing entry per issue.
- `publish` 3784–3790: batches of 64, with an `inflight_at` lookup per entry and no deduplication.
- Related: `present_group` 3314 calls `metal_resident` per SEND entry.

**Merged form.** `static void uses_of(struct metal_program *, const struct message *m, size_t n, int write)`. It batches
and drops consecutive repeats. A group's entries name a handful of distinct buffers: slot, send and receive.

**Net.** About −3. **Risk.** Exact: `useResource` is idempotent within an encoder.

**Rubric.** Hot path: O(entries) Objective-C calls become O(distinct buffers).

### 14. Two definition builders; WRITE definitions compiled and slotted at issue

**Instances**
- `ncclMeshCommPlan` 5058–5069: `compile_group`, then `slot_take` × GENERATIONS at declaration, then `prepared`.
- `write_definitions` 5407–5425: `compile_group` only.
- Its lazy recompile at issue: `definition_of` 2830–2832, after `cache_drop`, i.e. after `ncclMeshCommWriteRoutes` or
  `ncclMeshCommForms`.
- `bind_slot` 3054–3062 takes a WRITE definition's slot at its first executions. That is `slot_take(comm, 0)`, which
  allocates a page through `ncclMemAlloc`.
- `prepared` 4963 skips targets, so a WRITE's issues run `issue_make`'s mallocs, `issue_lay` and `place_group` at its
  first GENERATIONS executions. The binding is known at preparation (`plan->source_buffer`/`landing_buffer`).

**Merged form**
- `static ncclResult_t definition_prepare(struct ncclComm *, struct compiled *x, ...)` = `compile_group` + slots per
  generation + `prepared`. `prepared` takes a WRITE's bindings and `invocation = occurrence` from its plan.
- Both builders call it.
- `ncclMeshCommWriteRoutes` and `ncclMeshCommForms` recompile the dropped WRITE definitions on the spot, so
  `definition_of` never compiles.

**Net.** About 0: −8 from sharing, +6 for the configuration-time recompile.

**Risk.** Medium.
- A WRITE prepared before its routes exist now fails at preparation, or ends at its issue, instead of compiling
  lazily.
- WRITE plans hold slots from preparation.
- The intentional reuse `guard` for WRITEs (launch_metal 4394–4395) is unchanged.

**Rubric.** R09 ("unplanned compilation") and R10 (storage at execution) ≺ preparation.

### 15. Operand-making decisions re-derived every issue

**Instances**
- `launch_metal` 4412–4449 decides, per call per issue, whether to:
  - copy the own segment;
  - premultiply or copy the operand, rank by rank;
  - copy into scratch.

  Those choices come from `direct`, `partial`, the views and the binding.
- `complement_copy` 4179–4188 encodes the complement, which `plan_direct` already compiled into triples.

**Merged form.** At re-place, `bind_calls` compiles each call's operand operations into the same kind of triple list as
`complement`, with a kind field (copy or premultiply) and a phase (before the gate, or after publication). Both phases
are encoded by one loop, which replaces `complement_copy`.

**Net.** About −10. **Risk.** Low–medium: the dispatch order must be kept. **Rubric.** R09.

### 16. Entry→group lookups per entry; one-track order simulated per completion

**Instances**
- `inflight_at` (3722) is a linear search of the groups in flight. It runs for every entry in:
  - `publishable` 3745;
  - `channel_publication` 3759;
  - `publish` 3788;
  - `complete_track` 3872 and 3878;
  - `complete` 4012 and 4026.
- `complete_track` 3862–3886 re-simulates, at every completion, the serial order of a lone group. That order is a
  function of its schedule. The prefix that `issue_group` 4049–4050 can publish at issue is likewise fixed: it is the
  sends before the first send with an unconsumed prior.

**Merged form**
- A cursor `(g, at)` that moves to `g->next_group` only past `g->end[h][l]`, shared by these walks.
- Optionally, the lone-group order compiled at placement.

**Net.** About −5.

**Risk.** Medium. Equivalence across several groups in flight holds only because `issue_group` already published each
group's free prefix. Check before compiling the order.

**Rubric.** R09; hot path.

### 17. Stream fields round trip and the program bracket

**Instances**
- `admit` 4711–4716 unpacks an `ncclMeshStream` into call fields; `ncclGroupEnd` 4662–4664 packs it back.
- The group polls maximum is computed twice: `launch_metal` 4318–4321 and `ncclGroupEnd` 4662–4663.
- Choosing `metal_begin4` or `metal_begin` is written twice: `launch_metal` 4362–4364 and `ncclMeshComplete` 4503–4505.
- The completion tail (walk, retire, `program_end`, `walk_failed`) is written twice: `launch_metal` 4471–4481 and
  `ncclMeshComplete` 4510–4513.

**Merged form**
- Keep a `const ncclMeshStream *` (copied) in `struct call`.
- `polls_of(list, n)`.
- `program_begin(&operands, &stream, residency, violation, &message)`.

**Net.** About −12. **Risk.** Exact. **Rubric.** None (code shape).

### 18. Growable-array, copy and program-allocation helpers

**Instances**
- Growable arrays could use `grow` (2924):
  - `lease_room` 768–775;
  - `enqueue` 4745–4750;
  - `spans_add` 4087–4092.
- A `void *dup_of(const void *, size_t)` copy helper fits:
  - `program_copy` 365–373;
  - `ncclMeshCommWriteRoutes` 597–601;
  - `declared_members` 4855–4864;
  - `write_view_validate` 5218–5220;
  - `write_prepare` 5542–5547.
- `compile_entry` 381–392 and `collective_steps` 2408–2418 both allocate a program's four arrays from
  `mesh_compile_room`, compile, and free on failure. That suggests one `program_compile(trees, what, root, whole,
  contributors, recipients, members, out)`, calling `mesh_compile_members` or `mesh_compile` by `members`. Those two
  differ in `mesh_compile_sets`' last flag.

**Net.** About −33.

**Risk.** Exact. Only the initial growth constants change: 64 for leases, 16 for calls.

**Rubric.** None. This is the largest pure line win and carries no mandate or performance gain.

### 19. Failure chains and repeated checks

**Instances**
- `definition_of` 2833–2852 builds three `char why[]`/`snprintf`/`library_end` blocks. A variadic `library_end` would
  replace them.
- The revoked text appears twice: `revoked_fail` 1079 and `comm_failure` 1754.
- "Every channel's stream still held" is checked three times: `watch_turn` 699–700, `comm_failure` 1756–1760 and
  `gpu_wait` 3562–3563.
- The same realize failure message appears twice: `comm_init` 1334–1335 and `split` 5985–5986.
- `bootstrap_child_id` 1008–1021 and `ncclGetUniqueId` 1030–1037 both pick a fabric address or the hostname, listen,
  and format the id.

**Net.** About −20. **Risk.** Exact. **Rubric.** None (C6 weakly: one failure text each).

---

## Per-issue work the declaration already fixes (R09 index)

| Where | What | Finding |
|---|---|---|
| `schedule_laid` 2907–2916 | O(ins×outs) `need` search, also for fused groups that never read it | 3 |
| `present_group` 3257–3281 | `attach_lock` + `lease_at` ×2 + `mark_use` ×2 per message | 4 |
| `present_group` 3292–3315 | dependency search, `span_units` binary search per SEND | 11 |
| `launch_metal` 4328–4330, 4352; `launch_host` 4549–4557 | operator, operand check, host→Metal lease scans | 8 |
| `issue_bind` 3451–3471 | calls memcpy, segments copy, `issue_same` memcmp, `steps_rebind` memsets | 8 |
| `launch_metal` 4412–4449 | operand copy/premultiply decisions | 15 |
| `launch_metal` 4450–4451; `publish` 3784–3790 | `metal_uses` per entry | 13 |
| `launch_metal` 4454–4458 | fused publish/land counts by `channel_publication`/`own_landing` scans, fixed per definition | 9/16 |
| `definition_of` 2830–2832; `bind_slot` 3054–3062; `prepared` 4963 | WRITE definition compile, slot (`ncclMemAlloc`), and placement at first executions | 14 |
| `complete_track` 3862–3886 | lone-group order simulated per completion | 16 |
| `ncclGroupEnd` 4684–4699 | a thread per host part per group | 5 |
| `split` 5947/5970, `ncclCommGrow` 6064 | undeclared collectives, which end the program (`library_end`) | 1 |

## Considered and not proposed

- **Deleting `issue_same`/re-place for undeclared bindings.** eager-execution.md schedules this after every caller
  declares bindings. mesh_mpi without `Plan` and unplanned ProcessGroupMesh still reach it. Finding 8 only merges its
  walks.
- **The reuse `guard` on WRITE groups.** It is documented as intended (4394–4395), so this review leaves it alone.
- **Precomputing a WRITE's producer-ready word.** `ready_layout.instances` need not divide `occurrences` (finding 11).
- **The GPU `present` kernel's wait on `taken`.** It lives in nccl-mesh-kernels.h, outside this file's scope, and is
  the bridge ring's credit (R10 keeps it).
- **The public `ncclX`/`ncclXConfig` pairs.** These are public entry points, so they stay.


## Across files and frontends


Read-only analysis. Sources: the mesh repo at origin/main 4145d02f, extracted to
`scratchpad/dup` (D). metal-microbench is at origin/main e251d98a (`mesh_rank.m`, `mesh_rank.h`, `engine_recording.swift`,
plus `decode_crossings.swift` where it is the other half of a pattern). Line numbers are from those revisions. Nothing
was built or run, so every behaviour claim below comes from reading the code.

Kinds (rubric-actionable.md): **C**: nccl-mesh.c, mesh-*.c/h, nccl-mesh-metal.m, metal-recording.m/h,
mesh-tensor-layout.m, _stream.mm, backend.py, collective_views.py, mesh_mpi.py, assignment.py, mesh.py. **D**:
mesh_rank.m, engine_recording.swift, decode_crossings.swift, partition.py, dnn.py, declarations.py.

Value = net lines × mandate/performance gain ÷ risk. Net line figures are estimates of the code after the merge
(− removed, + added). They include no docstrings unless a docstring is deleted along with its function.

## Ranked summary

| # | Finding | Net lines | Risk |
|---|---|---|---|
| 1 | Per-call frontend paths that restate the declaration | about −380, after bindings are declared | med-high |
| 2 | One WRITE adapter for a view descriptor; one WRITE-group execute | −42 | low |
| 3 | Declared and issued call derived twice (declared_as divergence) | −15 | medium |
| 4 | Relay landings chosen and allocated by every frontend | −3 (memory and D7 are the gain) | medium |
| 5 | Python ncclMeshCall marshalling and plan-matrix queries | −27 | low |
| 6 | Crossing lowering: domains, agreed cuts, fragment counts | −27 (a −20 / −7 split) | low / med-high |
| 7 | Slot geometry and recorder words stated four ways | −3 (branches and one convention) | low-med |
| 8 | Channel count rebuilt from cables (D7) | −7 | low |
| 9 | Plan arithmetic restated outside mesh-plan.h / mesh-span-plan.h | −14 | low |
| 10 | Residency set with three owners | −9 | low-med |
| 11 | Producer-envelope division, two different rules | −5 | medium |
| 12 | Lease-coverage checks and retire pairs | −15 | low |
| 13 | Deadline helpers | −7 | low |

Below threshold, or within one file only, at the end.

---

### 1. Per-call frontend paths that restate the declaration (mandate, performance)

**Instances**
- torch-mesh, Python: `ProcessGroupMesh` collective methods, backend.py 335-434. They build a call tuple at every
  call and go through `_issue` 197-203. The `merge_partials` torch op, backend.py 518-526, leases `out` at every
  call. `partition.reduce_scatterv`, partition.py 398-411, leases and copies `y` and leases `out` at every call.
- torch-mesh, C++: `issue` 428-436, `prepare` 310-366, `encoded` 369-403, `submit` 405-426, `on_stream` 168-189,
  `complete` 191-208 and `MeshWork` 212-242 in _stream.mm. Every call re-derives its operand types, counts, MTLBuffer
  bindings, the bool/complex rules and the op.
- torch-mesh, `functional` and its op table, _stream.mm 478-560. These map the `_c10d_functional` ops to calls a
  second time; declarations.py `lower` 94-196 already maps them AOT. For example, the `all_to_all_single` counts are
  at _stream.mm 538-552 and again at declarations.py 160-165, and the `reduce_scatter_tensor` divisibility check is at
  530-534 and again at 155-158. `functional` also leases its result at every call (`region_empty`, 516/523/533/550)
  and calls `ncclCommCount` at every call.
- mesh_mpi.py collective methods 342-412. `Reduce` leases a target at every call (`empty`, 348), and so do `Barrier`
  (404) and `allreduce` (409). `Allgatherv` copies with `to_region` at every call (364). Each call also runs `kind()`
  and `segments()`.
- mesh_rank.m `MeshRankPlan` 893-973 declares every `ncclMeshCall` without `.bindings` (900-901, 939-940), although
  the fragment bindings are fixed by `merge_class` 709. So every prepared crossing is laid and placed at its issue
  (nccl-mesh.c `issue_bind` 3440-3500, once per generation and per distinct offset), and `issue()` 344-375 rederives
  the NCCL arguments at each invocation.
- The library compares every issue back against its declaration: `definition_of`/`declared_as`, nccl-mesh.c
  2813-2858, where a mismatch reaches `library_end`; and `issue_same`, 3417-3438.
- declarations.py `Function.__call__` 541-554 rebuilds a DTensor (`from_local`) per gradient at every call, although
  the shape, placement and stride are fixed by the declaration.

**What they share.** Each issue restates fields that `ncclMeshCommPlan` already holds: kind, type, count, op, root
and buffers. The library then checks them again. `Collectives` (_stream.mm 760-803), `ViewWrites` and
`MeshRankPrepare` already hold the same data, prepared once.

**Single form.** This is design/eager-execution.md "What follows": callers declare their bindings, then the run-time
paths are deleted, leaving the prepared functions (`Collectives`, `ViewWrites`, `Executions`, `MeshRankPrepare`) as
the only issuers. mesh_rank needs its fragment bindings in `ncclMeshCall.bindings` first. That works now for
fragments whose `CrossBinding.stride` is 0; strided slots also need finding 7's slot geometry in the binding.

**Net lines.** backend.py −100; _stream.mm −180; mesh_mpi.py −90; partition.py −15; mesh_rank.m +6.
declarations.py −4 if the gradient wrappers are built once at `declare`.

**Risk.** Medium-high. It removes API surface that eager callers use, so it has to follow the bindings migration.
The library already ends the program on undeclared groups (`definition_of`), so these paths only issue declared
groups a second time.

**Rubric.** For C: C4 and C5 (decisions at the declaration, not the call) and C6 (`declared_as` mismatches end the
program). For D: D2 (choices made earlier) and D8 (the per-call leases in `merge_partials`, `reduce_scatterv`,
`functional` and the mesh_mpi methods).

---

### 2. One WRITE adapter for a view descriptor; one WRITE-group execute

**Instances**
- Library: three public adapters over one `write_prepare`. They are `mesh_compile_spans` 5655-5663,
  `mesh_compile_rows` 5703-5712 and `mesh_compile_write` 5713-5721 in nccl-mesh.c. `write_prepare` already
  recognises rows in a span list (`write_view_rows` 5235-5251), so the frontends' choice of adapter adds nothing.
- mesh_mpi.py: `Write_view_prepare` 325-340 branches on rows versus spans after calling `mesh.view_layout`. Its rows
  branch calls `Write_rows_prepare` 295-311, which calls `_row_pitch` 101-116, which calls `mesh.view_layout` a
  second time to recover the pitch the first call returned. `_write_spans_prepare` 313-323 is used by
  `Write_view_prepare` alone.
- _stream.mm `ViewWrites::prepare` 1669-1698 has the same rows-versus-spans branch. Its two storage-bound
  TORCH_CHECKs (1681-1682, 1688-1690) restate `write_prepare`'s lease-extent check, nccl-mesh.c 5475-5486.
- _stream.mm `fused_writes_impl` 1515-1588 has three branches. Its tile branch spells a contiguous write as rows
  `(nbytes, 1, nbytes)`, because `mesh_compile_write` takes no layout (1561-1562). Its checks at 1519-1536
  (`framed`, repeats, stride, rows, row_bytes) and 1545-1549 (landing count and type) restate `write_prepare`'s
  EINVAL/EOVERFLOW checks at 5444-5471 and the cross-rank EPROTO comparison at 5498-5505.
- mesh_rank.m `columns_prepare` 815 (rows).
- WRITE-group execute (`ncclGroupStart`, `ncclMeshWriteExecute` × n, `ncclGroupEnd`): mesh_rank.m `columns_issue`
  839-845; _stream.mm `prepare_fused_writes` block 1107-1116 and `ViewWrites::encode` 1701-1706.
- The producer side: operand.py `TensorView.transfer` 123-135 branches between `mesh.view_layout` (C,
  nccl-mesh.c 5665-5701) and `layout.ordered_spans()` (`-[MetalTensorLayout orderedSpans]`, mesh-tensor-layout.m
  577-590). Both produce the logical-order runs of a view the same way: the trailing dense dimensions as one run, an
  odometer over the others, and adjacent runs merged.

**Single form.** Add one adapter to mesh-write-compiler.h that takes exactly what `mesh_view_layout` returns:
```c
ncclResult_t mesh_compile_view(const void *source, void *landing, int64_t shift, const uint64_t rows[3],
    const ncclMeshSpan *spans, size_t nspans, int source_rank, int destination, ncclComm_t comm, int metal,
    size_t max_packet, struct mesh_write_layout layout, ncclMeshWritePlan_t *out);
```
It uses rows when `rows[1]` is set and spans otherwise; `mesh_compile_rows` and `mesh_compile_spans` become its two
static cases. Add `ncclMeshWritesExecute(const ncclMeshWritePlan_t *plans, size_t n, size_t invocation,
cudaStream_t stream)` to nccl.h. Optionally, `orderedSpans` can call `mesh_view_layout` for affine layouts, and
`TensorView.transfer` then becomes one native `layout.transfer(element_size)` call.

**Net lines.** mesh_mpi.py −30 (`_row_pitch` and `_write_spans_prepare` go; `Write_rows_prepare` becomes a byte-count
check plus `Write_view_prepare`). _stream.mm −40 (−18 branch and bounds in `ViewWrites`, −15 restated checks in
`fused_writes_impl`, −7 execute bodies). mesh_rank.m −4. nccl-mesh.c +23. mesh-write-compiler.h and nccl.h +9.
operand.py −7 and mesh-tensor-layout.m +4 for the optional part. Total about −42.

**Risk.** Low. Error text moves from TORCH_CHECK to libnccl-mesh. The torch check against storage bounds is
stricter than the library's check against the lease, because a lease can outlast its page-rounded storage; that
difference is the only behaviour delta.

**Rubric.** C9 (one primitive, offered as a header); D3 (three branch sites gone); D4.

---

### 3. Declared and issued call derived twice (divergence ends the program)

**Instances**
- torch, declaration (Python): `backend.plan` 228-264 with `typed()` 64-69, the bool sum/prod→max/min rewrite
  232-233 and `counts × per` 234/238. `_binding` 87-94 computes the binding key.
- torch, issue (C++): _stream.mm `Group::moving` and `Group::reducing` 284-300 restate the same complex, bool and
  bytes-as-uint8 rules. `prepare` 310-366 restates `counts × per` at 344, 361 and 362-365. `Group::where` 272-277
  restates the binding key that `_binding` computes.
- torch: `backend.prepare_fragments` 437-450 encodes one call twice, as a `Call` namedtuple for `plan` and as an
  issue tuple for `Collectives`.
- native: mesh_rank.m `MeshRankPlan` 936-957 switches on the crossing kind to build an `ncclMeshCall`, and `issue()`
  354-371 switches on the same kinds to build NCCL arguments (`pieces × unit`, `elements × element`, uint8 for
  gathers and broadcasts).
- Both pairs must agree field by field with nccl-mesh.c `declared_as` 2813-2820. A mismatch reaches `library_end`
  2833-2852 and the program ends. A binding-key mismatch silently re-lays and re-places (`issue_same`).

**Single form.** Build each call's `ncclMeshCall` (with its bindings) once, and issue it through one library
switch:
```c
/* nccl.h, NCCL vocabulary: dispatches to ncclAllReduce … ncclMeshAlltoAllV with the call's bindings */
ncclResult_t ncclMeshCallIssue(const ncclMeshCall *call, ncclComm_t comm, cudaStream_t stream);
```
In torch, the C++ derivation is exposed as `_stream.declared(call) -> dict`, which `backend.plan` uses, and `Issue`
holds that record. In mesh_rank, `CrossFragment` stores the `ncclMeshCall` that `MeshRankPlan` builds, and `issue()`
calls `ncclMeshCallIssue`. Calls with list bindings (`*Buffers`) keep their own issue until `ncclMeshCall` carries
per-rank views.

**Net lines.** nccl-mesh.c +30; nccl.h +3; _stream.mm −21; backend.py −15; mesh_rank.m −12. Total about −15.

**Risk.** Medium. The record grows (list bindings). The gain is that declared and issued fields can no longer
diverge, more than the line count.

**Rubric.** C6 (a divergence ends the program); C9; D4.

---

### 4. Relay landings chosen and allocated by every frontend (D7, memory)

**Instances.** Each site loops over (source, destination) pairs. It uses the source view at the source and the
final view at the destination; on any other rank it allocates a compact relay lease and compiles a WRITE with it.
- mesh_rank.m `columns_prepare` 781-837. The relay branch is 804-813: `ncclMemAlloc(bytes × invocations)`,
  instances = invocations, pitch = row.
- _stream.mm `fused_allocate_ring` 1215-1256. The relay branch is 1239-1246: `region_empty(repeats × counts)` for
  each of 2 generations.
- _stream.mm `prepare_chain` 1371-1385. The relay branch is 1373-1377: `region_empty(count × numel)`.
- backend.py `bind_gather_views` 453-476. The relay is at 473-474 (`empty(segment bytes)`).
- partition.py `_follow_bytes` 713-790. The relay is at 767-769 (`_empty(run.bytes)`).

**How they differ.** In sizing (repeats, invocations or 1), in allocator (ncclMemAlloc, region_empty,
torch_mesh.empty, _empty), and in whether the transfer is rows or contiguous.

**What is wrong.** Every non-endpoint rank allocates a relay for every pair, whatever the installed weights send
through it. Default routes enumerate every simple path (mesh-write-routes.h 77-122), so a relay is used by default.
A solver weight of 0 leaves it unused but still allocated. The relay's phase also adds cuts to every rank's common
packet plan (`write_view_packets` reads every participant's phase). The library already owns relay storage for
routed collectives (mesh-plan.h 199-207: "separate, compact, uninitialized leases").

**Single form.** In nccl-mesh.c `write_prepare`, a rank that is neither endpoint passes `landing = NULL`. The
library counts that rank's transfers with `mesh_write_compile`. With none, it reports phase 0 and holds nothing. With
some, it allocates the compact relay itself (`bytes × max(repeats, layout.instances)`, pitch = row_bytes,
instance_stride = bytes) and frees it in `write_plan_drop` after the retire ledger ends. The signatures stay as they
are; the change is documented in mesh-write-compiler.h. A later step can size the relay to the fragments its paths
forward, as `mesh_routed_compile` does.

**Net lines.** mesh_rank.m −10; _stream.mm −12; backend.py −1; partition.py −2; nccl-mesh.c +20;
mesh-write-compiler.h +2. Total about −3.

**Risk.** Medium. The relay's lifetime moves into the plan. The off-path phase report changes the cuts, but it
changes them identically on every rank, because every rank derives path membership from the same fabric and paths.

**Rubric.** D7 (frontends know less of routes; they stop knowing that they are relays); D8; C9.

---

### 5. Python ncclMeshCall marshalling and plan-matrix queries

**Instances**
- backend.py `plan` 213-269: ffi keep-alive arrays, rank lists turned into bit masks (240-249), the bindings array
  (260), the default order (217-218), the `ncclMeshCommPlan` call and the entries list.
- assignment.py `declare` 8-38 marshals the same fields and derives the order from consecutive groups.
  mesh_mpi.py `Plan` 244-253 delegates to it.
- Directed-matrix queries, three loops over n×n turned into dicts: backend.py `plan_demands` 286-297,
  assignment.py `function_demands` 41-51 and `route_resolution` 54-60. A native fourth is mesh_rank.m
  `routing_report` 752-777.
- Rank list to bit mask: backend.py 241-249 and mesh_rank.m 314-315.

**Single form.** In rdma/mesh.py, add `plan(comm, calls, order=None) -> entries`. It takes calls as dicts with
Python lists, rank lists for masks, and (buffer, offset) pairs for bindings, and it owns the keep-alive. Add
`plan_matrix(comm, nodes, units=False) -> {(a, b): n}` as well. backend.plan keeps only its torch conversions
(after finding 3, `_stream.declared`).

**Net lines.** mesh.py +35; backend.py −30; assignment.py −32. Total about −27.

**Risk.** Low.

**Rubric.** C9; D4.

---

### 6. Crossing lowering: domains, agreed cuts, fragment counts (torch and native)

**Instances**
- torch: collective_views.py `_domains` 14-28 and `lower` 70-131. `lower` covers the all-to-all matrix exchange and
  pairing check (78-86), the cut union (110) and the fragment V-counts by overlap (117-128).
- backend.py `exchange_configuration` 317-325 and `union_cuts` 327-333. These run through the c10d store as JSON:
  a second configuration channel beside the library's.
- native: mesh_rank.m `cross_domains` 378-394 and `MeshRankCrossLayouts` 396-404. The cuts are in `mark_records`
  594-599 and `merge_records` 639-640, and the union rides the `ncclMeshCommConfigGather` JSON in `MeshRankPlan`
  856-869. The fragment counts by overlap are in `cross_fragments` 657-683.
- The library's own domains: nccl-mesh.c `segment_at` 2059-2062, `call_contributes`/`call_receives` 2034-2043 and
  `declared_members` 4849-4890. `prepare_counted` 4821-4843 does the same matrix exchange and pairing check as
  collective_views 78-86.

**What they share.** Each derives per-rank send and receive intervals from kind, counts, root and the
contributor/recipient masks. It then builds a `MetalTensorTraversal` (shared already), takes its cuts together with
{0, total}, unions them across ranks, and cuts the call into fragments, recomputing V counts by
`max(0, min(last, end) − max(first, start))`.

**How they differ.** Torch counts elements; native counts pieces (`counted ? pieces/element : pieces × elements`) and
bytes for gathers. Torch has ALLTOALL; native has holders and scatter.

**Single form.** In two steps.
- (a) Low risk. Torch's union and matrix exchange go through `ncclMeshCommConfigGather`, exposed as
  `mesh.config_gather`, and the matrix comes from `prepare_counted`'s exchange. The c10d-store side channel goes.
- (b) In nccl.h, preparation only:
  ```c
  ncclResult_t ncclMeshCallDomains(ncclComm_t, const ncclMeshCall *, int rank, uint64_t (*send)[2], size_t *sends,
      uint64_t (*receive)[2], size_t *receives, uint64_t *total);
  ncclResult_t ncclMeshCallCut(ncclComm_t, const ncclMeshCall *, uint64_t first, uint64_t end, ncclMeshCall *fragment,
      size_t *counts);
  ```
  These are built on the static helpers above. Both frontends build an `ncclMeshCall` first; mesh_rank already does,
  in `MeshRankPlan`.

**Net lines.** (a) backend.py −14, collective_views.py −9, mesh.py +8, nccl.h/nccl-mesh.c +5: about −10 here, about
−20 counting the removed store protocol. (b) collective_views.py −16, mesh_rank.m −20, nccl-mesh.c +35, mesh.py +4:
about −7 net.

**Risk.** (a) Low. (b) Medium-high, because of units (elements, pieces and bytes) and the ALLTOALL matrix.

**Rubric.** C7 (every member derives the domains one way, from the declaration); C9; D4; D7 for (a), the torch
frontend's own configuration channel.

---

### 7. Slot geometry and recorder words stated four ways

**Instances**
- The recorder formula `offset + (instances ? t % instances : t) * stride`: metal-recording.h `MetalInvocationOffset`
  30-32, used at metal-recording.m 475/483, 1034, 1144, 1157 and 1175, and at mesh_rank.m 1032.
- The WRITE layout's `(i % instances) * instance_stride` with instances ≥ 1: nccl-mesh.c `tile_offset` 1846-1849,
  `write_prepare_spans` 5368, the phases in `write_prepare` 5524-5526, and the extents at 5475-5476 and 1867.
  `!layout.instances` is EINVAL at 5447.
- mesh_rank.m converts the recorder's convention to the WRITE one: `stride ? (instances ?: invocations) : 1` at 802,
  819 and 827, and instances = invocations at 807.
- mesh_rank.m `MeshRankRecord` 542-556 has three scans that find a buffer's MetalOperand stride and instances. They
  duplicate the recorder's per-buffer association, metal-recording.m 1920-1925, which the recorder reads at 766,
  840-841 and 1105-1107.
- The landing record `8 + capacity × sizeof(ncclMeshBinding)` with its count word: decode_crossings.swift 99-104,
  _stream.mm 1555 and 1584-1588, mesh_rank.m 824-828, and nccl-mesh.c `ncclMeshWriteLandings` 5758.
- Access-word encode and decode: metal-recording.m 473, 851-861, 1167-1168, 1195-1196 and 1417-1418; _stream.mm 733
  and 818.

**Single form.** metal-capabilities.h is plain C, which C, ObjC, Swift and the Python extension already read. It
takes `mesh_slot_offset(offset, stride, instances, t)`, `mesh_access_mode(access, j)` and
`mesh_access_bits(mode, j)`. The WRITE layouts adopt the recorder's convention (instances 0 means a slot per
invocation), normalised once at `write_prepare` entry. metal-recording.h gains
`MetalOperandSlots(id<MTLBuffer>, NSUInteger *stride, NSUInteger *instances)`, which reads the association. nccl.h
gains `ncclMeshLandingRecordBytes(capacity)`.

**Net lines.** mesh_rank.m −10; metal-recording.m +6/−2; metal-capabilities.h +8; nccl-mesh.c 0; _stream.mm −3;
decode_crossings.swift −1; metal-recording.h 0. Total about −3, with four ternaries and one convention fewer.

**Risk.** Low-medium (the instances-0 normalisation and `tile_extent`).

**Rubric.** C9; D3. **Mandate link.** `ncclMeshCall.bindings` carries no slot geometry, so mesh_rank cannot declare
its per-invocation crossing bindings (finding 1).

---

### 8. Channel count rebuilt from cables (D7)

**Instances.** _stream.mm `physical_channels` 1075-1083, used at 1093 and 1176. mesh_rank.m `MeshRankChannels`
439-446, used by decode_crossings.swift 85. partition.py 745 (`len(group.fabric.adjacent[rank])`). The library's
count is `comm->count`, nccl-mesh.c `fabric_realize` 1277-1286, and it counts only the cables whose endpoint is
held. The frontends count every cable that touches the rank.

**Single form.** `ncclMeshCommChannels(ncclComm_t, int *channels)` in nccl.h, returning `comm->count`.

**Net lines.** nccl-mesh.c +5; nccl.h +2; _stream.mm −8; mesh_rank.m −6 (it keeps a one-line wrapper or the Swift
caller calls nccl.h); partition.py 0. Total about −7.

**Risk.** Low. Kernel specialisation reads this count, and it changes only where a cable is not held.

**Rubric.** D7 (frontends stop reading cables); C9.

---

### 9. Plan arithmetic restated outside mesh-plan.h / mesh-span-plan.h

**Instances**
- The page quantum `page / gcd(page, element)`: mesh-collective.c `mesh_quantum` 559-564, mesh-fabric.c 100-102
  (`mesh_fabric_refine`) and 179-181 (`mesh_write_compile`), nccl-mesh.c `plan_routes` 5099-5100.
- Registration-extent cuts of a piece. Both instances step `b = EXTENT − phase; b += EXTENT`, keep only
  element-aligned values, then insertion-sort and dedupe: nccl-mesh.c `extent_cuts` 2287-2307 and the inner loop of
  `counted_steps` 2366-2389. mesh-span-plan.h `mesh_span_residues` 59-105 has the sort and dedupe tail (97-103).
- `extent_cuts` 2297 sums the segments before rank r by hand, which `mesh_segment_first` already does.

**Single form.** `static inline uint64_t mesh_page_quantum(uint64_t page, uint32_t element)` in mesh-plan.h, beside
`mesh_segment_first`. In mesh-span-plan.h, `mesh_span_extent_cuts(phase, first, elements, element_bytes, out, room)`
and `mesh_span_cuts_unique(out, n)`; the second is shared with `mesh_span_residues`.

**Net lines.** mesh-collective.c −5; mesh-fabric.c −4; nccl-mesh.c −22; mesh-plan.h +3; mesh-span-plan.h +14. Total
about −14.

**Risk.** Low.

**Rubric.** C9.

---

### 10. Residency set with three owners

**Instances**
- nccl-mesh-metal.m: `metal_resident` 178-182 and `record4` 217-219 add every bound buffer to the crossing's
  residency, which is the schedule's own set (metal-recording.m `MetalScheduleCross` 1459-1469 passes
  `s->residency`). `metal_end` 621 then commits it.
- metal-recording.m adds at 1400, 1421 and 1488. 1488 copies a replay's residency into the schedule's set.
  `MetalScheduleFinishPart` 1767-1769 commits the same set again.
- mesh_rank.m `crossed_make` 504-507/516 and `merge_class` 724-728 add the crossing buffers to `replay->residency`
  and commit it. The library's `record4` already makes those buffers resident, and 1488 copies the replay's set
  anyway.

**Single form.** The schedule's set has one owner, metal-recording.m. The library adds to it (`metal_resident`) and
FinishPart (1769) commits it. mesh_rank's adds and commits go. `metal_end` keeps its commit only for a classic, unborrowed
set.

**Net lines.** mesh_rank.m −10; nccl-mesh-metal.m +1. Total about −9. One residency commit fewer per crossing during
preparation.

**Risk.** Low-medium. Check that the indirect (ICB) form never runs crossing buffers inside the ICB; crossings are
encoded directly.

**Rubric.** C9; on the D side (mesh_rank), D7, since it knows less of the library's buffers.

---

### 11. Producer-envelope division, two different rules

**Instances.** backend.py `plan` 219-225 counts `producer_groups` per root, once per declared group and block
repeat, and applies them at 256-258. mesh_rank.m `MeshRankPlan` 879-892 counts per mark, per source with a nonzero
piece, times invocations, and applies them at 918-921. Both then call nccl-mesh.c `ncclMeshProductionTimes`
5117-5126. A class with several tiled marks divides the same envelope differently in the two frontends.

**Single form.** `ncclMeshPlanProductionTimes(ncclMeshCall *calls, int count, const ncclMeshPlanRun *order,
size_t runs, const double *seconds)` in nccl-mesh.c, with one rule over the declared WRITE calls.

**Net lines.** nccl-mesh.c +20; backend.py −10; mesh_rank.m −15. Total about −5.

**Risk.** Medium. The rule changes for one frontend.

**Rubric.** C5 and D6 (one estimator input); C9.

---

### 12. Lease-coverage checks and retire pairs

**Instances**
- "A live lease of this attach covers [address, address + extent)", each under `attach_lock`: nccl-mesh.c
  `place_group` 3114-3119, `write_prepare` 5481-5486, `ncclMeshPageTableCreate` 5574-5588 and
  `ncclMeshWriteSourceReady` 5731-5738.
- Retire begin/mark/end over the payload and metadata leases: `write_prepare_spans` 5377-5397, `write_plan_drop`
  1884-1903 and `present_group` 3258-3279.
- Cross-file: the address-to-lease-origin step is in `tensor_pages` 5645-5653, in mesh-tensor-layout.m
  `MetalTensorPages host`/`bind` 610-629, and in _stream.mm `address()` 988-1000.

**Single form.** `static struct lease *lease_covering(const struct attach *, const void *, size_t extent)` and
`lease_uses_mark(...)` in nccl-mesh.c.

**Net lines.** nccl-mesh.c −15. Mostly one file; it is included because the brief lists it.

**Risk.** Low.

**Rubric.** C9.

---

### 13. Deadline helpers

**Instances.** nccl-mesh.c `deadline_left` 241-245 is the same as mesh-dataflow.c `left_of` 15-19, because
nccl-mesh.c registers `mesh_library_expired` as the expiry (724). mesh.h `mesh_deadline_left` 62-66 is the base of
both. mesh_rank.m 259, 285 and 852 have three polls on `ncclInProgress` with no deadline of their own; they are
bounded only by the library's init-phase deadline.

**Single form.** mesh.h's `mesh_deadline_left` consults the registered expiry, declared in mesh.h and defined in
mesh-dataflow.c, and both wrappers go. mesh_rank's three polls become one static helper in that file.

**Net lines.** nccl-mesh.c −5; mesh-dataflow.c −5; mesh.h +3. Total about −7 (mesh_rank.m −6, in one file).

**Risk.** Low.

**Rubric.** C6 (A4: every wait has a deadline).

---

## Below threshold or within one file (not ranked)

- **Binding-record word packing** (six u64 split into 12 u32 per channel, with null records): fused.py `_words` and
  `_encode_words` 41-42/89-96, and _stream.mm `compile_chain`'s `binding` and `input` lambdas 1431-1458. About −8.
- **Realization wrappers** (realize, enqueue and completion over a `next` cursor): _stream.mm `Collectives` 795-802,
  `ViewWrites` 1722-1730 and `Executions` 1645-1656, plus declarations.py `Execution.realize` 236-246. About −12,
  mostly within _stream.mm.
- **Builders of `MetalRecordedDispatch`** from bindings: metal-recording.m `MetalReplayCommands` 1181-1225,
  `MetalScheduleComputeNow` 1407-1433 and `recordDispatch`. All in one file.
- **The thread-width and grid computation** in nccl-mesh-metal.m `dispatch4` 237-239 and `dispatch` 268-270. One
  file.

## Ruled out

- **Dtype tables** in backend.py, mesh_mpi.py and mesh_rank.m: each is keyed by its own frontend's type vocabulary.
  The exception is torch's pair (C++ `kind` and Python `TYPES`/`typed`), which finding 3 covers.
- **Frontends issuing through NCCL calls**: this is the declared vocabulary (C1, eager-execution.md "Collectives are
  called inside a declaration"). Finding 3 merges only the field derivation.
- **`MetalTensorTraversal`**: collective_views and mesh_rank already share it.
