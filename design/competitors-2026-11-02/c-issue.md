<!-- design/page-maps-2026-11-02.md#the-runs-solve-each-choice-and-its-ground: a read-only inventory of a form the run's solve replaces, read by a subagent of the 2026-11-02 session at mesh-worktrees/joint 815b3af9 plus its uncommitted backend.py, collective_views.py and operand.py. It lists call sites to rewrite after the solve exists. It is a claim about the old code, not a requirement, and it decides nothing (design/rubric-actionable.md, procedure 5). -->

I've finished reading everything in scope. My findings follow.

# Inventory: allocators, schedulers and slot constructs in `rdma/nccl-mesh.c` 4000–7885, the write_binding block (2148–2269), and `rdma/mesh-write-compiler.h`

All lines are in `/Users/mdot/dox/mesh-worktrees/joint/rdma/nccl-mesh.c` unless another file is named.

**`mesh-write-compiler.h`.** There is no `mesh-write-compiler.c`. The header declares two things:
- `struct mesh_write_layout {tile_bytes, instances, instance_stride, invocations, invocation}` (line 10). This is the WRITE slot relation that `write_binding.layout` carries.
- `mesh_view_layout`, defined at nccl-mesh.c 7567–7605. It is pure geometry (affine rows or merged spans) and decides no capacity, reuse or order. It gets no entry below.

Some helpers that in-scope code depends on sit just before line 4000 and are cited where used: `group_at` 3983, `repeat_of` 3992, `write_order` 3813, `order_transfers` 3823.

---

## A. WRITE plan (write_binding)

**A1. `write_binding_prepare`, 7704–7750: allocator (relay lease) and slot relation**
- **Decides:** builds this endpoint's slot relation `local={.instances=slots.instances,.instance_stride=slots.stride,.invocations,.invocation}` (7717–7718) from the raw `d->bindings[endpoint]`. A relay rank with no binding that forwards gets its own storage, `mem_alloc(&lease,total,MESH_BACKING_PAGES)` (7733), with `total = last*instance_stride + frame_bytes`. The stride defaults to `frame_bytes` (7727), and `instances=0` (linear: one slot per invocation) applies when nothing is declared (7730).
- **Values:** relay landings, i.e. forwarded WRITE payload on relay ranks.
- **Safety:** linear storage means a distinct slot per invocation. Declared cyclic slots are reused by modulo and guarded downstream (B10, B12). The lease is held as `plan->payload`, kind 0 (7279–7280).
- **Lifetime:** allocated at `ncclMeshCommPlan` → landed by the bridge's DMA → read by the relay SEND → `write_binding_drop` frees it (2263) → given back by the retire ledger once its reads are done (`ncclMemFree` 1788, `leases_reap`).
- **Consumers:** `write_layout_prepare` → `write_prepare`, then `zoblimp` 6774, `place_data`, `write_prepare_spans`.
- **Note:** a linear relay makes `plan->occurrences = invocations` (7421), so `prepared` lays one issue per invocation (6802–6806).

**A2. `write_prepare`, 7292–7462: slot/modulo and capacity bound**
- **Decides:**
  - The slot extent must fit the table and lease (7342–7357): `write_slot_extent`, then `extent += slot*layout.instance_stride`.
  - `plan->occurrences` is the LCM over ranks of `layout.instances`; any linear rank makes it `invocations`; it is capped at invocations (7419–7425).
  - For each occurrence, each rank's phase is `(phase + InvocationIndex(occurrence,…)*instance_stride % MESH_DMA_EXTENT) % MESH_DMA_EXTENT` (7430–7431).
  - The packet bound is `max_packet ? max_packet : comm->page` (7299).
- **Sends to every rank:** `phase` taken from a region offset (7355), `origin` (7339), and storage runs as region offsets (7407–7411), via `prepare_exchange_locked` (7367, 7395, 7411).
- **Values:** the WRITE payload's DMA unit partition (`plan->packets[occurrence]`).
- **Safety:** none at run time; the cuts are immutable.
- **Lifetime:** preparation only; freed in `write_binding_drop`.
- **Consumers:** `write_prepare_spans`, plus `issue_lay`, `place_group` and `message_backing` through `write_packets` and `step_units`.

**A3. `write_view_packets` 7131–7181 and `write_storage_packets` 7187–7214: capacity bound (packetization)**
- **Decides:** cuts on the packet grid, restarted at each tile (7146–7148); a registration-extent lattice across repeats and participants (`mesh_span_residues` 7144, 7159); and per-occurrence physical-run breaks at `origin + InvocationIndex(occurrence,…)*instance_stride + repeat*frame_stride` (7197–7206).
- **Values:** DMA unit boundaries of WRITE sources, landings and relays.
- **Safety:** none. **Lifetime:** preparation only.
- **Consumers:** `write_prepare`, which stores the result in `plan->packets`.

**A4. `write_prepare_spans`, 7220–7290: allocator (descriptor table), slot formula, hold**
- **Decides:** one `mem_alloc(&plan->spans,…,MESH_BACKING_LINEAR)` (7251) holds every (step, occurrence, repeat, unit) descriptor.
  - Index: `span_first[write_span_index(plan,k,instance)] + repeat*units + unit` (7272).
  - Offset: from `origin + write_binding_slot(plan,instance)*instance_stride + repeat*stride + write_displacement(…)` (7266–7268).
- **Values:** DMA span tables for WRITE messages.
- **Safety:** an immutable table is reused by every invocation with the same `invocation % occurrences`. It is held by `plan->owned` (payload kind 0, spans `OWNED_TABLE`) through `owned_uses` and `owned_begin` (7280–7285). `present_group` overwrites `write_binding->owned.uses[2h+d].end = m->end.n` at every present (4760), so the last use recorded is the latest present's.
- **Lifetime:** written at plan time. Only the CPU reads it, at preparation: `place_group` 4324 and `message_backing` 6501 copy offsets out of it into the issue's own tables, which are what the bridge reads. It is released at `write_binding_drop` (2262).
- **Finding:** the table is a preparation-only intermediate, yet it is kept in region memory and held in the ledger as `OWNED_TABLE`.

**A5. `write_ready`, 7673–7702: slot/modulo readiness table and hold**
- **Decides:** `offsets[i*tiles+tile] = lease_page_offset(… offset + tile_offset(layout,tile,i))` for each slot `i < layout.instances` (7695–7696). The threshold `ready_value` is the same at every use.
- **Values:** source readiness (progress) words, producer → WRITE SEND.
- **Safety:** the same word and the same threshold are reused every `instances` invocations: entry `base = ready_value-1, step 0` (4795–4796). Hold: `owned_add(OWNED_TABLE|OWNED_SENDS)` and `owned_begin` (7693–7694).
- **Unsure:** nothing in scope resets the word between uses of a slot. Whatever makes a second use distinct is outside scope.
- **Lifetime:** written by the producer → read by the bridge's SEND gate and by `source_generation_effects` → held until `write_binding_drop`.
- **Consumers:** `present_group` 4793–4796, `source_generation_effects` 5639–5645, `source_group` 5862 (resident), `needs_publication` 3655.

**A6. `write_span_index` 2182–2184 and `write_packets` 2185–2187: slot/modulo**
- `step*p->occurrences + invocation%p->occurrences` and `p->packets + invocation%p->occurrences`.
- **Values:** which descriptor train and which packet plan an invocation uses.
- **Safety:** modulo index into immutable tables.
- **Consumers:** `issue_lay` 4087, `place_group` 4322, `message_backing` 6499, `write_prepare_spans` 7243/7244/7257/7259.

**A7. `write_binding_slot`, 2192–2194: slot/modulo**
- `ncclMeshInvocationIndex(invocation, layout.invocation, layout.instances)` gives the payload slot.
- **Values:** caller WRITE source and landing pages, at slot × `instance_stride`.
- **Safety:** none here. Landings into reused slots are guarded by B10–B12; sources are guarded by readiness words.
- **Consumers:** `place_data` 4200, `selected_effect` 5267, `write_source_copy` 5652, `ncclMeshCommSourceInputs` 6187, `write_prepare_spans` 7266.

**A8. `write_slot_extent`, 2195–2207: capacity bound**
- The last slot index touched over `[0, invocations)`: linear → `invocation+invocations-1`; cyclic → `instances-1` or the end slot.
- **Consumers:** `write_prepare` 7345, `write_binding_prepare` 7731 (relay size).

**A9. `tile_offset` 2214–2217 and `tile_extent` 2218–2238: slot/modulo and capacity bound**
- `(invocation % instances)*instance_stride + row/column strides`. `tile_extent` bounds the table.
- **Consumers:** `write_ready` 7678/7696, `source_generation_effects` 5641.

**A10. `write_binding_drop`, 2255–2269: reuse/retire rule**
- Under `attach_lock`, `owned_end(&p->owned, 1)` hands the recorded channel/walk uses to each lease's retire scope. Then it calls `ncclMemFree` on spans and relay, and frees the host arrays.
- **Consumers:** `calls_free` 5012 (from `compiled_drop` 3970 when a plan is replaced), `write_prepare` 7459, `write_binding_prepare` 7748.

---

## B. Definitions, period, issues, issue-owned pages

**B1. `ncclMeshCommPlan` 6879–6971, with `group_at` 3983–3989: order (occurrence → definition)**
- **Decides:** the runs `{first,count,repeat}` give `ends[k]`, `group[g].repeat += run.repeat` and `view.issues` (6914–6922). The plan starts at `view.first = comm->groups+1` (6961). `group_at` then computes `(holder-view.first) % view.issues`, then `% run.count`.
- **Values:** which definition (and so which issues and pages) each occurrence uses.
- **Safety:** modulo over the declared order. **Lifetime:** until the plan is replaced (`plan_free` 6968).
- **Consumers:** `definition_at` 4019, then `definition_of`, `ncclMeshCommLower` 5910, `occurrence_at` 5929; also `write_binding_prepare` (`x->repeat/frames`, 6941) and `prepared`.

**B2. `prepared`, 6791–6834: slot/modulo ("binding period") and allocation driver**
- **Decides:** `period` is the LCM of each call's slot count (`write_binding->occurrences` or `bindings[b].instances`; 0 means `invocations`), capped at invocations (6796–6808). It is joined across ranks by `prepare_exchange` (6812–6822) and refused above `UINT32_MAX` (6823).
- It then lays `period` issues, `zoblimp(s)` for s < period (6826); orders each with `issue_order` (6828), with `x->seconds` = the maximum (6829); runs `packets_prepare` (6831); and places each with `issue_place` (6832).
- **Values:** every first-period issue's own pages, and the caller-slot choice per instance.
- **Safety:** modulo. `x->laid[s]` for s < period becomes the template per instance.
- **Consumers:** `zorvek` 4982; `definition_of` 4029 and `ncclMeshCommLower` 5911 (period 0 means undeclared).

**B3. `zorvek`, 4980–5003: slot/modulo and reuse rule (issue selection)**
- **Decides:** `instance = invocation % x->period` (4982). It takes the first laid issue with that instance and `!used` (4984–4985). Otherwise it lays a new one with `zoblimp` and copies order and readers from the template `x->laid[instance]` (4987–4990).
- **`used` is never cleared.** It is only set at 5000 and read at 4985 and 5931. So past the first period, every lowering lays a fresh issue, and `x->laid` grows by one per lowering until the plan is dropped.
- **Values:** the occurrence's issue: zeeb, tables, progress words, input pages, input_requirements.
- **Safety:** library-owned pages are fresh per occurrence (renaming). Caller pages repeat per instance and are guarded by B12 and B13 (waits on `sent` of the latest scheduled issue of the same instance).
- **Lifetime:** from lowering to `compiled_drop`.
- **Consumers:** `lower_group` 5779, `occurrence_at` 5931, `landing_afters` 4685–4690, `arithmetic_targets` 5246–5251.
- **Unsure:** issues laid here never re-run `packets_prepare`. Their zeeb backing has to fit cuts that were unioned from the first period only, or `place_group` fails at 4363–4364.

**B4. `zoblimp` 6761–6787, with `issue_make` 5021–5039 and `issue_layout` 4878–4886: slot/modulo (caller slot)**
- **Decides:** `u->instance=instance` (6767) and `c->invocation=instance` (6772; zorvek writes the real invocation at 4996, after placement).
  - WRITE: `send_at`/`recv_at` = the plan's origin (6773–6774).
  - Collectives: `shifted(send_at, InvocationIndex(instance, bindings[0].invocation, bindings[0].instances)*bindings[0].stride)`, and the same for recv (6776–6777).
  - The host lists are sized `outs*repeats+1` (5033).
- **Values:** the caller's send/recv pages per instance.
- **Consumers:** `issue_layout` → `issue_lay`, `zappo`, `bind_calls`, `place_data`.

**B5. `slot_laid` and `zappo`, 4154–4187: allocator (bump within one lease)**
- **Decides:** size is the sum of:
  - `page_up` of each library-owned buffer, as chosen by `generation_buffer` 2619–2622 (scratch b≥1, a non-in-place result, all-to-all b≥2). Each is `elements*z` or `st->extent[b]` (4166).
  - `page_up(bytes)` per REDUCE landing (4169–4171).
- It allocates one `mem_alloc(&u->zeeb,…,MESH_BACKING_LINEAR)` per issue (4180), then bump-assigns `st->buffer[b]=base+used` (4165) and the landing area after them (4168).
- **Values:** scratch generations, relay generations, non-in-place results, reduce landing areas.
- **Safety:** distinct offsets within an issue and a fresh lease per issue. Held through `u->owned` when a message's data lies in it (4343).
- **Lifetime:** written by a landing's DMA, `operand_copies` 5748–5749 or a sum → read by SENDs, sums and `epilogue` → `issue_free` 5063.
- **Consumers:** `place_data`, `result_of` 4855, `operand_copies`, `zabbadoo_plan` 5136, `source_group` 5864.

**B6. `place_data`, 4190–4210: allocator (bump within the area) and placement**
- **Decides `x->data`:**
  - REDUCE landing: `shifted(area,used); used+=page_up(bytes)` (4206).
  - WRITE input-pages SEND: `frame*bytes + first*elt` (4201–4202).
  - WRITE: `frame*stride + write_displacement + write_binding_slot(write,invocation)*instance_stride` (4203).
  - Otherwise `x->offset` from the step's buffer or in-place read (4204–4205); all-to-all uses `view_at` (4208).
- **Values:** SEND sources, COPY destinations, reduce landing areas.
- **Safety:** slot formula. COPY landings into caller pages rely on B10.
- **Consumers:** `place_group`, `zinth` 4946, `zabbadoo_plan`, `zabbadoo`, `packets_prepare` 6689, `message_backing` 6509, `ncclMeshCommSourceInputs` 6195. It is re-run by `issue_order` 6617 and `issue_order_like` 6649.

**B7. `bind_calls`, `result_of` and `operand_at`, 4840–4873: placement decision**
- **Decides:** the result is the caller's `recv_at` when the call receives the whole operand. A reduce-scatter's result is the shifted recv. Otherwise it is `st->buffer[0]`, which lies in zeeb (4851–4856). `c->direct` comes from `plan_direct`.
- **Values:** results and partials.
- **Consumers:** `place_data`, `operand_copies`, `epilogue`, `zabbadoo_plan`.

**B8. `plan_direct` 5501–5571 and `source_generation` 5484–5500: allocator (source generation) and aliasing rule**
- **Decides:** physical overlap through `binding_relations` (5515–5521). It allocates `c->input_pages` (`ncclMemAlloc`, PAGES) when the call is premultiplied, or when the pages overlap but are not the same (5525–5527). A WRITE also gets `c->input_ready` (frames×tiles words, 5489–5494). It fills `st->read`/`st->fresh` (5534–5547) and the complement ranges (5551–5569).
- **Values:** input pages (source generations) and copied-readiness words.
- **Safety:** quote: "Actual prior DMA reads constrain their later COPY landing or arithmetic store through the prepared SEND-completion relation" (5548–5550). That relation is B10 and B13.
- **Lifetime:** written by `operand_copies` and `write_source_copy` (5649–5672, 5700–5707) → read by SENDs → `issue_free` 5054–5055.
- **Consumers:** `operand_copies`, `landing_progress` 4249/4279, `present_group` 4788–4792, `page_effects` 5385, `issue_order` 6610, `issue_order_like` 6632.

**B9. `place_group`, 4291–4411: allocator (span, bridge and after-list tables), capacity bound, hold**
- **Decides:** a two-pass bump layout in one `mem_alloc(&u->tables,…,LINEAR)` (4309):
  - per-message span tables, aligned to `MESH_RECORD_ALIGNMENT` (4358–4359);
  - per channel, a `mesh_zwoggle` and its entries (4371–4375);
  - per message, after-list room of `1+2*capacity` words. Capacity is `comm->count` for a landing and the prior-row count for a SEND (4381–4386).
- **Packet bound:** `write ? UINT64_MAX : comm->page` (4349). The registration extent is checked by `mesh_dma_span_valid` (4363).
- **Values:** DMA span tables, bridge tables, after-list records.
- **Safety:** re-placement first ends the previous holds (4300) and frees the old tables (4307), with reuse deferred by the ledger. It holds the payload leases (4343), the tables and the progress lease (`OWNED_TABLE` 4403–4404), then `owned_uses` and `owned_begin` (4406–4407).
- **Lifetime:** written by the CPU (spans; entries in `present_group`; afters in `ncclMeshCommSourceInputs`) → read by the bridge from `zinth`'s link until the walk leaves (`left=++ch->linked`, 4814–4815) → `issue_free`.
- **Consumers:** `present_group`, `zinth`, `source_group` 5838, `source_coverage` 6040, `landing_lists`, `landing_readers`, `arithmetic_readers`, `send_dependencies` (`unit_bytes`).

**B10. `landing_lists` with `overwrite_met`, 4519–4602: reuse/retire rule (wait on `sent`)**
- **Decides:** for each COPY landing, `own[i][q]` = (last k)+1 over this issue's SENDs on channel q whose physical spans meet the landing's spans and come earlier in (frame, call, consume order) (4532–4543, 4566–4579). Channels = `m->after | own` (4586–4587). Each channel gets an entry `{channels[q].view.sent - M, base}` (4595). The record names the issue's own SEND or NULL, in which case the base comes from `afters[q]` (4597).
- **Capacity:** `uint64_t` masks with `1ull<<q` imply at most 64 channels. **Unsure:** no check of this in scope.
- **Values:** COPY landings into caller pages (no wait for REDUCE scratch landings).
- **Safety:** a run-time wait in the bridge: the landing is held until `sent[q] ≥ base`.
- **Consumers:** `present_group` (bases 4746–4749), the bridge (entry `after`).

**B11. `landing_readers` and `landing_readers_of`, 4603–4677: reuse rule (who reads the landed pages)**
- **Decides:** for every laid issue of every group in the plan and each channel q, the last SEND k whose spans meet this issue's landing spans. It records `{x, v->instance, k, q}` (4655–4658) and sets `m->after |= 1<<q` (4611). Later issues copy this from the template (4667–4677).
- **Values:** the reader records for caller pages that landings overwrite.
- **Consumers:** `landing_afters`, `landing_lists`, and `arithmetic_readers` (called at 4663).

**B12. `landing_afters`, 4678–4692: turn-taking among laid issues**
- **Decides:** `afters[q]` = the maximum of `v->plans[q].out[k].end` over issues `v ≠ u` that are scheduled and have the same definition and instance.
- **Safety:** a wait on `sent` (through B10).
- **Consumers:** `present_group` 4745–4748.

**B13. `arithmetic_readers`, `arithmetic_readers_of` and `arithmetic_targets`, 5155–5260: reuse rule (store waits on `sent`)**
- **Decides:** for each REDUCE sum, the SENDs (earlier in this issue, or in any laid issue) whose source pages physically meet the sum's output pages. Each yields a `metal_wait` on `channels[q].sent` (5190–5198). The target is set at lowering from the latest scheduled same-instance issue: `last->end - pages + through` (5246–5253).
- **Values:** the outputs of the reduce store.
- **Safety:** an in-kernel wait on `sent`.
- **Consumers:** `zabbadoo` → `metal_zelloth` (`sum->wait`); `lower_group` 5796.

**B14. `landing_progress`, 4238–4289: allocator (progress and ready words)**
- **Decides:** one `mem_alloc(&u->progress,…,LINEAR)` (4262) holding:
  - per message, `spans_ready` (initialised to `pages.n`);
  - per channel that has SENDs, a `ready` word;
  - per non-WRITE input-pages original SEND, a `copied` word;
  - per non-WRITE REDUCE landing, a `summed` word (4267–4287).
- **Safety:** fresh per issue, monotone, held as `OWNED_TABLE` (4404). **Unsure:** the comment at 4240 ("Later uses of this prepared issue retain increasing global packet values") assumes issues are reused; B3 says each issue is used once.
- **Lifetime:** written by publishes, sums and copies, by the CPU stores at 6035/6199, and by mask commands → read by the bridge and by kernels → `issue_free`.
- **Consumers:** `present_group`, `send_dependencies` 4454, `required_word` 4470, `zabbadoo_plan`, `page_effects`/`source_publications`, `source_group` 5859.

**B15. `landing_runs`, 4212–4236: readiness map (host)**
- Breaks each non-WRITE landing's spans into element-width runs in `u->landings`.
- **Consumers:** `zabbadoo_plan` 5146, `zabbadoo` 5302–5308.

**B16. `metal_piece_hold` and `metal_piece_give`, 4133–4152: allocator (argument-record pieces)**
- Each piece is an `ncclMemAlloc` (PAGES) of `max(minimum, MESH_COMMAND_DATA_BYTES=65536)` bytes plus its buffer.
- **Values:** command argument/descriptor records. `nccl-mesh-metal.m` `argument_pages` 370–389 bump-allocates RECORD-aligned slices from the pieces.
- **Safety:** "never recycled while that function can execute" (nccl-mesh-metal.m 43–45). Pieces are given back on `MeshArgumentPages` dealloc (58). No ledger hold.

**B17. Host pools.** `kept()` 5066–5079 is a static `KEPT_SPANS` array used by `plan_direct` 5528. The thread-local `calls`/`call_capacity` (2375–2376) are reused at 5913–5918. `grow()` 4120–4131 doubles a list's room. All are host metadata.

**B18. `issue_free`, 5042–5065: reuse/retire rule**
- `owned_end(&u->owned,1)`, then `ncclMemFree` on tables, input_requirements, progress, zeeb, input_pages and input_ready. Leases come back through `leases_reap` once their reads are done.
- **Consumers:** `compiled_drop` 3968.

---

## C. Order, schedule and timing

**C1. `schedule_units`, 4055–4068: order (list ordinals)**
- Per-list cumulative page prefix: `end`; `listed=entry`, `laid=end`; `each[]` is the prefix after the first repeat.
- **Consumers:** `schedule_laid`, `send_dependencies`, `present_group`, the thresholds at 5388/5814 (`pages_to_word(laid)`), `ncclMeshCommSourceInputs` 6183, `source_coverage` 6051.

**C2. `issue_lay` 4070–4093 with `repeat_of` 3992–3999: order and modulo**
- `m[k%n]`, `frame=k/n`, `entry+=k/n*n`, `offset+=k/n*stride`. WRITE `pages = step_units[write_span_index(…, invocation)]` (4087). `consume[k] = k/p->ins*p->ins + p->consume[k%p->ins]` (4091).

**C3. `schedule_laid` 4095–4116, `commit_group` 4822–4829, `schedule` 4831–4838: order (global channel turn-taking)**
- `entry = listed + ch->sends|lands` and `end = ch->out|in + laid` (4107–4108). Writes `bound->channel/entry/consumption`. The commit advances `ch->sends/out/lands/in`, so positions follow the order in which occurrences are lowered across all definitions.
- **Consumers:** `present_group`, B12, B13, the `.target=x->end` thresholds (5146, 5258, 5318, 5367), `lower_group` 5788/5794, `ncclMeshCommRealized`.

**C4. `present_group`, 4741–4820: order, wait bases, lanes, retire marks**
- **SEND entry:**
  - default: the channel `ready` word, `base = laid - pages`, `step 1` (4778–4780);
  - relay: the `landed` word (or `dep_summed`) of `dep_h`, `base = channels[dep_h].in + dep_rel` (4782–4787);
  - copied words (4788–4792);
  - WRITE with `ready_value`: `ready_offsets[invocation % ready_layout.instances * ready_tiles + tile]` (4795).
- **Lanes:** `through = end - pages + unit + MESH_STREAM_LANES` (4777, 4804). This assumes the bridge's `p mod MESH_STREAM_LANES` lane assignment.
- **Requirement bases:** at 4746–4749.
- **Link chain:** `predecessor = (ch->link, ch->link_at)`, then `ch->link` = this table, and `left = ++ch->linked` (4812–4814).
- **Ledger marks:** `uses[2h+!send].end = m->end.n` (4759–4760) and walk uses `= ch->linked` (4815). Sets `scheduled=1`.

**C5. `zinth`, 4934–4956: order (link position)**
- Declares `metal_accesses` *Written* on every landing's data (4944–4948), then records `metal_open_gate`, which writes the tables' region offset into the predecessor's `next` (4951). The command index is `u->source_link[h]` (4949).
- **Effect:** the command solve places the link (when this group's receives are posted and its SENDs are released) after every earlier command touching the landing pages. Comment, 4936: "Link each prepared channel program after earlier readers of its destination pages."
- **Consumers:** `source_group` 5865, `ncclMeshCommSourceInputs` 6177–6198, `source_coverage` 6070–6071.

**C6. `send_dependencies`, 4413–4457: order (relay read-after-write)**
- `mesh_graph_forward` finds the forwarded landing. Its frame index is `s->entry[dep] - (repeats-1-repeat)*(ins/repeats)` (4444). `mesh_graph_cut_at` checks alignment; then `dep_h`, `dep_rel = (from->laid - from->pages) + units`, and `dep_summed` are set.
- **Consumer:** `present_group`.

**C7. `send_requirements`, 4459–4508: order (SEND after-lists)**
- `mesh_graph_frontier` over the prior receives gives words = `summed_word` or the channel's `landed`, at position `listed`. Bases come from `required_of(…)->end` at present time.

**C8. `zabbadoo_plan`, `zitem_order` and `reduction_fold`, 5081–5153: order**
- Sorted by (frame, call pointer, consume order) (5092–5100). All call pointers point into one issue's `u->calls`, so this is call-index order, not dependent on where memory was allocated; the same holds for `overwrite_met`, `write_order` and the pointer test in `arithmetic_readers`. Adjacent folds are merged. Inputs wait on `ch->landed ≥ x->end` (5145–5146).

**C9. Consumer and publication thresholds: order on words**
- `zabbadoo` 5283–5333: observe `landed ≥ end-pages+packet+1` (5318).
- `destination_effects` 5338–5376 and `page_effects` 5377–5396: publish on `ready` at `laid` (5388).
- `source_publications` 5806–5818.
- `source_generation_effects` 5626–5648: observe `source_ready` at the slot from `tile_offset(…, invocation)` (5641).
- `write_source_copy` 5649–5672: slot base at 5652; publishes the copied word with threshold 1 (5666–5669).
- **Consumers:** the `ncclMeshCommLower` effect callback and the source program.

**C10. `issue_order` 6593–6622, with `transfer_first` 6460, `issue_aliases` 6515–6581, `message_backing` 6493–6514 and `issue_order_finish` 6582–6592: order and allocation trigger**
- **Decides:** aliases are pairs (SEND read, landing write) in the same frame with physically overlapping page runs (6535–6546). They are exchanged with all ranks, sorted and deduplicated (6554–6577).
- `order_transfers` → `mesh_transfer_order` (3870) returns priorities, `consume_order`, `seconds` and `separate[]`. Each separated alias gets a `source_generation` (6605–6612) and is re-placed; the loop repeats until nothing changes. The finish step sets `entry=k`, `schedule_units` and `schedule_consumption`.
- **Values:** SEND/landing order, consumption order, source splits.

**C11. `issue_order_like`, 6623–6651: order copy**
- Copies `consume_order` and the per-step priority from the template, then `qsort(write_order)` by (frame, priority, round, call, offset). It splits a source wherever the template has one. No exchange with peers.

**C12. `packets_prepare`, 6653–6756: capacity bound (packetization)**
- Cuts at every first-period issue's physical backing (`MESH_DMA_EXTENT`), at coverage edges, and at `gcd(page, MESH_DMA_EXTENT)` for preserved WRITE sources (6696–6701). They are exchanged and unioned into the definition's packet cuts, which every steps copy shares (steps_copy 2501).
- This fixes `pages.n`, every threshold, and the ledger counts.

**C13. `ncclMeshCommSourceInputs`, 6084–6203: order, allocator, hold, initial ready**
- `source_input` coverage (6101–6111). `source_observations` keep the maximum threshold per word (6112–6130).
- `mem_alloc(&u->input_requirements,…,LINEAR)` (6133) holds the merged after-lists, which are patched into `entries[k].after` (6148–6149) after `present_group` has run. Hold: `OWNED_TABLE|OWNED_SENDS`, channel words excepted (6152–6173).
- The link command is declared as reading the SEND sources (6185–6197).
- The CPU stores a `ready` prefix equal to the leading input-only SENDs' `laid` (6199). Those SENDs are released by the link alone.

**C14. `source_group`, 5820–5880: binding phase and command order**
- `source_bindings = [u->instance, InvocationIndex(invocation, active.*), InvocationIndex(invocation, expected.*) …]` (5846–5856). It becomes `declaration.binding`, documented as the "authority for solved relation reuse" (nccl.h 1001–1006). Its reader is outside scope.
- Command order: link first, then per call: publications, copies, `zabbadoo`, complement, `epilogue` (5865–5875).

**C15. `source_coverage` 5974–6076, `coverage_slot` 5969–5973 and `coverage_declaration_order` 5945–5950: slot, order, hold**
- Slot: `offset + InvocationIndex(invocation, slots.invocation, slots.instances)*slots.stride`.
- The CPU stores `spans_ready = b->first` (6035). Mask commands publish `through` (6046–6049) and write into `m->spans` (6040).
- `metal_place_sources` moves the mask commands to `source_calls[before]` (6069) and relocates `source_link`.
- Holds the active/expected pages (6053–6064). Demand pointer: `values[invocation - demands.first]` (6012).

**C16. `lower_group` 5755–5802 and `ncclMeshCommLower` 5905–5926: occurrence cursor and poll budget (timing)**
- `polls = max(min(polls, NCCL_MESH_POLLS))` (5764–5768) is the in-kernel wait budget.
- `occurrence = comm->groups+1` (5774) and `comm->groups += repeats` (5799). `g->start`/`end` are channel cursors (5787–5795).

**C17. Timing and prediction**
- Production `{seconds, order}` records are copied per tile at declaration (6383–6397).
- `ncclMeshProductionTimes` (7006–7015) sets `seconds = total*(order+1)/extent`.
- These are consumed by `order_transfers` 3866 → priority → `write_order`.
- `x->seconds` (6828–6829) → `made->seconds[g]` (6953) → `view.seconds` → benchmarks.py 125 (reporting only).

**C18. `ncclMeshCommRealized`, 6270–6295: schedule check.** It gathers each cable's `sends`/`lands` from every rank and refuses if the peers disagree.

**C19. `declared()` slot normalisation, 6411–6443: slot bound**
- `if(!b->stride) b->instances=1` (6416), plus an overflow bound (6417); the same for coverage active/expected (6435–6436).
- **Consumers:** `zoblimp`, `prepared`, `coverage_slot`, `source_group`.

**C20. Preparation deadlines (timing, preparation only).** `comm->until = mesh_deadline(…, MESH_PREPARATION_NS)` at 6248, 6555, 6724 and 7366.

---

## Readers of slot-index formulas in scope

**`ncclMeshInvocationIndex`**
- 2193: WRITE payload slot.
- 2203–2204: first and last slot touched.
- 5854: coverage binding phase.
- 5971: coverage page offset (slot × stride).
- 6776–6777: collective send/recv offset (slot × stride).
- 7197: per-occurrence first physical byte.
- 7430: per-occurrence DMA-extent phase.

**`write_binding_slot`**
- 4200/4203: `place_data` displacement.
- 5267: page-effect base.
- 5652: input-copy source.
- 6187: link source-access base.
- 7266: span position.

**`% instances`**
- 2215: `tile_offset`.
- 4795: readiness word index.
- 7421–7424: LCM of instances (via gcd).

**`% occurrences`**
- 2183: `write_span_index`.
- 2186: `write_packets`.

**`% x->period`**
- 4982: `zorvek`.
- (`group_at` 3984/3988, just before scope: `% view.issues`, `% run.count`.)

**`instance_stride`**
- 2197, 2215, 2221–2222, 2236–2237: slot extent and readiness table.
- 4203, 5267, 5652, 6187, 7266: payload slot base.
- 7197, 7431: packet cuts.
- 7310: forced to 0 for an unbound or unstrided table.
- 7347–7348: extent check.
- 7718, 7727, 7731: relay slot layout and size.

**`stride * slot`**
- 5971, 6776–6777, 7347–7348, 7731 (`last*instance_stride`).

**`x->laid[instance]`**
- 4988, 4990: the template issue.
- Instance-matched scans of `x->laid`: 4655, 4687, 5248, 6680, 6827–6832, 6958.

**`u->instance` / `v->instance`**
- 4655, 4658, 4687, 4985, 5192, 5248, 5849; set at 6767.

**`source_bindings`**
- 5846–5856 (built); 5048 (freed).

**`c->invocation` as a slot input**
- 4087, 4322, 4795, 5641, 6012, 6499.

**Frame/repeat index arithmetic**
- 3993–3998, 4065, 4091, 4202–4203, 4444, 4465, 5355, 5659–5660, 7267.

---

## Shared state between entries

- **Channel cursors** (`ch->sends/lands/out/in/link/link_at/linked`): C3, C4, C6, C16, C18.
- **`x->laid` / `used` / `instance` / `scheduled`:** B2, B3, B4, B11, B12, B13, C12, C14, `occurrence_at`.
- **`u->owned` and `plan->owned` (retire ledger):** A1, A4, A5, A10, B9, B14, B18, C4, C13, C15.
- **The `u->tables` lease:** B9, B10, C4, C5, C7, C13, C15.
- **The `u->progress` lease:** B14, C4, C6, C7, C8, C9, C13, C15.
- **`u->zeeb` and the landing area:** B5, B6, B7, B18, C8.
- **write_binding `layout` / `occurrences` / `packets` / `spans`:** A1–A9, B2, B6, B9, C2, C9, C10.
- **`priority` / `consume_order` / `consume`:** C2, C8, C10, C11, B10, B13, C9.
- **`after` / `readers` / `afters` / `requirements`:** B10, B11, B12, C4, C7, C13.
- **Definition packet cuts (shared through `steps_copy`):** A2–A3, B9, C12, C15.