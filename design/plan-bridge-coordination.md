# D.5 and D.3.1: the implementation contract (draft)

The contract for building design/disjoint-pages-change-request.md, Appendix D, D.5 ("What the plan gives the bridge")
and D.3.1 ("Any number of programs at once"). A planning agent drafted it read-only on 2026-11-04 against mesh dc1555ff
and metal-microbench 63adb790; the D.3.2–D.3.6 unit (design/disjoint-pages-rulings-coordination.md) closed after it, at
mesh 1fd0cca0 and metal-microbench 3135184e, and the implementing coordinator reads the code from there. Claude reviewed
it. The implementing coordinator takes it as its contract once the operator has ruled on the points below and the
D.3.2–D.3.6 unit has closed. The change request holds wherever this file differs from it.

## Claude's review, and the rulings it asks for

**Withdrawn, 2026-11-04, after the operator's question ("wht is a shared carrier? why are you talking about waits?").**
§5's segment headers, sent on each of the bridge's lines ahead of their pages, and the trade-offs built on them (R-A,
R-B, and Claude's replacement for D7) are construction, not canon. The SCTP and QUIC citations don't support them: both
carry a stream's identity in the same packet as its data and reassemble from a receive buffer by copying, which is the
staging form data-model §3 rules out. §5 is to be rewritten from the canon and the failport's code before anything in
it is built. Everything outside §5 stands as reviewed.

Taken as written: the file split; the layouts; (group, epoch); the graph functions; attach taking nothing; each plan in
its claim; discovery through `plans`; the identity pages laid out with the region (D5); the GPU's placed pass and its
cost. D3 is sound: an announcement has to live in memory that reclamation can't reuse, as a hazard pointer does (M. M.
Michael, Hazard pointers: safe memory reclamation for lock-free objects, IEEE TPDS, 2004), so `prepared` stays as the
load's transient announcement.

The rulings, each with Claude's recommendation:
- **R-A, one form or two.** Recommended: D1, the one form. The sender writes a segment's header inline, ahead of its
  pages, on every carrier. The two-form alternative needs an agreed set of standing plans and an entry line withdrawn
  from data. D1 needs neither. It costs one 64-byte header per chain per round, which on e2b's path falls in the gap
  between rounds.
- **R-B, where a page on a shared carrier lands.** Recommended: (i), in its own page. Its cost is one receiver
  completion-to-post per event, and only where chains share a carrier. (ii) adds a pool, a free word per page and a
  dependent load on the GPU.
- **R-C, the wrap cut.** D.5 item 5 was wrong for chains of more than one event. It is corrected in the change request
  to the wrap cut of §4.
- **R-D, maps.** None is built until a program decides placement at run time; none does now. D.5 item 7 now says their
  tables go inside the window.
- **R-E, the start's port set.** Recommended: into the plan's state. The bridge parent writes one word once every port
  the plan needs holds it or has refused it, and `mesh_transfers_start` waits on that word alone. No port reaches
  library code.
- **R-F, grid.py.** Unchanged; it launches no rank.
- **D7, replaced by Claude.** Use SCTP's graceful shutdown (R. Stewart, ed., RFC 9260, §9.2) in place of taking a
  carrier out of service:
  - A sender finishes every segment it has written. Once its plan has ended, it sends the pages as they stand, then
    writes CLOSED.
  - The receiver lands those pages and moves no word for them, since its plan has ended too: no phase and no count.
  - Both ends' pages stay claimed until their `held` bits clear, which waits on CLOSED with nothing outstanding.
  - No carrier leaves service and no link ends because one plan has ended.

## 1. File split (headers first; one owner each)

| step | files | owner |
|---|---|---|
| 1 | mesh rdma/mesh.h, mesh-dataflow.h, mesh-call.h, mesh-collective.h, mesh-graph.h, mesh-metal.h; metal-microbench metal_recording.h | H (headers) |
| 2 | mesh-graph.c | G |
| 3 | mesh-collective.c | C |
| 3 | mesh-dataflow.c, mesh-call.c, mesh-map.h (layout), rdma/mesh.py | L (library) |
| 4 | mesh-records.c, mesh-link.c, mesh-stat.c, mesh-heartbeat.c (its holders read alone) | B1 (bridge parent) |
| 4 | mesh-flow.c, mesh-bridge.h | B2 (data path; after the rulings) |
| 4 | mesh-metal.m | M |
| 5 | metal_recording.m | R |
| 6 | tools/e2b_replay.m | E |
| 6 | tools/mesh/run_pair.py | P |
| — | reviews: mesh side; GPU and program side; the four rules and D.5 | 3 reviewers |

Unchanged:
- the lifecycle files mesh-children.c, mesh-child.c and mesh-lanes.c (B2 reads the port's identity page from
  `mesh_bridge_pages(M)->link[p]`, so CHILD_MAKE's `.page=MESH_ABSENT` stays);
- mesh-ports.c, mesh-events.c, MESH_CHILD_VERSION, and the wire messages (struct message);
- tools/prepare_e2b.py;
- tools/mesh/grid.py.

## 2. Shared names, exact layouts

**mesh.h, MESH_VERSION 110.** Its note: "110: the plan in its claim (D.3.1, D.5); hdr's one-plan fields vacant; `plans`
appended; struct mesh_hop 64 bytes; struct mesh_state 192; struct mesh_buffer `start`; struct mesh_place."

`struct hdr`. No field moves.
- Vacant, never read or written: `client` (160) → `vacant15`, `hop_off` (224) → `vacant16`, `hop_capacity` (232) →
  `vacant17`, `hops` (248) → `vacant18`, `plan` (256) → `vacant19`, `client_start` (304) → `vacant20`, `state` (400) →
  `vacant21`.
- Appended: `_Atomic uint64_t plans` at 408, so `sizeof(struct hdr)==416`.
  - It is a count that only rises. Every publication of a plan and every close of a claim adds 1 (release), then wakes
    it (`os_sync_wake_by_address_all`, SHARED).
  - It has many writers, as a count does, and is never reset.
- `serial` (184) stays. It mints client identities and claim generations (`atomic_fetch_add(&m->serial,1)+1`, masked to
  31 bits, as now).

`struct mesh_buffer` (32 bytes): `owner` 0, `closed` 8, `pages` 12, `constant` 16, and `uint64_t start` at 24 (the old
padding).

**A claim's head row** is the claim's first row; no value's rows include it.
- Its `struct mesh_buffer`: `owner` is the client word (stored last, release; 0 once released); `pages` is the claim's
  rows; `start` is `mesh_started(getpid())`.
- Its `struct mesh_page_entry` is unchanged: the window run, the local run, and `device` holding the generation.
- Its `struct mesh_publication`: `state` is the region offset of the plan's state section, stored with release once the
  plan stands (§6), 0 before that, never cleared. Every other field is 0.

A plan **stands** where `owner!=0 && closed==0 && publication.state!=0`.

`struct mesh_state` (192 bytes, `_Alignas(64)`):
- 0–63 as now: `copies, receives, ranges, releases` (u32); `phases, counts, supplied, status` (u64); `executions`
  (_Atomic u64) at 48.
- `places` (u64) at 56, in place of `padding`: the region offset of the placement tables.
- `group` 64 and `epoch` 72: the plan's name.
- `serial` 80: the map serial it was made on.
- `hops` 88: the region offset of the entries.
- `entries` 96: u × n.
- `_Atomic uint64_t held` 104: a bit per port, "this port holds the plan" (§6).
- `_Atomic int32_t refused[MESH_PORTS]` 112: port p's refusal code (0 none); its one writer is port p.
- `uint64_t padding[6]` 144.

The section's layout: the struct; phases, counts, supplied and status (the words mesh_plan starts once, D3); the
placement tables; the entries. Each is 64-aligned.

`struct mesh_place { uint32_t page,bytes; }` (8 bytes).
- `page` is an arena page (a multiple of `block`, inside the window).
- `bytes` is that exchange's bytes, from 1 to `block*pgsz`: `min(block*pgsz, range bytes − k*block*pgsz)` for page k.
- Each routed value at this node (sent, received or relayed) has u × P entries, copy-major: copy c, page k at `table +
  (c*P+k)*8`.
- Static placement is written once by mesh_plan and never again (D.5 item 7).

`struct mesh_hop` (64 bytes):
```c
struct mesh_hop { uint32_t node,direction,chain,place; uint64_t word,count,entry;
                  uint32_t field,pages,priority,event; uint64_t padding; };
/* node 0, direction 4, chain 8, place 12, word 16, count 24, entry 32, field 40, pages 44,
   priority 48, event 52, padding 56; sizeof 64 */
```
- `entry`: the region offset of its copy's `struct mesh_place`. It replaces `page`.
- `chain`: its chain among (`node`, `direction`)'s at this node, numbered alike at both ends (§4).
- `event`: the index in its chain of the release event behind it.
- `place`: its place in its chain within one round (events in chain order, an event's pages in page order).
- `priority`: its event's level (§4).
- `word`, `count`, `field` and `pages` keep their 109 meanings.

**Publications.** A routed range's first row gets `targets[1]` = {word: the region offset of its copy-0 table row,
count: P, stride: P*8}, at a source, a destination and a relay alike. `targets[0]` stays the source's supplied word.

**`struct mesh_port_info.prepared`** stays, narrowed to the generation of the claim this port is loading now (0 none).
It is set and cleared within one load (§6, D3).

**mesh-dataflow.h.**
- `struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; int fd; uint32_t padding; uint64_t epoch,state; }`,
  offsets 0/8/16/24/28/32/40, size 48.
  - `epoch`: this context's count of its mesh_plan calls, raised at each call's entry.
  - `state`: the region offset of this context's standing plan's state section (0 none).
- rdma/mesh.py `Context` gains `('padding',U),('epoch',Q),('state',Q)`.

**mesh-collective.h.**
- `int mesh_plan(struct mesh_ctx *, uint64_t group, const struct mesh_declared *, uint32_t count, struct mesh_value *,
  uint32_t values, const struct mesh_dependence *, uint32_t dependences);`
- `group` 0 is EINVAL. EEXIST where a standing plan at this node already has (group, epoch) (D6). ENOSPC for the page
  lists' room goes.

**mesh-metal.h.**
- `struct mesh_metal_transport { void *state,*window; struct mesh_state *words; uint64_t offset,bytes; }` (40 bytes).
  `window` is one MTLBuffer over arena page 0 to `wire_pages`, made with no copy.
- `struct mesh_metal_input { uint64_t phases,group,count,places; uint32_t pages,copies; }` (40 bytes). `places` is the
  byte offset of copy 0's table row in the state buffer.
- `struct mesh_metal_publication { uint64_t supplied,places; }` (16 bytes).
- Every function takes the plan from `context->state`, never from hdr.

**metal_recording.h.**
- `struct MetalRemote { uint64_t phases, group, count, places; uint32_t pages, copies; }`, asserted equal to
  mesh_metal_input.
- `- (void)window:(id)capture buffer:(id<MTLBuffer>)window pageBytes:(NSUInteger)bytes;`
- `- (void)published:(id)capture operand:(id<MTLBuffer>)operand supplied:(NSUInteger)supplied places:(NSUInteger)places
  pages:(NSUInteger)pages copies:(NSUInteger)copies;`
- `place:operand:in:offsets:copies:ahead:` raises on an operand that `received:` or `published:` names.

## 3. The plan's name

- A plan is (group, epoch). `group` is the launcher's 64-bit value, the same at every rank (NCCL's `ncclUniqueId`;
  PMIx's namespace).
- `epoch` is `context->epoch` after the call's raise, the same at every rank because mesh_plan is collective.
- Bridges pair one node's plan with the adjacent node's plan of equal (group, epoch).
- A client holds one standing plan: as now, step 8 closes its earlier claims.

## 4. Graph algorithms (mesh-graph.{h,c}; C9), called by mesh_plan

**Events.** Per adjacent node b and direction at this node, an event has two template vertices: in (what precedes it)
and out (what follows it).
- A source's send of range i: in = out = P(i), the range's supplied word.
- A forwarded page k of range i at node y (a relay, or a destination that sends it on), released by its phase: in =
  W(i,y), out = L(i,y).
- The receive direction's events are b's send events toward this node. Both ends compute them from the one template.
- e precedes f where a lag-0 walk of one or more edges joins out(e) to in(f). This is transitive, since in ⇝ out for
  every event. A relay's pages of one range are mutually unordered, as the template says.

**Functions.** Each cites its source in its comment.
```c
/* targets reached from each source by >= 1 edge of the rows (lag-0 rows: plan_acyclic's): bit t of
   before[s*words+t/64]; one pass in reverse topological order over bitsets.  0; EDEADLK; ENOMEM */
int mesh_graph_reach_sets(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *,size_t),
    const void *context,uint32_t sources,const uint32_t *source,uint32_t targets,const uint32_t *target,
    uint64_t *before);
/* Hopcroft-Karp (SIAM J. Comput. 1973): a maximum matching of left [0,left) to right [0,right),
   left l's edges adjacent[first[l]..first[l+1]); mate[l] its right or UINT32_MAX, taken[r] its left;
   its size, UINT32_MAX on ENOMEM.  Phases of BFS layers then vertex-disjoint DFS augmentations */
uint32_t mesh_graph_matching(uint32_t left,uint32_t right,const size_t *first,const uint32_t *adjacent,
    uint32_t *mate,uint32_t *taken);
/* Dilworth by Fulkerson (Proc. AMS 1956): elements [0,n) with element i before j where bit j of
   before[i*words+j/64]; the bipartite graph i_left -> j_right for each such pair, matched
   (mesh_graph_matching); chains = n - matching, each chain followed from its unmatched-on-the-right
   element by mate[].  chain[i], place[i]; *chains.  Chains numbered by their first element's index,
   so equal inputs number alike.  0; ENOMEM */
int mesh_graph_chains(uint32_t n,const uint64_t *before,uint32_t *chain,uint32_t *place,uint32_t *chains);
/* each vertex's level: 0 at a vertex with no edge out, else 1 + the greatest level of its heads
   (Adam, Chandy and Dickson, CACM 1974; Kwok and Ahmad, ACM Comput. Surv. 1999: the static b-level,
   unit weights).  0; EDEADLK; ENOMEM */
int mesh_graph_levels(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *,size_t),
    const void *context,uint32_t *level);
```

**mesh_plan's use** (plan_walk's new pass, step 1, before anything is written):
1. The lag-0 rows, the levels, and per (b, direction) the events in walk order (demand order, then page).
2. `before` from mesh_graph_reach_sets, then mesh_graph_chains.
3. The wrap.
   - Round t+1 of a chain may follow its round t only where D(out(last) → in(first)) ≤ 1, by mesh_graph_lags.
   - D(e → e) = 1 by the coordination's D5, so a chain of one event always passes.
   - Where a chain fails, it is cut greedily from its first element: extend while D(out(e_j) → in(e_s)) ≤ 1. This is
     monotone in j, since e_j ⇝ e_{j+1} at lag 0, so greedy makes the fewest cuts (R-C; D.5 item 5 as corrected).
4. Each hop's `chain`, `event` and `place`, and its `priority`: the level of the forward vertex it releases, F(i, this
   node, b).

**Costs**, at compile time, once a plan, never in the run's length. V and E are the template's; m is one
pair-direction's events.
- reach sets: O((V+E)·m/64);
- matching: O(E_c·√m), with E_c ≤ m² comparable pairs;
- levels: O(V+E);
- wraps: O(chains·(E+V) log V).

## 5. The bridge (its own files only: mesh-flow.c, mesh-bridge.h; mesh-records.c, mesh-link.c)

"Carrier" here is the bridge's ordered line, its private object.
- A port's cable has the carriers its pairing made: `link_plan`, both ends' smaller reported budget (the failport's
  rule).
- The SEND posted k-th on a carrier fills the receive posted k-th on it (data-model §3).
- The design takes any count of carriers ≥ 1 and branches on no OS. Evidence that it must: tag pre-return-20261103
  mesh-verbs.h:27–29 records macOS 26 devices at max_qp 11 and M5 Ultra macOS 27 devices at 3, less the provider's one.

**State per adjacent node and direction, in the port's child:**
1. **Chains.** Per loaded plan and chain: its records in `place` order, restricted to this cable's share, with one
   cursor (round, place).
   - Each event's pages are dealt over the pair's cables in key order by `mesh_graph_list_schedule`, with cables as
     machines at their bandwidth. Both ends compute the deal at load from the map at the plan's `serial`.
   - Records are one per hop, built at load. Each span is `wire_span(page*pgsz, bytes)` from the hop's placement entry,
     read once at load; the request is never changed afterwards.
2. **Ready set** over a carrier's chains, keyed by the head event's `priority`, then the chain.
   - It is a bit-tree over levels, with one bitmask of chains per level (van Emde Boas, FOCS 1975); up to 64, one word
     and `ctz`.
   - A chain enters when one acquire load of its head's release word passes. The pick is the highest set level, then its
     lowest chain: O(1).
3. **Free landings.** Per received page and copy, a bit set at its landing and cleared at its posting (R1, as now).
4. **Identity.** Per carrier, a FIFO of segments `{plan, chain, round, first, count}`. The carrier's k-th position,
   counted from a segment's start, is page `first + (k − start)` of that chain-round's share on this cable.
5. **Slots.** Per carrier, one in-slot and one out-slot of 64 bytes in the port's identity page.
   - Port p's identity page is arena page `p*block`, in the window's first `MESH_PORTS` blocks.
   - mesh-link.c sets those blocks' arena bits and records them in `mesh_bridge_pages(M)->link[p]` when it lays out or
     adopts the region.
   - Carrier q's in-slot is at byte `128q`, its out-slot at `128q+64`. `MESH_LINK_LANES*128 ≤ block*pgsz` is asserted.

**Entries.** 64 bytes, bridge-private: `{u32 kind, chain; u64 group, epoch, round; u32 first, count; pad}`. They are
SCTP's INIT, DATA chunk (stream identifier and sequence number; RFC 9260 §3.3.1) and SHUTDOWN, and QUIC's STREAM frame
and FIN (RFC 9000 §19.8). Rule 1 keeps them out of the data page, so they travel just ahead of it on the same carrier
instead of in its header.
- LOADED: this end holds (group, epoch).
- SEGMENT: a segment of (chain, round, first, count).
- CLOSED: the sender writes no more segments of the plan on this carrier.

**Where identity is implied, and where it travels ahead.**
- Implied: a page's cable (the cable share both ends compute); its landing page (the receiver's own placement entry for
  (range, page, copy t mod u)); its order within a segment (position).
- Travels ahead: only each segment's header, naming (plan, chain, round, first, count).
  - It is one 64-byte SEND into the carrier's in-slot, immediately ahead of the segment's first page.
  - The hardware holds that page until its receive is posted (data-model §3), so the header adds no wire trip. The MPI
    rendezvous's RTS/CTS round trip (Liu, Wu and Panda, ICS 2003) collapses to the receiver's post: the credit hold is
    the CTS.

**Sender, one rule per carrier.**
- A plan's segments go on a cable only after the peer's LOADED for it has landed on that cable.
- At each change of the cable's loaded plans, chains are dealt to carriers by Graham's LPT (1969) on bytes per round, in
  (group, epoch, chain) order. Chains are independent jobs, LPT's proper domain. A chain dealt several carriers splits
  its round's share into contiguous segments over them.
- Each carrier's next segment is written into its out-slot and sent as soon as it is determined and every earlier SEND
  on that carrier has been posted. Its pages follow as each release passes and the gate holds.
  - One chain dealt here: its next round's share, determined in advance. Its header is sent once round t's last page is
    posted.
  - Several chains: the ready set's pick, one event at a time, determined at its release (priority among ready heads,
    D.5).
- An out-slot is rewritten only once its last SEND has completed.
- Shutdown follows Claude's ruling on D7 (above): every written segment is finished, then CLOSED is written.

**Receiver, one rule per carrier.**
- It posts the in-slot. On its landing it reads the entry.
- For a SEGMENT, it posts the segment's receives in order, each once its page is free (R1), then posts the in-slot again
  (named once per landing).
- A landing:
  - `byte_len` must equal the entry's `bytes`, else EMSGSIZE fails the link;
  - the phase's XOR, the count's add and the wake are unchanged;
  - a completion's `wr_id` tells a slot from a record (records are 128-aligned; a slot's low bit is set).
- A plan closed here is let go on this cable once CLOSED has landed on every carrier and nothing of it is outstanding
  (§6 `held`).

**Obligations met.**
- Each chain progresses alone. A page waits only on its release and on its own chain's earlier pages; on a shared
  carrier it waits behind ready heads alone.
- By the allocator's order, a page's release at the sender follows the receiver's reads of its last use, so a segment
  written at release finds its pages free.
- Every landing is set first, in its named page, with no staging, and only words come back.
- Any number of plans and chains run on any count of carriers ≥ 1.

**Latency.**
- A carrier with one chain (e2b, whose chains per direction are expected to fit the carriers; checked at load, never
  assumed): round t+1's header lands just after round t's last page. The wrap puts round t's last release before round
  t+1's first. The added latency is max(0, the receiver's completion-to-post − the sender's gap between those releases),
  which is 0 for e2b, whose gap holds the tail of decode step t and the head of step t+1.
- A shared carrier: one receiver completion-to-post on each event's critical path, with no extra wire trip.
- With D.5's per-page `bytes`, a small partial moves `frames_of(bytes)`, not a whole page.

**Costs.**
- Wire: 64 bytes and one frame per segment.
- Memory: 128 bytes per carrier, plus `MESH_PORTS` blocks once.
- Per segment: one completion and one post. Per pick: O(1).

**What goes, in the bridge:**
- `struct flow flows[MESH_PORTS]`, which becomes per (port, loaded plan);
- link_prepare's units over every carrier of every cable;
- `struct cursor`'s lane-order walk;
- link_run's reads of hdr `hops`, `plan` and `client`, and port_progress's wait on hdr `plan` (it waits on `plans`);
- mesh-map.h `struct mesh_cable` `qps`, since the child uses its own link's plan.

## 6. D.3.1

**Attach takes nothing.** `mesh_attach`:
- maps the region and checks its magic, version and length;
- mints `client = (atomic_fetch_add(&serial,1)+1 & 0x7fffffff)<<32 | pid`, bit 63 staying `device_client`'s;
- sets `epoch=0` and `state=0`;
- reads the version again.

There is no CAS and no EADDRINUSE, and it writes nothing in hdr but `serial`.

**Each plan in its claim.** mesh_plan's state item grows by the tables and the entries. The writing order:
1. Compute everything, including §4.
2. Take the claim: the generation from `serial`, the head's `start` set.
3. Start the section's words once (D3), and write the tables, entries, `group`, `epoch`, `serial`, `hops`, `entries` and
   `places`. Bind the rows and write the publications, `targets[1]` included.
4. Read the map again. Where it moved: close this claim and every earlier one, set `context->state=0`, and return
   ESTALE.
5. Store the head's publication `state` (release), set `context->state`, then raise and wake `plans`.
6. Close every earlier claim of the client, then raise and wake `plans`.

A plan's pages are never rewritten while it stands, so the old re-checks for torn reads go.

**Discovery.**
- Each port child's progress thread reads `plans` once a pass, and when idle waits on it (SHARED, bounded by
  MESH_PAUSE_NS); `guard_wake` wakes it.
- Where `plans` moved since its last walk, it walks the heads (MESH_ROW_OWN, `mesh_head_after`). It loads each standing
  plan it has neither loaded nor refused, and ends each loaded plan whose claim is closed or reused.
- **A load** (D10 kept; D3):
  1. store `prepared` = the generation (release), then a seq_cst fence;
  2. read the head again (owner, generation, `closed`, publication `state`); where it changed, store `prepared=0` and
     skip;
  3. compute: ESTALE where `serial != map_serial`, EALREADY as now;
  4. on success `atomic_fetch_or(held, 1<<p)` (release); on refusal `refused[p]=code`;
  5. store `prepared=0`.
- A port holds every standing plan: it sets its `held` bit even where its cable carries nothing of the plan.
- The parent walks the heads on its event-loop thread, for plan_stale, the cancellations, `clients_turn` and the
  release.

**Retirement and the deferred release.**
- `mesh_retire(m, client)` closes every claim whose owner is `client` (release), then raises and wakes `plans`.
- A child ending a plan raises its not-landed counts (link_close's rule, per plan). Once nothing of the plan is
  outstanding and CLOSED has been heard on every carrier, it clears its bit (`fetch_and`, release). That is its last
  access to the plan's pages.
- A child's death: `streams_cancel(port)` with `!port->child` clears bit p in every plan not yet released, on the
  parent's thread, the releaser's.
- `mesh_retired_release`, for each closed claim: acquire `closed`, a seq_cst fence, then read every port's `prepared`
  and the plan's `held`. It keeps the claim where any `prepared` equals the generation or `held != 0`, and otherwise
  releases it as now.

**The start** (`mesh_transfers_start(ctx, executions)`, by `ctx->state`): its port set follows the ruling on R-E. Then
come the raise, the fence, and the read of `map_serial` against the plan's `serial` (R3).

**Cancellations, over every standing plan, on the parent's thread.**
- `plan_cancel(state, from)`: one plan's receive entries from map index `from` (UINT32_MAX for every one), with
  `executions` read once.
- `streams_cancel(port)`: every standing plan, from the lost peer.
- `plan_stale` (R2): a fence, a walk, then every plan whose `serial != map_serial` is cancelled.
- `streams_let_go` goes: a gone client's plans are closed by retirement, and their ports end them.

**Holders and clients.**
- mesh-heartbeat.c `replace_turn`: its holders are the bridge, its children, and each open claim's (owner pid, head
  `start`) by `mesh_holds`. The holders array is sized by count (malloc).
- `clients_turn` walks the heads. It adds an EVFILT_PROC watch for each open claim's pid, idempotent per pid, and
  retires at once a pid already gone. The event loop's EVFILT_PROC retires every owner of the event's pid.
- `busy_tell`: busy = 1 where any open claim names a live client other than the bridge.
- `region_open` retires each gone owner.
- mesh-stat prints `"clients":[pids]` in place of `client`.

## 7. The GPU side (mesh-metal.m, metal_recording.m)

- `mesh_metal_transport_create` wraps `ctx->state`'s section as now, plus `window`.
- `receive_prepare` and `publication_prepare` read `targets[1]` for `places` (made relative to the state buffer and
  checked inside it) and test `head->state==ctx->state`.

**What the recorder binds**, for every recorded dispatch that binds a routed operand (received or published, in any
capture):
- the operand's slot: `window`'s gpuAddress, plus the binding's own offset within the operand (a constant of the
  command);
- one free slot per routed operand for its table row: the state buffer at `places + (t mod u)*P*8`, a MetalStateWord as
  the phase group is (D4: nothing of the plan is a literal);
- one free slot per command for the capture's invariant row {pageBytes, 0, 0, 0}.

A command that can't be spliced (no recorded source, or a blit) and binds a routed operand raises at replay. Slots are
counted with the masked pass's 2d+3.

**What the spliced kernels compute.** `MetalMaskedPass` becomes `MetalPlacedPass`. Its first and later forms, and a
placed form for a dispatch with no received operand (a producer, or a source's own read), begin:
```metal
uint2 mesh_place_e_d = mesh_place_row_d[0];      // per routed operand d, before the phase loop
device T *name = (device T *)((device char *)mesh_place_base_d + (ulong)mesh_place_e_d.x * mesh_place_bytes);
```
- The operand's parameter is renamed `mesh_place_base_d`; `name` is its original name.
- The phase loop, the vote, the fence and the body follow unchanged.
- The entry loads issue before the phase loop, so the phase loads issue while they are outstanding.
- At replay the recorder reads each routed operand's table once. It raises where a copy's entries are not `page_0 +
  k*block`, since a delegate kernel reads a range as one run.

**Cost** (D.5's): per routed operand per threadgroup, one 8-byte load from an address fixed by its binding, the same for
every lane, beside the phase load. There's no dependent round trip, because the data loads already wait on the phase.
Then one multiply-add, held in a register. Nothing goes on the stack or the heap, and nothing is allocated at run time.

## 8. The program and the launchers

- **tools/e2b_replay.m:**
  - reads `MESH_GROUP` (`strtoull(s,&end,0)`, whole and nonzero; missing or invalid: exit 2, naming it) and passes it to
    `mesh_plan`;
  - calls `[recorder window:capture buffer:(id)transport.window pageBytes:block*pgsz]` for each capture;
  - names routed operands by `received:` (MetalRemote with `places`) and `published:...places:`, never by
    `place:operand:`.
- **tools/mesh/run_pair.py:**
  - `run()` draws `group = 1 + secrets.randbelow(2**64 - 1)` once per call;
  - `launch()` adds `MESH_GROUP={group}` for every rank of that run, and a solo run gets its own group;
  - the run's record carries `group`.

## 9. What goes, by file

- **mesh.h:**
  - hdr `client`, `hop_off`, `hop_capacity`, `hops`, `plan`, `client_start` and `state` (made vacant);
  - `mesh_hops()` and `mesh_plan_state()`;
  - struct mesh_hop's `page` and `serial`, and its 48-byte form;
  - the "Both ends" text in queue pairs, rewritten in chains, places and rounds;
  - "Writing a plan" steps 4–7 on hdr.
- **mesh-map.h:** `mesh_geometry.hops`; the page-list area in `mesh_layout`; mesh_cable `qps`.
- **mesh-link.c:**
  - `configuration_read`'s `hops` key (a stated `hops` is ignored; bin/mesh-bridge.sh still writes it, which is
    reported, not changed);
  - region_open's and the event loop's hdr `client` reads;
  - `link[p]=MESH_ABSENT`, since these are now the identity pages.
- **mesh-dataflow.{h,c}:**
  - `mesh_attach`'s CAS loop and EADDRINUSE;
  - `mesh_retire`'s hdr `client` CAS and its `hops`/`state` stores;
  - `mesh_ports_loaded`'s `prepared`, `phase` and `code` reads.
- **mesh-collective.{h,c}:** ENOSPC for `hop_capacity`; the generation from hdr `plan`; the writes to hdr `hops`,
  `state` and `plan` and to the hop area; plan_walk's constant page in entries.
- **mesh-flow.c:** one flow per port; link_prepare's spreading of units and its lane cursors; plan_cancel's hdr walk;
  `streams_let_go`; the waits and wakes on hdr `plan`.
- **mesh-records.c:** `busy_tell`'s client word; `clients_turn`'s one owner and its `streams_let_go` call.
- **mesh-heartbeat.c:** the hdr `client`/`client_start` holder read.
- **mesh-stat.c:** hdr `client`.
- **mesh-metal.m:** every read of hdr `state`.
- **mesh.py:** Context's four-field form.
- **metal_recording.{h,m}:** routed operands bound by copy offsets; MetalRemote's 32-byte form; `MetalMaskedPass`
  (renamed and generalized).
- **e2b_replay.m:** mesh_plan without a group; `placed_buffer`/`place:operand:` for routed operands.
- **D.4's entry on the ports' `prepared`:** narrowed, not removed (D3).

## 10. The planner's decisions

- **D1.** The receiver never computes a deal. The sender owns each carrier's order and writes segment headers inline.
  Sources: the stream identifiers of SCTP and QUIC, moved out of the page by rule 1; data-model §3's credit hold. It
  works at any count ≥ 1 and needs no separate line.
- **D2.** Chains come from the lag-0 order plus the wrap cut. Priority is the unit-weight b-level, since the template
  carries no times.
- **D3.** `prepared` is kept as the load's transient announcement. A bit ORed into the plan's state, after the port
  reads the head as open, could land in pages released and reused in between; D10's pairing closes that window only for
  a word in the bridge's own memory. This departs from D.4's entry.
- **D4.** Static tables are read once at load (records unchanged once written) and once at replay. Maps are not built
  (R-D).
- **D5.** The identity pages are `MESH_PORTS` window blocks taken at layout (A3: "the bridge's own pages are laid out
  once with the region"). mesh-bridge.h's 2026-11-03 note ("no bridge state takes arena pages") is reversed with a dated
  note.
- **D6.** EEXIST for a duplicate (group, epoch) at one node. `group` 0 is EINVAL.
- **D7.** Replaced by Claude's ruling (top of this file).

**Risks.**
- **Wide relayed ranges.** A relay's pages of one range are an antichain, so a wide relayed range gives many chains, and
  one header per page on shared carriers.
- **Striping.** A chain's share goes only on the carriers it is dealt, where before it was spread over every carrier.
  Per-carrier throughput against the cable's is unmeasured.
- **Running.** Nothing here runs until the operator authorizes it.
