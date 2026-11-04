# Disjoint-pages revision (C3): the coordinator's contract and its routing record

The coordinating agent's notes for carrying out design/disjoint-pages-change-request.md, Appendix C, C3 ("Claude's
proposal: the consolidated revision") and C4 ("What goes"), from mesh 360a6ccc and metal-microbench 472390a8. The
requirements are the change request's, not this file's: its operator's turns (C1) are the rulings, and C3 holds wherever
it differs from the rest of that document or from design/disjoint-pages-coordination.md (the first implementation's
contract, which this one replaces). This file holds the contract every sub-agent works to (the file split, the names
files share, and the decisions the contract takes where C3 leaves a choice or cannot be met as written), then the record
of each sub-assignment, its commits, each review finding and where it went, and what is left open.

## The four rules (every line is judged against them)

1. A data page holds payload only. Its producer or its landing writes each byte once per use, consumers read it, and
   nothing else touches it: no header, no flag, no zeroing, no tail handling in memory. Masks pick values within a page;
   masked-off lanes are guarded in registers (select, never a multiply by a 0/1 mask; a reduction puts its identity into
   masked lanes).
2. State lives apart from the data, in the plan's state section (the landed list's own section). Each state word has one
   writer; counts take arrivals. No state word is reset while its plan stands: phases are 2-bit Gray codes, supplied
   words and counts only rise.
3. Everything is placed once, at compile time, by the one allocator in `mesh_plan`, with modulo variable expansion by
   lifetime (u copies from happens-before; `executions` is no input of the plan). A program holds exactly the pages its
   plan uses. Nothing is claimed at attach.
4. No lane waits. Read-after-write is a masked first pass, then a later pass whose size comes through indirect dispatch
   and which the command buffer holds with `encodeWaitForEvent:value:` on a rising event the client signals from rising
   landed counts. Every remaining wait ends in bounded time.

## Rules (every sub-agent)

- Read, in this order: the change request in full (C1's turns are the rulings; C3 holds where it differs), this
  contract, your brief, then mesh `AGENTS.md` ("Collectives and tensor parallelism"), design/rubric-actionable.md,
  metal-microbench `AGENTS.md`, docs/principles.md and docs/measurement.md. Never open docs/facts.md at any revision.
  Never use or cite design/transport.md (old comments cite it; leave them, add none).
- Work in /Users/mdot/dox/mesh and /Users/mdot/metal-microbench, on `main`. No worktrees, no branches, no copies of
  source anywhere (no /tmp or scratchpad copies). Never `git reset`, `git clean`, `git stash` (nor read stashes), never
  copy source backups. Old source lives only in commit history.
- Edit only the files your brief gives you. Other sub-agents edit their files in the same working tree at the same
  time. If you need something in another owner's file, do not edit it: say so in your report and the coordinator routes
  it. You may read any file.
- Run nothing: no bridge, no e2b run, no program, no test, no measurement, no ssh, on any machine. Write no tests and no
  evaluators. Do not touch deployment, launch or scheduled-job scripts (rdma/mesh_hourly.py, rdma/mesh_preflight.py,
  rdma/mesh_graph.py, bin/, etc/, tools/mesh/ in metal-microbench unless your brief names a file there). Never edit
  xonotic/. The failport process lifecycle (rdma/mesh-heartbeat.c, mesh-children.c, mesh-child.c, mesh-lanes.c) is
  reused verbatim: do not change it. Never send SIGKILL to a bridge; never destroy queue pairs that have posted receives.
- Compiling is allowed. Judge your file by compiling it alone against the headers (commands below), because other
  owners' half-done files may break a whole build. The coordinator runs the three document builds at integration:
  `make -C rdma`, `make -C rdma native-audit` (in /Users/mdot/dox/mesh) and `make .build/e2b_replay` (in
  /Users/mdot/metal-microbench). Do not run them yourself unless your brief says so (concurrent makes race on the same
  outputs).
  - mesh C file: `cd /Users/mdot/dox/mesh/rdma && cc -DMESH_PROVIDER_RESERVED_QPS=1 -ffp-contract=off -fblocks -O2 -Wall
    -Wextra -Werror -fsyntax-only <file>.c`
  - mesh-metal.m: `cd /Users/mdot/dox/mesh/rdma && cc -std=c11 -fblocks -O2 -Wall -Wextra -Werror -fsyntax-only -I .
    mesh-metal.m`
  - metal_recording.m, metal_native.m: `cd /Users/mdot/metal-microbench && cc -std=c11 -fblocks -O2 -Wall -Wextra
    -Werror -fsyntax-only -I .build metal_recording.m metal_native.m`
  - e2b_replay.m: `cd /Users/mdot/metal-microbench && cc -std=c11 -fblocks -O2 -Wall -Wextra -Werror -fsyntax-only -I .
    -I .build/dawn-sdk/include -I .build/litert-sdk -I .build/LiteRT -I ../dox/mesh/rdma tools/e2b_replay.m`
  - prepare_e2b.py: `python3 -m py_compile tools/prepare_e2b.py` (writes nothing you keep: delete the __pycache__ entry
    it makes only if it was not there before; never commit it).
- Commit and push as you go. Commit your own files only, by explicit pathspec: `git -C <repo> add <your new file>` for a
  new file, then `git -C <repo> commit -m "..." -- <your files>`. Never `git add -A` or `git add .`. mesh has untracked
  build artifacts (rdma/_mesh_c.*, rdma/libnccl-mesh.*, rdma/libsleef.a, rdma/linear-chain, rdma/mesh-heartbeat,
  rdma/torch-mesh/, NODE-STANDUPS.md) and a secret tailscale.key: never commit any of them. Commit as soon as your file
  compiles alone; the message names the file and what changed, says which builds it passes ("compiles alone" where the
  whole build waits on other owners), and ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  Then `git -C <repo> push origin main`. If the push is refused because origin moved, run `git -C <repo> pull --rebase`
  and push again; if git refuses the pull because other agents' files are modified in the tree, do not stash, reset or
  commit their files: wait a minute and try again, and after three tries leave the push to the coordinator and say so in
  your report. If `.git/index.lock` exists, wait 1 s and retry, at most 10 times. A commit in mesh runs the C9 hook
  (bin/mesh-c9-review.py, up to 180 s): read its remarks; a graph computation outside mesh-graph.{h,c} moves into the
  graph module (tell the coordinator if that is another owner's file).
- Headers first: if you own a header, land its interface exactly as this contract writes it, in your first commit,
  within your first minutes, so other owners compile against it.
- Comments: follow your file's existing form. mesh rdma files carry block comments that cite the design by section;
  cite "design/disjoint-pages-change-request.md, C3.n" (and this contract by section where it decides). Where you
  reverse another session's behavior, leave a dated note at the spot (2026-11-04): what, why, what must stay. Where
  your code has to differ from C3, say so at the spot and in your report. None of this change's metal-microbench files
  is in the set tools/source_standard.py counts (ENGINE_SRCS and server/): keep each file's existing comment form there
  too (`// ../dox/mesh/...` citations and dated notes).
- No invented mechanisms. Every mechanism comes from C3, from the literature C3 cites, or from this contract's decisions
  below (each of which names its source). No hashing, receipts, credits, gates, filters, locks, leases, slots, rings,
  retries of a failed transfer, fallbacks or second forms. One configuration runs in a runtime; no call site handles two
  forms; a replacement covers what the old code did and the old code is removed in the same piece of work.
- Every failure is an error value, never a hang or a hardware fault. A spin or wait is allowed only where it ends in
  bounded time even if its event never comes. No GPU kernel spins.
- Queue pairs, lanes, streams and links are the bridge's alone: none reaches collective, data or GPU code.
- If something C3 or this contract requires is impossible with the real API or hardware, do not invent a substitute:
  finish everything else, and report the conflict with its evidence (the code, the SDK header, the API documentation).
- Report (your final message, short): files and functions written or removed; every difference from C3 or this
  contract, and why; what you need from other owners; your commit shas and whether they are pushed; compile results.

## Where things live (owner = sub-assignment)

| file | owner |
|---|---|
| mesh rdma/mesh.h, rdma/mesh-map.h | 1 |
| mesh rdma/mesh-graph.h, rdma/mesh-graph.c | 2 |
| mesh rdma/mesh-collective.h, rdma/mesh-collective.c | 3 |
| mesh rdma/mesh-dataflow.h, rdma/mesh-dataflow.c, rdma/mesh-call.h, rdma/mesh-call.c, rdma/mesh.py | 5 |
| mesh rdma/mesh-flow.c | 6 |
| mesh rdma/mesh-link.c, rdma/mesh-records.c, rdma/mesh-stat.c, rdma/mesh-bridge.h | 7 |
| mesh rdma/mesh-metal.h, rdma/mesh-metal.m | 8 |
| metal-microbench metal_recording.h, metal_recording.m | 9, then 10, then 11 |
| metal-microbench metal_native.h, metal_native.m, native_projection.metal, tools/native_bench.m | 12 |
| metal-microbench tools/e2b_replay.m | 13, then 4 |
| metal-microbench tools/prepare_e2b.py | 14 |
| reviews (written findings only) | 15 (mesh side), 16 (GPU and program side), 17 (the four rules across both) |
| rdma/Makefile, this file, INDEX.md | coordinator |

## Decisions this contract takes, and why

Each is either C3 made exact, or a place where C3's text cannot be met as written; the latter are departures, recorded
here with their evidence and in the final report.

- **D1 (departure from C3.1's target): counts per received range and copy, not one per node.** C3.1 holds execution t's
  later pass "until the node's landed count reaches execution t's target, (t + 1) x the node's receives per execution".
  A decode step of e2b has many exchanges in sequence: every layer's attention and contribution partials are published
  by each rank and read by the others (tools/prepare_e2b.py:1868-1892, the publications appended layer by layer;
  tools/e2b_replay.m:239-257, every rank's sends declared as routes), and rank m's exchange k + 1 is computed from its
  read of exchange k. A pass held for every receive of its execution would wait on exchange k + 1's receives, which the
  peers send only after their own reads of exchange k, which wait the same way: every pass would be held to its
  deadline. The literature C3 cites counts once per unit of work, not per execution: Portals 4's counting events, one
  counter per operation set with its own threshold (the change request, "Completion counted once per unit of work"),
  and PTX's mbarrier, one barrier per tile. So the count is per received range (one declared transfer at its
  destination), and per copy: with u >= 2, execution t + 1's landings into the other copy can arrive before t's are all
  in, so one count per range could reach (t + 1) x P with some of t's pages missing and read them as cancelled; within
  one copy the allocator's order puts every landing of t + u after t's reads. Execution t's target for range g is
  (floor(t / u) + 1) x P_g on the count of copy t mod u. The event value per pass still only rises (D7).
- **D2: the run's length is given at its start, not to the plan.** C3.3: "`executions` is no longer an input of
  `mesh_plan`: a plan runs any number of executions." The bridge must still post exactly the receives its peers will
  send: a SEND fills the oldest receive posted on its queue pair (data-model section 3), and a receive left posted when a
  plan ends is a lost link (mesh-flow.c link_close: "a posted receive is never taken back but by the child's exit").
  So the client's start says how many executions to serve: `mesh_transfers_start(context, executions)` raises the
  plan's state word `executions`, which only rises. Compile time never sees it.
- **D3: the state section is initialized once, where it is placed.** C3.7: "The fields start at 00 when the allocator
  places the list, once per plan, as `mbarrier.init` runs once per kernel." The plan's fresh pages hold whatever their
  last holder left, so mesh_plan writes the state section's words to their starting values once, when it places them,
  and never again while the plan stands. This is that initialization, not the zeroing C3.7 removes ("The only zeroing
  left is mesh-link.c's zeroing of the region's header"): no data page is written by the plan.
- **D4: capture, then plan, then bind.** C3.2 has the allocator place the pooled intermediates, whose sizes the
  recorder learns only by capturing the delegate's commands, while LiteRT needs a buffer to encode against when it is
  captured. So the program captures first, against stand-in memory that holds no value of the run (nothing reads it
  after the capture: the run executes only the recorded commands, bound to the placed pages), then plans every value,
  then binds each captured buffer to its placed pages and builds the indirect command buffers. Everything the recorder
  splices into a kernel that depends on the plan is a binding, never a literal, so nothing is compiled from the plan.
  The recorder's own greedy pool (metal_recording.m MetalPrepareDataflow's smallest-fit by lifetime) is a second
  allocator and goes: each value it renamed is reported with its lifetime and placed by mesh_plan.
- **D5: u from least lags, under one monotonicity condition.** C3.3 defines L per value as the smallest lag at which
  every read in t happens before the write in t + L, and u the largest L. Under happens-before (no fixed cycle times),
  a path at lag L implies one at every later lag only where each write's next instance follows it: so the allocator
  requires every routed value's write vertex to reach its own next instance at lag 1 (least lag 1 over a cycle through
  it), and fails the plan with ENOMEM naming the value where one does not, as C3.3 fails a value with no L. With that,
  the least lags (Dijkstra over the template, lags as weights) decide L, u and every interference test exactly, with
  no table over executions.
- **D6: the first pass predicates threadgroups.** C3.1 computes "the lanes whose pages have landed. The rest are
  predicated off." The delegate's kernels share work within a threadgroup (threadgroup_barrier, SIMD-group reductions),
  so a lane cannot compute while another of its threadgroup is off: the predicate is the threadgroup's (its SIMD-groups'
  ballots reduced), and the compaction is over threadgroups, by simd_ballot and simd_prefix_exclusive_sum (Blelloch).
- **D7: the decode is one Metal 3 command buffer.** `encodeWaitForEvent:value:` is MTLCommandBuffer's
  (Metal.framework/Headers/MTLCommandBuffer.h:409; "may only be called if there is no current command encoder");
  MTL4CommandBuffer has none, only MTL4CommandQueue's `waitForEvent:value:` between whole command buffers
  (MTL4CommandQueue.h:274). C3 names the first, so the decode's submission (tools/e2b_replay.m's struct Submission,
  MTL4CommitFeedback) becomes an MTLCommandBuffer on the recorder's queue: compute encoders execute the recorded
  ranges, and before each later pass the encoder ends and `encodeWaitForEvent:value:` holds the rest of the buffer.
  The event is an MTLSharedEvent; pass p of the run waits for value p + 1; the client alone sets its value.
- **D8: one client attached at a time stays.** C3.4's "any number of programs hold disjoint sections of a node's region
  at once" holds for pages: every plan's claim is its own, and a retired plan's claim coexists with the next plan's
  until the bridge releases it. The bridge serving several clients' plans at once (one page list each) is not among
  C3's items or C4's entries; hdr `client` and its compare-and-swap stay. Open at the end.
- **D9: the GPU raises a supplied word by one.** C3.7: "A supplied word has one writer, its range's producer, which
  stores t + 1 after its last write in execution t." A GPU producer's executions run in order, one release each, so it
  raises the word with an atomic add of 1 (the value is then t + 1) and needs no t; the host producer stores t + 1.
- **D10: the ports' announcement stays; the plan call's wait goes.** C4 removes mesh_plan_erased and its seq_cst fence
  pair. The bridge's turn releasing a retired plan's pages once no port holds them (C3.4's deferred reclamation: RCU's
  call_rcu; K. Fraser, Practical lock-freedom, 2004, epoch-based reclamation) still has to know a port that read the old
  plan cannot begin serving it after the release looked: a port announces the generation it loads (`prepared`), fences
  and reads `hops` and `plan` again, withdrawing where they moved (Fraser's announced epoch and its re-read); the
  release fences before it reads the announcements. Nobody waits.
- **D11: LiteRT's weights are LiteRT's.** e2b's model weights are the delegate's own GPU buffers inside its compiled
  models (metal_recording.m MetalNativeSubstitute reads them as the delegate bound them), never region pages; C3.2's
  "weights" in e2b are the region values the program binds (its native stages' operands, the KV cache, the prepared host
  inputs), all of which the plan places as invariants.

## Names files share

### The region (mesh.h, owner 1)

- `MESH_VERSION` 109. `#define MESH_LANE_BYTES 16u`: the bytes one lane loads of the phase list (64 receives' phases).
- `struct hdr`: no field moves. `executions` (offset 44) becomes `uint32_t vacant14`; `word_off`, `word_capacity`,
  `landed_off`, `landed_capacity`, `landed_stride` become `uint64_t vacant9 .. vacant13`, never read or written;
  `map_room` stays; appended after it: `_Atomic uint64_t state;` (sizeof 408): the region offset of the current plan's
  struct mesh_state, written by mesh_plan before `hops` (0: none).
- The phase code: `static inline uint32_t mesh_phase(uint64_t k){ k&=3; return (uint32_t)(k^(k>>1)); }` (Gray 00 01 11
  10). A receive of copy c lands in executions c, c + u, c + 2u, ...; after execution t's landing its field holds
  mesh_phase(t / u + 1); the landing flips bit `(t / u) & 1` of the field (one atomic XOR of one bit, which never
  carries into the next field). A reader of execution t compares the field with mesh_phase(t / u + 1): equal, landed;
  equal to mesh_phase(t / u), one behind (not yet, or cancelled); anything else, a write ran ahead of the allocator's
  order: an error.
- `struct mesh_status { _Atomic uint32_t behind,ahead,first,padding; };` (16 bytes), the GPU's alone: `behind` the later
  passes that found a field one behind, `ahead` those that found anything else, `first` the greatest ~t over them
  (0 none: the first failed execution is ~first).
- `struct mesh_state { _Alignas(64) uint32_t copies,receives,ranges,releases; uint64_t phases,counts,supplied,status;
  _Atomic uint64_t executions; uint64_t padding; };` (64 bytes), at the start of the plan's state section:
  - `copies` u; `receives` this node's receives a copy (the sum of its received ranges' pages); `ranges` this node's
    received ranges; `releases` this node's source ranges. The offsets below are region offsets.
  - `phases`: the phase list. For received range g (g < ranges, in the plan's demand order) and copy c, a group of
    `mesh_group_bytes(P_g)` bytes, `static inline uint64_t mesh_group_bytes(uint32_t pages){ return ((uint64_t)pages
    + 63) / 64 * MESH_LANE_BYTES; }`, groups in (g, c) order, g major; page k of the range (k < P_g) has its field at
    bits [2 (k % 32), 2 (k % 32) + 2) of uint64 word k / 32 of its group. A group never shares a lane's 16 bytes with
    another.
  - `counts`: ranges x copies uint64 words, range g's copy c at index g x copies + c. Each takes arrivals (+1 a
    landing, D1; a cancellation raises it to its final value, below) and only rises.
  - `supplied`: releases uint64 words, release r (this node's r-th source range in the plan's demand order). Its one
    writer is the range's producer, which raises it to t + 1 after its last write in execution t (D9).
  - `status`: one struct mesh_status.
  - `executions`: the executions the bridge serves; raised (never lowered) by the client's start (D2).
  - Every word starts at 0 when mesh_plan places the section (D3). No word of it is ever written to an earlier value.
- `static inline struct mesh_state *mesh_plan_state(struct hdr *m)`: hdr `state` (acquire) as a pointer, NULL where 0.
- `struct mesh_hop { uint32_t page,node,direction,field; uint64_t word,count; uint32_t pages,padding; uint64_t serial;
  };` (48 bytes; `word` at 16, `count` at 24, `pages` at 32, `serial` at 40). One entry is one hop of one page in one
  copy. `hops` = u x n: copy c's n entries at [c n, c n + n), every copy the same hops in the same order with the same
  `node`, `direction`, `serial`; `page`, `word`, `field`, `count` are copy c's own. Execution t runs copy t mod u.
  - A receive: `page` its page; `word` the region offset of the uint64 holding its field, `field` the field's low bit
    (even, < 64); `count` the region offset of its range's count word for this copy; `pages` its range's pages P.
  - A send from its range's source: `word` the region offset of the range's supplied word; `field` MESH_ABSENT;
    `count` 0; `pages` 0. Execution t's send is released once the word reads (acquire) at least t + 1.
  - A forwarding send (a relay's, or a destination's that sends the page on): `word` and `field` of its node's receive
    of the same page in the same copy; `count` 0; `pages` 0. Execution t's send is released once that field reads
    (acquire) mesh_phase(t / u + 1).
  - Both ends: for adjacent a and b, copy c's entries at a naming b with MESH_SEND and those at b naming a with
    MESH_RECEIVE are the same hops in the same order. u is the collective's (D5: every rank computes the same), so both
    ends hold the same copies.
- `struct mesh_publication { _Alignas(64) uint32_t sends,pages; uint64_t phases,count,state; struct mesh_target
  targets[]; };` (offsets 0, 4, 8, 16, 24, targets 32). mesh_plan writes every row it claims; all fields 0 but:
  - a received range's first row (copy 0's first row): `pages` P, `phases` the region offset of its copy-0 group (copy
    c's at phases + c x mesh_group_bytes(P)), `count` the region offset of its copy-0 count word (copy c's at count +
    8 c), `state` hdr `state`;
  - a source range's first row: `sends` 1, `targets[0]` = {stream: the region offset of its supplied word, count 1,
    stride 0}, `state` hdr `state`.
- Removed: `MESH_SUPPLIED`, `MESH_WORD_STRIDE`, `mesh_supplied`, `mesh_landed_record`, `struct mesh_cancel_range`,
  `struct mesh_cancellation`, `mesh_cancel`, `struct prepared_publication` (the GPU's publication records: the recorder
  binds the supplied word itself). `mesh_word_mask`, `mesh_bits_set`, `mesh_bits_clear` stay (the claims use them).

### The layout (mesh-map.h mesh_layout, owner 1)

As now without the supplied words' and landed list's areas: the row planes, the arena bitmap, the page table, the
buffers, the link table, the publications, the page lists (`hop_capacity` entries of 48 bytes), the bridge's pages, the
map's records (`map_room`), then the arena at a block boundary.

### The program's claim and its sections (mesh-dataflow.h, mesh-call.h, owner 5)

- `struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; int fd; };` rows, window, local and the cursors go.
  rdma/mesh.py mirrors it.
- `mesh_attach` takes the client word (compare-and-swap, a gone client retired, as now) and claims nothing: no rows, no
  pages, no wait for a retired client's release.
- A plan's claim (the only place rows and pages are taken; mesh_plan's one call):
  ```c
  struct mesh_claim { struct mesh_range rows,window,local; };
  /* rows: `rows` free page-table rows (MESH_ROW_OWN), one run; window: `window` free arena pages inside the registered
     window, local: `local` outside it (inside it where the window is the whole arena), each one block-aligned run, first
     fit; recorded in the rows' first row (its head: the client, `generation`, the three runs) for the bridge's release.
     Taken whole or not at all: 0, or ENOMEM with nothing taken.  rows >= 1 (the head) */
  int mesh_claim_take(struct mesh_ctx *,uint64_t generation,uint32_t rows,uint32_t window,uint32_t local,
                      struct mesh_claim *);
  /* this client's claim of `generation` closed (others 0), or every claim of this client but that one (others 1) */
  void mesh_claim_close(struct mesh_ctx *,uint64_t generation,int others);
  ```
  A run is taken by setting its bits atomically; a bit already set (the bridge's release and an attaching client run
  at once) undoes what this call set and scans on past it, so the scan ends in at most one pass over the bitmap.
- `mesh_retire(m, client)`: every claim of the client closed, then the client word cleared (as now).
- `uint32_t mesh_retired_release(struct hdr *)`: a seq_cst fence, then each closed claim whose generation no port's
  `prepared` holds released (its pages' bits, then its rows' bits, by the winner of the head's owner exchange, as now);
  the closed claims kept are counted. Runs in the bridge parent (mesh-records.c clients_turn, mesh-link.c region_adopt).
- `struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };` unchanged. `mesh_section_create`
  is removed: mesh_plan names every value's rows (below). `void *mesh_section_address(struct mesh_ctx *,struct
  mesh_section,uint32_t index)` and `mesh_section_constant` stay. `mesh_backing_bind` and `mesh_device_bind` stay.
- `int mesh_transfers_start(struct mesh_ctx *,uint64_t executions);`: the plan's state word `executions` raised to
  `executions` (a compare-and-swap loop that never lowers it; release), `plan` woken, then the bounded wait for the ports
  (as now). EINVAL where no plan is written (hdr `state` 0).
- Removed: `mesh_plan_erased`, `mesh_publication_prepare`, mesh_claim's claim at attach, the wait at attach, the
  `local` run and its laying out of values at creation.

### The plan call (mesh-collective.h, owner 3)

```c
enum { MESH_INVARIANT, MESH_EXECUTION };
/* a value at this node.  In: section.bytes (> 0), kind, wire, state, and for a local MESH_EXECUTION value its first and
   last positions on this node's own program order in one execution, begin <= end.  Out: section, every field */
struct mesh_value { struct mesh_section section; uint32_t kind,wire,state,begin,end,padding; };
/* one declared transfer, the same at every rank but `value`: the index in `values` of this node's value of the range
   where this node is its source or a destination, else MESH_ABSENT; `bytes` the range's bytes at every rank */
struct mesh_declared { uint32_t source,destination,identity,order,value,padding; size_t bytes; };
/* a program's data edge at one node: event `from` happens before event `to`, `to` in execution t + lag where `from`
   is in t.  An event is declared[i]'s production at its source (its kinds bit set: MESH_FROM_PRODUCED for `from`,
   MESH_TO_PRODUCED for `to`) or its consumption at declared[i]'s destination (bit clear); both events at one node */
#define MESH_FROM_PRODUCED 1u
#define MESH_TO_PRODUCED 2u
struct mesh_dependence { uint32_t from,to,lag,kinds; };
int mesh_plan(struct mesh_ctx *,const struct mesh_declared *,uint32_t count,struct mesh_value *,uint32_t values,
              const struct mesh_dependence *,uint32_t dependences);
/* a host producer's release of execution t of a source range: its supplied word stored t + 1 (release) */
void mesh_host_publish(struct mesh_ctx *,struct mesh_section,uint64_t execution);
/* execution t of a received range: 1 where every field of its copy t mod u reads mesh_phase(t / u + 1) (acquire);
   UINT64_MAX where a field reads neither that nor mesh_phase(t / u) (ahead), or where its count has reached
   (t / u + 1) x P with a field still behind (cancelled); else 0 (not yet) */
uint64_t mesh_host_arrived(struct mesh_ctx *,struct mesh_section,uint64_t execution);
```
- Every rank passes the same declared transfers and the same dependences in the same order (every rank's events, so
  every rank builds the one template); `values` are this node's own.
- A value a declared transfer names here is MESH_EXECUTION and wire, and one demand's alone (EINVAL otherwise). Wire
  values lie in the registered window; the rest outside it.
- `state` 1: the value is state (a GPU counter the program keeps): mesh_plan writes it to 0 once when it places it
  (D3). `state` 0: data, never written by the plan.
- Out, for each value: `section.first` its first row, `pages` P x block (P its blocks), `bytes` as given, `count` its
  copies (u for a routed value; 1 for every other), `stride` P: copy c's rows at first + c x P + k, k < P, and its first
  byte at mesh_section_address(section, c). A program binds execution t of a routed value to copy t mod count.
- `mesh_host_inputs` is removed (it did nothing).
- Returns 0; EINVAL (a malformed transfer, value or dependence; a lag-0 cycle); EHOSTUNREACH; ENOSPC (u x n entries past
  hdr `hop_capacity`); ENOMEM (no free run for the claim; a value with no unroll bound or whose write does not reach its
  next instance at lag 1 (D5), named on stderr by its index and the range's identity; the call's own memory); ESTALE;
  or the map reader's error. Every failure but ESTALE and the second map read's is found before anything is written,
  and leaves the previous plan standing. There is no ETIMEDOUT: nothing waits.

### The allocator (mesh-collective.c, owner 3; the graph algorithms owner 2's)

- The template: the demands' P, W, L, F, C vertices and the transport's lag-0 edges as now; each mesh_dependence one
  edge from its `from` event's vertex (P of the demand, or C at the destination) to its `to` event's, of its lag.
- Least lags D(x -> y): mesh_graph_lags over the template from every routed value's reads and write (at every node) to
  every routed value's write. A lag-0 cycle (D(x -> x) == 0) is EINVAL. Every routed write vertex w must have
  D(w -> w) == 1 (D5), else ENOMEM naming its value.
- L(v) = the largest D(r -> write v) over v's reads r; u = the largest L(v) over every routed value at every node (1
  where there is none). Every rank computes the same u.
- Items at this node: each routed value's copies (v, c), c < u, P blocks each (a relay's value among them: this node
  on a demand's trees but neither its source nor a destination, which no `values` entry names and no row holds, a
  routed value like any other); each local invariant (one); each local execution value (one); the state section (one,
  invariant, no rows). Two colourings by mesh_graph_left_edge, greedy by
  size (largest first, then value order, then copy): the wire items in the window space, every other item in the local
  space, each in blocks.
- Interference (two items interfere unless the rule lets them share):
  - an invariant (the state section among them) interferes with every other item;
  - two local execution values share only where [begin, end] do not overlap; a local execution value interferes with
    every routed copy;
  - routed copies (v, c) and (w, c'), with D_vw = the largest D(r -> write w) over v's reads r, and d = (c' - c) mod u:
    where d > 0 they share only if d >= D_vw and u - d >= D_wv; where d == 0 (and v != w) only if D_vw <= u and D_wv
    <= u and (D_vw == 0 or D_wv == 0). Two copies of one value (v == w, c != c') follow the d > 0 rule.
  These are the distances at which their executions meet (t and t' with t' - t = d mod u); by D5 an order at the least
  such distance holds at every larger one of the same class.
- Cost: Dijkstra from each source, O(sources x (E + V log V)) once a plan, and the colouring's walk, quadratic at worst
  in this node's items (values x u). Nothing depends on the run's length.

### Writing a plan (mesh-collective.c, owner 3)

1. Compute everything first: demands, routing, template, least lags, u, items, colourings, this node's entries of one
   copy (plan_walk) and every room: ENOSPC past `hop_capacity`.
2. The generation g = hdr `plan` + 1 (never 0). Claim: mesh_claim_take(context, g, rows, window blocks x block, local
   blocks x block): rows the sum of every value's count x P. ENOMEM leaves the previous plan standing.
3. Initialize the state section (its struct mesh_state, every other word 0) and every `state` value's bytes to 0 (D3);
   bind every value's rows (mesh_backing_bind, copy c's rows to copy c's pages); write every claimed row's publication.
   No data page is written.
4. `hops` 0 (release), a release fence, `plan` woken.
5. The entries, u x n; hdr `state`; `plan` = g.
6. The map read again: ESTALE where its serial moved (or the reader's error): mesh_claim_close(context, g, 0), `hops`
   stays 0, return the error (the previous plan no longer stands; close its claims too: mesh_claim_close(context, g,
   1)).
7. `hops` (release), `plan` woken.
8. mesh_claim_close(context, g, 1): every earlier claim of this client closed; the bridge releases each once no port
   holds its generation (D10). Nothing waits for it.

### The bridge (mesh-flow.c, owner 6)

- Load (link_prepare), on `hops` (acquire) with a plan state: u from the plan's struct mesh_state; the pair's units
  picked over the u x n template in list order and assigned by mesh_graph_list_schedule as now (both ends the same: u
  is the collective's); one record per template entry on this cable, each posted once per execution of its copy.
  `prepared` announced, a seq_cst fence, `hops` and `plan` read again, withdrawn where they moved (D10).
- Serve executions t < the plan's `executions` (acquire, read each pass), each lane in execution order: execution t's
  records are copy t mod u's on that lane. A receive is posted as its lane's room allows (MESH_LANE_UNITS, frames), in
  order, on nothing else. A record's j-th completion since the load is execution c + j u (a per-record counter, the
  child's alone).
- A landing of execution t: after the poll's acquire, `atomic_fetch_xor(word, 1 << (field + ((t / u) & 1)), release)`,
  then `atomic_fetch_add(count, 1, release)`; where the count it made is a multiple of `pages`, the count word's waiters
  woken (os_sync_wake_by_address_all, OS_SYNC_WAKE_BY_ADDRESS_SHARED). A failed completion, or one of other than a
  whole page, fails the link as now.
- A send of execution t: posted in its lane's order once released (source: supplied >= t + 1; forwarding: its field
  == mesh_phase(t / u + 1)), as the lane's gate and MESH_LANE_UNITS allow.
- Done while every record has served every execution below `executions`; a rise of `executions` serves again.
- A cancellation arrives on the count. Where a plan ends with receives not landed (link_close: a re-plan, the client
  gone, the link serving no lane, the child closing) and where the parent cancels (plan_cancel from streams_cancel and
  streams_let_go, the plan read whole as now), each such receive's count word is raised (a compare-and-swap loop that
  never lowers it; release) to its final value, P x the number of executions t < `executions` with t mod u == its
  copy, then woken. No phase is written for a cancellation, so a reader finds the receive one phase behind.
  plan_cancel and the release of retired claims (mesh_retired_release) both run on the bridge parent's one event-loop
  thread; owner 6 states where that is so, so that a cancellation's store computed from a plan's entries lands in that
  plan's pages while they are still held.
- Removed: landed_cancel, the cancel word, the OR into a landed record, the references to mesh_plan_erased.

### The GPU's side of the plan's words (mesh-metal.h, owner 8)

```c
/* the plan's state section wrapped once for the GPU: `state` an MTLBuffer over its pages (no copy), `words` its struct
   mesh_state, `offset` its region offset (hdr state), `bytes` its length; EINVAL where no plan is written */
struct mesh_metal_transport { void *state; struct mesh_state *words; uint64_t offset,bytes; };
int mesh_metal_transport_create(struct mesh_ctx *,void *device,struct mesh_metal_transport *);
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
/* a received range's words, byte offsets in the transport's buffer: copy c's phase group at phases + c x group, its
   count word at count + 8 c; pages P; copies u */
struct mesh_metal_input { uint64_t phases,group,count; uint32_t pages,copies; };
int mesh_metal_receive_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,struct mesh_metal_input *);
/* a source range's supplied word, its byte offset in the transport's buffer */
struct mesh_metal_publication { uint64_t supplied; };
int mesh_metal_publication_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,
                                   struct mesh_metal_publication *);
/* the client's signalling of the queue's event (C3.1): a wait of `value` is met once the count word at byte offset
   `count` of the transport's buffer reads at least `target`; the event (an id<MTLSharedEvent>) is set to each value once
   every wait of that value or less is met, in order.  The thread waits on each count word with
   os_sync_wait_on_address_with_timeout (OS_SYNC_WAIT_ON_ADDRESS_SHARED), woken by the bridge's arrivals; a value not
   met within MESH_PLATFORM_NS of the last value set, or a stop, sets the event to `final` at once and ends */
struct mesh_metal_wait { uint64_t count,target,value; };
struct mesh_metal_waiter;
int mesh_metal_waits_start(struct mesh_metal_transport *,void *event,const struct mesh_metal_wait *,size_t waits,
                           uint64_t final,struct mesh_metal_waiter **);
void mesh_metal_waits_stop(struct mesh_metal_waiter *);  /* async-signal-safe: one store and one wake */
int mesh_metal_waits_end(struct mesh_metal_waiter *);    /* joined and freed: 0, ETIMEDOUT (a deadline), ECANCELED (a stop) */
/* the GPU's status after the run (struct mesh_status): 0 where no pass failed; EPROTO where one found a field ahead;
   else `cause` where it is not 0, else ENOLINK (the bridge cancelled); *first the first failed execution */
int mesh_metal_status(struct mesh_metal_transport *,int cause,uint64_t *first);
```
The program's cancellation (`transport.cancel`, mesh_cancel) goes: a stop is mesh_metal_waits_stop. The direct
publication's statuses (struct mesh_output_status, the `invocations` argument) go: the recorder's restoring counter
replaces them (owner 11). Nothing here creates a section.

### The recorder (metal_recording.h, owner 9)

```objc
enum { METAL_INVARIANT, METAL_EXECUTION };     // mesh-collective.h MESH_INVARIANT, MESH_EXECUTION
// a value the capture holds, which the plan places (mesh-collective.h struct mesh_value): state 1 zeroed once where
// placed; begin, end its first and last dependency levels in one repetition, for METAL_EXECUTION
struct MetalValue { NSUInteger bytes; uint32_t kind, state, begin, end; };
// a received operand's words (mesh-metal.h struct mesh_metal_input), byte offsets in the state buffer
struct MetalRemote { uint64_t phases, group, count; uint32_t pages, copies; };
// a wait a pass is held on (mesh-metal.h struct mesh_metal_wait, the same layout)
struct MetalWait { uint64_t count, target, value; };
@interface MetalRecorder : NSObject
@property(readonly) id<MTLDevice> device;
@property(readonly) id<MTLCommandQueue> queue;
@property(readonly) id<MTLCommandQueue> recordingQueue;
- (instancetype)initWithDevice:(id<MTLDevice>)device;
// before the plan: encode's commands captured once for `repetitions` executions; nothing bound, nothing built
- (id)capture:(void (^)(void))encode repetitions:(NSUInteger)repetitions;
// inside encode: an operand a peer writes (its readers become masked first passes), an operand this node sends
- (void)receive:(id<MTLBuffer>)operand;
- (void)publish:(id<MTLBuffer>)operand;
// the capture's dependency levels, and its own values (its renamed intermediates, its passes' data, its counters) in
// one fixed order, their levels offset by `level`; values NULL counts them
- (uint32_t)levels:(id)capture;
- (NSUInteger)values:(id)capture level:(uint32_t)level into:(struct MetalValue *)values;
// after the plan, per capture: execution t binds an operand at offsets[t % copies] + t x stride of `buffer` (+ output
// where it writes); a value at `offset` of `buffer`; a received operand's words; a sent operand's supplied word; the
// plan's state buffer and its status word
- (void)place:(id)capture operand:(id<MTLBuffer>)operand in:(id<MTLBuffer>)buffer offsets:(const NSUInteger *)offsets
    copies:(NSUInteger)copies stride:(NSUInteger)stride output:(NSUInteger)output;
- (void)place:(id)capture value:(NSUInteger)index in:(id<MTLBuffer>)buffer offset:(NSUInteger)offset;
- (void)received:(id)capture operand:(id<MTLBuffer>)operand remote:(struct MetalRemote)remote;
- (void)published:(id)capture operand:(id<MTLBuffer>)operand supplied:(NSUInteger)offset;
- (void)state:(id)capture buffer:(id<MTLBuffer>)state status:(NSUInteger)offset;
// the capture's commands bound and its indirect command buffers built
- (struct MetalReplay)replay:(id)capture;
@end
void MetalReplayDestroy(struct MetalReplay *);
// a recording with no later pass (the index commands, the prefill); raises on a range that holds one
void MetalReplayExecute(id encoder, struct MetalReplay *, NSRange range);
void MetalReplayOptimize(id<MTLBlitCommandEncoder> encoder, struct MetalReplay *, NSRange range);
NSIndexSet *MetalReplayPublishing(struct MetalReplay *);
// the decode: repetitions [range) into `command`, each later pass after encodeWaitForEvent(event, base + p + 1), p
// counted from 0; its waits written (a received operand a pass); returns the passes encoded, *written the waits
NSUInteger MetalReplayEncode(id<MTLCommandBuffer> command, struct MetalReplay *, NSRange range, id<MTLSharedEvent> event,
    uint64_t base, struct MetalWait *waits, NSUInteger room, NSUInteger *written);
```
- `record:repetitions:storage:`, `operand:stride:outputOffset:`, the old `receive:landed:offset:stride:first:count:` and
  `publish:records:count:notice:completion:stride:ranges:bytes:rangeBytes:rangeStride:` go; so do MetalPrepareDataflow's
  pool (D4) and its placement diagnostics (E2B_DEBUG_UNPOOLED, E2B_DEBUG_POOL_LARGEST, E2B_DEBUG_POOL_AT).
- The indirect command buffers bind each operand and value where it is placed. A value two captured buffers share
  pages with is ordered by the level barriers within a repetition; every repetition's first command takes a barrier,
  since placements may share pages across repetitions.

### The masked first pass and the later pass (metal_recording.m, owner 10)

For each recorded dispatch that reads a received operand, in each repetition t:
- **First pass** (the recorded command, rewritten): at entry each threadgroup loads the phase group of copy t mod u of
  each received operand it reads (MESH_LANE_BYTES a lane, the group's ceil(P / 64) loads spread over lanes), compares
  every field k < P with mesh_phase(t / u + 1) by select-based masks, and reduces to one predicate (simd_ballot, then
  across SIMD-groups through threadgroup memory and one barrier). Passed: one system-scope acquire fence, then the body,
  its loads of the operand plain loads. Not passed: the threadgroup returns before any store. Every threadgroup writes
  its own flag (a data value, written once per use). The group's offset is a per-repetition binding of the state
  buffer; the expected phase is read through a per-repetition binding into a 16-byte table {0, 1, 3, 2} (an invariant
  value the recorder writes once when bound). Nothing in the kernel depends on the plan but these bindings.
- **Compaction**: one dispatch after a barrier reads the flags and writes, by simd_ballot and simd_prefix_exclusive_sum
  (Blelloch), the dense list of the threadgroups left and the later pass's MTLDispatchThreadgroupsIndirectArguments
  ({left, 1, 1}). Its list and arguments are data values, written once per use.
- **Later pass**: encoded by MetalReplayEncode after `encodeWaitForEvent:value:`, a
  `dispatchThreadgroupsWithIndirectBuffer:` of the same rewritten function with its threadgroup position (and every
  position derived from it) taken from the list; empty when every threadgroup passed. Each threadgroup reads the phases
  again: all landed, the acquire fence and the body; otherwise it adds 1 to the status's `behind` (every unequal field
  one behind) or `ahead` (else), raises `first` to ~t (atomic max), and returns before any store.
- **Its waits**: pass p of the run (repetition t, its j-th such dispatch) waits for every received operand it reads:
  {count: the operand's count word of copy t mod u, target: (t / u + 1) x P, value: base + p + 1}.
- Both lane waits in mesh_read_N go (the remote wait on the step's mask and the local countdown), with mesh_seen, its
  failed value 0xfffffffe, the 2^23 bounds and every UINT64_MAX store. A local input (a producer on the same queue) is
  ordered by the command order and the level barrier, and reads with a plain load.

### Publications (metal_recording.m, owner 9; the native partials owner 11, with metal_native, owner 12)

- A sent range's release: after its producer's last write of execution t, one system-scope release fence and an atomic
  add of 1 to the range's supplied word (D9), through a binding of the state buffer at the word's offset (no literal
  address in the kernel).
- A range written by several threadgroups (the direct publication): a restoring counter counts the range's stores up
  (each group's bytes, atomic add); the arrival that completes the range stores 0 back and then releases, as
  native_projection.metal:232's last arrival restores its count (an mbarrier's phase completion reloading it). The
  counter is a recorder value of `state` 1, placed by the plan and set to 0 once there (D3). It replaces the per-invocation
  statuses seeded by mesh-metal.m.
- The native projection's K-part buffer: its partial sums are a data value (no memset, line 1546 goes: every
  threadgroup writes its own before any are read), its arrival counters a state value; both placed by the plan.
  Owner 12 reports the counters' offset and bytes in struct MetalNativePlan beside partialBytes.

### The program (tools/e2b_replay.m, owners 13 then 4) and its manifest (tools/prepare_e2b.py, owner 14)

- The manifest: each rank stage of each region (regions[l].ranks[r]) gains `"dataflow"`, the routed operands that
  stage writes and reads, in the order its program computes them: a list of {"write": name} and {"read": name}
  (owner 14). Embedding stages and stages with no routed operand have none.
- The program, in order: the LiteRT setup and the operands' sizes as now; mesh_attach; the captures (phases 0, 1, 2)
  against stand-in memory (D4), receive: and publish: inside phase 2's; mesh_plan with every value (each operand: a
  routed one MESH_EXECUTION and wire; every other an invariant of its bytes, a host-prepared or prefill operand its
  stride x instances; each capture's own values, their levels offset per capture in phase order 0, 1, 2), every route
  as declared transfers, and the dependences from every rank's dataflow (each rank's events in stage order chained at
  lag 0, its last to its first at lag 1), in place of `NULL, 0`; each value's placement bound in each capture (place:,
  received:, published:, state:status:) and the replays built; the host inputs written into the placed pages; the
  preparation and the prefill as now; mesh_transfers_start(context, steps); the decode as one MTLCommandBuffer
  (MetalReplayEncode, D7) with an MTLSharedEvent, mesh_metal_waits_start before the commit, a stop handler that only
  calls mesh_metal_waits_stop; mesh_metal_waits_end, then the completion waited for at most MESH_PLATFORM_NS; the
  run's error value from the completion, the waiter and mesh_metal_status.
- Goes: `NULL, 0`; the refusals where steps differ from executions (24ec8cda, cfada099's and 472390a8's comments); the
  zeroing of every local operand's extent (line 627) and of every pooled intermediate (line 840); the MTL4 submission;
  the read of cancel words after the run.

## C3 and C4, item by item

| item | owners |
|---|---|
| C3.1 masked first pass, compaction, indirect later pass, event wait, counts, cancellations as arrivals, client's signal, mesh_read_N's waits gone | 10, 8, 6, 4, 1 |
| C3.2 every value placed by the allocator (pooled intermediates, counters, state section, no records) | 3, 9, 11, 12, 13 |
| C3.3 unrolling by modulo variable expansion, `executions` no input | 2, 3, 5, 13 |
| C3.4 nothing grabs; claims per plan; deferred release | 5, 3, 7, 6 |
| C3.5 the normal program's data edges | 14, 13 |
| C3.6 every race closed, every wait bounded | 3, 5, 6, 8, 10, 4 |
| C3.7 nothing zeroed or reset; phases, rising words, restoring counters | 1, 3, 6, 10, 11, 12, 13 |
| C3.8 what judges it: the builds | coordinator |
| C4 mesh-collective.c `executions` and its items; the memset (749-757); the word zeroing (759-763) | 3 |
| C4 mesh-call.h, mesh-dataflow.c mesh_plan_erased and its fence pair; mesh-flow.c's side | 5, 6 |
| C4 mesh.h `word_off`, `landed_off`; the ORed bits and the UINT64_MAX cancel word; MESH_SUPPLIED | 1, 6, 8 |
| C4 mesh-dataflow.c mesh_claim at attach, its wait, the local run | 5 |
| C4 metal_recording.m both lane waits; mesh_seen's failed value; a lane's UINT64_MAX and the host's read; line 1546 | 10, 11, 4 |
| C4 e2b_replay.m `NULL, 0`; the steps != executions refusals; lines 627 and 840 | 13, 9 |

## Record

(Each sub-assignment, its commits, each review finding and where it went, and what was left open, appended as the
work proceeds.)

### Sub-assignments

1. mesh.h, mesh-map.h: the region, the state section's structs, the hop entry, the publication's fields, removals.
2. mesh-graph.{h,c}: mesh_graph_lags; mesh_graph_happens_before removed once nothing calls it.
3. mesh-collective.{h,c}: the call's new signature, values, the template's program edges of four kinds, least lags
   and u, items, interference, the two colourings; the claim, the state section's placement and initialization, rows,
   publications, the entries of u copies, the writing order, the host functions; the memset and the word zeroing gone.
4. tools/e2b_replay.m, second (after 13): the decode's MTLCommandBuffer, its event and waiter, the stop handler, the
   bounded completion wait, the run's error value from the waiter and the status; the MTL4 submission and the read of
   cancel words gone.
5. mesh-dataflow.{h,c}, mesh-call.{h,c}, mesh.py: attach claims nothing; claims per plan, their close and release;
   sections named by the plan; the start's executions; mesh_plan_erased gone.
6. mesh-flow.c: the template's copies, phases, counts, the supplied threshold, cancellations as arrivals, rising
   executions; the cancel word gone.
7. mesh-link.c, mesh-records.c, mesh-stat.c, mesh-bridge.h: the layout's compare, the release's callers, any read of a
   vacated field.
8. mesh-metal.{h,m}: the transport over the state section, inputs and publications as offsets, the waiter, the status.
9. metal_recording.{h,m}, first: capture, values, place, replay; the pool gone; bindings in place of literals;
   publications by the supplied word and the restoring counter.
10. metal_recording.m, second: the masked first pass, the compaction, the later pass, the status, MetalReplayEncode and
    its waits; both lane waits in mesh_read_N gone (remote and local).
11. metal_recording.m, third: the native partials (data and state values; line 1546's memset gone), and the file read
    whole against the four rules.
12. metal_native.{h,m}, native_projection.metal, tools/native_bench.m: the K-part buffer's counters reported apart.
13. tools/e2b_replay.m, first: capture, plan, bind; values, declared transfers and data edges; the host inputs into
    placed pages; the zeroing and the refusals gone.
14. tools/prepare_e2b.py: each rank stage's dataflow in the manifest.
15, 16, 17. Reviews: written findings, routed to owners.

### Commits and routing

### Reviews and where their findings went

### Open at the end
