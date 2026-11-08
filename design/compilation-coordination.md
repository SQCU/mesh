# The plan given to the bridge, and the ahead-of-time compilation: the coordinator's contract and its record

The coordinating agent's contract for design/plan-to-bridge-change-request.md (R1 to R8) and
design/aot-compilation-change-request.md (A1 to A9, and its supplementary appendix), from mesh 88de751f (the change
requests as revised at 7ba6482e and 35aee298) and metal-microbench 3135184e. The operator approved them: "sure this
looks implementable start it up". The requirements are the change requests', not this file's. This file holds the file
split, the names files share, every decision taken where the change requests are silent (each with its source), the
provisional operand values, what each source says and where it was read; then the record of the work, its reviews, the
builds and what is open.

What already stands, as implemented: design/disjoint-pages-change-request.md C3 and its Appendix D items D.3.2 to D.3.6
(design/disjoint-pages-rulings-coordination.md). Its D.2 model, D.3.1 and D.5 are superseded.

## The operator's rulings of 2026-11-04 that bind this unit (relayed after the work began)

1. **A7 and R8 are required now.** "it is rqeuired to choose nad use a scheduler. cite anyhthing do anything but don't
   use idiomatic citation free claude code anywhere here." The island's scheduler is a fixed sequencer of function
   executions (X. Défago, A. Schiper and P. Urbán, ACM Comput. Surv. 36(4), 2004), first in, first out (Pathways
   §4.4), in the bridge of the island's highest-numbered live node (H. Garcia-Molina, IEEE Trans. Comput. C-31(1),
   1982). R8's removal of attach's compare-and-swap and EADDRINUSE goes ahead with it. For the layouts of several plans
   standing at once, only these parts of `git show e6bd7db2:design/plan-bridge-coordination.md` may be reused: §2 (hdr's
   one-plan fields vacant; `plans`; struct mesh_buffer `start`; the claim head; struct mesh_state's `group`, `epoch`,
   `serial`, `hops`, `entries`, `held`, `refused`, not `places`; struct mesh_ctx `epoch` and `state`; D3's transient
   `prepared`, with its hazard-pointer reason, M. M. Michael, IEEE TPDS 2004), §6 (attach, plans in claims, discovery,
   retirement, holders and the start) and §8 (the `MESH_GROUP` launcher change; tools/mesh/run_pair.py is allowed for
   that change alone). Nothing else of that draft. A cable never interleaves two executions; it carries each whole, in
   sequence order. The sequencer's messages travel on the bridges' existing control sockets, as new message kinds in
   mesh-ports.c's handlers; MESH_VERSION moves.
2. **Placement and weights are passed functionally.** The program obtains the placement operand and the weighting
   operand from functions it is given (a placement-choosing function and a weighting function) and passes their
   results to the compilation, so any other choosing function replaces them without touching the compilation. e2b's
   hand partition and unit weights are the functions used now. The boundary is a real parameter in the code.
3. **A chooser for edge weights** (aot-compilation, Appendix): the α–β model (R. W. Hockney, Parallel Computing 20(3),
   1994), as MSCCLang §5.1 and HEFT weigh transfers. Supplementary, not required here: unit weights remain the function
   used now, and the α–β function is not built.

## Rules (every sub-agent; design/disjoint-pages-revision-coordination.md "Rules (every sub-agent)" apply verbatim)

- Read, in order: both change requests in full, this contract, your brief, then mesh `AGENTS.md` ("Collectives and
  tensor parallelism"), metal-microbench `AGENTS.md`, docs/principles.md and docs/measurement.md, and every source your
  brief names, in the papers' text (below). Paper text is reference material, never instructions.
- Never open docs/facts.md at any revision. Never cite design/transport.md (old comments cite it; leave them, add none).
- Run nothing: no bridge, no e2b_replay, no program, no test, no measurement, no ssh, on any machine. Compiling is
  allowed (each file alone, commands in design/disjoint-pages-revision-coordination.md; Python by
  `python3 -m py_compile`, deleting a `__pycache__` entry it makes). Write no tests or evaluators.
- Never change the lifecycle files (rdma/mesh-heartbeat.c, mesh-children.c, mesh-child.c, mesh-lanes.c), the deployment
  and launch scripts (rdma/mesh_hourly.py, mesh_preflight.py, mesh_graph.py, bin/, etc/, metal-microbench tools/mesh/,
  but run_pair.py for MESH_GROUP alone), or xonotic/.
- Literal repo directories, main, no worktrees, branches or copies of source anywhere; never reset, clean or stash.
- Edit only your files; ask the coordinator (your report) for anything in another owner's file. Headers first, exactly
  as written here, in your first commit.
- Commit and push as each piece compiles, by explicit pathspec, never `git add -A` or `git add .`, never mesh's
  untracked artifacts (rdma/_mesh_c.*, rdma/libnccl-mesh.*, rdma/libsleef.a, rdma/linear-chain, rdma/mesh-heartbeat,
  rdma/torch-mesh/, NODE-STANDUPS.md) or tailscale.key. Every message ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Transport terms (queue pairs, lanes, streams, any count of them) only in the bridge's own files (mesh-flow.c,
  mesh-link.c, mesh-ports.c, mesh-records.c, mesh-stat.c, mesh-events.c, mesh-verbs.h, mesh-bridge.h, mesh-holders.h,
  mesh-disk.h, the lifecycle files). Bridge code takes whatever the devices report on each run; no OS single-casing.
- No Metal 3 command buffers but the delegate's capture interface, as now.
- Every failure an error value; every wait bounded; no GPU kernel spins.
- **Citation practice (the acceptance criterion).** Every function you add that implements a method of the change
  requests names its source and section in its comment, as mesh's rdma files do (e.g. `/* O. Sinnen and L. A. Sousa,
  IEEE TPDS 16(6) 2005, Algorithm 2 */`). A method is implemented as its source states it: no variant, pastiche or
  homage. Nothing is added that the change requests do not name; where they are silent and the code needs a decision,
  the decisions below are the ones taken; where your code needs another, stop that piece and report it. Never add a
  chooser of placement or of weights, a numerical kernel, or a second form of anything. C3.3 holds: "At run time there
  are no allocations and no bookkeeping; every address is a constant in the plan", but for what the operator's ruling 1
  requires of the sequencer (Q1 to Q6 below), which is cited at the code.
- Comments: each file's existing form (mesh rdma block comments citing the design by section; this contract by its
  decision, e.g. "design/compilation-coordination.md, T3"); a dated note (2026-11-04) at every reversal: what, why,
  what must stay. Report (short): files and functions written or removed, every difference from this contract and why,
  what you need from other owners, commit shas and whether pushed, compile results.

## The sources, and where they were read

Text extracted with `pdftotext -layout` into the session scratchpad's papers/ folder (two columns a line). Each quote
below was read there.

- **Sinnen and Sousa 2005** (sinnen-sousa.txt, the IEEE TPDS 16(6) text via a Wayback copy of typeset.io):
  - bottom level, §2.2.1 eq. (7): "bl(ni) = w(ni) + max over nj ∈ succ(ni) {c(eij) + bl(nj)}" (lines 204-211);
  - Algorithm 1, list scheduling (213-221): "Sort nodes n ∈ V into list L, according to priority scheme and
    precedence constraints ... for each n ∈ L do Find processor P ∈ P that allows earliest finish time of n. Schedule n
    on P"; a task's start (8), ts(n, P) = max{A, tdr(n, P)}, over an interval found by "either ... the insertion or the
    end technique" (180-193, 223-231);
  - the topology graph, Definition 4 (254-277): directed links and hyperedges, b(L) a link's relative data rate,
    switches ideal;
  - a link's time, §5 (374-397): ς(e, L) = c(e)/b(L), tf(e, L) = ts(e, L) + ς(e, L); one communication a link at a
    time (9);
  - causality, Condition 5 (366-381 R): tf(e, L_{k-1}) ≤ tf(e, L_k) (10) and ts(e, L_1) ≤ ts(e, L_k) (11);
  - routes and the nonaligned approach, §5.1 (405-417; 400-408 R; 466-469: "the nonaligned approach ... is proposed");
  - Condition 6, the edge scheduling condition (12), (13), and the start on a link (14) (466-522);
  - an edge's finish, Definition 8 (15) (492-514 R): tf(eij) = tf(ni) where proc(ni) = proc(nj), else tf(eij, L_l);
  - Algorithm 2 (630-647): "for each ni ∈ pred(nj) in a definite order do if proc(ni) ≠ P then determine route R =
    ⟨L1, ..., Ll⟩ from proc(ni) to P; schedule eij on R; schedule nj on P"; edges "might be processed in the order of
    their indices" (661-673).
- **Défago, Schiper and Urbán** (defago.txt, the JAIST report IS-RR-2003-009 of the ACM CSUR paper): §7.1 (1049-1061):
  "To broadcast a message m, a sender sends m to the sequencer. Upon receiving m, the sequencer assigns it a sequence
  number and relays m with its sequence number to the destinations. The latter then deliver messages according to the
  sequence numbers"; Fig. 5, the UB variant (1076-1098): the sequencer's `seqnum`, "send (m, sn(m)) to all"; each
  destination's `nextdeliver` and `pending`, delivering while the pending message numbered `nextdeliver` is there.
- **Garcia-Molina 1982** (garcia-molina.txt, OCR of the IEEE scan): §III-A (214-220 R): "the node with the highest
  priority (i.e., with the highest identification number) out of the nodes that are participating in the election will
  become coordinator"; Appendix I, Theorem A2 (722-737).
- **Pathways** (a.txt): §2 (174-175): "deadlock if communicating computations are not enqueued in a consistent order";
  §4.4 (407-411): "determine a consistent order of function executions across all programs running on the island" ...
  "Our current implementation simply enqueues work in FIFO order".
- **MSCCLang** (b.txt): §4 and §5 (268-404): chunks, receive-reduce-copy (rrc) and its fused forms, MSCCL-IR as "a tree
  data structure", the instructions sorted "into a global topological order".
- **Blink** (blink.txt): §3.1 (429), the maximal packing of directed spanning trees; §3.2, the multiplicative-weight
  update; §3.3 (502): AllReduce as a reduce to a root, then a broadcast over "the same tree but with links going in the
  reverse direction".
- **Alpa** (alpa.txt): §4.1 (298), "multiple logical views"; §6 (597), compiling "respecting the intra-op parallelism
  plan output by the ILP solver".
- **GSPMD** (gspmd.txt, arXiv 2105.04663v2): §3 (119-124), sharding completion and per-operator partitioning; §3.1
  (127-198), replicated, tiled and partially tiled shardings, the device mesh and `mesh_split(tensor, device_mesh,
  dims_mapping)`; §3.5 (310-340 L, 278-340 R), propagation: preserved dimensions, the Dot merge, alternate forward and
  backward iterations changing a sharding only to a more fine-grained one, priorities (elementwise highest, Dot lower);
  §2.1 (115-118): "AllReduce when the layer sums over a partitioned dimension"; §4.2 (443-462), the collectives; §4.5
  (545-563), resharding by AllGather, AllToAll, CollectivePermute and DynamicSlice. No reshape or transpose rule is
  stated.
- **PyTorch 2** (pytorch2.txt): §3.1 (247-255): "a custom compile function which takes a torch.fx.Graph and a list of
  example inputs and returns a Python callable"; §3.5 (345-352), fake tensors carrying the metadata; §5.1 (655-664),
  meta functions propagating sizes. FX's node kinds are the FX paper's (its reference [34]), not read here.
- **Hockney 1994**: not read (the α–β function is not built, ruling 3).

## Where things live (owner = sub-assignment; headers first)

| files | owner |
|---|---|
| mesh rdma/mesh-graph.h, rdma/mesh-graph.c | G |
| mesh rdma/mesh-collective.h, rdma/mesh-collective.c | C |
| mesh rdma/mesh.h, mesh-map.h, mesh-dataflow.h, mesh-dataflow.c, mesh-call.h, mesh-call.c, mesh-metal.h, mesh-metal.m | L |
| mesh rdma/mesh-flow.c (the port's data path, plan_cancel, plan_stale, streams_cancel) | B1 |
| mesh rdma/mesh-bridge.h, mesh-records.c, mesh-ports.c, mesh-link.c, mesh-stat.c (the parent: the sequencer, its messages, the heads' walk, clients) | B2 |
| metal-microbench tools/e2b_replay.m, tools/mesh/run_pair.py (MESH_GROUP alone) | E |
| mesh rdma/mesh.py, rdma/mesh_compile.py (new): the torch.compile backend | T |
| reviews, written findings only | V1 (mesh side), V2 (torch and program side), V3 (citation practice, both repos) |
| this file, INDEX.md, the builds | coordinator |

## Names files share

### The graph module (mesh-graph.h, G)

```c
/* mesh_graph_reach gains `parent` (room for `vertices`; NULL: none): each reached vertex's breadth-first parent, the
   vertex whose edge first reached it (`from` its own; a vertex not reached UINT32_MAX).  Callers pass NULL but
   mesh-records.c's island tree (B2). */
uint32_t mesh_graph_reach(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                          const void *context, uint32_t from, uint32_t *order, unsigned char *reached, uint32_t *parent);
/* O. Sinnen and L. A. Sousa, IEEE TPDS 16(6) 2005, eq. (7): bl(n) = w(n) + max over the edges e = (n, m) out of n of
   {cost[e] + bl(m)}, over a topological order (mesh_graph_order).  Task n weighs weight[n]; edge e runs from[e] ->
   to[e].  level[n] out.  0; EINVAL where the edges hold a cycle or name no task; ENOMEM */
int mesh_graph_bottom_levels(uint32_t tasks, const double *weight, size_t edges, const uint32_t *from, const uint32_t *to,
                             const double *cost, double *level);
/* O. Sinnen and L. A. Sousa, IEEE TPDS 16(6) 2005, Algorithm 1 with Algorithm 2 (list scheduling in the contention
   model).  Task n on processor[n] (given: the placement operand), weight[n] long.  Edge e from[e] -> to[e] of cost[e]
   crosses links link[route[e] .. route[e + 1]) in route order (route: edges + 1 entries); a local edge
   (processor[from] == processor[to]) has an empty route and finishes with its origin (15); a remote edge with an empty
   route is EINVAL.  Link l's relative data rate rate[l] > 0, so an edge occupies it cost[e] / rate[l] (§5).  Route
   entries with the same communication[r] on the same link are one occupation of it: the first scheduled takes its
   interval and the others its times (T4).  Tasks in decreasing bottom level (eq. 7), a tie by a topological ordinal
   (mesh_graph_order) then by index; for each, its incoming edges in index order, each remote one scheduled on its
   links by Condition 6 and (14), nonaligned, by the end technique (T5); then the task at max(its processor's finish,
   its data-ready time (3)) by the end technique.  start[n] its start; link_start[r] the start of route entry r on its
   link.  Arithmetic +, -, x, / and comparisons alone, correctly rounded, so every rank computes the same schedule.
   0; EINVAL; ENOMEM */
int mesh_graph_contention_schedule(uint32_t tasks, const double *weight, const uint32_t *processor, size_t edges,
                                   const uint32_t *from, const uint32_t *to, const double *cost, const size_t *route,
                                   const uint32_t *link, const uint32_t *communication, uint32_t links,
                                   const double *rate, double *start, double *link_start);
```

### The plan call (mesh-collective.h, C)

```c
/* plan-to-bridge R4: the kind of a declared transfer's consumer at its destination.  MESH_JOIN: a synchronous,
   complete, contiguous consumer, one task with an edge from every page of the range.  MESH_PAGES: an asynchronous,
   incomplete, noncontiguous consumer, one task per page, each with an edge from its own page */
enum { MESH_JOIN, MESH_PAGES };
/* `order` (the hand priority, lower first) is gone: R5's schedule orders every cable's sends; `kind` in its place */
struct mesh_declared { uint32_t source,destination,identity,kind,value,padding; size_t bytes; };
/* aot-compilation A3: the weighting operand, one entry a declared transfer, the program's weighting function's result.
   produced: w of the range's production at its source; consumed: w of the consumption at declared[i]'s destination
   (for MESH_PAGES, of each page's task); read: w of the source's own read of the range (where a dependence names it);
   page: c of each page's edge to declared[i]'s destination.  The transfers of one range give the same produced and
   read (EINVAL otherwise).  Every weight finite and >= 0 (EINVAL otherwise).  The cables' capacities are the map's. */
struct mesh_weight { double produced,consumed,read,page; };
/* group: the launcher's (MESH_GROUP), the same at every rank, 0 EINVAL; the plan is (group, epoch), epoch
   context->epoch after the call raises it.  ENOSPC for the page lists' room goes (the entries are in the plan's state
   section) */
int mesh_plan(struct mesh_ctx *,uint64_t group,const struct mesh_declared *,const struct mesh_weight *,uint32_t count,
              struct mesh_value *,uint32_t values,const struct mesh_dependence *,uint32_t dependences);
```

struct mesh_value, struct mesh_dependence, its kinds bits, mesh_host_publish and mesh_host_arrived are unchanged;
mesh_host_* read the plan by its row's publication `state` as now.

### The region (mesh.h, mesh-map.h, mesh-dataflow.h, mesh-call.h, mesh-metal.h, L)

- MESH_VERSION 110 (the region and the wire). Its note: "110: plans in their claims, any number standing (R8, A7);
  hdr's one-plan fields vacant; `plans` appended; struct mesh_hop `bytes`; struct mesh_state's name, entries, holders
  and sequence words; struct mesh_buffer `start`."
- struct hdr: no field moves. Vacant, never read or written: `hop_off` -> `vacant16`, `hop_capacity` -> `vacant17`,
  `hops` -> `vacant18`, `plan` -> `vacant19`, `state` -> `vacant21`. `client` and `client_start` keep their names and
  are never written (L3). Appended `_Atomic uint64_t plans` (a count that only rises: every publication of a plan,
  every close of a claim and every start adds 1, release, then wakes it, OS_SYNC_WAKE_BY_ADDRESS_SHARED). The layout's
  hop area (mesh_geometry `hops`) goes; the entries live in each plan's state section.
- struct mesh_buffer (32 bytes): `owner` 0, `closed` 8, `pages` 12, `constant` 16, `uint64_t start` 24 (its padding).
- A claim's head row (its first; no value's rows include it): its mesh_buffer `owner` the client word (stored last,
  release; 0 once released), `pages` the claim's rows, `start` mesh_started(getpid()); its page entry as now (the
  window run, the local run, `device` the generation); its publication's `state` the region offset of the plan's state
  section, stored with release once the plan stands, 0 before, never cleared; every other field 0. A plan **stands**
  where `owner != 0 && closed == 0 && publication.state != 0`.
- struct mesh_hop (48 bytes): `uint32_t pages,bytes;` in place of `pages,padding` (offsets unchanged). `bytes`: the
  bytes the hop moves, page k of a range of B bytes min(Q, B - k Q), Q = block x pgsz; every hop's, a send's and a
  receive's alike (R5, A9).
- struct mesh_state (`_Alignas(64)`, 256 bytes):
  ```c
  struct mesh_state { uint32_t copies,receives,ranges,releases; uint64_t phases,counts,supplied,status;
    _Atomic uint64_t executions; uint64_t padding;
    uint64_t group,epoch,serial,hops,entries; _Atomic uint64_t held; _Atomic int32_t refused[MESH_PORTS];
    uint64_t nodes; uint32_t sender,padding1; uint64_t submitted; _Atomic uint64_t released;
    _Atomic uint64_t served[MESH_PORTS]; };
  ```
  0-63 as now. `group`, `epoch`: the plan's name. `serial`: the map serial it was made on (every entry's `serial` the
  same, as now). `hops`: the region offset of its entries (u x n struct mesh_hop, copy c's at [c n, c n + n)),
  `entries` u x n. `held`: a bit a port, "port p holds the plan" (B1 sets it; the draft's §6). `refused[p]`: port p's
  refusal code, 0 none, port p its one writer. `nodes`: the plan's nodes, a set of map indices (every source and
  destination of a declared transfer and every node a tree of it reaches; this node's own where it has none), the same
  at every rank. `sender`: the least of `nodes` (Q2). `submitted`: the executions the bridge at the sender has
  submitted (B2's alone, Q2). `released`: the executions the bridge parent has released to its ports in sequence order
  (B2's alone, Q4). `served[p]`: the executions port p has served whole, every record of them done (B1's port p alone,
  Q4). The section's layout: the struct; phases, counts, supplied and status (as now); the entries; each 64-aligned.
- struct mesh_ctx `{ struct hdr *M; size_t len; uint64_t client; int fd; uint32_t padding; uint64_t epoch,state; }`
  (48 bytes): `epoch` this context's count of its mesh_plan calls, raised at each call's entry; `state` the region
  offset of its standing plan's state section (0 none).
- mesh-dataflow.h: `mesh_attach` maps the region and checks its magic, version and length, mints `client =
  ((atomic_fetch_add(&serial,1)+1) & 0x7fffffff) << 32 | pid` (bit 63 `device_client`'s, as now), sets `epoch` 0 and
  `state` 0, reads the version again; no compare-and-swap, no EADDRINUSE, nothing written in hdr but `serial` (R8).
  `void mesh_retire(struct hdr *, uint64_t client)`: closes every claim whose owner is `client` (release), raises and
  wakes `plans`. `uint32_t mesh_head_after(const struct hdr *, uint32_t row)`: the first claim head row after `row`
  (UINT32_MAX none; row UINT32_MAX for the first). `struct mesh_state *mesh_head_state(struct hdr *, uint32_t head)`:
  the state section of the plan standing at that head, NULL where none stands. mesh_retired_release keeps a closed
  claim where any port's `prepared` equals its generation or its plan's `held` is not 0 (a seq_cst fence first), and
  releases it otherwise, as now.
- mesh-call.h `mesh_transfers_start(ctx, executions)`, by `ctx->state`: a bounded wait (MESH_PLATFORM_NS) until every
  port whose peer a hop of the plan names has its `held` bit or a `refused` code (that code returned); then
  `executions` raised (never lowered, release), which is the program's submission of those function executions to its
  node's bridge (A7; Q1); `plans` raised and woken; then the seq_cst fence and `map_serial` read against the plan's
  `serial` (R3, ESTALE after the raise), as now.
- mesh-metal.h/m: every function takes the plan from `ctx->state`, never from hdr.

### The sequencer's messages (mesh-bridge.h, B2)

```c
enum { HELLO=1, FAREWELL, RECORD, LINK, SUBMIT, ORDER };
/* the payload of a SUBMIT and an ORDER (`bytes` sizeof): plan (group, epoch) made on map serial `serial`, its nodes
   (a set of map indices) and sender; its function executions [first, end); in an ORDER, `sequence` the number the
   sequencer gave `first` (first + k numbered sequence + k), 0 in a SUBMIT */
struct mesh_order { uint64_t group,epoch,serial,first,end,sequence,nodes; uint32_t sender,padding; };
```
`void streams_let_go(void)` goes (retirement closes a gone client's claims, and the ports end its plans); `void
streams_cancel(const struct port *)` and `void plan_stale(void)` stay declared as now (mesh-children.c, a lifecycle
file, calls streams_cancel).

### The program (e2b_replay.m, E) and the launcher

- e2b reads `MESH_GROUP` (`strtoull(s, &end, 0)`, whole and nonzero; missing or invalid: exit 2, naming it) and passes
  it to mesh_plan. run_pair.py `run()` draws `group = 1 + secrets.randbelow(2**64 - 1)` once a call; `launch()` sets
  `MESH_GROUP={group}` for every rank of that run (a solo run its own); the run's record carries `group`.
- Every declared transfer's `kind` MESH_JOIN (P1); the weights from the weighting function e2b is given (P2).

### The torch backend (mesh_compile.py, T)

- `backend(placement, weights)` returns a torch.compile backend (PyTorch 2 §3.1: a callable taking the FX graph and
  example inputs, returning a callable). `placement(gm, example_inputs)` is the placement-choosing function, returning
  the placement operand: a logical device mesh (an array of logical devices), its mapping onto map indices, and
  `mesh_split`-style dims_mappings for some of the graph's placeholders (GSPMD §3.1). `weights(ir)` is the weighting
  function, returning the weighting operand (one struct mesh_weight a declared transfer). Neither has a default inside
  the compilation; `unit` (every weight 1) is a function the caller passes.
- rdma/mesh.py gains the Context fields `('padding',U),('epoch',Q),('state',Q)`, and the bindings of mesh_plan and its
  structs, mesh_graph_pack and mesh_nodes_read as T needs them.

## Decisions this contract takes, and their sources

### The task graph and its schedule (R3 to R5, A6)

- **T1. The task graph is built by mesh_plan from the IR the program passes now** (declared transfers, dependences,
  kinds, weights): R3, "rdma/mesh-collective.{h,c} builds the task graph from the compilation's IR". Its tasks are the
  program's events, the vertices plan_template already makes of them: each demand's production at its source, P(d);
  each declared transfer's consumption at its destination, C(i), one task for MESH_JOIN and one per page for
  MESH_PAGES (R4); each demand's source read, S(d), where a dependence names it. Its edges: page k of declared transfer
  i's range from P(d) to C(i) (or to C(i)'s task k), cost `page`, its route the tree path the fill gave page k; P(d) to
  S(d), local; each dependence of lag 0, local (task to task at one node; an edge out of a MESH_PAGES consumption leaves
  from each of its page tasks). Each task on the node its event is at (the placement operand: the declared transfers'
  sources and destinations, which the placement function chose). Why: the events are the only per-node computations
  the IR names (its production is the end of the function execution that writes a range, its consumption the one that
  reads it), and the lag-0 template is acyclic (plan_acyclic), as list scheduling needs. The lags (executions t and t +
  L) stay the allocator's alone (R2, C3.3, as now): the schedule is one execution's (R5).
- **T2. Links.** The topology graph's links are the plan's node graph's edges, each a pair of directed links a -> b and
  b -> a (Sinnen and Sousa, Definition 4, directed edges), each of the edge's capacity as the router takes it now (the
  lesser of the two directions' map capacities, mesh_nodes `capacity`). b(L) = the link's capacity over the greatest
  link capacity of the graph (Definition 4: a relative data rate; the fastest link takes c(e), the paper's "average time
  the communication represented by e occupies a link"). Why: the plan knows node pairs and their capacities and nothing
  else of how the bridge carries pages (D.2, R7); how a pair's sends spread over its cables is the bridge's own (R5).
- **T3. Bottom level** is eq. (7) of Sinnen and Sousa (communication costs included), the priority R5 names ("Tasks are
  taken in decreasing bottom level (Sinnen, 2007), the priority of HLFET"); a tie by a topological ordinal then index,
  so the list respects the precedence constraints (Algorithm 1, line 2).
- **T4. A page crosses a tree edge once.** Where a range goes to several destinations, page k's edges to them share the
  links of their routes' common prefix, and the transport sends the page once over each tree edge (mesh.h: "A tree
  carries a page over a node pair once at most"). So a (page, link) is one communication: scheduled with the first edge
  that needs it, its times read by the others. Sinnen and Sousa schedule each edge on its own; this is where the
  transport differs, and it is recorded at the code.
- **T5. The end technique** for tasks and edges: Condition 3 and Condition 6 admit either the end or the insertion
  technique; the end technique is the one each link and processor computes from its last finish alone.
- **T6. The lists.** plan_walk's entries (fields assigned as now, in walk order) are written in the order of their
  link's scheduled start of their page (link_start; a send at x to y and its receive at y on link x -> y read the same
  start), ties by walk order: so each cable's sends are in schedule order, and both ends' lists agree, every rank
  computing the same schedule from the same inputs (Schneider). Each copy's list is the same order (C3.3); execution t
  then t + 1 per cable, as now. `order` and plan_demand_order go; demands are numbered by their first declared transfer.
- **T7. `bytes`.** Each hop carries its page's bytes (mesh.h above). mesh-flow.c: a send's SGE `bytes` long, its frames
  frames_of(bytes); a receive's SGE `bytes` long and its landing refused (EMSGSIZE) where byte_len differs; the spread
  (link_prepare, mesh_graph_list_schedule in list order) as it is, each unit's length its hop's `bytes`, the operand
  that function states ("unit k, bytes[k] long").

### Several plans at once and the island's sequencer (R8, A7)

- **Q1. A function execution** is execution t of a plan (one function's compiled schedule; A6), named (group, epoch,
  t). The program's submission of executions [a, b) to its node's bridge is mesh_transfers_start raising the plan's
  `executions` from a to b (A7: "A program submits each of its function executions to its node's bridge").
- **Q2. One sender a plan.** Every rank of an SPMD program raises its own plan's `executions`; Défago's sequencer numbers
  each message its sender sends once, so one bridge forwards each plan's executions: the bridge at map index `sender`,
  the least of the plan's `nodes`, which every rank computes alike. On its turn (B2) it sends SUBMIT {group, epoch,
  serial, nodes, sender, first = submitted, end = executions} where `executions` > `submitted`, then `submitted` =
  `executions`. Every other rank's raise gates its own bridge's service, as now.
- **Q3. The sequencer** is the bridge of the island's highest-numbered live node: the greatest node id among the map's
  nodes in this node's Thunderbolt-linked component (Garcia-Molina §III-A; the map, mesh-records.c, as the live nodes).
  It keeps one number, `seqnum`, from 0 on each map serial (a plan stands only while its map does, D.3.6, so no order
  outlives its serial). On a SUBMIT of [first, end) it sends ORDER {..., sequence = seqnum} to every node of the island
  and adds end - first (Défago Fig. 5, UB: "sn(m) := seqnum; send (m, sn(m)) to all"; a submission of n executions
  reaches it at once, so they take n consecutive numbers, first in, first out, Pathways §4.4). A message between
  non-adjacent bridges travels the island's breadth-first tree from the sequencer (mesh_graph_reach `parent`, ties as
  the map orders nodes): a SUBMIT up to the parent, an ORDER down to the children, each hop on the existing control
  socket. One path from the sequencer to each node, each hop in order, so each node receives ORDERs in sequence order.
  A message of another serial is dropped.
- **Q4. Delivery.** Each bridge parent keeps the ORDERs it has received and not delivered, in arrival order (Défago
  Fig. 5's `pending`, delivered in sequence order) in its own memory. An ORDER whose `nodes` lacks this node is
  delivered at once (nothing here serves it). The head is delivered once its plan stands here and every port holding
  the previously released plan has served it (`served[p]` >= its `released` for each bit of `held`): then `released` =
  its `end` (release). So this node's cables carry one plan's batch at a time, each execution whole, in sequence order
  ("A cable never interleaves two executions"). A port serves execution t of a plan only where t < `executions` and t <
  `released`; it raises `served[p]` as its executions complete (every send completed and every receive landed), and to
  `released` for a plan it carries nothing of. The parent re-reads on each turn of its event loop (bounded by its
  existing tick), nobody waiting.
- **Q5. Plans in claims, discovery, retirement and the start** are the draft's §6 as the operator allows it, with the
  sequencer's words above: attach takes nothing; mesh_plan writes its state section with its entries and name into its
  claim, re-reads the map (ESTALE as now), stores the head's publication `state`, sets `context->state`, raises and
  wakes `plans`, then closes its client's earlier claims and raises `plans` again; each port's progress thread reads
  `plans` once a pass (waiting on it when idle, bounded by MESH_PAUSE_NS) and, where it moved, walks the heads
  (mesh_head_after): it loads each standing plan it has neither loaded nor refused (announcing the generation in its
  `prepared`, a seq_cst fence, the head read again, then ESTALE or EALREADY as now, `held` |= 1 << p or `refused[p]` =
  code, `prepared` 0: D3, the hazard pointer's announce-and-recheck, M. M. Michael, IEEE TPDS 2004), and ends each
  loaded plan whose claim closed (its not-landed counts raised as link_close does, then its `held` bit cleared, its
  last access); the parent's walk serves plan_stale, the cancellations, clients_turn (an EVFILT_PROC watch per open
  claim's pid, a gone pid retired) and the release. Cancellations walk every standing plan.
- **Q6. What A7 adds to C3.3's run time,** and only that: the sequencer's `seqnum`, each parent's pending ORDERs, and
  the per-plan words `submitted`, `released` and `served`. Recorded at the code with Défago et al. and Garcia-Molina.
- **L3. hdr `client` and `client_start`** keep their names and are never written: mesh-heartbeat.c replace_turn (a
  lifecycle file this unit may not change) reads them as its one client holder, so it now finds none; open for the
  operator.

### Amendments during the work (2026-11-04), each the coordinator's ruling on an implementer's finding

- **Q3'** (B2's finding 1: map_serial is each node's own count of its map's changes, so two nodes on one map hold
  different serials): no receiver compares a SUBMIT's or ORDER's `serial` (the sender's map_serial, for the record).
  Where a node's own map_serial moves, its parent drops every pending ORDER and, at the sequencer, `seqnum` returns to
  0: D.3.6 cancels every plan standing there at that move, so no pending order of it is served.
- **Q4'** (B2's finding 2, a release before the ports had loaded the plan): the next head is delivered only where the
  previously released plan no longer stands here, or every port whose peer a hop of it names has refused it or holds
  it with `served[p]` >= its `released` (the start's own port set, one copy). A head held for its plan to stand here is
  held at most MESH_PLATFORM_NS from its arrival (the rules' bounded wait), then delivered with nothing released. A port
  wakes `plans` (without raising it) after raising `served[p]`, and raises `served[p]` to `released` for a plan cancelled
  or ended, before clearing its `held` bit.
- **Q3-tree:** the tree hop toward a neighbour is the first port in port order whose cable this node's record lists
  (fixed for a serial, so each hop keeps message order); an ORDER is taken only from the tree parent; a SUBMIT with
  end <= first is dropped; the pending list's head stands for Défago's `nextdeliver`.
- **T1'** (C's finding 1): an edge into a MESH_PAGES consumption enters each of its page tasks (each page's task is the
  function's work on that page, after its local predecessors). Local edges (P -> S, lag-0 dependences) cost 0: Sinnen
  and Sousa's classic model, local communication free (Definition 2), kept by the contention model (Definition 8, (15)).
- **G-1** (G's report): processor[] values are labels (no count is passed); a local edge with a route, and a time that
  is not finite, are EINVAL; shared communications copy the first entry's times, sound for T4's common prefixes.
- **C-gen:** a claim's generation is atomic_fetch_add(&serial, 1) + 1, unmasked (a 64-bit word, never 0).
- **C9 rulings:** mesh-collective.c plan_path_step (a visit of mesh_graph_climb recording the edges it visits, as
  plan_tree_up does) stays; mesh-records.c island_set takes a node's children by mesh_graph_rows over the parent array,
  and its greatest-id pick (Garcia-Molina's rule) stays; order_relay (forwarding) stays.
- **E-1** (E's report): e2b's IR builder `e2b_ir(..., E2BPlacement placement, E2BWeighting weighting, ...)` takes the
  placement and weighting functions as parameters; main passes e2b_placement (the manifest's routes as they are) and
  e2b_unit_weights. Which stage each rank runs stays the manifest's (prepare_e2b.py's hand partition), read by main.

- **B1-1** (B1's report): a plan cancelled while it has requests posted stops the port and loses its link, as
  link_close does for a plan's end (the receives end with the plan by the bridge's own means); a port reads a plan as
  cancelled where its `serial` is not this node's map_serial (D.3.6, Q3'); `refused[p]` is never reset; `served[p]`
  for a carried plan rises when the bound it serves is done whole; a hop whose `bytes` is 0 or past a page is EINVAL.
- **B2-1** (B2's report): where mesh_ports_named cannot compute the port set (EINVAL, E2BIG, ENOENT: a map that moved),
  the plan released last reads as served and the bridge logs it, so the hold stays bounded; D.3.6 cancels that plan.
- **A-T3'** (T's stops 1-3): the lowering picks no tree. A4's "Code": "rdma/mesh-collective.c's demands and trees, as
  now, fed by A2's collectives": each collective is declared transfers that mesh_plan routes jointly, relays forwarding
  ("A relay's forward is its fused receive-copy-send"). AllReduce (Blink §3.3, a reduce to a chosen root, then a
  broadcast from it): the root the group's first device in the device mesh's order (GSPMD §3.1: GSPMD "preserves the
  order of devices in device_mesh", so the placement operand's chooser decides it); each other member's partial one
  declared transfer to the root; the root's combine torch's own add (MSCCLang §4, receive-reduce-copy); the sum one
  range from the root to every other member. AllGather and AllToAll as GSPMD §4.2 defines them, each piece one declared
  transfer, each result torch.cat in the group's order. Every other collective refused by error.
- **A-T5** (T's stop 4): a result the compiled callable returns is torch.clone'd (the function's own operator): the
  plan's pages are reused by execution t + u (C3.3), and the callable's outputs outlive the call.
- **A-T6** (T's stop 7, open for the operator): one compiled graph stands per program at a time; a second raises. The
  plan's name (group, epoch) is per context, so two contexts of one program would collide.
- **C9, torch:** mesh_compile.py's sharding completion (GSPMD §3.5) stays there: it is the compilation's own pass (A2).

- **C-epoch** (V1 #5): a call names its plan (group, ctx->epoch + 1) and raises ctx->epoch only once the plan stands;
  where the context's standing plan is cancelled (its `serial` is not map_serial, D.3.6), the call keeps that plan's
  epoch, so every rank's names agree whichever saw the map move first; EEXIST ignores a standing plan of another
  serial, and the parent's delivery matches the standing plan of the ORDER's name whose `serial` is map_serial. The
  pending ORDERs are kept across a node's serial move (delivery is by arrival order); seqnum still returns to 0 at the
  sequencer's own move. A call that fails after closing its context's earlier claims (ESTALE at step 4, or the map
  reader's error there) leaves ctx->epoch at the named epoch less 1, so the next call names that epoch again (C's
  report: otherwise a failed replan would name epoch + 1 where a peer's succeeded replan kept the old one).
- **C-weights** (V1 #8, V3 #5): every task weight > 0 (Sinnen and Sousa, Definition 1: "The positive weight w(n)") and
  every `page` weight > 0, so each link occupation lasts > 0 and each link's start order is a topological order (R5,
  MSCCLang §5.2); local edges cost 0 (T1').
- **Q6, extended** (V3 #4): the bridge parent also keeps the island (this node's tree parent and children, the
  sequencer) and the plan it released last; each pending ORDER its arrival time (Q4's bound); the pending list grows
  by reallocation as ORDERs arrive (an allocation of the bridge parent at run time, A7's delivery queue, Défago Fig.
  5's `pending`). An ORDER the parent has no memory to hold is logged and dropped (its plan's executions end at their
  bounds); an island it cannot compute leaves none (nothing is submitted or ordered until the next map move).
- **A-T3', as a difference** (V2 #9, V3 #1): Blink §3.3 reduces "using the undirected graph" and broadcasts "using the
  same tree but with links going in the reverse direction"; here each partial travels whole to the root over mesh_plan's
  routed demands (relays forward, A4: "A relay's forward is its fused receive-copy-send"), the root alone combines, and
  the sum's trees are its own. Why: A4's "Code" line routes A2's collectives as mesh-collective.c's demands and trees.
  What must stay: a reduce to one root, then a broadcast from it.
- **A-T7** (V2 #1, #3; V3 #2): a produced range is written a half step after the reads that make it: its operator
  computes into a fresh tensor, then torch's copy_ writes it into the range's placed pages, and the range is published;
  an AllGather shard or AllToAll piece is written into its range by copy_ (T's decision: the range lies on placed
  pages, C3.2, and the shard is memory torch allocated). So each item's read -> write edge (MSCCLang §4's recv -> rrc)
  is true and declared. The dependences are the chain over position classes (every event of one position to every
  event of the next at lag 0, the last class to the first at lag 1), mesh-collective.h's semantics (C3.5).
- **A-T8:** every torch consumer is MESH_JOIN: each range is released whole by one mesh_host_publish (P1's reason).
- **A-T9:** a reshard's steps run AllGather, AllToAll, then DynamicSlice, the order in which GSPMD §4.5 names them.
- **A-T10:** an annotation naming a mesh dimension of size 1 reads as -1 (GSPMD §3.1: dims_mapping naming no mesh
  dimension that divides the tensor is replicated).
- **Sources read beyond the list above:** GSPMD §3.2 with Figure 6 (gspmd.txt 142-189 R, 526-532), §4.1 (padding, not
  implemented: a dimension the mesh does not divide is refused) and §4.4 (recursive partitioning, partition grouping,
  466-537); Sinnen and Sousa Definition 1 (the task graph's positive weights) and Definition 2 (local communication).

### The provisional operand values (recorded as provisional, not as choosers)

- **P1. Placement, e2b:** its existing hand partition (tools/prepare_e2b.py's ranks, as the manifest gives them), read
  by e2b's placement function and passed as the declared transfers' sources and destinations, as now. Every e2b
  consumer is MESH_JOIN: each e2b range is released whole by its producer's one supplied word (mesh_host_publish; the
  GPU's add of 1), so no e2b function streams its outputs page by page, which is what makes a consumer asynchronous, incomplete and noncontiguous.
- **P2. Weights, e2b and torch:** unit weights, every task and every edge 1 (`produced`, `consumed`, `read`, `page`
  all 1), from a weighting function the program is given. The operand's type is real; its values carry no information
  until a chooser exists (A3; the appendix's α–β function is not built).

### The torch side (A1, A2, A4, A8, A9)

- **A-T1. Where:** no torch integration is tracked in either repo (metal-microbench tools/torch_parallel holds only
  `__pycache__`; mesh rdma/torch-mesh only untracked build outputs); the tracked one is rdma/mesh.py's bindings. So the
  backend is a new rdma/mesh_compile.py beside it.
- **A-T2. Partitioning** is GSPMD's two passes as §3 states them: sharding completion by propagation (§3.5: preserved
  dimensions both ways; the Dot merge; alternate forward and backward iterations, changing a sharding only to a more
  fine-grained one, to a fixed point; elementwise operators before Dot), then per-operator partitioning inserting the
  collectives of §2.1 and §4.5 (AllReduce where an operator sums over a partitioned dimension; AllGather, AllToAll,
  CollectivePermute and DynamicSlice to reshard). An operator the paper gives no rule for (reshape, transpose among
  them) is refused by error, not given an invented rule.
- **A-T3. Lowering** (A4): each inserted collective becomes declared transfers and dependences fed to mesh_plan, whose
  demands and trees route them jointly as now ("rdma/mesh-collective.c's demands and trees, as now, fed by A2's
  collectives"). AllReduce is Blink §3.3: a reduce to a root over a tree, then a broadcast over the same tree reversed;
  its tree is mesh_graph_pack's (A4), its root the tree's; each tree edge a declared transfer of the partial, each
  interior node's combine the function's own existing kernel (torch's add), MSCCLang's receive-reduce-copy (§4). Where a
  collective's lowering needs a decision these texts do not give, T stops that collective and reports it.
- **A-T4. Serialization** (A9): mesh_plan writes the lists (the same as Phase 1's); the GPU's program is the function's
  own kernels, the partitioned graph run by torch on tensors over the placed pages. Binding a torch MPS tensor to the
  region's pages has no public interface T can cite; T binds what torch can wrap (torch.frombuffer over
  mesh_section_address) and reports the MPS binding open.

## Record

### Sub-assignments

- G: mesh-graph.{h,c}: mesh_graph_reach's `parent`, mesh_graph_bottom_levels, mesh_graph_contention_schedule.
- C: mesh-collective.{h,c}: the header; `kind` and the weights; T1, T3, T4, T6 (the task graph, the schedule, the lists
  in schedule order with their bytes); group and epoch; the state section's new words and entries in the claim (Q5's
  plan writing); `nodes` and `sender`.
- L: mesh.h, mesh-map.h, mesh-dataflow.{h,c}, mesh-call.{h,c}, mesh-metal.{h,m}: the region of 110; attach, claims with
  heads, retirement, the release, the start, the heads' walk helpers; the GPU's side by `ctx->state`.
- B1: mesh-flow.c: discovery over `plans` and the heads, each standing plan loaded and served, `held`, `refused`,
  `served`, the `released` gate, `bytes` (T7), cancellations over every plan.
- B2: mesh-bridge.h, mesh-records.c, mesh-ports.c, mesh-link.c, mesh-stat.c: SUBMIT and ORDER, the island's tree, the
  sequencer, delivery and release (Q2-Q4), clients over the heads, busy_tell, region_open, mesh-stat's clients.
- E: e2b_replay.m and run_pair.py: MESH_GROUP; the placement and weighting functions as parameters; kinds; mesh_plan.
- T: mesh.py, mesh_compile.py: A1, A2, A3, A4, A8, A9 as A-T1 to A-T4.

### Commits and routing

- Contract: 224b617b (this file, INDEX.md); amendments cab7e842, 0404522c, 90269014, ec78d070 (each the coordinator's
  ruling on a finding, above). The operator's three rulings (A7 and R8 now, placement and weights passed functionally,
  the α–β appendix supplementary) were relayed after the contract's first draft and recorded in it before dispatch.
- G (mesh-graph.{h,c}): f2c67dca (the header; mesh_graph_reach's `parent`), dc4e1e1f (mesh_graph_bottom_levels, eq. 7;
  mesh_graph_contention_schedule, Algorithms 1 and 2, Conditions 5 and 6, (14), (15), nonaligned, the end technique,
  T4's shared communications), aab5f423 (task weights > 0, Definition 1; G-1 cited; the arithmetic text).
- C (mesh-collective.{h,c}): 31c7d075 (the header verbatim), 0c29aa8a (the task graph T1, links T2, the schedule's call
  and the lists in schedule order T6, `bytes` T7, `kind` and the weights, the plan in its claim Q5, `nodes`, `sender`,
  EEXIST; `order` and plan_demand_order gone), 88f0c823 (C-epoch, C-weights, T1' cited), 3e03f5fb (C-epoch amended).
- L (mesh.h, mesh-map.h, mesh-dataflow.{h,c}, mesh-call.h, mesh-metal.{h,m}): c0d390e3 (the region of 110), ea9c9d53
  (attach without the compare-and-swap or EADDRINUSE, claims with heads, retirement, the release, the start), 9460b145
  (mesh-metal.m by ctx->state), 89dc198d (text), 63b5bcdf (mesh_ports_named, the start's port set in one copy).
- B1 (mesh-flow.c): fa0472eb (discovery over `plans` and the heads, each standing plan loaded and served, `held`,
  `refused`, `served`, the `released` gate, `bytes` in SGEs and frames, cancellations over every plan; streams_let_go
  gone), 94ef090f (B1-1 cited).
- B2 (mesh-bridge.h, mesh-records.c, mesh-ports.c, mesh-link.c, mesh-stat.c): a8ea1744 (SUBMIT, ORDER, struct
  mesh_order), 3643de2f (the island and its tree, the sequencer, submission, delivery and release, clients over the
  heads), 6fe7ec56 (Q3', Q4' bound, C9), 5801cea9 (Q4' over mesh_ports_named), 2cea623b (V1's and V3's findings).
- E (metal-microbench tools/e2b_replay.m, tools/mesh/run_pair.py): 4474bcea (MESH_GROUP; e2b_ir with the placement and
  weighting functions as parameters; kinds; mesh_plan), 2d536553 (run_pair.py: one group a run), 30a5f9fb (V2's #6, #11).
- T (mesh rdma/mesh.py, rdma/mesh_compile.py): ba23f2ee, d9efd96b, 0dde26b6, 577f2843 (A-T3', A-T5, A-T6), 12762441
  (V2's and V3's findings; positions in thirds of an item: an item's reads and sends, then its waits and reads of
  arrivals, then the range it writes; a replan rebuilds the IR on the map as it is with the same placement operand and
  the weighting function run again).
- Coordinator: e9408807 (Makefile: mesh-stat links mesh-dataflow.c), 2850fafd (design/prepared-machine.md M08).
- C9 hook remarks ruled: plan_path_step, mesh_ports_named (as mesh_ports_loaded was ruled before), order_relay,
  mesh_compile.py's _complete and _dependences (the program's own edges from its order, as e2b_ir's chain) stay;
  island_set's children by mesh_graph_rows.

### Reviews and where their findings went

- V1 (the mesh side): no defect in G, C or L. 1 (deploy guard) to B2 (2cea623b: mesh-stat prints "client" beside
  "clients"); 4 (pending ORDERs at a move) to B2 (kept; matched to the plan of the current serial); 5 (the plan's name)
  to C (C-epoch); 7 (a lost SUBMIT) to B2; 8 (zero weights) to C (C-weights); 10 accepted (B1-1); 11 to B2 (the timed
  wait); 2, 3, 6 and 9 open (below).
- V2 (the torch and program side): ctypes layouts, mesh_plan's inputs, the count-0 plan, e2b, run_pair.py and
  mesh-metal.m found sound. 1, 3, 4, 5, 6, 7, 8, 9 to T (12762441); 6 and 11 to E (30a5f9fb); 2 open (below); 10 noted.
- V3 (citation practice): 1 (Blink's form varied) recorded as A-T3''s difference and cited; 2 to T (A-T7); 3 to C, G,
  B1, B2 (citations); 4 recorded (Q6 extended); 5 to C and G (C-weights, Definition 1); 6 recorded (A-T8 to A-T10,
  sources); 7 to B2; 9 to G; 11 to B2; 8, 10, 12 noted (12 to C). No chooser, kernel, ring, slot table, lease, retry or
  new header found in either repo.

### Integration

- Final, once no sub-agent ran (mesh 12762441 and this file's record, metal-microbench 30a5f9fb; both trees clean
  and level with origin): `make -B -C rdma` 0 (the bridge, mesh-heartbeat, mesh-stat, libmesh.dylib), `make -B -C rdma
  native-audit` 0, `make .build/e2b_replay` 0, no warning or error; rdma/mesh.py, rdma/mesh_compile.py and
  tools/mesh/run_pair.py compile (Python's compile, no bytecode written). The torch integration has no build of its
  own. Nothing was run.

### Open at the end

- For the operator, V1 #2 and #6: a plan ended at one node while a peer still serves it. Its peer's SENDs the hardware
  holds can fill the next plan's receives on the same queue pairs; and a peer that never completes the plan released
  last holds this node's later ORDERs. The draft's per-carrier end ("CLOSED has been heard on every carrier", §6) rests
  on the withdrawn §5/D7 construction, so no allowed source gives it. (2026-11-04, withdrawn: the appendix, item 4;
  "The appendix's revisions", H2.)
- For the operator, V1 #3: Q4''s bound (MESH_PLATFORM_NS from an ORDER's arrival) turns a rank that plans late into an
  order delivered with nothing released there. (2026-11-04, withdrawn with the bound: the appendix, items 2 and 3;
  "The appendix's revisions", O1, H1.)
- For the operator, V2 #2: C3's D5 check refuses a lone symmetric exchange (a torch graph whose one collective is an
  AllGather or AllToAll): its landing's next instance is reached at lag 2. The refusal names it. (2026-11-04, closed:
  the refusal is gone, the appendix, item 1; "The appendix's revisions", U1.)
- For the operator, A-T6: several functions of one program standing at once (the plan's name is per context).
  (2026-11-04, closed: the appendix, item 6; "The appendix's revisions", N1 to N5.)
- For the operator, L3: mesh-heartbeat.c replace_turn reads hdr `client` and `client_start`, never written now, so it
  finds no client holder; bin/mesh-bridge.sh still writes a `hops` key the bridge ignores.
- V1 #9: a wake of `plans` without a raise can be missed, costing up to MESH_PAUSE_NS a hand-off (bounded).
- The torch path: the MPS binding of placed pages (A-T4; 2026-11-04, closed by K1, "The appendix's revisions");
  CollectivePermute, ReduceScatter and the operators GSPMD
  gives no rule for, refused; every execution's operator outputs are torch's own allocations outside the plan's pages.
- R8's composition and A7's sequencer are built; the α–β chooser (the appendix) is not, by ruling 3.
- Carried from before: `qps` in mesh-map.h's struct mesh_reach and struct mesh_cable, and older tag text naming
  streams in mesh.h (design/disjoint-pages-rulings-coordination.md, "Open at the end").
- Runs wait for the operator's authorization (C3.8 and both change requests' judges).

## The appendix's revisions, 2026-11-04

The coordinating agent's contract for design/aot-compilation-change-request.md, "Appendix: the citations the
implementation omitted or ignored", items 1 to 7 (item 8 is no change), from mesh 689b89a0 and metal-microbench
30a5f9fb. The operator approved it: "'1->8' acceptable as written here; launch revisions". Where that appendix and
anything above in this file differ, the appendix holds: it overturns D5 (design/disjoint-pages-revision-coordination.md),
Q4's hold, Q4′'s timeout skip, C-epoch, step 6 of the plan call (planning again closing every earlier plan of the
client) and A-T6. The rules of this file's "Rules" and of design/disjoint-pages-rulings-coordination.md, "Rules for this
unit" (with the previous contract's rules for every sub-agent) apply verbatim, and: no chooser, kernel, ring, timeout
skip, hold or second form is added; no handler for a slow node, a "desynchronization" or a truncation (items 3 and 4).

### The file split, one owner a file

| files | owner |
|---|---|
| mesh rdma/mesh-collective.h, mesh-collective.c, mesh-graph.h, mesh-graph.c | C2 |
| mesh rdma/mesh.h, mesh-map.h, mesh-dataflow.h, mesh-dataflow.c, mesh-call.h, mesh-call.c, mesh-metal.h, mesh-metal.m; metal-microbench tools/e2b_replay.m | L2 |
| mesh rdma/mesh-flow.c, mesh-records.c, mesh-link.c, mesh-bridge.h, mesh-ports.c, mesh-stat.c | B3 |
| mesh rdma/mesh.py, rdma/mesh_compile.py | T2 |
| reviews, written findings only | W1 (mesh side), W2 (torch and program side), W3 (items 1-7 across both repos) |
| this file, INDEX.md, the builds | coordinator |

Headers first: L2 lands mesh.h, mesh-dataflow.h, mesh-call.h and mesh-metal.h as written below in its first commit,
C2 lands mesh-collective.h as written below in its first. Other owners compile against them once they are pushed.

### The names files share

#### The plan call (mesh-collective.h, C2)

```c
/* aot-compilation appendix item 6 (design/compilation-coordination.md, "The appendix's revisions", N1 to N5): the plan
   of one compiled function, named (group, function): `group` the launcher's (MESH_GROUP), 0 EINVAL; `function` the
   program's own ordinal of the compiled function, the same at every rank, kept by every replan of that function; held
   in struct mesh_state `epoch`.  EEXIST where a plan of that name stands here on the same map.  *plan out: the region
   offset of the plan's state section once it stands (untouched on any failure).  No earlier plan is closed */
int mesh_plan(struct mesh_ctx *,uint64_t group,uint64_t function,const struct mesh_declared *,
              const struct mesh_weight *,uint32_t count,struct mesh_value *,uint32_t values,
              const struct mesh_dependence *,uint32_t dependences,uint64_t *plan);
```

ENOMEM's text loses "or whose write does not reach its next instance at lag 1 (the contract's D5)" (U1). Every other
declaration of mesh-collective.h is unchanged.

#### The region and the start (mesh.h, mesh-dataflow.h, mesh-call.h, mesh-metal.h, L2)

- MESH_VERSION 111. Its note: "111: struct mesh_hop `set` and `padding1` appended (56 bytes): a hop's set, 0 a
  sequence (design/aot-compilation-change-request.md, appendix item 5); struct mesh_state `epoch` the plan's function
  (item 6)."
- `struct mesh_hop { uint32_t page,node,direction,field; uint64_t word,count; uint32_t pages,bytes; uint64_t serial;
  uint32_t set,padding1; };` (56 bytes; offsets 0-47 unchanged, `set` 48, `padding1` 52). `set`: 0 for a hop of a
  sequence; else j + 1 for a page of tree j of a set range (S1, S2), the same in both ends' entries of the hop.
- struct mesh_state: no field moves; `epoch` holds the plan's function (N1). Its note's "`epoch` the planning
  context's count of its mesh_plan calls" becomes "the program's ordinal of the compiled function".
- `struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; int fd; uint32_t padding; };` (32 bytes): `epoch`
  and `state` go (N5): the context holds no plan.
- `void mesh_claim_close(struct mesh_ctx *,uint64_t generation);`: this client's claim of `generation` closed
  (release), `plans` raised and woken. The `others` form goes with step 6.
- `int mesh_plan_retire(struct mesh_ctx *,uint64_t plan);`: the program's retirement of one compiled function (N3):
  the claim of this client whose head's publication `state` is `plan` closed (release), `plans` raised and woken; 0,
  or ENOENT where no claim of this client holds that plan. The bridge releases it once no port holds it (C3.4, as
  now). The program's own retirement is mesh_detach's mesh_retire and the bridge's retirement of a gone pid, as now.
- `int mesh_transfers_start(struct mesh_ctx *,uint64_t plan,uint64_t executions);`: as now, by `plan` in place of
  `ctx->state` (EINVAL where `plan` is 0 or its section or entries lie outside the region).
- `int mesh_metal_transport_create(struct mesh_ctx *,uint64_t plan,void *device,struct mesh_metal_transport *);`: as
  now, by `plan`; mesh_metal_receive_prepare and mesh_metal_publication_prepare test a row's `state` against the
  transport's `offset` alone.

#### The torch bindings (mesh.py, T2)

Context loses `epoch` and `state`; the bindings of mesh_plan, mesh_transfers_start and mesh_plan_retire follow the
declarations above.

### Decisions this part takes, and their sources

#### Item 1, unrolling

- **U1.** plan_unroll's refusal of a write whose least cycle is not 1 (D5) goes; ENOMEM is only the value with a read
  that reaches no later write of it (C3.3: "A value with no such L fails the plan with `ENOMEM`"). L(v) and u stay
  the least lags' (L(v) the largest least lag over v's reads, u the largest L), and the interference tests stay the
  least lags' tests at the distances the copies' executions meet (C3.3). What makes the least lags C3.3's
  happens-before at every larger distance, in the code as it stands: every walk of the template into a write passes
  the write's range's production P (the transport's edges into a landing W leave only from P; edges into P are the
  program's dependences alone, mesh-collective.c plan_template), and both programs' dependences order every event of
  a node in execution t before each of its events in t + 1 (tools/e2b_replay.m e2b_ir: "r's last to its first at lag
  1"; rdma/mesh_compile.py _dependences: "every event no lag-0 edge leaves to every event none enters (lag 1)"), so a
  walk at lag L through P continues through P's next instances at every lag past L. The comments that cite D5
  (plan_unroll, plan_interferes, mesh-graph.h's and mesh-graph.c's source notes, mesh.h's "Copies") cite C3.3 and
  this decision instead. A lone AllGather or AllToAll plans (its landing reaches its next instance at lag 2, u 2).

#### Item 2, order

- **O1.** The sequencer's delivery is in sequence order, the pending list's head first (Défago et al., Fig. 5), with
  no timeout and no skip: Q4′'s bound (orders_deliver's skip after MESH_PLATFORM_NS, `orders_due`, the pending entry's
  `arrived`, mesh-link.c turn_due's wait for it) goes. A head whose plan does not stand here stays at the head.
- **O2. Not done: Q4's hold stays, and with it no execution of one plan interleaves with another plan's on a cable.**
  The evidence:
  - On each queue pair the k-th SEND fills the k-th receive posted (link_prepare's note; Hilland et al., RDMA
    Protocol Verbs Specification, §8.2.2, design/prepared-machine.md#M08), so both ends of a cable must post the
    released executions of several plans on each queue pair in one order: sequence order, the one both ends' parents
    deliver.
  - The ports are the bridge parent's child processes (mesh-flow.c's header: "in a port's child (mesh-child.c)"),
    sharing the region alone with it. What the parent's delivery leaves in the region is each plan's `released`, a
    count. Two deliveries of different plans between two passes of a port leave the port no record of which came
    first, and no word of the region holds it.
  - Carrying the delivery order to the ports needs one of two forms: a queue of the delivered ORDERs in the region
    from the parent to its ports, bounded (a ring) or unbounded; or a hand-off in which the parent waits for its ports
    to take each delivery (not to serve it), a hold. R1 names "queues of desired transfers" (U-Net, VIA: bounded
    descriptor queues) and A7's method is Défago's `pending` queue, so a queue is of the kind the change requests
    name; the brief of this part excludes rings and holds, so the form is the operator's to rule (2026-11-04, amended
    on review W3's finding 5, where this said "a structure no change request names"). Pathways §4.5 (a.txt 420-424):
    "computations may still be interleaved with those submitted by other concurrently executing programs". Hilland et
    al. §8.2.2 is cited at the code (link_prepare's note, design/prepared-machine.md#M08), not read in this part.
  - So the hold that keeps one plan's batch at a time per node stays (orders_served, B2-1, the ports' `served`), its
    comments citing the reason above in place of "A cable never interleaves two executions" (Claude's words, which the
    appendix withdraws). Open for the operator: what carries the sequencer's delivery order from the bridge parent to
    its ports.
- **O3. The deal as it stands.** link_prepare deals each plan's template once at its load (mesh_graph_list_schedule
  over the u copies' units in list order, carried from 0), and each lane serves its plan's executions in order, so
  consecutive executions of one plan interleave across the lines already (each line in execution order, the lines
  independent). The deal carries no bytes from one plan to the next; applying it across plans' consecutive executions
  would be a deal at run time, which it is not as it stands. Recorded, nothing changed.

#### Items 3 and 4

- **H1.** What the last unit added for a slow node is Q4′'s bound (O1); it goes. Every wait stays bounded by C3's
  rule as before (the start's wait, the host's and the GPU's waits on counts, the map reader).
- **H2.** No handler for a truncation is added. The open item "a plan ended at one node while a peer still serves it"
  (V1 #2 and #6) and V1 #3 are withdrawn (below, "Open at the end"), as is Claude's claim that the failport discarded
  such a channel. mesh-flow.c plan_outstanding (the tag's stream_end: a plan ended with requests outstanding loses the
  link) is the bridge's existing end of a plan, not a handler added for truncation, and stays.

#### Item 5, independent flows in uniformly random order

- **S1. Which pages.** A range is a set where every declared transfer of it is MESH_PAGES (an asynchronous, incomplete, noncontiguous
  consumer at every destination, R4). A range with any MESH_JOIN destination keeps the schedule's order (the appendix:
  "A synchronous, complete, contiguous consumer's operands (joins) keep the schedule's order").
- **S2. The set's unit is one tree's pages.** A set range's pages on tree j (the fill's cut, mesh_graph_fill: tree
  j's run) are one set on every edge of tree j, `set` = j + 1, j the tree's index in the solve's trees grouped by
  demand (mesh-collective.c p->tree order, the same at every rank). Every edge of tree j carries the same run of pages
  (plan_walk), so the set is the same pages on every edge of its tree and is named alike at both ends of each edge and
  at every rank by its tree's number. A cable carrying two trees of one range carries two sets.
  (2026-11-04, withdrawn on review W3's finding 6: this decision also argued that the grouping keeps each list a
  topological order, from "a forward follows its receive upstream, whose group start is earlier". The contention
  schedule starts a page on a route's links by Sinnen and Sousa's eq. (14), nonaligned (cut-through), so a page's start
  on a link can precede its finish on the link before; the start order is not shown to be a topological order of the
  bridge's store-and-forward waits, for T6's order itself as for the groups. Open for the operator, below.)
- **S3. The layout.** struct mesh_hop `set` (above). The entries' order (T6 amended): each entry's group is its set's
  entries of the same `node` and `direction` where `set` is not 0, else the entry alone; the entries sorted by their
  group's least `start`, then their group's least `walk`, then their own `walk`. So a set's entries are contiguous on
  their cable within each copy, in walk order (the tree's page order), both ends' alike (a send at x to y and its
  receive at y read the same start; walk order agrees per pair, mesh.h "Both ends"), and every other entry keeps the
  schedule's order. The set's extent is the maximal run of a direction's entries naming one peer within one copy with
  the same nonzero `set`.
- **S4. The generator.** Threefry-4×64-20 (J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, Parallel random
  numbers: as easy as 1, 2, 3, SC 2011, §4.2, Table 2, and §5: "Without AES-NI, we recommend Threefry-4×64-20 on CPUs";
  the bridge's progress threads run on the CPU). Philox4×32-10's key is 64 bits (Table 2: (4+2)×4), shorter than the
  plan's name; Philox4×64-10's is 128 bits ((4+2)×8), the name's width, so the choice rests on §5's recommendation
  (2026-11-04, review W3's finding 11). Threefry-N×W-R "follows the same pattern of S and P boxes as Threefish, but with R rounds"; "The key
  insertion schedule is the same as in Threefish, except that Threefry ignores the 'tweak'"; "For W = 64 and N ≥ 4,
  Threefry's rotation constants are taken from Threefish" (§4.2). So it is Threefish-256 (N. Ferguson et al., The Skein
  hash function family, v1.3, 2010, §3.3) with Nr = 20 and t0 = t1 = t2 = 0:
  - d = 0 .. 19: e_{d,i} = v_{d,i} + k_{d/4,i} mod 2^64 where d mod 4 = 0, else v_{d,i}; (f_{d,2j}, f_{d,2j+1}) =
    MIX_{d,j}(e_{d,2j}, e_{d,2j+1}), y0 = x0 + x1 mod 2^64, y1 = (x1 <<< R_{d mod 8, j}) ⊕ y0; v_{d+1,i} = f_{d,π(i)},
    π = (0, 3, 2, 1) (Table 3); output c_i = v_{20,i} + k_{5,i} mod 2^64;
  - R_{d,j} for Nw = 4 (Table 4): d = 0: 14, 16; 1: 52, 57; 2: 23, 40; 3: 5, 37; 4: 25, 33; 5: 46, 12; 6: 58, 22;
    7: 32, 32;
  - k_4 = C240 ⊕ k_0 ⊕ k_1 ⊕ k_2 ⊕ k_3, C240 = 0x1BD11BDAA9FC1A22; k_{s,i} = k_{(s+i) mod 5} for i = 0, 1, 2 (the tweak
    terms 0), k_{s,3} = k_{(s+3) mod 5} + s mod 2^64 (§3.3.2).
  - The key (k_0 .. k_3) = (struct mesh_state `group`, `epoch`, 0, 0), the plan's name zero-padded (§5: "zero-pad a
    smaller input if the full period length or key space is not needed"); a new random group each launch (R8).
  - The counter (p_0 .. p_3) = (execution t, the set's `set`, block b, 0) (§5.1: "concatenate the object identifier,
    i, and time, T, into a p-bit counter"); block b gives four 64-bit words, the (execution, set) stream's words 4b to
    4b + 3. §5: "(key, counter) tuples must never be improperly re-used". Within one plan, two cables of one pair never
    share a stream (S7′); every edge of tree j at the same place in its pair's cables draws the same stream, as S2's
    one set along the tree means; a replan keeps its name (N1) and restarts its executions at 0, so it redraws the
    cancelled plan's streams: open for the operator (2026-11-04, reviews W1's finding 2 and W3's finding 7, where this
    said no (key, counter) is used twice).
- **S5. The permutation.** D. E. Knuth, The Art of Computer Programming vol. 2, §3.4.2, Algorithm P (read in a
  reproduction citing the 3rd edition, pp. 145-146; the book itself not read): "P1. [Initialize.] Set j ← t. P2.
  [Generate U.] Generate a random number U, uniformly distributed between zero and one. P3. [Exchange.] Set k ←
  ⌊jU⌋ + 1. ... Exchange X_k ↔ X_j. P4. [Decrease j.] Decrease j by 1. If j > 1, return to step P2." Here t is the
  set's size m, X_1 .. X_m the set's members in their list order, the s-th U (s = 0, 1, ... as j runs m, m - 1, ...,
  2) the stream's word s read as r / 2^64, so k = ⌊j r / 2^64⌋ + 1, the high word of the 128-bit product j r, plus 1
  (integer arithmetic, so both ends compute the same; P3's note: k is not a remainder modulo j). The set's position i
  (its i-th entry in list order, from 1) carries member X_i in that execution. R. Durstenfeld, Algorithm 235, CACM 7(7)
  1964: not read (the ACM's copy behind a challenge page; a preview without the procedure).
- **S6. Where the permutation meets the deal.** The deal stays link_prepare's, once a load, over the template's
  positions in list order, each position's length its own hop's `bytes` (O3); each execution's permutation then
  chooses the member each of the set's positions carries, at both ends. This is the brief's "The permutation applies
  to both ends' entries of the set before the existing deal over lines" (the coordinator's instruction; neither change
  request names the deal; review W3's finding 9) exactly where a set's members have equal bytes (every set but one holding its range's last, shorter page); a deal
  of each execution's permuted list would be a deal at run time, which it is not as it stands. Recorded as a
  difference.
- **S7. At the bridge.** A member's record (its page, its words, its bytes) is the one link_prepare prepares, posted at
  the position its execution's permutation gives it, on that position's line. EMSGSIZE where a member is more frames
  than a line of one of its set's positions holds; EINVAL where a set's entries are not contiguous on the pair within a
  copy. Each lane computes an (execution, set)'s permutation once, into room taken at the load (the largest set it
  carries a position of): nothing is allocated at run time. A receive record's round test (`landings`) and a send's
  release are the member record's own. The bridge sees `set`, a set or a sequence, and no kind (R4).

#### Item 6, compiled functions coexist

- **N1. Names.** A plan is (group, function), `function` the program's ordinal of its compiled function, passed to
  mesh_plan and kept by every replan of that function. Every rank runs the same program and compiles its functions in
  the same order (the per-context count C-epoch read was this order; PyTorch 2 §3.1: the backend's compile function is
  called with each captured graph; e2b plans one function, 1). A replan of a function on the same map is EEXIST
  until the program retires the standing one. Sources for coexistence: R. Frostig, M. J. Johnson and C. Leary,
  Compiling machine learning programs via high-level tracing, SysML 2018 ("JAX trace caching creates a monomorphic
  signature for the parameters to the traced computation, such that newly encountered array element types, array
  dimensions, or tuple members trigger a re-compilation"); NVIDIA, CUDA Programming Guide §4.2.2 ("Work submission
  using graphs is separated into three distinct stages: definition, instantiation, and execution"; "It may be launched
  any number of times without repeating the instantiation") and §4.2.3.1 (`cudaGraphExecDestroy()`); Pathways §3 ("We
  refer to these computations with known resource requirements as 'compiled functions'"), §4.4.
- **N2. The handle** is the plan's state section offset, mesh_plan's `*plan` (as an instantiated graph's
  cudaGraphExec_t): the start, the GPU's transport and the retirement take it.
- **N3. Retirement.** A function's claim is closed only by mesh_plan_retire (the program retires the function), by
  mesh_detach or the bridge's retirement of a gone pid (the program itself), or by the plan call's own failure after
  its writes (ESTALE at step 4, its own claim alone).
- **N4. One execution a call.** mesh_transfers_start raises the plan's `executions`, one function execution each:
  torch's call t raises it to t + 1; e2b's run raises it to its steps at once, each step one execution of its one
  function, launched back to back as an instantiated graph is launched any number of times.
- **N5. What goes.** mesh_plan's step 6 (every earlier claim of the client closed), C-epoch (the epoch kept from a
  cancelled plan, the count raised once the plan stands, left at the named epoch less 1 on failure), struct mesh_ctx
  `epoch` and `state`, mesh_claim_close's `others`, and A-T6's single standing graph.

#### Item 7, torch's kernels

- **K1.** T2 finds torch's existing interface for a device tensor over existing memory in torch v2.14.0 (the
  installed .venv-torch's version): its source (aten/src/ATen/DLConvertor.cpp, torch/utils/dlpack.py) and
  documentation, read at that tag. The coordinator's first reading of the file on pytorch main: toDLPack exports an
  MPS tensor as kDLMetal with `data` its storage's base (the MTLBuffer) and `byte_offset` its offset;
  dlDeviceToTorchDevice maps kDLMetal to an MPS device; fromDLPackImpl passes `data` to at::from_blob on that device.
  If v2.14.0 holds the same, the MPS path is torch.from_dlpack of a DLPack tensor of device kDLMetal whose `data` is an
  id<MTLBuffer> over the copy's placed pages, made by newBufferWithBytesNoCopy as mesh-metal.m and e2b_replay.m's
  placed_buffer make theirs, the graph's operators then torch's own MPS kernels; a range is published only after the
  GPU's writes to it are complete (torch's own synchronization). Where v2.14.0 does not hold it, T2 reports with the
  lines read and writes no interface.

### What each source said, where it was read

Text extracted with `pdftotext -layout` into the session scratchpad's papers/ folder:
- **Salmon et al. 2011** (random123.txt): §4.2 (lines 360-424): Threefry-N×W-R, its key schedule, its rotation
  constants from Threefish for W = 64, N ≥ 4, "Threefry-4×64-72 is identical to Threefish-256 with zero 'tweak'";
  Table 2 (476-500): key and counter widths ((4+4)×8 for Threefry-4×64, (4+2)×4 for Philox4×32, (4+2)×8 for
  Philox4×64); §5 (552-553): "we recommend Threefry-4×64-20 on CPUs"; (565) zero-padding a smaller input; §5.1
  (586-640): keys and counters from application variables; (673-680) "(key, counter) tuples must never be improperly
  re-used" (line ranges corrected on review W3's finding 11).
- **Ferguson et al., Skein v1.3** (skein.txt): §3.3 (662-776): Threefish's rounds, subkey injection every four
  rounds, the MIX function, Table 3 (π), Table 4 (R_{d,j}), §3.3.2 the key schedule and C240.
- **Frostig, Johnson and Leary 2018** (jax-sysml.txt, lines 60-90): trace caching and re-compilation.
- **Pathways** (a.txt, lines 185-265): §3, compiled functions; §4.5 (420-424), interleaving with other programs.
- Read on the web, not saved: Knuth's Algorithm P as reproduced at gist.github.com/45cd3d09e545815cffe650ffe48eb23d;
  the CUDA Programming Guide, docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html, §4.2.2
  and §4.2.3.1; pytorch main's aten/src/ATen/DLConvertor.cpp (K1).
- Not read: Durstenfeld 1964 (S5); Boehm and Adve 2008 (item 4: nothing is built from it); J. W. S. Liu and C. L. Liu
  1974 (the deal is the existing code's).

### Amendments during the work (2026-11-04), each the coordinator's ruling on an implementer's finding

- **S7′** (B3's finding: the deal spreads a pair's positions over every cable of the pair, and each port's child holds
  records for its own cable's units alone; a receive record's round test and the bit its landing flips count its own
  `landings`, so a member placed on another cable in a later execution has no record there, or one out of step).
  The permutation is per cable: on each cable of the pair, the set's members whose positions the deal gave that cable
  are permuted among those positions (S5, with t their count), each execution, at both ends. This is the appendix's
  text at the bridge's level ("each execution of a set on a cable is sent and received in a uniformly random
  permutation of its pages"); how a pair's sequence is spread over its cables stays the bridge's own (R5, R7). Which of
  a set's pages a cable carries is the deal's, fixed at the load; their order on it is drawn each execution. The
  counter's fourth word is the cable's place in the pair's key order (mesh-map.h mesh_cables: "cables in key order",
  the same at both ends), so no two cables of one pair share a stream (Salmon et al. §5; S4). B3's ENOTSUP refusal
  goes. (Its last sentence here, a no-cycle argument resting on "each page's start below follows its finish above",
  is withdrawn: S2's note, review W3's finding 6.)

- **P3** (review W3's finding 3: no program declared a MESH_PAGES consumer, so item 5 was unreachable; P1's and
  A-T8's reason, one publish a range, is the producer's, while R4's kind is the consumer's: "function B can work on any
  pages received in any order"). In the torch lowering, a declared transfer is MESH_PAGES where its consumer at the
  destination is R4's asynchronous, incomplete, noncontiguous consumer: each AllReduce partial to the root, consumed by the root's combine (R4: "reductions'
  partial sums"; Gibbons, ⊕ associative and commutative), and a range whose every reader there is an operator of
  GSPMD's elementwise set (a map over pages, the appendix item 5's "a pointwise nonlinearity after a ffn"); MESH_JOIN
  otherwise (a Dot, a cat, a slice, a clone). A-T8 is reversed for those. Every consumer still reads a range only once
  it is complete (R4: "complete but not contiguous before function B is complete"). e2b's kinds stay P1's (open).
- **K1-a to K1-e** (T2's report, review W2): the DLPack structure is DLManagedTensorVersioned (dlpack.h: the legacy
  one "deprecated after DLPack v0.8"); torch's own synchronize before each write into a range on the GPU, besides
  before each publish (Metal orders accesses per buffer object, MTLResource.h `hazardTrackingMode`, MPSAllocator.h
  `UsageFlags::HAZARD`, and two values over one page are two buffer objects, so C3.3's read before write needs it);
  a plan whose binding fails after mesh_plan is retired at once (N3, R8); torch's MTLDevice is read from a probe MPS
  tensor's buffer (OperationUtils.h getMTLBufferStorage; torch has no Python call returning it); every copy, CPU and
  MPS, bound by torch.from_dlpack, kDLCPU or kDLMetal, and synchronized by its device module
  (torch.get_device_module(d).synchronize(); torch.cpu.synchronize documented "to facilitate device-agnostic code"),
  one form (review W3's finding 12, C3.8), torch.frombuffer gone.
- **E1** (review W2's finding 2): an execution is consumed at its submission. Once mesh_transfers_start has raised
  `executions` to t + 1, the next call is t + 1 whether or not this call's run fails (`executions` and the supplied
  words only rise: C3.7; mesh-call.h), so no call rewrites a published execution's pages; a start that fails before its
  raise submitted nothing.
- **N3′** (review W2's finding 3): torch retires a compiled function's plan when torch drops its callable, by Python's
  weakref.finalize on it (at its collection, or at exit before module teardown), after its devices' synchronize.
- **N2′** (review W1's finding 4): mesh_transfers_start refuses (EINVAL), before any raise, a handle that is not a
  plan standing in a claim of this client (the transport's tests, then the heads' walk); a second mesh_plan_retire is
  ENOENT.
- **O4** (review W3's finding 2): a delivery stores a plan's `released` only upward (C3: no state word written to an
  earlier value while its plan stands).

### Record of this part

#### Commits by owner

- Coordinator (mesh): 5ba3a080 (this part), 692bb5e7 (INDEX.md), 53c7233e (S7′), f9d8bb2c (the reviews' corrections
  to O2, S2, S4, S6, S7′ and the sources' lines), and this record.
- C2 (mesh rdma/mesh-collective.{h,c}, mesh-graph.{h,c}): e97d80ae (the header), dd57daa6 (U1, S1-S3, N1-N5),
  2a128d58 (mesh-graph's D5 citations to C3.3 and U1), b2e92cc6 and 7ec4df3f (U1's premise stated at struct
  mesh_dependence; the no-cycle argument withdrawn, with C-weights' reason for `page` > 0).
- L2 (mesh rdma/mesh.h, mesh-dataflow.{h,c}, mesh-call.h, mesh-metal.{h,m}; metal-microbench tools/e2b_replay.m):
  10b680a0 (the headers: MESH_VERSION 111, struct mesh_hop `set`, struct mesh_ctx of 32 bytes, mesh_claim_close,
  mesh_plan_retire, the start and the transport by plan), 77bd59dd (their code), aa112f5b (mesh.h: S3's order),
  e65f770b (N2′; the set's text; the no-cycle claim withdrawn); metal-microbench 3b996181 (e2b: function 1, the plan
  by its handle), 062a8474 (the plan call's comment).
- B3 (mesh rdma/mesh-flow.c, mesh-records.c, mesh-link.c, mesh-bridge.h): 5eb92c69 (O1/H1: Q4′'s bound gone; O2: the
  hold's comments re-cited), 24c6da33 (S4-S7: threefry4x64_20, set_draw, the sets' tables, the drains by position),
  e529e2c1 (S7′), 83dbacbb (D5's last citation; the streams' text; O4).
- T2 (mesh rdma/mesh.py, mesh_compile.py): 28897023 (U1; N1-N5: an ordinal a compiled graph, a handle each,
  STANDING and A-T6 gone, replans retiring), 16bb2864 (K1: DLPack over placed pages, torch's MPS kernels),
  091d2d05 (W2 #1-#6, W3 #12 and #3: E1, N3′, K1-e, P3, the float8 codes, citations).

#### Reviews and where their findings went

- W1 (the mesh side): 1 (a head whose plan never stands blocks later deliveries) open, below; 2 (the counter's text)
  to B3 (83dbacbb) and this part (S4); 3 (U1's premise) to C2 (b2e92cc6); 4 (the handle's checks, a second retire)
  to L2 (e65f770b, N2′). Threefry and Algorithm P found line by line as Skein §3.3 and S5 state them; S1-S3, S7′,
  N1-N5, O1 found as written.
- W2 (the torch and program side): 1, 2, 3, 4, 5, 6 to T2 (091d2d05: the synchronize before retiring, E1, N3′, the
  float8 codes, Header's citation, the deleter's citation); 7 to L2 (062a8474). ctypes layouts, the DLPack structures
  and torch v2.14.0's DLPack import of kDLMetal confirmed by W2's own reading at the tag; T2's four differences
  found sound.
- W3 (items 1-7 across both repos): 1 and 2 open, below (2's `released` to B3, O4); 3 to T2 (P3); 4 to C2 (U1's
  premise); 5 to this part (O2's reason amended); 6 to C2, L2 and this part (the no-cycle argument withdrawn), open;
  7 to B3 and this part (S4), open in part; 8 to B3 (83dbacbb); 9, 11 to this part (f9d8bb2c); 10 this record; 12 to
  L2 (062a8474, e65f770b) and T2 (K1-e). Found clean: items 3 and 4 (no handler for a slow node or a truncation);
  no chooser, kernel, ring, timeout skip, second form; transport terms in the bridge's files alone; no command buffer
  made; the overturned decisions gone from code (but dated notes).

#### What torch v2.14.0 holds (K1, read at the tag by T2 and again by W2)

DLConvertor.cpp: toDLPackImpl's MPS branch exports `data` = `src.storage().mutable_data()` and `byte_offset` =
`src.storage_offset() * c10::elementSize(src.scalar_type())`; dlDeviceToTorchDevice: `case DLDeviceType::kDLMetal:
return at::Device(DeviceType::MPS, index);`; fromDLPackImpl passes `dl_tensor.data` to at::from_blob with its strides
and toStorageOffset(byte_offset). aten/src/ATen/native/mps/OperationUtils.h 128-129: getMTLBufferStorage is
`__builtin_bit_cast(id<MTLBuffer>, tensor.storage().data())`; 536: MPS storage data is "a pointer to id<MTLBuffer>
object". MPSAllocator.h 292: the allocator's device is MPSDevice::getInstance()->device(); MPSDevice.mm: the first of
MTLCopyAllDevices not low-power and supporting Mac2. torch/csrc/utils/tensor_new.cpp tensor_fromDLPack: a
`dltensor_versioned` capsule of major version at most DLPACK_MAJOR_VERSION.

#### Integration

Once no implementer or reviewer ran (mesh 091d2d05 and this record; metal-microbench 062a8474; both trees clean and
level with origin): `make -B -C rdma` 0 (mesh-flow, mesh-heartbeat, mesh-stat, libmesh.dylib), `make -B -C rdma
native-audit` 0, `make .build/e2b_replay` 0 (libmesh_metal.dylib and e2b_replay rebuilt against the new libmesh), no
warning or error; rdma/mesh.py, rdma/mesh_compile.py and tools/mesh/run_pair.py compile (Python's compile, no
bytecode written). Nothing was run.

#### Open at the end, for the operator

- **Item 2 not done (O2).** Q4's hold stays: no execution of one plan interleaves with another plan's on a cable. What
  carries the sequencer's delivery order from the bridge parent to its ports: a queue of delivered ORDERs in the region
  (bounded or not) or a hand-off on each port's taking a delivery. The brief excluded both.
- **Delivery's liveness** (W1 #1, W3 #1). With Q4′'s bound gone (items 2 and 3), a pending ORDER whose plan never
  stands here (its rank exits, or retires the function, before the ORDER comes; a map move with no replan, the ORDERs
  being kept across this node's serial move by V1 #4's ruling) stays at the head, and every later ORDER at that node
  waits behind it. Défago's Fig. 5 delivers each message in turn; the "plan stands here" condition is how a delivery
  reaches its plan (the plan's `released`), and none of the change requests says what a delivery to a plan not
  standing is.
- **A plan's identity across a map move** (W3 #2, W1 #2, W3 #7). A replan keeps its name (N1), so an ORDER of the
  cancelled plan can release its replan (`released` now only rises, O4), and the replan redraws the cancelled plan's
  permutation streams (Salmon et al. §5's "improperly re-used"). No name both programs give alike at every rank tells
  two incarnations of one function apart (each node's map serial is its own count, Q3′).
- **No cycle on a cable** (W3 #6). The contention schedule starts a page on a route's links by Sinnen and Sousa's
  eq. (14), nonaligned (cut-through), so a page's start on a link can precede its finish on the link before; neither
  T6's start order nor S3's groups is shown to be a topological order of the bridge's store-and-forward waits (R5's
  and MSCCLang §5.2's condition). With it, C-weights' refusal of a `page` weight of 0 has lost its stated reason
  (Definition 1 allows c(e) ≥ 0); the refusal stays.
- **U1's premise.** The least lags are C3.3's order at every later distance because both programs' edges order each
  node's events of t before those of t + 1; mesh_plan does not test it (the appendix: ENOMEM refuses only a value with
  no such L). A program without such edges would get pages shared that its edges do not order.
- **Item 5's reach.** Torch's P3 kinds give sets (AllReduce partials; sums read only by elementwise operators); e2b's
  consumers stay MESH_JOIN (P1): which e2b stages take pages in any order is its manifest's to say
  (tools/prepare_e2b.py), not decided here. Recorded differences: S6 (the deal fixed at the load over positions, the
  brief's "before the deal" exact only where a set's members have equal bytes), S7′ (the permutation per cable; which
  pages a cable carries is the deal's), S2 (a cable carrying two trees of one range carries two sets), N4 (e2b's
  steps launched at once by one start, each step one execution).
- **The torch MPS path (K1)** is built on torch v2.14.0's DLPack import as read, and nothing was run: whether every MPS
  operator accepts a buffer its allocator did not make; whether MPS command buffers in flight retain the buffers they
  use after a failed call; torch.mps.synchronize's wait is torch's own, unbounded by this code (no kernel spins).
- **Torch retirement at exit** (N3′): a finalizer's failure at exit is printed, not raised (Python's documentation).
- Carried: L3; V1 #9; the torch path's refused collectives and operators; the α–β chooser not built (ruling 3); `qps`
  in mesh-map.h; runs wait for the operator's authorization (C3.8).

## The documents' revisions of edd4b583

The coordinating agent's contract for the three places design/plan-to-bridge-change-request.md and
design/aot-compilation-change-request.md changed at mesh edd4b583, from mesh edd4b583 and metal-microbench 062a8474:
R5's "The order of each cable's list, by the reference algorithm"; the aot appendix item 2's "Delivery at a node"; and
its item 5's "e2b declares its function B consumers". The documents are the requirement; where they and anything above
in this file differ, they hold. They overturn T6 and S3 (each node pair's list in the order of its link's scheduled
starts, a set's entries grouped at their least start) and Q4 with Q4′'s remainder, O2 and B2-1 (the hold: a node
releases the next plan only once every port has served the previous one; mesh-records.c orders_served). This file's
"Rules" and design/disjoint-pages-rulings-coordination.md, "Rules for this unit", apply verbatim, and: no chooser,
kernel, timeout skip, hold or second form is added; no handler for a slow node, a "desynchronization" or a truncation;
nothing the documents do not name; every function added names its source and section and is that source's method as
the source states it; where the documents are silent, the decision below, and where an implementer needs another, it
stops that piece and reports.

### The file split, one owner a file

| files | owner |
|---|---|
| mesh rdma/mesh-graph.h, mesh-graph.c, mesh-collective.h, mesh-collective.c, mesh.h (but the lines below the coordinator lands) | C3 |
| mesh rdma/mesh-flow.c, mesh-records.c, mesh-link.c, mesh-bridge.h | B4 |
| metal-microbench tools/prepare_e2b.py, tools/e2b_replay.m | E2 |
| reviews, written findings only | X1 (the mesh side), X2 (the program and torch side), X3 (across both: each gap closed as the documents and their sources state; T6, S3 and Q4 gone; nothing added) |
| this file, INDEX.md, mesh.h's MESH_VERSION and struct mesh_state (this part's commit), the builds | coordinator |

Torch (rdma/mesh.py, rdma/mesh_compile.py) changes nothing: mesh.py binds struct mesh_state through `entries`
(offsets 0 to 104), which nothing here moves, and the torch lowering's kinds are P3's.

### The names files share

#### The region (mesh.h; the coordinator lands these lines, C3 writes the notes around them)

- MESH_VERSION 112. Its note: "112: struct mesh_state `served` read as `taken` and `sequences` appended in the
  struct's tail (bytes 240-255; still 256 bytes): the plan's sequence words, one a copy, which the bridge parent writes
  and each port merges (design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node";
  design/compilation-coordination.md, "The documents' revisions of edd4b583", DL1)".
- ```c
  struct mesh_state { _Alignas(64) uint32_t copies; uint32_t receives,ranges,releases;
    uint64_t phases,counts,supplied,status; _Atomic uint64_t executions; uint64_t padding;
    uint64_t group,epoch,serial,hops,entries; _Atomic uint64_t held; _Atomic int32_t refused[MESH_PORTS];
    uint64_t nodes; uint32_t sender,padding1; uint64_t submitted; _Atomic uint64_t released;
    _Atomic uint64_t taken[MESH_PORTS]; uint64_t sequences,padding2; };
  ```
  `taken` at 176 (where `served` was), `sequences` at 240, `padding2` at 248; no other field moves.
- The state section's layout: the struct; the phases, counts, supplied words and status (as now); the sequence words,
  `copies` uint64 words at `sequences`, copy c's at `sequences` + 8 c; the entries; each 64-aligned. mesh_plan places
  and names them (plan_layout), and they start at 0 with every byte before the entries (D3).
- `released` (DL1): the executions of the plan whose sequence numbers the bridge parent has published, each in its
  copy's word, in sequence order: L. Lamport's ms, "the total number of messages sent" (Concurrent reading and writing,
  CACM 20(11), 1977, "Message Buffer"). The bridge parent's alone; only rises.
- `taken[p]` (DL1): Lamport's mr, port p's: no lane of port p reads the number of an execution of the plan below it
  again (the least execution a lane of port p carrying the plan may still read); for a plan port p carries nothing of,
  or has cancelled or ended, `released` as it rises. Port p's alone; only rises.
- Sequence word c (DL1): Lamport's B[c], P = u: the sequence number of the execution t (t mod u = c) the parent
  published last in copy c, struct mesh_order `sequence` + t - `first` of the ORDER that holds t. The bridge parent's
  alone.

#### The graph module (mesh-graph.h, C3)

```c
/* M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023, section 5.2, step 3: "Sort
   instructions into a global topological order with respect to their dependencies with a heap using the priority to
   order instructions."  Vertex v's priority priority[v]; edge e runs from[e] -> to[e] (an edge given twice counts
   twice and is removed twice, so it changes nothing).  A vertex enters the heap once every edge into it leaves a
   placed vertex; the heap's first is placed next, the greatest priority first and a tie to the lower vertex
   (design/compilation-coordination.md, "The documents' revisions of edd4b583", GO3).  place[v] out, v's place in the
   order.  0; EINVAL where an edge names no vertex or the edges hold a cycle (a vertex left unplaced); ENOMEM */
int mesh_graph_heap_order(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from,
                          const uint32_t *to, uint32_t *place);
```

Its heap is the binary heap mesh-graph.c already keeps for mesh_graph_lags (lags_up, lags_down), one copy (C9): those
take their comparison as an operand, mesh_graph_lags' behavior unchanged.

#### The bridge (mesh-flow.c, mesh-records.c, B4)

No message, wire form or declaration of mesh-bridge.h changes: struct mesh_order as it is. mesh-records.c's
orders_served and `last` go; served_tell is the port's tell of `taken[p]` (its name B4's).

#### e2b's manifest (tools/prepare_e2b.py, tools/e2b_replay.m, E2)

Each kept send route of a rank stage gains `"kinds"`: a list of strings, one an entry of its `"destinations"`, in that
order, each `"pages"` or `"join"` (EB1). e2b_replay.m's placement function carries it with the route, and e2b_ir gives
declared transfer (route, destination d) the kind MESH_PAGES for `"pages"` and MESH_JOIN for `"join"` (EB2).

### Decisions and their sources

#### R5, the order of each cable's list (C3)

- **GO1. The instructions and their dependencies.** The instructions are the task graph's tasks (T1, T1′, as
  plan_schedule builds them) and the plan's communications (T4: a page over a tree edge, plan_communication). A
  communication is one instruction: its send at the edge's upper node and its receive at its lower node take its one
  place, so a node pair's sends at one end and its receives at the other are the same hops in the same order (mesh.h,
  "Both ends"; R1: "the bridge's records name the same pages"). MSCCLang §4.2 expands a remote copy "into a send and a
  receive instruction" joined by "a communication edge"; joining them in one instruction is where this differs, so
  both ends' lists agree, recorded at the code. The dependencies: each local edge of the task graph (P(d) -> S(d), a
  lag-0 dependence), as it is; each remote edge, its tail task -> its route's first communication -> each next ->
  its last -> its head task: MSCCLang's communication and processing edges, a page sent before it lands and landed
  before it is forwarded or consumed (plan_template's P -> W(y), W(y) -> F(y, c), L(z) -> C(z)). Shared prefixes
  (T4) give an edge more than once, which changes no bottom level and no order.
- **GO2. The priority** is the bottom level (R5: "The priority is the bottom level"), eq. (7) of O. Sinnen and L. A.
  Sousa (IEEE TPDS 16(6) 2005, §2.2.1) over the instructions: bl(n) = w(n) + max over the dependencies e = (n, m)
  {c(e) + bl(m)} (mesh_graph_bottom_levels). A task weighs its weighting operand's weight (A3), as now; a
  communication on link L weighs its link time ς(e, L) = c(e) / b(L) (Sinnen and Sousa, §5; b(L) T2's rate), the
  greatest over the task graph's edges e whose routes cross it (T4: one occupation for every edge that shares it;
  eq. (7)'s bottom level is "the length of the longest path leaving the node", so the longest), and every dependency
  costs 0 (a remote edge's time is in its communications; a local one is free, Definition 2, T1′).
- **GO3. The heap.** MSCCLang §5.2, step 3 (mesh_graph_heap_order): the instructions with no dependency left unplaced
  in a heap, the greatest bottom level first (R5: "Tasks are taken in decreasing bottom level"; HLFET's priority), a tie
  to the lower instruction: the tasks as plan_schedule numbers them, then the communications as plan_trees numbers
  them, the same at every rank (F. B. Schneider, ACM Comput. Surv. 1990). MSCCLang names no tie; the lower index is
  the one every rank computes alike.
- **GO4. The lists.** Each of this node's entries (plan_walk; every field as assigned now) takes its communication's
  place. A set's entries on one node pair and direction (S1, S2: its nonzero `set`, `node` and `direction`, tree j's
  pages on that edge) take consecutive places at the place of the first of them in that order (R5: "They take
  consecutive places at the place of the set's first page in that order"), each among them by its own place; every
  other entry is at its own place. The entries are sorted by (their group's least place, their own place), a total
  order (each communication has one place). So each node pair's list at each end is a subsequence of the one global
  order (a set's at its first page's place), both ends' alike, every copy's the same; T6's and S3's order by
  scheduled starts goes, with the entries' `start`, `earliest` and `least`. Where it is a topological order: a set's
  pages on an edge depend only on its pages on the edge above (or the production), all of which take places before
  the least place of those below, so the order of the groups is a topological order of the instructions with each
  group whole (R5: "A set's pages ... share their predecessors"), and with every lag pointing forward each cable's
  sequence of executions is one (R5, MSCCLang §5.2: "By assigning instructions to thread blocks in a topological order
  that respects communication and processing edges, all implicit dependencies introduced by thread block sequential
  execution cannot produce cycles"). This answers the open item "No cycle on a cable" (W3 #6), now that the
  dependencies are store-and-forward ones (GO1).
- **GO5. The schedule stays, ordering nothing.** mesh_graph_contention_schedule is still called as now (R5's method
  and "Code: rdma/mesh-graph.{h,c}: ... the list schedule with edge scheduling"; A6: "One schedule per function"), and
  its starts order no list (R5: "Scheduled start times do not order the lists"). What reads them is the operator's to
  say (open). The demands stay numbered by their first declared transfer (T6's other half, not an order of the lists).

#### Item 2, delivery at a node (coordinator's lines in mesh.h; C3's layout; B4)

- **DL1. The words** (above): Lamport 1977, "Message Buffer": "Assume a buffer B which can hold P messages in
  locations B[0], ..., B[P - 1]"; "multidigit variables ms and mr to hold the total number of messages sent and
  received"; "sender: wait until ms < mr + P; put message in B[ms mod P]; ms := ms + 1"; "receiver: wait until mr < ms;
  read message in B[mr mod P]; mr := mr + 1"; each variable one writer (C3's rule 2; the item: "Each word has a single
  writer"). P is u (the item: "one word per copy"; R2, C3.3: "execution t runs copy t mod u"), the message of
  execution t its sequence number, ms `released`, mr port p's `taken[p]`. Where it differs from the paper, recorded at
  the code: every port is a receiver of every message (each port serves every standing plan), so the sender's wait is
  over every receiver's mr; and a port's mr is the least of its lanes' next reads, each lane reading the messages in
  order on its own.
- **DL2. The sender: delivery in sequence order, publication by the buffer** (mesh-records.c orders_deliver). The
  pending ORDERs in arrival order, which is sequence order (Défago et al., Fig. 5; Q3, Q3-tree), the head first, no
  timeout and no skip (O1). The head is matched to a standing plan of its (group, epoch) on this node's map serial, as
  now; not matched, it stays. For its next execution t = the greater of its `first` and the plan's `released`, number
  `sequence` + t - `first`: where t >= u and some port of the plan's port set (mesh-map.h mesh_ports_named, the start's
  set, one copy) that has not refused the plan (`refused[p]` 0) has `taken[p]` < t - u + 1, Lamport's sender waits
  ("wait until ms < mr + P"): the head stays, read again on a later turn of the parent's event loop, nobody blocking
  (as O1's head that does not stand here); else the number stored in word t mod u (an atomic store), then `released`
  t + 1 (release), "put message in B[ms mod P]; ms := ms + 1"; then on to t + 1. An ORDER done (its `end` reached, or
  every execution of it below `released` already) leaves the head; hdr `plans` is woken after each turn that published
  (region-shared), so the ports see it. A port set that cannot be computed (mesh_ports_named's error) leaves the head
  staying, said on the bridge's log once a head (B2-1's read-as-served goes: it would let the sender overwrite a word
  a port still reads). The head's wait holds every later ORDER behind it: the deliveries are in sequence order, and no
  port could serve a later number before it where it carries both; it is Lamport's sender's wait for the buffer the
  item names, not Q4's hold of a plan's whole batch (recorded; the operator's to judge, below).
- **DL3. The receivers: the k-way merge at each port** (mesh-flow.c). Each lane of a port, each direction, serves the
  executions of every plan it carries in sequence-number order: of each such plan (each loaded flow, not cancelled,
  whose records this lane carries that way), its next execution n there (the lane's cursor for it) is a sequence of
  that plan's, and the lane takes the one of least number among them (the item: "a k-way merge (D. E. Knuth, The Art
  of Computer Programming vol. 3, §5.4.1)", not read: the book, no reproduction found; built as the item states it,
  the least of the k sequences' first elements next, by comparing them, no tree of losers). Execution n of a plan is a
  candidate where n < its `released` (acquire), its number then its copy's word n mod u (Lamport's receiver: "wait
  until mr < ms; read message in B[mr mod P]"). Lamport's "writing data elements in one order and reading them in the
  opposite order": the parent publishes in sequence order, so the lane reads every candidate (first reading), then
  every plan's `released` and hdr `plans` again (second reading); the least number over the second reading is next
  (every number below one acquired in the first reading was published before it). Where hdr `plans` moved since the
  port's last walk, the lane takes nothing this pass (a plan standing, and so maybe published, that the port has not
  loaded would be missed); where the least is of an execution its node's program has not submitted (n at or past the
  plan's `executions`, D2, as now), the lane takes nothing (it cannot serve it, and nothing after it comes first).
  Once taken, the lane posts that execution's records on it in order, as now (each SEND once released by its word,
  each receive record's round once its last posting landed: D.3.5), then takes the next; an execution with no record
  on the lane is taken and passed alike. Each pass the port stores `taken[p]` for each plan it carries (release, only
  rising), the least execution any of its lanes carrying the plan may still read, and wakes hdr `plans` (no raise)
  where it rose. Both ends of a cable hold the same plans and the same records on each lane (link_prepare, one deal),
  and read the same numbers, so they post the same order on each queue pair (the item: "they merge alike"), the k-th
  SEND filling the k-th receive (Hilland et al. §8.2.2, design/prepared-machine.md#M08).
- **DL4. What goes**: Q4's hold, its test (mesh-records.c orders_served) and `last`; link_run's serving of one plan at
  a time and its batch bound per plan (`released` as a count of a batch; served_tell's hold note); the ports' `served`
  as "served whole"; O2's open item (what carries the delivery order to the ports: the sequence words, merged). A port
  ends, cancels or carries nothing of a plan as now, its `taken[p]` then following `released`.
- **DL5. Completions name their records** (as now), and each record names its plan (struct flow), so a pass's
  completions are counted to their own plans while several interleave on one queue pair (B4: room in the records'
  lines, M08, M29, without moving a field the post reads).
- **DL6. Q6, as this part leaves it:** what A7 adds to C3.3's run time is the sequencer's `seqnum`, each parent's
  pending ORDERs (the head's `first` and `sequence` advanced as its executions are published), and each plan's
  `submitted`, `released`, `taken` and sequence words.

#### Item 5, e2b's asynchronous, incomplete, noncontiguous consumers (E2)

- **EB1. The kinds** (tools/prepare_e2b.py). A kept send route's consumer at each destination is an asynchronous, incomplete, noncontiguous consumer,
  `"pages"`, where the route is a piece of a section cut over several contributors and its destinations' consumer is
  the sum of the pieces (prepare_e2b.py combine: the FFN row split's contributions and the head split's attention
  partials, added by ADD at each rank holding the stream): R4, "reductions' partial sums", ⊕ associative and
  commutative (Gibbons 1996); the item: "e2b's reductions of partial sums are function B". Every other route's
  consumers are `"join"` (synchronous, complete, contiguous consumers): a whole operand broadcast, a single owner's activation, the split-KV partials
  and normalizers (kv_merge), the vocabulary's routes. The kinds are given once the routes' destinations are cut (the
  publication's readers), one a destination.
- **EB2. e2b's IR** (tools/e2b_replay.m): e2b_placement carries `kinds` with each route; e2b_ir gives each declared
  transfer its destination's kind. A route without `kinds`, one a destination, or with another word, is refused (exit
  2, naming it: prepare the directory again), as a stage without its dataflow list is. P1's "every e2b consumer
  MESH_JOIN" is reversed, with a dated note: the kind is the consumer's (R4; P3's reason), not the producer's one
  publish.
- **EB3. Not declared asynchronous, incomplete and noncontiguous:** the split-KV merge (kv_merge) reads each rank's partial with its normalizer's
  rescale, a combine R4's FlashAttention example would make asynchronous, incomplete and noncontiguous; the item names e2b's reductions of partial
  sums, so it stays synchronous, complete and contiguous (open for the operator).

### What each source said, where it was read

Text extracted with `pdftotext -layout` into the session scratchpad's papers/ folder:
- **MSCCLang** (b.txt): §4.2 (279-292): "A remote copy expands into a send and a receive instruction", "a
  communication edge that indicates that the receiving side synchronizes with the sender", "processing edges, which
  represent the execution-order dependencies within ranks"; §5.2 (390-417): steps (1) to (4), step (3) as R5 quotes
  it, and "An Instruction DAG is guaranteed to have a global topological order ... cannot produce cycles so that the
  MSCCL-IR does not have deadlocks".
- **Sinnen and Sousa 2005** (sinnen-sousa.txt): eq. (7) and "the length of the longest path leaving the node"
  (204-211); ς(e, L) = c(e)/b(L) (§5, as above); §7.1.2 (673-690): the bottom level as the contention model's priority.
- **Lamport 1977** (lamport-rdwr.txt, from lamport.azurewebsites.net/pubs/rd-wr.pdf): "Message Buffer" (214-261, the
  right column): the buffer, ms and mr, the sender's and receiver's algorithms quoted in DL1, "If the buffer is full,
  then the sender must wait until the receiver empties a buffer position by reading the message in it"; the
  conclusion (386-392): "writing data elements in one order and reading them in the opposite order".
- **Défago et al.** (defago.txt): Fig. 5, as above.
- **Not read:** Knuth vol. 3 §5.4.1 (the book; searched, no reproduction of its text found): the merge is the item's
  own statement, the least first; the tree of losers that section is cited for elsewhere is not built.

### Amendments during the work (2026-11-05), each the coordinator's ruling on an implementer's finding

- **DL7** (B4's finding: a plan stands at mesh_plan's store of its head's publication `state`, before mesh_plan
  raises hdr `plans`, so a bridge parent can see it standing and publish its first number, and a later plan's, before
  the raise; a port that has not loaded it would then pass DL3's `plans` test and take the later number first while
  its peer takes the earlier). The bridge parent adds 1 to hdr `plans` (release, then wakes it) before it publishes a
  plan's first sequence number (its `released` 0), so a port that acquires any later number reads the move and walks
  the heads before it takes; a walk that reads the raise reads the plan standing, since the parent saw it standing
  before raising. Why: DL3's reading in Lamport's opposite order needs every plan that can hold a smaller number to be
  visible to the second reading; mesh_plan's own order (stand, then raise) leaves the first publication of a plan the
  one case it is not. mesh.h's note on hdr `plans` names the parent's first publication among its raisers.
- **DL8** (B4): a plan whose sequence words do not lie inside the region after its struct, 8-aligned, or whose
  `copies` is 0, is EINVAL: orders_deliver leaves its head staying, said once a head, and every port refuses it
  (link_prepare, through plan_inside; plan_sequences_inside, one copy, mesh-flow.c), as a plan's entries outside the
  region are refused. Why: an atomic store or load there would be a fault, not an error value.
- **GO4′** (C3): the no-cycle argument at plan_schedule rests on each group's least-placed member's predecessor (its
  place precedes that member's, so every group's least place follows the least place of each group it depends on),
  for any weights, where GO4's text said every page above takes its place before the least place of those below,
  which holds only where a tree's pages have equal bottom levels. The conclusion stands. `walk` goes with T6's ties.
- **DL2′** (reviews X1 #2 and X3 #4: DL2's "the greater of its `first` and the plan's `released`" lets `released` pass
  executions whose words were never written, where an earlier ORDER of the plan never reached this node, so a port
  reads them as published with a stale number, and Lamport's ms advances by one message at a time). The head's next
  execution is the plan's `released`: where `first` <= `released` < `end`, it is published as DL2 states (number
  `sequence` + `released` - `first`); where `end` <= `released`, the ORDER leaves the head (its executions all
  published); where `first` > `released`, the head stays, said on the bridge's log once a head (Défago et al., Fig. 5:
  the message numbered `nextdeliver` has not arrived). So an ORDER dropped here (order_take's ENOMEM, order_relay's
  port not greeted) leaves its plan's later executions unreleased here, as those drops' notes say, and the head waits
  (open, "Delivery's liveness").
- **DL3′** (X3 #3): where the first reading acquires no candidate but the second finds one, the lane takes nothing and
  its port passes again at once (link_run's 1): a number only the second reading found is not known least. Recorded at
  lane_take.
- **DL8′** (X3 #5): mesh-bridge.h declares plan_sequences_inside, the one copy (mesh-flow.c), which mesh-records.c's
  orders_deliver calls; and since plan_inside includes it, plan_cancel raises no count of such a plan (it names no
  count a port could serve: every port refuses it). This amends "No ... declaration of mesh-bridge.h changes" above.
- **DL9** (X1 #7, V1 #9): the delivery's two hand-offs raise hdr `plans` (mesh_notify: add 1, release, then wake),
  where they woke it without raising: the bridge parent after a turn of orders_deliver that published, and a port
  where its `taken[p]` rose (taken_tell, plan_end). Why: a wake with no raise that lands between a waiter's read of
  `plans` and its wait is lost until MESH_PAUSE_NS (3 s; V1 #9), and the delivery now hands off between the parent and
  its ports once per execution past the u-th, where Q4's came once per ORDER; os_sync_wait_on_address returns at once
  where the word has moved. The wake without a raise was Q4′'s ("A port wakes `plans` (without raising it) after
  raising `served[p]`"), overturned with Q4. Its cost, recorded: each raise moves `plans`, so each port walks the heads
  on its next pass (plans_walk) and DL3's `plans` test lets a lane that read the old value take nothing for one pass.
- **GO3′** (X3 #2): MSCCLang §5.2 step (1)'s own priority is "the depth (maximum hops from a root node) and reverse
  depth (maximum hops to a leaf node)", counted in hops; the priority here is R5's, the bottom level by eq. (7) over
  the weights (GO2). Recorded at the code as the difference from the reference algorithm, which R5 orders.
- **EB2′** (X2 #3): e2b's placement function carries each route's `kinds` with the route, though A2's placement
  operand (sharding specs, their mapping onto nodes) names no kind and A8 names "each consumer's kind" as the IR's: in
  e2b's manifest the hand partition and its consumers are one document, and the kinds follow the destinations it cuts.
  A placement function that replaces e2b's gives the kinds with its routes. Recorded as a difference.
- **Numbers across a reset** (X1 #1, open): the sequencer's `seqnum` returns to 0 at its own map move (Q3′), and a new
  sequencer starts at 0, while each node keeps its pending ORDERs across its own move (V1 #4's ruling); so at a node
  whose map has not yet moved, an ORDER numbered after the reset can match a plan of the old serial standing there
  (a replan keeps its name, N1), and the numbers published there go back, which DL3's opposite-order reading does not
  allow for. Every plan of the old serial is cancelled at that node's own move (D.3.6). No handler is added (the
  documents' items 3 and 4); lane_take's note qualifies its premise; open for the operator with "A plan's identity
  across a map move".
- **Noted, not this unit's:** the C9 hook's remark on tools/prepare_e2b.py:1680 (prepare()'s backward reachability
  from the logits over the operator graph, 864a0ae4, 2026-11-03) is on code this unit does not change.
- **DL10** (B4's finding on DL9's fixes: a port that ends a plan it carries, a halt among the causes, raises its
  `taken[p]` to `released` (plan_end), so the parent may publish up to `released` + u - 1 and overwrite words of
  executions below that; where the port's child loads the plan again before this node's run began, `executions` 0,
  link_prepare's EALREADY does not refuse it, and its lanes read words already overwritten). link_prepare refuses
  (EALREADY) a plan it carries units of whose `taken[p]` is above 0 at the load, as it refuses one whose `executions`
  is above 0: a load after this port has taken or been told past numbers that the parent may since have overwritten
  cannot know them, as a load after the run began cannot know which executions landed (the refusal's own reason,
  link_prepare's note). The refusal is the plan's `refused[p]`, which the parent's wait leaves out (DL2).

### Record of this part

#### Commits by owner

- Coordinator (mesh): 932d098d (this part; mesh.h MESH_VERSION 112 and struct mesh_state `taken`, `sequences`,
  `padding2`), 910c355d (INDEX.md), d56df2e8 (DL7, DL8, GO4′), a6875568 (DL2′, DL3′, DL8′, DL9, GO3′, EB2′, numbers
  across a reset, the C9 remark noted), 846bb073 (DL10), and this record.
- C3 (mesh rdma/mesh-graph.{h,c}, mesh-collective.{h,c}, mesh.h's text): e93937d0 (mesh_graph_heap_order declared),
  a8fd8d5f (mesh_graph_heap_order, MSCCLang §5.2 step 3; the binary heap one copy with mesh_graph_lags), c0d0e06f (the
  instructions, their bottom levels and the heap order, GO1-GO4; the entries by their group's least place; T6's and
  S3's order and `start`, `earliest`, `least`, `walk` gone; the sequence words placed, `sequences` written), 0195b9f6
  (mesh-collective.h's text; its message lost three backticked names to the shell, left as pushed), 3d1c4a4e (mesh.h's
  text), d22aa7f6 (DL7, GO4′ cited), 42ae8011 (GO3′; X1 #4, #6; X3 #8; DL9 and DL2′ in mesh.h's text).
- B4 (mesh rdma/mesh-flow.c, mesh-records.c, mesh-link.c, mesh-bridge.h; design/prepared-machine.md's M08 and M29
  rows for X3 #1): 3f6082c1 (`served` read as `taken`), c649d4bf (orders_deliver as Lamport's sender; orders_served
  and `last` gone), 2b853ec7 (lane_take, the k-way merge; the drains over every plan; records naming their flow, DL5;
  taken_tell; plan_sequences_inside; link_run; the notes), db035f40 (DL7, DL8), 6181fbb5 (DL2′, DL9, X1 #1, #3-#6, X3
  #1, #6, #8), 8428e27f (DL10).
- E2 (metal-microbench tools/prepare_e2b.py, tools/e2b_replay.m): 9486578d (EB1: `kinds`, "pages" for the pieces each
  holder's combine sums), 4dc3318c (EB2: kinds to MESH_PAGES and MESH_JOIN, refusals; P1's reversal), bb7f7426 (X2 #2,
  #4, #5, #6; X3 #7).
- A network error (ENOTFOUND) ended the coordinator's run partway; C3 and B4 were resumed from their transcripts on
  their own uncommitted edits (mesh-collective.h, mesh-flow.c), nothing re-dispatched, reset or discarded.

#### Reviews and where their findings went

- X1 (the mesh side): 1 (numbers going back across a sequencer reset) recorded, open ("Numbers across a reset"), and
  lane_take's premise qualified (B4, 6181fbb5); 2 (a gap published as filled) to B4 (DL2′, 6181fbb5); 3, 4, 5, 6 to B4
  and C3 (6181fbb5, 42ae8011); 7 (a lost wake per execution) to B4 and C3 (DL9). Found sound: the heap order, the
  instruction DAG's dependencies, both ends' agreement, contiguous sets, GO4′'s no-cycle argument, Lamport's sender,
  lane_take's readings, the words' single writers and monotony, no deadlock, the records' layouts, the rules.
- X2 (the program and torch side): 1 (e2b's partials are one page each in the 1-row decode, so item 5's permutation
  draws nothing there) open, below; 2, 4, 5, 6 to E2 (bb7f7426); 3 recorded (EB2′). Found sound: `summed` and `kinds`,
  e2b_ir's mapping and refusals, torch's bindings against mesh.h 112 (State through `entries`), the compiles. X2 also
  read three files under .build/ and output_data/ and ran a python3 one-liner that loaded a manifest's JSON, which the
  brief did not allow; nothing was written.
- X3 (across both): every gap closed; 1 to B4 (prepared-machine.md, 6181fbb5); 2 to C3 and this part (GO3′); 3 this
  part (DL3′); 4 to B4 (DL2′); 5 this part (DL8′); 6 and 8 to B4 and C3; 7 to E2 (bb7f7426), open beside EB3. Found
  clean: T6, S3, Q4, O2, Q4′, B2-1, `served` and orders_served only in dated notes and the earlier record; nothing
  added the documents do not name; the rules.
- B4's own findings on its work: the plan standing before mesh_plan raises `plans` (DL7) and the reload after
  plan_end (DL10), each ruled and built.

#### Integration

Once no implementer or reviewer ran (mesh 8428e27f and this record; metal-microbench bb7f7426; both trees clean and
level with origin): `make -B -C rdma` 0 (mesh-flow, mesh-heartbeat, mesh-stat, libmesh.dylib), `make -B -C rdma
native-audit` 0, `make .build/e2b_replay` 0 (libmesh_metal.dylib and e2b_replay against the new libmesh), no warning or
error; rdma/mesh.py, rdma/mesh_compile.py, tools/prepare_e2b.py and tools/mesh/run_pair.py compile (Python's compile,
no bytecode written). Nothing was run.

#### Open at the end, for the operator

- **DL2's head-of-line wait.** The pending head waits for Lamport's buffer of its own plan (u words), and every later
  ORDER at that node waits behind it, a port carrying only a later plan included; the reviews found it Lamport's
  sender's wait composed with Défago's in-order delivery, not Q4's hold, its pace the slowest port of the plan's set.
  The operator's to judge.
- **GO5.** The contention schedule is still computed and its starts order nothing; what reads them is the operator's.
- **Numbers across a reset** (X1 #1): ORDERs numbered after a sequencer's reset can match a plan of the old serial at
  a node whose map has not moved, and the numbers published there go back; with "A plan's identity across a map move".
- **Delivery's liveness**, now also: an ORDER dropped at a node (memory, a port not greeted) leaves its plan's head
  staying there (DL2′), holding every later ORDER behind it.
- **DL9's cost**: each hand-off raises `plans`, so each port walks the heads on its next pass and a lane may take
  nothing for one pass; not measured.
- **Compile time** (C3): mesh_graph_bottom_levels runs mesh_graph_order, O(V²), and V now counts every communication
  (pages x tree edges).
- **Item 5's reach in e2b** (X2 #1): in the 1-row decode each partial is one page, so each set is one page on each
  cable and Algorithm P draws nothing; the permutation shows only where a partial holds two pages or more on a cable.
- **Asynchronous, incomplete, noncontiguous consumers not declared**: e2b's split-KV merge (kv_merge, EB3) and the
  vocabulary's maxloc routes (X3 #7) stay synchronous, complete and contiguous, the item naming e2b's "reductions of partial sums".
- **Knuth vol. 3 §5.4.1 not read**: the merge is the item's statement, the least of the k first elements by
  comparison.
- Carried from the earlier parts: "A plan's identity across a map move"; U1's premise; the torch MPS path (K1); torch
  retirement at exit; L3; V1 #9 (its per-ORDER form, now DL9's raise for the delivery's hand-offs); the torch path's
  refused collectives and operators; the α–β chooser not built (ruling 3); `qps` in mesh-map.h; runs wait for the
  operator's authorization (C3.8).

## The revisions of 67ca86f8

The record of the unit whose instruction is design/plan-to-bridge-change-request.md and
design/aot-compilation-change-request.md as mesh 67ca86f8 holds them ("The document is the whole instruction to an
implementer"; "Where the document is silent, the implementer stops at that point and reports it. It takes no decision
of its own"), from mesh 67ca86f8 and metal-microbench bb7f7426. Its rules are design/disjoint-pages-revision-
coordination.md, "Rules (every sub-agent)", verbatim; that file's ownership table and decisions are not this unit's.
One implementer did the work and dispatched no sub-agent. What the code did not yet do was found by reading both
documents in full and comparing their text at 67ca86f8 with edd4b583's, which the code implemented ("The documents'
revisions of edd4b583", above), and with the code. This part takes no decision: each clause below is done as the
document states it, found already done, or stopped, with the document's words at that point.

### Done

- **SP. A set's pages keep their places** (plan-to-bridge R5): "A set's pages (design/aot-compilation-change-request.md,
  appendix item 5) keep their places in that order. The set's random permutation is drawn over those places.
  (Corrected 2026-11-05: the first text moved a set's pages to consecutive places, which nothing requires.)"
  - rdma/mesh-collective.c plan_schedule: GO4's groups go (struct plan_entry `at`, plan_entry_side, the least-place
    pass over mesh_graph_rows); every entry, a set's among them, is sorted by its own place (plan_entry_order), so each
    node pair's list at each end is a subsequence of the one global order, both ends' alike, every copy's the same.
  - rdma/mesh-flow.c link_prepare: a set is the picks of one copy and direction with the same nonzero `set`, wherever
    they lie in the list, its members in list order (where it was a maximal run of consecutive picks, S3), and the
    EINVAL for a `set` in two runs of one copy (S7) goes; the permutation stays per cable, over the positions the deal
    gave it (S7′). position_record: a lane that meets one set again after another in an execution draws it again, the
    same permutation.
  - rdma/mesh-collective.h and rdma/mesh.h: the notes, each reversal dated. MESH_VERSION stays 112 (no layout moves).
  - GO4's and GO4′'s argument (the groups in order of their least places a topological order with each group whole,
    so any permutation within a group one too) is withdrawn at each note; see stop 3.
- **EK. Every program declares each consumer's kind** (aot-compilation, appendix item 5): "Every program declares each
  consumer's kind. A consumer that reduces partial sums, or otherwise takes pages in any order, is an asynchronous, incomplete, noncontiguous consumer (R4)."
  - tools/prepare_e2b.py: prepare()'s set `summed` is `reduced`, the (partial, rank) pairs whose consumer at that rank
    reduces the partial with the other ranks': the pieces a combine sums (EB1, as before); kv_merge's split-KV partials
    and normalizers at each rank that merges them (R4's example, "online softmax (M. Milakov and N. Gimelshein ...)":
    kv_merge sums exp(lse_r - M) o_r and exp(lse_r - M) over the ranks); and each vocabulary contribution at every rank,
    since every rank runs maxloc_reduce, the max of the ranks' (value, index) pairs (the turns: "elementwise maps, and
    sum or max reductions"). Each such destination's kind is "pages"; EB3's and review X3 #7's "join" for them is
    reversed with dated notes. tools/e2b_replay.m: e2b_ir's comment on "pages". A directory prepared before keeps its
    manifest's kinds until it is prepared again.

### Found already done (the code at 67ca86f8 and bb7f7426 does what the document states)

- R4's correction: "Ordering coflows to minimize their completion time is Varys's smallest-effective-bottleneck first
  ... R5 does not require it." No code implements or names Varys; a join's pages carry its consumer's bottom level
  (GO2).
- R4: "The synchronous, complete, contiguous consumer's contiguity is placement: the allocator gives each value's copy one run of pages, as greedy-by-size offset
  assignment gives each tensor one contiguous range". mesh_plan's items are each value's copies, each coloured by
  mesh_graph_left_edge to one run [colour, colour + size) (plan_place).
- A4: "AllGather is one broadcast from each source over its packed trees (Blink §3.1). AllToAll is point-to-point
  demands, one per ordered pair of ranks, routed jointly as concurrent multicommodity flow". rdma/mesh_compile.py
  _all_gather declares each member's shard one range to every other member, which mesh_plan routes as one demand over
  its trees; _all_to_all declares one transfer an ordered pair; mesh_plan routes every demand at once by mesh_graph_flow
  (Garg and Könemann's rounds).
- Item 6: "name each compiled function alike at every rank: (the program's group, the function's ordinal in the
  program's order of compilation)". mesh_compile.py takes the next of FUNCTIONS (1, 2, ...) at each call of the
  backend's compile function; e2b plans one function, 1; the plan is (group, function) (N1).
- The scope, A1's, A9's and R8's new wording, and item 5's "For the program the code runs today, tools/prepare_e2b.py
  does the declaring": wording, no code.

### Where the work stopped

- **Stop 1. Item 2, "Delivery at a node".** The item:
  > A program's calls are a sequence, so each plan has at most one call outstanding. The sequencer numbers each call's
  > submission (Défago, Schiper and Urbán, 2004), and a call returns only once its executions are done (C3.1).
  >
  > The bridge parent writes, for each plan, its one outstanding call's sequence number and executions into words of
  > that plan's state, each with a single writer (C3's rule 2). No word is reused while its call is outstanding, so
  > nothing waits to write.
  >
  > Each port serves the outstanding calls of every standing plan in sequence-number order. Both ends of a cable read
  > the same numbers, so they serve alike.

  Where it is silent:
  - **When a node's words of a plan's next call are written.** The parent writes a call's words when the sequencer's
    ORDER of it reaches its node. The ORDER of call k + 1 follows the submission at the plan's sender (the least of the
    plan's nodes, Q2), whose program makes it once its own call k returned, and item 4 states what that means: "A call
    returns once its own landings and sends are done (C3.1). So at each end every page of an execution is matched".
    At another node of the plan, call k can still be outstanding then: its program not yet at call k, or a lane there
    still to post a forward of call k to a third node. The item says no word is reused while its call is outstanding
    and nothing waits to write; it does not say what the parent does with an ORDER of a plan's next call that reaches a
    node where the plan's call is outstanding: store its words over those a lane still reads (the lane's two ends would
    then read different numbers), or keep the ORDER pending (a wait, every later ORDER there behind it in sequence
    order), and if so until what (this node's program's next submission, or every lane there having taken call k).
  - **What a call is where one start submits several executions.** Item 6: "have each call launch one execution of its
    function"; this item: "its one outstanding call's sequence number and executions". e2b's one start raises
    `executions` to its steps at once (N4): one call of many executions, or many calls outstanding at once, which "each
    plan has at most one call outstanding" excludes.

  Nothing changed: the code is as edd4b583's item left it (Lamport's buffer: u sequence words a plan, `released` its
  ms, each port's `taken[p]` its mr, the parent's wait where the buffer is full, DL10's refusal on `taken[p]`), which
  the correction names "a ring whose writer waits when it is full". The item's other "The code must" lines were done
  before: orders delivered in sequence order with no timeout and no skip (O1), released executions interleaved on each
  cable (DL3), Q4's hold and Q4′'s skip gone (DL4, O1).
- **Stop 2. Edge scheduling.** R5's method: "Bottom levels count each edge's weight over its route's cables, as Sinnen
  and Sousa's eq. (7) does ... (Corrected 2026-11-05: the first text also required their edge scheduling, whose start
  times order nothing once the lists follow the reference algorithm below.)" R5's "Code": "rdma/mesh-graph.{h,c}: bottom
  levels, and the list schedule with edge scheduling." A6: "List scheduling by bottom level, with edge scheduling under
  contention, over the weights of A3." The method no longer requires edge scheduling; R5's "Code" line and A6 still
  name it. Nothing changed: plan_schedule still calls mesh_graph_contention_schedule and nothing reads its starts
  (GO5). The bottom levels are as the method states (GO2: a communication on link L weighs c(e) / b(L), so a remote
  edge's weight counts over its route's cables).
- **Stop 3. No cycle among the permuted sequences.** R5: "A set's pages ... keep their places in that order. The set's
  random permutation is drawn over those places"; and "Each cable carries execution t's sends in that order, then
  execution t + 1's. That order is a topological order of the unrolled executions, since every lag points forward. So
  the cables' sequences cannot wait on each other in a cycle". SP builds the first as written. The second holds for the
  places' order; each execution's permutation (item 5, per cable, S7′) changes which page a set's place carries, and
  the documents state no argument for the permuted sequences. GO4's groups made every permutation within a group a
  topological order too. With a set's places spread, that holds where every edge of the set's tree draws the same
  permutation over equally many members (S7′'s counter: one execution, set and cable place), and not otherwise. For
  example: nodes x, y, z; a set of pages a and b on the tree x -> y -> z at places 10 and 30 of (x, y), dealt to its
  second cable, and 20 and 40 of (y, z), whose one cable is at place 0; a demand from y over y -> z -> x at 21 and 22;
  one from z over z -> x -> y at 26 and 28, on (x, y)'s second cable (each lane in order, the places a topological
  order). Where (x, y) draws a at 10 and b at 30 and (y, z) draws b at 20, y's send at 20 waits for b's landing at 30,
  which is behind x's forward at 28, which waits for the landing at 26, behind z's forward at 22, which waits for the
  landing at 21, behind y's send at 20. No argument and no mechanism was added.

### Commits

- mesh cf5842fa (rdma/mesh-collective.{c,h}, rdma/mesh.h, rdma/mesh-flow.c: SP), and this record.
- metal-microbench 43e6302f (tools/prepare_e2b.py, tools/e2b_replay.m: EK).
- The rules' C9 hook did not run: mesh has none installed (.git/hooks holds only pre-commit.countermanded-20260922,
  and core.hooksPath is unset), so no remark was given.

### Integration

At mesh cf5842fa and metal-microbench 43e6302f, both trees clean and level with origin: `make -B -C rdma` 0 (mesh-flow,
mesh-heartbeat, mesh-stat, libmesh.dylib), `make -B -C rdma native-audit` 0, `make .build/e2b_replay` 0
(libmesh_metal.dylib and e2b_replay against the new libmesh), no warning or error; each changed C file compiled alone
first, and tools/prepare_e2b.py compiles (py_compile; its __pycache__ entries were there before). Nothing was run.

### What this changes among the earlier parts' open items

- "No cycle on a cable" (W3 #6), which GO4 answered, is open again: stop 3.
- "Item 5's reach in e2b" and "Asynchronous, incomplete, noncontiguous consumers not declared" (EB3, X3 #7): kv_merge's partials and normalizers and the
  vocabulary's contributions have asynchronous, incomplete, noncontiguous consumers (EK).
- "GO5" stands, with stop 2; "DL2's head-of-line wait" stands, with stop 1.

## The revisions of 46ae8fdf

The record of the unit whose instruction is design/plan-to-bridge-change-request.md and
design/aot-compilation-change-request.md as mesh 46ae8fdf holds them, from mesh 46ae8fdf and metal-microbench 43e6302f,
with two clauses as mesh 0e56d8da revises them on the operator's turn that the turns record now ends with
(plan-to-bridge R5, "Method" and "Code"; aot-compilation appendix item 2, "Delivery at a node", with A6 and A10's
output). Its rules are design/disjoint-pages-revision-coordination.md, "Rules (every sub-agent)", verbatim, but its
reading order and its file table. The documents now allow an implementer's minimal decision where they are silent,
recorded as its own ("How this document is written and used"); each such decision below is marked as the
implementer's (M1 to M7 the coordinating implementer's, LP1 to LP11 sub-agent L's, TC1 to TC6 sub-agent T's) and is
not a requirement of the documents. While the work ran, the coordinator held R5's priority and item 2's delivery for
revision, then gave their text at 0e56d8da; no code had been changed for either before the hold, and none was reverted.
Two sub-agents, each its files' one owner: L (rdma/mesh-graph.{h,c}, rdma/mesh-collective.{c,h}, rdma/Makefile) and T
(rdma/mesh_compile.py); the rest by the coordinating implementer. Nothing was run.

### Done

- **OP. One permutation of the whole set per execution** (aot-compilation, appendix item 5: "The counter is
  (execution, set), so a set is drawn once per execution. Every edge and every cable of the set's tree carries its share
  of the set's pages in that one permutation's order"; "A draw for each cable departs from the counter"). rdma/mesh-flow.c:
  set_draw draws Knuth's Algorithm P over the set's every position on the pair in the direction (M, in list order),
  counted by (execution, set, block, 0), with no cable's place; position_record gives each cable the members the deal
  gave it in that permutation's order (M1); struct flow_set `size` M, `local`, `ranks`; struct flow `place` gone; S7′'s
  per-cable stream reversed with dated notes (mesh-flow.c, mesh.h, mesh-collective.{c,h}, and metal-microbench
  tools/e2b_replay.m's comment).
- **LR. Lemire's rejection step** (item 5: "With Lemire's rejection step, which the code must add, the integer is
  exactly uniform"). set_draw takes each of Algorithm P's integers by D. Lemire, ACM TOMACS 29(1), 2019, section 4,
  Algorithm 5, with L = 64 and s = j, each x the stream's next word, where it took the product's high word alone. The
  algorithm's lines were read from arXiv 1805.10941v4 (its HTML), not from the journal's text.
- **LP. R5's priority as one linear program** (R5 at 0e56d8da), sub-agent L. rdma/mesh-graph.{h,c}
  mesh_graph_completion_times: Chen 2022's LP (14) with (14a), (1b) as each C_j's lower bound d_j, (1c) for each
  dependency, and Hall, Schulz, Shmoys and Wein's row (5) on each resource, by cutting planes with Queyranne's
  separation (each resource's members sorted by the current solution, the nested prefixes tested), each LP by HiGHS
  1.15.1's dual simplex through its C API; it builds rows and calls the library. rdma/mesh-collective.c plan_schedule:
  the program over GO1's instructions and dependencies, lengths w for a task and GO2's greatest c(e)/b(L) for a
  communication, resources each node's device and each T2 directed link (LP1); mesh_graph_heap_order by -C̄, the least
  C̄ first, a tie to the lower instruction (LP7). Edge scheduling's start times and the bottom levels are no longer
  computed for the order, and mesh_graph_contention_schedule and mesh_graph_bottom_levels, with their helpers, are
  removed (nothing else in either repo called them). rdma/Makefile links HiGHS (LP8). The notes quoting R5's earlier
  text carry dated notes (LP), mesh.h's "Both ends" among them.
- **CH. Delivery as the fixed sequencer's destinations deliver, through a FIFO channel** (item 2 at 0e56d8da: "Each
  node runs the destinations' code of their Fig. 5, as written"; "Delivery does not wait for any earlier call to
  finish"; "The delivered calls reach the ports, in delivery order, through a first-in, first-out channel whose writer
  never waits"; "Each port reads the channel in order"; "An entry is reclaimed once every port has read past it, by
  deferred reclamation"; "The code must ... publish them to each port through the channel above, in place of the
  sequence words (`sequences`), `taken[p]` and the parent's wait while they are full").
  - rdma/mesh-records.c orders_deliver: the pending head (arrival order is sequence order, as before) is delivered at
    once, its plan's name and executions appended to the deliveries' channel, then hdr `delivered` raised (release),
    and hdr `plans` raised and woken after a turn that delivered. The match to a standing plan, the port set, the
    sequence words, `released`, Lamport's sender's wait and their log lines are gone.
  - rdma/mesh.h, rdma/mesh-map.h, rdma/mesh-link.c: struct mesh_delivery, MESH_DELIVERY_ROOM, hdr `deliveries_off`,
    `deliveries_room`, `delivered` and `delivery_read[p]` appended, the area laid out after the map's records and
    compared by region_same; MESH_VERSION 113; struct mesh_state `released`, `taken` and `sequences` read and written
    no more, kept by name in place (M3).
  - rdma/mesh-flow.c: deliveries_copy (each port's child copies the channel into its own memory and tells
    `delivery_read[p]`); lane_take reads that copy in order (M5), every served lane reading it; taken_store,
    taken_tell, `told`, plan_end's raise of `taken[p]`, DL10's refusal on `taken[p]` and plan_sequences_inside gone.
  - rdma/mesh-collective.c (sub-agent L, LP9): plan_layout places no sequence words; mesh_plan writes `sequences` 0.
- **SN. The sequencer's number never goes back to 0** (item 2: "remove the coordination's Q3′: the sequencer's number
  never goes back to 0"). mesh-records.c island_set's `seqnum=0` gone; `seqnum` from 0 at the bridge's start only.
- **SQ. The sequencer's citation** (A7: "(Corrected 2026-11-05: the text called this the bully algorithm's coordinator
  ... No election runs, so that algorithm is not used.)"). island_set's note cites Défago, Schiper and Urbán §7.1, "The
  sequencer is unique, and the responsibility is not normally transfered to another processes" (checked in their
  text), and A7's "the code's choice, which the fixed-sequencer class leaves open", where it cited Garcia-Molina.
- **RS, CP. ReduceScatter and CollectivePermute lowered** (A4: "rdma/mesh_compile.py refuses CollectivePermute and
  ReduceScatter now; this requirement covers them"), sub-agent T. rdma/mesh_compile.py _reduce_scatter (GSPMD §4.2;
  each other member's slice one MESH_PAGES transfer to its keeper, R4's "reductions' partial sums", the keeper's
  receive-reduce-copy by torch.add, the first half of Blink §3.3's AllReduce) and _collective_permute (GSPMD §4.2,
  one point-to-point transfer a pair, routed as AllToAll's), each in LOWERINGS and _run; the "refused" comment fixed.
  The partitioner inserts neither collective (code fact: _reshard, "no CollectivePermute arises"; _partition replicates
  a contracting dimension the output's mesh dimension holds, where GSPMD §5 "replace[s] the AllReduce on the output BSM
  with a ReduceScatter"), so both are reached only once A2's partitioner inserts them; no clause asks that change.
- **R4's homomorphism cited as R4 cites it**, sub-agent T: mesh_compile.py's _consumer and _all_reduce cite Gibbons for
  associativity and Meertens's bag homomorphisms (CWI Monographs 1, 1986) for commutativity.

### Found already done (the code at 46ae8fdf and 43e6302f does what the document states)

- Item 2: "no node drops a pending order when its map changes": the pending ORDERs are kept across this node's map
  move (the ruling on review V1's finding 4). Q4's hold and Q4′'s skip are gone (DL4, O1), as the item says.
- Item 2: "The sequencer numbers each call ... A call is one execution of its compiled function": order_sequence gives
  a SUBMIT of [first, end) end - first numbers, one an execution; mesh_compile.py's call raises `executions` by one
  (N4). For "A program submits one call per execution", see M2.
- A7: "Each function execution is submitted once, by its plan's least node (`sender`)": orders_submit at the plan's
  `sender` (Q2). A9: "Nothing else is serialized but each entry's set mark": struct mesh_hop `set`.
- A2: GSPMD "produces a single program for all partitions": mesh_compile.py says nothing of one program a device.
- A10's "the order constraints between calls, which the program keeps when it submits": every program here submits
  its calls in program order (e2b_replay's start, mesh_compile.py's call), which keeps every such constraint.
- Item 6's JAX and CUDA wording, item 7 (K1) and R8's attach (mesh-dataflow.c, claiming nothing): as now.
- The appendix "a chooser for edge weights" is supplementary ("Using it means passing it in place of unit weights"):
  no clause requires it; the unit weights stay (P2).

### Decisions taken where the documents are silent (each the implementer's, with its reason)

- **M1 (item 5).** A cable's "share" of a set is the members whose positions the deal gave that cable, and each
  member's identity is its index among the set's positions on the pair in list order (the bridge's X_1 .. X_M, S5).
  Reason: each port's child holds records for its own cable's members only (link_prepare), the documents name no way
  for a member to be posted by a child that holds no record of it, and the hop names no index of a page within its
  set. Its consequence for item 5's argument is stop 3 below.
- **M2 (item 2).** A start that raises a plan's `executions` by k submits k calls, each one execution, each numbered
  on its own; e2b's one start of its steps (N4) is unchanged. Reason: the item defines a call as one execution and the
  sequencer already numbers each execution; how a program batches its submissions is stated nowhere.
- **M3 (item 2, "How the bridge holds the channel is the bridge's own").** The channel is an area of the region laid
  out with it, as the map's records are (disjoint-pages Appendix A, A3, "The bridge's own pages ... are laid out once
  with the region", which stands), of MESH_DELIVERY_ROOM = 4096 entries of 32 bytes (a plan's name and executions
  [first, end): M6), the k-th delivery in entry k mod 4096. A port's child begins reading at `delivered` as its first
  pass reads it. Reason: port children are spawned processes that share only the region (mesh-children.c
  posix_spawn, mesh-child.c shm_open); a plan a new child can serve stands on a map serial that its own pairing moved,
  so every delivery of it comes after the child's start.
- **M4 (item 2).** An entry is reused only once every port whose link is up with a queue pair in its plan (the test by
  which mesh-ports.c record_reach lists a cable) has read past its last delivery; where one has not, the ORDER stays at
  the head until a later turn, said once on the bridge's log. Reason: a finite area cannot hold more undelivered
  entries than its room, and dropping a delivery at one end of a cable and not the other would break "the k-th SEND on
  a queue pair fills the k-th receive" for every later call on that lane. This is a wait at the writer, which the
  clause's "whose writer never waits" does not allow; it arises only where a port's child stops reading for 4096
  deliveries, since each child copies every delivery into its own memory on each pass (M5).
- **M5 (item 2).** Each port's child copies the channel into its own memory as it reads it (realloc by doubling, the
  entries before every served lane's place let go), and each served lane reads that copy in order: a delivery of a
  loaded plan it carries is taken at its cursor's execution n where first <= n < end and n is submitted here (D2),
  passed where n >= end, and waited at where n < first (a delivery this node never made, order_take's ENOMEM); a
  delivery of a plan it does not carry, of a cancelled plan, or of one standing here with this port's refusal is
  passed; one of a plan not loaded otherwise is waited at. Reason: "Each port reads the channel in order", and a lane
  that passed a delivery its peer takes would break the k-th SEND's match.
- **M6 (item 2).** An entry holds the plan's name and [first, end) alone, no sequence number. Reason: the channel's
  order is the delivery order, so a port needs no number to take the calls in sequence-number order.
- **M7 (R3).** "V are tasks, each one operation of the captured graph executed at its node" is read as met by the
  template's events (T1: a production, a source read, a consumption, each the declared operation at its node).
  Reason: each task the code builds is one such operation's event; no clause asks every operation to be a task.
- **LP1 to LP11 (sub-agent L, R5).** LP1: a task's resource its node's device (by map index), a communication's its
  T2 directed link. LP2: each round adds each resource's most violated prefix alone. LP3: a prefix is violated only
  where it falls short by more than HiGHS's `primal_feasibility_tolerance` x (1 + its right side), so no row is added
  twice and the loop ends; no round bound. LP4: each dependency one row (duplicates from shared route prefixes, T4,
  skipped). LP5: the separation's sort ties to the lower vertex; zero-length vertices left out of the cuts. LP6: HiGHS
  options `solver` simplex, `simplex_strategy` 1, `parallel` off, `threads` 1, `output_flag` false, the rest its
  defaults; rows added to the solved model. LP7: the least C̄ first as priority -C̄ to the unchanged heap. LP8: the
  header included as <interfaces/highs_c_api.h> under -isystem $(HIGHS)/include/highs (HiGHS's pkg-config Cflags), since
  under -I/opt/homebrew/include its own includes do not resolve and its declared, undefined static
  Highs_compilationDate fails -Werror. LP9: the sequence words gone from the layout, `sequences` 0. LP10: EINVAL for a
  self-edge, a bad length or resource, or an infeasible program (a cycle of positive length); EDOM for HiGHS's error or
  any other status; ENOMEM past HiGHS's 32-bit indices. LP11: mesh_plan's "every weight > 0" test kept.
- **TC1 to TC6 (sub-agent T, A4).** TC1: a ReduceScatter's `dim` the tensor dimension it scatters, slice q to the
  member at place q in the device mesh's order. TC2: a CollectivePermute's `dim` its (source place, target place)
  pairs within the group, as XLA's source_target_pairs. TC3: a CollectivePermute's transfers MESH_JOIN (their consumer
  a clone, a synchronous, complete, contiguous consumer under P3). TC4: the permuted result torch.clone of what arrives, a tensor of its own, over no
  placed page. TC5: a pair p -> p keeps the member's value with no transfer; a member no pair targets holds zeros, as
  XLA's operation semantics state (checked on OpenXLA's page); the keeper adds its own slice first, then the landed
  slices in group order. TC6: the R4 citations above.

### Where the work stopped

- **Stop 1. A10's analysis.** A10: "The compilation must take a program's sequence of calls as a graph, decide at
  compile time which calls may be outstanding at once and in what orders, and compile that into the plans"; "The
  calls' graph. It is the program's captured trace, as A1 captures one function: TorchDynamo's FX graph of the calls
  and the tensors passing between them"; its output "for each compiled function, the most calls of it the analysis
  lets be outstanding at once in the program that submits them, which bounds how many the program issues ahead".
  Silent on: how the calls of compiled functions become an FX graph (the backend is called with each function's own
  graph, PyTorch 2 §3.1, not the program's calls; and for a program on another framework's delegate A10 names no
  capture); and what finite number the analysis gives where dependent calls are issued ahead as futures (Pathways §4.5,
  which A10 cites): Bernstein's conditions and commutativity analysis order such calls, and bound nothing. Nothing was
  built for the bound; the order constraints are found done above.
- **Stop 2. The asynchronous, incomplete, noncontiguous consumer's output pages.** R4: "The asynchronous, incomplete, noncontiguous consumer is one task per page, each with an edge from its own page, and its
  outputs are pages that leave as each is produced." The code gives each demand one production task, P(d), and a range
  one supplied word, which its producer raises after its last write in an execution (C3.7: "One word serves every
  execution"), so a range's pages leave together. A page leaving as it is produced needs a release of each page, which
  neither document nor C3 states. Not changed.
- **Stop 3 (narrowed). No cycle among the permuted sequences where the deal splits a set.** Item 5: "A set's pages
  have one producer and one route (above), so they share their predecessors, and each cable's sequence is the same
  renaming of them. So MSCCLang §5.2's argument against cycles holds." With OP and M1, a cable that carries every
  position of a set on its pair carries the one renaming, so the argument holds where one cable carries every position
  of the set on each pair of its tree. Where the deal splits a set's positions over a pair's cables, each cable carries
  the permutation restricted to its members, which is not one renaming. For example: a set of pages a and b on the tree
  x -> y -> z, at places 10 (cable A of (x, y)) and 30 (cable B of (x, y)), and 20 and 40 of (y, z), one cable; a demand
  from y over y -> z -> x at 21 and 22; one from z over z -> x -> y at 26 and 28, on (x, y)'s cable B. Where the
  permutation is (b, a), (y, z) carries b at 20 and a at 40, while each cable of (x, y) carries its one member; y's send
  at 20 waits for b's landing at 30, behind x's forward at 28, which waits for the landing at 26, behind z's forward at
  22, which waits for the landing at 21, behind y's send at 20. The one renaming on every cable would move a member to a
  cable whose child holds no record of it (M1), which the documents name no mechanism for. No argument and no
  mechanism was added.

### Commits

- mesh c2a226b0 (rdma/mesh-flow.c, rdma/mesh.h, rdma/mesh-collective.{c,h}: OP, LR).
- mesh 7430505c (rdma/mesh_compile.py: RS, CP, R4's citations; sub-agent T).
- mesh f4838c9d (rdma/mesh-graph.{h,c}, rdma/mesh-collective.{c,h}, rdma/Makefile: LP, LP9; sub-agent L).
- mesh 71e060ba (rdma/mesh.h, rdma/mesh-map.h, rdma/mesh-link.c, rdma/mesh-records.c, rdma/mesh-flow.c,
  rdma/mesh-bridge.h, rdma/mesh-dataflow.c: CH, SN, SQ).
- metal-microbench 4b0d2da2 (tools/e2b_replay.m: OP's citation in a comment).
- This record.
- The rules' C9 hook did not run: mesh has none installed.

### Integration

At mesh 71e060ba and metal-microbench 43e6302f, both trees clean but for untracked build artifacts and level with
origin: `make -B -C rdma` 0 (mesh-flow, mesh-heartbeat, mesh-stat and libmesh.dylib, the bridge image and libmesh
linking HiGHS), `make -B -C rdma native-audit` 0, `make .build/e2b_replay` 0, no warning or error. Each changed C file
compiled alone first, the lifecycle files unchanged and compiling against the new mesh.h, and rdma/mesh_compile.py
compiles (py_compile). After metal-microbench 4b0d2da2 (a comment), `make .build/e2b_replay` 0 again. Nothing was run.

### What this changes among the earlier parts' open items

- "The revisions of 67ca86f8", stop 1 (delivery at a node) and stop 2 (edge scheduling): answered by 0e56d8da's text
  and done (CH, LP); GO5's unread starts are gone.
- Stop 3 of that part: narrowed to sets the deal splits over a pair's cables (stop 3 above).
- "DL2's head-of-line wait", Lamport's sender's wait (DL1, DL2) and DL10: gone with the sequence words (CH); M4 is the
  one wait left at the writer.
- "Numbers across a reset" (review X1 #1): gone with the reset (SN).
- "Replans reuse names": a replanned function keeps its name (N1), and a lane matches a delivery to a loaded plan by
  name (M5), so a delivery made for a cancelled plan's executions that a lane meets after the cancelled plan has ended
  and its replan is loaded is read as the replan's. It arises only after a map move cancels a call in flight; no
  handler is added.

## The revisions of f3f7cf46

The record of the unit whose instruction is design/plan-to-bridge-change-request.md and
design/aot-compilation-change-request.md as mesh f3f7cf46 holds them, from mesh f3f7cf46 and metal-microbench 4b0d2da2.
Its rules are design/disjoint-pages-revision-coordination.md, "Rules (every sub-agent)", verbatim, but its reading order
(the two documents in full, then mesh AGENTS.md, design/rubric-actionable.md, metal-microbench AGENTS.md,
docs/principles.md, docs/measurement.md) and its file table. What the code did not yet do was found by reading both
documents in full against the code, beginning with the clauses revised after "The revisions of 46ae8fdf": plan-to-bridge
R4 (the asynchronous, incomplete, noncontiguous consumer's outputs) and R5 (the second objective; HiGHS from its pinned source), and the AOT document's A4, A10
and appendix items 2 and 5. That part's M1 (a cable's share of a set), M4 (the writer's wait) and M5 (each port's copy
of the channel) are overturned by name in the documents and are reversed below. Each decision taken where the documents
are silent is marked as the implementer's (HB, LX, SW, CH2, PM the coordinating implementer's; TA and TT sub-agent T3's)
and is not a requirement of the documents. One sub-agent, T3, was the one owner of rdma/mesh_compile.py; the rest is the
coordinating implementer's. Nothing was run but the compilers, the Makefile's own fetch of the pinned HiGHS source and
the three integration builds.

### Done

- **HB. HiGHS from its pinned source, linked statically** (plan-to-bridge R5, Code: "rdma/Makefile builds HiGHS from its
  source at a pinned release and links it statically. The release is tag v1.15.1, commit
  04024d701f79feb8e2f18bc3df0dffc04ef05088, fetched as https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.15.1.tar.gz
  and checked against SHA-256 a840d269dff2fafb371dd247df13ad5e026d7ce3b35ad3dc1eedd59bf0c2fb16"; "It is built with the
  tools every member has, cc and c++"; "It takes the same `-ffp-contract=off` as mesh-graph.c and no fast-math"; "The C
  API is highs/interfaces/highs_c_api.h of that source"; "Nothing is installed by hand, and nothing comes from
  Homebrew").
  - rdma/Makefile: `$(HIGHS_TARBALL)` fetched by curl --fail from that URL to a .part file, checked by `shasum -a 256
    --check` against HIGHS_SHA256, then moved, as metal-microbench's Makefile fetches `$(E2B_REFERENCE_MODEL)`
    (E2B_REFERENCE_REVISION with E2B_REFERENCE_SHA256). The tarball's pax header names commit
    04024d701f79feb8e2f18bc3df0dffc04ef05088 (read once from the downloaded copy, `comment=` in its global header); the
    SHA-256 is the check.
  - `$(HIGHS_BUILT)`: the tarball extracted under rdma/.build/highs, highs/HConfig.h.bazel.in copied to the source's
    root as HConfig.h, and every source of the BUILD.bazel target "highs" (its srcs globs, HIGHS_SOURCES) compiled by
    cc (.c) or c++ (the rest) with that target's copts (-fexceptions -Wno-unused-variable
    -Wno-unused-but-set-variable), its default define (CUPDLP_CPU), its includes (extern, extern/catch, highs, and the
    root where HConfig.h is), -O2 and -ffp-contract=off as mesh-graph.c (HB1, HB3); 216 objects, no warning.
  - mesh-flow and libmesh.dylib depend on it and link the objects with -lc++ and BUILD.bazel's linkopts -lpthread (HB2);
    mesh-graph.c includes <interfaces/highs_c_api.h> under -isystem of the source's highs directory and its root (LP8's
    reason stands: the header declares a static function it never defines). The Homebrew paths (HIGHS ?= /opt/homebrew,
    -lhighs) are gone; otool -L of libmesh.dylib names no HiGHS library.
  - A first form put the two rules before `all`, which made the tarball the default goal (`make -B -C rdma` fetched it
    and built nothing else); the rules now follow `all`.
- **LX. R5's second objective** (plan-to-bridge R5, "How: the program": "Then, over (14)'s optimal solutions, minimize
  Σ_j C_j: Chen's (1) and Hall et al.'s (1) with unit weights. The two are one lexicographic linear program (H. Isermann,
  Linear lexicographic optimization, OR Spektrum 4, 1982). They are solved in sequence, the second with C_max held at the
  first's optimum"). rdma/mesh-graph.c mesh_graph_completion_times: where the first objective's cutting-plane rounds end
  (no row violated), C_max's lower and upper bounds are both its value in that optimal solution
  (Highs_changeColBounds), its cost 0 and every C_j's cost 1 (Highs_changeColCost, Highs_changeColsCostByRange), and the
  rounds go on with the same rows, the same separation and the same termination test (LP2, LP3) until no row is violated
  again; completion[j] is the second objective's last solution (LX1). mesh-graph.h's declaration says so.
- **SW. The start's wait and the late-loading port's refusal removed** (aot-compilation, appendix item 2, "The code
  must: ... remove the start's wait for every port to load the plan, and the late-loading port's refusal of it
  (`mesh_ports_held`). Nothing requires them. "a send that arrives first is held until its receive is posted (§3), and a
  loss cancels the plan anyway""). rdma/mesh-dataflow.c: mesh_ports_held gone, and mesh_ports_named, its last caller's
  port set (SW1); mesh_transfers_start raises `executions`, raises and wakes `plans`, fences and reads the map's serial,
  ESTALE after the raise as before, with no wait before it. rdma/mesh-flow.c link_prepare: the EALREADY refusal of a
  plan this port carries units of whose `executions` read above 0 at the load is gone (SW2). rdma/mesh-map.h,
  rdma/mesh-call.h, rdma/mesh.h: dated notes; the start's error set is now 0, EINVAL or ESTALE.
- **CH2. The deliveries' channel read in place, with no copy and no wait** (aot-compilation, appendix item 2: "The
  delivered calls reach the ports, in delivery order, through a channel laid out once with the region ... Each port reads
  it in place, by view, and nothing copies it. Only the driver moves bytes"; "The writer never waits and never checks for
  a full channel"; "The code must: ... publish them to each port through the channel above, read in place, ... with no
  copy and no wait (M4 and M5 go)").
  - rdma/mesh-flow.c: each lane's place `at[d][q]` is a delivery's index, read in place as entry `at` mod hdr
    `deliveries_room` of the region's channel below hdr `delivered` (acquire, once a lane_take); deliveries_copy (M5:
    the copy into the child's memory, grown by doubling, and `delivery_read[p]` told after it) is deliveries_join, the
    child's first-pass place at `delivered` alone (M3's join stands); struct port_plans `orders`, `base`, `copied` and
    `room` gone.
  - rdma/mesh-records.c orders_deliver: entry k mod `deliveries_room` written for delivery k with nothing read before
    it; M4's test of every reading port's `delivery_read[p]`, the head staying while one trailed by the room, and its
    log line (`full`) gone.
  - rdma/mesh.h: hdr `delivery_read[p]` read and written no more, kept by name (CH2-1); notes dated. The room a plan
    claims is not built (stop 1); the room is MESH_DELIVERY_ROOM, as M3 laid it out (CH2-2).
- **PM. The deal fixes positions, not members** (aot-compilation, appendix item 5: "So the deal fixes positions, not
  members. Each lane keeps the positions the deal gives it. In each execution, the member at each position is the one σ
  puts there. Each port holds a record for every member of every set it carries a position of, each record a view of
  that member's page (R1). So any member can be posted at any of the set's positions on its cables"; "(Corrected
  2026-11-05: ... each cable keeps only its own members (the implementation's M1) ...)").
  - rdma/mesh-flow.c link_prepare: for each set with a position on this cable, a record of each of its M members on the
    pair, those whose own position the deal gave another cable built from their own hops after every lane's records
    (PM1); the frames test (EMSGSIZE) over every member against every position's line on this cable; `members[d]` the
    set's M records in list order, `slot_position` a position's index j among the set's M positions; struct flow_set's
    `local` and `ranks`, and `ranks[d]`, gone (M1 reversed).
  - position_record: position j posts member x[j] of set_draw's one permutation of the M positions, drawn per (execution,
    set) as before (OP, LR) (PM4).
  - The receive's round rule (design/disjoint-pages-change-request.md D.3.5, "named once per landing"): a record of round
    j is posted only where its page's phase field reads mesh_phase(j) (acquire), its round j - 1 landed through
    whichever port's child, where it read the record's own `landings`; the round is stored in the record as it is posted
    (`round`, in `landings`'s 8 bytes) and its landing flips bit field + (j & 1) and wakes by it (PM2).
  - plan_end's cancellation: each receive's count raised to its final value where it reads below it (count_raise), where
    it was raised where the record's own landings were below its copy's executions (PM3).
  - rdma/mesh.h (struct mesh_hop's notes, "Both ends") and metal-microbench tools/e2b_replay.m (the sets' comment):
    dated notes.

- **TA. ReduceScatter inserted where GSPMD inserts it** (aot-compilation A4: "A2's partitioner inserts these two where
  GSPMD's does. ReduceScatter goes where a reduction's result is sharded: "gradients will be communicated with
  ReduceScatter instead of AllReduce" (GSPMD §3.2)"), sub-agent T3. rdma/mesh_compile.py _partition: a Dot whose
  contracting dimension both operands shard along one mesh dimension that the output shards one of its own dimensions
  along gives its partial sums a ReduceScatter onto that output dimension (GSPMD §3.2; §5.1, "replace the AllReduce on
  the output BSM with a ReduceScatter", Figure 7's RS-y), where it gathered both operands' contracting dimension; the
  AllReduce where the output leaves the mesh dimension unused, and Figure 6's AllGather of a contracting dimension the
  operands shard unalike, as before (TA1 to TA3). Commits 7b429aa6, 7f0c60ea (a comment refilled).
- **TT. A program's calls as a dataflow graph, analysed at compile time** (aot-compilation A10), sub-agent T3.
  rdma/mesh_compile.py `program(block)`: Pathways' program tracer (Barham et al. §3) as torch.fx's symbolic tracer with
  each compiled function a leaf (a compiled module by `Tracer.is_leaf_module`, a compiled function by `torch.fx.wrap`,
  read in torch v2.14.0's source); each node reads its argument nodes and writes its result, so Bernstein's conditions
  order a call after one whose result it reads (TT2); each compiled function's bound is the width of that order on its
  calls, the most no two of which are ordered, by Dilworth's theorem through Fulkerson's matching, matched by Hopcroft
  and Karp's phases (the three the turns' D.5 names) (TT3); calls whose effects commute found by Rinard and Diniz's
  symbolic test (TOPLAS 19(6) §4.8, Figure 15, §4.9.2's rewrite rules) on the function's body traced by FX, a body FX
  cannot trace taken not to commute (§4.8) (TT4, TT5); the output, `Calls` (`mine`, `outstanding`, `orders`,
  `commuting`), carried in the IR's `calls` (A8), from the traced program under whose call the backend compiles the
  function (TT6 to TT8). mesh_plan's call and rdma/mesh.py unchanged: the bound reaches no plan (stop 1). Commit
  210f6cfe.

### Found already done (the code at f3f7cf46 and 4b0d2da2 does what the document states)

- **R4, the asynchronous, incomplete, noncontiguous consumer's outputs** ("An output page is available once its presence is set, as an element of an I-structure
  is ... Here the presence is C3's supplied word ... A producer's store releases whatever pages its work has written, any
  number at once and in any order. The bridge reads the words in place and sends each released page (R1)"; "The
  receiving side reads landings by masks (C3)"; "The release follows the producing work. The existing kernels write their
  pages, and the supplied store follows them. No kernel is written"). Each source range's one supplied word is stored by
  its producer after its last write (metal-microbench metal_recording.m: "A sent range's release: a system-scope fence
  after the producer's last store, then the range's supplied word raised"; rdma/mesh-collective.c mesh_host_publish);
  each send record of the range reads the word on its own (mesh-flow.c link_send_drain: "where the range's supplied word
  reads at least t + 1"), so every released page leaves as its lane reaches it, none waiting for another page; receives
  are read by the masked first pass (C3.1). "The revisions of 46ae8fdf", stop 2 (a release of each page) is answered by
  this text: one store releases whatever its work wrote.
- R5: "Every rank solves the same program with the same library, built from the same source with the same flags, on one
  thread": the source and flags are HB's, the one thread LP6's.
- Item 2: "deliver orders as Fig. 5's destinations do" (CH); "interleave released executions on each cable" (DL3, CH);
  "remove Q4's hold and Q4′'s timeout skip" (DL4, O1); "remove the coordination's Q3′" (SN; the pending ORDERs kept
  across a map move, V1 #4).
- Item 5: "Every program declares each consumer's kind" (EK); "The counter is (execution, set), so a set is drawn once
  per execution" (OP); Lemire's rejection step (LR). The item's fabric is a reading of the members, no code.
- A7's sequencer and its citation (SQ); item 6's names (N1); A9 and R8 as now.

### Decisions taken where the documents are silent (each the implementer's, with its reason)

- **HB1 (R5).** HiGHS's C++ sources compiled with -std=c++17, and every source with -DNDEBUG. Reason: BUILD.bazel sets
  neither, leaving the language standard to the toolchain and NDEBUG to Bazel's compilation mode, and every member's own
  clang has its own defaults; HiGHS's meson.build names `cpp_std=c++17`, and the HConfig.h BUILD.bazel copies defines
  `CMAKE_BUILD_TYPE "RELEASE"` and `HiGHSRELEASE`, a release build, which compiles with NDEBUG. So every member compiles
  the same source with the same flags (R5).
- **HB2 (R5).** "Links it statically" as the 216 objects themselves on each link line, with -lc++ for their runtime,
  not an archive. Reason: R5 names cc and c++ as the tools the build uses; an archive needs ar or libtool, and linking
  the objects is linking them statically.
- **HB3 (R5).** The source list, defines and config header are BUILD.bazel's target "highs" and its copy of
  highs/HConfig.h.bazel.in, where CMakeLists.txt's need CMake's configure step to make HConfig.h from HConfig.h.in.
  Reason: R5's cc and c++ alone, "cmake is not on the members"; BUILD.bazel is the source's own build of the library
  with a compiler and a copied header, so nothing of a configure is written here.
- **HB4 (R5).** The tarball, the extracted source and the objects live under rdma/.build/highs (`.build/` is ignored),
  and `make -B` fetches, checks and builds them again with everything else. Reason: -B remakes every target, and the
  fetch is the Makefile's own (the rules: "the Makefile's own fetch of the pinned HiGHS source").
- **LX1 (R5).** "C_max held at the first's optimum" is C_max's bounds both set to its value in the first objective's
  last solution, and the second objective's rounds start from the first's rows and basis. Reason: Isermann's sequential
  solution of a lexicographic program fixes each earlier objective at its optimal value before the next; the rows found
  are valid for the whole program, so keeping them only spares rounds. The first's last solution satisfies every row
  within HiGHS's tolerance (LP3), so the second program is feasible within that tolerance.
- **SW1 (item 2).** mesh_ports_named is removed with mesh_ports_held. Reason: the start's wait was its last caller (CH
  had removed the delivery's), and the rules: "a replacement covers what the old code did and the old code is removed".
- **SW2 (item 2).** "The late-loading port's refusal of it" is link_prepare's EALREADY where `executions` read above 0 at
  the load. Reason: it is the refusal the start's wait existed for (sub-assignment 5's reason, quoted at the removed
  code); a child that pairs again after a loss is refused ESTALE, the loss having moved the map's serial.
- **CH2-1 (item 2).** hdr `delivery_read[p]` is kept by name, read and written no more, and MESH_VERSION stays 113.
  Reason: no field moves (the header's rule, "a field is only ever appended"); bridge and children are one image.
- **CH2-2 (item 2).** The channel keeps M3's room, MESH_DELIVERY_ROOM entries laid out once with the region. Reason:
  the item's room by each plan's claim is stop 1 below; this is the room the code has.
- **PM1 (item 5).** A record of a member whose own position the deal gave another cable is built from that member's own
  hop (its page, words and bytes at this node) and stored after every lane's records, at no lane's place. Reason: "each
  record a view of that member's page (R1)"; the lanes' records stay in lane, copy and assignment order, which the
  cursors walk.
- **PM2 (item 5, D.3.5).** Every receive's round rule reads its page's phase field: round j posted only where the field
  reads mesh_phase(j). Reason: once a member's rounds are posted through several ports' children, a record's own
  landings no longer count its page's landings; the phase field is the page's one record of its landings, written by
  whichever child lands it (mesh.h, "Phases"), and for a page posted by one record alone the test is the old one (its
  round j - 1 posted only after j - 2 landed, so the field reads mesh_phase(j - 1) or mesh_phase(j)). Its reach,
  recorded (2026-11-05, the documents' answer: the field used outside its precondition, PTX's "valid only for" the
  current and the immediately preceding phase; appendix item 5 now names a page for round j by its range's rising count): the
  field is 2 bits (C3.7, "PTX's mbarrier phase bit, widened to two"), so it tells round j - 1's landing
  exactly while the page's earliest round not landed is at most three rounds before the one being posted; a member
  whose earlier rounds sit on another cable's lane four or more rounds behind would read as landed. The documents name
  no wider word, and nothing has been run.
- **PM3 (item 5).** The end's cancellation raises each receive's count to its final value wherever it reads below it.
  Reason: as PM2, no record counts its page's landings; count_raise leaves a count at or past its final as it is, and a
  plan ends at a port only as its claim closes or its link stops, when every receive of it not landed is cancelled
  (C3.1).
- **PM4 (item 5).** σ is x after set_draw read as x[j] = the member at position j, both counted in list order among the
  set's M positions on the pair. Reason: any fixed reading drawn alike at both ends gives each position one member in a
  uniformly random permutation; this one needs no inverse.

- **TA1 to TA4 (A4; sub-agent T3).** TA1: the ReduceScatter's local result keeps the output's sharding with the
  scattered dimension held whole, onto the output dimension that mesh dimension shards (GSPMD §3.2's and §5.1's
  examples). TA2: a partial's key is (node, its sharding, 'partial'), since lower passes _reduce_scatter its input's
  sharding from the key (TC1). TA3: each Dot form of _operator contracts one dimension, so a partial lies along at most
  one mesh dimension (`(m,) = partial` fails otherwise). TA4: CollectivePermute is inserted nowhere (stop 2).
- **TT1 to TT9 (A10; sub-agent T3).** TT1: a callable is compiled where Dynamo's own marks say so
  (`_torchdynamo_orig_callable`, and `_torchdynamo_wrapper_id` its own id, as eval_frame.py innermost_fn tests); a
  compiled module is a leaf where the block is a module that holds it (FX's path_of_module). TT2: a node reads all its
  argument nodes and writes its own result (symbolic tracing gives no types; in-place aliasing is not analysed), so of
  Bernstein's three conditions only a read of an earlier result can fail. TT3: "fewer where Bernstein's conditions order
  them" read as the order's width on the function's calls. TT4: the commutativity test on the body FX traces, its
  commutative and associative operators torch's add, mul, maximum and minimum. TT5: a commuting chain is calls of one
  function threading one accumulator through one argument, each result read by the next call alone; its calls are
  unordered among themselves in `orders` and still ordered for the bound, since A10's commuting calls "may run in any
  order" and its independent ones "in either order, or at once". TT6: `orders` the whole (transitive) order. TT7: the
  traced program runs by torch.fx.Interpreter, which names the call it is in (`CALLING`), read by the backend when
  Dynamo compiles within it; a function compiled outside any traced program carries None, and one compiled before keeps
  what it had. TT8: every IR carries the program's `orders` and `commuting` whole, `mine` its own calls. TT9: the
  analysis in Python, as the earlier ruling kept _complete there ("C9, torch").
- **C9a (the coordinating implementer's ruling on the C9 hook's remark on 210f6cfe).** The hook asks that _analyse (the
  order's closure and the chains), _width (Hopcroft and Karp's matching) and _execute (the symbolic execution's walk of
  a body's graph) move into the graph module. They stay in rdma/mesh_compile.py. Reason: the earlier ruling, "C9,
  torch: mesh_compile.py's sharding completion (GSPMD §3.5) stays there: it is the compilation's own pass (A2)", holds
  for A10's pass ("The compilation must take a program's sequence of calls as a graph, decide at compile time ..."); the
  C graph module has no closure, matching or width, so no second copy of one exists (rubric C9's concern); and Python's
  graph module, rdma/mesh_graph.py, is a file the rules bar from change. Open for the operator.
  Ruled 2026-11-05 by the operator: "inlined code is unilaterally bad and to be deduplicated and uninlined where
  possible and appropriate wrt graph operations module query." C9a is reversed: the closure, the chains' walk and the
  width move into rdma/mesh_graph.py, and the symbolic execution runs on torch.fx.Interpreter
  (design/aot-compilation-change-request.md A10, "Its graph operations are the graph module's").

### Where the work stopped

- **Stop 1. The room a plan claims in the deliveries' channel** (aot-compilation, appendix item 2):
  > **Its room is fixed at compile time**, as synchronous dataflow fixes its buffers before the program runs, from the
  > program's static schedule (E. A. Lee and D. G. Messerschmitt ...).

  > An entry stays until every lane that carries its plan has taken its call. A lane trails its node's deliveries by at
  > most two amounts added together. The first is A10's bound on the plan's calls outstanding at its submitter. The
  > second is C3.3's u copies on each hop of the plan's routes. ... This bound is this document's derivation.

  > So a plan claims room for that many entries, as it claims its pages (C3.4). A plan the room cannot hold fails with
  > `ENOMEM` when it is planned, as a value without a lag does in C3.3.

  No decision follows, for two reasons:
  - **Its first term.** A10's bound has a computation for torch (TT, below) and none for the program the code runs on
    another framework's delegate: A10 names Pathways' tracer and, "For torch", torch.fx's, and A1 gives that program's
    one function, not its calls. That program submits its whole run in one start (tools/e2b_replay.m; N4), so "the
    number of its calls in the block" would be its run's length, which C3.3 keeps out of the plan (mesh.h, "Copies": "the
    run's length is no input of mesh_plan"; design/disjoint-pages-revision-coordination.md, rule 3: "`executions` is no
    input of the plan").
  - **The claim against one channel in delivery order.** The channel is one area, delivery k in entry k mod its room
    ("in delivery order, through a channel laid out once with the region"), and a lane's place stays at the call it
    serves, so an entry is reused when the k + room-th delivery comes, not when its plan's lanes have taken it. The
    derivation bounds how far a lane trails its own plan's deliveries; another plan's deliveries at the node, on cables
    that lane does not serve, keep coming while it waits at its plan's call (for example, plan Q on one cable and plan P
    on another at one node, P's lane waiting at P's call n for P's program there to reach it, Q's calls delivered there
    meanwhile), so the plans' claims together do not bound the entries after it. Rooms one a plan would make "An entry
    stays until every lane that carries its plan has taken its call" hold, but then a call whose plan does not yet stand
    at a node has no room there, which "Each node runs the destinations' code of their Fig. 5, as written" delivers all
    the same (Fig. 5 has no such condition).

  What was built of the item's channel is CH2; what was not: the claim, its ENOMEM, and a room from the program's
  schedule.
  2026-11-05, the documents' answer: the stop's second reason holds, and the item's derivation was wrong for a channel
  shared by every plan. The item now gives each plan its own queue at each node, as synchronous dataflow gives each
  arc its FIFO queue, taken by a multiway merge on the sequence number; and A7 has every call of one program submitted
  by one node in the program's order, with no call of a plan submitted before every node of the plan holds it (as
  ncclCommInitRank synchronizes ranks), so a delivered call's plan always stands at its node. The first reason: a
  submission is one entry however many executions it carries (Pathways §4.5's "single message"), so e2b's run is one
  entry, and its length stays out of the plan.
- **Stop 2. CollectivePermute's insertion** (aot-compilation A4): "A2's partitioner inserts these two where GSPMD's
  does. ... CollectivePermute goes into halo exchange, "We use the CollectivePermute operator to exchange halo data
  between partitions" (§4.3), and into a pipeline's buffer shifting, "GSPMD will turn the buffer shifting into
  cross-device communication via CollectivePermute" (§3.3)." GSPMD places it in the halo exchange of windowed operators
  (§4.3: "e.g., Convolution, ReduceWindow") and of data formatting operators (Slice, Pad, Reverse, Reshape), in a
  pipeline's shifting by Pad and Slice (§3.3, §4.6: "can be done with a single CollectivePermute") and in resharding "to
  change device order" (§4.5). The partitioner partitions elementwise operators and Dot alone and refuses the rest
  (_operator, A-T2), and the placement operand names one device mesh, so none of those places arises among the
  operators it partitions, and _collective_permute is still reached by nothing. Reaching it needs the partitioner to
  partition one of those operators by GSPMD's rules, which A4 does not ask in words (sub-agent T3's stop).
  2026-11-05, the documents' answer: A2 asks it, "propagates them to every operator", and A4 now says so in words,
  with GSPMD §4.3's windowed operators; `_operator`'s "GSPMD gives no rule here for it" is false for them.

### Commits

- mesh 841cba44 (rdma/Makefile: HB), fe14e70a (rdma/mesh-graph.{c,h}: LX), e1525cc7 (rdma/mesh-dataflow.c,
  rdma/mesh-map.h, rdma/mesh-call.h: SW), c027460b (rdma/mesh-flow.c, rdma/mesh-records.c, rdma/mesh.h: CH2, SW's
  refusal), 8fd88d86 (rdma/mesh-flow.c, rdma/mesh.h: PM), ae433c04 (rdma/Makefile: the HiGHS rules after `all`),
  24746ef1 (rdma/mesh-flow.c: a note), 7b429aa6, 210f6cfe and 7f0c60ea (rdma/mesh_compile.py: TA, TT; sub-agent T3), and
  this record.
- metal-microbench a7642449 (tools/e2b_replay.m: PM's note, a comment).
- The rules' C9 hook (mesh .claude/settings.json, bin/mesh-c9-review.py on a commit) gave its remark on 210f6cfe
  (C9a); the earlier parts' "none installed" read only .git/hooks. No remark was shown on the coordinating
  implementer's commits.

### Integration

At mesh 7f0c60ea and metal-microbench a7642449, both trees clean but for untracked build artifacts and level with
origin: `make -B -C rdma` 0 (the pinned tarball fetched and its SHA-256 checked OK, HiGHS's 216 objects built,
mesh-flow, mesh-heartbeat, mesh-stat and libmesh.dylib, the bridge and libmesh linking HiGHS's objects), `make -B -C rdma
native-audit` 0 (mesh-flow.s, mesh-call.s, mesh-dataflow.s), `make .build/e2b_replay` 0 (libmesh_metal.dylib and
e2b_replay against the new libmesh), no warning or error. Each changed C file compiled alone first, the lifecycle files
unchanged and compiling against the new mesh.h; tools/e2b_replay.m compiled alone; rdma/mesh_compile.py compiles
(py_compile; the cache entry it made, not there before, removed). Nothing was run.

### What this changes among the earlier parts' open items

- "The revisions of 46ae8fdf", stop 1 (A10's analysis): answered by A10's revised text and built for torch (TT); its
  bound reaches no plan, and the program on a delegate has none (stop 1 above).
- That part's stop 2 (the asynchronous, incomplete, noncontiguous consumer's output pages): answered by R4's revised text and found already done.
- That part's stop 3 (no cycle where the deal splits a set): answered by item 5's revised text (σ one renaming over every
  lane, "So the deal fixes positions, not members") and built (PM); M1 is reversed. PM2's reach is recorded.
- That part's M4 (the one wait at the writer) and M5 (each port's copy of the channel): gone (CH2). The channel's room
  is M3's, its claim stop 1.
- "The start's wait for every port to load" (listed for removal at 46ae8fdf's end): gone (SW).
- "The revisions of 46ae8fdf", LP's objective (C_max alone, C̄_j off the critical path the solver's vertex's): the second
  objective fixes them (LX).
- "Before any run: the bridge now loads libhighs": no longer; HiGHS is in the bridge and libmesh, built from its pinned
  source by the Makefile (HB).

## The revisions of abecd389

The record of the unit whose instruction is design/plan-to-bridge-change-request.md and
design/aot-compilation-change-request.md as mesh abecd389 holds them, from mesh abecd389 and metal-microbench a7642449.
Its rules are design/disjoint-pages-revision-coordination.md, "Rules (every sub-agent)", verbatim, but its reading order
(the two documents in full, then mesh AGENTS.md, design/rubric-actionable.md, metal-microbench AGENTS.md,
docs/principles.md, docs/measurement.md) and its file table: where a document names a file to change, the document
stands over the table (A10 of rdma/mesh_graph.py: "The rules' bar on rdma/mesh_graph.py does not stand against it").
What the code did not yet do was found by reading both documents in full against the code, beginning with what mesh
9b2c0869, 8e403496 and abecd389 revised: A4 (the partitioner partitions every operator, so CollectivePermute is
reached), A7 whole (total order multicast by a fixed sequencer as Défago, Schiper and Urbán define it, one sender for
each program, a plan at every node of it before its first message), A10 (its graph operations the graph module's, the
symbolic execution on torch.fx.Interpreter, "The same holds wherever mesh's code walks a graph", which reverses C9a),
appendix item 2 (each plan's own queue of delivered calls at each node, its room, the multiway merge, nothing waiting
to write and nothing checking; this replaces CH2's one channel for every plan) and item 5 (a page's round rule by its
range's rising count; this replaces PM2). Each decision taken where the documents are silent is marked as the
implementer's (SQ, PS, PB, PQ, RC, PM5 and C9b the coordinating implementer's; T4-n sub-agent T4's) and is not a
requirement of the documents. Sub-agent T4 was the one owner of rdma/mesh_compile.py and rdma/mesh_graph.py; the
rest is the coordinating implementer's. Nothing was run but the compilers, `python3 -m py_compile`, the Makefile's own
fetch of the pinned HiGHS source and the three integration builds.

### Done

- **SQ1. The sequencer as Fig. 5 states it** (aot-compilation A7: "The algorithm is their Fig. 5, the "UB" variant of
  §7.1, as written"; "Sequencer: "Initialization: seqnum := 1"; "when receive (m): sn(m) := seqnum; send (m, sn(m)) to
  all; seqnum := seqnum + 1""; "m is one submission ... One submission is one message however many executions it
  carries"). rdma/mesh-records.c: `seqnum` from 1 at the bridge's start, one number a message (order_sequence), where
  it was from 0, one number an execution; mesh-bridge.h's note on struct mesh_order `sequence`. Commits 7eb93c51,
  c38ee188.
- **SQ2. The destinations as Fig. 5 states them, over §5.2.4's super-group** (A7: "Destinations: "Initialization:
  nextdeliver := 1; pending := ∅"; "when receive (m, seqnum): pending := pending ∪ {(m, seqnum)}; while ∃(m′, seqnum′)
  ∈ pending : seqnum′ = nextdeliver do deliver (m′); nextdeliver := nextdeliver + 1""; "Multicast to many groups is
  their §5.2.4, as written ... 3. processes not in Dest(m) discard m."  The super-group is the island's nodes";
  "deliver(m) at a node of Dest(m) appends m to its plan's queue there"). rdma/mesh-records.c: `nextdeliver` from 1;
  every ORDER held (order_take), where one whose `nodes` lacked this node was not; orders_deliver delivers the pending
  message whose `sequence` equals `nextdeliver`, wherever it sits, then the next, each at a node of Dest(m) appended to
  its plan's queue (order_deliver: entry k mod `room`, then the plan's `delivered`, release), elsewhere discarded; hdr
  `delivered` nextdeliver − 1 between two raises of hdr `plans` (SQ2-1, SQ2-2); stored 0 at the bridge's start
  (map_adopt). The delivery by arrival order, the pending head standing for nextdeliver and the sequence not compared
  (the recorded difference of Q3, Q3-tree and DL2), is gone. Commits 7eb93c51, c38ee188.
- **PS. One sender for each program, its submissions TO-broadcast in the order the program made them** (A7: "sender(m)
  is the program's one sender: the least of the program's nodes, the program being its launcher's `group`
  (rdma/mesh.h).  It TO-broadcasts the program's submissions in the order the program makes them"; "With one sender,
  whose messages reach the sequencer first in, first out, Fig. 5's sequencer numbers them in the order they were sent
  ... So FIFO Order holds, and with it A10's order constraints between the program's calls").
  - rdma/mesh-collective.{h,c} mesh_plan: `program` (the program's nodes, a set of map indices) after `function`;
    struct mesh_state `sender` the least of it, where it was the least of the plan's `nodes`; EINVAL where `program`
    is 0, lacks this node or lacks a node of the plan's (PS-1, PS-2).
  - rdma/mesh-dataflow.{h,c}: struct mesh_ctx `clock`, the program's Lamport clock (L. Lamport, CACM 21(7), 1978, IR1:
    "Each process Pi increments Ci between any two successive events"); mesh_transfers_start appends each raise of a
    plan's `executions` to the plan's submissions' queue (struct mesh_submission: the raise's end and the clock after
    one increment, then `submissions`, release), with no wait and no check of its room (PS-3, PS-4).
  - rdma/mesh-records.c orders_submit: at the program's sender, its plans' submissions' queues merged on the clock
    (D. E. Knuth, TAOCP vol. 3, §5.4.1, as item 2 states the merge), read in L. Lamport's opposite order (CACM 20(11),
    1977), one message a submission, [`submitted`, the entry's end), sent toward the sequencer only once the plan's
    barrier is complete (PB); `broadcast` counts the entries taken.
  - rdma/mesh.py: Context `clock`, mesh_plan's binding; metal-microbench tools/e2b_replay.m: `program` its ranks' map
    indices (a solo run's this node) and `outstanding` 1 (PS-5). Commits 7eb93c51, 8641a3e6, c38ee188, metal-microbench
    27093ce3.
- **PB. A plan at every node of it before its first message** (A7: "The program's sender counts the plan's nodes as
  each holds the plan, as a centralized barrier counts arrivals: "each processor updates a small amount of shared state
  to indicate its arrival and then polls that state to determine when all of the processors have arrived" (J. M.
  Mellor-Crummey and M. L. Scott ... §3.1).  It TO-broadcasts no submission of the plan before every node of the plan
  has arrived"). rdma/mesh-bridge.h: ARRIVE and struct mesh_arrival (MESH_VERSION 114); rdma/mesh-records.c: each node
  tells the sender that it holds each plan standing there (orders_announce), up the island's tree as a SUBMIT goes, to
  the sender where the path meets it, else to the sequencer and from there down the tree as an ORDER goes
  (arrival_up, arrival_down, order_heard); the sender holds what it hears (arrival_take) and counts it into its plan's
  `arrived` once that plan stands (arrivals_apply); orders_submit takes no submission of a plan whose `arrived` does
  not hold its `nodes`, and the program's later submissions wait behind it (PB-1 to PB-3); rdma/mesh-ports.c takes
  ARRIVE to order_heard. Commits 7eb93c51, c38ee188.
- **PQ. Each plan's own queue of delivered calls at each node, taken by a multiway merge** (appendix item 2: "Each
  plan's delivered calls reach the bridge at a node through a queue of the plan's own.  It is laid out with the plan's
  pages when the plan is placed, and claimed and released with them (C3.2, C3.4).  The bridge reads it in place, by
  view, and nothing copies it"; "Each entry is one delivered message, with its sequence number ... The plan's k-th
  delivery at the node is entry k mod the queue's room"; "Its room is fixed at compile time ... at most A10's bound on
  the plan's submissions outstanding at its sender, plus C3.3's u copies on each hop of the plan's routes ... A plan
  whose bound A10 cannot find fails with `ENOMEM` when it is planned"; "The bridge takes its plans' calls by a multiway
  merge on the sequence number (D. E. Knuth ... §5.4.1): of the first untaken entries of its standing plans, the least
  sequence number first"; "Nothing waits to write, and nothing checks for a full queue"; "This replaces ... the one
  channel for every plan (the implementation's CH2, with M3's room)").
  - rdma/mesh.h: struct mesh_delivery (sequence, first, end) an entry of a plan's queue; struct mesh_state's words
    from byte 156 renamed in place (`outstanding`, `delivered`, `submissions`, `broadcast`, `arrived`, `room`,
    `deliveries`, `submits`, `announced`), `released`, `taken` and `sequences` gone; hdr `deliveries_off`,
    `deliveries_room` and `delivery_read` kept by name; MESH_DELIVERY_ROOM gone; rdma/mesh-map.h: the channel's area
    gone from the layout.
  - rdma/mesh-collective.c mesh_plan: `outstanding`, A10's bound, after `program` (0 ENOMEM); the delivered calls'
    queue of `room` = `outstanding` + u × the hops of the plan's routes (PQ-1) and the submissions' queue of
    `outstanding` laid out in the state section between the status and the entries (plan_layout), started 0 with the
    section (D3).
  - rdma/mesh-flow.c: struct flow's view of the queue (`deliveries`, `room`) and each lane's place in it (`at`, 0 at
    the load; PQ-2), EINVAL at the load where the queue lies outside the region; lane_take the merge: hdr `delivered`
    first, then each carried plan's first untaken entry, the least numbered at most hdr `delivered` taken, nothing
    where hdr `plans` moved since the walk (PQ-3); struct port_plans `at` and `joined`, deliveries_join and
    plan_refused_here gone. Commits 7eb93c51, 8641a3e6, 930861b0.
- **RC and PM5. The round rule by the range's rising count** (appendix item 5: "So a page is named for round j where
  its range's count of its copy reads at least j times the pages this node receives of that range in a copy ... The
  code must name a page for round j by that count, in place of PM2's test of the phase field.  The phase fields stay,
  for C3.7's readers"). rdma/mesh-flow.c link_receive_drain: a record of round j posted only where its `count` reads
  (acquire) at least j × `pages`, where PM2 read the page's phase field; the landing's XOR stays. rdma/mesh-collective.c
  plan_layout and mesh_plan: every received range has its count words, a relay's own too (`counted`), so every receive
  entry names one (RC-1); link_prepare refuses a receive with none (EINVAL). Commits 7eb93c51, 8641a3e6, 930861b0.
- **T4-A10. A10's graph operations the graph module's** (aot-compilation A10: "The order's transitive closure, the walk
  of commuting chains, and the width by Fulkerson's matching in Hopcroft and Karp's phases are functions of
  rdma/mesh_graph.py, beside `walk` and `reach` ... rdma/mesh_compile.py calls them and keeps no copy"; "A body's
  symbolic execution runs on torch.fx.Interpreter, as the program's run already does (`_Program`), not on a second
  loop over a graph's nodes"; "The same holds wherever mesh's code walks a graph"; C9a reversed), sub-agent T4.
  rdma/mesh_graph.py: `closure` (A. Goralčíková and V. Koubek's reduct-and-closure over a topological order, MFCS 1979),
  `paths` (a linear forest's paths, read by `walk`), `matching` (Hopcroft and Karp's phases), `width` (Dilworth by
  Fulkerson's matching) and `propagate` (rounds to a fixed point alternating forward and backward passes, as GSPMD §3.5
  runs its completion; round-robin iteration, Kildall 1973, Kam and Ullman 1976), each over the caller's vertices and
  neighbours (and, for `propagate`, visit) functions. rdma/mesh_compile.py: `_analyse` calls `paths`, `closure` and
  `width`, its ancestors' and chains' loops and `_width` gone; `_execute`'s loop over a graph's nodes replaced by
  `_Symbolic`, a torch.fx.Interpreter computing Rinard and Diniz's expressions with the same rewrite rules; `_mesh`'s
  recursive walk of the nested device mesh by `walk`; `_complete`'s rounds by `propagate` (after the C9 hook's remark
  on cc4a3e12). Commits 624ecd6f, 92336795.
- **T4-A4. CollectivePermute reached by partitioning every operator** (aot-compilation A4: "The partitioner reaches them
  by partitioning every operator, as A2 states ("propagates them to every operator").  GSPMD gives the rule for the
  operators that need CollectivePermute: "The most typical use case of halo exchange is for partitinoning window-based
  operators (e.g., Convolution, ReduceWindow)" (§4.3).  The partitioner's refusal of an operator, "GSPMD gives no rule
  here for it" (rdma/mesh_compile.py `_operator`), is false for these"; "CollectivePermute goes into halo exchange ...
  and into a pipeline's buffer shifting ... (§3.3)"), sub-agent T4. rdma/mesh_compile.py: `_operator` builds each
  operator's rule: elementwise (broadcast operands now taken), Dot (torch's matrix products by matmul's labels,
  `F.linear` with a bias, `bmm`, `torch.einsum` by its equation; GSPMD §3.2), Convolution and transposed convolution
  (`_convolution`) and ReduceWindow (`_pooling`) with the halo exchange of GSPMD Figure 9b, §4.3 and A.2 (`_region`,
  `_extent`: each halo piece a narrow sent by one CollectivePermute, XLA's source_target_pairs, concatenated, a
  DynamicSlice at the partition's offset, positions outside the operand masked by a Select against an Iota, §4.1; a
  transposed convolution by A.2's base-dilation method and a DynamicSlice of its output), constant Pad (`_pad`) and
  Slice (`_slice`, getitem and narrow; §4.3), a Pad that its one Slice reads merged into it (`_merged`, §4.6), so the
  pipeline's "pad_left(state, 1)[:-1]" (§3.3) is one CollectivePermute; every other operator by the general rule
  (§4.5's resharding to §3.1's replicated sharding: operands gathered, run whole, output sliced; T4-5); "GSPMD gives no
  rule here" gone from the code; `_run` runs the new items with torch's own operators (A9). Commit cc4a3e12.
- **T4-P. The plan call** (the coordinating implementer's PS-1 and PQ): `serialize` passes `program`, the set of the
  placement operand's map indices (`ir.map`), and `outstanding`, A10's bound `ir.calls.outstanding`, 0 where the
  function was compiled under no traced program (so mesh_plan's ENOMEM, item 2). Commit 41cf06d6.

### Found already done (the code at abecd389 and a7642449 does what the document states)

- A7: "The island is the map's Thunderbolt-linked component" and "The sequencer is the bridge of the highest-numbered
  node the map reaches (rdma/mesh-records.c `island_set`)" (SQ); "Dest(m) is the nodes of m's plan (struct
  mesh_order's `nodes`)"; "sn(m) is struct mesh_order's `sequence`"; "Failures ... no standing plan's sequencer
  changes" (D.3.6's cancellation, plan_stale), with no handler added (items 3 and 4).
- Item 2: "remove the start's wait for every port to load the plan, and the late-loading port's refusal of it" (SW);
  "remove Q4's hold and Q4′'s timeout skip" (DL4, O1); "remove the coordination's Q3′" (SN, kept: no reset of `seqnum`,
  of `nextdeliver` or of the pending ORDERs at a map move); "M4 and M5 stay gone" (CH2); "interleave released executions
  on each cable" (the merge, now PQ's); "Recorded order: ... No collective or data code reads it" (MESH_LEDGER's
  traces, unchanged).
- Item 5: "So the deal fixes positions, not members" (PM); "The counter is (execution, set), so a set is drawn once per
  execution" (OP); Lemire's rejection step (LR); "Every program declares each consumer's kind" (EK).
- A10's tracer, Bernstein's order, the width bound and Rinard and Diniz's commuting calls (TT), now with its graph
  operations moved (T4, above).

### Decisions taken where the documents are silent (each the implementer's, with its reason)

- **SQ2-1 (A7).** The destinations' delivered count is published to the ports as hdr `delivered` = nextdeliver − 1,
  stored 0 at the bridge's start (map_adopt). Reason: item 2's merge needs "the node's delivery count" ("every entry
  below the node's delivery count is in its queue already"), which only the bridge parent holds; the field was the
  channel's count, kept by name; 0 at the start because Fig. 5's nextdeliver is 1 there and a region outlives a bridge.
- **SQ2-2 (A7, item 2).** hdr `plans` is raised before and after hdr `delivered`'s store. Reason: a port that acquires
  `delivered` must see `plans` moved where a plan stood since its last walk (the first raise follows the delivery into
  a plan the parent saw standing, so a lane that acquires the count reads it), and a port that read the first raise
  and passed before the count's store must be woken (the second); this is DL7 and DL9's pairing, kept.
- **SQ2-3 (A7).** A message whose plan does not stand at a node of its Dest(m) is discarded there, said on the bridge's
  log. Reason: deliver(m) appends m to its plan's queue, which exists only while the plan stands; the barrier (PB) makes
  the plan stand at every node of it before its first message, so this is a plan retired or cancelled meanwhile.
- **PS-1 (A7).** "The program's nodes" are an operand of the plan call, `program`, the same in every plan call of one
  program. Reason: A7 names the program by its launcher's `group` and its sender as the least of its nodes, but neither
  document gives a computation of those nodes, and mesh_plan sees one function's transfers; only the program knows all
  its nodes at its first plan, and every plan of one program must name one sender for FIFO Order to hold across its
  plans. tools/e2b_replay.m passes its ranks' map indices 0 .. R − 1 (rank i runs on map index i) and a solo run this
  node; rdma/mesh_compile.py passes the placement operand's nodes (A2's mapping of the logical mesh onto the map's
  nodes; T4's code, my choice).
- **PS-2 (A7).** mesh_plan is EINVAL where `program` lacks this node or a node of the plan's `nodes` (a relay outside
  the program). Reason: such a node runs no call of the program and never holds the plan, so the sender's barrier would
  never count it; an error value at planning, not a wait (the rules: "Every failure is an error value, never a hang").
- **PS-3 (A7).** The program's order reaches its sender's bridge through a queue of each plan's own submissions (each
  raise of `executions` and the program's Lamport clock at it), merged at the sender on the clock and read in Lamport's
  opposite order. Reason: A7 requires the sender to TO-broadcast "in the order the program makes them" and says nothing
  of how that order reaches the bridge; each plan's rising `executions` alone loses the order of two plans' starts that
  fall between two of the bridge's reads. The structures are the documents' and their sources': Lee and
  Messerschmitt's FIFO queue on the arc (here from the program's submission to its sender's TO-broadcast), Lamport's
  logical clock (1978, IR1) as the merge key, Knuth's multiway merge (§5.4.1) and Lamport's opposite-order reading
  (1977), as item 2 merges delivered calls.
- **PS-4 (A7, item 2).** The submissions' queue's room is `outstanding`, A10's bound. Reason: an entry stays until the
  sender's bridge TO-broadcasts it, and a call cannot complete before its submission is broadcast, delivered and
  served, so the untaken entries are at most the program's calls of the function outstanding, A10's bound; no writer
  waits and nothing checks (item 2's rule for its queue). At a node not the sender the queue is written and read by
  nobody.
- **PS-5 (item 2).** e2b's `outstanding` is 1. Reason: the documents' own: "e2b's run is one submission" (item 2), "e2b's
  run is one message" (A7); its program makes one start.
- **PB-1 (A7).** An arrival travels as a message, ARRIVE, up the island's tree to the sender where the path meets it,
  else to the sequencer, then down the tree to every node, kept at the node it names as the sender. Reason: the
  barrier's shared state is the sender's plan's word, written by its bridge parent alone, and the bridges share no
  memory but the control sockets; these are Q3's paths (a SUBMIT up, an ORDER down), so no routing of its own is added.
- **PB-2 (A7).** "Holds the plan" is the plan standing at the node (its claim head's publication `state`). Reason:
  delivery needs the plan's queue, which exists once the plan stands; a port's load is not needed, the queue keeping
  every entry until each lane carrying it has taken it.
- **PB-3 (A7).** The sender holds an arrival it hears before its own plan stands and counts it once that plan stands; it
  lets go of arrivals held and not counted at its own map move; every node tells each standing plan's arrival again on
  each liveness turn (MESH_PAUSE_NS), as record_retell tells a record again. Reason: a plan's nodes finish compiling
  at different times (A7: "this is trivially allowed"), so arrivals precede the sender's plan; at a map move every plan
  there is cancelled (D.3.6) and a replan's barrier must count the replans' arrivals, which may come before or after
  the sender's own move; the arrival told again recovers one let go or lost, by the tag's practice for its records.
- **PQ-1 (item 2).** "C3.3's u copies on each hop of the plan's routes" is u × the hops of the plan's routes, a hop a
  distinct tree edge carrying pages of one of its demands (each demand's `forwards`, summed). Reason: the literal
  reading, u for each hop; the sum over every hop is at least u × any path's hops, so it also bounds the item's
  per-node trail ("a node's part of a call trails the sender's by at most u copies a hop").
- **PQ-2 (item 2).** A lane's place in a plan's queue starts at 0 at the plan's load. Reason: by the barrier, the queue
  holds the plan's every message from its first, and by the room no entry is reused before every lane carrying it
  has taken it.
- **PQ-3 (item 2).** The merge takes nothing where hdr `plans` moved since the port's last walk, and sets `again`.
  Reason: a plan standing here and not yet loaded at the port would be missing from the merge, so a later number of a
  loaded plan could be taken first while the cable's other end takes the earlier; SQ2-2's first raise makes such a
  plan visible as a moved `plans` (DL3's `plans` test, kept).
- **RC-1 (item 5).** A relay's own received ranges get count words, where review 15's finding 1's ruling gave them none.
  Reason: the item's method needs the count at every receive it names ("a page is named for round j where its range's
  count of its copy reads at least j times the pages this node receives of that range in a copy"), and a relay's
  receives are named for rounds as a destination's are; C3.1's counted completion is the structure, one more word per
  relayed range and copy, raised by its landings. The item's "The plan keeps such a count already" and "adds no word"
  are false at a relay (below, "Where the documents are wrong").
- **PM5 (item 5).** The count test reads `count` with acquire and compares with the record's round j × `pages`; the
  record's `round` is still stored as it is posted, for the landing's phase flip. Reason: the item's statement; the
  phase fields stay for C3.7's readers.
- **C9b (the C9 hook's remark on c38ee188).** The hook read arrival_up and arrival_down as walks of the island's tree
  that belong in the graph module. They stay in rdma/mesh-records.c. Reason: each sends one message one hop, to this
  node's parent or children, as order_submit and order_relay do; no graph is walked in memory, the tree being
  island_set's, computed by mesh_graph_reach and mesh_graph_rows of the graph module; A10's "The same holds wherever
  mesh's code walks a graph" is met there.
- **T4-1 to T4-22 (A10, A4; sub-agent T4; each at its code, citing this part).** T4-1: `closure` by Goralčíková and
  Koubek, and `matching` a function of its own that `width` calls, so the primitive is one. T4-2: the commuting chains
  are a linear forest's `paths`. T4-3: the graph walks moved (`_mesh`'s, `_complete`'s rounds) and those kept as one
  pass or one step of adjacency (`_dependences`, `_groups`, `_readers`, `_consumer`, `partition`, `_partition`,
  `lower`). T4-4: graph walks in other rdma/*.py files reported, not edited (rdma/peers.py `observe_mesh` and
  `realize`, breadth-first; rdma/mesh_disk.py `free`, a parent chain; mesh_hourly.py and mesh_preflight.py already call
  `walk` and `reach`). T4-5: the general rule for an operator GSPMD gives no special rule (§4.5 with §3.1), nothing
  propagated through it, its preserved dimensions unknown. T4-6: windowed and data formatting operators propagate at
  level 1, since §3.5 gives the highest priority only where "there is no communication". T4-7: a halo wider than one
  shard takes one CollectivePermute per neighbour distance. T4-8: every operand's halos exchanged after its reshard.
  T4-9: no sharding carried onto a dimension its mesh dimension does not divide, or of size 0. T4-10: transposed
  convolution by A.2's superset-then-DynamicSlice method at every partition count. T4-11: an average pooling dimension
  whose divisor depends on the window's place held whole. T4-12: groups other than 1, or padding 'same' with an odd
  total, held whole. T4-13: a fused bias gathers its contracting dimension rather than reducing it, so the bias is not
  summed once a partition. T4-14: a contracting dimension sharded unalike whose mesh dimension the output leaves unused
  gathered, where it was refused. T4-15: broadcast labels aligned from the right. T4-16: in-place operators by their
  functional equivalents (PyTorch 2 §3.9), writes to inputs, attributes or views refused. T4-17: a Pad merged only
  where its Slice is its one reader. T4-18: the mask's fill 0 for convolution and average pooling, −inf (or the
  integer minimum) for max pooling, the pad's value for Pad. T4-19: `_consumer` gives an asynchronous, incomplete, noncontiguous consumer only where the read
  value's local shape is its reader's. T4-20: a get_attr held whole as a placeholder; call_module and non-tensor values
  by the general rule. T4-21: matmul's labels in general, `bmm`, einsum by its equation, at most one mesh dimension a
  partial. T4-22: `ir.reduced` keyed by the partial's key.

### Where the work stopped

- **Stop 1. Fig. 5 "as written" in an island whose bridges begin at different times** (aot-compilation A7):
  > The algorithm is their Fig. 5, the "UB" variant of §7.1, as written: ... Sequencer: "Initialization: seqnum := 1"
  > ... Destinations: "Initialization: nextdeliver := 1; pending := ∅"

  > A plan's destinations exist before its first message.  Section 3's specification is "the standard specification
  > of total order broadcast in a static system, that is, a system in which all processes are created at system
  > initialization" (§6).

  > The sequencer is the bridge of the highest-numbered node the map reaches (rdma/mesh-records.c `island_set`), the
  > code's choice, which the class leaves open.

  and item 2: "the sequencer's number never goes back to 0, and no node drops a pending order when its map changes."
  The facts of the code: each bridge is a process whose Fig. 5 initialization is its own start; bridges start and
  start again at different times; island_set computes the sequencer anew at every map move, so a node with a greater
  id joining moves it. The documents do not say what a destination's nextdeliver is when its bridge begins after the
  island's sequencer has numbered messages, nor what a destination does when the sequencer it hears from is a process
  begun after it (a bridge started again, or a new highest node). Défago, Schiper and Urbán leave both to a group
  membership service: "The recovery of processes and the dynamic join of new processes are discussed in Section 8.2"
  (§4.1, note 10); "Blocking can also be prevented by relying on a higher level mechanism, namely a group membership
  service" (§8.2); "A dynamic group is a group of processes with a membership that can change during the computation"
  (§6.1). Neither document names one, and the operator ruled the earlier stand-in (Q3′, the reset at a map move) out.
  What is built is Fig. 5 as written, initialized at each bridge's start (SQ1, SQ2). Its consequence, read from the
  code (nothing was run): only where every bridge of an island has started before the island's first message does
  every nextdeliver meet the sequencer's numbers; a bridge started after the sequencer has numbered messages, and every
  destination after the sequencer's bridge starts again or the sequencer moves to a bridge whose count is behind
  theirs, holds the ORDERs it receives pending and delivers none.
- **Stop 2. The delivered calls' queue's room where the program's sender holds no page of the plan** (appendix item 2):
  > An entry stays until the bridge at the node has taken its call.  That is at most A10's bound on the plan's
  > submissions outstanding at its sender, plus C3.3's u copies on each hop of the plan's routes: a copy is used again
  > only after the last reads of it (C3.3), so a node's part of a call trails the sender's by at most u copies a hop.

  against A7's revised sender: "sender(m) is the program's one sender: the least of the program's nodes". The
  derivation was written when the sender was the plan's least node. Now the sender is the program's, and a plan's nodes
  may be any of the program's (a collective among a subgroup that leaves the least device out). Where the sender holds
  no page of the plan, its part of each call of the plan is done at once (rdma/mesh_compile.py `run`: the start, then
  items with nothing to wait on), so its program issues the next call as soon as A10's bound lets it and nothing in the
  derivation bounds how far its submissions lead the plan's nodes; a node of the plan whose program is behind then
  receives deliveries past `room` before its lanes take the earlier, and item 2 forbids a check ("Nothing waits to
  write, and nothing checks for a full queue"). The room is built as the item states (PQ, PQ-1); no check and no
  refusal is added, and nothing was run. The documents do not say which bound holds there.

### Where the documents are wrong, or contradict the code's facts

- Appendix item 5: "The plan keeps such a count already: each range's landed count of each copy, which every landing
  raises, by whichever cable it lands (C3.1, counted completion)" and "The test holds at any distance and adds no word."
  At a relay it did not: rdma/mesh.h at abecd389, a receive's "`count` the region offset of its range's count word for
  this copy, or 0 where this node does not consume the range (a relay's own receive: its forwards are released by its
  phase, nobody waits on a count, and the state section has no count word for it)". RC-1 adds the relay's words.
- A7 cites §6's static system for "A plan's destinations exist before its first message", and requires Fig. 5's
  initializations as written, but the island's bridges are not "created at system initialization" (stop 1).
- Appendix item 2's room derivation assumes the sender's part of each call trails like any node's, which A7's
  program's sender need not (stop 2).
- A7's m carries "the map serial it was made on": each node's map serial is its own count (Q3′: "two nodes on one map
  hold different serials"), so the serial m carries is the sender's, and no receiver compares it (as before).
- A7: "the least of the program's nodes, the program being its launcher's `group` (rdma/mesh.h)": rdma/mesh.h names
  `group` as the launcher's but holds no program's nodes, and the plan call took none (PS-1).
- A4 routes "all of one function's collectives jointly over the Thunderbolt-linked nodes of the map", so a tree may
  relay through any node of the map, while a node holds a plan only where its program calls mesh_plan, and A7's barrier
  waits for every node of the plan (sub-agent T4's finding). With `program` the placement's nodes (PS-1), a plan whose
  route relays through a node outside the placement is EINVAL at planning (PS-2), where before it stood and its relay
  held no plan. The documents do not say what program runs at a relay outside the placement.
- Not done of A4, recorded by T4: GSPMD's own handling of Reverse, Reshape, Concatenate and rotation (§4.3, §4.6), of a
  reduction's masking (§4.1) and of Broadcast's priority (§3.5) is not built (those operators take the general rule,
  T4-5); `partition`'s refusal of a dimension its mesh dimension does not divide ("GSPMD section 4.1 pads it, which
  the contract does not give") stays, though GSPMD §4.1 gives that rule; CollectivePermute "to change device order"
  (§4.5) still arises nowhere, the placement naming one device mesh; a partitioned convolution kernel is held whole (A.2:
  "We skip the implementation details").

### Commits

- mesh 7eb93c51 (rdma/mesh.h, mesh-map.h, mesh-dataflow.h, mesh-bridge.h, mesh-collective.h, mesh.py: the headers and
  bindings), 8641a3e6 (rdma/mesh-collective.{c,h}, mesh-dataflow.c: PS, PQ, RC), c38ee188 (rdma/mesh-records.c,
  mesh-ports.c: SQ1, SQ2, PS, PB), 930861b0 (rdma/mesh-flow.c: PQ, PM5, RC), 624ecd6f, cc4a3e12, 92336795 and 41cf06d6
  (rdma/mesh_graph.py, mesh_compile.py: T4-A10, T4-A4, T4-P; sub-agent T4), and this record.
- metal-microbench 27093ce3 (tools/e2b_replay.m: the plan call's `program` and `outstanding`, PS-1, PS-5).
- The C9 hook (bin/mesh-c9-review.py on a mesh commit) gave remarks on c38ee188 (C9b, above) and on cc4a3e12 (T4 moved
  `_complete`'s rounds into rdma/mesh_graph.py, 92336795).

### Integration

At mesh 41cf06d6 and metal-microbench 27093ce3, both level with origin, the trees clean but for this record and
untracked build artifacts: `make -B -C rdma` 0 (the pinned HiGHS tarball fetched and its SHA-256 checked OK, its 216
objects built, mesh-flow, mesh-heartbeat, mesh-stat and libmesh.dylib), `make -B -C rdma native-audit` 0, `make
.build/e2b_replay` 0 in metal-microbench, no warning or error in any. Each changed C and Objective-C file compiled
alone first, the lifecycle files (mesh-heartbeat.c, mesh-children.c, mesh-child.c, mesh-lanes.c) unchanged and
compiling against the changed headers; rdma/mesh.py, mesh_compile.py and mesh_graph.py compile (py_compile; the
cache entry mesh_compile.cpython-314.pyc it made, not there before, removed). Nothing was run.

### What this changes among the earlier parts' open items

- "The revisions of f3f7cf46", stop 1 (the room a plan claims in the channel): answered by item 2's revised text and
  built (PQ): each plan's own queue, its room `outstanding` + u × its routes' hops, ENOMEM where A10's bound is not
  found; its derivation's premise fails where the program's sender holds no page of the plan (stop 2 above).
- That part's stop 2 (CollectivePermute's insertion): built (T4-A4) for windowed operators' halos and the pipeline's
  shift; not for device order (above).
- That part's C9a: reversed (T4-A10), as the operator ruled.
- That part's CH2, CH2-1 and CH2-2: the one channel and MESH_DELIVERY_ROOM gone (PQ); hdr `delivery_read` and the
  channel's offsets kept by name; hdr `delivered` is now the node's delivered count (SQ2-1).
- That part's PM2 and its recorded reach: gone (PM5); every received range has its count (RC-1).
- "Delivery's liveness" (an ORDER lost on the way to a node): under Fig. 5 as written, a node that misses a number
  holds every later ORDER pending and delivers none, where the arrival-order delivery passed over it; no handler
  (items 3 and 4).
- "Numbers across a reset" (46ae8fdf's): no number is reset (SN, SQ1); its successor is stop 1 above.
- "Replans reuse names": the barrier counts arrivals by the plan's name (PB-3), which a replan keeps (N1); a node that
  tells again the arrival of a plan of the old map, before its own map moves, can be counted toward the sender's
  replan, within one flooding of the map; recorded, no handler.

### Answered in the documents (2026-11-05, the coordinator)

The operator, after this part's report: "mostly i want the work done and i care far less about ceremony or
ritualization than about the delivery of source which meets several strict qualitative and several strict
quantitative requirements."

- **Stop 1** (Fig. 5 in an island whose bridges begin at different times): answered by aot-compilation A7, "The island
  is a dynamic group, and each of its views is a static system". The views are G. V. Chockler, I. Keidar and
  R. Vitenberg's (ACM Comput. Surv. 33(4), 2001), with their Properties 3.1 (Self Inclusion), 3.2 (Local
  Monotonicity), 3.3 (Initial View Event) and 4.3 (Sending View Delivery) verbatim. The membership service is the
  sequencer's unilateral decision (Amoeba, Défago, Schiper and Urbán §9.1.1). A bridge started again is a new process
  (Isis's incarnation number, Chockler et al. §3.1.1). The view identifier is Transis's integer, computed from the
  members' counters. Fig. 5 runs as written within each view, initialized at its creation and its installation. A view
  change blocks and flushes (Chockler et al. §4.2.1), and a plan stands only within the view its arrivals were counted
  in, which also answers "Replans reuse names" above.
- **Stop 2** (the room where the program's sender holds no page of the plan): answered by appendix item 2, "Its room is
  fixed at compile time, by the self-timed buffer bound". Bhattacharyya, Sriram and Lee (IEEE Trans. Signal Processing
  45(6), 1997) give the bound for an IPC edge on a cycle, Bfb(e) (§4.3, (5)), with BBS. A feedforward edge has none
  ("could be unbounded when the execution time of A is less than that of B", Fig. 2(a)), and is put on a cycle at
  compile time by Convert-to-SC-graph (§9, Fig. 6) and DetermineDelays (Fig. 8). The added edges' sinks wait on a
  rising count. The graph algorithms are rdma/mesh_graph.py's.
- **The documents' errors listed above:**
  - Item 5's "adds no word": corrected (RC-1 cited).
  - A7's static-system premise and item 2's derivation: corrected by the two answers above.
  - A7's map serial in m: replaced by the view's identifier, which every member compares.
  - A7's rdma/mesh.h for the program's nodes: replaced by the plan call's `program`, the nodes the launcher started the
    program at, which the launcher passes as it passes MESH_GROUP. That makes PS-1's "the placement's nodes" wrong for
    the torch path.
  - A4 against the barrier: every run is launched on the whole collective, so every node of the map, relays among
    them, is a node of the program and holds the plan (A4, A7). PS-2's EINVAL then arises only where a launcher passes
    fewer nodes than the map's.

## The revisions of 3f106747

The record of the unit whose instruction is design/plan-to-bridge-change-request.md and
design/aot-compilation-change-request.md as mesh 3f106747 holds them, from mesh 3f106747 and metal-microbench 27093ce3.
Its rules are design/disjoint-pages-revision-coordination.md, "Rules (every sub-agent)", but its reading order and its
file table, as the previous part's were. What the code did not yet do was found from the previous part's two stops and
the coordinator's "Answered in the documents", and from what 3f106747 revised: A7, "The island is a dynamic group, and
each of its views is a static system", whole; A7's sender(m) and A4 (the program's nodes the launcher's operand, every
relay one of them); appendix item 2, "Its room is fixed at compile time, by the self-timed buffer bound"; item 5's text
against RC-1. Each decision taken where the documents are silent is marked as the implementer's (VW, BS, PG the
coordinating implementer's; G-n sub-agent G's) and is not a requirement of the documents. Sub-agent G was the one owner
of rdma/mesh-graph.c; the rest is the coordinating implementer's. Nothing was run but the compilers, `python3 -m
py_compile` and the three integration builds.

### Done

- **VW. The island a dynamic group, each view a static system** (A7: "The view properties the code must have are G. V.
  Chockler, I. Keidar and R. Vitenberg's ... verbatim"; "A process is one start of a bridge"; "The membership service is
  the sequencer ... The sequencer creates a view at its bridge's start and at each move of its map"; "The view
  identifier is Transis's"; "A bridge installs a view when the view's first message reaches it"; "Fig. 5 runs within
  each view as written"; "A change of view blocks and flushes"; "A plan stands only within the view its nodes' arrivals
  were counted in").
  - rdma/mesh-map.h: struct mesh_record `view`, the bridge's counter (MESH_VERSION 115), which its record, its
    announcement of itself, floods; rdma/mesh-records.c record_links_same compares it, so a risen counter is written
    into the map with no serial moved.
  - rdma/mesh-records.c: `installed` (the view installed here, its creator's node id and start, `view` the counter);
    island_set at the sequencer calls view_create: identifier one more than the greatest counter among the members (the
    nodes the map reaches; the members' records' counters and its own), `last` seqnum - 1 where it created the view it
    has installed, else 0, then "seqnum := 1", its VIEW down the island's tree, installed here; view_heard (from the tree
    parent alone, as an ORDER) relays and installs; view_install: Self Inclusion and Local Monotonicity tested, the old
    view flushed to `last` where it came from the same process (orders_deliver(last)), "nextdeliver := 1; pending := ∅",
    the arrivals held let go, the counter the view's, hdr `map_serial` moved while `map` is odd and plan_stale, so every
    plan of the old view ends with ESTALE, then record_changed (the counter announced).
  - Every message carries its view's identifier: rdma/mesh-bridge.h struct mesh_order `view` where `serial` was,
    struct mesh_arrival `view`, struct mesh_view, struct mesh_count; order_sequence numbers only its own installed
    view's SUBMITs ("it numbers no further message of the old view"), order_take holds only the installed view's ORDERs
    (Sending View Delivery), arrivals_apply counts only the installed view's arrivals; orders_submit, orders_announce and
    counts_tell send nothing before the first view (Initial View Event).
  - hdr `delivered` and each delivered entry's `sequence` are `passed`, the node's delivery order over its views (VW-4).
  - rdma/mesh-ports.c takes VIEW and COUNT to order_heard. Commit 7b3afe21.
- **BS. Each queue's room its self-timed buffer bound** (appendix item 2: "The compilation builds the program's IPC graph
  from A10's analysis as BSL's §4.1 builds one"; "On a cycle, the queue's room is BSL's self-timed buffer bound";
  "Feedforward ... The compilation adds synchronization edges by BSL's Convert-to-SC-graph (§9, Fig. 6) and sets their
  delays by DetermineDelays (Fig. 8)"; "An added edge (vj, vi) holds "start(vi, k) ≥ end(vj, k − delay((vj, vi)))" ...
  Its source raises a count that only rises ... and its sink reads that count before its k-th start.  The bridge
  carries the count from source to sink, as it carries an arrival"; "A plan whose bound cannot be found fails with
  `ENOMEM` when it is planned ... where a cycle has no delay (BSL's Lemma 1: deadlock)").
  - rdma/mesh-graph.h: mesh_graph_components (Tarjan), mesh_graph_cycle_mean (BSL's BellmanFord, Lawler's search),
    mesh_graph_min_delay (MinDelay), mesh_graph_convert_sc (Convert-to-SC-graph, its edges in DetermineDelays' order
    (e_s, a_1 .. a_k, b_l .. b_1)), mesh_graph_determine_delays (DetermineDelays), each over the caller's vertices,
    edges, delays and times (commit 15b4e9d2); rdma/mesh-graph.c, sub-agent G: Tarjan's search with an explicit
    stack (components_search, one copy), the zero-delay deadlock test by it over the delay-0 edges, Bellman and Ford's
    positive-cycle test under time[from] - λ delay, Lawler's bisection, MinDelay's binary search over a copy of the
    delays, Convert-to-SC-graph's chains and DetermineDelays' loop over one edge array, reusing mesh_graph_rows and
    depth_head (commit 0bbdc5f2); the room of Convert-to-SC-graph's edges 2 x vertices (G-3, commit a91e0e82).
  - rdma/mesh-collective.c plan_rooms: the plan's part of the program's IPC graph, σ (the program's submission at the
    sender s), β (s's bridge's TO-broadcast), x_v (node v's part of the call, each node of the plan); σ -> β and β ->
    x_v of delay 0 (the two queues), σ -> x_s and x_s -> σ of delay `outstanding` (or σ -> σ where s is no node of
    the plan), x_v -> x_v of that delay elsewhere, each tree edge carrying pages both ways of delay u; times from the
    weighting operand (t(σ) = t(β) = 0). ENOMEM where a cycle has zero delay; the added edges and their delays; `room`
    ρ(x_self, β) and `submit_room` ρ(β, σ) by mesh_graph_lags (Theorem 2), each added edge's delay DetermineDelays'
    plus `outstanding` (BS-4); the edge into this node (`waits`, `wait_delay`) and out of it (`tells`) in struct
    mesh_state (320 bytes now). `spanned` computed before the rooms. The abecd389 rooms (`outstanding` + u x hops, and
    `outstanding` for the submissions) are gone.
  - rdma/mesh-dataflow.c mesh_transfers_start: the submissions' queue's room `submit_room`; at a sink, its k-th start
    waits until `wait_count` reaches k - `wait_delay`, bounded by MESH_PLATFORM_NS (ETIMEDOUT), ESTALE where the plan's
    serial moved, nothing raised in either case.
  - rdma/mesh-records.c: counts_tell at the source (each plan's `submissions` toward its edge's sink where it rose past
    `told`, and on every liveness turn), count_take at the sink (`wait_count` raised, its waiter woken); tree_toward and
    tree_down, the tree's routes one copy, where order_relay, arrival_up and arrival_down each held one. rdma/mesh-flow.c
    plan_cancel raises a cancelled plan's `wait_count` to UINT64_MAX and wakes it. Commits 7b3afe21, 6a1c6d3f, 069349d7.
- **PG. The plan call's `program` the launcher's** (A7, sender(m): "The program's nodes are the nodes its launcher
  started it at, an operand of the plan call (`program`) that the launcher passes as it passes `group` (MESH_GROUP)").
  rdma/mesh_compile.py _program: MESH_PROGRAM, read as _group reads MESH_GROUP, passed by serialize, where the
  placement operand's nodes were (PS-1 reversed). rdma/mesh-collective.h's note: `outstanding` the IPC graph's delay,
  `program` the launcher's. Commits 0a6f4e5e, 6a1c6d3f; metal-microbench e9fd7538 (tools/e2b_replay.m's note, PG-2).

### Found already done (the code at 3f106747 and 27093ce3 does what the document states)

- Item 5: "Each range's landed count of each copy is such a count ... A relay's receive has one too" (RC-1: every
  received range's count words, `counted`, rdma/mesh-collective.c plan_layout and mesh_plan; each receive entry's
  `count`; rdma/mesh-flow.c link_receive_drain's round test on it, plan_end's and plan_cancel's raises).
- A7, sender(m), for e2b: tools/e2b_replay.m passes its ranks' map indices 0 .. R - 1, the nodes its launcher started
  it at (PG-2).
- A4: "Every node of the map is a node of the program ... so every relay holds the plan": with `program` the
  launcher's every node, mesh_plan's EINVAL for a node of the plan outside `program` (PS-2) does not arise for a run on
  the whole collective (the torch path's case below, "Where the documents are wrong").

### Decisions taken where the documents are silent (each the implementer's, with its reason)

- **VW-1 (A7, flush).** The flush delivers the old view's messages this node holds, in Fig. 5's order, while each is
  the next and numbered at most `last`; one not yet arrived is never delivered. Reason: a VIEW and its view's ORDERs
  travel one tree, each hop in order, so the old view's messages ahead of it on an unchanged path arrive first; on a
  path the map's move changed nothing bounds when one would, a wait for it would be unbounded, and installation ends
  the old view's plans with ESTALE anyway.
- **VW-2 (A7, a plan within its view).** Installation moves hdr `map_serial` (while `map` is odd) and calls plan_stale,
  and no plan holds its view's identifier. Reason: A7: "Installing a view ends every plan of the old one with `ESTALE`,
  as D.3.6 ends a plan at its map's move"; every reader of a plan's serial (the ports' loads and cancellations, the
  start, mesh_plan's second map read, the programs' own tests) then ends the old view's plans by one path, and a plan
  standing on the serial was made in the installed view, so an arrival or a SUBMIT of it carries that view. The cost:
  a program that replans after its map moved and before the VIEW arrives plans again at the installation.
- **VW-3 (A7, the view's first message).** A VIEW of its own, sent at the view's creation, relayed down the island's
  tree from the sequencer as an ORDER is and taken from the tree parent alone, its members by node id. Reason: the
  arrivals and the program's first SUBMIT need the view installed before any ORDER of it (Initial View Event), and each
  node's map indices are its own.
- **VW-4 (A7 and item 2's merge).** A delivered entry's `sequence` and hdr `delivered` are the message's place in this
  node's delivery order over every view it installed (`passed`). Reason: the ports merge plans' entries by that number
  against `delivered` (lane_take), sn(m) begins again at 1 each view, so it would set the key back; within a view the
  order is sn(m)'s, the same at both ends of a cable, and the old view's plans are cancelled at installation.
- **VW-5 (A7, the announcement).** The counter travels in the bridge's record (struct mesh_record `view`), compared by
  record_links_same so a risen counter is written into the map without moving its serial. Reason: the record is the
  bridge's announcement of itself, flooded to every node and so to the sequencer.
- **VW-6 (A7, a process).** A process is (node id, hdr `boot`), `boot` the bridge's start on the wall clock. Reason:
  hdr `boot` is "the bridge's start on the wall clock (ns, CLOCK_REALTIME), its records' boot", new at every start, as
  Isis's incarnation number is.
- **VW-7 (A7).** An arrival or a COUNT of a view other than the one installed is let go, retold on its node's liveness
  turns. Reason: "the program's sender counts only the arrivals of the view it has installed"; installation lets the
  held ones go in any case.
- **BS-1 (item 2, "Where they live").** The five graph functions are the graph module's C half (rdma/mesh-graph.h,
  mesh-graph.c), and rdma/mesh_graph.py gains none. Reason: their one caller is mesh_plan, which alone holds the plan's
  routes (its trees, mesh_graph_flow's) and u (C3.3's allocator, plan_unroll) at planning; a Python copy no code calls
  would be a second copy of each primitive (C9). The coordinator's brief allows it: "If the C side needs any, they go in
  rdma/mesh-graph.h".
- **BS-2 (item 2, the graph's actors).** The program's submission σ and its bridge's TO-broadcast β are two actors at
  the sender, so the submissions' queue between them (PS-3) is an IPC edge with its own bound, `submit_room`; t(σ) =
  t(β) = 0. Reason: PS-4's room, `outstanding`, rested on the sender's call not completing before its submission is
  broadcast, which fails where the sender holds no page of the plan (its part done at once), stop 2's case; A3 weighs
  neither, and with both 0, σ, the lowest of the least, is the source SCC's link, so the sink-source edge's sink is the
  program's start, a point in control flow.
- **BS-3 (item 2, "the program's IPC graph").** The graph is the plan's part of the program's IPC graph: its own
  actors, queues and transfers, with the program's sender and A10's bound. Reason: each compiled function is planned at
  its first call (PyTorch 2 §3.1), before the program's later functions have trees or u, and the hops and u are
  mesh_plan's; every cycle C containing e bounds the tokens on e by Delay(C) (BSL Lemma 2), so (5) over a subgraph's
  cycles bounds it over the whole graph's, an upper bound, never too small.
- **BS-4 (item 2, the added edge's count).** A source's count is its node's program's starts of the plan
  (`submissions`), carried by COUNT, and the sink waits in its own program's start (mesh_transfers_start) until it
  reaches k - δ; in the rooms each added edge's delay is δ + `outstanding`. Reason: a node's part of a call ends at its
  sends' completions and landings, which only its port children see, and the program's starts are a rising word in the
  region at every node; a start of the plan follows the node's part of the call `outstanding` starts before (the delay
  on its own edge "from a call's completion to the program's later submission"), so the constraint the count gives is
  BSL (7) with δ + `outstanding`, never more than the bound assumes; the wait ends within MESH_PLATFORM_NS (ETIMEDOUT),
  or at ESTALE where the plan is cancelled, which raises the count (plan_cancel), so it ends "when its source moves on,
  or when the view ends".
- **BS-5 (item 2).** A plan with no declared transfer has the graph's nodes {s} at every rank, and a node of the
  program that is no node of the plan has `room` 1. Reason: such a plan's `nodes` is each rank's own (mesh_plan's
  fallback), so every rank builds one graph only this way; no message reaches the queue of a node outside Dest(m).
- **BS-6 (item 2, "The IPC edges are the plans' transfers, each with C3.3's u copies as its delay").** Each tree edge
  y -> c carrying a demand's pages (`child`) is two IPC edges, x_y -> x_c and x_c -> x_y, each of delay u; t(x_v) is v's
  task weights in one execution plus each page it sends or receives at the demand's greatest page weight. Reason: the
  earlier derivation's u on each hop, both ways (the item's correction: it "holds only where the queue lies on a cycle
  through the sender", its only feedforward case "where the program's sender holds no page"); BSL §4.1, "t(v) includes
  the time taken to execute all IPC operations (sends and receives)"; GO2's greatest. Under mesh_plan's premise (each
  node's events of execution t before those of t + 1) a node's take of a call trails its neighbour's by one call, at
  most u.
- **BS-7 (item 2).** One added edge at most enters and leaves each node's vertices, so struct mesh_state holds one edge
  each way. Reason: σ reaches every vertex, so its SCC is the only source SCC and Convert-to-SC-graph chains the sink
  SCCs alone, each link its node's one vertex, the sink-source edge entering σ (plan_rooms' note).
- **PG-1 (A7, sender(m)).** The torch path's launcher passes the program's nodes as MESH_PROGRAM, map indices apart by
  commas; mesh_compile.py refuses it missing or malformed, as MESH_GROUP. Reason: A7's "that the launcher passes as it
  passes `group` (MESH_GROUP)"; mesh_compile.py reads MESH_GROUP from its environment.
- **PG-2 (A7, sender(m), e2b).** tools/e2b_replay.m keeps its ranks 0 .. R - 1, and tools/mesh/run_pair.py is not
  changed. Reason: run_pair.py starts rank r at map index r for every node of the collective and refuses a directory
  not prepared for that many ranks, so those ranks are the nodes it started the program at; launch scripts are outside
  this unit's files (the rules).
- **G-1 (sub-agent G, mesh_graph_cycle_mean).** The mean returned is the bisection's upper end plus |V| τ. Reason: the
  tolerant test can miss a cycle of length up to |C| τ, so the search ends just below the true maximum, and a graph
  with edges added then fails at its own critical cycle (MinDelay ERANGE where BSL's answer exists); the margin makes a
  tie at λ_max pass, the mean too high by at most about |V| τ.
- **G-2 (G, the test's tolerance).** τ = 1e-9 x Σ time, an improvement of at most τ not counted. Reason: at λ_max the
  critical cycle's length is 0 in exact arithmetic, and the rounding of a sum of |V| terms (about |V| 2^-53 Σ time) is
  far below τ; scaled by the times, whose unit is the weighting operand's; 0 where every time is 0, the test exact.
- **G-3 (G's finding, the coordinating implementer's ruling).** Convert-to-SC-graph's room is 2 x vertices, where the
  header said `vertices` and G's code returned ENOMEM past it. Reason: Theorem 4's proof, "(n src + n snk – 1)",
  reaches 2|V| - 1 where SCCs of one vertex are both source and sink; with the room no count passes it, so the test
  goes (compile time, not run time). mesh_plan's graphs have one source SCC and add at most |V| - 1.
- **G-4 (G).** EINVAL where a time is negative or not finite, their sum not finite, or MinDelay's λ negative or not
  finite. Reason: the operands' test, as mesh_graph_completion_times tests its lengths.
- **G-5 (G).** mesh_graph_cycle_mean's 0 where no cycle is positive at λ = 0 (no cycle, or none of positive time); at
  most 200 bisection steps, or until the midpoint no longer moves (mesh_graph_fill's form); MinDelay tests r = bound
  first (ERANGE), then at most 65 tests; DetermineDelays computes λ_max even where nothing is added (EDEADLK for a
  zero-delay cycle), d_ub by an exact ceiling, its delays written once every δ is found. Reason: bounded loops, the
  header's statements, every party the same result. A graph with no cycle and positive times has λ_max 0 and MinDelay
  ERANGE; mesh_plan's graphs always have cycles (σ's, and every x_v's of delay `outstanding`).

### Where the work stopped

- Nowhere: each point the documents leave open took a minimal decision above.

### Where the documents are wrong, or contradict the code's facts

- Appendix item 2, "Where they live": "The strongly connected components, Convert-to-SC-graph, Bellman–Ford's cycle
  mean and MinDelay's binary search are functions of rdma/mesh_graph.py." The graph they run on needs each plan's
  routes and u, which mesh_plan alone computes, in C, at planning (mesh_graph_flow's trees; plan_unroll's u); so they
  are the graph module's C half (BS-1).
- Appendix item 2, "The compilation builds the program's IPC graph from A10's analysis": the torch path compiles each
  function at its first call (PyTorch 2 §3.1: the custom compile function "is called with each captured graph"), so when
  a plan is planned the program's later functions have no routes; the graph is the plan's part (BS-3), a sound bound by
  Lemma 2, not the program's whole graph's least.
- Appendix item 2, "Execution times t(x) are A3's weighting operand": A3 weighs no program submission and no
  TO-broadcast, which the graph needs as actors (BS-2); they take 0, where BSL's "t(v) assigns a positive integer
  execution time to each actor v" (§4.1).
- Appendix item 2, "Its source raises a count that only rises ... and its sink reads that count before its k-th start":
  the end of a node's part of a call is not in the region at that node (its sends' completions are its port children's
  alone); the count raised is the program's starts, so each added edge holds BSL (7) with its delay plus `outstanding`
  (BS-4).
- A4, "so every relay holds the plan", for the torch path: rdma/mesh_compile.py's IR refuses a node that is no device
  of the placement operand ("map index %d, this node, is no device of the placement operand (A2)"), so a node of the
  program the placement leaves out plans nothing and holds no plan; the sentence holds where the placement operand maps
  a device onto every node of the program, as e2b's does. The documents give no program for such a node.
- A7, "The sequencer creates a view at its bridge's start and at each move of its map", with the counters read from
  the members' records: a member's risen counter moves no map, so a view the sequencer creates before a member's latest
  record reaches it can have an identifier no greater than that member's counter, and that member ignores it (Local
  Monotonicity) until the sequencer's next view. Recorded; no handler (appendix items 3 and 4).
- A7, sender(m), "that the launcher passes as it passes `group` (MESH_GROUP)": e2b's launcher, tools/mesh/run_pair.py,
  passes each rank its rank and a directory prepared for the collective's ranks, no node set; the code's facts make
  those ranks the nodes it started the program at (PG-2).
- The previous part's PS-4 (the submissions' queue's room `outstanding`) rested on the same premise appendix item 2's
  correction names for the delivered calls' queue: it fails where the program's sender holds no page of the plan. Now
  its room is its own bound (BS-2).

### Commits

- mesh 15b4e9d2 (rdma/mesh-graph.h: the five declarations), 7b3afe21 (rdma/mesh.h, mesh-map.h, mesh-bridge.h,
  mesh-records.c, mesh-ports.c, mesh-flow.c: VW and the counts' carriage), 6a1c6d3f (rdma/mesh-collective.c,
  mesh-collective.h, mesh-dataflow.c: BS), 0a6f4e5e (rdma/mesh_compile.py: PG), 069349d7 (rdma/mesh.h: the layout
  sentence), 0bbdc5f2 (rdma/mesh-graph.c: the five functions, sub-agent G), a91e0e82 (rdma/mesh-graph.h,
  mesh-graph.c, mesh-collective.c: G-3), and this record.
- metal-microbench e9fd7538 (tools/e2b_replay.m: PG-2's note).
- The C9 hook (bin/mesh-c9-review.py on each mesh commit) returned no remark on any of them.

### Integration

At mesh a91e0e82 and metal-microbench e9fd7538, both level with origin, the trees clean but for this record and
untracked build artifacts: `make -B -C rdma` 0 (the pinned HiGHS tarball fetched and its SHA-256 checked OK, its
objects built, mesh-flow, mesh-heartbeat, mesh-stat and libmesh.dylib), `make -B -C rdma native-audit` 0, `make
.build/e2b_replay` 0 in metal-microbench, no warning or error in any. Each changed C and Objective-C file compiled alone
first, the lifecycle files (mesh-heartbeat.c, mesh-children.c, mesh-child.c, mesh-lanes.c) unchanged and compiling
against the changed headers; rdma/mesh_compile.py compiles (py_compile; the cache entry mesh_compile.cpython-314.pyc it
made, not there before, removed). Nothing was run.

### What this changes among the earlier parts' open items

- "The revisions of abecd389", stop 1 (Fig. 5 in an island whose bridges begin at different times): built (VW), as the
  coordinator's answer states it.
- That part's stop 2 (the delivered calls' queue's room where the program's sender holds no page of the plan): built
  (BS), and the submissions' queue's room with it (BS-2).
- That part's PS-1 (`program` the placement's nodes for the torch path): reversed (PG); PS-2's EINVAL stays, arising
  only where a launcher passes fewer nodes than the plan's.
- That part's "Replans reuse names": across a view's installation an old plan's arrival is not counted (its view, and
  the old plan's serial, are not the installed ones); within one view, a node whose map moves while the sequencer's
  does not can still tell an old plan's arrival before its own map moves, counted toward the sender's replan.
- "Delivery's liveness" (an ORDER lost on the way): unchanged within a view; at a view's change the old view's
  undelivered messages are never delivered (A7).

### Answered in the documents (2026-11-05, the coordinator, on this part's record)

Verified at mesh a71f5c87 and metal-microbench e9fd7538: `make -B -C rdma`, `make -B -C rdma native-audit` and `make
.build/e2b_replay` 0, no warning; rdma/mesh_compile.py compiles. Each finding of "Where the documents are wrong" is
answered in design/aot-compilation-change-request.md as follows.
- Item 2, "Where they live": the five functions are the graph module's C half, their one caller `mesh_plan` (BS-1
  kept, the text corrected).
- Item 2, "the program's IPC graph": each plan builds its own part when it is planned, and by BSL's Lemma 2 its bound
  is never less than the program's (BS-3 kept, the text states it).
- Item 2, the times of the program's submission and the TO-broadcast: 0, since their work is IPC operations, whose
  time BSL §4.1 counts in the actor that performs them, and A3 weighs neither. The rooms read delays alone (BS-2
  kept, the text states it).
- Item 2, the added edge's count: the program's starts at the source's node, the edge's delay DetermineDelays' plus
  A10's bound (BS-4 kept, the text states it).
- A4, a node of the program the placement maps no device: it compiles and plans each function with no section of its
  own and relays as its routes give it. NCCL: "Collective operations have to be called for each rank (hence CUDA
  device), using the same count and the same datatype, to form a complete collective operation. Failure to do so will
  result in undefined behavior, including hangs, crashes, or data corruption." **Code: rdma/mesh_compile.py's IR
  refuses such a node, and must not.**
- A7, a member's counter that rose after the sequencer counted it: the sequencer also creates a view whenever a
  member's announcement names a view other than its installed one with a counter not less than that view's
  identifier. Each announcement names the process that created its bridge's installed view as well as its counter, so
  a member that installed the sequencer's own view (its counter equal to the identifier) does not trigger one.
  **Code: rdma/mesh-map.h's record and rdma/mesh-records.c's view creation.**
- A7, sender(m), e2b's launcher: run_pair.py starts rank r at map index r on every node, so e2b's program's nodes are
  0 … R − 1 (PG-2 kept, the text states it); a torch program's launcher passes MESH_PROGRAM.

Outside the documents: the forced build took 113.2 s, against the operator's "nothing you're running should take more
than around 8000ms in any situation" (2026-10-31). HiGHS's 216 translation units were compiled one after another by a
shell loop. They now compile at once, one per hardware thread (rdma/Makefile, mesh 579f5b58), and the forced build
takes 16.4 s, HiGHS's part about 11.7 s on 18 threads. Its longest unit is 4.9 s, so the floor on one node is the
compiler time, 176 s under contention. Reaching the bound needs the other nodes' cores, or a smaller source set than
R5's BUILD.bazel globs. Open.

## The revisions of b5a3c6e6

The record of the unit whose instruction is design/plan-to-bridge-change-request.md and
design/aot-compilation-change-request.md as mesh b5a3c6e6 holds them, from mesh b5a3c6e6 and metal-microbench e9fd7538.
Its rules are design/disjoint-pages-revision-coordination.md, "Rules (every sub-agent)", but its reading order and its
file table, as the previous part's were. What the code did not yet do was the two items marked **Code:** in "Answered
in the documents (2026-11-05, the coordinator, on this part's record)" after "The revisions of 3f106747": A4's node of
the program to which the placement operand maps no device, and A7's view at a risen counter. Each decision taken where
the documents are silent is marked as the implementer's (ND, RV) and is not a requirement of the documents. No
sub-agent was started. Nothing was run but the compilers, `python3 -m py_compile` and the three integration builds.

### Done

- **ND. A node of the program to which the placement operand maps no device** (A4: "A node of the program to which the
  placement operand maps no device (A2) compiles and plans each function as every node does, with no section of its
  own. It declares the function's transfers as every rank does, and its part of each call is the relaying its routes
  give it."; NCCL, as A4 quotes it: "Collective operations have to be called for each rank (hence CUDA device), using
  the same count and the same datatype, to form a complete collective operation. Failure to do so will result in
  undefined behavior, including hangs, crashes, or data corruption.").
  - rdma/mesh_compile.py IR: the refusal "map index %d, this node, is no device of the placement operand (A2)" is
    gone, and `device` is None at such a node (ND-1). What the IR builds there is every rank's: the partitioned
    program (_partition reads no device); every device's declared transfers and events, over every group (the five
    lowerings, `_groups`), each declared transfer's `value`, and a layout, this node's only where its map index is an
    end (`transfer`), which at such a node none is; the dependences over every device's events (_dependences); the
    kinds (_consumer, over the program); the weights the weighting function gives over them. So `group`, `function`,
    `program`, `outstanding`, the declared transfers' ends, identities, kinds and bytes, the weights and the
    dependences that mesh_plan takes are every rank's; its `values` are none and each declared transfer's `value`
    MESH_ABSENT, as mesh_plan's contract has each rank pass its own (rdma/mesh-collective.c: "`value`: MESH_ABSENT
    where this node is neither the transfer's source nor its destination"). No lowering gives it a role.
  - serialize: the plan is made there with no value, and each call launches its execution by mesh_transfers_start, as
    at every node (its `executions` and `submissions` raised and its submission queued: its bridge serves the relays
    its routes give it once the ORDER is delivered, its starts are the count of any added edge leaving it (BS-4), and
    where it is the program's sender its bridge TO-broadcasts the submission). The call runs no item of the program
    and returns ND-2's outputs. Commit 80967a66.
- **RV. A view at a risen counter** (A7, "The membership service is the sequencer": "It also creates one whenever a
  member's announcement names a view other than the sequencer's installed one, with a counter not less than that
  view's identifier. ... The identifier is "computed based on the values of local counters" (below), so the sequencer
  computes it again from the counters it now holds. To tell such a member from one that installed the sequencer's own
  view, each announcement names the process that created its bridge's installed view as well as its counter.").
  - rdma/mesh-map.h struct mesh_record: `view_boot` and `view_sequencer` appended, the process that created the
    record's bridge's installed view (the creating sequencer's start, hdr `boot`, and its node id, as mesh-records.c
    struct installed names it), 0 both before its first view, and `padding` (RV-3); MESH_VERSION 116 (rdma/mesh.h).
  - rdma/mesh-records.c: record_changed writes them from `installed`; record_links_same compares them, so a record
    that names another view at an unchanged counter is written into the map; view_announced, at the sequencer alone
    (island.self == island.sequencer), holds of a record whose counter is not less than the installed view's
    identifier and whose view, counter and creator, is not the installed one (RV-2); map_put calls island_set where its
    write moved no serial and view_announced holds of the record it put (RV-1, RV-4), and island_set's view_create
    makes the view, its identifier one more than the greatest counter among the members' records the map now holds. A
    member that installed the sequencer's own view names it whole, its counter equal to the identifier, and creates
    none; nor does the sequencer's own record, which record_changed writes from `installed`. Commit c02318a3.

### Found already done (the code at b5a3c6e6 and e9fd7538 does what the document states)

- A4, "It declares the function's transfers as every rank does": the lowerings declared every device's transfers and
  events at every rank, and _dependences was "every rank's, by the program's order on that rank"; only the IR's
  refusal stood before them.
- mesh_plan takes a node with no value: its operand test is `(given && !values)`, every declared transfer's `value` is
  MESH_ABSENT at a node that is no end of it, and a node of `program` that is no node of the plan has `room` 1 (BS-5);
  `spanned` holds the plan's nodes alone, so its EINVAL past `program` does not arise for such a node.
- mesh_transfers_start: the same at such a node as at every node.
- A7, "the sequencer computes it again from the counters it now holds": view_create computes the identifier from the
  members' counters as the map holds them (VW); and a member installs the new view, flushing the old one where the same
  process created both (view_install, VW).

### Decisions taken where the documents are silent (each the implementer's, with its reason)

- **ND-1 (A4, `device`).** None at such a node. Reason: the lowerings' role tests and _run's coordinates are the IR's
  only readers of it; None is in no group and equal to no root, so the node holds no role, and _run is not run there.
- **ND-2 (A4, what the call returns at such a node).** Each of the graph's outputs as a tensor of its fake tensor's
  size, stride, element type and device, by torch.empty_strided ("Creates a tensor with the specified size and stride
  and filled with undefined data", torch's documentation, read in torch v2.11.0's torch/_torch_docs.py, the source at
  hand), which nothing writes, made once when the plan is made; an output that is no tensor is its example value as
  Dynamo recorded it. Reason: the documents give such a node no value ("with no section of its own") and say nothing of
  what its call returns. A torch.compile callable stands for its graph (PyTorch 2 §3.1), and Dynamo takes its outputs
  apart, so a call returning nothing fails at that node; an output of None reaches the program's next compiled function
  as a constant, which Dynamo captures into another graph, so that function would not be compiled and planned "as every
  node does" (A4). A tensor of the fake tensor's metadata, "the metadata of a tensor without its actual data" (PyTorch 2
  §3.5), gives that capture the shapes and types every rank's gives. Cost: such a node holds each output's bytes for the
  plan's life, where a device node makes its outputs' clones at each call (A-T5). Torch code of the program between its
  calls computes there on undefined data; A10's traced block has no control flow on values (torch.fx: "symbolically
  traced variables cannot be used as inputs to control flow"), so it makes the same calls there.
- **RV-1 (A7, where the view is made).** map_put calls island_set, the one path to view_create, not view_create
  itself. Reason: A7's members are "the nodes the sequencer's map reaches", which island_set computes; with the map
  unmoved it computes the island of the last move again, and its other work (the arrivals held let go) is
  view_install's too.
- **RV-2 (A7, "names a view other than the sequencer's installed one").** Two views are one where their identifiers
  and their creators (node id and start) are equal. Reason: A7's "each announcement names the process that created its
  bridge's installed view as well as its counter", for a counter equal to the identifier, where only the creator tells
  the views apart.
- **RV-3 (A7, the record's fields).** `view_boot` (64 bits), `view_sequencer` (32) and `padding` (32, 0) appended after
  `view`. Reason: "The record is only appended to" (the brief; mesh-records.c record_heard: "struct mesh_record is only
  appended to"); `padding` makes the tail a field every writer sets, as struct mesh_order's `padding` (mesh-bridge.h),
  so a record told on the wire carries no unset byte.
- **RV-4 (A7, when the test runs).** On the record map_put has just written, at that write. Reason: "whenever a
  member's announcement names ...": an announcement is one record heard (record_heard), and a change of its counter or
  creator is one record_links_same sees, so map_put writes it; a view the sequencer creates at a move already reads
  every counter the map holds.

### Where the work stopped

- Nowhere: each point the documents leave open took a minimal decision above.

### Where the documents are wrong, or contradict the code's facts

- A4, "its part of each call is the relaying its routes give it": A7 makes the program's sender "the least of the
  program's nodes", which can be a node the placement operand maps no device (map index 0, where the placement maps
  devices onto other nodes alone). There its part of each call is also the program's submission, which its bridge
  TO-broadcasts (A7, sender(m)). The code does both by the one start (mesh_transfers_start).

### Commits

- mesh c02318a3 (rdma/mesh.h, mesh-map.h, mesh-records.c: RV), 80967a66 (rdma/mesh_compile.py: ND), d0b816b0
  (rdma/mesh_compile.py: ND-2's note names the torch source it quotes), and this record.
- metal-microbench: none; no item names its files, and e9fd7538 builds against the changed headers.
- The C9 hook (bin/mesh-c9-review.py on each mesh commit) returned no remark on any of them.

### Integration

At mesh 80967a66 and metal-microbench e9fd7538, both level with origin (d0b816b0 after it a comment alone), the trees
clean but for this record and untracked build artifacts: `make -B -C rdma` 0 in 15 s (the pinned HiGHS tarball fetched
and its SHA-256 checked OK, its objects built, mesh-flow, mesh-heartbeat, mesh-stat and libmesh.dylib), `make -C rdma
native-audit` 0, `make .build/e2b_replay` 0 in metal-microbench (libmesh_metal.dylib and e2b_replay rebuilt against the
changed headers), no warning or error in any; codesign's "Mesh Bridge: no identity found" is the Makefile's ad hoc
fallback, not a compiler's. Each changed C file compiled alone first, and the lifecycle files (mesh-heartbeat.c,
mesh-children.c, mesh-child.c, mesh-lanes.c), unchanged, compiled alone against the changed headers;
rdma/mesh_compile.py compiles (py_compile; its cache entry was there before). Nothing was run.

### What this changes among the earlier parts' open items

- "The revisions of 3f106747", "Where the documents are wrong": A4's refusal of a node the placement leaves out, built
  (ND); A7's risen counter, which moved no map so no view followed it, built (RV).
- That part's PS-2 EINVAL (a node of the plan outside `program`) is unchanged; a node the placement leaves out is in
  `program` for a run on the whole collective.
- "Outside the documents", the forced build's time: unchanged by this unit, 15 s here.

### Answered in the documents (2026-11-05, the coordinator, on this part's record)

Verified at mesh 2c23692b: `make -C rdma`, `make -C rdma native-audit` and `make .build/e2b_replay` 0, no warning;
rdma/mesh_compile.py compiles. RV and its view_announced test meet A7 as written. A member that installed the
sequencer's own view names it whole, so no view follows its installation, and the sequencer's own record never
triggers one. ND meets A4.
- ND-2 reversed. `torch.empty_strided` on the function's device is memory outside the region, which the region-only
  requirement forbids. A4 now states the outputs at such a node: tensors on PyTorch's meta device, which "records only
  metadata, but no actual data".
- The documents' finding, that a node with no device can be the program's sender, needs no text. A7's sender
  broadcasts at its own start, as the code does.
- Outside this part, the torch path's own copies and allocations. The operator asked "is this a result of strictly
  reusing an upstream module or inlining?" They are rdma/mesh_compile.py `_run`'s, not torch's:
  - functional forms that allocate each result;
  - `write`'s `copy_` between two full device synchronizations;
  - `torch.clone` of each placed output.
  Appendix item 7 now states the out-variant form (ExecuTorch's out variant and memory planning, the planner C3's one
  allocator) and what stays, being torch's.
- The build. Every unit had been verified with `make -B`, and HiGHS's stamp had the whole Makefile as a
  prerequisite, so any edit of it rebuilt all 216 objects. The operator: "stop arbitrarily rebuilding unchanged
  dependencies as part of linking or assembling whatever runtime our compiler uses". Now (mesh 057174ba) the objects
  are remade only where their tarball, tag or flags change, and builds are verified incrementally.

## The revisions of e4a7398e

The record of the unit whose instruction is design/aot-compilation-change-request.md's appendix item 7, "What the torch
path adds beyond torch's own work, and removes", and its A4 line on a node the placement operand maps no device, from
mesh e4a7398e and metal-microbench e9fd7538. It follows item 7 as mesh 9c8c8865 revised it during the unit ("A publish
waits for nothing", "No output is copied") and as 0ffd880e revised it again (allocator.mm's event form cited alone).
Its rules are design/disjoint-pages-revision-coordination.md, "Rules (every sub-agent)". Each decision taken where the
documents are silent is marked as the implementer's and is not a requirement of the documents. No sub-agent was
started. Nothing was run but the compilers, `python3 -m py_compile`, the three integration builds (incremental) and the
extension's build against /Users/mdot/.venv-mesh-uv's torch 2.14.0, which the coordinator allowed. Torch claims below
were read in torch 2.14.0's sources and headers there (lib/python3.12/site-packages/torch, torchgen), not v2.11.0's.

### Done

- **OV. Each operator as its out variant** (item 7: "run each operator as its out variant into its result's placed
  pages"; ExecuTorch, "Concepts": "an operator's out variant will take in a pre-allocated tensor to its out kwarg").
  rdma/mesh_compile.py `_forms` runs each operator item on PyTorch's meta device (A4: "Most operations can be performed
  on meta tensors"), takes the ATen overload the call reaches (`_overload`) and its out variant by torch's own
  `torch._library._out_variant.to_out_variant` ("Given a functional operator overload, return its corresponding out
  variant"; inductor's use: "Try to lower single output functional custom ops to their out-variant", torch/_inductor/
  ir.py). `_run` calls that OpOverload with `out=` the result's placed pages. The program's own operators are out
  variants too: the AllReduce's and ReduceScatter's sums `torch.add(..., out=)`, the AllGather's and AllToAll's
  `torch.cat(..., out=)`, a halo region `torch.cat(..., out=)` of the narrows it covers and `fill_` of the parts outside
  the operand, the CollectivePermute's zeros `zero_()` of its pages. Commit 63638057.
- **PV. Every value placed by the one allocator** (item 7: "place every value of the node's partitioned graph by the one
  allocator. A view (a narrow, a slice) is its base's pages, and its base lives as long as it does"; C3.2, C3.3).
  - `_over` gives each key the set of keys whose pages it may lie over, a view its base's, an out-variant result its
    own. It replaces `placed`, which held an AllReduce's result and its slices only, and `_readers` and `_consumer` read
    it.
  - `_place` makes every key with pages of its own, and no range's, a local value of mesh_plan's allocator
    (MESH_EXECUTION, `wire` 0, `begin` its producer's item, `end` its last reader's), a reader of a view being its
    base's reader. `IR.value` is the one constructor of a routed or local value and its layout.
- **Sent values made in their sent pages** (item 7: "have each sent value's producer make it in its sent pages, so
  nothing copies it there"):
  - a member's AllReduce partial whose producer runs as its out variant, published after that item;
  - the root's sum, made by `torch.add`'s out variant in its sent pages;
  - an AllGather's operand whose producer writes pages of its own and which no other collective reads (`inplace`). Its
    production event moves to its producer's third step, and its reads there are the source's read
    (MESH_FROM_SOURCE_READ, `_all_gather`).
  Every other sent value is copied once (where the work stopped, below).
- **A collective's result is its parts' landed pages** (item 7: "A collective's result is the pages its parts land in"):
  the AllReduce member's result as before, and the CollectivePermute's result its arrival's pages, where `torch.clone`
  copied them. Its consumption event now spans its readers (`_collective_permute`).
- **TP. A publish waits for nothing** (item 7 at 9c8c8865: "a torch MPS event is recorded after the producer's
  out-variant kernel, the publish is done in that event's notification, and torch's command buffer is committed
  without waiting"; "On the CPU device ... the publish follows the call").
  - rdma/mesh-torch.mm `mesh_torch_publish` calls torch's own API in fba41159's event form: `acquireEvent(false,
    stream)`, `record`, `notify` with the publishes, `synchronize(SyncType::COMMIT)`. Each is as torch 2.14.0's
    MPSEvent.h and MPSStream.h state it, and each symbol is in its libtorch_cpu.
  - rdma/mesh-torch.mk builds libmesh-torch.dylib against TORCH, an operand (no venv path in source).
  - `_publish` publishes CPU values by mesh_host_publish and MPS values in one notification. No `synchronize()` is left
    on the publish path. `write`'s two synchronizations are gone. Commit 6b06a9c4, 63638057.
- **NC. No output is copied** (item 7 at 9c8c8865: "Each output is a view of its placed pages and holds until the plan
  reuses them"): the output item returns each key's tensor, and no `torch.clone` is left in the compiled path.
- **MD. A4, the meta device** (A4: "Its calls return tensors on PyTorch's meta device"): `_absent` makes each tensor
  output `torch.empty_strided(..., device='meta')`, where it used the function's device (ND-2, reversed). `_value` takes
  'meta', so that node's next function, traced from meta tensors, builds every rank's IR. The verification the brief
  asked for:
  - These operands are built at such a node exactly as at every rank: the declared transfers (ends from the placement
    and the map; identities in lowering order; kinds by `_consumer`, which reads targets and local shapes; bytes from
    element type and local shape, which a meta tensor carries alike) and the dependences (positions from the program,
    the forms by `_forms` on the meta device at every rank, `inplace`).
  - So are `group`, `program` and `outstanding` (A10's `_analyse` reads no device).
  - `layouts` and `values` are this node's and empty there, and `IR.value` refuses a meta tensor at a device of the
    placement.
  - `function`, the ordinal, is not, where the device changes between calls (below, documents).
- **C9.** The hook's remarks on 63638057 ("a topological pass over the program DAG ...", "last-use liveness ...") were
  met by rdma/mesh_graph.py `forward` (a forward data-flow problem's solution in one pass over a topological order) and
  `ends` (live intervals' ends over a linear order), with `_over`'s rule and `_place`'s uses as operands. Commit
  a3ab3044. The hook made no remark on a3ab3044 or 6b06a9c4.

### Found already done

- mesh_plan places a local value: MESH_EXECUTION, `begin <= end` "on this node's own program order", sharing pages
  only where those intervals do not meet (mesh-collective.h struct mesh_value; mesh-collective.c plan_interferes). No C
  change was needed to place every value.
- A received range is read only after mesh_host_arrived reports it (K1), and the AllReduce member's result was its
  landed pages.

### Decisions taken where the documents are silent (each the implementer's, with its reason)

- **OV-1 (`_call`).** _run's three forms of a call are one function, which `_forms` calls on meta tensors. Reason:
  one implementation of each computation (D4).
- **OV-2 (which overload a call takes).**
  - The ATen name is the call's own: a Tensor method's name; an operator module function's by OPERATORS; a function
    held by torch's generated bindings (torch._C._VariableFunctions, _nn, _linalg, _special, _fft; torch/__init__.py:
    `__obj = getattr(_C._VariableFunctions, __name)` ... `globals()[__name] = __obj`), its __name__.
  - The overload is the first whose schema takes the call's arguments, by
    `torch._C._check_schema_allow_fake_script_object`. That is torch's own resolution of a packet's call (torch/_ops.py
    `_call_overload_packet_from_python`) and its ONNX pass's `find_compatible_op_overload` (example: `(tensor, 2)` gives
    'Tensor_Scalar').
  - Reason: the out variant is defined on ATen overloads (`to_out_variant` takes an OpOverload), and Dynamo's graph holds
    torch-level calls.
  - Python functions that are no binding (F.relu, F.silu, F.max_pool1d to 3d) resolve to none.
- **OV-3 (an operator whose result may alias).** An overload whose return has an alias set (a view), or no overload, or
  no meta kernel, or a module the program calls: its result is taken to lie over every operand's pages. Reason: "its
  base lives as long as it does" (item 7) with no knowledge of which operand is the base.
  - As a consequence, the readers of an AllReduce's result and their spans now follow views made by operators, where
    `placed` followed `_reshard`'s and `_region`'s alone. The kinds are unchanged, since a view operator is no
    elementwise reader.
- **OV-4 (forms at every rank).** Forms are computed on the meta device at every rank, a node with no device among
  them, since `inplace` moves AllGather events that every rank declares.
- **PV-1 (local values' positions).** They are item indices. An item's reads and its write share one position, so an
  out variant never writes pages it reads. An output's last reader is the output item, so no value of its execution
  takes its pages.
- **PV-2 (one buffer a space).** An MPS value's tensors lie over one MTLBuffer for each space (window, outside) of a
  plan, where each value had its own buffer. Reason: the allocator shares a space's pages among values, and Metal
  orders accesses by each resource's hazard tracking (MTLResource.h; the old note's own reason for `write`'s first
  synchronization). Over one buffer, a write after another value's read of the same page is ordered with no wait.
  - The two spaces share no page.
  - Each is one run of the claim (mesh-dataflow.h struct mesh_claim).
  - torch 2.14.0 reads an MPS tensor at its storage offset in place on macOS 15 or later (OperationUtils.h
    `needsGather`).
- **PV-3 (inputs and module attributes).** They are not placed: they are the caller's and the module's own tensors, and
  placing them would be a staging copy. A sent input-derived value is copied once (below).
- **PV-4 (operators with no out variant).** Their results stay torch's and are placed only where they must be, a sent
  range, copied once there. Reason: item 7, "What stays": "an operator with no out variant, whose result is copied once
  into its placed pages where it must be there"; a local value nothing writes but a copy would be such a copy.
- **PV-5.** A result with no elements runs as it is (mesh_plan refuses a value of 0 bytes).
- **IP (AllGather in place).** Only where the operand's producer writes pages of its own (an out-variant call, a
  concatenated region, an AllGather, AllToAll or ReduceScatter result) and no other collective reads the operand.
  Reason: a value is "one range's alone" (mesh-collective.h), so an operand of two collectives can be made in one
  range only.
- **Regions.** A halo region is `torch.cat(out=)` of the narrows its window covers, then `fill_` of the parts with
  index < 0 or >= size. Reason: that is GSPMD section 4.1's Select by the same mask, with no mask tensor, where
  `masked_fill.Scalar_out` (autogen) would take one.
- **TP-1 (the extension's interface).** It takes mesh_host_publish's address and the context from Python and mirrors
  struct mesh_section, as rdma/mesh.py binds it. Reason: mesh's C headers do not compile as C++ (mesh.h's stdatomic.h
  and infiniband/verbs.h; compiled to check).
  - It is linked `-undefined dynamic_lookup` with `-std=c++20`, as torch 2.14.0's cpp_extension builds on macOS ("The
    darwin linker needs explicit consent to ignore unresolved symbols"). So torch's symbols bind to the program's
    loaded torch.
  - Its makefile is its own (rdma/mesh-torch.mk), so rdma/Makefile, the bridge's prerequisite, is unchanged.
  - EIO where torch throws or schedules no notification.
- **TP-2.** `record(true)` and `notify(true, ...)` take torch's lock at every call. Reason: fba41159 took it where its
  caller was off the stream's serial queue, and this caller is Python's thread.
- **TP-3 (mesh_host_publish).** It raises the supplied word by a compare-and-swap loop that never lowers it, where it
  stored it. Reason: notifications "may come in any order, so the count only rises" (fba41159's message); C3.7. Its
  only caller is mesh_compile.py. Commit 6b06a9c4.
- **TP-4 (retirement).** `_retire`, after its device synchronization, waits out the notifications still to run
  (`mesh_torch_notifying`), for at most MESH_PLATFORM_NS, before it closes the claim. Reason: a notification writes
  its plan's state section, which is the bridge's to release once the claim is closed. That is deferred reclamation's
  grace period (C3.4; McKenney and Slingwine). It is a retirement's wait, not a publish's.
- **TP-5.** libmesh-torch.dylib is loaded at mesh_compile's import, after torch, as mesh.py loads libmesh. A node
  without it fails that import with OSError. No deployment script builds it (they are not changed).
- **NC-1 (how long an output holds).** Until its pages are rewritten: the next execution for a local value (its one
  copy), u executions later for a routed one. Reason: "u executions later (C3.3)" read as the value's own copies. A
  program that passes an output into the next call of the same function, or keeps it longer, must clone it itself, as
  item 7 and CUDAGraph Trees state.
- **MD-2.** `_value` takes 'meta' everywhere, and `IR.value` refuses a meta tensor at a node the placement maps a
  device, where no binding exists (a compile-time refusal).
- **C9-1.** mesh_graph `forward`'s and `ends`' citations (Kam and Ullman's one pass on an acyclic graph; Poletto and
  Sarkar's live interval) are stated from memory, not read in this unit; neither is quoted.

### Where the work stopped

1. **No concatenation for AllGather and AllToAll.**
   - The text: "A collective's result is the pages its parts land in, so no concatenation copies them".
   - The code's facts: a destination's value of a range is "MESH_EXECUTION and wire, its bytes the range's, and one
     range's alone" (mesh-collective.h), and the colouring places each value apart. A torch tensor is one storage at
     affine offsets.
   - So a result over several ranges' pages exists only where those ranges are adjacent in the group's order. No value
     of the plan gives that, and no document states a value made of several ranges.
   - The concatenation stays, as `torch.cat`'s out variant into the result's placed pages.
2. **Parts of a value sent as several ranges.**
   - The text: "have each sent value's producer make it in its sent pages".
   - AllToAll's pieces, ReduceScatter's slices and CollectivePermute's halo pieces are parts of one producer's result.
     An out variant writes one result into one out tensor, and each part's range is a value of its own, placed apart.
   - Each part is copied once into its sent pages, as is an operand of two collectives, an input-derived operand and an
     operand no out variant makes.

### Where the documents are wrong, or contradict the code's facts

- **A4, "its next compiled function is traced with the shapes and types every rank's is".** This holds of shapes and
  types, but not of the compile count.
  - Dynamo guards each tensor input's device: torch 2.14.0 torch/_dynamo/guards.py `extract_tensor_metadata` returns
    `(t.shape, t.stride(), t.dtype, t.device, t.requires_grad)`, "Used by TENSOR_MATCH guard check spec".
  - At a node the placement maps no device, a compiled function called once with the program's own tensors and later
    with an earlier function's meta outputs is compiled twice. Every other rank compiles it once.
  - The second compile takes the next ordinal (FUNCTIONS), so plan names (group, function) differ across ranks from
    then on.
  - Strides are guarded too. A device node's out-variant outputs are contiguous, while the meta outputs take the fake
    tensor's strides.
  - No document states how the ordinals stay alike, so nothing was built for it.
- **A3 at such a node.** A weighting function that reads `ir.values` or `ir.layouts` gets none there. The unit weights
  read neither.

### Operators with no out variant met

Read in torch 2.14.0's torchgen/packaged/ATen/native/native_functions.yaml (explicit `.out` overloads and `autogen:`
out variants), for the ATen function each binding of the partitioner's operator sets names:
- **No out variant:** einsum; conv1d, conv2d, conv3d and conv_transpose1d to 3d (CompositeImplicitAutograd, decomposed
  to convolution, which has an autogen out the call does not reach); max_pool1d, max_pool2d, max_pool3d; pad
  (torch._C._nn.pad, F.pad). Each runs as it is, its result copied once into a sent range where it must be there.
- **Out variants:** add, sub, mul, div, neg, relu (autogen), tanh, sigmoid, exp, gelu, silu, matmul, mm, bmm, linear,
  avg_pool1d (autogen), avg_pool2d, avg_pool3d, cat, clone (autogen), zeros.
- **Views, which make no result of their own:** narrow and getitem.
- **Not resolved:** the Python functions F.relu, F.silu and F.max_pool1d to 3d, being no binding (OV-2). Each runs as
  it is, taken as a possible view.
None was met at run time: nothing was run.

### Commits

- mesh:
  - 6b06a9c4 (rdma/mesh-torch.mm, mesh-torch.mk, mesh-collective.{c,h}: TP, TP-3);
  - 63638057 (rdma/mesh_compile.py: OV, PV, IP, TP, NC, CP, MD);
  - a3ab3044 (rdma/mesh_graph.py, mesh_compile.py: C9);
  - 25410f84 (mesh-torch.mm's note, as 0ffd880e cites allocator.mm);
  - this record.
- metal-microbench: none. e9fd7538 builds against the changed header.

### Integration

At mesh 63638057, with a3ab3044 and 25410f84 changing Python and a comment only after it, and metal-microbench
e9fd7538. Every build was incremental, and none rebuilt HiGHS.
- `make -C rdma` exited 0. mesh-flow and libmesh.dylib were remade at 6b06a9c4, since mesh-collective.{c,h} are their
  inputs. The build took 2.4 s, and a later run was up to date.
- `make -C rdma native-audit` exited 0.
- `make .build/e2b_replay` exited 0 in metal-microbench. libmesh_metal.dylib and e2b_replay were remade against the
  changed mesh-collective.h.
- `make -C rdma -f mesh-torch.mk TORCH=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages/torch` exited 0 in
  1.2 to 1.6 s. Its undefined symbols are torch's MPS API and libc++ alone.
- The lines shown carried no warning or error. The C and Objective-C++ compiles are `-Werror`. codesign's "Mesh Bridge:
  no identity found" is the Makefile's ad hoc fallback.
- mesh-collective.c compiled alone first with `-Wall -Wextra -Werror`.
- rdma/mesh_compile.py and rdma/mesh_graph.py compile (`python3 -m py_compile`). mesh_compile's cache entry was there
  before; mesh_graph's 3.14 entry was rewritten, and whether it was there before is not known (ignored by .gitignore).
- Nothing was run: no form, overload resolution, binding or publish here has been exercised.

## The documents' corrections of 2026-11-05 (the coordinator)

The operator, 2026-11-05: "then correct every point in any change request docuemnt so far that feigns to supervene
upon the literature, the outer documentation and requirements of the project, etc. we now have a clear list of the
points where the change request documenst were externally or self inconsistent and what the corresponding hcanges must
be."

Each point below names where the text was wrong, the source it was wrong against, and what the code must now do. The
corrections are dated notes at each place, in design/aot-compilation-change-request.md (AOT),
design/plan-to-bridge-change-request.md (PB) and design/disjoint-pages-change-request.md (DP). The part "The revisions
of e4a7398e" above is not yet verified against the requirement lines. Its BS-3, ND, ND-2 and `_absent` forms rest on
points 1, 2 and 4, and are superseded with them.

1. **Capture at a function's first call (AOT A1).**
   - Wrong against: the operator's "lazy jit" and "AOT … serialized objects" turns, and torch's docs: `torch.compile`
     "is a JIT compiler" that falls back "to running the program in the eager Python runtime".
   - Origin: Claude's A1 chose a `torch.compile` backend with no requirement for it.
   - The code must: capture each torch program with `torch.export` before it runs, the same at every rank, with nothing
     run eagerly between compiled calls. rdma/mesh_compile.py's Dynamo backend (`serialize`) goes. mesh history's
     engines.py used `torch.export.export` (10-26 to 11-02).
2. **The meta device at a node with no section (AOT A4).**
   - Wrong against: point 1. It rested on eager code between calls and on Dynamo tracing again at that node.
   - The code must: at such a node, start its plans' executions and relay, and compute nothing. `_absent` goes.
3. **"Every program on the island" (AOT A7).**
   - Wrong against: the operator's turns, which give no several-program requirement. Pathways' multi-client island is
     Pathways' setting.
   - The code must: no change. The scheduler orders the one program's executions.
4. **Each plan builds its own part of the IPC graph (AOT appendix item 2).**
   - Wrong against: point 1.
   - The code must: build the program's whole IPC graph at compile time, every function together, with each queue's
     room (5) in it. `plan_rooms`' part graph goes.
5. **A claim per plan, placed by run-time first fit, and several programs at once (AOT appendix items 2 and 6; PB R8;
   DP A2, C3 item 4, D.3 item 1).**
   - Wrong against: the operator's turn 394 ("allocation known aot … do you think we're supporting dynamic runtime
     allocations"), and DP A2's own classification of first fit as the run-time strategy to replace. "Several programs"
     has no source but Claude's text.
   - The code must: place the whole program's values in one offline allocation at compile time. These go:
     - `mesh_claim_take`'s first fit and the claim's head record;
     - `mesh_plan_head`'s walk of every claim head in `mesh_transfers_start` and `mesh_plan_retire`, a run-time check;
     - `ENOMEM` from fragmentation.
     Release at retirement stays deferred reclamation (RCU). The program's group still names its plans, as NCCL's
     `ncclUniqueId` names a communicator.
   - Two things the coordinator's brief for the next unit called open are answered by the documents. The operator, of
     that brief: "but both of these sound idiomatic instead of citational and there are no groundings to literature,
     data structures, or algorithms? are they even questions?" They are not.
     - **The plan call's form for a whole program** is A8 and A9. A8: "The compilation must carry one IR from
       partitioning to serialization, as an operand of its own", its type R3's task graph, R4's kinds, and the
       placement and weighting operands. A9: the serialized operand, per node, in MSCCL-IR's shape. With the program
       exported whole (point 1), there is one IR of the program and one serialized operand of it.
     - **How the bridge finds a program's plans** is A9 and R1.
       - What the bridge runs is the self-timed schedule (A9, "What the bridge runs", 2026-11-05). It is Lee and Ha's
         strategy as BSL states it: "Each processor executes the actors assigned to it in the order specified at
         compile time. Before firing an actor, a processor waits for the data needed by that actor to become
         available". Its serialized form is MSCCL-IR's.
       - Every address in it is fixed at compile time, as ExecuTorch's memory planning "assigns a memory ID to each
         tensor and an offset in the buffer" and as CUDA graph trees keep "memory is always in the same location as it
         were recorded".
       - R1's method is U-Net's and VIA's queues of descriptors over registered memory, which the interface knows from
         the endpoint's creation; it does not search for them.
       - So the serialized operand's place is a constant the region's layout names once, as the bridge's own pages are
         "laid out once with the region" (disjoint-pages A3). Nothing walks heads to find it.
       - The operator, of "The bridge consumes the lists" and "every address is a constant in the plan" stated without
         these: "sounds idiomatic. citations? literature? algorithms? data structures?"
6. **Every value one run of pages (PB R2 and R4; DP C3.3).**
   - Wrong against: the operator's typing, "function B requires set of pages operand_b to be complete but not
     contiguous" and "domain:Async,noncontiguous,incomplete"; and the turn "collective functions only 'know' about
     pages".
   - The code must:
     - Represent a value as its ordered page-table rows (Kilburn et al.; Denning; PagedAttention's block tables, Kwon et
       al., SOSP 2023).
     - Place every value's pages anywhere, by the fewest chains of the interference order. That is Dilworth's theorem,
       computed by Fulkerson's bipartite matching, a function of rdma/mesh-graph.h or mesh_graph.py with caller
       operands.
     - Give an operand typed as one run (the operand of R4's synchronous, complete, contiguous consumer, or a torch MPS operator's operand, item 7) its run as a
       mapping written when the plan is made: consecutive virtual pages over its pages, as `ncclMeshPageTableCompose`'s
       `mach_vm_remap` view was on 2026-11-02 (PB R2).
     - Greedy by size goes, with `mesh_graph_left_edge`'s colouring.
7. **A received range is a value of its own, "one range's alone" (rdma/mesh-collective.h; the plan call of
   design/disjoint-pages-revision-coordination.md).**
   - Wrong against: NCCL's receives at offsets of one result ("receiving data from rank `i` at offset
     `i*sendcount`"). The contract cites nothing.
   - The code must: name, in each declared transfer, the pages of its destination value that it lands in. A
     collective's result is then the pages its parts land in, so no concatenation and no copy of a part remains (item 7;
     the stops of "The revisions of e4a7398e").
8. **Torch MPS operands (AOT appendix item 7).**
   - Added: MPSGraph takes an `MTLBuffer` with a shape and an optional row stride, and torch uses native strides from
     macOS 15.0. So such an operand is typed one strided progression (point 6).
   - The code must: the IR types each such operand, and the plan writes its mapping (point 6).
9. **"Lease" (AOT appendix item 7; the deleted allocator.mm).**
   - Wrong against: the coordination's rules ("No hashing, receipts, credits, gates, filters, locks, leases …"). The
     canonical lease is a grant limited in time (Gray and Cheriton, SOSP 1989).
   - The code must: no change. Item 9 (torch's own allocations in the region) is not written.

Open, outside these points:
- The forced HiGHS build's 11.7 s against the eight-second bound.
- Torch's own allocations (kernel scratch, and results of operators with no out variant) are not a requirement. The
  operator, 2026-11-05: ""- torch's own allocations (kernel scratch, and results of ops with no out= form);" i dunno
  who cares...?"
- Closed: typed values need no placement of their own. Their run is a mapping, and every value is placed by Dilworth's
  chains (PB R2). The operator: "ok so we're done with this topic because the constraints which seemed to require
  heroic intervention were fictional and involved only idiomatic code." Left is a measurement, when runs are
  authorized: whether TLB reach over a mapped run differs from the region's own mapping.

## The revisions of 4e78f6f4

The record of the unit whose instruction is "The documents' corrections of 2026-11-05 (the coordinator)" above, points
1, 2 and 4 to 8, from mesh 4e78f6f4 and metal-microbench e9fd7538, with the documents as 389c4072 and f715c249 revised
them during the unit (point 5's answers on the plan call's form and on how the bridge finds a program's plans; A9, "What
the bridge runs: the self-timed schedule"). Its rules are design/disjoint-pages-revision-coordination.md, "Rules (every
sub-agent)". The coordinating implementer fixed the shared interface first (dd215cbe), then five sub-agents worked on
disjoint files: G (rdma/mesh-graph.c), C (rdma/mesh-collective.c), D (rdma/mesh-dataflow.c, mesh-call.c, mesh-flow.c,
mesh-records.c, mesh-link.c, mesh-stat.c, mesh-bridge.h), T (rdma/mesh_compile.py, mesh.py, mesh-torch.mm) and E
(metal-microbench tools/e2b_replay.m). T's session ended on a network error after its last commit and before its report;
its decisions below are read from its notes at the code. Nothing was run but the compilers, `python3 -m py_compile` and
the four incremental builds.

### Done

- **Point 1, capture by `torch.export`** ("capture each torch program with `torch.export` before it runs, the same at
  every rank, with nothing run eagerly between compiled calls. rdma/mesh_compile.py's Dynamo backend (`serialize`)
  goes"). T, 44ebaa3b and 35711f88:
  - A compiled function is `Function(module, placement, weights)`, a leaf of A10's tracer. `backend`, `serialize`, the
    Dynamo markers and `CALLING` are gone.
  - `program(block, *example_inputs)` traces the block and runs it once over fake tensors (`_Export`, a
    torch.fx.Interpreter in a FakeTensorMode). It refuses at compile time any node that is not a placeholder, a call of
    a compiled function, an `operator.getitem` of such a call's result, or the output.
  - Each call is exported as mesh's engines.py captured, `torch.export.export(module, args, kwargs, strict=False)`.
    Then come `run_decompositions({})` (torch 2.14.0 exported_program.py: "If empty, we don't decompose any
    operator") and `module(check_guards=False)`. A function called with other metadata is another compiled function.
  - Ordinals follow program order, so every rank names them alike whatever its device. This answers "The revisions of
    e4a7398e", "Where the documents are wrong", A4's compile count.
  - The partitioner reads the exported graph's ATen operators: its tables are `torch.ops.aten` packets.
- **Point 2, a node with no section** ("start its plans' executions and relay, and compute nothing. `_absent` goes").
  - T's `_execute` raises the plan's executions (mesh_transfers_start) and returns no value where the function maps
    this node no device. `_absent` is gone.
  - A call with a section that reads such a call's result is refused at compile time (A4, withdrawn text: "it has no
    value for any program code to read").
- **Point 4, the program's whole IPC graph** ("build the program's whole IPC graph at compile time, every function
  together, with each queue's room (5) in it. `plan_rooms`' part graph goes"). C, c6c9c77f and 8d697c32, to the brief:
  - **Actors.** `plan_graph` builds one graph over every call of the block: σ_j and β_j for each call at the sender,
    and x_{j,v} for each node of its function's plan ("The actors are each node's parts of the program's calls").
  - **Edges.**
    - The sender's two sequences in program order, as BSL §4.1's processor cycle (delay 1 back).
    - Each call's queues and transfers (delay u, mapped to the block's executions).
    - A10's edges from each call's end to the start of its function's call `outstanding` calls later.
  - **Conversion and rooms.** Convert-to-SC-graph and DetermineDelays run over the whole graph. Each queue's room is
    the sum of its calls' bounds (5).
  - **Synchronization edges.** Each added edge is a struct mesh_sync entry, a wait at its sink's plan and a tell at
    its source's (mesh.h 117). D implemented the waits (mesh_transfers_start) and the carrying of the counts
    (counts_tell and count_take; struct mesh_count `function`, the sink plan).
- **Point 5, one offline allocation** ("place the whole program's values in one offline allocation at compile time.
  These go: `mesh_claim_take`'s first fit and the claim's head record; `mesh_plan_head`'s walk ...; `ENOMEM` from
  fragmentation. Release at retirement stays deferred reclamation (RCU). The program's group still names its plans").
  - **The plan call** takes A8's one IR of the whole program and writes A9's serialized operand, as 389c4072 answers:
    "The plan call's form for a whole program is A8 and A9". The IR is struct mesh_function per compiled function and
    `calls`, A10's calls in program order (dd215cbe).
  - **The serialized operand's root** is at the layout's one place: struct mesh_program at hdr `program_off`, laid out
    after the bridge's pages (mesh-map.h). Its `table` names each function's state section. The bridge reads the plans
    there (mesh_program_plan, D's `plan_from`), as 389c4072 and f715c249 answer: "the serialized operand's place is a
    constant the region's layout names once"; A9, "the bridge reads the serialized operand at a place its compilation
    fixed, and searches for nothing".
  - **Gone.** D removed mesh_claim_take, mesh_claim_close, mesh_head_after, mesh_head_state, mesh_plan_head and the
    first fit. Every walk of claim heads in mesh-flow.c, mesh-records.c, mesh-link.c and mesh-stat.c is gone, and so
    are mesh_transfers_start's tests of its handle.
  - **Retirement.** mesh_plan_retire closes one plan's state. mesh_retire closes the root. mesh_retired_release frees
    the root, as deferred reclamation, once no port's `prepared` is its generation and no plan's `held` is set.
  - **Numbering.** C numbers rows from row 0 and blocks from each space's first; there is no claim and no
    fragmentation.
- **Point 6, a value as its ordered page-table rows** ("Represent a value as its ordered page-table rows ... Place
  every value's pages anywhere, by the fewest chains of the interference order ... Dilworth's theorem, computed by
  Fulkerson's bipartite matching ... Give an operand typed as one run its run as a mapping written when the plan is
  made ... Greedy by size goes, with `mesh_graph_left_edge`'s colouring").
  - **The chains.** G wrote mesh_graph_chains: Fulkerson's network with each item one vertex of capacity its blocks,
    a maximum flow by Hopcroft and Karp's (Dinic's) phases, the chains numbered by their first elements (876b1d69).
    It removed mesh_graph_left_edge.
  - **The placement.** C places every item of every function by it, one call per space, under the order `plan_before`
    (decision PO-1 below). Rows are bound block by block, each block anywhere.
  - **The mapping.** `run` values get one mapping (mesh_view_compose: ncclMeshPageTableCompose's reservation and
    `mach_vm_remap`, restored from 1b372d09^ by D, 19c8f25e). 6b9d357b remaps a run of consecutive blocks at once, as
    that function did.
  - **Readers.** mesh_section_address names one row's block (D, 32a72160). E binds every e2b value through the
    mapping in one MTLBuffer (d81575db); T binds CPU and MPS tensors over it.
- **Point 7, transfers land in their value's pages** ("name, in each declared transfer, the pages of its destination
  value that it lands in, as NCCL's receives land at offsets of one result ... no concatenation and no copy of a part
  remains").
  - **The interface.** struct mesh_declared `first`. C maps each range to blocks [first, first + P) of its value, with
    its words at its own first row. The value's uncovered blocks are one item that shares no block.
  - **Torch.** T lands AllGather's and AllToAll's parts at their places (torch.cat goes). ReduceScatter's slices and
    the AllReduce root's partials land at their places in one value, summed by torch.add's out variant. A piece is sent
    from its place in its producer's value, and `send`'s copy of a part goes (#6's stop 2).
  - **e2b.** Its `first` is 0: each route is one whole operand at both ends, and every consumer binds each received
    operand as its own input (E's reading of e2b_ir and the manifest). No concatenation or copy was there to remove.
- **Point 8, torch MPS operands typed** ("the IR types each such operand, and the plan writes its mapping"). T types
  every value it binds as a tensor `run` 1 and binds MPS tensors over one MTLBuffer made over the whole mapping, which
  keeps PV-2's reason: one resource's accesses, which Metal's hazard tracking orders. `_Mapping` releases the view by
  reference count once no tensor over it remains (G. E. Collins, 1960, cited at the code).
- **The coordinator's findings on #6.**
  - rdma/mesh-section.h holds the one struct mesh_section, C/C++-safe, and mesh-call.h and mesh-torch.mm include it
    (dd215cbe, 32d9fcc3).
  - `send`'s `pages(value).copy_(source)` is gone for parts (point 7). A copy stays where item 7's "What stays" puts
    one ("an operator with no out variant, whose result is copied once into its placed pages where it must be there").

### Found already done

- e2b concatenates no received part (point 7): each consumer binds each received operand as its own input (E).
- The program's sender, its views and its queues (A7, appendix item 2) needed no change but the edges' entries.
- metal_recording.m compiled against the new headers unchanged; E's finding on its barrier is below.

### Decisions taken where the documents are silent (each the named implementer's, with its reason)

The two places the coordinator's first brief called open are not decisions: point 5's text (389c4072, f715c249) answers
both. They are under Done, as A8 and A9 (the plan call) and A9 and R1 (the root at the layout's place).

The coordinating implementer's:
- **PO-1, the order over the copies.** R2 states the relation for one execution: "Two items may share a page exactly
  when one's last access happens before the other's first write ... That relation is a partial order".
  - Over u copies a page is written again by its chain's first item at c_1 + u. So copies (v, c) and (w, c′) are
    ordered where c ≤ c′, c′ − c ≥ D_vw (D_vw = 0 and v ≠ w where c = c′), and c′ + E_w ≤ u, with E_w the largest
    least lag from w's reads to any write at the node.
  - The last clause gives every chain's wrap (c_k + E_{x_k} ≤ u ≤ c_1 + u) and keeps the relation transitive, by the
    least lags' triangle through y's write before its reads, as Dilworth's theorem needs.
  - C checked it and found it sound and conservative: it can cost pages.
- **PO-2.** Items of different functions are unordered: no edge of either template orders them. Invariants, state
  sections, the table, a lag-k value's copies and a named value's uncovered blocks are unordered with every item, as
  plan_interferes had them interfering.
- **PO-3, multiplicities.** An item's blocks are its multiplicity in Fulkerson's network (a vertex of capacity count on
  each side), so the network has one vertex an item, not one a block. This is the same matching, with interchangeable
  elements.
- **PO-4, the state sections and the table are one run each.** mesh_graph_chains numbers an item ordered with no
  other consecutively, so their readers (the bridge, mesh-metal.m) address them by region offset as before.
- **IG-1, per-call actors.** Each call of the block is its own actor, and a function's plan has `calls` of them at a
  node, its k-th start the call at place k mod `calls` (mesh.h struct mesh_state). Appendix item 6 makes a plan a
  compiled function ("each call launch one execution of its function"), and "The actors are each node's parts of the
  program's calls". So a plan can have several synchronization edges at a node, struct mesh_sync's entries.
- **IG-2, A10's and the transfers' delays.** An edge of delay d in a function's executions runs from call place i to
  (i + d) mod n, with delay ⌊(i + d) / n⌋ in the block's executions. For one call it is BS's x_s → σ and x_v → x_v.
- **IG-3, BS-4 for an actor of a call.** A count of starts at a node implies the end of the call A10's edge leads
  from. So a wait entry's `position` is the tail's A10 successor's place, and its delay in the rooms is DetermineDelays'
  δ plus that edge's delay.
- **IG-4.** A queue's room is the sum of its calls' bounds (5): one queue carries every call's tokens.
- **EB-1, EBUSY.** mesh_plan refuses where the root's `owner` is not 0, before anything is written. The run is one
  program on the whole collective (A7, corrected), and a page is reused only after deferred reclamation's release
  (C3.4; RCU). A replan after ESTALE waits for the bridge's release: the torch program raises EBUSY and its next call
  plans again.
- **MV-1, one mapping a program.** It has one release, and one MTLBuffer over it keeps PV-2's one resource.
- **Hunks in files no sub-agent held, each required by this unit's scope:**
  - metal-microbench metal_recording.m (dc862ecb). MetalEncoder's queue barrier's visibility is Device |
    ResourceAlias. E found that the mapping puts two values sharing a block at two virtual addresses of one page.
    MTL4CommandEncoder.h: MTL4VisibilityOptionResourceAlias "Flushes caches to ensure that aliased virtual addresses are
    memory consistent".
  - metal-microbench Makefile:
    - 0f3ae103 names libmesh's graph and section inputs, which the sub-make's rule omitted;
    - 8c3a8e06: 0f3ae103 made the SDK's stamp older than the Makefile, its prerequisite, so the e2b build remade an
      unchanged SDK, whose sparse-checkout step then failed on the existing clone. The stamp now takes its operands as
      text, as rdma/Makefile keeps HiGHS's (HIGHS_INPUTS). "A dependency is remade only when its own inputs change."
  - mesh rdma/Makefile:
    - 4bfa3b55 lists mesh-section.h;
    - 9c5210b3 makes mesh-stat from its own inputs (D's finding: it calls nothing of mesh-dataflow.c now).
  - rdma/mesh-torch.mk: a571a6a6 lists mesh-section.h.
  - rdma/mesh-ports.c: a comment's struct name.
  - mesh.h mesh_program_state: one reading of a table word, where three files read it (D's finding; 9c5210b3).

G's:
- Which elements of an item continue chains: the first in[j] of item j continue, and the rest start chains. The
  decomposition order is items by their predecessor counts, a linear extension. It is deterministic.
- The bound: counts and items below 2^32, so every flow and chain number fits in 64 bits.

C's:
- `count` or `ncalls` 0 is EINVAL: no actor, so no graph.
- One chains call where the window is the whole arena.
- ENOMEM where the rows pass the page table's.
- No run-time test that an added edge's tail is an x with an A10 edge: σ₀ reaches every vertex, so no sink SCC holds
  a σ or β (the argument is at the code).
- ρ is computed per call, so memory is O(V + E).
- The map is read before the operand tests, so a malformed operand may report the reader's error first.

D's:
- The bridge's VIEW payload is renamed struct mesh_group_view, where mesh-dataflow.h's struct mesh_view collided with
  it.
- `plan_from`, one iterator for seven readers.
- `clients_turn` keeps its exit watch beside mesh_holds.
- Only a start that submits waits.
- mesh_view_compose has no alignment operand.

E's:
- One MTLBuffer over the whole mapping. Metal 4's barriers name stages, not resources (MTL4CommandEncoder.h).
- The mapping is released after mesh_detach.
- The handle's EINVAL test after mesh_plan is dropped (point 5).

T's (from its notes):
- Example values are fake tensors. A function called with other metadata is another compiled function (JAX's trace
  cache).
- `_RemoveRuntimeAssertionsPass` (torch/_export/passes) strips the export's runtime assertions before partitioning.
- A part not a run of its producer's value is sent from a value of its own, copied once (item 7's "What stays").
- `_Mapping`'s reference count.

### Where the work stopped

- **IG-5, a node's actors in program order.** Appendix item 2: "The processors are the program's nodes. The actors are
  each node's parts of the program's calls, in program order", with BSL §4.1's cycle.
  - Against it: the same item, "Delivery does not wait for any earlier call to finish ... A call still running at a
    node delays no later call's delivery there"; and A9 (f715c249): "Its processors are the sends of each adjacent
    node and direction, R5's cables".
  - A node's parts of successive calls overlap, so edges forcing them in sequence would bound the queues by an order
    the code does not keep. They are not built. The sender's program thread and its TO-broadcast are sequential and
    are built so.
- **AllToAll's own piece** (T). The operand is one tensor its producer's out variant writes, and the other pieces are
  sent from it, so the own piece reaches its place in the result by one copy, as NCCL moves a rank's own block by a
  send to itself. Against point 7's "no copy of a part remains".
- **A part that is no run of blocks** (T, `_landing`). A collective whose parts along their dimension are not runs of
  the result's blocks, each beginning a block, is refused at compile time, citing point 7 and `first`. That covers a
  gather along an inner dimension, or parts of a partial block. Point 7 names pages, and a page lands whole.

### Where the documents are wrong, or contradict the code's facts

- **Appendix item 2 against itself and A9**, the processors (IG-5 above).
- **R2's "That relation is a partial order"** holds over one execution. Over u copies the wrap from a chain's last
  item to its first's next instance is not implied by the pairwise relation; PO-1's clause restores transitivity.
- **"Every failure but ESTALE and the second map read's is found before anything is written"** (mesh-collective.h)
  cannot hold for the mapping. mesh_view_compose reads the rows' bindings, so a mapping Mach refuses is found after
  them (C), and the root is then not published (d64b93e6).
- **The coordinator's first brief** called the plan call's form and the bridge's finding of plans open. The documents
  answer both (389c4072).

### Open

- **Aliased virtual addresses within one encoder.** The recorder's in-encoder barriers (the indirect command buffer's)
  name no visibility. Whether they make aliased addresses of one page consistent is not stated in the headers. The
  same holds for torch's MPS work over the one buffer. Both need a run.
- **The C9 hook's remarks:**
  - On dd215cbe, that the whole IPC graph "moves into the graph module": its algorithms are the graph module's
    (cycle_mean, convert_sc, determine_delays, lags), and mesh_plan builds their operands.
  - On 19c8f25e, on mesh_transfers_start's threshold (K − delay) × every + position + 1: BSL (7)'s arithmetic at the
    sink, not a walk; not moved.
  - On c6c9c77f, a walk over a parent table: moved to mesh_graph_rows (8d697c32).
  - On T's commit, a walk up the narrows: moved to mesh_graph.forward (35711f88).
- **Run-time checks left, outside these points** (D): mesh-records.c `queues_inside` and mesh-flow.c link_prepare's
  room and entry tests check facts the plan fixes at compile time.
- mesh-collective.h's IPC and order notes are the coordinating implementer's derivations, and they await the
  operator's ruling.

### Commits

- mesh:
  - dd215cbe (the interface: mesh.h, mesh-map.h, mesh-collective.h, mesh-dataflow.h, mesh-call.h, mesh-graph.h,
    mesh-section.h);
  - 4bfa3b55, 9c5210b3, d64b93e6, 6b9d357b, a571a6a6 (the coordinating implementer's);
  - G: 876b1d69;
  - C: c6c9c77f, 8d697c32;
  - D: 19c8f25e, 32a72160, e2effc95, 8e0ffb74, 84f569e2, a54e1774, d9b4e570;
  - T: 32d9fcc3, 44ebaa3b, 35711f88;
  - this record.
- metal-microbench: 0f3ae103, dc862ecb, 8c3a8e06 (the coordinating implementer's); E: d81575db.

### Integration

At mesh a571a6a6 and metal-microbench 8c3a8e06. Every build was incremental; HiGHS and the LiteRT/Dawn SDK were not
rebuilt.
- **`make -C rdma`:** exit 0, no warning.
  - First whole build after C's c6c9c77f: 2.8 s. mesh-flow, mesh-heartbeat (its mesh.h changed), mesh-stat and
    libmesh.dylib were remade.
  - Later runs: up to date.
- **`make -C rdma native-audit`:** exit 0, no warning.
- **`make -C rdma -f mesh-torch.mk TORCH=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages/torch`:** exit 0, no
  warning, about 1 s.
- **`make .build/e2b_replay`:** exit 0, no warning, 1.6 s. libmetal_recording, libmesh_metal and e2b_replay were
  remade.
  - The first try, before 8c3a8e06, stopped at the SDK's stamp, as above. Nothing was changed.
- **Compiled alone first** with `-Wall -Wextra -Werror`, exit 0: every changed C file, the failport lifecycle files
  (unchanged), mesh-ports.c, mesh-events.c, mesh-metal.m, e2b_replay.m, and metal_recording.m with metal_native.m.
- **Python:** `python3 -m py_compile` passes rdma/mesh_compile.py and rdma/mesh.py.
- **Nothing was run.** No plan, mapping, binding, export or publish here has been exercised.

### Answered in the documents (2026-11-05, the coordinator, on this part's record)

Verified at mesh d28b18e4 and metal-microbench 8c3a8e06: `make -C rdma`, `native-audit`, `mesh-torch.mk` with TORCH
set, and `make .build/e2b_replay` 0 incrementally, no warning; rdma/mesh_compile.py, mesh_graph.py and mesh.py compile.
Three read-only reviews of the diff against the requirement lines are recorded below when they report.
- **IG-5** (stop). Appendix item 2 now states the program-order cycle as each node's submissions, of time 0. Each
  call's part follows its submission, and A10's bound is the edge from a part's end to the submission b calls later.
  BSL counts the iterations a processor may hold by the delay on its back edge. So the parts overlap by up to b calls,
  and their sends are actors on R5's cables, A9's processors. The two statements describe different processors of one
  IPC graph.
- **AllToAll's own piece** (stop). PB R2 now states that the result's rows for the rank's own part name the operand's
  own pages: sharing by reference through the page table, as fbufs and IO-Lite pass buffers, with no copy.
- **A part that is not whole blocks** (stop). PB R2 now states that the result's layout is chosen, separately from its
  logical shape (XLA's `LayoutProto`, `minor_to_major` and padding). The collective's dimension is most-major, each part
  padded to whole blocks, and the result is read through its strided view. A consumer that cannot take that view has
  its own layout change, as an out-variant operator. The refusal goes. It refused Megatron's column-parallel gather
  along the last dimension, the common case.
- **R2's partial order over the u copies** (document error). PB R2 now carries PO-1's clause, cited to modulo variable
  expansion (Lam 1988) and cyclic interval graphs (Hendren et al., CC 1992).
- **"Every failure … before anything is written"** (document error, rdma/mesh-collective.h). The rule is that nothing
  is visible before the root's publication: one release store after every write, in C3's writing order, as RCU
  publishes a structure only once it is initialized (McKenney and Slingwine). A mapping that Mach refuses after the row
  writes leaves nothing published, which d64b93e6 does. The header's sentence is to be restated so.
- **EB-1** (a decision this part took, to be changed). An EBUSY that hands a retry to the program's next call is not
  deferred reclamation's form. In RCU's form the writer waits for the grace period to end (`synchronize_rcu`), here the
  bridge's release of the old root. So mesh_plan waits for it, bounded by MESH_PLATFORM_NS, and returns ETIMEDOUT past
  the bound.
- **Run-time checks left** (D's finding). mesh-records.c `queues_inside` and mesh-flow.c link_prepare's room and entry
  tests check facts the plan fixes at compile time, which the compile-time requirement forbids. They go in the next
  unit.
- **Still open, needing a run:** whether Metal keeps aliased virtual addresses of one page consistent within an
  encoder, and under torch's MPS work over the one buffer (MTL4VisibilityOptionResourceAlias, dc862ecb).

### Review of 4e78f6f4 against the requirement lines (2026-11-05, three read-only reviewers; the coordinator's answers)

Each finding is at d28b18e4, merged where two reviewers found the same thing. Each names the requirement line it fails
and what the code must do. Where a finding needs a method, the method is cited.

**Transport by view and region-only memory**
- V1. `_part`/`_send` (mesh_compile.py 1264–1292, the copy at 2152) copy a sent part into a value of its own in four
  cases: a slice along an inner dimension; an offset that is not block-aligned or a part overlapping another range's
  blocks; a CollectivePermute result; a program input. Item 7 allows one copy only for "an operator with no out
  variant". The code must send each part as a view of its producer's pages at their places, and lay parts that are not
  whole blocks out by PB R2's layout answer (the collective's dimension most-major, parts padded to whole blocks).
- V2. AllToAll's own piece (1447) and AllGather's own shard where `_directed` is false (1401) are copied. The code must
  make the result's rows for the own part name the operand's own pages (PB R2: sharing by reference through the page
  table). Only the no-out-variant case keeps its one copy.
- V3. The halo window concatenates its received pieces, `torch.cat(parts, dim, out=target)` (2192). The pieces must land
  at their places in the window region's value (point 7), so no concatenation remains.
- V4. The exported module's parameters and buffers are read from torch's allocations (2170, `_over`'s empty set for a
  constant). That fails region-only memory. C3.2: "Weights, the KV cache, pooled intermediates, exchanges … are all
  placed by `mesh_plan`'s allocator". The code must place them as invariants, as e2b does.
- V5. A program input is the caller's tensor, outside the region. Region-only memory makes every program value region
  pages, so the plan places each input. The caller writes into the input placed for it, as a host producer, ordered
  by the event (C3.1: "A host producer is ordered by the event"). ExecuTorch's memory planning can likewise plan the
  graph's inputs; the coordinator cites that from memory, not from a page read.

**Compile-time, never run-time checks**
- K1. Run-time searches where the plan knows the index. Each is to be read by index, as MSCCL-IR's steps name their
  dependences by index (`depid`, `deps`):
  - the start's scan of every wait entry (mesh-dataflow.c 562): the waits are stored in rows keyed by call place, and
    each edge's delay is folded into its threshold;
  - `count_take`'s search (mesh-records.c 896): the COUNT carries the sink entry's index;
  - the held-flow search (mesh-flow.c 1696): a flow table indexed by place.
- K2. Run-time bound and null tests of facts the plan fixes go:
  - mesh.h 1017 and 1019;
  - `state->sender>=n` (mesh-records.c 864);
  - `queues_inside` (mesh-records.c 958, 1170);
  - `plan_inside` and `deliveries>M->length` (mesh-flow.c 715, 719, 2053);
  - link_prepare's room and entry tests.
- K3. Values recomputed every execution from make()-time facts become constants of the plan (mesh_compile.py 2167,
  2220–2221, 2320, 2323): `ir.sends`' release lists, `own`, `ir.device`, and the received parts' landing order, which
  lowering records in `ir.at` and the runner re-derives (2215, 2219).
- K4. The alias flush on every encoder (metal_recording.m 207, dc862ecb) is paid by all, though the plan knows which
  blocks two values share and where one hands over to the next. The flush goes on those barriers alone.

**The eight-second bound and bounded waits**
- B1. An execution's waits each take a fresh MESH_PLATFORM_NS deadline: the start's edges, then each `_arrived`. So one
  execution is bounded by 5 s × (1 + arrivals). The code must give one execution one absolute deadline, shared by all
  its waits.
- B2. The program record's `owner` test (mesh-collective.c 1964) returns EBUSY on almost every replan after ESTALE,
  because the bridge releases the record a turn later. It is also a load then a later store, so two planners pass it.
  The replan must wait for the release, bounded by MESH_PLATFORM_NS, as RCU's writer waits out a grace period
  (`synchronize_rcu`), with ETIMEDOUT past it. The record has one writer by construction, the program's process, since
  a run is one program on the whole collective (A7, corrected), so it needs no exchange and no lock: a single-writer
  register (L. Lamport, On interprocess communication, Distributed Computing 1(2), 1986).
- B3. `_retire`'s device synchronization (mesh_compile.py 2268) is an unbounded device wait, and it runs on every
  replan. The record's release must follow torch's work in an MPSEvent notification, as torch_mesh's allocator.mm did
  its release (fba41159's event form, item 7). It is then bounded by the notification wait already there.
- B4. `mesh_graph_chains` (mesh-graph.c 864–885) makes items² comparisons and runs Dinic's maximum flow over the whole
  program's items, with nothing bounding its time. The matching is Hopcroft and Karp's where capacities are 1 (J. E.
  Hopcroft and R. M. Karp, An n^5/2 algorithm for maximum matchings in bipartite graphs, SIAM J. Comput. 2(4), 1973),
  Fulkerson's reduction. Its time on the largest program is a measurement for when runs are authorized.
- B5. The microbench Makefile's network fetches (git clone, uv pip install) are unbounded. They are rarer now, and open
  like HiGHS's build.

**The graph module, duplication, deciders and effectors, and names**
- G1. `mesh_graph_chains` orders its items by its own predecessor count (mesh-graph.c 875–894) where `mesh_graph_order`
  computes that order. The added edges' grouping by plan is a hand-written count and fill (mesh-collective.c 1500–1515)
  where `mesh_graph_rows` does it. Both are to call the module's own functions.
- G2. Duplicated patterns, each to be merged:
  - the two walks of a value's ranges (mesh-collective.c 770, 2039);
  - the copies expression (1947, 2026);
  - the second walk for a run value's place (2123), which `runs_at` already holds;
  - AllToAll's and ReduceScatter's one pairwise-exchange body, with CollectivePermute's send branch (mesh_compile.py
    1430–1502).
- G3. The planner calls `mach_vm_remap` and writes virtual addresses into the IR, and reads `getpid()`
  (mesh-collective.c 2111–2129; mesh-collective.h `void *view`). Deciders see pages. The plan writes rows, and the
  program's effector maps them.
- G4. `struct mesh_program`, "the root" (mesh.h 740–764), keeps the claim head record's `owner`, `closed` and `start`
  under a new name, which point 5 removes.
  - Its name is A9's: the serialized operand in MSCCL-IR's shape, whose top level is the `<algo>` element with one
    `<gpu>` per rank.
  - Its fields are what A9 and the failport's lifecycle need: the operand's publication (B2's single writer) and the
    holder the heartbeat already knows, reused as the failport's code is.
  - "Root" also names the trees' roots (mesh-collective.c 1749), and goes from this record.

**Answered earlier today, to be built in the same unit** (above):
- IG-5's submissions cycle;
- PO-1's text;
- the publication rule in mesh-collective.h's note;
- EB-1, which is B2.

## The revisions of 55b9cbad

The record of the unit whose work list is "The revisions of 4e78f6f4"'s last two parts, "Answered in the documents
(2026-11-05, the coordinator, on this part's record)" and "Review of 4e78f6f4 against the requirement lines". It
starts from mesh 55b9cbad and metal-microbench 8c3a8e06, with the document text of 3a7fe8b0 and 55b9cbad. Its rules
are design/disjoint-pages-revision-coordination.md, "Rules (every sub-agent)".
- The coordinating implementer fixed the shared interface first (b35234c3, b25fad5c) and took rdma/mesh-graph.c (G1).
- Four sub-agents then worked on disjoint files:
  - C: rdma/mesh-collective.c;
  - D: rdma/mesh-dataflow.c, mesh-call.c, mesh-records.c, mesh-flow.c, mesh-link.c, mesh-stat.c, and mesh-bridge.h past
    the interface's hunk;
  - T: rdma/mesh_compile.py, mesh.py and mesh-torch.mm;
  - E: metal-microbench tools/e2b_replay.m, metal_recording.m and metal_recording.h.
- The coordinating implementer reviewed each commit as it landed and sent D three findings, each fixed (045ac991,
  2765c562, 0a2fdf50).
- Nothing was run but the compilers, `python3 -m py_compile` and the four incremental builds.

### Done

**Answered in the documents**
- **IG-5, the submissions cycle** (appendix item 2, corrected: "The program-order cycle is each node's submissions,
  not its calls' parts ... Call k's part at the node, x_k, follows its submission (σ_k → x_k, delay 0). A10's bound b
  is the edge from x_k to the submission b calls later ... no edge orders the parts themselves"). C, 31919b97,
  plan_graph and plan_submission:
  - each node of a call's plan has a submission actor σ_{j,v} of time 0, and the sender's σ_j are its node's;
  - each node's submissions form BSL §4.1's cycle (delay 1 from the last to the first);
  - σ_{j,v} → x_{j,v} has delay 0, and A10's edge runs x → σ, where it ran x → x;
  - an added edge's tail may be a σ at a node: its position is its own call's place and its delay δ (its end is that
    start); an x tail keeps IG-3's form;
  - the argument that no sink SCC holds only a σ or a β is restated at the code. A sink SCC holding a β holds some σ at
    the sender, which reaches every vertex, so the graph is strongly connected and nothing is added.
- **A rank's own part shared through the page table** (PB R2: "the result's rows for the rank's own part name the
  operand's own pages ... as fbufs and IO-Lite pass buffers"): V2 below.
- **A part that is not whole blocks** (PB R2: "the collective's dimension most-major, each part padded to whole
  blocks ... The refusal goes"): V1 below.
- **PO-1's text.** plan_before cites PB R2's corrected clause ("Copies (v, c) and (w, c′) are ordered where c ≤ c′,
  ...", Lam 1988; Hendren, Gao, Altman and Mukerji, CC 1992) as the document's, where it called the clause the
  implementer's (C).
- **The publication rule** ("nothing is visible before the root's publication: one release store after every write,
  in C3's writing order, as RCU publishes a structure only once it is initialized").
  - mesh-collective.h's note is restated so (b35234c3), in place of "Every failure but ESTALE and the second map
    read's is found before anything is written".
  - mesh_plan writes every word in this order (C): every word of the operand, then the second map read, then struct
    mesh_algo `plans`, `group` and `table` and the holder (hdr `client`, `client_start`), then `published` last
    (release).
  - A failure before that store (ESTALE, the reader's error) leaves nothing published. The mapping is no longer
    mesh_plan's (G3), so no Mach failure follows the writes.
- **EB-1**, which is B2 below.
- **The run-time checks left**, which is K2 below.

**Transport by view and region-only memory** (T: 96cb578c, 0c980472, 75f3c1cd, 44212450)
- **V1** ("The code must send each part as a view of its producer's pages at their places, and lay parts that are
  not whole blocks out by PB R2's layout answer").
  - Each value has a layout (_layout, _form, _claim): one dimension most-major, each slice along it padded so every
    part is a run of whole blocks, kept as one strided view its producer's out variant writes.
  - Each part is sent from its producer's pages (_run_of). Where the parts do not fit the producer's layout, the key
    is laid out once into a value of its own (_laid): PB R2's "the change of layout is an operator of its own, run as
    its out variant into placed pages".
  - A collective's parts all come from one value (75f3c1cd). `_landing`'s refusal and `_part`'s four copies go.
- **V2** ("make the result's rows for the own part name the operand's own pages"; PB R2, as above).
  - The interface: struct mesh_share and struct mesh_function `share`/`shares` (b35234c3).
  - The planner, C (plan_function, plan_rows): validates each share before anything is written; binds each copy's
    shared rows to `of`'s pages; gives the shared items a lifetime covering both values' accesses (a decision, below).
  - The torch path, T: AllGather's own shard where it is not made in place, and AllToAll's own piece, are shares, with
    no copy.
- **V3** ("The pieces must land at their places in the window region's value (point 7), so no concatenation
  remains"). T: each halo piece lands at its `first` in its window's landing value (_region), and the window's middle
  is shared from the operand. `torch.cat` goes.
- **V4** ("The code must place them as invariants, as e2b does"; D.3.2: "For the weights, that write is the load").
  T: each parameter and buffer is a MESH_INVARIANT value, loaded once by `copy_` after the plan and before the first
  execution, and read there by every execution.
- **V5** ("the plan places each input. The caller writes into the input placed for it, as a host producer, ordered by
  the event"). T (44212450):
  - each tensor input of the traced block is a value of `lag` HOST_LAG (2, D.3.3);
  - `run.inputs()` gives the caller CPU tensors over the next execution's placed copies, once execution b − HOST_LAG
    has ended (its end an MPSEvent notification, mesh-torch.mm mesh_torch_end, the wait bounded);
  - `run()` takes no argument. This is the API change for the caller.

**Compile-time, never run-time checks**
- **K1** ("Each is to be read by index, as MSCCL-IR's steps name their dependences by index (`depid`, `deps`)").
  - **The waits.** The interface (b35234c3): struct mesh_sync is 32 bytes. A wait carries `base` = position + 1 −
    delay × every, and struct mesh_state `rows` holds its waits' compressed rows by call place.
    - C writes them grouped by mesh_graph_rows.
    - D (13c2d411, mesh_transfers_start, mesh_wait_row) reads only row k mod `calls`, each wait until `count` ≥ K ×
      every + base, with no branch against a delay.
  - **The COUNT.** A tell's `entry` is its edge's index among the sink's waits, which struct mesh_count carries.
    - C computes it.
    - D (d61df2a9, 045ac991): counts_tell sends it, and count_take raises edge[entry] alone, with no loop and no
      map_index_of search.
  - **The held flow.** D (d64fb13b): struct port_plans `loaded` is a table indexed by place (place_at), and struct
    flow `next` is gone.
- **K2** ("Run-time bound and null tests of facts the plan fixes go"; and "Answered in the documents", "Run-time
  checks left": "They go in the next unit").
  - mesh.h mesh_algo_plan: the k-against-`plans` test and the table word's NULL test go (b35234c3).
  - mesh-records.c (D): `queues_inside` and `state->sender>=n` go.
  - mesh-flow.c (D): `plan_inside` and its uses in link_prepare and plan_cancel go, as do link_prepare's room,
    deliveries, copies and entries tests and its `bytes`, `field`, `count` and `pages` EINVALs, and the null tests of
    a receive's count word. Each removal's note names its producer, mesh_plan.
  - Kept, as run-time state or a device's report: the map read and its serial (ESTALE), EPROTO, EHOSTUNREACH,
    EMSGSIZE, ENOMEM, the landing's byte-length check, and held_clear's test for a plan not yet released.
- **K3** ("Values recomputed every execution from make()-time facts become constants of the plan"). T (_steps): each
  item is built once as a closure of its release lists, receive order, places, own pair and the plan's state view. A
  node with no section has no steps.
- **K4** ("The flush goes on those barriers alone").
  - The interface: struct mesh_value `shares` out, 1 where a page bound in one of the value's rows is bound in another
    row at this node, another value's or another copy's (b25fad5c). C computes it once from the placement (plan_rows).
  - E (d329555a): MetalEncoder, MetalReplayEncode, MetalReplayRecorded and `replay:visibility:` take the queue
    barrier's visibility.
  - e2b gives each phase MTL4VisibilityOptionDevice, with MTL4VisibilityOptionResourceAlias added where a value the
    phase may touch has `shares` 1. The invariants' copy has its own, and the optimizer and the capture's scalar copy
    take Device alone.
  - E's Metal claims rest on MTL4CommandEncoder.h.

**The eight-second bound and bounded waits**
- **B1** ("one execution one absolute deadline, shared by all its waits").
  - mesh_transfers_start takes `deadline`, absolute clock_gettime_nsec_np(CLOCK_MONOTONIC) ns (b35234c3), and every
    wait of the start is bounded by it (D).
  - T's _execute takes one deadline and passes it to mesh_transfers_start, every `_arrived` and `_combine`
    (time.clock_gettime_ns(time.CLOCK_MONOTONIC), the same clock).
  - E's start passes the deadline of its first execution.
- **B2 / EB-1** ("The replan must wait for the release, bounded by MESH_PLATFORM_NS, as RCU's writer waits out a grace
  period (`synchronize_rcu`), with ETIMEDOUT past it ... a single-writer register").
  - C (plan_released): after everything is computed and before anything is written, mesh_plan waits on struct
    mesh_algo `released` until it is `published` or nothing is published (os_sync_wait_on_address_with_timeout,
    region-shared), ETIMEDOUT past MESH_PLATFORM_NS. There is no compare-and-swap, and EBUSY goes.
  - D: mesh_retired_release stores `released` = `published` (release) once no port announces it and no plan is held,
    and wakes it.
  - T raises ETIMEDOUT as it raised EBUSY, and the program still retires its own operand before it plans again.
- **B3** ("The record's release must follow torch's work in an MPSEvent notification"). T (bfc93712):
  - mesh-torch.mm after_torch is the one event form (acquireEvent, record, notify, SyncType::COMMIT; torch 2.14.0
    MPSEvent.h, MPSStream.h), for mesh_torch_publish, the new mesh_torch_retire and mesh_torch_end;
  - `_retire` synchronizes no device: on MPS it retires in the notification, on the CPU after the calls, and the
    notifications' wait stays bounded.

**The graph module, duplication, deciders and effectors, and names**
- **G1.**
  - mesh_graph_chains takes its items' linear extension from mesh_graph_order over the edges `before` gave, where it
    sorted them by its own predecessor count. Any topological order is a linear extension; EDEADLK where those edges
    hold a cycle, which a strict partial order cannot. The coordinating implementer, 3686d9c8.
  - The added edges' grouping by plan is mesh_graph_rows, keyed by the sink node and the call place, and the tells by
    plan. C.
- **G2.**
  - C: one walk of a value's covers, its ranges and shares together (plan_rows), and one copies function
    (plan_copies). The run-place walk went with G3.
  - T: _exchange is the one body of AllToAll, ReduceScatter and CollectivePermute's send.
  - mesh.h mesh_algo_open is the one test of the operand open, which mesh_algo_plan, clients_turn, clients_gone,
    region_adopt and mesh-stat each wrote (52a12eb0, 2765c562).
- **G3** ("The plan writes rows, and the program's effector maps them").
  - mesh_plan takes no `view`, and struct mesh_value `view` goes (b35234c3). C: no mesh_view_compose, mach_vm_remap,
    getpid or mesh_started in the planner.
  - E: e2b maps every value's rows with mesh_view_compose, keeps one MTLBuffer over the mapping, and releases it after
    mesh_detach.
  - T: _mapped maps every `run` value's rows in one mesh_view_compose call, and _bind binds at those places.
  - The process's start is read at attach (D, mesh_attach: struct mesh_ctx `start`).
- **G4** ("Its name is A9's ... Its fields are what A9 and the failport's lifecycle need: the operand's publication
  (B2's single writer) and the holder the heartbeat already knows, reused as the failport's code is"; "Root" goes).
  - struct mesh_algo at hdr `algo_off`, MESH_VERSION 118 (b35234c3), with three words:
    - `published`: the generation, the program's alone, stored last;
    - `retired`: the retirement;
    - `released`: the bridge parent's.
  - The holder is hdr `client` and `client_start`, which mesh_plan writes from the context. `owner`, `closed`, `start`
    and `generation` go.
  - D's readers (mesh_retire, mesh_retired_release, mesh_holding_retained, plan_here, clients_turn, clients_gone,
    region_adopt, plan_stands, plans_walk, held_clear, mesh-stat's clients_open) all read struct mesh_algo.
  - "root" names no record in any changed note.

### Found already done

- **B4** ("The matching is Hopcroft and Karp's where capacities are 1 ..., Fulkerson's reduction. Its time on the
  largest program is a measurement for when runs are authorized"): mesh_graph_chains is that network, solved by
  Dinic's phases, which are Hopcroft and Karp's where every count is 1. The answer is cited at its note (3686d9c8), and
  no code changed.
- **B5** ("They are rarer now, and open like HiGHS's build"): open, with no code change.
- e2b concatenates no part and reads no record word (E). mesh_graph.py needed no change (T).

### Decisions taken where the documents are silent (each the named implementer's, with its reason)

The coordinating implementer's (the interface):
- **struct mesh_algo's words, each one writer's** (C3's rule 2; B2's single-writer register).
  - `published` is the program's.
  - `released` is the bridge parent's: RCU's grace period ends there, and mesh_plan waits on it.
  - `retired` is stored by mesh_retire's two callers, the program at its detach or replan and the bridge for a gone
    holder (the failport's lifecycle, as the tag's mesh_retire retires a gone client). Both store the one value
    `published`.
  - A plan stands where `published` is not 0 and is not `retired`.
- **The holder is hdr `client` and `client_start`**, G4's "the holder the heartbeat already knows". mesh_plan copies
  them from struct mesh_ctx `client` (minted at attach) and `start` (read at attach, so the planner reads no process
  identity: G3). Nothing is written at attach (R8).
- **The waits' form.**
  - Each delay is folded as `base` = position + 1 − delay × every, so one bound K × every + base serves every K, and
    a bound of 0 or below holds at once.
  - The waits are rows by call place (`rows`), each an index among the plan's waits.
  - A tell names its sink's wait (`entry`), and struct mesh_count's `function` is 32 bits with `entry` beside it.
- **The deadline's clock** is CLOCK_MONOTONIC in nanoseconds, the clock mesh_transfers_start read already and Python's
  time.clock_gettime_ns reads.
- **struct mesh_share** is (value, first, of, at, blocks), both values of the same copies. C's two further EINVALs (a
  share of a share; two spaces) are accepted into the header (e627f61b).
- **`shares` counts another copy of the same value too**, since a mapping of every copy puts both rows at two virtual
  addresses (b25fad5c).
- **MESH_VERSION 118**, and hdr `program_off` renamed `algo_off` in its place.
- **mesh_algo_open** (52a12eb0), from the review of D's commits.
- **mesh_graph_chains' EDEADLK**, mesh_graph_order's own error.
- **metal-microbench Makefile** (4671b439): libmesh's rule lists mesh-map.h and mesh-holders.h, which mesh-dataflow.c
  includes and the rule omitted. "A dependency is remade only when its own inputs change", and is remade when one
  does.
- **plan_here** compares the operand's group before it reads a table word, so another launch's message is let go
  before the table (the ruling on D's finding, 0a2fdf50).

C's:
- **IG-5's cycle at a node** holds the calls whose plan's `nodes` include it. A start with no part there has time 0
  and no other edge, so it changes no cycle's delay or mean. The sender keeps σ_j for every call, and keeps σ_j →
  σ_later where it is no node of the plan.
- **V2's lifetime** is a superset: every item of `of` takes part in no order (PLAN_ALONE), which covers both values'
  accesses and can cost pages.
- **V2's further tests:** EINVAL for a share of a share, and for two values whose `wire` differs.
- **A `state` value's zeroing** covers shared rows too, the plan's own write before any producer's (506b8d5e).
- **K4's count** is kept by page / block, unique for every placed block.
- **New ENOMEMs** where n × J keys pass 32 bits or δ × every passes int64. Waits entering one place keep the added
  edges' order.
- **Values' `section` and `shares`** are set before the release wait, so on ETIMEDOUT the caller's structs are filled
  and nothing in the region is written.

D's:
- **The start's bound is compared in `__int128`**, so a bound of 0 or below holds at once and a cancellation's
  UINT64_MAX passes every bound.
- **plan_stands reads `published` before the plan**, so after a replan with fewer plans no place is read past the new
  table.
- **Downstream null tests go:** a receive's count word is tested nowhere, since every receive has carried one since
  114.
- **place_at grows the flow table** to the operand's `plans`, ENOMEM through plan_load's existing refusal.
- **region_adopt** leaves this bridge's own pid as no gone holder.

T's:
- **Padding:** each slice along the major dimension is padded to the least s ≥ its elements that is a multiple of
  E / gcd(m, E). The value stays one strided view its producer's out variant writes; where E and m are powers of two
  this is padding each part.
- **A value's layout** is fixed by the first collective that names it (_claim). A later collective needing another
  takes a layout change.
- **AllToAll's result** is copied once into its own value from its parts' view. A received part arrives in its
  sender's layout, so no single strided view fits both ends; this is PB R2's layout change, "an operator of its own".
- **A share refused by struct mesh_share's rules** is copied once at its item: two values not both named by a
  declared transfer here (a halo region at an edge partition), or a share of a share.
- **An input** is held by the function of the first call that reads it and has a section here. An AllReduce member
  holds no received buffer.

E's:
- **The mapping's rows** go value by value, each value's copies in order, the order mesh_plan's mapping had, so every
  binding stays where it was.
- **B1:** no other e2b wait changed. The step waits keep the waiter's per-value bound (mesh-metal.h), since holding
  step 0's end to the start's deadline would time out a step the waiter still considers live.
- **K4's granularity:** one visibility a phase is the finest this code allows, since `shares` is one bit with no
  partner. Each phase counts every earlier capture's own values.

### Where the work stopped

- **V1 and V2 for program inputs and parameters** (T).
  - V1 lists "a program input" among the copies to remove, and V2 says "Only the no-out-variant case keeps its one
    copy".
  - Against them, the interface's facts:
    - struct mesh_value: a `lag` k value is "named by no declared transfer (EINVAL otherwise)";
    - struct mesh_declared: the value is "MESH_EXECUTION and wire";
    - struct mesh_share: "of the same copies (both named by a declared transfer here, or both of the same `lag`)".
  - So a part of an input or of an invariant is still sent from its one layout change, a copy. No document states a
    lag-k or invariant value that a range lands in or is sent from, with copies other than u.
- **V1 for overlapping parts** (T). "send each part as a view of its producer's pages" is against struct
  mesh_declared's "no two ranges at this node share a block of it". A halo wider than a shard, or two halos over one
  shard, has a copy of its own.

### Where the documents are wrong, or contradict the code's facts

- **A plan's name in a view does not identify its compiled plan** (D's finding).
  - K1's index read takes "the COUNT carries the sink entry's index" as a fact both ends' plans fix.
  - But a replan keeps its plan's name (N1), and while a map move floods, two nodes can plan one name on two maps
    within one view.
  - An `entry` past the sink's waits is then a store outside its edges: a fault in the bridge, not an error value.
  - The search the index replaced matched by node and function, and stored nothing outside the edges. Nothing
    changed for it; it is for the operator.
- **V2 where `of`'s blocks land here** (C's finding).
  - The minimal lifetime is sound where `of`'s shared blocks are written at this node (U1's premise orders the next
    write after `value`'s reads).
  - Where they land here as a received range, no edge of the template orders the next landing after `value`'s reads.
  - The own parts of AllGather and AllToAll are written at their node, so the torch path's shares are the sound case.
    The documents state no form for the other.
- **Appendix item 8 against G4.** Item 8 is "Settled, with no change: The heartbeat's holders"; G4 says "the holder
  the heartbeat already knows, reused as the failport's code is".
  - With the holder in hdr `client` and `client_start`, mesh-heartbeat.c (unchanged) now finds the program as its
    client holder, where it found none from 110 (L3).
  - The coordinating implementer read G4 as naming those fields.
- **The coordinating implementer's brief, not a document:** T's compile line `clang++ -std=c++17` fails inside torch
  2.14.0's own c10/util/intrusive_ptr.h:775 (`operator<=>`). rdma/mesh-torch.mk's `-std=c++20` compiles it.

### Open

- **Aliased virtual addresses, needing a run** (as before).
  - e2b now flushes at the phases whose values share pages (K4).
  - torch's encoders are torch's, so mesh_compile.py sets no visibility. Whether one MTLBuffer's aliased addresses stay
    consistent across torch's encoders is unshown (T).
- **mesh-metal.m's waiter**, a file no sub-agent held, gives each value a fresh MESH_PLATFORM_NS after the previous
  one. B1 names the start's edges and `_arrived`, and e2b's step waits share no deadline with its start. Making them
  share one would be a joint change of mesh-metal.m and e2b (E).
- **The C9 hook's remarks, not moved** (D):
  - on 13c2d411, mesh_wait_row and the start's waits: an indexed read of a compiled row and BSL's self-timed firing,
    as the last unit answered on 19c8f25e;
  - on d61df2a9, orders_submit's merge: Knuth §5.4.1's multiway merge, older than this unit, of which only
    `queues_inside` was removed here.
- **B4's time** on the largest program, and B5's network fetches: measurements and fetches for when runs are
  authorized.
- **No-out-variant copies** stay where appendix item 7's "What stays" puts them.

### Commits

- mesh:
  - the coordinating implementer's: b35234c3 and b25fad5c (the interface), 3686d9c8 (G1), 52a12eb0 (mesh_algo_open),
    e627f61b (struct mesh_share's tests), and this record;
  - C: 31919b97, 506b8d5e;
  - D: 13c2d411, d61df2a9, 045ac991, d64fb13b, 949ac07f, 927b69ac, cdbff3bb, 2765c562, 0a2fdf50;
  - T: 0c27ecd2, bfc93712, 44212450, 96cb578c, 0c980472, 197f595d, 75f3c1cd.
- metal-microbench: 4671b439 (the coordinating implementer's); E: b84e27bc, d329555a.

### Integration

Every build was incremental, at mesh 75f3c1cd and metal-microbench d329555a. HiGHS and the LiteRT/Dawn SDK were not
rebuilt.
- **`make -C rdma`:** exit 0, no warning.
  - The first whole build after C's 31919b97 took 2.4 s. mesh-flow, mesh-heartbeat (its mesh.h changed), mesh-stat
    and libmesh.dylib were remade.
  - The run at the end: nothing remade.
  - The codesign line's "Mesh Bridge: no identity found" is the rule's fallback to an ad-hoc signature, as before.
- **`make -C rdma native-audit`:** exit 0, no warning; the run at the end: nothing to be done.
- **`make -C rdma -f mesh-torch.mk TORCH=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages/torch`:** exit 0, no
  warning, 1.0 s. libmesh-torch.dylib was remade.
- **`make .build/e2b_replay`:** exit 0, no warning, 1.7 s. libmetal_recording, libmesh_metal and e2b_replay were
  remade; the run at the end: nothing remade.
- **Compiled alone** with `-Wall -Wextra -Werror`, exit 0:
  - every changed C file;
  - the failport lifecycle files, unchanged (mesh-heartbeat.c, mesh-children.c, mesh-child.c, mesh-lanes.c);
  - mesh-ports.c, mesh-events.c, mesh-call.c, mesh-metal.m;
  - e2b_replay.m, and metal_recording.m with metal_native.m;
  - mesh-torch.mm with `-std=c++20`.
- **Python:** `python3 -m py_compile` passes rdma/mesh_compile.py, mesh.py and mesh_graph.py (T: pyflakes clean on
  mesh_compile.py).
- **Nothing was run.** No plan, wait, release, share, layout, mapping, flush or retirement here has been exercised.

### Answered in the documents (2026-11-06, the coordinator, on this part's record)

Verified at mesh 327df606 and metal-microbench d329555a: the four builds exit 0 incrementally with no warning, and
rdma/mesh_compile.py, mesh_graph.py and mesh.py compile. Three read-only reviews of this unit's diff are recorded below
when they report.
- **V1 and V2 for program inputs and parameters** (stop). PB R2 now states that any value a transfer names is sent from
  its own pages, whatever its copies (R1: "nothing copied on the way").
  - An invariant is placed once in the registered window and read by every execution's send, its supplied word raised
    once (C3.2).
  - An input is placed per execution and written by its caller as a host producer (C3.1).
  - The plan call's rules against this are the coordination's contract, not a requirement, and go.
- **V1 for overlapping parts** (stop). PB R2 now states that ranges may share a block where their bytes are disjoint.
  - C3 writes "each byte once per use", each byte, not each block.
  - Each range keeps its own landed count (item 5), and a page instruction names its bytes at their offset in the
    page, as MSCCL-IR's steps name a destination offset and count.
  - The contract's "no two ranges at this node share a block" goes.
- **A plan's name in a view does not identify its compiled plan** (document error, D's finding). A7 now states that
  every member of a view plans on the view's map.
  - The view's first message carries the records of the sequencer's map that the view was created from.
  - Every member computes the same plan from the same operands (Schneider 1990), so a signal's index names the same
    entry at both ends by construction, with no test at run time.
  - The fault D found, a store outside the sink's edges, cannot then arise.
- **V2 where the shared blocks land as a received range** (document error, C's finding). PB R2 now states that a
  share's pages are one item from the earlier first write to the later last read. Where the first write is a landing,
  the next landing follows the last read by C3.3's lag test. A share with no such lag fails with ENOMEM, as C3.3 fails
  any value. C's choice, shared blocks unordered for the whole program, is then not needed.
- **Appendix item 8 against G4.** These do not contradict. Item 8 settled the heartbeat's holders "with no change", and
  G4's holder in hdr `client` restores the heartbeat's finding of the program as its holder, which it had lost from 110.
- **mesh-metal.m's waiter** (open). B1's one deadline per execution is to cover it and e2b's step waits together, as
  a joint change in the next unit.

### Review of 55b9cbad against the requirement lines (2026-11-06, three read-only reviewers; the coordinator's answers)

**The earlier findings:** fixed: V4, V5, B2, B3, G1, G2's four, G3. Partly fixed: V1, V2, V3, K1, K2, K3, K4, B1, G4. Each
is restated below with what remains. Lines are at mesh 327df606 and metal-microbench d329555a.

**Transport by view and region-only memory**
- W1. `_laid` copies a whole key every execution (mesh_compile.py 1405–1414, 2548) wherever a part is not a run of its
  producer's blocks:
  - a `'slice'` (`_base` gives none, though the runner makes it a narrow);
  - a narrow along another dimension;
  - a window region or a CollectivePermute arrival;
  - a layout fixed by another collective along another dimension;
  - a base inside an AllGather result.
  
  The answer is PB R1's correction: a transfer is a list of byte ranges of its source's pages (MPI's type map; VIA's
  data segments). So every part is sent from its producer's pages, and `_laid` and its copies go. A layout change
  remains only as a consumer's own operator, where its operand cannot take the landed layout.
- W2. AllToAll regressed. Its pieces now land in `_buffer` and all k are copied into the result every execution (2629),
  where at 55b9cbad they landed in the result. Each piece must land at its byte ranges in the result (point 7; PB R1).
- W3. Share fallbacks copy (2655): the window's middle where no CollectivePermute offers a share (1824, 1834), and a share
  between two local values of one lag that the torch side refuses (2010), though struct mesh_share and the planner
  accept it. A window's middle is the local value's own pages, a share. `ir.fallbacks` goes, both as a second form and
  as a forbidden name ("fallbacks or second forms").
- W4. Operators with several outputs are classed `'functional'` (1124–1127), so torch allocates their results, though
  they have out variants (for example `native_layer_norm.out`). That fails region-only memory (item 7). Their out
  variants take one `out` per result.

**Compile-time, never run-time checks**
- W5. K2: the bridge reads `plans` before `mesh_algo_open` acquires `published` (mesh-bridge.h 436; mesh.h 1114–1116),
  so a replan with fewer plans can be read past. The count is read from the published operand after the acquire, so
  the index is bounded by construction.
- W6. K3: per-execution branches on make()-time facts remain (mesh_compile.py 2156, 2598, 2605, 2606, 2635). Each
  execution's steps are built once at make() as closures with no test.
- W7. K4: e2b flushes aliases in every encoder of a phase that touches a share (e2b_replay.m 1497–1507). `shares` is one
  bit with no partner. The plan names each share's partner, so the flush goes on the handover barrier alone.
- W8. The bridge's take and poll (mesh-flow.c 1263, 1879, 1891) visit every place to `plans`, empty ones included. The
  places with flows toward each adjacent node are fixed when the plan is made, so they are a list the plan writes.
- W9. The share rules are tested on both sides of the interface: mesh_compile.py 2010–2015 against mesh-collective.c 942
  and 1949. They are tested once, at the planner's interface, which runs when the program is made, at compile time.
  The torch side restates none.

**The eight-second bound**
- W10. B1: a deadline per execution is not the rule.
  - The rule is the operator's, on stalls: "nothing you're running should take more than around 8000ms", judged by
    time without progress.
  - So every wait ends at MESH_PLATFORM_NS without progress, and the first ETIMEDOUT ends the run: no later wait runs.
  - make()'s waits, `_retire` then `plan_released`, share one deadline.
  - e2b's step waits (`platform_bound(units)`, e2b_replay.m 250) and mesh-metal.m's waiter take one bound each, not
    one per unit.

**The graph module, duplication and names**
- W11. G4: `retired` has two writers, the program and the bridge for a dead holder (mesh-dataflow.c 78; mesh-records.c
  1430, 1493; mesh-link.c 381). One writer each: the program writes `retired`, and the bridge writes only `released`. A
  gone holder is read from the heartbeat's own holder words, never written as `retired`.
- W12. `_claim` and `slots` (mesh_compile.py 1368, 1445) are inherited idiom. The first is XLA's layout assignment
  ("The layout is determined by the `LayoutProto` proto"), and the second is a value's copies.
- W13. Duplicated patterns, each to be merged:
  - the producer-pages-else-layout decision (`_sent` 1420, `_halo` 1822, `_collective_permute` 1799), which W1 removes;
  - the two copy loops (2548, 2655), which W1 and W3 remove;
  - "published and not yet released", written three times (mesh-collective.c 2091; mesh-dataflow.c 354; mesh-flow.c
    2157), to be beside `mesh_algo_open`;
  - the holder's read and the gone-holder test (mesh-link.c 377–381; mesh-records.c 1425–1430, 1491–1493; mesh-stat.c
    118–120);
  - the mapping's row list in two effectors (mesh_compile.py 2289; e2b_replay.m 1437), to be one mesh-dataflow helper.

**Built in the same unit, from the answers on 55b9cbad's record:**
- every member of a view planning on the view's map (A7), which also makes K1's index safe;
- values sent from their own pages whatever their copies;
- ranges sharing a block by disjoint bytes, with offsets in instructions;
- a share's one item and its lag.

## The root cause of the copies, and the revised work list (2026-11-06, the coordinator)

The operator: "stop describing this in idiomatic terms. why does this happen in algorithmic terms. in data structural
terms. in terms of reference literature. once this is said, what we *should* be doing will be a lot more clear, esp
with al lof theo ther 'partial fixes', which all sound like invented problems and idiomatic sourceless code..."

**The root cause** (plan-to-bridge R1, "Why the code copies"):
- A torch tensor is a strided array: one storage, offset, sizes and strides. A view's part is an affine subset of its
  storage.
- The plan call describes a value as one run of rows (`struct mesh_section`), a transfer as one contiguous extent of
  one value (`struct mesh_declared`: `value`, `first`, `bytes`), and an alias as one run of blocks
  (`struct mesh_share`).
- So the compiler packs every part that is not one extent into a value of its own and copies it before each send.
  Communication libraries do this for noncontiguous datatypes only where their descriptor cannot describe the data;
  MPI implementations flatten a datatype into (offset, length) blocks and send from them in place (Ross, Miller and
  Gropp 2003; Wu, Wyckoff and Panda 2004).
- These are patches of that one descriptor, and they go with it, not one by one:
  - review 55b9cbad's W1–W3 and W9;
  - the layout fixed by the first collective (`_claim`);
  - the padded most-major layout (withdrawn in PB R2);
  - `struct mesh_share` and its rules;
  - `ir.fallbacks`;
  - AllToAll's landing buffer.

**The revised work list.** It replaces the earlier work lists' W1–W3, W9, the share items and the layout items. Each
item is written strictly to its source.
- **R-1. The transfer descriptor is a flattened view** (PB R1 and R2).
  - A transfer's source and destination are each a view (value, offset, sizes, strides). The compilation flattens each
    into its list of (row, offset, bytes) blocks, adjacent blocks merged, as datatype processing flattens a type map.
  - The bridge's instructions are those blocks: aot A9, MSCCL-IR's steps with offset and count.
  - Two values' rows may name one frame, the page table's sharing (fbufs; IO-Lite). A frame's item lives from the
    first write to the last read over every value naming it, with C3.3's lag test.
  - The plan call's interface changes accordingly. That removes `_laid`, `_claim`, `ir.fallbacks`,
    `struct mesh_share` with its tests on both sides, the padded layout and AllToAll's buffer. A value's layout is the
    program's own, torch's strides.
  - Every member of a view plans on the view's map (aot A7), so both ends' blocks agree by construction.
- **R-2. Operators with several outputs run their out variants** (W4; aot item 7), one `out` per result.
- **R-3. Publication is read as RCU's publish-subscribe** (W5): every field of the published operand is read through
  the acquired publication, never before it (McKenney and Slingwine).
- **R-4. The compiled program is instantiated once** (W6).
  - At make(), each execution's work becomes a straight-line sequence of its kernel calls and publishes, with no walk
    of the IR and no test of a make()-time fact. That is a graph instantiated once and launched by each call (aot
    item 6: "CUDA Graphs: a graph instantiated once and launched many times").
  - Its form is the specialization of an interpreter to a fixed program, partial evaluation (Y. Futamura, Partial
    evaluation of computation process: an approach to a compiler-compiler, 1971; cited from memory).
- **R-5. The alias flush is on the barriers where a frame changes virtual name** (W7). The plan knows each frame's
  virtual names through R-1's rows. Metal 4's `MTL4VisibilityOptionResourceAlias` ("Flushes caches to ensure that
  aliased virtual addresses are memory consistent", MTL4CommandEncoder.h) goes on the barrier between a write under
  one name and a read under another, and on no other.
- **R-6. The bridge runs A9's self-timed lists** (W8). Each cable's processor executes its list of instructions in the
  compile-time order, firing each on its dependency word (aot A9, "What the bridge runs"). It scans no table of plans.
- **R-7. Every wait ends at MESH_PLATFORM_NS without progress, and the first ETIMEDOUT ends the run** (W10):
  - make()'s waits share one deadline;
  - e2b's step waits and mesh-metal.m's waiter take one bound each, not one per unit.
  
  The operator's bound is judged by time without progress.
- **R-8. One writer a word** (W11; Lamport's single-writer registers): the program writes `retired`, the bridge writes
  only `released`, and a gone holder is read from the heartbeat's words.
- **R-9. Names** (W12): `slots` is a value's copies, and `_claim` goes with R-1.
- **R-10. The duplicates R-1 leaves** (W13), each merged:
  - "published and not yet released";
  - the holder's read and gone-holder test;
  - the mapping's row list in one mesh-dataflow helper over the plan's rows.

## The revisions of 14b9a757

The record of the unit whose work list is "Answered in the documents (2026-11-06, the coordinator, on this part's
record)", "Review of 55b9cbad against the requirement lines" (W1–W13) and, replacing W1–W3, W9 and the share and layout
items, "The root cause of the copies, and the revised work list" (R-1 to R-10, at mesh 56ede97e).
- It starts from mesh 14b9a757 and metal-microbench d329555a, with the document text of 1a2dc4fd, 14b9a757 and
  56ede97e.
- Its rules are design/disjoint-pages-revision-coordination.md, "Rules (every sub-agent)".
- **The interface came first.** The coordinating implementer fixed it and committed it (31ee5d4a, 11b6453b, later notes
  ea330a9f, 08997c49, 0a92cdd6) across rdma/mesh-collective.h, mesh.h, mesh-holders.h, mesh-bridge.h, mesh-map.h,
  mesh-call.h, mesh-metal.h and mesh.py's mirrors.
- **Five sub-agents then worked on disjoint files:**
  - C: rdma/mesh-collective.c;
  - D1: rdma/mesh-flow.c;
  - D2: rdma/mesh-records.c, mesh-dataflow.c, mesh-link.c, mesh-stat.c, mesh-call.c;
  - T: rdma/mesh_compile.py, mesh-torch.mm, mesh.py;
  - E: metal-microbench tools/e2b_replay.m, metal_recording.m and .h, and mesh rdma/mesh-metal.m.
- The coordinating implementer reviewed each commit as it landed. It sent D2 one finding (a kept region's view at
  adoption), fixed in a172596f. It took the merges and notes the sub-agents' reports named: 78557dca, 08997c49,
  0a92cdd6, c5e37153 and metal-microbench 9823303f.
- Nothing was run but the compilers, `python3 -m py_compile` and the four incremental builds.

### Done

**R-1. The transfer descriptor is a flattened view** ("A transfer's source and destination are each a view (value,
offset, sizes, strides). The compilation flattens each into its list of (row, offset, bytes) blocks, adjacent blocks
merged ... The bridge's instructions are those blocks").
- **The interface (31ee5d4a):**
  - struct mesh_strided {offset, element, dims, size[8], stride[8]}, in bytes and C order;
  - struct mesh_declared {source, destination, identity, kind, value, padding, from, to}, with `first` and `bytes`
    gone;
  - struct mesh_hop `offset`, a hop moving [page × pgsz + offset, + bytes);
  - struct mesh_share and struct mesh_function `share`/`shares` gone (64 bytes);
  - each range's words named by its declared transfer (struct mesh_state `transfers`, mesh_transfer_at), with
    mesh_host_publish and mesh_host_arrived taking (plan, transfer) and mesh-metal.h's prepares taking the transfer;
  - MESH_VERSION 119.
- **The planner (C, 5eeee1f9):**
  - Each view is tested at the plan (plan_view, plan_alike) and flattened in C order with adjacent extents merged
    (plan_extents; Ross, Miller and Gropp, as PB R1 cites).
  - Each range's instructions are the common refinement of the source's and every destination's extents cut at
    their values' block boundaries, and of the range's own bytes cut at Q (plan_instructions, mesh_graph_grid_cut).
  - The fill, the template, the schedule and the walk count instructions; each hop carries its frame, `offset` and
    `bytes`.
  - The frames' items are the sets of ranges covering a block whole, ordered by PO-1 generalized to sets (plan_sets,
    plan_before). A block holding uncovered bytes stays its value's whole-life item, and a MESH_INVARIANT source is
    one copy, every copy's sends naming it ("A frame's item lives from the first write to the last read over every
    value naming it, with C3.3's lag test").
  - One struct mesh_publication per declared transfer is written before `published`; no row publication is written.
  - struct mesh_share's handling and its EINVALs are gone, as are "no two ranges at this node share a block of it",
    "MESH_EXECUTION and wire" and the lag-k rule ("Ranges may share a block where their bytes are disjoint"; "Any value
    a transfer names is sent from its own pages, whatever its copies").
- **The bridge (D1, 80872890):** each record's span is at its hop's offset in its page.
- **The torch path (T, f2a1efd1, 70b35eb7, f0523b8d):**
  - each declared transfer's `from_`/`to` is torch's offset, sizes and strides × itemsize, with dims of size 1
    dropped and nested ones merged (torch 2.14.0 torch/_prims/__init__.py collapse_view's rule), every device's places
    computed alike at every rank;
  - `_laid`, `_claim`, the padded layout (`_form`, `_grain`, `_size`), `_sent`/`_run_of`/`_within`, `_buffer`,
    `_halo`'s decision, `ir.fallbacks`, `ir.shared`, `ir.share` and mesh.Share go;
  - an AllGather's own part and a halo window's middle are written in place by their producer (`_inside`);
  - a range sent from a parameter is published once after its load;
  - mesh-torch.mm publishes by (plan, transfer).

  copy_ calls went from 5 to 2. The two left are a no-out-variant operator's one copy (appendix item 7, "What stays")
  and the parameters' one load (V4). Every torch.cat left is in a comment.
- **e2b (E, a422854f, 81f1c08f):**
  - each route's `from` = `to` = {0, the operand's bytes, dims 0};
  - the prepares are by the declared transfer's index (`productions`, `consumptions`);
  - mesh-metal.m reads entry `transfer` with no run-time test of the plan's facts (K2).

**The answers built with R-1** ("Answered in the documents (2026-11-06)").
- **Every member of a view plans on the view's map** (A7, corrected: "The map a view is created from is the
  sequencer's, and the view's first message carries it ... Every member plans on those records").
  - The interface: hdr `view_map`, `view`, `view_off`, `view_nodes`; mesh-map.h's area and mesh_view_records; struct
    mesh_group_view `records`; plan_stale by the view.
  - D2 (a172596f, 8de3c86e, 1acf8b31):
    - the sequencer's VIEW carries its map's records, and view_install stores them under `view_map`, then calls
      plan_stale;
    - map_put and map_adopt end no plan;
    - at adoption no view is installed (the coordinating implementer's finding: "A bridge started again is a new
      process");
    - mesh_nodes_read reads the view's records, its `serial` the view's identifier, ESTALE before the first view;
    - every map index a plan holds is a place among the view's records (view_index_of);
    - the start's ESTALE is by hdr `view`.
  - D1 (80872890): the ports' cancellations and plan_stale compare with hdr `view`, and the peer's index is its place
    among the view's records. The cables stay the bridge's own map's.
- **Values sent from their own pages whatever their copies:** the interface and C, as above.
  - An invariant is one copy, its word raised once (mesh_host_publish with UINT64_MAX − 1, T's LOADED).
  - A lag-k source takes the plan's u copies, its host order declared in the dependences (mesh-collective.h). T found
    no reachable case (the stops).
- **Ranges sharing a block by disjoint bytes, with offsets in instructions:** R-1, as above.
- **A share's one item with its lag:** replaced by R-1's frame items.
- **mesh-metal.m's waiter with e2b:** R-7, below.

**R-2. Several outputs' out variants** (W4; "one `out` per result"). T (f2a1efd1): an operator returning k tensors
runs its out variant, each result a value of its own. This rests on torch 2.14.0 native_functions.yaml ("autogen:
native_layer_norm.out"), torchgen's `out{i}` naming and `out` tag, and torch/_library/_out_variant.py's matching.

**R-3. Publication read as RCU's publish-subscribe** (W5; "every field of the published operand is read through the
acquired publication, never before it").
- The interface (31ee5d4a): mesh_algo_standing; plan_from acquires first, then reads `plans`.
- D1: plans_walk acquires, announces, fences and reads the operand again before `plans`, `table`, `nodes` and
  `adjacent`; place_at takes the walk's `plans`; held_clear reads after mesh_algo_unreleased.
- D2: plan_here uses mesh_algo_standing after mesh_algo_open, and mesh_holding_retained reads `plans` after the acquire.

**R-4. Instantiated once** (W6; "each execution's work becomes a straight-line sequence of its kernel calls and
publishes, with no walk of the IR and no test of a make()-time fact"). T (f2a1efd1, 70b35eb7): each compiled function
is instantiated once a copy into straight-line steps (`_instantiate`, over mesh_graph.forward), with views of constants
evaluated at make(). W6's branches (2156, 2598, 2605, 2606, 2635) are make()-time choices.

**R-5. The alias flush where a frame changes virtual name** ("goes on the barrier between a write under one name and a
read under another, and on no other").
- The interface: struct mesh_value `shares`, the plan's statement.
- E (a422854f, bfe9b4b9):
  - each placement carries its value's `shares`;
  - replay: takes each shared value's handover, the first command of the level after its last level, from
    MetalPrepareDataflow's own walk, read by index;
  - the flush (MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias) goes on that barrier alone: an
    intra-pass barrier between two executes, a producer barrier where a range ends on it, or after the later pass at a
    cut;
  - every other barrier is Device alone.

  An indirect command's `setBarrier` takes no visibility (MTLIndirectCommandEncoder.h:102), so the flush is the
  encoder's barrier at the same level boundary (MTL4CommandEncoder.h, `barrierAfterEncoderStages:` and
  `barrierAfterStages:beforeQueueStages:`).

**R-6. The bridge runs A9's self-timed lists** (W8; "It scans no table of plans").
- The interface: struct mesh_algo `nodes` and `adjacent`, and mesh_algo_adjacent.
- C (5eeee1f9): the places with an entry toward each map index, mesh_graph_rows keyed by map index, placed in the table's
  item and written before `published`.
- D1 (80872890): lane_take, link_run's cancellation loop and busy test, the walk's ending pass and link_close visit only
  the places mesh_algo_adjacent gives for the peer's index, and plans_walk loads only those.

**R-7. Every wait ends at MESH_PLATFORM_NS without progress, the first ETIMEDOUT ending the run** (W10).
- The interface: mesh_plan's `deadline`, and mesh_transfers_start without one.
- C: plan_released waits until the caller's deadline.
- D2 (8de3c86e): each of the start's waits ends MESH_PLATFORM_NS after the last rise of its count.
- T (f2a1efd1):
  - make()'s retirement and mesh_plan share one deadline, taken after the compilation;
  - `_arrived` and the inputs' wait end without a rise of their count;
  - after the first ETIMEDOUT every later run() and inputs() raises at once.
- E:
  - step_wait waits for the step event's next value, MESH_PLATFORM_NS each time, with no multiple of units;
    `platform_bound`, `end_units` and `holds` go;
  - the waiter (mesh-metal.m, 81f1c08f) restarts its bound at each rise of the count it reads;
  - after the first ETIMEDOUT no later wait runs;
  - mesh_plan takes now + MESH_PLATFORM_NS.

**R-8. One writer a word** (W11; "the program writes `retired`, the bridge writes only `released`, and a gone holder is
read from the heartbeat's words").
- The interface: mesh_algo_open false once released; mesh-holders.h mesh_algo_holder.
- D2 (a172596f, 1acf8b31, 8de3c86e):
  - clients_turn, clients_gone and region_adopt retire nothing;
  - a gone holder raises hdr `plans` once;
  - mesh_retired_release releases a retired operand or one whose holder is gone.
- D1: a plan stands where its operand is open and its holder holds the region (plan_stands, in a walk: one
  proc_pidinfo a walk).

**R-9. Names** (W12). T: `slots` and `_claim` are gone. The coordinating implementer, on D1's finding (78557dca),
renamed mesh-flow.c's inherited `slot_set`, `slot_position` and `slot` to `position_set`, `position_index` and
`record_of` (the operator's "no 'lease', 'claim' or 'slot' from old code without the canonical structure and
citation").

**R-10. The duplicates** (W13).
- "Published and not yet released" is mesh_algo_unreleased, used by C, D1 and D2 at W13's three sites.
- The holder's read and the gone-holder test are mesh_algo_holder and mesh_holds at every site: clients_turn,
  clients_gone, region_adopt, mesh-stat, the release and plan_stands.
- The mapping's row list is mesh_view_values, defined by D2 and used by T's `_mapped` and E's mapping.
- From the reports:
  - the seqlock reader's test, written twice over `view_map`, is one form over its sequence word (mesh_seqlock_whole,
    78557dca);
  - C's supplied-word, phase-group and count-word arithmetic is one helper each.

### Found already done

- mesh_plan wrote neither `retired` nor `released` (C), and mesh-flow.c wrote no `retired` (D1).
- mesh-call.c needed no change (D2). mesh-graph.h and mesh-graph.c needed none (C): mesh_graph_rows and
  mesh_graph_grid_cut served.

### Decisions taken where the documents are silent (each the named implementer's, with its reason)

The coordinating implementer's (the interface):
- **The view's structure and name.** struct mesh_strided, offsets and strides in bytes and C order (the order torch's
  copy_ matches two tensors' elements).
  - It is named for torch's strided layout, since struct mesh_view already names the mapping.
  - Eight dims after coalescing, a compile-time shape.
  - `bytes` goes, the views stating it.
- **Both ends' views are every rank's.** A multicast's instructions must be the same at its source, its relays and its
  destinations, so every rank computes them from every end's view (R-1: "Every member of a view plans on the view's map
  ..., so both ends' blocks agree by construction"; Schneider 1990). The source view is repeated in every transfer of
  its range, tested alike.
- **The instructions are the common refinement**, each one extent within one block at every end, as an MSCCL-IR step
  names one offset and one count.
  - The range's own bytes are also cut at Q, so a relay holds the range in its bytes' order in ceil(B/Q) blocks.
  - That costs a cut only where a source and a destination share one misalignment.
- **A frame's item is the set of ranges covering a block whole**, ordered by D_XY = max D_ab and E_Y = max E_b
  (transitive through b's write). A block holding uncovered bytes lives the program's whole life, as `uncovered` did.
- **Invariant and input sources.**
  - An invariant named by a transfer keeps one copy, its word raised once with execution UINT64_MAX − 1.
  - A lag-k source takes the plan's u copies, its host order in the dependences: u must be every rank's, and a value's
    lag is one node's.
  - A destination is MESH_EXECUTION of lag 0, a landing writing its bytes every execution (C3.7).
  - Every value a transfer names is in the window whatever its `wire`.
- **A range's words by its declared transfer** (struct mesh_state `transfers`). Ranges sharing a block begin in one row,
  so a row names none of them alone; the declared transfer's index is the IR's own name.
- **R-6's list** is compressed rows by map index in struct mesh_algo `nodes` and `adjacent`.
- **R-8:**
  - mesh_algo_open is false once `released` is `published`, since a gone holder's operand is never retired now;
  - mesh_algo_holder is the one holder read;
  - the parent raises `plans` (a count) for a gone holder.
- **R-7:** mesh_plan takes the preparation's one deadline; mesh_transfers_start takes none.
- **A7:**
  - a plan's serial is the installed view's identifier;
  - a node's own map move ends no plan ("A plan then stands while its view does");
  - at adoption no view is installed and plan_stale runs (the finding to D2).
- **R-5's interface stays `shares`.** The flush goes on the barrier after a sharing value's last access, and the next
  name follows by the allocator's order, so the barrier's place needs no partner (W7 had said "The plan names each
  share's partner").
- **mesh_host_publish and mesh_host_arrived test no entry** (08997c49, on C's finding). The caller names only a transfer
  its node is an end of, which its compilation knows ("Compile-time, never runtime checks").
- **The seqlock test is one form** (mesh_seqlock_whole), and mesh-flow.c's 'slot' names are renamed (78557dca, on D1's
  findings).
- **The torch rule's inputs.** rdma/mesh-torch.mk no longer lists mesh-section.h (c5e37153, on T's finding), since
  mesh-torch.mm no longer includes it ("A dependency is remade only when its own inputs change").
- **e2b's refusal of a routed host input stays, its reason restated** (9823303f, on E's finding). The plan call's rule
  it cited is withdrawn, but e2b's dependences are each stage's dataflow (tools/prepare_e2b.py), which names no host
  write, and no prepared manifest routes such an operand.
- MESH_VERSION 119.

C's:
- **The fill and the weights.** The fill cuts instructions by count, one unit each, and each instruction weighs what a
  page weighed in the schedule and times. `down[d]` in the flow solve is B, where it was pages × Q.
- **The transfer table** lies after the supplied words and before the status, inside the GPU transport's wrap.
- **Tests kept and not added.** A named value with `state` 1 stays EINVAL. Overlapping destination bytes are not
  refused, since the header lists no such EINVAL.
- **Item order.** Blocks of one set of ranges are one item a copy, their elements in (value, block) order.

D1's:
- **The view test moved to the walk.** The walk compares the operand's view (its plan 0's `serial`) with the installed
  view once, before using the index, and link_prepare's per-plan test goes as redundant, with no `refused[p]` written.
- **The announcement is once a walk.** Its re-read and clear are once a walk; plan_load's own re-read and EAGAIN go.
- **One load an operand.** An operand's plans are loaded once, since `closed` only rises.
- **`holding`**, so a pass stops reading an operand's list once no flow of it is held.
- **No reload after a loss.** link_close keeps the operand's generation, so the same child never reloads its plans.
- **One seqlock reader** (records_read) for the map and the view.

D2's:
- **ESTALE before the first view** in mesh_nodes_read (A7's Initial View Event).
- **Plans' map indices are the view's records**, in arrivals_apply, orders_announce, counts_tell, orders_submit and
  orders_deliver. The island's tree stays on the map.
- **One store of the counter.** `installed.view` is merged into hdr `view`, one store of the bridge's counter.
- **One raise a gone holder** (`gone_seen`), so the parent does not spin.
- **The bridge's own-pid test goes.**
- **The VIEW's shape test** extends to its records (an exact byte count, at most MESH_MAP_ROOM). Its size is checked
  against MESH_TOPOLOGY_MESSAGE_BYTES by a _Static_assert.

T's:
- **mesh.Publication's mirror** (its fields before `targets`, 64 bytes), so the host's progress test reads the count
  word.
- **Producers that write an own part in place:**
  - an out variant;
  - one result of a multi-output operator;
  - a no-out-variant operator, its one copy going there;
  - a ReduceScatter sum.
- **No-out-variant results** get placed pages only where a collective reads them.
- **The deadline after the compilation.** make() compiles first, then takes its deadline.
- **ETIMEDOUT ends the program's run.**
- **_forms and the runner's instantiation** are on mesh_graph.forward, on the C9 hook's remarks on f2a1efd1.

E's:
- **Placements in (offset, stride).** The recorder's placements take (offset, stride), mesh_view_values' own statement
  of where copies lie.
- **A handover in each capture.** An earlier capture's value bound by a later capture hands over in each capture that
  binds it.
- **The profile** skips publishing commands, so a flush right after a skipped command is not issued; the profile's
  values are read by nothing.
- **A wait's bound** starts at its first read of the count.

### Where the work stopped

T refuses three cases at compile time, naming them. In each, a part must lie in another value's bytes, and R-1 removes
the copy and the share that put it there.
- **Every AllToAll.** A member's own piece lies in the result only by a copy or a share.
  - R-1: "That removes `_laid`, `_claim`, `ir.fallbacks`, struct mesh_share with its tests on both sides, the padded
    layout and AllToAll's buffer"; W2: "Each piece must land at its byte ranges in the result".
  - An operand strided so its own piece sat in the result would overlap the parts that arrivals write.
- **An AllGather's own part** whose producer writes no place of its own: a view an operator makes (`aten.view` after mm),
  a slice, or a collective's result other than ReduceScatter's.
- **A halo window's middle that cannot be written in place**, including a convolution over the sharded program input.
  Its operand is a slice of a placeholder that is whole at every node, so halos landing there write the same bytes
  twice. The brief's "an input is written there by its host producer" has no form against that fact. 2-D spatial halos
  stop here too.

Inputs and parameters as transfer sources are reached only through the halo stop. So no HOST_LAG edges are written, and
the publish-once path is unreachable.

### Where the documents are wrong, or contradict the code's facts

- **Rows alias at block granularity; views are byte-granular.**
  - R-1 and PB R2 (2026-11-06): "Two values' rows may name one frame, the page table's sharing ... with no record and
    no rules of its own".
  - The page table binds a row to a frame, a block, so two values can name one frame only where their bytes are whole
    blocks at the same block offsets in both.
  - An own piece or part at another offset has no frame to share. With the share record and the copies gone, the
    documents give it no form: T's stops.
- **A plan stands while its view does, and a link loss is not a view's end.**
  - A7 (corrected): "A plan then stands while its view does, and its view while the sequencer's map does".
  - The code's fact (D1): a port child spawned after a loss, before the next view is installed, finds the plans'
    serial equal to the view and loads them with cursors at execution 0. It cannot know which executions landed.
  - link_prepare's SW note relied on "the loss having moved the map's serial ... and refuses it, ESTALE". D1's
    link_close covers the same child; a newly spawned one is open.
- **A view's identifier can repeat** (D2): after every bridge of the island restarts, every counter is 0 again. A plan of
  an earlier process's view with the same identifier, unretired by its program, would stand again. The documents are
  silent.
- **The header's own notes, corrected as found** (ea330a9f, 08997c49, 0a92cdd6):
  - mesh_map_whole read `map` alone;
  - the release, retirement and map-reader notes still named the bridge as a retirer and the map serial;
  - mesh_host_publish's "Nothing where this node is not the transfer's source" contradicted "no run-time test";
  - the waiter's note still had the per-value bound.

### Open

- **The C9 hook's remarks on bfe9b4b9.** MetalPrepareDataflow's level walk predates this unit and is outside the graph
  module. design/disjoint-pages-revision-coordination.md:953 records that moving it needs libmetal_recording to link
  libmesh. replay: now only reads that walk's facts by index (E).
- **A layout change for a contiguous consumer.** A part written in place is a strided view, so a consumer needing it
  contiguous fails in torch at make(). PB R1's operator for a layout change ("an operator of the consumer's own, run as
  its out variant") is never inserted (T).
- **Zeros at edge partitions.** The zeros in an untargeted arrival slot and the window's fill write the same bytes twice,
  as before this unit (T).
- **A program before the first view.** mesh_nodes_read is ESTALE before the first view, so a program started before its
  bridge installs one fails at construction and plans again (T, D2).
- **e2b's routed host inputs** wait on its dataflow naming the host's write (the decision above).
- **Measurements for when runs are authorized:**
  - instruction counts of strided views;
  - the walk's proc_pidinfo;
  - aliased virtual addresses across torch's encoders, as before.

### Commits

- mesh:
  - the coordinating implementer's: 31ee5d4a and 11b6453b (the interface), ea330a9f, 08997c49 and 0a92cdd6 (notes),
    78557dca (mesh_seqlock_whole; the 'slot' names), c5e37153 (mesh-torch.mk), and this record;
  - C: 5eeee1f9;
  - D1: 80872890;
  - D2: 8de3c86e, a15b5c8f, 1acf8b31, a172596f;
  - T: f0523b8d, f2a1efd1, 70b35eb7;
  - E: 81f1c08f.
- metal-microbench: E's a422854f and bfe9b4b9; the coordinating implementer's 9823303f.

### Integration

Every build was incremental. HiGHS and the LiteRT/Dawn SDK were not rebuilt.
- **`make -C rdma`:** exit 0, no warning.
  - The first run after C, D1 and D2 took 2.5 s and remade mesh-flow, mesh-heartbeat (its mesh.h changed), mesh-stat
    and libmesh.dylib.
  - The run at the end remade nothing.
  - The codesign line's "Mesh Bridge: no identity found" is the rule's fallback to an ad-hoc signature, as before.
- **`make -C rdma native-audit`:** exit 0, no warning, 0.5 s, mesh-flow.s, mesh-call.s and mesh-dataflow.s remade. The
  run at the end: nothing to be done.
- **`make -C rdma -f mesh-torch.mk TORCH=/Users/mdot/.venv-mesh-uv/lib/python3.12/site-packages/torch`:** exit 0, no
  warning, 1.0 s. libmesh-torch.dylib was remade (mesh-torch.mm and mesh-torch.mk changed); the run at the end: up to
  date.
- **`make .build/e2b_replay`:** exit 0, no warning, 1.7 s. libmetal_recording, libmesh_metal and e2b_replay were remade;
  the run at the end remade nothing.
- **Compiled alone** with `-Wall -Wextra -Werror`, exit 0:
  - mesh-collective.c, mesh-flow.c, mesh-records.c, mesh-dataflow.c, mesh-link.c, mesh-stat.c and mesh-call.c;
  - mesh-graph.c, with HiGHS's include path as the Makefile gives it;
  - the failport lifecycle files, unchanged since 14b9a757 (mesh-heartbeat.c, mesh-children.c, mesh-child.c,
    mesh-lanes.c);
  - mesh-ports.c and mesh-events.c;
  - mesh-metal.m;
  - mesh-torch.mm with `-std=c++20`;
  - metal_recording.m with metal_native.m, and tools/e2b_replay.m.
- **Python:** `python3 -m py_compile` passes rdma/mesh_compile.py, mesh.py and mesh_graph.py (T: pyflakes clean).
- **Nothing was run.** No view, instruction, frame item, transfer entry, adjacency list, wait, release, flush or
  instantiated step here has been exercised.
