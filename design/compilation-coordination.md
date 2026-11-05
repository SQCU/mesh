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
/* plan-to-bridge R4: the kind of a declared transfer's consumer at its destination.  MESH_JOIN: function A, one task
   with an edge from every page of the range.  MESH_PAGES: function B, one task per page, each with an edge from its
   own page */
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
  GPU's add of 1), so no e2b function streams its outputs page by page, which is what makes a consumer function B.
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

- **S1. Which pages.** A range is a set where every declared transfer of it is MESH_PAGES (function B at every
  destination, R4). A range with any MESH_JOIN destination keeps the schedule's order (the appendix: "Function A's
  operands (joins) keep the schedule's order").
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
  destination is R4's function B: each AllReduce partial to the root, consumed by the root's combine (R4: "reductions'
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

#### Item 5, e2b's function B consumers (E2)

- **EB1. The kinds** (tools/prepare_e2b.py). A kept send route's consumer at each destination is function B,
  `"pages"`, where the route is a piece of a section cut over several contributors and its destinations' consumer is
  the sum of the pieces (prepare_e2b.py combine: the FFN row split's contributions and the head split's attention
  partials, added by ADD at each rank holding the stream): R4, "reductions' partial sums", ⊕ associative and
  commutative (Gibbons 1996); the item: "e2b's reductions of partial sums are function B". Every other route's
  consumers are `"join"` (function A): a whole operand broadcast, a single owner's activation, the split-KV partials
  and normalizers (kv_merge), the vocabulary's routes. The kinds are given once the routes' destinations are cut (the
  publication's readers), one a destination.
- **EB2. e2b's IR** (tools/e2b_replay.m): e2b_placement carries `kinds` with each route; e2b_ir gives each declared
  transfer its destination's kind. A route without `kinds`, one a destination, or with another word, is refused (exit
  2, naming it: prepare the directory again), as a stage without its dataflow list is. P1's "every e2b consumer
  MESH_JOIN" is reversed, with a dated note: the kind is the consumer's (R4; P3's reason), not the producer's one
  publish.
- **EB3. Not declared function B:** the split-KV merge (kv_merge) reads each rank's partial with its normalizer's
  rescale, a combine R4's FlashAttention example would make function B; the item names e2b's reductions of partial
  sums, so it stays function A (open for the operator).

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
