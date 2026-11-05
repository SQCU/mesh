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
  sequencer's own move.
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

(filled as the work lands)

### Reviews and where their findings went

(filled as the reviews land)

### Integration

(filled at the end)

### Open at the end

(filled at the end)
