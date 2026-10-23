# Extents and liveness: what a program's pages are, and when they are free

This document states, with the literature behind it, how the mesh knows which pages a collective program uses and when
each page is free. It is why the bridge assumes no arbitrary access pattern, uses no error state, and waits on data
alone. The memory model is [pages-and-functions.md](pages-and-functions.md); the transport is
[transport.md](transport.md) section 14.

The operator, 2026-10-22: "some algorithm tries to assume that all prorgams have arbitrary access patterns and
communication schedules for the sake of breadth, but in so doing relaxes the problem socpe they are working on so
severely that they describe a less stable or worse-converging algorithm". Sections a to e follow the points the
operator gave the same day, verbatim:

> a: a collective program can assert that it would hope to use extents a,b,c,d,e as scatters, writes, whatever.
> b: a collective program can dynamically pass less data than the extents it hopes to use.
> c: a collective program's access pattern can be derived from the compute graph through boring graph analysis
> algorithms and the explicit knowledge of its operands and function behaviors, up to the limits of data dependent
> functions returning extents.
> d: a program which has halted is in a special kind of callgraph state where we can treat its successive dependency
> and write relationships as 'maximally underused'; those extents *aren't* being 'requested' by that program anymore
> even if they might have been used if it wasn't halting.
> e: given these constraints, we always know more information about what extents could be under use or in-flight for
> a program the more we know about its operands and data-independent functions(pages)->pages; we can for some
> functions know so much that we can apply performance optimizations, like asynchronously freeing its pages partway
> through some data lflows.

## Terms

- **Page, row, extent.** A page is the unit the page table places, `block · pgsz` bytes. A row names one page, its
  stamp and its use count. An extent is a range of rows.
- **A collective program P.** A graph of functions over operands. Each function maps input pages to output pages.
- **D(P), the declared extents.** The extents P names when it is called: those it sends (gathers), those it lands
  into (scatters), and those it writes. They come from its operands and configuration.
- **U(P, t), the used extents.** The part of D(P) the calls of P actually move or write, as of time t.
- **L(P, t), the live extents.** The extents that some event of P after t may still read or write.
- **W(t), the in-flight extents.** The extents that requests already handed to the hardware reference: receives
  posted and not landed, SENDs posted and not completed.
- **Free(t).** The rows in neither L nor W. A page in Free(t) may be given to anyone.

## a. A program declares its extents when it is called

D(P) is known at the call. The operands and the configuration name every extent the program may scatter into, gather
from or write. Nothing the program does later adds to D(P). The operator, 2026-09-10 (pages-and-functions.md,
Setup): "all allocations needed within a page table for a complete compute graph should be known a priori at the time
a callgraph using these apis has been launched".

Literature:
- **MPI-4.** It declares buffers ahead of use. A partitioned send is set up once with `MPI_Psend_init` over a buffer
  and its partitions. `MPI_Pready` then marks each partition ready, and only ready partitions move ([MPI 4.1, 5.2,
  partitioned communication][mpi-part]). Persistent collectives such as `MPI_Allreduce_init` bind a collective's
  buffers and schedule once, for many starts ([partitioned collective communication, Sandia/MPI Forum][part-coll]).
- **Verbs.** A memory registration declares the extents a NIC may access, before any request names them.
- **GSPMD.** It shards tensors to statically known, capacity-shaped extents, including uneven ones, because the
  accelerator needs static shapes ([Xu et al. 2021][gspmd]).

## b. A program may move less than it declared

U(P, t) ⊆ D(P). A call may pass fewer elements than its extent holds. The counts are operands of the call, so the
difference is known when the call is made, not discovered later.

Literature:
- **v-collectives.** `MPI_Allgatherv` and its kind take per-rank counts within declared buffers. Our
  `ncclMeshAllGatherV` and `ncclMeshReduceScatterV` are the same.
- **Partitioned communication** moves only the partitions marked ready ([MPI 4.1][mpi-part]).
- **Uneven partitioning** pads each shard to a static capacity, and only the valid part carries values ([GSPMD][gspmd];
  torch-mesh `partition.py`, capacities and parts).

## c. The access pattern follows from the graph

For functions whose index maps do not depend on data, L(P, t) follows from the compute graph and the operands by
standard dataflow analysis:
- **Def-use and liveness.** Kildall's framework computes, for every program point, which values a later point may
  still use ([Kildall 1973][kildall]).
- **Arrays with affine index maps.** For each read, Feautrier's array dataflow analysis gives the exact write that
  produced the value, as a function of the iteration vector ([Feautrier 1991][feautrier]). For such functions the
  live extents are exact, not an over-approximation.
- **Region inference.** It assigns every value a region and a lifetime statically, with allocation and freeing
  points decided at compile time ([Tofte and Talpin; Tofte et al., retrospective][regions]).
- **Production compilers do this for tensors.** XLA's buffer assignment computes each buffer's live range from the
  schedule and shares memory between buffers whose ranges do not overlap ([XLA heap simulator][xla-heap]). Memory
  planners generally take a buffer's lifetime from its first write to its last read ([PlanMemory][planmemory]).

The limit is a data-dependent function that returns an extent, such as a routing count or a top-k index set. Its
output extent is still bounded by its declaration (a), and the analysis keeps that bound until the count is known (b).

## d. A halted program uses nothing after it halts

A program that has halted has no future events, by exit, by revoke, or by the page table's invalidation (the
operator, 2026-09-10, pages-and-functions.md, Setup: "our mesh api level primitives should let us invalidate a page
table in this manner whenever we want to"). So L(P, t) is empty from the halt on. Every extent it might have used, had
it run on, is no longer requested by it: in the operator's words (d, above), "maximally underused".

Only W(t) remains: requests already handed to the hardware.
- **Verbs.** The hardware owns a buffer that a posted work request names until that request completes. It may write a
  posted receive's buffer whenever a matching SEND arrives. So it is reused only after the completion, or after the
  queue pair that holds the request is gone ([rdmamojo, flushing work requests][rdmamojo];
  [Linux ib_drain_qp][drain]).
- **Our transport.** W(t) is the plan's in-flight window: at most a lane window of units per lane, each named by the
  plan.

So at a halt, Free = every row of the program except W(t). Those rows are free at once.

## e. More knowledge, earlier release

The live set computed with more knowledge is contained in the live set computed with less. Knowing a program's
operands and that its functions are data-independent (pages to pages) shrinks the over-approximation. In the limit it
is exact.
- **Last use.** When a row's last use is known statically, the row is released at that use. That can be partway
  through a dataflow, asynchronously, as the dependent's stamps arrive: steps 6 and 7 of pages-and-functions.md
  ("release ... when the dependent's rows are stamped").
- **Memory schedules.** Chen et al. trade recomputation for memory by schedule ([Chen et al. 2016][chen]), and
  Checkmate solves for optimal rematerialization schedules on arbitrary graphs ([Jain et al. 2020][checkmate]). Both
  rest on knowing every tensor's uses from the graph.
- **Over-approximation.** A design that assumes nothing about the program over-approximates. It treats every row
  the program owns as possibly live, and every request as possibly unfinished. That is the weakest analysis, with the
  most state and the most waiting.

## f. A call's end travels along its dependency graph

### The mesh is a graph that changes
The operator, 2026-10-22: "you need to treat the mesh as more distributional and having perpetually changing edges and
topologies for all of time. this will expose new and unseen errors and new and unseen requirements, which is why the
verbatim guidance from the user that all serivces must handle all graphs must be used rather than superficial evidence
from one configuration of one environment that programs 'happen to work'".

Nothing below relies on a configuration's shape, such as a pair or a triangle. Nodes join and leave, and links appear
and go. A call's routes are tables compiled onto the graph as it is when the call is made, as collective compilers do
for each topology ([TACCL][taccl]; [MSCCL-IR][msccl]).

### The plan's dependency graph
A call's plan is a DAG of hops. Each hop moves a piece over one link, from an extent at its sender to an extent at its
receiver. On a piece's route a node is one of:
- **a source**, which makes the piece;
- **a forwarder**, which receives it and sends it on unchanged, with no reduction;
- **a combiner**, which reduces several pieces into one before sending on;
- **a terminal**, which consumes it.

For every node, the plan says which received pieces feed which sent pieces: the identity for a forwarder, the
reduction's inputs for a combiner. This "feeds" relation is static plan data.

### Conclusion needs no message
When a call succeeds, every extent is released locally, at its last use (e):
- a forwarder's extent for a piece, when its out-hop's SEND completes;
- a combiner's input, when its reduction has read it;
- a terminal's extent, by its consumer's use count.

Nothing travels. This is the earliest possible release, and it costs nothing.

Dijkstra and Scholten detect when a diffusing computation has ended by sending acknowledgements back to its origin.
That is needed because such a computation's extent is not known in advance ([Dijkstra and Scholten 1980][ds]). A
compiled plan knows its extent, so each node knows its own last use and needs no acknowledgement.

### Failure: the halt's cones
The operator, 2026-10-22: "we must now explain to ourselves how we will propagate shutdowns at fastest possible speed at
the earliest point a dependency structure makes explicit that we can start freeing a concluded-or-failed call".

A halt at a node ends the sources of some hops. A halt is the node's holding ending, its call revoked, or a link on a
route lost. From the halted hops:
- **The forward cone F** is every node and piece the feeds relation reaches downstream. Those pieces will never arrive.
- **The backward cone B** is every node and piece upstream whose only consumers lie in F. Its work is now for nothing.

No extent in F or B has a future use (d). Each is free at once, except what that node's W(t) holds.

**Earliest knowledge.** By happened-before, a node can learn of the halt no earlier than a chain of messages from the
halt reaches it ([Lamport 1978][lamport]). The pieces themselves would have travelled the routes. So a notice that
travels the same routes, cut-through, reaches each node at the earliest time any signal could.

**The fastest propagation is:**
1. The halted hops' nodes originate a notice for the call that names the hops that ended.
2. A node that receives it forwards it at once, before anything else, with no batching and no wait. It goes along
   every out-hop in F (downstream) and every in-hop in B (upstream), by the plan's feeds relation.
3. As it forwards, the node acts locally:
   - it ends the call's holdings on those edges, posting nothing further for them;
   - it frees every extent of the cones outside its W(t);
   - it reports the status in its callers' `meta`.
4. A notice names its call, so a second copy is dropped. Each node acts once per call.

**What the notice is.**
- It is a fact about the call's plan, not a message about data: no node waits on it to use data, it only ends and
  frees (pages-and-functions.md, the addendum on messages about data).
- It travels on the bridges' control channel, never on a data lane, because a lane carries only plan pages.
- ULFM's revoke propagates over the communicator by reliable broadcast, and Ray's ownership carries a failure from
  owners to borrowers; both carry the failure along the system's own graph ([ULFM][ulfm-sc18];
  [Ray ownership][ray-own]). Here that graph is the plan's feeds relation, already compiled, so the cones are computed
  rather than discovered. The broader setting is distributed garbage collection's reference graphs
  ([Abdullahi and Ringwood 1998][dgc]).

### What W(t) holds under a halt
At a node, W(t) holds the requests already posted on the cones' edges: receives posted toward an upstream hop that
will never send, and SENDs its downstream will never receive. Their rows are released when the requests complete or
when their queue pair is gone (d).
- The notice reaches both ends of each cone edge, so both end that edge's stream. No stream is left dead at one end
  and live at the other.
- On a link with one stream, that stream's end loses the link: the port process exits, and the next one pairs from 0.
- On a link with several streams, a dead stream stays dead until its port process ends. Its W(t) is bounded by one
  window.

### Topology changes during a call
A link lost on a route is itself a halt. Its two ends originate notices into their cones. A node that joins during a
call holds no hop of that call's plan; it joins the next call's plan.

### What each layer does
- **The collective** compiles, with each call's plan, each node's feeds relation per piece. For each holding it gives
  its bridge the edges that depend on each edge.
- **The bridge**, on a holding's unclean end or on a notice:
  - ends the dependent holdings;
  - frees rows outside W(t);
  - forwards the notice on the dependent edges' control channels.
  
  No timing, no inference, no waiting.
- **The caller** reads the status in `meta` and decides whether to rerun.

### Not yet measured
- **The control channel's hop latency,** which bounds the notice: h hops cost h of it. Data lanes would be faster,
  and plan-only lanes rule them out.
- **The time from a halt to the last row freed** in its cones, on graphs of several shapes.

## What follows for the mesh

1. **The bridge assumes no arbitrary access pattern.** The rows any request on a queue pair can reference are
   W(t), the plan's window. Nothing else is pinned.
2. **When a holder gives or dies (d), every row of its holding outside W(t) is free at once.** A row in W(t) is free
   when its unit completes, or when the link's port process exits and its queue pairs are gone.
3. **No error state is used.** The documented queue-pair flush (move to Error, poll every `IBV_WC_WR_FLUSH_ERR`)
   would only end W(t) early. The operator, 2026-10-22, of that flush as the literature states it: "you can do this and nothing else with
   error states and if and only if it is exactly what is supported by the literature". The plan bounds W(t), and the
   flush is unused.
4. **A row is released at its last planned use (e), not at a holding's end.** The use count of pages-and-functions.md
   is this analysis's result.
5. **A prediction's mismatch is a statistic in `meta`.** A compiler's prediction of the page-table mapping can be
   checked against the facts the bridge reports: its page size, its rows' pages and stamps. Examples are contiguity, or
   which rows a crossing lands in. A mismatch is "an error channel return kind of statistic, not a data dependent
   control flow for a collective verb or a bridge transaction" (the operator, 2026-10-22), returned in `meta`.

## Dominating methods and choices

The operator, 2026-10-22: "connection to the prior literature is important for understanding what are choices and
what are strictly dominating superior methods we must use". A method is strictly dominating when it is never worse on
any measure, so there is nothing to trade. A choice trades one measure against another; a measurement or the balancer
decides it.

Strictly dominating:
- **Freeing every row outside W(t) at a halt (d).** It releases earlier at no cost: no request references those rows.
- **Releasing a row at its last use, when that use is known exactly (c, e).** It is never later than releasing at a
  holding's end. XLA's buffer assignment and region inference take this for granted.
- **Using the declaration and the counts (a, b) instead of inferring them at run time.** The facts are exact, an
  inference can only be equal or wrong, and the inference costs time.

Choices:
- **Rematerialization.** It trades recomputation for memory ([Chen et al. 2016][chen]; [Checkmate][checkmate]). Only a
  memory budget decides it.
- **The documented flush.** It ends W(t) early at the cost of a dead stream. The plan bounds W(t), so it is not chosen
  here.
- **The size of the posting window.** A larger window hides more latency and keeps more rows in W(t) at a halt.

## What the rejected designs did

Each took the most general program as its model, against (a)–(e):
- **A pin on every row a dead client owned (2fb2b270, corrected).** It treats D(P) as everything P held and ignores
  (d).
- **A stream waiting for a future, unrelated communicator's filler before its rows were free (2fb2b270, deleted).** It
  treats W(t) as unknown, which the plan already names.
- **Deadlines, keepalives and lane counts sampled at a holding's start (before 2fb2b270, deleted).** They treat time
  and racing samples as information about the program, which the program's declaration (a) and counts (b) already
  give exactly.

[mpi-part]: https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node87.htm
[part-coll]: https://www.sandia.gov/research/publications/details/partitioned-collective-communication-2021-10-01/
[gspmd]: https://arxiv.org/abs/2105.04663
[kildall]: https://courses.cs.washington.edu/courses/cse501/15sp/papers/kildall.pdf
[feautrier]: https://link.springer.com/doi/10.1007/BF01407931
[regions]: https://elsman.com/mlkit/pdf/retro.pdf
[xla-heap]: https://android.googlesource.com/platform/external/tensorflow/+/c56c7c878a8%5E%21
[planmemory]: https://ascendnpu-ir.gitcode.com/en/developer_guide/features/PlanMemory/PlanMemory.html
[rdmamojo]: https://rdmamojo.com/?p=1498
[drain]: https://lkml.iu.edu/1511.2/00504.html
[chen]: https://arxiv.org/abs/1604.06174
[checkmate]: https://arxiv.org/abs/1910.02653
[taccl]: https://arxiv.org/abs/2111.04867
[msccl]: https://arxiv.org/abs/2408.11008
[ds]: https://www.cs.utexas.edu/~EWD/transcriptions/EWD06xx/EWD687.html
[lamport]: https://amturing.acm.org/p558-lamport.pdf
[ulfm-sc18]: https://icl.utk.edu/files/print/2018/ulfm-sc18.pdf
[ray-own]: https://www.usenix.org/conference/nsdi21/presentation/cheng
[dgc]: https://www.cs.kent.ac.uk/people/staff/rej/gcbib/gcbib.pdf
