# Extents and liveness: what a program's pages are, and when they are free

This document states, with the literature behind it, how the mesh knows which pages a collective program uses and when
each page is free. It is the reason the bridge never assumes an arbitrary access pattern, uses no error state, and
waits on nothing but data. The memory model is [pages-and-functions.md](pages-and-functions.md); the transport is
[transport.md](transport.md) section 14.

The operator, 2026-10-22: "some algorithm tries to assume that all prorgams have arbitrary access patterns and
communication schedules for the sake of breadth, but in so doing relaxes the problem socpe they are working on so
severely that they describe a less stable or worse-converging algorithm".

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
from or write. Nothing the program does later adds to D(P). The memory model requires this: "All page-table
allocations needed by the complete compute graph are determined before launch".

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

A program that has halted has no future events, by exit, by revoke, or by the page table's invalidation ("The calling
process may invalidate the page table whenever it chooses", pages-and-functions.md). So L(P, t) is empty from the halt
on. Every extent it might have used, had it run on, is no longer requested by it. In the operator's words, it is
"maximally underused".

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
- **Over-approximation.** A design that assumes nothing about the program must over-approximate. It treats every row
  the program owns as possibly live, and every request as possibly unfinished. That is the weakest analysis, with the
  most state and the most waiting.

## What follows for the mesh

1. **The bridge never assumes an arbitrary access pattern.** The rows any request on a queue pair can reference are
   W(t), the plan's window. Nothing else is pinned.
2. **When a holder gives or dies (d), every row of its holding outside W(t) is free at once.** A row in W(t) is free
   when its unit completes, or when the link's port process exits and its queue pairs are gone.
3. **No error state is needed.** The documented queue-pair flush (move to Error, poll every `IBV_WC_WR_FLUSH_ERR`)
   would only end W(t) early. The plan bounds W(t), so the flush is permitted (the operator, 2026-10-22) and unused.
4. **A row is released at its last planned use (e), not at a holding's end.** The use count of pages-and-functions.md
   is this analysis's result.
5. **Predictions are reported, not obeyed.** A compiler's prediction of the page-table mapping can be checked against
   the facts the bridge reports: its page size, its rows' pages and stamps. Examples are contiguity, or which rows a
   crossing lands in. A mismatch is "an error channel return kind of statistic, not a data dependent control flow for
   a collective verb or a bridge transaction" (the operator, 2026-10-22), returned in `meta`.

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
