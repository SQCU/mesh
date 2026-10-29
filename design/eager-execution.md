# Eager execution: why an undeclared collective cannot be made safe

This document ranks the ways a program can call a collective. It shows why an eager call cannot be made safe on
one-sided writes, and why the programs we run never need one. An eager call names its buffers and counts only when it
runs. The memory model is [pages-and-functions.md](pages-and-functions.md). What a program declares, and when its pages
are free, is [extents-and-liveness.md](extents-and-liveness.md).

The operator, 2026-10-28, verbatim:

> the rules as written are almost certainly meant to *specifically prohibit* the use of eager undeclared calls! in all
> situations!

> basically, we must construct any situation which *requires* dynamic uncompile eager memory reads during a collective
> in order to reach a contradiction which implies the one-sided-send no-preemptive-guards policy should somehow be
> subverted!

> so there's no such thing as a correct, safe, guarded, unbounded domain and unbounded codomain, eager, data dependent
> operation, since trying to make this kind of call safe would paradoxically introduce a kind of unbounded halt that we
> don't want to deal with. on the other hand, implementing data dependent operations as a routing operand and a routed
> data/parameter range is easy and simple and what grouped_gemm et al do anyways

> meshlib is not for bad eager code on good collectives; why are you trying to use bad eager code on good collecives?
> you have pytorch at home.

## The ranking

There are four forms, from lowest to highest. Each pair below is ranked by the rubric's procedure
([rubric-actionable.md](rubric-actionable.md), step 2), with the code that decides it.

1. **An arbitrary function with no explicit indexing boundary, made safe by a guard.** Its volume is known only as it
   runs, so it reuses pages inside the call and waits for room.
2. **The grouped-GEMM form, called eagerly.** A routing operand and a routed range. The definition is compiled when the
   call arrives.
3. **The grouped-GEMM form, compiled.** The same operand and range, with pages and schedule fixed at the declaration.
4. **meshlib's compiled operations.** The same form as 3.

The pairs:
- **1 ≺ 2, by C6 and M4 (a failure's reach, containment) and C4 (asynchrony).** Form 1's guard is a wait on space.
  It can halt a call while every member is healthy (below: "A guard can halt with no party at fault"). Form 2 adds no
  wait. The code: the credit receipts of d4b8837c, 8b102b68 and e13eab3f, deleted by 22228bb3; the waits on `sent`
  inside a call, deleted by bdde7903.
- **2 ≺ 3, by D2 and C5 (when choices are made, when time may decide).** Form 2 chooses at the call: an undeclared
  group's definition is compiled at its first issue (`signature`, rdma/nccl-mesh.c). Form 3 chooses at the declaration:
  `ncclMeshCommPlan`, whose slots are taken there (3db5d137); `Plan` and `Execute` in rdma/mesh_mpi.py;
  `declarations.Function` in torch-mesh.
- **3 ∼ 4.** Neither ranks higher, because they are one form. The operator, 2026-10-28: "we can't use cuda ops on mps
  because mps isn't cuda but if we could we would be fond of the best cuda ops." The best compiled grouped GEMM on CUDA
  ranks with them.

The gap between 1 and 2 is a gap in kind: form 1 cannot be made correct. The gap between 2 and 3 is when the choice is
made.

## Terms

- **An eager call** names its buffers, counts and routes when it runs. Nothing earlier knows its pages.
- **A declared call** names its extents before it runs ([extents-and-liveness.md](extents-and-liveness.md), a).
- **A guard** is a wait inside a call on something other than the call's own input data: room, a receipt, a credit, a
  `sent` count.
- **A data wait** is a consumer waiting for its input to land. Every implementation has one. It is not a guard
  ([pages-and-functions.md](pages-and-functions.md): waiting on data instead of on messages about data).
- **A routing operand** is a tensor of counts, offsets or indices. It says where each part of a routed range goes.

## Assumptions

- **A1.** A call's pages are fixed when it is declared (rubrics.md D8; extents-and-liveness.md, a).
- **A2.** A page is written again only after it is released (pages-and-functions.md, Freeing 6 and 7).
- **A3.** A call waits only on data ([deliverables.md](deliverables.md) I3, I4 and I18).
- **A4.** Every wait has a deadline. Past it the program ends (the operator, 2026-10-23, in c6241813; the code is
  `mesh_expired` in rdma/mesh.h and `mesh_library_expired` in rdma/nccl-mesh.c).

## What a sender can know

Release needs proof. A sender can learn that its page was read in only two ways:
- a landing that comes after the remote read, along a chain of data back to the sender;
- its own NIC's completion of the DMA read, the `sent` prefix, which is a local event.

There are no others. A process learns of a remote event only through a chain of messages from that event
([Lamport 1978][lamport]; Chandy and Misra 1986). Freeing 6 names exactly these two proofs. A receipt is a message sent
only to make such a chain, and I3 excludes it.

## Where a guard would come from

A call needs a guard only if it must write a page again inside the call, and no chain of data comes back inside the
call. Then `sent` is the only proof, and the call must wait on it.

A declared call never has to do this. Its pages hold its whole volume (A1), so it writes no page twice. Reuse happens
across calls. There, the proof is a later landing, or `sent` read between calls without waiting: a choice made between
calls on known buffer depth (C5). For dense layers, where each step reads every node's last step, two copies of each
page are enough. NFE k+1 writes other pages while NFE k's are still being read (Freeing 7).

An eager call has no declaration, so its volume is not known until it runs. If its pages cannot hold all of it, it must
reuse pages inside the call. Then it needs a guard.

## A guard can halt with no party at fault

A guard waits for room. Collectives are cycles: in a ring or an all-to-all, each member both sends and receives.
Bounded buffers with blocking writes on a cycle can deadlock while every member is healthy. In Kahn's process networks
(Kahn 1974), which have exactly such buffers, this is called an artificial deadlock (Parks 1995). Whether a dataflow graph whose rates depend on data runs in bounded memory is
undecidable in general (Buck 1993). So no compiler can promise that a guarded eager collective finishes.

Under A4 the deadlock becomes an expiry, and the program ends although no party failed. That is the lowest pole of C6
and of M4. The operator, 2026-10-28, of such designs: "They always imply infinite loops, unbounded waits, unoptimizable
contradictions, and more importantly aren't needed for any kind of linear algebraic program we'll actually be running".

## The routing-operand form needs no guard

Suppose a call's input shape is declared. Then every output that depends on data is bounded by it: a routed operation
moves at most its input times its fan-out. So the operation is a routing operand plus a routed range, over pages sized
to that bound. The counts travel as data, inside pages already placed. A mask picks values within one page
([data-model.md](data-model.md), section 6). The call writes no page twice, so it needs no guard.

This is what grouped GEMM does:
- torch's `_grouped_mm` takes its group offsets as a device tensor;
- CUTLASS's grouped GEMM reads its problem sizes from device memory;
- both are bounded by their operands' declared sizes;
- GShard and the Switch Transformer route tokens to experts within a fixed capacity (Lepikhin et al. 2021; Fedus et
  al. 2022).

The only input left unbounded is one whose shape is not declared. That is a new declaration, compiled between calls.
The data model says so: "If we change who does which work, we compile new lists and steps before the next run"
(data-model.md, section 6). The theory has the same split. Rates that are fixed give a static schedule with bounded
buffers, decided before the run (Lee and Messerschmitt 1987). Rates that change only between iterations keep that
property (Bhattacharya and Bhattacharyya 2001; the supersteps of Valiant 1990). Rates that change inside an iteration
lose it (Buck 1993).

## Every case we tried

Each row is a case that seems to need reads chosen while a collective runs.

| Case | Why it seems to need a guard | The declared form |
|---|---|---|
| Counts that depend on data (MoE routing, sparse output, a search frontier) | The volume is unknown until the call runs | Worst-case pages with valid lengths carried in the data; or the counts in one call and the data in the next |
| Reads at addresses that depend on data (lookup by index) | The address is data | An index push, then a row push, into pages sized by the index count, which is declared |
| A one-way stream larger than its pages (a pipeline, a broadcast to a sink) | Pages reused inside the call, nothing coming back | Cut into calls. The host reads `sent` between calls, and G calls issued ahead hide that choice once G ≥ 1 + latency × link rate ÷ bytes per call |
| A producer faster than its link | Backpressure has to go somewhere | It is the issue rate, chosen between calls on known depth (C5) |
| A slow or dead peer | Someone must wait | A landing wait is a data wait; past its deadline the program ends (A4) |
| Decode loops and speculation | Control flow depends on data | The host chooses between NFEs; inside a call, the accepted length is a mask |
| A writer outside the plan | It writes whenever it likes | Excluded: every page is a declared producer's (region-only memory) |

Each answer costs memory, a call boundary, or numerics when a fixed capacity overflows. None costs a guard. A
contradiction would need all four of these at once: volume that depends on data, memory below the worst case, exact
numerics, and no added call boundary. A guard does not meet those four either: it turns overflow into a wait, and the
wait can deadlock.

## The programs we run

Every program the hourly runs, and every model the engine serves, has declared shapes:
- **Dense layers, the chain and the LSTM:** each step reads every node's last step, so two copies of each page.
- **tp and cp:** row and column splits of declared tensors, with partials landing in declared ranges.
- **ep and the MoE models:** tokens routed to experts, in the routing-operand form with capacity from the declaration.
  The balancer moves shares between evaluations, not inside one.
- **Attention:** KV pages are placed by the declared length. Speculation's accepted length is a mask.
- **bandwidth and fabric-dnn:** declared buffers.

None of them needs an eager call.

## What follows

- **Collectives are called inside a declaration.** NCCL's calls are the declaration's vocabulary there, the way CUDA
  graph capture keeps them. They are not entry points that run eagerly.
- **Eager entry points are removed, not guarded.** At 22228bb3 they are:
  - the undeclared group's definition compiled at its first issue (`signature` and `same_call`, rdma/nccl-mesh.c);
  - host groups compiled at their issue (`launch` and `launch_host`, rdma/nccl-mesh.c);
  - the MPS kernels `functional` registers for `_c10d_functional`, which run each op as it is called
    (rdma/torch-mesh/torch_mesh/_stream.mm). `declarations.Function` reads the same ops from its traced graph and
    declares them;
  - ProcessGroupMesh's eager methods (rdma/torch-mesh/torch_mesh/backend.py);
  - mesh_mpi's eager methods, `Allreduce` through `Barrier` (rdma/mesh_mpi.py), whose declared form is `Plan` and
    `Execute`.
- **A request for a new call** (`ncclSend` and `ncclRecv`, an undeclared AlltoAllV) is met as a declared call.
- **Page reuse is answered by page assignment.** Receipts, credits, waits on `sent`, spare pools and reuse queries are
  never the answer. The answer is page assignment over the declared function (Freeing 7).
- **A refusal check at run time is no remedy either** (rubrics.md C6: rejection filters). The entry point goes.

## References

- J. T. Buck. Scheduling Dynamic Dataflow Graphs with Bounded Memory Using the Token Flow Model. PhD thesis, UC
  Berkeley, 1993.
- B. Bhattacharya and S. S. Bhattacharyya. Parameterized dataflow modeling for DSP systems. IEEE Transactions on
  Signal Processing 49(10), 2001.
- K. M. Chandy and J. Misra. How processes learn. Distributed Computing 1(1), 1986.
- W. Fedus, B. Zoph and N. Shazeer. Switch Transformers. JMLR 23, 2022.
- G. Kahn. The semantics of a simple language for parallel programming. IFIP 1974.
- L. Lamport. Time, clocks, and the ordering of events in a distributed system. CACM 21(7), 1978 ([lamport]).
- E. A. Lee and D. G. Messerschmitt. Synchronous data flow. Proceedings of the IEEE 75(9), 1987.
- D. Lepikhin et al. GShard. ICLR 2021.
- T. M. Parks. Bounded Scheduling of Process Networks. PhD thesis, UC Berkeley, 1995.
- L. G. Valiant. A bridging model for parallel computation. CACM 33(8), 1990.

[lamport]: https://amturing.acm.org/p558-lamport.pdf
