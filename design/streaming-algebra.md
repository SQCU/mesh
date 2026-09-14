# Streaming algebra

The [asynchronous collective contract](async-collectives.md) is the complete scope.

The caller supplies numerical functions and mesh/tensor placement. Configuration
binds their input and output regions to actual registered storage before invocation.
The same configured region may be an output for one function, a transport source,
and an input for another. These are uses of the value, not additional copies.

For contraction, independent reduction-domain contributions have distinct storage.
Each can compute, publish and move while other contributions are unavailable.
Their configured reduction produces the final value at the corresponding output
coordinates. A nonlinear operation consumes that value once its actual sum exists;
it does not wait for unrelated output coordinates.

A producer's publication does not await transport completion, a consumer, or the
rest of its enclosing operation. Each consumer runs the partial computation whose
input regions are available. Configuration supplies distinct storage for live value
instances so reuse does not create a dependency between independent operations.

The public calling surface is documented in [streaming kernel calls](indexed-library.md).
Pallas supplies the calling structure and indexed forwarding mechanism; its GPU
backend details are not a separate implementation requirement here.

## End-to-end source derivation

Take configured output regions I and J with disjoint storage, and contraction
partitions Q0 and Q1. Let P[I,Q] = sum(k in Q, A[I,k] B[k]). The following
sequence follows the existing implementations, without a runtime evaluator.

1. `Program.kernel_call` and `_ExpressionRegions.parts` in
   `python/mesh/kernels.py` bind the supplied grid and operand regions. Each
   contraction contribution has its own output and actual operand dependencies.
   `_ReductionPlan` combines contributions through an addition tree. The tree
   is numerical dependency, not an enclosing-operation completion condition.
2. `mesh_issue_index` in `rdma/mesh-dataflow.c` examines those input maps and
   claims the output storage. `submit_ready` in `rdma/mesh-algebra.m` submits
   the prebound submission function. Synchronous CPU arithmetic is dispatched to
   the concurrent worker queue; asynchronous Metal/CoreML submission is direct.
   No CPU contraction runs on the serial presence handler.
3. Generated CPU section loops invoke `publish_cpu` after their stores. It calls
   `mesh_publish_partial`, which sets presence and notifies readers immediately.
   Later sections can still be executing. Native contractions publish their
   configured region through `complete_part`; Metal and Core ML reach that path
   through their device completion callbacks. The numerical call does not wait
   for a send, an acknowledgement or another output region.
4. `mesh_notify` puts the changed rows on canonical compute and send work lists.
   `link_ready` in `rdma/mesh-flow.c` selects only published transfer sources.
   An absent I therefore does not occupy a ready-list position ahead of J.
   `link_send_ready` posts the index description and queues its payload through
   `link_send_announced`, without awaiting a software acknowledgement. Lack of
   device capacity returns control to bridge progress; it does not stop the
   producer or the numerical presence handler.
5. `link_post` constructs the receive SGE from the configured destination page.
   Successful receive completion calls `mesh_receive_complete`, publishing those
   pages and notifying their consumers. `mesh_events` follows the affected rows'
   configured reader edges and attempts the corresponding numerical functions.
   It does not require an unrelated region to arrive or ask a caller to launch
   a partial consumer.
6. If Q1 is absent, P[I,Q0] can still compute, publish and move. Its addition
   parent waits only if that parent's other term is absent. Other tree branches
   and output regions remain issuable. A nonlinear consumer of the completed
   sum reads that sum, never an unfinished contribution substituted for it.

Thus absence of I does not appear in J's issue predicate unless the configured
arithmetic actually reads I. Transport completion releases its own source read;
it is not a prerequisite to publishing the source. `mesh_claimable` prevents
reuse of storage still being produced or read. Distinct live values have distinct
backing, so that reuse dependency does not connect independent values.

This uses the canonical changed-row publication list, operand-associated firing,
configured reduction tree and two-sided SEND/RECV implementation. Sources are
[publication work lists](algorithm-sources.md#publication-work-lists),
[presence-driven execution](algorithm-sources.md#presence-driven-execution),
[contraction accumulation](algorithm-sources.md#contraction-accumulation), and
[asynchronous index publication](algorithm-sources.md#async-index-push-contract).
The argument concerns dependency and control flow; it makes no latency or
throughput claim and introduces no acceptance or rejection criteria.

## Historical staged execution (superseded)

On September 14, 2026, `examples/streaming-chain.py` at commit `2bc27de`
executed on the MacBook (producer 0) and Mac mini (consumer 1) through the
canonical Thunderbolt SEND/RECV bridge. Both used shared-region ABI 26,
65,536 pages, four pages per block and one payload queue.

The supplied FP32 tensors were X[32,32], W_up[32,32] and W_down[32,16],
with tile rows, K and columns all 16. Input values were generated in X, W_up,
W_down order with NumPy `default_rng(73).standard_normal(shape)`, cast to FP32
and divided by 8. The actual calls were:

    producer 0: X → linear(X, W_up) → swish → copy to consumer 1
    consumer 1: received H → linear(H, W_down) → copy to producer 0

Both contractions bind two K contributions per output region. Each hidden
region becomes a transfer source after its numerical completion and an operand
of the consumer's corresponding contraction contribution after arrival. The
caller's terminal loop only reads returned results; it issues no intermediate
numerical work. This staged execution did not split either contraction across participants.
It does not demonstrate tensor-parallel work sharing.

The terminal received region (1,0) followed by (0,0). Their actual values are
[recorded here](../measurements/streaming-chain-2026-09-14.txt). The first entries
were -0.02199353650212288 and -0.012772580608725548 respectively. The returned
order is the observed execution result, not a required schedule. It does not
by itself establish the timing of every internal contribution; the source
bindings establish their partial-input dependencies.

The producer exited after consuming both regions. The consumer was then closed
with SIGTERM. No reference evaluator, timing comparison, pass/fail threshold or
runtime acceptance criterion was introduced.

## What the execution record does not demonstrate

The returned region order is not a demonstration of latency, low overhead,
absence of synchronization, or a bounded nonblocking operation. It records an
executed numerical chain. Those stronger properties were previously overstated.

The source exposes these distinct costs and waits:

| Boundary | Current mechanism | Consequence |
|---|---|---|
| Output claim | `mesh_issue_index` checks actual inputs and output reader ownership; `mesh_reset` clears presence and assigned read planes before setting producing | Work proportional to covered rows and assigned reader planes precedes dispatch. An unavailable operand/storage claim returns without spinning for it. |
| Publication | `mesh_publish_partial` / `mesh_publish` update atomic planes; `mesh_notify` pushes each affected row onto compute/send lists and calls `sendto(MSG_DONTWAIT)` | Publication does not wait for delivery, but incurs atomic contention, list work and a socket call. CAS retries have no stated per-call time bound. |
| Consumer discovery | `mesh_events` runs on a serial dispatch queue and traverses affected reader edges | Ready work can incur notification and queueing delay; asynchronous submission alone gives no bound on that delay. |
| Numerical issue | `submit_ready` enters a dispatch group and invokes the prebound submit function; only synchronous CPU work uses a worker queue | CPU arithmetic executes away from the presence handler. Dispatch internals and worker scheduling are not shown to have zero contention or bounded latency. |
| Transfer | Registered SEND/RECV with index descriptions and finite queue capacity | Capacity shortages defer posting. Reusing a receive destination depends on its actual readers. These are distinct from waiting for an unrelated tensor to finish. |
| Terminal application | The example supplies every input block before entering a busy-poll output loop; formatting/printing occurs in that loop | Observed output order omits intermediate timing and includes application observation delay. |
| Setup and destruction | Compiler process waits, synchronous registration/removal on the presence queue, and dispatch-group/semaphore waits during destruction | These boundaries explicitly wait; a claim about numerical publication must not be generalized to the complete program lifecycle. |

For this particular call configuration a 16×16 FP32 tile contains 1,024 useful
bytes. Its transferable backing block has four 16,384-byte pages. `mesh_realize`
rounds the useful transfer length up to a 4,096-byte frame; the payload therefore
uses 4,096 bytes before index traffic and device framing. These are source-derived
sizes, not measured timings and not a placement policy.

Publication-to-issue latency, enqueue overhead, transport delay, numerical
execution duration and contention cost have not been separated by the recorded
run. No numerical latency or overhead claim follows from it. The source shows
which mathematical dependencies are local to a region; it does not by itself
establish a wait-free guarantee for the complete implementation.

## Tensor-parallel operation use

The staged example was replaced with the MLX authors' column-sharded followed
by row-sharded linear decomposition. Caller configuration chooses J_0=[0,256)
and J_1=[256,512), with replicated X[1024,256]. Both participants own only their
weight slices in registered storage and compute

    U_p = X W_up[:,J_p]
    H_p = swish(U_p)
    D_p = H_p W_down[J_p,:]
    Y = D_0 + D_1

The root owns Y; the only network edge is peer D_1 to root receive storage.
Neither participant's U, H or D arithmetic requires an input from the other
participant. Each down-projection K contribution reads its own H region. Each
root addition reads one local D region and the matching received region, with
no dependency on the rest of either D tensor. The caller binds this entire chain
before publishing X. It launches no intermediate computation.

This chain ran over the two Macs' existing RDMA bridge with 128-sized tiles:
FP32/MPS at `68319f4`, FP16/BNNS and FP16-input/FP32-weight BNNS at `5c42fe2`.
All three runs returned sixteen 128×128 reduced regions. Actual output excerpts
and full-stdout digests are [recorded here](../measurements/tensor-parallel-chain-2026-09-14.txt).
The peers were closed with SIGTERM after the root consumed its results.

These runs establish use of the distributed decomposition and integrated
numerical/transport paths. The source above identifies region-local issue and
publication dependencies. The output record does not measure simultaneous device
execution, publication-to-consumption latency, or speedup. The implementation
still has a serial presence handler, socket notifications, and finite hardware
queues; removing one worker hop and unassigned reader-plane resets does not
establish zero scheduling overhead.
