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
   the bound numerical function to the existing concurrent dispatch queue and
   returns. No CPU contraction runs on the serial presence handler.
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

## Used operation chain

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
numerical work. These are the functions traced in the source derivation above.

The terminal received region (1,0) followed by (0,0). Their actual values are
[recorded here](../measurements/streaming-chain-2026-09-14.txt). The first entries
were -0.02199353650212288 and -0.012772580608725548 respectively. The returned
order is the observed execution result, not a required schedule. It does not
by itself establish the timing of every internal contribution; the source
bindings establish their partial-input dependencies.

The producer exited after consuming both regions. The consumer was then closed
with SIGTERM. No reference evaluator, timing comparison, pass/fail threshold or
runtime acceptance criterion was introduced.
