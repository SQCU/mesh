# Metal transfers in a repeated call graph, 18 November 2026

## A: What the compiler does

I gave the local compiler the same FX graph, `A(x) → B → A`, with two different placements on a two-rank topology at compiler-emit `48623c9`.

- With a transfer inside `A`, the frontend produced calls `[0, 1, 0]` and one transfer declaration for `A`. Metal stopped in `Bindings.endpoints` and asked the caller for `endpoint_operands`. The planner already groups calls 0 and 2 under `A`. The Metal code keys one pair of call indices by the identity of a line in the body that both calls share. [Endpoint lookup](../../../compiler-emit/compiler_emit/bindings.py) · [Recorder](../../../compiler-emit/compiler_emit/metal_install.py)
- With all of `A` on rank 0 and all of `B` on rank 1, the planner retained the edges `0 → 1` and `1 → 2`. On each rank its transfer counts for the two functions were `[0, 0]`. The frontend records shared storage across calls and collects sends inside each function. It omits the sends that these two edges require. [FX program and channel records](../../../compiler-emit/compiler_emit/source.py)

## B: What the compiler should do

For a transfer inside `A`, Metal reads the planner’s call group and binds the same compiled body’s transfer at calls 0 and 2 to their respective page ranges. For an edge between functions, placement reads the producing call’s output and the consuming call’s input. When they occupy different ranks, placement supplies their ranks and tensor views to the existing transport solver before it chooses routes and page schedules. Here the first `A` sends its result to `B`, then `B` sends its result back for the second `A`. The recorder uses those ranges and the bridge binds them to registered pages. The caller supplies the FX graph once. [Planner call groups and storage slots](../rdma/mesh-collective.h) · [Placement and bridge responsibilities, R6–R7](plan-to-bridge-change-request.md#r6-placement-determines-transfer-endpoints)

The live check is to run both placements through the public compiler on two ranks, call each program with two distinct successive inputs, and compare each result with eager FX. This review exercised compilation locally.

## Implementation and observations, 18 November

Placement reads each tensor’s producing rank and its readers from FX. For `A(left) → B(right) → A(left)` it supplies the two crossings before routing. The router reads the participating ranks from the plan, and Metal reads each callee’s occurrences from the planner’s call group. The caller supplies the graph and placement once. A rank with no local body records an empty entry. An invocation returns `None` for a tensor that lives only on a peer. [Placement](../../../compiler-emit/compiler_emit/source.py) · [Router](../rdma/mesh-collective.c) · [Endpoint declarations](../../../compiler-emit/compiler_emit/endpoints.py)

Three live failures identified where the recorder had discarded that information.

| What failed | What the compiler now does |
|---|---|
| `A` wrote the send pages. The next call took them as input, had zero local writers and never signaled the bridge. | Native `PUBLISH` takes the value and reads its selected byte ranges. The recorder orders it after the preceding GPU write. This publishes the whole transfer at GPU completion. A writer inside a kernel can publish completed cuts earlier. Writer steps that count contributions also bind the value’s arrival counter. |
| Two calls of the same compiled `A` shared an execution-index table. Metal initialized it once per recording object, overwrote the first ordinal and published generations 2, 4, 6 and 8. | Metal sizes the table from the planner’s call group and initializes its two ordinals once at their shared address. Four calls publish generations 1 through 4. |
| The recorder examined one invocation. The next invocation overwrote its two outgoing pages while RDMA was still reading them. | The recorder examines a full cycle of the send buffer. It orders a write that reuses a page after the overlapping DMA read completes. It uses the local completion event where that event already covers a local dependency. |

The failing runs distinguish the execution-index error from page reuse. [Execution indices](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-placements/repeated-1795048247.log) · [Actual page contents](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-placements/repeated-1795048402.log) · [Transport lowering](../../../metal-microbench/metal_transport.mm) · [Recorder](../../../metal-microbench/metal_program.m)

Studio 2 and Mini ran Metal `b3937308` through the public compiler. They ran both placements of `A → B → A`, reopened the between-call recording, and ran `(B(A(x)), A(y))` fresh and reopened. That last case uses one compiled `A` twice with distinct arguments. Each of the five runs exercised two consecutive invocations. All 57,344 float32 output values matched eager FX exactly. The bridge reported zero clients on each host after the runs. [Evaluator](../../../compiler-emit/artifacts/joint-selection-review-20261118/evaluate-calls.py) · [Five live logs](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-placements/b3937308)

In a separate fan-out graph, two independent calls on the same remote rank read one tensor produced by `A`. Placement requested a transfer for each reader. Sharing one receive across those calls remains an optimization.
