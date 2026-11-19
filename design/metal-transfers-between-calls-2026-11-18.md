# Metal transfers in a repeated call graph, 18 November 2026

## A: What the compiler does

I gave the local compiler the same FX graph, `A(x) → B → A`, with two different placements on a two-rank topology at compiler-emit `48623c9`.

- With a transfer inside `A`, the frontend produced calls `[0, 1, 0]` and one transfer declaration for `A`. Metal stopped in `Bindings.endpoints` and asked the caller for `endpoint_operands`. The planner already groups calls 0 and 2 under `A`. The Metal code keys one pair of call indices by the identity of a line in the body that both calls share. [Endpoint lookup](../../../compiler-emit/compiler_emit/bindings.py) · [Recorder](../../../compiler-emit/compiler_emit/metal_install.py)
- With all of `A` on rank 0 and all of `B` on rank 1, the planner retained the edges `0 → 1` and `1 → 2`. On each rank its transfer counts for the two functions were `[0, 0]`. The frontend records shared storage across calls and collects sends inside each function. It omits the sends that these two edges require. [FX program and channel records](../../../compiler-emit/compiler_emit/source.py)

## B: What the compiler should do

For a transfer inside `A`, Metal reads the planner’s call group and binds the same compiled body’s transfer at calls 0 and 2 to their respective page ranges. For an edge between functions, placement reads the producing call’s output and the consuming call’s input. When they occupy different ranks, placement supplies their ranks and tensor views to the existing transport solver before it chooses routes and page schedules. Here the first `A` sends its result to `B`, then `B` sends its result back for the second `A`. The recorder uses those ranges and the bridge binds them to registered pages. The caller supplies the FX graph once. [Planner call groups and storage slots](../rdma/mesh-collective.h) · [Placement and bridge responsibilities, R6–R7](plan-to-bridge-change-request.md#r6-placement-determines-transfer-endpoints)

The live check is to run both placements through the public compiler on two ranks, call each program with two distinct successive inputs, and compare each result with eager FX. This review exercised compilation locally.
