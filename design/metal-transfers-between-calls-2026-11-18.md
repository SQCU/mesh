# Metal transfers in a repeated call graph, 18 November 2026

## A: What the compiler does

I gave the local compiler the same FX graph, `A(x) → B → A`, with two different placements on a two-rank topology at compiler-emit `48623c9`.

- With a transfer inside `A`, the frontend produced calls `[0, 1, 0]` and one transfer declaration for `A`. Metal stopped in `Bindings.endpoints` and asked the caller for `endpoint_operands`. The planner already groups calls 0 and 2 under `A`. The Metal code keys one pair of call indices by the identity of a line in the body that both calls share. [Endpoint lookup](../../../compiler-emit/compiler_emit/bindings.py) · [Recorder](../../../compiler-emit/compiler_emit/metal_install.py)
- With all of `A` on rank 0 and all of `B` on rank 1, the planner retained the edges `0 → 1` and `1 → 2`. On each rank its transfer counts for the two functions were `[0, 0]`. The frontend records shared storage across calls and collects sends inside each function. It omits the sends that these two edges require. [FX program and channel records](../../../compiler-emit/compiler_emit/source.py)

## B: What the compiler should do

For a transfer inside `A`, Metal reads the planner’s call group and binds the same compiled body’s transfer at calls 0 and 2 to their respective page ranges. For an edge between functions, placement reads the producing call’s output and the consuming call’s input. When they occupy different ranks, placement supplies their ranks and tensor views to the existing transport solver before it chooses routes and page schedules. Here the first `A` sends its result to `B`, then `B` sends its result back for the second `A`. The recorder uses those ranges and the bridge binds them to registered pages. The caller supplies the FX graph once. [Planner call groups and storage slots](../rdma/mesh-collective.h) · [Placement and bridge responsibilities, R6–R7](plan-to-bridge-change-request.md#r6-placement-determines-transfer-endpoints)

The live check is to run both placements through the public compiler on two ranks, call each program with two distinct successive inputs, and compare each result with eager FX. This review exercised compilation locally.

## Implementation and observations, 18 November

At compiler-emit `e8e491d`, Metal reads the planner’s call group. For a graph that calls `A(x)` twice with the same placement, both calls use one compiled body. The recorder resolves that body’s send or receive against call 0’s plan pages at the first occurrence and call 2’s pages at the second. The caller supplies the graph and placement. [Endpoint declarations](../../../compiler-emit/compiler_emit/endpoints.py) · [Recording](../../../compiler-emit/compiler_emit/metal_install.py)

The plan also names calls that run wholly on a peer. With compiler-emit `bd5af0f` and metal-microbench `0acbe75d`, I placed all three calls on rank 0 and compiled and invoked each rank’s recording on one Mac. Rank 0 returned the eager FX result. Rank 1 recorded three empty entries and returned `None` for the remote tensor. Both ranks repeated the result after reopening their recordings. The native linker retains each task library when it links its object. Its previous second scan dereferenced an empty entry and crashed on rank 1. [Native linker](../../../metal-microbench/metal_lower.mm)

For `A(left) → B(right) → A(left)`, placement at compiler-emit `332a7dc` produced transfer counts `[0, 1, 1]` for the three calls on both rank projections. It retained the two numerical callees and distinguished the two placements of `A`: the first reads a caller input and the last reads a result from the other rank. Mesh `6ab15afb` also made routing use the program’s participating ranks when it reads link capacities. Before that change the solver could choose a link through a third rank, then reject its own route. The public call now takes the topology as the bridge reports it. [Placement](../../../compiler-emit/compiler_emit/source.py) · [Transport solver](../rdma/mesh-collective.c)
