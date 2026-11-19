# Scheduling repeated sharded calls, 19 November 2026

## The input program

`A(x,w)` multiplies an 8 × 16 matrix by a shared 16 × 32 weight. `B` takes ReLU. All shapes and calls are fixed. The three inputs to A vary independently.

```python
p = A(x, w)
q = B(p)
r = A(y, w)
s = B(r)
t = A(z, w)
return q + s, t, q[:, 1:29:2]
```

A owns 24 columns on one processor and eight on the other. B needs 16 on each. Both B results feed the sum. The strided output also reads the first B. The comparison first supplies a direct link and then compiles a topology on which every transfer between the numerical processors takes two hops. In a second program, a call updates the shared weight after the first multiply. The next multiply reads through the original argument, and the third reads through the updater's result. The caller observes the weight after return. The existing sharding client runs both programs with `--call-bounds`. [Executable inputs and comparisons](../../../../compiler-emit/artifacts/joint-sharding-20261119/evaluate.py)

## A: What the earlier implementation did

For the seven calls above, the partitioner made twenty physical calls and supplied twenty distinct function numbers to the native scheduler. Both the bound-one and bound-two programs returned the right tensors on three inputs. The bound rows constrained twenty different functions across iterations. They omitted the constraint between successive invocations of A in the same iteration. Two other substitutions lost information at the same boundary. An explicit fusion of the first and third updates in `add_(1), mul_(2), add_(1)` moved them across the multiply and changed 7 into 8 for input 2. A right-hand worker that needed a view of the caller's weight received an unbound operand even though its input table already held the backing tensor. [Prior scheduler trace](../../../../compiler-emit/artifacts/joint-sharding-20261119/source-call-bounds.md) · [Source and substitution](../../../../compiler-emit/compiler_emit/source.py)

The communication graph also chose the first rank as owner of a transfer-free function. Its rank-work vertex took its duration from the transfer geometry. Changing the duration or placement of purely numerical work could therefore leave a zero on the wrong rank. The queue target could emit two independent calls in one Inductor compilation while omitting the selected source issue constraint. A delayed first worker and a quick second worker then completed in reverse order even with a limit of one. [Native plan](../../rdma/mesh-collective.c) · [Torch target](../../../../compiler-emit/compiler_emit/torch_program.py)

## B: Which function supplies each operand

| Responsibility in the requests | Where the information comes from and where it goes |
|---|---|
| Identify occurrences, ordering and legal overlap | FX supplies the callee of each occurrence in the original graph. Bernstein read/write conflicts and ordered effects constrain substitution. A10 advances by the permitted number of calls in each callee's own repeating sequence. The result names both the successor occurrence and the number of blocks crossed. [AOT A1/A10](../../design/aot-compilation-change-request.md) |
| Place the work and form collective demands | Sharding propagation places numerical work. The reader's extents select the reshard. The collective lowering supplies page ranges, directed hops and their endpoints. A streaming reader depends on the ranges it can use independently. A joining reader depends on all ranges it needs. A shared weight view can serve all three multiplies. [A2–A4](../../design/aot-compilation-change-request.md) · [R1/R3/R4/R6](../../design/plan-to-bridge-change-request.md) |
| Constrain submission and storage | Project a source completion onto the physical calls that implement it and the later submissions it releases. Preserve the block distance. Apply the bound at each participating rank. Storage and queue sizing consume that relation together with data dependencies, resource order and copy lifetimes. The island sequencer orders delivery. Each rank starts eligible work from the compiled dependencies. [A5/A7/A10](../../design/aot-compilation-change-request.md) · [Disjoint pages C3.3–C3.7](../../design/disjoint-pages-change-request.md) |
| Choose among legal orders | The weighted graph includes numerical tasks and routed hops. R5 minimizes maximum completion time and then total completion time under precedence and shared-resource capacity. Its heap order gives each resource a subsequence. The chosen sharding, routing, fusion and allocation can change these inputs. A function table index supplies identity. The source issue bound supplies legality. Neither quantity estimates duration. [A3/A6](../../design/aot-compilation-change-request.md) · [Plan to bridge R5](../../design/plan-to-bridge-change-request.md) · [Composition 3–7](../../design/compilation-composition-change-request.md) |
| Emit and retain the selection | The numerical target consumes rank-local order and dependencies. The bridge consumes range descriptors and the selected hop order. The archive retains source incidence, delays and target bindings for an independent load. The synchronization signature relates actual issue and wait sites to their causes. [A8/A9](../../design/aot-compilation-change-request.md) · [Synchronization 1–4](../../design/synchronization-signature-change-request.md) |

## The changed program

The compiled family rows now read `A, B, A, B, A, add, slice`. The three A occurrences use the same weight and keep distinct completion events. With limit one, the first A releases the second A in the same block. With limit two, the first A releases the third. With limit four, the third A releases the first A two blocks later. B can have its own limit. On both numerical ranks and the relay, a fresh archive load reproduced the complete source and projected physical constraints. In each placement the two compiled CPU workers matched eager Torch on three inputs. The mutation variant also reproduced all three matrices and the caller's final weight. [Recorded run](../../../../compiler-emit/artifacts/joint-sharding-20261119/source-bounds-execution.txt)

The source partitioner accounts for the original storage conflicts before grouping calls. The planner retains numerical site and duration through the per-rank work graph. The target reads the source-bound edges from the analysis. On one worker, the strict limit added an Inductor dependency between two independent numerical kernels. On two workers at one Mesh rank, the strict limit added a queue between the calls. Raising the limit let the quicker second call finish first. The reference execution used local CPU workers. The relay observation exercised compilation and archive loading; this review supplies no physical-link timing.

The mutable weight also ran from offsets 3, 7 and 0 in its input storage after capture at offset 3. Across these calls, the maximum error was zero for each output and for the caller's weight. Numerical rank durations likewise respond to a change in constraints: two independent operations priced at 3 and 7 on rank 1 gave 10 on a shared resource and 7 on separate resources. These are costs in the units the caller supplied. [Offsets](../../../../compiler-emit/artifacts/joint-sharding-20261119/source-bounds-execution.txt) · [Rank costs and source limits](source-issue-and-rank-time.md)
