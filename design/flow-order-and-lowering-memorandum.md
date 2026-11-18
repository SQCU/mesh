# Flow order before target lowering

Mesh determines which inputs each piece of a function needs while the exported function and its tensor partitions are still IR. A producer that can deliver a partition separately and a consumer that can compute from it establish a dependency for that partition. A consumer that needs several partitions establishes a join on those partitions. A coflow groups transfers with a common completion objective. Group membership, physical contiguity, and required execution order answer different questions. [Plan to bridge, R4](plan-to-bridge-change-request.md) · [Compilation composition, §§5–6](compilation-composition-change-request.md)

For example, two column shards of a first projection can each compute their own pointwise activation from the shared input and their own weights. Either shard may finish first. An operation that reads both outputs joins them where it first needs both. The function's operand accesses and numerical semantics determine those edges, including any required order of accumulation. The same construction applies to a consumer of one independently arriving page. [AOT compilation, A2/A4/A6](aot-compilation-change-request.md)

The joint optimizer reads the annotated graph, the flow groups and their required edges together with placement, link capacity, storage and the supplied weighting function. It can choose among legal orders, overlap communication with arithmetic, and select fusion across a send or receive. It returns the selected schedule and retains its input so another round can make a different choice. Each separation, join or fusion keeps the correspondence to the original function operands. [Compilation composition, §§5–7](compilation-composition-change-request.md)

Final lowering consumes that selection. It can emit fixed commands with fixed operands, or fixed commands that read a varying list of eligible tiles and its count. In the latter case, compilation determines the legal membership and dependencies, and arrivals determine which eligible tiles have work. A change in measured latency can inform the weighting of a later round. [Operator clarification, 17 November 2026](#operator-clarification)

## Operator clarification

“anything with a 'free choice' can be optimized jointly by other concerns, so this tells us where in a data flow this must be and how that data flow must be structured”

## Operands and results

Placement evaluates `transfer(value, destination, readers)` on the original FX operations. The supplied function returns the collective identity and the declared consumption granularity. A page-wise reader of a byte range depends on the arriving pieces that overlap that range. A reader of the whole operand joins every required piece. Two independent writers in one coflow can contribute to the same completion objective. [Chowdhury and Stoica, *Coflow*](https://conferences.sigcomm.org/hotnets/2012/papers/hotnets12-final51.pdf)

The source dependence relation supplies legality. A provisional processor timeline can size buffers for that schedule. Using its chosen order to infer mandatory edges would make the next solver reproduce an earlier choice. The joint solver takes the source relation, communication cuts, task weights and resource assignments, and returns both its precedence relation and its chosen order. Assembly takes each processor's subsequence of that order. A subsequent fusion or expansion maps the affected occurrences to the original operands. [Plan to bridge, R3–R5](plan-to-bridge-change-request.md) · [Unger et al., *Unity*](https://www.usenix.org/conference/osdi22/presentation/unger)
