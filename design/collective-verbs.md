# Collective verbs on tensor regions

These are setup calls. They bind ordinary `Program.copy` edges and numerical
functions; publication and execution remain asynchronous at each usable region.
No verb waits for a whole tensor, another rank's invocation, or a remote consumer.
The [remaining receive-path limitations](lit/mlx-jaccl-structure.md) still apply
to the shared native implementation.

`value` is a mesh Tensor with global shape and block coordinates. `owners` maps
those coordinates to ranks. Local output tensors retain the same global geometry
and contain only blocks belonging to this participant. This expresses placement
without concatenating or copying a second dense operand. Every participant binds
the same ordered configuration before invocation.

| Call | Dataflow |
| --- | --- |
| `send(program, source.on(p), destination.on(q))` | Bind one transfer into an existing destination Ref/Tensor. |
| `recv(program, source.on(p), destination.on(q))` | The same paired-Ref declaration, expressed from the receiving context; use either name once per edge. |
| `recv_like(program, source.on(p), q)` | Reuse `Program.replicate` to allocate and bind matching tensor storage. |
| `broadcast(program, value, root=p, peers=ranks)` | Every source block at p goes to each recipient. |
| `scatter(program, value, root=p, owners=placement)` | Each block at p goes to its designated owner. |
| `gather(program, value, root=p, owners=placement)` | Each source owner's block goes to p, retaining its coordinates. |
| `all_gather(program, value, peers=ranks, owners=placement)` | Each source owner's block goes to every recipient. |
| `all_to_all(program, value, owners=sources, destinations=targets)` | Each source block goes to its configured destination; no arithmetic. |
| `reduce(program, value, root=p, peers=ranks, op=kernels.add)` | Reduce corresponding contributions into p. |
| `reduce_scatter(program, value, peers=ranks, owners=placement, op=kernels.add)` | Reduce each coordinate into its designated owner. |
| `all_reduce(program, value, peers=ranks, owners=placement, op=kernels.add)` | Reduce at the supplied owners and distribute each completed result to all recipients. |
| `all_sum`, `sum_scatter`, `all_max`, `all_min` | Sum, sum-scatter, maximum and minimum forms of the above reductions. |

Movement and reduction share storage and transfer primitives, not semantics.
Gather preserves distinct source values; reduction combines matching values.
Reduction binds a binary numerical tree over independent received contributions.
An available pair can reduce while another pair or output coordinate is absent.
The final value becomes available only after its actual required terms exist.

`nn.ffn` returns local numerical contributions. The caller selects a verb for the
next operation. In `examples/streaming-chain.py`, only the root has the downstream
weight and consumer, so `reduce` sends contributions there and performs the sum
there. A caller retaining reduced shards uses `reduce_scatter`; a caller needing
replicated sums uses `all_reduce`. None of these is a universal collective path.

Sources: MPI Forum's [collective semantics](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm),
MLX's [operations](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/ops.cpp),
and the [mechanism citations](algorithm-sources.md#collective-movement).
