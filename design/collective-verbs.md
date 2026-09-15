# Collective communication relations

The [user's requirements](collective-goals.md) require each collective's actual
semantics. This table describes value movement and combination; it prescribes no
Python signatures, scheduler, invocation protocol, or mandatory decomposition.

| Operation | Value relation |
| --- | --- |
| Send / receive | One configured source section goes to its destination. |
| Broadcast | A root's sections go to the configured recipients. |
| Scatter | A root's distinct sections go to their respective destinations. |
| Gather | Distinct source sections arrive at the root with their coordinates. |
| All-gather | Each participant receives the source sections. |
| All-to-all / all-scatter | Each source pushes its sections to their configured destinations. |
| Reduce | Matching contributions combine at the root using the selected numerical operation. |
| Reduce-scatter | Matching contributions combine at their respective destination owners. |
| All-reduce | The reduced values are available at each participating consumer. |

Sum, maximum and minimum select numerical reduction functions. Movement does not
imply reduction. Reduction does not imply gathering the entire tensor first.
The same partial publication and zero-copy transport support each relation.

Peer count, tensor type and operand arity do not select a collective. Scatter's
explicit destination list and gather's explicit destination remain authoritative
at every world size. Broadcast, all-gather, all-scatter, all-to-all and all-reduce
are selected by calling those verbs; additional peers do not promote a movement
into an "all" operation. A rank outside a transfer's two endpoints does not bind
that transfer or allocate a received copy.

References: MPI Forum's [collective communication](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm)
and MLX's [distributed operations](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/ops.cpp).
