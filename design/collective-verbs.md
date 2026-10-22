# Collective communication relations

This table describes each collective's value movement and combination.

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

The operator, 2026-09-15: "note that peer>1 prohibits the use of type inference of
broadcasts or alls rather than defined scatters and defined gathers". Peer count,
tensor type and operand arity select no collective. Scatter's destination list and
gather's destination are explicit at every world size. Broadcast, all-gather,
all-scatter, all-to-all and all-reduce are the verbs of those names; additional peers
do not make a movement an "all" operation. A rank outside a transfer's two endpoints
does not bind that transfer or allocate a received copy.

Repeated delivery of the same immutable part to the same destination and queue
shares the receive operand during setup. This does not merge input positions in
a numerical call: `reduce([x, x], using: combine)` still combines x with x.
Changing a destination or queue declares a different transfer. No collective is
inferred from this value sharing.

References: MPI Forum's [collective communication](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm)
and MLX's [distributed operations](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/ops.cpp).
