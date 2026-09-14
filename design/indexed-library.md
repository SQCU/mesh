# Streaming kernel calls

The [asynchronous collective contract](async-collectives.md) is the complete scope.

Install with `python -m pip install .`. The public numerical interface is
`Program.kernel_call`; composed expressions and realized backend kernels use the same
configuration. `BlockSpec` describes block shape and an index map independently
of the operand. `ShapeDtypeStruct` declares output shape and dtype.

```python
import numpy as np
from mesh import Program, BlockSpec, ShapeDtypeStruct, kernels

program = Program(backend='cpu')
x = program.tensor((4096, 256))
w = program.tensor((256, 256))
matmul = program.kernel_call(
    kernels.matmul,
    grid=(64,),
    in_specs=(BlockSpec((64, 256), lambda i: (i, 0)),
              BlockSpec((256, 256), lambda i: (0, 0))),
    out_specs=BlockSpec((64, 256), lambda i: (i, 0)),
    out_shape=ShapeDtypeStruct((4096, 256), np.float32),
)
y = matmul(x, w)
```

The call allocates output storage and binds independent region functions during
setup. `realize()` finalizes the program; the canonical presence owner issues
functions whose declared input regions are present. The completion path publishes that region to
consumers and configured sends. This implementation detail does not require
waiting for an enclosing operation to complete: library-owned in-operation
publication must expose usable partial outputs asynchronously while computation
continues. The numerical caller needs no separate publication scheduler,
completion callback, or tensor-wide join.

The current Python entry point accepts expressions built with `kernels.arguments`,
indexed operations and `kernels.expression`. Setup binds these to CPU, Metal/MPS,
or configured Core ML contraction implementations. The present expression frontend
is implementation, not an additional requirement to build a general compiler or
prohibit other supplied numerical implementations. The contract is asynchronous
region production and consumption through the supplied configuration.

`program.export(ref)` retains only the pages touched by `ref`, including sliced,
strided and transposed views. The resulting observation owns its own reader
members on those pages; `ready` observes their availability and `consume` releases
those memberships. `ref.present` also observes only the view's touched pages.
Neither operation requires the remainder of the allocation to arrive. These are
calling-context observations; they do not schedule numerical work.

Tensor storage is always canonical shared backing. Transpose, slices, and block
index maps describe that storage. A region must fit the configured backing block;
The current binder assigns complete publication quanta to output regions.
Independent outputs need separate presence ownership; memory binding supplies it. The current host
API supports two dimensions and clips boundary regions instead of padding them.
Pallas supplies the calling structure; JAX tracing and full compatibility are outside scope.

`program.copy(source.on(peer), destination.on(peer), queue=...)` configures a
transfer. Both participants declare corresponding destinations during setup.
The independent bridge sends each published section directly from registered
pages. Distinct outputs have distinct storage, and last-reader bits retain it
until numerical and NIC reads finish. Queue ordering is a configured transport
property; numerical kernels contain no synchronization protocol.

Constants are initialized with `program.constant` during setup. External input
production uses `program.write(ref)`; this attempts issue without waiting and
publishes on successful exit. An exception leaves the region unpublished and
releases its write claim so that production can retry. Writer bindings are owned
by native mesh for the program lifetime. `program.export(ref)` binds an output reader;
`ready` observes its stamps and `consume()` releases its reference. Returned
arrays remain valid until consumption. Close the program after pending native
work completes.

Native function, reader, route and transfer trace reconstruction is removed.
The private lowering boundary retains the function count needed to attach indexed
operands. Numerical callers receive tensor references and output observations;
page maps and row-function layouts remain inside canonical mesh.

## A composed producer and consumer

[The executable chain](../examples/streaming-chain.py) uses this interface for

    U = X W_up
    H = U / (1 + exp(-U))
    Y = H W_down

The caller supplies two participants and a split of the hidden dimension.
Participant p owns J_p and computes U_p = X W_up[:, J_p], H_p = swish(U_p),
and D_p = H_p W_down[J_p, :]. Both participants receive replicated X and can
start local arithmetic independently. The peer publishes D regions to the root;
the block owner computes Y = D_root + D_peer with the existing add kernel.
`reduce_scatter` places each sum at its configured owner; `all_gather` distributes
those completed blocks to both participants. Only the
local weight shard occupies canonical operand storage. Each reduced region feeds
swish and a further contraction Z = swish(Y) W_consumer at the root.

All calls and transfer edges are bound before `realize()`. Missing regions
constrain their own K contributions and matching output sums. Swish consumes
completed U values, never unfinished contraction sums. No intermediate caller
launch loop, hidden gather, or stage handoff schedules numerical work.

Both participants use the same files and placement arguments; `--region` selects
local mesh storage. For an input (rows, inner), weights (inner, hidden) and
(hidden, columns), a caller-selected hidden split of 256 is expressed as:

```sh
python examples/streaming-chain.py input.npy up.npy down.npy consumer.npy \
  --root 0 --peer 1 --split 256 --output-split 2048 --tile-rows 128 --tile-k 128 --tile-columns 128
```

The consumer weight has shape (columns, consumer_columns). The root prints
consumed regions as they become available. The peer remains
attached after publishing input, until terminated. File supply and terminal
observation are application I/O outside numerical functions. This uses the
[MLX tensor-parallel decomposition](algorithm-sources.md#nnffn)
with explicitly owned, region-wise reduction followed by all-gather.

`collective.reduce_scatter(program, value, peers=peers, owners=owners)` takes a
local contribution tensor and an explicit mapping from block coordinates to
owner ranks. It returns a Tensor view containing this participant's owned sums.
`collective.all_gather` takes that view and the same placement and returns all
blocks. `collective.all_reduce` composes the two. `collective.send` is
`Program.copy`. These are setup compositions, with no runtime launcher or poller.
`nn.ffn` takes the same `peers` and `owners` configuration and applies these
collectives to its down-projection contributions.
