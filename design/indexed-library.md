# Streaming kernel calls

Install with `python -m pip install .`. The public numerical interface is
`Program.kernel_call`; backend kernels and ordinary NumPy kernels use the same
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
setup. `realize()` finalizes the program; `scan()` issues functions whose declared
input regions are present. Completion publishes that region to consumers and
configured sends. No manual publish, completion callback, tensor-wide join, or
application scheduler is part of a numerical kernel.

`kernels.matmul`, `add`, `multiply`, `swish`, `tanh`, `exp`, `row_sum`, `rsqrt`,
and `affine(alpha, beta)` preserve the configured native backend functions.
CPU, Metal/MPS, and configured Core ML contractions share publication ownership.
For application arithmetic, pass an ordinary function taking borrowed input
arrays followed by writable output arrays, e.g. `np.matmul(x, w, out=y)`.
Input arrays are read-only; returning means that region's physical writes have
finished. Asynchronous backend implementations belong to the library's backend
binding path; callers do not manually submit or publish.

Tensor storage is always canonical shared backing. Transpose, slices, and block
index maps describe that storage. A region must fit the configured backing block;
output regions own full publication quanta so a stamp never exposes unwritten
payload. Setup may allocate separate backing for column tiles. The current host
API supports two dimensions and clips boundary regions instead of padding them.
This is not a JAX tracing or full Pallas compatibility claim.

`program.copy(source.on(peer), destination.on(peer), queue=...)` configures a
transfer. Both participants declare corresponding destinations during setup.
The independent bridge sends each published section directly from registered
pages. Distinct outputs have distinct storage, and last-reader bits retain it
until numerical and NIC reads finish. Queue ordering is a configured transport
property; numerical kernels contain no synchronization protocol.

Constants are initialized with `program.constant` during setup. External input
production uses `program.write(ref)`; this attempts issue without waiting and
publishes on successful exit. `program.export(ref)` binds an output reader;
`ready` observes its stamps and `consume()` releases its reference. Returned
arrays remain valid until consumption. Close the program after pending native
work completes. Arbitrary supplied NumPy code may itself allocate or copy;
the library guarantees its own backing and dependencies, not unknown kernel code.

## Streaming FFN

`mesh.nn.ffn` is a DNN composition of kernel calls outside the mesh runtime.
It accepts input K partitions, up-weight blocks indexed by hidden section and
K partition, and down-weight blocks indexed by hidden section:

```python
from mesh.nn import ffn

y = ffn(program, inputs, up_weights, down_weights, tile_rows=128)
```

For row section r and hidden section h:

    u[r,h] = sum_q x[r,q] @ up[h,q]
    a[r,h] = swish(u[r,h])
    p[r,h] = a[r,h] @ down[h]
    y[r]   = sum_h p[r,h]

Every product contribution has separate output storage. First-layer addition
finishes only the hidden section needed by swish. That section's second-linear
product can run while another hidden section is unfinished. Final additions
combine corresponding row sections. Swish is never applied separately to terms
of an unfinished sum. An optional setup-time `exchange` maps activated tensors
to configured receive tensors; it contains no runtime numerical control.

The numerical functions perform arithmetic and return. They contain no readiness
polls, semaphores, host waits, or completion guards. Mesh still enforces the actual
operand and storage-lifetime dependencies; absence is not treated as a numerical
zero and concurrent producers do not race on one accumulator.

Run `examples/streaming-algebra.py RANK --backend cpu` (or `metal`) on both peers
for the two-K-partition, two-hidden-section composition. Stop the consumer with
SIGTERM after the producer reports its checked result. The sustained performance
client remains `examples/streaming-overlap.py`; its archived measurements are in
[the overlap report](streaming-overlap-2026-09-13.md).

See [caller migration status](caller-migration.md) for the remaining Xonotic
application ports. They are part of the requested all-callers migration.
