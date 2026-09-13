# Streaming numerical composition

The unit of composition is an indexed numerical region with separately owned
output storage. Setup allocates its canonical registered backing, resolves each
function's input references, and binds peer destinations. Invocation scans those
functions against canonical page stamps. No whole-tensor readiness object exists.

Install with `python -m pip install .`, then `from mesh import Program, BlockSpec`.
The distribution builds and includes the native libraries and their public headers.
It requires NumPy; MLX workloads retain their previous dependencies in the `mlx`
default uv dependency group (`uv sync` retains the workload environment). A live canonical mesh bridge supplies the shared-memory region.

## Tensor algebra

`Program.tensor(shape, block_shape)` allocates a two-dimensional tensor as indexed,
independently reusable blocks, including ragged boundary blocks. Backing storage
is transferable by default. Block shape is realized configuration, not a runtime
choice or a semantic tensor boundary. Each block is an ordinary `Ref`.

The following is setup code, not an evaluator or a numerical invocation:

```python
import numpy as np
from mesh import Program, BlockSpec

program = Program(backend='cpu')
x = program.tensor((1024, 512), (128, 128))
remote = program.tensor(x.shape, x.block_shape)
w = program.tensor((512, 256), (128, 64))
p = x * 0.5 + 0.125
for peer in (0, 1):
    program.copy(p.on(peer), remote.on(1 - peer))
a = np.tanh(p + remote)
z = a @ w
statistics = z.sum(axis=1)
```

This lowers elementwise arithmetic per `(i, q)` and contraction per `(i, j, q)`.
`z.contributions[i, j][q]` is an ordinary reference that can feed another function
or a transfer before the other `q` contributions exist. A completed reduction
block can feed another contraction: composition has no root-operation special
case or generated C entry point. `Tensor.T` reindexes blocks and transposes their
views without moving payloads.

Contraction consumes each available contracted partition separately. Its final
sum necessarily depends on all terms in that sum. Nonlinear functions of the
final sum retain that mathematical dependency. Independent output blocks and
exposed contributions acquire no such dependency.

## Application numerical functions at both ends

`Program.call(kernel, grid=..., inputs=..., outputs=...)` enumerates `BlockSpec`
index maps during setup. At invocation the kernel receives prebound NumPy arrays:
read-only inputs followed by writable outputs. These arrays alias canonical pages.
The function is synchronous arithmetic for that region; returning publishes it.
A grid producer has no inputs. A consumer may also be the next producer:

```python
def produce(out):
    out.fill(1)

def transform(local, received, out):
    np.add(local, received, out=out)
    np.tanh(out, out=out)

identity = lambda i, q: (i, q)
program.call(produce, grid=x.grid, outputs=[BlockSpec(x, identity)])
activated = program.tensor(x.shape, x.block_shape)
program.call(transform, grid=x.grid,
    inputs=[BlockSpec(p, identity), BlockSpec(remote, identity)],
    outputs=[BlockSpec(activated, identity)])
result = activated @ w
```

These snippets illustrate alternative compositions. Configure constants, exports,
and all functions before `realize()`. Repeated `scan()` calls invoke ready regions;
the bridge independently progresses transfers. Kernels must finish their region's
writes before returning and must not wait for another region. NumPy operations
using `out=` permit direct writes, but arbitrary application kernels can allocate
or copy: the library cannot prove the internals of a supplied function.

`call_native(prepare, ...)` supplies the asynchronous form. Setup calls
`prepare(coordinate, inputs, outputs)` once per grid coordinate. It returns a
`Submission` and an opaque binding pointer whose storage the application retains
for the program lifetime. Submission has C signature
`void(void *binding, mesh_completion complete, void *context)` and returns after
issuing work. The backend calls `complete(context, error)` exactly once when its
physical reads/writes have completed. That completion publishes through mesh;
there is no application publish call to forget. Preparation owns compilation,
allocation, specialization, and device bindings. Submission must not wait for
inputs or allocate operand storage. The native C API is `mesh_algebra_function`.

A single submitted region cannot expose its unfinished interior. Express that
interior as independently writable blocks and region functions during setup.
This applies to every backend. Multiple outputs on one function are deliberately
joint completion; independently completed outputs use separate region functions.

## Source proof of permitted overlap

1. `Tensor.elementwise`, `Tensor.__matmul__`, `Tensor.sum`, and `Program.call`
   enumerate independent numerical functions during setup. `mesh_algebra_function`
   converts only the declared input views to dependency maps; built-in `bind_part`
   narrows them to the actual numerical output part. There is no tensor-wide join.
2. `mesh_algebra_scan` issues ready functions. CPU arithmetic finishes one part
   and calls `complete_part`; native asynchronous submissions return and call it
   on physical completion. Neither completion waits for later numerical parts.
3. `complete_part` calls `mesh_complete`. Its `mesh_publish` sets PRESENT and the
   configured send's ROW_HOT bits. `mesh-flow.c` runs independently and `link_post`
   uses the registered source pages directly. Later producer computation does not
   occur in this call chain and is not a prerequisite for sending.
4. Receive completion publishes the configured destination rows. A consumer's
   input maps name its own required blocks, so it may issue while other blocks
   remain in computation or transit. Its completion repeats the same publication
   path, allowing arbitrarily composed chains.
5. Last-reader bits keep each output alive through both numerical reads and NIC
   reads. Separate configured outputs have separate storage. There is no staging
   tensor, payload packing, or output alias used to emulate overlap.

This proves that the implementation permits producer/link/consumer overlap at
configured region boundaries. It does not establish a measured overlap duration
or speedup. CPU arithmetic is synchronous per region on the scanning thread;
the bridge and peer run concurrently. Asynchronous backend submissions additionally
allow the local scan to continue while computation executes. FIFO order on each
configured transport queue remains a real link constraint; independent traffic
must be assigned appropriate queues/order at setup, not inferred from arrivals.

## Lifetimes and granularity

Generic function outputs may be disjoint compact subregions of an allocated block,
aligned to its publication quantum. `BlockSpec.region_map(ref, *coordinate)`
selects such regions during setup. Inputs may be sliced, transposed, or broadcast.
Output regions own complete publication quanta so no unwritten payload becomes usable. `copy` transfers matching
whole blocks; indexed gather/scatter is composition of copies or numerical grid
functions whose index maps choose the source and destination blocks.

`export(ref)` binds a canonical reader during setup. `Result.ready` reads those
stamps; `consume()` releases that reader. Exported arrays are borrowed read-only
views, valid only until consumption. `write(ref)` is an external whole-block
producer: it attempts issue without waiting, yields the prebound writable array,
and publishes on successful exit. An exception does not publish incomplete data.
Normal streaming producers should be grid functions, so issue and publication
are automatic. Constants are initialized before realization.

Keep one attached program per process, and retain borrowed references only during
its lifetime. Close after asynchronous submissions physically finish. Callback
errors are reported without publishing incomplete output; they are diagnostics,
not another readiness protocol. The callback and native function remain retained
for the configured program lifetime.

The [source and literature review](region-streaming-review.md) records remaining
whole-K dependencies and the missing arrival-selected masked reduction.

## Independent launches into one full-sized output

Built-in numerical bindings accept the same compact, publication-aligned output
regions as custom functions. An allocation therefore need not be split into
separate tensor objects to obtain independent launches and publications.
For example, with FP32 storage and a 64 KiB publication quantum, each `(128,128)`
section below owns one quantum:

```python
local = program.tensor((512, 128))
received = program.tensor((512, 128))
output = program.tensor((512, 128))
for i in range(4):
    left = local[0, 0].slice(i * 128, 0, 128, 128)
    right = received[0, 0].slice(i * 128, 0, 128, 128)
    destination = output[0, 0].slice(i * 128, 0, 128, 128)
    program.bind('add', left, destination, right)
```

Each addition depends only on its two input sections. All four write into the
same output allocation, with disjoint write ownership. Configured transfers of
that allocation observe each completed section independently. Transmitted input
sections can therefore be consumed before the rest of the allocation arrives,
and those consumers immediately become streaming producers themselves.

`Program.contract(left_partitions, right_partitions, destination)` and
`Program.reduce_sum(contributions, destination)` likewise accept a destination
section. A contraction computes its partial contributions into separate registered
storage, then combines the contributions for each output section. A completed
section may be sent or consumed while another section's contributions are absent.
The remaining receive dependency for a function is its declared operand region,
not the containing allocation. Split the contracted dimension into the actual
partial operands at setup; different partials may finish independently.

Several launches must not perform uncoordinated read-modify-write on the same
output elements. Independent contributions have independent backing, and the
combining function owns the destination section. Its completion dependency is
legitimate: it needs those contributions. No whole-allocation completion is
required. This uses existing region functions and reader masks rather than an
additional arrival-selected accumulation scheduler.

Source addressing is consistent across backends: the CPU store uses
`z.offset + first + i`; Metal uses the output view's offset and strides; MPS uses
the sliced output matrix's offset; Core ML's supplied output pointer uses
`z.offset + first`. Publication starts at the corresponding output page offset.
All return through `complete_part`, which publishes and makes configured sends
eligible before unrelated launches finish.

## Pallas-style calls

Use operand-independent block specifications and a curried kernel call:

```python
import numpy as np
from mesh import BlockSpec, ShapeDtypeStruct


def matmul_kernel(x_ref, w_ref, y_ref):
    np.matmul(x_ref, w_ref, out=y_ref)


matmul = program.kernel_call(
    matmul_kernel,
    grid=(64,),
    in_specs=(BlockSpec((64, 256), lambda i: (i, 0)),
              BlockSpec((256, 256), lambda i: (0, 0))),
    out_specs=BlockSpec((64, 256), lambda i: (i, 0)),
    out_shape=ShapeDtypeStruct((4096, 256), np.float32),
)
y = matmul(x, w)
z = matmul(y, v)
```

The call configures storage and functions; it does not execute them eagerly.
`realize()` finishes configuration and `scan()` issues ready regions. The second
call depends on corresponding regions of `y`, not all of `y`. Numerical functions
receive borrowed arrays and return normally. No ctypes callback, manual completion,
or explicit publish is required. Inputs are read-only and outputs writable.
Use NumPy's `out=` operations to write the provided storage directly.

A BlockSpec index map returns block indices, as in Pallas. Shapes and index maps
can be reused with different operands. `spec.bind(tensor)` also works with the
lower-level `Program.call`. Existing `BlockSpec(tensor, index_map, region_map)`
clients remain supported. Multiple outputs use matching tuples of shape/dtype
objects and output specs; the configured call returns a tuple of tensors.

Current differences are explicit: tensors and specs are two-dimensional; boundary
blocks are clipped, not automatically padded; outputs must own publication-aligned
regions; and kernels are host NumPy functions rather than JAX-traced device code.
Asynchronous device implementations continue to use `call_native`. `kernel_call`
does not add a second execution engine or reinterpret an asynchronous function's
return as physical completion.

The existing [overlap demonstration](../examples/streaming-overlap.py) uses this
API for both producer and consumer. Its archived measurements record the source
revision used at the time; changing the call spelling does not retroactively
constitute a new measurement.

The migrated overlap client was run on both RDMA peers at revision `df50252`,
with 100 measured completions per mode, ten warmups, and two inputs in flight.
Both modes passed terminal numerical checks (maximum absolute error
`5.622414e-6`). Whole mode observed zero consumers before full reception;
streamed mode observed 6,935 such launches out of 7,040. This confirms that the
simplified call site retains regional consumption. It is a functional follow-up,
not a replacement for the four-pair performance record.
