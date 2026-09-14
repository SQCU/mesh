# MLX and JACCL: the small collective boundary

Reviewed September 14, 2026 against upstream `main` and mesh `a821c7a` before
the changes described below. The [async contract](../async-collectives.md)
governs the implementation. The operator's direction is to implement our own
partial-streaming path, guided by JACCL, and carefully avoid adding overhead.

## What callers need

MLX's [distributed operations](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/ops.cpp)
construct array operations with shape, dtype, inputs, a communication group and
a stream. `all_sum`, `all_max` and `all_min` share one reduction primitive;
`all_gather` concatenates rank contributions on axis zero; `sum_scatter` reduces
and distributes equal axis-zero slices. `send` names a destination; `recv` names
the source and output geometry; `recv_like` obtains that geometry from an array.
These operations do not require a model graph, placement optimizer or kernel selector.

Collective semantics follow the actual consumer. Broadcast, scatter, gather,
all-gather, all-to-all and reductions retain their distinct meanings. A reduction
binds arithmetic over available contributions; it does not first gather an entire
tensor. The operator explicitly rejected universalizing any one verb. The
[implemented calling surface](../collective-verbs.md) follows that direction.

MLX's [ShardedToAllLinear](https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/distributed.py)
computes `x @ weight.T`, then `all_sum`, then an optional bias. Its complementary
AllToShardedLinear performs a local projection with a column partition of the
weights. This is the same division used by Shoeybi et al.,
[Megatron-LM](https://arxiv.org/abs/1909.08053): local numerical functions compose
with communication. An FFN implementation belongs in the caller or numerical
library; the transport need not understand an FFN.

JAX's [Pallas collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
also composes an existing matmul with communication and uses distinct receive
slices. Our additional requirement is availability at independently usable
regions, on both sides of a transfer. Pallas syntax does not justify another
runtime grid after setup has already bound those regions.

## Necessary structures in our implementation

| Information | Existing owner | Why it exists |
| --- | --- | --- |
| Participant and endpoint configuration | `mesh_ctx`, realized transfer bindings | Caller-specified peers and placement. |
| Registered backing and tensor geometry | `mesh_extent`, `mesh_view`, canonical page table | The numerical operand is the actual send/receive storage. Distinct live values require distinct backing. |
| Numerical operation with input/output ranges | `mesh_row_function`, `mesh_row_map` | Bind the supplied function once; issue the arithmetic whose inputs exist. |
| Region presence and outstanding uses | Canonical presence and reader metadata | Partial availability and actual CPU/GPU/NIC lifetimes are different facts. |
| Posted transfers and completion queues | Bridge connection, WR and CQ records | Use the substrate's existing asynchronous posting and completion facts. |

The last two rows do not mandate the current per-page reader records. MLX's
[Metal evaluation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/eval.cpp)
retains input backing through its existing command-buffer completion callback;
[array::Data](https://github.com/ml-explore/mlx/blob/main/mlx/array.h) owns the
allocator buffer. This supports using actual operation completion for lifetime
management. Our [lifetime review](page-lifetimes-2026-09-14.md) identifies which
reader records still duplicate information and why early publication is not
evidence that a producer has finished every memory access.

## JACCL as implementation prior art

The standalone library lives inside MLX at
[`mlx/distributed/jaccl/lib`](https://github.com/ml-explore/mlx/tree/main/mlx/distributed/jaccl/lib).
Its [Group interface](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/group.h)
takes input/output pointers, byte counts, dtype where needed, and peers.
Its [RDMA layer](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/rdma.h)
owns connections and registered buffers, constructs SGEs, posts SEND/RECV, and
polls CQs. Mesh already uses those same verbs; another adapter would duplicate
an existing boundary.

The current [MeshImpl](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/mesh_impl.h)
preposts a small receive pipeline, encodes buffer/peer identity in WR IDs, and
refills from completions. Its public send/receive calls copy through internal
registered buffers and poll until the operation completes. Its reduction loop
also performs arithmetic. These source facts delimit reuse: retain direct
posting, indexed destinations and immediate draining in our own implementation;
use canonical operands and separately supplied numerical functions.

| Work | JACCL source | Consequence for mesh |
| --- | --- | --- |
| Connection and memory registration | Group setup | Keep outside numerical invocation. |
| Post, poll, identify completed storage, refill | RDMA/collective loop | Keep directly on dedicated TX/RX threads. |
| Operand copies through staging | MeshImpl send/receive | Avoid: canonical registered pages are the operands. |
| Poll until whole operation finishes | MeshImpl public operations | Do not put this on producer/consumer call paths. |
| Numerical reduction | MeshImpl reduction loop | Bind existing addition independently from transport progress. |
| Partial readiness | No public partial-result interface | Mesh needs region presence, not duplicate function or grid representations. |
| Ready-order transfer announcements | Additional mesh protocol | Remaining overhead to remove with coherent receive bindings; it is not justified merely by existing in mesh. |

This is a source work inventory, not measured latency parity. Current mesh still
has announcement-driven receive posting and destination reuse checks. Those
prevent a claim that it already meets the complete nonblocking contract or adds
no overhead relative to JACCL.

## Applied changes and source demonstration

- Removed native occurrence counts, affine occurrence strides, per-occurrence
  range pointers, reader offsets, selected-index arrays and callback index arguments.
  The Pallas grid is resolved once by setup.
- Removed `mesh_watch`. The bound function owns its dependency edges, submission
  callback and index membership directly. Realization receives pointers to those
  functions instead of copying their descriptors.
- Reduced `mesh_row_map` to first row, count and reader pointer. Reader storage
  uses one allocation per mutable range; constant-only ranges allocate none and
  no longer send consumption notifications. Presence reads reuse canonical bit
  operations. Mutable-generation bookkeeping remains explicit unfinished work.
- Implemented the movement verbs as indexed transfer relations and the reduction
  verbs as numerical trees over contributions, with explicit sum/max/min forms.
- Made `nn.ffn` return its local numerical contribution. The existing streaming
  example selects `reduce` to its root consumer, removing unused result replication.

The demonstration now reads: local projection and activation, local down
projection, `reduce`, elementwise consumer, subsequent `linear`. Setup
expands its blocks into the same bound functions used by native execution.
A completed region publishes canonical presence; the region index reaches its
dependent functions directly; issue reads only their configured input ranges;
CPU or device completion publishes their output ranges. Neither the second
runtime grid nor a watch-to-function relay participates. The example uses
separately allocated simultaneous instances. This demonstrates source use of
the simplification, not completion of the remaining receive/reuse refactor.
The final consumer runs only on the root. The example therefore binds a root
reduction directly; no completed result is sent back to an unused participant.

## Build and execution record

Mesh `f053612` built its native libraries and Python package on the M5 Max and
M4 Pro. Both machines also built the existing engine library and `forward_graph`
targets. Only existing deprecation/compiler warnings were emitted.

The existing streaming chain ran on both participants with Metal, using Apple's
MPS contraction path, the existing FP32 files in `/tmp/mesh-tp-consumer`,
`--root 0 --peer 1 --split 256 --tile-rows 128 --tile-k 128 --tile-columns 256`,
and two configured instances. The root returned 64 distinct output regions,
32 for each instance, and exited successfully. The peer's example lifetime ended
with ordinary SIGTERM and exited successfully. Both bridge processes remained up.
No reference evaluator or timing threshold was added. This run exercises the
root reduction chain; it does not claim execution coverage of every verb.

Across all changed maintained source files, this commit adds 191 lines and
removes 230: 39 net source lines removed while extending the collective surface.
Documentation and repository instructions add 229 lines and remove 35 separately;
no source comments were migrated into that count. This is not a claim that the
earlier whole-implementation halving has been achieved.
