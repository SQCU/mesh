# Storage lifetimes for streamed tensor partials

Reviewed September 14, 2026 against mesh main `d67bc85`, its pending direct-binding
refactor, metal-microbench `mesh_matrix.swift`, and the primary sources below.
The [async contract](../async-collectives.md) governs scope. This review adds no
collective, scheduler, cost model, public lifetime API, or acceptance program.

The evidence supports explicit value-instance storage and sparse lifetime dependencies
on existing operation completions. It does not establish a need for per-page consumer
stamps, a shared completion counter, or a general garbage collector. My earlier claim
that refcount-style background GC was necessarily the answer was premature.

## What the upstreams actually establish

| Source | Mechanism in the source | Consequence for mesh |
| --- | --- | --- |
| JAX [collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html), [implementation](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/gpu/collective_matmul_mgpu.py) | Allocates `(num_devices - 1, m_shard, k)` receive scratch; each incoming shard has its own slice. Reuses the existing local matmul with a forwarding callback. The example explicitly explains that shrinking this storage would require backpressure. It also contains GPU copy waits and semaphore synchronization. | Adopt distinct indexed destinations and numerical reuse. This is direct evidence for spending memory to remove reuse dependencies. It is not evidence that literal Pallas never synchronizes or that its whole kernel should become our default collective implementation. |
| JAX [async design note](https://docs.jax.dev/en/latest/pallas/design/async_note.html) | Shows a source whose arithmetic consumer finishes before an asynchronous send. Lifetime must extend through the send's completion. Explicit stateful Refs make this dependence visible without disguising it as an extra value use. The note also discusses unwanted copies from alias handling. | Record the actual transport use of the actual backing. A numerical completion alone cannot retire a source still read by the device. This design note supplies reasoning, not a stable runtime interface to transplant. |
| JAX [GPU pipelining](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html) | `max_concurrent_steps` adds buffering. `delay_release` extends a buffer's lifetime when asynchronous matrix instructions still read it. | The useful property is a lifetime tied to the actual execution structure. Arbitrarily retaining storage for a fixed number of wall-clock cycles is not the same guarantee. |
| MLX [Metal evaluation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/eval.cpp), [array ownership](https://github.com/ml-explore/mlx/blob/main/mlx/array.h) | Evaluation captures shared ownership of input backing in existing command-buffer completion handlers. `array::Data` invokes its allocator deleter when its owners disappear. | Reuse backend completion information and bind ownership to backing. Numerical kernels do not need to publish an additional per-page consumption protocol. MLX does use reference counting; this is not evidence that reference counting has no synchronization cost. |
| MLX [Metal allocator](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp) | `free` ordinarily returns a buffer to its cache; later allocation can reuse it. Both use an allocator mutex. | The spelling `free` does not imply a GPU wait, zeroing, or returning pages to the OS. Our question is where reclamation sits in the dependency path, not the name of a function. Copying this allocator wholesale would import runtime locking we do not need for a realized storage plan. |
| OpenXLA [buffer assignment](https://github.com/openxla/xla/blob/main/xla/service/buffer_assignment.cc) | Assigns backing using value live ranges, alias information and instruction ordering; reuse considers interference. | Static use information can eliminate runtime reference bookkeeping where the execution order proves non-overlap. A topological enumeration alone does not serialize independently running hardware. We need the principle, not XLA's compiler and allocation policy. |
| Apple [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt), rdma-core [posting contract](https://man7.org/linux/man-pages/man3/ibv_post_send.3.html) | Thunderbolt provides nonblocking SEND/RECV and completion queues. The posted WR names its actual memory. Reuse follows completion of device access. Apple distinguishes send completion from successful remote receipt. | Keep the physical span already stored with each posted operation. Local source reclamation does not require a remote-consumer acknowledgement. A receive completion establishes delivery into destination backing; it does not finish the destination's later numerical readers. |
| Linux [RCU documentation](https://www.kernel.org/doc/html/latest/RCU/whatisRCU.html) | `call_rcu` defers reclamation and returns control to the updater; `synchronize_rcu` waits. Reclamation follows the relevant read-side lifetimes. | Deferring reclamation is appropriate. A CPU grace-period library is not automatically a lifetime implementation for outstanding GPU/NIC work: those device uses must still be represented. This is an inference from the two contracts, not a proposed RCU dependency. |

MLX's [current JACCL adapter](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/jaccl.cpp)
passes whole-array pointers and sizes to collective operations on a CPU communication
stream. It does not expose tensor-partial publication in that interface. Reusing MLX's
ownership pattern does not establish that wrapping its collective call would meet our
partial-streaming contract. Its [distributed documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
also explicitly discusses CPU/GPU synchronization for communication.

## Actual mesh data and execution flow

The following are separate facts about one value, not separate competing value models:

- Its indexed contents are available to a particular numerical use.
- An operation has finished accessing its backing.
- No remaining operation or retained external view can access that backing, so the
  backing is eligible for another value.

`mesh_tensor_create` allocates registered backing and creates its CPU view at setup.
Generated CPU/Metal expressions address the canonical page table. BLAS pointers, BNNS
descriptors, MPS matrices, Core ML arrays and engine `MatrixView`s instead bind fixed
extent storage. `Ref.array` also constructs a persistent NumPy view of a fixed address.
Updating the page table alone would leave those bindings pointing to old storage.

| Path | Completion already available | Lifetime implication |
| --- | --- | --- |
| Generated CPU function | `submit_cpu` runs the bound body; its kernel returns to `complete_part` | Its declared input reads have ended. `publish_cpu` may expose output sections before that return, so early output presence is not equivalent evidence. |
| BLAS / BNNS | The configured numerical call returns, then `complete_part` runs | The actual library read has finished. |
| Metal, including supplied engine encoders | `submit_metal` uses `addCompletedHandler` | The command buffer has completed its accesses. Merely encoding or committing it is insufficient. |
| Core ML | The existing prediction completion callback | The asynchronous prediction has ended; existing backing/result validation remains. |
| SEND | `mesh_progress` retrieves the CQ entry; `mesh_posted` contains the exact row and page used | Device access to that source span ends independently of other readers. |
| RECV | The CQ entry publishes the received rows | The destination becomes a usable input. Subsequent CPU/GPU uses have their own lifetimes. |
| Host observation | `Result` exposes a NumPy view; `consume` records caller consumption | A view that escapes the configured graph is not covered by a static numerical last-use proof. Its existing program ownership must be retained unless its actual borrowing lifetime is represented. |

At `d67bc85`, `mesh_map_read` updates per-input-occurrence, per-row generation records,
then increments a shared completed-reader count. The last reader sets a READ plane.
`mesh_claimable` and `mesh_receive_postable` inspect retirement before overwriting the
same destinations. This is a storage-reuse protocol, not a Pallas requirement.
`MESH_PRODUCING` also prevents duplicate issue while a function is running; deleting it
without replacing that responsibility in the invocation representation changes behavior.

No payload zeroing occurs in `mesh_arena_release`: it clears page-pool ownership bits.
The allocator also excludes device-held pages. The current per-value retirement path
does not return arbitrary extents to that allocator: live programs keep their extent
bindings until teardown. Moving the reader counter to another thread would leave this
fundamental allocation policy unchanged.

## How to sparsify lifetime information

This is a derivation from the configured use graph, not a new runtime algorithm.
For a value region `v`, let `U(v)` be its actual memory-reading operations, including
sends and any external borrow. Remove a use from the reclamation condition only when
another retained completion is proven to occur after that use has finished reading
`v`. The remaining incomparable last uses form `L(v)`.

    reusable(v) = every completion in L(v) has occurred

A single last use needs one existing completion. Several incomparable last uses need
all their existing completion facts. This need not be implemented as a decrement per
reader per page. Regions with identical lifetime conditions can share one description;
contiguous spans can be represented as ranges; already implied dependencies can be
removed. Preserve partial publication granularity independently of this coalescing.

Example: a received partial `x[i]` feeds local matmul `f[i]` and forwarding send `s[i]`.
Those operations may run concurrently. Their actual completions together permit
reclamation of `x[i]`. Completion of `f[i]` alone does not imply completion of `s[i]`.
If `f[i]` publishes an early output and keeps reading `x[i]`, that early publication
also does not imply `f[i]` has finished. Meanwhile `x[i+1]` goes to another registered
span and can be produced without consulting either old completion.

Retaining all backing for one configured finite execution is an even simpler valid
policy when the allocated memory covers that execution: no interior reclamation is
needed. Overlapping executions get distinct backing and can retire separately.
For repeated use, reuse a storage instance only after its actual lifetime ends, while
new work uses another available instance. Never turn this into a barrier on the next
producer, an acknowledgement from the remote consumer, or a fixed-age assumption.
A modulo address alone does not establish that its previous value is dead.

## Source changes and remaining work

The uncommitted stamp-scanning collector was removed. It added an arena scan, a metadata
mutex and a thread, retained per-page member records, and did not provide different
backing to producers. It was not the required storage refactor.

The direct-binding refactor removes the following competing representations and relays:

- `MeshExtent`, its copied native extent, its lookup dictionary, the additional extent
  owner array and per-function wrapper operand arrays. The native extent owns the
  Metal buffer directly; views and matrix descriptors resolve that same extent.
- `submit_ready`, the execute closure forwarding to a C submit function, the extra
  Metal submission relay, execution-kind tags and realization-time CPU closure wrapping.
  Setup binds the native submission callback directly.
- The `prepare_part`/`bind_part` relay and mutable occurrence field. Algebra functions
  already have exactly one configured occurrence. The subsequent
  [MLX/JACCL review](mlx-jaccl-structure.md) removes the unused native occurrence
  interface and separate function-watch allocation as well.

The remaining implementation must change backing and uses together:

1. Represent the caller's simultaneous value instances in registered storage and their
   numerical bindings at setup. Preserve the selected backend. Fixed-view backends can
   use separately prebound instances; they do not all need to become page-table kernels.
2. Derive storage lifetimes from those same input/output/transfer regions. Use existing
   backend completion facts and retain unresolved storage off the producer path.
3. Use the actual receive landing spans as operands. Current receives are posted only
   after ready-order transfer descriptions arrive. SEND/RECV matching and frame sizes
   must agree; posting fixed destinations in a different order is not solved by a GC.
4. Remove the old per-page consumer protocol and destination-reuse checks when the
   value-instance representation replaces their lifetime and duplicate-issue duties.
5. Trace repeated streaming producer/consumer invocations through this complete source
   path. The existing one-write-per-instance example cannot establish repeated reuse.

A background worker is an implementation option only if the remaining pool-return
work justifies it. A general tracing collector, CPU RCU integration, per-page stamped
reader graph, automatic placement and a second scheduler are not strict dependencies
established by this review.

Native, package and engine compilation check integration of the direct-binding changes.
This review makes no latency measurements, no zero-overhead claim, and no claim that
repeated-generation storage or the whole collective objective is complete.
