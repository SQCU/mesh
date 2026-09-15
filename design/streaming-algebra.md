# Streaming algebra

The [asynchronous collective contract](async-collectives.md) is the complete scope.

The caller supplies numerical functions and mesh/tensor placement. Configuration
binds their input and output regions to actual registered storage before invocation.
The same configured region may be an output for one function, a transport source,
and an input for another. These are uses of the value, not additional copies.

For contraction, independent reduction-domain contributions have distinct storage.
Each can compute, publish and move while other contributions are unavailable.
Their configured reduction produces the final value at the corresponding output
coordinates. A nonlinear operation consumes that value once its actual sum exists;
it does not wait for unrelated output coordinates.

A producer's publication does not await transport completion, a consumer, or the
rest of its enclosing operation. Each consumer runs the partial computation whose
input regions are available. Configuration supplies distinct storage for live value
instances so reuse does not create a dependency between independent operations.

The public calling surface is documented in [streaming kernel calls](indexed-library.md).
Pallas supplies the calling structure and indexed forwarding mechanism; its GPU
backend details are not a separate implementation requirement here.

## End-to-end source derivation

The concrete caller is `examples/streaming-chain.py`. Both participants bind
`linear → swish → linear → reduce`; the root additionally
binds `swish → linear` on the reduced result. All these functions are realized
before either participant feeds its input regions.

The [MLX/JACCL structure review](lit/mlx-jaccl-structure.md) documents the applied
native simplification: each realized function owns its dependency index entries
directly, with no second grid or per-occurrence watch. The caller selects the verb
matching its numerical dependency and placement; no verb is the universal path.

For an output region I and participant p, let

    U_p[I] = X[I] W_up[:, J_p]
    H_p[I] = swish(U_p[I])
    D_p[I] = H_p[I] W_down[J_p, :]
    Y[I] = sum_p D_p[I]
    Z[I] = swish(Y[I]) W_consumer

The caller supplies J_p and the owner of each Y region. There is no dependency
between distinct I regions unless the supplied contraction reads across them.
The following are the publication sites for this chain:

| Stage | Publication in source | Consumers released |
|---|---|---|
| Input feed | `Program.write` finishes through `mesh_writer_complete` and `mesh_complete` | Local up-projection contributions reading those input rows |
| Each local up/down/consumer contraction contribution | Accelerate CPU calls finish through `complete_part`; Metal command-buffer completion calls `complete_part` | Its addition-tree parent, or the next operation if this is the complete result |
| Addition-tree nodes and both swish stages | Generated CPU stores call `publish_cpu → mesh_publish_partial` per section; final `complete_part → mesh_complete` releases inputs. Metal command-buffer completion calls `complete_part` | Numerical consumers and transfer edges of each published output region |
| Reduction transfer | `link_post` sends a published D region; successful `mesh_receive_complete` publishes its configured destination | The root's `kernel_call(add)` tree |
| Reduction sum | The root's add nodes use the generated-source publication paths above | The root's post-collective swish, without requiring other Y regions |
| Final contraction | The same contraction completion path above | Exported Z regions |

`mesh_complete` calls `mesh_publish` on output rows. `mesh_publish_partial`
and `mesh_publish` set presence and notify canonical readers. `mesh_notify`
queues affected compute and send rows; `mesh_events` follows their bound edges.
`mesh_fire` invokes the bound function's submission callback directly. CPU arithmetic runs on a concurrent
worker queue; Metal/CoreML submission returns to the presence handler after
submitting device work. None of these publication sites waits for transport or
for an unrelated region. Device completion precedes publication because writes
must be visible before a consumer reads them.

`_ExpressionRegions.parts` assigns distinct storage to K contributions.
Their addition tree completes the sum before nonlinear use. Each public Metal
kernel call has a queue bound at setup, so producer and consumer calls are not
placed on one shared command queue. This is a source-level account of eligible
interleaving, not a measurement of simultaneous device execution or latency.

## Contraction addresses

`matrix_parts` partitions each mutable BLAS/BNNS/MPS operand footprint so that
its first and last indexed elements lie in one registered backing block. Strides
are nonnegative, so all intervening accesses lie in that block. Partitioning a
row or column splits the corresponding output coordinates; partitioning K keeps
the output coordinates and uses the library's beta=0, then beta=1 accumulation.
This is setup work. It does not wait for an input or select a numerical backend.
Immutable constants retain their setup mappings for the program's lifetime.

For logical element offset `e`, scalar bytes `s` and page bytes `P`, the bound
address contains the canonical table entry for `extent.first + floor(e*s/P)`
and remainder `(e*s) mod P`. Invocation loads that entry and forms
`physical_page*P + remainder`. The block partition is essential: resolving only
the first page of an arbitrary multi-block matrix would not describe its other
rows after relocation. Each block's pages must remain a contiguous physical run,
as already required by the transport binding. The mapping must remain stable for
the value's active readers; this change does not implement that lifecycle.

BLAS receives resolved pointers with its setup dimensions and strides. BNNS
receives them through its two-input filter apply call; its descriptors retain no
mutable operand pointer. Filters live with the bound function and are destroyed
after its existing execution lifetime. Mesh creates no operand buffers or filters
during invocation. Internal BNNS scratch behavior is library-owned; the previous
explicit `BNNSMatMul` workspace is no longer used by this beta-capable API.

MPS receives setup-created matrix views selected by the resolved address. For
row stride `R`, setup enumerates possible address residues modulo `R`, spaced by
`gcd(P,R)`, and shares identical view tables. The quotient selects the matrix row
origin; the residue selects the view's buffer offset. Apple's matrix origin uses
`x` for rows and `y` for columns (also specified in the SDK's
`MPSMatrixFullyConnected.h`). No MPS matrix is constructed at invocation.

The arena buffers used for indexed kernels and MPS inputs are untracked Metal
resources. Actual producer completion supplies cross-function visibility. MPS
result buffers cover individual backing blocks, with tracking limited to that
block so the library can order its own accumulation passes. Tracking a whole
arena bank would serialize independent output blocks; it is not used for results.
These buffer objects alias registered pages and allocate no copied operand store.
This follows Apple's [hazard-tracking scope](https://developer.apple.com/documentation/metal/mtlhazardtrackingmode)
and [MPS matrix storage](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix).

Core ML arrays, supplied engine encoders and persistent NumPy views still retain
fixed extent addresses. Their integration, receive preposting and backing reuse
remain necessary before claiming G4. This address change establishes no latency
parity or coverage of those unfinished paths.

The existing two-participant `examples/streaming-chain.py` ran on September 14
with these bindings at `2a5b041`, Metal numerics, the FP32 inputs in
`/tmp/mesh-tp-consumer`, split 256, tiles 128 rows / 128 K / 256 columns and two
instances. The root returned all 64 output regions and exited successfully.
Their values matched the prior `931d438` run exactly when keyed by output region;
arrival order differed. Logs are `/tmp/mesh-indexed-streaming-root.log` and the
peer's `/tmp/mesh-indexed-streaming-peer.log`. Native, Python-package and engine
integration builds completed on both participants. This demonstrates use of the
changed MPS bindings in the producer/reduction/consumer chain, not relocation or
latency parity. No additional test program was added.

## Realized storage and the unresolved receive-posting requirement

`mesh_algebra_realize` prints `participant` and `planned_arena_bytes` before
registering numerical execution callbacks. The byte count sums every allocated
`mesh_extent.bytes`, including input, weight, intermediate, output and
receive storage, with page/block rounding already applied. Aliased Refs do not
add allocations. This is the invocation's operand footprint, not total bridge
registration, reader metadata or backend workspace.

G4's receive-posting requirement remains unsatisfied. The current bridge first
receives an index frame and then posts a payload receive on the specified planned
pages. It does not prepost every payload receive at realization. The upstream
constraints are recorded in [ledger D1–D5](collective-dependency-ledger.md#d1-the-verb-is-send-into-a-posted-recv-there-is-no-remote-write):
the substrate supports SEND/RECV, with FIFO receive consumption per queue and
finite receive capacity. Preposting fixed destination pages in declaration order
on one queue would mismatch arbitrarily ready sends. Ordering sends to match
would introduce a dependency between unrelated producers. The present index path
avoids that ordering dependency but retains an index-message receive-posting hop.

`mesh_issue` also retains its output-lifetime check. Deleting it alone would
allow a repeated firing to overwrite a value whose consumers have not finished.
G4 requires resolving both transfer destinations and distinct live invocation
storage; neither condition follows from a successful output or an empty grep.

## Historical staged execution (superseded)

On September 14, 2026, `examples/streaming-chain.py` at commit `2bc27de`
executed on the MacBook (producer 0) and Mac mini (consumer 1) through the
canonical Thunderbolt SEND/RECV bridge. Both used shared-region ABI 26,
65,536 pages, four pages per block and one payload queue.

The supplied FP32 tensors were X[32,32], W_up[32,32] and W_down[32,16],
with tile rows, K and columns all 16. Input values were generated in X, W_up,
W_down order with NumPy `default_rng(73).standard_normal(shape)`, cast to FP32
and divided by 8. The actual calls were:

    producer 0: X → linear(X, W_up) → swish → copy to consumer 1
    consumer 1: received H → linear(H, W_down) → copy to producer 0

Both contractions bind two K contributions per output region. Each hidden
region becomes a transfer source after its numerical completion and an operand
of the consumer's corresponding contraction contribution after arrival. The
caller's terminal loop only reads returned results; it issues no intermediate
numerical work. This staged execution did not split either contraction across participants.
It does not demonstrate tensor-parallel work sharing.

The terminal received region (1,0) followed by (0,0). Their actual values are
[recorded here](../measurements/streaming-chain-2026-09-14.txt). The first entries
were -0.02199353650212288 and -0.012772580608725548 respectively. The returned
order is the observed execution result, not a required schedule. It does not
by itself establish the timing of every internal contribution; the source
bindings establish their partial-input dependencies.

The producer exited after consuming both regions. The consumer was then closed
with SIGTERM. No reference evaluator, timing comparison, pass/fail threshold or
runtime acceptance criterion was introduced.

## What the execution record does not demonstrate

The returned region order is not a demonstration of latency, low overhead,
absence of synchronization, or a bounded nonblocking operation. It records an
executed numerical chain. Those stronger properties were previously overstated.

The source exposes these distinct costs and waits:

| Boundary | Current mechanism | Consequence |
|---|---|---|
| Output claim | `mesh_issue` checks actual inputs and output reader ownership; `mesh_reset` clears presence and assigned read planes before setting producing | Work proportional to covered rows and assigned reader planes precedes dispatch. An unavailable operand/storage claim returns without spinning for it. |
| Publication | `mesh_publish_partial` / `mesh_publish` update atomic planes; `mesh_notify` pushes each affected row onto compute/send lists using shared memory | Publication does not wait for delivery, but incurs atomic contention, and list work. CAS retries have no stated per-call time bound. |
| Consumer discovery | A dedicated `mesh.presence` thread spins on shared publication notices and invokes `mesh_events` on the numerical dispatch queue to traverse affected reader edges | Ready work can incur notification and queueing delay; asynchronous submission alone gives no bound on that delay. |
| Numerical issue | The bound CPU, Metal or Core ML submission callback enters a dispatch group and submits its numerical body; only synchronous CPU work uses a worker queue | CPU arithmetic executes away from the presence handler. Dispatch internals and worker scheduling are not shown to have zero contention or bounded latency. |
| Transfer | Dedicated spinning send and receive threads post registered SEND/RECV and drain their respective completion queues; initial receives are posted by the receive thread too | Capacity shortages defer posting. Reusing a receive destination depends on its actual readers. These are distinct from waiting for an unrelated tensor to finish. |
| Terminal application | The example supplies every input block before entering a busy-poll output loop; formatting/printing occurs in that loop | Observed output order omits intermediate timing and includes application observation delay. |
| Setup and destruction | Compiler process waits, synchronous registration/removal on the presence queue, and dispatch-group/semaphore waits during destruction | These boundaries explicitly wait; a claim about numerical publication must not be generalized to the complete program lifecycle. |

For this particular call configuration a 16×16 FP32 tile contains 1,024 useful
bytes. Its transferable backing block has four 16,384-byte pages. `mesh_realize`
rounds the useful transfer length up to a 4,096-byte frame; the payload therefore
uses 4,096 bytes before index traffic and device framing. These are source-derived
sizes, not measured timings and not a placement policy.

Publication-to-issue latency, enqueue overhead, transport delay, numerical
execution duration and contention cost have not been separated by the recorded
run. No numerical latency or overhead claim follows from it. The source shows
which mathematical dependencies are local to a region; it does not by itself
establish a wait-free guarantee for the complete implementation.

## Tensor-parallel operation use

The current example uses the MLX authors' column-sharded then row-sharded
linear decomposition. The input dataset used for the latest invocation has
X[4096,256], W_up[256,512], W_down[512,256] and W_consumer[256,128]. The caller
chooses J_0=[0,256), J_1=[256,512), 128-sized tiles, and assigns Y rows below
2048 to participant 0 and the remaining rows to participant 1.

Both participants compute their local D contribution. `reduce_scatter` sends
each contribution to its configured owner and adds it there. `all_gather`
replicates each completed Y region; participant 0 applies swish and the final
contraction. This is the chain traced above, including both directions of
communication and a consumer after the collective.

The invocation at `45902d8` returned 32 final 128×128 regions using Metal on
both Macs. It establishes use of the distributed arithmetic and the integrated
transport path after sparse-routing deletion. It does not measure overlap,
publication-to-consumption latency or speedup. Earlier root-only reduction runs
are retained in the historical measurement record; they are not evidence for
the current collective topology.

RDMA posting and completion draining belong exclusively to `mesh.rdma.send`
and `mesh.rdma.receive`. Neither worker executes numerical functions, application
callbacks, connection setup, sleeps or a general-purpose dispatch queue. The
control thread establishes and tears down connections; it never posts receives,
including the initial receive fill. Each receive completion immediately attempts
to replenish its queue, and each send completion immediately attempts further
posting. These ownership rules do not imply zero hardware latency or remove
existing dependencies on receive descriptions and reusable destination storage.

## Loading model shards into registered operands

`streaming-chain.py --model weights.gguf --numerics libgemma_metal.dylib`
interprets its up/down weight arguments as model tensor names. The existing
`ModelFile` reader supplies shapes during setup, and the caller's `--split`
selects the hidden partition. The engine loads each partition directly into the
transposed view of its configured Ref, then `Program.constant(ref)` marks those
pages constant. No intermediate NumPy weight array or alternate operand backing
is used by this path. The ordinary `.npy` input path remains available when
`--model` is absent.

The setup binding accepts the existing loader's FP16, BF16 and FP32
vectors and matrices. `--normalize TENSOR_NAME` loads a model normalization
vector directly into the configured FP16 scale Ref; bare `--normalize` uses
the example's all-ones scale. It adds no quantized decoder. The file handle is closed when the
example leaves its setup/program scope; numerical invocation uses only the
registered weight pages and previously bound numerical functions.

`--gate-weight TENSOR_NAME` selects the engine's existing FP16 gated GELU
activation. Gate and up projections bind independently; their complete regions
feed `gemma_mesh_gelu_mul`, whose output Ref has separate storage. Down projection,
reduce-scatter, all-gather and the downstream consumer use the same graph paths.
For this option the example's input and consumer weight files must use FP16;
model weights load into the configured FP16 projection Refs during setup.

`--residual` with FP16 inputs and `--normalize` uses the existing fused engine
normalization/residual kernel after all-gather. Each row block reads its original
input Ref and the completed reduced projection, then publishes a separate output
Ref. Input blocks cover the full feature width required by this row-wise operation;
other row blocks and other configured instances remain independent. The encoder
accepts independent operand offsets and a pipeline captured during setup.
