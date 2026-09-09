# Paged matrix execution and remote placement

The live policy uses `tensor.py`, `tensor_metal.py` and `tensor_runtime.py` for
symbolic capacity dimensions, fixed shader variants, lifetime-planned persistent
storage and recorded numerical phases. Routed expert forward and reverse products
reuse the generic 64-by-32-by-32 SIMD matrix implementation. Sparse neighborhood
integration consumes its complete indexed support. Full-axis reductions preserve
the model's mathematical contractions.

`tensor_mesh.py` partitions this same graph by declared ownership, including VJPs
and parameter updates. Numerical spans use native receive/transmit pages and GPU
copies; complete input bundles can be consumed directly through page tables.
Remote results are staged before publication. The current flow and local evidence
are in [POLICY-PROGRAM.md](POLICY-PROGRAM.md#persistent-execution-and-transport) and
[POLICY-IMPLEMENTATION-20260907.md](POLICY-IMPLEMENTATION-20260907.md).

## Historical MLX paging implementation

The following layout, implementation and deployment observations describe the
September 5–6 backend. `paged_matrix.py` retains those eager matrix primitives for
source-level calculations. Its independent MLX preparation and tensor-packet RPC
are no longer the live policy/learner execution path. Statements below about a
running placement describe the recorded run, not the current fleet state.

## Page layout

A physical page contains 64 rows. Features occupy multiples of 32 columns, with
unused columns zero-filled. A page-table entry contains four int32 fields:

| Field | Meaning |
|---|---|
| `PHYSICAL` | Physical page index in the arena |
| `START` | First position in the logical row-index array |
| `VALID` | Number of real rows, from zero through 64 |
| `EXPERT` | Weight bank used by this page |

The logical row-index array maps positions to input rows. Repeated indices express
top-k expert routing; scatter adds their contributions back to the input roster.
Physical pages and table entries may be reordered independently. Empty experts
have zero active rows; their allocated capacity remains in the execution schedule. Capacity is rounded to powers of two, so nearby populations
share allocation shapes. Page-slot arrays and dense table snapshots are cached;
MLX owns reuse of released allocation buffers. Live autograd values remain separate
immutable snapshots, so reuse cannot overwrite tensors needed by a pending VJP.

Gather initializes the entire arena to zero, including unused pages, row tails,
and feature tails. Projections preserve this padding. Contractions execute the configured table loop and select each expert
contribution in registers. Projection does not return early for empty pages. Gather
uses an explicit zero row and scatter an explicit sink for inactive entries.
Gram normalization uses configured rank, so padding cannot change the result.
Padding never becomes an additional observed participant or state word.

## Fixed kernels and prepared numerical programs

Five Metal variants implement the matrix backend: gather, scatter, projection,
transposed projection, and contraction. Their templates contain page/tile geometry
and the projection transpose bit. Their templates do not specialize on logical row occupancy or expert assignment.
Physical tensor capacities and feature widths determine the prepared program and
its launch extents. Input and metadata types are uniform; values and masks do not
change the launch schedule within that capacity.

Projection uses fixed 64-by-32 output tiles and 32-element reduction tiles.
Contraction uses fixed 32-by-32 output tiles and reduces 64 rows per mapped page.
Both keep partial sums in a cooperative tensor. This avoids an intermediate Gram
matrix per page and an intermediate weight-gradient tensor per page. All tensor
operations request full float32 precision. There is no scalar expert-boundary
fallback: routing puts each expert in homogeneous pages.

Dense matrix operations use the same page backend. Their mathematical products
remain dense; sparsity comes from active pages and selected experts, not from
discarding required Gram entries. Elementwise policy operations use MLX inside prepared compiled programs.

Stable Metal templates alone were insufficient: the initial implementation still
rebuilt graphs and skipped work according to page-table data. The September 6
[execution repair](STATE-REDUCTION-RCA.md) compiles local forward, sampling,
likelihood/integration and a combined loss/backward/accumulation/optimizer program, and
removes occupancy-dependent Metal returns and loop skips. Both-machine probes
measure trace reuse and device allocation calls. The expanded grouped-update probe
still observes intermittent allocations; static graph reuse does not establish a
fixed intermediate arena. Host input staging and capacity
growth remain outside that bounded numerical-solve result; the whole server loop
is not claimed allocation-free. Work reports estimate algebraic FLOPs and do not
measure the extra masked tile work or hardware instruction counts.

This follows the logical-to-physical block mapping described by
[PagedAttention](https://arxiv.org/abs/2309.06180) and
[vLLM's paged kernel design](https://docs.vllm.ai/en/latest/design/paged_attention/).
The GPU implementation uses Apple's fixed tile and cooperative accumulation APIs
from the [Metal Performance Primitives guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).

## Historical placement during the September 5 paging migration

The tethered runtime now runs the complete learner, both policy arms, optimizer
moments, and replay on the Mini. The dedicated game server and SDL client run on
the laptop. Game observations travel from bridge node 0 to learner node 1;
responses return to node 0. The learner uses the same paged matrix backend locally.
Weights and gradients stay on the Mini. No additional remote optimizer protocol
is needed. The model remains rank 128, hidden width 341, eight experts, top-2.

The Mini owns curriculum supervision with `--strategy-node 1 --peer-node 0
--server-host Ms-MacBook-Pro.local`, without `--distributed-scale`. These are
workload placements, not node classes or restrictions on either node's capability.
SSH has a connection deadline and keepalive failure detection. Deployment and
recovery details are in [MINI-LEARNER.md](MINI-LEARNER.md).

## Earlier block placement

The preceding tethered run used `--distributed-scale --distributed-scale-operation block`.
`RemoteScale` sends the complete residual expert block to the Mini, including its
input projection, routing, both expert projections, Gram fusion, output projection,
and reverse pass. The optimizer and the remaining policy operations stay on the
laptop. At the current 128/341/8/top-2 shape, a representative 8-player,
220-instrument forward moves approximately 34% of policy FLOPs to the Mini;
Gram-only placement moved approximately 2.5%.

`scale_rpc.py` packs each request's exact float32 tensor snapshot and computes the
forward result or VJP. `xonwire.def` names the four scale request/response types.
The numerical runtime (`matrix_worker.py`) owns this participant's mesh context
and receives tensor requests directly. The engine's `mesh_ipc.c` uses a local socket
for game traffic and no longer forwards numerical traffic. Neither the worker nor
the learner opens a verbs device. The current ownership and lifetime are described
in [POLICY-TRANSPORT.md](POLICY-TRANSPORT.md). Request
timeouts finish the same operation locally and report that fallback. Responses
retain transport session/request identity, preventing an old response from being
accepted as a new parameter snapshot.

Gram-only placement remains the CLI compatibility default. Full-block placement
retains versioned parameter storage per policy owner and uploads a generation when
missing; the optimizer remains on the learner. Activations and cotangents still
use CPU staging. In particular, the large default
model has a much larger transfer cost than this live 128-rank run. Paging fixes GPU
execution geometry; it does not remove that transport cost or establish a guarantee
of GPU saturation. Literal temperature comparison is also unavailable from the
current paired telemetry.

## Validation, September 5, 2026

Twenty-seven targeted tests passed on the M5 Max laptop and M4 Pro Mini. They covered
float64 product/gradient references, all seven block derivatives, changed parameter
snapshots, RPC fallback, ablation behavior, checkpoint/optimizer continuation,
reordered physical pages, empty experts, partial pages, repeated-row gradients,
capacity shrinkage, and the five-variant invariant across changing shapes. These
are historical observations from the subsequently removed suite, not current
completion criteria. The current policy flow is [POLICY-PROGRAM.md](POLICY-PROGRAM.md).

The original shape-specialized kernels took a median 572 ms on previously unseen
Mini row counts in a three-product fixture. Removing dimension specialization
reduced subsequent unseen-shape calls to 1.8 ms. The paged implementation then
replaced that intermediate runtime-shape implementation. Full-block benchmarks
show matching outputs and gradients; warm timings vary by population, with some
small forwards slower because of packing and padding. No universal warm-throughput
speedup is claimed.

Before paging, 18 paired samples of full-block offload averaged 20.8% GPU activity
on the Mini and 8.8% on the laptop. After the intermediate runtime-shape reload,
another window averaged 11.5% and 5.8%. These are live workload observations, not
controlled temperature or hardware-counter FLOP measurements. Final paged samples,
both-host benchmarks, and test logs are in `.build/scale-placement-20260905/`.

The final window with block placement averaged 2.2% Mini GPU activity and 9.0% laptop activity.
It followed a host reboot and used a different, smaller match population. Thus the
general goal of keeping the Mini busier than the laptop was not established by that placement. The
latest same-input forward comparison had zero output error but approximately
489 ms remote roundtrip versus 4.4 ms local computation.

At the time of that measurement the curriculum, client and training session were running. Normal worker shutdowns
preserved optimizer checkpoints. The Mini unexpectedly rebooted at 11:43 PDT;
the curriculum detected the broken server connection, saved training state, staged
the next match, and resumed. The same client timed out and reconnected automatically.
The deployment commands did not signal either mesh bridge or request a host reboot. Client network
and process recovery are documented separately in
[CLIENT-RELEASE.md](../xonotic/render/CLIENT-RELEASE.md).

The 11:43 panic and the earlier 10:16 panic both report a kernel data abort at
address `0x98`, with the faulting PC at `IOThunderboltFamily + 0x343c` and identical
AppleThunderboltRDMA backtrace offsets. The earlier crash predates this paging
change. The trigger and driver repair remain unresolved. Both reports and the
normalized comparison are preserved in the evidence directory.
