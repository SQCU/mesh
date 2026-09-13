# Streaming algebra over mesh

Operator scope, September 13, 2026: extend canonical mesh with generalized
linear algebra, scatters, all-gathers and reductions over independently
consumable tensor extents. DNN modules are compositions written by a separate
library. The first acceptance program is a streamed producer, peer sum,
elementwise transform and contraction, with an unrelated extent deliberately
withheld. This explicitly authorizes the executable example in `examples/`;
it is an observable numerical acceptance program, not another specification.

## Interface and ownership

`rdma/mesh-algebra.h` is a C interface; `libmesh-algebra.dylib` implements it
using Metal elementwise operations and MPS matrix multiplication. It builds
against `libmesh.dylib`, without the model loader, Engine module, checkpoint,
model dimensions, Python or environment-selected numerical configuration.
The `mesh-dataflow.h` source API remains compatible. Shared-memory version 20
publishes completed configuration before the bridge pairs; existing clients and
bridges must be rebuilt together. The startup exchange also has a new magic.

Create an algebra context on an already attached mesh context. Create tensors
as arrays of explicitly shaped extents. Configuration allocates every extent
in the canonical registered arena, maps its logical rows to those pages, and
creates a zero-copy Metal view of those same file pages. A transferable extent
is padded to the bridge's transfer block. No metadata gaps, transport copy,
separate dense payload or allocation during numerical invocation is introduced.

`mesh_tensor_view`, `mesh_view_slice` and `mesh_view_transpose` describe numerical
coordinates and strides. They never select a kernel. Readiness belongs to the
explicitly allocated extent; slicing a view does not invent finer readiness.
Every extent has its own pages, so independent subpage-sized numerical extents
are padded separately. This first implementation trades padding for preserving
readiness in the existing page table. Transport blocks, numerical shapes and
MPS internal tiles remain distinct units.

`mesh_algebra_bind` binds the arithmetic and dependencies before realization.
The built-ins are affine, weighted addition, multiplication, tanh, exponential,
row sum, reciprocal square root, and matrix contraction. Affine computes alpha*A+beta; addition computes
alpha*A+beta*B; contraction computes alpha*A@B into a fresh output. The remaining
operations ignore alpha/beta. An output view must cover its complete extent
with a non-overlapping dense or transposed layout. Each extent has one producer.
A partial contraction is a separate output extent; combine partials with explicit
addition functions. Thus neither concurrent accumulator mutation nor publication
of unfinished reductions is implicit in the interface.

Numerical functions are captured in a static list. Each currently has one
configured work extent and uses mesh_issue with that work index. The scan emits
one command buffer for each eligible function; its completion records GPU timing
and errors, then calls mesh_complete on that same function. No prediction task,
phase scheduler, whole-tensor join, or GPU wait is added. Missing inputs simply
do not issue. Readiness and reuse remain the existing present/reader bits.

A statically indexed scatter/gather over extents is composition of affine copies with alpha=1,
beta=0 between indexed extent views. Transfers bind each source/receive extent
to a numeric identity and queue. On two participants, all-gather exposes each
locally owned extent together with the peer-owned receive extents. Peer sum
is a weighted-add function over local and receive views. Larger collectives
can compose the same numerical operations, but the current bridge connects one
peer; this change does not implement a multi-peer topology.

## Producer and consumer boundaries

A producer can be a bound numerical function or externally supplied input.
A mutable external producer calls mesh_tensor_issue before writing an extent,
then mesh_tensor_complete to publish it. These use the same mesh_issue and
mesh_complete functions as GPU producers, including transfer publication.
mesh_tensor_publish is a convenience for bytes initialized before their first use;
constant parameters use mesh_tensor_constant before realization. Function
completion supplies all subsequent publications. The caller registers output
extents with mesh_algebra_return before realization, polls their availability,
reads them, and consumes each return after use. Different tensor extents are
independent invocations; there is no mandatory tensor-wide completion.

A consumer names the extent it reads, even if it only reads a slice. To start
on smaller pieces, define those pieces as independently produced extents.
The numerical library owns the split, not transport. A matrix contraction's
K slices produce separate partials; a fixed addition structure determines the
final association. A row normalization can compose square, row sum, reduction
of row statistics, and final elementwise operations; final normalization still
requires that row's complete statistic. No DNN-specific epilogue is embedded.

The MPS implementation retains its existing optimized GEMM loop. It binds
row-major or transposed input views before execution. Every configured output
extent is an independently submitted MPS call. This establishes streaming at
call boundaries, not visibility of internal tiles of a single opaque MPS call.
No faster local baseline is claimed. More aggressive producer publication must
be implemented in a backend that exposes those completion boundaries and must
be validated with the actual visibility guarantees of Metal and transport.

## Transport ordering

D5 remains a physical matching requirement: SEND and RECV have the same FIFO
order on each queue. An unproduced early binding blocks later transfers on that
queue. Independent queues continue. The example assigns the two independent
row groups to separate queues and withholds extent zero, which is first in
queue zero. Queue one must finish a complete contracted row group first.
This demonstrates real independence, but does not claim arbitrary ready-order
transfer within one queue or unbounded independent channels. Choosing more than
the configured queue count folds streams onto FIFO orders, as before.

Returns hold their storage until mesh_algebra_consume. Transport readers hold
send storage through NIC completion. Metal command completions retain the algebra
owner and its views. Callers must keep the attached mesh context alive through
completion and destroy the algebra before detaching. Configuration is single
threaded, scanning has one caller, and completion callbacks use atomic reporting
counters. Those counters are observational; readiness never reads them.

Configuration errors identify invalid geometry, overlapping output ownership,
unsupported MPS layouts or exhausted storage; they do not alter node availability
or transport policy. Existing geometry must be honored so a completed extent
cannot falsely publish unwritten bytes. No runtime repair selects another backend.

## Acceptance

Build with `make -C rdma .build/streaming-algebra`. Run the same committed main
on both participants with `rdma/.build/streaming-algebra 0 20 128` and
`rdma/.build/streaming-algebra 1 20 128`. Both bridges need matching block sizes
and at least two queues; 4096 arena pages with 4-page blocks cover the default
shape. MODEL_PATH and the metal-microbench checkout are not dependencies.

Four input extents represent two row groups and two contraction-axis panels.
Each GPU produces `P=0.5*X+0.125` directly in transferable pages. Each participant
adds the two peer contributions, applies tanh, and contracts each panel with
its corresponding constant weight view. Two partial contractions add into each
final row group. Each participant owns one group for the final all-gather.
Transposed weight views exercise contraction layout binding. Row sums exercise
a further streaming reduction consumer. Square, sum, affine, reciprocal square
root and broadcast multiplication compose row normalization, which is also checked. The host computes
an independent numerical expectation after completion and checks every output,
its peer-owned gathered replica, finiteness and row sums.

After four warmups, ordinary and withheld-input runs alternate. Withholding
extent zero must leave its producer output unavailable for a new read while group one finishes.
Only after observing that completed group does the caller supply extent zero.
This is an input-dependency experiment, not clock-coordinated device launch.
The 30-second limit is solely the acceptance process's failure deadline.
No sleeps, rendezvous messages or second participant scheduler establish work
readiness. Reports contain count, mean and sample variance for normal, delayed
and early-completion latency, plus numerical error and GPU command totals.
These are measured intervals, not a claimed speedup or transport-only cost.

## Registration reuse across clients

Successive acceptance clients exposed a provider lifetime bug: pair teardown retains
registered memory, but pair setup recomputed the registration count using the
caller’s data offset instead of the retained registration origin. For power-of-two
blocks this changed one region into two and wrote past the allocated MR array.
Setup now uses the realized registration origin for both initial registration and
reuse. The acceptance program must run across successive client attachments,
including different tensor extents, without restarting the bridge between them.

The acceptance input changes each trial, so stale transport data cannot pass by
coinciding with a previous invocation. External producers acquire their extent
through its configured output rows before overwriting it.

## Measurement status

[Recorded observations](data/streaming-algebra-2026-09-13.json) retain both the
earlier failures and the successful setup repair. At commit `9762d14`, six
consecutive client lifetimes passed on both the M5 and M4 without restarting
either bridge between clients: row counts 128, 37, 128, 1, 128, and 257.
The final shape spans multiple 64 KiB transport blocks per numerical extent.
Each client performs four warmups, twenty normal invocations and twenty
withheld-input invocations, changing input values every time.

That is 120 delayed-input observations per participant, all completing the
independent contracted row group before the withheld input was supplied. The
maximum absolute error across these runs is 1.38101313e-7. Gathered peer outputs
match exactly; row sums and the composed normalization pass their independent
numerical checks. The JSON records count, mean and sample variance for each
measured latency. These observations establish this acceptance scope; they do
not establish a performance gain or validate FP16 and larger peer topologies.

The earlier startup fault manifested as absent initial receive completions and
later data occupying earlier receive slots. The registration-origin repair fixes
an independently established out-of-bounds write. A symmetric metadata exchange,
fixed initial PSN, separate completion queues, and an RTR-only setup boundary
did not eliminate the startup fault. The retained repair posts each configured
initial receive window before completing the bilateral setup boundary and enabling
sends. Setup follows completed mesh_realize, using the configured-owner header
word. User authorization for fixed setup costs is recorded as ledger D16.

There is one setup-complete byte per peer per connection, on the existing bounded
TCP setup channel. The tensor progress path carries no acknowledgements, setup
checks, phases or completion tokens. link_receive is the same receive-posting
function during setup and ordinary transport progress. Failed setup releases its
posted occupancy through the existing link teardown.
