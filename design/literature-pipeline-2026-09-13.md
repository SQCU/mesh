# Literature-driven streaming implementation and operational evidence

Measured source: `82be91f` on main, September 13, 2026. Native packages on both
participants were built from `e5a1eee`; the intervening commit changes only the
existing streaming-algebra example's input supply. M5 Max local participant,
M4 Pro peer, Thunderbolt RDMA, one payload QP and one index QP per participant.

## Published mechanisms and their implementation

The JAX authors' [Pallas collective computation](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html)
composes local computation with asynchronous movement of explicit regions. Its
[software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)
explains how separate buffers remove dependencies introduced by storage reuse.
Mesh retains BlockSpecs, region references and source/destination transfer indices.
`nn.linear` partitions M, N and K, materializes independent K contributions in
canonical pages, and reduces matching output tiles. Swish consumes each completed
hidden tile; the following contraction consumes that tile as one K contribution.
No other hidden feature tile has to finish first. Setup owns allocations, bindings
and compilation. Nine independent gold instances have distinct storage.

The Linux authors' [llist publication list](https://raw.githubusercontent.com/torvalds/linux/master/include/linux/llist.h)
provides the push/detach mechanism used for changed-row notifications. Setup builds
row-to-function and row-to-transfer adjacency. Publication queues row indices;
compute examines affected functions and the bridge queues actual eligible transfer
indices. Neither repeatedly searches every configured transfer. Source readiness
still comes from canonical stamps. Coalesced publication is acquired when removing
membership, preserving publishers' writes before evaluating readiness.

[Triton's normalization implementation](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
provides the region-local fusion pattern. Mesh scalar expressions compile during
setup to CPU C or Metal source through the existing kernel-call interface. RMSNorm
fuses square/reduction per feature panel, combines FP32 statistics, then fuses
normalization per output panel. The statistic is a real dependency; unrelated rows
remain independent. The implementation does not apply nonlinearities to unfinished
sums as though the nonlinearities distributed over addition.

[Apple TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
specifies SEND/RECV, receive credits and frame-count matching. Transfers now retain
useful byte lengths and both endpoint indices. Matching SGEs reference canonical
registered pages directly. These gold tiles send 16 KiB instead of the 64 KiB
ownership allocation, without packing or staging. Index posting does not wait for
a software acknowledgement before payload posting. Correct ownership and hardware
credits remain necessary; unrelated unpublished values do not stop ready sends.

The detailed mechanisms and source citations are in
[algorithm sources](algorithm-sources.md#publication-work-lists),
[region expression fusion](algorithm-sources.md#region-expression-fusion), and
[transfer lengths](transfer-length-and-shape-curves-2026-09-13.md).

## Gold computation and measurement boundary

The existing `examples/streaming-algebra.py` runs:

`FFN → RMSNorm → summed learned embedding → FFN → RMSNorm`.

Inputs are 256 by 128 FP32, with 64-row, 64-K and 64-output-column sections.
Both FFNs contain their existing partitioned projections and swish composition.
The first exchanges node 0 to node 1; the second exchanges node 1 to node 0.
Local execution uses the same numerical composition and returns the exchange
operand directly. CPU uses the existing Accelerate contraction path; Metal uses
the existing MPS contraction path plus region kernels.

One instance warms up before timing. Eight independent instances are then supplied
together. Row section zero of the first measured instance is withheld until its
final output section `(1, 0)` is observed. The example checks that the withheld
section's output remains absent. Reference calculation and output validation run
after the batch finishes, outside timed computation. This demonstrates downstream
completion with an unrelated operand extent absent. The selected first-section
time is an observer upper bound, not necessarily the earliest output in the batch.

All local/paired CPU/Metal executions passed, with maximum absolute error
1.651473e-6 for CPU and 1.845171e-6 for Metal against the example's float64 reference.
These are FP32 results, not FP16 precision evidence.

| Backend / cohort | Local eight-instance batch, ms | Paired batch, ms |
| --- | ---: | ---: |
| CPU 1 | 14.330459 | 11.851792 |
| CPU 2 | 14.425417 | 12.711125 |
| CPU 3 | 16.802333 | 13.972875 |
| Metal 1 | 51.447250 | 46.353875 |

For each matched CPU cohort, throughput gain is
`100 * (local_batch_ms / paired_batch_ms - 1)`.
An online Welford summary gives **count 3, mean 18.216685 percent,
sample variance 16.890939 percentage-points squared**. Individual gains are
20.913858, 13.486548 and 20.249648 percent. This compares the distributed graph
against the improved local graph in this workload. There is no matched old/new
local-kernel measurement, so no separate local-kernel or compounded gain is claimed.
Three cohorts are modest evidence, not a general scaling curve. Metal has one
cohort, with sample variance undefined; CPU was faster for this small shape.

The earlier serial-input driver measured one supplied instance at a time and thus
could not establish batch pipeline throughput. Its results are retained separately,
without using them as a matched baseline for the concurrent-input driver.

## Causal traces and copy-free source review

Raw records and numerical summaries are in
[the measurement directory](../measurements/literature-pipeline-2026-09-13/).
Each compressed JSON includes observations and complete native compute/transport
traces; paired cases include the peer's records. `summary.json` retains all eight
batch cases and the online throughput comparison. Serial-input records identify
source `e5a1eee`; batch records identify `82be91f`.

Compute traces retain actual input row ranges, output identity, ready/start/complete
times and backend. Transfer traces retain both endpoint identities, useful bytes,
ready/post/CQ observation times and occurrence count. Host timestamps use
CLOCK_UPTIME_RAW on their own machine. GPU start/end timestamps are retained but
are not subtracted from host timestamps or another machine's clock.

For CPU cohort 1, 130 of 144 single-occurrence send intervals on each participant
overlap a numerical function's start-to-completion interval. Cohort 2 counts are
122 and 120; cohort 3 counts are 129 and 137. Those counts include warmup. An
interval is `[payload post, observed completion]`: this establishes outstanding
transport concurrent with computation, not precise physical wire occupancy.
Metal host intervals also overlap, but include command-buffer queuing and must
not be presented as physical GPU/wire concurrency.

Trace summaries deliberately retain cold events. In particular, the first node
can produce warmup values before link setup completes. Its all-event
producer-complete-to-post mean therefore includes setup delay and is not a steady
state dispatcher cost. Functions with repeated submissions are excluded from the
single-occurrence interval calculation; traces store the latest occurrence.

Copy-free execution follows from pointer flow: tensor blocks map canonical page
storage, native views carry those addresses, CPU/MPS/expression kernels write their
declared outputs there, and SGEs use the same registered pages. K partials and
reduction outputs are mathematical values with distinct storage, not transport
staging. An embedding gather necessarily writes selected elements to its declared
output. These conclusions come from source, not elapsed-time inference.

## Remaining boundaries

- FP16 contraction panels and their addition tree still round to FP16 between
  partials. RMSNorm statistics are FP32. A retained FP32 contraction accumulator
  through the entire K reduction requires a supported backend implementation;
  this FP32 gold does not settle that requirement.
- Specialized Xonotic expert/neighborhood operators still have whole-region
  lowering. Ordinary 2D contractions, pointwise arithmetic and row reductions use
  the shared tiled library path; universal operator lowering is unfinished.
- Shorter messages retain conservative maximum-block queue capacity. Recovering
  additional outstanding messages requires retaining actual per-request frame
  costs and sizing completion storage accordingly.
- Many small CPU jobs and MPS command buffers still incur dispatch costs. Fusion
  is available through the common expression interface, but not every numerical
  composition is fused. Setup-selected larger efficient regions must preserve
  independent publication where the algebra permits it.
- Larger shapes and heterogeneous CPU/ANE/MPS placement need matching local curves.
  Existing large FP16 curves are documented separately; they do not justify a
  backend choice for this small FP32 program.

No whole-tensor readiness barrier or alternate compatibility API was added to
address these boundaries. All measurement clients exited; both bridges remained
ready with zero attached clients. Each bridge registered 549,650,432 bytes for
this setup, including the metadata and traces, within the visible allocation
preflight already installed.
