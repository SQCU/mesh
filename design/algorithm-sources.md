# Algorithm sources and implementation obligations

The operator requires a citation to one of the authors/publications below for
every new function. Source citations may point to this document; prose stays in
documentation. A citation identifies the mechanism being implemented, not a
claim that the publication implements this repository verbatim.

## Operand matching and storage

Gregory M. Papadopoulos and David E. Culler, *Monsoon: An Explicit Token-Store
Architecture*, ISCA 1990, sections 2–3.
https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf

Compiler-assigned operand locations and storage-associated presence transitions
realize dataflow firing. Here the canonical RDMA page table is the sole authority
for operand presence, destination ownership and use counts. Configure numerical
functions and their input/output maps before invocation. Selection claims actual
destinations; completion publishes them after device visibility. No Job, stage,
parity cursor, completion token or caller-owned consumed mask may authorize work.
Monsoon's activation frames and token queues do not authorize equivalents here.

`mesh_pages_scan` checks destination stamps before matching input presence and
stamps, then claims the selected destinations. The ordering avoids repeatedly
reading operands for outputs already issued; it does not add state or change
the conjunction defining eligibility. The caller binds normalization's two
ownership spans to this same mechanism before invocation. Matching and claim
still rely on one scanner for overlapping destinations.

Each matching conjunction stops at its first failed condition. Reading the
remaining pages after a missing operand or already-issued destination cannot
make that candidate eligible. The prior eager loops nevertheless read every
page and dominated the sampled numerical thread during full-graph execution.
Short-circuit evaluation changes no persistent state and requires no readiness
cache or scheduler; later scans observe newly published values normally.

Every numerical intermediate, accumulator and index value occupies actual
sendable page payloads. Views gather/scatter through those pages. Lifetimes must
include residual reads, asynchronous device reads and hashing. Reuse follows
completed dependent reads; allocating every intermediate forever or renaming a
stage counter as a page index does not implement that lifetime rule.

### Reusing configured physical spans

`mesh_pages_slot.storage` names a one-based root slot for a shared physical
span; zero retains a separately allocated span. Roots name themselves and
cover the widest participating slot. Logical rows remain distinct. The
compiler allocates the physical span once and initializes its payloads.
This is configuration of ordinary sendable pages, not a storage privacy class.

`storage_ready` reads the existing physical-owner table; `claim_rows` writes
the destination stamp, restores its literal entry and records its physical
owner. As with the existing non-atomic multi-destination claim, one scanner
must own overlapping destinations, including physically overlapping spans.
The current caller assigns only normalization destinations to these spans;
all their claims occur on its single numerical scan thread.

`retire_storage` requires full publication and completion of every configured
dependent output at the corresponding generation. Whole-span completion is a
conservative read-lifetime proof; first-page publication is insufficient.
The existing use counts reflect outstanding dependents. Transported values
also retain storage through NIC completion and local hashing, without waiting
for a peer digest comparison. Retirement clears actual entries and submits
zeroing to the existing progress-to-helper queue. The helper makes the existing
physical-owner entry free only after zeroing the payload. It does not return
local arena pages to the receive bridge's free list.

This implements the storage lifetime obligation scoped by Papadopoulos and
Culler above. It does not infer lifetimes from numerical graph names, call
application kernels, or add caller phases. Correct dependency specifications
remain required. Broader storage reuse and link-recovery acceptance are still
unfinished; configuration-time zeroing during recovery does not establish
that all recovery paths are correct.

## Collective arithmetic and asynchronous reduction

Rolf Rabenseifner, *Optimization of Collective Reduction Operations*, ICCS 2004.
https://fs.hlrs.de/projects/rabenseifner/publ/myreduce_iccs2004_2.pdf

Pitch Patarasuk and Xin Yuan, *Bandwidth Optimal All-reduce Algorithms for
Clusters of Workstations*, JPDC 69(2), 117–124, 2009.
https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf

Use reduce-scatter followed by all-gather. For balanced ownership the sent
payload per participant is 2(n-1)|S|/n. Match input pages, accumulate in FP32,
and emit each result when its numerical inputs are complete. The collective
names do not impose whole-tensor execution barriers. A normalized row requires
all its feature contributions; independent rows need not wait for one another.

`rdma/mesh-pages.c::reduce_step` implements indexed partial addition using
configured maps and destination stamps. `reduce_run` scans that static list
on one runtime worker. `mesh_pages_reduces` realizes the maps; `mesh_pages_start`
and `mesh_pages_stop` manage worker lifetime, not numerical dependencies. This
single worker is an implementation choice supporting operand matching, not a
published guarantee of speedup. Transport progress remains independent of
arithmetic. The existing FP16 materialization and later normalization still
need replacement by the prescribed accumulator/index-page composition.

## Endpoint checking

Jerome H. Saltzer, David P. Reed and David D. Clark, *End-to-End Arguments in
System Design*, ACM TOCS 2(4), 277–288, 1984.
https://web.mit.edu/saltzer/www/publications/endtoend/endtoend.pdf

Endpoint checks establish integrity. The repository additionally requires
post-consumption checking: digests are ordinary pages, mismatch is an evaluation
failure value, and link failure is link status. Digests cannot gate numerical
execution or supply a missing numerical lifetime dependency.

## Overlap and performance evidence

The PyTorch authors, *Introducing Async Tensor Parallelism in PyTorch*, 2024.
https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487

Decompose actual communication/computation dependencies to overlap their work;
preserve efficient local numerical functions. Small operations can lose to launch
cost, poorer matrix utilization and resource contention. One scan issues all
currently ready work for a configured GPU function together. Measure matched
local and RDMA executions, including numerical error, latency, throughput,
submission count and actual communication cost. Published bandwidth bounds do
not attribute seconds of end-to-end time to RDMA.

## Remaining replacement

The caller's Job/stage control, consumed masks and embedding/head completion
words have been removed in committed revisions of metal-microbench. Distinct
configured functions now have distinct value rows and share the NFE stamp.
Native FFN input, MPS FFN intermediates, MPS attention input and vocabulary
input/output now address sendable payloads. This is not all operand storage:
other backend inputs and weights still require work. MPS/tensor attention Q/K/V
and attended intermediates now also occupy ordinary sendable page payloads;
the corrected indexed output stores were checked against the preceding logits.

Broader local-page recycling, complete read-lifetime proofs, accumulator/index
page reduction, independent admission without digest gates, and link-error
recovery/repetition remain unfinished. Normalized-input slots now share and
recycle configured physical spans after completed dependent reads and
asynchronous zeroing. Other local slots retain separate spans. These citations
provide no exemption.
Do not mark transport and asynchronous
map/reduce complete until both flows use that representation and actual RDMA
measurements establish correctness and performance on the supported workloads.
