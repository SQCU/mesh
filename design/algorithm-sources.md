# Algorithm sources and implementation obligations

The [complete replacement requirements](completion-requirements.md) consolidate
the current obligations, source disposition and acceptance evidence. They take
precedence over completion implications in the chronological notes below. The
active source callers now use the configured `mesh_rows_*` runtime; the excluded
`mesh_pages_*` and scheduler implementations have been deleted. The connected
source refactor still requires a fleet build and real-input execution evidence.

## ANE backing identity and registration

The operator supplied these additional implementation references on September 9,
2026:

- Ramchand Kumaresan, *Orion: Characterizing and Programming Apple's Neural
  Engine for LLM Training and Inference* (2026),
  https://arxiv.org/html/2603.06728v1, sections 2.2 and 4.2: IOSurface tensor I/O
  and the private runtime interface. This is not evidence that our CoreML
  wrapper exposes every allocation.
- Spencer H. Bryngelson, *Apple Neural Engine: Architecture, Programming, and
  Performance* (2026), https://arxiv.org/pdf/2606.22283, sections 27.6 and 28.6,
  tables 27.6 and 28.6: submission includes input/output/intermediate surface
  identities; the inspected loaded program maps weights, constants and working
  storage as well. Distinct device addresses can refer to one physical page.
  These observations make backing registration a concrete investigation;
  they do not demonstrate Thunderbolt registration of those allocations.
- Apple Machine Learning Research, *Deploying Transformers on the Apple Neural
  Engine* (2022), contributors Atila Orhon, Aseem Wadhwa, Youchang Kim,
  Francesco Rossi and Vignesh Jagadeesh,
  https://machinelearning.apple.com/research/neural-engine-transformers:
  channels-first layout, contiguous 64-byte-aligned last axis, and avoidance of
  transpose/reshape copies. Alignment and padding belong to configuration.

The operand identity remains the Papadopoulos–Culler operand-slot contract
documented below. Allocation order does not determine compliance: a surface
allocated by the numerical backend can supply the actual canonical mesh pages.
Different CPU/device virtual addresses do not imply different physical storage.
Neither a second copied arena nor a descriptor for unregistered memory suffices.

Source inspection at mesh `a22fe8e` and metal-microbench `5d4de29` identifies the
actual integration boundaries:

- `mesh_at` in `rdma/mesh.h` computes addresses from a single base and page
  stride. `region_sge` and registration in `rdma/mesh-flow.c` assume that same
  arena. Those assumptions must accommodate the actual surface-backed pages;
  adding an IOSurface wrapper alone does not connect transport.
- `mesh_metal_memory` already uses `mach_vm_remap` with copying disabled and
  `newBufferWithBytesNoCopy`. This establishes an existing alias mechanism,
  not ANE or NIC registration evidence.
- The installed macOS SDK's `IOSurfaceRef.h` exposes `IOSurfaceGetBaseAddress`,
  `IOSurfaceGetAllocSize`, `IOSurfaceCreateMachPort` and
  `IOSurfaceLookupFromMachPort`. A Mach right can carry the same surface into
  the separate bridge process during configuration; a raw pointer cannot.
- `CoreMLFunction.backingUsed` compares returned and supplied MLMultiArray
  object identities. It does not inspect internal weights or intermediates,
  establish physical backing identity, or demonstrate registration.

The connected implementation must realize surface ownership, mappings,
registration, literal page addresses, tensor geometry and lifetimes before
invocation. ANE completion publishes the configured output rows and error
metadata; it does not add a prediction-waiting scheduler. Invalidation retains
ownership of memory still accessible to a device rather than assigning that
same physical memory to a replacement computation. C07–C11, C21 and C28–C30
remain open until this is integrated and measured on both real participants.

### Installed IOSurface and Espresso evidence

Inspection on September 9, 2026, at mesh `e05804d` and caller `568e311`:

| Participant | OS | IOSurface wraps existing mesh address | Returned size |
|---|---|---|---|
| M5 Max | macOS 26.3.2, 25D2150 | Yes, exact same base address | 16384 bytes |
| M4 Pro | macOS 26.5.1, 25F80 | Yes, exact same base address | 16384 bytes |

The inspection opened the existing `/mesh0` shared mapping, read `hdr.data_off`,
rounded it upward to the 16384-byte OS-page boundary, and mapped 16384 bytes at
file offset 3211264. `IOSurfaceCreate` received two CFNumber properties:
`IOSurfaceAddress` was that mapping's base address and `IOSurfaceAllocSize` was
16384. On both machines creation succeeded, `IOSurfaceGetBaseAddress` returned
the supplied address exactly, and `IOSurfaceGetAllocSize` returned 16384.
No numerical values were written and no ANE execution was submitted. This
establishes wrapping that registered shared-memory range without a second
CPU-address allocation; it does not establish an ANE tensor layout, internal
parameter binding, coherence after device writes, or arbitrary-size surfaces.

`IOSurfaceAddress` is not declared in the installed public header. A primary
implementation using it is the `create_surface_with_address` function in
https://github.com/SolutionsExcite/darksword/blob/main/src/main.m. Only its
ordinary IOSurface property construction is relevant here; none of that
repository's exploit operations was executed or imported. The ANE backing
interpretation remains grounded in Bryngelson and Kumaresan above.

Read-only Objective-C runtime inspection of the installed Espresso framework
found these identical instance-method encodings on both machines:

| Selector | Encoding |
|---|---|
| `setExternalStorage:ioSurface:` | `v32@0:8Q16^{__IOSurface=}24` |
| `ioSurfaceForMultiBufferFrame:` | `^{__IOSurface=}24@0:8Q16` |
| `ane_io_surfaceForMultiBufferFrame:` | `@24@0:8Q16` |
| `metalBufferWithDevice:multiBufferFrame:` | `@32@0:8@16Q24` |
| `createIOSurfaceWithExtraProperties:` | `^{__IOSurface=}24@0:8@16` |
| `initWithIOSurfaceProperties:andPixelFormats:` | `@32@0:8@16@24` |

The class is `EspressoANEIOSurface`. Its `params_dict`, `width`, `height` and
`rowBytes` instance variables are present. The abbreviated selectors
`ioSurfaceForFrame:`, `IOSurfaceForFrame:`, `metalBufferWithDevice:`,
`setAliasingMem:`, `bytesPerFrame` and `totalBytes` are absent. Do not infer an
installed callable signature from a wrapper's method name. The source reference
https://github.com/mdaiter/ane at `e13f45818cbc8edeb6627d86a547f6e10d3883a6`
lists the external-storage mechanism; the actual encodings above were obtained
from the local runtime, not assumed from that list.

This makes wrapping already-registered mesh views an immediate implementation
route alongside reverse registration. Neither route requires copied operand
storage. Before changing the native numerical path, connect the actual model's
surface ownership, tensor properties and completion to the configured rows;
the presence of an external-storage method alone does not locate every internal
weight or intermediate surface.

After these inspections the normal `mesh-mini` SSH route failed through its
configured `ws-2` jump host. Direct SSH to `ms-mac-mini.local` succeeded and
reported a reboot at 21:26 on September 9, with approximately one minute of
uptime. No restart had been issued by this work, and no diagnostic-report file
from the preceding 15 minutes was found. The reboot cause is unestablished;
the successful surface calls alone must not be reported as stability evidence.
The new bridge process was paired and responsive with no attached numerical
client. Direct synchronization and both caller builds succeeded afterward.

## Native binding observation

The actual M4 FFN artifact loads as `MLDelegateModel` with `MLE5Engine`.
Installing Espresso's surface class is not evidence that this model uses that
class. The existing numerical evaluator now observes the retained E5 ports'
direct-binding flags after computation, without making them numerical control
dependencies. `coreMLStorageReport`, its `stored` accessor and
`MatrixView.coreMLArray` cite the Bryngelson/Kumaresan backing account and the
Saltzer–Reed–Clark endpoint principle through metal-microbench
`docs/parameter_groups.md#native-binding-observation` and `#native-iosurface-views`.
The private observation ABI comes from the installed runtime, not those papers.

Apple's installed `MLMultiArray.h` explicitly documents
`initWithPixelBuffer:shape:` for IOSurface-backed multiarrays that can avoid a
buffer copy. The implementation wraps compatible existing addresses with
IOSurface, then CVPixelBuffer, then MLMultiArray. It does not allocate another
numerical tensor. The native evaluator's previously ignored channels-first
configuration is now realized directly by the existing RMSNorm output-stride
specialization and native result views.

At metal source `6dca474`, a real 128-row M4 FFN reported direct binding for
input `x` and outputs `y`, `y_1920`; the earlier strided wrappers at `89c8507`
reported false while still satisfying wrapper identity. Exported inputs and
outputs are byte-identical. Local median latency changed from 2.822333 to
0.937292 ms in three measured calls after four warmups, with unchanged MPS
reference relative RMS error 0.0002229233. This is a local-shape result, not a
full NFE or TP performance claim. Full records and the separate old-caller
RDMA regression are in metal-microbench
`docs/data/native_iosurface_binding_2026-09-09.json`.

Internal weights/intermediates remain explicitly unverified. The existing mesh
page-gap layouts still require complete configuration and caller migration;
one successful local IOSurface binding does not justify hidden storage or
permit claiming C10 complete.

## Transport page addressing

Papadopoulos–Culler operand identity and the literal registered-buffer binding
documented by Apple TN3205 require transport to address the configured pages.
`link_worker`, `udp_link_worker` and `stream_link_worker` now retain the actual
mapping header and derive addresses through its page geometry. They no longer
carry a separately supplied base/span or multiply page indices by a private
4096-byte assumption. Receive capacity likewise follows `hdr.pgsz`.

This removes an implementation obstacle to changing configured page geometry;
the current allocator still chooses 4096 bytes. It does not claim alternate
geometry has been deployed or that headers can be removed from within an OS
page. CQ depth, MTU and loop budgets are distinct quantities and remain as such.
No readiness cache, transport framing or recovery mechanism is added. Existing
excluded control mechanisms still require the complete caller migration and
deletion prescribed by the requirements matrix.

Bridge deployment at `0f6e18a` exposed a separate launch-policy issue: M4
launchd recorded `OS_REASON_CODESIGNING` during reload before returning to a
paired process. Kernel logs identify AMFI launch-constraint rejection of new
PIDs 43984 and 46259 before bridge execution, at 21:58:14 and 21:59:51 on
September 9. The atomic build change did not eliminate that rejection; it
must not be described as fixing AMFI. Subsequent launchd starts succeeded.

The build previously wrote directly to the live executable path. Its recipe
now links and ad-hoc signs a new build artifact, then renames
it atomically into place. This preserves the inode mapped by the running
process until ordinary SIGTERM teardown. It is build publication, not a
computation recovery or synchronization protocol. No source backup or alternate
source version is created; source changes remain in commits on main.

At `6595130`, both bridges and both numerical callers built from synchronized
main sources. Before reload, lsof confirmed that the existing bridge processes
still mapped their old executable inodes while the new files had different
inodes. After ordinary SIGTERM restart, the running M5 PID 37048 mapped inode
89289704 and M4 PID 46539 mapped inode 1588004, each matching its current file;
both files passed strict code-signature verification. Their executable hashes
are preserved in metal-microbench
`docs/data/transport_page_addressing_2026-09-09.json`.

The six-layer, 1024-row, two-in-flight RDMA regression completed six NFEs per
participant with 72 agreements, zero disagreements/nonfinite outputs/retries,
and the prior exact logit hash. Both bridge PIDs remained alive and paired,
with zero bad completions and empty application rings afterward. This proves
the updated addressing at the currently deployed 4096-byte geometry, not
alternate geometry, native page-layout integration or full runtime replacement.

## Contiguous backing-page views

Operator clarification, September 9, 2026:

> mesh can be rewritten to allow zero copy views of contiguous backing pages for the sake of apis which want that

Papadopoulos and Culler (Monsoon, 1990) supply the named-operand model;
Bryngelson (2026), cited above, describes distinct addresses referring to the
same physical backing. Neither source supplies this macOS C ABI or proves its
RDMA performance. The implementation uses Mach's existing virtual-memory remap
operation with copying explicitly disabled.

`mesh_memory_view` realizes one contiguous CPU virtual range from an ordered
list of backing spans. `mesh_memory_span` is an address/length view description,
not operand storage, readiness, an invocation record or transport framing.
Every span starts and ends at an OS-page boundary. Mesh pages smaller than an
OS page retain their placement within that OS page; this operation cannot
remove in-page headers or independently rearrange subpages. Configuration must
choose a tensor layout compatible with those literal bytes.

The function reserves a virtual range, aliases each span into that range with
`mach_vm_remap(..., FALSE, ...)`, and returns the address and total length.
It does not copy operand bytes, register new physical storage, publish stamps,
or add work to numerical invocation. Configuration receives the literal Mach
error code. An incomplete mapping is released before returning that error.
`mesh_memory_release` releases only the alias; it neither frees canonical rows
nor cancels device accesses. The owner retains every view until its actual
readers/writers have finished, including across caller-selected invalidation.

The existing `mesh_metal_memory` now calls this canonical API before creating
its no-copy MTLBuffer and releases the alias through the matching API. Existing
receive/transmit pool and row-table Metal bindings therefore use this path;
there is no second Metal remapping implementation. Other APIs can use the CPU
view directly without requiring a Metal buffer as their storage owner.

This is configuration-time address interoperability. It does not yet import
IOSurfaces into the bridge, bind ANE internal allocations, replace the numerical
caller, or prove multi-span execution on the RDMA substrate. Those obligations
remain in the completion matrix.

Validation at source commit `0c15ee0` (measurement main `fda07b5`, numerical
caller `5d4de29`): both machines built the mesh libraries, bridge and numerical
caller. The existing six-layer RDMA runner, 1024 rows and two simultaneous
NFEs, completed six NFEs per participant with 72 agreements, zero disagreements,
nonfinite outputs and retries. Both final logit files have SHA-256
`2d890d398bd52555b1ded1d0508e62225f6f41e504abb578835b05d4e45b4a9d`.
The single-span path is exercised by this run. Multiple discontiguous spans
remain unmeasured. Details and exact configuration are in metal-microbench
`docs/data/contiguous_backing_views_2026-09-09.json`. Invocation times were
135.953–137.277 ms; completion gaps are not invocation latency. This is not
evidence of a replacement-runtime speedup or the 10-microsecond requirement.
The implementation adds 44 net source lines and 40 initial documentation lines;
no new evaluator was added.

## Explicit attention and projection weights

Papadopoulos and Culler (Monsoon, 1990), cited under operand matching below,
provide the operand-slot principle: a configured function consumes its named
values rather than allocating another representation of them. Rabenseifner
(2004) and Patarasuk–Yuan (2009), cited under collective arithmetic, supply the
partitioned-contraction/reduction context. Neither publication specifies Apple's
matrix-view ABI or proves this implementation's performance.

`AttentionWeightViews` names the Q, K, optional V, output projection and two
head-normalization weight values. Matrices have contraction orientation
`[input channels, output channels]`; normalization weights are `[1, head dimension]`.
`AttentionWeights.init` realizes head ownership and validates these shapes before
invocation. Explicit views bypass dense parameter loading. An absent V retains
the model's existing shared-K/V algebra; it is not a missing-operand fallback.

`realizeMetalAttentionRows` and the attention case of `realizeMetalParameter`
carry these views through the existing projections and attention launch path.
The bound `norm` operation uses the normalization view's buffer and byte offset,
so a weight contained in a mesh payload need not be copied to buffer offset zero.
That vector must be contiguous within its payload; configuration validates the
view before binding the kernel.

`realizeMetalParameter` also accepts an explicit projection/vocabulary weight
view, including the existing independent FP32 contraction ranges. This permits
the embedding and tied vocabulary projection to refer to the same parameter
storage. The numerical function does not load or select weights during execution.
The ordinary local/default binding retains its existing loader and arithmetic.

These bindings do not make arbitrary paged matrices acceptable to MPS. The
configured contraction must still expose valid payload-contained MPS subviews
or use the already selected backend's paged arithmetic. They do not supply
CoreML internal weight/intermediate bindings or replace the active mesh caller.
Completion requirements C07–C11 and C17 remain open until actual page-backed
execution through that caller is measured.

## Operator clarification: asynchronous error metadata

Operator instruction, September 9, 2026:

> error status returns are allowed by the specifications we use... but it can never be synchronous or blocking or consumed by a callgraph itself... it can only be passed monadically through a metadata channel... s.t. calling contexts can call `out, meta = meshfunction(x)`, with literal values corresponding to error codes and where/when they happened in a callgraph reported in the meta channel, not the output channel, for calling consumers to interpret and handle...

This supersedes blanket claims that error returns themselves are forbidden and
older requirements to terminate numerical execution on a negative status.
The [return-channel contract](pages-and-functions.md#asynchronous-error-metadata)
requires asynchronous propagation without error-dependent callgraph control.
Configured metadata pages carry the code and callgraph provenance; numerical
outputs remain numerical. Only the calling context interprets metadata. An
error report is not an execution-completion certificate. No new runtime
implementation is claimed by this clarification.

Current source disposition:

- `mesh_rows_receive` and `mesh_rows_progress` formerly propagated negative
  synchronous returns that stopped delivery or skipped transmission/retirement.
  Commit `af350bd` removes those paths from the unused replacement. It does not
  yet publish hardware-error metadata; removing the gates is only one part of
  the required replacement.
- `runReduceScatter` reads `mesh_pages_status` in numerical launch/completion
  paths and uses it to suppress work or cancel outputs. Those uses remain
  excluded. Reporting status in the final caller-facing metrics does not cure
  the internal control dependencies or provide per-occurrence provenance.
- Removing all status returns or converting functions to `void` does not by
  itself implement the required metadata channel. Add the configured return
  binding, migrate the single caller, then delete excluded implementations.

### Direct configured receive and NIC-completion consumption

The replacement `mesh_rows_receive` now performs the configured address gather,
page publication and dependent-input releases without descriptor validation,
epoch interpretation, duplicate-stamp suppression or synchronous error returns.
`mesh_rows_acknowledge` consumes each NIC-read completion and releases the source
named by the submitted page's immutable address. `mesh_rows_return` is called
only for received pages; it no longer rechecks that classification. Existing
page availability and ring capacity still determine whether the corresponding
page operation is currently possible.

`mesh_rows_progress` always visits NIC completion, receive, send and retirement;
its unsigned return is the number of page operations performed, not a status or
an operand. There is no error-dependent early exit. Removing these returns does
not implement hardware-error reporting: asynchronous metadata publication is
provided separately by `mesh_rows_report`, and the bridge/device binding to it
remains unfinished. The error-channel audit above describes the earlier source,
not the revised return types.

These functions require a realized, exclusively owned descriptor stream and
valid configured source/destination addresses. The transport binding must retain
the actual page mapping for every outstanding GPU/NIC access; invalidation cannot
rebind that memory or route its late completions into a replacement table. The
replacement is still unused, and its mapping-invalidation integration is still
missing. Removing epoch checks is not evidence that invalidation works. The old
active caller and its transport remain until the complete replacement is bound.

Papadopoulos–Culler (ISCA 1990) supply the assigned-storage/publication mechanism;
Rabenseifner (ICCS 2004) and Patarasuk–Yuan (JPDC 2009) supply the collective
exchange algebra; Saltzer–Reed–Clark (TOCS 1984) place acceptance at the endpoint.
All are cited below. These references do not certify this implementation's
mapping lifetime, error propagation or latency.

### Configuration owns receive input lifetimes

`mesh_rows_realize` now accounts for every logical row, including holes that
previously escaped validation because only produced outputs were examined.
Every row read by a configured function, a remote dependent-read binding or the
calling context's return maps must have a local producer or receive binding.
Its configured use count must equal those reads and its transmit uses. Unused
holes remain legal. This is configuration work before launch and allocates no
auxiliary ownership structure. It uses the existing maps and row-use algebra.

The receive path no longer rereads dependent input stamps or use counts to
validate a remote completion. A configured received output is the dependent-read
proof; the path performs the configured releases directly. Papadopoulos–Culler
(ISCA 1990) provide assigned operand storage and data-dependent firing, as cited
below; the repository's fixed maps supply the specific lifetime proof. Counting
uses alone does not establish acyclicity or protect against an invalidated
mapping receiving a late device write. Those remain distinct integration
obligations, not permission to add per-input runtime guards.

The subsequent direct-transport change removes receive descriptor/address
checks and negative return paths. Neither change establishes asynchronous error
propagation or complete caller migration.

### Asynchronous metadata publication

`mesh_rows_report` implements publication into a configured metadata output
page. Papadopoulos and Culler, *Monsoon: An Explicit Token-Store Architecture*
(ISCA 1990), supply the assigned-storage and presence-publication precedent;
Saltzer, Reed and Clark, *End-to-End Arguments in System Design* (TOCS 1984),
supply the endpoint responsibility for interpretation. Both publications are
listed below. Neither specifies this record layout or this API; the operator's
return-channel contract determines those choices.

The record contains the invocation stamp, occurrence time, configured function
identity, numerical index, peer identity and literal signed error code. The
producer supplies these values; publication neither samples a clock nor branches
on the code. `when` is a producer-local occurrence time in the clock domain and
units established by configuration. It does not imply synchronized peer clocks.
Zero and nonzero codes take the same publication path.

Before launch, realize an ordinary output map with one page per occurrence,
sufficient payload for `sizeof(struct mesh_row_metadata)`, one writer per row,
and return/transmit use counts. Include those outputs in the complete graph's
allocation and lifetime accounting. The `occurrence` argument selects that
configured output row; the record's `index` identifies the numerical operation
being reported. Different reporting sites and overlapping invocations require
distinct live destinations. There is no append counter, metadata queue, overwrite
of an unread event, or allocation during publication. Reuse obeys the same page
lifetime contract as every other output.

An asynchronous producer writes the record directly into the actual sendable
page payload, installs its configured page/use count and publishes the row stamp.
The function returns without waiting for execution, transport or a consumer.
The metadata output has no numerical dependents and publication releases no
numerical inputs. Only the calling context interprets it. Ordinary page transport
can carry it without inspecting its error code. This primitive is not yet bound
to the receive path or the single numerical caller, and does not establish full
error propagation, invalidation or a measured latency bound.

## Operator clarification: caller-owned repetition

### Complete allocation and unconditional invalidation

The operator further requires every page-table allocation for the complete
compute graph to be known before launch. The API must allow the caller to
invalidate its page table at any time, including for pages used by a hung
kernel. Consumption, freeing and explicit invalidation are permitted operations;
data-dependent branching, error recovery and related control flow are not
introduced into numerical functions, transport or reduction.

Every deployed node has NVMe storage or transitive RDMA access to a node with
NVMe. Missing or destroyed parameter pages are reproducible from that storage
and can be retransmitted. This invariant is supplied by the operator; the
numerical layer does not probe, infer or negotiate it.

Papadopoulos–Culler provide the configured operand-storage precedent, and the
caller-owned decision to reject and repeat remains an endpoint responsibility
under Saltzer–Reed–Clark. Neither citation authorizes the excluded control flow.
The implementation still owes a complete invalidation binding: clearing an
entry alone does not revoke an already-issued GPU or NIC memory access, and
late completion must not republish invalidated rows. A helper that only clears
entries would not establish the requested hung-kernel behavior.

Operator instruction, September 9, 2026:

> recovery is something which is only sanctioned through the mechanism of a totally feed forward nfe computation callgraph being re-run by the calling process because they don't like the results they got. there are to be no parity mechanisms or syncs, guards, waits, checks, or inferences in the mesh computation transport and reduction layer, and functions which use this layer are not to introduce data dependent control flow attempting to add overhead of this sort or any sort.

This governs the implementation and supersedes recovery obligations inferred
from older text below. The calling process evaluates the returned result and
may rerun the complete feed-forward NFE. Mesh computation, transport and reduction
do not recover or replay an invocation. Functions using them must not reintroduce
the excluded control flow. Feed-forward numerical dependencies and configured
page addressing do not authorize an additional runtime recovery protocol.

The proposed cancellation/drain/recovery functions were not implemented and are
withdrawn. `mesh_pages_recover` and the old caller's digest admission gate remain
implementation divergences, not required mechanisms. Existing defensive branches
in `mesh_rows_receive`, publication and release likewise do not acquire approval
merely because earlier additions compiled. The replacement must be reviewed
against this instruction before the single caller is migrated; excluded code is
deleted after that migration, in the operator's required order.

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

The first destination stamp provides a cheap rejection for already-issued
rows. Candidates surviving that check match their inputs before scanning the
complete output spans and physical ownership. The first stamp never authorizes
a claim: every destination still passes the full validation. This avoids
reading thousands of free destination pages for a future function whose first
operand is absent, while preserving the same matching predicate.

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

### Literal row functions

`rdma/mesh-dataflow.c` adds the required operations independently of the old
runtime. Papadopoulos and Culler, *Monsoon* (1990), sections 2–3, are the cited
source for storage-associated matching; the repository supplies the stricter
one-page-table representation. `mesh_row` contains exactly physical page, use
count and stamp. `mesh_rows` and `mesh_row_function` are immutable configuration
views, not per-evaluation state. They do not allocate an owner array, publication
bitmap, progress counter, task, token or queue.

`mesh_rows_validate` checks configuration, `mesh_rows_present` reads actual
rows, `mesh_rows_select` claims destinations in those same rows, and
`mesh_rows_publish` stamps completed visible output writes directly.
`mesh_row_data` resolves a row to its literal page payload. Selection requires
one scan owner for overlapping destinations; publication has one completion
owner per issued function. Selection output is transient numerical indices,
not a persistent consumed mask. `row_maps_overlap` checks the configured logical
and physical spans. `mesh_rows_realize` validates the complete configured
function list and bindings before initializing the canonical rows and zeroing
their assigned local pages. It excludes overlapping writers and overlapping
receive/output destinations. Its current realization assigns distinct physical
output spans to distinct functions; shared physical spans across functions are
not yet realized.

Output maps contain their fixed physical span and configured use count, in
addition to the logical row geometry. These are immutable allocation/lifetime
configuration, not a mutable ownership mirror. Selection accepts a newer NFE
only when the previous value's actual row has no page and zero uses. It retains
the previous completed stamp until claiming the newer value, then reinstalls
the configured physical page and use count. Clearing the stamp to zero on reuse
would incorrectly enable the old generation again, so that is not done.
The initial row state is likewise absent, zero uses and stamp zero.

`mesh_rows_uses` derives each row's use count during configuration from all
function-input occurrences, remote-read input maps, one NIC use per transmit
binding, and the graph's returned-page maps. Returned maps describe one caller
read of each row in their range; they are function-return storage, not a runtime
completion mechanism. Hashing must appear as a function input like every other
read. Neither the query nor use-count validation runs during an invocation.

The configuration helper `row_map_uses` counts the indices `i` satisfying
`0 <= i < rows` and `first + i*stride <= row < first + i*stride + count`.
For nonzero stride these form an integer interval, intersected with the configured
index range; zero stride contributes either all indices or none. Thus overlapping
input ranges and broadcast operands count every consuming function occurrence.
This realizes the compiler-known operand/lifetime relationships attributed to
Papadopoulos–Culler above without a mutable ownership mirror.

`mesh_rows_realize` compares every configured local-output and receive-row count
with that derived count before initializing the table. A map with nonuniform
uses must be split into the corresponding configured output maps. Counts that
exceed the row's 32-bit representation also fail this configuration comparison.
The only additional arguments are the immutable graph-return maps. Invocation
continues to decrement the literal use count directly; it neither recomputes
the graph nor validates a decrement. The caller still has to supply the complete
graph, including arithmetic, hashing, remote reads and returned values.

`mesh_metal_row_table` aliases this same array of physical-page/use-count/stamp
rows for the GPU using the existing no-copy mapping function. The configuration
owner supplies a page-aligned, page-rounded mapping whose lifetime covers the
alias. There is no GPU copy of the row table and no separate indices array.

`mesh_rows_publish` writes the configured output stamps and releases the
configured input uses. It performs no runtime operand validation, issued-stamp
test, or error return. Its loops traverse fixed configured maps. Configuration
and the numerical call graph must establish exactly one publication after each
writer completes; repeated publication is not intercepted by this primitive.
`mesh_row_release` directly decrements the actual row use count. It has no stamp
argument, validation branch, source-level compare/exchange retry loop or result
to interpret as permission. Each call consumes one use already accounted for
before launch. Atomic access applies to the literal row contents, not a separate
synchronization object. Papadopoulos–Culler remain the operand-storage citation.
The caller's complete use accounting and invalidation binding are still required;
removing guards alone does not establish those contracts.
Arrival integration must likewise provide exactly one release per configured
remote read proof. `mesh_row_zero`, called asynchronously after all
uses end, excludes writers through the row stamp, zeros the payload and removes
the physical page while retaining the completed stamp. The release owner retains
the physical page number through this operation and returns it to the appropriate
free list/bridge. Reinstallation occurs in selection while the row carries its
issued stamp. Hardware reads and hashing count toward the lifetime. A row whose
page is already absent is not zeroed or released a second time.

Receive bindings now carry the immutable input-row maps of the remote numerical
function whose output arrives in those rows. For received output index `j`, the
input map names `first + j * stride ..< first + j * stride + count`. Arrival of
that output with stamp `k` proves those configured remote reads have finished.
`mesh_rows_receive` validates the still-live input rows, publishes the arriving
output and releases one use for each mapped input occurrence. The destination's
existing stamp prevents a repeated arrival from releasing those uses again.
There is no acknowledgement page, consumed bitmap or per-function completion
counter. This is the dependent-output lifetime rule of the plain specification,
with Papadopoulos–Culler supplying the storage-associated presence principle.

For two-peer reduce-scatter, an arriving normalized output page can release the
corresponding sent partial page's remote-read use. The source also retains its
independent NIC-completion and hashing uses. Configuration must assign exactly
one output occurrence to each proven read, including when one numerical row has
several output pages. A proof for all inputs can attach to one returned output
only when that output's stamp proves all those reads finished. A digest verdict
is not substituted for the numerical output. The caller still has to realize
these input maps and counts; their existence is not evidence that the old
caller's lifetime logic has been replaced.

### Literal weight pages

`ModelFile.loadRows` in `metal-microbench/model_file.swift` realizes the
compiler-assigned operand placement described by Papadopoulos and Culler,
*Monsoon* (1990), sections 2–3, under the operator's stricter requirement that
every shared-memory operand occupy actual RDMA-sendable pages. Configuration
supplies a claimed output map, selected tensor coordinates and destination
precision. The loader decodes BF16, FP16 or FP32 directly from the model's file
mapping into those pages. It creates no intermediate dense operand buffer.
The source format and destination precision are configuration choices.

One numerical weight row occupies `ceil(columns / payloadElements)` consecutive
logical rows. The output-map stride selects the next numerical row. Each write
stops at the payload boundary; trailing payload elements remain as zeroed by
realization. Vector weights and scalar weights use the same mapping. The caller
owns the claimed pages throughout loading and publishes after loading finishes.
This is configuration-time initialization, not a function invoked during NFE
execution, and the cited paper does not specify model-file conversion.

The bridge currently uses 4096-byte physical pages. A 3840-element FP16 matrix
row exceeds one payload, so merely rebinding its previous dense buffer cannot
realize these maps. Matrix contraction bindings must consume this layout while
preserving the validated backend; the loader alone does not satisfy that work.
The existing caller still uses its prior loaders until the required bindings and
caller migration are complete.

`MatrixView` now describes page breaks along either matrix axis, with transpose
exchanging those axes and slices preserving aligned page origins. This is an
immutable numerical address map, using the same compiler-assigned operand
placement principle. `MatrixOperations` specializes the existing vector and
tensor kernels for weight-page width and physical stride during binding.
Vector loads gather each weight vector from its payload. The tensor kernel
rebases its operand tensor views at each weight-page boundary, with equal
contraction extents on both operands, and retains its FP32 cooperative
accumulator across those reads within one dispatch. Fixed reduction tiles must
divide the payload width. Final conversion remains after the full contraction.
No partial-sum command buffers, dense weight copies, or readiness state are
introduced. These are repository lowerings of indexed gathers and arithmetic;
Monsoon does not describe these Metal APIs. MPS page-break support and measured
numerical/performance validation remain outstanding.

`MatrixOperations.contract` binds one explicit contraction-index interval to
the existing multiplication implementation, writing an FP32 result into supplied
storage. A `MatrixView` slice entirely within one payload becomes an ordinary
strided view of the same bytes; a slice spanning payloads retains its page map.
This lets the configured MPS function address an individual weight payload
without a copy or a backend substitution. The indexed algebra is
`P_j = A[:, K_j] B[K_j, :]`, `C = sum_j P_j`, the local contraction counterpart
of the partial-sum reduction described by Rabenseifner and Patarasuk–Yuan.
Each `P_j` must have its own configured actual FP32 pages and input/output maps.
The binding encodes no sequence of partial contractions and inserts no waits;
each numerical function is independently issuable from its input stamps.
Canonical FP32 page reduction and final conversion remain caller integration
obligations. Splitting a contraction may change rounding and dispatch cost;
neither agreement nor speed follows from compilation.

Apple's [MPSMatrix documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix)
specifies row-major storage; its
[multiplication documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
specifies the optional transposes and scalar accumulation algebra. These API
references support the view mapping, not a claim that every input/result type
combination or this decomposition has been operationally validated here.

### Literal page reduction

The GPU `row_payload` address function follows Papadopoulos–Culler operand slots
and the page addressing required here: the table's physical page selects the
receive or transmit alias, with each alias's configured origin. It performs no
copy, residency repair or error-dependent scheduling. `bindRowNormalization`
binds both aliases before invocation, allowing its FP32 accumulator, gamma,
residual and scale reads to consume actual received pages as well as local pages.
Output rows are configured local producer pages. The accumulation and
normalization algebra remains the Rabenseifner/Patarasuk–Yuan partial-sum
composition described below; the papers do not specify this Metal address ABI.
This binder is not yet connected to the active caller, and shader compilation
alone does not demonstrate RDMA visibility or invalidation correctness.

Rabenseifner (2004) and Patarasuk–Yuan (2009), cited above, supply the
reduce-scatter/all-gather algebra. Papadopoulos–Culler supply the configured
operand maps and publication precedent. The addition entrypoints now consume
the existing `mesh_row_function` maps directly, without a second array of input
or accumulator row identities. Each input map describes one contributor's
ordered payload pages; output map zero describes the FP32 accumulator pages;
output map one describes one index page for each configured function index.
The index page's first uint64 value is the invocation stamp. Configuration gives
each such output one writer; it does not share a partially published index page.

`mesh_rows_validate_add` runs before invocation, after ordinary map validation.
For positive element count E, payload bytes B, and input element bytes s in
{2,4}, each input has exactly ceil(E/(B/s)) pages and the accumulator has exactly
ceil(E/(B/4)) pages. Output map one has exactly one page. No allocation or shape
validation occurs in addition. The selected f16/f32 entrypoint must match s.
Input and output map uses remain validated by `mesh_rows_realize` for the whole
graph. This validates storage extent, not the model's reduction partition.

For accumulator page q, first element a=q*(B/4), input reads use page
floor(a/(B/s)) and byte offset (a mod (B/s))*s. Each arithmetic span contains at
most B/4 elements. Since B is divisible by eight and s is two or four, every
span ends within its input payload, including tails and FP16-to-FP32 expansion.
The previous implementation instead offset a single input page by a*s; that
crossed headers or unrelated storage for sufficiently large operands.

Addition writes accumulator and index values only. After it returns, the same
`mesh_rows_publish` operation used by other configured functions publishes all
output row stamps and releases each configured input use exactly once. Addition
never releases inputs or publishes accumulator stamps itself. `mesh_rows_indexed`
requires the index page's published stamp before reading its literal values.
This removes the former competing publication/release ownership. No operation
uses the index contents to suppress or repeat an addition.

This ABI is still unused by the active numerical caller. Its page extent and
publication corrections do not establish caller migration, incremental reduction
on real RDMA inputs, or end-to-end performance; those remain integration work.

The addition entrypoints and `mesh_rows_normalize_f32` return no runtime status.
Their arithmetic contains no readiness scans, issued-stamp validation, duplicate
execution suppression or error-return branches. Counts, payload extents, index
locations, positive epsilon, claimed outputs and live inputs are configuration
and call-graph contracts. Arithmetic loops follow those fixed dimensions and
the fixed input order. This does not make the remaining selection, transport,
invalidation or caller implementations compliant; those still require replacement
and integration under the operator's no-added-control-flow instruction.

`mesh_rows_add_f32` applies the same operation to FP32 partial pages, including
the independent contraction partials above. Both entrypoints specialize the
inlined `row_add` at compilation for their source element width; invocation
does not choose an input format from a buffer. The FP32 variant adds directly
into FP32 accumulator pages and writes the same literal index value, without
an intervening FP16 materialization. Both use the configured input order and
require the same claimed destinations, index-page lifetime and single writer.
The shared implementation avoids duplicating the arithmetic/publication algorithm.
Rabenseifner and Patarasuk–Yuan are the cited partial-sum algebra; they do not
establish numerical equivalence to an unsplit floating-point matrix product.

`mesh_rows_normalize_f32` performs the plain specification's normalization and
residual addition from accumulator, gamma and residual pages into output pages.
Its caller supplies the numerical parameters, proves the full-row inputs from
the index/table, claims outputs and publishes only after the arithmetic finishes.
This is a numerical function, not a model-graph interpreter. The initial C
arithmetic is not a claim that CPU normalization is the fastest backend; the
configured GPU implementation must preserve the same page algebra and be
measured against the validated local function. The published collective papers
do not specify this model's normalization formula.

`metal-microbench/reduce_scatter.swift` supplies `bindRowNormalization` and
`rms_norm_accumulator_rows` for the GPU part of that same page algebra. The
configured maps gather FP32 accumulator elements, FP16 gamma/residual elements
and an FP32 scale from actual local sendable pages through `mesh_row.page`.
The kernel reduces squared FP32 values over the numerical row and scatters the
normalized residual result to FP16 output pages. No materialized FP16 reduced
input lies between addition and normalization. The within-threadgroup barrier
combines the eight SIMD partial sums; it is an arithmetic dependency, not a
transport wait. Papadopoulos–Culler supply the presence/matching principle;
Rabenseifner and Patarasuk–Yuan supply the surrounding collective algebra.
The model supplies RMS normalization, epsilon, gamma and scale.

The binding compiles its pipeline and fixes all operand maps before invocation.
Invocation encodes a supplied ready row range into the caller's command buffer;
it allocates no operand storage and performs no readiness polling or publication.
All four input maps and the output must name local arena pages covered by the
transmit alias. The caller proves full-row accumulator presence and all operand
lifetimes, then publishes on completion. Immutable geometry is encoder argument
data, not a separately allocated shared-memory operand buffer. This binding is
added before caller migration; compilation alone does not establish numerical
agreement or performance, and it has not yet been used in an RDMA evaluation.

These additions precede the caller migration. They do not constitute completed
transport integration, read-proof integration, recovery or RDMA validation.
The operator requires the existing caller to migrate before excluded
implementations and their interfaces are deleted.

### Literal page transport

Papadopoulos–Culler provide the storage-presence mechanism, and
Rabenseifner/Patarasuk–Yuan the page-exchange algebra cited above.
`mesh_rows_send`, `mesh_rows_receive`, `mesh_rows_acknowledge` and
`mesh_rows_return` bind actual pages to the existing SUB/CMP/ACK/REL rings.
`mesh_row_binding` is immutable correspondence between local and peer rows.
Realization requires bindings in increasing local-row order with disjoint
ranges. Receive lookup searches this same configured array by destination row;
it does not allocate a second lookup table. The selected binding still has to
match direction, peer, range and source-row correspondence.
`mesh_row_address` is the page's literal address/epoch/stamp header, not an
application frame-kind or handshake protocol. The transport worker owns those
rings. Each transmitted row has one configured transport use, released by its
actual NIC completion; local computation/checking uses are separate counts in
the same row. No flight array or additional descriptor queue is allocated.

Submission checks the existing page header to avoid sending the same row value
twice. On a full SUB ring it restores the prior header and retries on a later
scan; it does not publish an extra pending flag. The header and payload remain
unchanged through NIC completion. Each transmitted row has one bound adjacent
destination; configurations needing multiple copies must realize their
collective ownership/edges explicitly rather than overwrite an in-flight header.
Generation stamps must not be reused for another value under the same epoch.

Delivery installs the received physical page and configured binding's use count,
then stamps the actual destination. It never overwrites a live destination.
Each arrival restores the configured receive-use count; incoming data cannot
allocate or infer a numerical graph.
`mesh_rows_retire` implements the storage retirement part of Papadopoulos and
Culler’s operand-storage discipline through this specification’s literal use
count. An asynchronous owner scans the existing rows, skips unpublished values
and remaining uses, and zeroes completed values. It owns REL exclusively.
`mesh_rows_return` checks capacity in that existing ring before clearing a
received row. When capacity is absent, the actual page remains in the row;
the scan continues over other rows. When capacity exists, zeroing precedes
publishing the descriptor to REL. The bridge only advances the tail, so capacity
cannot disappear between the check and publication by the sole producer.
No blocking retry, pending list, or separate completion state is involved.
Local pages become absent only after zeroing; their fixed physical addresses
remain in the configured output maps for reuse. This does not yet provide
physical sharing between distinct configured outputs, remote dependent-read
proofs, or the worker integration needed by the caller.

Delivery scans one snapshot of the existing CMP ring and publishes every page
whose destination is available, returning the number published. It removes each
descriptor using the existing ring operation. A busy destination
does not prevent delivery to other available rows. No page or descriptor is moved
to a second pending queue. The single receive owner is required by that ring
operation. Error propagation, remote read proofs and recovery remain integration
obligations. These primitives are not yet used by the NFE.

`mesh_rows_progress` combines NIC-completion release, the receive snapshot,
publication-driven transmission and use-count retirement for the asynchronous
transport owner. Its arguments are the existing page table, fixed epoch and
immutable bindings. It owns no additional persistent state and invokes no model
kernels. Its unsigned return now counts completed page operations; it is not an
operand-readiness signal or an error return. Hardware errors require the separate
asynchronous metadata binding described above. Papadopoulos–Culler supply the operand
presence/lifetime principle; Rabenseifner/Patarasuk–Yuan supply the page-exchange
algebra. These references do not prove a latency bound for this implementation.

Snapshot erasure is safe with the existing single-consumer ring operation:
removing position `at` moves the old tail descriptor to `at` and advances the
tail. That moved descriptor was already examined in this pass. Every position
after `at` through the captured head remains unexamined and unchanged by erasure.
The producer may reuse released positions, but those lie outside the remaining
snapshot. Busy descriptors remain in the same ring for the next pass. No
persistent cursor, pending list, or caller-owned arrival bitmap is required.

### Literal page checking

Saltzer, Reed and Clark (1984), cited above, motivate endpoint checking.
`mesh_rows_digest` writes the digest of a live page directly into a configured
digest page; it retains no heap hash ring, count or generation cache. The two
hardware CRC polynomials are an implementation choice for error detection, not
a cryptographic claim or an algorithm attributed to those authors. Corresponding
peers must use the same configured seed and numerical payload extent.
`mesh_rows_equal` compares two present, equally stamped pages as a value. The
numerical graph must not depend on its result. A configured post-consumption
function owns the hash read, its output publication and input-use release;
comparison is an endpoint function whose disagreement concludes the NFE with
failure. These primitives alone do not implement replay or link recovery.

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

### FP32 contraction output specialization

`MatrixOperations.contract` requires FP32 destinations, but the initial vector
and tensor bindings rejected that type and their shaders always stored FP16.
That implementation gap is corrected by specializing the output element type
when the numerical binding is realized. The pipeline cache key includes the
output-type specialization. The invocation does not inspect output types or
choose a backend.

The existing vector and tensor kernels now write their FP32 contraction
accumulators directly into FP32 destinations. Ungated contractions do not round
the partial result through FP16. The existing fused gate/up expression retains
its specified FP16 rounding of the two projections before activation; changing
the destination type does not change that nonlinear expression. Ordinary FP16
outputs retain their previous conversion. This adds no scratch allocation,
host accumulation chain, or alternate kernel implementation.

Papadopoulos–Culler (ISCA 1990), cited above, provide configured functions over
assigned output storage. Rabenseifner (ICCS 2004) and Patarasuk–Yuan (JPDC 2009),
also cited above, provide the collective composition into which independent
contraction partials feed. The output precision and Metal specialization are
implementation choices here, not claims made by those publications. This change
does not prove that partitioning every contraction is faster than a full local
contraction, nor validate mixed-precision MPS multiplication or caller migration.

At caller commit `3f6f8e1` and mesh commit `2d91772`, both full client builds
passed. Direct Metal compilation passed for 20 variants per SoC (M5 Max and M4
Pro), including both output types and fused modes, plus fixed/dynamic tensor
reduction and both tensor orientations. The existing two-layer FP16 RDMA caller
reported 24 agreements per peer, no disagreements/retries/nonfinite values, and
byte-identical peer logits matching the earlier committed regression. These are
FP16 regression and FP32 compilation evidence, not numerical validation of the
FP32 partial path. The exact manifest and results are committed in
`metal-microbench/docs/data/fp32_output_rdma_2026-09-09.json`.

### Hardware error provenance still lost at the bridge boundary

At mesh commit `af350bd`, `rdma/mesh-links.h` puts literal failure codes into
`link_event.error`, including verbs completion status, post-operation returns
and socket errors. `rdma/mesh-flow.c` handles `L_FAULT` by incrementing `bad` and
clearing `link->up`; it does not preserve that literal code in a caller-visible
metadata page. The old `mesh-pages.c` subsequently infers failure from link and
bridge state. Such inference cannot recover the original code or its precise
operation provenance.

The required binding must preserve the error's code domain and operation
location/occurrence when reported by the device or bridge, then publish those
literal values into the configured metadata return. Verbs completion codes and
POSIX errno values cannot be silently treated as the same code domain. No new
inference, negative-status execution gate or internal recovery follows from
this reporting requirement. This bridge binding remains unfinished; neither
successful compilation nor removing receive validation provides it.

### Explicit FFN weight views

`MatrixOperations.bindFFN` owns the existing gate/up, activation and down-projection
composition over explicit `MatrixView` operands. The earlier implementation in
`DenseFFNWeights.bindActivation` is removed, and the dense normalized binding
now delegates to the same composition. `realizeMetalParameter` can receive
configured gate/up/down views instead of loading dense weight buffers. Supplied
views are in contraction orientation: hidden-by-width for gate/up and
width-by-hidden for down. They may describe actual page payloads within the
backend's supported layout; the binding does not copy them or select a backend
from their storage. Existing callers without supplied views keep their configured
weight loading and numerical behavior.

Papadopoulos–Culler (ISCA 1990), cited above, provide the explicit operand-storage
and configured-function precedent. Rabenseifner (ICCS 2004) and Patarasuk–Yuan
(JPDC 2009) provide the collective composition for the resulting partial values.
Those publications do not prescribe the FFN activation or its rounding. This
change preserves the existing fused and unfused expressions and selected
matrix backends, resolves their choices at binding time, and adds no invocation
allocation or synchronization.

The page-backed caller must supply activation/scratch storage as well as weight
views, and must include all of them in its complete graph allocation. The general
binding still supports allocation during realization when ordinary callers omit
scratch views. Full paged-K MPS interoperability still requires independent
payload-contained contractions and FP32 partial reduction; passing a spanning
paged view to MPS is not enabled by this refactoring. Attention/CoreML weight
interop and the mesh caller migration remain unfinished.

At caller commit `9c108d9` and mesh commit `fff31f3`, full builds passed on both
machines. The existing two-layer RDMA evaluation exercised the shared FFN
composition on M5 with ordinary dense weights; M4 retained its configured CoreML
FFN. Both peers reported 24 agreements, zero disagreements/retries/nonfinite
values, and logits byte-identical to the prior regression. Explicit page-backed
weight arguments and FP32 partial contractions were not exercised. The manifest
and metrics are committed in
`metal-microbench/docs/data/ffn_weight_views_rdma_2026-09-09.json`.

### Parameter output precision evaluation

The existing `runParameterBench` accepts `LM_BENCH_OUTPUT_ELEMENT=float32` for
local Metal, tensor and MPS parameter functions without the FP16 post-FFN
normalization. The default remains FP16. Selection, output allocation, matrix
views and binding occur before numerical invocation. The existing output dump,
finite-value scan and outside-timing FFN reference comparison interpret the
selected type. `scanNumericBuf` replaces the FP16-only scan under one shared
implementation; existing callers keep their default FP16 interpretation.

Saltzer–Reed–Clark (TOCS 1984), cited above, supply the endpoint verification
principle. This is an extension to the existing client evaluation, not a second
evaluator or a numerical dependency. The reference remains the existing MPS FFN
with FP16 output, so differences include output rounding; it is not an FP64
oracle. This option does not change the active RDMA caller's FP16 return layout,
exercise page-backed weights, or establish complete transport integration.

At caller commit `9463f05` and mesh commit `52bc871`, both full builds passed.
The existing evaluator ran twelve local FFN cases: MPS/vector/tensor, FP16/FP32
output, on M5 Max and M4 Pro, with 32 rows and 128 owned neurons. All completed
with finite output. Mixed FP16-operand/FP32-output MPS executed successfully on
both devices for this shape, resolving the earlier compilation-only uncertainty.
M5 tensor FP32 output exactly matched MPS FP32 output; M4 tensor relative RMS
error against MPS FP32 was approximately `3.17e-5`. Vector FP32 relative RMS was
approximately `2.59e-5` on M5 and `1.03e-5` on M4. Each FP32 result retained more
than 122,800 values not representable in FP16, out of 122,880 values, confirming
that the result is not merely FP16 output widened afterward.

The exact configurations, output hashes, timing samples and comparisons are in
`metal-microbench/docs/data/parameter_fp32_outputs_2026-09-09.json`. These are
small-shape dense numerical evaluations, not page-backed-weight or K-partial
validation, performance superiority, RDMA integration, or a full-NFE bound.

### Configured contraction ranges

`realizeMetalParameter` accepts an explicit reduction-coordinate range for a
projection or vocabulary function. It binds `MatrixOperations.contract` over
that range and the requested output coordinates, producing independent FP32
partial outputs. The range and all slices are resolved during realization;
numerical invocation performs the already-bound operation. There is no partial
accumulation loop, backend substitution or allocation inside invocation.

Papadopoulos–Culler (ISCA 1990), cited above, provide the configured operand and
function representation. Rabenseifner (ICCS 2004) and Patarasuk–Yuan (JPDC 2009)
provide the reduction/scatter/gather composition to which disjoint contraction
partials contribute. For a partition of K into disjoint ranges R, the exact
algebra is `C = sum_R A[:, R] B[R, :]`; floating-point association differences
must be measured rather than called bitwise equivalence.

The existing parameter evaluator accepts `LM_BENCH_REDUCTION_RANGE=first:end`
with FP32 output for those function kinds and records the resolved range. Its
FLOP count and estimated weight-read bandwidth reflect the selected range;
`weight_bytes` remains allocated weight storage and `weight_read_bytes` records
the range's weight operands. The evaluator uses the existing projection and dump paths, not
a second implementation. These local dense evaluations do not establish actual
page-backed operands, concurrent partial issue, transport integration or a
performance gain from splitting K.

At caller commit `8a01256` and mesh commit `72e8f26`, both full builds passed.
The existing client completed 30 finite projection evaluations across MPS,
tensor and vector backends on both SoCs. Each used 32 rows, 128 outputs and
K=3840, with a full contraction and partitions at K=1920 and K=1800. Input
hashes matched across cases on each device. Adding each pair of partial dumps
with FP32 rounding reconstructed the corresponding full backend output with
relative RMS error below `1e-6` in all twelve comparisons; the largest was
approximately `9.61e-7` for M4 tensor at K=1800. The latter partition exercises
partial reduction tiles. This is floating-point agreement, not exact equality.

The exact records and comparison method are committed in
`metal-microbench/docs/data/contraction_ranges_2026-09-09.json`. The summation
was measurement analysis outside the callgraph, not a replacement mesh reducer.
Actual page operands, concurrent issue and canonical page reduction remain
separate integration obligations; separate-run timings do not prove overlap.

### Complete page ownership

The connected replacement uses Papadopoulos and Culler's Monsoon operand
presence and use accounting, Rabenseifner and Patarasuk–Yuan's partitioned
page exchange, and Saltzer–Reed–Clark's endpoint interpretation of error values.
The API mechanisms below are this repository's realization of those principles,
not a claim that any paper specifies the C ABI.

The transport layout separates each page's address/error metadata from its
entire OS-page-aligned numerical payload. Both regions belong to the registered
shared mapping. A provider request gathers its header and payload directly;
receive scatters directly to the corresponding canonical header and payload.
Consequently consecutive payload pages form a contiguous native view without
removing embedded headers, padding copies, or a second numerical allocation.
The provider's two-SGE capability still requires validation on both actual
participants after the complete source change; no execution accompanied this
source pass.

`mesh_rows_create` installs a stable table identity during configuration and
allocates the row table from canonical pages. `mesh_rows_allocate` reserves
contiguous physical pages before invocation; `mesh_rows_map` binds exact-stamp
source values. The context's configured table registry preserves original
ownership for late physical completions. Identities are positional within each
participant. Every exchange binding explicitly names the remote table; its
immutable transmit record separately names the original local owner. The wire target is the literal remote binding array index; its index addresses
the page within that receiver binding, rather than assuming
that heterogeneous participants assign identical logical row numbers. Neither
local nor remote identities are reused while their original tables remain live. `mesh_rows_invalidate` removes the outer
caller's handle without changing the retained table or redirecting device
access. This protects replacement allocations from old accesses; it does not
claim device cancellation or immediate reclamation of a hung device's backing.
The present allocator is monotonic within its configured context; physical
lifetime reuse and final retirement remain obligations of the connected owner.

`mesh_rows_issue` stores the complete selected-index set in one or more
contiguous canonical pages. The first uint32 value is the count; the remaining
values are the selected function indices. Each configured reservation is sized
for all function rows, and worst-case concurrent reservations cover one issue
per row. `mesh_rows_complete` reads that same retained value, publishes exactly
its outputs, releases their configured input reads, then releases and zeros
that reservation. A later scan never overwrites the indices held by an earlier
completion. Configuration must provide the entire reservation; there is no
runtime allocation or smaller dispatch-capacity substitute.

Immutable source output maps explicitly permit shared physical backing. Such a
source has no numerical inputs and performs no write to its parameter values;
it publishes the invocation's stamp into a distinct logical row. Each logical
alias retains one caller use for the configured context's lifetime. Once its
numerical uses end, the same logical row can be stamped for a later invocation
without zeroing the immutable backing. Mutable output maps continue to require
distinct physical spans. Removing that restriction requires a complete static
lifetime proof and corresponding retirement ownership, not a boolean exception.

`mesh_rows_poll` consumes literal provider completion descriptors. Addressed
receives publish the original addressed table's rows and consume the configured
remote reads. NIC completion releases its source use regardless of the literal
error code; errors are preserved separately in registered metadata. A failed
receive has no trustworthy graph address, so its provenance names the physical
receive page and port rather than guessing a table from unwritten payload.
Only the outer calling context interprets these metadata values. The provider
completion rings describe physical work ownership; they do not contain another
queue of numerical functions, retry protocol, or recovery decision.

The final configured maps carry immutable `mesh_row_range` index values and
per-output read multiplicities in canonical pages. Realization generates affine
ranges or accepts already realized input ranges for padded native packs. The
numerical scan always gathers the configured range; it does not infer a layout
from an operand or allocate another index representation. Configuration derives
read multiplicities by interval differences and a prefix sum, including every
input occurrence, NIC read, remote read proof and caller return. Remaining uses
still occupy only the original three-field row. These index/count values are
static function configuration, not a mutable copy of readiness.

Each transmit occurrence has its own 128-byte record in canonical pages, with
an immutable 64-byte wire/address/error header and physical provider ownership
fields. Concurrent destinations never overwrite a header still read by the NIC.
The physical request's descriptor names that record directly; completion uses
the original local owner stored there. This permits fanout of the same payload
without duplicating its numerical storage or serializing sends behind one mutable
header. Separate records and the physical page itself remain retained until
actual NIC completion.

Retirement follows completed selection lists, original physical completions,
and configured returned-page maps as detailed below.
The existing physical REL operation names the released page and its original
canonical row offset. The bridge zeros the payload asynchronously, removes the
page from that row, and clears the retirement bit while retaining the old stamp.
It handles one such release alongside its normal physical completions and
submissions, rather than draining a bulk erase before communication. Receive
pages return to the receive pool; arena pages retain their configured address.
There is no extra acknowledgement or numerical readiness object. Selected index reservations are consumed by the scan owner and erased by the
same asynchronous physical REL operation. These are source semantics; their elapsed costs have not been measured.

`mesh_metal_rows_source` is the single GPU page-reduction implementation. It
gathers each participant's configured padded contraction layout and writes FP32
accumulator pages. Model-specific embedding and normalization remain numerical
functions in the caller. The former independent CPU add/normalize/index-query
implementation and unused count-query public APIs were removed with their callers.

The generic Python/game ingress binding now reserves canonical input pages,
encodes application records into those pages, and receives local socket payloads
directly into their registered destination views. It publishes those page values
through the same row transport. Application game-record decoding and retention
are outside the numerical graph; their historical event/snapshot semantics are
not erased to claim a numerical runtime reduction. The transport no longer
allocates copied send queues, reattaches itself, or substitutes another page
packing after a binding changes. Application configuration supplies the actual
peer set and matching ingress capacity before publication.

Remaining source obligations are explicit: the monotonic physical allocator does
not yet realize static lifetime reuse or final owner destruction; invalidation
retains old storage rather than revoking a hung device; an unaddressable failed
receive retains physical error provenance without guessing a graph destination.
The caller must expose that metadata and its ownership. Complete native parameter
and intermediate binding, all runtime
validation belong to the connected acceptance review, not these source edits.

Configuration classifies each input map's explicit ranges once: input `stride`
is zero exactly when every occurrence reads the same range, and one otherwise.
Runtime addressing uses only the realized ranges. Dennis's operand matching
therefore checks a shared range once per scan; Papadopoulos–Culler's read
accounting subtracts the completed occurrence count once per shared input page.
No readiness cache is introduced. Index values are released by completion and
retired through the same asynchronous physical REL operation by the scan owner.

`mesh_metal_regions` realizes bounded 1 GiB views and a canonical page containing
their GPU addresses; `mesh_metal_page_span` provides the exact bounded operand
view used consistently within each composed numerical function. Numerical views
retain Metal hazard tracking. Indirect region views belong only to single-encoder
page functions; their inter-function dependencies are literal completion stamps.

`mesh_metal_address_source` supplies the shared GPU physical-page address
arithmetic to both page functions and configured external tensor functions;
there is one region ABI and no parallel address decoder in generated models.

## Removed endpoint CRC feature

The operator explicitly removed endpoint CRC/digest comparison from the active
algorithm set. Earlier CRC/checking entries in this document are historical
implementation records, not current requirements. No digest kernels, CRC tables,
comparison functions, digest transport bindings, or CRC-generated metadata remain
in the canonical numerical path. Literal device/transport status metadata remains.

## Exact-range physical retirement

Papadopoulos–Culler read counts and the configured Dennis input/output maps
identify retirement without a second completion queue. Each static function
contains its already allocated selection-index map. Polling visits all completed
selection pages independently of new numerical readiness, retires precisely their
selected input/output ranges, consumes the list count, then releases its backing
pages from last to first. The first page remains the literal list owner until
all tail pages have entered asynchronous REL, even when REL is full.

A received page retains its original immutable address header. Before retiring
that page, its configured remote-read input ranges are retired; this follows the
feedforward page dependencies and retains the received page when physical REL
capacity is unavailable. Original NIC completion entries remain in the physical
CQ until their ownership release is submitted. Returned numerical and raw
metadata maps are copied into configuration pages and are the only endpoint
ranges polled for caller consumption. No full mutable-output or receive-table
retirement sweep remains. This source change has not been executed or measured.

## Context lifetime

Papadopoulos–Culler operand identities and actual read counts also govern outer
lifecycle operations. `mesh_rows_invalidate(&owner, held, count)` consumes the
caller capability and transfers its still-held return leases to the context.
Returned ranges are configured disjoint, with one caller lease per logical row;
multiple application references share that lease. Invalidation copies the held
subset into the existing canonical return-map storage and orders its intervals.
The caller must supply that subset, not infer ownership from aggregate use counts.

For each input occurrence, unequal nonzero input and consumer-output stamps
identify a configured read that has not issued for that value. Invalidation
cancels that multiplicity, unsent NIC reads and unrealized remote-read proofs.
Issued selection pages retain actual device ownership and original completion
records retain NIC ownership. Terminal callbacks publish the actual old pages;
metadata reporting only writes its already-claimed output bytes, so it cannot
restore canceled uses. The numerical invocation gains no cancellation guard.
Nested function and binding maps are copied into canonical configuration pages;
retirement does not borrow their caller's heap lifetime.

`mesh_context_metadata` exposes the original registered header of a failed or
unbound physical receive, without guessing a numerical destination.
`mesh_context_consume` transfers that page to asynchronous REL. Its canonical
physical row occupies 16 bytes of the default registered header record's unused
padding, outside the numerical payload; neither receiving nor zeroing payload
bytes can erase that ownership record.

`mesh_detach` is a nonblocking outer lifecycle operation. EBUSY reports pending
physical ownership or an unconsumed context metadata lease; it is not a graph
admission result or a retry of computation. The caller invalidates every owner
first and drives physical polling/metadata consumption outside numerical calls.
After actual selections and NIC reads end, held received pages and each allocated
arena page transfer exactly once to context-owned physical rows. Asynchronous
REL zeroes those payloads, including the old logical tables and configuration
maps, before detach clears the shared client claim and unmaps. Shared parameters
and region-address pages remain reserved until that whole-context transition.
Binary interval lookup avoids checking every held map for every output page.

Per-table interior arena reuse is not implemented. A hung uncancelled device
keeps its physical ownership; logical invalidation cannot revoke hardware memory
access. A new process/context in the demo uses an explicit physical bridge/QP
restart at configuration boundaries, so old table identity zero cannot address
new table identity zero. No numerical handshake or recovery protocol is added.
This lifecycle source has not yet been executed or measured.

Explicit ranges supplied from foreign configuration memory are copied into
canonical pages during realization; already canonical ranges remain views of the
same storage. Invalidation matches held ranges to the existing canonical return
descriptors and retains those descriptors, never the argument's pointers. Invalid
or duplicated held ranges return EINVAL to the outer lifecycle consumer before
any ownership changes. Successful invalidation returns zero.

## Link port metadata

`mesh_link_metadata(context, index)` exposes the existing physical port's literal
`when`, `code`, `domain` and `peer` fields through the outer metadata channel.
Its index is the configured port index, function is `UINT32_MAX`, and stamp is
zero because a port observation has no numerical function or generation.
`mesh_peers(context, NULL, 0)` returns the number of configured ports, including
multiple ports to the same peer; the calling context uses that bound.

This follows Saltzer–Reed–Clark's separation of endpoint interpretation from
transport mechanisms, already cited above. The accessor allocates no storage,
clears no status and makes no error or readiness decision. The outer consumer
may interpret the returned values; the numerical call graph does not read them.
These are observations of existing port fields, not a new atomic event record,
acknowledgement, progress counter or coherent multi-field snapshot protocol.
