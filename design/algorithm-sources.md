# Algorithm sources

## Independent verbs progress

Apple's [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
defines nonblocking `ibv_post_send`, `ibv_post_recv`, and polling of their
completion queue. The controller supplies receive credits; application receipts
are not required to submit another independent operation. Work-completion status
reports transport errors to the calling context's metadata.

`mesh-flow.c` gives each direction its own queue-capacity accounting. Each
submission is one literal WR. A failed receive post returns its unowned backing
to the free index; a failed send post releases the source's table occupancy and
records the error. Neither direction retains a pending batch that gates the other.
Every progress pass services both directions and completions with bounded work.
Normal termination enters verbs teardown without a software loop waiting for a
peer to consume committed sends. Driver destruction remains the operation that
releases the device resources.

The removed implementation confused a prepared host descriptor with an occupied
hardware queue and shared that condition across both directions. Its retry flag
could consequently stop receives behind a failed send, or stop sends behind a
failed receive. The descriptor is needed only during the literal post; accepted
work is represented by the device queue and its page-table indices.

## Operand matching and storage

Gregory Papadopoulos and David Culler, *Monsoon: an Explicit Token-Store
Architecture*, ISCA 1990, describe operand matching using indexed storage and
presence state. This is prior art for the function scan, not evidence of this
implementation's measured utilization. Logical row occupancy and physical page
occupancy describe different indexed resources. Reattachment cannot erase
physical ownership held by an outstanding device access.

Rolf Rabenseifner, *Optimization of Collective Reduction Operations*, ICCS 2004,
and Pitch Patarasuk and Xin Yuan, *Bandwidth Optimal All-reduce Algorithms for
Clusters of Workstations*, JPDC 2009, supply the reduction decomposition. They
do not require independently completed partials to wait for a host-side batch
receipt before becoming inputs to reduction.

## Receive storage before consumer binding

Apple TN3205 defines receive completion as completion of access to the posted
registered memory. Papadopoulos and Culler's indexed operand storage supplies
the separate local association of that memory with a numerical input. A missing
consumer binding cannot undo an already completed receive.

The receive pool uses the existing physical PAGE_OWN and PAGE_HOT planes.
Posting establishes both bits; completion clears HOT, records the physical
landed bit and initializes its inverse logical row to ABSENT. The bridge's
existing landed-block pass associates that block when its local binding exists.
Until then it retains the original pages, without a payload copy, extra queue,
sender message or readiness handshake. Once associated, normal reader masks
govern consumption. Association precedes release of retired ROW_BOUND bits so
a binding captured during consumer detach cannot address reallocated rows.

`mesh_receive_invalidate` clears receive-pool PAGE_OWN bits. It takes effect through subsequent bridge progress, not synchronous cancellation.
It neither waits nor modifies FREE, landed bits or inverse mappings. The bridge alone recycles an
invalidated block after its receive completes; completion does not restore its
ownership. Invalidation of associated blocks also removes their logical mapping.
Ordinary detach and dead-client takeover retire numerical rows and arena ownership,
preserving unassociated physical arrivals. Associated abandoned rows still follow
normal reclamation. No new table allocation or per-arrival structure is needed.

Invalidation covers current physical ownership intervals, including posted
receives, not arbitrarily future traffic using the same binding and index.
It is not remote-call cancellation. The calling context owns configured storage
and explicit destruction; replacing the bridge region destroys the whole receive
table through normal driver teardown. This API adds no transport recovery.

## Performance evidence

The acceptance target is less than five percent pipeline stall during the
linear-algebra-intensive interval on real inputs. Timing belongs to the calling
measurement context. A host interval with an outstanding command does not prove
that an accelerator was executing arithmetic. Metal execution timestamps and
host pending intervals must be reported separately; Core ML's opaque execution
must not be counted as measured hardware occupancy. Neither a timing simulation
nor command occupancy alone demonstrates mathematical FLOP utilization.

## Nonblocking table ownership

The indexed operand storage of Papadopoulos and Culler and the literal access
completion defined by TN3205 have different indices: logical rows belong to
functions, whereas backing pages belong to memory accesses. Configuration allocates
from the complement of assigned and outstanding indices. Submission occupies
both the physical source and its logical completion target; completion releases
those indices. Attach does not drain another caller's sends or erase its targets.

Issuing a function marks its output rows as being produced. Presence becomes true
at completion, and reader bits become consumed then. Clearing presence alone was
insufficient: the next scan could otherwise issue the unfinished operation again.

Only the bridge returns landed blocks to its free index. Numerical completion
sets reader bits; the bridge checks the configured reader mask through the inverse
landing-page index. This removes the competing return operations and the FREED
bit that was cleared before another reader had finished examining it.

Receive bindings retain their logical indices through ROW_BOUND. Detach removes
the binding and its assigned indices, but the bridge releases ROW_BOUND after its
current completion pass. Thus a previously captured binding address cannot become
another allocation halfway through that pass. These are bounded page-table masks;
no callback waits, reference-count drain, acknowledgement, or generation check is
needed. Binding addresses become visible only after their reader masks are set.

## Registered-span defect, closed

The provider registered aligned one-GiB address extents while a multi-page
block selected its key from its first byte, so a block could cross a
registration boundary; and a registration must not cross a 4 GiB
virtual-address boundary, the bank alias recorded in
[RDMA-KERNEL-RECOVERY.md](RDMA-KERNEL-RECOVERY.md). Both hold now by
construction: the bridge maps the region at a 4 GiB-aligned base and the data
origin is a multiple of the block; a power-of-two block gets 1 GiB regions from
the base, which divide the bank and which no block straddles; any other block
gets block-aligned regions from the data origin and the bridge refuses, saying
why, a mapping that reaches a bank boundary. The resolution is measured in the
next section.

## Regions follow blocks

A block is posted as one scatter-gather element whose key comes from its first
byte. Memory regions are therefore cut at block-aligned offsets from the data
origin — region 0 is the header, every data region spans a multiple of the
block — so a block is inside one region by construction. Regions cut at
absolute address boundaries put the tail of a straddling block outside its
key's region; the tail is where the tag lives, so the landing arrived with a
clean completion and no tag, and the bridge rejected it. Measured 2026-09-10
on the M5 (two regions: every landing from the M4 rejected) against the M4 (one
region: every landing accepted); with block-aligned regions both directions
carry every block (160 sent, 160 received, 0 bad on each side). A rejected
landing prints page, status, bytes, tag and base to the bridge log.

## Attach takes over a dead holder

The client word names the process that owns the region. A process that died
without detaching left its pid there and every later attach was refused: a
node that looked provisioned and was not. Attach now takes the word from a
holder that `kill(pid, 0)` reports gone and performs the release the dead
client never did — bases absent, row and page ownership cleared. A live holder
is still refused; the check prevents a demotion and never is one.


## Registered memory views

Apple's mmap MAP_SHARED mapping mechanism permits multiple virtual views of the
same shared-memory file pages. mesh_view_create reserves virtual address space
and maps configured physical-page runs from the existing mesh shared-memory fd
into it; mesh_view_destroy releases that view. No payload is copied and no new
registered memory is allocated. This is configuration-only address realization,
not a receiving-side remapping operation or an invocation-time allocator.

The two installed Thunderbolt providers report max_sge=1 through ibv_devinfo.
The view therefore joins numerical payload pages in virtual memory while the
bridge continues to submit their original registered addresses. Logical indices,
dense tensor offsets, and registered addresses remain separate representations
of the same underlying bytes.


## Independent configured programs

Papadopoulos and Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA
1990, separate indexed operand ownership from the functions that use the
operands. Each configured program here owns disjoint logical rows, backing
allocations and receive-binding indices in the existing table. Realizing a
second program begins with the table's existing reader masks, preserving the
readers already assigned to the first program. Newly allocated rows have zero
reader masks through the existing allocator.

`mesh_rows_release` clears only its logical ownership range and marks changed
receive-bound words for the bridge's existing retirement pass.
`mesh_arena_release` clears only its physical ownership range; transport HOT
indices remain occupied until intrinsic completion. `mesh_bindings_release`
removes only the configured binding indices. None detaches another program,
waits for a peer, or clears another program's reader registration.

The Swift caller shares one region storage object per configured region name.
Its weak registry does not retain unused regions. Configured values and graph
completion closures retain that object through their existing lexical owners;
only its final destruction detaches and unmaps the client region. Closing a
calling context drops that context's cached values, without detaching surviving
programs. Numerical callbacks retain their graph, so graph-owned values cannot
be retired before their final callback completes. This is storage ownership,
not a new execution readiness condition or a polling mechanism.

Binding ranges use the configured identity mapping described below. Slot reuse
retains the identity in the existing receive table, so local storage retirement
cannot cause an earlier transfer to address another program.

## Literal weight pages

Papadopoulos and Culler, *Monsoon: an Explicit Token-Store Architecture* (1990),
provide the indexed operand-store model used here. Configuration resolves the
parameter values and their ownership before numerical invocation. The port's
strided f16/f32 parameter reader materializes each declared shard into its actual
registered operand pages during configuration. Invocation neither consults the
original descriptor nor substitutes a file-backed parameter for a supplied one.
Apple's shared mapping mechanism described under registered memory views supplies
contiguous virtual tensor views over those same backing pages; it does not create
a second payload store. This citation identifies the storage and execution
separation, not a claim that Monsoon specifies today's tensor ABI or weight format.

## Configured binding identities

Papadopoulos and Culler, *Monsoon: an Explicit Token-Store Architecture*, ISCA
1990, supply the indexed operand-store principle. The configured binding identity
names an address interpretation in that store; it is not an arithmetic readiness
stamp, receipt, or peer progress counter. Apple TN3205 still supplies the same
literal SEND/RECV operations and completion semantics.

ABI 18 keeps the existing 16-byte transfer tag and its 32-bit binding field.
That field now names the complete configured identity. The fixed receive table
slot is `identity % 4096`; one atomic 64-bit table entry holds the identity and
logical base together. A reserved entry has no base until configuration realizes
its reader masks and receive mapping. Retirement removes that base while retaining
the identity. Reusing a slot requires a strictly greater identity and an unowned
slot. These are configuration address allocations, not per-invocation checks or
transport messages.

A receive whose identity equals the slot's active identity maps to its logical
base. A newer identity, an untouched slot, or the matching reservation retains
the existing early-arrival behavior until configuration supplies the base. An
older identity, or the matching retired identity, has no destination and its
landing pages return through the existing free index. Thus a late transfer cannot
become an operand of a new program that happens to reuse the same table slot.
The incoming payload is never copied or interpreted by this address resolution.

`mesh_bindings_reserve` accepts an explicit first identity, or `UINT32_MAX` for
the existing ordered-configuration allocator. Explicit identities allow different
handles in different table slots to compile in different orders on the peers;
the caller supplies the same identity and operation/version ordering to both.
Colliding live slots and non-increasing reuse are reported as compile errors.
Implicit allocation uses a high-water value in the shared region header, which
survives client detach/reattach and only resets when the bridge constructs a new
region and connection. Implicit callers must have matching configuration history
on both participants; independently ordered callers use explicit identities.

The 4096-entry table limits simultaneous occupied slots, not cumulative handle
creation. Retired slots are reusable with newer identities. The 32-bit identity
space never wraps: exhaustion reports compile metadata and requires a new
configured connection. An explicit identity range must not include UINT32_MAX.
Unused reserved slots and failed configuration reservations are retired. No
transport acknowledgement, handshake, wait, recovery message or payload side
channel was introduced.

## Column and row tensor composition

Shoeybi, Patwary, Puri, LeGresley, Casper and Catanzaro,
[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
(2019), describe intra-layer tensor parallelism. The port uses its complementary
matrix partitions: participant p computes `Z_p = X A_p` from its output-column
slice of A, then `Y_p = Z_p B_p` from the matching input-row slice of B.
Only `sum_p Y_p` requires distributed reduction. A column result is a local
numerical value; no all-gather is inserted between these matching contractions.
Papadopoulos and Culler's indexed operand matching supplies the execution rule
for these configured matrix functions. The existing local MatrixOperations
implementation and specialization remain the arithmetic owner.

FP32 interface values are explicitly converted at numerical boundaries when the
configured model uses FP16 operands. Conversion is an indexed elementwise function
over actual mesh pages, with one dependency extent per configured output extent.
This supports FP32 input/output representation; it does not claim an FP32 model
or replace the configured backend precision. No shadow copy or runtime backend
selection realizes the conversion.
