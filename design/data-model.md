# The data model

These sections say what the hardware is. Each number names its source. The files measurement.md and facts.md and the
folder output_data/ are in the metal-microbench repo. The papers are listed at the end.

The contract is one-sided, ahead-of-time addressed writes with the least attainable whole-function latency and
greatest useful bandwidth. No implementation interface may restrict the physical routes, parallel cables, placements,
or arithmetic decompositions available to a different solver. Remove protocol waits, acknowledgments, unnecessary
copies and guards by proving their invariants at realization. Necessary data availability and memory ownership remain explicit
dependencies; a source-level count of waits is not a performance proof. Section 6 distinguishes these cases.

## 1. The M-series accelerator

An M-series chip's CPU, GPU, Neural Engine and Thunderbolt RDMA engines share one DRAM. There is no device memory
(design/hardware-and-discipline.md). A Studio's M5 Ultra is two dies with 64 GPU cores (the chain's 64 threadgroups, one
a core, output_data/chain-20261022/t134-studios-rank0.log). Each core has one large on-chip cache for its registers,
threadgroup memory, stack and buffer data. The core hands it out as a shader runs. If a shader needs more than fits, the
core runs fewer SIMD groups to keep the data on chip (Apple tech talks 111375 and 111431). Past that cache come the GPU
L2, a system-level cache shared with the CPU and the RDMA engine, and DRAM (design/hardware-and-discipline.md). On an
M2 Pro's GPU, the system-level cache answers in 234 ns and DRAM in 342 ns or more (Chips and Cheese 2023).

An H100 has 132 SMs. Each SM has a 256 KB register file, then 256 KB of L1 and shared memory. The SMs share a split
50 MB L2, then 80 GB of HBM3. Host DRAM is past PCIe Gen 5, at 64 GB/s each way (NVIDIA 2022). So sm90 has one level
more than an Apple core, whose registers live in its cache. On a Hopper H800 a load takes 29 cycles from shared memory
and 32 from L1. It takes 258 from the near L2, 414 from the far L2, and 656 from HBM. At 1,755 MHz that is about 17,
18, 147, 236 and 374 ns (Luo et al. 2025).

sm90's cooperative warpgroup is four warps that run one matrix multiply out of shared memory (NVIDIA, CUTLASS). A copy
engine, the TMA, fills shared memory with tiles from HBM through the L2 (NVIDIA 2022). That tile path is where sm90
congests. 132 SMs share the split L2, and a far hit costs 1.6 times a near one (Luo et al. 2025).

The M-series analogue is a TensorOps matrix multiply that several SIMD groups of one threadgroup run together. Its
result stays in their registers. On the M5 it runs on a neural accelerator in each GPU core (Apple tech talk 111432).
There is no tile copy engine and no cluster. No threadgroup reads another's threadgroup memory. Threadgroups trade data
through device memory. Within one dispatch they see each other's writes only in coherent(device) memory. Plain memory
works on one die and fails across the Ultra's two (output_data/chain-20261022/RESULTS.md). So there is no tile path to
congest. The pressure is inside the core, where all data shares one cache.

The floors measured here are these. A GPU kernel that fences before each poll sees a store from outside the GPU in
0.58 µs. A dispatch in an open command buffer costs 0.9-1.4 µs. A new command buffer costs 13.7-18.7 µs
(measurement.md:414, 3233-3235). Apple gives the M5 Ultra 1.2 TB/s of memory bandwidth, for the Studios' 64-core
model and its 80-core model alike (Apple, Mac Studio specs). At width 16,384 the chain's kernel streams its weights at
about 620 GB/s, about half of that
(output_data/chain-20261022/RESULTS.md). An H800 reads HBM at 1,861.5 GB/s (Luo et al. 2025).

## 2. From one GPU's memory to another's

On sm90 with InfiniBand, bytes go from HBM over PCIe to the NIC and through the switch. Then they go over PCIe into the
far GPU's HBM, and a kernel there loads them into L1. Two hosts move 64 bytes CPU to CPU in 2.8 µs under one leaf
switch, and in 3.7 µs across leaves (Zhao et al. 2025). Between two H100s, a kernel-to-kernel ping-pong of 4 to 128
bytes takes 16.0 to 24.3 µs a round trip, by library and path (Hamidouche et al. 2025). The fastest is about 8 µs each
way. The NIC reaches GPU memory through a PCIe BAR. Each buffer is pinned for it in 64 KB pages, which can take
milliseconds. The GPU and the NIC need to share a PCIe root complex (NVIDIA, GPUDirect RDMA). NCCL's GPU-initiated path
registers each buffer on every rank (Hamidouche et al. 2025). A running kernel may see the NIC's writes stale, partial
or out of order until a CUDA sync call (NVIDIA, GPUDirect RDMA).

On the M-series there is no HBM, no BAR and no PCIe peer-to-peer path. The CPU, the GPU and the RDMA engine use one DRAM
(design/hardware-and-discipline.md). Each Thunderbolt controller sees memory through its own IOMMU (TN3205). The bridge
registers one region with each device once, when the port comes up. A Studio wires 82 of its 96 GiB this way, as 82
extents of 1 GiB, registered in 0.44-0.48 s on 2026-10-23 (etc/bridge/m5ultra-headless.toml; the bridge's log). An
86-extent region once registered in 0.165 s (design/HETEROGENEOUS-LINKS.md section 1).
Every page a program uses lies in that region, so nothing is pinned at a call. A SEND lands in the peer's DRAM about
3.6 µs after it is posted (measurement.md:3238). A GPU kernel that fences before each poll sees a store from outside
the GPU 0.58 µs after it is made (section 1). The GPU's plain stores reach the NIC only when their command buffer ends.
System-coherent stores reach it at once (design/hardware-and-discipline.md).

## 3. The Thunderbolt RDMA driver

The driver has SEND and RECV on unreliable-connected queue pairs. It has no remote write, no remote read and no atomics
(TN3205; design/HETEROGENEOUS-LINKS.md section 1). A SEND fills the oldest receive posted on its queue pair. Both must
span the same number of 4 KB credit frames. These credits do not require page-sized SGE payloads: the byte-extent
plan gives both sides equal exact lengths, and the driver credit charge is `ceil(length / 4096)`. The hardware holds the SEND in credit until that receive is posted, and it sends
no acknowledgement (TN3205; design/transport.md section 1). Each request names one contiguous registered range (max_sge 1); the range cannot cross its 1 GiB MR boundary. A device takes
100 registrations (max_mr 100). The driver keeps only an address's low 32 bits. So a registration across a 4 GiB
boundary lands its receives 4 GiB low (design/transport.md section 1). A device reports max_qp 3 on the Studios' macOS
27, and 11 on macOS 26 (design/HETEROGENEOUS-LINKS.md section 1). One queue pair carries one line of receives. There is
no shared receive queue.

So the receiver fixes each exact receive interval and its order before the payload comes. A program that picks addresses just in
time has two ways to send. It can ask first, which costs a round trip: 10.29 µs for 3 KB between hosts
(measurement.md:363). Or it can land the bytes in staging with a header, then copy them. That spends memory bandwidth
twice. A quiet controller is slow too. After 40-50 µs with nothing sent, a SEND took 6.2 µs to complete. On a link kept
busy it took 1.9 µs (measurement.md:3165-3177). A process that exits while it holds queue pairs can hang in the driver
until the node resets (README.md; facts.md).

On InfiniBand a reliable-connected queue pair also has RDMA WRITE, RDMA READ and atomics (IBTA). A sender writes any
registered address of its peer at any time. The peer posts nothing. With GPU-initiated networking, a GPU thread rings
the NIC's doorbell itself (Hamidouche et al. 2025). So programs on sm90 can pick addresses at run time. Programs here
name their pages and their order before the call. The bridge posts receives into the pages the receiver named.

## 4. What the chain and the LSTM cost

The chain and the LSTM are dependency chains. Each step needs the last step from every node. The chain takes 27.2-28.8
µs a step on two nodes, with 8 rows of width 1,024. That time includes its matrix work (measurement.md:3261-3266;
output_data/chain-20261022/t134-studios-rank0.log). A crossing takes about 3.6 µs from post to landing
(measurement.md:3238). A 16 KB all-reduce alone takes 10.4 µs (t134-studios-rank0.log). The two-layer LSTM of width
1,024 takes 84 µs a tick on three nodes (output_data/lstm-20261022/t134-n3-lstm-rank0.log).

On sm90 with InfiniBand, bytes take about 8 µs from one GPU's HBM to a kernel on another GPU. The load from HBM to L1
adds about 0.37 µs (sections 1 and 2). The chain's crossing, 3.6-10.4 µs, is in that same range. That holds when the
code fits the hardware. A 3 KB crossing takes 11.7 µs as plain dispatches in a fixed order. It takes 24.1 µs when each
completion runs a scheduling kernel, the operand is copied to scratch, and the waits read gate words
(measurement.md:3230-3238).

## 5. Three positions on moving memory

These three hold on CUDA clusters and on Apple silicon alike.

**1. Destination bindings and storage ownership are required.** The compiler knows each byte interval and its
backing allocation before execution. Logical page tables can express this memory model; transport descriptors
resolve it into exact byte spans. A span lands in storage whose prior uses are proved finished. Without a prior binding,
each move has to be asked for or described at run time. That costs a round trip or a header every time, and the cost
grows with every hop and every step.

**2. Copying memory to 20 other machines is faster than reading it from the same machine's NVMe, when the code is not
written wrong.** One preposted hop here lands a page in about 2-4 µs (sections 2 and 3). A random NVMe read takes tens
of µs to about 100 µs before the first byte can be used. One Thunderbolt 5 cable moves about 10 GB/s, as much as or more
than a Mini's own SSD. On an InfiniBand fat tree a remote node is one NIC hop and a few switch hops away, so remote
memory beats local NVMe there too (Infiniswap, NSDI 2017). On the Mac mesh each node is itself the switch. So each hop
adds its own few µs, and 20 hops are still faster than one random NVMe read.

**3. One-sided writes are faster than two-sided writes that run a protocol at run time.** A one-sided write needs 1. It
puts bytes in the interval the receiver bound before the call, and nobody answers.
Without a prior destination binding, an extra address-selection exchange would be needed. InfiniBand's RDMA WRITE is the same: the sender
must hold the receiver's address and key before it writes (IBTA). The Thunderbolt driver only has SEND and RECV. But
when every receive is posted ahead, in an order both ends fixed, a SEND lands like a one-sided write. Nothing on the
receiving side decides anything at run time. That holds only while every hop stays preposted and one-way, whatever else
changes.

**What each kind of write waits for.** Take one page going from node A to node B, where B's next step needs it. B can't
place the bytes before it knows where they go. It learns that before the call, from the table, or during the call, over
the same cable. There are three ways to write the page.

- Posted ahead. B posted a receive at the page before the call. A sends. The bytes cross once and land in the page. B
  sees them on its next poll.
- Asking first. A sends a request. B sees it, picks a page, posts a receive there and sends an answer. A sees the answer
  and sends. The bytes cross, and B sees them. The request and the answer each land in a receive posted ahead, so this
  form needs 1 as well.
- Landing with a header. A sends a header and the bytes into a staging receive that B posted ahead. B sees them, reads
  the header and picks a page. It copies the bytes there, and B's next step waits for the copy.

| Write | Crossings | Waits | Choices at run time | Passes over the bytes |
|---|---|---|---|---|
| Posted ahead | 1 | 1 | 0 | 1 |
| Asking first | 3 or more | 3 | 1, and a receive posted then | 1 |
| Landing with a header | 1 | 2 | 1 | 3 |

On this hardware a crossing lands about 3.6 µs after its post, and a poll sees it 0.58 µs later (sections 1 and 2). So
a write posted ahead takes about 4.2 µs. Asking first takes three of those, about 12.6 µs, before B's choice and its
post. A copy on the GPU adds a dispatch, 0.9-1.4 µs, or a command buffer, 13.7-18.7 µs, and moves the bytes twice more.
Measured, a 3 KB crossing took 11.7 µs in a fixed order and 24.1 µs with a choice and a copy (section 4). Every hop of
every step pays its row again.

**Why this is the hardware, not a choice.** No write can put bytes in a page before B knows the page. If B learns it
during the call, that fact must cross the cable too, or the bytes must move again after they land. Each crossing and
each pass costs what sections 1-3 measured. Faster parts make each one cheaper, but they don't change how many there
are. That count is the same on InfiniBand and sm90 as here.

**What this makes of a collective here.** On sm90 a collective pays HBM transactions, PCIe and BAR hops, and the
congested split L2 at every step (section 1). A Mac node has none of these between its memory and its cable. So every
node a partial passes through can spend its own FLOPs and bandwidth on it. It can add partials as they pass, then send
the sum on. A collective here is a set of nodes that compute while they forward, not a set of endpoints joined by a
fabric.

**What these leave out of code.** Avoid an extra request, receipt, staging pass or reciprocal producer prerequisite.
The availability of a needed input is already a data dependency. Observing an already satisfied monotone prefix
does not require another network crossing. Realizing that observation once instead of in every consumer threadgroup
can remove repeated fences. Price these mechanisms in the complete graph, including overlap and resource occupancy.

**Measurement does not grant permission to execute.** A TCPStore arrival exchange or a fabric barrier before each
trial adds a two-sided control prerequisite even when the payload primitive is one-sided. Moving that exchange just
outside a timer does not remove its effect on submission, overlap or link utilization. A realized sequence submits
without peer start/finish rendezvous. Initial operand construction and final evidence collection are separate from
that sequence. Local observations of input availability and typed retirement remain actual data dependencies.

Whole-function statistics are asynchronous rollout evidence. Each record retains its episode, realized proposal,
operands and sampling probabilities. Late peer records may arrive several episodes together; missing records cannot
halt execution. Importance weights are target/behavior sampling probabilities, not a penalty for age or congestion.
The optimizer's proposed vector and the executing generation are distinct: statistics do not authorize unilateral
changes to remote destination bindings. Preparing a new generation belongs to configuration construction.

**Exact sub-page writes are compiled spans.** The 2026-10-24 byte-extent revision uses
`mesh_dma_span {uint64_t offset, bytes}` and `mesh_entry {spans, units, word, base}`
([mesh.h](../rdma/mesh.h), wire version 135, region layout 154). A span names the exact source or destination byte
interval, including an unaligned start or short tail. Its 16-byte descriptor is local immutable metadata, not a
header carried with the payload. A full allocation page uses the same interface. Each SGE transfers exactly its
span length; a 19-byte span does not read or overwrite the other bytes of its allocation page.

The compiler fixes equal SEND/receive lengths and order across endpoints and relays, with cuts at physical
registration boundaries and its selected packet limit. `ncclMeshWritePrepare` binds repeated extents;
`ncclMeshWriteRowsPrepare` binds affine rows with each participant's own pitch; `ncclMeshWriteSpansPrepare` binds
an ordered list of exact intervals within one lease. Each participant may use different span counts and offsets,
with the same total logical bytes. Common cuts include every participant's view boundaries and registration
boundaries. Repeated source intervals are valid; overlapping destination intervals are rejected at configuration.
These interfaces prepare one plan before
execution; `ncclMeshWriteExecute` reuses it without exchanging placement metadata or acknowledgments
([API](../rdma/nccl.h)). An affine row maps packed byte k to `(k / row_bytes) * pitch + k % row_bytes`.
One immutable span table describes all rows. The compiler uses each row's actual registration phase, so an odd
pitch does not impose every other row's boundary cuts on this row.

This binding-aware preparation is not yet supplied by every ordinary collective signature. Its logical-only
`CommPlan` cannot jointly cut arbitrary unaligned bindings crossing 1 GiB registrations. Such layouts require
the shared prepared WRITE binding stage; a provider SGE list or hidden packing copy is not a substitute.

The earlier masked pack/unpack implementation is a dated intermediate realization. It demonstrated frame-local
ownership but is not a reason to force padded copies on byte ranges the bridge can name directly. Arithmetic
functions may still have explicit masks; transport does not add a hidden packing pass. Exact extent payload bytes,
`ceil(bytes / 4096)` credit frames, and physical wire framing are different quantities. Charge actual payload once
per hop, report credit frames separately, and leave physical framing overhead unknown until measured.

All future and posted physical readers retain their backing allocations. DMA completion can retire source reads;
it cannot retire the destination's arithmetic consumer. Immutable span tables have separate SEND/LAND completion
obligations and bridge CPU-read lifetimes. On a give, the bridge's subsequent local owner transition proves its
CPU reader quiescent; payload DMA pins alone do not prove that. This adds no remote ACK.

The [2026-10-24 live validation](transport.md#exact-byte-validation-on-2026-10-24) passed three-rank exact-byte,
ragged-row, registration-crossing and neighbor-preservation checks, including all six cables on larger transfers.
Two-node and three-node fused runs each passed 14 cases and 94 calls per rank. A complete three-byte numerical
function measured median 31.290 µs with exact writes versus 35.569 µs with 16,384-byte padded transfers, across five
trials of 128 calls per mode. That is whole-function latency, not isolated NIC latency. Wire 135/layout 154 was
temporarily deployed for validation, then the preceding bridge was restored. The earlier cluster measurements below
are not measurements of this revision.

## 6. Why code here only loses waits and guards

This section distinguishes unnecessary control dependencies from the data and ownership dependencies that a compiled
program must satisfy. It keeps the operator's requirements below and corrects the earlier claim that every local
observation necessarily imposes another fixed crossing or poll interval.

**What each one costs.**

- An unsatisfied input dependency holds its consumer until the input is available. The 0.58 µs result measures a
  particular fenced observation path; it is not a charge for every satisfied predicate or every persistent read.
  An additional message crosses a link; observing the existing landing prefix adds no message.
- A sync makes the host wait until the GPU's work ends. The next work then starts in a new command buffer, 13.7-18.7
  µs. Without the sync it could have been a dispatch in an open one, 0.9-1.4 µs (section 1).
- A separate barrier waits for arrivals unrelated to a consumer's actual pages. A remote barrier message consumes
  a crossing; a local observation's latency depends on its mechanism. A real data join can often be expressed by
  existing landing prefixes and must not be charged as an additional network message.
- A guard checks at run time something that could have been fixed before the call. So it is a choice at run time.
  Data-dependent control flow is the same kind of choice. In section 5's table, each choice at run time came with more
  of the rest. Asking first has 3 crossings and 3 waits. Landing with a header has 2 waits and 3 passes over the bytes.
- Persistent state is not by itself a wait or guard. An immutable compiled table and a monotone landing prefix are
  useful persistent state. Mutable protocol decisions and ownership checks that can be proved before execution
  should not remain on the data path.

Section 4 measured some of these together. A 3 KB crossing took 11.7 µs as plain dispatches in a fixed order. It took
24.1 µs when each completion ran a scheduling kernel, the operand was copied to scratch, and the waits read gate words
(measurement.md:3230-3238).

**Nothing in the hardware makes them cheaper.** The driver has no remote write, no remote read and no atomics. It
sends no acknowledgement (section 3). So a node learns what another node did only when a SEND lands, and that is a
crossing every time. On the GPU, no threadgroup reads another's threadgroup memory. Threadgroups see each other's writes
only through device memory, and within one dispatch only in coherent memory (section 1). So a wait between threadgroups
is a poll of device memory. Faster parts make each crossing and each pass cheaper. They don't change how many there are
(section 5).

**Compare the realized whole graph.** An added dependency cannot improve the same fixed schedule by itself, but
changing a collective, its decomposition, or its dispatch structure changes that schedule. A reduce-scatter followed
by all-gather can move fewer bytes than exchanging every full partial despite having two phases. Consolidating
repeated threadgroup waits into one readiness dispatch improved the measured DNN (2026-10-24 microprofile), even
though it added a dispatch. Measure full execution time and useful delivery against critical-path and capacity
bounds. Do not infer either a gain or a loss from local wait counts alone.

The operator, 2026-10-21: "we are here to REMOVE fences, REMOVE guards, REMOVE waits. it is the responsibility of DATA
PROCESSING FUNCTIONS to not overwrite each other's buffers." The operator, 2026-10-18: "all of these sound like attempts
to add statefulness to functions that don't need it, and in so doing create contagious brittleness"
(design/rubrics.md M4). Of one cable waiting on another's word before it begins, the operator, 2026-10-23: "i do not
want an ipc barrier happening inside of networking code" (mesh 7bbd1c22).

**When one is ever added.** The operator, 2026-10-23: "the user will appear and demand the minimum amount of
statefulness, side effects, globals, and guards b presenting algorithmic problems that cannot be handled in any way
wtihout them". So a wait, a sync, a barrier, a guard, a global or a side effect comes in only when the operator
presents an algorithmic problem that can't be solved any other way. Then exactly the minimum comes in, at the
operator's demand. None comes in as a precaution.

**Every wait that remains has a deadline.** The operator, 2026-10-23: "no wait may ever run longer than a deadline
operand presented at the time the wait was constructed; the presumed default is 5 seconds." The reason, in the
operator's words: "a 'communicator' cannot wait forever for a crashed or hung data processing client because this
would reserve resources forever and prevent other replacement data code from using the collective. the bridge cannot
wait forever either for the same reason." So the waits with deadlines are the collective's and the bridge's own. Each
gets its deadline when it is made. Its caller or its configuration gives it, and 5 s is the default. A GPU kernel has
no clock, so its deadline is a count of polls (mesh.metal mesh_wait_bounded). Data code never sees one. The operator:
"if you need to rewrite a data program to 'add' a deadline it is not relevant to the scope which was described; this
is one of many features which is opaque to library users for obvious reasons."

When the deadline passes, the program ends at once and names the wait and its deadline. It does not try again, wait
again or fall back. The operator: "we will add bounds and crash programs voluntarily by producing invalid results or
exceptions caused by immediate exit due to unbounded wait error rather than allow an unbounded wait of any kind
anywhere anyhow." The data processing function gets garbage or an error: "we don't care about data processing code
being correct, complete, getting results every time, etc. we always corrupt data processing code's numerics if it's a
choice between that, running slower, faulting, or hanging." The collective releases whatever it was using. It lives
only as long as the program it runs in, and no collective kernel is kept for successive data programs to reuse. The
bridge is resident and lasts; it is not what ends. The operator: "we dont' care about collective or data processing
function lifespan at all; there is no specification for how a resident collective would work". This is the minimum
the operator demanded for the waits that remain. It adds no wait, and code still only loses them.

**Data code and the driver's state.** The bridge turns fast. The driver's completions are a ring that the controller
writes, and the bridge's thread reads it in user space. That thread saw each completion within about 0.04 µs
(measurement.md:3169-3170). Data code turns slower. Its step goes through a dispatch, 0.9-1.4 µs, or a command buffer,
13.7-18.7 µs (section 1). Its host loop runs in the program's own language. A dispatch alone is more than 20 times the
bridge's turn, and a command buffer is more than 300 times it. So data code can't return as fast as bridge code.

Say data code had synchronous access to some driver state. The driver would hold that state until data code had read
it. Then the driver would wait at data code's pace, not its own. So data code never has synchronous access to any
driver state.

Data code does not inspect private queue-pair/CQ state or hold the bridge still. It may read a bridge-published
monotone availability prefix for its declared pages. Observing prefix 12 proves that target 10 landed even if the
prefix becomes 19 before the consumer acts. This proof requires the same stream generation and retained page value;
a reset counter or reused page is a different type. Historical observations likewise remain useful to the solver
between full evaluations. Neither case requires the state to stop changing. The earlier inference that every
asynchronous observation becomes useless when its counter advances was incorrect.

The operator, 2026-10-23: "data processing code cannot return as fast as bridge code returns, so data processing code
can never be permitted to have synchronous access to any driver state", and asynchronous access "is tautological,
contradictory, or otherwise is algorithmically impossible".

**The coupling that is left.** Data code names its pages, and
their order, before the call (section 3; section 5, position 1). The bridge posts receives into those pages, and words
land in them (sections 2 and 3). Data code reads its own pages. It never reads the driver. The operator, 2026-10-23:
"it is the bridge's job to transact as fast as possible, it is the collective's job to use every rank to the maximum
extent possible and measure the residual in completion time between ranks to allow for cross-function-evaluation
rebalancings, and it is the data processing job's code to declare enormous tensors or workloads and let the collective
and bridge api backend converge on the fastest possible solving time the hardware can ever support no matter what the
mesh graph is" (the whole passage is in section 8).

**Proposals that keep coming back.** Some proposals came up again and again on 2026-10-23. Each one adds a thing this
section has priced. Each is answered by what the document already shows.

- **A trait of today's code, taken for a limit of the hardware.** "A relay isn't a member, so it can't know." "A
  call's counts exist only at the callers." "One lane can't carry two flows." "Relays need lanes of their own." Each
  names how today's code is laid out, not what the hardware does. Where each page lands, and in what order, is fixed
  before the call (sections 3 and 5). The callgraph and the tensors' shapes are fixed when the program is compiled (the
  operator, 2026-10-22; design/rubrics.md D2). So every node on a path can know the same facts ahead of time. The shape
  of today's code limits nothing.
- **A check against the program's own error.** Examples are a check of each call against its declared plan, and a
  pass/fail test against a reference. Each is a guard: a choice at run time, with its read and its wait. The operator,
  2026-09-10: "there are to be no parity mechanisms or syncs, guards, waits, checks, or inferences in the mesh
  computation transport and reduction layer" (design/pages-and-functions.md). Of a reference, the operator,
  2026-10-23: "what is a reference and why are you running anything related to it?"
- **One shape tested, then a threshold or a fixed rule to protect it.** One example is a piece under 8 pages sent whole
  on the first cable (mesh b0c64f27). Another is half the GPU's cores for the chain's workers, chosen
  at width 1,024 (docs/writing-a-service.md, Workers). At width 16,384 on two nodes, the chain with it took 1,539 µs a
  step, where its record took 865, about 78% slower (output_data/hourly/log.jsonl, run 20261023T2305Z). A threshold on
  size is a choice at run time, and the bridge does not care how big a write is (section 8). The demo is judged across
  shapes. The operator, 2026-10-23: "we don't care about artifacts tah only appear at one tensor shape ever ... this is
  why the lstm-chain-demo demands that multiple shapes be run".
- **A wait or a handshake between processes or threads.** Examples are other cables waiting on the first cable's count
  before they begin, and two threads in one port handing each holding between them. Each is a barrier inside
  networking code, in the operator's words above.
- **State kept from one holding to the next.** An example is a standing ring of relays kept for a pairing's whole life.
  That is state one call leaves for the next, a global with side effects. The next call reads it and checks it, which
  is a wait and a guard (above).
- **A message, a header or a descriptor on each call.** Each is a crossing, a wait or a pass that a write posted ahead
  does not have. Asking first has 3 crossings and 3 waits. Landing with a header has 2 waits and 3 passes (section 5's
  table).
- **Large buffers outside the region.** Examples are private float64 copies, contiguous copies of the weights, and
  evaluation in private memory. Every page a program uses lies in the registered region, so nothing is pinned at a call
  (section 2). The page table both ends hold is the program's memory (section 5, position 1). A buffer outside it
  reaches a cable only by a copy into it, which is one more pass over the bytes.
- **"Busy links", and "a relay takes capacity from another pair".** These are guesses made in place of a measurement.
  The collective measures the residual, how far apart the ranks finish, and work moves by it (section 8). The operator,
  2026-10-23: "there is no such thing as link business if there is a transfer backend mediated schedule of
  transactions" (section 8).
- **One more resident process.** The heartbeat is the one process launchd keeps on a node (the operator, 2026-10-21).
  The bridge it opens is the one writer on each device (section 7). Another resident process is one more party whose
  state the others would read and wait on.

## 7. What the bridge must have

Each requirement below follows from a section above, which it names.

**Register once.** The bridge registers one region with each device when the port opens (section 2). It keeps that
region when it restarts. The next bridge takes the same memory, so nothing is wired twice
(design/transport.md section 2). Nothing is registered during a call. Registering then would cost what sm90 pays,
with each buffer pinned in 64 KB pages, which can take milliseconds (section 2).

**Post ahead.** The compiled plan fixes each receive's page and order before execution. The bridge fills its finite
receive window as early as queue capacity and the proved page lifetime allow. A fresh destination does not depend
on an unrelated local publication. A recycled destination must have a first-write ownership proof; it must not gain
a receipt handshake or a reciprocal producer gate as a substitute. The driver places bytes through these preposted
receives because it has no hardware remote-write operation (section 3).

**No choices during a call.** The bridge sends no header, makes no request and gives no answer. It picks no page from
what arrives (section 5, position 3). Each of these would cost another crossing or another pass over the bytes (the
table in section 5). So the bridge reads and writes words, and it never touches a byte of the pages it moves.

**One writer per device.** Only the bridge posts on each device (section 3), and only it counts how many frames each
queue has free. The driver takes a post past a queue's frames, and later frames come out corrupt. It does not check
the order of calls on a device, or it checks by faulting (design/transport.md section 2).

**Queue pairs made once.** Each queue pair is made once and connected once. Its only release is the exit of the
process that holds it (section 3). On the M5, a move back to RESET caused a panic on October 19, and a teardown call
left the device stuck for 17 hours on October 18 (design/transport.md sections 5 and 14). On October 20 the operator
ruled that no queue pair is reset, put in Error or destroyed while its process runs. On October 22 the operator
allowed one exception, the flush the literature uses: a queue pair is moved to Error so that each request it still
holds ends with a completion. The operator's words: "you can do this and nothing else with error states"
(design/rubrics.md M3).

**Minimize fixed cost per step.** The bridge must eliminate avoidable per-step work; zero overhead is not established
by writing that requirement. The measured floors make fixed work consequential. A
poll sees a store 0.58 µs after it is made, and a dispatch costs 0.9-1.4 µs (section 1). A crossing lands about
3.6 µs after its post (section 4). So a write posted ahead takes about 4.2 µs (section 5). Each µs the bridge adds is
a large share of that, about a quarter.

## 8. Writes over many hops and many paths

This section works out what a write costs when it takes more than one path. It uses three figures. B is one cable's
rate, about 9.2 GB/s each way with both ways in use (9.21 and 9.25 GB/s on 2026-10-01, measurement.md:364). P is the
16 KiB packet assumed by the earlier page-based measurements, not a minimum byte-extent payload. Δ is what each hop past the first adds, about 4.2 µs: 3.6 µs from post to landing, then a poll
(section 5).

**What a write over many hops is.**

- Each hop lands in a receive posted ahead, at a place fixed before the call. So nothing on the path decides anything
  at run time (section 5, positions 1 and 3).
- A relay's bridge sees the landing and sends the bytes on from the same memory. It never touches a byte of them. No
  CPU or GPU work runs on them there.
- A relay works piece by piece. Piece k moves on while piece k + 1 comes in. A path of h hops adds about
  (h − 1)·Δ to first arrival and also to finite-transfer completion. Once full, its rate is bounded by its slowest
  hop and the other traffic sharing each directed cable; enumerated paths do not acquire independent capacity.
- Each byte a relay sends on costs one DMA write and one DMA read in the relay's DRAM. A node that receives pays one
  DMA write a byte. Each port costs one polling core, since commit 60b00774 (design/transport.md section 14).

**The bound is a cut, not the cables.** A transfer between two nodes can use at most the max-flow between them (Ford
and Fulkerson 1956). It can be no more than the source's ports, and no more than the target's. The retained
2026-10-24 morning microprofile configuration had **seven cables**: four join m5ult1 and m5ult2, one joins m5ult1 and the Mini,
and two join m5ult2 and the Mini. Their incoming degrees are **5, 6, 3**. Provenance is the physical cable table in
`metal-microbench/output_data/review-20261024/microprofile/fixed.json`, with individual wire identities and advertised
80 Gbit/s directions, and the measured executions in its neighboring `RESULTS.md`. The four-cable record in
`output_data/spin-20261023` was an earlier topology, not a current solver input. Discovery supplies the graph;
neither the interface nor the solver assumes these ranks, degrees or equal capacities.

Read-only discovery at **2026-10-24 12:57:27 PDT** instead found **six paired cables**: 4/1/1, with
ingress degrees **5/5/2**. The second Studio's port 2 reported `STOPPED`; Mini port 1 reported `PAIRING`.
That observation establishes an unavailable pairing, not its cause. At that instant the advertised active wire
ceiling was **120 GB/s** over twelve directions. Re-discover before each realized configuration and retain the actual
wire identities and statuses with every measurement; do not carry the morning's missing cable into a later roofline.
The retained [discovery snapshot](../../../metal-microbench/output_data/review-20261024/interface-refactor/topology-current.json)
timestamps all three observations at 19:57:27 UTC. The
[Mini bridge log](../../../metal-microbench/output_data/review-20261024/interface-refactor/bridge-tail-rank2.log)
records the `en3` peer interface terminating at 17:34:51.987 UTC, before this refactor's native build and tests.
This chronology establishes pre-existing unavailability; it does not diagnose its cause.

| Pair | Morning direct cables / isolated max-flow in B | 12:57 direct cables / isolated max-flow in B |
|---|---|---|
| m5ult1 and m5ult2 | 4 / 5 | 4 / 5 |
| m5ult1 and the Mini | 1 / 3 | 1 / 2 |
| m5ult2 and the Mini | 2 / 3 | 1 / 2 |

These are isolated-pair bounds. Simultaneous demands share every traversed cable's capacity. The 9.2 GB/s value
above is a historical measured rate; 10 GB/s is the advertised per-direction ceiling. Fourteen advertised directions
gave 140 GB/s aggregate wire capacity for the morning measurement; twelve gave 120 GB/s at 12:57. Useful output
bytes exclude extra relay hops and redundant collective work.

**What taking in costs a node's memory.** A node that takes in on N ports takes N × 9.2 GB/s of DMA writes.

| N | GB/s | Of 620 GB/s | Of 1.2 TB/s | Of 273 GB/s |
|---|---|---|---|---|
| 3 | 27.6 | 4.5% | 2.3% | 10.1% |
| 6 | 55.2 | 8.9% | 4.6% | — |
| 12 | 110.4 | 17.8% | 9.2% | — |
| 18 | 165.6 | 26.7% | 13.8% | — |

620 GB/s is what the chain's kernel streams on a Studio, and 1.2 TB/s is Apple's figure for the M5 Ultra (section 1).
273 GB/s is Apple's figure for the Mini's M4 Pro (Apple, Mac mini specs). The Mini has 3 ports and a Studio has 6
(Apple, Mac mini specs and Mac Studio specs). So 12 and 18 are what-ifs. A relay that carries one cable's worth spends
about 18.4 GB/s, since it writes each byte once and reads it once. The two-cable 15.36 GB/s smoke test on 2026-10-23
(`output_data/smoke-20261023-11c52f5e`) is historical evidence. The 2026-10-24 microprofile records traffic in all
fourteen directions, including all six inputs at the second Studio, and a best retained large-run aggregate wire
rate of 113.08–113.51 GB/s. Positive traffic establishes coverage, not continuous saturation of each port. Neither
the memory ratios nor DMA's low FLOP cost establish compute interference; use measured communication dependencies
and whole-function residuals rather than assuming that relaying steals significant arithmetic capacity.

**When more paths win in an independent-path model.** Take one direct path with latency L1, and N − 1 relayed
paths with latency L2 = L1 + Δ, each with independent capacity B. Shared physical cables require the multicommodity
constraints instead; the expressions below must not be applied by counting overlapping paths. Split a transfer of
S bytes so that all paths end together, provided each calculated share is nonnegative.

- The direct path alone takes T1 = L1 + S/B.
- All N paths take T_N = S/(N·B) + mean(L).

All the paths beat the direct one once S > Δ·B, about 38.6 KB. That holds for any N. Now say each path carries at
least one packet of P useful bytes. Under that packet assumption all N paths are in use from
S_min(N) = Δ·B + N·P up. The historical table uses P = 16 KiB; exact byte spans require recomputing it from the
actual compiled packet sizes, without substituting rounded credit frames for useful payload.

| N | S_min | Time saved at S_min |
|---|---|---|
| 3 | 87,792 B | 3.6 µs |
| 6 | 136,944 B | 8.9 µs |
| 12 | 235,248 B | 19.6 µs |
| 18 | 333,552 B | 30.3 µs |

The time saved at S_min is (N − 1)·P/B. Far above S_min, the time comes close to S/(N·B). When every path is direct,
the Δ·B term drops out. Each hop past the first adds about Δ·B, 38.6 KB, to the size where its path pays off. A path
of 10 hops adds about 348 KB. One cable moves that in about 38 µs, well under a millisecond.

**Who balances the load.** The operator, 2026-10-23: "there is no such thing as link business if there is a transfer
backend mediated schedule of transactions. a link might be 'busy' and it might not be. it is not anyone's job to think
about this; it is the bridge's job to transact as fast as possible, it is the collective's job to use every rank to
the maximum extent possible and measure the residual in completion time between ranks to allow for
cross-function-evaluation rebalancings, and it is the data processing job's code to declare enormous tensors or
workloads and let the collective and bridge api backend converge on the fastest possible solving time the hardware can
ever support no matter what the mesh graph is."

| Who | Its job |
|---|---|
| The bridge | Transact as fast as it can. |
| The collective | Use every rank as fully as it can. Measure the residual, how far apart the ranks finish, so work can move between one evaluation and the next. |
| The data processing code | Declare huge tensors or workloads. Let the collective and the bridge find the fastest time the hardware allows, whatever the mesh's graph. |

When nodes compute at different speeds, a weaker node gets a smaller share. In the LSTM on three nodes the shares are
448, 448 and 128 of 1,024, and the Mini's is the 128 (output_data/lstm-20261022/t134-n3-lstm-rank0.log). So the Mini
ends its compute early. Then its DMA engines, links and memory have slack, and slack is just what a relay uses. The
balancer combines physical capacities and typed dependencies with measured whole-function residuals. A fitted
fixed-plus-work compute model and finite differences update compute and routing together between evaluations.
Relaying's resource cost must be measured rather than presumed; a fitted model is not an excuse to ignore residuals.

**The bridge does not care how big a write is.** The operator, 2026-10-23: "it is not the job of the bridge or the
collective to make programs run slower based on narrow case guesses for how big a transaction is; the bridge is not
there to 'care' about how big a write is; only to schedule writes and their swizzlings and so on s.t. they all happen
as fast as possible every single time. a bridge ideally does not even know what a 'page' is or which data dependencies
are related to which pages; as that is a namespace of data processing functions and collective functions."

So the sizes worked out above are facts about the hardware. They are not rules the bridge applies. Neither the bridge
nor the collective makes a program slower on a guess about how big a transfer is. The bridge schedules writes, and how
they are laid over the paths, so that all of them happen as fast as they can, every time. Ideally a bridge does not
even know what a page is, or which data goes with which page. Those belong to the data processing functions and the
collective functions.

## 9. Any collective as compiled one-sided writes

The operator, 2026-10-23: "if any of these are slow then the data model's eplanation of them is wrong and teh data
model docuemnt will be re-written until we explain how we can run any collective with one-sided writes with no
irreducible overhead, e.g. no overhead that could be reduced by writing a single program 'fused backend' for that one
program alone." "These" are the steps of expert parallelism, ring attention and Megatron, and the collectives that a
tensor-parallel program makes. This section explains how each one runs.

**What AOT knowledge permits and does not prove.** A fused backend is code written for one program alone. It knows
that program's shapes and the order of its calls. A generic backend can know them too. The operator, 2026-10-22: "we do not
intend to support any source level metaprogramming which tries to change what the syntax of the callgraph is or the
shapes of any tensors are partway through a tensor function execution" (design/rubrics.md D2). So the shapes, their
maximum extents and the callgraph are fixed when the program is compiled. Then every call's pages, and their order,
can be compiled ahead. That is done once, for every rank. Every rank and every relay on a path holds the same facts
(section 6). The bridge posts each receive at its page before its SEND comes (section 7).

This removes an information disadvantage; it does not select optimal routes, collective decomposition, reduction
placement, range size, dispatch structure, or overlap. A full partial exchange and reduce-scatter plus all-gather
compute the same sum with different byte counts and dependencies. For three equal direct-link ranks the former
sends 6S aggregate bytes and the latter 4S, before padding; the latter has an extra phase and is not automatically
faster for a small operand. Both must be representable by the same physical-write interface. AOT list scheduling
is a heuristic, and fluid max-flow is a lower bound with its assumptions, not proof of attained execution time.
Report useful work, whole-function latency, transaction-window bandwidth, actual wire amplification, and the gap to
critical-path and directed-capacity bounds. Setup and between-call optimization costs also count in sustained calls.

**What the implementation must remove.** These are requirements, not a claim that all historical paths already
satisfy them. Measure remaining realization, dispatch and ownership costs; do not hide them behind the word AOT.

- Planning per call when the complete graph is already known. Host compilation has its measured CPU and allocation
  cost; it is not automatically a three-crossing request protocol. Compile at configuration changes, realize all
  page bindings, and include amortized setup in sustained throughput.
- A fixed size rule that forbids a useful realization or silently selects a narrower interface. Shapes are valid
  compiler metadata and may inform AOT arithmetic/collective choice. They are not permission to discard cables or
  substitute a narrow empirical rule for complete-function optimization (section 6's 1,024/16,384 counterexample).
- Locks within an already realized data path. Measure their actual contention and dispatch costs instead of
  assigning every local lock a network crossing or a 0.58 µs GPU poll. Move ownership work to realization or
  asynchronous retirement where its proof permits that.
- Data-program waits on private driver state. The bridge owns queue progress and source DMA retirement. Data code
  consumes its own pages and their monotone availability evidence; neither that evidence nor between-call
  observations require freezing driver state (section 6).
- Host round trips each step. The host waits for the GPU and then starts a new command buffer, 13.7-18.7 µs. A
  dispatch in an open one costs 0.9-1.4 µs (section 1). In the chain's step of 27.2-28.8 µs, that adds about half to
  two thirds again (section 6).

**Each kind of collective, as compiled writes.** Below, n is the number of ranks and S is the operand's bytes. B is
one cable's rate, about 9.2 GB/s each way (section 8).

- All-gather and reduce-scatter over ragged shares. Rank r holds p_r of P units (partition.py, attach). The shares
  stay fixed between evaluations. So the plan knows each share's pages, and where each one lands in every output. In
  an all-gather, each share's pages go to every other rank, into their places. In a reduce-scatter, each rank's partial
  of a block goes to the block's owner. A node it passes through can add it on the way (section 5). Neither has a
  header, a count or a choice in the call.
- All-reduce, in two phases. A reduce-scatter leaves each rank with its block summed. Then an all-gather sends each
  sum to every rank. In each phase an equally partitioned node takes in (n − 1)/n of S. For an isolated two-rank
  example with cut C, the two phases carry S/2 each toward either rank, a fluid lower bound S/C before dependency
  startup, arithmetic and framing. Both dated Studio-pair observations above have isolated cut 5B, with the relay explicitly in the
  compiled topology. Concurrent Mini traffic shares that capacity and must enter the same solve.
- Broadcast and reduce from a root. In a broadcast, every other node takes in all of S. So it can go no faster than
  the smallest cut between the root and any one node (section 8). A reduce is the same writes run the other way.
  Partials flow toward the root, and each node they pass adds them (section 5).
- Expert routing. It is an all-to-all. Each rank sends each expert's rows to the rank that holds that expert, and the
  results come back. Its pages are fixed ahead in one of two ways. The counts can be fixed ahead for each layer. Or
  each slot can be sized at its maximum extent. Then the count the GPU wrote moves with the rows, and the padding past
  it is never read unmasked (partition.py's header).
- Context-parallel attention. The reviewed implementation gathers query blocks and reduce-scatters attention
  partials; it does not circulate K/V blocks despite the historical `_ring_attention` name. A K/V ring is a separate
  possible decomposition. Both express their actual page/range dependencies through compiled writes, and must be
  compared as complete functions rather than equated by naming.
- Megatron's column and row splits (partition.py, ColwiseParallel and RowwiseParallel). A column split gives each
  rank some of a weight's columns. Each rank's output is then its own share, and nothing crosses. A row split gives
  each rank some of the rows, so each rank's output is a partial sum. That ends in a reduce-scatter or an all-reduce,
  as above. So a column split and then a row split has one collective, at the place the callgraph fixes.
- A relay. A relay is the identity at a rank in the topology. What lands there is sent on unchanged, from the same
  memory, and its bridge never touches a byte (section 8). Where a node adds partials as they pass, the sum takes the
  identity's place. Either way, each path's relays are in the plan.

**Where the balancer fits.** The collective measures the residual, how far apart the ranks finish (section 8).
Between one evaluation and the next, the balancer sets the shares from it. A new share is then a new compiled plan,
with new pages and a new order, made once for every rank. It is not control at run time. Within an evaluation,
nothing about the shares is chosen.

**Implementation evidence and lifetime proof.** The pinned 2026-10-24 review is
`metal-microbench/output_data/review-20261024/data-model/REVIEW.md`. It located the old single-cable restriction in
fused lowering, rather than fabric discovery, and documented staging, repeated waits and receipt collectives.
Those are implementation defects, not sanctioned alternative interfaces. Current code changes must be validated
against that evidence without treating either an old source description or this specification as a performance
measurement. The page ownership solver and its two-slot recurrence proof are specified in
[whole-function.md](whole-function.md#typed-one-sided-lifetimes); they introduce no acknowledgment or data copy.

## Sources

- Apple. TN3205: Low-latency communication with RDMA over Thunderbolt. 2026.
- Apple. Mac Studio: Technical Specifications. apple.com/mac-studio/specs, read 2026-10-23.
- Apple. Mac mini (2024): Tech Specs. support.apple.com/121555, read 2026-10-23.
- Apple. Explore GPU advancements in M3 and A17 Pro. Tech talk 111375.
- Apple. Boost your graphics performance with the M5 and A19 GPUs. Tech talk 111431.
- Apple. Accelerate your machine learning workloads with the M5 and A19 GPUs. Tech talk 111432.
- C. Lam. A Brief Look at Apple's M2 Pro iGPU. Chips and Cheese, 2023.
- NVIDIA. NVIDIA Hopper Architecture In-Depth. 2022.
- NVIDIA. CUTLASS: Warpgroup MMA Programming Guide.
- NVIDIA. GPUDirect RDMA: Developing a Linux Kernel Module using GPUDirect RDMA.
- W. Luo et al. Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis.
  arXiv 2501.12084v1, 2025.
- K. Hamidouche et al. GPU-Initiated Networking for NCCL. arXiv 2511.15076v2, 2025.
- C. Zhao et al. Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures. ISCA
  2025.
- InfiniBand Trade Association. InfiniBand Architecture Specification, Volume 1.
- J. Gu et al. Efficient Memory Disaggregation with Infiniswap. NSDI 2017.
- L. R. Ford and D. R. Fulkerson. Maximal flow through a network. Canadian Journal of Mathematics 8, 1956.
