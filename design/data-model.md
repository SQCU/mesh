# The data model

These sections say what the hardware is. Each number names its source. The files measurement.md and facts.md and the
folder output_data/ are in the metal-microbench repo. The papers are listed at the end.

Code here only loses waits, syncs, barriers, guards, data-dependent control flow, globals and side effects. None of
them is ever added to code that did not have it. Section 6 works out why, from what sections 1-5 measure. It also says
when one is ever added, and why data code never reads the driver's state.

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
span the same number of 4 KB frames. The hardware holds the SEND in credit until that receive is posted, and it sends
no acknowledgement (TN3205; design/transport.md section 1). Each request names one range (max_sge 1). A device takes
100 registrations (max_mr 100). The driver keeps only an address's low 32 bits. So a registration across a 4 GiB
boundary lands its receives 4 GiB low (design/transport.md section 1). A device reports max_qp 3 on the Studios' macOS
27, and 11 on macOS 26 (design/HETEROGENEOUS-LINKS.md section 1). One queue pair carries one line of receives. There is
no shared receive queue.

So the receiver fixes where each frame lands, and in what order, before it comes. A program that picks addresses just in
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

**1. Paging and a memory abstraction are a hard requirement.** Every byte a program moves lives in a page of a page table
that both ends know before the call. A page lands in a free page, and its row says which page it is. Without the table,
each move has to be asked for or described at run time. That costs a round trip or a header every time, and the cost
grows with every hop and every step.

**2. Copying memory to 20 other machines is faster than reading it from the same machine's NVMe, when the code is not
written wrong.** One preposted hop here lands a page in about 2-4 µs (sections 2 and 3). A random NVMe read takes tens
of µs to about 100 µs before the first byte can be used. One Thunderbolt 5 cable moves about 10 GB/s, as much as or more
than a Mini's own SSD. On an InfiniBand fat tree a remote node is one NIC hop and a few switch hops away, so remote
memory beats local NVMe there too (Infiniswap, NSDI 2017). On the Mac mesh each node is itself the switch. So each hop
adds its own few µs, and 20 hops are still faster than one random NVMe read.

**3. One-sided writes are faster than two-sided writes that run a protocol at run time.** A one-sided write needs 1. It
puts bytes in a page that the receiver named before the call, in the table both ends hold, and nobody answers. Without
that table there is no page to name, so there is no one-sided write. InfiniBand's RDMA WRITE is the same: the sender
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

**What these leave out of code.** Sections 1-5 count each crossing, each wait and each pass over the bytes, and they
price each one. A wait, a sync, a barrier or a guard is always one more of these, or a fixed cost on every step.
Nothing in sections 1-3 makes one cheaper. So code here only loses them, and with them data-dependent control flow,
globals and side effects. None of them is added to code that did not have it. Section 6 works this out.

## 6. Why code here only loses waits and guards

This section works out, from sections 1-5, why no choice here ends in adding a wait, a sync, a barrier or a guard. It
says when one is ever added. Then it says why data code never reads the driver's state.

**What each one costs.**

- A wait holds a step until a word lands. A poll sees a store 0.58 µs after it is made, at the soonest (section 1).
  If the word comes from another node, it first crosses, in about 3.6 µs (section 2). Each wait is one more in the
  "Waits" column of section 5's table.
- A sync makes the host wait until the GPU's work ends. The next work then starts in a new command buffer, 13.7-18.7
  µs. Without the sync it could have been a dispatch in an open one, 0.9-1.4 µs (section 1).
- A barrier makes each party wait until every other one has arrived. Between nodes that is at least one more crossing
  and one more wait, and no data moves in it (sections 2 and 5). Between processes on one node, it is a poll of another
  process's word. That takes 0.58 µs at the soonest, and then as long as the other process takes.
- A guard checks at run time something that could have been fixed before the call. So it is a choice at run time.
  Data-dependent control flow is the same kind of choice. In section 5's table, each choice at run time came with more
  of the rest. Asking first has 3 crossings and 3 waits. Landing with a header has 2 waits and 3 passes over the bytes.
- A global or a side effect is state that one call leaves for another to find. The other call reads it and checks it
  before it acts. That read is a wait, and that check is a guard.

Section 4 measured some of these together. A 3 KB crossing took 11.7 µs as plain dispatches in a fixed order. It took
24.1 µs when each completion ran a scheduling kernel, the operand was copied to scratch, and the waits read gate words
(measurement.md:3230-3238).

**Nothing in the hardware makes them cheaper.** The driver has no remote write, no remote read and no atomics. It
sends no acknowledgement (section 3). So a node learns what another node did only when a SEND lands, and that is a
crossing every time. On the GPU, no threadgroup reads another's threadgroup memory. Threadgroups see each other's writes
only through device memory, and within one dispatch only in coherent memory (section 1). So a wait between threadgroups
is a poll of device memory. Faster parts make each crossing and each pass cheaper. They don't change how many there are
(section 5).

**So adding one can only slow a program.** A step with one more wait, crossing or pass can't end sooner than it did
without it. A write posted ahead takes about 4.2 µs (section 5). One more poll, 0.58 µs, adds about a seventh to that.
One more crossing and its poll double it. A new command buffer, 13.7-18.7 µs, is three to four and a half times it.
The chain's whole step takes 27.2-28.8 µs (section 4), and a new command buffer in each step would add about half to
two thirds again. Every hop of every step pays it again (section 5).

The operator, 2026-10-21: "we are here to REMOVE fences, REMOVE guards, REMOVE waits. it is the responsibility of DATA
PROCESSING FUNCTIONS to not overwrite each other's buffers." The operator, 2026-10-18: "all of these sound like attempts
to add statefulness to functions that don't need it, and in so doing create contagious brittleness"
(design/rubrics.md M4). Of one cable waiting on another's word before it begins, the operator, 2026-10-23: "i do not
want an ipc barrier happening inside of networking code" (design/transport.md section 14).

**When one is ever added.** The operator, 2026-10-23: "the user will appear and demand the minimum amount of
statefulness, side effects, globals, and guards b presenting algorithmic problems that cannot be handled in any way
wtihout them". So a wait, a sync, a barrier, a guard, a global or a side effect comes in only when the operator
presents an algorithmic problem that can't be solved any other way. Then exactly the minimum comes in, at the
operator's demand. None comes in as a precaution.

**Data code and the driver's state.** The bridge turns fast. The driver's completions are a ring that the controller
writes, and the bridge's thread reads it in user space. That thread saw each completion within about 0.04 µs
(measurement.md:3169-3170). Data code turns slower. Its step goes through a dispatch, 0.9-1.4 µs, or a command buffer,
13.7-18.7 µs (section 1). Its host loop runs in the program's own language. A dispatch alone is more than 20 times the
bridge's turn, and a command buffer is more than 300 times it. So data code can't return as fast as bridge code.

Say data code had synchronous access to some driver state. The driver would hold that state until data code had read
it. Then the driver would wait at data code's pace, not its own. So data code never has synchronous access to any
driver state.

Now say data code reads driver state while the driver goes on. The bridge moves on within about 0.04 µs. Data code acts
a dispatch or more later. By then the state it read has moved on, so the read can't inform any action. For the read to
inform one, the state would have to hold still until data code acts. That is synchronous access again. So asynchronous
access to driver state is a contradiction. It is either synchronous, or it informs nothing.

The operator, 2026-10-23: "data processing code cannot return as fast as bridge code returns, so data processing code
can never be permitted to have synchronous access to any driver state", and asynchronous access "is tautological,
contradictory, or otherwise is algorithmically impossible".

**The coupling that is left.** The data model already has the one coupling that works. Data code names its pages, and
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
  on the first cable (design/transport.md section 14). Another is half the GPU's cores for the chain's workers, chosen
  at width 1,024 (docs/writing-a-service.md, Workers). At width 16,384 on two nodes, the chain with it took 1,539 µs a
  step, where its record took 865, about 78% slower (output_data/hourly/log.jsonl, run 20261023T2305Z). A threshold on
  size is a choice at run time, and the bridge does not care how big a write is (section 8). The demo is judged across
  shapes. The operator, 2026-10-23: "we don't care about artifacts tah only appear at one tensor shape ever ... this is
  why the lstm-chain-demo demands that multiple shapes be run".
- **A wait or a handshake between processes or threads.** Examples are other cables waiting on the first cable's count
  before they begin, and two threads in one port handing each holding between them. Each is a barrier inside
  networking code, in the operator's words above. The count is deleted, and a port has one progress thread
  (design/transport.md sections 5 and 14).
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

**Post ahead.** The bridge posts each receive before its SEND arrives. It posts it at the page the receiver named, in
the order both ends fixed (section 3; section 5, position 1). The driver gives it no other way to place bytes, since it
has no remote write (section 3).

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

**No fixed cost per step.** The bridge adds no fixed cost to a step, because the floors under a step are small. A
poll sees a store 0.58 µs after it is made, and a dispatch costs 0.9-1.4 µs (section 1). A crossing lands about
3.6 µs after its post (section 4). So a write posted ahead takes about 4.2 µs (section 5). Each µs the bridge adds is
a large share of that, about a quarter.

## 8. Writes over many hops and many paths

This section works out what a write costs when it takes more than one path. It uses three figures. B is one cable's
rate, about 9.2 GB/s each way with both ways in use (9.21 and 9.25 GB/s on 2026-10-01, measurement.md:364). P is a
page, 16 KiB. Δ is what each hop past the first adds, about 4.2 µs: 3.6 µs from post to landing, then a poll
(section 5).

**What a write over many hops is.**

- Each hop lands in a receive posted ahead, at a place fixed before the call. So nothing on the path decides anything
  at run time (section 5, positions 1 and 3).
- A relay's bridge sees the landing and sends the bytes on from the same memory. It never touches a byte of them. No
  CPU or GPU work runs on them there.
- A relay works piece by piece. Piece k moves on while piece k + 1 comes in. So a path of h hops adds about
  (h − 1)·Δ to when the first byte lands. It does not add that to how long the whole transfer takes. Once it is full,
  each path delivers B.
- Each byte a relay sends on costs one DMA write and one DMA read in the relay's DRAM. A node that receives pays one
  DMA write a byte. Each port costs one polling core, since commit 60b00774 (design/transport.md section 14).

**The bound is a cut, not the cables.** A transfer between two nodes can use at most the max-flow between them (Ford
and Fulkerson 1956). It can be no more than the source's ports, and no more than the target's. The mesh has four
cables. One joins m5ult1 and m5ult2, one joins m5ult1 and the Mini, and two join m5ult2 and the Mini
(output_data/spin-20261023, each run's paired.log; commit 7876150c). So each pair has one more path's worth than its
own cables, through the third node.

| Pair | Its own cables | Paths' worth |
|---|---|---|
| m5ult1 and m5ult2 | 1 | 2 |
| m5ult1 and the Mini | 1 | 2 |
| m5ult2 and the Mini | 2 | 3 |

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
about 18.4 GB/s, since it writes each byte once and reads it once. No one has measured a Studio taking in on all 6
ports at once. The one figure for more than one cable is 15.36 GB/s each way on two cables, 1.67 times one cable. It
came from a smoke test on 2026-10-23 (output_data/smoke-20261023-11c52f5e; commit 7876150c).

**When more paths win.** Take one direct path with latency L1, and N − 1 relayed paths with latency L2 = L1 + Δ.
Split a transfer of S bytes so that all the paths end at the same time.

- The direct path alone takes T1 = L1 + S/B.
- All N paths take T_N = S/(N·B) + mean(L).

All the paths beat the direct one once S > Δ·B, about 38.6 KB. That holds for any N. Now say each path carries at
least one page. Then all N paths are in use from S_min(N) = Δ·B + N·P up.

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
balancer is to set the shares from the measured residuals, not from a model. If relaying ever slows a node, its
residual shows it, and the balancer moves work.

**Where today's code differs.** Today the bridge counts 16 KiB pages: each SEND and each receive is one whole page, and
it cuts a piece over a node's cables by its count of pages (design/transport.md section 14). And the bridges forward
nothing, so a transfer uses only the cables that join its two ends (rdma/nccl-mesh.c, map_of).

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

**Why a generic backend matches a fused one.** A fused backend is code written for one program alone. It knows that
program's shapes and the order of its calls. A generic backend can know them too. The operator, 2026-10-22: "we do not
intend to support any source level metaprogramming which tries to change what the syntax of the callgraph is or the
shapes of any tensors are partway through a tensor function execution" (design/rubrics.md D2). So the shapes, their
maximum extents and the callgraph are fixed when the program is compiled. Then every call's pages, and their order,
can be compiled ahead. That is done once, for every rank. Every rank and every relay on a path holds the same facts
(section 6). The bridge posts each receive at its page before its SEND comes (section 7).

So a step is two things. Writes posted ahead land, at one crossing and one wait a hop, about 4.2 µs (section 5). And
the program does its own arithmetic. A fused backend could do no less. It crosses the same cables, and the hardware
fixes how many crossings and waits a page needs (section 5). What is left is the bytes over the cut, and that bounds
the fused backend just as it bounds the generic one (section 8).

**What this rules out.** Each of these costs something a fused backend would not pay. So none of them is in a generic
one.

- Planning per call. A plan made at the call is a choice at run time. In section 5's table, asking first takes 3
  crossings, about 12.6 µs, where a write posted ahead takes about 4.2 µs. Measured, a 3 KB crossing took 11.7 µs in a
  fixed order. It took 24.1 µs with a scheduling kernel, a copy and gate words (section 4).
- A program chosen by a size threshold. That is a choice at run time too, tuned on one shape. A rule chosen at width
  1,024 made the chain about 78% slower at width 16,384 on two nodes: 1,539 µs a step against 865 (section 6). The
  bridge does not care how big a write is (section 8).
- Locks. A lock makes one party wait on another's word. On one node that is a poll, 0.58 µs at the soonest, and then
  as long as the other party takes. Between nodes it is at least one more crossing, about 3.6 µs, and no data moves in
  it (section 6).
- Waits on driver state. The bridge's thread sees a completion within about 0.04 µs. Data code's turn is a dispatch,
  0.9-1.4 µs, or a command buffer, 13.7-18.7 µs. A wait on driver state either holds the driver to data code's pace,
  or reads state that has already moved on (section 6).
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
  sum to every rank. In each phase a node takes in (n − 1)/n of S, so its cut bounds the time. Take two nodes with two
  paths' worth between them, as m5ult1 and m5ult2 have (section 8's table). Each phase takes in S/2 at 2B, which is
  S/(4B). Both phases take S/(2B).
- Broadcast and reduce from a root. In a broadcast, every other node takes in all of S. So it can go no faster than
  the smallest cut between the root and any one node (section 8). A reduce is the same writes run the other way.
  Partials flow toward the root, and each node they pass adds them (section 5).
- Expert routing. It is an all-to-all. Each rank sends each expert's rows to the rank that holds that expert, and the
  results come back. Its pages are fixed ahead in one of two ways. The counts can be fixed ahead for each layer. Or
  each slot can be sized at its maximum extent. Then the count the GPU wrote moves with the rows, and the padding past
  it is never read unmasked (partition.py's header).
- Ring attention. Each rank keeps its query block. The K/V blocks go around a fixed ring, one block a turn. At each
  turn a rank attends over the block it holds and sends that block on. Which block lands where, at each turn, is fixed
  ahead. So the ring is a fixed list of writes posted ahead.
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

**Where today's code is slower than this.**

- A communicator holds one stream toward each member's node, on one cable (rdma/nccl-mesh.c, comm_hold), so a
  collective between two nodes gets B where their cut is 2B or 3B (section 8's table).
- routed reads the received counts on the host and sizes each expert's rows from them, a host round trip in every
  layer (rdma/torch-mesh/torch_mesh/partition.py, routed).
- allgatherv copies this rank's block into its place in the output, and reduce_scatterv copies an input that lies
  outside the region, each one more pass over the bytes (partition.py, allgatherv and reduce_scatterv).

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
