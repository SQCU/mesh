# The data model

These sections say what the hardware is. Each number names its source. The files measurement.md and facts.md and the
folder output_data/ are in the metal-microbench repo. The papers are listed at the end.

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
(measurement.md:414, 3233-3235). Apple gives the M5 Ultra 1.2 TB/s of memory bandwidth for its 80-core model, and no separate figure for the Studios'
64-core model. At width 16,384 the chain's kernel streams its weights at about 620 GB/s, about half of that
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
extents of 1 GiB; 86 of them registered in 0.165 s (facts.md; design/HETEROGENEOUS-LINKS.md section 1; etc/bridge/README.md).
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

## 6. What the bridge must have

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

## Sources

- Apple. TN3205: Low-latency communication with RDMA over Thunderbolt. 2026.
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
