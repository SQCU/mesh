# The data model

These sections say what the hardware is. Each number names its source. The files measurement.md and facts.md and the
folder output_data/ are in the metal-microbench repo. The papers are listed at the end.

The hardware paths explain why programs here use pages and compile their writes before a call. Sections 5-9 derive
the limits this puts on code. The aim is the least whole-call time and the most useful bytes per second.

This document uses plain prose as close as possible to Flesch-Kincaid grades 4-6. A section may fall above or below
that range. Explain the hardware first, then the work it requires, then the choices that work rules out.

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
span the same number of 4 KB credit frames. A credit frame is how the driver counts queue space. It is not a rule
that all bytes in a page must be copied. Both ends fix the bytes to move and the same credit count before the call.
Each request stays within one registered range. The hardware holds the SEND in credit until that receive is posted,
and it sends no acknowledgement (TN3205; design/transport.md section 1). Each request names one range (max_sge 1). A device takes
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

## 5. Why pages and compilation

The driver needs a place for bytes to land before they arrive (section 3). We can settle that choice before the
call. The receiver posts to that place, and the sender sends when its data is ready. No peer needs to ask where
the bytes should go or wait for permission to send them. This is how SEND and RECV give us one-sided writes.

**Pages name the places.** Every byte a program sends belongs to a named page. A page table binds that name to
storage on each node. The storage is already there. The bindings are fixed before the call starts. The program
uses an index to find its page.

Each node has its own map. Nodes may have different page sizes, data types, and layouts. A page's type defines
what its indices mean. The compiler maps each part to the right local page. If values need a type change, the
plan names that step. An address on one node does not name storage on another node.

A write names source and target pages. A mask selects parts within one page, using indices in that page's type.
It cannot name another page or reach outside its page. A program cannot give a write a raw pointer and byte
count in place of a page and a mask. A move across pages names each page in its plan. This does not require a
separate device call for each page.

These limits let us send just the parts we need while keeping each write within its named pages. A value used
as a mask cannot turn the write into a move to some other place in a peer's memory.

**Compilation settles the choices.** Before the call, the compiler fixes the work shares, page bindings, masks,
and routes. It records which results each step needs. A solver may need a hard search to choose a good split.
The result is a plan that names pages by index.

A hard search and a simple rule can yield the same page list. Reading that list takes the same work in either
case. More pages can mean more work. A harder search does not make each lookup harder. The running program uses
the result; it does not repeat the choices that found it.

**One-sided writes spend less on each move.** Take one page going from A to B. There are three ways to place it:

| Method | What crosses the link | What B does after arrival |
|---|---|---|
| Place fixed ahead | The data | Uses the page where it landed |
| Ask first | A request, an answer, then the data | Uses the agreed page |
| Send a header with the data | The header and data | Reads the header, picks a page, and copies the data there |

The last two forms still need a posted receive for their first message. They add a choice after the first place
was fixed. Asking adds two crossings. Staging adds a copy, which reads and writes the bytes again. Sections 1-4
show the costs of those actions on this hardware. The page plan lets us avoid them at every hop.

Moving a page across the mesh can even beat a local NVMe read: a small transfer can finish while an SSD read
still waits for its first byte (sections 2 and 3; Infiniswap). A relay adds a hop, but it need not add a request
or a staging copy. A node can also compute with partial results as it sends them on. Its compute units and DMA
engines share the same memory (section 1). These paths let us share both copies and calculations across nodes.

## 6. Sharing work through one-sided writes

Nodes have different strengths. One may compute faster; another may have more links or more spare bandwidth.
We can devise rules to share copies and calculations among them. A fair split gives each node a share suited
to its rates, so the parts finish near the same time.

A routing function needs to know what must move, which pages hold it, and which links join the nodes. It also
needs their rates and which results each step needs. These are inputs to the function. It returns work shares
and a plan of page writes. A node's share of calculations and its share of relay traffic can differ.

Some rules let us split a message across a short path and a path through more neighbors. Each path gets a
share suited to its delay and rate. A relay passes each part on as it arrives. The parts can finish together
even though they took different numbers of hops. Each hop is still a one-sided write to a page fixed ahead.

Other rules tell us when work sent back by our peers means they have finished with a page. We can then reuse
that page or start the next calculation. The work they send back already tells us what we need to know. We
do not ask for a separate receipt. These rules are part of the plan, just like the rules that split a message.

Compilation turns these rules into page lists and steps. Once the plan is set, new input values can follow the
same writes. A function whose answers depend on those values can still keep its pages, routes, and work shares
fixed. The running call needs only that plan and its inputs. Each step reads its named inputs and writes its
named results when they are ready. It asks no peer to choose a place or grant permission between steps.

This leaves fewer ways to go wrong. A mask stays within its page. A write has its place before it starts. A
step waits only for the results it needs. We avoid choices and timing cases that would otherwise need checks
while the work runs. More time and link capacity are left for the work itself.

A compiled plan can still give us a poor split. Results from whole calls let us adjust shares and routes for
later calls. Compare both whole-call time and useful bytes per second. Each change becomes a new plan before
the call that uses it. We can keep making better choices without adding requests and replies to each step.

## 7. What the bridge must have

Each requirement follows from the hardware path and the page plan above.

**Register once.** Register the region when the port opens (section 2). Every page used by a call lies in that
region. A data call does not pin memory or register another buffer.

**Post ahead.** The plan fixes where each receive lands and its order (sections 3 and 5). The bridge posts as far
ahead as its queue space and the planned page uses allow. A fresh target page need not wait for an unrelated local
write. The plan also sets when shared results let us use a page again (section 6).

**Keep addresses local.** Before the call, the bridge binds pages and masks to local device addresses and byte
counts. Programs cannot supply these as write operands. The plan may split or group device transfers. They must
still cover only the bytes selected by the page masks. No header tells the bridge to pick a new page during the call.

**One writer per device.** The bridge posts on the device and tracks its queue space (section 3). Data programs
cannot post around it or inspect its private queues. The bridge passes pages onward through DMA; its CPU does not
copy their payload.

**Keep links across calls.** Queue pairs belong to the link's life, not to one data call. Make them when the link
opens. Keep their setup and teardown out of the data path. The driver limits how they can be flushed or released
(design/transport.md, sections 5 and 14).

**Do no extra work per step.** The page plan already holds the facts needed to place the bytes. Do not add a search,
a copy, a request, or a guard to rediscover them. The hardware costs in sections 1-4 make each such step costly.

## 8. Writes over many hops and many paths

A long transfer can use direct links and links through other nodes at once. The same page plan covers all of them.
Each hop lands at a place fixed before the call. Each relay sends from that same memory as its input arrives.
It does no CPU copy and asks no peer where the bytes should go.

**Hops can overlap.** Piece one moves onward while piece two arrives. More hops add startup delay. That delay also
counts toward the end of a finite transfer. Once the path is full, its slowest link bounds its rate. Other traffic
on that link shares the same capacity. A relay needs one DMA write and one DMA read per byte (section 2).

**Paths share physical links.** Two paths that use the same cable share its bandwidth. Drawing a second path does
not add a second cable. A cut is a set of links that all traffic between two sides must cross. The sum of those
links' rates bounds the flow across that cut (Ford and Fulkerson). The source's outgoing links and the target's
incoming links are two such limits.

The compiler uses the connected links and their capacities. It must not assume two nodes, one cable per pair, or
equal ports. Several transfers may use the same links. Plan their work together so each link's capacity is counted
once. A relay is a node that passes the data on unchanged. Its pages and links are part of the same plan.

**More paths can shorten the transfer.** Take two paths that share no cable. They each move B bytes per second.
One starts at time L. The other starts at time L plus d, due to its extra hop. A transfer of S bytes on the direct
path takes L + S/B. If S is greater than B times d, some bytes can use the other path and both paths can finish
sooner. The split gives more bytes to the path that starts first.

This example explains why an extra hop can help. It is not a rule that bars a path below a fixed size. The real
plan includes all paths, their shared links, page masks, and data needs. The solver must be able to shift writes
among them. Its goal is to make the whole call finish sooner and raise its useful bytes per second.

**Compute shares and routes change together.** A node can do less arithmetic and relay more data. DMA relays use
little arithmetic, though they still use links and memory. Do not infer their cost from a node's compute share.
Use the measured gaps between nodes' finish times to adjust the plan between whole calls.

Hop costs are measured too. Start nodes with the same estimate. Record both the mean delay and its spread, then
update them from new runs. A busy link or a slow estimate is a reason to improve the plan, not to hide that link
from the solver.

The bridge moves data as fast as the plan and hardware allow. The collective sets the work shares and routes.
Data code declares its work through the collective. It does not manage links or add a second transport path.

## 9. Any collective as compiled one-sided writes

A complete call has known shapes and a known graph of steps. These are fixed before it starts (design/rubrics.md,
D2). A library can use those facts just as code written for one program can. Its compiler must turn them into page
bindings and a write plan, rather than leave their meaning to be worked out during each step (section 5).

A plan's being compiled does not make its routes or work shares best. It makes the chosen plan cheap to enact.
Later whole-call results guide the next plan. Within a call, the work shares, page bindings, and routes stay fixed.
Every collective and solver uses this same page-and-mask model.

**Gather and scatter.** In an all-gather, each node's share goes into its place in every output. Shares may have
different sizes. Their pages and masks fix where each part goes. In a reduce-scatter, each node sends its partial
results to the nodes that own the result pages. A node can sum partials as they pass through it.

**Reduce and broadcast.** A reduce sends partials toward the root and sums them on the way. A broadcast sends a
root's data to all the other nodes. An all-reduce can reduce-scatter first, then all-gather the sums. This can move
fewer bytes than sending every full partial to every node. Its two phases still use the same one-sided writes.
The compiler must express either form and compare whole-call costs.

**Expert routing.** Each expert's input rows go to the node that holds it. The results go to the pages that need
them. Counts may be fixed, or the plan may reserve space for their known bounds. A mask selects the used parts
within each page. No peer must pick a new destination after the data arrives.

**Attention.** A ring can pass key and value blocks among nodes that hold query blocks. Another form can gather
query blocks and reduce-scatter partial results. Each form has a known sequence of writes. The compiler must use
that sequence and its real data needs, rather than add a barrier for the whole operand at each step.

**Tensor splits.** A column split gives each node a share of a layer's outputs. A row split gives it partial sums
that must be combined. The next step's needs fix which result pages go where. The same rule applies when nodes
have different compute rates, memory rates, or numbers of links.

**Results guide later calls.** Measure the time and useful bytes per second of the complete function. Count output
data once; extra relay hops do not create more useful output. Use those results to adjust work shares and routes
between calls across the measured shapes. Each new choice becomes another compiled page plan.

The records arrive on their own. Late peer records can describe several past calls at once. Each record names the
call and plan it measured. The solver uses them when they arrive. Their absence does not stop data execution or
make peers meet at a start or finish barrier. A new plan is bound before the call that will use it.

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
