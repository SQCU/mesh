# The data model

These sections say what the hardware is. Each number names its source. The files measurement.md and facts.md and the
folder output_data/ are in the metal-microbench repo. The papers are listed at the end.

The hardware paths explain how we share work through one-sided writes. Sections 5 and 6 connect that work to
page lists and compilation.

This document uses plain prose as close as possible to Flesch-Kincaid grades 4-6. A section may fall above or below
that range. Explain the hardware first, then how we use it to share work.

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

## 5. Sharing work through one-sided writes

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

This leaves fewer ways to go wrong. A mask stays within its page. A write has its place before it starts. We
avoid choices and timing cases that would otherwise need checks while the work runs. More time and link
capacity are left for the work itself.

A compiled plan can still give us a poor split. Results from whole calls let us adjust shares and routes for
later calls. Compare both whole-call time and useful bytes per second. Each change becomes a new plan before
the call that uses it. We can keep making better choices without adding requests and replies to each step.

## 6. Page lists and steps

The driver needs a place for bytes to land before they arrive (section 3). The memory in section 2 is already
set aside. We give each part of that memory a page name. A page table tells each node where to find a page by
its name.

A page list names the pages a step reads and writes, in the order it uses them. For a send, it names a page on
the source node and a page on the next node. Each node uses its own table to find its page. The pages can be
stored in different places on the two nodes. A mask picks values within one page, so we can send just those
values. It cannot name a place outside that page.

Our PyTorch code can already record a whole function as a graph of steps before it runs. The graph shows
which results each step needs. It can include the work that finds an answer and the work used to train the
model. With that graph, we can choose which nodes do each part of the work before we run it.

We turn those choices into page lists and steps. This is compilation. A hard rule and a simple rule can give
the same list, which takes the same time to read. The work spent choosing the list is done before the call.
Each step then finds its data through the list.

The receiver uses the lists to set the places where data will land (section 3). The sender can then write
there as each part is ready. This gives us one-sided writes. A node can use a part or send it on from that
same page, since its compute units and links share memory (section 1).

We can use a good plan for many sets of inputs. If we change who does which work, we compile new lists and
steps before the next run.

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
