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

## 5. Peak and what is reached

An H100's peak is far above a Studio's. But on real work it reaches only part of that peak.

Its 50 MB L2 is split in two halves. A load from the far half takes nearly twice as long as one from the near half. Read
across all 50 MB, L2 gives 3.8 TB/s, less than an A100's 4.5 TB/s (Chips and Cheese 2023). An L2 hit takes about 273
cycles, and about 508 when both halves are busy (Jarmusch et al. 2025). A B200 splits its L2 across its two dies.
Atomics take 90-100 ns on the near side and 190-220 ns across (Chips and Cheese, B200).

Dense math hits the power limit. A dense BF16 matmul on an H100 SXM holds about 680 TFLOPS at 700 W, with the clock near
69% of its top, against a 989 TFLOPS peak. Random inputs flip more bits than zeros, so they draw more power and slow the
clock more. Under long load some H100s have dropped from 1,755 MHz to 450-550 MHz (Spheron 2026). Where it fits,
sparse beats dense: 2:4 sparse FP8 reached about 1,900 TFLOPS against 1,500 dense at 16,384 cubed. At small layer sizes
it was slower than dense (Hu et al. 2024; Fisher-weighted SVD study 2026).

Small steps that wait on each other are the worst case. At batch 1 an H100 used about 27% of its memory bandwidth on a
7B model, where an L4 used about 81%. The cause was a fixed CPU launch cost each step (Batch-1 decode study 2026).
Below batch 100, MFU stays under 20% (Chen et al. 2024). Each launch ends in a PCIe write to the GPU's doorbell, and the
GPU idles until it comes (Blink 2026). On an A100, one decode layer took 0.38 ms on the GPU and 1.14 ms on the CPU
(Adrenaline 2025).

So for a chain of small steps, the share of the roofline reached each step decides the speed, not the peak. On CUDA
that share falls, because each step pays fixed costs: the launch and its doorbell, PCIe hops, far-L2 trips, and clocks
held down by the power limit. A Studio has one memory, no PCIe or BAR, a dispatch of about 1 µs in an open command
buffer, and a crossing that lands about 3.6 µs after it is posted (sections 1-3). It keeps that share high only while
every part of mesh adds no fixed cost to a step: no waits, no copies, pages and their order fixed and posted ahead, and
one region registered once. One part that breaks this pays a fixed cost every step, as CUDA does.

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
- C. Lam. Nvidia's H100: Funny L2, and Tons of Bandwidth. Chips and Cheese, 2023.
- Chips and Cheese. Nvidia's B200: Keeping the CUDA Juggernaut Rolling.
- A. Jarmusch et al. Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks. 2025.
- Spheron. GPU Performance Consistency in the Cloud: The Hidden Clock Tax. 2026.
- Y. Hu et al. S-STE: Continuous Pruning Function for Efficient 2:4 Sparse Pre-training. arXiv 2409.09099.
- Mind the Approximation: Fisher-Weighted SVD Compression for ViTs. arXiv 2609.07155.
- Memory-Bound but Not Bandwidth-Limited: The Physical AI Inference Gap in Batch-1 LLM Decode. arXiv 2605.30571.
- S. Chen et al. Efficient Heterogeneous LLM Decoding with Model-Attention Disaggregation. arXiv 2405.01814.
- Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC. arXiv 2604.07609.
- Injecting Adrenaline into LLM Serving. arXiv 2503.20552.
