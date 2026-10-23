# The hardware, in the operator's words and in measurements

The operator's five statements of 2026-10-22, each followed by the measured facts, the literature and the mechanisms
behind it. The operator's framing, the same day: "developing for this kind of backend feels only like restoring bad
hardware to surprising mediocrity rather than something qualitatively different to tuning dot cuh designs".

## 1. The hardware is unusual
"the hardware is weird and not like arm64 cpus or older apple silicon."

Compared with CUDA sm90 and sm100:
- **One cache level fewer.** The GPU's caches sit in front of a system-level cache shared with the CPU and the
  Thunderbolt RDMA engine, then DRAM. It is much less cache-congested than an HBM GPU.
- **Very low FLOPs and bandwidth relative to its memory, which is large.** Strong out-of-order CPUs, which can compile
  and plan at the node.
- **One memory.** CPU, GPU and the RDMA engine address the same DRAM. There is no device memory, no BAR, and no PCIe
  peer-to-peer path. The bridge registers its whole region once, so every page a program owns is already addressable
  by the NIC.

Measured on these machines:
- A GPU's plain stores reach the NIC only once their command buffer ends. System-coherent stores and loads reach it at
  once (metal-microbench docs/measurement.md, 2026-09-29).
- Data exchanged between threadgroups within one dispatch needs `coherent(device)`. Plain memory held on the M5's one
  die and failed on the M5 Ultra's two (mesh ebdef267).
- A dispatch boundary costs about 1–2 µs, and starting a command buffer about 13–19 µs. The GPU watchdog ends a command
  buffer after about 5 s.
- An M5 Ultra device holds 2 usable queue pairs a link.

With a dispatch boundary at 1–2 µs and a command buffer's start at 13–19 µs, a step whose compute is short is mostly
fixed overhead. The operator, 2026-10-20 (design/rubrics.md M8): "if there is 10ns of reducible overhead anywhere in our
code we reduce it to 0ns."

## 2. DMA is fundamental
"dma is fundamental and requires exceptional programming practices to make useful collectives from weak silicon."

The RDMA engine's contract (Apple TN3205):
- UC queue pairs, SEND and RECV only.
- A SEND fills the oldest receive posted on its queue pair, of the same frame count.
- A credit holds a SEND until its receive is posted.
- Every work request ends with a completion and a status. That completion is the only statement that bytes moved, and
  its error status is the only statement that they did not.

The operator's words on the practices, with their rubric dimensions:
- 2026-10-21: "zero copy methods must always be used over access by copy because memory bandwidth and flops are too
  scarce on this hardware api."
- M2 (2026-10-19): "if you have successfully 'asked permission' to send data" ≺ "you could have sent the data instead";
  receives on shared lanes driven by a header ≺ a pre-posted exclusive receive (2026-10-17).
- M9 (2026-10-18): "an in-order recieve protocol" ≺ "multiple concurrent streams which get written into the right
  expected output slots in mapped memory". Data movement is compiled from declared extents
  (design/extents-and-liveness.md).

## 3. The level of a streaming multiprocessor
"weak silicon is programmed at a similar level of abstraction to streaming multiprocessors, not abstractions attempting
to wrap entire gpus or nodes."

- M1: "the 'bridge' is not a grab bag of arbitrary ontologies anyone wants to put anywhere" (2026-10-19) ≺ "the bridge
  is there to behave like a system call all operating systems should have always been giving us" (2026-10-15).
- The operator, 2026-10-22: "data procesisng code gets to write their own guards in their own source. if source says to
  write continuously into ab uffer, the nic must write continuously into that buffer even if this means that dat
  aprocessing code can race or lose parts of a stream."

## 4. Different trade-offs from NCCL
"we economize on different tradeoffs from nccl, and use a more constrained (but still universally expressive and easy to
use) memory model in exchange for strictly fewer waits, copies, and protocol dependency relation depths. this is not
'nice-to-have' but is needed for baseline performance."

**NCCL.** NCCL owns staging buffers, runs a proxy, and copies through reduce-copy kernels in stream order, on CUDA GPUs
with another vendor's NIC: device memory sits behind PCIe, registering arbitrary buffers is costly, and NIC writes need
flushes.

**Here.**
- The memory model is a page table of rows, stamps and use counts (design/pages-and-functions.md), with extents
  declared when a program is called and plans compiled from them.
- A not-in-place reduction's extra pages are rows chosen in the plan and landed into in place. There is no staging pool
  and no ring with copies.
- On the collective's page facts, the operator, 2026-10-22: "data processing code may never discover these abstracted
  features"; "information hiding from data processing code authors on some level must exist for performance reasons and
  for decoupling backend complexity from linear algebraic function complexity rather than for 'security' or 'memory
  safety'".

## 5. Data semantics and the transport
"bridges and collectives are expressive enough to allow data corruption wrt a data producing or consuming program. we
never do anything to prevent what might be a data bug for some semantics of data. data processing lcients might be a
guitar pedal with a realtime guarantee or they might be a deferred data processing cron job. programmers have to write
their own syncs and guards or they get the fastest resutls possible."

- The transport writes landings as the plan presents them (design/transport.md section 14).
- The operator, 2026-10-22: "performance is harder than safety and produces safety anyways most of the time, and we're
  requiring performance rather than safety."
