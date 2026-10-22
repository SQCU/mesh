# The hardware, and the discipline it requires

Five points every agent must understand before changing the bridge, the collective or a program that uses them. The
operator stated them on 2026-10-22; each section gives the operator's words, then the measured facts behind them.

The operator's framing: "developing for this kind of backend feels only like restoring bad hardware to surprising
mediocrity rather than something qualitatively different to tuning dot cuh designs". Good compute graphs are good
everywhere. Placing data well across shared memory, SRAM and registers is good discipline on every CUDA device where
the trade can be made. This hardware only makes the price of doing it badly higher.

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

The consequence: a small step's compute is short, so a step is mostly fixed overhead. Every nanosecond left in a hot
loop is a direct share of utilization.

## 2. DMA is fundamental
"dma is fundamental and requires exceptional programming practices to make useful collectives from weak silicon."

The RDMA engine's contract (Apple TN3205):
- UC queue pairs, SEND and RECV only.
- A SEND fills the oldest receive posted on its queue pair, of the same frame count.
- A credit holds a SEND until its receive is posted.
- Every work request ends with a completion and a status. That completion is the only statement that bytes moved, and
  its error status is the only statement that they did not.

Useful collectives come from:
- zero copy;
- data movement compiled from declared extents (design/extents-and-liveness.md);
- receives posted ahead of their SENDs;
- system-coherent stores wherever a SEND reads;
- completions as the only signal.

## 3. Program at the level of a streaming multiprocessor
"weak silicon is programmed at a similar level of abstraction to streaming multiprocessors, not abstractions attempting
to wrap entire gpus or nodes."

- The bridge is a thin layer of hardware facts and posts.
- Collective ops and the kernels that use them write their own guards, explicitly, in their own code, as SM code does
  with barriers, fences and flags.
- Nothing below a kernel makes up for a guard its author left out.

## 4. Different trade-offs from NCCL
"we economize on different tradeoffs from nccl, and use a more constrained (but still universally expressive and easy to
use) memory model in exchange for strictly fewer waits, copies, and protocol dependency relation depths. this is not
'nice-to-have' but is needed for baseline performance."

**What NCCL is built for.** NCCL owns staging buffers, runs a proxy, and copies through reduce-copy kernels in stream
order. That is right for a CUDA GPU with another vendor's NIC: device memory sits behind PCIe, registering arbitrary
buffers is costly, and NIC writes need flushes.

**Here.**
- The memory model is a page table of rows, stamps and use counts (design/pages-and-functions.md), with extents
  declared when a program is called and plans compiled from them.
- A not-in-place reduction's extra pages are rows chosen in the plan and landed into in place. There is no staging pool
  and no ring with copies.
- Data code does not see the collective's page facts. That hiding exists for performance and to keep backend
  complexity out of linear algebra, not for safety.

## 5. The transport never protects a program's data semantics
"bridges and collectives are expressive enough to allow data corruption wrt a data producing or consuming program. we
never do anything to prevent what might be a data bug for some semantics of data. data processing lcients might be a
guitar pedal with a realtime guarantee or they might be a deferred data processing cron job. programmers have to write
their own syncs and guards or they get the fastest resutls possible."

- The transport writes landings as the plan presents them (design/transport.md section 14).
- A program that reuses a buffer guards it in its own source, or races.
- No gate, sync, retry, deadline or error correction in the bridge or the collective stands in for a program's own
  guard.
- Performance is the requirement. The operator: "performance is harder than safety and produces safety anyways most of
  the time".

## Before changing the bridge, the collective or a program that uses them
- Read design/pages-and-functions.md, design/extents-and-liveness.md, design/transport.md section 14, and design/lint.md.
- Run `make -C rdma lint`. A new class of defect gets its rule in the same commit as its fix.
- Use the literature's terms (design/lint.md, "Vocabulary"). A coined term usually marks a misunderstood API.
- Where a method strictly dominates by the literature, it is required. Only a real trade-off is chosen by measurement
  (design/extents-and-liveness.md, "Dominating methods and choices").
