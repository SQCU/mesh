# Facts collapsed to one value: the instances, and what this hardware lets us do instead (2026-10-31)

A fact that holds per page, per range or per link, carried as one value per call, stream or channel, cannot say what
the set says. Something then gets copied, serialized or refused. The remedy is the same each time: carry the tuple,
resolve it where it is used, and compile it into the final lowered form, a register or a table read once, not a
branch taken at run time. Below, each instance with its cost as measured and a recommendation in terms of the machines
we have.

## The machines (sysctl on the nodes, mesh-stat; measured rates from docs/facts.md and bandwidth.py)

| | Studio (m5ult1-4) | Mini |
|---|---|---|
| CPU | Apple M5 Ultra: 10 + 20 cores (hw.perflevel0/1) | Apple M4 Pro: 8 + 4 |
| A core's registers | 31 x 64-bit general, 32 x 128-bit vector (AArch64); SME2 and BF16 (hw.optional.arm.FEAT_SME2, FEAT_BF16: 1) | the same |
| A core's L1 | 128 KiB data, 192 KiB instruction; 128-byte lines | the same |
| L2 | 16 MiB per cluster of 5 | 16 MiB per cluster of 4 |
| Memory | 96 GiB, 82 GiB of it the bridge's region | 24 GiB, 16 GiB region |
| GPU | 64 cores | 16 cores |
| A cable | Thunderbolt RDMA, about 8.6-8.9 GB/s a direction in steady all-to-all (docs/facts.md) | the same |

Loose costs to reason with: a register read is free in the pipeline; an L1 hit a few cycles; L2 tens of cycles;
memory on the order of 100 ns; a small message from one node's region to another's tens of microseconds through the
bridge (bandwidth.py's 64 KiB broadcast: 17-21 us); a megabyte about 0.12 ms on one cable, less striped over a pair's
cables. So any planning or scheduling problem whose state fits a few KiB fits in one core's registers and L1, and
one whose state is megabytes moves to any of five machines and back in about a millisecond. A second of host time
anywhere in a service is three to six orders of magnitude off what these machines do.

## The instances

1. **A sum's own operand, one (buffer, offset)** (nccl-mesh-metal.h struct metal_zelloth_sum). Its pages lie in two
   places where earlier steps wrote part of its range, so the library copied the whole operand into one place first:
   bandwidth.py's reduce-scatter made 5 copies a call. *Done, ec69f367:* each run of a sum names where its own pages
   lie; a block reads its run's base into a register once, its threads stream from it. The sources are fixed at plan
   time; nothing is looked up per element. Reduce-scatter 1 MiB 178 -> 145 us, no copy made by any call.
2. **`direct`, one bool per call** (nccl-mesh.c plan_direct). A per-range fact (is this range read in place?) decided
   for the whole call: one step's range part written staged the whole operand. *Partly done, ec69f367* (a combine over
   a part-written range now stays direct). *Recommended:* the same for a SEND over such a range: its entry's spans are
   already a list (struct mesh_entry `spans`), so each span can name the buffer its bytes lie in; nothing forces one
   source per step. The in-place all-reduce and broadcast are still refused for their overlap and copy nothing, but
   lose the direct publication: their overlap is exact aliasing, a relation binding_relations already computes.
3. **`spans_meet`, a range as {-1, 0, 1}**, and **`read`/`fresh`, one map per step**. A part-written range can only
   say "refuse". *Recommended:* carry the written set's intersection (a few intervals; at most a step's count) to
   the steps that need it, as zsum_owns now does for sums.
4. **A landing's cross-issue `after`, a channel mask with one base a call** (f4086e35). Every landing waited for the
   latest SEND over any of its issue's pages. *Done, 76de94ce:* each landing's own list of requirements, the form a
   SEND's already had; one decode in the bridge.
5. **A channel is one cable** (struct channel `cable`, `wire`; a transfer's one cable index; routes as cable lists).
   A node pair's edge, with all its cables and relay paths, became the cable a piece was bound to, so the program's
   order became each cable's queue pairs' order (design/zorvek.md, design/transport.md section 2). *Recommended, the
   largest:* the collective names node pairs and positions in its rows; the bridge, which alone owns queue pairs,
   stripes a node pair's stream over every queue pair of every cable by a rule that depends on neither data nor
   program. The bridge already stripes a stream over its two lanes by unit index; the rule extends to all lanes of
   the pair, weighted by their measured rates, and terminates because each lane's sends and receives stay in unit
   order.
6. **A stream's `ready`, one prefix count stored by whichever publication runs** (nccl-mesh-kernels.h
   mesh_step_release). Correct only if publications execute in encoding order, which a schedule whose "encoding
   order is not execution order" (metal-recording.m) does not promise. *Recommended:* raise it with an atomic maximum
   (one instruction on these GPUs, as the fake-guards review proposed), so publications may complete in any order.
7. **MetalSchedule's after-scope, one queue-wide wait** (metal-recording.m MetalScheduleOrder). The first part that
   carries it holds every part committed after it, whether or not that part reads the earlier stream's data.
   *Recommended:* the wait on the parts whose declared reads meet the earlier stream's writes; the frontier that
   decides that is already kept.
8. **A configuration's observation record, one 4.4 MB line before the program starts** (mesh_statistics.py
   Observations). 3.3-6.4 s a rank through its pipe, the chain service's start skew past the 5 s a GPU wait is
   bounded. *Done, b4ab4394:* written at the first report; the ranks now start within 0.1 s. *Recommended:* the record
   itself is a structure every rank already holds: name it by its key, as completion lines already do.
9. **The region's length, a function of the structs' sizes** (mesh.h mesh_layout). A layout change needed a second
   region beside the first; with an exiting child still wiring the first, the node's memory ran out (seven resets
   since 10-29). *Done, 378dbd5f, 07949b28:* one region object a node, every layout fitted to it.
10. **Preparation as interpreted host work.** The chain's recording takes 2.0 s for 12,337 dispatches (about
    160 us each: a dictionary-keyed frontier per command, metal-recording.m), and bind deep-copies the plan's graphs
    twice (0.9 s, 1.6M calls). *Recommended:* the recorded schedule is a table of fixed-width rows (dispatch, bindings,
    reads, writes): its frontier a sorted interval array a core walks in L1 (12,337 rows of a few words each are well
    under 1 MiB, one core's L2 share), its dependencies computed in one pass, a few microseconds a command at most and
    parallel across a node's 30 cores where it is not; the plan's graphs shared, not copied, being immutable once made.
    A prepared program then starts milliseconds after its declaration, and a problem too large for one core's caches
    goes to an idle node's in about a millisecond a megabyte.
11. **A GPU wait's deadline, one bound fixed when the wait is made** (design/eager-execution.md A4). It assumes every
    rank starts the program together; host skew counted against it. *Recommended:* nothing in the wait. The program's
    start is the place: once preparation is milliseconds (10), the ranks that leave preparation's exchange together
    start together.
