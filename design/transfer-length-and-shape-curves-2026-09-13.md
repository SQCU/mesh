# Retained transfer lengths and measured setup choices

## Message length belongs to the transfer

Apple's [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
requires matching sender/receiver frame counts for a corresponding message.
It does not require a single payload length for the whole program. Hardware
queue capacity is measured in 4096-byte frames, and the configured queue's
actual capacity must be queried. Receive credits permit posting a payload
before its matching receive is posted. These facts support the asynchronous
index-push mechanism cited in
[algorithm sources](algorithm-sources.md#async-index-push-contract).

The numerical copy binding already knows the logical shape, scalar size, and
block offset. It now retains that useful byte length in `mesh_row_binding.bytes`.
`mesh_realize` divides the binding into its configured ownership blocks and
retains each block's frame-rounded transport length in `mesh_transfer.bytes`.
The setup exchange compares the paired lengths; index descriptors and posted
WR records carry the retained length to the actual send and receive SGEs.
No length is recovered from an allocation's padded size.

Ownership and mapping keep their independently allocated full blocks. A short
message uses a prefix of those same registered pages. Padding beyond the final
4096-byte frame is not transmitted; no dense staging array or subpage virtual
mapping is introduced. Publication retains the existing ownership range, whose
logical tensor shape excludes storage padding. A 64-by-128 FP32 section therefore
uses a 32768-byte message instead of sending its 65536-byte allocation. This is
a source-derived byte count, not a measured latency gain.

Bindings cannot claim more useful bytes than their storage or include trailing
ownership blocks with no useful data. The maximum configured block remains the
MR-boundary unit, so a shorter SGE stays within its already validated region.
A transfer descriptor is 40 bytes; 102 descriptors fit with the eight-byte header
in a 4096-byte index frame.

## Queue-depth scope

This change keeps the existing conservative capacity expressed as a count of
maximum-sized payloads. Every shorter payload fits that bound, and completion
arrays and pending vectors retain their existing capacity. This removes wire
padding without silently increasing the possible outstanding WR count.

A later depth change must retain actual per-QP frame capacities, sum the frames
of posted and announced requests, and decrement each stored request's frame cost
at completion. The completion queue and its result array must accommodate the
number of possible shorter WRs; dividing by the old maximum block would no
longer be correct. Index messages each cost one frame. Software entry capacity,
hardware frame capacity, and useful byte length are separate quantities.

## Retained large-shape curves

The following are existing measurements, read without running workloads. They
inform setup for their documented operations, shapes, precision and hardware.
They do not predict the tiny FP32 swish chain's timings.

The September 8 [kernel survey](../../../metal-microbench/docs/data/kernel_survey_2026-09-08.json)
contains full FP16 FFNs with hidden width 3840 and intermediate width 15360:

| Rows | M5 MPS invocation median (ms) | M4 MPS invocation median (ms) |
| ---: | ---: | ---: |
| 64 | 0.912708 | 4.658333 |
| 512 | 3.581792 | 31.624375 |
| 1024 | 6.664750 | 62.935875 |

The [backend record](../../../metal-microbench/docs/soc_compute_backends.md)
documents a matched 1024-by-3840-by-3456 FFN at 1.633 ms on M5 MPS and an
improved direct-I/O M4 CPU/ANE invocation median of 5.244708 ms. Its projection
curves favor configured 128-row native sections on M4; the measured 64-row and
256-row alternatives were slower for that operation. M5's corresponding
projection favors MPS. Placement is resolved before execution and preserves
those distinct local numerical implementations.

The September 11 [down-extent record](../../../metal-microbench/docs/mesh_f1b_down_extent_2026-09-11.md)
retains this M5 curve for the documented 4096-row FFN and 11904/3456 neuron split:

| Configured down rows | Down GPU time (ms) | Chain median (ms) |
| ---: | ---: | ---: |
| 128 | 7.78 | 28.71 |
| 512 | 7.48 | 27.10 |
| 1024 | 6.85 | 26.36 |
| 2048 | 6.34 | 26.18 |
| 4096 | 9.47 | 29.59 |

Those results support independently completing nonterminal sections while
retaining efficient local contractions. They do not justify collapsing the
operation to one terminal publication or transplanting those exact sizes into
a different computation. The current CPU gold chain keeps its Accelerate
contractions as the CPU baseline. No matching retained curve establishes an
accelerator advantage for that small FP32 shape.

No benchmark harness, trial configuration sweep, or performance percentage was
introduced by this review.
