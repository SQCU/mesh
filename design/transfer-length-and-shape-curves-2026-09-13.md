# Retained transfer lengths and measured setup choices

## Message length belongs to the transfer

Apple's [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
requires matching sender/receiver frame counts for a corresponding message.
It does not require a single payload length for the whole program. Hardware
queue capacity is measured in 4096-byte frames, and the configured queue's
actual capacity must be queried. Receive credits permit posting a payload
before its matching receive is posted. These facts support the asynchronous
index-push mechanism cited in
[algorithm sources](algorithm-sources.md#programcopy).

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
