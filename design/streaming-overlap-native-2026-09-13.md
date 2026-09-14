# Native streaming contraction-chain comparison

September 13, 2026. Source migration of the existing
the historical, now-removed `examples/streaming-overlap.py` harness.

## Corrected source paths

Previously the producer's streamed output had a `(tile, width)` partition but
its peer receive tensor had a whole `(rows, width)` partition. `Program.copy`
correctly rejected these mismatched layouts. Rank-dependent tensor allocation
also made the two descriptions differ. The numerical callback called NumPy in a
Python loop and inspected full-input presence from inside the computation.

Both ranks now construct the same tensors and native `kernel_call` grids. The
`peer` argument determines which participant binds each native matmul. Receive
and return storage use their source tensor's exact shape, block shape and dtype.
One data queue handles both directions, matching the ordinary one-QP bridge.
All returned sections have independent exported results. Ragged final row
regions use the existing clipped BlockSpec resolution and exact matching peer
allocation. No temporary packing or dense operand staging is added.

CPU executes the existing native CPU matmul and Metal the existing MPS matmul.
The old Python numerical callback, handwritten inner row loop, and in-function
presence/timing counters are gone. Setup constructs `depth` complete slots;
`trials` reuses that bounded storage after each slot's prior output has been
observed and consumed. Canonical writable state controls external input supply,
not numerical invocation readiness. No new task scheduler is part of the graph.

## Measurement meaning

The positional `rank` and `whole|streamed` modes and rows/tile/trials/depth flags
remain. `--backend cpu|metal` selects setup binding; `--trace PATH` retains the
existing compute and transfer traces. The receiving participant writes its
trace after ordinary SIGTERM/SIGINT. Whole mode explicitly compares one
whole-region contraction to independently published row-region contractions;
it is not an alternative production streaming interface. Different matmul
shapes can have different local kernel efficiencies, so a comparison must report
those costs rather than assign all differences to network overlap.

Warmup drains before the measured batch begins. The measured duration includes
external input writing, output observation and slot recycling. Reference
computation and terminal numerical comparisons lie outside that duration.
Only terminal invocations are numerically checked, as in the prior example;
the reported checked count makes this limit explicit. First-section timings are
host observations, not exact native publication times. Native traces supply
compute/input identities and transport events without instrumenting the matmul
body. Traces retain only each function's latest occurrence; analysis must not
combine different slot generations as if they belonged to one invocation.

The host input generator and observer keep their finite invocation bookkeeping;
mesh retains ownership of numerical readiness, transfer progress and scheduling.
The source allocates operand storage only while configuring the fixed slots.

## Validation
