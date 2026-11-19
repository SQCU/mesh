# One publication per result range, one transfer per destination

## A: Two readers caused two sends

Studio computed `y = sin(.75 * x)`. Mini called two functions that read the same 32 KiB of `y`. The compiler put a send in each reader's call. Neither call wrote `y` on Studio, so its recorder waited for the preceding GPU work to finish and published the whole tensor from the host. Mini read the same value twice and Studio sent it twice. The bridge already knew the two 16 KiB ranges in each transfer. [Earlier recording](metal-transfers-between-calls-2026-11-18.md) · [Two-host baseline](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-fanout/before-1795049031.log)

## B: The numerical writer publishes each completed range

Placement gives the call that writes `y` the outgoing transfer. The Metal recorder binds the transfer's source pages and completion words to that numerical kernel. The last group that writes a range signals its completion. The bridge can send those bytes while other groups still have work. The receiver can compute with the arrived pages that its operand requires. A reader that needs the whole tensor waits for all its ranges. [Placement](../../../compiler-emit/compiler_emit/source.py) · [Metal publication](../../../metal-microbench/metal_transport.mm) · [Consumer properties, R4](plan-to-bridge-change-request.md#r4-consumer-properties-determine-graph-edges-and-fusion)

Placement groups reads of the same produced version, tensor view, destination and collective. Both functions on Mini read the same landing pages. Each retains its own dependency on those pages. A write between the reads produces a new version. The allocator must give any simultaneously live versions distinct storage. [Reference placement](../../../schedule-emit/schedule_emit/placement.py) · [GPU operand binding](../../../compiler-emit/compiler_emit/endpoints.py) · [Allocation and lifetimes, R2](plan-to-bridge-change-request.md#r2-allocation-binds-every-value-to-pages)
