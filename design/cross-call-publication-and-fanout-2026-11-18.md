# One publication per result range, one transfer per destination

## A: Two readers caused two sends

Studio computed `y = sin(.75 * x)`. Mini called two functions that read the same 32 KiB of `y`. The compiler put a send in each reader's call. Neither call wrote `y` on Studio, so its recorder waited for the preceding GPU work to finish and published the whole tensor from the host. Mini read the same value twice and Studio sent it twice. The bridge already knew the two 16 KiB ranges in each transfer. [Earlier recording](metal-transfers-between-calls-2026-11-18.md) · [Two-host baseline](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-fanout/before-1795049031.log)

## B: The numerical writer publishes each completed range

Placement gives the call that writes `y` the outgoing transfer. The Metal recorder binds the transfer's source pages and completion words to that numerical kernel. The last group that writes a range signals its completion. The bridge can send those bytes while other groups still have work. The receiver can compute with the arrived pages that its operand requires. A reader that needs the whole tensor waits for all its ranges. [Placement](../../../compiler-emit/compiler_emit/source.py) · [Metal publication](../../../metal-microbench/metal_transport.mm) · [Consumer properties, R4](plan-to-bridge-change-request.md#r4-consumer-properties-determine-graph-edges-and-fusion)

Placement groups reads of the same produced version, tensor view, destination and collective. Both functions on Mini read the same landing pages. Each retains its own dependency on those pages. A write between the reads produces a new version. The allocator must give any simultaneously live versions distinct storage. [Reference placement](../../../schedule-emit/schedule_emit/placement.py) · [GPU operand binding](../../../compiler-emit/compiler_emit/endpoints.py) · [Allocation and lifetimes, R2](plan-to-bridge-change-request.md#r2-allocation-binds-every-value-to-pages)

## What two Macs measured

The public compiler ran the fan-out graph with two successive inputs. Both Mini readers matched eager FX on all 32,768 float32 outputs. The table counts outgoing payload at Studio across both invocations. Counting receive completions as well would double-count the same bytes. [Before](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-fanout/before-1795049031.log) · [After](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-fanout/f0ba23c/fanout.log)

| Studio publication | Physical 16 KiB sends | Bytes sent |
|---|---:|---:|
| Each reader requests the result | 8 | 131,072 |
| Both readers use the writer's transfer | 4 | 65,536 |

For the latency question, Studio published the first 16 KiB of an operand and withheld the remaining 16 KiB. Mini ran the compiled addition on the arriving data. Studio checked the first 4,096 results against `2*x+1` while the whole-result arrival word still read zero. Only then did Studio release the second half. All 8,192 final results matched. This measures a causal property. An end-to-end speedup depends on how much other work the program can overlap with the transfer. [Controlled two-host run](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-fanout/f0ba23c/causal-once.log) · [Evaluator](../../../compiler-emit/artifacts/joint-selection-review-20261118/evaluate-calls.py)

The mutation control puts `y.add_(.25)` between the two reads. An intermediate build allocated the old and new versions at the same destination address. The second reader returned 0.07623336 for its first element where eager FX returned −0.10472105. Giving the two versions distinct landing roots restored both reader outputs on both inputs. A separate run gave one reader page readiness and the other whole-tensor readiness. Both used one physical transfer. [Version collision](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-fanout/f2cc10a/fanout-mutation.log) · [Distinct versions](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-fanout/f0ba23c/fanout-mutation.log) · [Mixed reader policies](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-fanout/f0ba23c/fanout-mixed.log)
