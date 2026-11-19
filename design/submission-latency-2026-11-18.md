# One submission, one byte, and one remote reader

The September 24 principles name “one push per layer” and “no guard on ready data.” The September 24 measurement instructions count waits, crossings and dispatches, and compare useful kernel work against hardware rates. On October 1 the operator explicitly asked for bandwidth, latency and overhead on concrete operations. A7 assigns FIFO order to the sequencer and allows one message to admit a known range of executions. Each transfer then waits for its actual producer. [Principles, originally `90b4f6ce`](../../../metal-microbench/docs/principles.md) · [Measurement, originally `916f15ee`](../../../metal-microbench/docs/measurement.md) · [A7](aot-compilation-change-request.md#a7-the-island-sequencer-orders-function-executions)

## A: What added time or returned the wrong value

Starting an execution changed the same wake word as installing a program. The bridge parent used that word to rediscover installed programs and to scan for retired programs. Each active port used it to rediscover installed programs too. On a Studio each full scan visited 5,373,952 entries of 32 bytes. The active program and its page addresses stayed fixed across the 27 executions.

The Metal emitter passed the landing of a four-byte transfer to a numerical function as a `constant` pointer. With 27 distinct input values, the first remote reader returned the previous input's result on all 24 timed invocations. Exchanging the two numerical functions moved the error to whichever ran first. Starting the receiver 100 ms early and reading the result buffer again 50 ms after return reproduced it. [Inputs, output values and timing](../../../compiler-emit/artifacts/resident-latency-20261118/measurement.md)

## B: What the compiler and bridge do now

The compiler reads each operation's received-storage records when it declares the numerical arguments in Metal. For a read-only argument that names such a buffer it emits `volatile coherent(system) device const`. The first and second numerical readers then returned the current generation through both the public callable and its resident native entry. On strided inputs the compiler packs the requested elements into the transport blocks and both readers use the destination scatter. Sixteen separated floats used one 64-byte send. The same program with 4,095 floats used one 16,380-byte send. [Emitter, `4804602`](../../../compiler-emit/compiler_emit/fusion.py) · [Strided numerical comparisons and request lengths](../../../compiler-emit/artifacts/joint-selection-review-20261118/live-fanout/packed-stride-e95382f/measurements.md)

The bridge discovers new programs when the installation generation changes. It checks the preparer's lock before caching a scan because the installer reserves a generation before publishing its root. Ports continue to check the roots they hold. The parent reclaims storage on retirement, process exit and its periodic liveness check. A submission wakes the bridge to order that execution. [Discovery and reclamation, `52051d4e`](../rdma/mesh-records.c) · [Port discovery](../rdma/mesh-flow.c)

## Hardware results

Studio rank 2 sent to rank 1. Each row gives the median of 24 warm executions. The one-byte case checks a changing request and reply on one clock. The compiled cases run the numerical graph and check every returned scalar against eager PyTorch. Both revisions in that comparison use the corrected Metal declarations. [Raw samples, p95 and reproduction commands](../../../compiler-emit/artifacts/resident-latency-20261118/measurement.md)

| Measured work | Earlier bridge | Updated bridge |
|---|---:|---:|
| One byte out, one byte back, entire range admitted | 21.0 µs | 20.0 µs |
| Same exchange, new submission for every execution | 12,692 µs | 348 µs |
| One float through the public callable, one remote numerical reader | 16,641 µs | 345 µs |
| Same callable with two remote numerical readers | 18,526 µs | 448 µs |

On the updated bridge, twenty independent Accelerate dot products ran while a byte reply was in flight. Median completion was 83.08 µs with that ordering and 139.44 µs when the same arithmetic started after the reply. This comparison measures CPU arithmetic overlapping a real transfer. The packed multi-block GPU case established correct delivery to both readers. How much GPU execution overlaps individual blocks in that case remains unmeasured. [Matched arithmetic](../../../compiler-emit/artifacts/resident-latency-20261118/measurement.md)

A second pair, Studio rank 3 and Mini rank 4, shared one advertised 80 Gb/s cable. Each request sent a payload in 16 KiB blocks and returned a one-byte acknowledgement. With admission outside the clock, median useful payload rates before and after the bridge change were 3.57 and 4.22 GB/s at 256 KiB, 7.23 and 7.12 GB/s at 2 MiB, and 8.78 and 8.84 GB/s at 8 MiB. Each timing includes the reply. The receiver checked a changing byte each round and compared the full payload during warmup and on the final sample. Both runs used 24 warm samples. The 8 MiB result measures about 88% of the advertised line rate on that cable. [Probe and commands](../../../compiler-emit/artifacts/resident-latency-20261118/bulk.md)

All five bridge processes restarted with the new executable. Their original 13 paired cables and advertised rates returned. The direct byte exchange with fresh admission repeated at 282 µs in reverse run order. That remaining cost and the two-reader public callable's 1.00 ms p95 are material at this scale. [Fleet readback](../../../compiler-emit/artifacts/resident-latency-20261118/raw/bridge-publication-reload.json) · [Distributions](../../../compiler-emit/artifacts/resident-latency-20261118/measurement.md)

Rank 3 and rank 4 also installed a program, verified 27 changing echoes, retired it, and installed a new program under the same group number. All 27 new echoes matched. [First installation](../../../compiler-emit/artifacts/resident-latency-20261118/raw/publication-reinstall-1.log) · [Second installation](../../../compiler-emit/artifacts/resident-latency-20261118/raw/publication-reinstall-2.log)
