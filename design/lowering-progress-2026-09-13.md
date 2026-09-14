# Lowering implementation progress

The [nine-point plan](pallas-lowering-plan.md) remains active. This record covers
landed increments, not completion of the full plan or a general performance claim.

## Landed source increments

- `db5654b`: operation, derivative, dtype, shape and backend inventory in
  [lowering coverage](lowering-coverage.md).
- `b6583ff`: compiled integer indices, masks, indexed loads and selection; compiled
  embedding gathers; shared Xonotic two-dimensional comparisons and selects.
- `51145c5`: expression outputs retain separate actual input dependency lists.
  An absent operand for one output no longer delays a different independent output.
- `a7490b0`: Xonotic chooses shared lowering before emitting custom Metal code,
  removing eager emission's rejection of otherwise supported FP16 numerical paths.
- `dfb0d4b`: actual per-direction transport frame accounting and independently sized
  CQs. [Source proof and limits](actual-frame-capacity-2026-09-13.md).
- `6440f9c`, `13ed4bd`: existing streaming-overlap example migrated to native
  contractions, matching peer partitions and reusable slots; report serialization.
  [Migration contract](streaming-overlap-native-2026-09-13.md).
- `d1430f7`: [dynamic reader lifetime design](dynamic-reader-lifetimes.md), retaining
  selector/source retirement identity across independent source reuse. This design
  motivates ongoing implementation; a selected-readiness-only hook is insufficient.

## Operational evidence

[Raw observations and traces](../measurements/lowering-2026-09-13/) identify measured
source revisions. Both bridges were gracefully restarted with the new frame/CQ
implementation after confirming zero clients, preserving the 549,650,432-byte
registered region. The provider accepted separate CQs on both M5 Max and M4 Pro.

The gold chain passes on the two-node CPU and Metal paths with FP16 inputs. The
ordinary exact cancellation outputs remain 2 and 0. The new indexed expression
retains an int64 value of 2**53+1, masks its invalid table access, and produces
`[[16,18,20,22],[1,1,1,1]]`. A second output of the same expression kernel remains
absent until its separate input is supplied, then produces sixes. No Python
numerical callback is submitted in the recorded Metal gold.

The existing native overlap example also completed 15 warmup and 30 measured
invocations using three reusable slots, 4096-by-256 inputs and 16-row regions.
It configured 256 regions per invocation and completed 11520 native CPU numerical
submissions per participant across warmup and measurement. The three terminal
invocations checked against the numerical reference had maximum absolute error
5.508994e-6. This is terminal numerical evidence, not validation of every earlier
intermediate. Raw timing samples and online statistics are retained.

The overlap run exercises repeated storage reuse and a configured population
larger than the previous 255-message maximum-block cap. Latest-occurrence traces
do not establish peak historical occupancy, so it is not claimed to saturate or
measure the throughput benefit of the enlarged frame capacity. No matched old/new
polling-overhead conclusion follows from these runs.

## Remaining completion requirements

Dynamic selected-page dependencies and lifetime retirement, general indexed
scatter reductions, general logical-rank lowering, legal fusion/storage planning,
collective placement and full caller migration remain unfinished. Xonotic source
migration is not a substitute for its operational numerical/performance evidence.
The plan's no-feature-regression and no-overhead acceptance audit remains open.
Current gold evidence does not establish arbitrary dynamic table streaming or
complete Pallas-style lowering.
