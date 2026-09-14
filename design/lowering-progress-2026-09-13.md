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

General indexed scatter reductions, general logical-rank lowering, legal fusion/storage planning,
collective placement and full caller migration remain unfinished. Xonotic source
migration is not a substitute for its operational numerical/performance evidence.
The plan's no-feature-regression and no-overhead acceptance audit remains open.
The dynamic cases below establish their specific selected-source and reuse behavior;
they do not establish arbitrary dynamic indexing or complete Pallas-style lowering.


## Dynamic selected sources and constant specialization

`5175c7e` and `d5c41c0` implement whole-Tensor indexed loads, numerical selectors,
and canonical selected-source lifetime retirement. `c9c38f8` extends the existing
gold example: selected source rows produce fours while an unrelated source is
absent; the selected source then publishes its next occurrence before the old
unselected source arrives, and the next output is sixes. Both CPU and Metal local
runs pass. This checks a reuse error that ordinary final FFN accuracy would miss.

`056ebce` retains explicit constant declarations and eliminates dynamic selectors
only when every candidate belongs to that declared constant storage. Present
produced tables still require selectors. `228ea96` moves trace export after the
dynamic example, so numerical submissions from that demonstration are included.
Latest-occurrence traces still do not reconstruct full historical retirement.

The archived `constant-cpu.json.gz` and `constant-metal.json.gz` observations use
`228ea96`, FP16 inputs, three measured gold invocations plus warmup and the existing
side checks. Both pass with maximum gold error 0.001953125; exact cancellation,
masked integer indexing and both dynamic generations also pass. Both report 1652
completed numerical submissions. A local CPU run before constant specialization
reported 1844; this records removed selector work, not a matched throughput claim.
CPU completion latency has count 3, mean 13.249736333333333 ms and sample variance
2.097980727796333 ms²; Metal count 3, mean 24.254124666666666 ms and sample variance
3.5864129249723318 ms². These are local runs, not new distributed scaling evidence.


## Direct CPU register accumulation and indexed diagnostics

`dd19ebe` replaces the CPU half contraction's scalar dot path with direct-stride
4×4 NEON register accumulation in FP32. No full operand conversion storage is
introduced; configuration retains loader functions and canonical output addresses.
The existing all-FP32 Accelerate path remains. See
[cpu-register-contraction](cpu-register-contraction.md) for mechanism and limits.

`b8ab668` and `bf6128d` expose original indexed reader, selector, retirement and
output map identities in the existing trace. `7bcf4f5` and `a442808` add produced
selector subranges with retained vector/range lifetimes. This range extension is
compiled, but its dynamic runtime evidence remains to be supplied by the scatter
vertical slice. `0184b6a` and `9462daf` give expression lowering its whole configured
grid once, enabling shared routing preparation without repeated per-output setup.

Archived `neon-strided.json.gz` uses installed library `bf6128d`. CPU passes the
FP16 gold, exact cancellation, masked indexing, dynamic reuse and transposed
3×5 by 5×7 tail contraction. CPU batch time is 6.537459 ms; completion count 3,
mean 5.939403333333334 ms, sample variance 0.2719680602963338 ms². Three subsequent
Metal runs pass the same numerical cases. This is numerical and operational
evidence, not a matched speedup or broad kernel performance characterization.

An initial Metal observation failed before the example checked the independent
tail result's readiness. The example previously assumed main-chain completion
implied unrelated numerical completion. `a442808` explicitly observes the side
results after the timed work. Source inspection found the physical MPS descriptors
and transpose flags correct; no backend fallback or extra numerical dependency
was introduced. The first failure did not record the result values, so it alone
cannot distinguish unpublished output from numerical mismatch; the corrected
observer and three subsequent passing runs are the retained evidence.


The subsequent two-node CPU gold also passes with maximum error 0.001953125,
including cancellation, strided tails and dynamic indexed reuse. Raw observations
and both participants' traces are in `neon-paired-cpu.json.gz`, with each installed
library revision retained explicitly (`bf6128d` local, `4be627c` peer). The local
completion count is 3, mean 6.897945 ms, sample variance 0.000019781713 ms², and
batch time 6.90275 ms. These unmatched observations do not establish a distributed
speedup over the improved local baseline. The peer exited normally via SIGTERM
after the local numerical validation completed.


## Executable segmented indexed addition

`ae4183f` implements vector-row indexed addition through the existing expression
and kernel_call API. It groups retained destination keys/original ordinals,
computes per-segment partials in FP32 and selects only the partials contributing
to each final destination region. `25cfbd7` adds defined unsigned modular integer
accumulation. See [scatter lowering](scatter-lowering.md) for mechanism and limits.

The existing gold example now supplies three routing chunks, withholds the last
update chunk and observes the pointwise consumers of destinations 0, 1 and 3.
Destination 2 remains unavailable until its missing payload is supplied. Duplicate
keys across chunks sum correctly, the masked update is excluded, and the empty
destination produces zero. The second occurrence changes both routing and values
using the same storage, giving consumer rows [10, 0, 18, 12] after [8, 0, 14, 8].
All four rows repeat their scalar value across four features. CPU and Metal both
pass. This exercises produced selector ranges and their shared-vector lifetimes,
not just arithmetic over constant indices. Host observations occur after the
main timed gold and introduce no numerical dependency between independent rows.

Raw `scatter-reuse-cpu.json.gz` and `scatter-reuse-metal.json.gz` retain installed
library and example revisions, numerical observations and indexed trace maps.
Their observer times are upper bounds on availability and include host writes;
they are not physical compute/wire occupancy or matched performance gains.
Coverage is six updates, four features and four destinations over two occurrences.
General rank, wider radix tiles, production-scale memory/launch costs, fused
update expressions and the actual Xonotic migration remain open. Partial storage
is bounded by maximum segment count, but separately allocated canonical quanta
and per-reader candidate metadata still require storage/lowering optimization.


## Xonotic consumer migration and destination validity

`9b4ea55` migrates rank-one Xonotic scatter, scalar/vector pointwise expressions
and sum consumers. Remaining custom kernels retain explicit block pointers and
strides rather than demanding a single whole-region view. No dense operand copy
is introduced. [Xonotic source mapping](xonotic-indexed-add.md) records remaining
nonstreamed operations and axis coverage. This is an incremental migration, not
completion of the expert/neighborhood and derivative inventory.

The existing two-node planner completed normally at source `9b4ea55`: width 16,
eight bots, eight-row tiles, five-second duration, 3384 returned plans and 559
objective switches. `xonotic-planner.json` retains both outputs and configuration.
This validates the actual planner's compilation, transport and repeated execution;
it does not exercise every newly migrated scatter consumer or establish numerical
residuals, a throughput gain, or full Xonotic coverage. The peer Xonotic environment
uses Python 3.12 and the repository's declared MLX/NumPy dependency versions.

`e4d18c0` evaluates scatter destination validity before narrowing indices to U32.
`9a96306` extends the second existing scatter occurrence with index 2**32+2;
it must not wrap into destination 2. Both CPU and Metal pass, producing second
consumer rows [10, 0, 12, 12], rather than the earlier case's [10, 0, 18, 12].
`scatter-valid-cpu.json.gz` and `scatter-valid-metal.json.gz` retain these runs.
The earlier reuse artifacts remain evidence for their separately recorded inputs.


## Fused pointwise updates

`c4086e5` lowers pointwise update expressions directly inside segmented indexed
accumulation, using the same scalar emitter as ordinary expressions. Every used
operand retains its own selected source region and lifetime. Broadcast values
and mixed half/float sources preserve FP32 arithmetic; declared constants avoid
unnecessary dynamic lifetime bindings. There is no transformed-update tensor or
extra pointwise launch. Nested indexed loads and feature reductions inside the
update expression remain unfinished forms, not an eager-load fallback.

The existing example now uses updates * factors + 1 with separately streamed
FP32 factor rows. It supplies the final update rows before their factors; the
final consumer remains unavailable until the factor rows arrive. Independent
consumers have already completed. Both CPU and Metal pass two occurrences with
changing routes and factors: consumer rows [20, 0, 32, 18], then [32, 0, 38, 40].
Each row repeats its value across four features. Gold, cancellation, strided and
indexed reuse checks also pass. Archived scatter-fusion CPU/Metal observations
and traces retain exact library/example revisions. Both report 1704 completed
numerical submissions, the same count as the prior unfused expression case;
this establishes absence of an added transform launch in the covered program,
not a matched throughput improvement.
