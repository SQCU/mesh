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


## Shared contraction ownership and reader fanout

`89fec3f` moves K partitioning, FP32 partial contractions, pairwise reduction and
final casts into the shared `kernels.dot` expression lowering. `nn.linear` now
supplies output layout and placement. Existing CPU/MPS/Core ML operation owners
remain unchanged. `55eddbd` fixes direct mapped BlockSpecs by resolving their index
maps per coordinate, including mixed mapped/whole operands. The existing example
adds reversed output rows from transposed input views with a ragged K tail. Both
CPU and Metal pass that numerical check, the original cancellation case and gold.
This migration preserves the numerical launch formula for each output tile;
matched hardware nonregression remains a separate acceptance requirement.

`d0a12e2` documents [shared routing ownership](shared-routing-ownership.md).
`d8386f5` and `4ca4369` implement canonical reader groups for fanout that cannot fit
the direct reader-plane representation, with one actual member identity per
source row and reader occurrence. Direct planes remain the setup-selected path
when they suffice. `e5a7cd6` exposes those retained identities in Python traces.
`449ab9e` releases member storage with its owning program/maps and makes the
existing metadata owner the sole member-reset owner for both host and remote
publications. No new numerical launch, source copy, epoch guess or invocation
rendezvous implements group completion. Program teardown remains outside numerical
invocation. This source ownership fix is not a claim of a measured graph-churn
memory curve.

The configurable existing scatter case passes on both backends with 67 updates
in 33-row tiles, including a ragged final tile, SIMD radix work beyond one group,
FP32 fused factors, invalid/masked indices and two changing occurrences. Earlier
traces before the added fanout case show zero grouped readers for the six-update
program and two grouped source rows for the wider program, each with 166 distinct
retained member identities. Source grouping removes the reader-plane capacity
failure; it does not remove worst-case destination-by-candidate metadata or the
per-potential-segment physical allocation and launch costs.

The same example also sends a source through RDMA to 65 independent numerical
consumers and returns their results. Both CPU and Metal complete two source
occurrences: first/last branch outputs 2/130, then 3/195, with every branch checked.
The peer trace joins the grouped source row directly to a receive transfer with
two occurrences and 65 distinct member identities. This exercises remote receive
reset, not merely local directory reuse. Both peer workloads terminated normally.

Raw `reader-lifecycle-local-*` and `reader-lifecycle-paired-*` archives retain each
installed library revision, example revision, observations and both participants'
traces where applicable. The paired CPU gold maximum error is 0.001953125; Metal
is 0.00390625 under the unchanged absolute-plus-relative tolerance. These runs
establish covered execution and reuse, not matched throughput improvements or
physical GPU/wire overlap percentages. Nested dot epilogues, arbitrary-rank
lowering, sparse routing ownership, production storage/launch optimization, full
caller/derivative migration and the final performance audit remain unfinished.

## Shared sparse domain follow-up

The [sparse routing source and operational record](sparse-routing-progress-2026-09-13.md)
documents the shared candidate/directory ownership implementation through
`5843a8b`. It replaces per-destination candidate descriptors and duplicated
directory/table readers with explicit shared domain holds. Local CPU/Metal pass
1025 updates and 17 destinations; paired CPU/Metal pass the existing float16
gold with remote fanout reuse. The actual Xonotic gather/concatenate graph now
demonstrates independent indexed producers and consumers through the shared
representation. Source, precision, locality, derivative and performance limits
are recorded with the raw evidence. The bounded active-segment increment follows
below; physical partial storage and the broader plan remain unfinished.


## Bounded active numerical grids

`afd3b7b` retains the static maximum min(chunk updates, destination rows) for
scatter partial capacity. `b081848` implements native count/slot activation and
explicit produced/omitted dispositions; `4975a24` binds exact producer identities
in shared lowering; `dc4c80a` exposes count, omission and disposition reader traces.
Omission performs no numerical launch and publishes no payload. Routes consume
its explicit disposition without reading a fictitious payload occurrence.

The implemented indexed contract requires count and all selector/range views to
belong to one actual produced directory output, with compiler-supplied empty
inactive ranges. Existing indexed lifetimes retain late unused payloads;
ordinary unused reads retain separate retirement facts. The count and disposition
cannot recycle before their actual readers finish. Independent indexed-count
producers and arbitrary inactive selections remain outside this increment.

[Active-grid source and evidence](active-segments.md#archived-operational-observations)
now records local CPU/Metal passes for 1025 updates in 513-row chunks and 4097
updates in 2049-row chunks, both with 17 destinations. Installed library is
`dc4c80a`; examples are respectively `dc4c80a` and `e0ecb91`. Raw unchanged logs
and traces are archived with [explicit provenance](../measurements/lowering-2026-09-13/active-omission-provenance.json).
Four normal/changed/all-invalid/normal occurrences check independent consumers,
all-empty output before any update/factor payload, subsequent late-unused
retirement, and reuse. The larger case records the count, bounds and selector
on distinct directory pages under the same produced output ownership.

Each trace's 34 main scatter partial functions account for 50 numerical launches
and 86 omissions over 136 slot occurrences. Six Xonotic gradient partial functions
add 12 launches; whole-workflow submissions/completions are 2067/2067. The same
records include independent gradient outputs before an unrelated cotangent
arrives, without allocating the primal operand. This is bounded numerical and
source-lifetime evidence, not a matched throughput gain or complete derivative
coverage. Finer physical partial storage, repeated readiness traversal, general
rank/fusion, placement and the plan's final no-regression audit remain open.

## Composed contraction epilogues

`3288be4` replaces root-only dot dispatch with recursive pointwise epilogue
lowering in the existing expression compiler. Native FP32 K-panel contractions
and intermediate pairwise reductions are shared by identical contractions in
one grid coordinate. Repeated occurrences also retain the same expression input
identities. The last one or two partials feed the existing scalar emitter,
combining final summation, pointwise work and output conversion in one function.
A separately requested bare contraction retains independent publication and
canonical reader lifetime; it does not acquire the epilogue's extra operands.
Computed contraction operands and shape-changing dot reductions remain open.

The existing streaming-algebra example at `e8ebac1` requests
`expression(dot(a,b)*2+bias, dot(a,b))` over three independently supplied rows,
with a ragged two-panel K contraction and transposed weight backing. It supplies
row 1 without bias, observes the bare contraction while its epilogue remains
unavailable, then supplies that bias and observes the epilogue while rows 0 and
2 remain unpublished. It finishes those rows and repeats using the same storage.
Both CPU and Metal match exact dyadic reference values in both occurrences.

The native trace retains two shared contraction-panel functions, one fused
epilogue and one bare-result addition per row: twelve configured functions and
24 numerical submissions over two occurrences. For row 1 in both local traces,
partials start at canonical rows 196 and 200; epilogue output 168 reads those
and bias 152, while bare output 180 reads only the two partials. This is direct
dependency evidence, not a reconstruction from timing. No scalar callback,
extra host operand store or invocation-time allocation implements this change.
The remaining K-panel storage and native launch costs have not disappeared.

Local CPU/Metal runs use 4097 scatter updates, 2049-row chunks, 17 destinations,
three gold invocations and the existing Xonotic forward/derivative cases.
Both finish 2091 numerical submissions; unchanged gold maximum absolute errors
are approximately 1.65e-6 and 1.85e-6 respectively. `composed-provenance.json`
and compressed `composed-{cpu,metal}` observations/traces retain source and
configuration under `measurements/lowering-2026-09-13`. These observations do
not establish matched throughput improvement or complete general composition.

The paired CPU and Metal float16 runs also pass at example `88cba6a` with
67 updates in 33-row chunks and 17 destinations. Installed local library is
`e8ebac1`; installed peer library is `88cba6a` (identical numerical source).
The distributed gold maximum errors are 0.001953125 and 0.00390625 under the
unchanged tolerance, and all 65 remote fanout branches match both occurrences.
Rank zero finishes 1357 submissions in each backend. Indexed and composed-dot
side cases execute on rank zero; these runs do not establish remote scatter or
remote composed-dot candidate lifetime coverage. Both peer processes terminate
normally with SIGTERM after writing their traces. The paired observations,
both participants' traces and exact provenance are archived alongside the local
records. Broad expression/rank lowering, remaining caller migrations, storage
and launch optimization, and matched end-to-end performance remain open.

## Demand-driven computed contraction operands

`2aec0ae` replaces root-reference-only contraction construction with one
`_ContractionRegions` setup owner. Natural shapes, mapped/local versus whole
coordinate domains and backing cuts remain distinct metadata. An outer K-panel
requests only its left M×K and right K×N regions. Pointwise computations and
nested contractions recursively produce those canonical regions, preserving
FP32 intermediate values and independently ready native matmul inputs. Shared
region demands retain expression, source-view and origin identities; specialized
program IDs participate in those identities. No runtime graph builder or hidden
whole-intermediate operand is introduced.

The existing example now expresses a linear/swish/linear chain in one numerical
expression. X is 3×4 in single-row blocks, W1 is 4×6 in two-column blocks, and
W2 is 6×3 in two-row blocks. Both dots have K tile two. Supplying X row one,
W1's first feature panel and W2's first contraction panel permits the second
matmul to finish a contribution before the other two hidden panels exist.
The observer joins W2's actual canonical row range to its configured numerical
consumers and records their completion; it does not guess which function ran
from command order. After the missing weight panels arrive, row one finishes
while X rows zero and two remain unpublished. Both complete occurrences match
the independent float64 reference within the unchanged FP32 tolerance.

CPU and Metal local runs pass with installed source `7fc905d` and example
`ff3afa3`, before the subsequent FFN caller and mixed-dtype changes. Native
function 42 consumes W2's first canonical block (first row 236, four allocated
pages) in both early observations. Whole-workflow submissions are 2175, all
completed. The example initially shadowed its final timing dictionary with the
new consumer observation tuple; `ff3afa3` fixes that reporting error without
changing numerical execution. `nested-initial-provenance.json` and the compressed
`nested-final-{cpu,metal}` records preserve the passing runs. These establish
partial producer and partial consumer execution for this composition, not
matched throughput parity, remote nested contractions, or arbitrary-rank lowering.

Computed indexed loads, shape-changing reductions and mismatched non-singleton
mapped/global operand domains still require broader lowering. Native M/N panels
still need to fit their actual backing blocks. These limitations remain explicit;
there is no whole-operand copy or wait fallback for them.

## FFN caller composition and mixed operand types

`fab4ada` moves FFN hidden projections, their balanced partition sum and swish
into one shared expression call. It preserves the prior row/column partition
cuts, the activation's declared input dtype (including FP16 rounding), the
existing exchange boundary, and FP32 down-projection accumulation. The caller
no longer constructs separate completed hidden contractions and hidden sums.
An explicitly nested expression instead retains computed panels in FP32;
adding an expression-level cast remains necessary to spell the former FP16
rounding boundary inside an entirely nested FFN expression.

`1448f0e` removes the native contraction operand-dtype equality restriction.
The existing CPU implementation already has independent typed loaders and FP32
accumulators; MPS retains each original matrix buffer, offset, strides and dtype.
No conversion buffer or alternative numerical backend implements the change.
Core ML rectangle metadata and its generator now retain each operand's dtype
separately instead of reconstructing the right type from the left. Its existing
FP32 arithmetic conversion remains unchanged. Native libraries and the generator
compile; Core ML runtime/placement/internal-copy claims are not established by
that source evidence.

Installed source and example `1448f0e` pass the integrated CPU and Metal local
runs in both float32 and float16, plus the float16 two-node gold and remote
fanout workflow. The float16 nested expression specifically exercises
FP16×FP16→FP32, FP32 swish, then FP32×FP16→FP32 through native MPS on the Metal
backend. It observes the second contraction completing a partial before two
hidden panels arrive in each occurrence. This is evidence for the covered
heterogeneous MPS input combination, not a universal undocumented type guarantee.

The unchanged 4097-update float32 configuration completes 1919 submissions on
each backend, compared with 2175 before the FFN migration: 256 removed numerical
submissions. Float32 gold maximum errors are approximately 1.65e-6 on CPU and
1.58e-6 on Metal. Local float16 runs complete 2035 submissions, and paired rank
zero completes 1249; all four float16 runs have gold maximum error 0.001953125
under the unchanged tolerance. The paired gold still streams across the real
link and its 65 remote fanout branches reuse their inputs correctly. Nested
and indexed side operations remain local to rank zero. Both peer workloads
terminate normally after recording their traces.

`nested-integrated-provenance.json` and compressed `nested-integrated`,
`nested-mixed` and `nested-paired` observations/traces preserve configurations,
source and both participants where applicable. No matched throughput improvement
is inferred from the submission reduction or these short timings. General rank,
indexed/reduction composition, remaining derivatives and callers, storage/launch
optimization, collective placement and full performance acceptance remain open.

## Shared reductions and caller removal

`70b6511` extends the existing region owner to row reductions and renames it
`_ExpressionRegions`. A reduction demands the full resolved child feature domain,
computes each available feature partial independently, combines actual siblings
in a balanced tree, and caches the resulting row statistic across output tiles.
A computed dot can consume that cached statistic directly. Square-and-sum is
fused in each feature partial; there is no squared-tensor intermediate or
whole-tensor prerequisite. Scalar/intrinsic-only expressions have scalar domains;
indexed reduction domains must be supplied by explicit index-shaped operands.

`82c1ab3` expresses RMSNorm as one shared expression with the original output
cuts and dtype. `4841436` migrates the remaining Xonotic vector and matrix-row
sum/mean callers and deletes `_row_reduce` and its unused row-map helper.
Real means divide the FP32 sum before final conversion, eliminating intermediate
half rounding/overflow. Integer means retain their typed sum boundary before
integer division. These precision choices are documented in the canonical source
record, rather than hidden in caller-specific partial constructors.

`1aeb6cb` uses unsigned modular addition for integer-valued signed reduction
accumulators and partial merges, restoring signed interpretation at use. The
first Metal run then exposed an unsupported `simd_sum(ulong)` overload at
compilation. `b844daf` uses three supported UInt32 SIMD sums with exact limb
carry reconstruction; its modular arithmetic proof is in algorithm-sources.md.
The floating path is unchanged. The compile-failure log is retained alongside
the successful runs; no numerical fallback implements the correction.

The existing workflow supplies two of three feature panels for one normalization
row. A reduction consumer completes before the last panel arrives; afterward
that row normalizes while two unrelated rows remain unpublished. Both repeated
occurrences pass. The Xonotic graph now also requests matrix row means followed
by a vector sum, observing the early row means before the unrelated source block
arrives; the totals are 30 and 158. Integer cases preserve cancellation beyond
2**53 and signed 64-bit wrap. The final paired case also exercises low/middle
limb carries across SIMD lanes and overflow between separate feature partials,
returning 65536, 4294967298, -9223372036854775808 and 9223372036854775807.

Local CPU and Metal float32 workflows pass, including the 4097-update scatter
case. Both report 2045 completed submissions. Paired CPU and Metal float16 runs
report 1343 completed submissions on rank zero and gold maximum error
0.001953125, with remote fanout reuse intact. The paired reduction observer joins
canonical partial rows 516–523 to consumer function 79 in both occurrences.
Nested/reduction/indexed side cases remain local to rank zero; paired gold and
fanout exercise the actual link. Both peer workloads terminate normally.

`reduction-provenance.json` and compressed `reduction-*` observations/traces in
`measurements/lowering-2026-09-13` retain exact library/example revisions and
both participants. Before-migration observations use a smaller example scope;
added integer and Xonotic cases account for additional work, so total counts and
short timings do not establish a throughput comparison. General logical rank,
remaining indexed/derivative composition, physical storage/launch optimization,
collective placement and matched performance acceptance remain unfinished.
