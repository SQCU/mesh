# Bounded active segment domains

The JAX authors' [Megablox grouped multiplication](https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/kernels/megablox/backend.py)
retains group offsets, tile identities and the actual tile count. Its numerical
grid visits actual tiles; empty groups are visited when required to initialize
an output. Mesh applies this distinction to internal scatter partials. Final
destinations still return their base when they have no contributions.

## Static capacity

For a chunk containing U updates and a destination tensor containing D rows,
every validated key is one of D destinations. Stable grouping produces at most
S = min(U, D) segments. Invalid and masked keys create no additional segment.
`afd3b7b` therefore reserves S partial slots per feature stripe, preserving a
distinct output for every potentially simultaneous segment. It introduces no
reuse dependency and merges no publication boundaries.

The sorting directory retains its original U-based key, ordinal, bounds and
selector offsets. The partial tensor shape retains S separately. Reverse routing
enumerates those S actual candidate slots, rather than treating the directory
length as the partial capacity. Grouping also writes its actual segment count
at offset 8U; no host reads that value to construct or schedule numerical work.

For 1025 updates in chunks of 513 and 512 with 17 destinations, capacity becomes
34 partials instead of 1025. Installed source `6c41fe2` passes the existing local
CPU and Metal gold with this shape, including delayed factors and two routing
occurrences. Whole-workflow submissions change from 3915 to 1933: exactly two
occurrences of the 991 eliminated potential partials. This source/operation-count
result is not a measured throughput gain. It precedes runtime active omission.

## Produced and omitted occurrences

`b081848` implements the native active binding; `4975a24` attaches it in shared
scatter lowering and `dc4c80a` exposes its trace. `mesh_algebra_active` retains a
produced count view and exact slot ordinal for each configured numerical
function. A slot below the count executes
its existing numerical body and publishes its partial. A slot outside the count
has an explicit omitted disposition: no backend launch and no payload PRESENT.
This is a bounded numerical grid, not a second participant scheduler.

Shared routes retain the exact producer-function identity for every candidate.
An absent destination owner alone cannot prove omission: an active segment may
fall outside the requested output coverage and still produce a real payload.
An omitted producer disposition discharges the domain's candidate obligation
without reading or consuming payload storage. A produced disposition preserves
the actual payload reader until the consuming numerical function finishes.

`mesh_algebra_route_producers` retains each exact producer function and its
disposition reader map. Disposition and count lifetimes survive all actual readers.
A late source occurrence must not be confused with an omitted occurrence or the next producer
value. Inactive ordinary inputs retain explicit late-retirement facts; existing
indexed inputs retain their selected/unselected lifetime mechanism. Native
realization requires count and every indexed selector/range map to lie
within one actual produced output region. This is an explicit shared-publication
restriction, not an assumption that views happen to share backing storage. The
compiler supplies empty indexed ranges for inactive slots. The directory producer
cannot reuse any of that output while its indexed late-unselected obligations
remain. Independently produced counts/selectors and arbitrary inactive indexed
selections require a broader lifetime transformation; they are not covered here.

No final destination initialization is omitted. Repeated active/omitted/active
instances reuse the same configured storage and distinguish their actual
dispositions. Physical subpage packing and dynamic reuse across simultaneously
live slots remain separate work.

## Acceptance

Use the existing streaming-algebra workflow and Xonotic graph roots. Source must
establish exact count, slot, producer, candidate and lifetime ownership. Traces
must distinguish numerical submissions from omitted dispositions and must retain
actual identities. Delayed unrelated cotangents or updates may not become
dependencies of ready destinations. All-empty and changing active domains must
return the required base values without fictitious partial transmissions.

The native omission and covered CPU/Metal execution are implemented. The source
and observations below establish that bounded contract, without claiming complete
Pallas lowering, arbitrary conditional graphs or a throughput improvement.

## Source and trace ownership

In `rdma/mesh-dataflow.c`, `mesh_active_issue` decides omission before ordinary
operand readiness, marks a logical omitted result and publishes the disposition.
It neither issues the backend nor publishes the numerical output. Ordinary unused
input maps retire only when their actual occurrence arrives; their retained result
bits are written before releasing the reader. `mesh_active_event` releases the
count hold after those input obligations and every disposition reader finish.
The existing indexed retirement remains responsible for late unselected inputs.
Count producer preparation resets the disposition/omission/input-result rows.

`mesh_route_ready` requires the current produced disposition as well as selected
payload readiness. `mesh_route_retire` consumes the disposition for an omitted
candidate without setting payload READ. For a produced candidate it retires both
actual payload and disposition ownership. An old payload PRESENT bit is therefore
not evidence that the current optional occurrence produced a value. Several
routes can retain separate readers of one producer disposition.

`python/mesh/kernels.py` retains partial producer function IDs during compilation
and passes their exact candidate ordering to `mesh_algebra_route_producers`.
Traces expose the count/slot and disposition/omission/input-retirement rows for
each active function. The omission metric is separate from numerical submissions.
Route role 4 exposes the actual disposition reader map; role 1 still exposes the
candidate payload map. No candidate disappears from accounting merely because
its numerical function did not run.

## Archived operational observations

All four runs below use installed library `dc4c80a`, local float32 CPU or Metal,
17 destinations, four features, and the existing Xonotic forward/gradient side
checks. The example revision is recorded separately because the larger run uses
bounded input magnitudes. The [provenance record](../measurements/lowering-2026-09-13/active-omission-provenance.json)
identifies every raw artifact; gzip decompression preserves the original bytes.

| Updates / chunk | Backend | Example | Raw observations | Raw trace |
| --- | --- | --- | --- | --- |
| 1025 / 513 | CPU | `dc4c80a` | [log](../measurements/lowering-2026-09-13/active-cpu.jsonl.gz) | [trace](../measurements/lowering-2026-09-13/active-cpu-trace.json.gz) |
| 1025 / 513 | Metal | `dc4c80a` | [log](../measurements/lowering-2026-09-13/active-metal.jsonl.gz) | [trace](../measurements/lowering-2026-09-13/active-metal-trace.json.gz) |
| 4097 / 2049 | CPU | `e0ecb91` | [log](../measurements/lowering-2026-09-13/active-bounded-pages-cpu.jsonl.gz) | [trace](../measurements/lowering-2026-09-13/active-bounded-pages-cpu-trace.json.gz) |
| 4097 / 2049 | Metal | `e0ecb91` | [log](../measurements/lowering-2026-09-13/active-bounded-pages-metal.jsonl.gz) | [trace](../measurements/lowering-2026-09-13/active-bounded-pages-metal-trace.json.gz) |

Each run passes normal, changed routing, all-invalid and normal-again scatter
occurrences using the same storage. In the all-invalid occurrence, the example
checks every destination and pointwise consumer before supplying any update or
factor payload. It then supplies those unused payload occurrences for retirement.
The final normal occurrence checks reuse after that late retirement. On nonempty
occurrences, independent destinations complete before the last update/factor
chunk; the dependent destination remains unavailable until its factors arrive.
The logs retain observer times, not physical GPU/wire occupancy measurements.

Both shapes retain 34 main scatter partial functions. Across their four
occurrences the traces record 50 actual numerical launches and 86 omissions,
accounting for all 136 configured slot occurrences. Seventeen functions have
zero launches and four omissions each; sixteen have three launches and one
omission; one has two of each. Six further active functions belong to the
Xonotic gradient side case and each launch twice without omission. Thus the
whole trace has 40 active bindings, 62 numerical launches for those bindings and
86 omissions. These are distinct from the full workflow's 2067 submitted and
2067 completed numerical calls. CPU reports 2067 CPU submissions; Metal reports
zero CPU submissions. The full workflow also includes gold, precision, indexed
reuse, contractions and 65-consumer local fanout; its count cannot be compared
directly with an earlier workflow containing fewer side cases or occurrences.

The 4097-update trace establishes multi-page directory ownership directly. Its
first directory producer is function 18 with output rows 286 through 293. Its
first partial (function 19) retains ordinary directory rows 286 through 290,
selector rows 289 through 290, bounds row 288, and explicit count row 290. The
count reader has its own recorded member identity. These are one producer's
output region with different read coverages, not an assumed single-page alias.
CPU and Metal retain the same logical identities in these runs.

The Xonotic derivative observations check two changing generations: independent
gradient rows are returned while cotangent block 2 remains absent, then the
remaining row completes when it arrives. `primal_operand_allocated` is false.
Those observations are archived with the forward gather/concatenate records;
they do not establish coverage of every Xonotic derivative.

These runs establish the covered numerical outcomes, actual omitted launches,
late unused retirement and repeated storage reuse. They do not establish a
matched speedup, a scaling law, zero polling overhead, or a physical compute/I/O
overlap percentage. The two input families are not matched throughput baselines.

## Exact large-shape observations

The original demonstration used increasing positive update values. At 4097
updates those produce integer sums beyond float32's consecutive-integer range.
The reference's sequential `np.add.at` and the implementation's independently
rounded partial sums can then differ despite the intended FP32 contract. The
existing exact-equality observation is unsuitable for that input magnitude.

The example now cycles update magnitudes through 1 through 31, shifted by the
occurrence index. At 4097 updates the maximum possible four-occurrence example
term is 171, and twice the sum is bounded by 1401174, below 2**24. Every positive
integer partial, final sum and pointwise doubling is therefore exactly
representable in float32. Exact equality remains unchanged. This changes the
demonstration inputs; it does not change kernel precision, cast placement,
reduction order or comparison tolerance. Earlier archives retain their original
inputs and are not matched throughput baselines for the new data.
