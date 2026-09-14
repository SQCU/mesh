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

The planned native active binding retains a produced count view and exact slot
ordinal for each configured numerical function. A slot below the count executes
its existing numerical body and publishes its partial. A slot outside the count
has an explicit omitted disposition: no backend launch and no payload PRESENT.
This is a bounded numerical grid, not a second participant scheduler.

Shared routes retain the exact producer-function identity for every candidate.
An absent destination owner alone cannot prove omission: an active segment may
fall outside the requested output coverage and still produce a real payload.
An omitted producer disposition discharges the domain's candidate obligation
without reading or consuming payload storage. A produced disposition preserves
the actual payload reader until the consuming numerical function finishes.

Disposition and count lifetimes must survive all actual readers. A late source
occurrence must not be confused with an omitted occurrence or the next producer
value. Inactive ordinary inputs retain explicit late-retirement facts; existing
indexed inputs retain their selected/unselected lifetime mechanism. For the
initial scatter lowering, count and indexed bounds are views of the same produced
directory and inactive bounds are empty. Arbitrary inactive indexed selections
require a broader lifetime transformation before they can use this mechanism.

No final destination initialization is omitted. Repeated active/omitted/active
instances must reuse the same configured storage and distinguish their actual
dispositions. Physical subpage packing and dynamic reuse across simultaneously
live slots remain separate work.

## Acceptance

Use the existing streaming-algebra workflow and Xonotic graph roots. Source must
establish exact count, slot, producer, candidate and lifetime ownership. Traces
must distinguish numerical submissions from omitted dispositions and must retain
actual identities. Delayed unrelated cotangents or updates may not become
dependencies of ready destinations. All-empty and changing active domains must
return the required base values without fictitious partial transmissions.

This document records the capacity implementation and the active-disposition
contract. It does not claim native omission is implemented or measured yet.
