# Dynamic indexed reader lifetimes

Design proposal, September 13, 2026. No executor implementation is implied.
This extends the [lowering plan](pallas-lowering-plan.md#3-lower-index-expressions-to-region-dependencies)
and [coverage inventory](lowering-coverage.md). It must be reconciled with
[pages and functions](pages-and-functions.md) before implementation; changing
only the readiness predicate would be incorrect.

## Published mechanism and actual source boundary

[Pallas bounded dynamic slices](https://github.com/jax-ml/jax/blob/main/docs/pallas/tpu/pipelining.md)
separate a configured maximum block size from a runtime start/size produced by
numerical code. [Papadopoulos and Culler's Monsoon](https://www.cs.cmu.edu/~18742/papers/Papadopoulos1990.pdf)
uses explicitly addressed storage and presence bits for operand matching. These
support retained bounded index operands and address-based readiness. Neither
publication establishes that mutable-source lifetime can be omitted, nor does
Pallas guarantee that arbitrary dynamic gathers stream from independently absent
host operands. The mechanism below is mesh's proposed composition of those ideas.

[mesh_realize](../rdma/mesh-dataflow.c) reserves static reader planes across every
input map. `mesh_issue_index` requires each input range present and unread;
`mesh_complete` retires all input ranges. `mesh_execution_add` installs fixed
row-to-function adjacency, and `mesh_events` revisits only notified rows. These
owners should retain dynamic dependency information directly. No application
scheduler, graph rebuild, per-candidate kernel launch or transfer handshake is
needed.

## Identities and bounded storage

One selector occurrence describes exactly one occurrence of each candidate source
region. A candidate region has one producer stream and one stable logical-row
identity within this configured value slot. All addresses, valid counts, masks,
source ranges and selected row indices remain explicit; no min/max reconstruction
from an index vector substitutes for its actual members.

Several simultaneous selector invocations use distinct configured selector,
candidate and output value slots. Their numerical allocation is bounded at setup.
A slot can repeat after its preceding occurrence's lifetime is satisfied. Within
one slot, a source that has retired may independently publish its next occurrence
before other sources retire. That next source remains held for the next selector.
The same source must not be interpreted twice by the old selector. Sources cannot
skip occurrences or match arbitrary selector epochs under this contract; doing so
requires explicit additional value identities, not guesses from temporal order.

Retain these fields in setup metadata:

- The selector's canonical row range and dtype/layout, and its dedicated lifetime
  reader plane, separate from the numerical gather's ordinary selector reader.
- The exact bounded candidate source row ranges and their reserved reader planes.
- The selected source range/index vector and validity mask in the selector value;
  duplicates remain numerical contributions but share a source lifetime hold.
- Per selector/candidate **retirement result identity**. Its presence means this
  candidate occurrence has satisfied this selector's source obligation. This
  persists independently of that source row's current READ bit.
- Static adjacency from selector and candidate rows to their dynamic-reader
  descriptor; output/reader ownership remains the canonical executor's.

First use ordinary logical rows and their existing PRESENT/READ planes for the
retirement results. They are bounded, explicitly identified results of the
canonical reader-completion operation, not additional numerical tensors or a
second task queue. One result per candidate need not allocate one backing page:
logical identity and payload-page ownership are separate. Their aggregate lifetime
is one configured range, so existing bitmap AND-compare can identify completion.
Only introduce a dedicated compact bitmap if source ownership cannot represent
these ordinary result rows cleanly; it must encode the same identities and
lifetimes. No counter or occurrence number is required by this restricted contract.

## Why existing source READ bits are insufficient

Consider selector S selecting A while B and C are unselected. B arrives and retires.
C has not arrived. B is now reusable and publishes its next value. `mesh_reset`
clears B's READ bits. If S records retirement only in B's READ bit, S now sees B as
unread and retires the new B under the old selector. Repeated invocation loses an
operand without any numerical index changing.

The retained retirement result R(S,B) stays present across B's reset. The old
selector ignores B after that result is present. The next B remains unread until
the next selector occurrence begins. Thus R(S,B) describes a different fact from
B's READ bit: the former belongs to a selector occurrence, the latter to B's
current value. Keeping both preserves information that source reuse otherwise
discards. Holding B until C arrives would avoid this fact only by serializing
independent source reuse, which is not the proposed solution.

## Lifetime transitions

All transitions below belong to canonical publication/reader completion. They
operate on retained indices and bitmaps; they do not dispatch numerical no-ops.

| Arrival/completion | Required transition |
| --- | --- |
| Source before selector | Keep the source present and its reserved dynamic READ plane clear. There is no index value proving it selected or unselected yet. Notify its fixed adjacency. |
| Selector before source | Publish the retained selection. Inspect already-present, unresolved candidates once; missing candidates remain unresolved without preventing selected candidates from executing. |
| Already-present unselected source | Publish its retirement result, then mark its reserved source READ plane. Publication of the retirement result must precede source release so immediate producer reuse cannot erase the fact. |
| Unselected source after selector | The source notification performs the same retirement directly, provided its selector/candidate result is not already present. No numerical kernel is launched. |
| Selected source before or after selector | Leave its READ plane clear. The numerical function examines only actual selected ranges plus ordinary input dependencies; missing unselected regions are not readiness requirements. |
| Selected numerical work completes | Publish retirement results for its retained selected ranges before setting their source READ planes. Publish the numerical outputs at their declared independent boundaries. Retire the numerical selector reader once it no longer accesses indices. |
| Duplicate selected indices | Perform every numerical contribution, but retire each selected source obligation idempotently using the same result identity. A duplicate must not create an extra required retirement. |
| Last candidate retires | Once every retirement result in the selector's configured domain is present, satisfy the dedicated selector lifetime reader. Other numerical readers independently retain the selector if still active. |
| Source repeats while old selector remains live | Its old retirement result is present, so old-selector notifications ignore this new source value. The new source's READ plane stays clear until a new selector interprets it. |
| Selector repeats | Before publishing the new selector, clear/reset its owned retirement results as part of beginning that selector output occurrence. The preceding lifetime reader proves all old candidates resolved. Source values already published for the new occurrence are examined under the new selection. |

Selector preparation must reset retirement results before its PRESENT bit is set;
no event may read a half-written selector or half-reset domain. Use the existing
output PRODUCING/PRESENT ordering. The descriptor must not treat an old selector
whose lifetime reader is already satisfied as a new selection. The compiler and
executor own these relationships explicitly, rather than deriving them from an
old output's continued PRESENT bit.

For several numerical consumers of one selector, each consumer has its own source
reader obligation/retirement result, or the lowering proves one shared obligation
ends only after all its users finish. Separate concurrent functions must not clear
a source hold based solely on whichever completes first. Empty selection produces
its mathematical output without waiting for candidate data; lifetime cleanup
continues as each unselected source occurrence arrives. A candidate that never
arrives prevents reuse of that selector slot, not completion of its independent
numerical outputs or progress of other configured slots.

## Exact source disposition

`mesh_row_function`/`mesh_row_map` need an explicit dynamic-input descriptor instead
of disguising the possible source domain as an ordinary required input range.
`mesh_realize` reserves the possible-source READ planes, selector lifetime plane
and bounded retirement-result rows. It does not dynamically change reader masks.

`mesh_issue_index` first establishes selector availability, then visits the retained
selected ranges. It must neither dereference unpublished indices nor require all
possible sources. Output claimability remains unchanged.

`mesh_execution_add` retains fixed candidate/selector adjacency to the descriptor.
The existing serial event owner handles selector/source publication retirement.
Numerical completion must order retirement-result publication before releasing
selected source READ bits; if completion runs elsewhere, its atomic publications
must preserve that ordering and its final selector lifetime check must be
idempotent. There is no reason to rebuild adjacency for every selector occurrence.

`mesh_complete` and selector-output preparation must name the same retirement
result identities. Ordinary completion cannot blindly publish every associated
retirement row when the selector is produced: each candidate resolves separately.
This requires a documented canonical completion/owned-output reset extension,
not fabricated dummy functions or an inference from a pointer's current contents.

Tracing should retain selector/candidate/result row identities and selected ranges.
Current latest-occurrence traces alone cannot reconstruct earlier source retirement
once a slot repeats; do not use temporal proximity to manufacture that evidence.

## Irreducible work and remaining design check

Reading a dynamic index vector costs work proportional to its valid elements.
Matching a previously absent selected source requires some observation of its
publication. With static possible-reader masks, every candidate occurrence must
also discharge its selected or unselected lifetime obligation. This is metadata
work per candidate occurrence, not an O(candidate) collection of GPU/CPU launches.
Selector arrival can inspect the candidate domain once; source arrival uses its
fixed adjacency. A flat all-retired bitmap check per event may cost extra bitmap
scans; measure that owner before adding summaries, and never replace identities
with a counter that loses which source occurrence retired.

The implementation still needs to settle the exact ownership/reset API for
payload-free logical retirement rows. The proposed lifecycle above explains why
those rows exist and when they reset; it does not claim today's ordinary
mesh_row_function completion already implements them. If that ownership cannot
fit the existing logical-row model directly, document the narrowly equivalent
selector/candidate completion bitmap before adding a new field. Independent
source progress and occurrence identity are required in either representation.

## Implemented native contract

The native implementation now realizes the proposal through
`mesh_algebra_indexed(handle, function_index, selector, candidate_input_positions,
candidate_count)`. Input positions refer to the original numerical binding input
array. That array and an indexed-position set survive dependency merging, so an
ordinary alias of a dynamically read buffer retains its genuine ordinary
requirement. Several descriptors can attach to one function and independently
reserve source lifetime readers. Within a descriptor, candidate maps must be
disjoint canonical row domains; duplicate numerical indices share the same
candidate ordinal instead of duplicating physical candidates.

Selectors are locally produced U32 numerical outputs, containing candidate ordinals
or UINT32_MAX for masked accesses. Original indices and source values may arrive
remotely. Direct receives into selector storage and CONSTANT selector storage are
rejected during setup because their reset ownership differs from the local
producer contract. A selector source must cover every selector dependency row.
The compiler is responsible for emitting bounded ordinals and preserving all
ordinary/indexed access roles.

For C candidates the descriptor allocates 2C+2 canonical logical rows, without
operand payload pages: C retirement identities, C selected-membership identities,
one numerical-completion identity and one mapped-selection identity. Selection
membership is computed once from the retained numerical ordinal vector by the
existing metadata event owner. Unselected-source membership is then a bitmap
lookup, not a repeated scan of the vector per candidate. Selected input readiness
still visits the selected ordinal vector and its actual source maps; repeated
numerical indices are correct and can repeat those readiness checks.

Numerical completion stamps the completion result and notifies the selector's
existing adjacency. The serial metadata event owner subsequently retires selected
sources; unselected late sources use their own existing publication notifications.
There is no new synchronous dispatch or completion wait. Every retirement bit is
set before releasing its source READ plane. A selector-wide all-retired comparison
is needed only when an affected retirement bitmap word becomes complete. Selector
producer issue clears all its owned result bits before publishing the next selector.

The ordinary numerical selector reader prevents a numerically unfinished empty
selection from losing its indices even if every unselected source has already
retired. The separate lifetime reader prevents finished numerical work from reusing
selector storage while a late source occurrence remains unresolved. Retirement
identities persist across independent source resets. Output regions remain
independently publishable; their subsequent reuse obeys ordinary output-reader
lifetimes.

The native `mesh_row_function` structure gains an indexed-descriptor pointer;
libmesh and libmesh-algebra must be rebuilt together. The shared region header and
its version do not change. No bridge restart or workload was performed for this
source increment. Compilation establishes buildability only; operational evidence
belongs to the existing example and caller workflows.
