# Preserved occurrence and deployment identity

The September 13 source audit found two places where existing metadata was
lost and then reconstructed incorrectly.

## Occurrence-directed execution

The tagged-token firing mechanism cited in
[algorithm sources](algorithm-sources.md#presence-driven-execution) requires
both function identity and occurrence identity. A configured function already
provides its occurrence count and explicit or strided input/output ranges.
Registration now creates a watch for every occurrence and records that index.
Each range contributes edges to precisely that watch. An event tests only that
occurrence, and submission receives the same index used for ownership and
subsequent completion. No event searches the function for a different occurrence
or substitutes occurrence zero.

All watch and edge allocations finish before the registration becomes visible.
Allocation failure frees the unpublished metadata, leaving no partially active
function behind. Existing registrations remain active. This is configuration
realization, not a numerical barrier.

Both republication and constant publication now notify the existing presence
observer after changing the canonical bits. The event is a wakeup; the page table
remains the readiness source. Previously republication could leave a dependent
asleep until some unrelated publication woke it.

This audit does not claim that event delivery failures or arbitrary simultaneous
host ownership claims have been resolved. Those are distinct contracts from
retaining the configured occurrence index.

## Deployment identity

The deployment launcher already passes `MESH_APPLICATION_IDENTITY` as JSON,
including source revision, root, and native-client provenance. The runtime
identity reporter now retains that object directly. It no longer tries to infer
an active deployment from the removed bundle's `application.json`. An invocation
without a launcher identity reports its root and unbundled state explicitly.
This follows the configuration ownership described in
[algorithm sources](algorithm-sources.md#configuration-storage-layout).

## Source-review boundaries

Registered-memory views already preserve their page vectors. Consecutive page
runs are coalesced only while establishing virtual mappings; that optimization
does not discard page identity. Explicit `mesh_row_map.ranges` also remain
available and are now used for every occurrence's event edges.

No test harness was introduced. The changes are reviewed through ownership,
index, and event flow; compilation is an implementation check rather than a
substitute for those contracts.
