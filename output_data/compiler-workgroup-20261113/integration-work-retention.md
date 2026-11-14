# Finish unchanged function-analysis retention

Continue the same implementation and session as `integration-conflict-retention.md`. Root reviewed the connected
memory-only result and consumers; the remaining concrete loss is `Changed` clearing every `workCache` entry.
Complete that connection now. The existing operator assignment and compilation-composition requirements apply.

Determine the actual function mutation owner. Cloned callees existing is not by itself evidence that transformations
mutate those shared callees. Follow Inline/Epilogue/Discard and each direct function-body mutation. A replaced
function's old unreachable cache entry is distinct from an in-place changed function with live callers. Retain
unaffected completed Work results and invalidate the changed function and its actual dependents, if any.

Use existing Tint call/use correspondence and the canonical graph algorithms if a transitive relation is actually
needed. Do not introduce a new general analysis manager, string/digest reconstruction, blanket invalidation under
a different name, or a second compiler graph walk whose result is immediately discarded. Record the specific
mutation/sharing argument. Source/output reflection remains typed; no printed-source parsing.

Also address the query batching cost identified in your prior result: retain exact unknown pairs and, if simple,
group from-accesses with the same undecided partner set so one canonical call handles them without re-solving any
known pair. Do not add an elaborate batching framework. Use existing actual numerical construction/client inputs
to compare meaningful compile behavior once the coordinated build is ready; no new maintained tests or harness.

Ownership remains your exact conflict/work-result hunks in metal_ir and actual transformation mutation owners.
Compiler supervisor concurrently owns operation source edges and fusion legality; executor owns final argument
buffer emission. Do not stage their hunks. Check `integration-conflict-retention.inbox.md` for build/commit windows.
Publish source/check/remaining facts in `integration-conflict-retention.status.md`. Root will stage the already
reviewed conflict checkpoint separately; commits do not mean the remaining work is finished.
