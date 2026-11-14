# Actual use events through one lifetime calculation

SAME integration session59368744-da56-491a-871a-1be809c0cc93, explicit Opus5.5.
Root's preparation caller is committed df43fff9. This is a separate, narrow
connected correction under the operator's immediate graph-caller instruction.

Exact supported callers: rdma/mesh_compile.py lower at1669 and _place at1728.
Both call mesh_graph.ends(order, uses), but its current signature is
ends(order, reached, uses). Neither caller has or needs the new partial-order
heuristic in that helper: lower needs first use of a logical program value;
_place needs last use of a backing owner, including declared waits. This is a
concrete broken public path, not authority to remove that supported path.

Implement those actual producer/result/consumer boundaries. The source is the
existing ir.program order and its actual use events, plus _place's existing
owner correspondence/waits. Lower's first-use and _place's last-use calculation
must consume one canonical lifetime aggregation algorithm in mesh-graph.h/.c,
using the vocabulary of the already cited ExecuTorch
exir/memory_planning.py update_tensor_lifetime. Read existing code and references;
do not introduce a second lifetime allocator or transitive-closure problem.

Use a compact typed native operand: actual value/owner IDs with original program
positions. Retain the original key/owner correspondence and native event arrays.
Compute each ID's first and last position together once, as native result arrays;
unseen IDs have an explicit absence only where the actual input domain requires
it. Input membership may instead include exactly keys with a use. The caller's
chosen first/last is a view of that retained result, not a decoded-and-freed dict.
Keep result/input owners on the existing IR lifetime. Do not discard arrays and
then reconstruct them for an immediate consumer.

Preserve current semantics: lower assigns a collective's wait to its earliest
actual user; _place takes max(first definition, last owner use), with existing
landed/wire/invariant rules unchanged. _place's extra wait events and logical
value-to-owner mapping are genuine different input facts, not permission to
transcribe the aggregation twice. No runtime scheduling, extra waits, fallback,
new validation policy, or changes to collective numerics.

Owned sections: isolated lifetime declaration/function in rdma/mesh-graph.h/.c;
thin owned-result binding in rdma/mesh_graph.py replacing unused ends; only the
two actual calls and needed retained-result member in rdma/mesh_compile.py.
Coordinate section placement with compiler before editing C; current narrow
library build must finish first. Frontend owns finite sharding and labels;
numerical owns paths/width/_analyse. Preserve all their edits. No whole-file
rewrite, new module, generic Python-callback solver or maintained test harness.

Publish the exact operand/result API before adapting both actual callers, run
private strict C and Python AST/diff checks, write concise
integration-use-lifetimes.status.md and hold. Root coordinates build and commits.
No broad audit; complete this unit directly.
