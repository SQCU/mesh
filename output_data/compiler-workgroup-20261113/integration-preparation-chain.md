# Preparation caller: retained relation through publication placement

Same integration session 59368744-da56-491a-871a-1be809c0cc93, explicit
claude-opus-5-5. The source-map unit is committed in metal 79d7e7a6. It is complete
for this assignment; do not reopen its compiler/graph sections.

Operator: "\"Those callers are centralized only partway.\" then this must happen
immediately". Preserve the actual producer/result/consumer handoff, using the
existing mesh-graph.h/.c implementation. This is implementation work, not a new
review or a helper-only assignment.

Exact owned section: metal-microbench/tools/prepare_e2b.py, the producer/before
relations, fusing/kernel and publication-position consumer around lines1885–1960.
Keep all unrelated working edits. Frontend supervisor grants this section. Read
the repository instructions and the current code; no external processes, fleet
services, new evaluator or new graph module.

Current producer already has before, rank_of and head. It reconstructs after,
then runs a Python chain walk for each kernel(q). Reuse one ancestry Graph over
before, created before fusing. Its retained incoming rows give successors; select
the unique successor of the same rank under the existing rule. Remove the second
after adjacency construction. Frontend's concurrent incoming-row helper has
been agreed; coordinate through the actual on-disk Graph fields/API, without
editing mesh_graph.py or graph C yourself.

Construct once the finite directed relation over (node, remaining head budget
0/1). Its transition is exactly the current fusing rule: the unique same-rank
successor, only when its head fits the remaining budget, subtracting that head.
kernel(q) consumes canonical reach from (q, 1-head[q]) and projects reached states
to node IDs for the existing publication-position calculation. Retain the
relation and correspondence through that consumer. No graph reconstruction per
query, private DFS, source-text interpretation, added ordering, or altered rank,
head or placement semantics. The source graph is acyclic; preserve its semantics.

This is a finite-domain input to the existing graph algorithm, not a new generic
callback framework. If a smaller direct use of the same retained relation
expresses exactly the same bound and avoids duplicating state, explain it to the
supervisor before substituting. Do not silently change the bound.

Run Python syntax/AST and diff checks, review the actual publication consumer,
write a compact integration-preparation-chain.status.md with exact changed
boundary and evidence, then stop READY. Do not build shared libraries or commit;
root coordinates the next checkpoint. No claimed runtime or performance result
without the corresponding existing client run.

Frontend's API is now source-visible: `Graph.rows(incoming=True)` gives the
retained in_first/in_edge rows with original predecessor/successor identity;
`search`, `postorder`, `walk` and `reach` accept `incoming=True` too. Use that
actual API for the before relation's users. Frontend owns implementation and
matching native build, including mesh_graph_relation_tail. You need no C change.

Root review correction after the first READY draft: `spans[q] = {node for node,
_ in fused.reach(...)}` still discards the native search result and duplicates
its membership into a Python set. Retain the actual `fused.reach` result in
spans[q]. At the existing sole publication predicate, node p belongs to that
kernel iff either (p, 0) or (p, 1) belongs to the retained reach result. Consume
those two direct membership queries, without a new set/class/walk; preserve the
existing negative predicate and short-circuit behavior. The original search and
its relation then survive through the real consumer with no second allocation.
Your direct incoming CSR projection is acceptable: Graph.rows currently returns
native first/accessor operands, not a Python neighbours iterable. Do not invent
a competing accessor API just to obscure those direct view reads.

FINAL REVIEW CORRECTION — replaces the per-q reach/memo design above. A full
native search allocates arrays over all states; caching one per q would make
this preparation consume quadratic space. Do not retain that design.

The finite relation is acyclic with outdegree at most one. Its transpose is
therefore a forest. Run ONE incoming DFS from its sinks. Frontend now supplies
the existing DFS result's `end[v]` (preorder count when v finishes) and the
canonical ancestor predicate from CLRS section22.3; this is one O(N) owned
result. Exact agreed caller API:

    tree = fused.search(sinks, incoming=True)
    mesh_graph.ancestor(tree, (p, 0), (q, 1-head[q]))

The second argument is an ancestor and the third its proposed descendant;
check both p states0/1. Equality counts as ancestry, preserving start inclusion.
`sinks` are states with no outgoing edge, read from this relation's retained
rows (or its existing source/sink result). Retain tree and the finite relation
through publication placement. Delete spans and kernel entirely; no per-query
search, set, closure or memo. The actual publication predicate keeps its existing
negation/short-circuit semantics using the two canonical ancestor queries.

Frontend LiteRT owns `mesh_graph_search.end`, `mesh_graph_ancestor` and the thin
Python function; do not edit them. Graph.search(..., incoming=True) is already
source-visible. Update your status to replace the earlier memoized-search claim,
explain the forest premise and O(N) ownership, run syntax/diff checks and stop
READY for the coordinated matched graph build. This is the same connected
source/consumer assignment, with the performance defect corrected.
