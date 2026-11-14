# Preparation caller: status

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

Integration worker, session `59368744-da56-491a-871a-1be809c0cc93`, `claude-opus-5-5`. **READY** for the coordinated
matched graph build, and holding. This is uncommitted and syntax-checked only: no client run and no shared build.

## Changed boundary

The change is in metal-microbench `tools/prepare_e2b.py` `dataflow`, from the `before` normalization through the
publication predicate: the hunks at 1921–1956. The hunk at 1505 belongs to another worker and is untouched.

- **One ancestry relation.** `ancestry = mesh_graph.Graph.adjacency(range(len(before)), before.__getitem__)` is now
  made once, before fusing. Its edges run node → each producer it reads. The existing publication consumer uses the
  same object (`ancestry.reach([p])`).
- **Successors from its incoming rows.** The second adjacency is deleted, so `after` is gone. A node's readers are
  read from the retained relation's incoming rows (`data.in_first`, `data.in_edge`, and each edge's `source`), the
  on-disk fields that `mesh_graph.py` already uses in `paths` and `width`. `successors[node]` is the unique reader
  with the same `rank_of`, as before (a set, unique or `None`), computed once per node.
- **One finite relation for fusing.** `fused = Graph.adjacency(states, fusing)` over `(node, left)` with `left` 0 or
  1. Its transition is the current rule exactly: the unique same-rank successor m, only where `head[m] <= left`, to
  `(m, left - head[m])`. The relation is built once.
- **One incoming forest search (final review correction; replaces the per-q reach memo).**
  - The fusing relation is acyclic with at most one outgoing edge per state, so its transpose is a forest.
  - `tree = fused.search(sinks, incoming=True)` is one depth-first search over the incoming rows from the sinks:
    the states with no outgoing row, read from `fused.data.row_first`. It visits every state once, because each
    state's outgoing path ends at a sink.
  - Its owned result is O(N) over the 2N states, made once and retained with `fused` through placement.
  - q's kernel is the path from `(q, 1 - head[q])` to its sink, which is that state's ancestors in this tree, itself
    included (Cormen et al., 3rd ed., §22.3).
  - `spans` and `kernel` are deleted. There is no search, set, closure or memo for each query.
- **Publication predicate.** The sole predicate reads `q in ancestors and not mesh_graph.ancestor(tree, (p, 0), (q, 1 -
  head[q])) and not mesh_graph.ancestor(tree, (p, 1), (q, 1 - head[q]))`.
  - The second argument is the ancestor and the third its proposed descendant; equality counts, so the start
    is included.
  - This is the former "p not in q's kernel" with the same negation and short-circuit: the tree is asked only where
    `q in ancestors`, and the second query only after the first fails.
  - Rank, head, budget and placement semantics are unchanged.
- **Removed:** the `after` construction, the `mesh_graph.chain` call, and `spans`/`kernel`. `mesh_graph.py` and the
  graph C code are not edited.

## Publication consumer, reviewed

The placement loop is otherwise unchanged. `place[x] = k` where a reader is not in `ancestry.reach([p])`, or `p`
lies in that reader's kernel (the two ancestor queries above). The chain ordering and `home`/`stage` selection then
append `dataflow` entries as before.

## Evidence

- `ast.parse` and `py_compile` of `tools/prepare_e2b.py`, with the venv interpreter, both pass.
- `git diff --check tools/prepare_e2b.py` is clean.
- **Dependency (frontend LiteRT-owned, not on disk yet).** `mesh_graph.ancestor` and the search result's `end` field
  have not landed in `mesh_graph.py` or `mesh-graph.h`. The caller is written against the agreed API; it passes syntax
  checks now and runs once those land with the matched graph build.
- Not run: `tools/prepare_e2b.py` on a model, which needs that build and the existing E2B preparation client. No
  runtime or output-equality result is claimed.

## Supervisor checkpoint

The caller is committed in metal `df43fff9`. The source-visible graph producer
now includes `mesh_graph_search.end`, `mesh_graph_ancestor`, and the matching
Python binding. The worker's preceding not-on-disk and uncommitted observations
are superseded by this checkpoint. The matched Search56 library is still pending.

Source review confirms the finite relation has at most one outgoing edge per
state and is acyclic whenever the source dependency graph is acyclic. Reversing
it gives a forest rooted at the original sinks. The single retained DFS result
therefore answers the exact old bounded-path predicate, including its start,
with linear storage and constant-time membership. The actual publication
consumer reads that result; no per-query graph, traversal, set or cache remains.

The next observation uses the existing prepared E2B inputs at
`metal-microbench/.build/e2b-ffn/typed-20261109/decode-tp2-attn-7-1-mlp-ledger-bal`:
`operands.json` regions and the two `decode_rank_*.tflite` programs through
`tools.prepare_e2b.dataflow`. This is the existing preparation entrypoint; it
requires no target execution or service.
