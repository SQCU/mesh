# `_analyse` chain search and closure width: status

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

Session `64ddaf6e-3a70-4b6a-9d0b-50691a84c6cd`, model `claude-opus-5-5`. Assignment:
[call-analysis-views.md](call-analysis-views.md). **READY, HOLD.** The worker's checks below were private source checks. The supervisor build/checkpoint is recorded at the end.

## Published API (`rdma/mesh-graph.h`, beside `mesh_graph_chains`)

```c
int mesh_graph_closure_chains(const struct mesh_graph_closure *closed, uint32_t items, const uint32_t *member,
                              const uint64_t *count, uint64_t *chain, uint64_t *chains);
```

It calls the existing `mesh_graph_chains` once, with the same count and flow algorithm. Its native `before(a, b)` is
`reached[member[b] * words + member[a] / 64] >> member[a] % 64 & 1`, with no copied adjacency. `EINVAL` for a member at
or past `closed->vertices`. The graph relation, search and closure ABIs are unchanged.

## Changes

- **`mesh_graph.py` `paths`.**
  - One `Graph.adjacency` and one `graph.search(roots)`, the existing DFS. Each chain is the borrowed preorder slice
    `range(place[root], end[root])` of that search (Search56 `end`), viewed through `ordered` with the call objects.
    The per-root `graph.walk` (BFS) is removed.
  - The roots are chosen as before: a vertex with an outgoing edge and none incoming.
- **`mesh_graph.py` `closure_chains(closed, members)`** replaces `width`.
  - It resolves `LIB.mesh_graph_closure_chains` on use, so imports work against the installed libmesh.
  - It maps `members` through the closure's graph (`owned.graph.at`) and passes all counts as 1.
  - It returns `(chains, View(chain))`; the View retains the closure.
- **Removed:**
  - `width` and its Python `BEFORE` callback, which rescanned each row;
  - the `BEFORE` type and the `mesh_graph_chains` Python binding, which had no other user.
- **`mesh_compile.py` `_analyse`.** `outstanding[f] = mesh_graph.closure_chains(after, mine)[0]`, over the existing
  `after = mesh_graph.closure(range(len(calls)), ...)`. The comment now names `closure_chains`. The other values are
  unchanged: pairs, cause order, head labels and the source task identity.
- **`Calls`** (corrected by [call-analysis-retain.md](call-analysis-retain.md); see the correction below): appends
  `owner` (default None), the solved native results retained as made.

## Correction: Calls retains the solved results (call-analysis-retain.md)

**My earlier claim was wrong.** I said `Calls` needed no owner because nothing after `_analyse` reads the search,
closure or width results, so the integer `outstanding` values discharged them. Root rejected that: `Calls` is a
compilation result handed to the planner, so it must retain the native results it was made from.

- **`Calls`** is `namedtuple('Calls', 'outstanding orders causes norders chains owner', defaults=(None,))`. The
  archived load constructor (`saved['analysed']`, five fields) therefore stays valid with `owner` None.
- **The save path** writes its explicit five-field tuple (`outstanding`, the order and cause bytes, `norders`, the
  chains bytes), so the owner is never persisted, and a load solves nothing.
- **`_analyse` returns `owner = (chains, ordered, after, widths)`**, each a direct reference, with nothing copied or
  rebuilt:
  - `chains`: the chain Views, which hold the one follower Graph and its one search;
  - `ordered` and `after`: both existing closures, unchanged;
  - `widths[f]`: the pair `closure_chains` returned, `(count, View)`. `outstanding[f]` is that same count, so it is
    unchanged, and the pair is no longer dropped through `[0]`.
- **`closure_chains`' View owner** is `(owned closure, member, counts, chains)`: the native chain-count scalar itself,
  with the selected member and count arrays.
- **Edits:** only the `Calls` declaration, `_analyse`'s result construction and the `closure_chains` owner tuple. No
  C, header or other workers' hunks.
- **Checks:** `ast.parse` of `mesh_compile.py` and `mesh_graph.py`, and `git diff --check`: clean.

- **`paths` keeps its search with no chains.** It used to return a Python list of chain Views, so a block with no
  commuting chain got `[]` and its owned search was dropped before `Calls` could retain it. It now returns the
  existing `View(owned search, roots, range(len(roots)), at)`. Its element mapper returns the same borrowed preorder
  chain View, `ordered(owned, preorder, range(place[k], end[k]))`, in the original root order. With no roots it is an
  empty View that still holds the search. Only the `paths` return changed. Python AST and `git diff --check`:
  clean.

## Why the results are unchanged (from the source)

- **Disjoint chains.**
  - Each `b` sets at most one `follower[a] = b` (the first matching argument, then `break`), so in-degree is at most
    1.
  - A follower `a` has exactly one user, so it is followed at most once: out-degree is at most 1.
  - `b` uses `a`, so `b` comes later in graph order, and the graph is acyclic.
  - So each root's DFS subtree is exactly its chain, in chain order. That is the sequence the BFS from the same root
    gave. No vertex is reachable from two roots, so one search from all roots gives every chain disjointly, in the
    same root order. Isolated calls are never roots and never reached, as before.
- **Width.**
  - The old graph had an edge `a → x` exactly when `a in after[x]`. That is the closure row of `x`, bit `a`, because
    `closure` builds rows from each vertex to its predecessors, and `Closures` without `successors` reads the row.
  - The closure's vertices are `range(len(calls))`, so `member[k] = mine[k]`. The new predicate is therefore the same
    bit, and `mesh_graph_chains` gets the same items, counts and order. The chain count is the same.
- **Every caller:** `paths` and `width` had one live caller, `_analyse` (grep across `rdma` and `compiler`).

## Checks

- **`mesh-graph.c`:** strict private syntax and object compile (libmesh's flags plus `-std=c11 -Wall -Wextra
  -Werror`): clean.
- **`mesh-graph.h`:** compiled alone as C: clean.
- **Python:** `ast.parse` of `mesh_graph.py` and `mesh_compile.py`: clean.
- **Whitespace:** `git diff --check` on the four named files: clean.

## Supervisor checkpoint and remaining scope

Source is committed in `11e39c11`. The canonical `make libmesh.dylib` and fresh loaded-library ABI/symbol check passed;
`mesh_graph_closure_chains` is present. See [graph-consumers-review.md](graph-consumers-review.md) for exact logs and
the separate existing Engines partition/lower observation. That public path does not call `_analyse`. This unit's
chain and width correspondence is reviewed from source and privately compiled; no runtime `_analyse` comparison is
claimed. The worker remains held.
