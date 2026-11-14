# Use lifetimes: status

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

Integration worker, session `59368744-da56-491a-871a-1be809c0cc93`, `claude-opus-5-5`.
Final observation, 14 November 2026: committed in mesh `11e39c11`; canonical libmesh build and actual
`partition → lower` caller completed. The worker is held. This is a dated implementation record, not a specification.

## Published contract

- **Native (`rdma/mesh-graph.h`/`.c`, after `mesh_graph_rows`):**

      int mesh_graph_lifetimes(uint32_t values, size_t uses, const uint32_t *value, const size_t *position,
                               size_t *first, size_t *last);

  Use k is value `value[k]` at program position `position[k]`. For each value v < `values`, `first[v]` and `last[v]`
  are the least and greatest position of its uses, one linear pass (ExecuTorch `update_tensor_lifetime`). Returns 0.
- **Binding (`rdma/mesh_graph.py`, replacing the unused `ends`):** `Lifetimes(uses)` over `(key, position)` events.
  It holds `at` (key → value ID, exactly the keys with a use, in first-use order), the native `value` and `position`
  arrays and the native `first` and `last` result arrays. `.first` and `.last` are Mapping views through the original
  keys, read from the result arrays.
- **Callers (`rdma/mesh_compile.py`):** `lower` retains `ir.value_lifetimes` over `(read key, g)` and reads `.first`;
  `_place` retains `ir.owner_lifetimes` over `(owner, g)` from reads' owners and declared waits, and reads `.last`.

## Changed sections (only these)

- `rdma/mesh-graph.h` and `rdma/mesh-graph.c`: `mesh_graph_lifetimes`, directly after `mesh_graph_rows`, as
  published above.
  - It sets every row to `SIZE_MAX`, then lowers `first` and raises `last` once per use in one linear pass. The
    producer's key table holds only used keys, so no row stays `SIZE_MAX`.
  - There's no validation and no other policy.
- `rdma/mesh_graph.py`: the unused three-argument `ends` is replaced by `Endpoint` (a Mapping view of one result
  array, read through `at`) and `Lifetimes`.
  - `Lifetimes` holds `at`, the native `value`/`position` arrays and the `firsts`/`lasts` result arrays.
  - `first` and `last` are properties that return a new `Endpoint` view over `firsts` or `lasts` (final source
    review). `Lifetimes` owns only `at` and its four arrays, the IR owns `Lifetimes`, and an escaping `Endpoint` owns
    `Lifetimes`. There is no `Lifetimes` → `Endpoint` → `Lifetimes` cycle, and nothing is copied.
  - Its restype and argtypes are set at the call, as the existing line-278 binding does. The module therefore still
    imports against the current dylib, and the symbol is needed only when a caller runs.
- `rdma/mesh_compile.py`:
  - `IR.__init__` gains `value_lifetimes` and `owner_lifetimes`.
  - `lower` retains `ir.value_lifetimes` over each read `(key, g)` and reads `.first`, so a collective's wait is
    still at its earliest actual reader.
  - `_place` retains `ir.owner_lifetimes` over the same events as the old call: each read key's owners, then that
    position's declared-wait owners. It reads `.last`, and `max(first, last.get(key, first))` and the
    landed/wire/invariant rules are unchanged.

## Co-owned changes in the same checkpoint

- `mesh_graph_closure_chains` in `mesh-graph.h`/`.c`.
- The `_analyse` hunks in `mesh_compile.py`.
- The finite-sharding operands, solver and retained result in `mesh_graph.py` and `_complete`.

## Evidence

- `clang -std=c11 -Wall -Wextra -Werror -ffp-contract=off -fsyntax-only` of `mesh-graph.c` passes, with `-I.` and the
  Makefile's HiGHS and ISL include directories.
- The same flags on `mesh-graph.h` alone also pass.
- `ast.parse` and `py_compile` (output to `/tmp`) of `mesh_graph.py` and `mesh_compile.py` pass.
- `git diff --check` on all four files is clean.
- Compiler supervisor's canonical libmesh build and fresh symbol/ABI check passed for `11e39c11`.
- The actual existing `examples.engines.Engines` export, with input `[2,32]`, K32/N48/L1 and explicit two-node
  placement, completed `rdma.mesh_compile.partition → lower`. Its selected `Shard(0,(1,3))` produced counts
  `(8,24)`. The lowered program had 28 items and four declared transfers. Both actual lifetime consumers ran:
  27 logical value rows and 13 backing-owner rows, with `first <= last < 28` throughout. The IR retained both
  original event arrays and native endpoint results. Record:
  [partition-lower-check.json](frontends/partition-lower-check.json). This was compilation, not target execution.
