# Dead source relocation: status

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

Integration worker, session `59368744-da56-491a-871a-1be809c0cc93`, `claude-opus-5-5`. **HOLD** for root's review and
commit. Nothing is staged, built or committed. The next unit (signature node correspondence) waits for root to
release this move.

## Notice to root before editing `examples/tp.py`

The next edit takes these `examples/tp.py` lines at HEAD `1ac90a36` into `reference/retired/tp_cost.py`, verbatim:

- 102–116: the `# design/unequal-shards-change-request.md item 7` comment and `class Timed`.
- 119–126: the two citation comment lines and the malformed `taken(fixed, inputs):` with its body.

It also removes the blank lines 117–118 and 127–128. `chosen` (line 130), which calls `taken`, is left for its
owner.

The `tp.py` edit was made after this notice, as stated.

## Moves (verbatim, malformed headers unchanged)

| Production source at `1ac90a36` | Moved to | Change to the source |
|---|---|---|
| `rdma/mesh_compile.py` 7611–7686 (`_replays` and its citation comment), 8182–8279 (`_bound`), 8348–8512 (`program` and its comment), 8734–8764 (`load`), 8767–8821 (`_runner`) | `reference/retired/mesh_program.py` | 435 lines deleted, 0 added: the spans plus their trailing blank lines, and the 2 blank lines then left at the end of the file |
| `compiler/mesh_compiler/lower_coreml.py`, the whole file | `reference/retired/lower_coreml.py` | moved with `mv`; unstaged ` D`; `cmp` identical to HEAD |
| `examples/tp.py` 102–116 (`Timed`), 119–126 (`taken`) | `reference/retired/tp_cost.py` | 27 lines deleted, 0 added |

- Each reference file starts with a two-line comment naming its origin lines and checkpoint. After that, the spans
  are in source order, separated by two blank lines.
- A script confirmed that each `mesh_compile.py` span at HEAD appears byte-for-byte in `mesh_program.py`.
- `reference/retired/README.md` quotes both operator turns. It names checkpoint `1ac90a36` and `b25ec14b`, and lists
  the three dead files.
- No shared primitives, ctypes declarations or constants were between or inside the spans. `_execution_copy`,
  `CAUSES`, `_FREE` and all other live definitions stay in place.
- Imports used only by the moved bodies were not touched.

## Live consumers of the moved names

- `examples/tp.py:111`: `chosen` calls `taken(fixed, inputs)`. It is left for Frontend41, the owner of the
  replacement cost consumer.
- `compiler/mesh_compiler/native.py:442` mentions `_replays` in a citation comment only.
- No other `rdma`, `compiler/mesh_compiler` or `examples` Python source calls `_replays`, `_bound`, `_runner`,
  mesh's `program` or `load`, `Timed`, or imports `lower_coreml`.
- No live function was nested in the moved closures.

## Checks

- `ast.parse` passes on the changed live files `rdma/mesh_compile.py` and `examples/tp.py`.
- The expected `SyntaxError` from `ast.parse`, which only parses and doesn't compile or import:
  - `reference/retired/mesh_program.py` line 12, `_replays(...)`. The four later headers, at lines 83, 189, 350 and
    383, are also still malformed.
  - `reference/retired/lower_coreml.py` line 321, `lower(scheduled):`.
  - `reference/retired/tp_cost.py` line 24, `taken(fixed, inputs):`.
- `git diff --check` is clean on `rdma/mesh_compile.py` and `examples/tp.py`. `README.md` has no trailing whitespace.
