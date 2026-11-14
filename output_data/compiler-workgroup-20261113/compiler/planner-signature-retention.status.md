# Planner signature retention status

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `dc4b988b-a100-4e17-8c80-ff3f90ac97e3`.
READY for supervisor review. The source is held: not staged, not committed.

## Hunks

- **`compiler/mesh_compiler/ir.py`, `Plan`:** only two trailing fields, `signature: bytes = b""` and
  `nodes: tuple = ()`. The executor's uncommitted `ir.call` deletion in the same file is untouched. No other
  definition and no comment changed.
- **`compiler/mesh_compiler/schedule.py`, `solved(plan, planning)`:**
  - Line 134, after `mesh_plan`, `mesh_plan_save` and `plan.record`, and before `mesh_planned_free`:
    `plan.signature, plan.nodes = mesh_graph.archive(made.signature.contents), tuple(planning.nodes)`.
  - Lines 118-119: the citation comment gains `rdma/mesh-graph.h mesh_graph_archive` and
    composition-change-request items 3 and 6.4.
  - `completion`, `order`, `sections` and `record`, and the `mesh_planned_free`/`SYSTEM.free` ownership, are
    unchanged. No guard was added.

## Source and API checks (no planner execution)

- `py_compile` passed and `pyflakes` was clean for both files. Each diff is exactly the hunks above.
- `ir.Plan(*range(18))`, the existing 18-argument positional form that `planned` uses, constructs with
  `signature == b""` and `nodes == ()`. `archive.load`'s `Plan(**about["plan"])` still accepts archives written
  without the two fields.
- `mesh.Planned.signature` is `POINTER(mesh_graph.Signature)`, so `made.signature.contents` is the native
  `struct mesh_graph_signature`.
- Native ownership, which fixes the placement before the free: `plan_graph` sets `planned->signature`
  (`rdma/mesh-collective.c:1373`), and `mesh_planned_free` frees it (`1894-1895`).
- **Dependency, now present (updated):** `mesh_graph.archive(made)` is in `rdma/mesh_graph.py` (commits `22085c16`
  and `47fe9ee1`). It takes the `Signature` struct and passes it with `C.byref` to `mesh_graph_archive`, so
  `solved`'s `mesh_graph.archive(made.signature.contents)` matches its interface. The earlier note that a carried
  compile raises `AttributeError` there no longer applies. Planner execution is still not run (below).
- **Source finding:** `mesh_plan_save` already calls `mesh_graph_archive(planned->signature, ...)`
  (`mesh-collective.c:2309`, `2336`), so the same bytes are also embedded inside the opaque `plan.record`.
  `plan.signature` keeps them as an explicit result, so no consumer has to parse the record.

## Planner execution

Not run. No existing planning fixture exists without a bridge: `Planning` comes only from `bridge.planning()`
(`mesh_attach`). The context-free `ordered()` call to `mesh_plan(None, ...)` returns before `plan_graph`, so it
produces no signature. As instructed, I attached no bridge and built no fake planner.

## For the executor (owner of `archive.py` and `signature.py`)

`archive.save` currently writes `{**dataclasses.asdict(plan), "record": plan.record.hex()}`. With
`plan.signature: bytes` it raises a JSON `TypeError` for any carried plan until the field is encoded as `record`
is. `plan.nodes` would load back as a list. Both are in the executor's assigned encoding and decoding hunk.
