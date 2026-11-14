# Planner arena retention status

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `dc4b988b-a100-4e17-8c80-ff3f90ac97e3`.
READY for supervisor review. The source is held: not staged, not committed. The signature/nodes change stays as it
was, and the executor's `ir.call` deletion is untouched.

## Hunks

- **`compiler/mesh_compiler/ir.py`, `Plan`:** only two new trailing fields after `signature` and `nodes`:
  `arena_bytes: int = 0` and `offsets: tuple = ()`.
- **`compiler/mesh_compiler/schedule.py`, `solved(plan, planning)`:**
  - `sections = [value.section for value in function[0].value[:len(plan.values)]]` holds the actual post-plan
    sections. `plan.sections` is built from this one list as before: same fields, same order, not duplicated.
  - After `mesh_plan`/`mesh_plan_save`, `plan.record` and `plan.signature`, and before `mesh_planned_free`:
    - The output arrays are `first = (Q * (values + 1))()`, `offset = (Q * sum(section.count))()` and `view = View()`.
    - It calls `mesh.LIB.mesh_view_values(context, function, 1, first, offset, byref(view))`. This is the existing
      binding (`rdma/mesh.py:275`), on the same planning context and function that `mesh_plan` bound.
    - It sets `plan.arena_bytes = view.bytes` and
      `plan.offsets = tuple(tuple(offset[first[v]:first[v + 1]]) for v in values)`.
  - The citation comment adds `rdma/mesh-dataflow.h mesh_view_values` and composition items 3, 6.4 and 7.
  - Nothing else changes: no address is kept, no pointer is compared, no copy-0 assumption is made, and there is
    no new alias or lifetime solve and no guard.

## What the retained result is (source reading of `rdma/mesh-dataflow.c:80-101`)

- **`offsets[v]`** has one entry per copy of value `v`: `v.section.count` entries, ordered by copy `k`. Each run
  value's entry is the page-table address of row `section.first + k * section.stride`, minus the lowest such
  address over every run value of the function. Every residue is kept. A non-run value has `UINT64_MAX` for each
  copy, kept as is.
- **`arena_bytes`** is the highest run row end (address + `stride` × block) minus that lowest address, or 0 when
  there is no run value.
- The addresses are read from the planning context's page table, where `mesh_plan` bound the rows. The offsets are
  relative to the arena, so a uniform relocation of the run region by `plan_save`/`load` leaves them valid. This is
  the order's statement about relocation; I read it in source and did not observe it.

## Checks

- **Source/API, private:**
  - `py_compile` passed and `pyflakes` was clean for both files. No line I added exceeds 120 characters; the one
    longer line is `schedule.py`'s pre-existing header, line 1.
  - The diffs are exactly the hunks above.
  - `ir.Plan(*range(18))`, the positional form `planned` uses, constructs with `arena_bytes == 0` and
    `offsets == ()`. The trailing field order is `signature, nodes, arena_bytes, offsets`.
  - The output arrays for counts `[]`, `[0]` and `[2, 1, 0]` (a zero-length `offset` included) are accepted by
    `mesh_view_values.argtypes[3:6].from_param`, without calling the native function. `View.bytes` reads as an
    `int`, and array slices give tuples of `int`.
- **Execution:** not run. A non-empty result needs `mesh_plan` on a real `Planning` context, which comes only
  from `bridge.planning()` (`mesh_attach`). As instructed, I attached no bridge and built no fake planner. No value
  of `arena_bytes` or `offsets` has been observed.

## Consumers (not edited)

- **Frontends:** `Published` roots at the relocated arena base.
- **Executor:** its graph takes one byte root and forms typed views at `offsets[value][copy]`. The executor also
  owns `archive.py` encoding and decoding: `offsets` is a tuple of tuples of ints, so it loads back from JSON as
  lists, and `arena_bytes` is an int.

The signature status (`planner-signature-retention.status.md`) is updated: its `mesh_graph.archive` dependency is
now present (`22085c16`, `47fe9ee1`).
