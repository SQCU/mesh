# Planner archive bindings: worker status

Session `64ddaf6e-3a70-4b6a-9d0b-50691a84c6cd`, model `claude-opus-5-5`. Assignment:
[planner-archive-bindings.md](planner-archive-bindings.md). No staging or commits; no shared output built.

**READY.** The source is held for supervisor review and a named commit.

## Changed path

`rdma/mesh_graph.py` only (+30 lines), next to `record`/`from_record`:

- ctypes prototypes from `rdma/mesh-graph.h:278-286`: `mesh_graph_archive(const struct mesh_graph_signature *, void *,
  size_t, size_t *)`, `mesh_graph_archive_read(const void *, size_t, struct mesh_graph_signature *)` and
  `mesh_graph_compose(const struct mesh_graph_signature *, const struct mesh_graph_signature *, struct
  mesh_graph_signature *)`, each returning `int`.
- `archive(made) -> bytes`: a size query (`ERANGE` expected, otherwise `_check`), then the final call into a buffer of
  that size; returns immutable `bytes`.
- `from_archive(data) -> Signature`: `mesh_graph_archive_read(data, len(data), &made)` under `_check`.
- `compose(a, b) -> Signature`: `mesh_graph_compose(&a, &b, &made)` under `_check`.

The `Signature` structures that `from_archive` and `compose` return own their C arrays, and the caller releases them
with the existing `free()`. `compose` takes no ownership of `a` or `b`. On failure the C functions free their output
themselves (`record_restore` and `mesh_graph_compose`, `mesh-graph.c:2522` and `:2720`), so `_check` raising leaks
nothing. `mesh_graph_archive_read` copies the input bytes, so its buffer need not outlive the call. There is no
Python parsing, solving, graph derivation or printing, and no new classes.

`mesh.Planned.signature` is `C.POINTER(mesh_graph.Signature)` (`rdma/mesh.py:136`), so the schedule worker's
`mesh_graph.archive(made.signature.contents)` binds to this prototype directly.

## What the archive retains (from source)

`mesh_graph_record` keeps the primary arrays (processor, task, dependence, operation, edge, least, alias) and the
program metadata. `mesh_graph_record_read` restores them and then rederives the indices and counts with
`signature_derive`. `mesh_graph_archive` writes a header, the record and seven derived tables: position, previous,
waited, first, order, count and unknown. The header holds the counters uncovered, redundant, unwaited, the states
and the primitive count. `mesh_graph_archive_read` restores the record without derivation, copies the derived tables
back, sets the counters, and keeps the digest. It rejects a mismatched magic, primitive count, length or table size
with `EINVAL`.

## Compose scope (from source; not a union policy)

`mesh_graph_compose` unites processors by their canonical identity, puts `a`'s tasks and dependences first and `b`'s
after, and aliases a `b` BRIDGE_START to an `a` BRIDGE_START with the same processor identity, object and value. The
caller must align actual node and object coordinates before composing. These bindings add no identity guessing, and
they don't make arbitrary signatures unionable.

The planner's signature is `plan_signature` (`rdma/mesh-collective.c:1068`). It covers BSL submission waits and
`counts_tell`, BRIDGE_START, and the already selected program edges. Its `PLAN_COUNT + a` names an added edge, not a
transfer. This unit generates no per-transfer arrival or completion operations.

## Checks

| Check | Command | Result |
|---|---|---|
| Python syntax | `python3 -m py_compile rdma/mesh_graph.py` | clean |
| Library exports | `nm -gU rdma/libmesh.dylib` (the existing build, 17:40) | `_mesh_graph_archive`, `_mesh_graph_archive_read`, `_mesh_graph_compose` present |
| ABI roundtrip and compose | `/opt/homebrew/bin/python3 /tmp/planner-archive-check/check.py RECORD` (one-off, `/tmp`, not maintained), importing `rdma/mesh_graph.py` against the existing `rdma/libmesh.dylib` | below |

Records used are existing canonical signatures, read with `from_record`:

- `compiler/emitted/moe_three_processors.archive/signature.record`: 3 processors, 11 tasks, 12 operations,
  6 dependences, 6 synchronized edges.
  - Archive 2472 bytes, record 1040.
  - `archive -> from_archive -> archive` is byte-identical.
  - The read-back's `record` equals the original's.
  - Derived tables (position, previous, waited, first, order, count, unknown), counters (uncovered, redundant,
    unwaited, states) and digest are equal.
  - The printed forms are equal.
  - An archive truncated by 8 bytes fails with `[Errno 22] Invalid argument`.
  - `compose(a, a)`: 3 processors, 22 tasks, 24 operations, 12 dependences, 12 synchronized edges. Its archive
    roundtrip is byte-identical after both inputs were freed.
- `compiler/emitted/conv_cpu_mps.archive/signature.record`: 2 processors, 11 tasks, 11 operations, 9 dependences,
  7 synchronized edges.
  - Archive 2192 bytes, record 1104.
  - The roundtrip and table equality checks all pass, and the truncated archive gives `EINVAL`.
  - `compose(a, a)`: 2 processors, 22 tasks, 22 operations, 18 dependences, 14 synchronized edges, with an identical
    composed roundtrip.

**Limits.** No on-disk record contains a BRIDGE_START operation, so BRIDGE_START aliasing in `compose` is not
exercised. The candidates checked were the four `compiler/emitted/*/signature.record` files and
`executor/engines-linear-artifact/signature.record`, and none has one. No planner (`mesh_plan`) signature was
produced, because there is no bridge or run in this unit. Composing a signature with itself is an ABI check only.
It is not a meaningful program composition. Actual planner retention evidence belongs to the schedule worker's
`solved` change and its existing planning route.

## Consumers (other owners)

- Schedule worker: `Plan.signature = mesh_graph.archive(made.signature.contents)` in `schedule.solved`, before
  `mesh_planned_free`.
- Executor: `archive.py` encodes and decodes the bytes. `signature.py` reads them with `from_archive` and composes
  with `compose` after aligning identities.
