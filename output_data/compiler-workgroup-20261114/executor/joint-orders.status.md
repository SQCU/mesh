# Native joint call-order result consumer — status

Session e48f4bc7, explicit claude-opus-5-5. Unit: joint-orders.md, under the settled compiler grant. SOURCE-READY,
HOLD. No build, run or commit. The original `metal_recording.m` is unchanged (equal to HEAD).

## Agreed domains (by `cause[q].kind`, in every result)

| kind | `order[q]` endpoints |
|---|---|
| `MESH_PLANNED_PART` | part vertices `[0, 2·parts)` of call `pair = j` (its function's own edges) |
| `MESH_ORDER_FLOW`, `MESH_ORDER_CONFLICT`, `MESH_PLANNED_CHAIN` | call ordinals, with the edge's delay (CHAIN: 0 within a label, 1 last → head) |
| `MESH_PLANNED_PROGRAM`, `_SUBMIT`, `_OUTSTANDING`, `_TRANSFER` | signature tasks (network only) |

There is no new field or archive version. The full network signature, `planned->signature`, is unchanged.

## Producer hunk (mesh `rdma/mesh-collective.c` `plan_graph`, compiler grant)

**The `planned.order` copy, after the `plan_here` filter.** FLOW, CONFLICT and CHAIN edges are now written as
`{call[from[e]], call[to[e]], delay[e]}`. `plan_ordered` builds them work→work, `actor[a]+rank → actor[b]+rank`. The
bridge's other kinds copy unchanged.

**PART.** Was `{MESH_PLANNED_PART, UINT32_MAX}` with endpoints `base + f->edge[e]` (`base` starting at `V`). Now:
`{MESH_PLANNED_PART, j}`, endpoints `f->edge[e]` as given.

**Unchanged.** The order and cause arrays are still sized `built + within`. Nothing changes before the copy: the
built graph, the cycle mean, DetermineDelays and the waits.

**`rdma/mesh-collective.h`.** A comment above `struct mesh_planned_cause` now documents the domain table above.

## Decoder hunk (metal-microbench `metal_program.m` `MetalPlannedOrder`)

| kind | mapping |
|---|---|
| PART | `j = cause.pair`, `a = edge.from`, `b = edge.to`. Host part `a < parts` → `P + row[j] + a`; GPU part → `at[row[j] + a − parts]`. Skipped where `row[j]` is absent, and for a self-loop of delay 0. This is the existing correspondence. |
| FLOW/CONFLICT/CHAIN | `a = lastRow[edge.from]`, `b = row[edge.to]`; the edge is `{at[a], at[b], edge.delay}`, source last work → target first work. There is no start barrier. |
| other kinds | the existing signature-task branch, unchanged |

- **No-work rows.** A row with no numerical work (`MetalWork` false: a null replay/native row) is still a row and a
  protocol vertex. Its position is unkept, and the existing `mesh_graph_contract_unrolled` contracts through it. No
  wait or search is added.
- **Removed.** The global-concatenation base search, and the `planned->cause` NULL test. Every producer sets
  `cause`.
- **Unchanged.** The contraction, starts, tail/head and the relation solve.
- **Superseded wording.** My earlier draft wording ("destination first start", "the last working row", "a no-work
  protocol source") is superseded by the settled work→work role.

## Private strict checks (`-fsyntax-only -Wall -Wextra -Werror`, actual Makefile flags)

These checks are bounded evidence, not a current-source pass. The shared rdma headers are mid-edit by another
owner: `mesh.h`, modified at 01:24, defines `struct mesh_holder` twice, at :51 and :72.

- **`metal_program.m`.** Exactly 1 error, and it is that `mesh.h:72` redefinition. Nothing in this file.
- **`rdma/mesh-collective.c` (rdma Makefile flags).** The errors are all elsewhere, so none are in this hunk
  (:1386–1401):
  - `mesh.h:72`;
  - `struct mesh_planned` has no `rows` or `generation` (:2034–2040);
  - undeclared `plan_released` (:2433, :2991) and `mesh_algo` (:2791);
  - `plan_bind_values` arity (:2533, :2997).
- **Original `metal_recording.m`.** Still fails.

## Changed paths

- **metal-microbench:** `metal_program.m`.
- **mesh:** `rdma/mesh-collective.c` (the `plan_graph` copy hunk) and `rdma/mesh-collective.h` (the domain comment).
  Both files also carry other owners' uncommitted hunks, which I did not touch.
