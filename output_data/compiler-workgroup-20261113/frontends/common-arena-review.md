# Common planned arena: preparation and relocation

2026-11-13. The exact requirement-7 operand is the planner's complete mapped storage result, replacing the
intermediate per-value roots in checkpoint `7dc04827`.

Compiler producer `628852cf` records `Plan.arena_bytes` and `Plan.offsets[value][copy]` from `mesh_view_values`
immediately after `mesh_plan`. Every copy and every absent-value sentinel survives; no process pointer is archived.
Archive save/load retains both fields and restores the nested offset tuples.

The same frontend Claude session `9740398d-f513-4c78-8578-f80b79fa2225`, reported `claude-opus-5-5` on harness
2.1.321, implements `Published.arena`: one uint8 tensor over the actual loaded `mesh_view_values` address/span.
Zero span is an empty byte tensor. Per-transfer copies, roots, logical/distinct view maps and view reconstruction
are deleted, along with unused carried/here constructor operands. Source Plan, native function/held records,
context, loaded plan and invocation lifetime remain owned by Published. Raw turn 10 and `common-arena-5.md` are
under metal-microbench `output_data/compiler-workgroup-20261113/frontends/torch`.

Paired executor source now saves the actual arena input position and binds only Published.arena through
`Model.prepare`; Bound retains Published. The driver passes the complete offsets for each declared value and the
arena span. The body takes one arena input, retains the full offset table as model data, selects
`execution % copy_count`, and derives typed views inside the graph. The old getattr-based transfer-map consumer
is removed. Source inspection followed that dynamic consumer explicitly; an earlier direct-field grep alone did
not establish it had disappeared. This closes the source handoff; final lowering and execution evidence remain
separate.

## Why the saved offsets survive load

This follows from the current allocation/load owners, rather than a comparison of observed pointers:

1. `mesh-collective.c:plan_block` packs every `run` value/copy into the single `x.run` allocation. `plan_items`
   places that item in exactly one selected arena; `plan_frame` routes every run value through that same item.
2. `mesh-dataflow.c:mesh_view_values` computes its base/span using only run values. Non-run values contribute
   UINT64_MAX offsets and do not enter the span. Relay and state items in another arena therefore cannot make
   this span straddle two independently translated arenas.
3. `plan_saved_page` maps a saved page p to `new_arena.first + (p - saved_arena.first)`. `mesh_plan_load` preserves
   page/block size and whether arenas coincide. All included run pages receive the same translation.
4. Both each copy's address and the common minimum address move by that translation. Their difference and the
   common span remain unchanged. Thus all Plan.offsets entries, including physical reuse and all copy residues,
   remain valid when the actual relocated base is bound. Copy zero is not substituted for later residues.

The compiled view address is
`arena_base + offsets[value][execution % copy_count] + declared_byte_offset`.
Only the base and execution are live operands; the saved full offset table is already the solved result.

## Checks and limits

Supervisor bridge `py_compile` and `git diff --check` pass. The worker's harness denied its Bash checks; the
supervisor ran the authorized checks directly. This source proof is not a live bridge relocation measurement.
Actual native body/alias compilation and connected execution are the executor's paired validation work. No new
allocator, alias solver, source parser, runtime transfer adapter or publication lifecycle is introduced.
