# Scalar switch encoder — source held

Source checkpoint: `a00eb204`. SAME047 session
`047e12be-cd27-4637-a3bb-7d46f6e8216b`, explicit `claude-opus-5-5`.
No native import, build or invocation has validated this source yet.

- `lower_metal.instructed` consumes actual canonical branch GraphModules and
  explicit operands, recursively using the existing `_instruction` for leaves.
  The original source task retains every emitted leaf and branch result write.
- `Builder.switch` emits existing `_copy` writes inside each arm, before its end
  ordinal. Original yield/common ValueIDs use `_stored`; C registration moves no
  data and retains that provenance.
- One construction stack covers Repeat and Switch. Each constructor receives
  only its actual direct children as `{kind, region, arm}`, in source postorder.
  Equal/empty spans and mixed nesting need no interval hierarchy inference.
- Published C kinds are Repeat=1 and Switch=2. `RegionChild` is 12 bytes,
  `RepeatRegion` 44, `SwitchRegion` 48. Constructor argument order matches
  `metal_ir.h`; both append child count and child-array pointer.
- Repeat and switch APIs use `_compiler`'s canonical binding table. The displaced
  lazy repeat cache/helper is removed. No stale-library compatibility path.
- `native.py`, Bound, and the frontend output/destination normalization are
  unchanged by this unit.

Supervisor AST, pyflakes and diff checks passed on both edited Python files.
Native selector/range, result/completion and all-engine consumers remain the
paired C unit's obligation. Fresh imports stay held until its matching build.

## Update (native111: typed selector)

> Call at _Switch control point BEFORE arm operations ... Existing metal_ir_switch adds selector operation ID
> immediately before children.

- **Registration** (`_compiler()`, matching `metal_ir.h:306–315`, uncommitted):
  - `metal_ir_switch_selector` is `[P, U, U, U, Q] → U`;
  - `metal_ir_switch` is `[P, U, U, *U, *U, U, *U, *U, U selector, U children, POINTER(RegionChild)] → U`.
  - The kind values are the published `METAL_IR_REGION_REPEAT = 1` and `METAL_IR_REGION_SWITCH = 2` (already so in
    a00eb204). `RegionChild` is `struct metal_ir_region_child`, and both region layouts match the header's asserts
    (`parent_kind` at 36 in RepeatRegion, 44 bytes; at 44 in SwitchRegion, 48 bytes).
- **`Builder.switch`:**
  - It sets `identity = base | g.node`.
  - At the control point, before any arm operation, it calls `metal_ir_switch_selector(p, value(_stored(index)),
    branches, g.place, identity)`. The selector is an ordinary operation, kept as `Emitted(_Switch, selector, …)`, and
    lies inside the switch record's operation range, so lower_metal's emission loop and the source vertex own it.
  - Then come the arms, with in-arm `_copy` as before, and `metal_ir_switch(…, selector, children…)`.
  - The index ValueID is named once and reused.
  - The leaf consumers (`associate`) take only `_Generic` and `_Lapack` records, so the switch's `Emitted` changes
    nothing there.
- **Checks:** `py_compile`, `pyflakes` and `git diff --check` pass, with no new line over 120 characters (metal.py
  +19/−13 vs 774f08e4).

## Update (selector106: plan before final emission)

> Target.placed new order in EACH actual row

- **`Target.placed`** (native.py, comment `:442–448`). Each row now runs:
  1. transport (unchanged);
  2. `MetalRecorderPrepare` (`:494`), on the lowered typed program;
  3. value, step, operand and state placement, then `metal_ir_place` (`:509`);
  4. `checked(lib.MetalRecorderPlan(recorder, ready, error, ROOM), error, "replay plan")` (`:510`);
  5. the existing emitted call, `metal_ir_finalize` or `metal_ir_loaded`, moved here, not duplicated (`:511–512`);
  6. `captured.append`.

  The replay loop and the Owned order are unchanged.
- **`_recorder()`:** `('MetalRecorderPlan', C.c_int, [P, P, text, room])` (metal.py `:386`).
- **C pairing:** native122 now supplies the matching `MetalRecorderPlan` declaration and caught wrapper. Its retained
  plan reaches final emission and archive relocation; the existing tools use the same chronology. Imports remain
  held until the matching build. Full range-access geometry and other conditional consumers remain unfinished.
- **Checks:** `py_compile`, `pyflakes` and `git diff --check` pass. My hunks are native.py +5/−3 (3 hunks) and
  metal.py +1. metal.py's one line over 120 characters (`:685`) is SAME64's uncommitted scalar-if hunk.
