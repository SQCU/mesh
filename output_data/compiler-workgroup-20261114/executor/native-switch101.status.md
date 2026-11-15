# Indexed switch encoding through lower_metal and the Builder: status

Opus 5.5, SAME047. **SOURCE-READY; HOLD.** No import, build, run, stage or commit. Files: metal.py (+39/−1) and
lower_metal.py (+41/−6). native.py and Bound are untouched; SAME974's output and destination hunks are not touched.

> use EXISTING operations._copy ... to emit yielded Tensor -> common output Tensor writes INSIDE EACH arm, before
> recording its end operation ordinal. Keep original yield Tensor/ValueID in metal_ir_switch source correspondence.

## C pairing (metal.py, canonical `_compiler()` table; no lazy binding)

- **`metal_ir_switch`:** `[P, U, U, POINTER(U), POINTER(U), U, POINTER(U), POINTER(U)] → U`.
- **`metal_ir_switches`:** `[P, POINTER(size_t)] → POINTER(SwitchRegion)`.
- **`metal_ir_switch_values`:** `[P, POINTER(size_t)] → POINTER(U)`.
- **`SwitchRegion`:** 12 u32 fields (index, branches, branch, outputs, yielded, results, parent, parent_branch, select,
  order_first, order_last, padding), 48 bytes, `select` at 32. This matches `metal_ir.h:303–325` (uncommitted).
- These resolve when `_compiler()` is set up, so the library must be the matching build.

## Builder (`_Builder.generic` → `switch`, beside `repeat`)

For each branch, in order:
1. `first[k] = metal_ir_operations`.
2. Each record through `generic`, so nested switches and repeats register themselves.
3. For each (output, yield) pair, `operations._copy(output, yield)`, placed by `_placed` at the switch's own
   place and node, through `generic`. These are ordinary `Emitted` leaves inside the arm.
4. `end[k] = metal_ir_operations`.

Then:
- `yielded` is the yields' own `value()` ValueIDs, branch-major, and `results` is the outputs' `value()`.
- `metal_ir_switch(p, value(index), branches, first, end, outputs, yielded, results)` is called once.
- It returns None, since a switch is no operation. No region is retained, because nothing consumes one. Nothing is
  appended after the switch.

## Producer (`lower_metal.construct`)

- **`get_attr` nodes:** resolved through `fetched(module, target)`, `Interpreter.fetch_attr`'s attribute walk on
  assemble's owner-bound module, and on each branch GraphModule for nested `get_attr`.
- **`instructed(node, read, outs)`:** one recursion. A non-switch node goes to the existing `_local_call` and
  `_instruction`. A `switch_op` node becomes `map_arg(node.args)`, giving (index, GraphModules, operands); then:
  - each branch graph's placeholders are zipped strictly with the operands;
  - each other node is a `get_attr` object or `instructed` (nested switches through the same path);
  - the yields are the output node's leaves;
  - the result is one `_Switch(index, branches, outputs)`, with outputs `outs` (dests, as `_instruction` takes them)
    or `operations._made(node.meta["val"], node)`.
- **The main loop's call branch** uses `instructed`. Its `_placed`, `owners` and generator-advance handling are
  unchanged. The switch node's own task therefore owns its whole interval: the branch leaves and the in-arm copies,
  through `emitted`/`vertices`.

## Pairing notes (not workarounds)

- **e48:** each branch interval ends after that arm's result copies. C only retains (yield, result) pairs and
  membership, per the clarified contract.
- **SAME64:** `_Switch.index` must be a one-element integer tensor (the compiler's int32 index). `outputs` and yields
  must be tensors. `_made` gives None for a non-tensor leaf, which `value()` cannot take.
- **SAME974:** a destined switch node gets `outs = [dests[node]]` by the existing `_instruction` convention. Its
  multi-output semantics are that owner's.
- **Checks:** `py_compile`, `pyflakes` and `git diff --check` pass. No new line is over 120 characters
  (lower_metal's 2 were already there). The AST shows `_Builder.switch` exists and the three switch symbols are
  registered.

## Update (switch102: explicit parent)

> Nested switch registration returns region ID, attach to its actual immediate enclosing switch's children with that
> enclosing branch ordinal.

- **Registration:** `metal_ir_switch` is now `[P, U, U, *U, *U, U, *U, *U, U, *U, *U] → U`: after `results` come
  `children`, `child_regions` and `child_branches`. This matches `metal_ir.h:299–301` (uncommitted). `SwitchRegion`
  is unchanged at 48 bytes.
- **`_Builder.switching`:** a construction stack of the open switches' `[children, open branch]`, innermost last.
  - `switch` pushes its entry, sets the branch ordinal as each branch opens, and pops it after its last branch.
  - It then registers in postorder with its own direct children, `(region, branch)`.
  - If a switch is still open, the returned region goes into the innermost open switch's children, with that switch's
    open branch.
  - A `_Repeat` between two switches pushes nothing, so the enclosing switch's context holds through it.
  - There are no interval scans and no record filtering.
- **Operands:** the index, yields and outputs are named as `value(_stored(t))`, the complex/wide representation the
  Builder already uses for operands such as `dep_token`.
- **Checks:** `py_compile`, `pyflakes` and `git diff --check` pass, with no new line over 120 characters. The diff is
  metal.py +53/−1 and lower_metal.py +41/−6 (unchanged in this update).

## Update (encoder104: mixed region children)

> Use ONE compile-time source construction stack across BOTH kinds.

- **ctypes, all in `_compiler()`:**
  - **`RegionChild`:** `{kind, region, arm}`, 3 × u32 = 12 bytes, with kind constants `REGION_REPEAT = 0` and
    `REGION_SWITCH = 1`.
  - **`metal_ir_switch`:** its first 8 arguments, then `U` child count and `POINTER(RegionChild)`.
  - **`metal_ir_repeat`:** its first 6 arguments, then `U` child count and `POINTER(RegionChild)`.
  - **Moved from the lazy helper:** `metal_ir_repeats`, `metal_ir_repeat_carried` and `metal_ir_dispatch_repeats`.
    The `REPEATS`/`_repeats` cache is removed; `Builder.repeat` was its only caller.
  - **`RepeatRegion`:** its padding is now `parent_kind`, with `parent_branch` appended, for 44 bytes.
  - **`SwitchRegion`:** its padding is now `parent_kind`, still 48 bytes.
- **Pending e48's header:** `metal_ir.h` (uncommitted) still declares the earlier same-kind child arrays. The struct's
  C name and the kinds' numeric values are not published yet. **The kind values 0 and 1 above must be checked
  against the published enum before the build.** The layout is as the order states.
- **One stack, `_Builder.regions`:** the open regions, innermost last, each `[direct children, open arm]`.
  - `repeat` pushes `[children, UINT32_MAX]`. `switch` pushes `[children, k]` and updates k as each branch opens.
  - Each pops after its records and registers in postorder with only its direct mixed children.
  - `enclosed(kind, region)` then adds `RegionChild(kind, region, open arm)` to the innermost open region.
  - There are no interval scans and no same-kind trees. The stack exists only while records are encoded.
- **Unchanged:** the in-arm `_copy` of each yield into its common output, with `value(_stored(…))` provenance; the
  lower_metal recursion; native.py.
- **Checks:** `py_compile`, `pyflakes` and `git diff --check` pass, with no new line over 120 characters (metal.py
  +107/−32 vs HEAD).

Supervisor pairing: published metal_ir.h declares REGION_REPEAT=1 and REGION_SWITCH=2; metal.py corrected to those exact values. RegionChild12B, Repeat44B, Switch48B and constructor argument order match. Source only; native consumers/build remain pending.
