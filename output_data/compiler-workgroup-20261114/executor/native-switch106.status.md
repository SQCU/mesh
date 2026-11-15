# native-switch112: region query and observation corrected; switch execution NOT complete; source HOLD

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).
e48, claude-opus-5-5, source only. Strict rc=0: 8 METAL_IR_SRCS, capture, program, 2 tools. metal_recording.m untouched.

## This unit
- **One region forest:** `RegionForest` makes one `mesh_graph_search` from the supplied parent fields, at
  `metal_ir_construct` and the archive's load; specialize shares it.
  - `RegionAbove` uses `mesh_graph_climb`, and `RegionWithin` uses `mesh_graph_ancestor`.
  - `RepeatAbove`, `SwitchAbove`, `SwitchWithin`, `SwitchRepeat` and `RepeatWithin` are now one-line uses of these
    queries. Their walkers are deleted.
  - The ObjC `MetalRepeatAbove` is deleted and replaced by the new C ABI `metal_ir_region_above(p, kind, region,
    above, *arm)`.
- **Observation:** a ranged span keeps its existing GPU observation unit, so its actual timing is measured, including
  an empty selected range.

## Remaining, exact (same owner next)
1. **`metal_lower.mm` `Builder::Write`:** a `METAL_IR_RANGES` output emits a call to that kernel's retained typed
   function `mesh_switch_ranges` (a view of the table plus one u32 parameter), whose body stays empty. `Emit` then
   defers that kernel's pipeline.
2. **`metal_transport.mm`:** add to the same function the views of the arms' later INDIRECT arguments, and their
   effects.
3. **`metal_capture.m` replay:** one grouped-span plan from `cutting` plus parts, pass cuts and blits.
   `metal_program.m` `MetalNativeCommands` consumes that plan instead of deriving its own runs.
4. **New `metal_ir_switch_specialize(p, kernel, spans, count, …)`:** called by replay before encoding. It emits the
   unrolled stores `length[w] = k == arm ? count : 0` and `args[0] = 0` once, then compiles through `Emit`'s
   per-kernel owner. Replay allocates the table rows.
5. **`MetalNativeICB`:** set `ranges` from the plan's row addresses. That selects `NativeICBRangeEncode` at emission.
6. **`Obligations`/`Completions`:** one joined writer after the switch.
7. **`LowerProgram`:** CPU indexed branch.
8. **`LowerNeural`:** the tree through MIL `Cond` and storage SSA.
