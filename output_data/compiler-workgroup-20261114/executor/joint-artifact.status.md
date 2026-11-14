# Joint function artifact: status

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `047e12be-cd27-4637-a3bb-7d46f6e8216b`. Unit:
[joint-artifact.md](joint-artifact.md), with `joint-artifact.inbox.md` read top to bottom (233 lines, through
"Canonical local archive owner FINAL").

**Source state:** connected and source-checked, not run. The working tree holds `compiler/mesh_compiler/native.py`,
`archive.py` and the granted `lower_metal.py` hunk. Nothing is built, imported against a shared binary or committed.
The poisoned originals (`rdma/mesh_compile.py`, `lower_coreml.py`, `examples/tp.py`) are unchanged. Live imports are
`.metal` and `.operations` only.

## Codex review of this source checkpoint

This is an incomplete caller refactor, not executable joint-program evidence. Private stub checks do not validate
actual planning, native invocation or archive loading. The source now consumes resolved storage ownership, real
copy generations and explicit RNG initial words; the actual joint planner/archive producers are still changing.

A remaining error was found after the worker check: changing a prepared Range.owner to the current call does not
suffice because metal_ir_value_received matches the pre-lowering declaration's transfer AND owner. The authorized
replacement retains numerical construction before Transport, specializes it with actual per-call port declarations,
and only then produces the actor/storage result for the single joint plan. This will displace the current first-row
declaration and late owner rewrite. No transport or repeated-call runtime correctness is claimed here.

Likewise ports recorded with an absent output leaf are unresolved compiler operands, not executable returned values.
Actual output ownership comes from placement of the captured outer output Node, distinct from the local entry.
The output owner must receive its actual canonical transfer/root binding; non-output nodes retain protocol work.
The current Bound result path is not a fallback for those missing bindings.

## 1. The producer contracts as consumed (present facts)

- **`schedule.analyse(captured) -> CallAnalysis`:** native reads `nodes` (call j's original Node), `keys` (function
  ordinal F = position) and `calls[j]` (call j's F).
- **`schedule.solve_program(analysis, functions, bindings, quantum, planning=None, *, declarations, wire_values)`:**
  called once.
  - **Input `bindings`:** native's `Binding(call, function, inputs, outputs)`, with `Slot(slot, root, view, offset,
    reference)`.
  - **`declarations`:** the keyed Declarations.
  - **`wire_values[key]`:** `{declaration value index: ConstructedMachine.wires[value].root}`.
- **Result (`schedule.Solved`), the fields native reads:**
  - **Planning results:** `functions` (Function[F]), `planned` (`call[j].ordinal`/`.every`), `roots` (`mesh.Value[]`:
    `.run`, the allocation authority; `.shares`; `.section.bytes`), and `root_sources[s]`: `('input', outer flat
    placeholder position)`, `('result', s, bytes, device)` or `None` for a planned root.
  - **`bindings`:** `(call, root ValueID, storage s, delta, original reference, every, ordinal)`.
  - **First-views and offsets:** `function_value_first[f]` and `root_first` (first-views over `local.first`), and
    `offsets` (a View whose `.items` is the `local.offset` pointer).
  - **Archive and transport:** `saved()`; and, under planning, `context`/`view`.
- **`schedule.loaded_program(bytes)`:** the same complete owner, through `mesh_local_load`, with no planning.
- **`ConstructedMachine.initial_words`:** a `uint32` tuple. Root granted the field to this unit, and it is implemented
  (section 3).

## 2. `native.compile_program(captured, machines, declarations, analysis, node, planning=None)`

1. **Bodies and rows:** a `Body` per `analysis.keys` entry, with `machine = machines.get(key)` (`None` without a body on
   this node), and a `Row(j, calls[j], planned=j)` per call. Every key and call is kept.
2. **`Target.lowered(context, calls)`,** for present bodies only:
   - `interface(k)` gives the stable slots: distinct INPUT/STATE input roots, the generator state, then every further
     output root except a CONSTANT. Every input and output root is joint-bound, so its function-local run is 0.
   - It records each placeholder's and output's `(slot, root, view, byte offset)`.
   - Transport declarations, with the wire index `wire.root` (`metal_ir_value_planned(program, wire.root, k,
     wire.root, 0)`).
   - `metal_ir_value_external(program, root, slot)` for each slot, then `metal_ir_lower` once.
3. **`Target.functions`:** per present body, `storages(explicit = slots ∪ inputs ∪ state ∪ received ∪ sent)` and the
   retained `MetalRecorderActors` (parts and relation, freed with the target). An absent body gets a zero-local
   `mesh.Function()`. It also returns the quantum.
4. **`Target.bindings`:** `Binding`/`Slot` with the original Call references. An absent body's calls get `Slot(None, …,
   reference)`.
5. **`solve_program(…)`,** once.
6. **`Target.resolved(captured, analysis, joint)`:**
   - **Rows:** `(ordinal, every)` from `planned.call[j]`.
   - **Per resolved binding:**
     - `roots[s].run` places the root at s's copies (`joined`: root, s, delta, every);
     - `('input', p)` gives an operand `('input', p)` whose offset is outer placeholder p's byte offset minus delta;
     - `('result', s, bytes, device)` gives that operand with offset −delta. One allocation per s per invocation.
   - **Generators:** one operand `('generator', F)` per body with state, over all its rows.
   - **Constants:** each CONSTANT root's outer constant name, from `Call.inputs`.
   - **Outer outputs** come from the output node:
     - a call result is `(j, Output)`, or `(j, None)` with an entry in `ports` where the producer has no local body;
     - an outer placeholder is `(None, Output(position))`;
     - a literal is `(None, Output(literal))`.
7. **`Target.compile`:**
   - per row of a present body, `metal_ir_specialize`, then `metal_ir_value_external(prepared, root, a)` for that row's
     occurrences;
   - `placed`:
     - the row's own received ranges get owner = that call, generation `{1, 1, 0}`;
     - wires at `function_value_first[f]`;
     - run values at their full copies, with their shares;
     - each planned root's copy table projected through e = t·every + ordinal, entry i = root copy i // every, plus
       delta, with `roots[s].shares` (a view in place when every = 1 and delta = 0);
     - the step at `count`;
     - each operand's logical placement.

     A row with no program is `MetalCall(NULL, call + 1, NULL, 0)`.
   - one `MetalRecorderProgram`, `MetalProgramPhases` and `MetalProgramSignatureNode`, and one `MetalProgramSave`.
8. **Owners and prepare:**
   - `Arena`: the joint local result, its base `C.cast(offsets.items)`.
   - `Mapped`: the same base under planning.
   - `Target.prepare`: `metal_ir_archive_program(r)` per row (NULL stays `None`). Constants are re-attached from their
     outer names through `metal._lent` (`metal_ir_value_data` adopts the owner), then placement and one
     `MetalProgramLoad`. There is one generator buffer per body.
9. **`Bound`**, changes in LiteRT41's method that joint binding needs; observation is untouched:
   - one allocation per result s (a membership check), shared by its operands;
   - `generator[F]`;
   - `(None, leaf)` gives a literal or `given[position]`.

**`archive.py`:**
- `channels`/`invariant`/`carried` are saved as `[{"function": ordinal, …}]`, the ordinal taken from the completed
  target's body order.
- Function keys, which may be callables, are not saved, and the recipe drops `Body.key`. `restore` gives each body its
  ordinal.
- The target saves `{key}.native` (the artifact), `{key}.joint` (`solution.saved()`) and the typed recipe: bodies, rows,
  operands, `joined`, counts/steps, generators, constants, ports and outputs.

## 3. `lower_metal.py` (root's grant)

- `ConstructedMachine.initial_words: tuple = ()`, after `.decomposed`, which is unchanged.
- `construct` returns `initial_words=tuple(word & 0xFFFFFFFF for word in generator.state.tolist()) if state else ()`:
  the actual `Generator.state` snapshot, only when the program draws. There is no Generator retention and no reseed.

## 4. Checks (private: AST, syntax and stubs; no shared binary)

- `py_compile` passes for `native.py`, `archive.py` and `lower_metal.py`, and there is no line over 120 characters in
  `native.py`/`archive.py`. `git diff --check` is clean, and an AST scan finds no undefined name.
- `metal_owner_check.py`: every `M`/`metal` name is defined, and every bound symbol is declared.
- `/tmp/archive-worker/scripts/joint_final_check.py` loads `native.py` from source with stub `metal`/`mesh`/
  `schedule`/`operations` on a real FX call graph: `f` is called twice (x → y0 → y1, with an outer constant `w`), and
  `g` has no local body. All 23 checks pass:
  - **Bindings and rows:** the absent body's calls keep their references only; rows `(j, F, ordinal, every)`.
  - **Operands:** the input operand is `('input', 1)` at offset 8 − 8 = 0, and the result operand is `('result', 2, 16,
    'cpu')`. The generator operand covers both `f` rows.
  - **`joined`:** `(20, s1, 0, 2)` in row 0 and `(10, s1, 0, 2)` in row 1.
  - **Constants and outputs:** constants map through `Call.inputs` to `"w"`. The outputs are the result, a passthrough
    position and a literal.
  - **Placements:** a function value in place with shares 2; the planned root projected through every = 2 to
    `[1300, 1300, 1400, 1400]` with shares 3; and the step.
  - **Round trip:** the recipe has no callable key, and the JSON round trip equals the original in 10 fields, with
    bodies restored by ordinal.
  - **Ports:** a remote producer's output is `(j, None)` with an entry in `ports`, not an indexed absent body.

## 5. Missing producer fields (exact)

- **compiler979:**
  - `Solved.root_sources` and `solve_program(…, *, declarations, wire_values)` landing as specified (current
    `schedule.py` still unpacks 4-tuple bindings and lacks `root_sources`);
  - `schedule.loaded_program`;
  - under planning, `Solved.context`/`view` (planning currently raises in `solve_program`);
  - the cross-function call port owner/generation for received ranges: today only own transfers are projected per
    row;
  - the zero-local Function's exact fields for an absent body. Native passes `mesh.Function()`, all zeros.
- **frontend974:**
  - the joint output port operands for outer outputs produced on another machine. `Target.ports` holds `(position,
    call, output)` until they arrive, and `Bound` has no binding for them yet;
  - `machines` holding only this node's bodies, as the inbox settles.
- **compilerdc4:** `bridge.Published` with `span(first, index)`, `placements(row)` and `stepped(row)`, as the
  compile-time owners have, and its `planned`/`context`/arena.
- **LiteRT41:** the three `Bound` edits above are in its method. Its observation entry is unchanged.

Earlier checkpoints (13 and 14 November) are in git history and the session; this file states the current source only.
