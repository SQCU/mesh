# Native check dependency operand (native-check82) — SOURCE-READY, HOLD

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5. Pairing for
integration-matrix-exp-error.status.md, Correction 2.
- No build, target run, evaluator, stage or commit.
- Remote fleet state untouched. operations.py, driver, schedule, export and lower_metal untouched.
- Original malformed sources untouched.

## Producer → retained owner → consumer

1. **Producer (Python, `compiler/mesh_compiler/metal.py`).**
   - Registration: `('metal_ir_check_dep_token', None, [P, U, U])` beside `metal_ir_check_message`.
   - `Builder.encoded(_Check)`: after the existing `metal_ir_check(p, value(g.info), code, identity)` and its
     message, `if g.dep_token is not None: metal_ir_check_dep_token(p, operation, value(_stored(g.dep_token)))`.
   - `_Check.dep_token` is set by `_asserted` to the actual destination or result (operations.py, owned elsewhere).
2. **Retained owner (`metal_ir.h`/`metal_ir.mm` `metal_ir_check_dep_token`).** It appends `value` to that Check
   operation's `gathers` directly under the constructed-operation/value identity invariant. The supervisor removed
   the draft silent-drop guard after root review.
   - **Existing whole-value gathered read.** `OperandEffects` emits `Whole(value)` (read) for every `op.gathers` entry
     not in `op.inputs`. A Check's own `inputs[0]` status read is unchanged (`Loaded`: `k == 0`).
   - **Already retained by existing owners:** Accesses (operation effects, Transform's relation and `Proven`
     confinement), `Realize` (it keeps `op.gathers`), `AppendSource`/clone (they remap `op.gathers`), and
     specialization (it copies operations).
   - There is no fake map over the status box, numerical load, extra buffer, second relation solve or global wait.
3. **Consumers.**
   - **Ordinary CHECK effects.** `ConstructKernels` builds the CHECK kernel from the Check operation. `Artifact` takes
     its views from `Effects(op)` (the kernel's operations are non-empty), so the dispatch's effects include the
     dep_token's whole read. The dispatch relation then has the existing writer → CHECK conflict edge.
     - The existing host-dependency owner (planned waits; a host step reading GPU-written storage is host-obliged)
       orders the writer's completion before the status read. No new wait is added.
   - **The one concrete downstream requirement: a binding.** metal_capture.m keys every effect view by a slot binding
     its storage in that direction, raising "does not bind so" otherwise. So `ConstructKernels`' CHECK branch now also
     binds each gathered value's storage once:
     - access 1, alignment 1 (no typed access), bytes `Bytes(p, root)`;
     - the next slot. The status binding stays at slot 0.
     - This is a binding of the existing value, not inputs membership or new geometry.
   - **Failure and cancel unchanged.** `metal_ir_check_info` and `NativeChecked` still read only binding 0 (the status
     base, `pointers[0]`). The capture's CHECK record keeps `info` at `pointers[slots]`.
   - **Lifetime.** Storages takes the binding as the dep_token's lifetime and last use through the check.

## Files

- metal-microbench: `metal_ir.h`, `metal_ir.mm`, `metal_lower.mm`.
- mesh: `compiler/mesh_compiler/metal.py`.

## Checks

- **C:** strict `-fsyntax-only -Wall -Wextra -Werror` with the actual Makefile `TINT_CFLAGS`, rc=0 on:
  - all 8 METAL_IR_SRCS;
  - `tools/native_bench.m`, `tools/e2b_replay.m`;
  - `metal_capture.m`, `metal_program.m`.
- **Python:** `metal.py` parses with `python -I` `ast.parse`.
- `metal_recording.m` is unchanged. No shared library was rebuilt.

## Remaining gate

The root's public `out=` observation must be re-run on a matched canonical build. Not run here.

HOLD.

Supervisor review: actual Binding defaults access=1/alignment=1, capture consumes the token effect through its retained slot; root accepted this exact binding. Final metal_ir.mm strict syntax passed again after removing the silent guard. Header rationale reduced to source citation. Eager Python registration requires the matching library before fresh construction. No runtime success claimed.
