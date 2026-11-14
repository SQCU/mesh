# Emitted prefix/Step source snapshot: status

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `047e12be-cd27-4637-a3bb-7d46f6e8216b`. Unit:
[eigh-source-capture.md](eigh-source-capture.md), at mesh HEAD `beeb31f0`, with the installed libraries. **HOLD.** One
run, no production edit, no new ABI, no `Program` and no invocation.

## Run and artifacts

- **What ran:** `compiler/eigh-native-command.md`'s client, `compile_module` only.
- **The hook:** the same live `return` hook at `native.Target.placed` with `finalize=True`.
  - The getters, bound locally with header-exact ctypes: `metal_ir_source`, `metal_ir_bindings`,
    `metal_ir_effect_views`, `metal_ir_sources`, `metal_ir_operation_info`, `metal_ir_value_placement` and
    `metal_ir_value_shape`, plus `metal.py`'s dispatch and numerics getters.
  - They were read in that frame. Then a sentinel was raised.
- **Result:** process **exit 0**, with the **sentinel before ProgramMake**: `{"sentinel": true, "program_made":
  false}`.
- **Emitted source (for e48's paired review):**
  - [eigh-source-row0-kernel0.txt](eigh-source-row0-kernel0.txt), the prefix kernel: 18262 bytes, sha256 `8caefdb6…`;
  - [eigh-source-row1-kernel0.txt](eigh-source-row1-kernel0.txt), the first Step kernel 0: 38772 bytes, sha256
    `3630edad…`.
- **Facts:** [eigh-source-capture.json](eigh-source-capture.json); local stdout is 50,549 bytes, SHA-256
  `de03d8b0e4d12b3756c0214a7a65e8956b63ddba309dcd51ab43ec07f65d9151`; stderr is empty, SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. The final stdout record is
  `{"sentinel":true,"program_made":false,"stopped_before_ProgramMake":true}`.

## Typed correspondence (from the actual records)

**Step leaves:** the first Step body's retained emitted leaves (`ConstructedMachine.emitted`; 2496 leaves, the first 12
captured). Leaves 0–8 are one operation each, with op ID equal to the leaf index. Leaf 9, a `_Sort`, spans operations
9–14, so leaves 10 and 11 are operations 15 and 16. **All are in row 1's dispatch 0, kernel 0**, by
`metal_ir_sources` membership.

| Leaves | What they are |
|---|---|
| 0–2 | the copies: f32 4×4, f32 4×4, i32 5×2 |
| 3 | **the pop:** three i32 scalars (offset, size, below). It is identified exactly as the producer of the next leaf's symbol tensors. |
| 4 | **the `h` load, the symbol consumer:** f32 4×4, symbols (i32, bound 4) and (i32, bound 3), both loops clipped by symbol 0 |
| 5 | squares/offdiag: two f32 scalars, symbol bound 4 |
| 6 | terminal: two i32 |

So no other kernel was needed for the pop or the symbol consumer.

**The two dispatches:**

| Dispatch | Kind | Groups × threads | Bindings | Source ops | Effect records |
|---|---|---|---|---|---|
| Prefix, row 0, dispatch 0 (kernel 0) | KERNEL | 1 × 32 | 8 | 28 | 12 |
| First Step, row 1, dispatch 0 (kernel 0) | KERNEL | 1 × 32 | 22 | 21 | 76 |

Dispatch-to-numeric-kernel and member scopes are in `step_numerics`.

**Bindings and placements:** each binding lists `ValueInfo`, the compiled shape and layout, its root, and the root's
placement.

| Kernel | Binding (argument-buffer id) | Value | Access | Arena placement |
|---|---|---|---|---|
| Prefix | slot 2, `[[id(2)]]` int | value 6, the agenda output | write | @128, 40 bytes |
| Prefix | slot 1, `[[id(1)]]` int | value 8, the top output | write | @400 |
| Prefix | slot 5 | value 2 (h) | read/write | @64 |
| Prefix | slot 3 | value 4 (vectors) | write | @3584 |
| Prefix | slot 4 | value 0 (the input) | read | an operand, not in the arena |
| First Step | slot 4, `[[id(4)]]` const int | value 4, the agenda input | read | @128 |
| First Step | slot 6, `[[id(6)]]` const int | value 6, the top input | read | @400 |
| First Step | slot 0 | blocks input | read | @64 |
| First Step | slot 2 | vectors input | read | @3584 |

- **The match:** each Step input reads the arena range its prefix output writes.
- **Step kernel 0's temporaries:** 14 table-0 bindings, at disjoint arena ranges. None overlaps its inputs or outputs.

## As read in the emitted prefix source (evidence for review, not a diagnosis)

- **`top = 1`:** `mesh_member_1` stores int `1` to `[[id(1)]]`, top's binding, element 0.
- **Agenda:** `mesh_member_6` writes the 10 agenda elements (`[[id(2)]]`). Element `[row*2 + col]` gets `select(sym9[0],
  sym8[0], cond)`, where `cond` reads the threadgroup arrays `sym10[col]` and `sym11[row] == 1`.
- **Ordering:** in `mesh_kernel_body`, members 2–5, which fill those threadgroup scalars and masks, come before a
  `threadgroup_barrier(mem_threadgroup)`, and member 6 comes after it.
- **Paired review complete:** [eigh-emitted-review.status.md](eigh-emitted-review.status.md) reads members 2–5 and
  confirms the selected constant and mask, then checks the Step extraction and symbol loads against slots 4 and 6.
  The emitted expressions and recorded operands agree; runtime execution is not observed by this capture.

The numerical cause is not claimed.
