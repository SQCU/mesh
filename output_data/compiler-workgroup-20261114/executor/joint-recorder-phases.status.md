# Multi-function recorder lifecycle: status

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `047e12be-cd27-4637-a3bb-7d46f6e8216b`. Unit:
[joint-recorder-phases.md](joint-recorder-phases.md). **SOURCE-READY, HOLD.** No target run, no C, header or ABI edit,
no build. The original malformed files are untouched.

## The contract read (metal-microbench metal_capture.m)

- **`MetalRecorder -prepare:` (`:362–363`):** it raises `prepare: after this recorder's replay:` once `_replayed` is
  set.
- **`-replay:` (`:885–900`):** it sets `_replayed` on its first call, after `MetalPlaced` and `MetalInitialContents`
  read that capture's placements.
- **So, on one recorder:** every capture must be prepared before any replay. Placement belongs between a capture's
  prepare and its replay.
- **`metal_ir_value_placed` and `metal_ir_step_placed`** copy their offsets (`metal_ir.mm`, `offsets.assign`), so a
  placement's temporary arrays need not outlive its call.

## Correction (native.py `Target.placed` only)

- **First pass, every original row in order:**
  - a row with no program is kept as `(row, None)`;
  - every other row is transport-prepared from its marks, then `metal_ir_finalize` (compile) or `metal_ir_loaded`
    (load), then `MetalRecorderPrepare` (its release registered with `owned`), then its placements (wires, run values,
    joined roots, the step, the operands' logical copies), the state buffer, then `metal_ir_place`.
  - The capture is kept as `(row, ready)`, and the transport owners stay registered with `owned`, as before.
- **Second pass, the same captures in original row order:**
  - `MetalRecorderReplay` (its destroy registered with `owned`), then `M.Call(pointer(replay), row.planned + 1,
    None, 0)`;
  - a `None` row is `M.Call(None, row.planned + 1, None, 0)`.
- **Unchanged:** one recorder, one solution, no per-row allocation or plan, no compile on bind, no recreated
  capture, no alternate path. A partial failure in either pass releases everything registered so far through the
  existing `Owned`.
- Bound, Work, observe and numerical code are untouched.

## Checks

- `py_compile` and `pyflakes` pass for `native.py`, no line is over 120 characters, and `git diff --check` is clean.
- The diff is 14 insertions and 3 deletions in `Target.placed` alone.
- The same 4x4 MPS retry is the compiler supervisor's.
