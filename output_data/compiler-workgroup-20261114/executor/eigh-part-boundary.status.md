# Canonical part boundary handoff (eigh-part-boundary.md) — SOURCE-READY, HOLD

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5. Source checkpoint
`5b7fa388` on top of `1f605fa9`.
- No shared build, import, target run or commit.
- Original malformed `metal_recording.m` is untouched and was not compiled.

## Correction (metal-microbench `metal_capture.m`, `-prepare:`'s per-dispatch ordered loop, +3/−4)

**Before.** A part boundary (the capture's GPU command count) was appended only in the host-step branch, and at most
once per host step. That relied on every later part starting with a host step.
- A repeat exit can start an ordinary GPU part, and that transition was lost.

**Now.** At the head of the loop, before the kind switch, for every retained dispatch in `metal_ir_order`:
```objc
const uint32_t before = (uint32_t)(capture->dispatches.length / sizeof(struct MetalRecordedDispatch));
while (capture->boundaries.length / sizeof(uint32_t) < part[d])
    [capture->boundaries appendBytes:&before length:sizeof before];
```
- The host branch's own append is removed.
- `command.length = part[d]` is kept unchanged, as is every command, effect, order and ID.

**Resulting ranges** (`MetalOrder`: `parts[0] = 0`, `parts[q+1] = boundaries[q]`, `parts[bounds+1] = count`):
- each canonical part q ≥ 1 has exactly one boundary, the GPU command count before its first dispatch;
- so the capture has the P parts `metal_ir_parts` assigns, with the same IDs;
- a host-only part's GPU range is empty (equal boundaries);
- a GPU-first part, such as a repeat's exit or the last all-GPU tail, begins at its own first command;
- the final part ends at `count`.
- Later passes still record outside `capture->dispatches`, as before. `capture->boundaries` has no other reader.

## Decoder consequence (actual)

**What happened.**
- PrepareOrder made 5 parts per Step, matching the planned `call.parts = 5`.
- The capture recorded three internal boundaries, so `MetalReplayParts[0]` / `MetalPartEntries` gave 4 entries per Step.
- `MetalProgramRows`, which derives P from that count, expanded each Step into 4 rows: 30 rows in all, against 37.
- The Step's last canonical part, the accepted tail after the outer repeat, was therefore an **omitted row**: it was
  never offered to MetalQueued or emitted.
- The earlier parts' command ranges were shifted at the lost transitions.

**Source consequence.** The capture now consumes every canonical part transition; the next matched client should
retain 5 parts per Step and 37 rows.

The exact per-part runtime ranges are the next same-client check on a matched build. They are not observed here.

## Checks

`cc -std=c11 -fblocks -O2 -Wall -Wextra -Werror -fsyntax-only -I . -I ../dox/mesh/rdma`:
- `metal_capture.m`: rc=0;
- `metal_program.m`: rc=0.

**Changed:** `metal_capture.m` only. Supervisor omitted the added prose comment and reran the exact capture strict
command successfully; `eigh-part-boundary-syntax.log` is empty. HOLD.

The runtime state evidence is checkpoint `fbba3d56`; final zero conditions do not prove zero iterations. The earlier
retained repeat spans were captured during the pre-430a9fe1 root652 diagnosis and served only as a source lead. The
current planned five parts and prepared thirty rows directly establish the count mismatch. No new runtime range
claim is made from that earlier diagnostic.
