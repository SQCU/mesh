# Numerical construction and call-instance source checkpoint

14 November 2026. Native e48 and artifact047, same explicit Opus 5.5 sessions; Codex source review. This is a source
checkpoint, not a connected execution or performance result.

`metal_ir_construct` owns the existing Verify/Transform and numerical kernel construction. Its retained kernel
result includes selected mapping, locals and actual created ValueIDs. `metal_ir_specialize` accepts that result
through the existing typed Clone owner. `metal_ir_lower` consumes the constructed instance's ports, adds the existing
transport/check/step structure, then produces Artifact/Actors/Storages for the one joint solve. Device ownership is
fixed by construct; lower has no redundant device operand. Both direct C clients call the stages explicitly.

The current Python consumer constructs each present numeric body, specializes each original call, declares that
call's intrinsic ports and lowers it before solving. Original source function identity remains `Row.function`;
actual instance identity is `Row.instance`, from the planned call. Placements/counts/steps index the latter. The
saved row retains its own port records; the old first-call prepared-owner rewrite is deleted. All actual instances
and original call bindings go to one `solve_program(..., plan_calls=...)`. The actual source is
`metal_ir.h`, `metal_ir.mm`, `metal_lower.mm`, and `compiler/mesh_compiler/native.py`; status documents are evidence,
not API authorities.

Private strict syntax checks at the phase checkpoint passed all eight METAL_IR_SRCS under the actual Makefile Tint
flags and ARC, plus native_bench, e2b_replay, metal_capture and metal_program. The intentionally malformed original
metal_recording.m still fails and was unchanged. Native Python/ABI source passes AST/py_compile and diff checks.
Stub consistency observations are not execution evidence. Canonical matching build and current client/archive
invocation remain to run after the paired source ABI is coherent.

Still active: preserve exact per-HostEntry readiness/release through CPU composition instead of the existing
blanket transferred-root exclusion; replace Transport's later numerical re-Construct with retained typed Clone;
feed actual cross-call ports and genuine static geometry/visibility into their proper construction/instance owners;
complete network joint solve and Published/loader consumption of the same saved owner. No union of call port sets,
extra all-input wait, fake empty numerical body, or per-call replan is authorized. The existing singular Plan JSON
mirror is not the completed joint archive connection.
