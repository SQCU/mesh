# Retained source construction, 14 November 2026

The actual Target construction caller used `metal_ir_specialize` on an unconstructed source, which that API rejects.
It now calls `metal_ir_clone` before constructing any specialization class. The new entry consumes the same
`AppendSource` owner as `metal_ir_append`: typed region functions and handles are cloned through the existing Clone,
values/operations/regions preserve their IDs for this whole-source clone, and the source owner retains constants
and source relation facts. The existing append mapping still handles composition and STEP identity. Source-axis
remapping is frontend41's reviewed adjacent handoff, retained inside that same owner. No Transform or archive
roundtrip occurs in cloning.

Planned storage visibility now consumes the actual `Value.function != UINT32_MAX` identity in ExternallyVisible,
Realize and saved liveness. `planned` is an offset and may be zero; `.placed` still means an actual physical binding.
The native caller's interface IDs therefore remain valid across construction classes without invented buffers.

SAME native e48 supplied the C change; SAME archive047 supplied the central ABI and actual Target caller.
Codex inspected the shared owner, source/constant lifetime, unchanged append mapping, identity clone, and paired
caller. Strict private syntax of current `metal_ir.mm` and `metal_archive.mm` passed after frontend41 connected its
Flattened consumer; the other live native units had passed the preceding source check. Python AST/py_compile,
pyflakes and symbol declaration checks passed. No shared library build or runtime observation is claimed for this
new symbol. Original malformed files remain untouched. Work/formula/archive and Sort hunks are excluded from this
checkpoint, except the explicitly shared source-axis remap.

Remaining connected work is the prefix landing/body/suffix staging composition using actual append maps and
Prepared.remap/bind, and publication after all real writer scopes. The latter must retain one Arrivals completion
owner and actual per-function execution-copy lifetime; host publication after each writer is still open.
