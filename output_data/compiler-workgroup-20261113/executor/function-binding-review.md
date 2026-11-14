# Native function and external backing handoffs

Executor source review, 2026-11-13. Same Claude sessions: native `e48f4bc7` (resumes31–33), archive `047e12be`
(resumes26–28), each explicitly reported `claude-opus-5-5`. Codex reviewed the connected source. Runtime evidence
below remains pending the matching Function/load build.

The fixed GPU consumer is metal `0a506f6c`. `MetalNativeICB` reads the replay's retained argument records/storage
and emits existing `NativeExternalArgument` sites before the unchanged ICB spans. Each site retains the original
command-to-dispatch and member correspondence. Copy/fill sites use `NativeExternalBuffer` against their existing
endpoint binding arrays. No commands, schedule, copy counts or waits are introduced. This consumes the table ABI
already produced by Tint, retained by Kernel/Emitted and saved by the native archive.

`native.Target.prepare` borrows `MetalProgramExternals` with the native owner. Its completed use records identify
which external operands require GPU access. `Bound` preserves the caller's actual storage and root byte offset.
MPS storage supplies its Metal buffer; GPU-used CPU storage is wrapped over its system-page span by the existing
`mesh_compile._placed_buffer` owner, shared among views of that storage within the invocation. CPU-only operands
are not wrapped. Fresh results retain their existing logical device and ownership; no staging or device copy is
inserted.

The wrapper now delegates to `MetalBufferBorrow`. This is the same Metal `newBufferWithBytesNoCopy` operation,
with an optional backing owner released by Metal's buffer deallocator. The primitive explicitly adopts only after
a non-nil result. The caller lends its actual UntypedStorage; therefore error unwind cannot drop that backing while
native completion still holds the buffer. Mapped-plan callers pass a typed null release callback and keep their
existing lifetime. `_lent` shares the existing CPython reference-count transition across buffer and archive loaders;
there is no global pointer registry or retained invocation history. The unused Objective-C WRAP trampoline is gone.

`Solution.solve`, `Transported` and legacy `_bound` now pass the actual retained Actor quotient in
`mesh_function.relation`, alongside its cost and edge views. The canonical local/network loaders own the original
archive bytes and restored compiled relation/arrays. Local and bridge callers lend their existing Python bytes.
The legacy loader lends its aligned native-archive record span while retaining the actual parent archive. It no
longer copies that plan record. The six-field Python Actor cost/edge save recipe and load reconstruction are deleted.
The native plan archive supplies those compiled operands before the existing load validation.

Supervisor checks: exact diffs and Python AST/diff checks passed. The recorder primitive passed current-source
strict compilation; its first check against the prior header was superseded after the new header became coherent.
The new Function168, Planned104 and Local144/load callback ABI must be built together with these callers and the
compiler-owned C tool changes. Existing CPU/outer archive checks will be rerun because the native archive owner
changed, followed by the existing local Engines GPU client for the newly connected binding path. No live transport,
general alias/mutation, ANE or performance result is claimed by this source checkpoint.
