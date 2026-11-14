# Native function and external backing handoffs

Executor source review, 2026-11-13; matching consumer observations, 2026-11-14. Same Claude sessions: native
`e48f4bc7` (resumes31–33), archive `047e12be` (resumes26–28), each explicitly reported `claude-opus-5-5`. Codex
reviewed the connected source and ran the bounded observations below.

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

## Matching build and actual callers

The canonical build passed at mesh `246228d2` and metal `5889e933`; the Python placement caller checkpoint is
`d2c63c08`. The frozen build inventory is `compiler/function-relation-build-source.json`, with build and actual ABI
logs `compiler/function-relation-{libmesh-build,native-build,native-abi}.log`. Actual sizes were Function168,
Planned104, Local144, Search56, RelationData160, Actors48 and Storage24. Native archive version was 33.

The existing `examples/engines.py` ran with `OP=ffn ROWS=2 K=32 N=48 LAYERS=1 --nodes 0`, first
`ENGINES=cpu --device cpu`, then `ENGINES=mps --device mps`, using its normal CPU-created input and state. Both
exited 0, reported output `[2,32]` and normalized maximum error **0.0**. CPU preparation had one native entry;
GPU preparation had one retained command, 43 values and 11,984 bytes. Neither introduced host/fence/transport
waits. GPU execution preserved CPU output storage; no manual tensor moves were used.

For each target, the same existing Engines model was compiled and saved through the production outer archive,
the original Scheduled was deleted and collected, and `archive.load → Program` was invoked on two distinct CPU
input allocations (`x` and `x + 0.125`). Both targets reported errors **0.0** and **6.902263703523204e-05** against
the original model. Engines reports this metric without a fixed tolerance. Both archives preserved signature bytes,
loaded an actual Function relation, returned distinct fresh result storages, and left the first result exactly
unchanged after the second call. The GPU check's two inputs and two outputs all remained CPU tensors. Archive
sizes were 194,786 bytes (CPU) and 160,402 bytes (GPU). Native load reported restored positions without a relation
solve. All four processes exited 0 with empty stderr.

Logs are `typed-{cpu,mps}-{ffn,archive}-function.{json,stderr}`, and the two actual archive directories are
`typed-{cpu,mps}-archive-function/`. No evaluator was added. These observations cover the compiled fixed binding
and ordinary completion/lifetime paths; they do not measure cancellation, general alias/mutation, RNG, mixed
targets, ANE, network transport or performance. The blit endpoint connection is source-reviewed, not isolated by
this one-command FFN. No further GPU repeat is justified without a relevant implementation change.

| Library | SHA-256 |
| --- | --- |
| `.build/libmetal_ir.dylib` | `3187c2ef796607cec89de1e7ad12f0b72aa92131fdf6287cfc1ef21126f3b9fa` |
| `.build/libmetal_recording.dylib` | `0a0cf3c852fc1db8cb1176b00788164ee1a3cb32a13efc256613ea31976eae4e` |
| `rdma/libmesh.dylib` | `28ac159aa369011dd5fe941422a31938f660cd12bb7af2cf62aac679dca283ad` |
