# Live numerical construction owner and its actual consumer

Continue SAME Claude session 59368744-da56-491a-871a-1be809c0cc93, explicit claude-opus-5-5.
This replaces your earlier dead-source relocation orders. They are revoked.

The user's latest literal instructions are:
> i didn't say that the obsolete code should be moved, i said it should be mangled s.t. it is impossible to compile or interpret it without obvious errors
> this is the fastest way to cause all callers to be updated immediately and the instruction was very pragmetic and literal

The malformed definitions are restored and committed at their original paths. Do not edit, repair, move,
comment out, quote, parse around or import rdma/mesh_compile.py, compiler/mesh_compiler/lower_coreml.py,
examples/tp.py or metal-microbench/metal_recording.m. Their parse errors deliberately expose stale consumers.
Do not clone those modules under new names or create compatibility facades.

Implement this narrow live responsibility now:

1. compiler/mesh_compiler/operations.py owns the pure numerical/structured operation construction actually
   consumed by lower_metal and the placed-operation decomposition stage. It has no ctypes/ObjC/runtime,
   capture/export pipeline, native-program assembly, old program/load closures, source-text semantic parser,
   or duplicate graph solver. Inspect existing cited numerical formulas to retain required feature coverage,
   but the module's responsibility is source operands -> structured operation/region records.
2. Migrate compiler/mesh_compiler/lower_metal.py directly to these operation operands and the actual native
   encoder owner compiler/mesh_compiler/metal.py. No old mesh_compile import/sys.path shim or facade.

Read executor/live-metal-owner.status.md now. Settled dependency direction: metal imports operations; operations
does not import metal. Executor SAME047 owns metal.py and its _Builder. Do not edit that file. Frontend owns
driver/export/ir/runtime. Compiler SAME64 owns decompose.py; spectral edits in operations.py wait for your release.

Required exports to encoder047: _key, _stored, _Words, _Check, _Lapack, _Sort, _rows, _bits, SCALAR, KINDS, CPU.
Structured _Generic/_Region records carry the fields listed in executor/live-metal-owner.status.md §1.
operations owns scalar codes/tables and dtype groups, numerical op codes, GPU/CPU/ANE, BINARY/COMPARE/BOOLEAN/
COMPLEMENT/NEGATION/NOT/BUILTINS, PARALLEL/REDUCTION, ASSIGN/SUM/REGION, IDENTITY/ACCUMULATE, REASSOCIATE/CONTRACT.
It also owns actual lower_metal/driver consumers' Generator, seeded, _stepped, _instruction, _local_call, _copy,
_held, _placed, _metadata, _same_view. Preserve source-defined decomposition formulas used by decompose,
including _matrix_exponential, _polar and related existing cited numerical formulations; publish exact exports.
metal owns ABI role enums INPUT/CONSTANT/TEMPORARY/OUTPUT/STATE, BODY/COMBINE/FINALIZE/NONE, ctypes structures,
libraries, borrowed resources and _Builder. lower_metal imports each responsibility directly, not one facade.

Publish dependency/export boundary in integration-operations-owner.status.md immediately, then implement.
Do not spend a turn only drafting a proposal: this settled source boundary is authorized. Keep cited algorithms'
coverage, and distinguish surviving algorithm code from obsolete orchestration. Follow actual literature words.
Use the schedule-emit reference for stage composition, not exact module/signature copying. Existing documentation
claims are leads; source and actual stage inputs/outputs determine the implementation.

Check the live new source syntactically and its direct imported symbols once metal.py exists; the intentionally
malformed originals must continue failing. No native library build or new evaluator. Root coordinates existing
client observations after source coherence. No fleet/service operations. Root owns staging and commits.
When finished, record source/check evidence and release operations.py's exact decomposition/formula sections to
compiler64. Do not wait for permission to implement this assigned owner and its lower_metal consumer.
