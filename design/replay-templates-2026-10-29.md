# Replay templates: serialization, rehydration and reuse, 2026-10-29

The question (the operator, 2026-10-29): can recorded replays be serialized and rehydrated? Can operations of the same
shape be made parametric, so that replay segments are reused with the same architecture and different data, or a
different architecture in the same collective? Line numbers are rdma/metal-recording.m at 2425bbf7.

Conditional and while nodes (data-dependent early exit) stay deferred behind this. The operator's reason: no
architecture with state-of-the-art results uses early exit.

## What a replay holds today

`MetalReplayBuild` (1122) keeps the whole description beside the executable form, on the replay's owner (1143):

- the records, `MetalRecordedDispatch[]` (467): pipeline, buffer per binding, offset, stride and instances (the
  per-invocation slot), length, threadgroup memory, grid, group, and the access mask;
- the barrier bytes, a function of the reaches and the invocation index (`MetalOrder`, 1134);
- the reaches, `MetalReach[]`: buffer, offset, length, access and binding;
- the form and the repetition count.

The executable form is a list of indirect command buffer (ICB) segments with their buffers baked in
(`inheritBuffers = NO`, 1155; `setKernelBuffer` per command per invocation, 1188). Metal allocates those segments
outside the region (1165). The Metal 4 direct form instead binds every buffer at encode time, as
`table setAddress: buffer.gpuAddress + MetalInvocationOffset(...)` (1299).

Every record field is plain data except three kinds of object:

1. **the pipeline** (`id<MTLComputePipelineState>`);
2. **each buffer** (`id<MTLBuffer>`, in records and reaches);
3. **two kinds of closure:** host operations (`MetalRecordHost`, `NSError *(^)(void)`) and collective crossings
   (`MetalCrossing`).

## Serialization: name the three kinds of object

- **Pipelines** are named by (library source or metallib, function name, function-constant values). The compiled form
  is an `MTLBinaryArchive`, which mesh already serializes for its own kernels (nccl-mesh-kernels.m:31–39) and
  microbench already loads (common.swift:160–250). design/compile-once-2026-10-29.md makes layouts function constants,
  so a pipeline is exactly (source, constants).
- **Buffers** are named by operand: (operand k, byte offset, stride, instances, length). Rehydration binds operand k to
  whatever lease the loader supplies. A lease's `gpuAddress` differs per process, so nothing stores an address.
- **Barriers** are stored, so that rehydration skips the dependency compile. That compile cost cp 260 s per rank before
  the frontier was bucketed (docs/facts.md, mb 170dba48), and it still runs at every preparation.
- **ICB segments** are not stored. Metal owns their storage. Rehydration re-encodes them from the records, which costs
  encoding only, with no compile and no dependency pass.
- **Host operations** become named prepared operations: a kind, operand views and parameters (an SGEMM/BNNS binding,
  a CoreML model and its views). The header already says they are "prepared native operations, not callbacks into a
  frontend" (metal-recording.h:150). Naming them is the one interface change.
- **Crossings** are named by the collective call they belong to (declaration index, invocation, mark). The collective's
  program is compiled identically at every member (C7), so it is rebuilt from the declarations at load and never
  stored. Its pipelines are libnccl-mesh.metallib's.

Two costs leave preparation: pipeline compiles, which design/compile-once-2026-10-29.md names a candidate cause of
ranks starting seconds behind their peers, and the dependency compile.

## Templates: bindings by operand, not by buffer

A template is the serialized form above: operations, operand-relative bindings and reaches, barriers, pipeline names
and geometry. An instance is a template plus an operand table (operand k → lease) plus a device.

### Same architecture, different data

- **The direct form needs no change.** It rebinds by writing other base addresses into the argument table (1299).
  Today's `MetalOperand` slot (invocation t binds slot t % instances at a stride) is the special case where the
  operand table is affine in t.
- **The ICB form has two options.**
  - Re-encode the segments per operand table (encoding only).
  - Bind through one argument buffer per operand table. The commands then read operand addresses from it, and new data
    means writing k addresses with the ICB untouched. The cost is one extra indirection per binding inside kernels.
    `inheritBuffers = YES` is not general, because every command in a segment would have to share each slot's binding.
- **Validity.** Barriers stay valid while the operand table aliases the same way the template's did. Region leases
  declare their aliasing, so the check is a value at bind time, not a run-time gate.
- **Sizes.** Equal shapes reuse everything. Shapes that differ only in their parameters (the symbolic sizes of
  design/lowering-closure-2026-10-29.md) keep the template's structure. Grids and offsets become quasi-affine functions
  of the sizes. Sizes passed as function constants select other pipelines, all hits in the binary archive; sizes passed
  as constants in a buffer keep one pipeline for every size.

### Different architecture, same collective

- **Pipelines.** A metallib's AIR is portable across devices, and binary archives are per GPU family. A template names
  (source, constants); each member looks the pipeline up in its own family's archive, or compiles it once into that
  archive.
- **Geometry.** Threadgroup size, SIMD width and threadgroup memory become functions of the device's capabilities
  (metal-capabilities.h) instead of recorded literals. chain.py already takes its SIMD width from configuration (mesh
  a8b6e797).
- **What carries over unchanged.** Barriers come from data reaches, so they are independent of architecture.
- **Per device.** The form is chosen per device. A pipeline without `supportIndirectCommandBuffers` runs in the encoded
  form (1179). Host operation kinds (ANE, CPU) resolve to each device's implementation.
- **Members of one collective** run one template with their own operand tables and shares. Uneven shares (C2) need
  the parametric sizes above.

## Steps

| Step | Scope | Interfaces touched |
|---|---|---|
| R1 | An accessor that exports a replay's records, reaches and barriers, keyed by operand | one new function beside `MetalReplayBuild` |
| R2 | A template file: pipeline names, operand-relative records, reaches, barriers, geometry; writer and reader in a new file | none |
| R3 | Rehydration of GPU-only replays: archive lookup, operand table, ICB re-encode or direct form | none |
| R4 | Host operations named by kind; `MetalRecordHost` takes the name and its views | `MetalRecordHost`, `MetalScheduleHost` |
| R5 | Crossings named by their collective call; the program recompiled at load | `MetalCrossing` |
| R6 | Geometry as a function of device capabilities; per-family archives | the recording of grid and group |
| R7 | Parametric sizes from the lowering relations | depends on design/lowering-closure-2026-10-29.md T1–T3 |

R1–R3 are an encapsulated extension in the operator's sense of 2026-10-28: a new file and one accessor, no change to
an internal interface. A replay holding host operations or crossings fails to save, as a value, until R4–R5. R4 and
R5 change interfaces in metal-recording.*, which are the other group's hot files.
