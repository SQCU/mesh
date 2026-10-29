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

- **Pipelines** are named by their declaration: (library, function name, function-constant values as [index,
  MTLDataType, bytes], descriptor fields). `MetalPipelineMake` makes a pipeline from one and declares it. A frontend
  that makes pipelines itself calls `MetalPipelineDeclare`. Metal does not expose an `MTLFunction`'s library or
  constant values, so the name is taken where the pipeline is made. The library is written as itself
  (`serializeToURL:error:`, which microbench's common.swift `keptLibrary` already relies on), so loading parses no
  source. The pipelines are written as an `MTLBinaryArchive` for the writing device's architecture.
- **Buffers** are named by operand, and an operand is a region lease (`ncclMeshMemBuffer`). Torch's buffer for a tensor
  and the lease buffer of the same bytes (`MetalTensorPages`) are therefore one operand, and offsets are the lease's.
  - The caller names the *bound* operands (inputs, outputs, weights to replace). Each bound operand records where its
    buffer starts in its lease, and instantiation moves its offsets by the difference.
  - Every other operand is *kept*: its bytes are stored, and instantiation leases them anew (`MetalLeaseBuffer`).
  - Nothing stores an address, since a lease's `gpuAddress` differs per process.
- **Barriers** are compiled again at instantiation (`MetalDataflow`, `MetalOrder`). Since the frontier was bucketed
  (mesh a8856dd4), that pass is not where preparation's time goes. Storing its result would be a second record of
  what the reaches already determine.
- **ICB segments** are not stored. Metal owns their storage. Rehydration rebuilds them through the same
  `MetalReplayBuild` a recording uses, which costs encoding only.
- **Host operations** become named prepared operations: a kind, operand views and parameters (an SGEMM/BNNS binding,
  a CoreML model and its views). The header already says they are "prepared native operations, not callbacks into a
  frontend" (metal-recording.h:150). Naming them is the one interface change.
- **Crossings** are named by the collective call they belong to (declaration index, invocation, mark). The collective's
  program is compiled identically at every member (C7), so it is rebuilt from the declarations at load and never
  stored. Its pipelines are libnccl-mesh.metallib's.

The template is the function's building calls, not its encoded command buffer. `MetalScheduleCross` encodes a
collective into the Metal 4 command buffer during preparation, so the finished buffer holds what cannot be written.
A schedule keeps its calls as it is built (data, compute, append, host, cross), and instantiation makes the same calls
on a new schedule. Pipeline compiles leave preparation; design/compile-once-2026-10-29.md names them a candidate cause
of ranks starting seconds behind their peers.

## Templates: bindings by operand, not by buffer

A template is the serialized form above: its calls, operand-relative bindings and reaches, and pipeline declarations.
An instance is a template plus an operand table (operand k → lease) plus a device.

As landed (R1–R3):
- `MetalScheduleTemplate(schedule, bound)` exports the calls.
- `MetalTemplateWrite` and `MetalTemplateRead` write and read the directory: template.plist, library.<i>.metallib and
  pipelines.<architecture>.metallib.
- `MetalScheduleInstantiate(schedule, template, bound)` makes the calls on a new schedule.
- torch_mesh exposes these as `_stream.Schedule.save(directory, bound)` and `_stream.Schedule.load(directory, bound)`.
- A local check on the M5 Max ran one compute call and one replay (out = 4x + 3y over 2¹⁶ halves). Preparation took
  68.6 ms recorded, with its compile, and 1.4 ms loaded in a second process with no compile. Both runs were correct
  on their own data.

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
| R1 | Export: a schedule keeps its calls; pipelines are declared where made (landed) | `MetalPipelineMake`, `MetalPipelineDeclare`, `MetalScheduleTemplate` |
| R2 | The template directory: plist, libraries, per-architecture pipeline archive (landed) | `MetalTemplateWrite`, `MetalTemplateRead` |
| R3 | Instantiation of GPU-only functions: data, compute, append (landed) | `MetalScheduleInstantiate`; torch_mesh `Schedule.save`, `Schedule.load` |
| R4 | Host operations named by kind; `MetalRecordHost` takes the name and its views | `MetalRecordHost`, `MetalScheduleHost` |
| R5 | Crossings named by their collective call; the program recompiled at load | `MetalCrossing` |
| R6 | Geometry as a function of device capabilities; per-family archives | the recording of grid and group |
| R7 | Parametric sizes from the lowering relations | depends on design/lowering-closure-2026-10-29.md T1–T3 |

R1–R3 are an encapsulated extension in the operator's sense of 2026-10-28: a new file and one accessor, no change to
an internal interface. A replay holding host operations or crossings fails to save, as a value, until R4–R5. R4 and
R5 change interfaces in metal-recording.*, which are the other group's hot files.
