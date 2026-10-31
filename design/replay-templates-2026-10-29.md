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
- torch_mesh exposes these as `_stream.save(directory, functions, bound)` and
  `_stream.load(directory, bound, communicators, host)`. These are a program's functions, below.
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

## Declarations (R4, R5)

A host operation or crossing is made again from a declaration, not from its closure. The declaration is a kind, the
frontend's own bytes, and places: offsets in those bytes that hold a host address, an `ncclMeshBuffer`, or an
`ncclComm *`. At export, each place becomes (operand, offset in its lease) or (communicator index), and its bytes are
cleared. At instantiation, the places are written with the instance's operands and communicators, and the
frontend's resolver makes the operation or crossing from the bytes with the native constructor it always uses.
- **CPU programs.** These are declared before binding: `mesh_cpu_product_bind` rewrites a command, and its BNNS form
  holds its own pointers. Preparation and the rebuild bind through one function (engines `_cpu_program`).
- **Core ML predictions.** These are declared as their compiled model, units and views. The model is loaded from its
  path again.
- **Crossings.** A group of NCCL calls is declared as its issues. A crossing assumes its communicator's plan
  (`ncclMeshCommPlan`, made on every rank in one order, over the actual buffers), so in a new process the plan must be
  declared again before the crossing is encoded. Without that plan, a crossing fails as a value.
  - A plan's buffers can be kept operands that only the recorder holds, so the plan is captured where it is made.
    `ncclMeshCommPlan` keeps the declaration it accepted, flat (`plan_declaration`: each array in the length
    `declared()` reads, each binding's `ncclMeshBuffer` a place).
  - `MetalScheduleCommunicator` records a plan call wherever the communicator's declaration differs from the last one
    the function recorded. Instantiation declares it again (`ncclMeshCommPlanDeclared`) over the instance's operands
    before the crossings that follow. Every rank instantiates its own template in its preparation order, as every
    rank declares its plan in one order.
  - A plan declares a communicator's whole sequence. Functions that share one communicator's plan carry it whole, and
    the buffers they do not bind are kept by value. Declaring a plan again replaces the one in force, so a template
    with a plan is instantiated where that plan is not in use (a new process, or after the function it was made from
    is released).
  - A WRITE plan keeps the calls that made it (`write_declared`): its page table's storage, its prepare from masks or a
    layout with its arrays, and the readiness and landings set on it since. `ncclMeshWritePlanDeclared` makes the
    calls again.
    - Landing records are an output of preparation: the frontend reads them once into the fold kernel's words, which
      the template records as data. So a plan made from a declaration holds records of its own.
    - torch_mesh declares its fused crossings ("writes": polls, repeats, stride) and its view crossings ("views": the
      invocation) as their plans' declarations, and makes them again with fresh fused records.
- **Local checks** (M5 Max), since replaced by examples/template-e2e.py:
  - A CPU product rebuilt in a second process was exact on new data.
  - A 1-rank all-reduce crossing was loaded with the second process's communicator, and that process planned
    nothing. Its template is [plan, cross]. It took 2.2 ms and produced y = x on new data.

## Other architectures and other views (R6, R7)

**R6.** A template's libraries are AIR, which every Apple GPU loads, but they are compiled with the device's
threadgroup facts as macros (`MetalDeviceCompilerOptions`: `MESH_DEVICE_THREADGROUP_THREADS`, `_BYTES`).
- **Compile facts.** The template keeps those facts. An instance on a device with other facts fails as a value
  naming both. Every Apple GPU of this collective (M4 Pro, M5 Max, M5 Ultra) has 1024 threads and 32 KiB, so the AIR
  carries over.
- **Pipelines.** Pipeline binaries are per architecture. A load on an architecture without them compiles them from
  the AIR, with no source parse, and writes `pipelines.<architecture>.metallib` beside the others
  (`MetalTemplateArchive`).
- **Thread limits.** A pipeline's thread limit follows its register use, so it can differ between architectures. A
  dispatch whose recorded threadgroup exceeds this device's pipeline fails as a value naming it.
- **What is not re-derived.** Geometry that the frontend chose for a device (shares, splits by core count) is not
  re-derived. It is the frontend's choice, made at its preparation.

**R7.** A bound operand is a view: the template keeps where the view starts in its lease, and an instance's view
elsewhere moves every use of that lease by the difference. So one template recorded on one layer's slice of a
stacked weight runs on any other layer's slice of the same shape.
- **Restriction.** Bound views lie in distinct leases; two bound views of one lease are one operand.
- **Shape parameters.** Sizes that vary between members (uneven shares) are not parameters yet. Grids, offsets and
  data words are recorded as numbers. They become functions of sizes only where the frontend lowers sizes
  symbolically, through the relations of design/lowering-closure-2026-10-29.md, which torch_mesh's lowering does not
  yet emit.
- **Local checks** (M5 Max), since replaced by examples/template-e2e.py:
  - A two-stage function recorded on layer 0 of [4, 2¹⁶] stacks, loaded in new processes on layers 2 and 1, computed
    4x + 3y on those layers and left the others zero.
  - A load without this architecture's archive wrote one.
  - A template with altered compile facts failed, naming them.

## A program's functions, and the end-to-end test

A template holds a program: each of its functions' calls, over one shared table of operands, pipelines, libraries
and communicators. A plan belongs to the program, since `ncclMeshCommPlan` declares a communicator's whole sequence.
So a plan call is written where it differs from the last one written for its communicator, and every function's
crossings run under that one plan.
- **Why one function per template fails.** `bandwidth` plans its 45 cases in one call and realizes each as its own
  function. A template per function would declare the plan 45 times, and each declaration would free the storage of
  the functions loaded before it.

examples/template-e2e.py is the test. It runs on a real program, the bandwidth workload, through that program's own
interface.
- **Recording run.** A first process prepares the program as it always does (lowering, one plan, each function
  recorded), measures it and saves it.
- **Loading run.** A second process, with its own communicator, makes the program from the template on newly
  allocated operands, with no lowering and no planning, and measures it.
- **Evidence.** Each run's rows are the program's own: (us, GB/s) for each collective and size, labelled
  template-save or template-load, with its preparation seconds. Each run's outputs are checked against what each
  collective defines (max over ranks, the root's value, the gathered segments, the predecessor's exchange). A run
  whose outputs differ fails.
- **In the hourly.** The case `bandwidth-template` reports the saving run's tuples and then the loading run's, in one
  order.
- **A program that compiles kernels.** `fabric-dnn-template` runs the fabric-dnn workload the same way, on the entire
  collective, as every node run is. Its fold kernel compiles from source, and its products' pipelines, at every
  preparation, so the saving run compiles and the loading run does not. The loading run's preparation is the
  difference.
  - Its input and weights come from a fixed seed, so the loading run's output must equal the recording run's bit for
    bit. Each rank saved its output beside its template.
  - The loading run binds the recording run's assignment (its parts and operands, written beside the template), so
    both bind the same shapes.
  - Its partial, landing and hidden rows are bound, not kept, so the template stays small.
  - Its crossings are WRITE plans, so it is also the first end-to-end run of WRITE-plan capture.
- **One rank on the M5 Max (2026-10-29).**
  - 45 functions in each run, every output as defined.
  - Preparation 0.116–0.119 s recorded and 0.107–0.120 s loaded. This program compiles no kernels, and its
    preparation is the plan's compile, which a template declares again by design. So bandwidth gains no preparation
    time from a template.
  - The loaded run's rates were at or below the recorded run's in every shape, one sample each, so not evidence of
    any difference.

## Steps

| Step | Scope | Interfaces touched |
|---|---|---|
| R1 | Export: a schedule keeps its calls; pipelines are declared where made (landed) | `MetalPipelineMake`, `MetalPipelineDeclare`, `MetalScheduleTemplate` |
| R2 | The template directory: plist, libraries, per-architecture pipeline archive (landed) | `MetalTemplateWrite`, `MetalTemplateRead` |
| R3 | Instantiation of GPU-only functions: data, compute, append (landed) | `MetalScheduleInstantiate`; torch_mesh `Schedule.save`, `Schedule.load` |
| R4 | Host operations declared: (kind, bytes, places); the engine rebuilds them (landed: CPU programs, Core ML predictions) | `MetalDeclaration`, `MetalScheduleHost`; engines `rebuild` |
| R5 | Crossings refer to the complete communicator plan; typed views and progress bindings are serialized once. Instantiation restores that plan, and each crossing records its next declared group. | `MetalScheduleCross`; `ncclMeshCommPlanDeclaration`/`Declared`; `ncclMeshCommEncode` |
| R6 | Another architecture: per-architecture archives kept on first load; compile facts and pipeline thread limits checked as values (landed) | `MetalTemplateArchive`; `MetalScheduleInstantiate` |
| R7 | Bound operands are views, their byte offset the parameter (landed). Shape parameters need symbolic sizes from lowering | `MetalScheduleTemplate`/`Instantiate` offsets |

R1–R3 are an encapsulated extension in the operator's sense of 2026-10-28: a new file and one accessor, no change to
an internal interface. A replay holding host operations or crossings fails to save, as a value, until R4–R5. R4 and
R5 change interfaces in metal-recording.*, which are the other group's hot files.
