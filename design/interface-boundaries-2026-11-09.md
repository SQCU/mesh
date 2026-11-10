# Memorandum: what the compiler and provider modules written on 2026-11-09 take and return at their boundaries

2026-11-09. Written by Claude at the operator's request, from the session
`.claude-personal/projects/-Users-mdot-metal-microbench/55974018-f56d-4c8b-8973-6fc61b49140c.jsonl`. It accompanies
[implementation-order-2026-11-09.md](implementation-order-2026-11-09.md) and, like it, is an aid and not a source:
the change requests and the operator's turns govern.

## The turns

The operator, 2026-11-09, in order:

> actually neither of these are required sincew e can look at the overall functional flow of the old implementation trapped in mesh and metal microbench. if the data flow has the wrong directions at any point, we don't need to read any part of it to totally replace it, since it has no contents which relate to the requirements for any function which has the wrong causal direction. isn't that cool?

> "That reduces the review to boundaries. " then have a haiku 5.5 subagent-parallel workflow identify the interface-boundary-typing of all of the code implied in those inventories you were considering, with their output written to a memorandum in the repository scope.

## What was done

Ten readers (Claude Haiku), one per group of files, each read-only, ran in parallel. Each was told to work from
headers, includes, externally visible declarations and call sites, without reading function bodies in full, and to
state: what the group takes and from where; what it returns and to whom; which role it has in the order of stages
below; and whether its direction matches that order, with a file and line for every place it does not. Each was told
to answer "unknown" rather than guess, and to make no judgment of quality and no proposal.

The order of stages they compared against, each a function from one value to one value, a later one consuming what
an earlier one returns and never the reverse:

1. export: a torch module or a model file -> a graph of functions with a call signature
2. placement: that graph and a placement operand -> the graph with each function's processor, and send and receive
   nodes where a value crosses processors
3. devices: the placed graph and a device per processor -> the graph with devices filled in
4. schedule: that graph and a cost operand -> one ordered program per processor, and storage decisions, as one value
5. lower: the scheduled program -> one callable function per processor in one compiler's input format
6. runtime: the scheduled program and the lowered functions -> a callable; it binds addresses and calls; it derives
   nothing
7. archive: the scheduled program <-> files; loading solves nothing

Supplying numeric functions, or changing an external compiler, is a separate role and not a stage of this order.

A group's direction does not match where it (a) takes something a later stage produces, (b) produces something it
should receive from an earlier stage or a supplier, (c) reads back emitted output (source text, compiled binaries,
debug information) to recover a fact, or (d) performs several stage roles at once.

## What this is and is not

- The files were read in the metal-microbench working tree at 7483572e, which had three uncommitted files
  (`metal_lower.mm`, `metal_program.h`, `metal_recording.m`). Line numbers are of that working tree.
- The statements below are the readers' own, lightly reformatted. Claude checked 17 of the cited lines against the
  files (in `metal_ir.mm`, `metal_lower.mm`, `metal_native.mm`, `metal_neural.mm`, `metal_transform.mm`,
  `metal_recording.m`, `metal_submission.h`, `provider/litert_mesh.cc` and `provider/dawn.patch`); each showed what
  the reader said it showed. The rest are unchecked.
- A cited line is a fact; which of (a)–(d) it is an instance of is the reader's reading. Some readings are open to
  dispute, for example whether taking a device handle when lowering is taking a later stage's product.
- `metal_recording.{h,m}`, `rdma/mesh_compile.py`, `tools/e2b_replay.m` and the mesh C modules were not in this run.
  The implementation-order memorandum already records what was established about the first three.

## Summary

| group | lines | visible functions | roles found | direction |
|---|---:|---:|---|---|
| metal_ir | 4,669 | 92 | graph construction, send and receive declarations, schedule, lower entry, emission, archive, runtime binding and cost, supplying numeric functions | does not match |
| metal_transform | 1,452 | 1 | fusion and grouping into kernels, rewriting of regions, cost evaluation, host call entries | does not match |
| metal_lower | 2,859 | 49 | lower, mapping of each operation, running a compiled function while lowering, emission, linking, archive, binding on a device | does not match |
| metal_native | 1,233 | 2 | reading the machine's facts, schedule choices for matrix multiply, cost, lower | does not match |
| metal_clone and metal_transport | 925 | 8 | copying IR between modules (used by lower, transform and the numeric library); send and receive structure at lower and emission | mixed |
| metal_neural | 1,503 | 6 | lower, a storage decision, compile with a read-back of the device plan, archive, runtime bind and call | does not match |
| metal_program.h and metal_submission.h | 365 | 24 | runtime, declarations of lower and archive, reading identity and call sites back from the compiled binary | does not match |
| provider: litert_mesh and litert.patch | 5,477 | 2 | conversion of a model to a graph, lower, binding values to tensor addresses, storage decisions, supplying numeric functions | does not match |
| provider/dawn.patch | 20,081 | 49 | lower (printing), a runtime calling sequence inside the printer, supplying 64-bit numeric functions, shared types and validation | does not match |
| metal_binary64.wgsl and operator_shaders.metal | 1,971 | 139 | supplies numeric functions; no stage role | matches |

Nine of the ten groups have a direction that does not match, or matches only in part. The tenth, the two authored
shader sources, has no stage role: it supplies numeric functions.

## metal_ir

- **Files:** `metal_ir.h` (868 lines), `metal_ir.mm` (3,801 lines).
- **Externally visible functions:** 92.
- **Called from:**
  - /Users/mdot/dox/mesh/rdma/mesh_compile.py (about 51 metal_ir_* call sites: construction, metal_ir_lower,
    metal_ir_finalize, metal_ir_save/load, readbacks)
  - metal_recording.m (about 52 call sites: metal_ir_pipeline, metal_ir_source, metal_ir_dispatch_costs,
    metal_ir_prepare_order, readbacks)
  - provider/litert_mesh.cc (about 35 call sites: construction and value calls)
  - tools/e2b_replay.m (about 143 metal_ir_* name matches; not classified)
  - tools/native_bench.m (about 64 name matches; not classified)
  - metal_transform.mm, metal_lower.mm, metal_native.mm, metal_transport.mm, metal_lower.h, metal_recording.h,
    metal_program.h, docs/provider.md (name references; not classified)
- **Takes:**
  - C ABI value/operation operands: role, scalar, rank, shape pointers, loops/bounds, operand value and operation
    indices, byte payloads (void*, size_t) — from external producers through the C ABI (metal_ir.h:75-390);
    callers include /Users/mdot/dox/mesh/rdma/mesh_compile.py, metal_recording.m, provider/litert_mesh.cc,
    tools/e2b_replay.m, tools/native_bench.m (metal_ir.h:82, metal_ir.h:90, metal_ir.h:176, metal_ir.mm:2721,
    metal_ir.mm:2951)
  - struct mesh_graph_machine gpu and cpu machine configurations — from mesh-graph.h via caller (cost
    configuration) (metal_ir.h:261, metal_ir.mm:3792-3796)
  - void *device (id<MTLDevice>), retained and read for threadgroup memory length — from runtime caller of
    metal_ir_lower and metal_ir_load (metal_ir.h:262, metal_ir.mm:3285, metal_ir.mm:3293-3296, metal_ir.mm:3406)
  - serialized artifact bytes (const void*, length) to load — from file or buffer supplied by caller
    (metal_ir_save output read back) (metal_ir.h:277, metal_ir.mm:3406-3411)
  - dispatch-order parts (const uint32_t*, count) with earliest/latest bounds — from caller (prepare-order path;
    metal_recording.m is a caller of metal_ir_prepare_order per grep, not verified line by line) (metal_ir.h:286,
    metal_ir.mm:3530)
  - neural kernel input pointers (const void *const *, count) — from runtime caller (metal_ir.h:343,
    metal_ir.mm:3693-3695)
  - tint::core::ir::Module regions and Builder types (Program field) — from Tint library (metal_ir.h:403-404,
    metal_ir.h:667 (struct Program regions member), metal_ir.mm:1063-1071 (ir::Builder use))
  - mesh_graph_levels, mesh_graph_heap_order and mesh_graph_cost_times inputs from mesh-graph library — from mesh-
    graph (external helper) called from this file (metal_ir.mm:1903, metal_ir.mm:1929, metal_ir.mm:1975,
    metal_ir.mm:3713-3714)
  - Metal library WGSL text for binary64 soft-float functions — from embedded library source parsed by Tint WGSL
    reader (metal_ir.mm:2256-2262 (Library), metal_ir.mm:2280-2284 (Binary64))
- **Returns:**
  - opaque struct metal_ir_program* handle and status ints (0 or errno with error text) — to callers listed under
    callers (mesh_compile.py, metal_recording.m, provider/litert_mesh.cc, tools/e2b_replay.m) (metal_ir.h:75-76,
    metal_ir.mm:2702, metal_ir.mm:3285-3303)
  - per-kernel pipeline (void*), task function pointer, neural function pointer with context — to runtime callers
    (metal_recording.m references metal_ir_pipeline, metal_ir_task, metal_ir_neural per grep) (metal_ir.h:339-343,
    metal_ir.mm:3682-3691)
  - dispatch cost totals (double*), issued costs and status ints — to metal_recording.m (grep match for
    metal_ir_dispatch_costs) (metal_ir.h:361-362, metal_ir.mm:3708-3721)
  - emitted kernel source text (const char*, size) for diagnostics and digest — to metal_recording.m:653 (digest
    of metal_ir_source output) (metal_ir.h:363-364, metal_ir.mm:3722-3724, metal_recording.m:653)
  - saved artifact bytes (malloc'd void*, size_t length) — to caller that frees it (metal_ir.h:270 comment);
    consumer in mesh_compile.py via metal_ir_save/load pairing per grep (metal_ir.h:271, metal_ir.mm:3359-3404)
  - loaded program pointer (struct metal_ir_program*) or NULL with reason — to mesh_compile.py (metal_ir_load grep
    match) (metal_ir.h:277, metal_ir.mm:3406-3468)
  - sorted synchronization counts per kernel (uint32_t counts[4]) and device counts (counts[3]) — to runtime
    callers (metal_ir_kernel_counts, metal_ir_kernel_devices) (metal_ir.h:346, metal_ir.h:352,
    metal_ir.mm:3698-3706)
  - value and operation info structs, dispatch/binding/effect/order/dependence/control arrays as pointers into the
    program — to metal_recording.m and mesh_compile.py readbacks (metal_ir.h:370-390, metal_ir.mm:3727-3800)
- **Role:** more than one: export-like construction (graph built through metal_ir_value, metal_ir_binary, metal_ir_generic
  etc., metal_ir.mm:2721-3263); placement/transport declarations (metal_ir_machine, metal_ir_transport,
  metal_ir_value_sends/receives, metal_ir.mm:2779-2831, 3792); schedule (Order, ReplayOrder, Artifact,
  metal_ir.mm:1900-1960, 1982-2060); lower entry (metal_ir_lower calls Transform and Lower declared at
  metal_ir.h:851-852, metal_ir.mm:3297-3303); emission and finalization (metal_ir_finalize calls Emit,
  metal_ir.mm:3318); archive (metal_ir_save/load, metal_ir.mm:3359-3468); runtime binding and cost (pipeline,
  task, neural_bind, dispatch_costs, metal_ir.mm:3682-3721); numeric-function supplier (Binary64,
  metal_ir.mm:2284-2290). Several of these at once in one file.
- **Direction:** does not match.
- **Where, as the reader states it:**
  - (a) takes a later-stage product: the runtime device handle is taken at the lower stage and used to set memory
    and retain, metal_ir.mm:3293-3296 (p->device = CFRetain(device); p->memory = maxThreadgroupMemoryLength),
    declared metal_ir.h:262.
  - (a) load takes the device and rebinds kernels (runtime-stage input consumed by the archive stage),
    metal_ir.mm:3406 and metal_ir.mm:3444 (LoadTarget(*p, k, device, ...)).
  - (b) produces what an earlier stage should supply: Order computes the schedule (one ordered program per
    dispatch set) from the dependence relation inside this file, metal_ir.mm:1900-1932 (mesh_graph_levels at 1903,
    mesh_graph_heap_order at 1929), invoked from Artifact at metal_ir.mm:2055.
  - (b) Artifact computes dependence (Relation), control facts, footprints, actors/costs and later passes itself,
    metal_ir.mm:1982-2060 (Relation 2052, ControlFacts 2053, Order 2055, Realize 2056, Actor 2059, Laters 2060).
  - (b) supplies numeric functions itself: Binary64 parses a WGSL library and splices soft-float functions into
    the kernel module, metal_ir.mm:2256-2290.
  - (c) reads back emitted output to recover a fact: Counts walks the emitted module after Emit to derive
    synchronization counts, metal_ir.mm:1443-1460, called after Emit at metal_ir.mm:3318 and in the loop at
    metal_ir.mm:3322-3327.
  - (c) exposes emitted source text as an output, metal_ir.mm:3722-3724 (metal_ir_source), read back by
    metal_recording.m:653 for a digest; header metal_ir.h:364 calls it diagnostic.
  - (c) load restores kernel source text from the archive and then recomputes later passes, metal_ir.mm:3440
    (in.Str(kernel.source)) and metal_ir.mm:3464 (Laters(*p)); loaded also calls Laters at metal_ir.mm:3485.
  - (d) several stage roles in one function: metal_ir_lower runs Verify, Transform, Lower and Artifact
    (metal_ir.mm:3285-3303); metal_ir_finalize runs ReplayOrder, Emit, Counts and Laters (metal_ir.mm:3307-3330);
    Artifact runs schedule, storage relation, cost and later-pass derivation (metal_ir.mm:1982-2060).
  - (d) the header itself describes the file as holding construction, effects, ownership, and passes lowering
    across three files (metal_ir.h:5-9 comment, citing metal_transform.mm and metal_lower.mm).
- **Not determined:**
  - Bodies of Transform, Lower, Emit, SaveTarget, LoadTarget and Map, which are declared in metal_ir.h:851-860 and
    defined in other files (metal_transform.mm, metal_lower.mm), not read.
  - Whether tools/e2b_replay.m, tools/native_bench.m, litert_mesh.cc and metal_recording.m call the functions in
    the role their names suggest; only grep counts were taken.
  - Contents of /Users/mdot/dox/mesh/rdma/mesh-graph.h (mesh_graph_levels, mesh_graph_heap_order,
    mesh_graph_cost_times bodies) were not read.
  - Whether metal_ir_prepare_order's order-part input is produced by a later or earlier stage (caller side not
    read).
  - Whether mesh_compile.py's save/load pairing reads the archive for stage-7 use or recomputes; only the call
    sites were seen.
  - Bodies of Work, Actor, Relation, Realize, Laters, Footprints and Verify were not read in full; roles were
    inferred from their names and call sites.
  - Whether metal_program.h's metal_ir_native_* functions belong to this group (they match the name pattern but
    are not declared in metal_ir.h).

## metal_transform

- **Files:** `metal_transform.mm` (1,452 lines).
- **Externally visible functions:** 1.
- **Takes:**
  - Program &p (struct Program with operations[].place, operations[].kind, regions tint ir::Module, gpu/cpu
    mesh_graph_machine, memory, device, values, controls, region_table) — from metal_ir.h:666-700 struct Program;
    filled by metal_ir.mm caller (metal_transform.mm:1361; metal_ir.h:666)
  - Per-operation placement (place: METAL_IR_GPU/CPU) and sent/received/sendings value flags — from earlier
    placement stage data in Program (placement output) (metal_transform.mm:1044 (op.place),
    metal_transform.mm:1194-1197 (v.received, v.sent, v.sendings))
  - Cost machines p.gpu and p.cpu (mesh_graph_machine) read through mesh_graph_actor_time — from caller via
    metal_ir_machine (metal_ir.h:261) into Program::gpu/cpu (metal_ir.h:669) (metal_transform.mm:991, 994, 1004)
  - Device threadgroup memory p.memory — from metal_ir.mm:3296 (maxThreadgroupMemoryLength of the MTLDevice passed
    to metal_ir_lower) (metal_transform.mm:1045, metal_ir.h:673-675)
  - Regions tint ir::Module p.regions and per-operation region function, combine, finalize — from export/record
    stage output (metal_ir.h:506-514) (metal_transform.mm:1364-1367)
  - Binding-slot counts Bindings(...) and TransportBindings(...) — from metal_lower.mm:1136 (Bindings, declared
    metal_lower.h:162-163) and metal_transport.mm:674 (TransportBindings, declared metal_lower.h:178)
    (metal_transform.mm:1056, 1286-1287)
  - Mapping construction Map(p, op) and Map(p, selected, tiles|threads) — from metal_lower.mm:2153-2168 (declared
    metal_lower.h:156-157) (metal_transform.mm:1388, 1393, 1399, 973)
- **Returns:**
  - int status (0 on success); side effect on Program: p.kernels (Kernel with operations, mapping, tile, locals,
    hostEntries), per-operation implemented[] kernel indices, Value::internal storage marks, controlOwners
    (LocalControls), CPU composition (ComposeCPU) — to metal_ir.mm:3297-3298 (metal_ir_lower: status checked, then
    Lower(*p) at metal_ir.mm:3299-3300) (metal_transform.mm:1432-1449; metal_transform.mm:1182-1190 (Internal sets
    u.internal); metal_ir.mm:3297)
  - Host dispatch entries with threadgroup counts Groups(...) (runtime-facing data) — to
    Program::kernels[].hostEntries read by later stages (not determined from this file)
    (metal_transform.mm:1436-1444; metal_ir.h:253-254 comment)
- **Role:** more than one (fusion and grouping of operations into kernels with tile mappings: schedule-like; IR
  simplification and zero-point specialization over tint regions: a rewrite pass; cost-based grouping decisions
  through mesh_graph_actor_time: cost evaluation; host dispatch entries and CPU composition with
  LocalControls/ComposeCPU: lower/runtime-facing)
- **Direction:** does not match.
- **Where, as the reader states it:**
  - metal_transform.mm:1056 and 1286-1287: calls Bindings(p, members) and TransportBindings(p, members), which are
    defined in metal_lower.mm:1136 and metal_transport.mm:674 (later lower and transport stages, per
    metal_lower.h:162-178 comments), so fusion takes a count that a later stage produces.
  - metal_transform.mm:1388, 1393, 1399 and 973: calls Map(p, ...), defined at metal_lower.mm:2153-2168;
    metal_lower.h:156-158 documents it as the mapping "as Construct lowers it", so fusion consumes lower-stage
    output.
  - metal_transform.mm:1436-1444: the transform builds hostEntries (groups, last, operation) inside the kernel
    record, i.e. dispatch geometry that the runtime stage consumes; this performs runtime-stage data production in
    the same function as fusion.
  - metal_transform.mm:1447-1448 with 1192-1200 and 1313-1316: LocalControls and ComposeCPU run inside Transform
    and set control owners and CPU composition, which are lower-stage concerns in the same pass.
  - metal_transform.mm:1374-1375 and 1361-1366: Inline, Epilogue, Simplify and ZeroPoints perform IR rewriting
    over tint regions (specialization and fusion), a separate role from the schedule decision at 1388-1434.
  - No place found where metal_transform.mm reads emitted output (no MTL library, source text, binary, or debug-
    information use): grep for MTL, newLibrary, MTLLibrary, MTLComputePipeline, .metal, msl, source, binary,
    archive, dlopen, fopen, NSData, Reflection returned no matches in metal_transform.mm.
- **Not determined:**
  - The reference stage order's exact input/output types are not defined in the repo; this analysis treats Program
    (metal_ir.h:666) as one shared mutable value across stages, so a clean per-stage boundary cannot be read off
    the function signature alone.
  - Whether Map, Bindings, TransportBindings and Construct are pure functions of Program (their bodies in
    metal_lower.mm and metal_transport.mm were not read in full).
  - Whether hostEntries (metal_transform.mm:1436-1444) is consumed as a schedule output or as runtime input: the
    consumers were not traced beyond metal_ir.h comments.
  - Whether Transform's kernel grouping is the schedule stage output or a separate fusion decision: the Kernel
    struct fields (metal_ir.h:605-625) were not read.
  - The Joins overload: metal_transform.mm:1034 (bool Joins) and metal_transport.mm:625 (int Joins(Program&,
    char*, size_t)) share a name; which is called at metal_transform.mm:1394 and 1400 was judged from signature,
    not verified across namespaces.
  - Callers outside metal_ir.mm: only metal_ir.mm:3297 calls Transform; /Users/mdot/dox/mesh/rdma was searched for
    metal_transform and Transform( and returned no hits, but the search did not cover every file type in that
    tree.

## metal_lower

- **Files:** `metal_lower.h` (193 lines), `metal_lower.mm` (2,666 lines).
- **Externally visible functions:** 49.
- **Takes:**
  - Program& (metal_ir.h Program, the whole program state: operations, values, kernels, device) — from metal_ir.mm
    callers metal_ir.mm:3299 (Lower), 3318 (Emit), 3359/3396 (SaveTarget), 3444 (LoadTarget); declared
    metal_lower.h:156-184 and metal_ir.h:851-861 (metal_lower.mm:2499 int Lower(Program &p, ...);
    metal_lower.mm:2603 int Emit(Program &p, ...))
  - Kernel& and tint::core::ir::Module& / Function* (tint IR module of a kernel body being built) — from tint core
    IR headers included at metal_lower.mm:43-57; Builder fields metal_lower.h:18-22 (Program &p, Kernel &k,
    tint::core::ir::Module &m, tint::core::type::Manager &t) (metal_lower.h:17-22 struct Builder;
    metal_lower.mm:382 std::unique_ptr<ir::Module> Builder::Module())
  - Mapping (execution mapping/threads/tiles per operation) by const reference, and uint32_t thread counts — from
    metal_ir.h Mapping type, consumed here at metal_lower.mm:2153, 2160, 2168 and metal_lower.h:156-157
    (metal_lower.h:156-157 Mapping Map(const Program &, const Mapping &selected, const Mapping &tiles);
    metal_lower.h:157 Mapping Map(const Program &, const Mapping &selected, uint32_t threads))
  - uint32_t kernel index, operation index and input index — from caller-supplied indices into
    Program::kernels/operations/values (metal_ir.mm:3299, metal_transport.mm:335 for Construct) (metal_lower.h:167
    int Construct(Program &, uint32_t kernel, ...); metal_lower.h:168 Materialized(const Program &, uint32_t
    operation, uint32_t input))
  - void* device (Metal device handle) and const uint8_t* blob with size — from metal_ir_load arguments passed
    through metal_ir.mm:3406-3444; metal_ir_lower device argument metal_ir.mm:3285 (metal_lower.mm:2107 int
    LoadTarget(Program &p, uint32_t index, void *device, const uint8_t *blob, size_t bytes, ...))
  - std::vector<uint32_t> &created (values the construction adds, reused across passes) — from transport caller
    metal_transport.mm:335 Construct(p, laterIndex, created, error, room) (metal_lower.h:165-167 comment and
    declaration; metal_lower.mm:1246)
  - char* error and size_t room (error text buffer) — from all exported int-returning functions, e.g.
    metal_lower.mm:71 LowerFail, metal_lower.mm:2499, 2603 (metal_lower.mm:71 int LowerFail(char *error, size_t
    room, const std::string &text))
  - Device-side Metal objects (MTLDevice via p.device) read at Emit — from Program::device set before Emit
    (metal_ir.mm:3285-3299 path, field read at metal_lower.mm:2604) (metal_lower.mm:2604 id<MTLDevice> mtl =
    (__bridge id<MTLDevice>)p.device;)
  - Shader library bytes operator_shaders_metal (numeric function source for the CPU/native target) — from
    operator_shaders_metal.h, included at metal_lower.mm:25 and written as metal_stdlib in NativeBuild
    (metal_lower.mm:25 #include "operator_shaders_metal.h"; metal_lower.mm:1572 write("metal_stdlib",
    operator_shaders_metal, operator_shaders_metal_len))
  - Tuned mapping selector MapTuned (defined in metal_native.mm:1173, supplied from outside this group) — from
    metal_native.mm:1173 bool MapTuned(const Program &p, uint32_t op, Mapping &mapping) (metal_lower.mm:2162 if
    (MapTuned(p, op, tuned)) return tuned;)
  - Simplify (defined in metal_transform.mm:157, declared here) — from metal_transform.mm:157 void
    Simplify(ir::Module &m, ir::Function *f) (metal_lower.h:149 void Simplify(tint::core::ir::Module &,
    tint::core::ir::Function *);)
  - NativePlan / Projection / Construction records (plan values chosen by the tuned targets) — from
    metal_native.mm (Choose; Construction produced in metal_native.mm, per metal_lower.h:105-144 comments)
    (metal_lower.h:109-144 struct NativePlan, Projection, Construction; metal_native.h:8 comment referencing
    metal_lower.h)
- **Returns:**
  - int status (0 or errno-like) with the Program mutated in place: kernels appended (metal_lower.mm:2510-2520),
    op.implemented set, kernel.kind/checkValue set — to metal_ir.mm:3299 (metal_ir_lower)
    (metal_lower.mm:2499-2530 int Lower(...); metal_ir.mm:3299 if (!status) status = Lower(*p, error, room);)
  - int status with Kernel::pipeline, Kernel::binary, Kernel::body cleared; Program finalized in place — to
    metal_ir.mm:3318 (metal_ir_finalize) (metal_lower.mm:2603 int Emit(...); metal_lower.mm:2658-2663 k.pipeline =
    ...; k.body = nullptr; metal_ir.mm:3318 const int status = Emit(*p, error, room);)
  - serialized target bytes std::vector<uint8_t>& blob (a kernel's compiled archive/library bytes) — to
    metal_ir.mm:3396 (metal_ir_save) (metal_lower.mm:2085 int SaveTarget(...std::vector<uint8_t> &blob ...);
    metal_ir.mm:3396 if (int status = SaveTarget(*p, k, blob, error, room)))
  - Kernel::pipeline / Kernel::task / Kernel::neural restored from bytes (side effect), plus binary and object
    bytes — to metal_ir.mm:3444 (metal_ir_load) (metal_lower.mm:2107-2149 LoadTarget sets k.binary (2128), k.task
    (2148 region), k.pipeline (2149); metal_ir.mm:3444 if (LoadTarget(*p, k, device, ...)))
  - Mapping by value to Lower and Construct paths — to metal_lower.mm:2510 (Lower), metal_lower.mm:1274
    (Construct), metal_lower.mm:2009 (PrepareConstants region) (metal_lower.mm:2510 mapping = Map(p, operation);
    metal_lower.mm:1274 if (!k.mapping[i].construction) k.mapping[i] = Map(p, k.operations[i]);)
  - const Materialization* (prepared constant record or null) — to metal_native.mm:1094, 1114, 1148
    (Materialized); metal_neural.mm:781, 782, 790 (Materialized); metal_lower.mm:2023 (metal_lower.mm:2080-2083
    const Materialization *Materialized(...); metal_neural.mm:781 if (Materialized(p, index, contract->right))
    weightInput = contract->right;)
  - Construct: the kernel's tint IR module k.module and the created value list (side effect), int status — to
    metal_transport.mm:335 (Construct called for the later pass) (metal_lower.mm:1248-1250 k.module =
    Builder::Module(); metal_transport.mm:335 int status = Construct(p, laterIndex, created, error, room);)
  - uint32_t slot count from Bindings — to metal_transform.mm:1056 (Bindings in the fusion budget check)
    (metal_lower.mm:1136 uint32_t Bindings(...); metal_transform.mm:1056 return !gpu || Bindings(p, members) +
    TransportBindings(p, members) + extraBindings <= kBindings;)
  - LowerFail error text (side effect on error buffer), int status — to many callers in metal_lower.mm,
    metal_native.mm, metal_neural.mm (e.g. metal_neural.mm:1255 LowerFail) (metal_lower.mm:71-76 LowerFail;
    metal_neural.mm:1255 return LowerFail(error, room, ...))
- **Role:** more than one (name them): (5) lower: Builder, Construct and Lower build tint IR kernel bodies
  (metal_lower.mm:17-25 of .h, 1246, 2499); (2)/(4) mapping: Map chooses thread/tile/mapping per operation
  (metal_lower.mm:2153-2170, 2510, 1274); (6) runtime: PrepareConstants compiles a task and runs it at lowering
  time (metal_lower.mm:2052-2056), LoadTarget and Emit bind device objects and symbols (metal_lower.mm:2140-2151,
  1658); (7) archive: SaveTarget (2085) and LoadTarget (2107) move target bytes to and from files; (3) devices:
  Pipeline and Emit take the MTLDevice (metal_lower.mm:1491, 2604); supplier of numeric functions:
  operator_shaders_metal included and written to metal_stdlib (metal_lower.mm:25, 1572); external compiler
  invocation: xcrun clang++, lldb, dsymutil, Metal compiler (metal_lower.mm:1498, 1596, 1611, 1637). Also the
  header metal_lower.h declares shared types (Builder, NativePlan, Projection, Construction) at lines 17-144 and
  functions defined in other files (Transport, Joins, MapTuned, LowerNeural) at lines 169-191.
- **Direction:** does not match.
- **Where, as the reader states it:**
  - (a) takes a later-stage output: Emit consumes TransportMask and Wrap results of the transport pass, which
    metal_lower.h:169-176 describes as the transport structure after a kernel's body, realized at Emit;
    metal_lower.mm:2605 int status = TransportMask(p, error, room); the transport is produced in
    metal_transport.mm, which metal_ir.h:848-849 orders between metal_ir_lower and metal_ir_finalize.
  - (a) takes a later-stage output: Construct takes the later pass's Kernel and the created value list built by
    metal_transport.mm, not the reverse, at metal_lower.h:165-167 and metal_transport.mm:335 (the caller is a
    stage that runs after the first construction).
  - (b) produces what should come from an earlier stage: Lower computes the execution Mapping itself, mapping =
    Map(p, operation), at metal_lower.mm:2510, and Construct fills it at metal_lower.mm:1274; Map then selects a
    tuned mapping through MapTuned at metal_lower.mm:2162, a supplier outside the group (metal_native.mm:1173).
  - (b) produces what should come from a supplier: PrepareConstants executes a compiled batch to produce constant
    data during lowering, metal_lower.mm:2052-2056 (CompileTask then batch.task(pointers.data(),
    unsigned(pointers.size()))), a runtime execution inside the lower step.
  - (c) reads back emitted output: NativeBuild writes objects, links a dynamic library, then runs xcrun lldb with
    a generated Python script that writes a .sites file, and reads that file back to recover debug site records,
    metal_lower.mm:1611-1644 (invoke at 1637, fopen of library+".sites" at 1639, fread at 1644).
  - (c) reads back emitted output: after linking, the group reads the library and object bytes back from disk into
    result.binary and result.object, metal_lower.mm:1650-1660 (fopen(path) then fread into bytes->data()).
  - (c) reads back emitted output: Archive reads the serialized Metal archive file back into k.binary,
    metal_lower.mm:1515-1519 (serializeAsArchiveAndFlushToURL then dataWithContentsOfURL then k.binary.assign).
  - (c) reads back emitted output: LoadTarget writes the saved blob to a temporary file and loads the archive and
    library from that URL, metal_lower.mm:2128-2142 (writeToURL, newArchiveWithURL, newLibraryWithURL).
  - (d) several stage roles at once: Lower performs structure (2499-2530), operation mapping (2510), Check kernel
    construction (2530-2540), Unroll and PrepareConstants (2542-2543), with PrepareConstants compiling and running
    a task (2052-2056).
  - (d) several stage roles at once: Emit performs transport masks (2605), source generation through Source
    (1523-1545 via tint::msl::writer::Generate at 1541), pipeline compilation (1491-1510), archive capture
    (1512-1521) and binary selection (2611-2663) in one function, metal_lower.mm:2603-2665.
  - (d) several stage roles at once: NativeBuild performs linking with external tools, debug-metadata extraction,
    dlopen and dlsym binding of the entry symbol, metal_lower.mm:1563-1670 (dlopen at 1658, dlsym immediately
    after).
  - (d) several stage roles at once: LoadTarget restores bytes (2107-2128), then creates pipelines on a device
    (newComputePipelineStateWithDescriptor at 2150 area), in one function; this combines archive restore and
    runtime binding, metal_lower.mm:2107-2151.
- **Not determined:**
  - Program struct field definitions (device, kernels, materializations) in metal_ir.h were not read in full; the
    types Kernel, Operation, Value, Mapping, Materialization were not inspected beyond their use.
  - PrepareConstants body metal_lower.mm:1806-2050 was not read in full; only its CompileTask and batch.task call
    sites (2052-2056) were seen.
  - Lower body metal_lower.mm:2520-2540 was only partly read, and Source body metal_lower.mm:1523-1560 only
    partly.
  - Whether the program's device handle is set before Emit runs: the setter was not traced beyond
    metal_ir.mm:3285-3318 call order.
  - Callers of Bindings outside metal_transform.mm:1056 were not exhaustively enumerated; callers of Sequence,
    Map(Program&, Mapping&, Mapping&) and Map(Program&, Mapping&, uint32_t) were not searched outside this file.
  - Whether metal_transport.mm's transport output is produced before or after the lower stage at runtime: the
    ordering is taken from comments at metal_ir.h:848-849 and metal_lower.h:169-176, not from call order.
  - Whether the group's Construct results depend on Transport state beyond metal_transport.mm:335.

## metal_native

- **Files:** `metal_native.h` (17 lines), `metal_native.mm` (1,216 lines).
- **Externally visible functions:** 2.
- **Called from:**
  - metal_lower.mm (Map at line 2162 calls MapTuned; Construct at line 1366 calls NativeMember; line 704 calls
    MetalNativeCores)
  - tools/native_bench.m (lines 515 and 521 call MetalNativeCores and MetalNativeFamily)
  - metal_lower.h (declares MapTuned at line 183 and NativeMember at line 191; line 182 comment names
    metal_native.mm)
- **Takes:**
  - id<MTLDevice> device (MetalNativeFamily argument) — from caller of the export (tools/native_bench.m:521 passes
    device) (metal_native.h:11; metal_native.mm:44-46)
  - no operand for cores or family: MetalNativeCores() reads the IORegistry AGXAccelerator property gpu-core-
    count, default 8 — from IOKit (IOServiceGetMatchingService / IORegistryEntryCreateCFProperty)
    (metal_native.mm:26-42)
  - Family() takes the family from MTLCreateSystemDefaultDevice() internally, not from an operand — from Metal
    framework (metal_native.mm:371-376)
  - const Program &p (placed operation graph; reads p.operations[op].place against METAL_IR_CPU / METAL_IR_GPU)
    and uint32_t op — from metal_lower.mm:2162 (Map) via metal_lower.h:183; Program and Mapping types from
    metal_lower.h / metal_ir.h (not read in full) (metal_native.mm:1173-1178; metal_ir.h:64)
  - Builder &e (builder of the Tint IR module, entry, block) and an operand Mapping mapping with its construction
    — from metal_lower.mm:1366 (Construct) via metal_lower.h:191 (metal_native.mm:1200-1204)
  - Tint IR Builder state (e.b, e.m, e.entry) used to append instructions — from metal_lower.h Builder
    (metal_lower.h:17) (metal_native.mm:1119, 1160, 599)
  - Projection x from typed operation Recognize/RecognizeDense of Program p — from metal_native.mm:1181-1182
    (Recognize / RecognizeDense on the Program) (metal_native.mm:113, 196, 1181-1183)
- **Returns:**
  - NSUInteger GPU core count (default 8) — to metal_lower.mm:704; metal_native.mm:1184 (Choose);
    tools/native_bench.m:515 (metal_native.mm:26-42; metal_lower.mm:704; tools/native_bench.m:515)
  - NSUInteger Apple family (10 or 9) — to metal_native.mm:374 (Family); tools/native_bench.m:521
    (metal_native.mm:44-46; tools/native_bench.m:521)
  - bool with Mapping &mapping filled: operation, axis, tile, threads, lanes, order, partials, staged, memory,
    extraBindings, work (mesh_graph_work), algorithm METAL_IR_PROJECTION or GemmMapping, and mapping.construction
    — to metal_lower.mm:2162 (Map returns tuned mapping to the scheduler path) (metal_native.mm:1173-1198;
    567-597; metal_lower.mm:2162)
  - mesh_graph_work cost (operations per resource and width) built from the Program's prologue, epilogue and
    accumulator — to Mapping.work consumed by the caller (metal_lower.mm, not read) (metal_native.mm:545-565;
    306-307; 306-312)
  - int status; Tint IR instructions appended: msl::ir::NativeMatmul (kBlas/kBNNS) and msl::ir::Fence(kDevice),
    with value cursor updates in created — to Construct in metal_lower.mm:1366 and the emitted Tint IR module
    (metal_native.mm:1160-1161; 599; 1200-1214; metal_lower.mm:1366)
  - workspace byte count via BNNSMatMulWorkspaceSize — to HostMember (metal_native.mm:1155-1158) and host
    algorithm choice (metal_native.mm:287-294) (metal_native.mm:245-254)
- **Role:** more than one: (1) supplier of target facts (cores, family) as exported functions; (2) schedule stage work
  (Choose, Mapped, GemmMapping select lanes, threads, tiles, memory and partials from the typed operation); (3)
  cost computation (MapHost, TunedWork build mesh_graph_work); (4) lower stage work (NativeMember and
  Specialize/ProjectionMember/GemmMember append Tint IR instructions such as NativeMatmul and Fence); also a
  weight data relayout (Relayout, Weights at lines 399-461)
- **Direction:** does not match.
- **Where, as the reader states it:**
  - metal_native.mm:317-369 (Choose) takes the target cores and family and returns a NativePlan of schedule
    choices (lanes, simdgroups, parts, memory), so the group performs schedule work in the same unit that builds
    the lowered program: criterion (d).
  - metal_native.mm:1184-1194 (MapTuned) calls Choose, Specialize and Mapped and then returns a Mapping that
    carries cost (mapping.work from TunedWork, metal_native.mm:545-565) in one call: criterion (d), schedule and
    cost at once.
  - metal_native.mm:1200-1214 (NativeMember) appends Tint IR instructions (CreateInstruction<NativeMatmul> at
    metal_native.mm:1160-1161, Fence at metal_native.mm:599): criterion (d), lower stage role inside the same file
    as schedule.
  - metal_native.mm:374 and 28-33 read the target directly from MTLCreateSystemDefaultDevice and IORegistry rather
    than taking cores or family as an operand of MapTuned: criterion (b), a supplier value obtained internally
    instead of received from the stage that should supply it.
  - metal_native.mm:399-450 (Relayout called from Weights) rewrites constant weight bytes into a new layout inside
    the construction path called from metal_native.mm:614 and 939: criterion (d), a data transform performed
    during lowering that is neither a schedule nor a lowering-to-text step.
  - metal_native.mm:1173-1178 reads p.operations[op].place, a placed graph, which is the earlier placement stage
    output: this direction is consistent with the order and is not counted as a violation.
- **Not determined:**
  - The definitions of Program, Mapping, Construction and Projection were not read in full (metal_lower.h:109-141
    and metal_ir.h only partially inspected), so the exact type of the scheduled program each stage returns is not
    determined.
  - The downstream consumers of Mapping.work and the scheduler's cost operand in metal_lower.mm (around line
    2150-2170 and Map/Decide) were not read, so whether a cost operand flows into Choose is not determined.
  - Whether Weights/Relayout output (metal_native.mm:399-461) is later re-read from emitted output or from the
    runtime was not determined.
  - The Makefile reference to metal_native was found by grep but its target and build role were not read.
  - docs/kernels.md:42 mention of MetalNativeAttention was not checked against the current file, because
    metal_native.mm defines no MetalNativeAttention symbol in the list reviewed.
  - Whether metal_native.mm line 422 (dispatch_apply_f over KS) runs at lower time or at runtime was not
    determined from the boundary.

## metal_clone and metal_transport

- **Files:** `metal_clone.h` (27 lines), `metal_clone.mm` (203 lines), `metal_transport.mm` (695 lines).
- **Externally visible functions:** 8.
- **Called from:**
  - metal_lower.mm (Splice 315, 1257, 1817; AccessMode 183; Transport 2473, 2569, 2577, 2585, 2595; Joins 2589;
    TransportMask 2605 in Emit; Acquire 2197; Release 2199)
  - metal_native.mm (Splice 537; includes metal_clone.h line 14)
  - metal_transform.mm (Splice 259 via Spliced, called 277, 280, 689, 704, 713; TransportBindings 1056, 1286;
    includes metal_clone.h line 3)
  - metal_ir.mm (Splice 2375 with imported map; includes metal_clone.h line 22)
  - Makefile:104 and 114 (builds metal_clone.mm and metal_transport.mm into libmetal_ir.dylib)
- **Takes:**
  - tint::core::ir::Module& from (source module; a region or native module or the f64 library module) — from
    Program::regions (metal_ir.h:667), selected.module (metal_native.mm:537), library module (metal_ir.mm:2375),
    p.regions (metal_transform.mm:259) (metal_clone.h:22; metal_clone.mm:165-167)
  - tint::core::ir::Function* f to clone (region body or callee) — from caller's Program/Construction body
    (metal_lower.mm:315, metal_native.mm:537, metal_ir.mm:2375) (metal_clone.h:22; metal_clone.mm:165)
  - tint::core::ir::Builder& b (target insertion point and target module b.ir) — from caller's builder
    (metal_lower.mm:315, metal_native.mm:537, metal_transform.mm:259) (metal_clone.h:22; metal_clone.mm:168-169)
  - const std::vector<tint::core::ir::Value*>& arguments (values replacing the parameters; null means zero) — from
    caller operands (metal_lower.mm:1257 e.lid/e.group/e.lane; metal_native.mm:537 arguments) (metal_clone.h:23;
    metal_clone.mm:174-180)
  - std::function<ir::Value*(ir::Var*)> var mapper for module-scope vars of from — from caller callback
    (metal_lower.mm:315 binding lookup; metal_transform.mm:259 v->Result(); metal_ir.mm:2375 returns nullptr)
    (metal_clone.h:24; metal_clone.mm:86-88)
  - std::unordered_map<ir::Function*, ir::Function*>* imported (functions already cloned into b.ir, reused and
    extended) — from caller map (metal_ir.mm:2375 &imported) (metal_clone.h:25; metal_clone.mm:170-171)
  - tint::core::ir::Module& module and ir::Var* var, tint::core::Access (AccessMode) — from lower caller
    (metal_lower.mm:183 AccessMode(m, found->second, kReadWrite)) (metal_clone.h:14; metal_clone.mm:154-157)
  - Program& p (kernels, values, later_passes, controlEntries, regions, device) — from Program struct
    (metal_ir.h:666-667) passed by lower entries (metal_lower.mm:2473, 2585, 2589, 2595, 2605) and Emit
    (metal_lower.mm:2603) (metal_transport.mm:278, 574 (Builder& e), 612, 625, 674; metal_lower.h:173-178)
  - Builder& e (Program& p, Kernel&, tint builder e.b, e.m module, e.t types) and ir::Function* wrapper,
    ir::Value* lid/gid/lane — from lower entry points (metal_lower.mm:2197-2199 Acquire/Release; Builder declared
    metal_lower.h:17-21) (metal_transport.mm:431, 574; metal_lower.h:179-181)
  - const Program& p and const std::vector<uint32_t>& members (TransportBindings) — from metal_transform.mm
    grouping callers (metal_transform.mm:1056, 1286) (metal_transport.mm:674-675; metal_lower.h:178)
  - external mesh library: mesh_metal_mask, mesh_metal_release, mesh_metal_readiness, mesh_metal_range structs —
    from ../dox/mesh/rdma/mesh-metal.h (included as mesh-metal.h; Makefile MESH_RDMA) (metal_transport.mm:12,
    268-270, 461, 484, 588)
- **Returns:**
  - tint::core::ir::Value* result: the clone of the value of the one return of f, placed in b.ir — to
    metal_lower.mm:315 (Region/Spliced value), metal_lower.mm:1257, metal_lower.mm:1817, metal_native.mm:537
    (Activation), metal_transform.mm:259 (Spliced), metal_ir.mm:2375 (f64 call result) (metal_clone.h:22;
    metal_clone.mm:189-200)
  - void (AccessMode): mutates Var result type and PointerResult types in module — to metal_lower.mm:183
    (metal_clone.mm:154-163)
  - int status (0 or error) and inserted count; mutates Program p (kernels inserted at index+1, values, bindings,
    created list) — to metal_lower.mm:2473, 2569, 2577, 2585, 2595 (Lower) (metal_transport.mm:278, 326-336
    (InsertKernels, Construct call); metal_lower.h:173)
  - ir::Value* (Acquire): the body's threadgroup origin; emits acquisition into wrapper — to metal_lower.mm:2197
    (metal_transport.mm:431; metal_lower.mm:2197)
  - void (Release): emits release after the body — to metal_lower.mm:2199 (metal_transport.mm:574;
    metal_lower.mm:2199)
  - int status (TransportMask): sets each values[u].release.total and first-pass masks in Program p — to
    metal_lower.mm:2605 inside Emit (2603); then Wrap and dispatch (metal_lower.mm:2606-2609)
    (metal_transport.mm:612-619; metal_lower.mm:2603-2605)
  - int status (Joins): sets one release per joined sent value in Program p — to metal_lower.mm:2589 (Lower,
    before Control) (metal_transport.mm:625-670; metal_lower.mm:2588-2589)
  - uint32_t binding count (TransportBindings) — to metal_transform.mm:1056 (Joins grouping check),
    metal_transform.mm:1286 (Bindings + transport) (metal_transport.mm:674-693; metal_transform.mm:1056, 1286)
- **Role:** more than one: metal_clone.mm/.h is a shared IR-splice and clone helper used as lower-stage helper
  (metal_lower.mm:183, 315, 1257, 1817; metal_native.mm:537), a transform-stage helper (metal_transform.mm:259),
  and a splice of the numeric-function library (metal_ir.mm:2375, the supplier role); metal_transport.mm performs
  lower-stage structure and emission (Transport metal_transport.mm:278, Acquire 431, Release 574, Joins 625,
  Coherent 86), an Emit-time realization (TransportMask metal_transport.mm:612 called from metal_lower.mm:2605),
  and a count helper consumed by a grouping function (TransportBindings metal_transport.mm:674, used at
  metal_transform.mm:1056 and 1286)
- **Direction:** mixed.
- **Where, as the reader states it:**
  - metal_transport.mm:335: Transport (a lower-stage entry) calls Construct (declared metal_lower.h:167, the lower
    stage's own body construction) for the later pass, so the group performs the construct role internally as well
    as the transport role (performs several stage roles).
  - metal_transport.mm:612-619 and metal_lower.mm:2603-2605: TransportMask, the realization of masks and release
    totals, runs inside Emit, the final step after Lower, and it writes release.total into Program values that the
    Emit path then uses (metal_lower.mm:2606-2609); the group takes a later-step invocation for a value the lower
    entries did not produce (performs emit and lower roles).
  - metal_transport.mm:268-270 and 461-484: the group calls the external mesh_metal_mask and reads
    mesh_metal_readiness/range from the mesh library, a supplier outside the stage order, so the group consumes a
    supplier's function (performs a supplier-side computation at lower and emit time).
  - metal_transport.mm:674-693 with metal_transform.mm:1056 and 1286: TransportBindings, a function of the
    transport's per-value flags, is consumed by metal_transform.mm grouping (Joins at metal_transform.mm:1034) to
    decide group membership; whether that is a backward dependency depends on the stage of metal_transform.mm,
    which the boundary does not show.
  - metal_clone.h:22-25 and metal_clone.mm:165-200: Splice takes a from module, a Builder and arguments, and
    returns a Value; it does not read any later-stage output; metal_clone.mm:85 and 90-98 read only the source
    module's instructions (tint IR, not emitted text or binaries). This portion matches the direction of its
    callers.
  - No place found in any of the three files where the group reads back emitted output (source text, compiled
    binaries, debug information): grep of metal_transport.mm for
    source/emit/MSL/compile/binary/blob/getenv/dlsym/newLibrary/debug returned only
    CoherentStore/CoherentLoad/Fence creation (metal_transport.mm:107, 475, 548-602) and LowerFail strings
    (metal_transport.mm:308, 349), none reading an emitted artifact.
- **Not determined:**
  - The stage of metal_transform.mm, metal_native.mm and metal_ir.mm in the reference order (their own boundaries
    were not analysed), so whether the TransportBindings consumption at metal_transform.mm:1056 and 1286 is a
    backward dependency.
  - Whether Emit-time TransportMask (metal_transport.mm:612) belongs to the lower stage or to the target-compiler
    step, since the reference order does not name the Emit step.
  - Semantics of mesh_metal_mask, mesh_metal_release, mesh_metal_readiness and mesh_metal_range:
    ../dox/mesh/rdma/mesh-metal.h was not opened (only the call sites and comments were read).
  - Whether metal_ir.mm:2375 (Splice of metal_binary64.wgsl entries) is the supplier role in the sense of the task
    text or a lower-stage use; the library's content was not read.
  - The full bodies of Transport (metal_transport.mm:278-427), Joins (625-670), and the anonymous-namespace
    helpers (lines 36-273), which were not read in full by instruction; per-line claims about them are limited to
    the lines cited.
  - Who calls AccessMode and Splice beyond the cited lines (grep covered /Users/mdot/metal-microbench excluding
    tools/st-debug and /Users/mdot/dox/mesh/rdma; other trees were not searched).

## metal_neural

- **Files:** `metal_neural.mm` (1,503 lines).
- **Externally visible functions:** 6.
- **Called from:**
  - metal_lower.mm:2574 (LowerNeural, in Lower pass)
  - metal_lower.mm:2621 (EmitNeural, in Emit pass dispatch_apply)
  - metal_lower.mm:2087 (SaveNeural via SaveTarget)
  - metal_lower.mm:2109 (LoadNeural via LoadTarget)
  - metal_ir.mm:3688-3690 (metal_ir_neural returns Kernel::neural and context)
  - metal_ir.mm:3693-3695 (metal_ir_neural_bind calls BindNeural)
  - metal_lower.h:184-190 (declarations)
  - Makefile:104 (METAL_IR_SRCS wildcard build inclusion, not a call)
  - /Users/mdot/dox/mesh/rdma: no matches for the exported names
- **Takes:**
  - Program& (mutable): kernels, operations (with erased flag and inputs/outputs), values (storage, offset, shape,
    scalar, layout, internal), regions and constant contents, owners — from metal_ir.h Program (metal_ir.h:666),
    passed by metal_lower.mm Lower/Emit and SaveTarget/LoadTarget (metal_neural.mm:1250-1251, 1254, 1262-1264,
    1275, 1314-1315; metal_lower.h:186)
  - uint32_t kernel index into Program::kernels, with Kernel::operations as the ANE group — from
    metal_lower.mm:2519-2530 (neural grouping lambda on op.place == METAL_IR_ANE), metal_lower.mm:2574 caller
    (metal_neural.mm:1251, 1253; metal_lower.mm:2519-2530)
  - Operation region bodies as tint core IR (ir::Function, ir::Block, instructions CoreBinary, CoreUnary, Convert,
    CoreBuiltinCall, UserCall, Construct, Access, Let, If, ExitIf, Return) walked by Lowering::Region — from tint
    IR headers included at metal_neural.mm:15-31; metal_ir Contract/Materialization data from Program
    (metal_neural.mm:951, 983) (metal_neural.mm:15-31, 56-57, 581-752, 772-776)
  - Tensor/value types via p.regions.Types() and ty::Type (ComputeType, MilType) — from Program regions
    (metal_neural.mm:318, 515, 778) (metal_neural.mm:318-320, 515, 778)
  - Device pointers per binding slot (const void *const *pointers, count) — from metal_ir.mm:3693-3695
    metal_ir_neural_bind caller, runtime binding (metal_neural.mm:1199-1204)
  - Core ML artifact blob (bytes of a property list: files, inputs, outputs) — from metal_lower.mm:2109 LoadTarget
    (blob from disk, via archive) (metal_neural.mm:1448-1456; metal_lower.mm:2109)
  - Prediction feature provider inputs and output MLMultiArray backings from caller-owned pointers — from caller
    pointers via BindNeural (metal_neural.mm:1210-1213 view lambda) (metal_neural.mm:1203-1213, 1218-1242)
- **Returns:**
  - int status plus error text written into char* error; Kernel fields set: kind = METAL_IR_NEURAL, neural =
    Invoke, context = Neural*, bindings (Binding slot, storage, access, offset, bytes), embedded, owners entry,
    and per-value internal = kernel (storage marker) — to metal_lower.mm Lower (caller at metal_lower.mm:2574,
    LowerNeural result checked), and the Program used by later Transport and Emit passes
    (metal_neural.mm:1251-1257, 1268-1271, 1283-1291, 1318, 1359; metal_lower.mm:2574)
  - Neural::program (serialized Core ML model proto), Neural::weights, Neural::summary text (MIL description) held
    in the Neural context — to EmitNeural (same file, metal_neural.mm:1361-1418) (metal_neural.mm:1120-1124;
    metal_neural.mm:1347-1357; metal_neural.mm:1367-1378)
  - int status plus error text, and Kernel::devices (three counts: Neural Engine, CPU, GPU) and Kernel::source
    (summary and compute plan text) written into the Program — to metal_lower.mm Emit dispatch at
    metal_lower.mm:2621 (statuses and failure text), Kernel::devices/source consumers in metal_lower.mm
    (metal_neural.mm:1361, 1411-1414; metal_lower.mm:2619-2622)
  - Serialized Core ML artifact bytes (binary property list of compiled file contents and IO records) into blob —
    to metal_lower.mm SaveTarget (metal_lower.mm:2087) (metal_neural.mm:1420-1447; metal_lower.mm:2087)
  - Restored Kernel state: Neural context, compiled and loaded MLModel, Kernel::neural = Invoke — to
    metal_lower.mm LoadTarget (metal_lower.mm:2109) (metal_neural.mm:1448-1501 (esp. 1496-1499);
    metal_lower.mm:2109)
  - BindNeural: a heap MetalNeuralInvocation (MLDictionaryFeatureProvider, MLPredictionOptions with
    outputBackings, copy plans) returned through void **context; Invoke: int status 0 or errno after
    predictionFromFeatures and output copy — to metal_ir.mm:3688-3695 (metal_ir_neural and metal_ir_neural_bind),
    which the runtime calls (metal_neural.mm:1158-1197 (Invoke), 1199-1248 (BindNeural); metal_ir.mm:3688-3695)
- **Role:** more than one (lower; storage decision inside lower; emit/compile with device-plan readback; archive save and
  load; runtime bind and invoke)
- **Direction:** does not match.
- **Where, as the reader states it:**
  - metal_neural.mm:1313-1320 (LowerNeural sets p.values[value].internal = kernel for values not outside the
    kernel, a storage decision that the reference order assigns to stage 4 schedule, and the outside() predicate
    at 1260-1267 derives liveness from other operations' inputs): role (d) several stage roles in stage 5.
  - metal_neural.mm:1393-1413 (EmitNeural loads MLComputePlan from the compiled model and reads the per-operation
    preferred compute device back from the compiler's output, writing k.devices and k.source): role (c) reads back
    emitted compiler output to recover a device fact that stage 3 devices should supply, and role (d) combining
    compile, load and device recording.
  - metal_neural.mm:1431-1441 (SaveNeural reads compiled model files from disk and checks coremldata.bin): role
    (c) reading back compiled output, combined with archive role (stage 7).
  - metal_neural.mm:1496-1499 (LoadNeural calls MLModel modelWithContentsOfURL, then sets k.context and k.neural =
    Invoke): role (d) combining archive (stage 7) with runtime binding of the callable (stage 6); loading performs
    a compiler-side load, not only restore.
  - metal_neural.mm:1175-1187 (Invoke derives output copy source strides and contiguity from got.strides on each
    call, and metal_neural.mm:1199-1248 BindNeural derives copy plans at bind time): role (d) runtime deriving
    layout facts, which the reference order assigns to lower or schedule, not to runtime (stage 6).
- **Not determined:**
  - Whether the devices and source written at metal_neural.mm:1412-1413 are read by a later stage in
    metal_lower.mm (only the write site was checked).
  - Full definition of Program, Kernel and Value beyond the fields seen (metal_ir.h:435, 587-625, 666 only partly
    read); the scheduled-program shape the reference order calls stage 4 output was not confirmed as the input to
    LowerNeural.
  - Callers of SaveTarget and LoadTarget and whether the archive path is the only route for Core ML blobs (not
    traced).
  - Whether metal_neural.mm Lowering::Region (metal_neural.mm:581-752) reads any tint IR module beyond the
    operation regions (not fully read; function bodies were sampled by search only).
  - Whether Core ML compile and compute-plan results are deterministic across calls, which bears on whether the
    device readback is a fact or a planner choice (not measurable from the source).

## metal_program.h and metal_submission.h

- **Files:** `metal_program.h` (249 lines), `metal_submission.h` (116 lines).
- **Externally visible functions:** 24.
- **Called from:**
  - metal_recording.m (includes metal_program.h at line 3; about 15 MetalSubmissionTransition and about 14
    MetalSubmissionInstall call sites, e.g. metal_recording.m:2217, 3866-3879, 3929-3943, 3961-3977, 4284-4311,
    4375-4392; metal_recording.m:3712-3721 MetalNativeStep calls MetalSubmit)
  - metal_lower.mm (includes metal_program.h at line 27 and metal_program_source.h at line 28; uses NativePC
    layout at lines 1557-1559 and 1642-1644)
  - Makefile (lines 109-112 xxd -i of both headers into .build/metal_program_source.h; line 143 lists both headers
    as dependencies of libmetal_recording.dylib)
  - No caller found in metal_native.mm (no include of metal_program.h; lines 9-24 show no include) or
    metal_native.h
  - No reference to these names found in /Users/mdot/dox/mesh/rdma by grep
  - No definition of metal_ir_native_* found in the repository (only declarations at metal_program.h:45-51)
- **Takes:**
  - struct MetalSubmissionConfig by value {lag, positions, window, flags} (uint32 fields) — from caller of
    MetalSubmissionTransition; stored in NativeProgram.submission (metal_program.h:36) (metal_submission.h:8,
    metal_submission.h:14-15)
  - const struct MetalSubmissionState * (commits, encoded, signalled, tests, completed, completedCommits, pending,
    issued, commitEnds, progressCommits) — from caller-owned state; in metal_program.h it is NativeExecution.state
    (metal_program.h:117) (metal_submission.h:9-10, metal_submission.h:15)
  - uint32_t action (enum MetalSubmissionAction) and uint64_t value — from caller; enum values are defined in this
    group (metal_submission.h:5-7, metal_submission.h:15)
  - struct MetalSubmissionState * and const struct MetalSubmissionEffect * for MetalSubmissionInstall — from
    caller; effect produced by MetalSubmissionTransition (metal_submission.h:79-80)
  - const struct NativeProgram * (sites, controls, conditions, selections, guards, bindings, counts, submission
    config) — from caller of metal_ir_native_compile; no caller found in the repository (metal_program.h:27-37,
    metal_program.h:45)
  - const void * bytes and size_t length for load — from caller of metal_ir_native_load (archive bytes); no caller
    found in the repository (metal_program.h:46)
  - const struct NativeCompiled * for entry, identity, sites, save, free — from output of metal_ir_native_compile
    or load (metal_program.h:47-51)
  - struct NativeExecution * run context holding id<MTL4CommandQueue>, id<MTL4CommandBuffer>,
    id<MTL4ComputeCommandEncoder>, MTLSharedEvent objects, bindings, and function pointers
    start/started/commit_options — from Metal framework objects and caller-supplied callbacks
    (metal_program.h:107-126)
  - const void *const *bindings and per-binding structs (NativeICB, NativeAddress, NativeIndirect, NativeCopy,
    NativeFill, NativeCheck, NativePredicate) — from caller-supplied binding tables; NativeProgram.bindings and
    NativeProgram arrays (metal_program.h:34, metal_program.h:55-84, metal_program.h:199-248)
  - int (*predict)(void *, const void *const *, unsigned) callback and context pointers in NativeNeural — from
    external supplier of neural predict function (numeric function supplier) (metal_program.h:66-70)
  - char *error and size_t room for error text — from caller of compile, load, save and of NativeChecked
    (metal_program.h:45-47, metal_program.h:215-217)
  - const void *entry for NativeImageIdentity, resolved by dladdr — from dynamic loader (dladdr) and the mach-o
    header of the loaded image (metal_program.h:86-91)
- **Returns:**
  - MetalSubmissionEffect by value {wait, signal, ended, slot, stored, completed, completedCommits, encoded,
    action, active, submitted, array} — to metal_recording.m call sites (MetalSubmissionTransition, about 15
    calls) and metal_program.h NativeEncoderBegin, NativeSubmit, NativeProgressSignal, NativeProgressWait,
    NativeDrain (metal_submission.h:14-15, metal_submission.h:76; metal_recording.m:3866-3867, 3929-3930,
    3961-3962; metal_program.h:139, 163, 175, 182, 191, 194)
  - void; mutates MetalSubmissionState through the effect (commit ends, progress commits arrays, pending, commits,
    signalled, tests, issued, encoded) — to metal_recording.m call sites (MetalSubmissionInstall, about 14 calls)
    and metal_program.h Native* helpers (metal_submission.h:79-116; metal_recording.m:3879, 3967, 4286;
    metal_program.h:148, 171, 178)
  - int status and NativeCompiled ** handle from metal_ir_native_compile; int, bytes and length from
    metal_ir_native_save; NativeCompiled * from metal_ir_native_load — to callers of these extern C functions; no
    caller found in the repository (metal_program.h:45-47)
  - NativeEntry function pointer taking (struct NativeExecution *, uint64_t first, uint64_t executions) — to
    runtime invocation of compiled native code; no caller found in the repository (metal_program.h:41,
    metal_program.h:49)
  - const struct NativePC * array with count (site, padding, first, last) for debug site metadata — to
    metal_lower.mm reads the NativePC layout from debug metadata (metal_program.h:40, metal_program.h:51;
    metal_lower.mm:1557-1559, metal_lower.mm:1642-1644)
  - int status and NSUInteger-free void encoding into id<MTL4ComputeCommandEncoder> and id<MTL4ArgumentTable>
    (executeCommandsInBuffer, setAddress, dispatchThreadgroupsWithIndirectBuffer, copyFromBuffer, fillBuffer) — to
    Metal command encoder of NativeExecution; caller not found in metal_recording.m, metal_lower.mm, or
    metal_native.mm (metal_program.h:137-248)
  - 16-byte mach-o LC_UUID identity from the loaded image, or EINVAL/ENODATA — to unknown (no caller of
    NativeImageIdentity found) (metal_program.h:86-100)
  - Makefile xxd -i byte-array text of both headers into .build/metal_program_source.h — to metal_lower.mm
    includes metal_program_source.h (Makefile:109-112; metal_lower.mm:28)
- **Role:** More than one: (1) runtime (stage 6) for NativeExecution and the Native* encode and submit helpers
  (metal_program.h:107-248) and for the MetalSubmission transition/install pair (metal_submission.h:14-116); (2)
  lower (stage 5) declarations metal_ir_native_compile, entry (metal_program.h:45, 49); (3) archive (stage 7)
  declarations metal_ir_native_save and load (metal_program.h:46-47); (4) reading back emitted binary identity via
  LC_UUID (metal_program.h:86-100), which is not a stage role; (5) derivation of synchronization values (wait,
  signal, slot) at call time from config and state (metal_submission.h:24-27, 37, 43, 48-49, 53, 57-58), which the
  reference order assigns to schedule (stage 4) and to runtime with derives-nothing (stage 6).
- **Direction:** does not match.
- **Where, as the reader states it:**
  - (c) metal_program.h:86-99: NativeImageIdentity opens the loaded mach-o image of the compiled entry (dladdr at
    l.88, header at l.89, scan of load commands l.91-98) and returns the LC_UUID bytes at l.93-95, which recovers
    an identity from the emitted binary rather than receiving it.
  - (c) metal_program.h:40, 51: metal_ir_native_sites returns const NativePC debug-site metadata from the compiled
    object; the consumer metal_lower.mm:1557-1559 and 1642-1644 reads NativePC records from a debug metadata file
    to recover site information.
  - (d) metal_program.h:45-51: one header declares lower (compile, entry), archive (save, load, free) and
    identity/sites accessors for the same NativeCompiled handle, so compile, archive and runtime entry are in one
    boundary.
  - (d) metal_program.h:107-248: NativeExecution plus submission and encode helpers (runtime role) sit in the same
    header as the lower and archive declarations at l.45-51.
  - (b) metal_submission.h:24-27, 43, 48-49: the transition computes wait values, completedCommits, and ring slot
    indices at execution time from config.lag, positions, window and state; the reference assigns storage
    decisions and the ordered schedule to stage 4 (schedule) and says runtime derives nothing (stage 6).
  - (b) metal_program.h:66-70: NativeNeural carries a predict function pointer supplied from outside the group,
    and the supplier role for numeric functions is separate from this boundary; the group holds the callback in
    its runtime struct.
  - (a) none found: metal_program.h:27-37 takes NativeProgram, which is scheduled-program shaped (sites, controls,
    bindings, submission config), and metal_submission.h takes only config, state, action and value; no input is a
    product of a later stage in these two files.
- **Not determined:**
  - Definitions of metal_ir_native_compile, load, save, free, entry, identity and sites; none found in the
    repository by grep, so the implementing file and the stage they perform are unknown.
  - Callers of NativeProgram construction and of metal_ir_native_* outside the four files listed in callers; a
    whole-repo grep for the metal_ir_native_* names found only the declarations.
  - Callers of NativeEncoderBegin, NativeEncoderEnd, NativeSubmit, NativeProgressSignal, NativeProgressWait,
    NativeDrain, NativePredicateRead, NativeChecked, NativeICBEncode, NativeAddressEncode, NativeIndirectEncode,
    NativeCopyEncode, NativeFillEncode and NativeImageIdentity; grep of metal_recording.m, metal_lower.mm and
    metal_native.mm found no match for these names, and the whole repo was not searched for them.
  - Whether the NativeCompiled load path solves anything (the archive direction at stage 7), since the load
    definition is not in the repository.
  - Source of the NativeNeural predict callback and of NativeExecution.commit_options and start/started function
    pointers (supplier identity).
  - Whether MetalSubmissionConfig values in NativeProgram.submission are produced by a schedule stage; the
    producer of NativeProgram is not located.

## provider: litert_mesh and litert.patch

- **Files:** `provider/litert_mesh.h` (22 lines), `provider/litert_mesh.cc` (4,984 lines), `provider/litert.patch` (404 lines), `provider/BUILD.bazel` (63 lines), `provider/exports.lds` (4 lines).
- **Externally visible functions:** 2.
- **Called from:**
  - tools/e2b_replay.m (line 3 #include "litert_mesh.h"; lines 495-496 call LiteRtMeshDeclare and
    LiteRtMeshPublish from provide(); line 485 comment names both)
  - Makefile (lines 98-101: builds @mesh_provider//:libmesh_provider.dylib from provider/litert_mesh.cc,
    litert_mesh.h, exports.lds; line 90-95 applies provider/litert.patch; line 152 links .build/e2b_replay against
    libmesh_provider.dylib)
  - docs/provider.md (lines 45-56 describe the two exported functions; lines 22-38 describe the patch, BUILD,
    exports.lds)
  - /Users/mdot/dox/mesh/rdma: no reference to LiteRtMesh* or libmesh_provider found by grep
- **Takes:**
  - LiteRtCompiledModel compiled (LiteRT C API handle) — from caller, tools/e2b_replay.m:495-496 via
    litert_mesh.h:13-18; resolved to tflite::Interpreter by GetInterpreter(compiled)
    (provider/litert_mesh.cc:4864)
  - size_t signature index, into the interpreter's signature_keys — from caller tools/e2b_replay.m:495 (passes 0)
    (provider/litert_mesh.cc:4865-4867)
  - struct metal_ir_program *program (the Metal IR program handle into which the group emits) — from caller
    (tools/e2b_replay.m:495-496); declared in metal_ir.h (provider/litert_mesh.h:11, provider/litert_mesh.cc:4861)
  - uint32_t *values, one per subgraph input then output; UINT32_MAX means create, otherwise an existing value id
    — from caller tools/e2b_replay.m:491-496 and the value array it writes back at tools/e2b_replay.m:497-498
    (provider/litert_mesh.cc:4874-4877, 4884-4886)
  - const uint32_t *roles (metal_ir role per entry, optional) — from caller tools/e2b_replay.m:493 (role callback)
    (provider/litert_mesh.cc:4903)
  - size_t count, must equal inputs+outputs of the subgraph — from caller tools/e2b_replay.m:492
    (provider/litert_mesh.cc:4871-4873)
  - void *owner, void (*retain)(void *), void (*release)(void *) (lifetime callbacks for model-owned constant
    bytes; Publish only) — from caller tools/e2b_replay.m:496 (call->owner, model_retain, model_release)
    (provider/litert_mesh.cc:4921-4922, 4925)
  - TFLite subgraph: its inputs(), outputs(), execution_plan(), tensors and node_and_registration (the operations
    to lower, in plan order) — from TFLite runtime via tflite/core/subgraph.h and interpreter.h
    (litert_mesh.cc:44-46) (provider/litert_mesh.cc:4870-4878)
  - delegated partitions: DelegateKernel* per node whose custom_name is LITERT_WEBGPU, with mesh_graph()
    (md::GraphFloat32), mesh_ir_model() (ml_drift::ir::IrModel), mesh_context(), delegate_data(),
    quant_conversion_map(), GetInputIndices/GetOutputIndices — from ml_drift delegate, as modified by litert.patch
    (retained in DelegateKernel at litert.patch:174 and litert.patch:292; accessors at litert.patch:306-321)
    (provider/litert_mesh.cc:4878, 4955-4966, 4968-4974 (provider/litert_mesh.cc:4965 reads kernel->mesh_context))
  - TFLite node and registration per non-delegated node (builtin or custom), lowered by class Builtin — from
    TFLite subgraph (see above) (provider/litert_mesh.cc:4944-4954)
  - TfLiteContext tensors: dims, type, allocation_type, data.raw_const, bytes, quantization params — from TFLite
    runtime (provider/litert_mesh.cc:4898-4903, 4936-4937, 330-333 (via Bytes/Tensor), 390-395 (Quantize))
  - ml_drift types: md::GraphFloat32, md::Node, md::Value, md::ir::IrModel, md::TensorOrScalar, md::OperationType,
    md::DataType, md::BHWC — from ml_drift library (litert_mesh.cc:26-30) (provider/litert_mesh.cc:1012, 1074,
    1149-1156, 1194-1201, 3972-3999)
- **Returns:**
  - int status: 0 on success, or errno-style code (EINVAL, ENOTSUP, ENOMEM) with message written into char *error
    of length room — to caller tools/e2b_replay.m:495-496 (status, and failed/room); caller handles status at
    tools/e2b_replay.m:497 onward (provider/litert_mesh.cc:4910, 4918, 4972-4983;
    provider/litert_mesh.cc:4838-4841 (Errno))
  - side effect: values[k] written with the metal_ir value id for each subgraph input/output (created via
    metal_ir_value or reused) — to caller tools/e2b_replay.m:497-498 (writes back backing->ir[phase]=values[k] and
    metal_ir_value_view on mismatch) (provider/litert_mesh.cc:4905, 4884-4886)
  - side effect: Metal IR program populated with values (metal_ir_value, metal_ir_value_data,
    metal_ir_value_view), regions, operations (metal_ir_region, metal_ir_yield, metal_ir_scatter,
    metal_ir_operations) for every delegated graph and every lowered TFLite node — to the program object owned by
    the caller; program release callback is registered via Owner/Holder (provider/litert_mesh.cc:330-333,
    4954-4958, 4965-4966) (provider/litert_mesh.cc:330, 333, 4939, 4944-4966)
  - side effect: retain/release calls on owner (model_retain/model_release) per constant borrowed from the model —
    to caller callbacks passed at litert_mesh.h:17-18 (provider/litert_mesh.cc:326-334)
  - error text (char *error) when a node has no lowering: operation names list — to caller
    tools/e2b_replay.m:495-496 via failed buffer (provider/litert_mesh.cc:4968-4973, 4977-4981)
- **Role:** more than one: lower (Metal IR emission from ml_drift graphs and TFLite nodes,
  provider/litert_mesh.cc:4914-4984, 1149-3157, 3163-4836); bind of program values to TFLite tensor addresses
  (provider/litert_mesh.cc:4861-4910, 4939, uintptr identities at 4905, 4939); export-like model conversion from
  TFLite nodes to md::GraphFloat32 via BuildFinalModel (provider/litert_mesh.cc:3972-4010); supplier of numeric
  functions as Metal IR bodies (Tanh, Erf, Unary, Binary at provider/litert_mesh.cc:996-1133); storage decisions
  (dense row-major layout and TEMPORARY values at provider/litert_mesh.cc:4896-4905, 4937-4941, 477); placement-
  related partition setting in the patched delegate (provider/litert.patch:368, 393); changes to an external
  dependency (provider/litert.patch, all hunks)
- **Direction:** does not match.
- **Where, as the reader states it:**
  - (c) reads back emitted output: provider/litert_mesh.cc:279-282 Lowering constructor calls metal_ir_values and
    metal_ir_value_info on the program to recover constant identities already emitted;
    provider/litert_mesh.cc:356-364 Lowering::View calls metal_ir_value_info(p_, of) on an existing emitted value
    to recover its role, scalar and identity.
  - (c) reads back emitted output: provider/litert_mesh.cc:354-358 Know calls metal_ir_value_info to recover
    scalar of an emitted value.
  - (d) performs several stage roles: provider/litert_mesh.cc:4864-4870 Bind calls GetInterpreter and
    subgraph->AllocateTensors(), which is runtime allocation, inside the lowering entry point.
  - (d) export-like model conversion inside lowering: provider/litert_mesh.cc:3991-4010 Builtin::Parsed calls
    litert::ml_drift::BuildFinalModel(tfl_, ...) to build a md::GraphFloat32 from TFLite nodes and then lowers it.
  - (d) storage decisions inside lowering: provider/litert_mesh.cc:4896-4905 Bind derives dense row-major layout
    strides and assigns metal_ir_term layouts from TFLite tensor dims; provider/litert_mesh.cc:4937-4941 Publish
    creates METAL_IR_TEMPORARY values for unbound TFLite tensors; provider/litert_mesh.cc:314-315 Temporary
    creates TEMPORARY values.
  - (b) produces what it should receive from a supplier or earlier stage: provider/litert_mesh.cc:4937-4941 and
    314-315 create storage values itself rather than receiving them; provider/litert_mesh.cc:4861-4862 accepts
    TFLite execution order from subgraph->execution_plan() (provider/litert_mesh.cc:4871-4878) rather than a cost-
    ordered schedule.
  - (a) takes what a later stage produces: none found. Inputs are a compiled model, a caller-created program
    handle, and caller-supplied value and role arrays (provider/litert_mesh.h:13-18;
    provider/litert_mesh.cc:4861-4866).
  - supplier role inside lowering: provider/litert_mesh.cc:996-1010 Tanh and Erf and
    provider/litert_mesh.cc:1012-1072 Unary and provider/litert_mesh.cc:1074-1133 Binary emit numeric function
    bodies into Metal IR regions.
  - (d) placement role inside the patched delegate, outside the group's exported functions:
    provider/litert.patch:368 and provider/litert.patch:393 set max_delegated_partitions to
    std::numeric_limits<int>::max().
- **Not determined:**
  - The meaning and contract of metal_ir.h entry points (metal_ir_value, metal_ir_region, metal_ir_value_info) is
    defined in metal_ir.h, which was not read; their semantics are inferred from call sites only.
  - The body and behavior of litert::ml_drift::BuildFinalModel, GetInterpreter, and CustomOperationParserFactory
    (external ml_drift and LiteRT sources not in the repo) was not read.
  - Whether LiteRtMeshPublish's Lowering::Finish and the Graph/Builtin per-op bodies
    (provider/litert_mesh.cc:1149-4836, about 3700 lines) perform any further reads of emitted output beyond the
    call sites listed above was determined only by grep for metal_ir_value_info, metal_ir_value_view and
    metal_ir_values; the bodies were not read in full.
  - Whether the composite parsers in provider/litert_mesh.cc:31-41 and the BUILD dependencies
    (provider/BUILD.bazel:8-33) contribute placement or scheduling decisions was not determined.
  - Whether the caller (tools/e2b_replay.m) computes roles and values for each phase in a way that depends on the
    group's output beyond the value write-back (tools/e2b_replay.m:485-498) was not determined beyond that block.
  - Whether the program's later release or runtime call path (metal_ir_program_free, metal_ir_value_data owner
    release) binds addresses at runtime was not determined; only the owner/retain pairing at
    provider/litert_mesh.cc:326-334 was seen.

## provider/dawn.patch

- **Files:** `provider/dawn.patch` (20,081 lines).
- **Externally visible functions:** 49.
- **Called from:**
  - metal_lower.mm (Generate at lines 1541 and 1693; host_cpp set at 1690; Fence creation at 1063 and 1066; module
    properties at 384-386)
  - metal_native.mm (NativeMatmul creation at 1160; Algorithm chosen at 287 and 294)
  - metal_ir.mm (includes core/type/f64.h, i64.h and msl/ir/fence.h; ComputeType at 86-92; Fence counting at 1458)
  - Makefile (lines 77-82: git apply of provider/dawn.patch before the CMake build)
  - docs/provider.md (lines 29-31: describes the patch)
- **Takes:**
  - core::ir::Module with properties kAllow64BitIntegers and kAllow64BitFloats — from metal_lower.mm:384-386
    (provider/dawn.patch:14720 (property added); provider/dawn.patch:14741-14768 (structural validator checks F64
    against it))
  - msl::writer::Options: entry_point_name, bindings map from binding slot to BindingPoint, host_cpp bool,
    disable_* flags — from metal_lower.mm:1529-1541 (MSL path) and metal_lower.mm:1685-1693 (host C++ path)
    (provider/dawn.patch:19597-19625 (host_cpp field at 19608); metal_lower.mm:1690 (options.host_cpp = true))
  - msl::ir instructions Fence, CoherentLoad, CoherentStore, NativeMatmul (Algorithm, Layouts, operands a, b, c,
    workspace, status) built by lower and native code — from metal_lower.mm:1063, 1066 (Fence);
    metal_native.mm:1160 (NativeMatmul); metal_native.mm:287, 294 (Algorithm) (provider/dawn.patch:19507-19568
    (NativeMatmul class with Method() and Matrices()))
- **Returns:**
  - MSL source text and host C++ source text (Generate result msl field) and workgroup_allocations — to
    metal_lower.mm:1541-1546 (MSL path, kernel source and allocation check) and metal_lower.mm:1693-1699 (host C++
    path) (metal_lower.mm:1545-1546 (allocations compared to k.allocations); metal_lower.mm:1699 (loop over
    generated->workgroup_allocations))
  - Core IR type and constant overloads for i64 and f64 (BuildI64, BuildF64, i64(), f64(), Unsigned) — to Tint
    intrinsic table and type manager (inside Dawn, consumed by the patched Tint core)
    (provider/dawn.patch:14649-14661, 14879, 15288-15291)
- **Role:** more than one: lower (stage 5, msl printer emits MSL and host C++ text from the scheduled IR); runtime (stage 6
  role inside the emitter, EmitHostTrampoline binds binding pointers by slot); supplier of numeric functions (i64
  and f64 intrinsics, polyfill transforms); shared types and validation (core type, constant manager, structural
  validator, which are checks and not a stage)
- **Direction:** does not match.
- **Where, as the reader states it:**
  - provider/dawn.patch:19946-19949: EmitNativeMatmul derives the BLAS transpose flags (row, direct) and leading
    dimensions (leading) from the NativeMatmul Layouts at emit time, a derivation step in the emitter rather than
    a value received from an earlier stage.
  - provider/dawn.patch:19818-19830: EmitHostTrampoline emits void name_host(void* const* tint_bindings, uint3
    tint_group, uint3 tint_groups, void* tint_memory) and the calling sequence that passes each binding pointer to
    a parameter, which is a runtime (stage 6) role performed inside the lower-stage printer; the slot map it
    relies on is supplied by metal_lower.mm:1690-1692.
  - metal_lower.mm:1545-1546 and metal_lower.mm:1699: the lower stage reads back generated->workgroup_allocations
    from the emitter result and compares it with k.allocations that Lower published (metal_lower.mm:203); this is
    a read-back of an emitter-derived fact, not source text. The struct field is defined upstream and is not in
    the patch.
  - provider/dawn.patch:20067-20081: the Raise step of the msl writer skips SingleEntryPoint when host_cpp is set,
    so the emitter pipeline itself selects which entry points remain.
  - provider/dawn.patch:14741-14768: the core IR structural validator is a check stage, and it validates F64 and
    I64 against Property::kAllow64BitFloats, which the lower stage sets at metal_lower.mm:386.
- **Not determined:**
  - Bodies of the intrinsic data changes (provider/dawn.patch:216-14632, the msl intrinsic data at 348-19170), the
    polyfill transforms (binary_polyfill.cc at 14769, conversion_polyfill.cc at 14802, signed_integer_polyfill.cc
    at 14870) and the functional validator (14685) were not read in full; their roles are inferred from file names
    and diff headers.
  - The upstream Tint Generate Output struct (workgroup_allocations field and msg source) is not in the patch, so
    its computation was not verified.
  - Whether metal_native.mm:287 and 294 choose the Algorithm from a cost operand was not verified; the code shown
    uses a dense flag only.
  - Whether the printer's uses_internals_ flag (provider/dawn.patch:19726) is read by any caller outside the
    printer was not checked.
  - The contents of provider/dawn.patch were read by hunk header and selected hunks only, not by full line count
    per file.

## metal_binary64.wgsl and operator_shaders.metal

- **Files:** `metal_binary64.wgsl` (1,726 lines), `operator_shaders.metal` (245 lines).
- **Externally visible functions:** 139.
- **Called from:**
  - metal_ir.mm (#embed at line 2111; Binary64Library and Binary64 at 2266-2284; library.Find at 2373 and 2437)
  - metal_lower.mm (include of operator_shaders_metal.h at line 25; Binary64 call at line 1527; write of
    metal_stdlib at line 1575; xcrun clang++ at lines 1596-1597)
  - Makefile (lines 105-107 xxd rule for operator_shaders.metal; line 114 metal_binary64.wgsl as a
    libmetal_ir.dylib prerequisite)
  - metal_ir.h (declaration comment at lines 854-856; documentation only)
- **Takes:**
  - WGSL source text of metal_binary64.wgsl embedded as a byte array at build time (#embed), then parsed by Tint
    WgslToIR into an IR module (group 'library') — from build-time file metal_binary64.wgsl via metal_ir.mm
    (metal_ir.mm:2111 (#embed "metal_binary64.wgsl"); metal_ir.mm:2269-2272 (tint::Source::File,
    wgsl::reader::WgslToIR))
  - Caller-supplied operand values of type vec2<u32> (binary64 bit pair: low word, high word), plus u32, i32, f32,
    f16, bool scalars, at each library function signature; the caller's IR arguments are spliced in — from the
    caller's IR instruction operands, passed through metal_ir.mm Splice into the library functions
    (metal_binary64.wgsl:78 (u64_add(a: vec2<u32>, b: vec2<u32>)); metal_binary64.wgsl:672 (f64_from_u32_impl(a:
    u32, sign: u32)); metal_binary64.wgsl:711 (f64_from_f32_impl(f: f32)); metal_ir.mm:2373-2380
    (parsed.Find(call.entry); Splice(library, f, b, args, ...)))
  - The f64 operation to replace: a Kernel's IR module (CoreBinary/CoreUnary/Convert/CoreBuiltinCall on f64) is
    the input the splicing consumes, and it selects the library entry by name — from the kernel's IR module handed
    to Binary64(Kernel&) from the lowering path (metal_ir.mm:2284 (std::string Binary64(Kernel &k));
    metal_ir.mm:2320-2345 (per-instruction entry selection, e.g. f64_to_<kind>, f64_from_<kind>,
    Entry(binary->Op())))
  - Compile-time C++ inputs: the standard headers atomic, bit, cmath, cstddef, cstdint, type_traits and the
    toolchain's __bf16/_Float16 types and clang builtins — from the C++ standard library and clang toolchain
    (external) (operator_shaders.metal:7-12 (#include <atomic> ... <type_traits>); operator_shaders.metal:18-19
    (typedef _Float16 half; typedef __bf16 bfloat); operator_shaders.metal:215-216 (__atomic_thread_fence))
  - The shader file's bytes, unchanged, written to a per-task directory as the file name metal_stdlib, and the
    xxd-generated byte array operator_shaders_metal.h included by the lowering source — from
    operator_shaders.metal via Makefile xxd generation, read by metal_lower.mm (Makefile:105-107 (xxd -i
    operator_shaders.metal > .build/operator_shaders_metal.h); metal_lower.mm:25 (#include
    "operator_shaders_metal.h"); metal_lower.mm:1575 (write("metal_stdlib", operator_shaders_metal, ...)))
- **Returns:**
  - IR functions spliced by name into the caller's kernel IR module (each cloned once); the splice replaces each
    f64 instruction with a call to a library function, and the operation returns an error string or the empty
    string — to metal_ir.mm Binary64(Kernel&) and its caller Source() in metal_lower.mm (metal_ir.mm:2373-2381
    (call.inst->Result()->ReplaceAllUsesWith(result); Destroy); metal_ir.mm:2437-2438 (library.Find(entry) ->
    AddWork); metal_lower.mm:1527-1528 (if (std::string missing = Binary64(k); !missing.empty()) return
    LowerFail))
  - The name set of 139 top-level fn definitions (f64_* helpers and the SoftFloat/fdlibm routines), looked up by
    name, no entry points (no @compute, @group or @binding attributes in the file) — to metal_ir.mm Find(name)
    lookups in the Binary64 splice and the work queue (metal_binary64.wgsl:78-1726 (139 lines matching ^fn ); grep
    of metal_binary64.wgsl for '@' returns 0 matches; metal_ir.mm:2260-2262 (Find by name))
  - C++ declarations in the metal namespace (types, atomics, math using-declarations, select/all/any,
    scalar_min/max/abs, mem_flags, memory_order, thread_scope) that the emitted kernel source relies on through
    <metal_stdlib>; no values are returned to the pipeline — to the external clang++ compile of the emitted
    kernel.cpp (operator_shaders.metal:13 (namespace metal {); operator_shaders.metal:116-128
    (scalar_min/max/abs); operator_shaders.metal:205-212 (mem_flags, memory_order, thread_scope);
    metal_lower.mm:1596-1597 (xcrun clang++ -std=c++20 ... -isystem dir ... kernel.cpp))
- **Role:** supplies numeric functions (metal_binary64.wgsl: a SoftFloat binary64 function library spliced in place of f64
  instructions; operator_shaders.metal: the metal standard-library stand-in that the external clang++ compile of
  emitted MSL includes as <metal_stdlib>). Neither file is one of the seven stages. Not a change to the external
  compiler itself.
- **Direction:** matches.
- **Not determined:**
  - operator_shaders.metal lines 32-99 and 133-200 were not read (only grep-selected declarations and the ranges
    1-20, 100-130, 200-245 were seen), so the full set of metal:: math macros and the using-declarations there is
    not enumerated.
  - WGSL function bodies were not read in full; the boundary types are taken from signatures only
    (metal_binary64.wgsl:78-1726).
  - Whether the emitted MSL kernel actually calls each metal:: helper in operator_shaders.metal was not
    determined; only the include path and compile flags were checked.
  - metal_ir.mm Splice() and Entry() implementations (outside this group) were not read; the claim that splicing
    is in the lowering path rests on metal_lower.mm:1527 calling Binary64(k) before tint::msl::writer::Options at
    metal_lower.mm:1528.
  - No read-back of emitted MSL source, compiled binaries or debug information by either file was found, but the
    rest of the repository was searched only for the two filenames and the Binary64 symbol, not for every consumer
    of the library's Tint IR.
  - The rdma path /Users/mdot/dox/mesh/rdma was searched for references to both file names with no matches; the
    design document cited at metal_binary64.wgsl:1 and metal_lower.mm reference was not opened.
