# Change requests: which are outstanding

Every line below is a reading of the code as committed at mesh 434b9955 and metal-microbench 0b692ed8, made
on 2026-11-12, one reader a change request, each clause read sentence by sentence
([review-warnings.md](review-warnings.md), rule 6). The readings, with the lines of code each sentence rests on, are in
metal-microbench `output_data/change-request-audit-20261112/reread.json`; the reading it replaces, made at mesh
9aa74ed1, is `audit.json` beside it, and this index as that reading left it is at mesh 141fef7b. The standard every
change request follows is [change-requests.md](change-requests.md).

- The operator, 2026-11-12: "remove all undated or unversioned 'claims to facts' from any documents in any 'index' as
  there can clearly never be a coherent store of 'what the repo is' without versioning, dating, or namespacing of
  factual claims."
- The operator, 2026-11-12: "if they weren't finished, they still need to be finished beacuse they are not optional
  'would be nice to haves'. if any are finished, mark them as 'covered [datestamp,commitstamp]'."
- `covered [date, repository commit]`: every sentence of the item's clause was read as holding in code that has the
  data flow of [data-flow.md](data-flow.md); the commit is the one the reader found the last of it landed in, with
  that commit's date and repository.
- `outstanding`: at least one sentence does not hold there. The entry quotes each such sentence and says what the
  code does. A clause that holds only in the files [the-compiler.md](the-compiler.md) order 5 names, or in code the
  `compile_module` chain does not reach, is outstanding. Agreement of a check with a reference run makes nothing
  covered.
- `not determined [date]`: the clause states a condition of machines, services or records that no file shows; the
  item is outstanding until a reading shows it holds.
- A second reader then read each covered stamp for a sentence that does not hold, at mesh 6d379b7e; the entries it
  moved say so, and its readings are `refute.json` beside the others.
- [requirements-by-stage.md](requirements-by-stage.md) lists the same items under the stage that takes each, with
  each clause quoted.
- Of 611 items: 200 covered, 396 outstanding, 10 not determined,
  5 withdrawn. No request has every item covered. All eleven are outstanding.

## [compilation-composition-change-request.md](compilation-composition-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 82 items outstanding, 0 not determined, 10 covered, 1 withdrawn.

Outstanding:

  - The data flow (stage table and its closing sentence): "There is one schedule.": compile_module solves mesh_plan
    twice, driver.py:24 (planned -> solved, schedule.py:160 -> mesh_plan at schedule.py:111) and driver.py:26 (stored
    -> solved, schedule.py:196); the functions are ordered by a separate call, mesh_graph.heap_order at
    schedule.py:208, from the first solve's times (schedule.py:206), and the plan kept in Scheduled is the second
    solve's (driver.py:26-29).
  - 1 / data structure and Composes (export): "export produces both graphs": export.py:74 returns one Exported holding
    one graph_module (ir.py:45); no value of Exported or Scheduled is a graph of calls between compiled functions.
    "its collectives (the `send` and `recv` nodes placement added)": lower_metal keeps them as program entries that
    call the torch custom ops at run time (lower_metal.py:824-833, 901, 903; ir.py:12-24), lower_coreml the same
    (lower_coreml.py:269-272). "An intermediate conversion is completed before the program runs.":
    lower_inductor.py:72 returns torch.compile's callable, which converts at its first call inside Program
    (runtime.py:50, 72). "A missing lowering within the assigned domain is unfinished work, not a new restriction on
    that domain.": operators outside the handler table are reported missing (lower_metal.py:306-319, 332-336), float64
    has no Metal type (lower_metal.py:26-27), and export.py:32-35 raises "is out of scope; state must be a
    placeholder".
  - 1 / algorithm (full conversion): "every operation of the input is converted to the target, and the conversion
    succeeds only when all of them are": send and recv nodes are not converted, the returned function calls the torch
    custom ops SEND and RECV (lower_metal.py:901, 903; lower_coreml.py:269-272) and TO_COPY eagerly
    (lower_coreml.py:273-274); lower_metal returns an engine processor's module unconverted (lower_metal.py:1196-1198)
    and driver.lowered returns every module no lowering converted beside the converted ones (driver.py:32-40), which
    Program then runs as an FX GraphModule (runtime.py:15).
  - 1 / Write: every rank, its collectives, each placed CPU or GPU function: "export produces both graphs": one
    graph_module (export.py:74, ir.py:45). "its collectives (the `send` and `recv` nodes placement added)" / "Lowering
    covers every rank and its collectives": the lowered function calls the unlowered torch ops SEND and RECV
    (lower_metal.py:901, 903; ir.py:12-24). "An intermediate conversion is completed before the program runs.":
    lower_inductor.py:72 (torch.compile converts at first call). "A missing lowering ... is unfinished work":
    lower_metal.py:306-319, 332-336, 1204-1206; export.py:32-35.
  - 1 / Write: conversion completed before the program runs: "An intermediate conversion is completed before the
    program runs.": lower_inductor.lower returns torch.compile(adapted, backend="inductor") without calling it
    (lower_inductor.py:67-73); the conversion runs at the function's first call, which Program makes (runtime.py:50,
    72).
  - 1 / Write: the whole assigned domain: "A missing lowering within the assigned domain is unfinished work, not a new
    restriction on that domain.": lowerings are missing for every operator outside lower_metal.py:306-319 (POINTWISE
    :81-103 has ge and le for Scalar only) and for float64 on Metal (TYPES, lower_metal.py:26-27; untyped :347-350);
    the code states restrictions as scope: export.py:32-35 "get_attr ...: a graph that reads an attribute of its
    module is out of scope", lower_inductor.py:48-49, placement.py:666-667.
  - 2 / data structure: send and recv nodes in the one graph: "its edges are values, each with its sharding, consumer
    kind, accesses and lifetimes": no edge carries accesses (they are first made in lowering, lower_metal.py:224-225)
    and lifetimes are in Plan.values for one machine's values only (schedule.py:179-191). "One schedule orders
    functions and transfers together": functions by mesh_graph.heap_order (schedule.py:208), transfers by mesh_plan,
    solved twice (driver.py:24, 26; schedule.py:111). "The signature is printed from what lowering emits.":
    signature.py:100-101 uses scheduled.modules[processor] for every function without a `record` (lower_coreml's,
    lower_inductor's, unlowered).
  - 2 / data structure: what each edge carries: "its edges are values, each with its sharding, consumer kind, accesses
    and lifetimes": accesses are not in the scheduled value (lower_metal.py:224-225 makes them), lifetimes only in
    Plan.values (schedule.py:179-191). "One schedule orders functions and transfers together": schedule.py:208 and
    schedule.py:111, the latter twice (driver.py:24, 26). "The output of each analysis or lowering is the operand its
    consumers actually use.": Plan.values/places/sections from stored (schedule.py:165-196) have no reader
    (lower_metal.py:1188-1207 never reads scheduled.plan; runtime.py:25, 48, 71 assign ir.LOCAL.storage, which nothing
    reads; bridge.py:51-53 and :64-65 have no caller). "The signature describes that emitted program.":
    signature.py:100-101. "Each row's work is completed in this composition, including its specified deletion or
    replacement.": rdma/mesh_compile.py (program() :8242), rdma/allocate.py, rdma/mesh_code.py, rdma/mesh-torch.mm and
    metal-microbench metal_recording.m, tools/e2b_replay.m are at HEAD.
  - 2 / Composes: one schedule orders functions and transfers together: "One schedule orders functions and transfers
    together and produces the scheduled value.": functions are ordered by mesh_graph.heap_order (schedule.py:201-214),
    transfers by mesh_plan (schedule.py:107-121), which compile_module calls twice (driver.py:24 and :26); the
    function order uses the first solve's times (schedule.py:206) and Scheduled keeps the second solve's plan.
  - 2 / the signature: "The signature is printed from what lowering emits." / "The signature describes that emitted
    program.": for lower_coreml's and lower_inductor's functions, which have no `record`, signature.py:100-101 runs
    torch_run on scheduled.modules[processor]; the thread handoffs and bridge operations are composed in
    signature.py:61-87 and :148-151 from the scheduled value. "One schedule orders functions and transfers together":
    schedule.py:208; schedule.py:111 called twice (driver.py:24, 26). "The output of each analysis or lowering is the
    operand its consumers actually use.": Plan storage results unread (schedule.py:165-196; runtime.py:25, 48, 71;
    bridge.py:51-53). "Each row's work is completed in this composition": rdma/mesh_compile.py:8242 and
    metal-microbench metal_recording.m at HEAD.
  - 2 / Write: transformations keep information; outputs are operands; changes reflected: "The output of each analysis
    or lowering is the operand its consumers actually use.": stored writes each value's lifetime and place into the
    Plan (schedule.py:165-196) and nothing reads them (lower_metal.py:1188-1207 reads modules, devices, engines and
    carried; runtime.py:25, 48, 71 set ir.LOCAL.storage with no reader; Published.address bridge.py:51-53 and
    Carried.out bridge.py:64-65 have no caller); Plan.dependences (schedule.py:159) is read by no lowering, which
    derives its own relation (lower_metal.py:844). "Changes to the graph are reflected in those dependent results
    before emission.": the function order is fixed from the first solve (schedule.py:206, driver.py:25) and is not
    recomputed after the second solve changes the plan's values (driver.py:26).
  - 2 / Write: each row's work completed with its deletion: "Each row's work is completed in this composition,
    including its specified deletion or replacement.": the code the rows replace is at HEAD in both repositories: mesh
    rdma/mesh_compile.py (program() :8242, _replayed :7660, MetalRecorderProgram :7557), rdma/allocate.py,
    rdma/mesh_code.py, rdma/mesh-torch.mm; metal-microbench metal_recording.m (MetalRecorderProgram :2478,
    MetalProgramRun :2648), metal_transform.mm, tools/e2b_replay.m.
  - 3 / Write, second bullet: callers call them, own copies deleted: "the one schedule" / "the second scheduling
    solve": mesh_plan is solved at driver.py:24 and again at driver.py:26 (schedule.py:160, 196). "placement of
    program values at run time": result buffers are allocated at each call (lower_metal.py:1019-1026, 1149;
    lower_coreml.py:277-278). "alternative execution paths": rdma/mesh_compile.py program() (:8242) and
    metal-microbench metal_recording.m MetalRecorderProgram (:2478) are at HEAD beside the chain.
  - 3 / Write, second bullet: the four named deletions: "the second scheduling solve": driver.py:24 (planned) and
    driver.py:26 (stored) each call solved -> mesh_plan (schedule.py:160, 196, 111). "placement of program values at
    run time": lower_metal.py:1019-1026 allocates each result buffer at every call, :1029-1040 chooses each argument's
    handle at run time, Native :1149, lower_coreml.py:277-278. "alternative execution paths": rdma/mesh_compile.py
    program() (:8242), metal-microbench metal_recording.m (:2478, :2648) and tools/e2b_replay.m (:1517) at HEAD.
  - 4 / Write: placement and costs are operands: "placement and costs arrive as operands": no caller in examples/ or
    compiler/samples.py passes `cost`; every compile takes the default unit_cost, which returns 1.0 (driver.py:14;
    schedule.py:26-27). "compilation realizes the graph's storage": the plan's storage (schedule.py:165-196) is read
    by nothing; lower_metal allocates its own (lower_metal.py:984-985, 1019-1026, 1116-1117, 1149).
  - 4 / Write: compilation realizes transfers, ordering and synchronization: "placement and costs arrive as operands;
    compilation realizes the graph's storage, transfers, ordering and synchronization": no caller passes a cost
    (schedule.py:26-27 is used by every compile in examples/ and compiler/samples.py); the storage the compilation
    solves (schedule.py:165-196) is not realized: no lowering or runtime code reads Plan.values/places/sections, and
    buffers are allocated in lowering and at each call (lower_metal.py:984-985, 1019-1026, 1149).
  - 4 / Write: compilation realizes storage: "compilation realizes the graph's storage": for every value that is not a
    carried transfer the layout is unread: lower_metal.py:1188-1207 never reads scheduled.plan and gives each held
    value its own buffer (lower_metal.py:358-360, 984-985, 1116-1117); result buffers are allocated at every call
    (lower_metal.py:1019-1026, 1149; lower_coreml.py:277-278); ir.LOCAL.storage (runtime.py:25, 48, 71) and
    Published.address (bridge.py:51-53) have no reader. "costs arrive as operands": only the default unit_cost arrives
    (schedule.py:26-27).
  - 4 / Write: neither execution nor the bridge computes its own order or placement: "Every node's execution and the
    bridge's use the order and placement in the scheduled value; neither computes its own.": the bridge computes, per
    execution, a permutation of each set's positions at run time (rdma/mesh-flow.c:607-620 set_draw); that order is in
    no field of the scheduled value (ir.py:54-83).
  - 4 / Write: signature and external observation cover this execution: "The required signature and its external
    observation cover this execution.": for a processor lowered by lower_coreml or lower_inductor, or not lowered, the
    signature's tasks and operations are computed from scheduled.modules[processor] (signature.py:100-101, 33-44), not
    from the function Program calls (runtime.py:50, 72); Program's thread handoffs and the bridge's operations are
    written again in signature.py:61-87 and :148-151.
  - 4 / Write: performance and numerical comparison: "Performance and numerical comparison follow the operator's
    requirements in metal-microbench `docs/measurement.md`.": docs/measurement.md:13-14 names the readings (per-call
    overhead, bandwidth, E2B decode tokens per second after a prefill); no caller of compile_module reads any of them:
    examples/engines.py:114-118 reports the wall seconds of one call, examples/tp.py:100-127 times FX nodes for the
    shares, and no caller in examples/ or compiler/samples.py decodes E2B.
  - 5 / data structure and Composes: one compiled native program per node: "one compiled native program per node,
    which launches all CPU, ANE and MPS work of the graph": lowering returns one function per processor
    (lower_metal.py:1195-1207; lower_coreml.py:293-306), each a Python closure (lower_metal.py:892-905;
    lower_coreml.py:266-282); a machine's processors are launched by Python threads (runtime.py:41-52, 67-72). "Python
    capture and preparation are not its CPU execution.": each call's control (entry loop, send, recv, buffer
    allocation, address patching) runs in Python (lower_metal.py:892-905, 1019-1050, 1151-1155). "emits the native
    program": the same. "The signature is printed from that program": signature.py:100-101 uses the scheduled module
    where a function has no record.
  - 5 / Composes: signature printed by the graph module's printers: "The signature is printed from that program by the
    graph module's printers.": there is no native program per node to print from (lower_metal.py:892-905 returns a
    Python closure per processor); for lower_coreml's and lower_inductor's functions signature.py:100-101 prints from
    scheduled.modules[processor]; compiled(scheduled, ()) prints with nothing lowered (compiler/samples.py:1157).
  - 5 / Replaces, site 1: "`mesh_compile.py _outputs` and `_natives`: both traverse the same call graph ..." /
    "`program.make` also rebuilds the same bound-input list.": _outputs is at rdma/mesh_compile.py:7612 and is called
    at :8130; program() (:8242) still defines make (:8348).
  - 5 / Replaces, site 2: "`tools/e2b_replay.m` separately orchestrates prepared work." / "delete displaced
    orchestration": metal-microbench tools/e2b_replay.m is at HEAD and orchestrates MetalRecorderProgram and
    MetalProgramRun (tools/e2b_replay.m:1483-1517; Makefile:152-153). "consume it through the shared native program":
    the chain has no shared native program; each processor has a Python closure (lower_metal.py:892-905).
  - 5 / Replaces, site 3: signature schema and printers: "a copied Python structure or second scheduling solve is
    deleted": the second solve is at driver.py:26 (stored -> schedule.py:196 -> mesh_plan, after driver.py:24's); each
    lowering's synchronization is first a Python dict of tuples (lower_metal.py:881-882; signature.py:42-44) that
    signature.py:136-158 translates into mesh-graph's records; rdma/mesh_compile.py:7567-7575 remains as a second
    signature producer.
  - 5 / Write: one definition of each repeated behavior: "each repeated behavior has one definition ... every caller
    calls it; the duplicate bodies and their dead helpers are deleted": two compilers of Torch programs remain,
    rdma/mesh_compile.py program() (:8242) and compiler/mesh_compiler; within the chain a strided value's span is
    computed at schedule.py:102-103, lower_metal.py:354 and lower_metal.py:808; whether a node is a view is decided at
    schedule.py:91-98 (schema alias_info), placement.py:34-35 with :627-629 (a VIEWS set) and lower_metal.py:124-126
    with :686-695 (running the operator on a meta tensor); a channel's ends are collected at schedule.py:46-51 and
    signature.py:18-24; the caller and joined threads are derived at runtime.py:29-30 and again at signature.py:66-68;
    NotLowered is defined at lower_metal.py:231 and lower_coreml.py:36; Published.address and Carried.out
    (bridge.py:51-53, 64-65) have no caller.
  - 5 / Write: emitted program preserves domain, collectives, page placement, lifetimes: "The emitted program
    preserves the assigned numerical domain, collectives, page placement and lifetime semantics.": numerical domain:
    operators outside lower_metal.py:306-319 and float64 on Metal raise NotLowered (lower_metal.py:332-336,
    1204-1206); collectives: send and recv remain calls of the torch custom ops (lower_metal.py:901, 903) and a sent
    value is copied into the transfer's pages on the host (bridge.py:73-76); page placement and lifetimes: lowering
    never reads the plan's places or lifetimes (schedule.py:165-196; lower_metal.py:1188-1207), each held value has
    its own buffer for the function's life (lower_metal.py:358-360, 984-985) and results are allocated per call
    (lower_metal.py:1019-1026).
  - 5 / Write: producer and consumer agree at both ends: "page groups and sent counters, native result bindings, and
    signature identities agree at both ends": page groups and sent counters have no producer or consumer in the
    compile_module chain; they are in rdma/mesh-metal.h:82-90 and rdma/mesh-metal.m:158, 266 (mesh_metal_mask; struct
    mesh_metal_sending .sent), whose only Python caller is rdma/mesh_compile.py; lower_metal calls rdma/mesh-metal4.m
    and publishes by a host call (bridge.py:68-70).
  - 6 / Replaces 2: schedule results are operands of native preparation and of the transport: "The schedule's task
    identities, its dependence and order results, and the transformation that produced each are compilation operands
    of native preparation": lower_metal.lower reads scheduled.modules, devices, engines and carried only
    (lower_metal.py:1188-1207) and never scheduled.plan; Plan.dependences (schedule.py:83-86, 159) has no reader in
    any lowering, which derives a relation of its own from its steps' views (lower_metal.py:839-849); no field records
    the transformation that produced a result (ir.py:54-83).
  - 6 / Replaces 3: "The one graph is remapped through command transformation, gains the newly lowered vertices/edges,
    and is the operand of ordering and emission.": lowering does not remap the schedule's graph; it constructs another
    dependence relation from the views of the steps it emitted (lower_metal.py:839-849) beside the schedule's
    (Plan.dependences, schedule.py:83-86, 159), which no lowering reads (lower_metal.py:1188-1207).
  - 6 / Replaces 4: "Carry the actual access/effect representation into the consumers that need it": the schedule's
    consumers take whole values: its dependence events are per transferred value with a produced/read/consumed kind
    (schedule.py:22-23, 83-86, 155-157), a view is followed to its root and the root's whole extent is the value
    (schedule.py:91-99, 102-103, 174, 189-191), and mesh_plan receives those (schedule.py:108); strided accesses first
    exist in lowering (lower_metal.py:142-151).
  - 6 / Write: connections finished; identity mapping retained: "An unused new field, a new helper without a consumer,
    a second graph beside an ignored first graph, or diagnostic output alone does not finish a connection.":
    Plan.places and Plan.sections and the lifetimes stored writes (schedule.py:116-117, 178-195; ir.py:66-67) have no
    consumer; ir.LOCAL.storage is assigned (runtime.py:25, 48, 71) and never read; Published.address (bridge.py:51-53)
    and Carried.out (bridge.py:64-65) have no caller; the lowering's dependence relation (lower_metal.py:844-849)
    stands beside Plan.dependences (schedule.py:159), which lowering ignores.
  - 6 / Write: what the emitted program preserves: "The emitted program preserves independent calls, interleaved
    computation/transport, exact copy lifetimes, placed CPU/MPS work, the full assigned numerical domain and the
    LiteRT path.": the LiteRT path is not in the chain (export.py:15-16 takes a torch module; E2B runs through
    metal-microbench tools/e2b_replay.m and metal_recording.m); the numerical domain is partial
    (lower_metal.py:306-319, 26-27, 1204-1206); copy lifetimes are not taken from the plan (schedule.py:179-191
    unread; lower_metal.py:984-985 holds every buffer for the function's life). "A total order, waits, whole-buffer
    effects or another runtime do not replace them.": a send copies the whole value on the host after waiting for the
    commits it reads (lower_metal.py:860-865, 903; bridge.py:73-76).
  - Correction: generated target source / Composes: both frontends at export: "the Torch and LiteRT compiler
    interfaces produce the operations and their annotations at export": export.py:15-16 takes a torch module through
    torch.export.export; no file of compiler/mesh_compiler reads a LiteRT model. LiteRT operations are produced by
    metal-microbench provider/litert_mesh.cc into metal_ir, outside the compile_module chain.
  - Correction: generated target source / Composes and Emission boundary: "the Torch and LiteRT compiler interfaces
    produce the operations and their annotations at export": no LiteRT interface in the chain (export.py:15-16). "the
    producing ... interfaces supply the ... alias ... facts required by their consumers": lower_metal re-derives
    whether a node is a view by running the operator on a meta tensor (lower_metal.py:686-695). "A dependence edge of
    the schedule carries the value/effect/transfer relation ... until the lowering that reads it has represented that
    obligation": no lowering reads the schedule's edges (Plan.dependences, schedule.py:83-86;
    lower_metal.py:1188-1207).
  - Correction: generated target source / Write: frontends supply facts; lowerings preserve identities: "the producing
    Torch and LiteRT compiler interfaces supply the operation, binding, type, shape, indexing, effect, alias and
    transfer/generation facts required by their consumers": there is no LiteRT interface in the chain
    (export.py:15-16); alias facts are derived by each consumer: schedule.py:91-98 from the operator schema,
    placement.py:34-35 and :627-629 from a VIEWS set, lower_metal.py:686-695 by running the operator on a meta tensor.
  - Correction: generated target source / Write: an edge carries its cause: "A dependence edge of the schedule carries
    the value/effect/transfer relation that makes it necessary until the lowering that reads it has represented that
    obligation in the target execution.": no lowering reads the schedule's edges: lower_metal.lower reads
    scheduled.modules, devices, engines and carried (lower_metal.py:1188-1207), lower_coreml and lower_inductor the
    modules; the causes lowering records are on its own relation (lower_metal.py:209-212, 849).
  - Correction: generated target source / Property preserved: "the emitted program has the numerical, fusion,
    communication and synchronization behavior required by the cited requests": the chain emits no program for
    LiteRT/E2B (export.py:15-16) and raises NotLowered outside its handler table (lower_metal.py:306-319, 1204-1206);
    that program is still emitted by metal-microbench metal_transform.mm, metal_recording.m and tools/e2b_replay.m.
  - Performance correction / Emission: "The data structure is Tint's IR module.": the chain has no Tint module; a
    kernel is MSL or C++ text assembled by string formatting (lower_metal.py:927-933, 1076-1084). "each of these is a
    function from operations to operations inside lowering ...; `Print` comes after all of them": fusion, constants
    and reductions are not functions from operations to operations; each handler returns a View or a closure that
    returns target text (lower_metal.py:155-176, 441-457, 730-749) in one pass over the nodes
    (lower_metal.py:321-345). Reductions are a serial loop per output element (lower_metal.py:744-747), not subgroup
    reductions and workgroup exchanges.
  - Performance correction / Fusion and storage: "composes operand indexing maps and remaps operation
    operands/results": index maps are composed as text expressions (lower_metal.py:247-251, 679-683). "uses tensor
    use-def, alias and equivalence relations to determine storage (S8)": every materialized value gets a new buffer
    (lower_metal.py:358-360, 377) held for the function's life (lower_metal.py:984-985, 1116-1117); the lifetimes the
    schedule computed (schedule.py:175-191) are not read.
  - Performance correction / Constants: "lower reductions and packed data to the selected target operations":
    reductions are serial loops (lower_metal.py:744-747, 723-726, 768-769); no packed data in the chain. "express maps
    and epilogues as operation regions": they are closures returning target text (lower_metal.py:155-176, 451-457).
    "Compile producer publication and consumer acquisition from the same range/page dependencies used by transport
    scheduling.": publication and acquisition are host calls (bridge.py:68-70, 79-82) and lowering's dependences are
    not the plan's (lower_metal.py:844; schedule.py:159). Replaces: `_reduced` (rdma/mesh_compile.py:5060) and
    `_contract` (:3116) are at HEAD. "Operation equivalence ... not ... operator spelling": pipelines and native
    libraries are shared by emitted source text (lower_metal.py:264-267, 279-290).
  - Performance correction / Reductions: "constructs subgroup reductions and workgroup exchanges from reduction
    operations" / "lower reductions and packed data to the selected target operations": sum, mean, amax, amin, any,
    softmax and layer_norm are one thread's serial loop per output element (lower_metal.py:744-748, 723-727, 768-773);
    the chain has no subgroup reduction, no packed arithmetic and no system-scope access. "each of these is a function
    from operations to operations inside lowering": handlers return text-producing closures or stored views
    (lower_metal.py:155-176, 730-749). "Compile producer publication and consumer acquisition ...": host calls
    (bridge.py:68-70, 79-82). Replaces: `_reduced` and `_contract` remain (rdma/mesh_compile.py:5060, 3116).
  - Performance correction / Replaces: "`_pointwise`/`_reduced`/`_contract`'s premature commitment to generic
    dispatches": `_reduced` (rdma/mesh_compile.py:5060) and `_contract` (rdma/mesh_compile.py:3116) are at mesh HEAD,
    reached from program() (:8242); they now build _Region/_Generic objects (:5043-5052, 3129-3134), and whether what
    those lower to is still a generic dispatch was not followed into metal-microbench metal_ir.mm.
  - Performance correction / Write: numerical operations, regions, values, effects: "express maps and epilogues as
    operation regions": a map or epilogue is a Python closure that returns target source text (Fused.expression,
    Product.expression: lower_metal.py:155-176, 432-439, 451-457). "lower reductions and packed data to the selected
    target operations": reductions are serial loops (lower_metal.py:744-747); no packed data is lowered.
  - Performance correction / Write: publication and acquisition: "Compile producer publication and consumer
    acquisition from the same range/page dependencies used by transport scheduling.": publication is a host call made
    after the host copies the value into the transfer's pages (bridge.py:68-76) and acquisition is a host wait
    (bridge.py:79-82), neither compiled into the producer's or consumer's kernels; the relation lowering orders by is
    over its own storages' views (lower_metal.py:841-847), not the page dependences transport scheduling used
    (Plan.dependences, schedule.py:83-86, 159).
  - Performance correction / Write: applicability from structure, not spelling: "Operation equivalence and
    applicability follow the structured operands and effects, not an entry-point name, generated variable name or
    operator spelling.": two kernels are the same pipeline when their emitted source text is equal (PIPELINES keyed by
    source, lower_metal.py:264-267; LIBRARIES keyed by source, :279-290); the signature identifies tasks and stored
    inputs by generated node and placeholder names (signature.py:98-99, 120, 127-131).
  - Performance correction / Performance obligation: "subgroup lowering removes shared exchanges; publication fusion
    removes separate visibility/publication work": the chain has neither; reductions are serial loops
    (lower_metal.py:744-747) and publication is a separate host call after a host copy (bridge.py:68-76). "calling
    unchanged per-operation kernels does not accomplish these transformations": every other operator is one dispatch
    of its own (lower_metal.py:391-398, 727, 748, 773, 799). "The measurement standard is metal-microbench
    `docs/measurement.md`.": no caller reads its transport readings (docs/measurement.md:13-14;
    examples/engines.py:114-118).
  - Complete replacement / all in one commit workgroup: "the compiler interface is finished and the downstream bodies
    are removed together, by other code with the same feature coverage": the downstream bodies are at metal-microbench
    HEAD 0b692ed8: metal_recording.m (MetalRecorderProgram :2478, MetalProgramRun :2648), metal_transform.mm,
    metal_lower.mm, tools/e2b_replay.m; mesh rdma/mesh_compile.py, rdma/allocate.py, rdma/mesh_code.py,
    rdma/mesh-torch.mm are at HEAD; the chain does not cover what they cover (no LiteRT program: export.py:15-16;
    NotLowered: lower_metal.py:1204-1206).
  - Complete replacement / build from source: "Build from source the LiteRT runtime, delegate, ML Drift and Dawn/Tint
    dependencies that actually execute these lowerings.": what is built is linked into .build/libmetal_ir.dylib and
    libmesh_provider.dylib (Makefile:98-101, 114-116), which metal_ir.mm, metal_recording.m and tools/e2b_replay.m
    use; the compile_module chain's lowerings are executed by rdma/mesh-metal4.m, clang++ and Core ML
    (lower_metal.py:264-267, 279-290; lower_coreml.py:252) and link none of them.
  - Complete replacement / production E2B link: "Replaces: ... its generated-source interception as an optimization
    interface, the source transformations listed below, and their alternative preparation paths" / "the compiler
    interface is finished and the downstream bodies are removed together": the E2B executable is built from
    tools/e2b_replay.m with metal_recording.m and metal_ir (Makefile:143-153), all at HEAD, and calls no function of
    compiler/mesh_compiler; MetalRecorderProgram (metal_recording.m:2478) and MetalProgramLoad (:2519) remain as
    preparation paths.
  - Complete replacement / operand table row 1: Numerical operations: "LiteRT's typed model/operation construction":
    not in the chain (export.py:15-16). "regions with typed values, shapes, layouts, indexing maps, reduction axes,
    quantization and numerical precision semantics → specialization and fusion": the chain builds no regions or
    indexing maps; the forms fusion takes are closures returning target text (lower_metal.py:155-176), and nothing in
    ir.py:43-83 or lower_metal.py:26-28 carries quantization.
  - Complete replacement / operand table row 2: Kernel instructions: "Numerical lowering → typed shader
    modules/regions" / "Construct the mesh path directly through IR builders": the chain constructs kernels as MSL and
    C++ text by string formatting (lower_metal.py:400-416, 927-933, 1076-1084); it has no shader module, region or IR
    builder.
  - Complete replacement / operand table row 3: Values and effects: "Bind model values to kernel operands to placed
    storage explicitly.": the placed storage of the plan is bound to no kernel operand: lowering never reads
    Plan.places or Plan.sections (lower_metal.py:1188-1207) and binds buffers it allocates itself
    (lower_metal.py:984-985, 1002-1006, 1019-1026); only a carried transfer's pages are read from the layout, by a
    host copy (bridge.py:59-61, 73-76). "masking and publication": the chain lowers neither; publication is a host
    call (bridge.py:68-70).
  - Complete replacement / operand table row 4: Program relations: "Call, dependence and collective-schedule results →
    remapped relations through fusion, splitting and placement → the shared graph algorithms and native program": the
    schedule's dependence and collective-schedule results do not reach lowering (lower_metal.py:1188-1207 never reads
    scheduled.plan). "Update affected results when transformation changes their operands.": stored changes the plan's
    values and solves again (driver.py:26; schedule.py:178-196) and the function order made from the first solve's
    times (schedule.py:206; driver.py:25) is not updated.
  - Complete replacement / operand table row 5: Executable artifact: "→ the native executor": the executor is a Python
    closure per processor that loops over entries (lower_metal.py:892-905) on Python threads (runtime.py:44-52). "CPU
    execution is native Apple Silicon work; Python preparation is not the CPU target.": per call, Python allocates
    results, patches addresses and performs every send and recv (lower_metal.py:1019-1050, 1151-1155, 901-903); a
    program run without lowering executes FX GraphModules in Python (runtime.py:15; examples/tp.py:198).
  - Complete replacement / operands kept until finalization; publication before execution: "The compiler publishes the
    program and its bodies before execution": lower_inductor.py:72 returns torch.compile's callable; its bodies are
    compiled at the first execution (runtime.py:50, 72).
  - Complete replacement / many-to-many identities; producer-supplied meaning: "A tensor's reduction or quantization
    meaning is supplied by its producer": no producer in the chain supplies a quantization meaning: Exported and the
    node annotations lowering reads carry none (ir.py:43-50; lower_metal.py:26-28, 347-350), and the LiteRT producer
    is not in the chain (export.py:15-16).
  - Complete replacement / authored sources; emitted source never inspected; tuned projection/argmax: "Tuned
    projection/argmax arithmetic is carried through typed parameters and computation regions, including their
    map/epilogue operands.": the chain has no argmax lowering (lower_metal.py:306-319) and its product's epilogue is a
    closure returning text (lower_metal.py:432-439, 968-978), not a computation region; the tuned E2B projection and
    argmax are in metal-microbench files outside the chain.
  - Complete replacement / Tint objects behind handles; cache hits: "Tint objects are created and destroyed in the C++
    compiler; C/Objective-C/Python consumers use handles and typed operand/binding records through the program
    boundary. They do not serialize an operation into a string to cross it.": the chain has no C++ compiler and no
    Tint object; its compiler is Python and each kernel crosses to Objective-C as a source string (lower_metal.py:933
    -> rdma/mesh_metal4.py:36-38 -> rdma/mesh-metal4.m:51). The Tint-based compiler is metal-microbench metal_ir.mm
    and metal_lower.mm, outside the chain.
  - Complete replacement / replaced-implementation table row 1: "Torch's reading of static operator/layout metadata at
    run time" (replaced): a program run without lowering executes FX GraphModules, torch reading each operator and
    layout at every call (runtime.py:15; examples/tp.py:198), and rdma/mesh_compile.py remains (:8242); a lowered
    function rebuilds each result's strided view from static shape and stride at every call (lower_metal.py:293-297,
    905). "Constant propagation and specialization before target emission": lower_metal has no constant propagation
    (lower_coreml.py:113-116 has) and nothing in the chain takes a quantized operation (lower_metal.py:306-319).
  - Complete replacement / replaced-implementation table row 2: "Compose computation regions and indexing maps, remap
    uses and eliminate dead intermediates.": what is composed are closures returning target text and text index
    expressions (lower_metal.py:155-176, 247-251, 451-457); a node with no user is still stored by a dispatch of its
    own (lower_metal.py:345, 362-365, 371-382). The LiteRT program these functions transformed is not lowered by the
    chain (export.py:15-16).
  - Complete replacement / replaced-implementation table row 3: "Select/construct the tuned algorithms from typed
    operations, layouts and reduction domains. Preserve packed arithmetic, parallel reductions and split work": the
    chain has no argmax lowering (lower_metal.py:306-319), no packed arithmetic, and its reductions and normalizations
    are one thread's serial loop per output element (lower_metal.py:744-748, 768-773).
  - Complete replacement / replaced-implementation table row 4: "Lower typed memory effects and page readiness to
    coherent access, publication, worklists and later-pass kernels. Preserve computation/transport overlap and fused
    publication.": the chain lowers none of these: a send waits for the commits it reads, then the host copies the
    value into the transfer's pages and publishes by a host call (lower_metal.py:860-865, 903; bridge.py:68-76); a
    recv is a host wait (bridge.py:79-82); lower_metal.py has no worklist, mask or later-pass kernel.
  - Complete replacement / replaced-implementation table row 6: "Emit the shared native program's required
    synchronization regardless of the caller; delete that path and the orchestration it replaces.": the orchestration
    is at HEAD: metal-microbench tools/e2b_replay.m (:1483-1517) and metal_recording.m
    MetalRecorderProgram/MetalProgramRun (:2478, :2648), mesh rdma/mesh_compile.py _replayed (:7660); the chain has no
    shared native program, a Python closure per processor (lower_metal.py:892-905).
  - Complete replacement / after the replacement: "the source has no ... second compiler mode or path back to the
    removed implementation": two compilers of Torch programs are at mesh HEAD, rdma/mesh_compile.py program() (:8242)
    with rdma/allocate.py, rdma/mesh_code.py, rdma/mesh-torch.mm, and compiler/mesh_compiler; metal-microbench keeps
    the recorder's program path (metal_recording.m:2478, tools/e2b_replay.m).
  - Source bindings / Compact effects: the mesh-graph function: "scheduling, fusion and lifetime consumers take that
    relation": only the lowering's commit ordering takes it (lower_metal.py:844-856); the schedule's dependences come
    from mesh_graph.precedence over whole-value events (schedule.py:83-86), fusion is decided from use counts
    (lower_metal.py:362-369), and lifetimes from mesh_graph.ends over alias roots (schedule.py:174-176).
  - Source bindings / Compact effects: producer and consumers: "scheduling, fusion and lifetime consumers take that
    relation": the schedule stage does not: schedule.py:83-86 builds its dependences from per-transfer events through
    mesh_graph.precedence, and schedule.py:174-191 takes lifetimes from mesh_graph.ends over alias roots; fusion
    (lower_metal.py:362-369) reads use counts. The relation's one consumer is lower_metal.ordered
    (lower_metal.py:844-856).
  - Source bindings / Software binary64: "They implement the scalar operations of a typed binary64 kernel when the
    selected target lacks native binary64 instructions." / "lowering takes the typed binary64 instructions and emits
    target instructions": on Metal the chain has no binary64: TYPES omits torch.float64 (lower_metal.py:26-27), a
    float64 value is reported missing (lower_metal.py:347-350, 323, 335-336) and lower raises NotLowered (:1204-1206);
    neither SoftFloat nor FDLIBM source is in mesh. "Derived source carries the upstream notices.": the derived source
    is metal-microbench metal_binary64.wgsl with metal_lower.mm, outside the chain.
  - 7 / data structure: the scheduled value: "Analysis solves the abstract problem once and supplies a lowerable
    program to its optimizing consumers.": mesh_plan is solved twice per compile (driver.py:24, 26; schedule.py:160,
    196); dependences are analysed again in lowering from views, without the plan's (lower_metal.py:844;
    schedule.py:159); aliasing is analysed at schedule.py:91-98, placement.py:627-629 and lower_metal.py:686-695.
    "storage ... annotations stay available until their last compiler consumer": they have no compiler consumer
    (lower_metal.py:1188-1207 never reads scheduled.plan).
  - 7 / Composes: the archive stores the scheduled value: "Lowering takes it and emits the native program": lowering
    returns, per processor, a Python closure that loops over entries and branches on their kind
    (lower_metal.py:892-905; lower_coreml.py:266-282), or torch.compile's callable (lower_inductor.py:72); only
    kernels and segment entries are native (lower_metal.py:1141-1146) or Metal pipelines (:933).
  - 7 / Composes: lowering emits the native program, the runtime calls it: "Lowering takes it and emits the native
    program: the table below gives, per operand, what the emitted program preserves and what it does not read.": what
    is emitted per processor is a Python closure over a list of entries (lower_metal.py:892-905); at each call it
    reads entry kinds (:895-903), Storage dtype and count (:1019-1026) and View shape and stride (:293-297, 905),
    which the table lists as representations execution does not read.
  - 7 / Write: "Emit fixed target/API operations with bound operands, in the scheduled order": the Metal operations
    are not emitted fixed; at every call Python walks the entry list and branches on kind (lower_metal.py:895-903),
    patches addresses (:1042-1047), and mesh_metal4_commit encodes each dispatch again from a record array, branching
    on its barrier and groups flags (rdma/mesh-metal4.m:177-193). "The executor bodies this replaces are deleted with
    it.": metal-microbench metal_recording.m MetalProgramRun (:2648) and mesh rdma/mesh_compile.py _replayed (:7660)
    are at HEAD.
  - 7 / table row 1: typed numerical operations: "Representation that execution does not read: Type objects, shape
    inference, operation names, alias-analysis graphs": the lowered function reads Storage dtype and count and View
    shape and stride at every call (lower_metal.py:1019-1026, 293-297, 903-905); lower_coreml's reads each output
    node's meta val at every call (lower_coreml.py:277-278); a program run without lowering executes the FX
    GraphModules, each operator by name with torch's shape inference per call (runtime.py:15; examples/tp.py:198).
    "required lifetimes": the emitted program takes none from the schedule (schedule.py:179-191 unread;
    lower_metal.py:984-985).
  - 7 / table row 2: partitioned computation, transfers, compiled order: "Representation that execution does not read:
    Original vertices, edges, partitions": lower_coreml's function keeps a dictionary keyed by the graph's nodes and
    at each call reads each host node's args and kwargs (lower_coreml.py:266-282); a program run without lowering
    executes the original vertices (runtime.py:15; examples/tp.py:198). "legal overlap": a send's transport starts
    only after the host has waited for its commits and copied the value (lower_metal.py:860-865, 903;
    bridge.py:73-76).
  - 7 / table row 3: realized storage: "Realized storage, copies and completion dependencies | Correct value instance,
    non-overlapping live storage where required, reuse only after completion": the emitted program does not use the
    storage the schedule realized (schedule.py:165-196; lower_metal.py:1188-1207 never reads scheduled.plan); every
    held value has a separate buffer and none is reused (lower_metal.py:358-360, 984-985, 1116-1117). "Representation
    that execution does not read: ... parsing allocation metadata on each task": each call reads the Storage records
    and allocates the results (lower_metal.py:1019-1026, 1149; lower_coreml.py:277-278).
  - 7 / table row 4: target commands: "Representation that execution does not read: Static command-kind switches, pass
    scans, cut discovery and generic callback tables": each call branches on every entry's kind
    (lower_metal.py:895-903); mesh_metal4_commit re-encodes every dispatch from a record array, branching on barrier
    and groups and setting every address (rdma/mesh-metal4.m:177-193); the native target calls kernels through a table
    of function pointers and a generic callback (lower_metal.py:70-74, 1133-1139). The chain selects no ICB run.
  - 7 / table row 5: lowered synchronization and its cause: "Lowered synchronization and the cause recorded for each":
    a barrier is recorded as a boolean per step (lower_metal.py:856) and a host wait as the commits due
    (lower_metal.py:860-865), without the edges that cause them; the record's operations carry no cause
    (lower_metal.py:917-925, 876).
  - 7 / contract paragraph: "Dropping an annotation before its consumer ... or silently forgetting its property is an
    incomplete lowering.": the storage annotation's property (each value's place and lifetime, schedule.py:165-196) is
    realized by no lowering (lower_metal.py:1188-1207 never reads scheduled.plan; own buffers at :984-985, 1019-1026);
    lower_metal.hosted removes a copy node's `between` annotation (lower_metal.py:1182-1183) before signature.between
    reads it from that module (signature.py:28-29, 100-101).
  - 7 / residual runtime operands and waits: "Residual runtime operands are actual changing data": each call also
    reads static data: Program looks up each input's processors and positions by name (runtime.py:57-59, 63-64), the
    lowered function reads entry kinds, Storage sizes and View shapes (lower_metal.py:895-905, 1019-1026). "Readiness
    is not permission to repeat placement": result buffers are placed anew at every call (lower_metal.py:1019-1026,
    1149; lower_coreml.py:277-278).
  - 7 / performance paragraph: "Semantic equivalence does not excuse an extra run-time decoder, launch, transfer,
    guard or wait introduced only to maintain an erased abstraction.": a run-time decoder remains: Python's loop over
    entries (lower_metal.py:895-903) and mesh_metal4_commit's loop over dispatch records (rdma/mesh-metal4.m:177-193);
    an extra transfer: each sent value is copied into the transfer's pages (bridge.py:73-76). "Removing the outermost
    decoder while the same dispatch runs underneath it does not satisfy the requirement.": the native target's entries
    call each kernel through a part table and callback (lower_metal.py:70-74, 1137-1139).
  - 4 / Write: the bridge binds physical transport as R7 specifies: outstanding [2026-11-12, mesh f20efc91; review of
    f20efc91]. "The bridge binds physical transport as R7 specifies.": R7's requirement, "The code must keep every
    transport object and count inside the bridge", is outstanding in this index under plan-to-bridge (schedule.py
    reads the map's cable records in the compiling process).
  - 6 / Replaces 2: signature printed from the resulting compilation: outstanding [2026-11-12, mesh 6d379b7e; a second
    reader]. "The signature is printed from the resulting compilation."
    (design/compilation-composition-change-request.md:252): compiler/mesh_compiler/signature.py:100-101:
    `records[processor] = lowered_run(function.record, on_mps) if hasattr(function, "record") else torch_run(function
    if isinstance(function, torch.fx.GraphModule) else module)`. Only lower_metal's functions carry a record
    (lower_metal.py:1037, :1064). A processor lowered by lower_coreml (its `run`, lower_coreml.py:266-288, has no
    `record`) or by lower_inductor (lower_inductor.py:72) has its tasks, dependences and operations printed by
    torch_run (signature.py:33-44) from `scheduled.modules[processor]`, the scheduled fx module before lowering
    (driver.py:45-46 passes the unlowered `scheduled`), not from what that lowering produced. The chain runs this path
    with lower_coreml at compiler/samples.py:1116.

Covered:

  - 1 / Replaces: covered [2026-11-08, mesh d44e3d0c]
  - 3 / Write, first bullet: covered [2026-10-30, mesh 5ccf248a]
  - 4 / Write: runtime consumes programs and remaining inputs; self-timed calls: covered [2026-11-12, mesh 81c8e9da]
  - 5 / Replaces, site 3: strided-view traversal: covered [2026-11-08, mesh d44e3d0c]
  - 6 / Replaces 1: covered [2026-11-12, mesh ead08ba2]
  - Correction: generated target source / Replaces: covered [2026-11-09, metal-microbench f08d3c7b]
  - Complete replacement / replaced-implementation table row 5: covered [2026-11-12, mesh 69d5aa4d]
  - 7 / Replaces: MetalLoopTest and its whole-queue waits: covered [2026-11-10, metal-microbench 1b6f3182]
  - 7 / Replaces: run-time readers of positions, tasks, guards; static discovery: covered [2026-11-10,
    metal-microbench 1b6f3182]
  - Withdrawn, 2026-11-09: the stated scope: covered [2026-11-10, mesh ce2e8ac9]

Withdrawn:

  - Withdrawn, 2026-11-09: clauses on control flow that depends on data ("Nested communication regions: producer and
    consumer handoff", loop stack, predicate drains): withdrawn [2026-11-12 reading].
    design/compilation-composition-change-request.md:542 ("## Withdrawn, 2026-11-09") with the operator's turn at :554
    ("strike out anything and everything which contradicts the remarks reaffirmed ...") and the scope that replaces
    the clauses at :556; the request no longer contains the "Nested communication regions" clauses (its only "loop
    stack" outside the Replaces of requirement 7 is the operator's quotation at :489).

## [aot-compilation-change-request.md](aot-compilation-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 59 items outstanding, 1 not determined, 50 covered, 2 withdrawn.

Outstanding:

  - A1 every function of a program: "Every function of a torch program is captured and compiled before the program
    runs": lower_inductor.py:72 returns torch.compile(adapted, backend="inductor") callables, which compile at their
    first call inside Program (runtime.py:50,72); with no functions given, Program runs the exported fx GraphModules
    themselves (runtime.py:15). "The program's calls between functions are A10's graph.": nothing in
    compiler/mesh_compiler builds a graph of a program's calls; compile_module takes one module (driver.py:14).
  - A1 a program run through another framework's delegate: "Its graph is the recorder's capture of the delegate's
    commands (metal_recording.m)." and "For the program the code runs today on LiteRT, tools/prepare_e2b.py reads the
    model's graph.": export takes a torch module only (export.py:15-16); nothing in compiler/mesh_compiler reads a
    delegate's capture or a LiteRT model. The capture is in metal-microbench metal_recording.m and the model is read
    by tools/prepare_e2b.py for tools/e2b_replay.m, none of which reaches compile_module.
  - A3 produced by, consumed by (the caller passes the operand; mesh's type): "The caller passes the operand.": no
    call of compile_module in examples/ or compiler/samples.py passes `cost` (examples/tp.py:138;
    compiler/samples.py:327); each takes the default unit_cost of driver.py:14. metal-microbench has no caller of
    compile_module.
  - A4 algorithm: AllReduce: "to do an AllReduce operation on the directed graph, we first run a reduce operation to a
    chosen root vertex using the undirected graph and then do a broadcast operation from the root vertex using the
    same tree but with links going in the reverse direction.": place sums the shards' partial results by ADD on the
    keeping processor after each is sent there on its own channel (placement.py:596-599, 675-680); a plan_entry is a
    send or a receive (mesh-collective.c:112, 772-783) and none reduces at a relay; each transfer is its own demand
    with its own trees (schedule.py:152; mesh-collective.c:1687-1690,1758), not one tree used in both directions.
  - A4 algorithm: AllGather: "AllGather is one broadcast from each source over its packed trees: §3.1's broadcast,
    once for each source, as rdma/mesh_compile.py `_all_gather` declares it.": place gives each processor that reads a
    value its own channel (placement.py:675-677) and planned declares each channel as a transfer whose identity is the
    channel (schedule.py:152); mesh_plan joins transfers into one demand of several leaves only where source and
    identity are equal (mesh-collective.c:1681-1683, 1700), so a value read on k machines is k demands of one leaf
    each. `_all_gather` is in rdma/mesh_compile.py, which the compile_module chain does not call.
  - A4 algorithm: ReduceScatter: "Here each slice is reduced to the member that keeps it, over its packed trees: the
    first half of Blink §3.3's AllReduce, with that member as the root.": each shard's piece is sent to the keeper on
    its own channel, a demand of its own (placement.py:586-592, 675-680; schedule.py:152), and the sum is the keeper's
    ADD (placement.py:596-599); no entry of mesh_plan reduces along a tree toward a root (mesh-collective.c:772-783).
  - A4 where the partitioning inserts ReduceScatter and CollectivePermute: "and into a pipeline's buffer shifting,
    \"GSPMD will turn the buffer shifting into cross-device communication via CollectivePermute\" (§3.3)": RULES
    (placement.py:372-386) has no entry for aten.roll or aten.index_select; an operator with no rule takes
    `replicated` (placement.py:476), and its operands are resharded to one processor where it runs
    (placement.py:491-495).
  - A5 data structure and algorithm: "the order in which two items may share a page, Lamport's happens-before (...);
    and the fewest pages as the fewest chains of that order, by Dilworth's theorem, found by maximum bipartite
    matching": mesh_graph_chains is called with the order plan_alone, which returns 0 for every pair
    (mesh-collective.c:479-482, 658), so no two of its items share a page; items with a lifetime share pages by
    mesh_graph_fat_cover over their [start, end) intervals of operation places (mesh-collective.c:662-672).
  - A5 produced by, consumed by: "The bridge and the GPU's program take the bindings from the scheduled value (A9).":
    lower_metal.lower reads scheduled.devices, modules, engines and carried and no plan (lower_metal.py:1400-1406);
    its storages are its own buffers (lower_metal.py:1136,1174) or views of the tensors handed to it at run time
    (lower_metal.py:1183-1188); Program stores Published.address in ir.LOCAL.storage (runtime.py:25,48), which nothing
    reads at HEAD.
  - A6 data structure and algorithm: "The algorithm is design/plan-to-bridge-change-request.md R5's: R5's heap order
    by the completion times of one linear program": the order of each processor's operations, which `schedule` writes,
    is heap_order keyed by each function's longest-path level under `cost` (schedule.py:203, 208-210); only the recv
    of a channel between machines is keyed by the linear program's time (schedule.py:204-206). The linear program has
    tasks for the transfers and their communications alone (mesh-collective.c:836-848, 943-948).
  - A7 data structures: m, sn(m), Dest(m): "e2b's run is one message.": no caller of the compile_module chain runs
    e2b; e2b is submitted by metal-microbench tools/e2b_replay.m alone.
  - A7 sender(m) (the launcher passes the program's nodes): "e2b's launcher, tools/mesh/run_pair.py, passes no node
    set ... which tools/e2b_replay.m passes.": run_pair.py launches .build/e2b_replay (metal-microbench
    tools/mesh/run_pair.py:290), not a compile_module program. "A torch program's launcher passes them as MESH_PROGRAM
    beside MESH_GROUP (rdma/mesh_compile.py `_program`).": the one launcher that sets them for a compile_module
    program is compiler/samples.py:1296-1297, with the literals MESH_GROUP "1" and MESH_PROGRAM "0,1"; `_program` is
    in rdma/mesh_compile.py. "so the program's nodes are every member of the view": samples.py:1297 passes the fixed
    "0,1".
  - A7 views: the four view properties: "(Local Monotonicity) If a process p installs view V after installing view V′
    then the identifier of V is greater than that of V′.": view_install compares (id, sequencer, boot) by
    mesh_view_compare (mesh-records.c:682-683; rdma/mesh.h:97-100), so a view whose id equals the installed view's is
    installed where its sequencer's node number, or at the same sequencer its boot, is greater.
  - A7 views: the membership service is the sequencer: "It also creates one whenever a member's announcement names a
    view other than the sequencer's installed one, with a counter not less than that view's identifier.":
    view_announced creates one only where the named view compares greater than the installed one in the order (id,
    sequencer, boot) (mesh-records.c:757-759; rdma/mesh.h:97-100); an announcement of another view with the same id
    whose sequencer or boot is less creates none.
  - A7 views: a bridge installs a view: "if the view names it a member (Self Inclusion) and the view's identifier is
    greater than its counter (Local Monotonicity)": view_install installs where (id, sequencer, boot) compares greater
    than the installed view's (mesh-records.c:682-683; rdma/mesh.h:97-100), so a view whose id equals the counter is
    installed where its sequencer or boot is greater.
  - A7 views: a static schedule is used only within its view: "and `mesh_plan` computes it again in the new view":
    compile_module calls mesh_plan at compile time (driver.py:24,26; schedule.py:111) and Program loads the record
    once at its construction (bridge.py:25-28); after a view is installed mesh_transfers_start returns ESTALE
    (mesh-dataflow.c:252, 267), which bridge.py:42 raises through mesh.check (rdma/mesh.py:292-294). Nothing in
    runtime.py or bridge.py calls mesh_plan or mesh_plan_load again.
  - A8 data structure (the type mesh_plan takes): "A10's output.": the chain passes the constant 1 as `outstanding`
    (schedule.py:159) and no order constraints or chains (schedule.py:110: None, None, 0, None); no analysis produces
    them.
  - A9 here, per node: the GPU's program: "No kernel is written by the compilation.": lower_metal writes Metal kernel
    source for each lowered operation and compiles it (lower_metal.py:1079-1085 `kernel void k(...)`, pipeline at
    :273-275; the native form's source at :1349-1350). "For a delegate's program these are the recorder's command
    buffers.": the compile_module chain takes no delegate's program; the recorder is metal-microbench
    metal_recording.m.
  - A9 nothing else is serialized: "Nothing else is serialized but each entry's `set` (appendix item 5).":
    mesh_plan_save also writes each transfer's completion time, the synchronization edges, the publications, the
    adjacency, the planned calls and orders and a signature (mesh-collective.c:2209-2227, 2273-2283); archive.save
    also writes each processor's fx module as a .pt2 file, the plan's operands (declared, weights, values,
    dependences, times, places, sections), carried, blocks, subarrays and signature.record (archive.py:52-79).
  - A10 data structure: the calls' dataflow graph: "The data structure is the calls' dataflow graph." and "For torch,
    the tracer is torch.fx's symbolic tracer ... Each compiled function is a leaf, so a call of it is one node of the
    graph": nothing in compiler/mesh_compiler, compiler/samples.py or examples/ traces a block of calls of compiled
    functions; compile_module takes one module a call (driver.py:14) and mesh_plan is given one function and one call
    (schedule.py:109-110).
  - A10 algorithms: independence, commuting operations, the bound: "Independence: two calls may run in either order,
    or at once, where neither writes what the other reads or writes.", "Commuting operations: calls whose effects
    commute ... may run in any order" and "So the most calls of a function outstanding at once is the number of its
    calls in the block, or fewer where Bernstein's conditions order them.": the compile_module chain computes none of
    them; it passes one call, no orders, no chains and the constant bound 1 (schedule.py:109-110, 159).
  - A10 the graph operations are the graph module's (callers; Interpreter; wherever mesh walks a graph): "As that
    module states, \"A caller supplies its vertices and a neighbours function.\"": rdma/mesh_graph.py holds no such
    sentence (its head is line 1, "# design/rubrics.md C9"). "rdma/mesh_compile.py calls them and holds no copy.":
    nothing in the compile_module chain calls mesh_graph.paths or mesh_graph.width; rdma/mesh_compile.py is not called
    by it. "A body's symbolic execution runs on torch.fx.Interpreter, not on a second loop over a graph's nodes.":
    compiler/mesh_compiler uses no torch.fx.Interpreter; export refuses a graph with a control-flow operator
    (export.py:26-31).
  - A10 its output (the receiving types): "for each compiled function, the most calls of it the analysis lets be
    outstanding at once in the program that submits them" and "the order constraints between calls": no analysis in
    the compile_module chain produces either; schedule.py:159 writes the constant 1 and schedule.py:110 passes None
    for orders, causes and chains.
  - A10 produced by, consumed by (the schedule): "The schedule takes the bound as a delay of its IPC graph (appendix
    item 2).": the value the chain's schedule stage gives is the constant 1 (schedule.py:159), not a bound of A10's
    analysis, and it gives no order constraints (schedule.py:110).
  - A10 produced by, consumed by (the program issues calls ahead): "At run time the program issues calls ahead, as
    futures, in the order the analysis allows": Program.__call__ starts one execution, runs the functions, waits for
    every other processor to finish (runtime.py:73-75) and for each transfer's sends (runtime.py:76-77;
    bridge.py:45-48), then returns the results (runtime.py:80-86); it returns no future and issues no call ahead.
  - Appendix: a chooser for edge weights: "The algorithm is the α–β model of a transfer.", "an edge e of n bytes
    weighs α_k + n β_k on each cable k of its route" and "a function from (bytes, cable) to a weight, passed to the
    compilation as its edge-weight chooser": no such function is in compiler/, examples/ or metal-microbench, and no
    caller passes `cost` to compile_module (examples/tp.py:138; compiler/samples.py:327); the edge weight is
    unit_cost's 1.0 (schedule.py:26-27, 153).
  - Appendix item 2: the queue is laid out with the schedule's pages and read in place: "Only the driver moves bytes
    (design/plan-to-bridge-change-request.md R1).": Carried.put copies each sent value into its pages
    (bridge.py:73-76, `flat.copy_`), and Carried.out, which returns the pages, has no caller in
    compiler/mesh_compiler.
  - Appendix item 2: nothing waits to write; the only waits: "Each ends when its source moves on, or when the view
    ends and the static schedule with it.": the wait also ends by returning ETIMEDOUT when MESH_PLATFORM_NS passes
    with no rise of the count (mesh-dataflow.c:249, 253, 257).
  - Appendix item 5: the algorithm (Fisher and Yates over Threefry; Lemire): "whose group the launcher draws anew at
    each launch (metal-microbench tools/mesh/run_pair.py: `1 + secrets.randbelow(2**64 - 1)`, passed as
    `MESH_GROUP`)": run_pair.py:379, 287 draws and passes it to .build/e2b_replay (run_pair.py:290) alone; for a
    compile_module program the group is int(MESH_GROUP) << 32 plus a count (bridge.py:105), and the launcher that sets
    MESH_GROUP for one passes the literal "1" (compiler/samples.py:1296).
  - Appendix item 5: every program declares each consumer's kind: "Every program declares each consumer's kind." and
    "A consumer that reduces partial sums, or otherwise takes pages in any order, is an asynchronous, incomplete,
    noncontiguous consumer (R4).": on_devices marks every arrival JOIN (devices.py:35); PAGES (ir.py:34) is assigned
    nowhere, and no caller in examples/ or compiler/samples.py declares a kind. "For the program the code runs today,
    tools/prepare_e2b.py does the declaring": prepare_e2b.py prepares tools/e2b_replay.m's input and does not reach
    compile_module.
  - Appendix item 6: the compiled function's name; alike at every rank: "The data structure is the compiled function's
    name: (the program's group, the function's ordinal in the program's order of compilation), the same at every
    rank.": the chain names a function (int(MESH_GROUP) << 32) + next(PUBLISHED) with epoch always 1 (bridge.py:15,
    105; count 1 at bridge.py:26, so epoch is f + 1 = 1 at mesh-collective.c:2388); the count is of the Programs
    constructed with a plan in this process, in their order of construction, not the order of compilation; at compile
    time the group is int(MESH_GROUP) alone (schedule.py:111).
  - Appendix item 6, change: any number of compiled functions at once (mesh_plan): "let a program hold any number of
    compiled functions at once, each with its own static schedule": the chain passes one function a call
    (schedule.py:108-110; bridge.py:26); each Published first retires what this client published (bridge.py:24;
    mesh-dataflow.c:17-21) and then publishes a table of its one function (mesh-collective.c:1846-1852), so a second
    compiled function's Program ends the first's static schedule.
  - Appendix item 6, stage: a program holds one scheduled value for each compiled function (the compile chain): "A
    program holds one scheduled value for each compiled function, and the runtime calls each.": where two Programs of
    one process each have transfers between machines, the second's Published retires the first's published static
    schedule before loading its own (bridge.py:24-28; mesh-dataflow.c:17-21), and the bridge serves only the published
    group's (mesh-records.c:909-913), so the runtime cannot call each.
  - Appendix item 6, change: one allocation over the whole program; deferred reclamation: "place each static schedule
    with every other of the program by the one allocation over the whole exported program": each compile_module calls
    mesh_plan with its one function (schedule.py:108-110), so two functions of a program are placed by separate
    allocations. "A function's pages are released when the program retires it or itself": the retirement is made when
    the next Published is constructed (bridge.py:24); runtime.py retires nothing at a Program's end, and
    mesh_plan_retire (mesh-dataflow.c:28) has no caller in compiler/mesh_compiler.
  - The files and functions: rdma/mesh_compile.py: "`_all_gather` declares one broadcast from each source (A4)": the
    chain declares one transfer for each reading processor, each its own demand (placement.py:675-677;
    schedule.py:152; mesh-collective.c:1681-1683). "A10's analysis calls rdma/mesh_graph.py's functions and holds no
    copy of them.": the chain has no A10 analysis (schedule.py:110, 159). The functions the clause names are in
    rdma/mesh_compile.py, which compile_module does not call.
  - The files and functions: metal-microbench tools/mesh/run_pair.py: "draws the group and passes it as `MESH_GROUP`;
    starts rank r at map index r on every node of the collective (A7, item 5)": what it starts is .build/e2b_replay
    (run_pair.py:290), not a compile_module program; no launcher of a compile_module program draws a group
    (compiler/samples.py:1296-1297 passes "1" and "0,1").
  - The files and functions: metal-microbench tools/e2b_replay.m: "passes the program's nodes, map indices 0 … R − 1
    (A7)": the only code that does is metal-microbench tools/e2b_replay.m (:1322), which is not in the compile_module
    chain; the chain's program nodes are read from MESH_PROGRAM (schedule.py:158).
  - The files and functions: metal-microbench tools/prepare_e2b.py: "reads the model's graph (A1); declares each
    consumer's kind (item 5)": tools/prepare_e2b.py reads a tflite model and prepares tools/e2b_replay.m's input; it
    does not reach compile_module. In the chain the graph is a torch module's (export.py:16) and every consumer is
    marked JOIN (devices.py:35).
  - The files and functions: metal-microbench metal_recording.m: "the recorder's capture of a delegate's commands
    (A1)": the capture is in metal-microbench metal_recording.m alone; nothing in the compile_module chain takes a
    delegate's commands (export.py:15-16).
  - The stages each requirement belongs to (1-6): outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "There is
    one schedule." (design/aot-compilation-change-request.md:95): compiler/mesh_compiler/driver.py:24 and :26 solve
    mesh_plan twice in one compile_module (planned -> schedule.py:160, stored -> schedule.py:196, both through solved,
    schedule.py:111-113), the second with other values (schedule.py:178-195). driver.py:25 orders each processor's
    operations by the first solve's times (schedule.py:204-206), while the record the bridge loads is the second
    solve's (schedule.py:118; bridge.py:25-28).
  - A1 not a torch.compile backend: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "It is not a
    `torch.compile` backend." (design/aot-compilation-change-request.md:129):
    compiler/mesh_compiler/lower_inductor.py:72: `compiled[processor] = torch.compile(adapted, backend="inductor",
    options=options)`. Each processor's scheduled module is handed to torch.compile, which captures and compiles it at
    its first call inside Program (runtime.py:50, :72). The export stage itself is torch.export (export.py:16) and
    registers no backend; the refutation rests on the lowering the chain calls. The same index lists lines 150-151
    outstanding on this line of code.
  - A2 algorithm (GSPMD's partitioning): outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "It consumes
    sharding annotations on a few tensors, propagates them to every operator by compiler passes over a dataflow IR,
    and inserts the collectives." (design/aot-compilation-change-request.md:174-175): "every operator":
    compiler/mesh_compiler/placement.py:476-477 gives an operator with no entry in RULES (:372-386) the rule
    `replicated`, and :491-496 then reshards its operands to one processor, where it runs unpartitioned. "inserts the
    collectives": place inserts one mesh::send/mesh::recv pair for each reading processor (placement.py:674-680), and
    a reduction over shards is ADD nodes on the keeping processor (placement.py:595-598). The same index lists A4's
    AllReduce, AllGather and ReduceScatter outstanding for this.
  - A2 produced by, consumed by: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "Placement takes A1's graph
    and the sharding specs and returns the partitioned graph with its collectives, which A4 turns into sends and
    receives of pages." (design/aot-compilation-change-request.md:177-178); also "A6's schedule takes the result, in
    A8's IR." (line 179): compiler/mesh_compiler/placement.py:674-680 returns the graph with send/recv pairs already
    in it, one channel a reader, and placement.py:595-598 sums shards by ADD on the keeper. A4's AllReduce, AllGather
    and ReduceScatter are outstanding in the same index. The A8 and A6 it cites are outstanding too: the chain passes
    outstanding = 1 and no orders (schedule.py:110, :159), and the processors' order is not R5's
    (schedule.py:203-210).
  - A4 stage: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "Placement inserts the collectives (A2).
    `mesh_plan`, the schedule stage's function, routes them and lowers them to sends and receives of pages
    (rdma/mesh-collective.c)." (design/aot-compilation-change-request.md:219-220):
    compiler/mesh_compiler/placement.py:674-680: place adds send and recv nodes, as the clause's next sentence says
    the reference program does, one channel for each reading processor. A reduction is ADD on the keeper
    (placement.py:595-598), which mesh_plan never sees. mesh_plan is declared each channel between machines as a
    demand of one leaf (schedule.py:152; rdma/mesh-collective.c:1681-1691). The same index lists A4's AllReduce,
    AllGather and ReduceScatter outstanding.
  - A4 data structure and algorithm: packed spanning trees, packing: outstanding [2026-11-12, mesh 6d379b7e; a second
    reader]. "**Packing.** Blink §3.2: the packing is a multiplicative-weight update for fractional packing.
    `mesh_graph_pack`'s Garg and Könemann rounds are a multiplicative-weight method of the same kind."
    (design/aot-compilation-change-request.md:230-231): mesh_graph_pack is defined at rdma/mesh-graph.c:374 and
    declared at mesh-graph.h:50, and nothing calls it. The routes are made by mesh_graph_flow alone
    (rdma/mesh-collective.c:1758). With the chain as producer every demand has one leaf (schedule.py:152;
    mesh-collective.c:1681-1683), so each tree is a path from source to destination.
  - A4 which nodes hold a function's static schedule: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "Each
    compiled function has its own static schedule (appendix item 6). Every node of the map is a node of the program,
    since every run is launched on the whole collective, so every relay holds the static schedule (A7, sender(m))."
    (design/aot-compilation-change-request.md:264-266): The program's nodes are whatever MESH_PROGRAM names
    (compiler/mesh_compiler/schedule.py:158); the one launcher passes the literal "0,1" (compiler/samples.py:1296-1297
    by the index's reading). mesh_plan does not make every map node a node of the program: it returns EINVAL where a
    source, leaf or relay is outside `program` (rdma/mesh-collective.c:1794) or the node itself is (:1937). Both
    requirements the sentence cites are outstanding in the same index: appendix item 6 (each Published first retires
    what the client published, bridge.py:24) and A7 sender(m), the launcher's part.
  - A6 produced by, consumed by (the lists): outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "The schedule
    takes A8's IR." (design/aot-compilation-change-request.md:306): By citation only: the same index lists A8's data
    structure outstanding. The struct mesh_function the chain gives mesh_plan carries the constant 1 for `outstanding`
    and no orders, causes or chains (compiler/mesh_compiler/schedule.py:110, :159). The clause's other sentences hold:
    the lists are written at rdma/mesh-collective.c:2051-2078 and loaded at bridge.py:25-28.
  - A6 one schedule per function; the schedule takes A8's IR: outstanding [2026-11-12, mesh 6d379b7e; a second
    reader]. "There is one schedule." (design/aot-compilation-change-request.md:95) and "The schedule takes A8's IR."
    (line 306): compiler/mesh_compiler/driver.py:24 and :26: mesh_plan is solved twice for one function
    (schedule.py:160 and :196 through :111). The modules' order uses the first solve's times (driver.py:25;
    schedule.py:204-206) and the bridge's lists come from the second (schedule.py:118). A8's IR is outstanding in the
    same index (schedule.py:110, :159).
  - A7 stage: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "It takes the scheduled value's name and nodes
    and orders the calls of its functions." (design/aot-compilation-change-request.md:319): The scheduled value holds
    no name: ir.Plan has `program` and no group (compiler/mesh_compiler/ir.py:54-67). The runtime makes the name
    itself at compiler/mesh_compiler/bridge.py:105, `(int(os.environ["MESH_GROUP"]) << 32) + next(PUBLISHED)`, a count
    of this process's Programs. The schedule was solved under another name, int(MESH_GROUP) (schedule.py:111). The
    nodes are taken from the scheduled value (plan.program, bridge.py:26).
  - A7 one sender for each program: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "So FIFO Order holds,
    and with it A10's order constraints between the program's calls." (design/aot-compilation-change-request.md:402):
    By citation: A10's order constraints are outstanding in the same index, and the chain has none.
    compiler/mesh_compiler/schedule.py:110 passes None for orders, causes and chains, with one call of one function,
    and :159 passes the constant 1. The one sender itself holds (rdma/mesh-collective.c:2011; mesh-records.c:1103,
    :1124-1125).
  - A7 views: the view identifier: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "So Local Monotonicity
    holds at every member, a bridge started again among them." (design/aot-compilation-change-request.md:446-447):
    rdma/mesh-records.c:683: `if(!named || mesh_view_compare(id,installed)<=0)return;`. mesh_view_compare
    (rdma/mesh.h:97-100) orders by id, then sequencer, then boot, so a view whose identifier equals the bridge's
    counter is installed where its sequencer's node number or boot is greater. The same index lists Local Monotonicity
    outstanding on these lines.
  - A8 carried as one IR from partitioning to serialization: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    "The data structure is the task graph ... carried as one IR from partitioning to serialization, as an operand of
    its own." (design/aot-compilation-change-request.md:502-503) and "A2 partitions it. `mesh_plan` takes it for A4's
    routing and lowering, A5's allocator and A6's ordering" (lines 510-511): A2 partitions a torch.fx graph
    (compiler/mesh_compiler/placement.py:601-654). The task graph mesh_plan takes does not exist until the schedule
    stage, where planned() builds it from that fx graph as tuples in ir.Plan (schedule.py:139-159), and stored()
    builds it again with other values (schedule.py:178-195). The archive writes both forms side by side (archive.py:62
    and :72-77). A5's allocator and A6's ordering, which the sentence cites, are outstanding in the same index
    (rdma/mesh-collective.c:479-482, :658; schedule.py:203-210).
  - A9 stage and types: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "The bridge and the GPU's program
    take it." (design/aot-compilation-change-request.md:520-521), where "Here it is `struct mesh_algo`, the published
    table" (lines 521-522): compiler/mesh_compiler/lower_metal.py:1441-1459: lower(scheduled) reads scheduled.devices,
    modules, engines and carried, and neither scheduled.plan nor anything of struct mesh_algo; the file has no
    reference to `.plan`. Only the bridge takes the record (bridge.py:25-28). The "(A5, A6)" the clause cites are
    outstanding in the same index.
  - A9 the algorithm the bridge runs it by: the self-timed strategy: outstanding [2026-11-12, mesh 6d379b7e; a second
    reader]. "The graph is A8's task graph, the program's whole graph (A1, A10), as a CUDA graph is the whole of its
    captured work." (design/aot-compilation-change-request.md:557-558): compiler/mesh_compiler/schedule.py:108-110:
    mesh_plan is given one function and one call (`function, 1, calls, 1, None, None, 0, None`), from one module a
    compile_module (driver.py:14). No graph of the program's calls exists; A10 and A1's "every function" are
    outstanding in the same index. The bridge's part holds: list order and a stop at the first unready dependency
    (rdma/mesh-flow.c:1639-1650).
  - A10 the graph operations are the graph module's (the functions): outstanding [2026-11-12, mesh 6d379b7e; a second
    reader]. "**The graph operations:** the order's transitive closure, the walk of commuting chains, and the width by
    Fulkerson's matching in Hopcroft and Karp's phases." (design/aot-compilation-change-request.md:618-619): The
    functions exist (rdma/mesh_graph.py:44 closure, :161 paths, :167 matching, :204 width). Their only callers as
    A10's algorithms are in rdma/mesh_compile.py (:7771, :7783, :7819, :7823), which does not count. In the
    compile_module chain, paths, matching and width have no caller; closure is called once, for the dependences of one
    function (compiler/mesh_compiler/schedule.py:138), not for an order of calls.
  - Appendix item 1 (unrolling): outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "A5's allocator takes u."
    (design/aot-compilation-change-request.md:685): By citation: the allocator that takes u is not A5's, which the
    same index lists outstanding. rdma/mesh-collective.c:658 calls mesh_graph_chains with the order plan_alone, which
    returns 0 for every pair (:479-482), and items with a lifetime share pages by mesh_graph_fat_cover (:662-672).
    Also, u is not only "the largest L": plan_unroll starts it at horizon x calls (:380) and raises it by a value's
    declared lag before the lags are read (:384). No value is refused for its lag (:410).
  - Appendix item 5: the set: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "the pages of asynchronous,
    incomplete, noncontiguous consumers' operands ... are written as sets, not as a sequence"
    (design/aot-compilation-change-request.md:908-909) and "rdma/mesh-collective.c writes each entry's `set` into its
    `struct plan_entry`" (line 927): With the compile_module chain as producer no set is ever written.
    compiler/mesh_compiler/devices.py:35 marks every arrival JOIN; PAGES (ir.py:34) is assigned nowhere;
    schedule.py:152 declares that kind. rdma/mesh-collective.c:1695 then sets `d->set=0` for every demand, and :759
    gives every entry set 0. The j + 1 numbering exists only on a branch the chain never reaches. The same index lists
    "every program declares each consumer's kind" outstanding on devices.py:35.
  - The files and functions: rdma/mesh_graph.py: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    "**rdma/mesh_graph.py:** the order's transitive closure, the walk of commuting chains, and the width by
    Fulkerson's matching in Hopcroft and Karp's phases (A10)." (design/aot-compilation-change-request.md:1070-1071):
    The functions are in the file (rdma/mesh_graph.py:44, :161, :167, :204), but as A10's they are called only from
    rdma/mesh_compile.py (:7771-7823), which does not count. Nothing in compiler calls mesh_graph.paths, matching or
    width, and every A10 item it cites is outstanding in the same index (schedule.py:110, :159).
  - The files and functions: rdma/mesh-collective.c (fed by A2's collectives): outstanding [2026-11-12, mesh 6d379b7e;
    a second reader]. "the demands and trees are fed by A2's collectives, in place of ranges declared by hand"
    (design/aot-compilation-change-request.md:1075): The demands are fed by placement's send/recv channels, one
    transfer for each channel between machines (compiler/mesh_compiler/placement.py:674-680; schedule.py:140-152),
    each a demand of one leaf (rdma/mesh-collective.c:1681-1691). No AllReduce, AllGather or ReduceScatter reaches
    mesh_plan (placement.py:595-598 sums on the keeper); the index lists those outstanding. The clause's second half,
    one mesh_graph_flow call for every demand, holds at mesh-collective.c:1757-1759.

Not determined:

  - How this document is written and used (silence rule): not determined [2026-11-12]. The clause (request :54-55) is
    a rule for how an implementer records a decision; it has no sentence that code holds or fails.
    design/the-compiler.md:73 records one such decision as "Claude's choice, made because ...". Whether every decision
    taken where the request is silent is recorded, no file shows.

Covered:

  - A1 data structure and algorithm: covered [2026-11-12, mesh f8de5bbc]
  - A1 produced by, consumed by (the type): covered [2026-11-12, mesh f8de5bbc]
  - A2 data structure, part 1 (sharding specs): covered [2026-11-11, mesh 297c3cda]
  - A2 data structure, part 2 (mapping onto the map's nodes): covered [2026-11-12, mesh 81c8e9da]
  - A2 the chooser is separate: covered [2026-11-10, mesh ce2e8ac9]
  - A3 data structure: covered [2026-11-12, mesh 81c8e9da]
  - A3 no algorithm computes them: covered [2026-11-12, mesh 81c8e9da]
  - A3 produced by, consumed by (router and scheduler): covered [2026-11-05, mesh f4838c9d]
  - A3 the chooser of the weights: covered [2026-11-10, mesh ce2e8ac9]
  - A4 data structure: the instruction DAG: covered [2026-11-06, mesh 5eeee1f9]
  - A4 algorithm: joint routing: covered [2026-11-05, mesh c6c9c77f]
  - A4 algorithm: CollectivePermute: covered [2026-11-12, mesh 81c8e9da]
  - A4 algorithm: AllToAll: covered [2026-11-12, mesh 81c8e9da]
  - A4 produced by, consumed by (mesh_plan's types): covered [2026-11-05, mesh c6c9c77f]
  - A4 a node to which the placement operand maps no device (mesh_plan's part): covered [2026-11-12, mesh 81c8e9da]
  - A4 a node to which the placement operand maps no device (the program's part): covered [2026-11-12, mesh 8a1b0ea7]
  - A7 one order; total order multicast by a fixed sequencer (primitives, properties, steps of Fig. 5): covered
    [2026-11-05, mesh c38ee188]
  - A7 multicast to many groups: covered [2026-11-06, mesh a172596f]
  - A7 sender(m) (the bridge's and mesh_plan's part): covered [2026-11-12, mesh 81c8e9da]
  - A7 the sequencer: covered [2026-11-05, mesh 7b3afe21]
  - A7 deliver(m); produced by, consumed by: covered [2026-11-05, mesh 7eb93c51]
  - A7 failures: covered [2026-11-06, mesh fa089c56]
  - A7 views: a process: covered [2026-11-06, mesh 61435d4b]
  - A7 views: Fig. 5 runs within each view: covered [2026-11-06, mesh 61435d4b]
  - A7 views: a change of view blocks and flushes: covered [2026-11-06, mesh 61435d4b]
  - A7 views: every member computes its static schedules on the view's map: covered [2026-11-06, mesh a172596f]
  - A7 a static schedule's destinations exist before its first message: covered [2026-11-05, mesh c38ee188]
  - A9 here, per node: the lists: covered [2026-11-05, mesh c6c9c77f]
  - A9 here, per node: the bindings and the state section: covered [2026-11-09, mesh fdae772b]
  - A9 every address is fixed at compile time: covered [2026-11-09, mesh fdae772b]
  - Appendix item 2: data structures and algorithms (the queue, its size, the merge): covered [2026-11-05, mesh
    930861b0]
  - Appendix item 2: what is ordered, and what is not; interleaving: covered [2026-11-05, mesh 930861b0]
  - Appendix item 2: recorded order: covered [2026-11-03, mesh 0b3e78d7]
  - Appendix item 2: delivery at a node (Fig. 5's destinations; no wait on an earlier call): covered [2026-11-05, mesh
    c38ee188]
  - Appendix item 2: work and transfers run at once: covered [2026-11-12, mesh 0704bf95]
  - Appendix item 2: which files define them: covered [2026-11-08, mesh d44e3d0c]
  - Appendix item 2, change: the queues replace the sequence counters, taken[p] and the one channel: covered
    [2026-11-05, mesh 7eb93c51]
  - Appendix item 2, change: remove the start's wait and the late-loading port's refusal: covered [2026-11-05, mesh
    e1525cc7]
  - Appendix item 2, change: remove the coordination's Q3′: covered [2026-11-05, mesh 71e060ba]
  - Appendix item 3 (a slow node, "desynchronization": no handler): covered [2026-11-12, read at mesh 434b9955]
  - Appendix item 4 (no truncation at one end; no handler): covered [2026-11-09, mesh fdae772b]
  - Appendix item 5: one permutation of the whole set per execution: covered [2026-11-05, mesh c2a226b0]
  - Appendix item 5: a page's round rule reads a count that only rises: covered [2026-11-05, mesh 930861b0]
  - Appendix item 5: joins are sent in the schedule's order: covered [2026-11-04, mesh dd57daa6]
  - Appendix item 6, change: each call launches one execution: covered [2026-11-12, mesh 81c8e9da]
  - The files and functions: rdma/mesh-graph.h, mesh-graph.c: covered [2026-11-05, mesh 15b4e9d2]
  - The files and functions: rdma/mesh-collective.c (routing, u, queue sizes, sets): covered [2026-11-03, mesh
    80e2e4d3]
  - The files and functions: rdma/mesh-bridge.h: covered [2026-11-06, mesh 61435d4b]
  - The files and functions: rdma/mesh-records.c: covered [2026-11-04, mesh 3643de2f]
  - The files and functions: rdma/mesh-flow.c: covered [2026-11-03, mesh 0b3e78d7]

Withdrawn:

  - Appendix item 7 (torch's existing kernels on the GPU): withdrawn [2026-11-12 reading].
    design/aot-compilation-change-request.md:1030-1031: "design/torch-source-lowering-change-request.md item 1
    replaces this item: no torch Metal kernel is submitted or recorded."
  - Appendix item 8 (the heartbeat's holders): withdrawn [2026-11-12 reading].
    design/aot-compilation-change-request.md:1054 is the item's whole text: 'The operator: "settled."' The request
    keeps no requirement under this item, so there is no sentence to read against code.

## [torch-capture-change-request.md](torch-capture-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 26 items outstanding, 0 not determined, 6 covered, 0 withdrawn.

Outstanding:

  - 1, Write, first bullet (every Metal call recorded and replayed in order): "Every Metal call of the captured
    program is recorded, and the replay issues the recorded calls in the order they were recorded (apitrace;
    GFXReconstruct; RenderDoc's "every API call is serialised out in order"), each by the call that
    MTL4ComputeCommandEncoder.h or MTL4CommandQueue.h declares for it." The compile_module chain has no capture and no
    replay: lower_metal.lower (compiler/mesh_compiler/lower_metal.py:1188-1206) builds each processor's kernels and
    dispatches from the scheduled FX graph (Lowering.__init__, :303-345), and the function it returns (:892-905)
    encodes and commits those dispatches anew at every execution through mesh_metal4_commit
    (rdma/mesh-metal4.m:162-206). No Metal call is recorded.
  - 1, Write, second bullet (a call Metal 4 declares no call for): "A call for which Metal 4 declares none invalidates
    the capture (CUDA §4.2.2.1.2.3), and is documented in this request (GFXReconstruct :89)." There is no capture in
    the compile_module chain to invalidate (compiler/mesh_compiler/lower_metal.py:1188-1206 lowers the FX graph; an
    operator with no handler raises NotLowered at :1203-1205, which is not a Metal call).
    design/torch-capture-change-request.md names no such call: its only uses of the word are the quoted sources at
    :391 and :400 and the clause itself at :414.
  - 1, Write, third bullet (pipelines from a copy of the descriptor with supportIndirectCommandBuffers): "Each
    compute-pipeline creation method of MTLDevice.h (:834–864) creates its pipeline from a copy of the program's
    descriptor, with `supportIndirectCommandBuffers` set (GFXReconstruct :102; vulkan_capture_manager.cpp:889–896;
    MTLComputePipeline.h:70–73)." No program's descriptor is taken or copied: rdma/mesh-metal4.m:51-72
    (mesh_metal4_pipeline, called from compiler/mesh_compiler/lower_metal.py:264-267 and :933) makes its own
    MTL4ComputePipelineDescriptor from the lowered source and calls MTL4Compiler
    newComputePipelineStateWithDescriptor. `supportIndirectCommandBuffers` occurs nowhere in mesh rdma/ or compiler/
    at HEAD.
  - 1, "A call the program makes only while a condition holds", Write, first bullet: "The replay of every execution
    issues such a call where it was recorded: every call is serialised, and the replay executes the serialised calls
    verbatim (RenderDoc :21, :30; CUDA's snapshot)." and "Dawn's clear is a recorded fill, replayed at every
    execution; its condition is not evaluated at replay." No call is serialised and nothing is replayed in the
    compile_module chain: each execution runs the lowered function's own program of commits, waits, sends and receives
    (compiler/mesh_compiler/lower_metal.py:892-905); mesh-metal4.m encodes no fill.
  - 1, "A call the program makes only while a condition holds", Write, second bullet (initial contents): "The contents
    at the capture's start of each buffer the captured calls read before writing are saved with the capture (RenderDoc
    :21; GFXReconstruct's snapshot)." and "A buffer the captured calls also write has them applied before each
    execution's replay, so that the execution's writes are undone (RenderDoc :38; `ApplyInitialContents`;
    vulkan_replay_frame_loop_consumer.h:182–183)." There is no capture and no saved contents: archive.save writes each
    processor's FX graph and its state tensors (compiler/mesh_compiler/archive.py:72-79), and Program passes state
    tensors as arguments at each call (compiler/mesh_compiler/runtime.py:31-35, :67-72); nothing is applied before an
    execution to undo its writes.
  - 2, "It takes / It writes" (the IPC graph is a function of the static schedule): "The IPC graph is a function of
    the static schedule". The graph mesh_graph_buffer_bound takes is built at rdma/mesh-collective.c:1898-1911 from
    each struct mesh_function's parts, host, work and edge; rdma/mesh.py:313-314 (mesh.function, called from
    schedule.py:108) passes host, work and edge as None and parts and edges as 0, so it is one actor of times 0 with
    no dependence. It is computed inside mesh_plan before the plan is solved, and driver.py:24 (planned) runs before
    driver.py:25 (schedule) and :27 (argument_buffers); argument_buffers' result is not an operand of it.
  - 2, "The number of locations of a value the CPU writes", Write: "The value's number of locations is Bff(e): the
    smallest m for which adding d_m(snk(e), src(e)) leaves λ_max unchanged, found by the binary search from (6)'s
    value, with t(v) the execution time estimates." In the compile_module chain no t(v) is given: rdma/mesh.py:313-314
    passes host and work as None and parts as 0, so rdma/mesh-collective.c:1900-1902 leaves the one actor's host and
    work times 0, λ_max is 0 and mesh-graph.c:1581 sets bound 1 without a search. schedule.py:149, :189 and :191 give
    every value lag 0, so mesh-collective.c:1921 writes no value's lag; a value's number of locations is p->copies =
    horizon × calls raised by mesh_graph_lags (mesh-collective.c:380-412, :197), not a Bff from estimates. "the CPU's
    write of a value for execution t is src(e) and the GPU's reads of it snk(e) of a feedforward edge e of the node's
    IPC graph": no CPU write or GPU read of a value is given to that graph by the chain (no parts, no edges).
  - 3, torch scope: "**In the torch scope,** design/torch-source-lowering-change-request.md item 1 writes this item
    (plan-to-bridge-change-request.md R4: "A function that takes pages in any order has one task for each page.")." No
    function in the compile_module chain has one task for each page: compiler/mesh_compiler/devices.py:35 types every
    arrival's consumer JOIN and nothing in compiler/mesh_compiler sets PAGES; lower_metal.receive
    (compiler/mesh_compiler/lower_metal.py:824-828) makes one RECV entry for a channel, and its read (bridge.py:79-82)
    waits in mesh_host_wait for every page of the transfer (rdma/mesh-collective.c:28-48, :66-84) before the function
    reads any.
  - 3, LiteRT scope, Write, first bullet (index space analysis): "A recorded kernel's source is the `source` of the
    recorded `newLibraryWithSource:options:error:` call (item 1 records every call)." and "The index function of each
    of its loads and stores is computed from that source by use-def chains; a kernel whose index functions are affine,
    a · (W0 · w0 + l0), is a contiguous kernel, and any other is discontiguous (Lee §3.2.2)." No call is recorded in
    the compile_module chain and no kernel source is analysed: kernel source is emitted by
    compiler/mesh_compiler/lower_metal.py:927-935 from the FX graph and compiled by MTL4Compiler
    newLibraryWithDescriptor (rdma/mesh-metal4.m:60); nothing classes a kernel contiguous or discontiguous
    (`newLibraryWithSource` occurs nowhere in mesh rdma/ or compiler/).
  - 3, LiteRT scope, Write, second bullet (the pages a threadgroup reads): "The pages a threadgroup of a contiguous
    kernel reads are the partitions its index functions' values over its threads fall in ("Partitions are contiguous
    in memory")." and "A threadgroup of a discontiguous kernel reads every partition, as Lee transfers "the entire
    input" for one." The compile_module chain computes no pages per threadgroup: a received operand is read whole
    after mesh_host_wait returns for the transfer (compiler/mesh_compiler/bridge.py:79-82; lower_metal.py:899-900).
  - 3, LiteRT scope, Write, third bullet (first dispatch predicated by the mask): "Each recorded dispatch that reads a
    received operand is first executed predicated by the mask (Russell; RISC-V V).", "Each threadgroup tests the
    presence bits of the pages it reads (I-structures' "presence bits"), and no other.", "A threadgroup any of whose
    presence bits shows its page absent returns before any store and writes 1 to its element of the mask." and "It
    passes or returns as a unit, its threads' results joined by `simd_ballot` (MSL §6.10.1 and §6.10.2)." No dispatch
    is recorded or predicated: the kernels lower_metal emits (compiler/mesh_compiler/lower_metal.py:927-935) take no
    mask and no presence bits, `simd_ballot` occurs nowhere in mesh rdma/ or compiler/, and the presence fields are
    tested on the CPU by mesh_host_arrived (rdma/mesh-collective.c:28-48) for the whole transfer before the reader's
    dispatches are committed.
  - 3, LiteRT scope, Write, fourth bullet (stream compaction into the worklist): "A stream compaction after that
    dispatch packs the indices of the threadgroups whose mask element is 1 into a dense list by a scan and a scatter,
    and writes the list's length as the threadgroup count of the dispatch that executes the list (Harris, Sengupta and
    Owens; Blelloch)." and "The list is the worklist of a data-driven implementation (Nasre, Burtscher and Pingali)."
    No mask, compaction or worklist exists in the compile_module chain; every dispatch's grid is fixed at lowering
    (compiler/mesh_compiler/lower_metal.py:1006-1007; rdma/mesh-metal4.m:188-193).
  - 3, LiteRT scope, Write, fifth bullet (second dispatch executes the worklist): "A second dispatch of the kernel
    executes the worklist's threadgroups, its threadgroup count read from an indirect buffer
    (MTL4ComputeCommandEncoder.h:90–112)." and "Each of its threadgroups tests its presence bits again."
    rdma/mesh-metal4.m:190-193 encodes only dispatchThreadgroups:threadsPerThreadgroup: and
    dispatchThreads:threadsPerThreadgroup: with counts from struct mesh_metal4_dispatch; no dispatch takes an indirect
    buffer and no second dispatch of a kernel is made.
  - 3, LiteRT scope, Write, sixth bullet (task list in order of completion times): "The dispatches that execute
    worklists are a task list (Graham) in the order of non-decreasing completion times of the transfers they read, the
    times `mesh_plan`'s linear program computes, "where ties are broken by choosing an order that is consistent with
    the precedence relation" (Hall et al.)." There are no dispatches that execute worklists in the compile_module
    chain; what schedule.py:201-214 orders is the FX graph's functions.
  - 4, "It takes / It writes" (the event's value): "**It writes** the event's value, which the command queue's
    `waitForEvent:value:` reads." Nothing in the compile_module chain takes the starts and writes a Metal event's
    value. The one `waitForEvent:value:` (rdma/mesh-metal4.m:141) waits on a stream's `arrival` event, which torch's
    MPS command buffer signals with the stream's own counter (mesh-metal4.m:133-140; lower_metal.py:1038-1039); the
    other events are each commit's record, set in its feedback handler (mesh-metal4.m:203). Execution t's requests are
    awaited on the CPU by mesh_host_wait on the plan's count and completion words (bridge.py:47, :80;
    rdma/mesh-collective.c:66-84).
  - 4, "The value the event is set to at a failure", Write: "at a failure the event is set to UINT64_MAX, the greatest
    value of its `uint64_t`, as `complete_all` sets `done` to UINT_MAX: every wait on it, present and future, for any
    value, is satisfied (await(E, v) returns once E ≥ v), with no count of executions." and "The event is not
    signalled again (`reinit_completion` is `complete_all`'s condition for reuse; a revoked object "can never be
    repaired"); the next view's requests take a new event." No event is set at a failure in the compile_module chain:
    the only event values written are a stream's arrival counter (rdma/mesh-metal4.m:134-138) and a commit record's
    value in its feedback handler, set whether or not feedback.error is set (mesh-metal4.m:199-204); no event stands
    for a view's requests, so none is replaced at the next view.
  - 5, Write (the release's one 64-bit atomic write): "the fence, then one 64-bit atomic write of the word,
    `atomic_max_explicit` with t + 1, the 64-bit atomic write the header declares for it." No kernel source in the
    compile_module chain writes the word: compiler/mesh_compiler/lower_metal.py emits no release kernel, and
    `atomic_max_explicit` occurs nowhere in mesh rdma/ or compiler/. The write is mesh_host_publish's compare-exchange
    loop (rdma/mesh-collective.c:22-24: a relaxed load, then atomic_compare_exchange_weak_explicit with
    memory_order_release until the word is at least execution + 1), with no fence before it.
  - 6, Write, first bullet (offsets by Greedy by Size): "Each operand of a function typed `contiguous_input` or
    `contiguous_output` is given an offset in one memory block by Greedy by Size for Offset Calculation, as TensorFlow
    Lite's `GreedyBySizeAssignment` states it." No function or operand in the compile_module chain is typed
    `contiguous_input` or `contiguous_output` (the names occur nowhere in compiler/ or rdma/ at HEAD; devices.py:35
    types every arrival's consumer JOIN). compiler/mesh_compiler/schedule.py:149, :189 and :191 set `run` to 1 on
    every value they declare, so every value of the plan is given an offset in the block, whatever function reads or
    writes it.
  - 7, Write, first bullet (a captured buffer's interval): "A captured buffer's interval runs from its first write to
    its last read." There is no capture and no captured buffer in the compile_module chain; the positions of
    schedule.py:169-171 are those of the scheduled FX functions, not of a capture's commands. A value a transfer names
    is given the interval ending at the last position, len(local) − 1, not at its last read (schedule.py:179-182).
  - 7, Write, second bullet (a call's inputs and results): "A call's input's interval starts at the capture's first
    command, as a placeholder's `start = 0` (ExecuTorch), and ends at its last read." and "A call's result's interval
    starts at its first write and ends at the capture's last command, the output node being the graph's last
    (torch.fx) and a graph output "never ... scheduled for deallocation" (TensorFlow Lite)." There is no capture in
    the compile_module chain. In schedule.stored a placeholder that is state is declared INVARIANT with begin and end
    0 (compiler/mesh_compiler/schedule.py:188-189), a placeholder that is a user input is given no value (:190-193),
    and a result is given no value (:185, `root in results`), so neither a call's input nor its result has an interval
    of that form.
  - 8. The torch scope's final lowering: "**Replaces:** the torch scope's kernels, which are torch's MPS kernels, one
    ATen operator at a time (mesh 03297e52)." They are not gone: rdma/mesh_compile.py is at HEAD (8643 lines; `_apply`
    at :2363, `program` at :8242), and in the compile_module chain runtime.Program runs the scheduled FX GraphModules
    themselves where no lowered function is given (compiler/mesh_compiler/runtime.py:15, :50, :72, values moved to the
    processor's device at :54-55), which on an mps device are torch's kernels one operator at a time. "**Written** in
    design/torch-source-lowering-change-request.md, items 1–3." Item 1's typing there (:169-173, one task per page for
    the consumer that takes pages in any order) is not in the chain: compiler/mesh_compiler/devices.py:35 types every
    consumer JOIN.
  - 9. Every node replays: "every node runs its captured program by replay, its collectives included." No node of the
    compile_module chain has a captured program or a replay: Program calls each processor's function from Python at
    every execution (compiler/mesh_compiler/runtime.py:50, :72), and a Metal function encodes its dispatches at each
    call (lower_metal.py:892-905). "fcf5d18a's calls of torch from Python at every execution are deleted."
    rdma/mesh_compile.py, the file fcf5d18a changed, is at HEAD with `program` (:8242) and `_replayed` (:7660, :8620);
    and runtime.py:15 runs torch FX modules from Python at every execution where no lowered function is given.
  - Source for item 2's execution-time estimates, Write, first sentence (the lowering supplies the cost model's
    operands): "the target lowering supplies the operations, memory traffic and execution mapping of each compiled
    function to the analytical cost model, with the machine's rates and primitive costs as operands." No lowering in
    compiler/mesh_compiler supplies them and none calls the model: `actor_time`, `Actor` and `Machine` occur nowhere
    in compiler/ at HEAD; lower_metal.lower returns the functions alone
    (compiler/mesh_compiler/lower_metal.py:1188-1206). The chain's cost operand is compile_module's `cost`, by default
    unit_cost, which returns 1.0 for every task (compiler/mesh_compiler/driver.py:14; schedule.py:26-27).
  - Source for item 2's execution-time estimates, Write, second sentence (CPU and GPU rates): "CPU work, including the
    CPU's write of item 2, uses its CPU rates and traffic; a GPU function uses its GPU mapping and rates." Nothing in
    the compile_module chain calls mesh_graph_actor_time or gives it a CPU or a GPU record: place, schedule and lower
    make no such call (no `actor_time` in compiler/), and the times mesh_plan's buffer bound takes for host and device
    parts are 0 (rdma/mesh.py:313-314; rdma/mesh-collective.c:1900-1902).
  - Source for item 2's execution-time estimates, Write, third and fourth sentences (estimates stored; Bff supplies
    locations and reuse edges): "The estimates are stored with the compiled functions and are the t(v) of item 2's
    λ_max and Bff(e)." No estimate is stored in the compile_module chain: ir.Scheduled and ir.Plan have no field for
    one (compiler/mesh_compiler/ir.py:54-83), archive.save writes none (archive.py:52-71), and rdma/mesh.py:313-314
    passes host and work as None and parts as 0, so the t(v) of the bound are 0. "The resulting Bff(e) supplies the
    value's locations and the native program's corresponding reuse edges." With those operands the bound is 1
    (rdma/mesh-graph.c:1581) and no value's lag is written (schedule.py:149, :189, :191 give lag 0;
    mesh-collective.c:1921); the number of locations comes from horizon × calls raised by mesh_graph_lags
    (mesh-collective.c:380-412).
  - Source for item 2's execution-time estimates, Write, last sentence (transformations update costs):
    "Transformations update the costs of their resulting functions along with their effects." No function of the
    compile_module chain carries a cost for a transformation to update: placement, devices, schedule and the lowerings
    write none on a node or a function (the only cost is the caller's `cost` callable,
    compiler/mesh_compiler/driver.py:14, :24-25, applied to nodes at schedule.py:153 and :203), and driver.lowered
    replaces modules by lowered functions without any cost (driver.py:32-40).

Covered:

  - 2, Write (when execution t's requests into a location start): covered [2026-11-12, mesh 81c8e9da]
  - 4, Write, first bullet (execution t's waits; no fixed count): covered [2026-11-12, mesh 81c8e9da]
  - 4, Write, second bullet (the waits end at the free): covered [2026-11-12, mesh 81c8e9da]
  - 6, Write, second bullet (mesh_view_compose's remapping deleted): covered [2026-11-08, mesh b880b808]
  - 6, "Where the block lies, and the usage records of the u locations", Write, first bullet (the block is its own
    allocation): covered [2026-11-12, mesh 81c8e9da]
  - 6, "Where the block lies, and the usage records of the u locations", Write, second bullet (u usage records;
    do_arcs_overlap on the ring): covered [2026-11-12, mesh 81c8e9da]

## [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 29 items outstanding, 5 not determined, 12 covered, 0 withdrawn.

Outstanding:

  - 1 / Write: no torch Metal kernel: "**Write:** no torch Metal kernel is ever submitted or recorded again."
    runtime.py:15 runs scheduled.modules, torch.fx GraphModules of ATen operators, when no functions are given, on the
    processor's torch device (driver.py:16, runtime.py:55); examples/tp.py:198 and compiler/samples.py:770, :1146 call
    Program(scheduled) so. lower_inductor.py:70-72 compiles each processor's module with torch's inductor for
    scheduled.devices[processor]. lower_metal.py:76-79 and :1038-1039 commit torch's MPS command buffer
    (torch::mps::commit) and order the stream after it when an argument is on mps.
  - 1 / Write: the task graph: "A vertex is a task: a node of the graph the operator lowering produces." There is no
    operator lowering: driver.py:18-19 passes export's graph straight to place, and the nodes are those torch's
    `program.run_decompositions()` leaves (export.py:20). "An edge is a page passed between tasks." The edges are the
    fx nodes' operand edges and each channel's send-to-recv pair, each a whole tensor (schedule.py:39-41); pages
    appear only as the sections mesh_plan returns for carried values (schedule.py:116-117).
  - 1 / Write: the operator lowering produces them on every backend: "The operator lowering produces these vertices
    and edges, for each function on whichever Apple silicon backend it is placed, the CPU included." compile_module
    has no operator-lowering step (driver.py:18-19); the only decomposition is torch's `run_decompositions()` with no
    table of mesh's (export.py:20).
  - 1 / Write: the synchronous, complete, contiguous consumer: "the synchronous, complete, contiguous consumer is one
    task with an edge from every page of its operand, and only its operand, typed `contiguous_input` or
    `contiguous_output`, is one run of pages;" No `contiguous_input` or `contiguous_output` type exists in
    compiler/mesh_compiler or rdma/mesh-collective.h; devices.py:35 marks every arrival JOIN whatever its consumer is,
    and the node's edge is the one recv value (schedule.py:39-41), not an edge per page.
  - 1 / Write: the asynchronous, incomplete, noncontiguous consumer: "the asynchronous, incomplete, noncontiguous
    consumer is one task per page, each with an edge from its own page, and its outputs are pages that leave as each
    is produced;" Nothing in compiler/mesh_compiler assigns PAGES or makes a task per page: devices.py:35 sets JOIN on
    every arrival and each recv is one node (placement.py:678-680).
  - 2 / Write: the operator lowerings, then placement, consume it: "The operator lowerings (items 1, 5 and 6) and then
    placement consume it." No operator lowering runs between export and place: driver.py:18-19; compiler/mesh_compiler
    has no function that takes the exported graph and the placement operand and returns nodes of that graph before
    place.
  - 3 / Replaces: "and the second linear program over the GPU's commands (metal-microbench 3b719f10, `MetalOrder`)."
    metal-microbench HEAD (0b692ed8) still has it: metal_recording.m:201 `static void MetalOrder(MetalCapture
    *capture)`, called at metal_recording.m:719.
  - 3 / Write: the algorithm for the order: "The algorithm for the order is the completion-time linear program, which
    minimizes C_max and then ΣC_j over the completion times C_j [...]" schedule.py:203 computes every node's key with
    mesh_graph.longest (the longest path of costs); only the recv nodes of carried channels are replaced by
    mesh_plan's times (:204-206), and where nothing is carried between machines plan is None (:128-129) and no linear
    program is solved.
  - 3 / Write: same threadgroup: "same threadgroup: a threadgroup barrier;" rdma/mesh_graph.py:550-560 (orderings)
    returns three kinds only, an intra-pass barrier, a fence and a wait; the vertices of the relation are whole
    dispatches (lower_metal.py:396-398), and no kernel compiler/mesh_compiler writes contains a threadgroup barrier.
  - 5 / Write: one node, one LAPACK driver: "The algorithm is the LAPACK driver for the node's problem, called through
    Accelerate." and "`torch.linalg.svd`, `eigh`, `eig` and `eigvals`, and `lstsq` with the gelsd and gelss drivers,
    are each one such node." compiler/mesh_compiler calls no LAPACK driver: the native source includes only cblas
    (lower_metal.py:45-46), the lowering's handlers hold no linalg operator (lower_metal.py:306-319), and such a node
    is reported as missing (:332-334, :1204-1206). Whatever node these operators are in the graph is what torch's
    run_decompositions leaves (export.py:20). The LAPACK nodes are in rdma/mesh_compile.py only.
  - 5 / Write: it is an operator lowering: "This is an operator lowering. It produces that node in the exported graph"
    compile_module has no operator lowering (driver.py:18-19, export.py:20), and nothing in compiler/mesh_compiler
    produces the node.
  - 5 / Write: INFO: "The routine's `INFO` is a value the node returns; a positive `INFO` is reported." No node of the
    compile chain calls a LAPACK routine or returns or reports an INFO (lower_metal.py:306-319; nothing in
    compiler/mesh_compiler names INFO).
  - 5 / Write: pinv: "`torch.linalg.pinv` is the composition PyTorch states, of the node for `svd` or for `eigh`."
    compiler/mesh_compiler writes no composition for pinv and has no node for svd or eigh; the only decomposition is
    torch's run_decompositions (export.py:20).
  - 5 / Write: `_lstsq_metadata`: "`_lstsq_metadata` takes the four drivers PyTorch names, with the output shapes
    PyTorch gives each." `_lstsq_metadata` exists only at rdma/mesh_compile.py:3953-3962; the compile_module chain has
    no such function and does not import that file.
  - 5 / Write: the plain message: "Every operator whose method, on the processor the placement names, needs an
    iteration depth that tensor data decides gets, from the operator lowering, one plain message that names the
    operator and says that an iteration depth that depends on data is not compiled [...]" There is no operator
    lowering (driver.py:18-19). An operator the Metal or native lowering has no handler for is reported by
    lower_metal.py:332-334 and :1204-1206 as `processor: operator names`, with no statement about iteration depth;
    export.py:29-31 speaks of "data-dependent control flow" and only for higher order operators.
  - 6 / Write: nodes of the exported graph on every processor: "This is an operator lowering. Each operator below
    becomes nodes of the exported graph (item 2): matrix multiplications, QR factorizations, Cholesky factorizations,
    triangular solves and selections among computed values." compile_module has no operator lowering (driver.py:18-19,
    export.py:20) and nothing in compiler/mesh_compiler turns svd, eigh, pinv, lstsq or matrix_exp into such nodes;
    the lowering has no handler for QR, Cholesky or a triangular solve (lower_metal.py:306-319).
  - 6 / Write: the polar decomposition: "The algorithm for the polar decomposition is QDWH as `_qdwh` states it
    (jax/_src/tpu/linalg/qdwh.py)." and the sentences after it on its coefficients, steps, returned convergence value,
    refinement and Hermitian factor: compiler/mesh_compiler has no polar decomposition, no QDWH step and no
    `max_iterations` operand (no file there names them; lower_metal.py:306-319).
  - 6 / Write: eigh: "The algorithm for `eigh` is QDWH-eig (Nakatsukasa and Higham, §3.1), in which a polar
    decomposition splits the matrix into two blocks that share its eigenspaces below and above a split point, and the
    same is done to each block." compiler/mesh_compiler has no lowering of eigh (lower_metal.py:306-319; export.py:20
    is the only decomposition).
  - 6 / Write: svd, pinv, lstsq: "The algorithm for `svd` is Algorithm 2 (Nakatsukasa and Higham, §4): the polar
    decomposition, the eigendecomposition of its Hermitian factor, and one product." and "`pinv` and `lstsq` with
    gelsd and gelss are the compositions PyTorch states (item 5), of that `svd` or `eigh`." compiler/mesh_compiler has
    none of them (lower_metal.py:306-319; driver.py:18-19).
  - 6 / Write: matrix_exp: "The algorithm for `matrix_exp` is scaling and squaring as `_calc_P_Q`, `_squaring` and
    `expm` state it." and the sentences after it on `n_squarings`, `max_squarings`, the selections, the NaN result and
    the reported runtime error: compiler/mesh_compiler has no lowering of matrix_exp and no `max_squarings` operand
    (lower_metal.py:306-319; driver.py:14).
  - 6 / Write: the CPU for the whole operator: "Where the placement operand names the CPU for the whole operator, the
    operator is item 5's node." The compile chain has no item 5 node: on the CPU the native lowering reports such an
    operator as missing (lower_metal.py:332-334, :1190, :1204-1206).
  - 6 / Write: no body repeated by tensor data: "No lowering here repeats a body a number of times that tensor data
    decides." The lowerings this sentence is about, item 6's, are not in compiler/mesh_compiler
    (lower_metal.py:306-319), so no code of the compile_module chain is what it describes.
  - 1 / Write: the IR carries the kind: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "the IR carries
    which of the two each consumer is (design/aot-compilation-change-request.md A8)."
    (design/torch-source-lowering-change-request.md:173): compiler/mesh_compiler/devices.py:35 sets
    `arrival.meta["consumer"] = JOIN` on every arrival whatever its consumer is; PAGES (ir.py:34) is assigned nowhere
    in compiler/mesh_compiler, so the value says nothing about which of the two a consumer is. It is read once
    (schedule.py:152) and only for channels carried between machines; archive.py:63-66, :106-111 save and restore the
    block and subarray keys of a recv but not `consumer`. The cited A8 has a clause outstanding in the same reading
    ("A8 data structure (the type mesh_plan takes)").
  - 3 / Write: one schedule stage: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "This is the schedule
    stage, and there is one schedule. It takes the placed graph and the weighting operand
    (design/aot-compilation-change-request.md A3)." (design/torch-source-lowering-change-request.md:241-242):
    compiler/mesh_compiler/driver.py:24 and :26 solve mesh_plan twice (schedule.py:160 `solved` in planned, :196
    `solved` in stored, each mesh_plan + mesh_plan_save at :111-113). The order of driver.py:25 reads the first
    solve's times (schedule.py:206); the record the archive and the bridge take is the second's (schedule.py:118,
    archive.py:62, bridge.py:25-27). The cited A3 has a clause outstanding in the same reading ("The caller passes the
    operand": no caller of compile_module passes `cost`; default unit_cost at driver.py:14).
  - 3 / Write: commands written in that order: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "Each
    processor's commands are written in that order." (design/torch-source-lowering-change-request.md:250):
    compiler/mesh_compiler/lower_metal.py:367 with :384-395: a product or fused value with one user is deferred, and
    its dispatch is issued when the walk reaches that user (:413-415 reader -> :400-408 view -> :405 multiply / :407
    store -> :417-424 dispatch). The command of a function at place k is then written after the commands of the
    functions scheduled between it and its user. lower_inductor.py:72 hands each processor's module to torch.compile's
    inductor, which writes the kernels in its own order.
  - 3 / Write: Sriram's synchronization graph: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "Only the
    edges of a synchronization graph that preserves the inter-processor communication graph are written"
    (design/torch-source-lowering-change-request.md:251-252): compiler/mesh_compiler/lower_metal.py:1014-1019: at the
    end of each run the thread waits on every commit not yet waited (`range(len(segments)) if entry[0] is WAIT`),
    whether or not an edge reaches the host from it (run at :1054-1055, rdma/mesh-metal4.m:208-212).
    lower_metal.py:1076 with :1196-1197 also writes a queue wait on torch's command stream for an mps argument
    (mesh-metal4.m:136-141), which is no edge of the relation computed at :999-1007.
  - 3 / Write: same queue: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "same queue: a fence;"
    (design/torch-source-lowering-change-request.md:257): rdma/mesh-metal4.m:35 and :42: one static MTL4CommandQueue
    takes every stream's commits (:205). compiler/mesh_compiler/lower_metal.py:1443-1452 makes one Metal lowering,
    with its own stream (:1144), for each processor on mps, and compiler/samples.py:431 and :484 place two processors
    on mps. A dependence between two such processors has both ends on that one queue and is written as a send and recv
    with the thread's wait on the commit's shared event (lower_metal.py:1015, rdma/mesh_graph.py:553-554,
    mesh-metal4.m:212), not a fence. Fences are made only between two commits of one processor's stream
    (mesh_graph.py:555-556, lower_metal.py:1008-1009).
  - 3 / Write: independent pairs: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "Independent pairs get no
    ordering, and Metal then permits them to overlap." (design/torch-source-lowering-change-request.md:259):
    rdma/mesh-metal4.m:175-176 encodes each waitForFence at the start of the later commit's encoder, before all its
    dispatches, and :195 the updateFence after all of the earlier commit's dispatches, so one dependent pair orders
    every dispatch of the two commits. :180-183 the intra-pass barrier orders every earlier dispatch of the encoder
    before every later one. compiler/mesh_compiler/lower_metal.py:1015-1019 with :1054-1055 block the thread on a
    commit's event before a send, so later commits with no dependence on it are not issued until it completes.
  - 5 / Write: the cited functions are the CPU's: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "Item 6
    states what these operators are on the other processors."
    (design/torch-source-lowering-change-request.md:312-313): The sentence cites item 6, every clause of which is
    outstanding or not determined in the same reading. compiler/mesh_compiler has no lowering of svd, eigh, eig,
    eigvals, pinv, lstsq or matrix_exp on any processor: the handlers of lower_metal.py:322-341 hold none, the native
    drivers are solve and qr only (:1230-1231), and such a node is reported as missing (:354-356, :1457-1459).

Not determined:

  - 5 / Write: which routine: not determined [2026-11-12]. Both sentences state what the quoted sources and
    Accelerate's header name; they name nothing a file of either repository does. No code in compiler/mesh_compiler
    calls gelsd, gelss or any other LAPACK routine (lower_metal.py:45-46, :306-319).
  - 5 / Write: matrix_exp: not determined [2026-11-12]. The first sentence states what LAPACK has and the second
    points to item 6; neither names anything a file does. Item 6's matrix_exp entry is returned separately.
  - 6 / Write: the Halley continuation: not determined [2026-11-12]. The sentence states that no quoted source gives
    the number of steps to compile; it names nothing a file does. compiler/mesh_compiler has no QDWH code.
  - 6 / Write: eig and eigvals off the CPU: not determined [2026-11-12]. The sentence states that no quoted source
    gives these on the GPU or the Neural Engine; it names nothing a file does. compiler/mesh_compiler has no lowering
    of eig or eigvals.
  - 6 / Write: the Neural Engine: not determined [2026-11-12]. The sentence states that no quoted source says which
    operations the Neural Engine executes or at what precision; it names nothing a file does.

Covered:

  - 1 / Replaces: covered [2026-11-09, mesh dd2062d2]
  - 2 / Write: the exported graph: covered [2026-11-10, mesh ce2e8ac9]
  - 2 / Write: the export stage produces it: covered [2026-11-10, mesh ce2e8ac9]
  - 2 / Write: dynamic control flow is not compiled: covered [2026-11-10, mesh ce2e8ac9]
  - 2 / Write: a dimension known only at run time: covered [2026-11-10, mesh ce2e8ac9]
  - 3 / Write: each function's processor: covered [2026-11-11, mesh 297c3cda]
  - 3 / Write: each task's length: covered [2026-11-12, mesh 81c8e9da]
  - 3 / Write: the order is published: covered [2026-11-12, mesh 81c8e9da]
  - 3 / Write: same encoder: covered [2026-11-12, mesh 7d8092e1]
  - 3 / Write: another queue, another device or the CPU: covered [2026-11-12, mesh 7d8092e1]
  - 5 / Replaces: covered [2026-11-09, mesh 6904370b]
  - 5 / Write: no body repeated by tensor data: covered [2026-11-12, mesh a86c7dc4]

## [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 14 items outstanding, 0 not determined, 6 covered, 0 withdrawn.

Outstanding:

  - 1 / The algorithm: "**The algorithm** is the graph module's: `mesh_graph_dependences`,
    `mesh_graph_synchronization`, `mesh_graph_redundant` and `mesh_graph_schedule`." mesh_graph_dependences
    (rdma/mesh-graph.c:1678) and mesh_graph_schedule (rdma/mesh-graph.c:1977) have no caller in mesh: the lowering's
    dependence relation is mesh_graph_dependences_views (rdma/mesh_graph.py:532-538, called at lower_metal.py:844),
    and the order is mesh_graph.longest and mesh_graph.heap_order (compiler/mesh_compiler/schedule.py:203, :208).
    mesh_graph_signature itself calls mesh_graph_redundant and mesh_graph_lags (mesh-graph.c:2306, :2314, :2321).
  - 1 / Produced by: "**Produced by** the AOT compilation, from the lowered IR, and nothing else writes it." Other
    writers at HEAD: metal-microbench metal_recording.m:2939 (mesh_graph_signature), :2947, :2956 and
    tools/mesh_signature.h:22-25 (mesh_graph_record); mesh rdma/mesh-collective.c:1101 (plan_signature calls
    mesh_graph_signature for the plan); bin/mesh-sync-monitor.py:390-391 (two signatures built from the record's and
    the observed operations). Within the compile chain, signature.py:33-44 (torch_run) writes the operations of a
    processor whose function has no record from scheduled.modules, not from a lowered function, and signature.py:61-87
    (handed) writes the threads' queue waits and signals from the scheduled value's fields.
  - 4 / Write: the D program stops the process: "When it sees an operation the signature doesn't have at that point in
    the order of that command queue, thread or bridge, a destructive action stops the process at that call." For the
    bridge no destructive action is taken: its raises are printed by a second D program with no destructive option
    (bin/mesh-sync-monitor.py:176-182, :360) and compared with the compiled order after the run (:188-211, :383-385);
    the monitor refuses a target that holds the verbs device (:347-351). With `--observe` the action is `exit(1)`
    without `stop()` (:120, :403-405), and compiler/samples.py:1100-1104 passes `--observe` when `dtrace -w` is
    refused.
  - 4 / Write: the digest and the review record: "each measured run's digest is appended to the review record
    (metal-microbench output_data/change-request-reviews.jsonl)" Only bin/mesh-sync-monitor.py appends a digest. The
    measured runs of examples/tp.py (:126, :138, :197-198), examples/engines.py (:111-113) and the other examples call
    compile_module and Program and compute no signature or digest; no file under examples/ names the review record.
  - 5 / Write: a program with no wait: "For a program whose static schedule has no wait, the signature holds only the
    method's own operations." Whatever the schedule's waits are, the lowering records a host_signal for every commit
    (lower_metal.py:922) and, at the end of each run, a host_wait of the thread on every commit not yet waited
    (lower_metal.py:860-865 `range(len(segments)) if entry[0] is WAIT`, :925); signature.py:73-79 adds the threads'
    argument-queue waits and signals. "In the 252 tok/s replay (metal-microbench 864a0ae4), that was one submission
    per run." That replay is metal-microbench tools/e2b_replay.m with metal_recording.m; the compile_module chain has
    no run of it.
  - 6 / Write: "They belong to the compiled kernels in the artifact, so they are counted there, statically, as part of
    the same signature." The compile chain writes no operation on a kernel processor: lower_metal.py:917-925 emits
    only queue and host operations, signature.py:105-115 makes only the kinds found in those records plus host and
    bridge, and nothing in compiler/mesh_compiler reads a kernel's source for barriers, fences or atomics.
  - 1 / It holds: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "**It holds**, for each of the command
    queue's command stream, the thread and the bridge, and for each execution: the ordered list of synchronization
    operations: submissions, queue signals and waits, encoder fence updates, fence waits and barriers, and the
    thread's waits on a shared event;" (design/synchronization-signature-change-request.md:93-95):
    rdma/mesh-metal4.m:136-141 (mesh_metal4_after, called from compiler/mesh_compiler/lower_metal.py:1196-1197) issues
    two commits of torch's MPS command stream, an encodeSignalEvent:value: on torch's command buffer and a
    dispatch_sync, then waitForEvent; the record holds only the queue_wait for it (lower_metal.py:1076), so that queue
    signal and those submissions are in no list. compiler/mesh_compiler/signature.py:100-101 gives a processor whose
    function has no `record` (an fx module or an inductor function, lower_inductor.py:72, also on an mps device)
    torch_run's record (signature.py:33-44), which holds only host send/recv/copy operations and no lane for its
    command queue (signature.py:105). compiler/mesh_compiler/bridge.py:45-47 waits on every transfer at each
    execution's end (mesh_host_wait, rdma/mesh-collective.c:79) and signature.py has no operation for it.
  - 1 / Consumed by: the monitor's D program: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "the monitor's
    D program is generated from it (item 4)" (design/synchronization-signature-change-request.md:103-104): The
    sentence cites item 4, whose clauses "the D program stops the process" and "the digest and the review record" are
    outstanding in the same reading (reread.json). In code: the bridge's D program is fixed text taking only a group
    number, not generated from the signature (bin/mesh-sync-monitor.py:176-182, written at :358-359); the generated
    program is made only from the operations of class 'program' on one node (:88-97 sequences, :313-320), so the
    signature's feedback, bridge, kernel and other-node operations are not in it (:70-80).
  - 1 / Consumed by: SHA-256 in the artifact: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "its SHA-256
    is part of the compiled artifact" (design/synchronization-signature-change-request.md:104):
    compiler/mesh_compiler/archive.py:78-79 writes signature.record only `if scheduled.signature`; compile_module
    returns the scheduled value with signature None (driver.py:28-29, ir.py:83), and only driver.compiled
    (driver.py:44-48) sets it. Artifacts are saved from compile_module's value with no digest at
    compiler/samples.py:729 and at :849 -> :826 (save_for_review, the emitted archive). No file under examples/
    computes a signature or saves an archive. The per-processor .pt2 files and their scheduled.json (archive.py:52-77)
    hold no digest.
  - 3 / Write: the monitor is DTrace: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "Its Objective-C
    provider catches every Metal 4 call by selector, whatever class implements it. Its pid provider catches the
    thread's semaphore waits and shared-event waits." (design/synchronization-signature-change-request.md:130-131):
    bin/mesh-sync-monitor.py:18-38 probes eight named Metal 4 selectors only; MTL4CommandQueue's `-commit:count:`
    (MTL4CommandQueue.h:234, macOS 26.4 SDK) is not probed (the pattern at :19 is `-commit?count?options?:`), nor
    signalDrawable:/waitForDrawable:, so a submission by `commit:count:` is not caught. The shared-event wait is
    caught by the Objective-C provider (:30 `objc$target::-waitUntilSignaledValue?timeoutMS?:`), not the pid provider.
    The pid probe for semaphore waits (:42) fires only on a thread already bound to a compiled order and outside the
    interpreter lock and any probed call (:170 `/self->bound && self->inapi == 0/`).
  - 4 / Write: at the end: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "At the end, any count that
    differs fails the run" (design/synchronization-signature-change-request.md:141): bin/mesh-sync-monitor.py:381,
    :390-394 compare only the operations of classes 'program' and 'feedback' on the one node observed (:70-80, :313).
    The bridge's raises are counted only with --bridge (:382-389), other nodes' operations and kernel records never;
    with those left out the result is 'pass-scoped' (:406) and the run exits 0 (:418). compiler/samples.py:1123-1124
    runs it without --bridge. Differing counts themselves are only printed (:395-399); the failure is the digest's.
  - 4 / Write: every edge the image of a dependence, none redundant: outstanding [2026-11-12, mesh 6d379b7e; a second
    reader]. "every edge has to be the image of a data dependence between two of the command queue, the thread and the
    bridge" and "So a wait with no dependence under it shows before anything runs."
    (design/synchronization-signature-change-request.md:145-147): bin/mesh-sync-monitor.py:323-327 fails before
    running only for edges in state 'false' and for redundant edges. An edge whose dependence has another delay is
    'stronger', not the image (rdma/mesh-graph.c:2253), and passes. A wait with no signal at all is 'unmatched'
    (mesh-graph.c:2226, :2243) and passes; it appears only inside the printed states dictionary (monitor :330-333).
    The compile chain makes one itself: the queue_wait on torch's arrival (compiler/mesh_compiler/lower_metal.py:1076)
    has no signal operation in the signature.
  - 5 / Write: replay with the monitor: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "Replay a captured
    call graph through its compiled program with the monitor of items 3 and 4 attached."
    (design/synchronization-signature-change-request.md:153): The sentence cites items 3 and 4. Item 4's "the D
    program stops the process" and "the digest and the review record" are outstanding in the same reading, and item
    3's "every Metal 4 call" and item 4's "any count" fail above. In code: compiler/samples.py:1109-1111 passes
    --observe where `dtrace -w` is refused, which attaches no destructive action (bin/mesh-sync-monitor.py:119-122,
    :403-405), and :1123-1124 attaches no bridge observation. The player re-lowers the loaded value
    (samples.py:1085-1086) and runs it (:1089-1091).
  - 5 / Write: a wait the signature does not have: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "A wait
    the signature does not have stops the playback at the line that issues it. A playback that runs to completion has
    run exactly the compiler's signature." (design/synchronization-signature-change-request.md:156-157):
    bin/mesh-sync-monitor.py:170-172: an unlisted wait (dispatch_semaphore_wait, pthread_cond_wait,
    os_sync_wait_on_address) stops the process only on a thread already bound to a compiled order and outside the
    interpreter lock and probed calls; :172 with :319 drops both os_sync_wait_on_address probes whenever the record
    holds a plan count, and compiler/mesh_compiler/bridge.py:45-47 issues exactly such a wait per transfer each
    execution (rdma/mesh-collective.c:79) that signature.py has no operation for. Completion: :381, :390-394 compare
    classes 'program' and 'feedback' on one node; bridge, other-node and kernel operations are unobserved, the result
    is 'pass-scoped' (:406) and the exit status 0 (:418). With --observe the process is not stopped (:120).

Covered:

  - 1 / The data structure: covered [2026-11-12, mesh 507a1ec7]
  - 1 / Consumed by: printed: covered [2026-11-12, mesh 000c3787]
  - 2 / Matrix Market: covered [2026-11-08, mesh d44e3d0c]
  - 2 / DOT: covered [2026-11-08, mesh d44e3d0c]
  - 2 / Space-time diagram: covered [2026-11-08, mesh d44e3d0c]
  - 3 / Write: the program is not changed: covered [2026-11-08, mesh d44e3d0c]

## [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 50 items outstanding, 1 not determined, 10 covered, 0 withdrawn.

Outstanding:

  - R1 (requirement): "The code must move every page from the page that holds it to the page it lands in, with nothing
    copied on the way." The producer of the bridge's input copies each sent value into the transfer's pages before
    every send: compiler/mesh_compiler/ir.py:14 calls the channel's put, and compiler/mesh_compiler/bridge.py:73-76
    does flat.copy_(value...) and then publishes. Carried.out (bridge.py:64-65), which would hand out the pages to
    write in place, has no caller.
  - R1 data structure: "Each range is one block: (page-table row, offset, bytes)." The entry's page is the frame the
    row is bound to, not the row: rdma/mesh-collective.c:2058 and 2062 take frame from plan_page or from page[row],
    and :2064 stores it as .page; the bridge uses it as a region page number (rdma/mesh-flow.c:1032).
  - R1 algorithm: "Each view is flattened into its list of (row, offset, bytes) blocks, adjacent blocks merged, as
    datatype processing flattens a type map." The blocks written carry a frame, not a row:
    rdma/mesh-collective.c:2062-2064.
  - R1 a value's layout is the program's own: "A part is sent from its producer's pages, whatever its layout, and it
    lands at its byte ranges in its destination value's layout (R2)." The part is copied from the producer's tensor
    into the transfer's pages before each send (compiler/mesh_compiler/bridge.py:73-76), and each received range is a
    value of its own with a view at offset 0, not byte ranges of a destination value
    (compiler/mesh_compiler/schedule.py:145-151).
  - R1 Replaces: packing and unpacking; a layout chosen for a transfer's sake: "Replaces: packing and unpacking". The
    compile_module chain gives each transfer a value of its own (compiler/mesh_compiler/schedule.py:145-150) and the
    runtime copies the sent tensor into it before every send (compiler/mesh_compiler/bridge.py:73-76).
  - R2 (requirement): "The code must give every value its pages at compile time and name each page by its row in the
    page table." (1) Not every value: with no transfer between machines no plan is made
    (compiler/mesh_compiler/schedule.py:128-129, 166-167); results, mutated state and values of other machines are
    left out (schedule.py:169, 185-193); and the addresses of the placed values that are not transfers are handed to
    runtime.py:25,48,71 as LOCAL.storage, which nothing reads, so those values live where torch or the Metal lowering
    allocates them at run time (compiler/mesh_compiler/lower_metal.py:894, 1019-1022). (2) The lists name a page by
    its frame, not its row: rdma/mesh-collective.c:2062-2064.
  - R2 data structure: "A value is its ordered list of page-table rows, and its pages are placed anywhere." The
    compile_module chain marks every value run = 1 (the last field of the tuples at
    compile/mesh_compiler/schedule.py:149, 189, 191; rdma/mesh.py:76-78, 310), and mesh_plan lays each copy of a run
    value at consecutive blocks of one arena item at an offset chosen by mesh_graph_greedy_by_size
    (rdma/mesh-collective.c:583-617, 687).
  - R2 algorithm 1: the copies (C3.3): "At run time there are no allocations and no bookkeeping; every address is a
    constant in the plan." The Metal lowering allocates a buffer for each result storage at every call
    (compiler/mesh_compiler/lower_metal.py:894 context = begin(owner, made); :1019-1022 mesh_metal4_buffer), and
    values that are not carried are not at plan addresses (runtime.py:25, 48 set LOCAL.storage, which nothing reads).
  - R2 algorithm 2-3: the order and the chains: "Every item is one page." An item is a count of blocks
    (rdma/mesh-collective.c:484-490), and run values are records of bytes placed at offsets (:583-617). "Two items may
    share a page exactly when one's last access happens before the other's first write." and "It is found by maximum
    bipartite matching (D. R. Fulkerson, ...), exactly and in polynomial time." mesh_plan passes mesh_graph_chains the
    relation plan_alone, which orders no pair (rdma/mesh-collective.c:479-482, 658), so every block is its own chain;
    pages are shared between lifetimes by mesh_graph_fat_cover over circular arcs (:662-674) and, for run values,
    which is every value from compiler/mesh_compiler/schedule.py:149, 189, 191, by mesh_graph_greedy_by_size with an
    arc-overlap test (:575-578, 612).
  - R2 algorithm 4: the order over u copies: "Copies (v, c) and (w, c′) are ordered where c ≤ c′, c′ − c is at least
    the least lag from v to w, and c′ plus w's largest least lag to a write at the node is at most u." No such
    relation is given to mesh_graph_chains: its `before` is plan_alone, which returns 0
    (rdma/mesh-collective.c:479-482, 658). The table of least lags between values, p->between, is computed at :414-426
    and read nowhere.
  - R2 Replaces greedy by size: "No value is placed as one run, so greedy by size, and the contiguous-block problem it
    approximates (...), have no use here." mesh_graph_greedy_by_size is declared at rdma/mesh-graph.h:310, defined at
    rdma/mesh-graph.c:2931 and called by mesh_plan at rdma/mesh-collective.c:612 for every value with run set;
    compiler/mesh_compiler/schedule.py:149, 189, 191 sets run for every value it gives the plan.
  - R2 a run is a mapping, not a placement: "Such an operand's run is address translation, fixed when the static
    schedule is made." and "so the map from the operand's consecutive virtual pages to its pages is written once." No
    map is written: a run value's blocks are consecutive blocks of one arena item (rdma/mesh-collective.c:583-617,
    687), and mesh_view_values gives the run as a span of the region's own mapping, from the lowest to the highest row
    address (rdma/mesh-dataflow.c:80-101); compiler/mesh_compiler/bridge.py:33-37 reads addresses from that span.
    Neither rdma/ nor compiler/ calls a remap.
  - R2 a received range is pages of its destination value: "A range is not a value of its own." The compile_module
    chain makes each received range a value of its own, named by the receive node, with its view at offset 0
    (compiler/mesh_compiler/schedule.py:145-151).
  - R2 a rank's own part is shared, not copied: "A value is its page-table rows, so the result's rows for the rank's
    own part name the operand's own pages." From the compile_module chain no value's rows name another's pages:
    results are not given to the plan (compiler/mesh_compiler/schedule.py:144, 185), every value is run
    (schedule.py:149, 189, 191), and plan_sets skips run values (rdma/mesh-collective.c:1532, 1540, 1552).
  - R2 shared pages are one item of the allocation: "Shared pages are one item of the allocation, from the earlier of
    the two values' first writes to the later of their last reads." plan_interval takes begin and end from the first
    member's value alone (rdma/mesh-collective.c:551, 531-533). "Shared pages with no such lag are C3.3's case: 'A
    value with no such L fails the plan with ENOMEM, naming the value.'" A value with no walk is given lag 0 in
    plan_unroll (rdma/mesh-collective.c:410) and its arc the whole ring in plan_landing (:536-539); ENOMEM is returned
    only for a lag at or above UINT32_MAX (:411, 421-423).
  - R2 any value a transfer names is sent from its own pages: "Any value a transfer names is sent from its own pages,
    whatever its copies." The value is copied into the transfer's pages before each send
    (compiler/mesh_compiler/bridge.py:73-76). "An invariant, written before the first execution and read by every one
    (...), is placed in the registered window when a transfer names it. Every execution's send reads its one copy, and
    its supplied word is raised once." mesh_plan returns EINVAL for a named value whose kind is not MESH_EXECUTION
    (rdma/mesh-collective.c:1711-1714), and schedule.py:149 declares every transfer's value MESH_EXECUTION. "A
    program's input is placed per execution, and its caller writes it there. It is sent from those pages." Inputs are
    torch tensors bound by Program.bind (compiler/mesh_compiler/runtime.py:57-64) and are copied at the send
    (bridge.py:75). "A result whose rows name an invariant operand's pages names the operand's one copy from every
    copy of the result." No result is given rows (schedule.py:185).
  - R2 ranges may share a block where their bytes are disjoint: "A halo wider than a shard, or two halos over one
    shard, is then sent with no copy." In the compile_module chain a send of a view is given a value of its own
    (compiler/mesh_compiler/schedule.py:144-150) and every send copies into its value's pages
    (compiler/mesh_compiler/bridge.py:73-76).
  - R2 where it composes: "Takes: ... which consumers see a value as one run (R4)." schedule.py sets run = 1 for every
    value, whatever its consumers (compiler/mesh_compiler/schedule.py:149, 189, 191). "Writes: ... for each operand
    seen as one run, the map from its consecutive virtual pages to its pages." No map is written; a run is consecutive
    blocks of an arena item and a span of the region's mapping (rdma/mesh-collective.c:687;
    rdma/mesh-dataflow.c:80-101). "Consumed by R5's lists, whose entries name each page by its row (R1)". The entries
    name frames (rdma/mesh-collective.c:2062-2064).
  - R2 Replaces (mesh_plan's rules): "That value is MESH_EXECUTION and wire, its bytes the range's, and one range's
    alone". mesh_plan still returns EINVAL for a named value whose kind is not MESH_EXECUTION
    (rdma/mesh-collective.c:1711-1714). "a lag-k value 'named by no declared transfer'". A lag value named by a
    receive at this node is still EINVAL (:1711, `t->destination==p->self && v->lag`). "the share record (struct
    mesh_share) and its rules". struct mesh_value keeps a `shares` field (rdma/mesh-collective.h:25), which plan_rows
    sets (rdma/mesh-collective.c:703-711).
  - R3 the task graph: V and w: "The code must represent one execution as a task graph and nothing more." and "V are
    tasks, each one operation of the captured graph (aot-compilation A1) executed at its node." The graph mesh_plan
    builds has tasks only for the operations that write, read or consume a transfer between machines
    (rdma/mesh-collective.c:836-849); the other operations of the captured graph are not vertices and their weights
    are not passed, only their reachability as dependences (compiler/mesh_compiler/schedule.py:83-86, 138, 153-157).
    With no such transfer mesh_plan is not called (schedule.py:128-129), and the processors' order is made from the fx
    graph by mesh_graph.longest and mesh_graph.heap_order (schedule.py:201-214).
  - R4 (requirement): "The code must schedule both kinds of function, and must not tell the bridge which is which."
    The compile_module chain declares every transfer's consumer MESH_JOIN: compiler/mesh_compiler/devices.py:35 sets
    arrival.meta["consumer"] = JOIN for every receive, schedule.py:152 passes it, and nothing sets PAGES
    (compiler/mesh_compiler/ir.py:34 is its only mention).
  - R4 behaviour at run time, 2: "A function that takes pages in any order has one task for each page. Each task
    starts when its own page is present. It does not wait for another page of the same operand." No transfer from the
    compile_module chain is of this kind (compiler/mesh_compiler/devices.py:35), and every receive waits for all pages
    of its range (compiler/mesh_compiler/bridge.py:79-82; rdma/mesh-collective.c:37-46). "Each output page is released
    when it is written". The send publishes once, after the whole value is copied
    (compiler/mesh_compiler/bridge.py:73-76; rdma/mesh-collective.c:17-25).
  - R4 data structure: "its outputs are pages that leave as each is produced." A sent range has one supplied word
    (rdma/mesh-collective.c:460-462, 1724, 2036, 2067), which mesh_host_publish raises once an execution (:17-25),
    after the whole value is written (compiler/mesh_compiler/bridge.py:73-76); and the compile_module chain declares
    no consumer of this kind (compiler/mesh_compiler/devices.py:35).
  - R4 algorithm: presence is the supplied word: "A producer's store releases whatever pages its work has written, any
    number at once and in any order." There is one word for each sent range (rdma/mesh-collective.c:1724), holding a
    count of executions (:23-24), so a store releases all pages of a range for an execution and no part of them.
  - R4 nothing synchronizes outputs: the release follows the producing work: "The function's kernels write their
    pages, and the supplied store follows them." The kernels write buffers of torch's or of the Metal lowering's
    (compiler/mesh_compiler/lower_metal.py:894, 1019-1022), and the host copies the result into the transfer's pages
    (compiler/mesh_compiler/bridge.py:75; lower_metal.py:903). "The compilation writes no kernel (aot-compilation
    A9)." compiler/mesh_compiler/lower_metal.py:927-933 composes Metal kernel source (`kernel void k(...)`) for the
    operators it lowers.
  - R4 contiguity is a mapping: "Only the synchronous, complete, contiguous consumer's operand, and an operand whose
    consumer needs one strided progression, is seen as one run." compiler/mesh_compiler/schedule.py:149, 189, 191 sets
    run = 1 for every value. "That run is a mapping written when the static schedule is made, and its pages are placed
    anywhere, as every value's are." A run value's copy is consecutive blocks of one arena item at a greedy-by-size
    offset (rdma/mesh-collective.c:583-617, 687), read as a span of the region's mapping
    (rdma/mesh-dataflow.c:80-101); no mapping is written.
  - R5 data structure: one linear program over completion times: "A resource is a cable in one direction, or a node's
    device." The resources are one for each node (rdma/mesh-collective.c:882-883, 887) and one for each pair of nodes
    in each direction (:892-895, 945, 948), a pair's capacity being the sum of its cables' bandwidths
    (rdma/mesh-dataflow.c:203-208, 225); processors of one machine share the node's resource
    (compiler/mesh_compiler/schedule.py:142). "for a hop c(e) over its cable's bandwidth". The divisor is the pair's
    capacity over the largest pair's capacity (rdma/mesh-collective.c:891-895, 943).
  - R5 where it composes: "This is the one schedule of the data flow." compiler/mesh_compiler/schedule.py:201-214
    orders every processor's functions by mesh_graph.longest and mesh_graph.heap_order over the fx graph, taking from
    the plan only the receives' times (:204-206); with no transfer between machines mesh_plan is not called
    (:128-129). "each page's route, over trees packed by Garg and Könemann's rounds (...; rdma/mesh-graph.h
    `mesh_graph_pack`)". mesh_plan routes by mesh_graph_flow (rdma/mesh-collective.c:1758); it does not call
    mesh_graph_pack.
  - R7 (requirement and fact of the code): "The code must keep every transport object and count inside the bridge".
    The map's records carry each cable's queue-pair count (rdma/mesh-map.h:80 struct mesh_reach qps, written by the
    bridge at rdma/mesh-ports.c:111), and mesh_cables in rdma/mesh-dataflow.c:144, which is linked into libmesh.dylib
    (rdma/Makefile:124-125), reads it in the compiling process: compiler/mesh_compiler/schedule.py:132 calls
    mesh_nodes_read, which calls mesh_capacity and mesh_cables (rdma/mesh-dataflow.c:206, 225). mesh_capacity uses
    only the bandwidth.
  - R8 attaching holds nothing; no EADDRINUSE: "The code must hold nothing for a program but its pages." and "there is
    no client word". struct hdr has `client` and `client_start` (rdma/mesh.h:168-169, 180), struct mesh_ctx has
    `client` (rdma/mesh-dataflow.h:11), mesh_attach forms it (rdma/mesh-dataflow.c:48-52), the publication stores it
    (rdma/mesh-collective.c:1851-1852) and mesh_retire tests it (rdma/mesh-dataflow.c:20);
    compiler/mesh_compiler/bridge.py:24 and schedule.py:77 pass it. "rdma/mesh-dataflow.c: attaching only maps the
    region." mesh_attach also adds 1 to hdr `serial` and reads `device_client` (rdma/mesh-dataflow.c:48-50). "A bridge
    with no function execution to serve does nothing." Each port's thread wakes every MESH_PAUSE_NS and runs a pass
    (rdma/mesh-flow.c:2194-2199, 2105).
  - R8 there is no client word: "Attaching holds nothing, and there is no client word". rdma/mesh.h:169 declares hdr
    `client` (and :180 `client_start`); rdma/mesh-dataflow.h:11 declares the context's `client`;
    rdma/mesh-dataflow.c:48-52 makes it at attach; rdma/mesh-collective.c:1851-1852 stores it at publication;
    rdma/mesh-dataflow.c:20 and rdma/mesh-holders.h:43-46 read it; compiler/mesh_compiler/bridge.py:24 and
    schedule.py:77 pass it.
  - R8 a bridge with no function execution does nothing: "A bridge with no function execution to serve does nothing."
    A port's progress thread waits on hdr `plans` for at most MESH_PAUSE_NS and then runs a pass again, with
    link_receive_progress and link_send_progress, whether or not anything was released (rdma/mesh-flow.c:2194-2199,
    2082-2105).
  - R8 data structure and algorithm: one offline allocation at compile time: "one offline allocation over the whole
    exported program (...), at compile time." It covers one machine's values and leaves out results and mutated state,
    and is not made where no transfer is carried (compiler/mesh_compiler/schedule.py:128-129, 166-169, 185-193). "the
    algorithm is R2's" and "which consumers see one as a run, each such run a mapping." Every value is given run = 1
    (schedule.py:149, 189, 191) and is placed by mesh_graph_greedy_by_size as consecutive blocks
    (rdma/mesh-collective.c:583-617, 687). "Nothing takes pages at run time". The Metal lowering takes a buffer for
    each result at every call (compiler/mesh_compiler/lower_metal.py:894, 1019-1022).
  - R8 Replaces: any rule for several programs: "Replaces: ... any rule for several programs." The region records
    which program holds it and acts on it: the publication stores the publisher's client word and start
    (rdma/mesh-collective.c:1851-1852); mesh_retire retires the operand only where hdr `client` is the caller's
    (rdma/mesh-dataflow.c:20); and mesh_plan and mesh_plan_load wait for the operand published before to be released
    and return ETIMEDOUT otherwise (rdma/mesh-collective.c:1813-1822, 1986, 2390).
  - R1 where it composes: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. line 1974: "Writes: the blocks, as
    the entries of each cable's list (R5): `struct mesh_hop` `page`, `offset` and `bytes` (rdma/mesh.h).": mesh_plan
    writes one list for the node, each entry naming a neighbouring node and a direction, not a cable:
    rdma/mesh-collective.c:773-783 (.node=in / child[c]) and :2064. Which cable carries an entry is decided by the
    bridge at the load: rdma/mesh-flow.c:837 picks the entries naming the peer, :856-866 deals them over the pair's
    cables and queue pairs by mesh_graph_list_schedule. The sentence also cites R5, which the same index lists
    outstanding (R5 data structure: the resources are pairs of nodes, rdma/mesh-collective.c:892-895 and
    rdma/mesh-dataflow.c:203-208; R5 where it composes).
  - R3 the task graph: E are pages, with weight c: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. lines
    2143-2144: "They are given, not computed by this code: the weighting operand of
    design/aot-compilation-change-request.md A3.": The sentence cites A3, whose "The caller passes the operand" the
    index lists outstanding, and it is so at this head: no caller of compile_module passes `cost` (examples/tp.py:138,
    :197; examples/engines.py:111; compiler/samples.py:338, :839, :1010, :1035), so every w and c is the compiler's
    own default unit_cost, 1.0 (compiler/mesh_compiler/driver.py:14; compiler/mesh_compiler/schedule.py:26-27, :153).
    Also, for line 2142 ("E are the data passed between tasks: here, pages"): plan_schedule adds edges that are no
    page and gives them cost 0 itself (rdma/mesh-collective.c:919, :920-929).
  - R3 where it composes: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. lines 2147-2148: "Takes: the
    compilation's IR (aot-compilation A8), which is the captured graph with each task at its node by the placement
    operand (aot-compilation A2), and the weighting operand (aot-compilation A3).": What mesh_plan takes from
    compile_module is not the captured graph with each task: only the transfers between machines, four weights each
    and events of dependence (compiler/mesh_compiler/schedule.py:127-129, :140-157; passed at :108-111), and
    plan_schedule makes tasks only for a transfer's producer, own reader and consumer
    (rdma/mesh-collective.c:836-849). The cited A8 ("the type mesh_plan takes") is outstanding in the same index
    (schedule.py:110 passes None, None, 0, None and :158-159 the constant 1), as are A3's caller sentence and R5's
    data structure, which line 2153 cites.
  - R4 behaviour at run time, 3: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. lines 2189-2193: "In that
    program the two kinds are rows, not labels ... each of the second kind's tasks \"has one edge, from its own page,
    so its pages are independent flows\" (R5). So the order a cable sends in is computed from which pages a function
    needs together and which it needs singly.": From the compile_module chain there is one kind only:
    compiler/mesh_compiler/devices.py:35 sets arrival.meta["consumer"] = JOIN for every receive, schedule.py:152
    passes it, and PAGES (ir.py:34) is assigned nowhere. So rdma/mesh-collective.c:843 always makes one consumer task
    and :908 always targets task 0 of it; no page is "needed singly" in any program the chain solves. The sentence
    also cites R5, outstanding in the same index, and R4 (requirement) is listed outstanding for this same fact.
  - R4 nothing synchronizes outputs one page at a time: landings read by masks: outstanding [2026-11-12, mesh
    6d379b7e; a second reader]. line 2239: "The receiving side reads landings by masks (C3), so a landing interrupts
    no backend.": The cited C3 masks are the first pass that loads the receives' phases and predicates threads by
    select (disjoint-pages C3.1 algorithms 1 to 3, outstanding in the same index).
    compiler/mesh_compiler/lower_metal.py has no phase, mask or simd_ballot anywhere; its receive is a host call
    (lower_metal.py:1056-1057) that blocks in Carried.get (compiler/mesh_compiler/bridge.py:79-82) on mesh_host_wait,
    which sleeps on the range's count word until every page has landed (rdma/mesh-collective.c:73-82; :36-47). The
    receiving side is a blocking host wait on the whole range, not a read by masks.
  - R4 where it composes: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. lines 2250-2251: "Takes: each
    consumer's kind, which the IR carries (aot-compilation A8): `struct mesh_declared` `kind`, `MESH_JOIN` or
    `MESH_PAGES` (rdma/mesh-collective.h).": No consumer's kind is taken: compiler/mesh_compiler/devices.py:35 writes
    the constant JOIN on every receive node whatever its consumer, and schedule.py:152 passes that; PAGES (ir.py:34)
    is never assigned. The cited A8 data structure and appendix item 5 ("Every program declares each consumer's kind")
    are outstanding in the same index, as are R3 (V and w) and R5, which line 2252 cites.
  - R5 (requirement): outstanding [2026-11-12, mesh 6d379b7e; a second reader]. line 2266: "The code must order each
    cable's sends by a static schedule of one execution's task graph, minimizing its makespan.": The program solved is
    not one execution's task graph: its tasks are only each transfer's producer, own reader and consumer
    (rdma/mesh-collective.c:836-849, :879-888), with weights for those alone (compiler/mesh_compiler/schedule.py:153);
    every other operation of the execution enters only as a zero-cost precedence edge (schedule.py:83-86, :138,
    :154-157; rdma/mesh-collective.c:920-929), so the C_max minimized (rdma/mesh-graph.c:1180, :1185-1189) leaves
    their time out, and with no transfer between machines nothing is solved (schedule.py:127-129). The order is also
    not per cable: a resource is a pair of nodes with its cables' bandwidths summed (rdma/mesh-collective.c:892-895,
    :945; rdma/mesh-dataflow.c:203-208), and the list is per neighbouring node (rdma/mesh-collective.c:2064;
    rdma/mesh-flow.c:837).
  - R5 algorithm steps 3-5: priority, one global topological order, each cable's subsequence: outstanding [2026-11-12,
    mesh 6d379b7e; a second reader]. lines 2324 and 2326: "Each instruction's priority is C̄_j, its completion time in
    the program's optimal solution, the least first." and "One global topological order of the instructions is built":
    Two orders are built, and the tasks run in the one that is not by C̄_j. mesh_plan builds an order of every
    instruction by C̄ (rdma/mesh-collective.c:956-957) and keeps only the hops' places (:962-963). The functions'
    order, the one each processor executes, is built separately in compiler/mesh_compiler/schedule.py:203-210:
    priority is the longest-path cost (mesh_graph.longest, rdma/mesh_graph.py:116-119), with only the receive nodes
    given the plan's times (schedule.py:204-206), through a second call of heap_order (:208). The index lists the same
    fact outstanding under R5 where it composes.
  - R5 a set's pages keep their places: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. lines 2333-2334: "A
    set's pages (design/aot-compilation-change-request.md, appendix item 5) keep their places in that order. The set's
    random permutation is drawn over those places.": From the compile_module chain no set exists and no permutation is
    drawn: every transfer is JOIN (compiler/mesh_compiler/devices.py:35; schedule.py:152), so
    rdma/mesh-collective.c:1695 clears d->set for every demand, :759 writes set 0 on every entry, and
    rdma/mesh-flow.c:904 skips every pick. The code that keeps places and draws (rdma/mesh-flow.c:607-620, :649-658)
    is not reached by any list the chain produces. The cited appendix item 5 has two entries outstanding in the same
    index (the algorithm's key; every program declares each consumer's kind).
  - R5 executions: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. line 2337: "execution t runs copy t mod
    u, and every address is a constant of the static schedule (C3.3, R2).": Not every address: the Metal lowering
    allocates a buffer for each result storage at every call (compiler/mesh_compiler/lower_metal.py:1177-1184, begin:
    LIB.mesh_metal4_buffer, called from run at :1050) and maps argument storage by address at commit (:1187-1194). The
    cited C3.3 compile-time bound and R2 (requirement, data structure, algorithm 1) are outstanding in the same index
    for this. The copy rule itself is as stated (rdma/mesh-collective.c:70; compiler/mesh_compiler/bridge.py:53, :65,
    :74, :82).
  - R6 data structure and no run-time algorithm: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. lines
    2387-2388: "The data structure is the transfers of the static schedule: the send and receive nodes that placement
    adds to the graph, which R5 writes into each cable's lists.": Nothing is written into a cable's list at compile
    time: mesh_plan writes one list whose entries name a neighbouring node and a direction
    (rdma/mesh-collective.c:773-783, :2064), and which cable and queue pair carries each entry is computed by the
    bridge when it loads the plan (rdma/mesh-flow.c:813-823, :837, :856-866). The sentence cites R5, outstanding in
    the same index (data structure: pairs of nodes, not cables; where it composes). The transfers themselves are
    compiled at every rank (compiler/mesh_compiler/placement.py:676-679; schedule.py:140-152).
  - R6 where it composes: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. lines 2393-2395: "Produced by
    placement, as `struct mesh_declared` with its `source` and `destination` (rdma/mesh-collective.h), and by the
    schedule (R5), as `struct mesh_hop` with its `node` and `direction` (rdma/mesh.h).": Placement produces fx send
    and receive nodes with a processor name (compiler/mesh_compiler/placement.py:676-679); struct mesh_declared and
    its source and destination are produced in the schedule stage, from the machines map and the bridge's node list
    (compiler/mesh_compiler/schedule.py:131-133, :140-142, :152; rdma/mesh.py:306-308). The sentence also cites R5,
    outstanding in the same index. This refutation rests on those two grounds only; the bridge reading its own lists
    holds (rdma/mesh-flow.c:802, :837).
  - R8 pages released at retirement by deferred reclamation: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    lines 2433-2434: "A program's pages are released at its retirement by deferred reclamation (P. E. McKenney and J.
    D. Slingwine, Read-copy update, PDCS 1998; K. Fraser, Practical lock-freedom, PhD thesis, Cambridge, 2004;
    C3.4).": The sentence cites C3.4, whose (a), (b) and (c) are outstanding in the same index; (c) says "nothing
    waits for the release", and both mesh_plan and mesh_plan_load wait for it with a deadline and return ETIMEDOUT
    (rdma/mesh-collective.c:1813-1822, called at :1986 and :2390; the deadline from
    compiler/mesh_compiler/bridge.py:25-28). Retirement in the chain is called only before the same client's next
    publication (bridge.py:24; rdma/mesh-dataflow.c:17-21); compiler/mesh_compiler/runtime.py calls none when a
    program ends. The deferred release itself is at rdma/mesh-dataflow.c:122-131.
  - R8 function executions ordered by a fixed sequencer: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    lines 2435-2436: "Function executions are the one program's, ordered by a fixed sequencer:
    design/aot-compilation-change-request.md A7 (P. Barham et al., Pathways, MLSys 2022, §4.4; Défago, Schiper and
    Urbán, 2004).": The sentence is A7 by citation, and A7 has entries outstanding in the same index, one of which is
    in this chain: after a view is installed mesh_transfers_start returns ESTALE (rdma/mesh-dataflow.c:252, :267) and
    nothing computes the static schedule again, since Program loads the record once
    (compiler/mesh_compiler/bridge.py:25-28). "The one program's" also meets code for more than one: the publication
    stores the publisher's client word and mesh_retire tests it (rdma/mesh-collective.c:1851-1852;
    rdma/mesh-dataflow.c:20), which the index lists outstanding under R8 Replaces: any rule for several programs. The
    submission to the sequencer's queue is as stated (rdma/mesh-dataflow.c:237-268, from bridge.py:42).
  - R8 where it composes: placement and weights are operands: outstanding [2026-11-12, mesh 6d379b7e; a second
    reader]. line 2438: "Placement and weights are operands, passed in by the program from whatever function chooses
    them (aot-compilation A2, A3).": No program passes weights: every call of compile_module omits `cost`
    (examples/tp.py:138, :197; examples/engines.py:111; compiler/samples.py:338, :839, :1010, :1035) and takes the
    compiler's default unit_cost (compiler/mesh_compiler/driver.py:14; schedule.py:26-27). The cited A3 ("The caller
    passes the operand") is outstanding in the same index, as are R3 and R5, which line 2439 cites. The placement
    operand is passed and taken (driver.py:14, :19).
  - R8 where it composes: the pages are part of the scheduled value: outstanding [2026-11-12, mesh 6d379b7e; a second
    reader]. line 2440: "The pages are R2's rows, written when the static schedule is made and part of the scheduled
    value.": The sentence is R2 by citation, and nearly every R2 entry is outstanding in the same index. In this chain
    rows exist only for one machine's transferred and stored values: with no transfer between machines no plan and no
    row is made (compiler/mesh_compiler/schedule.py:127-129, :166-167), and results, mutated state and other machines'
    values get none (schedule.py:169, :185-193); every value is marked run and placed as consecutive blocks by
    mesh_graph_greedy_by_size (schedule.py:149, :189, :191; rdma/mesh-collective.c:612, :687). What rows are made are
    in Plan.record and archived (schedule.py:116-118; compiler/mesh_compiler/archive.py:62, :122).

Not determined:

  - R4 the record of a run: not determined [2026-11-12]. The clause's one sentence is about what the quoted sources
    say; it names no code.

Covered:

  - R1 Replaces: the share record: covered [2026-11-06, mesh 31ee5d4a]
  - R4 behaviour at run time, 1: covered [2026-11-12, mesh 0704bf95]
  - R4 behaviour at run time, 4: covered [2026-11-12, mesh 81c8e9da]
  - R5 algorithm steps 1-2: cutting planes, HiGHS dual simplex: covered [2026-11-05, mesh f4838c9d]
  - R5 every rank computes the same schedule: covered [2026-11-12, mesh 81c8e9da]
  - R5 changes by file: rdma/mesh-graph.{h,c}: covered [2026-11-05, mesh f4838c9d]
  - R5 changes by file: rdma/Makefile builds HiGHS: covered [2026-11-05, mesh 841cba44]
  - R5 changes by file: mesh-collective.c and mesh-flow.c: covered [2026-11-06, mesh 5eeee1f9]
  - R7 where it composes: covered [2026-11-12, mesh 81c8e9da]
  - R8 Replaces: mesh_claim_take, the claim-head walk, fragmentation ENOMEM: covered [2026-11-05, mesh c6c9c77f]

## [disjoint-pages-change-request.md](disjoint-pages-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 46 items outstanding, 0 not determined, 28 covered, 1 withdrawn.

Outstanding:

  - A3 item 1 (Items): "A value of several pages is several items, each placed on any page, not as a run":
    compiler/mesh_compiler/schedule.py:149, :189 and :191 give every value `run` 1, and mesh_plan places a `run` value
    at one offset of one run of blocks (rdma/mesh-collective.c:583-616 plan_block, mesh_graph_greedy_by_size at :612;
    :687 block k is the run's base + k). "An item is one page of one copy of one value.": the element placed is a
    block of hdr `block` pages (rdma/mesh-collective.c:189-193, :680).
  - A3 item 2 (The order): "Item a precedes item b when the dependence graph orders every read of a before the first
    write of b: happens-before, not time": for the values compile_module passes, two values share blocks where the
    arcs [begin, end] of their positions in the local scheduled order do not overlap (rdma/mesh-collective.c:560-578
    plan_location and plan_intersect; rdma/mesh-graph.c:2907 mesh_graph_arcs_overlap). begin and end are indices in
    the scheduled order (compiler/mesh_compiler/schedule.py:168-169, :180-182, :191).
  - A3 item 3 (The chains): "The items are split into the fewest chains, as many as the largest set of mutually
    unordered items.": mesh_plan calls mesh_graph_chains with plan_alone, which returns 0 for every pair
    (rdma/mesh-collective.c:479-482, :658), so every element is its own chain; items with an interval are coloured by
    mesh_graph_fat_cover (:670); every value compile_module passes has `run` and is given an offset by
    mesh_graph_greedy_by_size (:612).
  - A3 item 4 (Operands typed as one run): "A consumer typed to see one run of pages is given it as a mapping written
    when the static schedule is made": the consumer (compiler/mesh_compiler/bridge.py:35-37 and :59-61,
    torch.frombuffer at view.address + offset) reads a value that mesh_plan placed as one run of blocks in the region
    (rdma/mesh-collective.c:583-616, :687; rdma/mesh-dataflow.c:80-104 mesh_view_values takes the lowest and highest
    addresses of the `run` values). No mapping composes separately placed pages.
  - A3 "What it takes": "Every value one execution moves or computes, with its size in pages.":
    compiler/mesh_compiler/schedule.py:185-193 gives mesh_plan no value for a result, for state the program mutates,
    for a placeholder that is not state, or for a value whose producer is on another machine; the lowered Metal
    function holds its intermediates and results in Metal buffers that are no struct mesh_value
    (compiler/mesh_compiler/lower_metal.py:984, :1022 at HEAD).
  - A3 "Which function computes it, and which read it": "nothing allocates a program page while the program runs, not
    the bridge, not a call site, not a GPU kernel": compiler/mesh_compiler/lower_metal.py:1019-1023 (HEAD) allocates a
    Metal buffer for each result at each call of the lowered function, and :1036 makes a buffer view at a commit. "the
    recorded kernels (`metal_recording.m`) read the pages the page table names": the kernels lower_metal.py emits bind
    buffers by gpuAddress (lower_metal.py:1000-1008, :1046-1047; rdma/mesh-metal4.m:184-186), none taken from a struct
    mesh_page_entry row; Program.storage (compiler/mesh_compiler/runtime.py:25) has no reader.
  - A4 deleted (4): "`mesh_graph_left_edge`'s colouring, greedy by size.": rdma/mesh-graph.c:2931 is
    mesh_graph_greedy_by_size, which mesh_plan's plan_block calls (rdma/mesh-collective.c:612) for every value with
    `run`, and every value compiler/mesh_compiler/schedule.py passes has `run` (:149, :189, :191).
  - C3.1 data structure: "With it are one count per received range, one threadgroup count per later pass, and one
    status word per execution.": plan_layout places one struct mesh_status for the static schedule
    (rdma/mesh-collective.c:451-452) and no threadgroup count.
  - C3.1 algorithm 1 (masked first pass): "An execution's first pass loads its receives' phases, `MESH_LANE_BYTES` (a
    compile-time 16) bytes per GPU thread, and computes in the threads whose pages have been received.", "The other
    threads are predicated off." and "A mask is applied by select, never by multiplying by 0 or 1 (0 × NaN is NaN),
    and a reduction puts its identity into the masked-off threads.": compiler/mesh_compiler/lower_metal.py (HEAD)
    emits no pass that reads phases and no mask; a receive is a host step between commits (:824-828, :901-902) that
    waits in mesh_host_wait (compiler/mesh_compiler/bridge.py:80; rdma/mesh-collective.c:66-82).
  - C3.1 algorithm 2 (compaction): "The first pass compacts the work it left (`simd_ballot`,
    `simd_prefix_exclusive_sum`) and writes the later pass's threadgroup count.":
    compiler/mesh_compiler/lower_metal.py (HEAD) has no first pass, no simd_ballot or simd_prefix_exclusive_sum and no
    threadgroup count.
  - C3.1 algorithm 3 (indirect dispatch): "`dispatchThreadgroups(indirectBuffer:)` reads that count when the later
    pass starts, so the later pass is empty when every page has been received.": rdma/mesh-metal4.m:187-190 dispatches
    with dispatchThreadgroups:threadsPerThreadgroup: or dispatchThreads:threadsPerThreadgroup:; there is no indirect
    buffer and no later pass (compiler/mesh_compiler/lower_metal.py at HEAD).
  - C3.1 algorithm 4, bridge half: "so the later pass finds it one phase behind, and its threads end as an error that
    the later pass writes into the execution's status word": there is no later pass in
    compiler/mesh_compiler/lower_metal.py; the reader is mesh_host_arrived (rdma/mesh-collective.c:28-49), which
    returns UINT64_MAX, and mesh_host_wait returns ECANCELED (:77); nothing writes a struct mesh_status.
  - C3.1 algorithm 4, later-pass half: "A cancelled receive does not change its phase, so the later pass finds it one
    phase behind, and its threads end as an error that the later pass writes into the execution's status word.":
    compiler/mesh_compiler/lower_metal.py (HEAD) has no later pass; mesh_host_wait returns ECANCELED to the host
    (rdma/mesh-collective.c:77, from compiler/mesh_compiler/bridge.py:80), and nothing writes the status word.
  - C3.1 "Written by, read by": "The recorded kernels' first pass (`mesh_read_N`, `metal_recording.m`) reads the
    phases and writes the threadgroup count, and Metal's indirect dispatch reads that count.":
    compiler/mesh_compiler/lower_metal.py (HEAD) has no such pass and rdma/mesh-metal4.m:187-190 has no indirect
    dispatch. "The later pass writes the status word, which only the GPU writes, and `e2b_replay` reads it after the
    run.": nothing in compiler/mesh_compiler or the rdma functions it calls writes or reads the status that
    rdma/mesh-collective.c:451-452 places. "The threadgroup count is Metal's
    `MTLDispatchThreadgroupsIndirectArguments`.": not in compiler/mesh_compiler or rdma/mesh-metal4.m.
  - C3.2 (a) data structure and types: "**the algorithm** is A3's allocator": the values compile_module passes are
    placed by mesh_graph_greedy_by_size (rdma/mesh-collective.c:612). "and the bridge and the recorded kernels read it
    (A3)": the kernels compiler/mesh_compiler/lower_metal.py emits read no placement (lower_metal.py:1000-1008,
    :1046-1047 at HEAD).
  - C3.2 (b) what is placed: "It places the weights, the KV cache, pooled intermediates, the values sent between
    nodes, the GPU's publication records, the array of phases and the sent ranges' words (C3.7).": state the program
    mutates is given no value (compiler/mesh_compiler/schedule.py:186-187), and the lowered Metal function holds its
    intermediates in mesh_metal4_buffer allocations (compiler/mesh_compiler/lower_metal.py:984 at HEAD).
  - C3.2 (c) invariants placed once: "A value written before the first execution and read by every execution (the
    weights; the KV cache as one array) is placed once, for every execution": state the program mutates is not passed
    to mesh_plan (compiler/mesh_compiler/schedule.py:186-187).
  - C3.3 "Written by, read by": "A3's allocator places the copies' pages.": the copies of every value compile_module
    passes are placed by mesh_graph_greedy_by_size (rdma/mesh-collective.c:583-616, :612).
  - C3.3 compile-time bound: "At run time nothing is allocated, and every address is a constant of the static
    schedule.": compiler/mesh_compiler/lower_metal.py:1019-1023 (HEAD) allocates each result's Metal buffer at each
    call, and :1036 makes a buffer view at a commit.
  - C3.4 (a) no claim: "A3's allocator places every value of the whole exported program, at compile time":
    compiler/mesh_compiler/schedule.py:126-159 and :165-195 pass only this machine's transferred and stored values
    (:167-169, :185-193), and they are placed by mesh_graph_greedy_by_size (rdma/mesh-collective.c:612).
  - C3.4 (b) new state in new pages, no wait: "A new static schedule's state is in new pages, never the previous
    one's.": placement starts at the arena range's first page at every call (rdma/mesh-collective.c:642-643, :680),
    and mesh_plan_load copies the state to the same page relative to it (:2182-2189, :2393). "Nothing erases a static
    schedule or waits for one to be erased." and "nothing waits for the release": mesh_plan and mesh_plan_load wait in
    plan_released until the bridge stores `released` or the deadline passes (rdma/mesh-collective.c:1813-1822, :1986,
    :2390; compiler/mesh_compiler/bridge.py:27 gives the deadline).
  - C3.4 (c) deferred reclamation: "and nothing waits for the release": mesh_plan and mesh_plan_load wait for it in
    plan_released (rdma/mesh-collective.c:1813-1822, :1986, :2390), called with a deadline from
    compiler/mesh_compiler/bridge.py:25-28.
  - C3.5: "`tools/prepare_e2b.py` writes the dependences into the manifest.": metal-microbench tools/prepare_e2b.py
    writes no dependences; its manifest (:1446-1779) holds none. "`e2b_replay` passes them to `mesh_plan` in place of
    `NULL, 0`": tools/e2b_replay.m is outside compile_module and builds its edges itself (:722-779). "A3's allocator
    reads them as its order.": the order of the values compile_module passes is the overlap of their [begin, end]
    positions (rdma/mesh-collective.c:560-578).
  - C3.6 (1) late cancellation: "because no static schedule writes another's state": a later mesh_plan or
    mesh_plan_load writes its state over the pages of the one before, after plan_released (rdma/mesh-collective.c:1986
    and :1993, :2390 and :2393; placement from the range's first page, :642-643).
  - C3.6 (2) retired client's schedule: "A port that keeps a retired client's static schedule past `MESH_PLATFORM_NS`:
    removed by C3.4, with the claim at attach.": mesh_plan_load waits for the ports to drop the retired schedule until
    its deadline, now + PLATFORM_NS, and returns ETIMEDOUT past it (compiler/mesh_compiler/bridge.py:24-28;
    rdma/mesh-collective.c:2390, :1813-1822).
  - C3.6 (5) final wait bounded: "`e2b_replay.m`'s final wait for completion is bounded.": tools/e2b_replay.m is
    outside compile_module; Program's final waits are rdma/mesh-metal4.m:212
    (waitUntilSignaledValue:timeoutMS:UINT64_MAX, from compiler/mesh_compiler/lower_metal.py:1058 at HEAD) and
    compiler/mesh_compiler/runtime.py:74-75 (a queue get with no bound).
  - C3.7 "Written by, read by": "The recorded kernels' first pass reads the phases (C3.1).":
    compiler/mesh_compiler/lower_metal.py (HEAD) has no first pass; the phases are read on the host by
    mesh_host_arrived (rdma/mesh-collective.c:28-49).
  - C3.7 data pages: "No data page is written but by its producer or by the receive that fills it.": plan_bind_values,
    in mesh_plan and mesh_plan_load, zeroes the pages of each value whose `state` is set (rdma/mesh-collective.c:1835,
    called at :2085 and :2396).
  - C3.7 only zeroing: "The only zeroing is mesh-link.c's zeroing of the region's header, once, when the bridge
    creates the region.": mesh_plan zeroes the state section it places at each call (rdma/mesh-collective.c:1993), and
    plan_bind_values zeroes the pages of each value whose `state` is set (:1835).
  - C3.7 projection kernels: "The projection kernels' partial sums are not zeroed: every threadgroup writes its own
    before any are read." and "Their arrival counters restore the count on the last arrival
    (native_projection.metal:232), as an mbarrier's phase completion reloads it.": native_projection.metal is in
    neither repository at HEAD, and compiler/mesh_compiler/lower_metal.py has no projection kernel with partial sums
    or arrival counters.
  - C4 mesh deleted: rdma/mesh-collective.c (memset and word zeroing): "the `memset` of every declared instance's
    pages before `hops`": plan_bind_values zeroes the pages of each value whose `state` is set, before the schedule is
    published (rdma/mesh-collective.c:1835, called at :2085 before plan_publish at :2086).
  - C4 mesh deleted: rdma/mesh-dataflow.c claim: "the wait before it for the ports to drop a retired client's static
    schedule": mesh_plan and mesh_plan_load wait in plan_released for the bridge's release of the retired schedule,
    which it stores once no port holds it (rdma/mesh-collective.c:1813-1822, :1986, :2390;
    rdma/mesh-dataflow.c:110-131).
  - C4 metal-microbench written: tools/prepare_e2b.py: "tools/prepare_e2b.py: each stage's reads and writes, into the
    manifest (C3.5).": the manifest metal-microbench tools/prepare_e2b.py writes (:1446-1779) has no stage's reads and
    writes and no dependences.
  - C4 metal-microbench written: tools/e2b_replay.m: "tools/e2b_replay.m: the manifest's dependences, passed to
    `mesh_plan` (C3.5).": tools/e2b_replay.m is outside compile_module, and it builds the edges from the manifest's
    regions itself (:722-779); the manifest holds no dependences.
  - D.3.2: "Every buffer the capture binds is a value `mesh_plan` places": the lowered Metal function's buffers are
    mesh_metal4_buffer allocations and views of torch storage (compiler/mesh_compiler/lower_metal.py:984, :1022, :1036
    at HEAD), not values of mesh_plan. "An invariant's producer writes it into its placed pages once, after
    `mesh_plan` and before the first execution.": nothing writes state into the placed pages;
    compiler/mesh_compiler/runtime.py:31-35 binds the exported tensors, and the placed address (bridge.py:51-53,
    runtime.py:25) has no reader. "The recorded kernels read the placed pages, and after the bind no command reads a
    delegate buffer.": the lowered kernels read the buffers above. "**the algorithm** is A3's allocator": see A3 item
    3.
  - D.3.3: "The program declares the lag of the CPU code that writes them": compiler/mesh_compiler/schedule.py:149,
    :189 and :191 pass lag 0 for every value and no value for a user input (:192-193), and mesh_plan replaces a
    nonzero lag with the bound it computes (rdma/mesh-collective.c:1921). "The CPU code writes each copy, and the
    recorded kernels read it.": compiler/mesh_compiler/runtime.py:61-64 binds each call's inputs as the caller's
    tensors.
  - D.3.4 algorithm: "**The algorithm** is one commit per later pass (C3.1)", with its steps 1 to 3:
    compiler/mesh_compiler/lower_metal.py has no later pass; the one waitForEvent:value: (rdma/mesh-metal4.m:141)
    waits for torch's MPS commands in mesh_metal4_after (lower_metal.py:1038-1039), not at p + 1 before a later pass.
  - D.3.4 "Written by, read by": "`tools/e2b_replay.m`, `metal_recording.m` and `tools/native_bench.m` make and commit
    the command buffers.": in compile_module's flow the command buffers are made and committed by
    rdma/mesh-metal4.m:163-206 (mesh_metal4_commit), called from compiler/mesh_compiler/lower_metal.py:1049 (HEAD);
    the three files named are outside that flow.
  - D.3.4 delegate's capture: "The delegate's capture uses an `MTLCommandBuffer`: ml_drift encodes into one it is
    given, and commits its own loads during the capture." and "The recorder's queue for the delegate exists only for
    the capture, and nothing the run executes passes through it.": compiler/mesh_compiler has no capture of a
    delegate, and metal-microbench's .m, .mm, .h and .cc files at HEAD have no MTLCommandQueue and no
    id<MTLCommandBuffer>; the one MTLCommandBuffer in the flow is torch's, signalled in rdma/mesh-metal4.m:138.
  - D.3.6: "Where a node's map serial moves away from a static schedule's serial, its bridge cancels every receive of
    that static schedule that has not completed" and "The bridge (`rdma/mesh-flow.c`) compares the serial with the
    static schedule's and cancels.": plan_stale compares the schedule's struct mesh_view_id with the installed view
    (rdma/mesh-flow.c:2547-2552) and is called where a view is cleared or installed (rdma/mesh-records.c:141, :693),
    not where map_serial moves (:111). "Every node of the static schedule then reads ENOLINK, with no case for
    relays.": the readers Program calls return ECANCELED (rdma/mesh-collective.c:77) and ESTALE
    (rdma/mesh-dataflow.c:237-268); ENOLINK is only in rdma/mesh-metal.m:278. "The recorded kernels' later pass reads
    the counts and the phases (C3.1).": compiler/mesh_compiler/lower_metal.py has no later pass.
  - D.4 mesh: rdma/mesh-flow.c written: "Written, the cancellation of a static schedule's receives where the map
    serial moves (D.3.6).": plan_stale cancels where the schedule's view identifier is not the installed view's
    (rdma/mesh-flow.c:2552) and is called where a view is cleared or installed (rdma/mesh-records.c:141, :693); the
    move of map_serial (rdma/mesh-records.c:111) calls no cancellation.
  - C3.2 (d) nothing laid out when created: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    design/disjoint-pages-change-request.md:841-842 "Nothing is laid out when it is created, and no program writes a
    page to fault it in: the region is wired once (data-model §2).": compiler/mesh_compiler/lower_metal.py:1142-1143:
    when the lowered function is made, each held storage gets its own buffer from LIB.mesh_metal4_buffer; :1180: at
    every call each result gets a new one. rdma/mesh-metal4.m:85-88: mesh_metal4_buffer is [device
    newBufferWithLength:options:]. These values are laid out by Metal where they are created, not in pages mesh_plan
    placed; the placed address (bridge.py:51-53, runtime.py:25, :48, :71) has no reader. The index already records the
    same allocations under C3.2 (b), C3.3 compile-time bound and A3 "Which function computes it".
  - C3.3 algorithm 3 (copies, t mod u): outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    design/disjoint-pages-change-request.md:851 "One execution's graph is unrolled into u copies, and execution t runs
    copy t mod u.": Only the values carried between machines run copy t mod u (compiler/mesh_compiler/bridge.py:65,
    :74, :82; rdma/mesh-flow.c:518-521). The values compile_module gives the plan for the local functions
    (compiler/mesh_compiler/schedule.py:188-191) get u copies (rdma/mesh-collective.c:196-197) that no execution runs:
    Published.address (bridge.py:51-53) is stored as Program.storage (runtime.py:25) and ir.LOCAL.storage
    (runtime.py:48, :71) and read nowhere in compiler/mesh_compiler. The lowered function binds one Metal buffer per
    storage when it is made (lower_metal.py:1142-1143, :1162), the same address in every execution. With u = 1 the
    difference is only that those values are outside the plan's pages; with u > 1 the sentence fails for them
    outright.
  - C4 mesh deleted: mesh_plan_erased: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    design/disjoint-pages-change-request.md:923-924 "rdma/mesh-call.h and rdma/mesh-dataflow.c: `mesh_plan_erased`,
    and its seq_cst fence pair with the ports' code in rdma/mesh-flow.c.": The name is gone (comment only,
    rdma/mesh-call.h:109), but what it named is in the code as plan_released. mesh_plan_erased (mesh ae0bc2f1
    rdma/mesh-dataflow.c:447-465) was the plan call's wait, bounded by a deadline, until no port's `prepared` held the
    previous plan, before rewriting its state. rdma/mesh-collective.c:1813-1822 waits until the bridge stores
    `released` or the deadline passes; it is called at :1986 before mesh_plan rewrites the state at :1993, and at
    :2390 before mesh_plan_load does at :2393 (from compiler/mesh_compiler/schedule.py:111 and bridge.py:25-28).
    `released` is stored only once no port's `prepared` holds the plan (rdma/mesh-dataflow.c:110-116, :128-129),
    behind the seq_cst fence at :127 paired with the ports' fence at rdma/mesh-flow.c:1946-1947. The index lists C3.4
    (b) and (c) outstanding for this same wait.
  - D.3.4 data structure: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    design/disjoint-pages-change-request.md:1409-1410 "Every command buffer this code makes is one: the decode, the
    prefill, the preparation and fills, the profiling, the recorder's own copies and tools/native_bench.m.":
    rdma/mesh-metal4.m:136-140 (mesh_metal4_after): on torch's dispatch queue it calls torch's commit, takes torch's
    command buffer through get_command_buffer as an `id<MTLCommandBuffer>`, encodes its own signal into it
    (encodeSignalEvent:arrival value:arrived, :138) and commits it; the MTL4 queue then waits on that event (:141).
    compiler/mesh_compiler/lower_metal.py:1196-1197 calls it in each run of a lowered function that has an operand on
    mps (the three torch symbols are bound at :78-81). So a classic MTLCommandBuffer carries a command of this code in
    what the run executes, outside any delegate capture. The object is created by torch at this code's call; if
    "makes" is read as "allocates itself", the sentence holds at rdma/mesh-metal4.m:121-122, :170-171 and :205.
  - D.3.5: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. design/disjoint-pages-change-request.md:1428-1429
    "The allocator already orders the adjacent node's next send into that page after this node's reads of the last one
    (C3.3).": For a value whose read reaches no later write in the dependence graph, the allocator gives no order:
    rdma/mesh-collective.c:410 sets its lag to 0 and it adds no copy (the note at :374-377 says its reuse is held at
    run time). That is the case for a value carried one way through compile_module: rdma/mesh_graph.py:75-78 adds a
    lag-1 edge only between events of one processor, and the plan's own edges (rdma/mesh-collective.c:280-294) are all
    lag 0, so the destination's read (plan_consumed) has no path to the source's next write. What orders the next send
    after the reads is a run-time gate: the bridge takes execution t on a lane only below the plan's `executions`
    (rdma/mesh-flow.c:1354, for receives at :1499), which the node's own Program raises at each call
    (compiler/mesh_compiler/bridge.py:41; rdma/mesh-dataflow.c:274-277). The clause's other sentences hold: a record
    is posted again only once its range's count shows the last round landed (rdma/mesh-flow.c:1508); the lists are at
    rdma/mesh-collective.c:2052-2077.

Covered:

  - A3 "What it writes": covered [2026-11-12, mesh 81c8e9da]
  - A3 "The bridge's record pages": covered [2026-11-03, mesh b4e1915b]
  - A4 deleted (1): covered [2026-11-03, mesh ae0bc2f1]
  - A4 deleted (2): covered [2026-11-03, mesh fd49e3b5]
  - A4 deleted (3): covered [2026-11-04, mesh 04b5e561]
  - A4 deleted (5): covered [2026-11-05, mesh c6c9c77f]
  - C3.1 waits deleted: covered [2026-11-12, mesh a86c7dc4]
  - C3.3 algorithm 1 (L): covered [2026-11-12, mesh 81c8e9da]
  - C3.3 algorithm 2 (u): covered [2026-11-12, mesh 81c8e9da]
  - C3.4 "Written by, read by": covered [2026-11-12, mesh 81c8e9da]
  - C3.6 (3) local countdown: covered [2026-11-04, metal-microbench a159c59a]
  - C3.6 (4) link loss and poll bound: covered [2026-11-04, mesh 04b5e561]
  - C3.7 data structures: covered [2026-11-12, mesh 81c8e9da]
  - C3.7 algorithm 1: covered [2026-11-04, mesh 04b5e561]
  - C3.7 algorithm 2: covered [2026-11-11, mesh 6d4f007b]
  - C3.7 algorithm 3: covered [2026-11-05, mesh c6c9c77f]
  - C3.7 algorithm 4: covered [2026-11-12, mesh 0704bf95]
  - C3.7 algorithm 5: covered [2026-11-04, mesh 04b5e561]
  - C4 mesh deleted: rdma/mesh-collective.c (executions, per-execution items): covered [2026-11-04, mesh e2dcc0cf]
  - C4 mesh deleted: rdma/mesh.h areas and constants: covered [2026-11-04, mesh 533e55a8]
  - C4 metal-microbench deleted: metal_recording.m: covered [2026-11-09, metal-microbench f08d3c7b]
  - C4 metal-microbench deleted: tools/e2b_replay.m: covered [2026-11-04, metal-microbench 6fb94beb]
  - D.3.4 completion: covered [2026-11-12, mesh a86c7dc4]
  - D.4 mesh: rdma/mesh-flow.c deleted: covered [2026-11-04, mesh 1258766d]
  - D.4 mesh: coordination document: covered [2026-11-04, mesh 51523f66]
  - D.4 metal-microbench deleted (1): classic command queues and buffers: covered [2026-11-10, metal-microbench
    e1857e71]
  - D.4 metal-microbench deleted (2): per-step instances: covered [2026-11-04, metal-microbench 9a196d2a]
  - D.4 metal-microbench deleted (3): delegate buffers as captured: covered [2026-11-09, metal-microbench f08d3c7b]

Withdrawn:

  - D.3.1 (and the turn's item 1: several programs, MESH_GROUP queue-pair assignment, EBUSY): withdrawn [2026-11-12
    reading]. design/disjoint-pages-change-request.md:1136-1138: "I reasoned from a queue-pair count that belongs only
    to the bridge and differs between macOS 26 and 27, and I turned it into a limit programs would see (`EBUSY`). That
    is withdrawn, and I stopped the coordinator before it wrote anything."

## [page-queue-change-request.md](page-queue-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 26 items outstanding, 0 not determined, 12 covered, 0 withdrawn.

Outstanding:

  - The operator's statements: The E2B code, the bridge and two layers: "being told to change the e2b running code so
    that > it can interact with a bridge connected to some number of peers within a total collective of 5 machines?"
    The E2B running code is metal-microbench tools/e2b_replay.m (mesh_attach at :1039, mesh_plan at :1322); no caller
    of compile_module decodes E2B (mesh examples/ holds cp, crossing, engines, ep, pl and tp).
  - The operator's statements: The page queue: "there > must be some kind of zero copy queue" On the compile chain's
    send path every sent value is copied into the transfer's region pages before it is published:
    compiler/mesh_compiler/bridge.py:73-76 `flat.copy_(value.as_strided(flat.shape, (1,)))`, reached from ir.py:13-14
    `LOCAL.channels[channel].put(value)`.
  - The operator's statements: Replacing the --link form: "ensuring it *covers all features and requirements the
    previous implementation covered*" The previous implementation carried the E2B decode; at HEAD the only producer of
    the bridge's input for E2B is metal-microbench tools/e2b_replay.m:1322, and no caller of compile_module decodes
    E2B.
  - The operator's statements: On how parity is shown: ""3. Parity is shown by the normal program, " — yes." The E2B
    decode is run only by metal-microbench tools/e2b_replay.m (launched at tools/mesh/run_pair.py:280-291); no caller
    of compile_module is an E2B program.
  - The operator's statements: The bridge's process lifecycle: "then you have to verbatim reuse that process lifecycle
    code" rdma/mesh-heartbeat.c differs from tag pre-return-20261103 (git diff: 53 lines; its includes, the
    bridge-pages lookup, the holders array `3*MESH_PORTS+1` and the client holder read, `clock_ns`). The per-port
    children are not the tag's rdma/mesh-link.c: `git diff --stat pre-return-20261103 HEAD -- rdma/mesh-link.c` is
    2075 changed lines, the children being in rdma/mesh-children.c and rdma/mesh-child.c, where
    rdma/mesh-children.c:217-221 omits the tag's stores into the port's streams.
  - Claude's proposal: The page lists (the queue): the data structure: "An entry is one transfer of one page between
    this node and an adjacent node: `{ page, node, direction, word, position }`." The entry is rdma/mesh.h:80-81
    `struct mesh_hop { uint32_t page,node,direction,field; uint64_t word,count; uint32_t pages,bytes; uint64_t serial;
    uint32_t set,offset; }`: no `position`; it carries `bytes` from `offset` within the page
    (rdma/mesh-collective.c:2064-2066), 1 to block x pgsz bytes.
  - Claude's proposal: The page lists: word, position: "`word`, `position`: a word of the region and a value. A send
    is posted once its word reads its value." The entry holds no value: rdma/mesh-flow.c:1650 posts a send where
    `field` is absent once the word exceeds the execution's number (`word<=at.execution` holds it back), else once the
    two bits at `field` read `mesh_phase(at.round+1)`. "A receive's completion stores its value in its word."
    rdma/mesh-flow.c:1474-1475: a completion XORs one bit of the word at `field` and adds 1 to the entry's `count`
    word.
  - Claude's proposal: The page lists: no copy, not a ring: "Nothing is copied." Each value sent between machines is
    copied into the entry's region pages before it is published: compiler/mesh_compiler/bridge.py:73-76
    `flat.copy_(value.as_strided(flat.shape, (1,)))`, called by ir.py:13-14.
  - Claude's proposal: The page lists: Consumed by: "**Consumed by** the bridge alone, in `mesh-flow.c`." The client
    library also reads them: rdma/mesh-dataflow.c:288-298, in mesh_host_started (called from
    compiler/mesh_compiler/bridge.py:41), walks the plan's hops and raises each RECEIVE hop's `count` where the plan's
    view is not the installed one; rdma/mesh-metal.m:91 reads `mesh_hops(m,words)`.
  - Claude's proposal: The schedule: every live node runs a rank: "Every live node of the Thunderbolt-5-connected
    collective runs a rank, so an intermediate node's entries are written on that node." In the compile chain the
    ranks are the nodes MESH_PROGRAM lists (schedule.py:158), and mesh_plan refuses a route through a node outside
    that list (rdma/mesh-collective.c:1791-1794, EINVAL); the one launch of a compile_module program on machines sets
    MESH_PROGRAM to "0,1" (compiler/samples.py:1284-1285). run_pair.py's ranks are tools/e2b_replay.m.
  - Claude's proposal: The assignment: Its step: "is assigned to the (cable, queue pair) whose cumulative bytes over
    the cable's bandwidth finishes earliest" rdma/mesh-flow.c:820 gives each (cable, queue pair) the rate
    `cables[c].bandwidth/cables[c].qps`, and rdma/mesh-graph.c:529 divides its cumulative bytes by that rate: the
    bandwidth over the cable's queue-pair count, not the cable's bandwidth.
  - Claude's proposal: The assignment: It writes: "and a sends in that order as each page's `word` reads its
    `position`." The entry has no `position` (rdma/mesh.h:80-81); rdma/mesh-flow.c:1650 posts once the word exceeds
    the execution's number, or once the two bits at `field` read `mesh_phase(at.round+1)`. Members of a nonzero `set`
    are posted in each execution's permutation of their positions (rdma/mesh-flow.c:1496, 1647 position_record), not
    the list's order.
  - Claude's proposal: The changes, by file: The bridge: "post each send once its `word` reads its `position`, and
    store each completed receive's `position` in its `word`." The entry has no `position` (rdma/mesh.h:80-81): a send
    is posted once its word exceeds the execution's number or its two bits at `field` read `mesh_phase(at.round+1)`
    (rdma/mesh-flow.c:1650); a completed receive XORs one bit of its word at `field` and adds 1 to its `count` word
    (rdma/mesh-flow.c:1474-1475).
  - Claude's proposal: The changes, by file: The bridge: process lifecycle: "The process lifecycle is the code of tag
    `pre-return-20261103`, verbatim (`rdma/mesh-heartbeat.c`; `rdma/mesh-link.c` per-port children, their stop,
    reaping, flush and exit; the holder table)." rdma/mesh-heartbeat.c differs from the tag (git diff
    pre-return-20261103 HEAD: 53 lines, among them the bridge-pages lookup, `pid_t holders[3*MESH_PORTS+1]`, the
    single client holder, `clock_ns`). rdma/mesh-link.c differs from the tag's by 2075 lines; the children's code is
    in rdma/mesh-children.c and rdma/mesh-child.c, and rdma/mesh-children.c:217-221 leaves out the tag's stores into
    the port's streams.
  - Claude's proposal: The changes, by file: The callers: tools/e2b_replay.m: "`tools/e2b_replay.m` lines 497-500 pass
    the program's declared receives and sends (operands.json's names, ranks and ranges) to that function instead of
    binding transfers;" The pass is in metal-microbench tools/e2b_replay.m itself (operands.json read at :796,
    mesh_plan at :1322), a program outside the compile_module chain; no caller of compile_module passes E2B's declared
    receives and sends.
  - Claude's proposal: The changes, by file: The callers: tools/mesh/grid.py: "`tools/mesh/grid.py` prepares the
    programs for the collective's rank count with an explicit placement operand (shares per rank)." grid.py gives the
    shares to tools/prepare_e2b.py (`--placement {args.shares}`, tools/mesh/grid.py:22, 66-70), which prepares the
    programs outside the compile_module chain; no placement operand of compile_module
    (compiler/mesh_compiler/driver.py:14, 19) comes from it.
  - Claude's proposal: Order, bullet 1: "The replacement above is finished until it covers everything the `--link`
    form and the old calls did." The old calls' client, the E2B decode, reaches the bridge only from metal-microbench
    tools/e2b_replay.m:1322; no caller of compile_module decodes E2B. Of the replacement's own clauses, the entries
    have no `position` (rdma/mesh.h:80-81), the lifecycle code differs from the tag (rdma/mesh-heartbeat.c,
    rdma/mesh-children.c:217-221), and tools/mesh/grid.py:66-70 gives its placement to tools/prepare_e2b.py.
  - Claude's proposal: Order, bullet 3: "Parity is shown by the normal program, the E2B decode on the collective, in
    an end-to-end run." The E2B decode is metal-microbench tools/e2b_replay.m (launched at
    tools/mesh/run_pair.py:280-291); no caller of compile_module decodes E2B, so no run of the compile chain shows it.
  - Claude's proposal: Order, bullet 4: "Every wait is bounded" rdma/mesh-children.c:216
    `while(waitpid(pid,&status,0)<0 && errno==EINTR){}` has no bound. In the Program that reaches the bridge,
    compiler/mesh_compiler/runtime.py:75 `self.finished.get()` and :50 `self.argument_queues[processor].get()` have no
    timeout (a processor's thread that raises on mesh_host_wait's error never puts to `finished`, runtime.py:50-52),
    and rdma/mesh-metal4.m:212 waits with `timeoutMS:UINT64_MAX`.
  - The operator's statements: Collective size: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    design/page-queue-change-request.md:12-13: "all code now must operate on collectives of arbitrary > node count and
    there won't be any compromises here.": The node count is capped at 32: rdma/mesh-dataflow.h:42 `#define
    MESH_NODES_MOST 32`, and rdma/mesh-dataflow.c:189 `if(nodes>MESH_NODES_MOST)return E2BIG;` in mesh_view_copy,
    which mesh_nodes_read (:218) calls for every plan (compiler/mesh_compiler/schedule.py:132;
    rdma/mesh-collective.c:1935). A view of 33 nodes cannot be planned. Members are also bits of 64-bit words
    (mesh-collective.c:1700 `d->leaves|=UINT64_C(1)<<t->destination`; schedule.py:158 `1 << int(node)`). No pair-only
    code was found in the compile chain; the refutation is the literal bound against "arbitrary".
  - The operator's statements: Relays: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    design/page-queue-change-request.md:18: "routing within tb5 linked collectives is supported": The route is solved
    over the map, not within the collective. rdma/mesh-collective.c:1728-1731 builds the flow graph from every node
    pair of the map with capacity, whether or not the node is in `program`; mesh_graph_flow (:1758;
    rdma/mesh-graph.c:306-333) picks shortest trees over all n map nodes; then :1792-1794 returns EINVAL when any tree
    passes a node outside `program` (`if(spanned&~program)`). `program` is used nowhere else in plan_function. For a
    collective that is a proper subset of the map (MESH_PROGRAM, schedule.py:158; "0,1" at
    compiler/samples.py:1284-1285), a relay through a non-member fails the plan instead of being routed through
    members. Routing through member relays is written (plan_walk :771-783; relay page :2058).
  - The operator's statements: Queue pairs and links belong to the bridge: outstanding [2026-11-12, mesh 6d379b7e; a
    second reader]. design/page-queue-change-request.md:36-38: "whether 'queue pairs' or 'streams' or any attempt to
    expose ortransduce the global object > representing the literal connection between two physical and
    different-clock-domain literal computers are ever to be > exposed to collective or data processing code":
    Collective code reads the cables' ports and queue-pair counts in the compiling process.
    rdma/mesh-dataflow.c:134-153 mesh_cables, linked into libmesh.dylib (rdma/Makefile libmesh.dylib rule), matches
    cables by `port`/`peer_port` (:142) and takes `.qps=e->qps<f->qps?e->qps:f->qps` (:144) from the map's struct
    mesh_record copies (:215-218); it is reached from compiler/mesh_compiler/schedule.py:132 and bridge.py:99
    (mesh_nodes_read -> mesh_capacity :206, :225) and from mesh_plan (rdma/mesh-collective.c:1935). The same fact is
    why the index lists plan-to-bridge R7 outstanding. Also rdma/mesh-dataflow.c:111-113 reads the per-port link table
    (mesh_links(m)[p].port.prepared) in library code.
  - The operator's statements: Failing by error, not by hardware fault: outstanding [2026-11-12, mesh 6d379b7e; a
    second reader]. design/page-queue-change-request.md:130-131: "some spinwaits are allowed, as a treat, if we know
    they conclude in bounded time even if someone messes up their > complementary async event originators":
    compiler/mesh_compiler/schedule.py:71-75, image(): `while True: before = header.view_map; C.memmove(...); if
    mesh.LIB.mesh_view_whole(live.M, before): break`. A busy retry with no deadline: mesh_view_whole
    (rdma/mesh.h:288-291) is false while view_map is odd, so a bridge that stops mid-install leaves compile_module
    spinning forever. image() is on the compile chain (driver.py:24 planned -> schedule.py:130; driver.py:26 stored ->
    :196). The C reader of the same word has a deadline (rdma/mesh-dataflow.c:177). `down_pair` has no declaration or
    definition in either repository.
  - Claude's proposal: The schedule that writes the page lists: the algorithm: outstanding [2026-11-12, mesh 6d379b7e;
    a second reader]. design/page-queue-change-request.md:210-211: "every collective call and > pipeline hand-off,
    over the map's Thunderbolt-5-connected nodes, for any node count.": Not for any node count:
    rdma/mesh-dataflow.c:189 returns E2BIG above MESH_NODES_MOST (rdma/mesh-dataflow.h:42, 32), before mesh_plan
    reaches plan_function (rdma/mesh-collective.c:1935, 1946); destinations are bits of a 64-bit word (:1700). The
    rest was found as stated: one mesh_graph_flow over every demand (:1757-1758), run at compile time
    (compiler/mesh_compiler/schedule.py:111 from driver.py:24).
  - Claude's proposal: The schedule: It takes: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    design/page-queue-change-request.md:214-215: "It takes no cable, queue pair or link > identity.": mesh_plan reads
    the map's cable records itself: rdma/mesh-collective.c:1935 mesh_nodes_read -> rdma/mesh-dataflow.c:215-218 copies
    each node's struct mesh_record (its reach[]: port, peer_port, qps, bandwidth; rdma/mesh-map.h:80, 108-110) into
    the planning process, and :142-144 mesh_cables identifies each cable by its two port indices and takes its
    queue-pair count before :206 sums the bandwidths. The map gives it cables, not "each node pair's capacity"; the
    reduction to pairs is done inside the schedule's own call. struct mesh_declared and mesh_plan's parameters carry
    none (rdma/mesh-collective.h:37, 90).
  - Claude's proposal: The changes, by file: The callers: tools/mesh/run_pair.py: outstanding [2026-11-12, mesh
    6d379b7e; a second reader]. design/page-queue-change-request.md:259: "the driver starts no bridge, chooses no
    link, and runs one rank on every node of the collective": The rank the driver runs is the E2B replay program,
    outside the compile_module chain: metal-microbench tools/mesh/run_pair.py:290 `exec env ... .build/e2b_replay
    {REGION} ...` (launch, :280-291). No program of compile_module/runtime.Program is launched by it, so "runs one
    rank on every node" holds only through tools/e2b_replay.m, as the index already records for the sibling items
    (e2b_replay.m, grid.py). The deleted names (`discover_link`, `links`, `start_bridge0`, `start_bridge1`,
    `bridges_up`, `bridges_down`) are gone from both repositories, and :108-150 takes every member of the map.

Covered:

  - The operator's statements: Interleaving every transfer: covered [2026-11-12, mesh 81c8e9da]
  - Claude's proposal: The page lists: page, node, direction: covered [2026-11-12, mesh 81c8e9da]
  - Claude's proposal: The page lists: Produced by: covered [2026-11-12, mesh 81c8e9da]
  - Claude's proposal: The page lists: Replaces: covered [2026-11-03, mesh 4ac5cbdc]
  - Claude's proposal: The schedule: blocking transfers first: covered [2026-11-12, mesh 81c8e9da]
  - Claude's proposal: The schedule: It writes: covered [2026-11-12, mesh 81c8e9da]
  - Claude's proposal: The assignment of pages to cables and queue pairs: the algorithm: covered [2026-11-03, mesh
    c193b11a]
  - Claude's proposal: The assignment: It takes: covered [2026-11-03, mesh c193b11a]
  - Claude's proposal: The changes, by file: The bridge: queue-pair count: covered [2026-11-03, mesh 615aa4e7]
  - Claude's proposal: The changes, by file: The collective: one function: covered [2026-11-12, mesh 81c8e9da]
  - Claude's proposal: The changes, by file: The collective: replaced calls: covered [2026-11-03, mesh 1444f409]
  - Claude's proposal: Order, bullet 2: covered [2026-11-03, mesh a37d5a74]

## [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 34 items outstanding, 0 not determined, 32 covered, 0 withdrawn.

Outstanding:

  - 1, step 1: "Block j of member i (j ≠ i) is a transfer from the tensor view of block j in i's operand to the tensor
    view of block i in j's result": the received block is a value of its own that aten.cat joins into j's result
    (placement.py:212-213), not a tensor view of the result. In mesh_plan's operand,
    compiler/mesh_compiler/schedule.py:151-152 writes one layout per transfer with offset 0, and rdma/mesh.py:307
    gives that one layout as both `from` and `to`; schedule.py:144-145 gives a sent tensor view a value of its own
    (named after the send) instead of its operand's.
  - 1, step 2: "Block i is one local copy from the operand's tensor view to the result's tensor view, as
    MPIR_Localcopy does it." and "It is torch's `copy_` ..., a node of the program, with no synchronize and no
    clone.": placement.py writes no `copy_` node anywhere; block i is one operand of the aten.cat that writes the
    whole result (placement.py:212-213).
  - 1, step 3: "The local copy is absent only where the operand is the result with the same type map (MPI's in-place
    condition).": the `copy_` local copy of step 2 is absent everywhere (no `copy_` in placement.py); block i reaches
    the result through aten.cat (placement.py:212-213), or is the result itself where there is one piece
    (placement.py:212).
  - 2, step 1: "Where the producer of member i's input writes through an `out` tensor, its `out` is the tensor view of
    block i of the result.": no code of compiler/mesh_compiler/placement.py sets a node's `out`; the member's input is
    a piece of an aten.cat (placement.py:203-213). compiler/mesh_compiler/bridge.py:64-65 (`Carried.out`) has no
    caller in compiler/mesh_compiler.
  - 2, step 2: "Otherwise block i is one local copy into that tensor view, as MPICH's `sendbuf != MPI_IN_PLACE` branch
    does, with the same form as item 1's.": no `copy_` node is written; block i is an operand of the aten.cat that
    writes the whole result (placement.py:212-213).
  - 3, Change: the rule: "A tensor view node's entry in the memory plan is its base's entry, with the view's offset,
    sizes and strides as torch computes them." and "the placement stage writes a transfer's end that is a tensor view
    node as its base's value with that offset, those sizes and those strides (item 10's form), as the `from` or the
    `to` (`struct mesh_strided`) of a `struct mesh_declared` in `mesh_plan`'s operand.": schedule.py:144-150 gives a
    sent tensor view (`of(sent.args[0])` not None) a value of its own, named after the send node; schedule.py:151
    writes the layout with offset 0 and the received tensor's sizes and strides; rdma/mesh.py:307 uses it for both
    `from` and `to`. The operand is written in schedule.py `planned`, not in placement.py.
  - 3, Change: the result torch allocates: "Only a result torch allocates outside the memory plan's pages has no
    entry.": schedule.py:185-193 also gives no entry to a value among the program's results, to a placeholder that is
    not state (a user input) and to mutated state. "That case is item 2's local copy.": the copy is `flat.copy_`
    inside the channel's `put` at run time (bridge.py:75), called by `mesh::send` (ir.py:13-14) for every carried
    send; it is not a `copy_` node the placement stage writes in the exported graph.
  - 4, step 1: "A replicated operand's window is a DynamicSlice of the local operand: a tensor view of its pages, with
    no transfer.": where the window's read range passes the operand's edge, placement.py:337-342 makes the window an
    aten.cat of the slice with new_zeros/new_full parts, which is not a tensor view.
  - 4, step 2: "then masking": placement.py:337-341 writes no masking node; it concatenates new_zeros or new_full
    parts for the missing range (placement.py:342). "for each spatial dimension": one dimension only (see step 3).
  - 4, step 3: "the exchanges run one sharded dimension at a time, in increasing dimension order." and "Each
    exchange's operand is the previous exchange's result, so the second exchange's halos carry the first's, and the
    corners arrive with them.": placement.py:487-505 takes one window, of the one dimension `target[0]`, and calls
    `halos` once; a value holds one sharded dimension (placement.py:497), so no second exchange is written.
  - 4, step 4: "is in place where its producer's `out` is the tensor view of it in the window (item 2's condition),
    and one local copy otherwise.": no `out` is set and no `copy_` node is written; the own shard is one part of the
    aten.cat that makes the window (placement.py:335, 342).
  - 5, step 2: "A pair (p, p) is item 1's local copy at p.": no `copy_` node is written; p's own part goes into the
    aten.cat of placement.py:342, or is used as it is where it is the only part.
  - 5, step 3: "A replica that is no target outputs zeros.": placement.py:340-341 writes new_full of -inf or the
    type's minimum for max pooling (placement.py:273-274), and writes no node where `missing` is not positive.
  - 6, step 1: "Test torch's condition for the new tensor view's dimensions": the test runs only for an aten.reshape
    node whose operands are all invariant, and its result is read only for the `invariant` mark
    (placement.py:627-632); no other tensor view node is tested.
  - 6, step 2: "Where it holds, the node is a tensor view.": placement.py writes or copies aten.view and aten.reshape
    nodes without reading the test (placement.py:128, 132, 560); what the node is at run time is left to torch's own
    reshape.
  - 6, step 3: "Otherwise it is `contiguous` into a value in the memory plan, run as an out-variant copy.":
    placement.py writes no `contiguous` node and no out-variant copy; schedule.py gives no value for one.
  - 7.1, step 1: "A value's lifetime is that pair, as `update_tensor_lifetime` computes it": for a value that is a
    transfer's end, schedule.py:181-182 writes `end` = len(local) - 1, the machine's last operation, whichever node
    refers to it last; a received value is among them.
  - 7.1, step 2: "A transfer's ends each occupy their value from the graph node that starts the transfer to the graph
    node that waits for its completion": no graph node waits (7.2 step 1 is not written); schedule.py:181-182 takes
    the producer's index or 0 and the last local operation.
  - 7.2, step 1: "Each collective has one completion wait at each node: torch's `wait_tensor`, placed at the graph
    node before the first use of the collective's result there.": compiler/mesh_compiler writes no `wait_tensor` node.
    The waits are run-time calls: each receive's at its channel's `get` (bridge.py:79-82) and every send's after the
    functions return (bridge.py:45-48, runtime.py:76-77).
  - 7.2, step 2: "That wait completes the node's whole participation (MPI 4.1 §7.1): its sends, its forwards and its
    receives.": there is no one wait; receives are awaited one channel at a time in `Carried.get` (bridge.py:79-82),
    sends and forwards in `Published.end` at the end of the execution (runtime.py:76-77).
  - 7.2, step 3: "A send's source value and a forwarded block are occupied until that graph node": no such graph node
    exists; schedule.py:182 writes `end` = len(local) - 1.
  - 7.3, step 3: "The collective's `wait_tensor` at a node waits for its receive counts and its send-completion
    counters": there is no `wait_tensor`; the receive count is awaited in `Carried.get` (bridge.py:79-82) and the
    send-completion counters separately in `Published.end` (bridge.py:45-48).
  - 8, step 5: "Each is loaded once in make(), before execution 0: the node's slice is taken from the state_dict
    tensor ... and copied into the value's pages by torch's `copy_`": compiler/mesh_compiler/runtime.py:31-35 binds
    each state tensor whole as an argument; the slices stay nodes of the graph run at every execution; no code copies
    a slice into the pages of its MESH_INVARIANT value (runtime.py:25 keeps `storage`, which nothing in
    compiler/mesh_compiler reads).
  - 8, step 6: "The whole attribute is loaded only where the node reads it whole.": runtime.py:27-35 binds the whole
    state tensor on every processor whose graph has its placeholder, which schedule.py:220 gives to any processor with
    a user of it, a slice among them; archive.py:73-75 saves the whole tensor for each such processor.
  - 10, Composition: the placement stage writes it: "the placement stage writes it for each transfer's end, as `struct
    mesh_strided` in `mesh_plan`'s operand.": it is written in schedule.py `planned` (schedule.py:151), not in
    placement.py; one layout a transfer, with offset 0, which rdma/mesh.py:307 gives as both `from` and `to`.
  - 11, step 4: "The persistent request's replacement is compiled on the next membership view.": mesh_plan is called
    once, inside compile_module (schedule.py:111, driver.py:24-26), and its saved record is loaded once at the
    Program's construction (bridge.py:25, runtime.py:21-23); no code of compiler/mesh_compiler calls either on a later
    view (mesh_plan_load returns ESTALE there, rdma/mesh-collective.c:2247).
  - 13, step 2: "All its send-queue completions are consumed.": the loop stops at `until`, or once none has come for
    MESH_FLUSH_QUIET_NS, with `left[0]` still above 0 (rdma/mesh-child.c:63-66).
  - 14, step 3: "Each live range is one lifetime interval" of G: every value compile_module writes has `run` 1
    (schedule.py:149, 189, 191); mesh-collective.c:507 leaves a `run` value out of the PLAN_LOCAL items, and
    mesh-collective.c:596-613 places its live ranges by mesh_graph_greedy_by_size, not as intervals of G.
  - 7.3, step 2: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "The port's child process raises the
    counter by each work completion it retrieves for that range's sends or forwards."
    (design/problems-of-14b9a757-change-request.md:381-382): rdma/mesh-flow.c:1544-1549: sends_complete polls the
    link's one send completion queue (`l->provider.sent`, shared by its MESH_LINK_LANES = 10 lanes) and, for a
    completion whose lane is not in `served`, does `continue` before either prepared_send_counted call (1552, 1557).
    lane_lost (mesh-flow.c:345-347) clears one lane's bit of `served` while the other lanes stay served, so port_pass
    (2129-2130) keeps calling link_run and the lost lane's SEND completions are taken from the queue with no counter
    raised. Nothing counts them later: the other callers of prepared_send_counted are sends_cancel (unposted records,
    1572-1575) and lanes_flush (rdma/mesh-child.c:74), which counts only completions it polls itself.
  - 7.3, step 4: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "When the port's child process finds a
    persistent request cancelled, it first retrieves the completions of the sends it has posted. Each raises the
    count, or the error count where it completed in error, as fi_cancel has a completed operation report its own
    status." (design/problems-of-14b9a757-change-request.md:385-387): rdma/mesh-flow.c:2094-2099: on finding the
    flow's view replaced, link_run calls sends_complete once, which takes only the completions already in the queue
    (1543-1544, a poll that stops at the first short batch), then sends_cancel (2098) raises the error counts of the
    unposted sends (step 5) while posted sends are still outstanding (2099: `f->posted!=f->sent+f->landed`,
    plan_outstanding ends the link). Those sends' completions are retrieved after step 5, at the child's exit
    (rdma/mesh-child.c:63-75), where each raises the error count whatever its status (mesh-child.c:74). In the same
    sends_complete, a completion on a lane no longer in `served` raises neither count (mesh-flow.c:1549).
  - 8, step 2: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "A value made from invariants by tensor views
    alone is replicated too: a tensor view preserves its operand's dimensions, the sharding of a preserved dimension
    is propagated from input to output, and propagation creates no new sharding pattern (GSPMD §3.5), so a replicated
    operand's tensor view is replicated at every node." (design/problems-of-14b9a757-change-request.md:420-422):
    compiler/mesh_compiler/placement.py:627-629 marks a node invariant only where its target is one of the nine
    operators of VIEWS (placement.py:34-35: expand, reshape, slice, view, alias, permute, select, squeeze.dims,
    unsqueeze). A tensor view by any other operator is left out: `aten.split_with_sizes` with its `operator.getitem`
    results (what torch.split, the call the clause's Megatron source quotes, becomes) and `aten.diagonal`. Both are
    core ATen views that run_decompositions (export.py:19) keeps; lower_metal.py:340 lowers split_with_sizes. Such a
    view of a state tensor is not invariant, so placement.py:633-637 does not copy it onto its readers (it reaches a
    placeholder), it stays on the one processor placement.py:495-496 gave it, and placement.py:660-680 writes a send
    and a receive for a reader elsewhere. schedule.py:96-98 does treat these as views, by the schema's alias_info, so
    the two stages disagree.
  - 8, step 3: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "A collective whose operand is such a value
    is a resharding from replicated, so it is the DynamicSlice of its result's sharding, at each node, with no
    transfer." (design/problems-of-14b9a757-change-request.md:423-424): For a value made from a state tensor by a view
    operator outside VIEWS (compiler/mesh_compiler/placement.py:34-35; `aten.split_with_sizes` with getitem,
    `aten.diagonal`), the value is held on one processor (placement.py:495-496), reshard slices it at that holder
    (placement.py:197-198, `holder or processor`), and those slices are not invariant because their input is not
    (placement.py:627-629). They are therefore not copied onto the readers (633-646), and insert_send_recv writes a
    SEND at the holder and a RECV at each other processor (placement.py:660-680). The transfer's source is bytes of
    the state tensor.
  - 8, step 4: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "Each distinct DynamicSlice a node's program
    reads of an attribute, or of a value made from an attribute by tensor views alone, is its own never-freed value at
    that node (7.1, step 5)." (design/problems-of-14b9a757-change-request.md:425-426):
    compiler/mesh_compiler/schedule.py:91-98 and 174: the view chain stops only at a node placement marked invariant.
    A slice of a value made from an attribute by a view operator outside placement's VIEWS (placement.py:34-35,
    627-629; `aten.split_with_sizes` with getitem, `aten.diagonal`) is not marked, so `of` follows the schema's
    alias_info through the slice and the view to the state placeholder. schedule.py:188-189 then writes one
    MESH_INVARIANT value of the whole attribute's extent for (placeholder, processor); the slice has no value of its
    own. Separately, schedule.py:166-167 (with 127-129) writes no value at all where no nonempty transfer crosses
    machines.
  - 11, step 5: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "**The program:** it learns of the
    cancellation through the raised counts (7.3, steps 4 to 6)." (design/problems-of-14b9a757-change-request.md:501):
    The sentence cites 7.3 step 4, which does not hold: rdma/mesh-flow.c:2096-2099 raises the unposted sends' error
    counts before the posted sends' completions are retrieved, and mesh-flow.c:1549 takes a completion on a lane no
    longer served without raising either count. For such a send the range's count plus error count never reaches the
    execution's total, so the wait the program makes (compiler/mesh_compiler/bridge.py:47) returns 0 from
    mesh_host_sent (rdma/mesh-collective.c:60) until mesh-collective.c:76 returns ETIMEDOUT, not ECANCELED.

Covered:

  - 1, Change: the refusal is deleted: covered [2026-11-06, mesh 30afa696]
  - 2, Change: both refusals are deleted: covered [2026-11-06, mesh 30afa696]
  - 4, Change: the refusals are deleted: covered [2026-11-06, mesh 30afa696]
  - 5, step 1: covered [2026-11-12, mesh 3a30b7ec]
  - 5, Change: the refusal is deleted: covered [2026-11-06, mesh 30afa696]
  - 7 (Code): covered [2026-11-06, mesh 6a8a8524]
  - 7.1, step 3: covered [2026-11-08, mesh b880b808]
  - 7.1, step 4: covered [2026-11-08, mesh b880b808]
  - 7.1, step 5: covered [2026-11-12, mesh 28d7a032]
  - 7.2, Change: PLAN_SENT is deleted: covered [2026-11-06, mesh 6a8a8524]
  - 7.3, step 1: covered [2026-11-06, mesh 6a8a8524]
  - 7.3, step 5: covered [2026-11-06, mesh 8624d557]
  - 7.3, step 6: covered [2026-11-12, mesh 0704bf95]
  - 7.3, step 7: covered [2026-11-12, mesh 81c8e9da]
  - 8, step 1: covered [2026-11-11, mesh 297c3cda]
  - 8, Change: LOADED and the execution UINT64_MAX − 1 are deleted: covered [2026-11-06, mesh 2155c2b8]
  - 9: covered [2026-11-06, mesh 5eeee1f9]
  - 10, Change: the representation: covered [2026-11-06, mesh 496ad4ef]
  - 10, Change: the refusal is deleted: covered [2026-11-06, mesh 00f31f7f]
  - 11, step 1: covered [2026-11-06, mesh 028b27d6]
  - 11, step 2: covered [2026-11-06, mesh 028b27d6]
  - 11, step 3: covered [2026-11-06, mesh 028b27d6]
  - 12, step 1: covered [2026-11-06, mesh 61435d4b]
  - 12, step 2: covered [2026-11-06, mesh 61435d4b]
  - 13, step 1: covered [2026-11-03, mesh 53e3935c]
  - 13, step 3: covered [2026-11-06, mesh 4e4b1938]
  - 14, step 1: covered [2026-11-06, mesh 8c8a9dfe]
  - 14, step 2: covered [2026-11-06, mesh 8c8a9dfe]
  - 14, step 4: covered [2026-11-12, mesh 28d7a032]
  - 14, step 5: covered [2026-11-06, mesh 8c8a9dfe]
  - 14, step 6: covered [2026-11-06, mesh b78f81d0]
  - 14, Change: replaces plan_before and mesh_graph_chains for these items: covered [2026-11-06, mesh 8c8a9dfe]

## [unequal-shards-change-request.md](unequal-shards-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 17 items outstanding, 0 not determined, 29 covered, 1 withdrawn.

Outstanding:

  - 2 - Data structure: "MPI's two arrays, `counts` (MPI's `recvcounts` and `sendcounts`) and `displs`, each with one
    entry per device of the mesh dimension." (1) Where a value is resharded along the dimension it is already sharded
    on, counts and displs are each a tuple of one tuple per source device, each inner tuple of one entry per
    destination device, and the node carries the pair (i, j) as its block:
    compiler/mesh_compiler/placement.py:175-179, 202. (2) Where an operand's shards are joined for a product, counts
    and displs have one entry per block the reading processor takes (the blocks `unmasked` picks), not one per device:
    placement.py:540-546.
  - 2 - Stage: "each an integer array of one entry per device." Where a value is resharded along the dimension it is
    already sharded on, counts and displs are tuples of one tuple per source device
    (compiler/mesh_compiler/placement.py:175-179, 202); where an operand's shards are joined for a product they have
    one entry per block the reader takes (placement.py:540-546).
  - 4 - Data structure: "The buffer a device holds for a sharded dimension has one size for all devices, rounded up".
    For a dimension sharded in stripes, uneven() is False (compiler/mesh_compiler/placement.py:78-79) and each device
    holds len(range(j, extent, shards)) elements (placement.py:169-170, 184-186), which differ by device when the
    extent is not a multiple of the device count; nothing pads them.
  - 8 - Algorithm (b): the layout kept forward: "The sequence keeps that layout through every function of the program,
    forward and backward." An operator with no rule (compiler/mesh_compiler/placement.py:477, `RULES.get(node.target,
    replicated)`), a convolution or pooling over a striped value (:478-480) and a reduction over the striped dimension
    that is not a sum (e.g. mean.dim, :380, label None) leave no target (:481-486); the stripes are then sent to one
    processor and interleaved into the original order there (:491-496, :212-213, :137-143).
  - 8 - Algorithm (b): the layout kept backward: "The sequence keeps that layout through every function of the
    program, forward and backward." The backward's functions pass through the same partitioned: one with no rule
    (compiler/mesh_compiler/placement.py:477), an index_put that is not the accumulated form (:349-351), or a
    convolution or pooling over a striped value (:478-480) is given its operands whole on one processor, the stripes
    interleaved into the original order there (:491-496, :212-213).
  - 9 - Data structure / Algorithm: "Each of the four carries the counts of item 2 where its dimension is sharded by
    item 1." (1) A slice that shards a value the reading processor already holds (holder None or the processor itself,
    compiler/mesh_compiler/placement.py:193-198) makes no send or recv; its `blocks` are removed from every node when
    place ends (:651-652), so the slice node carries only its own bounds. (2) For a change of device order or of sizes
    along the held dimension, what is carried is a tuple of one tuple per source device (:175-179, 202), not item 2's
    array of one entry per device.
  - 12 - Algorithm (the all-gather's method): "Which of them an all-gather uses is chosen by their times at the
    sharding ratios, comm(i)(B) being \"determined based on the collective operation type, the sharding ratio B\" and
    the profiled network, and is chosen again whenever the sharding ratios are solved again". reshard has one method:
    every shard is cut to its own size by own and sent in its own channel
    (compiler/mesh_compiler/placement.py:153-156, 181-213); no time and no sharding ratio is read there, and no value
    records a method. comm in compiler/mesh_compiler/shares.py:59 is read only by the solve of the ratios.
  - 10 - Algorithm / Stage: outstanding [2026-11-12, mesh f20efc91; review of f20efc91]. "Halos are exchanged by a
    collective permute at the largest halo over the devices, and each device slices off what exceeds its own.":
    compiler/mesh_compiler/placement.py `halos` caps each send at the sender's own extent (`width = min(extent,
    ...)`), so a sender shorter than the largest halo sends what it holds.
  - 1 - Stage: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "`place` consumes it with the dimension's
    extent and writes the sizes, an integer array of one entry per device, which are the `counts` of items 2 and 3."
    (design/unequal-shards-change-request.md:97-99): Cites item 2, whose Data structure and Stage the same index lists
    outstanding. In code the counts on the nodes are not those sizes in two places:
    compiler/mesh_compiler/placement.py:175-179, 202 (resharding along the held dimension: counts is a tuple of one
    tuple per source device, the overlaps of old and new ranges, block is the pair (i, j)); placement.py:540-546
    (shards joined for a product: counts is the extents of the blocks the reading processor takes). The sizes are
    written at placement.py:614; they are the counts only on the path at :173, 209.
  - 2 - Algorithm: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "An all-gather, a reduce-scatter, an
    all-to-all, a gather and a scatter over a dimension sharded by item 1 each carry the two arrays. The block of
    device j is sent at `counts[j]` elements and placed at `displs[j]`."
    (design/unequal-shards-change-request.md:126-128): compiler/mesh_compiler/placement.py:175-179, 201-202: for an
    exchange along the dimension the value is already sharded on (other sizes or other processors), `given` and
    `displs` are tables of one tuple per source device, and the block is (i, j); the piece is cut at sum(given[i][:j])
    with given[i][j] elements (:188-190), so counts[j] is a tuple, not a number of elements. placement.py:540-546: for
    shards joined at a product, counts and displs have one entry per block the reader takes (index k), not per device
    j. This is the fact the index itself records as the failure of item 2's Data structure.
  - 3 - Data structure / Algorithm: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "the `counts` and
    `displs` of item 2." and "every device of the collective is given the same arrays."
    (design/unequal-shards-change-request.md:143-145): Cites item 2's arrays, which the same index lists outstanding
    (Data structure, Stage). In code: compiler/mesh_compiler/placement.py:528, 540-546 forms the arrays inside the
    loop over reading processors from the blocks `unmasked` picks for that reader (:417-435), so with a causal mask
    over three contiguous shards the sends of one key block from p0 carry (e0, e1) for p1 and (e0, e1, e2) for p2;
    devices of the same gather of keys hold different arrays. placement.py:175-179: on the held-dimension path what is
    given is a table per source, not item 2's array.
  - 3 - Stage: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "`place` writes the arrays once for a
    collective; that collective's nodes on every device carry them." (design/unequal-shards-change-request.md:147):
    compiler/mesh_compiler/placement.py:528-546: for operands joined at a product the arrays are written once per
    reading processor (inside `for j, processor in enumerate(processors)`), each from that reader's own picked blocks
    (`extents = tuple(shape(piece)[v[0]] for piece in pieces)`, :543), and differ between readers when the mask
    removes blocks; the send nodes on one holder then carry different arrays for different readers (:683-685 copies
    blocks[processor]). Also placement.py:203-207: for a replication to several processors `joined` and `placed` are
    formed again for each reader inside the loop at :181 (equal values there).
  - 8 - Algorithm (e): a block the mask inhibits entirely: outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    "A block of queries and keys that the mask inhibits entirely is not computed."
    (design/unequal-shards-change-request.md:299-300): compiler/mesh_compiler/placement.py:423-429: `unmasked` returns
    `every` (all key blocks for every processor's queries) unless the product has exactly one user, that user is
    aten.add.Tensor, its other operand reaches no placeholder (:424-425), and every node from there to the next bmm is
    in MASKED (:31-32). A mask that is a buffer or lifted constant of the module (a placeholder in the exported
    graph), or one applied by where/masked_fill on the scores, or with a finite fill, is not read (the test is `==
    -math.inf`, :435), and :540-547 then joins and computes every block, the wholly inhibited ones included.
  - 12 - Data structure: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "A device's buffer of item 4 is the
    full array and its shard of item 1 is a subarray of it: the buffer's size along the sharded dimension is at least
    the number of elements the device owns, as a local array's leading dimension is at least LOCr."
    (design/unequal-shards-change-request.md:374-376): Cites item 4's buffer, whose Data structure the same index
    lists outstanding: compiler/mesh_compiler/placement.py:78-79, 169-170, 184-186 (a striped dimension has no padded
    buffer). On the send, `sizes` is the shape of whatever the sent value was sliced from on the sender (:669-671),
    which is item 4's buffer only when the sent value is own's slice of it (:86-90); a stepped slice ends the chain
    (:669), so a stripe is sent with sizes equal to subsizes and starts 0. For shards of unequal sizes the buffer is
    max(sizes) (:193-194, 93-98) and that part held.
  - 12 - Algorithm (subarray sent): outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "a collective of item 2
    is given, for each device, the sizes of the full array, the sizes of the subarray and where it starts, and sends
    the subarray: `counts[j]` is the subarray's number of elements."
    (design/unequal-shards-change-request.md:377-378): Cites item 2, outstanding in the same index.
    compiler/mesh_compiler/placement.py:175-179, 188-190, 202: on the exchange along the held dimension counts[j] is a
    tuple (one entry per destination), and the subarray sent has given[i][j] elements along the dimension;
    placement.py:540-546: counts is indexed by the position k of the block among those the reader takes, not by device
    j. The three arrays are written at :669-673, 681-682, and no padded value crosses processors that I found
    (:153-156, 195-198, 525).
  - 12 - Stage: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "`place` adds the collective's `send` and
    `recv` nodes (item 2); for the subarray these carry the three arrays, each an array of integers with one entry per
    dimension of the buffer." (design/unequal-shards-change-request.md:386-387): Refuted only by its citation: item
    2's Stage ("`send` and `recv` nodes `place` adds ... with `counts` and `displs`, each an integer array of one
    entry per device") is outstanding in the same index, compiler/mesh_compiler/placement.py:175-179, 202, 540-546.
    The rest of the sentence I found in code: sizes, subsizes and starts, one entry per dimension, on both ends of
    every channel, placement.py:669-673, 681-682.
  - 15 - Stage: outstanding [2026-11-12, mesh 6d379b7e; a second reader]. "`place` writes these sizes where the
    placement operand carries no tuple for a sharded dimension; they are the `counts` of items 2 and 3."
    (design/unequal-shards-change-request.md:490-491): Cites items 2 and 3; item 2's Data structure and Stage are
    outstanding in the same index. In code the sizes written at compiler/mesh_compiler/placement.py:64-66, 614 are the
    counts on the nodes only on the path at :173, 209; at :175-179, 202 the counts are a table of overlaps per source
    device, and at :540-546 the extents of the blocks a reader takes.

Covered:

  - 1 - Data structure: covered [2026-11-11, mesh 297c3cda]
  - 1 - Algorithm: covered [2026-11-11, mesh 297c3cda]
  - 4 - Algorithm / Stage: covered [2026-11-12, mesh ed14734e]
  - 5 - Data structure: covered [2026-11-11, mesh 0d70ec49]
  - 5 - Algorithm: covered [2026-11-11, mesh 297c3cda]
  - 5 - Stage: covered [2026-11-11, mesh 297c3cda]
  - 6 - Data structure: covered [2026-11-12, mesh 60dc2097]
  - 6 - Algorithm: covered [2026-11-12, mesh 60dc2097]
  - 6 - Stage: covered [2026-11-12, mesh 60dc2097]
  - 7 - Data structure: covered [2026-11-12, mesh 60dc2097]
  - 7 - Algorithm: covered [2026-11-12, mesh 60dc2097]
  - 7 - Stage: covered [2026-11-12, mesh 60dc2097]
  - 8 - Data structure: covered [2026-11-11, mesh d3eecd18]
  - 8 - Algorithm (a): tokens permuted before the first embedding: covered [2026-11-12, mesh f8de5bbc]
  - 8 - Algorithm (c): position ids and target ids: covered [2026-11-12, mesh f8de5bbc]
  - 8 - Algorithm (d): the mask in original order: covered [2026-11-12, mesh 90843d93]
  - 8 - Stage: covered [2026-11-12, mesh 90843d93]
  - 9 - Stage: covered [2026-11-12, mesh 991742c8]
  - 10 - Data structure: covered [2026-11-12, mesh 3a30b7ec]
  - 11 - Data structure: covered [2026-11-12, mesh 53779b95]
  - 11 - Stage: covered [2026-11-12, mesh 53779b95]
  - 13 - Data structure: covered [2026-11-12, mesh 53779b95]
  - 13 - Algorithm: covered [2026-11-12, mesh 53779b95]
  - 13 - Stage: covered [2026-11-12, mesh 53779b95]
  - 14 - Data structure: covered [2026-11-10, mesh 4a9e1cf4]
  - 14 - Algorithm (the bound in the linear program): covered [2026-11-10, mesh 4a9e1cf4]
  - 14 - Algorithm (Whale's Algorithm 1): covered [2026-11-12, mesh 991742c8]
  - 14 - Stage: covered [2026-11-12, mesh 991742c8]
  - 15 - Algorithm: covered [2026-11-11, mesh 297c3cda]

Withdrawn:

  - Dropped - heterogeneity.md R11: withdrawn [2026-11-12 reading]. design/unequal-shards-change-request.md:495-498:
    "Dropped as not needed to implement the requirements above (the operator: \"if any of them are optional, drop
    them\"): `design/heterogeneity.md` R11, changing the tuples while a program runs".

## [integration-finishing-change-request.md](integration-finishing-change-request.md)

Outstanding [2026-11-12, mesh 434b9955, metal-microbench 0b692ed8]: 13 items outstanding, 3 not determined, 5 covered, 0 withdrawn.

Outstanding:

  - 1 (Write, bullet 4: e2b on all five machines to its end): "Run e2b on all five machines to its end: every rank's
    prefill and decode end, and the run's record states each rank's decoded tokens." The only program that decodes E2B
    is metal-microbench tools/e2b_replay.m (its mesh_plan call at tools/e2b_replay.m:1322), which run_pair.py launches
    (tools/mesh/run_pair.py:280-291); no caller of compile_module decodes E2B (the callers are mesh
    examples/tp.py:138,197, examples/engines.py:111, compiler/samples.py:327,827,998,1023).
  - 3 (Produced by): "**Produced by** the caller of the compilation." No caller of compile_module passes `cost` (mesh
    examples/tp.py:138,197; examples/engines.py:111; compiler/samples.py:327,827,998,1023); every weight is the
    schedule module's default unit_cost, 1.0 a task and an edge (compiler/mesh_compiler/schedule.py:26-27,
    driver.py:14).
  - 4 (Write): "**Write:** that walk is a function of mesh rdma/mesh-graph.{h,c}, with caller operands."
    rdma/mesh-graph.h and mesh-graph.c hold no such function at HEAD: mesh_graph_worklist_source, added by mesh
    e34e7a5e (2026-11-08), was removed by mesh dd2062d2 (2026-11-09); `worklist` matches nothing under rdma/ or
    compiler/.
  - 7 (The data structure: that commit's record): "and that commit's record: struct Feedback's semaphore, error and
    GPU times, and struct MetalRun's `failure`." The compile chain's record is MeshMetal4Record
    (rdma/mesh-metal4.m:25-30): an MTLSharedEvent and its value, `error`, `start`, `end`. It has no semaphore; there
    is no struct Feedback or struct MetalRun in mesh; the error reaches the caller as mesh_metal4_wait's message
    (mesh-metal4.m:208-216; rdma/mesh_metal4.py:44-47). The record is one a segment, written again by each execution's
    commit of that segment. struct Feedback in metal-microbench tools/e2b_replay.m:152 is two atomic times a phase.
  - 7 (Consumed by): "**Consumed by** `feedback_wait`, which receives one handler's call a commit." No function
    `feedback_wait` is in either repository; the waiter in the compile chain is mesh_metal4_wait
    (rdma/mesh-metal4.m:208-217), which waits for a record's latest committed value (`record.value`), with
    `timeoutMS:UINT64_MAX`.
  - 7 (Replaces, e2b_replay.m feedback_start): "in metal-microbench tools/e2b_replay.m, `feedback_start`: one
    `MTL4CommitOptions` a phase (`PREPARE`, `PREFILL`, `DECODE`, `PROFILE`), which every commit of that phase
    carries;" `feedback_start` is still in tools/e2b_replay.m:155-157 and called at :1484; it initialises one struct
    Feedback a phase (PREFILL, DECODE; :152-154), which feedback_record (:160-168) writes for every commit of that
    phase.
  - 2 (The algorithm): outstanding [2026-11-12, mesh f20efc91; review of f20efc91]. "**The algorithm** is list
    scheduling in the order of the linear program's completion times": compiler/mesh_compiler/schedule.py `schedule`
    keys every function by `mesh_graph.longest` and takes the linear program's time only for the recv of a carried
    channel.
  - 2 (It takes): outstanding [2026-11-12, mesh f20efc91; review of f20efc91]. "It **takes** a key for each of a
    node's replayed commands, and the commands' dependences": `schedule` takes a key for every function of every
    machine and calls one `heap_order` over them, not one node's commands.
  - 2 (It writes): outstanding [2026-11-12, mesh f20efc91; review of f20efc91]. "the node's list of commands:
    `mesh_graph_heap_order` over those keys": one order over every machine's functions is sliced per processor.
  - 2 (Consumed by): outstanding [2026-11-12, mesh f20efc91; review of f20efc91]. "**Consumed by** the node's replay,
    which executes its commands in that list's order": the list of the entries above is not made.
  - 3 (Replaces): outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    design/integration-finishing-change-request.md:55: "**Replaces:** the weighting operand of one `mesh.Weight` per
    transfer.": Not gone from both repositories. metal-microbench tools/e2b_replay.m:625-627 declares the weighting
    operand E2BWeighting, which writes one struct mesh_weight per declared transfer (:647-649
    `for(i<count)weights[i]=(struct mesh_weight){.produced=1,.consumed=1,.read=1,.page=1}`; passed to mesh_plan at
    :779). mesh rdma/mesh.py:299-309 `function(declared, weights, ...)` still takes one mesh.Weight per transfer
    (`Weight(*weight) for weight in weights`), filled per transfer at compiler/mesh_compiler/schedule.py:153. The
    earlier reading checked only the name `mesh.Weight` in metal-microbench.
  - 4 (Replaces): outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    design/integration-finishing-change-request.md:59: "**Replaces:** the walk in metal-microbench metal_recording.m
    `MetalMaskedPass`.": Only the name is gone. The walk (the masked first pass's vote and deferral, the later pass's
    worklist-listed threadgroup test) is still in metal-microbench: metal_transport.mm:395 `Acquire` (header comment
    :1-5 cites the same worklist sources; flags and list views at :139-153, compaction and later twin at :279-310),
    called from metal_lower.mm:2181. It is not a function of mesh rdma/mesh-graph.{h,c}: `worklist` matches nothing
    under mesh rdma/ or compiler/ (mesh_graph_worklist_source was removed by mesh dd2062d2).
  - 7 (Produced by): outstanding [2026-11-12, mesh 6d379b7e; a second reader].
    design/integration-finishing-change-request.md:77-78: "**Produced by** every commit of the replay: the handler
    writes the feedback data Metal calls it with to its own commit's record.": The record is a segment's, not a
    commit's: compiler/mesh_compiler/lower_metal.py:1154 makes one mesh_metal4_record per segment when the program is
    bound, and :1207-1208 passes that same segment.record[0] to every execution's commit of the segment, so
    rdma/mesh-metal4.m:199-203 overwrites error, start and end of one shared record (the index lists "that commit's
    record" outstanding for this). Not every commit either: rdma/mesh-metal4.m:136-140 (mesh_metal4_after, reached
    from lower_metal.py:1197) issues two torch MPS commits with no feedback handler and no record.

Not determined:

  - 1 (Write, bullet 1: diagnose and correct each machine's bridge configuration): not determined [2026-11-12]. The
    configuration the repository installs is in files (mesh bootstrap.sh, bin/mesh-bridge.sh, etc/bridge.conf).
  - 1 (Write, bullet 2: the mini's configuration.plist): not determined [2026-11-12].
  - 1 (Write, bullet 3: start every bridge): not determined [2026-11-12]. mesh bin/mesh-bridge.sh exists at HEAD.

Covered:

  - 3 (The data structure): covered [2026-11-12, mesh 81c8e9da]
  - 3 (Consumed by): covered [2026-11-12, mesh 81c8e9da]
  - 5: covered [2026-11-08, metal-microbench 5d5efc09]
  - 7 (The data structure: one MTL4CommitOptions a commit): covered [2026-11-12, mesh a86c7dc4]
  - 7 (Replaces, metal_recording.m MetalRunOf): covered [2026-11-08, metal-microbench 33be7703]
