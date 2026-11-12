# Change requests: which are outstanding

Every line below is a reading of the code at mesh 9aa74ed1 and metal-microbench 0b692ed8, made on 2026-11-12, one
reader a change request, each request read whole. The readings, with the clause each quotes and the lines of code
each rests on, are in metal-microbench `output_data/change-request-audit-20261112/audit.json`. The standard every
change request follows is [change-requests.md](change-requests.md).

- The operator, 2026-11-12: "remove all undated or unversioned 'claims to facts' from any documents in any 'index' as
  there can clearly never be a coherent store of 'what the repo is' without versioning, dating, or namespacing of
  factual claims." This index's earlier text is at mesh 9aa74ed1.
- The operator, 2026-11-12: "if they weren't finished, they still need to be finished beacuse they are not optional
  'would be nice to haves'. if any are finished, mark them as 'covered [datestamp,commitstamp]'."
- `covered [date, commit]`: the item's clause was read as holding in the code at mesh 9aa74ed1 or metal-microbench 0b692ed8; the date and
  commit, with the commit's repository, are those of the commit the reader found the code landed in, or of the reading
  where it named none.
- `outstanding`: the clause was not found to hold in code that has the data flow of [data-flow.md](data-flow.md). An
  item whose one implementation is in the files [the-compiler.md](the-compiler.md) order 5 names is outstanding.
- `not determined [date]`: the clause states the condition of machines or services, which the reading did not contact; the item
  is outstanding until a reading of the machines shows it holds.
- No request has every item covered. All eleven are outstanding.
- [requirements-by-stage.md](requirements-by-stage.md) lists the same items under the stage of
  [data-flow.md](data-flow.md) that takes each, with each clause quoted.

## [compilation-composition-change-request.md](compilation-composition-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 70 items outstanding, 22 covered, 1 withdrawn.

Outstanding:

  - 1 / Write: every rank, its collectives, each placed CPU or GPU function: A lowering of the compile_module chain (a
    `lower(scheduled)` in compiler/mesh_compiler/, applied by driver.lowered) converts each processor's placed CPU
    function; operand the scheduled value, result one function per CPU processor, called by Program.
  - 1 / Write: the whole assigned domain: lower_metal.lower (and lower_coreml.lower for `ane` processors) convert the
    remaining operators of the static-capture domain that export.read admits; operand the scheduled value's nodes,
    result the processor's function.
  - 1 / Replaces: rdma/mesh_compile.py program() is deleted in the commit of the-compiler.md order 5; nothing further
    is written in the chain for this clause.
  - 2 / data structure: what each edge carries: The schedule stage (compiler/mesh_compiler/schedule.py, with the
    functions of rdma/mesh_graph.py) writes each value's consumer kind, accesses and lifetimes on the scheduled value;
    lowering, the runtime's bridge and the archive read them from it.
  - 2 / the signature: lower (compiler/mesh_compiler/lower_metal.py, lower_coreml.py) returns, with each processor's
    function, the synchronizations it emitted as the operands of mesh_graph.signature (processors, tasks, dependences,
    operations); the signature is printed by rdma/mesh_graph.py's printers from that result.
  - 2 / Write: each row's work completed with its deletion: Each of the nine requests' clauses is carried by the stage
    the table's third column names (export, placement, schedule, lowering, runtime, archive); the old files are
    deleted in the commit of the-compiler.md order 5.
  - 3 / Write, second bullet: the four named deletions: The old files are deleted in the commit of the-compiler.md
    order 5. Nothing is written in schedule.py for this clause.
  - 4 / Write: compilation realizes storage: The schedule stage produces each value's storage (the pages of each
    carried channel among them) as fields of the scheduled value; bridge.entries, the lowerings and archive.save read
    them.
  - 4 / Write: neither execution nor the bridge computes its own order or placement: The schedule stage writes, in the
    scheduled value, each carried channel's page placement and the order of its page transfers; bridge.entries takes
    them as operands and publishes them.
  - 4 / Write: signature and external observation cover this execution: Same producer as item 2 / the signature;
    bin/mesh-sync-monitor.py takes the record printed from the lowered program.
  - 4 / Write: performance and numerical comparison: A program compiled by compile_module and run by Program decodes
    gemma-4-e2b and is compared as docs/measurement.md states.
  - 5 / data structure and Composes: one compiled native program per node: lower takes the scheduled value and emits,
    per machine, one compiled native program that launches the CPU, ANE and MPS work of that machine's processors;
    Program calls it.
  - 5 / Composes: signature printed by the graph module's printers: Same as item 2 / the signature.
  - 5 / Replaces, site 1: rdma/mesh_compile.py is deleted in the commit of the-compiler.md order 5.
  - 5 / Replaces, site 2: The model e2b_replay.m runs is compiled by compile_module and run by Program;
    tools/e2b_replay.m and the recorder it links are deleted in the commit of the-compiler.md order 5.
  - 5 / Replaces, site 3: signature schema and printers: Same as item 2 / the signature: the chain's producer calls
    mesh_graph.signature and its printers.
  - 5 / Write: one definition of each repeated behavior: Each behavior listed has one definition in the module that
    produces it (ir.py for the shared names and types, the lowering modules' common part for the node loop), its
    inputs explicit operands, and every caller calls it; the old duplicates go with the-compiler.md order 5.
  - 5 / Write: emitted program preserves domain, collectives, page placement, lifetimes: Follows from item 1 / whole
    domain, item 2 / edges and item 4 / storage; no further work is stated by this clause.
  - 5 / Write: producer and consumer agree at both ends: The native program's result bindings and the signature's
    identities are produced by lower and read by Program and the monitor from the same value.
  - 6 / Replaces 1: In lowering (compiler/mesh_compiler/lower_metal.py, Lowering.flush), the dependence edges are the
    result of mesh-graph's dependences function over the lowered work's accesses, and that edge result is the operand
    of mesh-graph's synchronization reduction; the caller is lower.
  - 6 / Replaces 2: signature printed from the resulting compilation: Same as item 2 / the signature.
  - 6 / Replaces 3: In lower_metal, the scheduled graph's edges are remapped onto the dispatches each node becomes,
    the newly lowered vertices and edges are added to that graph, and it is the operand of ordering (barriers, fences)
    and emission.
  - 6 / Replaces 4: lower_metal passes each dispatch's views (storage key, offset, strides, write) to mesh-graph's
    views dependence function and its ordering consumer takes the returned relation; a dispatch lowering adds
    contributes its own views.
  - 6 / Write: connections finished; identity mapping retained: lower_metal keeps, for each dispatch, segment and wait
    it emits, the identities of the scheduled nodes it came from, and returns that mapping with the function; its
    consumers are the signature and the ordering of Replaces 3.
  - 6 / Write: what the emitted program preserves: The LiteRT model enters the chain at export (see Correction /
    Composes); lower_metal's emitted function keeps independent dispatches unordered and waits as the dependence
    relation of Replaces 1, 3 and 4 gives.
  - Correction: generated target source / Composes: both frontends at export: An export function of
    compiler/mesh_compiler (beside export.export) takes a LiteRT model and returns ir.Exported: the operations with
    their types, shapes and annotations; place, schedule and lower take it as they take torch's.
  - Correction: generated target source / Composes and Emission boundary: In lower_metal.Lowering.multiply and
    dispatch, the three decisions at :254, :258 and :302 are taken from the operation's structured operands (the
    Product's form, the node's target, the dispatch's kind), not from the emitted text.
  - Correction: generated target source / Replaces: These transformations are functions from operations to operations
    inside lower (compiler/mesh_compiler/lower_metal.py), taking the scheduled value's typed nodes.
  - Correction: generated target source / Write: frontends supply facts; lowerings preserve identities: export writes
    the alias, effect and transfer/generation facts on the exported nodes; lower_metal reads them and updates
    identities, accesses and dependences where it fuses or rebinds.
  - Correction: generated target source / Property preserved: Follows from the Replaces entry above; no further work
    is stated.
  - Performance correction / Emission: lower builds, from the scheduled value's nodes, an IR module of typed
    instructions; the transformations below are functions on it; the MSL is printed after all of them.
  - Performance correction / Fusion and storage: Inside lower: fusion composes the operations' indexing maps and
    remaps their operands and results; storage of each intermediate is decided from use-def, alias and equivalence
    relations before the rewriting that consumes the decision.
  - Performance correction / Reductions: lower_metal lowers reduction operations to subgroup reductions and workgroup
    exchanges, and packed data to packed arithmetic, from the typed operation, its layout and its reduction axes.
  - Performance correction / Replaces: rdma/mesh_compile.py is deleted in the commit of the-compiler.md order 5; the
    operators it lowers are lowered by lower_metal (item 1 / whole domain).
  - Performance correction / Write: numerical operations, regions, values, effects: In lower, a map or epilogue is an
    operation region with typed operands and results (the IR module of the Emission entry); values are retained across
    their uses.
  - Performance correction / Write: publication and acquisition: lower takes, from the scheduled value, the range/page
    dependences the transport's schedule uses and emits the producer's publication and the consumer's acquisition from
    them.
  - Performance correction / Write: applicability from structure, not spelling: Same three sites as the Emission
    boundary entry.
  - Performance correction / Performance obligation: lower performs promotion, subgroup lowering and publication
    fusion as separate functions on operations, besides fusion.
  - Complete replacement / all in one commit workgroup: the-compiler.md order 5: the old files are deleted in the
    commits in which code compile_module and Program produce has their feature coverage at the same performance or
    better.
  - Complete replacement / production E2B link: The program that decodes E2B is compiled by compile_module and run by
    Program, linked to the source-built libraries.
  - Complete replacement / operand table row 1: Numerical operations: The LiteRT export of the Correction / Composes
    entry produces the same ir.Exported with quantization and precision on its values; lower's specialization and
    fusion take it.
  - Complete replacement / operand table row 2: Kernel instructions: Same as Performance correction / Emission.
  - Complete replacement / operand table row 3: Values and effects: Same as item 2 / edges and item 4 / storage: the
    facts are on the scheduled value and their consumers read them there.
  - Complete replacement / operand table row 4: Program relations: Same as item 6 / Replaces 3 and item 5 / native
    program.
  - Complete replacement / operand table row 5: Executable artifact: Same as item 5 / native program and item 2 / the
    signature.
  - Complete replacement / operands kept until finalization; publication before execution: The LiteRT export returns
    the whole program and its bodies as ir.Exported before any execution; lower runs after place and schedule have
    produced the calls and transfers.
  - Complete replacement / many-to-many identities; producer-supplied meaning: Same as item 6 / Write: identity
    mapping (the mapping may relate several nodes to several dispatches) and operand table row 1 (quantization on the
    exported value).
  - Complete replacement / authored sources; emitted source never inspected; tuned projection/argmax: The three sites
    of the Emission boundary entry; projection and argmax are lowered by lower_metal from typed parameters and
    regions.
  - Complete replacement / Tint objects behind handles; cache hits: Same as Performance correction / Emission; the
    cached artifact is keyed by its compilation inputs.
  - Complete replacement / replaced-implementation table row 1: lower specializes quantized operations by constant
    propagation before emission (with operand table row 1's quantization on the value).
  - Complete replacement / replaced-implementation table row 2: Same as Performance correction / Fusion and storage.
  - Complete replacement / replaced-implementation table row 3: Same as Performance correction / Reductions.
  - Complete replacement / replaced-implementation table row 4: Same as Performance correction / Write: publication
    and acquisition.
  - Complete replacement / replaced-implementation table row 5: Same as item 6 / Replaces 4 and item 2 / the
    signature.
  - Complete replacement / replaced-implementation table row 6: Same as item 5 / native program; metal_recording.* is
    deleted in the commit of the-compiler.md order 5.
  - Complete replacement / after the replacement: Follows from the all-in-one-workgroup entry and the Emission
    boundary entry.
  - Source bindings / Compact effects: producer and consumers: Same as item 6 / Replaces 4: lower_metal produces the
    views; rdma/mesh_graph.py exposes the header's views dependence function; the ordering, fusion and lifetime
    consumers take its relation.
  - Source bindings / Software binary64: lower_metal.lower takes the scheduled value's float64 operations and emits
    target instructions implementing them by the cited SoftFloat and FDLIBM algorithms.
  - 7 / Composes: lowering emits the native program, the runtime calls it: Same as item 5 / native program.
  - 7 / Replaces: run-time readers of positions, tasks, guards; static discovery: Same as item 7 / Write below.
  - 7 / Write: lower takes the scheduled value (the schedule and the effects) and emits fixed target/API operations
    with bound operands in the scheduled order, the CPU bodies composed with them while still lowerable; the old
    executor bodies are deleted with it.
  - 7 / table row 1: typed numerical operations: Same as item 7 / Write.
  - 7 / table row 2: partitioned computation, transfers, compiled order: Same as item 7 / Write, for
    lower_coreml.lower's function as well.
  - 7 / table row 3: realized storage: Same as item 4 / storage and item 7 / Write: storage is realized by compilation
    and the emitted program holds bound addresses.
  - 7 / table row 4: target commands: Same as item 7 / Write.
  - 7 / table row 5: lowered synchronization and its cause: lower_metal records, for each barrier, fence and wait it
    emits, the dependence that causes it, and the signature is printed from that record.
  - 7 / contract paragraph: Same as the Emission boundary entry, item 6 / Replaces 4 and item 6 / Write: identity
    mapping.
  - 7 / residual runtime operands and waits: lower_metal's wait before a send follows from the segments that write
    what the send reads (the dependence relation of item 6); the bridge's waits follow from the pages each publication
    reuses.
  - 7 / performance paragraph: Same as item 7 / Write and the preceding entry.
  - 4 / Write: the bridge binds physical transport as R7 specifies: outstanding [2026-11-12, mesh a7e2de47]. Read at that commit, the clause did not hold in the
    `compile_module` chain: `compiler/mesh_compiler/bridge.py` wrote the records itself (review of a7e2de47). Mesh
    81c8e9da has the schedule stage call `mesh_plan`; the clause is not yet read against it.

Covered:

  - The data flow (stage table and its closing sentence): covered [2026-11-10, mesh ce2e8ac9]
  - 1 / data structure and Composes (export): covered [2026-11-10, mesh ce2e8ac9]
  - 1 / algorithm (full conversion): covered [2026-11-12, mesh a86c7dc4]
  - 1 / Write: conversion completed before the program runs: covered [2026-11-12, mesh a86c7dc4]
  - 2 / data structure: send and recv nodes in the one graph: covered [2026-11-10, mesh ce2e8ac9]
  - 2 / Composes: one schedule orders functions and transfers together: covered [2026-11-11, mesh 29aed586]
  - 2 / Write: transformations keep information; outputs are operands; changes reflected: covered [2026-11-10,
    mesh ce2e8ac9]
  - 3 / Write, first bullet: covered [2026-11-11, mesh 29aed586]
  - 3 / Write, second bullet: callers call them, own copies deleted: covered [2026-11-11, mesh 29aed586]
  - 4 / Write: placement and costs are operands: covered [2026-11-10, mesh ce2e8ac9]
  - 4 / Write: compilation realizes transfers, ordering and synchronization: covered [2026-11-10, mesh ce2e8ac9]
  - 4 / Write: runtime consumes programs and remaining inputs; self-timed calls: covered [2026-11-10, mesh ce2e8ac9]
  - 5 / Replaces, site 3: strided-view traversal: covered [2026-11-08, mesh d44e3d0c]
  - 6 / Replaces 2: schedule results are operands of native preparation and of the transport: covered [2026-11-12,
    mesh a86c7dc4]
  - Correction: generated target source / Write: an edge carries its cause: covered [2026-11-10, mesh ce2e8ac9]
  - Performance correction / Constants: covered [2026-11-12, mesh a86c7dc4]
  - Complete replacement / build from source: covered [2026-11-09, metal-microbench 6eca7d7a]
  - Source bindings / Compact effects: the mesh-graph function: covered [2026-11-09, mesh 83c1e78e]
  - 7 / data structure: the scheduled value: covered [2026-11-10, mesh ce2e8ac9]
  - 7 / Composes: the archive stores the scheduled value: covered [2026-11-10, mesh ce2e8ac9]
  - 7 / Replaces: MetalLoopTest and its whole-queue waits: covered [2026-11-09, metal-microbench 07da9e91]
  - Withdrawn, 2026-11-09: the stated scope: covered [2026-11-10, mesh ce2e8ac9]

Withdrawn:

  - Withdrawn, 2026-11-09: clauses on control flow that depends on data ("Nested communication regions: producer and
    consumer handoff", loop stack, predicate drains): withdrawn [2026-11-12 reading].
    /Users/mdot/dox/mesh/design/compilation-composition-change-request.md:542-556 withdraws them; the struck section
    no longer appears in the document. /Users/mdot/dox/mesh/design/implementation-order-2026-11-09.md:77-80 names what
    was struck: "`MetalLoopTest`, `MetalDrain`, trip-count observation, and the scoped regions of
    compilation-composition-change-request.md \"Nested communication regions: producer and consumer handoff\", struck
    on 2026-11-09".

## [aot-compilation-change-request.md](aot-compilation-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 34 items outstanding, 77 covered, 1 withdrawn.

Outstanding:

  - A1 every function of a program: The compile chain (compiler/mesh_compiler/driver.py, with export.py) takes a torch
    program whose block calls several functions, captures each function with export before the program runs, and gives
    the calls between them to A10's graph. Operands: the program and its arguments; result: one scheduled value per
    function. Caller: the program's launcher at every rank.
  - A1 a program run through another framework's delegate: The export stage (compiler/mesh_compiler/export.py)
    returns, for a delegate's program, the graph of the delegate's captured commands as the value place takes.
    Operand: the delegate's program; result: its dataflow graph; caller: compile_module.
  - A3 produced by, consumed by (the caller passes the operand; mesh's type): The schedule stage of the compile chain
    passes the caller's cost operand to mesh_plan as struct mesh_weight of each struct mesh_declared
    (rdma/mesh-collective.h:41,53). Operand: cost; result: the weights field of struct mesh_function; caller:
    compile_module.
  - A4 stage: The schedule stage of the compile chain calls mesh_plan with the collectives place inserted, as struct
    mesh_function with its struct mesh_declared, the map and A3's operand; mesh_plan returns the published static
    schedule that runtime.Program's channels use. This takes the place of bridge.py entries() and publish().
  - A4 algorithm: AllReduce: placement.place inserts the AllReduce a sharded reduction requires as nodes of A8's IR;
    mesh_plan (rdma/mesh-collective.c) receives it as a reduce to a root over its packed trees and a broadcast from
    that root over the same trees reversed. Operands: the reduced value's shards and the map; result: struct
    mesh_declared demands.
  - A4 algorithm: AllGather: placement.place's AllGather reaches mesh_plan as one broadcast demand from each source
    (struct mesh_declared with that source and each destination). Operands: the shards; result: the declared
    transfers; caller: the schedule stage.
  - A4 algorithm: ReduceScatter: placement.place inserts ReduceScatter where a reduction's result is sharded;
    mesh_plan receives, per slice, a reduce to the member that keeps it over its packed trees.
  - A4 algorithm: CollectivePermute: placement.place inserts CollectivePermute as source-destination pairs; mesh_plan
    receives one point-to-point demand a pair and routes them jointly.
  - A4 algorithm: AllToAll: The pairs' demands of an AllToAll reach mesh_plan as struct mesh_declared, one per ordered
    pair of ranks, and are routed in its one mesh_graph_flow call.
  - A4 where the partitioning inserts ReduceScatter and CollectivePermute: placement.py's partitioning has a rule for
    every operator; a window-based operator's rule exchanges halo data by CollectivePermute, and a reduction whose
    result is sharded is communicated by ReduceScatter. Operands: the operator's node and its operands' shardings;
    result: the sharded nodes and the collectives.
  - A4 a node to which the placement operand maps no device (the program's part): runtime.Program on a node that holds
    no processor starts each call's execution of the function's static schedule (mesh_host_started /
    mesh_transfers_start) and returns no value. Operand: the scheduled value; caller: the program at that node.
  - A5 produced by, consumed by: The schedule stage gives mesh_plan each value of the scheduled graph as struct
    mesh_value (rdma/mesh-collective.h:25) with its lifetime; mesh_plan's allocator returns the bindings, which the
    scheduled value carries to the runtime and to the lowered functions.
  - A6 one schedule per function; the schedule takes A8's IR: The schedule stage (compiler/mesh_compiler/schedule.py
    with mesh_plan) produces one schedule of a function: the operations' order and the page instructions' order come
    from the one task graph of the placed function with the cost operand. Result: each processor's operations in order
    and the struct mesh_hop lists.
  - A7 sender(m) (the launcher passes the program's nodes): The runtime of the compile chain takes the program's nodes
    from its launcher beside the group and passes them as mesh_plan's `program` operand at every node of the
    collective.
  - A8 carried as one IR from partitioning to serialization: The value that passes from place to the schedule stage
    carries each consumer's kind, the placement and weighting operands and A10's output, and the schedule stage hands
    it to mesh_plan as struct mesh_function. Operand: the placed graph; result: struct mesh_function with its
    declared, weights and dependences.
  - A9 here, per node: the GPU's program: For a delegate's program: the delegate's recorded command buffers, bound to
    the placed pages, are the processor's function that runtime.Program calls for the scheduled value. For torch
    functions no work is stated here: the clause and the lowering request's item 1 disagree and the document does not
    say which holds.
  - A9 nothing else is serialized: Either the clause names `serial` and `offset` of struct mesh_hop (rdma/mesh.h) as
    part of an instruction's page and dependency, or mesh_plan (rdma/mesh-collective.c:2043-2045) stops writing them.
    The clause states no more.
  - A10 data structure: the calls' dataflow graph: The export stage (compiler/mesh_compiler/export.py) produces the
    graph of a program's calls with torch.fx's tracer, each compiled function a leaf. Operand: the program's block;
    result: the calls' graph, whose nodes' tensor arguments and results are their reads and writes; consumer: A10's
    analysis.
  - A10 algorithms: independence, commuting operations, the bound: An analysis in the compile chain, between export
    and the schedule stage, takes the calls' graph and returns the order constraints between calls and, for each
    function, the most calls outstanding at once. It calls rdma/mesh_graph.py closure, paths and width. Consumers: the
    schedule stage (mesh_plan's orders, causes, chains and struct mesh_function.outstanding) and the runtime.
  - A10 the graph operations are the graph module's (callers; Interpreter; wherever mesh walks a graph): A10's
    analysis in the compile chain calls rdma/mesh_graph.py's functions and runs a body's symbolic execution on
    torch.fx.Interpreter. compiler/mesh_compiler/schedule.py producer's walk is a call of a mesh_graph function with
    the node and its following-node function as operands.
  - A10 produced by, consumed by (the program issues calls ahead): runtime.Program issues a function's calls ahead of
    their results, up to the analysis's bound and in the analysis's order; its submission reads the function's struct
    mesh_sync entries (mesh_transfers_start, rdma/mesh-dataflow.c:234).
  - Appendix: a chooser for edge weights: A function, apart from the compilation, from (bytes, cable) to α_k + n β_k,
    with β_k from the map's bandwidth (rdma/mesh-map.h) and α_k fitted from measured transfer times of at least two
    sizes on the cable; its result is passed as A3's edge weights.
  - Appendix item 2: nothing waits to write; the only waits: mesh_transfers_start's wait on a struct mesh_sync count
    (rdma/mesh-dataflow.c:247-255) ends when the count rises to its need or when the installed view is no longer the
    plan's; the clause lists no other end.
  - Appendix item 3 (a slow node, "desynchronization": no handler): If the bound at rdma/mesh-dataflow.c:246-250 is a
    handler for a slow node, it is removed there (the same lines as the item 2 entry "nothing waits to write; the only
    waits").
  - Appendix item 5: every program declares each consumer's kind: The compile chain's IR carries each consumer's kind
    and the schedule stage writes it as struct mesh_declared.kind (MESH_JOIN or MESH_PAGES,
    rdma/mesh-collective.h:30,37) for mesh_plan. Operand: the placed graph's consumers; result: the kind of each
    declared transfer.
  - Appendix item 6, stage: a program holds one scheduled value for each compiled function (the compile chain):
    runtime.Program instances of one program's compiled functions stand at once across machines: each function's
    static schedule is published under (the program's group, the function's ordinal) by the one mesh_plan call over
    the program's functions. Caller: the program; operand: its scheduled values.
  - The files and functions: rdma/mesh_compile.py: As the A4 (every operator; AllGather), A7 (the launcher passes the
    program's nodes) and A10 (analysis) entries state.
  - The files and functions: rdma/mesh-collective.c (fed by A2's collectives): As the A4 stage entry states.
  - The files and functions: metal-microbench tools/e2b_replay.m: As the A7 entry "the launcher passes the program's
    nodes" states.
  - The files and functions: metal-microbench tools/prepare_e2b.py: As the A1 delegate entry and the item 5 "every
    program declares each consumer's kind" entry state.
  - The files and functions: metal-microbench metal_recording.m: As the A1 delegate entry states.
  - A9 here, per node: the lists: outstanding [2026-11-12, mesh a7e2de47]. Read at that commit, the clause did not hold in the
    `compile_module` chain: `compiler/mesh_compiler/bridge.py` wrote the records itself (review of a7e2de47). Mesh
    81c8e9da has the schedule stage call `mesh_plan`; the clause is not yet read against it.
  - A9 here, per node: the bindings and the state section: outstanding [2026-11-12, mesh a7e2de47]. Read at that commit, the clause did not hold in the
    `compile_module` chain: `compiler/mesh_compiler/bridge.py` wrote the records itself (review of a7e2de47). Mesh
    81c8e9da has the schedule stage call `mesh_plan`; the clause is not yet read against it.
  - A9 every address is fixed at compile time: outstanding [2026-11-12, mesh a7e2de47]. Read at that commit, the clause did not hold in the
    `compile_module` chain: `compiler/mesh_compiler/bridge.py` wrote the records itself (review of a7e2de47). Mesh
    81c8e9da has the schedule stage call `mesh_plan`; the clause is not yet read against it.

Covered:

  - How this document is written and used (silence rule): covered [2026-11-12, read at mesh 9aa74ed1]
  - The stages each requirement belongs to (1-6): covered [2026-11-12, mesh 52b2f91d]
  - A1 data structure and algorithm: covered [2026-11-10, mesh ce2e8ac9]
  - A1 not a torch.compile backend: covered [2026-11-10, mesh ce2e8ac9]
  - A1 produced by, consumed by (the type): covered [2026-11-12, mesh 5a2b211b]
  - A2 data structure, part 1 (sharding specs): covered [2026-11-11, mesh 297c3cda]
  - A2 data structure, part 2 (mapping onto the map's nodes): covered [2026-11-12, mesh 52b2f91d]
  - A2 algorithm (GSPMD's partitioning): covered [2026-11-12, mesh f03cb9ca]
  - A2 produced by, consumed by: covered [2026-11-12, mesh 52b2f91d]
  - A2 the chooser is separate: covered [2026-11-12, mesh 991742c8]
  - A3 data structure: covered [2026-11-10, mesh 60103676]
  - A3 no algorithm computes them: covered [2026-11-10, mesh 5889d898]
  - A3 produced by, consumed by (router and scheduler): covered [2026-11-10, mesh 5889d898]
  - A3 the chooser of the weights: covered [2026-11-12, mesh 52b2f91d]
  - A4 data structure: the instruction DAG: covered [2026-11-10, mesh 5889d898]
  - A4 data structure and algorithm: packed spanning trees, packing: covered [2026-11-10, mesh 5889d898]
  - A4 algorithm: joint routing: covered [2026-11-10, mesh 5889d898]
  - A4 produced by, consumed by (mesh_plan's types): covered [2026-11-10, mesh 60103676]
  - A4 which nodes hold a function's static schedule: covered [2026-11-10, mesh 5889d898]
  - A4 a node to which the placement operand maps no device (mesh_plan's part): covered [2026-11-10, mesh 5889d898]
  - A5 data structure and algorithm: covered [2026-11-10, mesh 5889d898]
  - A6 data structure and algorithm: covered [2026-11-10, mesh 5889d898]
  - A6 produced by, consumed by (the lists): covered [2026-11-10, mesh 5889d898]
  - A7 stage: covered [2026-11-11, mesh 1a3cfe60]
  - A7 one order; total order multicast by a fixed sequencer (primitives, properties, steps of Fig. 5): covered
    [2026-11-06, mesh a172596f]
  - A7 multicast to many groups: covered [2026-11-06, mesh 128f9bb5]
  - A7 data structures: m, sn(m), Dest(m): covered [2026-11-06, mesh 33a03e35]
  - A7 sender(m) (the bridge's and mesh_plan's part): covered [2026-11-10, mesh 5889d898]
  - A7 the sequencer: covered [2026-11-06, mesh 128f9bb5]
  - A7 deliver(m); produced by, consumed by: covered [2026-11-06, mesh 128f9bb5]
  - A7 failures: covered [2026-11-06, mesh 128f9bb5]
  - A7 one sender for each program: covered [2026-11-06, mesh 128f9bb5]
  - A7 views: the four view properties: covered [2026-11-06, mesh a172596f]
  - A7 views: a process: covered [2026-11-08, mesh 7517cfc8]
  - A7 views: the membership service is the sequencer: covered [2026-11-06, mesh 128f9bb5]
  - A7 views: the view identifier: covered [2026-11-06, mesh 128f9bb5]
  - A7 views: a bridge installs a view: covered [2026-11-06, mesh 128f9bb5]
  - A7 views: Fig. 5 runs within each view: covered [2026-11-06, mesh 128f9bb5]
  - A7 views: a change of view blocks and flushes: covered [2026-11-06, mesh 128f9bb5]
  - A7 views: a static schedule is used only within its view: covered [2026-11-06, mesh 128f9bb5]
  - A7 views: every member computes its static schedules on the view's map: covered [2026-11-06, mesh 128f9bb5]
  - A7 a static schedule's destinations exist before its first message: covered [2026-11-06, mesh 128f9bb5]
  - A8 data structure (the type mesh_plan takes): covered [2026-11-10, mesh 60103676]
  - A9 stage and types: covered [2026-11-08, mesh 7517cfc8]
  - A9 the algorithm the bridge runs it by: the self-timed strategy: covered [2026-11-06, mesh 4fe54e5f]
  - A10 the graph operations are the graph module's (the functions): covered [2026-11-08, mesh 38ce722e]
  - A10 its output (the receiving types): covered [2026-11-10, mesh 60103676]
  - A10 produced by, consumed by (the schedule): covered [2026-11-07, mesh 8c8094da]
  - Appendix item 1 (unrolling): covered [2026-11-10, mesh 5889d898]
  - Appendix item 2: data structures and algorithms (the queue, its size, the merge): covered [2026-11-07, mesh 8c8094da]
  - Appendix item 2: what is ordered, and what is not; interleaving: covered [2026-11-06, mesh 128f9bb5]
  - Appendix item 2: recorded order: covered [2026-11-06, mesh 4fe54e5f]
  - Appendix item 2: delivery at a node (Fig. 5's destinations; no wait on an earlier call): covered [2026-11-06,
    mesh a172596f]
  - Appendix item 2: work and transfers run at once: covered [2026-11-06, mesh 4fe54e5f]
  - Appendix item 2: the queue is laid out with the schedule's pages and read in place: covered [2026-11-10, mesh 5889d898]
  - Appendix item 2: which files define them: covered [2026-11-10, mesh 60103676]
  - Appendix item 2, change: the queues replace the sequence counters, taken[p] and the one channel: covered
    [2026-11-08, mesh 7517cfc8]
  - Appendix item 2, change: remove the start's wait and the late-loading port's refusal: covered [2026-11-06,
    mesh 4fe54e5f]
  - Appendix item 2, change: remove the coordination's Q3′: covered [2026-11-06, mesh 128f9bb5]
  - Appendix item 4 (no truncation at one end; no handler): covered [2026-11-10, mesh 5889d898]
  - Appendix item 5: the set: covered [2026-11-10, mesh 5889d898]
  - Appendix item 5: the algorithm (Fisher and Yates over Threefry; Lemire): covered [2026-11-06, mesh 4fe54e5f]
  - Appendix item 5: one permutation of the whole set per execution: covered [2026-11-06, mesh 4fe54e5f]
  - Appendix item 5: a page's round rule reads a count that only rises: covered [2026-11-05, mesh 930861b0]
  - Appendix item 5: joins are sent in the schedule's order: covered [2026-11-10, mesh 5889d898]
  - Appendix item 6: the compiled function's name; alike at every rank: covered [2026-11-10, mesh 5889d898]
  - Appendix item 6, change: any number of compiled functions at once (mesh_plan): covered [2026-11-10, mesh 5889d898]
  - Appendix item 6, change: one allocation over the whole program; deferred reclamation: covered [2026-11-10,
    mesh 5889d898]
  - Appendix item 6, change: each call launches one execution: covered [2026-11-08, mesh b880b808]
  - Appendix item 8 (the heartbeat's holders): covered [2026-11-06, mesh 31ee5d4a]
  - The files and functions: rdma/mesh_graph.py: covered [2026-11-08, mesh 38ce722e]
  - The files and functions: rdma/mesh-graph.h, mesh-graph.c: covered [2026-11-10, mesh 60103676]
  - The files and functions: rdma/mesh-collective.c (routing, u, queue sizes, sets): covered [2026-11-10, mesh 5889d898]
  - The files and functions: rdma/mesh-bridge.h: covered [2026-11-06, mesh 33a03e35]
  - The files and functions: rdma/mesh-records.c: covered [2026-11-06, mesh 128f9bb5]
  - The files and functions: rdma/mesh-flow.c: covered [2026-11-06, mesh 4fe54e5f]
  - The files and functions: metal-microbench tools/mesh/run_pair.py: covered [2026-11-09, metal-microbench f08d3c7b]

Withdrawn:

  - Appendix item 7 (torch's existing kernels on the GPU): withdrawn [2026-11-12 reading].
    design/aot-compilation-change-request.md:1030-1032; the replacing item is
    design/torch-source-lowering-change-request.md:136 "1. No torch kernel; every function from source".

## [torch-capture-change-request.md](torch-capture-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 27 items outstanding, 5 covered, 0 withdrawn.

Outstanding:

  - 1, Write, first bullet (every Metal call recorded and replayed in order): A capture holding the program's Metal
    calls with their arguments in order, and a replay that issues them in that order by the calls
    MTL4ComputeCommandEncoder.h and MTL4CommandQueue.h declare. The clause places it only in a flow that executes and
    records the program before its static schedule exists; it names no stage of compile_module and no shared header,
    and none of export, place, on_devices, schedule, argument_buffers has that flow.
  - 1, Write, second bullet (a call Metal 4 declares no call for): In the capture of the first bullet: a recorded call
    with no Metal 4 declaration invalidates the capture, and that call is listed in this request. No compile_module
    stage is named by the clause.
  - 1, Write, third bullet (pipelines from a copy of the descriptor with supportIndirectCommandBuffers): In the
    capture of the first bullet: each recorded compute-pipeline creation call makes its pipeline from a copy of the
    program's descriptor with supportIndirectCommandBuffers set. No compile_module stage is named by the clause.
  - 1, "A call the program makes only while a condition holds", Write, first bullet: In the capture and replay of item
    1: a call recorded under a condition is issued at its recorded position in every execution's replay, the condition
    not evaluated. No compile_module stage is named by the clause.
  - 1, "A call the program makes only while a condition holds", Write, second bullet (initial contents): In the
    capture of item 1: the start contents of each buffer read before written are saved with the capture, and for a
    buffer the captured calls also write they are applied before each execution's replay. No compile_module stage is
    named by the clause.
  - 2, "The number of locations of a value the CPU writes", Write: After schedule.argument_buffers: for each
    feedforward edge e that Scheduled.feedforward names, Bff(e) = mesh_graph.buffer_bound over the IPC graph of
    schedule.schedule's graphs with the t(v) estimates as operand; the result is the value's number of locations, a
    field of Scheduled that runtime.Program and bridge.entries read in place of the constants at bridge.py:23 and :78.
  - 3, torch scope: As torch-source-lowering-change-request.md item 1 states it: the operator lowering (before place)
    produces, for a consumer that takes pages in any order, one task per page with an edge from its own page, and the
    IR (ir.Exported/ir.Scheduled) carries which of the two kinds each consumer is; place, schedule and lower read it.
  - 3, LiteRT scope, Write, first bullet (index space analysis): Takes a recorded dispatch and the source of the
    recorded newLibraryWithSource call; writes each load's and store's index function and the kernel's class,
    contiguous or discontiguous. The item says its input is the recorded dispatch, so it is placed only in the capture
    flow of item 1; no compile_module stage is named.
  - 3, LiteRT scope, Write, second bullet (the pages a threadgroup reads): Takes the index functions of the first
    bullet and the partition size; writes, per threadgroup, the partitions it reads (every partition for a
    discontiguous kernel). Placed with the first bullet in the capture flow; no compile_module stage is named.
  - 3, LiteRT scope, Write, third bullet (first dispatch predicated by the mask): Takes a recorded dispatch that reads
    a received operand, the presence bits and the second bullet's pages per threadgroup; writes the mask, one element
    per threadgroup. Placed in the capture flow; no compile_module stage is named.
  - 3, LiteRT scope, Write, fourth bullet (stream compaction into the worklist): Takes the mask; writes the worklist
    (dense list of threadgroup indices) and its length in the indirect buffer of the dispatch that executes it. Placed
    in the capture flow; no compile_module stage is named.
  - 3, LiteRT scope, Write, fifth bullet (second dispatch executes the worklist): Takes the kernel, the worklist and
    its length in an indirect buffer; issues the kernel's second dispatch with the indirect threadgroup count, each
    threadgroup testing its presence bits again. Placed in the capture flow; no compile_module stage is named.
  - 3, LiteRT scope, Write, sixth bullet (task list in order of completion times): Takes the dispatches that execute
    worklists, the completion times of the transfers they read and the precedence relation; writes the task list in
    non-decreasing completion time, ties consistent with precedence. Placed in the capture flow; no compile_module
    stage is named.
  - 4, "It takes / It writes" (the event's value): Takes each execution's start (the store of submissions,
    rdma/mesh-dataflow.c:260); writes the Metal event's uint64_t value that the command queue's waitForEvent:value:
    reads. In the chain the command queue is rdma/mesh-metal4.m's, called from lower_metal's function.
  - 4, "The value the event is set to at a failure", Write: With the event of the entry "4, It takes / It writes": at
    a failure its value is set to UINT64_MAX, it is not signalled again, and the next view's requests take a new
    event. Where the chain holds a command queue: rdma/mesh-metal4.m, called by lower_metal's function.
  - 5, Write (the release's one 64-bit atomic write): lower_metal's emitted kernel source for a function whose result
    is sent takes the execution's number t and, after its stores, writes the release word: the fence, then
    atomic_max_explicit of the 64-bit word with t + 1. The word is the one the bridge reads (bridge.py:87
    publication.targets[0]).
  - 6, Write, first bullet (offsets by Greedy by Size): From schedule.schedule's result: one usage record per location
    of each operand of a function typed contiguous_input or contiguous_output; mesh_graph_greedy_by_size
    (rdma/mesh-graph.h:309) takes the records and writes each record's offset and total_size. The clause names no
    function that writes or reads the offsets and no type for an offset.
  - 6, "Where the block lies, and the usage records of the u locations", Write, first bullet (the block is its own
    allocation): With the offsets of item 6's first bullet: one allocation of total_size, consecutive pages holding no
    page of another operand, made where the chain places pages (the clause names no function).
  - 6, "Where the block lies, and the usage records of the u locations", Write, second bullet (u usage records;
    do_arcs_overlap on the ring): From schedule.schedule's result and item 2's u: u usage records per contiguous
    operand, each with the operand's size and its location's interval over the u executions; mesh_graph_greedy_by_size
    takes them with mesh_graph_arcs_overlap (rdma/mesh-graph.h:306) as its intersection test.
  - 7, Write, first bullet (a captured buffer's interval): Takes the capture's commands; writes one interval per
    buffer, from its first write to its last read, in positions of those commands. The clause places it only in the
    capture flow of item 1, names no type for the interval and no function that reads it, and names no compile_module
    stage.
  - 7, Write, second bullet (a call's inputs and results): With the intervals of the first bullet: an input's interval
    starts at the capture's first command and ends at its last read; a result's starts at its first write and ends at
    the capture's last command. Capture flow only; no compile_module stage is named.
  - 8. The torch scope's final lowering: As torch-source-lowering-change-request.md items 1-3 state it. In the chain:
    lower_metal.lower(scheduled) is the function that makes each mps processor's function from source; the callers
    that run mps processors without it are examples/tp.py:128 and :171 (Program(scheduled) and Program(fixed, timed
    GraphModules)).
  - 9. Every node replays: Every node runs its program by replay of a capture, collectives included, with no call of
    torch from Python per execution. The clause places it only in a flow that executes and records each node's program
    before its static schedule exists; it names no compile_module stage. In the chain the per-execution caller is
    runtime.Program (run, execute) with the functions lower returns.
  - Source for item 2's execution-time estimates, Write, first sentence (the lowering supplies the cost model's
    operands): Each target lowering (lower_metal.lower, lower_coreml.lower, lower_inductor.lower) supplies, per
    compiled function, its operations, bytes read and written and execution mapping as a struct mesh_graph_actor;
    mesh_graph_actor_time (rdma/mesh-graph.h:72) takes it with the machine's struct mesh_graph_machine as operand and
    writes t(v).
  - Source for item 2's execution-time estimates, Write, second sentence (CPU and GPU rates): In the supply of the
    first sentence: a CPU processor's function and the CPU's write of item 2 are timed with the cpu machine record; a
    GPU function with its GPU mapping and the gpu record (the two results of mesh_metal_machines).
  - Source for item 2's execution-time estimates, Write, third and fourth sentences (estimates stored; Bff supplies
    locations and reuse edges): The t(v) of the first sentence are stored with the compiled functions (what lower
    returns, or a field of Scheduled that archive.save writes) and are the operand of item 2's Bff(e); Bff(e) is the
    value's number of locations and the reuse edges of the scheduled program.
  - Source for item 2's execution-time estimates, Write, last sentence (transformations update costs): A
    transformation of the chain that changes a compiled function (lower_metal's fusion of nodes into one kernel,
    driver.lowered's successive lowerings) writes the resulting function's cost-model operands with it.

Covered:

  - 2, "It takes / It writes" (the IPC graph is a function of the static schedule): covered [2026-11-10, mesh ce2e8ac9]
  - 2, Write (when execution t's requests into a location start): covered [2026-11-11, mesh 1a3cfe60]
  - 4, Write, first bullet (execution t's waits; no fixed count): covered [2026-11-11, mesh 1a3cfe60]
  - 4, Write, second bullet (the waits end at the free): covered [2026-11-11, mesh 1a3cfe60]
  - 6, Write, second bullet (mesh_view_compose's remapping deleted): covered [2026-11-08, mesh b880b808]

## [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 34 items outstanding, 12 covered, 0 withdrawn.

Outstanding:

  - 1 / Replaces: `_apply` leaves with rdma/mesh_compile.py under design/the-compiler.md order 5; nothing in
    compiler/mesh_compiler calls it.
  - 1 / Write: no torch Metal kernel: Every function runtime.Program calls for a processor whose device is the GPU is
    one lower(scheduled) returned from source (lower_metal.lower), in examples/tp.py, pl.py, cp.py, ep.py, crossing.py
    and the check's paths as in examples/engines.py:112-113; no GraphModule is executed by torch on an mps device.
  - 1 / Write: the task graph: The graph that place and schedule consume has a vertex per task and an edge per page
    passed between tasks, with w and c (R3), carried in ir.Exported's graph; the operator lowering that runs between
    export and place produces it.
  - 1 / Write: the operator lowering produces them on every backend: An operator-lowering function between export and
    place in compile_module: operands the exported graph and the placement operand, result nodes of that same graph
    (the tasks and page edges) for each function on the GPU, the CPU and the Neural Engine; place and schedule take
    its result.
  - 1 / Write: the synchronous, complete, contiguous consumer: The operator lowering (between export and place) emits,
    for a consumer of this kind, one task with an edge from every page of its operand, and types that operand
    `contiguous_input` or `contiguous_output` as one run of pages.
  - 1 / Write: the asynchronous, incomplete, noncontiguous consumer: The operator lowering emits, for a consumer of
    this kind, one task per page with an edge from its own page, its outputs pages that leave as each is produced.
  - 1 / Write: the IR carries the kind: The graph ir.Exported / ir.Scheduled carry states, per consumer node, which of
    the two kinds it is; archive.save and load carry it.
  - 2 / Write: the operator lowerings, then placement, consume it: compile_module calls the operator lowerings of
    items 1, 5 and 6 on the exported graph and the placement operand, and passes their result to place.
  - 3 / Replaces: MetalOrder leaves with metal_recording.m under design/the-compiler.md order 5; the order of GPU
    commands in the chain is the schedule stage's (next entries).
  - 3 / Write: the algorithm for the order: schedule.schedule(module, cost) orders each processor's functions by
    mesh_graph_completion_times (operands: each task's length from `cost`, its processor as the resource, the
    dependences schedule.py's `preds` gives) followed by mesh_graph_heap_order on those completion times, both of
    rdma/mesh-graph.h, in place of mesh_graph.list_schedule.
  - 3 / Write: the order is published: bridge.entries / publish write, with the plans they publish, the order schedule
    produced (the completion times' heap order of the previous entry) for the sends and receives carried between
    machines.
  - 3 / Write: Sriram's synchronization graph: The lowering of ir.Scheduled forms the synchronization graph over its
    tasks (mesh_graph_synchronization of rdma/mesh-graph.h, operands the tasks and the dependences of the scheduled
    graphs) and writes a Metal ordering only for the edges that graph keeps.
  - 3 / Write: same threadgroup: lower_metal's kernel source: a dependence whose two ends are in the same threadgroup
    is written as a threadgroup barrier.
  - 3 / Write: independent pairs: In lower_metal.Lowering (send, receive, wait) and rdma/mesh-metal4.m
    (mesh_metal4_commit, mesh_metal4_wait): an event signal and wait are written only between a send or result and the
    commits whose writes it reads; a commit it does not depend on gets no ordering with it.
  - 5 / Replaces: See item 5 / Write: `_lstsq_metadata`.
  - 5 / Write: one node, one LAPACK driver: The operator lowering between export and place makes each of svd, eigh,
    eig, eigvals and lstsq (gelsd, gelss) one node of the exported graph whose function is the LAPACK driver called
    through Accelerate, with the node's fixed shapes as operands.
  - 5 / Write: it is an operator lowering: compile_module calls this lowering on the exported graph before place;
    place and schedule take the node as any other.
  - 5 / Write: which routine: For lstsq: the node's function is dgelsd_ / dgelss_ as the driver names. For svd, eigh,
    eig and eigvals the clause names no routine; nothing beyond the previous entries is stated.
  - 5 / Write: INFO: The LAPACK node's results include INFO; runtime.Program's caller is told of a positive INFO.
  - 5 / Write: pinv: The operator lowering writes pinv as PyTorch's composition over the svd node (hermitian=False) or
    the eigh node (hermitian=True).
  - 5 / Write: `_lstsq_metadata`: The operator lowering's lstsq node takes the four drivers PyTorch names and gives
    the output shapes PyTorch gives each.
  - 5 / Write: matrix_exp: See item 6 / Write: matrix_exp.
  - 5 / Write: the plain message: The operator lowering between export and place raises, for each such operator on the
    processor the placement operand names, one message that names the operator and says an iteration depth that
    depends on data is not compiled.
  - 5 / Write: the cited functions are the CPU's: The LAPACK node of this item is the operator's lowering where the
    placement operand names the CPU; on other processors, item 6.
  - 6 / Write: nodes of the exported graph on every processor: An operator lowering between export and place that
    rewrites each operator of this item into matrix multiplication, QR, Cholesky, triangular-solve and selection nodes
    of the exported graph; place gives each node its processor and sharding from the placement operand.
  - 6 / Write: the polar decomposition: The operator lowering writes the polar decomposition as the nodes of `_qdwh`'s
    steps: coefficients computed at compilation from l = eps with operand max_iterations (default 10), each step a QR
    node or a Cholesky node and two triangular-solve nodes with matrix multiplications, then the refinement and the
    Hermitian factor; whether it converged is a value among the results.
  - 6 / Write: the Halley continuation: None stated by the clause beyond item 5's plain message for the continuation.
  - 6 / Write: eigh: The clause names QDWH-eig and states no compilable depth; under item 5, eigh off the CPU gets the
    plain message from the operator lowering.
  - 6 / Write: svd, pinv, lstsq: The operator lowering writes svd as the polar-decomposition nodes, the
    eigendecomposition of the Hermitian factor and one product node; pinv and lstsq (gelsd, gelss) as PyTorch's
    compositions of that svd or eigh.
  - 6 / Write: matrix_exp: The operator lowering writes matrix_exp as the Padé nodes of `_calc_P_Q`, then
    max_squarings (operand, default 16) squaring steps each a selection between the square and its input by i <
    n_squarings, with a selection of NaN where n_squarings > max_squarings and a runtime error reported by the node.
  - 6 / Write: the CPU for the whole operator: The operator lowering reads the placement operand: where it names the
    CPU for the whole operator it produces item 5's LAPACK node, otherwise this item's nodes.
  - 6 / Write: no body repeated by tensor data: The lowerings of this item, when written between export and place,
    have step counts fixed at compilation (max_iterations, max_squarings) and selections in place of branches.
  - 6 / Write: eig and eigvals off the CPU: None stated by the clause; on the CPU, item 5's node.
  - 6 / Write: the Neural Engine: None stated by the clause.

Covered:

  - 2 / Write: the exported graph: covered [2026-11-10, mesh ce2e8ac9]
  - 2 / Write: the export stage produces it: covered [2026-11-10, mesh ce2e8ac9]
  - 2 / Write: dynamic control flow is not compiled: covered [2026-11-10, mesh ce2e8ac9]
  - 2 / Write: a dimension known only at run time: covered [2026-11-10, mesh ce2e8ac9]
  - 3 / Write: one schedule stage: covered [2026-11-10, mesh ce2e8ac9]
  - 3 / Write: each function's processor: covered [2026-11-10, mesh ce2e8ac9]
  - 3 / Write: each task's length: covered [2026-11-10, mesh ce2e8ac9]
  - 3 / Write: commands written in that order: covered [2026-11-12, mesh a86c7dc4]
  - 3 / Write: same encoder: covered [2026-11-12, mesh a86c7dc4]
  - 3 / Write: same queue: covered [2026-11-12, mesh 2c0e94ef]
  - 3 / Write: another queue, another device or the CPU: covered [2026-11-12, mesh a86c7dc4]
  - 5 / Write: no body repeated by tensor data: covered [2026-11-10, mesh ce2e8ac9]

## [synchronization-signature-change-request.md](synchronization-signature-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 13 items outstanding, 7 covered, 0 withdrawn.

Outstanding:

  - 1 / The algorithm: In the lowering of ir.Scheduled (lower_metal.lower and its per-processor program of commits,
    fence updates, fence waits, barriers and event waits), the dependences, the synchronization edges, the redundant
    marks and the order are the results of mesh_graph_dependences, mesh_graph_synchronization, mesh_graph_redundant
    and mesh_graph_schedule of rdma/mesh-graph.h, called with the lowering's own tasks and accesses as operands; their
    results are the operands of mesh_graph_signature.
  - 1 / Produced by: A function of the lowering of ir.Scheduled that returns the signature: its operands are the
    Scheduled and the lowered per-processor programs (lower_metal's COMMIT/WAIT/RECV/SEND entries, their barriers and
    fences; the SEND/RECV of machines carried by the bridge); its result is mesh_graph.signature's (processors, tasks,
    dependences, operations -> mesh_graph_signature). Its callers are the printers, the monitor's generator and
    archive.save.
  - 1 / Consumed by: printed: The signature produced from the lowering of ir.Scheduled is passed to
    mesh_graph_matrix_market, mesh_graph_dot and mesh_graph_space_time (rdma/mesh_graph.py forms) and printed.
  - 1 / Consumed by: the monitor's D program: The signature of the lowering of ir.Scheduled is the operand of the
    monitor's generator (see item 4, first Write clause).
  - 1 / Consumed by: SHA-256 in the artifact: archive.save(directory, scheduled) writes the signature's digest
    (mesh_graph_signature.digest) into the archive it writes, and archive.load returns it; operand: the signature
    produced from the lowering of that Scheduled.
  - 3 / Write: the monitor is DTrace: The monitor's probes catch the Metal 4 selectors and the thread's waits of a
    process that runs runtime.Program with the functions lower(scheduled) returned (the calls of rdma/mesh-metal4.m:
    commit:count:options:, signalEvent:value:, waitForEvent:value:, updateFence:, waitForFence:,
    barrierAfterEncoderStages:, waitUntilSignaledValue:timeoutMS:), with no condition on a function of
    metal_recording.m.
  - 4 / Write: the D program stops the process: The monitor's generator takes the signature produced from the lowering
    of ir.Scheduled (item 1) and writes the D program whose destructive action stops the process that runs
    runtime.Program at the first call that signature does not have at that point of that queue's, thread's or bridge's
    order, printing ustack().
  - 4 / Write: at the end: After a run of runtime.Program under the monitor, the observed operations are made a
    signature by mesh_graph.signature with the compiled signature's processors, tasks and dependences; a count that
    differs fails the run; both are printed by the three printers.
  - 4 / Write: the digest and the review record: A measured run of a program compiled by compile_module appends the
    digest of that program's signature to metal-microbench output_data/change-request-reviews.jsonl.
  - 5 / Write: replay with the monitor: A program compiled by compile_module (and one loaded by archive.load) is run
    by runtime.Program with the monitor of items 3 and 4 attached to that process, the monitor's operand the signature
    of that Scheduled.
  - 5 / Write: a program with no wait: The signature of the lowering of an ir.Scheduled whose schedule has no wait
    lists only the operations the lowering's method contains; it is read from the signature item 1 produces.
  - 5 / Write: a wait the signature does not have: With the monitor attached to a process running runtime.Program, a
    wait absent from that Scheduled's signature stops the process at the call; a run that completes has executed
    exactly the signature's operations.
  - 6 / Write: lower_metal.lower, which writes each kernel's source, gives per compiled kernel the count of
    threadgroup barriers, simdgroup barriers, fences and atomics that kernel contains, as operations of kind
    MESH_GRAPH_KERNEL of the signature item 1 produces.

Covered:

  - 1 / The data structure: covered [2026-11-08, mesh d44e3d0c]
  - 1 / It holds: covered [2026-11-08, mesh d44e3d0c]
  - 2 / Matrix Market: covered [2026-11-08, mesh d44e3d0c]
  - 2 / DOT: covered [2026-11-08, mesh d44e3d0c]
  - 2 / Space-time diagram: covered [2026-11-08, mesh d44e3d0c]
  - 3 / Write: the program is not changed: covered [2026-11-09, mesh 0b5f01bc]
  - 4 / Write: every edge the image of a dependence, none redundant: covered [2026-11-08, mesh d44e3d0c]

## [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 40 items outstanding, 21 covered, 0 withdrawn.

Outstanding:

  - R1 (requirement): The send of a channel carried between machines (bridge.Carried.put, called by ir.send through
    Program) names the pages that already hold the sent value: operands the value's page-table rows and its view;
    result hop entries (page, offset, bytes) over those pages, consumed by rdma/mesh-flow.c. The copy_ into the
    channel's pages goes.
  - R1 algorithm: In the schedule of compile_module's chain: each carried transfer's two views (offset, sizes,
    strides; Scheduled.carried holds dtype, shape, stride) are flattened to (row, offset, bytes) blocks with adjacent
    blocks merged; operands struct mesh_strided from/to, result struct mesh_hop page/offset/bytes;
    mesh_strided_extents (rdma/mesh-collective.h:35) is the shared function that does it.
  - R1 where it composes: From each send/recv pair placement adds (schedule.carried,
    compiler/mesh_compiler/schedule.py:64): one struct mesh_declared with from/to struct mesh_strided, in the
    `declared` array of a struct mesh_function given to mesh_plan; result the cables' struct mesh_hop lists, read by
    rdma/mesh-flow.c.
  - R1 a value's layout is the program's own: A transfer's source is the producer's pages under its view and its
    destination is byte ranges of the destination value's rows (struct mesh_declared from/to, `value`); the schedule
    writes the hops from those. No layout is made for the transfer.
  - R1 Replaces: packing and unpacking; a layout chosen for a transfer's sake: Same as R1 (requirement):
    bridge.Carried.put's copy goes when the hops name the value's own pages.
  - R2 (requirement): In compile_module's chain, when the static schedule is made: every value of the exported program
    gets page-table rows (struct mesh_value.section, rdma/mesh-section.h) and each row its page; operands the values
    with their lifetimes; result rows that are part of the scheduled value. mesh_plan (rdma/mesh-collective.h:90) is
    the named producer.
  - R2 algorithm 1: the copies (C3.3): The schedule stage computes each value's least lags with mesh_graph_lags
    (rdma/mesh-graph.h:78) from the dependences of the scheduled graph and takes u as the largest; result u and, per
    value and copy, its rows, constants of the scheduled value.
  - R2 algorithm 2-3: the order and the chains: plan_place (rdma/mesh-collective.c) gives mesh_graph_chains the
    happens-before of the items' lifetimes as its `before` operand, one item a page; result each item's chain, the
    page it holds; reached from compile_module's schedule.
  - R2 algorithm 4: the order over u copies: The `before` operand of mesh_graph_chains in plan_place, over items
    (value, copy), is the relation the clause states, from the least lags mesh_graph_lags returns and u.
  - R2 Replaces greedy by size: plan_block and mesh_graph_greedy_by_size's call go from mesh_plan; a value with `run`
    is placed page by page as every other value is.
  - R2 a run is a mapping, not a placement: When the static schedule is made, for each operand seen as one run: the
    map from its consecutive virtual pages to its placed pages, written once (the document names no type for it,
    :2116-2117); the consumer's kernel reads the mapped run.
  - R2 a received range is pages of its destination value: In the chain's schedule, a recv node's landing is byte
    ranges of the rows of the value the receiving function reads (struct mesh_declared.to and .value); no section is
    made for the range.
  - R2 a rank's own part is shared, not copied: In the chain's schedule, the rows of a result for the rank's own part
    are bound to the operand's pages; result rows in the scheduled value.
  - R2 shared pages are one item of the allocation: The allocation of the chain's schedule treats pages two values
    share as one item whose lifetime runs from the earlier first write to the later last read.
  - R2 any value a transfer names is sent from its own pages: In the chain's schedule: a value named by a transfer has
    its rows in the registered window (struct mesh_value kind MESH_INVARIANT or MESH_EXECUTION, wire); Program binds
    the caller's input and the state over those pages; the send's hops name them.
  - R2 ranges may share a block where their bytes are disjoint: In the chain's schedule, two ranges' hop entries may
    name one page at disjoint (offset, bytes); each range has its own landed count word per copy.
  - R2 where it composes: compile_module's schedule passes the exported program's values with their lifetimes (struct
    mesh_value begin, end, lag, run) to mesh_plan; the rows and u it writes are part of the scheduled value, read by
    the hop lists and by lowering.
  - R2 Replaces (mesh_plan's rules): plan_function and plan_landings in rdma/mesh-collective.c accept: a transferred
    value that is not MESH_EXECUTION and wire or is named by several ranges; a lag-k value named by a declared
    transfer; shared values with different copies; two ranges sharing a block with disjoint bytes.
  - R3 the task graph: E are pages, with weight c: The task graph the chain's schedule builds has one edge per page of
    each transfer, with the edge weight c from the weighting operand (struct mesh_weight.page).
  - R3 where it composes: compile_module's schedule gives the scheduled graph as struct mesh_function (declared,
    weights, dependence) to mesh_plan, whose plan_schedule hands the instructions' length/resource and the edges'
    from/to to mesh_graph_completion_times.
  - R4 (requirement): The chain's IR carries each consumer's kind (struct mesh_declared.kind, MESH_JOIN or MESH_PAGES)
    into the schedule; the hop lists carry none.
  - R4 behaviour at run time, 1: Program's execution of a processor's function: a task whose declared kind is
    MESH_JOIN starts on the presence of every page of its operand (the range's phases and count words) and on nothing
    else.
  - R4 behaviour at run time, 2: For a consumer of kind MESH_PAGES: the schedule makes one task per page (as
    plan_schedule does, rdma/mesh-collective.c:822); at run time each task reads its own page's presence; the
    producer's supplied store follows the pages it has written.
  - R4 behaviour at run time, 3: The chain's hop lists are written in the order mesh_graph_completion_times and
    mesh_graph_heap_order give for the task graph with R4's edges.
  - R4 the record of a run: A record of a run giving, per function, when each page of its operand became present and
    when the function started, and each cable's sequence of sends. The document names no source and no type for it.
  - R4 data structure: Same as R3 (E are pages) with the kind from struct mesh_declared.kind: a join's task has an
    edge from every page; a pages consumer has one task and one edge per page.
  - R4 nothing synchronizes outputs: the release follows the producing work: Same as R1 (requirement): the producing
    function's result is over the pages the hops name, and the supplied store (mesh_host_publish) follows the
    function's own writes.
  - R4 contiguity is a mapping: Same as R2 (a run is a mapping): only a MESH_JOIN operand or a one-strided-progression
    operand is given a run, as a mapping written when the static schedule is made.
  - R4 where it composes: The kind of each consumer is a field of the chain's IR for each recv, passed as struct
    mesh_declared.kind to mesh_plan; result task-graph edges and struct mesh_hop.set.
  - R5 (requirement): compile_module's one schedule orders each cable's sends: operands the task graph (R3) with
    lengths from the weighting operand and each page's route; result per copy each cable's struct mesh_hop list in
    schedule order.
  - R5 algorithm steps 3-5: priority, one global topological order, each cable's subsequence: compile_module's
    schedule takes each instruction's completion time as priority, calls mesh_graph_heap_order (rdma/mesh-graph.h:96)
    with the task graph's edges, and writes each cable's list as that order's subsequence.
  - R5 a set's pages keep their places: The chain's hop entries of a pages consumer carry their set mark (struct
    mesh_hop.set) at their places in the order.
  - R5 where it composes: compile_module's schedule is the caller of mesh_plan's solve: operands the task graph,
    placement, the weighting operand and the map's capacities (struct mesh_nodes); result each copy's cable lists in
    the scheduled value. The trees are packed by mesh_graph_pack as the clause names it.
  - R6 where it composes: Placement's send/recv nodes reach the schedule as struct mesh_declared (source,
    destination); the schedule writes struct mesh_hop node and direction.
  - R8 there is no client word: hdr `client` and `client_start`, their stores (mesh-collective.c plan_publish,
    compiler bridge.publish) and their readers (mesh-holders.h mesh_algo_holder, mesh-dataflow.c mesh_retire and
    mesh_retired_release) go.
  - R8 a bridge with no function execution does nothing: The port thread in rdma/mesh-flow.c, with no released
    execution of any plan, performs no work between changes of hdr `plans`.
  - R8 data structure and algorithm: one offline allocation at compile time: Same as R2 (requirement): the allocation
    of the whole exported program's values is made once in compile_module's chain and its rows are part of the
    scheduled value; Program takes no pages.
  - R8 function executions ordered by a fixed sequencer: Judged under aot-compilation A7: each execution a program
    starts is numbered by the island's sequencer and served in that order.
  - R8 where it composes: the pages are part of the scheduled value: Scheduled (compiler/mesh_compiler/ir.py) holds
    each value's rows as written by the schedule; archive.save/load carry them.
  - R8 Replaces: any rule for several programs: rdma/mesh-records.c, mesh-flow.c and mesh-collective.c carry no rule
    whose subject is several programs at once.

Covered:

  - R1 data structure: covered [2026-11-06, mesh 31ee5d4a]
  - R1 Replaces: the share record: covered [2026-11-06, mesh 31ee5d4a]
  - R2 data structure: covered [2026-11-05, mesh dd215cbe]
  - R3 the task graph: V and w: covered [2026-11-11, mesh 29aed586]
  - R4 behaviour at run time, 4: covered [2026-11-06, mesh 31ee5d4a]
  - R4 algorithm: presence is the supplied word: covered [2026-11-11, mesh 6d4f007b]
  - R4 nothing synchronizes outputs one page at a time: landings read by masks: covered [2026-11-11, mesh 6d4f007b]
  - R5 data structure: one linear program over completion times: covered [2026-11-05, mesh f4838c9d]
  - R5 algorithm steps 1-2: cutting planes, HiGHS dual simplex: covered [2026-11-05, mesh f4838c9d]
  - R5 executions: covered [2026-11-06, mesh 5eeee1f9]
  - R5 every rank computes the same schedule: covered [2026-11-10, mesh 4a9e1cf4]
  - R5 changes by file: rdma/mesh-graph.{h,c}: covered [2026-11-05, mesh f4838c9d]
  - R5 changes by file: rdma/Makefile builds HiGHS: covered [2026-11-05, mesh 841cba44]
  - R5 changes by file: mesh-collective.c and mesh-flow.c: covered [2026-11-06, mesh 5eeee1f9]
  - R6 data structure and no run-time algorithm: covered [2026-11-10, mesh ce2e8ac9]
  - R7 (requirement and fact of the code): covered [2026-11-03, mesh 615aa4e7]
  - R7 where it composes: covered [2026-11-12, mesh 5a2b211b]
  - R8 attaching holds nothing; no EADDRINUSE: covered [2026-11-04, mesh ea9c9d53]
  - R8 pages released at retirement by deferred reclamation: covered [2026-11-03, mesh c01fa79d]
  - R8 where it composes: placement and weights are operands: covered [2026-11-12, mesh 52b2f91d]
  - R8 Replaces: mesh_claim_take, the claim-head walk, fragmentation ENOMEM: covered [2026-11-05, mesh 19c8f25e]

## [disjoint-pages-change-request.md](disjoint-pages-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 31 items outstanding, 43 covered, 1 withdrawn.

Outstanding:

  - A3 item 1 (Items): The allocator (mesh_plan's plan_place, and whatever stage of compile_module calls it) takes
    each page of each copy of each value as its own item, including the values now flagged `run`, and returns one page
    per item.
  - A3 item 2 (The order): The allocator's order operand: a predicate before(a, b) over items, true where the
    dependence graph orders every read of a before the first write of b, evaluated at the copies' distances; produced
    from the dependence graph the scheduled program already holds (schedule.py preds / channel_ends) and passed to
    rdma/mesh-graph.h mesh_graph_chains (or mesh_graph.py) by the allocator's caller.
  - A3 item 3 (The chains): The allocator calls the graph module's chain decomposition (mesh-graph.h
    mesh_graph_chains, or mesh_graph.py matching) with A3 item 2's order over all items, and assigns one page per
    chain.
  - A3 item 4 (Operands typed as one run): Where the static schedule is made, the allocator writes, for each operand a
    consumer is typed to see as one run, the mapping from the operand's consecutive pages to its placed pages;
    lowering reads the mapping. Operands: the placed pages of the operand; result: the mapping, carried in the
    scheduled value.
  - A3 "Which function computes it, and which read it": A storage step of compile_module, after
    schedule/argument_buffers, takes the scheduled program's values (each carried channel's tensor from `carried`,
    each processor's state and results) and its dependence order, calls the allocator once, and puts each value's
    pages in Scheduled. bridge.entries, lower_metal and Program read those pages; bridge.entries' running sum and
    lower_metal's per-call buffers go.
  - C3.1 data structure: The later pass's threadgroup count and the status word are values the storage step of
    compile_module places in the state section with the phases and counts; lower_metal reads their places from
    Scheduled.
  - C3.1 algorithm 1 (masked first pass): lower_metal.lower, for a processor whose function reads a recv's pages: the
    emitted first pass loads the phases at the places Scheduled gives and computes only in the threads whose pages are
    received. Operands: the recv's phases place and expected phase; result: the kernel source of that pass.
  - C3.1 algorithm 2 (compaction): lower_metal.lower: the first pass it emits writes the later pass's threadgroup
    count and the compacted list of the work left, into the places of C3.1's data structure.
  - C3.1 algorithm 3 (indirect dispatch): rdma/mesh-metal4.m mesh_metal4_commit takes, for a later pass, the buffer
    address of its threadgroup count and encodes the indirect dispatch; lower_metal passes that address from
    Scheduled.
  - C3.1 algorithm 4, later-pass half: lower_metal.lower: the later pass it emits compares each receive's field with
    the expected phase and, where it is one behind with the count due, ends and writes the status word placed for the
    schedule.
  - C3.1 "Written by, read by": The functions lower_metal.lower returns are the readers of the phases and the writers
    of the threadgroup count and status; Program (runtime.py) reads the status after a call.
  - C3.1 waits deleted: The kernels lower_metal.lower emits contain no wait on a mask or countdown; a recv's reader is
    the masked first pass and later pass of C3.1.
  - C3.2 (b) what is placed: The storage step of compile_module passes every value of the scheduled program to the
    allocator: exported.state (weights, buffers, constants), each processor's results, the carried channels, and the
    state section's arrays; results are their pages in Scheduled.
  - C3.2 (d) nothing laid out when created: Same step as A3 "Which function computes it": lower_metal's functions take
    their storages' pages from Scheduled (mesh_metal4_view over placed pages) and create no buffer of their own.
  - C3.3 "Written by, read by": The storage step of compile_module computes u from the scheduled program's dependence
    graph through the graph module's lags function (mesh-graph.h mesh_graph_lags) and gives it, with the copies'
    pages, in Scheduled; bridge.entries writes that u into mesh_state.copies.
  - C3.4 (b) new state in new pages, no wait: The allocator's operand includes the pages standing schedules hold, and
    it places a new schedule's state and values in pages none of them holds; plan_released's wait and bridge.publish's
    wait on `released` go.
  - C3.5: The dependence graph the allocator takes is produced from the scheduled program: each function's reads and
    writes in the order schedule() gave (schedule.py), with each send/recv pair's edge (channel_ends), handed to the
    storage step of compile_module as its order operand.
  - C3.6 (1) late cancellation: Same as C3.4 (b): a new schedule's state is placed in pages no standing schedule
    holds.
  - C3.6 (3) local countdown: Same as C3.1 waits deleted.
  - C3.6 (5) final wait bounded: Program's wait for a call's completion (runtime.py __call__, through mesh_metal4_wait
    and bridge.Carried) is bounded.
  - C3.7 only zeroing: mesh-collective.c plan_bind_values' memset of `state` values' pages goes; a value that must
    start at a given content is written by its producer.
  - C4 metal-microbench written: tools/e2b_replay.m: Same as C3.5: the dependence operand comes from the scheduled
    program inside compile_module.
  - D.3.2: The storage step of compile_module places every value a lowering binds, with exported.state as invariants;
    Program writes each invariant into its placed pages once before the first call, and the lowered functions read
    only placed pages.
  - D.3.3: compile_module's storage step takes, for each input the CPU writes per execution (Exported.user_inputs),
    its declared lag, and gives it that many copies through C3.3's expansion; Program.bind writes copy t mod lag.
  - D.3.4 algorithm: lower_metal's function commits the commands before a later pass, has the queue wait on the event,
    and commits the later pass in the next command buffer, through rdma/mesh-metal4.m. The event value to wait for is
    not stated by the document.
  - D.3.4 completion: rdma/mesh-metal4.m mesh_metal4_wait: the wait on the stream's event is bounded, or completion is
    taken from the MTL4CommitFeedback handler.
  - D.3.4 "Written by, read by": The functions lower_metal.lower returns are the one maker and committer of command
    buffers for a scheduled program; Program calls them.
  - D.3.4 delegate's capture: None stated for the compile_module chain beyond the clause: if a lowering captures from
    the delegate, its MTLCommandBuffer is used only during lower() and no function the Program calls passes through
    it.
  - D.3.6: As the clause states it: rdma/mesh-flow.c compares hdr `map_serial` with the static schedule's serial where
    mesh-records.c map_put moves the serial, and calls plan_cancel for each schedule that differs. Whether
    aot-compilation A7's view rule replaces this is for the documents to say; this one does not.
  - D.4 mesh: rdma/mesh-flow.c written: Same as D.3.6.
  - D.4 metal-microbench deleted (2): per-step instances: Same as D.3.3 for the compile_module chain.

Covered:

  - A3 "What it takes": covered [2026-11-04, mesh baede218]
  - A3 "What it writes": covered [2026-11-10, mesh 5889d898]
  - A3 "The bridge's record pages": covered [2026-11-06, mesh fa089c56]
  - A4 deleted (1): covered [2026-11-03, mesh ae0bc2f1]
  - A4 deleted (2): covered [2026-11-03, mesh fd49e3b5]
  - A4 deleted (3): covered [2026-11-07, mesh dfabef5a]
  - A4 deleted (4): covered [2026-11-07, mesh 2e83f943]
  - A4 deleted (5): covered [2026-11-05, mesh c6c9c77f]
  - C3.1 algorithm 4, bridge half: covered [2026-11-04, mesh 04b5e561]
  - C3.2 (a) data structure and types: covered [2026-11-10, mesh 5889d898]
  - C3.2 (c) invariants placed once: covered [2026-11-10, mesh 5889d898]
  - C3.3 algorithm 1 (L): covered [2026-11-04, mesh baede218]
  - C3.3 algorithm 2 (u): covered [2026-11-04, mesh baede218]
  - C3.3 algorithm 3 (copies, t mod u): covered [2026-11-04, mesh baede218]
  - C3.3 compile-time bound: covered [2026-11-10, mesh 5889d898]
  - C3.4 (a) no claim: covered [2026-11-05, mesh c6c9c77f]
  - C3.4 (c) deferred reclamation: covered [2026-11-08, mesh b880b808]
  - C3.4 "Written by, read by": covered [2026-11-08, mesh b880b808]
  - C3.6 (2) retired client's schedule: covered [2026-11-05, mesh c6c9c77f]
  - C3.6 (4) link loss and poll bound: covered [2026-11-05, mesh 876b1d69]
  - C3.7 data structures: covered [2026-11-08, mesh 7517cfc8]
  - C3.7 algorithm 1: covered [2026-11-04, mesh 04b5e561]
  - C3.7 algorithm 2: covered [2026-11-04, mesh baede218]
  - C3.7 algorithm 3: covered [2026-11-10, mesh 5889d898]
  - C3.7 algorithm 4: covered [2026-11-04, mesh 04b5e561]
  - C3.7 algorithm 5: covered [2026-11-05, mesh 876b1d69]
  - C3.7 "Written by, read by": covered [2026-11-10, mesh 5889d898]
  - C3.7 data pages: covered [2026-11-04, mesh baede218]
  - C3.7 projection kernels: covered [2026-11-09, metal-microbench f08d3c7b]
  - C4 mesh deleted: rdma/mesh-collective.c (executions, per-execution items): covered [2026-11-04, mesh baede218]
  - C4 mesh deleted: rdma/mesh-collective.c (memset and word zeroing): covered [2026-11-04, mesh baede218]
  - C4 mesh deleted: mesh_plan_erased: covered [2026-11-04, mesh baede218]
  - C4 mesh deleted: rdma/mesh.h areas and constants: covered [2026-11-08, mesh 7517cfc8]
  - C4 mesh deleted: rdma/mesh-dataflow.c claim: covered [2026-11-04, mesh baede218]
  - C4 metal-microbench deleted: metal_recording.m: covered [2026-11-09, metal-microbench f08d3c7b]
  - C4 metal-microbench deleted: tools/e2b_replay.m: covered [2026-11-05, metal-microbench d81575db]
  - C4 metal-microbench written: tools/prepare_e2b.py: covered [2026-11-10, metal-microbench aa685a47]
  - D.3.4 data structure: covered [2026-11-12, mesh a86c7dc4]
  - D.3.5: covered [2026-11-05, mesh 930861b0]
  - D.4 mesh: rdma/mesh-flow.c deleted: covered [2026-11-05, mesh 930861b0]
  - D.4 mesh: coordination document: covered [2026-11-04, mesh a1d034b4]
  - D.4 metal-microbench deleted (1): classic command queues and buffers: covered [2026-11-10, metal-microbench e1857e71]
  - D.4 metal-microbench deleted (3): delegate buffers as captured: covered [2026-11-09, metal-microbench f08d3c7b]

Withdrawn:

  - D.3.1 (and the turn's item 1: several programs, MESH_GROUP queue-pair assignment, EBUSY): withdrawn [2026-11-12
    reading]. design/disjoint-pages-change-request.md:1136-1137 withdraws the turn's item 1; the document contains no
    D.2, D.3.1 or D.5 section; design/plan-to-bridge-change-request.md:8-9 states it replaces "D.2's model of a
    transfer, D.3.1 and D.5".

## [page-queue-change-request.md](page-queue-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 14 items outstanding, 24 covered, 0 withdrawn.

Outstanding:

  - The operator's statements: Relays: The entries Program publishes for a cross-machine channel
    (compiler/mesh_compiler/bridge.py, called from runtime.Program.__init__) pass through intermediate nodes where the
    two machines are not adjacent in the map.
  - The operator's statements: Interleaving every transfer: See 'The schedule that writes the page lists: the
    algorithm'.
  - The operator's statements: The E2B code, the bridge and two layers: The E2B program as a module compiled by
    compile_module and run by Program on the collective's machines, its cross-machine channels carried by the bridge
    (compiler/mesh_compiler/bridge.py).
  - The operator's statements: On how parity is shown: The E2B decode compiled by compile_module and run by Program on
    the collective, end to end.
  - Claude's proposal: The page lists: Produced by: The page lists of a Scheduled's cross-machine channels
    (Scheduled.carried: channel, sender, receiver, dtype, shape, stride) are produced by the collective's function in
    rdma/mesh-collective.{c,h}; its caller is compiler/mesh_compiler/bridge.py channels, called by runtime.Program.
  - Claude's proposal: The schedule that writes the page lists: the algorithm: The concurrent-flow solve
    (rdma/mesh-graph.h mesh_graph_flow) is run ahead of time over every cross-machine transfer of a Scheduled
    together; in the chain that is the producer of the page lists that bridge.channels calls.
  - Claude's proposal: The schedule: blocking transfers first: The producer of the page lists puts the transfers that
    block a pipeline stage first.
  - Claude's proposal: The schedule: It writes: The producer of a Scheduled's page lists writes, on each node a page
    passes through, the source's sends, each intermediate node's receive and send, and the destination's receive.
  - Claude's proposal: The schedule: every live node runs a rank: Every live node of the collective runs the Program,
    so an intermediate node's entries are written on that node.
  - Claude's proposal: The assignment: Its step: In rdma/mesh-flow.c link_prepare, the rate operand passed to
    mesh_graph_list_schedule for a (cable, queue pair) is the cable's bandwidth, as the clause states.
  - Claude's proposal: The changes, by file: The collective: one function: compiler/mesh_compiler/bridge.py channels
    (called by runtime.Program) passes a Scheduled's declared transfers to the one function of
    rdma/mesh-collective.{c,h} that writes the page lists.
  - Claude's proposal: The changes, by file: The callers: tools/e2b_replay.m: The caller is
    compiler/mesh_compiler/bridge.py channels, with Scheduled.carried as the declared receives and sends.
  - Claude's proposal: Order, bullet 3: The E2B decode compiled by compile_module and run by Program on the
    collective, end to end.
  - Claude's proposal: Order, bullet 4: The waits of compiler/mesh_compiler/bridge.py (publish, Carried.get,
    Carried.settle) and of rdma/mesh-metal4.m mesh_metal4_wait are bounded, each failure an error value.

Covered:

  - The operator's statements: Collective size: covered [2026-11-08, mesh b880b808]
  - The operator's statements: Queue pairs and links belong to the bridge: covered [2026-11-03, mesh 1444f409]
  - The operator's statements: The page queue: covered [2026-11-06, mesh 4fe54e5f]
  - The operator's statements: Replacing the --link form: covered [2026-11-06, mesh 1acf8b31]
  - The operator's statements: The bridge's process lifecycle: covered [2026-11-03, mesh 66beb887]
  - The operator's statements: Failing by error, not by hardware fault: covered [2026-11-08, mesh e6eda244]
  - Claude's proposal: The page lists (the queue): the data structure: covered [2026-11-08, mesh 7517cfc8]
  - Claude's proposal: The page lists: page, node, direction: covered [2026-11-12, read at mesh 9aa74ed1]
  - Claude's proposal: The page lists: word, position: covered [2026-11-12, read at mesh 9aa74ed1]
  - Claude's proposal: The page lists: no copy, not a ring: covered [2026-11-12, read at mesh 9aa74ed1]
  - Claude's proposal: The page lists: Consumed by: covered [2026-11-12, read at mesh 9aa74ed1]
  - Claude's proposal: The page lists: Replaces: covered [2026-11-03, mesh 4ac5cbdc]
  - Claude's proposal: The schedule: It takes: covered [2026-11-10, mesh 60103676]
  - Claude's proposal: The assignment of pages to cables and queue pairs: the algorithm: covered [2026-11-03,
    mesh c193b11a]
  - Claude's proposal: The assignment: It takes: covered [2026-11-06, mesh 33a03e35]
  - Claude's proposal: The assignment: It writes: covered [2026-11-12, read at mesh 9aa74ed1]
  - Claude's proposal: The changes, by file: The bridge: covered [2026-11-06, mesh 4fe54e5f]
  - Claude's proposal: The changes, by file: The bridge: process lifecycle: covered [2026-11-03, mesh 66beb887]
  - Claude's proposal: The changes, by file: The bridge: queue-pair count: covered [2026-11-08, mesh e6eda244]
  - Claude's proposal: The changes, by file: The collective: replaced calls: covered [2026-11-03, mesh 1444f409]
  - Claude's proposal: The changes, by file: The callers: tools/mesh/run_pair.py: covered [2026-11-03, metal-microbench dc454f50]
  - Claude's proposal: The changes, by file: The callers: tools/mesh/grid.py: covered [2026-11-03, metal-microbench 9bb797c0]
  - Claude's proposal: Order, bullet 1: covered [2026-11-12, read at mesh 9aa74ed1]
  - Claude's proposal: Order, bullet 2: covered [2026-11-12, read at mesh 9aa74ed1]

## [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 31 items outstanding, 35 covered, 0 withdrawn.

Outstanding:

  - 1, step 1: placement.place (reshard's branch for held ≠ dimension): the recv node of block (i→j) names as its
    destination the tensor view (offset, sizes, strides) of block i in j's result; the schedule reads the send and
    recv nodes as nodes of the graph; the pair of views is the `from` and `to` of one struct mesh_declared
    (rdma/mesh-collective.h:37).
  - 1, step 2: placement.place writes one aten.copy_ node at member i from the tensor view of block i of the operand
    to the tensor view of block i of the result; the schedule reads it as a node of the graph.
  - 1, step 3: placement.place, where it writes step 2's copy_ node: the node is not written where the operand's
    tensor view is the result's tensor view with the same sizes and strides.
  - 2, step 1: placement.place (reshard to replicated, dimension None): where the producer node of member i's input is
    an out variant, its `out` argument is the tensor view of block i of the result; the schedule reads it as a node of
    the graph.
  - 2, step 2: placement.place writes one aten.copy_ node from member i's input to the tensor view of block i of the
    result where step 1 does not apply.
  - 3, Change: the rule: placement.place: a send or recv whose value is a tensor view node carries its base value with
    the view's offset, sizes and strides as torch computes them; that is the `from` or `to` (struct mesh_strided,
    rdma/mesh-graph.h:162) of the transfer's struct mesh_declared, which mesh_plan reads.
  - 3, Change: the result torch allocates: With the rule above: a transfer's end whose value torch allocates outside
    the memory plan's pages is item 2 step 2's copy_ node into a value that has an entry.
  - 4, step 1: placement.place: for a windowed operator whose operand is replicated, each processor's operand is an
    aten.slice node (a tensor view) of the local operand over its window; no send or recv is written.
  - 4, step 2: placement.place: for each sharded spatial dimension of a windowed operator's operand, the send and recv
    nodes of the halo exchange of the maximum left and right halo sizes (item 5's CollectivePermute), then the slice
    node of the partition's valid region, then the masking nodes.
  - 4, step 3: placement.place: step 2's exchanges are written for the sharded dimensions in increasing order, each
    taking the previous exchange's result node as its operand; a dimension with one shard gets none.
  - 4, step 4: placement.place: the producer node's `out` is the tensor view of the partition's own shard in the
    window, or one copy_ node into that view (item 2's two steps).
  - 5, step 1: placement.place: for a CollectivePermute's pairs, a send node at p and a recv node at q for each (p, q)
    with p ≠ q, in the exported graph.
  - 5, step 2: placement.place: for a pair (p, p), item 1 step 2's copy_ node at p.
  - 5, step 3: placement.place: at a replica that is the target of no pair, a zeros node of the operand's shape is the
    CollectivePermute's result.
  - 6, step 1: placement.place: for each tensor view node, torch's condition on the operand's strides and sizes is
    tested for the new view's dimensions.
  - 6, step 2: placement.place: where step 1's condition holds the node stays a tensor view node.
  - 6, step 3: placement.place: where step 1's condition fails, the node is a copy node whose `out` is a value of the
    memory plan; the schedule reads it as a node of the graph and mesh_plan gives the `out` value its pages.
  - 7.1, step 1: A function of ir.Scheduled: for each processor's GraphModule in scheduled order, each value's pair
    [index of the first node referring to it, index of the last], as update_tensor_lifetime computes it; the pairs are
    the `begin` and `end` of each struct mesh_value in mesh_plan's operand. The document names no stage for this
    function.
  - 7.1, step 2: In step 1's function: the lifetime of a send's source value ends at the index of the wait node of
    7.2, and a recv's destination value begins at the index of the node that starts the receive.
  - 7.2, step 1: placement.place writes one wait_tensor node per collective at each processor, before the first node
    there that uses the collective's result.
  - 7.2, step 2: Step 1's wait_tensor node at a processor completes that processor's sends, forwards and receives of
    the collective.
  - 7.2, step 3: 7.1's function reads the index of step 1's wait_tensor node as the end of the lifetimes of the send's
    source value and of a forwarded block.
  - 7.3, step 3: runtime: the call of 7.2's wait_tensor node reads the collective's receive counts (mesh_host_arrived)
    and send-completion counters (mesh_host_sent) at that processor until they reach the execution's totals.
  - 8, step 2: placement.place: a node made from state placeholders by tensor views alone has the replicated sharding
    (held by every processor), as a placeholder has at placement.py:359.
  - 8, step 3: placement.place (reshard): with step 2's sharding, a reshard of a view of state is an aten.slice node
    at each processor and no send has such a value as its argument.
  - 8, step 4: placement.place writes, for each distinct slice a processor's graph reads of a state placeholder or of
    a view of one, one never-freed value in mesh_plan's operand: a struct mesh_value of kind MESH_INVARIANT
    (rdma/mesh-collective.h:21-25); mesh_plan gives it its pages.
  - 8, step 5: runtime.Program.__init__: for each of step 4's values at a processor, the slice of the state tensor (or
    of that view of it) by the partition dimension and the processor's rank is copied once into the value's pages by
    copy_, before the first call.
  - 8, step 6: runtime.Program.__init__ (with step 5): a processor is given the whole state tensor only where a node
    of its graph reads the placeholder whole.
  - 9: None stated: the clause determines no change and names no source for what mesh_plan does with such a pair.
  - 10, Composition: the placement stage writes it: placement.place writes, for each send's and recv's end, the view
    as dims and dims (size, stride) pairs with its offset (struct mesh_strided, rdma/mesh-graph.h:162), the `from` and
    `to` of the transfer's struct mesh_declared in mesh_plan's operand.
  - 11, step 4: runtime.Program / bridge.channels: on the next membership view the cancelled persistent request's
    replacement is written (mesh_plan, or bridge.publish for the chain's channels) from the same Scheduled, with that
    view's identifier as its serial.

Covered:

  - 1, Change: the refusal is deleted: covered [2026-11-06, mesh 30afa696]
  - 2, Change: both refusals are deleted: covered [2026-11-06, mesh 30afa696]
  - 4, Change: the refusals are deleted: covered [2026-11-06, mesh 30afa696]
  - 5, Change: the refusal is deleted: covered [2026-11-06, mesh 30afa696]
  - 7 (Code): covered [2026-11-05, mesh c6c9c77f]
  - 7.1, step 3: covered [2026-11-08, mesh b880b808]
  - 7.1, step 4: covered [2026-11-06, mesh 33cb11c2]
  - 7.1, step 5: covered [2026-11-05, mesh c6c9c77f]
  - 7.2, Change: PLAN_SENT is deleted: covered [2026-11-06, mesh 6a8a8524]
  - 7.3, step 1: covered [2026-11-06, mesh 6a8a8524]
  - 7.3, step 2: covered [2026-11-06, mesh 6a8a8524]
  - 7.3, step 4: covered [2026-11-06, mesh 8624d557]
  - 7.3, step 5: covered [2026-11-06, mesh 8624d557]
  - 7.3, step 6: covered [2026-11-06, mesh 8624d557]
  - 7.3, step 7: covered [2026-11-11, mesh 1a3cfe60]
  - 8, step 1: covered [2026-11-11, mesh 297c3cda]
  - 8, Change: LOADED and the execution UINT64_MAX − 1 are deleted: covered [2026-11-06, mesh 2155c2b8]
  - 10, Change: the representation: covered [2026-11-06, mesh 496ad4ef]
  - 10, Change: the refusal is deleted: covered [2026-11-06, mesh 00f31f7f]
  - 11, step 1: covered [2026-11-06, mesh 028b27d6]
  - 11, step 2: covered [2026-11-06, mesh 028b27d6]
  - 11, step 3: covered [2026-11-06, mesh 028b27d6]
  - 11, step 5: covered [2026-11-06, mesh 9bdaee8d]
  - 12, step 1: covered [2026-11-06, mesh 61435d4b]
  - 12, step 2: covered [2026-11-06, mesh 61435d4b]
  - 13, step 1: covered [2026-11-03, mesh 53e3935c]
  - 13, step 2: covered [2026-11-03, mesh 53e3935c]
  - 13, step 3: covered [2026-11-06, mesh 4e4b1938]
  - 14, step 1: covered [2026-11-06, mesh 33cb11c2]
  - 14, step 2: covered [2026-11-06, mesh 33cb11c2]
  - 14, step 3: covered [2026-11-06, mesh 33cb11c2]
  - 14, step 4: covered [2026-11-08, mesh b880b808]
  - 14, step 5: covered [2026-11-06, mesh 33cb11c2]
  - 14, step 6: covered [2026-11-06, mesh 30e6a6e7]
  - 14, Change: replaces plan_before and mesh_graph_chains for these items: covered [2026-11-07, mesh 8c8094da]

## [unequal-shards-change-request.md](unequal-shards-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 18 items outstanding, 28 covered, 1 withdrawn.

Outstanding:

  - 4 - Data structure: `place` (compiler/mesh_compiler/placement.py) gives the value a device holds along a sharded
    dimension one size for all devices, rounded up, and holds for it the device's own size of item 1, from which the
    mask is the comparison of the index along the dimension with the device's offset.
  - 4 - Algorithm / Stage: `place` adds, on each device, before an operator that reduces over the sharded dimension, a
    select node whose mask is computed from that device's size of item 1 and which replaces the elements past the
    device's shard by the reduction's identity. Operands: the placed graph, the device's size; result: the placed
    graph with the select nodes.
  - 7 - Data structure: The value item 6's solve is given for a processor and a kind of function is a speed as a
    function of the number of units, s(x) = x / t(x) from measured times t(x); `shares.computation`
    (compiler/mesh_compiler/shares.py:34) reads it where it now reads `time(node)`.
  - 7 - Stage: Before `place`, the times of the functions of processors on every machine of `processors` are taken and
    given to item 6's solve (`shares.shares(staged, processors, time, comm, upper)`,
    compiler/mesh_compiler/shares.py:41); operands: the fixed program's `Scheduled`; result: a time for every function
    node of every processor.
  - 8 - Algorithm (a): tokens permuted before the first embedding: `place` consumes a striped sharding stated on the
    token-id input of a program and carries it through the first embedding: the embedding's rule in `partitioned`
    (compiler/mesh_compiler/placement.py RULES), operands the ids' shards and the table, result the embedded shards in
    the same stripes.
  - 8 - Algorithm (b): the layout kept backward: `place` keeps the stripes through the operators of an exported
    backward pass: their rules in `partitioned` (compiler/mesh_compiler/placement.py RULES), operands the shards in
    stripes, results in the same stripes.
  - 8 - Algorithm (c): position ids and target ids: `place` shards the position-id and target-id inputs of a program
    in the stripes stated for the sequence and carries them through the operators that read them (the embedding of
    positions, the loss's read of targets): rules in `partitioned`, operands the ids' shards, results in the same
    stripes.
  - 10 - Data structure: `place` computes, for a windowed operator whose input is sharded by item 1 along the window's
    dimension, each device's halo size and the maximum over the devices.
  - 10 - Algorithm / Stage: `place` adds, for that operator, the `send` and `recv` nodes of a collective permute that
    carries each neighbour's halo at the largest halo size, and on each device a slice node that removes what exceeds
    the device's own halo; operands: the placed graph and the halo sizes; result: the placed graph.
  - 11 - Data structure: The form of a tensor whose rows differ in length, as an operand of item 13's partitioning:
    two tensors, `values` and nrows+1 nondecreasing `offsets`; a second ragged dimension a second array of offsets
    over the first.
  - 11 - Stage: Item 13's partitioning takes the offsets as its operand; nothing further is stated.
  - 12 - Data structure: For each device of a collective over a dimension held in item 4's buffer: the buffer's sizes,
    the shard's sizes and the shard's starts, one integer per dimension of the buffer, computed by `place` from item
    1's sizes.
  - 12 - Algorithm (subarray sent): The `send` and `recv` nodes `place` adds for a collective of item 2 over item 4's
    buffer carry the three arrays and the value sent is the subarray.
  - 12 - Algorithm (the all-gather's method): For an all-gather of unevenly sized shards: the two methods (shards
    padded to one size; each shard sent separately) and the choice between them by their times at the sharding ratios,
    made again whenever item 6's ratios are solved again. The document states that no source names the function that
    chooses or the type that records the choice.
  - 12 - Stage: `insert_send_recv`/`reshard` (compiler/mesh_compiler/placement.py) write on the `send` and `recv`
    nodes the three arrays of the subarray, one integer per dimension of the buffer.
  - 13 - Data structure: The array of P+1 separators as the sharding of the row dimension; each device's shard of the
    ragged tensor is the values of its rows concatenated and the lengths of its rows.
  - 13 - Algorithm: A function run before `place`: operands item 11's offsets and the devices' speeds in their given
    order; result the P+1 separators that minimise the largest load, by the probe (binary search in the offsets) and
    Nicol's search.
  - 13 - Stage: The separators written by item 13's function are carried in the placement operand as the sharding of
    the row dimension, and `place` (compiler/mesh_compiler/placement.py) consumes them.

Covered:

  - 1 - Data structure: covered [2026-11-11, mesh 297c3cda]
  - 1 - Algorithm: covered [2026-11-11, mesh 297c3cda]
  - 1 - Stage: covered [2026-11-11, mesh 297c3cda]
  - 2 - Data structure: covered [2026-11-11, mesh d3eecd18]
  - 2 - Algorithm: covered [2026-11-11, mesh d3eecd18]
  - 2 - Stage: covered [2026-11-11, mesh d3eecd18]
  - 3 - Data structure / Algorithm: covered [2026-11-11, mesh d3eecd18]
  - 3 - Stage: covered [2026-11-11, mesh d3eecd18]
  - 5 - Data structure: covered [2026-11-11, mesh 297c3cda]
  - 5 - Algorithm: covered [2026-11-11, mesh 297c3cda]
  - 5 - Stage: covered [2026-11-11, mesh 297c3cda]
  - 6 - Data structure: covered [2026-11-11, mesh 1b2faa3f]
  - 6 - Algorithm: covered [2026-11-10, mesh 4a9e1cf4]
  - 6 - Stage: covered [2026-11-11, mesh 1b2faa3f]
  - 7 - Algorithm: covered [2026-11-11, mesh 1b2faa3f]
  - 8 - Data structure: covered [2026-11-11, mesh d3eecd18]
  - 8 - Algorithm (b): the layout kept forward: covered [2026-11-11, mesh d3eecd18]
  - 8 - Algorithm (d): the mask in original order: covered [2026-11-11, mesh d3eecd18]
  - 8 - Algorithm (e): a block the mask inhibits entirely: covered [2026-11-12, mesh 90843d93]
  - 8 - Stage: covered [2026-11-11, mesh d3eecd18]
  - 9 - Data structure / Algorithm: covered [2026-11-11, mesh d3eecd18]
  - 9 - Stage: covered [2026-11-11, mesh d3eecd18]
  - 14 - Data structure: covered [2026-11-10, mesh 4a9e1cf4]
  - 14 - Algorithm (the bound in the linear program): covered [2026-11-10, mesh 4a9e1cf4]
  - 14 - Algorithm (Whale's Algorithm 1): covered [2026-11-12, mesh 991742c8]
  - 14 - Stage: covered [2026-11-11, mesh 1b2faa3f]
  - 15 - Algorithm: covered [2026-11-11, mesh 297c3cda]
  - 15 - Stage: covered [2026-11-11, mesh 297c3cda]

Withdrawn:

  - Dropped - heterogeneity.md R11: withdrawn [2026-11-12 reading]. design/unequal-shards-change-request.md:493-498
    (section "Dropped").

## [integration-finishing-change-request.md](integration-finishing-change-request.md)

Outstanding [2026-11-12, mesh 9aa74ed1, metal-microbench 0b692ed8]: 15 items outstanding, 6 covered, 0 withdrawn.

Outstanding:

  - 1 (Write, bullet 1: diagnose and correct each machine's bridge configuration): not determined [2026-11-12]. The
    clause is a state of five machines, not of code; the reading contacted no machine.
  - 1 (Write, bullet 2: the mini's configuration.plist): not determined [2026-11-12]. Machine state; the reading
    contacted no machine.
  - 1 (Write, bullet 3: start every bridge): not determined [2026-11-12]. A state of running services on five
    machines; this audit starts no process; the reading contacted no machine.
  - 1 (Write, bullet 4: e2b on all five machines to its end): E2B as a module given to compile_module(module, args,
    processors, placement, cost) with processors on the five machines, run by Program; the run's record states each
    rank's decoded tokens.
  - 2 (The algorithm): schedule.schedule(module, cost) (compiler/mesh_compiler/schedule.py), called by
    driver.compile_module, orders a processor's nodes by list scheduling in the order of the linear program's
    completion times, through rdma/mesh_graph.py.
  - 2 (It takes): The operands of the ordering in schedule.schedule: a key per node (a recv's node: the published
    completion time of its range; every other node: its dependence level) and the nodes' dependences.
  - 2 (It writes): schedule.schedule writes each processor's node list as mesh_graph.heap_order(priority, edges) over
    the keys above.
  - 2 (Consumed by): Program (runtime.py) and lower (lower_metal.py, lower_coreml.py) already execute each processor's
    nodes in Scheduled.modules' order; that order is to be the list of the entry above.
  - 3 (The data structure): The `cost` operand of compile_module/schedule.schedule carries a weight on each task and a
    weight on each edge, and schedule is given the cables' capacities from the map.
  - 3 (Consumed by): The stage of compile_module that routes cross-machine channels takes the weighting operand and
    the map's capacities and computes no weight; schedule already takes cost.
  - 4 (Replaces): See the next entry; once the walk is a graph-module function, the metal-microbench copy goes with
    the old code.
  - 4 (Write): A function of rdma/mesh-graph.{h,c} (and rdma/mesh_graph.py for a Python caller) that is the masked
    pass's walk, its operands the caller's; its caller in the compile_module chain is the lowering that emits that
    pass (lower_metal.lower).
  - 7 (The data structure: that commit's record): rdma/mesh-metal4.m mesh_metal4_commit's handler has a record for its
    own commit holding the feedback's error and GPU start and end times; the caller is lower_metal.py's function.
  - 7 (Produced by): As the entry above: each commit's handler in mesh_metal4_commit writes the feedback data to its
    own commit's record.
  - 7 (Consumed by): The function of rdma/mesh-metal4.m that lower_metal.py's WAIT entry calls (mesh_metal4_wait)
    receives one handler's call for each commit.

Covered:

  - 3 (Produced by): covered [2026-11-12, mesh 52b2f91d]
  - 3 (Replaces): covered [2026-11-08, mesh 01632014]
  - 5: covered [2026-11-08, metal-microbench 5d5efc09]
  - 7 (The data structure: one MTL4CommitOptions a commit): covered [2026-11-12, mesh a86c7dc4]
  - 7 (Replaces, e2b_replay.m feedback_start): covered [2026-11-10, metal-microbench e1857e71]
  - 7 (Replaces, metal_recording.m MetalRunOf): covered [2026-11-08, metal-microbench 33be7703]
