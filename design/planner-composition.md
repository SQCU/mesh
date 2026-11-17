# Planner composition work record — 16 November 2026

## Authority and scope

The operator directed this revision after source review of `analyze_local` and
`analyze_program`:

> shuck out all of the pure functions inlined into or defined inside of each of these deviant function bodies, delete the major funciton bodies, and recompose whatever we're supposed to be doing

> all graph algorithms should end up in an imported header, to avoid idiomatic plural multiple data structural representations of graphs

The [composition request](compilation-composition-change-request.md),
[compiler pseudocode](compiler-pseudocode.md),
[functional compilation memorandum](functional-compilation-memorandum.md), and
[access-by-view memorandum](access-by-view-memorandum.md) supply the existing
data-flow contract. The [schedule-emit driver](../../../schedule-emit/schedule_emit/driver.py)
demonstrates composition of functions with explicit inputs and returned owners.
This record assigns the reviewed planner replacement; it adds implementation
coverage to that contract.

## Source findings and complete replacement units

The review examined Mesh `02be0bb4`, `rdma/mesh-collective.c`, and the original
algorithms in `rdma/mesh-graph.h` and `rdma/mesh-graph.c`. The line references
below identify the reviewed revision; subsequent source defines current behavior.

| Reviewed computation | Producer and result | Direct consumers |
|---|---|---|
| `plan_analysis_new` groups calls; `plan_analyze_network` groups them again; `plan_graph` rebuilds ordinals twice | The graph module groups the call array once and returns row offsets, call indices, and ordinals | Local analysis, network analysis, call graph construction |
| Local completion and program analysis independently classify relation edges; local completion and `plan_graph` enumerate the same chain pairs | The graph module supplies relation-edge classification and chain edges from the original relation and chain operands | Local output and processor-edge expansion |
| `plan_trees`, `plan_template`, `plan_walk`, and `plan_times` reconstruct child rows from selected parent/root trees | The graph module constructs each selected tree's adjacency once and retains it with the routing result | Membership, template, transfer entries, transfer costs |
| `plan_graph` repeatedly materializes one graph for lag queries and supplies duplicate source vertices for nonlocal calls | The graph module prepares the graph once and computes the requested distinct-source queries | Queue capacity and synchronization consumers |
| `plan_horizon`, `plan_unroll`, and `plan_schedule` retain summaries and free full solved arrays | Each producer retains its complete bounds, lag matrix, completion times, order, and source correspondence | Copy expansion, allocation, scheduling, archives, later compilation rounds |
| `plan_sets` embeds interval union, membership deduplication, and equal-set grouping | The graph module performs these operations through its public header | Transport page-set construction |
| `plan_complete_network` solves graphs, reserves live pages, writes runtime records, and publishes a generation | Pure planning returns a complete compiled result; installation consumes that result and explicit live resources | Separate compiler and installation entrypoints |

The source anchors are `mesh-collective.c` lines 191, 344, 765, 797, 1007,
1072, 1232, 1352, 1361, 1453, 1666, 2267, 2317, 2488, 2606, and 2662;
`mesh-graph.c` lines 620 and 1046 contain the existing row and lag algorithms.
These observations concern repeated representation construction, repeated
queries, result retention, and responsibility assignment. Routing, buffer-bound,
allocation, and scheduling algorithms already have shared implementations.

## Composition before implementation

The following names describe calls to the original algorithms and their
extracted operations. The public C header supplies the concrete signatures.
Every argument representing an earlier result denotes a view of its complete
owner. Each returned result retains its predecessors and owns separate output
storage.

```text
analyze(functions, calls, relations, chains, roots, topology, storage, costs):
    rows = graph.group(calls)
    edges = graph.call_edges(relations, chains)
    routes = graph.route(functions, topology, costs)
    trees = graph.adjacency(routes)
    bounds = graph.buffer_bounds(functions, calls)
    expansion = graph.expansion(functions, edges, routes, trees, bounds)
    return Analysis(all_inputs, rows, edges, routes, trees, bounds, expansion)

compile(analysis, transformed_functions, transformed_demands, storage):
    graph = compose_graph(analysis, transformed_functions, transformed_demands)
    synchronization = graph_library.synchronization(graph)
    schedule = graph_library.schedule(graph, synchronization)
    allocation = graph_library.allocate(graph, schedule, storage)
    records = construct_records(analysis, graph, synchronization, schedule, allocation)
    return Compiled(analysis, graph, synchronization, schedule, allocation, records)

install(compiled, live_resources, deadline):
    reservation = reserve(live_resources, compiled.allocation, deadline)
    bindings = bind(compiled, reservation, live_resources)
    publication = publish(bindings)
    return Installed(compiled, reservation, bindings, publication)
```

The functions in the graph module implement graph and set operations. Planner
functions supply original operands and compose the returned results. The
installer performs live reservation, address binding, and publication. Callers
that request both compilation and installation invoke those functions in order.
Archive writers serialize retained results through their original layouts;
readers return views and relocation records with the archive owner retained.

Each implementation unit updates its direct consumers and ownership release
path together. The Python bindings mirror the C header and pass whole operand
bundles. Source review checks these interfaces against the complete inputs,
outputs, and predecessor ownership. Builds check the native declarations and
linkage; existing callers provide execution observations for their actual input
domains. The workgroup records observed coverage separately from this assignment.
