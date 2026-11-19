# Change request: the compiler supplies the bridge's transfer program

### R1. Transfers are page views

The compiler writes ordered transfer descriptors naming page views, ranges, dependencies and endpoints.
The bridge posts SEND/RECV requests over those registered views (Apple TN3205; MPI 4.1 persistent communication
§3.9; MSCCLang §§4–5). A source publication makes the corresponding produced range available for sending;
receive completion makes its destination range available to consumers.
The compiler flattens each endpoint's `(offset, sizes, strides)` view into `(page-table row, offset, bytes)` blocks,
merging adjacent blocks (MPI derived-datatype processing; Ross, Miller and Gropp, EuroPVM/MPI 2003;
PyTorch `c10/core/TensorImpl.h`). Transfers address the producer's and destination's own layouts. A required layout
conversion is a numerical operation whose output uses the consumer's declared layout.

### R2. Allocation binds every value to pages

The compiler allocates values through their lifetimes and dependency order (Lam 1988; Lamport 1978;
Dilworth/Fulkerson chain decomposition). Its page table binds logical sections to physical frames
(Kilburn et al. 1962; Denning 1970). Tensor views carry offset, dimensions, sizes and strides; consumers receive
the layout they require. Capture item 6 supplies Greedy by Size offsets for contiguous operands, and
collective-values item 14 supplies cyclic interval allocation. Lowering consumes the allocated views directly.
Logical rows share physical frames by page-table reference, as fbufs (Druschel and Peterson, SOSP 1993) and IO-Lite
(Pai, Druschel and Zwaenepoel, TOCS 2000) share buffers. The allocator combines shared-page lifetimes from their
first write through their last read. Disjoint byte ranges can occupy the same block. A contiguous virtual view
uses its prepared mapping over the allocated frames (`mach_vm_remap`; `design/page-maps-2026-11-02.md` at `6255e92b`).
The numerical operand's declared layout determines which mapping or contiguous arena it consumes.

### R3. An execution is a task graph

Placement produces the weighted task graph `G=(V,E,w,c)` of Sinnen 2007. Each vertex names a numerical or transfer
operation and its processor; each edge carries the value/access that causes its dependence.
Source numerical bodies and collective operations enter this same graph (Unity's parallel computation graph,
Unger et al., OSDI 2022). The graph module returns the derived ordering and source relation to the schedule.

### R4. Consumer properties determine graph edges and fusion

A blocking consumer has an edge from every required operand page; a streaming consumer has tasks for the pages
it can consume independently (Graefe, *Query evaluation techniques*, 1993; Chowdhury and Stoica, *Coflow*, 2012).
Streaming homomorphisms combine partial results with the cited associative/commutative operation (Gibbons,
*The third homomorphism theorem*, 1996; Meertens, *Algorithmics*, 1986).
The compiler carries those accesses through numerical prologue/epilogue and communication fusion (FLUX;
TileLink; CoCoNet, as cited in composition). The emitted producer publishes each completed range after its writes;
received-page masks and worklists drive the corresponding consumer work (capture item 3).

### R5. One solve orders computation and transfers

The scheduler creates one completion-time variable `C_j` for each task and each routed page hop, and `C_max`.
Instruction lengths come from AOT A3. The program imposes `C_j >= d_j`, `C_j >= C_i+d_j` on each dependence,
resource-capacity subset inequalities, and `C_max >= C_j` (Chen, *Scheduling coflows with precedence constraints*,
2022, §6; Hall, Schulz, Shmoys and Wein 1997). It lexicographically minimizes `C_max` then `sum(C_j)`
(Isermann, *Linear lexicographic optimization*, 1982). Queyranne's separation supplies violated resource rows;
HiGHS's dual revised simplex method solves the resulting linear programs (Huangfu and Hall 2018).
For each resource's instruction subset `S`, the row is
`sum(d_j * C_j) >= (sum(d_j)**2 + sum(d_j**2))/2`; a resource is a directed node-pair adjacency with aggregate capacity, or a node's device
(Hall et al., equation (5)). The second objective holds `C_max` at the first solve's optimum. The completion-time
solution supplies ordering priorities. Emission realizes that order by dependencies and actual data availability.

`mesh_graph_completion_times` produces completion priorities. `mesh_graph_heap_order` consumes them with the
dependence graph and returns a global topological order, with instruction identity resolving equal priorities
(MSCCLang §5.2). Each target and directed peer adjacency takes its subsequence. The retained result contains the input relation,
completion times, order, copy mappings and source identities for downstream consumers (composition item 6).
Ranks derive corresponding schedules from identical inputs through the same deterministic compilation
(Schneider, *Implementing fault-tolerant services using the state machine approach*, 1990).

### R6. Placement determines transfer endpoints

Placement writes explicit source and destination ranks and tensor views for every collective transfer. Routing
expands these into page hops through the supplied topology (GSPMD §4.5; Garg–Könemann 1998).
The resulting endpoint lists answer the sparse-data-exchange metadata problem described by Hoefler, Siebert and
Lumsdaine, PPoPP 2010, during compilation. Each descriptor names its adjacent peer, byte extents, producer publication and logical
completion. A relay uses the incoming range's completion to release that range on the selected next hop.
The bridge distributes eligible bytes and simultaneous independent transfers over the available cables on that hop.
It publishes logical receive completion after all bytes of the specified range have landed. Streaming and joining
consumers observe the ranges selected by R4. [Operator U59 and U71](operator-turns-2026-11-04-to-06.md) ·
[Bridge dealing and negotiated capacities](page-queue-coordination.md)

### R7. The bridge binds physical transport

The bridge reads device-reported cable and queue-pair resources, binds registered storage, and realizes the
compiled peer dependencies through bridge-private queue pairs and physical posting lists (Apple TN3205; `mesh-verbs.h`; MPI persistent
request binding). These objects remain owned by the bridge. The compiler passes page views and compiled ordering
across that boundary, as the operator's queue-pair ownership turns recorded at `cdd44406` specify.

### R8. Function calls own their storage and publication

A compiled function carries its pages, schedule and persistent requests. Submission uses the island's
fixed-sequencer order (Défago, Schiper and Urbán §7.1; Pathways §4.4). Each published function retains its own
identity and state across coexistence with other functions. Retirement releases its pages through deferred
reclamation after its outstanding users finish (McKenney and Slingwine 1998; Fraser 2004).
The same retained allocation results bind function parameters, numerical operands and transport values
(AOT A5/A8/A9; the operator's retained-result instruction).
