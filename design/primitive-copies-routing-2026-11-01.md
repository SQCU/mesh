# Primitive copies in the routing code, 2026-11-01

This is a read-only map of hand-written copies of primitives in the routing code. The rule is design/rubrics.md C9, "One
primitive, offered as a header". It was made at mesh ecdb4f2e. Its line numbers are that commit's.

## Method

- Eight Sonnet readers each read one slice of the routing code whole:
  - mesh-collective.c, in three slices;
  - nccl-mesh.c's forms, its steps and flow, its routing, and its schedule and plan;
  - mesh-routed.c, mesh-fabric.c, routing.py and plan_cycles.py, together.
- Each reader first read the catalog, mesh-graph.h and mesh_graph.py's definitions.
- Each reader grepped the whole repository for other copies of every primitive it found. It counted the copies and the
  layers each copy sits in.
- Each reader gave every function in its slice one verdict, so no function was skipped without a word.
- One Sonnet merger grouped the 119 findings into 43 primitives. It dropped none.
- The reports are kept whole in design/data/primitive-copies-routing-2026-11-01.json: every finding, every coverage
  verdict, every group and the merger's summary. Later work reads that file, not a retelling of it.
- The prose below is the merger's, in short sentences. Only its order of repair is ours: we added cost and the run
  records.

## Why there are quadratics

Every quadratic found has the same cause. A consumer needs the records of one key. It writes a scan of the whole table
for them inside a loop over keys, or over calls. The records were laid out by that key all along, or could be in one
pass. So the cost was never the problem's. It came from the problem's parameterization: no index by the key the work is
grouped by. This is C9 seen as cost. The missing primitive is copied inline as a linear scan, and the scan is nested.

These were found quadratic in the routing path. Each is calls, demands or trees times another of them.

| Where | What it scanned | Product | Status |
|---|---|---|---|
| route_tree_counts | every demand, every tree, at every call | calls x demands x trees | ecdb4f2e: calls x trees; a88732ac: the call's own trees |
| solved_steps, mesh_flow_trees | every tree of the group, and a calloc of every tree's parents, at every call | calls x trees | a88732ac: the call's own range (tree_first) |
| representative search, routing_prior | every demand against every demand | demands squared | 749c0257: a sort, then binary search |
| flow_solve | every occurrence of a repeated call was its own commodity | rounds x occurrences | 1945152f: equal demands are one commodity |
| collective_views.lower | a gather along h's units cut at every storage break | 16384 calls a tick, then every plan stage over them | 2a13732c: frames (Call outer), 16 calls a tick |
| NativeGraph.compose_reads | every kept operation's reads, at every removal | operations squared | e16e5f1e: mesh_graph.retained, one pass |
| mesh_solve, mesh_trees_pack (group 18) | every tree found so far, for an equal one, per demand per round | rounds x demands x trees | open |
| routing_prepare columns and expansion (group 9) | every solved or catalog tree, per demand | demands x trees | open |
| route_tree_basis_equal, route_path_basis_equal (group 19) | every tree or path of both routes, per demand or scope | demands x trees | open |
| routing_prior's bank test (group 43) | every bank-0 entry, per matched demand | demands x catalog | open |
| link pair lookups (group 5) | every link, per pair | pairs x links | open; links and nodes are few |
| children by scan (group 30) | every vertex, per vertex | nodes squared a tree | open; nodes are few |
| Dijkstra in dense form (group 17) | every vertex, per settled vertex | nodes squared a solve | open; dense is right at five nodes, but it is three copies |

Every open row has the same repair: one index built once by the key, from the graph module. Group 10 names the missing
builder, "build row offsets from counts". It has 17 hand copies. mesh_graph_row reads such rows, but no catalog function
builds them.

## Order of repair

1. Group 10's row builder goes into mesh-graph.h: records keyed by a number, grouped into compressed rows by one count,
   one prefix sum and one scatter. The consumer passes its records and key as operands. Then groups 9, 18 and 19
   become row reads, starting with routing_prepare, routing_prior, the two basis comparisons, mesh_solve and
   mesh_trees_pack.
2. Group 1's comparator: one key-array compare. It has 39 copies.
3. Group 2's run walker: sort, then visit each run of equal keys. It has 16 copies.
4. Group 11's lower bound. It has 14 copies, one of them mesh_graph_through.
5. Group 12's bit set test. It has 11 copies, two of them named functions (mesh_member, rank_member).
6. Groups 17, 36 and 37: the flow solve, its Dijkstra and its round count move into the graph module as one solver.
   Its consumers pass demands and capacities as operands. Equal commodities are merged inside it (1945152f did this at
   the call site).
7. The copy and free groups (4, 7, 8, 20, 27, 34, 38, 39): one owned-array form for the structs of arrays.

## What the plan cost, run by run

These are the 64-step LSTM on all five members (examples/lstm-service.py, lstm-k1-s64.json). The figures come from
rank 0's preparation record (nccl-mesh.c prepare_record, read with extras/prepare_report.py).

| Run | Commit | Where it stopped | Cost |
|---|---|---|---|
| d-r3 | ecdb4f2e | plan_calls, silent at 60 s | 45% route_tree_counts, 40% write_steps, 13% tree copies |
| d-r5 | a88732ac | definition, silent at 60 s | plan_calls 0.09 s a group; definition 19 s a group over 16384 calls (order_transfers 8 s) |
| d-r6 | 2a13732c | page_effects, KeyError tmp7 | native preparation 2.4 s whole; 16 calls a gather group |
| d-r7 | e16e5f1e | realization, silent at 60 s | native preparation under 1 s; 97% of samples in MeshIndexCompose |
| d-r10 | 2430d610 | realization, silent at 45 s | 144 page compositions, 40 s; one shader read's pair composed 5 times, 4.5 s each |
| d-r11 | 28971add | page-effects solve, silent at 45 s | 47 compositions, 13.4 s; then MetalPageEffectsSolve, under PageObservationGroups |
| d-r12 | d51ddcec | page-effects solve, silent at 45 s | the observation walk ended early, no change: the cost is PageGroups' MeshIndexSetRuns |
| d-r23 | 7f7cef0b | completed, all five ranks, 8 steps | page compositions 0.03 s a rank in all; 121 ms a step; preparation 3.4 s |
| d-r30 | 5695af53 | completed, 8 steps × 5 runs | 78–109 ms a warm step |
| d-r31 | 5695af53 | completed, 8 steps, checked | largest error 6.1e-5 against the CPU wavefront reference |
| d-r25, d-r26 | 40309819, 96737e42 | 64 steps, silent at 60 s | the page solve's dense domain × command × group tables (PageSlots) and their release |

What is left, in order:
1. GPU timestamps per dispatch in the recorder, to say which dispatch or which peer holds each 40–100 ms exchange.
2. One strided collective a tick in place of 16 one-row fragments (`Call.outer` taken by the native all-gather), and
   routes over more of rank 0's six links.
3. Preparation solved for one replay and bound for every occurrence, not over the unrolled program; that also removes
   the dense domain × command tables, which grow as the square of the steps.
4. The chain service and the trainable load-balancing LSTM on main.

## The addressing artifact in page relations

The slow composition's physical relation (d-r10, rank 0, a shader's read of its weights) maps a logical byte i to a
page byte p by

    p = 465960960 + i + 4094 floor(i / 2) - 3964926 floor(i / 1936).

Write i = 1936 r + 2 c + b, the row, the column and the byte of a 2048 x 968 half view. Then p = 465960960 + 4096 c +
2 r + b: a transposed view, affine in its coordinates, with no division at all. The floors come only from flattening
the view's coordinates to one byte number and recovering them again. isl then has to eliminate those divisions
exactly (isl_map_compute_divs, a parametric integer program with multi-precision gcds) each time two such relations
compose.

Floor division is not the algebra this needs. Strided views are integer-linear maps on mixed-radix coordinates (a
shape and a stride per mode, as in CuTe's layout algebra). Composing two of them is integer linear algebra on stride
tuples, with modes split where one shape does not divide the other. It is exact and has no divisions. The image of
such a map over a box is a union of arithmetic progressions, which is the four-field run (first, width, step, repeats)
the page compiler already uses, in closed form. Pages are power-of-two multiples, so the page of a byte is a shift and
its offset a mask.

At run time the same holds. Each SIMT lane has its coordinates from its thread and threadgroup position and forms an
address by multiply-adds with the strides. A division by a constant that cannot be avoided is a multiply-high and a
shift (Granlund and Montgomery), and a power of two is a shift and a mask.

The repair is to keep page relations in coordinates through composition, and to flatten only where runs are read
out. That changes _stream.page_region and the access relations' byte components (operand.py,
page_effects.logical_dispatch_accesses), and it is the next item for realization.

## The groups

### 1. Compare two records by an ordered list of keys, first difference decides

- Kind: comparator. Copies: 39. Deepest: 1.
- Catalog: none (mesh-graph.h has no compare or sort helper)
- Form: Add mesh_graph_compare(count, key, context, left, right) and mesh_graph_sort(records, count, size, keys, key, context) to mesh-graph.h. key(context, record, k) returns key k as a uint64. A descending key returns its complement. A double key, as in mesh_packed_heavier, is mapped to ordered bits by its consumer. The header owns the loop, the cast and the -1 or 1 rule. mesh_equal and route_demand_equal become compare equals 0. Python builds its key tuple from the same field list.
- Sites: rdma/mesh-collective.c:169 mesh_shorter; rdma/mesh-collective.c:170 mesh_equal; rdma/mesh-collective.c:371-376 mesh_message_order; rdma/mesh-collective.c:379-384 mesh_move_order; rdma/mesh-collective.c:733-737 mesh_packed_heavier; rdma/mesh-routed.c:17-24 physical_order; rdma/mesh-routed.c:26-32 local_order; rdma/mesh-routed.c:54-56 cut_end_order; rdma/mesh-routed.c:135-142 endpoint_order; rdma/mesh-fabric.c:24-27 mesh_cut_order; rdma/mesh-dataflow.c:138-141 mesh_node_order; rdma/mesh-diagnostics.c:11 order; rdma/mesh-graph.c:242 schedule_place_order; rdma/mesh-graph.c:487 classes_span_order; rdma/mesh-graph.c:497 classes_from_order; rdma/mesh-sparse.c:14 interval_order; rdma/mesh-sparse.c:19 support_order; rdma/mesh-sparse.c:29 tile_order; rdma/mesh-span-plan.h:37-40 mesh_span_cut_compare; rdma/nccl-mesh.c:925-929 cable_order; rdma/nccl-mesh.c:1756 composed_order; rdma/nccl-mesh.c:2943-2952 route_demand_equal; rdma/nccl-mesh.c:2959-2971 route_demand_order; rdma/nccl-mesh.c:2972 route_demand_sorting (thin adapter); rdma/nccl-mesh.c:3552-3558 consumption_order_compare; rdma/nccl-mesh.c:3679-3687 write_order; rdma/nccl-mesh.c:4384 overwrite_span_order; rdma/nccl-mesh.c:4461 landing_span_order; rdma/nccl-mesh.c:4948 zitem_order; rdma/nccl-mesh.c:5278 physical_order; rdma/nccl-mesh.c:5778 coverage_declaration_order; rdma/nccl-mesh.c:6302 transfer_alias_compare; rdma/nccl-mesh.c:6859 write_view_compare; rdma/nccl-mesh-metal.m:704 output_observation_order; rdma/metal-recording.m:163 MetalViewSelectionCompare; rdma/metal-recording.m:384 MetalViewAxisOrder; rdma/metal-recording.m:388 MetalViewSpanOrder; rdma/mesh-tensor-layout.m:7 TensorRunOrder; rdma/assignment.py:252-255 route_choices (same demand field list in Python)

Thirty-nine sites compare records key by key. mesh_shorter at mesh-collective.c:169 and mesh_equal at :170 compare two fields. mesh_message_order at :371 and mesh_move_order at :379 loop over a key array. The rest chain compare-and-return by hand. route_demand_equal at nccl-mesh.c:2943 and route_demand_order at :2959 list the same fields twice. assignment.py:252 lists them a third time. route_demand_sorting at :2972 is a thin adapter. mesh-graph.c holds three of these itself, at lines 242, 487 and 497.

### 2. Sort records, then scan runs of equal key to merge, drop or sum them

- Kind: sort and search index. Copies: 16. Deepest: 2.
- Catalog: none (mesh-graph.h has no run walker)
- Form: Add mesh_graph_runs(count, same, context, visit, consumer) to mesh-graph.h. same(context, i, j) says items i and j share a run. visit(consumer, begin, end) sees each run once. A dedupe passes equality and keeps begin. mesh_compile_sets passes its touch test and merges. mesh_trees_pack keeps the heaviest runs. The sort stays with the consumer, through mesh_graph_sort.
- Sites: rdma/mesh-collective.c:496-506 mesh_compile_sets; rdma/mesh-collective.c:938-951 mesh_trees_pack; rdma/nccl-mesh.c:3324-3334 routing_prepare; rdma/nccl-mesh.c:3025-3026 routing_prior; rdma/nccl-mesh.c:3106-3126 routing_paths; rdma/nccl-mesh.c:6708-6721 ncclMeshCommPlan; rdma/nccl-mesh.c:4985-4995 zabbadoo_plan; rdma/mesh-routed.c:66-68 refine_pieces; rdma/mesh-routed.c:94-99 refine_pieces; rdma/mesh-fabric.c:62-68 mesh_fabric_refine; rdma/mesh-graph.c:524-527 mesh_graph_classes; rdma/mesh-span-plan.h:40-50 mesh_span_residues_sort; rdma/mesh-sparse.c:82-88 mesh_sparse_product_compile; rdma/mesh-call.c:129 mesh_section_create; rdma/nccl-mesh-metal.m:163 metal_coverage; rdma/nccl-mesh-metal.m:395 fold_coverage

Sixteen sites sort a list and then scan runs of equal keys. mesh_compile_sets at mesh-collective.c:496-506 merges touching messages. mesh_trees_pack at :938-951 keeps the heaviest trees of each root. routing_prepare at nccl-mesh.c:3324-3334 picks one demand per equal run. ncclMeshCommPlan at :6708-6721 scans group runs twice. refine_pieces at mesh-routed.c:94-99 and mesh_fabric_refine at mesh-fabric.c:62-68 sort and drop repeats. So do mesh_graph_classes at mesh-graph.c:524-527 and mesh_span_residues_sort at mesh-span-plan.h:40-50. The deepest copies sit inside another helper that repeats the same scan.

### 3. Walk one compressed row of 32-bit words, first[v] to first[v + 1], by hand

- Kind: graph traversal. Copies: 15. Deepest: 2.
- Catalog: mesh_graph_row (mesh-graph.h:92)
- Form: Call mesh_graph_row(first, head, v, visit, consumer, NULL, NULL) where the loop reads head[k]. mesh_graph_row passes only head[k] to visit. Most of these loops need the index k. Add one twin, mesh_graph_row_at(first, v, visit, consumer), that calls visit(consumer, k). The visit adds a length, counts a use, takes a minimum or grows a length.
- Sites: rdma/mesh-collective.c:903 mesh_trees_pack; rdma/mesh-collective.c:906 mesh_trees_pack; rdma/mesh-collective.c:908-909 mesh_trees_pack; rdma/mesh-collective.c:930-936 mesh_trees_pack; rdma/mesh-collective.c:610 mesh_program_steps_cut; rdma/mesh-collective.c:711 mesh_program_check_members; rdma/mesh-fabric.c:98 mesh_paths_check; rdma/mesh-fabric.c:158 mesh_write_compile; rdma/mesh-routed.c:208 mesh_transfer_order; rdma/mesh-routed.c:323 mesh_routed_compile; rdma/mesh-routed.c:337 mesh_routed_compile; rdma/nccl-mesh.c:537 comm_compile; rdma/nccl-mesh.c:630 write_forwards; rdma/nccl-mesh.c:2554 bind_routed_steps; rdma/nccl-mesh.c:3729 order_transfers

Fifteen sites loop over a compressed row by hand. mesh_trees_pack walks the crossing list four times, at mesh-collective.c:903, 906, 908-909 and 930-936. Those four sit inside its packing loop. Eleven more row loops sit in mesh-collective.c, mesh-fabric.c, mesh-routed.c and nccl-mesh.c. mesh_graph_row visits such a row with a callback. These loops do not call it. The callback gets head[k] and not k, so most sites need the twin.

### 4. Free each element's arrays, then the array of elements

- Kind: copy or free boilerplate. Copies: 12. Deepest: 2.
- Catalog: none
- Form: Add mesh_release_array(array, count, size, release) to a small shared header. release(element) frees one element's arrays. The helper then frees the array. Each owner passes its release: program_free, trees_free, mesh_paths_release or a one-line free. Give the interval set (mesh_defined) its own free so mesh_program_check_members calls it.
- Sites: rdma/nccl-mesh.c:296-305 routes_free; rdma/nccl-mesh.c:2844-2863 routing_free; rdma/nccl-mesh.c:2107 write_packets_free; rdma/nccl-mesh.c:3130-3133 fold_basis_free; rdma/nccl-mesh.c:2394-2408 steps_free; rdma/nccl-mesh.c:3669-3675 schedules_free; rdma/nccl-mesh.c:2183 free of per-rank transfers; rdma/nccl-mesh.c:6371 free of sources' pages; rdma/mesh-collective.c:704-725 mesh_program_check_members; rdma/mesh-collective.c:953 mesh_trees_pack; rdma/mesh-routed.c:121 refine_pieces; rdma/mesh-routed.c:294-298 mesh_routed_free

Twelve sites free an array of structs that each own arrays. routes_free at nccl-mesh.c:296-305 writes the loop twice. routing_free at :2844-2863 nests two loops. mesh_program_check_members at mesh-collective.c:704-725 frees a set array by hand, since the set has no free. Each loop pairs with its own allocation code. routes_free and routing_free call free functions that are copies themselves. That puts them two layers deep. Most other sites are plain frees, so the match is loose.

### 5. Find, test or sum the links that join one unordered pair of nodes, by scanning the link list

- Kind: group-by and sum. Copies: 11. Deepest: 2.
- Catalog: none (mesh-graph.h has no pair lookup)
- Form: Add mesh_graph_group(count, key, context, order, first) to mesh-graph.h. It sorts item numbers by key(context, i) into order[]. Run g is [first[g], first[g + 1]). Links and cables pass key = lower end, then higher end. mesh_joins asks whether a run exists. mesh_parallel reads one run's length and weight sum. mesh_map_members, mesh_solve_links and the cable composers read each run once. mesh_transfer_order tests one cable and keeps its own test.
- Sites: rdma/mesh-collective.c:21-22 mesh_joins; rdma/mesh-collective.c:30-34 mesh_parallel; rdma/mesh-collective.c:233-238 mesh_map_members (pair test at 236); rdma/mesh-collective.c:261-274 mesh_map_members (pair test at 264); rdma/mesh-collective.c:749 mesh_solve_links; rdma/mesh-collective.c:753 mesh_solve_links; rdma/mesh-fabric.c:283-286 mesh_write_routes_compose; rdma/mesh-fabric.c:300-303 mesh_write_routes_compose; rdma/mesh-routed.c:215 mesh_transfer_order; rdma/nccl-mesh.c:2785-2789 path_hop_split; rdma/mesh-dataflow.c:109-118 mesh_paired

Eleven sites ask which links join one node pair. mesh_joins at mesh-collective.c:21-22 scans every link for the pair. mesh_parallel at :30-34 scans again to count and sum. mesh_map_members scans twice, at :236 and :264. mesh_solve_links scans at :749 and :753. mesh_write_routes_compose at mesh-fabric.c:285 and :302 scans twice. mesh_routes reaches mesh_parallel once per link at :206, so each link rescans all links. mesh_link_map_check calls mesh_joins N times N at :91. The two deepest copies sit inside other copies.

### 6. For each byte span, find the other spans on the same buffer that overlap it

- Kind: interval set. Copies: 10. Deepest: 2.
- Catalog: mesh_graph_span_reach and mesh_graph_meets (mesh-graph.h:146-148)
- Form: Call mesh_graph_span_reach once per buffer over the sorted spans. Call mesh_graph_meets for each step. Wrap both in one call, mesh_graph_overlaps(spans, count, group, context, visit). The consumer passes the spans, the buffer key and a meet callback. schedule_group keeps earlier non-SEND steps. mesh_transfer_order adds a follow edge. mesh_fabric_refine adds the cut points of the meeting span.
- Sites: rdma/nccl-mesh.c:3647-3664 schedule_group (with meet at 3588-3590); rdma/mesh-routed.c:228-239 mesh_transfer_order; rdma/mesh-fabric.c:55-61 mesh_fabric_refine; rdma/nccl-mesh.c:5283-5290 physical_overlap; rdma/nccl-mesh.c:5915-5919 input_intersects; rdma/nccl-mesh.c:6263 declared; rdma/metal-recording.m:666 piece scan; rdma/metal-recording.m:700 piece scan; rdma/metal-recording.m:735 piece scan; rdma/metal-recording.m:1804 piece scan

Ten sites find the earlier spans that overlap one span. schedule_group at nccl-mesh.c:3647-3664 tests every earlier step with meet, twice. meet at :3588-3590 is a hand overlap test, so the loop is a copy that calls a copy. mesh_transfer_order at mesh-routed.c:228-239 and mesh_fabric_refine at mesh-fabric.c:55-61 scan the same way. physical_overlap, input_intersects and declared repeat the test. metal-recording.m holds four more. nccl-mesh.c calls mesh_graph_meets at lines 4433 and 4506. These ten sites do not.

### 7. Grow an array by doubling its room, with an overflow check

- Kind: copy or free boilerplate. Copies: 10. Deepest: 2.
- Catalog: none (grow at nccl-mesh.c:3979 does it but is static)
- Form: Move grow to a shared header: grow(items, room, count, size, realloc_fn). It doubles room with an overflow check and calls the allocator operand. fold_basis_add passes mesh_isl_realloc. Every other consumer passes realloc. Each consumer passes its own array, room, count and item size.
- Sites: rdma/mesh-collective.c:647-653 mesh_defined_add; rdma/mesh-collective.c:918-923 mesh_trees_pack; rdma/nccl-mesh.c:3138-3144 fold_basis_add; rdma/nccl-mesh.c:3979-3988 grow (the one shared helper, static); rdma/mesh-span-plan.h:52-58 mesh_span_residue_add; rdma/mesh-span-plan.h:72-78 mesh_page_run_add; rdma/mesh-lease-retire.h:70-76 mesh_lease_retire_mark_use; rdma/nccl-mesh-metal.m:371-376 argument_pages; rdma/mesh-routed.c:85-88 refine_pieces; rdma/mesh-routed.c:145-152 order_edge

Ten sites grow an array by doubling. grow at nccl-mesh.c:3979-3988 is the shared helper, and it is static. No other file can call it. mesh_defined_add at mesh-collective.c:647-653 doubles its range array by hand. That function is itself an interval set copy, so it sits two layers deep. mesh_trees_pack at :918-923, fold_basis_add at nccl-mesh.c:3138-3144 and order_edge at mesh-routed.c:145-152 do it again. Two sites in mesh-span-plan.h and one in mesh-lease-retire.h do it too. Each picks its own start size.

### 8. Allocate, copy or free the first, cable and weight arrays of a struct mesh_paths

- Kind: copy or free boilerplate. Copies: 9. Deepest: 2.
- Catalog: none (mesh_write_routes_free in mesh-fabric.c holds the same three frees)
- Form: Add mesh_paths_make(out, paths, edges), mesh_paths_copy(out, from, begin, amount) and mesh_paths_release(paths) to mesh-fabric.h. mesh_write_routes_free calls mesh_paths_release for each pair. comm_free calls mesh_write_routes_free, as it already does for comm->composed. paths_free and weighed_free go away. writes_weigh shares first and cable through a flag operand of the copy.
- Sites: rdma/nccl-mesh.c:322-325 paths_free; rdma/nccl-mesh.c:342-343 comm_free; rdma/nccl-mesh.c:598-603 ncclMeshCommWriteRoutes; rdma/nccl-mesh.c:2768-2771 weighed_free; rdma/nccl-mesh.c:2800-2806 writes_weigh; rdma/nccl-mesh.c:3117-3121 routing_paths; rdma/mesh-fabric.c:188-196 mesh_write_routes_free; rdma/mesh-fabric.c:246-251 mesh_write_routes_default; rdma/mesh-fabric.c:289-297 mesh_write_routes_compose

Nine sites list the three arrays of struct mesh_paths by hand. paths_free at nccl-mesh.c:322-325 frees them for one pair. mesh_write_routes_free at mesh-fabric.c:188-196 frees the same three. comm_free at nccl-mesh.c:342-343 loops over paths_free instead of calling mesh_write_routes_free. That copies a helper that is itself a copy. ncclMeshCommWriteRoutes at :598-603 and mesh_write_routes_compose at mesh-fabric.c:289-297 copy the arrays. routing_paths at :3117-3121 and mesh_write_routes_default at mesh-fabric.c:246-251 allocate them. writes_weigh and weighed_free use a third ownership rule.

### 9. Scan a whole table for the records of one key (tree root, segment, demand, scope, cable)

- Kind: group-by and sum. Copies: 17. Deepest: 1.
- Catalog: none (mesh_graph_row reads rows by vertex, but nothing builds rows by key)
- Form: Use mesh_graph_group, as for the link pair group. The key operand reads tree root, segment, demand, scope or cable. Each consumer then reads a key's run, and reads it once. In Python, build one dict from cable to channels, as assignment.py already does. Pair this with mesh_graph_row for the reads.
- Sites: rdma/mesh-collective.c:317-321 mesh_trees_check; rdma/mesh-collective.c:336-341 mesh_trees_heaviest; rdma/mesh-collective.c:468-473 mesh_compile_sets; rdma/mesh-collective.c:579-590 mesh_program_ranges; rdma/mesh-collective.c:613 mesh_program_steps_cut; rdma/mesh-collective.c:679-681 mesh_program_check_members; rdma/nccl-mesh.c:400-404 root_carried; rdma/nccl-mesh.c:421-424 compile_form; rdma/nccl-mesh.c:435-438 compile_form; rdma/nccl-mesh.c:2750-2753 solved_steps; rdma/nccl-mesh.c:2814-2816 writes_weigh; rdma/nccl-mesh.c:3063 routing_paths; rdma/nccl-mesh.c:3087 routing_paths; rdma/nccl-mesh.c:3370-3402 routing_prepare; rdma/plan_cycles.py:99-107 main (node_of); rdma/plan_cycles.py:109-117 main (paired); rdma/assignment.py:229-231 route_axes (a dict group-by: the target shape, no change)

Seventeen sites scan a whole table for the records of one key. root_carried at nccl-mesh.c:400-404 scans all trees for one root. compile_form scans again at :421-424 and :435-438. mesh_trees_check at mesh-collective.c:317-321 and mesh_trees_heaviest at :336-341 scan all trees for each node. routing_prepare at nccl-mesh.c:3370-3402 and routing_paths at :3063 and :3087 scan per demand and per scope. mesh_program_ranges at mesh-collective.c:579-590 scans all parts four times per segment. Comments near nccl-mesh.c:3436 say such a scan held a 64-step plan for a minute. assignment.py:229 is the dict form and needs no change.

### 10. Build row offsets from counts: count each row's items, add the counts, optionally scatter the items

- Kind: group-by and sum. Copies: 17. Deepest: 1.
- Catalog: none (mesh_graph_mark and mesh_graph_row read rows, but nothing builds them)
- Form: Add mesh_graph_rows(rows, count, context, first) to mesh-graph.h. count(context, row) gives a row's size. It writes first[rows + 1] and returns the total, or EOVERFLOW. Add mesh_graph_scatter for edge heads. mesh_graph_schedule, mesh_graph_classes, mesh_graph_join and mesh_graph_depth call both. Plain running sums such as span_first and view_first pass a count of bytes or units.
- Sites: rdma/mesh-collective.c:514-518 mesh_compile_sets; rdma/nccl-mesh.c:2573-2575 bind_routed_steps; rdma/nccl-mesh.c:2599-2608 routed_steps; rdma/nccl-mesh.c:2614-2625 counted_steps; rdma/nccl-mesh.c:3701-3717 order_transfers; rdma/nccl-mesh.c:5865 cuts; rdma/nccl-mesh.c:6282-6301 transfer_first; rdma/nccl-mesh.c:6453 issue_order_like; rdma/nccl-mesh.c:7058 span_first; rdma/nccl-mesh.c:7253 view_first; rdma/mesh-routed.c:168-171 order_closes; rdma/mesh-routed.c:282-283 dependency_order; rdma/mesh-graph.c:287-291 mesh_graph_schedule; rdma/mesh-graph.c:310-312 mesh_graph_schedule; rdma/mesh-graph.c:544-546 mesh_graph_classes; rdma/mesh-graph.c:559-575 mesh_graph_join; rdma/mesh-graph.c:586-588 mesh_graph_depth

Seventeen sites build row offsets from counts. order_transfers at nccl-mesh.c:3701-3717 and transfer_first at :6282-6301 run the same walk with the same guard. issue_order_like at :6453 recomputes what transfer_first holds. routed_steps, counted_steps and bind_routed_steps repeat the sum. order_closes and dependency_order at mesh-routed.c:168-171 and :282-283 repeat it. The catalog file mesh-graph.c writes it five times itself, at lines 287-291, 310-312, 544-546, 559-575 and 586-588.

### 11. Binary search a sorted list for the first item not before a key

- Kind: sort and search index. Copies: 14. Deepest: 1.
- Catalog: mesh_graph_through (mesh-graph.c:435, uint64 values only)
- Form: Add mesh_graph_lower_bound(count, before, context, key) to mesh-graph.h. It returns the first index i where before(context, i, key) is false. Each consumer passes its own order test. mesh_graph_through and mesh_graph_meets call it. plan_cycles.py frontier calls bisect_right over a list of ends built once per channel.
- Sites: rdma/nccl-mesh.c:2986-2990 route_demand_first; rdma/nccl-mesh.c:1779 lease_at; rdma/nccl-mesh.c:2162 write_selection; rdma/nccl-mesh.c:2192 span_after; rdma/nccl-mesh.c:3843 group_at; rdma/nccl-mesh.c:5786 coverage_begin; rdma/mesh-span-plan.h:97 mesh_page_run_at; rdma/mesh-tensor-layout.m:61 fragment:end:; rdma/nccl-mesh-metal.m:173 metal_coverage; rdma/metal-recording.m:186 MetalViewBlockGather; rdma/mesh.h:320 mesh_layout_within; rdma/mesh-graph.c:438 mesh_graph_through; rdma/mesh-graph.c:481 mesh_graph_meets; rdma/plan_cycles.py:92-95 main.frontier (a linear scan where bisect_right fits)

Fourteen sites search a sorted list by hand. route_demand_first at nccl-mesh.c:2986-2990 finds the first demand not before a key. lease_at, write_selection, span_after, group_at and coverage_begin do the same in that file. Five more sit in mesh-span-plan.h, mesh-tensor-layout.m, nccl-mesh-metal.m, metal-recording.m and mesh.h. The catalog holds two itself, at mesh-graph.c:438 and :481. mesh_graph_through takes uint64 values only. The copies differ only in their order test. plan_cycles.py:92 scans a list that bisect_right would search.

### 12. Test or set one vertex in a bit set (a bit a vertex, 64 to a word, NULL means every vertex)

- Kind: set or bitset. Copies: 11. Deepest: 1.
- Catalog: none (mesh.h mesh_bits_set is atomic and works on region planes, so it does not fit)
- Form: Add inline mesh_set_has(set, v), mesh_set_add(set, v) and mesh_set_clear(set, v) to mesh-plan.h, which both C files include. A NULL set means every vertex in has. mesh_member and rank_member become mesh_set_has. Each consumer passes its own word array. backend.py calls a small helper with the same layout.
- Sites: rdma/mesh-collective.c:145 mesh_member; rdma/nccl-mesh.c:6143-6145 rank_member; rdma/nccl-mesh.c:2816 writes_weigh; rdma/nccl-mesh.c:2876 routing_tree_check; rdma/nccl-mesh.c:3155 fold_basis_seed; rdma/nccl-mesh.c:3175 fold_basis_validate; rdma/nccl-mesh.c:496-497 comm_routes (set and clear); rdma/nccl-mesh.c:2708 demands_of (set); rdma/nccl-mesh.c:2718 demands_of (set); rdma/nccl-mesh.c:2870 routing_members (mask fill); rdma/torch-mesh/torch_mesh/backend.py:278 (set in Python)

Eleven sites test or set one bit in a word array. mesh_member at mesh-collective.c:145 and rank_member at nccl-mesh.c:6143 are the same test under two names. Four more places write the test inline, at nccl-mesh.c:2816, 2876, 3155 and 3175. Five places set or fill bits by hand. mesh_member is static, so nccl-mesh.c cannot call it. rank_member is declared at nccl-mesh.c:2537, so line 2816 could call it today.

### 13. Cut a range into pieces at a sorted set of points or at a fixed period

- Kind: interval set. Copies: 5. Deepest: 2.
- Catalog: none in mesh-graph.h (mesh-span-plan.h holds mesh_packet_begin and mesh_packet_next for this job)
- Form: Use mesh_packet_begin and mesh_packet_next over a cut set, and give mesh-routed.c and mesh-fabric.c access to that header. The consumer passes the range, the points or the period, and a callback that builds one piece from a length. mesh_write_compile passes the tile as the period and an empty cut set.
- Sites: rdma/mesh-fabric.c:62-74 mesh_fabric_refine; rdma/mesh-routed.c:106-117 refine_pieces; rdma/mesh-fabric.c:161-171 mesh_write_compile; rdma/mesh-span-plan.h:81-92 mesh_page_runs_read; rdma/mesh-span-plan.h:138-146 mesh_packet_next (the walk's own home)

Five sites cut a range into pieces. mesh_fabric_refine at mesh-fabric.c:62-74 sorts cut points and splits the range. refine_pieces at mesh-routed.c:106-117 cuts again the pieces that mesh_fabric_refine made. That puts it two layers deep. mesh_write_compile at mesh-fabric.c:161-171 counts pieces first and then emits them. The count pass and the emit pass can disagree. mesh_packet_next at mesh-span-plan.h:138-146 already walks a range through cuts with a cursor. mesh-fabric.c does not include that header.

### 14. Walk up a parent table from a vertex to the root, within n steps

- Kind: tree walk. Copies: 4. Deepest: 2.
- Catalog: mesh_graph_climb (mesh-graph.h:57) and mesh_graph_ancestors (mesh-graph.h:66)
- Form: Call mesh_graph_climb(parent, vertices, root, from, visit, consumer). mesh_trees_check and routing_tree_check pass a visit that tests each hop with mesh_link_between and keeps the last vertex. mesh_solve passes a visit that sets tree[at] and treel[at], or calls mesh_graph_ancestors per leaf. mesh_route_nodes reads a column of the next table, so add an up(context, v) operand beside the parent array. Its visit writes path[step].
- Sites: rdma/mesh-collective.c:327-329 mesh_trees_check; rdma/mesh-collective.c:61-67 mesh_route_nodes (with the path reversal at 69); rdma/mesh-collective.c:811-816 mesh_solve; rdma/nccl-mesh.c:2877-2883 routing_tree_check

Four sites walk up a parent table by hand. mesh_trees_check does it at mesh-collective.c:327-329. routing_tree_check does it at nccl-mesh.c:2877-2883. mesh_route_nodes does it on the next table at mesh-collective.c:61-67. mesh_solve does it at :811-816, inside its weights loop, so that copy is two layers deep. mesh_link_map_check calls mesh_graph_climb at :109. mesh_tree_shape calls mesh_graph_ancestors at :353. These four sites do neither. The walk at line 328 can read a parent that no check has seen.

### 15. Turn an element count into allocation units (ceiling by the page quantum) and checked bytes

- Kind: other. Copies: 8. Deepest: 1.
- Catalog: none (mesh_page_quantum at mesh-plan.h:176 has no units function beside it)
- Form: Add mesh_page_units(elements, quantum) to mesh-plan.h beside mesh_page_quantum. Add mesh_operand_bytes(elements, element_bytes, out), which checks the multiply. Each consumer passes its elements, quantum and element size.
- Sites: rdma/nccl-mesh.c:3415-3421 route_path_counts; rdma/nccl-mesh.c:3433-3435 route_tree_counts; rdma/nccl-mesh.c:3314 routing_prepare; rdma/nccl-mesh.c:6799 plan_routes; rdma/mesh-routed.c:360 mesh_routed_compile; rdma/mesh-fabric.c:116 mesh_write_ranges; rdma/mesh-collective.c:578 mesh_program_ranges; rdma/mesh-sparse.c:34-36 divided

Eight sites turn a count into units by a rounded-up divide. route_path_counts at nccl-mesh.c:3415-3421 does it with a checked byte count. route_tree_counts repeats both at :3433-3435. routing_prepare repeats the divide at :3314 and plan_routes at :6799. mesh_routed_compile, mesh_write_ranges, mesh_program_ranges and divided repeat it in other files. mesh_page_quantum sits at mesh-plan.h:176 with no units function beside it. This is a weak match, since each site is one expression.

### 16. Split a count of units over parts: evenly, or by weight with the remainder handed out by a rule

- Kind: other. Copies: 7. Deepest: 1.
- Catalog: none (mesh_graph_cells cuts into fixed-size cells, not equal pieces)
- Form: Add one apportion function in C and one in mesh_graph.py: apportion(units, parts, weight, remainder_rule). An even cut passes weight 1. The remainder rule is an operand: first parts, or largest fraction. mesh_cut_next adds only the quantum and the size override. mesh_program_ranges passes the exp of the shifted log weight. mesh_write_ranges passes the path weight.
- Sites: rdma/mesh-collective.c:554-559 mesh_cut_next; rdma/torch-mesh/torch_mesh/partition.py:277 attach; rdma/torch-mesh/torch_mesh/chain.py:59-61 _share; rdma/mesh-collective.c:585-597 mesh_program_ranges; rdma/mesh-fabric.c:123-134 mesh_write_ranges; rdma/allocate.py:155-190 project; rdma/allocate.py:11-20 partition_bounds

Seven sites split a count over parts. mesh_cut_next at mesh-collective.c:554-559 cuts evenly and gives the first remainder parts one more. partition.py:277 and chain.py:59-61 write the same even rule. mesh_program_ranges at mesh-collective.c:585-597 floors weighted shares and hands the remainder to the first parts. mesh_write_ranges at mesh-fabric.c:123-134 floors with integers and hands it to the largest fraction. allocate.py project and partition_bounds do the same in Python. The remainder rules differ. The floor step and the sum check agree.

### 17. Dijkstra shortest paths from one root, picking the nearest unfinished vertex each step

- Kind: shortest path. Copies: 3. Deepest: 2.
- Catalog: none (mesh-graph.h has no shortest-path function)
- Form: Add mesh_graph_shortest(vertices, edges, end, context, source, cost_size, extend, less, cost, parent) to mesh-graph.h. extend(context, edge, side, from_cost, to_cost) builds a neighbour's cost. less(context, a, b) orders two costs. mesh_distances passes an (hops, inverse bandwidth) cost. mesh_solve passes a double length. mesh_routes calls extend again to find each node's tight parents.
- Sites: rdma/mesh-collective.c:173-187 mesh_distances; rdma/mesh-collective.c:798-810 mesh_solve; rdma/mesh-collective.c:216 mesh_routes (the hop-extend sum of line 183, written again)

Three sites write the shortest-path step. mesh_distances at mesh-collective.c:173-187 is Dijkstra in dense form. mesh_solve repeats the loop at :798-810 with a double length. mesh_solve's copy sits inside its weights loop, so it is two layers deep. mesh_routes writes the hop-extend sum again at :216. design/rubrics.md names a Dijkstra in the collective solver as a C9 case. Each copy scans all vertices to pick the nearest, so each costs N squared.

### 18. Find an equal tree row in a list of distinct rows, or add it, and count a hit

- Kind: group-by and sum. Copies: 3. Deepest: 2.
- Catalog: none
- Form: Add one find-or-add function over a table of distinct u32 rows. Operands are the table, the row width, a leading key and the row. It owns the rows, frees them and returns the slot. mesh_solve adds a hit to the slot. mesh_trees_pack adds flow. fold_basis_add adds a share.
- Sites: rdma/mesh-collective.c:818-826 mesh_solve; rdma/mesh-collective.c:911-928 mesh_trees_pack; rdma/nccl-mesh.c:3134-3148 fold_basis_add

Three sites look for an equal tree row, and add it if missing. mesh_solve does it at mesh-collective.c:818-826 with memcmp over n words. mesh_trees_pack does it at :911-928 and fold_basis_add at nccl-mesh.c:3134-3148. Each scan reads the list in order. So the cost grows with the tree count, for every demand and round. The two mesh-collective.c copies sit inside the solve loops, two layers deep.

### 19. Compare the rows of one key in two flat tables, row by row, with two cursors

- Kind: comparator. Copies: 3. Deepest: 2.
- Catalog: none
- Form: Add one function that walks the rows of key A in table a and the rows of key B in table b together. It calls equal(context, i, j) on each pair and fails on a length mismatch. Tree and path each pass their row compare. Build the first-row offsets once with mesh_graph_group so the cursor moves need no scan.
- Sites: rdma/nccl-mesh.c:2991-3002 route_tree_basis_equal; rdma/nccl-mesh.c:3003-3015 route_path_basis_equal; rdma/assignment.py:268-273 route_choices (the weights check in Python)

Two functions share one shape. route_tree_basis_equal at nccl-mesh.c:2991-3002 and route_path_basis_equal at :3003-3015 each move two cursors to the next row of a key. Each compares the two rows and stops when both tables end. Only the row compare differs. The cursor moves at lines 2995-2996 and 3006-3007 are full key scans, so each sits one layer inside the key-scan copy. assignment.py:268 checks the same weights in Python.

### 20. Allocate, copy or free the root, parent and log_weight arrays of a struct mesh_trees

- Kind: copy or free boilerplate. Copies: 6. Deepest: 1.
- Catalog: none
- Form: Add mesh_trees_make(nodes, room), mesh_trees_copy(made, from) and mesh_trees_free(trees) to mesh-plan.h. Put the three arrays in one block so one free ends them. Add mesh_trees_split(trees, first, count) for the view that comm_routes carves out and does not own. Callers pass the struct pointer and the room.
- Sites: rdma/nccl-mesh.c:267-270 trees_free; rdma/nccl-mesh.c:273-283 trees_copy; rdma/nccl-mesh.c:429-431 compile_form; rdma/nccl-mesh.c:489-494 comm_routes; rdma/nccl-mesh.c:2740-2741 solved_steps (allocation); rdma/nccl-mesh.c:2757 solved_steps (inline frees)

Six sites list the three arrays of struct mesh_trees. trees_free at nccl-mesh.c:267-270 frees them and zeroes the struct. trees_copy at :273-283 sizes and copies them. compile_form at :429-431, comm_routes at :489-494 and solved_steps at :2740-2741 allocate them, each with its own room and failure check. solved_steps frees them inline at :2757. routes_free relies on comm->rooted pointing inside comm->trees. That rule lives in two functions.

### 21. Check that compressed rows are well formed: first[0] is 0, rows never shrink, the last entry equals the total

- Kind: other. Copies: 6. Deepest: 1.
- Catalog: none (mesh_graph_row reads compressed rows but nothing checks them)
- Form: Add mesh_graph_rows_check(first, rows, total) to mesh-graph.h. It returns 0 when first[0] is 0, no row shrinks and first[rows] equals total, else EINVAL. Each consumer passes its table, row count and total.
- Sites: rdma/nccl-mesh.c:2900-2903 routing_catalog_check (source_first); rdma/nccl-mesh.c:2911 routing_catalog_check (first); rdma/mesh-collective.c:669-670 mesh_program_check_members; rdma/mesh-routed.c:186-188 mesh_transfer_order; rdma/mesh-routed.c:306-309 mesh_routed_compile; rdma/mesh-fabric.c:84-91 mesh_paths_check

Six checks validate compressed rows by hand. routing_catalog_check does it twice, at nccl-mesh.c:2900-2903 and :2911. mesh_program_check_members, mesh_transfer_order, mesh_routed_compile and mesh_paths_check repeat the check. Each needs first[0] to be 0 and rows not to shrink. Each needs the last entry to equal the total. The catalog reads such rows and never checks them.

### 22. Weights exp(x - max): find the largest value, then exponentiate each value minus it

- Kind: other. Copies: 3. Deepest: 2.
- Catalog: none
- Form: Add one function that writes exp((x[i] - max) * scale) for a list and returns the sum. mesh_solve and mesh_program_ranges pass their lists and scales. Python calls the same rule through numpy in mesh_graph.py.
- Sites: rdma/mesh-collective.c:787-789 mesh_solve; rdma/mesh-collective.c:579-581 mesh_program_ranges (again at 587-591); rdma/assignment.py:474-475 envelope

Three sites shift by the maximum before they exponentiate. mesh_solve finds the top load at mesh-collective.c:787-788 and uses it at :789. That line sits inside the weights loop, so it is two layers deep. mesh_program_ranges does the same for log weights at :579-581 and again at :587-591. assignment.py:474 does it with numpy. mesh_exp is called three times per part in mesh_program_ranges. This is a weak match.

### 23. Wait on a shared word in short slices up to a deadline, test whether the bridge is gone, and re-read a record set until its serial holds still

- Kind: other. Copies: 3. Deepest: 2.
- Catalog: none
- Form: Add two functions in one header: wait_changed(word, seen, deadline, bridge), which returns ESRCH when the bridge is gone, and read_stable(serial, copy, consumer), which retries until the serial is even and unchanged. Each of the three callers passes its own word and copy.
- Sites: rdma/mesh-dataflow.c:208-223 mesh_nodes_health; rdma/mesh-dataflow.c:84-104 mesh_records_read; rdma/mesh-dataflow.c:247-253 mesh_map_wait

Three functions in mesh-dataflow.c share a wait loop. mesh_nodes_health at lines 208-223 runs a serial-read loop. If the serial is odd it waits and tests mesh_gone. mesh_records_read at :84-104 runs the same loop on the map serial. mesh_map_wait at :247-253 repeats the wait and the gone test. mesh_nodes_health calls mesh_records_read inside its loop. So it is a copy around a copy, two layers deep.

### 24. Add per-pair or per-key values into a table with an overflow guard (sum or maximum)

- Kind: group-by and sum. Copies: 5. Deepest: 1.
- Catalog: none (mesh_graph_frontier groups by key but keeps a maximum, not a sum)
- Form: Add a checked accumulate, mesh_sum_into(cell, value, mode), to mesh-plan.h. Mode picks sum or maximum, and it returns EOVERFLOW on a wrap. Each consumer passes its table and the cell its key names. Build the key table with mesh_graph_group where a scan is still needed.
- Sites: rdma/nccl-mesh.c:3436-3455 route_tree_counts; rdma/nccl-mesh.c:3359-3368 routing_prepare; rdma/nccl-mesh.c:3389-3401 routing_prepare; rdma/nccl-mesh.c:6789-6811 plan_routes; rdma/mesh-routed.c:355-362 mesh_routed_compile

Five sites add values into a table with an overflow check. route_tree_counts at nccl-mesh.c:3436-3455 sums units and bytes for each demand. routing_prepare sums up and down bytes at :3359-3368 and shares at :3389-3401. plan_routes at :6789-6811 adds a weighted vector per call, or takes the maximum. mesh_routed_compile at mesh-routed.c:355-362 does the same for one transaction. Commit a88732ac changed route_tree_counts to read its own demands' trees alone.

### 25. Keep a running page sum over a message list, storing each message's entry number and running end

- Kind: group-by and sum. Copies: 5. Deepest: 1.
- Catalog: none (mesh-graph.h has no prefix-sum function)
- Form: Add one prefix function to mesh-graph.h. It takes a list, a weight callback and a store callback for the entry and running end. Each consumer passes its message list and its page weight. ncclMeshCommPlan passes run extents with the overflow guard. group_at reads the result through mesh_graph_lower_bound.
- Sites: rdma/nccl-mesh.c:3632-3644 schedule_group; rdma/nccl-mesh.c:3748-3758 order_transfers; rdma/nccl-mesh.c:3917-3921 schedule_units; rdma/nccl-mesh.c:3965 schedule_laid (variant: base plus laid); rdma/nccl-mesh.c:6729-6742 ncclMeshCommPlan (running total of run extents)

Five sites keep a running page sum. schedule_group does it at nccl-mesh.c:3632-3644 with pages_add on each placed message. order_transfers sorts at :3755 and then recomputes entry and end at :3756-3757. schedule_units repeats the sum at :3917-3921. schedule_laid at :3965 is a variant. ncclMeshCommPlan keeps a running total of run extents at :6729-6742, and group_at searches it. The three sums must stay in step.

### 26. Order item numbers by an integer key, by sort or by one rescan per key value, and read back the permutation

- Kind: sort and search index. Copies: 5. Deepest: 1.
- Catalog: none
- Form: Add one function that lists item indices in the order of a key, ties by index. Operands are the count and key(context, i). mesh_lay passes the step's round. schedule_consumption and zabbadoo_plan pass their key tuples through mesh_graph_sort. The sortedness check of mesh_rounds_sorted and write_view_validate becomes one test of the same function.
- Sites: rdma/mesh-collective.c:963-975 mesh_rounds_sorted and mesh_lay; rdma/nccl-mesh.c:3565-3577 schedule_consumption; rdma/mesh-graph.c:361 mesh_graph_schedule; rdma/nccl-mesh.c:4982 zabbadoo_plan; rdma/nccl-mesh.c:6870-6889 write_view_validate

Five sites order items by a key and read back the permutation. mesh_lay at mesh-collective.c:968-978 checks that rounds ascend. If not, it finds the top round and rescans all steps once per round. schedule_consumption at nccl-mesh.c:3565-3577 sorts keyed records and copies the indexes back. mesh_graph_schedule does the same at mesh-graph.c:361. write_view_validate at nccl-mesh.c:6870-6889 checks order and then sorts. This is a weak match.

### 27. Allocate and free the link, bandwidth, next and mark arrays of a struct mesh_link_map

- Kind: copy or free boilerplate. Copies: 5. Deepest: 1.
- Catalog: none
- Form: Add mesh_link_map_make(kind, nodes, links) and mesh_link_map_free(map) to mesh-collective.c, beside mesh_link_map_check. One block holds link, bandwidth and mark. Each consumer fills link[e] and bandwidth[e] from its own list. map_free and mesh_whole_free are deleted.
- Sites: rdma/nccl-mesh.c:285-288 map_free; rdma/nccl-mesh.c:970-981 fabric_discover; rdma/mesh-dataflow.c:142 mesh_whole_free; rdma/mesh-dataflow.c:148-158 mesh_whole_make; rdma/mesh.py:44-56 LinkMap.__init__

Five sites build or free a link map by field. map_free at nccl-mesh.c:285-288 frees link, bandwidth, next and mark. mesh_whole_free at mesh-dataflow.c:142 frees the same four. fabric_discover at nccl-mesh.c:970-981 callocs three of them and copies each cable in. mesh_whole_make at mesh-dataflow.c:148-158 builds the same struct from node records. LinkMap.__init__ in mesh.py builds it a third way. The storage rule is written five times.

### 28. Page divided by gcd(page, element bytes): the elements in one page-aligned unit

- Kind: other. Copies: 5. Deepest: 1.
- Catalog: none in mesh-graph.h (mesh_page_quantum at mesh-plan.h:176 returns this value)
- Form: Call mesh_page_quantum(page, element_bytes) in mesh_routed_compile. Make mesh_page_quantum call mesh_span_gcd. TensorGCD and the loop at mesh-tensor-layout.m:1097 call the same gcd. Keep one gcd in one header.
- Sites: rdma/mesh-routed.c:358-359 mesh_routed_compile; rdma/mesh-plan.h:176 mesh_page_quantum; rdma/mesh-span-plan.h:19-22 mesh_span_gcd; rdma/mesh-tensor-layout.m:140 TensorGCD; rdma/mesh-tensor-layout.m:1097 inline gcd loop

Five sites hold a gcd loop. mesh_routed_compile runs one at mesh-routed.c:358-359 and divides at :360. That value is mesh_page_quantum(page, element_bytes). mesh-routed.c includes mesh-plan.h, which declares it. mesh_page_quantum has an inline gcd of its own. mesh_span_gcd at mesh-span-plan.h:19-22 is the same loop. mesh-tensor-layout.m holds two more.

### 29. Pair each send with its receive by sorting and matching, or by a linear search for the first unmatched receive

- Kind: sort and search index. Copies: 4. Deepest: 1.
- Catalog: mesh_graph_through (a rank query; it counts and does not match equals)
- Form: Add a header function that sorts two arrays by a consumer comparator and calls pair(consumer, i, j) for the k-th of each. It returns an error where counts or keys differ. Each consumer passes the arrays, the comparator and the pair callback. If write_steps needs it, extend the function with a matched-index output.
- Sites: rdma/mesh-routed.c:36-44 number_pieces; rdma/mesh-routed.c:244-251 mesh_transfer_order; rdma/mesh-routed.c:337-348 mesh_routed_compile; rdma/nccl-mesh.c:2514-2528 write_steps (a forward merge; the weakest fit)

Four sites pair a send with its receive. number_pieces at mesh-routed.c:36-44 sorts all pieces and pairs neighbours. mesh_transfer_order at :244-251 sorts two lists and pairs the k-th of each. mesh_routed_compile at :337-348 searches the destination's whole step list for each send, so it costs steps squared per rank. write_steps at nccl-mesh.c:2514-2528 merges sorted receives with one forward pointer. I found no other copy of that one, so it is the weakest fit.

### 30. Find the children of each vertex of a parent table by scanning every vertex

- Kind: tree walk. Copies: 4. Deepest: 1.
- Catalog: none (mesh_graph_tree_levels gives depth and height only)
- Form: Add mesh_graph_children(parent, vertices, root, first, child) to mesh-graph.h. It builds compressed child rows in one pass. Consumers read rows with mesh_graph_row: a count in fold_basis_validate, a recursion in mesh_fold_visit, a filtered first child in mesh_first_child.
- Sites: rdma/mesh-collective.c:360-366 mesh_first_child; rdma/mesh-collective.c:355 mesh_tree_shape (a child marks its parent); rdma/mesh-fold.h:24-27 mesh_fold_visit; rdma/nccl-mesh.c:3180-3184 fold_basis_validate

Three sites scan a whole parent table for a vertex's children. mesh_first_child at mesh-collective.c:360-366 scans n vertices per call. It is called at lines 456 and 489 inside loops over every node, so one tree costs n squared. mesh_fold_visit at mesh-fold.h:24-27 and fold_basis_validate at nccl-mesh.c:3180-3184 scan the same way. mesh_tree_shape at mesh-collective.c:355 marks a parent from a child and needs no scan. The catalog has tree levels but no child rows.

### 31. Walk a cable path hop by hop, reading the far end of each cable

- Kind: graph traversal. Copies: 4. Deepest: 1.
- Catalog: none (mesh_graph_tree_path takes a step operand but follows a parent tree)
- Form: Add a header walk over a hop list: step(consumer, k, at) gives the next node and visit(consumer, k, at, next) sees each hop. A far_end(cable, at) helper serves path_hop_next, path_hop_split and mesh_paths_check. Each consumer passes the cables and a callback.
- Sites: rdma/mesh-fabric.c:98-106 mesh_paths_check; rdma/mesh-fabric.c:158-174 mesh_write_compile; rdma/nccl-mesh.c:2775-2779 path_hop_next; rdma/nccl-mesh.c:2783 path_hop_split

Four sites read the far end of a cable or walk hops. mesh_paths_check at mesh-fabric.c:98-106 steps to the far end and marks visited nodes. mesh_write_compile at :158-174 walks the same hops. path_hop_next at nccl-mesh.c:2775-2779 reads the far end. path_hop_split repeats that read at :2783. This is a weak match.

### 32. Check that every edge names two different vertices below the vertex count

- Kind: other. Copies: 4. Deepest: 1.
- Catalog: none
- Form: Add mesh_graph_edges_bad(vertices, edges, end, context) to mesh-graph.h. It returns the first bad edge number, or edges if none. Each consumer chooses to fail or to skip.
- Sites: rdma/mesh-collective.c:195 mesh_routes; rdma/mesh-collective.c:87 mesh_link_map_check; rdma/mesh-collective.c:746 mesh_solve_links; rdma/mesh-fabric.c:18 mesh_fabric_check

Four loops check an edge range. mesh_routes does it at mesh-collective.c:195. mesh_link_map_check does it at :87. mesh_solve_links does it at :746 and skips bad links. mesh_fabric_check does it at mesh-fabric.c:18 for cables. The loops differ only in field names and in what they do on a bad edge. This is a weak match.

### 33. Turn a share into a log weight that every node computes alike, non-positive shares to minus infinity

- Kind: numeric approximation. Copies: 4. Deepest: 1.
- Catalog: none (mesh_log is file-static and in no header)
- Form: Declare mesh_exp and mesh_log in mesh-plan.h as inline functions and drop static. compile_form and ncclMeshFoldPrepare call mesh_log. The three libm calls become that call. They take no operands.
- Sites: rdma/mesh-collective.c:294-310 mesh_exp and mesh_log (static, built from exact operations); rdma/nccl-mesh.c:424 compile_form (libm log); rdma/nccl-mesh.c:437 compile_form (libm log); rdma/nccl-mesh.c:3251 ncclMeshFoldPrepare (libm log)

Four sites make a log weight. mesh_log at mesh-collective.c:303-310 uses exact operations only. Its comment says every node must compute the same shares whatever its libm. mesh_log is static, so no header offers it. compile_form at nccl-mesh.c:424 and :437 and ncclMeshFoldPrepare at :3251 call libm log. Those values become log_weight, which mesh_exp reads later. Nodes may then disagree on a share.

### 34. Allocate, copy or free the four arrays of a struct mesh_program

- Kind: copy or free boilerplate. Copies: 3. Deepest: 1.
- Catalog: none
- Form: Add mesh_program_make(program, parts, nodes, moves), mesh_program_copy(made, from) and mesh_program_free(program) to mesh-collective.c, beside mesh_compile_room, which knows the sizes. program_copy and program_compile call make and free. Callers pass the program pointer.
- Sites: rdma/nccl-mesh.c:263-265 program_free; rdma/nccl-mesh.c:353-364 program_copy; rdma/nccl-mesh.c:369-381 program_compile

Three functions list the four program arrays. program_free at nccl-mesh.c:263-265 frees segment, log_weight, first and move. program_copy at :353-364 allocates and copies the same four. program_compile at :369-381 callocs them and frees all four on failure. A new field in mesh_program needs edits in all three.

### 35. A set of integer intervals: add one with merging, and test whether one is covered

- Kind: interval set. Copies: 3. Deepest: 1.
- Catalog: none (mesh_graph_span and mesh_graph_meets test overlap only; they do not union or test cover)
- Form: Add one interval set to a shared header: add, covers and meets. The set is the consumer's storage. mesh_program_check_members and the nccl-mesh.c source relations both call it. The doubling in add uses the shared grow.
- Sites: rdma/mesh-collective.c:630-658 mesh_defined_has and mesh_defined_add; rdma/nccl-mesh.c:5248-5272 spans_meet and spans_add; rdma/mesh-sparse.c:67-77 (sort intervals, merge touching ones)

Three sites keep sorted disjoint ranges and merge on add. mesh_defined_add and mesh_defined_has sit at mesh-collective.c:630-658. spans_add and spans_meet sit at nccl-mesh.c:5248-5272. mesh-sparse.c:67-77 sorts intervals and merges touching ones. The first two track the same fact: which element ranges of a buffer are written. They can drift apart. The catalog's span functions answer overlap questions only.

### 36. Multiplicative-weights concurrent flow: lengths from loads, route every demand, add loads, repeat until a stop rule

- Kind: flow or packing solve. Copies: 2. Deepest: 1.
- Catalog: mesh_graph_arborescence (mesh-graph.h:51) serves only as mesh_trees_pack's tree step
- Form: Add one packing function to mesh-graph.h. Operands are the edge capacities, the demands, eps, a stop rule and a tree oracle called with the current lengths. mesh_solve passes a Dijkstra oracle. mesh_trees_pack passes mesh_graph_arborescence. It returns the distinct trees with their flows, through the find-or-add function.
- Sites: rdma/mesh-collective.c:786-835 mesh_solve; rdma/mesh-collective.c:899-937 mesh_trees_pack

Two solves share one loop family. mesh_solve at mesh-collective.c:786-835 sets edge lengths from loads, routes every demand and adds loads. mesh_trees_pack at :899-937 grows lengths until their weighted sum reaches 1. Only its tree step calls the catalog, through mesh_graph_arborescence at :904. The loop, the length update and the stop rule are written out in each. They differ in the tree step and the stop rule.

### 37. Round count T = floor(ln 2E / eps squared) + 1 of the multiplicative-weights solve

- Kind: flow or packing solve. Copies: 2. Deepest: 1.
- Catalog: none
- Form: Write the round count once, as rounds(edges, eps), beside the solver. mesh_solve_room and mesh_solve both call it. The eps guard stays with each caller.
- Sites: rdma/mesh-collective.c:763-766 mesh_solve_room; rdma/mesh-collective.c:784 mesh_solve

Two lines compute the round count. mesh_solve_room does it at mesh-collective.c:765. mesh_solve does it at :784. Both use ln(2E) over eps squared, plus 1. The guards differ. Line 764 replaces a bad eps with 1, and line 779 makes mesh_solve fail on a bad eps. mesh-plan.h says the trees made are at most the room. A change to one formula can break that.

### 38. Allocate the demand, parent and share arrays of a struct flow

- Kind: copy or free boilerplate. Copies: 2. Deepest: 1.
- Catalog: none
- Form: Add flow_make(f, room, nodes) beside flow_free at nccl-mesh.c:2685. flow_solve and routing_prepare call it with their own room. Keep the column data once and let the routing view point at it.
- Sites: rdma/nccl-mesh.c:3379-3382 routing_prepare; rdma/nccl-mesh.c:2688-2690 flow_solve

Two places allocate a struct flow by hand. routing_prepare does it at nccl-mesh.c:3379-3382. flow_solve does it at :2688-2690. Each makes three arrays and checks each. flow_free at :2685 is the only shared piece. routing_prepare then copies each tree column into r->flow at :3396-3397 and again into the view's trees and parents at :3397-3398. The flow and the view hold the same data twice.

### 39. Flatten an array of row arrays into one array, with a flag for each row it must free

- Kind: copy or free boilerplate. Copies: 2. Deepest: 1.
- Catalog: none
- Form: Add one header function that takes row pointers, row counts and an owned mask. It returns the flat array with its offsets. Each consumer passes its rows. It uses mesh_graph_rows for the offsets.
- Sites: rdma/nccl-mesh.c:3694-3735 order_transfers (frees owned rows at 3760); rdma/nccl-mesh.c:2572-2581 collective_steps

Two sites flatten rows of struct mesh_transfer into one array. order_transfers keeps a pointer and an owned flag for each row. It copies the rows at nccl-mesh.c:3728 and frees the owned ones at :3760. collective_steps at :2572-2581 builds the same flat array and offsets by hand. Neither calls a shared helper.

### 40. Count a program's segments as one more than the highest segment any part names

- Kind: other. Copies: 2. Deepest: 1.
- Catalog: none
- Form: Export mesh_program_segments(const struct mesh_program *) from mesh-plan.h. collective_steps calls it and compares the result with the rank count.
- Sites: rdma/mesh-collective.c:541-546 mesh_program_segments (static); rdma/nccl-mesh.c:2644-2649 collective_steps

Two places count segments. mesh_program_segments at mesh-collective.c:541-546 is static. collective_steps at nccl-mesh.c:2644-2649 has the same body. If the program sets segments, use it. Else take the highest part segment plus one. The function is not in mesh-plan.h, so collective_steps wrote it out again.

### 41. Map reduction laws to a fold order (1 without reorder, 2 with reorder only, else 0)

- Kind: other. Copies: 2. Deepest: 1.
- Catalog: none
- Form: Add fold_order(laws) beside the fold code in mesh-fold.h. fold_basis_validate and ncclMeshFoldPrepare both call it.
- Sites: rdma/nccl-mesh.c:3176 fold_basis_validate; rdma/nccl-mesh.c:3254 ncclMeshFoldPrepare

Two lines map laws to a fold order. fold_basis_validate does it at nccl-mesh.c:3176. ncclMeshFoldPrepare does it at :3254. Both give 1 without reorder. Both give 2 with reorder but no reassociate. Both give 0 otherwise. This is a weak match.

### 42. Ask whether a report holds an endpoint toward a node on a wire

- Kind: set or bitset. Copies: 2. Deepest: 1.
- Catalog: none
- Form: Add report_holds(report, node, wire) in nccl-mesh.c, or one scan with a predicate operand. fabric_discover passes (y, x->node, u->wire) at line 951 and (x, y->node, v->wire) at line 965.
- Sites: rdma/nccl-mesh.c:949-952 fabric_discover; rdma/nccl-mesh.c:961-967 fabric_discover

fabric_discover writes one match twice. The loop at nccl-mesh.c:949-952 tests node and wire in one direction. The loop at lines 961-967 tests the same pair with x and y swapped. The inner loop at line 965 tests node and wire for any match. Each loop is short, so this is a weak match.

### 43. Test whether a route pointer lies in bank 0 of the catalog entries, by looping over the bank

- Kind: other. Copies: 2. Deepest: 1.
- Catalog: none
- Form: Add route_supplied(catalog, route), which returns 1 when route lies in bank 0. Both places call it. A pointer range test would also do.
- Sites: rdma/nccl-mesh.c:3029-3030 routing_prior; rdma/nccl-mesh.c:3045-3046 routing_prior

routing_prior loops over bank 0 to test one route at nccl-mesh.c:3029-3030. It writes the same loop again at :3045-3046. Bank 0 is one array. A pointer range test or one helper would do. This is a weak match.

## The merger's summary

Eight reports merged into 43 groups. The groups hold 340 sites, and each site is listed once per group. I dropped no finding. I read the code for the climb, Dijkstra, pair-scan, bit-set, row-walk, overlap and offset sites, and for the comparator definitions. Each is a hand loop. None is only a call to a catalog function. The real catalog calls stay out of the groups. They are mesh_graph_climb at mesh-collective.c:109, mesh_graph_ancestors at :353, mesh_graph_arborescence at :904, and the span, forward, row, cut_at and frontier calls in nccl-mesh.c at lines 2822, 4239, 4292, 4306, 4350, 4433 and 4506. Lines are those of ecdb4f2e, which I read with git show. The worktree is at a88732ac, two commits later, and only nccl-mesh.c differs. Its lines past 2780 have moved. For example, consumption_order_compare is at 3605 there. One reader cited six nccl-mesh.c comparators 40 to 48 lines off. I used the ecdb4f2e lines. At a88732ac, route_tree_counts and routing_prepare read trees by tree_first. That removes the scan at the old route_tree_counts site. solved_steps still scans. The catalog itself repeats primitives. mesh-graph.c holds three comparators, five offset builds, two binary searches and one sort-and-drop. mesh-graph.h has no compare, group, run, row-offset, lower-bound or shortest-path function. Order follows copies times deepest depth. Ties go to code that is hot in preparation. The LSTM plan comments name collective_steps, write_steps and the key scans as hot. Copies equals the sites listed. Some listed sites are homes or references, flagged in the site text: grow, mesh_log, mesh_packet_next, route_demand_sorting and the assignment.py dict. Low-confidence groups are the units, hop walk, edge range, laws, endpoint and bank groups. The comparator group is wide on purpose. One header function serves all 39 sites.
