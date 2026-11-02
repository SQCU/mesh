# Permissions in the compiler layers (2026-11-02)

The operator's rule, 2026-11-02: choosers know pages and never pointers. Effectors know pointers and pages and never
decide. "This system of constraint is both performant and memory safe *for the same shared single reason*." Final
runtime code may read literal pointers, but a pointer enters the data flow only at the final lowering, as the
dereference of a page-table index.

This inventory lists the sites in the C and Objective-C compiler layers that break the rule. It changes no code. Every
row is in [design/data/permissions-compiler-2026-11-02.json](data/permissions-compiler-2026-11-02.json). Each reader's
rows are in [design/data/permissions-compiler-2026-11-02/](data/permissions-compiler-2026-11-02/).

## Method

The files were read whole, by static reading only, at origin/main 212f8e8e. nccl-mesh.c, mesh-collective.c,
mesh-plan.h, mesh-graph.*, metal-page-effects.* and nccl.h are unchanged since 2a336eec. mesh-tensor-layout.* and
mesh-index-relation.* were read at 502f02c9, after that day's layout change. The bridge, the kernels, the recorder and
the Python and ObjC++ frontend are other inventories.

A row is one function's use of one abstraction for one purpose. It names the scope (chooser or effector), the
violation, the abstraction, the problem the code handles and the page-map operation that answers it. There are five
violations:

- **chooser holds pointer**: a chooser computes or decides with a pointer, a byte offset into storage, a (buffer,
  offset) pair, a region offset or `contents`.
- **chooser picks a map by hand**: a ring, `% instances`, slot × stride, an in-place landing, a fixed generation or a
  numbered scratch buffer.
- **chooser has access**: a chooser scope imports, receives, stores or can call a pointer-yielding API or a pointer
  type, even where it does not decide with it yet.
- **effector decides**: run-time code chooses a path, a wait, a threshold or a placement.
- **effector computes a map**: run-time or generated code evaluates an address map: div and mod by a layout extent,
  `mesh_storage_N`, `mesh_index`.

A layout that maps logical element indices to virtual coordinates is the target model's own piece. It is not a row.
Graph algorithms over vertices and items are not rows.

## Counts

| Problem | Rows | Holds pointer | Map by hand | Has access | Effector decides | Effector computes a map |
|---|---|---|---|---|---|---|
| progress-words | 51 | 46 | 3 | 2 | 0 | 0 |
| resolve-backing | 37 | 30 | 0 | 7 | 0 | 0 |
| dma-spans | 34 | 28 | 0 | 6 | 0 | 0 |
| operand-identity | 29 | 19 | 4 | 6 | 0 | 0 |
| slot-selection | 25 | 0 | 25 | 0 | 0 | 0 |
| alias-overlap | 20 | 18 | 1 | 1 | 0 | 0 |
| operand-geometry | 20 | 19 | 0 | 1 | 0 | 0 |
| scratch-placement | 19 | 6 | 11 | 2 | 0 | 0 |
| war-reuse | 18 | 9 | 9 | 0 | 0 | 0 |
| bind-backend | 18 | 8 | 0 | 10 | 0 | 0 |
| identity-by-address | 17 | 17 | 0 | 0 | 0 | 0 |
| in-place | 16 | 6 | 10 | 0 | 0 | 0 |
| lease-lifetime | 14 | 13 | 0 | 1 | 0 | 0 |
| runtime-map | 12 | 1 | 0 | 0 | 0 | 11 |
| host-data | 11 | 11 | 0 | 0 | 0 | 0 |
| header-scope | 7 | 0 | 0 | 7 | 0 | 0 |
| effector-decides | 1 | 0 | 0 | 0 | 1 | 0 |
| all | 349 | 231 | 63 | 43 | 1 | 11 |

| Reader | Files | Rows |
|---|---|---|
| nm1 | rdma/nccl-mesh.c 1-4100, rdma/nccl.h | 82 |
| nm2 | rdma/nccl-mesh.c 4100-6140 | 114 |
| nm3 | rdma/nccl-mesh.c 6140-7921 | 60 |
| planner | rdma/mesh-collective.c, rdma/mesh-plan.h, rdma/mesh-graph.c, rdma/mesh-graph.h | 17 |
| effects | rdma/metal-page-effects.m, rdma/metal-page-effects.h | 55 |
| layout | rdma/mesh-tensor-layout.m/.h, rdma/mesh-index-relation.m/.h | 21 |

## The problems and their one answer

Each problem gets one answer in the joint change. The groups are in order of rows.

### progress-words: 51 rows

**Problem.** Locating progress, ready, summed, copied, coverage, failure and count words by byte offsets, region offsets
or pointer differences.

**Answer.** Words are pages of the same map. A word layout (tile, frame, channel -> word index) is composed with the
page map. The final lowering emits the literal word address.

Sites: `nccl-mesh.c:2090 struct write_binding`, `nccl-mesh.c:2143 tile_offset`, `nccl-mesh.c:2148 tile_extent`,
`nccl-mesh.c:2292 struct call`, `nccl-mesh.c:3563 struct message`, `nccl-mesh.c:3581 struct schedule`, `nccl-mesh.c:3608
copied_word`, `nccl-mesh.c:3868 struct issue`, and 43 more.

### resolve-backing: 37 rows

**Problem.** Turning a pointer or a (buffer, offset) back into region pages, runs or offsets: lease_at, lease_offset,
data_off, buffer.contents + offset into ncclMeshRegionRuns.

**Answer.** No inverse in the compiler. The allocator hands out the page table with the lease. The backing is the image
of the map, read from the table. Region offsets are dropped.

Sites: `nccl-mesh.c:1637 lease_runs`, `nccl-mesh.c:1651 lease_page_offset`, `nccl-mesh.c:1655 view_row`,
`nccl-mesh.c:1766 composed_order`, `nccl-mesh.c:1800 lease_offset`, `nccl-mesh.c:1882 stream_page`, `nccl-mesh.c:2071
address_of`, `nccl-mesh.c:2367 in_region`, and 29 more.

### dma-spans: 34 rows

**Problem.** Building DMA spans, SGEs, packets and bridge tables from region offsets, backing runs and casts into table
memory.

**Answer.** Specialization at the final lowering. An entry's spans are the image of its page range, coalesced into runs.
Cuts are the image's page boundaries joined with the declared logical cuts. Tables are pages of the map written with
page indices.

Sites: `nccl-mesh.c:3560 struct message`, `nccl-mesh.c:3856 struct issue`, `nccl-mesh.c:145 struct channel`,
`nccl-mesh.c:1373 fabric_realize`, `nccl.h:977 ncclMeshTransport`, `nccl-mesh.c:4143 landing_runs`, `nccl-mesh.c:4236
place_group`, `nccl-mesh.c:4253 place_group`, and 25 more.

### operand-identity: 29 rows

**Problem.** Naming an operand's storage: where, ncclMeshBuffer and ncclMeshBufferSlots fields in chooser records.

**Answer.** An operand is a page-map value: a page table id and the map invocation -> (virtual page -> physical page),
with a position in the operand's logical coordinates. Choosers compose these values. The final lowering specializes them
into (region buffer, page index x page bytes) or a pointer literal.

Sites: `nccl-mesh.c:2066 struct where`, `nccl-mesh.c:2080 struct write_binding`, `nccl-mesh.c:2265 struct call`,
`nccl-mesh.c:3559 struct message`, `nccl-mesh.c:153 struct ncclComm`, `nccl.h:245 ncclMeshBuffer`, `nccl.h:875
ncclMeshCall`, `nccl-mesh.c:4137 place_data`, and 21 more.

### slot-selection: 25 rows

**Problem.** Choosing which storage an invocation uses by hand: % instances, % occurrences, slot x stride, displacement,
a fixed generation.

**Answer.** Lifetime allocation produces the map invocation -> (virtual page -> physical page). The final lowering
specializes it per invocation into literal page indices. No modulus remains in any interface.

Sites: `nccl-mesh.c:2111 write_span_index`, `nccl-mesh.c:2121 write_binding_slot`, `nccl-mesh.c:3701 schedule_group`,
`nccl.h:279 ncclMeshStream`, `nccl.h:773 ncclMeshInvocationIndex`, `nccl.h:788 ncclMeshBufferSlots`, `nccl-mesh.c:4129
place_data`, `nccl-mesh.c:4251 place_group`, and 17 more.

### alias-overlap: 20 rows

**Problem.** Knowing when two operands, accesses or steps share bytes, decided by address, byte range or buffer number.

**Answer.** Intersect the two maps' images. The preimage of the intersection in each operand's logical coordinates is
the dependency. Buffer numbers and byte ranges do not take part.

Sites: `nccl-mesh.c:2393 binding_meets`, `nccl-mesh.c:3651 meet`, `nccl.h:735 ncclMeshAccess`, `nccl-mesh.c:4868 zinth`,
`nccl-mesh.c:5033 reduction_fold`, `nccl-mesh.c:5267 destination_effects`, `nccl-mesh.c:5362 physical_order`,
`nccl-mesh.c:5378 binding_relations`, and 12 more.

### operand-geometry: 20 rows

**Problem.** Byte positions of views, segments, rows, frames and pieces inside choosers.

**Answer.** Compose layouts in the operand's logical (element) coordinates with the page map. Bytes appear only at the
final lowering.

Sites: `nccl-mesh.c:2097 struct write_binding`, `nccl-mesh.c:2168 write_selection`, `nccl-mesh.c:2350 view_at`,
`nccl-mesh.c:3595 effect_first`, `nccl-mesh.c:3705 schedule_group`, `nccl-mesh.c:3921 repeat_of`, `nccl-mesh.c:4130
place_data`, `nccl-mesh.c:4378 send_dependencies`, and 12 more.

### scratch-placement: 19 rows

**Problem.** Storage for partials, relays, landing areas, generations and an issue's own pages, placed at byte offsets
or as numbered buffers.

**Answer.** These are virtual pages of the function's map with lifetimes. Lifetime allocation assigns their physical
pages. Numbered buffers and byte cursors are deleted.

Sites: `nccl-mesh.c:2087 struct write_binding`, `nccl-mesh.c:2226 struct steps`, `nccl-mesh.c:2289 struct call`,
`nccl-mesh.c:2548 generation_buffer`, `nccl-mesh.c:2573 bind_routed_steps`, `nccl-mesh.c:2673 partial_split`,
`nccl-mesh.c:3856 struct issue`, `nccl-mesh.c:4087 slot_laid`, and 11 more.

### war-reuse: 18 rows

**Problem.** Ordering a write to storage after every reader of its old contents: a landing into bytes a SEND reads, an
overwrite after GPU and DMA readers, a cyclic instance taken to mean the same pages.

**Answer.** Lifetime allocation over the composed map. A physical page takes a new virtual page only after the last
reader of the old one in the dependency order. The WAR edge is the preimage of the reused physical page.

Sites: `nccl-mesh.c:4473 landing_lists`, `nccl-mesh.c:4542 landing_readers`, `nccl-mesh.c:4596 landing_readers_of`,
`nccl-mesh.c:4610 landing_afters`, `nccl-mesh.c:5087 arithmetic_met`, `nccl-mesh.c:5097 arithmetic_readers`,
`nccl-mesh.c:5148 arithmetic_readers_of`, `nccl-mesh.c:5169 arithmetic_targets`, and 10 more.

### bind-backend: 18 rows

**Problem.** Making a Metal buffer, an argument entry or a host pointer for a kernel or CPU leaf.

**Answer.** Effector: permitted at the final lowering only, with a decided page index as its input. No chooser sees the
buffer or the pointer.

Sites: `nccl-mesh.c:1890 lease_buffer`, `nccl-mesh.c:4064 struct piece`, `nccl.h:708 ncclMeshPageTableBinding`,
`nccl.h:722 ncclMeshPageTableAddress`, `nccl-mesh.c:5749 source_group`, `nccl-mesh.c:5824 as_metal`, `nccl-mesh.c:6699
zoblimp`, `nccl-mesh.c:7424 ncclMeshPageTableCreate`, and 10 more.

### identity-by-address: 17 rows

**Problem.** Identifying, deduplicating or ordering objects (calls, leases, buffers, words) by their host address.

**Answer.** Ordinals and page table ids. Equality of storage is equality of page images.

Sites: `nccl-mesh.c:3619 consumption_order_compare`, `nccl-mesh.c:3742 write_order`, `nccl-mesh.c:714 struct lease`,
`nccl-mesh.c:4467 overwrite_met`, `nccl-mesh.c:5021 zitem_order`, `nccl-mesh.c:5097 arithmetic_readers`,
`nccl-mesh.c:6020 ncclMeshCommSourceInputs`, `metal-page-effects.m:18 PageOperandIdentity`, and 9 more.

### in-place: 16 rows

**Problem.** Avoiding a copy by using the send operand's storage as the result's: reading the send buffer in place,
landing in recvbuff, complement copies of the ranges no step writes.

**Answer.** Lifetime allocation decides the sharing. Where each page's last read of the send value precedes the first
write of the result, the result's virtual page maps to the same physical page. The complement is the preimage of the
result pages no step writes.

Sites: `nccl-mesh.c:2232 struct steps`, `nccl-mesh.c:2273 struct call`, `nccl-mesh.c:3589 original_send`,
`nccl-mesh.c:3663 forwarded`, `nccl-mesh.c:4134 place_data`, `nccl-mesh.c:4780 result_of`, `nccl-mesh.c:4792
bind_calls`, `nccl-mesh.c:5066 zabbadoo_plan`, and 7 more.

### lease-lifetime: 14 rows

**Problem.** Deciding when a returned or owned lease's pages may be reused after their DMA and GPU readers, keyed by
address.

**Answer.** Lifetime allocation keyed by page identity. The retirement ledger lists page tables and their pages, not
addresses.

Sites: `nccl-mesh.c:1670 lease_give`, `nccl-mesh.c:1696 lease_take`, `nccl-mesh.c:1814 struct owned_lease`,
`nccl-mesh.c:1842 owned_begin`, `nccl-mesh.c:4229 place_group`, `nccl-mesh.c:4971 issue_free`, `nccl-mesh.c:5903
source_coverage`, `nccl-mesh.c:6020 ncclMeshCommSourceInputs`, and 6 more.

### runtime-map: 12 rows

**Problem.** Generated or run-time code that evaluates an address map: div/mod by a layout extent, % instances,
mesh_storage_N, mesh_index.

**Answer.** Specialize at the final lowering: a pointer literal, or base + affine offset, read from the page table.

Sites: `metal-page-effects.m:879 MetalPageFields compose`, `metal-page-effects.m:971 PageCoverageSource`,
`metal-page-effects.m:1014 PageCoverageSource`, `metal-page-effects.m:1043 PageCoverageSource`,
`metal-page-effects.m:1385 PageObservations / MetalPageEffectsSource`, `mesh-tensor-layout.m:321
TensorExpressionSource`, `mesh-tensor-layout.m:1384 TensorStorageDeclaration`, `mesh-tensor-layout.m:2008
declaration:origin:`, and 4 more.

### host-data: 11 rows

**Problem.** A chooser reading or writing payload or metadata through a pointer: scalars, identities, copies, fills,
progress initial values.

**Answer.** A data function with page operands, decided by the compiler and run by the effector at the final lowering.

Sites: `nccl-mesh.c:2282 struct call`, `nccl-mesh.c:123 struct redop`, `nccl.h:792 ncclMeshDemandBindings`,
`nccl-mesh.c:4201 landing_progress`, `nccl-mesh.c:4830 reduction`, `nccl-mesh.c:5515 scalar_page`, `nccl-mesh.c:5903
source_coverage`, `nccl-mesh.c:6020 ncclMeshCommSourceInputs`, and 2 more.

### header-scope: 7 rows

**Problem.** A chooser's headers or translation unit bring pointer types and pointer-yielding calls into scope: Metal.h,
nccl.h, mesh-call.h, the bridge's layout.h, the static helpers of nccl-mesh.c.

**Answer.** Split the headers. A chooser header declares page-map types only. Pointer types and pointer-yielding calls
live in the final lowering's header and translation unit, which no chooser includes.

Sites: `nccl.h:1040 (file scope) ncclMemAlloc`, `nccl-mesh.c:2071 (file scope) lease_at`, `mesh-plan.h:5 header include
of etc/bridge/layout.h`, `mesh-collective.c:1 file include of mesh-call.h`, `mesh-tensor-layout.h:2 (file scope)`,
`mesh-tensor-layout.m:1 (file scope)`, `mesh-index-relation.h:2 (file scope)`.

### effector-decides: 1 row

**Problem.** Run-time code that chooses a path, a wait, a threshold or a placement: the kernel scan that picks how far a
completed prefix reaches before it publishes.

**Answer.** The chooser compiles the publication points from the dependency order. The kernel stores at those points and
decides nothing.

Sites: `mesh-graph.h:128 MESH_GRAPH_PREFIX_METAL`.

## One representation

Every answer above uses one value: the address map invocation → (virtual page → physical page), composed with layouts.
Lifetime allocation produces it. Choosers compose it, take images and preimages of it, and order by it. The final
lowering specializes it into literal page indices, pointer literals or base + affine offset. The groups depend on each
other, so the change lands at once:

1. The compiler's interface types in nccl.h carry page-map operands instead of `ncclMeshBuffer`.
2. Lifetime allocation replaces numbered scratch buffers, cyclic slots, generations and in-place choices.
3. Hazards, aliases and step dependencies come from intersections of images and their preimages.
4. Progress, coverage and failure words are pages of the same map, located through a word layout.
5. One final-lowering module holds every pointer-yielding call. It takes page indices only. No chooser header or
   translation unit can reach it.
6. Emitted kernels read literals or base + affine offset. No div, mod or table lookup evaluates a map at run time.

## nccl.h interfaces that become incompatible

These change, because each hands a chooser a pointer, a byte offset or a hand-picked map:

- `ncclMeshBuffer` (nccl.h:245), the (`void *buffer`, `size_t offset`) pair. Every compiler interface that holds one
  changes with it: `ncclMeshPageCoverage` (250), `ncclMeshPageEffect.word` and `.violation` (262),
  `ncclMeshAccess.storage` (735), `ncclMeshCall.source_ready` (910) and `ncclMeshTransport.spans` (981).
- `ncclMeshBufferSlots` (788) and `ncclMeshInvocationIndex` (773): slot × stride and `% instances`. They become a
  page-map operand, a page table and its invocation → page map. `ncclMeshCall.bindings` (899) and
  `ncclMeshCoverage.active` and `.expected` (811) change with them.
- `ncclMeshStream.invocation` (285). It indexes the solved map, not a modulus.
- `ncclMeshTileLayout` (739): byte strides and `% instances`. It becomes an element word layout, composed with the
  words' page map.
- `ncclMeshDemandBindings.values` (793), a `const void *const *`.
- `ncclMeshPageTableCreate` (695) takes `const void *local`. The allocator hands out the table with the lease instead.
- `ncclMeshRegionRuns` and `ncclMeshRegionRun` (700-701) give region offsets. `ncclMeshPageTableBacking` already gives
  the page runs.
- `ncclMeshPageTableBinding` (708) and `ncclMeshPageTableAddress` (722) yield pointers. They move to the final
  lowering's header.
- nccl.h as a whole declares `ncclMemAlloc` and `ncclMeshMemBuffer` to every compiler file that includes it. It is split
  into the data programs' NCCL header and a compiler header of page-map types.

These already have the target form and stay: the `ncclMeshPageTable_t` handle, `ncclMeshPageTableShape`,
`ncclMeshPageBacking`, `ncclMeshPageMask`, `ncclMeshPageSelection` and the signature of `ncclMeshPageTableCompose` (its
implementation remaps by address and changes), `ncclMeshLayout` in element coordinates, `ncclMeshSourceInput`, the
logical fields of `ncclMeshPageEffect` (call, first, bytes, frame), and the routing records. NCCL's own API keeps its
pointers for data programs: `ncclMemAlloc`, `ncclMemFree`, `ncclCommRegister`, the `ncclScalarDevice` scalar of
`ncclRedOpCreatePreMulSum` and `ncclMeshMemBuffer`. Only the compiler stops seeing them.

## Findings to read apart

- **Peers learn this rank's region offsets.** `write_prepare` exchanges its storage runs (offset − `data_off`) with
  every rank at nccl-mesh.c:7319-7333, and its registration phases in the report at 7284-7301. The comment at 6960 says
  no process pointer crosses a socket. No process pointer does, but region offsets do, so a peer's preparation depends
  on this rank's physical placement.
- **A pointer enters the data flow during preparation.** `ncclMeshPageTableCompose` reserves a view with `mmap` and
  remaps each run with `mach_vm_remap` at `M + run.offset` (nccl-mesh.c:7547-7564). Later choosers read the composed
  view's address.
- **Order depends on host addresses.** `write_order` (3748), `consumption_order_compare` (3619), `overwrite_met`
  (4470), `zitem_order` (5024) and `arithmetic_readers` (5117) compare `struct call` addresses.
  `MetalPagePublicationDomains` orders domains by region offset. `MetalPageEffectsRebind` keys by `valueWithPointer`.
  `MetalPageFields` dedupes by a `%p` key. `TensorExpressionPiece` caches by the expression's address. Both ends of a
  cable lay their lists alike only while these orders agree.
- **The host writes progress words through `metal_contents`** at nccl-mesh.c:5964 and 6119, during preparation.
- **One kernel text decides.** `MESH_GRAPH_PREFIX_METAL` (mesh-graph.h:128) scans completion bits at run time and
  picks how far to publish.
- **Emitted kernels evaluate maps at run time.** `mesh_index` and `mesh_storage_N` divide and take remainders by
  layout extents (7 rows in mesh-tensor-layout.m). The cut, announcement and threshold tables of `PageCoverageSource`
  and `MetalPageEffectsSource` are looked up per access (4 rows in metal-page-effects.m). Their values are known at
  preparation. The affine contribution path at metal-page-effects.m:996-1013 already emits literals.
- **Clean.** mesh-index-relation.m has no row: its relation algebra works in pages and positions. The routing,
  solver and graph algorithms have no row except `mesh_graph_settle`'s `void **items`, whose one caller passes lease
  addresses. The reader's key `new:header-scope` is the group header-scope here.
