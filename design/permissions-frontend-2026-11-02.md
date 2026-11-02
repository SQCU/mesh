# Permissions in the frontend, 2026-11-02

The rule is the operator's, of 2026-11-02. Choosers know pages and never pointers. Effectors know pointers and pages and
decide nothing. A pointer literal enters the data flow only at the final lowering, as a dereference of a page-table
index. Generated code that evaluates an address map at run time is a lowering that did not specialize.

This inventory covers torch_mesh's Python and Objective-C++ (`rdma/torch-mesh/torch_mesh/*.py`, `_stream.mm`,
`allocator.mm`), `rdma/mesh_relation.py`, `rdma/mesh_graph.py`, `examples/lstm.py`, `examples/lstm-service.py` and
`examples/chain.py`, at 212f8e8e. The rows are in
[design/data/permissions-frontend-2026-11-02.json](data/permissions-frontend-2026-11-02.json). The C compiler layers
and the bridge, kernels and recorder have their own inventories.

## Method

Every row was found by reading the source. Nothing was built or run. A row is one function, or one contiguous part of
one, that holds or computes an address, a byte offset or a byte width, or that fixes a map by hand, or that can reach a
pointer-yielding call. Each row names its scope, its violation, the abstraction it uses, the problem it handles and the
page-map operation that replaces it.

The violations are five. "Chooser holds pointer" is an address, a byte offset, a `(buffer, offset)` pair or a byte
width in a chooser. "Chooser picks a map by hand" is a ring, a generation, a slot stride, an invocation number or an
in-place landing written by the program or the library. "Chooser has access" is a chooser scope that imports, receives
or can call a pointer-yielding API without deciding with it yet. "Effector decides" is evaluation code that chooses
placement or storage. "Effector computes a map" is generated or native run-time code that evaluates a map the final
lowering could have specialized. Six rows are marked "none (effector)". They mark where the permitted boundary is now.
Their pointers come from torch storage or `page_backing`, not from a page table.

## Counts

There are 166 rows. 87 are "chooser holds pointer", 36 "chooser has access", 19 "chooser picks a map by hand", 14
"effector computes a map", 4 "effector decides" and 6 "none (effector)".

| Group | Problem the code handles | Rows | Replacement |
|---|---|---|---|
| P4 | relate producer and consumer selections of different scalar types | 26 | typed layouts composed onto one page map; the width multiplies only at the final lowering |
| P14 | reach layouts, device facts and native helpers that live beside pointer-yielding calls | 23 | a pointer-free module for choosers; `_stream`'s pointer calls reachable from the final lowering only |
| P2 | start an argument at a byte other than its layout's first element | 15 | the origin is a layout coordinate over the page map; the lowering emits `page_table[page] + offset` |
| P9 | find the region pages a tensor or an address lies in | 14 | region tensors carry their page map from the lease; nothing maps an address back |
| P12 | address a view's elements inside a kernel | 14 | the final lowering specializes the map into literals or base + affine offset |
| P1 | hand an operand to the native compiler or recorder | 10 | pass the page map and the layout in elements |
| P7 | storage for repeated occurrences of one call | 10 | an invocation -> (virtual page -> physical page) map solved from liveness |
| P10 | give a host, CPU or provider entry its operands | 10 | declare the entry over page maps; the lowering writes its pointers |
| P6 | operate on the same bytes at another scalar width | 9 | another layout over the same page map |
| P8 | update recurrent state each step without a new tensor | 8 | functional steps and collectives; the invocation page map decides any sharing |
| P5 | decide which values share an allocation, and its size | 8 | aliasing is page-map image intersection; extent is the map's domain |
| P11 | name the part of a collective's operand a segment covers | 7 | segments are layouts over the page map, in elements |
| P15 | give each value region storage, and constants a place | 5 | leases taken once from the compiled page map at load |
| P3 | meet an API's alignment and capacity; decide copy or direct use | 4 | alignment and capacity are constraints of the page-map solve |
| P13 | describe a torch view's position in its allocation | 3 | a layout composed onto the lease's page map |

P4 and P6 together are 35 rows. They exist because pages are bytes and values are typed. Once the dtype is a property
of the layout over a page map, no chooser needs a byte width.

P2, P3, P9 and P13 are 36 rows. They share one cause. A chooser's operand is a torch storage handle plus an offset,
not a page map. `TensorView` (operand.py:13-19) carries `storage`, an element offset and `binding_origin`.
`TensorOperand` (`_stream.mm`:546-559) recovers pages from the MTLBuffer and adds the byte origin.

P12 rows are in generated text: `mesh_index`, `mesh_storage_N`, `mesh_page_read` and `mesh_page_write`.
`native.py`:1120-1142 divides a byte map by the scalar width on every access.

## The program halves

The joint change has a program half. These are the in-place landings and the hand-picked generations.

- `examples/lstm.py` keeps one `h [L, rows, H]` (lines 33-35). `mine` is `h.narrow(2, first, HR)`. Every tick,
  `allgatherv_out(self.mine, self.h)` (line 55) lands the gather in `h`. Its own source `mine` is a slice of `h`. The
  next tick's layers read `h`. The cell writes `c` and `mine` in place with `copy_` (lines 10-15, 50).
- `torch_mesh/chain.py` keeps two generations: `empty(2, batch, ranks, R, K)` and `len(windows) % 2` (lines 139-149).
  Landings and reads are picked by `generations[generation, j]` (lines 210-235). Its slots are hand-picked too:
  `dict(instances=0, stride=bytes)` for inputs, `dict(instances=1, stride=0)` for the recurrence, an invocation counter
  and frame strides (lines 158-217).
- `examples/lstm-service.py` broadcasts `X` into `X` (lines 113-119).
- `examples/chain.py` draws through a fixed scratch window (lines 25-31).

The library halves accept these forms. `partition.py`:96-105 defines `allgatherv_out`. `declarations.py` takes `out` as
the result (485-509), declares the out-variants (579-580, 592-593) and turns the landing into an effect on an input
(850-875, 1208-1210). `backend._slots` (118-122) passes instances, stride and invocation to the compiler.
`Plan.append` (page_effects.py:406-415) lowers each occurrence at its hand-counted invocation.

## User-facing interfaces that change

- `torch.ops.torch_mesh.allgatherv_out` and `all_gather_into_tensor_out` are removed. `allgatherv` returns the gathered
  value.
- `torch_mesh.address(t)` and `torch_mesh.records()` stop returning addresses. Residency is having a page map.
- `TensorView` loses `binding_origin`, `storage_offset()`, `element_size()`, `storage_as()`, `tensor()` and
  `write_layout()`. It becomes a page map and a layout.
- `backend.Call.bindings` and `prepare_fragments` take page-map operands, not tensors. `prepare_writes` loses
  `frame_strides`.
- Python no longer supplies `ncclMeshBufferSlots.instances`, `stride` or `invocation`.
- `_stream.page_backing`, `host_page_backing`, `page_storage`, `occurrence_source_views`, `occurrence_publications`
  and `PageEffectsPlan.bind`'s page tensors leave the chooser's reach. The page-effects solve returns page-map operands.
- `engines.host(t)` is replaced by declared host fills and reads.
- `Function.prepare(repetitions)` and `Executions.run` give way to the invocation domain. The schedules recorded in
  advance no longer fix how many runs there can be.
- `torch_mesh/chain.operands` declares one recurrent value per window. It has no generations argument to give.

## Clean modules

`mesh_graph.py`, `ownership.py`, `reduction.py`, `scan.py`, `sparse.py`, `streamed_relations.py`, `configuration.py`,
`random.py`, `shader.py` and `attention.py` hold no address and pick no map. `mesh_relation.py` holds no data address.
Its one row is the isl handle that crosses into `_stream` as an integer (mesh_relation.py:241-263). `streamed.py`'s RING
is a producer-consumer ring in threadgroup memory inside one kernel. It is not a tensor's address map and has no row.
