# Pointer and offset lowering, 2026-10-28

## Why this inventory exists

A codebase with no eager execution names no buffer at the call. Every operand is declared first. Preparation assigns its
storage (pages) and lowers every access once, from logical shape, strides and element offset to page indices and then to
DMA spans and recorded bindings. After that, raw addresses belong in two places only: the allocator's leases, which back
the page tables, and the compiled tables the run indexes (design/eager-execution.md, design/pages-and-functions.md).

Any other code that computes, passes or recovers a pointer or byte offset is either re-deriving at run time what
preparation fixed, or a frontend lowering layout by hand. Each such site is one more copy of the address mapping, and
copies disagree:
- The 2026-10-28 hourlies failed on typed WRITE counts against byte views, and on rows, spans and typed forms each
  reporting its own count and type for the same bytes.
- A dtype cannot name sub-byte values (int4, int2, fp4).

R11 (design/runtime-interface-removal-2026-10-25.md) made the public WRITE contract page tables and masks, but left
"local tensor addresses resolved at the frontend boundary". This is that boundary's inventory.

This is a read-only map at mesh origin/main f5357487 and metal-microbench origin/main bfb93c25; its line numbers drift.
8f1fa367 has since removed the WRITE's type from the declaration, plan, report and issue (most of G5 below).

## Summary

| # | Operation | Sites | Needs | Known at |
|---|---|---|---|---|
| G1 | tensor/array → address, or (MTLBuffer, storage_offset×element_size) | 18 | storage identity, element offset, element bits | declaration |
| G2 | view → base + shift / rows / byte spans / byte reaches | 24 | shape, strides, element offset, element bits | declaration (`MetalTensorLayout` already holds them in element units) |
| G3 | address → lease → page index, plus checks | 28 | storage identity, element offset | allocation (the lease), recovered by a linear address search |
| G4 | host pointer ↔ Metal buffer + offset | 17 | storage identity | allocation |
| G5 | dtype → count or bytes on a WRITE | 34 | element bits only | declaration |
| G6 | collective (non-WRITE) pointers passed at issue | 19 | storage identity, extent; a dtype only to reduce | declaration (`ncclMeshCall.bindings`) |
| G7 | recorder / compiled-descriptor byte offsets | 17 | outputs of lowering | the allowed layer |
| G8 | allocator / lease layer | 9 | none | the allowed layer |

One preparation repeats the lookup several times:
- a single `mesh_compile_*` call runs the address-to-lease search twice in `tensor_pages`;
- `write_prepare` and `write_prepare_spans` look up the same binding again;
- `ncclMeshWriteExecute` turns the plan back into a pointer, and `admit` converts that pointer again.

Sub-byte types are blocked at `SIZE[datatype]`, `itemsize`, `elementBytes` and the `torch.uint8`/`ncclUint8` stand-ins.

## The single lowering

The lowering lives in the library, beside `page_spans` in nccl-mesh.c, and is declared in nccl.h:

```c
typedef struct { ncclMeshPageTable_t storage; uint32_t element_bits, rank;
  const uint64_t *shape; const int64_t *strides; uint64_t offset;   /* elements */
  const ncclMeshAxisMap *maps; } ncclMeshLayout;
ncclResult_t ncclMeshLayoutPages(const ncclMeshLayout*, ncclMeshPageSelection *out);
```

- **Storage.** The page table is made once, from the lease handle, when the storage is allocated, not from an address.
- **Algorithm.** The lowering runs `mesh_view_layout` / `write_view_rows` arithmetic in bits. It keeps affine rows compact
  (start page, first byte, row, rows, pitch) and otherwise emits page masks. It rejects span edges that do not fall on a
  byte.
- **Callers.** A WRITE takes only a table and a selection, with bytes = numel × bits / 8 and no dtype. Each frontend
  needs one thin helper that fills `ncclMeshLayout`: torch's `TensorOperand`, and numpy's base, offset and strides.

It replaces:
- G1, G2, G5 and G6's bindings in full;
- G3 and G4, except `lease_at`/`lease_buffer` inside the allocator.

G7 keeps byte offsets, taken from the selection.

## Progress

Landed on mesh main:
- 8f1fa367: a WRITE has no type. Its declaration, plan, report and issue carry bytes (G5, library side).
- bf4f1454: `ncclMeshLayout` and `ncclMeshWriteLayoutPrepare`, a WRITE from abstract layout over its storage's page
  table, element bits included. mesh_mpi's WRITE methods take that path, with one position read in `_layout`; numpy
  has no storage offset (G1 #11-14, G2 #4-5).
- d13f13be: torch-mesh `ViewWrites` passes `TensorView.write_layout()`, a storage, element bits, shape, strides and
  offset, or a mapped view's element runs. `TensorView.transfer`, `mesh_compile_view` and `mesh_compile_spans` are
  deleted (G2 #6, #8, #19; G3 #6, #8).

- e327689a: a tiled fused write binds each rank's own tensor by its layout (`tensor_write`).
- 1370f15a: frame trains are layout (`ncclMeshLayout.frames`, `frame_stride` in elements): the fused ring, the chain
  and fused writes with repeats. `mesh_compile_write` is deleted.

- f1cdef43: migration runs are element views of each rank's own tensor. `fused.write_rows` and `fused_writes_impl`
  take no byte geometry, and the uint8 byte views in `_follow_bytes` are gone.

Open, in order:
3. *(landed, f1cdef43)* **Migration.** partition.py `_follow_bytes` narrows uint8 views at the byte runs `migration.compile_rows` computes
   (G2 #24, G1 #16-18). It should emit element layouts of the real source and destination views, after which
   `fused.write_rows` and `mesh_compile_rows` go.
4. *(landed: mesh 92af4f48, metal-microbench f16bdcb7; mesh_compile_rows and tensor_pages deleted after)* **mesh_rank.m** (metal-microbench): its crossings' `mesh_compile_rows` and fragment offsets (G2 #22-23, G4 #14-16).
   Its slots repeat at a byte stride per invocation (`CrossBinding.stride`). The layout entry takes that period from
   `ncclMeshWriteLayout.instance_pages`, in whole pages. The period is local, like frames, so it moves into
   `ncclMeshLayout` in elements. `ncclMeshWriteLayout` then keeps what every rank shares: tile bytes and invocations.
5. **G6, reconsidered.** `ncclAllReduce(sendbuff, recvbuff, ...)` takes pointers because it is NCCL's API, and C1
   (design/rubric-actionable.md) ranks higher the form that keeps NCCL's API and semantics as its interface. So an
   issue's pointers stay in the API, unread: a group of collectives is laid and placed for every slot of its declared
   bindings (`prepared`), its issue takes the slot of its stream's invocation, and `declared_as` holds it to its
   declaration's kind and counts (mesh 4ae7ed5f; `issue_same` deleted).
   - What remains eager is the frontends' per-call paths that restate a declaration at every call
     (design/duplication-review-2026-10-28.md, across-files 1).
   - Declared bindings in `ncclMeshCall.bindings` can move to layouts once those paths are gone.

## Inventory

Roots: mesh `rdma/` (C, ObjC, Python), `rdma/torch-mesh/torch_mesh/`, metal-microbench `mesh_rank.m`.

### G1. tensor/array → address or (buffer, byte offset): 18

Needs storage identity, element offset and element bits. All are in torch or numpy metadata at declaration.

1. _stream.mm `Group::where` 273-277: MPS gives {getMTLBufferStorage, storage_offset×element_size}; CPU gives data_ptr()
2. _stream.mm `encoded` (Dispatch) 615-631: bound, offsets and lengths from storage_offset×element_size and nbytes
3. _stream.mm `prepare_chain` 1364-1367: input/output base and pitch in bytes
4. _stream.mm `prepare_chain`/append 1393-1405: per-step offsets = base + t×pitch
5. _stream.mm `tensorPages` 677-683: storage().data_ptr().get(), or the MTLBuffer
6. _stream.mm `address` 990-1000: contents + storage_offset×element_size, checked with ncclMeshMemBuffer
7. _stream.mm `import_attributes` 1037-1039: raw from/to pointers for the import copy
8. _stream.mm `fused_writes_impl`/framed 1531-1538: byte offset bounds checked against storage nbytes
9. _stream.mm `workspace` 131-139: scratch MTLBuffer handed to the library
10. backend.py `_binding` 87-94: MPS untyped_storage().data_ptr() used as the MTLBuffer id, plus byte offset; CPU data_ptr()
11. mesh_mpi.py `at` 70-76: ctypes.data
12. mesh_mpi.py `Comm.Pages` 241-245: at(array) → ncclMeshPageTableCreate
13. mesh_mpi.py `Comm.Write_prepare` 269
14. mesh_mpi.py `Comm.Write_view_prepare` 310: ctypes.data + shift
15. assignment.py `declare` 20-25: {NULL, a.ctypes.data} bindings
16. partition.py `_in_region` 171-174: address(y) != 0
17. partition.py `_follow_bytes` 734: address(local) presence check
18. partition.py `_follow_bytes` 752-753: gcd of absolute addresses used as the vector width

### G2. view → base + shift / rows / spans / byte reaches: 24

Needs shape, strides, element offset and element bits. All are known at declaration.

1. nccl-mesh.c `mesh_view_layout` 5643-5680
2. mesh-write-compiler.h `mesh_compile_view` / `mesh_view_layout` 24-37
3. mesh.py `view_layout` 203-219
4. mesh_mpi.py `Comm.Write_view_prepare` 306-313
5. mesh_mpi.py `Comm.Write_rows_prepare` 286-295
6. operand.py `TensorView.transfer` 123-135: strides×size; first = offset + shift//size; spans scaled to bytes; base built with as_strided
7. operand.py `TensorView.storage_as` 67-72: nbytes//element_size
8. backend.py `ProcessGroupMesh.prepare_writes` 304-315: transfer() → ViewWrites
9. collective_views.py `_alias` 65-67: fragment → as_strided at an element offset
10. collective_views.py `lower` 101-102: empty alias
11. mesh-tensor-layout.m `reaches:base:elementBytes:` 723-735
12. mesh-tensor-layout.m `MetalTensorPages bind:elementBytes:` 757-760: shift by origin/bytes, asserts byte alignment
13. metal-recording.m `MetalViewReaches` / `MetalViewMappedReaches` 299-363
14. _stream.mm `TensorOperand` ctor 688-693
15. _stream.mm `Commands` ctor 733-741: {buffer, offset×element_size, count×element_size} plus reaches
16. _stream.mm `ScheduleBinding::host` 1609-1616
17. _stream.mm `TensorLayout` "spans" 1821-1825
18. _stream.mm `import_attributes` 1016-1026, 1045-1052
19. _stream.mm `ViewWrites::prepare` 1671-1686
20. _stream.mm `fused_allocate_ring` 1241-1249: slot stride in bytes
21. _stream.mm `prepare_chain` 1371-1381: stride in bytes
22. mesh_rank.m `cross_fragments` 672-679: offset += part.offset×element
23. mesh_rank.m `columns_prepare` 797-804: pitch×element, offset += first
24. partition.py `_follow_bytes` 739-770: uint8 views narrowed at byte run offsets

### G3. address → lease → page index / checks: 28

Needs storage identity and element offset. The lease is known at allocation.

1. nccl-mesh.c `lease_at` 1646-1657: linear search
2. nccl-mesh.c `lease_buffer` 1660-1670
3. nccl-mesh.c `ncclMeshMemBuffer` 1672-1678
4. nccl-mesh.c `ncclMeshPageTableCreate` 5548-5569
5. nccl-mesh.c `tensor_pages` 5623-5631: creates the table, then repeats where_of/address_of
6. nccl-mesh.c `mesh_compile_spans` 5633-5641
7. nccl-mesh.c `mesh_compile_rows` 5681-5690
8. nccl-mesh.c `mesh_compile_view` 5692-5697
9. nccl-mesh.c `mesh_compile_write` 5698-5706
10. nccl-mesh.c `write_prepare` 5437-5456: shifted base → address_of → in_region → lease_at; phase computed from the address
11. nccl-mesh.c `write_prepare_spans` 5314-5333: origin = address − region base
12. nccl-mesh.c `write_prepare_spans` 5343-5360: lease_at on the payload
13. nccl-mesh.c `write_plan_drop` 1897-1915
14. nccl-mesh.c `ncclMeshWriteSourceReady` 5714-5726
15. nccl-mesh.c `in_region` 2105-2109
16. nccl-mesh.c `operands_check` 2124-2140
17. nccl-mesh.c `place_group` 3112-3121
18. nccl-mesh.c `bind_retirements` 3164-3185: overlap test on addresses
19. nccl-mesh.c `present_group` 3253-3258
20. nccl-mesh.c `present_group` 3293-3296: ready word address
21. nccl-mesh.c `declared` 4891-4900: binding address → phase
22. nccl-mesh.c `plan_direct` 4101-4112: aliasing by address
23. mesh-tensor-layout.m `MetalTensorPages host:` 741-751: origin = address − contents
24. mesh-tensor-layout.m `MetalTensorPages storage:` 752-755
25. _stream.mm `same_operand` 1829-1832: identity by buffer equality
26. mesh_rank.m `MeshRankRecord` 540-556: identity by MTLBuffer pointer
27. metal-recording.m `MetalAllocationKey` / `MetalAllocationBase` 454-464
28. metal-recording.m `MetalViewAccesses` 489-496

### G4. host pointer ↔ Metal buffer + offset: 17

Needs storage identity. Known at allocation.

1. nccl-mesh.c `where` / `shifted` / `address_of` 1797-1809
2. nccl-mesh.c `where_of` 4045-4048
3. nccl-mesh.c `admit` 4688-4700
4. nccl-mesh.c `as_metal` 4507-4517
5. nccl-mesh.c `host_commit` 4533-4539
6. nccl-mesh.c `prepared` 4939-4942
7. nccl-mesh.c `declared` 4909-4912
8. nccl-mesh.c `bind_views` 5798-5803 (callers 5804-5811, 5819-5826, 5843-5844)
9. nccl-mesh.c `view_at` 2080-2090 (callers 3104-3105, 4121, 4161, 4399, 4408)
10. nccl-mesh.c `ncclMeshPageTableBinding` 5575-5578
11. nccl-mesh.c `write_prepare` 5533-5536: plan->source = shifted base
12. nccl-mesh.c `ncclMeshWriteExecute` 5731-5737: rebuilds the send/recv pointer for enqueue
13. nccl-mesh.c `fabric_realize` 1287-1291, `piece_take` 2955-2970, `bind_slot` 3065-3068: library-internal allocations
14. mesh_rank.m `issue` 351-352: contents + offset
15. mesh_rank.m `columns_prepare` 808-811: relay ncclMemAlloc → ncclMeshMemBuffer
16. mesh_rank.m `columns_prepare` 823-828: landing table pointer
17. _stream.mm `Commands` ctor 722-725: constants storage → ncclMeshMemBuffer

### G5. dtype → count or bytes on a WRITE: 34

Needs element bits only. Known at declaration. The library and adapter entries are resolved by 8f1fa367: 3, 7, 8, 9,
10, 13, 15, 16, and 2 and 18 for mesh_compile_write.

1. nccl.h `ncclMeshCall` 1097, 1102
2. mesh-write-compiler.h `mesh_compile_write` 17-19
3. nccl-mesh.c `mesh_compile_write` 5698-5706
4. nccl-mesh.c `write_pages_prepare` 5607: ncclUint8
5. nccl-mesh.c `mesh_compile_spans` 5638: ncclUint8
6. nccl-mesh.c `mesh_compile_rows` 5687: ncclUint8
7. nccl-mesh.c `write_prepare` 5411-5432: SIZE[datatype]; tile_bytes % SIZE; indexed views require ncclUint8
8. nccl-mesh.c `write_prepare` 5462-5474: type compared across ranks
9. nccl-mesh.c `write_prepare` 5537-5539
10. nccl-mesh.c `write_prepare_spans` 5289-5301: mesh_operand{type, SIZE, count}; tile_bytes/SIZE
11. nccl-mesh.c `write_steps` 2211-2226
12. nccl-mesh.c `order_writes` 2664-2687
13. nccl-mesh.c `declared_as` 2821-2826
14. nccl-mesh.c `declared` 4871-4882
15. nccl-mesh.c `write_definitions` 5384
16. nccl-mesh.c `ncclMeshWriteExecute` 5736
17. mesh_mpi.py `kind` 79-83
18. mesh_mpi.py `Comm.Write_prepare` 269-273
19. mesh_mpi.py `Comm.Write_rows_prepare` 289-290
20. _stream.mm `kind` 143-160
21. _stream.mm `Group::moving` 284-288
22. _stream.mm `fused_allocate_ring` 1247-1253
23. _stream.mm `prepare_chain` 1381-1384
24. _stream.mm `fused_writes_impl` 1552-1556
25. _stream.mm `fused_writes_impl` 1567-1573
26. backend.py `typed` 64-69
27. backend.py `ProcessGroupMesh.plan` 229-262
28. backend.py `bind_gather_views` 472-474: uint8 relay
29. partition.py `_gather_writes` 437-447
30. partition.py `gathered` 450-462
31. partition.py `_scatter_spans` 836
32. declarations.py `lower` 122, 128, 130, 147, 165
33. mesh_rank.m `MeshRankPlan` 925-930
34. mesh_rank.m `columns_prepare` 792-794

### G6. collective issue pointers: 19

1-8. mesh_mpi.py: Allreduce 316-318, Reduce 320-323, Bcast 325-327, Allgatherv 332-342, Reduce_scatter 347-355,
     Alltoall 357-361, Gather 363-368, Scatter 370-375
9. _stream.mm `Group::reducing` 289-299
10. _stream.mm `prepare` 310-366
11. _stream.mm `encoded` (Issue) 369-397
12. _stream.mm `issue` 428-440
13. _stream.mm `Collectives` 768-797
14. backend.py `prepare_fragments` 437-450
15. backend.py `ProcessGroupMesh.plan` 260
16. mesh_rank.m `issue` 344-375
17. mesh_rank.m `MeshRankPlan` 940-955
18. nccl-mesh.c `launch_metal` 4390-4412
19. nccl-mesh.c `complement_copy` 4156-4165

### G7. recorder / descriptor offsets: 17 (the allowed layer)

1. metal-recording.h 29-32, 47
2. metal-recording.m `setBuffer` / `setBufferOffset` / `setBytes` 759-786
3. metal-recording.m `recordDispatch` 836-845
4. metal-recording.m `recordDispatch` 860-871
5. metal-recording.m `MetalRecordReplay` 1094-1108
6. metal-recording.m `MetalRecord` 1918-1925
7. metal-recording.m `MetalAccesses` 470-487
8. metal-recording.m `MetalReplayBuild` 1031-1035
9. metal-recording.m `MetalReplayDispatch4` 1140-1150
10. metal-recording.m `MetalReplayDispatch` 1152-1179
11. metal-recording.m `MetalReplayCommands` 1191-1197
12. metal-recording.m `MetalScheduleData` 1395-1405
13. metal-recording.m `MetalScheduleComputeNow` 1407-1420
14. metal-recording.m `MetalScheduleCompute` 1444-1455
15. _stream.mm `encode_in` 807-812
16. _stream.mm `prepare_dispatch` 816-823
17. mesh_rank.m `MeshRankPrepare` 1029-1033

### G8. allocator / lease layer: 9 (the allowed layer)

mesh.py `lease` 192-200; mesh_mpi.py `empty` 53-58 and `to_region` 61-67; _stream.mm `returned` 967-971, `region_empty`
973-986 and `records` 1058-1063; partition.py `_empty` 164-168; mesh_rank.m `MeshRankAllocate` 448-470;
nccl-mesh.c `ncclMemAlloc`/`ncclMemFree`.

Callers outside the listed files:
- torch_mesh/__init__.py `address` and `records` (182-191);
- fused.py `writes`/`write_rows` (45-90), which go through `ScheduleBinding.writes`;
- `migration.compile_rows`, which produces the byte runs used in G2 #24.
