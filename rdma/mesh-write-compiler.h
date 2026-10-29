/* The view-layout arithmetic every WRITE binding is lowered with, during preparation only. Program-facing operands
   are nccl.h's: a page table and its masks (ncclMeshWritePagesPrepare), or a view's abstract layout over a page
   table (ncclMeshLayout, ncclMeshWriteLayoutPrepare); no address is an operand. Peers may use different row or
   span forms of the same logical bytes (design/pointer-lowering-2026-10-28.md). */
#ifndef MESH_WRITE_COMPILER_H
#define MESH_WRITE_COMPILER_H
#include "nccl.h"
typedef struct ncclMeshSpan { uint64_t offset, bytes; } ncclMeshSpan;
/* A binding's local layout as the common WRITE compiler takes it, in bytes (ncclMeshWriteLayoutPrepare fills it from
   ncclMeshLayout's elements; ncclMeshWritePagesPrepare from its pages). */
struct mesh_write_layout { size_t tile_bytes, instances, instance_stride, invocations; };
#ifdef __cplusplus
extern "C" {
#endif
/* A strided view's bytes in logical (C) element order, the transfer order every WRITE binds (nccl.h ncclMeshWriteLayoutPrepare):
   an affine view's rows (rows[0] bytes each, rows[1] of them, rows[2] apart, ascending) and no
   span; else spans ({offset, bytes}, adjacent ones merged, a broadcast view's repeated bytes repeated) relative to its
   lowest addressed byte, *shift bytes from its first element, up to `room` of them written.  Strides are bytes;
   trailing dense dimensions are one row.  The spans needed (0 for an affine or empty view, its rows 0 when empty), or
   -1 where shape or strides are missing.  Preparation only: no value is read. */
long mesh_view_layout(const uint64_t* shape, const int64_t* strides, size_t rank, uint64_t itemsize,
    int64_t* shift, uint64_t* rows, ncclMeshSpan* spans, size_t room);
#ifdef __cplusplus
}
#endif
#endif
