/* Tensor frontend lowering, used only during preparation. Program-facing
   communication operands are ncclMeshPageMask in nccl.h. These adapters resolve
   local tensor views to that page domain before the common WRITE compiler.
   Affine rows and repeated typed values remain compact operands. Indexed
   views retain only their declared selections; none expand to per-page masks.
   Peers may use different row/indexed forms of the same logical bytes. */
#ifndef MESH_WRITE_COMPILER_H
#define MESH_WRITE_COMPILER_H
#include "nccl.h"
typedef struct ncclMeshSpan { uint64_t offset, bytes; } ncclMeshSpan;
/* Local tensor binding strides. Public page programs use ncclMeshWriteLayout's
   page indices instead; no tensor address becomes a peer destination address. */
struct mesh_write_layout { size_t tile_bytes, instances, instance_stride, invocations; };
#ifdef __cplusplus
extern "C" {
#endif
ncclResult_t mesh_compile_write(const void* source_buffer, void* landing_buffer, size_t count, ncclDataType_t datatype,
    int source, int destination, ncclComm_t comm, int metal, uint32_t repeats, size_t stride, size_t max_packet,
    ncclMeshWritePlan_t* out);
ncclResult_t mesh_compile_rows(const void* source_buffer, void* landing_buffer, size_t row_bytes, size_t rows,
    size_t pitch, int source, int destination, ncclComm_t comm, int metal, size_t max_packet, struct mesh_write_layout layout, ncclMeshWritePlan_t* out);
ncclResult_t mesh_compile_spans(const void* source_buffer, void* landing_buffer,
    const ncclMeshSpan* spans, size_t nspans, int source, int destination, ncclComm_t comm,
    int metal, size_t max_packet, struct mesh_write_layout layout, ncclMeshWritePlan_t* out);
/* A view as mesh_view_layout gives it, `source_buffer`/`landing_buffer` at its lowest addressed byte (its first element
   moved by *shift): its rows where rows[1] (mesh_compile_rows), else its spans (mesh_compile_spans). */
ncclResult_t mesh_compile_view(const void* source_buffer, void* landing_buffer, const uint64_t rows[3], const ncclMeshSpan* spans,
    size_t nspans, int source, int destination, ncclComm_t comm, int metal, size_t max_packet, struct mesh_write_layout layout,
    ncclMeshWritePlan_t* out);
/* A strided view's bytes in logical (C) element order, the transfer order every frontend binds (mesh_compile_view):
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
