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
#ifdef __cplusplus
}
#endif
#endif
