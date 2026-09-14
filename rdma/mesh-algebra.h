#ifndef MESH_ALGEBRA_H
#define MESH_ALGEBRA_H
#include <stddef.h>
#include <stdint.h>
struct mesh_ctx;
#ifdef __OBJC__
@protocol MTLCommandBuffer;
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* design/streaming-algebra.md */
enum mesh_scalar { MESH_F16, MESH_F32, MESH_I32, MESH_U32, MESH_I64, MESH_U64, MESH_U8, MESH_BOOL };
enum mesh_algebra_op { MESH_CONTRACT = 6 };
struct mesh_algebra;
struct mesh_tensor;
struct mesh_shape { size_t rows,columns; enum mesh_scalar scalar; };
struct mesh_view {
  struct mesh_tensor *tensor;
  uint32_t extent;
  size_t offset,rows,columns,row_stride,column_stride;
};
struct mesh_copy_region { struct mesh_view source; size_t row,column; };
/* design/algorithm-sources.md#view-scoped-host-production */
struct mesh_writer;
struct mesh_algebra_report { uint64_t submitted,completed,native_submitted,native_backings; int64_t code; double gpu_seconds; uint64_t cpu_submitted; };

struct mesh_algebra *mesh_algebra_create(struct mesh_ctx *);
struct mesh_algebra *mesh_algebra_create_cpu(struct mesh_ctx *);
int mesh_algebra_kernel(struct mesh_algebra *);
int mesh_algebra_coreml(struct mesh_algebra *,const char *python,const char *generator,const char *cache);
void mesh_algebra_destroy(struct mesh_algebra *);
size_t mesh_algebra_page_bytes(struct mesh_algebra *);
size_t mesh_algebra_publication_bytes(struct mesh_algebra *);
uint32_t mesh_algebra_node(struct mesh_algebra *);
struct mesh_tensor *mesh_tensor_create(struct mesh_algebra *,const struct mesh_shape *,size_t extents,int transferable,int contiguous);
int mesh_algebra_materialize(struct mesh_algebra *,const struct mesh_copy_region *,size_t,struct mesh_view);
struct mesh_view mesh_tensor_view(struct mesh_tensor *,uint32_t extent);
struct mesh_view mesh_view_slice(struct mesh_view,size_t row,size_t column,size_t rows,size_t columns);
struct mesh_view mesh_view_transpose(struct mesh_view);
struct mesh_view mesh_view_broadcast(struct mesh_view,size_t rows,size_t columns);
size_t mesh_tensor_publication_bytes(struct mesh_tensor *,uint32_t extent);
void *mesh_tensor_data(struct mesh_tensor *,uint32_t extent);
int mesh_tensor_constant(struct mesh_tensor *,uint32_t extent);
int mesh_algebra_writer(struct mesh_algebra *,struct mesh_view,struct mesh_writer **);
int mesh_writer_writable(struct mesh_writer *);
int mesh_writer_issue(struct mesh_writer *);
void mesh_writer_complete(struct mesh_writer *,int publish);
#ifdef __OBJC__
/* design/algorithm-sources.md#programkernel_call */
int mesh_algebra_encode(struct mesh_algebra *,const struct mesh_view *inputs,size_t input_count,const struct mesh_view *outputs,size_t output_count,void (^encode)(id<MTLCommandBuffer>));
#endif
int mesh_algebra_source(struct mesh_algebra *,const char *cpu_source,const char *metal_source,const struct mesh_view *inputs,size_t input_count,struct mesh_view output,const uint8_t *access_axes,size_t row_begin,size_t row_count,size_t column_begin,size_t column_count);
int mesh_algebra_indexed(struct mesh_algebra *,size_t function,struct mesh_view selector,const size_t *candidate_inputs,size_t input_count,const struct mesh_view *candidates,size_t count);
int mesh_algebra_bind(struct mesh_algebra *,enum mesh_algebra_op,struct mesh_view a,struct mesh_view b,struct mesh_view output,float alpha,float beta);
/* design/algorithm-sources.md#selected-native-contractions */
int mesh_algebra_view_pages(struct mesh_algebra *,struct mesh_view,struct mesh_view *pages,size_t capacity,size_t *count);
int mesh_algebra_copy(struct mesh_algebra *,struct mesh_view source,uint32_t sender,struct mesh_view destination,uint32_t receiver,uint16_t queue);
int mesh_algebra_export(struct mesh_algebra *,struct mesh_view,size_t *first,size_t *count);
int mesh_algebra_realize(struct mesh_algebra *);
int mesh_algebra_available(struct mesh_algebra *,size_t output);
void mesh_algebra_consume(struct mesh_algebra *,size_t output);
size_t mesh_algebra_function_count(struct mesh_algebra *);
struct mesh_algebra_report mesh_algebra_report(struct mesh_algebra *);

#ifdef __cplusplus
}
#endif
#endif
