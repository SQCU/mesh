#ifndef MESH_ALGEBRA_H
#define MESH_ALGEBRA_H
#include "mesh-dataflow.h"
#ifdef __cplusplus
extern "C" {
#endif

/* design/streaming-algebra.md */
enum mesh_scalar { MESH_F16, MESH_F32, MESH_I32, MESH_U32 };
enum mesh_algebra_op { MESH_AFFINE, MESH_ADD, MESH_MULTIPLY, MESH_TANH, MESH_EXP, MESH_SUM, MESH_CONTRACT, MESH_RSQRT, MESH_SWISH };
typedef void (*mesh_completion)(void *context,int64_t error);
typedef void (*mesh_submission)(void *binding,mesh_completion complete,void *context);
struct mesh_algebra;
struct mesh_tensor;
struct mesh_shape { size_t rows,columns; enum mesh_scalar scalar; };
struct mesh_view {
  struct mesh_tensor *tensor;
  uint32_t extent;
  size_t offset,rows,columns,row_stride,column_stride;
};
struct mesh_endpoint { struct mesh_tensor *tensor; uint32_t peer,first,stride; };
struct mesh_algebra_report { uint64_t submitted,completed,native_submitted,native_backings,ne_planned_operations; int64_t code; double gpu_seconds; uint64_t cpu_submitted; };

struct mesh_algebra *mesh_algebra_create(struct mesh_ctx *);
struct mesh_algebra *mesh_algebra_create_cpu(struct mesh_ctx *);
int mesh_algebra_coreml(struct mesh_algebra *,const char *python,const char *generator,const char *cache);
void mesh_algebra_destroy(struct mesh_algebra *);
size_t mesh_algebra_publication_bytes(struct mesh_algebra *);
struct mesh_tensor *mesh_tensor_create(struct mesh_algebra *,const struct mesh_shape *,size_t extents,int transferable);
struct mesh_view mesh_tensor_view(struct mesh_tensor *,uint32_t extent);
struct mesh_view mesh_view_slice(struct mesh_view,size_t row,size_t column,size_t rows,size_t columns);
struct mesh_view mesh_view_transpose(struct mesh_view);
struct mesh_view mesh_view_broadcast(struct mesh_view,size_t rows,size_t columns);
void *mesh_tensor_data(struct mesh_tensor *,uint32_t extent);
struct mesh_row_map mesh_tensor_rows(struct mesh_tensor *,uint32_t extent);
int mesh_tensor_present(struct mesh_tensor *,uint32_t extent);
int mesh_tensor_constant(struct mesh_tensor *,uint32_t extent);
int mesh_tensor_issue(struct mesh_tensor *,uint32_t extent);
void mesh_tensor_complete(struct mesh_tensor *,uint32_t extent);
int mesh_algebra_function(struct mesh_algebra *,const struct mesh_view *inputs,size_t input_count,const struct mesh_view *outputs,size_t output_count,mesh_submission,void *binding);
int mesh_algebra_bind(struct mesh_algebra *,enum mesh_algebra_op,struct mesh_view a,struct mesh_view b,struct mesh_view output,float alpha,float beta);
int mesh_algebra_copy(struct mesh_algebra *,struct mesh_endpoint source,struct mesh_endpoint destination,size_t count,uint16_t queue);
int mesh_algebra_export(struct mesh_algebra *,struct mesh_tensor *,uint32_t extent,size_t *index);
int mesh_algebra_return(struct mesh_algebra *,struct mesh_tensor *,uint32_t extent);
int mesh_algebra_realize(struct mesh_algebra *);
void mesh_algebra_scan(struct mesh_algebra *);
int mesh_algebra_available(struct mesh_algebra *,size_t output);
void mesh_algebra_consume(struct mesh_algebra *,size_t output);
struct mesh_algebra_report mesh_algebra_report(struct mesh_algebra *);

#ifdef __cplusplus
}
#endif
#endif
