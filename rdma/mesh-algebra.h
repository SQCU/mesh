#ifndef MESH_ALGEBRA_H
#define MESH_ALGEBRA_H
#include "mesh-dataflow.h"
#ifdef __cplusplus
extern "C" {
#endif

/* design/streaming-algebra.md */
enum mesh_scalar { MESH_F16, MESH_F32, MESH_I32, MESH_U32, MESH_I64, MESH_U64, MESH_U8, MESH_BOOL };
enum mesh_algebra_op { MESH_AFFINE, MESH_ADD, MESH_MULTIPLY, MESH_TANH, MESH_EXP, MESH_SUM, MESH_CONTRACT, MESH_RSQRT, MESH_SWISH };
struct mesh_metal_dispatch { const char *name; size_t grid[3],group[3],argument_buffer,argument_offset; };
struct mesh_metal_constant { const void *bytes; size_t length; };
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
struct mesh_writer { struct mesh_ctx *context; struct mesh_row_map output; struct mesh_row_function function; };
struct mesh_endpoint { struct mesh_tensor *tensor; uint32_t peer,first,stride; };
struct mesh_algebra_event { uint64_t ready_ns,start_ns,complete_ns,gpu_start_ns,gpu_end_ns,submissions; uint32_t first_output,output_maps,kind,input_maps; };
/* design/algorithm-sources.md#function-cost-profiles */
enum mesh_algebra_backend { MESH_BACKEND_EXTERNAL, MESH_BACKEND_CPU_SGEMM, MESH_BACKEND_CPU_NEON_CONTRACT, MESH_BACKEND_CPU_BUILTIN, MESH_BACKEND_CPU_COMPILED, MESH_BACKEND_METAL_COMPILED, MESH_BACKEND_METAL_MPS, MESH_BACKEND_METAL_BUILTIN, MESH_BACKEND_COREML, MESH_BACKEND_SELECTED_MIXED };
struct mesh_algebra_profile { uint64_t successful,failed,gpu_samples; double dispatch_mean_ns,dispatch_m2_ns2,execution_mean_ns,execution_m2_ns2,gpu_mean_ns,gpu_m2_ns2; uint32_t kind,backend; };
/* design/algorithm-sources.md#function-cost-profiles */
struct mesh_algebra_plan { struct mesh_view left,right,output; uint64_t first,count; uint32_t backend,operation,left_scalar,right_scalar,output_scalar,rectangles; float alpha,beta; };
struct mesh_indexed_event { uint64_t input; uint32_t descriptor,role,candidate,first,count,plane,retired,selected,completed,mapped,flags; };
struct mesh_active_event { uint64_t function,omissions; uint32_t slot,count_first,count_maps,disposition,omitted,retired,inputs,flags; };
struct mesh_route_event { uint64_t function; uint32_t domain,role,index,first,count,plane,retired,completed,prepared,consumer,flags; };
struct mesh_algebra_report { uint64_t submitted,completed,native_submitted,native_backings,ne_planned_operations; int64_t code; double gpu_seconds; uint64_t cpu_submitted; };

struct mesh_algebra *mesh_algebra_create(struct mesh_ctx *);
struct mesh_algebra *mesh_algebra_create_cpu(struct mesh_ctx *);
int mesh_algebra_coreml(struct mesh_algebra *,const char *python,const char *generator,const char *cache);
void mesh_algebra_destroy(struct mesh_algebra *);
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
struct mesh_row_map mesh_tensor_rows(struct mesh_tensor *,uint32_t extent);
int mesh_tensor_constant(struct mesh_tensor *,uint32_t extent);
int mesh_algebra_writer(struct mesh_algebra *,struct mesh_view,struct mesh_writer *);
int mesh_writer_writable(struct mesh_writer *);
int mesh_writer_issue(struct mesh_writer *);
void mesh_writer_complete(struct mesh_writer *);
int mesh_algebra_metal(struct mesh_algebra *,const char *,const struct mesh_metal_dispatch *,size_t,const struct mesh_metal_constant *,size_t,const struct mesh_view *,size_t,const struct mesh_view *,size_t);
int mesh_algebra_source(struct mesh_algebra *,const char *cpu_source,const char *metal_source,const struct mesh_view *inputs,size_t input_count,struct mesh_view output,const uint8_t *row_inputs,size_t row_begin,size_t row_count);
int mesh_algebra_indexed(struct mesh_algebra *,size_t function,struct mesh_view selector,const size_t *candidate_inputs,size_t candidate_count);
int mesh_algebra_indexed_range(struct mesh_algebra *,size_t function,struct mesh_view selector,struct mesh_view range,const size_t *candidate_inputs,size_t candidate_count);
struct mesh_route *mesh_algebra_route_create(struct mesh_algebra *,struct mesh_view owners,struct mesh_view ordinals,struct mesh_view offsets,const struct mesh_view *candidates,size_t candidate_count,size_t consumer_count);
struct mesh_view mesh_algebra_route_table(struct mesh_algebra *,struct mesh_route *);
int mesh_algebra_active(struct mesh_algebra *,size_t function,struct mesh_view count,size_t slot);
int mesh_algebra_route_producers(struct mesh_algebra *,struct mesh_route *,const size_t *functions,size_t count);
int mesh_algebra_route_hold(struct mesh_algebra *,struct mesh_route *,const struct mesh_view *,size_t count);
int mesh_algebra_route_attach(struct mesh_algebra *,size_t function,struct mesh_route *,size_t consumer);
int mesh_algebra_bind(struct mesh_algebra *,enum mesh_algebra_op,struct mesh_view a,struct mesh_view b,struct mesh_view output,float alpha,float beta);
/* design/algorithm-sources.md#selected-native-contractions */
int mesh_algebra_view_pages(struct mesh_algebra *,struct mesh_view,struct mesh_view *pages,size_t capacity,size_t *count);
int mesh_algebra_contract_select(struct mesh_algebra *,struct mesh_view selector,const struct mesh_view *left,const struct mesh_view *right,size_t plan_count,const struct mesh_view *inputs,size_t input_count,struct mesh_view output,float alpha,size_t *function_index);
int mesh_algebra_copy(struct mesh_algebra *,struct mesh_endpoint source,struct mesh_endpoint destination,size_t count,uint16_t queue);
int mesh_algebra_present(struct mesh_algebra *,struct mesh_view);
int mesh_algebra_export(struct mesh_algebra *,struct mesh_view,size_t *first,size_t *count);
int mesh_algebra_realize(struct mesh_algebra *);
int mesh_algebra_available(struct mesh_algebra *,size_t output);
void mesh_algebra_consume(struct mesh_algebra *,size_t output);
size_t mesh_algebra_trace_count(struct mesh_algebra *);
struct mesh_algebra_event mesh_algebra_trace(struct mesh_algebra *,size_t function);
struct mesh_algebra_profile mesh_algebra_profile(struct mesh_algebra *,size_t function);
const char *mesh_algebra_environment(struct mesh_algebra *);
const char *mesh_algebra_specialization(struct mesh_algebra *,size_t function);
const char *mesh_algebra_source_text(struct mesh_algebra *,size_t function,uint32_t language);
size_t mesh_algebra_plan_count(struct mesh_algebra *,size_t function);
struct mesh_algebra_plan mesh_algebra_plan(struct mesh_algebra *,size_t function,size_t plan);
struct mesh_row_range mesh_algebra_trace_input(struct mesh_algebra *,size_t function,size_t input);
struct mesh_row_range mesh_algebra_trace_output(struct mesh_algebra *,size_t function,size_t output);
size_t mesh_algebra_trace_indexed_count(struct mesh_algebra *,size_t function);
struct mesh_indexed_event mesh_algebra_trace_indexed(struct mesh_algebra *,size_t function,size_t entry);
struct mesh_reader_event mesh_algebra_trace_input_reader(struct mesh_algebra *,size_t function,size_t input,uint32_t row);
struct mesh_reader_event mesh_algebra_trace_indexed_reader(struct mesh_algebra *,size_t function,size_t entry,uint32_t row);
struct mesh_active_event mesh_algebra_trace_active(struct mesh_algebra *,size_t function);
struct mesh_row_range mesh_algebra_trace_active_count(struct mesh_algebra *,size_t function,size_t map);
struct mesh_reader_event mesh_algebra_trace_active_reader(struct mesh_algebra *,size_t function,size_t map,uint32_t row);
struct mesh_active_event mesh_algebra_trace_route_producer(struct mesh_algebra *,size_t entry);
size_t mesh_algebra_trace_route_count(struct mesh_algebra *);
struct mesh_route_event mesh_algebra_trace_route(struct mesh_algebra *,size_t entry);
struct mesh_reader_event mesh_algebra_trace_route_reader(struct mesh_algebra *,size_t entry,uint32_t row);
struct mesh_algebra_report mesh_algebra_report(struct mesh_algebra *);

#ifdef __cplusplus
}
#endif
#endif
