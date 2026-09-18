#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_calls;
struct mesh_call;
struct mesh_function;
/* design/algorithm-sources.md#programkernel_call */
typedef __attribute__((swiftcall)) void (*mesh_invoke)(struct mesh_call *,void * __attribute__((swift_context)));
/* design/prepared-machine.md#M41 */
struct mesh_submission {
  _Alignas(128) struct mesh_arrival *destinations[MESH_COMPUTE_THREADS];
  uint32_t counts[MESH_COMPUTE_THREADS];
  uint32_t generation,stride,count;
};
_Static_assert(sizeof(struct mesh_submission)==128 && _Alignof(struct mesh_submission)==128,"M41");
/* design/prepared-machine.md#M41 */
/* design/algorithm-sources.md#program */
static inline __attribute__((always_inline)) uint32_t mesh_submit(struct mesh_submission *submission){
  uint32_t generation=submission->generation,count=submission->count;
  submission->generation=generation+submission->stride;
  for(uint32_t i=0;i<count;i++){
    struct mesh_arrival *cells=submission->destinations[i];
    uint32_t extent=submission->counts[i];
    for(uint32_t k=0;k<extent;k++)atomic_store_explicit(&cells[k].stamp,(uint64_t)generation+1,memory_order_release);
  }
  return generation;
}
/* design/prepared-machine.md#M42 */
/* design/algorithm-sources.md#meshresult */
static inline __attribute__((always_inline)) uint64_t mesh_result(struct mesh_instance *instance,uint32_t generation){
  struct mesh_status status=atomic_load_explicit(&instance->status,memory_order_acquire);
  if(status.completed==(uint64_t)generation+1 || status.completed==UINT64_MAX)return 0;
  return status.value>>62==MESH_RESULT_SUCCESS?MESH_RESULT(MESH_RESULT_BUSY,0,0):status.value;
}
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride,channel; };
/* design/prepared-machine.md#M43 */
struct mesh_operand {
  _Alignas(16) void *data;
  size_t bytes;
  uint32_t index,row;
  uint32_t *sequence;
};
_Static_assert(sizeof(struct mesh_operand)==32 && _Alignof(struct mesh_operand)==16 && offsetof(struct mesh_operand,sequence)==24,"M43");
/* design/prepared-machine.md#M20 */
struct mesh_call {
  _Alignas(128) struct mesh_operand *operands;
  struct mesh_instance *instance;
  _Atomic uint32_t *completed;
  _Alignas(16) uint32_t index;
  uint32_t pending,invocation,identity;
  mesh_invoke submit;
  void *argument;
  uint32_t publication_count,recurring;
  _Alignas(128) struct prepared_publication publications[];
};
_Static_assert(sizeof(struct mesh_call)==128 && _Alignof(struct mesh_call)==128,"mesh_call record");
_Static_assert(offsetof(struct mesh_call,pending)==36 && offsetof(struct mesh_call,invocation)==40 &&
  offsetof(struct mesh_call,submit)==48 && offsetof(struct mesh_call,argument)==56 &&
  offsetof(struct mesh_call,publication_count)==64 && offsetof(struct mesh_call,recurring)==68 && offsetof(struct mesh_call,publications)==128,"M20");
/* design/prepared-machine.md#M26 */
struct prepared_residency {
  _Alignas(32) uint64_t completed;
  uint32_t reserved[6];
};
_Static_assert(sizeof(struct prepared_residency)==32 && offsetof(struct prepared_residency,completed)==0,"M26");
/* design/prepared-machine.md#M27 */
/* design/prepared-machine.md#M28 */
/* design/prepared-machine.md#M33 */
/* design/prepared-machine.md#M34 */
/* design/prepared-machine.md#M35 */
_Static_assert(4*sizeof(uint32_t)+2*sizeof(uint64_t)==32 && 6*sizeof(uint32_t)+sizeof(uint64_t)==32,"M27 M28 M33 M34 M35");
/* design/prepared-machine.md#M36 */
_Static_assert(2*sizeof(uint32_t)==8,"M36");
/* design/prepared-machine.md#M29 */
_Static_assert(2*sizeof(uint32_t)+3*sizeof(uint64_t)==32,"M29");
/* design/prepared-machine.md#M30 */
_Static_assert(sizeof(float)==4,"M30");
/* design/prepared-machine.md#M38 */
_Static_assert(sizeof(uint32_t)==4 && sizeof(float)==4 && sizeof(uint8_t)==1,"M38");
/* design/prepared-machine.md#M31 */
/* design/prepared-machine.md#M32 */
_Static_assert(sizeof(uint32_t)==4 && sizeof(uint64_t)==8 && sizeof(uint16_t)==2,"M31 M32");
/* design/prepared-machine.md#M39 */
_Static_assert(sizeof(void *)==8,"M39 native handle");
/* design/prepared-machine.md#M40 */
struct mesh_resident_control {
  _Alignas(32) _Atomic uint32_t shutdown;
  uint32_t reserved[7];
};
_Static_assert(sizeof(struct mesh_resident_control)==32 && offsetof(struct mesh_resident_control,shutdown)==0,"M40");
/* design/prepared-machine.md#M40 */
/* design/algorithm-sources.md#resident-metal */
static inline void mesh_residency_stop(struct mesh_resident_control *control){
  atomic_store_explicit(&control->shutdown,1,memory_order_release);
}
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
/* design/prepared-machine.md#M03 */
_Static_assert(sizeof(uint16_t)==2,"M01 M02 M03 scalar storage");
/* design/algorithm-sources.md#programkernel_call */
static inline __attribute__((always_inline)) uint32_t mesh_call_index(const struct mesh_call *call){return call->index;}
/* design/algorithm-sources.md#programkernel_call */
static inline __attribute__((always_inline,returns_nonnull)) struct mesh_operand *mesh_call_operands(const struct mesh_call *call){return call->operands;}
typedef void (*mesh_dispose)(void *);

/* design/algorithm-sources.md#programkernel_call */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *,uint32_t workers,uint32_t count,void *owner,mesh_dispose);
struct mesh_function *mesh_call_bind(struct mesh_calls *,uint32_t worker,
  const struct mesh_section *inputs,size_t input_count,size_t dependency_count,
  const struct mesh_section *outputs,size_t output_count,const void *submit,void *context);
/* design/algorithm-sources.md#resident-metal */
uint64_t *mesh_function_completion(struct mesh_function *,uint32_t frame);
int mesh_calls_prepare(struct mesh_calls *);
int mesh_calls_start(struct mesh_calls *);
/* design/prepared-machine.md#M41 */
struct mesh_submission *mesh_submission_at(struct mesh_calls *,uint32_t slot,struct mesh_instance **);
void mesh_call_complete(struct mesh_call *,int error);
void mesh_call_fail(struct mesh_call *,int error);
void mesh_calls_destroy(struct mesh_calls *);

/* design/algorithm-sources.md#programcopy */
int mesh_transfer_bind(struct mesh_ctx *,uint32_t queue,int receive,uint32_t identity,struct mesh_section);
int mesh_transfers_prepare(struct mesh_ctx *);
int mesh_transfers_start(struct mesh_ctx *);

/* design/algorithm-sources.md#programtensor */
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,uint32_t channel,struct mesh_section *);
uint32_t mesh_row_page(struct mesh_ctx *,uint32_t row,uint32_t chunk);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
void mesh_section_constant(struct mesh_ctx *,struct mesh_section);

/* design/algorithm-sources.md#collectivesync_on_remote_fill */
void mesh_sync_on_remote_fill(struct mesh_ctx *,const struct mesh_section *,size_t count,uint32_t index);
#endif
