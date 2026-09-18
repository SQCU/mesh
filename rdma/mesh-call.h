#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"

struct mesh_calls;
struct mesh_call;
struct mesh_function;
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride,channel; };
struct mesh_operand {
  size_t bytes;
  struct mesh_page_entry *pages; struct mesh_publication *publication; uint32_t index;
  uint32_t *sequence; size_t quantum;
};
_Static_assert(sizeof(struct mesh_operand)==48,"mesh_operand");
struct mesh_call {
  _Alignas(64) struct mesh_function *function;
  struct mesh_operand *operands;
  uint64_t return_slot;
  uint32_t index,remaining,pending,invocation,return_index;
  int error;
  struct hdr *memory;
  uint32_t input_count,output_count;
  uint64_t completion_slot;
};
_Static_assert(sizeof(struct mesh_call)==128 && _Alignof(struct mesh_call)==64,"mesh_call record");
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
/* design/prepared-machine.md#M37 */
_Static_assert(sizeof(struct mesh_stream)==128 && offsetof(struct mesh_stream,slots)==8,"M37");
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
/* design/prepared-machine.md#M13 */
struct prepared_publication { uint64_t destination; uint64_t value,scale,reserved; };
_Static_assert(sizeof(struct prepared_publication)==32 && offsetof(struct prepared_publication,scale)==16,"M13");
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
/* design/prepared-machine.md#M03 */
_Static_assert(sizeof(uint16_t)==2,"M01 M02 M03 scalar storage");
/* design/algorithm-sources.md#programkernel_call */
static inline __attribute__((always_inline)) uint32_t mesh_call_index(const struct mesh_call *call){return call->index;}
/* design/algorithm-sources.md#programkernel_call */
static inline __attribute__((always_inline,returns_nonnull)) struct mesh_operand *mesh_call_operands(const struct mesh_call *call){return call->operands;}
/* design/algorithm-sources.md#programtensor */
static inline __attribute__((always_inline)) void *mesh_operand_data(const struct mesh_page_entry *pages){return (void *)atomic_load_explicit(&pages->address,memory_order_relaxed);}
/* design/algorithm-sources.md#programtensor */
static inline void *mesh_operand_address(struct mesh_operand operand,size_t offset){
  size_t block=offset/operand.quantum;
  return (char *)mesh_operand_data(operand.pages+block)+(offset-block*operand.quantum);
}
/* design/algorithm-sources.md#programtensor */
static inline uint32_t mesh_operand_page(struct mesh_operand operand){return atomic_load_explicit(&operand.pages->mapping,memory_order_relaxed);}
typedef void (*mesh_dispose)(void *);

/* design/algorithm-sources.md#programkernel_call */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *,uint32_t workers,uint32_t count,void *owner,mesh_dispose);
struct mesh_function *mesh_call_bind(struct mesh_calls *,uint32_t worker,
  const struct mesh_section *inputs,size_t input_count,size_t dependency_count,int completion_publication,
  const struct mesh_section *outputs,size_t output_count,const void *submit,void *context,const void *rearm,void *rearm_context);
/* design/algorithm-sources.md#resident-metal */
uint64_t *mesh_function_completion(struct mesh_function *,uint32_t frame,uint64_t *value);
void mesh_call_finish(struct mesh_call *);
int mesh_calls_prepare(struct mesh_calls *);
int mesh_calls_start(struct mesh_calls *);
uint64_t mesh_calls_submit(struct mesh_calls *,uint32_t index);
/* design/algorithm-sources.md#meshresult */
uint64_t mesh_calls_result(struct mesh_calls *,uint32_t index);
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
void mesh_section_release(struct mesh_ctx *,struct mesh_section);

/* design/algorithm-sources.md#collectivesync_on_remote_fill */
void mesh_sync_on_remote_fill(struct mesh_ctx *,const struct mesh_section *,size_t count,uint32_t index);
#endif
