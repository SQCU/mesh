#include "mesh-call.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* design/algorithm-sources.md#programkernel_call */
typedef __attribute__((swiftcall)) void (*mesh_rearm)(uint32_t,void * __attribute__((swift_context)));

/* design/prepared-machine.md#M23 */
struct mesh_function {
  struct mesh_calls *calls;
  struct mesh_section *inputs;
  unsigned char *values;
  struct mesh_operand *operands;
  uint32_t *consumed;
  size_t input_count,output_count,consumed_count,dependency_count;
  uint32_t worker,identity,pending,initial;
  mesh_invoke submit;
  void *argument;
  mesh_rearm rearm;
  void *rearm_argument;
  struct mesh_function *next;
};
_Static_assert(sizeof(struct mesh_function)==128,"M23");
/* design/prepared-machine.md#M22 */
struct mesh_call_worker {
  _Alignas(128) struct mesh_calls *calls;
  uint32_t index;
  _Atomic uint32_t completed;
  struct mesh_arrival *arrivals;
  struct mesh_event_reader returns;
  uint32_t arrival_count;
  _Atomic uintptr_t frame;
};
_Static_assert(sizeof(struct mesh_call_worker)==128 && _Alignof(struct mesh_call_worker)==128,"mesh_call_worker");
_Static_assert(14*sizeof(uintptr_t)==112,"M25 input ABI frame");
struct mesh_calls {
  struct mesh_ctx *context;
  struct mesh_function *functions;
  struct mesh_call_worker workers[MESH_COMPUTE_THREADS];
  uint32_t count,extent,first,function_count;
  unsigned char *values;
  size_t stride;
  struct mesh_instance *instances;
  struct mesh_submission *submissions;
  _Atomic uint32_t references;
  _Atomic int running;
  void *owner;
  mesh_dispose dispose;
};
_Static_assert(sizeof(struct mesh_calls)==1280,"M24");

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* design/algorithm-sources.md#programtensor */
static void mesh_calls_release(struct mesh_calls *calls,uint32_t references){
  if(atomic_fetch_sub_explicit(&calls->references,references,memory_order_acq_rel)!=references)return;
  for(uint32_t i=0;i<calls->count;i++)free(calls->workers[i].returns.inputs);
  for(struct mesh_function *function=calls->functions;function;){
    struct mesh_function *next=function->next;
    free(function->operands);free(function->inputs);free(function);
    function=next;
  }
  if(calls->first!=MESH_ABSENT)mesh_rows_release(calls->context,calls->first,calls->extent);
  if(calls->dispose)calls->dispose(calls->owner);
  free(calls->submissions);free(calls->values);free(calls);
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *context,uint32_t workers,uint32_t count,void *owner,mesh_dispose dispose){
  if(!context || !context->M || !workers || workers>MESH_COMPUTE_THREADS || !count){errno=EINVAL;return NULL;}
  /* design/prepared-machine.md#M24 */
  struct mesh_calls *calls=aligned_alloc(_Alignof(struct mesh_calls),sizeof *calls);
  if(!calls)return NULL;
  memset(calls,0,sizeof *calls);
  calls->context=context;calls->count=workers;calls->extent=count;calls->first=MESH_ABSENT;atomic_init(&calls->references,1);
  /* design/prepared-machine.md#M22 */
  for(uint32_t i=0;i<workers;i++)calls->workers[i]=(struct mesh_call_worker){.calls=calls,.index=i};
  calls->first=mesh_rows_alloc(context,count);
  if(calls->first==MESH_ABSENT){mesh_calls_release(calls,1);return NULL;}
  calls->instances=mesh_instances(context->M,context->client);
  /* design/prepared-machine.md#M41 */
  calls->submissions=aligned_alloc(_Alignof(struct mesh_submission),(size_t)count*sizeof *calls->submissions);
  if(!calls->submissions){mesh_calls_release(calls,1);return NULL;}
  context->M->instance_count[context->client>>63]=count;
  /* design/prepared-machine.md#M42 */
  for(uint32_t i=0;i<count;i++){
    calls->submissions[i]=(struct mesh_submission){.instance=calls->instances+i,.generation=i,.stride=count};
    atomic_store_explicit(&calls->instances[i].status,((struct mesh_status){MESH_RESULT(MESH_RESULT_BUSY,0,0),0}),memory_order_relaxed);
  }
  atomic_store_explicit(&context->M->result[context->client>>63],((struct mesh_status){MESH_RESULT(MESH_RESULT_BUSY,0,0),0}),memory_order_release);
  calls->owner=owner;calls->dispose=dispose;
  return calls;
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_function *mesh_call_bind(struct mesh_calls *calls,uint32_t worker,
  const struct mesh_section *inputs,size_t input_count,size_t dependency_count,
  const struct mesh_section *outputs,size_t output_count,const void *submit,void *argument,const void *rearm,void *rearm_argument){
  if(worker>=calls->count || dependency_count>input_count || input_count>UINT32_MAX || output_count>UINT32_MAX-input_count || atomic_load(&calls->running)){errno=EINVAL;return NULL;}
  /* design/prepared-machine.md#M23 */
  struct mesh_function *function=calloc(1,sizeof *function);
  if(!function)return NULL;
  size_t count=input_count+output_count;
  function->inputs=calloc(1,(count?count:1)*sizeof *inputs+input_count*sizeof *function->consumed);
  if(!function->inputs){free(function);return NULL;}
  function->consumed=(uint32_t *)(function->inputs+count);
  memcpy(function->inputs,inputs,input_count*sizeof *inputs);
  memcpy(function->inputs+input_count,outputs,output_count*sizeof *outputs);
  function->calls=calls;function->worker=worker;function->input_count=input_count;function->output_count=output_count;
  function->dependency_count=dependency_count;
  function->submit=(mesh_invoke)submit;function->argument=argument;
  function->rearm=(mesh_rearm)rearm;function->rearm_argument=rearm_argument;
  for(size_t i=0;i<input_count;i++){
    if(inputs[i].stride)function->consumed[function->consumed_count++]=(uint32_t)i;
    for(uint32_t index=0;index<(inputs[i].stride?calls->extent:1);index++)
      mesh_buffer_retain(calls->context->M,mesh_section_row(inputs[i],index));
  }
  function->identity=calls->function_count++;
  function->next=calls->functions;calls->functions=function;
  return function;
}

/* design/algorithm-sources.md#resident-metal */
/* design/prepared-machine.md#M37 */
uint64_t *mesh_function_completion(struct mesh_function *function,uint32_t frame,uint64_t *value){
  struct mesh_call *call=(void *)(function->values+frame*function->calls->stride);
  *value=(UINT64_C(1)<<63)|((uint64_t)call->return_index+1);
  return (uint64_t *)((char *)call->memory+call->completion_slot+8);
}

/* design/algorithm-sources.md#programkernel_call */
static void *mesh_call_progress(void *argument){
  struct mesh_call_worker *worker=argument;
  struct mesh_calls *calls=worker->calls;
  unsigned char *values=calls->values;
  size_t stride=calls->stride;
  struct mesh_arrival *arrivals=worker->arrivals;
  uint32_t count=worker->arrival_count,cursor=0,submitted=0;

  pthread_setname_np("mesh.numerical");
  /* design/prepared-machine.md#M25 */
  atomic_store_explicit(&worker->frame,(uintptr_t)__builtin_frame_address(0),memory_order_release);
  while(atomic_load_explicit(&calls->running,memory_order_acquire) || atomic_load_explicit(&worker->completed,memory_order_acquire)!=submitted){
    /* design/prepared-machine.md#M18 */
    for(uint32_t visited=0;visited<count;visited++){
      struct mesh_arrival *cell=arrivals+cursor;
      if(++cursor==count)cursor=0;
      uint64_t value=atomic_load_explicit(&cell->stamp,memory_order_acquire);
      if(!value)continue;
      atomic_store_explicit(&cell->stamp,0,memory_order_relaxed);
      /* design/prepared-machine.md#M20 */
      struct mesh_call *call=(void *)(values+(size_t)cell->call*stride);
      call->invocation^=(call->invocation^(uint32_t)(value-1))&cell->mask;
      if(--call->pending)continue;
      call->submit(call,call->argument);
      submitted++;
    }
    {
      struct hdr *m=calls->context->M;
      uint64_t event;
      while((event=mesh_event_take(&worker->returns))!=MESH_EVENT_ABSENT){
        struct mesh_call *call=(void *)(values+(size_t)(uint32_t)event*stride);
        if(event>>63){mesh_call_finish(call);submitted++;continue;}
        if(!call->error && --call->remaining)continue;
        struct mesh_function *function=call->function;
        if(call->error)mesh_result_conclude(&calls->instances[call->index].status,
          MESH_RESULT(MESH_RESULT_FUNCTION,call->function->identity,call->error));
        else {
          if(function->rearm)function->rearm(call->index,function->rearm_argument);
          for(size_t i=0;i<function->output_count;i++)mesh_buffer_reset(m,call->operands[function->input_count+i].row);
          call->remaining=(uint32_t)(function->output_count?function->output_count:1);
          call->pending=function->pending;
          mesh_instance_release(&calls->instances[call->index],1);
        }
      }
    }
  }
  mesh_calls_release(calls,1);
  return NULL;
}

/* design/prepared-machine.md#M22 */
/* design/algorithm-sources.md#index-hand-off */
static int mesh_events_prepare(struct mesh_calls *calls){
  struct hdr *m=calls->context->M;
  uint32_t queue=mesh_notice_queue(m,calls->context->client,m->links*(m->qps+1)+MESH_COMPUTE_THREADS);
  for(uint32_t i=0;i<calls->count;i++){
    int error=mesh_event_reader_init(&calls->workers[i].returns,m,queue+i);
    if(error)return error;
  }
  for(uint32_t row=0;row<m->rows;row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    struct mesh_publication *publication=mesh_publication_at(m,row);
    if(atomic_load_explicit(&buffer->owner,memory_order_relaxed)==calls->context->client && publication->sends &&
       atomic_load_explicit(&mesh_page(m)[row].stamp,memory_order_relaxed)){
      /* design/prepared-machine.md#M13 */
      uint32_t count=mesh_publication_prepare(m,row,1,NULL);
      struct prepared_publication *stores=aligned_alloc(32,(size_t)count*sizeof *stores);
      if(!stores)return ENOMEM;
      mesh_publication_prepare(m,row,1,stores);
      for(uint32_t i=0;i<count;i++)atomic_store_explicit((_Atomic uint64_t *)(uintptr_t)stores[i].destination,1,memory_order_release);
      free(stores);
    }
  }
  return 0;
}

/* design/algorithm-sources.md#programkernel_call */
int mesh_calls_prepare(struct mesh_calls *calls){
  struct hdr *m=calls->context->M;
  uint32_t rows=mesh_rows(m);
  for(uint32_t row=0;row<rows;row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_relaxed)==calls->context->client)buffer->initial=atomic_load_explicit(&buffer->references,memory_order_relaxed);
  }
  size_t width=(size_t)rows+1;
  size_t *offsets=calloc(calls->count*width,sizeof *offsets);
  if(!offsets)return ENOMEM;
  for(int pass=0;pass<2;pass++){
    for(struct mesh_function *function=calls->functions;function;function=function->next){
      struct mesh_call_worker *worker=&calls->workers[function->worker];
      size_t *indices=offsets+function->worker*width;
      uint32_t pending=0,initial=0;int varying=0;
      for(size_t i=0;function->submit && i<=function->dependency_count;i++){
        struct mesh_section section;
        if(i==function->dependency_count){
          if(varying)continue;
          section=(struct mesh_section){.first=calls->first,.count=calls->extent,.stride=1};
        } else {
          section=function->inputs[i];
          varying|=section.stride!=0;
          if(atomic_load_explicit(&mesh_page(m)[section.first].stamp,memory_order_relaxed))continue;
        }
        initial++;pending+=section.stride!=0;
        for(uint32_t index=0;index<calls->extent;index++){
          uint32_t row=mesh_section_row(section,index);
          if(pass){
            /* design/prepared-machine.md#M18 */
            worker->arrivals[indices[row]++]=(struct mesh_arrival){
              .call=function->identity*calls->extent+index,.mask=section.stride?UINT32_MAX:0};
          } else {
            indices[row+1]++;
            mesh_publish_bind(calls->context,row,m->links*(m->qps+1)+function->worker,0);
          }
        }
      }
      if(!pass){
        function->pending=pending;
        function->initial=initial;
      }
    }
    for(uint32_t i=0;i<calls->count;i++){
      struct mesh_call_worker *worker=&calls->workers[i];
      size_t *indices=offsets+i*width;
      if(pass){
        for(uint32_t row=0,first=0;row<rows;row++){
          uint32_t end=(uint32_t)indices[row];
          if(first!=end){
            struct mesh_target *target=mesh_publish_bind(calls->context,row,m->links*(m->qps+1)+i,0);
            target->stream=(uint64_t)((char *)(worker->arrivals+first)-(char *)m);
            target->count=end-first;
          }
          first=end;
        }
      } else {
        for(uint32_t row=0;row<rows;row++)indices[row+1]+=indices[row];
        if(indices[rows]>UINT32_MAX){free(offsets);return EOVERFLOW;}
        /* design/prepared-machine.md#M18 */
        worker->arrival_count=(uint32_t)indices[rows];
        if(worker->arrival_count){
          uint64_t bytes=indices[rows]*sizeof *worker->arrivals,quantum=(uint64_t)m->block*m->pgsz;
          uint64_t pages=(bytes+quantum-1)/quantum*m->block;
          if(pages>m->rows){free(offsets);return ENOMEM;}
          uint32_t page=mesh_arena_alloc(calls->context,(uint32_t)pages,m->block);
          if(page==MESH_ABSENT){free(offsets);return errno;}
          struct mesh_pool *pool=mesh_pools(m)+page/m->block;pool->pages=(uint32_t)pages;
          atomic_store_explicit(&pool->owner,calls->context->client,memory_order_release);
          worker->arrivals=(void *)mesh_at(m,page);
        }
      }
    }
  }
  free(offsets);
  /* design/prepared-machine.md#M20 */
  size_t publications=0;
  for(struct mesh_function *function=calls->functions;function;function=function->next)if(function->submit)
    for(uint32_t frame=0;frame<calls->extent;frame++){
      size_t count=0;
      for(size_t i=0;i<function->output_count;i++)
        count+=mesh_publication_prepare(m,mesh_section_row(function->inputs[function->input_count+i],frame),1,NULL);
      if(count>publications)publications=count;
    }
  calls->stride=(sizeof(struct mesh_call)+publications*sizeof(struct prepared_publication)+127)&~(size_t)127;
  calls->values=aligned_alloc(128,(calls->function_count?calls->function_count:1)*calls->extent*calls->stride);
  if(!calls->values)return ENOMEM;
  for(struct mesh_function *function=calls->functions;function;function=function->next){
    size_t count=function->input_count+function->output_count;
    function->values=calls->values+(size_t)function->identity*calls->extent*calls->stride;
    /* design/prepared-machine.md#M43 */
    function->operands=aligned_alloc(32,calls->extent*(count?count:1)*sizeof *function->operands);
    if(!function->operands)return ENOMEM;
    uint32_t queue=mesh_notice_queue(m,calls->context->client,m->links*(m->qps+1)+MESH_COMPUTE_THREADS+function->worker);
    for(uint32_t index=0;index<calls->extent;index++){
      struct mesh_call *call=(void *)(function->values+index*calls->stride);
      uint32_t slot=function->identity*calls->extent+index;
      *call=(struct mesh_call){.function=function,.operands=function->operands+index*count,.index=index,
        .remaining=(uint32_t)(function->output_count?function->output_count:1),.pending=function->initial,
        .memory=m,.input_count=(uint32_t)function->input_count,.output_count=(uint32_t)function->output_count,
        .return_index=slot,.submit=function->submit,.argument=function->argument};
      if(!call->submit)call->completion_slot=mesh_event_bind(m,queue);
      for(size_t i=0;i<count;i++){
        struct mesh_section section=function->inputs[i];
        uint32_t row=mesh_section_row(section,index);
        call->operands[i]=(struct mesh_operand){.data=(void *)atomic_load_explicit(&mesh_page(m)[row].address,memory_order_relaxed),
          .bytes=section.bytes,.index=section.stride?index:0,.row=row,.sequence=&call->invocation};
        if(i<function->input_count)continue;
        struct mesh_buffer *buffer=mesh_buffers(m)+row;
        buffer->channel=m->links*m->qps+MESH_COMPUTE_THREADS+function->worker;buffer->return_index=slot;
        buffer->return_slot=mesh_event_bind(m,queue);
        /* design/prepared-machine.md#M13 */
        if(call->submit)call->publication_count+=mesh_publication_prepare(m,row,1,call->publications+call->publication_count);
      }
      call->return_slot=function->output_count?mesh_buffers(m)[call->operands[function->input_count].row].return_slot:mesh_event_bind(m,queue);
    }
  }
  fprintf(stderr,"mesh arena: in_flight=%u bytes_per_instance=%llu shared_bytes=%llu allocated_bytes=%llu\n",calls->extent,
    (unsigned long long)(calls->context->arena-calls->context->shared_pages)*m->pgsz/calls->extent,
    (unsigned long long)calls->context->shared_pages*m->pgsz,(unsigned long long)calls->context->arena*m->pgsz);
  uint32_t dynamic=calls->function_count,shared=0;
  for(uint32_t q=0;q<m->links*m->qps;q++)for(int direction=0;direction<2;direction++){
    uint32_t count=atomic_load(mesh_order_length(m,calls->context->client,q,direction));
    struct mesh_transfer *transfers=mesh_transfers(m,calls->context->client,q,direction);
    for(uint32_t i=0;i<count;i++){if(transfers[i].stride)dynamic++;else shared++;}
  }
  for(uint32_t frame=0;frame<calls->extent;frame++){
    calls->instances[frame].count=dynamic;
    atomic_store_explicit(&calls->instances[frame].remaining,dynamic+shared,memory_order_relaxed);
    if(!dynamic && !shared)atomic_store_explicit(&calls->instances[frame].status,((struct mesh_status){0,UINT64_MAX}),memory_order_relaxed);
  }
  if(!dynamic && !shared)atomic_store_explicit(&m->result[calls->context->client>>63],((struct mesh_status){0,UINT64_MAX}),memory_order_release);
  int event_error=mesh_events_prepare(calls);
  if(event_error)return event_error;
  /* design/prepared-machine.md#M41 */
  for(uint32_t slot=0;slot<calls->extent;slot++){
    struct mesh_publication *root=mesh_publication_at(m,calls->first+slot);
    struct mesh_submission *submission=calls->submissions+slot;
    submission->count=root->uses;
    for(uint32_t i=0;i<root->uses;i++){
      submission->destinations[i]=(struct mesh_arrival *)((char *)m+root->targets[i].stream);
      submission->counts[i]=root->targets[i].count;
    }
  }
  /* design/prepared-machine.md#M10 */
  for(uint32_t q=0;q<m->links*m->qps;q++){
    struct mesh_transfer *in=mesh_transfers(m,calls->context->client,q,MESH_RECEIVE);
    uint32_t count=atomic_load(mesh_order_length(m,calls->context->client,q,MESH_RECEIVE));
    for(uint32_t i=0;i<count;i++)for(uint32_t slot=0;slot<in[i].count;slot++){
      uint32_t row=in[i].local_row+slot*in[i].stride,chunks=mesh_buffers(m)[row].pages/m->block;
      struct mesh_publication *first=mesh_publication_at(m,row),*last=mesh_publication_at(m,row+chunks-1);
      if(first!=last){
        memcpy(last,first,(size_t)m->target_stride);
        first->sends=first->uses=0;
      }
      for(uint32_t k=0;k+1<chunks;k++)mesh_publication_at(m,row+k)->row=MESH_ABSENT;
    }
  }
  return 0;
}

/* design/prepared-machine.md#M17 */
/* design/algorithm-sources.md#programkernel_call */
int mesh_calls_start(struct mesh_calls *calls){
  uint32_t workers=0;
  for(uint32_t i=0;i<calls->count;i++)if(calls->workers[i].arrival_count || calls->workers[i].returns.count)workers|=UINT32_C(1)<<i;
  atomic_store_explicit(&calls->running,1,memory_order_release);
  atomic_fetch_add_explicit(&calls->references,(uint32_t)__builtin_popcount(workers),memory_order_relaxed);
  while(workers){
    uint32_t i=(uint32_t)__builtin_ctz(workers);
    pthread_t thread;
    int error=pthread_create(&thread,NULL,mesh_call_progress,&calls->workers[i]);
    if(error){
      atomic_store(&calls->running,0);
      mesh_calls_release(calls,(uint32_t)__builtin_popcount(workers));return error;
    }
    /* design/prepared-machine.md#M25 */
    while(!atomic_load_explicit(&calls->workers[i].frame,memory_order_acquire)){}
    workers&=workers-1;
    pthread_detach(thread);
  }
  return 0;
}

/* design/prepared-machine.md#M41 */
/* design/algorithm-sources.md#program */
struct mesh_submission *mesh_submission_at(struct mesh_calls *calls,uint32_t slot,struct mesh_instance **instance){
  *instance=calls->instances+slot;
  return calls->submissions+slot;
}

/* design/algorithm-sources.md#programkernel_call */
static __attribute__((noinline)) void mesh_call_cleanup(struct mesh_call *call,int error){
  struct hdr *m=call->memory;
  struct mesh_operand *outputs=call->operands+call->input_count;
  size_t output_count=call->output_count;
  struct mesh_function *function=call->function;
  struct mesh_calls *calls=function->calls;
  _Atomic uint32_t *completed=&calls->workers[function->worker].completed;
  for(size_t i=0;i<function->consumed_count;i++){
    uint32_t input=function->consumed[i];
    mesh_buffer_release(m,call->operands[input].row);
  }
  if(error || !output_count)
    mesh_event_push(m,call->return_slot,call->return_index,0);
  else for(size_t i=0;i<output_count;i++)mesh_buffer_release(m,outputs[i].row);
  atomic_fetch_add_explicit(completed,1,memory_order_release);
}

/* design/algorithm-sources.md#resident-metal */
void mesh_call_finish(struct mesh_call *call){mesh_call_cleanup(call,0);}

/* design/algorithm-sources.md#programkernel_call */
void mesh_call_complete(struct mesh_call *call,int error){
  if(error)call->error=error;
  else {
    /* design/prepared-machine.md#M13 */
    uint64_t invocation=call->invocation;
    for(uint32_t i=0,count=call->publication_count;i<count;i++){
      struct prepared_publication store=call->publications[i];
      atomic_store_explicit((_Atomic uint64_t *)(uintptr_t)store.destination,1+invocation*store.scale,memory_order_release);
    }
  }
  __attribute__((musttail)) return mesh_call_cleanup(call,error);
}

/* design/algorithm-sources.md#meshresult */
void mesh_call_fail(struct mesh_call *call,int error){mesh_call_complete(call,error?error:EIO);}

/* design/algorithm-sources.md#programkernel_call */
void mesh_calls_destroy(struct mesh_calls *calls){
  if(!calls)return;
  atomic_store_explicit(&calls->running,0,memory_order_release);
  mesh_calls_release(calls,1);
}

/* design/algorithm-sources.md#programcopy */
int mesh_transfer_bind(struct mesh_ctx *context,uint32_t queue,int receive,uint32_t identity,struct mesh_section section){
  struct hdr *m=context->M;
  if(queue>=m->links*m->qps)return EINVAL;
  _Atomic uint32_t *length=mesh_order_length(m,context->client,queue,receive);
  uint32_t index=atomic_load_explicit(length,memory_order_relaxed);
  if(index==mesh_blocks(m))return ENOSPC;
  if(!receive)for(uint32_t value=0;value<section.count;value++){
    uint32_t row=mesh_section_row(section,value);
    mesh_buffer_retain(m,row);
    uint32_t link=queue/m->qps;
    mesh_publish_bind(context,row,link,1)->count++;
  }
  mesh_transfers(m,context->client,queue,receive)[index]=(struct mesh_transfer){section.first,identity,section.count,section.stride,MESH_ABSENT,0,section.bytes};
  atomic_store_explicit(length,index+1,memory_order_release);
  return 0;
}

/* design/prepared-machine.md#M04 */
/* design/algorithm-sources.md#programcopy */
static int mesh_transfer_compare(const void *a,const void *b){
  const struct mesh_transfer *left=*(const struct mesh_transfer *const *)a,*right=*(const struct mesh_transfer *const *)b;
  return (left->local_row>right->local_row)-(left->local_row<right->local_row);
}

/* design/algorithm-sources.md#programcopy */
int mesh_transfers_prepare(struct mesh_ctx *context){
  struct hdr *m=context->M;
  /* design/prepared-machine.md#M04 */
  uint32_t slots=m->instance_count[context->client>>63];
  for(uint32_t p=0;p<m->links;p++){
    uint64_t base=m->notice_off+(uint64_t)mesh_notice_queue(m,context->client,p)*m->notice_bytes;
    struct mesh_tx *tx=(void *)((char *)m+base);
    uint32_t count=0,once=0,length=0;
    for(uint32_t q=0;q<m->qps;q++)length+=atomic_load_explicit(mesh_order_length(m,context->client,p*m->qps+q,MESH_SEND),memory_order_relaxed);
    struct mesh_transfer **ordered=malloc((length?length:1)*sizeof *ordered);
    if(!ordered)return ENOMEM;
    uint32_t position=0;
    for(uint32_t q=0;q<m->qps;q++){
      struct mesh_transfer *out=mesh_transfers(m,context->client,p*m->qps+q,MESH_SEND);
      uint32_t length=atomic_load_explicit(mesh_order_length(m,context->client,p*m->qps+q,MESH_SEND),memory_order_relaxed);
      for(uint32_t i=0;i<length;i++){
        ordered[position++]=out+i;
        if(out[i].stride)count++;else once++;
      }
    }
    *tx=(struct mesh_tx){.count=count,.slots=slots,.once=once};
    memset(tx->cells,0,((size_t)count*slots+once)*sizeof *tx->cells);
    qsort(ordered,length,sizeof *ordered,mesh_transfer_compare);
    uint32_t next[2]={0,once};
    for(uint32_t i=0;i<length;i++){
      struct mesh_transfer *out=ordered[i];
      uint32_t varying=out->stride!=0;
      out->first=next[varying];
      for(uint32_t slot=0;slot<out->count;slot++){
        struct mesh_publication *publication=mesh_publication_at(m,out->local_row+slot*out->stride);
        for(uint32_t j=0;j<publication->sends;j++)if(publication->targets[j].stream==base){
          publication->targets[j].stream=base+offsetof(struct mesh_tx,cells)+sizeof(struct mesh_send)*(slot*count+next[varying]);
        }
      }
      next[varying]++;
    }
    free(ordered);
  }

  /* design/prepared-machine.md#M02 */
  for(uint32_t q=0;q<m->links*m->qps;q++){
    uint32_t count=atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE)),pages=0;
    struct mesh_transfer *transfers=mesh_transfers(m,context->client,q,MESH_RECEIVE);
    for(uint32_t i=0;i<count;i++)pages+=transfers[i].count*mesh_buffers(m)[transfers[i].local_row].pages;
    if(!pages)continue;
    uint32_t page=mesh_arena_alloc(context,pages,m->block);
    if(page==MESH_ABSENT)return errno;
    struct mesh_pool *pool=&mesh_pools(m)[page/m->block];pool->pages=pages;
    atomic_store_explicit(&pool->owner,context->client,memory_order_release);
    for(uint32_t i=0;i<count;i++){
      transfers[i].pool=page;
      uint32_t span=mesh_buffers(m)[transfers[i].local_row].pages;
      for(uint32_t slot=0;slot<transfers[i].count;slot++){
        mesh_backing_bind(context,transfers[i].local_row+slot*transfers[i].stride,span,page,slot);
        page+=span;
      }
    }
  }
  return 0;
}

/* design/prepared-machine.md#M06 */
/* design/algorithm-sources.md#programcopy */
int mesh_transfers_start(struct mesh_ctx *context){
  struct hdr *m=context->M;
  atomic_store_explicit(&m->configured,context->client,memory_order_release);
  for(uint32_t p=0;p<m->links;p++){
    uint32_t transfers=0;
    for(uint32_t q=0;q<m->qps;q++)for(int d=0;d<2;d++)transfers+=atomic_load(mesh_order_length(m,context->client,p*m->qps+q,d));
    if(!transfers)continue;
    struct mesh_port_info *port=&mesh_links(m)[p].port;
    while(atomic_load_explicit(&port->prepared,memory_order_acquire)!=context->client){}
    if(atomic_load_explicit(&port->phase,memory_order_relaxed)==MESH_STOPPED)
      return (int)atomic_load_explicit(&port->code,memory_order_relaxed);
  }
  return 0;
}

/* design/algorithm-sources.md#programtensor */
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,uint32_t channel,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || !count)return EINVAL;
  size_t quantum=(size_t)m->block*m->pgsz;
  if(bytes>(size_t)mesh_rows(m)*m->pgsz)return ENOMEM;
  size_t span=(bytes+quantum-1)/quantum*m->block;
  if(span>mesh_rows(m)/count)return ENOMEM;
  uint32_t stride=(uint32_t)span/m->block,rows=count*stride,first=mesh_rows_alloc(context,rows);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t row=first;row<first+rows;row+=stride){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    atomic_store_explicit(&buffer->references,2,memory_order_relaxed);buffer->pages=(uint32_t)span;buffer->channel=channel;
    if(channel!=MESH_ABSENT){
      buffer->return_slot=mesh_event_bind(m,mesh_notice_queue(m,context->client,m->links+channel));
    }
    atomic_store_explicit(&buffer->owner,context->client,memory_order_release);
  }
  mesh_bits_set(m,MESH_ROW_HOT,first,rows);
  /* design/prepared-machine.md#M01 */
  /* design/prepared-machine.md#M03 */
  if(channel==MESH_ABSENT){
    uint32_t page=mesh_arena_alloc(context,(uint32_t)span*count,m->block);
    if(page==MESH_ABSENT){
      int error=errno;
      for(uint32_t r=first;r<first+rows;r+=stride){mesh_buffer_release(m,r);mesh_buffer_release(m,r);}
      mesh_rows_release(context,first,rows);return error;
    }
    for(uint32_t slot=0;slot<count;slot++)
      mesh_backing_bind(context,first+slot*stride,(uint32_t)span,page+slot*(uint32_t)span,slot);
  }
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,stride,channel};
  return 0;
}
/* design/algorithm-sources.md#programtensor */
uint32_t mesh_row_page(struct mesh_ctx *context,uint32_t row,uint32_t chunk){return atomic_load_explicit(&mesh_page(context->M)[row+chunk].mapping,memory_order_acquire);}
/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return mesh_at(context->M,mesh_row_page(context,mesh_section_row(section,index),0));}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){
  atomic_store_explicit(&mesh_page(context->M)[section.first].stamp,1,memory_order_release);
  mesh_buffer_release(context->M,section.first);
}
/* design/algorithm-sources.md#programtensor */
void mesh_section_release(struct mesh_ctx *context,struct mesh_section section){
  for(uint32_t index=0;index<section.count;index++)mesh_buffer_release(context->M,mesh_section_row(section,index));
  mesh_rows_release(context,section.first,section.count*(section.pages/context->M->block));
}
/* design/algorithm-sources.md#collectivesync_on_remote_fill */
void mesh_sync_on_remote_fill(struct mesh_ctx *context,const struct mesh_section *sections,size_t count,uint32_t index){
  for(size_t i=0;i<count;i++){
    struct mesh_section section=sections[i];
    uint64_t stamp=section.stride?(uint64_t)index+1:1;
    for(;;)for(uint32_t slot=0;slot<section.count;slot++)
      if(atomic_load_explicit(&mesh_page(context->M)[mesh_section_row(section,slot)].stamp,memory_order_acquire)==stamp)goto filled;
filled:;
  }
}
