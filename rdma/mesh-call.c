#include "mesh-call.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct mesh_call {
  _Alignas(64) struct mesh_function *function;
  struct mesh_operand *operands;
  uint64_t return_slot;
  uint32_t index,remaining,pending,invocation,return_row;
  int error;
  struct hdr *memory;
  uint32_t input_count,output_count;
};
_Static_assert(sizeof(struct mesh_call)==64 && _Alignof(struct mesh_call)==64,"mesh_call record");
struct mesh_function {
  struct mesh_calls *calls;
  struct mesh_section *inputs;
  struct mesh_call *values;
  struct mesh_operand *operands;
  uint32_t *consumed;
  size_t input_count,output_count,consumed_count;
  uint32_t worker,identity,pending;
  mesh_submit submit;
  mesh_rearm rearm;
  mesh_dispose dispose;
  void *argument;
  struct mesh_function *next;
};
/* design/algorithm-sources.md#programkernel_call */
struct mesh_use {
  _Alignas(64) mesh_submit submit;
  void *argument;
  struct mesh_call *call;
  struct mesh_operand *operands;
  uint32_t invocation_mask,input_count,end;
};
_Static_assert(sizeof(struct mesh_use)==64 && _Alignof(struct mesh_use)==64,"mesh_use record");
struct mesh_call_worker {
  struct mesh_calls *calls;
  struct mesh_use *targets;
  uint32_t index;
  _Atomic uint32_t active;
  struct mesh_event_reader arrivals,returns;
};
struct mesh_calls {
  struct mesh_ctx *context;
  struct mesh_function *functions;
  struct mesh_call_worker workers[MESH_COMPUTE_THREADS];
  uint32_t count,extent,first,return_first,return_count,root_workers,function_count;
  struct mesh_call **slots;
  struct mesh_instance *instances;
  _Atomic uint32_t references;
  _Atomic int running;
  void *owner;
  mesh_dispose dispose;
};

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* design/algorithm-sources.md#programtensor */
static void mesh_calls_release(struct mesh_calls *calls,uint32_t references){
  if(atomic_fetch_sub_explicit(&calls->references,references,memory_order_acq_rel)!=references)return;
  for(uint32_t i=0;i<calls->count;i++){free(calls->workers[i].targets);free(calls->workers[i].arrivals.inputs);free(calls->workers[i].returns.inputs);}
  for(struct mesh_function *function=calls->functions;function;){
    struct mesh_function *next=function->next;
    if(function->dispose)function->dispose(function->argument);
    free(function->operands);free(function->values);free(function->inputs);free(function);
    function=next;
  }
  if(calls->first!=MESH_ABSENT)mesh_rows_release(calls->context,calls->first,calls->extent);
  if(calls->return_first!=MESH_ABSENT)mesh_rows_release(calls->context,calls->return_first,calls->return_count);
  if(calls->dispose)calls->dispose(calls->owner);
  free(calls->slots);free(calls);
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *context,uint32_t workers,uint32_t count,void *owner,mesh_dispose dispose){
  if(!context || !context->M || !workers || workers>MESH_COMPUTE_THREADS || !count){errno=EINVAL;return NULL;}
  struct mesh_calls *calls=calloc(1,sizeof *calls);
  if(!calls)return NULL;
  calls->context=context;calls->count=workers;calls->extent=count;calls->first=calls->return_first=MESH_ABSENT;atomic_init(&calls->references,1);
  for(uint32_t i=0;i<workers;i++)calls->workers[i]=(struct mesh_call_worker){.calls=calls,.index=i};
  calls->first=mesh_rows_alloc(context,count);
  if(calls->first==MESH_ABSENT){mesh_calls_release(calls,1);return NULL;}
  calls->instances=mesh_instances(context->M,context->client);
  context->M->instance_count[context->client>>63]=count;
  for(uint32_t i=0;i<count;i++){
    atomic_store_explicit(&calls->instances[i].status,((struct mesh_status){MESH_RESULT(MESH_RESULT_BUSY,0,0),0}),memory_order_relaxed);
    atomic_store_explicit(&calls->instances[i].available,1,memory_order_relaxed);
  }
  atomic_store_explicit(&context->M->result[context->client>>63],((struct mesh_status){MESH_RESULT(MESH_RESULT_BUSY,0,0),0}),memory_order_release);
  calls->owner=owner;calls->dispose=dispose;
  return calls;
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_function *mesh_call_bind(struct mesh_calls *calls,uint32_t worker,
  const struct mesh_section *inputs,const struct mesh_section *views,size_t input_count,
  const struct mesh_section *outputs,size_t output_count,mesh_submit submit,mesh_rearm rearm,void *argument,mesh_dispose dispose){
  if(worker>=calls->count || !submit || input_count>UINT32_MAX || output_count>UINT32_MAX-input_count || atomic_load(&calls->running)){errno=EINVAL;return NULL;}
  struct mesh_function *function=calloc(1,sizeof *function);
  if(!function)return NULL;
  size_t count=input_count+output_count,extent=calls->extent;
  function->inputs=calloc(1,(input_count?input_count:1)*sizeof *inputs+input_count*sizeof *function->consumed);
  function->operands=calloc(extent*(count?count:1),sizeof *function->operands);
  function->values=aligned_alloc(_Alignof(struct mesh_call),extent*sizeof *function->values);
  if(!function->inputs || !function->operands || !function->values){errno=ENOMEM;goto failed;}
  function->consumed=(uint32_t *)(function->inputs+input_count);
  for(size_t i=0;i<input_count;i++){
    if(inputs[i].stride)function->consumed[function->consumed_count++]=(uint32_t)i;
  }
  memcpy(function->inputs,inputs,input_count*sizeof *inputs);
  function->calls=calls;function->worker=worker;function->input_count=input_count;function->output_count=output_count;
  function->submit=submit;function->rearm=rearm;function->argument=argument;function->dispose=dispose;
  struct hdr *m=calls->context->M;
  for(size_t i=0;i<input_count;i++)for(uint32_t index=0;index<(inputs[i].stride?extent:1);index++)
    mesh_buffer_retain(m,mesh_section_row(inputs[i],index));
  for(uint32_t index=0;index<extent;index++){
    struct mesh_call *call=&function->values[index];
    *call=(struct mesh_call){.function=function,.operands=function->operands+index*count,.index=index,
      .remaining=(uint32_t)(output_count?output_count:1),.memory=m,.input_count=(uint32_t)input_count,.output_count=(uint32_t)output_count};
    for(size_t i=0;i<count;i++){
      struct mesh_section section=i<input_count?views[i]:outputs[i-input_count];
      uint32_t row=mesh_section_row(i<input_count?inputs[i]:section,index);
      function->operands[index*count+i]=(struct mesh_operand){.arena=mesh_at(m,0),.bytes=section.bytes,
        .index=section.stride?index:0,.row=row,.sequence=&call->invocation,
        .pages=mesh_page(m)+mesh_section_row(section,index),.page_size=m->pgsz,.block_pages=m->block};
    }
  }
  function->identity=calls->function_count++;
  function->next=calls->functions;calls->functions=function;
  return function;
failed:
  free(function->inputs);free(function->operands);free(function->values);free(function);
  return NULL;
}

/* design/algorithm-sources.md#programkernel_call */
static void *mesh_call_progress(void *argument){
  struct mesh_call_worker *worker=argument;
  struct mesh_calls *calls=worker->calls;
  struct hdr *m=calls->context->M;
  struct mesh_buffer *buffers=mesh_buffers(m);

  pthread_setname_np("mesh.numerical");
  while(atomic_load_explicit(&calls->running,memory_order_acquire) || atomic_load_explicit(&worker->active,memory_order_acquire)){
    uint64_t event;
    if(atomic_load_explicit(&calls->running,memory_order_acquire)){
      while((event=mesh_event_take(&worker->arrivals))!=MESH_EVENT_ABSENT){
        for(uint32_t i=(uint32_t)event,end=worker->targets[i].end;i<end;i++){
          struct mesh_use use=worker->targets[i];
          use.call->invocation^=(use.call->invocation^(uint32_t)(event>>32))&use.invocation_mask;
          if(--use.call->pending)continue;
          atomic_fetch_add_explicit(&worker->active,1,memory_order_relaxed);
          use.submit(use.call,use.call->index,use.argument,use.operands,use.operands+use.input_count);
        }
      }
    }
    while((event=mesh_event_take(&worker->returns))!=MESH_EVENT_ABSENT){
      struct mesh_call *call=calls->slots[buffers[(uint32_t)event].binding];
      if(!call->error && --call->remaining)continue;
      struct mesh_function *function=call->function;
      if(call->error)mesh_result_conclude(&calls->instances[call->index].status,
        MESH_RESULT(MESH_RESULT_FUNCTION,call->function->identity,call->error));
      else {
        if(function->rearm)function->rearm(call->index,function->argument);
        for(size_t i=0;i<function->output_count;i++)mesh_buffer_reset(m,call->operands[function->input_count+i].row);
        call->remaining=(uint32_t)(function->output_count?function->output_count:1);
        call->pending=function->pending;
        mesh_instance_release(&calls->instances[call->index],1);
      }
    }
  }
  mesh_calls_release(calls,1);
  return NULL;
}

/* design/algorithm-sources.md#index-hand-off */
struct mesh_event_binding { uint64_t source; uint32_t index; };
/* design/algorithm-sources.md#index-hand-off */
static int mesh_event_compare(const void *a,const void *b){
  const struct mesh_event_binding *left=a,*right=b;
  if(left->source!=right->source)return (left->source>right->source)-(left->source<right->source);
  return (left->index>right->index)-(left->index<right->index);
}
/* design/algorithm-sources.md#index-hand-off */
static uint32_t mesh_event_root(uint32_t *parents,uint32_t index){
  while(parents[index]!=index){parents[index]=parents[parents[index]];index=parents[index];}
  return index;
}
/* design/algorithm-sources.md#index-hand-off */
static uint64_t mesh_event_remap(struct hdr *m,uint64_t first,const uint64_t *mapping,uint64_t slot){
  uint64_t relative=slot-first;
  return mapping[relative/m->notice_bytes*m->rows+(relative%m->notice_bytes-sizeof(struct mesh_events))/sizeof(struct mesh_stream)];
}
/* design/algorithm-sources.md#index-hand-off */
static int mesh_events_prepare(struct mesh_calls *calls){
  struct hdr *m=calls->context->M;
  uint32_t rows=m->rows,local=m->links*(m->qps+1),queues=local+2*MESH_COMPUTE_THREADS;
  uint32_t first_queue=mesh_notice_queue(m,calls->context->client,0);
  uint64_t first=m->notice_off+(uint64_t)first_queue*m->notice_bytes;
  uint32_t *last=calloc(rows,sizeof *last),*parents=malloc((size_t)rows*sizeof *parents);
  unsigned char *successors=calloc(rows,1);
  uint64_t *mapping=malloc((size_t)queues*rows*sizeof *mapping);
  struct mesh_event_binding *bindings=malloc((size_t)rows*sizeof *bindings);
  int error=0;
  if(!last || !parents || !successors || !mapping || !bindings){error=ENOMEM;goto done;}
  for(struct mesh_function *function=calls->functions;function;function=function->next)if(function->output_count){
    for(uint32_t frame=0;frame<calls->extent;frame++){
      struct mesh_operand *outputs=function->values[frame].operands+function->input_count;
      last[outputs[function->output_count-1].row]=outputs[0].row+1;
    }
  }
  for(uint32_t row=0;row<rows;row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_relaxed)!=calls->context->client)continue;
    struct mesh_target *targets=mesh_targets(m,row);
    for(uint32_t i=0;i<buffer->sends+buffer->uses;i++)((struct mesh_stream *)((char *)m+targets[i].stream))->slots=buffer->publisher;
  }
  uint64_t total_bindings=0,total_streams=0;
  for(uint32_t q=0;q<queues;q++){
    struct mesh_events *events=mesh_events(m,first_queue+q);
    uint32_t count=events->count;
    total_bindings+=count;
    for(uint32_t row=0;row<rows;row++)parents[row]=row;
    memset(successors,0,rows);
    if(q<m->links || (q>=local && q<local+MESH_COMPUTE_THREADS)){
      for(struct mesh_function *function=calls->functions;function;function=function->next)if(function->output_count){
        for(uint32_t frame=0;frame<calls->extent;frame++){
          uint32_t target=function->values[frame].operands[function->input_count].row;
          for(size_t i=0;i<function->input_count;i++){
            uint32_t row=mesh_section_row(function->inputs[i],frame),source=last[row];
            if(!function->inputs[i].stride || atomic_load_explicit(&mesh_presence(m)[row],memory_order_relaxed))continue;
            if(!source || successors[source-1])continue;
            source--;
            if(q>=local){
              struct mesh_buffer *buffer=&mesh_buffers(m)[row];
              struct mesh_target *targets=mesh_targets(m,row);
              uint32_t end=buffer->sends+buffer->uses,before=end,consumer=end;
              for(uint32_t j=buffer->sends;j<end;j++){
                uint32_t destination=(uint32_t)((targets[j].stream-first)/m->notice_bytes);
                if(destination==q)before=j;
                if(destination==local+function->worker)consumer=j;
              }
              if(consumer==end || (before!=end && before>consumer))continue;
            }
            uint32_t a=mesh_event_root(parents,source),b=mesh_event_root(parents,target);
            if(a==b)continue;
            parents[b]=a;successors[source]=1;break;
          }
        }
      }
    }
    for(uint32_t i=0;i<count;i++){
      uint64_t source=events->streams[i].slots;
      if(source && source<=rows)source=(uint64_t)mesh_event_root(parents,(uint32_t)source-1)+1;
      bindings[i]=(struct mesh_event_binding){source,i};
    }
    qsort(bindings,count,sizeof *bindings,mesh_event_compare);
    uint32_t streams=0;uint64_t used=0;
    uint64_t slots=(uint64_t)((char *)(events->streams+rows)-(char *)m);
    for(uint32_t i=0;i<count;){
      uint32_t end=i+1,capacity=1;
      while(end<count && bindings[end].source==bindings[i].source)end++;
      while(capacity<end-i)capacity*=2;
      struct mesh_stream *stream=&events->streams[streams++];
      *stream=(struct mesh_stream){.slots=slots+used*sizeof(uint64_t),.mask=capacity-1};
      uint64_t offset=(uint64_t)((char *)stream-(char *)m);
      for(uint32_t j=i;j<end;j++)mapping[(size_t)q*rows+bindings[j].index]=offset;
      used+=capacity;i=end;
    }
    events->count=streams;total_streams+=streams;
  }
  for(uint32_t row=0;row<rows;row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_relaxed)!=calls->context->client)continue;
    struct mesh_target *targets=mesh_targets(m,row);
    for(uint32_t i=0;i<buffer->sends+buffer->uses;i++)targets[i].stream=mesh_event_remap(m,first,mapping,targets[i].stream);
    if(buffer->return_slot)buffer->return_slot=mesh_event_remap(m,first,mapping,buffer->return_slot);
  }
  for(struct mesh_function *function=calls->functions;function;function=function->next)
    for(uint32_t frame=0;frame<calls->extent;frame++){
      struct mesh_call *call=&function->values[frame];call->return_slot=mesh_event_remap(m,first,mapping,call->return_slot);
    }
  for(uint32_t i=0;i<calls->count;i++){
    struct mesh_call_worker *worker=&calls->workers[i];
    error=mesh_event_reader_init(&worker->arrivals,m,first_queue+local+i);
    if(!error)error=mesh_event_reader_init(&worker->returns,m,first_queue+local+MESH_COMPUTE_THREADS+i);
    if(error)goto done;
  }
  for(uint32_t row=0;row<rows;row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_relaxed)==calls->context->client && buffer->sends &&
       atomic_load_explicit(&mesh_presence(m)[row],memory_order_relaxed))mesh_publish(m,row,1);
  }
  fprintf(stderr,"mesh events: bindings=%llu streams=%llu\n",(unsigned long long)total_bindings,(unsigned long long)total_streams);
done:
  free(bindings);free(mapping);free(successors);free(parents);free(last);
  return error;
}

/* design/algorithm-sources.md#programkernel_call */
int mesh_calls_start(struct mesh_calls *calls){
  struct hdr *m=calls->context->M;
  uint32_t rows=mesh_rows(m),workers=0;
  for(uint32_t row=0;row<rows;row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_relaxed)==calls->context->client)buffer->initial=atomic_load_explicit(&buffer->references,memory_order_relaxed);
  }
  if(calls->function_count>rows/calls->extent)return ENOMEM;
  calls->slots=calloc(calls->function_count?calls->function_count*calls->extent:1,sizeof *calls->slots);
  if(!calls->slots)return ENOMEM;
  for(struct mesh_function *function=calls->functions;function;function=function->next)
    if(!function->output_count)calls->return_count+=calls->extent;
  if(calls->return_count){
    calls->return_first=mesh_rows_alloc(calls->context,calls->return_count);
    if(calls->return_first==MESH_ABSENT)return errno;
  }
  uint32_t returned=0;
  for(struct mesh_function *function=calls->functions;function;function=function->next){
    size_t count=function->input_count+function->output_count;
    for(uint32_t index=0;index<calls->extent;index++){
      uint32_t slot=function->identity*calls->extent+index;
      uint32_t queue=mesh_notice_queue(m,calls->context->client,m->links*(m->qps+1)+MESH_COMPUTE_THREADS+function->worker);
      struct mesh_call *call=&function->values[index];
      calls->slots[slot]=call;
      for(size_t j=0;j<function->output_count;j++){
        uint32_t row=function->operands[index*count+function->input_count+j].row;
        struct mesh_buffer *buffer=&mesh_buffers(m)[row];
        buffer->channel=m->links*m->qps+MESH_COMPUTE_THREADS+function->worker;buffer->binding=slot;
        buffer->return_slot=mesh_event_bind(m,queue);
        buffer->publisher=(uint64_t)call->operands[function->input_count].row+1;
      }
      call->return_row=function->output_count?call->operands[function->input_count].row:calls->return_first+returned++;
      struct mesh_buffer *buffer=&mesh_buffers(m)[call->return_row];buffer->binding=slot;
      call->return_slot=function->output_count?buffer->return_slot:mesh_event_bind(m,queue);
    }
  }
  size_t width=(size_t)rows+1;
  size_t *offsets=calloc(calls->count*width,sizeof *offsets);
  if(!offsets)return ENOMEM;
  for(int pass=0;pass<2;pass++){
    for(struct mesh_function *function=calls->functions;function;function=function->next){
      struct mesh_call_worker *worker=&calls->workers[function->worker];
      size_t *indices=offsets+function->worker*width;
      workers|=UINT32_C(1)<<function->worker;
      uint32_t pending=0,initial=0;int varying=0;
      for(size_t i=0;i<=function->input_count;i++){
        struct mesh_section section;
        if(i==function->input_count){
          if(varying)continue;
          section=(struct mesh_section){.first=calls->first,.count=calls->extent,.stride=1};
          calls->root_workers|=UINT32_C(1)<<function->worker;
        } else {
          section=function->inputs[i];
          varying|=section.stride!=0;
          if(atomic_load_explicit(&mesh_presence(m)[section.first],memory_order_relaxed))continue;
        }
        initial++;pending+=section.stride!=0;
        for(uint32_t index=0;index<calls->extent;index++){
          uint32_t row=mesh_section_row(section,index);
          if(pass){
            struct mesh_call *call=&function->values[index];
            worker->targets[indices[row]++]=(struct mesh_use){
              function->submit,function->argument,call,call->operands,
              section.stride?UINT32_MAX:0,
              (uint32_t)function->input_count,0};
          } else {
            indices[row+1]++;
            mesh_publish_bind(calls->context,row,m->links*(m->qps+1)+function->worker,0);
          }
        }
      }
      if(!pass){
        function->pending=pending;
        for(uint32_t frame=0;frame<calls->extent;frame++)function->values[frame].pending=initial;
      }
    }
    for(uint32_t i=0;i<calls->count;i++){
      struct mesh_call_worker *worker=&calls->workers[i];
      size_t *indices=offsets+i*width;
      if(pass){
        for(uint32_t row=0,first=0;row<rows;row++){
          uint32_t end=(uint32_t)indices[row];
          if(first!=end){
            mesh_publish_bind(calls->context,row,m->links*(m->qps+1)+i,0)->index=first;
            worker->targets[first].end=end;
          }
          first=end;
        }
      } else {
        for(uint32_t row=0;row<rows;row++)indices[row+1]+=indices[row];
        if(indices[rows]>UINT32_MAX){free(offsets);return EOVERFLOW;}
        worker->targets=aligned_alloc(_Alignof(struct mesh_use),(indices[rows]?indices[rows]:1)*sizeof *worker->targets);
        if(!worker->targets){free(offsets);return ENOMEM;}
      }
    }
  }
  free(offsets);
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
    workers&=workers-1;
    pthread_detach(thread);
  }
  return 0;
}

/* design/algorithm-sources.md#program */
uint64_t mesh_calls_submit(struct mesh_calls *calls,uint32_t index){
  struct hdr *m=calls->context->M;
  uint64_t status=atomic_load_explicit(&m->result[calls->context->client>>63],memory_order_acquire).value;
  if(status>>62!=MESH_RESULT_BUSY)return status;
  if(!calls->root_workers)return 0;
  uint32_t frame=index%calls->extent;
  struct mesh_instance *instance=&calls->instances[frame];
  if(!atomic_load_explicit(&instance->available,memory_order_acquire))return MESH_RESULT(MESH_RESULT_BUSY,0,0);
  atomic_store_explicit(&instance->available,0,memory_order_relaxed);
  atomic_store_explicit(&instance->invocation,index,memory_order_relaxed);
  atomic_store_explicit(&instance->status,((struct mesh_status){MESH_RESULT(MESH_RESULT_BUSY,0,0),0}),memory_order_release);
  status=atomic_load_explicit(&m->result[calls->context->client>>63],memory_order_acquire).value;
  if(status>>62!=MESH_RESULT_BUSY){mesh_result_conclude(&instance->status,status);return status;}
  mesh_publish(m,calls->first+frame,(uint64_t)index+1);
  return 0;
}

/* design/algorithm-sources.md#meshresult */
uint64_t mesh_calls_result(struct mesh_calls *calls,uint32_t index){
  struct mesh_status status=atomic_load_explicit(&calls->instances[index%calls->extent].status,memory_order_acquire);
  if(status.completed==(uint64_t)index+1 || status.completed==UINT64_MAX)return 0;
  return status.value>>62==MESH_RESULT_SUCCESS?MESH_RESULT(MESH_RESULT_BUSY,0,0):status.value;
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_call_complete(struct mesh_call *call,int error){
  struct hdr *m=call->memory;
  struct mesh_operand *outputs=call->operands+call->input_count;
  size_t output_count=call->output_count;uint64_t stamp=(uint64_t)call->invocation+1;
  if(error)call->error=error;
  else for(size_t i=0;i<output_count;i++)mesh_publish(m,outputs[i].row,stamp);
  struct mesh_function *function=call->function;
  struct mesh_calls *calls=function->calls;
  _Atomic uint32_t *active=&calls->workers[function->worker].active;
  for(size_t i=0;i<function->consumed_count;i++){
    uint32_t input=function->consumed[i];
    mesh_buffer_release(m,call->operands[input].row);
  }
  if(error || !output_count)
    mesh_event_push(m,call->return_slot,call->return_row,0);
  else for(size_t i=0;i<output_count;i++)mesh_buffer_release(m,outputs[i].row);
  atomic_fetch_sub_explicit(active,1,memory_order_release);
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
    uint64_t payload=(uint64_t)m->block*m->pgsz;
    mesh_publish_bind(context,row,link,1)->count+=(uint32_t)((section.bytes+payload-1)/payload);
  }
  mesh_transfers(m,context->client,queue,receive)[index]=(struct mesh_transfer){section.first,identity,section.count,section.stride,MESH_ABSENT,section.bytes};
  atomic_store_explicit(length,index+1,memory_order_release);
  return 0;
}

/* design/algorithm-sources.md#programcopy */
int mesh_transfers_prepare(struct mesh_ctx *context){
  struct hdr *m=context->M;
  uint32_t *positions=calloc(m->links?m->links:1,sizeof *positions);
  if(!positions)return ENOMEM;
  uint64_t first=m->notice_off+(uint64_t)mesh_notice_queue(m,context->client,0)*m->notice_bytes;
  for(uint32_t row=0;row<m->rows;row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_relaxed)!=context->client)continue;
    struct mesh_target *targets=mesh_targets(m,row);
    for(uint32_t i=0;i<buffer->sends;i++){
      uint32_t link=(uint32_t)((targets[i].stream-first)/m->notice_bytes);
      targets[i].index=positions[link];positions[link]+=targets[i].count;
    }
  }
  free(positions);
  for(uint32_t q=0;q<m->links*m->qps;q++){
    uint32_t count=atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE)),pages=0;
    struct mesh_transfer *transfers=mesh_transfers(m,context->client,q,MESH_RECEIVE);
    for(uint32_t i=0;i<count;i++)pages+=transfers[i].count*mesh_buffers(m)[transfers[i].local_row].pages;
    if(!pages)continue;
    uint32_t page=mesh_arena_alloc(context,pages,m->block);
    if(page==MESH_ABSENT)return errno;
    struct mesh_pool *pool=&mesh_pools(m)[page/m->block];pool->pages=pages;
    atomic_store_explicit(&pool->owner,context->client,memory_order_release);
    for(uint32_t i=0;i<count;i++)transfers[i].pool=page;
  }
  return 0;
}

/* design/algorithm-sources.md#programcopy */
void mesh_transfers_start(struct mesh_ctx *context){
  struct hdr *m=context->M;
  atomic_store_explicit(&m->configured,context->client,memory_order_release);
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
      buffer->publisher=(UINT64_C(1)<<63)|channel/m->qps;
    }
    atomic_store_explicit(&buffer->owner,context->client,memory_order_release);
  }
  mesh_bits_set(m,MESH_ROW_HOT,first,rows);
  if(channel==MESH_ABSENT)for(uint32_t row=first;row<first+rows;row+=stride){
    uint32_t page=mesh_arena_alloc(context,(uint32_t)span,m->block);
    if(page==MESH_ABSENT){
      int error=errno;
      for(uint32_t r=first;r<first+rows;r+=stride){mesh_buffer_release(m,r);mesh_buffer_release(m,r);}
      mesh_rows_release(context,first,rows);return error;
    }
    mesh_backing_bind(context,row,(uint32_t)span,page,(row-first)/stride);
  }
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,stride,channel};
  return 0;
}
/* design/algorithm-sources.md#programtensor */
uint32_t mesh_row_page(struct mesh_ctx *context,uint32_t row,uint32_t chunk){return atomic_load_explicit(mesh_page(context->M)+row+chunk,memory_order_acquire);}
/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return mesh_at(context->M,mesh_row_page(context,mesh_section_row(section,index),0));}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){
  mesh_buffers(context->M)[section.first].publisher=UINT64_MAX;
  atomic_store_explicit(&mesh_presence(context->M)[section.first],1,memory_order_release);
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
      if(atomic_load_explicit(&mesh_presence(context->M)[mesh_section_row(section,slot)],memory_order_acquire)==stamp)goto filled;
filled:;
  }
}
