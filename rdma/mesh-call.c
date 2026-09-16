#include "mesh-call.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct mesh_call {
  struct mesh_function *function;
  struct mesh_operand *operands;
  uint32_t pending,index,invocation;
};
struct mesh_slot { struct mesh_call *call; uint32_t remaining; };
struct mesh_function {
  struct mesh_calls *calls;
  struct mesh_section *inputs;
  struct mesh_call *values;
  struct mesh_operand *operands;
  uint32_t *consumed,*available,*references;
  unsigned char *placed;
  size_t input_count,output_count,consumed_count;
  uint32_t worker,identity,free_count;
  mesh_submit submit;
  mesh_rearm rearm;
  mesh_dispose dispose;
  void *argument;
  struct mesh_function *next;
};
struct mesh_use { uint32_t function,input; };
struct mesh_call_worker {
  struct mesh_calls *calls;
  size_t *offsets;
  struct mesh_use *targets;
  uint32_t index;
};
struct mesh_calls {
  struct mesh_ctx *context;
  struct mesh_function *functions,**program;
  struct mesh_call_worker workers[MESH_COMPUTE_THREADS];
  uint32_t count,extent,first,return_first,root_workers,function_count;
  struct mesh_slot *slots;
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
  for(uint32_t i=0;i<calls->count;i++){free(calls->workers[i].offsets);free(calls->workers[i].targets);}
  for(struct mesh_function *function=calls->functions;function;){
    struct mesh_function *next=function->next;
    if(function->dispose)function->dispose(function->argument);
    free(function->available);free(function->operands);free(function->values);free(function->inputs);free(function);
    function=next;
  }
  if(calls->first!=MESH_ABSENT)mesh_rows_release(calls->context,calls->first,calls->extent);
  if(calls->return_first!=MESH_ABSENT)mesh_rows_release(calls->context,calls->return_first,calls->function_count*calls->extent);
  if(calls->dispose)calls->dispose(calls->owner);
  free(calls->slots);free(calls->program);free(calls);
}

/* design/algorithm-sources.md#meshresult */
static void mesh_calls_cancel(struct mesh_calls *calls,uint32_t first,uint32_t count){
  for(struct mesh_function *function=calls->functions;function;function=function->next)
    if(function->worker>=first && function->worker<first+count)for(uint32_t index=0;index<calls->extent;index++)
      if(function->values[index].pending && (uint32_t)atomic_fetch_sub_explicit(&calls->instances[index].references,1,memory_order_acq_rel)==1)
        mesh_calls_release(calls,1);
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *context,uint32_t workers,uint32_t count,void *owner,mesh_dispose dispose){
  if(!context || !context->M || !workers || workers>MESH_COMPUTE_THREADS || !count){errno=EINVAL;return NULL;}
  struct mesh_calls *calls=calloc(1,sizeof *calls);
  if(!calls)return NULL;
  calls->context=context;calls->count=workers;calls->extent=count;calls->first=calls->return_first=MESH_ABSENT;atomic_init(&calls->references,1);
  for(uint32_t i=0;i<workers;i++){
    calls->workers[i]=(struct mesh_call_worker){.calls=calls,.index=i,.offsets=calloc((size_t)mesh_rows(context->M)+1,sizeof(size_t))};
    if(!calls->workers[i].offsets){mesh_calls_release(calls,1);return NULL;}
  }
  calls->first=mesh_rows_alloc(context,count);
  if(calls->first==MESH_ABSENT){mesh_calls_release(calls,1);return NULL;}
  calls->instances=mesh_instances(context->M,context->client);
  context->M->instance_count[context->client>>63]=count;
  for(uint32_t i=0;i<count;i++)atomic_store_explicit(&calls->instances[i].status,MESH_RESULT(MESH_RESULT_BUSY,0,0),memory_order_relaxed);
  calls->owner=owner;calls->dispose=dispose;
  return calls;
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_function *mesh_call_bind(struct mesh_calls *calls,uint32_t worker,
  const struct mesh_section *inputs,const struct mesh_section *views,size_t input_count,
  const struct mesh_section *outputs,size_t output_count,mesh_submit submit,mesh_rearm rearm,void *argument,mesh_dispose dispose){
  if(worker>=calls->count || !submit || atomic_load(&calls->running)){errno=EINVAL;return NULL;}
  struct mesh_function *function=calloc(1,sizeof *function);
  if(!function)return NULL;
  size_t count=input_count+output_count,extent=calls->extent;
  function->inputs=calloc(1,(input_count?input_count:1)*sizeof *inputs+input_count*(sizeof *function->consumed+sizeof *function->placed));
  function->operands=calloc(extent*(count?count:1),sizeof *function->operands);
  function->values=calloc(extent,sizeof *function->values);
  function->available=calloc(extent+output_count,sizeof *function->available);
  if(!function->inputs || !function->operands || !function->values || !function->available){errno=ENOMEM;goto failed;}
  function->references=function->available+extent;function->free_count=(uint32_t)extent;
  for(uint32_t i=0;i<extent;i++)function->available[i]=(uint32_t)extent-i-1;
  function->consumed=(uint32_t *)(function->inputs+input_count);
  function->placed=(unsigned char *)(function->consumed+input_count);
  for(size_t i=0;i<input_count;i++){
    function->placed[i]=views[i].first!=inputs[i].first;
    if(inputs[i].stride)function->consumed[function->consumed_count++]=(uint32_t)i;
  }
  memcpy(function->inputs,inputs,input_count*sizeof *inputs);
  function->calls=calls;function->worker=worker;function->input_count=input_count;function->output_count=output_count;
  function->submit=submit;function->rearm=rearm;function->argument=argument;function->dispose=dispose;
  struct hdr *m=calls->context->M;
  for(size_t i=0;i<input_count;i++)for(uint32_t index=0;index<(inputs[i].stride?extent:1);index++)
    mesh_buffer_retain(m,mesh_section_row(inputs[i],index),inputs[i].pages);
  for(uint32_t index=0;index<extent;index++){
    struct mesh_call *call=&function->values[index];
    *call=(struct mesh_call){.function=function,.invocation=index};
    for(size_t i=0;i<count;i++){
      struct mesh_section section=i<input_count?views[i]:outputs[i-input_count];
      uint32_t page=atomic_load_explicit(&mesh_page(m)[mesh_section_row(section,index)],memory_order_acquire);
      function->operands[index*count+i]=(struct mesh_operand){.data=page==MESH_ABSENT?NULL:mesh_at(m,page),.bytes=section.bytes,.page=page,.index=section.stride?index:0,.row=mesh_section_row(section,index)};
    }
  }
  function->identity=calls->function_count++;
  function->next=calls->functions;calls->functions=function;
  return function;
failed:
  free(function->available);free(function->inputs);free(function->operands);free(function->values);free(function);
  return NULL;
}

/* design/algorithm-sources.md#programkernel_call */
static void mesh_call_input(struct mesh_function *function,struct mesh_operand *operand,uint32_t input,uint32_t row){
  operand->row=row;
  if(!function->placed[input]){
    struct hdr *m=function->calls->context->M;
    struct mesh_section section=function->inputs[input];
    operand->page=atomic_load_explicit(&mesh_page(m)[row],memory_order_acquire);
    operand->data=mesh_at(m,operand->page);operand->index=section.stride?(row-section.first)/section.stride:0;
  }
}

/* design/algorithm-sources.md#programkernel_call */
static void *mesh_call_progress(void *argument){
  struct mesh_call_worker *worker=argument;
  struct mesh_calls *calls=worker->calls;
  struct hdr *m=calls->context->M;
  uint32_t queue=mesh_notice_queue(m,calls->context->client,m->links*(m->qps+1)+worker->index);
  struct mesh_notice_reader reader=mesh_notice_reader_init(m,queue),returns=mesh_notice_reader_init(m,queue+MESH_COMPUTE_THREADS);
  pthread_setname_np("mesh.numerical");
  while(atomic_load_explicit(&calls->running,memory_order_acquire)){
    uint32_t row;
    for(;;){
      row=mesh_notice_take(&returns);
      if(row!=MESH_ABSENT){
        struct mesh_slot *slot=&calls->slots[mesh_buffers(m)[row].binding];
        if(!--slot->remaining){
          struct mesh_call *call=slot->call;
          struct mesh_function *function=call->function;
          if(function->rearm)function->rearm(call->index,function->argument);
          function->available[function->free_count++]=call->index;
        }
      }
      row=mesh_notice_take(&reader);
      if(row==MESH_ABSENT)break;
      for(size_t i=worker->offsets[row];i<worker->offsets[row+1];i++){
        struct mesh_use use=worker->targets[i];
        struct mesh_function *function=calls->program[use.function];
        size_t count=function->input_count+function->output_count;
        int shared=use.input!=MESH_ABSENT && !function->inputs[use.input].stride;
        uint32_t first=shared?0:mesh_buffers(m)[row].invocation,end=shared?calls->extent:first+1;
        if(shared)
          for(uint32_t index=0;index<calls->extent;index++)mesh_call_input(function,&function->operands[index*count+use.input],use.input,row);
        for(uint32_t index=first;index<end;index++){
          struct mesh_call *call=&function->values[index];
          if(!shared){
            if(!call->operands){
              call->index=function->available[--function->free_count];
              calls->slots[function->identity*calls->extent+call->index]=(struct mesh_slot){call,(uint32_t)function->output_count+1};
              call->operands=function->operands+call->index*count;
              for(size_t j=0;j<count;j++)call->operands[j].invocation=call->invocation;
              for(size_t j=0;j<function->output_count;j++){
                uint32_t output=call->operands[function->input_count+j].row;
                atomic_fetch_add_explicit(&mesh_buffers(m)[output].ownership,function->references[j],memory_order_relaxed);
                atomic_store_explicit(&mesh_presence(m)[output],0,memory_order_relaxed);
                mesh_buffers(m)[output].invocation=call->invocation;
              }
            }
            if(use.input!=MESH_ABSENT)mesh_call_input(function,&call->operands[use.input],use.input,row);
          }
          if(!--call->pending)function->submit(call,call->index,function->argument,call->operands,call->operands+function->input_count);
        }
      }
    }
  }
  mesh_calls_cancel(calls,worker->index,1);
  mesh_calls_release(calls,1);
  return NULL;
}

/* design/algorithm-sources.md#programkernel_call */
int mesh_calls_start(struct mesh_calls *calls){
  struct hdr *m=calls->context->M;
  uint32_t rows=mesh_rows(m),workers=0;
  if(calls->function_count>rows/calls->extent)return ENOMEM;
  calls->program=calloc(calls->function_count?calls->function_count:1,sizeof *calls->program);
  calls->slots=calloc(calls->function_count?calls->function_count*calls->extent:1,sizeof *calls->slots);
  if(!calls->program || !calls->slots)return ENOMEM;
  if(calls->function_count){
    calls->return_first=mesh_rows_alloc(calls->context,calls->function_count*calls->extent);
    if(calls->return_first==MESH_ABSENT)return errno;
  }
  for(struct mesh_function *function=calls->functions;function;function=function->next){
    size_t count=function->input_count+function->output_count;
    for(uint32_t index=0;index<calls->extent;index++){
      uint32_t slot=function->identity*calls->extent+index;
      mesh_buffers(m)[calls->return_first+slot].binding=slot;
      for(size_t j=0;j<function->output_count;j++){
        uint32_t row=function->operands[index*count+function->input_count+j].row;
        struct mesh_buffer *buffer=&mesh_buffers(m)[row];
        uint32_t references=(uint32_t)atomic_load_explicit(&buffer->ownership,memory_order_relaxed);
        function->references[j]=references;
        buffer->channel=m->links*m->qps+MESH_COMPUTE_THREADS+function->worker;buffer->binding=slot;
        atomic_fetch_sub_explicit(&buffer->ownership,references,memory_order_relaxed);
      }
    }
  }
  for(uint32_t index=0;index<calls->extent;index++){
    uint32_t transfers_count=0;
    for(uint32_t q=0;q<m->links*m->qps;q++)for(int d=0;d<2;d++){
      uint32_t count=atomic_load_explicit(mesh_order_length(m,calls->context->client,q,d),memory_order_relaxed);
      struct mesh_transfer *transfers=mesh_transfers(m,calls->context->client,q,d);
      for(uint32_t i=0;i<count;i++)transfers_count+=transfers[i].stride?index<transfers[i].count:transfers[i].count;
    }
    uint64_t references=((uint64_t)transfers_count<<32)|calls->function_count;
    atomic_store_explicit(&calls->instances[index].references,references,memory_order_relaxed);
    atomic_store_explicit(&calls->instances[index].status,references?MESH_RESULT(MESH_RESULT_BUSY,0,0):0,memory_order_relaxed);
  }
  for(int pass=0;pass<2;pass++){
    for(struct mesh_function *function=calls->functions;function;function=function->next){
      struct mesh_call_worker *worker=&calls->workers[function->worker];
      calls->program[function->identity]=function;
      workers|=UINT32_C(1)<<function->worker;
      uint32_t pending=0;int varying=0;
      for(size_t i=0;i<=function->input_count;i++){
        struct mesh_section section;
        uint32_t input;
        if(i==function->input_count){
          if(varying)continue;
          section=(struct mesh_section){.first=calls->first,.count=calls->extent,.stride=1};input=MESH_ABSENT;
          calls->root_workers|=UINT32_C(1)<<function->worker;
        } else {
          section=function->inputs[i];input=(uint32_t)i;
          varying|=section.stride!=0;
          if(atomic_load_explicit(&mesh_presence(m)[section.first],memory_order_relaxed))continue;
        }
        pending++;
        for(uint32_t index=0;index<section.count;index++){
          uint32_t row=mesh_section_row(section,index);
          mesh_buffers(m)[row].uses|=UINT32_C(1)<<function->worker;
          if(pass)worker->targets[worker->offsets[row]++]=(struct mesh_use){function->identity,input};
          else worker->offsets[row+1]++;
        }
      }
      if(!pass)for(uint32_t index=0;index<calls->extent;index++)function->values[index].pending=pending;
    }
    for(uint32_t i=0;i<calls->count;i++){
      struct mesh_call_worker *worker=&calls->workers[i];
      if(pass){memmove(worker->offsets+1,worker->offsets,(size_t)rows*sizeof *worker->offsets);worker->offsets[0]=0;}
      else {
        for(uint32_t row=0;row<rows;row++)worker->offsets[row+1]+=worker->offsets[row];
        worker->targets=calloc(worker->offsets[rows]?worker->offsets[rows]:1,sizeof *worker->targets);
        if(!worker->targets)return ENOMEM;
      }
    }
  }
  atomic_store_explicit(&calls->running,1,memory_order_release);
  atomic_fetch_add_explicit(&calls->references,(uint32_t)__builtin_popcount(workers)+(calls->function_count?calls->extent:0),memory_order_relaxed);
  while(workers){
    uint32_t i=(uint32_t)__builtin_ctz(workers);
    pthread_t thread;
    int error=pthread_create(&thread,NULL,mesh_call_progress,&calls->workers[i]);
    if(error){
      atomic_store(&calls->running,0);
      mesh_calls_cancel(calls,i,calls->count-i);
      mesh_calls_release(calls,(uint32_t)__builtin_popcount(workers));return error;
    }
    workers&=workers-1;
    pthread_detach(thread);
  }
  return 0;
}

/* design/algorithm-sources.md#program */
void mesh_calls_submit(struct mesh_calls *calls,uint32_t index){
  mesh_buffers(calls->context->M)[calls->first+index].invocation=index;
  uint32_t workers=calls->root_workers;
  while(workers){
    uint32_t worker=(uint32_t)__builtin_ctz(workers);workers&=workers-1;
    mesh_notice_push(calls->context->M,mesh_notice_queue(calls->context->M,calls->context->client,calls->context->M->links*(calls->context->M->qps+1)+worker),calls->first+index);
  }
}

/* design/algorithm-sources.md#meshresult */
uint64_t mesh_calls_result(struct mesh_calls *calls,uint32_t index){
  return atomic_load_explicit(&calls->instances[index].status,memory_order_acquire);
}

/* design/algorithm-sources.md#programkernel_call */
static void mesh_call_finish(struct mesh_call *call){
  struct mesh_function *function=call->function;
  struct mesh_calls *calls=function->calls;
  for(size_t i=0;i<function->consumed_count;i++){
    uint32_t input=function->consumed[i];
    mesh_buffer_release(calls->context->M,call->operands[input].row,function->inputs[input].pages);
  }
  uint32_t invocation=call->invocation;
  mesh_notice_push(calls->context->M,mesh_notice_queue(calls->context->M,calls->context->client,
    calls->context->M->links*(calls->context->M->qps+1)+MESH_COMPUTE_THREADS+function->worker),
    calls->return_first+function->identity*calls->extent+call->index);
  if((uint32_t)mesh_instance_release(&calls->instances[invocation],1)==1)mesh_calls_release(calls,1);
}

/* design/algorithm-sources.md#meshresult */
void mesh_call_fail(struct mesh_call *call,int error){
  struct mesh_function *function=call->function;
  mesh_instance_conclude(&function->calls->instances[call->invocation],MESH_RESULT(MESH_RESULT_FUNCTION,function->identity,error));
  mesh_call_finish(call);
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_call_complete(struct mesh_call *call){
  struct mesh_function *function=call->function;
  for(size_t i=0;i<function->output_count;i++)mesh_publish(function->calls->context->M,call->operands[function->input_count+i].row);
  mesh_call_finish(call);
}

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
    mesh_buffer_retain(m,row,section.pages);
    uint32_t link=queue/m->qps;
    uint64_t bit=UINT64_C(1)<<(link%64),uses=atomic_fetch_or_explicit(&mesh_send_uses(m,row)[link/64],bit,memory_order_relaxed);
    if(!(uses&bit) && atomic_load_explicit(&mesh_presence(m)[row],memory_order_relaxed))
      mesh_notice_push(m,mesh_notice_queue(m,context->client,link),row);
  }
  mesh_transfers(m,context->client,queue,receive)[index]=(struct mesh_transfer){section.first,identity,section.count,section.stride,m->block,MESH_ABSENT,section.bytes};
  atomic_store_explicit(length,index+1,memory_order_release);
  return 0;
}

/* design/algorithm-sources.md#programcopy */
int mesh_transfers_prepare(struct mesh_ctx *context){
  struct hdr *m=context->M;
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
  uint32_t pages=count*(uint32_t)span,first=mesh_rows_alloc(context,pages);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t row=first;row<first+pages;row+=(uint32_t)span)
    mesh_buffers(m)[row]=(struct mesh_buffer){.ownership=2,.first=row,.pages=(uint32_t)span,.channel=channel,.owner=context->client};
  mesh_bits_set(m,MESH_ROW_HOT,first,pages);
  if(channel==MESH_ABSENT)for(uint32_t row=first;row<first+pages;row+=(uint32_t)span){
    uint32_t page=mesh_arena_alloc(context,(uint32_t)span,m->block);
    if(page==MESH_ABSENT){
      int error=errno;
      mesh_buffer_release(context->M,first,pages);mesh_buffer_release(context->M,first,pages);
      mesh_rows_release(context,first,pages);return error;
    }
    mesh_backing_bind(context,row,(uint32_t)span,page);
  }
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,(uint32_t)span,channel};
  return 0;
}
/* design/algorithm-sources.md#programtensor */
uint32_t mesh_row_page(struct mesh_ctx *context,uint32_t row){return atomic_load_explicit(&mesh_page(context->M)[row],memory_order_acquire);}
/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return mesh_at(context->M,mesh_row_page(context,mesh_section_row(section,index)));}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){mesh_publish(context->M,section.first);}
/* design/algorithm-sources.md#programtensor */
void mesh_section_release(struct mesh_ctx *context,struct mesh_section section){
  mesh_buffer_release(context->M,section.first,section.count*section.pages);
  mesh_rows_release(context,section.first,section.count*section.pages);
}
/* design/algorithm-sources.md#collectivesync_on_remote_fill */
void mesh_sync_on_remote_fill(struct mesh_ctx *context,const struct mesh_section *sections,size_t count,uint32_t index){
  for(size_t i=0;i<count;i++){
    struct mesh_section section=sections[i];
    uint32_t stamp=section.stride?index+1:1;
    for(;;)for(uint32_t slot=0;slot<section.count;slot++)
      if(atomic_load_explicit(&mesh_presence(context->M)[mesh_section_row(section,slot)],memory_order_acquire)==stamp)goto filled;
filled:;
  }
}
