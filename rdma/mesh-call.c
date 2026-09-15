#include "mesh-call.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct mesh_call {
  struct mesh_function *function;
  struct mesh_operand *operands;
  uint32_t pending,index;
  int completed;
};
struct mesh_function {
  struct mesh_calls *calls;
  struct mesh_section *inputs,*outputs;
  struct mesh_call *values;
  struct mesh_operand *operands;
  uint32_t *remote;
  size_t input_count,output_count,remote_count;
  uint32_t worker;
  mesh_submit submit;
  mesh_dispose dispose;
  void *argument;
  struct mesh_function *next;
};
struct mesh_call_worker {
  struct mesh_calls *calls;
  size_t *offsets;
  struct mesh_call **targets;
  uint32_t index;
};
struct mesh_calls {
  struct mesh_ctx *context;
  struct mesh_function *functions;
  struct mesh_call_worker workers[MESH_COMPUTE_THREADS];
  uint32_t count,extent,first,root_workers;
  _Atomic uint32_t references;
  _Atomic int running;
  void *owner;
  mesh_dispose dispose;
};

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* design/algorithm-sources.md#programtensor */
static void mesh_calls_release(struct mesh_calls *calls){
  if(atomic_fetch_sub_explicit(&calls->references,1,memory_order_acq_rel)!=1)return;
  struct hdr *m=calls->context->M;
  for(uint32_t i=0;i<calls->count;i++){free(calls->workers[i].offsets);free(calls->workers[i].targets);}
  for(struct mesh_function *function=calls->functions;function;){
    struct mesh_function *next=function->next;
    for(uint32_t index=0;index<calls->extent;index++)for(size_t i=0;i<function->input_count;i++){
      struct mesh_section section=function->inputs[i];
      uint32_t row=mesh_section_row(section,index);
      mesh_buffers(m)[row].uses&=~(UINT32_C(1)<<function->worker);
      if(!function->values[index].completed)mesh_buffer_release(m,row,section.pages);
    }
    if(function->dispose)function->dispose(function->argument);
    free(function->operands);free(function->values);free(function->inputs);free(function);
    function=next;
  }
  if(calls->first!=MESH_ABSENT)mesh_rows_release(calls->context,calls->first,calls->extent);
  if(calls->dispose)calls->dispose(calls->owner);
  free(calls);
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *context,uint32_t workers,uint32_t count,void *owner,mesh_dispose dispose){
  if(!context || !context->M || !workers || workers>MESH_COMPUTE_THREADS || !count){errno=EINVAL;return NULL;}
  struct mesh_calls *calls=calloc(1,sizeof *calls);
  if(!calls)return NULL;
  calls->context=context;calls->count=workers;calls->extent=count;calls->first=MESH_ABSENT;atomic_init(&calls->references,1);
  for(uint32_t i=0;i<workers;i++){
    calls->workers[i]=(struct mesh_call_worker){.calls=calls,.index=i,.offsets=calloc((size_t)mesh_rows(context->M)+1,sizeof(size_t))};
    if(!calls->workers[i].offsets){mesh_calls_release(calls);return NULL;}
  }
  calls->first=mesh_rows_alloc(context,count);
  if(calls->first==MESH_ABSENT){mesh_calls_release(calls);return NULL;}
  calls->owner=owner;calls->dispose=dispose;
  return calls;
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_function *mesh_call_bind(struct mesh_calls *calls,uint32_t worker,
  const struct mesh_section *inputs,size_t input_count,
  const struct mesh_section *outputs,size_t output_count,mesh_submit submit,void *argument,mesh_dispose dispose){
  if(worker>=calls->count || !submit || atomic_load(&calls->running)){errno=EINVAL;return NULL;}
  struct mesh_function *function=calloc(1,sizeof *function);
  if(!function)return NULL;
  size_t count=input_count+output_count,extent=calls->extent;
  function->inputs=calloc(1,(count?count:1)*sizeof *inputs+input_count*sizeof *function->remote);
  function->operands=calloc(extent*(count?count:1),sizeof *function->operands);
  function->values=calloc(extent,sizeof *function->values);
  if(!function->inputs || !function->operands || !function->values){errno=ENOMEM;goto failed;}
  function->outputs=function->inputs+input_count;
  function->remote=(uint32_t *)(function->outputs+output_count);
  for(size_t i=0;i<input_count;i++)if(inputs[i].receive)function->remote[function->remote_count++]=(uint32_t)i;
  memcpy(function->inputs,inputs,input_count*sizeof *inputs);
  memcpy(function->outputs,outputs,output_count*sizeof *outputs);
  function->calls=calls;function->worker=worker;function->input_count=input_count;function->output_count=output_count;
  function->submit=submit;function->argument=argument;function->dispose=dispose;
  struct hdr *m=calls->context->M;
  size_t retained=0;
  for(uint32_t index=0;index<extent;index++)for(size_t i=0;i<input_count;i++){
    int error=mesh_buffer_retain(m,mesh_section_row(inputs[i],index),inputs[i].pages);
    if(error){
      for(size_t j=0;j<retained;j++){
        struct mesh_section section=inputs[j%input_count];
        mesh_buffer_release(m,mesh_section_row(section,(uint32_t)(j/input_count)),section.pages);
      }
      errno=error;goto failed;
    }
    retained++;
  }
  for(uint32_t index=0;index<extent;index++){
    struct mesh_call *call=&function->values[index];
    *call=(struct mesh_call){.function=function,.index=index,.operands=function->operands+index*count};
    for(size_t i=0;i<count;i++){
      struct mesh_section section=function->inputs[i];
      uint32_t page=atomic_load_explicit(&mesh_page(m)[mesh_section_row(section,index)],memory_order_acquire);
      call->operands[i]=(struct mesh_operand){mesh_at(m,page),section.bytes,page,index};
    }
  }
  function->next=calls->functions;calls->functions=function;
  return function;
failed:
  free(function->inputs);free(function->operands);free(function->values);free(function);
  return NULL;
}

/* design/algorithm-sources.md#programkernel_call */
static void mesh_call_submit(struct mesh_call *call){
  struct mesh_function *function=call->function;
  struct mesh_ctx *context=function->calls->context;
  for(size_t i=0;i<function->remote_count;i++){
    uint32_t operand=function->remote[i];
    struct mesh_section section=function->inputs[operand];
    uint32_t page=atomic_load_explicit(&mesh_page(context->M)[mesh_section_row(section,call->index)],memory_order_acquire);
    call->operands[operand].data=mesh_at(context->M,page);call->operands[operand].page=page;
  }
  atomic_fetch_add_explicit(&function->calls->references,1,memory_order_relaxed);
  function->submit(call,call->index,function->argument,call->operands,call->operands+function->input_count);
}

/* design/algorithm-sources.md#programkernel_call */
static void *mesh_call_progress(void *argument){
  struct mesh_call_worker *worker=argument;
  struct mesh_calls *calls=worker->calls;
  struct hdr *m=calls->context->M;
  uint32_t queue=mesh_notice_queue(m,calls->context->client,m->links+worker->index);
  pthread_setname_np("mesh.numerical");
  while(atomic_load_explicit(&calls->running,memory_order_acquire)){
    uint32_t row=mesh_notice_take(m,queue);
    while(row!=MESH_ABSENT){
      uint32_t next=mesh_notice_next(m,queue,row);
      for(size_t i=worker->offsets[row];i<worker->offsets[row+1];i++){
        struct mesh_call *call=worker->targets[i];
        if(!--call->pending)mesh_call_submit(call);
      }
      row=next;
    }
  }
  mesh_calls_release(calls);
  return NULL;
}

/* design/algorithm-sources.md#programkernel_call */
int mesh_calls_start(struct mesh_calls *calls){
  struct hdr *m=calls->context->M;
  uint32_t rows=mesh_rows(m);
  for(int pass=0;pass<2;pass++){
    for(struct mesh_function *function=calls->functions;function;function=function->next){
      struct mesh_call_worker *worker=&calls->workers[function->worker];
      for(uint32_t index=0;index<calls->extent;index++){
        struct mesh_call *call=&function->values[index];
        int varying=0;
        for(size_t i=0;i<=function->input_count;i++){
          uint32_t row;
          if(i==function->input_count){
            if(varying)continue;
            row=calls->first+index;calls->root_workers|=UINT32_C(1)<<function->worker;
          } else {
            struct mesh_section section=function->inputs[i];
            row=mesh_section_row(section,index);
            if(mesh_bit(m,MESH_PRESENT,row))continue;
            varying|=section.stride!=0;
            mesh_buffers(m)[row].uses|=UINT32_C(1)<<function->worker;
          }
          if(pass)worker->targets[worker->offsets[row]++]=call;
          else {worker->offsets[row+1]++;call->pending++;}
        }
      }
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
  for(uint32_t i=0;i<calls->count;i++){
    pthread_t thread;
    atomic_fetch_add_explicit(&calls->references,1,memory_order_relaxed);
    int error=pthread_create(&thread,NULL,mesh_call_progress,&calls->workers[i]);
    if(error){atomic_store(&calls->running,0);mesh_calls_release(calls);return error;}
    pthread_detach(thread);
  }
  return 0;
}

/* design/algorithm-sources.md#program */
void mesh_calls_submit(struct mesh_calls *calls,uint32_t index){
  uint32_t workers=calls->root_workers;
  while(workers){
    uint32_t worker=(uint32_t)__builtin_ctz(workers);workers&=workers-1;
    mesh_notice_push(calls->context->M,mesh_notice_queue(calls->context->M,calls->context->client,calls->context->M->links+worker),calls->first+index);
  }
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_call_fail(struct mesh_call *call,int error){
  struct mesh_function *function=call->function;
  struct mesh_calls *calls=function->calls;
  struct mesh_ctx *context=calls->context;
  context->M->port.code=error;context->M->port.domain=4;
  for(size_t i=0;i<function->input_count;i++)
    mesh_buffer_release(context->M,mesh_section_row(function->inputs[i],call->index),function->inputs[i].pages);
  call->completed=1;
  mesh_calls_release(calls);
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_call_complete(struct mesh_call *call){
  struct mesh_function *function=call->function;
  struct mesh_calls *calls=function->calls;
  struct mesh_ctx *context=calls->context;
  for(size_t i=0;i<function->output_count;i++){
    struct mesh_section section=function->outputs[i];
    uint32_t row=mesh_section_row(section,call->index);
    mesh_publish(context->M,row);
  }
  for(size_t i=0;i<function->input_count;i++)
    mesh_buffer_release(context->M,mesh_section_row(function->inputs[i],call->index),function->inputs[i].pages);
  call->completed=1;
  mesh_calls_release(calls);
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_calls_destroy(struct mesh_calls *calls){
  if(!calls)return;
  atomic_store_explicit(&calls->running,0,memory_order_release);
  mesh_calls_release(calls);
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
    int error=mesh_buffer_retain(m,row,section.pages);if(error)return error;
    uint32_t link=queue/m->qps;
    uint64_t bit=UINT64_C(1)<<(link%64),uses=atomic_fetch_or_explicit(&mesh_send_uses(m,row)[link/64],bit,memory_order_relaxed);
    if(!(uses&bit) && mesh_bit(m,MESH_PRESENT,row))
      mesh_notice_push(m,mesh_notice_queue(m,context->client,link),row);
  }
  mesh_transfers(m,context->client,queue,receive)[index]=(struct mesh_transfer){section.first,identity,section.count,section.stride,m->block,section.bytes};
  atomic_store_explicit(length,index+1,memory_order_release);
  return 0;
}

/* design/algorithm-sources.md#programcopy */
int mesh_transfers_prepare(struct mesh_ctx *context){
  struct hdr *m=context->M;
  context->receives=calloc(m->links?m->links*m->qps:1,sizeof *context->receives);
  if(!context->receives)return ENOMEM;
  for(uint32_t q=0;q<m->links*m->qps;q++){
    uint32_t count=atomic_load(mesh_order_length(m,context->client,q,MESH_RECEIVE)),pages=0;
    struct mesh_transfer *transfers=mesh_transfers(m,context->client,q,MESH_RECEIVE);
    for(uint32_t i=0;i<count;i++)pages+=transfers[i].count*mesh_buffers(m)[transfers[i].local_row].pages;
    if(!pages)continue;
    uint32_t page=mesh_arena_alloc(context,pages,m->block);
    if(page==MESH_ABSENT)return errno;
    context->receives[q]=(struct mesh_range){page,pages};
    for(uint32_t i=0;i<count;i++)for(uint32_t value=0;value<transfers[i].count;value++){
      uint32_t row=transfers[i].local_row+value*transfers[i].stride,span=mesh_buffers(m)[row].pages;
      mesh_backing_bind(context,row,span,page);
      page+=span;
    }
  }
  return 0;
}

/* design/algorithm-sources.md#programcopy */
void mesh_transfers_start(struct mesh_ctx *context){
  struct hdr *m=context->M;
  atomic_store_explicit(&m->configured,context->client,memory_order_release);
}

/* design/algorithm-sources.md#programtensor */
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,int receive,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || !count)return EINVAL;
  size_t quantum=(size_t)m->block*m->pgsz;
  if(bytes>(size_t)mesh_rows(m)*m->pgsz)return ENOMEM;
  size_t span=(bytes+quantum-1)/quantum*m->block;
  if(span>mesh_rows(m)/count)return ENOMEM;
  uint32_t pages=count*(uint32_t)span,first=mesh_rows_alloc(context,pages);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t row=first;row<first+pages;row+=(uint32_t)span)
    mesh_buffers(m)[row]=(struct mesh_buffer){.ownership=2|MESH_BUFFER_FLAG(MESH_BUFFER_SEALED),.first=row,.pages=(uint32_t)span,.owner=context->client};
  mesh_bits_set(m,MESH_ROW_HOT,first,pages);
  if(!receive)for(uint32_t row=first;row<first+pages;row+=(uint32_t)span){
    uint32_t page=mesh_arena_alloc(context,(uint32_t)span,m->block);
    if(page==MESH_ABSENT){
      int error=errno;mesh_backing_release(context,first,pages);mesh_rows_release(context,first,pages);return error;
    }
    mesh_backing_bind(context,row,(uint32_t)span,page);
  }
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,(uint32_t)span,(uint32_t)receive};
  return 0;
}
/* design/algorithm-sources.md#programtensor */
uint32_t mesh_section_page(struct mesh_ctx *context,struct mesh_section section,uint32_t index){
  return atomic_load_explicit(&mesh_page(context->M)[mesh_section_row(section,index)],memory_order_acquire);
}
/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return mesh_at(context->M,mesh_section_page(context,section,index));}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){mesh_publish(context->M,section.first);}
/* design/algorithm-sources.md#programtensor */
void mesh_section_release(struct mesh_ctx *context,struct mesh_section section){
  mesh_buffer_release(context->M,section.first,section.count*section.pages);
  mesh_rows_release(context,section.first,section.count*section.pages);
}
/* design/algorithm-sources.md#collectivesync_on_remote_fill */
void mesh_sync_on_remote_fill(struct mesh_ctx *context,const struct mesh_section *sections,size_t count,uint32_t index){
  for(size_t i=0;i<count;i++)while(!mesh_bit(context->M,MESH_PRESENT,mesh_section_row(sections[i],index))){}
}
