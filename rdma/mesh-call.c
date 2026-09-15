#include "mesh-call.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct mesh_use { struct mesh_call *call; struct mesh_use *next; };
struct mesh_call {
  struct mesh_calls *calls;
  struct mesh_section *inputs,*outputs;
  struct mesh_operand *operands;
  struct mesh_use *uses;
  size_t input_count,output_count;
  uint32_t pending,worker;
  int completed;
  mesh_submit submit;
  mesh_dispose dispose;
  void *argument;
  struct mesh_call *next,*root_next;
};
struct mesh_call_worker {
  struct mesh_calls *calls;
  struct mesh_use **uses;
  struct mesh_call *roots;
  uint32_t index;
};
struct mesh_calls {
  struct mesh_ctx *context;
  struct mesh_call *functions;
  struct mesh_call_worker workers[MESH_COMPUTE_THREADS];
  uint32_t count;
  _Atomic uint32_t references;
  _Atomic int running;
  void *owner;
  mesh_dispose dispose;
};

/* design/algorithm-sources.md#programtensor */
static void mesh_calls_release(struct mesh_calls *calls){
  if(atomic_fetch_sub_explicit(&calls->references,1,memory_order_acq_rel)!=1)return;
  struct hdr *m=calls->context->M;
  for(uint32_t i=0;i<calls->count;i++)free(calls->workers[i].uses);
  for(struct mesh_call *call=calls->functions;call;){
    struct mesh_call *next=call->next;
    for(size_t i=0;i<call->input_count;i++){
      struct mesh_section section=call->inputs[i];
      mesh_buffers(m)[section.first].uses&=~(UINT32_C(1)<<call->worker);
      if(!call->completed)mesh_buffer_release(m,section.first,section.pages);
    }
    if(call->dispose)call->dispose(call->argument);
    free(call->uses);free(call->operands);free(call->inputs);free(call);
    call=next;
  }
  if(calls->dispose)calls->dispose(calls->owner);
  free(calls);
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *context,uint32_t workers,void *owner,mesh_dispose dispose){
  if(!context || !context->M || !workers || workers>MESH_COMPUTE_THREADS){errno=EINVAL;return NULL;}
  struct mesh_calls *calls=calloc(1,sizeof *calls);
  if(!calls)return NULL;
  calls->context=context;calls->count=workers;atomic_init(&calls->references,1);
  for(uint32_t i=0;i<workers;i++){
    calls->workers[i]=(struct mesh_call_worker){.calls=calls,.index=i,.uses=calloc(mesh_rows(context->M),sizeof(struct mesh_use *))};
    if(!calls->workers[i].uses){mesh_calls_release(calls);return NULL;}
  }
  calls->owner=owner;calls->dispose=dispose;
  return calls;
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_call *mesh_call_bind(struct mesh_calls *calls,uint32_t worker,
  const struct mesh_section *inputs,size_t input_count,
  const struct mesh_section *outputs,size_t output_count,mesh_submit submit,void *argument,mesh_dispose dispose){
  if(worker>=calls->count || !submit || atomic_load(&calls->running)){errno=EINVAL;return NULL;}
  struct mesh_call *call=calloc(1,sizeof *call);
  if(!call)return NULL;
  size_t count=input_count+output_count;
  call->inputs=calloc(count?count:1,sizeof *inputs);
  call->operands=calloc(count?count:1,sizeof *call->operands);
  call->uses=calloc(input_count?input_count:1,sizeof *call->uses);
  if(!call->inputs || !call->operands || !call->uses){free(call->inputs);free(call->operands);free(call->uses);free(call);return NULL;}
  call->outputs=call->inputs+input_count;
  memcpy(call->inputs,inputs,input_count*sizeof *inputs);
  memcpy(call->outputs,outputs,output_count*sizeof *outputs);
  call->calls=calls;call->worker=worker;call->input_count=input_count;call->output_count=output_count;
  call->submit=submit;call->argument=argument;call->dispose=dispose;
  struct hdr *m=calls->context->M;
  for(size_t i=0;i<output_count;i++){
    uint32_t page=atomic_load_explicit(&mesh_page(m)[outputs[i].first],memory_order_acquire);
    call->operands[input_count+i]=(struct mesh_operand){mesh_at(m,page),outputs[i].bytes,page};
  }
  for(size_t i=0;i<input_count;i++){
    int error=mesh_buffer_retain(m,inputs[i].first,inputs[i].pages);
    if(error){
      for(size_t j=0;j<i;j++)mesh_buffer_release(m,inputs[j].first,inputs[j].pages);
      free(call->inputs);free(call->operands);free(call->uses);free(call);errno=error;return NULL;
    }
  }
  for(size_t i=0;i<input_count;i++){
    uint32_t row=inputs[i].first;
    if(mesh_bit(m,MESH_PRESENT,row))continue;
    struct mesh_use *use=&call->uses[i];
    *use=(struct mesh_use){call,calls->workers[worker].uses[row]};
    calls->workers[worker].uses[row]=use;
    mesh_buffers(m)[row].uses|=UINT32_C(1)<<worker;
    call->pending++;
  }
  if(!call->pending){call->root_next=calls->workers[worker].roots;calls->workers[worker].roots=call;}
  call->next=calls->functions;calls->functions=call;
  return call;
}

/* design/algorithm-sources.md#programkernel_call */
static void mesh_call_submit(struct mesh_call *call){
  struct mesh_ctx *context=call->calls->context;
  for(size_t i=0;i<call->input_count;i++){
    struct mesh_section section=call->inputs[i];
    uint32_t page=atomic_load_explicit(&mesh_page(context->M)[section.first],memory_order_acquire);
    call->operands[i]=(struct mesh_operand){mesh_at(context->M,page),section.bytes,page};
  }
  atomic_fetch_add_explicit(&call->calls->references,1,memory_order_relaxed);
  call->submit(call,call->argument,call->operands,call->operands+call->input_count);
}

/* design/algorithm-sources.md#programkernel_call */
static void *mesh_call_progress(void *argument){
  struct mesh_call_worker *worker=argument;
  struct mesh_calls *calls=worker->calls;
  struct hdr *m=calls->context->M;
  uint32_t queue=mesh_notice_queue(calls->context->client,MESH_NOTICE_COMPUTE+worker->index);
  pthread_setname_np("mesh.numerical");
  for(struct mesh_call *call=worker->roots;call;call=call->root_next)mesh_call_submit(call);
  while(atomic_load_explicit(&calls->running,memory_order_acquire)){
    uint32_t row=mesh_notice_take(m,queue);
    while(row!=MESH_ABSENT){
      uint32_t next=mesh_notice_next(m,queue,row);
      for(struct mesh_use *use=worker->uses[row];use;use=use->next)
        if(!--use->call->pending)mesh_call_submit(use->call);
      row=next;
    }
  }
  mesh_calls_release(calls);
  return NULL;
}

/* design/algorithm-sources.md#programkernel_call */
int mesh_calls_start(struct mesh_calls *calls){
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

/* design/algorithm-sources.md#programkernel_call */
void mesh_call_fail(struct mesh_call *call,int error){
  struct mesh_calls *calls=call->calls;
  struct mesh_ctx *context=calls->context;
  context->M->port.code=error;context->M->port.domain=4;
  for(size_t i=0;i<call->input_count;i++)
    mesh_buffer_release(context->M,call->inputs[i].first,call->inputs[i].pages);
  call->completed=1;
  mesh_calls_release(calls);
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_call_complete(struct mesh_call *call){
  struct mesh_calls *calls=call->calls;
  struct mesh_ctx *context=calls->context;
  for(size_t i=0;i<call->output_count;i++){
    struct mesh_section section=call->outputs[i];
    mesh_publish_partial(context,section.first,section.pages);
    mesh_buffer_produced(context->M,section.first,section.pages);
  }
  for(size_t i=0;i<call->input_count;i++)
    mesh_buffer_release(context->M,call->inputs[i].first,call->inputs[i].pages);
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
  if(queue>=m->qps || section.pages!=m->block || section.bytes>(size_t)m->block*m->pgsz)return EINVAL;
  _Atomic uint32_t *length=mesh_order_length(m,queue,receive);
  uint32_t index=atomic_load_explicit(length,memory_order_relaxed);
  if(index==mesh_blocks(m))return ENOSPC;
  if(!receive){
    int error=mesh_buffer_retain(m,section.first,section.pages);if(error)return error;
    if(!mesh_bit(m,MESH_SEND_SOURCE,section.first) && mesh_bit(m,MESH_PRESENT,section.first))
      mesh_notice_push(m,mesh_notice_queue(context->client,MESH_NOTICE_SEND),section.first);
    mesh_bits_set(m,MESH_SEND_SOURCE,section.first,1);
  }
  mesh_transfers(m,queue,receive)[index]=(struct mesh_transfer){section.first,identity,0,(uint32_t)section.bytes};
  atomic_store_explicit(length,index+1,memory_order_release);
  return 0;
}

/* design/algorithm-sources.md#programcopy */
void mesh_transfers_start(struct mesh_ctx *context){atomic_store_explicit(&context->M->configured,context->client,memory_order_release);}

/* design/algorithm-sources.md#programtensor */
int mesh_section_create(struct mesh_ctx *context,size_t bytes,struct mesh_section *section){
  struct hdr *m=context->M;
  if(!bytes || bytes>(size_t)m->block*m->pgsz)return EINVAL;
  uint32_t first=mesh_rows_alloc(context,m->block);
  if(first==MESH_ABSENT)return errno;
  int error=mesh_backing_alloc(context,first,m->block,m->block,1);
  if(error){mesh_rows_release(context,first,m->block);return error;}
  *section=(struct mesh_section){first,m->block,bytes};
  mesh_buffer_retain(m,first,m->block);
  mesh_buffer_seal(m,first,m->block);
  return 0;
}
/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section){return mesh_row_data(context,section.first);}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){mesh_constant(context,section.first,section.pages);}
/* design/algorithm-sources.md#programtensor */
void mesh_section_release(struct mesh_ctx *context,struct mesh_section section){
  mesh_buffer_release(context->M,section.first,section.pages);
  mesh_rows_release(context,section.first,section.pages);
}
/* design/algorithm-sources.md#programtensor */
size_t mesh_section_capacity(struct mesh_ctx *context){return (size_t)context->M->block*context->M->pgsz;}

/* design/algorithm-sources.md#programtensor */
size_t mesh_receive_pages(struct mesh_ctx *context,uint32_t queue,uint32_t *pages){
  struct hdr *m=context->M;
  uint32_t count=atomic_load_explicit(mesh_order_length(m,queue,MESH_RECEIVE),memory_order_acquire);
  if(pages)for(uint32_t i=0;i<count;i++)
    pages[i]=atomic_load_explicit(&mesh_page(m)[mesh_transfers(m,queue,MESH_RECEIVE)[i].local_row],memory_order_acquire);
  return count;
}

/* design/algorithm-sources.md#collectivesync_on_remote_fill */
void mesh_sync_on_remote_fill(struct mesh_ctx *context,const struct mesh_section *sections,size_t count){
  for(size_t i=0;i<count;i++)while(!mesh_bits_all(context->M,MESH_PRESENT,sections[i].first,sections[i].pages)){}
}
