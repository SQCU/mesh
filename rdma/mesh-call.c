#include "mesh-call.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct mesh_call {
  struct mesh_function *function;
  struct mesh_operand *operands;
  uint32_t pending,initial,index;
  int completed;
};
struct mesh_function {
  struct mesh_calls *calls;
  struct mesh_section *inputs,*outputs,*consumed;
  struct mesh_call *values;
  struct mesh_operand *operands;
  uint32_t *remote;
  size_t input_count,output_count,remote_count,consumed_count;
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
  uint32_t index,references;
};
struct mesh_calls {
  struct mesh_ctx *context;
  struct mesh_function *functions;
  struct mesh_call *values;
  uint32_t *rows,*declared,roots[MESH_COMPUTE_THREADS];
  size_t row_count,function_count;
  struct mesh_call_worker workers[MESH_COMPUTE_THREADS];
  uint32_t count,extent,first,root_workers;
  _Atomic uint32_t references;
  _Atomic int running;
  void *owner;
  mesh_dispose dispose;
};

/* design/algorithm-sources.md#program */
static inline uint32_t mesh_slot(const struct mesh_calls *calls,uint32_t index){return index%calls->extent;}

/* design/algorithm-sources.md#programtensor */
static uint32_t mesh_section_row(struct mesh_section section,uint32_t index){return section.first+index*section.stride;}

/* design/algorithm-sources.md#programtensor */
static void mesh_calls_release(struct mesh_calls *calls,uint32_t references){
  if(atomic_fetch_sub_explicit(&calls->references,references,memory_order_acq_rel)!=references)return;
  struct hdr *m=calls->context->M;
  for(uint32_t i=0;i<calls->count;i++){free(calls->workers[i].offsets);free(calls->workers[i].targets);}
  for(struct mesh_function *function=calls->functions;function;){
    struct mesh_function *next=function->next;
    for(size_t i=0;i<function->input_count;i++)for(uint32_t index=0;index<(function->inputs[i].stride?calls->extent:1);index++){
      struct mesh_section section=function->inputs[i];
      uint32_t row=mesh_section_row(section,index);
      mesh_buffers(m)[row].uses&=~(UINT32_C(1)<<function->worker);
      if(!section.stride || !calls->values || !function->values[index*calls->function_count].completed)mesh_buffer_release(m,row,section.pages);
    }
    if(function->dispose)function->dispose(function->argument);
    free(function->operands);free(function->inputs);free(function);
    function=next;
  }
  if(calls->first!=MESH_ABSENT)mesh_rows_release(calls->context,calls->first,calls->extent);
  if(calls->dispose)calls->dispose(calls->owner);
  free(calls->values);free(calls->rows);free(calls->declared);
  free(calls);
}

/* design/algorithm-sources.md#programkernel_call */
struct mesh_calls *mesh_calls_create(struct mesh_ctx *context,uint32_t workers,uint32_t inFlight,void *owner,mesh_dispose dispose){
  if(!context || !context->M || !workers || workers>MESH_COMPUTE_THREADS || !inFlight){errno=EINVAL;return NULL;}
  struct mesh_calls *calls=calloc(1,sizeof *calls);
  if(!calls)return NULL;
  calls->context=context;calls->count=workers;calls->extent=inFlight;calls->first=MESH_ABSENT;atomic_init(&calls->references,1);
  for(uint32_t i=0;i<workers;i++){
    calls->workers[i]=(struct mesh_call_worker){.calls=calls,.index=i,.offsets=calloc((size_t)mesh_rows(context->M)+1,sizeof(size_t))};
    if(!calls->workers[i].offsets){mesh_calls_release(calls,1);return NULL;}
  }
  calls->first=mesh_rows_alloc(context,inFlight);
  if(calls->first==MESH_ABSENT){mesh_calls_release(calls,1);return NULL;}
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
  function->inputs=calloc(1,(count+input_count?count+input_count:1)*sizeof *inputs+input_count*sizeof *function->remote);
  function->operands=calloc(extent*(count?count:1),sizeof *function->operands);
  if(!function->inputs || !function->operands){errno=ENOMEM;goto failed;}
  function->outputs=function->inputs+input_count;
  function->consumed=function->outputs+output_count;
  function->remote=(uint32_t *)(function->consumed+input_count);
  for(size_t i=0;i<input_count;i++){
    if(inputs[i].receive)function->remote[function->remote_count++]=(uint32_t)i;
    if(inputs[i].stride)function->consumed[function->consumed_count++]=inputs[i];
  }
  memcpy(function->inputs,inputs,input_count*sizeof *inputs);
  memcpy(function->outputs,outputs,output_count*sizeof *outputs);
  function->calls=calls;function->worker=worker;function->input_count=input_count;function->output_count=output_count;
  function->submit=submit;function->argument=argument;function->dispose=dispose;
  struct hdr *m=calls->context->M;
  size_t retained=0;
  for(size_t i=0;i<input_count;i++)for(uint32_t index=0;index<(inputs[i].stride?extent:1);index++){
    int error=mesh_buffer_retain(m,mesh_section_row(inputs[i],index),inputs[i].pages);
    if(error){
      for(size_t j=0;j<input_count;j++)for(uint32_t value=0;value<(inputs[j].stride?extent:1) && retained;value++,retained--)
        mesh_buffer_release(m,mesh_section_row(inputs[j],value),inputs[j].pages);
      errno=error;goto failed;
    }
    retained++;
  }
  calls->function_count++;
  calls->workers[worker].references+=calls->extent;
  function->next=calls->functions;calls->functions=function;
  return function;
failed:
  free(function->inputs);free(function->operands);free(function);
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
  uint32_t unused=1;
  for(struct mesh_function *function=calls->functions;function;function=function->next)
    if(function->worker==worker->index)for(uint32_t index=0;index<calls->extent;index++)unused+=function->values[index*calls->function_count].pending!=0;
  mesh_calls_release(calls,unused);
  return NULL;
}

/* design/algorithm-sources.md#programkernel_call */
int mesh_calls_start(struct mesh_calls *calls,const struct mesh_section *sections,size_t count){
  struct hdr *m=calls->context->M;
  uint32_t rows=mesh_rows(m);
  calls->values=calloc(calls->extent*(calls->function_count?calls->function_count:1),sizeof *calls->values);
  calls->rows=calloc(calls->extent*(count?count:1),sizeof *calls->rows);
  calls->declared=calloc(rows,sizeof *calls->declared);
  if(!calls->values || !calls->rows || !calls->declared){free(calls->values);calls->values=NULL;return ENOMEM;}
  calls->row_count=count;
  for(uint32_t index=0;index<calls->extent;index++){
    uint32_t slot=mesh_slot(calls,index),root=calls->first+slot;
    mesh_buffers(m)[root]=(struct mesh_buffer){.ownership=count,.first=root,.pages=1,.uses=1,.owner=calls->context->client};
    for(size_t i=0;i<count;i++){
      uint32_t row=mesh_section_row(sections[i],slot);
      struct mesh_buffer *buffer=&mesh_buffers(m)[row];
      calls->rows[slot*count+i]=row;buffer->slot=root+1;
      calls->declared[row]=(uint32_t)atomic_load_explicit(&buffer->ownership,memory_order_relaxed)-1;
      atomic_store_explicit(&buffer->ownership,calls->declared[row]|MESH_BUFFER_FLAG(MESH_BUFFER_SEALED),memory_order_relaxed);
    }
  }
  mesh_bits_set(m,MESH_ROW_HOT,calls->first,calls->extent);
  size_t ordinal=0;
  for(int pass=0;pass<2;pass++){
    for(struct mesh_function *function=calls->functions;function;function=function->next){
      if(!pass)function->values=calls->values+ordinal++;
      struct mesh_call_worker *worker=&calls->workers[function->worker];
      for(uint32_t index=0;index<calls->extent;index++){
        uint32_t slot=mesh_slot(calls,index);
        struct mesh_call *call=&function->values[slot*calls->function_count];
        if(!pass){
          size_t operands=function->input_count+function->output_count;
          *call=(struct mesh_call){.function=function,.index=slot,.operands=function->operands+slot*operands};
          for(size_t i=0;i<operands;i++){
            struct mesh_section section=function->inputs[i];
            uint32_t page=atomic_load_explicit(&mesh_page(m)[mesh_section_row(section,slot)],memory_order_acquire);
            call->operands[i]=(struct mesh_operand){mesh_at(m,page),section.bytes,page,slot};
          }
        }
        int varying=0;
        for(size_t i=0;i<=function->input_count;i++){
          uint32_t row;
          if(i==function->input_count){
            if(varying)continue;
            row=calls->first+slot;calls->root_workers|=UINT32_C(1)<<function->worker;
          } else {
            struct mesh_section section=function->inputs[i];
            row=mesh_section_row(section,slot);
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
  uint32_t roots=calls->root_workers;calls->root_workers=0;
  for(uint32_t i=0;i<calls->count;i++)if(roots&(UINT32_C(1)<<i))
    calls->roots[calls->root_workers++]=mesh_notice_queue(m,calls->context->client,m->links+i);
  for(size_t i=0;i<calls->extent*calls->function_count;i++)calls->values[i].initial=calls->values[i].pending;
  atomic_store_explicit(&calls->running,1,memory_order_release);
  for(uint32_t i=0;i<calls->count;i++){
    pthread_t thread;
    uint32_t references=calls->workers[i].references+1;
    atomic_fetch_add_explicit(&calls->references,references,memory_order_relaxed);
    int error=pthread_create(&thread,NULL,mesh_call_progress,&calls->workers[i]);
    if(error){atomic_store(&calls->running,0);mesh_calls_release(calls,references);return error;}
    pthread_detach(thread);
  }
  return 0;
}

/* design/algorithm-sources.md#program */
int mesh_calls_submit(struct mesh_calls *calls,uint32_t index){
  struct hdr *m=calls->context->M;
  uint32_t slot=mesh_slot(calls,index),root=calls->first+slot;
  struct mesh_buffer *event=&mesh_buffers(m)[root];
  uint32_t stamp=atomic_load_explicit(&event->uses,memory_order_acquire);
  if(!stamp)return 0;
  atomic_store_explicit(&event->uses,0,memory_order_relaxed);
  atomic_store_explicit(&event->ownership,calls->row_count,memory_order_relaxed);
  for(size_t i=0;i<calls->function_count;i++){
    struct mesh_call *call=&calls->values[slot*calls->function_count+i];
    call->pending=call->initial;call->completed=0;
  }
  atomic_fetch_add_explicit(&calls->references,(stamp>>1)*calls->function_count,memory_order_relaxed);
  for(size_t i=0;i<calls->row_count;i++){
    uint32_t row=calls->rows[slot*calls->row_count+i];
    atomic_store_explicit(&mesh_buffers(m)[row].ownership,calls->declared[row]|MESH_BUFFER_FLAG(MESH_BUFFER_SEALED),memory_order_relaxed);
    atomic_fetch_and_explicit(&mesh_plane(m,MESH_PRESENT)[row/64],~(UINT64_C(1)<<(row%64)),memory_order_relaxed);
  }
  for(uint32_t i=0;i<calls->root_workers;i++)mesh_notice_push(m,calls->roots[i],root);
  return 1;
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_call_fail(struct mesh_call *call,int error){
  struct mesh_function *function=call->function;
  struct mesh_calls *calls=function->calls;
  struct mesh_ctx *context=calls->context;
  uint32_t index=call->index;
  call->completed=1;
  context->M->port.code=error;context->M->port.domain=4;
  for(size_t i=0;i<function->consumed_count;i++){
    struct mesh_section section=function->consumed[i];
    mesh_buffer_release(context->M,mesh_section_row(section,index),section.pages);
  }
  mesh_calls_release(calls,1);
}

/* design/algorithm-sources.md#programkernel_call */
void mesh_call_complete(struct mesh_call *call){
  struct mesh_function *function=call->function;
  struct mesh_calls *calls=function->calls;
  struct mesh_ctx *context=calls->context;
  uint32_t index=call->index;
  call->completed=1;
  for(size_t i=0;i<function->output_count;i++){
    struct mesh_section section=function->outputs[i];
    uint32_t row=mesh_section_row(section,index);
    mesh_publish(context->M,row);
  }
  for(size_t i=0;i<function->consumed_count;i++){
    struct mesh_section section=function->consumed[i];
    mesh_buffer_release(context->M,mesh_section_row(section,index),section.pages);
  }
  mesh_calls_release(calls,1);
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
int mesh_transfers_prepare(struct mesh_ctx *context,const struct mesh_section *sections,size_t count){
  struct hdr *m=context->M;
  for(size_t i=0;i<count;i++)if(!sections[i].receive)for(uint32_t index=0;index<sections[i].count;index++){
    struct mesh_section section=sections[i];
    uint32_t page=mesh_arena_alloc(context,section.pages,m->block);
    if(page==MESH_ABSENT)return errno;
    mesh_backing_bind(context,mesh_section_row(section,index),section.pages,page);
  }
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
int mesh_section_create(struct mesh_ctx *context,size_t bytes,uint32_t count,int receive,int shared,struct mesh_section *section){
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
  if(!receive && shared)for(uint32_t row=first;row<first+pages;row+=(uint32_t)span){
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
  if(!section.stride || !mesh_buffers(context->M)[section.first].slot)
    mesh_buffer_release(context->M,section.first,section.count*section.pages);
  mesh_rows_release(context,section.first,section.count*section.pages);
}
/* design/algorithm-sources.md#collectivesync_on_remote_fill */
void mesh_sync_on_remote_fill(struct mesh_calls *calls,const struct mesh_section *sections,size_t count,uint32_t index){
  uint32_t slot=mesh_slot(calls,index);
  for(size_t i=0;i<count;i++)while(!mesh_bit(calls->context->M,MESH_PRESENT,mesh_section_row(sections[i],slot))){}
}
