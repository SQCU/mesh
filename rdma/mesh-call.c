#include "mesh-call.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct mesh_call {
  struct mesh_function *function;
  struct mesh_operand *operands;
  uint32_t index,remaining;
  uint64_t invocation;
  int error;
};
struct mesh_join { uint64_t invocation; uint32_t pending; };
struct mesh_function {
  struct mesh_calls *calls;
  struct mesh_section *inputs;
  struct mesh_call *values;
  struct mesh_join *joins;
  struct mesh_operand *operands;
  uint32_t *consumed,*available,*references,*matches,*join_free,*join_rows;
  unsigned char *placed;
  size_t input_count,output_count,consumed_count;
  uint32_t worker,identity,pending,mask;
  uint64_t free_head,free_tail,join_head,join_tail;
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
  _Atomic uint32_t active;
};
struct mesh_record { _Atomic uint64_t key; _Atomic uint32_t frame; uint32_t remaining; uint64_t error; };
struct mesh_calls {
  struct mesh_ctx *context;
  struct mesh_function *functions,**program;
  struct mesh_call_worker workers[MESH_COMPUTE_THREADS];
  uint32_t count,extent,first,return_first,root_workers,function_count;
  struct mesh_call **slots;
  struct mesh_instance *instances;
  struct mesh_record *records;
  uint32_t *available,mask,free_mask,shared,dynamic;
  uint64_t *frame_keys;
  uint64_t free_head,*event_heads;
  _Atomic uint64_t free_tail;
  uint32_t event_first,event_count;
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
    free(function->joins);free(function->matches);free(function->available);free(function->operands);free(function->values);free(function->inputs);free(function);
    function=next;
  }
  if(calls->first!=MESH_ABSENT)mesh_rows_release(calls->context,calls->first,calls->extent);
  if(calls->return_first!=MESH_ABSENT)mesh_rows_release(calls->context,calls->return_first,calls->function_count*calls->extent);
  if(calls->event_count)mesh_rows_release(calls->context,calls->event_first,calls->event_count);
  if(calls->dispose)calls->dispose(calls->owner);
  free(calls->records);free(calls->available);free(calls->frame_keys);free(calls->event_heads);
  free(calls->slots);free(calls->program);free(calls);
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
  for(uint32_t i=0;i<count;i++){
    atomic_store_explicit(&calls->instances[i].status,MESH_RESULT(MESH_RESULT_BUSY,0,0),memory_order_relaxed);
    atomic_store_explicit(&calls->instances[i].invocation,UINT64_MAX,memory_order_relaxed);
  }
  atomic_store_explicit(&context->M->result[context->client>>63],MESH_RESULT(MESH_RESULT_BUSY,0,0),memory_order_release);
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
  size_t varying=0;for(size_t i=0;i<input_count;i++)varying+=inputs[i].stride!=0;
  uint32_t capacity=1;while(capacity<2*extent*(varying?varying:1))capacity*=2;function->mask=capacity-1;
  function->available=calloc(2*(size_t)capacity+output_count,sizeof *function->available);
  function->joins=calloc(capacity,sizeof *function->joins+input_count*sizeof *function->join_rows);
  function->matches=malloc(capacity*sizeof *function->matches);
  if(!function->inputs || !function->operands || !function->values || !function->available || !function->joins || !function->matches){errno=ENOMEM;goto failed;}
  for(uint32_t i=0;i<capacity;i++)function->matches[i]=MESH_ABSENT;
  function->join_free=function->available+capacity;function->join_tail=capacity;
  function->join_rows=(uint32_t *)(function->joins+capacity);
  function->references=function->join_free+capacity;function->free_tail=extent;
  for(uint32_t i=0;i<extent;i++)function->available[i]=i;
  for(uint32_t i=0;i<capacity;i++)function->join_free[i]=i;
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
    mesh_buffer_retain(m,mesh_section_row(inputs[i],index));
  for(uint32_t index=0;index<extent;index++){
    struct mesh_call *call=&function->values[index];
    *call=(struct mesh_call){.function=function,.operands=function->operands+index*count,.index=index};
    for(size_t i=0;i<count;i++){
      struct mesh_section section=i<input_count?views[i]:outputs[i-input_count];
      uint32_t page=atomic_load_explicit(mesh_buffer_pages(m,mesh_section_row(section,index)),memory_order_acquire);
      function->operands[index*count+i]=(struct mesh_operand){.data=page==MESH_ABSENT?NULL:mesh_at(m,page),.bytes=section.bytes,
        .page=page,.index=section.stride?index:0,.row=mesh_section_row(section,index),
        .pages=mesh_buffer_pages(m,mesh_section_row(section,index)),.page_size=m->pgsz,.block_pages=m->block};
    }
  }
  function->identity=calls->function_count++;
  function->next=calls->functions;calls->functions=function;
  return function;
failed:
  free(function->joins);free(function->matches);free(function->available);free(function->inputs);free(function->operands);free(function->values);free(function);
  return NULL;
}

/* design/algorithm-sources.md#programkernel_call */
static void mesh_call_input(struct mesh_function *function,struct mesh_operand *operand,uint32_t input,uint32_t row){
  operand->row=row;
  if(!function->placed[input]){
    struct hdr *m=function->calls->context->M;
    struct mesh_section section=function->inputs[input];
    operand->page=atomic_load_explicit(mesh_buffer_pages(m,row),memory_order_acquire);
    operand->pages=mesh_buffer_pages(m,row);
    operand->data=mesh_at(m,operand->page);operand->index=section.stride?(row-section.first)/section.stride:0;
  }
}

/* design/algorithm-sources.md#programkernel_call */
static uint32_t mesh_call_match(struct mesh_function *function,uint64_t invocation){
  uint32_t at=mesh_hash(invocation)&function->mask;
  while(function->matches[at]!=MESH_ABSENT && function->joins[function->matches[at]].invocation!=invocation)at=(at+1)&function->mask;
  return at;
}

/* design/algorithm-sources.md#programkernel_call */
static uint32_t mesh_call_join(struct mesh_function *function,uint64_t invocation){
  uint32_t at=mesh_call_match(function,invocation),index=function->matches[at];
  if(index==MESH_ABSENT){
    index=function->join_free[function->join_head++&function->mask];function->matches[at]=index;
    function->joins[index]=(struct mesh_join){invocation,function->pending};
  }
  return index;
}

/* design/algorithm-sources.md#programkernel_call */
static void mesh_call_ready(struct mesh_function *function,uint32_t index){
  struct mesh_join *join=&function->joins[index];
  if(--join->pending)return;
  uint32_t slot=function->available[function->free_head++&function->mask];
  struct mesh_call *call=&function->values[slot];
  call->invocation=join->invocation;call->error=0;call->remaining=(uint32_t)(function->output_count?function->output_count:1);
  struct hdr *m=function->calls->context->M;
  for(size_t i=0;i<function->input_count+function->output_count;i++)call->operands[i].invocation=call->invocation;
  for(size_t i=0;i<function->consumed_count;i++){
    uint32_t input=function->consumed[i];
    mesh_call_input(function,&call->operands[input],input,function->join_rows[index*function->input_count+input]);
  }
  for(size_t i=0;i<function->output_count;i++){
    uint32_t row=call->operands[function->input_count+i].row;
    atomic_fetch_add_explicit(&mesh_buffers(m)[row].ownership,function->references[i],memory_order_relaxed);
    atomic_store_explicit(&mesh_presence(m)[row],0,memory_order_relaxed);
    mesh_buffers(m)[row].invocation=call->invocation;
  }
  uint32_t hole=mesh_call_match(function,join->invocation),at=(hole+1)&function->mask;
  while(function->matches[at]!=MESH_ABSENT){
    uint32_t from=mesh_hash(function->joins[function->matches[at]].invocation)&function->mask;
    if(((at-from)&function->mask)>=((at-hole)&function->mask)){function->matches[hole]=function->matches[at];hole=at;}
    at=(at+1)&function->mask;
  }
  function->matches[hole]=MESH_ABSENT;
  function->join_free[function->join_tail++&function->mask]=index;
  atomic_fetch_add_explicit(&function->calls->workers[function->worker].active,1,memory_order_relaxed);
  function->submit(call,call->index,function->argument,call->operands,call->operands+function->input_count);
}

/* design/algorithm-sources.md#programkernel_call */
static void *mesh_call_progress(void *argument){
  struct mesh_call_worker *worker=argument;
  struct mesh_calls *calls=worker->calls;
  struct hdr *m=calls->context->M;
  uint32_t queue=mesh_notice_queue(m,calls->context->client,m->links*(m->qps+1)+worker->index);
  struct mesh_notice_reader reader=mesh_notice_reader_init(m,queue),returns=mesh_notice_reader_init(m,queue+MESH_COMPUTE_THREADS);
  pthread_setname_np("mesh.numerical");
  while(atomic_load_explicit(&calls->running,memory_order_acquire) || atomic_load_explicit(&worker->active,memory_order_acquire)){
    uint32_t row;
    while((row=mesh_notice_take(&returns))!=MESH_ABSENT){
      struct mesh_call *call=calls->slots[mesh_buffers(m)[row].binding];
      if(!call->error && --call->remaining)continue;
      struct mesh_function *function=call->function;
      for(size_t i=0;i<function->consumed_count;i++){
        uint32_t input=function->consumed[i];
        mesh_buffer_release(m,call->operands[input].row);
      }
      if(call->error)mesh_event_push(m,calls->context->client,1+worker->index,
        (struct mesh_event){MESH_RESULT(MESH_RESULT_FUNCTION,call->function->identity,call->error),call->invocation,MESH_EVENT_ERROR});
      else {
        if(function->rearm)function->rearm(call->index,function->argument);
        function->available[function->free_tail++&function->mask]=call->index;
        mesh_event_push(m,calls->context->client,1+worker->index,(struct mesh_event){0,call->invocation,MESH_EVENT_RETURN});
      }
    }
    if(!atomic_load_explicit(&calls->running,memory_order_acquire))continue;
    row=mesh_notice_take(&reader);
    if(row==MESH_ABSENT)continue;
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    for(size_t i=worker->offsets[buffer->definition];i<worker->offsets[buffer->definition+1];i++){
      struct mesh_use use=worker->targets[i];
      struct mesh_function *function=calls->program[use.function];
      if(use.input!=MESH_ABSENT && !function->inputs[use.input].stride){
        function->pending--;
        for(uint32_t index=0;index<calls->extent;index++){
          struct mesh_call *call=&function->values[index];
          mesh_call_input(function,&call->operands[use.input],use.input,row);
        }
        for(uint32_t index=0;index<=function->mask;index++)if(function->joins[index].pending)mesh_call_ready(function,index);
      } else {
        uint32_t index=mesh_call_join(function,buffer->invocation);
        if(use.input!=MESH_ABSENT)function->join_rows[index*function->input_count+use.input]=row;
        mesh_call_ready(function,index);
      }
    }
  }
  mesh_calls_release(calls,1);
  return NULL;
}

/* design/algorithm-sources.md#meshresult */
static struct mesh_record *mesh_record(struct mesh_calls *calls,uint64_t invocation){
  uint64_t key=(uint64_t)invocation+1;
  uint32_t at=mesh_hash(invocation)&calls->mask,available=MESH_ABSENT;
  for(uint32_t n=0;n<=calls->mask;n++,at=(at+1)&calls->mask){
    uint64_t current=atomic_load_explicit(&calls->records[at].key,memory_order_relaxed);
    if(current==key)return &calls->records[at];
    if(current==UINT64_MAX && available==MESH_ABSENT)available=at;
    if(!current){if(available==MESH_ABSENT)available=at;break;}
  }
  struct mesh_record *record=&calls->records[available];
  record->remaining=calls->dynamic+calls->shared;record->error=0;
  atomic_store_explicit(&record->frame,MESH_ABSENT,memory_order_relaxed);
  atomic_store_explicit(&record->key,key,memory_order_release);
  return record;
}

/* design/algorithm-sources.md#meshresult */
static void mesh_instance_frame(struct mesh_calls *calls,struct mesh_record *record,uint32_t frame,uint64_t invocation){
  uint64_t previous=calls->frame_keys[frame];
  if(previous!=UINT64_MAX)atomic_store_explicit(&mesh_record(calls,previous)->key,UINT64_MAX,memory_order_release);
  calls->frame_keys[frame]=invocation;
  atomic_store_explicit(&record->frame,frame,memory_order_release);
  if(record->error)mesh_result_conclude(&calls->instances[frame].status,record->error);
}

/* design/algorithm-sources.md#meshresult */
static void mesh_instance_return(struct mesh_calls *calls,struct mesh_record *record){
  uint32_t frame=atomic_load_explicit(&record->frame,memory_order_relaxed);
  if(frame==MESH_ABSENT)return;
  mesh_result_conclude(&calls->instances[frame].status,record->error);
  uint64_t tail=atomic_load_explicit(&calls->free_tail,memory_order_relaxed);
  calls->available[tail&calls->free_mask]=frame;
  atomic_store_explicit(&calls->free_tail,tail+1,memory_order_release);
}

/* design/algorithm-sources.md#meshresult */
static void *mesh_instance_progress(void *argument){
  struct mesh_calls *calls=argument;struct hdr *m=calls->context->M;
  uint64_t owner=calls->context->client;
  pthread_setname_np("mesh.lifecycle");
  while(atomic_load_explicit(&calls->running,memory_order_acquire)){
    for(uint32_t producer=0;producer<mesh_event_queues(m);producer++){
      struct mesh_event_queue *queue=mesh_event_queue(m,owner,producer);
      uint64_t head=calls->event_heads[producer];
      if(head==atomic_load_explicit(&queue->tail,memory_order_acquire))continue;
      struct mesh_event event=mesh_events(m,owner)[queue->first+(head&queue->mask)];
      calls->event_heads[producer]=head+1;
      if(event.kind==MESH_EVENT_SHARED){
        calls->shared--;
        for(uint32_t i=0;i<=calls->mask;i++){
          struct mesh_record *record=&calls->records[i];
          uint64_t key=atomic_load_explicit(&record->key,memory_order_relaxed);
          if(key && key!=UINT64_MAX && record->remaining && !--record->remaining)mesh_instance_return(calls,record);
        }
        if(!calls->shared && !calls->dynamic)mesh_result_conclude(&m->result[owner>>63],0);
        continue;
      }
      struct mesh_record *record=mesh_record(calls,event.invocation);
      if(!calls->root_workers && atomic_load_explicit(&record->frame,memory_order_relaxed)==MESH_ABSENT){
        uint32_t frame=calls->available[calls->free_head++&calls->free_mask];
        atomic_store_explicit(&calls->instances[frame].invocation,event.invocation,memory_order_relaxed);
        atomic_store_explicit(&calls->instances[frame].status,MESH_RESULT(MESH_RESULT_BUSY,0,0),memory_order_release);
        uint64_t failure=atomic_load_explicit(&m->result[owner>>63],memory_order_acquire);
        if(failure>>62!=MESH_RESULT_BUSY)mesh_result_conclude(&calls->instances[frame].status,failure);
        mesh_instance_frame(calls,record,frame,event.invocation);
      }
      if(event.kind==MESH_EVENT_SUBMIT){
        mesh_instance_frame(calls,record,(uint32_t)event.status,event.invocation);
        if(!record->remaining)mesh_instance_return(calls,record);
      } else if(event.kind==MESH_EVENT_ERROR){
        record->error=event.status;
        uint32_t frame=atomic_load_explicit(&record->frame,memory_order_relaxed);
        if(frame!=MESH_ABSENT)mesh_result_conclude(&calls->instances[frame].status,event.status);
      } else if(!--record->remaining)mesh_instance_return(calls,record);
    }
  }
  mesh_calls_release(calls,1);return NULL;
}

/* design/algorithm-sources.md#meshresult */
static int mesh_instances_prepare(struct mesh_calls *calls){
  struct hdr *m=calls->context->M;
  uint32_t queues=mesh_event_queues(m),capacity=1,total=0;
  while(capacity<4*calls->extent)capacity*=2;calls->mask=capacity-1;
  calls->records=calloc(capacity,sizeof *calls->records);
  capacity=1;while(capacity<calls->extent)capacity*=2;calls->free_mask=capacity-1;
  calls->available=calloc(capacity,sizeof *calls->available);
  calls->frame_keys=malloc(calls->extent*sizeof *calls->frame_keys);
  calls->event_heads=calloc(queues,sizeof *calls->event_heads);
  if(!calls->records || !calls->available || !calls->frame_keys || !calls->event_heads)return ENOMEM;
  for(uint32_t i=0;i<calls->extent;i++){calls->available[i]=i;calls->frame_keys[i]=UINT64_MAX;}
  atomic_store_explicit(&calls->free_tail,calls->extent,memory_order_relaxed);
  uint32_t *counts=calloc(queues,sizeof *counts);
  if(!counts)return ENOMEM;
  counts[0]=calls->extent;calls->dynamic=calls->function_count;
  for(uint32_t i=0;i<calls->function_count;i++)counts[1+calls->program[i]->worker]+=calls->extent;
  for(uint32_t q=0;q<m->links*m->qps;q++)for(int direction=0;direction<2;direction++){
    uint32_t count=atomic_load(mesh_order_length(m,calls->context->client,q,direction));
    struct mesh_transfer *transfers=mesh_transfers(m,calls->context->client,q,direction);
    for(uint32_t i=0;i<count;i++){
      counts[1+MESH_COMPUTE_THREADS+2*(q/m->qps)+direction]+=transfers[i].count;
      if(transfers[i].stride)calls->dynamic++;else calls->shared++;
    }
  }
  for(uint32_t q=0;q<queues;q++){
    capacity=1;while(capacity<counts[q])capacity*=2;counts[q]=capacity;total+=capacity;
  }
  uint32_t first=mesh_rows_alloc(calls->context,total);
  if(first==MESH_ABSENT){free(counts);return errno;}
  calls->event_first=first;calls->event_count=total;
  for(uint32_t q=0;q<queues;q++){
    struct mesh_event_queue *queue=mesh_event_queue(m,calls->context->client,q);
    queue->first=first;queue->mask=counts[q]-1;atomic_store_explicit(&queue->tail,0,memory_order_relaxed);first+=counts[q];
  }
  free(counts);
  if(!calls->shared && !calls->dynamic)atomic_store_explicit(&m->result[calls->context->client>>63],0,memory_order_release);
  return 0;
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
      mesh_buffers(m)[calls->return_first+slot].binding=slot;calls->slots[slot]=&function->values[index];
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
        if(pass)worker->targets[worker->offsets[section.first]++]=(struct mesh_use){function->identity,input};
        else {
          worker->offsets[section.first+1]++;
          for(uint32_t index=0;index<section.count;index++)mesh_buffers(m)[mesh_section_row(section,index)].uses|=UINT32_C(1)<<function->worker;
        }
      }
      if(!pass)function->pending=pending;
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
  fprintf(stderr,"mesh arena: in_flight=%u bytes_per_instance=%llu shared_bytes=%llu allocated_bytes=%llu\n",calls->extent,
    (unsigned long long)(calls->context->arena-calls->context->shared_pages)*m->pgsz/calls->extent,
    (unsigned long long)calls->context->shared_pages*m->pgsz,(unsigned long long)calls->context->arena*m->pgsz);
  int preparation=mesh_instances_prepare(calls);
  if(preparation)return preparation;
  atomic_store_explicit(&calls->running,1,memory_order_release);
  pthread_t lifecycle;
  atomic_fetch_add_explicit(&calls->references,1,memory_order_relaxed);
  int error=pthread_create(&lifecycle,NULL,mesh_instance_progress,calls);
  if(error){atomic_store(&calls->running,0);mesh_calls_release(calls,1);return error;}
  pthread_detach(lifecycle);
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
uint64_t mesh_calls_submit(struct mesh_calls *calls,uint64_t index){
  if(!calls->root_workers)return 0;
  uint64_t status=atomic_load_explicit(&calls->context->M->result[calls->context->client>>63],memory_order_acquire);
  if(status>>62!=MESH_RESULT_BUSY)return status;
  if(calls->free_head==atomic_load_explicit(&calls->free_tail,memory_order_acquire))return MESH_RESULT(MESH_RESULT_BUSY,0,0);
  uint32_t frame=calls->available[calls->free_head++&calls->free_mask];
  atomic_store_explicit(&calls->instances[frame].invocation,index,memory_order_relaxed);
  atomic_store_explicit(&calls->instances[frame].status,MESH_RESULT(MESH_RESULT_BUSY,0,0),memory_order_release);
  status=atomic_load_explicit(&calls->context->M->result[calls->context->client>>63],memory_order_acquire);
  if(status>>62!=MESH_RESULT_BUSY){mesh_result_conclude(&calls->instances[frame].status,status);return status;}
  mesh_event_push(calls->context->M,calls->context->client,0,(struct mesh_event){frame,index,MESH_EVENT_SUBMIT});
  mesh_buffers(calls->context->M)[calls->first+frame].invocation=index;
  uint32_t workers=calls->root_workers;
  while(workers){
    uint32_t worker=(uint32_t)__builtin_ctz(workers);workers&=workers-1;
    mesh_notice_push(calls->context->M,mesh_notice_queue(calls->context->M,calls->context->client,calls->context->M->links*(calls->context->M->qps+1)+worker),calls->first+frame);
  }
  return status>>62==MESH_RESULT_BUSY?0:status;
}

/* design/algorithm-sources.md#meshresult */
uint64_t mesh_calls_result(struct mesh_calls *calls,uint64_t index){
  uint32_t at=mesh_hash(index)&calls->mask;
  for(uint32_t n=0;n<=calls->mask;n++,at=(at+1)&calls->mask){
    uint64_t key=atomic_load_explicit(&calls->records[at].key,memory_order_acquire);
    if(!key)break;
    if(key==(uint64_t)index+1){
      uint32_t frame=atomic_load_explicit(&calls->records[at].frame,memory_order_acquire);
      if(frame==MESH_ABSENT)break;
      uint64_t status=atomic_load_explicit(&calls->instances[frame].status,memory_order_acquire);
      return atomic_load_explicit(&calls->instances[frame].invocation,memory_order_acquire)==index?status:MESH_RESULT(MESH_RESULT_BUSY,0,0);
    }
  }
  return atomic_load_explicit(&calls->context->M->result[calls->context->client>>63],memory_order_acquire);
}

/* design/algorithm-sources.md#programkernel_call */
static void mesh_call_finish(struct mesh_call *call,int error){
  struct mesh_function *function=call->function;
  struct mesh_calls *calls=function->calls;
  struct hdr *m=calls->context->M;
  _Atomic uint32_t *active=&calls->workers[function->worker].active;
  struct mesh_operand *outputs=call->operands+function->input_count;
  size_t output_count=function->output_count;
  if(error || !output_count){
    call->error=error;
    mesh_notice_push(m,mesh_notice_queue(m,calls->context->client,m->links*(m->qps+1)+MESH_COMPUTE_THREADS+function->worker),
      calls->return_first+function->identity*calls->extent+call->index);
  } else for(size_t i=0;i<output_count;i++)mesh_publish(m,outputs[i].row);
  atomic_fetch_sub_explicit(active,1,memory_order_release);
}

/* design/algorithm-sources.md#meshresult */
void mesh_call_fail(struct mesh_call *call,int error){mesh_call_finish(call,error?error:EIO);}
/* design/algorithm-sources.md#programkernel_call */
void mesh_call_complete(struct mesh_call *call){mesh_call_finish(call,0);}

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
    uint64_t bit=UINT64_C(1)<<(link%64),uses=atomic_fetch_or_explicit(&mesh_send_uses(m,row)[link/64],bit,memory_order_relaxed);
    if(!(uses&bit) && atomic_load_explicit(&mesh_presence(m)[row],memory_order_relaxed))
      mesh_notice_push(m,mesh_notice_queue(m,context->client,link),row);
  }
  mesh_transfers(m,context->client,queue,receive)[index]=(struct mesh_transfer){section.first,identity,section.count,section.stride,MESH_ABSENT,section.bytes};
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
  uint32_t stride=(uint32_t)span/m->block,rows=count*stride,first=mesh_rows_alloc(context,rows);
  if(first==MESH_ABSENT)return errno;
  for(uint32_t row=first;row<first+rows;row+=stride)
    mesh_buffers(m)[row]=(struct mesh_buffer){.ownership=2,.rows=stride,.pages=(uint32_t)span,.channel=channel,.definition=first,.owner=context->client,.mapping=m->page_off+(uint64_t)row*sizeof(uint32_t)};
  mesh_bits_set(m,MESH_ROW_HOT,first,rows);
  if(channel==MESH_ABSENT)for(uint32_t row=first;row<first+rows;row+=stride){
    uint32_t page=mesh_arena_alloc(context,(uint32_t)span,m->block);
    if(page==MESH_ABSENT){
      int error=errno;
      for(uint32_t r=first;r<first+rows;r+=stride){mesh_buffer_release(m,r);mesh_buffer_release(m,r);}
      mesh_rows_release(context,first,rows);return error;
    }
    mesh_backing_bind(context,row,(uint32_t)span,page);
  }
  *section=(struct mesh_section){first,(uint32_t)span,bytes,count,stride,channel};
  return 0;
}
/* design/algorithm-sources.md#programtensor */
uint32_t mesh_row_page(struct mesh_ctx *context,uint32_t row,uint32_t chunk){return atomic_load_explicit(mesh_buffer_pages(context->M,row)+chunk,memory_order_acquire);}
/* design/algorithm-sources.md#programtensor */
void *mesh_section_address(struct mesh_ctx *context,struct mesh_section section,uint32_t index){return mesh_at(context->M,mesh_row_page(context,mesh_section_row(section,index),0));}
/* design/algorithm-sources.md#programwrite */
void mesh_section_constant(struct mesh_ctx *context,struct mesh_section section){mesh_publish(context->M,section.first);}
/* design/algorithm-sources.md#programtensor */
void mesh_section_release(struct mesh_ctx *context,struct mesh_section section){
  for(uint32_t index=0;index<section.count;index++)mesh_buffer_release(context->M,mesh_section_row(section,index));
  mesh_rows_release(context,section.first,section.count*mesh_buffers(context->M)[section.first].rows);
}
/* design/algorithm-sources.md#collectivesync_on_remote_fill */
void mesh_sync_on_remote_fill(struct mesh_ctx *context,const struct mesh_section *sections,size_t count,uint64_t index){
  for(size_t i=0;i<count;i++){
    struct mesh_section section=sections[i];
    uint64_t stamp=section.stride?index+1:1;
    for(;;)for(uint32_t slot=0;slot<section.count;slot++)
      if(atomic_load_explicit(&mesh_presence(context->M)[mesh_section_row(section,slot)],memory_order_acquire)==stamp)goto filled;
filled:;
  }
}
