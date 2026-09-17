#include "mesh-verbs.h"
#include "mesh-dataflow.h"
#include <pthread.h>
#include <sys/event.h>

/* design/algorithm-sources.md#programcopy */
struct mesh_send_edge {
  _Alignas(128) struct ibv_sge span;
  struct ibv_qp *pair;
  int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **);
  uint32_t queue,tag_row,end,chunks;
  struct ibv_send_wr request;
  struct mesh_instance *instances;
  uint32_t row,completions,remaining;
};
_Static_assert(sizeof(struct mesh_send_edge)==256 && _Alignof(struct mesh_send_edge)==128 && offsetof(struct mesh_send_edge,request)+offsetof(struct ibv_send_wr,wr)<=128,"mesh_send_edge native request");
struct mesh_receive_request {_Alignas(64) struct ibv_sge span;struct ibv_recv_wr request;};
_Static_assert(sizeof(struct mesh_receive_request)==64 && _Alignof(struct mesh_receive_request)==64,"mesh_receive_request");
struct mesh_ready {uint64_t head,tail;size_t first,mask;};
/* design/algorithm-sources.md#programcopy */
struct mesh_send_range {_Alignas(16) uint32_t first;uint32_t end,invocation;};
_Static_assert(sizeof(struct mesh_send_range)==16 && _Alignof(struct mesh_send_range)==16,"mesh_send_range");
/* design/algorithm-sources.md#programcopy */
struct mesh_send_binding {_Alignas(16) struct ibv_sge *target;uint32_t key;};
_Static_assert(sizeof(struct mesh_send_binding)==16 && _Alignof(struct mesh_send_binding)==16,"mesh_send_binding");
struct mesh_receive_record {
  _Alignas(32) struct mesh_page_entry *entry;
  struct mesh_send_binding *sends;
  uint32_t row,send_count,send_notices,use_notices;
};
_Static_assert(sizeof(struct mesh_receive_record)==32 && _Alignof(struct mesh_receive_record)==32,"mesh_receive_record");
struct mesh_receive {
  _Alignas(128) struct mesh_receive_record *records;
  uint32_t region_base,page_base,region_blocks,block,page_shift,block_reciprocal,pool_offset;
  uintptr_t client_data;
  uint32_t *pages;
  struct mesh_receive_request *requests;
  struct mesh_send_binding *sends;
  uint64_t head;
  size_t mask;
  _Alignas(128) _Atomic uint64_t tail;
  struct mesh_event_reader returns;
};
_Static_assert(offsetof(struct mesh_receive,pages)==48 && offsetof(struct mesh_receive,tail)==128 && sizeof(struct mesh_receive)==256 && _Alignof(struct mesh_receive)==128,"mesh_receive completion geometry");
/* design/algorithm-sources.md#transport-retirement */
struct mesh_free_ring {
  _Alignas(128) size_t position;
  size_t mask;
  _Alignas(128) _Atomic uint32_t slots[];
};
_Static_assert(sizeof(struct mesh_free_ring)==128 && offsetof(struct mesh_free_ring,slots)==128,"mesh_free_ring");
struct mesh_link {
  pthread_t workers[3];
  uint32_t worker_count,index;
  pthread_t controller;
  int events;
  char *configuration;
  _Atomic int progressing,retiring;
  struct hdr *M;struct mesh_verbs provider;int qps;uint64_t client;
  struct mesh_receive receive[MESH_QPS];
  struct mesh_send_range *send_ready;
  struct mesh_send_edge *send_edges;
  struct mesh_free_ring *retirements[2];
  size_t send_count;
  struct mesh_link *links;
  struct mesh_ready ready[MESH_QPS];
  struct mesh_event_reader notices;
  struct mesh_instance *instances;
  uint32_t instance_count;
};
static inline __attribute__((always_inline)) int link_receive_post(struct mesh_queue *queue,struct mesh_receive *receive);
/* design/algorithm-sources.md#programcopy */
static void link_receive_destroy(struct mesh_receive *receive){
  free(receive->requests);free(receive->records);free(receive->sends);free(receive->pages);free(receive->returns.inputs);
}
/* design/algorithm-sources.md#meshresult */
static void link_stop(struct mesh_link *link){
  atomic_store_explicit(&link->progressing,0,memory_order_release);
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
/* design/algorithm-sources.md#programcopy */
static void link_error(struct mesh_link *link,int64_t code,uint32_t domain){
  struct mesh_port_info *port=&mesh_links(link->M)[link->index].port;port->code=code;port->domain=domain;
  mesh_result_conclude(&link->M->result[link->client>>63],MESH_RESULT(MESH_RESULT_LINK,link->index,code));
  atomic_store_explicit(&port->phase,MESH_STOPPED,memory_order_release);
  link_stop(link);
  for(uint32_t i=0;i<link->instance_count;i++)mesh_result_conclude(&link->instances[i].status,MESH_RESULT(MESH_RESULT_LINK,link->index,code));
}
/* design/algorithm-sources.md#programcopy */
static int link_prepare(struct mesh_link *link){
  struct hdr *m=link->M;uint64_t payload=(uint64_t)m->block*m->pgsz;
  size_t count=0,ready_count=0,queue_counts[MESH_QPS]={0};
  uint32_t *offsets=calloc(mesh_rows(m),sizeof *offsets);
  if(!offsets)return ENOMEM;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++)for(uint32_t i=0;i<atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_SEND));i++){
    struct mesh_transfer transfer=mesh_transfers(m,link->client,link->index*m->qps+q,MESH_SEND)[i];
    size_t chunks=(size_t)((transfer.bytes+payload-1)/payload);
    count+=transfer.count*chunks;queue_counts[q]+=transfer.count;
  }
  if(count>UINT32_MAX){free(offsets);return EOVERFLOW;}
  link->send_count=count;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    size_t capacity=1;
    while(capacity<queue_counts[q])capacity*=2;
    link->ready[q]=(struct mesh_ready){.first=ready_count,.mask=capacity-1};
    ready_count+=capacity;
  }
  free(link->send_edges);link->send_edges=aligned_alloc(_Alignof(struct mesh_send_edge),(count?count:1)*sizeof *link->send_edges);
  free(link->send_ready);link->send_ready=calloc(ready_count,sizeof *link->send_ready);
  if(!link->send_edges || !link->send_ready){free(offsets);return ENOMEM;}
  size_t releases[2]={count,0};
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    struct mesh_transfer *in=mesh_transfers(m,link->client,link->index*m->qps+q,MESH_RECEIVE);
    for(uint32_t i=0;i<atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_RECEIVE));i++)releases[MESH_RECEIVE]+=in[i].count;
  }
  for(uint32_t d=0;d<2;d++){
    size_t capacity=1;
    while(capacity<releases[d])capacity*=2;
    size_t bytes=(sizeof(struct mesh_free_ring)+capacity*sizeof(_Atomic uint32_t)+127)&~(size_t)127;
    free(link->retirements[d]);link->retirements[d]=aligned_alloc(_Alignof(struct mesh_free_ring),bytes);
    if(!link->retirements[d]){free(offsets);return ENOMEM;}
    memset(link->retirements[d],0,bytes);link->retirements[d]->mask=capacity-1;
  }
  uint64_t first=m->notice_off+(uint64_t)mesh_notice_queue(m,link->client,link->index)*m->notice_bytes;
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_relaxed)!=link->client)continue;
    struct mesh_publication *publication=mesh_publication_at(m,row);
    struct mesh_target *targets=publication->targets;
    for(uint32_t i=0;i<publication->sends;i++)if(targets[i].stream>=first && targets[i].stream<first+m->notice_bytes){
      offsets[row]=targets[i].index;
      uint32_t end=targets[i].index+targets[i].count;
      for(uint32_t at=targets[i].index;at<end;at++)link->send_edges[at].end=end;
    }
  }
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    uint32_t count=atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_SEND));
    struct mesh_transfer *out=mesh_transfers(m,link->client,link->index*m->qps+q,MESH_SEND);
    for(uint32_t i=0;i<count;i++)for(uint32_t value=0;value<out[i].count;value++){
      uint32_t row=out[i].local_row+value*out[i].stride,chunks=(uint32_t)((out[i].bytes+payload-1)/payload);
      uint32_t first=offsets[row];offsets[row]+=chunks;
      for(uint32_t chunk=0;chunk<chunks;chunk++){
        struct mesh_send_edge *source=&link->send_edges[first+chunk];uint32_t end=source->end;
        *source=(struct mesh_send_edge){.queue=q,.tag_row=row+chunk,.end=end,.instances=&link->instances[value],
          .row=row,.chunks=chunks,.remaining=chunks,.completions=out[i].stride?1:link->instance_count,
          .request={.wr_id=first,.sg_list=&source->span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED}};
      }
    }
  }
  free(offsets);return 0;
}
/* design/algorithm-sources.md#programcopy */
static int link_configure(void *state,int socket,uint64_t client){
  struct mesh_link *link=state;struct hdr *m=link->M;
  int notice_error=mesh_event_reader_init(&link->notices,m,mesh_notice_queue(m,client,link->index));
  if(notice_error){errno=notice_error;return -1;}
  uint64_t payload=(uint64_t)m->block*m->pgsz;
  uint32_t bytes[2*MESH_QPS]={0};
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    uint32_t counts[3]={atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_SEND)),atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,MESH_RECEIVE)),mesh_rows(m)},peer_counts[3];
    if(exchange(socket,counts,peer_counts,sizeof counts,sizeof peer_counts,m,client,link->provider.deadline))return -1;
    uint32_t sends=counts[MESH_SEND],receives=counts[MESH_RECEIVE];
    if(sends!=peer_counts[MESH_RECEIVE] || receives!=peer_counts[MESH_SEND]){
      fprintf(stderr,"transfer count mismatch queue=%u local=%u,%u peer=%u,%u\n",q,sends,receives,peer_counts[MESH_SEND],peer_counts[MESH_RECEIVE]);
      errno=EPROTO;return -1;
    }
    struct mesh_transfer *bindings=calloc(sends+receives+1,sizeof *bindings),*peer=calloc(receives?receives:1,sizeof *peer);
    if(!bindings || !peer){free(bindings);free(peer);return -1;}
    struct mesh_transfer *out=bindings,*in=bindings+sends;
    memcpy(out,mesh_transfers(m,link->client,link->index*m->qps+q,MESH_SEND),sends*sizeof *out);
    memcpy(in,mesh_transfers(m,link->client,link->index*m->qps+q,MESH_RECEIVE),receives*sizeof *in);
    for(int direction=0;direction<2;direction++){
      struct mesh_transfer *transfers=direction==MESH_SEND?out:in;
      uint64_t largest=0;
      for(uint32_t i=0;i<counts[direction];i++)if(transfers[i].bytes>largest)largest=transfers[i].bytes;
      bytes[2*q+direction]=(uint32_t)(largest<payload?largest:payload)+sizeof(struct mesh_wire_tag);
      if(counts[direction] && (bytes[2*q+direction]+4095)/4096>link->provider.capacity[q][direction]){free(bindings);free(peer);errno=EMSGSIZE;return -1;}
    }
    int error=exchange(socket,out,peer,sends*sizeof *out,receives*sizeof *peer,m,client,link->provider.deadline);
    uint32_t chunks_count=0;
    for(uint32_t i=0;i<receives && !error;i++){
      if(in[i].binding!=peer[i].binding || in[i].count!=peer[i].count || in[i].bytes!=peer[i].bytes){
        fprintf(stderr,"transfer mismatch queue=%u index=%u local=%u,%u,%llu peer=%u,%u,%llu\n",q,i,in[i].binding,in[i].count,(unsigned long long)in[i].bytes,peer[i].binding,peer[i].count,(unsigned long long)peer[i].bytes);
        errno=EPROTO;error=-1;break;
      }
      uint32_t chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      chunks_count+=in[i].count*chunks;
    }
    if(error){free(bindings);free(peer);return error;}
    size_t send_count=0;
    uint64_t send_first=m->notice_off+(uint64_t)mesh_notice_queue(m,client,0)*m->notice_bytes;
    for(uint32_t i=0;i<receives;i++)for(uint32_t value=0;value<in[i].count;value++){
      uint32_t row=in[i].local_row+value*in[i].stride;
      struct mesh_publication *publication=mesh_publication_at(m,row);
      struct mesh_target *targets=publication->targets;
      for(uint32_t j=0;j<publication->sends;j++){
        struct mesh_link *out=&link->links[(targets[j].stream-send_first)/m->notice_bytes];
        if(device_up(out->provider.device,out->provider.wire,m)){free(bindings);free(peer);return -1;}
        send_count+=targets[j].count;
      }
    }
    struct mesh_receive *receive=&link->receive[q];
    link_receive_destroy(receive);
    size_t capacity=1;
    while(capacity<chunks_count)capacity*=2;
    uint32_t stride=(m->block+1)*m->pgsz,region_blocks=(uint32_t)(link->provider.wire->region_extent/stride);
    uint32_t first=receives?in[0].pool:0,first_region=first/m->block/region_blocks;
    uint32_t regions=chunks_count?(first/m->block+chunks_count-1)/region_blocks-first_region+1:0;
    *receive=(struct mesh_receive){
      .region_base=(uint32_t)((uintptr_t)link->provider.wire->data>>32)+first_region,
      .page_base=first_region*region_blocks*m->block,.region_blocks=region_blocks,.pool_offset=first_region*region_blocks-first/m->block,
      .block_reciprocal=(uint32_t)(((UINT64_C(1)<<32)+m->block)/(m->block+1)),
      .page_shift=(uint32_t)__builtin_ctz(m->pgsz),.block=m->block,.client_data=m->client_data[client>>63],.tail=chunks_count,.mask=capacity-1};
    int reader_error=mesh_event_reader_init(&receive->returns,m,mesh_notice_queue(m,client,m->links+link->index*m->qps+q));
    if(reader_error){free(bindings);free(peer);errno=reader_error;return -1;}
    receive->pages=calloc(capacity,sizeof *receive->pages);
    receive->requests=aligned_alloc(_Alignof(struct mesh_receive_request),(chunks_count?chunks_count:1)*sizeof *receive->requests);
    receive->records=aligned_alloc(_Alignof(struct mesh_receive_record),(receives?peer_counts[2]:1)*sizeof *receive->records);
    receive->sends=malloc((send_count?send_count*regions:1)*sizeof *receive->sends);
    if(!receive->pages || !receive->requests || !receive->records || !receive->sends){free(bindings);free(peer);return -1;}
    for(uint32_t i=0;i<chunks_count;i++){
      uint32_t page=first+i*m->block;receive->pages[i]=i;
      struct mesh_receive_request *request=&receive->requests[i];
      *request=(struct mesh_receive_request){.span=link->provider.device->spans[page/m->block],.request={.sg_list=&request->span,.num_sge=1}};
      request->request.wr_id=request->span.addr;
      request->span.length=bytes[2*q+MESH_RECEIVE];
    }
    size_t next_send=0;
    for(uint32_t i=0;i<receives;i++){
      uint32_t chunks=(uint32_t)((in[i].bytes+payload-1)/payload);
      for(uint32_t value=0;value<in[i].count;value++){
        uint32_t row=in[i].local_row+value*in[i].stride;
        struct mesh_buffer *buffer=&mesh_buffers(m)[row];
        buffer->frame=value;buffer->completions=in[i].stride!=0;
        struct mesh_publication *publication=mesh_publication_at(m,row);
        struct mesh_target *targets=publication->targets;
        for(uint32_t chunk=0;chunk<chunks;chunk++){
          struct mesh_receive_record *record=&receive->records[peer[i].local_row+value*peer[i].stride+chunk];
          *record=(struct mesh_receive_record){
            .entry=mesh_page(m)+row+chunk,.row=chunk+1==chunks?row:MESH_ABSENT,
            .sends=receive->sends+next_send,.send_notices=publication->sends,.use_notices=publication->uses};
          size_t first_send=next_send;
          for(uint32_t region=first_region;region<first_region+regions;region++)for(uint32_t j=0;j<publication->sends;j++){
            struct mesh_link *out=&link->links[(targets[j].stream-send_first)/m->notice_bytes];
            for(uint32_t at=targets[j].index;at<targets[j].index+targets[j].count;at+=out->send_edges[at].chunks){
              if(chunk<out->send_edges[at].chunks){
                receive->sends[next_send++]=(struct mesh_send_binding){&out->send_edges[at+chunk].span,out->provider.device->regions[region]->lkey};
              }
            }
          }
          record->send_count=(uint32_t)((next_send-first_send)/regions);
        }
      }
    }
    for(size_t i=0;i<link->send_count;i++){
      struct mesh_send_edge *source=&link->send_edges[i];
      if(source->queue!=q)continue;
      source->pair=link->provider.queues[q].pair;source->post=link->provider.queues[q].send;
      source->span.length=bytes[2*q+MESH_SEND];
      if(mesh_buffers(m)[source->row].channel==MESH_ABSENT){
        uint32_t page=atomic_load_explicit(&mesh_page(m)[source->tag_row].mapping,memory_order_relaxed);
        struct ibv_sge span=link->provider.device->spans[page/m->block];
        source->span.addr=span.addr;source->span.lkey=span.lkey;
      }
    }
    free(bindings);free(peer);
  }
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    int error=link_receive_post(&link->provider.queues[q],&link->receive[q]);
    if(error<0)error=-error;
    if(error && error!=ENOMEM && error!=EAGAIN){errno=error;return -1;}
  }
  uint32_t posted=1,peer_posted;
  return exchange(socket,&posted,&peer_posted,sizeof posted,sizeof peer_posted,m,client,link->provider.deadline);
}
/* design/algorithm-sources.md#programcopy */
static inline __attribute__((always_inline)) int link_receive_post(struct mesh_queue *queue,struct mesh_receive *receive){
  while(receive->head!=atomic_load_explicit(&receive->tail,memory_order_acquire)){
    struct ibv_recv_wr *bad;
    int error=queue->receive(queue->pair,&receive->requests[receive->pages[receive->head&receive->mask]].request,&bad);
    if(error)return error;
    receive->head++;
  }
  return 0;
}
/* design/algorithm-sources.md#programcopy */
static inline __attribute__((always_inline)) int link_send_request(struct mesh_send_edge *source,uint32_t invocation){
  struct mesh_wire_tag *tag=(struct mesh_wire_tag *)(uintptr_t)source->span.addr;
  atomic_store_explicit(&tag->value,((uint64_t)invocation<<32)|source->tag_row,memory_order_relaxed);
  struct ibv_send_wr *bad;
  int error=source->post(source->pair,&source->request,&bad);
  return error<0?-error:error;
}
/* design/algorithm-sources.md#programcopy */
static int mesh_send_progress(struct mesh_link *link,struct mesh_send_edge *edges){
  struct mesh_verbs *v=&link->provider;
  struct ibv_wc *completions=v->completions+MESH_SEND;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    struct mesh_queue *queue=&v->queues[q];
    int count=queue->poll[MESH_SEND](queue->completions[MESH_SEND],1,completions);
    if(count<0){link_error(link,count,3);return count;}
    struct mesh_ready *ready=&link->ready[q];
    while(ready->head!=ready->tail){
      struct mesh_send_range pending=link->send_ready[ready->first+(ready->head&ready->mask)];
      int error=link_send_request(&edges[pending.first],pending.invocation);
      if(error){
        if(error!=ENOMEM && error!=EAGAIN){link_error(link,error,1);return error;}
        break;
      }
      ready->head++;
      if(++pending.first!=pending.end)link->send_ready[ready->first+(ready->tail++&ready->mask)]=pending;
    }
    if(count){
      struct ibv_wc *wc=completions;
      if(wc->status){link_error(link,wc->status,2);return wc->status;}
      struct mesh_free_ring *retirements=link->retirements[MESH_SEND];
      atomic_store_explicit(&retirements->slots[retirements->position++&retirements->mask],(uint32_t)wc->wr_id+1,memory_order_release);
    }
  }
  return 0;
}
/* design/algorithm-sources.md#programcopy */
static int mesh_receive_progress(struct mesh_link *link,struct mesh_free_ring *retirements,struct mesh_page_entry *pages){
  struct hdr *m=link->M;struct mesh_verbs *v=&link->provider;
  struct ibv_wc *completions=v->completions+MESH_RECEIVE;
  for(uint32_t q=0;q<(uint32_t)link->qps;q++){
    struct mesh_queue *queue=&v->queues[q];
    int count=queue->poll[MESH_RECEIVE](queue->completions[MESH_RECEIVE],1,completions);
    if(count<0){link_error(link,count,3);return count;}
    if(count){
      struct ibv_wc *wc=completions;
      if(wc->status){link_error(link,wc->status,2);return wc->status;}
      struct mesh_receive *receive=&link->receive[q];
      struct mesh_wire_tag *tag=(struct mesh_wire_tag *)(uintptr_t)wc->wr_id;
      uint64_t tag_value=atomic_load_explicit(&tag->value,memory_order_relaxed);
      uint64_t device_address=*((uint64_t *)tag-1);
      uint32_t region=(uint32_t)(wc->wr_id>>32)-receive->region_base;
      uint32_t slot=(uint32_t)(((uint64_t)((uint32_t)wc->wr_id>>receive->page_shift)*receive->block_reciprocal)>>32);
      uint32_t block=region*receive->region_blocks+slot,page=receive->page_base+block*receive->block;
      uint32_t invocation=(uint32_t)(tag_value>>32);
      struct mesh_receive_record record=receive->records[(uint32_t)tag_value];
      struct mesh_page_entry *entry=record.entry;
      atomic_store_explicit(&entry->mapping,((uint64_t)(block+receive->pool_offset)<<32)|page,memory_order_relaxed);
      atomic_store_explicit(&entry->address,receive->client_data+((uintptr_t)page<<receive->page_shift),memory_order_relaxed);
      atomic_store_explicit(&entry->device,device_address,memory_order_relaxed);
      struct mesh_send_binding *sends=record.sends+(size_t)region*record.send_count;
      for(uint32_t j=0;j<record.send_count;j++){
        struct mesh_send_binding binding=sends[j];
        binding.target->addr=(uintptr_t)tag;binding.target->lkey=binding.key;
      }
      if(record.row!=MESH_ABSENT){
        const struct mesh_target *targets=record.send_notices || record.use_notices?mesh_publication_at(m,record.row)->targets:NULL;
        mesh_publish(m,targets,record.send_notices,record.use_notices,pages+record.row,(uint64_t)invocation+1);
        atomic_store_explicit(&retirements->slots[retirements->position++&retirements->mask],record.row+1,memory_order_release);
      }
    }
    int error=link_receive_post(queue,&link->receive[q]);
    if(error<0)error=-error;
    if(error && error!=ENOMEM && error!=EAGAIN){link_error(link,error,1);return error;}
  }
  return 0;
}

/* design/algorithm-sources.md#programkernel_call */
static void link_publications(struct mesh_link *link,struct mesh_send_edge *edges,struct mesh_event_reader *notices){
  uint64_t event;
  while((event=mesh_event_take(notices))!=MESH_EVENT_ABSENT){
    uint32_t first=(uint32_t)event;
    for(uint32_t at=first,end=edges[first].end;at<end;){
      struct mesh_send_edge *source=&edges[at];
      uint32_t chunk=at,next=at+source->chunks,invocation=(uint32_t)(event>>32);
      do {
        int error=link_send_request(&edges[chunk],invocation);
        if(error){
          if(error!=ENOMEM && error!=EAGAIN){link_error(link,error,1);return;}
          struct mesh_ready *ready=&link->ready[source->queue];
          link->send_ready[ready->first+(ready->tail++&ready->mask)]=(struct mesh_send_range){chunk,next,invocation};
          if(mesh_send_progress(link,edges))return;
          break;
        }
      } while(++chunk<next);
      at=next;
    }
  }
}

/* design/algorithm-sources.md#programcopy */
static void *link_send_progress(void *argument){
  struct mesh_link *link=argument;
  struct mesh_send_edge *edges=link->send_edges;
  struct mesh_event_reader notices=link->notices;
  pthread_setname_np("mesh.rdma.send");
  while(atomic_load_explicit(&link->progressing,memory_order_acquire)){
    if(mesh_send_progress(link,edges))break;
    link_publications(link,edges,&notices);
  }
  return NULL;
}
/* design/algorithm-sources.md#transport-retirement */
static void *link_retire_progress(void *argument){
  struct mesh_link *link=argument;
  struct mesh_free_ring *rings[2]={link->retirements[MESH_SEND],link->retirements[MESH_RECEIVE]};
  struct mesh_send_edge *edges=link->send_edges;
  struct hdr *m=link->M;
  size_t heads[2]={0},masks[2]={rings[0]->mask,rings[1]->mask};
  pthread_setname_np("mesh.rdma.retire");
  for(;;){
    int running=atomic_load_explicit(&link->retiring,memory_order_acquire),progress=0;
    for(uint32_t d=0;d<2;d++){
      _Atomic uint32_t *slot=&rings[d]->slots[heads[d]&masks[d]];
      uint32_t index=atomic_load_explicit(slot,memory_order_acquire);
      if(!index)continue;
      atomic_store_explicit(slot,0,memory_order_relaxed);heads[d]++;progress=1;
      uint32_t row=index-1,completions;
      struct mesh_instance *instances;
      if(d==MESH_SEND){
        struct mesh_send_edge *source=&edges[row];
        if(--source->remaining)continue;
        source->remaining=source->chunks;
        row=source->row;instances=source->instances;completions=source->completions;
      } else {
        instances=link->instances;
        completions=mesh_buffers(m)[row].completions?0:link->instance_count;
      }
      mesh_buffer_release(m,row);
      mesh_instance_release(instances,completions);
    }
    for(uint32_t q=0;q<(uint32_t)link->qps;q++){
      struct mesh_receive *receive=&link->receive[q];
      uint64_t event=mesh_event_take(&receive->returns);
      if(event==MESH_EVENT_ABSENT)continue;
      progress=1;
      uint32_t row=(uint32_t)event;
      struct mesh_buffer *buffer=&mesh_buffers(m)[row];
      struct mesh_page_entry *entries=mesh_page(m)+row;
      uint32_t frame=buffer->frame,count=buffer->completions;
      uint32_t invocation=(uint32_t)(atomic_load_explicit(&entries->stamp,memory_order_relaxed)-1);
      uint64_t tail=atomic_load_explicit(&receive->tail,memory_order_relaxed);
      for(uint32_t chunk=0,chunks=buffer->pages/m->block;chunk<chunks;chunk++){
        receive->pages[tail++&receive->mask]=(uint32_t)(atomic_load_explicit(&entries[chunk].mapping,memory_order_relaxed)>>32);
        atomic_store_explicit(&entries[chunk].mapping,MESH_ABSENT,memory_order_relaxed);
      }
      mesh_buffer_reset(m,row);
      for(uint32_t i=0;i<count;i++){
        struct mesh_instance *instance=&link->instances[frame+i];
        atomic_store_explicit(&instance->invocation,invocation,memory_order_relaxed);
        mesh_instance_release(instance,1);
      }
      atomic_store_explicit(&receive->tail,tail,memory_order_release);
    }
    if(!running && !progress)return NULL;
  }
}
/* design/algorithm-sources.md#programcopy */
static void *link_receive_progress(void *argument){
  struct mesh_link *link=argument;
  struct mesh_free_ring *retirements=link->retirements[MESH_RECEIVE];
  struct mesh_page_entry *pages=mesh_page(link->M);
  pthread_setname_np("mesh.rdma.receive");
  while(atomic_load_explicit(&link->progressing,memory_order_acquire))
    if(mesh_receive_progress(link,retirements,pages))break;
  return NULL;
}

/* design/algorithm-sources.md#meshresult */
static void link_close(struct mesh_link *link,int *control){
  atomic_store_explicit(&link->progressing,0,memory_order_release);
  if(*control>=0)shutdown(*control,SHUT_RDWR);
  while(link->worker_count>1)pthread_join(link->workers[--link->worker_count],NULL);
  atomic_store_explicit(&link->retiring,0,memory_order_release);
  if(link->worker_count)pthread_join(link->workers[--link->worker_count],NULL);
  if(*control>=0){close(*control);*control=-1;}
  while(!down_pair(&link->provider))link_error(link,errno?errno:EIO,1);
  if(link->provider.listener>=0){close(link->provider.listener);link->provider.listener=-1;}
  atomic_store_explicit(&mesh_links(link->M)[link->index].port.phase,MESH_STOPPED,memory_order_release);
}

/* design/algorithm-sources.md#programcopy */
static void *link_run(void *argument){
  struct mesh_link *link=argument;struct hdr *m=link->M;
  struct mesh_port_info *port=&mesh_links(m)[link->index].port;
  uint32_t transfers=0;
  for(uint32_t q=0;q<m->qps;q++)for(int d=0;d<2;d++)transfers+=atomic_load(mesh_order_length(m,link->client,link->index*m->qps+q,d));
  struct kevent64_s event;
  EV_SET64(&event,(uint32_t)link->client,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,0,0,0);
  int error=kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?errno:0,control=-1;
  if(error){link_error(link,error,4);if(error==ESRCH)mesh_retire(m,link->client);}
  if(!error && transfers){
    atomic_store_explicit(&port->phase,MESH_PAIRING,memory_order_release);
    control=verbs_up(&link->provider,m,link->qps,link_configure,link,link->client);
    if(control<0)link_error(link,errno?errno:EIO,1);
    else {
      EV_SET64(&event,control,EVFILT_READ,EV_ADD|EV_CLEAR,0,0,0,0,0);
      error=kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)?errno:0;
      if(!error){
        __atomic_store_n(&mesh_links(m)[link->index].bandwidth,link->provider.bandwidth,__ATOMIC_RELAXED);
        atomic_store_explicit(&port->phase,MESH_PAIRED,memory_order_release);
      }
      atomic_store_explicit(&link->retiring,1,memory_order_release);
      void *(*progress[3])(void *)={link_retire_progress,link_send_progress,link_receive_progress};
      for(uint32_t d=0;d<3 && !error;d++){
        error=pthread_create(&link->workers[d],NULL,progress[d],link);
        if(error)break;
        link->worker_count++;
      }
      if(error)link_error(link,error,1);
    }
  } else atomic_store_explicit(&link->progressing,0,memory_order_release);
  for(;;){
    if(!atomic_load_explicit(&link->progressing,memory_order_acquire))link_close(link,&control);
    if(stop || atomic_load_explicit(&m->client,memory_order_acquire)!=link->client)break;
    int count=kevent64(link->events,NULL,0,&event,1,0,NULL);
    if(count<0){if(errno==EINTR)continue;link_error(link,errno,4);break;}
    if(event.flags&EV_ERROR)link_error(link,event.data,4);
    else if(event.filter==EVFILT_PROC){link_error(link,ECANCELED,4);mesh_retire(m,link->client);}
    else if(event.filter==EVFILT_READ)
      link_error(link,event.flags&EV_EOF?(event.fflags?event.fflags:ECONNRESET):EPROTO,4);
  }
  link_close(link,&control);
  EV_SET64(&event,(uint32_t)link->client,EVFILT_PROC,EV_DELETE,0,0,0,0,0);
  kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  return NULL;
}

/* design/algorithm-sources.md#programcopy */
int main(int argc,char **argv){
  const char *name=MESH_NAME;int me=0,layout=0;double pct=0;
  uint64_t arena_pages=0,block_pages=0;
  uint32_t link_count=0,device_count=0,qps=getenv("MESH_QPS")?(uint32_t)atoi(getenv("MESH_QPS")):1;
  struct mesh_link *links=aligned_alloc(_Alignof(struct mesh_link),(size_t)argc*sizeof *links);
  struct mesh_device *devices=calloc((size_t)argc,sizeof *devices);
  if(!links || !devices)die("bridge configuration allocation");
  memset(links,0,(size_t)argc*sizeof *links);
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I") && i+1<argc)me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M") && i+1<argc)pct=atof(argv[++i]);
    else if((!strcmp(argv[i],"-A") || !strcmp(argv[i],"-B")) && i+1<argc){
      char kind=argv[i][1],*end;uint64_t pages=strtoull(argv[++i],&end,10);
      if(*end || !pages || pages>INT32_MAX)die("configured page count");
      if(kind=='A')arena_pages=pages;else block_pages=pages;
    }
    else if(!strcmp(argv[i],"--layout"))layout=1;
    else if(!strcmp(argv[i],"-s") && i+1<argc)name=argv[++i];
    else if(!strcmp(argv[i],"--link") && i+1<argc){
      struct mesh_link *link=&links[link_count];
      char *fields=strdup(argv[++i]);link->configuration=fields;
      if(!fields)die("link configuration allocation");
      char *device=strsep(&fields,","),*peer=strsep(&fields,","),*local=strsep(&fields,","),*remote=strsep(&fields,","),*service=strsep(&fields,",");
      if(!device || !*device || !peer || !*peer || !local || !*local || !remote || !*remote || fields)die("link requires device,peer,local-address,remote-address[,service]");
      uint32_t d=0;
      while(d<device_count && strcmp(devices[d].name,device))d++;
      if(d==device_count){devices[d].name=device;pthread_mutex_init(&devices[d].setup,NULL);device_count++;}
      link->index=link_count++;link->provider=(struct mesh_verbs){.device=&devices[d],.peer=(uint32_t)strtoul(peer,NULL,10),.local_address=local,.remote_address=remote,.service=service?service:MESH_PORT,.listener=-1};
    }
    else die("unknown bridge option");
  }
  if(me<0 || !isfinite(pct) || pct<0 || pct>100 || !arena_pages || !block_pages || arena_pages<block_pages || !qps || qps>MESH_QPS)die("bridge geometry");
  const uint32_t pg=(uint32_t)getpagesize();
  /* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
  if((uint64_t)block_pages*pg+4096>16773120)die("transport chunk exceeds native request capacity");
  struct hdr geometry={0};
  uint64_t length=mesh_layout(&geometry,pg,(uint32_t)block_pages,(uint32_t)arena_pages,link_count,qps);
  uint64_t ram=0;size_t rl=sizeof ram;sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  if(pct && length>(uint64_t)(pct/100*(double)ram))die("configured graph exceeds page capacity");
  if(layout){printf("%llu\n",(unsigned long long)length);return 0;}
  atexit(down);struct sigaction sa={0};sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL);sigaction(SIGTERM,&sa,NULL);sigaction(SIGHUP,&sa,NULL);signal(SIGPIPE,SIG_IGN);
  shm_unlink(name);int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE);if(fd<0)die("shm");
  if(ftruncate(fd,(off_t)length))die("ftruncate");fchmod(fd,MESH_MODE);
  struct hdr *m=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(m==MAP_FAILED)die("mmap");shm=name;
  *m=geometry;m->node=(uint32_t)me;m->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(m);r++)atomic_store_explicit(&mesh_page(m)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  struct mesh_wire wire={0};
  if(wire_map(&wire,m,fd))die("transport page aliases");
  close(fd);
  for(uint32_t i=0;i<link_count;i++){
    struct mesh_link *link=&links[i];
    link->M=m;link->qps=(int)qps;link->provider.wire=&wire;link->links=links;
    link->provider.completions=calloc(2,sizeof *link->provider.completions);
    if(!link->provider.completions)die("link allocation");
    link->events=kqueue();
    struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,0,0,0);
    if(link->events<0 || kevent64(link->events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("link control events");
    mesh_links(m)[i].peer=link->provider.peer;
    snprintf(mesh_links(m)[i].device,sizeof mesh_links(m)[i].device,"%s",link->provider.device->name);
    atomic_store(&mesh_links(m)[i].port.phase,MESH_PAIRING);
  }
  int status=0;
  uint64_t retired=0;
  atomic_store(&m->bridge_pid,(uint64_t)getpid());atomic_store(&m->port.phase,MESH_PAIRING);
  __sync_synchronize();m->magic=MESH_MAGIC;
  fprintf(stderr,"bridge node %d: %u links, %u queue pairs per link\n",me,link_count,qps);
  while(!stop){
    uint64_t retirement=atomic_load_explicit(&m->retired,memory_order_acquire);
    if(retired!=retirement){mesh_retired_release(m);retired=retirement;}
    uint64_t client=atomic_load_explicit(&m->client,memory_order_acquire);
    if(!client || atomic_load_explicit(&m->configured,memory_order_acquire)!=client)continue;
    atomic_store_explicit(&m->device_client,client,memory_order_seq_cst);
    if(atomic_load_explicit(&m->client,memory_order_seq_cst)!=client || atomic_load_explicit(&m->configured,memory_order_seq_cst)!=client){atomic_store_explicit(&m->device_client,0,memory_order_seq_cst);continue;}
    uint32_t started=0;
    for(uint32_t i=0;i<link_count;i++){
      links[i].client=client;atomic_store_explicit(&links[i].progressing,1,memory_order_release);
      links[i].instances=mesh_instances(m,client);links[i].instance_count=m->instance_count[client>>63];
    }
    int preparation=0;
    for(uint32_t i=0;i<link_count && !preparation;i++){
      preparation=link_prepare(&links[i]);
      if(preparation)link_error(&links[i],preparation,1);
    }
    for(uint32_t i=0;i<link_count && !preparation;i++){
      int error=pthread_create(&links[i].controller,NULL,link_run,&links[i]);
      if(error){status=error;stop=1;break;}
      started++;
    }
    while(!stop && atomic_load_explicit(&m->client,memory_order_acquire)==client){}
    for(uint32_t i=0;i<started;i++)link_stop(&links[i]);
    for(uint32_t i=0;i<started;i++)pthread_join(links[i].controller,NULL);
    atomic_store_explicit(&m->device_client,0,memory_order_seq_cst);
  }
  for(uint32_t i=0;i<device_count;i++)if(!down_device(&devices[i])){fprintf(stderr,"verbs teardown failed: %s\n",strerror(errno));return 1;}
  atomic_store_explicit(&m->device_client,0,memory_order_seq_cst);
  mesh_retired_release(m);
  for(uint32_t i=0;i<link_count;i++){
    struct mesh_link *link=&links[i];
    atomic_store(&mesh_links(m)[i].port.phase,MESH_STOPPED);
    if(link->provider.listener>=0)close(link->provider.listener);
    close(link->events);
    free(link->provider.completions);free(link->send_edges);free(link->send_ready);free(link->retirements[0]);free(link->retirements[1]);free(link->notices.inputs);
    for(uint32_t q=0;q<qps;q++)link_receive_destroy(&link->receive[q]);
    free(link->configuration);
  }
  for(uint32_t i=0;i<device_count;i++)pthread_mutex_destroy(&devices[i].setup);
  free(devices);free(links);munmap(wire.data,wire.length);free(wire.spans);
  atomic_store(&m->port.phase,MESH_STOPPED);munmap(m,length);return status;
}
