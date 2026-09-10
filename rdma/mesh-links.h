#ifndef MESH_LINKS_H
#define MESH_LINKS_H
#define LINK_QUEUE 16384
#define LINK_LIMIT 16
enum { L_RECV=1, L_SEND, L_UP, L_RETIRED, L_FAULT };
struct link_event { uint32_t kind, page, bytes, error; uint64_t generation, header; uint32_t domain; };
struct link_queue { struct ring cursor; struct link_event entries[LINK_QUEUE]; };
struct mesh_link {
  const char *device, *local, *peer, *name;
  int node, peer_node;
  struct hdr *pages;
  pthread_t thread;
  struct link_queue command, completion;
  _Atomic int reset, stopped;
  _Atomic uint64_t heartbeat, operation, phase;
  _Atomic uint64_t generation;
  int up, receives, sends, send_capacity, receive_capacity;
  uint64_t pending;
};
// ../design/algorithm-sources.md#transport-page-addressing
static int link_push(struct link_queue *q,struct link_event event){
  uint64_t head=atomic_load_explicit(&q->cursor.head,memory_order_relaxed);
  if(head-atomic_load_explicit(&q->cursor.tail,memory_order_acquire)>=LINK_QUEUE) return -1;
  q->entries[head%LINK_QUEUE]=event;
  atomic_store_explicit(&q->cursor.head,head+1,memory_order_release); return 0;
}
// ../design/algorithm-sources.md#transport-page-addressing
static int link_pop(struct link_queue *q,struct link_event *event){
  uint64_t tail=atomic_load_explicit(&q->cursor.tail,memory_order_relaxed);
  if(tail==atomic_load_explicit(&q->cursor.head,memory_order_acquire)) return -1;
  *event=q->entries[tail%LINK_QUEUE];
  atomic_store_explicit(&q->cursor.tail,tail+1,memory_order_release); return 0;
}
// ../design/algorithm-sources.md#transport-page-addressing
static void *link_worker(void *argument){
  struct mesh_link *link=argument;
  struct hdr *pages=link->pages;
  char *memory=(char*)pages;
  size_t span=pages->data_off+(size_t)pages->pgsz*((size_t)pages->pool+pages->arena);
  selected_device=link->device; listen_address=link->local; expected_peer=link->peer_node;
  arc4random_buf(&mynonce,sizeof mynonce);
  flight_open(link->name,link->node,span);
  while(!stop){
    atomic_store(&link->phase,MESH_PAIRING);
    const char *peer=expected_peer>=0 && link->node<expected_peer?NULL:link->peer;
    while(!stop && ((lsock<0 && listener_up()) || verbs_up(peer,memory,span,link->node))){
      while(!down_pair()) usleep(20000);
      atomic_store(&link->heartbeat,flight_time()); usleep(100000);
    }
    if(stop) break;
    size_t capacity=(size_t)send_capacity+receive_capacity;
    struct ibv_wc *completions=calloc(capacity,sizeof *completions);
    struct ibv_sge (*sges)[2]=calloc(capacity,sizeof *sges);
    struct ibv_recv_wr *receives=calloc(receive_capacity,sizeof *receives);
    struct ibv_send_wr *sends=calloc(send_capacity,sizeof *sends);
    if(!completions || !sges || !receives || !sends) die("provider descriptors");
    link->peer_node=expected_peer; link->send_capacity=send_capacity; link->receive_capacity=receive_capacity; link->generation++;
    link_push(&link->completion,(struct link_event){.kind=L_UP,.generation=link->generation});
    atomic_store(&link->phase,MESH_PAIRED);
    while(!stop && !atomic_load(&link->reset)){
      if(atomic_load(&link->completion.cursor.head)-atomic_load(&link->completion.cursor.tail)>LINK_QUEUE-2*capacity-2) continue;
      atomic_store(&link->operation,F_POLL_CQ);
      int count=ibv_poll_cq(cq,(int)capacity,completions);
      atomic_store(&link->operation,0);
      if(count<0){
        link_push(&link->completion,(struct link_event){.kind=L_FAULT,.error=(uint32_t)count,.domain=3});
        while(!stop && !atomic_load(&link->reset)) usleep(1000);
        break;
      }
      for(int i=0;i<count;i++){
        struct ibv_wc *wc=&completions[i];
        int receive=(int)(wc->wr_id>>63);
        uint64_t offset=receive?pages->headers_off+(size_t)(uint32_t)wc->wr_id*MESH_HEADER_STRIDE:wc->wr_id;
        struct mesh_send *record=(struct mesh_send*)(memory+offset);
        link_push(&link->completion,(struct link_event){.kind=receive?L_RECV:L_SEND,
          .page=receive?(uint32_t)wc->wr_id:record->page,.header=offset,
          .bytes=wc->byte_len,.error=wc->status,.domain=wc->status?2:0});
      }
      size_t count_send=0,count_receive=0,count_command=0;
      struct link_event value;
      while(count_command<capacity && !link_pop(&link->command,&value)){
        struct link_event *command=&value;
        sges[count_command][0]=region_sge(memory,command->header,MESH_HEADER_BYTES);
        sges[count_command][1]=region_sge(memory,(size_t)((char*)mesh_at(pages,command->page)-memory),pages->pgsz);
        if(command->kind==L_RECV){
          receives[count_receive]=(struct ibv_recv_wr){.wr_id=(UINT64_C(1)<<63)|command->page,
            .sg_list=sges[count_command],.num_sge=2};
          if(count_receive) receives[count_receive-1].next=&receives[count_receive];
          count_receive++;
        } else {
          sends[count_send]=(struct ibv_send_wr){.wr_id=command->header,.sg_list=sges[count_command],
            .num_sge=2,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
          if(count_send) sends[count_send-1].next=&sends[count_send];
          count_send++;
        }
        count_command++;
      }
      if(count_receive){
        struct ibv_recv_wr *bad=NULL;
        int error=ibv_post_recv(qp,receives,&bad);
        for(struct ibv_recv_wr *wr=error?bad:NULL;wr;wr=wr->next){
          uint32_t page=(uint32_t)wr->wr_id;
          link_push(&link->completion,(struct link_event){.kind=L_RECV,.page=page,
            .header=pages->headers_off+(size_t)page*MESH_HEADER_STRIDE,.error=(uint32_t)error,.domain=1});
        }
      }
      if(count_send){
        struct ibv_send_wr *bad=NULL;
        int error=ibv_post_send(qp,sends,&bad);
        for(struct ibv_send_wr *wr=error?bad:NULL;wr;wr=wr->next){
          struct mesh_send *record=(struct mesh_send*)(memory+wr->wr_id);
          link_push(&link->completion,(struct link_event){.kind=L_SEND,.page=record->page,
            .header=wr->wr_id,.error=(uint32_t)error,.domain=1});
        }
      }
    }
    atomic_store(&link->phase,MESH_RETIRING);
    while(!down_verbs()){ atomic_store(&link->heartbeat,flight_time()); usleep(20000); }
    free(completions); free(sges); free(receives); free(sends);
    struct link_event discarded; while(!link_pop(&link->command,&discarded)){}
    link_push(&link->completion,(struct link_event){.kind=L_RETIRED});
    atomic_store(&link->reset,0);
  }
  while(!down_verbs()){ atomic_store(&link->heartbeat,flight_time()); usleep(20000); }
  if(lsock>=0) close(lsock);
  flight_heartbeat(MESH_STOPPED); atomic_store(&link->phase,MESH_STOPPED); atomic_store(&link->stopped,1);
  return NULL;
}
struct mesh_route { uint16_t link; };
// ../design/algorithm-sources.md#transport-page-addressing
static int link_submit(struct mesh_link *link,uint32_t kind,uint32_t page,uint64_t header){
  if(!link->up || (kind==L_SEND?link->sends>=link->send_capacity:link->receives>=link->receive_capacity)) return -1;
  struct mesh_send *record=(struct mesh_send*)((char*)link->pages+header);
  record->page=page;
  if(link_push(&link->command,(struct link_event){.kind=kind,.page=page,.header=header})) return -1;
  if(kind==L_SEND){
    record->previous=0; record->next=link->pending;
    if(link->pending) ((struct mesh_send*)((char*)link->pages+link->pending))->previous=header;
    link->pending=header; link->sends++;
  } else link->receives++;
  return 0;
}
// ../design/algorithm-sources.md#transport-page-addressing
static void link_release(struct mesh_link *link,uint64_t offset){
  char *base=(char*)link->pages;
  struct mesh_send *record=(struct mesh_send*)(base+offset);
  if(record->previous) ((struct mesh_send*)(base+record->previous))->next=record->next;
  else link->pending=record->next;
  if(record->next) ((struct mesh_send*)(base+record->next))->previous=record->previous;
  link->sends--;
}
#endif
