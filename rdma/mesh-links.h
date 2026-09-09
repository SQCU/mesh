#ifndef MESH_LINKS_H
#define MESH_LINKS_H
#define LINK_QUEUE 16384
#define LINK_LIMIT 16
enum { L_RECV=1, L_SEND, L_UP, L_RETIRED, L_FAULT, L_READY };
struct link_event { uint32_t kind, page, bytes, error; uint64_t generation; };
struct link_queue { struct ring cursor; struct link_event entries[LINK_QUEUE]; };
struct mesh_link {
  const char *device, *local, *peer, *name;
  int node, peer_node;
  char *memory; size_t span;
  pthread_t thread;
  struct link_queue command, completion;
  _Atomic int reset, stopped;
  _Atomic uint64_t heartbeat, operation, phase;
  uint64_t cursor, accepted;
  int up, receives, sends;
  uint32_t probe_page;
  double peer_seen, probe_at;
};
static int link_push(struct link_queue *q,struct link_event event){
  uint64_t head=atomic_load_explicit(&q->cursor.head,memory_order_relaxed);
  if(head-atomic_load_explicit(&q->cursor.tail,memory_order_acquire)>=LINK_QUEUE) return -1;
  q->entries[head%LINK_QUEUE]=event;
  atomic_store_explicit(&q->cursor.head,head+1,memory_order_release); return 0;
}
static int link_pop(struct link_queue *q,struct link_event *event){
  uint64_t tail=atomic_load_explicit(&q->cursor.tail,memory_order_relaxed);
  if(tail==atomic_load_explicit(&q->cursor.head,memory_order_acquire)) return -1;
  *event=q->entries[tail%LINK_QUEUE];
  atomic_store_explicit(&q->cursor.tail,tail+1,memory_order_release); return 0;
}
static void link_emit(struct mesh_link *link,struct link_event event){
  while(link_push(&link->completion,event)){ atomic_store(&link->heartbeat,flight_time()); usleep(100); }
}
static void *link_worker(void *argument){
  struct mesh_link *link=argument;
  selected_device=link->device; listen_address=link->local; expected_peer=link->peer_node;
  arc4random_buf(&mynonce,sizeof mynonce); if(!mynonce) mynonce=1;
  char label[128]; snprintf(label,sizeof label,"%s-%s",link->name,link->device?link->device:"automatic");
  flight_open(label,link->node,link->span);
  double retry_at=0; uint64_t generation=0;
  while(!stop){
    atomic_store(&link->heartbeat,flight_time()); atomic_store(&link->phase,MESH_PAIRING);
    if(monotime()<retry_at){ usleep(100000); continue; }
    if(lsock<0 && listener_up()){ retry_at=monotime()+retry_delay(); continue; }
    const char *peer=expected_peer>=0 && link->node<expected_peer?NULL:link->peer;
    int ready=!verbs_up(peer,link->memory,link->span,link->node);
    if(ready && !stop){
      backoff_n=0; generation++; atomic_store(&link->phase,MESH_PAIRED);
      link_emit(link,(struct link_event){.kind=L_UP,.generation=generation});
      double checked=monotime(),awake_until=0; int alive=1;
      while(alive && !stop && !atomic_load(&link->reset)){
        atomic_store(&link->heartbeat,flight_time());
        uint64_t backlog=atomic_load(&link->completion.cursor.head)-atomic_load(&link->completion.cursor.tail);
        if(backlog>LINK_QUEUE-128){ usleep(IDLE_POLL_US); continue; }
        struct ibv_wc completions[32];
        atomic_store(&link->operation,F_POLL_CQ);
        int count=TRACE(POLL_CQ,cq,32,0,ibv_poll_cq(cq,32,completions));
        atomic_store(&link->operation,0);
        if(count<0){ alive=0; link_emit(link,(struct link_event){.kind=L_FAULT,.error=EIO,.generation=generation}); }
        for(int i=0;i<count;i++){
          struct ibv_wc *wc=&completions[i];
          if(wc->status!=IBV_WC_SUCCESS){
            alive=0; link_emit(link,(struct link_event){.kind=L_FAULT,.error=wc->status,.generation=generation}); break;
          }
          link_emit(link,(struct link_event){.kind=wc->opcode==IBV_WC_RECV?L_RECV:L_SEND,
            .page=(uint32_t)wc->wr_id,.bytes=wc->byte_len,.generation=generation});
        }
        struct link_event command; int activity=count>0;
        for(int budget=0;alive && budget<64 && !link_pop(&link->command,&command);budget++){
          if(command.generation!=generation) continue;
          activity=1;
          struct ibv_sge sge=region_sge(link->memory,(size_t)command.page*4096,command.bytes);
          int error;
          if(command.kind==L_RECV){
            struct ibv_recv_wr wr={.wr_id=command.page,.sg_list=&sge,.num_sge=1},*bad;
            error=TRACE(POST_RECV,qp,command.page,command.bytes,ibv_post_recv(qp,&wr,&bad));
          } else {
            struct ibv_send_wr wr={.wr_id=command.page,.sg_list=&sge,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
            error=TRACE(POST_SEND,qp,command.page,command.bytes,ibv_post_send(qp,&wr,&bad));
          }
          if(error){ alive=0; link_emit(link,(struct link_event){.kind=L_FAULT,.error=(uint32_t)error,.generation=generation}); }
        }
        if(monotime()-checked>=1){
          flight_heartbeat(MESH_PAIRED); checked=monotime();
          int incoming=accept(lsock,NULL,NULL);
          if(incoming>=0){ close(incoming); alive=0; }
          if(TRACE(QUERY_PORT,ctx,1,0,ibv_query_port(ctx,1,&pa)) || pa.state!=IBV_PORT_ACTIVE){ alive=0; retire_device=1; }
        }
        idle_poll(activity,&awake_until);
      }
    }
    atomic_store(&link->phase,stop?MESH_STOPPING:MESH_RETIRING);
    while(!(stop||retire_device?down_verbs():down_pair())){
      atomic_store(&link->heartbeat,flight_time()); usleep(20000);
    }
    retire_device=0; atomic_store(&link->reset,0);
    struct link_event discarded; while(!link_pop(&link->command,&discarded)){}
    if(ready) link_emit(link,(struct link_event){.kind=L_RETIRED,.generation=generation});
    retry_at=monotime()+retry_delay();
  }
  while(!down_verbs()){ atomic_store(&link->heartbeat,flight_time()); usleep(20000); }
  if(lsock>=0) close(lsock);
  flight_heartbeat(MESH_STOPPED); atomic_store(&link->phase,MESH_STOPPED); atomic_store(&link->stopped,1);
  return NULL;
}
static void *(*bridge_link_worker)(void *)=link_worker;
struct mesh_route { uint16_t mask,first; };
static int route_link(const struct mesh_link *links,int count,const struct mesh_route *routes,uint16_t target,int incoming){
  struct mesh_route route=routes[target];
  if(route.mask){
    if(route.first<count && route.first!=incoming && links[route.first].up) return route.first;
    for(int i=0;i<count;i++) if((route.mask&(1u<<i)) && i!=incoming && links[i].up) return i;
    return -1;
  }
  for(int i=0;i<count;i++) if(i!=incoming && links[i].up && links[i].peer_node==target) return i;
  return count==1 && incoming<0 && links[0].up && links[0].peer_node<0?0:-1;
}
static int link_space(const struct mesh_link *link,uint32_t kind){
  return link->up && (kind==L_SEND?link->sends:link->receives)<QD;
}
static int link_submit(struct mesh_link *link,uint32_t kind,uint32_t page,uint32_t bytes){
  if(!link_space(link,kind)) return -1;
  if(link_push(&link->command,(struct link_event){kind,page,bytes,0,link->accepted})) return -1;
  if(kind==L_SEND) link->sends++; else link->receives++; return 0;
}
#endif
