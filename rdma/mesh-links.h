#ifndef MESH_LINKS_H
#define MESH_LINKS_H
#define LINK_LIMIT 16
enum { L_RECV=1, L_SEND };
enum { LINK_SETUP, LINK_ACTIVE, LINK_RELEASED, LINK_RETIRED, LINK_ACKNOWLEDGED };
struct mesh_link {
  const char *device, *local, *peer, *name;
  int node, peer_node;
  struct hdr *pages;
  struct mesh_port_info *status;
  struct mesh_verbs provider;
  pthread_t thread;
  _Atomic int ownership, reset, stopped;
  _Atomic uint64_t heartbeat, operation, phase, generation;
  int up, faulted, receives, sends;
  uint64_t pending;
};
// ../design/algorithm-sources.md#transport-page-addressing
static void *link_worker(void *argument){
  struct mesh_link *link=argument;
  struct hdr *pages=link->pages;
  provider=&link->provider;
  char *memory=(char*)pages;
  size_t span=pages->data_off+(size_t)pages->pgsz*((size_t)pages->pool+pages->arena);
  selected_device=link->device; listen_address=link->local; expected_peer=link->peer_node;
  arc4random_buf(&mynonce,sizeof mynonce);
  flight_open(link->name,link->node,span);
  while(!stop){
    atomic_store(&link->phase,MESH_PAIRING);
    const char *peer=expected_peer>=0 && link->node<expected_peer?NULL:link->peer;
    while(!stop && ((lsock<0 && listener_up()) || verbs_up(peer,memory,span,link->node))){
      while(!(retire_device?down_verbs():down_pair())) usleep(20000);
      retire_device=0;
      atomic_store(&link->heartbeat,flight_time()); usleep(100000);
    }
    if(stop) break;
    size_t capacity=(size_t)provider->send_capacity+provider->receive_capacity;
    provider->completions=calloc(2*capacity,sizeof *provider->completions);
    provider->sges=calloc(capacity,sizeof *provider->sges);
    provider->receives=calloc(2*provider->receive_capacity,sizeof *provider->receives);
    provider->sends=calloc(2*provider->send_capacity,sizeof *provider->sends);
    if(!provider->completions || !provider->sges || !provider->receives || !provider->sends) die("provider descriptors");
    link->peer_node=expected_peer; link->generation++;
    atomic_store_explicit(&link->ownership,LINK_ACTIVE,memory_order_release);
    while(atomic_load_explicit(&link->ownership,memory_order_acquire)!=LINK_RELEASED) usleep(1000);
    atomic_store(&link->phase,MESH_RETIRING);
    while(!down_verbs()){ atomic_store(&link->heartbeat,flight_time()); usleep(20000); }
    atomic_store_explicit(&link->ownership,LINK_RETIRED,memory_order_release);
    while(atomic_load_explicit(&link->ownership,memory_order_acquire)!=LINK_ACKNOWLEDGED) usleep(1000);
    free(provider->completions); free(provider->sges); free(provider->receives); free(provider->sends);
    provider->completed=provider->receiving=provider->sending=0;
    atomic_store(&link->reset,0);
    atomic_store_explicit(&link->ownership,LINK_SETUP,memory_order_release);
  }
  while(!down_verbs()){ atomic_store(&link->heartbeat,flight_time()); usleep(20000); }
  if(lsock>=0) close(lsock);
  flight_heartbeat(MESH_STOPPED); atomic_store(&link->phase,MESH_STOPPED); atomic_store(&link->stopped,1);
  return NULL;
}
struct mesh_route { uint16_t link; };
// ../design/algorithm-sources.md#transport-page-addressing
static int link_submit(struct mesh_link *link,uint32_t kind,uint32_t page,uint64_t header){
  struct mesh_verbs *v=&link->provider;
  if(!link->up || (kind==L_SEND?link->sends>=v->send_capacity:link->receives>=v->receive_capacity)) return -1;
  provider=v;
  int index=kind==L_SEND?v->receive_capacity+v->sending/2:v->receiving/2;
  v->sges[index][0]=region_sge((char*)link->pages,header,MESH_HEADER_BYTES);
  v->sges[index][1]=region_sge((char*)link->pages,link->pages->data_off+(size_t)page*link->pages->pgsz,link->pages->pgsz);
  struct mesh_send *record=(struct mesh_send*)((char*)link->pages+header);
  record->page=page; record->header.code=0; record->header.domain=0;
  if(kind==L_SEND){
    for(int part=0;part<2;part++){
      struct ibv_sge *span=&v->sges[index][part];
      mesh_transport_send_span(&v->sends[v->sending],span,header|(part?0:UINT64_C(1)<<62),
        (void*)(uintptr_t)span->addr,span->length,span->lkey,NULL);
      if(v->sending) v->sends[v->sending-1].next=&v->sends[v->sending];
      v->sending++;
    }
    record->previous=0; record->next=link->pending;
    if(link->pending) ((struct mesh_send*)((char*)link->pages+link->pending))->previous=header;
    link->pending=header; link->sends++;
  } else {
    for(int part=0;part<2;part++){
      struct ibv_sge *span=&v->sges[index][part];
      mesh_transport_receive_span(&v->receives[v->receiving],span,(UINT64_C(1)<<63)|(part?0:UINT64_C(1)<<62)|page,
        (void*)(uintptr_t)span->addr,span->length,span->lkey,NULL);
      if(v->receiving) v->receives[v->receiving-1].next=&v->receives[v->receiving];
      v->receiving++;
    }
    link->receives++;
  }
  return 0;
}
// ../design/algorithm-sources.md#transport-page-addressing
static void link_flush(struct mesh_link *link){
  struct mesh_verbs *v=&link->provider;
  if(!link->up) return;
  if(v->receiving){
    struct ibv_recv_wr *bad=NULL;
    int error=mesh_transport_receive(v->pair,v->receives,&bad);
    if(!error && atomic_load(&link->phase)==MESH_PAIRING) atomic_store(&link->phase,MESH_PAIRED);
    if(error){ link->status->when=flight_time(); link->status->code=error; link->status->domain=1; }
    v->receiving=0;
  }
  if(v->sending){
    struct ibv_send_wr *bad=NULL;
    int error=mesh_transport_send(v->pair,v->sends,&bad);
    if(error){ link->status->when=flight_time(); link->status->code=error; link->status->domain=1; }
    v->sending=0;
  }
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
