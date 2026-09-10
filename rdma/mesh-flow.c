#include "mesh-verbs.h"

#include "mesh-links.h"
#define COUNT(field) atomic_fetch_add_explicit(&M->field,1,memory_order_relaxed)
#define MOVE(page,to) do{ counts[owner[page]]--; owner[page]=(to); counts[to]++; }while(0)
#define RELEASE(page) do{ MOVE(page,FREE); free_pages[counts[FREE]-1]=(page); }while(0)
// ../design/algorithm-sources.md#transport-page-addressing
static void mesh_submit(struct hdr *M,struct mesh_link *links,int link_count,
  const struct mesh_route *routes,uint64_t *submit_cursor,int *arena_pending){
  int me=(int)M->node; struct desc descriptor;
  size_t credits=0;
  for(int index=0;index<link_count;index++) if(links[index].up)
    credits+=(size_t)(links[index].provider.send_capacity-links[index].sends);
  uint64_t position;
  uint64_t available=atomic_load_explicit(&M->r[SUB].head,memory_order_acquire)-atomic_load_explicit(&M->r[SUB].tail,memory_order_relaxed);
  for(uint64_t budget=0;budget<available && credits && !stop &&
    atomic_load(&M->r[ACK].head)-atomic_load(&M->r[ACK].tail)+(uint64_t)(*arena_pending)<MESH_RING &&
    ring_select(&M->r[SUB],submit_cursor,&position);budget++){
    descriptor=*slot(M,SUB,position); uint32_t page=descriptor.page;
    struct mesh_page_header *header=(struct mesh_page_header*)((char*)M+descriptor.header);
    header->wire=(struct wire){.src=(uint16_t)me,.dst=descriptor.node};
    int next=routes[descriptor.node].link;
    if(link_submit(&links[next],L_SEND,page,descriptor.header)) continue;
    (*arena_pending)++; credits--; COUNT(sent);
    if(links[next].sends==links[next].provider.send_capacity) link_flush(&links[next]);
    ring_erase(&M->r[SUB],slot(M,SUB,0),sizeof descriptor,MESH_RING,position);
  }
  for(int index=0;index<link_count;index++){
    link_flush(&links[index]);
  }
}
// ../design/algorithm-sources.md#transport-page-addressing
static int mesh_progress(struct hdr *M,struct mesh_link *links,int link_count,
  const struct mesh_route *routes,int *free_pages,unsigned char *owner,
  unsigned char *pool_link,int *counts,uint64_t *submit_cursor,int *arena_pending){
  int stopped=0;
  mesh_submit(M,links,link_count,routes,submit_cursor,arena_pending);
  int me=(int)M->node,pool=(int)M->pool,receive_share=pool/link_count;
  struct mesh_port_info *ports=mesh_ports(M);
  for(int index=0;index<link_count;index++){
    struct mesh_link *link=&links[index]; struct mesh_verbs *v=&link->provider;
    if(atomic_exchange(&ports[index].reset_request,0)) atomic_store(&link->reset,1);
    int ownership=atomic_load_explicit(&link->ownership,memory_order_acquire);
    if(ownership==LINK_ACTIVE){
      if(stop || atomic_load(&link->reset)){
        link->up=0; ownership=LINK_RELEASED;
        atomic_store_explicit(&link->ownership,ownership,memory_order_release);
      } else if(!link->faulted){
        link->up=1; ports[index].peer=(uint16_t)link->peer_node;
      }
    }
    if(link->up && v->completed<2*(v->send_capacity+v->receive_capacity)){
#ifdef MESH_TRANSPORT_TIMING
      double poll_begin=monotime();
#endif
      int count=mesh_transport_complete(v->completion_queue,2*(v->send_capacity+v->receive_capacity)-v->completed,v->completions+v->completed);
#ifdef MESH_TRANSPORT_TIMING
      measured_poll_seconds+=monotime()-poll_begin; measured_polls++;
#endif
      if(count<0){
        ports[index].when=flight_time(); ports[index].code=count; ports[index].domain=3;
        COUNT(bad); link->up=0; link->faulted=1; atomic_store(&link->phase,MESH_RETIRING);
      } else v->completed+=count;
    }
    int accessible=ownership==LINK_ACTIVE || ownership==LINK_RELEASED || ownership==LINK_RETIRED;
    int retained=0;
    for(int position=0;accessible && position<v->completed;position++){
      struct ibv_wc *wc=&v->completions[position];
      int receive=(int)(wc->wr_id>>63),header_completion=(int)((wc->wr_id>>62)&1);
      uint64_t offset=receive?M->headers_off+(size_t)(uint32_t)wc->wr_id*MESH_HEADER_STRIDE:wc->wr_id&~(UINT64_C(1)<<62);
      struct mesh_send *record=(struct mesh_send*)((char*)M+offset);
      uint32_t page=receive?(uint32_t)wc->wr_id:record->page;
      uint32_t error=wc->status,domain=error?2:0;
      if(error && !record->header.code){
        struct mesh_page_header *header=(struct mesh_page_header*)((char*)M+offset);
        header->when=flight_time(); header->code=error; header->domain=domain;
        header->index=page; header->peer=(uint32_t)link->peer_node;
        if(receive) header->function=UINT32_MAX;
      }
      if(header_completion) goto accepted;
      error=(uint32_t)record->header.code; domain=record->header.domain;
      if(!receive){
        struct desc ack={.page=page,.header=offset,.error=error,.domain=domain};
        if(offset>=M->data_off){
          link_release(link,offset); (*arena_pending)--; push(M,ACK,&ack);
        } else {
          link_release(link,offset);
          if(error){ push(M,CMP,&ack); MOVE(page,APP); }else RELEASE(page);
        }
        goto accepted;
      }
      if(error){
        struct desc failure={.page=page,.header=offset,.error=error,.domain=domain};
        push(M,CMP,&failure);
        MOVE(page,APP); goto accepted;
      }
      struct mesh_page_header *header=mesh_header(M,page);
      uint32_t bytes=wc->byte_len;
      if(header->wire.dst!=(uint16_t)me){
        if(stop){ header->when=flight_time(); header->code=ECANCELED; header->domain=1; RELEASE(page); goto accepted; }
        int next=routes[header->wire.dst].link;
        if(link_submit(&links[next],L_SEND,page,offset)){ v->completions[retained++]=*wc; continue; }
        MOVE(page,SEND); pool_link[page]=(unsigned char)next; COUNT(sent);
      } else {
        struct desc receive={.page=page,.header=offset,.bytes=bytes,.node=header->wire.src};
        push(M,CMP,&receive);
        MOVE(page,APP); COUNT(recvd);
      }
accepted:
      if(receive && !header_completion) link->receives--;
    }
    if(accessible) v->completed=retained;
    if(ownership==LINK_RETIRED && !v->completed){
      for(int i=0;i<pool;i++) if(pool_link[i]==index && owner[i]==RECV){
        struct mesh_page_header *header=mesh_header(M,(uint32_t)i);
        if(header->code){
          struct desc failure={.page=(uint32_t)i,.error=(uint32_t)header->code,.domain=header->domain};
          push(M,CMP,&failure); MOVE(i,APP);
        }else RELEASE(i);
      }
      while(link->pending){
        uint64_t offset=link->pending;
        struct mesh_send *record=(struct mesh_send*)((char*)M+offset);
        if(!record->header.code){
          record->header.when=flight_time(); record->header.code=ECANCELED; record->header.domain=1;
        }
        struct desc completion={.page=record->page,.header=offset,.error=(uint32_t)record->header.code,.domain=record->header.domain};
        link_release(link,offset);
        if(offset>=M->data_off){ (*arena_pending)--; push(M,ACK,&completion); }
        else { push(M,CMP,&completion); MOVE(completion.page,APP); }
      }
      if(!link->pending){
        link->receives=0; link->faulted=0;
        atomic_store_explicit(&link->ownership,LINK_ACKNOWLEDGED,memory_order_release);
      }
    }
    stopped+=atomic_load(&link->stopped);
    int receive_limit=link->up?(receive_share<v->receive_capacity?receive_share:v->receive_capacity):0;
    while(!stop && counts[FREE] && link->receives<receive_limit &&
      atomic_load_explicit(&M->r[CMP].head,memory_order_relaxed)-atomic_load_explicit(&M->r[CMP].tail,memory_order_acquire)+(uint64_t)counts[RECV]+(uint64_t)counts[SEND]<MESH_RING){
      int page=free_pages[counts[FREE]-1];
      if(link_submit(link,L_RECV,(uint32_t)page,M->headers_off+(size_t)page*MESH_HEADER_STRIDE)) break;
      MOVE(page,RECV); pool_link[page]=(unsigned char)index;
    }
    link_flush(link);
  }
  mesh_submit(M,links,link_count,routes,submit_cursor,arena_pending);
  return stopped;
}
// ../design/algorithm-sources.md#transport-page-addressing
int main(int argc,char**argv){
  const char *peer=NULL,*name=MESH_NAME; int me=0,link_count=0,layout=0; double pct=0;
  uint64_t arena_pages=0,receive_pages=0;
  struct mesh_link *links=calloc(LINK_LIMIT,sizeof *links);
  struct mesh_route *routes=calloc(65536,sizeof *routes);
  if(!links||!routes) die("link metadata");
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-I") && i+1<argc) me=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-M") && i+1<argc) pct=atof(argv[++i]);
    else if((!strcmp(argv[i],"-A") || !strcmp(argv[i],"-R")) && i+1<argc){
      int arena=!strcmp(argv[i],"-A"); char *end;
      uint64_t pages=strtoull(argv[++i],&end,10);
      if(*end || !pages || pages>INT32_MAX) die("configured page count");
      if(arena) arena_pages=pages; else receive_pages=pages;
    }
    else if(!strcmp(argv[i],"--layout")) layout=1;
    else if(!strcmp(argv[i],"-s") && i+1<argc) name=argv[++i];
    else if(!strcmp(argv[i],"--route") && i+1<argc){
      unsigned destination,index; char extra;
      if(sscanf(argv[++i],"%u:%u%c",&destination,&index,&extra)!=2 || destination>=65535 || index>=LINK_LIMIT) die("route descriptor");
      routes[destination].link=(uint16_t)index;
    } else if(!strcmp(argv[i],"--link") && i+1<argc){
      if(link_count==LINK_LIMIT) die("link descriptor capacity");
      char *fields=strdup(argv[++i]),*cursor=fields,*part[4];
      for(int j=0;j<4;j++) part[j]=strsep(&cursor,",");
      if(!part[0]||!part[1]||!part[2]||!part[3]||cursor) die("link requires device,peer,local-address,peer-address");
      char *end; unsigned long node=strtoul(part[1],&end,10);
      if(*end || node>=65535) die("link peer identity");
      struct mesh_link *link=&links[link_count];
      link->device=part[0]; link->peer_node=(int)node; link->local=part[2]; link->peer=part[3];
      routes[node].link=(uint16_t)link_count++;
    } else if(argv[i][0]=='-') die("unknown bridge option");
    else peer=argv[i];
  }
  if(me<0 || me>=65535 || !isfinite(pct) || pct<0 || pct>100 || !arena_pages || !receive_pages) die("bridge geometry");
  if(!link_count){ link_count=1; links[0].peer=peer; links[0].peer_node=-1; }
  for(int i=0;i<65536;i++) if(routes[i].link>=link_count) die("route names absent link");
  atexit(down); struct sigaction sa={0}; sa.sa_handler=onsig;
  sigaction(SIGINT,&sa,NULL); sigaction(SIGTERM,&sa,NULL); sigaction(SIGHUP,&sa,NULL); signal(SIGPIPE,SIG_IGN);
  uint64_t ram=0; size_t rl=sizeof ram; sysctlbyname("hw.memsize",&ram,&rl,NULL,0);
  const uint32_t pg=(uint32_t)getpagesize();
  size_t h0=(RINGS+NRING*MESH_RING*sizeof(struct desc)+LINK_LIMIT*sizeof(struct mesh_port_info)+pg-1)/pg*pg;
  uint64_t wanted=arena_pages+receive_pages;
  if(wanted>INT32_MAX) die("page index capacity");
  int np=(int)wanted,pool=(int)receive_pages;
  size_t d0=(h0+(size_t)np*MESH_HEADER_STRIDE+pg-1)/pg*pg,span=(size_t)pg*np;
  if(pct && d0+span>(uint64_t)(pct/100*(double)ram)) die("configured graph exceeds page capacity");
  if(layout){ printf("%zu\n",d0+span); return 0; }
  shm_unlink(name); int fd=shm_open(name,O_CREAT|O_RDWR,MESH_MODE); if(fd<0) die("shm");
  if(ftruncate(fd,(off_t)(d0+span))) die("ftruncate"); fchmod(fd,MESH_MODE);
  struct hdr *M=mmap(NULL,d0+span,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);
  if(M==MAP_FAILED) die("mmap"); shm=name;
  M->pgsz=pg; M->pool=(uint32_t)pool; M->arena=(uint32_t)(np-pool); M->node=(uint32_t)me;
  M->version=MESH_VERSION; M->headers_off=h0; M->data_off=d0; atomic_store(&M->port_count,(uint64_t)link_count);
  __sync_synchronize(); M->magic=MESH_MAGIC;
  int *free_pages=malloc((size_t)pool*sizeof *free_pages);
  unsigned char *owner=calloc((size_t)pool,1),*pool_link=calloc((size_t)pool,1);
  if(!free_pages||!owner||!pool_link) die("page ownership metadata");
  int counts[NOWN]={0}; counts[FREE]=pool;
  for(int i=0;i<pool;i++) free_pages[i]=i;
  struct mesh_port_info *ports=mesh_ports(M);
  flight_status=M; atomic_store(&M->bridge_pid,(uint64_t)getpid());
  for(int i=0;i<link_count;i++){
    links[i].node=me; links[i].name=name; links[i].pages=M; links[i].status=&ports[i];
    snprintf(ports[i].device,sizeof ports[i].device,"%s",links[i].device?links[i].device:"automatic");
    ports[i].peer=(uint16_t)links[i].peer_node;
    int error=pthread_create(&links[i].thread,NULL,link_worker,&links[i]);
    if(error){ errno=error; perror("link worker"); atomic_store(&links[i].stopped,1); stop=1; }
  }
  double began=now(),telemetry=began; int arena_pending=0;
  uint64_t submit_cursor=0;
  for(;;){
#ifdef MESH_TRANSPORT_TIMING
    double pass_begin=monotime();
#endif
    uint64_t release_tail=atomic_load_explicit(&M->r[REL].tail,memory_order_relaxed);
    uint64_t released=atomic_load_explicit(&M->r[REL].head,memory_order_acquire)-release_tail;
    int stopped;
    do{
      stopped=mesh_progress(M,links,link_count,routes,free_pages,owner,pool_link,counts,&submit_cursor,&arena_pending);
      if(released){
        struct desc descriptor=*slot(M,REL,release_tail);
        memset(mesh_at(M,descriptor.page),0,pg);
        struct mesh_row *row=(struct mesh_row*)((char*)M+descriptor.header);
        uint64_t stamp=__atomic_load_n(&row->stamp,__ATOMIC_RELAXED);
        __atomic_store_n(&row->page,MESH_ROW_ABSENT,__ATOMIC_RELEASE);
        __atomic_store_n(&row->stamp,stamp&~MESH_ROW_WRITING,__ATOMIC_RELEASE);
        if(descriptor.page<(uint32_t)pool) RELEASE(descriptor.page);
        atomic_store_explicit(&M->r[REL].tail,++release_tail,memory_order_release);
        released--;
      }
    }while(released);
    double stamp=now();
    if(stamp-telemetry>=0.25){
      int live=0;
      for(int index=0;index<link_count;index++){
        struct mesh_link *link=&links[index];
        live+=link->up;
        atomic_store(&ports[index].phase,atomic_load(&link->phase));
        atomic_store(&ports[index].heartbeat_us,atomic_load(&link->heartbeat));
        atomic_store(&ports[index].operation,atomic_load(&link->operation));
        atomic_store(&ports[index].generation,link->generation);
      }
      flight_heartbeat(stop?MESH_STOPPING:live?MESH_PAIRED:MESH_PAIRING);
      for(int i=0;i<NOWN;i++) atomic_store(&M->mean[i],counts[i]);
      atomic_store(&M->up_ms,(uint64_t)((stamp-began)*1000)); telemetry=stamp;
#ifdef MESH_TRANSPORT_TIMING
      fprintf(stderr,"{\"transport_timing\":{\"passes\":%llu,\"pass_seconds\":%.9f,\"polls\":%llu,\"poll_seconds\":%.9f}}\n",
        (unsigned long long)measured_passes,measured_pass_seconds,
        (unsigned long long)measured_polls,measured_poll_seconds);
#endif
    }
#ifdef MESH_TRANSPORT_TIMING
    measured_pass_seconds+=monotime()-pass_begin; measured_passes++;
#endif
    if(stop && stopped==link_count &&
      atomic_load_explicit(&M->r[REL].head,memory_order_acquire)==
      atomic_load_explicit(&M->r[REL].tail,memory_order_relaxed)) break;
  }
  for(int i=0;i<link_count;i++) pthread_join(links[i].thread,NULL);
#ifdef MESH_TRANSPORT_TIMING
  fprintf(stderr,"{\"transport_timing\":{\"passes\":%llu,\"pass_seconds\":%.9f,\"polls\":%llu,\"poll_seconds\":%.9f}}\n",
    (unsigned long long)measured_passes,measured_pass_seconds,
    (unsigned long long)measured_polls,measured_poll_seconds);
#endif
  flight_heartbeat(MESH_STOPPED);
  for(int i=0;i<link_count;i++) free((void*)links[i].device);
  free(links); free(routes); free(free_pages); free(owner); free(pool_link);
  flight_status=NULL; munmap(M,d0+span); return 0;
}
