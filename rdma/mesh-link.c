#include "mesh-bridge.h"
/* design/transport.md section 15 */
static _Atomic sig_atomic_t stop;
static struct hdr *M;
static struct wire_mapping wire;
static struct port ports[MESH_PORTS];
static uint32_t port_count,device_count,ones=MESH_ABSENT,link_page[MESH_PORTS],terminations_told,map_page,map_pages,map_room,
                known_count,child_asked,child_got;
static int events=-1,listener=-1,hello=-1,network=-1,addresses=-1,unreleased,finding,find_wanted,boots_set,child_halted;
static uint64_t now,sequence,memory,held_pages,boots_since,health_sent,child_bandwidth,owners[MESH_CLIENTS];
static int64_t stop_reason;
static struct mesh_ctx bridge={.fd=-1};
static dispatch_queue_t finder;
static _Atomic int stalled,found,arrived_elsewhere;
static _Atomic uint32_t faulted,terminated,terminations,arrived;
static _Atomic uint64_t exit_at;
static _Atomic uintptr_t retried[MESH_PORTS];
static uintptr_t provider[2];
static char devices[MESH_PORTS][32],boots_path[PATH_MAX],self_path[PATH_MAX],model[MESH_MODEL_BYTES];
static unsigned char child_in[sizeof(struct child_message)];
static struct known known[MESH_KNOWN];
static struct mesh_health health;
static const struct in6_addr everyone={{{0xff,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}};
_Static_assert(MESH_PORTS<=32,"a bit a port");

static uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_MONOTONIC); }
static uint64_t backoff(uint32_t n){ return MIN(UINT64_C(60)<<MIN(n,6u),UINT64_C(3600)); }
static void say(const char *format,...){
  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);
  struct tm utc;gmtime_r(&t.tv_sec,&utc);
  char line[2048];
  size_t n=strftime(line,sizeof line,"%Y-%m-%dT%H:%M:%S",&utc);
  n+=(size_t)snprintf(line+n,sizeof line-n,".%03ldZ %.3f ",t.tv_nsec/1000000,(double)clock_gettime_nsec_np(CLOCK_MONOTONIC)/1e9);
  va_list arguments;va_start(arguments,format);vsnprintf(line+n,sizeof line-n,format,arguments);va_end(arguments);
  fputs(line,stderr);
}
static void port_log(const struct port *port,const char *format,...){
  char line[2048];
  va_list arguments;va_start(arguments,format);vsnprintf(line,sizeof line,format,arguments);va_end(arguments);
  say("link %u (%s): %s",port->index,port->name,line);
}
static void die(const char *m){ say("%s\n",m); exit(1); }
static int watch(uint64_t ident,int16_t filter,uint16_t flags,uint32_t fflags,uint64_t udata){
  struct kevent64_s event;EV_SET64(&event,ident,filter,flags,fflags,0,udata,0,0);
  return kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
static void wake(void){ watch(0,EVFILT_USER,0,NOTE_TRIGGER,0); }
/* design/transport.md section 15, "Windows" */
static void quiet_until(_Atomic uint64_t *at,uint64_t until){
  uint64_t was=atomic_load(at);
  while(was<until && !atomic_compare_exchange_weak(at,&was,until)){}
}
static void quiet_all(uint64_t until){
  quiet_until(&exit_at,until);
  for(uint32_t p=0,count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);p<count;p++)quiet_until(&ports[p].quiet_at,until);
}
static uint64_t windows_end(void){
  uint64_t latest=atomic_load(&exit_at);
  for(uint32_t p=0;p<port_count;p++){const uint64_t quiet=atomic_load(&ports[p].quiet_at);latest=MAX(latest,quiet);}
  return latest;
}
static void port_halt(struct port *port,_Atomic uint32_t *which){
  atomic_fetch_add(&port->link.pairing,1);
  atomic_store(&port->link.served,0);
  if(which)atomic_fetch_or(which,UINT32_C(1)<<port->index);
}
/* design/transport.md section 9; section 15, "The boot ledger" */
#define BOOTS_SETTLE_NS (UINT64_C(600)*1000000000u)
static void boots_clear(void){ if(boots_set && !unlink(boots_path))boots_set=0; }
static void boots_hold(const char *configuration){
  const char *slash=strrchr(configuration,'/');
  snprintf(boots_path,sizeof boots_path,"%.*s/boots",slash?(int)(slash-configuration):1,slash?configuration:".");
  char boot[64]={0},prior[64]={0},next[PATH_MAX+8];
  size_t size=sizeof boot-1;
  unsigned abrupt=0;
  if(sysctlbyname("kern.bootsessionuuid",boot,&size,NULL,0)){
    say("bridge boots: this boot's identity (kern.bootsessionuuid): %s; no boot is counted\n",strerror(errno));return;}
  FILE *in=fopen(boots_path,"r");
  if(in){
    if(fscanf(in,"%63s %u",prior,&abrupt)!=2){prior[0]=0;abrupt=0;}
    fclose(in);
    if(strcmp(prior,boot))abrupt++;
  }
  snprintf(next,sizeof next,"%s.new",boots_path);
  FILE *out=fopen(next,"w");
  if(out){
    fprintf(out,"%s %u\n",boot,abrupt);
    const int kept=!fflush(out) && !fsync(fileno(out));
    if(!fclose(out) && kept && !rename(next,boots_path))boots_set=1;
  }
  if(!boots_set)say("bridge boots: %s not written (%s); no boot is counted\n",boots_path,strerror(errno));
  if(abrupt){
    const uint64_t hold=backoff(abrupt-1);
    say("bridge boots: %u boot%s in a row ended within %llu s of a bridge's start with %s set (a panic, a power loss or a "
        "reset); no device is opened for %llu s, and a stop signal ends the bridge meanwhile\n",abrupt,abrupt==1?"":"s",
        (unsigned long long)(BOOTS_SETTLE_NS/1000000000u),boots_path,(unsigned long long)hold);
    struct timespec rest={(time_t)hold,0};
    struct kevent64_s got;
    if(kevent64(events,NULL,0,&got,1,0,&rest)>0 && got.filter==EVFILT_SIGNAL){
      boots_clear();
      say("bridge boots: stopped while holding (signal %llu)\n",(unsigned long long)got.ident);
      exit(0);
    }
  }
  boots_since=clock_ns();
}
static void farewell_text(int64_t code,char *out,size_t room){
  if(code==SIGHUP)snprintf(out,room,"SIGHUP: its heartbeat restarts it");
  else if(code==SIGTERM)snprintf(out,room,"SIGTERM: its heartbeat's release or stop");
  else if(code>0)snprintf(out,room,"signal %lld",(long long)code);
  else snprintf(out,room,"no reason given");
}

/* design/transport.md section 14, "The stream"; section 15, "The data path" */
static uint64_t page_bytes(void){ return (uint64_t)M->block*M->pgsz; }
static struct mesh_stream *stream_of(const struct port *port,uint32_t s){ return mesh_stream_at(M,port->index*MESH_LINK_STREAMS+s); }
static _Atomic uint64_t *stream_words(uint64_t offset){ return (_Atomic uint64_t *)((char *)M+offset); }
static int owned(uint64_t owner){ return owner && atomic_load_explicit(&mesh_client_at(M,owner)->owner,memory_order_acquire)==owner; }
static void lane_lost(struct port *port,uint32_t lane,int64_t code){
  struct link *l=&port->link;
  if(atomic_fetch_and_explicit(&l->served,~(UINT32_C(1)<<lane),memory_order_acq_rel)>>lane&1)
    port_log(port,"lane %u failed: %s; the link toward node %u lost\n",lane,strerror((int)(code>0?code:EIO)),port->peer);
  atomic_fetch_or_explicit(&l->broken,UINT32_C(1)<<lane,memory_order_release);
  wake();
}
static void stream_time(_Atomic uint64_t *ring,uint64_t e,uint64_t at){
  _Atomic uint64_t *entry=ring+2*(e%MESH_STREAM_TIMES);
  atomic_store_explicit(entry,0,memory_order_relaxed);
  atomic_store_explicit(entry+1,at,memory_order_release);
  atomic_store_explicit(entry,1+e,memory_order_release);
}
static int row_at(struct mesh_row r,uint64_t *at){
  const uint64_t a=atomic_load_explicit(&mesh_page(M)[r.r].address,memory_order_relaxed),arena=(uint64_t)mesh_arena_pages(M)*M->pgsz;
  if(a<M->data_off || a-M->data_off>arena-page_bytes())return 0;
  *at=a-M->data_off;
  return 1;
}
static struct ibv_sge page_span(const struct port *port,uint64_t at){
  return (struct ibv_sge){.addr=(uintptr_t)wire.data+at,.length=(uint32_t)(page_bytes()-at%page_bytes()),
                          .lkey=port->device.extent_mrs[at/MESH_EXTENT]->lkey};
}
static const _Atomic uint64_t *region_word(uint64_t offset){
  return offset && offset<=M->length-8 && !(offset%8)?stream_words(offset):NULL;
}
static void walk_take(struct walk *w,struct mesh_stream *x,int land){
  const uint64_t n=atomic_load_explicit(land?&x->lands:&x->sends,memory_order_acquire);
  const struct mesh_entry *list=(const struct mesh_entry *)((char *)M+x->lists)+(uint64_t)land*MESH_STREAM_ENTRIES;
  uint64_t t=w->taken;
  for(;t<n && t-w->done<MESH_STREAM_ENTRIES;t++){
    struct taken *k=w->ring+t%MESH_STREAM_ENTRIES;
    const struct mesh_entry e=list[t%MESH_STREAM_ENTRIES];
    const _Atomic uint64_t *word=land?NULL:e.word?region_word(e.word):&x->ready;
    k->row=mesh_entry_row(&e);k->pages=mesh_entry_pages(&e);k->base=mesh_entry_base(&e);
    k->whole=mesh_rows_hold(M,k->row,k->pages) && (land || word);
    k->word=k->whole?word:NULL;
    k->page=w->pages;
    w->pages=mesh_pages_add(w->pages,k->pages);
  }
  if(t!=w->taken){w->taken=t;atomic_store_explicit(&x->taken[land],t,memory_order_release);}
}
static struct taken *walk_page(struct walk *w,struct mesh_pages q){
  if(w->at<w->done)w->at=w->done;
  for(;w->at<w->taken;w->at++){
    struct taken *t=w->ring+w->at%MESH_STREAM_ENTRIES;
    if(mesh_pages_lt(q,mesh_pages_add(t->page,t->pages)))return t;
  }
  return NULL;
}
static void walk_count(struct walk *w,struct mesh_pages through,_Atomic uint64_t *count){
  if(through.n!=w->count.n){w->count=through;mesh_pages_store(count,through);}
  while(w->done<w->taken){
    const struct taken *e=w->ring+w->done%MESH_STREAM_ENTRIES;
    if(mesh_pages_lt(through,mesh_pages_add(e->page,e->pages)))break;
    w->done++;
  }
}
static struct mesh_pages past(struct mesh_pages count,struct mesh_pages base){ return (struct mesh_pages){count.n>base.n?count.n-base.n:0}; }
static void stream_zero(struct mesh_stream *x,int posts){
  _Atomic uint64_t *words[]={&x->sends,&x->lands,&x->taken[0],&x->taken[1],&x->ready,&x->sent,&x->landed};
  for(size_t i=0;i<sizeof words/sizeof *words;i++)atomic_store_explicit(words[i],0,memory_order_relaxed);
  for(int way=0;posts && way<2;way++)
    for(int c=0;c<2;c++){atomic_store_explicit(&x->posts[way].posted[c],0,memory_order_relaxed);atomic_store_explicit(&x->posts[way].done[c],0,memory_order_relaxed);}
}
static void stream_free(struct mesh_stream *x,uint64_t owner){
  stream_zero(x,0);
  atomic_store_explicit(&x->owner,owner,memory_order_release);
}
static int holding_kept(const struct mesh_stream *x,uint64_t owner){
  return atomic_load_explicit(&x->owner,memory_order_acquire)==owner && owned(owner);
}
static inline void unit_told(struct mesh_stream *x,int way,uint32_t c,uint64_t u,uint32_t row){
  atomic_store_explicit(mesh_unit_row(M,x->rows,way,(int)c,u),row,memory_order_relaxed);
  atomic_store_explicit(&x->posts[way].posted[c],u+1,memory_order_release);
}
/* design/transport.md section 14, "Holding it" and "A holding's end" */
static void stream_turn_one(struct port *port,uint32_t t){
  struct link *l=&port->link;
  struct mesh_stream *x=stream_of(port,t);
  const uint32_t k=MESH_LANE_FIRST+2*t;
  if(l->hold[t]==HOLD_GONE)return;
  const uint64_t owner=atomic_load_explicit(&x->owner,memory_order_acquire);
  if(l->hold[t]==HOLD_NONE){
    if(!owner)return;
    if(owner&MESH_STREAM_GIVEN){stream_free(x,0);return;}
    l->held[t]=owner;l->number[t]++;l->hold[t]=HOLD_SERVE;
    l->sending[t]=(struct walk){.ring=l->sending[t].ring,.base=stream_pages(l->lane[k].sent,l->lane[k+1].sent)};
    atomic_store_explicit(&l->begun[t],l->number[t],memory_order_release);
  }
  if(l->hold[t]==HOLD_SERVE && owner==l->held[t] && owned(owner)){walk_take(l->sending+t,x,0);return;}
  if(l->hold[t]==HOLD_SERVE){l->hold[t]=HOLD_END;atomic_store_explicit(&l->ended[t],l->number[t],memory_order_release);}
  if(atomic_load_explicit(&l->stopped[t],memory_order_acquire)!=l->number[t])return;
  l->held[t]=0;
  uint64_t sends=0,receives=0;
  for(uint32_t c=0;c<2;c++){
    sends+=l->lane[k+c].sent-l->lane[k+c].done;
    receives+=atomic_load_explicit(&x->posts[1].posted[c],memory_order_acquire)-atomic_load_explicit(&x->posts[1].done[c],memory_order_acquire);
  }
  if(!sends && !receives){l->hold[t]=HOLD_NONE;stream_free(x,0);return;}
  l->hold[t]=HOLD_GONE;
  atomic_store_explicit(&x->live,0,memory_order_release);
  stream_free(x,MESH_STREAM_GIVEN);
  port_log(port,"stream %u toward node %u: its holding ended with %llu SENDs and %llu receives outstanding: the stream not live\n",t,
           port->peer,(unsigned long long)sends,(unsigned long long)receives);
  uint32_t live=0;
  for(uint32_t s=0;s<plan_streams(l->plan);s++)live|=atomic_load_explicit(&stream_of(port,s)->live,memory_order_acquire);
  if(live)return;
  port_log(port,"no stream toward node %u live: the link lost\n",port->peer);
  atomic_store_explicit(&l->lost,LINK_STREAMS_ENDED,memory_order_relaxed);
  atomic_fetch_or_explicit(&l->broken,l->plan>>MESH_LANE_FIRST<<MESH_LANE_FIRST,memory_order_release);
  wake();
}
static void stream_receive_turn(struct port *port,uint32_t t){
  struct link *l=&port->link;
  struct mesh_stream *x=stream_of(port,t);
  const struct lane_state *s=l->lane+MESH_LANE_FIRST+2*t;
  const uint64_t begun=atomic_load_explicit(&l->begun[t],memory_order_acquire),ended=atomic_load_explicit(&l->ended[t],memory_order_acquire);
  if(l->rnumber[t] && ended==l->rnumber[t]){l->rnumber[t]=0;atomic_store_explicit(&l->stopped[t],ended,memory_order_release);}
  if(l->rnumber[t]){walk_take(l->landing+t,x,1);return;}
  if(begun==atomic_load_explicit(&l->stopped[t],memory_order_relaxed))return;
  if(ended==begun){atomic_store_explicit(&l->stopped[t],begun,memory_order_release);return;}
  l->rnumber[t]=begun;l->rheld[t]=__atomic_load_n(&l->held[t],__ATOMIC_RELAXED);
  l->landing[t]=(struct walk){.ring=l->landing[t].ring,.base=stream_pages(s[0].posted,s[1].posted)};
  walk_take(l->landing+t,x,1);
}
static uint32_t stream_post(struct port *port,uint32_t t,uint32_t served,int way){
  struct link *l=&port->link;
  const uint32_t k=MESH_LANE_FIRST+2*t;
  struct mesh_stream *x=stream_of(port,t);
  if((served>>k&3)!=3 || (way?!l->rnumber[t] || !holding_kept(x,l->rheld[t]):l->hold[t]!=HOLD_SERVE))return 0;
  struct walk *w=way?l->landing+t:l->sending+t;
  const struct frames each=frames_of(page_bytes());
  uint32_t posted=0;
  for(;;){
    const struct mesh_pages p=stream_pages(*lane_posts(l->lane+k,way),*lane_posts(l->lane+k+1,way)),q=mesh_pages_sub(p,w->base);
    const uint32_t lane=k+page_lane(p);
    struct lane_state *s=l->lane+lane;
    if(way && s->posted>=s->got+s->window)break;
    const struct taken *e=walk_page(w,q);
    if(!e)break;
    const struct mesh_pages i=mesh_pages_sub(q,e->page);
    uint64_t at=(uint64_t)(way?l->page:ones)*M->pgsz;
    uint32_t row=MESH_ABSENT;
    if(e->whole){
      if(!way && !mesh_pages_lt(mesh_pages_add(e->base,i),mesh_pages_load(e->word)))break;
      const struct mesh_row r=mesh_row_after(e->row,i);
      if(row_at(r,&at))row=(uint32_t)r.r;
    }
    if(!way && !gate_holds(l->gate+lane,each))break;
    struct ibv_sge span=page_span(port,at);
    const struct mesh_queue *queue=l->provider.queues+lane;
    unit_told(x,way,page_lane(p),*lane_posts(s,way),row);
    int error;
    if(way){struct ibv_recv_wr request={.wr_id=lane,.sg_list=&span,.num_sge=1},*bad;error=queue->receive(queue->pair,&request,&bad);}
    else{
      struct ibv_send_wr posting={.wr_id=gate_post(l->gate+lane,lane,each),.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,
                                  .send_flags=IBV_SEND_SIGNALED},*bad;
      error=queue->send(queue->pair,&posting,&bad);
    }
    if(error){lane_lost(port,lane,error<0?-error:error);break;}
    if(way)s->posted++;
    else __atomic_store_n(&s->sent,s->sent+1,__ATOMIC_RELEASE);
    posted|=UINT32_C(1)<<lane;
  }
  return posted;
}
static __attribute__((always_inline)) inline uint32_t link_lane(const struct link *l,uint32_t qp_num){
  uint32_t k=0;
  while(k<MESH_LINK_LANES && l->qpn[k]!=qp_num)k++;
  return k;
}
static void link_failed(struct port *port,int64_t code,uint32_t domain){
  struct link *l=&port->link;
  const uint32_t served=atomic_exchange_explicit(&l->served,0,memory_order_acq_rel);
  if(served)port_log(port,"a completion failed (code %lld, domain %u): no lane served, the link toward node %u lost\n",(long long)code,
                     domain,port->peer);
  atomic_fetch_or_explicit(&l->broken,served,memory_order_release);
  wake();
}
static void walk_complete(struct port *port,int way,uint32_t moved,uint32_t posted){
  struct link *l=&port->link;
  uint32_t timing=0;
  for(uint32_t t=0;t<plan_streams(l->plan);t++){
    const uint32_t k=MESH_LANE_FIRST+2*t,lanes=UINT32_C(3)<<k;
    if(!((moved|posted)&lanes) || (way?!l->rnumber[t]:l->hold[t]!=HOLD_SERVE))continue;
    struct walk *w=way?l->landing+t:l->sending+t;
    struct mesh_stream *x=stream_of(port,t);
    walk_count(w,past(stream_through(*lane_done(l->lane+k,way),*lane_done(l->lane+k+1,way)),w->base),way?&x->landed:&x->sent);
    timing|=way?w->timed!=w->done:(posted&lanes)!=0;
  }
  const uint64_t at=timing?mach_absolute_time():0;
  for(uint32_t t=0;t<plan_streams(l->plan);t++){
    const uint32_t k=MESH_LANE_FIRST+2*t,lanes=UINT32_C(3)<<k;
    if(!((moved|posted)&lanes))continue;
    struct mesh_stream *x=stream_of(port,t);
    for(uint32_t c=0;c<2;c++)if(moved>>(k+c)&1)atomic_store_explicit(&x->posts[way].done[c],*lane_done(l->lane+k+c,way),memory_order_release);
    if((way?!l->rnumber[t]:l->hold[t]!=HOLD_SERVE) || !((way?moved:posted)&lanes))continue;
    struct walk *w=way?l->landing+t:l->sending+t;
    const uint64_t last=way?w->done:w->taken;
    for(;w->timed<last;w->timed++){
      const struct taken *e=w->ring+w->timed%MESH_STREAM_ENTRIES;
      if(!way && e->pages.n && !mesh_pages_lt(e->page,mesh_pages_sub(stream_pages(l->lane[k].sent,l->lane[k+1].sent),w->base)))break;
      stream_time(way?l->landed_at[t]:l->posted_at[t],w->timed,at);
    }
  }
}
static void port_pass(struct port *port,int way){
  struct link *l=&port->link;
  const uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  uint32_t posted=0,moved=0;
  for(uint32_t t=0;served>>MESH_LANE_FIRST && t<plan_streams(l->plan);t++){
    if(way)stream_receive_turn(port,t);
    else stream_turn_one(port,t);
    posted|=stream_post(port,t,served,way);
  }
  if(!served)return;
  const uint32_t live=atomic_load_explicit(&l->served,memory_order_acquire);
  struct ibv_wc done[16];
  const int count=l->provider.poll(way?l->provider.completion:l->provider.sent,16,done);
  if(count<0){link_failed(port,EIO,2+(uint32_t)way);return;}
  if(way && count)atomic_thread_fence(memory_order_acquire);
  for(int i=0;i<count;i++){
    const uint32_t lane=link_lane(l,done[i].qp_num);
    if(lane>=MESH_LINK_LANES || !(live>>lane&1))continue;
    uint64_t status_opcode;
    memcpy(&status_opcode,&done[i].status,sizeof status_opcode);
    if((uint32_t)status_opcode){link_failed(port,(uint32_t)status_opcode,2);return;}
    if(!way){gate_retire(l->gate+lane,done[i].wr_id);l->lane[lane].done++;}
    else if((status_opcode>>32)&IBV_WC_RECV)l->lane[lane].got++;
    else continue;
    moved|=UINT32_C(1)<<lane;
  }
  if(way)posted=0;
  if(moved || posted)walk_complete(port,way,moved,posted);
}
/* design/transport.md section 14, "A progress thread with nothing to do" */
static void *port_progress(void *argument){
  struct port *port=((struct progress *)argument)->port;
  const int way=((struct progress *)argument)->way;
  pthread_setname_np(way?"mesh.rdma.receive":"mesh.rdma.send");
  stack_t alternate={.ss_sp=malloc(SIGSTKSZ),.ss_size=SIGSTKSZ};
  if(alternate.ss_sp)sigaltstack(&alternate,NULL);
  atomic_store_explicit(&port->thread[way],pthread_self(),memory_order_release);
  while(!atomic_load_explicit(&port->closing,memory_order_acquire)){
    if(atomic_load_explicit(&port->link.served,memory_order_acquire)){port_pass(port,way);continue;}
    if(atomic_load_explicit(&faulted,memory_order_relaxed))wake();
    os_sync_wait_on_address_with_timeout(&port->link.served,0,sizeof port->link.served,OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                         OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
  }
  return NULL;
}
static void served_wake(struct port *port){
  os_sync_wake_by_address_all(&port->link.served,sizeof port->link.served,OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
static int port_threads(struct port *port){
  int error=0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  for(pthread_t created;port->threads<2 && !error;){
    port->progress[port->threads]=(struct progress){port,port->threads};
    if(!(error=pthread_create(&created,&attributes,port_progress,port->progress+port->threads)))port->threads++;
  }
  pthread_attr_destroy(&attributes);
  return error;
}

/* design/transport.md section 3; section 15, "The control socket" */
static void port_lost(struct port *port,int64_t code);
static void port_watch(struct port *port,int16_t filter){
  if(watch((uint64_t)port->control,filter,EV_ADD|(filter==EVFILT_WRITE?EV_ONESHOT:0),0,port->index))port_lost(port,errno);
}
static void port_flush(struct port *port){
  while(port->control>=0 && port->out_sent<port->out_length){
    ssize_t sent=send(port->control,port->out+port->out_sent,port->out_length-port->out_sent,0);
    if(sent<0 && (errno==EAGAIN || errno==EINTR)){
      if(!port->writing){port->writing=1;port_watch(port,EVFILT_WRITE);}
      return;
    }
    if(sent<=0){port_lost(port,sent?errno:EIO);return;}
    port->out_sent+=(size_t)sent;
  }
  port->out_sent=port->out_length=0;
}
static int port_coalesce(struct port *port,const struct message *said,const void *payload,uint32_t bytes){
  struct mesh_record fresh,queued;
  if(said->kind!=RECORD || bytes!=sizeof fresh)return 0;
  memcpy(&fresh,payload,sizeof fresh);
  for(size_t at=0;at+sizeof(struct message)<=port->out_length;){
    struct message header;
    memcpy(&header,port->out+at,sizeof header);
    if(at>=port->out_sent && header.kind==RECORD && header.bytes==sizeof queued){
      memcpy(&queued,port->out+at+sizeof header,sizeof queued);
      if(queued.node==fresh.node){
        if(fresh.boot>queued.boot || (fresh.boot==queued.boot && fresh.sequence>=queued.sequence))
          memcpy(port->out+at+sizeof header,&fresh,sizeof fresh);
        return 1;
      }
    }
    at+=sizeof header+header.bytes;
  }
  return 0;
}
static void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes){
  if(port->control<0 || (said.kind!=HELLO && !port->greeted) || port_coalesce(port,&said,payload,bytes))return;
  said.magic=XMAGIC;said.version=MESH_VERSION;said.node=M->node;said.boot=M->boot;said.block=(uint32_t)page_bytes();said.port=port->index;said.bytes=bytes;
  said.lid=port->lid;memcpy(said.gid,port->gid,16);
  size_t length=port->out_length+sizeof said+bytes;
  if(length>port->out_room){
    unsigned char *grown=realloc(port->out,length*2);
    if(!grown){port_lost(port,ENOMEM);return;}
    port->out=grown;port->out_room=length*2;
  }
  memcpy(port->out+port->out_length,&said,sizeof said);
  if(bytes)memcpy(port->out+port->out_length+sizeof said,payload,bytes);
  port->out_length=length;
  port_flush(port);
}

/* design/transport.md section 3, RECORD; section 15, "The map" */
static struct mesh_record *record_held(uint32_t node){
  for(uint64_t i=0;i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node==node)return mesh_records(M)+i;
  return NULL;
}
static void records_say(const struct port *from,const struct mesh_record *record,int held){
  for(uint32_t p=0;(!held || record_held(record->node)) && p<port_count;p++)
    if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},record,sizeof *record);
}
static uint64_t available_now(void){
  int level=0;size_t size=sizeof level;
  return sysctlbyname("kern.memorystatus_level",&level,&size,NULL,0) || level<0?0:(uint64_t)level*(memory/100);
}
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  struct known *at=NULL;
  if(known_count<MESH_KNOWN)at=known+known_count++;
  else for(uint32_t i=0;i<known_count;i++)if(!record_held(known[i].node) && (!at || known[i].heard<at->heard))at=known+i;
  if(at)*at=(struct known){.node=node};
  return at;
}
static int map_grow(uint64_t records){
  if(records<=map_room)return 0;
  uint32_t pages=(uint32_t)((records*2*sizeof(struct mesh_record)+M->pgsz-1)/M->pgsz),page=mesh_arena_alloc(&bridge,pages,1);
  if(page==MESH_ABSENT)return ENOMEM;
  memcpy(mesh_at(M,page),mesh_records(M),(size_t)atomic_load(&M->map_nodes)*sizeof(struct mesh_record));
  atomic_store(&M->map_off,(uint64_t)(mesh_at(M,page)-(unsigned char *)M));
  mesh_bridge_pages(M)->map=page;mesh_bridge_pages(M)->map_pages=pages;
  if(map_pages)mesh_bits_clear(mesh_arena_bits(M),map_page,map_pages);
  held_pages+=pages;held_pages-=map_pages;
  map_page=page;map_pages=pages;map_room=(uint32_t)((uint64_t)pages*M->pgsz/sizeof(struct mesh_record));
  return 0;
}
static void map_put(const struct mesh_record *record){
  const uint64_t n=atomic_load(&M->map_nodes);
  struct mesh_record *r=mesh_records(M);
  uint64_t at=0;
  while(at<n && r[at].node!=record->node)at++;
  const uint64_t count=at==n?n+1:n;
  struct mesh_record *next=malloc(count*sizeof *next);
  unsigned char *reached=calloc(count,1);
  uint32_t *order=malloc(count*sizeof *order),head=0,tail=0;
  if(!next || !reached || !order){free(next);free(reached);free(order);say("map: no memory to put node %u's record\n",record->node);return;}
  memcpy(next,r,(size_t)n*sizeof *r);
  memcpy(next+at,record,sizeof *record);
  for(uint64_t k=0;k<count;k++)if(next[k].node==M->node){reached[k]=1;order[tail++]=(uint32_t)k;}
  while(head<tail){
    uint32_t a=order[head++];double beta;
    for(uint64_t b=0;b<count;b++)if(!reached[b] && mesh_joined(next+a,next+b,&beta)){reached[b]=1;order[tail++]=(uint32_t)b;}
  }
  for(uint64_t k=0;k<count;k++)if(!reached[k] && !(at==n && k==at)){
    struct known *gone=known_at(next[k].node);
    say("node %u unreachable, heard %.1f s before\n",next[k].node,gone && gone->heard?(double)(now-gone->heard)/1e9:-1.0);
    if(gone)gone->gone=now;
  }
  uint64_t kept=0;
  for(uint64_t k=0;k<count;k++)if(reached[k])memmove(next+kept++,next+k,sizeof *next);
  if(kept!=n || memcmp(next,r,(size_t)n*sizeof *r)){
    mesh_notify(&M->map);
    if(map_grow(kept))say("map: no lease for %llu records; %llu kept\n",(unsigned long long)kept,(unsigned long long)map_room);
    if(kept>map_room)kept=map_room;
    memcpy(mesh_records(M),next,(size_t)kept*sizeof *next);
    atomic_store(&M->map_nodes,kept);
    mesh_notify(&M->map);
  }
  free(next);free(reached);free(order);
}
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  int same=a->node==b->node && a->reached==b->reached && a->boot==b->boot &&
    a->region==b->region && a->memory==b->memory && !memcmp(a->model,b->model,sizeof a->model);
  for(uint32_t k=0;same && k<a->reached && k<MESH_STREAMS;k++)
    same=a->reach[k].node==b->reach[k].node && a->reach[k].bandwidth==b->reach[k].bandwidth;
  return same;
}
static void record_put(struct mesh_record *held,const struct mesh_record *record){
  if(!held || !record_links_same(held,record)){map_put(record);return;}
  mesh_notify(&M->health);
  held->sequence=record->sequence;held->health=record->health;
  mesh_notify(&M->health);
}
static void record_changed(void){
  struct mesh_record own={.node=M->node,.boot=M->boot,.sequence=++sequence,.region=wire.length-held_pages*M->pgsz,.memory=memory,.health=health};
  memcpy(own.model,model,sizeof own.model);
  for(uint32_t p=0;p<port_count;p++){
    const struct port *port=ports+p;
    if(port->link.state!=LINK_UP || !plan_streams(port->link.plan))continue;
    uint32_t k=0;
    while(k<own.reached && own.reach[k].node!=port->peer)k++;
    if(k==own.reached){
      if(own.reached==MESH_STREAMS)continue;
      own.reach[own.reached++]=(struct mesh_reach){.node=port->peer};
    }
    own.reach[k].bandwidth+=port->bandwidth;
  }
  record_put(record_held(M->node),&own);
  records_say(NULL,&own,0);
}
static void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard={0};
  memcpy(&heard,payload,MIN((size_t)bytes,sizeof heard));
  if(!heard.node || heard.node==M->node)return;
  heard.reached=MIN(heard.reached,(uint32_t)MESH_STREAMS);
  heard.model[sizeof heard.model-1]=0;
  struct mesh_record *held=record_held(heard.node);
  struct known *k=known_at(heard.node);
  if(k && k->boot && (heard.boot<k->boot || (heard.boot==k->boot && heard.sequence<=k->sequence))){
    if(!held && heard.boot==k->boot && heard.sequence==k->sequence){map_put(&heard);records_say(from,&heard,1);}
    return;
  }
  if(held && (heard.boot<held->boot || (heard.boot==held->boot && heard.sequence<=held->sequence)))return;
  if(!held && k && k->gone)say("node %u back after %.1f s\n",heard.node,(double)(now-k->gone)/1e9);
  else if(!held && k && !k->heard)say("node %u joined\n",heard.node);
  if(k){k->heard=now;k->gone=0;k->boot=heard.boot;k->sequence=heard.sequence;}
  record_put(held,&heard);
  records_say(from,&heard,0);
}
static void link_set(struct port *port,uint32_t peer,uint32_t phase,int64_t code){
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  int changed=atomic_load(&link->peer)!=peer || atomic_load(&link->phase)!=phase;
  atomic_store_explicit(&link->peer,peer,memory_order_relaxed);atomic_store_explicit(&link->code,code,memory_order_relaxed);
  atomic_store_explicit(&link->phase,phase,memory_order_release);
  if(changed)record_changed();
}

/* design/transport.md section 14, "A process a port"; section 15, "The port's device" */
static int device_gid(struct mesh_device *device,const char *interface){
  enum { HELD=16, ENTRIES=256 };
  struct in6_addr held[HELD];
  union ibv_gid table[ENTRIES];
  int helds=0,chosen=-1,matched=0,linklocal=-1,first=-1;
  struct ifaddrs *all=NULL;
  if(!getifaddrs(&all)){
    for(const struct ifaddrs *a=all;a && helds<HELD;a=a->ifa_next){
      if(!a->ifa_addr || a->ifa_addr->sa_family!=AF_INET6 || strcmp(a->ifa_name,interface))continue;
      struct in6_addr x=((const struct sockaddr_in6 *)(const void *)a->ifa_addr)->sin6_addr;
      if(!IN6_IS_ADDR_LINKLOCAL(&x))continue;
      x.s6_addr[2]=x.s6_addr[3]=0;
      held[helds++]=x;
    }
    freeifaddrs(all);
  }
  const int entries=device->port.gid_tbl_len<ENTRIES?device->port.gid_tbl_len:ENTRIES;
  for(int i=0;i<entries;i++){
    int nonzero=0;
    if(ibv_query_gid(device->context,1,i,table+i))continue;
    for(int b=0;b<16;b++)nonzero|=table[i].raw[b];
    if(!nonzero)continue;
    if(first<0)first=i;
    if(linklocal<0 && table[i].raw[0]==0xfe && (table[i].raw[1]&0xc0)==0x80)linklocal=i;
    for(int h=0;h<helds && chosen<0;h++)if(!memcmp(table[i].raw,held[h].s6_addr,16)){chosen=i;matched=1;}
  }
  if(chosen<0)chosen=linklocal>=0?linklocal:first;
  if(chosen<0)return ENXIO;
  device->gid=table[chosen];device->gid_index=(uint8_t)chosen;device->gid_matched=(uint8_t)matched;
  return 0;
}
static int wire_map(struct wire_mapping *mapping,struct hdr *m,int file){
  mapping->length=(size_t)mesh_arena_pages(m)*m->pgsz;
  mapping->data=NULL;
  char *reserved=mmap(NULL,mapping->length+MESH_BANK,PROT_NONE,MAP_PRIVATE|MAP_ANON|MAP_NORESERVE,-1,0);
  if(reserved==MAP_FAILED)return -1;
  char *base=(char *)(((uintptr_t)reserved+MESH_BANK-1)&~(uintptr_t)(MESH_BANK-1));
  if(base>reserved)munmap(reserved,(size_t)(base-reserved));
  size_t tail=(size_t)((reserved+mapping->length+MESH_BANK)-(base+mapping->length));
  if(tail)munmap(base+mapping->length,tail);
  mapping->data=mmap(base,mapping->length,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,file,(off_t)m->data_off);
  if(mapping->data==MAP_FAILED){mapping->data=NULL;return -1;}
  return 0;
}
static int device_up(struct mesh_device *device,int *refused){
  *refused=0;
  if(!device->context){
    struct ibv_device **list=ibv_get_device_list(NULL);
    for(int i=0;list && list[i];i++)if(!strcmp(device->name,ibv_get_device_name(list[i]))){device->context=ibv_open_device(list[i]);break;}
    if(list)ibv_free_device_list(list);
    if(!device->context)return ENODEV;
  }
  if(ibv_query_port(device->context,1,&device->port))return errno?errno:EIO;
  if(device->port.state!=IBV_PORT_ACTIVE)return ENETDOWN;
  if(device->frame_capacity)return 0;
  struct ibv_device_attr capabilities;
  if(ibv_query_device(device->context,&capabilities))return errno?errno:EIO;
  if(!device->domain && !(device->domain=ibv_alloc_pd(device->context)))return errno?errno:EIO;
  const uint32_t extents=(uint32_t)((wire.length+MESH_EXTENT-1)/MESH_EXTENT);
  if((uint64_t)extents>(uint64_t)(capabilities.max_mr>0?capabilities.max_mr:0)){
    say("%s: the region of %zu bytes is %u extents of %zu bytes, past the device's max_mr %d: refused\n",device->name,
      wire.length,extents,MESH_EXTENT,capabilities.max_mr);
    *refused=EFBIG;return EFBIG;
  }
  if(!device->extent_mrs && !(device->extent_mrs=calloc(extents,sizeof *device->extent_mrs)))return ENOMEM;
  const uint64_t began=clock_ns();
  for(size_t e=0,first=0;e<extents;e++,first+=MESH_EXTENT){
    const size_t bytes=wire.length-first<MESH_EXTENT?wire.length-first:MESH_EXTENT;
    if(!device->extent_mrs[e] && !(device->extent_mrs[e]=ibv_reg_mr(device->domain,wire.data+first,bytes,IBV_ACCESS_LOCAL_WRITE))){
      const int error=errno?errno:EIO;
      uint32_t held=0;
      for(uint32_t k=0;k<extents;k++)held+=device->extent_mrs[k]!=NULL;
      say("%s: extent %zu of %u (%zu bytes at %zu) refused: %s; %u of the region's extents registered, max_mr %d\n",
        device->name,e,extents,bytes,first,strerror(error),held,capabilities.max_mr);
      *refused=error;return error;
    }
  }
  say("%s: region of %zu bytes (%u extents) registered in %.3f s\n",device->name,wire.length,extents,(double)(clock_ns()-began)/1e9);
  device->max_pairs=capabilities.max_qp>1?(uint32_t)capabilities.max_qp-1:0;
  uint32_t capacity=capabilities.max_qp_wr<QD?capabilities.max_qp_wr:QD;
  if(capabilities.max_cqe<=1 || !capacity)return EOPNOTSUPP;
  device->frame_capacity=capacity>=(uint32_t)capabilities.max_cqe?(uint32_t)capabilities.max_cqe-1:capacity;
  return 0;
}
static int pair_up(struct mesh_verbs *p,uint32_t plan){
  struct mesh_device *device=p->device;
  const uint32_t count=(uint32_t)__builtin_popcount(plan),entries=plan?32-(uint32_t)__builtin_clz(plan):0,frames=device->frame_capacity;
  if(device->pairs+count>device->max_pairs)return ENOSPC;
  if(!(p->queues=calloc(entries?entries:1,sizeof *p->queues)))return ENOMEM;
  p->psn=arc4random()&0xffffff;
  if(!p->completion && !(p->completion=ibv_create_cq(device->context,(int)(frames+1),NULL,NULL,0)))return errno?errno:EIO;
  if(!p->sent && !(p->sent=ibv_create_cq(device->context,(int)(frames+1),NULL,NULL,0)))return errno?errno:EIO;
  p->poll=p->completion->context->ops.poll_cq;
  for(uint32_t q=0;q<entries;q++){
    if(!(plan>>q&1))continue;
    struct mesh_queue *queue=&p->queues[q];
    struct ibv_qp_init_attr qi={.send_cq=p->sent,.recv_cq=p->completion,.qp_type=IBV_QPT_UC,
      .cap={.max_send_wr=frames,.max_recv_wr=frames,.max_send_sge=1,.max_recv_sge=1}},actual;
    if(!(queue->pair=ibv_create_qp(device->domain,&qi)))return errno?errno:EIO;
    device->pairs++;
    queue->send=queue->pair->context->ops.post_send;queue->receive=queue->pair->context->ops.post_recv;
    struct ibv_qp_attr queried,a={.qp_state=IBV_QPS_INIT,.port_num=1};
    if(ibv_query_qp(queue->pair,&queried,IBV_QP_CAP,&actual))return errno?errno:EIO;
    queue->receive_capacity=actual.cap.max_recv_wr;queue->send_capacity=actual.cap.max_send_wr;
    if(ibv_modify_qp(queue->pair,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS))return errno?errno:EIO;
  }
  return 0;
}
static uint64_t port_bandwidth(const struct ibv_port_attr *port){
  static const uint64_t speeds[256]={[1]=2500000000,[2]=5000000000,[4]=10000000000,[8]=10000000000,[16]=14000000000,[32]=25000000000,[64]=50000000000,[128]=100000000000};
  static const uint8_t widths[256]={[1]=1,[2]=4,[4]=8,[8]=12};
  return speeds[port->active_speed]*widths[port->active_width];
}
static void job_begin(struct link *l,int job){
  l->job=job;l->job_pairing=atomic_load_explicit(&l->pairing,memory_order_relaxed);l->job_at=now;
  atomic_store_explicit(&l->done,0,memory_order_relaxed);
}
static int link_lapsed(const struct link *l){ return atomic_load_explicit(&l->pairing,memory_order_relaxed)!=l->job_pairing?ECANCELED:0; }
static int link_make(struct port *port){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  struct ibv_port_attr attributes;
  int error=0;
  if(p->queues)return EALREADY;
  if(ibv_query_port(port->device.context,1,&attributes))return errno?errno:EIO;
  port->device.port=attributes;
  if((error=device_gid(&port->device,port->name)) || (error=link_lapsed(l)) || (error=pair_up(p,l->plan)))return error;
  l->share=(struct frames){(uint64_t)MIN(p->completion->cqe,p->sent->cqe)/MAX(plan_lanes(l->plan>>MESH_LANE_FIRST),1u)};
  return 0;
}
static uint64_t stream_wire(const struct port *port,uint32_t s){
  const uint32_t k=MESH_LANE_FIRST+2*s*MESH_STREAM_PAIRS;
  uint8_t end[2][20];
  memcpy(end[0],port->gid,16);memcpy(end[0]+16,&port->link.qpn[k],4);
  memcpy(end[1],port->heard.gid,16);memcpy(end[1]+16,&port->link.peer[k][0],4);
  const int first=memcmp(end[0],end[1],20)>0;
  uint64_t h=UINT64_C(14695981039346656037);
  for(int e=0;e<2;e++)
    for(int i=0;i<20;i++){h^=end[e^first][i];h*=UINT64_C(1099511628211);}
  return h?h:1;
}
static int link_connect(struct port *port){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  const uint32_t plan=l->plan;
  int error=0;
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1){
    struct ibv_qp_attr r={.qp_state=IBV_QPS_RTR,.path_mtu=IBV_MTU_4096,.rq_psn=l->peer[k][1],.dest_qp_num=l->peer[k][0],
      .ah_attr={.dlid=port->heard.lid,.port_num=1,.is_global=1,.grh={.hop_limit=1,.sgid_index=port->device.gid_index}}};
    memcpy(&r.ah_attr.grh.dgid,port->heard.gid,16);
    if(ibv_modify_qp(p->queues[k].pair,&r,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN))error=errno?errno:EIO;
  }
  if(!error)error=link_lapsed(l);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1){
    struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=(p->psn+k)&0xffffff};
    if(ibv_modify_qp(p->queues[k].pair,&t,IBV_QP_STATE|IBV_QP_SQ_PSN))error=errno?errno:EIO;
  }
  const struct frames share={MAX(l->share.n,1)},unit=frames_of(page_bytes()),ring={(uint64_t)MESH_LANE_UNITS*unit.n};
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++){
    l->qpn[k]=plan>>k&1?p->queues[k].pair->qp_num:UINT32_MAX;
    l->gate[k]=(struct send_gate){.capacity=plan>>k&1?frames_min(frames_min((struct frames){p->queues[k].send_capacity},share),ring):(struct frames){0}};
  }
  if(!error)error=link_lapsed(l);
  for(uint32_t k=MESH_LANE_FIRST;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1){
    const struct frames room=frames_min(l->share,(struct frames){p->queues[k].receive_capacity});
    l->lane[k]=(struct lane_state){.window=(uint32_t)MIN(MESH_LANE_UNITS,MAX(1,room.n/unit.n))};
    atomic_thread_fence(memory_order_release);
  }
  for(uint32_t s=0;s<plan_streams(plan) && !error;s++)l->wire[s]=stream_wire(port,s);
  return error;
}
static void streams_open(struct port *port,uint32_t node,uint64_t bandwidth){
  struct link *l=&port->link;
  for(uint32_t s=0;s<plan_streams(l->plan);s++){
    struct mesh_stream *x=stream_of(port,s);
    atomic_store_explicit(&x->owner,0,memory_order_release);
    stream_zero(x,1);
    l->posted_at[s]=stream_words(x->posted_at);l->landed_at[s]=stream_words(x->landed_at);
    for(uint32_t e=0;e<2*MESH_STREAM_TIMES;e++){atomic_store_explicit(l->posted_at[s]+e,0,memory_order_relaxed);atomic_store_explicit(l->landed_at[s]+e,0,memory_order_relaxed);}
    l->held[s]=l->number[s]=l->rnumber[s]=0;l->hold[s]=HOLD_NONE;
    atomic_store_explicit(&l->begun[s],0,memory_order_relaxed);atomic_store_explicit(&l->ended[s],0,memory_order_relaxed);
    atomic_store_explicit(&l->stopped[s],0,memory_order_relaxed);
    l->sending[s]=(struct walk){.ring=l->sending[s].ring};l->landing[s]=(struct walk){.ring=l->landing[s].ring};
    x->child_start=mesh_started(getpid());
    atomic_store_explicit(&x->child,(uint32_t)getpid(),memory_order_release);
    atomic_store_explicit(&x->peer,node,memory_order_relaxed);
    atomic_store_explicit(&x->bandwidth,bandwidth/MESH_LINK_STREAMS,memory_order_relaxed);
    atomic_store_explicit(&x->wire,l->wire[s],memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&x->live,1,memory_order_release);
  }
}
static void link_job(struct port *port,int job){
  struct link *l=&port->link;
  job_begin(l,job);
  dispatch_async(port->work,^{
    atomic_store_explicit(&port->thread[2],pthread_self(),memory_order_release);
    int result=link_lapsed(l);
    if(!result)result=job==LINK_JOB_MAKE?link_make(port):link_connect(port);
    if(!result && job==LINK_JOB_CONNECT)result=port_threads(port);
    l->result=result;
    atomic_store_explicit(&port->thread[2],NULL,memory_order_release);
    atomic_store_explicit(&l->done,1,memory_order_release);
    wake();
  });
}
static void device_open(struct port *port){
  port->job=1;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    int refused=0,result=device_up(&port->device,&refused);
    if(!result)result=device_gid(&port->device,port->name);
    if(!result && !port->device.gid_matched){
      const uint8_t *g=port->device.gid.raw;
      port_log(port,"no GID of %s is a link-local address %s holds; GID[%u] %02x%02x:%02x%02x:%02x%02x:%02x%02x:"
               "%02x%02x:%02x%02x:%02x%02x:%02x%02x used\n",port->device.name,port->name,port->device.gid_index,
               g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],g[8],g[9],g[10],g[11],g[12],g[13],g[14],g[15]);
    }
    port->result=result;port->refused=refused;
    atomic_store_explicit(&port->done,1,memory_order_release);
    wake();
  });
}

/* design/transport.md section 14, "A process a port" */
static int stream_lists(struct port *port);
static uint32_t own_lease(uint32_t *recorded,uint32_t pages){
  if(*recorded==MESH_ABSENT)*recorded=mesh_arena_alloc(&bridge,pages,M->block);
  if(*recorded!=MESH_ABSENT)held_pages+=pages;
  return *recorded;
}
static void message_send(int fd,struct child_message said){
  const char *at=(const char *)&said;
  size_t left=sizeof said;
  while(fd>=0 && left){
    const ssize_t sent=send(fd,at,left,0);
    if(sent<0 && errno==EINTR)continue;
    if(sent<=0)return;
    at+=sent;left-=(size_t)sent;
  }
}
static int message_read(int fd,unsigned char *in,uint32_t *held,struct child_message *heard){
  for(;;){
    const ssize_t got=recv(fd,in+*held,sizeof *heard-*held,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return 0;
    if(got<=0)return -1;
    if((*held+=(uint32_t)got)<sizeof *heard)continue;
    *held=0;memcpy(heard,in,sizeof *heard);
    return 1;
  }
}
static void child_job(struct port *port,int job){
  struct link *l=&port->link;
  job_begin(l,job);
  struct child_message said={.kind=job==LINK_JOB_MAKE?CHILD_MAKE:CHILD_CONNECT,.pairing=l->job_pairing,.page=l->page,.lanes=l->plan};
  if(job==LINK_JOB_CONNECT){
    memcpy(said.peer,l->peer,sizeof said.peer);said.lid=port->heard.lid;memcpy(said.gid,port->heard.gid,16);
    said.node=port->peer;said.bandwidth=port->bandwidth;
  }
  message_send(port->child_fd,said);
}
static void link_turn(struct port *port){
  struct link *l=&port->link;
  if(l->job){
    if(!atomic_load_explicit(&l->done,memory_order_acquire))return;
    const int job=l->job,result=l->result;
    l->job=0;
    if(l->job_pairing!=atomic_load_explicit(&l->pairing,memory_order_relaxed)){}
    else if(result){
      port_log(port,"lanes toward node %u not %s: %s; paired again with the port's next child\n",port->peer,
               job==LINK_JOB_MAKE?"made":"connected",strerror(result));
      port_lost(port,result);
      return;
    }
    else if(job==LINK_JOB_MAKE){
      struct message said={.kind=LINK,.count=plan_lanes(l->plan)};
      l->state=LINK_MADE;
      for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(l->plan>>k&1){said.pair[k][0]=l->qpn[k];said.pair[k][1]=(l->psn+k)&0xffffff;}
      port_say(port,said,NULL,0);
    }
    else{
      l->state=LINK_UP;
      port_log(port,"%u lanes up toward node %u (plan %#x: %u stream%s), each lane's share %llu frames\n",plan_lanes(l->plan),port->peer,
               l->plan,plan_streams(l->plan),plan_streams(l->plan)==1?"":"s",(unsigned long long)l->share.n);
      record_changed();
    }
  }
  const int held=atomic_load_explicit(&stalled,memory_order_relaxed) || now<port->quiet_at;
  if(!port->greeted)return;
  if(l->state==LINK_UP){
    const uint32_t broken=atomic_exchange_explicit(&l->broken,0,memory_order_acq_rel);
    const int lost=atomic_exchange_explicit(&l->lost,0,memory_order_relaxed);
    if(broken && lost==LINK_STREAMS_ENDED)
      port_log(port,"no stream toward node %u live, each holding ended with units outstanding: the link lost\n",port->peer);
    else if(broken)port_log(port,"lanes %#x toward node %u failed: the link lost\n",broken,port->peer);
    if(broken){port_lost(port,lost?lost:EIO);return;}
  }
  if(l->job || held)return;
  if(l->state==LINK_DOWN){
    if(l->page==MESH_ABSENT)l->page=own_lease(mesh_bridge_pages(M)->link+port->index,l->pages);
    if(l->page==MESH_ABSENT || (l->lists==MESH_ABSENT && stream_lists(port))){
      port_log(port,"lanes not made: the region holds no room for their blocks and rings; paired again on hearing the peer\n");
      port_lost(port,ENOMEM);
      return;
    }
    l->lanes=1;
    child_job(port,LINK_JOB_MAKE);
  }
  else if(l->state==LINK_MADE && l->heard){l->state=LINK_CONNECTING;child_job(port,LINK_JOB_CONNECT);}
}
static void ports_stalled(void){
  int any=0;
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    const struct link *l=&port->link;
    const int opening=port->job && !atomic_load_explicit(&port->done,memory_order_acquire) && now-port->job_at>=MESH_PLATFORM_NS;
    const int linking=l->job && !atomic_load_explicit(&l->done,memory_order_acquire) && now-l->job_at>=MESH_PLATFORM_NS;
    any|=opening||linking;
    if(port->stuck){
      if(!opening && !linking){port->stuck=0;if(!port->job)port->opening=1;}
      continue;
    }
    if(!opening && !linking)continue;
    port->stuck=1;
    port_log(port,"its %s out %.1f s: its port concluded around it (design/transport.md)\n",opening?"open's job":
             l->job==LINK_JOB_MAKE?"make job":"connect job",(double)(now-(opening?port->job_at:l->job_at))/1e9);
    port->open=0;
    port_lost(port,ETIMEDOUT);
  }
  atomic_store_explicit(&stalled,any,memory_order_relaxed);
}
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  struct link *l=&port->link;
  if(port->greeted){
    if(heard->kind==RECORD)record_heard(port,payload,heard->bytes);
    else if(heard->kind==FAREWELL){port->parted=1;port->farewell=heard->code;port_lost(port,ECONNRESET);}
    else if(heard->kind!=LINK || l->heard || (l->state!=LINK_DOWN && l->state!=LINK_MADE)){}
    else if(heard->count!=plan_lanes(l->plan)){
      port_log(port,"node %u's LINK names %u lanes, this end's plan %u (plan %#x): %s\n",port->peer,heard->count,plan_lanes(l->plan),
               l->plan,port->heard.count?"the two ends planned apart":"its bridge states no budget and makes every lane, "
               "and this port holds fewer: the link pairs once that node runs a bridge that states its budget");
      port_lost(port,EPROTO);
    }
    else{
      memcpy(l->peer,heard->pair,sizeof l->peer);
      port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
      l->heard=1;
      link_turn(port);
    }
    return;
  }
  if(heard->kind!=HELLO)return;
  if(heard->block!=page_bytes()){port_lost(port,EPROTO);return;}
  port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;port->greeted=1;
  const uint32_t theirs=hello_budget(heard->count);
  l->plan=link_plan(MIN(port->budget,theirs));
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
  atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
  port_log(port,"paired with node %u (queue pairs for the link: this port's %u, its %u%s; plan %#x, %u lanes, %u stream%s)\n",port->peer,
           port->budget,theirs,heard->count?"":", its bridge stating none",l->plan,plan_lanes(l->plan),plan_streams(l->plan),
           plan_streams(l->plan)==1?"":"s");
  link_set(port,port->peer,MESH_PAIRED,0);
  for(uint64_t i=0;port->greeted && i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node!=M->node)
    port_say(port,(struct message){.kind=RECORD},mesh_records(M)+i,sizeof(struct mesh_record));
  if(port->greeted)link_turn(port);
}
static void streams_cancel(const struct port *port){
  for(uint32_t p=0;p<port_count;p++){
    if((port && ports+p!=port) || ports[p].link.lists==MESH_ABSENT)continue;
    for(uint32_t s=0;s<MESH_LINK_STREAMS;s++){
      struct mesh_stream *x=stream_of(ports+p,s);
      atomic_store_explicit(&x->live,0,memory_order_release);
      mesh_pages_store(&x->sent,MESH_PAGES_LOST);
      mesh_pages_store(&x->landed,MESH_PAGES_LOST);
    }
  }
}
static void port_lost(struct port *port,int64_t code){
  struct link *l=&port->link;
  quiet_until(&port->quiet_at,now+MESH_PLATFORM_NS);
  atomic_fetch_add_explicit(&l->pairing,1,memory_order_relaxed);l->state=LINK_DOWN;l->heard=0;
  atomic_store_explicit(&l->served,0,memory_order_release);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
  atomic_store_explicit(&l->lost,0,memory_order_relaxed);
  if(port->child){
    message_send(port->child_fd,(struct child_message){.kind=CHILD_HALT,.pairing=atomic_load(&l->pairing)});
    if(l->lanes && !stop){port->recycle=1;port->open=0;}
  }
  if(port->control>=0){close(port->control);port->control=-1;}
  if(port->greeted)port->next=now;
  if(port->greeted||port->connecting){
    char reason[96];farewell_text(port->farewell,reason,sizeof reason);
    if(stop)port_log(port,"closed toward node %u: this bridge is stopping\n",port->peer);
    else if(port->peer && port->parted)port_log(port,"node %u parted, its bridge stopping (%s): code %lld\n",port->peer,reason,(long long)code);
    else if(port->peer)port_log(port,"node %u lost: code %lld\n",port->peer,(long long)code);
    else port_log(port,"lost: code %lld\n",(long long)code);
  }
  streams_cancel(port);
  port->greeted=port->parted=port->connecting=port->writing=0;port->farewell=0;port->peer=0;port->held=0;port->out_length=port->out_sent=0;
  link_set(port,0,port->open?MESH_PAIRING:MESH_STOPPED,code);
}
static void port_greet(struct port *port){
  int on=1,flags=fcntl(port->control,F_GETFL);
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on)){
    port_lost(port,errno?errno:EIO);return;
  }
  port_watch(port,EVFILT_READ);
  port_say(port,(struct message){.kind=HELLO,.count=1+port->budget},NULL,0);
}
static struct port *port_at(uint32_t interface){
  for(uint32_t p=0;interface && p<port_count;p++)if(ports[p].interface==interface)return ports+p;
  return NULL;
}
static void hello_join(const struct port *port){
  struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=port->interface};
  if(port->interface)setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
}
static void hello_send(const struct port *port,const struct sockaddr_in6 *to){
  const struct message said={.magic=XMAGIC,.version=MESH_VERSION,.kind=HELLO,.node=M->node,.boot=M->boot,.block=(uint32_t)page_bytes(),.port=port->index};
  sendto(hello,&said,sizeof said,0,(const struct sockaddr *)to,sizeof *to);
}
static int child_spawn(struct port *port);
static void port_open(struct port *port){
  if(port->job || port->stuck || !port->opening || port->control>=0 || now<port->quiet_at || port->recycle || now<port->hold_to ||
     now<port->retry_at || atomic_load_explicit(&stalled,memory_order_relaxed))return;
  if(!port->child && child_spawn(port))return;
  port->opening=0;
  port->interface=if_nametoindex(port->name);
  hello_join(port);
  port->job=1;port->job_at=now;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  message_send(port->child_fd,(struct child_message){.kind=CHILD_OPEN});
}
static void port_opened(struct port *port){
  const int result=port->result;
  port->job=0;
  if(result){
    if(port->refused){
      port->refusals++;
      const uint64_t hold=backoff(port->refusals-1);
      port->retry_at=now+hold*UINT64_C(1000000000);
      port_log(port,"the region's registration refused (%s), %u in a row: opened again at a hello heard on it from %llu s on, or at "
               "its link's coming on\n",strerror(port->refused),port->refusals,(unsigned long long)hold);
    }
    port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);
    return;
  }
  port->open=1;port->refusals=0;
  if(port->control>=0)return;
  link_set(port,0,MESH_PAIRING,0);
  port->next=now;
}
static void port_hello(struct port *port){
  if(!port->open || port->control>=0 || port->job || now<port->next)return;
  port->next=now+MESH_PAUSE_NS;
  setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_IF,&port->interface,sizeof port->interface);
  struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=everyone,.sin6_scope_id=port->interface};
  hello_send(port,&to);
}
static void hello_heard(void){
  struct message heard;struct sockaddr_in6 from;socklen_t length=sizeof from;
  while(recvfrom(hello,&heard,sizeof heard,0,(struct sockaddr *)&from,&length)==(ssize_t)sizeof heard){
    struct port *port=port_at(from.sin6_scope_id);
    length=sizeof from;
    if(!port || heard.magic!=XMAGIC || heard.kind!=HELLO || heard.node==M->node || port->control>=0)continue;
    if(!port->open){if(!port->job)port->opening=1;continue;}
    if(M->node>heard.node){hello_send(port,&from);continue;}
    from.sin6_port=htons((uint16_t)atoi(MESH_PORT));
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) ||
       (connect(port->control,(struct sockaddr *)&from,sizeof from) && errno!=EINPROGRESS)){port_lost(port,errno);continue;}
    port->peer=heard.node;port->connecting=1;
    port_watch(port,EVFILT_WRITE);
  }
}
static void control_heard(struct port *port){
  for(;;){
    size_t need=sizeof(struct message);
    if(port->held>=need){
      const struct message *header=(const void *)port->in;
      if(header->magic!=XMAGIC || header->bytes>(UINT32_C(64)<<20)){port_lost(port,EPROTO);return;}
      need+=header->bytes;
    }
    if(need>port->in_room){
      unsigned char *grown=realloc(port->in,need);
      if(!grown){port_lost(port,ENOMEM);return;}
      port->in=grown;port->in_room=need;
    }
    if(port->held<need){
      ssize_t got=recv(port->control,port->in+port->held,need-port->held,0);
      if(got<0 && (errno==EAGAIN || errno==EINTR))return;
      if(got<=0){port_lost(port,got?errno:ENETRESET);return;}
      port->held+=(size_t)got;
      if(port->held<need || need==sizeof(struct message))continue;
    }
    struct message heard;memcpy(&heard,port->in,sizeof heard);
    port->held=0;
    port_heard(port,&heard,port->in+sizeof heard);
    if(port->control<0)return;
  }
}
/* design/transport.md section 2, the bridge's own pages; section 10, duds */
static uint64_t page_round(uint64_t bytes){ return (bytes+page_bytes()-1)/page_bytes()*page_bytes(); }
static int stream_lists(struct port *port){
  const uint64_t lists=page_round(2*(uint64_t)MESH_STREAM_ENTRIES*sizeof(struct mesh_entry)),times=page_round((uint64_t)MESH_STREAM_TIMES*16),
                 rows=page_round(4*(uint64_t)MESH_LANE_UNITS*sizeof(uint32_t));
  const uint32_t page=own_lease(mesh_bridge_pages(M)->lists+port->index,(uint32_t)(MESH_LINK_STREAMS*(lists+2*times+rows)/M->pgsz));
  if(page==MESH_ABSENT)return ENOMEM;
  record_changed();
  uint64_t at=M->data_off+(uint64_t)page*M->pgsz;
  for(uint32_t s=0;s<MESH_LINK_STREAMS;s++){
    struct mesh_stream *x=stream_of(port,s);
    x->lists=at;at+=lists;
    x->posted_at=at;at+=times;
    x->landed_at=at;at+=times;
    x->rows=at;at+=rows;
  }
  port->link.lists=page;
  return 0;
}
static void link_reserve(void){
  struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  for(uint32_t p=0;p<MESH_PORTS;p++)link_page[p]=own_lease(own->link+p,M->block);
  if((ones=own_lease(&own->ones,M->block))==MESH_ABSENT)die("the duds' block");
  memset(mesh_at(M,ones),0xff,(size_t)page_bytes());
}
static void streams_let_go(void){
  for(uint32_t p=0;p<port_count;p++)
    for(uint32_t s=0;s<MESH_LINK_STREAMS;s++){
      struct mesh_stream *x=stream_of(ports+p,s);
      uint64_t owner=atomic_load_explicit(&x->owner,memory_order_acquire);
      if(owner && !(owner&MESH_STREAM_GIVEN) && !owned(owner))atomic_compare_exchange_strong(&x->owner,&owner,owner|MESH_STREAM_GIVEN);
    }
}
/* design/transport.md section 14, "Plug to link, at the hardware's pace" */
static void port_named(struct port *port,uint32_t index,const char *device){
  *port=(struct port){.index=index,.control=-1,.child_fd=-1};
  port->link.provider.device=&port->device;port->link.lists=MESH_ABSENT;port->link.pages=M->block;
  snprintf(port->device.name,sizeof port->device.name,"%s",device);snprintf(port->name,sizeof port->name,"%s",device+5);
}
static void ports_find(void){
  if(finding)return;
  finding=1;
  dispatch_async(finder,^{
    struct ibv_device **list=ibv_get_device_list(NULL);
    uint32_t n=0;
    for(int i=0;list && list[i] && n<MESH_PORTS;i++)if(!strncmp(ibv_get_device_name(list[i]),"rdma_",5))
      snprintf(devices[n++],sizeof devices[0],"%s",ibv_get_device_name(list[i]));
    if(list)ibv_free_device_list(list);
    device_count=n;
    atomic_store_explicit(&found,1,memory_order_release);
    wake();
  });
}
static void ports_found(void){
  finding=0;atomic_store_explicit(&found,0,memory_order_relaxed);
  for(uint32_t i=0;i<device_count && port_count<MESH_PORTS;i++){
    uint32_t known=0;
    for(uint32_t p=0;p<port_count;p++)known|=!strcmp(ports[p].device.name,devices[i]);
    if(known)continue;
    struct port *port=ports+port_count;
    port_named(port,port_count,devices[i]);
    port->opening=1;port->link.page=link_page[port_count];
    if(!strncmp(port->name,"en",2)){port->family=APPLE_IF_FAM_ETHERNET;port->unit=(uint32_t)strtoul(port->name+2,NULL,10);}
    __atomic_store_n(&port_count,port_count+1,__ATOMIC_SEQ_CST);
    quiet_until(&port->quiet_at,atomic_load(&exit_at));
    link_set(port,0,MESH_STOPPED,0);
  }
}
static void network_heard(int stopping){
  union mesh_network_event event;
  int unknown=0;
  while(recv(network,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    const uint32_t code=event.header.event_code;
    char name[IFNAMSIZ+16];
    snprintf(name,sizeof name,"%.*s%u",(int)sizeof interface->if_name,interface->if_name,interface->if_unit);
    int known=0;
    for(uint32_t p=0;p<port_count;p++)if(!strcmp(name,ports[p].name)){
      known=1;
      if(code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED){ports[p].opening=1;ports[p].retry_at=0;}
      else if(code!=KEV_DL_LINK_OFF && code!=KEV_DL_IF_DETACHED){}
      else if(stopping)quiet_until(&ports[p].quiet_at,now+MESH_PLATFORM_NS);
      else{ports[p].open=0;port_lost(ports+p,ENETDOWN);}
    }
    unknown|=!known && (code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED);
  }
  if(unknown && !stopping)find_wanted=1;
}
static void addresses_heard(void){
  union { struct kern_event_msg header; unsigned char bytes[512]; } event;
  while(recv(addresses,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    for(uint32_t p=0;p<port_count;p++)if(interface->if_family==ports[p].family && interface->if_unit==ports[p].unit){
      hello_join(ports+p);
      ports[p].next=now;
    }
  }
}
/* design/transport.md sections 7 and 8; section 15, "Terminations and faults" */
static struct port *peer_port(io_service_t peer){
  CFTypeRef rid=IORegistryEntryCreateCFProperty(peer,CFSTR("RID"),kCFAllocatorDefault,0);
  io_iterator_t interfaces=IO_OBJECT_NULL;
  struct port *found=NULL;
  if(rid && IOServiceGetMatchingServices(kIOMainPortDefault,IOServiceMatching("AppleThunderboltRDMAInterface"),&interfaces)==KERN_SUCCESS){
    for(io_service_t interface;(interface=IOIteratorNext(interfaces));IOObjectRelease(interface)){
      CFTypeRef its=found?NULL:IORegistryEntryCreateCFProperty(interface,CFSTR("RID"),kCFAllocatorDefault,0);
      io_name_t name;
      if(its && CFEqual(its,rid) && IORegistryEntryGetName(interface,name)==KERN_SUCCESS){
        const uint32_t count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);
        for(uint32_t p=0;p<count;p++)if(!strcmp(ports[p].device.name,name))found=ports+p;
      }
      if(its)CFRelease(its);
    }
    IOObjectRelease(interfaces);
  }
  if(rid)CFRelease(rid);
  return found;
}
static void peer_terminated(void *unused,io_iterator_t iterator){
  (void)unused;
  io_service_t services[MESH_PORTS];
  uint32_t count=0;
  for(io_service_t service;(service=IOIteratorNext(iterator));){
    quiet_all(clock_ns()+MESH_PLATFORM_NS);
    atomic_fetch_add(&terminations,1);
    wake();
    if(count<MESH_PORTS)services[count++]=service;
    else IOObjectRelease(service);
  }
  for(uint32_t k=0;k<count;k++){
    struct port *port=peer_port(services[k]);
    IOObjectRelease(services[k]);
    if(!port)continue;
    port_halt(port,&terminated);
    wake();
  }
}
static void peer_arrived(void *unused,io_iterator_t iterator){
  (void)unused;
  for(io_service_t service;(service=IOIteratorNext(iterator));IOObjectRelease(service)){
    struct port *port=peer_port(service);
    if(port)atomic_fetch_or(&arrived,UINT32_C(1)<<port->index);
    else atomic_store(&arrived_elsewhere,1);
    wake();
  }
}
static void terminations_watch(int arrivals){
  IONotificationPortRef notifications=IONotificationPortCreate(kIOMainPortDefault);
  io_iterator_t iterator=IO_OBJECT_NULL,came=IO_OBJECT_NULL;
  kern_return_t error=KERN_FAILURE,heard=KERN_FAILURE;
  if(notifications){
    IONotificationPortSetDispatchQueue(notifications,dispatch_queue_create("mesh.terminations",DISPATCH_QUEUE_SERIAL));
    error=IOServiceAddMatchingNotification(notifications,kIOTerminatedNotification,IOServiceMatching("AppleThunderboltRDMAPeerInterface"),
                                           peer_terminated,NULL,&iterator);
    if(arrivals)heard=IOServiceAddMatchingNotification(notifications,kIOFirstMatchNotification,
                                                      IOServiceMatching("AppleThunderboltRDMAPeerInterface"),peer_arrived,NULL,&came);
  }
  if(arrivals && heard)say("bridge: Thunderbolt RDMA peer interfaces' arrivals not heard (%#x): a port whose device was not yet "
                           "present at its link's coming on opens at the next hello heard on it\n",(unsigned)heard);
  if(arrivals && !heard)peer_arrived(NULL,came);
  if(error){say("bridge: Thunderbolt RDMA peer interfaces' terminations not heard (%#x): a link's going off is its first sign\n",(unsigned)error);return;}
  peer_terminated(NULL,iterator);
}
static void fault_heard(int number,siginfo_t *info,void *context){
  const uintptr_t pc=__darwin_arm_thread_state64_get_pc(((ucontext_t *)context)->uc_mcontext->__ss);
  const uintptr_t page=trunc_page((uintptr_t)info->si_addr);
  const pthread_t self=pthread_self();
  struct port *port=NULL;
  for(uint32_t p=0,count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);p<count;p++)
    for(int t=0;t<3;t++)if(atomic_load_explicit(&ports[p].thread[t],memory_order_acquire)==self)port=ports+p;
  if(port && pc>=provider[0] && pc<provider[1]){
    void *got=mmap((void *)page,vm_page_size,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
    if(got!=MAP_FAILED && got!=(void *)page)munmap(got,vm_page_size);
    if(got==(void *)page || atomic_load(&faulted)>>port->index&1){port_halt(port,&faulted);return;}
    if(atomic_exchange(&retried[port->index],page)!=page)return;
  }
  struct sigaction fallback={.sa_handler=SIG_DFL};
  sigaction(number,&fallback,NULL);
}
static void faults_install(void){
  for(uint32_t i=0;i<_dyld_image_count();i++){
    const char *image=_dyld_get_image_name(i);
    unsigned long size=0;
    const uint8_t *text=image && strstr(image,"/libthunderboltrdma.")?
      getsegmentdata((const void *)_dyld_get_image_header(i),"__TEXT",&size):NULL;
    if(text){provider[0]=(uintptr_t)text;provider[1]=(uintptr_t)text+size;}
  }
  struct sigaction faults={.sa_sigaction=fault_heard,.sa_flags=SA_SIGINFO|SA_ONSTACK};
  sigemptyset(&faults.sa_mask);
  if(sigaction(SIGSEGV,&faults,NULL) || sigaction(SIGBUS,&faults,NULL))die("bridge faults");
}
static void terminations_turn(void){
  const uint32_t heard=atomic_load(&terminations),lost=atomic_exchange(&terminated,0),fresh=heard-terminations_told;
  const uint32_t faults=atomic_exchange(&faulted,0);
  if(fresh){
    const uint64_t until=atomic_load(&exit_at);
    say("bridge node %u: %u Thunderbolt RDMA peer interface%s terminating: no port job and no exit for %.1f s "
        "(design/transport.md)\n",M->node,fresh,fresh==1?"":"s",(double)(until>now?until-now:0)/1e9);
    terminations_told=heard;
  }
  for(uint32_t p=0;p<port_count;p++)if(lost>>p&1){
    port_log(ports+p,"its peer interface terminating: its link lost\n");
    port_lost(ports+p,ETIMEDOUT);
  }
  for(uint32_t p=0;p<port_count;p++)if(faults>>p&1){
    port_log(ports+p,"a fault of its provider's doorbell absorbed: its link lost (design/transport.md)\n");
    ports[p].open=0;port_lost(ports+p,ENETDOWN);
  }
  const uint32_t came=atomic_exchange(&arrived,0);
  for(uint32_t p=0;p<port_count;p++)if(came>>p&1 && ports[p].control<0){ports[p].opening=1;ports[p].retry_at=0;}
  if(atomic_exchange(&arrived_elsewhere,0))find_wanted=1;
}

/* design/transport.md section 14, "A process a port"; section 15, "The children" */
static void child_reaped(struct port *port);
static pid_t child_elsewhere(const struct port *port){
  static pid_t pids[16384];
  static char *arguments;
  static size_t room;
  int most=0,argmax[2]={CTL_KERN,KERN_ARGMAX};
  size_t length=sizeof most;
  if(!arguments && (sysctl(argmax,2,&most,&length,NULL,0) || most<=0 || !(arguments=malloc(room=(size_t)most))))return 0;
  const int count=proc_listallpids(pids,(int)sizeof pids);
  for(int i=0;i<count && i<(int)(sizeof pids/sizeof *pids);i++){
    const pid_t pid=pids[i];
    int own=pid==getpid();
    for(uint32_t p=0;p<port_count;p++)own|=pid==ports[p].child;
    if(own)continue;
    int mib[3]={CTL_KERN,KERN_PROCARGS2,pid},argc=0;
    size_t size=room;
    if(sysctl(mib,3,arguments,&size,NULL,0) || size<sizeof argc)continue;
    memcpy(&argc,arguments,sizeof argc);
    const char *at=arguments+sizeof argc,*end=arguments+size,*argv[4]={0};
    at+=strnlen(at,(size_t)(end-at));
    while(at<end && !*at)at++;
    for(int a=0;a<argc && a<4 && at<end;a++){argv[a]=at;at+=strnlen(at,(size_t)(end-at))+1;}
    if(argc>=4 && argv[1] && argv[3] && !strcmp(argv[1],"--child") && !strcmp(argv[3],port->device.name))return pid;
  }
  return 0;
}
static int child_spawn(struct port *port){
  int pair[2],on=1;
  if(port->elsewhere)return EBUSY;
  const pid_t elsewhere=child_elsewhere(port);
  if(elsewhere){
    if(watch((uint64_t)elsewhere,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,TAG_CHILD|port->index)){if(errno==ESRCH)wake();return EBUSY;}
    port_log(port,"process %d, a child of its device from an earlier launch, holds its queue pairs until its exit; its own child spawned "
             "after it\n",(int)elsewhere);
    port->elsewhere=elsewhere;
    return EBUSY;
  }
  struct mesh_holder *holders=mesh_bridge_pages(M)->child;
  uint32_t entry=0;
  while(entry<2*MESH_PORTS && mesh_holds((pid_t)holders[entry].pid,holders[entry].start))entry++;
  if(entry==2*MESH_PORTS){port->hold_to=now+MESH_PAUSE_NS;return EBUSY;}
  int error=socketpair(AF_UNIX,SOCK_STREAM,0,pair)?errno?errno:EIO:0;
  pid_t pid=0;
  if(!error){
    setsockopt(pair[0],SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on);
    char index[16],page[16],version[16];
    snprintf(index,sizeof index,"%u",port->index);snprintf(page,sizeof page,"%u",ones);snprintf(version,sizeof version,"%u",MESH_CHILD_VERSION);
    char *arguments[]={self_path,"--child",version,port->device.name,index,page,NULL};
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions,0,"/dev/null",O_RDONLY,0);
    posix_spawn_file_actions_addopen(&actions,1,"/dev/null",O_WRONLY,0);
    posix_spawn_file_actions_addinherit_np(&actions,2);
    posix_spawn_file_actions_adddup2(&actions,pair[1],3);
    posix_spawnattr_init(&attributes);
    posix_spawnattr_setflags(&attributes,POSIX_SPAWN_CLOEXEC_DEFAULT|POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attributes,0);
    error=posix_spawn(&pid,self_path,&actions,&attributes,arguments,*_NSGetEnviron());
    posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attributes);
    close(pair[1]);
    if(error)close(pair[0]);
  }
  if(error){
    port_log(port,"its child not spawned: %s\n",strerror(error));
    port->hold_to=now+MESH_PAUSE_NS;
    return error;
  }
  port->child=pid;port->child_fd=pair[0];port->child_at=now;port->child_held=0;port->ending=0;
  holders[entry]=(struct mesh_holder){(uint32_t)pid,0,mesh_started(pid)};
  port->holder=entry;
  port_log(port,"its child %d spawned\n",(int)pid);
  if(watch((uint64_t)pair[0],EVFILT_READ,EV_ADD,0,TAG_CHILD|port->index)){close(port->child_fd);port->child_fd=-1;}
  if(watch((uint64_t)pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,TAG_CHILD|port->index)){child_reaped(port);return ECHILD;}
  return 0;
}
static void child_heard(struct port *port){
  struct link *l=&port->link;
  const uint32_t bit=UINT32_C(1)<<port->index;
  struct child_message heard;
  int got;
  while((got=message_read(port->child_fd,port->child_in,&port->child_held,&heard))>0){
    const int current=heard.pairing==atomic_load_explicit(&l->pairing,memory_order_relaxed);
    if(heard.kind==CHILD_OPENED && port->job){
      port->result=heard.result;port->refused=heard.refused;
      if(!heard.result){port->lid=heard.lid;memcpy(port->gid,heard.gid,16);port->bandwidth=heard.bandwidth;port->budget=heard.lanes;}
      atomic_store_explicit(&port->done,1,memory_order_release);
    }
    else if(heard.kind==CHILD_MADE || heard.kind==CHILD_CONNECTED){
      if(heard.kind==CHILD_CONNECTED && !current)streams_cancel(port);
      if(l->job!=(heard.kind==CHILD_MADE?LINK_JOB_MAKE:LINK_JOB_CONNECT) || heard.pairing!=l->job_pairing)continue;
      l->result=heard.result;
      if(heard.kind==CHILD_MADE && !heard.result){
        memcpy(l->qpn,heard.qpn,sizeof l->qpn);l->psn=heard.psn;l->share=(struct frames){heard.share};
        port->lid=heard.lid;memcpy(port->gid,heard.gid,16);
      }
      atomic_store_explicit(&l->done,1,memory_order_release);
    }
    else if(heard.kind==CHILD_BROKEN && current && l->state==LINK_UP){
      if(heard.result)atomic_store_explicit(&l->lost,heard.result,memory_order_relaxed);
      atomic_fetch_or(&l->broken,heard.broken);
    }
    else if(heard.kind==CHILD_FAULTED)atomic_fetch_or(&faulted,bit);
    else if(heard.kind==CHILD_TERMINATED)atomic_fetch_or(&terminated,bit);
  }
  if(got<0){close(port->child_fd);port->child_fd=-1;}
}
static void child_turn(struct port *port){
  if(!port->recycle || !port->child || port->ending || port->job || port->link.job || now<atomic_load(&port->quiet_at) ||
     now<atomic_load(&exit_at) || atomic_load_explicit(&stalled,memory_order_relaxed))return;
  message_send(port->child_fd,(struct child_message){.kind=CHILD_END});
  port->ending=1;
  port_log(port,"its child %d asked to end, its lanes released at its exit (design/transport.md section 14)\n",(int)port->child);
}
static void child_reaped(struct port *port){
  struct link *l=&port->link;
  const pid_t pid=port->child;
  int status=0;
  if(port->child_fd>=0)child_heard(port);
  if(port->child_fd>=0){close(port->child_fd);port->child_fd=-1;}
  while(waitpid(pid,&status,0)<0 && errno==EINTR){}
  port->child=0;
  if(port->holder<2*MESH_PORTS && mesh_bridge_pages(M)->child[port->holder].pid==(uint32_t)pid)
    mesh_bridge_pages(M)->child[port->holder]=(struct mesh_holder){0};
  const int asked=port->ending;
  port->ending=port->recycle=0;
  if(asked){port->aborts=0;port_log(port,"its child %d ended\n",(int)pid);}
  else{
    port->aborts=now-port->child_at<MESH_CHILD_SETTLE_NS?port->aborts+1:0;
    const uint64_t hold=port->aborts>1?backoff(port->aborts-2):0;
    port->hold_to=now+hold*UINT64_C(1000000000);
    port_log(port,"its child %d ended unbidden (%s %d); its next spawned in %llu s\n",(int)pid,WIFSIGNALED(status)?"signal":"status",
             WIFSIGNALED(status)?WTERMSIG(status):WEXITSTATUS(status),(unsigned long long)hold);
    port->open=0;
    port_lost(port,ECHILD);
  }
  if(port->job){port->result=ECHILD;port->refused=0;atomic_store_explicit(&port->done,1,memory_order_release);}
  if(l->job){l->result=ECHILD;atomic_store_explicit(&l->done,1,memory_order_release);}
  streams_cancel(port);
  unreleased=1;
  l->state=LINK_DOWN;l->lanes=l->heard=0;l->share=(struct frames){0};l->plan=0;
  memset(l->qpn,0,sizeof l->qpn);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
  port->opening=1;
}
/* design/transport.md section 2, the client slots; section 14, "A holding's end" */
static void clients_turn(int liveness){
  const pid_t self=getpid();
  int changed=0;
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    uint64_t owner=atomic_load(&client->owner);
    if(owner && (pid_t)(uint32_t)owner==self)continue;
    if(owner && owner!=owners[s] && !(owner&MESH_RETIRING) && watch((uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,s)<0 &&
       errno==ESRCH){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    changed|=owner!=owners[s];owners[s]=owner;
  }
  unreleased|=changed;
  if(unreleased || liveness)unreleased=mesh_retired_release(M,0)!=0;
  streams_let_go();
  uint64_t busy=0;
  for(uint32_t p=0;p<port_count*MESH_LINK_STREAMS;p++)
    busy+=(atomic_load_explicit(&mesh_stream_at(M,p)->owner,memory_order_relaxed)&~MESH_STREAM_GIVEN)!=0;
  if(busy)atomic_store_explicit(&M->idle_since,0,memory_order_relaxed);
  else if(!atomic_load_explicit(&M->idle_since,memory_order_relaxed))atomic_store_explicit(&M->idle_since,clock_gettime_nsec_np(CLOCK_REALTIME),memory_order_relaxed);
  atomic_store_explicit(&M->busy,busy,memory_order_release);
}
static void *control_watch(void *unused){
  (void)unused;
  while(!stop){
    uint64_t seen=atomic_load_explicit(&M->control,memory_order_acquire);
    wake();
    os_sync_wait_on_address(&M->control,seen,sizeof M->control,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
  }
  return NULL;
}
static const char *configuration_read(const char *path,char *name,size_t room,uint64_t *region_bytes){
  FILE *file=fopen(path,"rb");
  if(!file)return strerror(errno);
  unsigned char bytes[65536];
  const size_t length=fread(bytes,1,sizeof bytes,file);
  const int whole=feof(file);
  fclose(file);
  if(!whole)return "not a configuration property list (too long)";
  CFDataRef data=CFDataCreate(NULL,bytes,(CFIndex)length);
  CFPropertyListRef list=data?CFPropertyListCreateWithData(NULL,data,kCFPropertyListImmutable,NULL,NULL):NULL;
  if(data)CFRelease(data);
  const char *refused=NULL;
  CFTypeRef named=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("name")):NULL;
  CFTypeRef sized=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("region_bytes")):NULL;
  int64_t value=0;
  if(!list || CFGetTypeID(list)!=CFDictionaryGetTypeID())refused="not a property list of a configuration (rdma/bridge_configuration.py writes it)";
  else if(!named || CFGetTypeID(named)!=CFStringGetTypeID() || !CFStringGetCString(named,name,(CFIndex)room,kCFStringEncodingUTF8))refused="its name is missing";
  else if(!sized || CFGetTypeID(sized)!=CFNumberGetTypeID() || !CFNumberGetValue(sized,kCFNumberSInt64Type,&value) || value<=0)
    refused="its region_bytes, a positive number of bytes, is missing";
  if(list)CFRelease(list);
  *region_bytes=(uint64_t)value;
  return refused;
}

/* design/transport.md section 14, "A process a port"; section 15, "The children" */
static __attribute__((noreturn)) void child_leave(struct port *port,int asked){
  port_halt(port,NULL);
  atomic_store_explicit(&port->closing,1,memory_order_release);
  served_wake(port);
  const uint64_t start=clock_ns(),bound=start+(asked?1:2)*MESH_PLATFORM_NS;
  for(;;){
    const uint64_t at=clock_ns(),windows=MAX(atomic_load(&exit_at),atomic_load(&port->quiet_at));
    const uint64_t until=MIN(bound,MAX(asked?start:start+MESH_PLATFORM_NS,windows));
    if(at>=until)break;
    struct timespec rest={(time_t)((until-at)/1000000000u),(long)((until-at)%1000000000u)};
    nanosleep(&rest,NULL);
  }
  exit(0);
}
static void child_command(struct port *port,const struct child_message *heard){
  struct link *l=&port->link;
  if(heard->kind==CHILD_OPEN){if(!port->job)device_open(port);return;}
  if(heard->kind==CHILD_HALT){port_halt(port,NULL);child_halted=0;return;}
  if(heard->kind==CHILD_END)child_leave(port,1);
  if(heard->kind!=CHILD_MAKE && heard->kind!=CHILD_CONNECT)return;
  const int job=heard->kind==CHILD_MAKE?LINK_JOB_MAKE:LINK_JOB_CONNECT;
  int refused=l->job?EBUSY:child_halted || atomic_load(&faulted) || atomic_load(&terminated)?ECANCELED:0;
  if(!refused && job==LINK_JOB_CONNECT){
    memcpy(l->peer,heard->peer,sizeof l->peer);port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
    port->peer=heard->node;child_bandwidth=heard->bandwidth;
    for(uint32_t s=0;s<MESH_LINK_STREAMS && !refused;s++)
      if((!l->sending[s].ring && !(l->sending[s].ring=calloc(MESH_STREAM_ENTRIES,sizeof(struct taken)))) ||
         (!l->landing[s].ring && !(l->landing[s].ring=calloc(MESH_STREAM_ENTRIES,sizeof(struct taken)))))refused=ENOMEM;
  }
  if(refused){
    message_send(3,(struct child_message){.kind=job==LINK_JOB_MAKE?CHILD_MADE:CHILD_CONNECTED,.pairing=heard->pairing,.result=refused});
    return;
  }
  child_asked=heard->pairing;
  if(job==LINK_JOB_MAKE){l->page=heard->page;l->plan=heard->lanes;}
  link_job(port,job);
}
static void said_address(struct port *port,struct child_message *said){
  port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
  said->lid=port->lid;memcpy(said->gid,port->gid,16);
}
static void child_jobs(struct port *port){
  struct link *l=&port->link;
  if(port->job && atomic_load_explicit(&port->done,memory_order_acquire)){
    port->job=0;
    struct child_message said={.kind=CHILD_OPENED,.result=port->result,.refused=port->refused};
    if(!port->result){said_address(port,&said);said.bandwidth=port_bandwidth(&port->device.port);said.lanes=port->device.max_pairs;}
    message_send(3,said);
  }
  if(!l->job || !atomic_load_explicit(&l->done,memory_order_acquire))return;
  const int job=l->job;
  l->job=0;
  struct child_message said={.kind=job==LINK_JOB_MAKE?CHILD_MADE:CHILD_CONNECTED,.pairing=child_asked,.result=l->result};
  if(!said.result && l->job_pairing!=atomic_load(&l->pairing))said.result=ECANCELED;
  if(!said.result && job==LINK_JOB_MAKE){
    for(uint32_t k=0;k<MESH_LINK_LANES;k++)said.qpn[k]=l->plan>>k&1?l->provider.queues[k].pair->qp_num:UINT32_MAX;
    said.psn=l->provider.psn;said.share=(uint32_t)l->share.n;
    said_address(port,&said);
  }
  else if(!said.result){
    streams_open(port,port->peer,child_bandwidth);
    atomic_store(&l->served,l->plan);
    if(atomic_load(&l->pairing)!=l->job_pairing){atomic_store(&l->served,0);said.result=ECANCELED;}
    served_wake(port);
  }
  message_send(3,said);
}
static int child_main(int argc,char **argv){
  signal(SIGPIPE,SIG_IGN);
  const unsigned long index=argc>=6?strtoul(argv[4],NULL,10):MESH_PORTS;
  if(index>=MESH_PORTS){say("child of %s: port %lu is past the bridge's, or its command line short: it ends\n",argc>3?argv[3]:"?",index);return 1;}
  faults_install();
  const int fd=shm_open(MESH_NAME,O_RDWR);
  struct stat shape;
  if(fd<0 || fstat(fd,&shape))die("child: its bridge's region");
  M=mmap(NULL,(size_t)shape.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(M==MAP_FAILED || M->magic!=MESH_MAGIC || M->version!=MESH_LAYOUT || atomic_load(&M->bridge_pid)!=(uint64_t)getppid())
    die("child: its bridge's region is not its parent's");
  if(wire_map(&wire,M,fd))die("child: transport page aliases");
  close(fd);
  struct port *port=ports;
  port_named(port,(uint32_t)index,argv[3]);
  port->work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL);port->link.page=MESH_ABSENT;
  ones=(uint32_t)strtoul(argv[5],NULL,10);
  __atomic_store_n(&port_count,1,__ATOMIC_SEQ_CST);
  if((events=kqueue())<0 || watch(0,EVFILT_USER,EV_ADD|EV_CLEAR,0,UINT64_MAX) || watch(3,EVFILT_READ,EV_ADD,0,UINT64_MAX))die("child events");
  terminations_watch(0);
  for(;;){
    now=clock_ns();
    child_jobs(port);
    const uint32_t broken=atomic_exchange_explicit(&port->link.broken,0,memory_order_acq_rel);
    if(broken)message_send(3,(struct child_message){.kind=CHILD_BROKEN,.pairing=child_asked,.broken=broken,
                                                   .result=atomic_exchange_explicit(&port->link.lost,0,memory_order_relaxed)});
    if(atomic_exchange(&faulted,0)){child_halted=1;message_send(3,(struct child_message){.kind=CHILD_FAULTED});}
    if(atomic_exchange(&terminated,0)){child_halted=1;message_send(3,(struct child_message){.kind=CHILD_TERMINATED});}
    struct kevent64_s ready[4];
    struct timespec tick={(time_t)(MESH_PAUSE_NS/1000000000u),(long)(MESH_PAUSE_NS%1000000000u)};
    const int count=kevent64(events,NULL,0,ready,4,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++)if(ready[i].filter==EVFILT_READ && ready[i].ident==3){
      struct child_message heard;
      int got;
      while((got=message_read(3,child_in,&child_got,&heard))>0)child_command(port,&heard);
      if(got<0)child_leave(port,0);
    }
  }
}

/* design/transport.md section 2; section 15, "The region" */
static int region_open(const struct hdr *geometry,int *out){
  int fd=shm_open(MESH_NAME,O_RDWR,0);
  if(fd<0 && errno!=ENOENT){say("bridge: region %s not opened (%s): this bridge ends without touching it\n",MESH_NAME,strerror(errno));exit(1);}
  if(fd>=0){
    struct stat shape;
    if(fstat(fd,&shape)){say("bridge: region %s not read (%s): this bridge ends without touching it\n",MESH_NAME,strerror(errno));exit(1);}
    const size_t size=(size_t)shape.st_size;
    struct hdr *old=size>=sizeof(struct hdr)?mmap(NULL,size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0):NULL;
    if(old==MAP_FAILED){say("bridge: region %s not mapped (%s): this bridge ends without touching it\n",MESH_NAME,strerror(errno));exit(1);}
    if(old && old->magic==MESH_MAGIC){
      if(__atomic_load_n(&old->version,__ATOMIC_ACQUIRE)==MESH_LAYOUT && !memcmp(&old->pgsz,&geometry->pgsz,3*sizeof old->pgsz) &&
         !memcmp(&old->streams,&geometry->streams,offsetof(struct hdr,boot)-offsetof(struct hdr,streams))){M=old;*out=fd;return 1;}
      __atomic_store_n(&old->version,0,__ATOMIC_SEQ_CST);
      say("bridge: region %s is expired (another layout, or released): this bridge ends, and its heartbeat unlinks it once no "
          "holder of it lives and opens a bridge again\n",MESH_NAME);
      boots_clear();
      exit(EX_TEMPFAIL);
    }
    if(old)munmap(old,size);
    close(fd);
    shm_unlink(MESH_NAME);
  }
  if((fd=shm_open(MESH_NAME,O_CREAT|O_EXCL|O_RDWR,MESH_MODE))<0)die("shm");
  if(ftruncate(fd,(off_t)geometry->length))die("ftruncate");
  fchmod(fd,MESH_MODE);
  M=mmap(NULL,(size_t)geometry->length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  if(M==MAP_FAILED)die("mmap");
  *M=*geometry;M->version=MESH_LAYOUT;
  for(uint32_t r=0;r<mesh_rows(M);r++)atomic_store_explicit(&mesh_page(M)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  for(uint32_t p=0;p<MESH_PORTS;p++)own->link[p]=own->lists[p]=MESH_ABSENT;
  own->ones=own->map=MESH_ABSENT;own->map_pages=0;
  *out=fd;
  return 0;
}
static void region_adopt(void){
  for(uint32_t s=0;s<MESH_STREAMS;s++){
    struct mesh_stream *x=mesh_stream_at(M,s);
    atomic_store(&x->live,0);atomic_store(&x->peer,0);atomic_store(&x->wire,0);
    mesh_pages_store(&x->sent,MESH_PAGES_LOST);mesh_pages_store(&x->landed,MESH_PAGES_LOST);
  }
  struct mesh_link_info *links=mesh_links(M);
  for(uint32_t p=0;p<MESH_PORTS;p++){
    atomic_store(&links[p].peer,0);atomic_store(&links[p].phase,MESH_UNKNOWN);atomic_store(&links[p].code,0);atomic_store(&links[p].bandwidth,0);
  }
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    const uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
    if(owner&MESH_RETIRING)mesh_retire(M,owner);
  }
  mesh_retired_release(M,0);
  if(atomic_load(&M->map)&1)atomic_fetch_add(&M->map,1);
  if(atomic_load(&M->health)&1)atomic_fetch_add(&M->health,1);
  const struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  if(own->map!=MESH_ABSENT){
    map_page=own->map;map_pages=own->map_pages;map_room=(uint32_t)((uint64_t)map_pages*M->pgsz/sizeof(struct mesh_record));
    held_pages+=map_pages;
    atomic_store(&M->map_nodes,0);
    atomic_store(&M->map_off,(uint64_t)(mesh_at(M,map_page)-(unsigned char *)M));
  }
  atomic_store(&M->busy,0);atomic_store(&M->stopping,0);
}
/* design/transport.md section 15, "Windows" */
static inline void due_at(uint64_t *due,uint64_t at){ if(at>now && at<*due)*due=at; }
static uint64_t turn_due(uint64_t liveness){
  uint64_t due=MIN(liveness,now+MESH_PAUSE_NS);
  due_at(&due,health_sent+MESH_HEALTH_NS);
  if(boots_set)due_at(&due,boots_since+BOOTS_SETTLE_NS);
  if(find_wanted)due_at(&due,windows_end());
  due_at(&due,atomic_load(&exit_at));
  for(uint32_t p=0;p<port_count;p++){
    const struct port *port=ports+p;
    due_at(&due,atomic_load(&port->quiet_at));due_at(&due,port->hold_to);
    if(port->opening)due_at(&due,port->retry_at);
    if(port->open && port->control<0)due_at(&due,port->next);
    if(port->job)due_at(&due,port->job_at+MESH_PLATFORM_NS);
    if(port->link.job)due_at(&due,port->link.job_at+MESH_PLATFORM_NS);
  }
  return due;
}
int main(int argc,char **argv){
  if(argc>=3 && !strcmp(argv[1],"--child"))return child_main(argc,argv);
  if(argc!=2)die("usage: mesh-flow CONFIGURATION: the canonical property list bin/mesh-user.sh writes from the node's bridge configuration document (etc/bridge/README.md); the bridge has no default and does not start without one");
  char name[256];uint64_t region_bytes;
  const char *refused=configuration_read(argv[1],name,sizeof name,&region_bytes);
  if(refused){say("bridge configuration %s refused: %s\n",argv[1],refused);exit(1);}
  uint32_t room=sizeof self_path;
  char resolved[PATH_MAX];
  if(_NSGetExecutablePath(self_path,&room) || !realpath(self_path,resolved))die("the bridge's own image");
  snprintf(self_path,sizeof self_path,"%s",resolved);
  static const int stopping[]={SIGINT,SIGTERM,SIGHUP,SIGQUIT,SIGUSR1,SIGUSR2,SIGALRM,SIGVTALRM,SIGPROF,SIGXCPU,SIGXFSZ};
  if((events=kqueue())<0)die("bridge events");
  for(size_t i=0;i<sizeof stopping/sizeof *stopping;i++){
    if(watch((uint64_t)stopping[i],EVFILT_SIGNAL,EV_ADD,0,UINT64_MAX))die("bridge events");
    signal(stopping[i],SIG_IGN);
  }
  signal(SIGPIPE,SIG_IGN);
  boots_hold(argv[1]);
  uuid_t uuid;struct timespec bound={5,0};
  if(gethostuuid(uuid,&bound)){say("host identity: %s; this bridge's node is drawn at random\n",strerror(errno));arc4random_buf(uuid,sizeof uuid);}
  uint32_t node=0;
  for(int i=0;i<16;i++)node=(node*0x01000193u)^uuid[i];
  node=(node&0x7fffffffu)?node&0x7fffffffu:1;
  const uint16_t service=(uint16_t)atoi(MESH_PORT);
  const uint32_t pg=(uint32_t)getpagesize();
  if((uint64_t)pg+4096>16773120)die("transport chunk exceeds native request capacity");
  if(region_bytes<pg || region_bytes/pg>UINT32_MAX){say("bridge configuration %s refused: its region of %llu bytes is not between one page and 2^32 pages\n",name,(unsigned long long)region_bytes);exit(1);}
  struct hdr geometry={0};
  mesh_layout(&geometry,pg,1,(uint32_t)(region_bytes/pg));
  int on=1;
  struct sockaddr_in6 any={.sin6_len=sizeof any,.sin6_family=AF_INET6,.sin6_port=htons(service),.sin6_addr=IN6ADDR_ANY_INIT};
  listener=socket(AF_INET6,SOCK_STREAM,0);hello=socket(AF_INET6,SOCK_DGRAM,0);network=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
  struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS},inet6={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_INET6_SUBCLASS};
  unsigned loop=0;
  if(listener<0 || hello<0 || network<0 || setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) ||
     setsockopt(hello,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_LOOP,&loop,sizeof loop) ||
     bind(listener,(struct sockaddr *)&any,sizeof any) || listen(listener,SOMAXCONN) || bind(hello,(struct sockaddr *)&any,sizeof any) ||
     fcntl(listener,F_SETFL,O_NONBLOCK) || fcntl(hello,F_SETFL,O_NONBLOCK) || fcntl(network,F_SETFL,O_NONBLOCK) || ioctl(network,SIOCSKEVFILT,&filter))
    die("bridge sockets");
  if((addresses=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT))<0 || fcntl(addresses,F_SETFL,O_NONBLOCK) || ioctl(addresses,SIOCSKEVFILT,&inet6)){
    say("bridge: the kernel's IPv6 address events not heard (%s): a port's hello after its address comes waits its pace\n",strerror(errno));
    if(addresses>=0)close(addresses);
    addresses=-1;
  }
  int fd=-1;
  const int adopted=region_open(&geometry,&fd);
  if(adopted)region_adopt();
  M->node=node;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);M->idle_since=M->boot;
  say("bridge node %u: region %s %s\n",node,MESH_NAME,adopted?"adopted from the bridge before this one (its clients' leases stand)":"made new");
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  bridge.M=M;
  finder=dispatch_queue_create("mesh.ports",DISPATCH_QUEUE_SERIAL);
  size_t size=sizeof memory;
  if(sysctlbyname("hw.memsize",&memory,&size,NULL,0))memory=0;
  size=sizeof model;
  if(sysctlbyname("hw.model",model,&size,NULL,0))model[0]=0;
  model[sizeof model-1]=0;
  struct timeval booted;size=sizeof booted;
  if(!sysctlbyname("kern.boottime",&booted,&size,NULL,0))health.booted=(uint64_t)booted.tv_sec*1000000000u+(uint64_t)booted.tv_usec*1000u;
  health.available=available_now();
  record_changed();
  if(!atomic_load(&M->map_nodes))die("the map's lease");
  link_reserve();
  int watched[4]={listener,hello,network,addresses};
  for(int i=0;i<4 && watched[i]>=0;i++)if(watch((uint64_t)watched[i],EVFILT_READ,EV_ADD,0,UINT64_MAX))die("bridge events");
  if(watch(0,EVFILT_USER,EV_ADD|EV_CLEAR,0,UINT64_MAX))die("bridge events");
  mesh_bridge_pages(M)->bridge_start=mesh_started(getpid());
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();M->magic=MESH_MAGIC;
  pthread_t watcher;
  if(pthread_create(&watcher,NULL,control_watch,NULL))die("control watcher");
  terminations_watch(1);
  mach_timespec_t quiet={(unsigned)(MESH_PLATFORM_NS/1000000000u),(clock_res_t)(MESH_PLATFORM_NS%1000000000u)};
  const kern_return_t busy=IOKitWaitQuiet(kIOMainPortDefault,&quiet);
  if(busy)say("bridge node %u: IOKit's registry not quiet within %.1f s (%#x): its devices read as they stand\n",node,
              (double)MESH_PLATFORM_NS/1e9,(unsigned)busy);
  ports_find();
  say("bridge node %u: configuration %s, region %llu bytes, arena %llu pages, rows %u\n",node,name,
      (unsigned long long)region_bytes,(unsigned long long)mesh_arena_pages(M),M->rows);
  uint64_t liveness=0;
  while(!stop){
    now=clock_ns();
    if(atomic_load_explicit(&found,memory_order_acquire))ports_found();
    terminations_turn();
    ports_stalled();
    if(find_wanted && !finding && !atomic_load_explicit(&stalled,memory_order_relaxed) && now>=windows_end()){find_wanted=0;ports_find();}
    for(uint32_t p=0;p<port_count;p++){
      struct port *port=ports+p;
      if(port->job && atomic_load_explicit(&port->done,memory_order_acquire))port_opened(port);
      port_open(port);
      port_hello(port);
      link_turn(port);
      child_turn(port);
    }
    if(now-health_sent>=MESH_HEALTH_NS){health_sent=now;health.available=available_now();record_changed();}
    int alive=now>=liveness;
    if(alive){
      liveness=now+MESH_PAUSE_NS;
      const struct mesh_record *held=record_held(M->node);
      struct mesh_record own;
      if(held){memcpy(&own,held,sizeof own);records_say(NULL,&own,0);}
    }
    clients_turn(alive);
    struct kevent64_s ready[16];
    const uint64_t due=turn_due(liveness),at=clock_ns(),rest=due>at?due-at:0;
    struct timespec tick={(time_t)(rest/1000000000u),(long)(rest%1000000000u)};
    int count=kevent64(events,NULL,0,ready,16,0,&tick);
    now=clock_ns();
    if(boots_set && now-boots_since>=BOOTS_SETTLE_NS)boots_clear();
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL){stop=1;stop_reason=(int64_t)e->ident;}
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_CHILD)){
        struct port *port=(e->udata&0xff)<port_count?ports+(e->udata&0xff):NULL;
        if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->child)child_reaped(port);
        else if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->elsewhere)port->elsewhere=0;
        else if(port && e->filter==EVFILT_READ && (int)e->ident==port->child_fd)child_heard(port);
      }
      else if(e->filter==EVFILT_PROC){
        for(uint32_t s=0;s<MESH_CLIENTS;s++){
          uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
          if(owner && !(owner&MESH_RETIRING) && (uint32_t)owner==(uint32_t)e->ident)mesh_retire(M,owner);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)listener){
        struct sockaddr_in6 from;socklen_t size=sizeof from;int accepted;
        while((accepted=accept(listener,(struct sockaddr *)&from,&size))>=0){
          struct port *port=port_at(from.sin6_scope_id);size=sizeof from;
          if(!port || port->control>=0 || !port->open){close(accepted);continue;}
          port->control=accepted;
          port_greet(port);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)hello)hello_heard();
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard(0);
      else if(e->filter==EVFILT_READ && addresses>=0 && e->ident==(uint64_t)addresses)addresses_heard();
      else if(e->udata<port_count && (int)e->ident==ports[e->udata].control){
        struct port *port=ports+e->udata;
        if(e->filter==EVFILT_WRITE && port->connecting){
          int error=0;socklen_t size=sizeof error;
          if(getsockopt(port->control,SOL_SOCKET,SO_ERROR,&error,&size))error=errno;
          if(error)port_lost(port,error);
          else {port->connecting=0;port_greet(port);}
        }
        else if(e->filter==EVFILT_WRITE){port->writing=0;port_flush(port);}
        else control_heard(port);
      }
    }
  }
  char why[96];farewell_text(stop_reason,why,sizeof why);
  say("bridge node %u: stopping (%s)\n",node,why);
  boots_clear();
  for(uint32_t p=0;p<port_count;p++){
    message_send(ports[p].child_fd,(struct child_message){.kind=CHILD_HALT});
    port_say(ports+p,(struct message){.kind=FAREWELL,.code=stop_reason},NULL,0);
    if(ports[p].control>=0){close(ports[p].control);ports[p].control=-1;}
  }
  streams_cancel(NULL);
  atomic_store(&M->bridge_pid,0);
  mesh_notify(&M->control);
  const uint64_t held_to=clock_ns()+MESH_PLATFORM_NS;
  for(int told=0;;){
    now=clock_ns();
    const uint64_t windows=windows_end(),latest=MIN(windows,held_to);
    if(latest<=now)break;
    if(!told++)say("bridge node %u: its exit held %.1f s, to its quiet windows' end (design/transport.md)\n",node,(double)(latest-now)/1e9);
    struct pollfd heard={.fd=network,.events=POLLIN};
    if(poll(&heard,1,(int)((latest-now+999999)/1000000))>0){now=clock_ns();network_heard(1);}
  }
  for(uint32_t p=0;p<port_count;p++)message_send(ports[p].child_fd,(struct child_message){.kind=CHILD_END});
  const int exits=kqueue();
  for(uint32_t p=0;exits>=0 && p<port_count;p++)if(ports[p].child){
    struct kevent64_s watch;EV_SET64(&watch,(uint64_t)ports[p].child,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,p,0,0);
    kevent64(exits,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  }
  for(const uint64_t until=clock_ns()+MESH_PLATFORM_NS;;){
    uint32_t left=0;
    for(uint32_t p=0;p<port_count;p++)if(ports[p].child){
      if(!waitpid(ports[p].child,NULL,WNOHANG))left++;
      else ports[p].child=0;
    }
    const uint64_t at=clock_ns();
    if(!left || at>=until)break;
    struct kevent64_s ended;
    struct timespec rest={(time_t)((until-at)/1000000000u),(long)((until-at)%1000000000u)};
    if(exits<0 || (kevent64(exits,NULL,0,&ended,1,0,&rest)<0 && errno!=EINTR))poll(NULL,0,(int)((until-at+999999)/1000000));
  }
  return 0;
}
