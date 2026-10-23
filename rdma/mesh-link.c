#include "mesh-bridge.h"
/* design/transport.md section 15 */
static _Atomic sig_atomic_t stop;
static struct hdr *M;
static struct wire_mapping wire;
static struct port ports[MESH_PORTS];
static uint32_t port_count,device_count,ones=MESH_ABSENT,link_page[MESH_PORTS],pool_page[MESH_PORTS],map_page,map_pages,
                known_count,known_room,child_asked,names_held;
static int events=-1,listener=-1,hello=-1,network=-1,addresses=-1,unreleased,finding,find_wanted,child_halted;
static uint64_t now,sequence,memory,held_pages,find_at,owners[MESH_CLIENTS];
static int64_t stop_reason;
static struct mesh_ctx bridge={.fd=-1};
static dispatch_queue_t finder;
static _Atomic int stalled,found,arrived_elsewhere;
static _Atomic uint32_t terminated,arrived,faulted;
static _Atomic int job_parked,naming;
static uintptr_t provider[2];
static _Atomic uintptr_t mapped,job_thread;
static _Atomic uint64_t exit_at;
static char devices[MESH_PORTS][32],self_path[PATH_MAX],model[MESH_MODEL_BYTES];
static struct mesh_record *known;
static uint64_t (**known_name)[2],(*names)[2];
static struct mesh_health health;
static const struct in6_addr everyone={{{0xff,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}};
_Static_assert(MESH_PORTS<=32,"a bit a port");

static uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static uint32_t fnv(const void *bytes,size_t n){ uint32_t h=0;for(const unsigned char *b=bytes;n--;b++)h=(h*0x01000193u)^*b;return h; }
static void say(const char *format,...){ va_list arguments;va_start(arguments,format);mesh_say("",format,arguments);va_end(arguments); }
static void port_log(const struct port *port,const char *format,...){
  char prefix[64];snprintf(prefix,sizeof prefix,"link %u (%s): ",port->index,port->name);
  va_list arguments;va_start(arguments,format);mesh_say(prefix,format,arguments);va_end(arguments);
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

/* design/transport.md section 14, "The queue"; section 15, "The data path" */
static uint64_t page_bytes(void){ return (uint64_t)M->block*M->pgsz; }
static void lanes_lost(struct port *port,uint32_t lanes,int64_t code){
  struct link *l=&port->link;
  const uint32_t served=atomic_fetch_and_explicit(&l->served,~lanes,memory_order_acq_rel)&lanes;
  if(served)port_log(port,"lanes %#x failed (code %lld): the link toward node %u lost\n",served,(long long)code,port->peer);
  atomic_fetch_or_explicit(&l->broken,served,memory_order_release);
  wake();
}
static int row_at(struct mesh_row r,uint64_t *at){
  const uint64_t a=atomic_load_explicit(&mesh_page(M)[r.r].address,memory_order_relaxed),arena=(uint64_t)mesh_arena_pages(M)*M->pgsz;
  if(a<M->data_off || a-M->data_off>arena-page_bytes())return 0;
  *at=a-M->data_off;
  return 1;
}
static struct ibv_sge span_of(const struct port *port,uint64_t at,uint64_t bytes){
  return (struct ibv_sge){.addr=(uintptr_t)wire.data+at,.length=(uint32_t)bytes,.lkey=port->device.extent_mrs[at>>port->device.extent_shift]->lkey};
}
static struct ibv_sge page_span(const struct port *port,uint64_t at){ return span_of(port,at,page_bytes()-at%page_bytes()); }
static const _Atomic uint64_t *region_word(uint64_t offset){
  return offset && offset<=M->length-8 && !(offset%8)?(const _Atomic uint64_t *)((char *)M+offset):NULL;
}
static uint64_t link_at(const struct link *l){ return (uint64_t)l->page*M->pgsz; }
static int owned(uint64_t owner){ return owner && !(owner&MESH_RETIRING); }
static struct mesh_plan *queue_first(const struct mesh_client *slot){ return mesh_plan_at(M,atomic_load_explicit(&slot->queue,memory_order_acquire)); }
static struct mesh_plan *queue_next(const struct mesh_plan *t){ return mesh_plan_at(M,atomic_load_explicit(&t->next,memory_order_acquire)); }
static uint32_t k32(uint64_t k){ return (uint32_t)k; }
/* design/transport.md section 14, "Walks" */
static void walk_mark(uint32_t who,int way,int begin){
  atomic_fetch_add_explicit(&mesh_bridge_pages(M)->walking[who][way],1,begin?memory_order_seq_cst:memory_order_release);
  if(begin)atomic_thread_fence(memory_order_seq_cst);
}
/* design/transport.md section 14, "A registration" and "Sending" */
static int key_order(const uint64_t *x,const uint64_t *y,int words){
  for(int i=0;i<words;i++)if(x[i]!=y[i])return x[i]<y[i]?-1:1;
  return 0;
}
static int named_order(const void *a,const void *b){ return key_order(((const struct named *)a)->name,((const struct named *)b)->name,2); }
static void routes_free(struct walker *w){ free(w->name);free(w->node);free(w->hop);free(w->joined);free(w->relaying);free(w->path);free(w->paths); }
static void routes_fresh(struct walker *w){
  const uint64_t serial=atomic_load_explicit(&M->map,memory_order_acquire);
  if(serial==w->serial)return;
  struct mesh_record *r=NULL;
  uint64_t (*named)[2]=NULL;
  const int count=mesh_map_copy(M,&r,&named);
  if(count<0)return;
  const uint32_t n=(uint32_t)count;
  uint32_t names=0;
  for(uint32_t i=0;named && i<n;i++)names+=r[i].names;
  struct named *name=malloc((names+1)*sizeof *name);
  uint32_t *node=malloc((n+1)*sizeof *node),*hop=calloc(n+1,sizeof *hop),*relaying=calloc(n+1,sizeof *relaying),
           *path=malloc(((size_t)n*n+1)*sizeof *path),*paths=calloc(n+1,sizeof *paths);
  struct joined *joined=calloc(n+1,sizeof *joined);
  for(uint32_t a=0,at=0;name && a<n;a++)for(uint32_t j=0;named && j<r[a].names;j++,at++)name[at]=(struct named){{named[at][0],named[at][1]},r[a].node,0};
  struct mesh_whole whole;
  const int error=mesh_whole_make(r,n,&whole);
  struct walker made={.serial=serial,.name=name,.names=names,.nodes=n,.node=node,.hop=hop,.joined=joined,.relaying=relaying,.path=path,.paths=paths};
  if(error || !name || !node || !hop || !joined || !relaying || !path || !paths){free(r);free(named);routes_free(&made);mesh_whole_free(&whole);return;}
  uint32_t self=n;
  for(uint32_t a=0;a<n;a++){node[a]=r[a].node;if(r[a].node==M->node)self=a;}
  qsort(name,names,sizeof *name,named_order);
  for(uint32_t b=0;self<n && b<n;b++){
    uint32_t *to=path+(size_t)b*n;
    paths[b]=b!=self?mesh_route_path(&whole.map,self,b,to):0;
    for(uint32_t i=0;i<paths[b];i++)to[i]=node[to[i]];
    hop[b]=paths[b]>=2?to[1]:0;
  }
  for(uint32_t b=0;self<n && b<n;b++){
    struct joined *c=joined+b;
    uint64_t g=0;
    c->count=b==self?0:MIN(mesh_joined(r+self,r+b,c->cable,c->weight,MESH_REACH),(uint32_t)MESH_REACH);
    for(uint32_t i=1;i<c->count;i++)
      for(uint32_t j=i;j && c->cable[j-1]>c->cable[j];j--){
        const uint32_t id=c->cable[j];const uint64_t bits=c->weight[j];
        c->cable[j]=c->cable[j-1];c->weight[j]=c->weight[j-1];c->cable[j-1]=id;c->weight[j-1]=bits;
      }
    for(uint32_t i=0;i<c->count;i++)for(uint64_t x=c->weight[i],y;x;x=y)y=g%x,g=x;
    for(uint32_t i=0;i<c->count;i++)c->cycle+=c->weight[i]=g?c->weight[i]/g:1;
  }
  for(uint32_t i=0;self<n && i<names;i++){
    uint32_t d=0;
    while(d<n && node[d]!=name[i].node)d++;
    for(uint32_t b=0;d<n && d!=self && b<n;b++)relaying[b]|=b!=d && joined[b].count && whole.next[(size_t)b*n+d]==self;
  }
  routes_free(w);
  made.index=w->index;made.index_room=w->index_room;made.table=w->table;made.table_room=w->table_room;made.fresh=w->fresh;
  *w=made;
  free(r);free(named);mesh_whole_free(&whole);
}
static uint32_t name_node(const struct walker *w,const uint64_t *key){
  uint32_t lo=0,hi=w->names;
  while(lo<hi){
    const uint32_t mid=lo+(hi-lo)/2;
    const int order=key_order(w->name[mid].name,key,2);
    if(!order)return w->name[mid].node;
    if(order<0)lo=mid+1;else hi=mid;
  }
  return 0;
}
static uint32_t hop_of(const struct walker *w,uint32_t node){
  for(uint32_t i=0;i<w->nodes;i++)if(w->node[i]==node)return w->hop[i];
  return 0;
}
static int joined_same(const struct walker *a,const struct walker *b,uint32_t node){
  const uint32_t hop=hop_of(a,node);
  uint32_t i=0,j=0;
  while(i<a->nodes && a->node[i]!=hop)i++;
  while(j<b->nodes && b->node[j]!=hop)j++;
  return hop && hop==hop_of(b,node) && i<a->nodes && j<b->nodes && !memcmp(a->joined+i,b->joined+j,sizeof *a->joined);
}
static int path_same(const struct walker *a,const struct walker *b,uint32_t node){
  uint32_t i=0,j=0;
  while(i<a->nodes && a->node[i]!=node)i++;
  while(j<b->nodes && b->node[j]!=node)j++;
  if(i==a->nodes || j==b->nodes || a->paths[i]!=b->paths[j])return 0;
  return !memcmp(a->path+(size_t)i*a->nodes,b->path+(size_t)j*b->nodes,a->paths[i]*sizeof *a->path);
}
/* design/transport.md section 14, "The wire"; SplitMix64's finalizer (Steele, Lea and Flood, OOPSLA 2014) */
static uint64_t key_mix(const uint64_t *key){
  uint64_t h=key[0]^key[1]*UINT64_C(0x9e3779b97f4a7c15)^key[2]*UINT64_C(0xbf58476d1ce4e5b9);
  h=(h^h>>30)*UINT64_C(0xbf58476d1ce4e5b9);h=(h^h>>27)*UINT64_C(0x94d049bb133111eb);
  return h^h>>31;
}
struct share { uint64_t base,cycle,lo,hi; uint32_t lanes; };
static int share_of(const struct walker *w,const struct port *port,uint32_t hop,const uint64_t *key,struct share *s){
  uint32_t b=0,i=0;
  while(b<w->nodes && w->node[b]!=hop)b++;
  const struct joined *c=b<w->nodes?w->joined+b:NULL;
  *s=(struct share){.lanes=plan_lanes(port->link.plan)};
  while(c && i<c->count && c->cable[i]!=port->link.cable)s->lo+=c->weight[i++];
  if(!c || i==c->count || !s->lanes || !c->weight[i])return 0;
  s->hi=s->lo+c->weight[i];s->cycle=c->cycle;s->base=key_mix(key)%c->cycle;
  return 1;
}
static uint64_t share_runs(const struct share *s,uint64_t x){
  const uint64_t r=x%s->cycle;
  return x/s->cycle*(s->hi-s->lo)+(r>s->lo?MIN(r,s->hi)-s->lo:0);
}
static int placed(const struct walker *w,const struct port *port,uint32_t hop,const uint64_t *key,uint64_t run,uint32_t *lane){
  struct share s;
  if(!share_of(w,port,hop,key,&s) || (s.base+run)%s.cycle<s.lo || (s.base+run)%s.cycle>=s.hi)return 0;
  *lane=(uint32_t)((share_runs(&s,s.base+run)-share_runs(&s,s.base))%s.lanes);
  return 1;
}
static uint64_t placed_runs(const struct walker *w,const struct port *port,uint32_t hop,const uint64_t *key,uint64_t runs,uint32_t lane){
  struct share s;
  if(!share_of(w,port,hop,key,&s))return 0;
  const uint64_t on=share_runs(&s,s.base+runs)-share_runs(&s,s.base);
  return on>lane?(on-lane+s.lanes-1)/s.lanes:0;
}
/* design/transport.md section 14, "Completion" and "Lost" */
static void tuple_count(struct mesh_plan *t,struct mesh_entry *e,uint64_t k,int way,uint64_t n){
  const uint64_t now=atomic_fetch_add_explicit(&e->count,n,memory_order_seq_cst)+n;
  if(mesh_called(now)!=e->pages || (uint32_t)(now>>32)!=k32(k))return;
  uint64_t done=atomic_load_explicit(&e->done,memory_order_seq_cst);
  if(done==MESH_STAMP_LOST || done>=k || !atomic_compare_exchange_strong_explicit(&e->done,&done,k,memory_order_seq_cst,memory_order_seq_cst))return;
  const uint64_t ways=atomic_fetch_add_explicit(&t->completed[way],1,memory_order_seq_cst)+1;
  if(mesh_called(ways)!=(way?t->lands:t->sends))return;
  _Atomic uint64_t *stamp=way?&t->landed:&t->sent;
  uint64_t was=atomic_load_explicit(stamp,memory_order_seq_cst);
  if(was!=MESH_STAMP_LOST && was<k)atomic_compare_exchange_strong_explicit(stamp,&was,k,memory_order_seq_cst,memory_order_seq_cst);
}
static uint32_t tuple_take(struct mesh_entry *e,uint64_t k,uint32_t most,uint32_t *first){
  uint64_t was=atomic_load_explicit(&e->posted,memory_order_seq_cst);
  for(;;){
    const uint32_t taken=mesh_called(was);
    if((uint32_t)(was>>32)!=k32(k) || taken>=e->pages)return 0;
    const uint32_t n=(uint32_t)MIN((uint64_t)most,e->pages-taken);
    if(!atomic_compare_exchange_weak_explicit(&e->posted,&was,was+n,memory_order_seq_cst,memory_order_seq_cst))continue;
    *first=taken;
    return n;
  }
}
static int tuple_post(struct mesh_entry *e,uint64_t k,uint32_t n){
  uint64_t was=atomic_load_explicit(&e->posted,memory_order_seq_cst);
  do if((uint32_t)(was>>32)!=k32(k))return -1;
     else if(mesh_called(was)+(uint64_t)n>e->pages)return 0;
  while(!atomic_compare_exchange_weak_explicit(&e->posted,&was,was+n,memory_order_seq_cst,memory_order_seq_cst));
  if(atomic_load_explicit(&e->done,memory_order_seq_cst)!=MESH_STAMP_LOST)return 1;
  atomic_fetch_add_explicit(&e->count,n,memory_order_seq_cst);
  return 0;
}
/* design/transport.md section 14, "Rows"; section 15, "The SEND gate" */
static uint64_t region_of(const void *p){ return p?(uint64_t)((const char *)p-(const char *)M):0; }
static uint64_t wired_of(const void *p){ return p?M->data_off+(uint64_t)((const char *)p-wire.data):0; }
static struct mesh_entry *unit_entry(const struct unit *u){ return u->entry?(struct mesh_entry *)((char *)M+u->entry):NULL; }
static struct mesh_plan *unit_table(const struct unit *u){ return u->table?(struct mesh_plan *)((char *)M+u->table):NULL; }
static struct relay *unit_relay(const struct unit *u){ return u->relay?(struct relay *)(wire.data+(u->relay-M->data_off)):NULL; }
static struct unit unit_of(const struct run *r){
  return (struct unit){.table=region_of(r->table),.entry=region_of(r->entry),.relay=wired_of(r->relay),.k=r->k,.kind=r->kind};
}
static void run_note(struct lane_state *s,int way,uint64_t start){
  const struct run *r=way?&s->landing:&s->sending;
  const int pages=r->state==RUN_PAGES || (r->state==RUN_KEY && !way);
  s->record->run[way]=(struct run_record){pages?unit_of(r):(struct unit){0},start,pages?r->unit.count:0,0};
}
static int post_send(struct port *port,uint32_t lane,struct ibv_sge span,struct unit u){
  struct link *l=&port->link;
  struct lane_state *s=l->lane+lane;
  const struct mesh_queue *queue=l->provider.queues+lane;
  struct ibv_send_wr posting={.wr_id=lane,.sg_list=&span,.num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
  s->record->ring[0][s->sent%MESH_RING]=u;
  const int error=queue->send(queue->pair,&posting,&bad);
  if(error){lanes_lost(port,UINT32_C(1)<<lane,error<0?-error:error);return 0;}
  atomic_store_explicit(&s->record->posted[0],++s->sent,memory_order_release);
  return 1;
}
static int post_receive(struct port *port,uint32_t lane,struct ibv_sge span,struct unit u){
  struct link *l=&port->link;
  struct lane_state *s=l->lane+lane;
  const struct mesh_queue *queue=l->provider.queues+lane;
  struct ibv_recv_wr request={.wr_id=lane,.sg_list=&span,.num_sge=1},*bad;
  s->record->ring[1][s->posted%MESH_RING]=u;
  const int error=queue->receive(queue->pair,&request,&bad);
  if(error){lanes_lost(port,UINT32_C(1)<<lane,error<0?-error:error);return 0;}
  atomic_store_explicit(&s->record->posted[1],++s->posted,memory_order_release);
  return 1;
}
static uint64_t relay_at(const struct relay *x,uint32_t page){ return ((uint64_t)x->page+(uint64_t)page*M->block)*M->pgsz; }
/* design/transport.md section 14, "Sending" */
static void run_send(struct port *port,uint32_t lane){
  struct link *l=&port->link;
  struct lane_state *s=l->lane+lane;
  struct run *r=&s->sending;
  if(r->state==RUN_KEY && s->sent-s->done<s->sends && s->keys-s->keys_done<MESH_KEY_SLOTS){
    const uint32_t slot=(uint32_t)(s->keys%MESH_KEY_SLOTS);
    const uint64_t at=link_at(l)+link_keys_at(page_bytes())+((uint64_t)lane*MESH_KEY_SLOTS+slot)*sizeof(struct key_unit);
    memcpy(wire.data+at,&r->unit,sizeof r->unit);
    if(!post_send(port,lane,span_of(port,at,sizeof r->unit),(struct unit){.kind=UNIT_KEY,.slot=slot}))return;
    s->keys++;
    r->state=RUN_PAGES;
    run_note(s,0,s->sent);
  }
  while(r->state==RUN_PAGES && r->posted<r->unit.count && s->sent-s->done<s->sends){
    uint64_t at=(uint64_t)ones*M->pgsz;
    if(r->kind==UNIT_RELAY)at=relay_at(r->relay,r->posted);
    else if(r->entry && region_word(r->entry->word) && atomic_load_explicit(&r->entry->done,memory_order_acquire)!=MESH_STAMP_LOST &&
            mesh_rows_hold(M,(struct mesh_row){r->first},(struct mesh_pages){r->unit.count}))
      row_at((struct mesh_row){r->first+r->posted},&at);
    if(!post_send(port,lane,page_span(port,at),unit_of(r)))return;
    r->posted++;
  }
  if(r->state==RUN_PAGES && r->posted==r->unit.count){*r=(struct run){0};run_note(s,0,0);}
}
static int blocked(const struct walker *w,uint32_t node,uint32_t before){
  for(uint32_t i=0;i<before;i++)if(w->blocked[i]==node)return 1;
  return w->blocks==MESH_CLIENTS;
}
static void block(struct walker *w,uint32_t node){ if(!blocked(w,node,w->blocks))w->blocked[w->blocks++]=node; }
static int table_order(const void *a,const void *b){
  const uint64_t x=atomic_load_explicit(&(*(struct mesh_plan *const *)a)->start,memory_order_relaxed),
                 y=atomic_load_explicit(&(*(struct mesh_plan *const *)b)->start,memory_order_relaxed);
  return (x>y)-(x<y);
}
static void tables_send(struct port *port,struct walker *w,uint32_t served){
  for(uint32_t c=0;c<MESH_CLIENTS;c++){
    const struct mesh_client *slot=mesh_slot(M,c);
    if(!owned(atomic_load_explicit(&slot->owner,memory_order_acquire)))continue;
    w->tables=0;
    uint32_t tables=mesh_rows(M);
    for(struct mesh_plan *t=queue_first(slot);t && tables--;t=queue_next(t)){
      const uint64_t k=atomic_load_explicit(&t->start,memory_order_acquire),sent=k?atomic_load_explicit(&t->sent,memory_order_acquire):0;
      if((t->name[0] || t->name[1]) && w->told!=w->serial && name_node(w,t->name)!=M->node){
        w->told=w->serial;
        atomic_store_explicit(&naming,1,memory_order_release);
        wake();
      }
      if(!k || !t->sends || sent==k || sent==MESH_STAMP_LOST)continue;
      if(w->tables==w->table_room){
        struct mesh_plan **grown=realloc(w->table,(w->table_room?2*w->table_room:64)*sizeof *grown);
        if(!grown)return;
        w->table=grown;w->table_room=w->table_room?2*w->table_room:64;
      }
      w->table[w->tables++]=t;
    }
    qsort(w->table,w->tables,sizeof *w->table,table_order);
    w->blocks=0;
    for(uint32_t i=0;i<w->tables;i++){
      struct mesh_plan *t=w->table[i];
      const uint64_t k=atomic_load_explicit(&t->start,memory_order_acquire);
      const uint32_t before=w->blocks;
      for(uint32_t e=0;e<t->sends;e++){
        struct mesh_entry *x=t->entry+e;
        const uint64_t done=atomic_load_explicit(&x->done,memory_order_acquire);
        const uint32_t node=x->pages && (done<k || done==MESH_STAMP_LOST)?name_node(w,x->key):0,hop=node?hop_of(w,node):0;
        uint32_t lane=MESH_LINK_LANES,first,n;
        if(!hop || blocked(w,node,before) || hop!=port->peer)continue;
        const _Atomic uint64_t *word=region_word(x->word);
        if(done!=MESH_STAMP_LOST && word && atomic_load_explicit(word,memory_order_acquire)<x->need){block(w,node);continue;}
        while(placed(w,port,hop,x->key,mesh_called(atomic_load_explicit(&x->posted,memory_order_acquire))/MESH_RUN,&lane) && (served>>lane&1) &&
              port->link.lane[lane].sending.state==RUN_NONE && (n=tuple_take(x,k,MESH_RUN,&first))){
          struct run *r=&port->link.lane[lane].sending;
          *r=(struct run){.table=t,.entry=x,.k=k,.first=x->row+first,.state=RUN_KEY,.kind=UNIT_PAGE};
          memcpy(r->unit.key,x->key,sizeof r->unit.key);r->unit.offset=first;r->unit.count=n;
          run_note(port->link.lane+lane,0,UINT64_MAX);
          run_send(port,lane);
        }
        if(mesh_called(atomic_load_explicit(&x->posted,memory_order_acquire))<x->pages)block(w,node);
      }
    }
  }
}
/* design/transport.md section 14, "Relays" */
static void relays_send(struct port *port,struct walker *w,uint32_t served){
  struct link *l=&port->link;
  const uint32_t lanes=plan_lanes(l->plan);
  for(uint32_t p=0;lanes && p<MESH_PORTS;p++){
    const uint32_t page=mesh_bridge_pages(M)->link[p];
    if(page==MESH_ABSENT)continue;
    struct relay *relay=(struct relay *)(wire.data+(uint64_t)page*M->pgsz+link_relays_at(page_bytes()));
    for(uint32_t i=0;i<MESH_RELAYS;i++){
      struct relay *x=relay+i;
      if(atomic_load_explicit(&x->state,memory_order_acquire)!=RELAY_READY)continue;
      const uint32_t hop=hop_of(w,x->node);
      uint32_t lane=MESH_LINK_LANES;
      if(!hop || hop!=port->peer || !placed(w,port,hop,x->key,x->offset/MESH_RUN,&lane) || !(served>>lane&1) || l->lane[lane].sending.state!=RUN_NONE)
        continue;
      int first=1;
      for(uint32_t j=0;first && j<MESH_RELAYS;j++){
        const uint32_t state=atomic_load_explicit(&relay[j].state,memory_order_acquire);
        first=!((state==RELAY_LANDING || state==RELAY_READY) && relay[j].lane==x->lane && relay[j].node==x->node && relay[j].order<x->order);
      }
      uint32_t ready=RELAY_READY;
      if(!first || !atomic_compare_exchange_strong_explicit(&x->state,&ready,relay_sending(port->index),memory_order_acq_rel,memory_order_acquire))continue;
      struct run *r=&l->lane[lane].sending;
      *r=(struct run){.relay=x,.state=RUN_KEY,.kind=UNIT_RELAY};
      memcpy(r->unit.key,x->key,sizeof r->unit.key);r->unit.offset=x->offset;r->unit.count=x->count;
      run_note(l->lane+lane,0,UINT64_MAX);
      run_send(port,lane);
    }
  }
}
/* design/transport.md section 14, "The index" */
static int indexed_order(const void *a,const void *b){ return key_order(((const struct indexed *)a)->key,((const struct indexed *)b)->key,3); }
static void index_make(struct walker *w){
  w->fresh=1;
  w->indexed=0;
  for(uint32_t c=0;c<MESH_CLIENTS;c++){
    const struct mesh_client *slot=mesh_slot(M,c);
    if(!owned(atomic_load_explicit(&slot->owner,memory_order_acquire)))continue;
    uint32_t tables=mesh_rows(M);
    for(struct mesh_plan *t=queue_first(slot);t && tables--;t=queue_next(t)){
      const uint64_t k=atomic_load_explicit(&t->start,memory_order_acquire),landed=k?atomic_load_explicit(&t->landed,memory_order_acquire):0;
      if(!k || !t->lands || landed==k)continue;
      for(uint32_t e=0;e<t->lands;e++){
        struct mesh_entry *x=t->entry+t->sends+e;
        if(!x->pages || atomic_load_explicit(&x->done,memory_order_acquire)==k)continue;
        if(w->indexed==w->index_room){
          struct indexed *grown=realloc(w->index,(w->index_room?2*w->index_room:64)*sizeof *grown);
          if(!grown)break;
          w->index=grown;w->index_room=w->index_room?2*w->index_room:64;
        }
        struct indexed *y=w->index+w->indexed++;
        memcpy(y->key,x->key,sizeof y->key);y->table=t;y->entry=x;y->k=k;
      }
    }
  }
  qsort(w->index,w->indexed,sizeof *w->index,indexed_order);
}
static struct indexed *index_find(struct walker *w,const uint64_t *key){
  if(!w->fresh)index_make(w);
  uint32_t lo=0,hi=w->indexed;
  while(lo<hi){
    const uint32_t mid=lo+(hi-lo)/2;
    const int order=key_order(w->index[mid].key,key,3);
    if(!order)return w->index+mid;
    if(order<0)lo=mid+1;else hi=mid;
  }
  return NULL;
}
static struct relay *relay_take(struct port *port,uint32_t lane,uint32_t node,const struct key_unit *unit){
  struct link *l=&port->link;
  struct relay *relay=(struct relay *)(wire.data+link_at(l)+link_relays_at(page_bytes()));
  uint32_t entry=MESH_RELAYS,first=MESH_POOL_BLOCKS;
  for(uint32_t i=0;entry==MESH_RELAYS && i<MESH_RELAYS;i++)if(atomic_load_explicit(&relay[i].state,memory_order_acquire)==RELAY_FREE)entry=i;
  for(uint32_t b=0,run=0;entry<MESH_RELAYS && first==MESH_POOL_BLOCKS && b<MESH_POOL_BLOCKS;b++){
    run=l->pool_free[b/64]>>(b%64)&1?0:run+1;
    if(run==unit->count)first=b+1-run;
  }
  if(entry==MESH_RELAYS || first==MESH_POOL_BLOCKS)return NULL;
  for(uint32_t b=first;b<first+unit->count;b++)l->pool_free[b/64]|=UINT64_C(1)<<(b%64);
  struct relay *x=relay+entry;
  x->lane=lane;x->order=++l->order[lane];memcpy(x->key,unit->key,sizeof x->key);x->offset=unit->offset;x->count=unit->count;
  x->page=l->pool+first*M->block;x->node=node;
  atomic_store_explicit(&x->landed,0,memory_order_relaxed);atomic_store_explicit(&x->sent,0,memory_order_relaxed);
  atomic_store_explicit(&x->state,RELAY_LANDING,memory_order_release);
  return x;
}
static void relays_free(struct port *port){
  struct link *l=&port->link;
  struct relay *relay=(struct relay *)(wire.data+link_at(l)+link_relays_at(page_bytes()));
  for(uint32_t i=0;i<MESH_RELAYS;i++){
    struct relay *x=relay+i;
    if((atomic_load_explicit(&x->state,memory_order_acquire)&0xff)!=RELAY_SENDING || atomic_load_explicit(&x->sent,memory_order_acquire)<x->count)continue;
    for(uint32_t b=(x->page-l->pool)/M->block;b<(x->page-l->pool)/M->block+x->count;b++)l->pool_free[b/64]&=~(UINT64_C(1)<<(b%64));
    atomic_store_explicit(&x->state,RELAY_FREE,memory_order_release);
  }
}
static uint32_t *arrival_runs(struct lane_state *s,const struct mesh_entry *entry,uint64_t k){
  for(uint32_t i=0;i<s->arrivals;i++)if(s->arrival[i].entry==entry && s->arrival[i].k==k)return &s->arrival[i].runs;
  return NULL;
}
static void arrival_add(struct lane_state *s,struct mesh_entry *entry,uint64_t k){
  uint32_t *runs=arrival_runs(s,entry,k);
  if(!runs && s->arrivals==s->arrival_room){
    struct arrival *grown=realloc(s->arrival,(s->arrival_room?2*s->arrival_room:16)*sizeof *grown);
    if(!grown)return;
    s->arrival=grown;s->arrival_room=s->arrival_room?2*s->arrival_room:16;
  }
  if(!runs){s->arrival[s->arrivals]=(struct arrival){entry,k,0,0};runs=&s->arrival[s->arrivals++].runs;}
  ++*runs;
}
/* design/transport.md section 14, "The index" */
static void run_place(struct port *port,struct walker *w,uint32_t lane){
  struct run *r=&port->link.lane[lane].landing;
  const struct key_unit *u=&r->unit;
  const struct indexed *x=index_find(w,u->key);
  const uint32_t node=x?M->node:name_node(w,u->key);
  int placed=0;
  r->posted=0;r->state=RUN_PAGES;
  if(u->count>MESH_RUN)r->kind=UNIT_SINK;
  else if(x && (uint64_t)u->offset+u->count<=x->entry->pages && (placed=tuple_post(x->entry,x->k,u->count))>0){
    r->kind=UNIT_PAGE;r->table=x->table;r->entry=x->entry;r->k=x->k;r->first=x->entry->row+u->offset;
    if(!mesh_rows_hold(M,(struct mesh_row){x->entry->row},(struct mesh_pages){x->entry->pages}))r->first=UINT64_MAX;
    run_note(port->link.lane+lane,1,port->link.lane[lane].posted);
  }
  else if((!x && node==M->node) || placed<0)r->state=RUN_HELD;
  else if(!x && node && hop_of(w,node)){
    r->kind=UNIT_RELAY;
    if(!(r->relay=relay_take(port,lane,node,u)))r->state=RUN_HELD;
  }
  else r->kind=UNIT_SINK;
  if(x && r->state==RUN_PAGES && u->count<=MESH_RUN && (uint64_t)u->offset+u->count<=x->entry->pages)arrival_add(port->link.lane+lane,x->entry,x->k);
}
static int lane_expects(struct port *port,struct walker *w,uint32_t lane){
  struct lane_state *s=port->link.lane+lane;
  uint32_t b=0;
  while(b<w->nodes && w->node[b]!=port->peer)b++;
  if(b<w->nodes && w->relaying[b])return 1;
  if(!w->fresh)index_make(w);
  for(uint32_t i=0;i<s->arrivals;){
    const struct arrival *a=s->arrival+i;
    const uint64_t posted=atomic_load_explicit(&a->entry->posted,memory_order_acquire),done=atomic_load_explicit(&a->entry->done,memory_order_acquire);
    if((uint32_t)(posted>>32)==k32(a->k) && done!=a->k)i++;
    else s->arrival[i]=s->arrival[--s->arrivals];
  }
  for(uint32_t i=0;i<w->indexed;i++){
    const struct indexed *y=w->index+i;
    const uint64_t posted=atomic_load_explicit(&y->entry->posted,memory_order_acquire);
    const uint32_t from=name_node(w,y->entry->from),*runs=arrival_runs(s,y->entry,y->k);
    if((uint32_t)(posted>>32)!=k32(y->k) || mesh_called(posted)>=y->entry->pages || !from || hop_of(w,from)!=port->peer)continue;
    if(placed_runs(w,port,port->peer,y->key,(y->entry->pages+MESH_RUN-1)/MESH_RUN,lane)>(runs?*runs:0))return 1;
  }
  return 0;
}
static void run_receive(struct port *port,struct walker *w,uint32_t lane){
  struct link *l=&port->link;
  struct lane_state *s=l->lane+lane;
  struct run *r=&s->landing;
  if(r->state==RUN_HELD)run_place(port,w,lane);
  while(r->state==RUN_PAGES && r->posted<r->unit.count && s->posted-s->got<s->window){
    uint64_t at=link_at(l);
    if(r->kind==UNIT_PAGE && r->first!=UINT64_MAX)row_at((struct mesh_row){r->first+r->posted},&at);
    else if(r->kind==UNIT_RELAY)at=relay_at(r->relay,r->posted);
    if(!post_receive(port,lane,page_span(port,at),unit_of(r)))return;
    r->posted++;
  }
  if(r->state==RUN_PAGES && r->posted==r->unit.count){*r=(struct run){0};run_note(s,1,0);}
  if(r->state==RUN_NONE && s->posted-s->got<s->window && lane_expects(port,w,lane)){
    const uint64_t at=link_at(l)+link_heard_at(page_bytes())+(uint64_t)lane*sizeof(struct key_unit);
    if(post_receive(port,lane,span_of(port,at,sizeof(struct key_unit)),(struct unit){.kind=UNIT_KEY}))r->state=RUN_KEY;
  }
}
/* design/transport.md section 15, "The data path" */
static __attribute__((always_inline)) inline uint32_t link_lane(const struct link *l,uint32_t qp_num){
  uint32_t k=0;
  while(k<MESH_LINK_LANES && l->qpn[k]!=qp_num)k++;
  return k;
}
static void completed(struct port *port,struct walker *w,int way,uint32_t lane){
  struct link *l=&port->link;
  struct lane_state *s=l->lane+lane;
  const struct unit u=s->record->ring[way][*lane_done(s,way)%MESH_RING];
  ++*lane_done(s,way);
  struct mesh_entry *entry=unit_entry(&u);
  struct relay *relay=unit_relay(&u);
  if(u.kind==UNIT_KEY && !way)s->keys_done++;
  else if(u.kind==UNIT_KEY){
    memcpy(&s->landing.unit,wire.data+link_at(l)+link_heard_at(page_bytes())+(uint64_t)lane*sizeof(struct key_unit),sizeof s->landing.unit);
    if(s->landing.unit.count)run_place(port,w,lane);
    else s->landing=(struct run){0};
  }
  else if(u.kind==UNIT_PAGE && entry)tuple_count(unit_table(&u),entry,u.k,way,1);
  else if(u.kind==UNIT_RELAY && !way)atomic_fetch_add_explicit(&relay->sent,1,memory_order_acq_rel);
  else if(u.kind==UNIT_RELAY && atomic_fetch_add_explicit(&relay->landed,1,memory_order_acq_rel)+1==relay->count)
    atomic_store_explicit(&relay->state,RELAY_READY,memory_order_release);
  atomic_store_explicit(&s->record->done[way],*lane_done(s,way),memory_order_release);
}
static void port_pass(struct port *port,int way,struct walker *w){
  struct link *l=&port->link;
  const uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire),lanes=plan_lanes(l->plan);
  if(!served)return;
  routes_fresh(w);
  walk_mark(port->index,way,1);
  w->fresh=0;
  struct ibv_wc done[16];
  const int count=l->provider.poll(way?l->provider.completion:l->provider.sent,16,done);
  int failed=count<0;
  if(failed)lanes_lost(port,~UINT32_C(0),EIO);
  if(way && count>0)atomic_thread_fence(memory_order_acquire);
  for(int i=0;!failed && i<count;i++){
    if(done[i].status){lanes_lost(port,~UINT32_C(0),done[i].status);failed=1;break;}
    const uint32_t lane=link_lane(l,done[i].qp_num);
    if(lane<MESH_LINK_LANES)completed(port,w,way,lane);
  }
  for(uint32_t lane=0;!failed && lane<lanes;lane++)if(served>>lane&1){
    if(way)run_receive(port,w,lane);
    else run_send(port,lane);
  }
  if(!failed && way)relays_free(port);
  else if(!failed){
    tables_send(port,w,served);
    relays_send(port,w,served);
  }
  walk_mark(port->index,way,0);
}
/* design/transport.md section 14, "Lost" and "Rows" */
static void unit_lost(struct hdr *m,const struct unit *u,int way){
  struct mesh_entry *entry=u->kind==UNIT_PAGE && u->entry?(struct mesh_entry *)((char *)m+u->entry):NULL;
  struct mesh_plan *table=u->table?(struct mesh_plan *)((char *)m+u->table):NULL;
  if(!entry || !table)return;
  mesh_stamp_lost(&entry->done,u->k);
  mesh_stamp_lost(way?&table->landed:&table->sent,u->k);
}
static uint32_t run_unposted(const struct lane_record *x,int way){
  const struct run_record *r=x->run+way;
  const uint64_t posted=atomic_load_explicit(&x->posted[way],memory_order_acquire);
  const uint64_t done=r->start==UINT64_MAX || posted<r->start?0:posted-r->start;
  return r->unit.kind==UNIT_PAGE && done<r->count?r->count-(uint32_t)done:0;
}
static void lanes_halted(struct port *port,int way){
  for(uint32_t lane=0;lane<MESH_LINK_LANES;lane++){
    struct lane_record *x=port->link.lane[lane].record;
    for(uint64_t i=x?atomic_load_explicit(&x->done[way],memory_order_acquire):0;x && i<atomic_load_explicit(&x->posted[way],memory_order_acquire);i++)
      unit_lost(M,x->ring[way]+i%MESH_RING,way);
    if(x && run_unposted(x,way))unit_lost(M,&x->run[way].unit,way);
  }
}
/* design/transport.md section 14, "A progress thread with nothing to do" */
static void *port_progress(void *argument){
  struct progress *self=argument;
  struct port *port=self->port;
  const int way=self->way;
  pthread_setname_np(way?"mesh.rdma.receive":"mesh.rdma.send");
  stack_t alternate={.ss_sp=malloc(SIGSTKSZ),.ss_size=SIGSTKSZ};
  if(alternate.ss_sp)sigaltstack(&alternate,NULL);
  for(uint64_t pass=0,halted=1;!atomic_load_explicit(&port->closing,memory_order_acquire);){
    atomic_store_explicit(&self->passes,++pass,memory_order_relaxed);
    if(atomic_load_explicit(&port->link.served,memory_order_acquire)){halted=0;port_pass(port,way,&self->walker);continue;}
    if(!halted){lanes_halted(port,way);halted=1;}
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
    port->progress[port->threads]=(struct progress){.port=port,.way=port->threads};
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
static void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes){
  if(port->control<0 || (said.kind!=HELLO && !port->greeted))return;
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
static void record_say(struct port *port,uint32_t k){
  const size_t bytes=sizeof *known+(size_t)known[k].names*sizeof *names;
  unsigned char *said=malloc(bytes);
  if(!said){port_lost(port,ENOMEM);return;}
  memcpy(said,known+k,sizeof *known);
  if(known[k].names)memcpy(said+sizeof *known,known_name[k],(size_t)known[k].names*sizeof *names);
  port_say(port,(struct message){.kind=RECORD},said,(uint32_t)bytes);
  free(said);
}
static void records_say(const struct port *from,uint32_t k){
  for(uint32_t p=0;p<port_count;p++)
    if(ports+p!=from && ports[p].greeted)record_say(ports+p,k);
}
static uint64_t available_now(void){
  int level=0;size_t size=sizeof level;
  return sysctlbyname("kern.memorystatus_level",&level,&size,NULL,0) || level<0?0:(uint64_t)level*(memory/100);
}
static struct mesh_record *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  return NULL;
}
static int known_put(const struct mesh_record *record,const uint64_t (*named)[2],uint32_t *index){
  struct mesh_record *at=known_at(record->node);
  uint64_t (*copy)[2]=malloc(((size_t)record->names+1)*sizeof *copy);
  if(!copy){say("map: no memory to hold node %u's record\n",record->node);return ENOMEM;}
  if(!at && known_count==known_room){
    const uint32_t room=known_room?2*known_room:16;
    struct mesh_record *grown=realloc(known,room*sizeof *known);
    uint64_t (**grown_name)[2]=grown?realloc(known_name,room*sizeof *known_name):NULL;
    if(grown)known=grown;
    if(!grown_name){free(copy);say("map: no memory to hold node %u's record\n",record->node);return ENOMEM;}
    known_name=grown_name;known_room=room;
  }
  if(!at){at=known+known_count;known_name[known_count++]=NULL;}
  const uint32_t k=(uint32_t)(at-known);
  memcpy(copy,named,(size_t)record->names*sizeof *copy);
  free(known_name[k]);known_name[k]=copy;
  *at=*record;
  *index=k;
  return 0;
}
static int map_write(const struct mesh_record *records,const uint32_t *index,uint64_t n){
  uint64_t total=0;
  for(uint64_t i=0;i<n;i++)total+=records[i].names;
  const uint64_t bytes=n*sizeof *records+total*sizeof *names;
  const uint32_t pages=(uint32_t)((bytes+M->pgsz-1)/M->pgsz)+!n,page=mesh_arena_alloc(&bridge,pages,1);
  if(page==MESH_ABSENT)return ENOMEM;
  memcpy(mesh_at(M,page),records,(size_t)n*sizeof *records);
  unsigned char *at=mesh_at(M,page)+n*sizeof *records;
  for(uint64_t i=0;i<n;i++){memcpy(at,known_name[index[i]],(size_t)records[i].names*sizeof *names);at+=(size_t)records[i].names*sizeof *names;}
  atomic_store_explicit(&M->map_at,(uint64_t)page<<32|n,memory_order_release);
  atomic_fetch_add_explicit(&M->map,1,memory_order_release);
  mesh_bridge_pages(M)->map=page;mesh_bridge_pages(M)->map_pages=pages;
  if(map_pages)mesh_bits_clear(mesh_arena_bits(M),map_page,map_pages);
  held_pages+=pages;held_pages-=map_pages;
  map_page=page;map_pages=pages;
  return 0;
}
/* design/transport.md section 14, "Lost" */
static struct walker routes;
static void tuple_stamped(struct mesh_plan *t,struct mesh_entry *x,uint64_t k,int landing){
  mesh_stamp_lost(&x->done,k);
  mesh_stamp_lost(landing?&t->landed:&t->sent,k);
}
static void tuples_walk(const struct walker *now,uint32_t peer){
  walk_mark(MESH_PORTS,0,1);
  for(uint32_t c=0;c<MESH_CLIENTS;c++){
    const struct mesh_client *slot=mesh_slot(M,c);
    uint32_t tables=mesh_rows(M);
    for(struct mesh_plan *t=owned(atomic_load_explicit(&slot->owner,memory_order_acquire))?queue_first(slot):NULL;t && tables--;t=queue_next(t)){
      const uint64_t k=atomic_load_explicit(&t->start,memory_order_acquire);
      for(uint32_t e=0;k && e<t->sends+t->lands;e++){
        struct mesh_entry *x=t->entry+e;
        const int landing=e>=t->sends;
        if(!x->pages || atomic_load_explicit(&x->done,memory_order_acquire)>=k)continue;
        const uint64_t *name=landing?x->from:x->key;
        const uint32_t was=name_node(&routes,name),is=now?name_node(now,name):was;
        if(!was || was==M->node)continue;
        if(!now){if(landing && hop_of(&routes,was)==peer)tuple_stamped(t,x,k,landing);continue;}
        const int moved=!is || !path_same(&routes,now,was) || !joined_same(&routes,now,was);
        if(!is || (moved && (landing || mesh_called(atomic_load_explicit(&x->posted,memory_order_acquire)))))tuple_stamped(t,x,k,landing);
      }
    }
  }
  walk_mark(MESH_PORTS,0,0);
}
static void landings_from(uint32_t peer){ tuples_walk(NULL,peer); }
static void tuples_rerouted(void){
  struct walker now={0};
  routes_fresh(&now);
  if(!now.serial)return;
  if(routes.serial && now.serial!=routes.serial)tuples_walk(&now,0);
  routes_free(&routes);
  routes=now;
}
static void map_publish(void){
  const uint64_t published=atomic_load(&M->map_at),n=(uint32_t)published;
  const struct mesh_record *r=mesh_records(M,published);
  struct mesh_record *next=malloc((known_count+1)*sizeof *next);
  unsigned char *reached=calloc(known_count+1,1);
  uint32_t *order=malloc((known_count+1)*sizeof *order),*index=malloc((known_count+1)*sizeof *index),head=0,tail=0;
  if(!next || !reached || !order || !index){free(next);free(reached);free(order);free(index);say("map: no memory to make it\n");return;}
  for(uint32_t k=0;k<known_count;k++)if(known[k].node==M->node){reached[k]=1;order[tail++]=k;}
  while(head<tail){
    const uint32_t a=order[head++];
    for(uint32_t b=0;b<known_count;b++)if(!reached[b] && mesh_joined(known+a,known+b,NULL,NULL,0)){reached[b]=1;order[tail++]=b;}
  }
  uint64_t kept=0;
  for(uint32_t k=0;k<known_count;k++){
    int held=0;
    for(uint64_t i=0;i<n;i++)held|=r[i].node==known[k].node;
    if(held!=reached[k] && known[k].node!=M->node)say(reached[k]?"node %u reached\n":"node %u unreachable\n",known[k].node);
    if(reached[k]){index[kept]=k;next[kept++]=known[k];}
  }
  const int changed=kept!=n || memcmp(next,r,(size_t)n*sizeof *r);
  if(changed && map_write(next,index,kept))say("map: no lease for %llu records\n",(unsigned long long)kept);
  else if(changed)tuples_rerouted();
  free(next);free(reached);free(order);free(index);
}
static void record_changed(void){
  health.available=available_now();
  struct mesh_record own={.node=M->node,.boot=M->boot,.sequence=++sequence,.region=wire.length-held_pages*M->pgsz,.memory=memory,.health=health};
  memcpy(own.model,model,sizeof own.model);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].link.state==LINK_UP && ports[p].link.plan)
    own.reach[own.reached++]=(struct mesh_reach){.node=ports[p].peer,.cable=ports[p].link.cable,.bandwidth=ports[p].bandwidth};
  own.names=names_held;
  uint32_t k;
  if(known_put(&own,(const uint64_t (*)[2])names,&k))return;
  map_publish();
  records_say(NULL,k);
}
static void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard={0};
  memcpy(&heard,payload,MIN((size_t)bytes,sizeof heard));
  if(!heard.node || heard.node==M->node)return;
  heard.reached=MIN(heard.reached,(uint32_t)MESH_REACH);
  heard.names=(uint32_t)MIN((uint64_t)heard.names,bytes>sizeof heard?(bytes-sizeof heard)/sizeof *names:0);
  heard.model[sizeof heard.model-1]=0;
  const struct mesh_record *known_one=known_at(heard.node);
  if(known_one && (heard.boot<known_one->boot || (heard.boot==known_one->boot && heard.sequence<=known_one->sequence)))return;
  uint32_t k;
  if(known_put(&heard,(const uint64_t (*)[2])(payload+sizeof heard),&k))return;
  map_publish();
  records_say(from,k);
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
static int device_up(struct mesh_device *device){
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
  if(!device->extent_mrs){
    uint32_t shift=(uint32_t)__builtin_ctzll(page_bytes());
    while(shift<32 && ((wire.length-1)>>shift)+1>(uint64_t)(capabilities.max_mr>0?capabilities.max_mr:1))shift++;
    device->extent_shift=(uint8_t)shift;
    if(!(device->extent_mrs=calloc(((wire.length-1)>>shift)+1,sizeof *device->extent_mrs)))return ENOMEM;
  }
  const size_t extent=(size_t)1<<device->extent_shift;
  const uint32_t extents=(uint32_t)(((wire.length-1)>>device->extent_shift)+1);
  const uint64_t began=clock_ns();
  for(size_t e=0,first=0;e<extents;e++,first+=extent){
    const size_t bytes=wire.length-first<extent?wire.length-first:extent;
    if(!device->extent_mrs[e] && !(device->extent_mrs[e]=ibv_reg_mr(device->domain,wire.data+first,bytes,IBV_ACCESS_LOCAL_WRITE)))
      return errno?errno:EIO;
  }
  say("%s: region of %zu bytes (%u extents of %zu bytes) registered in %.3f s\n",device->name,wire.length,extents,extent,
      (double)(clock_ns()-began)/1e9);
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
  static const enum ibv_rate lane[256]={[1]=IBV_RATE_2_5_GBPS,[2]=IBV_RATE_5_GBPS,[4]=IBV_RATE_10_GBPS,[8]=IBV_RATE_10_GBPS,[16]=IBV_RATE_14_GBPS,
                                         [32]=IBV_RATE_25_GBPS,[64]=IBV_RATE_50_GBPS,[128]=IBV_RATE_100_GBPS};
  static const uint8_t widths[256]={[1]=1,[2]=4,[4]=8,[8]=12,[16]=2};
  const int mbps=ibv_rate_to_mbps(lane[port->active_speed]);
  return mbps>0?(uint64_t)mbps*UINT64_C(1000000)*widths[port->active_width]:0;
}
static void job_begin(struct port *port,int job){
  port->job=job;port->job_pairing=atomic_load_explicit(&port->link.pairing,memory_order_relaxed);port->job_at=now;
  atomic_store_explicit(&port->done,0,memory_order_relaxed);
}
static int link_lapsed(const struct port *port){ return atomic_load_explicit(&port->link.pairing,memory_order_relaxed)!=port->job_pairing?ECANCELED:0; }
static int link_make(struct port *port){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  struct ibv_port_attr attributes;
  int error=0;
  if(p->queues)return EALREADY;
  if(ibv_query_port(port->device.context,1,&attributes))return errno?errno:EIO;
  port->device.port=attributes;
  if((error=device_gid(&port->device,port->name)) || (error=link_lapsed(port)) || (error=pair_up(p,l->plan)))return error;
  l->share=(struct frames){(uint64_t)MIN(p->completion->cqe,p->sent->cqe)/MAX(plan_lanes(l->plan),1u)};
  return 0;
}
static uint32_t link_cable(const struct port *port){
  uint8_t end[2][20];
  memcpy(end[0],port->gid,16);memcpy(end[0]+16,&port->link.qpn[0],4);
  memcpy(end[1],port->heard.gid,16);memcpy(end[1]+16,&port->link.peer[0][0],4);
  const int first=memcmp(end[0],end[1],20)>0;
  uint64_t h=UINT64_C(14695981039346656037);
  for(int e=0;e<2;e++)
    for(int i=0;i<20;i++){h^=end[e^first][i];h*=UINT64_C(1099511628211);}
  const uint32_t cable=(uint32_t)(h^h>>32);
  return cable?cable:1;
}
/* design/transport.md section 14, "Lost" and "Rows" */
static void lanes_reaped(uint32_t port,uint32_t walker){
  const uint32_t page=mesh_bridge_pages(M)->link[port];
  struct lane_record *record=page==MESH_ABSENT?NULL:(struct lane_record *)(mesh_at(M,page)+link_lanes_at(page_bytes()));
  walk_mark(walker,0,1);
  for(uint32_t lane=0;record && lane<MESH_LINK_LANES;lane++)
    for(int way=0;way<2;way++){
      struct lane_record *x=record+lane;
      const uint64_t posted=atomic_load_explicit(&x->posted[way],memory_order_acquire);
      for(uint64_t i=atomic_load_explicit(&x->done[way],memory_order_acquire);i<posted && posted-i<=MESH_RING;i++){
        const struct unit *u=x->ring[way]+i%MESH_RING;
        unit_lost(M,u,way);
        if(u->kind==UNIT_PAGE && u->entry)atomic_fetch_add_explicit(&((struct mesh_entry *)((char *)M+u->entry))->count,1,memory_order_seq_cst);
      }
      const uint32_t unposted=run_unposted(x,way);
      unit_lost(M,&x->run[way].unit,way);
      if(unposted && x->run[way].unit.entry)
        atomic_fetch_add_explicit(&((struct mesh_entry *)((char *)M+x->run[way].unit.entry))->count,unposted,memory_order_seq_cst);
      atomic_store_explicit(&x->done[way],posted,memory_order_release);
      x->run[way]=(struct run_record){0};
    }
  walk_mark(walker,0,0);
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
  if(!error)error=link_lapsed(port);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1){
    struct ibv_qp_attr t={.qp_state=IBV_QPS_RTS,.sq_psn=(p->psn+k)&0xffffff};
    if(ibv_modify_qp(p->queues[k].pair,&t,IBV_QP_STATE|IBV_QP_SQ_PSN))error=errno?errno:EIO;
  }
  const struct frames unit=frames_of(page_bytes());
  if(!error)lanes_reaped(port->index,port->index);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)l->qpn[k]=plan>>k&1?p->queues[k].pair->qp_num:UINT32_MAX;
  if(!error)l->cable=link_cable(port);
  if(!error)error=link_lapsed(port);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1){
    const struct frames room=frames_min(l->share,(struct frames){p->queues[k].receive_capacity});
    const struct frames sends=frames_min((struct frames){p->queues[k].send_capacity},(struct frames){MAX(l->share.n,1)});
    struct lane_record *record=(struct lane_record *)(wire.data+link_at(l)+link_lanes_at(page_bytes()))+k;
    memset(record,0,sizeof *record);
    l->lane[k]=(struct lane_state){.window=(uint32_t)MIN(MESH_RING,MAX(2,room.n/unit.n)),.sends=(uint32_t)MIN(MESH_RING,MAX(2,sends.n/unit.n)),
                                   .record=record};
    atomic_thread_fence(memory_order_release);
  }
  struct relay *relay=(struct relay *)(wire.data+link_at(l)+link_relays_at(page_bytes()));
  for(uint32_t i=0;!error && i<MESH_RELAYS;i++){
    uint32_t state=atomic_load_explicit(&relay[i].state,memory_order_acquire);
    while(((state&0xff)==RELAY_LANDING || (state&0xff)==RELAY_READY) &&
          !atomic_compare_exchange_weak_explicit(&relay[i].state,&state,RELAY_FREE,memory_order_acq_rel,memory_order_acquire)){}
    if((state&0xff)==RELAY_SENDING)
      for(uint32_t b=(relay[i].page-l->pool)/M->block;b<(relay[i].page-l->pool)/M->block+relay[i].count && b<MESH_POOL_BLOCKS;b++)
        l->pool_free[b/64]|=UINT64_C(1)<<(b%64);
  }
  return error;
}
static void job_run(struct port *port,int job){
  job_begin(port,job);
  dispatch_async(port->work,^{
    atomic_store(&job_thread,(uintptr_t)pthread_self());
    int result=job==CHILD_OPEN?device_up(&port->device):link_lapsed(port);
    if(!result)result=job==CHILD_OPEN?device_gid(&port->device,port->name):job==CHILD_MAKE?link_make(port):link_connect(port);
    if(!result && job==CHILD_CONNECT)result=port_threads(port);
    if(!result && job==CHILD_OPEN && !port->device.gid_matched){
      const uint8_t *g=port->device.gid.raw;
      port_log(port,"no GID of %s is a link-local address %s holds; GID[%u] %02x%02x:%02x%02x:%02x%02x:%02x%02x:"
               "%02x%02x:%02x%02x:%02x%02x:%02x%02x used\n",port->device.name,port->name,port->device.gid_index,
               g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],g[8],g[9],g[10],g[11],g[12],g[13],g[14],g[15]);
    }
    port->result=result;
    atomic_store_explicit(&port->done,1,memory_order_release);
    wake();
  });
}

/* design/transport.md section 14, "A process a port" */
static uint32_t own_lease(uint32_t *recorded,uint32_t pages){
  if(*recorded==MESH_ABSENT)*recorded=mesh_arena_alloc(&bridge,pages,M->block);
  if(*recorded!=MESH_ABSENT)held_pages+=pages;
  return *recorded;
}
static uint32_t own_zeroed(uint32_t *recorded,uint32_t pages){
  const int fresh=*recorded==MESH_ABSENT;
  const uint32_t page=own_lease(recorded,pages);
  if(fresh && page!=MESH_ABSENT)memset(mesh_at(M,page),0,(size_t)pages*M->pgsz);
  return page;
}
static int message_send(int fd,struct child_message said){
  const char *at=(const char *)&said;
  size_t left=sizeof said;
  while(left){
    const ssize_t sent=fd<0?-1:send(fd,at,left,0);
    if(sent<0 && errno==EINTR)continue;
    if(sent<=0)return -1;
    at+=sent;left-=(size_t)sent;
  }
  return 0;
}
static int child_say(struct port *port,struct child_message said){
  if(!port->deaf && message_send(port->child_fd,said))port->deaf=1;
  return port->deaf;
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
  job_begin(port,job);
  struct child_message said={.kind=(uint32_t)job,.pairing=port->job_pairing,.page=l->page,.pool=l->pool,.lanes=l->plan};
  if(job==CHILD_CONNECT){
    memcpy(said.peer,l->peer,sizeof said.peer);said.lid=port->heard.lid;memcpy(said.gid,port->heard.gid,16);
    said.node=port->peer;
  }
  if(child_say(port,said)){port->result=EPIPE;atomic_store_explicit(&port->done,1,memory_order_release);}
}
static void port_opened(struct port *port,int result);
static void job_finish(struct port *port){
  struct link *l=&port->link;
  if(!port->job || !atomic_load_explicit(&port->done,memory_order_acquire))return;
  const int job=port->job,result=port->result;
  port->job=0;
  if(job==CHILD_OPEN)port_opened(port,result);
  else if(port->job_pairing!=atomic_load_explicit(&l->pairing,memory_order_relaxed)){}
  else if(result){
    port_log(port,"lanes toward node %u not %s: %s; paired again with the port's next child\n",port->peer,
             job==CHILD_MAKE?"made":"connected",strerror(result));
    port_lost(port,result);
  }
  else if(job==CHILD_MAKE){
    struct message said={.kind=LINK,.count=plan_lanes(l->plan)};
    l->state=LINK_MADE;
    for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(l->plan>>k&1){said.pair[k][0]=l->qpn[k];said.pair[k][1]=(l->psn+k)&0xffffff;}
    port_say(port,said,NULL,0);
  }
  else{
    l->state=LINK_UP;
    l->cable=link_cable(port);
    port_log(port,"%u lanes up toward node %u (plan %#x), each lane's share %llu frames\n",plan_lanes(l->plan),port->peer,l->plan,
             (unsigned long long)l->share.n);
    record_changed();
  }
}
static void link_turn(struct port *port){
  struct link *l=&port->link;
  const int held=atomic_load_explicit(&stalled,memory_order_relaxed) || now<port->quiet_at;
  if(!port->greeted)return;
  if(l->state==LINK_UP){
    const uint32_t broken=atomic_exchange_explicit(&l->broken,0,memory_order_acq_rel);
    const int lost=atomic_exchange_explicit(&l->lost,0,memory_order_relaxed);
    if(broken)port_log(port,"lanes %#x toward node %u failed: the link lost\n",broken,port->peer);
    if(broken){port_lost(port,lost?lost:EIO);return;}
  }
  if(port->job || held)return;
  if(l->state==LINK_DOWN){
    if(l->page==MESH_ABSENT)l->page=own_zeroed(mesh_bridge_pages(M)->link+port->index,l->pages);
    if(l->pool==MESH_ABSENT)l->pool=own_lease(mesh_bridge_pages(M)->pool+port->index,MESH_POOL_BLOCKS*M->block);
    if(l->page==MESH_ABSENT || l->pool==MESH_ABSENT){
      port_log(port,"lanes not made: the region holds no room for its link pages and pool; paired again on hearing the peer\n");
      port_lost(port,ENOMEM);
      return;
    }
    l->lanes=1;
    child_job(port,CHILD_MAKE);
  }
  else if(l->state==LINK_MADE && l->heard){l->state=LINK_CONNECTING;child_job(port,CHILD_CONNECT);}
}
static void ports_stalled(void){
  int any=0;
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    const int out=port->job && !atomic_load_explicit(&port->done,memory_order_acquire) && now-port->job_at>=MESH_PLATFORM_NS;
    any|=out;
    if(port->stuck){
      if(!out)port->stuck=0;
      continue;
    }
    if(!out)continue;
    port->stuck=1;
    port_log(port,"its %s out %.1f s: its port concluded around it (design/transport.md)\n",port->job==CHILD_OPEN?"open's job":
             port->job==CHILD_MAKE?"make job":"connect job",(double)(now-port->job_at)/1e9);
    port->open=0;
    port_lost(port,ETIMEDOUT);
  }
  atomic_store_explicit(&stalled,any,memory_order_relaxed);
}
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  struct link *l=&port->link;
  if(port->greeted){
    if(heard->kind==RECORD)record_heard(port,payload,heard->bytes);
    else if(heard->kind!=LINK || l->heard || (l->state!=LINK_DOWN && l->state!=LINK_MADE)){}
    else if(heard->count!=plan_lanes(l->plan)){
      port_log(port,"node %u's LINK names %u lanes, this end's plan %u (plan %#x)\n",port->peer,heard->count,plan_lanes(l->plan),l->plan);
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
  port->peer=heard->node;port->heard=*heard;port->greeted=1;
  const uint32_t theirs=heard->count-1;
  l->plan=link_plan(MIN(port->budget,theirs));
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
  atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
  port_log(port,"paired with node %u (queue pairs for the link: this port's %u, its %u; plan %#x, %u lanes)\n",port->peer,
           port->budget,theirs,l->plan,plan_lanes(l->plan));
  link_set(port,port->peer,MESH_PAIRED,0);
  for(uint32_t i=0;port->greeted && i<known_count;i++)if(known[i].node!=M->node)record_say(port,i);
  if(port->greeted)link_turn(port);
}
static void landings_from(uint32_t peer);
static void port_lost(struct port *port,int64_t code){
  struct link *l=&port->link;
  if(l->lanes && port->peer)landings_from(port->peer);
  quiet_until(&port->quiet_at,now+MESH_PLATFORM_NS);
  atomic_fetch_add_explicit(&l->pairing,1,memory_order_relaxed);l->state=LINK_DOWN;l->heard=0;
  atomic_store_explicit(&l->served,0,memory_order_release);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
  atomic_store_explicit(&l->lost,0,memory_order_relaxed);
  if(port->child){
    child_say(port,(struct child_message){.kind=CHILD_HALT,.pairing=atomic_load(&l->pairing)});
    if(l->lanes && !stop){port->recycle=1;port->open=0;}
  }
  if(port->control>=0){close(port->control);port->control=-1;}
  if(port->greeted)port->next=now;
  if(port->greeted||port->connecting){
    if(stop)port_log(port,"closed toward node %u: this bridge is stopping\n",port->peer);
    else if(port->peer)port_log(port,"node %u lost: code %lld\n",port->peer,(long long)code);
    else port_log(port,"lost: code %lld\n",(long long)code);
  }
  port->greeted=port->connecting=port->writing=0;port->peer=0;port->held=0;port->out_length=port->out_sent=0;
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
  if(port->job || port->stuck || !port->opening || port->control>=0 || now<port->quiet_at || port->recycle ||
     atomic_load_explicit(&stalled,memory_order_relaxed))return;
  if(!port->child && child_spawn(port)){port->opening=0;return;}
  port->opening=0;
  port->interface=if_nametoindex(port->name);
  hello_join(port);
  child_job(port,CHILD_OPEN);
}
static void port_opened(struct port *port,int result){
  if(result){
    port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);
    return;
  }
  port->open=1;
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
      need+=((const struct message *)(const void *)port->in)->bytes;
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
static void link_reserve(void){
  struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  for(uint32_t p=0;p<MESH_PORTS;p++){
    link_page[p]=own_zeroed(own->link+p,link_blocks(page_bytes())*M->block);
    pool_page[p]=own_lease(own->pool+p,MESH_POOL_BLOCKS*M->block);
  }
  if((ones=own_lease(&own->ones,M->block))==MESH_ABSENT)die("the duds' block");
  memset(mesh_at(M,ones),0xff,(size_t)page_bytes());
}
/* design/transport.md section 14, "Plug to link, at the hardware's pace" */
static void port_named(struct port *port,uint32_t index,const char *device){
  *port=(struct port){.index=index,.control=-1,.child_fd=-1};
  port->link.provider.device=&port->device;port->link.pages=link_blocks(page_bytes())*M->block;
  snprintf(port->device.name,sizeof port->device.name,"%s",device);snprintf(port->name,sizeof port->name,"%s",device+5);
}
static void ports_find(void){
  if(finding)return;
  finding=1;find_at=now;
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
    port->opening=1;port->link.page=link_page[port_count];port->link.pool=pool_page[port_count];
    if(!strncmp(port->name,"en",2)){port->family=APPLE_IF_FAM_ETHERNET;port->unit=(uint32_t)strtoul(port->name+2,NULL,10);}
    __atomic_store_n(&port_count,port_count+1,__ATOMIC_SEQ_CST);
    quiet_until(&port->quiet_at,atomic_load(&exit_at));
    link_set(port,0,MESH_STOPPED,0);
  }
}
static void network_heard(void){
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
      if(code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED)ports[p].opening=1;
      else if(code!=KEV_DL_LINK_OFF && code!=KEV_DL_IF_DETACHED){}
      else{ports[p].open=0;port_lost(ports+p,ENETDOWN);}
    }
    unknown|=!known && (code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED);
  }
  if(unknown)find_wanted=1;
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
  IONotificationPortSetDispatchQueue(notifications,dispatch_queue_create("mesh.terminations",DISPATCH_QUEUE_SERIAL));
  IOServiceAddMatchingNotification(notifications,kIOTerminatedNotification,IOServiceMatching("AppleThunderboltRDMAPeerInterface"),
                                   peer_terminated,NULL,&iterator);
  if(arrivals)IOServiceAddMatchingNotification(notifications,kIOFirstMatchNotification,IOServiceMatching("AppleThunderboltRDMAPeerInterface"),
                                               peer_arrived,NULL,&came);
  peer_arrived(NULL,came);
  peer_terminated(NULL,iterator);
}
static void terminations_turn(void){
  const uint32_t lost=atomic_exchange(&terminated,0);
  for(uint32_t p=0;p<port_count;p++)if(lost>>p&1){
    port_log(ports+p,"its peer interface terminating: its link lost\n");
    port_lost(ports+p,ENXIO);
  }
  const uint32_t came=atomic_exchange(&arrived,0);
  for(uint32_t p=0;p<port_count;p++)if(came>>p&1 && ports[p].control<0)ports[p].opening=1;
  if(atomic_exchange(&arrived_elsewhere,0))find_wanted=1;
}

/* design/transport.md section 14, "A process a port"; section 15, "The children" */
static void child_reaped(struct port *port);
static void names_publish(void);
static void walks_even(uint32_t port);
static void relays_sent(uint32_t port);
static int child_spawn(struct port *port){
  int pair[2],on=1;
  struct mesh_holder *holders=mesh_bridge_pages(M)->child;
  const uint32_t device=fnv(port->device.name,strlen(port->device.name));
  uint32_t entry=2*MESH_PORTS;
  for(uint32_t k=2*MESH_PORTS;!port->elsewhere && k--;){
    const pid_t pid=(pid_t)holders[k].pid;
    if(!mesh_holds(pid,holders[k].start))entry=k;
    else if(holders[k].device==device && !watch((uint64_t)pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,TAG_CHILD|port->index)){
      port_log(port,"process %d, a child of its device from an earlier launch, holds its queue pairs until its exit; its own child spawned "
               "after it\n",(int)pid);
      port->elsewhere=pid;port->elsewhere_start=holders[k].start;port->late_from=port->late_at=now+2*MESH_PLATFORM_NS;port->stage=0;
    }
  }
  if(port->elsewhere)return EBUSY;
  if(entry==2*MESH_PORTS)return EBUSY;
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
    error=fcntl(pair[0],F_SETFL,O_NONBLOCK)?errno:posix_spawn(&pid,self_path,&actions,&attributes,arguments,*_NSGetEnviron());
    posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attributes);
    close(pair[1]);
    if(error)close(pair[0]);
  }
  if(error){port_log(port,"its child not spawned: %s\n",strerror(error));return error;}
  port->child=pid;port->child_fd=pair[0];port->child_at=now;port->child_held=0;port->ending=port->deaf=port->hung=port->stage=0;
  holders[entry]=(struct mesh_holder){(uint32_t)pid,device,mesh_started(pid)};
  port->holder=entry;
  port_log(port,"its child %d spawned\n",(int)pid);
  if(watch((uint64_t)pair[0],EVFILT_READ,EV_ADD,0,TAG_CHILD|port->index)){close(port->child_fd);port->child_fd=-1;}
  if(watch((uint64_t)pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,TAG_CHILD|port->index)){child_reaped(port);return ECHILD;}
  return 0;
}
static void child_heard(struct port *port){
  struct link *l=&port->link;
  struct child_message heard;
  int got;
  while((got=message_read(port->child_fd,port->child_in,&port->child_held,&heard))>0){
    const int current=heard.pairing==atomic_load_explicit(&l->pairing,memory_order_relaxed);
    const int job=(int)heard.kind-CHILD_OPENED+CHILD_OPEN;
    if(job>=CHILD_OPEN && job<=CHILD_CONNECT){
      if(port->job!=job || heard.pairing!=port->job_pairing)continue;
      port->result=heard.result;
      if(!heard.result && job!=CHILD_CONNECT){port->lid=heard.lid;memcpy(port->gid,heard.gid,16);}
      if(!heard.result && job==CHILD_OPEN){port->bandwidth=heard.bandwidth;port->budget=heard.lanes;}
      if(!heard.result && job==CHILD_MAKE){memcpy(l->qpn,heard.qpn,sizeof l->qpn);l->psn=heard.psn;l->share=(struct frames){heard.share};}
      atomic_store_explicit(&port->done,1,memory_order_release);
    }
    else if(heard.kind==CHILD_NAMED)names_publish();
    else if(heard.kind==CHILD_FAULTED){port_log(port,"the provider's fault absorbed in the port's child: its link lost\n");port->open=0;port_lost(port,ENETDOWN);}
    else if(heard.kind==CHILD_BROKEN && current && l->state==LINK_UP){
      if(heard.result)atomic_store_explicit(&l->lost,heard.result,memory_order_relaxed);
      atomic_fetch_or(&l->broken,heard.broken);
    }
  }
  if(got<0){close(port->child_fd);port->child_fd=-1;}
}
static uint64_t child_seen(const struct port *port){
  return MAX(atomic_load_explicit(&mesh_bridge_pages(M)->progress[port->index],memory_order_relaxed),port->child_at);
}
static int child_may_end(const struct port *port){
  return !port->job && now>=atomic_load(&port->quiet_at) && now>=atomic_load(&exit_at) && !atomic_load_explicit(&stalled,memory_order_relaxed);
}
static void child_turn(struct port *port){
  const uint64_t seen=child_seen(port);
  if(port->child && !port->ending && !port->hung && (port->deaf || now>=seen+MESH_PLATFORM_NS)){
    port_log(port,"its child %d %s, its last progress %.1f s before: its link lost, and the child ended once its windows have passed "
             "(design/transport.md section 16)\n",(int)port->child,port->deaf?"took no whole message":"made no progress",
             (double)(int64_t)(now-seen)/1e9);
    port->hung=1;port->open=0;port_lost(port,ETIMEDOUT);port->recycle=1;
  }
  const pid_t late=port->ending?port->child:port->elsewhere;
  if(late && port->stage<2 && now>=port->late_at && child_may_end(port) && (port->ending || mesh_holds(late,port->elsewhere_start))){
    port_log(port,"process %d, %s, has not ended: %s (design/transport.md section 16)\n",(int)late,
             port->ending?"its child asked to end":"a child of an earlier launch",port->stage?"SIGKILL":"SIGSTOP, and SIGKILL once quiet");
    kill(late,port->stage++?SIGKILL:SIGSTOP);port->late_at=now+MESH_PLATFORM_NS;
  }
  if(!port->recycle || !port->child || port->ending || !child_may_end(port))return;
  child_say(port,(struct child_message){.kind=CHILD_END});
  port->ending=1;port->late_from=port->late_at=now+MESH_PLATFORM_NS+MESH_PAUSE_NS;port->stage=0;
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
  const int asked=port->ending && !port->hung;
  port->ending=port->recycle=0;
  if(asked)port_log(port,"its child %d ended\n",(int)pid);
  else{
    port_log(port,"its child %d ended %s (%s %d)\n",(int)pid,port->hung?"after its hang":"unbidden",
             WIFSIGNALED(status)?"signal":"status",WIFSIGNALED(status)?WTERMSIG(status):WEXITSTATUS(status));
    port->open=0;
    if(!port->hung)port_lost(port,ECHILD);
  }
  if(port->job){port->result=ECHILD;atomic_store_explicit(&port->done,1,memory_order_release);}
  lanes_reaped(port->index,MESH_PORTS);
  walks_even(port->index);
  relays_sent(port->index);
  unreleased=1;
  l->state=LINK_DOWN;l->lanes=l->heard=0;l->share=(struct frames){0};l->plan=0;
  memset(l->qpn,0,sizeof l->qpn);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
  port->opening=1;
}
/* design/transport.md section 14, "A registration" and "Walks" */
static int names_order(const void *a,const void *b){
  const uint64_t *x=a,*y=b;
  return x[0]!=y[0]?(x[0]>y[0])-(x[0]<y[0]):(x[1]>y[1])-(x[1]<y[1]);
}
static int names_read(void){
  uint64_t (*read)[2]=NULL;
  uint32_t count=0,room=0;
  for(uint32_t c=0;c<MESH_CLIENTS;c++){
    const struct mesh_client *slot=mesh_slot(M,c);
    if(!owned(atomic_load_explicit(&slot->owner,memory_order_acquire)))continue;
    uint32_t tables=mesh_rows(M);
    for(const struct mesh_plan *t=queue_first(slot);t && tables--;t=queue_next(t)){
      if(!t->name[0] && !t->name[1])continue;
      int seen=0;
      for(uint32_t i=0;i<count && !seen;i++)seen=read[i][0]==t->name[0] && read[i][1]==t->name[1];
      if(!seen && count==room){
        uint64_t (*grown)[2]=realloc(read,(room?2*room:16)*sizeof *read);
        if(!grown){free(read);return 0;}
        read=grown;room=room?2*room:16;
      }
      if(!seen){read[count][0]=t->name[0];read[count][1]=t->name[1];count++;}
    }
  }
  if(read)qsort(read,count,sizeof *read,names_order);
  if(count==names_held && (!count || !memcmp(read,names,(size_t)count*sizeof *read))){free(read);return 0;}
  free(names);
  names=read;names_held=count;
  return 1;
}
static void names_publish(void){
  walk_mark(MESH_PORTS,0,1);
  const int named=names_read();
  walk_mark(MESH_PORTS,0,0);
  if(named)record_changed();
}
static void walks_even(uint32_t port){
  for(int way=0;way<2;way++){
    _Atomic uint64_t *word=&mesh_bridge_pages(M)->walking[port][way];
    if(atomic_load_explicit(word,memory_order_acquire)&1)atomic_fetch_add_explicit(word,1,memory_order_release);
  }
}
static void relays_sent(uint32_t port){
  for(uint32_t p=0;p<MESH_PORTS;p++){
    const uint32_t page=mesh_bridge_pages(M)->link[p];
    struct relay *relay=page==MESH_ABSENT?NULL:(struct relay *)(mesh_at(M,page)+link_relays_at(page_bytes()));
    for(uint32_t i=0;relay && i<MESH_RELAYS;i++)
      if(atomic_load_explicit(&relay[i].state,memory_order_acquire)==relay_sending(port))
        atomic_store_explicit(&relay[i].sent,relay[i].count,memory_order_release);
  }
}
/* design/transport.md section 14, "Walks"; section 15, "The library" */
static void slots_vacate(void){
  static struct mesh_walk walk[MESH_CLIENTS];
  static int settled[MESH_CLIENTS];
  for(uint32_t c=0;c<MESH_CLIENTS;c++){
    struct mesh_client *slot=mesh_slot(M,c);
    uint64_t queue=atomic_load_explicit(&slot->queue,memory_order_acquire);
    if(atomic_load_explicit(&slot->owner,memory_order_seq_cst) || !queue){settled[c]=0;continue;}
    if(settled[c]){
      if(mesh_walked(M,walk+c) && atomic_compare_exchange_strong_explicit(&slot->queue,&queue,0,memory_order_acq_rel,memory_order_acquire))settled[c]=0;
      continue;
    }
    int balanced=1;
    uint32_t tables=mesh_rows(M);
    for(const struct mesh_plan *t=queue_first(slot);balanced && t && tables--;t=queue_next(t))balanced=mesh_plan_balanced(t);
    if(balanced){mesh_walks(M,walk+c);settled[c]=1;}
  }
}
/* design/transport.md section 2, the client slots; section 14, "Rows" */
static void clients_turn(void){
  const pid_t self=getpid();
  int changed=0;
  walk_mark(MESH_PORTS,0,1);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    uint64_t owner=atomic_load(&client->owner);
    if(owner && (pid_t)(uint32_t)owner==self)continue;
    if(owner && owner!=owners[s] && !(owner&MESH_RETIRING) && watch((uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,s)<0 &&
       errno==ESRCH){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    changed|=owner!=owners[s];owners[s]=owner;
  }
  unreleased|=changed;
  slots_vacate();
  if(unreleased)unreleased=mesh_retired_release(M,0)!=0;
  const int named=names_read();
  walk_mark(MESH_PORTS,0,0);
  if(named)record_changed();
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
  const int dictionary=list && CFGetTypeID(list)==CFDictionaryGetTypeID();
  CFTypeRef named=dictionary?CFDictionaryGetValue(list,CFSTR("name")):NULL,sized=dictionary?CFDictionaryGetValue(list,CFSTR("region_bytes")):NULL;
  int64_t value=0;
  if(!dictionary)refused="not a property list of a configuration (rdma/bridge_configuration.py writes it)";
  else if(!named || CFGetTypeID(named)!=CFStringGetTypeID() || !CFStringGetCString(named,name,(CFIndex)room,kCFStringEncodingUTF8))refused="its name is missing";
  else if(!sized || CFGetTypeID(sized)!=CFNumberGetTypeID() || !CFNumberGetValue(sized,kCFNumberSInt64Type,&value) || value<=0)
    refused="its region_bytes, a positive number of bytes, is missing";
  if(list)CFRelease(list);
  *region_bytes=(uint64_t)value;
  return refused;
}

/* design/transport.md section 14, "A process a port"; section 15, "The children" */
static void fault_heard(int number,siginfo_t *info,void *context){
  const uintptr_t pc=__darwin_arm_thread_state64_get_pc(((ucontext_t *)context)->uc_mcontext->__ss),page=trunc_page((uintptr_t)info->si_addr);
  if(pc>=provider[0] && pc<provider[1]){
    void *got=mmap((void *)page,vm_page_size,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
    if(got!=MAP_FAILED && got!=(void *)page)munmap(got,vm_page_size);
    if(got==(void *)page)atomic_store(&mapped,page);
    const int lands=atomic_load(&mapped)==page;
    if(!lands && (uintptr_t)pthread_self()==atomic_load(&job_thread))atomic_store(&job_parked,1);
    port_halt(ports,&faulted);
    if(lands)return;
    for(;;)pause();
  }
  struct sigaction fallback={.sa_handler=SIG_DFL};
  sigaction(number,&fallback,NULL);
}
static __attribute__((noreturn)) void child_leave(struct port *port,int asked){
  port_halt(port,NULL);
  atomic_store_explicit(&port->closing,1,memory_order_release);
  served_wake(port);
  const uint64_t start=clock_ns(),bound=start+(asked?1:2)*MESH_PLATFORM_NS;
  for(;;){
    const uint64_t at=clock_ns(),windows=MAX(atomic_load(&exit_at),atomic_load(&port->quiet_at));
    const uint64_t until=MIN(bound,MAX(asked?start:start+MESH_PLATFORM_NS,windows)),rest_ns=MIN(until-at,MESH_PAUSE_NS);
    atomic_store_explicit(&mesh_bridge_pages(M)->progress[port->index],at,memory_order_relaxed);
    if(at>=until)break;
    struct timespec rest={(time_t)(rest_ns/1000000000u),(long)(rest_ns%1000000000u)};
    nanosleep(&rest,NULL);
  }
  exit(0);
}
static void child_command(struct port *port,const struct child_message *heard){
  struct link *l=&port->link;
  const int job=(int)heard->kind;
  if(job==CHILD_HALT){port_halt(port,NULL);child_halted=0;return;}
  if(job==CHILD_END)child_leave(port,1);
  if(job<CHILD_OPEN || job>CHILD_CONNECT)return;
  int refused=job!=CHILD_OPEN && (child_halted || atomic_load(&terminated) || atomic_load(&faulted))?ECANCELED:0;
  if(!refused && job==CHILD_CONNECT){
    memcpy(l->peer,heard->peer,sizeof l->peer);port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
    port->peer=heard->node;
  }
  if(refused){
    message_send(3,(struct child_message){.kind=(uint32_t)(job-CHILD_OPEN+CHILD_OPENED),.pairing=heard->pairing,.result=refused});
    return;
  }
  child_asked=heard->pairing;
  if(job==CHILD_MAKE){l->page=heard->page;l->pool=heard->pool;l->plan=heard->lanes;}
  job_run(port,job);
}
static void child_jobs(struct port *port){
  struct link *l=&port->link;
  if(!port->job || !atomic_load_explicit(&port->done,memory_order_acquire))return;
  const int job=port->job;
  port->job=0;
  struct child_message said={.kind=(uint32_t)(job-CHILD_OPEN+CHILD_OPENED),.pairing=child_asked,.result=port->result};
  if(!said.result && job!=CHILD_OPEN && port->job_pairing!=atomic_load(&l->pairing))said.result=ECANCELED;
  if(!said.result && job!=CHILD_CONNECT){
    port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
    said.lid=port->lid;memcpy(said.gid,port->gid,16);
  }
  if(!said.result && job==CHILD_OPEN){said.bandwidth=port->bandwidth=port_bandwidth(&port->device.port);said.lanes=port->device.max_pairs;}
  else if(!said.result && job==CHILD_MAKE){
    for(uint32_t k=0;k<MESH_LINK_LANES;k++)said.qpn[k]=l->plan>>k&1?l->provider.queues[k].pair->qp_num:UINT32_MAX;
    said.psn=l->provider.psn;said.share=(uint32_t)l->share.n;
  }
  else if(!said.result){
    atomic_store(&l->served,l->plan);
    if(atomic_load(&l->pairing)!=port->job_pairing){atomic_store(&l->served,0);said.result=ECANCELED;}
    served_wake(port);
  }
  message_send(3,said);
}
static int child_main(int argc,char **argv){
  signal(SIGPIPE,SIG_IGN);
  const unsigned long index=argc>=6?strtoul(argv[4],NULL,10):MESH_PORTS;
  if(index>=MESH_PORTS){say("child of %s: port %lu is past the bridge's, or its command line short: it ends\n",argc>3?argv[3]:"?",index);return 1;}
  for(uint32_t i=0;i<_dyld_image_count();i++){
    const char *image=_dyld_get_image_name(i);
    unsigned long size=0;
    const uint8_t *text=image && strstr(image,"/libthunderboltrdma.")?getsegmentdata((const void *)_dyld_get_image_header(i),"__TEXT",&size):NULL;
    if(text){provider[0]=(uintptr_t)text;provider[1]=(uintptr_t)text+size;}
  }
  struct sigaction faults={.sa_sigaction=fault_heard,.sa_flags=SA_SIGINFO|SA_ONSTACK};
  sigemptyset(&faults.sa_mask);
  if(sigaction(SIGSEGV,&faults,NULL) || sigaction(SIGBUS,&faults,NULL))die("child faults");
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
  port->work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL);port->link.page=port->link.pool=MESH_ABSENT;
  ones=(uint32_t)strtoul(argv[5],NULL,10);
  __atomic_store_n(&port_count,1,__ATOMIC_SEQ_CST);
  if((events=kqueue())<0 || watch(0,EVFILT_USER,EV_ADD|EV_CLEAR,0,UINT64_MAX) || watch(3,EVFILT_READ,EV_ADD,0,UINT64_MAX))die("child events");
  terminations_watch(0);
  uint64_t passed[2]={0};
  uint32_t serving=0;
  for(;;){
    now=clock_ns();
    const int fault=atomic_load(&faulted)!=0;
    if(port->job && atomic_load(&job_parked)){port->result=ECANCELED;atomic_store_explicit(&port->done,1,memory_order_release);}
    child_jobs(port);
    const uint64_t sent=atomic_load_explicit(&port->progress[0].passes,memory_order_relaxed),got=atomic_load_explicit(&port->progress[1].passes,memory_order_relaxed);
    const uint32_t served=atomic_load_explicit(&port->link.served,memory_order_relaxed);
    if(!served || served!=serving || (sent!=passed[0] && got!=passed[1])){
      serving=served;passed[0]=sent;passed[1]=got;
      atomic_store_explicit(&mesh_bridge_pages(M)->progress[port->index],now,memory_order_relaxed);
    }
    const uint32_t broken=atomic_exchange_explicit(&port->link.broken,0,memory_order_acq_rel);
    if(broken)message_send(3,(struct child_message){.kind=CHILD_BROKEN,.pairing=child_asked,.broken=broken,
                                                   .result=atomic_exchange_explicit(&port->link.lost,0,memory_order_relaxed)});
    if(atomic_exchange(&terminated,0))child_halted=1;
    if(atomic_exchange_explicit(&naming,0,memory_order_acq_rel))message_send(3,(struct child_message){.kind=CHILD_NAMED});
    if(fault){message_send(3,(struct child_message){.kind=CHILD_FAULTED});child_leave(port,0);}
    struct kevent64_s ready[4];
    struct timespec tick={(time_t)(MESH_PAUSE_NS/1000000000u),(long)(MESH_PAUSE_NS%1000000000u)};
    const int count=kevent64(events,NULL,0,ready,4,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++)if(ready[i].filter==EVFILT_READ && ready[i].ident==3){
      struct child_message heard;
      int got;
      while((got=message_read(3,port->child_in,&port->child_held,&heard))>0)child_command(port,&heard);
      if(got<0)child_leave(port,0);
    }
  }
}

/* design/transport.md section 2; section 15, "The region" */
static int region_open(const struct hdr *geometry,int *out){
  int fd=shm_open(MESH_NAME,O_RDWR,0);
  struct stat shape={0};
  struct hdr *old=NULL;
  const char *failed=fd<0?errno==ENOENT?NULL:"opened":fstat(fd,&shape)?"read":(size_t)shape.st_size<sizeof *old?NULL:
                     (old=mmap(NULL,(size_t)shape.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0))==MAP_FAILED?"mapped":NULL;
  if(failed){say("bridge: region %s not %s (%s): this bridge ends without touching it\n",MESH_NAME,failed,strerror(errno));exit(1);}
  if(fd>=0){
    if(old && old->magic==MESH_MAGIC){
      if(__atomic_load_n(&old->version,__ATOMIC_ACQUIRE)==MESH_LAYOUT && !memcmp(&old->pgsz,&geometry->pgsz,3*sizeof old->pgsz) &&
         !memcmp(&old->spare,&geometry->spare,offsetof(struct hdr,boot)-offsetof(struct hdr,spare))){M=old;*out=fd;return 1;}
      __atomic_store_n(&old->version,0,__ATOMIC_SEQ_CST);
      say("bridge: region %s is expired (another layout, or released): this bridge ends, and its heartbeat unlinks it once no "
          "holder of it lives and opens a bridge again\n",MESH_NAME);
      exit(EX_TEMPFAIL);
    }
    if(old)munmap(old,(size_t)shape.st_size);
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
  for(uint32_t p=0;p<MESH_PORTS;p++)own->link[p]=own->pool[p]=MESH_ABSENT;
  own->ones=own->map=MESH_ABSENT;own->map_pages=0;
  *out=fd;
  return 0;
}
static void region_adopt(void){
  int earlier=0;
  for(uint32_t k=0;k<2*MESH_PORTS;k++)earlier|=mesh_holds((pid_t)mesh_bridge_pages(M)->child[k].pid,mesh_bridge_pages(M)->child[k].start);
  for(uint32_t p=earlier?MESH_PORTS:0;p<=MESH_PORTS;p++)walks_even(p);
  for(uint32_t p=0;!earlier && p<MESH_PORTS;p++){lanes_reaped(p,MESH_PORTS);relays_sent(p);}
  walk_mark(MESH_PORTS,0,1);
  mesh_tuples_lost(M,0);
  struct mesh_link_info *links=mesh_links(M);
  for(uint32_t p=0;p<MESH_PORTS;p++){
    atomic_store(&links[p].peer,0);atomic_store(&links[p].phase,MESH_UNKNOWN);atomic_store(&links[p].code,0);atomic_store(&links[p].bandwidth,0);
  }
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    const uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
    if(owner&MESH_RETIRING)mesh_retire(M,owner);
  }
  mesh_retired_release(M,0);
  walk_mark(MESH_PORTS,0,0);
  const struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  if(own->map!=MESH_ABSENT){
    map_page=own->map;map_pages=own->map_pages;
    held_pages+=map_pages;
  }
}
/* design/transport.md section 16, "The watchdog" */
static void turned(void){
  struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  uint64_t from=finding?find_at+MESH_PLATFORM_NS:UINT64_MAX;
  for(uint32_t p=0;p<port_count;p++){
    const struct port *port=ports+p;
    if(port->job && !atomic_load_explicit(&port->done,memory_order_acquire))from=MIN(from,port->job_at+MESH_PLATFORM_NS);
    if(port->child)from=MIN(from,port->ending?port->late_from:child_seen(port)+MESH_PLATFORM_NS);
    if(port->elsewhere)from=MIN(from,port->late_from);
  }
  atomic_store_explicit(&own->hung_from,from<=now && !stop?from:0,memory_order_relaxed);
  atomic_store_explicit(&own->turned,now,memory_order_relaxed);
  atomic_store_explicit(&own->turned_by,own->bridge_start,memory_order_release);
}
/* design/transport.md section 15, "Windows" */
static inline void due_at(uint64_t *due,uint64_t at){ if(at>now && at<*due)*due=at; }
static uint64_t turn_due(uint64_t liveness){
  uint64_t due=MIN(liveness,now+MESH_PAUSE_NS);
  if(find_wanted)due_at(&due,windows_end());
  due_at(&due,atomic_load(&exit_at));
  for(uint32_t p=0;p<port_count;p++){
    const struct port *port=ports+p;
    due_at(&due,atomic_load(&port->quiet_at));
    if(port->open && port->control<0)due_at(&due,port->next);
    if(port->job)due_at(&due,port->job_at+MESH_PLATFORM_NS);
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
  uuid_t uuid;struct timespec bound={5,0};
  if(gethostuuid(uuid,&bound)){say("host identity: %s; this bridge's node is drawn at random\n",strerror(errno));arc4random_buf(uuid,sizeof uuid);}
  uint32_t node=fnv(uuid,sizeof uuid)&0x7fffffffu;node+=!node;
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
    if(addresses>=0)close(addresses);
    addresses=-1;
  }
  int fd=-1;
  const int adopted=region_open(&geometry,&fd);
  if(adopted)region_adopt();
  M->node=node;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);
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
  record_changed();
  if(!(uint32_t)atomic_load(&M->map_at))die("the map's lease");
  link_reserve();
  int watched[4]={listener,hello,network,addresses};
  for(int i=0;i<4 && watched[i]>=0;i++)if(watch((uint64_t)watched[i],EVFILT_READ,EV_ADD,0,UINT64_MAX))die("bridge events");
  if(watch(0,EVFILT_USER,EV_ADD|EV_CLEAR,0,UINT64_MAX))die("bridge events");
  mesh_bridge_pages(M)->bridge_start=mesh_started(getpid());
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();M->magic=MESH_MAGIC;
  terminations_watch(1);
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
      job_finish(port);
      port_open(port);
      port_hello(port);
      link_turn(port);
      child_turn(port);
    }
    if(now>=liveness)liveness=now+MESH_PAUSE_NS;
    clients_turn();
    turned();
    struct kevent64_s ready[16];
    const uint64_t due=turn_due(liveness),at=clock_ns(),rest=due>at?due-at:0;
    struct timespec tick={(time_t)(rest/1000000000u),(long)(rest%1000000000u)};
    int count=kevent64(events,NULL,0,ready,16,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL){stop=1;stop_reason=(int64_t)e->ident;}
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_CHILD)){
        struct port *port=(e->udata&0xff)<port_count?ports+(e->udata&0xff):NULL;
        if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->child)child_reaped(port);
        else if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->elsewhere){
          port->elsewhere=0;port->opening=1;lanes_reaped(port->index,MESH_PORTS);walks_even(port->index);relays_sent(port->index);
        }
        else if(port && e->filter==EVFILT_READ && (int)e->ident==port->child_fd)child_heard(port);
      }
      else if(e->filter==EVFILT_PROC){
        walk_mark(MESH_PORTS,0,1);
        for(uint32_t s=0;s<MESH_CLIENTS;s++){
          uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
          if(owner && !(owner&MESH_RETIRING) && (uint32_t)owner==(uint32_t)e->ident)mesh_retire(M,owner);
        }
        walk_mark(MESH_PORTS,0,0);
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
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard();
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
  say("bridge node %u: stopping (signal %lld%s)\n",node,(long long)stop_reason,stop_reason==SIGHUP?": its heartbeat restarts it":
      stop_reason==SIGTERM?": its heartbeat's release or stop":"");
  for(uint32_t p=0;p<port_count;p++){
    if(ports[p].child)child_say(ports+p,(struct child_message){.kind=CHILD_END});
    if(ports[p].stage==1)kill(ports[p].ending?ports[p].child:ports[p].elsewhere,SIGKILL);
    if(ports[p].control>=0){close(ports[p].control);ports[p].control=-1;}
  }
  walk_mark(MESH_PORTS,0,1);mesh_tuples_lost(M,0);walk_mark(MESH_PORTS,0,0);
  atomic_store(&M->bridge_pid,0);
  return 0;
}
