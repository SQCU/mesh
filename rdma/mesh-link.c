#include "mesh-bridge.h"
#include <spawn.h>
#include <sys/wait.h>
#include <crt_externs.h>
#include <libproc.h>
#include <sysexits.h>
#include <mach/mach_time.h>

/* mesh-verbs.h: the stop flag every loop reads */
_Atomic sig_atomic_t stop;

struct hdr *M;
struct wire_mapping wire;
struct port ports[MESH_PORTS];
uint32_t port_count;
int events=-1,listener=-1,hello=-1,network=-1,unreleased,finding;
uint64_t now,sequence;
char host[MESH_NAME_BYTES];
struct mesh_ctx bridge={.fd=-1};
static dispatch_queue_t finder;
/* design/transport.md: the devices to be read again (an interface that no port is attached or its link came on), read
   on the event loop's turn once no window stands and no call is stuck (ports_find) */
static int find_wanted;
/* design/transport.md B12: a port's job out past MESH_PLATFORM_NS, during which no port dispatches a driver
   call and the devices are not read (ports_stalled's, each event-loop turn); and the ports whose progress threads did
   not hold for a settle or a guarded change within MESH_PLATFORM_NS (`unguarded`, a bit a port: port_wait), concluded
   alike */
static _Atomic int stalled;
static _Atomic uint32_t unguarded;
/* design/transport.md B9b: the window of a peer interface's termination (`exit_at`), as every port's quiet
   window holds, before which the bridge does not exit nor read the devices */
static _Atomic uint64_t exit_at;
/* a window's end moved to `until` where it is earlier, by any thread (the event loop's conclusions, a termination's
   queue, B9b), never earlier */
static void quiet_until(_Atomic uint64_t *at,uint64_t until){
  uint64_t was=atomic_load(at);
  while(was<until && !atomic_compare_exchange_weak(at,&was,until)){}
}
/* the latest of the exit's window and every port's (B9, B9b): the stop's hold, and the devices' read (ports_find) */
static uint64_t windows_end(void){
  uint64_t latest=atomic_load(&exit_at);
  for(uint32_t p=0;p<port_count;p++){const uint64_t quiet=atomic_load(&ports[p].quiet_at);latest=MAX(latest,quiet);}
  return latest;
}
_Atomic int found;
static char devices[MESH_PORTS][32];
static uint32_t device_count;
static const struct in6_addr everyone={{{0xff,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}};

static uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_MONOTONIC); }
/* design/transport.md section 9: the bridge's boot ledger beside its configuration (`boots`): the boot it started in
   (kern.bootsessionuuid) and how many boots in a row ended while a bridge was within BOOTS_SETTLE_NS of its start, no
   stop of its having cleared the ledger (a panic, a power loss, a reset); n of them hold every device unopened 60 2^(n-1)
   s, at most an hour, so a fault the bridge's start or pairing meets recurs at most that often and the node stays
   reachable meanwhile; the bridge's stop clears it, and so does BOOTS_SETTLE_NS of running */
#define BOOTS_SETTLE_NS (UINT64_C(600)*1000000000u)
static char boots_path[PATH_MAX];
static int boots_set;
static uint64_t boots_since;
static void boots_clear(void){ if(boots_set && !unlink(boots_path))boots_set=0; }
static __attribute__((unused)) void boots_hold(const char *configuration){
  const char *slash=strrchr(configuration,'/');
  snprintf(boots_path,sizeof boots_path,"%.*s/boots",slash?(int)(slash-configuration):1,slash?configuration:".");
  char boot[64]={0},prior[64]={0};
  size_t size=sizeof boot-1;
  if(sysctlbyname("kern.bootsessionuuid",boot,&size,NULL,0)){
    say("bridge boots: this boot's identity (kern.bootsessionuuid): %s; no boot is counted\n",strerror(errno));return;}
  unsigned abrupt=0;
  FILE *in=fopen(boots_path,"r");
  if(in){
    if(fscanf(in,"%63s %u",prior,&abrupt)!=2){prior[0]=0;abrupt=0;}
    fclose(in);
    if(strcmp(prior,boot))abrupt++;
  }
  char next[PATH_MAX+8];
  snprintf(next,sizeof next,"%s.new",boots_path);
  FILE *out=fopen(next,"w");
  if(out){
    fprintf(out,"%s %u\n",boot,abrupt);
    const int kept=!fflush(out) && !fsync(fileno(out));
    if(!fclose(out) && kept && !rename(next,boots_path))boots_set=1;
  }
  if(!boots_set)say("bridge boots: %s not written (%s); no boot is counted\n",boots_path,strerror(errno));
  if(abrupt){
    const uint64_t hold=MIN(UINT64_C(60)<<MIN(abrupt-1,6u),UINT64_C(3600));
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
uint64_t memory;
static int64_t stop_reason;
static void farewell_text(int64_t code,char *out,size_t room){
  if(code==SIGHUP)snprintf(out,room,"SIGHUP: its heartbeat restarts it");
  else if(code==SIGTERM)snprintf(out,room,"SIGTERM: its heartbeat's release or stop");
  else if(code>0)snprintf(out,room,"signal %lld",(long long)code);
  else snprintf(out,room,"no reason given");
}
struct known known[MESH_KNOWN];
void wake(void){
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}
/* design/transport.md: a port halted from any thread (a peer interface's termination heard, a raiser's wait
   at its bound; atomics alone): its pairing lost, so no job of it makes a further call (B10), then no lane of its link
   served, both sequentially consistent (link_turn's connect stores served and reads the pairing again), and its bit set
   in `which` for the event loop's conclusion (terminations_turn, ports_stalled) */
_Static_assert(MESH_PORTS<=32,"a bit a port");
static void port_halt(struct port *port,_Atomic uint32_t *which){
  atomic_fetch_add(&port->link.pairing,1);
  atomic_store(&port->link.served,0);
  atomic_fetch_or(which,UINT32_C(1)<<port->index);
}

/* design/HETEROGENEOUS-LINKS.md section 10: in a port's child, the most mach ticks a link goes without a SEND while it
   holds no stream (`alive_ticks[0]`, MESH_PLATFORM_NS/5) and while it holds one (`alive_ticks[1]`, MESH_KEEPALIVE_NS):
   link_alive */
static uint64_t alive_ticks[2];

/* design/transport.md: a port's link pages: a block no lane names (the keepalive lane's before the keepalive took the
   streams' small lanes, kept so the region's layout stands), then the spare page the data lanes' spare receives land in
   (link_spare),
   leased for every port the region has, at the bridge's start beside the map's (link_reserve), else as its link first
   pairs (link_turn) */
static uint32_t link_pages(void){ return 2*M->block; }
static uint64_t link_block(const struct port *port){ return (uint64_t)port->link.page*M->pgsz; }
/* a lane's failure, found by a progress thread: served no more, and the link lost by the event loop on reading
   `broken` (link_turn) */
static void lane_lost(struct port *port,uint32_t lane,int64_t code){
  struct link *l=&port->link;
  if(atomic_fetch_and_explicit(&l->served,~(UINT32_C(1)<<lane),memory_order_acq_rel)>>lane&1)
    say("link %u (%s): lane %u failed: %s; the link toward node %u lost\n",port->index,port->name,lane,
        strerror((int)(code>0?code:EIO)),port->peer);
  atomic_fetch_or_explicit(&l->broken,UINT32_C(1)<<lane,memory_order_release);
  wake();
}

/* design/transport.md section 14: port p's stream as the region tells it, and a word of the region */
static inline struct mesh_stream *stream_of(const struct port *port,uint32_t s){ return mesh_stream_at(M,port->index*MESH_LINK_STREAMS+s); }
static inline _Atomic uint64_t *stream_words(uint64_t offset){ return (_Atomic uint64_t *)((char *)M+offset); }
/* design/transport.md section 14: entry e's slot of a time ring (mesh.h struct mesh_stream `posted_at`, `landed_at`) given
   the ticks `at`: its first word 0, then `at`, then 1 + e, each store ordered after the one before */
static inline void stream_time(_Atomic uint64_t *ring,uint64_t e,uint64_t at){
  _Atomic uint64_t *entry=ring+2*(e%MESH_STREAM_TIMES);
  atomic_store_explicit(entry,0,memory_order_relaxed);
  atomic_store_explicit(entry+1,at,memory_order_release);
  atomic_store_explicit(entry,1+e,memory_order_release);
}
/* the bytes of a page, a large lane's SEND and receive */
static inline uint64_t page_bytes(void){ return (uint64_t)M->block*M->pgsz; }
/* the arena offset of a port's spare page: its link block's second page, which receives what no holding presented */
static uint64_t link_block(const struct port *port);
static inline uint64_t link_spare(const struct port *port){ return link_block(port)+page_bytes(); }
/* design/transport.md section 14: whether bytes [at, at + bytes) of the region touch a page the bridge holds itself
   (the duds' block, a port's link pages or lists, the map), which no holding may name */
static int bridge_touched(uint64_t at,uint64_t bytes){
  const struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  const uint64_t pgsz=M->pgsz,first=(at-M->data_off)/pgsz,last=(at-M->data_off+(bytes?bytes:1)-1)/pgsz;
  const uint32_t lists=(uint32_t)((uint64_t)MESH_LINK_STREAMS*(2*MESH_STREAM_ENTRIES*sizeof(struct mesh_entry)+4*MESH_STREAM_TIMES*8)/pgsz);
  uint32_t page[2*MESH_PORTS+2],pages[2*MESH_PORTS+2],n=0;
  for(uint32_t p=0;p<MESH_PORTS;p++){page[n]=own->link[p];pages[n++]=2*M->block;page[n]=own->lists[p];pages[n++]=lists;}
  page[n]=own->ones;pages[n++]=M->block;page[n]=own->map;pages[n++]=own->map_pages;
  for(uint32_t i=0;i<n;i++)if(page[i]!=MESH_ABSENT && first<(uint64_t)page[i]+pages[i] && last>=page[i])return 1;
  return 0;
}
/* design/transport.md section 14: the entries a holder has presented on one of its lists taken into the walk, as far
   as its ring has room: each copied once, whether it is the holder's to name (struct taken `whole`) read, its units cut
   (whole pages, then the last page's frames), and the list's `taken` stored past them */
static void walk_take(struct walk *w,struct mesh_stream *x,int land){
  const uint64_t n=atomic_load_explicit(land?&x->lands:&x->sends,memory_order_acquire),page=page_bytes();
  const struct mesh_entry *list=(const struct mesh_entry *)((char *)M+x->lists)+(uint64_t)land*MESH_STREAM_ENTRIES;
  uint64_t t=w->taken;
  for(;t<n && t-w->done<MESH_STREAM_ENTRIES;t++){
    struct taken *k=w->ring+t%MESH_STREAM_ENTRIES;
    k->entry=list[t%MESH_STREAM_ENTRIES];
    const struct mesh_entry *e=&k->entry;
    const uint64_t pages=(e->bytes+page-1)/page*page;
    k->whole=e->at>=M->data_off && !((e->at-M->data_off)%page) && pages<=M->length-e->at && !bridge_touched(e->at,pages) &&
             (!e->word || (e->word<=M->length-8 && !(e->word%8)));
    const uint64_t tail=e->bytes%page;
    k->units[0]=e->bytes/page;k->units[1]=(tail+MESH_FRAME-1)/MESH_FRAME;
    k->first[0]=w->units[0];k->first[1]=w->units[1];k->offset=w->bytes;
    w->units[0]+=k->units[0];w->units[1]+=k->units[1];w->bytes+=e->bytes;
  }
  if(t!=w->taken){w->taken=t;atomic_store_explicit(&x->taken[land],t,memory_order_release);}
}
/* the walk's entry holding unit u of class c (its index over the holding), from the walk's place in the class, which it
   moves past every entry whose units of the class are all before u; NULL where no entry taken holds it */
static struct taken *walk_unit(struct walk *w,uint32_t c,uint64_t u){
  for(;w->at[c]<w->taken;w->at[c]++){
    struct taken *t=w->ring+w->at[c]%MESH_STREAM_ENTRIES;
    if(u<t->first[c]+t->units[c])return t;
  }
  return NULL;
}
/* unit j of class c of an entry: its bytes [*lo, *hi) of the entry's (a whole page; a frame of the last page, the last
   one its bytes alone) */
static inline void unit_span(const struct taken *t,uint32_t c,uint64_t j,uint64_t *lo,uint64_t *hi){
  const uint64_t page=page_bytes();
  *lo=c?t->units[0]*page+j*MESH_FRAME:j*page;
  *hi=c?MIN(*lo+MESH_FRAME,t->entry.bytes):*lo+page;
}
/* design/transport.md section 14: a walk's entries whose units have all completed passed (`got` per class: the lane's
   completions, SENDs or landings, of this holding), and its count (`sent` or `landed`) stored where it moved: an entry's
   bytes count through its last page completed in order, then its frames; the time ring given each entry passed */
static void walk_count(struct walk *w,const uint64_t got[2],_Atomic uint64_t *count,_Atomic uint64_t *ring,uint64_t at){
  const uint64_t page=page_bytes();
  uint64_t moved=w->count;
  while(w->done<w->taken){
    const struct taken *e=w->ring+w->done%MESH_STREAM_ENTRIES;
    const uint64_t pages=got[0]>e->first[0]?MIN(got[0]-e->first[0],e->units[0]):0;
    const uint64_t frames=got[1]>e->first[1]?MIN(got[1]-e->first[1],e->units[1]):0;
    const uint64_t covered=pages*page+(pages==e->units[0]?MIN(frames*MESH_FRAME,e->entry.bytes-pages*page):0);
    moved=e->offset+covered;
    if(pages<e->units[0] || frames<e->units[1])break;
    if(ring)stream_time(ring,w->done,at);
    w->done++;
  }
  if(moved!=w->count){w->count=moved;atomic_store_explicit(count,moved,memory_order_release);}
}
/* a lane's units of its holding completed: its count past the holding's first index, 0 before it */
static inline uint64_t past(uint64_t count,uint64_t base){ return count>base?count-base:0; }
/* design/transport.md section 14: stream t's holding at its port's send thread, each pass.  An owner taken while the
   stream is free begins one (its number one more, asked of the receive thread, which freezes its receives' indices);
   once frozen, the holding's words are made new and its start stored (per lane pair: the large lane's SENDs and receives
   so far, then the small lane's), then `begun`; once its holder has told its peer's start, each lane's first index is
   the greater of its own and its peer's (`target` for its SENDs, `reach` for its receives: the end behind sends duds into
   the peer's posted receives, or posts spares for the peer's SENDs), told the receive thread (`serving`); its entries
   then taken as presented; an owner word no longer the one served (let go to 0 by its holder, or by the event loop for a
   holder gone: streams_let_go) ends it, every presented byte going (the unready as duds), told the receive thread
   (`ended`), and once all have gone the stream is free */
static void stream_turn_one(struct port *port,uint32_t t){
  struct link *l=&port->link;
  struct mesh_stream *x=stream_of(port,t);
  struct walk *w=l->sending+t;
  const uint32_t k=MESH_LANE_FIRST+2*t;
  const uint64_t owner=atomic_load_explicit(&x->owner,memory_order_acquire);
  if(l->hold[t]==HOLD_NONE){
    if(!owner || !atomic_load_explicit(&x->free,memory_order_acquire))return;
    atomic_store_explicit(&x->free,0,memory_order_relaxed);
    l->held[t]=owner;l->number[t]++;l->hold[t]=HOLD_ASKED;
    atomic_store_explicit(&l->asked[t],l->number[t],memory_order_release);
    return;
  }
  if(l->hold[t]!=HOLD_END && owner!=l->held[t]){
    l->hold[t]=HOLD_END;
    atomic_store_explicit(&l->ended[t],l->number[t],memory_order_release);
  }
  if(l->hold[t]==HOLD_ASKED){
    if(atomic_load_explicit(&l->frozen_at[t],memory_order_acquire)!=l->number[t])return;
    for(int c=0;c<2;c++)atomic_store_explicit(&x->taken[c],0,memory_order_relaxed);
    atomic_store_explicit(&x->sends,0,memory_order_relaxed);atomic_store_explicit(&x->lands,0,memory_order_relaxed);
    atomic_store_explicit(&x->ready,0,memory_order_relaxed);atomic_store_explicit(&x->sent,0,memory_order_relaxed);
    atomic_store_explicit(&x->landed,0,memory_order_relaxed);atomic_store_explicit(&x->told,0,memory_order_relaxed);
    atomic_store_explicit(&x->deadline,UINT64_MAX,memory_order_relaxed);
    const uint64_t start[4]={l->lane[k].sent,l->frozen[t][0],l->lane[k+1].sent,l->frozen[t][1]};
    for(uint32_t i=0;i<MESH_STREAM_START;i++)atomic_store_explicit(&x->start[i],i<4?start[i]:0,memory_order_relaxed);
    *w=(struct walk){.ring=w->ring};
    l->hold[t]=HOLD_BEGUN;
    atomic_store_explicit(&x->begun,owner,memory_order_release);
    return;
  }
  if(l->hold[t]==HOLD_BEGUN){
    if(atomic_load_explicit(&x->told,memory_order_acquire)!=l->held[t])return;
    uint64_t peer[4];
    for(int i=0;i<4;i++)peer[i]=atomic_load_explicit(&x->peer_start[i],memory_order_relaxed);
    for(int c=0;c<2;c++){
      l->target[t][c]=w->base[c]=MAX(l->lane[k+c].sent,peer[2*c+1]);
      l->reach[t][c]=MAX(l->frozen[t][c],peer[2*c]);
    }
    l->hold[t]=HOLD_SERVE;
    atomic_store_explicit(&l->serving[t],l->number[t],memory_order_release);
  }
  const int served=atomic_load_explicit(&l->serving[t],memory_order_relaxed)==l->number[t];
  if(l->hold[t]==HOLD_SERVE || (l->hold[t]==HOLD_END && served))walk_take(w,x,0);
  if(l->hold[t]==HOLD_END){
    if(served && (past(l->lane[k].sent,w->base[0])<w->units[0] || past(l->lane[k+1].sent,w->base[1])<w->units[1] ||
                  w->taken<atomic_load_explicit(&x->sends,memory_order_acquire)))return;
    l->held[t]=0;l->hold[t]=HOLD_NONE;
    atomic_store_explicit(&x->begun,0,memory_order_relaxed);
    atomic_store_explicit(&x->free,1,memory_order_release);
  }
}
static void stream_turn(struct port *port){ for(uint32_t s=0;s<plan_streams(port->link.plan);s++)stream_turn_one(port,s); }
static int link_held(const struct link *l){
  for(uint32_t s=0;s<MESH_LINK_STREAMS;s++)if(l->held[s])return 1;
  return 0;
}
/* design/transport.md section 14: a data lane's SENDs, one pass of its port's send thread: while a holding is served or
   ending, first duds up to its first index (`target`: the peer's receives posted before the holding, which a dud fills),
   then each unit of its class in its send walk's order, from the holder's own pages once its entry's word says its
   bytes are ready, else as a dud from the all-ones block once the holder's deadline has passed or the holding is
   ending, an entry not the holder's to name all duds; each while the lane's gate holds its frames (ibv_post_send
   accepts past the queue's frames and corrupts later, so the gate is the caller's); the completions are the link's
   (link_send); a failed post is the lane's (lane_lost).  Whether it posted a unit */
static int lane_send(struct port *port,uint32_t k){
  struct link *l=&port->link;
  struct lane_state *s=l->lane+k;
  const uint32_t t=lane_stream(k),c=lane_class(k),state=l->hold[t];
  if(state!=HOLD_SERVE && !(state==HOLD_END && atomic_load_explicit(&l->serving[t],memory_order_relaxed)==l->number[t]))return 0;
  const struct mesh_stream *x=stream_of(port,t);
  struct walk *w=l->sending+t;
  struct send_gate *gate=l->gate+k;
  const struct mesh_queue *q=l->provider.queues+k;
  const uint64_t first=s->sent,unit=c?MESH_FRAME:page_bytes();
  uint64_t clock=0;
  for(;;){
    uint64_t at=(uint64_t)ones*M->pgsz,bytes=unit;
    if(s->sent>=l->target[t][c]){
      const uint64_t u=s->sent-w->base[c];
      const struct taken *e=walk_unit(w,c,u);
      if(!e)break;
      uint64_t lo,hi;
      unit_span(e,c,u-e->first[c],&lo,&hi);
      bytes=hi-lo;
      const _Atomic uint64_t *word=e->entry.word?stream_words(e->entry.word):&x->ready;
      if(e->whole && atomic_load_explicit(word,memory_order_acquire)>=e->entry.base+hi)at=e->entry.at-M->data_off+lo;
      else if(e->whole && state==HOLD_SERVE){
        const uint64_t deadline=atomic_load_explicit(&x->deadline,memory_order_relaxed);
        if(deadline==UINT64_MAX)break;
        if(!clock)clock=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        if(clock<deadline)break;
      }
    }
    const uint32_t frames=(uint32_t)((bytes+MESH_FRAME-1)/MESH_FRAME);
    if(((gate->posted-gate->retired)&MESH_GATE_MASK)+frames>gate->capacity)break;
    gate->posted+=frames;
    struct ibv_sge span=wire_span(&port->device,page_bytes(),at,bytes);
    struct ibv_send_wr posting={.wr_id=((uint64_t)k<<48)|(gate->posted&MESH_GATE_MASK),.sg_list=&span,.num_sge=1,
                                .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
    const int error=q->send(q->pair,&posting,&bad);
    if(error){lane_lost(port,k,error<0?-error:error);break;}
    __atomic_store_n(&s->sent,s->sent+1,__ATOMIC_RELEASE);gate->requests++;
  }
  return s->sent!=first;
}
/* design/transport.md section 14: stream t's holding at its port's receive thread, each pass: a holding asked (begun at
   the send thread) stops the spares and freezes each lane's receives posted (`frozen`, told in `frozen_at`); its end
   (`ended`) turns the stream to spares, which receive the peer's SENDs to it, its receives posted so far kept (`last`);
   `touching`, the owner of the holding whose pages they are, cleared once all of them have landed (the peer's SENDs
   to it, or its duds as the next holding aligns); a holding whose start is told (`serving`) is served once none touches,
   its landing walk beginning at each lane's `reach`, its entries taken as presented */
static void stream_receive_turn(struct port *port,uint32_t t){
  struct link *l=&port->link;
  struct mesh_stream *x=stream_of(port,t);
  struct walk *w=l->landing+t;
  const uint32_t k=MESH_LANE_FIRST+2*t;
  const uint64_t asked=atomic_load_explicit(&l->asked[t],memory_order_acquire);
  if(asked!=l->rnumber[t]){
    if(l->rstate[t]==RECV_SERVE)for(int c=0;c<2;c++)l->last[t][c]=l->lane[k+c].posted;
    l->rnumber[t]=asked;l->rstate[t]=RECV_NONE;
    for(int c=0;c<2;c++)l->frozen[t][c]=l->lane[k+c].posted;
    atomic_store_explicit(&l->frozen_at[t],asked,memory_order_release);
  }
  if(l->rnumber[t] && atomic_load_explicit(&l->ended[t],memory_order_acquire)==l->rnumber[t] && l->rstate[t]!=RECV_SPARE){
    if(l->rstate[t]==RECV_SERVE)for(int c=0;c<2;c++)l->last[t][c]=l->lane[k+c].posted;
    l->rstate[t]=RECV_SPARE;
  }
  if(atomic_load_explicit(&x->touching,memory_order_relaxed) && l->rstate[t]!=RECV_SERVE &&
     l->lane[k].got>=l->last[t][0] && l->lane[k+1].got>=l->last[t][1])
    atomic_store_explicit(&x->touching,0,memory_order_release);
  if(l->rstate[t]==RECV_NONE && l->rnumber[t] && atomic_load_explicit(&l->serving[t],memory_order_acquire)==l->rnumber[t] &&
     !atomic_load_explicit(&x->touching,memory_order_relaxed)){
    *w=(struct walk){.ring=w->ring,.base={l->reach[t][0],l->reach[t][1]}};
    atomic_store_explicit(&x->touching,atomic_load_explicit(&x->owner,memory_order_relaxed),memory_order_release);
    l->rstate[t]=RECV_SERVE;
  }
  if(l->rstate[t]==RECV_SERVE)walk_take(w,x,1);
}
/* design/transport.md: a data lane's receives posted, at most its window ahead of its landings, on its port's receive
   thread each pass: while a holding is served, first spares up to its first index (`reach`: the peer's SENDs before the
   holding, its duds or its earlier holding's), then each unit of its class in its landing walk's order, into the
   holder's own pages once its entry's word allows (an entry not the holder's to name into the spare page); while the
   stream turns to spares (a holding ended), the spare page; at a fresh pairing, one spare on the small lane, which the
   peer's keepalive fills (link_alive).  Every receive is a whole unit (a page; a frame, which lies
   in its entry's last page, since an entry starts on a page): a peer's SEND is never longer than the receive it fills,
   so ends whose entries differ garble their own pages and never fail the link, and the frames match (TN3205).  0, or the
   post's error */
static int lane_post(struct port *port,uint32_t k){
  struct link *l=&port->link;
  struct lane_state *s=l->lane+k;
  const uint32_t t=lane_stream(k),c=lane_class(k),state=l->rstate[t];
  if(state==RECV_NONE && (l->rnumber[t] || !c))return 0;
  struct walk *w=l->landing+t;
  const struct mesh_queue *q=l->provider.queues+k;
  const uint64_t unit=c?MESH_FRAME:page_bytes(),window=state==RECV_NONE?1:s->window;
  while(s->posted<s->got+window){
    uint64_t at=link_spare(port),bytes=unit;
    if(state==RECV_SERVE && s->posted>=l->reach[t][c]){
      const uint64_t u=s->posted-w->base[c];
      const struct taken *e=walk_unit(w,c,u);
      if(!e)break;
      if(e->entry.word && e->whole && atomic_load_explicit(stream_words(e->entry.word),memory_order_acquire)<e->entry.base)break;
      uint64_t lo,hi;
      unit_span(e,c,u-e->first[c],&lo,&hi);
      if(e->whole)at=e->entry.at-M->data_off+lo;
    }
    struct ibv_sge span=wire_span(&port->device,page_bytes(),at,bytes);
    struct ibv_recv_wr request={.wr_id=k,.sg_list=&span,.num_sge=1},*bad;
    const int error=q->receive(q->pair,&request,&bad);
    if(error)return error<0?-error:error;
    s->posted++;
  }
  return 0;
}
/* design/transport.md: a data lane's next unit landed (receives complete in the order posted) */
static void lane_landed(struct port *port,uint32_t k){ port->link.lane[k].got++; }
/* design/HETEROGENEOUS-LINKS.md section 10: the link's keepalive, at both ends: once the link has posted no SEND for
   MESH_PLATFORM_NS/5 (holding no stream here) or MESH_KEEPALIVE_NS (holding one), a frame of the all-ones block on the
   small lane of its least stream that serves no holding here and has nothing in flight there: ALIVE_BYTES of it on a
   stream that has served no holding here this pairing (the peer's receive for it can only be a spare), else MESH_FRAME
   bytes, a dud.
   It is a SEND of the lane like any other, counted in the lane's SENDs, so a holding's start (its lanes' counts:
   stream_turn_one) takes it in; it waits in the hardware's credit until the peer posts a receive (TN3205), and fills a
   spare, or, where the peer still serves a holding this end ended, that holding's next unit as a dud would.  A link
   whose every stream serves a holding here sends none: a timed frame on a held stream's lane would fill the receive
   that stream's next unit is owed */
#define ALIVE_BYTES 16u
static void link_alive(struct port *port,uint32_t served){
  struct link *l=&port->link;
  const uint32_t streams=plan_streams(l->plan);
  uint32_t t=0,k=MESH_LANE_FIRST+1;
  for(;t<streams;t++,k+=2)if(l->hold[t]==HOLD_NONE && served>>k&1 && l->lane[k].done==l->lane[k].sent)break;
  if(t==streams)return;
  const uint64_t at=mach_absolute_time();
  if(at-l->sent_at<alive_ticks[link_held(l)])return;
  struct send_gate *gate=l->gate+k;
  if(((gate->posted-gate->retired)&MESH_GATE_MASK)+1>gate->capacity)return;
  gate->posted++;
  struct ibv_sge span=wire_span(&port->device,page_bytes(),(uint64_t)ones*M->pgsz,l->number[t]?MESH_FRAME:ALIVE_BYTES);
  struct ibv_send_wr posting={.wr_id=((uint64_t)k<<48)|(gate->posted&MESH_GATE_MASK),.sg_list=&span,.num_sge=1,
                              .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
  const struct mesh_queue *q=l->provider.queues+k;
  const int error=q->send(q->pair,&posting,&bad);
  if(error){lane_lost(port,k,error<0?-error:error);return;}
  __atomic_store_n(&l->lane[k].sent,l->lane[k].sent+1,__ATOMIC_RELEASE);gate->requests++;
  l->sent_at=at;
}
/* the lane a completion's queue pair number names: one of the link's lanes, else MESH_LINK_LANES (dropped; a lane its
   plan leaves out has the number UINT32_MAX, which no queue pair has) */
static __attribute__((always_inline)) inline uint32_t link_lane(const struct link *l,uint32_t qp_num){
  uint32_t k=0;
  while(k<MESH_LINK_LANES && l->qpn[k]!=qp_num)k++;
  return k;
}
/* design/transport.md: the link's first failed completion, on any lane, or a failed poll: no lane of the
   link served from here, so nothing is posted on it again, and the link lost by the event loop on reading `broken`
   (link_turn), the stream's words ~0 (port_lost) */
static void link_failed(struct port *port,int64_t code,uint32_t domain){
  struct link *l=&port->link;
  const uint32_t served=atomic_exchange_explicit(&l->served,0,memory_order_acq_rel);
  if(served)say("link %u (%s): a completion failed (code %lld, domain %u): no lane served, the link toward node %u lost\n",
                port->index,port->name,(long long)code,domain,port->peer);
  atomic_fetch_or_explicit(&l->broken,served,memory_order_release);
  wake();
}
/* design/transport.md: the link's SENDs, one pass of the port's send thread: its SEND completions taken (a lane no
   longer served drops its own; a failed one fails the link, link_failed): a lane's frames retired through its gate
   (each request names its lane and the frames posted through it) and its SENDs completed counted; the pass's time the
   link's last SEND where it posted one, then the keepalive where it is due (link_alive); then each stream whose holding
   is served or ending: each entry whose first unit the pass posted given its posting time (one read of the ticks after
   every post of the pass, design/transport.md section 14, mesh.h `posted_at`), and its `sent` stored where its completions moved
   it (walk_count) */
static void link_send(struct port *port,uint32_t posted){
  struct link *l=&port->link;
  const uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  struct ibv_wc done[16];
  const int count=l->provider.poll(l->provider.sent,16,done);
  if(count<0){link_failed(port,EIO,2);return;}
  uint32_t completed=0;
  for(int i=0;i<count;i++){
    const uint32_t lane=link_lane(l,done[i].qp_num);
    if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
    if(done[i].status){link_failed(port,(int64_t)done[i].status,2);return;}
    struct send_gate *gate=l->gate+lane;
    gate->retired=done[i].wr_id&MESH_GATE_MASK;gate->completions++;l->lane[lane].done++;completed|=UINT32_C(1)<<lane;
  }
  const uint64_t at=posted?mach_absolute_time():0;
  if(posted)l->sent_at=at;
  link_alive(port,served);
  if(!posted && !completed)return;
  for(uint32_t t=0;t<plan_streams(l->plan);t++){
    const uint32_t k=MESH_LANE_FIRST+2*t,lanes=UINT32_C(3)<<k;
    if(!((posted|completed)&lanes) || (l->hold[t]!=HOLD_SERVE && l->hold[t]!=HOLD_END))continue;
    struct walk *w=l->sending+t;
    struct mesh_stream *x=stream_of(port,t);
    const uint64_t sent[2]={past(l->lane[k].sent,w->base[0]),past(l->lane[k+1].sent,w->base[1])};
    for(;posted&lanes && w->timed<w->taken;w->timed++){
      const struct taken *e=w->ring+w->timed%MESH_STREAM_ENTRIES;
      if(e->units[0]?sent[0]<=e->first[0]:e->units[1] && sent[1]<=e->first[1])break;
      stream_time(l->posted_at[t],w->timed,at);
    }
    const uint64_t got[2]={past(l->lane[k].done,w->base[0]),past(l->lane[k+1].done,w->base[1])};
    walk_count(w,got,&x->sent,NULL,0);
  }
}
/* the link's receive completions, one pass of the port's receive thread: a lane's unit landed (lane_landed), a failed
   one failing the link (link_failed); then each stream whose
   holding is served and a unit of which landed: its `landed` stored where its landings moved it, and each entry landed
   whole given its landing time (one read of the ticks after the pass's landings, design/transport.md section 14, mesh.h
   `landed_at`) */
static void link_receive(struct port *port){
  struct link *l=&port->link;
  const uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  struct ibv_wc done[16];
  const int count=l->provider.poll(l->provider.completion,16,done);
  if(count<0){link_failed(port,EIO,3);return;}
  if(count)atomic_thread_fence(memory_order_acquire);
  uint32_t landed=0;
  for(int i=0;i<count;i++){
    const uint32_t lane=link_lane(l,done[i].qp_num);
    if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
    uint64_t status_opcode;
    memcpy(&status_opcode,&done[i].status,sizeof status_opcode);
    if((uint32_t)status_opcode){link_failed(port,(uint32_t)status_opcode,2);return;}
    if(lane>=MESH_LANE_FIRST && (status_opcode>>32)&IBV_WC_RECV){lane_landed(port,lane);landed|=UINT32_C(1)<<lane;}
  }
  if(!landed)return;
  const uint64_t at=mach_absolute_time();
  for(uint32_t t=0;t<plan_streams(l->plan);t++){
    const uint32_t k=MESH_LANE_FIRST+2*t;
    if(!(landed&(UINT32_C(3)<<k)) || l->rstate[t]!=RECV_SERVE)continue;
    struct walk *w=l->landing+t;
    const uint64_t got[2]={past(l->lane[k].got,w->base[0]),past(l->lane[k+1].got,w->base[1])};
    walk_count(w,got,&stream_of(port,t)->landed,l->landed_at[t],at);
  }
}

/* a progress thread's tell (the epoch it has seen, the guard it holds at), its raiser woken (port_wait) */
static void port_tell(_Atomic uint64_t *word,uint64_t value){
  atomic_store_explicit(word,value,memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
/* design/transport.md section 14: one pass of a port's progress thread, `side` 0 its receives' and 1 its SENDs': the
   lanes the link serves (`served` read once a pass: none posts on a lane its link no longer serves), the stream's
   holding and each data lane's SENDs (the send thread) or each one's first receives (the receive thread), then every
   lane's completions.  It tells the port's epoch it began at (a conclusion around a stuck call moves it:
   ports_stalled) */
static void port_pass(struct port *port,int side,uint64_t *told){
  const uint64_t epoch=atomic_load(&port->epoch);
  struct link *l=&port->link;
  const uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  if(served>>MESH_LANE_FIRST){
    if(side)stream_turn(port);
    else for(uint32_t t=0;t<plan_streams(l->plan);t++)stream_receive_turn(port,t);
  }
  uint32_t posted=0;
  for(uint32_t k=MESH_LANE_FIRST;k<MESH_LINK_LANES;k++){
    if(!(served>>k&1))continue;
    if(side)posted|=(uint32_t)lane_send(port,k)<<k;
    else{
      const int error=lane_post(port,k);
      if(error)lane_lost(port,k,error);
    }
  }
  if(served){if(side)link_send(port,posted);else link_receive(port);}
  if(epoch!=*told){port_tell(port->seen+side,epoch);*told=epoch;}
}
static void *port_progress(void *argument){
  struct port *port=((struct progress *)argument)->port;
  const int side=((struct progress *)argument)->side;
  free(argument);
  pthread_setname_np(side?"mesh.rdma.send":"mesh.rdma.receive");
  /* fault_heard: this thread its port's, and its handler on a stack of its own */
  stack_t alternate={.ss_sp=malloc(SIGSTKSZ),.ss_size=SIGSTKSZ};
  if(alternate.ss_sp)sigaltstack(&alternate,NULL);
  atomic_store_explicit(&port->thread[side],pthread_self(),memory_order_release);
  uint64_t told=atomic_load_explicit(&port->seen[side],memory_order_relaxed);
  while(!atomic_load_explicit(&port->closing,memory_order_acquire)){
    /* while the port's guard stands, or its link serves no lane, no pass (no post on any queue pair of its device) and
       no lane read: the thread waits on the guard word (design/transport.md B13: a guarded call in the driver costs no
       core), woken by its raising and lowering, by a settle and by the link's coming up (guard_wake), looking again
       every MESH_PAUSE_NS (the stop's `closing`), the guard told and the epoch at each waking, each tell waking its
       raiser (port_tell) */
    const uint64_t guard=atomic_load_explicit(&port->guard,memory_order_acquire);
    if(guard&1 || !atomic_load_explicit(&port->link.served,memory_order_acquire)){
      if(guard&1)port_tell(port->guarded+side,guard);
      const uint64_t epoch=atomic_load_explicit(&port->epoch,memory_order_acquire);
      if(epoch!=told){port_tell(port->seen+side,epoch);told=epoch;}
      os_sync_wait_on_address_with_timeout(&port->guard,guard,sizeof port->guard,OS_SYNC_WAIT_ON_ADDRESS_NONE,
        OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
      continue;
    }
    port_pass(port,side,&told);
  }
  return NULL;
}
/* design/transport.md B13: the progress threads parked on the port's guard look again */
static void guard_wake(struct port *port){ os_sync_wake_by_address_all(&port->guard,sizeof port->guard,OS_SYNC_WAKE_BY_ADDRESS_NONE); }
/* design/transport.md: the port's two progress threads started, once; 0, or pthread_create's error */
static int port_threads(struct port *port){
  int error=0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  while(port->threads<2 && !error){
    struct progress *progress=malloc(sizeof *progress);
    if(!progress){error=ENOMEM;break;}
    *progress=(struct progress){port,port->threads};
    atomic_store(&port->seen[port->threads],atomic_load(&port->epoch));
    pthread_t created;
    if((error=pthread_create(&created,&attributes,port_progress,progress)))free(progress);
    else port->threads++;
  }
  pthread_attr_destroy(&attributes);
  return error;
}
/* design/transport.md: the raiser of a settle or a guard parked, never spinning, until each started
   progress thread's word (`told`: the epoch it has seen, or the guard it holds at) reaches `value`, woken by the
   thread's tell (port_tell), its guard's waiters woken as it begins and at each waking; at most
   MESH_PLATFORM_NS: a thread that has not told by then is in a call that has not returned, and the port is halted for
   the event loop to conclude as around a stuck call (`unguarded`, ports_stalled).  0, or ETIMEDOUT */
static int port_wait(struct port *port,_Atomic uint64_t *told,uint64_t value){
  const uint64_t until=clock_ns()+MESH_PLATFORM_NS;
  for(int side=0;side<port->threads;side++)
    for(uint64_t seen,at;(seen=atomic_load_explicit(told+side,memory_order_acquire))<value;){
      if((at=clock_ns())>=until){port_halt(port,&unguarded);return ETIMEDOUT;}
      guard_wake(port);
      os_sync_wait_on_address_with_timeout(told+side,seen,sizeof *told,OS_SYNC_WAIT_ON_ADDRESS_NONE,OS_CLOCK_MACH_ABSOLUTE_TIME,
        until-at);
    }
  return 0;
}
/* design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1": the guard of a port's device, on the port's queue,
   around a change of its verbs objects (queue pairs and completion queues made, connected or destroyed): raised, it
   returns once each started progress thread holds (it told the guard's value before its next pass, and passes no more
   while the guard stands), so no queue pair of the device is posted on while another is made, changed or destroyed
   (Apple's provider: a post on a long-lived queue pair across another's teardown stores to the device's shared doorbell
   mapping after it moved, tbt_post_recv); lowered, the threads parked on it woken and going on
   (design/transport.md B13).  It stands for every change once the port's threads run, the link lanes'
   long-lived queue pairs being posted on while the link is up.  Raised: 1; 0 where no progress thread runs; -1 where
   one did not hold within MESH_PLATFORM_NS (port_wait), the guard lowered again and the change not to be made */
static void guard_lower(struct port *port,int raised){
  if(raised>0){atomic_fetch_add_explicit(&port->guard,1,memory_order_release);guard_wake(port);}
}
static int guard_raise(struct port *port){
  if(!port->threads)return 0;
  if(!port_wait(port,port->guarded,atomic_fetch_add_explicit(&port->guard,1,memory_order_acq_rel)+1))return 1;
  guard_lower(port,1);
  return -1;
}

/* design/transport.md: a message to the port's peer, queued behind what the socket has not taken yet (a node's record
   in place of its older one: port_coalesce); a send that fails, or a queue the process cannot grow, is the link's loss */
static void port_watch(struct port *port,int filter){
  struct kevent64_s event;EV_SET64(&event,(uint64_t)port->control,filter,EV_ADD|(filter==EVFILT_WRITE?EV_ONESHOT:0),0,0,port->index,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))port_lost(port,errno);
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
/* a RECORD whose node has a RECORD waiting wholly unsent in the port's queue takes that one's place where it is no older
   (RFC 2328 s13: a node's newest record is the one that counts), so a peer that reads nothing holds a record a node in
   the queue; 1 where it did */
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
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes){
  if(port->control<0 || (said.kind!=HELLO && !port->greeted) || port_coalesce(port,&said,payload,bytes))return;
  said.magic=XMAGIC;said.node=M->node;said.boot=M->boot;said.block=M->block*M->pgsz;said.port=port->index;said.bytes=bytes;
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
/* each client slot's owner as clients_turn last read it */
uint64_t owners[MESH_CLIENTS];
/* design/transport.md: this node's record, its region, memory and model with its ports and health */
char model[MESH_MODEL_BYTES];
/* a port's link as the port table tells it; a change is a new record */
static void link_set(struct port *port,uint32_t peer,uint32_t phase,int64_t code){
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  int changed=atomic_load(&link->peer)!=peer || atomic_load(&link->phase)!=phase;
  atomic_store_explicit(&link->peer,peer,memory_order_relaxed);atomic_store_explicit(&link->code,code,memory_order_relaxed);
  atomic_store_explicit(&link->phase,phase,memory_order_release);
  if(changed)record_changed();
}

/* design/transport.md section 14: a link job on the port's queue, guarded (no queue pair of the device is posted on
   while one is made or connected: guard_raise), its verbs calls made on hearing the peer, from the peer's numbers as the
   job was started (`numbers`, its LID and GID among them: the event loop takes the peer's LINK and hello meanwhile), and
   none once its pairing is lost (B10: read as the guard stands and between its steps; a guard not held within
   MESH_PLATFORM_NS halts the port, its pairing with it, port_wait), each in the port's child and once in its life
   (design/transport.md section 14, "A process a port"): the lanes made (LINK_JOB_MAKE: the port's own address read
   again, then a UC queue pair in INIT for each lane of the link's plan, d6dceac's verbs_up, each lane's share of the
   completion queues the plan's lanes alone dividing them: design/HETEROGENEOUS-LINKS.md; EALREADY for a make asked
   again); or connected to the peer's (LINK_JOB_CONNECT, the plan's lanes alone: RTR, then RTS, each lane's number and
   gate set, every lane's progress made new, and the port's progress threads started).  No queue pair is moved to
   another state, destroyed or forgotten after it: the in-place RESET of 127 panicked the M5 in AppleThunderboltRDMA on its
   first run (2026-10-19, design/RDMA-KERNEL-RECOVERY.md), and the kernel releases every object at the child's exit, its
   one end */
struct link_numbers { uint32_t peer[MESH_LINK_LANES][2],pairing; uint16_t lid; uint8_t gid[16]; };
static int link_lapsed(const struct link *l,const struct link_numbers *numbers){
  return atomic_load_explicit(&l->pairing,memory_order_relaxed)!=numbers->pairing?ECANCELED:0;
}
static int link_make(struct port *port,const struct link_numbers *numbers){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  struct ibv_port_attr attributes;
  int error=0;
  if(p->queues)return EALREADY;
  /* design/transport.md: the port's own address (its attributes and GID) read again at the make, as its opening read
     it: a peer's restart can move the port's GID table, and LINKs naming the address of the port's opening failed every
     lane's connection at the peer with EIO (2026-10-19, after the Mini's power loss) */
  if(ibv_query_port(port->device.context,1,&attributes))return errno?errno:EIO;
  port->device.port=attributes;
  if((error=device_gid(&port->device,port->name)) || (error=link_lapsed(l,numbers)) || (error=pair_up(p,l->plan)))return error;
  l->share=(uint32_t)MIN(p->completion->cqe,p->sent->cqe)/MAX(plan_lanes(l->plan),1u);
  return 0;
}
static void lane_reset(struct port *port,uint32_t k);
/* design/transport.md section 14: stream s's identity, alike at both ends: FNV-1a over its first lane's two ends, each
   its port's GID and its queue pair's number, the lesser end first; never 0 */
static uint64_t stream_wire(const struct port *port,const struct link_numbers *numbers,uint32_t s){
  const uint32_t k=MESH_LANE_FIRST+2*s*MESH_STREAM_PAIRS;
  uint8_t end[2][20];
  memcpy(end[0],port->gid,16);memcpy(end[0]+16,&port->link.qpn[k],4);
  memcpy(end[1],numbers->gid,16);memcpy(end[1]+16,&numbers->peer[k][0],4);
  const int first=memcmp(end[0],end[1],20)>0;
  uint64_t h=UINT64_C(14695981039346656037);
  for(int e=0;e<2;e++)
    for(int i=0;i<20;i++){h^=end[e^first][i];h*=UINT64_C(1099511628211);}
  return h?h:1;
}
static int link_connect(struct port *port,const struct link_numbers *numbers){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  const uint32_t plan=l->plan;
  int error=0;
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)error=queue_receive(p,(int)k,numbers->peer[k],numbers->lid,numbers->gid);
  if(!error)error=link_lapsed(l,numbers);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)error=queue_send(p,(int)k);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++){
    l->qpn[k]=plan>>k&1?p->queues[k].pair->qp_num:UINT32_MAX;
    l->gate[k]=(struct send_gate){.capacity=plan>>k&1?MIN(p->queues[k].send_capacity,l->share>1?l->share:1):0};
  }
  if(!error)error=link_lapsed(l,numbers);
  l->sent_at=0;
  for(uint32_t k=MESH_LANE_FIRST;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)lane_reset(port,k);
  for(uint32_t s=0;s<plan_streams(plan) && !error;s++)l->wire[s]=stream_wire(port,numbers,s);
  return error;
}
/* design/transport.md section 14, "A process a port": the link's streams opened as its lanes connect, by the child
   before any lane is served, every one new: a holding of a lost pairing ended with it (its owner word 0: its holder's
   give, a compare-and-swap from its own client, then matches nothing, and its communicator reads its holding gone,
   mesh-call.h mesh_stream_held), its ready and landed words 0, its next units 0, its peer node, its share of the link's
   bandwidth and its identity told, then free and live; and its time rings' addresses the link's own (`posted_at`,
   `landed_at`: the progress threads read no line of the region a client writes for them).  None is kept for a holder
   of a lost pairing: no holder can, by letting go or by never letting go, keep a stream of the link from its next
   pairing (design/rubrics.md M4).  Nothing of
   the lost pairing is sent on it: that pairing's groups ended at their next look at a landed word ~0 (streams_cancel),
   this connect comes MESH_PLATFORM_NS and a child's exit after the loss, and its communicators start no group after it.
   The streams opened are those the link's plan has lanes for (design/HETEROGENEOUS-LINKS.md): the rest stay as the
   loss, the adoption or the region's making left them, not live and not free, so no client takes one.  Each stream's
   share of the bandwidth is the link's over MESH_LINK_STREAMS however many are live, so the map's figure for a link
   (mesh-plan.h mesh_map, the lesser of its ends') is alike for links of every plan */
static void streams_open(struct port *port,uint32_t node,uint64_t bandwidth){
  struct link *l=&port->link;
  for(uint32_t s=0;s<plan_streams(l->plan);s++){
    struct mesh_stream *x=stream_of(port,s);
    atomic_store_explicit(&x->owner,0,memory_order_release);
    _Atomic uint64_t *words[]={&x->begun,&x->told,&x->deadline,&x->sends,&x->lands,&x->ready,&x->sent,&x->landed,&x->taken[0],
                               &x->taken[1],&x->touching};
    for(size_t i=0;i<sizeof words/sizeof *words;i++)atomic_store_explicit(words[i],0,memory_order_relaxed);
    for(uint32_t i=0;i<MESH_STREAM_START;i++){atomic_store_explicit(&x->start[i],0,memory_order_relaxed);atomic_store_explicit(&x->peer_start[i],0,memory_order_relaxed);}
    l->posted_at[s]=stream_words(x->posted_at);l->landed_at[s]=stream_words(x->landed_at);
    for(uint32_t e=0;e<2*MESH_STREAM_TIMES;e++){atomic_store_explicit(l->posted_at[s]+e,0,memory_order_relaxed);atomic_store_explicit(l->landed_at[s]+e,0,memory_order_relaxed);}
    l->held[s]=l->number[s]=l->rnumber[s]=0;l->hold[s]=HOLD_NONE;l->rstate[s]=RECV_NONE;
    l->last[s][0]=l->last[s][1]=0;
    atomic_store_explicit(&l->asked[s],0,memory_order_relaxed);atomic_store_explicit(&l->serving[s],0,memory_order_relaxed);
    atomic_store_explicit(&l->ended[s],0,memory_order_relaxed);atomic_store_explicit(&l->frozen_at[s],0,memory_order_relaxed);
    l->sending[s]=(struct walk){.ring=l->sending[s].ring};l->landing[s]=(struct walk){.ring=l->landing[s].ring};
    atomic_store_explicit(&x->peer,node,memory_order_relaxed);
    atomic_store_explicit(&x->bandwidth,bandwidth/MESH_LINK_STREAMS,memory_order_relaxed);
    atomic_store_explicit(&x->wire,l->wire[s],memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&x->free,1,memory_order_release);
    atomic_store_explicit(&x->live,1,memory_order_release);
  }
}
static void link_job(struct port *port,int job){
  struct link *l=&port->link;
  struct link_numbers numbers={.pairing=atomic_load_explicit(&l->pairing,memory_order_relaxed),.lid=port->heard.lid};
  memcpy(numbers.peer,l->peer,sizeof numbers.peer);memcpy(numbers.gid,port->heard.gid,sizeof numbers.gid);
  l->job=job;l->job_pairing=numbers.pairing;l->job_at=now;atomic_store_explicit(&l->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    atomic_store_explicit(&port->thread[2],pthread_self(),memory_order_release);
    const int raised=guard_raise(port);
    int result=link_lapsed(l,&numbers);
    if(!result)result=job==LINK_JOB_MAKE?link_make(port,&numbers):link_connect(port,&numbers);
    if(!result && job==LINK_JOB_CONNECT)result=port_threads(port);
    guard_lower(port,raised);
    l->result=result;
    atomic_store_explicit(&port->thread[2],NULL,memory_order_release);
    atomic_store_explicit(&l->done,1,memory_order_release);
    wake();
  });
}
/* a LINK telling the peer every lane of this end's pairing, the lanes of its plan (`count` theirs): each one's queue pair
   number and first sequence number, as its child told them (CHILD_MADE), 0 for a lane the plan leaves out */
static void link_tell(struct port *port){
  const struct link *l=&port->link;
  struct message said={.kind=LINK,.count=plan_lanes(l->plan)};
  for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(l->plan>>k&1){said.pair[k][0]=l->qpn[k];said.pair[k][1]=(l->psn+k)&0xffffff;}
  port_say(port,said,NULL,0);
}
/* the link of a port lost (its peer's loss, a failure, the bridge stopping): its lanes served no more and left as they
   are, no verbs call made, and, once connected, never paired again in this bridge's life (design/transport.md section
   14); a job of this pairing still on the port's queue makes no call and is dropped as it ends (B10) */
static void link_lost(struct port *port){
  struct link *l=&port->link;
  atomic_fetch_add_explicit(&l->pairing,1,memory_order_relaxed);l->state=LINK_DOWN;l->heard=0;
  atomic_store_explicit(&l->served,0,memory_order_release);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
}
/* design/transport.md section 14: data lane k's progress made new at its link's connect, its sequences 0 and its receive
   window the lane's share of the completion queue in its unit's frames (a page, or a frame); on the port's queue in the
   connect's job, before any progress thread serves it.  Its stream's words are the stream's opening's (streams_open) */
static void lane_reset(struct port *port,uint32_t k){
  struct link *l=&port->link;
  const uint32_t frames=(uint32_t)((lane_class(k)?MESH_FRAME:page_bytes())/MESH_FRAME);
  l->lane[k]=(struct lane_state){.window=MAX(1,MIN(l->share/frames,l->provider.queues[k].receive_capacity))};
  atomic_thread_fence(memory_order_release);
}
/* design/transport.md section 14: the pages of a port's streams' lists and time rings: each stream's two lists of
   MESH_STREAM_ENTRIES entries, then its two time rings of MESH_STREAM_TIMES entries of two words, each part whole pages */
static uint64_t whole_pages(uint64_t bytes){ return (bytes+page_bytes()-1)/page_bytes()*page_bytes(); }
static uint32_t lists_pages(void){
  const uint64_t each=whole_pages(2*(uint64_t)MESH_STREAM_ENTRIES*sizeof(struct mesh_entry))+2*whole_pages((uint64_t)MESH_STREAM_TIMES*16);
  return (uint32_t)(MESH_LINK_STREAMS*each/M->pgsz);
}
/* design/transport.md section 14: the port's streams' lists and time rings, leased from the arena by the parent as the
   port first pairs (recorded in the region, so a bridge adopting the region takes the same pages: section 2) and held
   for the region's life, every child and bridge of the port's, told in the region (mesh.h struct mesh_stream `lists`,
   `posted_at`, `landed_at`).  0, or ENOMEM where the region holds no room for them */
static int stream_lists(struct port *port){
  struct link *l=&port->link;
  struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  const uint32_t pages=lists_pages();
  if(own->lists[port->index]==MESH_ABSENT)own->lists[port->index]=mesh_arena_alloc(&bridge,pages,M->block);
  const uint32_t page=own->lists[port->index];
  if(page==MESH_ABSENT)return ENOMEM;
  held_pages+=pages;record_changed();
  uint64_t at=M->data_off+(uint64_t)page*M->pgsz;
  for(uint32_t s=0;s<MESH_LINK_STREAMS;s++){
    struct mesh_stream *x=stream_of(port,s);
    x->lists=at;at+=whole_pages(2*(uint64_t)MESH_STREAM_ENTRIES*sizeof(struct mesh_entry));
    x->posted_at=at;at+=whole_pages((uint64_t)MESH_STREAM_TIMES*16);
    x->landed_at=at;at+=whole_pages((uint64_t)MESH_STREAM_TIMES*16);
  }
  l->lists=page;
  return 0;
}
/* design/transport.md section 14, "A process a port": the child's walks of each stream's two lists, made at its first
   connect; 0, or ENOMEM */
static int lanes_room(struct port *port){
  struct link *l=&port->link;
  for(uint32_t s=0;s<MESH_LINK_STREAMS;s++){
    if(!l->sending[s].ring && !(l->sending[s].ring=calloc(MESH_STREAM_ENTRIES,sizeof(struct taken))))return ENOMEM;
    if(!l->landing[s].ring && !(l->landing[s].ring=calloc(MESH_STREAM_ENTRIES,sizeof(struct taken))))return ENOMEM;
  }
  return 0;
}
/* design/transport.md section 14: the streams of a port (`port` NULL: of every port) live no more, free no more, and
   their `sent` and `landed` ~0, as its link is lost or the bridge stops: a client waiting on one reads the loss */
void streams_cancel(const struct port *port){
  for(uint32_t p=0;p<port_count;p++){
    if((port && ports+p!=port) || ports[p].link.lists==MESH_ABSENT)continue;
    for(uint32_t s=0;s<MESH_LINK_STREAMS;s++){
      struct mesh_stream *x=stream_of(ports+p,s);
      atomic_store_explicit(&x->live,0,memory_order_release);
      atomic_store_explicit(&x->free,0,memory_order_release);
      atomic_store_explicit(&x->sent,UINT64_MAX,memory_order_release);
      atomic_store_explicit(&x->landed,UINT64_MAX,memory_order_release);
    }
  }
}
/* design/transport.md section 14: a stream whose owner word names a client gone (its process exited, its slot retired)
   let go, its word 0, as that client's rows are released; its port's send thread then ends the holding (stream_turn) */
void streams_let_go(void){
  for(uint32_t p=0;p<port_count;p++)
    for(uint32_t s=0;s<MESH_LINK_STREAMS;s++){
      struct mesh_stream *x=stream_of(ports+p,s);
      uint64_t owner=atomic_load_explicit(&x->owner,memory_order_acquire);
      if(owner && !owned(owner))atomic_compare_exchange_strong(&x->owner,&owner,0);
    }
}
/* design/transport.md section 14: the link's turn, on the event loop: a job's end taken (made: LINK told the peer every
   lane; connected: up, every lane served, the stream live); a job that failed loses the link (port_lost); while it is
   up, a lane a progress thread found failed (`broken`) loses it; then, while no port's job is out past MESH_PLATFORM_NS (B12) and once the port's last
   quiet window has ended (B9), the lanes made once the port is paired (`lanes`: its child asked for them), and connected
   once both ends' are made, each by the port's child and once in its life: a link lost once its lanes were asked for
   pairs again with the port's next child (design/transport.md section 14, "A process a port": port_lost, child_turn,
   child_reaped) */
static void child_job(struct port *port,int job);
static void link_turn(struct port *port){
  struct link *l=&port->link;
  if(l->job){
    if(!atomic_load_explicit(&l->done,memory_order_acquire))return;
    const int job=l->job,result=l->result;
    l->job=0;
    if(l->job_pairing!=atomic_load_explicit(&l->pairing,memory_order_relaxed)){}
    else if(result){
      say("link %u (%s): lanes toward node %u not %s: %s; paired again with the port's next child\n",port->index,port->name,
          port->peer,job==LINK_JOB_MAKE?"made":"connected",strerror(result));
      port_lost(port,result);
      return;
    }
    else if(job==LINK_JOB_MAKE){l->state=LINK_MADE;link_tell(port);}
    else{
      l->state=LINK_UP;
      say("link %u (%s): %u lanes up toward node %u (plan %#x: %u stream%s, the keepalive on the small lane of a stream "
          "that serves no holding), each lane's share %u frames\n",port->index,port->name,plan_lanes(l->plan),port->peer,
          l->plan,plan_streams(l->plan),plan_streams(l->plan)==1?"":"s",l->share);
      record_changed();
    }
  }
  const int held=atomic_load_explicit(&stalled,memory_order_relaxed) || now<port->quiet_at;
  if(!port->greeted)return;
  if(l->state==LINK_UP){
    const uint32_t broken=atomic_exchange_explicit(&l->broken,0,memory_order_acq_rel);
    if(broken){
      say("link %u (%s): lanes %#x toward node %u failed: the link lost\n",port->index,port->name,broken,port->peer);
      port_lost(port,EIO);
      return;
    }
  }
  if(l->job || held)return;
  if(l->state==LINK_DOWN){
    if((l->page==MESH_ABSENT && ((l->page=mesh_bridge_pages(M)->link[port->index]=mesh_arena_alloc(&bridge,l->pages,M->block))==MESH_ABSENT ||
                                 !(held_pages+=l->pages))) ||
       (l->lists==MESH_ABSENT && stream_lists(port))){
      say("link %u (%s): lanes not made: the region holds no room for their blocks and rings; paired again on hearing the peer\n",
          port->index,port->name);
      port_lost(port,ENOMEM);
      return;
    }
    l->lanes=1;
    child_job(port,LINK_JOB_MAKE);
  }
  else if(l->state==LINK_MADE && l->heard){l->state=LINK_CONNECTING;child_job(port,LINK_JOB_CONNECT);}
}
/* design/transport.md B12 (22891630^ mesh-flight.h's pending-operation age, per port: its dispatch's
   `job_at` in place of flight_begin's stamp, its `done` in place of flight_end's): a port whose open's or link's job
   is still out MESH_PLATFORM_NS after it began is concluded around
   it, once: open 0 and port_lost (ETIMEDOUT: its socket closed, the map's phase STOPPED, its stream's words ~0), no
   verbs call.  The call itself cannot
   be ended.  While any port's job is so out, no port dispatches a driver call and the devices are not read (`stalled`:
   port_open, link_turn, ports_find).  A port whose progress threads did not hold for a settle or a
   guarded change within MESH_PLATFORM_NS (`unguarded`, port_wait: the change not made) is concluded alike, once, unless
   already concluded around a call.  Each conclusion moves the port's epoch and wakes its threads; the port stays
   `stuck` until the call returns and each progress thread has told the conclusion's epoch (a thread tells it on its
   first pass after its call returns, port_pass, or as it wakes where it is in none), then opens again (`opening`,
   port_open past its quiet window, B9; an open's own return is port_opened's) and so takes jobs again.  Read on the
   event loop's turn */
static void ports_stalled(void){
  int any=0;
  const uint32_t unheld=atomic_exchange(&unguarded,0);
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    const struct link *l=&port->link;
    const int opening=port->job && !atomic_load_explicit(&port->done,memory_order_acquire) && now-port->job_at>=MESH_PLATFORM_NS;
    const int linking=l->job && !atomic_load_explicit(&l->done,memory_order_acquire) && now-l->job_at>=MESH_PLATFORM_NS;
    const int out=opening||linking,held=unheld>>p&1;
    any|=out;
    if(port->stuck){
      int returned=!out;
      for(int side=0;returned && side<port->threads;side++)
        returned=atomic_load_explicit(port->seen+side,memory_order_acquire)>=atomic_load(&port->epoch);
      if(returned){port->stuck=0;if(!port->job)port->opening=1;}
      continue;
    }
    if(!out && !held)continue;
    port->stuck=1;atomic_fetch_add(&port->epoch,1);guard_wake(port);
    if(!out)say("link %u (%s): a progress thread did not hold within %.1f s for a settle or a verbs change, which was not "
                "made: its port concluded around it (design/transport.md)\n",port->index,port->name,
                (double)MESH_PLATFORM_NS/1e9);
    else say("link %u (%s): its %s out %.1f s: its port concluded around it (design/transport.md)\n",port->index,
             port->name,opening?"open's job":l->job==LINK_JOB_MAKE?"make job":"connect job",
             (double)(now-(opening?port->job_at:l->job_at))/1e9);
    port->open=0;
    port_lost(port,ETIMEDOUT);
  }
  atomic_store_explicit(&stalled,any,memory_order_relaxed);
}
/* a peer's LINK: every lane of its pairing and the address they were made on (its port's LID and GID as its make read
   them, which its hello's may predate: the connect's address, and an end of each stream's identity alike at both ends,
   stream_wire), taken once, before this end's connect (a LINK heard again is let be: the first stands), its lanes this
   end's plan's (both ends plan from the same two budgets: design/HETEROGENEOUS-LINKS.md); one naming other lanes loses
   the link, since a lane can be connected only to a queue pair its peer made (a port whose child was asked for its
   lanes greets no peer once its link is lost until its next child opens it: design/transport.md section 14, "A
   process a port") */
static void link_heard(struct port *port,const struct message *heard){
  struct link *l=&port->link;
  if(l->heard || (l->state!=LINK_DOWN && l->state!=LINK_MADE))return;
  if(heard->count!=plan_lanes(l->plan)){
    say("link %u (%s): node %u's LINK names %u lanes, this end's plan %u (plan %#x): %s\n",port->index,port->name,port->peer,heard->count,
        plan_lanes(l->plan),l->plan,port->heard.count?"the two ends planned apart":"its bridge states no budget and makes every lane, "
        "and this port holds fewer: the link pairs once that node runs a bridge that states its budget");
    port_lost(port,EPROTO);
    return;
  }
  memcpy(l->peer,heard->pair,sizeof l->peer);
  port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
  l->heard=1;
  link_turn(port);
}


/* design/transport.md: the bridge's wire, a message a kind: a record, the peer's farewell (its "release
   everything": the link lost at once, so nothing is posted toward it again, design/transport.md) or its
   LINK; a message of any other kind is let be (AGENTS.md, "Fail toward access": unknown input, keep going) */
static void message_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(heard->kind==RECORD)record_heard(port,payload,heard->bytes);
  else if(heard->kind==FAREWELL){port->parted=1;port->farewell=heard->code;port_lost(port,ECONNRESET);}
  else if(heard->kind==LINK)link_heard(port,heard);
}
/* a message heard on a port's control socket: before its peer's hello, that hello alone (any other let be); the hello
   names the peer (the node it states, whatever node's hello this end dialed), and pairs the link, its plan the fewer
   queue pairs of the two ends', design/HETEROGENEOUS-LINKS.md, its lanes made (link_turn); a peer whose blocks are
   another size loses the link, since its SENDs and receives would cover other frames than this end's (TN3205); then
   each taken (message_heard) */
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(!port->greeted){
    if(heard->kind!=HELLO)return;
    if(heard->block!=M->block*M->pgsz){port_lost(port,EPROTO);return;}
    port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;port->greeted=1;
    const uint32_t theirs=hello_budget(heard->count);
    port->link.plan=link_plan(MIN(port->budget,theirs));
    struct mesh_link_info *link=&mesh_links(M)[port->index];
    atomic_store_explicit(&link->bandwidth,port->bandwidth,memory_order_relaxed);
    atomic_fetch_add_explicit(&link->pairings,1,memory_order_relaxed);
    say("link %u (%s): paired with node %u (queue pairs for the link: this port's %u, its %u%s; plan %#x, %u lanes, %u stream%s)\n",
        port->index,port->name,port->peer,port->budget,theirs,heard->count?"":", its bridge stating none",port->link.plan,
        plan_lanes(port->link.plan),plan_streams(port->link.plan),plan_streams(port->link.plan)==1?"":"s");
    link_set(port,port->peer,MESH_PAIRED,0);
    /* a port newly paired is sent every record the map holds, this node's first */
    for(uint64_t i=0;port->greeted && i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node!=M->node)
      port_say(port,(struct message){.kind=RECORD},mesh_records(M)+i,sizeof(struct mesh_record));
    if(port->greeted)link_turn(port);
    return;
  }
  message_heard(port,heard,payload);
}

/* design/transport.md section 14, #loss: a link lost has its stream's landed words ~0 (streams_cancel: every client
   waiting on one reads the loss); its lanes are served no more before its control socket closes (its child told,
   CHILD_HALT: no post on them once the peer can read the loss); no verbs call is made (design/RDMA-KERNEL-RECOVERY.md
   "October 18").  A link whose child was asked for its lanes (`lanes`) waits for that child's end (`recycle`: child_turn
   asks it once the port's quiet window has ended), whose exit the kernel's release of the lanes is, the port closed and
   greeting no peer meanwhile, and pairs again with the port's next child ("A process a port"); a port lost before its
   lanes were asked for and still open says hello at once (port_hello: the records go on).  Its quiet window begins (B9:
   no port job on it, and no exit, for MESH_PLATFORM_NS, the operator's bound on the termination a loss may follow) */
static void child_send(struct port *port,struct child_message said);
void port_lost(struct port *port,int64_t code){
  quiet_until(&port->quiet_at,now+MESH_PLATFORM_NS);
  link_lost(port);
  if(port->child){
    child_send(port,(struct child_message){.kind=CHILD_HALT,.pairing=atomic_load(&port->link.pairing)});
    if(port->link.lanes && !stop){port->recycle=1;port->open=0;}
  }
  if(port->control>=0){close(port->control);port->control=-1;}
  if(port->greeted)port->next=now;
  /* design/transport.md: the peer named, and a peer that said farewell told from one lost */
  if(port->greeted||port->connecting){
    char reason[96];farewell_text(port->farewell,reason,sizeof reason);
    if(stop)say("link %u (%s): closed toward node %u: this bridge is stopping\n",port->index,port->name,port->peer);
    else if(port->peer && port->parted)say("link %u (%s): node %u parted, its bridge stopping (%s): code %lld\n",port->index,port->name,
                                            port->peer,reason,(long long)code);
    else if(port->peer)say("link %u (%s): node %u lost: code %lld\n",port->index,port->name,port->peer,(long long)code);
    else say("link %u (%s): lost: code %lld\n",port->index,port->name,(long long)code);
  }
  streams_cancel(port);
  port->greeted=port->parted=port->connecting=port->writing=0;port->farewell=0;port->peer=0;port->held=0;port->out_length=port->out_sent=0;
  link_set(port,0,port->open?MESH_PAIRING:MESH_STOPPED,code);
}
/* the control socket: non-blocking, its sends queued (port_say), its HELLO stating the queue pairs the port's device
   holds for a link (hello_budget); its peer's loss is the kernel's to tell (its end closed, a reset, the link's going
   off) */
static void port_greet(struct port *port){
  int on=1,flags=fcntl(port->control,F_GETFL);
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on)){
    port_lost(port,errno?errno:EIO);return;
  }
  port_watch(port,EVFILT_READ);
  port_say(port,(struct message){.kind=HELLO,.count=1+port->budget},NULL,0);
}
/* design/transport.md: the port a link-local peer reached, by its interface; none for a scope of 0 (no
   interface: a LAN address, or the index of an interface that is gone) */
static struct port *port_at(uint32_t interface){
  for(uint32_t p=0;interface && p<port_count;p++)if(ports[p].interface==interface)return ports+p;
  return NULL;
}
/* design/transport.md: a port's device opened, on its own queue (guarded), where it is to be (at the start,
   at its link's coming on or its interface's attaching, at a hello heard while it is not open, while it has no peer,
   and as the call its port was concluded around returns, ports_stalled), once its quiet window has ended and no port's
   job is out past MESH_PLATFORM_NS (design/transport.md B9, B12): the hellos' group joined on its
   interface, so the port hears them open or not (hello_heard, B16); its port read (none where the guard was not held
   within MESH_PLATFORM_NS: ETIMEDOUT, port_wait); a context that fails stays as it is, and the port with it, until its
   child ends (design/transport.md section 14: nothing made is forgotten or made again in a process).  The device is
   its child's: spawned here where the port has none (child_spawn: not while its last child's end is awaited, `recycle`,
   nor before the hold a child that ended unbidden set, `hold_to`), and asked to open it (CHILD_OPEN), its answer the
   job's end (child_heard) */
static int child_spawn(struct port *port);
static void port_open(struct port *port){
  if(port->job || port->stuck || !port->opening || port->control>=0 || now<port->quiet_at || port->recycle || now<port->hold_to ||
     now<port->retry_at || atomic_load_explicit(&stalled,memory_order_relaxed))return;
  if(!port->child && child_spawn(port))return;
  port->opening=0;
  port->interface=if_nametoindex(port->name);
  struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=port->interface};
  if(port->interface)setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
  port->job=1;port->job_at=now;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  child_send(port,(struct child_message){.kind=CHILD_OPEN});
}
/* the child's open of its port's device, on its port's queue, guarded; its end told the child's loop (child_main) */
static void device_open(struct port *port){
  port->job=1;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    const int raised=guard_raise(port);
    int refused=0,result=raised<0?ETIMEDOUT:device_up(&port->device,&wire,&refused);
    if(!result)result=device_gid(&port->device,port->name);
    if(!result && !port->device.gid_matched){
      const uint8_t *g=port->device.gid.raw;
      say("link %u (%s): no GID of %s is a link-local address %s holds; GID[%u] %02x%02x:%02x%02x:%02x%02x:%02x%02x:"
          "%02x%02x:%02x%02x:%02x%02x:%02x%02x used\n",port->index,port->name,port->device.name,port->name,port->device.gid_index,
          g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7],g[8],g[9],g[10],g[11],g[12],g[13],g[14],g[15]);
    }
    guard_lower(port,raised);
    port->result=result;port->refused=refused;
    atomic_store_explicit(&port->done,1,memory_order_release);
    wake();
  });
}
/* design/transport.md: a port's opening taken (its address and bandwidth as its child told them, CHILD_OPENED): a
   device that failed to open opens again at the next hello heard on it (hello_heard) or its link's next coming on; one
   whose registration of the region it refused (it keeps the extents it took: device_up logged the extent and why) does
   so too, the k-th refusal in a row no sooner than 60 2^(k-1) s on, at most an hour, unless its link comes on again
   (network_heard), since each try wires more of the node's memory (design/rubrics.md, Failure across interfaces 4 and
   6: the Mini's watchdog halted it once its memory was wired whole, 10-20); one opened says hello at once (port_hello) */
static void port_opened(struct port *port){
  const int result=port->result;
  port->job=0;
  if(result){
    if(port->refused){
      port->refusals++;
      const uint64_t hold=MIN(UINT64_C(60)<<MIN(port->refusals-1,6u),UINT64_C(3600));
      port->retry_at=now+hold*UINT64_C(1000000000);
      say("link %u (%s): the region's registration refused (%s), %u in a row: opened again at a hello heard on it from %llu s "
          "on, or at its link's coming on\n",port->index,port->name,strerror(port->refused),port->refusals,(unsigned long long)hold);
    }
    port->open=0;if(port->control<0)link_set(port,0,MESH_STOPPED,result);
    return;
  }
  port->open=1;port->refusals=0;
  if(port->control>=0)return;
  link_set(port,0,MESH_PAIRING,0);
  port->next=now;
}
/* design/transport.md: an open port with no peer says hello to every node on its link, every MESH_PAUSE_NS (a
   paced emission, deciding nothing: a peer that starts later is heard as it says its own) */
static void port_hello(struct port *port){
  if(!port->open || port->control>=0 || port->job || now<port->next)return;
  port->next=now+MESH_PAUSE_NS;
  setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_IF,&port->interface,sizeof port->interface);
  struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=everyone,.sin6_scope_id=port->interface};
  struct message said={.magic=XMAGIC,.kind=HELLO,.node=M->node,.boot=M->boot,.block=M->block*M->pgsz,.port=port->index};
  sendto(hello,&said,sizeof said,0,(struct sockaddr *)&to,sizeof to);
}
/* design/transport.md: a node heard on a port: the lower node dials the higher's listener on the address
   it heard from, the higher answers its hello so the lower hears it at once; a port not open is opened (its open's job
   out aside), so a port whose link job or open failed pairs again on hearing its peer, never on a clock */
static void hello_heard(void){
  struct message heard;struct sockaddr_in6 from;socklen_t length=sizeof from;
  while(recvfrom(hello,&heard,sizeof heard,0,(struct sockaddr *)&from,&length)==(ssize_t)sizeof heard){
    struct port *port=port_at(from.sin6_scope_id);
    length=sizeof from;
    if(!port || heard.magic!=XMAGIC || heard.kind!=HELLO || heard.node==M->node || port->control>=0)continue;
    if(!port->open){if(!port->job)port->opening=1;continue;}
    if(M->node>heard.node){
      struct message said={.magic=XMAGIC,.kind=HELLO,.node=M->node,.boot=M->boot,.block=M->block*M->pgsz,.port=port->index};
      sendto(hello,&said,sizeof said,0,(struct sockaddr *)&from,sizeof from);
      continue;
    }
    from.sin6_port=htons((uint16_t)atoi(MESH_PORT));
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) ||
       (connect(port->control,(struct sockaddr *)&from,sizeof from) && errno!=EINPROGRESS)){port_lost(port,errno);continue;}
    port->peer=heard.node;port->connecting=1;
    port_watch(port,EVFILT_WRITE);
  }
}
/* a control socket's messages, each a header and its `bytes`, each whole one taken */
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
/* design/transport.md: the link pages of every port the region has, leased at the bridge's start beside the
   map's, so a port found later (an interface attached once clients hold the region) has them; a port whose pages the
   region could not give takes them as its link first pairs (link_turn).  Each is recorded in the region
   (mesh-observe.h struct mesh_bridge_pages), so a bridge adopting the region takes the same pages (section 2) */
static uint32_t link_page[MESH_PORTS];
/* design/transport.md section 10: the all-ones block every dud is sent from (lane_send), taken and filled at the
   bridge's start, never written again */
uint32_t ones=MESH_ABSENT;
uint64_t held_pages;
static void link_reserve(void){
  struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  for(uint32_t p=0;p<MESH_PORTS;p++){
    if(own->link[p]==MESH_ABSENT)own->link[p]=mesh_arena_alloc(&bridge,link_pages(),M->block);
    if((link_page[p]=own->link[p])!=MESH_ABSENT)held_pages+=link_pages();
  }
  if(own->ones==MESH_ABSENT && (own->ones=mesh_arena_alloc(&bridge,M->block,M->block))==MESH_ABSENT)die("the duds' block");
  ones=own->ones;
  held_pages+=M->block;
  memset(mesh_at(M,ones),0xff,(size_t)M->block*M->pgsz);
}
/* design/transport.md: every RDMA device of the node a port, in the order found, read on the finder's queue
   (at the start, behind IOKit's quiet registry, and where an interface attached or its link came on that no port has,
   on the event loop's turn once the windows have ended and no call is stuck: find_wanted, design/transport.md
   B9, B9b, B12) and added on the event loop */
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
    *port=(struct port){.index=port_count,.control=-1,.child_fd=-1,.opening=1};
    /* design/transport.md: the link's block, the pages leased for the port's index */
    port->link.pages=link_pages();
    port->link.page=link_page[port_count];
    port->link.lists=MESH_ABSENT;
    port->link.provider.device=&port->device;
    snprintf(port->device.name,sizeof port->device.name,"%s",devices[i]);snprintf(port->name,sizeof port->name,"%s",devices[i]+5);
    /* design/transport.md B9b: counted for a termination's queue (quiet_all, peer_port) once whole, then
       held to the window a termination heard meanwhile set (`exit_at`) */
    __atomic_store_n(&port_count,port_count+1,__ATOMIC_SEQ_CST);
    quiet_until(&port->quiet_at,atomic_load(&exit_at));
    link_set(port,0,MESH_STOPPED,0);
  }
}
/* design/transport.md: a port's link off or its interface detached (the link lost, its lanes left as they
   are, design/transport.md B6 (b), and the port down until its link comes on: its device opened then,
   port_open); an interface of any name that attached or whose link came on and is no port yet: the node's devices to
   be read again (a link's coming on stays a reading, for a device registered after its interface attached), the read
   put off to the windows' end (find_wanted, kept while a read is out).  At the stop (`stopping`), a port's link going off only extends
   its quiet window, which the exit waits out (B9) */
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

/* design/transport.md B9b (EXIT-RECLAIM.md P2, and P3 as a window in place of IOServiceWaitQuiet): a
   Thunderbolt RDMA peer interface's termination, which IOKit tells in its first phase (kIOTerminatedNotification), on
   a queue of its own: the terminations heard (`terminations`), and the ports whose peer interface is terminating
   (`terminated`, a bit a port index), which the event loop concludes (terminations_turn) */
static _Atomic uint32_t terminated,terminations;
/* the exit's window and every found port's held to `until` at least (a port found meanwhile takes `exit_at`'s as it is
   counted: ports_found) */
static void quiet_all(uint64_t until){
  quiet_until(&exit_at,until);
  const uint32_t count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);
  for(uint32_t p=0;p<count;p++)quiet_until(&ports[p].quiet_at,until);
}
/* the port whose interface is the terminating peer interface's, by mesh-fabric-init.sh's rule (an
   AppleThunderboltRDMAInterface rdma_<port> and its peer interface carry the same "RID"); NULL where none is found
   (the property gone with the terminating service, or no port of that interface yet) */
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
/* the iterator drained first, which arms the notification again: for each terminated service the exit's window and
   every port's held to MESH_PLATFORM_NS from now (no port job and no exit inside the termination, whichever port's it
   is) and the event loop woken, before any property is read; then the port each maps to halted (port_halt: its
   pairing lost, served no more, marked for the event loop's conclusion); a service that maps to no port holds the
   windows alone; each released */
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
/* armed at the start, before IOKit's registry is waited on and the devices are read: the notification on a serial
   queue of its own, its iterator drained once (which arms it); without it a link's going off is the first sign */
static void terminations_watch(void){
  IONotificationPortRef notifications=IONotificationPortCreate(kIOMainPortDefault);
  io_iterator_t iterator=IO_OBJECT_NULL;
  kern_return_t error=KERN_FAILURE;
  if(notifications){
    IONotificationPortSetDispatchQueue(notifications,dispatch_queue_create("mesh.terminations",DISPATCH_QUEUE_SERIAL));
    error=IOServiceAddMatchingNotification(notifications,kIOTerminatedNotification,IOServiceMatching("AppleThunderboltRDMAPeerInterface"),
                                           peer_terminated,NULL,&iterator);
  }
  if(error){say("bridge: Thunderbolt RDMA peer interfaces' terminations not heard (%#x): a link's going off is its first sign\n",(unsigned)error);return;}
  peer_terminated(NULL,iterator);
}
/* design/transport.md (EXIT-RECLAIM.md P4): the provider's doorbell fault absorbed.  A SIGSEGV or SIGBUS
   whose pc lies in the provider's text (`provider`, read at the start) on a thread that posts on a port's queue
   pairs (struct port `thread`) has one anonymous page mapped at the faulting page by hint, never MAP_FIXED, so nothing
   mapped is replaced: where it got that page (no mapping was there: the doorbell the driver unmapped in a termination),
   or the port's bit in `faulted` is set (its other thread got the page first; else the store faults again until the
   event loop takes the bit), the store lands in that page, no doorbell rings, and the port is halted for the event loop
   to lose (port_halt, terminations_turn: ENETDOWN, no verbs call).  Where the page could not be mapped and the bit is
   clear, the store is made once more (`retried`, the port's last such page: its other thread may have mapped it, its
   bit not yet set or already taken by the event loop), and only a second fault at that page is any other fault.  The
   bridge does not exit.  Any other fault: the default action restored, the store made again, the process
   ended.  On the thread's alternate stack where it has one (port_progress); mmap, munmap and sigaction alone */
static uintptr_t provider[2];
static _Atomic uint32_t faulted;
static _Atomic uintptr_t retried[MESH_PORTS];
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
/* the provider's text read and the faults' handler installed, by a child before it opens its device */
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
/* the terminations and the absorbed faults taken on the event loop: each termination logged, and each port whose peer
   interface is terminating lost (port_lost, ETIMEDOUT: its socket closed, its stream's words ~0), its lanes left as they
   are, no verbs call; each port a fault was absorbed on lost as at its link's going off (open 0, ENETDOWN:
   network_heard) */
static uint32_t terminations_told;
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
    say("link %u (%s): its peer interface terminating: its link lost\n",ports[p].index,ports[p].name);
    port_lost(ports+p,ETIMEDOUT);
  }
  for(uint32_t p=0;p<port_count;p++)if(faults>>p&1){
    say("link %u (%s): a fault of its provider's doorbell absorbed: its link lost (design/transport.md)\n",
        ports[p].index,ports[p].name);
    ports[p].open=0;port_lost(ports+p,ENETDOWN);
  }
}

/* design/transport.md section 14, "A process a port": the parent's side of a port's child.  The bridge is a parent
   (the region, the map and records, the clients, discovery, hellos and the control sockets) and a child a port (its
   device, its link's queue pairs and completion queues, its progress threads): mesh-flow spawned again, the same image
   (`self_path`), told the contract's version it is spawned for (mesh-bridge.h MESH_CHILD_VERSION), in a process group of its own (a job's group signalled at its end reaches no verbs owner), with
   the parent's socket at descriptor 3, the parent's stderr, and nothing else.  The child's exit is the one release of
   its port's verbs objects (the kernel's), and a link once connected and then lost pairs again with the port's next
   child */
static char self_path[PATH_MAX];
/* a message to the port's child, whole; one not taken (the child gone) is dropped, its end reaped (child_reaped) */
static void child_send(struct port *port,struct child_message said){
  const char *at=(const char *)&said;
  size_t left=sizeof said;
  while(port->child_fd>=0 && left){
    const ssize_t sent=send(port->child_fd,at,left,0);
    if(sent<0 && errno==EINTR)continue;
    if(sent<=0)return;
    at+=sent;left-=(size_t)sent;
  }
}
static void child_reaped(struct port *port);
/* a process, not this bridge nor one of its children, that is a child of the port's device of any version (its argv
   `--child VERSION DEVICE ...`, the three words every version keeps): a child of a launch whose parent ended unbidden, which holds the device's queue
   pairs until its exit (child_leave, at most 2 MESH_PLATFORM_NS); its pid, or 0.  A device's queue pairs are one
   child's at a time: a link takes as many as its device holds for it, up to MESH_LINK_LANES (ten of a macOS 26
   device's ten, design/HETEROGENEOUS-LINKS.md) */
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
/* the port's child spawned (its socket's messages and its exit watched, tagged by the port); 0, or the error, the
   port's next spawn held MESH_PAUSE_NS */
static int child_spawn(struct port *port){
  int pair[2],on=1;
  const pid_t elsewhere=child_elsewhere(port);
  if(elsewhere){
    if(elsewhere!=port->elsewhere)say("link %u (%s): process %d, a child of its device from an earlier launch, holds its queue "
                                      "pairs until its exit; its own child spawned after it\n",port->index,port->name,(int)elsewhere);
    port->elsewhere=elsewhere;
    return EBUSY;
  }
  /* design/transport.md section 2: the child's entry among the region's holders, one no live child holds; where every
     one is held, the child waits as for an earlier launch's (a holder of the region is never left unrecorded) */
  struct mesh_holder *holders=mesh_bridge_pages(M)->child;
  uint32_t entry=0;
  while(entry<2*MESH_PORTS && mesh_holds((pid_t)holders[entry].pid,holders[entry].start))entry++;
  if(entry==2*MESH_PORTS){port->hold_to=now+MESH_PAUSE_NS;return EBUSY;}
  if(socketpair(AF_UNIX,SOCK_STREAM,0,pair)){
    const int error=errno?errno:EIO;
    say("link %u (%s): its child not spawned: %s\n",port->index,port->name,strerror(error));
    port->hold_to=now+MESH_PAUSE_NS;
    return error;
  }
  setsockopt(pair[0],SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on);
  char index[16],page[16],alive[24];
  snprintf(index,sizeof index,"%u",port->index);snprintf(page,sizeof page,"%u",ones);
  snprintf(alive,sizeof alive,"%llu",(unsigned long long)MESH_KEEPALIVE_NS);
  char version[16];
  snprintf(version,sizeof version,"%u",MESH_CHILD_VERSION);
  char *arguments[]={self_path,"--child",version,port->device.name,index,page,alive,NULL};
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
  pid_t pid=0;
  const int error=posix_spawn(&pid,self_path,&actions,&attributes,arguments,*_NSGetEnviron());
  posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attributes);
  close(pair[1]);
  if(error){
    close(pair[0]);
    say("link %u (%s): its child not spawned: %s\n",port->index,port->name,strerror(error));
    port->hold_to=now+MESH_PAUSE_NS;
    return error;
  }
  port->child=pid;port->child_fd=pair[0];port->child_at=now;port->child_held=0;port->ending=0;
  holders[entry]=(struct mesh_holder){(uint32_t)pid,0,mesh_started(pid)};
  port->holder=entry;
  say("link %u (%s): its child %d spawned\n",port->index,port->name,(int)pid);
  struct kevent64_s watch;
  EV_SET64(&watch,(uint64_t)pair[0],EVFILT_READ,EV_ADD,0,0,TAG_CHILD|port->index,0,0);
  if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){close(port->child_fd);port->child_fd=-1;}
  EV_SET64(&watch,(uint64_t)pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,TAG_CHILD|port->index,0,0);
  if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){child_reaped(port);return ECHILD;}
  return 0;
}
/* a link job asked of the port's child: its lanes made (the link's pages named) or connected to the peer's
   (the peer's lanes as its LINK told them, its address as its hello did, its node and the link's bandwidth), under
   the link's pairing as it is asked; its end the child's answer (child_heard), taken by link_turn */
static void child_job(struct port *port,int job){
  struct link *l=&port->link;
  l->job=job;l->job_pairing=atomic_load_explicit(&l->pairing,memory_order_relaxed);l->job_at=now;
  atomic_store_explicit(&l->done,0,memory_order_relaxed);
  struct child_message said={.kind=job==LINK_JOB_MAKE?CHILD_MAKE:CHILD_CONNECT,.pairing=l->job_pairing,.page=l->page,.lanes=l->plan};
  if(job==LINK_JOB_CONNECT){
    memcpy(said.peer,l->peer,sizeof said.peer);said.lid=port->heard.lid;memcpy(said.gid,port->heard.gid,16);
    said.node=port->peer;said.bandwidth=port->bandwidth;
  }
  child_send(port,said);
}
/* the child's messages, each whole one taken: an open's, a make's or a connect's end for the job out (a connect of a
   pairing since lost has its streams cancelled again: the child may have opened them after the loss's cancel); the
   lanes a progress thread found failed, for the link's pairing while it is up; a fault absorbed, a guard not held, a
   termination heard, each its port's bit for the event loop's conclusion (terminations_turn, ports_stalled).  The
   socket's end closes it; the child's exit is reaped (child_reaped) */
static void child_heard(struct port *port){
  struct link *l=&port->link;
  const uint32_t bit=UINT32_C(1)<<port->index;
  for(;;){
    const ssize_t got=recv(port->child_fd,port->child_in+port->child_held,sizeof port->child_in-port->child_held,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0){close(port->child_fd);port->child_fd=-1;return;}
    if((port->child_held+=(uint32_t)got)<sizeof port->child_in)continue;
    port->child_held=0;
    struct child_message heard;
    memcpy(&heard,port->child_in,sizeof heard);
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
        memcpy(l->qpn,heard.qpn,sizeof l->qpn);l->psn=heard.psn;l->share=heard.share;
        port->lid=heard.lid;memcpy(port->gid,heard.gid,16);
      }
      atomic_store_explicit(&l->done,1,memory_order_release);
    }
    else if(heard.kind==CHILD_BROKEN && current && l->state==LINK_UP)atomic_fetch_or(&l->broken,heard.broken);
    else if(heard.kind==CHILD_FAULTED)atomic_fetch_or(&faulted,bit);
    else if(heard.kind==CHILD_UNGUARDED)atomic_fetch_or(&unguarded,bit);
    else if(heard.kind==CHILD_TERMINATED)atomic_fetch_or(&terminated,bit);
  }
}
/* a port whose link was lost once its lanes were asked for (`recycle`) asks its child to end once its quiet window and
   the exit's have ended (B9, B9b: no release inside a termination), no port's call is stuck (B12) and no job of its own
   is out */
static void child_turn(struct port *port){
  if(!port->recycle || !port->child || port->ending || port->job || port->link.job || now<atomic_load(&port->quiet_at) ||
     now<atomic_load(&exit_at) || atomic_load_explicit(&stalled,memory_order_relaxed))return;
  child_send(port,(struct child_message){.kind=CHILD_END});
  port->ending=1;
  say("link %u (%s): its child %d asked to end, its lanes released at its exit (design/transport.md section 14)\n",
      port->index,port->name,(int)port->child);
}
/* the port's child ended: its last messages taken, its exit reaped, a job of it out ended (ECHILD); one that ended
   unbidden loses the link (port_lost, ECHILD) and is spawned again once the port's quiet window has passed, and the
   k-th in a row within MESH_CHILD_SETTLE_NS of its spawn, k from 2, holds the port's next spawn 60 2^(k-2) s, at most an
   hour (the boot ledger's rule, a port's: restarted, "not crashing again of course", the operator 10-20).  Its streams cancelled again,
   its link down with no lanes, and the port to be opened by its next child */
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
  if(asked){port->aborts=0;say("link %u (%s): its child %d ended\n",port->index,port->name,(int)pid);}
  else{
    port->aborts=now-port->child_at<MESH_CHILD_SETTLE_NS?port->aborts+1:0;
    const uint64_t hold=port->aborts>1?MIN(UINT64_C(60)<<MIN(port->aborts-2,6u),UINT64_C(3600)):0;
    port->hold_to=now+hold*UINT64_C(1000000000);
    say("link %u (%s): its child %d ended unbidden (%s %d); its next spawned in %llu s\n",port->index,port->name,(int)pid,
        WIFSIGNALED(status)?"signal":"status",WIFSIGNALED(status)?WTERMSIG(status):WEXITSTATUS(status),(unsigned long long)hold);
    port->open=0;
    port_lost(port,ECHILD);
  }
  if(port->job){port->result=ECHILD;port->refused=0;atomic_store_explicit(&port->done,1,memory_order_release);}
  if(l->job){l->result=ECHILD;atomic_store_explicit(&l->done,1,memory_order_release);}
  streams_cancel(port);
  l->state=LINK_DOWN;l->lanes=l->heard=0;l->share=0;l->plan=0;
  memset(l->qpn,0,sizeof l->qpn);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
  port->opening=1;
}

/* design/transport.md: the clients' control word, watched for the event loop */
static void *control_watch(void *unused){
  (void)unused;
  while(!stop){
    uint64_t seen=atomic_load_explicit(&M->control,memory_order_acquire);
    wake();
    os_sync_wait_on_address(&M->control,seen,sizeof M->control,OS_SYNC_WAIT_ON_ADDRESS_SHARED);
  }
  return NULL;
}

/* design/transport.md: the bridge's configuration, the canonical property list bin/mesh-user.sh writes from
   the consumer's document (rdma/bridge_configuration.py): its name and its region in bytes (`region_bytes`); a `keepalive`
   it states is not read (the keepalive's pace is the bridge's: MESH_KEEPALIVE_NS); NULL, or why it is refused */
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

/* design/transport.md: the node's name, its LocalHostName (else its host name), read at the start and, on
   each liveness turn, on the finder's queue (configd answers there, never on the event loop: host_read); the event loop
   takes a change (host_taken), so a renamed node's next record carries its new name (the map's label of it; a node is
   its number) */
static void host_name(char *name){
  CFStringRef local=SCDynamicStoreCopyLocalHostName(NULL);
  if(!local || !CFStringGetCString(local,name,MESH_NAME_BYTES,kCFStringEncodingUTF8))gethostname(name,MESH_NAME_BYTES);
  if(local)CFRelease(local);
  name[MESH_NAME_BYTES-1]=0;
}
static char host_heard[MESH_NAME_BYTES];
static _Atomic int host_reading,host_read_done;
static void host_read(void){
  if(atomic_load(&host_read_done) || atomic_exchange(&host_reading,1))return;
  dispatch_async(finder,^{
    char name[MESH_NAME_BYTES]={0};
    host_name(name);
    memcpy(host_heard,name,sizeof name);
    atomic_store_explicit(&host_read_done,1,memory_order_release);
    atomic_store(&host_reading,0);
    wake();
  });
}
static void host_taken(void){
  if(!atomic_load_explicit(&host_read_done,memory_order_acquire))return;
  if(strcmp(host_heard,host)){
    say("node %u: renamed %s to %s\n",M->node,host,host_heard);
    memcpy(host,host_heard,sizeof host);
    record_changed();
  }
  atomic_store_explicit(&host_read_done,0,memory_order_release);
}

/* design/transport.md section 14, "A process a port": a port's child, `mesh-flow --child VERSION DEVICE INDEX ONES
   KEEPALIVE` as its parent spawned it (child_spawn), VERSION the contract's (mesh-bridge.h MESH_CHILD_VERSION): the parent's region mapped (its magic, version and
   bridge the parent's), its port the parent's port INDEX on DEVICE (its streams' words the region's), the duds' block
   ONES the parent's, KEEPALIVE the parent's pace (a child of this build keeps its own, MESH_KEEPALIVE_NS, so a parent of
   an earlier build that stated 0 there sets no pace), the provider's faults absorbed (fault_heard) and its peer interfaces'
   terminations heard (B9b: its lanes served no more at once, port_halt).  It takes the parent's messages on descriptor 3
   in order: an open, a make or a connect of its link as a job on its port's queue (link_job, device_open; one asked
   while a job is out, or while a halt of its own is not yet answered by the parent's, is refused), a halt (no lane
   served, a job out lapsed), an end, and a message of any other kind let be; and tells each job's end, naming the pairing it was asked for.  A connect that
   ends while its pairing stands opens the link's streams (streams_open) and then serves every lane, its pairing read
   again (port_halt's order).  The child ends at its parent's word, once its own windows have ended (at most
   MESH_PLATFORM_NS on), or at its parent's loss (the socket's end), its lanes served no more, MESH_PLATFORM_NS and its
   windows on (at most twice that): its exit, the kernel's release of its device's objects, no verbs call made */
static uint32_t child_asked;
static uint64_t child_bandwidth;
static int child_halted;
static unsigned char child_in[sizeof(struct child_message)];
static uint32_t child_got;
static void child_tell(struct child_message said){
  const char *at=(const char *)&said;
  size_t left=sizeof said;
  while(left){
    const ssize_t sent=send(3,at,left,0);
    if(sent<0 && errno==EINTR)continue;
    if(sent<=0)return;
    at+=sent;left-=(size_t)sent;
  }
}
static __attribute__((noreturn)) void child_leave(struct port *port,int asked){
  atomic_fetch_add(&port->link.pairing,1);
  atomic_store(&port->link.served,0);
  atomic_store_explicit(&port->closing,1,memory_order_release);
  guard_wake(port);
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
  if(heard->kind==CHILD_HALT){
    atomic_fetch_add(&l->pairing,1);atomic_store(&l->served,0);guard_wake(port);
    child_halted=0;
    return;
  }
  if(heard->kind==CHILD_END)child_leave(port,1);
  if(heard->kind!=CHILD_MAKE && heard->kind!=CHILD_CONNECT)return;
  const int job=heard->kind==CHILD_MAKE?LINK_JOB_MAKE:LINK_JOB_CONNECT;
  int refused=l->job?EBUSY:child_halted || atomic_load(&faulted) || atomic_load(&unguarded) || atomic_load(&terminated)?ECANCELED:0;
  if(!refused && job==LINK_JOB_CONNECT){
    memcpy(l->peer,heard->peer,sizeof l->peer);port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
    port->peer=heard->node;child_bandwidth=heard->bandwidth;
    refused=lanes_room(port);
  }
  if(refused){
    child_tell((struct child_message){.kind=job==LINK_JOB_MAKE?CHILD_MADE:CHILD_CONNECTED,.pairing=heard->pairing,.result=refused});
    return;
  }
  child_asked=heard->pairing;
  if(job==LINK_JOB_MAKE){l->page=heard->page;l->plan=heard->lanes;}
  link_job(port,job);
}
static void child_jobs(struct port *port){
  struct link *l=&port->link;
  if(port->job && atomic_load_explicit(&port->done,memory_order_acquire)){
    port->job=0;
    struct child_message said={.kind=CHILD_OPENED,.result=port->result,.refused=port->refused};
    if(!port->result){
      port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
      said.lid=port->lid;memcpy(said.gid,port->gid,16);said.bandwidth=port_bandwidth(&port->device.port);
      said.lanes=port->device.max_pairs;
    }
    child_tell(said);
  }
  if(!l->job || !atomic_load_explicit(&l->done,memory_order_acquire))return;
  const int job=l->job;
  l->job=0;
  struct child_message said={.kind=job==LINK_JOB_MAKE?CHILD_MADE:CHILD_CONNECTED,.pairing=child_asked,.result=l->result};
  if(!said.result && l->job_pairing!=atomic_load(&l->pairing))said.result=ECANCELED;
  if(!said.result && job==LINK_JOB_MAKE){
    for(uint32_t k=0;k<MESH_LINK_LANES;k++)said.qpn[k]=l->plan>>k&1?l->provider.queues[k].pair->qp_num:UINT32_MAX;
    said.psn=l->provider.psn;said.share=l->share;
    port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
    said.lid=port->lid;memcpy(said.gid,port->gid,16);
  }
  else if(!said.result){
    streams_open(port,port->peer,child_bandwidth);
    atomic_store(&l->served,l->plan);
    if(atomic_load(&l->pairing)!=l->job_pairing){atomic_store(&l->served,0);said.result=ECANCELED;}
    guard_wake(port);
  }
  child_tell(said);
}
static void child_reports(struct port *port){
  const uint32_t broken=atomic_exchange_explicit(&port->link.broken,0,memory_order_acq_rel);
  if(broken)child_tell((struct child_message){.kind=CHILD_BROKEN,.pairing=child_asked,.broken=broken});
  if(atomic_exchange(&faulted,0)){child_halted=1;child_tell((struct child_message){.kind=CHILD_FAULTED});}
  if(atomic_exchange(&unguarded,0)){child_halted=1;child_tell((struct child_message){.kind=CHILD_UNGUARDED});}
  if(atomic_exchange(&terminated,0)){child_halted=1;child_tell((struct child_message){.kind=CHILD_TERMINATED});}
}
static void child_read(struct port *port){
  for(;;){
    const ssize_t got=recv(3,child_in+child_got,sizeof child_in-child_got,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0)child_leave(port,0);
    if((child_got+=(uint32_t)got)<sizeof child_in)continue;
    child_got=0;
    struct child_message heard;
    memcpy(&heard,child_in,sizeof heard);
    child_command(port,&heard);
  }
}
static int child_main(int argc,char **argv){
  signal(SIGPIPE,SIG_IGN);
  const unsigned long index=argc>=7?strtoul(argv[4],NULL,10):MESH_PORTS;
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
  *port=(struct port){.index=(uint32_t)index,.control=-1,.child_fd=-1,.work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL)};
  port->link.provider.device=&port->device;port->link.page=port->link.lists=MESH_ABSENT;
  snprintf(port->device.name,sizeof port->device.name,"%s",argv[3]);snprintf(port->name,sizeof port->name,"%s",argv[3]+5);
  ones=(uint32_t)strtoul(argv[5],NULL,10);
  mach_timebase_info_data_t base;
  mach_timebase_info(&base);
  alive_ticks[0]=MESH_PLATFORM_NS/5*base.denom/base.numer;alive_ticks[1]=MESH_KEEPALIVE_NS*base.denom/base.numer;
  __atomic_store_n(&port_count,1,__ATOMIC_SEQ_CST);
  struct kevent64_s event;
  if((events=kqueue())<0)die("child events");
  EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
  EV_SET64(&event,3,EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
  terminations_watch();
  for(;;){
    now=clock_ns();
    child_jobs(port);
    child_reports(port);
    struct kevent64_s ready[4];
    struct timespec tick={(time_t)(MESH_PAUSE_NS/1000000000u),(long)(MESH_PAUSE_NS%1000000000u)};
    const int count=kevent64(events,NULL,0,ready,4,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++)if(ready[i].filter==EVFILT_READ && ready[i].ident==3)child_read(port);
  }
}

/* design/transport.md section 2, the region and the heartbeat: the node's region (MESH_NAME) opened.  The region a
   bridge before this one kept (a restart, SIGHUP; a crash) is adopted where its magic, layout version (MESH_LAYOUT) and
   layout (`geometry`, mesh_layout of this configuration's region) are this bridge's: the same memory, its clients'
   slots, rows and leases standing and the GPU buffers over them valid, its pages registered again by the port children
   (the same pages: nothing is wired twice, nothing reloaded).  A region a live bridge serves is never read: a second
   bridge of the node ends at its port's bind, before region_open.  A region of another layout, or one a stop released (layout version
   0), is made expired (its layout version 0: no client's mesh_open takes it, and each client's next lease or
   communicator returns the expiry, nccl-mesh.c attach_expired) and left named, and this bridge ends with EX_TEMPFAIL:
   its heartbeat unlinks it once no process holds it and opens a bridge again, so no second region is ever made beside
   a live client of the first (the 2026-10-20 watchdog panic, design/rubrics.md "Failure across interfaces"); a bridge
   unlinks only a shared memory object that never became a region (no magic: a bridge ended while making it).  A region that cannot
   be read is left as it is, and this bridge ends.  1 adopted, 0 made; *out its descriptor */
static int region_same(const struct hdr *a,const struct hdr *b){
  return a->pgsz==b->pgsz && a->block==b->block && a->rows==b->rows && a->streams==b->streams && a->clients==b->clients &&
         a->planes_off==b->planes_off && a->arena_off==b->arena_off && a->page_off==b->page_off && a->buffer_off==b->buffer_off &&
         a->bridge_off==b->bridge_off && a->stream_off==b->stream_off && a->client_off==b->client_off && a->data_off==b->data_off &&
         a->length==b->length;
}
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
      if(__atomic_load_n(&old->version,__ATOMIC_ACQUIRE)==MESH_LAYOUT && region_same(old,geometry)){M=old;*out=fd;return 1;}
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

/* the adopted region's bridge state made this bridge's, before any client reads it as live (`bridge_pid` set last): every
   stream not live, not free and held by none, its landed words ~0 (a lost pairing's holding ended with its bridge, and a
   wait on it reads the loss; the next pairing opens each new, streams_open), the links' table cleared, a client slot
   whose retirement a process that died left half done finished (mesh_retire's walk, which any party can repeat to the
   same end), the closed rows released, a map or health serial a bridge that died mid-write left odd made even (its
   readers wait while it is odd), and the map's records this bridge's to write again (map_adopt) */
static void region_adopt(void){
  for(uint32_t s=0;s<MESH_STREAMS;s++){
    struct mesh_stream *x=mesh_stream_at(M,s);
    atomic_store(&x->live,0);atomic_store(&x->free,0);atomic_store(&x->owner,0);atomic_store(&x->peer,0);atomic_store(&x->wire,0);
    atomic_store(&x->touching,0);
    atomic_store_explicit(&x->sent,UINT64_MAX,memory_order_release);atomic_store_explicit(&x->landed,UINT64_MAX,memory_order_release);
  }
  struct mesh_link_info *links=mesh_links(M);
  for(uint32_t p=0;p<MESH_PORTS;p++){
    atomic_store(&links[p].peer,0);atomic_store(&links[p].phase,MESH_UNKNOWN);atomic_store(&links[p].code,0);atomic_store(&links[p].bandwidth,0);
  }
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *slot=(struct mesh_client *)((char *)M+M->client_off)+s;
    const uint64_t owner=atomic_load(&slot->owner);
    if(!(owner&MESH_RETIRING))continue;
    for(uint32_t row=0;row<mesh_rows(M);row++){
      struct mesh_buffer *buffer=&mesh_buffers(M)[row];
      if(atomic_load(&buffer->owner)!=(owner&~MESH_RETIRING))continue;
      if(buffer->pages)atomic_store(&buffer->closed,1);
      mesh_bits_clear(mesh_plane(M,MESH_ROW_OWN),row,1);
    }
    atomic_store(&slot->owner,0);
  }
  mesh_retired_release(M);
  if(atomic_load(&M->map)&1)atomic_fetch_add(&M->map,1);
  if(atomic_load(&M->health)&1)atomic_fetch_add(&M->health,1);
  map_adopt();
  atomic_store(&M->busy,0);atomic_store(&M->stopping,0);
}

/* design/transport.md: the node's resident bridge; its one operand the path of its configuration (no default); a port's
   child as its parent spawns it (child_main) */
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
  /* design/transport.md: every catchable signal whose default ends the process takes the stop, its filter
     added before its default is ignored, so none is lost (one arriving before takes its default action, before anything
     shared is made) */
  struct kevent64_s event;
  static const int stopping[]={SIGINT,SIGTERM,SIGHUP,SIGQUIT,SIGUSR1,SIGUSR2,SIGALRM,SIGVTALRM,SIGPROF,SIGXCPU,SIGXFSZ};
  if((events=kqueue())<0)die("bridge events");
  for(size_t i=0;i<sizeof stopping/sizeof *stopping;i++){
    EV_SET64(&event,(uint64_t)stopping[i],EVFILT_SIGNAL,EV_ADD,0,0,UINT64_MAX,0,0);
    if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
    signal(stopping[i],SIG_IGN);
  }
  signal(SIGPIPE,SIG_IGN);
  boots_hold(argv[1]);
  uuid_t uuid;struct timespec bound={5,0};
  if(gethostuuid(uuid,&bound)){say("host identity: %s; this bridge's node is drawn at random\n",strerror(errno));arc4random_buf(uuid,sizeof uuid);}
  uint32_t node=0;
  for(int i=0;i<16;i++)node=(node*0x01000193u)^uuid[i];
  node=(node&0x7fffffffu)?node&0x7fffffffu:1;
  /* design/transport.md: the node's hello port, the mesh's */
  const uint16_t service=(uint16_t)atoi(MESH_PORT);
  const uint32_t pg=(uint32_t)getpagesize();
  /* design/transport.md */
  if((uint64_t)pg+4096>16773120)die("transport chunk exceeds native request capacity");
  if(region_bytes<pg || region_bytes/pg>UINT32_MAX){say("bridge configuration %s refused: its region of %llu bytes is not between one page and 2^32 pages\n",name,(unsigned long long)region_bytes);exit(1);}
  struct hdr geometry={0};
  mesh_layout(&geometry,pg,1,(uint32_t)(region_bytes/pg));
  /* design/transport.md section 2: the bridge's port bound before its region is read, so a second bridge of the node
     (one started beside the heartbeat's, or while the last one ends) ends at its bind, never touching the region */
  int on=1;
  struct sockaddr_in6 any={.sin6_len=sizeof any,.sin6_family=AF_INET6,.sin6_port=htons(service),.sin6_addr=IN6ADDR_ANY_INIT};
  listener=socket(AF_INET6,SOCK_STREAM,0);hello=socket(AF_INET6,SOCK_DGRAM,0);network=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
  struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS};
  unsigned loop=0;
  if(listener<0 || hello<0 || network<0 || setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) ||
     setsockopt(hello,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_LOOP,&loop,sizeof loop) ||
     bind(listener,(struct sockaddr *)&any,sizeof any) || listen(listener,SOMAXCONN) || bind(hello,(struct sockaddr *)&any,sizeof any) ||
     fcntl(listener,F_SETFL,O_NONBLOCK) || fcntl(hello,F_SETFL,O_NONBLOCK) || fcntl(network,F_SETFL,O_NONBLOCK) || ioctl(network,SIOCSKEVFILT,&filter))
    die("bridge sockets");
  int fd=-1;
  const int adopted=region_open(&geometry,&fd);
  if(adopted)region_adopt();
  /* a boot is the bridge's start on the wall clock, so a node's later boot is a newer record */
  M->node=node;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);M->idle_since=M->boot;
  say("bridge node %u: region %s %s\n",node,MESH_NAME,adopted?"adopted from the bridge before this one (its clients' leases stand)":"made new");
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  bridge.M=M;
  host_name(host);
  finder=dispatch_queue_create("mesh.ports",DISPATCH_QUEUE_SERIAL);
  /* design/transport.md: the node's memory and hardware model, which its record carries with its region */
  size_t size=sizeof memory;
  if(sysctlbyname("hw.memsize",&memory,&size,NULL,0))memory=0;
  sysctl_text("hw.model",model,sizeof model);
  health_init();
  record_changed();
  if(!atomic_load(&M->map_nodes))die("the map's lease");
  link_reserve();
  int watched[3]={listener,hello,network};
  for(int i=0;i<3;i++){EV_SET64(&event,(uint64_t)watched[i],EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");}
  EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  mesh_bridge_pages(M)->bridge_start=mesh_started(getpid());
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();M->magic=MESH_MAGIC;
  pthread_t watcher;
  if(pthread_create(&watcher,NULL,control_watch,NULL))die("control watcher");
  /* design/transport.md B9b, B9: the peer interfaces' terminations heard from here on; the devices first read
     once IOKit's registry is quiet (no termination in its course: XNU's waitForState, its deadline MESH_PLATFORM_NS),
     else as they stand at the bound */
  terminations_watch();
  mach_timespec_t quiet={(unsigned)(MESH_PLATFORM_NS/1000000000u),(clock_res_t)(MESH_PLATFORM_NS%1000000000u)};
  const kern_return_t busy=IOKitWaitQuiet(kIOMainPortDefault,&quiet);
  if(busy)say("bridge node %u: IOKit's registry not quiet within %.1f s (%#x): its devices read as they stand\n",node,
              (double)MESH_PLATFORM_NS/1e9,(unsigned)busy);
  ports_find();
  say("bridge node %u (%s): configuration %s, region %llu bytes, arena %llu pages, rows %u, a link's keepalive after %llu ns "
      "without a SEND while it holds a stream (design/HETEROGENEOUS-LINKS.md section 10)\n",node,host,name,
      (unsigned long long)region_bytes,(unsigned long long)mesh_arena_pages(M),M->rows,(unsigned long long)MESH_KEEPALIVE_NS);
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
    health_turn();
    host_taken();
    int alive=now>=liveness;
    if(alive){
      liveness=now+MESH_PAUSE_NS;
      host_read();
      record_retell();
    }
    clients_turn(alive);
    struct kevent64_s ready[16];
    struct timespec tick={0,250000000};
    int count=kevent64(events,NULL,0,ready,16,0,&tick);
    now=clock_ns();
    if(boots_set && now-boots_since>=BOOTS_SETTLE_NS)boots_clear();
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL){stop=1;stop_reason=(int64_t)e->ident;}
      /* design/transport.md section 14, "A process a port": a port's child's message or exit */
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_CHILD)){
        struct port *port=(e->udata&0xff)<port_count?ports+(e->udata&0xff):NULL;
        if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->child)child_reaped(port);
        else if(port && e->filter==EVFILT_READ && (int)e->ident==port->child_fd)child_heard(port);
      }
      else if(e->filter==EVFILT_PROC){
        for(uint32_t s=0;s<MESH_CLIENTS;s++){
          uint64_t owner=atomic_load(&((struct mesh_client *)((char *)M+M->client_off))[s].owner);
          if(owner && (uint32_t)owner==(uint32_t)e->ident)mesh_retire(M,owner);
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
  /* design/transport.md: FAREWELL on every control socket (queued as the socket takes it; the kernel sends
     what it holds after the close), every program's words ~0 and its status the stop, stored from this thread, then the
     bridge gone; no queue, thread or verbs call waited on, and every object the kernel's to release at the exit
     (design/RDMA-KERNEL-RECOVERY.md "October 18"), the exit no sooner than every port's quiet window ends
     (design/transport.md B9: its reclaim never inside the termination a conclusion may follow) and a peer
     interface's termination's (`exit_at`, B9b), the kernel's link events read meanwhile, so a link going off then
     extends its port's (network_heard), as a termination then extends them all (peer_terminated), each no later than
     MESH_PLATFORM_NS past the stop's start (`held_to`: every window set before the stop ends by then, so the hold is
     the one 5 s budget, well inside its heartbeat's bound of 3 MESH_PLATFORM_NS) */
  for(uint32_t p=0;p<port_count;p++){
    child_send(ports+p,(struct child_message){.kind=CHILD_HALT});
    port_say(ports+p,(struct message){.kind=FAREWELL,.code=stop_reason},NULL,0);
    if(ports[p].control>=0){close(ports[p].control);ports[p].control=-1;}
  }
  streams_cancel(NULL);
  /* design/transport.md section 2: every stop keeps the region for the next bridge to adopt; a release is the
     heartbeat's, which expires the region once this bridge has ended (mesh-heartbeat.c region_expire) */
  atomic_store(&M->bridge_pid,0);
  mesh_notify(&M->control);
  const uint64_t held_to=clock_ns()+MESH_PLATFORM_NS;
  for(int told=0;;){
    now=clock_ns();
    const uint64_t windows=windows_end(),latest=MIN(windows,held_to);
    if(latest<=now)break;
    if(!told++)say("bridge node %u: its exit held %.1f s, to its quiet windows' end (design/transport.md)\n",node,
                   (double)(latest-now)/1e9);
    struct pollfd heard={.fd=network,.events=POLLIN};
    if(poll(&heard,1,(int)((latest-now+999999)/1000000))>0){now=clock_ns();network_heard(1);}
  }
  /* design/transport.md section 14, "A process a port": each child asked to end, and its exit awaited, at most
     MESH_PLATFORM_NS (its own windows its own: child_leave), so no child of this launch holds its device's queue pairs
     as the next launch's children make theirs */
  for(uint32_t p=0;p<port_count;p++)child_send(ports+p,(struct child_message){.kind=CHILD_END});
  for(const uint64_t until=clock_ns()+MESH_PLATFORM_NS;;){
    uint32_t left=0;
    for(uint32_t p=0;p<port_count;p++)if(ports[p].child){
      if(!waitpid(ports[p].child,NULL,WNOHANG))left++;
      else ports[p].child=0;
    }
    if(!left || clock_ns()>=until)break;
    poll(NULL,0,50);
  }
  return 0;
}
