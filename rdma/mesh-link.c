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
int events=-1,listener=-1,hello=-1,network=-1,addresses=-1,unreleased,finding;
uint64_t now,sequence;
struct mesh_ctx bridge={.fd=-1};
static dispatch_queue_t finder;
/* design/transport.md: the devices to be read again (an interface that no port is attached or its link came on), read
   on the event loop's turn once no window stands and no call is stuck (ports_find) */
static int find_wanted;
/* design/transport.md B12: a port's job out past MESH_PLATFORM_NS, during which the devices are not read
   (ports_stalled's, each event-loop turn) */
static _Atomic int stalled;
/* the ports a provider's doorbell fault was absorbed on (fault_heard), a bit a port */
static _Atomic uint32_t faulted;
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
/* design/transport.md: a port halted from any thread (a peer interface's termination heard, a provider's fault
   absorbed; atomics alone): its pairing lost, so no job of it makes a further call (B10), then no lane of its link
   served, both sequentially consistent (link_turn's connect stores served and reads the pairing again), and its bit set
   in `which` for the event loop's conclusion (terminations_turn, ports_stalled) */
_Static_assert(MESH_PORTS<=32,"a bit a port");
static void port_halt(struct port *port,_Atomic uint32_t *which){
  atomic_fetch_add(&port->link.pairing,1);
  atomic_store(&port->link.served,0);
  atomic_fetch_or(which,UINT32_C(1)<<port->index);
}

/* design/transport.md: a port's link page, the spare page, which receives what no holding's row takes (spare_span),
   leased for every port the region has, at the bridge's start beside the map's (link_reserve), else as its link first
   pairs (link_turn): its arena pages, and the arena offset of the first */
static uint32_t link_pages(void){ return M->block; }
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
/* the bytes of a page: every SEND and receive of a stream's lane */
static inline uint64_t page_bytes(void){ return (uint64_t)M->block*M->pgsz; }
/* design/pages-and-functions.md: the arena offset of row r's page (the row's own word: the page table read in the
   order of an entry's rows, its block table) */
/* 0 where the row holds no page of the arena (a row with no page bound reads address 0), so no SGE ever names memory
   outside the registered arena: a bound on memory */
static inline int row_at(struct mesh_row r,uint64_t *at){
  const uint64_t a=atomic_load_explicit(&mesh_page(M)[r.r].address,memory_order_relaxed),arena=(uint64_t)mesh_arena_pages(M)*M->pgsz;
  if(a<M->data_off || a-M->data_off>arena-page_bytes())return 0;
  *at=a-M->data_off;
  return 1;
}
/* the verbs boundary, the one place a page is bytes: the SGE of row r's page (0 where the row holds none: row_at); of
   the spare page, the link block; and of `bytes` of the all-ones block */
static inline int page_span(const struct port *port,struct mesh_row r,struct ibv_sge *span){
  uint64_t at;
  if(!row_at(r,&at))return 0;
  *span=wire_span(&port->device,page_bytes(),at,page_bytes());
  return 1;
}
static inline struct ibv_sge spare_span(const struct port *port){
  return wire_span(&port->device,page_bytes(),link_block(port),page_bytes());
}
static inline struct ibv_sge ones_span(const struct port *port,uint64_t bytes){
  return wire_span(&port->device,page_bytes(),(uint64_t)ones*M->pgsz,bytes);
}
/* the word at region offset `offset`, NULL where the region has none there (0, past its end, or off a word) */
static inline const _Atomic uint64_t *region_word(uint64_t offset){
  return offset && offset<=M->length-8 && !(offset%8)?stream_words(offset):NULL;
}
/* design/transport.md section 14: the entries a holder has presented on one of its lists taken into the walk, as far
   as its ring has room: each copied once and typed (struct taken), whether it is the holder's to name (`whole`) read,
   its first page over the holding counted, and the list's `taken` stored past them */
static void walk_take(struct walk *w,struct mesh_stream *x,int land){
  const uint64_t n=atomic_load_explicit(land?&x->lands:&x->sends,memory_order_acquire);
  const struct mesh_entry *list=(const struct mesh_entry *)((char *)M+x->lists)+(uint64_t)land*MESH_STREAM_ENTRIES;
  uint64_t t=w->taken;
  for(;t<n && t-w->done<MESH_STREAM_ENTRIES;t++){
    struct taken *k=w->ring+t%MESH_STREAM_ENTRIES;
    const struct mesh_entry e=list[t%MESH_STREAM_ENTRIES];
    const _Atomic uint64_t *word=e.word?region_word(e.word):land?NULL:&x->ready;
    k->row=mesh_entry_row(&e);k->pages=mesh_entry_pages(&e);k->base=mesh_entry_base(&e);
    k->whole=mesh_rows_hold(M,k->row,k->pages) && (word || !e.word);
    k->word=k->whole?word:NULL;
    k->page=w->pages;
    w->pages=mesh_pages_add(w->pages,k->pages);
  }
  if(t!=w->taken){w->taken=t;atomic_store_explicit(&x->taken[land],t,memory_order_release);}
}
/* the walk's entry holding page q of the holding, from the walk's place, which it moves past every entry done and
   every entry whose pages are all before q; NULL where no entry taken holds it */
static struct taken *walk_page(struct walk *w,struct mesh_pages q){
  if(w->at<w->done)w->at=w->done;
  for(;w->at<w->taken;w->at++){
    struct taken *t=w->ring+w->at%MESH_STREAM_ENTRIES;
    if(mesh_pages_lt(q,mesh_pages_add(t->page,t->pages)))return t;
  }
  return NULL;
}
/* design/transport.md section 14: a walk's entries whose pages have all completed passed (`through`: the stream's
   pages completed in order, SENDs or landings, lanes_through), and its count (`sent` or `landed`, in pages of the
   holding) stored where it moved */
static void walk_count(struct walk *w,struct mesh_pages through,_Atomic uint64_t *count){
  through=mesh_pages_sub(through,w->base);
  while(w->done<w->taken){
    const struct taken *e=w->ring+w->done%MESH_STREAM_ENTRIES;
    if(mesh_pages_lt(through,mesh_pages_add(e->page,e->pages)))break;
    w->done++;
  }
  if(through.n!=w->count.n){w->count=through;mesh_pages_store(count,through);}
}
/* design/transport.md section 14: a holding's words made new (its lists' counts, `ready`, `sent`, `landed`), then its
   owner word `owner`, by the port's progress thread once it posts nothing more of the holding: 0, the stream free,
   where nothing of the holding is outstanding; MESH_STREAM_GIVEN alone, naming no client, so none takes it, where
   units are (stream_end) */
static void stream_free(struct mesh_stream *x,uint64_t owner){
  _Atomic uint64_t *words[]={&x->sends,&x->lands,&x->taken[0],&x->taken[1],&x->ready,&x->sent,&x->landed};
  for(size_t i=0;i<sizeof words/sizeof *words;i++)atomic_store_explicit(words[i],0,memory_order_relaxed);
  atomic_store_explicit(&x->owner,owner,memory_order_release);
}
/* design/transport.md section 14: whether a holding is still its holder's, `owner`: the stream's owner word unchanged
   (not given) and the client's slot still its own (not retired) */
static int holding_kept(const struct mesh_stream *x,uint64_t owner){
  return atomic_load_explicit(&x->owner,memory_order_seq_cst)==owner && owned(owner);
}
/* design/transport.md section 14: unit u of stream x's lane c one way (`way`: 0 a SEND, 1 a receive) told before it is
   posted: the row it names (MESH_ABSENT none) in its lane's ring, then the lane's units posted past it (sequentially
   consistent, so a reader of the count reads the row), then whether its holding is still `owner`'s (holding_kept, its
   reads sequentially consistent).  A client that gives its stream or retires and then reads the lanes' units
   (mesh-dataflow.c mesh_pinned) so sees the unit, or the unit is not posted: where the holding is no longer kept the
   count goes back and 0 is returned.  1: post it */
static int unit_told(struct mesh_stream *x,int way,uint32_t c,uint64_t u,uint32_t row,uint64_t owner){
  atomic_store_explicit(mesh_unit_row(M,x->rows,way,(int)c,u),row,memory_order_relaxed);
  atomic_store_explicit(&x->posts[way].posted[c],u+1,memory_order_seq_cst);
  if(holding_kept(x,owner))return 1;
  atomic_store_explicit(&x->posts[way].posted[c],u,memory_order_release);
  return 0;
}
/* design/transport.md section 14: stream t's holding ended a pass ago (its owner word given, or its client retired), so
   every completion its lanes had before the end was taken (port_pass).  Where nothing its lanes posted is outstanding
   (each SEND posted completed, each receive posted landed), its words made new and the stream free (stream_free).
   Else, the plan's window in flight (no queue pair moved to another state, nothing awaited): its words made new and its
   owner word MESH_STREAM_GIVEN alone (stream_free), the stream not live for the rest of its pairing, so no
   communicator takes it; each outstanding unit's row stays held until the unit completes or the port's child exits
   (mesh.h struct mesh_posts).  A link none of whose streams is then live is lost by a failed lane's path (`broken`: the
   child tells its parent, which loses the link; the child's exit releases its queue pairs and with them the window's
   rows, and the port's next child pairs from 0), booked as what it is (`lost`, LINK_STREAMS_ENDED, not a lane's EIO) */
static void stream_end(struct port *port,uint32_t t){
  struct link *l=&port->link;
  struct mesh_stream *x=stream_of(port,t);
  const struct lane_state *s=l->lane+MESH_LANE_FIRST+2*t;
  const uint64_t sends=s[0].sent-s[0].done+s[1].sent-s[1].done,receives=s[0].posted-s[0].got+s[1].posted-s[1].got;
  l->held[t]=0;
  l->hold[t]=sends||receives?HOLD_GONE:HOLD_NONE;
  if(!sends && !receives){stream_free(x,0);return;}
  atomic_store_explicit(&x->live,0,memory_order_release);
  stream_free(x,MESH_STREAM_GIVEN);
  say("link %u (%s): stream %u toward node %u: its holding ended with %llu SENDs and %llu receives outstanding: the stream "
      "not live\n",port->index,port->name,t,port->peer,(unsigned long long)sends,(unsigned long long)receives);
  uint32_t live=0;
  for(uint32_t s=0;s<plan_streams(l->plan);s++)live|=atomic_load_explicit(&stream_of(port,s)->live,memory_order_acquire);
  if(live)return;
  say("link %u (%s): no stream toward node %u live: the link lost\n",port->index,port->name,port->peer);
  atomic_store_explicit(&l->lost,LINK_STREAMS_ENDED,memory_order_relaxed);
  atomic_fetch_or_explicit(&l->broken,l->plan>>MESH_LANE_FIRST<<MESH_LANE_FIRST,memory_order_release);
  wake();
}
/* design/transport.md section 14: stream t's holding at its port's progress thread, each pass.  An owner word holding
   a client's while none is served begins a holding, its walks' bases its lanes' SENDs and receives posted so far (no
   unit of an earlier holding is outstanding: each ended with none, or its stream serves no holding again in the
   pairing, stream_end); its entries then taken as presented.  The owner word given (MESH_STREAM_GIVEN: by its holder,
   or by the event loop for a holder gone, streams_let_go), or its client retired, ends it at once: nothing more of it
   is posted (unit_told), its entries taken and not posted are dropped, and on the next pass, its completions taken,
   the stream is freed or ended (stream_end).  A holding given before it began is freed at once.  The bridge posts
   nothing of its own on a stream: no unit between holdings and none for a holding that ended (the operator,
   2026-10-22: "the bridge has as little code as possible and does as little as possible") */
static void stream_turn(struct port *port,uint32_t t){
  struct link *l=&port->link;
  struct mesh_stream *x=stream_of(port,t);
  const struct lane_state *s=l->lane+MESH_LANE_FIRST+2*t;
  if(l->hold[t]==HOLD_GONE)return;
  if(l->hold[t]==HOLD_END){stream_end(port,t);return;}
  const uint64_t owner=atomic_load_explicit(&x->owner,memory_order_acquire);
  if(l->hold[t]==HOLD_NONE){
    if(!owner)return;
    if(owner&MESH_STREAM_GIVEN){stream_free(x,0);return;}
    l->held[t]=owner;l->hold[t]=HOLD_SERVE;
    l->sending[t]=(struct walk){.ring=l->sending[t].ring,.base={s[0].sent+s[1].sent}};
    l->landing[t]=(struct walk){.ring=l->landing[t].ring,.base={s[0].posted+s[1].posted}};
  }
  if(owner==l->held[t] && owned(owner)){walk_take(l->sending+t,x,0);walk_take(l->landing+t,x,1);}
  else l->hold[t]=HOLD_END;
}
/* design/transport.md section 14: stream t's SENDs, one pass of its port's progress thread, while its holding is served and
   both its lanes are: each page of the holding in order (the stream's page p on its lane p mod MESH_STREAM_LANES), the
   whole page,
   from the page its row names once its entry's word says the page is ready, and an entry not the holder's to name
   (rows outside the page table, a holder's error) at once as duds from the all-ones block; each while its lane's gate
   holds its frames (ibv_post_send accepts past the queue's frames and corrupts later, so the gate is the caller's);
   each unit told before it is posted and posted only while its holding is its holder's (unit_told); the completions
   are the link's (link_send); a failed post is the lane's (lane_lost: its child's exit releases its row,
   mesh-dataflow.c mesh_pinned) */
static void stream_send(struct port *port,uint32_t t,uint32_t served){
  struct link *l=&port->link;
  const uint32_t k=MESH_LANE_FIRST+2*t;
  if(l->hold[t]!=HOLD_SERVE || (served>>k&3)!=3)return;
  struct mesh_stream *x=stream_of(port,t);
  struct walk *w=l->sending+t;
  const struct frames each=frames_of(page_bytes());
  for(;;){
    const uint32_t j=MESH_STREAM_LANES>1 && lane_page(1,l->lane[k+1].sent)<lane_page(0,l->lane[k].sent),lane=k+j;
    const struct mesh_pages q=mesh_pages_sub((struct mesh_pages){lane_page(j,l->lane[lane].sent)},w->base);
    const struct taken *e=walk_page(w,q);
    if(!e)break;
    const struct mesh_pages i=mesh_pages_sub(q,e->page);
    struct ibv_sge span=ones_span(port,page_bytes()),page;
    uint32_t row=MESH_ABSENT;
    if(e->whole){
      if(!mesh_pages_lt(mesh_pages_add(e->base,i),mesh_pages_load(e->word)))break;
      const struct mesh_row r=mesh_row_after(e->row,i);
      if(page_span(port,r,&page)){span=page;row=(uint32_t)r.r;}
    }
    struct send_gate *gate=l->gate+lane;
    if(!gate_holds(gate,each) || !unit_told(x,0,j,l->lane[lane].sent,row,l->held[t]))break;
    struct ibv_send_wr posting={.wr_id=gate_post(gate,lane,each),.sg_list=&span,.num_sge=1,
                                .opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED},*bad;
    const struct mesh_queue *queue=l->provider.queues+lane;
    const int error=queue->send(queue->pair,&posting,&bad);
    if(error){lane_lost(port,lane,error<0?-error:error);break;}
    __atomic_store_n(&l->lane[lane].sent,l->lane[lane].sent+1,__ATOMIC_RELEASE);gate->requests++;
  }
}
/* design/pages-and-functions.md "The NFE" step 2: stream t's receives, one pass of its port's progress thread, while
   its holding is served and both its lanes are: each page of the holding in order (the stream's page p on its lane p mod
   MESH_STREAM_LANES), each lane at most its window ahead of its landings, into the page its row names once its entry's word
   allows (the row released by its reader), and an entry not the holder's to name (rows outside the page table, a
   holder's error) into the spare page; each unit told before it is posted and posted only while its holding is its
   holder's (unit_told); a failed post is the lane's (lane_lost: its child's exit releases its row, mesh-dataflow.c
   mesh_pinned) */
static void stream_post(struct port *port,uint32_t t,uint32_t served){
  struct link *l=&port->link;
  const uint32_t k=MESH_LANE_FIRST+2*t;
  if(l->hold[t]!=HOLD_SERVE || (served>>k&3)!=3)return;
  struct mesh_stream *x=stream_of(port,t);
  struct walk *w=l->landing+t;
  for(;;){
    const uint32_t j=MESH_STREAM_LANES>1 && lane_page(1,l->lane[k+1].posted)<lane_page(0,l->lane[k].posted),lane=k+j;
    struct lane_state *s=l->lane+lane;
    if(s->posted>=s->got+s->window)break;
    const struct mesh_pages q=mesh_pages_sub((struct mesh_pages){lane_page(j,s->posted)},w->base);
    const struct taken *e=walk_page(w,q);
    if(!e)break;
    if(e->word && mesh_pages_lt(mesh_pages_load(e->word),e->base))break;
    struct ibv_sge span=spare_span(port),page;
    uint32_t row=MESH_ABSENT;
    if(e->whole){
      const struct mesh_row r=mesh_row_after(e->row,mesh_pages_sub(q,e->page));
      if(page_span(port,r,&page)){span=page;row=(uint32_t)r.r;}
    }
    if(!unit_told(x,1,j,s->posted,row,l->held[t]))break;
    struct ibv_recv_wr request={.wr_id=lane,.sg_list=&span,.num_sge=1},*bad;
    const struct mesh_queue *queue=l->provider.queues+lane;
    const int error=queue->receive(queue->pair,&request,&bad);
    if(error){lane_lost(port,lane,error<0?-error:error);break;}
    s->posted++;
  }
}
/* design/transport.md: a lane's next unit landed (receives complete in the order posted) */
static void lane_landed(struct port *port,uint32_t k){ port->link.lane[k].got++; }
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
/* design/transport.md: the link's SENDs, one pass of the port's progress thread, the lanes `served` the pass's: its SEND
   completions taken (a lane no longer served drops its own; a failed one fails the link, link_failed): a lane's frames
   retired through its gate
   (each request names its lane and the frames posted through it) and its SENDs completed counted; then each stream
   whose lanes completed: each lane's units completed told (`posts`), and while its holding is served, its `sent`
   stored where its completions moved it (walk_count) */
static void link_send(struct port *port,uint32_t served){
  struct link *l=&port->link;
  struct ibv_wc done[16];
  const int count=l->provider.poll(l->provider.sent,16,done);
  if(count<0){link_failed(port,EIO,2);return;}
  uint32_t completed=0;
  for(int i=0;i<count;i++){
    const uint32_t lane=link_lane(l,done[i].qp_num);
    if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
    if(done[i].status){link_failed(port,(int64_t)done[i].status,2);return;}
    gate_retire(l->gate+lane,done[i].wr_id);l->lane[lane].done++;completed|=UINT32_C(1)<<lane;
  }
  if(!completed)return;
  for(uint32_t t=0;t<plan_streams(l->plan);t++){
    const uint32_t k=MESH_LANE_FIRST+2*t;
    if(!(completed&(UINT32_C(3)<<k)))continue;
    struct mesh_stream *x=stream_of(port,t);
    const struct lane_state *s=l->lane+k;
    for(uint32_t c=0;c<2;c++)if(completed>>(k+c)&1)atomic_store_explicit(&x->posts[0].done[c],s[c].done,memory_order_release);
    if(l->hold[t]!=HOLD_SERVE)continue;
    walk_count(l->sending+t,(struct mesh_pages){lanes_through((const uint64_t[]){s[0].done,s[1].done})},&x->sent);
  }
}
/* the link's receive completions, one pass of the port's progress thread, the lanes `served` the pass's: a lane's unit
   landed (lane_landed), a failed one failing the link (link_failed); then each stream a unit of which landed: each
   lane's units landed told (`posts`), and while its holding is served, its `landed` stored where its landings moved it
   (the stamp of every row it covers) */
static void link_receive(struct port *port,uint32_t served){
  struct link *l=&port->link;
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
    if((status_opcode>>32)&IBV_WC_RECV){lane_landed(port,lane);landed|=UINT32_C(1)<<lane;}
  }
  if(!(landed>>MESH_LANE_FIRST))return;
  for(uint32_t t=0;t<plan_streams(l->plan);t++){
    const uint32_t k=MESH_LANE_FIRST+2*t;
    if(!(landed&(UINT32_C(3)<<k)))continue;
    struct mesh_stream *x=stream_of(port,t);
    const struct lane_state *s=l->lane+k;
    for(uint32_t c=0;c<2;c++)if(landed>>(k+c)&1)atomic_store_explicit(&x->posts[1].done[c],s[c].got,memory_order_release);
    if(l->hold[t]!=HOLD_SERVE)continue;
    walk_count(l->landing+t,(struct mesh_pages){lanes_through((const uint64_t[]){s[0].got,s[1].got})},&x->landed);
  }
}

/* the progress thread's tell (the epoch it has seen, the guard it holds at), its raiser woken (port_wait) */
static void port_tell(_Atomic uint64_t *word,uint64_t value){
  atomic_store_explicit(word,value,memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
/* design/transport.md section 14: one pass of a port's progress thread: the lanes the link serves (`served` read once a
   pass: none posts on a lane its link no longer serves), each stream's holding, its SENDs and its receives, then every
   lane's completions.  It tells the port's epoch it began at (a conclusion around a stuck call moves it:
   ports_stalled) */
static void port_pass(struct port *port,uint64_t *told){
  const uint64_t epoch=atomic_load(&port->epoch);
  struct link *l=&port->link;
  const uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  for(uint32_t t=0;served>>MESH_LANE_FIRST && t<plan_streams(l->plan);t++){
    stream_turn(port,t);stream_send(port,t,served);stream_post(port,t,served);
  }
  if(served){link_receive(port,served);link_send(port,served);}
  if(epoch!=*told){port_tell(&port->seen,epoch);*told=epoch;}
}
/* design/transport.md section 14: a port's progress thread, its SENDs' and its receives': it passes without pause while
   its link serves a lane (rubric M8: the queues read as fast as a hardware thread allows), and waits on the port's
   guard word while it serves none, looking again every MESH_PAUSE_NS (the stop's `closing`) */
static void *port_progress(void *argument){
  struct port *port=argument;
  pthread_setname_np("mesh.rdma");
  /* fault_heard: this thread its port's, and its handler on a stack of its own */
  stack_t alternate={.ss_sp=malloc(SIGSTKSZ),.ss_size=SIGSTKSZ};
  if(alternate.ss_sp)sigaltstack(&alternate,NULL);
  atomic_store_explicit(&port->thread[0],pthread_self(),memory_order_release);
  uint64_t told=atomic_load_explicit(&port->seen,memory_order_relaxed);
  while(!atomic_load_explicit(&port->closing,memory_order_acquire)){
    /* while the port's guard stands, or its link serves no lane, no pass (no post on any queue pair of its device) and
       no lane read: the thread waits on the guard word (design/transport.md B13: a guarded call in the driver costs no
       core), woken by its raising and lowering, by a settle and by the link's coming up (guard_wake), looking again
       every MESH_PAUSE_NS (the stop's `closing`), the guard told and the epoch at each waking, each tell waking its
       raiser (port_tell) */
    const uint64_t guard=atomic_load_explicit(&port->guard,memory_order_acquire);
    if(guard&1 || !atomic_load_explicit(&port->link.served,memory_order_acquire)){
      /* a fault absorbed on the port (fault_heard: the handler, which may make no call but mmap, munmap and sigaction,
         halted the port) wakes its child's loop, which tells its parent at once, not at its next tick */
      if(atomic_load_explicit(&faulted,memory_order_relaxed))wake();
      if(guard&1)port_tell(&port->guarded,guard);
      const uint64_t epoch=atomic_load_explicit(&port->epoch,memory_order_acquire);
      if(epoch!=told){port_tell(&port->seen,epoch);told=epoch;}
      /* the wait on the word that holds the thread: the guard while it stands, else the lanes served while they are
         none, so a change of that word between the read above and the wait ends the wait at once (a wait on the guard
         for a link's coming up slept MESH_PAUSE_NS where the lanes were served between the two) */
      if(guard&1)os_sync_wait_on_address_with_timeout(&port->guard,guard,sizeof port->guard,OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                                      OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
      else os_sync_wait_on_address_with_timeout(&port->link.served,0,sizeof port->link.served,OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                                OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
      continue;
    }
    port_pass(port,&told);
  }
  return NULL;
}
/* design/transport.md B13: the progress thread parked on the port's guard looks again */
static void guard_wake(struct port *port){
  os_sync_wake_by_address_all(&port->guard,sizeof port->guard,OS_SYNC_WAKE_BY_ADDRESS_NONE);
  os_sync_wake_by_address_all(&port->link.served,sizeof port->link.served,OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
/* design/transport.md: the port's progress thread started, once; 0, or pthread_create's error */
static int port_threads(struct port *port){
  if(port->threads)return 0;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
  atomic_store(&port->seen,atomic_load(&port->epoch));
  pthread_t created;
  const int error=pthread_create(&created,&attributes,port_progress,port);
  if(!error)port->threads=1;
  pthread_attr_destroy(&attributes);
  return error;
}
/* design/transport.md: the raiser of a settle or a guard parked, never spinning, until the port's progress thread,
   where started, has its word (`told`: the epoch it has seen, or the guard it holds at) at `value`, woken by the
   thread's tell (port_tell), its guard's waiters woken as it begins and at each waking, within its deadline,
   MESH_PLATFORM_NS from its start: a thread that has not told by then is in a call that has not returned, and the
   port's child, the waiting process, ends at once naming the wait (design/data-model.md section 6), its exit the
   kernel's release of its port's objects and its parent's loss of the port (child_reaped) */
static void port_wait(struct port *port,_Atomic uint64_t *told,uint64_t value){
  const struct mesh_deadline deadline=mesh_deadline("mesh-flow child: its port's progress thread holding a settle or the guard (port_wait)",0);
  for(uint64_t seen;port->threads && (seen=atomic_load_explicit(told,memory_order_acquire))<value;){
    const uint64_t left=mesh_deadline_left(&deadline);
    guard_wake(port);
    os_sync_wait_on_address_with_timeout(told,seen,sizeof *told,OS_SYNC_WAIT_ON_ADDRESS_NONE,OS_CLOCK_MACH_ABSOLUTE_TIME,left);
  }
}
/* design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1": the guard of a port's device, on the port's queue,
   around a change of its verbs objects (queue pairs and completion queues made, connected or destroyed): raised, it
   returns once each started progress thread holds (it told the guard's value before its next pass, and passes no more
   while the guard stands), so no queue pair of the device is posted on while another is made, changed or destroyed
   (Apple's provider: a post on a long-lived queue pair across another's teardown stores to the device's shared doorbell
   mapping after it moved, tbt_post_recv); lowered, the threads parked on it woken and going on
   (design/transport.md B13).  It stands for every change once the port's threads run, the link lanes'
   long-lived queue pairs being posted on while the link is up.  Raised: 1; 0 where no progress thread runs (a thread
   that does not hold within its deadline ends the child: port_wait) */
static void guard_lower(struct port *port,int raised){
  if(raised>0){atomic_fetch_add_explicit(&port->guard,1,memory_order_release);guard_wake(port);}
}
static int guard_raise(struct port *port){
  if(!port->threads)return 0;
  port_wait(port,&port->guarded,atomic_fetch_add_explicit(&port->guard,1,memory_order_acq_rel)+1);
  return 1;
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
/* each client slot's owner as clients_turn last read it */
uint64_t owners[MESH_CLIENTS];
/* design/transport.md: this node's record, its region, memory and model with its ports and health */
char model[MESH_MODEL_BYTES];
/* design/transport.md section 14: this node's reach as its record tells it (mesh-records.c record_changed): an entry
   a cable whose link is up with a stream in its plan, its peer node and bandwidth, at most `room`; their count.  The
   cables, not the streams live this instant, so two nodes several cables join are as many parallel links of the map
   (mesh-dataflow.c mesh_join), which a cable's link coming up or lost moves (link_turn, link_set) */
uint32_t record_reach(struct mesh_reach *reach,uint32_t room){
  uint32_t count=0;
  for(uint32_t p=0;p<port_count && count<room;p++)
    if(ports[p].link.state==LINK_UP && plan_streams(ports[p].link.plan))
      reach[count++]=(struct mesh_reach){.node=ports[p].peer,.bandwidth=ports[p].bandwidth};
  return count;
}
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
   completion queues the plan's streams' lanes alone dividing them, lanes 0 and 1 posting nothing:
   design/HETEROGENEOUS-LINKS.md; EALREADY for a make asked
   again); or connected to the peer's (LINK_JOB_CONNECT, the plan's lanes alone: RTR, then RTS, each lane's number and
   gate set, every lane's progress made new, and the port's progress thread started).  No queue pair is moved to
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
  l->share=(struct frames){(uint64_t)MIN(p->completion->cqe,p->sent->cqe)/MAX(plan_lanes(l->plan>>MESH_LANE_FIRST),1u)};
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
  const struct frames share={MAX(l->share.n,1)},ring={(uint64_t)MESH_LANE_UNITS*frames_of(page_bytes()).n};
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++){
    l->qpn[k]=plan>>k&1?p->queues[k].pair->qp_num:UINT32_MAX;
    l->gate[k]=(struct send_gate){.capacity=plan>>k&1?frames_min(frames_min((struct frames){p->queues[k].send_capacity},share),ring):(struct frames){0}};
  }
  if(!error)error=link_lapsed(l,numbers);
  for(uint32_t k=MESH_LANE_FIRST;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)lane_reset(port,k);
  for(uint32_t s=0;s<plan_streams(plan) && !error;s++)l->wire[s]=stream_wire(port,numbers,s);
  return error;
}
/* design/transport.md section 14, "A process a port": the link's streams opened as its lanes connect, by the child
   before any lane is served, every one new: a holding of a lost pairing ended with it (its owner word 0: its holder's
   give, a compare-and-swap from its own client, then matches nothing, and its communicator reads its holding gone,
   mesh-call.h mesh_stream_held), its words 0 (its lists' counts, `ready`, `sent`, `landed`) and its lanes' units
   posted and completed each way 0 (`posts`: its lanes' units are 0 as their queue pairs are made, lane_reset), this
   child its poster (`child`, `child_start`: its exit ends every unit it posted, mesh-dataflow.c mesh_pinned), its peer
   node, its share of the link's bandwidth and its identity told, then live.  None is
   kept for a holder of a lost pairing: no holder can, by letting go or by never letting go, keep a stream of the link
   from its next pairing (design/rubrics.md M4).  Nothing of the lost pairing is sent on it: that pairing's groups
   ended at their next look at a landed word ~0 (streams_cancel), this connect comes MESH_PLATFORM_NS and a child's exit
   after the loss, and its communicators start no group after it.  The streams opened are those the link's plan has
   lanes for (design/HETEROGENEOUS-LINKS.md): the rest stay as the loss, the adoption or the region's making left
   them, not live, so no client takes one.  Each stream's share of the bandwidth is the link's over MESH_LINK_STREAMS
   however many are live, so the map's figure for a link (mesh-plan.h mesh_map, the lesser of its ends') is alike for
   links of every plan */
static void streams_open(struct port *port,uint32_t node,uint64_t bandwidth){
  struct link *l=&port->link;
  for(uint32_t s=0;s<plan_streams(l->plan);s++){
    struct mesh_stream *x=stream_of(port,s);
    atomic_store_explicit(&x->owner,0,memory_order_release);
    _Atomic uint64_t *words[]={&x->sends,&x->lands,&x->ready,&x->sent,&x->landed,&x->taken[0],&x->taken[1],
                               &x->posts[0].posted[0],&x->posts[0].posted[1],&x->posts[0].done[0],&x->posts[0].done[1],
                               &x->posts[1].posted[0],&x->posts[1].posted[1],&x->posts[1].done[0],&x->posts[1].done[1]};
    for(size_t i=0;i<sizeof words/sizeof *words;i++)atomic_store_explicit(words[i],0,memory_order_relaxed);
    l->held[s]=0;l->hold[s]=HOLD_NONE;
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
  struct link_numbers numbers={.pairing=atomic_load_explicit(&l->pairing,memory_order_relaxed),.lid=port->heard.lid};
  memcpy(numbers.peer,l->peer,sizeof numbers.peer);memcpy(numbers.gid,port->heard.gid,sizeof numbers.gid);
  l->job=job;l->job_pairing=numbers.pairing;l->job_at=now;atomic_store_explicit(&l->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    atomic_store_explicit(&port->thread[1],pthread_self(),memory_order_release);
    const int raised=guard_raise(port);
    int result=link_lapsed(l,&numbers);
    if(!result)result=job==LINK_JOB_MAKE?link_make(port,&numbers):link_connect(port,&numbers);
    if(!result && job==LINK_JOB_CONNECT)result=port_threads(port);
    guard_lower(port,raised);
    l->result=result;
    atomic_store_explicit(&port->thread[1],NULL,memory_order_release);
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
  atomic_store_explicit(&l->lost,0,memory_order_relaxed);
}
/* design/transport.md section 14: a stream's lane k's progress made new at its link's connect, its sequences 0 and
   its receive window the receives the fewer frames of its share of the completion queue and its receive queue hold, a
   receive a page's frames, at most a ring of rows' (MESH_LANE_UNITS: mesh.h struct mesh_posts); on the port's queue in
   the connect's job, before any progress thread serves it.  Its stream's words are the stream's opening's
   (streams_open) */
static void lane_reset(struct port *port,uint32_t k){
  struct link *l=&port->link;
  const struct frames unit=frames_of(page_bytes());
  const struct frames room=frames_min(l->share,(struct frames){l->provider.queues[k].receive_capacity});
  l->lane[k]=(struct lane_state){.window=(uint32_t)MIN(MESH_LANE_UNITS,MAX(1,room.n/unit.n))};
  atomic_thread_fence(memory_order_release);
}
/* design/transport.md section 14: the arena pages of a port's streams' lists and rings: each stream's two lists of
   MESH_STREAM_ENTRIES entries and its rings of rows,
   MESH_LANE_UNITS rows of 4 bytes a lane each way (mesh.h mesh_unit_row), each part's bytes rounded up to whole pages
   (page_round) */
static uint64_t page_round(uint64_t bytes){ return (bytes+page_bytes()-1)/page_bytes()*page_bytes(); }
static uint64_t rows_bytes(void){ return page_round(4*(uint64_t)MESH_LANE_UNITS*sizeof(uint32_t)); }
static uint32_t lists_pages(void){
  const uint64_t each=page_round(2*(uint64_t)MESH_STREAM_ENTRIES*sizeof(struct mesh_entry))+rows_bytes();
  return (uint32_t)(MESH_LINK_STREAMS*each/M->pgsz);
}
/* design/transport.md section 14: the port's streams' lists and rings, leased from the arena by the parent as the
   port first pairs (recorded in the region, so a bridge adopting the region takes the same pages: section 2) and held
   for the region's life, every child and bridge of the port's, told in the region (mesh.h struct mesh_stream `lists`,
   `rows`).  0, or ENOMEM where the region holds no room for them */
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
    x->lists=at;at+=page_round(2*(uint64_t)MESH_STREAM_ENTRIES*sizeof(struct mesh_entry));
    x->rows=at;at+=rows_bytes();
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
/* design/transport.md section 14: the streams of a port (`port` NULL: of every port) live no more, and their `sent`
   and `landed` ~0, as its link is lost or the bridge stops: a client waiting on one reads the loss */
void streams_cancel(const struct port *port){
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
/* design/transport.md section 14: a stream whose owner word names a client gone (its process exited, its slot retired)
   let go as its holder's give does (MESH_STREAM_GIVEN set), as that client's rows are released; its port's progress
   thread then ends the holding at once and frees the stream (stream_turn) */
void streams_let_go(void){
  for(uint32_t p=0;p<port_count;p++)
    for(uint32_t s=0;s<MESH_LINK_STREAMS;s++){
      struct mesh_stream *x=stream_of(ports+p,s);
      uint64_t owner=atomic_load_explicit(&x->owner,memory_order_acquire);
      if(owner && !(owner&MESH_STREAM_GIVEN) && !owned(owner))atomic_compare_exchange_strong(&x->owner,&owner,owner|MESH_STREAM_GIVEN);
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
      say("link %u (%s): %u lanes up toward node %u (plan %#x: %u stream%s), each lane's share %llu frames\n",
          port->index,port->name,plan_lanes(l->plan),port->peer,l->plan,plan_streams(l->plan),plan_streams(l->plan)==1?"":"s",
          (unsigned long long)l->share.n);
      record_changed();
    }
  }
  const int held=now<port->quiet_at;
  if(!port->greeted)return;
  if(l->state==LINK_UP){
    const uint32_t broken=atomic_exchange_explicit(&l->broken,0,memory_order_acq_rel);
    const int lost=atomic_exchange_explicit(&l->lost,0,memory_order_relaxed);
    if(broken && lost==LINK_STREAMS_ENDED)
      say("link %u (%s): no stream toward node %u live, each holding ended with units outstanding: the link lost\n",
          port->index,port->name,port->peer);
    else if(broken)say("link %u (%s): lanes %#x toward node %u failed: the link lost\n",port->index,port->name,broken,port->peer);
    if(broken){port_lost(port,lost?lost:EIO);return;}
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
   be ended.  While any port's job is so out the devices are not read (`stalled`: ports_find, the parent's own driver
   entry), and the port itself takes no job until the call returns (`stuck`, its job's word); each port's calls are its
   own child's, a process of its own (section 14), so one port's stuck call holds no other port's (a hold of every
   port's was 10-18's, when one process made every port's calls).  Each conclusion moves the port's epoch and wakes its threads; the port stays
   `stuck` until the call returns and each progress thread has told the conclusion's epoch (a thread tells it on its
   first pass after its call returns, port_pass, or as it wakes where it is in none), then opens again (`opening`,
   port_open past its quiet window, B9; an open's own return is port_opened's) and so takes jobs again.  Read on the
   event loop's turn */
static void ports_stalled(void){
  int any=0;
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    const struct link *l=&port->link;
    const int opening=port->job && !atomic_load_explicit(&port->done,memory_order_acquire) && now-port->job_at>=MESH_PLATFORM_NS;
    const int linking=l->job && !atomic_load_explicit(&l->done,memory_order_acquire) && now-l->job_at>=MESH_PLATFORM_NS;
    const int out=opening||linking;
    any|=out;
    if(port->stuck){
      const int returned=!out && (!port->threads || atomic_load_explicit(&port->seen,memory_order_acquire)>=atomic_load(&port->epoch));
      if(returned){port->stuck=0;if(!port->job)port->opening=1;}
      continue;
    }
    if(!out)continue;
    port->stuck=1;atomic_fetch_add(&port->epoch,1);guard_wake(port);
    say("link %u (%s): its %s out %.1f s: its port concluded around it (design/transport.md)\n",port->index,
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
    if(heard->block!=page_bytes()){port_lost(port,EPROTO);return;}
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
   and as the call its port was concluded around returns, ports_stalled), once its quiet window has ended and no job of
   its own is out (design/transport.md B9, B12): the hellos' group joined on its
   interface, so the port hears them open or not (hello_heard, B16); its port read (none where the guard was not held
   within MESH_PLATFORM_NS: ETIMEDOUT, port_wait); a context that fails stays as it is, and the port with it, until its
   child ends (design/transport.md section 14: nothing made is forgotten or made again in a process).  The device is
   its child's: spawned here where the port has none (child_spawn: not while its last child's end is awaited, `recycle`,
   nor before the hold a spawn that failed set, `hold_to`), and asked to open it (CHILD_OPEN), its answer the
   job's end (child_heard) */
static int child_spawn(struct port *port);
static void port_open(struct port *port){
  if(port->job || port->stuck || !port->opening || port->control>=0 || now<port->quiet_at || port->recycle || now<port->hold_to ||
     now<port->retry_at)return;
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
    int refused=0,result=device_up(&port->device,&wire,&refused);
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
  struct message said={.magic=XMAGIC,.version=MESH_VERSION,.kind=HELLO,.node=M->node,.boot=M->boot,.block=(uint32_t)page_bytes(),.port=port->index};
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
      struct message said={.magic=XMAGIC,.version=MESH_VERSION,.kind=HELLO,.node=M->node,.boot=M->boot,.block=(uint32_t)page_bytes(),.port=port->index};
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
/* design/transport.md section 10: the all-ones block every dud is sent from (stream_send), taken and filled at the
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
  memset(mesh_at(M,ones),0xff,(size_t)page_bytes());
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
    /* the interface as the kernel's events name it, its name read once here into typed values (addresses_heard) */
    if(!strncmp(port->name,"en",2)){port->family=APPLE_IF_FAM_ETHERNET;port->unit=(uint32_t)strtoul(port->name+2,NULL,10);}
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
/* design/transport.md: an IPv6 address of a port's interface added, changed or deleted (the kernel's KEV_INET6_SUBCLASS
   events on `addresses`: a link-local address bin/mesh-fabric-init.sh puts in place, or its duplicate address detection
   done): the hellos' group joined on the interface again (a join while it held no IPv6 address took nothing) and the
   port's hello said at this turn, so a hello that left before the address could carry it is said again as the address
   comes, not a pace (MESH_PAUSE_NS) later.  The port is the event's interface family and unit, the port's own read once
   (struct port `family`, `unit`): no name is built or compared per event (design/rubrics.md, Strings are messages).
   Another Ethernet-family interface of the same unit (bridge0, awdl0, anpi<N>) is taken as the port's, and its event
   says the port's hello early, as the pace would: a hello on an open port with no peer is never wrong */
static void addresses_heard(void){
  union { struct kern_event_msg header; unsigned char bytes[512]; } event;
  while(recv(addresses,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    for(uint32_t p=0;p<port_count;p++)if(interface->if_family==ports[p].family && interface->if_unit==ports[p].unit){
      struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=ports[p].interface};
      if(ports[p].interface)setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
      ports[p].next=now;
    }
  }
}

/* design/transport.md B9b (EXIT-RECLAIM.md P2, and P3 as a window in place of IOServiceWaitQuiet): a
   Thunderbolt RDMA peer interface's termination, which IOKit tells in its first phase (kIOTerminatedNotification), on
   a queue of its own: the terminations heard (`terminations`), and the ports whose peer interface is terminating
   (`terminated`, a bit a port index), which the event loop concludes (terminations_turn) */
static _Atomic uint32_t terminated,terminations;
/* design/transport.md: a Thunderbolt RDMA peer interface's arrival (kIOFirstMatchNotification: a cable's far end come
   to RDMA), on the terminations' queue, so a port's termination and a later arrival are taken in their order: the ports
   whose peer interface arrived (`arrived`, a bit a port index) and whether one arrived whose interface is no port
   (`arrived_elsewhere`), which the event loop takes (arrivals_turn) */
static _Atomic uint32_t arrived;
static _Atomic int arrived_elsewhere;
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
/* the arrivals drained (which arms the notification again), each service released: the port its interface is marked
   to be opened, one whose interface is no port of this process marked for the devices' read */
static void peer_arrived(void *unused,io_iterator_t iterator){
  (void)unused;
  for(io_service_t service;(service=IOIteratorNext(iterator));IOObjectRelease(service)){
    struct port *port=peer_port(service);
    if(port)atomic_fetch_or(&arrived,UINT32_C(1)<<port->index);
    else atomic_store(&arrived_elsewhere,1);
    wake();
  }
}
/* armed at the start, before IOKit's registry is waited on and the devices are read: the notification on a serial
   queue of its own, its iterator drained once (which arms it); without it a link's going off is the first sign.  The
   parent (`arrivals`) hears the peer interfaces' arrivals on the same port and queue, its iterator drained once too:
   without them a port whose open met its device's peer not yet present (ENETDOWN at both ends, no hello sent) opened
   only at its link's next coming on or the next hello heard on it, which might never come */
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
static _Atomic uintptr_t retried[MESH_PORTS];
static void fault_heard(int number,siginfo_t *info,void *context){
  const uintptr_t pc=__darwin_arm_thread_state64_get_pc(((ucontext_t *)context)->uc_mcontext->__ss);
  const uintptr_t page=trunc_page((uintptr_t)info->si_addr);
  const pthread_t self=pthread_self();
  struct port *port=NULL;
  for(uint32_t p=0,count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);p<count;p++)
    for(int t=0;t<2;t++)if(atomic_load_explicit(&ports[p].thread[t],memory_order_acquire)==self)port=ports+p;
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
/* the arrivals taken on the event loop, after the terminations (a peer interface gone and then come is lost, then
   opened): each port whose peer interface arrived and that has no peer is to be opened as at its link's coming on
   (network_heard), so its device is opened at the hardware's word that its far end is present and not only at the en
   interface's link, which can come on before it (bin/mesh-fabric-init.sh: link active, no RDMA peer); an arrival at an
   interface that is no port has the devices read again (find_wanted) */
static void arrivals_turn(void){
  const uint32_t came=atomic_exchange(&arrived,0);
  for(uint32_t p=0;p<port_count;p++)if(came>>p&1 && ports[p].control<0){ports[p].opening=1;ports[p].retry_at=0;}
  if(atomic_exchange(&arrived_elsewhere,0))find_wanted=1;
}

/* design/transport.md section 14, "A process a port": the parent's side of a port's child.  The bridge is a parent
   (the region, the map and records, the clients, discovery, hellos and the control sockets) and a child a port (its
   device, its link's queue pairs and completion queues, its progress thread): mesh-flow spawned again, the same image
   (`self_path`), told the contract's version it is spawned for (mesh-bridge.h MESH_CHILD_VERSION), in a process group of its own (a job's group signalled at its end reaches no verbs owner), with
   the parent's socket at descriptor 3, the parent's stderr, and nothing else.  The child's exit is the one release of
   its port's verbs objects (the kernel's), and a link once connected and then lost pairs again with the port's next
   child */
static char self_path[PATH_MAX];
/* a message whole on the socket between a port's child and its parent, within MESH_PLATFORM_NS of the call (the operator,
   2026-10-23: no wait runs longer than a deadline presented when it is made, 5 s by default): a peer gone drops it, and
   one that takes no more by the deadline ends this process at once, the wait named (`what`) */
static void message_send(int fd,const struct child_message *said,const char *what){
  const char *at=(const char *)said;
  size_t left=sizeof *said;
  const uint64_t until=clock_ns()+MESH_PLATFORM_NS;
  while(left){
    const ssize_t sent=send(fd,at,left,MSG_DONTWAIT);
    if(sent>0){at+=sent;left-=(size_t)sent;continue;}
    if(sent<0 && errno==EINTR)continue;
    if(sent==0 || errno!=EAGAIN)return;
    const uint64_t t=clock_ns();
    if(t>=until){say("%s: its peer took no message for %.0f s; this process exits\n",what,(double)MESH_PLATFORM_NS/1e9);exit(EX_SOFTWARE);}
    struct pollfd room={.fd=fd,.events=POLLOUT};
    poll(&room,1,(int)((until-t+999999)/1000000));
  }
}
/* a message to the port's child, whole (message_send); one not taken (the child gone) is dropped, its end reaped
   (child_reaped) */
static void child_send(struct port *port,struct child_message said){
  if(port->child_fd>=0)message_send(port->child_fd,&said,"bridge: child_send to a port's child");
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
   port's next spawn held MESH_PAUSE_NS.  A child of its device from an earlier launch (child_elsewhere) holds the spawn
   until its exit, which the event loop hears (EVFILT_PROC, tagged by the port: `elsewhere` cleared) and spawns at once;
   the processes are read again only then, not at every turn while it lives.  One whose exit cannot be watched is read
   again at the next turn (at once where it has already gone or is in its exit, which the scan does not see) */
static int child_spawn(struct port *port){
  int pair[2],on=1;
  if(port->elsewhere)return EBUSY;
  const pid_t elsewhere=child_elsewhere(port);
  if(elsewhere){
    struct kevent64_s watch;
    EV_SET64(&watch,(uint64_t)elsewhere,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,TAG_CHILD|port->index,0,0);
    if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){if(errno==ESRCH)wake();return EBUSY;}
    say("link %u (%s): process %d, a child of its device from an earlier launch, holds its queue pairs until its exit; its "
        "own child spawned after it\n",port->index,port->name,(int)elsewhere);
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
  char index[16],page[16];
  snprintf(index,sizeof index,"%u",port->index);snprintf(page,sizeof page,"%u",ones);
  char version[16];
  snprintf(version,sizeof version,"%u",MESH_CHILD_VERSION);
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
  port->child=pid;port->child_fd=pair[0];port->child_held=0;port->ending=0;
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
}
/* a port whose link was lost once its lanes were asked for (`recycle`) asks its child to end once its quiet window and
   the exit's have ended (B9, B9b: no release inside a termination) and no job of its own is out (B12) */
static void child_turn(struct port *port){
  if(!port->recycle || !port->child || port->ending || port->job || port->link.job || now<atomic_load(&port->quiet_at) ||
     now<atomic_load(&exit_at))return;
  child_send(port,(struct child_message){.kind=CHILD_END});
  port->ending=1;
  say("link %u (%s): its child %d asked to end, its lanes released at its exit (design/transport.md section 14)\n",
      port->index,port->name,(int)port->child);
}
/* the port's child ended: its last messages taken, its exit reaped, a job of it out ended (ECHILD); one that ended
   unbidden loses the link (port_lost, ECHILD) and is spawned again once the port's quiet window has passed, at the
   events that open any port, however often its children ended (the operator, 2026-10-23 01:03Z: "this is not a real feature.
   delete it and anything like it").  Its streams cancelled again,
   the region's closed leases looked at again (`unreleased`: the rows its units named are held no more, its queue
   pairs gone with it, mesh-dataflow.c mesh_pinned), its link down with no lanes, and the port to be opened by its next
   child */
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
  if(asked)say("link %u (%s): its child %d ended\n",port->index,port->name,(int)pid);
  else{
    say("link %u (%s): its child %d ended unbidden (%s %d)\n",port->index,port->name,(int)pid,
        WIFSIGNALED(status)?"signal":"status",WIFSIGNALED(status)?WTERMSIG(status):WEXITSTATUS(status));
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
   the consumer's document (rdma/bridge_configuration.py): its name and its region in bytes (`region_bytes`); NULL, or
   why it is refused */
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

/* design/transport.md section 14, "A process a port": a port's child, `mesh-flow --child VERSION DEVICE INDEX ONES` as
   its parent spawned it (child_spawn; a parent of an earlier build adds its keepalive pace, which is not read: no
   child sends a keepalive), VERSION the contract's (mesh-bridge.h MESH_CHILD_VERSION): the parent's region mapped (its magic, version and
   bridge the parent's), its port the parent's port INDEX on DEVICE (its streams' words the region's), the duds' block
   ONES the parent's, the provider's faults absorbed (fault_heard) and its peer interfaces'
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
static void child_tell(struct child_message said){ message_send(3,&said,"child: child_tell to its parent"); }
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
  int refused=l->job?EBUSY:child_halted || atomic_load(&faulted) || atomic_load(&terminated)?ECANCELED:0;
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
    said.psn=l->provider.psn;said.share=(uint32_t)l->share.n;
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
  if(broken)child_tell((struct child_message){.kind=CHILD_BROKEN,.pairing=child_asked,.broken=broken,
                                              .result=atomic_exchange_explicit(&port->link.lost,0,memory_order_relaxed)});
  if(atomic_exchange(&faulted,0)){child_halted=1;child_tell((struct child_message){.kind=CHILD_FAULTED});}
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
  *port=(struct port){.index=(uint32_t)index,.control=-1,.child_fd=-1,.work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL)};
  port->link.provider.device=&port->device;port->link.page=port->link.lists=MESH_ABSENT;
  snprintf(port->device.name,sizeof port->device.name,"%s",argv[3]);snprintf(port->name,sizeof port->name,"%s",argv[3]+5);
  ones=(uint32_t)strtoul(argv[5],NULL,10);
  __atomic_store_n(&port_count,1,__ATOMIC_SEQ_CST);
  struct kevent64_s event;
  if((events=kqueue())<0)die("child events");
  EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
  EV_SET64(&event,3,EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
  terminations_watch(0);
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
   stream not live and held by none, its landed words ~0 (a lost pairing's holding ended with its bridge, and a
   wait on it reads the loss; the next pairing opens each new, streams_open), the links' table cleared, a client slot
   whose retirement a process that died left half done finished (mesh_retire's walk, which any party can repeat to the
   same end), the closed rows released, a map or health serial a bridge that died mid-write left odd made even (its
   readers wait while it is odd), and the map's records this bridge's to write again (map_adopt) */
static void region_adopt(void){
  for(uint32_t s=0;s<MESH_STREAMS;s++){
    struct mesh_stream *x=mesh_stream_at(M,s);
    atomic_store(&x->live,0);atomic_store(&x->owner,0);atomic_store(&x->peer,0);atomic_store(&x->wire,0);
    mesh_pages_store(&x->sent,MESH_PAGES_LOST);mesh_pages_store(&x->landed,MESH_PAGES_LOST);
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

/* design/transport.md: when the event loop's next turn is due if no event comes first: the earliest of the times its turn
   waits for that are still ahead (the next liveness turn, `liveness`; the next health sample;
   where the devices are to be read, the windows' end; the exit's window; each port's quiet window, spawn hold, open's
   retry, next hello, and the bound of its open's or link's job out), and no later than MESH_PAUSE_NS on (design/rubrics.md M3: the wait ends in finite
   time).  Every other change a turn takes wakes the loop where it happens (wake, `control`, a socket, a signal and an
   exit on the queue), so a turn comes at the change or at the time it waits for, where a fixed tick (250 ms) took each
   such time up to a tick late */
static inline void due_at(uint64_t *due,uint64_t at){ if(at>now && at<*due)*due=at; }
static uint64_t turn_due(uint64_t liveness){
  uint64_t due=MIN(liveness,now+MESH_PAUSE_NS);
  due_at(&due,health_due());
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
  /* design/transport.md: the kernel's IPv6 address events (addresses_heard); a bridge that cannot hear them says a
     port's hello at its pace alone */
  struct kev_request inet6={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_INET6_SUBCLASS};
  if((addresses=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT))<0 || fcntl(addresses,F_SETFL,O_NONBLOCK) || ioctl(addresses,SIOCSKEVFILT,&inet6)){
    say("bridge: the kernel's IPv6 address events not heard (%s): a port's hello after its address comes waits its pace\n",strerror(errno));
    if(addresses>=0)close(addresses);
    addresses=-1;
  }
  int fd=-1;
  const int adopted=region_open(&geometry,&fd);
  if(adopted)region_adopt();
  /* a boot is the bridge's start on the wall clock, so a node's later boot is a newer record */
  M->node=node;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);M->idle_since=M->boot;
  say("bridge node %u: region %s %s\n",node,MESH_NAME,adopted?"adopted from the bridge before this one (its clients' leases stand)":"made new");
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  bridge.M=M;
  finder=dispatch_queue_create("mesh.ports",DISPATCH_QUEUE_SERIAL);
  /* design/transport.md: the node's memory and hardware model, which its record carries with its region */
  size_t size=sizeof memory;
  if(sysctlbyname("hw.memsize",&memory,&size,NULL,0))memory=0;
  sysctl_text("hw.model",model,sizeof model);
  health_init();
  record_changed();
  if(!atomic_load(&M->map_nodes))die("the map's lease");
  link_reserve();
  int watched[4]={listener,hello,network,addresses};
  for(int i=0;i<4 && watched[i]>=0;i++){EV_SET64(&event,(uint64_t)watched[i],EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");}
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
    arrivals_turn();
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
    int alive=now>=liveness;
    if(alive){
      liveness=now+MESH_PAUSE_NS;
      record_retell();
    }
    clients_turn(alive);
    struct kevent64_s ready[16];
    const uint64_t due=turn_due(liveness),at=clock_ns(),rest=due>at?due-at:0;
    struct timespec tick={(time_t)(rest/1000000000u),(long)(rest%1000000000u)};
    int count=kevent64(events,NULL,0,ready,16,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL){stop=1;stop_reason=(int64_t)e->ident;}
      /* design/transport.md section 14, "A process a port": a port's child's message or exit */
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_CHILD)){
        struct port *port=(e->udata&0xff)<port_count?ports+(e->udata&0xff):NULL;
        if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->child)child_reaped(port);
        else if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->elsewhere)port->elsewhere=0;
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
     as the next launch's children make theirs: each exit heard on a queue of the children's exits alone (EVFILT_PROC),
     where a poll every 50 ms took each up to 50 ms late (a queue not made waits out the bound) */
  for(uint32_t p=0;p<port_count;p++)child_send(ports+p,(struct child_message){.kind=CHILD_END});
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
