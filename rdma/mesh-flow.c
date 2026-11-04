/* The bridge's data path, in a port's child (mesh-child.c): one cable and its queue pairs (the tag's lanes, the link's
   `plan`), serving this node's share of the region's page lists (mesh.h struct mesh_hop) by the rule both ends compute
   (design/page-queue-change-request.md, "Claude's proposal": "The page lists (the queue)", "What the bridge decides",
   "The rule both ends compute", "The functions"; and the operator's "The page queue" and "Interleaving every
   transfer": every inbound link used, a ready page waiting only behind earlier pages of its own queue pair).
   Its native forms are the d6dceac transport's (MESH_VERSION 104, restored 2026-10-01: the source of the fastest
   recorded E2B decode, metal-microbench docs/measurement.md): SENDs prepared once a plan and chained per queue pair,
   each posted as its word releases it; receives prepared once a plan, posted ahead in a window and refilled as they
   land; completions polled through the provider's own functions (mesh-verbs.h struct mesh_verbs `poll`, struct
   mesh_queue `send`, `receive`); traced variants under MESH_LEDGER.  Their inputs are the page lists, in place of the
   per-link, per-queue transfer lists (mesh_transfers, mesh_order_length) and the --link form's link_configure.  The
   progress thread, the guard around its device's verbs calls and a port's halts are tag pre-return-20261103
   rdma/mesh-link.c's, verbatim but at interfaces.  No main (the binary's is mesh-link.c's, the child's mesh-child.c's)
   and no verbs teardown: a halt clears `served` and moves `pairing`, and the child's exit is the teardown (mesh-child.c
   child_leave, lanes_flush; down_pair stays deleted). */
#include "mesh-bridge.h"
#include "mesh-graph.h"

/* design/prepared-machine.md#M29: d6dceac's prepared SEND (its span and request, one cache line's worth of what the
   post reads), its release now its word of execution t and `position` (mesh.h struct mesh_hop) in place of the
   client's `ready` continuation (struct mesh_send): the word is the region's, the record the child's */
struct prepared_send {
  _Alignas(128) _Atomic uint64_t *word;
  uint64_t position;
  struct ibv_sge span;
  struct ibv_send_wr request;
};
_Static_assert(sizeof(struct prepared_send)==256 && _Alignof(struct prepared_send)==128 &&
  offsetof(struct prepared_send,span)==16 && offsetof(struct prepared_send,request)==32 &&
  offsetof(struct prepared_send,request.send_flags)+sizeof(unsigned int)<=128,"M29");
/* design/prepared-machine.md#M08: d6dceac's prepared receive, verbatim, its landed word `input` and the value its
   landing stores `argument` (`position`), `next` the next receive on its queue pair; `hop` and `need` added: its hop's
   place in the pair's receives (struct flow_hop) and the leading executions of that hop landed before it is posted */
struct prepared_receive {
  _Alignas(128) struct ibv_recv_wr request;
  struct ibv_sge span;
  _Alignas(32) struct ibv_recv_wr *next;
  struct ibv_qp *pair;
  _Atomic uint64_t *input;
  uint64_t argument;
  uint32_t hop,need;
};
_Static_assert(sizeof(struct prepared_receive)==128 && _Alignof(struct prepared_receive)==128 &&
  offsetof(struct prepared_receive,span)==32 && offsetof(struct prepared_receive,next)==64 &&
  offsetof(struct prepared_receive,pair)==72 && offsetof(struct prepared_receive,input)==80 &&
  offsetof(struct prepared_receive,argument)==88,"M08 M09");
/* design/prepared-machine.md#M27 */
struct mesh_trace { _Alignas(32) uint64_t identity; uint64_t begin,middle,end; };
_Static_assert(sizeof(struct mesh_trace)==32 && _Alignof(struct mesh_trace)==32,"M27");
/* a receive hop of the node pair, its execution-0 landed word (a region offset), its position, and the leading
   executions of it this child has read landed (`through`: mesh.h, a receive of execution t is posted once its
   receives of executions 0 .. t - (depth - 1) have landed) */
struct flow_hop { uint64_t word,position,through; };
/* A port's plan as its child serves it, the progress thread's alone: none loaded; refused (the view `hops`, `plan` it
   refused, until the plan call writes again); served; done (every SEND of its share completed, every receive landed);
   or ended with work outstanding on the link's queue pairs, which only the child's exit releases (LINK_STREAMS_ENDED),
   so nothing more is loaded in the child's life.  Its records by lane: SENDs [send_at[k], send_end[k]) still to post,
   receives from receive_next[k] on (a chain), the pair's receive hops (`landing`), the plan's word stride, the
   completions of its share (`sent`, `landed`) against its counts, and the traces (MESH_LEDGER) */
enum { FLOW_NONE, FLOW_REFUSED, FLOW_SERVING, FLOW_DONE, FLOW_ENDED };
struct flow {
  int state,ledger,refused;
  uint32_t lanes;
  uint64_t hops,plan,client,stride,send_count,receive_count,sent,landed;
  struct prepared_send *sends,*send_at[MESH_LINK_LANES],*send_end[MESH_LINK_LANES];
  struct prepared_receive *receives,*receive_next[MESH_LINK_LANES];
  struct flow_hop *landing;
  struct mesh_trace *trace[2];
  size_t traced[2],trace_capacity[2];
};
static struct flow flows[MESH_PORTS];

/* tag pre-return-20261103 rdma/mesh-link.c:56-65 port_halt (3f72cbde, 306b22be), verbatim but not static */
/* design/transport.md: a port halted from any thread (a peer interface's termination heard, a provider's fault
   absorbed; atomics alone): its pairing lost, so no job of it makes a further call (B10), then no lane of its link
   served, both sequentially consistent (link_turn's connect stores served and reads the pairing again), and its bit set
   in `which` for the event loop's conclusion (terminations_turn, ports_stalled) */
_Static_assert(MESH_PORTS<=32,"a bit a port");
void port_halt(struct port *port,_Atomic uint32_t *which){
  atomic_fetch_add(&port->link.pairing,1);
  atomic_store(&port->link.served,0);
  atomic_fetch_or(which,UINT32_C(1)<<port->index);
}
/* tag pre-return-20261103 rdma/mesh-link.c:71-81 lane_lost (d556370e), verbatim */
/* a lane's failure, found by a progress thread: served no more, and the link lost by the event loop on reading
   `broken` (link_turn) */
static void lane_lost(struct port *port,uint32_t lane,int64_t code){
  struct link *l=&port->link;
  if(atomic_fetch_and_explicit(&l->served,~(UINT32_C(1)<<lane),memory_order_acq_rel)>>lane&1)
    mesh_say("bridge","link %u (%s): lane %u failed (code %lld): the link toward node %u lost",port->index,port->name,lane,
             (long long)code,port->peer);
  atomic_store_explicit(&l->lost,code,memory_order_relaxed);
  atomic_fetch_or_explicit(&l->broken,UINT32_C(1)<<lane,memory_order_release);
  wake();
}
/* tag pre-return-20261103 rdma/mesh-link.c:275-286 link_failed (d556370e), verbatim */
/* design/transport.md: the link's first failed completion, on any lane, or a failed poll: no lane of the
   link served from here, so nothing is posted on it again, and the link lost by the event loop on reading `broken`
   (link_turn), the stream's words ~0 (port_lost) */
static void link_failed(struct port *port,int64_t code){
  struct link *l=&port->link;
  const uint32_t served=atomic_exchange_explicit(&l->served,0,memory_order_acq_rel);
  if(served)mesh_say("bridge","link %u (%s): a completion or its poll failed (code %lld): no lane served, the link toward node %u lost",
                     port->index,port->name,(long long)code,port->peer);
  atomic_store_explicit(&l->lost,code,memory_order_relaxed);
  atomic_fetch_or_explicit(&l->broken,served,memory_order_release);
  wake();
}

/* tag pre-return-20261103 rdma/mesh-dataflow.c:79-103 mesh_records_read (e3c4c986), adapted: the region the bridge's
   own (M), the map's serial read inside the same reading, an error value in place of -errno, and its deadline the
   bridge's clock (main's mesh.h has no struct mesh_deadline); the bridge that writes the map is this child's parent, so
   a write that stays open past MESH_PLATFORM_NS is ETIMEDOUT, not the tag's ESRCH */
/* the map's records as one serial left them, the seqlock's reader (mesh_map_whole): a copy in *out the caller frees,
   its count and the map's serial with it, or an error value; the read within its deadline, MESH_PLATFORM_NS from its
   start */
static int map_read(struct mesh_record **out,uint64_t *count,uint64_t *serial){
  const uint64_t until=clock_ns()+MESH_PLATFORM_NS;
  for(;;){
    const uint64_t before=atomic_load_explicit(&M->map,memory_order_acquire),at=clock_ns();
    if(at>=until)return ETIMEDOUT;
    if(before&1){
      const uint64_t left=until-at;
      os_sync_wait_on_address_with_timeout(&M->map,before,sizeof M->map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                           OS_CLOCK_MACH_ABSOLUTE_TIME,left<MESH_PAUSE_NS?left:MESH_PAUSE_NS);
      continue;
    }
    const uint64_t n=atomic_load_explicit(&M->map_nodes,memory_order_relaxed),off=atomic_load_explicit(&M->map_off,memory_order_relaxed);
    const int fits=off<=M->length && n<=(M->length-off)/sizeof(struct mesh_record);
    struct mesh_record *copy=fits?malloc((n?n:1)*sizeof *copy):NULL;
    if(copy)memcpy(copy,(char *)M+off,n*sizeof *copy);
    const uint64_t s=atomic_load_explicit(&M->map_serial,memory_order_relaxed);
    if(mesh_map_whole(M,before)){
      if(!copy)return fits?ENOMEM:EINVAL;
      *out=copy;*count=n;*serial=s;return 0;
    }
    free(copy);
  }
}
/* a node's index in the map: its place in ascending node-id order among the map's records (mesh.h hdr `map_serial`);
   UINT32_MAX where the map holds no record of it */
static uint32_t map_index(const struct mesh_record *records,uint64_t count,uint32_t node){
  uint32_t index=0,held=0;
  for(uint64_t r=0;r<count;r++){index+=records[r].node<node;held|=records[r].node==node;}
  return held?index:UINT32_MAX;
}
/* the plan written as the bridge reads it (mesh.h, "Executions"): its executions (main's invocations, at least one),
   its ring depth (d6dceac link_configure's: depth == executions where unset or past them, the unrung machine) and its
   hops a slot */
static void plan_shape(uint64_t hops,uint32_t *executions,uint32_t *depth,uint64_t *slot){
  *executions=M->executions?M->executions:1;
  *depth=atomic_load_explicit(&M->depth,memory_order_relaxed);
  if(!*depth || *depth>*executions)*depth=*executions;
  *slot=hops/ *depth;
}
static inline _Atomic uint64_t *region_word(uint64_t offset){ return (_Atomic uint64_t *)(void *)((char *)M+offset); }

/* The rule both ends compute (design/page-queue-change-request.md; mesh.h struct mesh_hop, "Both ends"), and this
   port's share of the plan built from it.  The plan written (hdr `plan` its generation) is refused (ESTALE) where its
   entries' `serial` is not the map's serial as one whole reading of the map left it.  The cables between this node and
   the port's peer are the reach entries both records list for each other (each from its own side, the ports crossed),
   one cable an entry of each; a cable's key is (the lower
   node id's port index, the higher's), cables in key order; its bandwidth the lesser of the two records' and its queue
   pairs the lesser of the two (the pairing's, the same at both ends).  The pair's links are each cable's queue pairs
   0 .. qps - 1 in (key, queue pair) order, each of rate the cable's bandwidth over its queue pairs.  Each direction's
   hops of the pair (slot 0's entries naming the peer's index, in list order) are units of execution 0 in list order,
   then execution 1's, ..., every one a page (block x pgsz bytes), assigned by mesh_graph_list_schedule (Graham's list
   scheduling, mesh-graph.h): this node's SENDs to the peer are the peer's receives from this node, so both ends assign
   one list alike.  Kept: the units on this port's cable, each on its queue pair (lane q, link_plan), in assignment
   order, so the k-th SEND on a queue pair fills the k-th receive posted on it.  A SEND of execution t sends slot (t mod
   depth)'s page of its hop, released by its word of execution t (word + t hop_stride) reaching `position`; a relay is
   no special case, its SEND naming its receive's landed word.  A receive of execution t lands in slot (t mod depth)'s
   page, storing `position` into its word of execution t, and where the plan rings (depth < executions) is posted once
   its hop's executions 0 .. t - window have landed, the window d6dceac link_configure's (depth - 1, at least 1: a slot
   short of the ring; mesh.h says t - (depth - 1), which for depth 1 asks t's own landing, so main's 1 stands).  No
   verbs call: the progress thread posts as it serves (link_receive_progress, link_send_progress).  0, or an error
   value: ESTALE, ETIMEDOUT or ENOMEM reading the map, EHOSTUNREACH where the pair has units but no link of rate,
   EPROTO where the map gives this cable a queue pair its link does not hold, EMSGSIZE where a page is more frames than
   a lane it is assigned can hold, ENOTSUP where the plan rings and a page this cable receives into is one this node
   sends from (below) */
static int link_prepare(struct port *port){
  struct link *l=&port->link;
  struct flow *f=flows+port->index;
  const uint64_t hops=atomic_load_explicit(&M->hops,memory_order_acquire);
  const uint64_t plan=atomic_load_explicit(&M->plan,memory_order_relaxed);
  f->hops=hops;f->plan=plan;f->client=atomic_load_explicit(&M->client,memory_order_acquire);f->stride=M->hop_stride;
  uint32_t executions,depth;uint64_t n;
  plan_shape(hops,&executions,&depth,&n);
  const struct mesh_hop *list=mesh_hops(M);
  struct mesh_record *records=NULL;uint64_t count=0,serial=0;
  int error=map_read(&records,&count,&serial);
  if(error)return error;
  if(hops && list[0].serial!=serial){free(records);return ESTALE;}
  const uint32_t self=M->node,peer=port->peer,index=map_index(records,count,peer);
  const struct mesh_record *mine=NULL,*theirs=NULL;
  for(uint64_t r=0;r<count;r++){if(records[r].node==self)mine=records+r;if(records[r].node==peer)theirs=records+r;}
  struct cable { uint32_t low,high,qps,port; uint64_t bandwidth; } cables[MESH_PORTS],swap;
  uint32_t cable_count=0;
  for(uint32_t e=0;mine && theirs && e<mine->reached && e<MESH_PORTS;e++){
    const struct mesh_reach *a=mine->reach+e;
    if(a->node!=peer)continue;
    for(uint32_t g=0;g<theirs->reached && g<MESH_PORTS;g++){
      const struct mesh_reach *b=theirs->reach+g;
      if(b->node!=self || b->port!=a->peer_port || b->peer_port!=a->port)continue;
      cables[cable_count++]=(struct cable){.low=self<peer?a->port:a->peer_port,.high=self<peer?a->peer_port:a->port,
        .qps=a->qps<b->qps?a->qps:b->qps,.port=a->port,.bandwidth=a->bandwidth<b->bandwidth?a->bandwidth:b->bandwidth};
      break;
    }
  }
  free(records);
  for(uint32_t i=1;i<cable_count;i++)
    for(uint32_t j=i;j && (cables[j-1].low>cables[j].low || (cables[j-1].low==cables[j].low && cables[j-1].high>cables[j].high));j--){
      swap=cables[j-1];cables[j-1]=cables[j];cables[j]=swap;
    }
  double rate[MESH_PORTS*MESH_LINK_LANES];
  uint32_t on[MESH_PORTS*MESH_LINK_LANES],queue[MESH_PORTS*MESH_LINK_LANES],links=0;
  for(uint32_t c=0;c<cable_count;c++){
    if(cables[c].qps>MESH_LINK_LANES)return EPROTO;
    for(uint32_t q=0;q<cables[c].qps;q++){
      rate[links]=(double)cables[c].bandwidth/(double)cables[c].qps;on[links]=c;queue[links]=q;links++;
    }
  }
  /* each direction's hops of the pair, slot 0's in list order (every slot holds the same hops in the same order) */
  uint32_t *pick[2]={NULL,NULL};uint64_t picks[2]={0,0},units[2]={0,0};
  uint64_t *bytes=NULL,*carried=NULL;uint32_t *assigned[2]={NULL,NULL};
  for(int d=0;d<2 && index!=UINT32_MAX;d++){
    if(!(pick[d]=malloc((n?n:1)*sizeof *pick[d]))){error=ENOMEM;goto failed;}
    for(uint64_t j=0;j<n;j++)if(list[j].node==index && list[j].direction==(uint32_t)d)pick[d][picks[d]++]=(uint32_t)j;
    units[d]=picks[d]*executions;
  }
  /* A page this node both receives into and sends from (a relay's, or a destination's that forwards it) is used again
     every `depth` executions, so where the plan rings, its receive of execution t may be posted only once the page's
     SENDs of execution t - depth have completed, not only its receives landed (the coordinator, 2026-11-03, from
     mesh_plan 80e2e4d3).  Those SENDs are another port's child's (the page goes on to another node, over another
     cable), and no word of the region tells their completion: a hop names its release word and its landed word, no
     sent word (mesh.h struct mesh_hop; mesh-collective.c plan_word lays the words 32 a line, and a landed word reads
     exactly 1 once landed).  So such a plan is refused (ENOTSUP) where it rings, not served into a page a SEND may
     still be reading.  Every slot holds the same hops, a relay's SENDs from its receive's page in each, so slot 0
     tells.  What must stay: the refusal, until the plan carries a sent word per SEND and execution, stored by the
     SEND's child at its completion, which this child reads before posting the page's next receive */
  for(uint64_t i=0;depth<executions && i<picks[MESH_RECEIVE];i++)
    for(uint64_t j=0;j<n;j++)
      if(list[j].direction==MESH_SEND && list[j].page==list[pick[MESH_RECEIVE][i]].page){error=ENOTSUP;goto failed;}
  const uint64_t most=units[0]>units[1]?units[0]:units[1];
  if(most){
    bytes=malloc(most*sizeof *bytes);carried=malloc((links?links:1)*sizeof *carried);
    if(!bytes || !carried){error=ENOMEM;goto failed;}
    for(uint64_t k=0;k<most;k++)bytes[k]=page_bytes();
  }
  /* the assignment, and this cable's units on each of its queue pairs */
  uint64_t lane_units[2][MESH_LINK_LANES]={{0}};
  for(int d=0;d<2;d++){
    if(!units[d])continue;
    if(!(assigned[d]=malloc(units[d]*sizeof *assigned[d]))){error=ENOMEM;goto failed;}
    if(isinf(mesh_graph_list_schedule(links,rate,units[d],bytes,assigned[d],carried))){error=EHOSTUNREACH;goto failed;}
    for(uint64_t k=0;k<units[d];k++){
      const uint32_t a=assigned[d][k];
      if(cables[on[a]].port!=port->index){assigned[d][k]=UINT32_MAX;continue;}
      const uint32_t q=queue[a];
      if(!(l->plan>>q&1) || !l->provider.queues || (int)q>=l->provider.count || !l->provider.queues[q].pair){error=EPROTO;goto failed;}
      assigned[d][k]=q;lane_units[d][q]++;
    }
  }
  /* a page a lane's queue cannot hold refused, not left unposted: its frames past the lane's receive room (lane_reset's
     frame_capacity) or past its SEND gate's (link_connect) */
  const uint64_t frames=frames_of(page_bytes()).n;
  for(uint32_t q=0;q<MESH_LINK_LANES;q++)
    if((lane_units[MESH_RECEIVE][q] && frames>l->lane[q].frame_capacity) || (lane_units[MESH_SEND][q] && frames>l->gate[q].capacity.n)){
      error=EMSGSIZE;goto failed;
    }
  /* the records, by lane in assignment order */
  f->send_count=f->receive_count=0;
  for(uint32_t q=0;q<MESH_LINK_LANES;q++){f->send_count+=lane_units[MESH_SEND][q];f->receive_count+=lane_units[MESH_RECEIVE][q];}
  f->sends=f->send_count?aligned_alloc(128,f->send_count*sizeof *f->sends):NULL;
  f->receives=f->receive_count?aligned_alloc(128,f->receive_count*sizeof *f->receives):NULL;
  f->landing=picks[MESH_RECEIVE]?calloc(picks[MESH_RECEIVE],sizeof *f->landing):NULL;
  if((f->send_count && !f->sends) || (f->receive_count && !f->receives) || (picks[MESH_RECEIVE] && !f->landing)){error=ENOMEM;goto failed;}
  for(uint64_t i=0;i<picks[MESH_RECEIVE];i++)
    f->landing[i]=(struct flow_hop){.word=list[pick[MESH_RECEIVE][i]].word,.position=list[pick[MESH_RECEIVE][i]].position};
  struct prepared_send *send_fill[MESH_LINK_LANES];
  struct prepared_receive *receive_fill[MESH_LINK_LANES],*receive_last[MESH_LINK_LANES];
  f->lanes=0;
  for(uint64_t q=0,s=0,r=0;q<MESH_LINK_LANES;q++){
    f->send_at[q]=send_fill[q]=f->sends?f->sends+s:NULL;f->send_end[q]=f->sends?f->sends+s+lane_units[MESH_SEND][q]:NULL;
    s+=lane_units[MESH_SEND][q];
    f->receive_next[q]=receive_fill[q]=lane_units[MESH_RECEIVE][q]?f->receives+r:NULL;receive_last[q]=NULL;
    r+=lane_units[MESH_RECEIVE][q];
    if(lane_units[MESH_SEND][q] || lane_units[MESH_RECEIVE][q])f->lanes|=UINT32_C(1)<<q;
  }
  const uint32_t window=depth<executions?(depth>1?depth-1:1):executions;
  for(int d=0;d<2;d++)for(uint64_t k=0;k<units[d];k++){
    const uint32_t q=assigned[d][k];
    if(q==UINT32_MAX)continue;
    const uint64_t t=k/picks[d],i=k%picks[d],j=pick[d][i];
    const struct mesh_hop *hop=list+(t%depth)*n+j;
    const struct ibv_sge span=wire_span(&port->device,(uint64_t)hop->page*M->pgsz,page_bytes());
    _Atomic uint64_t *word=region_word(list[j].word+t*f->stride);
    if(d==MESH_SEND){
      struct prepared_send *record=send_fill[q]++;
      *record=(struct prepared_send){.word=word,.position=list[j].position,.span=span};
      record->request=(struct ibv_send_wr){.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1,.opcode=IBV_WR_SEND,
        .send_flags=IBV_SEND_SIGNALED};
    } else {
      struct prepared_receive *record=receive_fill[q]++;
      *record=(struct prepared_receive){.span=span,.pair=l->provider.queues[q].pair,.input=word,.argument=list[j].position,
        .hop=(uint32_t)i,.need=depth<executions && t+1>window?(uint32_t)(t+1-window):0};
      record->request=(struct ibv_recv_wr){.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1};
      if(receive_last[q])receive_last[q]->next=&record->request;
      receive_last[q]=record;
    }
  }
  /* design/prepared-machine.md#M27 */
  const char *ledger=getenv("MESH_LEDGER");
  f->ledger=ledger && ledger[0] && !(ledger[0]=='0' && !ledger[1]);
  const uint64_t traces[2]={f->send_count,f->receive_count};
  for(int d=0;d<2;d++){
    f->traced[d]=0;f->trace_capacity[d]=f->ledger?traces[d]:0;f->trace[d]=NULL;
    if(!f->trace_capacity[d])continue;
    if(!(f->trace[d]=aligned_alloc(32,(f->trace_capacity[d]+1)*sizeof(struct mesh_trace)))){error=ENOMEM;goto failed;}
  }
  if(f->ledger)for(uint32_t q=0;q<MESH_LINK_LANES;q++)if(f->lanes>>q&1)
    fprintf(stderr,"{\"trace_layout\":%u,\"rank\":%u,\"lane\":%u,\"send_base\":%llu,\"sends\":%llu,\"receive_base\":%llu,\"receives\":%llu,"
      "\"send_record_bytes\":%zu,\"receive_record_bytes\":%zu,\"executions\":%u,\"depth\":%u,\"window\":%u}\n",port->index,M->node,q,
      (unsigned long long)(uintptr_t)f->send_at[q],(unsigned long long)lane_units[MESH_SEND][q],
      (unsigned long long)(uintptr_t)f->receive_next[q],(unsigned long long)lane_units[MESH_RECEIVE][q],
      sizeof(struct prepared_send),sizeof(struct prepared_receive),executions,depth,window);
  f->sent=f->landed=0;
  for(int d=0;d<2;d++){free(pick[d]);free(assigned[d]);}
  free(bytes);free(carried);
  return 0;
failed:
  for(int d=0;d<2;d++){free(pick[d]);free(assigned[d]);free(f->trace[d]);f->trace[d]=NULL;}
  free(bytes);free(carried);
  free(f->sends);free(f->receives);free(f->landing);
  f->sends=NULL;f->receives=NULL;f->landing=NULL;f->send_count=f->receive_count=0;f->lanes=0;
  return error;
}

/* whether the receive's hop has landed its leading executions through the receive's `need` (mesh.h: its word of each
   reaches its position, below UINT64_MAX), read from the region's words, which every cable of the pair lands into */
static int landed_through(struct flow *f,const struct prepared_receive *record){
  struct flow_hop *hop=f->landing+record->hop;
  while(hop->through<record->need){
    const uint64_t value=atomic_load_explicit(region_word(hop->word+hop->through*f->stride),memory_order_acquire);
    if(value<hop->position || value==UINT64_MAX)return 0;
    hop->through++;
  }
  return 1;
}

/* design/prepared-machine.md#M08 */
/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#programcopy */
/* d6dceac's receive drain (its unordered form: a completion names its record), now one pass of the port's progress
   thread over the link's receive completions (the queue pairs' shared queue, `completion`), in the tag's batches
   (rdma/mesh-link.c link_receive): each landing of a lane served stores its record's `position` (release) into its
   landed word of its execution, after the acquire the poll's DMA asks, and is counted (the lane's `got` and frames,
   which lanes_flush reads); a failed one fails the link (link_failed), never retried.  Then each served lane's
   receives posted ahead from its chain, in order, while its queue has room (the tag's requests and frames) and, where
   the plan rings, its hop's leading executions have landed; at most MESH_POST_QUANTUM a pass.  0 where the link failed */
static __attribute__((always_inline)) inline int link_receive_drain(struct port *port,struct flow *f,uint32_t served,int traced){
  struct link *l=&port->link;
  struct ibv_wc done[MESH_COMPLETION_BATCH];
  int (*poll)(struct ibv_cq *,int,struct ibv_wc *)=l->provider.poll;
  const uint64_t each=frames_of(page_bytes()).n;
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace=traced?f->trace[MESH_RECEIVE]:NULL;
  const size_t capacity=traced?f->trace_capacity[MESH_RECEIVE]:0;
  for(int count=MESH_COMPLETION_BATCH;count==MESH_COMPLETION_BATCH;){
    count=poll(l->provider.completion,MESH_COMPLETION_BATCH,done);
    if(count<0){link_failed(port,EIO);return 0;}
    if(count)atomic_thread_fence(memory_order_acquire);
    for(int i=0;i<count;i++){
      const uint32_t lane=link_lane(l,done[i].qp_num);
      if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
      uint64_t status_opcode;
      memcpy(&status_opcode,&done[i].status,sizeof status_opcode);
      if((uint32_t)status_opcode){link_failed(port,(uint32_t)status_opcode);return 0;}
      if(!((status_opcode>>32)&IBV_WC_RECV))continue;
      const uint64_t polled=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
      struct prepared_receive *record=(void *)(uintptr_t)done[i].wr_id;
      atomic_store_explicit(record->input,record->argument,memory_order_release);
      struct lane_state *state=l->lane+lane;
      state->got++;state->frames_retired+=each;f->landed++;
      if(traced && f->traced[MESH_RECEIVE]<capacity){
        const uint64_t published=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        trace[f->traced[MESH_RECEIVE]++]=(struct mesh_trace){(uintptr_t)record,polled,published,published};
      }
    }
  }
  for(uint32_t lanes=f->lanes&served;lanes;lanes&=lanes-1){
    const uint32_t k=(uint32_t)__builtin_ctz(lanes);
    struct lane_state *state=l->lane+k;
    const struct mesh_queue *queue=l->provider.queues+k;
    struct prepared_receive *record=f->receive_next[k];
    for(uint32_t budget=MESH_POST_QUANTUM;record && budget;budget--){
      if(state->posted-state->got>=MESH_LANE_UNITS || state->frames_posted-state->frames_retired+each>state->frame_capacity)break;
      if(record->need && !landed_through(f,record))break;
      struct ibv_recv_wr *bad;
      const int error=queue->receive(record->pair,&record->request,&bad);
      if(error){lane_lost(port,k,error<0?-error:error);break;}
      state->frames_posted+=each;state->posted++;
      record=(struct prepared_receive *)(void *)record->next;
    }
    f->receive_next[k]=record;
  }
  return 1;
}
/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#programcopy */
/* d6dceac link_receive_progress, now a pass and not a thread's entry: its traced variant under MESH_LEDGER */
static int link_receive_progress(struct port *port,uint32_t served){
  struct flow *f=flows+port->index;
  /* design/prepared-machine.md#M27 */
  if(f->trace[MESH_RECEIVE])return link_receive_drain(port,f,served,1);
  return link_receive_drain(port,f,served,0);
}

/* design/prepared-machine.md#M04 */
/* design/prepared-machine.md#M06 */
/* design/prepared-machine.md#M15 */
/* design/algorithm-sources.md#independent-native-queues */
/* d6dceac's send drain (each queue pair's chain from its cursor, a record posted as its release word holds, through the
   provider's own post), now one pass of the port's progress thread: the SEND completions first, on their own queue
   (`sent`, mesh-verbs.h pair_up), in the tag's batches (rdma/mesh-link.c link_send), each of a lane served retiring its
   frames through the lane's gate and counted (`done`), a failed one failing the link; then each served lane's SENDs in
   order, each once its word of its execution reaches its `position` (acquire; UINT64_MAX releases nothing), while the
   lane's gate holds its frames and its requests are under MESH_LANE_UNITS, at most MESH_POST_QUANTUM a pass.  A ready
   page waits only behind earlier pages of its own queue pair.  0 where the link failed */
static __attribute__((always_inline)) inline int link_send_drain(struct port *port,struct flow *f,uint32_t served,int traced){
  struct link *l=&port->link;
  struct ibv_wc done[MESH_COMPLETION_BATCH];
  const struct frames each=frames_of(page_bytes());
  /* design/prepared-machine.md#M27 */
  struct mesh_trace *trace=traced?f->trace[MESH_SEND]:NULL;
  const size_t capacity=traced?f->trace_capacity[MESH_SEND]:0;
  for(int count=MESH_COMPLETION_BATCH;count==MESH_COMPLETION_BATCH;){
    count=l->provider.poll(l->provider.sent,MESH_COMPLETION_BATCH,done);
    if(count<0){link_failed(port,EIO);return 0;}
    for(int i=0;i<count;i++){
      const uint32_t lane=link_lane(l,done[i].qp_num);
      if(lane>=MESH_LINK_LANES || !(served>>lane&1))continue;
      if(done[i].status){link_failed(port,(int64_t)done[i].status);return 0;}
      gate_retire(l->gate+lane,each);l->lane[lane].done++;f->sent++;
    }
  }
  for(uint32_t lanes=f->lanes&served;lanes;lanes&=lanes-1){
    const uint32_t k=(uint32_t)__builtin_ctz(lanes);
    struct lane_state *state=l->lane+k;
    struct send_gate *gate=l->gate+k;
    const struct mesh_queue *queue=l->provider.queues+k;
    int (*post)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **)=queue->send;
    struct prepared_send *record=f->send_at[k],*const end=f->send_end[k];
    for(uint32_t budget=MESH_POST_QUANTUM;record<end && budget;budget--){
      const uint64_t value=atomic_load_explicit(record->word,memory_order_acquire);
      if(value<record->position || value==UINT64_MAX)break;
      if(state->sent-state->done>=MESH_LANE_UNITS || !gate_holds(gate,each))break;
      const uint64_t observed=traced?clock_gettime_nsec_np(CLOCK_UPTIME_RAW):0;
      struct ibv_send_wr *bad;
      const int error=post(queue->pair,&record->request,&bad);
      if(traced && f->traced[MESH_SEND]<capacity){
        const uint64_t sent=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        trace[f->traced[MESH_SEND]++]=(struct mesh_trace){(uintptr_t)record,observed,observed,sent};
      }
      if(error){lane_lost(port,k,error<0?-error:error);break;}
      gate_post(gate,each);
      __atomic_store_n(&state->sent,state->sent+1,__ATOMIC_RELEASE);gate->requests++;
      record++;
    }
    f->send_at[k]=record;
  }
  return 1;
}
/* design/prepared-machine.md#M08 */
/* design/algorithm-sources.md#independent-native-queues */
/* d6dceac link_send_progress, now a pass and not a thread's entry: its traced variant under MESH_LEDGER */
static int link_send_progress(struct port *port,uint32_t served){
  struct flow *f=flows+port->index;
  /* design/prepared-machine.md#M27 */
  if(f->trace[MESH_SEND])return link_send_drain(port,f,served,1);
  return link_send_drain(port,f,served,0);
}

/* design/prepared-machine.md#M11 */
/* design/algorithm-sources.md#meshresult */
/* The end of the plan a port's child serves, on its progress thread: where the plan was written again (hdr `hops` 0,
   the plan call's first store, or another plan's), where the link serves no lane (a halt, a loss), and as the child
   closes.  Its landed words not landed are cancelled (mesh.h: UINT64_MAX, stored only where a word reads 0, so a
   landing is never overwritten); its traces told (d6dceac link_close's lines); its records freed; and the port's
   `prepared` 0, no plan loaded.  Where a SEND or receive of the link's queue pairs is outstanding (the lanes' counts),
   nothing more is loaded in the child's life, and where the link still serves, it is lost as the tag loses a link whose
   holdings ended with units outstanding (rdma/mesh-link.c stream_end: `lost` LINK_STREAMS_ENDED, every lane `broken`,
   so the child tells CHILD_BROKEN and its parent loses the link): a posted receive is never taken back but by the
   child's exit (lanes_flush), so no later plan's page is posted beside it, and `prepared` stays the plan's until the
   parent reaps the child.  No verbs call: d6dceac's down_pair and listener close are the child's exit's */
static void link_close(struct port *port,uint32_t served){
  struct link *l=&port->link;
  struct flow *f=flows+port->index;
  if(f->state==FLOW_NONE || f->state==FLOW_ENDED)return;
  if(f->state==FLOW_REFUSED){f->state=FLOW_NONE;return;}
  for(uint64_t r=0;r<f->receive_count;r++){
    uint64_t idle=0;
    atomic_compare_exchange_strong_explicit(f->receives[r].input,&idle,UINT64_MAX,memory_order_release,memory_order_relaxed);
  }
  /* design/prepared-machine.md#M27 */
  for(int d=0;d<2;d++){
    for(size_t i=0;i<f->traced[d];i++){
      const struct mesh_trace t=f->trace[d][i];
      fprintf(stderr,"{\"native_trace\":%u,\"direction\":%d,\"event\":%zu,\"identity\":%llu,\"ns\":[%llu,%llu,%llu]}\n",
        port->index,d,i,(unsigned long long)t.identity,(unsigned long long)t.begin,(unsigned long long)t.middle,(unsigned long long)t.end);
    }
    free(f->trace[d]);f->trace[d]=NULL;f->traced[d]=0;f->trace_capacity[d]=0;
  }
  uint32_t outstanding=0;
  for(uint32_t k=0;k<MESH_LINK_LANES;k++)
    if(l->plan>>k&1 && (l->lane[k].sent!=l->lane[k].done || l->lane[k].posted!=l->lane[k].got))outstanding|=UINT32_C(1)<<k;
  free(f->sends);free(f->receives);free(f->landing);
  f->sends=NULL;f->receives=NULL;f->landing=NULL;f->send_count=f->receive_count=0;f->lanes=0;
  memset(f->send_at,0,sizeof f->send_at);memset(f->send_end,0,sizeof f->send_end);memset(f->receive_next,0,sizeof f->receive_next);
  f->state=outstanding?FLOW_ENDED:FLOW_NONE;
  /* `prepared` 0 at once where nothing is outstanding; else it stays the plan's until the parent reaps the child
     (mesh-children.c child_reaped, streams_cancel), so a retired client's rows its requests name are kept
     (mesh-dataflow.c mesh_retired_release reads every port's `prepared`: the coordinator, 2026-11-03) */
  if(!outstanding)atomic_store_explicit(&mesh_links(M)[port->index].port.prepared,0,memory_order_release);
  if(outstanding && served){
    mesh_say("bridge","link %u (%s): its plan ended with work outstanding on lanes %#x: the link toward node %u lost",
             port->index,port->name,outstanding,port->peer);
    atomic_store_explicit(&l->lost,LINK_STREAMS_ENDED,memory_order_relaxed);
    atomic_fetch_or_explicit(&l->broken,l->plan,memory_order_release);
    wake();
  }
}

/* The plan's turn on the port's progress thread, each pass while its link serves: the plan served ended where the
   region's was written again (link_close: hdr `hops` 0, or another generation `plan`) or the client it was loaded for
   holds the region no more (hdr `client` another: mesh-dataflow.c mesh_retire); a plan written while a client holds
   the region and not yet loaded loaded (link_prepare), its generation then stored in the port's `prepared` (release: the client's start
   waits on it at every port whose cable reaches a node a hop names, so it is stored where the rule gave this cable no
   page too), or refused, the port's phase MESH_STOPPED and its `code` the refusal's (CONTRACT "A plan loaded"), until
   the plan call writes again; a plan loaded
   served, its receives, then its SENDs (a landed relay page's SEND goes in the same pass), until every SEND of its share
   completed and every receive landed.  1 while it serves a plan with work left, 0 idle */
static int link_run(struct port *port,uint32_t served){
  struct flow *f=flows+port->index;
  const uint64_t hops=atomic_load_explicit(&M->hops,memory_order_acquire);
  const uint64_t plan=atomic_load_explicit(&M->plan,memory_order_relaxed);
  const uint64_t client=atomic_load_explicit(&M->client,memory_order_acquire);
  if(f->state!=FLOW_NONE && f->state!=FLOW_ENDED && (!hops || hops!=f->hops || plan!=f->plan || client!=f->client))
    link_close(port,served);
  if(f->state==FLOW_NONE && hops && client){
    struct mesh_port_info *info=&mesh_links(M)[port->index].port;
    const int error=link_prepare(port);
    if(error){
      f->state=FLOW_REFUSED;f->refused=1;
      mesh_say("bridge","link %u (%s): plan %llu refused (code %d)",port->index,port->name,
               (unsigned long long)f->plan,error);
      atomic_store_explicit(&info->prepared,0,memory_order_release);
      atomic_store_explicit(&info->code,error,memory_order_relaxed);
      atomic_store_explicit(&info->phase,MESH_STOPPED,memory_order_release);
      return 0;
    }
    /* a refusal of this child's undone where a later plan loads: its phase stopped and code were the refusal's */
    if(f->refused){
      f->refused=0;
      atomic_store_explicit(&info->code,0,memory_order_relaxed);
      atomic_store_explicit(&info->phase,MESH_PAIRED,memory_order_release);
    }
    f->state=FLOW_SERVING;
    atomic_store_explicit(&info->prepared,f->plan,memory_order_release);
  }
  if(f->state!=FLOW_SERVING)return 0;
  if(!link_receive_progress(port,served) || !link_send_progress(port,served))return 0;
  if(f->sent==f->send_count && f->landed==f->receive_count){f->state=FLOW_DONE;return 0;}
  return 1;
}

/* tag pre-return-20261103 rdma/mesh-link.c:367-371 port_tell (60b00774), verbatim */
/* the progress thread's tell (the epoch it has seen, the guard it holds at), its raiser woken (port_wait) */
static void port_tell(_Atomic uint64_t *word,uint64_t value){
  atomic_store_explicit(word,value,memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_NONE);
}
/* tag pre-return-20261103 rdma/mesh-link.c:372-386 port_pass (8b757084), verbatim but its work: the plan's turn
   (link_run) in place of each stream's holding, SENDs, receives and completions, and whether it has work left told */
/* design/transport.md section 14: one pass of a port's progress thread: the lanes the link serves (`served` read once a
   pass: none posts on a lane its link no longer serves), the plan's turn on them (its receives first: a peer's SEND is
   held until its receive is posted, TN3205, so ours let its SENDs go).  It tells the port's epoch it began at (a
   conclusion around a stuck call moves it: ports_stalled).  1 where the plan has work left */
static int port_pass(struct port *port,uint64_t *told){
  const uint64_t epoch=atomic_load(&port->epoch);
  struct link *l=&port->link;
  const uint32_t served=atomic_load_explicit(&l->served,memory_order_acquire);
  const int busy=served?link_run(port,served):0;
  if(epoch!=*told){port_tell(&port->seen,epoch);*told=epoch;}
  return busy;
}
/* tag pre-return-20261103 rdma/mesh-link.c:387-424 port_progress (3a82ede9), verbatim but the plan: while the link
   serves and its plan has no work left (none written, or done), the thread waits on the plan's word (hdr `plan`, its
   generation, region-shared) as it waits on the guard, looking again every MESH_PAUSE_NS (so a client's retirement,
   which wakes `control` and not `plan`, is seen within it); a link that serves no lane ends the plan it served
   (link_close: a halt); and the thread's end ends it too */
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
    const uint32_t served=atomic_load_explicit(&port->link.served,memory_order_acquire);
    if(guard&1 || !served){
      /* a fault absorbed on the port (fault_heard: the handler, which may make no call but mmap, munmap and sigaction,
         halted the port) wakes its child's loop, which tells its parent at once, not at its next tick */
      if(atomic_load_explicit(&faulted,memory_order_relaxed))wake();
      if(guard&1)port_tell(&port->guarded,guard);
      /* a link that serves no lane ends the plan it served (no verbs call: link_close) */
      if(!served)link_close(port,0);
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
    /* the plan's generation read before the pass, so a plan written during it ends the wait below at once (the plan
       call wakes it, region-shared, after its last store: mesh-collective.c mesh_plan) */
    const uint64_t plan=atomic_load_explicit(&M->plan,memory_order_acquire);
    if(port_pass(port,&told))continue;
    os_sync_wait_on_address_with_timeout(&M->plan,plan,sizeof M->plan,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                         OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
  }
  link_close(port,0);
  return NULL;
}
/* tag pre-return-20261103 rdma/mesh-link.c:425-429 guard_wake (60b00774), verbatim but not static, and the plan's word
   woken too (region-shared: the progress thread's third wait, port_progress), so a raise holds at once */
/* design/transport.md B13: the progress thread parked on the port's guard looks again */
void guard_wake(struct port *port){
  os_sync_wake_by_address_all(&port->guard,sizeof port->guard,OS_SYNC_WAKE_BY_ADDRESS_NONE);
  os_sync_wake_by_address_all(&port->link.served,sizeof port->link.served,OS_SYNC_WAKE_BY_ADDRESS_NONE);
  if(M)os_sync_wake_by_address_all(&M->plan,sizeof M->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* tag pre-return-20261103 rdma/mesh-link.c:430-442 port_threads (3a82ede9), verbatim but not static */
/* design/transport.md: the port's progress thread started, once; 0, or pthread_create's error */
int port_threads(struct port *port){
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
/* tag pre-return-20261103 rdma/mesh-link.c:443-456 port_wait (e3c4c986), verbatim but its deadline: main's mesh.h has
   no struct mesh_deadline, so the tag's mesh_deadline, mesh_deadline_left and mesh_expired are written out (the same
   MESH_PLATFORM_NS from the wait's start, the same end: the wait named on the log and the child's exit, status 1) */
/* design/transport.md: the raiser of a settle or a guard parked, never spinning, until the port's progress thread,
   where started, has its word (`told`: the epoch it has seen, or the guard it holds at) at `value`, woken by the
   thread's tell (port_tell), its guard's waiters woken as it begins and at each waking, within its deadline,
   MESH_PLATFORM_NS from its start: a thread that has not told by then is in a call that has not returned, and the
   port's child, the waiting process, ends at once naming the wait (design/eager-execution.md, assumption A4), its exit the
   kernel's release of its port's objects and its parent's loss of the port (child_reaped) */
static void port_wait(struct port *port,_Atomic uint64_t *told,uint64_t value){
  const uint64_t until=clock_ns()+MESH_PLATFORM_NS;
  for(uint64_t seen;port->threads && (seen=atomic_load_explicit(told,memory_order_acquire))<value;){
    const uint64_t at=clock_ns();
    if(at>=until){
      mesh_say("bridge","mesh-flow child: its port's progress thread holding a settle or the guard (port_wait): past its deadline "
               "of %.3f s; the program ends (design/eager-execution.md, assumption A4)",(double)MESH_PLATFORM_NS/1e9);
      _exit(1);
    }
    guard_wake(port);
    os_sync_wait_on_address_with_timeout(told,seen,sizeof *told,OS_SYNC_WAIT_ON_ADDRESS_NONE,OS_CLOCK_MACH_ABSOLUTE_TIME,until-at);
  }
}
/* tag pre-return-20261103 rdma/mesh-link.c:457-473 guard_lower and guard_raise (306b22be), verbatim but not static */
/* design/RDMA-KERNEL-RECOVERY.md "The stripe writes of October 1": the guard of a port's device, on the port's queue,
   around a change of its verbs objects (queue pairs and completion queues made, connected or destroyed): raised, it
   returns once each started progress thread holds (it told the guard's value before its next pass, and passes no more
   while the guard stands), so no queue pair of the device is posted on while another is made, changed or destroyed
   (Apple's provider: a post on a long-lived queue pair across another's teardown stores to the device's shared doorbell
   mapping after it moved, tbt_post_recv); lowered, the threads parked on it woken and going on
   (design/transport.md B13).  It stands for every change once the port's threads run, the link lanes'
   long-lived queue pairs being posted on while the link is up.  Raised: 1; 0 where no progress thread runs (a thread
   that does not hold within its deadline ends the child: port_wait) */
void guard_lower(struct port *port,int raised){
  if(raised>0){atomic_fetch_add_explicit(&port->guard,1,memory_order_release);guard_wake(port);}
}
int guard_raise(struct port *port){
  if(!port->threads)return 0;
  port_wait(port,&port->guarded,atomic_fetch_add_explicit(&port->guard,1,memory_order_acq_rel)+1);
  return 1;
}

/* this node's landed words of the plan written not landed cancelled (mesh.h: UINT64_MAX, where a word reads 0), every
   execution's, of its receives from the node of map index `from` (UINT32_MAX: every receive) */
static void plan_cancel(uint32_t from){
  const uint64_t hops=atomic_load_explicit(&M->hops,memory_order_acquire);
  if(!hops)return;
  uint32_t executions,depth;uint64_t n;
  plan_shape(hops,&executions,&depth,&n);
  const uint64_t stride=M->hop_stride;
  const struct mesh_hop *list=mesh_hops(M);
  for(uint64_t j=0;j<n;j++){
    if(list[j].direction!=MESH_RECEIVE || (from!=UINT32_MAX && list[j].node!=from))continue;
    for(uint64_t t=0;t<executions;t++){
      uint64_t idle=0;
      atomic_compare_exchange_strong_explicit(region_word(list[j].word+t*stride),&idle,UINT64_MAX,memory_order_release,memory_order_relaxed);
    }
  }
}
/* tag pre-return-20261103 rdma/mesh-link.c:619-654 streams_open (11807bec), reduced to what it is under the page lists:
   the tag's streams made new and live for a new pairing; here the port's serving opened, as its lanes connect and
   before any is served (mesh-child.c child_jobs): no plan loaded on the new pairing (`prepared` 0: its progress thread
   loads the plan written as the lanes are served, link_run), and the link's bandwidth told in the port's table, as
   d6dceac link_run told it at its pairing.  `node` is the port's peer, which the child already holds (`port->peer`,
   CHILD_CONNECT): the tag stored it in each stream */
void streams_open(struct port *port,uint32_t node,uint64_t bandwidth){
  (void)node;
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  __atomic_store_n(&link->bandwidth,bandwidth,__ATOMIC_RELAXED);
  atomic_store_explicit(&link->port.prepared,0,memory_order_release);
}
/* tag pre-return-20261103 rdma/mesh-link.c:699-711 streams_cancel (11807bec), reduced to what it is under the page
   lists: the tag's streams of a port live no more and their words ~0; here, as its link is lost or the bridge stops
   (mesh-ports.c port_lost, mesh-children.c child_reaped, the stop), this node's landed words not landed cancelled for
   the receives from the port's peer (its index in the map; none where the map holds no record of it), every one where
   `port` is NULL, and the port's (every port's) plan no longer loaded (`prepared` 0) where no child of it lives: a
   live child's requests may still name the plan's pages, so its `prepared` is its own to clear (link_close) or its
   reap's (child_reaped, after `port->child` is 0).  A receive the port's child lands after it stores its position over
   the cancellation: the page's bytes did land */
void streams_cancel(const struct port *port){
  if(!port){
    plan_cancel(UINT32_MAX);
    for(uint32_t p=0;p<port_count;p++)
      if(!ports[p].child)atomic_store_explicit(&mesh_links(M)[ports[p].index].port.prepared,0,memory_order_release);
    return;
  }
  struct mesh_record *records=NULL;uint64_t count=0,serial=0;
  if(port->peer && !map_read(&records,&count,&serial)){
    const uint32_t from=map_index(records,count,port->peer);
    free(records);
    if(from!=UINT32_MAX)plan_cancel(from);
  } else if(port->peer)plan_cancel(UINT32_MAX);
  if(!port->child)atomic_store_explicit(&mesh_links(M)[port->index].port.prepared,0,memory_order_release);
}
/* tag pre-return-20261103 rdma/mesh-link.c:712-722 streams_let_go (60b00774), reduced to what it is under the page
   lists: the tag let a gone client's streams go; here, on the bridge's turn over its clients (mesh-records.c
   clients_turn), where no client holds the region (hdr `client` 0: its holder retired, mesh-dataflow.c mesh_retire) and
   a plan stays written, this node's landed words of it not landed are cancelled, so no relay of this node forwards a
   page of a plan whose client is gone */
void streams_let_go(void){
  if(atomic_load_explicit(&M->client,memory_order_acquire))return;
  plan_cancel(UINT32_MAX);
}
