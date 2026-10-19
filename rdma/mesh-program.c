#include "mesh-bridge.h"

static void busy_tell(void){
  const uint64_t busy=flow_count;
  if(busy)atomic_store_explicit(&M->idle_since,0,memory_order_relaxed);
  else if(!atomic_load_explicit(&M->idle_since,memory_order_relaxed))atomic_store_explicit(&M->idle_since,clock_gettime_nsec_np(CLOCK_REALTIME),memory_order_relaxed);
  atomic_store_explicit(&M->busy,busy,memory_order_release);
}
static uint32_t map_page,map_pages,map_room;
/* design/bridge-and-ipc.md#records: the bytes the kernel can give without pressure (kern.memorystatus_level percent of
   the node's memory), read now */
static uint64_t available_now(void){
  int level=0;size_t size=sizeof level;
  return sysctlbyname("kern.memorystatus_level",&level,&size,NULL,0) || level<0?0:(uint64_t)level*(memory/100);
}
/* design/bridge-and-ipc.md#records: what this node's record carries of its health (mesh-plan.h struct mesh_health), its
   boot read once and its available memory at most MESH_HEALTH_NS apart (health_turn) */
static struct mesh_health health;
static uint64_t health_sent;
void sysctl_text(const char *name,char *out,size_t room){
  size_t size=room;
  if(sysctlbyname(name,out,&size,NULL,0))out[0]=0;
  out[room-1]=0;
}
void health_init(void){
  struct timeval booted;size_t size=sizeof booted;
  if(!sysctlbyname("kern.boottime",&booted,&size,NULL,0))health.booted=(uint64_t)booted.tv_sec*1000000000u+(uint64_t)booted.tv_usec*1000u;
  health.available=available_now();
}

static uint32_t known_count;
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  struct known *at=NULL;
  if(known_count<MESH_KNOWN)at=known+known_count++;
  else for(uint32_t i=0;i<known_count;i++)if(!record_held(known[i].node) && (!at || known[i].heard<at->heard))at=known+i;
  if(at)*at=(struct known){.node=node};
  return at;
}
/* design/bridge-and-ipc.md#flows: a flow's program toward its destination: its slot's program, and its lists of queue k
   each way */
static struct mesh_tx *flow_tx(const struct flow *f){ return &mesh_client_at(M,f->owner)->tx[f->destination]; }
static uint32_t flow_length(const struct flow *f,int d,uint32_t k){ return atomic_load(mesh_order_length(M,f->owner,f->destination*MESH_QUEUES+k,d)); }
static struct mesh_transfer *flow_transfers(const struct flow *f,int d,uint32_t k){ return mesh_transfers(M,f->owner,f->destination*MESH_QUEUES+k,d); }
/* design/bridge-and-ipc.md#clients: a status told, and its client's start woken */
static void status_prepared(struct mesh_port_info *status,uint64_t owner){
  atomic_store_explicit(&status->prepared,owner,memory_order_release);
  os_sync_wake_by_address_all(&status->prepared,sizeof status->prepared,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
static void status_failed(struct mesh_port_info *status,uint64_t owner,int64_t code,uint32_t domain,uint32_t where){
  atomic_store_explicit(&status->code,code,memory_order_relaxed);atomic_store_explicit(&status->domain,domain,memory_order_relaxed);
  atomic_store_explicit(&status->failed,where,memory_order_relaxed);atomic_store_explicit(&status->phase,MESH_STOPPED,memory_order_relaxed);
  status_prepared(status,owner);
}

/* The ring slot a transfer's invocation t addresses: (t - begin) mod its depth, at its slot pages. */
static uint32_t transfer_row(struct hdr *m,uint32_t fallback,const struct mesh_transfer *transfer,uint32_t row,uint32_t invocation,uint32_t chunk){
  uint32_t depth=transfer->depth?transfer->depth:fallback;
  if(!depth)depth=1;
  return row+(uint32_t)((size_t)((invocation-transfer->begin)%depth)*transfer->invocation_pages/m->block)+chunk;
}

/* design/prepared-machine.md#M09: a chunk's span in the window its port's device registered when it came up; 0, or
   EINVAL for a page outside the window */
static int chunk_span(struct flow *link,uint32_t address,uint64_t remaining,struct ibv_sge *span){
  struct hdr *m=link->M;
  uint64_t offset=atomic_load_explicit(&mesh_page(m)[address].address,memory_order_relaxed)-m->data_off;
  if(offset>=wire.length)return EINVAL;
  *span=wire_span(&link->port->device,(uint64_t)m->block*m->pgsz,offset,remaining);
  return 0;
}

/* design/prepared-machine.md#M07, #M08, #M12: the word a landed chunk of invocation u of the transfer's slot `index` (0
   where each slot has its own rows) stores: its slot's completion word (mesh_host_inputs, the Metal transport), where
   the row's delivery names a word among the receiving program's own toward the flow's node (`words`, its cancellation
   range, [offset, offset + 8 count): a delivery another program wrote on the row, which outlives that program, is
   none of this one's); or, on a row the receiving program (`owner`) also publishes (a member that sends on what it
   receives: a relay rank's rows), that slot's publication's SEND cell once the record's last chunk landed, so the
   landing releases it to the next hop (the index hand-off, *release set); else the record's own (`sink`), which nothing
   reads.  NULL where the word is not in the window (#M09) */
static _Atomic uint64_t *receive_input(struct hdr *m,uint64_t owner,const struct mesh_cancel_range *words,uint32_t row,uint32_t index,
                                       uint32_t chunk,uint32_t chunks,uint32_t u,_Atomic uint64_t *sink,uint32_t *release){
  const struct mesh_delivery *delivery=mesh_delivery_at(m,row+chunk);
  uint64_t at=0;
  const uint64_t word=delivery->device_input+delivery->device_stride*u+sizeof(uint64_t)*(uint64_t)index*chunks;
  *release=0;
  if(delivery->device_input && words && word>=words->offset && (word-words->offset)/sizeof(uint64_t)<words->count)at=word;
  else if(chunk+1==chunks){
    uint32_t sends;
    const struct mesh_target *target=mesh_row_targets(m,owner,row,&sends);
    if(index<sends){at=target[index].cells+sizeof(struct mesh_send)*(uint64_t)u;*release=1;}
  }
  if(!at)return sink;
  return at>=m->data_off && at<=m->length-sizeof(uint64_t)?(void *)((char *)m+at):NULL;
}

/* design/bridge-and-ipc.md#cables: whether the flow carries the program's invocation t, and how many of a transfer's
   active invocations [begin, begin + active) it carries */
static int stripe_of(const struct flow *link,uint32_t t){ return t%link->stripes==link->stripe; }
static uint32_t stripe_count(const struct flow *link,uint32_t begin,uint32_t active){
  uint32_t first=(link->stripe+link->stripes-begin%link->stripes)%link->stripes;
  return first<active?(active-first+link->stripes-1)/link->stripes:0;
}

/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M04, #M06, #M29 */
/* The SEND side: every SEND cell of the flow's invocations (#cables), its chain of chunk requests over its invocation's
   ring slot and its stream successor, a stream a queue of the program, posted on that queue's lane (the cell's queue
   pair and `queue` its lane, whose gate counts its frames: struct link `gate`; the cells' layout is
   mesh_transfers_prepare's).  A stream's cells run invocation by invocation, the publications of each in binding order,
   as the peer's ring of the same queue posts its receives on the lane's other end: TN3205's credit flow control holds a
   SEND until its matching receive is posted (metal-microbench docs/kernels.md Transport (git 7300d1e^)).  0, or the
   error (EINVAL: a transfer of more than the lane's one slot). */
static int sends_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->owner);
  struct mesh_tx *tx=flow_tx(link);
  const struct link *lanes=&link->port->link;
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire);
  const int cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0;
  const uint32_t queues=link->queues;
  size_t requests=0;
  for(uint32_t k=0;k<queues;k++){
    struct mesh_transfer *out=flow_transfers(link,MESH_SEND,k);
    for(uint32_t i=0;i<flow_length(link,MESH_SEND,k);i++){
      if(out[i].count>1)return EINVAL;
      const size_t own=(size_t)out[i].count*stripe_count(link,out[i].begin,mesh_transfer_active(out+i,invocations));
      requests+=(size_t)(mesh_row_chunks(m,out[i].local_row,out[i].bytes)-1)*own;
    }
  }
  link->publications=(void *)((char *)m+tx->cells);
  /* design/prepared-machine.md#M29 */
  link->requests=aligned_alloc(128,(requests+1)*sizeof *link->requests);
  link->streams=calloc(queues,sizeof *link->streams);
  struct mesh_send **last=calloc(queues,sizeof *last),**terminal=calloc(queues,sizeof *terminal);
  if(!link->requests||!link->streams||!last||!terminal){free(last);free(terminal);return ENOMEM;}
  link->request_count=requests;
  size_t next=0;int error=0;
  for(uint32_t k=0;k<queues && !error;k++){
    const uint32_t count=flow_length(link,MESH_SEND,k),lane=link->lane[k];
    struct mesh_transfer *out=flow_transfers(link,MESH_SEND,k);
    struct ibv_qp *pair=lanes->provider.queues[lane].pair;
    /* each cell's chain over its invocation's ring slot */
    for(uint32_t i=0;i<count && !error;i++){
      if(!out[i].count)continue;
      const uint32_t active=mesh_transfer_active(out+i,invocations),row=out[i].local_row,chunks=mesh_row_chunks(m,row,out[i].bytes);
      struct mesh_send *cells=link->publications+out[i].first;
      for(uint32_t u=0;u<active && !error;u++){
        if(!stripe_of(link,out[i].begin+u))continue;
        struct mesh_send *cell=cells+u;
        cell->pair=(uintptr_t)pair;cell->queue=lane;cell->chunks=chunks;
        atomic_store_explicit(&cell->port,link->port->index+1,memory_order_release);
        uint64_t remaining=out[i].bytes;
        for(uint32_t c=0;c<chunks && !error;c++){
          struct ibv_sge span;
          if((error=chunk_span(link,transfer_row(m,fallback,out+i,row,out[i].begin+u,c),remaining,&span)))break;
          remaining-=span.length;
          struct ibv_sge *entry=c?&link->requests[next+c-1].span:&cell->span;
          struct ibv_send_wr *request=c?&link->requests[next+c-1].request:&cell->request;
          *entry=span;
          /* the first request's wr_id stays the producer's nonzero argument (mesh_transfers_prepare),
             which a host producer reads at publication (mesh_host_publish) */
          *request=(struct ibv_send_wr){.wr_id=c?0:1,.next=c+1==chunks?NULL:&link->requests[next+c].request,
            .sg_list=entry,.num_sge=1,.opcode=IBV_WR_SEND};
        }
        next+=chunks-1;
      }
      terminal[k]=cells+active;
    }
    /* the stream's cells in invocation order, its publications in binding order within one */
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && stripe_of(link,t);i++){
      const uint32_t active=mesh_transfer_active(out+i,invocations);
      if(!out[i].count || t<out[i].begin || t-out[i].begin>=active)continue;
      struct mesh_send *cell=link->publications+out[i].first+(t-out[i].begin);
      if(last[k])last[k]->successor=(uintptr_t)cell;
      else link->streams[k].cell=cell;
      last[k]=cell;
    }
  }
  /* design/prepared-machine.md#M30: a cyclic program's last cell is followed by its first */
  for(uint32_t k=0;k<queues && !error;k++)if(last[k])last[k]->successor=cyclic?(uintptr_t)link->streams[k].cell:(uintptr_t)terminal[k];
  for(uint32_t k=0;k<queues;k++){link->streams[k].first=link->streams[k].cell;link->streams[k].cycle=0;}
  link->stream_count=0;
  for(uint32_t k=0;k<queues;k++)if(link->streams[k].cell)link->streams[link->stream_count++]=link->streams[k];
  free(last);free(terminal);
  return error;
}

/* design/prepared-machine.md#M12: a client's cancellation at its offset `at` (0: none), only where its header, its
   destination `d`'s range and that range's words all lie in the window, as each word the bridge stores does (#M09);
   else NULL */
static struct mesh_cancellation *cancel_at(uint64_t at,uint32_t d){
  const uint64_t range=sizeof(struct mesh_cancellation)+sizeof(struct mesh_cancel_range)*((uint64_t)d+1);
  if(!at || at<M->data_off || range>M->length || at>M->length-range)return NULL;
  struct mesh_cancellation *cancel=(void *)((char *)M+at);
  const uint64_t offset=cancel->ranges[d].offset,count=cancel->ranges[d].count;
  return offset>=M->data_off && offset<=M->length && count<=(M->length-offset)/sizeof(uint64_t)?cancel:NULL;
}

/* design/prepared-machine.md#M08 */
/* The receive side: a ring a queue of the program, on that queue's lane, its records invocation by invocation (the
   flow's invocations, #cables), the transfers of each in binding order, a transfer's chunks in order, each storing its
   word (receive_input); each record posted once the record of its transfer `span` of the flow's own invocations before
   has landed (the ring's span over the program's invocations, divided among the stripes; a cyclic ring's first records
   wait on its last of the cycle before), so a ring's posted frames stay within its lane's queue pair and its share of
   the link's completion queues.  Its first records are posted by the port's receive thread once the flow is listed
   (flow_arm), never here.  0, or the error (EINVAL: a transfer of more than the lane's one slot, a word outside the
   window). */
static int receives_configure(struct flow *link){
  struct hdr *m=link->M;struct mesh_client *client=mesh_client_at(m,link->owner);
  const struct link *lanes=&link->port->link;
  /* design/prepared-machine.md#M09, #M10: the program's targets lie in the window, as each word it stores does */
  if(client->target_count && (client->targets<m->data_off || client->targets>m->length ||
     client->target_count>(m->length-client->targets)/sizeof(struct mesh_target)))return EINVAL;
  struct mesh_tx *tx=flow_tx(link);
  /* design/prepared-machine.md#M12: the program's own words toward the flow's node, where it has any */
  const struct mesh_cancellation *cancel=cancel_at(tx->cancel,link->destination);
  const struct mesh_cancel_range *words=cancel?cancel->ranges+link->destination:NULL;
  uint32_t invocations=tx->invocations?tx->invocations:1,fallback=atomic_load_explicit(&client->depth,memory_order_acquire),depth=fallback;
  if(!depth||depth>invocations)depth=invocations;
  link->cyclic=atomic_load_explicit(&client->cyclic,memory_order_acquire)!=0;link->invocations=invocations;
  const uint32_t queues=link->queues;
  size_t records=0;
  for(uint32_t k=0;k<queues;k++){
    struct mesh_transfer *in=flow_transfers(link,MESH_RECEIVE,k);
    for(uint32_t i=0;i<flow_length(link,MESH_RECEIVE,k);i++){
      if(in[i].count>1)return EINVAL;
      records+=(uint64_t)mesh_row_chunks(m,in[i].local_row,in[i].bytes)*in[i].count*stripe_count(link,in[i].begin,mesh_transfer_active(in+i,invocations));
    }
  }
  link->receive=aligned_alloc(128,(records+1)*sizeof *link->receive);
  link->rings=calloc(queues,sizeof *link->rings);
  if(!link->receive||!link->rings)return ENOMEM;
  link->receive_count=records;
  /* a record is one frame at least, so a lane's share of frames also bounds the completions it can have pending in the
     receive completion queue the link's lanes share */
  for(uint32_t q=0;q<queues;q++)link->rings[q]=(struct receive_ring){.completed=-1,.span=UINT32_MAX,
    .capacity=MIN(lanes->provider.queues[link->lane[q]].receive_capacity,lanes->share>1?lanes->share:1)};
  size_t at=0;int error=0;
  for(uint32_t ring=0;ring<queues && !error;ring++){
    const uint32_t count=flow_length(link,MESH_RECEIVE,ring);
    struct mesh_transfer *in=count?flow_transfers(link,MESH_RECEIVE,ring):NULL;
    struct ibv_qp *pair=lanes->provider.queues[link->lane[ring]].pair;
    struct receive_ring *r=link->rings+ring;
    r->first=link->receive+at;
    /* firsts: the ring index of each transfer's own records, own invocation by own invocation (base[i] the first's) */
    size_t *base=calloc((size_t)count+1,sizeof *base),*firsts=NULL;
    for(uint32_t i=0;base&&i<count;i++)
      base[i+1]=base[i]+(in[i].count?stripe_count(link,in[i].begin,mesh_transfer_active(in+i,invocations)):0);
    if(base)firsts=malloc((base[count]+1)*sizeof *firsts);
    if(!base||!firsts){free(base);free(firsts);return ENOMEM;}
    for(uint32_t i=0;i<count;i++){
      if(!in[i].count)continue;
      uint32_t active=mesh_transfer_active(in+i,invocations),own=in[i].depth?in[i].depth:depth;
      /* receive u+span lands in slot (u+span) mod depth while the local reader may still be at u:
         the ring's span is held a slot short of the ring; a transfer whose every invocation has its
         own slot bounds nothing */
      if(own<active)r->span=MIN(r->span,own>1?own-1:1);
    }
    for(uint32_t t=0;t<invocations && !error;t++)for(uint32_t i=0;i<count && !error && stripe_of(link,t);i++){
      uint32_t active=mesh_transfer_active(in+i,invocations);
      if(!in[i].count || t<in[i].begin || t-in[i].begin>=active)continue;
      uint32_t u=t-in[i].begin,row=in[i].local_row,chunks=mesh_row_chunks(m,row,in[i].bytes);
      const uint32_t first=(link->stripe+link->stripes-in[i].begin%link->stripes)%link->stripes,mine=(u-first)/link->stripes;
      uint64_t remaining=in[i].bytes;
      for(uint32_t c=0;c<chunks;c++){
        struct ibv_sge span;
        if((error=chunk_span(link,transfer_row(m,fallback,in+i,row,t,c),remaining,&span)))break;
        remaining-=span.length;
        struct prepared_receive *record=link->receive+at++;
        if(!c)firsts[base[i]+mine]=(size_t)(record-r->first);
        uint32_t release;
        *record=(struct prepared_receive){.request={.wr_id=(uintptr_t)record,.sg_list=&record->span,.num_sge=1},
          .span=span,.input=receive_input(m,link->owner,words,row,0,c,chunks,u,&record->sink,&release),
          .pair=pair,.argument=1,.frames=(span.length+4095)/4096,.invocation=t,.queue=ring,.link=link};
        record->release=release;
        if(!record->input){error=EINVAL;break;}
      }
    }
    r->count=(size_t)(link->receive+at-r->first);
    /* design/prepared-machine.md#M30 */
    if(link->cyclic)r->span=MIN(r->span,invocations>1?invocations-1:1);
    /* #cables: the span in the flow's own invocations, a stripe's share of the program's (one at least) */
    const uint32_t span=r->span==UINT32_MAX?UINT32_MAX:MAX(r->span/link->stripes,1u);
    for(uint32_t i=0;i<count && !error;i++){
      uint32_t own=(uint32_t)(base[i+1]-base[i]);
      if(!own)continue;
      uint32_t chunks=mesh_row_chunks(m,in[i].local_row,in[i].bytes);
      for(uint32_t j=0;j<own;j++){
        size_t here=firsts[base[i]+j],behind=0;
        if(span!=UINT32_MAX && j>=span)behind=here-firsts[base[i]+j-span];
        else if(span!=UINT32_MAX && link->cyclic && j+own>=span)behind=here+r->count-firsts[base[i]+j+own-span];
        else if(span!=UINT32_MAX && link->cyclic)behind=r->count;
        for(uint32_t c=0;c<chunks;c++)r->first[here+c].behind=(uint32_t)behind;
      }
    }
    r->span=span;
    free(base);free(firsts);
  }
  if(at)link->receive[at]=link->receive[at-1];
  return error;
}

/* design/bridge-and-ipc.md#flows: a flow's one job, on its port's queue: its receive records and its SEND chains
   prepared over its lanes, then the flow listed in the port's table (its rings' first records posted by the receive
   thread's first look at it, flow_arm).  No verbs call is made: its lanes' queue pairs are the link's, made at its
   pairing (design/bridge-and-ipc.md#lanes) */
static void flow_job(struct flow *f){
  f->job=1;atomic_store_explicit(&f->done,0,memory_order_relaxed);
  dispatch_async(f->port->work,^{
    int error=receives_configure(f);
    if(!error)error=sends_configure(f);
    if(!error)error=port_list(f);
    f->result=error;
    atomic_store_explicit(&f->done,1,memory_order_release);
    wake();
  });
}
/* design/algorithm-sources.md#meshresult: a flow stopped: where it failed, its program's pending words toward the far
   end cancelled (M12) and its status the failure naming the node `where`, stored here on the event loop as
   programs_stop stores them (no queue waited on: design/recovery.md#five-seconds B4's order, ~0 and the status before
   B6, never behind a call held on the port's queue); then on its port's queue, after its job, its lanes let go
   (lanes_release: no completion of them is taken as its own after it, and a lane it left with positions outstanding
   stays DIRTY, moved to the error state unless the stop is its link's loss, `lost`); once no progress thread reads it
   (`listed` 0: a flow a thread may still read is kept whole, mesh-link.c lanes_release), the census printed (each of
   its queues' lane: the SEND requests and frames posted and retired through its gate, and its ring's records posted and
   landed and their bytes); `stopped` 2 where a dirty lane's requests may still land
   in its rows (design/recovery.md#five-seconds B7), else 1; its storage freed with it (flow_free) */
static void flow_stop(struct flow *f,int64_t code,uint32_t domain,uint32_t where,int lost){
  const uint64_t owner=f->owner;const uint32_t destination=f->destination;struct mesh_cancellation *cancel=f->cancel;
  f->stopping=1;atomic_store_explicit(&f->stopped,0,memory_order_relaxed);
  if(code && owned(owner)){
    if(cancel)mesh_cancel(M,cancel,destination);
    status_failed(&mesh_client_at(M,owner)->status[destination],owner,code,domain,where);
  }
  dispatch_async(f->port->work,^{
    const uint32_t dirty=lanes_release(f,lost);
    const struct link *l=&f->port->link;
    for(uint32_t q=0;f->rings && !f->listed && q<f->queues;q++){
      const struct send_gate *g=l->gate+f->lane[q];
      say("link %u census queue=%u requests=%llu frames=%llu retired=%llu send_completions=%llu records=%zu posted=%zu landed=%llu bytes=%llu "
          "lane=%u%s\n",f->port->index,q,(unsigned long long)g->requests,(unsigned long long)g->posted,(unsigned long long)g->retired,
          (unsigned long long)g->completions,f->rings[q].count,f->rings[q].posted,(unsigned long long)f->rings[q].landed,
          (unsigned long long)f->rings[q].bytes,f->lane[q],f->held>>f->lane[q]&1?" dirty, its rows held":dirty>>f->lane[q]&1?" dirty, flushed":"");
    }
    atomic_store_explicit(&f->stopped,f->held?2:1,memory_order_release);
    wake();
  });
}
/* a flow freed with its storage, its index in its port's table given back */
static void flow_free(struct flow *f){
  for(struct flow **at=&flows;*at;at=&(*at)->next)if(*at==f){*at=f->next;break;}
  f->port->flows--;
  f->port->reserved[f->slot/64]&=~(UINT64_C(1)<<(f->slot%64));
  free(f->receive);free(f->streams);free(f->rings);free(f->requests);free(f);
  flow_count--;busy_tell();
}
/* design/algorithm-sources.md#meshresult: a program on a port failed as a value, its status the failure naming the node
   `where`, and every flow of the same client program toward the node (its other cables) with it; a flow on `lost`, the
   port whose link was lost (port_lost's; NULL none), stops with no verbs call.  No message tells the peer: its programs
   end by what ends them at that end (the link's loss there, its caller's revoke as its words,
   design/bridge-and-ipc.md#protocol) */
void flow_fail(struct flow *f,int64_t code,uint32_t domain,uint32_t where,const struct port *lost){
  if(f->ended)return;
  f->ended=1;
  say("link %u: program %016llx toward node %u (stripe %u of %u) failed at node %u: code %lld domain %u\n",f->port->index,
    (unsigned long long)f->key,f->far,f->stripe,f->stripes,where,(long long)code,domain);
  flow_stop(f,code,domain,where,f->port==lost);
  for(struct flow *g=flows;g;g=g->next)if(g->owner==f->owner && g->destination==f->destination)flow_fail(g,code,domain,where,lost);
}
/* design/algorithm-sources.md#meshresult: a client's program toward its destination `d` refused or failed as a value
   naming the node `where`: its status the failure, then ~0 into its cancel words (M12, where they lie in the window:
   cancel_at), so a client reading a landing of it reads the status the ~0 names */
static void program_refused(struct mesh_client *client,uint64_t owner,uint32_t d,int64_t code,uint32_t where){
  status_failed(&client->status[d],owner,code,4,where);
  struct mesh_cancellation *cancel=cancel_at(client->tx[d].cancel,d);
  if(cancel)mesh_cancel(M,cancel,d);
}
/* a client's program toward its destination `d` failed as a value naming the node `where`: its flows failed, and the
   program refused (program_refused) */
static void program_fail(uint64_t owner,uint32_t d,int64_t code,uint32_t where){
  for(struct flow *g=flows;g;g=g->next)if(g->owner==owner && g->destination==d)flow_fail(g,code,4,where,NULL);
  if(owned(owner))program_refused(mesh_client_at(M,owner),owner,d,code,where);
}
/* a program that ended, its client gone or its slot given back: its flows stopped, nothing told */
static void flow_end(struct flow *f){
  if(f->ended)return;
  f->ended=1;
  flow_stop(f,0,0,0,0);
}
/* design/bridge-and-ipc.md#clients: a client program's status once every flow of it toward the node runs */
static void program_paired(struct flow *f){
  for(struct flow *g=flows;g;g=g->next)
    if(g->owner==f->owner && g->destination==f->destination && (!g->running || g->job))return;
  if(!owned(f->owner))return;
  struct mesh_port_info *status=&mesh_client_at(M,f->owner)->status[f->destination];
  atomic_store_explicit(&status->code,0,memory_order_relaxed);atomic_store_explicit(&status->failed,0,memory_order_relaxed);
  atomic_store_explicit(&status->phase,MESH_PAIRED,memory_order_relaxed);
  status_prepared(status,f->owner);
}
/* a flow's job's end taken, on the event loop: a failure the flow's, else it runs */
static void flow_done(struct flow *f){
  f->job=0;
  if(f->ended)return;
  if(f->result){flow_fail(f,f->result,1,M->node,NULL);return;}
  f->running=1;
  program_paired(f);
}
/* design/bridge-and-ipc.md#flows: a flow made on a port toward `far` for the client's program `owner`'s destination `d`
   (its execution `key`), its `queues` queues on the lanes `lane` (held by it from now, their lease words DIRTY:
   lanes_take, lanes_leased having found them `owner`'s), the program's invocations t with t mod stripes = stripe
   (#cables), an index of the port's table its own (MESH_FLOWS a port); NULL where the table is full (*refused E2BIG)
   or a lease was let go meanwhile (ENOLCK) */
static struct flow *flow_make(struct port *port,uint64_t owner,uint32_t d,uint64_t key,uint32_t far,uint32_t queues,const uint32_t *lane,
                              uint32_t stripe,uint32_t stripes,int64_t *refused){
  uint32_t slot=0;
  while(slot<MESH_FLOWS && port->reserved[slot/64]>>(slot%64)&1)slot++;
  *refused=E2BIG;
  if(slot==MESH_FLOWS)return NULL;
  struct flow *f=calloc(1,sizeof *f);
  if(!f)return NULL;
  if(!lanes_take(port,owner,lane,queues)){free(f);*refused=ENOLCK;return NULL;}
  const struct mesh_queue *queue=port->link.provider.queues+lane[0];
  const uint64_t cancel=mesh_client_at(M,owner)->tx[d].cancel;
  *f=(struct flow){.next=flows,.far=far,.queues=queues,.stripe=stripe,.stripes=stripes,.destination=d,.lane={lane[0],queues>1?lane[1]:0},
    .slot=slot,.generation=++port->generation[slot],.lanes=UINT32_C(1)<<lane[0]|(queues>1?UINT32_C(1)<<lane[1]:0),.key=key,.owner=owner,
    .M=M,.port=port,.post_send=queue->send,.post_receive=queue->receive,.cancel=cancel_at(cancel,d)};
  port->reserved[slot/64]|=UINT64_C(1)<<(slot%64);
  if(slot>=atomic_load(&port->high))atomic_store(&port->high,slot+1);
  for(uint32_t q=0;q<queues;q++)atomic_store_explicit(&port->link.holder[lane[q]],f,memory_order_release);
  flows=f;port->flows++;flow_count++;busy_tell();
  return f;
}

/* the queues of a client's program toward its destination `d`: its highest queue with a list, plus one */
static uint32_t program_queues(uint64_t owner,uint32_t d){
  uint32_t queues=0;
  for(uint32_t q=0;q<MESH_QUEUES;q++)for(int w=0;w<2;w++)if(atomic_load(mesh_order_length(M,owner,d*MESH_QUEUES+q,w)))queues=q+1;
  return queues;
}
/* design/bridge-and-ipc.md#ports: the ports paired with `node`, in the cable order both ends read alike, by the index
   the lower node gave each cable */
static uint32_t cable_rank(const struct port *port){ return M->node<port->peer?port->index:port->peer_port; }
static uint32_t cables(uint32_t node,struct port **out){
  uint32_t n=0;
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted && ports[p].peer==node)out[n++]=ports+p;
  for(uint32_t i=1;i<n;i++)for(uint32_t j=i;j>0 && cable_rank(out[j])<cable_rank(out[j-1]);j--){struct port *t=out[j];out[j]=out[j-1];out[j-1]=t;}
  return n;
}
/* design/bridge-and-ipc.md#clients, #lanes, #cables: a started program's destination taken onto its lanes.  Its queues
   (at most two, of one slot each: its communicator's lane pair, queue k on the lane its slot names, mesh.h struct
   mesh_tx `lane`, which its client leased at the communicator's init, mesh.h struct hdr `lanes`) ride each cable to its
   node on which the client holds that lease, in the cable order both ends read alike, flow j of n carrying the
   program's invocations t with t mod n = j; no message is sent, the peer's program of the communicator riding the same
   lanes from the other end, each end posting its receives in its program's order and TN3205's credit holding a SEND
   until its receive is posted.  A program reaches only a node on a cable (a relay is a member rank of the communicator
   with no contribution: DESIGN.md decision 1).  Refused as a value (program_refused: its status, then ~0 into its
   cancel words) naming the node where it is this node (EINVAL) or no cable reaches it (EHOSTUNREACH); naming this node
   where its shape is not a lane pair's (EINVAL), no cable carries its lease or a lease was let go as its flow was made
   (ENOLCK), or a port's table is full (E2BIG) */
static void program_take(struct mesh_client *client,uint64_t owner,uint32_t d){
  const uint32_t node=client->tx[d].node,queues=program_queues(owner,d);
  const uint32_t lane[2]={client->tx[d].lane[0],client->tx[d].lane[1]};
  if(!node || node==M->node){program_refused(client,owner,d,EINVAL,node);return;}
  if(client->tx[d].slots!=1 || !queues || queues>2 || (queues==2 && lane[0]==lane[1])){
    program_refused(client,owner,d,EINVAL,M->node);return;
  }
  struct port *cable[MESH_PORTS];
  const uint32_t n=cables(node,cable);
  if(!n){program_refused(client,owner,d,EHOSTUNREACH,node);return;}
  uint32_t leased=0;
  for(uint32_t j=0;j<n;j++)if(lanes_leased(cable[j],owner,lane,queues))cable[leased++]=cable[j];
  if(!leased){program_refused(client,owner,d,ENOLCK,M->node);return;}
  atomic_store_explicit(&client->status[d].phase,MESH_PAIRING,memory_order_relaxed);
  for(uint32_t j=0;j<leased;j++){
    int64_t refused;
    struct flow *f=flow_make(cable[j],owner,d,client->key,node,queues,lane,j,leased,&refused);
    if(!f){program_fail(owner,d,refused,M->node);return;}
    atomic_fetch_or_explicit(&client->ports,UINT32_C(1)<<cable[j]->index,memory_order_release);
    flow_job(f);
  }
}

/* design/bridge-and-ipc.md#ports, #region: the map, a record a node: this node's own, rebuilt from its port table when
   that changes (its sequence one more), and every other node's newest heard; each change written between two
   notifications of the map serial (odd while it writes), the records in arena pages the bridge leases and leases again
   larger as the mesh grows; nodes no paired path reaches dropped */
static int map_grow(uint64_t records){
  if(records<=map_room)return 0;
  uint32_t pages=(uint32_t)((records*2*sizeof(struct mesh_record)+M->pgsz-1)/M->pgsz),page=mesh_arena_alloc(&bridge,pages,1);
  if(page==MESH_ABSENT)return ENOMEM;
  memcpy(mesh_at(M,page),mesh_records(M),(size_t)atomic_load(&M->map_nodes)*sizeof(struct mesh_record));
  atomic_store(&M->map_off,(uint64_t)(mesh_at(M,page)-(unsigned char *)M));
  if(map_pages)mesh_bits_clear(mesh_arena_bits(M),map_page,map_pages);
  map_page=page;map_pages=pages;map_room=(uint32_t)((uint64_t)pages*M->pgsz/sizeof(struct mesh_record));
  return 0;
}
/* RFC 2328 s14: the map with `record` put in (its node's held record replaced, else added) and every record no paired
   path from this node reaches left out, written only where it differs from the map held (so a record that changes
   nothing, or that the prune drops at once, moves no serial) */
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
  /* design/recovery.md#diagnosis: a node the prune drops is logged (one put now and never reached yet is not a loss) */
  for(uint64_t k=0;k<count;k++)if(!reached[k] && !(at==n && k==at)){
    struct known *gone=known_at(next[k].node);
    say("node %u (%s) unreachable, heard %.1f s before\n",next[k].node,next[k].name,gone && gone->heard?(double)(now-gone->heard)/1e9:-1.0);
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
/* design/recovery.md#diagnosis: a record whose links (each port's peer, phase and bandwidth), name, boot and facts are
   the held one's changes only its health, sequence and ports' counts (their codes, pairings and queue pairs), written
   under the region's health serial and never the map's: the map serial moves only where the mesh changes (a
   communicator's session reads the map again where it moved, and reopens only where its own routes or their links'
   costs changed: nccl-mesh.c map_digest, session_ensure), and neither a health sample nor a queue pair is a change of
   the mesh */
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  int same=a->node==b->node && a->ports==b->ports && a->boot==b->boot && !memcmp(a->name,b->name,sizeof a->name) &&
    a->window==b->window && a->memory==b->memory && !memcmp(a->model,b->model,sizeof a->model);
  for(uint32_t p=0;same && p<MESH_PORTS;p++)
    same=atomic_load(&a->port[p].peer)==atomic_load(&b->port[p].peer) && atomic_load(&a->port[p].phase)==atomic_load(&b->port[p].phase) &&
      atomic_load(&a->port[p].bandwidth)==atomic_load(&b->port[p].bandwidth);
  return same;
}
static void health_put(struct mesh_record *held,const struct mesh_record *record){
  mesh_notify(&M->health);
  held->sequence=record->sequence;held->health=record->health;
  memcpy(held->port,record->port,sizeof held->port);
  mesh_notify(&M->health);
}
struct mesh_record *record_held(uint32_t node){
  for(uint64_t i=0;i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node==node)return mesh_records(M)+i;
  return NULL;
}
void record_changed(void){
  struct mesh_record own={.node=M->node,.ports=port_count,.boot=M->boot,.sequence=++sequence,.window=wire.length,.memory=memory};
  memcpy(own.name,host,sizeof own.name);
  memcpy(own.model,model,sizeof own.model);
  memcpy(own.port,mesh_links(M),sizeof own.port);
  own.health=health;
  struct mesh_record *held=record_held(M->node);
  if(held && record_links_same(held,&own))health_put(held,&own);
  else map_put(&own);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
/* design/recovery.md#five-seconds: this node's record told again at its current sequence on every greeted port, each
   liveness turn (MESH_PAUSE_NS), so each control socket carries a message well within its peer's MESH_PLATFORM_NS
   (mesh-link.c port_silent); a receiver drops a record no newer than its newest (record_heard, RFC 2328 s13), its map and
   serials unmoved.  Told from a copy: a port lost meanwhile may write the map */
void record_retell(void){
  const struct mesh_record *held=record_held(M->node);
  if(!held)return;
  struct mesh_record own;memcpy(&own,held,sizeof own);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
void health_turn(void){
  if(now-health_sent<MESH_HEALTH_NS)return;
  health_sent=now;health.available=available_now();
  record_changed();
}
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard;
  if(bytes!=sizeof heard)return;
  memcpy(&heard,payload,sizeof heard);
  heard.name[sizeof heard.name-1]=0;
  if(!heard.node || heard.node==M->node || heard.ports>MESH_PORTS)return;
  heard.model[sizeof heard.model-1]=0;
  struct mesh_record *held=record_held(heard.node);
  struct known *k=known_at(heard.node);
  /* design/bridge-and-ipc.md#records, RFC 2328 s13: a record no newer than the newest heard of its node, the map holding
     it or not, is dropped and not sent on, so a record the map pruned never circulates (a node's later boot is always
     newer) */
  if(k && k->boot && (heard.boot<k->boot || (heard.boot==k->boot && heard.sequence<=k->sequence))){
    /* the newest known of a node the map dropped while no paired path reached it: put again where a path now does,
       and sent on only then (a pruned record that no path reaches stops here) */
    if(!held && heard.boot==k->boot && heard.sequence==k->sequence){
      map_put(&heard);
      for(uint32_t p=0;record_held(heard.node) && p<port_count;p++)
        if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
    }
    return;
  }
  if(held && (heard.boot<held->boot || (heard.boot==held->boot && heard.sequence<=held->sequence)))return;
  /* design/recovery.md#diagnosis: a node that joins or comes back is logged here */
  if(!held && k && k->gone)say("node %u (%s) back after %.1f s\n",heard.node,heard.name,(double)(now-k->gone)/1e9);
  else if(!held && k && !k->heard)say("node %u (%s) joined\n",heard.node,heard.name);
  if(k){k->heard=now;k->gone=0;k->boot=heard.boot;k->sequence=heard.sequence;}
  if(held && record_links_same(held,&heard))health_put(held,&heard);
  else map_put(&heard);
  for(uint32_t p=0;p<port_count;p++)if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
}
/* design/bridge-and-ipc.md#stop: the bridge stopping (`port` NULL), or a port concluded around a call out past
   MESH_PLATFORM_NS (design/recovery.md#five-seconds B12: its flows whose stops have not run, those waiting behind the
   call): every such program's words toward its destinations cancelled (~0) and its status the code, stored from the
   event loop's thread as plain memory stores, no queue, thread or verbs call waited on */
void programs_stop(int64_t code,const struct port *port){
  for(struct flow *f=flows;f;f=f->next){
    if(!owned(f->owner) || (port && (f->port!=port || atomic_load_explicit(&f->stopped,memory_order_acquire))))continue;
    if(f->cancel)mesh_cancel(M,f->cancel,f->destination);
    status_failed(&mesh_client_at(M,f->owner)->status[f->destination],f->owner,code,4,M->node);
  }
}
/* design/bridge-and-ipc.md#clients: every client slot read: a slot's process watched from the turn its slot is first
   seen (its exit retires every slot it holds, the event loop's EVFILT_PROC), one already gone retired at once, and on a
   liveness turn any client whose process exited retired; a program that ended (its slot given back or retired) has its
   flows stopped; a program started is taken toward each destination it binds, once no flow of the slot's earlier client
   runs or stops (one kept for a dirty lane, its stop run, touches no word of the slot).  Rows of a retired client are
   released once no flow of one remains (one kept for a dirty lane among them: design/recovery.md#five-seconds B7) */
void clients_turn(int liveness){
  const pid_t self=getpid();
  int changed=0,zombie=0;
  for(struct flow *f=flows;f;f=f->next)
    if(!owned(f->owner) || atomic_load(&mesh_client_at(M,f->owner)->configured)!=f->owner)flow_end(f);
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    uint64_t owner=atomic_load(&client->owner);
    if(owner && (pid_t)(uint32_t)owner==self)continue;
    if(liveness && owner && !(owner&MESH_RETIRING) && mesh_gone((pid_t)(uint32_t)owner)){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    if(owner && owner!=owners[s] && !(owner&MESH_RETIRING)){
      struct kevent64_s watch;EV_SET64(&watch,(uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,s,0,0);
      if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)<0 && errno==ESRCH){mesh_retire(M,owner);owner=atomic_load(&client->owner);}
    }
    changed|=owner!=owners[s];owners[s]=owner;
    if(!owner || (owner&MESH_RETIRING) || atomic_load_explicit(&client->configured,memory_order_acquire)!=owner || accepted[s]==owner)continue;
    int earlier=0;
    for(struct flow *f=flows;f;f=f->next)
      earlier|=f->owner!=owner && ((f->owner>>32)&(MESH_CLIENTS-1))==s && atomic_load_explicit(&f->stopped,memory_order_acquire)!=2;
    if(earlier)continue;
    accepted[s]=owner;
    atomic_store_explicit(&client->ports,0,memory_order_relaxed);
    for(uint32_t d=0,n=mesh_destination(client,0);d<n;d++)if(mesh_destination_transfers(M,owner,d))program_take(client,owner,d);
  }
  for(struct flow *f=flows;f;f=f->next)zombie|=!owned(f->owner);
  unreleased|=changed;
  if(unreleased && !zombie){unreleased=0;mesh_retired_release(M);}
}
/* design/bridge-and-ipc.md#flows: every flow's turn: its job's end taken, a stopped flow freed (one kept for a lane it
   left dirty once its link's lanes' generation moved, link_make's or a flush's that moved every queue pair, or once the
   move a window put off moved its lanes and stopped it: design/recovery.md#five-seconds B7, 4df30806 flow_release's
   rule), and a progress thread's finding (a post or a completion that failed) its failure; nothing here waits on a
   clock */
void flows_turn(void){
  for(struct flow *f=flows,*next;f;f=next){
    next=f->next;
    if(f->job && atomic_load_explicit(&f->done,memory_order_acquire))flow_done(f);
    if(f->stopping){
      const int stopped=atomic_load_explicit(&f->stopped,memory_order_acquire);
      if(!f->job && (stopped==1 || (stopped==2 && atomic_load_explicit(&f->port->link.generation,memory_order_acquire)!=f->lane_epoch)))
        flow_free(f);
      continue;
    }
    if(f->ended)continue;
    const int64_t failed=atomic_load(&f->failed);
    if(failed)flow_fail(f,failed,atomic_load(&f->domain),M->node,NULL);
  }
}
