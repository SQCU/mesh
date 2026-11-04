/* The bridge's map (design/page-queue-change-request.md, "State of the code": the resident bridge keeps a map): this
   node's record and every other node's newest heard, in the region for the collective and the bridges' rule to read
   (mesh.h struct mesh_record, struct mesh_reach, hdr `map`, `map_serial`), its health samples, and the client's
   watch.  From tag pre-return-20261103 rdma/mesh-records.c, whole; only interfaces are adapted (main's region fields,
   its one client word, the map's area laid out with the region in place of the tag's arena lease), and each line that
   is not the tag's has a note at its spot.  The map holds
   Thunderbolt-5-linked members only: a record no paired path from this node reaches is left out (map_put), and every
   reach entry is a cable a port paired (mesh-ports.c record_reach). */
#include "mesh-bridge.h"
/* not the tag's: main's mesh-bridge.h does not include the library's header (the tag's included mesh-call.h), and
   mesh_retire and mesh_retired_release are declared there (mesh_arena_alloc, the tag's third, is called here no
   longer: map_grow's note) */
#include "mesh-dataflow.h"
/* MESH_HEALTH_NS is mesh-bridge.h's (review R2, B2) */
#include "mesh-graph.h"

/* tag pre-return-20261103 rdma/mesh-records.c busy_tell (3f72cbde), adapted: a client's hold is main's client word */
/* design/transport.md section 14: the streams held, an owner word naming a client (a stream whose holding ended with
   units outstanding names none: MESH_STREAM_GIVEN alone), told in the region (`busy`), and the wall clock at which
   none last came to be (`idle_since`, 0 while any is) */
/* differs from the tag: main has no streams (mesh-bridge.h: the data path is the page lists'), and what a client holds of
   the bridge is its one client word, hdr `client` (the client's pid in its low 32 bits), so `busy` is 1 while that word
   names a client, 0 where it names none or this bridge (main's mesh_retire writes its retirer's pid there until it
   stores 0), in place of the tag's count of streams whose owner names a client */
static void busy_tell(void){
  const uint64_t owner=atomic_load_explicit(&M->client,memory_order_relaxed);
  uint64_t busy=owner && (pid_t)(uint32_t)owner!=getpid();
  if(busy)atomic_store_explicit(&M->idle_since,0,memory_order_relaxed);
  else if(!atomic_load_explicit(&M->idle_since,memory_order_relaxed))atomic_store_explicit(&M->idle_since,clock_gettime_nsec_np(CLOCK_REALTIME),memory_order_relaxed);
  atomic_store_explicit(&M->busy,busy,memory_order_release);
}
/* tag pre-return-20261103 rdma/mesh-records.c, the map's lease (3f72cbde), `static uint32_t map_page,map_pages,map_room;`,
   is not here: no lease exists (map_grow's note); the map's room is hdr `map_room` */
/* tag pre-return-20261103 rdma/mesh-records.c available_now (3f72cbde), verbatim */
/* design/transport.md: the bytes the kernel can give without pressure (kern.memorystatus_level percent of
   the node's memory), read now */
static uint64_t available_now(void){
  int level=0;size_t size=sizeof level;
  return sysctlbyname("kern.memorystatus_level",&level,&size,NULL,0) || level<0?0:(uint64_t)level*(memory/100);
}
/* tag pre-return-20261103 rdma/mesh-records.c, the health sampled (3f72cbde), verbatim */
/* design/transport.md: what this node's record carries of its health (mesh-plan.h struct mesh_health), its
   boot read once and its available memory at most MESH_HEALTH_NS apart (health_turn) */
static struct mesh_health health;
static uint64_t health_sent;
/* tag pre-return-20261103 rdma/mesh-records.c sysctl_text (3f72cbde), verbatim */
void sysctl_text(const char *name,char *out,size_t room){
  size_t size=room;
  if(sysctlbyname(name,out,&size,NULL,0))out[0]=0;
  out[room-1]=0;
}
/* tag pre-return-20261103 rdma/mesh-records.c health_init (3f72cbde), verbatim */
void health_init(void){
  struct timeval booted;size_t size=sizeof booted;
  if(!sysctlbyname("kern.boottime",&booted,&size,NULL,0))health.booted=(uint64_t)booted.tv_sec*1000000000u+(uint64_t)booted.tv_usec*1000u;
  health.available=available_now();
}

/* tag pre-return-20261103 rdma/mesh-records.c known_at (3f72cbde), verbatim */
static uint32_t known_count;
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].node==node)return known+i;
  struct known *at=NULL;
  if(known_count<MESH_KNOWN)at=known+known_count++;
  else for(uint32_t i=0;i<known_count;i++)if(!record_held(known[i].node) && (!at || known[i].heard<at->heard))at=known+i;
  if(at)*at=(struct known){.node=node};
  return at;
}
/* tag pre-return-20261103 rdma/mesh-records.c map_grow (3f72cbde), adapted then to main's mesh_arena_alloc, is not
   here.  2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 ("Sharing the region between programs":
   "The bridge's own pages (its records) are laid out once with the region, not taken from any program's section") and
   A4 (mesh-records.c's call of mesh_arena_alloc goes), reversing this file's first form, in which map_grow leased the
   records arena pages outside the registered window (mesh_arena_alloc of the bridge's own context `bridge`), leased
   again twice as large when the map outgrew them, and recorded them in struct mesh_bridge_pages `map`, `map_pages` and
   `held_pages`.  Why: the arena is the programs' sections alone, and a page the bridge took there was one call-site
   allocation among those A4 removes.  Now the map's records are its area of the layout: hdr `map_room` records
   (MESH_MAP_ROOM) at hdr `map_off`, both set by mesh_layout (mesh-map.h) before the arena and never moved, so nothing
   grows, nothing is copied and the bridge holds no arena page.  What must stay: the map is written at `map_off` and
   within `map_room` alone; a map past the room leaves its records past it out (map_put), as the tag left out what a
   failed lease could not hold */
/* tag pre-return-20261103 rdma/mesh-records.c map_adopt (3f72cbde), adapted: the records' clearing is a change of the
   map's nodes */
/* design/transport.md section 2: an adopted region's map pages, recorded by the bridge before (mesh-observe.h struct
   mesh_bridge_pages), taken again as this bridge's, its records none until this bridge writes its own */
/* differs from the tag (2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3, "Sharing the region
   between programs"): the map's area is the layout's (hdr `map_off`, `map_room`), which region_open's region_same
   (mesh-link.c) found the same in the adopted region as in this bridge's layout, so nothing is taken again: the tag's
   test of `map` (MESH_ABSENT where none was leased), its taking of `map`, `map_pages` and the room, the held pages'
   count and the store of `map_off` go, and the area is adopted as it lies */
void map_adopt(void){
  /* differs from the tag: the records cleared between two notifications of `map`, and `map_serial` moved where any
     were held, since every change of the map's nodes moves it (mesh.h `map_serial`): a plan made against the earlier
     bridge's map is refused.  region_adopt (mesh-link.c) has made `map` even before this, as at the tag */
  mesh_notify(&M->map);
  if(atomic_load(&M->map_nodes))atomic_fetch_add(&M->map_serial,1);
  atomic_store(&M->map_nodes,0);
  mesh_notify(&M->map);
}
/* tag pre-return-20261103 rdma/mesh-records.c map_put (3f72cbde, abeffc1b), adapted: node-id order and the map serial */
/* RFC 2328 s14: the map with `record` put in (its node's held record replaced, else added) and every record no paired
   path from this node reaches left out, written only where it differs from the map held (so a record that changes
   nothing, or that the prune drops at once, moves no serial) */
/* differs from the tag: a record added goes to its place in ascending node-id order, not to the end, so the map's
   records stay in that order (the held map is in it, and the prune keeps the order), and a node's index is its place
   (mesh.h `map_serial`); and `map_serial` moves, while `map` is odd, where the write changes the map's nodes (which,
   or how many) or any cable (a record's `reached` or any field of a reach entry: peer node, ports, queue pairs,
   bandwidth), and not where it changes only the rest of a record (its sequence, boot, region, memory, model); and the
   reach from this node is the graph module's (mesh_graph_reach over the joins, C9), not a walk of its own here, its
   reached set the tag's walk's */
/* an edge's end for mesh_graph_reach: join e's record on `side` (map_put) */
static uint32_t map_end(const void *context,uint32_t e,uint32_t side){ return ((const uint32_t (*)[2])context)[e][side]; }
static void map_put(const struct mesh_record *record){
  const uint64_t n=atomic_load(&M->map_nodes);
  struct mesh_record *r=mesh_records(M);
  uint64_t at=0;
  while(at<n && r[at].node<record->node)at++;
  const uint64_t added=at==n || r[at].node!=record->node;
  const uint64_t count=added?n+1:n;
  struct mesh_record *next=malloc(count*sizeof *next);
  unsigned char *reached=calloc(count,1);
  uint32_t *order=malloc(count*sizeof *order),(*join)[2]=malloc((count*(count-1)/2+1)*sizeof *join);
  if(!next || !reached || !order || !join){free(next);free(reached);free(order);free(join);return;}
  memcpy(next,r,(size_t)at*sizeof *r);
  memcpy(next+at,record,sizeof *record);
  memcpy(next+at+1,r+at+!added,(size_t)(n-at-!added)*sizeof *r);
  /* not the tag's lines: the joins (mesh_joined, symmetric) as edges, and this node's record the reach's start; none
     reached where the map holds no record of this node, as the tag's walk seeded none */
  uint64_t self=count,joins=0;
  for(uint64_t a=0;a<count;a++){
    if(next[a].node==M->node)self=a;
    for(uint64_t b=a+1;b<count;b++){double beta;if(mesh_joined(next+a,next+b,&beta)){join[joins][0]=(uint32_t)a;join[joins][1]=(uint32_t)b;joins++;}}
  }
  if(self<count)mesh_graph_reach((uint32_t)count,(uint32_t)joins,map_end,join,(uint32_t)self,order,reached);
  free(join);
  uint64_t kept=0;
  for(uint64_t k=0;k<count;k++)if(reached[k])memmove(next+kept++,next+k,sizeof *next);
  if(kept!=n || memcmp(next,r,(size_t)n*sizeof *r)){
    /* not the tag's: whether the nodes or a cable change, read before the write (the write replaces the records read) */
    int moved=kept!=n;
    for(uint64_t k=0;!moved && k<kept;k++)
      moved=next[k].node!=r[k].node || next[k].reached!=r[k].reached ||
        memcmp(next[k].reach,r[k].reach,(size_t)MIN(next[k].reached,(uint32_t)MESH_PORTS)*sizeof *next[k].reach);
    mesh_notify(&M->map);
    /* not the tag's (2026-11-03, map_grow's note): the tag's `map_grow(kept);` gone, and the room the layout's hdr
       `map_room`, the tag's leased `map_room`; `moved=1` not the tag's: records past the room, left out as the tag left
       out those a failed lease could not hold, are a change of the map's nodes */
    if(kept>M->map_room){kept=M->map_room;moved=1;}
    memcpy(mesh_records(M),next,(size_t)kept*sizeof *next);
    atomic_store(&M->map_nodes,kept);
    if(moved)atomic_fetch_add(&M->map_serial,1);
    mesh_notify(&M->map);
  }
  free(next);free(reached);free(order);
}
/* tag pre-return-20261103 rdma/mesh-records.c record_links_same (3f72cbde), adapted: a reach entry compared whole */
/* design/transport.md: a record whose reached nodes (each with its stream's bandwidth), boot and facts are the
   held one's changes only its health and sequence, written under the region's health serial and never the map's: the
   map serial moves only where the mesh changes, and a health sample is no change of the mesh */
/* differs from the tag: the reach entry is mesh.h's per cable, and every field of it (peer node, both ports, queue
   pairs, bandwidth) is the rule's input (struct mesh_hop, "Both ends"), so all are compared, not the tag's node and
   bandwidth alone; its bound MESH_PORTS, the tag's MESH_STREAMS */
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  int same=a->node==b->node && a->reached==b->reached && a->boot==b->boot &&
    a->region==b->region && a->memory==b->memory && !memcmp(a->model,b->model,sizeof a->model);
  for(uint32_t k=0;same && k<a->reached && k<MESH_PORTS;k++)
    same=!memcmp(a->reach+k,b->reach+k,sizeof *a->reach);
  return same;
}
/* tag pre-return-20261103 rdma/mesh-records.c health_put (3f72cbde), verbatim */
static void health_put(struct mesh_record *held,const struct mesh_record *record){
  mesh_notify(&M->health);
  held->sequence=record->sequence;held->health=record->health;
  mesh_notify(&M->health);
}
/* tag pre-return-20261103 rdma/mesh-records.c record_held (3f72cbde), verbatim */
struct mesh_record *record_held(uint32_t node){
  for(uint64_t i=0;i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node==node)return mesh_records(M)+i;
  return NULL;
}
/* tag pre-return-20261103 rdma/mesh-records.c record_changed (3f72cbde), adapted: the arena's bytes, MESH_PORTS */
/* differs from the tag: `region` is the arena's bytes, as at the tag, whose wire mapping was the arena (tag
   mesh-verbs.h wire_map); main's `wire` is the registered window alone (mesh-verbs.h wire_map).  The reach's room is
   MESH_PORTS cables (mesh.h struct mesh_record), the tag's MESH_STREAMS streams */
/* 2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 ("Sharing the region between programs"): the
   tag's `- held_pages*M->pgsz` goes, since the bridge holds no arena page (its map is the layout's area, map_grow's
   note, and it leases no link pages or duds' block, mesh-link.c): every arena byte is the programs'.  What must stay:
   `region` the bytes the programs' sections may take */
void record_changed(void){
  struct mesh_record own={.node=M->node,.boot=M->boot,.sequence=++sequence,.region=(uint64_t)mesh_arena_pages(M)*M->pgsz,.memory=memory};
  memcpy(own.model,model,sizeof own.model);
  own.reached=record_reach(own.reach,MESH_PORTS);
  own.health=health;
  struct mesh_record *held=record_held(M->node);
  if(held && record_links_same(held,&own))health_put(held,&own);
  else map_put(&own);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
/* tag pre-return-20261103 rdma/mesh-records.c record_retell (3f72cbde), verbatim */
/* design/transport.md: this node's record told again at its current sequence on every greeted port, each
   liveness turn (MESH_PAUSE_NS), so each control socket carries a message well within MESH_PLATFORM_NS; a receiver drops
   a record no newer than its newest (record_heard, RFC 2328 s13), its map and
   serials unmoved.  Told from a copy: a port lost meanwhile may write the map */
void record_retell(void){
  const struct mesh_record *held=record_held(M->node);
  if(!held)return;
  struct mesh_record own;memcpy(&own,held,sizeof own);
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
/* tag pre-return-20261103 rdma/mesh-records.c health_due (3f72cbde), verbatim */
/* when the next health sample is due (the event loop's turn waits for it: mesh-link.c turn_due) */
uint64_t health_due(void){ return health_sent+MESH_HEALTH_NS; }
/* tag pre-return-20261103 rdma/mesh-records.c health_turn (3f72cbde), verbatim */
void health_turn(void){
  if(now-health_sent<MESH_HEALTH_NS)return;
  health_sent=now;health.available=available_now();
  record_changed();
}
/* tag pre-return-20261103 rdma/mesh-records.c record_heard (3f72cbde, 60b00774, abeffc1b), adapted: MESH_PORTS */
/* a record heard: as much of it as this bridge's record holds, the rest 0 (design/HETEROGENEOUS-LINKS.md section 6,
   item 3: struct mesh_record is only appended to, so a longer one's first bytes are this one; a change of MESH_STREAMS or
   of the model's room is not an append, and moves MESH_VERSION), its reach at most MESH_STREAMS nodes;
   one naming no node, or this node, is not another node's */
/* differs from the tag: the reach's bound is MESH_PORTS cables (mesh.h struct mesh_record), the tag's MESH_STREAMS */
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard={0};
  memcpy(&heard,payload,MIN((size_t)bytes,sizeof heard));
  if(!heard.node || heard.node==M->node)return;
  heard.reached=MIN(heard.reached,(uint32_t)MESH_PORTS);
  heard.model[sizeof heard.model-1]=0;
  struct mesh_record *held=record_held(heard.node);
  struct known *k=known_at(heard.node);
  /* design/transport.md, RFC 2328 s13: a record no newer than the newest heard of its node, the map holding
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
  if(k){k->heard=now;k->boot=heard.boot;k->sequence=heard.sequence;}
  if(held && record_links_same(held,&heard))health_put(held,&heard);
  else map_put(&heard);
  for(uint32_t p=0;p<port_count;p++)if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
}
/* tag pre-return-20261103 rdma/mesh-records.c clients_turn (3f72cbde), adapted: main's one client word, not slots */
/* design/transport.md: every client slot read: a slot's process watched from the turn its slot is first
   seen (its exit, the event loop's EVFILT_PROC, retires every slot it holds: a holder's death is that event, not a
   poll), one already gone retired at once; the rows of a retired client released but those a request outstanding on a
   lane names (mesh-dataflow.c mesh_retired_release, mesh_pinned), a stream a gone client held let go (mesh-link.c
   streams_let_go: its port's progress thread, which posts nothing of it once the client retired, ends the holding), and
   the streams held told (busy_tell) */
/* differs from the tag: main has one client word, hdr `client` (its pid in the low 32 bits), not MESH_CLIENTS slots, so
   the tag's loop is its one slot, s = 0: `owners[0]` its last read, its watch's udata 0 (the tag's slot index; the
   event loop's EVFILT_PROC of udata 0 retires hdr `client` where its low 32 bits are the event's ident).  Main's word has
   no MESH_RETIRING bit: a retirement writes its retirer's pid there (mesh-dataflow.c mesh_retire), this bridge's
   skipped as the tag skips its own, another's watched as a client's.  mesh_retired_release keeps the rows a port's plan
   may still name (any port's `prepared` nonzero, mesh-dataflow.c c01fa79d) and returns how many it kept, as the tag's
   did, so `unreleased` stays set until none is kept */
/* 2026-11-04, sub-assignment 7 (design/disjoint-pages-change-request.md, C3.4: "The bridge's own turn releases a
   retired plan's pages once no port holds them", deferred reclamation; design/disjoint-pages-revision-coordination.md,
   "The program's claim and its sections" and D10), the note above made exact for the claims per plan: what
   mesh_retired_release keeps is each closed claim whose generation a port's `prepared` holds (after a seq_cst fence),
   not every closed claim while any port holds any plan, and the count it returns is of the closed claims kept.  And
   the order of this turn's last calls is reversed from the tag's (release, then let go): streams_let_go first, then
   the release, so a cancellation this turn computes from a plan's entries (mesh-flow.c plan_cancel) lands in that
   plan's pages before this turn's release can let them go.  What must stay: the cancellation before the release */
void clients_turn(int liveness){
  const pid_t self=getpid();
  int changed=0;
  uint64_t owner=atomic_load(&M->client);
  if(!(owner && (pid_t)(uint32_t)owner==self)){
    if(owner && owner!=owners[0]){
      struct kevent64_s watch;EV_SET64(&watch,(uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,0,0,0);
      if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)<0 && errno==ESRCH){mesh_retire(M,owner);owner=atomic_load(&M->client);}
    }
    changed|=owner!=owners[0];owners[0]=owner;
  }
  unreleased|=changed;
  /* where no client holds the region, this node's receives of the plan written that have not landed are cancelled:
     each one's count word raised to its final value, no phase written (mesh-flow.c streams_let_go, plan_cancel;
     design/disjoint-pages-change-request.md, C3.1 and C3.6), before the release below (the note above, 2026-11-04) */
  streams_let_go();
  /* the tag's line again (the coordinator, at integration, once mesh-dataflow.c's mesh_retired_release returns the rows
     it kept, c01fa79d): a retired client's rows a port's plan may still name are kept, and looked at again next turn */
  /* 2026-11-04, sub-assignment 7, for mesh-flow.c plan_cancel (design/disjoint-pages-change-request.md, C3.6;
     design/disjoint-pages-revision-coordination.md, "The bridge": "plan_cancel and the release of retired claims
     (mesh_retired_release) both run on the bridge parent's one event-loop thread"): they do.  In the bridge,
     mesh_retired_release is called here and in mesh-link.c region_adopt, and plan_cancel only from streams_cancel and
     streams_let_go; each of those runs on the thread of mesh-link.c main alone: region_adopt before the event loop, its
     streams_cancel(NULL) before its release; streams_let_go here, in clients_turn, the loop's turn; streams_cancel from
     mesh-ports.c port_lost and mesh-children.c child_reaped and child_heard, which only the loop's turn and its events
     reach (port_lost's callers: the loop's events, child_reaped, control_heard, hello_heard, network_heard and a
     control socket's connect; its turn's terminations_turn, ports_stalled, link_turn, port_open by child_spawn,
     health_turn and record_retell by port_say; and network_heard from the stop's hold after the loop), and
     streams_cancel(NULL) at the stop, after the loop.
     The parent's other threads call none of them: control_watch only wakes the loop; the finder's queue (mesh-ports.c
     ports_find) only lists the devices; the terminations' queue (mesh-events.c peer_terminated, peer_arrived) only
     stores atomics (port_halt) and wakes the loop.  The port children's threads, which land and send, are other
     processes and call neither.  So a cancellation's stores, computed from a plan's entries read whole, are done before
     any later call of the release can let that plan's pages go */
  if(unreleased || liveness)unreleased=mesh_retired_release(M)!=0;
  busy_tell();
}
