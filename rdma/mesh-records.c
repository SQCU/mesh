#include "mesh-bridge.h"

/* the frames the switch's lanes have queued, told in the region (`busy`), and the wall clock at which none last came to
   be (`idle_since`, 0 while any is) */
static void busy_tell(void){
  const uint64_t busy=lanes_busy();
  if(busy)atomic_store_explicit(&M->idle_since,0,memory_order_relaxed);
  else if(!atomic_load_explicit(&M->idle_since,memory_order_relaxed))atomic_store_explicit(&M->idle_since,clock_gettime_nsec_np(CLOCK_REALTIME),memory_order_relaxed);
  atomic_store_explicit(&M->busy,busy,memory_order_release);
}
/* design/transport.md: the bytes the kernel can give without pressure (kern.memorystatus_level percent of
   the node's memory), read now */
static uint64_t available_now(void){
  int level=0;size_t size=sizeof level;
  return sysctlbyname("kern.memorystatus_level",&level,&size,NULL,0) || level<0?0:(uint64_t)level*(memory/100);
}
/* design/transport.md: what this node's record carries of its health (mesh-view.h struct mesh_health), its
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

/* design/transport.md, RFC 2328 s13's database: the newest record of each node, this node's own among them, kept
   whether a paired path from this node reaches it or not, with when this node last heard one (its clock, 0 for its
   own), when the map last lost the node (`lost_at`), and the ports whose peers have that record (`told`, a bit a port
   of `ports`: it was heard on the port or sent on it since the port's pairing), so a record is sent only where its
   peer lacks it; a full table gives the longest unheard of the nodes the map lacks its place */
struct known { uint64_t heard,lost_at; uint32_t told; struct mesh_record record; };
_Static_assert(MESH_PORTS<=32,"a port a bit of `told`");
static struct known known[MESH_RECORDS];
static uint32_t known_count;
static uint32_t port_bit(const struct port *port){ return UINT32_C(1)<<(uint32_t)(port-ports); }
/* the map's published copy (step 0) and the one not published (step 1, the next publish's: map_publish), this thread
   the one writer of both and of `map_version` */
static struct mesh_map_copy *map_copy(uint64_t step){
  return mesh_map_copies(M)+((atomic_load_explicit(&M->map_version,memory_order_relaxed)+step)&1);
}
/* the published map's record of `node`, NULL where the map has none */
static const struct mesh_record *record_of(uint32_t node){
  const struct mesh_map_copy *map=map_copy(0);
  for(uint32_t i=0;i<mesh_map_nodes(map);i++)if(map->record[i].node==node)return map->record+i;
  return NULL;
}
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].record.node==node)return known+i;
  struct known *at=NULL;
  if(known_count<MESH_RECORDS)at=known+known_count++;
  else for(uint32_t i=0;i<known_count;i++)if(!record_of(known[i].record.node) && (!at || known[i].heard<at->heard))at=known+i;
  if(at)*at=(struct known){.record.node=node};
  return at;
}
/* record k sent on every greeted port whose peer lacks it: one not told it, and not the record's own node (a bridge
   drops its own record, record_heard), each port it is sent on told.  Sent from a copy: a send that loses its port
   makes this node's record and the map again */
static void record_tell(struct known *k){
  const struct mesh_record record=k->record;
  for(uint32_t p=0;p<port_count;p++){
    if(!port_greeted(ports+p) || k->told>>p&1 || ports[p].peer==record.node)continue;
    port_say(ports+p,(struct message){.kind=RECORD},&record,sizeof record);
    k->told|=UINT32_C(1)<<p;
  }
}
/* a port paired anew: its peer told none of the table's records (it may be another bridge, or one that kept none) */
void records_forget(const struct port *port){
  const uint32_t bit=port_bit(port);
  for(uint32_t k=0;k<known_count;k++)known[k].told&=~bit;
}
/* design/transport.md, #region: the map, a record a node a paired path from this node reaches, each its node's record
   in the table above (this node's own rebuilt from its port table when that changes, its sequence one more; every
   other node's the newest heard), written in the copy not published and then published (mesh.h struct
   mesh_map_copy): the copy published by one release store of `map_version`, and where the mesh changed (`mesh`), the
   map's serial moved and its waiters woken (mesh.h mesh_map_moved).  An adopted region's published copy is the bridge
   before's until this bridge's first publish, its own record (node_start), replaces it */
static void map_publish(int mesh){
  atomic_store_explicit(&M->map_version,atomic_load_explicit(&M->map_version,memory_order_relaxed)+1,memory_order_release);
  if(mesh)mesh_map_moved(M);
}
/* design/transport.md: whether two records' reached nodes (each with its stream's bandwidth), boot and facts are the same */
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  int same=a->node==b->node && a->reached==b->reached && a->boot==b->boot &&
    a->region==b->region && a->memory==b->memory && !memcmp(a->model,b->model,sizeof a->model);
  for(uint32_t k=0;same && k<a->reached && k<MESH_STREAMS;k++)
    same=a->reach[k].node==b->reach[k].node && a->reach[k].padding==b->reach[k].padding &&
         a->reach[k].bandwidth==b->reach[k].bandwidth;
  return same;
}
/* a record whose links the published map has as they are: the map copied into the copy not published with that
   record's sequence and health, and published, the map's serial unmoved (a health sample is no change of the mesh) */
static void health_put(const struct mesh_record *record){
  const struct mesh_map_copy *was=map_copy(0);
  struct mesh_map_copy *next=map_copy(1);
  const uint32_t n=mesh_map_nodes(was);
  memcpy(next->record,was->record,(size_t)n*sizeof *next->record);
  next->nodes=n;
  for(uint32_t i=0;i<n;i++)if(next->record[i].node==record->node){next->record[i].sequence=record->sequence;next->record[i].health=record->health;}
  map_publish(0);
}
/* RFC 2328 s16.1 over the table: the map made again, every record known that a paired path from this node reaches (two
   nodes paired where each one's record names the other: mesh_joined), in the table's order, in the copy not
   published, and published only where it differs from the published one (so a change no path reaches moves no
   serial).  A node the map loses or gains is logged, and each record it gains is sent where a peer lacks it
   (record_tell): a port paired while the map lacked the record was never sent it (map_tell sends the map alone), and
   every other port's peer was told it or sent it.  A map published has the switch's view and the node table made from
   it at once (mesh-link.c switch_view) */
static void map_make(void){
  const struct mesh_map_copy *was=map_copy(0);
  struct mesh_map_copy *next=map_copy(1);
  const uint32_t n=mesh_map_nodes(was);
  unsigned char reached[MESH_RECORDS]={0};
  uint32_t order[MESH_RECORDS],gained[MESH_RECORDS],head=0,tail=0,gains=0;
  for(uint32_t k=0;k<known_count;k++)if(known[k].record.node==M->node){reached[k]=1;order[tail++]=k;}
  while(head<tail){
    const uint32_t a=order[head++];double beta;
    for(uint32_t b=0;b<known_count;b++)if(!reached[b] && mesh_joined(&known[a].record,&known[b].record,&beta)){reached[b]=1;order[tail++]=b;}
  }
  uint32_t count=0;
  for(uint32_t k=0;k<known_count;k++){
    struct known *node=known+k;
    int mapped=0;
    for(uint32_t i=0;i<n && !mapped;i++)mapped=was->record[i].node==node->record.node;
    if(reached[k])next->record[count++]=node->record;
    if(node->record.node==M->node || mapped==reached[k])continue;
    if(mapped){
      say("node %u unreachable, heard %.1f s before\n",node->record.node,node->heard?(double)(now-node->heard)/1e9:-1.0);
      node->lost_at=now;
      continue;
    }
    if(node->lost_at)say("node %u back after %.1f s\n",node->record.node,(double)(now-node->lost_at)/1e9);
    else say("node %u joined\n",node->record.node);
    node->lost_at=0;
    gained[gains++]=k;
  }
  if(count!=n || memcmp(next->record,was->record,(size_t)n*sizeof *next->record)){
    next->nodes=count;
    map_publish(1);
    switch_view();
  }
  /* each from its table entry (record_tell copies it): a port lost meanwhile makes the map again, and moves no entry */
  for(uint32_t g=0;g<gains;g++)record_tell(known+gained[g]);
}
/* design/transport.md: `record` stored as its node's newest (`k` its place in the table), heard on port `from` (NULL:
   this node's own), whose peer is then the one that has it.  One whose node the map has, and whose reached nodes,
   boot and facts are the map's record's, changes only its health and sequence there (health_put), published with the
   map's serial unmoved: the serial moves only where the mesh changes, and a health sample is no change of the mesh.
   Any other makes the map again (map_make) */
static void map_put(struct known *k,const struct mesh_record *record,const struct port *from){
  const struct mesh_record *mapped=record_of(record->node);
  k->record=*record;k->told=from?port_bit(from):0;
  if(mapped && record_links_same(mapped,record))health_put(record);
  else map_make();
}
/* design/transport.md: the map sent on `port` as its link comes up: each record of it whose peer lacks it
   (record_tell's rule), this node's own apart (record_changed sends it, naming the link), the peer keeping each until
   its own record names the link too (map_make) */
void map_tell(struct port *port){
  const uint32_t bit=port_bit(port);
  for(uint32_t k=0;port_greeted(port) && k<known_count;k++){
    struct known *node=known+k;
    if(node->record.node==M->node || node->record.node==port->peer || node->told&bit || !record_of(node->record.node))continue;
    const struct mesh_record record=node->record;
    port_say(port,(struct message){.kind=RECORD},&record,sizeof record);
    node->told|=bit;
  }
}
void record_changed(void){
  struct mesh_record own={.node=M->node,.boot=M->boot,.sequence=++sequence,.region=pool_bytes(),.memory=memory};
  memcpy(own.model,model,sizeof own.model);
  own.reached=record_reach(own.reach,MESH_STREAMS);
  own.health=health;
  struct known *k=known_at(M->node);
  map_put(k,&own,NULL);
  record_tell(k);
}
/* design/transport.md: this node's record told again at its current sequence on every greeted port, each
   liveness turn (MESH_PAUSE_NS), so each control socket carries a message well within MESH_PLATFORM_NS; a receiver drops
   a record no newer than its newest (record_heard, RFC 2328 s13), its map and
   serials unmoved.  Told from a copy: a port lost meanwhile may write the map */
void record_retell(void){
  const struct mesh_record *stored=record_of(M->node);
  if(!stored)return;
  struct mesh_record own;memcpy(&own,stored,sizeof own);
  for(uint32_t p=0;p<port_count;p++)if(port_greeted(ports+p))port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
/* when the next health sample is due (the event loop's turn waits for it: mesh-link.c turn_due) */
uint64_t health_due(void){ return health_sent+MESH_HEALTH_NS; }
void health_turn(void){
  if(now-health_sent<MESH_HEALTH_NS)return;
  health_sent=now;health.available=available_now();
  record_changed();
}
/* a record heard: as much of it as this bridge's record has, the rest 0 (design/HETEROGENEOUS-LINKS.md section 6,
   item 3: struct mesh_record is only appended to, so a longer one's first bytes are this one; a change of MESH_STREAMS or
   of the model's room is not an append, and moves MESH_VERSION), its reach at most MESH_STREAMS nodes;
   one naming no node, or this node, is not another node's */
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard={0};
  memcpy(&heard,payload,MIN((size_t)bytes,sizeof heard));
  if(!heard.node || heard.node==M->node)return;
  heard.reached=MIN(heard.reached,(uint32_t)MESH_STREAMS);
  heard.model[sizeof heard.model-1]=0;
  struct known *k=known_at(heard.node);
  /* design/transport.md, RFC 2328 s13: a record no newer than the newest heard of its node is dropped and not sent on,
     so no record circulates (a node's later boot is always newer), one the same as that newest telling that `from`'s
     peer has it; so is one the table has no place for (every node it holds is in the map) */
  if(!k)return;
  if(k->heard && heard.boot==k->record.boot && heard.sequence==k->record.sequence){k->told|=port_bit(from);return;}
  if(k->heard && (heard.boot<k->record.boot || (heard.boot==k->record.boot && heard.sequence<k->record.sequence)))return;
  k->heard=now;
  map_put(k,&heard,from);
  record_tell(k);
}
/* design/transport.md: every client slot read, on every turn of the event loop: a slot's process watched from the turn
   its slot is first seen as it stands, owned or leaving (its exit, the event loop's EVFILT_PROC, makes every slot it
   has leave: a client's death is that event, not a poll), one already exited made to leave at once (mesh.h
   mesh_leave, which also finishes a leaving its process left half done as it died: its indices freed and their pages
   dropped, the writes on its ring left for the port children to carry); and the frames queued told (busy_tell).  A
   slot taken is seen at the turn after, whatever wakes the loop (at most MESH_PAUSE_NS on) */
void clients_turn(void){
  const pid_t self=getpid();
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    uint64_t owner=atomic_load(&client->owner);
    if(owner && (pid_t)(uint32_t)owner==self)continue;
    if(owner && owner!=owners[s]){
      struct kevent64_s watch;EV_SET64(&watch,(uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,s,0,0);
      if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)<0 && errno==ESRCH){mesh_leave(M,owner);owner=atomic_load(&client->owner);}
    }
    owners[s]=owner;
  }
  busy_tell();
}
