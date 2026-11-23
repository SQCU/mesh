#include "mesh-bridge.h"

/* the frames the switch's lanes have queued, told in the region (`busy`), and the wall clock at which none last came to
   be (`idle_since`, 0 while any is) */
static void busy_tell(void){
  const uint64_t busy=lanes_busy();
  if(busy)atomic_store_explicit(&M->idle_since,0,memory_order_relaxed);
  else if(!atomic_load_explicit(&M->idle_since,memory_order_relaxed))atomic_store_explicit(&M->idle_since,clock_gettime_nsec_np(CLOCK_REALTIME),memory_order_relaxed);
  atomic_store_explicit(&M->busy,busy,memory_order_release);
}
static uint32_t map_page,map_pages,map_room;
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
#define MESH_KNOWN 256
static struct known known[MESH_KNOWN];
static uint32_t known_count;
static uint32_t port_bit(const struct port *port){ return UINT32_C(1)<<(uint32_t)(port-ports); }
/* the map's record of `node`, NULL where the map has none */
static struct mesh_record *record_of(uint32_t node){
  for(uint64_t i=0;i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node==node)return mesh_records(M)+i;
  return NULL;
}
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].record.node==node)return known+i;
  struct known *at=NULL;
  if(known_count<MESH_KNOWN)at=known+known_count++;
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
    if(!ports[p].greeted || k->told>>p&1 || ports[p].peer==record.node)continue;
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
   other node's the newest heard); each change written between two notifications of the map serial (odd while it
   writes), the records in arena pages the bridge leases and leases again larger as the mesh grows */
static int map_grow(uint64_t records){
  if(records<=map_room)return 0;
  uint32_t pages=(uint32_t)((records*2*sizeof(struct mesh_record)+M->pgsz-1)/M->pgsz),page=mesh_arena_alloc(&bridge,pages,1);
  if(page==MESH_ABSENT)return ENOMEM;
  memcpy(mesh_at(M,page),mesh_records(M),(size_t)atomic_load(&M->map_nodes)*sizeof(struct mesh_record));
  atomic_store(&M->map_off,(uint64_t)(mesh_at(M,page)-(unsigned char *)M));
  mesh_bridge_pages(M)->map=page;mesh_bridge_pages(M)->map_pages=pages;
  if(map_pages)mesh_arena_give(M,map_page,map_pages);
  map_page=page;map_pages=pages;map_room=(uint32_t)((uint64_t)pages*M->pgsz/sizeof(struct mesh_record));
  return 0;
}
/* design/transport.md section 2: an adopted region's map pages, recorded by the bridge before (mesh-observe.h struct
   mesh_bridge_pages), taken again as this bridge's, its records none until this bridge writes its own */
void map_adopt(void){
  const struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  if(own->map==MESH_ABSENT)return;
  map_page=own->map;map_pages=own->map_pages;map_room=(uint32_t)((uint64_t)map_pages*M->pgsz/sizeof(struct mesh_record));
  atomic_store(&M->map_nodes,0);
  atomic_store(&M->map_off,(uint64_t)(mesh_at(M,map_page)-(unsigned char *)M));
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
static void health_put(struct mesh_record *stored,const struct mesh_record *record){
  mesh_notify(&M->health);
  stored->sequence=record->sequence;stored->health=record->health;
  mesh_notify(&M->health);
}
/* RFC 2328 s16.1 over the table: the map made again, every record known that a paired path from this node reaches (two
   nodes paired where each one's record names the other: mesh_joined), in the table's order, written only where it
   differs from the map as it stands (so a change no path reaches moves no serial).  A node the map loses or gains is
   logged, and each record it gains is sent where a peer lacks it (record_tell): a port paired while the map lacked
   the record was never sent it (map_tell sends the map alone), and every other port's peer was told it or sent it.
   A map written has the switch's view and the node table made from it at once (mesh-link.c switch_view) */
static void map_make(void){
  const uint64_t n=atomic_load(&M->map_nodes);
  const struct mesh_record *r=mesh_records(M);
  struct mesh_record *next=malloc(MESH_KNOWN*sizeof *next);
  unsigned char *reached=calloc(MESH_KNOWN,1);
  uint32_t *order=malloc(MESH_KNOWN*sizeof *order),*gained=malloc(MESH_KNOWN*sizeof *gained),head=0,tail=0,gains=0;
  if(!next || !reached || !order || !gained){
    free(next);free(reached);free(order);free(gained);say("map: no memory to make it again\n");return;
  }
  for(uint32_t k=0;k<known_count;k++)if(known[k].record.node==M->node){reached[k]=1;order[tail++]=k;}
  while(head<tail){
    const uint32_t a=order[head++];double beta;
    for(uint32_t b=0;b<known_count;b++)if(!reached[b] && mesh_joined(&known[a].record,&known[b].record,&beta)){reached[b]=1;order[tail++]=b;}
  }
  uint64_t count=0;
  for(uint32_t k=0;k<known_count;k++){
    struct known *node=known+k;
    int was=0;
    for(uint64_t i=0;i<n && !was;i++)was=r[i].node==node->record.node;
    if(reached[k])next[count++]=node->record;
    if(node->record.node==M->node || was==reached[k])continue;
    if(was){
      say("node %u unreachable, heard %.1f s before\n",node->record.node,node->heard?(double)(now-node->heard)/1e9:-1.0);
      node->lost_at=now;
      continue;
    }
    if(node->lost_at)say("node %u back after %.1f s\n",node->record.node,(double)(now-node->lost_at)/1e9);
    else say("node %u joined\n",node->record.node);
    node->lost_at=0;
    gained[gains++]=k;
  }
  if(count!=n || memcmp(next,r,(size_t)n*sizeof *r)){
    mesh_notify(&M->map);
    if(map_grow(count))say("map: no lease for %llu records; %llu remaining\n",(unsigned long long)count,(unsigned long long)map_room);
    if(count>map_room)count=map_room;
    memcpy(mesh_records(M),next,(size_t)count*sizeof *next);
    atomic_store(&M->map_nodes,count);
    mesh_notify(&M->map);
    switch_view();
  }
  free(next);free(reached);free(order);
  /* each from its table entry (record_tell copies it): a port lost meanwhile makes the map again, and moves no entry */
  for(uint32_t g=0;g<gains;g++)record_tell(known+gained[g]);
  free(gained);
}
/* design/transport.md: `record` stored as its node's newest (`k` its place in the table), heard on port `from` (NULL:
   this node's own), whose peer is then the one that has it.  One whose node the map has, and whose reached nodes,
   boot and facts are the map's record's, changes only its health and sequence there, written under the region's
   health serial and never the map's: the map serial moves only where the mesh changes, and a health sample is no
   change of the mesh.  Any other makes the map again (map_make) */
static void map_put(struct known *k,const struct mesh_record *record,const struct port *from){
  struct mesh_record *mapped=record_of(record->node);
  k->record=*record;k->told=from?port_bit(from):0;
  if(mapped && record_links_same(mapped,record))health_put(mapped,record);
  else map_make();
}
/* design/transport.md: the map sent on `port` as its link comes up: each record of it whose peer lacks it
   (record_tell's rule), this node's own apart (record_changed sends it, naming the link), the peer keeping each until
   its own record names the link too (map_make) */
void map_tell(struct port *port){
  const uint32_t bit=port_bit(port);
  for(uint32_t k=0;port->greeted && k<known_count;k++){
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
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
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
/* design/transport.md: every client slot read: a slot's process watched from the turn its slot is first seen (its exit,
   the event loop's EVFILT_PROC, makes every slot it has leave: a client's death is that event, not a poll), one
   already exited made to leave at once (mesh.h mesh_leave: its indices freed and their pages dropped, the writes on
   its ring left for the port children to carry); on a liveness turn, a slot whose leaving its process left half done as it died finished; and the frames
   queued told (busy_tell) */
void clients_turn(int liveness){
  const pid_t self=getpid();
  for(uint32_t s=0;s<MESH_CLIENTS;s++){
    struct mesh_client *client=(struct mesh_client *)((char *)M+M->client_off)+s;
    uint64_t owner=atomic_load(&client->owner);
    if(owner && (pid_t)(uint32_t)owner==self)continue;
    if(owner && owner!=owners[s] && !(owner&MESH_LEAVING)){
      struct kevent64_s watch;EV_SET64(&watch,(uint32_t)owner,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,s,0,0);
      if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)<0 && errno==ESRCH){mesh_leave(M,owner);owner=atomic_load(&client->owner);}
    }
    if(liveness && owner&MESH_LEAVING && !mesh_alive((pid_t)(uint32_t)owner,client->start)){mesh_leave(M,owner);owner=atomic_load(&client->owner);}
    owners[s]=owner;
  }
  busy_tell();
}
