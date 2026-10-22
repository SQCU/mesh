#include "mesh-bridge.h"

/* design/transport.md section 14: the streams held, told in the region (`busy`), and the wall clock at which none
   last came to be (`idle_since`, 0 while any is) */
static void busy_tell(void){
  uint64_t busy=0;
  for(uint32_t p=0;p<port_count*MESH_LINK_STREAMS;p++)busy+=atomic_load_explicit(&mesh_stream_at(M,p)->owner,memory_order_relaxed)!=0;
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
/* design/transport.md: what this node's record carries of its health (mesh-plan.h struct mesh_health), its
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

/* design/transport.md, RFC 2328 s12, s13: the database, held apart from the map: of each node heard, this node's own
   first, the newest record heard of it (`record`: a node's later boot is always newer), whether a paired path reaches it
   or not, when it was last heard (this node's clock), when the map last let it go (`gone`, 0 while the map holds it or
   never held it), and the last host told sharing its number (`told`, number_shared).  A record no paired path reaches is
   kept whole, as OSPF keeps its database, so the map takes it the moment a path does, whichever of the path's records
   arrives last (map_publish), and is never sent on again (record_heard sends on only a newer one) nor told a newly paired
   peer (mesh-link.c port_heard tells the map's).  The database grows as nodes are heard; where it cannot, the longest
   unheard of the nodes the map does not hold gives its place */
struct known { uint64_t heard,gone; uint8_t told[16]; struct mesh_record record; };
static struct known *known;
static uint32_t known_count,known_room;
static uint8_t own_told[16];
static struct known *known_at(uint32_t node){
  for(uint32_t i=0;i<known_count;i++)if(known[i].record.node==node)return known+i;
  struct known *at=NULL,*grown=NULL;
  const uint32_t room=known_room?2*known_room:64;
  if(known_count==known_room && (grown=realloc(known,(size_t)room*sizeof *known))){known=grown;known_room=room;}
  if(known_count<known_room)at=known+known_count++;
  else for(uint32_t i=0;i<known_count;i++)
    if(known[i].record.node!=M->node && !record_held(known[i].record.node) && (!at || known[i].heard<at->heard))at=known+i;
  if(at)*at=(struct known){.record.node=node};
  return at;
}
/* design/transport.md, #region: the map, a record a node: this node's own, rebuilt from its port table when
   that changes (its sequence one more), and every other node's newest heard that a paired path reaches (map_publish);
   each change written between two notifications of the map serial (odd while it writes), the records in arena pages the
   bridge leases and leases again larger as the mesh grows */
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
/* design/transport.md section 2: an adopted region's map pages, recorded by the bridge before (mesh-observe.h struct
   mesh_bridge_pages), taken again as this bridge's, its records none until this bridge writes its own */
void map_adopt(void){
  const struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  if(own->map==MESH_ABSENT)return;
  map_page=own->map;map_pages=own->map_pages;map_room=(uint32_t)((uint64_t)map_pages*M->pgsz/sizeof(struct mesh_record));
  held_pages+=map_pages;
  atomic_store(&M->map_nodes,0);
  atomic_store(&M->map_off,(uint64_t)(mesh_at(M,map_page)-(unsigned char *)M));
}
/* RFC 2328 s16: the map made from the database, every record a path of joined records reaches from this node's
   (mesh_joined: each names the other) in the database's order, this node's first, and no other, written only where it
   differs from the map held (so a change that brings no node in, lets none go and alters no held record moves no
   serial); a node the map comes to hold (joined, or back) or lets go (unreachable) logged as it does
   (design/transport.md) */
static void map_publish(void){
  const uint64_t n=atomic_load(&M->map_nodes);
  struct mesh_record *r=mesh_records(M),*next=malloc(((size_t)known_count+1)*sizeof *next);
  unsigned char *reached=calloc((size_t)known_count+1,1);
  uint32_t *order=malloc(((size_t)known_count+1)*sizeof *order),head=0,tail=0;
  if(!next || !reached || !order){free(next);free(reached);free(order);say("map: no memory to make it from %u records\n",known_count);return;}
  for(uint32_t k=0;k<known_count;k++)if(known[k].record.node==M->node){reached[k]=1;order[tail++]=k;}
  while(head<tail){
    const uint32_t a=order[head++];double beta;
    for(uint32_t b=0;b<known_count;b++)if(!reached[b] && mesh_joined(&known[a].record,&known[b].record,&beta)){reached[b]=1;order[tail++]=b;}
  }
  uint64_t kept=0;
  for(uint32_t k=0;k<known_count;k++){
    struct known *x=known+k;
    const int held=record_held(x->record.node)!=NULL;
    if(reached[k])next[kept++]=x->record;
    if(x->record.node==M->node || reached[k]==held)continue;
    char id[37];uuid_unparse_lower(x->record.uuid,id);
    if(reached[k] && x->gone)say("node %u (host %s) back after %.1f s\n",x->record.node,id,(double)(now-x->gone)/1e9);
    else if(reached[k])say("node %u (host %s) joined\n",x->record.node,id);
    else say("node %u (host %s) unreachable, heard %.1f s before\n",x->record.node,id,(double)(now-x->heard)/1e9);
    x->gone=reached[k]?0:now;
  }
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
/* design/transport.md: a record whose links (each its node, streams and bandwidth), boot, host and facts are the
   held one's changes only its health and sequence, written under the region's health serial and never the map's: the
   map serial moves only where the mesh changes, and a health sample is no change of the mesh (nor of which nodes a
   path reaches: that is the nodes and their links alone) */
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  int same=a->node==b->node && a->reached==b->reached && a->boot==b->boot && a->region==b->region && a->memory==b->memory &&
    !memcmp(a->model,b->model,sizeof a->model) && !memcmp(a->uuid,b->uuid,sizeof a->uuid);
  for(uint32_t k=0;same && k<a->reached && k<MESH_STREAMS;k++)
    same=a->reach[k].node==b->reach[k].node && a->reach[k].streams==b->reach[k].streams && a->reach[k].bandwidth==b->reach[k].bandwidth;
  return same;
}
static void health_put(struct mesh_record *held,const struct mesh_record *record){
  mesh_notify(&M->health);
  held->sequence=record->sequence;held->health=record->health;
  mesh_notify(&M->health);
}
struct mesh_record *record_held(uint32_t node){
  for(uint64_t i=0;i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node==node)return mesh_records(M)+i;
  return NULL;
}
/* a record newer than the database's of its node taken in as its node's: where it changes only its health and sequence,
   the map's copy of it, where the map holds it, given them (health_put); else the map made again (map_publish) */
static void record_put(struct known *k,const struct mesh_record *record){
  const int same=k->record.boot && record_links_same(&k->record,record);
  k->record=*record;k->heard=now;
  struct mesh_record *held=record_held(record->node);
  if(!same)map_publish();
  else if(held)health_put(held,record);
}
void record_changed(void){
  struct mesh_record own={.node=M->node,.boot=M->boot,.sequence=++sequence,.region=wire.length-held_pages*M->pgsz,.memory=memory};
  memcpy(own.model,model,sizeof own.model);
  memcpy(own.uuid,node_uuid,sizeof own.uuid);
  own.reached=record_reach(own.reach,MESH_STREAMS);
  own.health=health;
  struct known *k=known_at(M->node);
  if(k)record_put(k,&own);
  else say("map: no memory to hold this node's record\n");
  for(uint32_t p=0;p<port_count;p++)if(ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
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
/* when the next health sample is due (the event loop's turn waits for it: mesh-link.c turn_due) */
uint64_t health_due(void){ return health_sent+MESH_HEALTH_NS; }
void health_turn(void){
  if(now-health_sent<MESH_HEALTH_NS)return;
  health_sent=now;health.available=available_now();
  record_changed();
}
/* AGENTS.md, "A node is its hardware": node `node`'s number heard from host `heard` (`what`, on `port`) where host `held`
   holds it (this node, or the database's record of the node): two hosts share one number.  Two such cabled together
   never pair (mesh-link.c hello_heard, port_heard: a hello of this node's own number), and elsewhere the newer record of
   the two stands for both in every map, each host's links joining only where its record stands; so it is told loudly,
   with both hosts, once for each host so heard (`told`, the last told) */
void number_shared(uint32_t node,const uint8_t *held,const uint8_t *heard,uint8_t *told,const struct port *port,const char *what){
  if(!memcmp(held,heard,16) || !memcmp(told,heard,16))return;
  memcpy(told,heard,16);
  char a[37],b[37];
  uuid_unparse_lower(held,a);uuid_unparse_lower(heard,b);
  say("node %u: two hosts share this number (AGENTS.md: a node is its host UUID): host %s holds it, and %s heard on link "
      "%u (%s) is host %s's; cabled together the two do not pair, and elsewhere the newer record of the two stands for both\n",
      node,a,what,port->index,port->name,b);
}
/* design/transport.md, RFC 2328 s13: a record heard, taken into the database where it is newer than the newest heard of
   its node (a node's later boot is always newer), held or not, and sent on to every other paired peer; one no newer is
   dropped and not sent on (so a record no path reaches never circulates).  A record of this node's own number is this
   node's own come back, or another host's (number_shared), and is not taken */
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard;
  if(bytes!=sizeof heard)return;
  memcpy(&heard,payload,sizeof heard);
  heard.model[sizeof heard.model-1]=0;
  if(!heard.node || heard.reached>MESH_STREAMS)return;
  if(heard.node==M->node){number_shared(heard.node,node_uuid,heard.uuid,own_told,from,"a record");return;}
  struct known *k=known_at(heard.node);
  if(!k){say("map: no memory to hold node %u's record\n",heard.node);return;}
  if(k->record.boot)number_shared(heard.node,k->record.uuid,heard.uuid,k->told,from,"a record");
  if(k->record.boot && (heard.boot<k->record.boot || (heard.boot==k->record.boot && heard.sequence<=k->record.sequence)))return;
  record_put(k,&heard);
  for(uint32_t p=0;p<port_count;p++)if(ports+p!=from && ports[p].greeted)port_say(ports+p,(struct message){.kind=RECORD},&heard,sizeof heard);
}
/* design/transport.md: every client slot read: a slot's process watched from the turn its slot is first
   seen (its exit retires every slot it holds, the event loop's EVFILT_PROC), one already gone retired at once, and on a
   liveness turn any client whose process exited retired; the rows of a retired client released (no DMA lands in a
   client's rows: a stream's rings are the bridge's), a stream a gone client held let go (mesh-link.c streams_let_go:
   its port's send thread ends the holding), and the streams held told (busy_tell) */
void clients_turn(int liveness){
  const pid_t self=getpid();
  int changed=0;
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
  }
  unreleased|=changed;
  if(unreleased){unreleased=0;mesh_retired_release(M);}
  streams_let_go();
  busy_tell();
}
