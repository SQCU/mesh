#include "mesh-bridge.h"

static void busy_tell(void){
  const uint64_t busy=lanes_busy();
  if(busy)atomic_store_explicit(&M->idle_since,0,memory_order_relaxed);
  else if(!atomic_load_explicit(&M->idle_since,memory_order_relaxed))atomic_store_explicit(&M->idle_since,clock_gettime_nsec_np(CLOCK_REALTIME),memory_order_relaxed);
  atomic_store_explicit(&M->busy,busy,memory_order_release);
}
static uint64_t available_now(void){
  int level=0;size_t size=sizeof level;
  return sysctlbyname("kern.memorystatus_level",&level,&size,NULL,0) || level<0?0:(uint64_t)level*(memory/100);
}
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

struct known { uint64_t heard,lost_at; uint32_t told; struct mesh_record record; };
_Static_assert(MESH_PORTS<=32,"a port a bit of `told`");
static struct known known[MESH_RECORDS];
static uint32_t known_count;
static uint32_t port_bit(const struct port *port){ return UINT32_C(1)<<(uint32_t)(port-ports); }
static struct mesh_map_copy *map_copy(uint64_t step){
  return mesh_map_copies(M)+((atomic_load_explicit(&M->map_version,memory_order_relaxed)+step)&1);
}
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
static void record_tell(struct known *k){
  const struct mesh_record record=k->record;
  for(uint32_t p=0;p<port_count;p++){
    if(!port_greeted(ports+p) || k->told>>p&1 || ports[p].peer==record.node)continue;
    port_say(ports+p,(struct message){.kind=RECORD},&record,sizeof record);
    k->told|=UINT32_C(1)<<p;
  }
}
void records_forget(const struct port *port){
  const uint32_t bit=port_bit(port);
  for(uint32_t k=0;k<known_count;k++)known[k].told&=~bit;
}
static void map_publish(int mesh){
  atomic_store_explicit(&M->map_version,atomic_load_explicit(&M->map_version,memory_order_relaxed)+1,memory_order_release);
  if(mesh)mesh_map_moved(M);
}
static int record_links_same(const struct mesh_record *a,const struct mesh_record *b){
  int same=a->node==b->node && a->reached==b->reached && a->boot==b->boot &&
    a->region==b->region && a->memory==b->memory && !memcmp(a->model,b->model,sizeof a->model);
  for(uint32_t k=0;same && k<a->reached && k<MESH_STREAMS;k++)
    same=a->reach[k].node==b->reach[k].node && a->reach[k].padding==b->reach[k].padding &&
         a->reach[k].bandwidth==b->reach[k].bandwidth;
  return same;
}
static void health_put(const struct mesh_record *record){
  const struct mesh_map_copy *was=map_copy(0);
  struct mesh_map_copy *next=map_copy(1);
  const uint32_t n=mesh_map_nodes(was);
  memcpy(next->record,was->record,(size_t)n*sizeof *next->record);
  next->nodes=n;
  for(uint32_t i=0;i<n;i++)if(next->record[i].node==record->node){next->record[i].sequence=record->sequence;next->record[i].health=record->health;}
  map_publish(0);
}
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
  for(uint32_t g=0;g<gains;g++)record_tell(known+gained[g]);
}
static void map_put(struct known *k,const struct mesh_record *record,const struct port *from){
  const struct mesh_record *mapped=record_of(record->node);
  k->record=*record;k->told=from?port_bit(from):0;
  if(mapped && record_links_same(mapped,record))health_put(record);
  else map_make();
}
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
void record_retell(void){
  const struct mesh_record *stored=record_of(M->node);
  if(!stored)return;
  struct mesh_record own;memcpy(&own,stored,sizeof own);
  for(uint32_t p=0;p<port_count;p++)if(port_greeted(ports+p))port_say(ports+p,(struct message){.kind=RECORD},&own,sizeof own);
}
uint64_t health_due(void){ return health_sent+MESH_HEALTH_NS; }
void health_turn(void){
  if(now-health_sent<MESH_HEALTH_NS)return;
  health_sent=now;health.available=available_now();
  record_changed();
}
void record_heard(struct port *from,const unsigned char *payload,uint32_t bytes){
  struct mesh_record heard={0};
  memcpy(&heard,payload,MIN((size_t)bytes,sizeof heard));
  if(!heard.node || heard.node==M->node)return;
  heard.reached=MIN(heard.reached,(uint32_t)MESH_STREAMS);
  heard.model[sizeof heard.model-1]=0;
  struct known *k=known_at(heard.node);
  if(!k)return;
  if(k->heard && heard.boot==k->record.boot && heard.sequence==k->record.sequence){k->told|=port_bit(from);return;}
  if(k->heard && (heard.boot<k->record.boot || (heard.boot==k->record.boot && heard.sequence<k->record.sequence)))return;
  k->heard=now;
  map_put(k,&heard,from);
  record_tell(k);
}
void clients_turn(void){
  busy_tell();
}
