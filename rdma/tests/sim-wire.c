/* design/switched-multipath-dma.md "Tests": two simulated nodes' bridges in one process.  Each node's region is made as
   the bridge makes it (mesh.h mesh_layout), named $MESH_REGION-style /mesh-sim-<node>, so a client process run with
   MESH_REGION set attaches it as it attaches a bridge's (libmesh, libnccl-mesh unchanged).  Each node's switch is served
   by the bridge's own steps (mesh-switched.h: staging, forwarding by name, delivery, the estimators, the refresh) over a
   simulated wire of CABLES cables of LANES queue pairs, as the hardware is: a queue pair's SENDs fill the receives posted
   on it strictly in their order, a SEND waits until a receive is posted (the transport's credit), and every SEND and
   receive is one whole slot (one frame count).  Queue pairs are served in a rotating, jittered order; with LAG set (a
   queue pair's index: direction, cable, lane), that one is served once in 64 turns at most, so it lags far behind the
   others (the switch moves pieces off it: its queue counts in every draw); with BURST set, each queue pair served makes
   up to that many matches a turn (else 1 to 8 at random).  The refresh runs every REFRESH turns (default 64).

   SIGUSR2 loses cable 1 (or 0 with one cable) at both ends, as a link lost: its queue pairs and every unit on them go
   (the receives posted given back to their pools, the SENDs not yet matched switched again from their slots: a piece
   that may have crossed is sent again, and its destination counts each unit once), and it comes up again UP ms later
   (500).  Every loss is said on stderr.  Each node's clients gone are retired, as the bridge does on their exit.

   usage: sim-wire NODE_A NODE_B CABLES LANES MIB   (runs until SIGTERM; prints "ready" once both regions are live) */
#include "../mesh-switched.h"
#include "../mesh-dataflow.h"
#include "../mesh-call.h"
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <sys/mman.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>

enum { CABLES=8, LANES=8, POSTED=16 };
struct qp { uint32_t send[MESH_SWITCH_RING],receive[POSTED*4]; uint64_t sends_head,sends_tail,head,tail; };
struct node { struct hdr *m; struct mesh_switch *sw; uint32_t *held,holds,room; };
static struct node node[2];
static struct qp qp[2][CABLES][LANES];   /* qp[d][c][j]: node d's SENDs on cable c lane j, node 1 - d's receives there */
static uint32_t cables,lanes;static uint8_t down[CABLES];static uint64_t up_at[CABLES];
static uint64_t matched,bytes,random_state=0x9e3779b97f4a7c15u,burst;
static uint64_t next_random(void){ random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return random_state; }
static uint64_t clock_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000+(uint64_t)t.tv_nsec/1000000; }
static uint64_t clock_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000u+(uint64_t)t.tv_nsec; }
static uint32_t lane_of(uint32_t c,uint32_t j){ return c*2*MESH_LINK_STREAMS+j; }
static void hold(struct node *v,uint32_t s){
  if(v->holds==v->room){v->room=v->room?2*v->room:256;v->held=realloc(v->held,v->room*sizeof *v->held);if(!v->held)exit(1);}
  v->held[v->holds++]=s;
}
/* a SEND of slot s on lane l of a lost cable: what it carried held for forwarding (mesh_switch_again) */
static void again(struct node *v,uint32_t s,uint32_t l){
  const uint32_t t=mesh_switch_again(v->m,v->sw,s,l);
  if(t!=UINT32_MAX)hold(v,t);
}
static void cable_lose(uint32_t c){
  if(down[c])return;
  fprintf(stderr,"sim-wire: cable %u lost\n",c);
  down[c]=1;up_at[c]=clock_ms()+(getenv("UP")?strtoull(getenv("UP"),NULL,0):500);
  for(int d=0;d<2;d++)for(uint32_t j=0;j<lanes;j++){
    struct node *v=node+d,*r=node+1-d;
    const uint32_t l=lane_of(c,j);
    atomic_store(&v->sw->lane[l].live,0);
    struct qp *q=&qp[d][c][j];
    for(;q->sends_head<q->sends_tail;q->sends_head++)again(v,q->send[q->sends_head%MESH_SWITCH_RING],l);
    for(uint32_t s;(s=mesh_switch_peek(v->sw,l))!=UINT32_MAX;){mesh_switch_pop(v->sw,l);again(v,s,l);}
    atomic_store(&v->sw->lane[l].queued,0);
    for(;q->head<q->tail;q->head++)mesh_switch_give(r->sw,q->receive[q->head%(POSTED*4)]);
  }
}
static void cable_make(uint32_t c){
  fprintf(stderr,"sim-wire: cable %u up\n",c);
  down[c]=0;up_at[c]=UINT64_MAX;
  for(int d=0;d<2;d++)for(uint32_t j=0;j<lanes;j++)atomic_store(&node[d].sw->lane[lane_of(c,j)].live,1);
}
/* node n's child pass: what its programs release staged and switched, what it holds retried, its lanes' outboxes
   moved onto their queue pairs, every queue pair's receives at the far end topped up */
static void pass(uint32_t n){
  struct node *v=node+n;
  const uint64_t now=clock_ns();
  for(uint32_t k=0;k<MESH_NODES;k++){
    struct mesh_node *x=mesh_node_at(v->m,k);
    if(!atomic_load(&x->node))continue;
    for(uint32_t p=0;p<MESH_PROGRAMS;p++){
      struct mesh_declared *d=x->program+p;
      if(!mesh_switch_owned(d))continue;
      uint32_t s;
      if(mesh_switch_credit(v->m,v->sw,x,d,now,&s) && !mesh_switch_forward(v->m,v->sw,s,now,MESH_SWITCH_LOGS-1))hold(v,s);
      for(int burst_=0;burst_<16 && mesh_switch_stage(v->m,v->sw,x,d,now,&s)==1;burst_++)if(!mesh_switch_forward(v->m,v->sw,s,now,MESH_SWITCH_LOGS-1))hold(v,s);
    }
  }
  uint32_t kept=0;
  for(uint32_t i=0;i<v->holds;i++)if(!mesh_switch_forward(v->m,v->sw,v->held[i],now,MESH_SWITCH_LOGS-1))v->held[kept++]=v->held[i];
  v->holds=kept;
  for(uint32_t c=0;c<cables;c++)for(uint32_t j=0;j<lanes;j++){
    if(down[c])continue;
    struct qp *q=&qp[n][c][j];struct node *r=node+1-n;
    while(q->tail-q->head<POSTED){
      const uint32_t s=mesh_switch_take(r->sw);
      if(s==UINT32_MAX)break;
      atomic_store(&r->sw->slot[s].state,SLOT_RECEIVE);r->sw->slot[s].lane=lane_of(c,j);
      q->receive[q->tail++%(POSTED*4)]=s;
    }
    for(uint32_t s;q->sends_tail-q->sends_head<MESH_SWITCH_RING && (s=mesh_switch_peek(v->sw,lane_of(c,j)))!=UINT32_MAX;){
      mesh_switch_pop(v->sw,lane_of(c,j));
      q->send[q->sends_tail++%MESH_SWITCH_RING]=s;
    }
  }
}
/* node d's queue pair (c, j): each SEND at its head fills the receive at its head */
static void match(uint32_t d,uint32_t c,uint32_t j,uint64_t most){
  struct qp *q=&qp[d][c][j];struct node *v=node+d,*r=node+1-d;
  if(down[c])return;
  for(;most && q->sends_head!=q->sends_tail && q->head!=q->tail;most--){
    const uint32_t s=q->send[q->sends_head++%MESH_SWITCH_RING],into=q->receive[q->head++%(POSTED*4)];
    memcpy(mesh_switch_slot_at(r->m,r->sw,into),mesh_switch_slot_at(v->m,v->sw,s),v->sw->slot_bytes);
    atomic_thread_fence(memory_order_release);
    const uint64_t now=clock_ns();
    mesh_switch_sent(v->sw,lane_of(c,j),s,v->sw->slot_bytes,now,now,now+1);
    if(!mesh_switch_landed(r->m,r->sw,lane_of(c,j),into,now,MESH_SWITCH_LOGS-1))hold(r,into);
    matched++;bytes+=v->sw->slot_bytes;
  }
}
static char names[2][64];
static volatile sig_atomic_t stopping,dumping,losing;
static void stop(int signal){ (void)signal; stopping=1; }
static void dump_asked(int signal){ (void)signal; dumping=1; }
static void lose_asked(int signal){ (void)signal; losing=1; }
static void dump(void){
  for(uint32_t n=0;n<2;n++){
    fprintf(stderr,"node %u: %llu slots free of %u, %u held\n",node[n].m->node,(unsigned long long)atomic_load(&node[n].sw->free_count),node[n].sw->slots,node[n].holds);
    for(uint32_t c=0;c<cables;c++)for(uint32_t j=0;j<lanes;j++){
      const struct qp *q=&qp[n][c][j];
      fprintf(stderr,"  cable %u lane %u: %llu SENDs waiting, %llu receives posted at the peer, outbox %llu\n",c,j,(unsigned long long)(q->sends_tail-q->sends_head),
              (unsigned long long)(q->tail-q->head),(unsigned long long)(atomic_load(&node[n].sw->lane[lane_of(c,j)].tail)-atomic_load(&node[n].sw->lane[lane_of(c,j)].head)));
    }
  }
}
static void node_make(uint32_t n,uint32_t me,uint32_t peer,uint64_t mib){
  snprintf(names[n],sizeof names[n],"/mesh-sim-%u",me);
  shm_unlink(names[n]);
  const uint32_t pg=(uint32_t)getpagesize();
  struct hdr geometry={0};
  const uint64_t length=mesh_layout(&geometry,pg,1,(uint32_t)((mib<<20)/pg));
  const int fd=shm_open(names[n],O_CREAT|O_EXCL|O_RDWR,0600);
  if(fd<0 || ftruncate(fd,(off_t)length)){perror("sim-wire: region");exit(1);}
  struct hdr *m=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  close(fd);
  if(m==MAP_FAILED){perror("sim-wire: mmap");exit(1);}
  *m=geometry;m->version=MESH_LAYOUT;m->node=me;m->boot=1;
  for(uint32_t r=0;r<mesh_rows(m);r++)atomic_store(&mesh_page(m)[r].mapping,MESH_ABSENT);
  atomic_store(&m->bridge_pid,(uint64_t)getpid());
  m->magic=MESH_MAGIC;
  struct mesh_node *x=mesh_node_at(m,0);
  atomic_store(&x->generation,1);atomic_store(&x->node,peer);atomic_store(&x->live,1);
  atomic_store(&x->bandwidth,UINT64_C(80000000000)*cables);
  /* the switch's pool, as the bridge leases it (mesh-link.c switch_make) */
  struct mesh_ctx ctx={.M=m};
  const uint64_t slot=mesh_switch_slot_bytes(m);
  const uint32_t slots=1024,pages=(uint32_t)((slots*slot+pg-1)/pg),page=mesh_arena_alloc(&ctx,pages,1);
  if(page==MESH_ABSENT){fprintf(stderr,"sim-wire: no pages for the pool\n");exit(1);}
  struct mesh_switch *sw=mesh_switch_of(m);
  mesh_switch_init(sw,slots,slot,m->data_off+(uint64_t)page*pg);
  for(uint32_t c=0;c<cables;c++)for(uint32_t j=0;j<lanes;j++){atomic_store(&sw->lane[lane_of(c,j)].peer,peer);atomic_store(&sw->lane[lane_of(c,j)].live,1);}
  const uint32_t nodes[2]={me,peer};
  const struct mesh_switch_fabric_link links[2]={{me,peer,cables*lanes,10.0*cables*lanes,6000},{peer,me,cables*lanes,10.0*cables*lanes,6000}};
  mesh_switch_view_set(sw,me,nodes,2,links,2);
  node[n]=(struct node){m,sw,NULL,0,0};
}
/* the map each region holds, as the bridges' floods leave it (mesh-records.c map_put, mesh-link.c record_reach): both
   nodes' records, each one's reach an entry a cable toward the other at the cable's 80 Gbit/s */
static void records_make(uint32_t n){
  struct hdr *m=node[n].m;
  struct mesh_ctx c={.M=m};
  const uint32_t pages=(uint32_t)((2*sizeof(struct mesh_record)+m->pgsz-1)/m->pgsz),page=mesh_arena_alloc(&c,pages,1);
  if(page==MESH_ABSENT){fprintf(stderr,"sim-wire: no pages for the map\n");exit(1);}
  struct mesh_record *r=(struct mesh_record *)mesh_at(m,page);
  for(uint32_t k=0;k<2;k++){
    r[k]=(struct mesh_record){.node=node[k].m->node,.reached=cables,.boot=1,.sequence=1};
    for(uint32_t e=0;e<cables;e++)r[k].reach[e]=(struct mesh_reach){.node=node[1-k].m->node,.padding=1,.bandwidth=UINT64_C(80000000000)};
  }
  atomic_store(&m->map_off,(uint64_t)((unsigned char *)r-(unsigned char *)m));
  atomic_store(&m->map_nodes,2);
  atomic_store(&m->map,2);
}
static void clients_reap(void){
  for(int n=0;n<2;n++){
    struct hdr *m=node[n].m;
    for(uint32_t s=0;s<MESH_CLIENTS;s++){
      const uint64_t owner=atomic_load(&((struct mesh_client *)((char *)m+m->client_off)+s)->owner);
      if(owner && !(owner&MESH_RETIRING) && kill((pid_t)(uint32_t)owner,0) && errno==ESRCH)mesh_retire(m,owner);
    }
  }
}
int main(int argc,char **argv){
  if(argc!=6){fprintf(stderr,"usage: sim-wire NODE_A NODE_B CABLES LANES MIB\n");return 2;}
  const uint32_t a=(uint32_t)strtoul(argv[1],NULL,0),b=(uint32_t)strtoul(argv[2],NULL,0);
  cables=(uint32_t)strtoul(argv[3],NULL,0);lanes=(uint32_t)strtoul(argv[4],NULL,0);
  if(!a || !b || a==b || !cables || cables>CABLES || !lanes || lanes>LANES || lanes>2*MESH_LINK_STREAMS)return 2;
  const char *lagging=getenv("LAG");
  const uint32_t lag=lagging && *lagging?(uint32_t)strtoul(lagging,NULL,0):UINT32_MAX;
  const uint64_t refresh=getenv("REFRESH")?strtoull(getenv("REFRESH"),NULL,0):64;
  burst=getenv("BURST")?strtoull(getenv("BURST"),NULL,0):0;
  signal(SIGTERM,stop);signal(SIGINT,stop);signal(SIGUSR1,dump_asked);signal(SIGUSR2,lose_asked);
  node_make(0,a,b,strtoull(argv[5],NULL,0));node_make(1,b,a,strtoull(argv[5],NULL,0));
  for(uint32_t n=0;n<2;n++){records_make(n);up_at[0]=UINT64_MAX;}
  for(uint32_t c=0;c<CABLES;c++)up_at[c]=UINT64_MAX;
  for(uint32_t n=0;n<2;n++)mesh_switch_refresh(node[n].sw,NULL,32,1,n,(double)node[n].sw->slot_bytes,0);
  printf("ready\n");fflush(stdout);
  const uint32_t q=2*cables*lanes;
  for(uint64_t turn=0;!stopping;turn++){
    if(dumping){dumping=0;dump();}
    if(losing){losing=0;cable_lose(cables>1?1:0);}
    if(!(turn%64)){const uint64_t ms=clock_ms();for(uint32_t c=0;c<cables;c++)if(up_at[c]<=ms)cable_make(c);}
    if(!(turn%refresh))for(uint32_t n=0;n<2;n++)mesh_switch_refresh(node[n].sw,NULL,32,0.5f,turn^n,(double)node[n].sw->slot_bytes,0);
    if(!(turn%16384))clients_reap();
    for(uint32_t n=0;n<2;n++)pass(n);
    for(uint32_t k=0;k<q;k++){
      const uint32_t at=(uint32_t)((k+turn)%q);
      const uint64_t r=next_random();
      if(r%4==0)continue;
      if(at==lag && (r>>8)%64)continue;
      match(at/(cables*lanes),at/lanes%cables,at%lanes,burst?burst:1+(r>>20)%8);
    }
    if(!(turn%4096))for(uint32_t n=0;n<2;n++)mesh_retired_release(node[n].m);
  }
  fprintf(stderr,"sim-wire: %llu SENDs, %llu bytes crossed\n",(unsigned long long)matched,(unsigned long long)bytes);
  for(int n=0;n<2;n++)shm_unlink(names[n]);
  return 0;
}
