/* design/declared-programs.md "Testing": two simulated nodes' bridges in one process.  Each node's region is made as
   the bridge makes it (mesh.h mesh_layout), named $MESH_REGION-style /mesh-sim-<node>, so a client process run with
   MESH_REGION set attaches it as it attaches a bridge's (libmesh, libnccl-mesh unchanged).  Each node's order toward the
   other is served by the bridge's own steps (mesh-order.h order_next, order_raise_pass) over the simulated wire of
   tests/wire.h, CABLES cables of LANES queue pairs, which enforces receive-before-send, each queue pair's post order and
   equal lengths, and that both ends took every piece for the same piece of the same transfer (any mismatch ends the
   simulator).  Queue pairs are served in a rotating, jittered order; with LAG set (a queue pair's index: direction,
   cable, lane), that one is served once in 64 turns at most, so it lags far behind the others; with BURST set, each queue
   pair served makes up to that many matches a turn (else 1 to 8 at random).

   With RESTARTS set, the two bridges' parents run as mesh-link.c order_turn runs them: no epoch at the start, the cables
   coming up one at a time (cable 0 first, then each STAGGER ms later), each end's epochs agreed by mesh-order.h
   order_agree_turn over ORDER messages (delivered while a cable is up at both ends, a turn later); a cable of an epoch
   lost restarts the order there (order_restart) and loses every cable toward the peer (order_lose); a restart a client's
   declaration asked (`restart`: it met a program of a client gone with pieces outstanding) does the same; every cable
   lost comes up again UP ms later (500; each further cable STAGGER ms after, 200), made new (a new identity, its lanes' counts 0, its port's child serving the
   generation its node is at then: mesh-link.c lanes_open). SIGUSR2 loses cable 1 (or 0 with one cable): node A's
   bridge notices at once, node B's NOTICE ms later (0: at once), so a client may declare at B before B restarts.
   Each segment's digest is told the peer as a SEGMENT (mesh-order.h order_segments_turn) and its pieces posted once both
   ends' agree. Every restart, epoch and loss is said on stderr. Each node's clients gone are retired, as the bridge does on their
   exit.

   usage: sim-wire NODE_A NODE_B CABLES LANES MIB   (runs until SIGTERM; prints "ready" once both regions are live) */
#include "wire.h"
#include "../mesh-dataflow.h"
#include "../mesh-call.h"
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>

static struct wire w;
static volatile sig_atomic_t losing;
static void lose_asked(int signal){ (void)signal; losing=1; }

/* ---- RESTARTS: the bridges' parents (mesh-link.c order_turn, order_lose, lanes_open) -------------------------------- */
static int restarts;
static uint64_t stagger=200,up_delay=500,notice;   /* milliseconds */
static struct order_agree agree[2];
/* ORDER messages toward node n, in flight: delivered a turn after sent, in order, while a cable is up at both ends */
static struct { struct order_message message[16]; uint64_t at[16]; uint32_t head,tail; } inbox[2];
static uint64_t identity[WIRE_CABLES],made,up_at[WIRE_CABLES],lost_at[2][WIRE_CABLES];
/* SEGMENT messages toward node n, in flight (as ORDER's), and each node's agreement of its segments */
static struct { struct segment_message message[64]; uint64_t at[64]; uint32_t head,tail; } segbox[2];
static struct order_segments segmented[2];
static uint64_t turn_now,ms_now;
static uint64_t clock_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000+(uint64_t)t.tv_nsec/1000000; }
static int cable_up(uint32_t c){ return !w.node[0].down[c] && !w.node[1].down[c]; }
static int any_up(void){ for(uint32_t c=0;c<w.cables;c++)if(cable_up(c))return 1; return 0; }
/* cable c lost (both ends: its queue pairs gone, every unit on them with them), made again `up_delay` turns on */
static void cable_lose(uint32_t c,const char *why){
  /* a cable down already (still to come up, or lost before) is made again after this loss too */
  if(w.node[0].down[c] && w.node[1].down[c]){up_at[c]=ms_now+up_delay+c*stagger;return;}
  fprintf(stderr,"sim-wire: turn %llu: cable %u (identity %llu) lost: %s\n",(unsigned long long)turn_now,c,(unsigned long long)identity[c],why);
  for(int n=0;n<2;n++){w.node[n].down[c]=1;lost_at[n][c]=ms_now;}
  for(int d=0;d<2;d++)for(uint32_t j=0;j<w.lanes;j++){
    struct wire_fifo *f=&w.fifo[d][c][j];f->sent_head=f->sent_tail;f->received_head=f->received_tail;
  }
  up_at[c]=ms_now+up_delay+c*stagger;
}
/* cable c made again: a new identity, its lanes' counts 0, each end's child serving the generation its node is at now */
static void cable_make(uint32_t c){
  identity[c]=++made;up_at[c]=UINT64_MAX;
  for(int n=0;n<2;n++){
    struct wire_node *v=w.node+n;
    v->down[c]=0;v->generation[c]=atomic_load(&v->x->generation);
    for(int way=0;way<2;way++)for(uint32_t j=0;j<w.lanes;j++){v->posted[way][c][j]=0;atomic_store(wire_done(v,way,c,j),0);v->post[way][c][j]=(struct order_post){0};}
    for(int way=0;way<2;way++)order_raiser_clear(&v->raiser[c][way]);
  }
  fprintf(stderr,"sim-wire: turn %llu: cable %u up (identity %llu)\n",(unsigned long long)turn_now,c,(unsigned long long)identity[c]);
}
/* node n's order restarted (order_restart) and every cable toward the peer lost (order_lose) */
static void order_restart_at(uint32_t n,const char *why){
  fprintf(stderr,"sim-wire: turn %llu: node %u: %s: the order restarts\n",(unsigned long long)turn_now,w.node[n].m->node,why);
  order_restart(w.node[n].m,w.node[n].x);agree[n]=(struct order_agree){0};segmented[n]=(struct order_segments){0};
  for(uint32_t c=0;c<w.cables;c++)cable_lose(c,"the order restarted");
}
struct frontier_context { uint32_t n; };
static uint64_t sim_frontier(const struct mesh_node *x,int way,void *context){
  const struct wire_node *v=w.node+((struct frontier_context *)context)->n;
  const uint32_t epochs=atomic_load(&x->epochs);
  if(!epochs)return 0;
  const struct mesh_epoch *last=x->epoch+(epochs<MESH_EPOCHS?epochs:MESH_EPOCHS)-1;
  uint64_t posted[WIRE_CABLES*WIRE_LANES];uint32_t k=0;
  for(uint32_t c=0;c<last->cables;c++)for(uint32_t j=0;j<last->cable[c].lanes;j++)posted[k++]=v->posted[way][last->cable[c].port][j];
  return order_posted_through(x,epochs,way,posted);
}
/* one turn of node n's parent (mesh-link.c order_turn) */
static void order_turn_at(uint32_t n){
  struct wire_node *v=w.node+n;struct mesh_node *x=v->x;
  /* a message in flight is lost with the last cable (the ports' sockets it went on) */
  if(!any_up()){inbox[n].head=inbox[n].tail;segbox[n].head=segbox[n].tail;}
  while(segbox[n].head!=segbox[n].tail && segbox[n].at[segbox[n].head%64]<turn_now){
    order_segments_heard(segmented+n,x,&segbox[n].message[segbox[n].head%64]);segbox[n].head++;
  }
  while(inbox[n].head!=inbox[n].tail && inbox[n].at[inbox[n].head%16]<turn_now){
    order_agree_heard(agree+n,x,&inbox[n].message[inbox[n].head%16]);inbox[n].head++;
  }
  if(atomic_load(&x->restart)){order_restart_at(n,"a declaration met a program of a client gone with pieces outstanding");return;}
  uint64_t wire[WIRE_CABLES];uint32_t port[WIRE_CABLES],lanes[WIRE_CABLES],up=0;
  for(uint32_t c=0;c<w.cables;c++)if(!v->down[c]){wire[up]=identity[c];port[up]=c;lanes[up]=w.lanes;up++;}
  for(uint32_t a=1;a<up;a++)for(uint32_t b=a;b>0 && wire[b]<wire[b-1];b--){
    uint64_t t=wire[b];wire[b]=wire[b-1];wire[b-1]=t;uint32_t q=port[b];port[b]=port[b-1];port[b-1]=q;
  }
  const uint32_t epochs=atomic_load(&x->epochs);
  const struct mesh_epoch *last=epochs?x->epoch+(epochs<MESH_EPOCHS?epochs:MESH_EPOCHS)-1:NULL;
  for(uint32_t c=0;last && c<last->cables;c++){
    uint32_t k=0;
    while(k<up && wire[k]!=last->cable[c].wire)k++;
    /* node B's bridge notices a loss `notice` ms late (node A's at once) */
    if(k==up && (n==0 || ms_now>=lost_at[n][last->cable[c].port]+notice)){order_restart_at(n,"a cable of its epoch lost");return;}
  }
  struct order_message told;struct frontier_context context={n};
  const int done=order_agree_turn(x,v->m->node<w.node[1-n].m->node,agree+n,up,wire,port,lanes,sim_frontier,&context,&told);
  if(done&ORDER_LOST){order_restart_at(n,"its agreed epoch names a cable not up here");return;}
  if(done&ORDER_SAY){
    if(any_up() && inbox[1-n].tail-inbox[1-n].head<16){
      inbox[1-n].message[inbox[1-n].tail%16]=told;inbox[1-n].at[inbox[1-n].tail%16]=turn_now;inbox[1-n].tail++;
    }else if(told.kind==ORDER_PROPOSE)agree[n].sent.cables=UINT32_MAX;
    else{order_restart_at(n,"an accept not sent");return;}
  }
  /* the segments' digests told and agreed (mesh-link.c order_turn) */
  struct segment_message said[ORDER_TELLS];uint32_t says=0;
  if(order_segments_turn(x,segmented+n,said,&says)&ORDER_LOST){
    order_restart_at(n,"its segments as declared here and there cannot agree");return;
  }
  for(uint32_t k=0;k<says;k++){
    if(!any_up() || segbox[1-n].tail-segbox[1-n].head>=64){order_segments_untold(segmented+n,said+k);continue;}
    segbox[1-n].message[segbox[1-n].tail%64]=said[k];segbox[1-n].at[segbox[1-n].tail%64]=turn_now;segbox[1-n].tail++;
  }
  if(done&ORDER_EXTENDED){
    const struct mesh_epoch *e=x->epoch+atomic_load(&x->epochs)-1;
    fprintf(stderr,"sim-wire: turn %llu: node %u: epoch %u over %u cable%s from pieces %llu sent, %llu received (%s)\n",(unsigned long long)turn_now,
            v->m->node,atomic_load(&x->epochs),e->cables,e->cables==1?"":"s",(unsigned long long)e->first[0],(unsigned long long)e->first[1],
            v->m->node<w.node[1-n].m->node?"proposed":"accepted");
  }
}
/* each node's clients gone retired (the bridge's watch of their exits: mesh-records.c, mesh-link.c mesh_retire) */
static void clients_reap(void){
  for(int n=0;n<2;n++){
    struct hdr *m=w.node[n].m;
    for(uint32_t s=0;s<MESH_CLIENTS;s++){
      const uint64_t owner=atomic_load(&((struct mesh_client *)((char *)m+m->client_off)+s)->owner);
      if(owner && !(owner&MESH_RETIRING) && kill((pid_t)(uint32_t)owner,0) && errno==ESRCH)mesh_retire(m,owner);
    }
  }
}
static void restarts_turn(void){
  if(losing){losing=0;cable_lose(w.cables>1?1:0,"SIGUSR2");}
  for(uint32_t c=0;c<w.cables;c++)if(up_at[c]<=ms_now)cable_make(c);
  for(uint32_t n=0;n<2;n++)order_turn_at(n);
}
static char names[2][64];
static volatile sig_atomic_t stopping,dumping;
static void stop(int signal){ (void)signal; stopping=1; }
static void dump_asked(int signal){ (void)signal; dumping=1; }
/* SIGUSR1: every queue pair's counts and depth, and each order's raised pieces, on stderr */
static void dump(void){
  for(uint32_t n=0;n<2;n++){
    const struct mesh_node *x=w.node[n].x;
    const struct order_segments *a=w.told?segmented+n:w.segments+n;
    fprintf(stderr,"node %u: segments %llu agreed %llu raised %llu/%llu; placing %llu told %llu scanned %llu open %llu/%llx/%llu\n",w.node[n].m->node,
            (unsigned long long)atomic_load(&x->segments),(unsigned long long)atomic_load(&x->agreed),
            (unsigned long long)atomic_load(&x->raised[0]),(unsigned long long)atomic_load(&x->raised[1]),(unsigned long long)a->placing,
            (unsigned long long)a->told,(unsigned long long)a->scanned,(unsigned long long)a->open[0],(unsigned long long)a->open[1],(unsigned long long)a->open[2]);
    for(uint64_t k=0;k<a->placing+1 && k<8;k++)
      fprintf(stderr,"  heard place %llu: %llu %llx committed %llu rounds %lld; local %llu\n",(unsigned long long)k,(unsigned long long)a->heard[k][0],
              (unsigned long long)a->heard[k][1],(unsigned long long)a->heard[k][2],(long long)a->heard[k][3],(unsigned long long)(k<a->placing?a->local[k]:UINT64_MAX));
    for(uint64_t s=0;s<atomic_load(&x->segments) && s<8;s++)
      fprintf(stderr,"  segment %llu digest %llx\n",(unsigned long long)s,(unsigned long long)order_segment(x,s)->digest);
    for(uint32_t c=0;c<w.cables;c++)for(int way=0;way<2;way++)for(uint32_t j=0;j<w.lanes;j++)
      fprintf(stderr,"  cable %u %s lane %u: posted %llu done %llu\n",c,way?"receive":"send",j,(unsigned long long)w.node[n].posted[way][c][j],
              (unsigned long long)atomic_load(wire_done(w.node+n,way,c,j)));
  }
}
static void node_make(uint32_t n,uint32_t node,uint32_t peer,uint64_t mib){
  snprintf(names[n],sizeof names[n],"/mesh-sim-%u",node);
  shm_unlink(names[n]);
  const uint32_t pg=(uint32_t)getpagesize();
  struct hdr geometry={0};
  const uint64_t length=mesh_layout(&geometry,pg,1,(uint32_t)((mib<<20)/pg));
  const int fd=shm_open(names[n],O_CREAT|O_EXCL|O_RDWR,0600);
  if(fd<0 || ftruncate(fd,(off_t)length)){perror("sim-wire: region");exit(1);}
  struct hdr *m=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  close(fd);
  if(m==MAP_FAILED){perror("sim-wire: mmap");exit(1);}
  *m=geometry;m->version=MESH_LAYOUT;m->node=node;m->boot=1;
  for(uint32_t r=0;r<mesh_rows(m);r++)atomic_store(&mesh_page(m)[r].mapping,MESH_ABSENT);
  atomic_store(&m->bridge_pid,(uint64_t)getpid());
  m->magic=MESH_MAGIC;
  struct mesh_node *x=mesh_node_at(m,0);
  order_restart(m,x);
  atomic_store(&x->node,peer);
  atomic_store(&x->bandwidth,UINT64_C(80000000000)*w.cables);
  w.node[n].m=m;w.node[n].x=x;
}
/* the map each region holds, as the bridges' floods leave it (mesh-records.c map_put, mesh-link.c record_reach): both
   nodes' records, each one's reach an entry a cable toward the other at the cable's 80 Gbit/s, in arena pages leased
   for them as the bridge leases its map's; the communicator's map is made from them (mesh_map) */
static void records_make(uint32_t n,uint32_t cables){
  struct hdr *m=w.node[n].m;
  struct mesh_ctx c={.M=m};
  const uint32_t pages=(uint32_t)((2*sizeof(struct mesh_record)+m->pgsz-1)/m->pgsz),page=mesh_arena_alloc(&c,pages,1);
  if(page==MESH_ABSENT){fprintf(stderr,"sim-wire: no pages for the map\n");exit(1);}
  struct mesh_record *r=(struct mesh_record *)mesh_at(m,page);
  for(uint32_t k=0;k<2;k++){
    r[k]=(struct mesh_record){.node=w.node[k].m->node,.reached=cables,.boot=1,.sequence=1};
    for(uint32_t e=0;e<cables;e++)r[k].reach[e]=(struct mesh_reach){.node=w.node[1-k].m->node,.padding=1,.bandwidth=UINT64_C(80000000000)};
  }
  atomic_store(&m->map_off,(uint64_t)((unsigned char *)r-(unsigned char *)m));
  atomic_store(&m->map_nodes,2);
  atomic_store(&m->map,2);
}
int main(int argc,char **argv){
  if(argc!=6){fprintf(stderr,"usage: sim-wire NODE_A NODE_B CABLES LANES MIB\n");return 2;}
  const uint32_t a=(uint32_t)strtoul(argv[1],NULL,0),b=(uint32_t)strtoul(argv[2],NULL,0);
  const uint32_t cables=(uint32_t)strtoul(argv[3],NULL,0),lanes=(uint32_t)strtoul(argv[4],NULL,0);
  if(!a || !b || a==b || !cables || cables>WIRE_CABLES || !lanes || lanes>WIRE_LANES)return 2;
  const char *lagging=getenv("LAG");
  const uint32_t lag=lagging && *lagging?(uint32_t)strtoul(lagging,NULL,0):UINT32_MAX;
  signal(SIGTERM,stop);signal(SIGINT,stop);signal(SIGUSR1,dump_asked);
  wire_make(&w,cables,lanes,getenv("ROOM")?(uint32_t)strtoul(getenv("ROOM"),NULL,0):256);
  /* BURST: every queue pair's matches a turn drained up to it (the wire's jitter then in which queue pairs are served) */
  w.burst=getenv("BURST")?strtoull(getenv("BURST"),NULL,0):0;
  node_make(0,a,b,strtoull(argv[5],NULL,0));node_make(1,b,a,strtoull(argv[5],NULL,0));
  for(uint32_t n=0;n<2;n++)records_make(n,cables);
  /* the one epoch both ends agree: every cable from piece 0, cables named by port c, identity c + 1 at both ends; with
     FRESH (an order just restarted, as the bridges re-form it: its first cable up alone, then every one), an epoch over
     cable 0 from piece 0 and then one over every cable from that same piece, before any client */
  restarts=getenv("RESTARTS") && *getenv("RESTARTS");
  if(getenv("STAGGER"))stagger=strtoull(getenv("STAGGER"),NULL,0);
  if(getenv("UP"))up_delay=strtoull(getenv("UP"),NULL,0);
  if(getenv("NOTICE"))notice=strtoull(getenv("NOTICE"),NULL,0);
  signal(SIGUSR2,lose_asked);
  if(restarts){
    /* no epoch: every cable comes up in turn, cable 0 first, and the bridges agree the epochs */
    for(int n=0;n<2;n++)w.node[n].served=1;
    w.told=1;
    for(uint32_t c=0;c<cables;c++){for(int n=0;n<2;n++)w.node[n].down[c]=1;up_at[c]=clock_ms()+c*stagger;}
  }
  else if(getenv("FRESH") && *getenv("FRESH") && cables>1){wire_epoch(&w,1,0);wire_epoch(&w,cables,0);}
  else wire_epoch(&w,cables,0);
  printf("ready\n");fflush(stdout);
  for(uint64_t turn=0;!stopping;turn++){
    if(dumping){dumping=0;dump();}
    turn_now=turn;
    if(restarts){if(!(turn%64))ms_now=clock_ms();restarts_turn();}
    if(!(turn%16384))clients_reap();
    wire_turn(&w,turn,lag,64);
    if(!(turn%4096))for(uint32_t n=0;n<2;n++)mesh_retired_release(w.node[n].m);
  }
  fprintf(stderr,"sim-wire: %llu pieces, %llu bytes crossed\n",(unsigned long long)w.matched,(unsigned long long)w.bytes);
  for(int n=0;n<2;n++)shm_unlink(names[n]);
  return 0;
}
