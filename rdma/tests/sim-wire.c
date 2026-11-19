/* design/node-addressed-transfers.md "Testing": two simulated nodes' bridges in one process.  Each node's region is
   made as the bridge makes it (mesh.h mesh_layout), named $MESH_REGION-style /mesh-sim-<node>, so a client process run
   with MESH_REGION set attaches it as it attaches a bridge's (libmesh, libnccl-mesh unchanged).  Each node's order
   toward the other is served by the bridge's own walkers (mesh-order.h order_walk, order_completed, order_retire),
   over a simulated wire of CABLES cables of LANES queue pairs each way that enforces what the hardware does: a receive
   names its bytes before the SEND that fills it, each queue pair's SENDs fill its posted receives strictly in post
   order, a SEND waits (the transport's holding) until a receive is posted, and the two lengths must be equal (any
   mismatch ends the simulator: the two ends' orders differed).  Queue pairs are served in a rotating, jittered order,
   so completions arrive out of order across them, as on several cables.

   usage: sim-wire NODE_A NODE_B CABLES LANES MIB   (runs until SIGTERM; prints "ready" once both regions are live) */
#include "../mesh-order.h"
#include "../mesh-dataflow.h"
#include "../mesh-call.h"
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

enum { QUEUE = 64, MOST_CABLES = 8, MOST_LANES = 8 };
struct post { struct mesh_dma_span span; uint64_t slot; };
struct fifo { struct post sends[QUEUE], receives[QUEUE]; uint32_t sent_head,sent_tail,received_head,received_tail; };
static struct node { struct hdr *m; struct mesh_ctx bridge; struct mesh_node *x; uint64_t dud; char name[64]; } nodes[2];
/* the FIFO of direction d (from node d to node 1-d), cable c, queue pair j */
static struct fifo fifos[2][MOST_CABLES][MOST_LANES];
static struct order_walker walkers[2][MOST_CABLES][2][MOST_LANES];
static uint32_t cables,lanes;
static volatile sig_atomic_t stopping;
static uint64_t matched,bytes;

struct port_context { uint32_t node,cable; };
static int post(void *context,int way,uint32_t lane,struct mesh_dma_span span,int dud,uint64_t slot){
  const struct port_context *p=context;
  if(dud)span.offset=nodes[p->node].dud;
  struct fifo *f=&fifos[way?1-p->node:p->node][p->cable][lane];
  if(way){
    if(f->received_tail-f->received_head==QUEUE)return 0;
    f->receives[f->received_tail++%QUEUE]=(struct post){span,slot};
  }else{
    if(f->sent_tail-f->sent_head==QUEUE)return 0;
    f->sends[f->sent_tail++%QUEUE]=(struct post){span,slot};
  }
  return 1;
}
/* the wire: each SEND at a queue pair's head fills the receive at its head, the two ends' byte counts equal */
static void wire(uint32_t d,uint32_t c,uint32_t j){
  struct fifo *f=&fifos[d][c][j];
  while(f->sent_head!=f->sent_tail && f->received_head!=f->received_tail){
    const struct post s=f->sends[f->sent_head%QUEUE],r=f->receives[f->received_head%QUEUE];
    if(s.span.bytes!=r.span.bytes){
      fprintf(stderr,"sim-wire: node %u -> %u cable %u queue pair %u: a SEND of %llu bytes met a receive of %llu: the orders differ\n",
              nodes[d].m->node,nodes[1-d].m->node,c,j,(unsigned long long)s.span.bytes,(unsigned long long)r.span.bytes);
      exit(3);
    }
    memcpy((char *)nodes[1-d].m+r.span.offset,(const char *)nodes[d].m+s.span.offset,s.span.bytes);
    atomic_thread_fence(memory_order_release);
    f->sent_head++;f->received_head++;matched++;bytes+=s.span.bytes;
    order_completed(nodes[d].m,nodes[d].x,0,s.slot);
    order_completed(nodes[1-d].m,nodes[1-d].x,1,r.slot);
  }
}
static void stop(int signal){ (void)signal; stopping=1; }
static volatile sig_atomic_t dumping;
static void dump_asked(int signal){ (void)signal; dumping=1; }
/* SIGUSR1: every walker's place and every queue pair's depth, on stderr */
static void dump(void){
  for(uint32_t n=0;n<2;n++)for(uint32_t c=0;c<cables;c++)for(int way=0;way<2;way++)for(uint32_t j=0;j<lanes;j++){
    const struct order_walker *w=&walkers[n][c][way][j];
    fprintf(stderr,"node %u cable %u way %d lane %u: at %llu k %llu n %llu piece %llu posted %llu\n",n,c,way,j,
            (unsigned long long)w->at,(unsigned long long)w->k,(unsigned long long)w->n,(unsigned long long)w->piece,(unsigned long long)w->posted);
  }
  for(uint32_t d=0;d<2;d++)for(uint32_t c=0;c<cables;c++)for(uint32_t j=0;j<lanes;j++){
    const struct fifo *f=&fifos[d][c][j];
    if(f->sent_tail!=f->sent_head || f->received_tail!=f->received_head)
      fprintf(stderr,"fifo %u cable %u lane %u: %u sends, %u receives\n",d,c,j,f->sent_tail-f->sent_head,f->received_tail-f->received_head);
  }
}
static void node_make(struct node *n,uint32_t node,uint32_t peer,uint64_t mib){
  snprintf(n->name,sizeof n->name,"/mesh-sim-%u",node);
  shm_unlink(n->name);
  const uint32_t pg=(uint32_t)getpagesize();
  struct hdr geometry={0};
  const uint64_t length=mesh_layout(&geometry,pg,1,(uint32_t)((mib<<20)/pg));
  const int fd=shm_open(n->name,O_CREAT|O_EXCL|O_RDWR,0600);
  if(fd<0 || ftruncate(fd,(off_t)length)){perror("sim-wire: region");exit(1);}
  n->m=mmap(NULL,length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  close(fd);
  if(n->m==MAP_FAILED){perror("sim-wire: mmap");exit(1);}
  *n->m=geometry;n->m->version=MESH_LAYOUT;n->m->node=node;n->m->boot=1;
  for(uint32_t r=0;r<mesh_rows(n->m);r++)atomic_store(&mesh_page(n->m)[r].mapping,MESH_ABSENT);
  atomic_store(&n->m->bridge_pid,(uint64_t)getpid());
  n->m->magic=MESH_MAGIC;
  n->bridge=(struct mesh_ctx){.M=n->m,.len=length,.fd=-1};
  if(mesh_slot_take(&n->bridge,n->m)){fprintf(stderr,"sim-wire: no slot\n");exit(1);}
  const uint32_t ring=(uint32_t)((2*(uint64_t)MESH_RING*sizeof(struct mesh_slot)+pg-1)/pg);
  const uint32_t page=mesh_arena_alloc(&n->bridge,ring+1,1);
  if(page==MESH_ABSENT){fprintf(stderr,"sim-wire: no room for the rings\n");exit(1);}
  n->x=mesh_node_at(n->m,0);
  n->x->ring=n->m->data_off+(uint64_t)page*pg;
  n->dud=n->x->ring+(uint64_t)ring*pg;
  order_rings_new(n->m,n->x);
  atomic_store(&n->x->generation,1);
  atomic_store(&n->x->node,peer);
  atomic_store(&n->x->bandwidth,UINT64_C(80000000000)*cables);
}
int main(int argc,char **argv){
  if(argc!=6){fprintf(stderr,"usage: sim-wire NODE_A NODE_B CABLES LANES MIB\n");return 2;}
  const uint32_t a=(uint32_t)strtoul(argv[1],NULL,0),b=(uint32_t)strtoul(argv[2],NULL,0);
  cables=(uint32_t)strtoul(argv[3],NULL,0);lanes=(uint32_t)strtoul(argv[4],NULL,0);
  if(!a || !b || a==b || !cables || cables>MOST_CABLES || !lanes || lanes>MOST_LANES)return 2;
  signal(SIGTERM,stop);signal(SIGINT,stop);signal(SIGUSR1,dump_asked);
  node_make(nodes,a,b,strtoull(argv[5],NULL,0));node_make(nodes+1,b,a,strtoull(argv[5],NULL,0));
  /* the one epoch both ends agree: every cable from piece 0, cables named by port c, identity c + 1 at both ends */
  uint32_t port[MOST_CABLES],lane_count[MOST_CABLES];uint64_t identity[MOST_CABLES];
  for(uint32_t c=0;c<cables;c++){port[c]=c;lane_count[c]=lanes;identity[c]=c+1;}
  const uint64_t first[2]={0,0};
  for(int n=0;n<2;n++)order_extend(nodes[n].x,first,cables,port,lane_count,identity);
  printf("ready\n");fflush(stdout);
  struct port_context context[2][MOST_CABLES];
  for(uint32_t n=0;n<2;n++)for(uint32_t c=0;c<cables;c++)context[n][c]=(struct port_context){n,c};
  uint64_t pass=0,random=0x9e3779b97f4a7c15u;
  while(!stopping){
    pass++;
    if(dumping){dumping=0;dump();}
    for(uint32_t n=0;n<2;n++)for(uint32_t c=0;c<cables;c++)for(int way=0;way<2;way++)for(uint32_t j=0;j<lanes;j++)
      order_walk(nodes[n].m,nodes[n].x,way,c,j,&walkers[n][c][way][j],post,&context[n][c]);
    for(uint32_t k=0;k<2*cables*lanes;k++){
      random^=random<<13;random^=random>>7;random^=random<<17;
      const uint32_t at=(uint32_t)((k+pass)%(2*cables*lanes));
      if(random%4==0)continue;
      wire(at/(cables*lanes),at/lanes%cables,at%lanes);
    }
    for(uint32_t n=0;n<2;n++){
      for(int way=0;way<2;way++)order_retire(nodes[n].m,nodes[n].x,way);
      if(!(pass%4096))mesh_retired_release(nodes[n].m);
    }
  }
  fprintf(stderr,"sim-wire: %llu pieces, %llu bytes crossed\n",(unsigned long long)matched,(unsigned long long)bytes);
  for(int n=0;n<2;n++)shm_unlink(nodes[n].name);
  return 0;
}
