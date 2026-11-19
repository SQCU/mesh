/* design/declared-programs.md "Testing": two simulated nodes' bridges in one process.  Each node's region is made as
   the bridge makes it (mesh.h mesh_layout), named $MESH_REGION-style /mesh-sim-<node>, so a client process run with
   MESH_REGION set attaches it as it attaches a bridge's (libmesh, libnccl-mesh unchanged).  Each node's order toward the
   other is served by the bridge's own steps (mesh-order.h order_next, order_raise_pass) over the simulated wire of
   tests/wire.h, CABLES cables of LANES queue pairs, which enforces receive-before-send, each queue pair's post order and
   equal lengths, and that both ends took every piece for the same piece of the same transfer (any mismatch ends the
   simulator).  Queue pairs are served in a rotating, jittered order; with LAG set (a queue pair's index: direction,
   cable, lane), that one is served once in 64 turns at most, so it lags far behind the others.

   usage: sim-wire NODE_A NODE_B CABLES LANES MIB   (runs until SIGTERM; prints "ready" once both regions are live) */
#include "wire.h"
#include "../mesh-dataflow.h"
#include "../mesh-call.h"
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

static struct wire w;
static char names[2][64];
static volatile sig_atomic_t stopping,dumping;
static void stop(int signal){ (void)signal; stopping=1; }
static void dump_asked(int signal){ (void)signal; dumping=1; }
/* SIGUSR1: every queue pair's counts and depth, and each order's raised pieces, on stderr */
static void dump(void){
  for(uint32_t n=0;n<2;n++){
    const struct mesh_node *x=w.node[n].x;
    fprintf(stderr,"node %u: segments %llu raised %llu/%llu\n",w.node[n].m->node,(unsigned long long)atomic_load(&x->segments),
            (unsigned long long)atomic_load(&x->raised[0]),(unsigned long long)atomic_load(&x->raised[1]));
    for(uint32_t c=0;c<w.cables;c++)for(int way=0;way<2;way++)for(uint32_t j=0;j<w.lanes;j++)
      fprintf(stderr,"  cable %u %s lane %u: posted %llu done %llu\n",c,way?"receive":"send",j,(unsigned long long)w.node[n].posted[way][c][j],
              (unsigned long long)w.node[n].done[way][c][j]);
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
int main(int argc,char **argv){
  if(argc!=6){fprintf(stderr,"usage: sim-wire NODE_A NODE_B CABLES LANES MIB\n");return 2;}
  const uint32_t a=(uint32_t)strtoul(argv[1],NULL,0),b=(uint32_t)strtoul(argv[2],NULL,0);
  const uint32_t cables=(uint32_t)strtoul(argv[3],NULL,0),lanes=(uint32_t)strtoul(argv[4],NULL,0);
  if(!a || !b || a==b || !cables || cables>WIRE_CABLES || !lanes || lanes>WIRE_LANES)return 2;
  const char *lagging=getenv("LAG");
  const uint32_t lag=lagging && *lagging?(uint32_t)strtoul(lagging,NULL,0):UINT32_MAX;
  signal(SIGTERM,stop);signal(SIGINT,stop);signal(SIGUSR1,dump_asked);
  wire_make(&w,cables,lanes,256);
  node_make(0,a,b,strtoull(argv[5],NULL,0));node_make(1,b,a,strtoull(argv[5],NULL,0));
  /* the one epoch both ends agree: every cable from piece 0, cables named by port c, identity c + 1 at both ends */
  wire_epoch(&w,cables,0);
  printf("ready\n");fflush(stdout);
  for(uint64_t turn=0;!stopping;turn++){
    if(dumping){dumping=0;dump();}
    wire_turn(&w,turn,lag,64);
    if(!(turn%4096))for(uint32_t n=0;n<2;n++)mesh_retired_release(w.node[n].m);
  }
  fprintf(stderr,"sim-wire: %llu pieces, %llu bytes crossed\n",(unsigned long long)w.matched,(unsigned long long)w.bytes);
  for(int n=0;n<2;n++)shm_unlink(names[n]);
  return 0;
}
