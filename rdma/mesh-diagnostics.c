#include "mesh-diagnostics.h"
#include "mesh-call.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* docs/measurement.md#the-quiet-probe: the quiet-to-active latency toward a NODE, its rounds each one transfer each
   way of one block, declared toward the node (design/declared-programs.md); the bridge alone places them on its
   cables.  Not an application routing or collective interface */
static uint64_t ns(void){ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static int order(const void *a,const void *b){ const uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b; return (x>y)-(x<y); }
static double at(uint64_t *v,size_t n,size_t num,size_t den){ qsort(v,n,sizeof *v,order); return n?v[n*num/den]/1e3:0; }
static int lost(_Atomic uint64_t *word){ return atomic_load_explicit(word,memory_order_acquire)==UINT64_MAX; }
static int wait_for(_Atomic uint64_t *word,uint64_t value){
  while(atomic_load_explicit(word,memory_order_acquire)<value)if(lost(word))return 1;
  return lost(word);
}
/* the probe's words (ready, sent, landed), its program's tables (a SEND and a receive a round) and spans, in its table
   lease */
struct probe { _Atomic uint64_t ready,sent,landed,padding; struct mesh_crossing table[]; };

/* No timed request reads or changes a descriptor. One immutable source is read repeatedly; every landing has its own
   backing. The probe is one declared program of one execution, `rounds` steps: step k a receive into block k+1 and a
   SEND of block 0 gated on `ready` reaching k+1, all issued at once (design/declared-programs.md) */
static int present(struct mesh_ctx *c,const struct mesh_order *o,uint32_t rounds,struct mesh_section data,struct probe *p,struct mesh_run *run){
  const uint64_t source=(char *)mesh_section_address(c,data,0)-(char *)c->M,base=(char *)p-(char *)c->M;
  struct mesh_dma_span *span=(struct mesh_dma_span *)(p->table+2*(size_t)rounds);
  const uint64_t spans=base+offsetof(struct probe,table)+2*(uint64_t)rounds*sizeof(struct mesh_crossing);
  for(uint32_t k=0;k<rounds;k++){
    span[2*k]=(struct mesh_dma_span){.offset=source,.bytes=data.bytes};
    span[2*k+1]=(struct mesh_dma_span){.offset=(char *)mesh_section_address(c,data,k+1)-(char *)c->M,.bytes=data.bytes};
    p->table[k]=(struct mesh_crossing){.spans=spans+2*k*sizeof *span,.units=1,.first=k,.step=k,.gate=base+offsetof(struct probe,ready),
                                       .gate_base=k,.done=base+offsetof(struct probe,sent),.done_value=k+1};
    p->table[rounds+k]=(struct mesh_crossing){.spans=spans+(2*k+1)*sizeof *span,.units=1,.first=k,.step=k,
                                              .done=base+offsetof(struct probe,landed),.done_value=k+1};
  }
  const uint64_t tables=base+offsetof(struct probe,table);
  const struct mesh_declaration program={.key=1,.period=rounds,.count=1,.table={tables,tables+rounds*sizeof(struct mesh_crossing)},
                                         .transfers={rounds,rounds},.pieces={rounds,rounds}};
  const int error=mesh_declare(c,o,&program,run);
  if(!error)mesh_issue(run,rounds);
  return error;
}
static void release_probe(struct mesh_ctx *c,struct mesh_run *run,struct mesh_section data,struct mesh_section table){
  /* the probe ended (its program closed), and a lease its program still names held by the bridge until it is done (mesh-dataflow.c mesh_pinned) */
  if(run->issued)mesh_close(c,run);
  if(table.count)mesh_lease_return(c,table.first);
  for(uint32_t k=0;k<data.count;k++)mesh_lease_return(c,data.first+k*data.stride);
  mesh_detach(c);
}

/* docs/measurement.md#the-quiet-probe */
int mesh_diagnostic_quiet(int argc,char **argv){
  struct mesh_ctx c={.fd=-1};
  if(argc<3 || mesh_attach(&c)){fprintf(stderr,"quiet list | quiet run ROLE NODE ROUNDS GAP,...: no region\n");return 2;}
  if(!strcmp(argv[1],"list")){
    struct mesh_reach reach[MESH_NODES];
    const uint32_t n=mesh_nodes_reached(c.M,reach,MESH_NODES);
    for(uint32_t i=0;i<n;i++)printf("%u\n",reach[i].node);
    mesh_detach(&c);
    return 0;
  }
  if(argc<6){mesh_detach(&c);return 2;}
  const int role=atoi(argv[2]);
  const uint32_t node=(uint32_t)strtoul(argv[3],NULL,10);
  const uint32_t rounds=(uint32_t)strtoul(argv[4],NULL,10);
  uint64_t gaps[16];
  uint32_t ngaps=0;
  for(char *g=strtok(argv[5],",");g && ngaps<16;g=strtok(NULL,","))gaps[ngaps++]=strtoull(g,NULL,10)*1000;
  const uint64_t page=(uint64_t)c.M->block*c.M->pgsz;
  struct mesh_section data={0},table={0};
  struct mesh_order o;
  struct mesh_run run={0};
  if(role<0 || role>1 || !rounds || !ngaps || rounds>=4096 || mesh_order_open(&c,node,&o) ||
     mesh_section_create(&c,page,rounds+1,&data) ||
     mesh_section_create(&c,sizeof(struct probe)+2*rounds*(sizeof(struct mesh_crossing)+sizeof(struct mesh_dma_span)),1,&table)){
    fprintf(stderr,"quiet: no lease, no order toward node %u, or rounds past %u\n",node,4096);
    release_probe(&c,&run,data,table);
    return 3;
  }
  struct probe *v=mesh_section_address(&c,table,0);
  memset(v,0,sizeof *v);
  memset(mesh_section_address(&c,data,0),0x5a,page);
  if(present(&c,&o,rounds,data,v,&run)){
    fprintf(stderr,"quiet: a transfer refused\n");release_probe(&c,&run,data,table);return 4;
  }
  uint64_t *done=calloc(rounds,sizeof *done),*trip=calloc(rounds,sizeof *trip),*a=calloc(rounds,sizeof *a),*b=calloc(rounds,sizeof *b);
  uint32_t *gap=calloc(rounds,sizeof *gap);
  if(!done || !trip || !gap || !a || !b){
    free(done);free(trip);free(gap);free(a);free(b);release_probe(&c,&run,data,table);return 3;
  }
  int failed=0;
  for(uint32_t k=0;k<rounds && !failed;k++){
    gap[k]=k%ngaps;
    if(role){
      failed=wait_for(&v->landed,k+1);
      if(failed)break;
      const uint64_t t0=ns();
      atomic_store_explicit(&v->ready,k+1,memory_order_release);
      failed|=wait_for(&v->sent,k+1);
      done[k]=ns()-t0;
    }else{
      const uint64_t from=ns();
      while(ns()-from<gaps[gap[k]]){}
      const uint64_t t0=ns();
      atomic_store_explicit(&v->ready,k+1,memory_order_release);
      failed=wait_for(&v->sent,k+1);
      done[k]=ns()-t0;
      failed|=wait_for(&v->landed,k+1);
      trip[k]=ns()-t0;
    }
  }
  if(failed){
    fprintf(stderr,"quiet: the order toward node %u was lost\n",node);
    free(done);free(trip);free(gap);free(a);free(b);release_probe(&c,&run,data,table);return 5;
  }
  for(uint32_t g=0;g<ngaps;g++){
    size_t n=0;
    for(uint32_t k=8;k<rounds;k++)if(gap[k]==g){a[n]=done[k];b[n++]=trip[k];}
    printf("{\"scope\": \"node_quiet_probe\", \"peer\": %u, \"payload_bytes\": %llu, \"role\": %d, \"node\": %u, \"quiet_microseconds\": %llu, \"trips\": %zu, \"send_p10\": %.2f, \"send_median\": %.2f, \"send_p90\": %.2f",
           node,(unsigned long long)page,role,c.M->node,(unsigned long long)(role?0:gaps[g]/1000),n,at(a,n,1,10),at(a,n,1,2),at(a,n,9,10));
    if(!role)printf(", \"trip_p10\": %.2f, \"trip_median\": %.2f, \"trip_p90\": %.2f",at(b,n,1,10),at(b,n,1,2),at(b,n,9,10));
    printf("}\n");
    if(role)break;
  }
  free(done);free(trip);free(gap);free(a);free(b);release_probe(&c,&run,data,table);
  return 0;
}

// docs/measurement.md#the-retire-run
#include "mesh-dataflow.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

int mesh_diagnostic_retire(int n) {
  if (n < 1) return 2;
  const int fd = shm_open(mesh_name(), O_RDONLY);
  struct stat shape;
  if (fd < 0 || fstat(fd, &shape)) return 1;
  const struct hdr *m = mmap(NULL, (size_t)shape.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) return 1;
  for (int i = 0; i < n; i++) {
    int p[2];
    if (pipe(p)) return 1;
    const pid_t pid = fork();
    if (!pid) {
      struct mesh_ctx c = {.fd = -1};
      uint64_t client = mesh_attach(&c) ? 0 : c.client;
      (void)!write(p[1], &client, sizeof client);
      _exit(0);
    }
    uint64_t client = 0;
    (void)!read(p[0], &client, sizeof client);
    close(p[0]);
    close(p[1]);
    waitpid(pid, NULL, 0);
    const uint64_t ended = clock_gettime_nsec_np(CLOCK_MONOTONIC);
    const _Atomic uint64_t *owner = &((const struct mesh_client *)((const char *)m + m->client_off) + (client >> 32 & 63))->owner;
    uint64_t at = ended;
    while (client && atomic_load(owner) == client && (at = clock_gettime_nsec_np(CLOCK_MONOTONIC)) - ended < UINT64_C(10000000000)) usleep(200);
    printf("{\"i\": %d, \"slot\": %llu, \"retired_ms\": %.3f}\n", i, (unsigned long long)(client >> 32 & 63), client ? (double)(at - ended) / 1e6 : -1.0);
    fflush(stdout);
    usleep(100000 + (useconds_t)(arc4random_uniform(3000)) * 1000);
  }
  return 0;
}
