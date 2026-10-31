#include "mesh-diagnostics.h"
#include "mesh-call.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* docs/measurement.md#the-quiet-probe: an explicitly selected physical WIRE's
   quiet-to-active latency, not an application routing or collective interface. */
static uint64_t ns(void){ return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static int order(const void *a,const void *b){ const uint64_t x=*(const uint64_t *)a,y=*(const uint64_t *)b; return (x>y)-(x<y); }
static double at(uint64_t *v,size_t n,size_t num,size_t den){ qsort(v,n,sizeof *v,order); return n?v[n*num/den]/1e3:0; }
static int lost(_Atomic uint64_t *word){ return atomic_load_explicit(word,memory_order_acquire)==UINT64_MAX; }
/* design/transport.md section 5: a wait on `landed` alone, within MESH_PLATFORM_NS, past which the probe ends, the wait
   named; the moment `sent` reaches `through` read as it passes (*at, 0 not yet), so no wait reads `sent` */
static int wait_for(const struct mesh_stream_view *v,uint64_t value,uint64_t through,uint64_t *at,const char *wait){
  const struct mesh_deadline deadline=mesh_deadline(wait,0);
  for(;;){
    if(!*at && atomic_load_explicit(v->sent,memory_order_acquire)>=through)*at=ns();
    if(atomic_load_explicit(v->landed,memory_order_acquire)>=value || lost(v->landed))return lost(v->landed);
    mesh_deadline_left(&deadline);
  }
}

/* No timed request reads or changes a descriptor. One immutable source is read
   repeatedly; every preposted landing has separate backing, so even two hardware
   lanes can never overwrite the same destination. Tables are local metadata.
   design/zorvek.md zinth: the probe is one call, its tables (mesh.h struct mesh_zwoggle: its SEND entry and its landing
   entry, then their spans) linked as the holding's first, the stream's `zwoggle`; its SENDs go as `ready` allows */
static void lay(struct mesh_ctx *c,const struct mesh_stream_view *v,uint32_t rounds,
                struct mesh_section data,struct mesh_section table){
  struct mesh_zwoggle *tables=mesh_section_address(c,table,0);
  struct mesh_zwoggle_entry *entries=(struct mesh_zwoggle_entry *)(tables+1);
  struct mesh_dma_span *spans=(struct mesh_dma_span *)(entries+2);
  const uint64_t source=mesh_page_offset(c->M,data.first,0);
  for(uint32_t k=0;k<rounds;k++){
    spans[k]=(struct mesh_dma_span){.offset=source,.bytes=data.bytes};
    spans[rounds+k]=(struct mesh_dma_span){
      .offset=mesh_page_offset(c->M,data.first+(k+1)*data.stride,0),.bytes=data.bytes};
  }
  const uint64_t metadata=mesh_page_offset(c->M,table.first,0),first=metadata+(uint64_t)((char *)spans-(char *)tables);
  entries[0]=(struct mesh_zwoggle_entry){.entry={.spans=first,.units=rounds},.relative=0};
  entries[1]=(struct mesh_zwoggle_entry){.entry={.spans=first+rounds*sizeof *spans,.units=rounds},.relative=0};
  atomic_store_explicit(&tables->next,0,memory_order_relaxed);
  tables->entries[0]=metadata+sizeof *tables;tables->entries[1]=tables->entries[0]+sizeof *entries;
  tables->count[0]=tables->count[1]=1;tables->bases=0;
  atomic_store_explicit(v->zwoggle,metadata,memory_order_release);
}

static void release_probe(struct mesh_ctx *c,uint32_t stream,int completed,
                          struct mesh_section data,struct mesh_section table){
  if(stream!=MESH_ABSENT){
    mesh_stream_give(c,stream);
    /* Give prevents future posts, but is not itself a CPU-reader completion.
       On error retain tables until the old client's owner has disappeared.
       A different owner implies that the bridge ended this holding first.
       These observations are local teardown, never remote receipt messages. */
    /* design/zorvek.md zinth: the bridge's walk reads the probe's tables until the holding ends, whatever completed,
       so they are kept until then on every path (the ring's entries were copies the bridge held) */
    (void)completed;
    const struct mesh_deadline deadline=mesh_deadline("quiet: the bridge's end of the probe's holding (release_probe)",0);
    while((atomic_load_explicit(&mesh_stream_at(c->M,stream)->owner,memory_order_acquire)&
          ~MESH_STREAM_GIVEN)==c->client){mesh_deadline_left(&deadline);usleep(1000);}
  }
  if(table.count)mesh_lease_return(c,table.first);
  for(uint32_t k=0;k<data.count;k++)mesh_lease_return(c,data.first+k*data.stride);
  mesh_detach(c);
}

/* docs/measurement.md#the-quiet-probe */
int mesh_diagnostic_quiet(int argc,char **argv){
  struct mesh_ctx c={.fd=-1};
  if(argc<3 || mesh_attach(&c)){fprintf(stderr,"quiet list PEER | quiet run ROLE WIRE ROUNDS GAP,...: no region\n");return 2;}
  if(!strcmp(argv[1],"list")){
    uint32_t node[MESH_STREAMS];
    uint64_t wire[MESH_STREAMS];
    const uint32_t n=mesh_streams_free(c.M,node,wire,MESH_STREAMS);
    for(uint32_t i=0;i<n;i++)if(node[i]==(uint32_t)strtoul(argv[2],NULL,10))printf("%llu\n",(unsigned long long)wire[i]);
    mesh_detach(&c);
    return 0;
  }
  if(argc<6){mesh_detach(&c);return 2;}
  const int role=atoi(argv[2]);
  const uint64_t wire=strtoull(argv[3],NULL,10);
  const uint32_t rounds=(uint32_t)strtoul(argv[4],NULL,10);
  uint64_t gaps[16];
  uint32_t ngaps=0;
  for(char *g=strtok(argv[5],",");g && ngaps<16;g=strtok(NULL,","))gaps[ngaps++]=strtoull(g,NULL,10)*1000;
  const uint64_t page=(uint64_t)c.M->block*c.M->pgsz;
  struct mesh_section data={0},table={0};
  uint32_t stream=MESH_ABSENT;
  struct mesh_stream_view v;
  if(role<0 || role>1 || !rounds || !ngaps ||
     mesh_section_create(&c,page,rounds+1,MESH_BACKING_PAGES,&data) ||
     mesh_section_create(&c,sizeof(struct mesh_zwoggle)+2*sizeof(struct mesh_zwoggle_entry)+2*rounds*sizeof(struct mesh_dma_span),1,
                         MESH_BACKING_LINEAR,&table) ||
     mesh_stream_take(&c,wire,&stream) || mesh_stream_view(&c,stream,&v)){
    fprintf(stderr,"quiet: no lease or no stream %llu\n",(unsigned long long)wire);
    release_probe(&c,stream,0,data,table);
    return 3;
  }
  memset(mesh_section_address(&c,data,0),0x5a,page);
  uint64_t *done=calloc(rounds,sizeof *done),*trip=calloc(rounds,sizeof *trip),*a=calloc(rounds,sizeof *a),*b=calloc(rounds,sizeof *b);
  uint32_t *gap=calloc(rounds,sizeof *gap);
  if(!done || !trip || !gap || !a || !b){
    free(done);free(trip);free(gap);free(a);free(b);release_probe(&c,stream,0,data,table);return 3;
  }
  lay(&c,&v,rounds,data,table);
  /* a SEND's time: its unit's `sent` seen while the next landing is awaited (the responder's, at its next round, so its
     last round's is not read), or that landing's time where `sent` had not moved by then */
  int failed=0;
  uint64_t t0=0;
  for(uint32_t k=0;k<rounds && !failed;k++){
    uint64_t seen=0;
    gap[k]=k%ngaps;
    if(role){
      failed=wait_for(&v,k+1,k,&seen,"quiet: the peer's unit landed (wait_for landed)");
      if(k)done[k-1]=(seen?seen:ns())-t0;
      if(failed)break;
    }else for(const uint64_t from=ns();ns()-from<gaps[gap[k]];){}
    t0=ns();
    atomic_store_explicit(v.ready,k+1,memory_order_release);
    if(!role){
      failed=wait_for(&v,k+1,k+1,&seen,"quiet: the peer's unit landed (wait_for landed)");
      done[k]=(seen?seen:ns())-t0;trip[k]=ns()-t0;
    }
  }
  if(failed){
    fprintf(stderr,"quiet: the link was lost\n");
    free(done);free(trip);free(gap);free(a);free(b);release_probe(&c,stream,0,data,table);return 5;
  }
  for(uint32_t g=0;g<ngaps;g++){
    size_t n=0;
    for(uint32_t k=8;k+(uint32_t)role<rounds;k++)if(gap[k]==g){a[n]=done[k];b[n++]=trip[k];}
    printf("{\"scope\": \"physical_link_quiet_probe\", \"wire\": \"%016llx\", \"payload_bytes\": %llu, \"role\": %d, \"node\": %u, \"quiet_microseconds\": %llu, \"trips\": %zu, \"send_p10\": %.2f, \"send_median\": %.2f, \"send_p90\": %.2f",
           (unsigned long long)wire,(unsigned long long)page,role,c.M->node,(unsigned long long)(role?0:gaps[g]/1000),n,at(a,n,1,10),at(a,n,1,2),at(a,n,9,10));
    if(!role)printf(", \"trip_p10\": %.2f, \"trip_median\": %.2f, \"trip_p90\": %.2f",at(b,n,1,10),at(b,n,1,2),at(b,n,9,10));
    printf("}\n");
    if(role)break;
  }
  free(done);free(trip);free(gap);free(a);free(b);release_probe(&c,stream,1,data,table);
  return 0;
}

// docs/measurement.md#the-retire-run
#include "mesh-dataflow.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <poll.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* design/transport.md section 5: the attaching child's report and exit, each within MESH_PLATFORM_NS of the fork (a
   child's attach or exit can wait on its region or its device), past which it is killed and the run ends, the wait
   named */
static void child_left(pid_t pid, const struct mesh_deadline *deadline) {
  if (mesh_clock_ns() < deadline->at) return;
  kill(pid, SIGKILL);
  mesh_expired(deadline->wait, deadline->ns);
}

int mesh_diagnostic_retire(int n) {
  if (n < 1) return 2;
  const int fd = shm_open(MESH_NAME, O_RDONLY);
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
    if (pid < 0) { close(p[0]); close(p[1]); return 1; }
    uint64_t client = 0;
    const struct mesh_deadline deadline = mesh_deadline("retire: the attaching child's report and exit (mesh_diagnostic_retire)", 0);
    for (struct pollfd look = {p[0], POLLIN, 0}; poll(&look, 1, 10) <= 0;) child_left(pid, &deadline);
    (void)!read(p[0], &client, sizeof client);
    close(p[0]);
    close(p[1]);
    while (!waitpid(pid, NULL, WNOHANG)) { child_left(pid, &deadline); usleep(1000); }
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
