#include <signal.h>
#include "mesh-dataflow.h"
#include "mesh-map.h"
#include "mesh-call.h"
#include "mesh-collective.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
/* design/algorithm-sources.md#programtensor */
/* design/plan-to-bridge-change-request.md R8; P. E. McKenney and J. D. Slingwine, Read-copy update, 1998; K. Fraser,
   Practical lock-freedom, 2004 */
void mesh_retire(struct hdr *m,uint64_t client){
  struct mesh_algo *algo=mesh_algo(m);
  const uint64_t published=atomic_load_explicit(&algo->published,memory_order_acquire);
  if(atomic_load_explicit(&m->client,memory_order_acquire)==client && published)
    atomic_store_explicit(&algo->retired,published,memory_order_release);
  mesh_notify(&m->plans);
  /* design/prepared-machine.md#M26 */
  mesh_control_notify(m);
}

/* design/aot-compilation-change-request.md, appendix item 6; NVIDIA, CUDA Programming Guide §4.2.3.1 */
void mesh_plan_retire(struct mesh_ctx *c,uint64_t plan){
  atomic_store_explicit(&mesh_state_at(c->M,plan)->closed,1,memory_order_release);
  mesh_notify(&c->M->plans);
}

/* design/plan-to-bridge-change-request.md R8 */
int mesh_attach(struct mesh_ctx *c,const char *name){
  if(c->M) return 0;
  if(!name) name=getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  int file=shm_open(name,O_RDWR,MESH_MODE);
  if(file<0) return errno;
  struct stat info;
  if(fstat(file,&info)){ int error=errno; close(file); return error; }
  struct hdr *memory=mmap(NULL,(size_t)info.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,file,0);
  int error=errno;
  if(memory==MAP_FAILED){ close(file); return error; }
  if((size_t)info.st_size<sizeof *memory || memory->magic!=MESH_MAGIC || memory->version!=MESH_VERSION || memory->length>(uint64_t)info.st_size){
    munmap(memory,(size_t)info.st_size); close(file); return EINVAL;
  }
  uint64_t client=(((atomic_fetch_add_explicit(&memory->serial,1,memory_order_relaxed)+1)&UINT64_C(0x7fffffff))<<32)|(uint32_t)getpid();
  const uint64_t device=atomic_load_explicit(&memory->device_client,memory_order_seq_cst);
  client|=(~device)&(UINT64_C(1)<<63);
  if(__atomic_load_n(&memory->version,__ATOMIC_SEQ_CST)!=MESH_VERSION){ munmap(memory,(size_t)info.st_size); close(file); return EINVAL; }
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.client=client,.fd=file,.start=mesh_started(getpid())};
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  mesh_retire(c->M,c->client);
  int status=munmap(c->M,c->len);
  int error=status?errno:0;
  if(close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* design/algorithm-sources.md#programtensor */
void mesh_backing_bind(struct mesh_ctx *c,uint32_t first,uint32_t pages,uint32_t page,uint32_t index){
  for(uint32_t offset=0;offset<pages;offset+=c->M->block){
    struct mesh_page_entry *entry=&mesh_page(c->M)[first+offset/c->M->block];
    atomic_store_explicit(&entry->mapping,((uint64_t)index<<32)|(page+offset),memory_order_relaxed);
    atomic_store_explicit(&entry->address,(uintptr_t)mesh_at(c->M,page+offset)-(uintptr_t)c->M,memory_order_relaxed);
  }
}

/* design/torch-capture-change-request.md item 6; TensorFlow Lite 6cf18d88,
   tensorflow/lite/delegates/gpu/common/memory_management.h:130-131 and tensorflow/lite/simple_memory_arena.cc:441-445 */
static uint64_t mesh_view_row(struct hdr *m,const struct mesh_value *v,uint32_t copy){
  return atomic_load_explicit(&mesh_page(m)[v->section.first+copy*v->section.stride].address,memory_order_relaxed);
}
void mesh_view_values(struct mesh_ctx *c,const struct mesh_function *functions,uint32_t count,uint64_t *first,
                      uint64_t *offset,struct mesh_view *view){
  struct hdr *m=c->M;
  const uint64_t block=(uint64_t)m->block*m->pgsz;
  uint64_t low=UINT64_MAX,high=0,at=0,values=0;
  for(uint32_t f=0;f<count;f++)
    for(uint32_t i=0;i<functions[f].values;i++){
      const struct mesh_value *v=functions[f].value+i;
      for(uint32_t k=0;v->run && k<v->section.count;k++){
        const uint64_t a=mesh_view_row(m,v,k),z=a+(uint64_t)v->section.stride*block;
        if(a<low)low=a;
        if(z>high)high=z;
      }
    }
  for(uint32_t f=0;f<count;f++)
    for(uint32_t i=0;i<functions[f].values;i++){
      const struct mesh_value *v=functions[f].value+i;
      first[values++]=at;
      for(uint32_t k=0;k<v->section.count;k++)offset[at++]=v->run?mesh_view_row(m,v,k)-low:UINT64_MAX;
    }
  first[values]=at;
  *view=high>low?(struct mesh_view){(char *)c->M+low,(size_t)(high-low)}:(struct mesh_view){NULL,0};
}

/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *c,uint32_t row,uint64_t address){
  if(row!=MESH_ABSENT)atomic_store_explicit(&mesh_page(c->M)[row].device,address,memory_order_relaxed);
}

/* M. M. Michael, IEEE TPDS 15(6), 2004 */
static int mesh_holding_retained(struct hdr *m,const struct mesh_algo *algo,uint64_t published){
  const uint32_t links=m->links>MESH_PORTS?m->links:MESH_PORTS;
  for(uint32_t p=0;p<links;p++)
    if(atomic_load_explicit(&mesh_links(m)[p].port.prepared,memory_order_acquire)==published)return 1;
  for(uint32_t k=0;k<algo->plans;k++)
    if(atomic_load_explicit(&mesh_algo_state(m,algo,k)->held,memory_order_acquire))return 1;
  return 0;
}

/* design/algorithm-sources.md#programtensor */
/* design/aot-compilation-change-request.md, appendix item 6; P. E. McKenney and J. D. Slingwine, Read-copy update,
   1998; K. Fraser, Practical lock-freedom, 2004; C++20 [atomics.order] p4.4 */
uint32_t mesh_retired_release(struct hdr *m){
  struct mesh_algo *algo=mesh_algo(m);
  const uint64_t published=mesh_algo_unreleased(m);
  const struct mesh_algo_holder holder=mesh_algo_holder(m);
  if(!published || mesh_holds(holder.pid,holder.start))return 0;
  atomic_thread_fence(memory_order_seq_cst);
  if(mesh_holding_retained(m,algo,published))return 1;
  atomic_store_explicit(&algo->released,published,memory_order_release);
  os_sync_wake_by_address_all(&algo->released,sizeof algo->released,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  return 0;
}

uint32_t mesh_cables(const struct mesh_record *a,const struct mesh_record *b,struct mesh_cable *out){
  const int lower=a->node<b->node;
  uint32_t n=0;
  for(uint32_t i=0;i<a->reached && i<MESH_PORTS;i++){
    const struct mesh_reach *e=a->reach+i;
    if(e->node!=b->node)continue;
    for(uint32_t k=0;k<b->reached && k<MESH_PORTS;k++){
      const struct mesh_reach *f=b->reach+k;
      if(f->node!=a->node || f->port!=e->peer_port || f->peer_port!=e->port)continue;
      const struct mesh_cable cable={.key={lower?e->port:e->peer_port,lower?e->peer_port:e->port},
        .qps=e->qps<f->qps?e->qps:f->qps,.bandwidth=e->bandwidth<f->bandwidth?e->bandwidth:f->bandwidth};
      uint32_t at=n++;
      for(;at && (out[at-1].key[0]>cable.key[0] || (out[at-1].key[0]==cable.key[0] && out[at-1].key[1]>cable.key[1]));at--)
        out[at]=out[at-1];
      out[at]=cable;
      break;
    }
  }
  return n;
}

static uint32_t mesh_join(struct mesh_record *a,struct mesh_record *b,uint64_t *bandwidth){
  struct mesh_cable cables[MESH_PORTS];
  const uint32_t joined=mesh_cables(a,b,cables);
  uint64_t slowest=0;
  for(uint32_t c=0;c<joined;c++)if(cables[c].bandwidth && (!slowest || cables[c].bandwidth<slowest))slowest=cables[c].bandwidth;
  if(joined)*bandwidth=slowest;
  return joined;
}
int mesh_joined(struct mesh_record *a,struct mesh_record *b,double *beta){
  uint64_t bandwidth=0;
  const uint32_t joined=mesh_join(a,b,&bandwidth);
  if(joined)*beta=bandwidth?8e9/(double)bandwidth:0;
  return (int)joined;
}

/* design/aot-compilation-change-request.md A7; G. V. Chockler, I. Keidar and R. Vitenberg, ACM Comput. Surv. 33(4),
   2001, Property 3.3 */
static int mesh_view_copy(struct mesh_ctx *c,uint64_t deadline,struct mesh_record *r,uint32_t *count,
                          struct mesh_view_id *view){
  struct hdr *m=c->M;
  for(;;){
    const uint64_t before=atomic_load_explicit(&m->view_map,memory_order_acquire),now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    if(now>=deadline)return mesh_gone((pid_t)atomic_load_explicit(&m->bridge_pid,memory_order_acquire))?ESRCH:ETIMEDOUT;
    if(before&1){
      os_sync_wait_on_address_with_timeout(&m->view_map,before,sizeof m->view_map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                           OS_CLOCK_MACH_ABSOLUTE_TIME,deadline-now);
      continue;
    }
    const uint64_t nodes=atomic_load_explicit(&m->view_nodes,memory_order_relaxed),at=m->view_off;
    *view=mesh_view_installed(m,memory_order_relaxed);
    const int fits=at>=sizeof *m && at<=c->len && nodes<=(c->len-at)/sizeof *r;
    if(fits && nodes<=MESH_NODES_MOST)memcpy(r,(char *)m+at,nodes*sizeof *r);
    if(!mesh_seqlock_whole(&m->view_map,before))continue;
    if(!fits)return EINVAL;
    if(nodes>MESH_NODES_MOST)return E2BIG;
    if(!view->id)return ESTALE;
    *count=(uint32_t)nodes;return 0;
  }
}

static void mesh_map_order(const struct mesh_record *r,uint32_t count,uint32_t *order){
  for(uint32_t i=0;i<count;i++){
    uint32_t k=i;
    for(;k && r[order[k-1]].node>r[i].node;k--)order[k]=order[k-1];
    order[k]=i;
  }
}

static uint64_t mesh_capacity(const struct mesh_record *a,const struct mesh_record *b){
  struct mesh_cable cables[MESH_PORTS];
  uint64_t sum=0;
  for(uint32_t c=mesh_cables(a,b,cables);c;c--)sum+=cables[c-1].bandwidth;
  return sum;
}

/* design/aot-compilation-change-request.md A7 */
int mesh_nodes_read(struct mesh_ctx *c,struct mesh_nodes *out){
  struct mesh_record r[MESH_NODES_MOST];
  uint32_t count,order[MESH_NODES_MOST];
  struct mesh_view_id view;
  int error=mesh_view_copy(c,clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS,r,&count,&view);
  if(error)return error;
  mesh_map_order(r,count,order);
  *out=(struct mesh_nodes){.serial=view.id,.count=count,.self=MESH_ABSENT,.sequencer=view.sequencer,.boot=view.boot};
  for(uint32_t i=0;i<count;i++){
    out->node[i]=r[order[i]].node;
    if(out->node[i]==c->M->node)out->self=i;
    for(uint32_t j=0;j<i;j++)out->capacity[i][j]=out->capacity[j][i]=mesh_capacity(r+order[i],r+order[j]);
  }
  return out->self==MESH_ABSENT?ENOENT:0;
}

/* design/disjoint-pages-change-request.md, D.3.6; design/aot-compilation-change-request.md A7 and appendix items 2 and
   6; X. Défago, A. Schiper and P. Urbán, ACM Comput. Surv. 36(4), 2004, §7.1; P. Barham et al., Pathways, MLSys 2022,
   §4.4; L. Lamport, CACM 21(7), 1978, IR1; S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing
   45(6), 1997, (7); K. Fraser, Practical lock-freedom, 2004 */
static inline const uint32_t *mesh_wait_row(struct hdr *m,const struct mesh_state *state,uint64_t place){
  return (const uint32_t *)(const void *)((const char *)m+state->rows)+place;
}
int mesh_transfers_start(struct mesh_ctx *context,uint64_t plan,uint64_t executions){
  struct hdr *m=context->M;
  struct mesh_state *state=mesh_state_at(m,plan);
  struct mesh_submission *queue=(struct mesh_submission *)(void *)((char *)m+state->submits);
  const uint64_t k=atomic_load_explicit(&state->submissions,memory_order_relaxed),K=k/state->calls,
                 served=k?queue[(k-1)%state->submit_room].end:0;
  if(served<executions){
    struct mesh_sync *edge=mesh_syncs(m,state);
    const uint32_t *row=mesh_wait_row(m,state,k%state->calls);
    for(uint32_t e=row[0];e<row[1];e++){
      const __int128 need=(__int128)K*edge[e].every+edge[e].base;
      uint64_t seen=atomic_load_explicit(&edge[e].count,memory_order_acquire),
               until=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;
      while(seen<need){
        const uint64_t now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
        if(mesh_view_compare(mesh_view_installed(m,memory_order_acquire),mesh_plan_view(state)))return ESTALE;
        if(now>=until)return ETIMEDOUT;
        os_sync_wait_on_address_with_timeout(&edge[e].count,seen,sizeof edge[e].count,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                             OS_CLOCK_MACH_ABSOLUTE_TIME,until-now);
        const uint64_t risen=atomic_load_explicit(&edge[e].count,memory_order_acquire);
        if(risen!=seen){seen=risen;until=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;}
      }
    }
  }
  if(served<executions){
    queue[k%state->submit_room]=(struct mesh_submission){.end=executions,.clock=++context->clock};
    atomic_store_explicit(&state->submissions,k+1,memory_order_release);
  }
  mesh_notify(&m->plans);
  atomic_thread_fence(memory_order_seq_cst);
  return mesh_view_compare(mesh_view_installed(m,memory_order_relaxed),mesh_plan_view(state))?ESTALE:0;
}
/* MPI 4.1 §3.9, MPI_START; ISO/IEC 9899:2011 (N1570) 7.17.3p11; libfabric man/fi_endpoint.3.md, fi_cancel
   (https://raw.githubusercontent.com/ofiwg/libfabric/main/man/fi_endpoint.3.md) */
void mesh_host_started(struct mesh_ctx *context,uint64_t plan,uint32_t transfer,uint64_t execution){
  struct hdr *m=context->M;
  struct mesh_state *state=mesh_state_at(m,plan);
  uint64_t seen=atomic_load_explicit(&state->executions,memory_order_relaxed);
  (void)transfer;
  while(seen<execution+1 && !atomic_compare_exchange_weak_explicit(&state->executions,&seen,execution+1,
                                                                   memory_order_release,memory_order_relaxed));
  mesh_notify(&m->plans);
  atomic_thread_fence(memory_order_seq_cst);
  if(!mesh_view_compare(mesh_view_installed(m,memory_order_relaxed),mesh_plan_view(state)))return;
  struct mesh_sync *edge=mesh_syncs(m,state);
  for(uint32_t e=0;e<state->waits;e++){
    atomic_store_explicit(&edge[e].count,UINT64_MAX,memory_order_release);
    os_sync_wake_by_address_all(&edge[e].count,sizeof edge[e].count,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  }
  const uint32_t copies=state->copies;
  const uint64_t executions=atomic_load_explicit(&state->executions,memory_order_acquire),n=state->entries/copies;
  const struct mesh_hop *list=mesh_hops(m,state);
  for(uint32_t c=0;c<copies;c++){
    const uint64_t serves=executions>c?(executions-c-1)/copies+1:0;
    for(uint64_t j=c*n;serves && j<c*n+n;j++){
      if(list[j].direction!=MESH_RECEIVE)continue;
      _Atomic uint64_t *count=(_Atomic uint64_t *)(void *)((char *)m+list[j].count);
      const uint64_t final=(uint64_t)list[j].pages*serves;
      uint64_t was=atomic_load_explicit(count,memory_order_relaxed);
      while(was<final && !atomic_compare_exchange_weak_explicit(count,&was,final,memory_order_release,
                                                                memory_order_relaxed));
      if(was<final)os_sync_wake_by_address_all(count,sizeof *count,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
    }
  }
}
