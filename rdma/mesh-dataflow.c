#include "mesh-dataflow.h"
#include "mesh-map.h"
#include "mesh-call.h"
#include "mesh-collective.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

void mesh_retire(struct hdr *m,uint64_t client){
  struct mesh_algo *algo=mesh_algo(m);
  const uint64_t published=atomic_load_explicit(&algo->published,memory_order_acquire);
  if(atomic_load_explicit(&m->client,memory_order_acquire)==client)
    atomic_store_explicit(&algo->retired,published,memory_order_release);
  mesh_notify(&m->plans);
}

void mesh_plan_retire(struct mesh_ctx *c,uint64_t plan){
  atomic_store_explicit(&mesh_state_at(c->M,plan)->closed,1,memory_order_release);
  mesh_notify(&c->M->plans);
}

int mesh_attach(struct mesh_ctx *c,const char *name){
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
  const uint64_t client=((atomic_fetch_add_explicit(&memory->serial,1,memory_order_relaxed)+1)<<32)|(uint32_t)getpid();
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.client=client,.fd=file,.start=mesh_started(getpid())};
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  mesh_retire(c->M,c->client);
  int error=munmap(c->M,c->len)?errno:0;
  if(close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

void mesh_backing_bind(struct mesh_ctx *c,uint32_t first,uint32_t pages,uint32_t page){
  for(uint32_t offset=0;offset<pages;offset+=c->M->block)
    atomic_store_explicit(&mesh_page(c->M)[first+offset/c->M->block].address,
                          (uintptr_t)mesh_at(c->M,page+offset)-(uintptr_t)c->M,memory_order_relaxed);
}

/* The SDK's mach/mach_vm.h:270: "kern_return_t mach_vm_remap ( vm_map_t target_task, mach_vm_address_t
   *target_address, mach_vm_size_t size, mach_vm_offset_t mask, int flags, vm_map_t src_task, mach_vm_address_t
   src_address, boolean_t copy, vm_prot_t *cur_protection, vm_prot_t *max_protection, vm_inherit_t inheritance )";
   mach/vm_statistics.h:307, VM_FLAGS_OVERWRITE: "delete any existing mappings first"; mach/vm_inherit.h:87,
   VM_INHERIT_NONE: "absent from child" */
static int mesh_view_compose(struct mesh_ctx *c,const uint32_t *rows,size_t count,struct mesh_view *view){
  *view=(struct mesh_view){NULL,0};
  if(!count)return 0;
  struct hdr *m=c->M;
  const size_t block=(size_t)m->block*m->pgsz,bytes=count*block;
  char *base=mmap(NULL,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
  if(base==MAP_FAILED)return errno;
  for(size_t k=0,run;k<count;k+=run){
    mach_vm_address_t at=(mach_vm_address_t)(uintptr_t)(base+k*block);
    const uint64_t first=atomic_load_explicit(&mesh_page(m)[rows[k]].address,memory_order_relaxed);
    for(run=1;k+run<count &&
              atomic_load_explicit(&mesh_page(m)[rows[k+run]].address,memory_order_relaxed)==first+run*block;run++);
    const uintptr_t source=(uintptr_t)m+first;
    vm_prot_t current,maximum;
    const kern_return_t mapped=mach_vm_remap(mach_task_self(),&at,run*block,0,VM_FLAGS_FIXED|VM_FLAGS_OVERWRITE,
                                             mach_task_self(),source,FALSE,&current,&maximum,VM_INHERIT_NONE);
    if(mapped!=KERN_SUCCESS){
      munmap(base,bytes);
      return mapped==KERN_NO_SPACE || mapped==KERN_RESOURCE_SHORTAGE?ENOMEM:EINVAL;
    }
  }
  *view=(struct mesh_view){base,bytes};
  return 0;
}
void mesh_view_release(struct mesh_view view){
  munmap(view.address,view.bytes);
}
/* row(c, k) = section.first + c × section.stride + k, for c < section.count and k < section.stride */
int mesh_view_values(struct mesh_ctx *c,const struct mesh_function *functions,uint32_t count,uint64_t *offset,
                     struct mesh_view *view){
  const size_t block=(size_t)c->M->block*c->M->pgsz;
  size_t rows=0,values=0;
  for(uint32_t f=0;f<count;f++)
    for(uint32_t i=0;i<functions[f].values;i++,values++){
      const struct mesh_section s=functions[f].value[i].section;
      offset[values]=functions[f].value[i].run?rows*block:UINT64_MAX;
      if(functions[f].value[i].run)rows+=(size_t)s.count*s.stride;
    }
  uint32_t *row=malloc(rows*sizeof *row);
  if(!row)return ENOMEM;
  for(uint32_t f=0,at=0;f<count;f++)
    for(uint32_t i=0;i<functions[f].values;i++){
      const struct mesh_section s=functions[f].value[i].section;
      for(uint32_t k=0;functions[f].value[i].run && k<s.count*s.stride;k++)row[at++]=s.first+k;
    }
  const int error=mesh_view_compose(c,row,rows,view);
  free(row);
  return error;
}

/* M. M. Michael, Hazard pointers: safe memory reclamation for lock-free objects, IEEE TPDS 15(6), 2004 */
static int mesh_holding_retained(struct hdr *m,const struct mesh_algo *algo,uint64_t published){
  const uint32_t links=m->links>MESH_PORTS?m->links:MESH_PORTS;
  for(uint32_t p=0;p<links;p++)
    if(atomic_load_explicit(&mesh_links(m)[p].port.prepared,memory_order_acquire)==published)return 1;
  for(uint32_t k=0;k<algo->plans;k++)
    if(atomic_load_explicit(&mesh_algo_state(m,algo,k)->held,memory_order_acquire))return 1;
  return 0;
}

/* P. E. McKenney and J. D. Slingwine, Read-copy update, PDCS 1998; K. Fraser, Practical lock-freedom, 2004; C++20
   [atomics.order] p4.4 */
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

int mesh_joined(struct mesh_record *a,struct mesh_record *b,double *beta){
  struct mesh_cable cables[MESH_PORTS];
  (void)beta;
  return (int)mesh_cables(a,b,cables);
}

static int mesh_view_copy(struct mesh_ctx *c,uint64_t deadline,struct mesh_record *r,uint32_t *count,
                          struct mesh_view_id *view){
  struct hdr *m=c->M;
  for(;;){
    const uint64_t before=atomic_load_explicit(&m->view_map,memory_order_acquire),now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    if(now>=deadline)return ETIMEDOUT;
    if(before&1){
      os_sync_wait_on_address_with_timeout(&m->view_map,before,sizeof m->view_map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                           OS_CLOCK_MACH_ABSOLUTE_TIME,deadline-now);
      continue;
    }
    const uint64_t nodes=atomic_load_explicit(&m->view_nodes,memory_order_relaxed);
    *view=mesh_view_installed(m,memory_order_relaxed);
    if(nodes<=MESH_NODES_MOST)memcpy(r,(char *)m+m->view_off,nodes*sizeof *r);
    if(!mesh_seqlock_whole(&m->view_map,before))continue;
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

static inline const uint32_t *mesh_wait_row(struct hdr *m,const struct mesh_state *state,uint64_t place){
  return (const uint32_t *)(const void *)((const char *)m+state->rows)+place;
}
/* Bhattacharyya, Sriram and Lee 1997, (7): "start(vi, k) ≥ end(vj, k − delay((vj, vi)))"; the waits of start k are
   row k mod calls, each until count ≥ ⌊k / calls⌋ × every + base over ℤ */
int mesh_transfers_start(struct mesh_ctx *context,uint64_t plan,uint64_t executions){
  struct hdr *m=context->M;
  struct mesh_state *state=mesh_state_at(m,plan);
  struct mesh_submission *queue=(struct mesh_submission *)(void *)((char *)m+state->submits);
  const uint64_t k=atomic_load_explicit(&state->submissions,memory_order_relaxed),K=k/state->calls;
  struct mesh_sync *edge=mesh_syncs(m,state);
  const uint32_t *row=mesh_wait_row(m,state,k%state->calls);
  for(uint32_t e=row[0];e<row[1];e++){
    const __int128 need=(__int128)K*edge[e].every+edge[e].base;
    uint64_t seen=atomic_load_explicit(&edge[e].count,memory_order_acquire),
             until=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;
    while(seen<need){
      const uint64_t now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
      if(now>=until)return ETIMEDOUT;
      os_sync_wait_on_address_with_timeout(&edge[e].count,seen,sizeof edge[e].count,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                           OS_CLOCK_MACH_ABSOLUTE_TIME,until-now);
      const uint64_t risen=atomic_load_explicit(&edge[e].count,memory_order_acquire);
      if(risen!=seen){seen=risen;until=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;}
    }
  }
  queue[k%state->submit_room]=(struct mesh_submission){.end=executions,.clock=++context->clock};
  atomic_store_explicit(&state->submissions,k+1,memory_order_release);
  mesh_notify(&m->plans);
  return mesh_view_compare(mesh_view_installed(m,memory_order_relaxed),mesh_plan_view(state))?ESTALE:0;
}
void mesh_host_started(struct mesh_ctx *context,uint64_t plan,uint32_t transfer,uint64_t execution){
  struct mesh_state *state=mesh_state_at(context->M,plan);
  uint64_t seen=atomic_load_explicit(&state->executions,memory_order_relaxed);
  (void)transfer;
  while(seen<execution+1 && !atomic_compare_exchange_weak_explicit(&state->executions,&seen,execution+1,
                                                                   memory_order_release,memory_order_relaxed));
  mesh_notify(&context->M->plans);
}
