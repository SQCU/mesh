#include "mesh-reduce.h"
#include <errno.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>

enum { R_WAITING, R_BORROWED, R_READY, R_COMPLETING };
struct mesh_reduce_function {
  _Atomic size_t refs;
  size_t count;
  size_t *parent;
  struct mesh_reduce_node nodes[];
};
struct mesh_reduction {
  mesh_reduce_function *function;
  uint64_t program, invocation, owner;
  _Atomic int error;
  _Atomic int states[];
};
static int contributor_order(const void *a,const void *b){
  uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b; return (x>y)-(x<y); }

mesh_reduce_function *mesh_reduce_compile(size_t contributors, mesh_reduce_index_fn index, void *capture){
  if(!contributors || contributors>(size_t)INT_MAX/2 || !index){ errno=EINVAL; return NULL; }
  size_t count=contributors*2-1;
  mesh_reduce_function *f=calloc(1,sizeof *f+count*sizeof *f->nodes);
  size_t *parent=malloc(count*sizeof *parent);
  uint64_t *ids=malloc(contributors*sizeof *ids);
  int error=0; size_t leaves=0;
  if(!f||!parent||!ids){ error=ENOMEM; goto finish; }
  for(size_t i=0;i<count;i++) parent[i]=SIZE_MAX;
  for(size_t i=0;i<count&&!error;i++){
    int status=index(capture,i,&f->nodes[i]);
    if(status){ error=status>0?status:EINVAL; break; }
    const struct mesh_reduce_node *n=&f->nodes[i];
    if(n->input[0]==SIZE_MAX && n->input[1]==SIZE_MAX){
      if(leaves==contributors || n->contributor>=contributors){ error=EINVAL; break; }
      ids[leaves++]=n->contributor;
    } else for(size_t j=0;j<2;j++){
      size_t child=n->input[j];
      if(child>=i || parent[child]!=SIZE_MAX){ error=EINVAL; break; }
      parent[child]=i;
    }
  }
  if(!error && leaves!=contributors) error=EINVAL;
  if(!error) for(size_t i=0;i+1<count;i++) if(parent[i]==SIZE_MAX){ error=EINVAL; break; }
  if(!error){
    qsort(ids,contributors,sizeof *ids,contributor_order);
    for(size_t i=1;i<contributors;i++) if(ids[i]==ids[i-1]){ error=EEXIST; break; }
  }
finish:
  free(ids);
  if(error){ free(parent); free(f); errno=error; return NULL; }
  atomic_init(&f->refs,1); f->count=count; f->parent=parent; return f;
}
size_t mesh_reduce_count(const mesh_reduce_function *f){ return f->count; }
const struct mesh_reduce_node *mesh_reduce_node(const mesh_reduce_function *f, size_t i){ return i<f->count?&f->nodes[i]:NULL; }
void mesh_reduce_free(mesh_reduce_function *f){
  if(f && atomic_fetch_sub(&f->refs,1)==1){ free(f->parent); free(f); } }
mesh_reduction *mesh_reduce_bind(mesh_reduce_function *f, uint64_t program, uint64_t invocation, uint64_t owner){
  mesh_reduction *r=calloc(1,sizeof *r+f->count*sizeof *r->states);
  if(!r){ errno=ENOMEM; return NULL; }
  atomic_fetch_add(&f->refs,1);
  r->function=f; r->program=program; r->invocation=invocation; r->owner=owner;
  return r;
}
struct mesh_reduce_token mesh_reduce_token(const mesh_reduction *r, size_t i){
  return (struct mesh_reduce_token){r->program,r->invocation,i}; }
int mesh_reduce_needed(const mesh_reduction *r, size_t i){
  if(i>=r->function->count) return 0;
  size_t parent=r->function->parent[i];
  return r->function->nodes[i].owner==r->owner || parent==SIZE_MAX || r->function->nodes[parent].owner==r->owner;
}
int mesh_reduce_ready(const mesh_reduction *r, size_t i){
  if(!mesh_reduce_needed(r,i)) return -EINVAL;
  int error=atomic_load_explicit(&r->error,memory_order_acquire);
  if(error) return -error;
  int state=atomic_load_explicit(&r->states[i],memory_order_acquire);
  return state<0?state:state==R_READY;
}
static int reduction_token(const mesh_reduction *r, struct mesh_reduce_token token){
  if(token.program!=r->program || token.invocation!=r->invocation) return -ESTALE;
  if(token.node>=r->function->count) return -EINVAL;
  return mesh_reduce_needed(r,(size_t)token.node)?0:-EXDEV;
}
int mesh_reduce_claim(mesh_reduction *r, struct mesh_reduce_token token){
  int error=reduction_token(r,token); if(error) return error;
  error=atomic_load_explicit(&r->error,memory_order_acquire); if(error) return -error;
  size_t i=(size_t)token.node;
  const struct mesh_reduce_node *n=&r->function->nodes[i];
  if(n->owner==r->owner && n->input[0]!=SIZE_MAX){
    for(size_t j=0;j<2;j++){
      int state=mesh_reduce_ready(r,n->input[j]);
      if(state!=1) return state;
    }
  }
  int waiting=R_WAITING;
  if(atomic_compare_exchange_strong_explicit(&r->states[i],&waiting,R_BORROWED,memory_order_acquire,memory_order_relaxed)) return 1;
  return waiting<0?waiting:0;
}
int mesh_reduce_commit(mesh_reduction *r, struct mesh_reduce_token token, int error){
  int status=reduction_token(r,token); if(status) return status;
  if(error<0) return -EINVAL;
  int borrowed=R_BORROWED;
  if(!atomic_compare_exchange_strong(&r->states[token.node],&borrowed,R_COMPLETING)) return -EINVAL;
  atomic_store_explicit(&r->states[token.node],error?-error:R_READY,memory_order_release);
  return 0;
}
int mesh_reduce_status(const mesh_reduction *r){
  int complete=1;
  for(size_t i=0;i<r->function->count;i++) if(mesh_reduce_needed(r,i)){
    int state=mesh_reduce_ready(r,i); if(state<0) return state; complete&=state;
  }
  return complete;
}
void mesh_reduce_cancel(mesh_reduction *r, int error){
  int clear=0; atomic_compare_exchange_strong(&r->error,&clear,error>0?error:ECANCELED); }
int mesh_reduce_retire(mesh_reduction *r){
  for(size_t i=0;i<r->function->count;i++){
    int state=atomic_load_explicit(&r->states[i],memory_order_acquire);
    if(state==R_BORROWED || state==R_COMPLETING) return EBUSY;
  }
  mesh_reduce_free(r->function); free(r); return 0;
}
