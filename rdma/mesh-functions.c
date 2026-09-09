#include "mesh-functions.h"
#include "mesh-wire.h"
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum { WAITING, AVAILABLE, BORROWED, COMPLETING, COMMITTED, SETTLED, INVALID };
struct mesh_function { _Atomic size_t refs; size_t count; uint64_t *channels; struct mesh_extent extents[]; };
struct mesh_argument {
  struct mstream stream;
  struct mesh_view view;
  uint32_t *pages;
  size_t page_count;
  int active, released, leased;
  _Atomic int state, error, requested;
};
struct mesh_call {
  struct mesh_executor *executor;
  struct mesh_function *function;
  uint32_t channel_base;
  int retain_transmit;
  uint64_t period, last;
  struct mesh_scope scope;
  uint32_t *indices; size_t indices_bytes;
  uint64_t *work, *runnable;
  size_t settled;
  _Atomic int status;
  struct mesh_argument arguments[];
};
struct mesh_executor {
  struct mesh_ctx *context;
  pthread_mutex_t lock;
  struct mesh_call **calls;
  size_t count;
  struct mstream **streams;
  size_t capacity, window, received;
  struct mesh_executor_policy policy;
};

static void *allocation(size_t bytes){
  void *p=calloc(1,bytes); if(!p) errno=ENOMEM; return p; }
static int channel_order(const void *a,const void *b){
  uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b; return (x>y)-(x<y); }
static void change_argument(mesh_call *c,size_t i){
  mesh_stream_changed(&c->arguments[i].stream);
}
static void change_call(mesh_call *c){
  for(size_t word=0;word<mesh_page_words(c->function->count);word++){
    size_t bits=c->function->count-word*64; if(bits>64) bits=64;
    __atomic_fetch_or(&c->work[word],UINT64_MAX>>(64-bits),__ATOMIC_RELEASE);
    __atomic_fetch_or(&c->runnable[word],UINT64_MAX>>(64-bits),__ATOMIC_RELEASE);
  }
}
static void bind_argument_work(mesh_call *c,size_t i){
  c->arguments[i].stream.changed=c->work+i/64;
  c->arguments[i].stream.runnable=c->runnable+i/64;
  c->arguments[i].stream.changed_bit=UINT64_C(1)<<(i%64);
}

mesh_function *mesh_compile(size_t count, mesh_index_fn index, void *capture){
  if(count>INT_MAX || count>(SIZE_MAX-sizeof(mesh_function))/sizeof(struct mesh_extent) || (count&&!index)){
    errno=EINVAL; return NULL; }
  mesh_function *f=allocation(sizeof *f+count*sizeof(struct mesh_extent)); if(!f) return NULL;
  atomic_init(&f->refs,1); f->count=count;
  for(size_t i=0;i<count;i++){
    int status=index(capture,i,&f->extents[i]);
    struct mesh_extent *x=&f->extents[i];
    if(status || !x->bytes || x->offset>SIZE_MAX-x->bytes || x->peer<0 || x->peer>=UINT16_MAX || (x->receive!=0&&x->receive!=1)){
      errno=status>0?status:EINVAL; free(f); return NULL; }
  }
  uint64_t *channels=allocation((count?count:1)*sizeof *channels);
  if(!channels){ free(f); return NULL; }
  for(size_t i=0;i<count;i++) channels[i]=((uint64_t)f->extents[i].channel<<17)|((uint64_t)f->extents[i].peer<<1)|(unsigned)f->extents[i].receive;
  qsort(channels,count,sizeof *channels,channel_order);
  int duplicate=0; for(size_t i=1;i<count;i++) duplicate|=channels[i]==channels[i-1];
  if(duplicate){ errno=EADDRINUSE; free(channels); free(f); return NULL; }
  f->channels=channels;
  return f;
}
const struct mesh_extent *mesh_function_extent(const mesh_function *f, size_t i){ return i<f->count?&f->extents[i]:NULL; }
size_t mesh_function_count(const mesh_function *f){ return f->count; }
void mesh_function_free(mesh_function *f){ if(f&&atomic_fetch_sub(&f->refs,1)==1){ free(f->channels); free(f); } }

static int channel_overlap(const mesh_function *a,uint32_t abase,const mesh_function *b,uint32_t bbase){
  if(!a->count || !b->count) return 0;
  uint64_t x=(uint64_t)abase<<17, y=(uint64_t)bbase<<17;
  if(a->channels[a->count-1]+x<b->channels[0]+y || b->channels[b->count-1]+y<a->channels[0]+x) return 0;
  size_t i=0,j=0;
  while(i<a->count && j<b->count){
    uint64_t left=a->channels[i]+x,right=b->channels[j]+y;
    if(left==right) return 1;
    if(left<right) i++; else j++;
  }
  return 0;
}

mesh_executor *mesh_executor_create(struct mesh_ctx *context, size_t window_pages){
  return mesh_executor_create_with(context,window_pages,(struct mesh_executor_policy){0});
}
mesh_executor *mesh_executor_create_with(struct mesh_ctx *context, size_t window_pages, struct mesh_executor_policy policy){
  if(!context || !context->M || context->mapping_pinned<0 || !window_pages || window_pages>=context->M->pool || window_pages>=context->M->arena){
    errno=EINVAL; return NULL; }
  mesh_executor *e=allocation(sizeof *e); if(!e) return NULL;
  int status=pthread_mutex_init(&e->lock,NULL);
  if(status){ free(e); errno=status; return NULL; }
  int vacant=0;
  if(!atomic_compare_exchange_strong(&context->executor_attached,&vacant,1)){
    pthread_mutex_destroy(&e->lock); free(e); errno=EBUSY; return NULL;
  }
  e->context=context; e->window=window_pages; e->policy=policy; context->mapping_pinned=1;
  return e;
}

static mesh_call *bind_function(mesh_executor *e, mesh_function *f, uint32_t base, struct mesh_scope scope){
  mesh_call *c=allocation(sizeof *c+f->count*sizeof(struct mesh_argument)+2*mesh_page_words(f->count)*sizeof(uint64_t)); if(!c) return NULL;
  c->executor=e; c->function=f; c->channel_base=base; c->scope=scope;
  c->work=(uint64_t*)(c->arguments+f->count);
  c->runnable=c->work+mesh_page_words(f->count);
  if(!f->count){ atomic_fetch_add(&f->refs,1); atomic_store(&c->status,1); c->executor=NULL; return c; }
  if(!e){ free(c); errno=EINVAL; return NULL; }
  size_t header=mesh_epoch_set(scope.epoch)?sizeof(struct mesh_frame):MESH_OFF;
  size_t payload=mesh_pay(e->context->M)-header;
  size_t pages=0, alignment=(size_t)getpagesize();
  for(size_t i=0;i<f->count;i++){
    size_t chunk=f->extents[i].chunk?f->extents[i].chunk:payload;
    if(chunk>payload){ free(c); errno=EINVAL; return NULL; }
    size_t count=f->extents[i].bytes/chunk+(f->extents[i].bytes%chunk!=0);
    size_t copies=f->extents[i].receive?2:1;
    if(count>((SIZE_MAX-alignment+1)/sizeof(uint32_t)-pages)/copies){ free(c); errno=EOVERFLOW; return NULL; }
    pages+=count*copies;
  }
  c->indices_bytes=(pages*sizeof(uint32_t)+alignment-1)/alignment*alignment;
  c->indices=mmap(NULL,c->indices_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
  if(c->indices==MAP_FAILED){ free(c); return NULL; }
  pages=0;
  int error=0; size_t receives=0;
  pthread_mutex_lock(&e->lock);
  if(e->context->mapping_pinned<0) error=ESTALE;
  if(f->count>INT_MAX-e->capacity) error=EOVERFLOW;
  for(size_t i=0;i<f->count&&!error;i++){
    const struct mesh_extent *x=&f->extents[i];
    struct mesh_argument *a=&c->arguments[i];
    size_t chunk=x->chunk?x->chunk:payload;
    a->page_count=x->bytes/chunk+(x->bytes%chunk!=0);
    if(x->channel>UINT32_MAX-base || a->page_count>e->window){ error=EOVERFLOW; break; }
    if(x->receive){
      size_t capacity=e->context->M->pool-e->window;
      if(a->page_count>capacity-e->received-receives){ error=EAGAIN; break; }
      receives+=a->page_count;
    }
    a->pages=c->indices+pages; pages+=a->page_count*(x->receive?2:1);
    a->view=(struct mesh_view){.base=mesh_at(e->context->M,0),.pages=a->pages,.epoch=&a->stream.scope.epoch,
      .offset=x->offset,.bytes=x->bytes,.stride=e->context->M->pgsz,.payload=(uint32_t)(sizeof(struct wire)+header),.capacity=(uint32_t)chunk};
    a->stream=(struct mstream){.n=x->bytes,.sid=base+x->channel,.node=x->peer,.rx=x->receive,
        .scope=scope,.logical_offset=x->offset,.chunk=x->chunk,.closing=1,.retain_seen=1,.open_retry_ns=e->policy.open_retry_ns,
        .arrivals=x->receive?a->pages+a->page_count:NULL,.parked=1};
    if(!error && (mesh_epoch_set(scope.epoch) || x->chunk)){
      if(x->receive && mesh_stream_receive(e->context,&a->stream,a->pages)) error=ENOMEM;
      a->active=1; a->stream.parked=0;
    }
    bind_argument_work(c,i);
  }
  for(size_t j=0;j<e->count&&!error;j++){
    mesh_call *other=e->calls[j];
    if((mesh_epoch_equal(scope.epoch,other->scope.epoch) || (other->period && scope.epoch.high==other->scope.epoch.high)) &&
       channel_overlap(f,base,other->function,other->channel_base)) error=EADDRINUSE;
  }
  size_t capacity=e->capacity+f->count;
  if(!error) error=mesh_stream_reserve(e->context,capacity);
  if(!error){
    mesh_call **calls=realloc(e->calls,(e->count+1)*sizeof *calls);
    if(calls) e->calls=calls; else error=ENOMEM;
  }
  if(!error){
    struct mstream **streams=realloc(e->streams,capacity*sizeof *streams);
    if(streams) e->streams=streams; else error=ENOMEM;
  }
  if(!error){
    for(size_t i=0;i<f->count;i++) e->streams[e->capacity+i]=&c->arguments[i].stream;
    e->capacity=capacity; e->received+=receives;
    atomic_fetch_add(&f->refs,1);
    e->calls[e->count++]=c;
    change_call(c);
    mesh_stream_register(e->context,e->streams,e->capacity);
  }
  pthread_mutex_unlock(&e->lock);
  if(error){ for(size_t i=0;i<f->count;i++) free(c->arguments[i].stream.seen); munmap(c->indices,c->indices_bytes); free(c); errno=error; return NULL; }
  return c;
}
mesh_call *mesh_bind(mesh_executor *e, mesh_function *f, uint32_t base){
  return bind_function(e,f,base,(struct mesh_scope){0});
}
mesh_call *mesh_bind_scoped(mesh_executor *e, mesh_function *f, uint32_t base, struct mesh_scope scope){
  if(!mesh_epoch_set(scope.epoch)){ errno=EINVAL; return NULL; }
  return bind_function(e,f,base,scope);
}

void mesh_call_retain_transmit(mesh_call *c){ c->retain_transmit=1; }
int mesh_call_cycle(mesh_call *c,uint64_t period,uint64_t last){
  if(period<2 || last<c->scope.epoch.low || !mesh_epoch_set(c->scope.epoch)) return EINVAL;
  c->period=period; c->last=last; c->retain_transmit=1;
  for(size_t i=0;i<c->function->count;i++){
    struct mstream *s=&c->arguments[i].stream;
    s->resident=1; s->closing=0;
    if(c->function->extents[i].receive) continue;
    if(s->paged) continue;
    s->work=calloc(mesh_page_words(c->arguments[i].page_count),sizeof *s->work);
    if(!s->work) return ENOMEM;
    s->paged=1; s->retain_seen=1;
  }
  return 0;
}

static void reset_argument(mesh_call *c,size_t i,struct mesh_epoch epoch,int borrowed){
  struct mesh_argument *a=&c->arguments[i]; const struct mesh_extent *x=&c->function->extents[i];
  struct mstream *s=&a->stream;
  int retained=c->retain_transmit && !x->receive && s->stride;
  s->off=0; s->done=0; s->retry_ns=0; s->fin_after_ns=0; s->scan_word=0;
  s->st=MS_RUN; s->agreed=borrowed?s->agreed:0; s->error=0; s->abort_ack=0; s->closed=0; s->parked=0;
  s->scope.epoch=epoch; s->nb=a->page_count; s->pages=x->receive?a->pages:NULL;
  atomic_store(&s->available,0);
  if(s->seen) memset(s->seen,0,s->paged?mesh_page_words(a->page_count)*sizeof *s->work:(a->page_count+7)/8);
  a->active=1; a->released=0; a->leased=retained;
  atomic_store(&a->error,0); atomic_store(&a->requested,borrowed);
  atomic_store_explicit(&a->state,retained?(borrowed?BORROWED:AVAILABLE):WAITING,memory_order_release);
  change_argument(c,i);
}

int mesh_call_rearm(mesh_call *c, struct mesh_epoch epoch){
  if(!mesh_epoch_set(c->scope.epoch) || epoch.high!=c->scope.epoch.high || epoch.low<=c->scope.epoch.low) return EINVAL;
  mesh_executor *e=c->executor;
  if(!e){ c->scope.epoch=epoch; return 0; }
  pthread_mutex_lock(&e->lock);
  int error=e->context->mapping_pinned<0?ESTALE:0;
  size_t receives=0;
  for(size_t i=0;i<c->function->count&&!error;i++){
    if(atomic_load(&c->arguments[i].state)!=SETTLED) error=EBUSY;
    if(epoch.low<=c->arguments[i].stream.scope.epoch.low) error=EINVAL;
    if(mesh_stream_conflict(e->context,&c->arguments[i].stream,epoch)) error=EADDRINUSE;
    if(c->function->extents[i].receive) receives+=c->arguments[i].page_count;
  }
  if(!error && receives>e->context->M->pool-e->window-e->received) error=EAGAIN;
  if(!error){
    c->scope.epoch=epoch; e->received+=receives;
    for(size_t i=0;i<c->function->count;i++) reset_argument(c,i,epoch,0);
    c->settled=0;
    atomic_store_explicit(&c->status,0,memory_order_release);
  }
  pthread_mutex_unlock(&e->lock); return error;
}

const struct mesh_view *mesh_call_view(const mesh_call *c, size_t i){ return i<c->function->count?&c->arguments[i].view:NULL; }
const uint32_t *mesh_call_indices(const mesh_call *c, size_t *bytes){ *bytes=c->indices_bytes; return c->indices; }

static void argument_fail(mesh_call *c, struct mesh_argument *a, int error){
  int status=atomic_load(&c->status);
  while(status>=0&&!atomic_compare_exchange_weak(&c->status,&status,-error)){}
  if(status>=0) change_call(c);
  atomic_store(&a->error,-atomic_load(&c->status));
  if(a->active){
    if(!a->stream.error){ a->stream.error=error; a->stream.fin_after_ns=0; a->stream.retry_ns=0; }
    a->stream.st=MS_FAIL;
  }
  int state=atomic_load(&a->state);
  while(state!=BORROWED && state!=COMPLETING && state!=INVALID && state!=SETTLED &&
        !atomic_compare_exchange_weak(&a->state,&state,INVALID)){}
}

static int advance_argument(mesh_call *c,size_t i){
  mesh_executor *e=c->executor; struct mesh_ctx *ctx=e->context;
  struct mesh_argument *a=&c->arguments[i]; const struct mesh_extent *x=&c->function->extents[i];
  int state=atomic_load_explicit(&a->state,memory_order_acquire), previous=state;
  if(ctx->mapping_pinned<0) argument_fail(c,a,ESTALE);
  if(a->active && a->stream.st==MS_FAIL) argument_fail(c,a,a->stream.error?a->stream.error:EIO);
  if(atomic_load(&c->status)<0) argument_fail(c,a,-atomic_load(&c->status));
  if(a->active && x->receive && a->stream.done==x->bytes){
    int waiting=WAITING;
    atomic_compare_exchange_strong_explicit(&a->state,&waiting,AVAILABLE,memory_order_release,memory_order_relaxed);
  }
  state=atomic_load_explicit(&a->state,memory_order_acquire);
  if(state==COMMITTED && atomic_load(&a->error)){ argument_fail(c,a,atomic_load(&a->error)); state=atomic_load(&a->state); }
  if(!a->released && (state==INVALID || (state==COMMITTED && (x->receive?a->stream.done==x->bytes:a->stream.st==MS_DONE)))){
    if(!a->active || (c->retain_transmit && !x->receive && state==COMMITTED) || !mesh_release_view(ctx,&a->stream)){
      a->released=1;
      if(x->receive) e->received-=a->page_count;
    } else change_argument(c,i);
  }
  if(a->released && (!a->active || (a->stream.st==MS_DONE && a->stream.closed &&
     (!a->stream.stride || mesh_stream_idle(ctx,&a->stream))) ||
     (atomic_load(&c->status)<0 && (!mesh_epoch_set(c->scope.epoch) || a->stream.abort_ack || ctx->mapping_pinned<0)))){
    a->stream.parked=1;
    atomic_store_explicit(&a->state,SETTLED,memory_order_release); state=SETTLED;
  } else if(a->released && a->stream.stride && a->stream.st==MS_DONE && a->stream.closed) change_argument(c,i);
  if(state==SETTLED && c->period && atomic_load(&c->status)>=0 &&
     c->period<=c->last && a->stream.scope.epoch.low<=c->last-c->period){
    struct mesh_epoch epoch=a->stream.scope.epoch; epoch.low+=c->period;
    reset_argument(c,i,epoch,1);
    if(x->receive) e->received+=a->page_count;
    state=atomic_load_explicit(&a->state,memory_order_acquire);
  }
  c->settled+=(state==SETTLED)-(previous==SETTLED);
  if(c->settled==c->function->count){
    int running=0; atomic_compare_exchange_strong_explicit(&c->status,&running,1,memory_order_release,memory_order_relaxed);
  }
  return state;
}

static void progress_arguments(mesh_executor *e){
  struct mesh_ctx *ctx=e->context;
  for(size_t j=0;j<e->count;j++){
    mesh_call *c=e->calls[j];
    for(size_t word=0;word<mesh_page_words(c->function->count);word++){
      if(!__atomic_load_n(&c->work[word],__ATOMIC_RELAXED)) continue;
      uint64_t work=__atomic_exchange_n(&c->work[word],0,__ATOMIC_ACQUIRE);
      for(;work;work&=work-1){
        size_t i=word*64+(size_t)__builtin_ctzll(work);
        struct mesh_argument *a=&c->arguments[i]; const struct mesh_extent *x=&c->function->extents[i];
        int state=advance_argument(c,i);
        if(state==SETTLED) continue;
        if(state==WAITING && atomic_load(&a->requested) && atomic_load(&c->status)>=0 &&
           (x->receive?!a->active:!a->leased)){
          if(x->receive){
            int error=mesh_lissen_view(ctx,&a->stream,a->pages,x->bytes,c->channel_base+x->channel);
            bind_argument_work(c,i);
            if(error){
              argument_fail(c,a,ENOMEM); continue; }
            a->stream.node=x->peer;
            a->stream.arrivals=a->pages+a->page_count;
          } else {
            void *p=a->active?mesh_stream_lease(ctx,&a->stream):mesh_yell_view(ctx,&a->stream,x->bytes,x->peer,c->channel_base+x->channel);
            bind_argument_work(c,i);
            if(!p){ change_argument(c,i); continue; }
            size_t first=((unsigned char*)p-mesh_at(ctx->M,0))/ctx->M->pgsz;
            for(size_t j=0;j<a->page_count;j++) a->pages[j]=(uint32_t)(first+j);
            a->leased=1;
            atomic_store_explicit(&a->state,AVAILABLE,memory_order_release);
          }
          a->stream.closing=!a->stream.resident; a->stream.parked=0; a->active=1;
          mesh_stream_schedule(&a->stream);
        }
      }
    }
  }
}

static void progress_streams(mesh_executor *e,uint64_t now_ns){
  size_t turn=e->context->stream_cursor++;
  for(size_t j=0;j<e->count;j++){
    mesh_call *c=e->calls[(turn%e->count+j)%e->count];
    size_t words=mesh_page_words(c->function->count), step=turn/e->count;
    for(size_t w=0;w<words;w++){
      size_t word=(step/64%words+w)%words;
      if(!__atomic_load_n(&c->runnable[word],__ATOMIC_RELAXED)) continue;
      uint64_t work=__atomic_exchange_n(&c->runnable[word],0,__ATOMIC_ACQUIRE);
      size_t shift=step%64;
      work=(work>>shift)|(work<<((64-shift)%64));
      for(;work;work&=work-1){
        size_t i=word*64+((size_t)__builtin_ctzll(work)+shift)%64;
        struct mstream *s=&c->arguments[i].stream;
        if(mesh_progress_stream(e->context,s,e->window,now_ns)) mesh_stream_schedule(s);
      }
    }
  }
}

int mesh_progress(mesh_executor *e){
  if(pthread_mutex_trylock(&e->lock)) return 0;
  struct mesh_ctx *ctx=e->context;
  progress_arguments(e);
  uint64_t now_ns;
  if(mesh_poll_streams(ctx,e->streams,(int)e->capacity,&now_ns)) progress_streams(e,now_ns);
  progress_arguments(e);
  int completed=0;
  for(size_t j=0;j<e->count;j++){
    mesh_call *c=e->calls[j];
    completed+=atomic_load(&c->status)==1;
  }
  pthread_mutex_unlock(&e->lock);
  return completed;
}

void mesh_request(mesh_call *c, size_t i){
  if(i<c->function->count && !atomic_exchange_explicit(&c->arguments[i].requested,1,memory_order_acq_rel)) change_argument(c,i);
}
int mesh_acquire(mesh_call *c, size_t i, struct mesh_view *view){
  if(i>=c->function->count) return -EINVAL;
  int error=atomic_load(&c->status); if(error<0) return error;
  struct mesh_argument *a=&c->arguments[i]; int available=AVAILABLE;
  mesh_request(c,i);
  if(!atomic_compare_exchange_strong_explicit(&a->state,&available,BORROWED,memory_order_acquire,memory_order_relaxed)) return 0;
  *view=a->view; return 1;
}
size_t mesh_join_indices(struct mesh_join_state *state,uint64_t generation,uint64_t required,
  uint64_t contribution,uint32_t offset,const uint32_t *indices,size_t count,uint32_t *ready){
  size_t n=0;
  for(size_t i=0;i<count;i++){
    uint32_t row=offset+indices[i]; struct mesh_join_state *s=&state[row];
    uint64_t previous=s->coverage*(s->generation==generation), coverage=previous|contribution;
    *s=(struct mesh_join_state){generation,coverage};
    ready[n]=row; n+=(coverage==required && previous!=required);
  }
  return n;
}

static void publish_word(struct mesh_argument *a,size_t index,uint64_t mask){
  struct mesh_page_bits *word=&a->stream.work[index];
  uint64_t added=mask&~__atomic_fetch_or(&word->published,mask,__ATOMIC_RELEASE);
  if(!added) return;
  size_t tail=a->page_count-1, capacity=a->view.capacity;
  size_t omitted=(tail/64==index && ((added>>(tail%64))&1))*(capacity-(a->view.bytes-tail*capacity));
  atomic_fetch_add_explicit(&a->stream.available,(size_t)__builtin_popcountll(added)*capacity-omitted,memory_order_release);
  __atomic_fetch_or(&word->pending,added,__ATOMIC_RELEASE);
  mesh_stream_schedule(&a->stream);
}
int mesh_publish(mesh_call *c, size_t i){
  if(i>=c->function->count || c->function->extents[i].receive) return -EINVAL;
  struct mesh_argument *a=&c->arguments[i];
  if(atomic_load_explicit(&a->state,memory_order_acquire)!=BORROWED) return -EINVAL;
  if(a->stream.paged) for(size_t word=0;word<mesh_page_words(a->page_count);word++){
    size_t bits=a->page_count-word*64; if(bits>64) bits=64;
    publish_word(a,word,UINT64_MAX>>(64-bits));
  }
  if(!a->stream.paged){
    atomic_store_explicit(&a->stream.available,a->view.bytes,memory_order_release);
    mesh_stream_schedule(&a->stream);
  }
  return 0;
}
int mesh_acquire_page(mesh_call *c, size_t i, size_t page){
  if(i>=c->function->count) return -EINVAL;
  int error=atomic_load(&c->status); if(error<0) return error;
  struct mesh_argument *a=&c->arguments[i];
  if(page>=a->page_count) return -EINVAL;
  if(!c->function->extents[i].receive) return mesh_page_published(c,i,page);
  if(a->pages[page]==UINT32_MAX) return 0;
  int state=atomic_load_explicit(&a->state,memory_order_acquire);
  if(state==WAITING || state==AVAILABLE)
    atomic_compare_exchange_strong_explicit(&a->state,&state,BORROWED,memory_order_acquire,memory_order_relaxed);
  return state==WAITING || state==AVAILABLE || state==BORROWED?1:-EINVAL;
}
ptrdiff_t mesh_ready_pages(const mesh_call *c,size_t i){
  if(i>=c->function->count) return -EINVAL;
  int error=atomic_load(&c->status); if(error<0) return error;
  const struct mesh_argument *a=&c->arguments[i];
  size_t bytes=c->function->extents[i].receive?a->stream.done:atomic_load_explicit(&a->stream.available,memory_order_acquire);
  return (ptrdiff_t)(bytes/a->view.capacity+(bytes%a->view.capacity!=0));
}
const uint32_t *mesh_arrived_pages(const mesh_call *c,size_t i,size_t *count){
  *count=0;
  if(i>=c->function->count || !c->function->extents[i].receive) return NULL;
  const struct mesh_argument *a=&c->arguments[i];
  *count=a->stream.done/a->view.capacity+(a->stream.done%a->view.capacity!=0);
  return a->stream.arrivals;
}
ptrdiff_t mesh_select_arrivals(mesh_call *c,size_t received,size_t produced,struct mesh_selection *selection,const uint32_t **selected){
  if(received>=c->function->count || produced>=c->function->count ||
     !c->function->extents[received].receive || c->function->extents[produced].receive) return -EINVAL;
  struct mesh_argument *rx=&c->arguments[received],*tx=&c->arguments[produced];
  if(!tx->stream.paged || tx->page_count<rx->page_count) return -EINVAL;
  if(!mesh_epoch_equal(rx->stream.scope.epoch,tx->stream.scope.epoch)){ *selected=NULL; return 0; }
  size_t count=0; mesh_arrived_pages(c,received,&count);
  size_t consumed=selection->consumed;
  if(consumed>count || selection->arrivals>count) return -EINVAL;
  size_t published=atomic_load_explicit(&tx->stream.available,memory_order_acquire);
  uint32_t *indices=rx->stream.arrivals;
  size_t begin=published?(published==selection->published?selection->arrivals:consumed):count;
  size_t end=consumed;
  if(published==tx->stream.n){ end=count; begin=count; }
  for(size_t i=begin;i<count;i++){
    uint32_t enabled=(uint32_t)mesh_stream_published(&tx->stream,indices[i]);
    uint32_t swap=(indices[i]^indices[end])*(uint32_t)enabled;
    indices[i]^=swap; indices[end]^=swap; end+=enabled;
  }
  if(end>consumed && !consumed){
    int status=mesh_acquire_page(c,received,indices[0]);
    if(status!=1) return status?status:-EAGAIN;
  }
  selection->arrivals=count; selection->published=published;
  *selected=indices?indices+consumed:NULL;
  return (ptrdiff_t)(end-consumed);
}
int mesh_publish_page(mesh_call *c, size_t i, size_t page){
  if(i>=c->function->count || c->function->extents[i].receive) return -EINVAL;
  struct mesh_argument *a=&c->arguments[i]; struct mstream *s=&a->stream;
  if(!s->paged || page>=a->page_count || atomic_load_explicit(&a->state,memory_order_acquire)!=BORROWED) return -EINVAL;
  publish_word(a,page/64,UINT64_C(1)<<(page%64));
  return 0;
}
int mesh_page_published(const mesh_call *c,size_t i,size_t page){
  if(i>=c->function->count || c->function->extents[i].receive) return -EINVAL;
  const struct mesh_argument *a=&c->arguments[i];
  if(!a->stream.paged || page>=a->page_count) return -EINVAL;
  if(!a->stream.work) return -EINVAL;
  int published=mesh_stream_published(&a->stream,page);
  if(!published && atomic_load_explicit(&a->state,memory_order_acquire)!=BORROWED) return -EINVAL;
  return published;
}
ptrdiff_t mesh_flush(mesh_call *c,size_t i){
  if(i>=c->function->count || c->function->extents[i].receive) return -EINVAL;
  struct mesh_argument *a=&c->arguments[i];
  if(!a->stream.paged) return -EINVAL;
  return mesh_stream_push_pages(c->executor->context,&a->stream,c->executor->window);
}
int mesh_complete(mesh_call *c, size_t i, int error){
  if(i>=c->function->count || error<0) return -EINVAL;
  struct mesh_argument *a=&c->arguments[i];
  int borrowed=BORROWED;
  if(!atomic_compare_exchange_strong(&a->state,&borrowed,COMPLETING)) return -EINVAL;
  atomic_store(&a->error,error);
  if(!error && !c->function->extents[i].receive && !a->stream.paged)
    atomic_store_explicit(&a->stream.available,a->view.bytes,memory_order_release);
  atomic_store_explicit(&a->state,COMMITTED,memory_order_release);
  change_argument(c,i);
  return 0;
}
size_t mesh_call_unsent(const mesh_call *c){
  size_t pending=0;
  for(size_t i=0;i<c->function->count;i++){
    const struct mesh_argument *a=&c->arguments[i]; const struct mstream *s=&a->stream;
    if(c->function->extents[i].receive || !a->active || s->st!=MS_RUN) continue;
    if(s->paged){
      size_t bytes=atomic_load_explicit(&s->available,memory_order_acquire)-s->off;
      pending+=bytes/a->view.capacity+(bytes%a->view.capacity!=0);
    }
    else pending+=s->off<atomic_load_explicit(&s->available,memory_order_acquire);
  }
  return pending;
}
int mesh_call_status(const mesh_call *c){ return atomic_load_explicit(&c->status,memory_order_acquire); }
void mesh_call_cancel(mesh_call *c, int error){
  int status=atomic_load(&c->status);
  while(status>=0&&!atomic_compare_exchange_weak(&c->status,&status,-(error>0?error:ECANCELED))){}
  if(status>=0) change_call(c);
}

static int retire_call(mesh_call *c, int abandon){
  if(abandon && (!mesh_epoch_set(c->scope.epoch) || mesh_call_status(c)>=0)) return EINVAL;
  mesh_executor *e=c->executor;
  if(!e){ mesh_function_free(c->function); free(c); return 0; }
  pthread_mutex_lock(&e->lock);
  int busy=0;
  for(size_t i=0;i<c->function->count;i++){
    struct mesh_argument *a=&c->arguments[i];
    int state=atomic_load(&a->state);
    busy|=abandon?(!a->released || state==BORROWED || state==COMPLETING):state!=SETTLED;
  }
  if(busy){ pthread_mutex_unlock(&e->lock); return EBUSY; }
  size_t at=0,first=0;
  for(;at<e->count && e->calls[at]!=c;at++) first+=e->calls[at]->function->count;
  memmove(e->calls+at,e->calls+at+1,(e->count-at-1)*sizeof *e->calls); e->count--;
  e->capacity-=c->function->count;
  memmove(e->streams+first,e->streams+first+c->function->count,(e->capacity-first)*sizeof *e->streams);
  mesh_stream_register(e->context,e->streams,e->capacity);
  for(size_t i=0;i<c->function->count;i++)
    if(c->arguments[i].stream.stride) mesh_release_view(e->context,&c->arguments[i].stream);
  pthread_mutex_unlock(&e->lock);
  for(size_t i=0;i<c->function->count;i++) free(c->arguments[i].stream.seen);
  munmap(c->indices,c->indices_bytes);
  mesh_function_free(c->function); free(c); return 0;
}
int mesh_call_retire(mesh_call *c){ return retire_call(c,0); }
int mesh_call_abandon(mesh_call *c){ return retire_call(c,1); }
int mesh_executor_free(mesh_executor *e){
  pthread_mutex_lock(&e->lock);
  int busy=e->count!=0 || (e->context->mapping_pinned>=0 && e->context->inflight!=0);
  pthread_mutex_unlock(&e->lock);
  if(busy) return EBUSY;
  atomic_store(&e->context->executor_attached,0);
  mesh_stream_register(e->context,NULL,0);
  pthread_mutex_destroy(&e->lock); free(e->calls); free(e->streams); free(e); return 0;
}

static void view_copy(const struct mesh_view *v, size_t offset, void *data, size_t bytes, int write){
  size_t payload=v->capacity?v->capacity:v->stride-v->payload;
  while(bytes){
    size_t page=offset/payload, position=offset%payload, n=payload-position; if(n>bytes) n=bytes;
    unsigned char *p=v->base+(size_t)v->pages[page]*v->stride+v->payload+position;
    if(write) memcpy(p,data,n); else memcpy(data,p,n);
    offset+=n; data=(unsigned char*)data+n; bytes-=n;
  }
}
void mesh_view_read(const struct mesh_view *v, size_t offset, void *out, size_t bytes){ view_copy(v,offset,out,bytes,0); }
void mesh_view_write(const struct mesh_view *v, size_t offset, const void *in, size_t bytes){ view_copy(v,offset,(void*)in,bytes,1); }
