#include "mesh-pages.h"
#include "mesh-wire.h"
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define CONTROL UINT32_MAX
#define UNUSED (UINT32_MAX-1)
#define ABSENT UINT32_MAX
#define RETRY_CAP 1000000000ull
#define WINDOW (MESH_RING/2)
#define MIX 0xff51afd7ed558ccdULL
#define MIX2 0xc4ceb9fe1a85ec53ULL

struct slot {
  struct mesh_pages_slot spec;
  uint32_t *table; uint64_t *stamp; unsigned char *inflight;
  _Atomic uint64_t *publish[2]; _Atomic uint64_t publishing[2]; uint64_t *pending; size_t words;
  size_t base;
  uint32_t flying;
  _Atomic uint64_t fill_generation[2], fill_count[2], highest, complete, producible;
  uint64_t released;
  uint32_t dependent[MESH_PAGES_DEPENDENCIES], dependents;
  uint64_t fault_generation; uint32_t fault_first, fault_second, fault_have;
  int agreed, open_due, abort_due, transported;
  uint64_t open_ns, open_retry, peer_nonce;
};
struct mesh_pages {
  struct mesh_ctx *context; struct hdr *M;
  struct mesh_epoch epoch; unsigned char plan[32];
  struct slot *slots; size_t count;
  uint32_t versions; struct mesh_pages_policy policy;
  uint32_t *table; size_t table_bytes; uint64_t *stamps;
  uint32_t *owner; uint32_t *control; size_t control_count, control_free;
  uint32_t *later; size_t later_count;
  size_t flying;
  pthread_t thread; _Atomic int running, status; int announced;
  uint64_t nonce, incarnation;
  mesh_pages_hook hook; void *capture;
  uint64_t now, refreshed, integrity, stale, duplicates, overwrites, faults;
};

static uint64_t clock_ns(void){
  struct timespec clock; clock_gettime(CLOCK_MONOTONIC,&clock);
  return (uint64_t)clock.tv_sec*1000000000u+clock.tv_nsec; }
static size_t header_bytes(void){ return sizeof(struct wire)+sizeof(struct mesh_frame); }
static void *zeroed(size_t n){ void *p=calloc(1,n?n:1); if(!p) errno=ENOMEM; return p; }
static uint64_t mix(uint64_t x){ x^=x>>33; x*=MIX; x^=x>>33; x*=MIX2; x^=x>>33; return x; }
static uint64_t fresh_nonce(void){ uint64_t v=mix(clock_ns()^((uint64_t)getpid()<<32)); return v?v:1; }
static void agreement(const mesh_pages *p, const struct slot *s, unsigned char body[48]){
  memcpy(body,p->plan,32); memcpy(body+32,&s->spec.pages,4); memcpy(body+36,&p->versions,4); memcpy(body+40,&p->nonce,8); }
static unsigned char *payload_at(const mesh_pages *p, uint32_t page){ return mesh_at(p->M,page)+header_bytes(); }
static uint32_t parity(const mesh_pages *p, uint64_t generation){ return (uint32_t)(((generation-1)/p->versions)&1); }
static int faulted(const mesh_pages *p, const struct slot *s, uint64_t generation, uint32_t *first, uint32_t *second){
  if(!p->policy.fault_period || s->spec.pages<2) return 0;
  uint64_t r=mix(p->policy.fault_seed^((uint64_t)s->spec.sid<<32)^generation*0x9e3779b97f4a7c15ULL);
  if(r%p->policy.fault_period) return 0;
  uint64_t a=mix(r+1)%s->spec.pages, b=mix(r+2)%(s->spec.pages-1);
  if(b>=a) b++;
  *first=(uint32_t)a; *second=(uint32_t)b; return 1;
}

__attribute__((target("crc")))
uint64_t mesh_pages_hash(const void *data, size_t bytes, uint64_t seed){
  const unsigned char *d=data; size_t n=bytes/32;
  uint32_t a=(uint32_t)seed, b=(uint32_t)(seed>>32), c=~a, e=~b;
  for(size_t i=0;i<n;i++){
    uint64_t w0,w1,w2,w3; memcpy(&w0,d+i*32,8); memcpy(&w1,d+i*32+8,8); memcpy(&w2,d+i*32+16,8); memcpy(&w3,d+i*32+24,8);
    a=__builtin_arm_crc32cd(a,w0); b=__builtin_arm_crc32cd(b,w1); c=__builtin_arm_crc32cd(c,w2); e=__builtin_arm_crc32cd(e,w3);
  }
  for(size_t i=n*32;i<bytes;i++) a=__builtin_arm_crc32cb(a,d[i]);
  return mix(((uint64_t)a<<32|b)^mix((uint64_t)c<<32|e));
}

mesh_pages *mesh_pages_compile(struct mesh_ctx *context, struct mesh_epoch epoch, const unsigned char plan[32],
  const struct mesh_pages_slot *slots, size_t count, uint32_t versions, struct mesh_pages_policy policy){
  if(!context || !context->M || !mesh_epoch_set(epoch) || !count || versions<2 || !policy.control_pages){ errno=EINVAL; return NULL; }
  struct hdr *M=context->M;
  mesh_pages *p=zeroed(sizeof *p); if(!p) return NULL;
  p->context=context; p->M=M; p->epoch=epoch; memcpy(p->plan,plan,32);
  p->versions=versions; p->policy=policy; p->nonce=fresh_nonce();
  if(!p->policy.open_retry_ns) p->policy.open_retry_ns=50000;
  p->count=count;
  p->slots=zeroed(count*sizeof *p->slots);
  p->owner=zeroed((size_t)M->arena*sizeof *p->owner);
  if(p->owner) for(uint32_t j=0;j<M->arena;j++) p->owner[j]=UNUSED;
  p->control=zeroed(policy.control_pages*sizeof *p->control);
  p->later=zeroed(MESH_RING*sizeof *p->later);
  if(!p->slots || !p->owner || !p->control || !p->later){ mesh_pages_free(p); return NULL; }
  size_t entries=0, arena=0;
  for(size_t i=0;i<count;i++){
    const struct mesh_pages_slot *x=&slots[i];
    if(!x->pages || x->pages>M->pool || x->depends>MESH_PAGES_DEPENDENCIES){ mesh_pages_free(p); errno=EINVAL; return NULL; }
    for(uint8_t d=0;d<x->depends;d++) if(x->dependency[d]>=count){ mesh_pages_free(p); errno=EINVAL; return NULL; }
    entries+=x->pages;
    if(!x->receive) arena+=x->pages;
  }
  if(arena+policy.control_pages>M->arena){ mesh_pages_free(p); errno=ENOMEM; return NULL; }
  size_t alignment=(size_t)getpagesize();
  p->table_bytes=(entries*sizeof(uint32_t)+alignment-1)/alignment*alignment;
  p->table=mmap(NULL,p->table_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
  if(p->table==MAP_FAILED){ p->table=NULL; mesh_pages_free(p); return NULL; }
  memset(p->table,0xff,p->table_bytes);
  p->stamps=zeroed(entries*sizeof *p->stamps);
  if(!p->stamps){ mesh_pages_free(p); return NULL; }
  arena=0; entries=0;
  for(size_t i=0;i<count;i++){
    struct slot *s=&p->slots[i]; s->spec=slots[i];
    s->table=p->table+entries; s->stamp=p->stamps+entries; entries+=s->spec.pages;
    s->words=mesh_page_words(s->spec.pages);
    s->inflight=zeroed(s->spec.pages);
    s->publish[0]=zeroed(s->words*sizeof *s->publish[0]); s->publish[1]=zeroed(s->words*sizeof *s->publish[1]);
    s->pending=zeroed(s->words*sizeof *s->pending);
    if(!s->inflight || !s->publish[0] || !s->publish[1] || !s->pending){ mesh_pages_free(p); return NULL; }
    s->transported=!s->spec.receive && s->spec.peer!=MESH_PAGES_LOCAL;
    if(!s->spec.receive){
      s->base=arena;
      for(uint32_t j=0;j<s->spec.pages;j++){ s->table[j]=M->pool+(uint32_t)(arena+j); p->owner[arena+j]=(uint32_t)i; }
      arena+=s->spec.pages;
    }
    if(!s->transported) s->agreed=1;
    for(uint8_t d=0;d<s->spec.depends;d++){
      struct slot *a=&p->slots[s->spec.dependency[d]];
      if(a->dependents==MESH_PAGES_DEPENDENCIES){ mesh_pages_free(p); errno=E2BIG; return NULL; }
      a->dependent[a->dependents++]=(uint32_t)i;
    }
  }
  for(uint32_t j=0;j<policy.control_pages;j++){ p->owner[arena+j]=CONTROL; p->control[j]=(uint32_t)(arena+j); }
  p->control_count=p->control_free=policy.control_pages;
  return p;
}

int mesh_pages_free(mesh_pages *p){
  if(!p) return 0;
  mesh_pages_stop(p);
  for(size_t i=0;i<p->count;i++){ struct slot *s=&p->slots[i]; free(s->inflight); free((void*)s->publish[0]); free((void*)s->publish[1]); free(s->pending); }
  if(p->table) munmap(p->table,p->table_bytes);
  free(p->stamps); free(p->slots); free(p->owner); free(p->control); free(p->later); free(p);
  return 0;
}

size_t mesh_pages_header(const mesh_pages *p){ (void)p; return header_bytes(); }
size_t mesh_pages_payload(const mesh_pages *p){ return p->M->pgsz-header_bytes(); }
const uint32_t *mesh_pages_entries(const mesh_pages *p, uint32_t slot){ return slot<p->count?p->slots[slot].table:NULL; }
const uint64_t *mesh_pages_stamps(const mesh_pages *p, uint32_t slot){ return slot<p->count?p->slots[slot].stamp:NULL; }
uint32_t mesh_pages_filled(const mesh_pages *p, uint32_t slot, uint64_t generation){
  if(slot>=p->count || !generation) return 0;
  const struct slot *s=&p->slots[slot]; uint32_t k=parity(p,generation);
  if(atomic_load_explicit(&s->fill_generation[k],memory_order_acquire)!=generation) return 0;
  return (uint32_t)atomic_load_explicit(&s->fill_count[k],memory_order_acquire);
}
uint64_t mesh_pages_highest(const mesh_pages *p, uint32_t slot){ return slot<p->count?atomic_load_explicit(&p->slots[slot].highest,memory_order_acquire):0; }
void mesh_pages_produce_hook(mesh_pages *p, mesh_pages_hook hook, void *capture){ p->hook=hook; p->capture=capture; }
uint64_t mesh_pages_producible(const mesh_pages *p, uint32_t slot){
  return slot<p->count?atomic_load_explicit(&p->slots[slot].producible,memory_order_acquire):0; }
int mesh_pages_faulted(const mesh_pages *p, uint32_t slot, uint64_t generation, uint32_t *first, uint32_t *second){
  return slot<p->count && p->slots[slot].spec.receive && faulted(p,&p->slots[slot],generation,first,second); }
int mesh_pages_agreed(const mesh_pages *p, uint32_t slot){ return slot<p->count && __atomic_load_n(&p->slots[slot].agreed,__ATOMIC_ACQUIRE); }
int mesh_pages_status(const mesh_pages *p){ return atomic_load_explicit(&p->status,memory_order_acquire); }
uint64_t mesh_pages_incarnation(const mesh_pages *p){ return p->incarnation; }
void mesh_pages_fail(mesh_pages *p, int error){
  int expected=0; atomic_compare_exchange_strong(&p->status,&expected,-(error>0?error:ECANCELED));
}

int mesh_pages_publish(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation){
  if(slot>=p->count || p->slots[slot].spec.receive || !generation) return -EINVAL;
  struct slot *s=&p->slots[slot];
  if(first>s->spec.pages || count>s->spec.pages-first) return -EINVAL;
  if(atomic_load_explicit(&s->producible,memory_order_acquire)<generation) return -EBUSY;
  uint32_t k=parity(p,generation);
  atomic_store_explicit(&s->publishing[k],generation,memory_order_release);
  for(size_t at=first,end=first+count;at<end;){
    size_t bit=at%64, bits=end-at; if(bits>64-bit) bits=64-bit;
    atomic_fetch_or_explicit(&s->publish[k][at/64],(UINT64_MAX>>(64-bits))<<bit,memory_order_release);
    at+=bits;
  }
  return 0;
}

static uint32_t take_control(mesh_pages *p){ return p->control_free?p->control[--p->control_free]:CONTROL; }
static int control(mesh_pages *p, const struct slot *s, uint32_t kind, uint64_t offset, uint64_t generation, const void *body, size_t bytes){
  uint32_t page=take_control(p); if(page==CONTROL) return -1;
  unsigned char *q=mesh_at(p->M,p->M->pool+page)+sizeof(struct wire);
  struct mesh_frame frame={.h={offset,s->spec.sid,kind},.epoch={p->epoch.high,generation},.source=(uint16_t)p->M->node,.target=s->spec.peer};
  size_t n=mesh_frame_encode(q,&frame,0);
  if(bytes) memcpy(q+n,body,bytes);
  struct desc d={.page=p->M->pool+page,.bytes=(uint32_t)(n+bytes),.node=s->spec.peer};
  if(push(p->M,SUB,&d)){ p->control[p->control_free++]=page; return -1; }
  p->flying++; return 0;
}
static int push_page(mesh_pages *p, struct slot *s, uint32_t page){
  if(s->inflight[page] || p->flying>=WINDOW) return 0;
  unsigned char *q=mesh_at(p->M,s->table[page])+sizeof(struct wire);
  struct mesh_frame frame={.h={page,s->spec.sid,K_DATA},.epoch={p->epoch.high,s->stamp[page]},.source=(uint16_t)p->M->node,.target=s->spec.peer};
  size_t n=mesh_frame_encode(q,&frame,mesh_pages_payload(p));
  struct desc d={.page=s->table[page],.bytes=(uint32_t)n,.node=s->spec.peer};
  if(push(p->M,SUB,&d)) return 0;
  s->inflight[page]=1; s->flying++; p->flying++;
  return 1;
}
static void flush_later(mesh_pages *p){
  while(p->later_count){
    struct desc d={.page=p->later[0]};
    if(push(p->M,REL,&d)) break;
    memmove(p->later,p->later+1,(--p->later_count)*sizeof *p->later);
  }
}
static void release_page(mesh_pages *p, uint32_t page){
  struct desc d={.page=page};
  if(push(p->M,REL,&d) && p->later_count<MESH_RING) p->later[p->later_count++]=page;
}
static void filled(mesh_pages *p, struct slot *s, uint64_t generation){
  uint32_t k=parity(p,generation);
  if(atomic_load_explicit(&s->fill_generation[k],memory_order_relaxed)!=generation){
    atomic_store_explicit(&s->fill_count[k],0,memory_order_relaxed);
    atomic_store_explicit(&s->fill_generation[k],generation,memory_order_release);
  }
  uint64_t count=atomic_fetch_add_explicit(&s->fill_count[k],1,memory_order_release)+1;
  if(generation>atomic_load_explicit(&s->highest,memory_order_relaxed)) atomic_store_explicit(&s->highest,generation,memory_order_release);
  if(count==s->spec.pages && generation>atomic_load_explicit(&s->complete,memory_order_relaxed)) atomic_store_explicit(&s->complete,generation,memory_order_release);
}
static void release_entry(mesh_pages *p, struct slot *a, uint32_t j, uint64_t generation){
  if(a->stamp[j]!=generation || a->table[j]==ABSENT) return;
  release_page(p,a->table[j]); __atomic_store_n(&a->table[j],ABSENT,__ATOMIC_RELEASE);
}
static void release_dependencies(mesh_pages *p, struct slot *s, uint32_t page, uint64_t generation){
  if(s->spec.pagewise){
    for(uint8_t d=0;d<s->spec.depends;d++){
      struct slot *a=&p->slots[s->spec.dependency[d]];
      if(a->spec.receive && page<a->spec.pages) release_entry(p,a,page,generation);
    }
    return;
  }
  if(generation<=s->released) return;
  s->released=generation;
  for(uint8_t d=0;d<s->spec.depends;d++){
    struct slot *a=&p->slots[s->spec.dependency[d]];
    if(!a->spec.receive) continue;
    for(uint32_t j=0;j<a->spec.pages;j++) release_entry(p,a,j,generation);
  }
}
static void arrive(mesh_pages *p, struct slot *s, uint32_t page, uint32_t physical, uint64_t generation){
  __atomic_store_n(&s->stamp[page],generation,__ATOMIC_RELEASE);
  __atomic_store_n(&s->table[page],physical,__ATOMIC_RELEASE);
  if(s->fault_generation!=generation){
    s->fault_generation=generation; s->fault_have=0;
    if(!faulted(p,s,generation,&s->fault_first,&s->fault_second)) s->fault_have=4;
  }
  if(s->fault_have<4 && (page==s->fault_first || page==s->fault_second)){
    s->fault_have|=page==s->fault_first?1:2;
    if(s->fault_have==3){
      uint32_t first=s->table[s->fault_first], second=s->table[s->fault_second];
      size_t bytes=mesh_pages_payload(p);
      if(first!=ABSENT && second!=ABSENT){
        unsigned char *a=payload_at(p,first), *b=payload_at(p,second);
        for(size_t i=0;i<bytes;i++){ unsigned char x=a[i]^b[i]; a[i]=x; b[i]=x; }
      } else {
        unsigned char *a=payload_at(p,first!=ABSENT?first:second);
        for(size_t i=0;i<bytes;i++) a[i]^=0xa5;
      }
      p->faults++; s->fault_have=4;
    }
  }
  filled(p,s,generation);
}
static void offer(mesh_pages *p, size_t i){
  struct slot *s=&p->slots[i];
  if(s->spec.receive) return;
  uint64_t limit;
  if(s->dependents){
    limit=UINT64_MAX;
    for(uint32_t d=0;d<s->dependents;d++){
      const struct slot *b=&p->slots[s->dependent[d]];
      uint64_t shown=b->spec.pagewise && !s->transported?atomic_load_explicit(&b->complete,memory_order_acquire):atomic_load_explicit(&b->highest,memory_order_acquire);
      if(shown<limit) limit=shown;
    }
    limit+=p->versions;
  } else limit=(s->transported && s->flying?0:atomic_load_explicit(&s->complete,memory_order_acquire))+p->versions;
  if(limit<=atomic_load_explicit(&s->producible,memory_order_relaxed)) return;
  atomic_store_explicit(&s->producible,limit,memory_order_release);
  if(p->hook) p->hook(p->capture,(uint32_t)i,limit);
}

static struct slot *lookup(mesh_pages *p, uint32_t sid, int receive, int from){
  for(size_t i=0;i<p->count;i++){
    struct slot *s=&p->slots[i];
    if(s->spec.sid==sid && (s->spec.receive!=0)==(receive!=0) && s->spec.peer==from) return s;
  }
  return NULL;
}
static void receive(mesh_pages *p, uint32_t page, size_t bytes, int from){
  int keep=0;
  unsigned char *q=mesh_at(p->M,page)+sizeof(struct wire);
  struct mesh_frame frame; size_t header=mesh_frame_decode(q,bytes,&frame);
  if(header!=sizeof frame || frame.source!=from || frame.target!=p->M->node){ p->integrity++; goto done; }
  if(frame.epoch.high!=p->epoch.high){ p->stale++; goto done; }
  struct slot *s=lookup(p,frame.h.sid,mesh_frame_receive(frame.h.k),from);
  if(!s){ p->stale++; goto done; }
  uint64_t g=frame.epoch.low;
  switch(frame.h.k){
  case K_OPEN: {
    unsigned char body[48]; agreement(p,s,body);
    uint64_t nonce; memcpy(&nonce,q+header+40,8);
    if(bytes!=header+48 || memcmp(q+header,body,40)){ mesh_pages_fail(p,EPROTO); break; }
    if(s->peer_nonce && s->peer_nonce!=nonce){ mesh_pages_fail(p,ECONNRESET); break; }
    s->peer_nonce=nonce;
    __atomic_store_n(&s->agreed,1,__ATOMIC_RELEASE); s->open_due=1;
    break; }
  case K_READY: {
    uint64_t nonce=0; if(bytes==header+8) memcpy(&nonce,q+header,8);
    if(s->peer_nonce && nonce && s->peer_nonce!=nonce){ mesh_pages_fail(p,ECONNRESET); break; }
    if(nonce) s->peer_nonce=nonce;
    __atomic_store_n(&s->agreed,1,__ATOMIC_RELEASE);
    break; }
  case K_DATA: {
    uint32_t at=(uint32_t)frame.h.off;
    if(at>=s->spec.pages || bytes!=header+mesh_pages_payload(p)){ p->integrity++; break; }
    if(g<=s->stamp[at]){ if(g==s->stamp[at]) p->duplicates++; else p->stale++; break; }
    if(s->table[at]!=ABSENT){ p->overwrites++; release_page(p,s->table[at]); }
    keep=1; arrive(p,s,at,page,g);
    break; }
  case K_ABORT_RX: case K_ABORT_TX:
    mesh_pages_fail(p,frame.h.off>0&&frame.h.off<=INT32_MAX?(int)frame.h.off:ECANCELED);
    break;
  default: p->stale++; break;
  }
done:
  if(!keep) release_page(p,page);
}

static void acknowledge(mesh_pages *p){
  struct desc d;
  while(!pop(p->M,ACK,&d)){
    if(d.page<p->M->pool || d.page>=p->M->pool+p->M->arena) continue;
    uint32_t index=d.page-p->M->pool, owner=p->owner[index];
    if(owner!=CONTROL && owner>=p->count) continue;
    if(p->flying) p->flying--;
    if(owner==CONTROL){ p->control[p->control_free++]=index; continue; }
    struct slot *s=&p->slots[owner]; size_t page=index-s->base;
    if(s->inflight[page]){ s->inflight[page]=0; s->flying--; }
  }
}
static void transmit(mesh_pages *p, size_t i){
  struct slot *s=&p->slots[i];
  if(s->spec.receive) return;
  for(uint32_t k=0;k<2;k++){
    for(size_t w=0;w<s->words;w++){
      uint64_t bits=atomic_exchange_explicit(&s->publish[k][w],0,memory_order_acquire);
      if(!bits) continue;
      uint64_t generation=atomic_load_explicit(&s->publishing[k],memory_order_acquire);
      for(;bits;bits&=bits-1){
        uint32_t page=(uint32_t)(w*64+(size_t)__builtin_ctzll(bits));
        if(s->stamp[page]>=generation) continue;
        s->stamp[page]=generation;
        release_dependencies(p,s,page,generation);
        if(s->transported) s->pending[w]|=UINT64_C(1)<<(page%64);
        filled(p,s,generation);
      }
    }
  }
  if(!s->transported) return;
  if(!s->agreed){
    if(p->now>=s->open_ns){
      unsigned char body[48]; agreement(p,s,body);
      if(!control(p,s,K_OPEN,s->spec.pages,0,body,48)){
        s->open_retry=s->open_retry?(s->open_retry<RETRY_CAP/2?s->open_retry*2:RETRY_CAP):p->policy.open_retry_ns;
        s->open_ns=p->now+s->open_retry;
      }
    }
    return;
  }
  for(size_t w=0;w<s->words && p->flying<WINDOW;w++){
    uint64_t bits=s->pending[w];
    for(;bits && p->flying<WINDOW;bits&=bits-1){
      uint32_t page=(uint32_t)(w*64+(size_t)__builtin_ctzll(bits));
      if(push_page(p,s,page)) s->pending[w]&=~(UINT64_C(1)<<(page%64));
    }
  }
}
static void answer(mesh_pages *p, size_t i){
  struct slot *s=&p->slots[i];
  if(!s->spec.receive) return;
  if(s->open_due && !control(p,s,K_READY,0,0,&p->nonce,8)) s->open_due=0;
}
static void abort_peers(mesh_pages *p){
  int error=-atomic_load_explicit(&p->status,memory_order_acquire);
  for(size_t i=0;i<p->count;i++){
    struct slot *s=&p->slots[i];
    if(!s->abort_due || (!s->spec.receive && !s->transported)) continue;
    if(!control(p,s,s->spec.receive?K_ABORT_TX:K_ABORT_RX,(uint64_t)error,0,NULL,0)) s->abort_due=0;
  }
}
static int linked(const mesh_pages *p){
  uint64_t ports=atomic_load_explicit(&p->M->port_count,memory_order_acquire);
  for(uint64_t i=0;i<ports;i++) if(atomic_load_explicit(&mesh_ports(p->M)[i].phase,memory_order_acquire)==MESH_PAIRED) return 1;
  return ports==0;
}
static void refresh(mesh_pages *p){
  if(atomic_load(&p->M->phase)>=MESH_STOPPING){ mesh_pages_fail(p,ESTALE); return; }
  if(p->now-p->refreshed<250000000ull) return;
  p->refreshed=p->now;
  uint64_t pid=atomic_load(&p->M->bridge_pid);
  if(pid && kill((pid_t)pid,0)<0 && errno==ESRCH){ mesh_pages_fail(p,ESTALE); return; }
  if(!linked(p)) mesh_pages_fail(p,ENOTCONN);
}

int mesh_pages_progress(mesh_pages *p){
  p->now=clock_ns();
  int status=atomic_load_explicit(&p->status,memory_order_acquire);
  if(status<0 && !p->announced){ p->announced=1; for(size_t i=0;i<p->count;i++) p->slots[i].abort_due=1; }
  refresh(p);
  acknowledge(p);
  flush_later(p);
  for(int turn=0;turn<256;turn++){
    struct desc d; if(pop(p->M,CMP,&d)) break;
    if(d.page>=p->M->pool){ p->integrity++; continue; }
    receive(p,d.page,d.bytes,d.node);
  }
  if(status<0){ abort_peers(p); return status; }
  for(size_t i=0;i<p->count;i++){ transmit(p,i); answer(p,i); }
  for(size_t i=0;i<p->count;i++) offer(p,i);
  return atomic_load_explicit(&p->status,memory_order_acquire);
}
static void *run(void *argument){
  mesh_pages *p=argument;
  pthread_setname_np("mesh-pages");
  unsigned idle=0;
  while(atomic_load_explicit(&p->running,memory_order_relaxed)){
    size_t before=p->flying; uint64_t highest=0;
    for(size_t i=0;i<p->count;i++) highest+=atomic_load_explicit(&p->slots[i].highest,memory_order_relaxed);
    mesh_pages_progress(p);
    uint64_t after=0; for(size_t i=0;i<p->count;i++) after+=atomic_load_explicit(&p->slots[i].highest,memory_order_relaxed);
    idle=before==p->flying && highest==after?idle+1:0;
    if(idle>4096){ sched_yield(); idle=4096; }
  }
  return NULL;
}
int mesh_pages_start(mesh_pages *p){
  if(atomic_exchange(&p->running,1)) return 0;
  int error=pthread_create(&p->thread,NULL,run,p);
  if(error) atomic_store(&p->running,0);
  return error;
}
void mesh_pages_stop(mesh_pages *p){ if(atomic_exchange(&p->running,0)) pthread_join(p->thread,NULL); }

int mesh_pages_recover(mesh_pages *p){
  if(atomic_load_explicit(&p->running,memory_order_acquire)) return EBUSY;
  if(atomic_load_explicit(&p->status,memory_order_acquire)==-ESTALE) return ESTALE;
  uint64_t began=clock_ns();
  while(p->flying && clock_ns()-began<2000000000ull){ p->now=clock_ns(); acknowledge(p); sched_yield(); }
  for(size_t i=0;i<p->count;i++){
    struct slot *s=&p->slots[i];
    if(s->spec.receive) for(uint32_t j=0;j<s->spec.pages;j++) if(s->table[j]!=ABSENT){ release_page(p,s->table[j]); s->table[j]=ABSENT; }
    memset(s->stamp,0,s->spec.pages*sizeof *s->stamp);
    memset(s->inflight,0,s->spec.pages);
    for(size_t w=0;w<s->words;w++){ atomic_store(&s->publish[0][w],0); atomic_store(&s->publish[1][w],0); s->pending[w]=0; }
    atomic_store(&s->publishing[0],0); atomic_store(&s->publishing[1],0);
    for(int k=0;k<2;k++){ atomic_store(&s->fill_generation[k],0); atomic_store(&s->fill_count[k],0); }
    atomic_store(&s->highest,0); atomic_store(&s->complete,0); atomic_store(&s->producible,0);
    s->flying=0; s->released=0; s->fault_generation=0; s->fault_have=0;
    s->agreed=!s->transported; s->open_due=s->abort_due=0; s->open_ns=s->open_retry=0; s->peer_nonce=0;
  }
  flush_later(p);
  p->flying=0; p->control_free=p->control_count;
  p->nonce=fresh_nonce(); p->incarnation++; p->announced=0;
  atomic_store_explicit(&p->status,0,memory_order_release);
  return 0;
}

int mesh_pages_settled(const mesh_pages *p){
  if(p->flying) return 0;
  for(size_t i=0;i<p->count;i++){
    const struct slot *s=&p->slots[i];
    if(s->spec.receive) continue;
    for(size_t w=0;w<s->words;w++)
      if(s->pending[w] || atomic_load_explicit(&s->publish[0][w],memory_order_acquire) || atomic_load_explicit(&s->publish[1][w],memory_order_acquire)) return 0;
  }
  return 1;
}
size_t mesh_pages_describe(const mesh_pages *p, char *out, size_t bytes){
  size_t n=(size_t)snprintf(out,bytes,"status=%d incarnation=%llu flying=%zu integrity=%llu stale=%llu duplicates=%llu overwrites=%llu faults=%llu control_free=%zu/%zu\n",
    atomic_load(&p->status),(unsigned long long)p->incarnation,p->flying,(unsigned long long)p->integrity,(unsigned long long)p->stale,
    (unsigned long long)p->duplicates,(unsigned long long)p->overwrites,(unsigned long long)p->faults,p->control_free,p->control_count);
  for(size_t i=0;i<p->count && n<bytes;i++){
    const struct slot *s=&p->slots[i];
    uint32_t present=0;
    for(uint32_t j=0;j<s->spec.pages;j++) present+=s->table[j]!=ABSENT;
    n+=(size_t)snprintf(out+n,bytes-n,"slot %zu sid=%u %s peer=%u pages=%u agreed=%d producible=%llu highest=%llu complete=%llu present=%u flying=%u released=%llu fills=[%llu:%llu %llu:%llu]\n",
      i,s->spec.sid,s->spec.receive?"rx":s->transported?"tx":"local",s->spec.peer,s->spec.pages,s->agreed,
      (unsigned long long)atomic_load(&s->producible),(unsigned long long)atomic_load(&s->highest),(unsigned long long)atomic_load(&s->complete),present,s->flying,
      (unsigned long long)s->released,(unsigned long long)atomic_load(&s->fill_generation[0]),(unsigned long long)atomic_load(&s->fill_count[0]),
      (unsigned long long)atomic_load(&s->fill_generation[1]),(unsigned long long)atomic_load(&s->fill_count[1]));
  }
  return n;
}
