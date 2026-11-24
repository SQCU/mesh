#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <libproc.h>
#include "mesh-view.h"
#include "mesh-call.h"

/* design/SPEC-flat-page-dma-bridge.md */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_ULA_PREFIX "fd6d:6573:68:"
#define MESH_MODE 0666
#define MESH_VERSION 168u
#define MESH_LAYOUT 200u
#define MESH_ABSENT UINT32_MAX

#define MESH_STREAMS 32
#define MESH_LINK_STREAMS 4u
#define MESH_PORTS (MESH_STREAMS/MESH_LINK_STREAMS)
#define MESH_NODES 8u
#define MESH_CLIENTS 64
#define MESH_BRIDGE_BYTES (MESH_STREAMS*32u)
#define MESH_SWITCH_BYTES (UINT64_C(8)<<20)
#define MESH_PAUSE_NS UINT64_C(3000000000)
#define MESH_PLATFORM_NS UINT64_C(5000000000)
#define MESH_DMA_EXTENT (UINT64_C(1)<<30)

#define MESH_TABLE_EMPTY UINT32_MAX
#define MESH_REF_CONTIG (UINT32_C(1)<<31)
#define MESH_REF_MASK (MESH_REF_CONTIG-1u)
#define MESH_WIRE_BATCH 255u
#define MESH_DESC_FLUSH 63u

struct mesh_dma_span { uint64_t offset,bytes; };
_Static_assert(sizeof(struct mesh_dma_span)==16,"mesh_dma_span ABI");

/* design/SPEC-flat-page-dma-bridge.md section 4: uniform 4096-byte DESC wire frame */
struct mesh_wire_entry {
  uint32_t dst_node;
  uint32_t dst_index;
  uint16_t bytes;
  uint8_t flags;
  uint8_t hops;
  uint32_t src_page;
};
_Static_assert(sizeof(struct mesh_wire_entry)==16,"mesh_wire_entry");

struct mesh_wire_desc {
  _Atomic uint32_t next;
  uint32_t count;
  uint32_t origin_node;
  uint32_t from_node;
  struct mesh_wire_entry e[MESH_WIRE_BATCH];
};
_Static_assert(sizeof(struct mesh_wire_desc)==MESH_PAGE_BYTES,"mesh_wire_desc is one 4096 B page");

struct mesh_node { _Alignas(64) _Atomic uint32_t node; _Atomic uint32_t live; _Atomic uint64_t generation,bandwidth; };
_Static_assert(sizeof(struct mesh_node)==64,"mesh_node");

struct mesh_posts { _Alignas(64) _Atomic uint64_t posted[2]; _Atomic uint64_t done[2]; };
_Static_assert(sizeof(struct mesh_posts)==64 && offsetof(struct mesh_posts,done)==16,"mesh_posts ABI");
struct mesh_lanes {
  _Alignas(64) _Atomic uint32_t peer;
  _Atomic uint32_t child;
  _Atomic uint64_t bandwidth,wire;
  _Atomic uint64_t passes;
  _Atomic uint64_t send_ns,sends;
  uint64_t child_start;
  struct mesh_posts posts[2];
};
_Static_assert(sizeof(struct mesh_lanes)==192 && offsetof(struct mesh_lanes,posts)==64,"mesh_lanes");

struct mesh_record { uint32_t node,reached; uint64_t boot,sequence,region,memory;
  char model[MESH_MODEL_BYTES]; struct mesh_reach reach[MESH_STREAMS]; struct mesh_health health; };

struct mesh_client {
  _Alignas(64) _Atomic uint64_t owner;
  uint64_t start,padding[6];
};

/* Every field keeps its offset for mesh-heartbeat and mesh-install-state ABI compatibility.
   table_off -> W[0..entries-1], arena_off -> u[0..flat_pages-1], ring_off -> struct mesh_bitmaps,
   attached -> pending llist head, unused -> flat_pages, spare -> contig_pages. */
struct hdr {
  uint32_t magic,version,pgsz,block,entries,node,streams,clients;
  uint64_t table_off,arena_off,ring_off,pool_off,bridge_off,stream_off,client_off,data_off,length,boot;
  _Atomic uint64_t bridge_pid,serial,attached,map;
  uint64_t map_off;
  _Atomic uint64_t map_version;
  uint64_t unused;
  _Atomic uint64_t busy,idle_since;
  _Atomic uint64_t spare;
  uint64_t node_off,switch_off;
};
_Static_assert(offsetof(struct hdr,version)==4 && offsetof(struct hdr,pgsz)==8 && offsetof(struct hdr,block)==12 &&
               offsetof(struct hdr,entries)==16 && offsetof(struct hdr,node)==20 && offsetof(struct hdr,clients)==28 &&
               offsetof(struct hdr,table_off)==32 && offsetof(struct hdr,arena_off)==40 && offsetof(struct hdr,ring_off)==48 &&
               offsetof(struct hdr,pool_off)==56 && offsetof(struct hdr,bridge_off)==64 && offsetof(struct hdr,stream_off)==72 &&
               offsetof(struct hdr,client_off)==80 && offsetof(struct hdr,data_off)==88 && offsetof(struct hdr,length)==96 &&
               offsetof(struct hdr,bridge_pid)==112 && offsetof(struct hdr,attached)==128 && offsetof(struct hdr,map)==136 &&
               offsetof(struct hdr,map_off)==144 && offsetof(struct hdr,map_version)==152 && offsetof(struct hdr,busy)==168 &&
               offsetof(struct hdr,spare)==184 &&
               offsetof(struct hdr,node_off)==192 && offsetof(struct hdr,switch_off)==200 && sizeof(struct hdr)==208,
               "struct hdr: every field at its offset");
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");

#define MESH_RECORDS 256u
struct mesh_map_copy { uint64_t nodes; struct mesh_record record[MESH_RECORDS]; };
static inline struct mesh_map_copy *mesh_map_copies(struct hdr *m){ return (struct mesh_map_copy *)((char *)m+m->map_off); }
static inline const struct mesh_map_copy *mesh_map_published(struct hdr *m){
  return mesh_map_copies(m)+(atomic_load_explicit(&m->map_version,memory_order_acquire)&1);
}
static inline uint32_t mesh_map_nodes(const struct mesh_map_copy *copy){ return copy->nodes<MESH_RECORDS?(uint32_t)copy->nodes:MESH_RECORDS; }
static inline void mesh_map_moved(struct hdr *m){ atomic_fetch_add_explicit(&m->map,1,memory_order_release); }
static inline void mesh_expire(struct hdr *m){ __atomic_store_n(&m->version,0,__ATOMIC_SEQ_CST); }
static inline struct mesh_lanes *mesh_lanes_at(struct hdr *m,uint32_t p){ return (struct mesh_lanes *)((char *)m+m->stream_off)+p; }
static inline struct mesh_node *mesh_node_at(struct hdr *m,uint32_t n){ return (struct mesh_node *)((char *)m+m->node_off)+n; }
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz))*m->block; }
static inline uint32_t mesh_flat_pages(const struct hdr *m){ return (uint32_t)m->unused; }
static inline _Atomic uint64_t *mesh_pending_head(struct hdr *m){ return &m->attached; }
static inline _Atomic uint64_t *mesh_contig_pages(struct hdr *m){ return &m->spare; }

/* ---- Flat page array P[0..N-1], page table W[0..entries-1], refcounts u[0..N-1] ---- */
static inline unsigned char *mesh_page_at(struct hdr *m,uint32_t p){ return (unsigned char *)m+m->data_off+(uint64_t)p*MESH_PAGE_BYTES; }
static inline uint64_t mesh_page_offset(const struct hdr *m,uint32_t p){ return m->data_off+(uint64_t)p*MESH_PAGE_BYTES; }
static inline struct mesh_wire_desc *mesh_desc_at(struct hdr *m,uint32_t p){ return (struct mesh_wire_desc *)mesh_page_at(m,p); }
static inline _Atomic uint64_t *mesh_table(struct hdr *m){ return (_Atomic uint64_t *)((char *)m+m->table_off); }
static inline _Atomic uint32_t *mesh_refs(struct hdr *m){ return (_Atomic uint32_t *)((char *)m+m->arena_off); }

static inline uint64_t mesh_word_pack(uint32_t p,uint32_t bytes){
  return p==MESH_PAGE_NONE?(uint64_t)MESH_TABLE_EMPTY:(((uint64_t)bytes<<32)|(uint64_t)(p+1u));
}
static inline uint32_t mesh_word_page(uint64_t w){ const uint32_t lo=(uint32_t)w; return (!lo || lo==MESH_TABLE_EMPTY)?MESH_PAGE_NONE:lo-1u; }
static inline uint32_t mesh_word_bytes(uint64_t w){ return (uint32_t)(w>>32); }

/* ---- 3-level lock-free free bitmap (design/SPEC-flat-page-dma-bridge.md section 2, 5.1) ---- */
struct mesh_bmp_desc {
  uint32_t items,l0_words,l1_words,l2_words;
  uint64_t l0_off,l1_off,l2_off;
};
struct mesh_bitmaps {
  struct mesh_bmp_desc pages,indices;
  _Atomic uint32_t initialized;
  uint32_t pad;
};
static inline struct mesh_bitmaps *mesh_bitmaps_of(struct hdr *m){ return (struct mesh_bitmaps *)((char *)m+m->ring_off); }
static inline _Atomic uint64_t *mesh_bmp_l0(struct hdr *m,const struct mesh_bmp_desc *d){ return (_Atomic uint64_t *)((char *)m+d->l0_off); }
static inline _Atomic uint64_t *mesh_bmp_l1(struct hdr *m,const struct mesh_bmp_desc *d){ return (_Atomic uint64_t *)((char *)m+d->l1_off); }
static inline _Atomic uint64_t *mesh_bmp_l2(struct hdr *m,const struct mesh_bmp_desc *d){ return (_Atomic uint64_t *)((char *)m+d->l2_off); }

static inline uint64_t mesh_bmp_tail_mask(uint32_t count){ return count>=64?~UINT64_C(0):((UINT64_C(1)<<count)-1u); }

static inline void mesh_bmp_fill(struct hdr *m,const struct mesh_bmp_desc *d){
  _Atomic uint64_t *l0=mesh_bmp_l0(m,d),*l1=mesh_bmp_l1(m,d),*l2=mesh_bmp_l2(m,d);
  for(uint32_t w=0;w<d->l0_words;w++){
    const uint32_t rem=d->items-w*64u;
    atomic_store_explicit(&l0[w],mesh_bmp_tail_mask(rem),memory_order_relaxed);
  }
  for(uint32_t w=0;w<d->l1_words;w++){
    const uint32_t rem=d->l0_words-w*64u;
    atomic_store_explicit(&l1[w],mesh_bmp_tail_mask(rem),memory_order_relaxed);
  }
  for(uint32_t w=0;w<d->l2_words;w++){
    const uint32_t rem=d->l1_words-w*64u;
    atomic_store_explicit(&l2[w],mesh_bmp_tail_mask(rem),memory_order_relaxed);
  }
}

static inline void mesh_bmp_set(struct hdr *m,const struct mesh_bmp_desc *d,uint32_t idx){
  if(idx>=d->items)return;
  const uint32_t w0=idx>>6,w1=w0>>6,w2=w1>>6;
  const uint64_t m0=UINT64_C(1)<<(idx&63u),m1=UINT64_C(1)<<(w0&63u),m2=UINT64_C(1)<<(w1&63u);
  _Atomic uint64_t *l0=mesh_bmp_l0(m,d),*l1=mesh_bmp_l1(m,d),*l2=mesh_bmp_l2(m,d);
  const uint64_t prev0=atomic_fetch_or_explicit(&l0[w0],m0,memory_order_release);
  if(prev0==0 || !(atomic_load_explicit(&l1[w1],memory_order_relaxed)&m1)){
    atomic_thread_fence(memory_order_seq_cst);
    const uint64_t prev1=atomic_fetch_or_explicit(&l1[w1],m1,memory_order_seq_cst);
    if(prev1==0 || !(atomic_load_explicit(&l2[w2],memory_order_relaxed)&m2))
      atomic_fetch_or_explicit(&l2[w2],m2,memory_order_seq_cst);
  }
}

static inline void mesh_bmp_clear_empty(struct hdr *m,const struct mesh_bmp_desc *d,uint32_t w0){
  const uint32_t w1=w0>>6,w2=w1>>6;
  const uint64_t m1=UINT64_C(1)<<(w0&63u),m2=UINT64_C(1)<<(w1&63u);
  _Atomic uint64_t *l0=mesh_bmp_l0(m,d),*l1=mesh_bmp_l1(m,d),*l2=mesh_bmp_l2(m,d);
  const uint64_t prev1=atomic_fetch_and_explicit(&l1[w1],~m1,memory_order_seq_cst);
  if(atomic_load_explicit(&l0[w0],memory_order_seq_cst)!=0){
    if(atomic_fetch_or_explicit(&l1[w1],m1,memory_order_seq_cst)==0)
      atomic_fetch_or_explicit(&l2[w2],m2,memory_order_seq_cst);
    return;
  }
  if((prev1&~m1)!=0)return;
  atomic_fetch_and_explicit(&l2[w2],~m2,memory_order_seq_cst);
  if(atomic_load_explicit(&l1[w1],memory_order_seq_cst)!=0)
    atomic_fetch_or_explicit(&l2[w2],m2,memory_order_seq_cst);
}

static inline void mesh_bmp_clear_mask(struct hdr *m,const struct mesh_bmp_desc *d,uint32_t w0,uint64_t mask){
  _Atomic uint64_t *l0=mesh_bmp_l0(m,d);
  const uint64_t prev0=atomic_fetch_and_explicit(&l0[w0],~mask,memory_order_seq_cst);
  if((prev0&~mask)==0)mesh_bmp_clear_empty(m,d,w0);
}

static inline uint32_t mesh_rot_ctz(uint64_t v,uint32_t start){
  const uint32_t s=start&63u;
  const uint64_t r=s?((v>>s)|(v<<(64u-s))):v;
  return ((uint32_t)__builtin_ctzll(r)+s)&63u;
}

static inline uint32_t mesh_bmp_find_word(struct hdr *m,const struct mesh_bmp_desc *d,uint32_t start_w0,uint64_t *word_out){
  _Atomic uint64_t *l0=mesh_bmp_l0(m,d),*l1=mesh_bmp_l1(m,d),*l2=mesh_bmp_l2(m,d);
  const uint32_t n0=d->l0_words,n2=d->l2_words;
  if(!n0 || !n2)return MESH_ABSENT;
  const uint32_t hint=start_w0%n0;
  const uint64_t vh=atomic_load_explicit(&l0[hint],memory_order_relaxed);
  if(vh){*word_out=vh;return hint;}
  const uint32_t s2=(hint>>12)%n2,s1=(hint>>6)&63u,s0=hint&63u;
  for(uint32_t i=0;i<n2;i++){
    const uint32_t w2=(s2+i)%n2;
    uint64_t v2=atomic_load_explicit(&l2[w2],memory_order_relaxed);
    while(v2){
      const uint32_t b2=mesh_rot_ctz(v2,i==0?s1:0u);
      v2&=~(UINT64_C(1)<<b2);
      const uint32_t w1=(w2<<6)+b2;
      if(w1>=d->l1_words)continue;
      uint64_t v1=atomic_load_explicit(&l1[w1],memory_order_relaxed);
      while(v1){
        const uint32_t b1=mesh_rot_ctz(v1,(i==0 && b2==s1)?s0:0u);
        v1&=~(UINT64_C(1)<<b1);
        const uint32_t w0=(w1<<6)+b1;
        if(w0>=n0)continue;
        const uint64_t v0=atomic_load_explicit(&l0[w0],memory_order_relaxed);
        if(v0){*word_out=v0;return w0;}
      }
    }
  }
  return MESH_ABSENT;
}

/* ---- Physical page allocator and elective refcounts (kref_get_unless_zero) ---- */
static inline int mesh_page_hold(struct hdr *m,uint32_t p){
  if(p>=mesh_flat_pages(m))return 0;
  _Atomic uint32_t *u=mesh_refs(m)+p;
  uint32_t v=atomic_load_explicit(u,memory_order_relaxed);
  for(;;){
    if((v&MESH_REF_MASK)==0)return 0;
    if(atomic_compare_exchange_weak_explicit(u,&v,v+1u,memory_order_relaxed,memory_order_relaxed))return 1;
  }
}

static inline void mesh_page_release(struct hdr *m,uint32_t p){
  if(p>=mesh_flat_pages(m))return;
  _Atomic uint32_t *u=mesh_refs(m)+p;
  uint32_t v=atomic_load_explicit(u,memory_order_relaxed);
  for(;;){
    const uint32_t c=v&MESH_REF_MASK;
    if(c==0)return;
    const uint32_t nv=c==1u?0u:(v-1u);
    if(atomic_compare_exchange_weak_explicit(u,&v,nv,memory_order_acq_rel,memory_order_relaxed)){
      if(c==1u){
        if(v&MESH_REF_CONTIG)atomic_fetch_sub_explicit(mesh_contig_pages(m),1,memory_order_relaxed);
        mesh_bmp_set(m,&mesh_bitmaps_of(m)->pages,p);
      }
      return;
    }
  }
}

static inline uint32_t mesh_page_alloc_batch(struct hdr *m,uint32_t *out,uint32_t max_n,uint32_t seed){
  const struct mesh_bmp_desc *d=&mesh_bitmaps_of(m)->pages;
  _Atomic uint64_t *l0=mesh_bmp_l0(m,d);
  _Atomic uint32_t *refs=mesh_refs(m);
  uint32_t got=0,cur=seed;
  for(uint32_t tries=0;got<max_n && tries<64;tries++){
    uint64_t v0=0;
    const uint32_t w0=mesh_bmp_find_word(m,d,cur,&v0);
    if(w0==MESH_ABSENT)break;
    uint64_t want=0;
    uint32_t need=max_n-got;
    for(uint64_t tmp=v0;tmp && need;need--){
      const uint64_t lsb=tmp&-tmp;
      want|=lsb;
      tmp^=lsb;
    }
    const uint64_t prev=atomic_fetch_and_explicit(&l0[w0],~want,memory_order_acq_rel);
    if((prev&~want)==0)mesh_bmp_clear_empty(m,d,w0);
    uint64_t claimed=prev&want;
    while(claimed){
      const uint32_t b=(uint32_t)__builtin_ctzll(claimed);
      claimed&=claimed-1u;
      const uint32_t p=(w0<<6)+b;
      if(p>=d->items)continue;
      uint32_t zero=0;
      if(atomic_compare_exchange_strong_explicit(&refs[p],&zero,1u,memory_order_acq_rel,memory_order_relaxed))
        out[got++]=p;
    }
    cur=w0+1u;
  }
  return got;
}

static inline uint32_t mesh_page_alloc(struct hdr *m,uint32_t seed){
  uint32_t p=MESH_PAGE_NONE;
  return mesh_page_alloc_batch(m,&p,1,seed)?p:MESH_PAGE_NONE;
}

static inline uint32_t mesh_page_alloc_contiguous(struct hdr *m,uint32_t count){
  const uint32_t pages=mesh_flat_pages(m);
  if(!count || count>pages){errno=EINVAL;return MESH_PAGE_NONE;}
  const uint64_t limit=(uint64_t)pages*9u/20u;
  uint64_t cur=atomic_load_explicit(mesh_contig_pages(m),memory_order_relaxed);
  for(;;){
    if(cur+count>limit){
      fputs("ERR:NPHARD: contiguous allocations exceed 45% of memory space\n",stderr);
      abort();
    }
    if(atomic_compare_exchange_weak_explicit(mesh_contig_pages(m),&cur,cur+count,memory_order_acq_rel,memory_order_relaxed))break;
  }
  const struct mesh_bmp_desc *d=&mesh_bitmaps_of(m)->pages;
  _Atomic uint64_t *l0=mesh_bmp_l0(m,d);
  _Atomic uint32_t *refs=mesh_refs(m);
  uint32_t run_start=0,run_len=0;
  for(uint32_t w=0;w<d->l0_words;w++){
    const uint64_t v=atomic_load_explicit(&l0[w],memory_order_relaxed);
    if(v==~UINT64_C(0)){
      if(!run_len)run_start=w<<6;
      run_len+=64u;
      if(run_start+run_len>pages)run_len=pages-run_start;
    }else if(v==0){
      run_len=0;continue;
    }else{
      for(uint32_t b=0;b<64u && (w<<6)+b<pages;b++){
        if((v>>b)&1u){
          if(!run_len)run_start=(w<<6)+b;
          run_len++;
          if(run_len>=count)goto try_claim;
        }else run_len=0;
      }
      continue;
    }
    if(run_len>=count){
try_claim:;
      const uint32_t base=run_start;
      uint32_t k=0;
      for(;k<count;k++){
        uint32_t zero=0;
        if(!atomic_compare_exchange_strong_explicit(&refs[base+k],&zero,MESH_REF_CONTIG|1u,memory_order_acq_rel,memory_order_relaxed))break;
      }
      if(k==count){
        for(uint32_t p=base;p<base+count;){
          const uint32_t w0=p>>6,off=p&63u,take=count-(p-base)<64u-off?count-(p-base):64u-off;
          mesh_bmp_clear_mask(m,d,w0,mesh_bmp_tail_mask(take)<<off);
          p+=take;
        }
        return base;
      }
      for(uint32_t r=0;r<k;r++){
        atomic_store_explicit(&refs[base+r],0u,memory_order_release);
        mesh_bmp_set(m,d,base+r);
      }
      run_start=base+k+1u;
      run_len=0;
      w=run_start>>6;
      if(w<d->l0_words && (run_start&63u)){
        const uint64_t vw=atomic_load_explicit(&l0[w],memory_order_relaxed);
        for(uint32_t b=run_start&63u;b<64u && (w<<6)+b<pages;b++){
          if((vw>>b)&1u){if(!run_len)run_start=(w<<6)+b;run_len++;if(run_len>=count)goto try_claim;}
          else run_len=0;
        }
      }
    }
  }
  atomic_fetch_sub_explicit(mesh_contig_pages(m),count,memory_order_relaxed);
  errno=ENOMEM;
  return MESH_PAGE_NONE;
}

/* ---- Page-table index allocator and atomic exchange ---- */
static inline uint64_t mesh_table_load(struct hdr *m,uint32_t index){
  return atomic_load_explicit(&mesh_table(m)[index],memory_order_acquire);
}

static inline void mesh_table_land(struct hdr *m,uint32_t index,uint32_t page,uint32_t bytes){
  const uint64_t old=atomic_exchange_explicit(&mesh_table(m)[index&(MESH_TABLE-1u)],mesh_word_pack(page,bytes),memory_order_acq_rel);
  const uint32_t old_p=mesh_word_page(old);
  if(old_p!=MESH_PAGE_NONE)mesh_page_release(m,old_p);
}

static inline uint32_t mesh_idx_alloc_one(struct hdr *m,int back_page,uint32_t seed,uint32_t *out_page){
  const struct mesh_bmp_desc *d=&mesh_bitmaps_of(m)->indices;
  _Atomic uint64_t *tab=mesh_table(m);
  uint32_t p=MESH_PAGE_NONE;
  if(back_page){
    p=mesh_page_alloc(m,seed);
    if(p==MESH_PAGE_NONE)return MESH_ABSENT;
  }
  const uint64_t val=back_page?mesh_word_pack(p,0):MESH_TABLE_EMPTY;
  for(uint32_t tries=0;tries<32;tries++){
    uint64_t v0=0;
    const uint32_t w0=mesh_bmp_find_word(m,d,seed+tries,&v0);
    if(w0==MESH_ABSENT)break;
    uint64_t clear=0;
    while(v0){
      const uint32_t b=(uint32_t)__builtin_ctzll(v0);
      v0&=v0-1u;
      const uint32_t idx=(w0<<6)+b;
      clear|=UINT64_C(1)<<b;
      if(idx>=d->items)continue;
      uint64_t zero=0;
      if(atomic_compare_exchange_strong_explicit(&tab[idx],&zero,val,memory_order_release,memory_order_relaxed)){
        mesh_bmp_clear_mask(m,d,w0,clear);
        if(out_page)*out_page=p;
        return idx;
      }
    }
    if(clear)mesh_bmp_clear_mask(m,d,w0,clear);
  }
  if(p!=MESH_PAGE_NONE)mesh_page_release(m,p);
  return MESH_ABSENT;
}

static inline void mesh_idx_free_one(struct hdr *m,uint32_t index){
  if(index>=m->entries)return;
  const uint64_t old=atomic_exchange_explicit(&mesh_table(m)[index],0,memory_order_acq_rel);
  const uint32_t p=mesh_word_page(old);
  if(p!=MESH_PAGE_NONE)mesh_page_release(m,p);
  if(old!=0)mesh_bmp_set(m,&mesh_bitmaps_of(m)->indices,index);
}

/* ---- Linux llist.h lock-free singly linked list over DESC pages (next in P[q].next) ---- */
static inline void mesh_llist_add(struct hdr *m,_Atomic uint64_t *head,uint32_t q){
  _Atomic uint32_t *next=&mesh_desc_at(m,q)->next;
  uint64_t first=atomic_load_explicit(head,memory_order_relaxed);
  do {
    atomic_store_explicit(next,(uint32_t)first,memory_order_relaxed);
  } while(!atomic_compare_exchange_weak_explicit(head,&first,(uint64_t)(q+1u),memory_order_seq_cst,memory_order_relaxed));
}

static inline uint32_t mesh_llist_del_all(struct hdr *m,_Atomic uint64_t *head){
  (void)m;
  return (uint32_t)atomic_exchange_explicit(head,0,memory_order_seq_cst);
}

static inline uint32_t mesh_llist_reverse(struct hdr *m,uint32_t head){
  const uint32_t pages=mesh_flat_pages(m);
  uint32_t rev=0;
  while(head){
    const uint32_t q=head-1u;
    if(q>=pages)break;
    const uint32_t nxt=atomic_load_explicit(&mesh_desc_at(m,q)->next,memory_order_relaxed);
    atomic_store_explicit(&mesh_desc_at(m,q)->next,rev,memory_order_relaxed);
    rev=head;
    head=nxt;
  }
  return rev;
}

/* ---- Processes of the region ---- */
static inline int mesh_exited(pid_t process){ return !process || (kill(process,0) && errno==ESRCH); }
static inline uint64_t mesh_started(pid_t process){
  struct proc_bsdinfo info;
  return process && proc_pidinfo(process,PROC_PIDTBSDINFO,0,&info,(int)sizeof info)==(int)sizeof info?
         (uint64_t)info.pbi_start_tvsec*UINT64_C(1000000)+info.pbi_start_tvusec:0;
}
static inline int mesh_alive(pid_t pid,uint64_t start){ return start?mesh_started(pid)==start:!mesh_exited(pid); }

int mesh_joined(struct mesh_record *,struct mesh_record *,double *beta);

static inline uint64_t mesh_bmp_size(uint32_t items,struct mesh_bmp_desc *d,uint64_t base){
  const uint32_t w0=(items+63u)/64u,w1=(w0+63u)/64u,w2=(w1+63u)/64u;
  uint64_t at=base;
  d->items=items;d->l0_words=w0;d->l1_words=w1;d->l2_words=w2;
  d->l0_off=at;at+=(uint64_t)w0*sizeof(uint64_t);at=(at+63u)&~UINT64_C(63);
  d->l1_off=at;at+=(uint64_t)w1*sizeof(uint64_t);at=(at+63u)&~UINT64_C(63);
  d->l2_off=at;at+=(uint64_t)w2*sizeof(uint64_t);at=(at+63u)&~UINT64_C(63);
  return at;
}

static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t pages){
  const uint32_t arena=pages/block*block;
  const uint64_t page=pgsz,bytes=(uint64_t)block*pgsz;
  const uint32_t flat=(uint32_t)(((uint64_t)arena*pgsz)/MESH_PAGE_BYTES);
  uint64_t at=(sizeof *h+page-1)/page*page;
  h->pgsz=pgsz;h->block=block;h->entries=MESH_TABLE;h->streams=MESH_STREAMS;h->clients=MESH_CLIENTS;
  h->unused=flat;
  h->table_off=at;at+=(uint64_t)MESH_TABLE*sizeof(uint64_t);at=(at+page-1)/page*page;
  h->arena_off=at;at+=(uint64_t)flat*sizeof(uint32_t);at=(at+page-1)/page*page;
  h->ring_off=at;
  at+=sizeof(struct mesh_bitmaps);at=(at+63u)&~UINT64_C(63);
  struct mesh_bmp_desc tmp;
  at=mesh_bmp_size(flat,&tmp,at);
  at=mesh_bmp_size(MESH_TABLE,&tmp,at);
  at=(at+page-1)/page*page;
  h->pool_off=0;
  h->bridge_off=at;at+=MESH_BRIDGE_BYTES;at=(at+page-1)/page*page;
  h->stream_off=at;at+=(uint64_t)MESH_STREAMS*sizeof(struct mesh_lanes);at=(at+page-1)/page*page;
  h->node_off=at;at+=(uint64_t)MESH_NODES*sizeof(struct mesh_node);at=(at+page-1)/page*page;
  h->map_off=at;at+=2*sizeof(struct mesh_map_copy);at=(at+page-1)/page*page;
  h->switch_off=at;at+=MESH_SWITCH_BYTES;at=(at+page-1)/page*page;
  h->client_off=at;at+=(uint64_t)MESH_CLIENTS*sizeof(struct mesh_client);
  at=(at+bytes-1)/bytes*bytes;
  h->data_off=at;at+=(uint64_t)arena*pgsz;
  h->length=at;return at;
}

static inline void mesh_bitmaps_init(struct hdr *m){
  struct mesh_bitmaps *b=mesh_bitmaps_of(m);
  if(atomic_load_explicit(&b->initialized,memory_order_acquire))return;
  uint64_t at=m->ring_off+sizeof(struct mesh_bitmaps);at=(at+63u)&~UINT64_C(63);
  at=mesh_bmp_size(mesh_flat_pages(m),&b->pages,at);
  mesh_bmp_size(m->entries,&b->indices,at);
  mesh_bmp_fill(m,&b->pages);
  mesh_bmp_fill(m,&b->indices);
  atomic_store_explicit(&b->initialized,1,memory_order_release);
}
#endif
