#ifndef MESH_REDUCE_H
#define MESH_REDUCE_H
#include "mesh.h"
#include <math.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* Switch-side algebraic page morphisms (Pages^k -> Pages^m) and lock-free reduction landing.
   Algebraic classes (top 2 bits of op):
     C0 (MESH_ALG_C0_COMM):     commutative monoid, tag = leaf weight (any order, any Steiner merge tree).
     C1 (MESH_ALG_C1_LIFTED):   lifted commutative monoid (e.g. online softmax (m,l,o)), tag = leaf weight.
     C2 (MESH_ALG_C2_INTERVAL): associative non-commutative monoid, tag = lo | (hi << 8) contiguous interval. */

static inline int mesh_red_can_merge(uint8_t op,uint16_t tag_a,uint16_t tag_b,uint16_t *tag_out,int *b_is_left){
  const uint8_t alg=MESH_RED_ALG(op);
  if(alg==MESH_ALG_C0_COMM || alg==MESH_ALG_C1_LIFTED){
    const uint32_t sum=(uint32_t)tag_a+(uint32_t)tag_b;
    if(!tag_a || !tag_b || sum>0xffffu)return 0;
    *tag_out=(uint16_t)sum;
    *b_is_left=0;
    return 1;
  }
  if(alg==MESH_ALG_C2_INTERVAL){
    const uint32_t lo_a=tag_a&0xffu,hi_a=tag_a>>8;
    const uint32_t lo_b=tag_b&0xffu,hi_b=tag_b>>8;
    if(hi_a<=lo_a || hi_b<=lo_b)return 0;
    if(hi_a==lo_b){
      *tag_out=MESH_TAG_INTERVAL(lo_a,hi_b);
      *b_is_left=0; /* a is left [lo_a,hi_a), b is right [lo_b,hi_b) -> dst = b o a */
      return 1;
    }
    if(hi_b==lo_a){
      *tag_out=MESH_TAG_INTERVAL(lo_b,hi_a);
      *b_is_left=1; /* b is left [lo_b,hi_b), a is right [lo_a,hi_a) -> dst = a o b */
      return 1;
    }
  }
  return 0;
}

static inline void mesh_red_apply_page(uint8_t op,void *dst_v,const void *src_v,uint32_t bytes,int src_is_left){
  if(!bytes)return;
  if(bytes>MESH_PAGE_BYTES)bytes=MESH_PAGE_BYTES;
  switch(op){
  case MESH_RED_SUM_F32:{
    float *d=(float *)dst_v;
    const float *s=(const float *)src_v;
    uint32_t n=bytes>>2,i=0;
#if defined(__ARM_NEON)
    for(;i+16u<=n;i+=16u){
      vst1q_f32(d+i,    vaddq_f32(vld1q_f32(d+i),    vld1q_f32(s+i)));
      vst1q_f32(d+i+4,  vaddq_f32(vld1q_f32(d+i+4),  vld1q_f32(s+i+4)));
      vst1q_f32(d+i+8,  vaddq_f32(vld1q_f32(d+i+8),  vld1q_f32(s+i+8)));
      vst1q_f32(d+i+12, vaddq_f32(vld1q_f32(d+i+12), vld1q_f32(s+i+12)));
    }
    for(;i+4u<=n;i+=4u)vst1q_f32(d+i,vaddq_f32(vld1q_f32(d+i),vld1q_f32(s+i)));
#endif
    for(;i<n;i++)d[i]+=s[i];
    return;
  }
  case MESH_RED_MAX_F32:{
    float *d=(float *)dst_v;
    const float *s=(const float *)src_v;
    uint32_t n=bytes>>2,i=0;
#if defined(__ARM_NEON)
    for(;i+4u<=n;i+=4u)vst1q_f32(d+i,vmaxq_f32(vld1q_f32(d+i),vld1q_f32(s+i)));
#endif
    for(;i<n;i++)if(s[i]>d[i])d[i]=s[i];
    return;
  }
  case MESH_RED_MIN_F32:{
    float *d=(float *)dst_v;
    const float *s=(const float *)src_v;
    uint32_t n=bytes>>2,i=0;
#if defined(__ARM_NEON)
    for(;i+4u<=n;i+=4u)vst1q_f32(d+i,vminq_f32(vld1q_f32(d+i),vld1q_f32(s+i)));
#endif
    for(;i<n;i++)if(s[i]<d[i])d[i]=s[i];
    return;
  }
  case MESH_RED_SUM_I32:{
    int32_t *d=(int32_t *)dst_v;
    const int32_t *s=(const int32_t *)src_v;
    uint32_t n=bytes>>2,i=0;
#if defined(__ARM_NEON)
    for(;i+4u<=n;i+=4u)vst1q_s32(d+i,vaddq_s32(vld1q_s32(d+i),vld1q_s32(s+i)));
#endif
    for(;i<n;i++)d[i]+=s[i];
    return;
  }
  case MESH_RED_SUM_I64:{
    int64_t *d=(int64_t *)dst_v;
    const int64_t *s=(const int64_t *)src_v;
    uint32_t n=bytes>>3,i=0;
#if defined(__ARM_NEON)
    for(;i+2u<=n;i+=2u)vst1q_s64(d+i,vaddq_s64(vld1q_s64(d+i),vld1q_s64(s+i)));
#endif
    for(;i<n;i++)d[i]+=s[i];
    return;
  }
  case MESH_RED_ONLINE_SOFTMAX_F32:{
    /* C1 lifted commutative monoid: 8 rows of 128 floats [m, l, o_0..o_125] per 4 KiB page. */
    float *d=(float *)dst_v;
    const float *s=(const float *)src_v;
    const uint32_t nf=bytes>>2;
    for(uint32_t base=0;base+128u<=nf;base+=128u){
      float *a=d+base;
      const float *b=s+base;
      const float m=a[0]>b[0]?a[0]:b[0];
      const float ea=expf(a[0]-m),eb=expf(b[0]-m);
      a[0]=m;
      a[1]=a[1]*ea+b[1]*eb;
      uint32_t k=2u;
#if defined(__ARM_NEON)
      const float32x4_t vea=vdupq_n_f32(ea),veb=vdupq_n_f32(eb);
      a[2]=a[2]*ea+b[2]*eb;
      a[3]=a[3]*ea+b[3]*eb;
      k=4u;
      for(;k+4u<=128u;k+=4u){
        const float32x4_t va=vld1q_f32(a+k),vb=vld1q_f32(b+k);
        vst1q_f32(a+k,vfmaq_f32(vmulq_f32(va,vea),vb,veb));
      }
#endif
      for(;k<128u;k++)a[k]=a[k]*ea+b[k]*eb;
    }
    return;
  }
  case MESH_RED_AFFINE_SCAN_F32:{
    /* C2 non-commutative associative monoid: pairs (a, b) composing f_R o f_L = (a_R*a_L, a_R*b_L + b_R).
       If src_is_left == 0: dst is L, src is R -> dst = src o dst.
       If src_is_left == 1: src is L, dst is R -> dst = dst o src. */
    float *d=(float *)dst_v;
    const float *s=(const float *)src_v;
    const uint32_t nf=bytes>>2;
    if(!src_is_left){
      for(uint32_t i=0;i+1u<nf;i+=2u){
        const float al=d[i],bl=d[i+1u],ar=s[i],br=s[i+1u];
        d[i]=ar*al;
        d[i+1u]=ar*bl+br;
      }
    }else{
      for(uint32_t i=0;i+1u<nf;i+=2u){
        const float ar=d[i],br=d[i+1u],al=s[i],bl=s[i+1u];
        d[i]=ar*al;
        d[i+1u]=ar*bl+br;
      }
    }
    return;
  }
  default: return;
  }
}

static inline int mesh_page_is_exclusive(struct hdr *m,uint32_t p){
  return (atomic_load_explicit(&mesh_refs(m)[p],memory_order_relaxed)&MESH_REF_MASK)==1u;
}
static inline int mesh_page_run_exclusive(struct hdr *m,uint32_t p0,uint32_t count){
  for(uint32_t k=0;k<count;k++)if(!mesh_page_is_exclusive(m,p0+k))return 0;
  return 1;
}

static inline void mesh_red_apply_run(struct hdr *m,uint8_t op,uint32_t dst_p0,uint32_t src_p0,
                                      const struct mesh_wire_entry *ge,int src_is_left){
  for(uint32_t k=0;k<8u;k++){
    const uint32_t b=ge[k].bytes<=MESH_PAGE_BYTES?ge[k].bytes:MESH_PAGE_BYTES;
    mesh_red_apply_page(op,mesh_page_at(m,dst_p0+k),mesh_page_at(m,src_p0+k),b,src_is_left);
  }
}

/* Lock-free reduction landing into W[index] for C0/C1, with per-bucket interval stash for out-of-order C2. */
#define MESH_C2_BUCKETS 64u
#define MESH_C2_SLOTS   32u
struct mesh_c2_slot {
  uint32_t idx,page;
  uint16_t bytes,tag;
  uint8_t  op,used,pad[2];
};
struct mesh_c2_bucket {
  _Alignas(64) _Atomic uint32_t lock;
  uint32_t pad;
  struct mesh_c2_slot slot[MESH_C2_SLOTS];
};

static inline uint32_t mesh_red_fold_pair(struct hdr *m,uint8_t op,uint32_t p_a,uint32_t p_b,
                                          uint32_t bytes,int b_is_left,uint32_t seed){
  if(mesh_page_is_exclusive(m,p_a)){
    mesh_red_apply_page(op,mesh_page_at(m,p_a),mesh_page_at(m,p_b),bytes,b_is_left);
    mesh_page_release(m,p_b);
    return p_a;
  }
  if(mesh_page_is_exclusive(m,p_b)){
    /* Swap target: fold a into b, inverting src_is_left! */
    mesh_red_apply_page(op,mesh_page_at(m,p_b),mesh_page_at(m,p_a),bytes,!b_is_left);
    mesh_page_release(m,p_a);
    return p_b;
  }
  const uint32_t acc=mesh_page_alloc(m,seed);
  if(acc==MESH_PAGE_NONE){
    /* Fallback: fold into p_a */
    mesh_red_apply_page(op,mesh_page_at(m,p_a),mesh_page_at(m,p_b),bytes,b_is_left);
    mesh_page_release(m,p_b);
    return p_a;
  }
  memcpy(mesh_page_at(m,acc),mesh_page_at(m,p_a),bytes);
  mesh_red_apply_page(op,mesh_page_at(m,acc),mesh_page_at(m,p_b),bytes,b_is_left);
  mesh_page_release(m,p_a);
  mesh_page_release(m,p_b);
  return acc;
}

static inline void mesh_table_reduce_land_c2(struct hdr *m,struct mesh_c2_bucket *buckets,uint32_t index,
                                             uint32_t page,uint32_t bytes,uint8_t op,uint16_t tag){
  const uint32_t slot_idx=index&(MESH_TABLE-1u);
  struct mesh_c2_bucket *bk=buckets+(slot_idx&(MESH_C2_BUCKETS-1u));
  while(atomic_exchange_explicit(&bk->lock,1u,memory_order_acquire)){}
  _Atomic uint64_t *wslot=&mesh_table(m)[slot_idx];
  const uint64_t old=atomic_exchange_explicit(wslot,(uint64_t)MESH_TABLE_EMPTY,memory_order_acq_rel);
  uint32_t cur_p=page,cur_b=bytes;
  uint16_t cur_tag=tag;
  const uint32_t old_p=mesh_word_page(old);
  if(old_p!=MESH_PAGE_NONE){
    const uint16_t old_tag=mesh_word_tag(old);
    const uint32_t old_b=mesh_word_bytes(old);
    uint16_t merged_tag=0;int b_is_left=0;
    if(mesh_red_can_merge(op,old_tag,cur_tag,&merged_tag,&b_is_left)){
      cur_p=mesh_red_fold_pair(m,op,old_p,cur_p,cur_b>old_b?cur_b:old_b,b_is_left,slot_idx);
      cur_tag=merged_tag;
      if(old_b>cur_b)cur_b=old_b;
    }else{
      int parked=0;
      for(uint32_t s=0;s<MESH_C2_SLOTS;s++){
        if(!bk->slot[s].used){
          bk->slot[s]=(struct mesh_c2_slot){.idx=slot_idx,.page=old_p,.bytes=(uint16_t)old_b,.tag=old_tag,.op=op,.used=1u};
          parked=1;break;
        }
      }
      if(!parked)mesh_page_release(m,old_p);
    }
  }
  /* Cascade-merge any stashed adjacent intervals for this slot_idx. */
  for(int progress=1;progress;){
    progress=0;
    for(uint32_t s=0;s<MESH_C2_SLOTS;s++){
      if(!bk->slot[s].used || bk->slot[s].idx!=slot_idx || bk->slot[s].op!=op)continue;
      uint16_t merged_tag=0;int b_is_left=0;
      if(mesh_red_can_merge(op,cur_tag,bk->slot[s].tag,&merged_tag,&b_is_left)){
        const uint32_t sb=bk->slot[s].bytes;
        cur_p=mesh_red_fold_pair(m,op,cur_p,bk->slot[s].page,cur_b>sb?cur_b:sb,b_is_left,slot_idx);
        cur_tag=merged_tag;
        if(sb>cur_b)cur_b=sb;
        bk->slot[s].used=0;
        progress=1;
      }
    }
  }
  atomic_store_explicit(wslot,mesh_word_pack_red(cur_p,cur_b,cur_tag),memory_order_release);
  atomic_store_explicit(&bk->lock,0u,memory_order_release);
}

static inline void mesh_table_reduce_land(struct hdr *m,struct mesh_c2_bucket *c2_buckets,uint32_t index,
                                          uint32_t page,uint32_t bytes,uint8_t op,uint16_t tag){
  if(op==MESH_RED_NONE || !tag){
    mesh_table_land(m,index,page,bytes);
    return;
  }
  if(MESH_RED_ALG(op)==MESH_ALG_C2_INTERVAL){
    mesh_table_reduce_land_c2(m,c2_buckets,index,page,bytes,op,tag);
    return;
  }
  /* C0 / C1: 100% lock-free parallel reduction accumulator directly inside 64-bit W[index]. */
  const uint32_t slot_idx=index&(MESH_TABLE-1u);
  _Atomic uint64_t *wslot=&mesh_table(m)[slot_idx];
  uint32_t cur_p=page,cur_b=bytes;
  uint16_t cur_tag=tag;
  for(;;){
    uint64_t expected=(uint64_t)MESH_TABLE_EMPTY;
    const uint64_t val=mesh_word_pack_red(cur_p,cur_b,cur_tag);
    if(atomic_compare_exchange_strong_explicit(wslot,&expected,val,memory_order_acq_rel,memory_order_acquire))return;
    const uint64_t old=atomic_exchange_explicit(wslot,(uint64_t)MESH_TABLE_EMPTY,memory_order_acq_rel);
    const uint32_t old_p=mesh_word_page(old);
    if(old_p==MESH_PAGE_NONE)continue;
    const uint16_t old_tag=mesh_word_tag(old);
    const uint32_t old_b=mesh_word_bytes(old);
    uint16_t merged_tag=0;int b_is_left=0;
    if(mesh_red_can_merge(op,old_tag,cur_tag,&merged_tag,&b_is_left)){
      const uint32_t nb=cur_b>old_b?cur_b:old_b;
      cur_p=mesh_red_fold_pair(m,op,old_p,cur_p,nb,b_is_left,slot_idx);
      cur_b=nb;
      cur_tag=merged_tag;
    }else{
      mesh_page_release(m,old_p);
    }
  }
}
#endif
