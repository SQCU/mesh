#include "mesh-switched.h"
#include <pthread.h>

static inline uint32_t thread_seed(void){
  static _Thread_local uint32_t base,cursor;
  if(!base){
    uint64_t tid=0;
    pthread_threadid_np(NULL,&tid);
    const uint32_t h=(uint32_t)((tid^(tid>>32)^(uint64_t)getpid())*2654435761u);
    base=1u+((h>>16)&1023u)*32u;
    cursor=base;
  }
  const uint32_t w=cursor;
  cursor=base+((cursor-base+1u)&31u);
  return w;
}

uint32_t mesh_nodes_reached(struct hdr *m,struct mesh_reach *reach,uint32_t room){
  uint32_t count=0;
  for(uint32_t n=0;m && n<MESH_NODES && count<room;n++){
    const struct mesh_node *x=mesh_node_at(m,n);
    const uint32_t node=atomic_load_explicit(&x->node,memory_order_relaxed);
    if(!node || !atomic_load_explicit(&x->live,memory_order_acquire))continue;
    reach[count++]=(struct mesh_reach){.node=node,.bandwidth=atomic_load_explicit(&x->bandwidth,memory_order_relaxed)};
  }
  return count;
}

int mesh_alloc(struct mesh_ctx *ctx,uint32_t count,int allocate_pages,
               uint32_t *out_indices,uint32_t *out_pages){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m || !count || (!out_indices && !out_pages))return EINVAL;
  if(!out_indices){
    if(!allocate_pages)return EINVAL;
    uint32_t got=0;
    while(got<count){
      const uint32_t n=mesh_page_alloc_batch(m,out_pages+got,count-got,thread_seed());
      if(!n){
        for(uint32_t r=0;r<got;r++)mesh_page_release(m,out_pages[r]);
        return ENOMEM;
      }
      got+=n;
    }
    return 0;
  }
  if(!allocate_pages){
    uint32_t got=0;
    while(got<count){
      const uint32_t n=mesh_idx_alloc_batch(m,out_indices+got,count-got,thread_seed());
      if(!n){
        for(uint32_t r=0;r<got;r++)mesh_idx_free_one(m,out_indices[r]);
        return ENOMEM;
      }
      got+=n;
    }
    if(out_pages)for(uint32_t i=0;i<count;i++)out_pages[i]=MESH_PAGE_NONE;
    return 0;
  }
  for(uint32_t i=0;i<count;i++){
    uint32_t p=MESH_PAGE_NONE;
    const uint32_t idx=mesh_idx_alloc_one(m,allocate_pages,thread_seed(),&p);
    if(idx==MESH_ABSENT){
      for(uint32_t r=0;r<i;r++)mesh_idx_free_one(m,out_indices[r]);
      return ENOMEM;
    }
    out_indices[i]=idx;
    if(out_pages)out_pages[i]=p;
  }
  return 0;
}

int mesh_alloc_contiguous(struct mesh_ctx *ctx,uint32_t count,
                          uint32_t *out_indices,void **out_base){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m || !count)return EINVAL;
  const uint32_t base=mesh_page_alloc_contiguous(m,count);
  if(base==MESH_PAGE_NONE)return errno?errno:ENOMEM;
  if(out_indices){
    const struct mesh_bmp_desc *d=&mesh_bitmaps_of(m)->indices;
    _Atomic uint64_t *tab=mesh_table(m);
    for(uint32_t i=0;i<count;i++){
      uint32_t got=MESH_ABSENT;
      const uint64_t val=mesh_word_pack(base+i,0);
      for(uint32_t tries=0;tries<32 && got==MESH_ABSENT;tries++){
        uint64_t v0=0;
        const uint32_t w0=mesh_bmp_find_word(m,d,thread_seed()+tries,&v0);
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
            got=idx;break;
          }
        }
        if(clear)mesh_bmp_clear_mask(m,d,w0,clear);
      }
      if(got==MESH_ABSENT){
        for(uint32_t r=0;r<i;r++)mesh_idx_free_one(m,out_indices[r]);
        for(uint32_t r=i;r<count;r++)mesh_page_release(m,base+r);
        return ENOMEM;
      }
      out_indices[i]=got;
    }
  }
  if(out_base)*out_base=mesh_page_at(m,base);
  return 0;
}

int mesh_hold(struct mesh_ctx *ctx,uint32_t page){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m)return EINVAL;
  return mesh_page_hold(m,page)?0:ENOENT;
}

uint32_t mesh_hold_index(struct mesh_ctx *ctx,uint32_t index){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m || index>=m->entries)return MESH_PAGE_NONE;
  for(;;){
    const uint64_t w=mesh_table_load(m,index);
    const uint32_t p=mesh_word_page(w);
    if(p==MESH_PAGE_NONE)return MESH_PAGE_NONE;
    if(!mesh_page_hold(m,p))continue;
    if(mesh_word_page(mesh_table_load(m,index))==p)return p;
    mesh_page_release(m,p);
  }
}

void mesh_release(struct mesh_ctx *ctx,uint32_t page){
  struct hdr *m=ctx?ctx->M:NULL;
  if(m)mesh_page_release(m,page);
}

void mesh_clear_indices(struct mesh_ctx *ctx,const uint32_t *indices,uint32_t count){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m || !indices)return;
  uint32_t run_p=MESH_PAGE_NONE,run_n=0;
  for(uint32_t i=0;i<count;i++){
    const uint32_t idx=indices[i];
    if(idx>=m->entries)continue;
    const uint64_t prev=atomic_exchange_explicit(&mesh_table(m)[idx],(uint64_t)MESH_TABLE_EMPTY,memory_order_acq_rel);
    const uint32_t old_p=mesh_word_page(prev);
    if(old_p==MESH_PAGE_NONE)continue;
    if(run_n && old_p==run_p+run_n){run_n++;}
    else{
      if(run_n)mesh_page_release_run(m,run_p,run_n);
      run_p=old_p;run_n=1u;
    }
  }
  if(run_n)mesh_page_release_run(m,run_p,run_n);
}

void mesh_free_indices(struct mesh_ctx *ctx,const uint32_t *indices,uint32_t count){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m || !indices)return;
  for(uint32_t i=0;i<count;i++)mesh_idx_free_one(m,indices[i]);
}

int mesh_defragment(struct mesh_ctx *ctx,const uint32_t *indices,uint32_t count,void **out_base){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m || !indices || !count)return EINVAL;
  for(uint32_t i=0;i<count;i++)if(indices[i]>=m->entries)return EINVAL;
  const uint32_t base=mesh_page_alloc_contiguous(m,count);
  if(base==MESH_PAGE_NONE)return errno?errno:ENOMEM;
  for(uint32_t r=0;r<count;r++){
    const uint32_t idx=indices[r];
    const uint32_t p=mesh_hold_index(ctx,idx);
    uint32_t b=0;
    if(p!=MESH_PAGE_NONE){
      b=mesh_word_bytes(mesh_table_load(m,idx));
      if(b>MESH_PAGE_BYTES)b=MESH_PAGE_BYTES;
      if(b)memcpy(mesh_page_at(m,base+r),mesh_page_at(m,p),b);
    }
    mesh_table_land(m,idx,base+r,b);
    if(p!=MESH_PAGE_NONE)mesh_page_release(m,p);
  }
  if(out_base)*out_base=mesh_page_at(m,base);
  return 0;
}

int mesh_write(struct mesh_ctx *ctx,const struct mesh_xfer *xfers,uint32_t count){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m || !xfers || !count)return EINVAL;
  const uint32_t self=m->node,pages=mesh_flat_pages(m);
  struct mesh_wire_desc *desc=NULL;
  uint32_t desc_page=MESH_PAGE_NONE,n=0;
  for(uint32_t i=0;i<count;i++){
    const struct mesh_xfer *x=xfers+i;
    if(x->src_page>=pages || x->bytes>MESH_PAGE_BYTES || x->dst_index>=m->entries){
      if(desc_page!=MESH_PAGE_NONE){
        for(uint32_t k=0;k<n;k++)mesh_page_release(m,desc->e[k].src_page);
        mesh_page_release(m,desc_page);
      }
      return EINVAL;
    }
    if(x->dst_node==self){
      if(!mesh_page_hold(m,x->src_page))continue;
      mesh_table_land(m,x->dst_index,x->src_page,x->bytes);
      continue;
    }
    if(desc_page==MESH_PAGE_NONE){
      desc_page=mesh_page_alloc(m,thread_seed());
      if(desc_page==MESH_PAGE_NONE)return ENOMEM;
      desc=mesh_desc_at(m,desc_page);
      desc->origin_node=self;desc->from_node=self;
      n=0;
    }
    if(!mesh_page_hold(m,x->src_page))continue;
    desc->e[n++]=(struct mesh_wire_entry){
      .dst_node=x->dst_node,.dst_index=x->dst_index,
      .bytes=(uint16_t)x->bytes,.flags=0,.hops=0,.src_page=x->src_page
    };
    if(n==MESH_DESC_FLUSH){
      desc->count=n;
      mesh_llist_add(m,mesh_pending_head(m),desc_page);
      desc_page=MESH_PAGE_NONE;desc=NULL;n=0;
    }
  }
  if(desc_page!=MESH_PAGE_NONE){
    if(n){
      desc->count=n;
      mesh_llist_add(m,mesh_pending_head(m),desc_page);
    }else mesh_page_release(m,desc_page);
  }
  return 0;
}

int mesh_reduce_write(struct mesh_ctx *ctx,const struct mesh_red_xfer *xfers,uint32_t count){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m || !xfers || !count)return EINVAL;
  const uint32_t self=m->node,pages=mesh_flat_pages(m);
  struct mesh_switch *sw=mesh_switch_of(m);
  struct mesh_wire_desc *desc=NULL;
  uint32_t desc_page=MESH_PAGE_NONE,n=0;
  uint8_t cur_op=0;
  uint16_t cur_tag=0;
  for(uint32_t i=0;i<count;i++){
    const struct mesh_red_xfer *x=xfers+i;
    if(x->src_page>=pages || x->bytes>MESH_PAGE_BYTES || x->dst_index>=m->entries){
      if(desc_page!=MESH_PAGE_NONE){
        for(uint32_t k=0;k<n;k++)mesh_page_release(m,desc->e[k].src_page);
        mesh_page_release(m,desc_page);
      }
      return EINVAL;
    }
    if(x->dst_node==self){
      if(!mesh_page_hold(m,x->src_page))continue;
      mesh_table_reduce_land(m,sw->c2_stash,x->dst_index,x->src_page,x->bytes,x->op,x->tag);
      continue;
    }
    if(desc_page!=MESH_PAGE_NONE && (n==MESH_DESC_FLUSH || x->op!=cur_op || x->tag!=cur_tag)){
      desc->e[254].flags=cur_op;
      desc->e[254].bytes=cur_tag;
      desc->count=n;
      mesh_llist_add(m,mesh_pending_head(m),desc_page);
      desc_page=MESH_PAGE_NONE;desc=NULL;n=0;
    }
    if(desc_page==MESH_PAGE_NONE){
      desc_page=mesh_page_alloc(m,thread_seed());
      if(desc_page==MESH_PAGE_NONE)return ENOMEM;
      desc=mesh_desc_at(m,desc_page);
      desc->origin_node=self;desc->from_node=self;
      cur_op=x->op;cur_tag=x->tag;
      n=0;
    }
    if(!mesh_page_hold(m,x->src_page))continue;
    desc->e[n++]=(struct mesh_wire_entry){
      .dst_node=x->dst_node,.dst_index=x->dst_index,
      .bytes=(uint16_t)x->bytes,.flags=(cur_op?MESH_WIRE_REDUCE:0u),.hops=0,.src_page=x->src_page
    };
  }
  if(desc_page!=MESH_PAGE_NONE){
    if(n){
      desc->e[254].flags=cur_op;
      desc->e[254].bytes=cur_tag;
      desc->count=n;
      mesh_llist_add(m,mesh_pending_head(m),desc_page);
    }else mesh_page_release(m,desc_page);
  }
  return 0;
}

int mesh_entry_red(struct mesh_ctx *ctx,uint32_t index,struct mesh_entry_info *out,uint16_t *out_tag){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m || !out || index>=m->entries)return EINVAL;
  const uint64_t w=mesh_table_load(m,index);
  const uint32_t p=mesh_word_page(w);
  if(p==MESH_PAGE_NONE || p>=mesh_flat_pages(m)){
    *out=(struct mesh_entry_info){.page=MESH_PAGE_NONE,.bytes=0,.data=NULL};
    if(out_tag)*out_tag=0u;
  }else{
    *out=(struct mesh_entry_info){.page=p,.bytes=mesh_word_bytes(w),.data=mesh_page_at(m,p)};
    if(out_tag)*out_tag=mesh_word_tag(w);
  }
  return 0;
}

int mesh_entry(struct mesh_ctx *ctx,uint32_t index,struct mesh_entry_info *out){
  return mesh_entry_red(ctx,index,out,NULL);
}

void *mesh_page_ptr(struct mesh_ctx *ctx,uint32_t page){
  struct hdr *m=ctx?ctx->M:NULL;
  if(!m || page>=mesh_flat_pages(m))return NULL;
  return mesh_page_at(m,page);
}
