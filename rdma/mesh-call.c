#include "mesh.h"

/* the client's region, where it holds a slot of it; else NULL */
static struct hdr *attached(struct mesh_ctx *context){ return context && context->M && context->client?context->M:NULL; }

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

/* The indices claimed (mesh.h mesh_index_take), then each one's word made anew: naming a page popped from the pool
   (its uses 1, the word's) or none, landed 0.  A page a stale landing left at a freed index is dropped as the word is
   made anew.  Where the pool runs short, the pages taken go back and the indices are freed */
uint32_t mesh_pages(struct mesh_ctx *context,uint32_t count,uint32_t bytes){
  struct hdr *m=attached(context);
  const uint32_t frames=mesh_class_frames(bytes);
  if(!m || !count || count>m->entries || !frames){errno=EINVAL;return MESH_ABSENT;}
  const uint32_t first=mesh_index_take(m,mesh_owner_byte(context->client),count,&context->index_cursor);
  if(first==MESH_ABSENT){errno=ENOSPC;return MESH_ABSENT;}
  for(uint32_t i=0;i<count;i++){
    uint32_t pd=MESH_ABSENT;
    if(bytes){
      pd=mesh_pool_pop(m,frames);
      if(pd==MESH_ABSENT){
        for(uint32_t k=0;k<i;k++)mesh_pd_drop(m,mesh_entry_reset(m,first+k,MESH_ABSENT));
        mesh_index_release(m,first,count);
        errno=ENOMEM;return MESH_ABSENT;
      }
      struct mesh_pd *p=mesh_pd_at(m,pd);
      atomic_store_explicit(&p->bytes,0,memory_order_relaxed);
      atomic_store_explicit(&p->state,MESH_PD_LIVE,memory_order_relaxed);
      atomic_store_explicit(&p->uses,1,memory_order_relaxed);
    }
    mesh_pd_drop(m,mesh_entry_reset(m,first+i,pd));
  }
  return first;
}

/* One ring cell: the page's use for the write counted by a compare-exchange from 1 (the word's use alone), so a page
   with a SEND or a write out is refused, and dropped again where the ring is full */
int mesh_write(struct mesh_ctx *context,uint32_t index,uint32_t bytes,const struct mesh_dest *dests,uint32_t n){
  struct hdr *m=attached(context);
  if(!m || index>=m->entries || !dests || !n || n>MESH_DESTS)return EINVAL;
  for(uint32_t i=0;i<n;i++)if(dests[i].index>=MESH_TABLE)return EINVAL;
  if(atomic_load_explicit(&mesh_owners(m)[index],memory_order_acquire)!=mesh_owner_byte(context->client))return EINVAL;
  const uint32_t pd=mesh_word_pd(mesh_entry_load(m,index));
  if(!mesh_pd_valid(m,pd))return EINVAL;
  struct mesh_pd *p=mesh_pd_at(m,pd);
  if(bytes>MESH_PAYLOAD(p->cls))return EINVAL;
  uint32_t alone=1;
  if(!atomic_compare_exchange_strong_explicit(&p->uses,&alone,2,memory_order_acq_rel,memory_order_relaxed))return EBUSY;
  struct mesh_write w={.pd=pd,.bytes=bytes,.dests=n};
  for(uint32_t i=0;i<n;i++)w.dest[i]=dests[i];
  if(!mesh_ring_push(mesh_ring_at(m,mesh_owner_byte(context->client)-1u),&w)){mesh_pd_drop(m,pd);return EAGAIN;}
  return 0;
}

int mesh_invalidate(struct mesh_ctx *context,uint32_t first,uint32_t count,int indices){
  struct hdr *m=attached(context);
  if(!m || first>=m->entries || count>m->entries-first)return EINVAL;
  const uint8_t own=mesh_owner_byte(context->client);
  for(uint32_t i=first;i<first+count;i++)if(atomic_load_explicit(&mesh_owners(m)[i],memory_order_acquire)!=own)return EINVAL;
  for(uint32_t i=first;i<first+count;i++){
    mesh_pd_drop(m,mesh_entry_put(m,i,MESH_ABSENT,0));
    if(indices)atomic_store_explicit(&mesh_owners(m)[i],0,memory_order_release);
  }
  return 0;
}

int mesh_entry(struct mesh_ctx *context,uint32_t index,struct mesh_view_entry *entry){
  struct hdr *m=context?context->M:NULL;
  if(!m || !entry || index>=m->entries)return EINVAL;
  const uint64_t word=mesh_entry_load(m,index);
  const uint32_t pd=mesh_word_pd(word);
  *entry=(struct mesh_view_entry){.landed=mesh_word_landed(word)};
  if(mesh_pd_valid(m,pd)){
    const struct mesh_pd *p=mesh_pd_at(m,pd);
    entry->bytes=atomic_load_explicit(&p->bytes,memory_order_relaxed);
    entry->uses=atomic_load_explicit(&p->uses,memory_order_relaxed);
    entry->data=mesh_pd_data(m,pd);
  }
  return 0;
}
void *mesh_page_data(struct mesh_ctx *context,uint32_t index){
  struct hdr *m=context?context->M:NULL;
  if(!m || index>=m->entries)return NULL;
  const uint32_t pd=mesh_word_pd(mesh_entry_load(m,index));
  return mesh_pd_valid(m,pd)?mesh_pd_data(m,pd):NULL;
}

void mesh_leave(struct hdr *m,uint64_t client){
  const uint64_t held=client&~MESH_LEAVING;
  struct mesh_client *slot=mesh_client_at(m,held);
  if(!held)return;
  if(client&MESH_LEAVING){if(atomic_load_explicit(&slot->owner,memory_order_acquire)!=client)return;}
  else if(!atomic_compare_exchange_strong(&slot->owner,&client,held|MESH_LEAVING))return;
  const uint8_t own=mesh_owner_byte(held);
  _Atomic uint8_t *owners=mesh_owners(m);
  for(uint32_t i=0;i<m->entries;i++)
    if(atomic_load_explicit(&owners[i],memory_order_relaxed)==own){
      mesh_pd_drop(m,mesh_entry_put(m,i,MESH_ABSENT,0));
      atomic_store_explicit(&owners[i],0,memory_order_release);
    }
  uint64_t leaving=held|MESH_LEAVING;
  atomic_compare_exchange_strong(&slot->owner,&leaving,0);
  mesh_notify(&m->control);
}
