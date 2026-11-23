#include "mesh.h"

/* the client's region, where it has a slot of it; else NULL */
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

static uint32_t new_page(struct hdr *m,uint32_t frames){
  const uint32_t pd=mesh_pool_pop(m,frames);
  if(pd!=MESH_ABSENT){
    struct mesh_pd *p=mesh_pd_at(m,pd);
    atomic_store_explicit(&p->bytes,0,memory_order_relaxed);
    atomic_store_explicit(&p->state,MESH_PD_LIVE,memory_order_relaxed);
    atomic_store_explicit(&p->uses,1,memory_order_relaxed);
  }
  return pd;
}

/* The writer alone assigns this source index. A queued write holds the old backing page through its completion. */
void *mesh_page_writable(struct mesh_ctx *context,uint32_t index,uint32_t bytes){
  struct hdr *m=attached(context);
  const uint32_t frames=mesh_class_frames(bytes);
  if(!m || index>=m->entries || !frames ||
     atomic_load_explicit(&mesh_owners(m)[index],memory_order_acquire)!=mesh_owner_byte(context->client)){
    errno=EINVAL;return NULL;
  }
  uint32_t pd=mesh_word_pd(mesh_entry_load(m,index));
  if(mesh_pd_valid(m,pd)){
    struct mesh_pd *p=mesh_pd_at(m,pd);
    if(p->cls==frames && atomic_load_explicit(&p->uses,memory_order_acquire)==1)return mesh_pd_data(m,pd);
  }
  pd=new_page(m,frames);
  if(pd==MESH_ABSENT){errno=EAGAIN;return NULL;}
  mesh_pd_drop(m,mesh_entry_reset(m,index,pd));
  return mesh_pd_data(m,pd);
}

/* The indices taken (mesh.h mesh_index_take), then each one's word made anew: naming a page popped from the pool
   (its uses 1, the word's) or none, landed 0.  A page a stale landing left at a freed index is dropped as the word is
   made anew.  Where the pool has too few pages, those taken go back and the indices are freed */
uint32_t mesh_pages(struct mesh_ctx *context,uint32_t count,uint32_t bytes){
  struct hdr *m=attached(context);
  const uint32_t frames=mesh_class_frames(bytes);
  if(!m || !count || count>m->entries || !frames){errno=EINVAL;return MESH_ABSENT;}
  const uint32_t first=mesh_index_take(m,mesh_owner_byte(context->client),count);
  if(first==MESH_ABSENT){errno=ENOSPC;return MESH_ABSENT;}
  for(uint32_t i=0;i<count;i++){
    uint32_t pd=MESH_ABSENT;
    if(bytes){
      pd=new_page(m,frames);
      if(pd==MESH_ABSENT){
        for(uint32_t k=0;k<i;k++)mesh_pd_drop(m,mesh_entry_reset(m,first+k,MESH_ABSENT));
        mesh_index_release(m,first,count);
        errno=ENOMEM;return MESH_ABSENT;
      }

    }
    mesh_pd_drop(m,mesh_entry_reset(m,first+i,pd));
  }
  return first;
}

/* The destinations on other nodes one ring cell, those on this node set here.  The page's uses counted first, by one
   compare-exchange from 1 (the word's use alone, so a page with a SEND or a write out is refused): one for the cell
   where it has destinations, one for each destination here, so the page is held while it is set here whatever the
   port children do with the cell meanwhile.  The cell pushed first (its destinations not held up by these), its ring
   flagged for the bridge's port children (mesh.h mesh_ring_flag); then the page given the write's bytes and each
   destination here set: its word naming the page, its landed one more (mesh.h mesh_entry_put), the page it named
   before dropped.  A full ring sets nothing, and its uses are dropped again.  A write whose destinations are all here
   never enters the ring and needs no link */
int mesh_write(struct mesh_ctx *context,uint32_t index,uint32_t bytes,const struct mesh_dest *dests,uint32_t n){
  struct hdr *m=attached(context);
  if(!m || index>=m->entries || !dests || !n || n>MESH_DESTS)return EINVAL;
  for(uint32_t i=0;i<n;i++)if(dests[i].index>=MESH_TABLE)return EINVAL;
  if(atomic_load_explicit(&mesh_owners(m)[index],memory_order_acquire)!=mesh_owner_byte(context->client))return EINVAL;
  const uint32_t pd=mesh_word_pd(mesh_entry_load(m,index));
  if(!mesh_pd_valid(m,pd))return EINVAL;
  struct mesh_pd *p=mesh_pd_at(m,pd);
  if(bytes>MESH_PAYLOAD(p->cls))return EINVAL;
  struct mesh_write w={.pd=pd,.bytes=bytes};
  uint32_t here[MESH_DESTS],local=0;
  for(uint32_t i=0;i<n;i++){
    if(dests[i].node==m->node)here[local++]=dests[i].index;
    else w.dest[w.dests++]=dests[i];
  }
  const uint32_t held=(w.dests?1u:0u)+local;
  uint32_t alone=1;
  if(!atomic_compare_exchange_strong_explicit(&p->uses,&alone,1+held,memory_order_acq_rel,memory_order_relaxed))return EBUSY;
  if(w.dests){
    const uint32_t slot=mesh_owner_byte(context->client)-1u;
    if(!mesh_ring_push(mesh_ring_at(m,slot),&w)){
      for(uint32_t k=0;k<held;k++)mesh_pd_drop(m,pd);
      return EAGAIN;
    }
    mesh_ring_flag(m,slot);
  }
  if(local)atomic_store_explicit(&p->bytes,bytes,memory_order_relaxed);
  for(uint32_t k=0;k<local;k++)mesh_pd_drop(m,mesh_entry_put(m,here[k],pd,1));
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
  const uint64_t owner=client&~MESH_LEAVING;
  struct mesh_client *slot=mesh_client_at(m,owner);
  if(!owner)return;
  if(client&MESH_LEAVING){if(atomic_load_explicit(&slot->owner,memory_order_acquire)!=client)return;}
  else if(!atomic_compare_exchange_strong(&slot->owner,&client,owner|MESH_LEAVING))return;
  const uint8_t own=mesh_owner_byte(owner);
  _Atomic uint8_t *owners=mesh_owners(m);
  for(uint32_t i=0;i<m->entries;i++)
    if(atomic_load_explicit(&owners[i],memory_order_relaxed)==own){
      mesh_pd_drop(m,mesh_entry_put(m,i,MESH_ABSENT,0));
      atomic_store_explicit(&owners[i],0,memory_order_release);
    }
  uint64_t leaving=owner|MESH_LEAVING;
  atomic_compare_exchange_strong(&slot->owner,&leaving,0);
}
