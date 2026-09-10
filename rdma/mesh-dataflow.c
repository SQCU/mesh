#include "mesh-dataflow.h"
#include <stdlib.h>
#include <unistd.h>

// ../design/algorithm-sources.md#configuration-storage-layout
size_t mesh_storage_pages(size_t bytes,uint32_t pagebytes){
  if(!pagebytes || bytes>SIZE_MAX-(pagebytes-1)){ errno=EOVERFLOW; return SIZE_MAX; }
  return (bytes+pagebytes-1)/pagebytes;
}
// ../design/algorithm-sources.md#contiguous-backing-page-views
uint32_t mesh_rows_allocate(struct mesh_rows *p,size_t count,size_t alignment){
  struct mesh_ctx *c=p->context;
  size_t pg=c->M->pgsz;
  if(!count || !alignment || (alignment&(alignment-1))){ errno=EINVAL; return MESH_ROW_ABSENT; }
  if(alignment<pg) alignment=pg;
  size_t words=((size_t)c->M->pool+c->M->arena+63)/64;
  mesh_hot_blocks(c->M,c->hot,words);
  uintptr_t base=(uintptr_t)mesh_at(c->M,c->M->pool);
  size_t first=0;
  while(first<c->M->arena){
    uintptr_t begin=base+first*pg;
    first=(((begin+alignment-1)&~(uintptr_t)(alignment-1))-base)/pg;
    if(c->extent_bytes){
      size_t offset=((size_t)c->M->pool+first)*pg;
      if(count>c->extent_bytes/pg){ errno=EOVERFLOW; return MESH_ROW_ABSENT; }
      if(offset/c->extent_bytes!=(offset+count*pg-1)/c->extent_bytes)
        first=((offset/c->extent_bytes+1)*c->extent_bytes)/pg-c->M->pool;
    }
    if(first>c->M->arena || count>c->M->arena-first){ errno=ENOMEM; return MESH_ROW_ABSENT; }
    size_t i=0;
    for(;i<count;i++){
      size_t page=c->M->pool+first+i;
      if((c->blocks[page/64]|c->hot[page/64])&(UINT64_C(1)<<(page%64))) break;
    }
    if(i==count) break;
    first+=i+1;
  }
  if(first>=c->M->arena){ errno=ENOMEM; return MESH_ROW_ABSENT; }
  for(size_t i=0;i<count;i++){
    size_t page=c->M->pool+first+i;
    uint64_t bit=UINT64_C(1)<<(page%64);
    c->blocks[page/64]|=bit; p->blocks[page/64]|=bit;
  }
  if(first+count>c->allocation) c->allocation=first+count;
  return c->M->pool+(uint32_t)first;
}
// ../design/algorithm-sources.md#literal-row-functions
struct mesh_rows *mesh_rows_create(struct mesh_ctx *c,size_t rows,uint64_t identity){
  if(!c || !c->M || !rows || rows>UINT32_MAX){ errno=EINVAL; return NULL; }
  struct mesh_rows *p=calloc(1,sizeof *p);
  if(!p) return NULL;
  *p=(struct mesh_rows){.memory=c->M,.count=rows,.bytes=c->M->pgsz,.context=c,.identity=identity};
  p->blocks=calloc(((size_t)c->M->pool+c->M->arena+63)/64,sizeof(uint64_t));
  if(!p->blocks){ free(p); return NULL; }
  uint32_t at=mesh_rows_allocate(p,mesh_storage_pages(rows*sizeof(struct mesh_row),p->bytes),p->bytes);
  if(at==MESH_ROW_ABSENT){ free(p); return NULL; }
  p->table=(void*)mesh_at(c->M,at);
  for(size_t i=0;i<rows;i++) p->table[i]=(struct mesh_row){.page=MESH_ROW_ABSENT};
  struct mesh_rows **tables=realloc(c->tables,(c->table_count+1)*sizeof *tables);
  if(!tables){ free(p); return NULL; }
  c->tables=tables; tables[c->table_count++]=p; return p;
}
// ../design/algorithm-sources.md#literal-row-functions
void mesh_rows_map(struct mesh_rows *p,uint32_t first,uint32_t physical,uint32_t count,uint32_t uses,uint64_t stamp){
  for(uint32_t i=0;i<count;i++){
    struct mesh_row *r=&p->table[first+i];
    r->page=physical+i; r->uses=uses;
    __atomic_store_n(&r->stamp,stamp,__ATOMIC_RELEASE);
    if(stamp && stamp<MESH_ROW_WRITING) mesh_send_row(p->memory,r,stamp);
  }
}
// ../design/algorithm-sources.md#literal-row-functions
void mesh_rows_constant(struct mesh_rows *p,uint32_t first,uint32_t count){
  for(uint32_t i=0;i<count;i++){
    struct mesh_row *row=&p->table[first+i];
    row->uses=row->send?((struct mesh_send*)((unsigned char*)p->memory+row->send))->count:0;
    __atomic_store_n(&row->stamp,MESH_ROW_CONSTANT,__ATOMIC_RELEASE);
    mesh_send_row(p->memory,row,MESH_ROW_CONSTANT);
  }
}
// ../design/algorithm-sources.md#literal-row-functions
static int mesh_map_valid(const struct mesh_rows *p,struct mesh_row_map m,uint32_t occurrences,int output){
  for(uint32_t i=0;i<occurrences;i++){
    struct mesh_row_range r=mesh_range(m,i);
    if(!r.count || (uint64_t)r.first+r.count>p->count) return EINVAL;
    if(output && ((uint64_t)m.physical+(uint64_t)i*m.physical_stride+r.count>(uint64_t)p->memory->pool+p->memory->arena || m.physical<p->memory->pool)) return EINVAL;
  }
  return 0;
}
// ../design/algorithm-sources.md#literal-row-functions
int mesh_rows_realize(const struct mesh_rows *p0,const struct mesh_row_function *functions,size_t count,
  struct mesh_row_binding *bindings,size_t binding_count,struct mesh_row_map *returns,size_t return_count){
  struct mesh_rows *p=(struct mesh_rows*)p0;
  if(!p || !functions || !count || (binding_count && !bindings) || (return_count && !returns)) return EINVAL;
  p->functions=functions; p->function_count=count; p->bindings=bindings; p->binding_count=binding_count;
  p->returns=returns; p->return_count=return_count;
  for(size_t i=0;i<p->count;i++) p->table[i].uses=0;
  for(size_t i=0;i<count;i++){
    const struct mesh_row_function *f=&functions[i];
    if(!f->rows || !f->outputs || !f->output || (f->inputs && !f->input)) return EINVAL;
    for(uint32_t side=0;side<2;side++){
      struct mesh_row_map *maps=side?f->output:f->input;
      uint32_t n=side?f->outputs:f->inputs;
      for(uint32_t j=0;j<n;j++){
        int e=mesh_map_valid(p,maps[j],f->rows,side); if(e) return e;
        for(uint32_t k=0;k<f->rows;k++){
          struct mesh_row_range r=mesh_range(maps[j],k);
          if(side){
            for(uint32_t x=0;x<r.count;x++){
              struct mesh_row *row=&p->table[r.first+x];
              if(row->stamp==MESH_ROW_WRITING) return EINVAL;
              if(row->stamp!=MESH_ROW_CONSTANT){ row->stamp=MESH_ROW_WRITING; row->page=MESH_ROW_ABSENT; }
            }
          }else for(uint32_t x=0;x<r.count;x++) p->table[r.first+x].uses++;
        }
      }
    }
  }
  size_t sends=0;
  for(size_t i=0;i<binding_count;i++){
    struct mesh_row_binding *b=&bindings[i];
    if((uint64_t)b->first+b->count>p->count || b->receive>1) return EINVAL;
    if(b->receive){
      for(uint32_t j=0;j<b->count;j++) if(p->table[b->first+j].stamp==MESH_ROW_WRITING) return EINVAL;
      for(size_t j=0;j<i;j++) if(bindings[j].receive &&
        (uint64_t)b->first< (uint64_t)bindings[j].first+bindings[j].count &&
        (uint64_t)bindings[j].first<(uint64_t)b->first+b->count) return EINVAL;
    }else { sends+=b->count; for(uint32_t j=0;j<b->count;j++) p->table[b->first+j].uses++; }
  }
  for(size_t i=0;i<return_count;i++){
    if(mesh_map_valid(p,returns[i],1,0)) return EINVAL;
    struct mesh_row_range r=mesh_range(returns[i],0);
    for(uint32_t j=0;j<r.count;j++) p->table[r.first+j].uses++;
  }
  for(size_t i=0;i<count;i++) for(uint32_t j=0;j<functions[i].outputs;j++){
    struct mesh_row_map *m=&functions[i].output[j];
    m->uses=p->table[mesh_range(*m,0).first].uses;
    for(uint32_t k=0;k<functions[i].rows;k++){
      struct mesh_row_range r=mesh_range(*m,k);
      for(uint32_t x=0;x<r.count;x++) if(p->table[r.first+x].uses!=m->uses) return EINVAL;
    }
  }
  for(size_t i=0;i<binding_count;i++){
    struct mesh_row_binding *b=&bindings[i];
    b->uses=b->count?p->table[b->first].uses:0;
    if(b->receive) for(uint32_t j=0;j<b->count;j++) if(p->table[b->first+j].uses!=b->uses) return EINVAL;
  }
  size_t metadata=sizeof(struct mesh_table)+binding_count*sizeof(struct mesh_receive_binding);
  uint32_t base=mesh_rows_allocate(p,mesh_storage_pages(metadata,p->bytes),p->bytes);
  if(base==MESH_ROW_ABSENT) return errno;
  struct mesh_table *shared=(void*)mesh_at(p->memory,base);
  p->shared=(uint64_t)((unsigned char*)shared-(unsigned char*)p->memory);
  struct mesh_receive_binding *receive=(void*)(shared+1);
  *shared=(struct mesh_table){.identity=p->identity,.rows=(uint64_t)((unsigned char*)p->table-(unsigned char*)p->memory),
    .bindings=(uint64_t)((unsigned char*)receive-(unsigned char*)p->memory),.count=(uint32_t)p->count,.binding_count=(uint32_t)binding_count};
  for(size_t i=0;i<binding_count;i++) receive[i]=(struct mesh_receive_binding){bindings[i].first,bindings[i].receive?bindings[i].count:0,bindings[i].uses,0};
  if(sends){
    uint32_t at=mesh_rows_allocate(p,mesh_send_storage_pages(sends,p->bytes),p->bytes);
    if(at==MESH_ROW_ABSENT) return errno;
    uint64_t record_base=(uint64_t)(mesh_at(p->memory,at)-(unsigned char*)p->memory);
    size_t cursor=0;
    for(uint32_t r=0;r<p->count;r++){
      size_t first=cursor;
      for(size_t i=0;i<binding_count;i++){
        const struct mesh_row_binding *b=&bindings[i];
        if(b->receive || r<b->first || r-b->first>=b->count) continue;
        *mesh_record(p->memory,record_base,cursor++)=(struct mesh_send){.header={.wire={p->memory->node,b->peer,0},.table=b->remote_table,
          .source=r,.target=b->remote,.index=r-b->first,.peer=b->peer},
          .row=(uint64_t)((unsigned char*)&p->table[r]-(unsigned char*)p->memory)};
      }
      if(cursor!=first){ struct mesh_send *record=mesh_record(p->memory,record_base,first); record->count=(uint32_t)(cursor-first); p->table[r].send=(uint64_t)((unsigned char*)record-(unsigned char*)p->memory); }
    }
  }
  for(size_t i=0;i<p->count;i++) if(p->table[i].stamp!=MESH_ROW_CONSTANT){
    if(p->table[i].uses && p->table[i].stamp!=MESH_ROW_WRITING && p->table[i].page==MESH_ROW_ABSENT){
      int receive=0;
      for(size_t j=0;j<binding_count;j++) receive|=bindings[j].receive && i>=bindings[j].first && i-bindings[j].first<bindings[j].count;
      if(!receive) return EINVAL;
    }
    p->table[i].stamp=0; p->table[i].uses=0;
  }
  shared->next=atomic_load_explicit(&p->memory->tables,memory_order_relaxed);
  atomic_store_explicit(&p->memory->tables,p->shared,memory_order_release);
  for(uint32_t i=0;i<p->memory->pool;i++) mesh_bind_receive(p->memory,i);
  return 0;
}
// ../design/algorithm-sources.md#literal-row-functions
void *mesh_row_data(const struct mesh_rows *p,uint32_t row){
  uint32_t page=__atomic_load_n(&p->table[row].page,__ATOMIC_ACQUIRE);
  return page==MESH_ROW_ABSENT?NULL:mesh_at(p->memory,page)+p->offset;
}
// ../design/algorithm-sources.md#literal-row-functions
int mesh_rows_present(const struct mesh_rows *p,struct mesh_row_map m,uint32_t index,uint64_t stamp){
  struct mesh_row_range r=mesh_range(m,index);
  for(uint32_t i=0;i<r.count;i++){
    const struct mesh_row *row=&p->table[r.first+i];
    uint64_t value=__atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE);
    if((value!=stamp && value!=MESH_ROW_CONSTANT) || __atomic_load_n(&row->page,__ATOMIC_ACQUIRE)==MESH_ROW_ABSENT) return 0;
  }
  return 1;
}
// ../design/algorithm-sources.md#literal-row-functions
size_t mesh_rows_issue(const struct mesh_rows *p,const struct mesh_row_function *f,uint64_t stamp,uint32_t *indices,size_t capacity){
  size_t selected=0;
  for(uint32_t i=0;i<f->rows && selected<capacity;i++){
    int ready=1;
    for(uint32_t j=0;j<f->inputs && ready;j++) ready=mesh_rows_present(p,f->input[j],i,stamp);
    for(uint32_t j=0;j<f->outputs && ready;j++){
      struct mesh_row_range r=mesh_range(f->output[j],i);
      for(uint32_t k=0;k<r.count && ready;k++){
        const struct mesh_row *row=&p->table[r.first+k];
        ready=__atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE)<stamp && __atomic_load_n(&row->page,__ATOMIC_ACQUIRE)==MESH_ROW_ABSENT;
      }
    }
    if(!ready) continue;
    for(uint32_t j=0;j<f->outputs;j++){
      struct mesh_row_map m=f->output[j]; struct mesh_row_range r=mesh_range(m,i);
      for(uint32_t k=0;k<r.count;k++){
        struct mesh_row *row=&p->table[r.first+k];
        row->page=m.physical+i*m.physical_stride+k; row->uses=m.uses;
        __atomic_store_n(&row->stamp,MESH_ROW_WRITING|stamp,__ATOMIC_RELEASE);
      }
    }
    indices[selected++]=i;
  }
  return selected;
}
// ../design/algorithm-sources.md#literal-row-functions
void mesh_rows_complete(const struct mesh_rows *p,const struct mesh_row_function *f,uint64_t stamp,const uint32_t *indices,size_t count){
  for(size_t n=count;n>0;n--){
    size_t i=n-1;
    for(uint32_t j=0;j<f->inputs;j++){
      struct mesh_row_range r=mesh_range(f->input[j],indices[i]);
      for(uint32_t k=0;k<r.count;k++) mesh_release(p->memory,&p->table[r.first+k]);
    }
    for(uint32_t j=0;j<f->outputs;j++){
      struct mesh_row_range r=mesh_range(f->output[j],indices[i]);
      for(uint32_t k=0;k<r.count;k++){
        struct mesh_row *row=&p->table[r.first+k];
        if(!row->uses) __atomic_store_n(&row->page,MESH_ROW_ABSENT,__ATOMIC_RELAXED);
        __atomic_store_n(&row->stamp,stamp,__ATOMIC_RELEASE);
        mesh_send_row(p->memory,row,stamp);
      }
    }
  }
}
// ../design/algorithm-sources.md#asynchronous-metadata-publication
void mesh_rows_report(const struct mesh_rows *p,struct mesh_row_map output,uint32_t occurrence,struct mesh_row_metadata metadata){
  memcpy(mesh_row_data(p,mesh_range(output,occurrence).first),&metadata,sizeof metadata);
}
// ../design/algorithm-sources.md#literal-row-functions
void mesh_row_release(const struct mesh_rows *p,uint32_t row){ mesh_release(p->memory,&p->table[row]); }
// ../design/algorithm-sources.md#context-lifetime
void mesh_rows_retire(struct mesh_rows *p){
  struct mesh_table *shared=(struct mesh_table*)((unsigned char*)p->memory+p->shared);
  uint64_t at=atomic_load_explicit(&p->memory->tables,memory_order_acquire),previous=0;
  while(at && at!=p->shared){
    previous=at;
    at=__atomic_load_n(&((struct mesh_table*)((unsigned char*)p->memory+at))->next,__ATOMIC_ACQUIRE);
  }
  if(at){
    if(previous) __atomic_store_n(&((struct mesh_table*)((unsigned char*)p->memory+previous))->next,shared->next,__ATOMIC_RELEASE);
    else atomic_store_explicit(&p->memory->tables,shared->next,memory_order_release);
  }
  __atomic_store_n(&shared->identity,UINT64_MAX,__ATOMIC_RELEASE);
  size_t words=((size_t)p->memory->pool+p->memory->arena+63)/64;
  for(size_t i=0;i<words;i++) p->context->blocks[i]&=~p->blocks[i];
  for(size_t i=0;i<p->count;i++){
    uint32_t page=__atomic_exchange_n(&p->table[i].page,MESH_ROW_ABSENT,__ATOMIC_ACQ_REL);
    if(page<p->memory->pool) push(p->memory,REL,&(struct desc){.page=page});
  }
}
// ../design/algorithm-sources.md#context-lifetime
int mesh_rows_close(struct mesh_ctx *c){
  atomic_store_explicit(&c->M->tables,0,memory_order_release);
  for(size_t i=0;i<c->table_count;i++){
    mesh_rows_retire(c->tables[i]);
    free(c->tables[i]->blocks);
    free(c->tables[i]);
  }
  free(c->tables); c->tables=NULL; c->table_count=0; return 0;
}
