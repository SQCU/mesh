#include "mesh-dataflow.h"
#include <errno.h>
#include <stdlib.h>
#include <unistd.h>

// ../design/algorithm-sources.md#complete-page-ownership
uint32_t mesh_rows_allocate(struct mesh_rows *p, size_t count, size_t alignment){
  struct mesh_ctx *c=p->context;
  if(!count || !alignment || (alignment&(alignment-1))){ errno=EINVAL; return MESH_ROW_ABSENT; }
  if(alignment<p->bytes) alignment=p->bytes;
  uintptr_t base=(uintptr_t)mesh_at(c->M,c->M->pool);
  uintptr_t begin=base+c->allocation*p->bytes;
  if(begin>UINTPTR_MAX-alignment+1){ errno=EOVERFLOW; return MESH_ROW_ABSENT; }
  size_t first=(((begin+alignment-1)&~(uintptr_t)(alignment-1))-base)/p->bytes;
  if(first>c->M->arena || count>c->M->arena-first){ errno=ENOMEM; return MESH_ROW_ABSENT; }
  c->allocation=first+count;
  return c->M->pool+(uint32_t)first;
}

// ../design/algorithm-sources.md#complete-page-ownership
struct mesh_rows *mesh_rows_create(struct mesh_ctx *c, size_t rows, uint64_t identity){
  if(!c || !c->M || !rows || rows>UINT32_MAX || identity!=c->table_count){ errno=EINVAL; return NULL; }
  struct mesh_rows *p=calloc(1,sizeof *p);
  if(!p) return NULL;
  *p=(struct mesh_rows){.memory=c->M,.count=rows,.bytes=c->M->pgsz,.context=c,.identity=identity};
  size_t count=mesh_storage_pages(rows*sizeof(struct mesh_row),p->bytes);
  uint32_t physical=mesh_rows_allocate(p,count,(size_t)getpagesize());
  if(physical==MESH_ROW_ABSENT){ free(p); return NULL; }
  p->table=(void*)mesh_at(c->M,physical);
  for(size_t i=0;i<rows;i++) p->table[i]=(struct mesh_row){.page=MESH_ROW_ABSENT};
  struct mesh_rows **tables=realloc(c->tables,(c->table_count+1)*sizeof *tables);
  if(!tables){ free(p); return NULL; }
  c->tables=tables;
  c->tables[c->table_count++]=p;
  return p;
}

// ../design/algorithm-sources.md#complete-page-ownership
void mesh_rows_map(struct mesh_rows *p,uint32_t first,uint32_t physical,
  uint32_t count,uint32_t uses,uint64_t stamp){
  for(uint32_t i=0;i<count;i++) p->table[first+i]=(struct mesh_row){physical+i,uses,stamp};
}

// ../design/algorithm-sources.md#complete-page-ownership
int mesh_rows_invalidate(struct mesh_rows **pages,const struct mesh_row_map *held,size_t count){
  struct mesh_rows *p=*pages;
  if(count>p->return_count) return EINVAL;
  for(size_t i=0;i<count;i++){
    struct mesh_row_range value=held[i].ranges?held[i].ranges[0]:(struct mesh_row_range){held[i].first,held[i].count};
    int found=0;
    for(size_t j=0;j<p->return_count;j++) found|=p->returns[j].ranges[0].first==value.first && p->returns[j].ranges[0].count==value.count;
    if(!found) return EINVAL;
    for(size_t j=0;j<i;j++){
      struct mesh_row_range previous=held[j].ranges?held[j].ranges[0]:(struct mesh_row_range){held[j].first,held[j].count};
      if(value.first==previous.first) return EINVAL;
    }
  }
  *pages=NULL;
  for(size_t i=0;i<p->function_count;i++){
    const struct mesh_row_function *f=&p->functions[i];
    for(uint32_t index=0;index<f->rows;index++){
      const struct mesh_row_map *out=f->output;
      uint64_t consumed=__atomic_load_n(&p->table[out->first+index*out->stride].stamp,__ATOMIC_ACQUIRE)&~MESH_ROW_WRITING;
      for(uint32_t j=0;j<f->inputs;j++){
        struct mesh_row_range range=f->input[j].ranges[index];
        for(uint32_t k=0;k<range.count;k++){
          struct mesh_row *row=&p->table[range.first+k];
          uint64_t stamp=__atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE)&~MESH_ROW_WRITING;
          if(stamp && stamp!=consumed &&
            __atomic_load_n(&row->page,__ATOMIC_ACQUIRE)!=MESH_ROW_ABSENT) mesh_row_release(p,range.first+k);
        }
      }
    }
  }
  for(size_t i=0;i<p->binding_count;i++){
    struct mesh_row_binding *b=(struct mesh_row_binding*)&p->bindings[i];
    for(uint32_t index=0;index<b->count;index++){
      struct mesh_row *row=&p->table[b->first+index];
      if(b->receive){
        ((uint32_t*)b->uses)[index]=0;
      }else{
        const struct mesh_send *record=(struct mesh_send*)mesh_at(p->memory,b->headers)+index;
        uint64_t stamp=__atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE)&~MESH_ROW_WRITING;
        if(stamp && record->header.stamp!=stamp &&
          __atomic_load_n(&row->page,__ATOMIC_ACQUIRE)!=MESH_ROW_ABSENT) mesh_row_release(p,b->first+index);
      }
    }
    if(!b->receive) b->count=0;
  }
  struct mesh_row_map *owned=(struct mesh_row_map*)p->returns;
  // held=owned+d => touched_before(i)={k,d+k | 0<=k<i} < d+i
  for(size_t i=0;i<count;i++){
    struct mesh_row_range value=held[i].ranges?held[i].ranges[0]:(struct mesh_row_range){held[i].first,held[i].count};
    size_t j=i;
    while(owned[j].ranges[0].first!=value.first || owned[j].ranges[0].count!=value.count) j++;
    struct mesh_row_map previous=owned[i]; owned[i]=owned[j]; owned[j]=previous;
  }
  p->return_count=count;
  for(size_t i=1;i<count;i++){
    struct mesh_row_map value=owned[i]; size_t j=i;
    while(j && owned[j-1].ranges[0].first>value.ranges[0].first){ owned[j]=owned[j-1]; j--; }
    owned[j]=value;
  }
  return 0;
}

// ../design/algorithm-sources.md#configuration-storage-layout
size_t mesh_storage_pages(size_t bytes,uint32_t page_bytes){
  if(!page_bytes || bytes>SIZE_MAX-(page_bytes-1)){ errno=EOVERFLOW; return SIZE_MAX; }
  return (bytes+page_bytes-1)/page_bytes;
}

// ../design/algorithm-sources.md#configuration-storage-layout
size_t mesh_region_table_pages(size_t pages,uint32_t page_bytes){
  if(!page_bytes || pages>SIZE_MAX/page_bytes){ errno=EOVERFLOW; return SIZE_MAX; }
  size_t regions=mesh_storage_pages(pages*page_bytes,UINT32_C(1)<<30);
  if(regions==SIZE_MAX || regions>SIZE_MAX/sizeof(uint64_t)){ errno=EOVERFLOW; return SIZE_MAX; }
  return mesh_storage_pages(regions*sizeof(uint64_t),page_bytes);
}

// ../design/algorithm-sources.md#configuration-storage-layout
static void *row_storage(const struct mesh_rows *p,uint32_t page_bytes,size_t *total,size_t count,size_t width){
  if(*total==SIZE_MAX) return NULL;
  if(count && width>SIZE_MAX/count){ *total=SIZE_MAX; errno=EOVERFLOW; return NULL; }
  size_t pages=mesh_storage_pages(count*width,page_bytes);
  if(pages==SIZE_MAX || *total>SIZE_MAX-pages){ *total=SIZE_MAX; errno=EOVERFLOW; return NULL; }
  *total+=pages;
  if(!p || !pages) return NULL;
  uint32_t physical=mesh_rows_allocate((struct mesh_rows*)p,pages,page_bytes);
  if(physical==MESH_ROW_ABSENT){ *total=SIZE_MAX; return NULL; }
  return mesh_at(p->memory,physical);
}

// ../design/algorithm-sources.md#configuration-storage-layout
static void row_ranges(const struct mesh_rows *p,uint32_t page_bytes,size_t *total,
  struct mesh_row_map *map,uint32_t rows){
  struct mesh_row_range *ranges=row_storage(p,page_bytes,total,rows,sizeof *ranges);
  if(!p || !ranges) return;
  if(map->ranges) memcpy(ranges,map->ranges,(size_t)rows*sizeof *ranges);
  else for(uint32_t i=0;i<rows;i++){
    uint64_t first=(uint64_t)map->first+(uint64_t)i*map->stride;
    if(first+map->count>p->count){ *total=SIZE_MAX; errno=EINVAL; return; }
    ranges[i]=(struct mesh_row_range){(uint32_t)first,map->count};
  }
  map->ranges=ranges;
}

// ../design/algorithm-sources.md#configuration-storage-layout
static size_t row_layout(const struct mesh_rows *p,uint32_t page_bytes,size_t rows,
  const struct mesh_row_function *functions,size_t count,struct mesh_row_binding *bindings,
  size_t binding_count,struct mesh_row_map *returns,size_t return_count,uint64_t **delta){
  if(!page_bytes || (page_bytes&(page_bytes-1)) || !rows || rows>UINT32_MAX || !count || !functions ||
    (binding_count && !bindings) || (return_count && !returns)){ errno=EINVAL; return SIZE_MAX; }
  size_t total=0;
  for(size_t i=0;i<count;i++){
    if(!functions[i].rows || !functions[i].outputs || !functions[i].output ||
      (functions[i].inputs && !functions[i].input)){ errno=EINVAL; return SIZE_MAX; }
    for(uint32_t side=0;side<2;side++){
      struct mesh_row_map *maps=side?functions[i].output:functions[i].input;
      uint32_t n=side?functions[i].outputs:functions[i].inputs;
      for(uint32_t j=0;j<n;j++) row_ranges(p,page_bytes,&total,&maps[j],functions[i].rows);
    }
  }
  for(size_t i=0;i<return_count;i++) row_ranges(p,page_bytes,&total,&returns[i],1);
  void *counts=row_storage(p,page_bytes,&total,rows+1,sizeof(uint64_t));
  if(p) *delta=counts;
  for(size_t i=0;i<count;i++) for(uint32_t j=0;j<functions[i].outputs;j++){
    struct mesh_row_map *map=&functions[i].output[j];
    void *uses=row_storage(p,page_bytes,&total,(size_t)functions[i].rows*map->count,sizeof(uint32_t));
    if(p) map->uses=uses;
  }
  for(size_t i=0;i<binding_count;i++){
    struct mesh_row_binding *b=&bindings[i];
    if(!b->count) continue;
    void *uses=row_storage(p,page_bytes,&total,b->count,sizeof(uint32_t));
    if(p) b->uses=uses;
    if(!b->receive){
      void *records=row_storage(p,page_bytes,&total,b->count,sizeof(struct mesh_send));
      if(p && records) b->headers=(uint32_t)(((unsigned char*)records-mesh_at(p->memory,0))/page_bytes);
    }
  }
  struct mesh_row_function *configured=row_storage(p,page_bytes,&total,count,sizeof *configured);
  if(p && configured) memcpy(configured,functions,count*sizeof *configured);
  for(size_t i=0;i<count;i++) for(uint32_t side=0;side<2;side++){
    size_t n=side?functions[i].outputs:functions[i].inputs;
    struct mesh_row_map *copy=row_storage(p,page_bytes,&total,n,sizeof *copy);
    if(p && copy && configured){
      memcpy(copy,side?functions[i].output:functions[i].input,n*sizeof *copy);
      if(side) configured[i].output=copy; else configured[i].input=copy;
    }
  }
  struct mesh_row_binding *links=row_storage(p,page_bytes,&total,binding_count,sizeof *links);
  if(p && links) memcpy(links,bindings,binding_count*sizeof *links);
  struct mesh_row_map *out=row_storage(p,page_bytes,&total,return_count,sizeof *out);
  if(p && out) memcpy(out,returns,return_count*sizeof *out);
  if(p && total!=SIZE_MAX){
    struct mesh_rows *owned=(struct mesh_rows*)p;
    owned->functions=configured; owned->function_count=count;
    owned->bindings=links; owned->binding_count=binding_count;
    owned->returns=out; owned->return_count=return_count;
  }
  return total;
}

// ../design/algorithm-sources.md#configuration-storage-layout
size_t mesh_rows_configuration_pages(uint32_t page_bytes,size_t rows,
  const struct mesh_row_function *functions,size_t count,const struct mesh_row_binding *bindings,
  size_t binding_count,const struct mesh_row_map *returns,size_t return_count){
  return row_layout(NULL,page_bytes,rows,functions,count,(struct mesh_row_binding*)bindings,
    binding_count,(struct mesh_row_map*)returns,return_count,NULL);
}

// ../design/algorithm-sources.md#literal-row-functions
static int row_maps_overlap(struct mesh_row_map a,uint32_t na,struct mesh_row_map b,uint32_t nb,int physical){
  uint64_t a0=physical?a.physical:a.ranges[0].first,b0=physical?b.physical:b.ranges[0].first;
  uint64_t az=physical?(uint64_t)a.physical+(uint64_t)(na-1)*a.physical_stride+a.count:
    (uint64_t)a.ranges[na-1].first+a.ranges[na-1].count;
  uint64_t bz=physical?(uint64_t)b.physical+(uint64_t)(nb-1)*b.physical_stride+b.count:
    (uint64_t)b.ranges[nb-1].first+b.ranges[nb-1].count;
  if(az<=b0 || bz<=a0) return 0;
  for(uint32_t i=0;i<na;i++) for(uint32_t j=0;j<nb;j++){
    uint64_t first_a=physical?(uint64_t)a.physical+(uint64_t)i*a.physical_stride:a.ranges[i].first;
    uint64_t first_b=physical?(uint64_t)b.physical+(uint64_t)j*b.physical_stride:b.ranges[j].first;
    uint64_t count_a=physical?a.count:a.ranges[i].count,count_b=physical?b.count:b.ranges[j].count;
    if(first_a<first_b+count_b && first_b<first_a+count_a) return 1;
  }
  return 0;
}

// ../design/algorithm-sources.md#literal-row-functions
static int mesh_rows_validate(const struct mesh_rows *p, const struct mesh_row_function *f){
  if(!p || !p->memory || !p->table || !p->bytes || p->offset || p->bytes!=p->memory->pgsz ||
     p->offset%8 || p->bytes%8 || !f || !f->rows || !f->outputs || !f->output || (f->inputs && !f->input)) return EINVAL;
  const struct mesh_row_map *indices=&f->indices;
  if(indices->count && ((uint64_t)indices->first+indices->count>p->count ||
    indices->physical<p->memory->pool ||
    (uint64_t)indices->physical+indices->count>(uint64_t)p->memory->pool+p->memory->arena ||
    (uint64_t)indices->count*p->bytes<((uint64_t)f->rows+1)*sizeof(uint32_t))) return EINVAL;
  for(uint32_t side=0;side<2;side++){
    const struct mesh_row_map *maps=side?f->output:f->input;
    uint32_t count=side?f->outputs:f->inputs;
    for(uint32_t i=0;i<count;i++){
      const struct mesh_row_map *m=&maps[i];
      uint64_t end=(uint64_t)m->first+(uint64_t)(f->rows-1)*m->stride+m->count;
      if((side && (!m->count || end>p->count || end>UINT32_MAX)) || (m->immutable && (!side || f->inputs))) return EINVAL;
      for(uint32_t row=0;row<f->rows;row++){
        const struct mesh_row_range range=m->ranges[row];
        if(!range.count || (uint64_t)range.first+range.count>p->count ||
          (side && (range.first!=(uint64_t)m->first+(uint64_t)row*m->stride || range.count!=m->count))) return EINVAL;
      }
      if(side && f->rows>1 && m->stride<m->count) return EINVAL;
      if(side && (m->physical<p->memory->pool ||
         (uint64_t)m->physical+(uint64_t)(f->rows-1)*m->physical_stride+m->count>(uint64_t)p->memory->pool+p->memory->arena ||
         (f->rows>1 && m->physical_stride<m->count && !m->immutable))) return EINVAL;
      if(side) for(uint32_t j=0;j<i;j++)
        if(row_maps_overlap(*m,f->rows,maps[j],f->rows,0) || (!(m->immutable && maps[j].immutable) && row_maps_overlap(*m,f->rows,maps[j],f->rows,1))) return EINVAL;
    }
  }
  return 0;
}

// ../design/algorithm-sources.md#complete-page-ownership
static void row_count_delta(uint64_t *delta,struct mesh_row_range range){
  delta[range.first]++;
  delta[(uint64_t)range.first+range.count]--;
}

// ../design/algorithm-sources.md#literal-row-functions
int mesh_rows_realize(const struct mesh_rows *p, const struct mesh_row_function *functions,
  size_t count, struct mesh_row_binding *bindings, size_t binding_count,
  struct mesh_row_map *returns, size_t return_count){
  if(!p || p->count>UINT32_MAX || !count || !functions || (binding_count && !bindings) || (return_count && !returns)) return EINVAL;
  uint64_t *delta=NULL;
  if(row_layout(p,p->bytes,p->count,functions,count,bindings,binding_count,returns,return_count,&delta)==SIZE_MAX) return errno;
  for(size_t i=0;i<count;i++){
    const struct mesh_row_function *f=&functions[i];
    int error=mesh_rows_validate(p,f);
    if(error) return error;
    for(size_t j=0;j<i;j++) for(uint32_t a=0;a<f->outputs;a++) for(uint32_t b=0;b<functions[j].outputs;b++)
      if(row_maps_overlap(f->output[a],f->rows,functions[j].output[b],functions[j].rows,0) ||
         (!(f->output[a].immutable && functions[j].output[b].immutable) &&
          row_maps_overlap(f->output[a],f->rows,functions[j].output[b],functions[j].rows,1))) return EINVAL;
  }
  for(size_t i=0;i<binding_count;i++){
    const struct mesh_row_binding *b=&bindings[i];
    if((uint64_t)b->first+b->count>p->count || b->receive>1) return EINVAL;
    if(!b->count) continue;
    if(b->receive) for(size_t j=0;j<i;j++) if(bindings[j].receive && bindings[j].count &&
      (row_maps_overlap((struct mesh_row_map){.first=b->first,.count=b->count,.ranges=&(struct mesh_row_range){b->first,b->count}},1,
         (struct mesh_row_map){.first=bindings[j].first,.count=bindings[j].count,.ranges=&(struct mesh_row_range){bindings[j].first,bindings[j].count}},1,0))) return EINVAL;
    if(b->receive) for(size_t j=0;j<count;j++) for(uint32_t k=0;k<functions[j].outputs;k++)
      if(row_maps_overlap((struct mesh_row_map){.first=b->first,.count=b->count,.ranges=&(struct mesh_row_range){b->first,b->count}},1,
          functions[j].output[k],functions[j].rows,0)) return EINVAL;
    if(!b->receive) for(uint32_t row=b->first;row<b->first+b->count;row++){
      int produced=0;
      for(size_t j=0;j<count && !produced;j++) for(uint32_t k=0;k<functions[j].outputs && !produced;k++){
        const struct mesh_row_map *m=&functions[j].output[k];
        if(row<m->first) continue;
        uint32_t at=row-m->first;
        if(m->stride?(at/m->stride<functions[j].rows && at%m->stride<m->count):at<m->count){
          produced=1;
        }
      }
      for(size_t j=0;j<binding_count && !produced;j++)
        produced=bindings[j].receive && row>=bindings[j].first &&
          (uint64_t)row<(uint64_t)bindings[j].first+bindings[j].count;
      if(!produced) return EINVAL;
    }
  }
  for(size_t i=0;i<return_count;i++) if(!returns[i].ranges[0].count ||
    (uint64_t)returns[i].ranges[0].first+returns[i].ranges[0].count>p->count) return EINVAL;
  for(size_t i=0;i<return_count;i++) for(size_t j=0;j<i;j++)
    if(row_maps_overlap(returns[i],1,returns[j],1,0)) return EINVAL;
  memset(delta,0,(p->count+1)*sizeof(uint64_t));
  for(size_t i=0;i<count;i++) for(uint32_t j=0;j<functions[i].inputs;j++)
    for(uint32_t row=0;row<functions[i].rows;row++) row_count_delta(delta,functions[i].input[j].ranges[row]);
  for(size_t i=0;i<binding_count;i++){
    const struct mesh_row_binding *b=&bindings[i];
    if(!b->receive) row_count_delta(delta,(struct mesh_row_range){b->first,b->count});
  }
  for(size_t i=0;i<return_count;i++) row_count_delta(delta,returns[i].ranges[0]);
  uint64_t uses=0;
  for(uint32_t row=0;row<p->count;row++){
    uses+=delta[row];
    int produced=p->table[row].page!=MESH_ROW_ABSENT;
    for(size_t i=0;i<count && !produced;i++) for(uint32_t j=0;j<functions[i].outputs && !produced;j++){
      const struct mesh_row_map *map=&functions[i].output[j];
      if(row<map->first) continue;
      uint32_t offset=row-map->first;
      produced=map->stride?(offset/map->stride<functions[i].rows && offset%map->stride<map->count):offset<map->count;
    }
    for(size_t i=0;i<binding_count;i++){
      const struct mesh_row_binding *b=&bindings[i];
      produced|=b->receive && row>=b->first && (uint64_t)row<b->first+(uint64_t)b->count;
    }
    if((uses && !produced) || uses>UINT32_MAX) return EINVAL;
    p->table[row].uses=(uint32_t)uses;
  }
  for(size_t i=0;i<count;i++) for(uint32_t j=0;j<functions[i].outputs;j++){
    struct mesh_row_map *m=&functions[i].output[j];
    uint32_t *uses=(uint32_t*)m->uses;
    for(uint32_t row=0;row<functions[i].rows;row++) for(uint32_t k=0;k<m->count;k++){
      uses[(size_t)row*m->count+k]=p->table[m->first+row*m->stride+k].uses;
      if(m->immutable && !uses[(size_t)row*m->count+k]) return EINVAL;
    }
    m->uses=uses;
  }
  for(size_t i=0;i<binding_count;i++){
    struct mesh_row_binding *b=&bindings[i];
    if(!b->count) continue;
    uint32_t *uses=(uint32_t*)b->uses;
    for(uint32_t j=0;j<b->count;j++) uses[j]=p->table[b->first+j].uses;
    b->uses=uses;
  }
  for(size_t i=0;i<count;i++){
    const struct mesh_row_function *f=&functions[i];
    for(uint32_t j=0;j<f->outputs;j++){
      const struct mesh_row_map *m=&f->output[j];
      for(uint32_t r=0;r<f->rows;r++) for(uint32_t k=0;k<m->count;k++){
        uint32_t physical=m->physical+r*m->physical_stride+k;
        if(m->immutable) p->table[m->first+r*m->stride+k]=(struct mesh_row){.page=physical,.uses=1};
        else {
          memset(mesh_at(p->memory,physical)+p->offset,0,p->bytes);
          p->table[m->first+r*m->stride+k]=(struct mesh_row){.page=MESH_ROW_ABSENT};
        }
      }
    }
  }
  for(size_t i=0;i<binding_count;i++){
    struct mesh_row_binding *b=&bindings[i];
    if(b->receive || !b->count) continue;
    struct mesh_send *records=(void*)mesh_at(p->memory,b->headers);
    for(uint32_t j=0;j<b->count;j++) records[j]=(struct mesh_send){
      .header={.table=b->remote_table,.source=b->first+j,.target=b->remote,.index=j,.peer=b->peer},
      .owner=p->identity};
  }
  for(size_t i=0;i<count;i++){
    const struct mesh_row_map *indices=&functions[i].indices;
    for(uint32_t j=0;j<indices->count;j++){
      memset(mesh_at(p->memory,indices->physical+j),0,p->bytes);
      p->table[indices->first+j]=(struct mesh_row){.page=MESH_ROW_ABSENT};
    }
  }
  for(size_t i=0;i<count;i++) for(uint32_t j=0;j<functions[i].inputs;j++){
    struct mesh_row_map *map=&functions[i].input[j];
    map->stride=0;
    for(uint32_t k=1;k<functions[i].rows;k++)
      map->stride|=(map->ranges[k].first!=map->ranges[0].first || map->ranges[k].count!=map->ranges[0].count);
  }
  for(size_t i=0;i<count;i++) for(uint32_t j=0;j<functions[i].inputs;j++)
    ((struct mesh_row_function*)p->functions)[i].input[j].stride=functions[i].input[j].stride;
  struct mesh_row_map *held=(struct mesh_row_map*)p->returns;
  for(size_t i=1;i<p->return_count;i++){
    struct mesh_row_map value=held[i]; size_t j=i;
    while(j && held[j-1].ranges[0].first>value.ranges[0].first){ held[j]=held[j-1]; j--; }
    held[j]=value;
  }
  return 0;
}

// ../design/algorithm-sources.md#literal-row-functions
void *mesh_row_data(const struct mesh_rows *p, uint32_t row){
  uint32_t page=__atomic_load_n(&p->table[row].page,__ATOMIC_ACQUIRE);
  return page==MESH_ROW_ABSENT?NULL:mesh_at(p->memory,page)+p->offset;
}

// ../design/algorithm-sources.md#literal-row-functions
int mesh_rows_present(const struct mesh_rows *p, struct mesh_row_map map, uint32_t index, uint64_t stamp){
  struct mesh_row_range range=map.ranges[index];
  for(uint32_t i=0;i<range.count;i++){
    const struct mesh_row *r=&p->table[range.first+i];
    if(__atomic_load_n(&r->stamp,__ATOMIC_ACQUIRE)!=stamp ||
       __atomic_load_n(&r->page,__ATOMIC_ACQUIRE)==MESH_ROW_ABSENT) return 0;
  }
  return 1;
}

// ../design/algorithm-sources.md#literal-row-functions
static size_t mesh_rows_select(const struct mesh_rows *p, const struct mesh_row_function *f,
  uint64_t stamp, uint32_t *indices, size_t capacity){
  if(!stamp || stamp>=MESH_ROW_WRITING) return 0;
  int common_checked=0;
  size_t selected=0;
  for(uint32_t index=0;index<f->rows && selected<capacity;index++){
    const struct mesh_row_map *first=f->output;
    if(__atomic_load_n(&p->table[first->first+index*first->stride].stamp,__ATOMIC_ACQUIRE)>=stamp) continue;
    int ready=1;
    for(uint32_t i=0;i<f->inputs && ready;i++)
      if(f->input[i].stride) ready=mesh_rows_present(p,f->input[i],index,stamp);
    for(uint32_t i=0;i<f->outputs && ready;i++){
      const struct mesh_row_map *m=&f->output[i];
      for(uint32_t j=0;j<m->count && ready;j++){
        const struct mesh_row *r=&p->table[m->first+index*m->stride+j];
        ready=__atomic_load_n(&r->stamp,__ATOMIC_ACQUIRE)<stamp &&
          (m->immutable?(__atomic_load_n(&r->uses,__ATOMIC_ACQUIRE)<=1 &&
           __atomic_load_n(&r->page,__ATOMIC_ACQUIRE)!=MESH_ROW_ABSENT):
           (__atomic_load_n(&r->page,__ATOMIC_ACQUIRE)==MESH_ROW_ABSENT &&
            __atomic_load_n(&r->uses,__ATOMIC_ACQUIRE)==0));
      }
    }
    if(!ready) continue;
    if(!common_checked){
      for(uint32_t i=0;i<f->inputs;i++)
        if(!f->input[i].stride && !mesh_rows_present(p,f->input[i],0,stamp)) return 0;
      common_checked=1;
    }
    for(uint32_t i=0;i<f->outputs;i++){
      const struct mesh_row_map *m=&f->output[i];
      for(uint32_t j=0;j<m->count;j++){
        struct mesh_row *r=&p->table[m->first+index*m->stride+j];
        __atomic_store_n(&r->stamp,MESH_ROW_WRITING|stamp,__ATOMIC_RELEASE);
        __atomic_store_n(&r->uses,m->uses[(size_t)index*m->count+j],__ATOMIC_RELAXED);
        if(!m->immutable) __atomic_store_n(&r->page,m->physical+index*m->physical_stride+j,__ATOMIC_RELEASE);
      }
    }
    indices[selected++]=index;
  }
  return selected;
}

// ../design/algorithm-sources.md#literal-row-functions
void mesh_rows_publish(const struct mesh_rows *p, const struct mesh_row_function *f, uint32_t index, uint64_t stamp){
  for(uint32_t i=0;i<f->outputs;i++){
    const struct mesh_row_map *m=&f->output[i];
    for(uint32_t j=0;j<m->count;j++)
      __atomic_store_n(&p->table[m->first+index*m->stride+j].stamp,stamp,__ATOMIC_RELEASE);
  }
  for(uint32_t i=0;i<f->inputs;i++){
    const struct mesh_row_map *m=&f->input[i];
    struct mesh_row_range range=m->ranges[index];
    for(uint32_t j=0;j<range.count;j++) mesh_row_release(p,range.first+j);
  }
}

static int mesh_rows_return(const struct mesh_rows *p,uint32_t row,uint64_t stamp);

// ../design/algorithm-sources.md#complete-page-ownership
uint32_t mesh_rows_issue(const struct mesh_rows *p,const struct mesh_row_function *f,
  uint64_t stamp){
  struct mesh_row_map indices=f->indices;
  uint32_t width=(uint32_t)((((uint64_t)f->rows+1)*sizeof(uint32_t)+p->bytes-1)/p->bytes);
  for(uint32_t i=0;i+width<=indices.count;i+=width){
    struct mesh_row *row=&p->table[indices.first+i];
    int available=1;
    for(uint32_t j=0;j<width;j++)
      if(__atomic_load_n(&row[j].page,__ATOMIC_ACQUIRE)!=MESH_ROW_ABSENT ||
        __atomic_load_n(&row[j].stamp,__ATOMIC_ACQUIRE)>=MESH_ROW_WRITING) available=0;
    if(!available) continue;
    uint32_t physical=indices.physical+i;
    uint32_t *values=(void*)mesh_at(p->memory,physical);
    size_t count=mesh_rows_select(p,f,stamp,values+1,f->rows);
    if(!count) return MESH_ROW_ABSENT;
    values[0]=(uint32_t)count;
    for(uint32_t j=0;j<width;j++){
      __atomic_store_n(&row[j].uses,1,__ATOMIC_RELAXED);
      __atomic_store_n(&row[j].page,physical+j,__ATOMIC_RELEASE);
      __atomic_store_n(&row[j].stamp,stamp,__ATOMIC_RELEASE);
    }
    return indices.first+i;
  }
  return MESH_ROW_ABSENT;
}

// ../design/algorithm-sources.md#complete-page-ownership
void mesh_rows_complete(const struct mesh_rows *p,const struct mesh_row_function *f,
  uint64_t stamp,uint32_t indices){
  const uint32_t *values=mesh_row_data(p,indices);
  for(uint32_t i=0;i<values[0];i++) for(uint32_t j=0;j<f->outputs;j++){
    const struct mesh_row_map *map=&f->output[j];
    for(uint32_t k=0;k<map->count;k++)
      __atomic_store_n(&p->table[map->first+values[i+1]*map->stride+k].stamp,stamp,__ATOMIC_RELEASE);
  }
  for(uint32_t i=0;i<f->inputs;i++){
    const struct mesh_row_map *map=&f->input[i];
    if(!map->stride){
      struct mesh_row_range range=map->ranges[0];
      for(uint32_t j=0;j<range.count;j++) __atomic_fetch_sub(&p->table[range.first+j].uses,values[0],__ATOMIC_ACQ_REL);
    }else for(uint32_t j=0;j<values[0];j++){
      struct mesh_row_range range=map->ranges[values[j+1]];
      for(uint32_t k=0;k<range.count;k++) mesh_row_release(p,range.first+k);
    }
  }
  uint32_t width=(uint32_t)((((uint64_t)f->rows+1)*sizeof(uint32_t)+p->bytes-1)/p->bytes);
  for(uint32_t i=0;i<width;i++){
    mesh_row_release(p,indices+i);
  }
}

// ../design/algorithm-sources.md#asynchronous-metadata-publication
void mesh_rows_report(const struct mesh_rows *p, struct mesh_row_map output,
  uint32_t occurrence, struct mesh_row_metadata metadata){
  uint32_t page=output.physical+occurrence*output.physical_stride;
  memcpy(mesh_at(p->memory,page)+p->offset,&metadata,sizeof metadata);
}

// ../design/algorithm-sources.md#literal-row-functions
void mesh_row_release(const struct mesh_rows *p, uint32_t row){
  __atomic_fetch_sub(&p->table[row].uses,1,__ATOMIC_ACQ_REL);
}

// ../design/algorithm-sources.md#complete-page-ownership
static size_t mesh_rows_send(const struct mesh_rows *p,
  const struct mesh_row_binding *bindings,size_t count){
  size_t sent=0;
  for(size_t i=0;i<count;i++){
    const struct mesh_row_binding *b=&bindings[i];
    if(b->receive) continue;
    for(uint32_t j=0;j<b->count;j++){
      const struct mesh_row *row=&p->table[b->first+j];
      uint64_t stamp=__atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE);
      uint32_t page=__atomic_load_n(&row->page,__ATOMIC_ACQUIRE);
      if(!stamp || stamp>=MESH_ROW_WRITING || page==MESH_ROW_ABSENT) continue;
      struct mesh_send *record=(struct mesh_send*)mesh_at(p->memory,b->headers)+j;
      struct mesh_page_header *header=&record->header;
      if(header->stamp==stamp){
        mesh_rows_return(p,b->first+j,stamp);
        continue;
      }
      struct ring *ring=&p->memory->r[SUB];
      uint64_t head=atomic_load_explicit(&ring->head,memory_order_relaxed);
      if(head-atomic_load_explicit(&ring->tail,memory_order_acquire)==MESH_RING) return sent;
      header->stamp=stamp; record->page=page;
      *slot(p->memory,SUB,head)=(struct desc){.page=page,.bytes=p->bytes,.node=b->peer,
        .header=(uint64_t)((unsigned char*)record-(unsigned char*)p->memory)};
      atomic_store_explicit(&ring->head,head+1,memory_order_release);
      sent++;
    }
  }
  return sent;
}

// ../design/algorithm-sources.md#context-lifetime
static int mesh_release_page(struct hdr *memory,struct mesh_row *value,uint32_t page){
  if(page<memory->pool){
    struct ring *ring=&memory->r[REL];
    uint64_t head=atomic_load_explicit(&ring->head,memory_order_relaxed);
    if(head-atomic_load_explicit(&ring->tail,memory_order_acquire)>=MESH_RING) return -1;
    __atomic_store_n(&value->page,MESH_ROW_ABSENT,__ATOMIC_RELEASE);
    *slot(memory,REL,head)=(struct desc){.page=page};
    atomic_store_explicit(&ring->head,head+1,memory_order_release);
  }else __atomic_store_n(&value->page,MESH_ROW_ABSENT,__ATOMIC_RELEASE);
  return 1;
}

// ../design/algorithm-sources.md#complete-page-ownership
static int mesh_rows_return(const struct mesh_rows *p,uint32_t row,uint64_t stamp){
  struct mesh_row *value=&p->table[row];
  uint32_t page=__atomic_load_n(&value->page,__ATOMIC_ACQUIRE);
  if(page==MESH_ROW_ABSENT || !stamp || stamp>=MESH_ROW_WRITING ||
    __atomic_load_n(&value->uses,__ATOMIC_ACQUIRE)) return 0;
  return mesh_release_page(p->memory,value,page);
}

// ../design/algorithm-sources.md#complete-page-ownership
static int mesh_rows_retire_range(const struct mesh_rows *p,struct mesh_row_range range){
  int available=1;
  for(uint32_t i=0;i<range.count;i++){
    uint32_t row=range.first+i;
    if(mesh_rows_return(p,row,__atomic_load_n(&p->table[row].stamp,__ATOMIC_ACQUIRE))<0) available=0;
  }
  return available;
}

// ../design/algorithm-sources.md#complete-page-ownership
static void mesh_rows_retire(const struct mesh_rows *p){
  for(size_t i=0;i<p->function_count;i++){
    const struct mesh_row_function *f=&p->functions[i];
    uint32_t width=(uint32_t)((((uint64_t)f->rows+1)*sizeof(uint32_t)+p->bytes-1)/p->bytes);
    for(uint32_t first=0;first+width<=f->indices.count;first+=width){
      uint32_t logical=f->indices.first+first;
      const struct mesh_row *row=&p->table[logical];
      uint64_t stamp=__atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE);
      // first_fit(i) => issued(j>i) => stamp(i)!=0
      if(!stamp) break;
      if(__atomic_load_n(&row->uses,__ATOMIC_ACQUIRE) ||
        __atomic_load_n(&row->page,__ATOMIC_ACQUIRE)==MESH_ROW_ABSENT || stamp>=MESH_ROW_WRITING) continue;
      uint32_t *indices=mesh_row_data(p,logical);
      int available=1;
      for(uint32_t side=0;indices[0] && side<2;side++){
        const struct mesh_row_map *maps=side?f->output:f->input;
        uint32_t count=side?f->outputs:f->inputs;
        for(uint32_t j=0;j<count;j++){
          if(maps[j].immutable) continue;
          uint32_t selected=(!side && !maps[j].stride)?1:indices[0];
          for(uint32_t k=0;k<selected;k++)
            available&=mesh_rows_retire_range(p,maps[j].ranges[indices[k+1]]);
        }
      }
      for(uint32_t j=0;j<width;j++) if(__atomic_load_n(&row[j].uses,__ATOMIC_ACQUIRE)) available=0;
      if(available){
        indices[0]=0;
        for(uint32_t j=width;j>0;j--)
          if(mesh_rows_return(p,logical+j-1,__atomic_load_n(&row[j-1].stamp,__ATOMIC_ACQUIRE))<0) break;
      }
    }
  }
  for(size_t i=0;i<p->return_count;i++) if(!p->returns[i].immutable)
    mesh_rows_retire_range(p,p->returns[i].ranges[0]);
}

// ../design/algorithm-sources.md#complete-page-ownership
size_t mesh_rows_poll(struct mesh_ctx *context){
  struct hdr *memory=context->M;
  size_t changed=0;
  struct desc completion;
  struct ring *ack=&memory->r[ACK];
  uint64_t acknowledged=atomic_load_explicit(&ack->tail,memory_order_relaxed);
  uint64_t completed=atomic_load_explicit(&ack->head,memory_order_acquire);
  while(acknowledged<completed){
    completion=*slot(memory,ACK,acknowledged);
    struct mesh_send *record=(void*)((unsigned char*)memory+completion.header);
    struct mesh_page_header *header=&record->header;
    const struct mesh_rows *p=context->tables[record->owner];
    header->code=completion.error; header->domain=completion.domain;
    mesh_row_release(p,header->source);
    atomic_store_explicit(&ack->tail,++acknowledged,memory_order_release);
    changed++;
  }
  struct ring *ring=&memory->r[CMP];
  uint64_t tail=atomic_load_explicit(&ring->tail,memory_order_relaxed);
  uint64_t head=atomic_load_explicit(&ring->head,memory_order_acquire);
  for(uint64_t at=tail;at<head;at++){
    completion=*slot(memory,CMP,at);
    if(completion.error || !context->table_count){
      struct mesh_page_header *header=mesh_header(memory,completion.page);
      if(completion.error){ header->code=completion.error; header->domain=completion.domain; }
      struct mesh_row *error=mesh_context_row(memory,completion.page);
      *error=(struct mesh_row){completion.page,1,1};
      ring_erase(ring,slot(memory,CMP,0),sizeof(struct desc),MESH_RING,at);
      changed++;
      continue;
    }
    const struct mesh_page_header *header=mesh_header(memory,completion.page);
    const struct mesh_rows *p=context->tables[header->table];
    const struct mesh_row_binding *binding=&p->bindings[header->target];
    uint32_t index=header->index;
    if(!binding->uses[index]){
      *mesh_context_row(memory,completion.page)=(struct mesh_row){completion.page,1,1};
      ring_erase(ring,slot(memory,CMP,0),sizeof(struct desc),MESH_RING,at);
      changed++;
      continue;
    }
    struct mesh_row *row=&p->table[binding->first+index];
    if(__atomic_load_n(&row->page,__ATOMIC_ACQUIRE)!=MESH_ROW_ABSENT ||
      __atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE)>=MESH_ROW_WRITING) continue;
    __atomic_store_n(&row->uses,binding->uses[index],__ATOMIC_RELAXED);
    __atomic_store_n(&row->page,completion.page,__ATOMIC_RELEASE);
    __atomic_store_n(&row->stamp,header->stamp,__ATOMIC_RELEASE);
    ring_erase(ring,slot(memory,CMP,0),sizeof(struct desc),MESH_RING,at);
    changed++;
  }
  for(size_t i=0;i<context->table_count;i++){
    const struct mesh_rows *p=context->tables[i];
    changed+=mesh_rows_send(p,p->bindings,p->binding_count);
    mesh_rows_retire(p);
  }
  return changed;
}

// ../design/algorithm-sources.md#context-lifetime
const struct mesh_page_header *mesh_context_metadata(struct mesh_ctx *c,uint32_t page){
  return __atomic_load_n(&mesh_context_row(c->M,page)->uses,__ATOMIC_ACQUIRE)?mesh_header(c->M,page):NULL;
}

// ../design/algorithm-sources.md#context-lifetime
int mesh_context_consume(struct mesh_ctx *c,uint32_t page){
  struct mesh_row *row=mesh_context_row(c->M,page);
  __atomic_fetch_sub(&row->uses,1,__ATOMIC_ACQ_REL);
  if(mesh_release_page(c->M,row,page)<0){
    __atomic_fetch_add(&row->uses,1,__ATOMIC_RELEASE);
    return EBUSY;
  }
  return 0;
}

// ../design/algorithm-sources.md#context-lifetime
static uint32_t mesh_rows_held(const struct mesh_rows *p,uint32_t row){
  size_t first=0,last=p->return_count;
  while(first<last){
    size_t middle=first+(last-first)/2;
    if(p->returns[middle].ranges[0].first<=row) first=middle+1; else last=middle;
  }
  if(!first) return 0;
  struct mesh_row_range range=p->returns[first-1].ranges[0];
  return row-range.first<range.count;
}

// ../design/algorithm-sources.md#context-lifetime
int mesh_rows_close(struct mesh_ctx *c){
  mesh_rows_poll(c);
  for(size_t i=0;i<c->table_count;i++){
    const struct mesh_rows *p=c->tables[i];
    for(size_t j=0;j<p->function_count;j++){
      const struct mesh_row_function *f=&p->functions[j];
      for(uint32_t k=0;k<f->indices.count;k++)
        if(__atomic_load_n(&p->table[f->indices.first+k].uses,__ATOMIC_ACQUIRE)) return EBUSY;
      for(uint32_t k=0;k<f->outputs;k++) for(uint32_t index=0;index<f->rows;index++)
        for(uint32_t n=0;n<f->output[k].count;n++){
          uint32_t logical=f->output[k].first+index*f->output[k].stride+n,held=0;
          const struct mesh_row *row=&p->table[logical];
          held=mesh_rows_held(p,logical);
          if(__atomic_load_n(&row->uses,__ATOMIC_ACQUIRE)>held ||
            (__atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE)>=MESH_ROW_WRITING &&
             __atomic_load_n(&row->page,__ATOMIC_ACQUIRE)!=MESH_ROW_ABSENT)) return EBUSY;
        }
    }
  }
  for(size_t i=0;i<c->table_count;i++){
    const struct mesh_rows *p=c->tables[i];
    for(size_t j=0;j<p->binding_count;j++) if(p->bindings[j].receive)
      for(uint32_t k=0;k<p->bindings[j].count;k++){
        uint32_t logical=p->bindings[j].first+k,held=0;
        held=mesh_rows_held(p,logical);
        if(__atomic_load_n(&p->table[logical].uses,__ATOMIC_ACQUIRE)>held) return EBUSY;
      }
  }
  for(uint32_t i=0;i<c->M->pool;i++)
    if(__atomic_load_n(&mesh_context_row(c->M,i)->uses,__ATOMIC_ACQUIRE)) return EBUSY;
  for(unsigned i=0;i<NRING;i++) if(
    atomic_load_explicit(&c->M->r[i].head,memory_order_acquire)!=
    atomic_load_explicit(&c->M->r[i].tail,memory_order_acquire)) return EBUSY;
  if(c->table_count){
    for(size_t i=0;i<c->table_count;i++){
      const struct mesh_rows *p=c->tables[i];
      for(size_t j=0;j<p->count;j++){
        uint32_t page=__atomic_load_n(&p->table[j].page,__ATOMIC_ACQUIRE);
        if(page<c->M->pool) *mesh_context_row(c->M,page)=(struct mesh_row){page,0,1};
      }
    }
    for(size_t i=0;i<c->table_count;i++) free(c->tables[i]);
    free(c->tables); c->tables=NULL; c->table_count=0;
  }
  int pending=0;
  for(uint32_t i=0;i<c->M->pool;i++){
    struct mesh_row *row=mesh_context_row(c->M,i);
    uint64_t stamp=__atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE);
    uint32_t page=__atomic_load_n(&row->page,__ATOMIC_ACQUIRE);
    if(stamp && page!=MESH_ROW_ABSENT){
      if(mesh_release_page(c->M,row,page)<0){ pending=1; break; }
    }
  }
  return pending?EBUSY:0;
}
