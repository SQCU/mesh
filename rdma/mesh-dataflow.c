#include "mesh-dataflow.h"
#include <errno.h>
#include <math.h>

typedef _Float16 mesh_half8 __attribute__((ext_vector_type(8)));
typedef float mesh_float8 __attribute__((ext_vector_type(8)));

// ../design/algorithm-sources.md#literal-row-functions
static int row_maps_overlap(struct mesh_row_map a, uint32_t na, struct mesh_row_map b, uint32_t nb, int physical){
  uint64_t first_a=physical?a.physical:a.first, first_b=physical?b.physical:b.first;
  uint64_t stride_a=physical?a.physical_stride:a.stride, stride_b=physical?b.physical_stride:b.stride;
  if(first_a+(na-1)*stride_a+a.count<=first_b || first_b+(nb-1)*stride_b+b.count<=first_a) return 0;
  uint32_t i=0,j=0;
  while(i<na && j<nb){
    uint64_t x=first_a+i*stride_a, y=first_b+j*stride_b;
    if(x<y+b.count && y<x+a.count) return 1;
    if(x<y) i++; else j++;
  }
  return 0;
}

// ../design/algorithm-sources.md#literal-row-functions
int mesh_rows_validate(const struct mesh_rows *p, const struct mesh_row_function *f){
  if(!p || !p->memory || !p->table || !p->bytes || p->offset<sizeof(struct wire)+sizeof(struct mesh_row_address) ||
     p->offset>p->memory->pgsz || p->bytes>p->memory->pgsz-p->offset ||
     p->offset%8 || p->bytes%8 || !f || !f->rows || !f->outputs || !f->output || (f->inputs && !f->input)) return EINVAL;
  for(uint32_t side=0;side<2;side++){
    const struct mesh_row_map *maps=side?f->output:f->input;
    uint32_t count=side?f->outputs:f->inputs;
    for(uint32_t i=0;i<count;i++){
      const struct mesh_row_map *m=&maps[i];
      uint64_t end=(uint64_t)m->first+(uint64_t)(f->rows-1)*m->stride+m->count;
      if(!m->count || end>p->count || end>UINT32_MAX) return EINVAL;
      if(side && f->rows>1 && m->stride<m->count) return EINVAL;
      if(side && (m->physical<p->memory->pool ||
         (uint64_t)m->physical+(uint64_t)(f->rows-1)*m->physical_stride+m->count>(uint64_t)p->memory->pool+p->memory->arena ||
         (f->rows>1 && m->physical_stride<m->count))) return EINVAL;
      if(side) for(uint32_t j=0;j<i;j++)
        if(row_maps_overlap(*m,f->rows,maps[j],f->rows,0) || row_maps_overlap(*m,f->rows,maps[j],f->rows,1)) return EINVAL;
    }
  }
  return 0;
}

// ../design/algorithm-sources.md#literal-row-functions
static uint64_t row_map_uses(struct mesh_row_map map, uint32_t rows, uint32_t row){
  if(!rows || row<map.first) return 0;
  uint64_t offset=(uint64_t)row-map.first;
  if(!map.stride) return offset<map.count?rows:0;
  uint64_t first=offset<map.count?0:(offset-map.count)/map.stride+1;
  uint64_t last=offset/map.stride;
  if(last>=rows) last=rows-1;
  return first<=last?last-first+1:0;
}

// ../design/algorithm-sources.md#literal-row-functions
uint64_t mesh_rows_uses(const struct mesh_row_function *functions, size_t count,
  const struct mesh_row_binding *bindings, size_t binding_count,
  const struct mesh_row_map *returns, size_t return_count, uint32_t row){
  uint64_t uses=0;
  for(size_t i=0;i<count;i++) for(uint32_t j=0;j<functions[i].inputs;j++)
    uses+=row_map_uses(functions[i].input[j],functions[i].rows,row);
  for(size_t i=0;i<binding_count;i++){
    const struct mesh_row_binding *b=&bindings[i];
    if(!b->receive) uses+=(row>=b->first && (uint64_t)row<b->first+(uint64_t)b->count);
    for(uint32_t j=0;j<b->inputs;j++) uses+=row_map_uses(b->input[j],b->count,row);
  }
  for(size_t i=0;i<return_count;i++) uses+=row_map_uses(returns[i],1,row);
  return uses;
}

// ../design/algorithm-sources.md#literal-row-functions
int mesh_rows_realize(const struct mesh_rows *p, const struct mesh_row_function *functions,
  size_t count, const struct mesh_row_binding *bindings, size_t binding_count,
  const struct mesh_row_map *returns, size_t return_count){
  if(!p || p->count>UINT32_MAX || !count || !functions || (binding_count && !bindings) || (return_count && !returns)) return EINVAL;
  for(size_t i=0;i<count;i++){
    const struct mesh_row_function *f=&functions[i];
    int error=mesh_rows_validate(p,f);
    if(error) return error;
    for(size_t j=0;j<i;j++) for(uint32_t a=0;a<f->outputs;a++) for(uint32_t b=0;b<functions[j].outputs;b++)
      if(row_maps_overlap(f->output[a],f->rows,functions[j].output[b],functions[j].rows,0) ||
         row_maps_overlap(f->output[a],f->rows,functions[j].output[b],functions[j].rows,1)) return EINVAL;
  }
  for(size_t i=0;i<binding_count;i++){
    const struct mesh_row_binding *b=&bindings[i];
    if(!b->count || (uint64_t)b->first+b->count>p->count || (uint64_t)b->remote+b->count>UINT32_MAX || b->receive>1) return EINVAL;
    if(b->inputs && (!b->receive || !b->input)) return EINVAL;
    for(uint32_t j=0;j<b->inputs;j++){
      const struct mesh_row_map *m=&b->input[j];
      uint64_t end=(uint64_t)m->first+(uint64_t)(b->count-1)*m->stride+m->count;
      if(!m->count || end>p->count || end>UINT32_MAX) return EINVAL;
    }
    if(i && (uint64_t)bindings[i-1].first+bindings[i-1].count>b->first) return EINVAL;
    if(b->receive) for(size_t j=0;j<count;j++) for(uint32_t k=0;k<functions[j].outputs;k++)
      if(row_maps_overlap((struct mesh_row_map){.first=b->first,.count=b->count},1,
          functions[j].output[k],functions[j].rows,0)) return EINVAL;
    if(!b->receive) for(uint32_t row=b->first;row<b->first+b->count;row++){
      int produced=0;
      for(size_t j=0;j<count && !produced;j++) for(uint32_t k=0;k<functions[j].outputs && !produced;k++){
        const struct mesh_row_map *m=&functions[j].output[k];
        if(row<m->first) continue;
        uint32_t at=row-m->first;
        if(m->stride?(at/m->stride<functions[j].rows && at%m->stride<m->count):at<m->count){
          if(!m->uses) return EINVAL;
          produced=1;
        }
      }
      if(!produced) return EINVAL;
    }
  }
  for(size_t i=0;i<return_count;i++)
    if(!returns[i].count || (uint64_t)returns[i].first+returns[i].count>p->count ||
       (uint64_t)returns[i].first+returns[i].count>UINT32_MAX) return EINVAL;
  for(uint32_t row=0;row<p->count;row++){
    uint64_t uses=mesh_rows_uses(functions,count,bindings,binding_count,returns,return_count,row);
    uint32_t configured=0;
    int produced=0;
    for(size_t i=0;i<count;i++) for(uint32_t j=0;j<functions[i].outputs;j++){
      const struct mesh_row_map *m=&functions[i].output[j];
      if(row_map_uses(*m,functions[i].rows,row)){
        produced=1;
        configured=m->uses;
      }
    }
    for(size_t i=0;i<binding_count;i++){
      const struct mesh_row_binding *b=&bindings[i];
      if(b->receive && row>=b->first && (uint64_t)row<b->first+(uint64_t)b->count){
        produced=1;
        configured=b->uses;
      }
    }
    if((uses && !produced) || uses!=configured) return EINVAL;
  }
  for(size_t i=0;i<p->count;i++) p->table[i]=(struct mesh_row){.page=MESH_ROW_ABSENT};
  for(size_t i=0;i<count;i++){
    const struct mesh_row_function *f=&functions[i];
    for(uint32_t j=0;j<f->outputs;j++){
      const struct mesh_row_map *m=&f->output[j];
      for(uint32_t r=0;r<f->rows;r++) for(uint32_t k=0;k<m->count;k++)
        memset(mesh_at(p->memory,m->physical+r*m->physical_stride+k)+sizeof(struct wire),0,
          p->offset+p->bytes-sizeof(struct wire));
    }
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
  for(uint32_t i=0;i<map.count;i++){
    const struct mesh_row *r=&p->table[map.first+index*map.stride+i];
    if(__atomic_load_n(&r->stamp,__ATOMIC_ACQUIRE)!=stamp ||
       __atomic_load_n(&r->page,__ATOMIC_ACQUIRE)==MESH_ROW_ABSENT) return 0;
  }
  return 1;
}

// ../design/algorithm-sources.md#literal-row-functions
size_t mesh_rows_select(const struct mesh_rows *p, const struct mesh_row_function *f,
  uint64_t stamp, uint32_t *indices, size_t capacity){
  if(!stamp || stamp>=MESH_ROW_WRITING) return 0;
  size_t selected=0;
  for(uint32_t index=0;index<f->rows && selected<capacity;index++){
    const struct mesh_row_map *first=f->output;
    if(__atomic_load_n(&p->table[first->first+index*first->stride].stamp,__ATOMIC_ACQUIRE)>=stamp) continue;
    int ready=1;
    for(uint32_t i=0;i<f->inputs && ready;i++) ready=mesh_rows_present(p,f->input[i],index,stamp);
    for(uint32_t i=0;i<f->outputs && ready;i++){
      const struct mesh_row_map *m=&f->output[i];
      for(uint32_t j=0;j<m->count && ready;j++){
        const struct mesh_row *r=&p->table[m->first+index*m->stride+j];
        ready=__atomic_load_n(&r->stamp,__ATOMIC_ACQUIRE)<stamp &&
          __atomic_load_n(&r->page,__ATOMIC_ACQUIRE)==MESH_ROW_ABSENT &&
          __atomic_load_n(&r->uses,__ATOMIC_ACQUIRE)==0;
      }
    }
    if(!ready) continue;
    for(uint32_t i=0;i<f->outputs;i++){
      const struct mesh_row_map *m=&f->output[i];
      for(uint32_t j=0;j<m->count;j++){
        struct mesh_row *r=&p->table[m->first+index*m->stride+j];
        __atomic_store_n(&r->stamp,MESH_ROW_WRITING|stamp,__ATOMIC_RELEASE);
        __atomic_store_n(&r->uses,m->uses,__ATOMIC_RELAXED);
        __atomic_store_n(&r->page,m->physical+index*m->physical_stride+j,__ATOMIC_RELEASE);
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
    for(uint32_t j=0;j<m->count;j++) mesh_row_release(p,m->first+index*m->stride+j);
  }
}

// ../design/algorithm-sources.md#asynchronous-metadata-publication
void mesh_rows_report(const struct mesh_rows *p, struct mesh_row_map output,
  uint32_t occurrence, struct mesh_row_metadata metadata){
  uint32_t page=output.physical+occurrence*output.physical_stride;
  struct mesh_row *row=&p->table[output.first+occurrence*output.stride];
  memcpy(mesh_at(p->memory,page)+p->offset,&metadata,sizeof metadata);
  __atomic_store_n(&row->uses,output.uses,__ATOMIC_RELAXED);
  __atomic_store_n(&row->page,page,__ATOMIC_RELEASE);
  __atomic_store_n(&row->stamp,metadata.stamp,__ATOMIC_RELEASE);
}

// ../design/algorithm-sources.md#literal-row-functions
void mesh_row_release(const struct mesh_rows *p, uint32_t row){
  __atomic_fetch_sub(&p->table[row].uses,1,__ATOMIC_ACQ_REL);
}

// ../design/algorithm-sources.md#literal-row-functions
int mesh_row_zero(const struct mesh_rows *p, uint32_t row, uint64_t stamp){
  struct mesh_row *r=&p->table[row];
  if(__atomic_load_n(&r->uses,__ATOMIC_ACQUIRE) ||
     __atomic_load_n(&r->page,__ATOMIC_ACQUIRE)==MESH_ROW_ABSENT) return 0;
  uint64_t expected=stamp;
  if(!stamp || stamp>=MESH_ROW_WRITING ||
     !__atomic_compare_exchange_n(&r->stamp,&expected,MESH_ROW_WRITING|stamp,0,__ATOMIC_ACQ_REL,__ATOMIC_RELAXED)) return 0;
  uint32_t page=__atomic_load_n(&r->page,__ATOMIC_ACQUIRE);
  if(page!=MESH_ROW_ABSENT) memset(mesh_at(p->memory,page)+p->offset,0,p->bytes);
  __atomic_store_n(&r->page,MESH_ROW_ABSENT,__ATOMIC_RELEASE);
  __atomic_store_n(&r->stamp,stamp,__ATOMIC_RELEASE);
  return 1;
}

// ../design/algorithm-sources.md#literal-page-reduction
static __attribute__((always_inline)) inline void row_add(const struct mesh_rows *p, const uint32_t *inputs, size_t count,
  const uint32_t *accumulators, size_t elements, uint32_t index_row, uint32_t index, uint64_t stamp, size_t bytes){
  uint64_t *indices=mesh_row_data(p,index_row);
  size_t width=p->bytes/sizeof(float), pages=(elements+width-1)/width;
  for(size_t page=0;page<pages;page++){
    size_t first=page*width, n=elements-first;
    if(n>width) n=width;
    float *output=mesh_row_data(p,accumulators[page]);
    for(size_t input=0;input<count;input++){
      const unsigned char *source=(const unsigned char*)mesh_row_data(p,inputs[input])+first*bytes;
      size_t i=0;
      for(;i+8<=n;i+=8){
        mesh_float8 sum;
        if(bytes==sizeof(_Float16)){
          mesh_half8 half;
          memcpy(&half,source+i*bytes,sizeof half);
          sum=__builtin_convertvector(half,mesh_float8);
        } else memcpy(&sum,source+i*bytes,sizeof sum);
        if(input){ mesh_float8 prior; memcpy(&prior,output+i,sizeof prior); sum+=prior; }
        memcpy(output+i,&sum,sizeof sum);
      }
      for(;i<n;i++) output[i]=(input?output[i]:0)+
        (bytes==sizeof(_Float16)?(float)((const _Float16*)source)[i]:((const float*)source)[i]);
    }
  }
  for(size_t i=0;i<pages;i++) __atomic_store_n(&p->table[accumulators[i]].stamp,stamp,__ATOMIC_RELEASE);
  __atomic_store_n(indices+index,stamp,__ATOMIC_RELEASE);
  for(size_t i=0;i<count;i++) mesh_row_release(p,inputs[i]);
}

// ../design/algorithm-sources.md#literal-page-reduction
void mesh_rows_add_f16(const struct mesh_rows *p, const uint32_t *inputs, size_t count,
  const uint32_t *accumulators, size_t elements, uint32_t index_row, uint32_t index, uint64_t stamp){
  row_add(p,inputs,count,accumulators,elements,index_row,index,stamp,sizeof(_Float16));
}

// ../design/algorithm-sources.md#literal-page-reduction
void mesh_rows_add_f32(const struct mesh_rows *p, const uint32_t *inputs, size_t count,
  const uint32_t *accumulators, size_t elements, uint32_t index_row, uint32_t index, uint64_t stamp){
  row_add(p,inputs,count,accumulators,elements,index_row,index,stamp,sizeof(float));
}

// ../design/algorithm-sources.md#literal-page-reduction
int mesh_rows_indexed(const struct mesh_rows *p, uint32_t index_row, uint32_t first, uint32_t count, uint64_t stamp){
  if(first>p->bytes/sizeof(uint64_t) || count>p->bytes/sizeof(uint64_t)-first) return 0;
  const uint64_t *indices=mesh_row_data(p,index_row);
  if(!indices) return 0;
  for(uint32_t i=0;i<count;i++) if(__atomic_load_n(indices+first+i,__ATOMIC_ACQUIRE)!=stamp) return 0;
  return 1;
}

// ../design/algorithm-sources.md#literal-page-reduction
void mesh_rows_normalize_f32(const struct mesh_rows *p, const uint32_t *accumulators,
  const uint32_t *gamma, const uint32_t *residual, const uint32_t *outputs,
  size_t elements, float epsilon, float scale){
  size_t floats=p->bytes/sizeof(float), halves=p->bytes/sizeof(_Float16);
  float squared=0;
  for(size_t i=0;i<elements;i++){
    float value=((const float*)mesh_row_data(p,accumulators[i/floats]))[i%floats];
    squared+=value*value;
  }
  float norm=1.0f/sqrtf(squared/(float)elements+epsilon);
  for(size_t i=0;i<elements;i++){
    float value=((const float*)mesh_row_data(p,accumulators[i/floats]))[i%floats];
    float weight=(float)((const _Float16*)mesh_row_data(p,gamma[i/halves]))[i%halves];
    float add=(float)((const _Float16*)mesh_row_data(p,residual[i/halves]))[i%halves];
    ((_Float16*)mesh_row_data(p,outputs[i/halves]))[i%halves]=(_Float16)((value*norm*weight+add)*scale);
  }
}

// ../design/algorithm-sources.md#literal-page-transport
size_t mesh_rows_send(const struct mesh_rows *p, uint64_t epoch,
  const struct mesh_row_binding *bindings, size_t count){
  size_t sent=0;
  for(size_t i=0;i<count;i++){
    const struct mesh_row_binding *b=&bindings[i];
    if(b->receive) continue;
    for(uint32_t j=0;j<b->count;j++){
      const struct mesh_row *r=&p->table[b->first+j];
      uint64_t stamp=__atomic_load_n(&r->stamp,__ATOMIC_ACQUIRE);
      uint32_t page=__atomic_load_n(&r->page,__ATOMIC_ACQUIRE);
      if(!stamp || stamp>=MESH_ROW_WRITING || page==MESH_ROW_ABSENT) continue;
      struct mesh_row_address *address=(void*)(mesh_at(p->memory,page)+sizeof(struct wire));
      struct mesh_row_address previous;
      memcpy(&previous,address,sizeof previous);
      if(previous.epoch==epoch && previous.stamp==stamp && previous.source==b->first+j) continue;
      struct mesh_row_address next={epoch,stamp,b->first+j,b->remote+j};
      memcpy(address,&next,sizeof next);
      struct desc d={page,p->offset+p->bytes-(uint32_t)sizeof(struct wire),b->peer};
      if(push(p->memory,SUB,&d)){
        memcpy(address,&previous,sizeof previous);
        return sent;
      }
      sent++;
    }
  }
  return sent;
}

// ../design/algorithm-sources.md#literal-page-transport
size_t mesh_rows_receive(const struct mesh_rows *p,
  const struct mesh_row_binding *bindings, size_t count){
  struct ring *ring=&p->memory->r[CMP];
  uint64_t tail=atomic_load_explicit(&ring->tail,memory_order_relaxed);
  if(tail==atomic_load_explicit(&ring->head,memory_order_acquire)) return 0;
  uint64_t head=atomic_load_explicit(&ring->head,memory_order_acquire);
  size_t received=0;
  for(uint64_t at=tail;at<head;at++){
    struct desc d=*slot(p->memory,CMP,at);
    struct mesh_row_address address;
    memcpy(&address,mesh_at(p->memory,d.page)+sizeof(struct wire),sizeof address);
    size_t first=0,last=count;
    while(first<last){
      size_t middle=first+(last-first)/2;
      if(bindings[middle].first<=address.target) first=middle+1; else last=middle;
    }
    const struct mesh_row_binding *matched=&bindings[first-1];
    struct mesh_row *r=&p->table[address.target];
    if(__atomic_load_n(&r->page,__ATOMIC_ACQUIRE)!=MESH_ROW_ABSENT) continue;
    uint32_t index=address.target-matched->first;
    __atomic_store_n(&r->uses,matched->uses,__ATOMIC_RELAXED);
    __atomic_store_n(&r->page,d.page,__ATOMIC_RELEASE);
    __atomic_store_n(&r->stamp,address.stamp,__ATOMIC_RELEASE);
    for(uint32_t i=0;i<matched->inputs;i++){
      const struct mesh_row_map *m=&matched->input[i];
      for(uint32_t j=0;j<m->count;j++) mesh_row_release(p,m->first+index*m->stride+j);
    }
    ring_erase(ring,slot(p->memory,CMP,0),sizeof(struct desc),MESH_RING,at);
    received++;
  }
  return received;
}

// ../design/algorithm-sources.md#literal-page-transport
size_t mesh_rows_acknowledge(const struct mesh_rows *p){
  size_t released=0;
  struct desc d;
  while(!pop(p->memory,ACK,&d)){
    struct mesh_row_address address;
    memcpy(&address,mesh_at(p->memory,d.page)+sizeof(struct wire),sizeof address);
    mesh_row_release(p,address.source);
    released++;
  }
  return released;
}

// ../design/algorithm-sources.md#literal-page-transport
int mesh_rows_return(const struct mesh_rows *p, uint32_t row, uint64_t stamp){
  uint32_t page=__atomic_load_n(&p->table[row].page,__ATOMIC_ACQUIRE);
  struct ring *ring=&p->memory->r[REL];
  uint64_t head=atomic_load_explicit(&ring->head,memory_order_relaxed);
  if(head-atomic_load_explicit(&ring->tail,memory_order_acquire)>=MESH_RING) return 0;
  if(!mesh_row_zero(p,row,stamp)) return 0;
  *slot(p->memory,REL,head)=(struct desc){.page=page};
  atomic_store_explicit(&ring->head,head+1,memory_order_release);
  return 1;
}

// ../design/algorithm-sources.md#literal-page-transport
size_t mesh_rows_retire(const struct mesh_rows *p){
  size_t released=0;
  for(size_t row=0;row<p->count;row++){
    const struct mesh_row *r=&p->table[row];
    uint64_t stamp=__atomic_load_n(&r->stamp,__ATOMIC_ACQUIRE);
    if(!stamp || stamp>=MESH_ROW_WRITING || __atomic_load_n(&r->uses,__ATOMIC_ACQUIRE)) continue;
    uint32_t page=__atomic_load_n(&r->page,__ATOMIC_ACQUIRE);
    if(page==MESH_ROW_ABSENT) continue;
    int result=page<p->memory->pool?mesh_rows_return(p,(uint32_t)row,stamp):mesh_row_zero(p,(uint32_t)row,stamp);
    released+=result>0;
  }
  return released;
}

// ../design/algorithm-sources.md#literal-page-transport
size_t mesh_rows_progress(const struct mesh_rows *p, uint64_t epoch,
  const struct mesh_row_binding *bindings, size_t count){
  size_t changed=mesh_rows_acknowledge(p);
  changed+=mesh_rows_receive(p,bindings,count);
  changed+=mesh_rows_send(p,epoch,bindings,count);
  changed+=mesh_rows_retire(p);
  return changed;
}

// ../design/algorithm-sources.md#literal-page-checking
__attribute__((target("crc")))
int mesh_rows_digest(const struct mesh_rows *p, uint32_t input, uint32_t output, uint32_t index, uint64_t seed){
  if(index>=p->bytes/sizeof(uint64_t)) return EINVAL;
  const unsigned char *source=mesh_row_data(p,input);
  uint64_t *destination=mesh_row_data(p,output);
  if(!source || !destination) return EINVAL;
  uint32_t first=(uint32_t)seed, second=(uint32_t)(seed>>32)^UINT32_MAX;
  for(uint32_t i=0;i<p->bytes;i+=sizeof(uint64_t)){
    uint64_t word;
    memcpy(&word,source+i,sizeof word);
    first=__builtin_arm_crc32cd(first,word);
    second=__builtin_arm_crc32d(second,word);
  }
  destination[index]=((uint64_t)first<<32)|second;
  return 0;
}

// ../design/algorithm-sources.md#literal-page-checking
int mesh_rows_equal(const struct mesh_rows *p, uint32_t first, uint32_t second, uint64_t stamp){
  if(!mesh_rows_present(p,(struct mesh_row_map){.first=first,.count=1},0,stamp) ||
     !mesh_rows_present(p,(struct mesh_row_map){.first=second,.count=1},0,stamp)) return -EAGAIN;
  return memcmp(mesh_row_data(p,first),mesh_row_data(p,second),p->bytes)==0;
}
