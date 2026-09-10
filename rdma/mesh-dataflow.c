#include "mesh-dataflow.h"
#include <errno.h>
#include <math.h>
#include <sched.h>

typedef _Float16 mesh_half8 __attribute__((ext_vector_type(8)));
typedef float mesh_float8 __attribute__((ext_vector_type(8)));

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
      if(side) for(uint32_t j=0;j<i;j++){
        uint32_t a=0,b=0;
        while(a<f->rows && b<f->rows){
          uint64_t x=(uint64_t)m->first+(uint64_t)a*m->stride;
          uint64_t y=(uint64_t)maps[j].first+(uint64_t)b*maps[j].stride;
          if(x<y+maps[j].count && y<x+m->count) return EINVAL;
          if(x<y) a++; else b++;
        }
      }
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
        ready=__atomic_load_n(&r->stamp,__ATOMIC_ACQUIRE)==0 &&
          __atomic_load_n(&r->page,__ATOMIC_ACQUIRE)!=MESH_ROW_ABSENT;
      }
    }
    if(!ready) continue;
    for(uint32_t i=0;i<f->outputs;i++){
      const struct mesh_row_map *m=&f->output[i];
      for(uint32_t j=0;j<m->count;j++)
        __atomic_store_n(&p->table[m->first+index*m->stride+j].stamp,MESH_ROW_WRITING|stamp,__ATOMIC_RELEASE);
    }
    indices[selected++]=index;
  }
  return selected;
}

// ../design/algorithm-sources.md#literal-row-functions
int mesh_rows_publish(const struct mesh_rows *p, const struct mesh_row_function *f, uint32_t index, uint64_t stamp){
  if(index>=f->rows || !stamp || stamp>=MESH_ROW_WRITING) return EINVAL;
  for(uint32_t i=0;i<f->inputs;i++){
    const struct mesh_row_map *m=&f->input[i];
    if(!mesh_rows_present(p,*m,index,stamp)) return EINVAL;
    for(uint32_t j=0;j<m->count;j++)
      if(!__atomic_load_n(&p->table[m->first+index*m->stride+j].uses,__ATOMIC_ACQUIRE)) return EINVAL;
  }
  for(uint32_t i=0;i<f->outputs;i++){
    const struct mesh_row_map *m=&f->output[i];
    for(uint32_t j=0;j<m->count;j++)
      if(__atomic_load_n(&p->table[m->first+index*m->stride+j].stamp,__ATOMIC_ACQUIRE)!=(MESH_ROW_WRITING|stamp)) return EINVAL;
  }
  for(uint32_t i=0;i<f->outputs;i++){
    const struct mesh_row_map *m=&f->output[i];
    for(uint32_t j=0;j<m->count;j++)
      __atomic_store_n(&p->table[m->first+index*m->stride+j].stamp,stamp,__ATOMIC_RELEASE);
  }
  for(uint32_t i=0;i<f->inputs;i++){
    const struct mesh_row_map *m=&f->input[i];
    for(uint32_t j=0;j<m->count;j++) mesh_row_release(p,m->first+index*m->stride+j,stamp);
  }
  return 0;
}

// ../design/algorithm-sources.md#literal-row-functions
int mesh_row_release(const struct mesh_rows *p, uint32_t row, uint64_t stamp){
  struct mesh_row *r=&p->table[row];
  if(__atomic_load_n(&r->stamp,__ATOMIC_ACQUIRE)!=stamp) return -EINVAL;
  uint32_t uses=__atomic_load_n(&r->uses,__ATOMIC_RELAXED);
  while(uses){
    if(__atomic_compare_exchange_n(&r->uses,&uses,uses-1,0,__ATOMIC_ACQ_REL,__ATOMIC_RELAXED)) return uses==1;
  }
  return -EINVAL;
}

// ../design/algorithm-sources.md#literal-row-functions
int mesh_row_zero(const struct mesh_rows *p, uint32_t row, uint64_t stamp){
  struct mesh_row *r=&p->table[row];
  if(__atomic_load_n(&r->uses,__ATOMIC_ACQUIRE)) return 0;
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
int mesh_rows_add_f16(const struct mesh_rows *p, const uint32_t *inputs, size_t count,
  const uint32_t *accumulators, size_t elements, uint32_t index_row, uint32_t index, uint64_t stamp){
  if(!count || !elements || elements>p->bytes/sizeof(_Float16) || index>=p->bytes/sizeof(uint64_t) ||
     !stamp || stamp>=MESH_ROW_WRITING) return -EINVAL;
  uint64_t *indices=mesh_row_data(p,index_row);
  if(!indices) return 0;
  if(__atomic_load_n(indices+index,__ATOMIC_ACQUIRE)==stamp) return 0;
  for(size_t i=0;i<count;i++)
    if(!mesh_rows_present(p,(struct mesh_row_map){inputs[i],1,0},0,stamp)) return 0;
  size_t width=p->bytes/sizeof(float), pages=(elements+width-1)/width;
  for(size_t i=0;i<pages;i++)
    if(!mesh_row_data(p,accumulators[i]) ||
       __atomic_load_n(&p->table[accumulators[i]].stamp,__ATOMIC_ACQUIRE)!=(MESH_ROW_WRITING|stamp)) return 0;
  for(size_t page=0;page<pages;page++){
    size_t first=page*width, n=elements-first;
    if(n>width) n=width;
    float *output=mesh_row_data(p,accumulators[page]);
    for(size_t input=0;input<count;input++){
      const _Float16 *source=(const _Float16*)mesh_row_data(p,inputs[input])+first;
      size_t i=0;
      for(;i+8<=n;i+=8){
        mesh_half8 half;
        mesh_float8 sum;
        memcpy(&half,source+i,sizeof half);
        sum=__builtin_convertvector(half,mesh_float8);
        if(input){ mesh_float8 prior; memcpy(&prior,output+i,sizeof prior); sum+=prior; }
        memcpy(output+i,&sum,sizeof sum);
      }
      for(;i<n;i++) output[i]=(input?output[i]:0)+(float)source[i];
    }
  }
  for(size_t i=0;i<pages;i++) __atomic_store_n(&p->table[accumulators[i]].stamp,stamp,__ATOMIC_RELEASE);
  __atomic_store_n(indices+index,stamp,__ATOMIC_RELEASE);
  for(size_t i=0;i<count;i++) mesh_row_release(p,inputs[i],stamp);
  return 1;
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
int mesh_rows_normalize_f32(const struct mesh_rows *p, const uint32_t *accumulators,
  const uint32_t *gamma, const uint32_t *residual, const uint32_t *outputs,
  size_t elements, float epsilon, float scale){
  if(!elements || !(epsilon>0)) return EINVAL;
  size_t floats=p->bytes/sizeof(float), halves=p->bytes/sizeof(_Float16);
  for(size_t i=0;i<(elements+floats-1)/floats;i++) if(!mesh_row_data(p,accumulators[i])) return EINVAL;
  for(size_t i=0;i<(elements+halves-1)/halves;i++)
    if(!mesh_row_data(p,gamma[i]) || !mesh_row_data(p,residual[i]) || !mesh_row_data(p,outputs[i])) return EINVAL;
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
  return 0;
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
int mesh_rows_receive(const struct mesh_rows *p, uint64_t epoch,
  const struct mesh_row_binding *bindings, size_t count){
  struct ring *ring=&p->memory->r[CMP];
  uint64_t tail=atomic_load_explicit(&ring->tail,memory_order_relaxed);
  if(tail==atomic_load_explicit(&ring->head,memory_order_acquire)) return 0;
  uint64_t head=atomic_load_explicit(&ring->head,memory_order_acquire);
  for(uint64_t at=tail;at<head;at++){
    struct desc d=*slot(p->memory,CMP,at);
    struct mesh_row_address address;
    if(d.page>=p->memory->pool || d.bytes!=p->offset+p->bytes-sizeof(struct wire)) return -EPROTO;
    memcpy(&address,mesh_at(p->memory,d.page)+sizeof(struct wire),sizeof address);
    if(address.epoch!=epoch || !address.stamp || address.stamp>=MESH_ROW_WRITING || address.target>=p->count) return -ESTALE;
    int matched=0;
    for(size_t i=0;i<count;i++){
      const struct mesh_row_binding *b=&bindings[i];
      if(b->receive && b->peer==d.node && address.target>=b->first && address.target-b->first<b->count &&
         address.source==(uint64_t)b->remote+address.target-b->first){ matched=1; break; }
    }
    if(!matched) return -EPROTO;
    struct mesh_row *r=&p->table[address.target];
    if(__atomic_load_n(&r->page,__ATOMIC_ACQUIRE)!=MESH_ROW_ABSENT) continue;
    if(__atomic_load_n(&r->stamp,__ATOMIC_ACQUIRE)>=address.stamp) return -ESTALE;
    __atomic_store_n(&r->page,d.page,__ATOMIC_RELEASE);
    __atomic_store_n(&r->stamp,address.stamp,__ATOMIC_RELEASE);
    ring_erase(ring,slot(p->memory,CMP,0),sizeof(struct desc),MESH_RING,at);
    return 1;
  }
  return 0;
}

// ../design/algorithm-sources.md#literal-page-transport
size_t mesh_rows_acknowledge(const struct mesh_rows *p, uint64_t epoch){
  size_t released=0;
  struct desc d;
  while(!pop(p->memory,ACK,&d)){
    if(d.page<p->memory->pool || d.page>=p->memory->pool+p->memory->arena) continue;
    struct mesh_row_address address;
    memcpy(&address,mesh_at(p->memory,d.page)+sizeof(struct wire),sizeof address);
    if(address.epoch!=epoch || address.source>=p->count ||
       __atomic_load_n(&p->table[address.source].page,__ATOMIC_ACQUIRE)!=d.page) continue;
    released+=mesh_row_release(p,address.source,address.stamp)>=0;
  }
  return released;
}

// ../design/algorithm-sources.md#literal-page-transport
int mesh_rows_return(const struct mesh_rows *p, uint32_t row, uint64_t stamp){
  uint32_t page=__atomic_load_n(&p->table[row].page,__ATOMIC_ACQUIRE);
  if(page>=p->memory->pool) return -EINVAL;
  if(!mesh_row_zero(p,row,stamp)) return 0;
  struct desc d={.page=page};
  while(push(p->memory,REL,&d)) sched_yield();
  return 1;
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
  if(!mesh_rows_present(p,(struct mesh_row_map){first,1,0},0,stamp) ||
     !mesh_rows_present(p,(struct mesh_row_map){second,1,0},0,stamp)) return -EAGAIN;
  return memcmp(mesh_row_data(p,first),mesh_row_data(p,second),p->bytes)==0;
}
