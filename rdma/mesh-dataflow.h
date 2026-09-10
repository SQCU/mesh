#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include <errno.h>
#define MESH_ROW_ABSENT UINT32_MAX
#define MESH_ROW_WRITING (UINT64_C(1)<<63)
#define MESH_ROW_CONSTANT UINT64_MAX
struct mesh_page_header { struct wire wire; uint16_t padding; uint64_t table,stamp; uint32_t source,target; uint64_t when; int64_t code; uint32_t domain,function,index,peer; };
struct mesh_row { uint32_t page,uses; uint64_t stamp,send; };
struct mesh_send { struct mesh_page_header header; uint32_t page,count; uint64_t row; };
struct mesh_receive_binding { uint32_t first,count,uses,reserved; };
struct mesh_table { uint64_t next,identity,rows,bindings; uint32_t count,binding_count; };
struct mesh_rows {
  struct hdr *memory; struct mesh_row *table; size_t count; uint32_t offset,bytes;
  struct mesh_ctx *context; uint64_t identity,shared; uint64_t *blocks;
  const struct mesh_row_binding *bindings; size_t binding_count;
  const struct mesh_row_function *functions; size_t function_count;
  const struct mesh_row_map *returns; size_t return_count;
};
struct mesh_row_range { uint32_t first,count; };
struct mesh_row_map { uint32_t first,count,stride,physical,physical_stride,uses; const struct mesh_row_range *ranges; };
struct mesh_row_function { struct mesh_row_map *input,*output; uint32_t inputs,outputs,rows; };
struct mesh_row_binding { uint32_t first,count,remote,uses; uint16_t peer,receive; uint64_t remote_table; };
struct mesh_row_metadata { uint64_t stamp,when; uint32_t function,index,peer; int64_t code; uint32_t domain,reserved; };
// ../design/algorithm-sources.md#configuration-storage-layout
static inline size_t mesh_send_storage_pages(size_t count,uint32_t bytes){
  size_t slots=bytes/sizeof(struct mesh_send); return (count+slots-1)/slots;
}
// ../design/algorithm-sources.md#configuration-storage-layout
static inline struct mesh_send *mesh_record(struct hdr *m,uint64_t base,size_t index){
  size_t slots=m->pgsz/sizeof(struct mesh_send);
  return (struct mesh_send*)((unsigned char*)m+base+(index/slots)*m->pgsz+(index%slots)*sizeof(struct mesh_send));
}
// ../design/algorithm-sources.md#transport-page-addressing
static inline struct mesh_page_header *mesh_header(struct hdr *m,uint32_t i){ return &mesh_record(m,m->headers_off,i)->header; }
// ../design/algorithm-sources.md#transport-active-request-indices
static inline size_t mesh_hot_prefix(uint32_t pool,uint32_t arena,uint32_t bytes){
  size_t receives=((size_t)pool+63)/64;
  size_t sends=((size_t)arena*(bytes/sizeof(struct mesh_send))+63)/64;
  return RINGS+NRING*MESH_RING*sizeof(struct desc)+sizeof(struct mesh_port_info)+(receives+sends)*sizeof(uint64_t);
}
// ../design/algorithm-sources.md#transport-active-request-indices
static inline uint64_t *mesh_hot_rx(struct hdr *m){
  return (uint64_t*)(mesh_ports(m)+1);
}
// ../design/algorithm-sources.md#transport-active-request-indices
static inline uint64_t *mesh_hot_tx(struct hdr *m){
  return mesh_hot_rx(m)+((size_t)m->pool+63)/64;
}
// ../design/algorithm-sources.md#transport-active-request-indices
static inline size_t mesh_hot_send_index(struct hdr *m,uint64_t offset){
  size_t local=offset-m->data_off-(size_t)m->pool*m->pgsz;
  return (local/m->pgsz)*(m->pgsz/sizeof(struct mesh_send))+(local%m->pgsz)/sizeof(struct mesh_send);
}
// ../design/algorithm-sources.md#transport-active-request-indices
static inline void mesh_hot_blocks(struct hdr *m,uint64_t *mask,size_t words){
  memset(mask,0,words*sizeof *mask);
  size_t receive_words=((size_t)m->pool+63)/64;
  for(size_t i=0;i<receive_words;i++) mask[i]=__atomic_load_n(mesh_hot_rx(m)+i,__ATOMIC_ACQUIRE);
  size_t send_words=((size_t)m->arena*(m->pgsz/sizeof(struct mesh_send))+63)/64;
  for(size_t i=0;i<send_words;i++){
    uint64_t active=__atomic_load_n(mesh_hot_tx(m)+i,__ATOMIC_ACQUIRE);
    while(active){
      size_t index=64*i+(size_t)__builtin_ctzll(active);
      struct mesh_send *record=mesh_record(m,m->data_off+(size_t)m->pool*m->pgsz,index);
      uint32_t page=__atomic_load_n(&record->page,__ATOMIC_ACQUIRE);
      mask[page/64]|=UINT64_C(1)<<(page%64);
      size_t header=((unsigned char*)record-(unsigned char*)m-m->data_off)/m->pgsz;
      mask[header/64]|=UINT64_C(1)<<(header%64);
      size_t row=(record->row-m->data_off)/m->pgsz;
      mask[row/64]|=UINT64_C(1)<<(row%64);
      active&=active-1;
    }
  }
}
// ../design/algorithm-sources.md#literal-row-functions
static inline struct mesh_row_range mesh_range(struct mesh_row_map m,uint32_t index){ return m.ranges?m.ranges[index]:(struct mesh_row_range){m.first+index*m.stride,m.count}; }
// ../design/algorithm-sources.md#source-access-completion
static inline void mesh_release(struct hdr *m,struct mesh_row *row){
  if(__atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE)==MESH_ROW_CONSTANT) return;
  if(__atomic_fetch_sub(&row->uses,1,__ATOMIC_ACQ_REL)!=1) return;
  uint32_t page=__atomic_exchange_n(&row->page,MESH_ROW_ABSENT,__ATOMIC_ACQ_REL);
  if(page<m->pool && push(m,REL,&(struct desc){.page=page})){
    mesh_ports(m)->code=ENOBUFS; mesh_ports(m)->domain=1;
  }
}
// ../design/algorithm-sources.md#source-access-completion
static inline void mesh_rows_sent(struct hdr *m,const struct mesh_send *record){
  struct mesh_row *row=(struct mesh_row*)((unsigned char*)m+record->row);
  if(__atomic_load_n(&row->stamp,__ATOMIC_ACQUIRE)==MESH_ROW_CONSTANT) __atomic_fetch_sub(&row->uses,1,__ATOMIC_ACQ_REL);
  else mesh_release(m,row);
}
// ../design/algorithm-sources.md#local-completion-bindings
static inline void mesh_send_row(struct hdr *m,struct mesh_row *row,uint64_t stamp){
  uint64_t offset=row->send;
  if(!offset) return;
  struct mesh_send *records=(struct mesh_send*)((unsigned char*)m+offset);
  uint32_t count=records->count;
  for(uint32_t i=0;i<count;i++){
    struct mesh_send *record=(struct mesh_send*)((unsigned char*)m+offset); record->header.stamp=stamp;
    record->page=__atomic_load_n(&row->page,__ATOMIC_ACQUIRE);
    size_t request=mesh_hot_send_index(m,offset);
    __atomic_fetch_or(mesh_hot_tx(m)+request/64,UINT64_C(1)<<(request%64),__ATOMIC_RELEASE);
    if(push(m,SUB,&(struct desc){.page=record->page,.bytes=m->pgsz,.node=(uint16_t)record->header.peer,
      .header=offset})){
      __atomic_fetch_and(mesh_hot_tx(m)+request/64,~(UINT64_C(1)<<(request%64)),__ATOMIC_RELEASE);
      mesh_ports(m)->code=ENOBUFS; mesh_ports(m)->domain=1;
      mesh_rows_sent(m,record);
    }
    offset+=sizeof *record;
    if(offset%m->pgsz+sizeof *record>m->pgsz) offset+=m->pgsz-offset%m->pgsz;
  }
}
// ../design/algorithm-sources.md#local-completion-bindings
static inline void mesh_bind_receive(struct hdr *m,uint32_t page){
  struct mesh_send *record=(struct mesh_send*)mesh_header(m,page);
  struct mesh_page_header *h=&record->header;
  uint64_t received=MESH_ROW_WRITING|page;
  if(__atomic_load_n(&record->row,__ATOMIC_ACQUIRE)!=received) return;
  uint64_t at=atomic_load_explicit(&m->tables,memory_order_acquire);
  while(at){
    struct mesh_table *t=(struct mesh_table*)((unsigned char*)m+at);
    if(__atomic_load_n(&t->identity,__ATOMIC_ACQUIRE)==h->table){
      struct mesh_receive_binding *bindings=(struct mesh_receive_binding*)((unsigned char*)m+t->bindings);
      struct mesh_receive_binding b=bindings[h->target];
      struct mesh_row *row=(struct mesh_row*)((unsigned char*)m+t->rows)+b.first+h->index;
      uint64_t target=(uint64_t)((unsigned char*)row-(unsigned char*)m);
      if(!__atomic_compare_exchange_n(&record->row,&received,target,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE)) return;
      __atomic_store_n(&row->uses,b.uses,__ATOMIC_RELAXED);
      __atomic_store_n(&row->page,page,__ATOMIC_RELAXED);
      uint64_t stamp=h->stamp;
      __atomic_store_n(&row->stamp,stamp,__ATOMIC_RELEASE);
      mesh_send_row(m,row,stamp);
      return;
    }
    at=t->next;
  }
}
// ../design/algorithm-sources.md#local-completion-bindings
static inline void mesh_rows_received(struct hdr *m,uint32_t page){
  struct mesh_send *record=(struct mesh_send*)mesh_header(m,page);
  __atomic_store_n(&record->row,MESH_ROW_WRITING|page,__ATOMIC_RELEASE);
  mesh_bind_receive(m,page);
}
struct mesh_row_metadata mesh_link_metadata(struct mesh_ctx *,size_t);
size_t mesh_storage_pages(size_t,uint32_t);
struct mesh_rows *mesh_rows_create(struct mesh_ctx *,size_t,uint64_t);
uint32_t mesh_rows_allocate(struct mesh_rows *,size_t,size_t);
void mesh_rows_map(struct mesh_rows *,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t);
void mesh_rows_constant(struct mesh_rows *,uint32_t,uint32_t);
int mesh_rows_close(struct mesh_ctx *);
void mesh_rows_retire(struct mesh_rows *);
size_t mesh_rows_issue(const struct mesh_rows *,const struct mesh_row_function *,uint64_t,uint32_t *,size_t);
void mesh_rows_complete(const struct mesh_rows *,const struct mesh_row_function *,uint64_t,const uint32_t *,size_t);
int mesh_rows_realize(const struct mesh_rows *,const struct mesh_row_function *,size_t,struct mesh_row_binding *,size_t,struct mesh_row_map *,size_t);
void *mesh_row_data(const struct mesh_rows *,uint32_t);
int mesh_rows_present(const struct mesh_rows *,struct mesh_row_map,uint32_t,uint64_t);
void mesh_rows_report(const struct mesh_rows *,struct mesh_row_map,uint32_t,struct mesh_row_metadata);
void mesh_row_release(const struct mesh_rows *,uint32_t);
#endif
