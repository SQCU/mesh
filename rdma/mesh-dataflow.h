#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_ctx { struct hdr *M; size_t len; uint32_t rows,arena; int fd; void *execution,*readers; };
struct mesh_row_range { uint32_t first,count; };
struct mesh_row_map { uint32_t first,count,stride,plane; const struct mesh_row_range *ranges; const uint32_t *members; const size_t *member_offsets; };
struct mesh_active;
struct mesh_index_candidate { struct mesh_row_map *maps; uint32_t count; size_t input; struct mesh_active *producer; struct mesh_row_map disposition; size_t function; };
struct mesh_indexed_read {
  struct mesh_row_map *selector; uint32_t selectors,vector_maps;
  const uint32_t *indices,*bounds; size_t rows,columns,row_stride,column_stride,bounds_stride;
  struct mesh_index_candidate *candidate; uint32_t candidates,retired,selected,completed,mapped;
  struct mesh_active *domain;
  struct mesh_indexed_read *next;
};
struct mesh_route_vector { const uint32_t *values; size_t columns,row_stride,column_stride,length; };
struct mesh_route {
  struct mesh_route_vector owners,ordinals,offsets;
  struct mesh_row_map *metadata; uint32_t metadata_count;
  struct mesh_index_candidate *candidate; uint32_t candidates,consumers,retired,completed,prepared;
  void **watches; size_t *functions; void *table,*authority;
  struct mesh_route *next;
};
struct mesh_route_use { struct mesh_route *domain; uint32_t consumer; struct mesh_route_use *next; };
struct mesh_active {
  struct mesh_row_map *count_maps; uint32_t maps,slot,disposition,omitted,retired,inputs;
  const uint32_t *count; struct mesh_row_function *function; _Atomic uint64_t omissions;
};
struct mesh_row_function { struct mesh_row_map *input,*output; uint32_t inputs,outputs,rows; struct mesh_indexed_read *indexed; struct mesh_route_use *routes; struct mesh_active *active; };
/* ledger D5: `binding` orders blocks within `queue`; both participants declare the same identities and queues */
struct mesh_row_binding { uint32_t first,count,binding,plane; uint16_t queue,receive; uint64_t bytes; };
struct mesh_transfer_event { uint32_t queue,direction; struct mesh_transfer transfer; uint64_t ready_ns,post_ns,cq_ns,occurrences; };
struct mesh_reader_event {uint32_t source,member,plane,completed,flags;};
struct mesh_row_metadata { uint64_t stamp,when; uint32_t function,index,peer; int64_t code; uint32_t domain,reserved; };
static inline struct mesh_row_range mesh_range(struct mesh_row_map m,uint32_t index){ return m.ranges?m.ranges[index]:(struct mesh_row_range){m.first+index*m.stride,m.count}; }
static inline uint64_t mesh_length(const struct mesh_ctx *c){ return c->M->length; }
static inline uint64_t mesh_data_offset(const struct mesh_ctx *c){ return c->M->data_off; }
static inline uint64_t mesh_page_offset(const struct mesh_ctx *c){ return c->M->page_off; }
static inline uint32_t mesh_block_pages(const struct mesh_ctx *c){ return c->M->block; }
static inline uint32_t mesh_arena_pages(const struct mesh_ctx *c){ return c->M->rows; }
static inline uint32_t mesh_window(const struct mesh_ctx *c){ return mesh_window_blocks(c->M); }
static inline void *mesh_page_address(struct mesh_ctx *c,uint32_t page){ return mesh_at(c->M,page); }
struct mesh_ctx *mesh_context(void);
struct hdr *mesh_region(struct mesh_ctx *);
int mesh_execution_add(struct mesh_ctx *,struct mesh_row_function *,void *owner,void (*submit)(void *,uint32_t),void *argument);
void mesh_execution_remove(struct mesh_ctx *,void *owner);
int mesh_execution_active(struct mesh_ctx *,struct mesh_active *,void *owner);
int mesh_execution_route(struct mesh_ctx *,struct mesh_route *,void *owner);
int mesh_execution_indexed(struct mesh_ctx *,struct mesh_indexed_read *,void *owner);
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
void *mesh_view_create(struct mesh_ctx *,uint32_t first,size_t count);
int mesh_view_destroy(void *address,size_t length);
struct mesh_row_metadata mesh_link_metadata(struct mesh_ctx *,size_t);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
int mesh_backing_alloc(struct mesh_ctx *,uint32_t first,uint32_t count,uint32_t quantum,int contiguous);
void mesh_backing_release(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_arena_release(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_map(struct mesh_ctx *,uint32_t first,uint32_t count,uint32_t page);
void mesh_constant(struct mesh_ctx *,uint32_t first,uint32_t count);
int mesh_realize(struct mesh_ctx *,struct mesh_row_function *functions,size_t count,
  struct mesh_row_binding *bindings,size_t binding_count,struct mesh_row_map *returns,size_t return_count);

void *mesh_row_data(struct mesh_ctx *,uint32_t row);
int mesh_present(struct mesh_ctx *,struct mesh_row_map,uint32_t index);
int mesh_available(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index);
int mesh_writable(struct mesh_ctx *,uint32_t first,uint32_t count);
int mesh_republish(struct mesh_ctx *,uint32_t first,uint32_t count);
size_t mesh_issue(struct mesh_ctx *,const struct mesh_row_function *,uint32_t *indices,size_t capacity);
void mesh_publish_partial(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_complete(struct mesh_ctx *,const struct mesh_row_function *,const uint32_t *indices,size_t count);
void mesh_consume(struct mesh_ctx *,struct mesh_row_map,uint32_t index);
void mesh_reader_unbind(struct mesh_ctx *,struct mesh_row_map *);
struct mesh_reader_event mesh_reader_trace(struct mesh_ctx *,struct mesh_row_map,uint32_t index,uint32_t row);
size_t mesh_transfer_trace_count(struct mesh_ctx *);
struct mesh_transfer_event mesh_transfer_trace(struct mesh_ctx *,size_t index);
#endif
