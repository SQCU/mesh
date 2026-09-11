#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_ctx { struct hdr *M; size_t len; uint32_t rows,landing,arena; };
struct mesh_row_range { uint32_t first,count; };
struct mesh_row_map { uint32_t first,count,stride,plane; const struct mesh_row_range *ranges; };
struct mesh_row_function { struct mesh_row_map *input,*output; uint32_t inputs,outputs,rows; };
struct mesh_row_binding { uint32_t first,count,binding,plane; uint16_t peer,receive; };
struct mesh_row_metadata { uint64_t stamp,when; uint32_t function,index,peer; int64_t code; uint32_t domain,reserved; };
static inline struct mesh_row_range mesh_range(struct mesh_row_map m,uint32_t index){ return m.ranges?m.ranges[index]:(struct mesh_row_range){m.first+index*m.stride,m.count}; }
static inline uint64_t mesh_length(const struct mesh_ctx *c){ return c->M->length; }
static inline uint64_t mesh_data_offset(const struct mesh_ctx *c){ return c->M->data_off; }
static inline uint64_t mesh_page_offset(const struct mesh_ctx *c){ return c->M->page_off; }
static inline uint32_t mesh_block_pages(const struct mesh_ctx *c){ return c->M->block; }
static inline uint32_t mesh_pool_pages(const struct mesh_ctx *c){ return c->M->pool; }
static inline uint32_t mesh_arena_pages(const struct mesh_ctx *c){ return c->M->arena; }
static inline uint32_t mesh_window(const struct mesh_ctx *c){ return mesh_window_blocks(c->M); }
static inline void *mesh_page_address(struct mesh_ctx *c,uint32_t page){ return mesh_at(c->M,page); }
struct mesh_ctx *mesh_context(void);
struct hdr *mesh_region(struct mesh_ctx *);
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
struct mesh_row_metadata mesh_link_metadata(struct mesh_ctx *,size_t);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
uint32_t mesh_landing_alloc(struct mesh_ctx *,uint32_t count);
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
void mesh_map(struct mesh_ctx *,uint32_t first,uint32_t count,uint32_t page);
void mesh_constant(struct mesh_ctx *,uint32_t first,uint32_t count);
int mesh_realize(struct mesh_ctx *,struct mesh_row_function *functions,size_t count,
  struct mesh_row_binding *bindings,size_t binding_count,struct mesh_row_map *returns,size_t return_count);

void *mesh_row_data(struct mesh_ctx *,uint32_t row);
int mesh_present(struct mesh_ctx *,struct mesh_row_map,uint32_t index);
int mesh_available(struct mesh_ctx *c,struct mesh_row_map map,uint32_t index);
int mesh_republish(struct mesh_ctx *,uint32_t first,uint32_t count);
size_t mesh_issue(struct mesh_ctx *,const struct mesh_row_function *,uint32_t *indices,size_t capacity);
void mesh_complete(struct mesh_ctx *,const struct mesh_row_function *,const uint32_t *indices,size_t count);
void mesh_consume(struct mesh_ctx *,struct mesh_row_map,uint32_t index);
#endif
