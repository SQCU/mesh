#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_ctx { struct hdr *M; size_t len; uint32_t rows,arena; int fd; void *execution,*readers; };
struct mesh_reader_member { _Atomic uint64_t generation; uint32_t row; };
struct mesh_row_map { uint32_t first,count; struct mesh_reader_member *members; };
struct mesh_row_function {
  struct mesh_row_map *input,*output;
  uint32_t inputs,outputs;
  void (*submit)(void *);
  void *argument;
};
/* ledger D5: `binding` orders blocks within `queue`; both participants declare the same identities and queues */
struct mesh_row_binding { uint32_t first,count,binding,plane; uint16_t queue,receive; uint64_t bytes; };
static inline uint64_t mesh_length(const struct mesh_ctx *c){ return c->M->length; }
static inline uint64_t mesh_data_offset(const struct mesh_ctx *c){ return c->M->data_off; }
static inline uint64_t mesh_page_offset(const struct mesh_ctx *c){ return c->M->page_off; }
static inline uint32_t mesh_block_pages(const struct mesh_ctx *c){ return c->M->block; }
static inline uint32_t mesh_arena_pages(const struct mesh_ctx *c){ return c->M->rows; }
static inline uint32_t mesh_window(const struct mesh_ctx *c){ return mesh_window_blocks(c->M); }
static inline void *mesh_page_address(struct mesh_ctx *c,uint32_t page){ return mesh_at(c->M,page); }
struct mesh_ctx *mesh_context(void);
struct hdr *mesh_region(struct mesh_ctx *);
int mesh_execution_start(struct mesh_ctx *,struct mesh_row_function **,size_t count,void *owner);
void mesh_execution_remove(struct mesh_ctx *,void *owner);
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);
void *mesh_view_create(struct mesh_ctx *,uint32_t first,size_t count,_Atomic uint32_t *mapped);
int mesh_view_bind(struct mesh_ctx *,uint32_t first,size_t count,void *address,_Atomic uint32_t *mapped);
int mesh_view_destroy(void *address,size_t length);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
int mesh_backing_alloc(struct mesh_ctx *,uint32_t first,uint32_t count,uint32_t quantum,int contiguous);
void mesh_backing_release(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_arena_release(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_map(struct mesh_ctx *,uint32_t first,uint32_t count,uint32_t page);
void mesh_constant(struct mesh_ctx *,uint32_t first,uint32_t count);
int mesh_realize(struct mesh_ctx *,struct mesh_row_function *const *functions,size_t count,
  struct mesh_row_binding *bindings,size_t binding_count,struct mesh_row_map *returns,size_t return_count);

void *mesh_row_data(struct mesh_ctx *,uint32_t row);
int mesh_available(struct mesh_ctx *,struct mesh_row_map);
int mesh_issue(struct mesh_ctx *,const struct mesh_row_function *);
void mesh_publish_partial(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_complete(struct mesh_ctx *,const struct mesh_row_function *);
void mesh_consume(struct mesh_ctx *,struct mesh_row_map);
void mesh_reader_unbind(struct mesh_ctx *,struct mesh_row_map *);
#endif
