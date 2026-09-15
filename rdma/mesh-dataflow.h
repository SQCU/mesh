#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; uint32_t rows,arena; int fd; };
static inline uint64_t mesh_length(const struct mesh_ctx *c){ return c->M->length; }
static inline uint64_t mesh_data_offset(const struct mesh_ctx *c){ return c->M->data_off; }
static inline uint64_t mesh_page_offset(const struct mesh_ctx *c){ return c->M->page_off; }
static inline uint32_t mesh_block_pages(const struct mesh_ctx *c){ return c->M->block; }
static inline uint32_t mesh_arena_pages(const struct mesh_ctx *c){ return c->M->rows; }
static inline uint32_t mesh_window(const struct mesh_ctx *c){ return mesh_window_blocks(c->M); }
static inline void *mesh_page_address(struct mesh_ctx *c,uint32_t page){ return mesh_at(c->M,page); }
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_detach(struct mesh_ctx *);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
int mesh_backing_alloc(struct mesh_ctx *,uint32_t first,uint32_t count,uint32_t quantum,int contiguous);
void mesh_backing_release(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
void mesh_map(struct mesh_ctx *,uint32_t first,uint32_t count,uint32_t page);
void mesh_constant(struct mesh_ctx *,uint32_t first,uint32_t count);
int mesh_buffer_retain(struct hdr *,uint32_t first,uint32_t count);
void mesh_buffer_seal(struct hdr *,uint32_t first,uint32_t count);
uint32_t mesh_collect(struct hdr *,uint32_t pending);
void *mesh_row_data(struct mesh_ctx *,uint32_t row);
void mesh_publish_partial(struct mesh_ctx *,uint32_t first,uint32_t count);
#endif
