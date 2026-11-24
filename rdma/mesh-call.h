#ifndef MESH_CALL_H
#define MESH_CALL_H
#include <stddef.h>
#include <stdint.h>
#include "mesh-view.h"
#ifdef __cplusplus
extern "C" {
#endif

#define MESH_PAGE_BYTES 4096u
#define MESH_FRAME 4096u
#define MESH_PAGE_NONE UINT32_MAX
#define MESH_TABLE (1u<<20)
#define MESH_DESTS 8u
#define MESH_WIRE_SAME_DATA 1u

/* Algebraic classes for switch-side page morphisms (Pages^k -> Pages^m) */
#define MESH_ALG_NONE         0u
#define MESH_ALG_C0_COMM      1u
#define MESH_ALG_C1_LIFTED    2u
#define MESH_ALG_C2_INTERVAL  3u

#define MESH_RED_OP(alg, id)        ((uint8_t)(((alg)<<6)|((id)&0x3fu)))
#define MESH_RED_ALG(op)            ((uint8_t)((op)>>6))
#define MESH_RED_NONE               0u
#define MESH_RED_SUM_F32            MESH_RED_OP(MESH_ALG_C0_COMM, 1u)
#define MESH_RED_MAX_F32            MESH_RED_OP(MESH_ALG_C0_COMM, 2u)
#define MESH_RED_MIN_F32            MESH_RED_OP(MESH_ALG_C0_COMM, 3u)
#define MESH_RED_SUM_I32            MESH_RED_OP(MESH_ALG_C0_COMM, 4u)
#define MESH_RED_SUM_I64            MESH_RED_OP(MESH_ALG_C0_COMM, 5u)
#define MESH_RED_ONLINE_SOFTMAX_F32 MESH_RED_OP(MESH_ALG_C1_LIFTED, 1u)
#define MESH_RED_AFFINE_SCAN_F32    MESH_RED_OP(MESH_ALG_C2_INTERVAL, 1u)

#define MESH_TAG_WEIGHT(w)          ((uint16_t)(w))
#define MESH_TAG_INTERVAL(lo, hi)   ((uint16_t)(((uint16_t)(lo)&0xffu)|(((uint16_t)(hi)&0xffu)<<8)))

struct hdr;
struct mesh_ctx { struct hdr *M; size_t len; int fd; };

struct mesh_entry_info {
  uint32_t page;
  uint32_t bytes;
  void *data;
};

struct mesh_xfer {
  uint32_t src_page;
  uint32_t bytes;
  uint32_t dst_node;
  uint32_t dst_index;
};

struct mesh_red_xfer {
  uint32_t src_page;
  uint32_t bytes;
  uint32_t dst_node;
  uint32_t dst_index;
  uint8_t  op;
  uint8_t  pad;
  uint16_t tag;
};

int mesh_attach(struct mesh_ctx *ctx);
int mesh_detach(struct mesh_ctx *ctx);

int mesh_alloc(struct mesh_ctx *ctx, uint32_t count, int allocate_pages,
               uint32_t *out_indices, uint32_t *out_pages);
int mesh_alloc_contiguous(struct mesh_ctx *ctx, uint32_t count,
                          uint32_t *out_indices, void **out_base);

int mesh_hold(struct mesh_ctx *ctx, uint32_t page);
uint32_t mesh_hold_index(struct mesh_ctx *ctx, uint32_t index);
void mesh_release(struct mesh_ctx *ctx, uint32_t page);
void mesh_clear_indices(struct mesh_ctx *ctx, const uint32_t *indices, uint32_t count);
void mesh_free_indices(struct mesh_ctx *ctx, const uint32_t *indices, uint32_t count);

int mesh_defragment(struct mesh_ctx *ctx, const uint32_t *indices, uint32_t count, void **out_base);
int mesh_write(struct mesh_ctx *ctx, const struct mesh_xfer *xfers, uint32_t count);
int mesh_reduce_write(struct mesh_ctx *ctx, const struct mesh_red_xfer *xfers, uint32_t count);

int mesh_entry(struct mesh_ctx *ctx, uint32_t index, struct mesh_entry_info *out);
int mesh_entry_red(struct mesh_ctx *ctx, uint32_t index, struct mesh_entry_info *out, uint16_t *out_tag);
void *mesh_page_ptr(struct mesh_ctx *ctx, uint32_t page);
uint32_t mesh_nodes_reached(struct hdr *M, struct mesh_reach *out, uint32_t room);

#ifdef __cplusplus
}
#endif
#endif
