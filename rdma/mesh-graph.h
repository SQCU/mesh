#ifndef MESH_GRAPH_H
#define MESH_GRAPH_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Zero-copy single-command-buffer Metal + flat-DMA FX graph execution engine (libmesh-graph.dylib).
   Compiled once from an explicit functional IR (BufferSpec[] + MeshGraphOp[]), then replayed with
   zero Python per op, zero ATen/MPSGraph dispatch, zero command-buffer splits, and zero Python per
   token during multi-step autoregressive decode. */

enum mesh_graph_buf_kind {
  MG_BUF_WEIGHT = 0, /* static parameter/constant (borrowed zero-copy if 16 KiB aligned, else 1x upload) */
  MG_BUF_ARENA  = 1, /* static shared Metal activation arena (sub-allocated via MG_BUF_VIEW) */
  MG_BUF_STATE  = 2, /* persistent input/output/KV-cache buffer (host-accessible zero-copy view) */
  MG_BUF_VIEW   = 3  /* zero-copy view into parent buffer at parent_offset (access by view) */
};

enum mesh_graph_dtype {
  MG_DTYPE_BF16 = 0,
  MG_DTYPE_F16  = 1,
  MG_DTYPE_F32  = 2,
  MG_DTYPE_I32  = 3,
  MG_DTYPE_U32  = 4
};

enum mesh_graph_wire_mode {
  MG_WIRE_BF16       = 0, /* MESH_RED_SUM_BF16 in-switch NEON */
  MG_WIRE_F32        = 1, /* MESH_RED_SUM_F32 in-switch NEON */
  MG_WIRE_GATHER     = 2, /* chunk-major Steiner-multicast mesh_write, rank-ordered fp32 GPU sum */
  MG_WIRE_HAD8       = 3, /* MESH_RED_SUM_HAD_I8 in-switch NEON */
  MG_WIRE_HAD8_GATHER = 4, /* Steiner-multicast had_int8 + linear Hadamard-domain GPU sum (0 CPU touch) */
  MG_WIRE_AUTO       = 5  /* Pareto auto: gather for <= 2 pages, had8g for > 2 pages */
};

enum mesh_graph_op_kind {
  MG_OP_EMBEDDING         = 1,  /* out[B,D] = W[tok[B],D] * fparam0 (fparam0==0 -> 1.0) */
  MG_OP_LINEAR            = 2,  /* out[M,N] = x[M,K] @ W[N,K]^T */
  MG_OP_LINEAR_PUBLISH    = 3,  /* out[M,N] = x[M,K] @ W[N,K]^T + fused /mesh0 publish + doorbell */
  MG_OP_CROSSING_SUM_NORM = 4,  /* await /mesh0 W[] + sum + optional RMSNorm(w1) + res [+ RMSNorm(w2)] + retire */
  MG_OP_RMS_NORM          = 5,  /* RMSNorm(x, w1, eps) or fused AddNorm(p, res, w1, scalar, w2, eps) */
  MG_OP_HEAD_NORM_ROPE    = 6,  /* Q/(K,V) per-head RMSNorm + RoPE (base or freqs) + optional KV cache write */
  MG_OP_KV_SDPA_DECODE    = 7,  /* GQA decode attention over KV cache [B,n_kv,max_seq,D] using pos_buf[0] */
  MG_OP_GATE_UP_ACT       = 8,  /* fused act(x @ Wg^T) * (x @ Wu^T) (u0=0: GeGLU gelu_approx, u0=1: SwiGLU silu) */
  MG_OP_PLE_GATE          = 9,  /* fused gelu_approx(x @ Wg^T) * pli[:, layer, :] */
  MG_OP_ELEMENTWISE       = 10, /* unary/binary elementwise (u0 = subop) */
  MG_OP_VOCAB_ARGMAX      = 11, /* row_argmax(lg) + flat-DMA crossing + in-GPU tok_buf/out_tok/pos_buf update */
  MG_OP_ALLREDUCE         = 12  /* standalone flat-DMA all-reduce (publish + await + sum + retire) */
};

enum mesh_graph_ew_subop {
  MG_EW_ADD       = 0, /* out = a + b */
  MG_EW_MUL       = 1, /* out = a * b */
  MG_EW_SCALE     = 2, /* out = a * f0 */
  MG_EW_ADD_SCALE = 3, /* out = (a + b) * f0 */
  MG_EW_GELU      = 4, /* out = gelu_approx(a) */
  MG_EW_SILU      = 5, /* out = silu(a) */
  MG_EW_RELU      = 6, /* out = max(a, 0) */
  MG_EW_COPY      = 7, /* out = a */
  MG_EW_MUL_SLICE = 8, /* out[b,d] = gelu_approx(a[b,d]) * b_full[b, layer_off + d] */
  MG_EW_RES_SCALE = 9  /* out = (a + b) * scalar_buf[0] */
};

/* Explicit 96-byte flat instruction descriptor (zero heap allocations inside execution loop). */
struct mesh_graph_op {
  uint32_t kind;        /* enum mesh_graph_op_kind */
  uint32_t dtype;       /* enum mesh_graph_dtype */
  int32_t  in[6];       /* input buffer IDs (-1 = unused) */
  int32_t  out[3];      /* output buffer IDs (-1 = unused) */
  uint32_t u[8];        /* integer shape / config params */
  float    f[4];        /* float params (eps, scale, rope_base_log2, etc.) */
};

typedef struct mesh_graph mesh_graph_t;

mesh_graph_t *mesh_graph_create(int world_size, int rank, const char *comm_id, int default_wire_mode);
void          mesh_graph_destroy(mesh_graph_t *g);

int           mesh_graph_add_buffer(mesh_graph_t *g, uint64_t nbytes, int kind,
                                    const void *init_data, int parent_buf, uint64_t parent_off);
void         *mesh_graph_buffer_ptr(mesh_graph_t *g, int buf_id);

int           mesh_graph_add_crossing(mesh_graph_t *g, uint32_t n_elems, uint32_t esize,
                                      int wire_mode, int is_argmax);
int           mesh_graph_add_op(mesh_graph_t *g, const struct mesh_graph_op *op);
int           mesh_graph_compile(mesh_graph_t *g);

/* Execute the compiled FX graph `n_steps` times in pure C++/Metal (1 single MTLCommandBuffer per step,
   pipelined 1 step ahead, zero Python dispatch per op or per token).
   If out_tok_buf_id >= 0 and out_tokens != NULL, copies out_tokens[0..n_steps*B-1].
   Returns steady-state ms/step (median of interior steps when n_steps >= 4, else wall_ms / n_steps). */
double        mesh_graph_run(mesh_graph_t *g, uint32_t n_steps, uint32_t warm_steps,
                             int32_t *out_tokens, uint32_t out_tok_stride, double *out_wall_ms);

void          mesh_graph_barrier(mesh_graph_t *g);
uint32_t      mesh_graph_node_id(mesh_graph_t *g);

#ifdef __cplusplus
}
#endif
#endif
