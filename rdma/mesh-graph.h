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
  MG_OP_ALLREDUCE         = 12, /* standalone flat-DMA all-reduce (publish + await + sum + retire) */
  MG_OP_CONCAT            = 13, /* concat along last dim: out[M, K0+K1] = cat(a[M,K0], b[M,K1]) */
  MG_OP_SPEC_VERIFY_COMMIT = 14, /* in-GPU lossless greedy spec verify + pos/anchor update + hidden select + stats */
  MG_OP_WRITE_TOK_SLOT    = 15, /* write 1 int32 token from src[0] into dst[slot_idx] */
  MG_OP_PLAN_PRENORM      = 16, /* fused CGT features + prenormalize obs/carts/teams + asinh + mp_norm + NAX pad */
  MG_OP_PLAN_EMBED_CAT    = 17, /* fused mp_norm + type_embed mp_lerp + concat [N,J,K] + mp_norm into x[E,M,D] */
  MG_OP_MP_NORM           = 18, /* pure BF16 rms_norm(x, eps) without weight tensor */
  MG_OP_MP_LERP           = 19, /* fused mp_norm(mp_lerp(a, norm_b?mp_norm(b):b, sigmoid_buf[idx])) */
  MG_OP_MP_SWIGLU         = 20, /* fused mp_silu(gate, scale) * up from packed [M, 2*H] -> [M, H] */
  MG_OP_CGT_GRAM_STAGE1   = 21, /* fused global Gram (16KB TG SRAM) + pool_score + team_pool + blend_pool */
  MG_OP_CGT_TEAM_RIVAL    = 22, /* fused norm(q_t,k_t,v_t) + team rival Gram (q@k^T/sqrt(K))@v -> t_mixed */
  MG_OP_CGT_BCAST_MERGE   = 23, /* fused norm(t_bcast) + t_global + broadcast + mp_norm(mp_norm(global_raw) + team_vals) */
  MG_OP_PLAN_SLICE_NORM   = 24, /* fused mp_norm(x) + split [E,M,D] -> p_h[E*N,D], c_h[E*J,D], t_h[E*K,D] */
  MG_OP_BILINEAR_LOGITS   = 25, /* fused temp * dot(unit(Q), unit(K)) with optional own-team mask */
  MG_OP_PLAN_POST_PACK    = 26, /* fused cont_mean, values, dyn + optional det/stoch sample & pack_plan[E,N,16] */
  MG_OP_SIMDT_STEP        = 27, /* fused TG-SRAM DeltaTimePayloadEngine step (score, carts, goals, kinematics, combat) */
  MG_OP_SIMDT_OBS         = 28, /* emit BF16 obs[E,N,84], carts[E,J,18], teams[E,K,7] from simdt SoA state */
  MG_OP_LINEAR_DX         = 29, /* backward dX[M,K] = dY[M,N] @ W[N,K] (+= if u[3]==1) via Metal 4 NAX */
  MG_OP_LINEAR_DW         = 30, /* backward grad_W_f32[N,K] += dY[M,N]^T @ X[M,K] via Metal 4 NAX (FP32 accum!) */
  MG_OP_PLAN_LOSS_BWD     = 31, /* fused PPO/GRPO + entropy + Winnie/Lou + Dynamics loss & head backward */
  MG_OP_BILINEAR_BWD      = 32, /* exact backward of bilinear logits -> dQ, dK, d_log_temp */
  MG_OP_PLAN_SLICE_BWD    = 33, /* merge d_p_h/d_c_h/d_t_h + mp_norm_bwd(x_final) -> d_x */
  MG_OP_MP_NORM_BWD       = 34, /* exact backward of mp_norm (accum_dx optional) */
  MG_OP_MP_LERP_BWD       = 35, /* exact backward of mp_norm(mp_lerp(a, norm_b?mp_norm(b):b, sigmoid(logit))) */
  MG_OP_MP_SWIGLU_BWD     = 36, /* exact backward of mp_silu(gate)*up -> d_gate_up[M, 2*H] */
  MG_OP_CGT_BCAST_BWD     = 37, /* exact backward of cgt_bcast_merge -> d_global_raw, d_tb_raw */
  MG_OP_CGT_RIVAL_BWD     = 38, /* exact backward of cgt_team_rival -> d_qt_raw, d_kt_raw, d_vt_raw */
  MG_OP_CGT_STAGE1_BWD    = 39, /* exact backward of team_pool + global_gram + norm(v,r) -> d_v_raw, d_r_raw, d_xn, d_w_pool */
  MG_OP_PLAN_EMBED_BWD    = 40, /* exact backward of plan_embed_cat -> d_obs/cart/team_enc, d_type_embed, d_type_blend */
  MG_OP_UNIT_W_BWD        = 41, /* tangent projection of FP32 grad_Wu through F.normalize(W) -> grad_W */
  MG_OP_ZERO_F32          = 42, /* fast GPU zero of FP32 buffer */
  MG_OP_FWHT_PACK_I8      = 43, /* Rademacher-FWHT-64 + per-row dynamic int8_t + f32 scale (u[2]=transpose) */
  MG_OP_LINEAR_I8         = 44, /* native Metal 4 MPP int8@int8->int32 NAX GEMM (optional accum=1) */
  MG_OP_SIMDT_REWARD      = 45, /* fused in-GPU CGT step reward + Winnie/Lou role masks */
  MG_OP_OPT_TANGENT_NORM  = 46, /* unit-norm/mp-norm/scalar tangent proj + atomic L2 norm^2 */
  MG_OP_OPT_MOM_PREP      = 47, /* clip + Nesterov mom + optional Aurora D init + BF16 X0 pack */
  MG_OP_OPT_NS_AXPBY      = 48, /* BF16 y = f0*a + f1*b for Newton-Schulz polynomial */
  MG_OP_OPT_AURORA_DREF   = 49, /* Aurora tall-matrix row-leverage D refinement + X0 repack */
  MG_OP_OPT_APPLY_2D      = 50, /* polar step + EDM2 unit-norm + EMA + live BF16 pack + zero gw */
  MG_OP_OPT_APPLY_1D      = 51  /* Adam step for 1D UnitLinear (u[7]=0) or scalar bank (u[7]=1) */
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
int           mesh_graph_import_buffer(mesh_graph_t *g, mesh_graph_t *src_graph, int src_buf_id);
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

/* Execute n_steps inside ONE single MTLCommandBuffer + ONE MTLComputeCommandEncoder (zero per-step CB latency). */
double        mesh_graph_run_fused(mesh_graph_t *g, uint32_t n_steps, uint32_t warm_steps, double *out_wall_ms);

/* Non-blocking single-step submit + wait for Saguaro / SSD async overlap (verifier CB in flight while speculator runs). */
void         *mesh_graph_submit_step(mesh_graph_t *g, uint32_t step_idx);
void          mesh_graph_wait_step(void *cb_token);

void          mesh_graph_barrier(mesh_graph_t *g);
uint32_t      mesh_graph_node_id(mesh_graph_t *g);

#ifdef __cplusplus
}
#endif
#endif
