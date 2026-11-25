#ifndef MESH_CODEC_H
#define MESH_CODEC_H

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Unified interconnect/SHM compression codecs & Pareto 3D lower-hull selector.
   Consolidated from metal-microbench/had_int8_metal/pareto_codec_selector.h and
   sm120-chudlets/fabric/python/sm120_act_router.py for Apple Silicon M5 + TB5/TB4 flat-page RDMA. */

enum mesh_codec_id {
  MESH_CODEC_BF16              = 0,  /* Uncompressed FP16/BF16 (16.00 bpe) */
  MESH_CODEC_HAD_INT8          = 1,  /* FWHT-64 + symmetric INT8, 2 fp16 scales/64 elems (68 B/64, 8.50 bpe) */
  MESH_CODEC_HAD_INT6_PK       = 2,  /* FWHT-64 + bit-packed INT6, 1 fp32 scale/64 elems (52 B/64, 6.50 bpe) */
  MESH_CODEC_HAD_INT6_ANS      = 3,  /* FWHT-64 + INT6 + 32-lane rANS (5.28 bpe) */
  MESH_CODEC_CHSPLIT_ANS       = 4,  /* Channel-split + 32-lane rANS (3.55 bpe) */
  MESH_CODEC_MST_MEALY         = 5,  /* 2D MST Mealy-Machine 2-context rANS (3.20 bpe) */
  MESH_CODEC_RVQ4_LUT2S        = 6,  /* AOT 2-stage RVQ-4 TG-SRAM LUT (4.00 bpe) */
  MESH_CODEC_RVQ4_B8_3S        = 7,  /* AOT 3-stage RVQ-4 (7.25 bpe) */
  MESH_CODEC_KALOMAZE_27B_DRAM = 8,  /* Kalomaze 27-bit Dual-RVQ DRAM body decode (3.75 bpe) */
  MESH_CODEC_KALOMAZE_27B_ADC  = 9,  /* Kalomaze 27-bit Dual-RVQ zero-DRAM-decompress ADC-LUT Q*K^T (3.75 bpe) */
  MESH_CODEC_KALOMAZE_27B_G4   = 10  /* Kalomaze 27-bit G=4 prefix-copy ADC KV (2.52 bpe) */
};

enum mesh_codec_workload {
  MESH_WORKLOAD_ACT = 0,  /* Activations / collectives / MoE */
  MESH_WORKLOAD_KV  = 1   /* Disaggregated KV-cache pages */
};

enum mesh_codec_mode {
  MESH_CODEC_SYNC  = 0,   /* Immediate synchronous (same command buffer / stream) */
  MESH_CODEC_ASYNC = 1    /* Slack-overlapped pipelined (hides encode behind compute slack_us) */
};

/* Self-contained 4-byte-aligned wire blocks packed inside 4096-byte /mesh0 pages.
   Because FWHT-64 (Walsh-Hadamard over GF(2)^6) is linear, the RDMA switch reduces these
   blocks directly in the Hadamard domain via NEON without ever running an inverse FWHT! */
#define MESH_HAD_I8_BLK_ELEMS   64u
#define MESH_HAD_I8_BLK_BYTES   68u
#define MESH_HAD_I8_BLKS_PAGE   60u  /* 60 * 68 = 4080 B <= 4096 B (3840 elems/page) */

#define MESH_HAD_I6_BLK_ELEMS   64u
#define MESH_HAD_I6_BLK_BYTES   52u
#define MESH_HAD_I6_BLKS_PAGE   78u  /* 78 * 52 = 4056 B <= 4096 B (4992 elems/page) */

struct mesh_had_i8_blk {
  uint16_t s_f16[2];  /* IEEE-754 binary16 scales for q[0..31] and q[32..63] */
  int8_t   q[64];     /* symmetric quantized FWHT-64 coeffs in [-127, 127] */
};

struct mesh_had_i6_blk {
  float    s;         /* FP32 scale for all 64 quantized coeffs in [-31, 31] */
  uint32_t w[12];     /* 4 groups x 3 u32 = 384 bits = 64 x 6-bit unsigned (v + 31) */
};

_Static_assert(sizeof(struct mesh_had_i8_blk) == MESH_HAD_I8_BLK_BYTES, "mesh_had_i8_blk must be 68 B");
_Static_assert(sizeof(struct mesh_had_i6_blk) == MESH_HAD_I6_BLK_BYTES, "mesh_had_i6_blk must be 52 B");

struct mesh_codec_spec {
  enum mesh_codec_id id;
  float bpe_inf;
  float hdr_bytes_fixed;
  float enc_gbps_bf16;
  float dec_gbps_bf16;
  float alpha_enc_us;
  float alpha_dec_us;
  float kl_act;
  float kl_kv;
  bool  requires_aot;
};

struct mesh_codec_decision {
  enum mesh_codec_id id;
  enum mesh_codec_mode mode;
  float bpe;
  uint32_t num_chunks;
  bool preserve_bos;
  double predicted_us;
};

/* Empirically calibrated M5 silicon hull (metal-microbench/had_int8_metal/pareto_codec_selector.h). */
static const struct mesh_codec_spec MESH_M5_CODEC_TABLE[] = {
  {MESH_CODEC_BF16,              16.00f,   0.0f,   0.0f,   0.0f,  0.0f,  0.0f, 0.00000f, 0.00000f, false},
  {MESH_CODEC_HAD_INT8,           8.50f,   0.0f, 238.3f, 262.1f,  3.0f,  3.0f, 0.00103f, 0.00083f, false},
  {MESH_CODEC_HAD_INT6_PK,        6.50f,   0.0f, 117.4f, 231.9f,  3.0f,  3.0f, 0.00420f, 0.00150f, false},
  {MESH_CODEC_HAD_INT6_ANS,       5.28f, 160.0f, 118.5f,  83.3f,  7.0f,  7.0f, 0.00420f, 0.00150f, true},
  {MESH_CODEC_CHSPLIT_ANS,        3.55f, 224.0f,  83.3f,  88.3f,  9.0f,  9.0f, 0.00450f, 0.00200f, false},
  {MESH_CODEC_MST_MEALY,          3.20f, 320.0f,  53.0f,  33.5f, 14.0f, 14.0f, 0.02500f, 0.00669f, false},
  {MESH_CODEC_RVQ4_LUT2S,         4.00f,   0.0f,   2.0f, 551.8f, 12.0f,  5.0f, 0.03000f, 0.01176f, true},
  {MESH_CODEC_RVQ4_B8_3S,         7.25f,   0.0f,   1.5f, 175.0f, 30.0f, 16.0f, 0.00900f, 0.00263f, true},
  {MESH_CODEC_KALOMAZE_27B_DRAM,  3.75f,   0.0f,   2.0f, 245.0f, 14.0f,  6.0f, 0.02500f, 0.00633f, true},
  {MESH_CODEC_KALOMAZE_27B_ADC,   3.75f,   0.0f,   2.0f, 810.3f, 14.0f,  6.0f, 0.02500f, 0.00633f, true},
  {MESH_CODEC_KALOMAZE_27B_G4,    2.52f,   0.0f,   2.0f, 810.3f, 14.0f,  6.0f, 0.02500f, 0.00633f, true},
};

static inline double mesh_codec_time_us(
    const struct mesh_codec_spec *c,
    uint64_t nbytes_bf16,
    double link_gbps,
    double slack_us,
    bool pipelined_rx,
    double rtt_us) {
  const double bf16_bytes = (double)nbytes_bf16;
  const double t_enc = (c->enc_gbps_bf16 > 0.0f)
      ? ((double)c->alpha_enc_us + (bf16_bytes / ((double)c->enc_gbps_bf16 * 1e9)) * 1e6)
      : 0.0;
  const double t_dec = (c->dec_gbps_bf16 > 0.0f)
      ? ((double)c->alpha_dec_us + (bf16_bytes / ((double)c->dec_gbps_bf16 * 1e9)) * 1e6)
      : 0.0;
  const double wire_bytes = (double)c->hdr_bytes_fixed + bf16_bytes * ((double)c->bpe_inf / 16.0);
  const double link_Bps = (link_gbps > 0.0 ? link_gbps : 1.0) * 1e9 / 8.0;
  const double t_wire = (wire_bytes / link_Bps) * 1e6;
  const double exposed_enc = (t_enc > slack_us) ? (t_enc - slack_us) : 0.0;
  if (pipelined_rx) {
    const double rx = (t_wire > t_dec) ? t_wire : t_dec;
    return rtt_us + exposed_enc + rx;
  }
  return rtt_us + exposed_enc + t_wire + t_dec;
}

/* Pure deterministic Sync vs Async Pareto codec selector with <=16 KiB crossover & early-layer guard. */
static inline struct mesh_codec_decision mesh_select_codec(
    const struct mesh_codec_spec *table,
    size_t count,
    uint64_t nbytes_bf16,
    enum mesh_codec_mode mode,
    enum mesh_codec_workload workload,
    double link_gbps,
    double max_kl,
    double slack_us,
    int layer_idx,
    int n_layers,
    bool allow_aot,
    bool allow_error_feedback,
    bool preserve_bos) {
  struct mesh_codec_decision dec = {
    .id = MESH_CODEC_BF16,
    .mode = mode,
    .bpe = 16.0f,
    .num_chunks = 1u,
    .preserve_bos = false,
    .predicted_us = 0.0
  };
  const double link_Bps = (link_gbps > 0.0 ? link_gbps : 1.0) * 1e9 / 8.0;
  const bool early_sensitive = (layer_idx >= 0 && n_layers >= 6 && layer_idx < (n_layers / 3));

  /* Small payload crossover: <=16 KiB stays exact uncompressed to avoid kernel launch tax. */
  if (nbytes_bf16 <= 16384u) {
    dec.predicted_us = ((double)nbytes_bf16 / link_Bps) * 1e6;
    return dec;
  }
  /* Depth-Jacobian early-layer guard under strict KL budget (without error feedback). */
  if (workload == MESH_WORKLOAD_ACT && early_sensitive && max_kl <= 0.015 && !allow_error_feedback) {
    dec.predicted_us = ((double)nbytes_bf16 / link_Bps) * 1e6;
    return dec;
  }

  const bool pipelined = (mode == MESH_CODEC_ASYNC);
  const double eff_slack = pipelined ? slack_us : 0.0;
  const double kl_scale = (workload == MESH_WORKLOAD_ACT && layer_idx >= 0 && !early_sensitive) ? 0.189 : 1.0;

  double best_t = DBL_MAX;
  for (size_t i = 0; i < count; i++) {
    const struct mesh_codec_spec *c = &table[i];
    const double kl = (workload == MESH_WORKLOAD_ACT ? (double)c->kl_act : (double)c->kl_kv) * kl_scale;
    if (!allow_error_feedback && kl > max_kl) continue;
    if (c->requires_aot && !allow_aot) continue;
    const double t = mesh_codec_time_us(c, nbytes_bf16, link_gbps, eff_slack, pipelined, 0.0);
    if (t < best_t) {
      best_t = t;
      dec.id = c->id;
      dec.bpe = c->bpe_inf;
      dec.predicted_us = t;
    }
  }
  dec.num_chunks = (pipelined && nbytes_bf16 >= 524288u && dec.id != MESH_CODEC_BF16) ? 4u : 1u;
  dec.preserve_bos = preserve_bos && (dec.id != MESH_CODEC_BF16);
  return dec;
}

#ifdef __cplusplus
}
#endif
#endif
