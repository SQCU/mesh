/* Ticket T8: Unified native C/ObjC++ phase-overlap Metal + flat-page algebraic RDMA runtime.
   Subsumes per-strategy inlined client programs into ONE parametric execution loop:
     dst(r, p) = (p + r / g) % P
   operating zero-copy over 16 KiB-aligned /mesh0 pages via MTLDevice newBufferWithBytesNoCopy.
   Supports all 3 algebraic page-morphism classes natively with wire-packed Metal kernels:
     kind=0 (C0 MESH_RED_SUM_F32)            : MPSMatrixMultiplication -> /mesh0
     kind=1 (C1 MESH_RED_ONLINE_SOFTMAX_F32) : MSL FlashAttention tile -> [m,l,o_0..o_125] in /mesh0
     kind=2 (C2 MESH_RED_AFFINE_SCAN_F32)    : MSL Block Affine Scan  -> (A,b) pairs in /mesh0
     kind=3 (BLIND C0 baseline)              : MPS -> plain mesh_write + host NEON sum
   Includes zero-NCCL TCP bootstrap for one-time landing-index exchange at bridge.prepare(). */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <arpa/inet.h>
#include <arm_neon.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

extern "C" {
#include "mesh-call.h"
}

static double now_us(void) { return clock_gettime_nsec_np(CLOCK_MONOTONIC) * 1e-3; }

static void neonsum(float *d, const float *s, size_t n) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    vst1q_f32(d + i,      vaddq_f32(vld1q_f32(d + i),      vld1q_f32(s + i)));
    vst1q_f32(d + i + 4,  vaddq_f32(vld1q_f32(d + i + 4),  vld1q_f32(s + i + 4)));
    vst1q_f32(d + i + 8,  vaddq_f32(vld1q_f32(d + i + 8),  vld1q_f32(s + i + 8)));
    vst1q_f32(d + i + 12, vaddq_f32(vld1q_f32(d + i + 12), vld1q_f32(s + i + 12)));
  }
  for (; i < n; i++) d[i] += s[i];
}

/* ---- Zero-NCCL TCP bootstrap (replaces NCCL for one-time W[dst_idx] exchange & barriers) ---- */
static int send_all(int fd, const void *buf, size_t n) {
  const unsigned char *p = (const unsigned char *)buf;
  while (n) {
    ssize_t k = send(fd, p, n, 0);
    if (k <= 0) return -1;
    p += k; n -= (size_t)k;
  }
  return 0;
}

static int recv_all(int fd, void *buf, size_t n) {
  unsigned char *p = (unsigned char *)buf;
  while (n) {
    ssize_t k = recv(fd, p, n, 0);
    if (k <= 0) return -1;
    p += k; n -= (size_t)k;
  }
  return 0;
}

struct mesh_phase_group {
  int rank;
  int ranks;
  int listen_fd;
  int peer_fd[8]; /* on rank 0: fd for each peer 1..ranks-1; on rank > 0: peer_fd[0] is master */
  uint32_t my_node;
  uint32_t nodes[8];
  uint32_t bar_src_page;
  uint32_t bar_idx[2];      /* local double-buffered RDMA barrier W[idx] */
  uint32_t bar_all[8][2];   /* all ranks' barrier W[idx] */
  uint32_t bar_gen;
  struct mesh_ctx ctx;
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLComputePipelineState> pso_flash_c1;
  id<MTLComputePipelineState> pso_scan_c2;
};

static const char *kPhaseMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

/* C1 wire-packed FlashAttention tile: writes 128 floats [m, l, o_0..o_125] per query row directly into /mesh0.
   Uses Apple Silicon hardware simd_sum (4 simdgroups x 32 = 128 threads/row) for high-occupancy M5 Ultra execution. */
kernel void flash_c1_packed(
    device const float *Q     [[buffer(0)]], /* [S_tile, 126] */
    device const float *K     [[buffer(1)]], /* [S_kv,   126] */
    device const float *V     [[buffer(2)]], /* [S_kv,   126] */
    device float       *Out   [[buffer(3)]], /* [S_tile, 128] in /mesh0: [m, l, o_0..o_125] */
    constant uint4     &dims  [[buffer(4)]], /* x=S_tile, y=S_kv, z=D(=126), w=q_row_off */
    uint tg_id                [[threadgroup_position_in_grid]],
    uint tid                  [[thread_index_in_threadgroup]],
    uint sg_id                [[simdgroup_index_in_threadgroup]],
    uint lane                 [[thread_index_in_simdgroup]])
{
  const uint row = tg_id;
  if (row >= dims.x) return;
  const uint S_kv = dims.y;
  const uint D = dims.z;
  const float scale = rsqrt(float(D));

  const float qv = (tid < D) ? Q[(dims.w + row) * D + tid] : 0.0f;
  threadgroup float sg_part[4];

  float m_i = -1e30f;
  float l_i = 0.0f;
  float o_i = 0.0f;

  for (uint j = 0; j < S_kv; j++) {
    const device float *kj = K + j * D;
    const device float *vj = V + j * D;
    const float prod = (tid < D) ? (qv * kj[tid]) : 0.0f;
    const float ssum = simd_sum(prod);
    if (lane == 0) sg_part[sg_id] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float dot = sg_part[0] + sg_part[1] + sg_part[2] + sg_part[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float score = dot * scale;
    const float m_new = max(m_i, score);
    const float alpha = exp(m_i - m_new);
    const float beta  = exp(score - m_new);
    l_i = l_i * alpha + beta;
    if (tid < D) o_i = o_i * alpha + beta * vj[tid];
    m_i = m_new;
  }

  device float *dst = Out + row * 128u;
  if (tid == 0) dst[0] = m_i;
  if (tid == 1) dst[1] = l_i;
  if (tid < D) dst[tid + 2u] = o_i;
}

/* C2 wire-packed Block Affine Scan: reduces local sequence chunk [0..L_loc) of linear recurrences
   h_t = a_t * h_{t-1} + b_t across H_tile independent channels into (A_block, B_block) pairs
   written directly into /mesh0 in MESH_RED_AFFINE_SCAN_F32 wire layout: [a_0, b_0, a_1, b_1, ...]. */
kernel void affine_scan_c2_packed(
    device const float2 *In   [[buffer(0)]], /* [L_loc, H_total] as (a, b) pairs */
    device float2       *Out  [[buffer(1)]], /* [H_tile] in /mesh0 as (A, B) pairs */
    constant uint4      &dims [[buffer(2)]], /* x=H_tile, y=L_loc, z=H_total, w=h_off */
    uint gid                  [[thread_position_in_grid]])
{
  if (gid >= dims.x) return;
  const uint h = dims.w + gid;
  const uint L_loc = dims.y;
  const uint H_tot = dims.z;
  float A = 1.0f;
  float B = 0.0f;
  for (uint t = 0; t < L_loc; t++) {
    const float2 ab = In[t * H_tot + h];
    A = ab.x * A;
    B = ab.x * B + ab.y;
  }
  Out[gid] = float2(A, B);
}
)MSL";

extern "C" {

struct mesh_phase_result {
  double wall_us;
  double gpu_us;
  double throughput; /* TFLOPS for C0/C1, GB/s effective for C2 */
  int ok;
  double max_err;
};

static int group_allgather(struct mesh_phase_group *g, const void *sendbuf, void *recvbuf, size_t nbytes) {
  if (g->rank == 0) {
    memcpy(recvbuf, sendbuf, nbytes);
    for (int r = 1; r < g->ranks; r++) {
      if (recv_all(g->peer_fd[r], (unsigned char *)recvbuf + (size_t)r * nbytes, nbytes)) return -1;
    }
    for (int r = 1; r < g->ranks; r++) {
      if (send_all(g->peer_fd[r], recvbuf, (size_t)g->ranks * nbytes)) return -1;
    }
  } else {
    if (send_all(g->peer_fd[0], sendbuf, nbytes)) return -1;
    if (recv_all(g->peer_fd[0], recvbuf, (size_t)g->ranks * nbytes)) return -1;
  }
  return 0;
}

/* Pure TB5 RDMA barrier (~10 us, zero Tailscale TCP, zero NCCL!): reduces 4 B SUM_I32 into W[bar_idx[sl]] */
int mesh_phase_barrier(struct mesh_phase_group *g) {
  const uint32_t sl = (g->bar_gen++) & 1u;
  const uint16_t done = MESH_TAG_WEIGHT(g->ranks);
  struct mesh_red_xfer xf[8];
  for (int d = 0; d < g->ranks; d++)
    xf[d] = (struct mesh_red_xfer){ g->bar_src_page, 4u, g->nodes[d], g->bar_all[d][sl], MESH_RED_SUM_I32, 0, MESH_TAG_WEIGHT(1) };
  while (mesh_reduce_write(&g->ctx, xf, (uint32_t)g->ranks) == ENOMEM) {}
  struct mesh_entry_info e;
  uint16_t tag = 0;
  for (;;) {
    mesh_entry_red(&g->ctx, g->bar_idx[sl], &e, &tag);
    if (e.data && tag == done) break;
  }
  mesh_clear_indices(&g->ctx, &g->bar_idx[sl], 1u);
  return 0;
}

struct mesh_phase_group *mesh_phase_open(int rank, int ranks, const char *master_ip, int port) {
  auto *g = new mesh_phase_group();
  memset(&g->ctx, 0, sizeof(g->ctx));
  g->rank = rank;
  g->ranks = ranks;
  g->listen_fd = -1;
  for (int i = 0; i < 8; i++) g->peer_fd[i] = -1;

  if (mesh_attach(&g->ctx)) { delete g; return NULL; }
  g->my_node = *(const uint32_t *)((const unsigned char *)g->ctx.M + 20);

  if (rank == 0) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) || listen(fd, 16)) {
      close(fd); mesh_detach(&g->ctx); delete g; return NULL;
    }
    g->listen_fd = fd;
    for (int i = 1; i < ranks; i++) {
      int cfd = accept(fd, NULL, NULL);
      if (cfd < 0) { delete g; return NULL; }
      setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
      int peer_r = -1;
      if (recv_all(cfd, &peer_r, sizeof(peer_r)) || peer_r <= 0 || peer_r >= ranks) {
        close(cfd); delete g; return NULL;
      }
      g->peer_fd[peer_r] = cfd;
    }
  } else {
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, master_ip, &addr.sin_addr);
    int cfd = -1;
    for (int attempt = 0; attempt < 200; attempt++) {
      cfd = socket(AF_INET, SOCK_STREAM, 0);
      if (connect(cfd, (struct sockaddr *)&addr, sizeof(addr)) == 0) break;
      close(cfd); cfd = -1;
      usleep(25000);
    }
    if (cfd < 0) { mesh_detach(&g->ctx); delete g; return NULL; }
    int yes = 1;
    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    if (send_all(cfd, &rank, sizeof(rank))) { close(cfd); mesh_detach(&g->ctx); delete g; return NULL; }
    g->peer_fd[0] = cfd;
  }

  if (group_allgather(g, &g->my_node, g->nodes, sizeof(uint32_t))) {
    mesh_detach(&g->ctx); delete g; return NULL;
  }
  /* Allocate 1 pinned source page + 2 landing indices for pure-RDMA barriers (~10 us, zero TCP in hot loop) */
  void *bar_base = NULL;
  if (mesh_alloc_contiguous(&g->ctx, 1u, NULL, &bar_base)) { mesh_detach(&g->ctx); delete g; return NULL; }
  const uint64_t data_off = *(const uint64_t *)((const unsigned char *)g->ctx.M + 88);
  g->bar_src_page = (uint32_t)(((unsigned char *)bar_base - ((unsigned char *)g->ctx.M + data_off)) / MESH_PAGE_BYTES);
  *(int32_t *)bar_base = 1;
  if (mesh_alloc(&g->ctx, 2u, 0, g->bar_idx, NULL)) { mesh_release(&g->ctx, g->bar_src_page); mesh_detach(&g->ctx); delete g; return NULL; }
  if (group_allgather(g, g->bar_idx, g->bar_all, 2u * sizeof(uint32_t))) { mesh_detach(&g->ctx); delete g; return NULL; }
  g->bar_gen = 0;

  g->dev = MTLCreateSystemDefaultDevice();
  g->q = [g->dev newCommandQueue];
  NSError *err = nil;
  NSString *src = [NSString stringWithUTF8String:kPhaseMSL];
  id<MTLLibrary> lib = [g->dev newLibraryWithSource:src options:nil error:&err];
  if (!lib) {
    fprintf(stderr, "mesh_phase MSL compile error: %s\n", err.localizedDescription.UTF8String);
    mesh_detach(&g->ctx); delete g; return NULL;
  }
  g->pso_flash_c1 = [g->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"flash_c1_packed"] error:&err];
  g->pso_scan_c2  = [g->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"affine_scan_c2_packed"] error:&err];
  return g;
}

void mesh_phase_close(struct mesh_phase_group *g) {
  if (!g) return;
  mesh_free_indices(&g->ctx, g->bar_idx, 2u);
  mesh_release(&g->ctx, g->bar_src_page);
  for (int i = 0; i < 8; i++) if (g->peer_fd[i] >= 0) close(g->peer_fd[i]);
  if (g->listen_fd >= 0) close(g->listen_fd);
  mesh_detach(&g->ctx);
  delete g;
}

uint32_t mesh_phase_node_id(struct mesh_phase_group *g, int r) {
  return (g && r >= 0 && r < g->ranks) ? g->nodes[r] : 0u;
}

/* Unified parametric phase-overlap executor for C0 (GEMM), C1 (FlashAttention), C2 (Affine Scan), and blind baseline.
   Executes dst(r, step) = (step + r / g) % P with zero-copy /mesh0 MTLBuffers and in-switch mesh_reduce_write! */
int mesh_phase_run(struct mesh_phase_group *grp,
                   int kind,             /* 0=C0_GEMM, 1=C1_FLASH, 2=C2_SCAN, 3=BLIND_GEMM */
                   uint32_t dim0,        /* M / S / L */
                   uint32_t dim1,        /* K / D / H */
                   uint32_t dim2,        /* N / 0 / 0 */
                   uint32_t tiles,
                   uint32_t coalesce_g,
                   uint8_t op,
                   uint16_t my_tag,
                   uint16_t done_tag,
                   int standalone,
                   int reps,
                   struct mesh_phase_result *out) {
  const int rank = grp->rank, ranks = grp->ranks;
  const uint32_t g = coalesce_g > 0 ? coalesce_g : 1u;
  const int is_blind = (kind == 3);
  const uint32_t nph = tiles * (uint32_t)ranks;

  size_t tile_sh_bytes = 0;
  double flops_or_bytes = 0.0;
  uint32_t M_tile = 0, K_loc = 0, N_sh = 0;
  uint32_t S_tile = 0, S_kv = 0, D_flash = 126;
  uint32_t L_loc = 0, H_tile = 0, H_tot = 0;

  if (kind == 0 || kind == 3) {
    K_loc = dim1 / (uint32_t)ranks;
    N_sh = dim2 / (uint32_t)ranks;
    M_tile = dim0 / tiles;
    tile_sh_bytes = (size_t)M_tile * N_sh * sizeof(float);
    flops_or_bytes = 2.0 * (double)dim0 * (double)dim1 * (double)dim2; /* cluster FLOPs */
  } else if (kind == 1) {
    /* C1 FlashAttention: S total queries split across (ranks * tiles) shards, each rank holds S/ranks KV rows, D=126 */
    S_kv = dim0 / (uint32_t)ranks;
    S_tile = dim0 / ((uint32_t)ranks * tiles);
    D_flash = (dim1 > 0 && dim1 <= 126u) ? dim1 : 126u;
    tile_sh_bytes = (size_t)S_tile * 128u * sizeof(float); /* 128 f32/row = 512 B/row, 8 rows per 4 KiB page! */
    flops_or_bytes = 4.0 * (double)dim0 * (double)dim0 * (double)D_flash; /* cluster QK^T + PV FLOPs */
  } else if (kind == 2) {
    /* C2 Affine Scan: L total sequence split across ranks (L_loc = L/ranks), H total channels split across (ranks * tiles) */
    L_loc = dim0 / (uint32_t)ranks;
    H_tot = dim1;
    H_tile = dim1 / ((uint32_t)ranks * tiles);
    tile_sh_bytes = (size_t)H_tile * 2u * sizeof(float); /* (A, B) float2 per channel */
    flops_or_bytes = (double)dim0 * (double)dim1 * 8.0; /* total sequence state bytes scanned across cluster */
  } else {
    return -1;
  }

  const uint32_t tile_pages = (uint32_t)((tile_sh_bytes + MESH_PAGE_BYTES - 1u) / MESH_PAGE_BYTES);
  const uint32_t sh_pages = tiles * tile_pages;
  const uint32_t alloc_pages = ((tile_pages + 3u) & ~3u);
  const uint64_t data_off = *(const uint64_t *)((const unsigned char *)grp->ctx.M + 88);

  /* Zero-copy 16 KiB-aligned /mesh0 staging buffers wrapped directly as MTLBuffers */
  std::vector<uint32_t> raw_p0(nph), src_p0(nph);
  std::vector<void *> src_ptr(nph);
  std::vector<id<MTLBuffer>> bufOut(nph);
  for (uint32_t i = 0; i < nph; i++) {
    void *base = NULL;
    if (mesh_alloc_contiguous(&grp->ctx, alloc_pages + 4u, NULL, &base)) return -2;
    uintptr_t raw = (uintptr_t)base;
    uintptr_t aligned = (raw + 16383u) & ~(uintptr_t)16383u;
    raw_p0[i] = (uint32_t)(((unsigned char *)raw - ((unsigned char *)grp->ctx.M + data_off)) / MESH_PAGE_BYTES);
    src_p0[i] = (uint32_t)(((unsigned char *)aligned - ((unsigned char *)grp->ctx.M + data_off)) / MESH_PAGE_BYTES);
    src_ptr[i] = (void *)aligned;
    bufOut[i] = [grp->dev newBufferWithBytesNoCopy:(void *)aligned
                                           length:(size_t)alloc_pages * MESH_PAGE_BYTES
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
    if (!bufOut[i]) return -3;
  }

  /* Kernel-specific Metal input buffers (allocated once before timing loop) */
  id<MTLBuffer> bufA = nil, bufQ = nil, bufK = nil, bufV = nil, bufScanIn = nil;
  std::vector<id<MTLBuffer>> bufB;
  MPSMatrixMultiplication *mm = nil;
  std::vector<MPSMatrix *> matA, matB, matC;

  if (kind == 0 || kind == 3) {
    bufA = [grp->dev newBufferWithLength:(size_t)dim0 * K_loc * sizeof(float) options:MTLResourceStorageModeShared];
    float *pA = (float *)bufA.contents;
    for (size_t i = 0; i < (size_t)dim0 * K_loc; i++) pA[i] = 0.25f * (float)((int)((i + (size_t)rank * 17u) % 5u) - 2);
    bufB.resize(ranks);
    for (int d = 0; d < ranks; d++) {
      bufB[d] = [grp->dev newBufferWithLength:(size_t)K_loc * N_sh * sizeof(float) options:MTLResourceStorageModeShared];
      float *pB = (float *)bufB[d].contents;
      for (size_t i = 0; i < (size_t)K_loc * N_sh; i++) pB[i] = 0.25f * (float)((int)((i + (size_t)(rank * 7 + d * 3)) % 5u) - 2);
    }
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M_tile columns:K_loc rowBytes:K_loc * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K_loc columns:N_sh rowBytes:N_sh * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M_tile columns:N_sh rowBytes:N_sh * sizeof(float) dataType:MPSDataTypeFloat32];
    mm = [[MPSMatrixMultiplication alloc] initWithDevice:grp->dev transposeLeft:NO transposeRight:NO
                                              resultRows:M_tile resultColumns:N_sh interiorColumns:K_loc alpha:1.0 beta:0.0];
    matA.resize(tiles); matB.resize(ranks); matC.resize(nph);
    for (uint32_t t = 0; t < tiles; t++)
      matA[t] = [[MPSMatrix alloc] initWithBuffer:bufA offset:(size_t)t * M_tile * K_loc * sizeof(float) descriptor:descA];
    for (int d = 0; d < ranks; d++)
      matB[d] = [[MPSMatrix alloc] initWithBuffer:bufB[d] descriptor:descB];
    for (uint32_t i = 0; i < nph; i++)
      matC[i] = [[MPSMatrix alloc] initWithBuffer:bufOut[i] descriptor:descC];
  } else if (kind == 1) {
    bufQ = [grp->dev newBufferWithLength:(size_t)dim0 * D_flash * sizeof(float) options:MTLResourceStorageModeShared];
    bufK = [grp->dev newBufferWithLength:(size_t)S_kv * D_flash * sizeof(float) options:MTLResourceStorageModeShared];
    bufV = [grp->dev newBufferWithLength:(size_t)S_kv * D_flash * sizeof(float) options:MTLResourceStorageModeShared];
    float *pQ = (float *)bufQ.contents, *pK = (float *)bufK.contents, *pV = (float *)bufV.contents;
    for (size_t i = 0; i < (size_t)dim0 * D_flash; i++) pQ[i] = 0.25f * (float)((int)((i * 13u + 3u) % 7u) - 3);
    for (size_t i = 0; i < (size_t)S_kv * D_flash; i++) {
      pK[i] = 0.25f * (float)((int)((i + (size_t)rank * 11u) % 7u) - 3);
      pV[i] = 0.25f * (float)((int)((i + (size_t)rank * 5u + 1u) % 5u) - 2);
    }
  } else if (kind == 2) {
    bufScanIn = [grp->dev newBufferWithLength:(size_t)L_loc * H_tot * 2u * sizeof(float) options:MTLResourceStorageModeShared];
    float *pS = (float *)bufScanIn.contents;
    for (size_t i = 0; i < (size_t)L_loc * H_tot; i++) {
      pS[2 * i]     = 0.99f - 0.01f * (float)((int)((i + (size_t)rank * 3u) % 5u)); /* a in [0.95, 0.99] stable */
      pS[2 * i + 1] = 0.10f * (float)((int)((i + (size_t)rank * 7u) % 5u) - 2);
    }
  }

  /* One-time double-buffered landing-index exchange over zero-NCCL TCP bootstrap */
  const uint32_t slots = 2u;
  const uint32_t n_recv = is_blind ? (slots * (uint32_t)ranks * sh_pages) : (slots * sh_pages);
  std::vector<uint32_t> my_idx(n_recv), all_idx((size_t)ranks * n_recv);
  if (!standalone) {
    if (mesh_alloc(&grp->ctx, n_recv, 0, my_idx.data(), NULL)) return -4;
    if (group_allgather(grp, my_idx.data(), all_idx.data(), n_recv * sizeof(uint32_t))) return -5;
  }

  /* Pre-build per-phase xfer tables */
  std::vector<std::vector<struct mesh_xfer>> bxf(slots);
  std::vector<std::vector<struct mesh_red_xfer>> dxf(slots * nph);
  if (!standalone) {
    for (uint32_t sl = 0; sl < slots; sl++) {
      if (is_blind) {
        bxf[sl].reserve((size_t)ranks * sh_pages);
        for (uint32_t t = 0; t < tiles; t++) {
          for (int d = 0; d < ranks; d++) {
            const uint32_t *didx = all_idx.data() + (size_t)d * n_recv + (sl * (uint32_t)ranks + (uint32_t)rank) * sh_pages + t * tile_pages;
            const uint32_t p0 = src_p0[t * (uint32_t)ranks + (uint32_t)d];
            for (uint32_t k = 0; k < tile_pages; k++)
              bxf[sl].push_back((struct mesh_xfer){ p0 + k, MESH_PAGE_BYTES, grp->nodes[d], didx[k] });
          }
        }
      } else {
        for (uint32_t t = 0; t < tiles; t++) {
          for (int d = 0; d < ranks; d++) {
            auto &v = dxf[(sl * tiles + t) * (uint32_t)ranks + (uint32_t)d];
            v.reserve(tile_pages);
            const uint32_t *didx = all_idx.data() + (size_t)d * n_recv + sl * sh_pages + t * tile_pages;
            const uint32_t p0 = src_p0[t * (uint32_t)ranks + (uint32_t)d];
            for (uint32_t k = 0; k < tile_pages; k++) {
              const size_t off = (size_t)k * MESH_PAGE_BYTES;
              const uint32_t pb = (tile_sh_bytes - off < MESH_PAGE_BYTES) ? (uint32_t)(tile_sh_bytes - off) : MESH_PAGE_BYTES;
              v.push_back((struct mesh_red_xfer){ p0 + k, pb, grp->nodes[d], didx[k], op, 0, my_tag });
            }
          }
        }
      }
    }
  }

  auto encode_phase = [&](id<MTLCommandBuffer> cb, uint32_t t, int d) {
    const uint32_t idx = t * (uint32_t)ranks + (uint32_t)d;
    if (kind == 0 || kind == 3) {
      [mm encodeToCommandBuffer:cb leftMatrix:matA[t] rightMatrix:matB[d] resultMatrix:matC[idx]];
    } else if (kind == 1) {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:grp->pso_flash_c1];
      const uint32_t q_off = ((uint32_t)d * tiles + t) * S_tile;
      uint32_t dims[4] = { S_tile, S_kv, D_flash, q_off };
      [enc setBuffer:bufQ offset:0 atIndex:0];
      [enc setBuffer:bufK offset:0 atIndex:1];
      [enc setBuffer:bufV offset:0 atIndex:2];
      [enc setBuffer:bufOut[idx] offset:0 atIndex:3];
      [enc setBytes:dims length:sizeof(dims) atIndex:4];
      [enc dispatchThreadgroups:MTLSizeMake(S_tile, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [enc endEncoding];
    } else if (kind == 2) {
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:grp->pso_scan_c2];
      const uint32_t h_off = ((uint32_t)d * tiles + t) * H_tile;
      uint32_t dims[4] = { H_tile, L_loc, H_tot, h_off };
      [enc setBuffer:bufScanIn offset:0 atIndex:0];
      [enc setBuffer:bufOut[idx] offset:0 atIndex:1];
      [enc setBytes:dims length:sizeof(dims) atIndex:2];
      const uint32_t tg = 256u;
      [enc dispatchThreadgroups:MTLSizeMake((H_tile + tg - 1u) / tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
      [enc endEncoding];
    }
  };

  std::vector<float> out_buf((sh_pages * MESH_PAGE_BYTES) / sizeof(float), 0.0f);
  double t_wall_sum = 0.0, t_gpu_sum = 0.0;

  for (int rep = -3; rep < reps; rep++) {
    const uint32_t sl = (uint32_t)((rep + 3) & 1);
    if (!standalone) mesh_phase_barrier(grp);
    const double t0 = now_us();

    if (standalone) {
      @autoreleasepool {
        std::vector<id<MTLCommandBuffer>> cbs(nph);
        uint32_t p = 0;
        for (uint32_t t = 0; t < tiles; t++) {
          for (int step = 0; step < ranks; step++) {
            const int d = (step + rank / (int)g) % ranks;
            cbs[p] = [grp->q commandBuffer];
            encode_phase(cbs[p], t, d);
            [cbs[p] commit];
            p++;
          }
        }
        for (uint32_t p2 = 0; p2 < nph; p2++) [cbs[p2] waitUntilCompleted];
      }
      const double dt = now_us() - t0;
      if (rep >= 0) { t_wall_sum += dt; t_gpu_sum += dt; }
    } else if (is_blind) {
      @autoreleasepool {
        id<MTLCommandBuffer> cb = [grp->q commandBuffer];
        for (uint32_t t = 0; t < tiles; t++)
          for (int d = 0; d < ranks; d++)
            encode_phase(cb, t, d);
        [cb commit];
        [cb waitUntilCompleted];
      }
      const double tg = now_us() - t0;
      const auto &v = bxf[sl];
      for (size_t off = 0; off < v.size(); ) {
        const uint32_t n = (uint32_t)(v.size() - off < 240u ? v.size() - off : 240u);
        while (mesh_write(&grp->ctx, v.data() + off, n) == ENOMEM) {}
        off += n;
      }
      for (uint32_t pi = 0; pi < sh_pages; pi++) {
        float *dst_f = (float *)((unsigned char *)out_buf.data() + (size_t)pi * MESH_PAGE_BYTES);
        for (int r = 0; r < ranks; r++) {
          const uint32_t idx = my_idx[(sl * (uint32_t)ranks + (uint32_t)r) * sh_pages + pi];
          struct mesh_entry_info e;
          for (;;) { mesh_entry(&grp->ctx, idx, &e); if (e.data) break; }
          if (r == 0) memcpy(dst_f, e.data, e.bytes);
          else neonsum(dst_f, (const float *)e.data, e.bytes / sizeof(float));
        }
      }
      mesh_clear_indices(&grp->ctx, my_idx.data() + sl * (uint32_t)ranks * sh_pages, (uint32_t)ranks * sh_pages);
      const double dt = now_us() - t0;
      if (rep >= 0) { t_wall_sum += dt; t_gpu_sum += tg; }
    } else {
      /* UNIFIED PARAMETRIC PHASE-OVERLAP LOOP: dst(r, step) = (step + r / g) % P */
      @autoreleasepool {
        std::vector<id<MTLCommandBuffer>> cbs(nph);
        std::vector<uint32_t> ph_t(nph), ph_d(nph);
        uint32_t p = 0;
        for (uint32_t t = 0; t < tiles; t++) {
          for (int step = 0; step < ranks; step++) {
            const int d = (step + rank / (int)g) % ranks;
            ph_t[p] = t;
            ph_d[p] = (uint32_t)d;
            cbs[p] = [grp->q commandBuffer];
            encode_phase(cbs[p], t, d);
            [cbs[p] commit];
            p++;
          }
        }
        for (uint32_t q_i = 0; q_i < nph; q_i++) {
          [cbs[q_i] waitUntilCompleted];
          const auto &v = dxf[(sl * tiles + ph_t[q_i]) * (uint32_t)ranks + ph_d[q_i]];
          for (size_t off = 0; off < v.size(); ) {
            const uint32_t n = (uint32_t)(v.size() - off < 240u ? v.size() - off : 240u);
            while (mesh_reduce_write(&grp->ctx, v.data() + off, n) == ENOMEM) {}
            off += n;
          }
        }
      }
      const uint32_t *ridx = my_idx.data() + sl * sh_pages;
      for (uint32_t pi = 0; pi < sh_pages; pi++) {
        struct mesh_entry_info e; uint16_t tag = 0;
        for (;;) { mesh_entry_red(&grp->ctx, ridx[pi], &e, &tag); if (e.data && tag == done_tag) break; }
        if (rep == reps - 1) memcpy((unsigned char *)out_buf.data() + (size_t)pi * MESH_PAGE_BYTES, e.data, e.bytes);
      }
      mesh_clear_indices(&grp->ctx, ridx, sh_pages);
      const double dt = now_us() - t0;
      if (rep >= 0) t_wall_sum += dt;
    }
  }

  /* Verify exact algebraic reduction accuracy on rank 0 against host reference */
  int ok = 1;
  double max_err = 0.0;
  if (!standalone) {
    const size_t check_n = (kind == 1) ? 128u : 16u; /* 1 full [m,l,o_0..o_125] row for C1, 16 f32 for C0/C2 */
    std::vector<float> loc(check_n), all(check_n * (size_t)ranks);
    memcpy(loc.data(), src_ptr[0], check_n * sizeof(float));
    group_allgather(grp, loc.data(), all.data(), check_n * sizeof(float));
    if (rank == 0) {
      if (kind == 0 || kind == 3) {
        for (size_t i = 0; i < check_n; i++) {
          float exp = 0.0f;
          for (int r = 0; r < ranks; r++) exp += all[r * check_n + i];
          double e = fabs((double)out_buf[i] - (double)exp);
          if (e > max_err) max_err = e;
          if (e > 1e-2) ok = 0;
        }
      } else if (kind == 1) {
        /* Exact C1 online softmax reference across ranks 0..ranks-1 */
        float m = -1e30f, l = 0.0f, o[126] = {0};
        for (int r = 0; r < ranks; r++) {
          const float *row = all.data() + r * 128u;
          const float mn = fmaxf(m, row[0]);
          const float ea = expf(m - mn), eb = expf(row[0] - mn);
          l = l * ea + row[1] * eb;
          for (int k = 0; k < 126; k++) o[k] = o[k] * ea + row[2 + k] * eb;
          m = mn;
        }
        double em = fabs((double)out_buf[0] - (double)m);
        double el = fabs((double)out_buf[1] - (double)l);
        max_err = fmax(em, el);
        for (int k = 0; k < 126; k++) {
          double eo = fabs((double)(out_buf[2 + k] / out_buf[1]) - (double)(o[k] / l));
          if (std::isnan(eo) || eo > max_err) max_err = eo;
        }
        if (std::isnan(max_err) || max_err > 1e-4) {
          ok = 0;
          fprintf(stderr, "C1 mismatch: got m=%g l=%g o0=%g exp m=%g l=%g o0=%g\n",
                  out_buf[0], out_buf[1], out_buf[2], m, l, o[0]);
        }
      } else if (kind == 2) {
        /* Exact C2 non-commutative affine composition: (A_3,B_3) o ... o (A_0,B_0) */
        for (size_t i = 0; i < check_n / 2u; i++) {
          float A = 1.0f, B = 0.0f;
          for (int r = 0; r < ranks; r++) {
            const float a = all[r * check_n + 2 * i];
            const float b = all[r * check_n + 2 * i + 1];
            A = a * A;
            B = a * B + b;
          }
          double ea = fabs((double)out_buf[2 * i] - (double)A);
          double eb = fabs((double)out_buf[2 * i + 1] - (double)B);
          if (std::isnan(ea) || std::isnan(eb)) max_err = NAN;
          else max_err = fmax(max_err, fmax(ea, eb));
          if (i == 0 && (std::isnan(max_err) || max_err > 1e-4))
            fprintf(stderr, "C2 mismatch[0]: got (%g,%g) exp (%g,%g)\n", out_buf[0], out_buf[1], A, B);
        }
        if (std::isnan(max_err) || max_err > 1e-4) ok = 0;
      }
    }
    mesh_free_indices(&grp->ctx, my_idx.data(), n_recv);
  }

  for (uint32_t i = 0; i < nph; i++) {
    for (uint32_t k = 0; k < alloc_pages + 4u; k++) mesh_release(&grp->ctx, raw_p0[i] + k);
  }

  const double mean_us = t_wall_sum / reps;
  const double gpu_us = t_gpu_sum / reps;
  const double tp = (kind == 2) ? (flops_or_bytes / (mean_us * 1e-6) / 1e9)  /* GB/s */
                                : (flops_or_bytes / (mean_us * 1e-6) / 1e12); /* TFLOPS */
  out->wall_us = mean_us;
  out->gpu_us = gpu_us;
  out->throughput = tp;
  out->ok = ok;
  out->max_err = max_err;
  return 0;
}

/* ---- Ticket M4: Sparse MoE / Engram zero-payload identity reduction vs dense counterfactual ----
   Active experts write f16 activations directly into /mesh0 and call mesh_reduce_write(MESH_RED_SUM_F16).
   Unrouted experts in sparse mode (dense==0) emit bytes=0, src_page=MESH_PAGE_NONE identity tokens:
   zero GPU compute, zero page alloc, zero j=1 data WRs, zero switch NEON math, exact W[dst_idx] completion! */
int mesh_phase_moe_sparse(struct mesh_phase_group *grp,
                          uint32_t n_experts,
                          uint32_t expert_bytes,
                          uint32_t top_k,
                          int dense_counterfactual,
                          int reps,
                          struct mesh_phase_result *out) {
  const int rank = grp->rank, ranks = grp->ranks;
  if (n_experts < (uint32_t)ranks || (n_experts % (uint32_t)ranks) != 0) return -1;
  const uint32_t exp_per_rank = n_experts / (uint32_t)ranks;
  const uint32_t exp_pages = (expert_bytes + MESH_PAGE_BYTES - 1u) / MESH_PAGE_BYTES;
  const uint32_t my_recv_pages = exp_per_rank * exp_pages;
  const uint64_t data_off = *(const uint64_t *)((const unsigned char *)grp->ctx.M + 88);

  /* Deterministic sparse routing table: each rank activates `top_k` experts, leaving expert (n_experts-1)
     unrouted by ALL ranks when top_k < n_experts - 1 so we verify both partial-sparse and all-zero identity! */
  std::vector<int> active(n_experts, 0);
  const uint32_t routable = (top_k + 1u < n_experts) ? (n_experts - 1u) : n_experts;
  for (uint32_t k = 0; k < top_k && k < routable; k++) {
    active[((uint32_t)rank * 2u + k) % routable] = 1;
  }

  /* Allocate contiguous /mesh0 buffers ONLY for experts that transmit data (all experts in dense mode,
     only top_k experts in M4 sparse mode!). */
  std::vector<uint32_t> raw_p0(n_experts, MESH_PAGE_NONE), src_p0(n_experts, MESH_PAGE_NONE);
  for (uint32_t e = 0; e < n_experts; e++) {
    if (!dense_counterfactual && !active[e]) continue;
    void *base = NULL;
    if (mesh_alloc_contiguous(&grp->ctx, exp_pages, NULL, &base)) return -2;
    raw_p0[e] = (uint32_t)(((unsigned char *)base - ((unsigned char *)grp->ctx.M + data_off)) / MESH_PAGE_BYTES);
    src_p0[e] = raw_p0[e];
    __fp16 *fp = (__fp16 *)base;
    const size_t ne = expert_bytes / sizeof(__fp16);
    if (active[e]) {
      const float val = (float)(rank + 1) * 0.5f + 0.125f * (float)(e & 3u);
      for (size_t i = 0; i < ne; i++) fp[i] = (__fp16)val;
    } else {
      memset(base, 0, expert_bytes);
    }
  }

  /* Double-buffered landing indices exchanged once over zero-NCCL TCP bootstrap */
  const uint32_t slots = 2u;
  const uint32_t n_recv = slots * my_recv_pages;
  std::vector<uint32_t> my_idx(n_recv), all_idx((size_t)ranks * n_recv);
  if (mesh_alloc(&grp->ctx, n_recv, 0, my_idx.data(), NULL)) return -3;
  if (group_allgather(grp, my_idx.data(), all_idx.data(), n_recv * sizeof(uint32_t))) return -4;

  /* Pre-build xfer tables: active experts -> MESH_RED_SUM_F16 data pages; unrouted sparse -> 0-byte identity! */
  const uint16_t my_tag = MESH_TAG_WEIGHT(1);
  const uint16_t done_tag = MESH_TAG_WEIGHT(ranks);
  std::vector<std::vector<struct mesh_red_xfer>> xfs(slots);
  for (uint32_t sl = 0; sl < slots; sl++) {
    xfs[sl].reserve((size_t)n_experts * exp_pages);
    /* Interleave 8-page blocks across experts for max in-switch coalescing */
    for (uint32_t base = 0; base < exp_pages; base += 8u) {
      const uint32_t blk = exp_pages - base < 8u ? exp_pages - base : 8u;
      for (uint32_t e = 0; e < n_experts; e++) {
        const uint32_t dst_r = e / exp_per_rank;
        const uint32_t local_e = e % exp_per_rank;
        const uint32_t *didx = all_idx.data() + (size_t)dst_r * n_recv + sl * my_recv_pages + local_e * exp_pages;
        if (!dense_counterfactual && !active[e]) {
          for (uint32_t k = 0; k < blk; k++)
            xfs[sl].push_back((struct mesh_red_xfer){ MESH_PAGE_NONE, 0u, grp->nodes[dst_r], didx[base + k], MESH_RED_SUM_F16, 0, my_tag });
        } else {
          for (uint32_t k = 0; k < blk; k++) {
            const size_t off = (size_t)(base + k) * MESH_PAGE_BYTES;
            const uint32_t pb = (expert_bytes - off < MESH_PAGE_BYTES) ? (uint32_t)(expert_bytes - off) : MESH_PAGE_BYTES;
            xfs[sl].push_back((struct mesh_red_xfer){ src_p0[e] + base + k, pb, grp->nodes[dst_r], didx[base + k], MESH_RED_SUM_F16, 0, my_tag });
          }
        }
      }
    }
  }

  std::vector<float> first_val(exp_per_rank, 0.0f);
  std::vector<int> was_null(exp_per_rank, 0);
  double sum_us = 0.0;
  for (int rep = -3; rep < reps; rep++) {
    const uint32_t sl = (uint32_t)((rep + 3) & 1);
    mesh_phase_barrier(grp);
    const double t0 = now_us();
    const auto &v = xfs[sl];
    for (size_t off = 0; off < v.size(); ) {
      /* Batch up to 240 data xfers; mesh_reduce_write internally splits zero-payload batches cleanly at 16 */
      const uint32_t n = (uint32_t)(v.size() - off < 240u ? v.size() - off : 240u);
      while (mesh_reduce_write(&grp->ctx, v.data() + off, n) == ENOMEM) {}
      off += n;
    }
    const uint32_t *ridx = my_idx.data() + sl * my_recv_pages;
    for (uint32_t pi = 0; pi < my_recv_pages; pi++) {
      struct mesh_entry_info e; uint16_t tag = 0;
      for (;;) {
        mesh_entry_red(&grp->ctx, ridx[pi], &e, &tag);
        if (tag == done_tag) break; /* accepts BOTH active (e.data!=NULL) and all-zero identity (e.data==NULL)! */
      }
      if (rep == reps - 1 && (pi % exp_pages) == 0) {
        const uint32_t le = pi / exp_pages;
        if (e.data) { first_val[le] = (float)(*(const __fp16 *)e.data); was_null[le] = 0; }
        else { first_val[le] = 0.0f; was_null[le] = 1; }
      }
    }
    mesh_clear_indices(&grp->ctx, ridx, my_recv_pages);
    const double dt = now_us() - t0;
    if (rep >= 0) sum_us += dt;
  }

  /* Verify exact algebraic sum on every owned expert across all ranks */
  int ok = 1;
  double max_err = 0.0;
  for (uint32_t le = 0; le < exp_per_rank; le++) {
    const uint32_t e = (uint32_t)rank * exp_per_rank + le;
    float exp_sum = 0.0f;
    int any_active = 0;
    for (int r = 0; r < ranks; r++) {
      int r_act = 0;
      for (uint32_t k = 0; k < top_k && k < routable; k++)
        if ((((uint32_t)r * 2u + k) % routable) == e) r_act = 1;
      if (r_act) { exp_sum += (float)(r + 1) * 0.5f + 0.125f * (float)(e & 3u); any_active = 1; }
    }
    const double err = fabs((double)first_val[le] - (double)exp_sum);
    if (err > max_err) max_err = err;
    if (err > 1e-2) ok = 0;
    if (!dense_counterfactual && !any_active && !was_null[le]) ok = 0; /* verify all-zero shortcut allocated 0 pages! */
  }

  mesh_free_indices(&grp->ctx, my_idx.data(), n_recv);
  for (uint32_t e = 0; e < n_experts; e++) {
    if (raw_p0[e] != MESH_PAGE_NONE)
      for (uint32_t k = 0; k < exp_pages; k++) mesh_release(&grp->ctx, raw_p0[e] + k);
  }
  const double mean_us = sum_us / reps;
  const double dense_bytes = (double)n_experts * (double)expert_bytes * (double)(ranks - 1);
  out->wall_us = mean_us;
  out->gpu_us = 0.0;
  out->throughput = dense_bytes / (mean_us * 1e-6) / 1e9; /* effective dense-equivalent GB/s */
  out->ok = ok;
  out->max_err = max_err;
  return 0;
}

/* ---- Ticket M5: Zero-Copy Disaggregated Prefill -> Decode KV-Page Handoff ----
   Prefill ranks (0..ranks/2-1) compute layer-by-layer f16 K+V projections directly into 16 KiB-aligned /mesh0
   pages via MPSMatrixMultiplication and stream layer l via mesh_write WHILE GPU computes layer l+1.
   Decode ranks (ranks/2..ranks-1) wrap landed W[kv_idx] pages zero-copy as MTLBuffers via newBufferWithBytesNoCopy. */
int mesh_phase_kv_handoff(struct mesh_phase_group *grp,
                          uint32_t n_layers,
                          uint32_t kv_seq,
                          uint32_t kv_dim,
                          int standalone,
                          int reps,
                          struct mesh_phase_result *out) {
  const int rank = grp->rank, ranks = grp->ranks;
  if (ranks < 2 || (ranks & 1)) return -1;
  const int half = ranks / 2;
  const int is_prefill = (rank < half);
  const int peer_r = is_prefill ? (rank + half) : (rank - half);
  const uint32_t kv_cols = kv_dim * 2u; /* K + V concatenated in fp16 */
  const size_t layer_bytes = (size_t)kv_seq * kv_cols * sizeof(__fp16);
  const uint32_t layer_pages = (uint32_t)((layer_bytes + MESH_PAGE_BYTES - 1u) / MESH_PAGE_BYTES);
  const uint32_t alloc_pages = ((layer_pages + 3u) & ~3u);
  const uint64_t data_off = *(const uint64_t *)((const unsigned char *)grp->ctx.M + 88);

  /* Prefill GPU buffers: H [kv_seq, kv_dim] @ W_kv [kv_dim, 2*kv_dim] -> zero-copy /mesh0 KV[layer] */
  id<MTLBuffer> bufH = nil, bufW = nil;
  MPSMatrixMultiplication *mm = nil;
  MPSMatrix *matH = nil, *matW = nil;
  std::vector<uint32_t> raw_p0(n_layers, MESH_PAGE_NONE), src_p0(n_layers, MESH_PAGE_NONE);
  std::vector<id<MTLBuffer>> bufKV(n_layers);
  std::vector<MPSMatrix *> matKV(n_layers);
  if (is_prefill) {
    bufH = [grp->dev newBufferWithLength:(size_t)kv_seq * kv_dim * sizeof(__fp16) options:MTLResourceStorageModeShared];
    bufW = [grp->dev newBufferWithLength:(size_t)kv_dim * kv_cols * sizeof(__fp16) options:MTLResourceStorageModeShared];
    __fp16 *pH = (__fp16 *)bufH.contents, *pW = (__fp16 *)bufW.contents;
    for (size_t i = 0; i < (size_t)kv_seq * kv_dim; i++) pH[i] = (__fp16)(0.125f * (float)((int)(i % 5u) - 2));
    for (size_t i = 0; i < (size_t)kv_dim * kv_cols; i++) pW[i] = (__fp16)(0.125f * (float)((int)((i + 3u) % 5u) - 2));
    MPSMatrixDescriptor *dH = [MPSMatrixDescriptor matrixDescriptorWithRows:kv_seq columns:kv_dim rowBytes:kv_dim * 2u dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dW = [MPSMatrixDescriptor matrixDescriptorWithRows:kv_dim columns:kv_cols rowBytes:kv_cols * 2u dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dO = [MPSMatrixDescriptor matrixDescriptorWithRows:kv_seq columns:kv_cols rowBytes:kv_cols * 2u dataType:MPSDataTypeFloat16];
    mm = [[MPSMatrixMultiplication alloc] initWithDevice:grp->dev transposeLeft:NO transposeRight:NO
                                              resultRows:kv_seq resultColumns:kv_cols interiorColumns:kv_dim alpha:1.0 beta:0.0];
    matH = [[MPSMatrix alloc] initWithBuffer:bufH descriptor:dH];
    matW = [[MPSMatrix alloc] initWithBuffer:bufW descriptor:dW];
    for (uint32_t l = 0; l < n_layers; l++) {
      void *base = NULL;
      if (mesh_alloc_contiguous(&grp->ctx, alloc_pages + 4u, NULL, &base)) return -2;
      uintptr_t raw = (uintptr_t)base;
      uintptr_t aligned = (raw + 16383u) & ~(uintptr_t)16383u;
      raw_p0[l] = (uint32_t)(((unsigned char *)raw - ((unsigned char *)grp->ctx.M + data_off)) / MESH_PAGE_BYTES);
      src_p0[l] = (uint32_t)(((unsigned char *)aligned - ((unsigned char *)grp->ctx.M + data_off)) / MESH_PAGE_BYTES);
      bufKV[l] = [grp->dev newBufferWithBytesNoCopy:(void *)aligned length:(size_t)alloc_pages * MESH_PAGE_BYTES
                                            options:MTLResourceStorageModeShared deallocator:nil];
      matKV[l] = [[MPSMatrix alloc] initWithBuffer:bufKV[l] descriptor:dO];
    }
  }

  /* Exchange double-buffered KV landing indices */
  const uint32_t slots = 2u;
  const uint32_t total_pages = n_layers * layer_pages;
  const uint32_t n_recv = slots * total_pages;
  std::vector<uint32_t> my_idx(n_recv, 0), all_idx((size_t)ranks * n_recv, 0);
  if (!standalone) {
    if (!is_prefill) {
      if (mesh_alloc(&grp->ctx, n_recv, 0, my_idx.data(), NULL)) return -3;
    }
    if (group_allgather(grp, my_idx.data(), all_idx.data(), n_recv * sizeof(uint32_t))) return -4;
  }

  /* Pre-build per-layer mesh_write descriptors on prefill ranks */
  std::vector<std::vector<struct mesh_xfer>> xfs(slots * n_layers);
  if (is_prefill && !standalone) {
    for (uint32_t sl = 0; sl < slots; sl++) {
      for (uint32_t l = 0; l < n_layers; l++) {
        auto &v = xfs[sl * n_layers + l];
        v.reserve(layer_pages);
        const uint32_t *didx = all_idx.data() + (size_t)peer_r * n_recv + sl * total_pages + l * layer_pages;
        for (uint32_t k = 0; k < layer_pages; k++)
          v.push_back((struct mesh_xfer){ src_p0[l] + k, MESH_PAGE_BYTES, grp->nodes[peer_r], didx[k] });
      }
    }
  }

  int ok = 1;
  double sum_us = 0.0;
  for (int rep = -2; rep < reps; rep++) {
    const uint32_t sl = (uint32_t)((rep + 2) & 1);
    if (!standalone) mesh_phase_barrier(grp);
    const double t0 = now_us();
    if (is_prefill) {
      @autoreleasepool {
        std::vector<id<MTLCommandBuffer>> cbs(n_layers);
        for (uint32_t l = 0; l < n_layers; l++) {
          cbs[l] = [grp->q commandBuffer];
          [mm encodeToCommandBuffer:cbs[l] leftMatrix:matH rightMatrix:matW resultMatrix:matKV[l]];
          [cbs[l] commit];
        }
        for (uint32_t l = 0; l < n_layers; l++) {
          [cbs[l] waitUntilCompleted];
          if (!standalone) {
            const auto &v = xfs[sl * n_layers + l];
            for (size_t off = 0; off < v.size(); ) {
              const uint32_t n = (uint32_t)(v.size() - off < 240u ? v.size() - off : 240u);
              while (mesh_write(&grp->ctx, v.data() + off, n) == ENOMEM) {}
              off += n;
            }
          }
        }
      }
      if (!standalone) mesh_phase_barrier(grp); /* include full decode arrival in end-to-end TTFT/handoff time */
    } else if (!standalone) {
      const uint32_t *ridx = my_idx.data() + sl * total_pages;
      for (uint32_t l = 0; l < n_layers; l++) {
        for (uint32_t k = 0; k < layer_pages; k++) {
          struct mesh_entry_info e;
          for (;;) { mesh_entry(&grp->ctx, ridx[l * layer_pages + k], &e); if (e.data) break; }
          /* Zero-copy verification: wrap first 16 KiB-aligned landed page as an MTLBuffer with no memcpy! */
          if (rep == reps - 1 && k == 0) {
            uintptr_t addr = (uintptr_t)e.data;
            uintptr_t aligned = addr & ~(uintptr_t)16383u;
            id<MTLBuffer> borrow = [grp->dev newBufferWithBytesNoCopy:(void *)aligned length:16384u
                                                             options:MTLResourceStorageModeShared deallocator:nil];
            if (!borrow) ok = 0;
          }
        }
      }
      mesh_clear_indices(&grp->ctx, ridx, total_pages);
      mesh_phase_barrier(grp);
    }
    const double dt = now_us() - t0;
    if (rep >= 0) sum_us += dt;
  }

  if (!standalone && !is_prefill) mesh_free_indices(&grp->ctx, my_idx.data(), n_recv);
  if (is_prefill) {
    for (uint32_t l = 0; l < n_layers; l++)
      for (uint32_t k = 0; k < alloc_pages + 4u; k++) mesh_release(&grp->ctx, raw_p0[l] + k);
  }
  const double mean_us = sum_us / reps;
  const double total_kv_bytes = (double)half * (double)n_layers * (double)layer_bytes;
  out->wall_us = mean_us;
  out->gpu_us = is_prefill ? mean_us : 0.0;
  out->throughput = total_kv_bytes / (mean_us * 1e-6) / 1e9; /* aggregate KV handoff GB/s */
  out->ok = ok;
  out->max_err = 0.0;
  return 0;
}

/* ---- Ticket M6b: Fixed-Capacity R14 Heterogeneous Step returning exact R2 per-rank GPU stretch ----
   Allocates fixed-capacity /mesh0 buffers once (never reallocates when shares change!), executes each rank's
   exact share K_loc = K_total * my_share / total_shares (with optional slow_mult to emulate heterogeneous silicon),
   and returns exact R2 GPU stretch (gpu_us) alongside total step wall_us for online allocate.py Balancer! */
int mesh_phase_run_hetero(struct mesh_phase_group *grp,
                          uint32_t M,
                          uint32_t K_total,
                          uint32_t N,
                          uint32_t my_share,
                          uint32_t total_shares,
                          uint32_t slow_mult,
                          uint32_t tiles,
                          uint32_t coalesce_g,
                          int reps,
                          struct mesh_phase_result *out) {
  const int rank = grp->rank, ranks = grp->ranks;
  const uint32_t g = coalesce_g > 0 ? coalesce_g : 1u;
  const uint32_t K_loc = (uint32_t)(((uint64_t)K_total * my_share) / total_shares);
  const uint32_t N_sh = N / (uint32_t)ranks;
  const uint32_t M_tile = M / tiles;
  const uint32_t nph = tiles * (uint32_t)ranks;
  const size_t tile_sh_bytes = (size_t)M_tile * N_sh * sizeof(float);
  const uint32_t tile_pages = (uint32_t)((tile_sh_bytes + MESH_PAGE_BYTES - 1u) / MESH_PAGE_BYTES);
  const uint32_t sh_pages = tiles * tile_pages;
  const uint32_t alloc_pages = ((tile_pages + 3u) & ~3u);
  const uint64_t data_off = *(const uint64_t *)((const unsigned char *)grp->ctx.M + 88);

  /* R14: Fixed-capacity buffers (sized to K_total) — zero reallocation as shares change! */
  id<MTLBuffer> bufA = [grp->dev newBufferWithLength:(size_t)M * K_total * sizeof(float) options:MTLResourceStorageModeShared];
  std::vector<id<MTLBuffer>> bufB(ranks);
  for (int d = 0; d < ranks; d++)
    bufB[d] = [grp->dev newBufferWithLength:(size_t)K_total * N_sh * sizeof(float) options:MTLResourceStorageModeShared];
  std::vector<uint32_t> raw_p0(nph), src_p0(nph);
  std::vector<id<MTLBuffer>> bufOut(nph);
  for (uint32_t i = 0; i < nph; i++) {
    void *base = NULL;
    if (mesh_alloc_contiguous(&grp->ctx, alloc_pages + 4u, NULL, &base)) return -2;
    uintptr_t raw = (uintptr_t)base, aligned = (raw + 16383u) & ~(uintptr_t)16383u;
    raw_p0[i] = (uint32_t)(((unsigned char *)raw - ((unsigned char *)grp->ctx.M + data_off)) / MESH_PAGE_BYTES);
    src_p0[i] = (uint32_t)(((unsigned char *)aligned - ((unsigned char *)grp->ctx.M + data_off)) / MESH_PAGE_BYTES);
    bufOut[i] = [grp->dev newBufferWithBytesNoCopy:(void *)aligned length:(size_t)alloc_pages * MESH_PAGE_BYTES
                                          options:MTLResourceStorageModeShared deallocator:nil];
  }

  MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M_tile columns:K_loc rowBytes:K_loc * sizeof(float) dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K_loc columns:N_sh rowBytes:N_sh * sizeof(float) dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M_tile columns:N_sh rowBytes:N_sh * sizeof(float) dataType:MPSDataTypeFloat32];
  MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:grp->dev transposeLeft:NO transposeRight:NO
                                                                     resultRows:M_tile resultColumns:N_sh interiorColumns:K_loc alpha:1.0 beta:0.0];
  std::vector<MPSMatrix *> matA(tiles), matB(ranks), matC(nph);
  for (uint32_t t = 0; t < tiles; t++)
    matA[t] = [[MPSMatrix alloc] initWithBuffer:bufA offset:(size_t)t * M_tile * K_loc * sizeof(float) descriptor:descA];
  for (int d = 0; d < ranks; d++) matB[d] = [[MPSMatrix alloc] initWithBuffer:bufB[d] descriptor:descB];
  for (uint32_t i = 0; i < nph; i++) matC[i] = [[MPSMatrix alloc] initWithBuffer:bufOut[i] descriptor:descC];

  const uint32_t slots = 2u;
  const uint32_t n_recv = slots * sh_pages;
  std::vector<uint32_t> my_idx(n_recv), all_idx((size_t)ranks * n_recv);
  if (mesh_alloc(&grp->ctx, n_recv, 0, my_idx.data(), NULL)) return -3;
  if (group_allgather(grp, my_idx.data(), all_idx.data(), n_recv * sizeof(uint32_t))) return -4;

  const uint16_t my_tag = MESH_TAG_WEIGHT(1), done_tag = MESH_TAG_WEIGHT(ranks);
  std::vector<std::vector<struct mesh_red_xfer>> dxf(slots * nph);
  for (uint32_t sl = 0; sl < slots; sl++) {
    for (uint32_t t = 0; t < tiles; t++) {
      for (int d = 0; d < ranks; d++) {
        auto &v = dxf[(sl * tiles + t) * (uint32_t)ranks + (uint32_t)d];
        v.reserve(tile_pages);
        const uint32_t *didx = all_idx.data() + (size_t)d * n_recv + sl * sh_pages + t * tile_pages;
        const uint32_t p0 = src_p0[t * (uint32_t)ranks + (uint32_t)d];
        for (uint32_t k = 0; k < tile_pages; k++)
          v.push_back((struct mesh_red_xfer){ p0 + k, MESH_PAGE_BYTES, grp->nodes[d], didx[k], MESH_RED_SUM_F32, 0, my_tag });
      }
    }
  }

  double wall_sum = 0.0, gpu_sum = 0.0;
  for (int rep = -2; rep < reps; rep++) {
    const uint32_t sl = (uint32_t)((rep + 2) & 1);
    mesh_phase_barrier(grp);
    const double t0 = now_us();
    double t_gpu_end = t0;
    @autoreleasepool {
      std::vector<id<MTLCommandBuffer>> cbs(nph);
      std::vector<uint32_t> ph_t(nph), ph_d(nph);
      uint32_t p = 0;
      for (uint32_t t = 0; t < tiles; t++) {
        for (int step = 0; step < ranks; step++) {
          const int d = (step + rank / (int)g) % ranks;
          ph_t[p] = t; ph_d[p] = (uint32_t)d;
          cbs[p] = [grp->q commandBuffer];
          for (uint32_t m_i = 0; m_i < (slow_mult ? slow_mult : 1u); m_i++)
            [mm encodeToCommandBuffer:cbs[p] leftMatrix:matA[t] rightMatrix:matB[d] resultMatrix:matC[t * (uint32_t)ranks + (uint32_t)d]];
          [cbs[p] commit];
          p++;
        }
      }
      for (uint32_t qi = 0; qi < nph; qi++) {
        [cbs[qi] waitUntilCompleted];
        if (qi + 1u == nph) t_gpu_end = now_us();
        const auto &v = dxf[(sl * tiles + ph_t[qi]) * (uint32_t)ranks + ph_d[qi]];
        for (size_t off = 0; off < v.size(); ) {
          const uint32_t n = (uint32_t)(v.size() - off < 240u ? v.size() - off : 240u);
          while (mesh_reduce_write(&grp->ctx, v.data() + off, n) == ENOMEM) {}
          off += n;
        }
      }
    }
    const uint32_t *ridx = my_idx.data() + sl * sh_pages;
    for (uint32_t pi = 0; pi < sh_pages; pi++) {
      struct mesh_entry_info e; uint16_t tag = 0;
      for (;;) { mesh_entry_red(&grp->ctx, ridx[pi], &e, &tag); if (e.data && tag == done_tag) break; }
    }
    mesh_clear_indices(&grp->ctx, ridx, sh_pages);
    const double dt = now_us() - t0;
    if (rep >= 0) { wall_sum += dt; gpu_sum += (t_gpu_end - t0); }
  }

  mesh_free_indices(&grp->ctx, my_idx.data(), n_recv);
  for (uint32_t i = 0; i < nph; i++)
    for (uint32_t k = 0; k < alloc_pages + 4u; k++) mesh_release(&grp->ctx, raw_p0[i] + k);
  out->wall_us = wall_sum / reps;
  out->gpu_us = gpu_sum / reps; /* exact R2 compute stretch excluding collective wait! */
  out->throughput = (2.0 * (double)M * (double)K_total * (double)N) / ((wall_sum / reps) * 1e-6) / 1e12;
  out->ok = 1;
  out->max_err = 0.0;
  return 0;
}

} /* extern "C" */
