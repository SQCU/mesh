// Native single-command-buffer Metal + flat-DMA FX graph execution engine (libmesh-graph.dylib).
// Executes arbitrary lowered torch.export FX graphs in ONE MTLCommandBuffer + ONE MTLComputeCommandEncoder
// per step over zero-copy /mesh0 W[] + P[] MTLBuffers with Steiner multicast & GF(2)^6 Hadamard-domain reduction.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <arpa/inet.h>
#include <mach/mach_time.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mesh-graph.h"
extern "C" {
#include "mesh-call.h"
#include "mesh-codec.h"
}

namespace {

constexpr uint32_t kGenSlots = 4;
constexpr uint32_t kDbRingSize = 4096;
constexpr uint32_t kDbRingMask = kDbRingSize - 1;
constexpr uint32_t kMaxRanks = 8;

enum DbKind : uint32_t {
  DB_REDUCE = 0,
  DB_GATHER = 1,
  DB_ARGMAX = 2,
};

struct alignas(64) Doorbell {
  std::atomic<uint64_t> pub_seq;
  std::atomic<uint64_t> ret_seq;
  uint32_t kind;
  uint32_t cid;
  uint32_t g_slot;
  uint32_t n_pages;
  uint32_t last_bytes;
  uint32_t red_op;
  uint32_t pad[6];
};
static_assert(sizeof(Doorbell) == 64, "Doorbell cache line");

inline uint32_t hdr_entries(const struct hdr *m) { return *(const uint32_t *)((const uint8_t *)m + 16); }
inline uint32_t hdr_node(const struct hdr *m) { return *(const uint32_t *)((const uint8_t *)m + 20); }
inline uint64_t hdr_table_off(const struct hdr *m) { return *(const uint64_t *)((const uint8_t *)m + 32); }
inline uint64_t hdr_data_off(const struct hdr *m) { return *(const uint64_t *)((const uint8_t *)m + 88); }
inline uint64_t hdr_length(const struct hdr *m) { return *(const uint64_t *)((const uint8_t *)m + 96); }
inline uint64_t now_ns() { return clock_gettime_nsec_np(CLOCK_MONOTONIC); }

int send_all(int fd, const void *buf, size_t n) {
  const uint8_t *p = static_cast<const uint8_t *>(buf);
  while (n) {
    ssize_t k = send(fd, p, n, 0);
    if (k <= 0) return -1;
    p += k;
    n -= static_cast<size_t>(k);
  }
  return 0;
}

int recv_all(int fd, void *buf, size_t n) {
  uint8_t *p = static_cast<uint8_t *>(buf);
  while (n) {
    ssize_t k = recv(fd, p, n, 0);
    if (k <= 0) return -1;
    p += k;
    n -= static_cast<size_t>(k);
  }
  return 0;
}

struct BufSlot {
  id<MTLBuffer> mtl = nil;
  NSUInteger off = 0;
  uint64_t nbytes = 0;
  int kind = MG_BUF_STATE;
  void *host_ptr = nullptr;
};

struct CrossingSlot {
  uint32_t n_elems = 0;
  uint32_t esize = 2;
  int wire_mode = MG_WIRE_GATHER;
  int is_argmax = 0;
  uint8_t red_op = MESH_RED_SUM_BF16;
  uint32_t pages_per_frame = 1;
  uint32_t last_page_bytes = 4096;
  uint32_t src_base_page = MESH_PAGE_NONE;
  uint32_t total_src_pages = 0;
  uint32_t land_per_gen = 0; // pages_per_frame (reduce) or world*pages_per_frame (gather) or world (argmax)
  std::vector<uint32_t> land[kMaxRanks]; // [rank][kGenSlots * land_per_gen]
  id<MTLBuffer> land_buf = nil;
  uint32_t active_db_slot = 0;
  uint64_t active_db_seq = 0;
};

// M5 Ultra decode launch table (exact shapes from mesh-mlx)
struct Launch { uint32_t sg, s, rows, vec, unroll; };
struct LaunchEntry { uint32_t tp; const char *kernel; uint32_t N, K; Launch l; };
constexpr Launch kWideRows{4, 1, 1, 4, 8};
constexpr LaunchEntry kLaunches[] = {
  {2, "gemv_publish", 1536, 1024, {8, 4, 2, 4, 2}},
  {2, "gemv_publish", 1536, 6144, {8, 2, 1, 8, 4}},
  {2, "gate_up_geglu", 3072, 1536, {2, 2, 1, 8, 2}},
  {2, "gate_up_geglu", 6144, 1536, {4, 2, 1, 8, 2}},
  {2, "ple_gate", 256, 1536, {8, 4, 1, 8, 4}},
  {4, "gemv_publish", 1536, 512, {8, 1, 2, 8, 4}},
  {4, "gemv_publish", 1536, 1536, {8, 1, 2, 4, 4}},
  {4, "gemv_rows", 512, 1536, {2, 1, 1, 4, 4}},
  {4, "gemv_rows", 1024, 1536, {2, 1, 1, 8, 2}},
  {4, "gemv_rows", 2048, 1536, {4, 1, 1, 8, 4}},
  {4, "gemv_rows", 65536, 1536, {4, 2, 1, 8, 2}},
  {4, "ple_gate", 256, 1536, {4, 4, 1, 4, 2}},
};

Launch launch_for(uint32_t world, const char *kernel, uint32_t N, uint32_t K) {
  for (const auto &x : kLaunches) {
    if (x.tp == world && !strcmp(x.kernel, kernel) && x.N == N && x.K == K) {
      const uint32_t rows = x.l.sg / x.l.s * x.l.rows;
      if (N % rows == 0 && K % (x.l.s * x.l.vec) == 0) return x.l;
    }
  }
  return kWideRows;
}

std::string launch_name(const char *kernel, uint32_t B, const Launch &l) {
  return std::string(kernel) + "_" + std::to_string(B) + "_" + std::to_string(l.sg) + "_" +
         std::to_string(l.s) + "_" + std::to_string(l.rows) + "_" + std::to_string(l.vec) + "_" +
         std::to_string(l.unroll);
}

std::string launch_instantiations() {
  const std::map<std::string, std::string> args = {
    {"ple_gate", "(device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*, constant uint*, uint, uint, uint)"},
    {"gate_up_act", "(device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*, constant uint*, uint, uint, uint)"},
    {"gemv_rows", "(device const bfloat*, device const bfloat*, device bfloat*, constant uint*, uint, uint, uint)"},
    {"gemv_publish", "(device const bfloat*, device const bfloat*, device bfloat*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint, uint, uint, uint, uint)"},
  };
  std::vector<std::pair<std::string, Launch>> all;
  for (const auto &k : args) all.push_back({k.first, kWideRows});
  for (const auto &x : kLaunches) {
    std::string kn = x.kernel;
    if (kn == "gate_up_geglu") kn = "gate_up_act";
    all.push_back({kn, x.l});
  }
  std::set<std::string> made;
  std::string out;
  for (uint32_t B = 1; B <= 2; B++) {
    for (const auto &[kernel, l] : all) {
      const std::string name = launch_name(kernel.c_str(), B, l);
      if (!args.count(kernel) || !made.insert(name).second) continue;
      out += "template [[host_name(\"" + name + "\")]] [[kernel]] void " + kernel + "<" + std::to_string(B) + ", " +
             std::to_string(l.sg) + ", " + std::to_string(l.s) + ", " + std::to_string(l.rows) + ", " +
             std::to_string(l.vec) + ", " + std::to_string(l.unroll) + ">" + args.at(kernel) + ";\n";
    }
  }
  return out;
}

std::string build_core_msl() {
  std::string s = R"MSL(
#pragma METAL internals : enable
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline void mesh_sys_write(device uchar* p, ulong idx, T v) {
  ((volatile coherent(system) device T*)p)[idx] = v;
}
template <typename T>
inline T mesh_sys_read(const device uchar* p, ulong idx) {
  return ((const volatile coherent(system) device T*)p)[idx];
}
inline void mesh_sys_fence() {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
}
inline void mesh_ring_pub(device ulong* db, uint slot, ulong seq) {
  mesh_sys_fence();
  ((volatile coherent(system) device ulong*)db)[ulong(slot) * 8ul] = seq;
  mesh_sys_fence();
}
inline void mesh_ring_ret(device ulong* db, uint slot, ulong seq) {
  mesh_sys_fence();
  ((volatile coherent(system) device ulong*)db)[ulong(slot) * 8ul + 1ul] = seq;
  mesh_sys_fence();
}
inline uint mesh_poll_land(const device ulong* W, uint idx, uint want_tag) {
  const volatile coherent(system) device ulong* wp = (const volatile coherent(system) device ulong*)W + idx;
  for (uint spin = 0; spin < 4500000u; spin++) {
    mesh_sys_fence();
    ulong w = *wp;
    uint lo = uint(w & 0xfffffffful);
    uint tag = uint(w >> 48);
    if (lo != 0u && lo != 0xffffffffu && (want_tag == 0u || tag >= want_tag)) return lo - 1u;
  }
  return 0xffffffffu;
}
inline void fwht64_simd_pair(thread float& a, thread float& b, uint lane) {
  float s = a + b, d = a - b;
  a = s; b = d;
  #pragma unroll
  for (uint mask = 1u; mask <= 16u; mask <<= 1u) {
    float oa = simd_shuffle_xor(a, mask);
    float ob = simd_shuffle_xor(b, mask);
    bool hi = (lane & mask) != 0u;
    a = hi ? (oa - a) : (a + oa);
    b = hi ? (ob - b) : (b + ob);
  }
  a *= 0.125f;
  b *= 0.125f;
}
inline float gelu_approx(float gv) {
  return 0.5f * gv * (1.0f + metal::precise::tanh(0.7978845608028654f * (gv + 0.044715f * gv * gv * gv)));
}
inline float silu_fn(float x) { return x / (1.0f + metal::fast::exp(-x)); }

// ---- 1. Embedding lookup with optional scale ----
[[kernel]] void mg_embedding_bf16(const device int* tok [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant uint* u [[buffer(3)]], constant float& scale [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint D = u[0], B = u[1];
  const uint d = gid.x, b = gid.y;
  if (d >= D || b >= B) return;
  const uint t = uint(max(0, tok[b]));
  const float v = float(W[ulong(t) * D + d]) * scale;
  out[ulong(b) * D + d] = bfloat(v);
}

// ---- 2. Generic tiled linear (bf16 & f32) for arbitrary (M, N, K) ----
[[kernel]] void mg_linear_generic_bf16(const device bfloat* x [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    device bfloat* y [[buffer(2)]], constant uint* u [[buffer(3)]], // M, N, K
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint M = u[0], N = u[1], K = u[2];
  const uint m = tg.y;
  const uint n = tg.x * 4u + sg;
  if (m >= M || n >= N) return;
  const device bfloat* xr = x + ulong(m) * K;
  const device bfloat* wr = W + ulong(n) * K;
  float acc = 0.0f;
  for (uint k = lane; k < K; k += 32u) acc += float(xr[k]) * float(wr[k]);
  for (ushort off = 16; off >= 1; off >>= 1) acc += simd_shuffle_down(acc, off);
  if (lane == 0) y[ulong(m) * N + n] = bfloat(acc);
}

[[kernel]] void mg_linear_generic_f32(const device float* x [[buffer(0)]], const device float* W [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint* u [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint M = u[0], N = u[1], K = u[2];
  const uint m = tg.y;
  const uint n = tg.x * 4u + sg;
  if (m >= M || n >= N) return;
  const device float* xr = x + ulong(m) * K;
  const device float* wr = W + ulong(n) * K;
  float acc = 0.0f;
  for (uint k = lane; k < K; k += 32u) acc += xr[k] * wr[k];
  for (ushort off = 16; off >= 1; off >>= 1) acc += simd_shuffle_down(acc, off);
  if (lane == 0) y[ulong(m) * N + n] = acc;
}

// ---- 3. M5-Ultra tuned decode tiled_rows (B=1,2) ----
template <uint B, uint SG, uint S, uint ROWS, uint VEC, uint UNROLL>
inline void tiled_rows(device const bfloat* x, device const bfloat* W, uint K, uint tg, uint sg, uint lane,
                       threadgroup float* red, thread float (&res)[ROWS][B], thread uint& row0, thread bool& owner) {
  constexpr uint H = VEC / 4;
  const uint part = sg % S, Kp = K / S;
  row0 = (tg * (SG / S) + sg / S) * ROWS;
  device const vec<bfloat, 4>* x4 = (device const vec<bfloat, 4>*)(x + part * Kp);
  device const bfloat* Wp = W + part * Kp;
  const uint n_v = Kp / VEC, n_main = n_v - n_v % (32 * UNROLL), x_ld4 = K / 4;
  float acc[ROWS][B];
  for (uint r = 0; r < ROWS; r++) for (uint b = 0; b < B; b++) acc[r][b] = 0.0f;
  for (uint base = 0; base < n_main; base += 32 * UNROLL) {
    float4 wf[UNROLL][ROWS][H];
    for (uint u = 0; u < UNROLL; u++) for (uint r = 0; r < ROWS; r++) for (uint h = 0; h < H; h++)
      wf[u][r][h] = float4(((device const vec<bfloat, 4>*)(Wp + ulong(row0 + r) * K))[(base + u * 32 + lane) * H + h]);
    for (uint b = 0; b < B; b++) {
      float a[ROWS];
      for (uint r = 0; r < ROWS; r++) a[r] = 0.0f;
      for (uint u = 0; u < UNROLL; u++) for (uint h = 0; h < H; h++) {
        const float4 xv = float4(x4[b * x_ld4 + (base + u * 32 + lane) * H + h]);
        for (uint r = 0; r < ROWS; r++) a[r] += dot(wf[u][r][h], xv);
      }
      for (uint r = 0; r < ROWS; r++) acc[r][b] += a[r];
    }
  }
  for (uint v = n_main + lane; v < n_v; v += 32)
    for (uint h = 0; h < H; h++) {
      float4 wf[ROWS];
      for (uint r = 0; r < ROWS; r++) wf[r] = float4(((device const vec<bfloat, 4>*)(Wp + ulong(row0 + r) * K))[v * H + h]);
      for (uint b = 0; b < B; b++) { const float4 xv = float4(x4[b * x_ld4 + v * H + h]); for (uint r = 0; r < ROWS; r++) acc[r][b] += dot(wf[r], xv); }
    }
  for (uint r = 0; r < ROWS; r++) for (uint b = 0; b < B; b++) {
    float v = acc[r][b];
    for (ushort off = 16; off >= 1; off >>= 1) v += simd_shuffle_down(v, off);
    res[r][b] = v;
    if (S > 1 && lane == 0) red[(sg * ROWS + r) * B + b] = res[r][b];
  }
  owner = part == 0 && lane == 0;
  if (S > 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (owner)
      for (uint r = 0; r < ROWS; r++) for (uint b = 0; b < B; b++) {
        float t = 0.0f;
        for (uint q = 0; q < S; q++) t += red[((sg + q) * ROWS + r) * B + b];
        res[r][b] = t;
      }
  }
}

template <uint B, uint SG, uint S, uint ROWS, uint VEC, uint UNROLL>
[[kernel]] void gate_up_act(device const bfloat* x [[buffer(0)]], device const bfloat* Wg [[buffer(1)]], device const bfloat* Wu [[buffer(2)]],
    device bfloat* act [[buffer(3)]], constant uint* dims [[buffer(4)]], // N, K, act_kind (0=gelu_approx, 1=silu)
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {
  const uint N = dims[0], K = dims[1], act_kind = dims[2];
  threadgroup float red[2][SG * ROWS * B];
  float g[ROWS][B], u[ROWS][B]; uint row0; bool owner;
  tiled_rows<B, SG, S, ROWS, VEC, UNROLL>(x, Wg, K, tg, sg, lane, red[0], g, row0, owner);
  tiled_rows<B, SG, S, ROWS, VEC, UNROLL>(x, Wu, K, tg, sg, lane, red[1], u, row0, owner);
  if (!owner) return;
  for (uint r = 0; r < ROWS; r++) for (uint b = 0; b < B; b++) {
    const float gv = float(bfloat(g[r][b])), uv = float(bfloat(u[r][b]));
    const float av = (act_kind == 1u) ? silu_fn(gv) : gelu_approx(gv);
    act[ulong(b) * N + row0 + r] = bfloat(float(bfloat(av)) * uv);
  }
}

template <uint B, uint SG, uint S, uint ROWS, uint VEC, uint UNROLL>
[[kernel]] void ple_gate(device const bfloat* x [[buffer(0)]], device const bfloat* Wg [[buffer(1)]], device const bfloat* pli [[buffer(2)]],
    device bfloat* gate [[buffer(3)]], constant uint* dims [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {
  const uint N = dims[0], K = dims[1], stride = dims[2], layer = dims[3];
  threadgroup float red[SG * ROWS * B];
  float g[ROWS][B]; uint row0; bool owner;
  tiled_rows<B, SG, S, ROWS, VEC, UNROLL>(x, Wg, K, tg, sg, lane, red, g, row0, owner);
  if (!owner) return;
  for (uint r = 0; r < ROWS; r++) for (uint b = 0; b < B; b++) {
    const uint row = row0 + r;
    gate[ulong(b) * N + row] = bfloat(float(bfloat(gelu_approx(float(bfloat(g[r][b]))))) * float(pli[ulong(b) * stride + ulong(layer) * N + row]));
  }
}

template <uint B, uint SG, uint S, uint ROWS, uint VEC, uint UNROLL>
[[kernel]] void gemv_publish(device const bfloat* x [[buffer(0)]], device const bfloat* W [[buffer(1)]], device bfloat* y [[buffer(2)]],
    device uchar* send [[buffer(3)]], device ulong* db [[buffer(4)]], device atomic_uint* done [[buffer(5)]],
    constant uint* dims [[buffer(6)]], // N, K, db_slot, seq_lo, seq_hi, wmode
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint i [[thread_index_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]]) {
  const uint N = dims[0], K = dims[1], wmode = dims[5];
  threadgroup float red[SG * ROWS * B];
  float a[ROWS][B]; uint row0; bool owner;
  tiled_rows<B, SG, S, ROWS, VEC, UNROLL>(x, W, K, tg, sg, lane, red, a, row0, owner);
  if (owner) {
    for (uint r = 0; r < ROWS; r++) for (uint b = 0; b < B; b++) {
      const bfloat v = bfloat(a[r][b]);
      const ulong at = ulong(b) * N + row0 + r;
      y[at] = v;
      if (wmode == 1u) mesh_sys_write<float>(send, at, float(v));
      else if (wmode < 3u) mesh_sys_write<bfloat>(send, at, v);
    }
  }
  if (wmode >= 3u) return;
  mesh_sys_fence();
  threadgroup_barrier(mem_flags::mem_device);
  threadgroup bool last;
  if (i == 0) {
    mesh_sys_fence();
    last = atomic_fetch_add_explicit(done, 1u, memory_order_relaxed) == groups - 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last && i == 0) {
    atomic_store_explicit(done, 0u, memory_order_relaxed);
    ulong seq = ulong(dims[3]) | (ulong(dims[4]) << 32);
    mesh_ring_pub(db, dims[2], seq);
  }
}

template <uint B, uint SG, uint S, uint ROWS, uint VEC, uint UNROLL>
[[kernel]] void gemv_rows(device const bfloat* x [[buffer(0)]], device const bfloat* W [[buffer(1)]], device bfloat* y [[buffer(2)]],
    constant uint* dims [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {
  const uint N = dims[0], K = dims[1];
  threadgroup float red[SG * ROWS * B];
  float a[ROWS][B]; uint row0; bool owner;
  tiled_rows<B, SG, S, ROWS, VEC, UNROLL>(x, W, K, tg, sg, lane, red, a, row0, owner);
  if (owner)
    for (uint r = 0; r < ROWS; r++) for (uint b = 0; b < B; b++) y[ulong(b) * N + row0 + r] = bfloat(a[r][b]);
}

// ---- 4. Standalone publish kernels (bf16/f32 & GF(2)^6 had_int8) ----
template <typename T> struct four { typedef uint2 type; };
template <> struct four<float> { typedef uint4 type; };

template <typename T>
[[kernel]] void mg_write_pub(device const T* x [[buffer(0)]], device uchar* send [[buffer(1)]],
    device ulong* db [[buffer(2)]], device atomic_uint* done [[buffer(3)]],
    constant uint* cfg [[buffer(4)]], // n, db_slot, seq_lo, seq_hi, widen_f32
    uint k [[thread_position_in_grid]], uint i [[thread_index_in_threadgroup]], uint groups [[threadgroups_per_grid]]) {
  const uint n = cfg[0], widen = cfg[4];
  if (widen) {
    const uint base = k * 4u;
    if (base < n) {
      float4 v = float4(float(x[base]), float(x[base + 1u]), float(x[base + 2u]), float(x[base + 3u]));
      mesh_sys_write<uint4>(send, k, as_type<uint4>(v));
    }
  } else {
    typedef typename four<T>::type V;
    if (k < n / 4u) mesh_sys_write<V>(send, k, ((device const V*)x)[k]);
  }
  mesh_sys_fence();
  threadgroup_barrier(mem_flags::mem_device);
  threadgroup bool last;
  if (i == 0) {
    mesh_sys_fence();
    last = atomic_fetch_add_explicit(done, 1u, memory_order_relaxed) == groups - 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last && i == 0) {
    atomic_store_explicit(done, 0u, memory_order_relaxed);
    ulong seq = ulong(cfg[2]) | (ulong(cfg[3]) << 32);
    mesh_ring_pub(db, cfg[1], seq);
  }
}
template [[host_name("mg_write_pub_bf16")]] [[kernel]] void mg_write_pub<bfloat>(device const bfloat*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint, uint, uint);
template [[host_name("mg_write_pub_f32")]] [[kernel]] void mg_write_pub<float>(device const float*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint, uint, uint);

[[kernel]] void mg_had8_pub_bf16(device const bfloat* x [[buffer(0)]], device uchar* send [[buffer(1)]],
    device ulong* db [[buffer(2)]], device atomic_uint* done [[buffer(3)]],
    constant uint* cfg [[buffer(4)]], // n_blks, db_slot, seq_lo, seq_hi
    uint tg [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint i [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint n_blks = cfg[0];
  const uint blk = tg * 4u + sg;
  if (blk < n_blks) {
    const ulong base = ulong(blk) * 64ul;
    float a = float(x[base + lane * 2u]);
    float b = float(x[base + lane * 2u + 1u]);
    fwht64_simd_pair(a, b, lane);
    const bool upper = (lane >= 16u);
    const float ab = max(fabs(a), fabs(b));
    const float amax_lo = simd_max(upper ? 0.0f : ab);
    const float amax_hi = simd_max(upper ? ab : 0.0f);
    const half sh_lo = (amax_lo > 0.0f) ? half(max(amax_lo * (1.0f / 127.0f), 6.103515625e-5f)) : half(0.0f);
    const half sh_hi = (amax_hi > 0.0f) ? half(max(amax_hi * (1.0f / 127.0f), 6.103515625e-5f)) : half(0.0f);
    const float sc = float(upper ? sh_hi : sh_lo);
    const float inv_s = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
    const ulong dst_off = ulong(blk / 60u) * 4096ul + ulong(blk % 60u) * 68ul;
    if (lane == 0u) {
      mesh_sys_write<half>(send + dst_off, 0, sh_lo);
      mesh_sys_write<half>(send + dst_off, 1, sh_hi);
    }
    mesh_sys_write<char>(send + dst_off + 4ul, lane * 2u, char(clamp(int(rint(a * inv_s)), -127, 127)));
    mesh_sys_write<char>(send + dst_off + 4ul, lane * 2u + 1u, char(clamp(int(rint(b * inv_s)), -127, 127)));
  }
  mesh_sys_fence();
  threadgroup_barrier(mem_flags::mem_device);
  threadgroup bool last;
  if (i == 0) {
    mesh_sys_fence();
    last = atomic_fetch_add_explicit(done, 1u, memory_order_relaxed) == groups - 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last && i == 0) {
    atomic_store_explicit(done, 0u, memory_order_relaxed);
    ulong seq = ulong(cfg[2]) | (ulong(cfg[3]) << 32);
    mesh_ring_pub(db, cfg[1], seq);
  }
}

// ---- 5. Fused Crossing Sum + RMSNorm + Residual [+ RMSNorm2] (and local RMSNorm / AddNorm) ----
template <typename T>
[[kernel]] void mg_sum_norm(const device ulong* W [[buffer(0)]], const device uchar* arena [[buffer(1)]],
    const device uint* land_idx [[buffer(2)]], device const T* own [[buffer(3)]],
    device const T* residual [[buffer(4)]], device const T* w1 [[buffer(5)]], device const T* w2 [[buffer(6)]],
    device T* h [[buffer(7)]], device T* normed [[buffer(8)]], device const T* scalar [[buffer(9)]],
    device ulong* db [[buffer(10)]], device atomic_uint* rdone [[buffer(11)]],
    constant uint* cfg [[buffer(12)]], // [0]=n,[1]=ranks,[2]=me,[3]=dim,[4]=twice,[5]=scaled,[6]=wmode,[7]=ppf,[8]=lbase,[9]=db_slot,[10]=seq_lo,[11]=seq_hi,[12]=flags(bit0=has_res,bit1=has_w1,bit2=pure_sum)
    constant float& eps [[buffer(13)]],
    uint i [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]], uint rows [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint ranks = cfg[1], me = cfg[2], dim = cfg[3], twice = cfg[4], scaled = cfg[5];
  const uint wmode = cfg[6], ppf = cfg[7], lbase = cfg[8], flags = cfg[12];
  const bool has_res = (flags & 1u) != 0u;
  const bool has_w1 = (flags & 2u) != 0u;
  const bool pure_sum = (flags & 4u) != 0u;
  const bool do_wait = (ranks > 1u);
  const T by = scaled ? scalar[0] : T(1.0f);
  constexpr uint READS = 4;
  const ulong base = ulong(row) * dim;
  const uint k0 = i * READS;

  float mine[READS], res[READS], g1[READS], g2[READS];
  for (uint e = 0; e < READS; e++) {
    const bool in = k0 + e < dim;
    mine[e] = in ? float(own[base + k0 + e]) : 0.0f;
    res[e]  = (in && has_res) ? float(residual[base + k0 + e]) : 0.0f;
    g1[e]   = (in && has_w1) ? float(w1[k0 + e]) : 1.0f;
    g2[e]   = (in && twice) ? float(w2[k0 + e]) : 1.0f;
  }

  const uint wire_eb = (wmode == 1u) ? 4u : sizeof(T);
  const uint pg0 = uint((base * wire_eb) >> 12);
  const uint pg1 = uint(((base + dim) * wire_eb - 1ul) >> 12);
  const uint npg = pg1 - pg0 + 1u;

  threadgroup uint tg_phys[32];
  if (do_wait) {
    if (wmode == 2u) {
      const uint total_polls = ranks * npg;
      for (uint idx = i; idx < total_polls; idx += 32u) {
        const uint q = idx / npg;
        const uint po = idx % npg;
        tg_phys[q * 4u + po] = (q == me) ? 0xffffffffu : mesh_poll_land(W, land_idx[lbase + q * ppf + pg0 + po], 0u);
      }
    } else {
      if (i < npg) tg_phys[i] = mesh_poll_land(W, land_idx[lbase + pg0 + i], ranks - 1u);
    }
    threadgroup_barrier(mem_flags::mem_device);
    mesh_sys_fence();
  }

  typedef typename four<T>::type V;
  float s[READS] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (k0 < dim) {
    if (!do_wait) {
      for (uint e = 0; e < READS; e++) s[e] = mine[e];
    } else if (wmode == 2u) {
      const ulong boff = (base + k0) * sizeof(T);
      const uint po = uint(boff >> 12) - pg0;
      const uint in_pg = uint(boff & 4095u);
      for (uint q = 0; q < ranks; q++) {
        if (q == me) {
          for (uint e = 0; e < READS; e++) s[e] += mine[e];
        } else {
          const uint phys = tg_phys[q * 4u + po];
          if (phys != 0xffffffffu) {
            const vec<T, 4> got = as_type<vec<T, 4>>(mesh_sys_read<V>(arena + ulong(phys) * 4096ul + in_pg, 0));
            for (uint e = 0; e < READS; e++) s[e] += float(got[e]);
          }
        }
      }
    } else if (wmode == 1u) {
      const ulong boff = (base + k0) * 4ul;
      const uint po = uint(boff >> 12) - pg0;
      const uint in_pg = uint(boff & 4095u);
      const uint phys = tg_phys[po];
      for (uint e = 0; e < READS; e++) s[e] = mine[e];
      if (phys != 0xffffffffu) {
        const float4 got = as_type<float4>(mesh_sys_read<uint4>(arena + ulong(phys) * 4096ul + in_pg, 0));
        for (uint e = 0; e < READS; e++) s[e] += got[e];
      }
    } else {
      const ulong boff = (base + k0) * sizeof(T);
      const uint po = uint(boff >> 12) - pg0;
      const uint in_pg = uint(boff & 4095u);
      const uint phys = tg_phys[po];
      for (uint e = 0; e < READS; e++) s[e] = mine[e];
      if (phys != 0xffffffffu) {
        const vec<T, 4> got = as_type<vec<T, 4>>(mesh_sys_read<V>(arena + ulong(phys) * 4096ul + in_pg, 0));
        for (uint e = 0; e < READS; e++) s[e] += float(got[e]);
      }
    }
  }

  if (pure_sum) {
    for (uint e = 0; e < READS; e++) if (k0 + e < dim) h[base + k0 + e] = T(s[e]);
  } else {
    threadgroup float sums[32];
    threadgroup float inv[1];
    float v[READS];
    float acc = 0.0f;
    for (uint e = 0; e < READS; e++) {
      const float x = k0 + e < dim ? float(T(s[e])) : 0.0f;
      v[e] = x;
      acc += x * x;
    }
    for (uint pass = 0; pass < 1u + twice; pass++) {
      acc = simd_sum(acc);
      if (sg == 0) sums[lane] = 0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0) sums[sg] = acc;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) {
        const float t = simd_sum(sums[lane]);
        if (lane == 0) inv[0] = metal::precise::rsqrt(t / dim + eps);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float scale = inv[0];
      acc = 0.0f;
      for (uint e = 0; e < READS; e++) {
        const uint k = k0 + e;
        if (k >= dim) continue;
        if (pass == 0) {
          const T nv = has_w1 ? T(float(T(g1[e])) * float(T(v[e] * scale))) : T(v[e] * scale);
          T x = has_res ? T(res[e] + float(nv)) : nv;
          if (scaled) x = T(float(x) * float(by));
          h[base + k] = x;
          v[e] = float(x);
          acc += v[e] * v[e];
        } else {
          normed[base + k] = T(float(T(g2[e])) * float(T(v[e] * scale)));
        }
      }
    }
  }

  if (do_wait) {
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup bool last;
    if (i == 0) {
      mesh_sys_fence();
      last = atomic_fetch_add_explicit(rdone, 1u, memory_order_relaxed) == rows - 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last && i == 0) {
      atomic_store_explicit(rdone, 0u, memory_order_relaxed);
      ulong seq = ulong(cfg[10]) | (ulong(cfg[11]) << 32);
      mesh_ring_ret(db, cfg[9], seq);
    }
  }
}
template [[host_name("mg_sum_norm_bf16")]] [[kernel]] void mg_sum_norm<bfloat>(const device ulong*, const device uchar*, const device uint*, device const bfloat*, device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*, device bfloat*, device const bfloat*, device ulong*, device atomic_uint*, constant uint*, constant float&, uint, uint, uint, uint, uint);
template [[host_name("mg_sum_norm_f32")]] [[kernel]] void mg_sum_norm<float>(const device ulong*, const device uchar*, const device uint*, device const float*, device const float*, device const float*, device const float*, device float*, device float*, device const float*, device ulong*, device atomic_uint*, constant uint*, constant float&, uint, uint, uint, uint, uint);

// GF(2)^6 FWHT-64 had_int8 consumer (in-switch wmode==3 or zero-CPU-touch linear Hadamard gather wmode==4).
[[kernel]] void mg_sum_norm_had8_bf16(const device ulong* W [[buffer(0)]], const device uchar* arena [[buffer(1)]],
    const device uint* land_idx [[buffer(2)]], device const bfloat* own [[buffer(3)]],
    device const bfloat* residual [[buffer(4)]], device const bfloat* w1 [[buffer(5)]], device const bfloat* w2 [[buffer(6)]],
    device bfloat* h [[buffer(7)]], device bfloat* normed [[buffer(8)]],
    device ulong* db [[buffer(10)]], device atomic_uint* rdone [[buffer(11)]],
    constant uint* cfg [[buffer(12)]], constant float& eps [[buffer(13)]],
    uint i [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]], uint rows [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint ranks = cfg[1], me = cfg[2], dim = cfg[3], twice = cfg[4], wmode = cfg[6], ppf = cfg[7], lbase = cfg[8], flags = cfg[12];
  const bool has_res = (flags & 1u) != 0u;
  const bool has_w1 = (flags & 2u) != 0u;
  const bool pure_sum = (flags & 4u) != 0u;
  const bool do_wait = (ranks > 1u);
  const uint bpr = dim / 64u;
  const uint blk0 = row * bpr;
  const uint pg0 = blk0 / 60u;
  const uint pg1 = (blk0 + bpr - 1u) / 60u;
  const uint npg = pg1 - pg0 + 1u;

  const ulong base = ulong(row) * dim;
  const uint k0 = sg * 64u + lane * 2u;
  float mine0 = (k0 < dim) ? float(own[base + k0]) : 0.0f;
  float mine1 = (k0 + 1u < dim) ? float(own[base + k0 + 1u]) : 0.0f;
  float res0  = (k0 < dim && has_res) ? float(residual[base + k0]) : 0.0f;
  float res1  = (k0 + 1u < dim && has_res) ? float(residual[base + k0 + 1u]) : 0.0f;
  float g10   = (k0 < dim && has_w1) ? float(w1[k0]) : 1.0f;
  float g11   = (k0 + 1u < dim && has_w1) ? float(w1[k0 + 1u]) : 1.0f;
  float g20   = (k0 < dim && twice) ? float(w2[k0]) : 1.0f;
  float g21   = (k0 + 1u < dim && twice) ? float(w2[k0 + 1u]) : 1.0f;

  threadgroup uint tg_phys[32];
  if (do_wait) {
    if (wmode == 4u) {
      const uint total_polls = ranks * npg;
      for (uint idx = i; idx < total_polls; idx += 32u) {
        const uint q = idx / npg;
        const uint po = idx % npg;
        tg_phys[q * 4u + po] = (q == me) ? 0xffffffffu : mesh_poll_land(W, land_idx[lbase + q * ppf + pg0 + po], 0u);
      }
    } else {
      if (i < npg) tg_phys[i] = mesh_poll_land(W, land_idx[lbase + pg0 + i], ranks - 1u);
    }
    threadgroup_barrier(mem_flags::mem_device);
    mesh_sys_fence();
  }

  float a = 0.0f, b = 0.0f;
  if (do_wait && sg < bpr) {
    const uint blk = blk0 + sg;
    const uint pg = blk / 60u;
    const uint blk_in_pg = blk % 60u;
    if (wmode == 4u) {
      for (uint q = 0; q < ranks; q++) {
        if (q == me) continue;
        const uint phys = tg_phys[q * 4u + (pg - pg0)];
        if (phys == 0xffffffffu) continue;
        const ulong off = ulong(phys) * 4096ul + ulong(blk_in_pg) * 68ul;
        const float sc = float(mesh_sys_read<half>(arena + off, lane >= 16u ? 1u : 0u));
        a += float(mesh_sys_read<char>(arena + off + 4ul, lane * 2u)) * sc;
        b += float(mesh_sys_read<char>(arena + off + 4ul, lane * 2u + 1u)) * sc;
      }
      fwht64_simd_pair(a, b, lane);
    } else {
      const uint phys = tg_phys[pg - pg0];
      if (phys != 0xffffffffu) {
        const ulong off = ulong(phys) * 4096ul + ulong(blk_in_pg) * 68ul;
        const float sc = float(mesh_sys_read<half>(arena + off, lane >= 16u ? 1u : 0u));
        a = float(mesh_sys_read<char>(arena + off + 4ul, lane * 2u)) * sc;
        b = float(mesh_sys_read<char>(arena + off + 4ul, lane * 2u + 1u)) * sc;
        fwht64_simd_pair(a, b, lane);
      }
    }
  }

  if (pure_sum) {
    if (k0 < dim) h[base + k0] = bfloat(mine0 + a);
    if (k0 + 1u < dim) h[base + k0 + 1u] = bfloat(mine1 + b);
  } else {
    float v0 = (k0 < dim) ? float(bfloat(mine0 + a)) : 0.0f;
    float v1 = (k0 + 1u < dim) ? float(bfloat(mine1 + b)) : 0.0f;
    float acc = v0 * v0 + v1 * v1;
    threadgroup float sums[32];
    threadgroup float inv[1];
    for (uint pass = 0; pass < 1u + twice; pass++) {
      acc = simd_sum(acc);
      if (sg == 0) sums[lane] = 0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0) sums[sg] = acc;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) {
        const float t = simd_sum(sums[lane]);
        if (lane == 0) inv[0] = metal::precise::rsqrt(t / dim + eps);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float scale = inv[0];
      acc = 0.0f;
      if (k0 < dim) {
        if (pass == 0) {
          bfloat nv0 = has_w1 ? bfloat(float(bfloat(g10)) * float(bfloat(v0 * scale))) : bfloat(v0 * scale);
          bfloat nv1 = has_w1 ? bfloat(float(bfloat(g11)) * float(bfloat(v1 * scale))) : bfloat(v1 * scale);
          bfloat x0 = has_res ? bfloat(res0 + float(nv0)) : nv0;
          bfloat x1 = has_res ? bfloat(res1 + float(nv1)) : nv1;
          h[base + k0] = x0; h[base + k0 + 1u] = x1;
          v0 = float(x0); v1 = float(x1);
          acc = v0 * v0 + v1 * v1;
        } else {
          normed[base + k0] = bfloat(float(bfloat(g20)) * float(bfloat(v0 * scale)));
          normed[base + k0 + 1u] = bfloat(float(bfloat(g21)) * float(bfloat(v1 * scale)));
        }
      }
    }
  }

  if (do_wait) {
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup bool last;
    if (i == 0) {
      mesh_sys_fence();
      last = atomic_fetch_add_explicit(rdone, 1u, memory_order_relaxed) == rows - 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (last && i == 0) {
      atomic_store_explicit(rdone, 0u, memory_order_relaxed);
      ulong seq = ulong(cfg[10]) | (ulong(cfg[11]) << 32);
      mesh_ring_ret(db, cfg[9], seq);
    }
  }
}

// ---- 6. Fused Head RMSNorm + RoPE + Direct KV Cache Write (reads pos_buf[0] on GPU!) ----
[[kernel]] void mg_head_norm_rope_kv(const device bfloat* x [[buffer(0)]], const device bfloat* w0 [[buffer(1)]],
    const device bfloat* w1 [[buffer(2)]], device bfloat* q_out [[buffer(3)]],
    device bfloat* k_cache [[buffer(4)]], device bfloat* v_cache [[buffer(5)]],
    const device int* pos_buf [[buffer(6)]], const device float* freqs [[buffer(7)]],
    constant uint* p [[buffer(8)]],  // h0, h1, h2, D, rope(0=none,1=base,2=freqs), S, max_seq, window
    constant float* f [[buffer(9)]], // eps, base_log2
    uint i [[thread_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint h0 = p[0], h1 = p[1], h2 = p[2], D = p[3], rope = p[4], S = p[5], max_seq = p[6], window = p[7];
  const float eps = f[0], base_log2 = f[1];
  const uint heads = h0 + h1 + h2, b = tg / heads, g = tg % heads;
  const uint seg = g < h0 ? 0u : g < h0 + h1 ? 1u : 2u;
  const uint h = seg == 0u ? g : seg == 1u ? g - h0 : g - h0 - h1;
  const device bfloat* xr = x + ulong(b) * S + ulong(g) * D;
  const device bfloat* w = seg == 0u ? w0 : w1;
  const bool has_w = seg < 2u, rotate = rope && seg < 2u;
  const uint pos = uint(max(0, pos_buf[0]));
  const uint slot = (window > 0u) ? (pos % max_seq) : min(pos, max_seq - 1u);

  device bfloat* y = (seg == 0u)
                         ? (q_out + (ulong(b) * h0 + h) * D)
                         : ((seg == 1u ? k_cache : v_cache) + ((ulong(b) * h1 + h) * max_seq + slot) * D);

  constexpr uint READS = 4;
  const uint k0 = i * READS;
  float v[READS], acc = 0.0f;
  for (uint e = 0; e < READS; e++) { v[e] = float(xr[k0 + e]); acc += v[e] * v[e]; }
  threadgroup float sums[32];
  threadgroup float inv[1];
  threadgroup float normed[1024];
  acc = simd_sum(acc);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) { const float t = simd_sum(sums[lane]); if (lane == 0) inv[0] = metal::precise::rsqrt(t / D + eps); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = inv[0];
  bfloat nv[READS];
  for (uint e = 0; e < READS; e++) nv[e] = has_w ? bfloat(float(w[k0 + e]) * float(bfloat(v[e] * scale))) : bfloat(v[e] * scale);
  if (!rotate) { for (uint e = 0; e < READS; e++) y[k0 + e] = nv[e]; return; }
  for (uint e = 0; e < READS; e++) normed[k0 + e] = float(nv[e]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_d = D / 2u;
  const float L = float(pos);
  for (uint e = 0; e < READS; e++) {
    const uint k = k0 + e, j = k < half_d ? k : k - half_d;
    const float inv_freq = (rope == 1u) ? metal::exp2(-(float(j) / float(half_d)) * base_log2) : (1.0f / freqs[j]);
    const float theta = L * inv_freq, c = metal::fast::cos(theta), sn = metal::fast::sin(theta);
    const float a = normed[j], bb = normed[j + half_d];
    y[k] = bfloat(k < half_d ? a * c - bb * sn : a * sn + bb * c);
  }
}

// ---- 7. High-Throughput Fused GQA Decode Attention (8 simdgroups/TG, 0 barriers in KV loop!) ----
// Supports D=256 (8 elems/lane) and D=512 (16 elems/lane), sliding window or full causal, reads pos_buf[0] on GPU.
template <uint D>
[[kernel]] void mg_kv_sdpa_decode(const device bfloat* Q [[buffer(0)]], const device bfloat* Kc [[buffer(1)]],
    const device bfloat* Vc [[buffer(2)]], device bfloat* Out [[buffer(3)]], const device int* pos_buf [[buffer(4)]],
    constant uint* u [[buffer(5)]], // B, h_q, h_kv, max_seq, window
    constant float& scale [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint E = D / 32u; // 8 for D=256, 16 for D=512
  constexpr uint H = E / 4u;  // 2 for D=256, 4 for D=512
  const uint h_q = u[1], h_kv = u[2], max_seq = u[3], window = u[4];
  const uint qh = tg.x, b = tg.y;
  const uint kvh = (qh * h_kv) / h_q;
  const uint pos = uint(max(0, pos_buf[0]));
  const uint len = pos + 1u;
  const uint start = (window > 0u && len > window) ? (len - window) : 0u;

  const device vec<bfloat, 4>* q4 = (const device vec<bfloat, 4>*)(Q + (ulong(b) * h_q + qh) * D);
  float4 qv[H];
  float4 ov[H];
  for (uint h = 0; h < H; h++) {
    qv[h] = float4(q4[lane * H + h]);
    ov[h] = float4(0.0f);
  }
  const ulong kv_base = (ulong(b) * h_kv + kvh) * max_seq;
  float m_i = -INFINITY;
  float l_i = 0.0f;

  for (uint t = start + sg; t < len; t += 8u) {
    const uint slot = (window > 0u) ? (t % max_seq) : t;
    const device vec<bfloat, 4>* k4 = (const device vec<bfloat, 4>*)(Kc + (kv_base + slot) * D);
    const device vec<bfloat, 4>* v4 = (const device vec<bfloat, 4>*)(Vc + (kv_base + slot) * D);
    float dotp = 0.0f;
    for (uint h = 0; h < H; h++) dotp += dot(qv[h], float4(k4[lane * H + h]));
    const float sc = simd_sum(dotp) * scale;
    const float m_new = max(m_i, sc);
    const float alpha = (m_i == -INFINITY) ? 0.0f : metal::fast::exp(m_i - m_new);
    const float beta = metal::fast::exp(sc - m_new);
    l_i = l_i * alpha + beta;
    for (uint h = 0; h < H; h++) ov[h] = ov[h] * alpha + beta * float4(v4[lane * H + h]);
    m_i = m_new;
  }

  // Merge 8 simdgroups via threadgroup memory.
  threadgroup float sm_m[8];
  threadgroup float sm_l[8];
  threadgroup float sm_o[8][D];
  if (lane == 0u) { sm_m[sg] = m_i; sm_l[sg] = l_i; }
  for (uint h = 0; h < H; h++) {
    const uint idx = (lane * H + h) * 4u;
    sm_o[sg][idx + 0u] = ov[h].x; sm_o[sg][idx + 1u] = ov[h].y;
    sm_o[sg][idx + 2u] = ov[h].z; sm_o[sg][idx + 3u] = ov[h].w;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float gm = -INFINITY;
    for (uint s = 0; s < 8u; s++) gm = max(gm, sm_m[s]);
    float gl = 0.0f;
    float w[8];
    for (uint s = 0; s < 8u; s++) {
      w[s] = (sm_l[s] > 0.0f) ? metal::fast::exp(sm_m[s] - gm) : 0.0f;
      gl += sm_l[s] * w[s];
    }
    const float inv_l = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    device bfloat* dst = Out + (ulong(b) * h_q + qh) * D;
    for (uint h = 0; h < H; h++) {
      const uint idx = (lane * H + h) * 4u;
      for (uint c = 0; c < 4u; c++) {
        float acc = 0.0f;
        for (uint s = 0; s < 8u; s++) acc += sm_o[s][idx + c] * w[s];
        dst[idx + c] = bfloat(acc * inv_l);
      }
    }
  }
}
template [[host_name("mg_kv_sdpa_decode_256")]] [[kernel]] void mg_kv_sdpa_decode<256>(const device bfloat*, const device bfloat*, const device bfloat*, device bfloat*, const device int*, constant uint*, constant float&, uint2, uint, uint);
template [[host_name("mg_kv_sdpa_decode_512")]] [[kernel]] void mg_kv_sdpa_decode<512>(const device bfloat*, const device bfloat*, const device bfloat*, device bfloat*, const device int*, constant uint*, constant float&, uint2, uint, uint);

// ---- 8. General elementwise ops (bf16 & f32) ----
template <typename T>
[[kernel]] void mg_elementwise(const device T* a [[buffer(0)]], const device T* b [[buffer(1)]],
    device T* out [[buffer(2)]], constant uint* u [[buffer(3)]], constant float& f0 [[buffer(4)]],
    uint t [[thread_position_in_grid]]) {
  const uint n = u[0], subop = u[1], dim = u[2], stride = u[3], layer = u[4];
  if (t >= n) return;
  float av = float(a[t]);
  float r = av;
  if (subop == 0u) r = av + float(b[t]);
  else if (subop == 1u) r = av * float(b[t]);
  else if (subop == 2u) r = av * f0;
  else if (subop == 3u) r = (av + float(b[t])) * f0;
  else if (subop == 4u) r = gelu_approx(av);
  else if (subop == 5u) r = silu_fn(av);
  else if (subop == 6u) r = max(av, 0.0f);
  else if (subop == 7u) r = av;
  else if (subop == 8u) {
    const uint row = t / dim, col = t % dim;
    r = float(bfloat(gelu_approx(av))) * float(b[ulong(row) * stride + ulong(layer) * dim + col]);
  } else if (subop == 9u) {
    r = (av + float(b[t])) * f0;
  }
  out[t] = T(r);
}
template [[host_name("mg_elementwise_bf16")]] [[kernel]] void mg_elementwise<bfloat>(const device bfloat*, const device bfloat*, device bfloat*, constant uint*, constant float&, uint);
template [[host_name("mg_elementwise_f32")]] [[kernel]] void mg_elementwise<float>(const device float*, const device float*, device float*, constant uint*, constant float&, uint);

// ---- 9. Vocab-Parallel Row Argmax + Flat-DMA Crossing + In-GPU tok_buf/pos_buf update ----
constant constexpr uint ARGMAX_CHUNK = 2048;
inline void best_of(thread float& v, thread uint& at, float v2, uint at2) { if (v2 > v || (v2 == v && at2 < at)) { v = v2; at = at2; } }
inline void best_reduce(thread float& v, thread uint& at, uint lane, uint sg, uint nsg, threadgroup float* tv, threadgroup uint* ti) {
  for (ushort off = 16; off >= 1; off >>= 1) best_of(v, at, simd_shuffle_down(v, off), simd_shuffle_down(at, off));
  if (lane == 0) { tv[sg] = v; ti[sg] = at; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    v = lane < nsg ? tv[lane] : -INFINITY; at = lane < nsg ? ti[lane] : 0xffffffffu;
    for (ushort off = 16; off >= 1; off >>= 1) best_of(v, at, simd_shuffle_down(v, off), simd_shuffle_down(at, off));
  }
}
[[kernel]] void mg_argmax_parts(device const bfloat* lg [[buffer(0)]], device uint2* part [[buffer(1)]], constant uint* g [[buffer(2)]],
    uint i [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]], uint2 tgs [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint2 nt2 [[threads_per_threadgroup]]) {
  const uint nt = nt2.x;
  const uint V = g[0], p = tg.x, b = tg.y, P = tgs.x, k0 = p * ARGMAX_CHUNK, k1 = min(V, k0 + ARGMAX_CHUNK);
  device const bfloat* row = lg + ulong(b) * V;
  float v = -INFINITY; uint at = 0xffffffffu;
  for (uint k = k0 + i * 4; k < k1; k += nt * 4) {
    const vec<bfloat, 4> x = *(device const vec<bfloat, 4>*)(row + k);
    for (uint e = 0; e < 4; e++) best_of(v, at, float(x[e]), k + e);
  }
  threadgroup float tv[32]; threadgroup uint ti[32];
  best_reduce(v, at, lane, sg, (nt + 31) / 32, tv, ti);
  if (i == 0) part[ulong(b) * P + p] = uint2(as_type<uint>(v), at);
}

[[kernel]] void mg_argmax_pick_pub(device const uint2* part [[buffer(0)]], device uint2* local_best [[buffer(1)]],
    device uchar* send [[buffer(2)]], device ulong* db [[buffer(3)]], device int* out_tok [[buffer(4)]],
    device int* tok_buf [[buffer(5)]], device int* pos_buf [[buffer(6)]],
    constant uint* g [[buffer(7)]], // P, offset, B, ranks, db_slot, seq_lo, seq_hi, update_inplace
    uint i [[thread_index_in_threadgroup]], uint b [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
  const uint P = g[0], offset = g[1], ranks = g[3], upd = g[7];
  float v = -INFINITY; uint at = 0xffffffffu;
  for (uint p = i; p < P; p += nt) { const uint2 q = part[ulong(b) * P + p]; best_of(v, at, as_type<float>(q.x), q.y); }
  threadgroup float tv[32]; threadgroup uint ti[32];
  best_reduce(v, at, lane, sg, (nt + 31) / 32, tv, ti);
  if (i == 0) {
    const uint idx = at + offset;
    local_best[b] = uint2(as_type<uint>(v), idx);
    if (ranks > 1u) {
      mesh_sys_write<uint>(send, ulong(b) * 2ul, as_type<uint>(v));
      mesh_sys_write<uint>(send, ulong(b) * 2ul + 1ul, idx);
    } else {
      out_tok[b] = int(idx);
      if (upd) {
        tok_buf[b] = int(idx);
        if (b == 0u) pos_buf[0] += 1;
      }
    }
  }
}

[[kernel]] void mg_argmax_ring_pub(device ulong* db [[buffer(0)]], constant uint* g [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i == 0) {
    ulong seq = ulong(g[1]) | (ulong(g[2]) << 32);
    mesh_ring_pub(db, g[0], seq);
  }
}

[[kernel]] void mg_argmax_gather_pick(const device ulong* W [[buffer(0)]], const device uchar* arena [[buffer(1)]],
    const device uint* arg_land [[buffer(2)]], const device uint2* local_best [[buffer(3)]],
    device int* out_tok [[buffer(4)]], device int* tok_buf [[buffer(5)]], device int* pos_buf [[buffer(6)]],
    device ulong* db [[buffer(7)]],
    constant uint* g [[buffer(8)]], // rows, ranks, me, lbase, db_slot, seq_lo, seq_hi, update_inplace
    uint i [[thread_index_in_threadgroup]], uint b [[thread_position_in_grid]]) {
  const uint rows = g[0], ranks = g[1], me = g[2], lbase = g[3], upd = g[7];
  threadgroup uint phys[8];
  if (i < ranks) {
    phys[i] = (i == me) ? 0xffffffffu : mesh_poll_land(W, arg_land[lbase + i], 0u);
  }
  threadgroup_barrier(mem_flags::mem_device);
  mesh_sys_fence();
  if (b < rows) {
    const uint2 lb = local_best[b];
    float best = as_type<float>(lb.x);
    uint at = lb.y;
    for (uint q = 0; q < ranks; q++) {
      if (q == me || phys[q] == 0xffffffffu) continue;
      const device uchar* p = arena + ulong(phys[q]) * 4096ul;
      const float v = as_type<float>(mesh_sys_read<uint>(p, ulong(b) * 2ul));
      const uint k = mesh_sys_read<uint>(p, ulong(b) * 2ul + 1ul);
      if (v > best || (v == best && k < at)) { best = v; at = k; }
    }
    out_tok[b] = int(at);
    if (upd) {
      tok_buf[b] = int(at);
      if (b == 0u) pos_buf[0] += 1;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (i == 0) {
    ulong seq = ulong(g[5]) | (ulong(g[6]) << 32);
    mesh_ring_ret(db, g[4], seq);
  }
}
)MSL";
  s += launch_instantiations();
  return s;
}

std::string build_gemm_msl() {
  return std::string("#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n#pragma METAL internals : enable\n") + R"MSL(
using namespace metal;
#define UNROLL_ALL _Pragma("clang loop unroll(full)")
template <typename T>
inline void mesh_sys_write(device uchar* p, ulong idx, T v) { ((volatile coherent(system) device T*)p)[idx] = v; }
inline void mesh_sys_fence() {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, static_cast<thread_scope>(3));
}
inline void mesh_ring_pub(device ulong* db, uint slot, ulong seq) {
  mesh_sys_fence();
  ((volatile coherent(system) device ulong*)db)[ulong(slot) * 8ul] = seq;
  mesh_sys_fence();
}
inline void group_arrive_db(device ulong* db, device atomic_uint* done, constant uint* pub_cfg, uint arrivals, uint i, threadgroup bool& last, bool stored) {
  if (stored) mesh_sys_fence();
  threadgroup_barrier(mem_flags::mem_device);
  if (i == 0) {
    mesh_sys_fence();
    last = atomic_fetch_add_explicit(done, 1u, memory_order_relaxed) == arrivals - 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last && i == 0) {
    atomic_store_explicit(done, 0u, memory_order_relaxed);
    ulong seq = ulong(pub_cfg[1]) | (ulong(pub_cfg[2]) << 32);
    mesh_ring_pub(db, pub_cfg[0], seq);
  }
}

template <typename T, int vecs_per_tg, int k_lanes>
[[kernel]] void gemm_rows_wide(const device T* mat [[buffer(0)]], const device T* in_vec [[buffer(1)]], device T* out_vec [[buffer(3)]],
    constant int* dims [[buffer(4)]], uint3 tid [[threadgroup_position_in_grid]], uint3 tgpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]], uint simd_lid [[thread_index_in_simdgroup]]) {
  typedef float AccT;
  constexpr int unroll = 8, num_simdgroups = k_lanes / 8;
  const int in_vec_size = dims[0], out_vec_size = dims[1], M = dims[2], matrix_ld = dims[3], vector_ld = dims[4];
  constexpr int rows_per_simdgroup = 32 / k_lanes, rows_per_tg = rows_per_simdgroup * num_simdgroups;
  const short k_lane = simd_lid % k_lanes, sg_row = simd_lid / k_lanes;
  const int out_row = tid.y * rows_per_tg + rows_per_simdgroup * simd_gid + sg_row;
  const int row = min(out_row, out_vec_size - 1);
  const device vec<T, 4>* w4 = (const device vec<T, 4>*)(mat + int64_t(row) * matrix_ld);
  const int n_v4 = in_vec_size / 4, n_main = n_v4 - n_v4 % (k_lanes * unroll);
  const device vec<T, 4>* x4 = (const device vec<T, 4>*)in_vec;
  const int x_ld4 = vector_ld / 4, n_chunks = (M + vecs_per_tg - 1) / vecs_per_tg;
  for (int chunk = tid.x; chunk < n_chunks; chunk += tgpg.x) {
    const int vec0 = chunk * vecs_per_tg;
    int x_off4[vecs_per_tg];
    for (int v = 0; v < vecs_per_tg; v++) x_off4[v] = min(vec0 + v, M - 1) * x_ld4;
    AccT result[vecs_per_tg] = {0};
    for (int base = 0; base < n_main; base += k_lanes * unroll) {
      vec<AccT, 4> wf[unroll];
      UNROLL_ALL for (int i = 0; i < unroll; i++) wf[i] = vec<AccT, 4>(w4[base + i * k_lanes + k_lane]);
      UNROLL_ALL for (int v = 0; v < vecs_per_tg; v++) {
        AccT acc = 0;
        UNROLL_ALL for (int i = 0; i < unroll; i++) acc += dot(wf[i], vec<AccT, 4>(x4[x_off4[v] + base + i * k_lanes + k_lane]));
        result[v] += acc;
      }
    }
    for (int idx = n_main + k_lane; idx < n_v4; idx += k_lanes) {
      const vec<AccT, 4> wf = vec<AccT, 4>(w4[idx]);
      UNROLL_ALL for (int v = 0; v < vecs_per_tg; v++) result[v] += dot(wf, vec<AccT, 4>(x4[x_off4[v] + idx]));
    }
    UNROLL_ALL for (int v = 0; v < vecs_per_tg; v++)
      UNROLL_ALL for (ushort off = k_lanes / 2; off >= 1; off >>= 1) result[v] += simd_shuffle_down(result[v], off);
    if (k_lane == 0 && out_row < out_vec_size)
      for (int v = 0; v < vecs_per_tg; v++)
        if (vec0 + v < M) out_vec[int64_t(vec0 + v) * out_vec_size + out_row] = static_cast<T>(result[v]);
  }
}

struct ThinParams { int M, N, K, lda, ldb, ldd, tiles_n, tiles_m, k_iters; };
template <typename T, int UM, int UN, int SM, int SN, int KS>
[[kernel, max_total_threads_per_threadgroup(SM * SN * KS * 32)]] void gemm_rows_nax(
    const device T* A [[buffer(0)]], const device T* B [[buffer(1)]], device T* D [[buffer(3)]],
    const constant ThinParams* params [[buffer(4)]], uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using tensor_t = metal::tensor<device T, extents_t, metal::tensor_inline>;
  constexpr int TSM = 16 * UM, TSN = 16 * UN, BK = 64;
  const int K = params->K, lda = params->lda, ldb = params->ldb;
  const int tgp = tid.x, tile_m = tgp % params->tiles_m, tile_n = tgp / params->tiles_m;
  const int k_group = simd_group_id / (SM * SN), sg = simd_group_id % (SM * SN);
  const int m_off = (tile_m * SM + sg % SM) * TSM, n_off = (tile_n * SN + sg / SM) * TSN;
  const int rows = params->M - m_off, cols = params->N - n_off;
  A += int64_t(m_off) * lda; B += int64_t(n_off) * ldb; D += int64_t(m_off) * params->ldd + n_off;
  tensor_t tA((device T*)A, extents_t(K, rows), strides_t{1, lda});
  tensor_t tB((device T*)B, extents_t(K, cols), strides_t{1, ldb});
  auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(TSM, TSN, BK, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto Dacc = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
  UNROLL_ALL for (uint16_t i = 0; i < Dacc.get_capacity(); i++) Dacc[i] = 0.0f;
  auto Areg = op.template get_left_input_cooperative_tensor<T, T, float>();
  for (int it = 0; it < params->k_iters; it++) {
    if ((it & 3) == 0) threadgroup_barrier(mem_flags::mem_none);
    const int k = (it * KS + k_group) * BK;
    if (k >= K) continue;
    auto a = a_chunk(k); Areg.load(a);
    auto b = b_chunk(k); op.run(Areg, b, Dacc);
  }
  if (KS > 1) {
    using tg_tensor_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;
    constexpr int kBlockSize = TSM * TSN;
    threadgroup float partials[(KS > 1 ? KS - 1 : 1) * SM * SN * kBlockSize];
    if (k_group > 0) {
      tg_tensor_t tP(partials + ((k_group - 1) * SM * SN + sg) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
      Dacc.store(tP);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_group == 0)
      for (int g = 0; g < KS - 1; g++) {
        tg_tensor_t tP(partials + (g * SM * SN + sg) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
        auto part = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
        part.load(tP);
        UNROLL_ALL for (uint16_t i = 0; i < Dacc.get_capacity(); i++) Dacc[i] += part[i];
      }
  }
  if (k_group == 0) {
    auto Dout = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, T>();
    UNROLL_ALL for (uint16_t i = 0; i < Dacc.get_capacity(); i++) Dout[i] = static_cast<T>(Dacc[i]);
    tensor_t tD(D, extents_t(cols, rows), strides_t{1, params->ldd});
    auto d = tD.template slice<TSN, TSM>(0, 0);
    Dout.store(d);
  }
}

template <typename T, int vecs_per_tg, int k_lanes>
[[kernel]] void gemm_publish_wide(const device T* mat [[buffer(0)]], const device T* in_vec [[buffer(1)]], device T* out_vec [[buffer(3)]],
    constant int* dims [[buffer(4)]], device uchar* send [[buffer(5)]], device ulong* db [[buffer(6)]],
    device atomic_uint* done [[buffer(7)]], constant uint* pub_cfg [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]], uint3 tgpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]], uint simd_lid [[thread_index_in_simdgroup]], uint i [[thread_index_in_threadgroup]]) {
  typedef float AccT;
  constexpr int unroll = 8, num_simdgroups = k_lanes / 8;
  const int in_vec_size = dims[0], out_vec_size = dims[1], M = dims[2], matrix_ld = dims[3], vector_ld = dims[4];
  const uint wmode = pub_cfg[3];
  constexpr int rows_per_simdgroup = 32 / k_lanes, rows_per_tg = rows_per_simdgroup * num_simdgroups;
  const short k_lane = simd_lid % k_lanes, sg_row = simd_lid / k_lanes;
  const int out_row = tid.y * rows_per_tg + rows_per_simdgroup * simd_gid + sg_row;
  const int row = min(out_row, out_vec_size - 1);
  const device vec<T, 4>* w4 = (const device vec<T, 4>*)(mat + int64_t(row) * matrix_ld);
  const int n_v4 = in_vec_size / 4, n_main = n_v4 - n_v4 % (k_lanes * unroll);
  const device vec<T, 4>* x4 = (const device vec<T, 4>*)in_vec;
  const int x_ld4 = vector_ld / 4, n_chunks = (M + vecs_per_tg - 1) / vecs_per_tg;
  for (int chunk = tid.x; chunk < n_chunks; chunk += tgpg.x) {
    const int vec0 = chunk * vecs_per_tg;
    int x_off4[vecs_per_tg];
    for (int v = 0; v < vecs_per_tg; v++) x_off4[v] = min(vec0 + v, M - 1) * x_ld4;
    AccT result[vecs_per_tg] = {0};
    for (int base = 0; base < n_main; base += k_lanes * unroll) {
      vec<AccT, 4> wf[unroll];
      UNROLL_ALL for (int q = 0; q < unroll; q++) wf[q] = vec<AccT, 4>(w4[base + q * k_lanes + k_lane]);
      UNROLL_ALL for (int v = 0; v < vecs_per_tg; v++) {
        AccT acc = 0;
        UNROLL_ALL for (int q = 0; q < unroll; q++) acc += dot(wf[q], vec<AccT, 4>(x4[x_off4[v] + base + q * k_lanes + k_lane]));
        result[v] += acc;
      }
    }
    for (int idx = n_main + k_lane; idx < n_v4; idx += k_lanes) {
      const vec<AccT, 4> wf = vec<AccT, 4>(w4[idx]);
      UNROLL_ALL for (int v = 0; v < vecs_per_tg; v++) result[v] += dot(wf, vec<AccT, 4>(x4[x_off4[v] + idx]));
    }
    UNROLL_ALL for (int v = 0; v < vecs_per_tg; v++)
      UNROLL_ALL for (ushort off = k_lanes / 2; off >= 1; off >>= 1) result[v] += simd_shuffle_down(result[v], off);
    if (k_lane == 0 && out_row < out_vec_size)
      for (int v = 0; v < vecs_per_tg; v++)
        if (vec0 + v < M) {
          const T y = static_cast<T>(result[v]);
          const ulong at = ulong(vec0 + v) * ulong(out_vec_size) + ulong(out_row);
          out_vec[at] = y;
          if (wmode == 1u) mesh_sys_write<float>(send, at, float(y));
          else if (wmode < 3u) mesh_sys_write<T>(send, at, y);
        }
  }
  threadgroup bool last;
  if (wmode < 3u) group_arrive_db(db, done, pub_cfg, tgpg.x * tgpg.y, i, last, k_lane == 0 && out_row < out_vec_size);
}

template <typename T, int UM, int UN, int SM, int SN, int KS>
[[kernel, max_total_threads_per_threadgroup(SM * SN * KS * 32)]] void gemm_publish_nax(
    const device T* A0 [[buffer(0)]], const device T* B0 [[buffer(1)]], device T* D0 [[buffer(3)]],
    const constant ThinParams* params [[buffer(4)]], device uchar* send [[buffer(5)]], device ulong* db [[buffer(6)]],
    device atomic_uint* done [[buffer(7)]], constant uint* pub_cfg [[buffer(8)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]], uint3 tid [[threadgroup_position_in_grid]],
    uint3 tgpg [[threadgroups_per_grid]], uint i [[thread_index_in_threadgroup]]) {
  using extents_t = metal::dextents<int32_t, 2>;
  using strides_t = metal::array<int32_t, 2>;
  using tensor_t = metal::tensor<device T, extents_t, metal::tensor_inline>;
  constexpr int TSM = 16 * UM, TSN = 16 * UN, BK = 64;
  const uint wmode = pub_cfg[3];
  const int K = params->K, lda = params->lda, ldb = params->ldb;
  const int tgp = tid.x, tile_m = tgp % params->tiles_m, tile_n = tgp / params->tiles_m;
  const int k_group = simd_group_id / (SM * SN), sg = simd_group_id % (SM * SN);
  const int m_off = (tile_m * SM + sg % SM) * TSM, n_off = (tile_n * SN + sg / SM) * TSN;
  const int rows = params->M - m_off, cols = params->N - n_off;
  const device T* A = A0 + int64_t(m_off) * lda;
  const device T* B = B0 + int64_t(n_off) * ldb;
  device T* D = D0 + int64_t(m_off) * params->ldd + n_off;
  tensor_t tA((device T*)A, extents_t(K, rows), strides_t{1, lda});
  tensor_t tB((device T*)B, extents_t(K, cols), strides_t{1, ldb});
  auto a_chunk = [&](int k) { return tA.template slice<BK, TSM>(k, 0); };
  auto b_chunk = [&](int k) { return tB.template slice<BK, TSN>(k, 0); };
  using a_chunk_t = decltype(a_chunk(0));
  using b_chunk_t = decltype(b_chunk(0));
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(TSM, TSN, BK, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto Dacc = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
  UNROLL_ALL for (uint16_t e = 0; e < Dacc.get_capacity(); e++) Dacc[e] = 0.0f;
  auto Areg = op.template get_left_input_cooperative_tensor<T, T, float>();
  for (int it = 0; it < params->k_iters; it++) {
    if ((it & 3) == 0) threadgroup_barrier(mem_flags::mem_none);
    const int k = (it * KS + k_group) * BK;
    if (k >= K) continue;
    auto a = a_chunk(k); Areg.load(a);
    auto b = b_chunk(k); op.run(Areg, b, Dacc);
  }
  if (KS > 1) {
    using tg_tensor_t = metal::tensor<threadgroup float, extents_t, metal::tensor_inline>;
    constexpr int kBlockSize = TSM * TSN;
    threadgroup float partials[(KS > 1 ? KS - 1 : 1) * SM * SN * kBlockSize];
    if (k_group > 0) {
      tg_tensor_t tP(partials + ((k_group - 1) * SM * SN + sg) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
      Dacc.store(tP);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_group == 0)
      for (int gk = 0; gk < KS - 1; gk++) {
        tg_tensor_t tP(partials + (gk * SM * SN + sg) * kBlockSize, extents_t(TSN, TSM), strides_t{1, TSN});
        auto part = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, float>();
        part.load(tP);
        UNROLL_ALL for (uint16_t e = 0; e < Dacc.get_capacity(); e++) Dacc[e] += part[e];
      }
  }
  if (k_group == 0) {
    auto Dout = op.template get_destination_cooperative_tensor<a_chunk_t, b_chunk_t, T>();
    UNROLL_ALL for (uint16_t e = 0; e < Dacc.get_capacity(); e++) Dout[e] = static_cast<T>(Dacc[e]);
    tensor_t tD(D, extents_t(cols, rows), strides_t{1, params->ldd});
    auto d = tD.template slice<TSN, TSM>(0, 0);
    Dout.store(d);
    if (wmode < 3u) {
      UNROLL_ALL for (uint16_t e = 0; e < Dacc.get_capacity(); e++) {
        auto idx = Dacc.get_multidimensional_index(e);
        const ulong at = ulong(m_off + idx[1]) * ulong(params->N) + ulong(n_off + idx[0]);
        if (wmode == 1u) mesh_sys_write<float>(send, at, float(static_cast<T>(Dacc[e])));
        else mesh_sys_write<T>(send, at, static_cast<T>(Dacc[e]));
      }
    }
  }
  threadgroup bool last;
  if (wmode < 3u) group_arrive_db(db, done, pub_cfg, tgpg.x, i, last, true);
}

#define WIDE(NV, KL) template [[host_name("gemm_rows_wide_" #NV "_" #KL)]] [[kernel]] void gemm_rows_wide<bfloat, NV, KL>(const device bfloat*, const device bfloat*, device bfloat*, constant int*, uint3, uint3, uint, uint);
WIDE(2, 16) WIDE(3, 16) WIDE(4, 16) WIDE(5, 16) WIDE(2, 32) WIDE(3, 32) WIDE(4, 32) WIDE(5, 32)
#define NAX(UM, SM, KS) template [[host_name("gemm_rows_nax_" #UM "_" #SM "_" #KS)]] [[kernel]] void gemm_rows_nax<bfloat, UM, 2, SM, 1, KS>(const device bfloat*, const device bfloat*, device bfloat*, const constant ThinParams*, uint, uint3);
NAX(1, 1, 4) NAX(2, 1, 4) NAX(2, 2, 3) NAX(2, 3, 2) NAX(2, 4, 2) NAX(2, 5, 1) NAX(2, 6, 1) NAX(2, 7, 1) NAX(2, 8, 1)
#define WIDE_P(NV, KL) template [[host_name("gemm_publish_wide_" #NV "_" #KL)]] [[kernel]] void gemm_publish_wide<bfloat, NV, KL>(const device bfloat*, const device bfloat*, device bfloat*, constant int*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint3, uint3, uint, uint, uint);
WIDE_P(2, 16) WIDE_P(3, 16) WIDE_P(4, 16) WIDE_P(5, 16) WIDE_P(2, 32) WIDE_P(3, 32) WIDE_P(4, 32) WIDE_P(5, 32)
#define NAX_P(UM, SM, KS) template [[host_name("gemm_publish_nax_" #UM "_" #SM "_" #KS)]] [[kernel]] void gemm_publish_nax<bfloat, UM, 2, SM, 1, KS>(const device bfloat*, const device bfloat*, device bfloat*, const constant ThinParams*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint, uint3, uint3, uint);
NAX_P(1, 1, 4) NAX_P(2, 1, 4) NAX_P(2, 2, 3) NAX_P(2, 3, 2) NAX_P(2, 4, 2) NAX_P(2, 5, 1) NAX_P(2, 6, 1) NAX_P(2, 7, 1) NAX_P(2, 8, 1)
)MSL";
}

} // namespace

struct mesh_graph {
  int world_size = 1;
  int rank = 0;
  int default_wire = MG_WIRE_AUTO;
  struct mesh_ctx ctx{};
  bool attached = false;
  int listen_fd = -1;
  int peer_fd[kMaxRanks] = {-1, -1, -1, -1, -1, -1, -1, -1};
  uint32_t my_node = 0;
  uint32_t nodes[kMaxRanks] = {};

  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> q = nil;
  id<MTLLibrary> lib_core = nil;
  id<MTLLibrary> lib_gemm = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> psos;

  id<MTLBuffer> table_buf = nil;
  NSUInteger table_bias = 0;
  id<MTLBuffer> arena_buf = nil;
  NSUInteger arena_bias = 0;
  id<MTLBuffer> db_buf = nil;
  Doorbell *db = nullptr;
  id<MTLBuffer> counter_buf = nil; // [0..4095]: pub_done (64x64B), [4096..8191]: ret_done

  uint64_t db_encode_seq = 0;
  std::atomic<uint64_t> pub_head{0};
  std::atomic<uint64_t> ret_head{0};
  std::atomic<uint64_t> active_until_ns{0};
  std::atomic<bool> stop_worker{false};
  std::thread worker;

  std::vector<BufSlot> bufs;
  std::vector<CrossingSlot> crossings;
  std::vector<mesh_graph_op> ops;
  uint64_t step_gen = 0;
  int out_tok_buf_id = -1;
};

static id<MTLComputePipelineState> get_pso(mesh_graph *g, id<MTLLibrary> lib, const std::string &name) {
  auto it = g->psos.find(name);
  if (it != g->psos.end()) return it->second;
  NSError *err = nil;
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!fn) {
    fprintf(stderr, "mesh-graph: missing Metal function '%s'\n", name.c_str());
    return nil;
  }
  id<MTLComputePipelineState> pso = [g->dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    fprintf(stderr, "mesh-graph: PSO error for '%s': %s\n", name.c_str(), err.localizedDescription.UTF8String);
    return nil;
  }
  g->psos[name] = pso;
  return pso;
}

static int bootstrap_allgather(mesh_graph *g, const void *sendbuf, void *recvbuf, size_t nbytes) {
  if (g->world_size <= 1) {
    memcpy(recvbuf, sendbuf, nbytes);
    return 0;
  }
  uint8_t *rb = static_cast<uint8_t *>(recvbuf);
  if (g->rank == 0) {
    memcpy(rb, sendbuf, nbytes);
    for (int r = 1; r < g->world_size; r++) {
      if (recv_all(g->peer_fd[r], rb + static_cast<size_t>(r) * nbytes, nbytes) != 0) return -1;
    }
    for (int r = 1; r < g->world_size; r++) {
      if (send_all(g->peer_fd[r], rb, static_cast<size_t>(g->world_size) * nbytes) != 0) return -1;
    }
  } else {
    if (send_all(g->peer_fd[0], sendbuf, nbytes) != 0) return -1;
    if (recv_all(g->peer_fd[0], rb, static_cast<size_t>(g->world_size) * nbytes) != 0) return -1;
  }
  return 0;
}

static int bootstrap_connect(mesh_graph *g, const char *comm_id) {
  if (g->world_size <= 1) return 0;
  std::string host = "127.0.0.1";
  int port = 29700;
  if (comm_id && *comm_id) {
    std::string s(comm_id);
    auto col = s.rfind(':');
    if (col != std::string::npos) {
      host = s.substr(0, col);
      port = std::stoi(s.substr(col + 1));
    }
  }
  if (g->rank == 0) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 || listen(fd, 32) != 0) {
      close(fd);
      return -1;
    }
    g->listen_fd = fd;
    for (int k = 1; k < g->world_size; k++) {
      int cfd = accept(fd, nullptr, nullptr);
      if (cfd < 0) return -1;
      int one = 1;
      setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      uint32_t pr = 0;
      if (recv_all(cfd, &pr, sizeof(pr)) != 0 || pr == 0 || pr >= static_cast<uint32_t>(g->world_size)) {
        close(cfd);
        return -1;
      }
      g->peer_fd[pr] = cfd;
    }
  } else {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    const std::string pstr = std::to_string(port);
    if (getaddrinfo(host.c_str(), pstr.c_str(), &hints, &res) != 0 || !res) return -1;
    int fd = -1;
    for (int attempt = 0; attempt < 200; attempt++) {
      fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
      if (fd >= 0 && connect(fd, res->ai_addr, res->ai_addrlen) == 0) break;
      if (fd >= 0) { close(fd); fd = -1; }
      usleep(25000);
    }
    freeaddrinfo(res);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    uint32_t r = static_cast<uint32_t>(g->rank);
    if (send_all(fd, &r, sizeof(r)) != 0) { close(fd); return -1; }
    g->peer_fd[0] = fd;
  }
  return 0;
}

static void submit_db(mesh_graph *g, const Doorbell &d) {
  if (g->world_size <= 1) return;
  const uint32_t me = static_cast<uint32_t>(g->rank);
  const uint32_t nr = static_cast<uint32_t>(g->world_size);
  const CrossingSlot &c = g->crossings[d.cid];
  static thread_local std::vector<struct mesh_red_xfer> rxfers;
  static thread_local std::vector<struct mesh_xfer> xfers;
  const uint32_t ppf = d.n_pages;
  const uint32_t src0 = c.src_base_page + d.g_slot * ppf;

  if (d.kind == DB_REDUCE) {
    rxfers.clear();
    const size_t idx_off = static_cast<size_t>(d.g_slot) * ppf;
    for (uint32_t r = 0; r < nr; r++) {
      if (r == me) continue;
      const uint32_t dst_node = g->nodes[r];
      const uint32_t *dland = c.land[r].data() + idx_off;
      for (uint32_t p = 0; p < ppf; p++) {
        rxfers.push_back(mesh_red_xfer{
            .src_page = src0 + p,
            .bytes = (p + 1u == ppf) ? d.last_bytes : MESH_PAGE_BYTES,
            .dst_node = dst_node,
            .dst_index = dland[p],
            .op = static_cast<uint8_t>(d.red_op),
            .pad = 0,
            .tag = MESH_TAG_WEIGHT(1),
        });
      }
    }
    if (!rxfers.empty()) mesh_reduce_write(&g->ctx, rxfers.data(), static_cast<uint32_t>(rxfers.size()));
  } else if (d.kind == DB_GATHER) {
    // Chunk-major (8-page bulk8 runs outer, peers inner) for Steiner multicast dedup in mesh-link.c!
    xfers.clear();
    const size_t frame_off = static_cast<size_t>(d.g_slot) * nr * ppf + static_cast<size_t>(me) * ppf;
    uint32_t p = 0;
    for (; p + 8u <= ppf; p += 8u) {
      for (uint32_t r = 0; r < nr; r++) {
        if (r == me) continue;
        const uint32_t dst_node = g->nodes[r];
        const uint32_t *dland = c.land[r].data() + frame_off + p;
        for (uint32_t k = 0; k < 8u; k++) {
          xfers.push_back(mesh_xfer{
              .src_page = src0 + p + k,
              .bytes = (p + k + 1u == ppf) ? d.last_bytes : MESH_PAGE_BYTES,
              .dst_node = dst_node,
              .dst_index = dland[k],
          });
        }
      }
    }
    for (; p < ppf; p++) {
      const uint32_t nb = (p + 1u == ppf) ? d.last_bytes : MESH_PAGE_BYTES;
      for (uint32_t r = 0; r < nr; r++) {
        if (r == me) continue;
        const uint32_t dst_node = g->nodes[r];
        const uint32_t *dland = c.land[r].data() + frame_off;
        xfers.push_back(mesh_xfer{
            .src_page = src0 + p,
            .bytes = nb,
            .dst_node = dst_node,
            .dst_index = dland[p],
        });
      }
    }
    if (!xfers.empty()) mesh_write(&g->ctx, xfers.data(), static_cast<uint32_t>(xfers.size()));
  } else if (d.kind == DB_ARGMAX) {
    struct mesh_xfer axfers[kMaxRanks];
    uint32_t nx = 0;
    for (uint32_t r = 0; r < nr; r++) {
      if (r == me) continue;
      axfers[nx++] = mesh_xfer{
          .src_page = src0,
          .bytes = d.last_bytes,
          .dst_node = g->nodes[r],
          .dst_index = c.land[r][d.g_slot * nr + me],
      };
    }
    if (nx) mesh_write(&g->ctx, axfers, nx);
  }
}

static void retire_db(mesh_graph *g, const Doorbell &d) {
  if (g->world_size <= 1) return;
  const uint32_t me = static_cast<uint32_t>(g->rank);
  const CrossingSlot &c = g->crossings[d.cid];
  const size_t off = static_cast<size_t>(d.g_slot) * c.land_per_gen;
  mesh_clear_indices(&g->ctx, c.land[me].data() + off, c.land_per_gen);
}

static void worker_fn(mesh_graph *g) {
  uint64_t phead = 0, rhead = 0;
  while (!g->stop_worker.load(std::memory_order_acquire)) {
    bool did = false;
    for (;;) {
      Doorbell &pd = g->db[phead & kDbRingMask];
      if (pd.pub_seq.load(std::memory_order_acquire) == phead + 1) {
        submit_db(g, pd);
        phead++;
        g->pub_head.store(phead, std::memory_order_release);
        did = true;
      } else {
        break;
      }
    }
    for (;;) {
      Doorbell &rd = g->db[rhead & kDbRingMask];
      if (rd.ret_seq.load(std::memory_order_acquire) == rhead + 1) {
        retire_db(g, rd);
        rhead++;
        g->ret_head.store(rhead, std::memory_order_release);
        did = true;
      } else {
        break;
      }
    }
    if (did) {
      g->active_until_ns.store(now_ns() + 25000000ull, std::memory_order_release);
    } else if (now_ns() < g->active_until_ns.load(std::memory_order_relaxed)) {
      asm volatile("yield");
    } else {
      usleep(25);
    }
  }
}

static uint32_t alloc_db(mesh_graph *g, uint32_t kind, uint32_t cid, uint32_t g_slot,
                         uint32_t n_pages, uint32_t last_bytes, uint8_t red_op, uint64_t &out_seq) {
  const uint64_t seq = ++g->db_encode_seq;
  while (seq - g->ret_head.load(std::memory_order_acquire) >= kDbRingSize - 4) usleep(10);
  const uint32_t slot = static_cast<uint32_t>((seq - 1) & kDbRingMask);
  Doorbell &d = g->db[slot];
  d.pub_seq.store(0, std::memory_order_relaxed);
  d.ret_seq.store(0, std::memory_order_relaxed);
  d.kind = kind;
  d.cid = cid;
  d.g_slot = g_slot;
  d.n_pages = n_pages;
  d.last_bytes = last_bytes;
  d.red_op = red_op;
  std::atomic_thread_fence(std::memory_order_release);
  out_seq = seq;
  g->active_until_ns.store(now_ns() + 25000000ull, std::memory_order_release);
  return slot;
}

static void encode_linear_core(mesh_graph *g, id<MTLComputeCommandEncoder> enc, const mesh_graph_op &op,
                               bool publish, uint32_t g_slot) {
  const BufSlot &bx = g->bufs[op.in[0]], &bw = g->bufs[op.in[1]], &by = g->bufs[op.out[0]];
  const uint32_t M = op.u[0], N = op.u[1], K = op.u[2];
  const bool do_pub = publish && (g->world_size > 1);
  CrossingSlot *c = do_pub ? &g->crossings[op.u[3]] : nullptr;
  uint32_t db_slot = 0;
  uint64_t seq = 0;
  const bool is_had8 = do_pub && (c->wire_mode == MG_WIRE_HAD8 || c->wire_mode == MG_WIRE_HAD8_GATHER);
  const uint32_t wmode = do_pub ? (is_had8 ? 3u : static_cast<uint32_t>(c->wire_mode)) : 3u;

  if (do_pub && !is_had8) {
    const uint32_t kind = (c->wire_mode == MG_WIRE_GATHER) ? DB_GATHER : DB_REDUCE;
    db_slot = alloc_db(g, kind, op.u[3], g_slot, c->pages_per_frame, c->last_page_bytes, c->red_op, seq);
    c->active_db_slot = db_slot;
    c->active_db_seq = seq;
  }

  const NSUInteger send_off = do_pub ? (g->arena_bias + static_cast<NSUInteger>(c->src_base_page + g_slot * c->pages_per_frame) * MESH_PAGE_BYTES) : 0;
  const NSUInteger done_off = do_pub ? (static_cast<NSUInteger>(op.u[3] & 63u) * 64u) : 0;

  if (op.dtype == MG_DTYPE_BF16) {
    // 1. M=1,2: M5-Ultra tuned tiled_rows (gemv_rows / gemv_publish)
    if (M <= 2 && K % 32 == 0) {
      const char *kbase = (do_pub && !is_had8) ? "gemv_publish" : "gemv_rows";
      const Launch l = launch_for(static_cast<uint32_t>(g->world_size), kbase, N, K);
      const uint32_t rows_per_tg = (l.sg / l.s) * l.rows;
      if (N % rows_per_tg == 0 && K % (l.s * l.vec) == 0) {
        id<MTLComputePipelineState> pso = get_pso(g, g->lib_core, launch_name(kbase, M, l));
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx.mtl offset:bx.off atIndex:0];
        [enc setBuffer:bw.mtl offset:bw.off atIndex:1];
        [enc setBuffer:by.mtl offset:by.off atIndex:2];
        if (do_pub && !is_had8) {
          [enc setBuffer:g->arena_buf offset:send_off atIndex:3];
          [enc setBuffer:g->db_buf offset:0 atIndex:4];
          [enc setBuffer:g->counter_buf offset:done_off atIndex:5];
          const uint32_t dims[6] = {N, K, db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32), wmode};
          [enc setBytes:dims length:sizeof(dims) atIndex:6];
        } else {
          const uint32_t dims[2] = {N, K};
          [enc setBytes:dims length:sizeof(dims) atIndex:3];
        }
        [enc dispatchThreadgroups:MTLSizeMake(N / rows_per_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * l.sg, 1, 1)];
        return;
      }
    }
    // 2. M=3..15: gemm_rows_wide / gemm_publish_wide
    if (M >= 3 && M <= 15 && K % 4 == 0) {
      const int passes = static_cast<int>((M + 4) / 5);
      const int nv = static_cast<int>((M + passes - 1) / passes);
      const int kl = (passes == 1) ? 32 : 16;
      const std::string kn = std::string((do_pub && !is_had8) ? "gemm_publish_wide_" : "gemm_rows_wide_") +
                             std::to_string(nv) + "_" + std::to_string(kl);
      id<MTLComputePipelineState> pso = get_pso(g, g->lib_gemm, kn);
      [enc setComputePipelineState:pso];
      [enc setBuffer:bw.mtl offset:bw.off atIndex:0];
      [enc setBuffer:bx.mtl offset:bx.off atIndex:1];
      [enc setBuffer:by.mtl offset:by.off atIndex:3];
      const int dims[5] = {static_cast<int>(K), static_cast<int>(N), static_cast<int>(M), static_cast<int>(K), static_cast<int>(K)};
      [enc setBytes:dims length:sizeof(dims) atIndex:4];
      if (do_pub && !is_had8) {
        [enc setBuffer:g->arena_buf offset:send_off atIndex:5];
        [enc setBuffer:g->db_buf offset:0 atIndex:6];
        [enc setBuffer:g->counter_buf offset:done_off atIndex:7];
        const uint32_t pcfg[4] = {db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32), wmode};
        [enc setBytes:pcfg length:sizeof(pcfg) atIndex:8];
      }
      const int rows_per_tg = (32 / kl) * (kl / 8);
      [enc dispatchThreadgroups:MTLSizeMake(passes, (static_cast<int>(N) + rows_per_tg - 1) / rows_per_tg, 1)
          threadsPerThreadgroup:MTLSizeMake(32 * (kl / 8), 1, 1)];
      return;
    }
    // 3. M=16..128 aligned (M%16==0, N%32==0, K%64==0): Metal 4 MPP NAX tensor ops!
    if ((M == 16 || (M >= 32 && M <= 128 && M % 32 == 0)) && N % 32 == 0 && K % 64 == 0) {
      const int um = (M == 16) ? 1 : 2;
      const int sm = (um == 2 && N > 64 && M > 64) ? std::min<int>(8, (static_cast<int>(M) + 31) / 32) : 1;
      const int ks = (sm == 1) ? 4 : 1 + 4 / sm;
      const std::string kn = std::string((do_pub && !is_had8) ? "gemm_publish_nax_" : "gemm_rows_nax_") +
                             std::to_string(um) + "_" + std::to_string(sm) + "_" + std::to_string(ks);
      id<MTLComputePipelineState> pso = get_pso(g, g->lib_gemm, kn);
      const int tsm = 16 * um;
      const int tm = (static_cast<int>(M) + tsm * sm - 1) / (tsm * sm);
      const int tn = static_cast<int>(N) / 32;
      const int k_iters = ((static_cast<int>(K) + 63) / 64 + ks - 1) / ks;
      const int params[9] = {static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), static_cast<int>(K),
                             static_cast<int>(K), static_cast<int>(N), tn, tm, k_iters};
      [enc setComputePipelineState:pso];
      [enc setBuffer:bx.mtl offset:bx.off atIndex:0];
      [enc setBuffer:bw.mtl offset:bw.off atIndex:1];
      [enc setBuffer:by.mtl offset:by.off atIndex:3];
      [enc setBytes:params length:sizeof(params) atIndex:4];
      if (do_pub && !is_had8) {
        [enc setBuffer:g->arena_buf offset:send_off atIndex:5];
        [enc setBuffer:g->db_buf offset:0 atIndex:6];
        [enc setBuffer:g->counter_buf offset:done_off atIndex:7];
        const uint32_t pcfg[4] = {db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32), wmode};
        [enc setBytes:pcfg length:sizeof(pcfg) atIndex:8];
      }
      [enc dispatchThreadgroups:MTLSizeMake(tm * tn, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * sm * ks, 1, 1)];
      return;
    }
  }

  // 4. Generic fallback for any arbitrary (M, N, K) or F32
  id<MTLComputePipelineState> pso = get_pso(g, g->lib_core, op.dtype == MG_DTYPE_F32 ? "mg_linear_generic_f32" : "mg_linear_generic_bf16");
  [enc setComputePipelineState:pso];
  [enc setBuffer:bx.mtl offset:bx.off atIndex:0];
  [enc setBuffer:bw.mtl offset:bw.off atIndex:1];
  [enc setBuffer:by.mtl offset:by.off atIndex:2];
  const uint32_t u[3] = {M, N, K};
  [enc setBytes:u length:sizeof(u) atIndex:3];
  [enc dispatchThreadgroups:MTLSizeMake((N + 3u) / 4u, M, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  if (do_pub && !is_had8) {
    id<MTLComputePipelineState> ppub = get_pso(g, g->lib_core, op.dtype == MG_DTYPE_F32 ? "mg_write_pub_f32" : "mg_write_pub_bf16");
    [enc setComputePipelineState:ppub];
    [enc setBuffer:by.mtl offset:by.off atIndex:0];
    [enc setBuffer:g->arena_buf offset:send_off atIndex:1];
    [enc setBuffer:g->db_buf offset:0 atIndex:2];
    [enc setBuffer:g->counter_buf offset:done_off atIndex:3];
    const uint32_t pcfg[5] = {M * N, db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32),
                              (c->wire_mode == MG_WIRE_F32 && op.dtype == MG_DTYPE_BF16) ? 1u : 0u};
    [enc setBytes:pcfg length:sizeof(pcfg) atIndex:4];
    const NSUInteger n4 = (M * N) / 4u;
    [enc dispatchThreads:MTLSizeMake(n4, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(n4, 256), 1, 1)];
  }
}

static void encode_publish_only(mesh_graph *g, id<MTLComputeCommandEncoder> enc, const BufSlot &bx,
                                uint32_t cid, uint32_t g_slot, uint32_t dtype) {
  CrossingSlot &c = g->crossings[cid];
  const bool is_gather = (c.wire_mode == MG_WIRE_GATHER || c.wire_mode == MG_WIRE_HAD8_GATHER);
  uint64_t seq = 0;
  const uint32_t db_slot = alloc_db(g, is_gather ? DB_GATHER : DB_REDUCE, cid, g_slot,
                                    c.pages_per_frame, c.last_page_bytes, c.red_op, seq);
  c.active_db_slot = db_slot;
  c.active_db_seq = seq;
  const NSUInteger send_off = g->arena_bias + static_cast<NSUInteger>(c.src_base_page + g_slot * c.pages_per_frame) * MESH_PAGE_BYTES;
  const NSUInteger done_off = static_cast<NSUInteger>(cid & 63u) * 64u;
  if (c.wire_mode == MG_WIRE_HAD8 || c.wire_mode == MG_WIRE_HAD8_GATHER) {
    const uint32_t n_blks = c.n_elems / 64u;
    const uint32_t tgs = (n_blks + 3u) / 4u;
    [enc setComputePipelineState:get_pso(g, g->lib_core, "mg_had8_pub_bf16")];
    [enc setBuffer:bx.mtl offset:bx.off atIndex:0];
    [enc setBuffer:g->arena_buf offset:send_off atIndex:1];
    [enc setBuffer:g->db_buf offset:0 atIndex:2];
    [enc setBuffer:g->counter_buf offset:done_off atIndex:3];
    const uint32_t cfg[4] = {n_blks, db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32)};
    [enc setBytes:cfg length:sizeof(cfg) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  } else {
    [enc setComputePipelineState:get_pso(g, g->lib_core, dtype == MG_DTYPE_F32 ? "mg_write_pub_f32" : "mg_write_pub_bf16")];
    [enc setBuffer:bx.mtl offset:bx.off atIndex:0];
    [enc setBuffer:g->arena_buf offset:send_off atIndex:1];
    [enc setBuffer:g->db_buf offset:0 atIndex:2];
    [enc setBuffer:g->counter_buf offset:done_off atIndex:3];
    const uint32_t cfg[5] = {c.n_elems, db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32),
                             (c.wire_mode == MG_WIRE_F32 && dtype == MG_DTYPE_BF16) ? 1u : 0u};
    [enc setBytes:cfg length:sizeof(cfg) atIndex:4];
    const NSUInteger n4 = c.n_elems / 4u;
    [enc dispatchThreads:MTLSizeMake(n4, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(n4, 256), 1, 1)];
  }
}

// Encode the ENTIRE lowered FX graph into ONE single MTLCommandBuffer + ONE single MTLComputeCommandEncoder!
static id<MTLCommandBuffer> encode_step(mesh_graph *g, uint32_t step_idx) {
  const uint32_t g_slot = static_cast<uint32_t>((g->step_gen++) % kGenSlots);
  id<MTLCommandBuffer> cb = [g->q commandBufferWithUnretainedReferences];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

  for (const auto &op : g->ops) {
    switch (op.kind) {
      case MG_OP_EMBEDDING: {
        const BufSlot &bt = g->bufs[op.in[0]], &bw = g->bufs[op.in[1]], &bo = g->bufs[op.out[0]];
        const uint32_t B = op.u[0], D = op.u[1];
        const float sc = (op.f[0] == 0.0f) ? 1.0f : op.f[0];
        [enc setComputePipelineState:get_pso(g, g->lib_core, "mg_embedding_bf16")];
        [enc setBuffer:bt.mtl offset:bt.off atIndex:0];
        [enc setBuffer:bw.mtl offset:bw.off atIndex:1];
        [enc setBuffer:bo.mtl offset:bo.off atIndex:2];
        const uint32_t u[2] = {D, B};
        [enc setBytes:u length:sizeof(u) atIndex:3];
        [enc setBytes:&sc length:sizeof(sc) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(D, B, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(D, 256), 1, 1)];
        break;
      }
      case MG_OP_LINEAR:
        encode_linear_core(g, enc, op, false, g_slot);
        break;
      case MG_OP_LINEAR_PUBLISH:
        encode_linear_core(g, enc, op, true, g_slot);
        break;
      case MG_OP_CROSSING_SUM_NORM:
      case MG_OP_ALLREDUCE: {
        const bool is_allred = (op.kind == MG_OP_ALLREDUCE);
        const uint32_t cid = op.u[0];
        const uint32_t rows = op.u[1], dim = op.u[2];
        const uint32_t published = is_allred ? 0u : op.u[3];
        const uint32_t twice = (op.out[1] >= 0 && op.in[3] >= 0) ? 1u : 0u;
        const uint32_t flags = is_allred ? 4u : ((op.in[1] >= 0 ? 1u : 0u) | (op.in[2] >= 0 ? 2u : 0u));
        const BufSlot &by = g->bufs[op.in[0]];
        const BufSlot &bh = g->bufs[op.out[0]];
        const BufSlot &bn = (twice ? g->bufs[op.out[1]] : bh);
        const BufSlot &bres = (op.in[1] >= 0 ? g->bufs[op.in[1]] : by);
        const BufSlot &bw1 = (op.in[2] >= 0 ? g->bufs[op.in[2]] : by);
        const BufSlot &bw2 = (twice ? g->bufs[op.in[3]] : bw1);

        const bool has_net = (g->world_size > 1);
        CrossingSlot *c = has_net ? &g->crossings[cid] : nullptr;
        const bool is_had8 = has_net && (c->wire_mode == MG_WIRE_HAD8 || c->wire_mode == MG_WIRE_HAD8_GATHER);
        if (has_net && (!published || is_had8)) {
          encode_publish_only(g, enc, by, cid, g_slot, op.dtype);
        }
        const uint32_t wmode = has_net ? static_cast<uint32_t>(c->wire_mode) : 0u;
        const uint32_t ppf = has_net ? c->pages_per_frame : 1u;
        const uint32_t lbase = has_net ? (g_slot * c->land_per_gen) : 0u;
        const uint32_t dbs = has_net ? c->active_db_slot : 0u;
        const uint64_t seq = has_net ? c->active_db_seq : 0u;
        const uint32_t cfg[13] = {rows * dim, static_cast<uint32_t>(g->world_size), static_cast<uint32_t>(g->rank),
                                  dim, twice, 0u, wmode, ppf, lbase, dbs, static_cast<uint32_t>(seq),
                                  static_cast<uint32_t>(seq >> 32), flags};
        const float eps = op.f[0];
        const NSUInteger rdone_off = 4096u + static_cast<NSUInteger>(cid & 63u) * 64u;
        if (is_had8) {
          [enc setComputePipelineState:get_pso(g, g->lib_core, "mg_sum_norm_had8_bf16")];
          [enc setBuffer:g->table_buf offset:g->table_bias atIndex:0];
          [enc setBuffer:g->arena_buf offset:g->arena_bias atIndex:1];
          [enc setBuffer:c->land_buf offset:0 atIndex:2];
          [enc setBuffer:by.mtl offset:by.off atIndex:3];
          [enc setBuffer:bres.mtl offset:bres.off atIndex:4];
          [enc setBuffer:bw1.mtl offset:bw1.off atIndex:5];
          [enc setBuffer:bw2.mtl offset:bw2.off atIndex:6];
          [enc setBuffer:bh.mtl offset:bh.off atIndex:7];
          [enc setBuffer:bn.mtl offset:bn.off atIndex:8];
          [enc setBuffer:g->db_buf offset:0 atIndex:10];
          [enc setBuffer:g->counter_buf offset:rdone_off atIndex:11];
          [enc setBytes:cfg length:sizeof(cfg) atIndex:12];
          [enc setBytes:&eps length:sizeof(eps) atIndex:13];
          [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake((dim / 64u) * 32u, 1, 1)];
        } else {
          [enc setComputePipelineState:get_pso(g, g->lib_core, op.dtype == MG_DTYPE_F32 ? "mg_sum_norm_f32" : "mg_sum_norm_bf16")];
          [enc setBuffer:(has_net ? g->table_buf : by.mtl) offset:(has_net ? g->table_bias : 0) atIndex:0];
          [enc setBuffer:(has_net ? g->arena_buf : by.mtl) offset:(has_net ? g->arena_bias : 0) atIndex:1];
          [enc setBuffer:(has_net ? c->land_buf : by.mtl) offset:0 atIndex:2];
          [enc setBuffer:by.mtl offset:by.off atIndex:3];
          [enc setBuffer:bres.mtl offset:bres.off atIndex:4];
          [enc setBuffer:bw1.mtl offset:bw1.off atIndex:5];
          [enc setBuffer:bw2.mtl offset:bw2.off atIndex:6];
          [enc setBuffer:bh.mtl offset:bh.off atIndex:7];
          [enc setBuffer:bn.mtl offset:bn.off atIndex:8];
          [enc setBuffer:bw1.mtl offset:bw1.off atIndex:9];
          [enc setBuffer:(has_net ? g->db_buf : by.mtl) offset:0 atIndex:10];
          [enc setBuffer:(has_net ? g->counter_buf : by.mtl) offset:(has_net ? rdone_off : 0) atIndex:11];
          [enc setBytes:cfg length:sizeof(cfg) atIndex:12];
          [enc setBytes:&eps length:sizeof(eps) atIndex:13];
          const NSUInteger width = (((dim + 3u) / 4u + 31u) / 32u) * 32u;
          [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        }
        break;
      }
      case MG_OP_RMS_NORM: {
        // Supports standalone RMSNorm(x, w1, eps) and fused AddNorm(p, res, w1, scalar, w2, eps)!
        const BufSlot &bp = g->bufs[op.in[0]];
        const BufSlot &bres = (op.in[1] >= 0 ? g->bufs[op.in[1]] : bp);
        const BufSlot &bw1 = (op.in[2] >= 0 ? g->bufs[op.in[2]] : bp);
        const BufSlot &bscal = (op.in[3] >= 0 ? g->bufs[op.in[3]] : bp);
        const BufSlot &bw2 = (op.in[4] >= 0 ? g->bufs[op.in[4]] : bw1);
        const BufSlot &bh = g->bufs[op.out[0]];
        const BufSlot &bn = (op.out[1] >= 0 ? g->bufs[op.out[1]] : bh);
        const uint32_t rows = op.u[0], dim = op.u[1];
        const uint32_t twice = (op.out[1] >= 0 && op.in[4] >= 0) ? 1u : 0u;
        const uint32_t scaled = (op.in[3] >= 0) ? 1u : 0u;
        const uint32_t flags = (op.in[1] >= 0 ? 1u : 0u) | (op.in[2] >= 0 ? 2u : 0u);
        const uint32_t cfg[13] = {rows * dim, 1u, 0u, dim, twice, scaled, 0u, 1u, 0u, 0u, 0u, 0u, flags};
        const float eps = op.f[0];
        [enc setComputePipelineState:get_pso(g, g->lib_core, op.dtype == MG_DTYPE_F32 ? "mg_sum_norm_f32" : "mg_sum_norm_bf16")];
        [enc setBuffer:bp.mtl offset:bp.off atIndex:0];
        [enc setBuffer:bp.mtl offset:bp.off atIndex:1];
        [enc setBuffer:bp.mtl offset:bp.off atIndex:2];
        [enc setBuffer:bp.mtl offset:bp.off atIndex:3];
        [enc setBuffer:bres.mtl offset:bres.off atIndex:4];
        [enc setBuffer:bw1.mtl offset:bw1.off atIndex:5];
        [enc setBuffer:bw2.mtl offset:bw2.off atIndex:6];
        [enc setBuffer:bh.mtl offset:bh.off atIndex:7];
        [enc setBuffer:bn.mtl offset:bn.off atIndex:8];
        [enc setBuffer:bscal.mtl offset:bscal.off atIndex:9];
        [enc setBuffer:bp.mtl offset:bp.off atIndex:10];
        [enc setBuffer:bp.mtl offset:bp.off atIndex:11];
        [enc setBytes:cfg length:sizeof(cfg) atIndex:12];
        [enc setBytes:&eps length:sizeof(eps) atIndex:13];
        const NSUInteger width = (((dim + 3u) / 4u + 31u) / 32u) * 32u;
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        break;
      }
      case MG_OP_HEAD_NORM_ROPE: {
        const BufSlot &bx = g->bufs[op.in[0]], &bw0 = g->bufs[op.in[1]];
        const BufSlot &bw1 = (op.in[2] >= 0 ? g->bufs[op.in[2]] : bw0);
        const BufSlot &bpos = g->bufs[op.in[3]];
        const BufSlot &bfreqs = (op.in[4] >= 0 ? g->bufs[op.in[4]] : bx);
        const BufSlot &bq = g->bufs[op.out[0]];
        const BufSlot &bk = (op.out[1] >= 0 ? g->bufs[op.out[1]] : bq);
        const BufSlot &bv = (op.out[2] >= 0 ? g->bufs[op.out[2]] : bq);
        const uint32_t B = op.u[0], h0 = op.u[1], h1 = op.u[2], h2 = op.u[3], D = op.u[4], rope = op.u[5], max_seq = op.u[6], window = op.u[7];
        const uint32_t S = (h0 + h1 + h2) * D;
        const uint32_t p[8] = {h0, h1, h2, D, rope, S, max_seq ? max_seq : 1u, window};
        const float f[2] = {op.f[0], op.f[1]};
        [enc setComputePipelineState:get_pso(g, g->lib_core, "mg_head_norm_rope_kv")];
        [enc setBuffer:bx.mtl offset:bx.off atIndex:0];
        [enc setBuffer:bw0.mtl offset:bw0.off atIndex:1];
        [enc setBuffer:bw1.mtl offset:bw1.off atIndex:2];
        [enc setBuffer:bq.mtl offset:bq.off atIndex:3];
        [enc setBuffer:bk.mtl offset:bk.off atIndex:4];
        [enc setBuffer:bv.mtl offset:bv.off atIndex:5];
        [enc setBuffer:bpos.mtl offset:bpos.off atIndex:6];
        [enc setBuffer:bfreqs.mtl offset:bfreqs.off atIndex:7];
        [enc setBytes:p length:sizeof(p) atIndex:8];
        [enc setBytes:f length:sizeof(f) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake(B * (h0 + h1 + h2), 1, 1) threadsPerThreadgroup:MTLSizeMake(D / 4u, 1, 1)];
        break;
      }
      case MG_OP_KV_SDPA_DECODE: {
        const BufSlot &bq = g->bufs[op.in[0]], &bk = g->bufs[op.in[1]], &bv = g->bufs[op.in[2]], &bpos = g->bufs[op.in[3]];
        const BufSlot &bo = g->bufs[op.out[0]];
        const uint32_t B = op.u[0], h_q = op.u[1], h_kv = op.u[2], D = op.u[3], max_seq = op.u[4], window = op.u[5];
        const float sc = (op.f[0] == 0.0f) ? 1.0f : op.f[0];
        const uint32_t u[5] = {B, h_q, h_kv, max_seq, window};
        [enc setComputePipelineState:get_pso(g, g->lib_core, D == 512u ? "mg_kv_sdpa_decode_512" : "mg_kv_sdpa_decode_256")];
        [enc setBuffer:bq.mtl offset:bq.off atIndex:0];
        [enc setBuffer:bk.mtl offset:bk.off atIndex:1];
        [enc setBuffer:bv.mtl offset:bv.off atIndex:2];
        [enc setBuffer:bo.mtl offset:bo.off atIndex:3];
        [enc setBuffer:bpos.mtl offset:bpos.off atIndex:4];
        [enc setBytes:u length:sizeof(u) atIndex:5];
        [enc setBytes:&sc length:sizeof(sc) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(h_q, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        break;
      }
      case MG_OP_GATE_UP_ACT: {
        const BufSlot &bx = g->bufs[op.in[0]], &bwg = g->bufs[op.in[1]], &bwu = g->bufs[op.in[2]], &bo = g->bufs[op.out[0]];
        const uint32_t B = op.u[0], N = op.u[1], K = op.u[2], act_kind = op.u[3];
        if (B <= 2 && K % 32 == 0) {
          const Launch l = launch_for(static_cast<uint32_t>(g->world_size), "gate_up_geglu", N, K);
          const uint32_t rows_per_tg = (l.sg / l.s) * l.rows;
          [enc setComputePipelineState:get_pso(g, g->lib_core, launch_name("gate_up_act", B, l))];
          [enc setBuffer:bx.mtl offset:bx.off atIndex:0];
          [enc setBuffer:bwg.mtl offset:bwg.off atIndex:1];
          [enc setBuffer:bwu.mtl offset:bwu.off atIndex:2];
          [enc setBuffer:bo.mtl offset:bo.off atIndex:3];
          const uint32_t dims[3] = {N, K, act_kind};
          [enc setBytes:dims length:sizeof(dims) atIndex:4];
          [enc dispatchThreadgroups:MTLSizeMake(N / rows_per_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * l.sg, 1, 1)];
        } else {
          // B >= 3: reuse fast NAX/Wide linear + elementwise using out[1] scratch for up!
          mesh_graph_op lin_g{};
          lin_g.kind = MG_OP_LINEAR; lin_g.dtype = op.dtype;
          lin_g.in[0] = op.in[0]; lin_g.in[1] = op.in[1]; lin_g.out[0] = op.out[0];
          lin_g.u[0] = B; lin_g.u[1] = N; lin_g.u[2] = K;
          encode_linear_core(g, enc, lin_g, false, g_slot);
          mesh_graph_op lin_u = lin_g;
          lin_u.in[1] = op.in[2]; lin_u.out[0] = op.out[1];
          encode_linear_core(g, enc, lin_u, false, g_slot);
          // gelu/silu(out[0])
          const uint32_t tot = B * N;
          const uint32_t eu1[5] = {tot, act_kind == 1u ? MG_EW_SILU : MG_EW_GELU, 0, 0, 0};
          const float fz = 0.0f;
          [enc setComputePipelineState:get_pso(g, g->lib_core, "mg_elementwise_bf16")];
          [enc setBuffer:bo.mtl offset:bo.off atIndex:0];
          [enc setBuffer:bo.mtl offset:bo.off atIndex:1];
          [enc setBuffer:bo.mtl offset:bo.off atIndex:2];
          [enc setBytes:eu1 length:sizeof(eu1) atIndex:3];
          [enc setBytes:&fz length:sizeof(fz) atIndex:4];
          [enc dispatchThreads:MTLSizeMake(tot, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(tot, 256), 1, 1)];
          // out[0] *= out[1]
          const BufSlot &bu = g->bufs[op.out[1]];
          const uint32_t eu2[5] = {tot, MG_EW_MUL, 0, 0, 0};
          [enc setBuffer:bo.mtl offset:bo.off atIndex:0];
          [enc setBuffer:bu.mtl offset:bu.off atIndex:1];
          [enc setBuffer:bo.mtl offset:bo.off atIndex:2];
          [enc setBytes:eu2 length:sizeof(eu2) atIndex:3];
          [enc setBytes:&fz length:sizeof(fz) atIndex:4];
          [enc dispatchThreads:MTLSizeMake(tot, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(tot, 256), 1, 1)];
        }
        break;
      }
      case MG_OP_PLE_GATE: {
        const BufSlot &bx = g->bufs[op.in[0]], &bwg = g->bufs[op.in[1]], &bpli = g->bufs[op.in[2]], &bo = g->bufs[op.out[0]];
        const uint32_t B = op.u[0], N = op.u[1], K = op.u[2], stride = op.u[3], layer = op.u[4];
        if (B <= 2 && K % 32 == 0) {
          const Launch l = launch_for(static_cast<uint32_t>(g->world_size), "ple_gate", N, K);
          const uint32_t rows_per_tg = (l.sg / l.s) * l.rows;
          [enc setComputePipelineState:get_pso(g, g->lib_core, launch_name("ple_gate", B, l))];
          [enc setBuffer:bx.mtl offset:bx.off atIndex:0];
          [enc setBuffer:bwg.mtl offset:bwg.off atIndex:1];
          [enc setBuffer:bpli.mtl offset:bpli.off atIndex:2];
          [enc setBuffer:bo.mtl offset:bo.off atIndex:3];
          const uint32_t dims[4] = {N, K, stride, layer};
          [enc setBytes:dims length:sizeof(dims) atIndex:4];
          [enc dispatchThreadgroups:MTLSizeMake(N / rows_per_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * l.sg, 1, 1)];
        } else {
          mesh_graph_op lin{};
          lin.kind = MG_OP_LINEAR; lin.dtype = op.dtype;
          lin.in[0] = op.in[0]; lin.in[1] = op.in[1]; lin.out[0] = op.out[0];
          lin.u[0] = B; lin.u[1] = N; lin.u[2] = K;
          encode_linear_core(g, enc, lin, false, g_slot);
          const uint32_t tot = B * N;
          const uint32_t eu[5] = {tot, MG_EW_MUL_SLICE, N, stride, layer};
          const float fz = 0.0f;
          [enc setComputePipelineState:get_pso(g, g->lib_core, "mg_elementwise_bf16")];
          [enc setBuffer:bo.mtl offset:bo.off atIndex:0];
          [enc setBuffer:bpli.mtl offset:bpli.off atIndex:1];
          [enc setBuffer:bo.mtl offset:bo.off atIndex:2];
          [enc setBytes:eu length:sizeof(eu) atIndex:3];
          [enc setBytes:&fz length:sizeof(fz) atIndex:4];
          [enc dispatchThreads:MTLSizeMake(tot, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(tot, 256), 1, 1)];
        }
        break;
      }
      case MG_OP_ELEMENTWISE: {
        const BufSlot &ba = g->bufs[op.in[0]];
        const BufSlot &bb = (op.in[1] >= 0 ? g->bufs[op.in[1]] : ba);
        const BufSlot &bo = g->bufs[op.out[0]];
        const uint32_t n = op.u[0];
        const uint32_t eu[5] = {n, op.u[1], op.u[2], op.u[3], op.u[4]};
        const float f0 = op.f[0];
        [enc setComputePipelineState:get_pso(g, g->lib_core, op.dtype == MG_DTYPE_F32 ? "mg_elementwise_f32" : "mg_elementwise_bf16")];
        [enc setBuffer:ba.mtl offset:ba.off atIndex:0];
        [enc setBuffer:bb.mtl offset:bb.off atIndex:1];
        [enc setBuffer:bo.mtl offset:bo.off atIndex:2];
        [enc setBytes:eu length:sizeof(eu) atIndex:3];
        [enc setBytes:&f0 length:sizeof(f0) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(n, 256), 1, 1)];
        break;
      }
      case MG_OP_VOCAB_ARGMAX: {
        const BufSlot &blg = g->bufs[op.in[0]];
        const BufSlot &btok = (op.in[1] >= 0 ? g->bufs[op.in[1]] : blg);
        const BufSlot &bpos = (op.in[2] >= 0 ? g->bufs[op.in[2]] : blg);
        const BufSlot &bout = g->bufs[op.out[0]];
        const BufSlot &bpart = g->bufs[op.out[1]];
        const BufSlot &blbest = g->bufs[op.out[2]];
        const uint32_t B = op.u[0], V = op.u[1], offset = op.u[2], cid = op.u[3], upd = op.u[4];
        const uint32_t P = (V + 2047u) / 2048u;
        const bool has_net = (g->world_size > 1);
        g->out_tok_buf_id = op.out[0];
        // Write step's output tokens at step_idx * B so multi-step decode preserves full history!
        const NSUInteger out_off = bout.off + static_cast<NSUInteger>(step_idx) * B * sizeof(int32_t);

        // 1. row_argmax_parts
        [enc setComputePipelineState:get_pso(g, g->lib_core, "mg_argmax_parts")];
        [enc setBuffer:blg.mtl offset:blg.off atIndex:0];
        [enc setBuffer:bpart.mtl offset:bpart.off atIndex:1];
        [enc setBytes:&V length:sizeof(V) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(P, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // 2. row_argmax_pick (+ write to /mesh0 if world_size > 1, else finish directly!)
        uint32_t db_slot = 0;
        uint64_t seq = 0;
        CrossingSlot *c = has_net ? &g->crossings[cid] : nullptr;
        NSUInteger send_off = 0;
        if (has_net) {
          db_slot = alloc_db(g, DB_ARGMAX, cid, g_slot, 1u, B * 8u, 0u, seq);
          send_off = g->arena_bias + static_cast<NSUInteger>(c->src_base_page + g_slot) * MESH_PAGE_BYTES;
        }
        const uint32_t gp[8] = {P, offset, B, static_cast<uint32_t>(g->world_size), db_slot,
                                static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32), upd};
        [enc setComputePipelineState:get_pso(g, g->lib_core, "mg_argmax_pick_pub")];
        [enc setBuffer:bpart.mtl offset:bpart.off atIndex:0];
        [enc setBuffer:blbest.mtl offset:blbest.off atIndex:1];
        [enc setBuffer:(has_net ? g->arena_buf : blbest.mtl) offset:send_off atIndex:2];
        [enc setBuffer:(has_net ? g->db_buf : blbest.mtl) offset:0 atIndex:3];
        [enc setBuffer:bout.mtl offset:out_off atIndex:4];
        [enc setBuffer:btok.mtl offset:btok.off atIndex:5];
        [enc setBuffer:bpos.mtl offset:bpos.off atIndex:6];
        [enc setBytes:gp length:sizeof(gp) atIndex:7];
        const NSUInteger pth = std::min<NSUInteger>(1024, ((P + 31u) / 32u) * 32u);
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(pth, 1, 1)];

        if (has_net) {
          // 3. Ring doorbell once all B threadgroups of mg_argmax_pick_pub finished (serial compute encoder orders it!)
          const uint32_t rg[3] = {db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32)};
          [enc setComputePipelineState:get_pso(g, g->lib_core, "mg_argmax_ring_pub")];
          [enc setBuffer:g->db_buf offset:0 atIndex:0];
          [enc setBytes:rg length:sizeof(rg) atIndex:1];
          [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

          // 4. Await all peers' argmax pages + pick global winner + update tok_buf/pos_buf on GPU + retire!
          const uint32_t ag[8] = {B, static_cast<uint32_t>(g->world_size), static_cast<uint32_t>(g->rank),
                                  g_slot * static_cast<uint32_t>(g->world_size), db_slot,
                                  static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32), upd};
          const NSUInteger wth = std::max<NSUInteger>(B, 32);
          [enc setComputePipelineState:get_pso(g, g->lib_core, "mg_argmax_gather_pick")];
          [enc setBuffer:g->table_buf offset:g->table_bias atIndex:0];
          [enc setBuffer:g->arena_buf offset:g->arena_bias atIndex:1];
          [enc setBuffer:c->land_buf offset:0 atIndex:2];
          [enc setBuffer:blbest.mtl offset:blbest.off atIndex:3];
          [enc setBuffer:bout.mtl offset:out_off atIndex:4];
          [enc setBuffer:btok.mtl offset:btok.off atIndex:5];
          [enc setBuffer:bpos.mtl offset:bpos.off atIndex:6];
          [enc setBuffer:g->db_buf offset:0 atIndex:7];
          [enc setBytes:ag length:sizeof(ag) atIndex:8];
          [enc dispatchThreads:MTLSizeMake(wth, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(wth, 256), 1, 1)];
        }
        break;
      }
      default:
        break;
    }
  }

  [enc endEncoding];
  return cb;
}

extern "C" {

mesh_graph_t *mesh_graph_create(int world_size, int rank, const char *comm_id, int default_wire_mode) {
  auto *g = new mesh_graph();
  g->world_size = world_size < 1 ? 1 : world_size;
  g->rank = rank;
  g->default_wire = default_wire_mode;

  g->dev = MTLCreateSystemDefaultDevice();
  g->q = [g->dev newCommandQueue];
  NSError *err = nil;
  NSString *src_core = [NSString stringWithUTF8String:build_core_msl().c_str()];
  g->lib_core = [g->dev newLibraryWithSource:src_core options:nil error:&err];
  if (!g->lib_core) {
    fprintf(stderr, "mesh-graph: core MSL compile error: %s\n", err.localizedDescription.UTF8String);
    delete g;
    return nullptr;
  }
  NSString *src_gemm = [NSString stringWithUTF8String:build_gemm_msl().c_str()];
  g->lib_gemm = [g->dev newLibraryWithSource:src_gemm options:nil error:&err];
  if (!g->lib_gemm) {
    fprintf(stderr, "mesh-graph: gemm MSL compile error: %s\n", err.localizedDescription.UTF8String);
    delete g;
    return nullptr;
  }

  if (g->world_size > 1) {
    if (mesh_attach(&g->ctx) != 0) {
      fprintf(stderr, "mesh-graph: mesh_attach(/mesh0) failed\n");
      delete g;
      return nullptr;
    }
    g->attached = true;
    g->my_node = hdr_node(g->ctx.M);
    if (bootstrap_connect(g, comm_id) != 0 || bootstrap_allgather(g, &g->my_node, g->nodes, sizeof(uint32_t)) != 0) {
      fprintf(stderr, "mesh-graph: bootstrap failed\n");
      mesh_graph_destroy(g);
      return nullptr;
    }
    const uintptr_t t_addr = reinterpret_cast<uintptr_t>(g->ctx.M) + static_cast<uintptr_t>(hdr_table_off(g->ctx.M));
    const uintptr_t t_aln = t_addr & ~static_cast<uintptr_t>(16383u);
    const size_t t_bytes = ((static_cast<size_t>(hdr_entries(g->ctx.M)) * 8u + (t_addr - t_aln) + 16383u) & ~static_cast<size_t>(16383u));
    g->table_bias = static_cast<NSUInteger>(t_addr - t_aln);
    g->table_buf = [g->dev newBufferWithBytesNoCopy:reinterpret_cast<void *>(t_aln)
                                            length:t_bytes
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];

    const uintptr_t d_addr = reinterpret_cast<uintptr_t>(g->ctx.M) + static_cast<uintptr_t>(hdr_data_off(g->ctx.M));
    const uintptr_t d_aln = d_addr & ~static_cast<uintptr_t>(16383u);
    const uintptr_t d_end = reinterpret_cast<uintptr_t>(g->ctx.M) + static_cast<uintptr_t>(hdr_length(g->ctx.M));
    const size_t d_bytes = (d_end > d_aln) ? (static_cast<size_t>(d_end - d_aln) & ~static_cast<size_t>(16383u)) : 0u;
    g->arena_bias = static_cast<NSUInteger>(d_addr - d_aln);
    g->arena_buf = [g->dev newBufferWithBytesNoCopy:reinterpret_cast<void *>(d_aln)
                                            length:d_bytes
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];

    g->db_buf = [g->dev newBufferWithLength:kDbRingSize * sizeof(Doorbell) options:MTLResourceStorageModeShared];
    g->db = static_cast<Doorbell *>([g->db_buf contents]);
    memset(g->db, 0, kDbRingSize * sizeof(Doorbell));
    g->counter_buf = [g->dev newBufferWithLength:8192 options:MTLResourceStorageModeShared];
    memset([g->counter_buf contents], 0, 8192);
    g->stop_worker.store(false);
    g->worker = std::thread(worker_fn, g);
  }
  return g;
}

int mesh_graph_add_buffer(mesh_graph_t *g, uint64_t nbytes, int kind,
                          const void *init_data, int parent_buf, uint64_t parent_off) {
  BufSlot b{};
  b.nbytes = nbytes;
  b.kind = kind;
  if (kind == MG_BUF_VIEW) {
    const BufSlot &p = g->bufs.at(static_cast<size_t>(parent_buf));
    b.mtl = p.mtl;
    b.off = p.off + static_cast<NSUInteger>(parent_off);
    b.host_ptr = p.host_ptr ? (static_cast<uint8_t *>(p.host_ptr) + parent_off) : nullptr;
  } else {
    const size_t alloc_bytes = std::max<size_t>(static_cast<size_t>(nbytes), 64u);
    // Zero-copy borrow if 16 KiB-aligned weight pointer & 16 KiB multiple length; else single shared upload!
    if (kind == MG_BUF_WEIGHT && init_data &&
        (reinterpret_cast<uintptr_t>(init_data) & 16383u) == 0 &&
        (alloc_bytes & 16383u) == 0) {
      b.mtl = [g->dev newBufferWithBytesNoCopy:const_cast<void *>(init_data)
                                        length:alloc_bytes
                                       options:MTLResourceStorageModeShared
                                   deallocator:nil];
    }
    if (!b.mtl) {
      b.mtl = [g->dev newBufferWithLength:alloc_bytes options:MTLResourceStorageModeShared];
      if (init_data && nbytes) memcpy([b.mtl contents], init_data, static_cast<size_t>(nbytes));
      else memset([b.mtl contents], 0, alloc_bytes);
    }
    b.off = 0;
    b.host_ptr = [b.mtl contents];
  }
  const int id = static_cast<int>(g->bufs.size());
  g->bufs.push_back(b);
  return id;
}

void *mesh_graph_buffer_ptr(mesh_graph_t *g, int buf_id) {
  if (!g || buf_id < 0 || static_cast<size_t>(buf_id) >= g->bufs.size()) return nullptr;
  return g->bufs[static_cast<size_t>(buf_id)].host_ptr;
}

int mesh_graph_add_crossing(mesh_graph_t *g, uint32_t n_elems, uint32_t esize,
                            int wire_mode, int is_argmax) {
  CrossingSlot c{};
  c.n_elems = n_elems;
  c.esize = esize;
  c.is_argmax = is_argmax;
  int wm = (wire_mode < 0) ? g->default_wire : wire_mode;
  if (is_argmax) {
    c.wire_mode = MG_WIRE_GATHER;
    c.pages_per_frame = 1;
    c.last_page_bytes = n_elems * 8u;
  } else {
    if (wm == MG_WIRE_AUTO) {
      wm = (esize == 2 && (n_elems % 64u) == 0 && static_cast<uint64_t>(n_elems) * esize > 2u * MESH_PAGE_BYTES)
               ? MG_WIRE_HAD8_GATHER
               : MG_WIRE_GATHER;
    }
    if ((wm == MG_WIRE_HAD8 || wm == MG_WIRE_HAD8_GATHER) && (esize != 2 || (n_elems % 64u) != 0)) {
      wm = MG_WIRE_GATHER;
    }
    c.wire_mode = wm;
    if (wm == MG_WIRE_HAD8 || wm == MG_WIRE_HAD8_GATHER) {
      c.red_op = MESH_RED_SUM_HAD_I8;
      const uint32_t n_blks = n_elems / 64u;
      c.pages_per_frame = (n_blks + 59u) / 60u;
      const uint32_t rem = n_blks % 60u ? n_blks % 60u : 60u;
      c.last_page_bytes = rem * 68u;
    } else if (wm == MG_WIRE_F32) {
      c.red_op = MESH_RED_SUM_F32;
      const uint64_t bytes = static_cast<uint64_t>(n_elems) * 4u;
      c.pages_per_frame = static_cast<uint32_t>((bytes + MESH_PAGE_BYTES - 1) / MESH_PAGE_BYTES);
      c.last_page_bytes = static_cast<uint32_t>(bytes - static_cast<uint64_t>(c.pages_per_frame - 1) * MESH_PAGE_BYTES);
    } else {
      c.red_op = (esize == 2) ? MESH_RED_SUM_BF16 : MESH_RED_SUM_F32;
      const uint64_t bytes = static_cast<uint64_t>(n_elems) * esize;
      c.pages_per_frame = static_cast<uint32_t>((bytes + MESH_PAGE_BYTES - 1) / MESH_PAGE_BYTES);
      c.last_page_bytes = static_cast<uint32_t>(bytes - static_cast<uint64_t>(c.pages_per_frame - 1) * MESH_PAGE_BYTES);
    }
  }
  if (c.pages_per_frame == 0) c.pages_per_frame = 1;
  const int cid = static_cast<int>(g->crossings.size());
  g->crossings.push_back(std::move(c));
  return cid;
}

int mesh_graph_add_op(mesh_graph_t *g, const struct mesh_graph_op *op) {
  if (!g || !op) return -1;
  g->ops.push_back(*op);
  return static_cast<int>(g->ops.size()) - 1;
}

int mesh_graph_compile(mesh_graph_t *g) {
  if (!g) return -1;
  if (g->world_size > 1 && !g->crossings.empty()) {
    // Allocate contiguous TX pages + W[] landing indices for ALL crossings in ONE batch + ONE TCP allgather!
    uint32_t total_tx_pages = 0;
    uint32_t total_land_idx = 0;
    for (auto &c : g->crossings) {
      c.total_src_pages = kGenSlots * c.pages_per_frame;
      total_tx_pages += c.total_src_pages;
      const bool is_gather = c.is_argmax || (c.wire_mode == MG_WIRE_GATHER || c.wire_mode == MG_WIRE_HAD8_GATHER);
      c.land_per_gen = is_gather ? (static_cast<uint32_t>(g->world_size) * c.pages_per_frame) : c.pages_per_frame;
      total_land_idx += kGenSlots * c.land_per_gen;
    }
    void *tx_base = nullptr;
    if (mesh_alloc_contiguous(&g->ctx, total_tx_pages, nullptr, &tx_base) != 0) return -1;
    uint32_t p_cur = static_cast<uint32_t>(
        (reinterpret_cast<uint8_t *>(tx_base) - (reinterpret_cast<uint8_t *>(g->ctx.M) + hdr_data_off(g->ctx.M))) / MESH_PAGE_BYTES);

    std::vector<uint32_t> my_all_land(total_land_idx, 0);
    if (mesh_alloc(&g->ctx, total_land_idx, 0, my_all_land.data(), nullptr) != 0) return -1;
    mesh_clear_indices(&g->ctx, my_all_land.data(), total_land_idx);

    std::vector<uint32_t> gathered(static_cast<size_t>(g->world_size) * total_land_idx, 0);
    if (bootstrap_allgather(g, my_all_land.data(), gathered.data(), total_land_idx * sizeof(uint32_t)) != 0) return -1;

    uint32_t l_off = 0;
    for (auto &c : g->crossings) {
      c.src_base_page = p_cur;
      p_cur += c.total_src_pages;
      const uint32_t cnt = kGenSlots * c.land_per_gen;
      for (int r = 0; r < g->world_size; r++) {
        const uint32_t *base = gathered.data() + static_cast<size_t>(r) * total_land_idx + l_off;
        c.land[r].assign(base, base + cnt);
      }
      c.land_buf = [g->dev newBufferWithBytes:c.land[g->rank].data()
                                       length:cnt * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];
      l_off += cnt;
    }
  }
  mesh_graph_barrier(g);
  return 0;
}

void mesh_graph_barrier(mesh_graph_t *g) {
  if (!g || g->world_size <= 1) return;
  uint32_t v = 1, out[kMaxRanks] = {};
  bootstrap_allgather(g, &v, out, sizeof(v));
}

uint32_t mesh_graph_node_id(mesh_graph_t *g) { return g ? g->my_node : 0u; }

double mesh_graph_run(mesh_graph_t *g, uint32_t n_steps, uint32_t warm_steps,
                      int32_t *out_tokens, uint32_t out_tok_stride, double *out_wall_ms) {
  if (!g || n_steps == 0) return 0.0;
  @autoreleasepool {
    for (uint32_t w = 0; w < warm_steps; w++) {
      id<MTLCommandBuffer> cb = encode_step(g, 0);
      [cb commit];
      [cb waitUntilCompleted];
    }
    if (g->world_size > 1) {
      while (g->ret_head.load(std::memory_order_acquire) < g->db_encode_seq) usleep(5);
      mesh_graph_barrier(g);
    }

    // Pipelined single-CB execution: encode step k+1 while step k runs on GPU (ZERO Python per op/token!).
    std::vector<double> done_ns;
    done_ns.reserve(n_steps + 1);
    const uint64_t t0 = now_ns();
    done_ns.push_back(static_cast<double>(t0));

    id<MTLCommandBuffer> prev = encode_step(g, 0);
    [prev commit];
    for (uint32_t k = 1; k < n_steps; k++) {
      id<MTLCommandBuffer> nxt = encode_step(g, k);
      [nxt commit];
      [prev waitUntilCompleted];
      done_ns.push_back(static_cast<double>(now_ns()));
      prev = nxt;
    }
    [prev waitUntilCompleted];
    done_ns.push_back(static_cast<double>(now_ns()));

    if (out_tokens && g->out_tok_buf_id >= 0 && out_tok_stride > 0) {
      const void *src = mesh_graph_buffer_ptr(g, g->out_tok_buf_id);
      if (src) memcpy(out_tokens, src, static_cast<size_t>(n_steps) * out_tok_stride * sizeof(int32_t));
    }

    const double wall_ms = (done_ns.back() - done_ns.front()) * 1e-6;
    if (out_wall_ms) *out_wall_ms = wall_ms;
    if (n_steps >= 4) {
      std::vector<double> step_ms;
      step_ms.reserve(n_steps - 2);
      for (uint32_t k = 1; k + 1 < n_steps; k++) step_ms.push_back((done_ns[k + 1] - done_ns[k]) * 1e-6);
      std::sort(step_ms.begin(), step_ms.end());
      return step_ms[step_ms.size() / 2];
    }
    return wall_ms / static_cast<double>(n_steps);
  }
}

void mesh_graph_destroy(mesh_graph_t *g) {
  if (!g) return;
  if (g->worker.joinable()) {
    while (g->ret_head.load(std::memory_order_acquire) < g->db_encode_seq) usleep(10);
    mesh_graph_barrier(g);
    g->stop_worker.store(true, std::memory_order_release);
    g->worker.join();
  }
  if (g->attached) {
    for (auto &c : g->crossings) {
      if (c.src_base_page != MESH_PAGE_NONE) {
        for (uint32_t k = 0; k < c.total_src_pages; k++) mesh_release(&g->ctx, c.src_base_page + k);
      }
      if (!c.land[g->rank].empty()) {
        mesh_clear_indices(&g->ctx, c.land[g->rank].data(), static_cast<uint32_t>(c.land[g->rank].size()));
        mesh_free_indices(&g->ctx, c.land[g->rank].data(), static_cast<uint32_t>(c.land[g->rank].size()));
      }
    }
    mesh_detach(&g->ctx);
    g->attached = false;
  }
  for (int r = 0; r < kMaxRanks; r++) {
    if (g->peer_fd[r] >= 0) { close(g->peer_fd[r]); g->peer_fd[r] = -1; }
  }
  if (g->listen_fd >= 0) { close(g->listen_fd); g->listen_fd = -1; }
  delete g;
}

} // extern "C"
