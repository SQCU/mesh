// Flat-page DMA + in-switch algebraic reduction MLX primitive backend for mesh (origin/main).
// Runs Gemma-4-E2B (and any tensor-parallel MLX program) directly over /mesh0 zero-copy MTLBuffers
// with zero command-buffer splits, zero host memcpy, and native MESH_RED_SUM_{BF16,F32,HAD_I8} in-switch coalescing.
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <arpa/inet.h>
#include <mach/mach_time.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <objc/runtime.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

extern "C" {
#include "mesh-call.h"
#include "mesh-codec.h"
}

namespace nb = nanobind;
namespace mx = mlx::core;

namespace {

constexpr uint32_t kGenSlots = 4;
constexpr uint32_t kDbRingSize = 2048;
constexpr uint32_t kDbRingMask = kDbRingSize - 1;
constexpr uint32_t kSpinGroups = 32;
constexpr uint32_t kMaxRanks = 8;

enum DbKind : uint32_t {
  DB_FUSED_REDUCE = 0,
  DB_FUSED_GATHER = 1,
  DB_FUSED_ARGMAX = 2,
  DB_ALLSUM_F32 = 3,
};

enum WireMode : uint32_t {
  WIRE_BF16 = 0,   // MESH_RED_SUM_BF16 (or F32 if esize==4)
  WIRE_F32 = 1,    // MESH_RED_SUM_F32 (widen bf16 -> f32 on publish)
  WIRE_GATHER = 2, // per-peer mesh_write, rank-ordered fp32 sum on GPU
  WIRE_HAD8 = 3,   // MESH_RED_SUM_HAD_I8 (GF(2)^6 FWHT-64 1.0625 B/elem)
};

struct alignas(64) Doorbell {
  std::atomic<uint64_t> pub_seq;
  std::atomic<uint64_t> ret_seq;
  uint32_t kind;
  uint32_t pid;
  uint32_t g_slot;
  uint32_t j;
  uint32_t n_pages;
  uint32_t last_bytes;
  uint32_t extra; // wire_op or allsum slot
  uint32_t pad[5];
};
static_assert(sizeof(Doorbell) == 64, "Doorbell is one cache line");

// Helpers to read /mesh0 struct hdr without pulling C11 _Atomic headers into C++20.
inline uint32_t hdr_entries(const struct hdr* m) { return *(const uint32_t*)((const uint8_t*)m + 16); }
inline uint32_t hdr_node(const struct hdr* m) { return *(const uint32_t*)((const uint8_t*)m + 20); }
inline uint64_t hdr_table_off(const struct hdr* m) { return *(const uint64_t*)((const uint8_t*)m + 32); }
inline uint64_t hdr_data_off(const struct hdr* m) { return *(const uint64_t*)((const uint8_t*)m + 88); }
inline uint64_t hdr_length(const struct hdr* m) { return *(const uint64_t*)((const uint8_t*)m + 96); }
inline uint64_t hdr_map_off(const struct hdr* m) { return *(const uint64_t*)((const uint8_t*)m + 144); }
inline uint64_t hdr_map_ver(const struct hdr* m) {
  return __atomic_load_n((const uint64_t*)((const uint8_t*)m + 152), __ATOMIC_ACQUIRE);
}

struct MeshRecordWire {
  uint32_t node, reached;
  uint64_t boot, sequence, region, memory;
  char model[MESH_MODEL_BYTES];
  struct mesh_reach reach[32];
  struct mesh_health health;
};
struct MeshMapCopyWire {
  uint64_t nodes;
  MeshRecordWire record[256];
};

// Zero-NCCL TCP bootstrap (rank 0 star topology, one-time index exchange at init/fused_init).
int send_all(int fd, const void* buf, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  while (n) {
    ssize_t k = send(fd, p, n, 0);
    if (k <= 0) return -1;
    p += k;
    n -= static_cast<size_t>(k);
  }
  return 0;
}

int recv_all(int fd, void* buf, size_t n) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  while (n) {
    ssize_t k = recv(fd, p, n, 0);
    if (k <= 0) return -1;
    p += k;
    n -= static_cast<size_t>(k);
  }
  return 0;
}

struct AllSumSlot {
  uint32_t src_base_page = MESH_PAGE_NONE;
  uint32_t max_pages = 0;
  std::vector<uint32_t> my_land;            // [max_pages]
  std::vector<uint32_t> all_land[kMaxRanks]; // [compute_rank][max_pages]
};

struct FusedPlan {
  size_t n = 0;
  size_t esize = 2;
  uint32_t R = 0;
  uint32_t nranks = 0;
  uint32_t rank = 0; // compute rank index
  WireMode wire = WIRE_BF16;
  uint8_t red_op = MESH_RED_SUM_BF16;
  uint32_t pages_per_frame = 0;
  uint32_t last_page_bytes = 0;
  uint32_t total_src_pages = 0;
  uint32_t src_base_page = MESH_PAGE_NONE;

  // Landing indices per compute rank:
  // Reduce mode (BF16/F32/HAD8): [kGenSlots][R-1][pages_per_frame]
  // Gather mode:                 [kGenSlots][R-1][nranks][pages_per_frame]
  uint32_t sum_idx_per_rank = 0;
  std::vector<uint32_t> sum_land[kMaxRanks];
  // Argmax landing indices (frame R-1): [kGenSlots][nranks] (1 page per sender)
  std::vector<uint32_t> argmax_land[kMaxRanks];

  // Shared GPU buffers for this plan's landing-index tables and arrival counters.
  MTL::Buffer* sum_land_buf = nullptr;
  MTL::Buffer* argmax_land_buf = nullptr;
  MTL::Buffer* counter_buf = nullptr; // [0..4095]: pub_done (64 slots x 64B), [4096..8191]: ret_done

  uint64_t next_gen = 0;
  uint64_t active_gen = 0;
  uint32_t active_db_slot[256] = {};
  uint64_t active_db_seq[256] = {};

  void release_resources(struct mesh_ctx* ctx) {
    if (sum_land_buf) { sum_land_buf->release(); sum_land_buf = nullptr; }
    if (argmax_land_buf) { argmax_land_buf->release(); argmax_land_buf = nullptr; }
    if (counter_buf) { counter_buf->release(); counter_buf = nullptr; }
    if (ctx && ctx->M) {
      if (src_base_page != MESH_PAGE_NONE) {
        for (uint32_t k = 0; k < total_src_pages; k++) mesh_release(ctx, src_base_page + k);
        src_base_page = MESH_PAGE_NONE;
      }
      if (!sum_land[rank].empty()) {
        mesh_clear_indices(ctx, sum_land[rank].data(), static_cast<uint32_t>(sum_land[rank].size()));
        mesh_free_indices(ctx, sum_land[rank].data(), static_cast<uint32_t>(sum_land[rank].size()));
      }
      if (!argmax_land[rank].empty()) {
        mesh_clear_indices(ctx, argmax_land[rank].data(), static_cast<uint32_t>(argmax_land[rank].size()));
        mesh_free_indices(ctx, argmax_land[rank].data(), static_cast<uint32_t>(argmax_land[rank].size()));
      }
    }
    for (uint32_t r = 0; r < kMaxRanks; r++) {
      sum_land[r].clear();
      argmax_land[r].clear();
    }
    total_src_pages = 0;
  }
};

struct State {
  struct mesh_ctx ctx{};
  bool attached = false;
  int listen_fd = -1;
  int peer_fd[kMaxRanks] = {-1, -1, -1, -1, -1, -1, -1, -1};
  uint32_t my_node = 0;
  uint32_t world_nodes[kMaxRanks] = {};
  uint32_t compute_nodes[kMaxRanks] = {};
  uint32_t world_rank = 0;
  uint32_t ranks = 0;  // WORLD_SIZE
  uint32_t nranks = 0; // compute ranks (TP)
  int32_t index = -1;  // compute rank index, -1 if relay
  uint32_t before = 0;

  MTL::Buffer* table_buf = nullptr;
  size_t table_bias = 0;
  MTL::Buffer* arena_buf = nullptr;
  size_t arena_bias = 0;

  MTL::Buffer* db_buf = nullptr;
  Doorbell* db = nullptr;
  uint64_t db_encode_seq = 0;
  std::atomic<uint64_t> pub_head{0};
  std::atomic<uint64_t> ret_head{0};
  std::atomic<uint64_t> active_until_ns{0};
  std::atomic<bool> stop_worker{false};
  std::thread worker;

  // AllSum ring (prefill + meshmlx_bench)
  size_t slot_bytes = 0;
  uint32_t allsum_slots = 0;
  uint64_t allsum_next = 0;
  std::vector<AllSumSlot> ring;
  MTL::Buffer* allsum_land_buf = nullptr; // [allsum_slots * max_pages]
  MTL::Buffer* allsum_counter_buf = nullptr;
  uint32_t allsum_max_pages = 0;

  std::vector<FusedPlan> plans;
  WireMode default_wire = WIRE_BF16;
  bool noawait = false;
  uint64_t encode_ns = 0;
  uint64_t calls = 0;
} g;

inline uint64_t now_ns() { return clock_gettime_nsec_np(CLOCK_MONOTONIC); }

WireMode parse_wire_mode() {
  const char* e = getenv("MESHMLX_WIRE");
  if (!e || !*e || !strcmp(e, "bf16")) return WIRE_BF16;
  if (!strcmp(e, "f32")) return WIRE_F32;
  if (!strcmp(e, "gather")) return WIRE_GATHER;
  if (!strcmp(e, "had8")) return WIRE_HAD8;
  throw std::runtime_error(std::string("meshmlx: unknown MESHMLX_WIRE='") + e + "' (use bf16, f32, gather, had8)");
}

void bootstrap_allgather(const void* sendbuf, void* recvbuf, size_t nbytes) {
  if (g.ranks <= 1) {
    memcpy(recvbuf, sendbuf, nbytes);
    return;
  }
  uint8_t* rb = static_cast<uint8_t*>(recvbuf);
  if (g.world_rank == 0) {
    memcpy(rb, sendbuf, nbytes);
    for (uint32_t r = 1; r < g.ranks; r++) {
      if (recv_all(g.peer_fd[r], rb + r * nbytes, nbytes) != 0)
        throw std::runtime_error("meshmlx: bootstrap recv failed");
    }
    for (uint32_t r = 1; r < g.ranks; r++) {
      if (send_all(g.peer_fd[r], rb, g.ranks * nbytes) != 0)
        throw std::runtime_error("meshmlx: bootstrap broadcast failed");
    }
  } else {
    if (send_all(g.peer_fd[0], sendbuf, nbytes) != 0)
      throw std::runtime_error("meshmlx: bootstrap send failed");
    if (recv_all(g.peer_fd[0], rb, g.ranks * nbytes) != 0)
      throw std::runtime_error("meshmlx: bootstrap recv broadcast failed");
  }
}

void bootstrap_barrier() {
  uint32_t v = 1, out[kMaxRanks] = {};
  bootstrap_allgather(&v, out, sizeof(v));
}

void bootstrap_connect(int nranks, int rank) {
  if (nranks <= 1) return;
  std::string host = "127.0.0.1";
  int port = 29500;
  if (const char* cid = getenv("NCCL_COMM_ID")) {
    std::string s(cid);
    auto col = s.rfind(':');
    if (col != std::string::npos) {
      host = s.substr(0, col);
      port = std::stoi(s.substr(col + 1));
    }
  } else if (const char* ma = getenv("MASTER_ADDR")) {
    host = ma;
    if (const char* mp = getenv("MASTER_PORT")) port = std::stoi(mp);
  }
  if (rank == 0) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("meshmlx: socket failed");
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(fd, 32) != 0) {
      close(fd);
      throw std::runtime_error("meshmlx: bootstrap bind/listen failed on port " + std::to_string(port));
    }
    g.listen_fd = fd;
    for (int k = 1; k < nranks; k++) {
      int cfd = accept(fd, nullptr, nullptr);
      if (cfd < 0) throw std::runtime_error("meshmlx: bootstrap accept failed");
      int one = 1;
      setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      uint32_t peer_r = 0;
      if (recv_all(cfd, &peer_r, sizeof(peer_r)) != 0 || peer_r == 0 || peer_r >= static_cast<uint32_t>(nranks)) {
        close(cfd);
        throw std::runtime_error("meshmlx: invalid peer rank in bootstrap");
      }
      g.peer_fd[peer_r] = cfd;
    }
  } else {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    const std::string pstr = std::to_string(port);
    if (getaddrinfo(host.c_str(), pstr.c_str(), &hints, &res) != 0 || !res)
      throw std::runtime_error("meshmlx: getaddrinfo failed for " + host + ":" + pstr);
    int fd = -1;
    for (int attempt = 0; attempt < 200; attempt++) {
      fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
      if (fd >= 0 && connect(fd, res->ai_addr, res->ai_addrlen) == 0) break;
      if (fd >= 0) { close(fd); fd = -1; }
      usleep(25000);
    }
    freeaddrinfo(res);
    if (fd < 0) throw std::runtime_error("meshmlx: bootstrap connect failed to " + host + ":" + pstr);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    uint32_t r = static_cast<uint32_t>(rank);
    if (send_all(fd, &r, sizeof(r)) != 0) throw std::runtime_error("meshmlx: bootstrap rank send failed");
    g.peer_fd[0] = fd;
  }
}

void wake_worker() {
  g.active_until_ns.store(now_ns() + 25000000ull, std::memory_order_release);
}

void submit_doorbell(const Doorbell& d) {
  if (g.nranks <= 1 || g.index < 0) return;
  const uint32_t me = static_cast<uint32_t>(g.index);
  if (d.kind == DB_FUSED_REDUCE) {
    const FusedPlan& f = g.plans[d.pid];
    const uint32_t ppf = d.n_pages;
    const uint32_t src0 = f.src_base_page + (d.g_slot * f.R + d.j) * ppf;
    const size_t idx_off = (static_cast<size_t>(d.g_slot) * (f.R - 1) + d.j) * ppf;
    std::vector<struct mesh_red_xfer> xfers;
    xfers.reserve(static_cast<size_t>(f.nranks - 1) * ppf);
    for (uint32_t r = 0; r < f.nranks; r++) {
      if (r == me) continue;
      const uint32_t dst_node = g.compute_nodes[r];
      const uint32_t* dland = f.sum_land[r].data() + idx_off;
      for (uint32_t p = 0; p < ppf; p++) {
        xfers.push_back(mesh_red_xfer{
            .src_page = src0 + p,
            .bytes = (p + 1 == ppf) ? d.last_bytes : MESH_PAGE_BYTES,
            .dst_node = dst_node,
            .dst_index = dland[p],
            .op = static_cast<uint8_t>(d.extra),
            .pad = 0,
            .tag = MESH_TAG_WEIGHT(1),
        });
      }
    }
    if (!xfers.empty()) mesh_reduce_write(&g.ctx, xfers.data(), static_cast<uint32_t>(xfers.size()));
  } else if (d.kind == DB_FUSED_GATHER) {
    const FusedPlan& f = g.plans[d.pid];
    const uint32_t ppf = d.n_pages;
    const uint32_t src0 = f.src_base_page + (d.g_slot * f.R + d.j) * ppf;
    const size_t frame_off = (static_cast<size_t>(d.g_slot) * (f.R - 1) + d.j) * f.nranks * ppf + static_cast<size_t>(me) * ppf;
    std::vector<struct mesh_xfer> xfers;
    xfers.reserve(static_cast<size_t>(f.nranks - 1) * ppf);
    for (uint32_t r = 0; r < f.nranks; r++) {
      if (r == me) continue;
      const uint32_t dst_node = g.compute_nodes[r];
      const uint32_t* dland = f.sum_land[r].data() + frame_off;
      for (uint32_t p = 0; p < ppf; p++) {
        xfers.push_back(mesh_xfer{
            .src_page = src0 + p,
            .bytes = (p + 1 == ppf) ? d.last_bytes : MESH_PAGE_BYTES,
            .dst_node = dst_node,
            .dst_index = dland[p],
        });
      }
    }
    if (!xfers.empty()) mesh_write(&g.ctx, xfers.data(), static_cast<uint32_t>(xfers.size()));
  } else if (d.kind == DB_FUSED_ARGMAX) {
    const FusedPlan& f = g.plans[d.pid];
    const uint32_t src0 = f.src_base_page + (d.g_slot * f.R + d.j) * f.pages_per_frame;
    struct mesh_xfer xfers[kMaxRanks];
    uint32_t nx = 0;
    for (uint32_t r = 0; r < f.nranks; r++) {
      if (r == me) continue;
      xfers[nx++] = mesh_xfer{
          .src_page = src0,
          .bytes = d.last_bytes,
          .dst_node = g.compute_nodes[r],
          .dst_index = f.argmax_land[r][d.g_slot * f.nranks + me],
      };
    }
    if (nx) mesh_write(&g.ctx, xfers, nx);
  } else if (d.kind == DB_ALLSUM_F32) {
    const AllSumSlot& sl = g.ring[d.extra];
    const uint32_t np = d.n_pages;
    std::vector<struct mesh_red_xfer> xfers;
    xfers.reserve(static_cast<size_t>(g.nranks - 1) * np);
    for (uint32_t r = 0; r < g.nranks; r++) {
      if (r == me) continue;
      const uint32_t dst_node = g.compute_nodes[r];
      const uint32_t* dland = sl.all_land[r].data();
      for (uint32_t p = 0; p < np; p++) {
        xfers.push_back(mesh_red_xfer{
            .src_page = sl.src_base_page + p,
            .bytes = (p + 1 == np) ? d.last_bytes : MESH_PAGE_BYTES,
            .dst_node = dst_node,
            .dst_index = dland[p],
            .op = MESH_RED_SUM_F32,
            .pad = 0,
            .tag = MESH_TAG_WEIGHT(1),
        });
      }
    }
    if (!xfers.empty()) mesh_reduce_write(&g.ctx, xfers.data(), static_cast<uint32_t>(xfers.size()));
  }
}

void retire_doorbell(const Doorbell& d) {
  if (g.nranks <= 1 || g.index < 0) return;
  const uint32_t me = static_cast<uint32_t>(g.index);
  if (d.kind == DB_FUSED_REDUCE) {
    const FusedPlan& f = g.plans[d.pid];
    const size_t off = (static_cast<size_t>(d.g_slot) * (f.R - 1) + d.j) * d.n_pages;
    mesh_clear_indices(&g.ctx, f.sum_land[me].data() + off, d.n_pages);
  } else if (d.kind == DB_FUSED_GATHER) {
    const FusedPlan& f = g.plans[d.pid];
    const size_t off = (static_cast<size_t>(d.g_slot) * (f.R - 1) + d.j) * f.nranks * d.n_pages;
    mesh_clear_indices(&g.ctx, f.sum_land[me].data() + off, f.nranks * d.n_pages);
  } else if (d.kind == DB_FUSED_ARGMAX) {
    const FusedPlan& f = g.plans[d.pid];
    mesh_clear_indices(&g.ctx, f.argmax_land[me].data() + d.g_slot * f.nranks, f.nranks);
  } else if (d.kind == DB_ALLSUM_F32) {
    const AllSumSlot& sl = g.ring[d.extra];
    mesh_clear_indices(&g.ctx, sl.my_land.data(), d.n_pages);
  }
}

void worker_loop() {
  uint64_t phead = 0, rhead = 0;
  while (!g.stop_worker.load(std::memory_order_acquire)) {
    bool did = false;
    for (;;) {
      Doorbell& pd = g.db[phead & kDbRingMask];
      if (pd.pub_seq.load(std::memory_order_acquire) == phead + 1) {
        submit_doorbell(pd);
        phead++;
        g.pub_head.store(phead, std::memory_order_release);
        did = true;
      } else {
        break;
      }
    }
    for (;;) {
      Doorbell& rd = g.db[rhead & kDbRingMask];
      if (rd.ret_seq.load(std::memory_order_acquire) == rhead + 1) {
        retire_doorbell(rd);
        rhead++;
        g.ret_head.store(rhead, std::memory_order_release);
        did = true;
      } else {
        break;
      }
    }
    if (did) {
      wake_worker();
    } else if (now_ns() < g.active_until_ns.load(std::memory_order_relaxed)) {
      asm volatile("yield");
    } else {
      usleep(25);
    }
  }
}

uint32_t alloc_doorbell(uint32_t kind, uint32_t pid, uint32_t g_slot, uint32_t j, uint32_t n_pages, uint32_t last_bytes,
                        uint32_t extra, uint64_t& out_seq) {
  const uint64_t seq = ++g.db_encode_seq;
  // Wait if ring is ever full (2048 slots >> ~200 in flight)
  while (seq - g.ret_head.load(std::memory_order_acquire) >= kDbRingSize - 4) usleep(10);
  const uint32_t slot = static_cast<uint32_t>((seq - 1) & kDbRingMask);
  Doorbell& d = g.db[slot];
  d.pub_seq.store(0, std::memory_order_relaxed);
  d.ret_seq.store(0, std::memory_order_relaxed);
  d.kind = kind;
  d.pid = pid;
  d.g_slot = g_slot;
  d.j = j;
  d.n_pages = n_pages;
  d.last_bytes = last_bytes;
  d.extra = extra;
  std::atomic_thread_fence(std::memory_order_release);
  out_seq = seq;
  wake_worker();
  return slot;
}

std::string tname(mx::Dtype t) {
  if (t == mx::bfloat16) return "bfloat";
  if (t == mx::float16) return "half";
  if (t == mx::float32) return "float";
  throw std::invalid_argument("meshmlx: expected bfloat16, float16 or float32");
}

// Common MSL prelude for system-coherent /mesh0 access, doorbells, W[] polling, and GF(2)^6 FWHT-64.
const char* kMeshMslPrelude = R"MSL(
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
)MSL";

std::string convert_source() {
  return std::string(kMeshMslPrelude) + R"MSL(
template <typename T>
[[kernel]] void to_f32_pub(device const T* x [[buffer(0)]], device uchar* src [[buffer(1)]],
    device ulong* db [[buffer(2)]], device atomic_uint* done [[buffer(3)]],
    constant uint4& info [[buffer(4)]], // x=n, y=db_slot, z=seq_lo, w=seq_hi
    uint t [[thread_position_in_grid]], uint i [[thread_index_in_threadgroup]], uint groups [[threadgroups_per_grid]]) {
  const uint n = info.x;
  if (t < n) mesh_sys_write<float>(src, t, float(x[t]));
  mesh_sys_fence();
  threadgroup_barrier(mem_flags::mem_device);
  threadgroup bool last;
  if (i == 0) {
    mesh_sys_fence();
    last = atomic_fetch_add_explicit(done, 1u, memory_order_relaxed) == groups - 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!last) return;
  if (i == 0) {
    atomic_store_explicit(done, 0u, memory_order_relaxed);
    ulong seq = ulong(info.z) | (ulong(info.w) << 32);
    mesh_ring_pub(db, info.y, seq);
  }
}

template <typename T>
[[kernel]] void from_f32_await(const device ulong* W [[buffer(0)]], const device uchar* arena [[buffer(1)]],
    const device uint* land_idx [[buffer(2)]], const device T* own [[buffer(3)]], device T* out [[buffer(4)]],
    device ulong* db [[buffer(5)]], device atomic_uint* rdone [[buffer(6)]],
    constant uint4& info [[buffer(7)]],  // x=n, y=want_tag, z=n_pages, w=land_base
    constant uint4& dbinf [[buffer(8)]], // x=db_slot, y=seq_lo, z=seq_hi, w=noawait
    uint tg [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint i [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
  const uint n = info.x, want_tag = info.y, n_pages = info.z, land_base = info.w;
  const bool do_wait = (want_tag > 0u) && (dbinf.w == 0u);
  const uint p_start = (tg * n_pages) / groups;
  const uint p_end = ((tg + 1u) * n_pages) / groups;
  for (uint pg = p_start; pg < p_end; pg++) {
    threadgroup uint phys;
    if (i == 0) phys = do_wait ? mesh_poll_land(W, land_idx[land_base + pg], want_tag) : 0xffffffffu;
    threadgroup_barrier(mem_flags::mem_device);
    mesh_sys_fence();
    const uint base = pg * 1024u;
    const uint end = min(n, base + 1024u);
    const uint p = phys;
    for (uint k = base + i; k < end; k += nt) {
      float v = float(own[k]);
      if (p != 0xffffffffu) v += mesh_sys_read<float>(arena + ulong(p) * 4096ul, k - base);
      out[k] = T(v);
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
  threadgroup bool last;
  if (i == 0) {
    mesh_sys_fence();
    last = atomic_fetch_add_explicit(rdone, 1u, memory_order_relaxed) == groups - 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last && i == 0) {
    atomic_store_explicit(rdone, 0u, memory_order_relaxed);
    ulong seq = ulong(dbinf.y) | (ulong(dbinf.z) << 32);
    mesh_ring_ret(db, dbinf.x, seq);
  }
}

template [[host_name("to_f32_pub_bfloat")]] [[kernel]] void to_f32_pub<bfloat>(device const bfloat*, device uchar*, device ulong*, device atomic_uint*, constant uint4&, uint, uint, uint);
template [[host_name("to_f32_pub_half")]] [[kernel]] void to_f32_pub<half>(device const half*, device uchar*, device ulong*, device atomic_uint*, constant uint4&, uint, uint, uint);
template [[host_name("to_f32_pub_float")]] [[kernel]] void to_f32_pub<float>(device const float*, device uchar*, device ulong*, device atomic_uint*, constant uint4&, uint, uint, uint);
template [[host_name("from_f32_await_bfloat")]] [[kernel]] void from_f32_await<bfloat>(const device ulong*, const device uchar*, const device uint*, const device bfloat*, device bfloat*, device ulong*, device atomic_uint*, constant uint4&, constant uint4&, uint, uint, uint, uint);
template [[host_name("from_f32_await_half")]] [[kernel]] void from_f32_await<half>(const device ulong*, const device uchar*, const device uint*, const device half*, device half*, device ulong*, device atomic_uint*, constant uint4&, constant uint4&, uint, uint, uint, uint);
template [[host_name("from_f32_await_float")]] [[kernel]] void from_f32_await<float>(const device ulong*, const device uchar*, const device uint*, const device float*, device float*, device ulong*, device atomic_uint*, constant uint4&, constant uint4&, uint, uint, uint, uint);
)MSL";
}

// ---- M5 Ultra tuned decode GEMV launch table (preserved verbatim from mesh-resume-mlx) ----
struct Launch { uint32_t sg, s, rows, vec, unroll; };
struct LaunchEntry { uint32_t tp; std::string kernel; uint32_t N, K; Launch l; };
const Launch kWideRows{4, 1, 1, 4, 8};
const LaunchEntry kLaunches[] = {
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

const std::vector<LaunchEntry>& launch_overrides() {
  static const std::vector<LaunchEntry> overrides = [] {
    std::vector<LaunchEntry> v;
    const char* e = getenv("MESHMLX_LAUNCH");
    for (std::string rest = e ? e : ""; !rest.empty();) {
      const size_t end = rest.find(';');
      const std::string item = rest.substr(0, end);
      rest = end == std::string::npos ? "" : rest.substr(end + 1);
      LaunchEntry x{};
      char name[64];
      if (sscanf(item.c_str(), "%63[^:]:%ux%u=%u,%u,%u,%u,%u", name, &x.N, &x.K, &x.l.sg, &x.l.s, &x.l.rows, &x.l.vec, &x.l.unroll) != 8)
        throw std::runtime_error("meshmlx: invalid MESHMLX_LAUNCH item '" + item + "'");
      x.kernel = name;
      v.push_back(x);
    }
    return v;
  }();
  return overrides;
}

Launch launch_for(const std::string& kernel, uint32_t N, uint32_t K) {
  for (const auto& x : launch_overrides()) if (x.kernel == kernel && x.N == N && x.K == K) return x.l;
  for (const auto& x : kLaunches) if (x.tp == g.nranks && x.kernel == kernel && x.N == N && x.K == K) return x.l;
  return kWideRows;
}

std::string launch_name(const std::string& kernel, uint32_t B, const Launch& l) {
  return kernel + "_" + std::to_string(B) + "_" + std::to_string(l.sg) + "_" + std::to_string(l.s) + "_" +
         std::to_string(l.rows) + "_" + std::to_string(l.vec) + "_" + std::to_string(l.unroll);
}

std::string launch_source() {
  const std::map<std::string, std::string> args = {
    {"ple_gate", "(device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*, constant uint*, uint, uint, uint)"},
    {"gate_up_geglu", "(device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*, constant uint*, uint, uint, uint)"},
    {"gemv_rows", "(device const bfloat*, device const bfloat*, device bfloat*, constant uint*, uint, uint, uint)"},
    {"gemv_publish", "(device const bfloat*, device const bfloat*, device bfloat*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint, uint, uint, uint, uint)"},
  };
  std::vector<std::pair<std::string, Launch>> all;
  for (const auto& k : args) all.push_back({k.first, kWideRows});
  for (const auto& x : kLaunches) all.push_back({x.kernel, x.l});
  for (const auto& x : launch_overrides()) all.push_back({x.kernel, x.l});
  std::set<std::string> made;
  std::string out;
  for (uint32_t B = 1; B <= 2; B++) {
    for (const auto& [kernel, l] : all) {
      const std::string name = launch_name(kernel, B, l);
      if (!args.count(kernel) || !made.insert(name).second) continue;
      out += "template [[host_name(\"" + name + "\")]] [[kernel]] void " + kernel + "<" + std::to_string(B) + ", " +
             std::to_string(l.sg) + ", " + std::to_string(l.s) + ", " + std::to_string(l.rows) + ", " +
             std::to_string(l.vec) + ", " + std::to_string(l.unroll) + ">" + args.at(kernel) + ";\n";
    }
  }
  return out;
}

std::string launch_kernel(const std::string& kernel, uint32_t B, const Launch& l, uint32_t N, uint32_t K, size_t& groups) {
  const uint32_t rows = l.sg / l.s * l.rows;
  if (N % rows || K % (l.s * l.vec)) throw std::runtime_error("meshmlx: " + kernel + "'s launch does not divide its shape");
  groups = N / rows;
  return launch_name(kernel, B, l);
}

uint32_t decode_rows(const mx::array& x) {
  const uint32_t B = static_cast<uint32_t>(x.size() / x.shape(-1));
  if (B != 1 && B != 2) throw std::runtime_error("meshmlx: decode products take 1 or 2 rows");
  return B;
}

std::string fused_source() {
  std::string s = kMeshMslPrelude;
  s += R"MSL(
template <typename T> struct four { typedef uint2 type; };
template <> struct four<float> { typedef uint4 type; };

// Copy partial into contiguous /mesh0 source pages (in wire dtype: bf16 or f32) and ring doorbell.
template <typename T>
[[kernel]] void fused_write_pub(device const T* x [[buffer(0)]], device uchar* send [[buffer(1)]],
    device ulong* db [[buffer(2)]], device atomic_uint* done [[buffer(3)]],
    constant uint* cfg [[buffer(4)]], // [0]=n, [1]=db_slot, [2]=seq_lo, [3]=seq_hi, [4]=widen_f32
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
template [[host_name("fused_write_pub_bfloat")]] [[kernel]] void fused_write_pub<bfloat>(device const bfloat*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint, uint, uint);
template [[host_name("fused_write_pub_half")]] [[kernel]] void fused_write_pub<half>(device const half*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint, uint, uint);
template [[host_name("fused_write_pub_float")]] [[kernel]] void fused_write_pub<float>(device const float*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint, uint, uint);

// GF(2)^6 FWHT-64 had_int8 pack into /mesh0 pages (60 blocks/page x 68 B = 4080 B) + doorbell.
[[kernel]] void fused_had8_pub_bfloat(device const bfloat* x [[buffer(0)]], device uchar* send [[buffer(1)]],
    device ulong* db [[buffer(2)]], device atomic_uint* done [[buffer(3)]],
    constant uint* cfg [[buffer(4)]], // [0]=n_blks, [1]=db_slot, [2]=seq_lo, [3]=seq_hi
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

// Standalone fused_all_sum consumer over /mesh0 W[] + arena.
template <typename T>
[[kernel]] void fused_sum(const device ulong* W [[buffer(0)]], const device uchar* arena [[buffer(1)]],
    const device uint* land_idx [[buffer(2)]], device const T* own [[buffer(3)]], device T* out [[buffer(4)]],
    device ulong* db [[buffer(5)]], device atomic_uint* rdone [[buffer(6)]],
    constant uint* cfg [[buffer(7)]], // [0]=n, [1]=ranks, [2]=me, [3]=wire_mode, [4]=ppf, [5]=land_base, [6]=db_slot, [7]=seq_lo, [8]=seq_hi, [9]=noawait
    uint tg [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
    uint i [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
  const uint n = cfg[0], ranks = cfg[1], me = cfg[2], wmode = cfg[3], ppf = cfg[4], lbase = cfg[5];
  const bool do_wait = (ranks > 1u) && (cfg[9] == 0u);
  const uint wire_eb = (wmode == 1u) ? 4u : sizeof(T);
  const uint elems_pg = 4096u / wire_eb;
  const uint p_start = (tg * ppf) / groups;
  const uint p_end = ((tg + 1u) * ppf) / groups;
  for (uint pg = p_start; pg < p_end; pg++) {
    threadgroup uint phys[8];
    if (wmode == 2u) {
      if (i < ranks) {
        phys[i] = (i == me || !do_wait) ? 0xffffffffu : mesh_poll_land(W, land_idx[lbase + i * ppf + pg], 0u);
      }
    } else {
      if (i == 0) phys[0] = do_wait ? mesh_poll_land(W, land_idx[lbase + pg], ranks - 1u) : 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_device);
    mesh_sys_fence();
    const uint base = pg * elems_pg;
    const uint end = min(n, base + elems_pg);
    for (uint k = base + i; k < end; k += nt) {
      float acc = 0.0f;
      if (wmode == 2u) {
        for (uint q = 0; q < ranks; q++) {
          if (q == me) acc += float(own[k]);
          else if (phys[q] != 0xffffffffu) acc += float(mesh_sys_read<T>(arena + ulong(phys[q]) * 4096ul, k - base));
        }
      } else {
        acc = float(own[k]);
        if (phys[0] != 0xffffffffu) {
          if (wmode == 1u) acc += mesh_sys_read<float>(arena + ulong(phys[0]) * 4096ul, k - base);
          else acc += float(mesh_sys_read<T>(arena + ulong(phys[0]) * 4096ul, k - base));
        }
      }
      out[k] = T(acc);
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
  threadgroup bool last;
  if (i == 0) {
    mesh_sys_fence();
    last = atomic_fetch_add_explicit(rdone, 1u, memory_order_relaxed) == groups - 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (last && i == 0 && ranks > 1u) {
    atomic_store_explicit(rdone, 0u, memory_order_relaxed);
    ulong seq = ulong(cfg[7]) | (ulong(cfg[8]) << 32);
    mesh_ring_ret(db, cfg[6], seq);
  }
}
template [[host_name("fused_sum_bfloat")]] [[kernel]] void fused_sum<bfloat>(const device ulong*, const device uchar*, const device uint*, device const bfloat*, device bfloat*, device ulong*, device atomic_uint*, constant uint*, uint, uint, uint, uint);
template [[host_name("fused_sum_half")]] [[kernel]] void fused_sum<half>(const device ulong*, const device uchar*, const device uint*, device const half*, device half*, device ulong*, device atomic_uint*, constant uint*, uint, uint, uint, uint);
template [[host_name("fused_sum_float")]] [[kernel]] void fused_sum<float>(const device ulong*, const device uchar*, const device uint*, device const float*, device float*, device ulong*, device atomic_uint*, constant uint*, uint, uint, uint, uint);

// Fused crossing await + sum + RMSNorm_w1 + residual [+ RMSNorm_w2] over /mesh0 W[] + arena.
// Supports reduce (bf16/f32), gather (rank-ordered fp32), and local add_norm (ranks == 1).
template <typename T>
[[kernel]] void fused_sum_norm(const device ulong* W [[buffer(0)]], const device uchar* arena [[buffer(1)]],
    const device uint* land_idx [[buffer(2)]], device const T* own [[buffer(3)]],
    device const T* residual [[buffer(4)]], device const T* w1 [[buffer(5)]], device const T* w2 [[buffer(6)]],
    device T* h [[buffer(7)]], device T* normed [[buffer(8)]], device const T* scalar [[buffer(9)]],
    device ulong* db [[buffer(10)]], device atomic_uint* rdone [[buffer(11)]],
    constant uint* cfg [[buffer(12)]], // [0]=n,[1]=ranks,[2]=me,[3]=dim,[4]=twice,[5]=scaled,[6]=wmode,[7]=ppf,[8]=lbase,[9]=db_slot,[10]=seq_lo,[11]=seq_hi,[12]=noawait
    constant float& eps [[buffer(13)]],
    uint i [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]], uint rows [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint ranks = cfg[1], me = cfg[2], dim = cfg[3], twice = cfg[4], scaled = cfg[5];
  const uint wmode = cfg[6], ppf = cfg[7], lbase = cfg[8];
  const bool do_wait = (ranks > 1u) && (cfg[12] == 0u);
  const T by = scaled ? scalar[0] : T(1.0f);
  constexpr uint READS = 4;
  const ulong base = ulong(row) * dim;
  const uint k0 = i * READS;

  // Preload local operands under the network wait.
  float mine[READS], res[READS], g1[READS], g2[READS];
  for (uint e = 0; e < READS; e++) {
    const bool in = k0 + e < dim;
    mine[e] = in ? float(own[base + k0 + e]) : 0.0f;
    res[e]  = in ? float(residual[base + k0 + e]) : 0.0f;
    g1[e]   = in ? float(w1[k0 + e]) : 0.0f;
    g2[e]   = in && twice ? float(w2[k0 + e]) : 0.0f;
  }

  const uint wire_eb = (wmode == 1u) ? 4u : sizeof(T);
  const uint pg0 = uint((base * wire_eb) >> 12);
  const uint pg1 = uint(((base + dim) * wire_eb - 1ul) >> 12);
  const uint npg = pg1 - pg0 + 1u; // <= 2 for bf16 dim=1536, <= 3 for f32

  threadgroup uint tg_phys[32]; // reduce: tg_phys[0..npg-1]; gather: tg_phys[q*4 + p_off]
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

  threadgroup float sums[32];
  threadgroup float inv[1];
  float v[READS];
  float acc = 0.0f;
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
        const T nv = T(float(T(g1[e])) * float(T(v[e] * scale)));
        T x = T(res[e] + float(nv));
        if (scaled) x = T(float(x) * float(by));
        h[base + k] = x;
        v[e] = float(x);
        acc += v[e] * v[e];
      } else {
        normed[base + k] = T(float(T(g2[e])) * float(T(v[e] * scale)));
      }
    }
  }

  if (ranks > 1u) {
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
template [[host_name("fused_sum_norm_bfloat")]] [[kernel]] void fused_sum_norm<bfloat>(const device ulong*, const device uchar*, const device uint*, device const bfloat*, device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*, device bfloat*, device const bfloat*, device ulong*, device atomic_uint*, constant uint*, constant float&, uint, uint, uint, uint, uint);
template [[host_name("fused_sum_norm_half")]] [[kernel]] void fused_sum_norm<half>(const device ulong*, const device uchar*, const device uint*, device const half*, device const half*, device const half*, device const half*, device half*, device half*, device const half*, device ulong*, device atomic_uint*, constant uint*, constant float&, uint, uint, uint, uint, uint);
template [[host_name("fused_sum_norm_float")]] [[kernel]] void fused_sum_norm<float>(const device ulong*, const device uchar*, const device uint*, device const float*, device const float*, device const float*, device const float*, device float*, device float*, device const float*, device ulong*, device atomic_uint*, constant uint*, constant float&, uint, uint, uint, uint, uint);

// GF(2)^6 FWHT-64 had_int8 consumer + RMSNorm + residual [+ RMSNorm]:
// 1 threadgroup per row, 24 simdgroups (768 threads) for dim=1536 (1 simdgroup = 64 elements = 1 FWHT-64 block!).
[[kernel]] void fused_sum_norm_had8_bfloat(const device ulong* W [[buffer(0)]], const device uchar* arena [[buffer(1)]],
    const device uint* land_idx [[buffer(2)]], device const bfloat* own [[buffer(3)]],
    device const bfloat* residual [[buffer(4)]], device const bfloat* w1 [[buffer(5)]], device const bfloat* w2 [[buffer(6)]],
    device bfloat* h [[buffer(7)]], device bfloat* normed [[buffer(8)]],
    device ulong* db [[buffer(10)]], device atomic_uint* rdone [[buffer(11)]],
    constant uint* cfg [[buffer(12)]], constant float& eps [[buffer(13)]],
    uint i [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]], uint rows [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint ranks = cfg[1], dim = cfg[3], twice = cfg[4], lbase = cfg[8];
  const bool do_wait = (ranks > 1u) && (cfg[12] == 0u);
  const uint bpr = dim / 64u;
  const uint blk0 = row * bpr;
  const uint pg0 = blk0 / 60u;
  const uint pg1 = (blk0 + bpr - 1u) / 60u;
  const uint npg = pg1 - pg0 + 1u;

  const ulong base = ulong(row) * dim;
  const uint k0 = sg * 64u + lane * 2u;
  float mine0 = (k0 < dim) ? float(own[base + k0]) : 0.0f;
  float mine1 = (k0 + 1u < dim) ? float(own[base + k0 + 1u]) : 0.0f;
  float res0  = (k0 < dim) ? float(residual[base + k0]) : 0.0f;
  float res1  = (k0 + 1u < dim) ? float(residual[base + k0 + 1u]) : 0.0f;
  float g10   = (k0 < dim) ? float(w1[k0]) : 0.0f;
  float g11   = (k0 + 1u < dim) ? float(w1[k0 + 1u]) : 0.0f;
  float g20   = (k0 < dim && twice) ? float(w2[k0]) : 0.0f;
  float g21   = (k0 + 1u < dim && twice) ? float(w2[k0 + 1u]) : 0.0f;

  threadgroup uint tg_phys[4];
  if (do_wait) {
    if (i < npg) tg_phys[i] = mesh_poll_land(W, land_idx[lbase + pg0 + i], ranks - 1u);
    threadgroup_barrier(mem_flags::mem_device);
    mesh_sys_fence();
  }

  float a = 0.0f, b = 0.0f;
  if (do_wait && sg < bpr) {
    const uint blk = blk0 + sg;
    const uint pg = blk / 60u;
    const uint phys = tg_phys[pg - pg0];
    if (phys != 0xffffffffu) {
      const ulong off = ulong(phys) * 4096ul + ulong(blk % 60u) * 68ul;
      const float sc = float(mesh_sys_read<half>(arena + off, lane >= 16u ? 1u : 0u));
      a = float(mesh_sys_read<char>(arena + off + 4ul, lane * 2u)) * sc;
      b = float(mesh_sys_read<char>(arena + off + 4ul, lane * 2u + 1u)) * sc;
      fwht64_simd_pair(a, b, lane);
    }
  }

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
        bfloat x0 = bfloat(res0 + float(bfloat(float(bfloat(g10)) * float(bfloat(v0 * scale)))));
        bfloat x1 = bfloat(res1 + float(bfloat(float(bfloat(g11)) * float(bfloat(v1 * scale)))));
        h[base + k0] = x0;
        h[base + k0 + 1u] = x1;
        v0 = float(x0); v1 = float(x1);
        acc = v0 * v0 + v1 * v1;
      } else {
        normed[base + k0] = bfloat(float(bfloat(g20)) * float(bfloat(v0 * scale)));
        normed[base + k0 + 1u] = bfloat(float(bfloat(g21)) * float(bfloat(v1 * scale)));
      }
    }
  }

  if (ranks > 1u) {
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

// Argmax write + doorbell (1 page for all B <= 512 rows, 8 B/row: float val + uint idx).
[[kernel]] void argmax_write_pub(device const float* val [[buffer(0)]], device const uint* idx [[buffer(1)]],
    device uchar* send [[buffer(2)]], device ulong* db [[buffer(3)]],
    constant uint* g [[buffer(4)]], // [0]=rows, [1]=db_slot, [2]=seq_lo, [3]=seq_hi
    uint b [[thread_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  if (b < g[0]) {
    mesh_sys_write<uint>(send, ulong(b) * 2ul, as_type<uint>(val[b]));
    mesh_sys_write<uint>(send, ulong(b) * 2ul + 1ul, idx[b]);
  }
  mesh_sys_fence();
  threadgroup_barrier(mem_flags::mem_device);
  if (i == 0) {
    ulong seq = ulong(g[2]) | (ulong(g[3]) << 32);
    mesh_ring_pub(db, g[1], seq);
  }
}

[[kernel]] void argmax_pick(const device ulong* W [[buffer(0)]], const device uchar* arena [[buffer(1)]],
    const device uint* arg_land [[buffer(2)]], device const float* own_val [[buffer(3)]],
    device const uint* own_idx [[buffer(4)]], device int* out [[buffer(5)]],
    device ulong* db [[buffer(6)]],
    constant uint* g [[buffer(7)]], // [0]=rows, [1]=ranks, [2]=me, [3]=lbase, [4]=db_slot, [5]=seq_lo, [6]=seq_hi, [7]=noawait
    uint i [[thread_index_in_threadgroup]], uint b [[thread_position_in_grid]]) {
  const uint rows = g[0], ranks = g[1], me = g[2], lbase = g[3];
  const bool do_wait = (ranks > 1u) && (g[7] == 0u);
  threadgroup uint phys[8];
  if (i < ranks) {
    phys[i] = (i == me || !do_wait) ? 0xffffffffu : mesh_poll_land(W, arg_land[lbase + i], 0u);
  }
  threadgroup_barrier(mem_flags::mem_device);
  mesh_sys_fence();
  if (b < rows) {
    float best = own_val[b];
    uint at = own_idx[b];
    for (uint q = 0; q < ranks; q++) {
      if (q == me || phys[q] == 0xffffffffu) continue;
      const device uchar* p = arena + ulong(phys[q]) * 4096ul;
      const float v = as_type<float>(mesh_sys_read<uint>(p, ulong(b) * 2ul));
      const uint k = mesh_sys_read<uint>(p, ulong(b) * 2ul + 1ul);
      if (v > best || (v == best && k < at)) { best = v; at = k; }
    }
    out[b] = int(at);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (i == 0 && ranks > 1u) {
    ulong seq = ulong(g[5]) | (ulong(g[6]) << 32);
    mesh_ring_ret(db, g[4], seq);
  }
}

// ---- Verbatim M5-Ultra-tuned pure compute kernels from mesh-resume-mlx ----
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
[[kernel]] void row_argmax_parts(device const bfloat* lg [[buffer(0)]], device uint2* part [[buffer(1)]], constant uint* g [[buffer(2)]],
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
[[kernel]] void row_argmax_pick(device const uint2* part [[buffer(0)]], device float* val [[buffer(1)]], device uint* idx [[buffer(2)]],
    constant uint* g [[buffer(3)]], uint i [[thread_index_in_threadgroup]], uint b [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
  const uint P = g[0], offset = g[1];
  float v = -INFINITY; uint at = 0xffffffffu;
  for (uint p = i; p < P; p += nt) { const uint2 q = part[ulong(b) * P + p]; best_of(v, at, as_type<float>(q.x), q.y); }
  threadgroup float tv[32]; threadgroup uint ti[32];
  best_reduce(v, at, lane, sg, (nt + 31) / 32, tv, ti);
  if (i == 0) { val[b] = v; idx[b] = at + offset; }
}

template <typename T>
[[kernel]] void head_norm_rope(device const T* x [[buffer(0)]], device const T* w0 [[buffer(1)]], device const T* w1 [[buffer(2)]],
    device const T* w2 [[buffer(3)]], device T* y0 [[buffer(4)]], device T* y1 [[buffer(5)]], device T* y2 [[buffer(6)]],
    device const int* offset [[buffer(7)]], device const float* freqs [[buffer(8)]], constant uint* p [[buffer(9)]],
    constant float* f [[buffer(10)]],
    uint i [[thread_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint h0 = p[0], h1 = p[1], h2 = p[2], D = p[3], rope = p[4], has_w2 = p[5], S = p[6];
  const float eps = f[0], base_log2 = f[1];
  const uint heads = h0 + h1 + h2, b = tg / heads, g = tg % heads;
  const uint seg = g < h0 ? 0 : g < h0 + h1 ? 1 : 2, h = seg == 0 ? g : seg == 1 ? g - h0 : g - h0 - h1;
  const uint hs = seg == 0 ? h0 : seg == 1 ? h1 : h2;
  device const T* xr = x + ulong(b) * S + ulong(g) * D;
  device const T* w = seg == 0 ? w0 : seg == 1 ? w1 : w2;
  device T* y = (seg == 0 ? y0 : seg == 1 ? y1 : y2) + (ulong(b) * hs + h) * D;
  const bool has_w = seg < 2 || has_w2, rotate = rope && seg < 2;
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
  T nv[READS];
  for (uint e = 0; e < READS; e++) nv[e] = has_w ? T(float(w[k0 + e]) * float(T(v[e] * scale))) : T(v[e] * scale);
  if (!rotate) { for (uint e = 0; e < READS; e++) y[k0 + e] = nv[e]; return; }
  for (uint e = 0; e < READS; e++) normed[k0 + e] = float(nv[e]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint half_d = D / 2;
  const float L = 1.0f * static_cast<float>(offset[0]);
  for (uint e = 0; e < READS; e++) {
    const uint k = k0 + e, j = k < half_d ? k : k - half_d;
    const float inv_freq = rope == 1 ? metal::exp2(-(static_cast<float>(j) / static_cast<float>(half_d)) * base_log2) : 1.0 / (freqs[j]);
    const float theta = L * inv_freq, c = metal::fast::cos(theta), sn = metal::fast::sin(theta);
    const float a = normed[j], bb = normed[j + half_d];
    y[k] = T(k < half_d ? a * c - bb * sn : a * sn + bb * c);
  }
}
template [[host_name("head_norm_rope_bfloat")]] [[kernel]] void head_norm_rope<bfloat>(device const bfloat*, device const bfloat*, device const bfloat*, device const bfloat*, device bfloat*, device bfloat*, device bfloat*, device const int*, device const float*, constant uint*, constant float*, uint, uint, uint, uint);

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
inline float gelu_approx(float gv) {
  return 0.5f * gv * (1.0f + metal::precise::tanh(0.7978845608028654f * (gv + 0.044715f * gv * gv * gv)));
}
template <uint B, uint SG, uint S, uint ROWS, uint VEC, uint UNROLL>
[[kernel]] void gate_up_geglu(device const bfloat* x [[buffer(0)]], device const bfloat* Wg [[buffer(1)]], device const bfloat* Wu [[buffer(2)]],
    device bfloat* act [[buffer(3)]], constant uint* dims [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint tg [[threadgroup_position_in_grid]]) {
  const uint N = dims[0], K = dims[1];
  threadgroup float red[2][SG * ROWS * B];
  float g[ROWS][B], u[ROWS][B]; uint row0; bool owner;
  tiled_rows<B, SG, S, ROWS, VEC, UNROLL>(x, Wg, K, tg, sg, lane, red[0], g, row0, owner);
  tiled_rows<B, SG, S, ROWS, VEC, UNROLL>(x, Wu, K, tg, sg, lane, red[1], u, row0, owner);
  if (!owner) return;
  for (uint r = 0; r < ROWS; r++) for (uint b = 0; b < B; b++) {
    const float gv = float(bfloat(g[r][b])), uv = float(bfloat(u[r][b]));
    act[ulong(b) * N + row0 + r] = bfloat(float(bfloat(gelu_approx(gv))) * uv);
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
    constant uint* dims [[buffer(6)]], // [0]=N, [1]=K, [2]=db_slot, [3]=seq_lo, [4]=seq_hi, [5]=wmode (0=bf16,1=f32,2=gather,3=no_write)
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
      else if (wmode != 3u) mesh_sys_write<bfloat>(send, at, v);
    }
  }
  if (wmode == 3u) return;
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
)MSL";
  s += launch_source();
  return s;
}

std::string gemm_source() {
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

[[kernel]] void dep_copy(const device uint2* x [[buffer(0)]], const device uchar* dep [[buffer(1)]], device uint2* y [[buffer(2)]],
                         constant uint& n4 [[buffer(3)]], uint t [[thread_position_in_grid]]) {
  if (t < n4) y[t] = x[t];
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
[[kernel]] void gemm_publish_wide(const device T* mat [[buffer(0)]], const device T* in_vec [[buffer(1)]], device T* out_vec [[buffer(3)]],
    constant int* dims [[buffer(4)]], device uchar* send [[buffer(5)]], device ulong* db [[buffer(6)]],
    device atomic_uint* done [[buffer(7)]], constant uint* pub_cfg [[buffer(8)]], // [0]=db_slot,[1]=seq_lo,[2]=seq_hi,[3]=wmode
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
          else if (wmode != 3u) mesh_sys_write<T>(send, at, y);
        }
  }
  threadgroup bool last;
  if (wmode != 3u) group_arrive_db(db, done, pub_cfg, tgpg.x * tgpg.y, i, last, k_lane == 0 && out_row < out_vec_size);
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
    if (wmode != 3u) {
      UNROLL_ALL for (uint16_t e = 0; e < Dacc.get_capacity(); e++) {
        auto idx = Dacc.get_multidimensional_index(e);
        const ulong at = ulong(m_off + idx[1]) * ulong(params->N) + ulong(n_off + idx[0]);
        if (wmode == 1u) mesh_sys_write<float>(send, at, float(static_cast<T>(Dacc[e])));
        else mesh_sys_write<T>(send, at, static_cast<T>(Dacc[e]));
      }
    }
  }
  threadgroup bool last;
  if (wmode != 3u) group_arrive_db(db, done, pub_cfg, tgpg.x, i, last, true);
}

#define WIDE(NV, KL) template [[host_name("gemm_rows_wide_" #NV "_" #KL)]] [[kernel]] void gemm_rows_wide<bfloat, NV, KL>(const device bfloat*, const device bfloat*, device bfloat*, constant int*, uint3, uint3, uint, uint);
WIDE(2, 16) WIDE(3, 16) WIDE(4, 16) WIDE(5, 16) WIDE(2, 32) WIDE(3, 32) WIDE(4, 32) WIDE(5, 32)
#define NAX(SM, KS) template [[host_name("gemm_rows_nax_" #SM "_" #KS)]] [[kernel]] void gemm_rows_nax<bfloat, 2, 2, SM, 1, KS>(const device bfloat*, const device bfloat*, device bfloat*, const constant ThinParams*, uint, uint3);
NAX(1, 4) NAX(2, 3) NAX(3, 2) NAX(4, 2) NAX(5, 1) NAX(6, 1) NAX(7, 1) NAX(8, 1)
#define WIDE_P(NV, KL) template [[host_name("gemm_publish_wide_" #NV "_" #KL)]] [[kernel]] void gemm_publish_wide<bfloat, NV, KL>(const device bfloat*, const device bfloat*, device bfloat*, constant int*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint3, uint3, uint, uint, uint);
WIDE_P(2, 16) WIDE_P(3, 16) WIDE_P(4, 16) WIDE_P(5, 16) WIDE_P(2, 32) WIDE_P(3, 32) WIDE_P(4, 32) WIDE_P(5, 32)
#define NAX_P(SM, KS) template [[host_name("gemm_publish_nax_" #SM "_" #KS)]] [[kernel]] void gemm_publish_nax<bfloat, 2, 2, SM, 1, KS>(const device bfloat*, const device bfloat*, device bfloat*, const constant ThinParams*, device uchar*, device ulong*, device atomic_uint*, constant uint*, uint, uint3, uint3, uint);
NAX_P(1, 4) NAX_P(2, 3) NAX_P(3, 2) NAX_P(4, 2) NAX_P(5, 1) NAX_P(6, 1) NAX_P(7, 1) NAX_P(8, 1)
)MSL";
}

// Helper: prepare generation & doorbell for frame j of FusedPlan pid.
void prepare_fused_frame(FusedPlan& f, int pid, uint32_t j, uint32_t& out_gslot, uint32_t& out_db_slot, uint64_t& out_seq) {
  if (j == 0) f.active_gen = f.next_gen++;
  out_gslot = static_cast<uint32_t>(f.active_gen % kGenSlots);
  const uint32_t kind = (f.wire == WIRE_GATHER) ? DB_FUSED_GATHER : DB_FUSED_REDUCE;
  out_db_slot = alloc_doorbell(kind, static_cast<uint32_t>(pid), out_gslot, j, f.pages_per_frame, f.last_page_bytes,
                               static_cast<uint32_t>(f.red_op), out_seq);
  f.active_db_slot[j & 255] = out_db_slot;
  f.active_db_seq[j & 255] = out_seq;
}

void encode_fused_publish(FusedPlan& f, int pid, mx::metal::Device& d, mx::metal::CommandEncoder& enc,
                          const mx::array& x, uint32_t j, uint32_t& gslot, uint32_t& db_slot, uint64_t& seq) {
  if (x.size() != f.n) throw std::runtime_error("meshmlx: fused partial size mismatch");
  prepare_fused_frame(f, pid, j, gslot, db_slot, seq);
  auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
  const size_t src_off = g.arena_bias + static_cast<size_t>(f.src_base_page + (gslot * f.R + j) * f.pages_per_frame) * MESH_PAGE_BYTES;
  const size_t done_off = static_cast<size_t>(j & 63) * 64u;
  if (f.wire == WIRE_HAD8) {
    const uint32_t n_blks = static_cast<uint32_t>(f.n / 64);
    const uint32_t tgs = (n_blks + 3u) / 4u;
    enc.set_compute_pipeline_state(d.get_kernel("fused_had8_pub_bfloat", lib));
    enc.set_input_array(x, 0);
    enc.set_buffer(g.arena_buf, 1, src_off);
    enc.set_buffer(g.db_buf, 2, 0);
    enc.set_buffer(f.counter_buf, 3, done_off);
    const uint32_t cfg[4] = {n_blks, db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32)};
    enc.set_bytes(cfg, 4, 4);
    enc.dispatch_threadgroups(MTL::Size(tgs, 1, 1), MTL::Size(128, 1, 1));
  } else {
    enc.set_compute_pipeline_state(d.get_kernel("fused_write_pub_" + tname(x.dtype()), lib));
    enc.set_input_array(x, 0);
    enc.set_buffer(g.arena_buf, 1, src_off);
    enc.set_buffer(g.db_buf, 2, 0);
    enc.set_buffer(f.counter_buf, 3, done_off);
    const uint32_t cfg[5] = {static_cast<uint32_t>(f.n), db_slot, static_cast<uint32_t>(seq),
                             static_cast<uint32_t>(seq >> 32), (f.wire == WIRE_F32 && f.esize == 2) ? 1u : 0u};
    enc.set_bytes(cfg, 5, 4);
    const size_t n4 = f.n / 4;
    const size_t wg = std::min<size_t>(n4, 256);
    enc.dispatch_threads(MTL::Size(n4, 1, 1), MTL::Size(wg, 1, 1));
  }
}

// ---- Primitives ----

class MeshAllSum : public mx::Primitive {
 public:
  explicit MeshAllSum(mx::Stream s) : mx::Primitive(s) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    const mx::array& x = inputs[0];
    mx::array& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    const size_t n = x.size();
    const uint32_t n_pages = static_cast<uint32_t>((n * 4 + MESH_PAGE_BYTES - 1) / MESH_PAGE_BYTES);
    if (n_pages > g.allsum_max_pages) throw std::runtime_error("meshmlx: all_sum larger than slot_bytes");
    const uint32_t last_bytes = static_cast<uint32_t>((n * 4) - static_cast<size_t>(n_pages - 1) * MESH_PAGE_BYTES);
    const uint32_t sl_idx = static_cast<uint32_t>(g.allsum_next++ % g.allsum_slots);
    const AllSumSlot& sl = g.ring[sl_idx];
    uint64_t seq = 0;
    const uint32_t db_slot = alloc_doorbell(DB_ALLSUM_F32, 0, 0, 0, n_pages, last_bytes, sl_idx, seq);

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_convert", []() { return convert_source(); });
    auto& enc = mx::metal::get_command_encoder(s);
    const uint64_t t0 = now_ns();
    const size_t src_off = g.arena_bias + static_cast<size_t>(sl.src_base_page) * MESH_PAGE_BYTES;
    const size_t pdone_off = static_cast<size_t>(sl_idx & 63) * 64u;
    const size_t rdone_off = 4096u + static_cast<size_t>(sl_idx & 63) * 64u;

    // 1. Convert input -> fp32 in /mesh0 + ring doorbell
    enc.set_compute_pipeline_state(d.get_kernel("to_f32_pub_" + tname(x.dtype()), lib));
    enc.set_input_array(x, 0);
    enc.set_buffer(g.arena_buf, 1, src_off);
    enc.set_buffer(g.db_buf, 2, 0);
    enc.set_buffer(g.allsum_counter_buf, 3, pdone_off);
    const uint32_t pinfo[4] = {static_cast<uint32_t>(n), db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32)};
    enc.set_bytes(pinfo, 4, 4);
    enc.dispatch_threads(MTL::Size(n, 1, 1), MTL::Size(std::min<size_t>(n, 256), 1, 1));

    // 2. Await W[] reduced pages + add own + convert to output dtype + retire doorbell
    auto* fk = d.get_kernel("from_f32_await_" + tname(out.dtype()), lib);
    enc.set_compute_pipeline_state(fk);
    enc.set_buffer(g.table_buf, 0, g.table_bias);
    enc.set_buffer(g.arena_buf, 1, g.arena_bias);
    enc.set_buffer(g.allsum_land_buf, 2, 0);
    enc.set_input_array(x, 3);
    enc.set_output_array(out, 4);
    enc.set_buffer(g.db_buf, 5, 0);
    enc.set_buffer(g.allsum_counter_buf, 6, rdone_off);
    const uint32_t ainfo[4] = {static_cast<uint32_t>(n), g.nranks > 1 ? g.nranks - 1u : 0u, n_pages, sl_idx * g.allsum_max_pages};
    const uint32_t dbinf[4] = {db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32), g.noawait ? 1u : 0u};
    enc.set_bytes(ainfo, 4, 7);
    enc.set_bytes(dbinf, 4, 8);
    const size_t tgs = std::min<size_t>(n_pages, kSpinGroups);
    enc.dispatch_threadgroups(MTL::Size(tgs, 1, 1), MTL::Size(256, 1, 1));

    g.encode_ns += now_ns() - t0;
    g.calls++;
  }
  const char* name() const override { return "MeshAllSum"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
};

class FusedAllSum : public mx::Primitive {
 public:
  FusedAllSum(mx::Stream s, uint32_t j, int pid) : mx::Primitive(s), j_(j), pid_(pid) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    FusedPlan& f = g.plans.at(pid_);
    const mx::array& x = inputs[0];
    mx::array& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto& enc = mx::metal::get_command_encoder(s);
    uint32_t gslot = 0, db_slot = 0;
    uint64_t seq = 0;
    encode_fused_publish(f, pid_, d, enc, x, j_, gslot, db_slot, seq);
    auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
    auto* k = d.get_kernel("fused_sum_" + tname(out.dtype()), lib);
    enc.set_compute_pipeline_state(k);
    enc.set_buffer(g.table_buf, 0, g.table_bias);
    enc.set_buffer(g.arena_buf, 1, g.arena_bias);
    enc.set_buffer(f.sum_land_buf, 2, 0);
    enc.set_input_array(x, 3);
    enc.set_output_array(out, 4);
    enc.set_buffer(g.db_buf, 5, 0);
    enc.set_buffer(f.counter_buf, 6, 4096u + static_cast<size_t>(j_ & 63) * 64u);
    const uint32_t lbase = (f.wire == WIRE_GATHER)
                               ? (gslot * (f.R - 1) + j_) * f.nranks * f.pages_per_frame
                               : (gslot * (f.R - 1) + j_) * f.pages_per_frame;
    const uint32_t cfg[10] = {static_cast<uint32_t>(f.n), f.nranks, f.rank, static_cast<uint32_t>(f.wire),
                              f.pages_per_frame, lbase, db_slot, static_cast<uint32_t>(seq),
                              static_cast<uint32_t>(seq >> 32), g.noawait ? 1u : 0u};
    enc.set_bytes(cfg, 10, 7);
    const size_t tgs = std::min<size_t>(f.pages_per_frame, kSpinGroups);
    enc.dispatch_threadgroups(MTL::Size(tgs, 1, 1), MTL::Size(256, 1, 1));
  }
  const char* name() const override { return "FusedAllSum"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
 private:
  uint32_t j_;
  int pid_;
};

class FusedSumNorm : public mx::Primitive {
 public:
  FusedSumNorm(mx::Stream s, uint32_t j, int pid, float eps, bool published)
      : mx::Primitive(s), j_(j), pid_(pid), eps_(eps), published_(published) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    FusedPlan& f = g.plans.at(pid_);
    const mx::array &y = inputs[0], &res = inputs[1], &w1 = inputs[2];
    const bool twice = inputs.size() > 3;
    for (auto& o : outputs) o.set_data(mx::allocator::malloc(o.nbytes()));
    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
    auto& enc = mx::metal::get_command_encoder(s);

    uint32_t gslot = 0, db_slot = 0;
    uint64_t seq = 0;
    if (!published_ || f.wire == WIRE_HAD8) {
      encode_fused_publish(f, pid_, d, enc, y, j_, gslot, db_slot, seq);
    } else {
      gslot = static_cast<uint32_t>(f.active_gen % kGenSlots);
      db_slot = f.active_db_slot[j_ & 255];
      seq = f.active_db_seq[j_ & 255];
    }

    const uint32_t dim = static_cast<uint32_t>(y.shape(-1));
    const uint32_t rows = static_cast<uint32_t>(y.size() / dim);
    const uint32_t lbase = (f.wire == WIRE_GATHER)
                               ? (gslot * (f.R - 1) + j_) * f.nranks * f.pages_per_frame
                               : (gslot * (f.R - 1) + j_) * f.pages_per_frame;
    const uint32_t cfg[13] = {static_cast<uint32_t>(f.n), f.nranks, f.rank, dim, twice ? 1u : 0u, 0u,
                              static_cast<uint32_t>(f.wire), f.pages_per_frame, lbase, db_slot,
                              static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32), g.noawait ? 1u : 0u};

    if (f.wire == WIRE_HAD8) {
      auto* k = d.get_kernel("fused_sum_norm_had8_bfloat", lib);
      enc.set_compute_pipeline_state(k);
      enc.set_buffer(g.table_buf, 0, g.table_bias);
      enc.set_buffer(g.arena_buf, 1, g.arena_bias);
      enc.set_buffer(f.sum_land_buf, 2, 0);
      enc.set_input_array(y, 3);
      enc.set_input_array(res, 4);
      enc.set_input_array(w1, 5);
      enc.set_input_array(twice ? inputs[3] : w1, 6);
      enc.set_output_array(outputs[0], 7);
      enc.set_output_array(twice ? outputs[1] : outputs[0], 8);
      enc.set_buffer(g.db_buf, 10, 0);
      enc.set_buffer(f.counter_buf, 11, 4096u + static_cast<size_t>(j_ & 63) * 64u);
      enc.set_bytes(cfg, 13, 12);
      enc.set_bytes(eps_, 13);
      const size_t width = (dim / 64u) * 32u;
      enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(width, 1, 1));
    } else {
      auto* k = d.get_kernel("fused_sum_norm_" + tname(y.dtype()), lib);
      enc.set_compute_pipeline_state(k);
      enc.set_buffer(g.table_buf, 0, g.table_bias);
      enc.set_buffer(g.arena_buf, 1, g.arena_bias);
      enc.set_buffer(f.sum_land_buf, 2, 0);
      enc.set_input_array(y, 3);
      enc.set_input_array(res, 4);
      enc.set_input_array(w1, 5);
      enc.set_input_array(twice ? inputs[3] : w1, 6);
      enc.set_output_array(outputs[0], 7);
      enc.set_output_array(twice ? outputs[1] : outputs[0], 8);
      enc.set_input_array(w1, 9); // unread scalar
      enc.set_buffer(g.db_buf, 10, 0);
      enc.set_buffer(f.counter_buf, 11, 4096u + static_cast<size_t>(j_ & 63) * 64u);
      enc.set_bytes(cfg, 13, 12);
      enc.set_bytes(eps_, 13);
      const size_t width = (((dim + 3) / 4 + 31) / 32) * 32;
      enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(width, 1, 1));
    }
  }
  const char* name() const override { return "FusedSumNorm"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
 private:
  uint32_t j_;
  int pid_;
  float eps_;
  bool published_;
};

class GemvPublish : public mx::Primitive {
 public:
  GemvPublish(mx::Stream s, uint32_t j, int pid) : mx::Primitive(s), j_(j), pid_(pid) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    FusedPlan& f = g.plans.at(pid_);
    out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
    if (out[0].size() != f.n) throw std::runtime_error("meshmlx: gemv_publish size mismatch");
    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
    auto& enc = mx::metal::get_command_encoder(s);

    uint32_t gslot = 0, db_slot = 0;
    uint64_t seq = 0;
    if (f.wire != WIRE_HAD8) {
      prepare_fused_frame(f, pid_, j_, gslot, db_slot, seq);
    }
    const uint32_t B = decode_rows(in[0]), K = static_cast<uint32_t>(in[0].shape(-1)), N = static_cast<uint32_t>(in[1].shape(0));
    const Launch l = launch_for("gemv_publish", N, K);
    size_t groups = 0;
    enc.set_compute_pipeline_state(d.get_kernel(launch_kernel("gemv_publish", B, l, N, K, groups), lib));
    enc.set_input_array(in[0], 0);
    enc.set_input_array(in[1], 1);
    enc.set_output_array(out[0], 2);
    const size_t src_off = g.arena_bias + static_cast<size_t>(f.src_base_page + (gslot * f.R + j_) * f.pages_per_frame) * MESH_PAGE_BYTES;
    enc.set_buffer(g.arena_buf, 3, src_off);
    enc.set_buffer(g.db_buf, 4, 0);
    enc.set_buffer(f.counter_buf, 5, static_cast<size_t>(j_ & 63) * 64u);
    const uint32_t dims[6] = {N, K, db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32),
                              static_cast<uint32_t>(f.wire)};
    enc.set_bytes(dims, 6, 6);
    enc.dispatch_threadgroups(MTL::Size(groups, 1, 1), MTL::Size(32 * l.sg, 1, 1));
  }
  const char* name() const override { return "GemvPublish"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
 private:
  uint32_t j_;
  int pid_;
};

class GemmPublish : public mx::Primitive {
 public:
  GemmPublish(mx::Stream s, uint32_t j, int pid) : mx::Primitive(s), j_(j), pid_(pid) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    FusedPlan& f = g.plans.at(pid_);
    out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
    if (out[0].size() != f.n) throw std::runtime_error("meshmlx: gemm_publish size mismatch");
    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_gemm", []() { return gemm_source(); });
    auto& enc = mx::metal::get_command_encoder(s);

    uint32_t gslot = 0, db_slot = 0;
    uint64_t seq = 0;
    if (f.wire != WIRE_HAD8) {
      prepare_fused_frame(f, pid_, j_, gslot, db_slot, seq);
    }
    const mx::array &x = in[0], &w = in[1];
    const int K = x.shape(-1), N = w.shape(0), M = static_cast<int>(x.size() / K);
    const size_t src_off = g.arena_bias + static_cast<size_t>(f.src_base_page + (gslot * f.R + j_) * f.pages_per_frame) * MESH_PAGE_BYTES;
    const uint32_t pub_cfg[4] = {db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32), static_cast<uint32_t>(f.wire)};
    auto bind_pub = [&]() {
      enc.set_buffer(g.arena_buf, 5, src_off);
      enc.set_buffer(g.db_buf, 6, 0);
      enc.set_buffer(f.counter_buf, 7, static_cast<size_t>(j_ & 63) * 64u);
      enc.set_bytes(pub_cfg, 4, 8);
    };
    if (M <= 15) {
      const int passes = (M + 4) / 5, nv = (M + passes - 1) / passes, kl = passes == 1 ? 32 : 16;
      if (M < 2 || passes > 3 || K % 4) throw std::runtime_error("meshmlx: gemm_publish 2..15 rows needs K % 4");
      enc.set_compute_pipeline_state(d.get_kernel("gemm_publish_wide_" + std::to_string(nv) + "_" + std::to_string(kl), lib));
      enc.set_input_array(w, 0);
      enc.set_input_array(x, 1);
      enc.set_output_array(out[0], 3);
      const int dims[5] = {K, N, M, K, K};
      enc.set_bytes(dims, 5, 4);
      bind_pub();
      const int rows_per_tg = (32 / kl) * (kl / 8);
      enc.dispatch_threadgroups(MTL::Size(passes, (N + rows_per_tg - 1) / rows_per_tg, 1), MTL::Size(32 * (kl / 8), 1, 1));
      return;
    }
    if (M > 128 || M % 32 || N % 32 || K % 64) throw std::runtime_error("meshmlx: gemm_publish needs M%32,N%32,K%64");
    const int sm = (N > 64 && M > 64) ? std::min(8, (M + 31) / 32) : 1, ks = sm == 1 ? 4 : 1 + 4 / sm;
    enc.set_compute_pipeline_state(d.get_kernel("gemm_publish_nax_" + std::to_string(sm) + "_" + std::to_string(ks), lib));
    const int tm = (M + 32 * sm - 1) / (32 * sm), tn = N / 32, k_iters = ((K + 63) / 64 + ks - 1) / ks;
    const int params[9] = {M, N, K, K, K, N, tn, tm, k_iters};
    enc.set_input_array(x, 0);
    enc.set_input_array(w, 1);
    enc.set_output_array(out[0], 3);
    enc.set_bytes(params, 9, 4);
    bind_pub();
    enc.dispatch_threadgroups(MTL::Size(tm * tn, 1, 1), MTL::Size(32 * sm * ks, 1, 1));
  }
  const char* name() const override { return "GemmPublish"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
 private:
  uint32_t j_;
  int pid_;
};

class FusedArgmax : public mx::Primitive {
 public:
  FusedArgmax(mx::Stream s, uint32_t j, int pid) : mx::Primitive(s), j_(j), pid_(pid) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    FusedPlan& f = g.plans.at(pid_);
    const mx::array &val = inputs[0], &idx = inputs[1];
    mx::array& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
    auto& enc = mx::metal::get_command_encoder(s);

    const uint32_t rows = static_cast<uint32_t>(val.size());
    const uint32_t gslot = static_cast<uint32_t>(f.active_gen % kGenSlots);
    uint64_t seq = 0;
    const uint32_t db_slot = alloc_doorbell(DB_FUSED_ARGMAX, static_cast<uint32_t>(pid_), gslot, j_, 1u, rows * 8u, 0u, seq);
    const size_t src_off = g.arena_bias + static_cast<size_t>(f.src_base_page + (gslot * f.R + j_) * f.pages_per_frame) * MESH_PAGE_BYTES;

    const uint32_t gw[4] = {rows, db_slot, static_cast<uint32_t>(seq), static_cast<uint32_t>(seq >> 32)};
    const uint32_t wthreads = std::max<uint32_t>(rows, 32);
    enc.set_compute_pipeline_state(d.get_kernel("argmax_write_pub", lib));
    enc.set_input_array(val, 0);
    enc.set_input_array(idx, 1);
    enc.set_buffer(g.arena_buf, 2, src_off);
    enc.set_buffer(g.db_buf, 3, 0);
    enc.set_bytes(gw, 4, 4);
    enc.dispatch_threads(MTL::Size(wthreads, 1, 1), MTL::Size(std::min<uint32_t>(wthreads, 256), 1, 1));

    const uint32_t gp[8] = {rows, f.nranks, f.rank, gslot * f.nranks, db_slot, static_cast<uint32_t>(seq),
                            static_cast<uint32_t>(seq >> 32), g.noawait ? 1u : 0u};
    enc.set_compute_pipeline_state(d.get_kernel("argmax_pick", lib));
    enc.set_buffer(g.table_buf, 0, g.table_bias);
    enc.set_buffer(g.arena_buf, 1, g.arena_bias);
    enc.set_buffer(f.argmax_land_buf, 2, 0);
    enc.set_input_array(val, 3);
    enc.set_input_array(idx, 4);
    enc.set_output_array(out, 5);
    enc.set_buffer(g.db_buf, 6, 0);
    enc.set_bytes(gp, 8, 7);
    enc.dispatch_threads(MTL::Size(wthreads, 1, 1), MTL::Size(std::min<uint32_t>(wthreads, 256), 1, 1));
  }
  const char* name() const override { return "FusedArgmax"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
 private:
  uint32_t j_;
  int pid_;
};

// ---- Pure compute primitives (verbatim from mesh-resume-mlx) ----

class DepCopy : public mx::Primitive {
 public:
  explicit DepCopy(mx::Stream s) : mx::Primitive(s) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
    auto& s = stream(); auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_gemm", []() { return gemm_source(); });
    auto& enc = mx::metal::get_command_encoder(s);
    const uint32_t n4 = static_cast<uint32_t>(in[0].nbytes() / 8);
    enc.set_compute_pipeline_state(d.get_kernel("dep_copy", lib));
    enc.set_input_array(in[0], 0); enc.set_input_array(in[1], 1); enc.set_output_array(out[0], 2); enc.set_bytes(n4, 3);
    enc.dispatch_threads(MTL::Size(n4, 1, 1), MTL::Size(std::min<uint32_t>(n4, 256), 1, 1));
  }
  const char* name() const override { return "DepCopy"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
};

class GemmRows : public mx::Primitive {
 public:
  explicit GemmRows(mx::Stream s) : mx::Primitive(s) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    for (auto& o : out) o.set_data(mx::allocator::malloc(o.nbytes()));
    auto& s = stream(); auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_gemm", []() { return gemm_source(); });
    auto& enc = mx::metal::get_command_encoder(s);
    const mx::array &x = in[0], &w = in[1];
    const int K = x.shape(-1), N = w.shape(0), M = static_cast<int>(x.size() / K);
    if (M <= 15) {
      const int passes = (M + 4) / 5, nv = (M + passes - 1) / passes, kl = passes == 1 ? 32 : 16;
      enc.set_compute_pipeline_state(d.get_kernel("gemm_rows_wide_" + std::to_string(nv) + "_" + std::to_string(kl), lib));
      enc.set_input_array(w, 0); enc.set_input_array(x, 1); enc.set_output_array(out[0], 3);
      const int dims[5] = {K, N, M, K, K}; enc.set_bytes(dims, 5, 4);
      const int rows_per_tg = (32 / kl) * (kl / 8);
      enc.dispatch_threadgroups(MTL::Size(passes, (N + rows_per_tg - 1) / rows_per_tg, 1), MTL::Size(32 * (kl / 8), 1, 1));
      return;
    }
    const int sm = (N > 64 && M > 64) ? std::min(8, (M + 31) / 32) : 1, ks = sm == 1 ? 4 : 1 + 4 / sm;
    enc.set_compute_pipeline_state(d.get_kernel("gemm_rows_nax_" + std::to_string(sm) + "_" + std::to_string(ks), lib));
    const int tm = (M + 32 * sm - 1) / (32 * sm), tn = N / 32, k_iters = ((K + 63) / 64 + ks - 1) / ks;
    const int params[9] = {M, N, K, K, K, N, tn, tm, k_iters};
    enc.set_input_array(x, 0); enc.set_input_array(w, 1); enc.set_output_array(out[0], 3); enc.set_bytes(params, 9, 4);
    enc.dispatch_threadgroups(MTL::Size(tm * tn, 1, 1), MTL::Size(32 * sm * ks, 1, 1));
  }
  const char* name() const override { return "GemmRows"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
};

class GateUpGeglu : public mx::Primitive {
 public:
  explicit GateUpGeglu(mx::Stream s) : mx::Primitive(s) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
    auto& s = stream(); auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
    auto& enc = mx::metal::get_command_encoder(s);
    const uint32_t B = decode_rows(in[0]), K = static_cast<uint32_t>(in[0].shape(-1)), N = static_cast<uint32_t>(in[1].shape(0));
    const Launch l = launch_for("gate_up_geglu", N, K); size_t groups;
    enc.set_compute_pipeline_state(d.get_kernel(launch_kernel("gate_up_geglu", B, l, N, K, groups), lib));
    enc.set_input_array(in[0], 0); enc.set_input_array(in[1], 1); enc.set_input_array(in[2], 2); enc.set_output_array(out[0], 3);
    const uint32_t dims[2] = {N, K}; enc.set_bytes(dims, 2, 4);
    enc.dispatch_threadgroups(MTL::Size(groups, 1, 1), MTL::Size(32 * l.sg, 1, 1));
  }
  const char* name() const override { return "GateUpGeglu"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
};

class GemvRows : public mx::Primitive {
 public:
  explicit GemvRows(mx::Stream s) : mx::Primitive(s) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
    auto& s = stream(); auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
    auto& enc = mx::metal::get_command_encoder(s);
    const uint32_t B = decode_rows(in[0]), K = static_cast<uint32_t>(in[0].shape(-1)), N = static_cast<uint32_t>(in[1].shape(0));
    const Launch l = launch_for("gemv_rows", N, K); size_t groups;
    enc.set_compute_pipeline_state(d.get_kernel(launch_kernel("gemv_rows", B, l, N, K, groups), lib));
    enc.set_input_array(in[0], 0); enc.set_input_array(in[1], 1); enc.set_output_array(out[0], 2);
    const uint32_t dims[2] = {N, K}; enc.set_bytes(dims, 2, 3);
    enc.dispatch_threadgroups(MTL::Size(groups, 1, 1), MTL::Size(32 * l.sg, 1, 1));
  }
  const char* name() const override { return "GemvRows"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
};

class PleGate : public mx::Primitive {
 public:
  PleGate(mx::Stream s, uint32_t layer) : mx::Primitive(s), layer_(layer) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
    auto& s = stream(); auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
    auto& enc = mx::metal::get_command_encoder(s);
    const uint32_t B = decode_rows(in[0]), K = static_cast<uint32_t>(in[0].shape(-1)), N = static_cast<uint32_t>(in[1].shape(0));
    const uint32_t layers = static_cast<uint32_t>(in[2].shape(-2));
    const Launch l = launch_for("ple_gate", N, K); size_t groups;
    enc.set_compute_pipeline_state(d.get_kernel(launch_kernel("ple_gate", B, l, N, K, groups), lib));
    enc.set_input_array(in[0], 0); enc.set_input_array(in[1], 1); enc.set_input_array(in[2], 2); enc.set_output_array(out[0], 3);
    const uint32_t dims[4] = {N, K, layers * N, layer_}; enc.set_bytes(dims, 4, 4);
    enc.dispatch_threadgroups(MTL::Size(groups, 1, 1), MTL::Size(32 * l.sg, 1, 1));
  }
  const char* name() const override { return "PleGate"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
 private:
  uint32_t layer_;
};

class AddNorm : public mx::Primitive {
 public:
  AddNorm(mx::Stream s, float eps) : mx::Primitive(s), eps_(eps) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    for (auto& o : out) o.set_data(mx::allocator::malloc(o.nbytes()));
    auto& s = stream(); auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
    auto& enc = mx::metal::get_command_encoder(s);
    const mx::array& p = in[0];
    const uint32_t dim = static_cast<uint32_t>(p.shape(-1)), rows = static_cast<uint32_t>(p.size() / dim);
    auto* k = d.get_kernel("fused_sum_norm_" + tname(p.dtype()), lib);
    enc.set_compute_pipeline_state(k);
    enc.set_input_array(p, 0); enc.set_input_array(p, 1); enc.set_input_array(p, 2); // unread (ranks == 1)
    enc.set_input_array(p, 3);
    enc.set_input_array(in[1], 4); enc.set_input_array(in[2], 5); enc.set_input_array(in[4], 6);
    enc.set_output_array(out[0], 7); enc.set_output_array(out[1], 8);
    enc.set_input_array(in[3], 9);
    enc.set_input_array(p, 10); enc.set_input_array(p, 11); // unread
    const uint32_t cfg[13] = {static_cast<uint32_t>(p.size()), 1u, 0u, dim, 1u, 1u, 0u, 1u, 0u, 0u, 0u, 0u, 1u};
    enc.set_bytes(cfg, 13, 12);
    enc.set_bytes(eps_, 13);
    const size_t width = (((dim + 3) / 4 + 31) / 32) * 32;
    enc.dispatch_threadgroups(MTL::Size(rows, 1, 1), MTL::Size(width, 1, 1));
  }
  const char* name() const override { return "AddNorm"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
 private:
  float eps_;
};

class HeadNormRope : public mx::Primitive {
 public:
  HeadNormRope(mx::Stream s, uint32_t h0, uint32_t h1, uint32_t h2, uint32_t D, uint32_t rope, bool w2, float eps, float base_log2)
      : mx::Primitive(s), h0_(h0), h1_(h1), h2_(h2), D_(D), rope_(rope), w2_(w2), eps_(eps), base_log2_(base_log2) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    for (auto& o : out) o.set_data(mx::allocator::malloc(o.nbytes()));
    auto& s = stream(); auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
    auto& enc = mx::metal::get_command_encoder(s);
    const mx::array& x = in[0];
    const uint32_t B = static_cast<uint32_t>(x.shape(0)), S = static_cast<uint32_t>(x.shape(-1));
    enc.set_compute_pipeline_state(d.get_kernel("head_norm_rope_" + tname(x.dtype()), lib));
    enc.set_input_array(x, 0); enc.set_input_array(in[1], 1); enc.set_input_array(in[4], 2); enc.set_input_array(in[5], 3);
    enc.set_output_array(out[0], 4);
    enc.set_output_array(out.size() > 1 ? out[1] : out[0], 5);
    enc.set_output_array(out.size() > 1 ? out[2] : out[0], 6);
    enc.set_input_array(in[2], 7); enc.set_input_array(in[3], 8);
    const uint32_t p[7] = {h0_, h1_, h2_, D_, rope_, w2_ ? 1u : 0u, S}; enc.set_bytes(p, 7, 9);
    const float f[2] = {eps_, base_log2_}; enc.set_bytes(f, 2, 10);
    enc.dispatch_threadgroups(MTL::Size(B * (h0_ + h1_ + h2_), 1, 1), MTL::Size(D_ / 4, 1, 1));
  }
  const char* name() const override { return "HeadNormRope"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
 private:
  uint32_t h0_, h1_, h2_, D_, rope_;
  bool w2_;
  float eps_, base_log2_;
};

class RowArgmax : public mx::Primitive {
 public:
  RowArgmax(mx::Stream s, uint32_t offset) : mx::Primitive(s), offset_(offset) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override { throw std::runtime_error("GPU only"); }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    for (auto& o : out) o.set_data(mx::allocator::malloc(o.nbytes()));
    auto& s = stream(); auto& d = mx::metal::device(s.device);
    auto* lib = d.get_library("meshmlx_fused", []() { return fused_source(); });
    auto& enc = mx::metal::get_command_encoder(s);
    const uint32_t V = static_cast<uint32_t>(in[0].shape(-1)), B = static_cast<uint32_t>(in[0].size() / V), P = static_cast<uint32_t>(out[2].shape(1));
    enc.set_compute_pipeline_state(d.get_kernel("row_argmax_parts", lib));
    enc.set_input_array(in[0], 0); enc.set_output_array(out[2], 1); enc.set_bytes(V, 2);
    enc.dispatch_threadgroups(MTL::Size(P, B, 1), MTL::Size(256, 1, 1));
    enc.set_compute_pipeline_state(d.get_kernel("row_argmax_pick", lib));
    enc.set_input_array(out[2], 0); enc.set_output_array(out[0], 1); enc.set_output_array(out[1], 2);
    const uint32_t gp[2] = {P, offset_}; enc.set_bytes(gp, 2, 3);
    enc.dispatch_threadgroups(MTL::Size(B, 1, 1), MTL::Size(std::min<uint32_t>(1024, (P + 31) / 32 * 32), 1, 1));
  }
  const char* name() const override { return "RowArgmax"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
 private:
  uint32_t offset_;
};

// No-op relay primitive (our flat-DMA mesh-flow daemon relays automatically with zero client work).
class RelayNoop : public mx::Primitive {
 public:
  explicit RelayNoop(mx::Stream s) : mx::Primitive(s) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {}
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>& out) override {
    for (auto& o : out) o.set_data(mx::allocator::malloc(o.nbytes()));
  }
  const char* name() const override { return "RelayNoop"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
};

double host_seconds() {
  static mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t t; mach_timebase_info(&t); return t; }();
  return static_cast<double>(mach_absolute_time()) * tb.numer / tb.denom / 1e9;
}

}  // namespace

NB_MODULE(_meshmlx, m) {
  nb::module_::import_("mlx.core");

  m.def("host_seconds", []() { return host_seconds(); });
  m.def("set_noawait", [](bool on) { g.noawait = on; }, nb::arg("on"));
  m.def("encode_stats", []() { auto r = std::make_tuple(g.encode_ns, g.calls); g.encode_ns = g.calls = 0; return r; });
  m.def("clock", []() { return now_ns(); });
  m.def("paths_newer", []() { return false; });
  m.def("resume", []() { mx::synchronize(); return false; });
  m.def("resume_pending", []() { return false; });
  m.def("rerun_sums", [](uint32_t) {});
  m.def("rerun", [](uint32_t, int) {}, nb::arg("n"), nb::arg("plan") = -1);
  m.def("relay_resume", [](int) { return 0u; }, nb::arg("plan") = -1);

  m.def("nodes", []() {
    std::vector<uint32_t> n(g.ranks);
    for (uint32_t i = 0; i < g.ranks; i++) n[i] = g.world_nodes[i];
    return n;
  });

  m.def("schedule", []() {
    return std::make_tuple(0ull, static_cast<uint64_t>(MESH_PAGE_BYTES), 0ull, 0ull, 0ull, 0ull);
  });

  m.def("compute", []() { return std::make_tuple(g.nranks, g.index); });

  m.def("whole_map", []() {
    std::vector<uint32_t> nodes;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> links;
    if (!g.attached) return std::make_tuple(nodes, links);
    const uint64_t ver = hdr_map_ver(g.ctx.M);
    const auto* copies = reinterpret_cast<const MeshMapCopyWire*>(reinterpret_cast<const uint8_t*>(g.ctx.M) + hdr_map_off(g.ctx.M));
    const MeshMapCopyWire& c = copies[ver & 1];
    const uint32_t n = c.nodes < 256 ? static_cast<uint32_t>(c.nodes) : 256u;
    for (uint32_t i = 0; i < n; i++) nodes.push_back(c.record[i].node);
    for (uint32_t i = 0; i < n; i++) {
      const uint32_t a = c.record[i].node;
      const uint32_t rc = c.record[i].reached < 32 ? c.record[i].reached : 32u;
      for (uint32_t j = 0; j < rc; j++) {
        const uint32_t b = c.record[i].reach[j].node;
        if (a < b) links.emplace_back(a, b, c.record[i].reach[j].bandwidth);
      }
    }
    return std::make_tuple(nodes, links);
  });

  m.def("fabric", []() {
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t, uint64_t>> out;
    if (!g.attached) return std::make_tuple(out, 1u);
    const uint64_t ver = hdr_map_ver(g.ctx.M);
    const auto* copies = reinterpret_cast<const MeshMapCopyWire*>(reinterpret_cast<const uint8_t*>(g.ctx.M) + hdr_map_off(g.ctx.M));
    const MeshMapCopyWire& c = copies[ver & 1];
    const uint32_t n = c.nodes < 256 ? static_cast<uint32_t>(c.nodes) : 256u;
    auto rank_of = [&](uint32_t node) -> int {
      for (uint32_t r = 0; r < g.ranks; r++) if (g.world_nodes[r] == node) return static_cast<int>(r);
      return -1;
    };
    uint64_t wire = 0;
    for (uint32_t i = 0; i < n; i++) {
      const int ra = rank_of(c.record[i].node);
      if (ra < 0) continue;
      const uint32_t rc = c.record[i].reached < 32 ? c.record[i].reached : 32u;
      for (uint32_t j = 0; j < rc; j++) {
        const int rb = rank_of(c.record[i].reach[j].node);
        if (rb > ra) out.emplace_back(static_cast<uint32_t>(ra), static_cast<uint32_t>(rb), wire++, c.record[i].reach[j].bandwidth);
      }
    }
    return std::make_tuple(out, 1u);
  });

  m.def("init", [](int nranks, int rank, size_t slot_bytes, int slots, uint32_t /*polls*/, std::vector<int> relays) {
    if (nranks < 1 || nranks > static_cast<int>(kMaxRanks)) throw std::runtime_error("meshmlx: invalid nranks");
    g.default_wire = parse_wire_mode();
    g.ranks = static_cast<uint32_t>(nranks);
    g.world_rank = static_cast<uint32_t>(rank);
    std::vector<bool> is_relay(nranks, false);
    for (int r : relays) {
      if (r < 0 || r >= nranks || is_relay[r]) throw std::runtime_error("meshmlx: invalid RELAY_RANKS");
      is_relay[r] = true;
    }
    g.nranks = 0;
    g.index = -1;
    for (int r = 0; r < nranks; r++) {
      if (r == rank) g.before = g.nranks;
      if (is_relay[r]) continue;
      if (r == rank) g.index = static_cast<int32_t>(g.nranks);
      g.nranks++;
    }
    if (!g.nranks) throw std::runtime_error("meshmlx: every rank a relay");

    if (mesh_attach(&g.ctx) != 0) throw std::runtime_error("meshmlx: mesh_attach(/mesh0) failed");
    g.attached = true;
    g.my_node = hdr_node(g.ctx.M);

    bootstrap_connect(nranks, rank);
    bootstrap_allgather(&g.my_node, g.world_nodes, sizeof(uint32_t));
    uint32_t ci = 0;
    for (int r = 0; r < nranks; r++) {
      if (!is_relay[r]) g.compute_nodes[ci++] = g.world_nodes[r];
    }

    // Borrow /mesh0 W[] table and P[] flat page arena zero-copy as Metal buffers.
    auto* dev = mx::metal::device(mx::Device::gpu).mtl_device();
    const uintptr_t t_addr = reinterpret_cast<uintptr_t>(g.ctx.M) + static_cast<uintptr_t>(hdr_table_off(g.ctx.M));
    const uintptr_t t_aln = t_addr & ~static_cast<uintptr_t>(16383u);
    const size_t t_bytes = ((static_cast<size_t>(hdr_entries(g.ctx.M)) * 8u + (t_addr - t_aln) + 16383u) & ~static_cast<size_t>(16383u));
    g.table_bias = static_cast<size_t>(t_addr - t_aln);
    g.table_buf = dev->newBuffer(reinterpret_cast<void*>(t_aln), t_bytes, MTL::ResourceStorageModeShared, nullptr);
    if (!g.table_buf) throw std::runtime_error("meshmlx: failed to wrap /mesh0 W[] as MTLBuffer");

    const uintptr_t d_addr = reinterpret_cast<uintptr_t>(g.ctx.M) + static_cast<uintptr_t>(hdr_data_off(g.ctx.M));
    const uintptr_t d_aln = d_addr & ~static_cast<uintptr_t>(16383u);
    const uintptr_t d_end = reinterpret_cast<uintptr_t>(g.ctx.M) + static_cast<uintptr_t>(hdr_length(g.ctx.M));
    const size_t d_bytes = (d_end > d_aln) ? (static_cast<size_t>(d_end - d_aln) & ~static_cast<size_t>(16383u)) : 0u;
    g.arena_bias = static_cast<size_t>(d_addr - d_aln);
    g.arena_buf = dev->newBuffer(reinterpret_cast<void*>(d_aln), d_bytes, MTL::ResourceStorageModeShared, nullptr);
    if (!g.arena_buf) throw std::runtime_error("meshmlx: failed to wrap /mesh0 arena as MTLBuffer");

    // Shared doorbell ring buffer + submitter thread.
    g.db_buf = dev->newBuffer(kDbRingSize * sizeof(Doorbell), MTL::ResourceStorageModeShared);
    g.db = static_cast<Doorbell*>(g.db_buf->contents());
    memset(g.db, 0, kDbRingSize * sizeof(Doorbell));
    g.db_encode_seq = 0;
    g.pub_head.store(0);
    g.ret_head.store(0);
    g.stop_worker.store(false);
    g.worker = std::thread(worker_loop);

    // Preallocate AllSum ring (capped at 8 slots to keep /mesh0 footprint compact).
    g.slot_bytes = slot_bytes;
    g.allsum_slots = static_cast<uint32_t>(std::min(std::max(slots, 4), 8));
    g.allsum_max_pages = static_cast<uint32_t>((slot_bytes + MESH_PAGE_BYTES - 1) / MESH_PAGE_BYTES);
    g.allsum_next = 0;
    g.ring.clear();
    g.ring.resize(g.allsum_slots);
    std::vector<uint32_t> my_all_idx(static_cast<size_t>(g.allsum_slots) * g.allsum_max_pages, 0);
    for (uint32_t s = 0; s < g.allsum_slots; s++) {
      void* base = nullptr;
      if (mesh_alloc_contiguous(&g.ctx, g.allsum_max_pages, nullptr, &base) != 0)
        throw std::runtime_error("meshmlx: mesh_alloc_contiguous failed for all_sum ring");
      g.ring[s].src_base_page = static_cast<uint32_t>((reinterpret_cast<uint8_t*>(base) - (reinterpret_cast<uint8_t*>(g.ctx.M) + hdr_data_off(g.ctx.M))) / MESH_PAGE_BYTES);
      g.ring[s].max_pages = g.allsum_max_pages;
      g.ring[s].my_land.resize(g.allsum_max_pages);
      if (mesh_alloc(&g.ctx, g.allsum_max_pages, 0, g.ring[s].my_land.data(), nullptr) != 0)
        throw std::runtime_error("meshmlx: mesh_alloc failed for all_sum landing indices");
      mesh_clear_indices(&g.ctx, g.ring[s].my_land.data(), g.allsum_max_pages);
      memcpy(my_all_idx.data() + static_cast<size_t>(s) * g.allsum_max_pages, g.ring[s].my_land.data(),
             g.allsum_max_pages * sizeof(uint32_t));
    }
    std::vector<uint32_t> gathered(static_cast<size_t>(g.ranks) * my_all_idx.size());
    bootstrap_allgather(my_all_idx.data(), gathered.data(), my_all_idx.size() * sizeof(uint32_t));
    uint32_t c_idx = 0;
    for (int r = 0; r < nranks; r++) {
      if (is_relay[r]) continue;
      const uint32_t* base = gathered.data() + static_cast<size_t>(r) * my_all_idx.size();
      for (uint32_t s = 0; s < g.allsum_slots; s++) {
        g.ring[s].all_land[c_idx].assign(base + static_cast<size_t>(s) * g.allsum_max_pages,
                                         base + static_cast<size_t>(s + 1) * g.allsum_max_pages);
      }
      c_idx++;
    }
    g.allsum_land_buf = dev->newBuffer(my_all_idx.data(), my_all_idx.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    g.allsum_counter_buf = dev->newBuffer(8192, MTL::ResourceStorageModeShared);
    memset(g.allsum_counter_buf->contents(), 0, 8192);
    bootstrap_barrier();
  }, nb::arg("nranks"), nb::arg("rank"), nb::arg("slot_bytes") = 4 << 20, nb::arg("slots") = 256,
     nb::arg("polls") = 0, nb::arg("relays") = std::vector<int>{});

  m.def("fused_init", [](size_t n, uint32_t R, int esize, int pid) {
    // Synchronize GPU & worker before reallocating a reused plan slot.
    mx::synchronize();
    while (g.ret_head.load(std::memory_order_acquire) < g.db_encode_seq) usleep(10);
    bootstrap_barrier();

    if (pid < 0) {
      g.plans.emplace_back();
      pid = static_cast<int>(g.plans.size()) - 1;
    }
    FusedPlan& f = g.plans.at(pid);
    f.release_resources(&g.ctx);
    f.n = n;
    f.R = R;
    f.esize = static_cast<size_t>(esize);
    f.nranks = g.nranks;
    f.rank = g.index >= 0 ? static_cast<uint32_t>(g.index) : 0u;
    f.wire = g.default_wire;
    if (f.wire == WIRE_HAD8 && (f.esize != 2 || (f.n % 64) != 0)) f.wire = WIRE_BF16;

    if (f.wire == WIRE_HAD8) {
      f.red_op = MESH_RED_SUM_HAD_I8;
      const uint32_t n_blks = static_cast<uint32_t>(f.n / 64);
      f.pages_per_frame = (n_blks + 59u) / 60u;
      const uint32_t rem_blks = n_blks % 60u ? n_blks % 60u : 60u;
      f.last_page_bytes = rem_blks * 68u;
    } else if (f.wire == WIRE_F32) {
      f.red_op = MESH_RED_SUM_F32;
      const size_t bytes = f.n * 4u;
      f.pages_per_frame = static_cast<uint32_t>((bytes + MESH_PAGE_BYTES - 1) / MESH_PAGE_BYTES);
      f.last_page_bytes = static_cast<uint32_t>(bytes - static_cast<size_t>(f.pages_per_frame - 1) * MESH_PAGE_BYTES);
    } else {
      f.red_op = (f.esize == 2) ? MESH_RED_SUM_BF16 : MESH_RED_SUM_F32;
      const size_t bytes = f.n * f.esize;
      f.pages_per_frame = static_cast<uint32_t>((bytes + MESH_PAGE_BYTES - 1) / MESH_PAGE_BYTES);
      f.last_page_bytes = static_cast<uint32_t>(bytes - static_cast<size_t>(f.pages_per_frame - 1) * MESH_PAGE_BYTES);
    }
    if (f.pages_per_frame == 0) f.pages_per_frame = 1;

    // Allocate contiguous source pages for kGenSlots * R frames.
    f.total_src_pages = kGenSlots * f.R * f.pages_per_frame;
    void* src_base = nullptr;
    if (mesh_alloc_contiguous(&g.ctx, f.total_src_pages, nullptr, &src_base) != 0)
      throw std::runtime_error("meshmlx: mesh_alloc_contiguous failed for fused plan");
    f.src_base_page = static_cast<uint32_t>(
        (reinterpret_cast<uint8_t*>(src_base) - (reinterpret_cast<uint8_t*>(g.ctx.M) + hdr_data_off(g.ctx.M))) / MESH_PAGE_BYTES);

    // Allocate landing W[] indices for R-1 sum frames + 1 argmax frame across kGenSlots generations.
    const uint32_t sum_frames = f.R > 1 ? f.R - 1 : 1;
    const uint32_t mult = (f.wire == WIRE_GATHER) ? f.nranks : 1u;
    f.sum_idx_per_rank = kGenSlots * sum_frames * mult * f.pages_per_frame;
    const uint32_t arg_idx_per_rank = kGenSlots * f.nranks;

    std::vector<uint32_t> my_sum(f.sum_idx_per_rank, 0);
    std::vector<uint32_t> my_arg(arg_idx_per_rank, 0);
    if (mesh_alloc(&g.ctx, f.sum_idx_per_rank, 0, my_sum.data(), nullptr) != 0)
      throw std::runtime_error("meshmlx: mesh_alloc failed for sum landing indices");
    mesh_clear_indices(&g.ctx, my_sum.data(), f.sum_idx_per_rank);
    if (mesh_alloc(&g.ctx, arg_idx_per_rank, 0, my_arg.data(), nullptr) != 0)
      throw std::runtime_error("meshmlx: mesh_alloc failed for argmax landing indices");
    mesh_clear_indices(&g.ctx, my_arg.data(), arg_idx_per_rank);

    std::vector<uint32_t> all_sum(static_cast<size_t>(g.ranks) * f.sum_idx_per_rank);
    std::vector<uint32_t> all_arg(static_cast<size_t>(g.ranks) * arg_idx_per_rank);
    bootstrap_allgather(my_sum.data(), all_sum.data(), my_sum.size() * sizeof(uint32_t));
    bootstrap_allgather(my_arg.data(), all_arg.data(), my_arg.size() * sizeof(uint32_t));

    uint32_t ci = 0;
    for (uint32_t r = 0; r < g.ranks; r++) {
      // Map world ranks to compute ranks by matching world_nodes against compute_nodes.
      bool is_comp = false;
      for (uint32_t k = 0; k < g.nranks; k++) if (g.compute_nodes[k] == g.world_nodes[r]) { is_comp = true; break; }
      if (!is_comp) continue;
      f.sum_land[ci].assign(all_sum.data() + static_cast<size_t>(r) * f.sum_idx_per_rank,
                            all_sum.data() + static_cast<size_t>(r + 1) * f.sum_idx_per_rank);
      f.argmax_land[ci].assign(all_arg.data() + static_cast<size_t>(r) * arg_idx_per_rank,
                               all_arg.data() + static_cast<size_t>(r + 1) * arg_idx_per_rank);
      ci++;
    }

    auto* dev = mx::metal::device(mx::Device::gpu).mtl_device();
    f.sum_land_buf = dev->newBuffer(my_sum.data(), my_sum.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    f.argmax_land_buf = dev->newBuffer(my_arg.data(), my_arg.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    f.counter_buf = dev->newBuffer(8192, MTL::ResourceStorageModeShared);
    memset(f.counter_buf->contents(), 0, 8192);
    f.next_gen = 0;
    f.active_gen = 0;
    bootstrap_barrier();
    return pid;
  }, nb::arg("n"), nb::arg("frames"), nb::arg("esize") = 4, nb::arg("plan") = -1);

  m.def("all_sum", [](const mx::array& x) {
    return mx::array(x.shape(), x.dtype(), std::make_shared<MeshAllSum>(mx::default_stream(mx::Device::gpu)), {x});
  }, nb::arg("x"));

  m.def("fused_all_sum", [](const mx::array& x, uint32_t j, int pid) {
    return mx::array(x.shape(), x.dtype(), std::make_shared<FusedAllSum>(mx::default_stream(mx::Device::gpu), j, pid), {x});
  }, nb::arg("x"), nb::arg("frame"), nb::arg("plan") = 0);

  m.def("fused_sum_norm", [](const mx::array& y, uint32_t j, const mx::array& residual, const mx::array& w1,
                             std::optional<mx::array> w2, float eps, int pid, bool published) {
    std::vector<mx::array> in{y, residual, w1};
    if (w2) in.push_back(*w2);
    const int outs = w2 ? 2 : 1;
    return mx::array::make_arrays(std::vector<mx::Shape>(outs, y.shape()), std::vector<mx::Dtype>(outs, y.dtype()),
                                  std::make_shared<FusedSumNorm>(mx::default_stream(mx::Device::gpu), j, pid, eps, published), in);
  }, nb::arg("y"), nb::arg("frame"), nb::arg("residual"), nb::arg("w1"), nb::arg("w2").none(), nb::arg("eps"),
     nb::arg("plan") = 0, nb::arg("published") = false);

  m.def("gemv_publish", [](const mx::array& x0, const mx::array& w, uint32_t j, int pid) {
    const mx::array x = mx::contiguous(x0);
    mx::Shape shape = x.shape(); shape.back() = w.shape(0);
    return mx::array(shape, x.dtype(), std::make_shared<GemvPublish>(mx::default_stream(mx::Device::gpu), j, pid), {x, mx::contiguous(w)});
  }, nb::arg("x"), nb::arg("w"), nb::arg("frame"), nb::arg("plan") = 0);

  m.def("gemm_publish", [](const mx::array& x0, const mx::array& w, uint32_t j, int pid) {
    const mx::array x = mx::contiguous(x0);
    mx::Shape shape = x.shape(); shape.back() = w.shape(0);
    return mx::array(shape, x.dtype(), std::make_shared<GemmPublish>(mx::default_stream(mx::Device::gpu), j, pid), {x, mx::contiguous(w)});
  }, nb::arg("x"), nb::arg("w"), nb::arg("frame"), nb::arg("plan") = 0);

  m.def("dep_copy", [](const mx::array& x0, const mx::array& dep) {
    const mx::array x = mx::contiguous(x0);
    return mx::array(x.shape(), x.dtype(), std::make_shared<DepCopy>(mx::default_stream(mx::Device::gpu)), {x, dep});
  }, nb::arg("x"), nb::arg("dep"));

  m.def("gemm_rows", [](const mx::array& x0, const mx::array& w, bool /*epilogue*/) {
    const mx::array x = mx::contiguous(x0);
    mx::Shape shape = x.shape(); shape.back() = w.shape(0);
    return mx::array(shape, x.dtype(), std::make_shared<GemmRows>(mx::default_stream(mx::Device::gpu)), {x, mx::contiguous(w)});
  }, nb::arg("x"), nb::arg("w"), nb::arg("epilogue") = false);

  m.def("gate_up_geglu", [](const mx::array& x0, const mx::array& wg, const mx::array& wu) {
    const mx::array x = mx::contiguous(x0);
    mx::Shape shape = x.shape(); shape.back() = wg.shape(0);
    return mx::array(shape, x.dtype(), std::make_shared<GateUpGeglu>(mx::default_stream(mx::Device::gpu)), {x, wg, wu});
  }, nb::arg("x"), nb::arg("wg"), nb::arg("wu"));

  m.def("ple_gate", [](const mx::array& x0, const mx::array& wg, const mx::array& pli, uint32_t layer) {
    const mx::array x = mx::contiguous(x0);
    mx::Shape shape = x.shape(); shape.back() = wg.shape(0);
    return mx::array(shape, x.dtype(), std::make_shared<PleGate>(mx::default_stream(mx::Device::gpu), layer), {x, wg, mx::contiguous(pli)});
  }, nb::arg("x"), nb::arg("wg"), nb::arg("pli"), nb::arg("layer"));

  m.def("add_norm", [](const mx::array& p, const mx::array& residual, const mx::array& w1, const mx::array& scalar,
                       const mx::array& w2, float eps) {
    return mx::array::make_arrays({p.shape(), p.shape()}, {p.dtype(), p.dtype()},
                                  std::make_shared<AddNorm>(mx::default_stream(mx::Device::gpu), eps),
                                  {mx::contiguous(p), mx::contiguous(residual), mx::contiguous(w1), mx::contiguous(scalar), mx::contiguous(w2)});
  }, nb::arg("p"), nb::arg("residual"), nb::arg("w1"), nb::arg("scalar"), nb::arg("w2"), nb::arg("eps"));

  m.def("head_norm_rope", [](const mx::array& x0, const mx::array& w0, uint32_t h0, const mx::array& offset, float eps,
                             std::optional<float> base, std::optional<mx::array> freqs, std::optional<mx::array> w1,
                             uint32_t h1, std::optional<mx::array> w2, uint32_t h2, uint32_t head_dim) {
    const mx::array x = mx::contiguous(x0);
    const uint32_t B = static_cast<uint32_t>(x.shape(0));
    const uint32_t rope = freqs ? 2u : (base ? 1u : 0u);
    const float base_log2 = base ? std::log2(*base) : 0.0f;
    std::vector<mx::array> in{x, mx::contiguous(w0), mx::contiguous(offset), freqs ? mx::contiguous(*freqs) : x,
                              w1 ? mx::contiguous(*w1) : x, w2 ? mx::contiguous(*w2) : x};
    std::vector<mx::Shape> shapes{{static_cast<int>(B), static_cast<int>(h0), 1, static_cast<int>(head_dim)}};
    std::vector<mx::Dtype> dtypes{x.dtype()};
    if (h1) {
      shapes.push_back({static_cast<int>(B), static_cast<int>(h1), 1, static_cast<int>(head_dim)});
      shapes.push_back({static_cast<int>(B), static_cast<int>(h2), 1, static_cast<int>(head_dim)});
      dtypes.push_back(x.dtype());
      dtypes.push_back(x.dtype());
    }
    return mx::array::make_arrays(shapes, dtypes,
                                  std::make_shared<HeadNormRope>(mx::default_stream(mx::Device::gpu), h0, h1, h2, head_dim, rope, w2.has_value(), eps, base_log2),
                                  in);
  }, nb::arg("x"), nb::arg("w0"), nb::arg("h0"), nb::arg("offset"), nb::arg("eps"), nb::arg("base") = nb::none(),
     nb::arg("freqs") = nb::none(), nb::arg("w1") = nb::none(), nb::arg("h1") = 0, nb::arg("w2") = nb::none(),
     nb::arg("h2") = 0, nb::arg("head_dim") = 256);

  m.def("gemv_rows", [](const mx::array& x0, const mx::array& w) {
    const mx::array x = mx::contiguous(x0);
    mx::Shape shape = x.shape(); shape.back() = w.shape(0);
    return mx::array(shape, x.dtype(), std::make_shared<GemvRows>(mx::default_stream(mx::Device::gpu)), {x, mx::contiguous(w)});
  }, nb::arg("x"), nb::arg("w"));

  m.def("row_argmax", [](const mx::array& lg0, uint32_t offset) {
    const mx::array lg = mx::contiguous(lg0);
    const int V = lg.shape(-1), B = static_cast<int>(lg.size() / V), P = (V + 2047) / 2048;
    auto out = mx::array::make_arrays({{B}, {B}, {B, P, 2}}, {mx::float32, mx::uint32, mx::uint32},
                                      std::make_shared<RowArgmax>(mx::default_stream(mx::Device::gpu), offset), {lg});
    return std::make_tuple(out[0], out[1]);
  }, nb::arg("lg"), nb::arg("offset") = 0);

  m.def("fused_argmax", [](const mx::array& val, const mx::array& idx, uint32_t j, int pid) {
    return mx::array(val.shape(), mx::int32, std::make_shared<FusedArgmax>(mx::default_stream(mx::Device::gpu), j, pid), {val, idx});
  }, nb::arg("val"), nb::arg("idx"), nb::arg("frame"), nb::arg("plan") = 0);

  m.def("relay_fused", [](const mx::array& after, int) {
    return mx::array({1}, mx::uint32, std::make_shared<RelayNoop>(mx::default_stream(mx::Device::gpu)), {after});
  }, nb::arg("after"), nb::arg("plan") = 0);

  m.def("relay_all_sum", [](const mx::array& after, size_t) {
    return mx::array({1}, mx::uint32, std::make_shared<RelayNoop>(mx::default_stream(mx::Device::gpu)), {after});
  }, nb::arg("after"), nb::arg("n"));

  m.def("declare_begin", []() {});
  m.def("declare_all_sum", [](size_t) {}, nb::arg("n"));
  m.def("declare_fused", [](int) {}, nb::arg("plan"));
  m.def("declare_end", []() {});

  m.def("finalize", []() {
    mx::synchronize();
    if (g.worker.joinable()) {
      while (g.ret_head.load(std::memory_order_acquire) < g.db_encode_seq) usleep(10);
      bootstrap_barrier();
      g.stop_worker.store(true, std::memory_order_release);
      g.worker.join();
    }
    for (auto& f : g.plans) f.release_resources(&g.ctx);
    g.plans.clear();
    if (g.attached) {
      for (auto& sl : g.ring) {
        if (sl.src_base_page != MESH_PAGE_NONE) {
          for (uint32_t k = 0; k < sl.max_pages; k++) mesh_release(&g.ctx, sl.src_base_page + k);
        }
        if (!sl.my_land.empty()) {
          mesh_clear_indices(&g.ctx, sl.my_land.data(), static_cast<uint32_t>(sl.my_land.size()));
          mesh_free_indices(&g.ctx, sl.my_land.data(), static_cast<uint32_t>(sl.my_land.size()));
        }
      }
      g.ring.clear();
      if (g.allsum_land_buf) { g.allsum_land_buf->release(); g.allsum_land_buf = nullptr; }
      if (g.allsum_counter_buf) { g.allsum_counter_buf->release(); g.allsum_counter_buf = nullptr; }
      if (g.db_buf) { g.db_buf->release(); g.db_buf = nullptr; g.db = nullptr; }
      if (g.table_buf) { g.table_buf->release(); g.table_buf = nullptr; }
      if (g.arena_buf) { g.arena_buf->release(); g.arena_buf = nullptr; }
      mesh_detach(&g.ctx);
      g.attached = false;
    }
    for (uint32_t r = 0; r < kMaxRanks; r++) {
      if (g.peer_fd[r] >= 0) { close(g.peer_fd[r]); g.peer_fd[r] = -1; }
    }
    if (g.listen_fd >= 0) { close(g.listen_fd); g.listen_fd = -1; }
  });
}
