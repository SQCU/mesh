// torch.distributed's "mesh" backend: a c10d::Backend whose collectives are libnccl-mesh's (../../nccl.h),
// as PyTorch's "Customize Process Group Backends Using Cpp Extensions" tutorial prescribes.  Rank 0
// makes the communicator's unique id and hands it out through the group's store.  The communicator's
// link map is the bridge's link table (ncclMeshConfig_t), which the group's options name (torch_mesh.
// Options: the region, a link-map file stated into it, each rank's node).  The reductions are
// libnccl-mesh's Metal kernels.  A failed call (a link lost, a bridge stalled, the link map's epoch
// moved: revoked) revokes the communicator; the error path, a call's own failure when it is issued or an
// earlier call's at a Work's wait, first agrees with every rank (ncclMeshCommAgree: ULFM's
// MPI_Comm_agree), then raises the agreed failure; agree() is that agreement, agreed() the last one.
//   MPS tensors in the window (window_heaps below): once the first group exists, the heaps PyTorch's MPS
// allocator makes place each buffer in a window allocation of its own, whose one Metal buffer is the
// tensor's MTLBuffer, so every MPS tensor made from then on (an op's output, x * 2, as much as a
// factory's) is window memory and its release is the allocator's own.  A window tensor (its parts
// contiguous and adjacent in one allocation) is passed in place; any other (made before the group, or
// where the window had no room) is copied by GPU blits on the MPS stream (MPSStream::copy) into a window
// allocation of the call before it and out after it (the NIC reads and writes only registered memory).
// Every call on MPS tensors is out of place where the caller's API is (the library reads the send
// buffer in place: nccl.h), in place where it is in place.  The MPS work enqueued before the call is
// declared the writer of every allocation the call touches (the fence's value, ncclMeshMemUse), so the
// call waits for exactly that: the MPS kernels' plain stores reach the NIC only once their command
// buffer completes, which the fence's event marks (the fence: one commit a call).  It runs on a deferred
// stream of its own (ncclMeshStreamDefer): the library's worker starts its transfers once the GPU has
// reached the fence, and the call's GPU work after them (the combines as the pieces land, on the
// completion words the bridge sets; its completion value) is kept until its Work's wait() encodes it into
// the current MPS command buffer (ncclMeshStreamEncodeWait), uncommitted: the next fence, or PyTorch's own
// commits, submit it, and the MPS work after it follows it in that command buffer.  No host
// synchronization and no other queue sits between the MPS work before and after the call.
//   CPU tensors: a tensor, or parts contiguous and adjacent, are the buffer itself (libnccl-mesh copies
// memory outside the window in and out by GPU blits through a Metal buffer over its pages); parts that
// are not are copied by the backend's own GPU blits, through Metal buffers over their pages, into a
// window allocation and out of it.  A collective on them completes before it returns; a send or recv
// runs on a stream of its own (as NCCL's point-to-point calls on separate streams), so isend/irecv
// pairs progress together, and its Work waits for that stream.
//   MESH_TRACE=<file>: each call's host times and bytes, when the GPU reached the MPS fence before it, the
// call's stream reached its value, the MPS stream reached its kept work and passed it (words the GPU
// publishes before and after it, system-coherent, polled by a thread of the trace's: no command buffer of
// its own), the GPU times of the MPS command buffer that ends at the fence, and libnccl-mesh's tallies of
// the call (ncclMeshGroupTally), written as JSON lines when the group is destroyed or the process exits.
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <pybind11/chrono.h>
#include <dlfcn.h>
#include <objc/runtime.h>

#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <time.h>
#include <unistd.h>

#include "nccl.h"

namespace c10d {

static std::pair<uint64_t, uint64_t> agree_on(ncclComm_t comm, const char *after);
// A failure raised; a failure of the communicator (ncclRemoteError, ncclTimeout, ncclSystemError: the
// network, a revoked call) only once every rank has agreed on it, the first failed call and each rank's
// link-table epoch in its message.
static void check(ncclResult_t result, ncclComm_t comm, const char *what) {
  if (result == ncclSuccess) return;
  const std::string cause = ncclGetLastError(comm);
  if (comm && (result == ncclRemoteError || result == ncclTimeout || result == ncclSystemError)) {
    auto agreed = agree_on(comm, " after the failure");
    TORCH_CHECK(false, "mesh: ", what, ": ", ncclGetErrorString(result), ": ", cause, " [agreed by every rank: call ", agreed.first,
                " since the previous agreement failed; this rank plans on its link-table epoch ", agreed.second, "]");
  }
  TORCH_CHECK(false, "mesh: ", what, ": ", ncclGetErrorString(result), ": ", cause);
}

static ncclDataType_t datatype(const at::Tensor &t) {
  switch (t.scalar_type()) {
  case at::kChar: return ncclInt8;
  case at::kByte: case at::kBool: return ncclUint8;
  case at::kInt: return ncclInt32;
  case at::kUInt32: return ncclUint32;
  case at::kLong: return ncclInt64;
  case at::kUInt64: return ncclUint64;
  case at::kHalf: return ncclFloat16;
  case at::kFloat: return ncclFloat32;
  case at::kDouble: return ncclFloat64;
  case at::kBFloat16: return ncclBfloat16;
  case at::kFloat8_e4m3fn: return ncclFloat8e4m3;
  case at::kFloat8_e5m2: return ncclFloat8e5m2;
  default: TORCH_CHECK(false, "mesh: no NCCL datatype for ", t.scalar_type());
  }
}

// The backend's own copies (libnccl-mesh counts its own): bytes the CPU copied (a non-contiguous CPU
// tensor made contiguous, and back) and bytes the GPU copied (an MPS tensor outside the window, CPU
// parts, into and out of the call's window allocation; an all-to-all's own rows); its handoffs: the MPS
// command buffers it committed, the fences (an event signalled at a command buffer's end); and the MPS
// allocator's heaps made window heaps, their buffers in the window and those the device made (no room).
static std::atomic<uint64_t> cpu_copied{0}, gpu_copied{0}, commits{0}, fences{0}, window_heaps_made{0}, window_buffers{0}, device_buffers{0};

static id<MTLDevice> device() {
  static id<MTLDevice> made = at::mps::is_available() ? at::mps::MPSDevice::getInstance()->device() : MTLCreateSystemDefaultDevice();
  return made;
}

// ---- the trace (MESH_TRACE) ----
static bool tracing() {
  static const bool on = getenv("MESH_TRACE") != nullptr;
  return on;
}
static uint64_t uptime() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }  // Python's time.monotonic_ns
struct Traced {
  std::string op, dtype;
  bool mps = false;
  uint64_t bytes = 0, in_place = 0, blit_in = 0, blit_out = 0, enter = 0, issued = 0, fence = 0, reach = 0, resume = 0, commits = 0;
  std::vector<int64_t> splits;
  void *tally = nullptr, *stream_event = nullptr;
  uint64_t stream_value = 0;
};
// The trace's state, never destroyed (its thread polls it until the process ends).
struct TraceState {
  std::mutex lock;
  std::vector<Traced> calls;
  std::map<std::pair<void *, uint64_t>, uint64_t> reached;          // (event or word, value): when the GPU reached it
  std::vector<std::tuple<void *, uint64_t, bool>> pending;          // (event or word, value, a word) not yet reached
  std::map<uint64_t, std::pair<double, double>> gpu;                // fence value: its command buffer's GPU times (s)
  uint64_t *words = nullptr;                                        // the words the GPU publishes (a window allocation)
  size_t used = 0;
};
static TraceState &trace = *new TraceState;
static void trace_dump();
// The value an event or a word holds (a word the GPU stores system-coherent, seen within microseconds; an
// event signalled inside a command buffer is seen only as the command buffer ends).
static uint64_t value_of(void *at, bool word) {
  return word ? __atomic_load_n((uint64_t *)at, __ATOMIC_ACQUIRE) : [(__bridge id<MTLSharedEvent>)at signaledValue];
}
// The trace's thread: each pending point noted in trace.reached once it is reached, polled.
static void poll_trace() {
  for (;;) {
    bool idle;
    {
      std::lock_guard<std::mutex> guard(trace.lock);
      for (auto it = trace.pending.begin(); it != trace.pending.end();) {
        auto [at, value, word] = *it;
        if (value_of(at, word) < value) { ++it; continue; }
        trace.reached[{at, value}] = uptime();
        if (!word) [(__bridge id<MTLSharedEvent>)at release];
        it = trace.pending.erase(it);
      }
      idle = trace.pending.empty();
    }
    if (idle) usleep(200);
    else sched_yield();
  }
}
static void pend(void *at, uint64_t value, bool word) {
  static std::once_flag once;
  std::call_once(once, [] { std::thread(poll_trace).detach(); });
  std::lock_guard<std::mutex> guard(trace.lock);
  trace.pending.emplace_back(at, value, word);
}
// When `event` reaches `value`, noted in trace.reached.
static void note_reached(id<MTLSharedEvent> event, uint64_t value) {
  [event retain];
  pend((__bridge void *)event, value, false);
}
// A word of the trace's, published (1) by the GPU in `cb` after the work encoded so far (libnccl-mesh's
// publish kernel, ncclMeshEncodeCopies of no copies), noted in trace.reached once seen; its index (0: none).
static uint64_t note_word(id<MTLCommandBuffer> cb) {
  uint64_t index;
  {
    std::lock_guard<std::mutex> guard(trace.lock);
    if (!trace.words) {
      void *memory = nullptr;
      if (ncclMemAlloc(&memory, (size_t)1 << 20) != ncclSuccess) return 0;
      std::memset(memory, 0, (size_t)1 << 20);
      trace.words = (uint64_t *)memory;
    }
    if (trace.used + 1 >= ((size_t)1 << 17)) return 0;
    index = ++trace.used;
  }
  if (ncclMeshEncodeCopies((__bridge void *)cb, 0, nullptr, nullptr, nullptr, nullptr, trace.words + index, 1) != ncclSuccess) return 0;
  pend(trace.words + index, 1, true);
  return index;
}
// The GPU times of the command buffer `cb` (not yet committed), noted under the fence value `value`.
static void note_gpu(id<MTLCommandBuffer> cb, uint64_t value) {
  [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
    std::lock_guard<std::mutex> guard(trace.lock);
    trace.gpu[value] = {done.GPUStartTime, done.GPUEndTime};
  }];
}

// ---- the MPS stream's fence ----
// `body` on the MPS stream's serial queue (inline when already on it: a release closure can run there).
template <typename F> static void on_mps(at::mps::MPSStream *s, F body) {
  const char *here = dispatch_queue_get_label(DISPATCH_CURRENT_QUEUE_LABEL), *mps = dispatch_queue_get_label(s->queue());
  if (here && mps && !strcmp(here, mps)) body();
  else dispatch_sync(s->queue(), ^{ body(); });
}
static id<MTLSharedEvent> fence_event() {
  static id<MTLSharedEvent> event = [device() newSharedEvent];
  return event;
}
static std::atomic<uint64_t> fence_next{0};
// A signal of the fence event after every command enqueued on the MPS stream `s` so far (on its queue),
// not committed; its value.
static uint64_t mps_signal(at::mps::MPSStream *s) {
  s->endKernelCoalescing();
  const uint64_t value = ++fence_next;
  [s->commandBuffer() encodeSignalEvent:fence_event() value:value];
  return value;
}
// A signal of the fence event after every command enqueued on the current MPS stream so far, committed
// (the MPS kernels' plain stores reach the NIC once their command buffer completes); its value.
static uint64_t mps_fence() {
  auto *s = at::mps::getCurrentMPSStream();
  uint64_t value = 0;
  on_mps(s, [&] {
    value = mps_signal(s);
    if (tracing()) {
      note_gpu(s->commandBuffer(), value);
      note_reached(fence_event(), value);
    }
    s->synchronize(at::mps::SyncType::COMMIT);
  });
  commits++;
  fences++;
  return value;
}

// ---- MPS tensors in the window ----
// torch 2.14's MPS allocator (aten/src/ATen/mps/MPSAllocator.mm, MPSAllocator.h HeapBlock) makes its heaps
// with -[MTLDevice newHeapWithDescriptor:] (placement heaps of shared storage: its pools' hazard-tracked,
// the scalar pool's untracked) and places each buffer with -[MTLHeap newBufferWithLength:options:offset:];
// of a heap it otherwise asks only its size and purgeable state, and releases it.  window_heaps, run by
// the first group, gives the MPS device's class its own newHeapWithDescriptor:: a heap libtorch_cpu asks
// for that is a tracked, shared placement heap (the pools') is a WindowHeap, whose every buffer is a window
// allocation of its own (ncclMeshMemAllocBuffer: its one Metal buffer is the one the allocator owns, and
// the allocation is freed once that buffer is deallocated); where the window has no room (or would be
// three quarters the allocator's) the buffer is the device's own, a tensor the backend copies through the
// window.  Heaps made before the group stay Metal's.
} // namespace c10d
@interface MeshWindowHeap : NSObject {
 @public
  id<MTLDevice> device_;
  NSUInteger size_;
  MTLPurgeableState state_;
}
@end
@implementation MeshWindowHeap
- (NSUInteger)size { return size_; }
- (id<MTLDevice>)device { return device_; }
- (MTLPurgeableState)setPurgeableState:(MTLPurgeableState)state {
  const MTLPurgeableState was = state_;
  if (state != MTLPurgeableStateKeepCurrent) state_ = state;
  return was;
}
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options offset:(NSUInteger)offset {
  void *memory = nullptr, *buffer = nullptr;
  if (ncclMeshMemAllocBuffer(&memory, length, options, &buffer) == ncclSuccess) {
    c10d::window_buffers++;
    return (__bridge id<MTLBuffer>)buffer;
  }
  c10d::device_buffers++;
  return [device_ newBufferWithLength:length options:options];
}
@end
namespace c10d {
static IMP metal_heap = nullptr;
static id<MTLHeap> window_heap(id self, SEL selector, MTLHeapDescriptor *d) {
  Dl_info info;
  if (d.type == MTLHeapTypePlacement && d.storageMode == MTLStorageModeShared && d.hazardTrackingMode == MTLHazardTrackingModeTracked &&
      dladdr(__builtin_return_address(0), &info) && info.dli_fname && strstr(info.dli_fname, "libtorch_cpu")) {
    MeshWindowHeap *heap = [MeshWindowHeap new];
    heap->device_ = self;
    heap->size_ = d.size;
    heap->state_ = MTLPurgeableStateNonVolatile;
    window_heaps_made++;
    return (id<MTLHeap>)heap;
  }
  return ((id<MTLHeap>(*)(id, SEL, MTLHeapDescriptor *))metal_heap)(self, selector, d);
}
static void window_heaps() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (!at::mps::is_available()) return;
    id<MTLDevice> d = at::mps::MPSDevice::getInstance()->device();
    metal_heap = method_setImplementation(class_getInstanceMethod(object_getClass(d), @selector(newHeapWithDescriptor:)), (IMP)window_heap);
  });
}

// ---- window allocations ----
// A CPU window tensor's release closure: at once (the CPU's writes are done; the library's recorded
// points still hold its pages).
static void release_host(void *memory) { check(ncclMemFree(memory), nullptr, "ncclMemFree"); }
static id<MTLBuffer> buffer_of(void *memory, size_t *offset) {
  void *buffer = nullptr;
  check(ncclMeshMemBuffer(memory, &buffer, offset), nullptr, "ncclMeshMemBuffer");
  return (__bridge id<MTLBuffer>)buffer;
}
// A window tensor: an MPS tensor of the MPS allocator (its heaps window heaps), or a CPU tensor of a window
// allocation's bytes, its storage's deleter the allocation's release closure.
static at::Tensor window_tensor(at::IntArrayRef sizes, at::ScalarType dtype, bool mps) {
  auto options = at::TensorOptions().dtype(dtype);
  if (mps) return at::empty(sizes, options.device(at::kMPS));
  const size_t bytes = std::max<size_t>(1, (size_t)c10::multiply_integers(sizes) * c10::elementSize(dtype));
  void *memory = nullptr;
  check(ncclMemAlloc(&memory, bytes), nullptr, "ncclMemAlloc");
  return at::from_blob(memory, sizes, [memory](void *) { release_host(memory); }, options.device(at::kCPU));
}
static at::Tensor empty(std::vector<int64_t> sizes, const at::Tensor &like, const std::string &device) {
  TORCH_CHECK(device == "cpu" || device == "mps", "mesh: empty() makes cpu or mps tensors, not ", device);
  return window_tensor(sizes, like.scalar_type(), device == "mps");
}
// Where an MPS tensor's bytes are in the window: its MTLBuffer the Metal buffer of the allocation its
// contents lie in (nullptr: not a window tensor).
static char *window_bytes(const at::Tensor &t) {
  id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(t);
  char *contents = buffer ? (char *)[buffer contents] : nullptr;
  void *theirs = nullptr;
  size_t offset = 0;
  if (!contents || ncclMeshMemBuffer(contents, &theirs, &offset) != ncclSuccess || theirs != (__bridge void *)buffer) return nullptr;
  return contents + t.storage_offset() * t.element_size();
}

// Where MPS tensors laid end to end are in the window: the first's bytes where every one is contiguous
// and they are adjacent in one window allocation (an empty one lies anywhere), else nullptr.
static char *window_span(const std::vector<at::Tensor> &parts) {
  char *first = nullptr, *end = nullptr;
  id<MTLBuffer> buffer = nil;
  for (auto &t : parts) {
    if (!t.nbytes()) continue;
    char *at = t.is_mps() && t.is_contiguous() ? window_bytes(t) : nullptr;
    if (!at || (end && at != end) || (buffer && at::native::mps::getMTLBufferStorage(t) != buffer)) return nullptr;
    if (!first) { first = at; buffer = at::native::mps::getMTLBufferStorage(t); }
    end = at + t.nbytes();
  }
  return first;
}

// The backend's queue for CPU parts' blits, and its event.
static id<MTLCommandQueue> host_queue() {
  static id<MTLCommandQueue> queue = [device() newCommandQueue];
  return queue;
}
static id<MTLSharedEvent> host_event() {
  static id<MTLSharedEvent> event = [device() newSharedEvent];
  return event;
}
static std::atomic<uint64_t> host_next{0};
// A Metal buffer over the host pages holding `bytes` at `pointer`, no copy; `offset` the pointer's place.
static id<MTLBuffer> host_pages(void *pointer, size_t bytes, size_t *offset) {
  const uintptr_t page = (uintptr_t)getpagesize(), at = (uintptr_t)pointer, first = at & ~(page - 1),
                  end = (at + std::max<size_t>(bytes, 1) + page - 1) & ~(page - 1);
  *offset = at - first;
  id<MTLBuffer> buffer = [device() newBufferWithBytesNoCopy:(void *)first length:end - first options:MTLResourceStorageModeShared deallocator:nil];
  TORCH_CHECK(buffer, "mesh: the host pages of a CPU tensor as a Metal buffer (newBufferWithBytesNoCopy)");
  return buffer;
}

// Streams for calls that run asynchronously (a send or recv on CPU tensors, every call on MPS tensors):
// one a call, each with a queue of its own, reused once the Work holding it is gone and its work done.
// An MPS call's stream is deferred (ncclMeshStreamDefer): the library keeps the GPU work that follows its
// transfers (the combines, the stream's completion) and the Work's wait encodes it on the MPS stream.
struct StreamPool {
  std::mutex lock;
  std::vector<cudaStream_t> all, idle[2];  // idle[defer]: a stream's mode is set once, when it is made
  std::map<cudaStream_t, bool> deferred;
  cudaStream_t acquire(bool defer) {
    std::lock_guard<std::mutex> guard(lock);
    cudaStream_t s = nullptr;
    auto &free = idle[defer];
    for (size_t i = 0; i < free.size() && !s; i++)
      if (ncclMeshStreamQuery(free[i]) == ncclSuccess) {
        s = free[i];
        free.erase(free.begin() + (long)i);
      }
    if (!s) {
      check(ncclMeshStreamCreate(&s, nullptr), nullptr, "ncclMeshStreamCreate");
      check(ncclMeshStreamDefer(s, defer), nullptr, "ncclMeshStreamDefer");
      all.push_back(s);
      deferred[s] = defer;
    }
    return s;
  }
  void release(cudaStream_t s) {
    std::lock_guard<std::mutex> guard(lock);
    idle[deferred[s]].push_back(s);
  }
  void synchronize() {
    std::lock_guard<std::mutex> guard(lock);
    for (auto s : all) ncclMeshStreamSynchronize(s);
  }
  ~StreamPool() {
    for (auto s : all) { ncclMeshStreamSynchronize(s); ncclMeshStreamDestroy(s); }
  }
};

// One torch call's buffers where libnccl-mesh reads and writes them, and the stream it runs on.  A place
// is tensors laid end to end (one tensor, or a list): in place where they already are (CPU: contiguous
// and adjacent; MPS: contiguous and adjacent in one window allocation), else a slice of the call's
// window allocation, filled and emptied by GPU blits.  On MPS a place may also be filled from another
// tensor at an offset once the call's fence is committed (fill: an all-to-all's own rows, which the
// network neither reads nor writes, copied while its transfers run).
class Call {
 public:
  Call(const at::Tensor &like, bool async, std::shared_ptr<StreamPool> pool, const char *op = nullptr)
      : mps_(like.is_mps()), async_(async || like.is_mps()), pool_(std::move(pool)), op_(op), dtype_(like.scalar_type()),
        enter_(tracing() && op ? uptime() : 0) {}
  int add(const at::Tensor &t, bool in, bool out) { return add(std::vector<at::Tensor>{t}, in, out); }
  int add(std::vector<at::Tensor> parts, bool in, bool out) {
    Place p;
    p.parts = std::move(parts);
    p.in = in;
    p.out = out;
    for (auto &t : p.parts) p.bytes += t.nbytes();
    places_.push_back(std::move(p));
    return (int)places_.size() - 1;
  }
  void fill(const at::Tensor &from, int place, size_t at) { fills_.push_back({from, place, at}); }
  bool mps() const { return mps_; }
  void *ptr(int i) const { return places_[i].pointer; }
  cudaStream_t stream() const { return stream_; }

  // Every place's pointer: the call's window allocation for those not in place, and the blits in; on MPS
  // the fence after them declared the writer of every allocation the call touches, then the fills.
  void begin() {
    size_t used = 0;
    for (auto &p : places_)
      if (!(mps_ ? window_place(p) : host_place(p))) { p.at = used; used += (p.bytes + 255) & ~(size_t)255; }
    if (used) {
      check(ncclMemAlloc(&scratch_, used), nullptr, "ncclMemAlloc");
      size_t offset = 0;
      buffer_ = buffer_of(scratch_, &offset);
      scratch_bytes_ = used;
    }
    for (auto &p : places_)
      if (p.at != SIZE_MAX) p.pointer = (char *)scratch_ + p.at;
    if (!mps_) {
      host_blits(true);
      if (async_) stream_ = pool_->acquire(false);
      return;
    }
    auto *s = at::mps::getCurrentMPSStream();
    for (auto &p : places_)
      if (p.at != SIZE_MAX && p.in) mps_blits(s, p, true);
    const uint64_t fence = fence_ = mps_fence();
    commits_++;
    for (auto &p : places_)
      if (p.bytes) check(ncclMeshMemUse(p.pointer, p.bytes, (__bridge void *)fence_event(), fence, 1), nullptr, "ncclMeshMemUse");
    for (auto &f : fills_) mps_fill(s, f);
    stream_ = pool_->acquire(true);
  }
  // Once the call is complete (CPU) or its completion is ordered on the current MPS stream (the stream's
  // kept GPU work encoded into its command buffer, not committed: ncclMeshStreamEncodeWait): the places
  // not in place copied back, and the call's window allocation released after them (at a signal encoded
  // after them, reached whenever that command buffer runs).  A call that issued nothing has nothing to
  // order.
  void after() {
    if (!mps_) {
      host_blits(false);
      return;
    }
    if (!stream_) return;
    auto *s = at::mps::getCurrentMPSStream();
    on_mps(s, [&] {
      s->endKernelCoalescing();
      if (traced_ != SIZE_MAX) {  // words published before and after the kept work: when the MPS stream reached and passed it
        const uint64_t reach = note_word(s->commandBuffer());
        std::lock_guard<std::mutex> guard(trace.lock);
        trace.calls[traced_].reach = reach;
      }
      check(ncclMeshStreamEncodeWait(stream_, (__bridge void *)s->commandBuffer()), nullptr, "ncclMeshStreamEncodeWait");
      if (traced_ != SIZE_MAX) {
        const uint64_t resume = note_word(s->commandBuffer());
        std::lock_guard<std::mutex> guard(trace.lock);
        trace.calls[traced_].resume = resume;
      }
    });
    for (auto &p : places_)
      if (p.at != SIZE_MAX && p.out) mps_blits(s, p, false);
    if (!scratch_) return;
    uint64_t value = 0;
    on_mps(s, [&] { value = mps_signal(s); });
    check(ncclMeshMemRelease(scratch_, (__bridge void *)fence_event(), value), nullptr, "ncclMeshMemRelease");
    scratch_ = nullptr;
  }
  ~Call() {
    if (scratch_) ncclMeshMemRelease(scratch_, nullptr, 0);
  }
  // The call's trace record once it is issued (MESH_TRACE): its places' bytes, in place or blitted, its
  // host times, its fence, and the library's tallies of its group, held.
  void traced(std::vector<int64_t> splits = {}) {
    if (!enter_) return;
    Traced t;
    t.op = op_;
    t.dtype = c10::toString(dtype_);
    t.mps = mps_;
    for (auto &p : places_) {
      t.bytes += p.bytes;
      if (p.at == SIZE_MAX) t.in_place += p.bytes;
      else {
        if (p.in) t.blit_in += p.bytes;
        if (p.out) t.blit_out += p.bytes;
      }
    }
    for (auto &f : fills_) t.blit_in += f.tensor.nbytes();
    t.splits = std::move(splits);
    t.enter = enter_;
    t.issued = uptime();
    t.fence = fence_;
    t.commits = commits_;
    ncclMeshGroupTally(&t.tally);
    if (stream_) {
      t.stream_event = (__bridge void *)[(__bridge id<MTLSharedEvent>)stream_->event retain];  // read at the dump, past the stream
      t.stream_value = stream_->value;
      note_reached((__bridge id<MTLSharedEvent>)stream_->event, stream_->value);
    }
    std::lock_guard<std::mutex> guard(trace.lock);
    traced_ = trace.calls.size();
    trace.calls.push_back(std::move(t));
  }

 private:
  struct Place {
    std::vector<at::Tensor> parts;
    bool in = false, out = false;
    size_t bytes = 0, at = SIZE_MAX;
    void *pointer = nullptr;
  };
  struct Segment {
    at::Tensor tensor;
    int place;
    size_t at;
  };
  // CPU: the parts themselves where they are contiguous and adjacent (an empty part lies anywhere).
  bool host_place(Place &p) {
    char *first = nullptr, *end = nullptr;
    for (auto &t : p.parts) {
      if (!t.nbytes()) continue;
      if (!t.is_contiguous() || (end && (char *)t.data_ptr() != end)) return false;
      if (!first) first = (char *)t.data_ptr();
      end = (char *)t.data_ptr() + t.nbytes();
    }
    p.pointer = first;
    return true;
  }
  // MPS: the parts in place where they are contiguous and adjacent in one window allocation (an empty
  // part lies anywhere).
  bool window_place(Place &p) {
    p.pointer = window_span(p.parts);
    return p.pointer != nullptr;
  }
  // Each MPS part's bytes between its MTLBuffer and the call's window allocation, blitted on the MPS
  // stream (a non-contiguous part through a contiguous MPS copy).
  void mps_blits(at::mps::MPSStream *s, Place &p, bool in) {
    size_t at = p.at;
    for (auto &t : p.parts) {
      if (!t.nbytes()) continue;
      at::Tensor c = t.is_contiguous() ? t : in ? t.contiguous() : at::empty(t.sizes(), t.options());
      if (in && !c.is_same(t)) gpu_copied += c.nbytes();
      id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(c);
      size_t offset = c.storage_offset() * c.element_size();
      if (in) s->copy(buffer, buffer_, c.nbytes(), offset, at, 0, at::mps::SyncType::NONE);
      else s->copy(buffer_, buffer, c.nbytes(), at, offset, 0, at::mps::SyncType::NONE);
      if (!in && !c.is_same(t)) { t.copy_(c); gpu_copied += c.nbytes(); }
      gpu_copied += c.nbytes();
      at += c.nbytes();
    }
  }
  // A fill: a tensor's bytes into a place's at an offset, blitted on the MPS stream.
  void mps_fill(at::mps::MPSStream *s, Segment &g) {
    at::Tensor &t = g.tensor;
    if (!t.nbytes()) return;
    at::Tensor c = t.is_contiguous() ? t : t.contiguous();
    void *buffer = nullptr;
    size_t offset = 0;
    check(ncclMeshMemBuffer((char *)places_[g.place].pointer + g.at, &buffer, &offset), nullptr, "ncclMeshMemBuffer");
    s->copy(at::native::mps::getMTLBufferStorage(c), (__bridge id<MTLBuffer>)buffer, c.nbytes(), c.storage_offset() * c.element_size(), offset, 0,
            at::mps::SyncType::NONE);
    gpu_copied += c.nbytes();
  }
  // Each CPU part's bytes between its pages and the call's window allocation, blitted on the backend's
  // queue through a Metal buffer over its pages (a non-contiguous part through a contiguous CPU copy):
  // in, declared the writer of the allocation; out, after the call's recorded points, waited for on the
  // host, then the allocation released.
  void host_blits(bool in) {
    if (!scratch_) return;
    @autoreleasepool {
      host_blits_(in);
    }
  }
  void host_blits_(bool in) {
    id<MTLCommandBuffer> cb = [host_queue() commandBuffer];
    std::vector<id<MTLBuffer>> wrapped;
    if (!in) {
      void *events[16];
      uint64_t values[16];
      int count = 0;
      check(ncclMeshMemWaits(scratch_, scratch_bytes_, 1, events, values, 16, &count), nullptr, "ncclMeshMemWaits");
      for (int i = 0; i < count && i < 16; i++) [cb encodeWaitForEvent:(__bridge id<MTLSharedEvent>)events[i] value:values[i]];
    }
    std::vector<std::pair<at::Tensor, at::Tensor>> back;
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    for (auto &p : places_) {
      if (p.at == SIZE_MAX || !(in ? p.in : p.out)) continue;
      size_t at = p.at;
      for (auto &t : p.parts) {
        if (!t.nbytes()) continue;
        at::Tensor c = t;
        if (!t.is_contiguous()) {
          c = in ? t.contiguous() : at::empty(t.sizes(), t.options());
          if (in) cpu_copied += c.nbytes();
          else back.emplace_back(t, c);
        }
        size_t offset = 0;
        id<MTLBuffer> pages = host_pages(c.data_ptr(), c.nbytes(), &offset);
        wrapped.push_back(pages);
        if (c.nbytes()) {
          if (in) [blit copyFromBuffer:pages sourceOffset:offset toBuffer:buffer_ destinationOffset:at size:c.nbytes()];
          else [blit copyFromBuffer:buffer_ sourceOffset:at toBuffer:pages destinationOffset:offset size:c.nbytes()];
        }
        gpu_copied += c.nbytes();
        at += c.nbytes();
      }
    }
    [blit endEncoding];
    const uint64_t value = ++host_next;
    [cb encodeSignalEvent:host_event() value:value];
    [cb commit];
    for (id<MTLBuffer> pages : wrapped) [pages release];  // the command buffer holds them until it has run
    if (in) {
      check(ncclMeshMemUse(scratch_, scratch_bytes_, (__bridge void *)host_event(), value, 1), nullptr, "ncclMeshMemUse");
      return;
    }
    [cb waitUntilCompleted];
    for (auto &b : back) { b.first.copy_(b.second); cpu_copied += b.second.nbytes(); }
    check(ncclMeshMemRelease(scratch_, (__bridge void *)host_event(), value), nullptr, "ncclMeshMemRelease");
    scratch_ = nullptr;
  }
  bool mps_, async_;
  std::shared_ptr<StreamPool> pool_;
  const char *op_;
  at::ScalarType dtype_;
  uint64_t enter_, fence_ = 0, commits_ = 0;
  size_t traced_ = SIZE_MAX;
  cudaStream_t stream_ = nullptr;
  std::vector<Place> places_;
  std::vector<Segment> fills_;
  void *scratch_ = nullptr;
  size_t scratch_bytes_ = 0;
  id<MTLBuffer> buffer_ = nil;
  friend class WorkMesh;
};

// A call's Work: done at once for a CPU collective; else done once its stream reaches the call's value,
// wait() waiting on the host (CPU) or ordering the current MPS stream after it (MPS), then copying back,
// and raising (once every rank has agreed on it) an earlier call's failure the communicator holds.
class WorkMesh : public Work {
 public:
  WorkMesh(OpType type, std::vector<at::Tensor> outputs, std::shared_ptr<Call> call, ncclComm_t comm)
      : Work(-1, type), outputs_(std::move(outputs)), call_(std::move(call)), comm_(comm),
        future_(c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()))) {
    if (!call_->stream()) finish();
  }
  ~WorkMesh() override {
    try {
      if (call_) finish();
    } catch (const std::exception &) {
    }
  }
  bool isCompleted() override { return !call_ || ncclMeshStreamQuery(call_->stream()) == ncclSuccess; }
  bool isSuccess() const override { return true; }
  bool wait(std::chrono::milliseconds) override {
    if (call_) finish();
    ncclResult_t failed = ncclSuccess;
    if (comm_) ncclCommGetAsyncError(comm_, &failed);
    check(failed, comm_, "a call issued earlier");
    return true;
  }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override { return future_; }

 private:
  void finish() {
    cudaStream_t stream = call_->stream();
    if (stream && !call_->mps_) check(ncclMeshStreamSynchronize(stream), nullptr, "the stream");
    call_->after();
    if (stream) call_->pool_->release(stream);
    call_.reset();
    future_->markCompleted(c10::IValue(outputs_));
  }
  std::vector<at::Tensor> outputs_;
  std::shared_ptr<Call> call_;
  ncclComm_t comm_;
  c10::intrusive_ptr<c10::ivalue::Future> future_;
};

// The communicators of the live groups in the order they were made (the default group's first).
static std::mutex comms_lock;
static std::vector<ncclComm_t> comms;
// The last agreement this process made (the error path's or agree()'s): how many so far, the first failed
// call it found (0: none) and this rank's link-table epoch.
static struct {
  std::mutex lock;
  uint64_t count = 0, failed = 0, epoch = 0;
} agreement;
static std::pair<uint64_t, uint64_t> agree_on(ncclComm_t comm, const char *after) {
  uint64_t failed = 0, epoch = 0;
  const ncclResult_t result = ncclMeshCommAgree(comm, &failed, &epoch);
  TORCH_CHECK(result == ncclSuccess, "mesh: ncclMeshCommAgree", after, ": ", ncclGetErrorString(result), ": ", ncclGetLastError(comm));
  std::lock_guard<std::mutex> guard(agreement.lock);
  agreement.count++;
  agreement.failed = failed;
  agreement.epoch = epoch;
  return {failed, epoch};
}

class ProcessGroupMesh : public Backend {
 public:
  ProcessGroupMesh(const c10::intrusive_ptr<Store> &store, int rank, int size, const std::string &region, const std::string &links,
                   const std::vector<int> &nodes)
      : Backend(rank, size) {
    TORCH_CHECK((int)nodes.size() == size, "mesh: the options name ", nodes.size(), " nodes for ", size, " ranks");
    check(ncclMeshLinksAttach(region.empty() ? nullptr : region.c_str(), &links_), nullptr, "ncclMeshLinksAttach");
    if (!links.empty()) check(ncclMeshLinksState(links_, links.c_str()), nullptr, "ncclMeshLinksState");
    ncclUniqueId id;
    const std::string key = "mesh_nccl_unique_id";
    if (rank == 0) {
      check(ncclGetUniqueId(&id), nullptr, "ncclGetUniqueId");
      store->set(key, std::vector<uint8_t>(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES));
    } else {
      auto bytes = store->get(key);
      TORCH_CHECK(bytes.size() == NCCL_UNIQUE_ID_BYTES, "mesh: the unique id in the store has ", bytes.size(), " bytes");
      std::memcpy(id.internal, bytes.data(), NCCL_UNIQUE_ID_BYTES);
    }
    ncclMeshConfig_t config = NCCL_MESH_CONFIG_INITIALIZER;
    config.links = links_;
    config.nodes = nodes.data();
    check(ncclCommInitRankConfig(&comm_, size, id, rank, &config.base), nullptr, "ncclCommInitRankConfig");
    window_heaps();
    std::lock_guard<std::mutex> guard(comms_lock);
    comms.push_back(comm_);
  }
  ~ProcessGroupMesh() override {
    streams_->synchronize();
    {
      std::lock_guard<std::mutex> guard(comms_lock);
      comms.erase(std::remove(comms.begin(), comms.end(), comm_), comms.end());
    }
    if (comm_) ncclCommDestroy(comm_);
    if (links_) ncclMeshLinksDetach(links_);
    trace_dump();
  }
  const std::string getBackendName() const override { return "mesh"; }

  static c10::intrusive_ptr<Backend> create(const c10::intrusive_ptr<Store> &store, int rank, int size, const std::chrono::duration<float> &,
                                            const std::string &region, const std::string &links, const std::vector<int> &nodes) {
    return c10::make_intrusive<ProcessGroupMesh>(store, rank, size, region, links, nodes);
  }

  c10::intrusive_ptr<Work> broadcast(std::vector<at::Tensor> &tensors, const BroadcastOptions &opts) override {
    auto call = start(tensors[0], "broadcast");
    std::vector<int> at;
    for (auto &t : tensors) at.push_back(call->add(t, getRank() == opts.rootRank, true));
    call->begin();
    group([&] {
      for (size_t i = 0; i < tensors.size(); i++)
        check(ncclBroadcast(call->ptr(at[i]), call->ptr(at[i]), tensors[i].numel(), datatype(tensors[i]), (int)opts.rootRank, comm_,
                            call->stream()), comm_, "ncclBroadcast");
    });
    return work(OpType::BROADCAST, tensors, call);
  }

  c10::intrusive_ptr<Work> allreduce(std::vector<at::Tensor> &tensors, const AllreduceOptions &opts) override {
    auto call = start(tensors[0], "all_reduce");
    std::vector<int> at;
    for (auto &t : tensors) at.push_back(call->add(t, true, true));
    call->begin();
    for (size_t i = 0; i < tensors.size(); i++) reduce_call("ncclAllReduce", opts.reduceOp, tensors[i], [&](ncclRedOp_t op) {
      return ncclAllReduce(call->ptr(at[i]), call->ptr(at[i]), tensors[i].numel(), datatype(tensors[i]), op, comm_, call->stream());
    });
    return work(OpType::ALLREDUCE, tensors, call);
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(std::vector<at::Tensor> &tensors, const AllreduceCoalescedOptions &opts) override {
    return allreduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> reduce(std::vector<at::Tensor> &tensors, const ReduceOptions &opts) override {
    auto call = start(tensors[0], "reduce");
    std::vector<int> at;
    for (auto &t : tensors) at.push_back(call->add(t, true, getRank() == opts.rootRank));
    call->begin();
    for (size_t i = 0; i < tensors.size(); i++) reduce_call("ncclReduce", opts.reduceOp, tensors[i], [&](ncclRedOp_t op) {
      return ncclReduce(call->ptr(at[i]), call->ptr(at[i]), tensors[i].numel(), datatype(tensors[i]), op, (int)opts.rootRank, comm_,
                        call->stream());
    });
    return work(OpType::REDUCE, tensors, call);
  }

  // The list's tensors are the ranks' counts (MPI_Allgatherv): equal, ncclAllGather; else the planner's cut.
  // In place where the input already is its segment of the output, else out of place (the library reads
  // the input in place and copies it into its segment once the transfers are under way).
  c10::intrusive_ptr<Work> allgather(std::vector<std::vector<at::Tensor>> &outputs, std::vector<at::Tensor> &inputs,
                                     const AllgatherOptions &) override {
    TORCH_CHECK(inputs.size() == 1 && outputs.size() == 1 && (int)outputs[0].size() == getSize(), "mesh: allgather takes one tensor and a list of world_size");
    std::vector<size_t> counts = numels(outputs[0]);
    TORCH_CHECK(inputs[0].numel() == (int64_t)counts[getRank()], "mesh: all_gather's input is this rank's tensor of the list: ", inputs[0].numel(),
                " elements, the list's ", counts[getRank()]);
    auto call = start(inputs[0], "all_gather");
    size_t at = 0;
    for (int r = 0; r < getRank(); r++) at += counts[r] * inputs[0].element_size();
    int in = -1, out = call->add(outputs[0], false, true);
    if (!(call->mps() && lies_at(inputs[0], outputs[0], at))) in = call->add(inputs[0], true, false);
    call->begin();
    check(ncclMeshAllGatherV(in < 0 ? (char *)call->ptr(out) + at : call->ptr(in), call->ptr(out), counts.data(), datatype(inputs[0]), comm_,
                             call->stream()), comm_, "ncclMeshAllGatherV");
    return work(OpType::ALLGATHER, outputs[0], call, uneven(counts));
  }

  c10::intrusive_ptr<Work> all_gather_single(at::Tensor &output, at::Tensor &input, const AllgatherOptions &) override {
    TORCH_CHECK(output.numel() == getSize() * input.numel(), "mesh: all_gather_into_tensor's output is world_size inputs");
    auto call = start(input, "all_gather_into_tensor");
    const int in = gather_into(*call, output, input);
    call->begin();
    gather_issue(*call, 0, in, input);
    std::vector<at::Tensor> result{output};
    return work(OpType::_ALLGATHER_BASE, result, call);
  }

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs,
                                                           const AllgatherOptions &) override {
    TORCH_CHECK(!inputs.empty() && inputs.size() == outputs.size(), "mesh: allgather_into_tensor_coalesced takes as many outputs as inputs");
    for (size_t i = 0; i < inputs.size(); i++)
      TORCH_CHECK(outputs[i].numel() == getSize() * inputs[i].numel(), "mesh: all_gather_into_tensor's output is world_size inputs");
    auto call = start(inputs[0], "all_gather_into_tensor_coalesced");
    std::vector<int> outs, ins;
    for (size_t i = 0; i < inputs.size(); i++) {
      outs.push_back(call->add(outputs[i], false, true));
      ins.push_back(call->mps() && lies_at(inputs[i], {outputs[i]}, getRank() * inputs[i].nbytes()) ? -1 : call->add(inputs[i], true, false));
    }
    call->begin();
    group([&] {
      for (size_t i = 0; i < inputs.size(); i++) gather_issue(*call, outs[i], ins[i], inputs[i]);
    });
    return work(OpType::COALESCED, outputs, call);
  }

  // The root's list is the ranks' counts (MPI_Gatherv, MPI_Scatterv): equal, ncclGather or ncclScatter; else
  // grouped ncclSend/ncclRecv of each rank's own, as libnccl-mesh lowers those two.
  c10::intrusive_ptr<Work> gather(std::vector<std::vector<at::Tensor>> &outputs, std::vector<at::Tensor> &inputs,
                                  const GatherOptions &opts) override {
    TORCH_CHECK(inputs.size() == 1, "mesh: gather takes one tensor");
    const bool root = getRank() == opts.rootRank;
    TORCH_CHECK(!root || (outputs.size() == 1 && (int)outputs[0].size() == getSize()), "mesh: gather's root takes a list of world_size");
    std::vector<size_t> counts = root ? numels(outputs[0]) : std::vector<size_t>{};
    TORCH_CHECK(!root || inputs[0].numel() == (int64_t)counts[getRank()], "mesh: gather's input is the root's tensor of its list");
    auto call = start(inputs[0], "gather");
    int in = call->add(inputs[0], true, false), out = root ? call->add(outputs[0], false, true) : -1;
    call->begin();
    const ncclDataType_t type = datatype(inputs[0]);
    if (!root || uneven(counts).empty())
      check(ncclGather(call->ptr(in), root ? call->ptr(out) : nullptr, inputs[0].numel(), type, (int)opts.rootRank, comm_, call->stream()), comm_,
            "ncclGather");
    else
      group([&] {
        check(ncclSend(call->ptr(in), inputs[0].numel(), type, (int)opts.rootRank, comm_, call->stream()), comm_, "ncclSend");
        size_t at = 0;
        for (int r = 0; r < getSize(); r++) {
          check(ncclRecv((char *)call->ptr(out) + at, counts[r], type, r, comm_, call->stream()), comm_, "ncclRecv");
          at += counts[r] * inputs[0].element_size();
        }
      });
    std::vector<at::Tensor> result = root ? outputs[0] : std::vector<at::Tensor>{};
    return work(OpType::GATHER, result, call, uneven(counts));
  }

  c10::intrusive_ptr<Work> scatter(std::vector<at::Tensor> &outputs, std::vector<std::vector<at::Tensor>> &inputs,
                                   const ScatterOptions &opts) override {
    TORCH_CHECK(outputs.size() == 1, "mesh: scatter takes one tensor");
    const bool root = getRank() == opts.rootRank;
    TORCH_CHECK(!root || (inputs.size() == 1 && (int)inputs[0].size() == getSize()), "mesh: scatter's root takes a list of world_size");
    std::vector<size_t> counts = root ? numels(inputs[0]) : std::vector<size_t>{};
    TORCH_CHECK(!root || outputs[0].numel() == (int64_t)counts[getRank()], "mesh: scatter's output is the root's tensor of its list");
    auto call = start(outputs[0], "scatter");
    int out = call->add(outputs[0], false, true), in = root ? call->add(inputs[0], true, false) : -1;
    call->begin();
    const ncclDataType_t type = datatype(outputs[0]);
    if (!root || uneven(counts).empty())
      check(ncclScatter(root ? call->ptr(in) : nullptr, call->ptr(out), outputs[0].numel(), type, (int)opts.rootRank, comm_, call->stream()), comm_,
            "ncclScatter");
    else
      group([&] {
        size_t at = 0;
        for (int r = 0; r < getSize(); r++) {
          check(ncclSend((char *)call->ptr(in) + at, counts[r], type, r, comm_, call->stream()), comm_, "ncclSend");
          at += counts[r] * outputs[0].element_size();
        }
        check(ncclRecv(call->ptr(out), outputs[0].numel(), type, (int)opts.rootRank, comm_, call->stream()), comm_, "ncclRecv");
      });
    return work(OpType::SCATTER, outputs, call, uneven(counts));
  }

  // The list's tensors are the ranks' counts (MPI_Reduce_scatter): equal, ncclReduceScatter; else the planner's cut.
  // In place on the input where the output is its rank's segment in the window, else out of place (the
  // library reads the input in place and leaves it unchanged).
  c10::intrusive_ptr<Work> reduce_scatter(std::vector<at::Tensor> &outputs, std::vector<std::vector<at::Tensor>> &inputs,
                                          const ReduceScatterOptions &opts) override {
    TORCH_CHECK(outputs.size() == 1 && inputs.size() == 1 && (int)inputs[0].size() == getSize(), "mesh: reduce_scatter takes one tensor and a list of world_size");
    std::vector<size_t> counts = numels(inputs[0]);
    TORCH_CHECK(outputs[0].numel() == (int64_t)counts[getRank()], "mesh: reduce_scatter's output is this rank's tensor of the list: ",
                outputs[0].numel(), " elements, the list's ", counts[getRank()]);
    auto call = start(outputs[0], "reduce_scatter");
    size_t at = 0;
    for (int r = 0; r < getRank(); r++) at += counts[r] * outputs[0].element_size();
    int out = -1, in = scatter_from(*call, outputs[0], inputs[0], at, &out);
    call->begin();
    reduce_call("ncclMeshReduceScatterV", opts.reduceOp, outputs[0], [&](ncclRedOp_t op) {
      return ncclMeshReduceScatterV(call->ptr(in), out < 0 ? (char *)call->ptr(in) + at : call->ptr(out), counts.data(), datatype(outputs[0]), op,
                                    comm_, call->stream());
    });
    return work(OpType::REDUCE_SCATTER, outputs, call, uneven(counts));
  }

  c10::intrusive_ptr<Work> reduce_scatter_single(at::Tensor &output, at::Tensor &input, const ReduceScatterOptions &opts) override {
    TORCH_CHECK(input.numel() == getSize() * output.numel(), "mesh: reduce_scatter_tensor's input is world_size outputs");
    auto call = start(output, "reduce_scatter_tensor");
    const size_t at = getRank() * output.nbytes();
    int out = -1, in = scatter_from(*call, output, {input}, at, &out);
    call->begin();
    reduce_call("ncclReduceScatter", opts.reduceOp, output, [&](ncclRedOp_t op) {
      return ncclReduceScatter(call->ptr(in), out < 0 ? (char *)call->ptr(in) + at : call->ptr(out), output.numel(), datatype(output), op, comm_,
                               call->stream());
    });
    std::vector<at::Tensor> result{output};
    return work(OpType::_REDUCE_SCATTER_BASE, result, call);
  }

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs,
                                                           const ReduceScatterOptions &opts) override {
    TORCH_CHECK(!inputs.empty() && inputs.size() == outputs.size(), "mesh: reduce_scatter_tensor_coalesced takes as many outputs as inputs");
    auto call = start(outputs[0], "reduce_scatter_tensor_coalesced");
    std::vector<int> ins, outs(inputs.size(), -1);
    for (size_t i = 0; i < inputs.size(); i++) {
      TORCH_CHECK(inputs[i].numel() == getSize() * outputs[i].numel(), "mesh: reduce_scatter_tensor's input is world_size outputs");
      ins.push_back(scatter_from(*call, outputs[i], {inputs[i]}, getRank() * outputs[i].nbytes(), &outs[i]));
    }
    call->begin();
    group([&] {
      for (size_t i = 0; i < inputs.size(); i++) reduce_call("ncclReduceScatter", opts.reduceOp, outputs[i], [&](ncclRedOp_t op) {
        const size_t at = getRank() * outputs[i].nbytes();
        return ncclReduceScatter(call->ptr(ins[i]), outs[i] < 0 ? (char *)call->ptr(ins[i]) + at : call->ptr(outs[i]), outputs[i].numel(),
                                 datatype(outputs[i]), op, comm_, call->stream());
      });
    });
    return work(OpType::COALESCED, outputs, call);
  }

  // Equal splits are ncclAlltoAll; others, grouped ncclSend/ncclRecv of each rank's rows.  On MPS this
  // rank's own rows are copied into their place in the output on the MPS stream after the call's fence
  // (a fill, while the transfers run) and only the other ranks' travel, so the library's group has no
  // copy before its transfers.
  c10::intrusive_ptr<Work> all_to_all_single(at::Tensor &output, at::Tensor &input, std::vector<int64_t> &outputSplits,
                                             std::vector<int64_t> &inputSplits, const AllToAllOptions &) override {
    const int n = getSize(), me = getRank();
    auto call = start(input, "all_to_all_single");
    const int64_t inRow = input.dim() ? input.numel() / std::max<int64_t>(input.size(0), 1) : 1;
    const int64_t outRow = output.dim() ? output.numel() / std::max<int64_t>(output.size(0), 1) : 1;
    auto split_of = [&](std::vector<int64_t> s, const at::Tensor &t) {
      if (s.empty()) s.assign(n, t.dim() ? t.size(0) / n : 0);
      return s;
    };
    std::vector<int64_t> ins = split_of(inputSplits, input), outs = split_of(outputSplits, output);
    int64_t mineIn = 0, mineOut = 0;
    for (int r = 0; r < me; r++) { mineIn += ins[r]; mineOut += outs[r]; }
    const bool own = call->mps() && input.dim() && ins[me] * inRow == outs[me] * outRow;
    int in = call->add(input, true, false), out = call->add(output, false, true);
    if (own && ins[me]) call->fill(input.narrow(0, mineIn, ins[me]), out, (size_t)(mineOut * outRow) * output.element_size());
    call->begin();
    if (outputSplits.empty() && inputSplits.empty() && !own) {
      TORCH_CHECK(input.numel() % n == 0 && input.numel() == output.numel(), "mesh: all_to_all_single's tensors split evenly");
      check(ncclAlltoAll(call->ptr(in), call->ptr(out), input.numel() / n, datatype(input), comm_, call->stream()), comm_, "ncclAlltoAll");
    } else {
      const size_t element = input.element_size();
      group([&] {
        int64_t at = 0;
        for (int r = 0; r < n; r++) {
          if (!(own && r == me))
            check(ncclSend((char *)call->ptr(in) + at * inRow * element, ins[r] * inRow, datatype(input), r, comm_, call->stream()), comm_, "ncclSend");
          at += ins[r];
        }
        at = 0;
        for (int r = 0; r < n; r++) {
          if (!(own && r == me))
            check(ncclRecv((char *)call->ptr(out) + at * outRow * element, outs[r] * outRow, datatype(output), r, comm_, call->stream()), comm_, "ncclRecv");
          at += outs[r];
        }
      });
    }
    std::vector<int64_t> splits(inputSplits);
    splits.insert(splits.end(), outputSplits.begin(), outputSplits.end());
    std::vector<at::Tensor> result{output};
    return work(OpType::ALLTOALL_BASE, result, call, splits);
  }

  c10::intrusive_ptr<Work> alltoall(std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs, const AllToAllOptions &) override {
    TORCH_CHECK((int)outputs.size() == getSize() && (int)inputs.size() == getSize(), "mesh: alltoall takes lists of world_size");
    const int me = getRank();
    auto call = start(inputs[0], "all_to_all");
    const bool own = call->mps() && inputs[me].nbytes() == outputs[me].nbytes();  // this rank's own tensor: a fill (all_to_all_single)
    std::vector<int> ins, outs;
    for (int r = 0; r < getSize(); r++) {
      ins.push_back(call->add(inputs[r], true, false));
      outs.push_back(call->add(outputs[r], false, true));
    }
    if (own && inputs[me].nbytes()) call->fill(inputs[me], outs[me], 0);
    call->begin();
    group([&] {
      for (int r = 0; r < getSize(); r++) {
        if (own && r == me) continue;
        check(ncclSend(call->ptr(ins[r]), inputs[r].numel(), datatype(inputs[r]), r, comm_, call->stream()), comm_, "ncclSend");
        check(ncclRecv(call->ptr(outs[r]), outputs[r].numel(), datatype(outputs[r]), r, comm_, call->stream()), comm_, "ncclRecv");
      }
    });
    return work(OpType::ALLTOALL, outputs, call);
  }

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor> &tensors, int dst, int) override {
    auto call = std::make_shared<Call>(tensors[0], true, streams_, "send");
    std::vector<int> at;
    for (auto &t : tensors) at.push_back(call->add(t, true, false));
    call->begin();
    for (size_t i = 0; i < tensors.size(); i++)
      check(ncclSend(call->ptr(at[i]), tensors[i].numel(), datatype(tensors[i]), dst, comm_, call->stream()), comm_, "ncclSend");
    return work(OpType::SEND, tensors, call);
  }

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor> &tensors, int src, int) override {
    auto call = std::make_shared<Call>(tensors[0], true, streams_, "recv");
    std::vector<int> at;
    for (auto &t : tensors) at.push_back(call->add(t, false, true));
    call->begin();
    for (size_t i = 0; i < tensors.size(); i++)
      check(ncclRecv(call->ptr(at[i]), tensors[i].numel(), datatype(tensors[i]), src, comm_, call->stream()), comm_, "ncclRecv");
    return work(OpType::RECV, tensors, call);
  }

  // Out of place, for torch's functional collectives (below): `out` a fresh tensor, `input` read in place
  // and left as it is.
  c10::intrusive_ptr<Work> allreduce_into(std::vector<at::Tensor> &outs, const std::vector<at::Tensor> &inputs, const ReduceOp &reduceOp) {
    auto call = start(inputs[0], outs.size() > 1 ? "all_reduce_coalesced" : "all_reduce");
    std::vector<int> ins, os;
    for (size_t i = 0; i < inputs.size(); i++) {
      ins.push_back(call->add(inputs[i], true, false));
      os.push_back(call->add(outs[i], false, true));
    }
    call->begin();
    group([&] {
      for (size_t i = 0; i < inputs.size(); i++) reduce_call("ncclAllReduce", reduceOp, outs[i], [&](ncclRedOp_t op) {
        return ncclAllReduce(call->ptr(ins[i]), call->ptr(os[i]), outs[i].numel(), datatype(outs[i]), op, comm_, call->stream());
      });
    });
    return work(OpType::ALLREDUCE, outs, call);
  }
  // MPI_Alltoallv with the GPU's counts (torch_mesh.all_to_all_counted, ncclMeshAlltoAllCounted): `input`'s rows
  // grouped by destination; `counts` int64, sends[q] entries for rank q (an MPS tensor, or a CPU window tensor);
  // `out` [capacity, ...]; `host` int64, a CPU window tensor (the library's records hold it to the call's end), its
  // first element the arrival word and the rest each rank's segment for this rank, receives[q] entries, written by
  // the library's worker as they land; `landed` the same counts on MPS, written by the GPU (for GPU work after the
  // call's wait).
  c10::intrusive_ptr<Work> all_to_all_counted(at::Tensor &out, const at::Tensor &input, const at::Tensor &counts, const std::vector<size_t> &sends,
                                              const std::vector<size_t> &receives, at::Tensor &host, at::Tensor &landed) {
    auto call = start(input, "all_to_all_counted");
    const int in = call->add(input, true, false), o = call->add(out, false, true), c = counts.is_mps() ? call->add(counts, true, false) : -1,
              l = call->add(landed, false, true);
    call->begin();
    const int64_t rows = input.dim() ? input.size(0) : 1, row = rows ? input.numel() / std::max<int64_t>(rows, 1) : 1;
    uint64_t *arrived = (uint64_t *)host.data_ptr<int64_t>();
    check(ncclMeshAlltoAllCounted(call->ptr(in), (size_t)rows, (const int64_t *)(c < 0 ? counts.data_ptr() : call->ptr(c)), sends.data(), call->ptr(o),
                                  (size_t)(out.dim() ? out.size(0) : 1), host.data_ptr<int64_t>() + 1, (int64_t *)call->ptr(l), receives.data(), arrived, (size_t)row,
                                  datatype(input), comm_, call->stream()),
          comm_, "ncclMeshAlltoAllCounted");
    std::vector<at::Tensor> result{out, landed};
    return work(OpType::ALLTOALL_BASE, result, call);
  }
  c10::intrusive_ptr<Work> broadcast_into(at::Tensor &out, const at::Tensor &input, int root) {
    auto call = start(input, "broadcast");
    const int in = getRank() == root ? call->add(input, true, false) : -1, o = call->add(out, false, true);
    call->begin();
    check(ncclBroadcast(in < 0 ? nullptr : call->ptr(in), call->ptr(o), out.numel(), datatype(out), root, comm_, call->stream()), comm_, "ncclBroadcast");
    std::vector<at::Tensor> result{out};
    return work(OpType::BROADCAST, result, call);
  }

  c10::intrusive_ptr<Work> barrier(const BarrierOptions &) override {
    at::Tensor one = at::ones({1}, at::kFloat);
    check(ncclAllReduce(one.data_ptr(), one.data_ptr(), 1, ncclFloat32, ncclSum, comm_, nullptr), comm_, "barrier");
    std::vector<at::Tensor> none;
    return c10::make_intrusive<WorkMesh>(OpType::BARRIER, none, std::make_shared<Call>(one, false, streams_), comm_);
  }

 private:
  ncclComm_t comm_ = nullptr;
  void *links_ = nullptr;
  std::shared_ptr<StreamPool> streams_ = std::make_shared<StreamPool>();

  // A collective's call: synchronous on CPU tensors, on a stream of its own on MPS tensors.
  std::shared_ptr<Call> start(const at::Tensor &like, const char *op) { return std::make_shared<Call>(like, false, streams_, op); }
  // Whether `t`'s bytes are the window bytes `at` into the list laid end to end.
  static bool lies_at(const at::Tensor &t, const std::vector<at::Tensor> &list, size_t at) {
    char *base = window_span(list), *mine = t.is_mps() && t.is_contiguous() ? window_bytes(t) : nullptr;
    return base && mine && mine == base + at;
  }
  // An all-gather into `output`: in place where the input is its segment of the output in the window,
  // else the input a place of its own (out of place).  gather_issue issues it (the coalesced form's i-th
  // pair of places: `ins[i]` the input's, -1 in place).
  int gather_into(Call &call, at::Tensor &output, at::Tensor &input) {
    call.add(output, false, true);
    return call.mps() && lies_at(input, {output}, getRank() * input.nbytes()) ? -1 : call.add(input, true, false);
  }
  void gather_issue(Call &call, int out, int in, at::Tensor &input) {
    void *from = in < 0 ? (char *)call.ptr(out) + getRank() * input.nbytes() : call.ptr(in);
    check(ncclAllGather(from, call.ptr(out), input.numel(), datatype(input), comm_, call.stream()), comm_, "ncclAllGather");
  }
  // A reduce-scatter's places: the input itself where `output` is its segment `at` in the window (in
  // place: *out -1), else the input and the output (out of place: the library leaves the input as it is).
  // Returns the input's place; *out the output's.
  int scatter_from(Call &call, at::Tensor &output, const std::vector<at::Tensor> &input, size_t at, int *out) {
    *out = -1;
    if (call.mps() && lies_at(output, input, at)) return call.add(input, true, true);
    *out = call.add(output, false, true);
    return call.add(input, true, false);
  }
  // A list's element counts; uneven: those counts where they differ (the trace's splits), else none.
  static std::vector<size_t> numels(const std::vector<at::Tensor> &list) {
    std::vector<size_t> counts;
    for (auto &t : list) counts.push_back((size_t)t.numel());
    return counts;
  }
  static std::vector<int64_t> uneven(const std::vector<size_t> &counts) {
    for (auto c : counts)
      if (c != counts[0]) return std::vector<int64_t>(counts.begin(), counts.end());
    return {};
  }
  c10::intrusive_ptr<Work> work(OpType type, std::vector<at::Tensor> &outputs, std::shared_ptr<Call> call, std::vector<int64_t> splits = {}) {
    call->traced(std::move(splits));
    return c10::make_intrusive<WorkMesh>(type, outputs, std::move(call), comm_);
  }
  template <typename F> void group(F body) {
    check(ncclGroupStart(), comm_, "ncclGroupStart");
    try { body(); } catch (...) { ncclGroupEnd(); throw; }
    check(ncclGroupEnd(), comm_, "ncclGroupEnd");
  }
  // A reduction under torch's ReduceOp: SUM, PRODUCT, MIN, MAX, AVG as NCCL's; PREMUL_SUM a PreMulSum
  // operator of its factor in the tensor's type; a bool tensor's SUM its MAX (logical or).
  template <typename F> void reduce_call(const char *what, const ReduceOp &reduceOp, const at::Tensor &t, F call) {
    ncclRedOp_t op;
    bool created = false;
    switch ((ReduceOp::RedOpType)reduceOp) {
    case ReduceOp::SUM: op = t.scalar_type() == at::kBool ? ncclMax : ncclSum; break;
    case ReduceOp::PRODUCT: op = ncclProd; break;
    case ReduceOp::MIN: op = ncclMin; break;
    case ReduceOp::MAX: op = ncclMax; break;
    case ReduceOp::AVG: op = ncclAvg; break;
    case ReduceOp::PREMUL_SUM: {
      auto supplement = reinterpret_cast<PreMulSumSupplement *>(reduceOp.supplement_.get());
      at::Tensor factor = supplement->tensor_factor.defined() ? supplement->tensor_factor.to(at::kCPU).to(t.scalar_type())
                                                              : at::scalar_tensor(supplement->double_factor, t.options().device(at::kCPU));
      check(ncclRedOpCreatePreMulSum(&op, factor.data_ptr(), datatype(t), ncclScalarHostImmediate, comm_), comm_, "ncclRedOpCreatePreMulSum");
      created = true;
      break;
    }
    default: TORCH_CHECK(false, "mesh: ", what, " has no NCCL operator for ReduceOp ", (int)(ReduceOp::RedOpType)reduceOp);
    }
    ncclResult_t result = call(op);
    if (created) ncclRedOpDestroy(op, comm_);
    check(result, comm_, what);
  }
};

// MPS tensors: PyTorch registers the c10d operators for CPU (and CUDA) alone, so a collective on an MPS
// tensor reaches no backend.  This kernel of the MPS key runs the operator's CPU kernel, which reaches
// the group's backend for "cpu" (this one), with the MPS tensors themselves.
static void mps_to_backend(const c10::OperatorHandle &op, torch::jit::Stack *stack) {
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
}

// torch_mesh.all_to_all_counted (above) on the group named `group`.
static c10::intrusive_ptr<Work> all_to_all_counted(const std::string &group, at::Tensor out, at::Tensor input, at::Tensor counts,
                                                   std::vector<size_t> sends, std::vector<size_t> receives, at::Tensor host, at::Tensor landed) {
  auto *mesh = dynamic_cast<ProcessGroupMesh *>(resolve_process_group(group)->getBackend(c10::DeviceType::CPU).get());
  TORCH_CHECK(mesh, "mesh: all_to_all_counted runs on a group of the mesh backend");
  TORCH_CHECK(host.is_cpu() && host.is_contiguous() && host.scalar_type() == at::kLong, "mesh: all_to_all_counted's host counts are a contiguous CPU int64 tensor");
  return mesh->all_to_all_counted(out, input, counts, sends, receives, host, landed);
}
// Its received counts once they have landed: waited for on the host (the arrival word), not on the GPU.
static std::vector<int64_t> counted(at::Tensor host) {
  const uint64_t *arrived = (const uint64_t *)host.data_ptr<int64_t>();
  for (uint64_t deadline = uptime() + 300000000000ull; !__atomic_load_n(arrived, __ATOMIC_ACQUIRE);) {
    TORCH_CHECK(uptime() < deadline, "mesh: all_to_all_counted's counts did not arrive by the deadline");
    sched_yield();
  }
  TORCH_CHECK(__atomic_load_n(arrived, __ATOMIC_ACQUIRE) == 1, "mesh: all_to_all_counted failed before its counts arrived (the Work's wait raises why)");
  const int64_t *at = host.data_ptr<int64_t>() + 1;
  return std::vector<int64_t>(at, at + host.numel() - 1);
}

// What libnccl-mesh and this backend copied, sent and waited for so far (counted, not timed).
static pybind11::dict counts() {
  ncclMeshCounts_t c;
  check(ncclMeshGetCounts(&c), nullptr, "ncclMeshGetCounts");
  pybind11::dict d;
  d["library_cpu_copy_bytes"] = c.cpuCopyBytes;
  d["library_gpu_copy_bytes"] = c.gpuCopyBytes;
  d["library_gpu_kernels"] = c.gpuKernels;
  d["library_host_waits"] = c.hostWaits;
  d["library_input_waits"] = c.inputWaits;
  d["library_sent_bytes"] = c.sentBytes;
  d["library_received_bytes"] = c.receivedBytes;
  d["library_gpu_event_waits"] = c.gpuEventWaits;
  d["library_gpu_word_waits"] = c.gpuWordWaits;
  d["library_host_word_waits"] = c.hostWordWaits;
  d["library_commits"] = c.commits;
  d["library_wakeups"] = c.wakeups;
  d["library_buffers"] = c.buffers;
  d["backend_cpu_copy_bytes"] = cpu_copied.load();
  d["backend_gpu_copy_bytes"] = gpu_copied.load();
  d["backend_commits"] = commits.load();
  d["backend_fences"] = fences.load();
  d["backend_window_heaps"] = window_heaps_made.load();
  d["backend_window_buffers"] = window_buffers.load();
  d["backend_device_buffers"] = device_buffers.load();
  return d;
}
// The window allocator's records (ncclMeshMemRecords): address, bytes, freed, and each recorded point's
// value and its event's value now.
static pybind11::list records() {
  std::vector<ncclMeshMemRecord_t> table(4096);
  int count = 0;
  check(ncclMeshMemRecords(table.data(), (int)table.size(), &count), nullptr, "ncclMeshMemRecords");
  pybind11::list out;
  for (int i = 0; i < count && i < (int)table.size(); i++) {
    pybind11::dict d;
    d["address"] = table[i].address;
    d["bytes"] = table[i].bytes;
    d["freed"] = (bool)table[i].freed;
    pybind11::list points;
    for (int j = 0; j < table[i].points; j++) points.append(pybind11::make_tuple(table[i].value[j], table[i].reached[j]));
    d["points"] = points;
    out.append(d);
  }
  return out;
}
// ULFM's MPI_Comm_agree on the default group's communicator (the first made): every rank calls it; the
// first call that failed on any rank since the previous agreement (None: none did) and this rank's
// link-table epoch, on which every rank's table holds the same map of the group.
static pybind11::tuple agree() {
  ncclComm_t comm;
  {
    std::lock_guard<std::mutex> guard(comms_lock);
    TORCH_CHECK(!comms.empty(), "mesh: agree() needs a process group of the backend");
    comm = comms[0];
  }
  auto agreed = agree_on(comm, "");
  return pybind11::make_tuple(agreed.first ? pybind11::object(pybind11::int_(agreed.first)) : pybind11::none(), agreed.second);
}
// The last agreement this process made: (how many so far, the first failed call or None, the epoch).
static pybind11::tuple agreed() {
  std::lock_guard<std::mutex> guard(agreement.lock);
  return pybind11::make_tuple(agreement.count, agreement.failed ? pybind11::object(pybind11::int_(agreement.failed)) : pybind11::none(),
                              agreement.epoch);
}
// A snapshot of the link table of `region` (empty: MESH_REGION) of N nodes: (epoch, the bridge's node,
// present [N], links [N, N, 4] of alpha, beta, stated, up, reported [N]: each node's last report's
// sequence, 0 none).
static pybind11::tuple links(const std::string &region) {
  void *table = nullptr;
  check(ncclMeshLinksAttach(region.empty() ? nullptr : region.c_str(), &table), nullptr, "ncclMeshLinksAttach");
  uint32_t n = 0, node = 0;
  uint64_t epoch = 0;
  check(ncclMeshLinksRead(table, &n, nullptr, nullptr, nullptr, nullptr, nullptr), nullptr, "ncclMeshLinksRead");
  std::vector<ncclMeshLink_t> snapshot((size_t)n * n);
  std::vector<uint32_t> present(n);
  std::vector<uint64_t> reported(n);
  check(ncclMeshLinksRead(table, &n, snapshot.data(), present.data(), reported.data(), &node, &epoch), nullptr, "ncclMeshLinksRead");
  ncclMeshLinksDetach(table);
  at::Tensor t = at::empty({(int64_t)n, (int64_t)n, 4}, at::kFloat), p = at::empty({(int64_t)n}, at::kInt);
  for (size_t i = 0; i < (size_t)n * n; i++) {
    float *row = t.data_ptr<float>() + 4 * i;
    row[0] = snapshot[i].alpha, row[1] = snapshot[i].beta, row[2] = (float)snapshot[i].stated, row[3] = (float)snapshot[i].up;
  }
  pybind11::list sequences;
  for (uint32_t i = 0; i < n; i++) {
    p.data_ptr<int>()[i] = (int)present[i];
    sequences.append(reported[i]);
  }
  return pybind11::make_tuple(epoch, node, p, t, sequences);
}
// Where a tensor's bytes are, as the records name them (0 for a tensor outside the window).
static uint64_t address(const at::Tensor &t) {
  if (t.is_mps()) return (uint64_t)(uintptr_t)window_bytes(t);
  void *buffer = nullptr;
  size_t offset = 0;
  return ncclMeshMemBuffer(t.data_ptr(), &buffer, &offset) == ncclSuccess ? (uint64_t)(uintptr_t)t.data_ptr() : 0;
}

// The trace (MESH_TRACE) written out, once the points still to be noted are (at most 1 s): a header line
// (the clock, uptime ns, and the library's clock's offset from it), a line per call (its fence's command
// buffer's GPU times; the host times its fence, its kept work's reach and resume signals and its stream's
// value were seen reached).
static void trace_dump() {
  const char *path = getenv("MESH_TRACE");
  if (!path) return;
  auto pending = [] {
    std::lock_guard<std::mutex> guard(trace.lock);
    auto waiting = [&](void *event, uint64_t value, bool gpu) {
      if (!event || !value || [(__bridge id<MTLSharedEvent>)event signaledValue] < value) return false;
      return !trace.reached.count({event, value}) || (gpu && !trace.gpu.count(value));
    };
    void *fence = (__bridge void *)fence_event();
    auto unseen = [&](uint64_t index) { return index && trace.words[index] && !trace.reached.count({trace.words + index, 1}); };
    for (auto &c : trace.calls)
      if (waiting(fence, c.fence, true) || unseen(c.reach) || unseen(c.resume) || waiting(c.stream_event, c.stream_value, false)) return true;
    return false;
  };
  for (int i = 0; i < 1000 && pending(); i++) usleep(1000);
  std::lock_guard<std::mutex> guard(trace.lock);
  const uint64_t offset = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) - uptime();
  void *fence = (__bridge void *)fence_event();
  auto reached = [&](void *event, uint64_t value) -> std::string {
    auto it = trace.reached.find({event, value});
    return value && it != trace.reached.end() ? std::to_string(it->second) : "null";
  };
  auto gpu = [&](uint64_t value) -> std::string {
    auto it = trace.gpu.find(value);
    if (!value || it == trace.gpu.end()) return "null";
    return "[" + std::to_string((uint64_t)(it->second.first * 1e9)) + "," + std::to_string((uint64_t)(it->second.second * 1e9)) + "]";
  };
  std::ofstream out(path);
  out << "{\"clock\":\"uptime_ns\",\"library_offset_ns\":" << offset << ",\"pid\":" << getpid() << "}\n";
  for (size_t i = 0; i < trace.calls.size(); i++) {
    auto &c = trace.calls[i];
    out << "{\"call\":" << i << ",\"op\":\"" << c.op << "\",\"dtype\":\"" << c.dtype << "\",\"device\":\"" << (c.mps ? "mps" : "cpu")
        << "\",\"bytes\":" << c.bytes << ",\"in_place\":" << c.in_place << ",\"blit_in\":" << c.blit_in << ",\"blit_out\":" << c.blit_out
        << ",\"splits\":[";
    for (size_t k = 0; k < c.splits.size(); k++) out << (k ? "," : "") << c.splits[k];
    out << "],\"enter_ns\":" << c.enter << ",\"issued_ns\":" << c.issued << ",\"fence\":" << c.fence << ",\"fence_reached_ns\":"
        << reached(fence, c.fence) << ",\"fence_gpu_ns\":" << gpu(c.fence) << ",\"stream_done_ns\":" << reached(c.stream_event, c.stream_value)
        << ",\"reach_ns\":" << reached(c.reach ? trace.words + c.reach : nullptr, 1) << ",\"resume_ns\":"
        << reached(c.resume ? trace.words + c.resume : nullptr, 1) << ",\"commits\":" << c.commits << ",\"library\":[";
    ncclMeshCounts_t counts[64];
    int n = 0;
    ncclMeshTallyCounts(c.tally, counts, 64, &n);
    for (int k = 0; k < n && k < 64; k++) {
      auto &t = counts[k];
      auto time = [&](uint64_t ns) { return ns ? std::to_string(ns - offset) : std::string("null"); };
      out << (k ? "," : "") << "{\"start_ns\":" << time(t.startNs) << ",\"end_ns\":" << time(t.endNs) << ",\"arrived_ns\":" << time(t.arrivedNs)
          << ",\"sent\":" << t.sentBytes
          << ",\"received\":" << t.receivedBytes << ",\"host_waits\":" << t.hostWaits << ",\"input_waits\":" << t.inputWaits
          << ",\"kernels\":" << t.gpuKernels << ",\"gpu_copy\":" << t.gpuCopyBytes << ",\"cpu_copy\":" << t.cpuCopyBytes
          << ",\"gpu_event_waits\":" << t.gpuEventWaits << ",\"gpu_word_waits\":" << t.gpuWordWaits << ",\"host_word_waits\":" << t.hostWordWaits
          << ",\"commits\":" << t.commits << ",\"wakeups\":" << t.wakeups << ",\"buffers\":" << t.buffers << "}";
    }
    out << "]}\n";
  }
}

// ---- functional collectives on MPS tensors ----
// torch's _c10d_functional ops (torch/csrc/distributed/c10d/Functional.cpp, DTensor's redistributions
// and context parallelism) return a fresh tensor: all_reduce and broadcast clone their input and run in
// place on the clone, the others allocate their output.  These MPS kernels are Functional.cpp's with the
// output a fresh MPS tensor (window memory: window_heaps) and the call out of place on the input, read in
// place, so no clone: the library sends the input's bytes and combines them with the pieces it receives
// straight into the output (nccl.h).  The work is registered on the output as Functional.cpp registers it,
// for wait_tensor.  A group whose backend is not this one takes Functional.cpp's own path.
static c10::intrusive_ptr<ProcessGroup> group_of(const c10::IValue &v) {
  return v.isString() ? resolve_process_group(v.toStringRef()) : v.toCustomClass<ProcessGroup>();
}
static ProcessGroupMesh *mesh_of(const c10::intrusive_ptr<ProcessGroup> &group) {
  return dynamic_cast<ProcessGroupMesh *>(group->getBackend(c10::DeviceType::CPU).get());
}
static ReduceOp reduce_of(const std::string &name) {
  if (name == "sum") return ReduceOp::SUM;
  if (name == "avg") return ReduceOp::AVG;
  if (name == "product") return ReduceOp::PRODUCT;
  if (name == "min") return ReduceOp::MIN;
  if (name == "max") return ReduceOp::MAX;
  TORCH_CHECK(false, "mesh: functional collectives on MPS take sum, avg, product, min and max, not ", name);
}
static std::vector<int64_t> rows(at::IntArrayRef sizes, int64_t first) {
  std::vector<int64_t> shape(sizes.begin(), sizes.end());
  shape[0] = first;
  return shape;
}
static void functional_all_reduce(const c10::OperatorHandle &, torch::jit::Stack *stack) {
  auto group = group_of((*stack)[2]);
  const std::string op = (*stack)[1].toStringRef();
  at::Tensor input = (*stack)[0].toTensor().contiguous();
  torch::jit::drop(*stack, 3);
  at::Tensor out = at::empty(input.sizes(), input.options());
  if (auto *mesh = mesh_of(group)) {
    std::vector<at::Tensor> outs{out};
    register_work(out, mesh->allreduce_into(outs, {input}, reduce_of(op)));
  } else {
    out.copy_(input);
    std::vector<at::Tensor> tensors{out};
    AllreduceOptions opts;
    opts.reduceOp = reduce_of(op);
    register_work(out, group->allreduce(tensors, opts));
  }
  torch::jit::push(*stack, out);
}
static void functional_all_gather(const c10::OperatorHandle &, torch::jit::Stack *stack) {
  auto group = group_of((*stack)[2]);
  const int64_t size = (*stack)[1].toInt();
  at::Tensor input = (*stack)[0].toTensor().contiguous();
  torch::jit::drop(*stack, 3);
  at::Tensor out = at::empty(rows(input.sizes(), input.size(0) * size), input.options());
  register_work(out, group->_allgather_base(out, input));
  torch::jit::push(*stack, out);
}
static void functional_reduce_scatter(const c10::OperatorHandle &, torch::jit::Stack *stack) {
  auto group = group_of((*stack)[3]);
  const int64_t size = (*stack)[2].toInt();
  const std::string op = (*stack)[1].toStringRef();
  at::Tensor input = (*stack)[0].toTensor().contiguous();
  torch::jit::drop(*stack, 4);
  TORCH_CHECK(input.dim() && input.size(0) % size == 0, "mesh: reduce_scatter_tensor's first dimension is a multiple of the group size");
  at::Tensor out = at::empty(rows(input.sizes(), input.size(0) / size), input.options());
  ReduceScatterOptions opts;
  opts.reduceOp = reduce_of(op);
  register_work(out, group->_reduce_scatter_base(out, input, opts));
  torch::jit::push(*stack, out);
}
static void functional_all_to_all(const c10::OperatorHandle &, torch::jit::Stack *stack) {
  auto group = group_of((*stack)[3]);
  std::vector<int64_t> in_splits, out_splits;
  for (auto &v : (*stack)[2].toListRef()) in_splits.push_back(v.isInt() ? v.toInt() : v.toSymInt().guard_int(__FILE__, __LINE__));
  for (auto &v : (*stack)[1].toListRef()) out_splits.push_back(v.isInt() ? v.toInt() : v.toSymInt().guard_int(__FILE__, __LINE__));
  at::Tensor input = (*stack)[0].toTensor().contiguous();
  torch::jit::drop(*stack, 4);
  int64_t first = input.dim() ? input.size(0) : 1;
  if (!out_splits.empty()) {
    first = 0;
    for (auto v : out_splits) first += v;
  }
  at::Tensor out = at::empty(input.dim() ? rows(input.sizes(), first) : std::vector<int64_t>{}, input.options());
  register_work(out, group->alltoall_base(out, input, out_splits, in_splits));
  torch::jit::push(*stack, out);
}
static void functional_broadcast(const c10::OperatorHandle &, torch::jit::Stack *stack) {
  auto group = group_of((*stack)[2]);
  const int64_t src = (*stack)[1].toInt();
  at::Tensor input = (*stack)[0].toTensor().contiguous();
  torch::jit::drop(*stack, 3);
  at::Tensor out = at::empty(input.sizes(), input.options());
  if (auto *mesh = mesh_of(group)) register_work(out, mesh->broadcast_into(out, input, (int)src));
  else {
    out.copy_(input);
    std::vector<at::Tensor> tensors{out};
    BroadcastOptions opts;
    opts.rootRank = src;
    register_work(out, group->broadcast(tensors, opts));
  }
  torch::jit::push(*stack, out);
}
static void functional_all_gather_coalesced(const c10::OperatorHandle &, torch::jit::Stack *stack) {
  auto group = group_of((*stack)[2]);
  const int64_t size = (*stack)[1].toInt();
  std::vector<at::Tensor> inputs, outs;
  for (auto &t : (*stack)[0].toTensorVector()) inputs.push_back(t.contiguous());
  torch::jit::drop(*stack, 3);
  for (auto &t : inputs) outs.push_back(at::empty(rows(t.sizes(), t.size(0) * size), t.options()));
  auto work = group->allgather_into_tensor_coalesced(outs, inputs);
  for (auto &t : outs) register_work(t, work);
  torch::jit::push(*stack, outs);
}
static void functional_reduce_scatter_coalesced(const c10::OperatorHandle &, torch::jit::Stack *stack) {
  auto group = group_of((*stack)[3]);
  const int64_t size = (*stack)[2].toInt();
  const std::string op = (*stack)[1].toStringRef();
  std::vector<at::Tensor> inputs, outs;
  for (auto &t : (*stack)[0].toTensorVector()) inputs.push_back(t.contiguous());
  torch::jit::drop(*stack, 4);
  for (auto &t : inputs) {
    TORCH_CHECK(t.dim() && t.size(0) % size == 0, "mesh: reduce_scatter_tensor's first dimension is a multiple of the group size");
    outs.push_back(at::empty(rows(t.sizes(), t.size(0) / size), t.options()));
  }
  ReduceScatterOptions opts;
  opts.reduceOp = reduce_of(op);
  auto work = group->reduce_scatter_tensor_coalesced(outs, inputs, opts);
  for (auto &t : outs) register_work(t, work);
  torch::jit::push(*stack, outs);
}
static void functional_all_reduce_coalesced(const c10::OperatorHandle &, torch::jit::Stack *stack) {
  auto group = group_of((*stack)[2]);
  const std::string op = (*stack)[1].toStringRef();
  std::vector<at::Tensor> inputs, outs;
  for (auto &t : (*stack)[0].toTensorVector()) inputs.push_back(t.contiguous());
  torch::jit::drop(*stack, 3);
  for (auto &t : inputs) outs.push_back(at::empty(t.sizes(), t.options()));
  c10::intrusive_ptr<Work> work;
  if (auto *mesh = mesh_of(group)) work = mesh->allreduce_into(outs, inputs, reduce_of(op));
  else {
    for (size_t i = 0; i < inputs.size(); i++) outs[i].copy_(inputs[i]);
    AllreduceCoalescedOptions opts;
    opts.reduceOp = reduce_of(op);
    work = group->allreduce_coalesced(outs, opts);
  }
  for (auto &t : outs) register_work(t, work);
  torch::jit::push(*stack, outs);
}

} // namespace c10d

TORCH_LIBRARY_IMPL(_c10d_functional, MPS, m) {
  m.impl("all_reduce", torch::CppFunction::makeFromBoxedFunction<&c10d::functional_all_reduce>());
  m.impl("all_gather_into_tensor", torch::CppFunction::makeFromBoxedFunction<&c10d::functional_all_gather>());
  m.impl("reduce_scatter_tensor", torch::CppFunction::makeFromBoxedFunction<&c10d::functional_reduce_scatter>());
  m.impl("all_to_all_single", torch::CppFunction::makeFromBoxedFunction<&c10d::functional_all_to_all>());
  m.impl("broadcast", torch::CppFunction::makeFromBoxedFunction<&c10d::functional_broadcast>());
  m.impl("all_gather_into_tensor_coalesced", torch::CppFunction::makeFromBoxedFunction<&c10d::functional_all_gather_coalesced>());
  m.impl("reduce_scatter_tensor_coalesced", torch::CppFunction::makeFromBoxedFunction<&c10d::functional_reduce_scatter_coalesced>());
  m.impl("all_reduce_coalesced", torch::CppFunction::makeFromBoxedFunction<&c10d::functional_all_reduce_coalesced>());
}

TORCH_LIBRARY_IMPL(c10d, MPS, m) {
  for (const char *name : {"broadcast_", "allreduce_", "allreduce_coalesced_", "reduce_", "allgather_", "_allgather_base_",
                           "allgather_coalesced_", "allgather_into_tensor_coalesced_", "gather_", "scatter_", "reduce_scatter_",
                           "_reduce_scatter_base_", "reduce_scatter_tensor_coalesced_", "alltoall_", "alltoall_base_",
                           "barrier", "send", "recv_"})
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&c10d::mps_to_backend>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupMesh", &c10d::ProcessGroupMesh::create);
  m.def("empty", &c10d::empty);
  m.def("counts", &c10d::counts);
  m.def("records", &c10d::records);
  m.def("address", &c10d::address);
  m.def("trace_dump", &c10d::trace_dump);
  m.def("agree", &c10d::agree);
  m.def("agreed", &c10d::agreed);
  m.def("links", &c10d::links);
  m.def("all_to_all_counted", &c10d::all_to_all_counted);
  m.def("counted", &c10d::counted);
}
