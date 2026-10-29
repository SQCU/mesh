// torch's allocators and Python's object arenas over region leases (ncclMemAlloc), so no tensor and no small Python
// object of a mesh process lies outside the region: every pymalloc arena is a lease (PyObject_SetArenaAllocator); every CPU
// tensor is a lease, its DataPtr the lease's address, returned (ncclMemFree) when its storage is released; every MPS
// tensor is a lease, its DataPtr the lease's no-copy MTLBuffer.  Torch 2.14 has no MPS allocator hook, so this one is constructed in place over the fixed instance
// at::mps::getIMPSAllocator(), whose 16 bytes are kept as the delegate for pointers it did not make.  A block let go
// returns its lease (ncclMemFree: the pages kept until every DMA read from them ends) when the command buffer open on
// torch's stream completes, as torch's allocator returns a buffer in use, and commits it (torch's kernels never do).
#include <Python.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <c10/core/CPUAllocator.h>
#include <ATen/mps/MPSEvent.h>
#include <ATen/mps/MPSStream.h>
#include <c10/core/Storage.h>
#include <c10/util/Exception.h>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#import <Metal/Metal.h>
#include "nccl.h"

namespace {

struct Lease { void *memory; size_t bytes; id_t id; std::vector<int64_t> shape; std::shared_ptr<at::mps::MPSEvent> event; };

std::mutex *lock_ = new std::mutex;
auto *leases_ = new std::unordered_map<const void *, Lease>;
auto *hosts_ = new std::unordered_map<const void *, size_t>;   // the CPU tensors' leases: each its bytes
id_t made_ = 0;
alignas(16) unsigned char torchs_[16];

at::mps::IMPSAllocator *torchs() { return reinterpret_cast<at::mps::IMPSAllocator *>(torchs_); }

// `ours(lease)` on the lease of `buffer`, under the lock; `theirs()` where torch's own allocator made it
template <typename Ours, typename Theirs> auto with_lease(const void *buffer, Ours ours, Theirs theirs) {
  {
    std::lock_guard<std::mutex> guard(*lock_);
    if (auto found = leases_->find(buffer); found != leases_->end()) return ours(found->second);
  }
  return theirs();
}
bool ours(const void *buffer) { return with_lease(buffer, [](Lease &) { return true; }, [] { return false; }); }

// Whether this thread runs on `stream`'s serial queue, where a dispatch_sync onto it would never return.
char queue_key_;
bool on_queue(at::mps::MPSStream *stream) {
  if (dispatch_queue_get_specific(stream->queue(), &queue_key_) != stream)
    dispatch_queue_set_specific(stream->queue(), &queue_key_, stream, nullptr);
  return dispatch_get_specific(&queue_key_) == stream;
}

void release(void *buffer) {
  if (!buffer) return;
  void *memory = nullptr;
  {
    std::lock_guard<std::mutex> guard(*lock_);
    auto found = leases_->find(buffer);
    TORCH_INTERNAL_ASSERT(found != leases_->end(), "region allocator: no lease of buffer ", buffer);
    memory = found->second.memory;
    leases_->erase(found);
  }
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MTLCommandBufferHandler returned = ^(id<MTLCommandBuffer>) { ncclMemFree(memory); };
  if (on_queue(stream)) {
    [stream->commandBuffer() addCompletedHandler:returned];
    return;
  }
  dispatch_sync(stream->queue(), ^{
    @autoreleasepool {
      [stream->commandBuffer() addCompletedHandler:returned];
      stream->synchronize(at::mps::SyncType::COMMIT);
    }
  });
}

c10::DataPtr lease(size_t bytes) {
  if (!bytes) return {nullptr, nullptr, &release, at::Device(at::DeviceType::MPS, 0)};
  void *memory = nullptr, *buffer = nullptr;
  size_t offset = 0;
  TORCH_CHECK_WITH(OutOfMemoryError, !ncclMemAlloc(&memory, bytes), "MPS backend out of memory: a region lease of ", bytes,
                   " bytes (torch_mesh allocator.mm): ", ncclGetLastError(nullptr));
  if (ncclMeshMemBuffer(memory, &buffer, &offset)) {
    const std::string why = ncclGetLastError(nullptr);
    ncclMemFree(memory);
    TORCH_CHECK_WITH(OutOfMemoryError, false, "MPS backend out of memory: the Metal buffer of a region lease of ", bytes,
                     " bytes (torch_mesh allocator.mm): ", why);
  }
  std::lock_guard<std::mutex> guard(*lock_);
  leases_->emplace(buffer, Lease{memory, bytes, ++made_, {}, nullptr});
  return {buffer, buffer, &release, at::Device(at::DeviceType::MPS, 0)};
}

// pymalloc's arenas (its 1 MiB blocks of small objects): each a lease, those made before the region's held by the
// allocator that made them, which also returns them
PyObjectArenaAllocator pythons_;
auto *arenas_ = new std::unordered_set<void *>;
void *arena_take(void *, size_t bytes) {
  void *memory = nullptr;
  if (ncclMemAlloc(&memory, bytes)) return nullptr;   // pymalloc's out-of-memory: a MemoryError
  std::lock_guard<std::mutex> guard(*lock_);
  arenas_->insert(memory);
  return memory;
}
void arena_return(void *, void *memory, size_t bytes) {
  bool ours;
  {
    std::lock_guard<std::mutex> guard(*lock_);
    ours = arenas_->erase(memory) == 1;
  }
  if (ours) ncclMemFree(memory);
  else pythons_.free(pythons_.ctx, memory, bytes);
}

void release_host(void *memory) {
  if (!memory) return;
  {
    std::lock_guard<std::mutex> guard(*lock_);
    hosts_->erase(memory);
  }
  ncclMemFree(memory);
}

// torch's CPU allocator: a CPU tensor's bytes are a lease's pages (page aligned, beyond torch's 64-byte alignment)
struct RegionHostAllocator final : c10::Allocator {
  c10::DataPtr allocate(size_t bytes) override {
    const c10::Device cpu(c10::DeviceType::CPU);
    if (!bytes) return {nullptr, nullptr, &release_host, cpu};
    void *memory = nullptr;
    TORCH_CHECK_WITH(OutOfMemoryError, !ncclMemAlloc(&memory, bytes), "CPU out of memory: a region lease of ", bytes,
                     " bytes (torch_mesh allocator.mm): ", ncclGetLastError(nullptr));
    {
      std::lock_guard<std::mutex> guard(*lock_);
      hosts_->emplace(memory, bytes);
    }
    return {memory, memory, &release_host, cpu};
  }
  c10::DeleterFnPtr raw_deleter() const override { return &release_host; }
  void copy_data(void *dest, const void *src, std::size_t count) const override { std::memcpy(dest, src, count); }
};

struct RegionAllocator final : at::mps::IMPSAllocator {
  c10::DataPtr allocate(size_t bytes) override { return lease(bytes); }
  c10::DeleterFnPtr raw_deleter() const override { return &release; }
  void copy_data(void *dest, const void *src, std::size_t count) const override {
    std::memcpy([(__bridge id<MTLBuffer>)dest contents], [(__bridge id<MTLBuffer>)src contents], count);
  }
  bool initialized() override { return true; }
  void recordStream(const c10::DataPtr &, c10::Stream) override {}

  // torch's heaps' own accounts and policies, unchanged
  void emptyCache(c10::MempoolId_t pool) override { torchs()->emptyCache(pool); }
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex d) override { return torchs()->getDeviceStats(d); }
  void resetAccumulatedStats(c10::DeviceIndex d) override { torchs()->resetAccumulatedStats(d); }
  void resetPeakStats(c10::DeviceIndex d) override { torchs()->resetPeakStats(d); }
  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex d) override { return torchs()->getMemoryInfo(d); }
  void freeInactiveBuffers() const override { torchs()->freeInactiveBuffers(); }
  std::string formatSize(size_t bytes) const override { return torchs()->formatSize(bytes); }
  void setLowWatermarkRatio(double ratio) const override { torchs()->setLowWatermarkRatio(ratio); }
  void setHighWatermarkRatio(double ratio) const override { torchs()->setHighWatermarkRatio(ratio); }
  ssize_t getLowWatermarkValue() const override { return torchs()->getLowWatermarkValue(); }
  size_t getLowWatermarkLimit() const override { return torchs()->getLowWatermarkLimit(); }
  size_t getHighWatermarkLimit() const override { return torchs()->getHighWatermarkLimit(); }
  size_t getTotalAllocatedMemory() const override { return torchs()->getTotalAllocatedMemory(); }
  size_t getCurrentAllocatedMemory() const override { return torchs()->getCurrentAllocatedMemory(); }
  size_t getDriverAllocatedMemory() const override { return torchs()->getDriverAllocatedMemory(); }
  size_t getRecommendedMaxMemory() const override { return torchs()->getRecommendedMaxMemory(); }

  // a block's own queries: its lease's where it is one, torch's otherwise
  ssize_t getUnalignedBufferSize(const void *b) const override {
    return with_lease(b, [](Lease &l) { return (ssize_t)l.bytes; }, [b] { return torchs()->getUnalignedBufferSize(b); });
  }
  c10::IntArrayRef getBufferShape(const void *b) const override {
    return with_lease(b, [](Lease &l) { return c10::IntArrayRef(l.shape); }, [b] { return torchs()->getBufferShape(b); });
  }
  id_t getBufferId(const void *b) const override {
    return with_lease(b, [](Lease &l) { return l.id; }, [b] { return torchs()->getBufferId(b); });
  }
  void setBufferShape(const void *b, const c10::IntArrayRef &shape) const override {
    with_lease(b, [&](Lease &l) { l.shape = shape.vec(); }, [&] { torchs()->setBufferShape(b, shape); });
  }
  bool isSharedBuffer(const void *b) const override { return ours(b) || torchs()->isSharedBuffer(b); }
  c10::DataPtr allocScalarBufferWithValue(void *value, size_t bytes) const override {
    c10::DataPtr scalar = lease(bytes);
    std::memcpy([(__bridge id<MTLBuffer>)scalar.get() contents], value, bytes);
    return scalar;
  }
  std::pair<const void *, uint32_t> getSharedBufferPtr(const void *b) const override {
    if (!ours(b)) return torchs()->getSharedBufferPtr(b);
    id<MTLBuffer> metal = (__bridge id<MTLBuffer>)b;
    return {[metal contents], (uint32_t)[metal retainCount]};
  }
  c10::Storage getHostAliasStorage(const c10::Storage &storage) const override {
    if (!ours(storage.data())) return torchs()->getHostAliasStorage(storage);
    void *contents = [(__bridge id<MTLBuffer>)storage.data() contents];
    c10::DataPtr alias(contents, new c10::Storage(storage), [](void *held) { delete static_cast<c10::Storage *>(held); },
                       c10::Device(c10::DeviceType::CPU));
    return c10::Storage(c10::Storage::use_byte_size_t(), storage.nbytes(), std::move(alias), nullptr, false);
  }
  // Host reads of a block wait on an event recorded on torch's current stream after the GPU's last use of it.
  bool recordEvents(c10::ArrayRef<const void *> buffers) const override {
    std::vector<const void *> theirs;
    bool recorded = false;
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    for (const void *b : buffers) {
      if (!ours(b)) { theirs.push_back(b); continue; }
      std::shared_ptr<at::mps::MPSEvent> event = at::mps::getMPSEventPool()->acquireEvent(false, stream);
      event->record(!on_queue(stream));
      recorded |= with_lease(b, [&](Lease &l) { l.event = std::move(event); return true; }, [] { return false; });
    }
    return (!theirs.empty() && torchs()->recordEvents(theirs)) || recorded;
  }
  bool waitForEvents(c10::ArrayRef<const void *> buffers) const override {
    using Event = std::optional<std::shared_ptr<at::mps::MPSEvent>>;
    std::vector<const void *> theirs;
    bool waited = false;
    for (const void *b : buffers) {
      const Event event = with_lease(b, [](Lease &l) { return Event(l.event); }, [] { return Event(); });
      if (!event) theirs.push_back(b);
      else if (*event) waited |= (*event)->synchronize();
    }
    return (!theirs.empty() && torchs()->waitForEvents(theirs)) || waited;
  }
};

static_assert(sizeof(RegionAllocator) == sizeof(at::mps::IMPSAllocator) && sizeof(at::mps::IMPSAllocator) == sizeof(void *),
              "the region allocator is constructed over torch's instance, whose vtable pointer alone it replaces");

} // namespace

// The region allocators: the CPU one by torch's own hook, above its default; the MPS one in place of torch's instance,
// which must be of torch's class (the layout the delegate copies).
void install_region_allocator() {
  static RegionHostAllocator host;
  c10::SetCPUAllocator(&host, 1);
  PyObject_GetArenaAllocator(&pythons_);
  PyObjectArenaAllocator region = {nullptr, arena_take, arena_return};
  PyObject_SetArenaAllocator(&region);
  at::mps::IMPSAllocator *instance = at::mps::getIMPSAllocator();
  const char *type = typeid(*instance).name();
  TORCH_CHECK(!strcmp(type, "N2at3mps12MPSAllocatorE"), "torch_mesh: torch's MPS allocator instance (at::mps::getIMPSAllocator) "
              "is of class ", type, ", not torch's at::mps::MPSAllocator, so the region allocator cannot take its place (allocator.mm)");
  std::memcpy(torchs_, static_cast<const void *>(instance), sizeof torchs_);
  new (instance) RegionAllocator();
}

// The region allocators' leases live now: each its address and the bytes asked of it.
std::vector<std::pair<uint64_t, uint64_t>> region_allocations() {
  std::lock_guard<std::mutex> guard(*lock_);
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (const auto &[buffer, lease] : *leases_) out.emplace_back((uint64_t)(uintptr_t)lease.memory, lease.bytes);
  for (const auto &[memory, bytes] : *hosts_) out.emplace_back((uint64_t)(uintptr_t)memory, bytes);
  return out;
}
