// torch's MPS allocator over region leases: every MPS tensor torch makes in a mesh program (its factories', its eager
// temporaries', Inductor's buffers, `.to('mps')`'s) is a lease of this node's bridge's region (ncclMemAlloc), visible
// to the collectives, and none lies in torch's heaps, which the GPU wires whole beside the region.
//
// Torch 2.14 has no allocator hook for MPS: its factories call at::mps::getIMPSAllocator(), a fixed instance, and
// c10::SetAllocator(kMPS) reaches no factory. Every caller outside torch's MPSAllocator.mm dispatches through the
// instance's vtable, so the region allocator is constructed in place over that instance, whose 16 bytes (a vtable
// pointer and its usage flags; its state is torch's MPSHeapAllocatorImpl) are kept aside as the delegate for pointers
// this allocator did not make. MPSAllocator.mm's own calls (pinned CPU memory, isMPSPinnedPtr) bind torch's class
// directly and stay in torch's heaps.
//
// A block is a lease of its own, with the lease's no-copy shared MTLBuffer (ncclMeshMemBuffer) as its DataPtr, as an
// MPS DataPtr is the MTLBuffer itself. A block let go returns its lease (ncclMemFree, which keeps the pages until any
// DMA read posted from them has completed) once the command buffer open on torch's current stream completes, so every
// command bound to it before then has run. The release commits that buffer (SyncType::COMMIT): torch's Metal kernels
// never commit, so an eager loop with no host wait would hold every lease it let go until its next wait (fewer than 150
// steps of a 64 MiB elementwise op leased all of a 14 GiB region). A block let go on the stream's own queue, where an
// encoder may be open, returns at the next commit. A lease the region cannot meet is torch's out-of-memory error.
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSEvent.h>
#include <ATen/mps/MPSStream.h>
#include <c10/core/Storage.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>
#import <Metal/Metal.h>
#include "nccl.h"

namespace {

struct Lease {
  void *memory;
  size_t bytes;
  id_t id;
  std::vector<int64_t> shape;
  std::shared_ptr<at::mps::MPSEvent> event;
};

std::mutex *lock_ = new std::mutex;
auto *leases_ = new std::unordered_map<const void *, Lease>;
id_t made_ = 0;
alignas(16) unsigned char torchs_[16];

at::mps::IMPSAllocator *torchs() { return reinterpret_cast<at::mps::IMPSAllocator *>(torchs_); }

bool ours(const void *buffer) {
  std::lock_guard<std::mutex> guard(*lock_);
  return leases_->count(buffer);
}

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
  const ncclResult_t leased = ncclMemAlloc(&memory, bytes);
  TORCH_CHECK_WITH(OutOfMemoryError, !leased, "MPS backend out of memory: a region lease of ", bytes,
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

// The address of the local symbol `name` in the image that holds `inside` (its symbol table, as nm lists it); 0 where
// it has none.
uintptr_t local_symbol(const void *inside, const char *name) {
  Dl_info info{};
  if (!dladdr(inside, &info) || !info.dli_fbase) return 0;
  const auto *header = static_cast<const mach_header_64 *>(info.dli_fbase);
  const segment_command_64 *text = nullptr, *linkedit = nullptr;
  const symtab_command *symtab = nullptr;
  const auto *command = reinterpret_cast<const load_command *>(header + 1);
  for (uint32_t k = 0; k < header->ncmds; k++) {
    if (command->cmd == LC_SEGMENT_64) {
      const auto *segment = reinterpret_cast<const segment_command_64 *>(command);
      if (!strcmp(segment->segname, SEG_TEXT)) text = segment;
      if (!strcmp(segment->segname, SEG_LINKEDIT)) linkedit = segment;
    } else if (command->cmd == LC_SYMTAB) {
      symtab = reinterpret_cast<const symtab_command *>(command);
    }
    command = reinterpret_cast<const load_command *>(reinterpret_cast<const char *>(command) + command->cmdsize);
  }
  if (!text || !linkedit || !symtab) return 0;
  const uintptr_t slide = reinterpret_cast<uintptr_t>(header) - text->vmaddr;
  const char *base = reinterpret_cast<const char *>(linkedit->vmaddr + slide - linkedit->fileoff);
  const auto *symbols = reinterpret_cast<const nlist_64 *>(base + symtab->symoff);
  const char *strings = base + symtab->stroff;
  for (uint32_t k = 0; k < symtab->nsyms; k++)
    if ((symbols[k].n_type & N_TYPE) == N_SECT && !strcmp(strings + symbols[k].n_un.n_strx, name)) return symbols[k].n_value + slide;
  return 0;
}

struct RegionAllocator final : at::mps::IMPSAllocator {
  c10::DataPtr allocate(size_t bytes) override { return lease(bytes); }
  c10::DeleterFnPtr raw_deleter() const override { return &release; }
  void copy_data(void *dest, const void *src, std::size_t count) const override {
    std::memcpy([(__bridge id<MTLBuffer>)dest contents], [(__bridge id<MTLBuffer>)src contents], count);
  }

  bool initialized() override { return true; }
  void emptyCache(c10::MempoolId_t pool) override { torchs()->emptyCache(pool); }
  void recordStream(const c10::DataPtr &, c10::Stream) override {}
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override { return torchs()->getDeviceStats(device); }
  void resetAccumulatedStats(c10::DeviceIndex device) override { torchs()->resetAccumulatedStats(device); }
  void resetPeakStats(c10::DeviceIndex device) override { torchs()->resetPeakStats(device); }
  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override { return torchs()->getMemoryInfo(device); }

  void freeInactiveBuffers() const override { torchs()->freeInactiveBuffers(); }
  ssize_t getUnalignedBufferSize(const void *buffer) const override {
    {
      std::lock_guard<std::mutex> guard(*lock_);
      auto found = leases_->find(buffer);
      if (found != leases_->end()) return (ssize_t)found->second.bytes;
    }
    return torchs()->getUnalignedBufferSize(buffer);
  }
  c10::IntArrayRef getBufferShape(const void *buffer) const override {
    {
      std::lock_guard<std::mutex> guard(*lock_);
      auto found = leases_->find(buffer);
      if (found != leases_->end()) return found->second.shape;
    }
    return torchs()->getBufferShape(buffer);
  }
  id_t getBufferId(const void *buffer) const override {
    {
      std::lock_guard<std::mutex> guard(*lock_);
      auto found = leases_->find(buffer);
      if (found != leases_->end()) return found->second.id;
    }
    return torchs()->getBufferId(buffer);
  }
  void setBufferShape(const void *buffer, const c10::IntArrayRef &shape) const override {
    {
      std::lock_guard<std::mutex> guard(*lock_);
      auto found = leases_->find(buffer);
      if (found != leases_->end()) {
        found->second.shape = shape.vec();
        return;
      }
    }
    torchs()->setBufferShape(buffer, shape);
  }
  bool isSharedBuffer(const void *buffer) const override { return ours(buffer) || torchs()->isSharedBuffer(buffer); }
  c10::DataPtr allocScalarBufferWithValue(void *value, size_t bytes) const override {
    c10::DataPtr scalar = lease(bytes);
    std::memcpy([(__bridge id<MTLBuffer>)scalar.get() contents], value, bytes);
    return scalar;
  }
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
  std::pair<const void *, uint32_t> getSharedBufferPtr(const void *buffer) const override {
    if (!ours(buffer)) return torchs()->getSharedBufferPtr(buffer);
    id<MTLBuffer> metal = (__bridge id<MTLBuffer>)buffer;
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
    for (const void *buffer : buffers) {
      if (!ours(buffer)) {
        theirs.push_back(buffer);
        continue;
      }
      std::shared_ptr<at::mps::MPSEvent> event = at::mps::getMPSEventPool()->acquireEvent(false, stream);
      event->record(!on_queue(stream));
      std::lock_guard<std::mutex> guard(*lock_);
      auto found = leases_->find(buffer);
      if (found != leases_->end()) found->second.event = std::move(event);
      recorded = true;
    }
    return (!theirs.empty() && torchs()->recordEvents(theirs)) || recorded;
  }
  bool waitForEvents(c10::ArrayRef<const void *> buffers) const override {
    std::vector<const void *> theirs;
    bool waited = false;
    for (const void *buffer : buffers) {
      std::shared_ptr<at::mps::MPSEvent> event;
      {
        std::lock_guard<std::mutex> guard(*lock_);
        auto found = leases_->find(buffer);
        if (found == leases_->end()) {
          theirs.push_back(buffer);
          continue;
        }
        event = found->second.event;
      }
      if (event) waited |= event->synchronize();
    }
    return (!theirs.empty() && torchs()->waitForEvents(theirs)) || waited;
  }
};

static_assert(sizeof(RegionAllocator) == sizeof(at::mps::IMPSAllocator) && sizeof(at::mps::IMPSAllocator) == sizeof(void *),
              "the region allocator is constructed over torch's instance, whose vtable pointer alone it replaces");

} // namespace

// The region allocator in place of torch's MPS allocator instance, which must be of torch's class: its vtable pointer
// that of `vtable for at::mps::MPSAllocator` (the address point, past the offset to top and the type info), the layout
// the delegate's copy reads.
void install_region_allocator() {
  at::mps::IMPSAllocator *instance = at::mps::getIMPSAllocator();
  const uintptr_t vtable = local_symbol(reinterpret_cast<const void *>(&at::mps::getIMPSAllocator), "__ZTVN2at3mps12MPSAllocatorE");
  const uintptr_t pointer = *reinterpret_cast<const uintptr_t *>(instance);
  TORCH_CHECK(vtable, "torch_mesh: the region allocator takes the place of torch's MPS allocator instance only where it is "
              "of torch's class, and libtorch_cpu has no symbol `vtable for at::mps::MPSAllocator` to check it by (allocator.mm)");
  TORCH_CHECK(pointer == vtable + 2 * sizeof(void *), "torch_mesh: torch's MPS allocator instance (at::mps::getIMPSAllocator) has "
              "vtable pointer ", reinterpret_cast<void *>(pointer), ", not that of torch's class at::mps::MPSAllocator (",
              reinterpret_cast<void *>(vtable + 2 * sizeof(void *)), "), so the region allocator cannot take its place (allocator.mm)");
  std::memcpy(torchs_, static_cast<const void *>(instance), sizeof torchs_);
  new (instance) RegionAllocator();
}

// The region allocator's leases live now: each its address and the bytes asked of it.
std::vector<std::pair<uint64_t, uint64_t>> region_allocations() {
  std::lock_guard<std::mutex> guard(*lock_);
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (const auto &[buffer, lease] : *leases_) out.emplace_back((uint64_t)(uintptr_t)lease.memory, lease.bytes);
  return out;
}
