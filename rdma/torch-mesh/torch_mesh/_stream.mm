#include "mesh-write-compiler.h"
#include "mesh-plan.h"
#include "layout.h"
// PyTorch frontend bindings to the common collective compiler and native
// Metal schedule. Prepared calls resolve routes, pages and dispatch operands
// before submission. The compatibility MPS stream is flushed at a whole
// function boundary; it has no publication-driven buffering policy.
#include <c10/core/CPUAllocator.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSEvent.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <array>
#include <vector>
#include <unistd.h>
#define MESH_SCALAR_PREPARATION
#include "matrix_operand.metal"
#include "layout.metal"
#undef MESH_SCALAR_PREPARATION
#include "metal-recording.h"
#include "metal-page-effects.h"
#include "nccl.h"
#include "layout.h" /* MESH_PREPARATION_NS: the preparation deadline a kernel compile waits within */

// allocator.mm: torch's MPS allocator made the region's, and its leases live now (address, bytes)
void install_region_allocator();
std::vector<std::pair<uint64_t, uint64_t>> region_allocations();

// Typed immutable scalar evaluation during preparation. Address/support
// relations are caller operands; all numerical conversions below come from
// the same source compiled into the matrix reader and layout shader.
template<typename T> struct ScalarTypeTag { using type=T; };
template<typename Function>
static std::string scalar_type(at::ScalarType type,const Function &function) {
#define MESH_SCALAR_TYPE(Name,Type) case at::ScalarType::Name:return function(ScalarTypeTag<Type>{});
  switch(type) {
    MESH_SCALAR_TYPE(Bool,bool)
    MESH_SCALAR_TYPE(Byte,uint8_t)
    MESH_SCALAR_TYPE(Char,int8_t)
    MESH_SCALAR_TYPE(Short,int16_t)
    MESH_SCALAR_TYPE(UInt16,uint16_t)
    MESH_SCALAR_TYPE(Int,int32_t)
    MESH_SCALAR_TYPE(UInt32,uint32_t)
    MESH_SCALAR_TYPE(Long,int64_t)
    MESH_SCALAR_TYPE(UInt64,uint64_t)
    MESH_SCALAR_TYPE(Half,c10::Half)
    MESH_SCALAR_TYPE(BFloat16,c10::BFloat16)
    MESH_SCALAR_TYPE(Float,float)
    MESH_SCALAR_TYPE(Double,double)
    MESH_SCALAR_TYPE(Float8_e4m3fn,c10::Float8_e4m3fn)
    MESH_SCALAR_TYPE(Float8_e5m2,c10::Float8_e5m2)
    default:TORCH_CHECK(false,"preparation scalar reader has no declared scalar type ",type);
  }
#undef MESH_SCALAR_TYPE
}
template<typename T> static T scalar_bytes(const std::string &bytes,size_t offset) {
  T value;
  __builtin_memcpy(&value,bytes.data()+offset,sizeof value);
  return value;
}
template<bool Signed>
static int scalar_packed(uint8_t byte,unsigned position,unsigned bits) {
#define MESH_SCALAR_BITS(Bits) case Bits:return mesh_matrix_bits<Bits,Signed>(byte,position);
  switch(bits) {
    MESH_SCALAR_BITS(1) MESH_SCALAR_BITS(2) MESH_SCALAR_BITS(3) MESH_SCALAR_BITS(4)
    MESH_SCALAR_BITS(5) MESH_SCALAR_BITS(6) MESH_SCALAR_BITS(7) MESH_SCALAR_BITS(8)
    default:TORCH_CHECK(false,"packed preparation scalar width must be between 1 and 8 bits");
  }
#undef MESH_SCALAR_BITS
}
static pybind11::bytes scalar_read(const pybind11::bytes &source,at::ScalarType source_type,
    at::ScalarType output_type,const std::vector<int64_t> &offsets,unsigned bits,bool signed_code,
    const std::optional<pybind11::bytes> &scales,const std::optional<pybind11::bytes> &zeros,
    int64_t absent,bool bitcast,const std::optional<std::vector<bool>> &present) {
  const std::string bytes=source,scale_bytes=scales?std::string(*scales):std::string(),
    zero_bytes=zeros?std::string(*zeros):std::string();
  const size_t count=offsets.size(),width=c10::elementSize(source_type);
  TORCH_CHECK(!bits || bits<=8,"packed preparation scalar width must be between 1 and 8 bits");
  TORCH_CHECK(!present || present->size()==count,"support mask must match the declared output order");
  TORCH_CHECK(count<=SIZE_MAX/sizeof(int64_t),"preparation scalar count exceeds address space");
  TORCH_CHECK(!scales || scale_bytes.size()==count*sizeof(float),"scales must be output-ordered FLOAT32 bytes");
  TORCH_CHECK(!zeros || zero_bytes.size()==count*sizeof(int64_t),"zeros must be output-ordered INT64 bytes");
  TORCH_CHECK(!bitcast || (!bits&&!scales&&!zeros&&width==c10::elementSize(output_type)),
    "bitcast requires ordinary equal-width scalar types without quantization");
  for(size_t i=0;i<count;i++) {
    TORCH_CHECK(offsets[i]>=0,"bit positions must be nonnegative; support is a separate operand");
    if(present&&!(*present)[i])continue;
    const uint64_t at=(uint64_t)offsets[i]/8,lane=(uint64_t)offsets[i]%8;
    TORCH_CHECK(bits?lane+bits<=8:lane==0,"scalar field crosses its declared byte boundary");
    const size_t bytes_needed=bits?1:width;
    TORCH_CHECK(at<=bytes.size()&&bytes_needed<=bytes.size()-at,"scalar field exceeds supplied initializer bytes");
  }
  const std::string result=scalar_type(source_type,[&](auto input_tag) {
    using Input=typename decltype(input_tag)::type;
    return scalar_type(output_type,[&](auto output_tag) {
      using Output=typename decltype(output_tag)::type;
      TORCH_CHECK(count<=SIZE_MAX/sizeof(Output),"preparation scalar output exceeds address space");
      std::string made(count*sizeof(Output),'\0');
      for(size_t i=0;i<count;i++) {
        const bool support=!present||(*present)[i];
        const size_t at=(uint64_t)offsets[i]/8;
        const float scale=scales?scalar_bytes<float>(scale_bytes,i*sizeof(float)):1.0f;
        const int64_t zero=zeros?scalar_bytes<int64_t>(zero_bytes,i*sizeof(int64_t)):0;
        Output value;
        if(bits) {
          const int code=support?(signed_code?scalar_packed<true>((uint8_t)bytes[at],offsets[i]%8,bits):
                                               scalar_packed<false>((uint8_t)bytes[at],offsets[i]%8,bits)):
                                  mesh_layout_cast<int>(absent);
          value=scales||zeros?mesh_matrix_scaled<Output>(code,scale,zero):mesh_layout_cast<Output>(code);
        } else {
          const Input code=support?scalar_bytes<Input>(bytes,at):mesh_layout_cast<Input>(absent);
          if(bitcast) {
            if constexpr(sizeof(Input)==sizeof(Output))value=mesh_layout_bitcast<Output>(code);
          } else value=scales||zeros?mesh_matrix_scaled<Output>(code,scale,zero):mesh_layout_cast<Output>(code);
        }
        __builtin_memcpy(made.data()+i*sizeof(Output),&value,sizeof value);
      }
      return made;
    });
  });
  return pybind11::bytes(result);
}

// Framework stream dependencies and compatibility-call storage. The shared
// native schedule owns prepared execution; this state names MPS data events.
struct MeshStream {
  at::mps::MPSStream *const mps;
  id<MTLSharedEvent> const progress;
  id<MTL4CommandQueue> const prepared;
  // the buffers this stream committed, the k-th signalling k on progress as the GPU reaches its end
  uint64_t committed = 0;

  explicit MeshStream(at::mps::MPSStream *stream) : mps(stream), progress([stream->device() newSharedEvent]),
      prepared([stream->device() newMTL4CommandQueue]) {
    if (!progress || !prepared) {
      [progress release]; [prepared release];
      TORCH_CHECK(false,"preparing Metal context: shared event or command queue unavailable");
    }
  }

  // On torch's queue: its work so far committed (and waited for, where `wait`), torch's own event after it (MPSEvent)
  // notifying `progress` of the next count for consumers of its declared external input pages.  Notifications
  // may come in any order; `progress` only rises.
  std::mutex rising;
  void signalled(bool wait) {
    const uint64_t value = ++committed;
    id<MTLSharedEvent> event = progress;
    std::mutex *order = &rising;
    std::shared_ptr<at::mps::MPSEvent> ended = at::mps::getMPSEventPool()->acquireEvent(false, mps);
    ended->record(false);
    ended->notify(false, ^(id<MTLSharedEvent>, uint64_t) {
      std::lock_guard<std::mutex> guard(*order);
      if (event.signaledValue < value) event.signaledValue = value;
      (void)ended;
    });
    mps->synchronize(wait ? at::mps::SyncType::COMMIT_AND_WAIT : at::mps::SyncType::COMMIT);
  }
};

static std::mutex *streams_lock_ = new std::mutex;
static auto *streams_ = new std::unordered_map<at::mps::MPSStream *, MeshStream *>;

// torch's stream's MeshStream, made at its first use.
static MeshStream &stream_of(at::mps::MPSStream *mps) {
  thread_local std::pair<at::mps::MPSStream *, MeshStream *> last{nullptr, nullptr};
  if (last.first == mps) return *last.second;
  std::lock_guard<std::mutex> guard(*streams_lock_);
  auto found = streams_->find(mps);
  if (found == streams_->end()) found = streams_->emplace(mps, new MeshStream(mps)).first;
  last = {mps, found->second};
  return *found->second;
}

// Every MeshStream made so far: those of the torch streams a group was issued on (none before any was, so the host
// path makes no MPS stream).
static std::vector<MeshStream *> streams_made() {
  std::lock_guard<std::mutex> guard(*streams_lock_);
  std::vector<MeshStream *> made;
  for (auto &[mps, s] : *streams_) made.push_back(s);
  return made;
}

// A host wait of the library's on stream s, on torch's queue: a publication still waiting committed, and with `wait`
// the stream committed and waited for (torch's COMMIT_AND_WAIT, taken here on its queue), on the stream queue.
static void host_wait(MeshStream &s, bool wait) {
  MeshStream *self = &s;
  at::mps::dispatch_sync_with_rethrow(s.mps->queue(), ^{
    @autoreleasepool {
      self->signalled(wait);
    }
  });
}

// Before a host wait outside torch's streams (a recording's compile meeting): every stream's waiting
// publication committed.
static void commit() {
  for (MeshStream *s : streams_made()) host_wait(*s, false);
}

// Configuration-time migration may read existing tensors produced by ordinary
// GPU stores. End that producer command buffer to establish NIC visibility;
// enqueue the next buffer on the same serial queue, without a CPU/GPU wait.
static void publish_boundary() {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{ s->signalled(false); });
}


static void checked(int64_t result) {
  TORCH_CHECK(!result, "libnccl-mesh: ", ncclGetErrorString((ncclResult_t)result), ": ", ncclGetLastError(nullptr));
}

static ncclComm_t communicator(int64_t comm) { return (ncclComm_t)(uintptr_t)comm; }

static at::Tensor region_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps);
static MetalTensorPages *tensorPages(const at::Tensor &t);

using Tensors = std::vector<std::optional<at::Tensor>>;

// A native host entry (prepare, context, retain, release) as the recorder's host declaration, the storage it reads held
// with it; `owned`: the caller's reference to its context passes to the operation, else the operation takes its own.
static MetalHostOperation host_operation(const std::array<uintptr_t,4> &entry, bool owned, std::vector<at::Tensor> held) {
  for (auto pointer : entry) TORCH_CHECK(pointer, "host operation requires a complete native binding");
  using Lifetime = void (*)(void *);
  auto prepare = reinterpret_cast<MeshHostPrepare>(entry[0]);
  auto context = reinterpret_cast<void *>(entry[1]);
  if (!owned) reinterpret_cast<Lifetime>(entry[2])(context);
  auto bound = std::shared_ptr<void>(context, reinterpret_cast<Lifetime>(entry[3]));
  MetalHostOperation operation = ^const char *(MeshHostCompletion complete,void *context,MeshHostTask *task) {
    // Tensor handles retain the actual storage, including CPU-only weights.
    (void)held;
    return prepare(bound.get(),complete,context,task);
  };
  return [[operation copy] autorelease];
}

using Schedule = std::shared_ptr<MetalSchedule>;

static Schedule prepared_schedule() {
  at::mps::MPSStream *stream=at::mps::getCurrentMPSStream();
  MeshStream *source=&stream_of(stream);
  Schedule schedule(MetalScheduleCreateOn(source->prepared,MetalFormEncoded),MetalScheduleFree);
  TORCH_CHECK(schedule,"creating the prepared function command stream");
  return schedule;
}

static pybind11::capsule schedule_completion(const Schedule &schedule) {
  return pybind11::capsule(new Schedule(schedule),"mesh.prepared-function",[](PyObject *token) {
    delete static_cast<Schedule *>(PyCapsule_GetPointer(token,"mesh.prepared-function"));
  });
}

// each mesh process group's communicator, found by its group (attach) with no Python between (collective declaration)
static std::unordered_map<const c10d::ProcessGroup *, std::pair<c10::weak_intrusive_ptr<c10d::ProcessGroup>, int64_t>> communicators_;

static void attach(const c10::intrusive_ptr<c10d::ProcessGroup> &group, int64_t comm) {
  communicators_.insert_or_assign(group.get(), std::make_pair(c10::weak_intrusive_ptr<c10d::ProcessGroup>(group), comm));
}

static int64_t communicator_of(const c10::IValue &name) {
  auto group = c10d::resolve_process_group(name.toStringRef());
  auto found = communicators_.find(group.get());
  if (found == communicators_.end()) return 0;
  if (found->second.first.lock() != group) {
    communicators_.erase(found);
    return 0;
  }
  return found->second.second;
}

// The mesh library's own kernels on torch's stream (torch_mesh/sparse.py): a kernel compiled from its source at Metal
// 4.0 (tensor operations), and a dispatch of it encoded in torch's open compute encoder on its queue, ordered with
// torch's own work and waited on by no host: buffers by index (a tensor's storage at its offset), 32-bit constants at
// one index, the grid in threadgroups.
// A kernel compiles from the moment it is asked for, on the Metal compiler service, so a preparation's compiles run
// together across the node's cores instead of one after another; its first use (state_of) waits for it, within the
// preparation's deadline (MESH_PREPARATION_NS), and raises its compile error there.
struct Compiled { dispatch_semaphore_t done; std::atomic<bool> ready{false}; id<MTLComputePipelineState> state = nil; std::string error; };
static std::vector<std::shared_ptr<Compiled>> pipelines_;
static std::mutex pipelines_lock_;

static int64_t pipeline(const std::string &source, const std::string &name) {
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  MTLCompileOptions *options = MetalDeviceCompilerOptions(device);
  auto made = std::make_shared<Compiled>();
  made->done = dispatch_semaphore_create(0);
  const std::string function_name = name;
  [device newLibraryWithSource:@(source.c_str()) options:options completionHandler:^(id<MTLLibrary> library, NSError *error) {
    NSError *failed = nil;
    // Arithmetic pipeline preparation uses its retained source declaration.
    id<MTLComputePipelineState> state = library ? MetalPipelineMake(library, @(function_name.c_str()), @[], nil, nil, &failed) : nil;
    if (!library) made->error = std::string("kernel source: ") + error.localizedDescription.UTF8String;
    else if (!state) made->error = "kernel pipeline " + function_name + ": " + failed.localizedDescription.UTF8String;
    made->state = state;
    dispatch_semaphore_signal(made->done);
  }];
  std::lock_guard<std::mutex> guard(pipelines_lock_);
  pipelines_.push_back(made);
  return (int64_t)pipelines_.size() - 1;
}

// design/compile-once-2026-10-29.md: one library of a source per process, compiled once on the compiler service, and
// each command's pipeline specialized from it by its layout's function constants ([index, type, value]: 0 bool, 1 uint,
// 2 long, 3 ulong), so commands that differ only in layout share one compile.
struct Library { dispatch_semaphore_t done; id<MTLLibrary> library = nil; std::string error; };
static std::unordered_map<std::string, std::shared_ptr<Library>> libraries_;

static std::shared_ptr<Library> library_of(const std::string &source) {
  std::lock_guard<std::mutex> guard(pipelines_lock_);
  auto &held = libraries_[source];
  if (held) return held;
  held = std::make_shared<Library>();
  held->done = dispatch_semaphore_create(0);
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  auto made = held;
  [device newLibraryWithSource:@(source.c_str()) options:MetalDeviceCompilerOptions(device) completionHandler:^(id<MTLLibrary> library, NSError *error) {
    if (library) made->library = [library retain];
    else made->error = std::string("kernel source: ") + error.localizedDescription.UTF8String;
    dispatch_semaphore_signal(made->done);
  }];
  return held;
}

// `made` specialized from `library` once it is compiled: function `name` with `constants` ([index, type, value])
static void specialize(std::shared_ptr<Library> library, const std::string &name, const std::vector<std::array<int64_t, 3>> &constants,
                       std::shared_ptr<Compiled> made) {
  const std::string function_name = name;
  const std::vector<std::array<int64_t, 3>> values = constants;
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    @autoreleasepool {
      if (dispatch_semaphore_wait(library->done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)MESH_PREPARATION_NS))) {
        made->error = "kernel source: not compiled within the preparation's deadline";
      } else {
        dispatch_semaphore_signal(library->done);   /* the library stays ready for every later specialization */
        if (!library->library) made->error = library->error;
        else {
          // Each constant retains [index, MTLDataType, bytes] in the pipeline declaration.
          NSMutableArray *constants = [NSMutableArray array];
          static const MTLDataType types[4] = {MTLDataTypeBool, MTLDataTypeUInt, MTLDataTypeLong, MTLDataTypeULong};
          for (const auto &[index, type, value] : values) {
            const bool flag = value != 0; const uint32_t word = (uint32_t)value; const int64_t wide = value; const uint64_t unsigned_wide = (uint64_t)value;
            const void *at = type == 0 ? (const void *)&flag : type == 1 ? (const void *)&word : type == 2 ? (const void *)&wide : (const void *)&unsigned_wide;
            const size_t bytes = type == 0 ? sizeof flag : type == 1 ? sizeof word : sizeof wide;
            [constants addObject:@[@(index), @(types[type & 3]), [NSData dataWithBytes:at length:bytes]]];
          }
          NSError *error = nil;
          id<MTLComputePipelineState> state = MetalPipelineMake(library->library, @(function_name.c_str()), constants, nil, nil, &error);
          if (!state) made->error = "kernel pipeline " + function_name + (error ? std::string(": ") + error.localizedDescription.UTF8String : "");
          made->state = state;
        }
      }
      dispatch_semaphore_signal(made->done);
    }
  });
}

// A specialized pipeline is its source's text without instances (`base`) and its instance line. The pipelines of a base
// not yet compiled wait in its batch; the first of them a caller needs (state_of) compiles one library of the base and
// every instance line waiting, and each is specialized from it. So a source is parsed once for all the instances a
// preparation names before it reads a pipeline (design/compile-once-2026-10-29.md).
struct Waiting { std::string instance, name; std::vector<std::array<int64_t, 3>> constants; std::shared_ptr<Compiled> made; };
static std::unordered_map<std::string, std::vector<Waiting>> batches_;
static std::unordered_map<const Compiled *, std::string> waiting_on_;

static int64_t pipeline_specialized(const std::string &base, const std::string &instance, const std::string &name,
                                    const std::vector<std::array<int64_t, 3>> &constants) {
  auto made = std::make_shared<Compiled>();
  made->done = dispatch_semaphore_create(0);
  std::lock_guard<std::mutex> guard(pipelines_lock_);
  batches_[base].push_back({instance, name, constants, made});
  waiting_on_[made.get()] = base;
  pipelines_.push_back(made);
  return (int64_t)pipelines_.size() - 1;
}

// the batch `made` waits in compiled as one library and specialized; nothing where it waits in none
static void batch_compile(const Compiled *made) {
  std::string base;
  std::vector<Waiting> batch;
  {
    std::lock_guard<std::mutex> guard(pipelines_lock_);
    const auto found = waiting_on_.find(made);
    if (found == waiting_on_.end()) return;
    base = found->second;
    batch = std::move(batches_[base]);
    batches_.erase(base);
    for (const auto &w : batch) waiting_on_.erase(w.made.get());
  }
  std::string text = base + "\n";
  std::unordered_set<std::string> lines;
  for (const auto &w : batch) if (lines.insert(w.instance).second) text += w.instance + "\n";
  auto library = library_of(text);
  for (const auto &w : batch) specialize(library, w.name, w.constants, w.made);
}

static id<MTLComputePipelineState> state_of(int64_t handle, const char *what) {
  std::shared_ptr<Compiled> made;
  {
    std::lock_guard<std::mutex> guard(pipelines_lock_);
    if (handle < 0 || handle >= (int64_t)pipelines_.size()) throw std::runtime_error(std::string(what) + ": no kernel " + std::to_string(handle));
    made = pipelines_[(size_t)handle];
  }
  if (!made->ready.load(std::memory_order_acquire)) {
    batch_compile(made.get());
    if (dispatch_semaphore_wait(made->done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)MESH_PREPARATION_NS)))
      throw std::runtime_error(std::string(what) + ": kernel " + std::to_string(handle) + " not compiled within the preparation's deadline");
    made->ready.store(true, std::memory_order_release);
  }
  if (!made->state) throw std::runtime_error(made->error);
  return made->state;
}

// What Metal reports of the device a kernel runs on (its MTLDevice: its Apple GPU family, the n of the newest
// MTLGPUFamilyApple<n> it supports, asked from Apple1 up, since a GPU of a family supports every earlier one's ("a higher
// GPU version is always a superset of an earlier version in the same GPU family": Apple, Detecting GPU features and Metal
// software versions); maxThreadsPerThreadgroup's width; maxThreadgroupMemoryLength; in that order) and of a compiled
// kernel (its pipeline: maxTotalThreadsPerThreadgroup, which its registers may hold below the device's,
// threadExecutionWidth, staticThreadgroupMemoryLength, in that order): the facts a specializer selects and checks its
// configuration by, and reads where no document states them.
static std::tuple<int64_t, int64_t, int64_t> device_facts() {
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  int64_t family = 0;
  while ([device supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + family)]) family++;
  return {family, (int64_t)device.maxThreadsPerThreadgroup.width, (int64_t)device.maxThreadgroupMemoryLength};
}

static std::tuple<int64_t, int64_t, int64_t> pipeline_facts(int64_t handle) {
  id<MTLComputePipelineState> state = state_of(handle, "pipeline_facts");
  return {(int64_t)state.maxTotalThreadsPerThreadgroup, (int64_t)state.threadExecutionWidth,
          (int64_t)state.staticThreadgroupMemoryLength};
}

// A kernel that runs a fused crossing's steps (`fused`) reaches the library's counts, landing buffers and records by
// address, through the residency set, which Metal does not hazard-track: its encoder declares every buffer of the set,
// so its pass is ordered against the library's dispatches that name them (nccl-mesh-metal.h metal_uses).
// A dispatch made ready on the host (encoded): its pipeline, its buffers bound at their offsets, its constants' words
// set at `at`, its grid; encoded into an open encoder on torch's queue (encode_in)
struct Dispatch {
  id<MTLComputePipelineState> state;
  std::vector<id<MTLBuffer>> bound;
  std::vector<NSUInteger> offsets, lengths;
  std::vector<uint32_t> words;
  NSUInteger at;
  MTLSize grid, group;
};
// A word a kernel reads as uint (or, two's complement, int): a stride, offset, count or bound past 32 bits does not
// narrow silently into another value.
static uint32_t word_of(int64_t value, size_t k) {
  TORCH_CHECK(value >= INT32_MIN && value <= (int64_t)UINT32_MAX, "a kernel's word ", k, " is ", value,
              ", which no 32-bit word holds: an offset, stride or bound past it would address another element");
  return (uint32_t)value;
}
static Dispatch encoded(int64_t handle, const std::vector<std::optional<at::Tensor>> &buffers, const std::vector<int64_t> &constants,
                        int64_t at, int64_t gx, int64_t gy, int64_t threads) {
  std::vector<uint32_t> words(constants.size());
  for (size_t k = 0; k < constants.size(); k++) words[k] = word_of(constants[k], k);
  Dispatch d{state_of(handle, "encode"), std::vector<id<MTLBuffer>>(buffers.size(), nil), std::vector<NSUInteger>(buffers.size(), 0), std::vector<NSUInteger>(buffers.size(), 0),
             words, (NSUInteger)at, MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, 1),
             MTLSizeMake((NSUInteger)threads, 1, 1)};
  for (size_t i = 0; i < buffers.size(); i++)
    if (buffers[i]) {
      const at::Tensor &t = *buffers[i];
      if (!t.is_mps()) throw std::runtime_error("encode: buffer " + std::to_string(i) + " is not an MPS tensor");
      d.bound[i] = at::native::mps::getMTLBufferStorage(t);
      d.offsets[i] = (NSUInteger)(t.storage_offset() * t.element_size());
      d.lengths[i] = t.nbytes();
    }
  return d;
}

// The shared preparation layout is also the Python frontend's layout type.
// Its object and maps never enter a replay record.
static NSArray<NSNumber *> *tensorNumbers(const std::vector<NSInteger> &values) {
  NSMutableArray *result=[NSMutableArray array];for(auto value:values)[result addObject:@(value)];return result;
}
static std::vector<NSInteger> tensorNumbers(NSArray<NSNumber *> *values) {
  std::vector<NSInteger> result;for(NSNumber *value in values)result.push_back(value.integerValue);return result;
}
static NSData *tensorWords(const std::vector<NSInteger> &values) {
  return [NSData dataWithBytes:values.data() length:values.size()*sizeof(NSInteger)];
}
struct TensorLayout {
  MetalTensorLayout *value;
  explicit TensorLayout(MetalTensorLayout *layout):value([layout retain]){}
  TensorLayout(const TensorLayout &other):value([other.value retain]){}
  TensorLayout(TensorLayout &&other):value(other.value){other.value=nil;}
  ~TensorLayout(){[value release];}
  static TensorLayout make(const std::vector<NSInteger> &shape,const std::vector<NSInteger> &strides,NSInteger offset,
      const std::map<NSUInteger,std::vector<NSInteger>> &axes,const std::optional<std::vector<NSInteger>> &indices,
      const std::optional<std::vector<NSInteger>> &indexShape,const std::optional<std::string> &coordinates) {
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];
    for(const auto &[axis,values]:axes)maps[@(axis)]=tensorWords(values);
    MetalTensorLayout *layout=[MetalTensorLayout shape:tensorNumbers(indexShape?*indexShape:shape)
        strides:tensorNumbers(strides) offset:offset maps:maps indices:indices?tensorWords(*indices):nil];
    if(coordinates) {
      NSData *data=[NSData dataWithBytes:coordinates->data() length:coordinates->size()];
      NSArray *values=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
      return TensorLayout([layout shaped:tensorNumbers(shape) coordinates:values]);
    }
    return TensorLayout([layout reshape:tensorNumbers(shape)]);
  }
  static TensorLayout tensor(const at::Tensor &t) {
    return make({t.sizes().begin(),t.sizes().end()},{t.strides().begin(),t.strides().end()},t.storage_offset(),{},std::nullopt,std::nullopt,std::nullopt);
  }
};
struct TensorTraversal {
  MetalTensorTraversal *value;
  TensorTraversal(const std::vector<TensorLayout> &layouts,const TensorLayout &domain,const TensorLayout &selection) {
    NSMutableArray *parts=[NSMutableArray array];
    for(const auto &layout:layouts)[parts addObject:layout.value];
    @try { value=[[MetalTensorTraversal layouts:parts domain:domain.value selection:selection.value] retain]; }
    @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
  }
  TensorTraversal(const TensorTraversal &other):value([other.value retain]){}
  ~TensorTraversal(){[value release];}
};
// Resolve a frontend storage handle to its bridge pages once, while preparing.
// A tensor containing changing page indices remains an ordinary data operand;
// neither this adapter nor TensorLayout reads those indices from its payload.
static MetalTensorPages *tensorPages(const at::Tensor &t) {
  TORCH_CHECK(t.is_mps()||t.device().is_cpu(),"page operands require CPU or Metal views of bridge storage");
  @try {
    return [(t.is_mps()?[MetalTensorPages storage:at::native::mps::getMTLBufferStorage(t)]
                       :[MetalTensorPages host:t.storage().data_ptr().get()]) retain];
  } @catch(NSException *error) { TORCH_CHECK(false,error.reason.UTF8String); }
}
struct TensorOperand {
  at::Tensor storage;
  MetalTensorPages *pages;
  TensorLayout layout;
  NSUInteger origin;
  explicit TensorOperand(const pybind11::object &object):
    storage((THPVariable_Check(object.ptr())?object:object.attr("storage")).cast<at::Tensor>()),
    pages(tensorPages(storage)),
    layout([pages bind:(THPVariable_Check(object.ptr())?TensorLayout::tensor(storage):object.attr("layout").cast<TensorLayout>()).value
                 elementBytes:storage.element_size()]),
    origin(THPVariable_Check(object.ptr())||!pybind11::hasattr(object,"binding_origin")||object.attr("binding_origin").is_none()
      ?(NSUInteger)layout.value.offset*storage.element_size():object.attr("binding_origin").cast<NSUInteger>()){}
  ~TensorOperand(){[pages release];}
};

// These preparation declarations retain the tensor leases behind their page
// operands. Final emitted source owns them; no Python object or lookup remains
// in a numerical entry.
@interface PythonPageCoverage : MetalPageCoverage {
@public std::vector<at::Tensor> tensors;
}
@end
@implementation PythonPageCoverage
@end
@interface PythonPageContribution : MetalPageContribution {
@public std::vector<at::Tensor> tensors;
}
@end
@implementation PythonPageContribution
@end

static PythonPageCoverage *page_coverage(const pybind11::handle &value) {
  const TensorOperand active(pybind11::reinterpret_borrow<pybind11::object>(value.attr("active")));
  const TensorOperand expected(pybind11::reinterpret_borrow<pybind11::object>(value.attr("expected")));
  auto *made=[PythonPageCoverage declaration:(ncclMeshPageCoverage){
    {active.pages.buffer,active.origin},{expected.pages.buffer,expected.origin},value.attr("cuts").cast<size_t>()}];
  made->tensors={active.storage,expected.storage};return made;
}

// Import only the pages produced by the framework before preparation.  The
// recorder connects their event to intersecting consumers; unrelated commands
// have no dependency on the framework queue.
static void schedule_input_reaches(MetalSchedule *schedule,const std::vector<MetalReach> &inputs) {
  std::vector<MetalReach> reaches;
  for(auto reach:inputs)if(reach.access&MetalReachRead) {
    reach.access=MetalReachWritten;reach.binding=-1;reaches.push_back(reach);
  }
  if(reaches.empty())return;
  auto *stream=at::mps::getCurrentMPSStream();auto *source=&stream_of(stream);
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{ source->signalled(false); });
  MetalScheduleInput(schedule,reaches.data(),reaches.size(),source->progress,source->committed);
}
static void schedule_inputs(MetalSchedule *schedule,const std::vector<pybind11::object> &inputs) {
  std::vector<MetalReach> reaches;
  for(const auto &value:inputs) {
    const TensorOperand operand(value);
    if(!operand.storage.is_mps())continue;
    NSData *data=[operand.layout.value reaches:operand.pages.buffer base:0
      elementBytes:operand.storage.element_size() access:MetalReachRead binding:-1];
    const auto *spans=(const MetalReach *)data.bytes;
    reaches.insert(reaches.end(),spans,spans+data.length/sizeof *spans);
  }
  schedule_input_reaches(schedule,reaches);
}

// The canonical bridge backing is already allocated. This preparation handle
// retains the contributing leases and exposes their one page table to ordinary
// tensor views; execution receives the same root buffer and compiled indices.
static NSArray<MetalTensorPages *> *page_sources(const std::vector<at::Tensor> &owners) {
  NSMutableArray *sources=[NSMutableArray array];
  for(const auto &owner:owners){MetalTensorPages *pages=tensorPages(owner);[sources addObject:pages];[pages release];}
  return sources;
}
static at::Tensor page_storage(const std::vector<at::Tensor> &owners,
    const TensorLayout &order,NSUInteger alignment) {
  TORCH_CHECK(!owners.empty(),"a page backing requires its declared lease operands");
  struct Storage {
    MetalTensorPages *pages;
    ~Storage(){[pages release];}
  };
  auto held=std::make_unique<Storage>();held->pages=nil;
  NSArray<MetalTensorPages *> *sources=page_sources(owners);
  @try {held->pages=[[MetalTensorPages compose:sources order:order.value alignment:alignment] retain];}
  @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
  const bool metal=owners.front().is_mps();
  const auto device=owners.front().device();
  const size_t bytes=held->pages.buffer.length;
  void *pointer=metal?(__bridge void *)held->pages.buffer:held->pages.buffer.contents;
  c10::DataPtr data(pointer,held.release(),[](void *value){delete static_cast<Storage *>(value);},device);
  c10::Storage storage(c10::Storage::use_byte_size_t(),bytes,std::move(data),nullptr,false);
  return at::empty({0},at::TensorOptions().dtype(at::kByte).device(device)).set_(storage,0,{(int64_t)bytes});
}

// FX's output is the native recorder's input. This constructor only binds
// storage and constant operands; it neither runs nor records a framework call.
using CommandRecord = std::tuple<int64_t, std::vector<pybind11::object>,
    std::map<uint32_t,pybind11::bytes>, std::array<NSUInteger,3>, std::array<NSUInteger,3>, bool, uint64_t>;
// A slot's actual selected read/write views are independent of its argument
// address and full binding layout. Both become the existing command ABI here.
using CommandReach = std::tuple<int32_t,uint32_t,pybind11::object>;
using CommandReaches = std::vector<std::optional<std::vector<CommandReach>>>;
// A record (words, constants) is refused short of its declared type; a buffer, holding part of one declared element: an
// array of none is a ragged share of nothing (a rank given no columns).
static void fits_binding(id<MTLComputePipelineState> state, NSUInteger index, NSUInteger bound, bool record) {
  const NSUInteger declared = MetalPipelineDeclaredBytes(state, index);
  if (declared <= bound || (!record && !bound)) return;
  NSString *name = MetalPipelineName(state);
  TORCH_CHECK(false, "kernel ", name ? name.UTF8String : "(unnamed)", " declares ", declared, " bytes at buffer ", index,
              ", and its dispatch binds ", bound, " bytes of ", record ? "its record" : "a buffer");
}
using CommandHost=std::tuple<size_t,std::array<uintptr_t,4>,std::vector<CommandReach>>;
// Both direct host declarations and mixed command streams retain the same
// page accesses and native entry at preparation.
struct PreparedHost {
  std::vector<MetalReach> views;
  MetalHostOperation operation;
  PreparedHost(const std::array<uintptr_t,4> &entry,const std::vector<CommandReach> &reaches) {
    std::vector<at::Tensor> held;
    for(const auto &[slot,access,object]:reaches) {
      const TensorOperand operand(object);const auto &t=operand.storage;held.push_back(t);
      NSData *data=[operand.layout.value reaches:operand.pages.buffer base:0 elementBytes:t.element_size() access:access binding:slot];
      const auto *spans=(const MetalReach *)data.bytes;
      views.insert(views.end(),spans,spans+data.length/sizeof *spans);
    }
    operation=[host_operation(entry,false,held) copy];
  }
  ~PreparedHost(){[operation release];}
  void append(MetalSchedule *schedule) {
    MetalScheduleHost(schedule,views.data(),views.size(),operation);
  }
};
struct Commands {
  std::vector<MetalReplay *> replays;
  std::vector<std::unique_ptr<PreparedHost>> hosts;
  std::vector<at::Tensor> operands;
  explicit Commands(const std::vector<CommandRecord> &records,const std::optional<CommandReaches> &declared=std::nullopt,
      const std::vector<CommandHost> &hostRecords={}) {
    TORCH_CHECK(!declared||declared->size()==records.size(),"each command has one declared access list");
    if(records.empty()&&hostRecords.empty())return;
    auto device=at::mps::MPSDevice::getInstance()->device();
    std::vector<std::vector<MetalCommandBinding>> bindings(records.size());
    std::vector<std::vector<MetalReach>> reaches(records.size());
    std::vector<MetalCommandDescription> descriptions;
    std::vector<uint8_t> words;
    std::vector<std::vector<std::array<NSUInteger,3>>> constants(records.size());
    for(size_t k=0;k<records.size();k++)for(const auto &[index,scalar]:std::get<2>(records[k])) {
      const std::string bytes=scalar;
      words.resize((words.size()+MESH_COMMAND_DATA_ALIGNMENT-1)/MESH_COMMAND_DATA_ALIGNMENT*MESH_COMMAND_DATA_ALIGNMENT);
      constants[k].push_back({index,words.size(),bytes.size()});
      words.insert(words.end(),bytes.begin(),bytes.end());
    }
    id<MTLBuffer> constantBuffer=nil;size_t constantOffset=0;
    if(!words.empty()) {
      auto storage=region_empty({(int64_t)words.size()},at::empty({0},at::TensorOptions().dtype(at::kByte)),false);
      memcpy(storage.data_ptr(),words.data(),words.size());
      void *mapped=nullptr;checked(ncclMeshMemBuffer(storage.data_ptr(),&mapped,&constantOffset));
      constantBuffer=(id<MTLBuffer>)mapped;operands.push_back(std::move(storage));
    }
    for(size_t k=0;k<records.size();k++) {
      const auto &[handle,buffers,scalars,grid,group,threadGrid,access]=records[k];

      auto &bound=bindings[k];
      auto &ranges=reaches[k];
      const auto *selected=declared&&(*declared)[k]?&*(*declared)[k]:nullptr;
      for(size_t j=0;j<buffers.size();j++)if(!buffers[j].is_none()) {
        const TensorOperand operand(buffers[j]);const auto &t=operand.storage;const auto &layout=operand.layout;
        id<MTLBuffer> buffer=operand.pages.buffer;
        const uint32_t mode=((access>>j)&1?MetalReachRead:0)|((access>>(MESH_ACCESS_WRITE_SHIFT+j))&1?MetalReachWritten:0);
        bound.push_back({buffer,operand.origin,layout.value.count*t.element_size(),(uint32_t)j,mode,0});
        if(mode) {
          NSData *data=[layout.value reaches:buffer base:0 elementBytes:t.element_size()
            access:MetalReachBinding|(selected?MetalReachResident:mode) binding:(int32_t)j];
          const auto *spans=(const MetalReach *)data.bytes;
          ranges.insert(ranges.end(),spans,spans+data.length/sizeof *spans);
        }
        operands.push_back(t);
      }
      if(selected)for(const auto &[slot,mode,value]:*selected) {
        const TensorOperand operand(value);const auto &t=operand.storage;
        NSData *data=[operand.layout.value reaches:operand.pages.buffer base:0 elementBytes:t.element_size()
          access:(slot>=0?MetalReachBinding:0)|mode binding:slot];
        const auto *spans=(const MetalReach *)data.bytes;ranges.insert(ranges.end(),spans,spans+data.length/sizeof *spans);
        operands.push_back(t);
      }
      for(const auto &constant:constants[k]) {
        bound.push_back({constantBuffer,constantOffset+constant[1],constant[2],(uint32_t)constant[0],MetalReachRead,0});
        ranges.push_back({constantBuffer,constantOffset+constant[1],constant[2],MetalReachBinding|MetalReachRead,(int32_t)constant[0]});
      }
      // every binding the type its kernel declares (fits_binding): constants records, tensors arrays
      const id<MTLComputePipelineState> state=state_of(handle,"commands");
      for(const auto &b:bound)fits_binding(state,b.index,b.length,b.buffer==constantBuffer&&constantBuffer);
      descriptions.push_back({state,bound.data(),bound.size(),nullptr,0,ranges.data(),ranges.size(),nullptr,0,
        MTLSizeMake(grid[0],grid[1],grid[2]),MTLSizeMake(group[0],group[1],group[2]),(BOOL)threadGrid});
    }
    size_t first=0;
    auto replay=[&](size_t end) {
      TORCH_CHECK(first<=end&&end<=descriptions.size(),"host positions must follow declared command order");
      auto *made=first==end?nullptr:MetalReplayCommands(device,descriptions.data()+first,end-first,MetalFormEncoded);
      TORCH_CHECK(first==end||made,"native command preparation failed");
      replays.push_back(made);first=end;
    };
    for(const auto &[at,entry,reaches]:hostRecords) {
      replay(at);hosts.push_back(std::make_unique<PreparedHost>(entry,reaches));
    }
    replay(descriptions.size());
  }
  ~Commands(){for(auto *replay:replays)MetalReplayFree(replay);}
  void append(MetalSchedule *schedule) {
    for(size_t i=0;i<replays.size();i++) {
      if(replays[i]) {
        NSError *error=MetalScheduleAppend(schedule,replays[i],NSMakeRange(0,1),@[],nil,nil);
        TORCH_CHECK(!error,error?error.localizedDescription.UTF8String:"");
      }
      if(i<hosts.size())hosts[i]->append(schedule);
    }
  }
};
// One declaration-owned occurrence for every collective kind. The surrounding
// compiler consumes effects before append records this exact occurrence.
// This object has no executor, completion observer or independent realization.
struct Occurrence {
  ncclComm_t comm=nullptr;
  uint32_t polls=0;
  uint64_t occurrence=UINT64_MAX;
  std::vector<at::Tensor> operands;
  Occurrence()=default;
  Occurrence(int64_t handle,std::vector<at::Tensor> values):comm(communicator(handle)),operands(std::move(values)){}
  void lower(uint64_t invocation=0) {
    checked(ncclMeshCommLower(comm,invocation,polls,nullptr,nullptr,&occurrence));
  }
  const ncclMeshSources *sources(isl_ctx *context) const {
    const ncclMeshSources *result=nullptr;
    checked(ncclMeshCommSources(comm,occurrence,context,&result));return result;
  }
  void append(MetalSchedule *schedule,const std::vector<size_t> &commands) {
    MetalScheduleCommunicator(schedule,comm);
    const ncclMeshGroup publication{comm,0,1};
    Occurrence *held=this;
    checked(MetalScheduleCross(schedule,nullptr,0,&publication,
      ^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,id<MTLResidencySet> residency,NSUInteger,NSUInteger){
        ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
          .residencySet=(__bridge void *)residency,.polls=held->polls};
        return (int)ncclMeshCommEncode(held->comm,held->occurrence,commands.data(),commands.size(),&on);
      }));
  }
};

// Every binding of a dispatch holds the type its kernel declares there (the pipeline's reflection, the one statement of
// it): a record of words shorter than the kernel's struct, or a buffer shorter than one of its elements, is refused,
// since the kernel would read its next fields or elements from whatever follows (60b84953: linear_dims' ten words bound
// as eight, its strides read from the next record and every step's output written tens of MB past y).
static void fits(const Dispatch &d) {
  fits_binding(d.state, d.at, d.words.size() * sizeof(uint32_t), true);
  for (size_t i = 0; i < d.bound.size(); i++) if (d.bound[i]) fits_binding(d.state, i, d.lengths[i], false);
}
// Configuration binds kernels from the Torch frontend into the native recorder.
// Access modes are explicit operands; the recorder orders the declared byte views.
static void prepare_dispatch(MetalSchedule *schedule,const Dispatch &d,uint64_t access) {
  fits(d);
  std::vector<MetalCommandBinding> bindings;
  for(size_t j=0;j<d.bound.size();j++)if(d.bound[j])
    bindings.push_back({d.bound[j],d.offsets[j],d.lengths[j],(uint32_t)j,
      (uint32_t)(((access>>j)&1?MetalReachRead:0)|((access>>(MESH_ACCESS_WRITE_SHIFT+j))&1?MetalReachWritten:0)),0});
  bindings.push_back(MetalScheduleData(schedule,d.words.data(),d.words.size()*sizeof(uint32_t),(uint32_t)d.at));
  MetalScheduleCompute(schedule,d.state,bindings.data(),bindings.size(),d.grid,d.group);

}

// Linear command geometry is compiled from the shared manifest and discovered GPU cores.
static int64_t linear_groups(int64_t rows, int64_t columns) {
  return ((rows + MESH_LINEAR_ROWS - 1) / MESH_LINEAR_ROWS) * ((columns + MESH_LINEAR_COLUMNS - 1) / MESH_LINEAR_COLUMNS);
}


TORCH_LIBRARY_FRAGMENT(torch_mesh, m) {
  m.def("linear(Tensor x, Tensor w, Tensor? bias=None, int act=0, Tensor? x1=None) -> Tensor");
  m.def("linears(Tensor[] x, Tensor[] w, Tensor?[] bias, int[] act, Tensor?[] x1) -> Tensor[]");
}

// Region tensors (torch_mesh.empty, records, address): a tensor whose bytes are a lease of this node's bridge's region
// (ncclMemAlloc: pages registered on every device when it came up), so the library reaches it with no staging copy
// where it reads or writes its operands in place.  An MPS one is torch's: torch's MPS allocator is the region's
// (allocator.mm), so every MPS tensor is a lease, returned once the command buffer open on torch's stream at its release
// has completed; region_empty leases one even for no elements.  A CPU one is torch::from_blob over the lease, returned
// at its release (a host-path call is complete when it returns).
// A region tensor: torch's own allocators are the region's (allocator.mm), so this is a tensor of at least one byte
// whose storage is a lease, on the CPU or the GPU.
static at::Tensor region_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps) {
  const at::ScalarType dtype = like.scalar_type();
  const size_t bytes = std::max<size_t>(1, (size_t)c10::multiply_integers(sizes) * c10::elementSize(dtype));
  const auto options = at::TensorOptions().dtype(dtype).device(mps ? at::kMPS : at::kCPU);
  c10::Allocator *allocator = mps ? at::mps::GetMPSAllocator() : c10::GetCPUAllocator();
  return at::empty({0}, options).set_(c10::Storage(c10::Storage::use_byte_size_t(), bytes, allocator), 0, sizes);
}

// Where a tensor's bytes lie in the region (0: outside it): an MPS tensor's where its MTLBuffer is the Metal buffer of the
// lease its contents lie in, a CPU tensor's where its pointer lies in a lease.
static uint64_t address(const at::Tensor &t) {
  void *theirs = nullptr;
  size_t offset = 0;
  if (!t.is_mps()) return ncclMeshMemBuffer(t.data_ptr(), &theirs, &offset) == ncclSuccess ? (uint64_t)(uintptr_t)t.data_ptr() : 0;
  id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(t);
  char *contents = buffer ? (char *)[buffer contents] : nullptr;
  if (!contents) return 0;
  char *at = contents + t.storage_offset() * t.element_size();
  if (ncclMeshMemBuffer(at, &theirs, &offset) != ncclSuccess || theirs != (__bridge void *)buffer) return 0;
  return (uint64_t)(uintptr_t)at;
}

// Fused crossings bind the same immutable communicator Call declarations.
// GPU bindings name physical channels, including every parallel cable and
// compiled relay path. Producers publish source ranges; consumers observe
// only their data dependencies. Registration cuts are agreed at realization,
// and no buffer must begin on a transport page. Unsupported inputs fail.
// The library's Metal prelude (ncclMeshMetalPrelude): rdma/mesh.metal as the loaded library was built with it, then its
// steps; the text a kernel that runs them is compiled after.
static std::string metal_prelude() { return ncclMeshMetalPrelude(); }

// Generic gathers and reductions retain their typed local page operands.
// The communicator plan owns every compiled transfer.
static int physical_channels(ncclComm_t comm) {
  int channels = 0;
  checked(ncclMeshCommChannels(comm, &channels));
  return channels;
}

using ViewBinding = std::tuple<at::Tensor, int64_t, std::vector<int64_t>, std::vector<int64_t>, int64_t, bool, std::vector<int64_t>, int, int>;
static std::shared_ptr<Occurrence> view_write_operands(const std::vector<ViewBinding> &bindings,
    int64_t comm,int64_t polls) {
  std::vector<at::Tensor> operands;operands.reserve(bindings.size());
  for(const auto &binding:bindings)operands.push_back(std::get<0>(binding));
  auto made=std::make_shared<Occurrence>(comm,std::move(operands));made->polls=(uint32_t)polls;
  return made;
}

// A Python binding of the shared native schedule. All methods except enqueue
// receive configuration; enqueue is the native header's submission operation.
struct ScheduleBinding {
  Schedule schedule;
  ScheduleBinding():schedule(prepared_schedule()){}
  explicit ScheduleBinding(MetalSchedule *prepared):schedule(prepared,[](MetalSchedule *){}){}
  std::vector<at::Tensor> operands;
  void inputs(const std::vector<pybind11::object> &values) {
    for(const auto &value:values)operands.push_back(TensorOperand(value).storage);
    schedule_inputs(schedule.get(),values);
  }
  void results(const std::vector<std::pair<at::Tensor,uint64_t>> &values) {
    std::vector<MetalResult> words;words.reserve(values.size());
    for(const auto &value:values) {
      const at::Tensor &word=value.first;operands.push_back(word);
      MetalTensorPages *pages=tensorPages(word);
      words.push_back({pages.buffer,pages.origin+(NSUInteger)(word.storage_offset()*word.element_size()),value.second});
      MetalScheduleRetainObject(schedule.get(),pages);[pages release];
    }
    MetalScheduleResults(schedule.get(),words.data(),words.size());
  }
  // Preparation supplies native run/context/retain/release operands. The
  // completion worker neither owns a Python callback nor acquires the GIL.
  void host(const std::array<uintptr_t,4> &entry,const std::vector<CommandReach> &reaches) { @autoreleasepool {
    PreparedHost(entry,reaches).append(schedule.get());
  }}
  void copy(const at::Tensor &source,const at::Tensor &destination) {
    TORCH_CHECK(source.is_mps()&&destination.is_mps()&&source.scalar_type()==at::kByte&&
      destination.scalar_type()==at::kByte&&source.is_contiguous()&&destination.is_contiguous()&&
      source.numel()==destination.numel(),"prepared import copy requires matching contiguous MPS byte operands");
    operands.push_back(source);operands.push_back(destination);
    MetalScheduleCopy(schedule.get(),at::native::mps::getMTLBufferStorage(source),source.storage_offset(),
      at::native::mps::getMTLBufferStorage(destination),destination.storage_offset(),source.numel());
  }
  void compute(int64_t kernel,const Tensors &buffers,const std::vector<int64_t> &words,
      int64_t at,int64_t gx,int64_t gy,int64_t threads,uint64_t access) {
    for(const auto &buffer:buffers)if(buffer)operands.push_back(*buffer);
    if(!gx||!gy)return;
    prepare_dispatch(schedule.get(),encoded(kernel,buffers,words,at,gx,gy,threads),access);
  }
  // A missing local endpoint is represented by the communicator's declared relay.
  // The schedule retains declaration operands; its communicator owns compiled transfers.
  std::vector<std::shared_ptr<Occurrence>> prepared;
  void transfer(const std::shared_ptr<Occurrence> &declaration,const std::vector<size_t> &commands) {
    prepared.push_back(declaration);declaration->append(schedule.get(),commands);
  }

  void finish() {
    const auto held=operands;const auto plans=prepared;
    checked(MetalScheduleFinish(schedule.get(),^(NSError *){(void)held;(void)plans;}));
  }
  void enqueue() { MetalScheduleEnqueue(schedule.get()); }
  pybind11::capsule completion() { return schedule_completion(schedule); }

};

// Typed page effects retain their existing region storage through preparation.
struct PageEffectPages {
  struct Storage {void *buffer;size_t first,bytes;at::Tensor value;};
  static ncclMeshBuffer resolved(ncclMeshBuffer binding) {
    if(!binding.buffer) {
      void *buffer=nullptr;size_t offset=0;
      checked(ncclMeshMemBuffer((void *)binding.offset,&buffer,&offset));binding={buffer,offset};
    }
    return binding;
  }
  static Storage storage(ncclMeshBuffer binding,size_t bytes) {
    id<MTLBuffer> buffer=(id<MTLBuffer>)binding.buffer;
    [buffer retain];
    auto base=at::from_blob((char *)buffer.contents+binding.offset,{(int64_t)bytes},
      [buffer](void *){[buffer release];},at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    return {binding.buffer,binding.offset,bytes,std::move(base)};
  }
  static at::Tensor page(ncclMeshBuffer binding,std::vector<int64_t> shape,at::ScalarType type,
      std::vector<Storage> &storages) {
    binding=resolved(binding);
    size_t bytes=c10::elementSize(type);for(int64_t extent:shape)bytes*=extent;
    auto found=std::find_if(storages.begin(),storages.end(),[&](const Storage &held){
      return held.buffer==binding.buffer&&held.first<=binding.offset&&
        binding.offset+bytes<=held.first+held.bytes;
    });
    if(found==storages.end()) {
      storages.push_back(storage(binding,bytes));found=storages.end()-1;
    }
    std::vector<int64_t> strides(shape.size());int64_t width=1;
    for(size_t i=shape.size();i-->0;){strides[i]=width;width*=shape[i];}
    const size_t scalar=c10::elementSize(type);
    return found->value.as_strided({(int64_t)(found->bytes/scalar*scalar)},{1},0).view(type)
      .as_strided(shape,strides,(binding.offset-found->first)/scalar);
  }
};

// Source effects remain typed page operands. This preparation interface never
// reverse-decodes a backend binding or makes an execution-time address lookup.
static pybind11::list occurrence_source_views(const Occurrence &prepared,uintptr_t context) {
  @try {
    const ncclMeshSources *source=prepared.sources(reinterpret_cast<isl_ctx *>(context));
    pybind11::list result;
    for(MetalPageTransfer *transfer in (NSArray *)source->transfers) {
      auto storage=PageEffectPages::storage({transfer->payload,0},transfer->payload.length);
      result.append(pybind11::make_tuple(storage.value,TensorLayout(transfer->layout)));
    }
    return result;
  } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
}

static pybind11::list occurrence_transports(const Occurrence &prepared,uintptr_t context) {
  const ncclMeshSources *source=prepared.sources(reinterpret_cast<isl_ctx *>(context));
  pybind11::list result;
  for(size_t i=0;i<source->transports;i++) {
    const auto &t=source->transport[i];pybind11::dict row;
    row["transport"]=i;row["call"]=t.call;row["first"]=t.first;row["bytes"]=t.bytes;row["units"]=t.units;
    row["frame"]=t.frame;row["domain"]=t.domain;row["channel"]=t.channel;row["direction"]=t.direction;
    row["transaction"]=t.transaction;row["value_first"]=t.value_first;
    row["entry"]=t.entry;row["call_begin"]=source->call_first[t.call];row["call_end"]=source->call_first[t.call+1];
    result.append(row);
  }
  return result;
}

static pybind11::list occurrence_publications(const Occurrence &prepared,uintptr_t context) {
  @try {
    const ncclMeshSources *source=prepared.sources(reinterpret_cast<isl_ctx *>(context));
    std::vector<PageEffectPages::Storage> pages;pybind11::list result;
    for(NSArray *domain in MetalPagePublicationDomains((NSArray *)source->transfers)) {
      pybind11::list cuts;
      for(NSUInteger index=0;index<domain.count;index++) {
        MetalPageTransfer *t=domain[index];pybind11::dict row;
        auto storage=PageEffectPages::storage({t->payload,0},t->payload.length);
        row["cut"]=index;row["call"]=t->effect.call;row["frame"]=t->effect.frame;
        row["operand"]=t->effect.operand;row["first"]=t->effect.first;row["bytes"]=t->effect.bytes;
        row["threshold"]=t->effect.threshold;row["transport"]=t->effect.transport;
        row["transport_first"]=t->effect.transport_first;row["coverage_cut"]=t->coverageCut;
        row["storage"]=storage.value;row["layout"]=TensorLayout(t->layout);cuts.append(row);
      }
      MetalPageTransfer *first=domain[0];pybind11::dict row;
      row["word"]=PageEffectPages::page(first->effect.word,{1},at::kUInt64,pages);
      row["cuts"]=cuts;result.append(row);
    }
    return result;
  } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
}

// The frontend supplies its retained source and normalized page relations.
// Dependency solving and effect emission belong to the shared native compiler.
static uintptr_t page_group_relation(uintptr_t relation,const std::array<NSUInteger,3> &threads,
    const std::array<NSUInteger,3> &counts,const std::vector<NSInteger> &axes,const std::vector<NSUInteger> &reductions) {
  @try {
    return reinterpret_cast<uintptr_t>(MetalPageGroupRelation(reinterpret_cast<isl_map *>(relation),
      threads.data(),counts.data(),axes.data(),axes.size(),reductions.data(),reductions.size()));
  } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
}

static pybind11::dict compile_page_effects(uintptr_t context,const pybind11::list &descriptions,const pybind11::list &occurrences) {
  @try {
    NSMutableArray<MetalPageCommand *> *commands=[NSMutableArray array];
    NSMutableArray<MetalPageTransfer *> *transfers=[NSMutableArray array];
    std::unordered_map<MetalPageCommand *,size_t> commandIndices;
    struct Block {NSMutableArray *commands=[NSMutableArray array],*transfers=[NSMutableArray array];};
    struct NativeSource {std::shared_ptr<Occurrence> owner;NSArray *commands,*transfers;};
    std::map<NSUInteger,Block> blocks;
    std::vector<NativeSource> native;
    pybind11::list recordings;
    std::unordered_map<MetalPageTransfer *,size_t> transferOwners;
    auto relation=[](const pybind11::handle &value){
      return reinterpret_cast<isl_map *>(value.attr("pointer").cast<uintptr_t>());
    };
    auto accesses_of=[&](const pybind11::handle &values) {
      NSMutableArray<MetalPageAccess *> *accesses=[NSMutableArray array];
      for(const auto &entry:values.cast<pybind11::tuple>()) {
        const auto access=pybind11::reinterpret_borrow<pybind11::dict>(entry);
        [accesses addObject:[MetalPageAccess root:MetalPageRegionRoot()
          slot:access["slot"].cast<NSInteger>() access:access["access"].cast<uint32_t>() relation:relation(access["relation"])]];
      }
      return accesses;
    };
    std::unordered_map<PyObject *,PythonPageCoverage *> coverages;
    auto coverage_of=[&](const pybind11::handle &value) {
      auto found=coverages.find(value.ptr());
      if(found!=coverages.end())return found->second;
      auto *made=page_coverage(value);coverages[value.ptr()]=made;return made;
    };
    for(const auto &item:descriptions) {
      const auto row=pybind11::reinterpret_borrow<pybind11::dict>(item);
      NSArray<MetalPageAccess *> *accesses=accesses_of(row["accesses"]);
      const auto grid=row["grid"].cast<std::array<NSUInteger,3>>();
      const auto source=row["source"].cast<std::string>(),function=row["function"].cast<std::string>();
      auto &block=blocks[row["order"].cast<NSUInteger>()];
      MetalPageCommand *command=[MetalPageCommand order:block.commands.count
        groups:row["groups"].cast<NSUInteger>() backend:row["backend"].cast<MetalPageBackend>()
        source:@(source.c_str()) function:@(function.c_str()) grid:MTLSizeMake(grid[0],grid[1],grid[2])
        metadataSlot:row["metadata_slot"].cast<NSUInteger>() accesses:accesses];
      command->coherentAccesses=row["coherent_accesses"].cast<bool>();
      NSMutableArray *names=[NSMutableArray array];
      for(const auto &name:row["argument_names"].cast<std::vector<std::string>>())[names addObject:@(name.c_str())];
      command->argumentNames=[names copy];
      for(const auto &entry:row["sites"].cast<pybind11::tuple>()) {
        const auto site=pybind11::reinterpret_borrow<pybind11::dict>(entry);
        const auto name=site["name"].cast<std::string>();
        MetalPageCommandSite(command,@(name.c_str()),site["cells"].cast<NSUInteger>(),
          site["scope"].cast<MetalPageSiteScope>(),accesses_of(site["accesses"]));
      }
      NSMutableArray *contributions=[NSMutableArray array],*announcements=[NSMutableArray array];
      for(const auto &entry:row["contributions"].cast<pybind11::tuple>()) {
        const auto part=pybind11::reinterpret_borrow<pybind11::dict>(entry);
        const TensorOperand word(part["word"].cast<pybind11::object>()),indices(part["indices"].cast<pybind11::object>());
        auto *made=[PythonPageContribution coverage:coverage_of(part["coverage"])
          word:(ncclMeshBuffer){word.pages.buffer,word.origin} phase:part["phase"].cast<NSUInteger>()
          cells:part["cells"].cast<NSUInteger>() width:part["width"].cast<NSUInteger>()
          indices:(ncclMeshBuffer){indices.pages.buffer,indices.origin}];
        made->tensors={word.storage,indices.storage};[contributions addObject:made];
      }
      for(const auto &entry:row["announcements"].cast<pybind11::tuple>()) {
        const auto part=pybind11::reinterpret_borrow<pybind11::dict>(entry);
        [announcements addObject:[MetalPageAnnouncement coverage:coverage_of(part["coverage"])
          phase:part["phase"].cast<NSUInteger>() flags:part["flags"].cast<uint32_t>() relation:relation(part["relation"])]];
      }
      command->contributions=[contributions copy];command->announcements=[announcements copy];
      commandIndices[command]=commandIndices.size();[block.commands addObject:command];
    }
    for(const auto &item:occurrences) {
      const auto row=pybind11::reinterpret_borrow<pybind11::dict>(item);
      const auto owner=row["owner"].cast<std::shared_ptr<Occurrence>>();
      const auto *source=owner->sources(reinterpret_cast<isl_ctx *>(context));
      NSArray *declared=(NSArray *)source->commands,*effects=(NSArray *)source->transfers;
      const auto frames=row["frame_orders"].cast<std::vector<NSUInteger>>();
      const auto base=row["order"].cast<NSUInteger>();
      std::map<NSUInteger,std::vector<size_t>> selected;
      for(NSUInteger index=0;index<declared.count;index++) {
        MetalPageCommand *command=declared[index];
        const auto order=frames.empty()||command->frame==UINT32_MAX?base:frames.at(command->frame);
        [blocks[order].commands addObject:command];selected[order].push_back(index);
      }
      if(selected.empty())selected[base]={};
      for(const auto &part:selected)recordings.append(pybind11::make_tuple(part.first,owner,part.second));
      for(MetalPageTransfer *transfer in effects)
        [blocks[frames.empty()?base:frames.at(transfer->effect.frame)].transfers addObject:transfer];
      for(MetalPageTransfer *transfer in effects)transferOwners[transfer]=native.size();
      native.push_back({owner,declared,effects});
    }
    NSUInteger next=0;
    for(auto &entry:blocks) {
      MetalPageEffectsOrder(entry.second.commands,entry.second.transfers,&next);
      [commands addObjectsFromArray:entry.second.commands];[transfers addObjectsFromArray:entry.second.transfers];
    }
    MetalPageEffects *effects=MetalPageEffectsSolve(commands,transfers);
    pybind11::list sources,results;
    std::unordered_map<MetalPageCommand *,MetalPageSource *> emitted;
    auto device=at::mps::MPSDevice::getInstance()->device();
    for(NSUInteger index=0;index<effects->commands.count;index++) {
      MetalPageSource *source=MetalPageEffectsSource(effects,index,device);
      MetalPageCommand *command=effects->commands[index];emitted[command]=source;
      const auto frontend=commandIndices.find(command);if(frontend==commandIndices.end())continue;
      // Every returned page view retains the complete composition, including
      // prefix pages. Native host bindings can retain any of their exact uses.
      auto page=[source](id<MTLBuffer> buffer,NSUInteger first,NSUInteger bytes){
        [source retain];
        return at::from_blob((char *)buffer.contents+first,{(int64_t)bytes},
          [source](void *){[source release];},at::TensorOptions().dtype(at::kByte).device(at::kCPU));
      };
      pybind11::dict row;
      row["index"]=frontend->second;row["source"]=source->source.UTF8String;
      row["metadata"]=source->metadata?pybind11::cast(page(source->metadata,source->metadataOffset,
        source->metadata.length-source->metadataOffset)):pybind11::none();
      pybind11::list reaches,observed,published;
      const auto *reach=(const MetalReach *)source->reaches.bytes;
      for(NSUInteger r=0;r<source->reaches.length/sizeof *reach;r++)
        reaches.append(pybind11::make_tuple(reach[r].binding,reach[r].access,page(reach[r].buffer,reach[r].offset,reach[r].length)));
      for(NSNumber *slot in source->observedSlots)observed.append(slot.unsignedIntegerValue);
      for(NSNumber *slot in source->publishedSlots)published.append(slot.unsignedIntegerValue);
      row["reaches"]=reaches;row["observed_slots"]=observed;row["published_slots"]=published;sources.append(row);
    }
    for(size_t index=0;index<native.size();index++) {
      const auto &source=native[index];
      std::vector<ncclMeshSourceInput> inputs;
      std::vector<std::vector<ncclMeshPageEffect>> prior;prior.reserve(effects->initial.count);
      for(MetalPageInitial *input in effects->initial) {
        const auto *transfer=input->transfer;
        if(transfer->effect.operand!=ncclMeshPageSource||transferOwners.at(input->transfer)!=index)continue;
        prior.emplace_back();for(MetalPageTransfer *arrival in input->observations)prior.back().push_back(arrival->effect);
        inputs.push_back({transfer->effect.call,transfer->effect.first+input->first,input->bytes,
          transfer->effect.frame,prior.back().data(),prior.back().size()});
      }
      checked(ncclMeshCommSourceInputs(source.owner->comm,source.owner->occurrence,inputs.data(),inputs.size()));
      NSMutableArray *specialized=[NSMutableArray arrayWithCapacity:source.commands.count];
      for(MetalPageCommand *command in source.commands)[specialized addObject:emitted.at(command)];
      checked(ncclMeshCommSpecialize(source.owner->comm,source.owner->occurrence,specialized));
    }
    std::vector<PageEffectPages::Storage> resultPages;
    for(MetalPageTransfer *transfer in effects->results)
      results.append(pybind11::make_tuple(PageEffectPages::page(transfer->effect.word,{1},at::kUInt64,resultPages),
        transfer->effect.threshold));
    pybind11::dict result;result["sources"]=sources;result["results"]=results;result["recordings"]=recordings;return result;
  } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
}

// Invocation slots and results are operands. Runtime selects the next prepared
// native function; it does not enter the framework's encoding queue.
struct Executions {
  std::vector<Schedule> schedules;
  pybind11::object output;
  size_t next=0;
  Schedule latest;
  explicit Executions(pybind11::object value):output(std::move(value)){}
  void add(ScheduleBinding &binding){
    size_t ended=0;
    while(ended<next) {
      NSError *error=nil;const int status=MetalScheduleStatus(schedules[ended].get(),&error);
      TORCH_CHECK(status>=0,"prepared function failed: ",error.localizedDescription.UTF8String);
      if(!status)break;
      ended++;
    }
    schedules.erase(schedules.begin(),schedules.begin()+ended);next-=ended;
    schedules.push_back(binding.schedule);
  }
  pybind11::object run(){latest=schedules[next++];MetalScheduleEnqueue(latest.get());return output;}
  pybind11::capsule completion(){return schedule_completion(latest);}
};

// Observe a whole function without waiting. Tuple observations retain each
// component and report the latest native end only when all have completed.
static void completion_schedules(pybind11::handle observation, std::vector<Schedule> &into) {
  if(pybind11::isinstance<pybind11::tuple>(observation)) {
    for(auto item:observation.cast<pybind11::tuple>())completion_schedules(item,into);
    return;
  }
  auto token=pybind11::reinterpret_borrow<pybind11::capsule>(observation);
  TORCH_CHECK(!strcmp(token.name(),"mesh.prepared-function"),"torch_mesh.completion_observe: ",token.name()," is no prepared function's completion");
  into.push_back(*static_cast<Schedule *>(token.get_pointer()));
}
static std::optional<double> completion_observe(pybind11::object observation) {
  std::vector<Schedule> schedules;
  completion_schedules(observation,schedules);
  double latest=0;
  bool pending=false;
  for(const Schedule &schedule:schedules) {
    double ended=0;
    NSError *error=nil;
    const int status=MetalScheduleEnd(schedule.get(),&ended,&error);
    TORCH_CHECK(status>=0,"the prepared function failed: ",error.localizedDescription.UTF8String);
    pending|=status==0;
    latest=std::max(latest,ended);
  }
  return pending?std::nullopt:std::optional<double>(latest);
}

// The module: prepared functions, declared collectives and fused crossings; no collective is
// issued at a call (design/eager-execution.md).
// Every call from Python drains the autorelease pool it pushed: Python's thread has no run loop, so an object
// autoreleased with no pool of its own goes to the thread's implicit pool, which is popped only at the thread's end, and
// lives, outside the region, for the process's life (2026-10-29: 600,000 such objects in a chain rank on m5ult1).
extern "C" void *objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void *);
struct Drained {
  void *pool = objc_autoreleasePoolPush();
  Drained() = default;
  Drained(const Drained &) = delete;
  ~Drained() { objc_autoreleasePoolPop(pool); }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  // An Objective-C exception from any binding reaches Python as a ValueError naming its reason. Without it, pybind
  // reports a binding with no @catch of its own as "Caught an unknown exception!" (the chain's first distribute on
  // main, 2026-11-01, TensorLayout select_runs), its cause gone. A C++ exception leaves this translator for pybind's own.
  pybind11::register_exception_translator([](std::exception_ptr raised) {
    @try { if (raised) std::rethrow_exception(raised); }
    @catch (NSException *error) {
      PyErr_SetString(PyExc_ValueError, [NSString stringWithFormat:@"%@: %@", error.name, error.reason].UTF8String);
    }
  });
  module.def("scalar_read",&scalar_read,pybind11::arg("source"),pybind11::arg("source_type"),
    pybind11::arg("output_type"),pybind11::arg("offsets"),pybind11::arg("bits")=0,
    pybind11::arg("signed")=true,pybind11::arg("scales")=std::nullopt,pybind11::arg("zeros")=std::nullopt,
    pybind11::arg("absent")=0,pybind11::arg("bitcast")=false,pybind11::arg("present")=std::nullopt);
  install_region_allocator();
  module.def("place_layout",[](NSUInteger count,const std::vector<TensorLayout> &uses,
      const std::vector<TensorLayout> &targets,const std::vector<std::vector<NSInteger>> &footprints,
      const std::vector<TensorLayout> &values,const std::vector<NSInteger> &widths)->pybind11::object {
    auto layouts=[](const std::vector<TensorLayout> &items) {
      NSMutableArray *result=[NSMutableArray array];for(const auto &item:items)[result addObject:item.value];return result;
    };
    @try {
      NSMutableArray *reserved=[NSMutableArray array];for(const auto &span:footprints)[reserved addObject:tensorNumbers(span)];
      MetalTensorLayout *placed=MetalTensorPlace(count,layouts(uses),layouts(targets),reserved,layouts(values),tensorNumbers(widths));
      return placed?pybind11::cast(TensorLayout(placed)):pybind11::none();
    } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
  }, pybind11::call_guard<Drained>());
  module.def("prefer_layout",[](NSUInteger count,const pybind11::list &hard,const pybind11::list &groups,
      const std::vector<double> &preferences,const std::vector<TensorLayout> &values,
      const std::vector<NSInteger> &widths)->pybind11::object {
    auto demands=[](const pybind11::handle &items) {
      NSMutableArray *result=[NSMutableArray array];
      for(const auto &item:items.cast<pybind11::list>()) {
        auto record=item.cast<pybind11::dict>();std::string flexible=record["flexible"].cast<std::string>();
        [result addObject:@{@"use":record["use"].cast<TensorLayout>().value,
          @"target":record["target"].cast<TensorLayout>().value,
          @"footprint":tensorNumbers(record["footprint"].cast<std::vector<NSInteger>>()),
          @"width":@(record["width"].cast<NSUInteger>()),@"flexible":@(!flexible.empty()),
          @"requirements":flexible.empty()?@[]:@[@{@"kind":@(flexible.c_str())}]}];
      }
      return result;
    };
    @try {
      NSMutableArray *choices=[NSMutableArray array],*weights=[NSMutableArray array],*typed=[NSMutableArray array];
      for(const auto &group:groups)[choices addObject:demands(group)];
      for(double weight:preferences)[weights addObject:@(weight)];
      for(const auto &value:values)[typed addObject:value.value];
      NSDictionary *solved=MetalTensorPlacePreferred(count,demands(hard),choices,weights,typed,tensorNumbers(widths));
      if(!solved)return pybind11::none();
      pybind11::dict result;result["layout"]=TensorLayout(solved[@"layout"]);
      result["extent"]=[solved[@"extent"] unsignedIntegerValue];result["alignment"]=[solved[@"alignment"] unsignedIntegerValue];
      std::vector<NSUInteger> accepted;for(NSNumber *index in solved[@"accepted"])accepted.push_back(index.unsignedIntegerValue);
      std::vector<std::vector<NSUInteger>> conflicts;
      for(NSArray *group in solved[@"conflicts"]) {std::vector<NSUInteger> row;for(NSNumber *index in group)row.push_back(index.unsignedIntegerValue);conflicts.push_back(row);}
      result["accepted"]=accepted;result["conflicts"]=conflicts;return std::move(result);
    } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
  },pybind11::call_guard<Drained>());
  module.def("page_backing",[](const pybind11::object &value) {
    TensorOperand operand(value);
    return std::make_tuple((uint64_t)(uintptr_t)operand.pages.buffer.contents,
      operand.pages.buffer.length,TensorLayout([operand.layout.value components:operand.storage.element_size()]));
  }, pybind11::call_guard<Drained>());
  module.def("page_region",[](const pybind11::object &value,uintptr_t context) {
    const TensorOperand operand(value);
    MetalTensorLayout *bytes=[operand.layout.value components:operand.storage.element_size()];
    auto *relation=MetalPageRegionSelection(reinterpret_cast<isl_ctx *>(context),operand.pages.buffer,bytes);
    return std::make_pair(bytes.count,reinterpret_cast<uintptr_t>(relation));
  },pybind11::call_guard<Drained>());
  module.def("page_storage",&page_storage,pybind11::call_guard<Drained>());
  module.def("page_domain",[](const std::vector<at::Tensor> &owners){
    NSArray<MetalTensorPages *> *sources=page_sources(owners);
    return std::make_pair(TensorLayout([MetalTensorPages domain:sources]),sources[0].pageBytes);
  },pybind11::call_guard<Drained>());
  module.def("host_page_backing",[](uint64_t address) {
    @try {
      MetalTensorPages *pages=[MetalTensorPages host:(const void *)(uintptr_t)address];
      return std::array<uint64_t,3>{(uint64_t)(uintptr_t)pages.buffer.contents,pages.buffer.length,pages.origin};
    } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
  }, pybind11::call_guard<Drained>());
  module.def("completion_observe", &completion_observe, pybind11::call_guard<Drained>());
  module.def("pipeline", &pipeline, pybind11::call_guard<Drained>());
  module.def("pipeline_specialized", &pipeline_specialized, pybind11::call_guard<Drained>());
  module.def("index_constant_prelude", [](){return std::string(MetalViewIndexConstantPrelude().UTF8String);}, pybind11::call_guard<Drained>());
  pybind11::class_<TensorTraversal>(module,"TensorTraversal")
    .def(pybind11::init<const std::vector<TensorLayout> &,const TensorLayout &,const TensorLayout &>(), pybind11::call_guard<Drained>())
    .def_property_readonly("cuts",[](const TensorTraversal &v){return tensorNumbers(v.value.cuts);})
    .def("fragment",[](const TensorTraversal &v,NSUInteger first,NSUInteger end){
      @try { auto f=[v.value fragment:first end:end];return std::make_tuple(f.part,f.offset,f.count); }
      @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
    }, pybind11::call_guard<Drained>());
  pybind11::class_<TensorLayout>(module,"TensorLayout")
    .def_static("identify",[](NSUInteger count,const TensorLayout &source,const TensorLayout &target) {
      @try {NSArray *result=MetalTensorIdentify(count,source.value,target.value);
        return std::make_pair(TensorLayout(result[0]),TensorLayout(result[1]));
      } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
    },pybind11::call_guard<Drained>())
    .def_static("assign",[](NSUInteger count,const std::vector<TensorLayout> &uses,const std::vector<TensorLayout> &targets) {
      NSMutableArray *read=[NSMutableArray array],*write=[NSMutableArray array];
      for(const auto &value:uses)[read addObject:value.value];for(const auto &value:targets)[write addObject:value.value];
      @try { NSArray *assigned=MetalTensorAssignments(count,read,write);
        return std::make_tuple(TensorLayout(assigned[0]),TensorLayout(assigned[1]),TensorLayout(assigned[2]));
      } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
    }, pybind11::call_guard<Drained>())
    .def_static("unite",[](const TensorLayout &sourceRoot,const TensorLayout &targetRoot,
        const TensorLayout &source,const TensorLayout &target) {
      @try {
        return std::make_tuple(TensorLayout(MeshIndexUnion(sourceRoot.value,targetRoot.value,source.value,target.value,nil,YES)),
          TensorLayout(MeshIndexUnion(sourceRoot.value,targetRoot.value,source.value,target.value,sourceRoot.value,YES)),
          TensorLayout(MeshIndexUnion(sourceRoot.value,targetRoot.value,source.value,target.value,targetRoot.value,NO)));
      } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
    }, pybind11::call_guard<Drained>())
    .def_static("runs",[](const std::vector<std::vector<NSInteger>> &runs){
      NSMutableArray *values=[NSMutableArray array];for(const auto &run:runs)[values addObject:tensorNumbers(run)];
      return TensorLayout([MetalTensorLayout runs:values]);
    }, pybind11::call_guard<Drained>())
    .def(pybind11::init(&TensorLayout::make),pybind11::arg("shape"),pybind11::arg("strides"),pybind11::arg("offset")=0,
         pybind11::arg("maps")=std::map<NSUInteger,std::vector<NSInteger>>{},pybind11::arg("indices")=pybind11::none(),
         pybind11::arg("index_shape")=pybind11::none(),pybind11::arg("index_coordinates")=pybind11::none(), pybind11::call_guard<Drained>())
    .def_property_readonly("shape",[](const TensorLayout &v){return tensorNumbers(v.value.shape);})
    .def_property_readonly("index_shape",[](const TensorLayout &v){return tensorNumbers(v.value.indexShape);})
    .def_property_readonly("strides",[](const TensorLayout &v){return tensorNumbers(v.value.strides);})
    .def_property_readonly("offset",[](const TensorLayout &v){return v.value.offset;})
    .def_property_readonly("count",[](const TensorLayout &v){return v.value.count;})
    .def_property_readonly("affine",[](const TensorLayout &v){return bool(v.value.affine);})
    .def_property_readonly("state",[](const TensorLayout &v){
      std::vector<std::vector<NSInteger>> maps(v.value.indexShape.count);
      for(NSNumber *axis in v.value.maps){NSData *data=v.value.maps[axis];const auto *p=(const NSInteger *)data.bytes;maps[axis.unsignedIntegerValue]={p,p+data.length/sizeof *p};}
      NSData *data=[NSJSONSerialization dataWithJSONObject:v.value.coordinates options:0 error:nil];
      return pybind11::make_tuple(tensorNumbers(v.value.shape),tensorNumbers(v.value.strides),v.value.offset,maps,
          tensorNumbers(v.value.indexShape),std::string((const char *)data.bytes,data.length));
    })
    .def("index",[](const TensorLayout &v,NSUInteger i){return [v.value index:i];}, pybind11::call_guard<Drained>())
    .def("shift",[](const TensorLayout &v,NSInteger offset){return TensorLayout([v.value shift:offset]);}, pybind11::call_guard<Drained>())
    .def("equivalent",[](const TensorLayout &v,const TensorLayout &other){return bool([v.value equivalent:other.value]);}, pybind11::call_guard<Drained>())
    .def("intersection",[](const TensorLayout &v,const TensorLayout &other){return TensorLayout([v.value intersection:other.value]);}, pybind11::call_guard<Drained>())
    .def("select",[](const TensorLayout &v,NSUInteger axis,const std::vector<NSInteger> &indices){return TensorLayout([v.value select:axis indices:tensorWords(indices)]);}, pybind11::call_guard<Drained>())
    .def("select_runs",[](const TensorLayout &v,NSUInteger axis,const std::vector<std::vector<NSInteger>> &runs){
      NSMutableArray *parts=[NSMutableArray array];for(const auto &run:runs)[parts addObject:tensorNumbers(run)];
      return TensorLayout([v.value select:axis runs:parts]);
    }, pybind11::call_guard<Drained>())
    .def("slice",[](const TensorLayout &v,NSUInteger axis,NSInteger first,NSUInteger count,NSInteger step){return TensorLayout([v.value slice:axis first:first count:count step:step]);}, pybind11::call_guard<Drained>())
    .def("element",[](const TensorLayout &v,NSUInteger axis,NSUInteger i){return TensorLayout([v.value element:axis at:i]);}, pybind11::call_guard<Drained>())
    .def("permute",[](const TensorLayout &v,const std::vector<NSInteger> &axes){return TensorLayout([v.value permute:tensorNumbers(axes)]);}, pybind11::call_guard<Drained>())
    .def("reshape",[](const TensorLayout &v,const std::vector<NSInteger> &shape){return TensorLayout([v.value reshape:tensorNumbers(shape)]);}, pybind11::call_guard<Drained>())
    .def("components",[](const TensorLayout &v,NSUInteger count){return TensorLayout([v.value components:count]);}, pybind11::call_guard<Drained>())
    .def("elements",[](const TensorLayout &v,NSUInteger count){return TensorLayout([v.value elements:count]);}, pybind11::call_guard<Drained>())
    .def("gather",[](const TensorLayout &v,const std::vector<NSInteger> &shape,const std::vector<NSInteger> &indices){return TensorLayout([v.value gather:tensorNumbers(shape) indices:tensorWords(indices)]);}, pybind11::call_guard<Drained>())
    .def("rebase",[](const TensorLayout &v,const TensorLayout &declared,const TensorLayout &placed){
      @try{return TensorLayout([v.value rebase:declared.value onto:placed.value]);}
      @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
    }, pybind11::call_guard<Drained>())
    .def("matches",[](const TensorLayout &v,const std::vector<NSInteger> &steps){return bool([v.value matches:tensorNumbers(steps)]);}, pybind11::call_guard<Drained>())
    .def("required",[](const TensorLayout &v,const std::string &kind,const std::vector<NSInteger> &steps){return TensorLayout([v.value required:@(kind.c_str()) strides:tensorNumbers(steps)]);},pybind11::arg("kind"),pybind11::arg("strides")=std::vector<NSInteger>{}, pybind11::call_guard<Drained>())
    .def("requirements",[](const TensorLayout &v,const pybind11::list &requirements){
      NSMutableArray *values=[NSMutableArray array];
      for(const auto &requirement:requirements) {
        if(pybind11::isinstance<TensorLayout>(requirement)) {
          [values addObject:@{@"layout":requirement.cast<TensorLayout>().value}];continue;
        }
        const auto [kind,steps]=requirement.cast<std::pair<std::string,std::vector<NSInteger>>>();
        [values addObject:@{@"kind":@(kind.c_str()),@"strides":tensorNumbers(steps)}];
      }
      @try { return TensorLayout([v.value requirements:values]); }
      @catch(NSException *error){TORCH_CHECK(false,error.reason.UTF8String);}
    }, pybind11::call_guard<Drained>())
    .def("declaration",[](const TensorLayout &v,NSUInteger binding,const std::optional<NSInteger> &origin){
      return std::string([v.value declaration:binding origin:origin?*origin:v.value.offset].UTF8String);
      },pybind11::arg("binding"),pybind11::arg("origin")=pybind11::none(),pybind11::call_guard<Drained>())
    .def("constants",[](const TensorLayout &v,NSUInteger binding,const std::optional<NSInteger> &origin)->pybind11::object{
      NSArray<NSArray<NSNumber *> *> *values=[v.value constants:binding origin:origin?*origin:v.value.offset];
      if(!values)return pybind11::none();
      pybind11::list out;
      for(NSArray<NSNumber *> *c in values)out.append(pybind11::make_tuple(c[0].longLongValue,c[1].longLongValue,c[2].longLongValue));
      return out;},pybind11::arg("binding"),pybind11::arg("origin")=pybind11::none(),pybind11::call_guard<Drained>())
    .def("storage_declaration",[](const TensorLayout &v,NSUInteger binding,const TensorLayout &source){
      return std::string([v.value declaration:binding fromStorage:source.value].UTF8String);}, pybind11::call_guard<Drained>())
    .def("storage_byte_declaration",[](const TensorLayout &v,NSUInteger binding,const TensorLayout &source,NSUInteger width){
      return std::string([v.value scalarStorageDeclaration:binding fromBytes:source.value elementBytes:width].UTF8String);}, pybind11::call_guard<Drained>())
    .def("storage_span",[](const TensorLayout &v){
      const MetalTensorSpan span=v.value.storageSpan;return std::make_pair(span.offset,span.count);
    }, pybind11::call_guard<Drained>())
    .def("ordered_spans",[](const TensorLayout &v){
      NSData *data=[v.value orderedSpans];const auto *spans=(const MetalTensorSpan *)data.bytes;
      std::vector<std::pair<NSInteger,NSUInteger>> result;
      for(NSUInteger i=0;i<data.length/sizeof *spans;i++)result.emplace_back(spans[i].offset,spans[i].count);
      return result;
    }, pybind11::call_guard<Drained>())
    .def("spans",[](const TensorLayout &v,NSUInteger bytes){
      NSData *data=[v.value reaches:nil base:0 elementBytes:bytes access:MetalReachRead binding:0];
      const MetalReach *spans=(const MetalReach *)data.bytes;std::vector<std::pair<NSUInteger,NSUInteger>> result;
      for(NSUInteger i=0;i<data.length/sizeof *spans;i++)result.push_back({spans[i].offset,spans[i].length});return result;
    }, pybind11::call_guard<Drained>())
    .def("__eq__",[](const TensorLayout &v,const TensorLayout &w){return bool([v.value isEqual:w.value]);}, pybind11::call_guard<Drained>())
    .def("__hash__",[](const TensorLayout &v){return v.value.hash;}, pybind11::call_guard<Drained>());
  module.def("operand_layout",[](const pybind11::object &object){return TensorOperand(object).layout;}, pybind11::call_guard<Drained>());
  module.def("physical_layout",[](const pybind11::object &object){
    const TensorOperand operand(object);
    return TensorLayout([operand.pages physical:operand.layout.value elementBytes:operand.storage.element_size()]);
  },pybind11::call_guard<Drained>());
  module.def("same_operand",[](const pybind11::object &left,const pybind11::object &right){
    const TensorOperand a(left),b(right);
    return a.storage.scalar_type()==b.storage.scalar_type()&&
      [[a.pages physical:a.layout.value elementBytes:a.storage.element_size()]
        equivalent:[b.pages physical:b.layout.value elementBytes:b.storage.element_size()]];
  }, pybind11::call_guard<Drained>());
  module.def("view_prelude", [](){return std::string(MetalViewIndexPrelude().UTF8String);}, pybind11::call_guard<Drained>());
  auto viewSelection=[](NSDictionary *result) {
    auto values=[](NSData *data) {
      std::vector<int32_t> words(data.length/sizeof(int32_t));
      if(data.length)memcpy(words.data(),data.bytes,data.length);
      return words;
    };
    return std::make_tuple(values(result[@"blocks"]),values(result[@"present"]),values(result[@"order"]));
  };
  // an axis's domain (present stored positions and each one's logical index) of `stored` entries, selected: the
  // retained blocks and the selection's own domain (metal-recording.h MetalViewBlockSlice)
  module.def("view_block_slice", [viewSelection](const std::vector<int32_t> &present,const std::vector<int32_t> &order,NSUInteger stored,
                                                 NSInteger first,NSInteger end,NSInteger step,NSUInteger block) {
    TORCH_CHECK(present.size()==order.size(),"a domain gives each present position its logical index");
    return viewSelection(MetalViewBlockSlice(present.data(),order.data(),order.size(),stored,first,end,step,block));
  }, pybind11::call_guard<Drained>());
  module.def("view_block_gather", [viewSelection](const std::vector<int32_t> &present,const std::vector<int32_t> &order,NSUInteger stored,
                                                  const std::vector<int32_t> &selected,NSUInteger block) {
    TORCH_CHECK(present.size()==order.size(),"a domain gives each present position its logical index");
    return viewSelection(MetalViewBlockGather(present.data(),order.data(),order.size(),stored,selected.data(),selected.size(),block));
  }, pybind11::call_guard<Drained>());
  module.def("view_tiles", [](const std::vector<NSUInteger> &shape,NSUInteger elements) {
    using Tile=std::pair<std::vector<NSUInteger>,std::vector<NSUInteger>>;
    if(shape.empty())return std::vector<Tile>{{{}, {}}};
    NSData *data=MetalViewTiles(shape.data(),shape.size(),elements);
    const NSUInteger *words=(const NSUInteger *)data.bytes;
    std::vector<Tile> tiles;
    for(size_t i=0;i<data.length/sizeof *words;i+=2*shape.size())
      tiles.push_back({{words+i,words+i+shape.size()},{words+i+shape.size(),words+i+2*shape.size()}});
    return tiles;
  }, pybind11::call_guard<Drained>());
  module.def("butterfly_plan", [](NSUInteger width,NSUInteger localElements) {
    NSData *data=MetalButterflyPlan(width,localElements);
    const MetalButterflyStage *stages=(const MetalButterflyStage *)data.bytes;
    std::vector<std::pair<NSUInteger,NSUInteger>> plan;
    for(size_t i=0;i<data.length/sizeof *stages;i++)plan.push_back({stages[i].width,stages[i].step});
    return plan;
  }, pybind11::call_guard<Drained>());
  module.def("hadamard_local", [](NSUInteger columns,NSUInteger mix) {
    auto device=at::mps::MPSDevice::getInstance()->device();
    return bool(MetalHadamardLocal(columns,mix,device.maxThreadsPerThreadgroup.width,device.maxThreadgroupMemoryLength));
  }, pybind11::call_guard<Drained>());
  module.def("view_index", [](const std::vector<NSUInteger> &shape,const std::vector<NSInteger> &strides,const std::string &index) {
    TORCH_CHECK(shape.size()==strides.size(),"an index view needs one stride per axis");
    return std::string([MetalViewIndexExpression(shape.data(),strides.data(),shape.size(),@(index.c_str())) UTF8String]);
  }, pybind11::call_guard<Drained>());
  module.def("index_values", [](NSUInteger binding,const std::vector<NSInteger> &values) {
    return std::string(MetalViewIndexValuesDeclaration(binding,values.data(),values.size()).UTF8String);
  }, pybind11::call_guard<Drained>());
  pybind11::dict compilation;
#define MESH_METAL_VALUE(name, value) compilation[#name] = value;
#include "metal.def"
#undef MESH_METAL_VALUE
  module.attr("compilation") = compilation;
  module.def("linear_groups", &linear_groups, pybind11::call_guard<Drained>());
  pybind11::class_<Commands>(module,"Commands").def(pybind11::init<const std::vector<CommandRecord> &,const std::optional<CommandReaches> &,const std::vector<CommandHost> &>(),
    pybind11::arg("records"),pybind11::arg("reaches")=pybind11::none(),pybind11::arg("hosts")=std::vector<CommandHost>{});
  module.def("device_facts", &device_facts, pybind11::call_guard<Drained>());
  module.def("pipeline_facts", &pipeline_facts, pybind11::call_guard<Drained>());
  module.attr("polls") = (int64_t)NCCL_MESH_POLLS;

  module.def("attach", &attach, pybind11::call_guard<Drained>());
  module.def("commit", &commit, pybind11::call_guard<Drained>());
  module.def("publish_boundary", &publish_boundary, pybind11::call_guard<Drained>());
  module.def("region_empty", &region_empty, pybind11::call_guard<Drained>());
  module.def("address", &address, pybind11::call_guard<Drained>());
  module.def("records", &region_allocations, pybind11::call_guard<Drained>());   // the region tensors' leases live now: each its address and bytes
  module.def("metal_prelude", &metal_prelude, pybind11::call_guard<Drained>());
  pybind11::class_<ScheduleBinding>(module,"Schedule")
    .def(pybind11::init<>(), pybind11::call_guard<Drained>())
    .def("append",[](ScheduleBinding &s,Commands &commands){commands.append(s.schedule.get());
        s.operands.insert(s.operands.end(),commands.operands.begin(),commands.operands.end());}, pybind11::call_guard<Drained>())
    .def("inputs",&ScheduleBinding::inputs,pybind11::call_guard<Drained>())
    .def("results",&ScheduleBinding::results,pybind11::call_guard<Drained>())
    .def("copy",&ScheduleBinding::copy,pybind11::call_guard<Drained>())
    .def("compute",&ScheduleBinding::compute, pybind11::call_guard<Drained>())
    .def("host",&ScheduleBinding::host, pybind11::arg("entry"), pybind11::arg("reaches"),
         pybind11::call_guard<Drained>())
    .def("finish",&ScheduleBinding::finish, pybind11::call_guard<Drained>())
    .def("enqueue",&ScheduleBinding::enqueue, pybind11::call_guard<Drained>())
    .def("completion",&ScheduleBinding::completion, pybind11::call_guard<Drained>());
  pybind11::class_<Occurrence,std::shared_ptr<Occurrence>>(module,"Occurrence")
    .def(pybind11::init<int64_t,std::vector<at::Tensor>>(),pybind11::arg("comm"),pybind11::arg("operands"),pybind11::call_guard<Drained>())
    .def("lower",&Occurrence::lower,pybind11::arg("invocation")=0,pybind11::call_guard<Drained>())
    .def("source_views",&occurrence_source_views,pybind11::call_guard<Drained>())
    .def("transports",&occurrence_transports,pybind11::call_guard<Drained>())
    .def("publications",&occurrence_publications,pybind11::call_guard<Drained>())
    .def("append",[](const std::shared_ptr<Occurrence> &c,ScheduleBinding &s,const std::vector<size_t> &commands){s.transfer(c,commands);},pybind11::call_guard<Drained>());
  pybind11::enum_<MetalPageBackend>(module,"PageBackend")
    .value("Metal",MetalPageMetal).value("CPU",MetalPageCPU).value("ANE",MetalPageANE);
  pybind11::enum_<MetalPageSiteScope>(module,"PageSiteScope")
    .value("Lane",MetalPageSiteLane).value("Threadgroup",MetalPageSiteThreadgroup).value("Simdgroup",MetalPageSiteSimdgroup);
  module.def("page_group_relation",&page_group_relation,pybind11::call_guard<Drained>());
  module.def("compile_page_effects",&compile_page_effects,pybind11::call_guard<Drained>());
  pybind11::class_<Executions>(module,"Executions")
    .def(pybind11::init<pybind11::object>(), pybind11::call_guard<Drained>())
    .def("add",&Executions::add, pybind11::call_guard<Drained>())
    .def("run",&Executions::run, pybind11::call_guard<Drained>())
    .def("completion",&Executions::completion, pybind11::call_guard<Drained>());
  // the physical channels this rank's lists lay (a cable each that ends here): the one count of them, which a
  // kernel's CHANNELS and the host's record layout both take
  module.def("channels", [](const std::string &group) {
    const int64_t handle = communicator_of(c10::IValue(group));
    TORCH_CHECK(handle, "torch_mesh.channels: ", group, " is no live mesh group of this process");
    return physical_channels(communicator(handle));
  }, pybind11::call_guard<Drained>());
  module.def("view_write_operands",&view_write_operands,pybind11::call_guard<Drained>());

}
