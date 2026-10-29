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
#include "metal-recording.h"
#include "nccl.h"
#include "layout.h" /* MESH_PREPARATION_NS: the preparation deadline a kernel compile waits within */

// allocator.mm: torch's MPS allocator made the region's, and its leases live now (address, bytes)
void install_region_allocator();
std::vector<std::pair<uint64_t, uint64_t>> region_allocations();

// Framework stream dependencies and compatibility-call storage. The shared
// native schedule owns prepared execution; this state names MPS data events.
struct MeshStream {
  at::mps::MPSStream *const mps;
  id<MTLSharedEvent> const progress;
  id<MTL4CommandQueue> const prepared;
  // the buffers this stream committed, the k-th signalling k on progress as the GPU reaches its end
  uint64_t committed = 0;
  // whether a group was issued on this stream since the host path last waited for it
  bool streamed = false;
  // the workspace of the group being encoded (the library's workspace allocator's context is the MeshStream, which
  // outlives every call), then each issued group's, by ticket, until the library has retired it (ncclMeshRetired)
  std::vector<at::Tensor> scratch;
  std::map<uint64_t, std::vector<at::Tensor>> holders;

  explicit MeshStream(at::mps::MPSStream *stream) : mps(stream), progress([stream->device() newSharedEvent]),
      prepared([stream->device() newMTL4CommandQueue]) {
    if (!progress || !prepared) {
      [progress release]; [prepared release];
      TORCH_CHECK(false,"preparing Metal context: shared event or command queue unavailable");
    }
  }

  // On torch's queue: the open buffer committed (and waited for, where `wait`), its end signalling the next count.
  void signalled(bool wait) {
    mps->endKernelCoalescing();
    [mps->commandBuffer() encodeSignalEvent:progress value:++committed];
    mps->synchronize(wait ? at::mps::SyncType::COMMIT_AND_WAIT : at::mps::SyncType::COMMIT);
  }

  // On the host: the workspace `held` of the group issued as `ticket` kept until the library retires it (0: a group
  // complete in the stream's order once encoded, whose workspace is let go here, its leases returned once the stream's
  // work on them completes), and the workspace of every group retired by now released.
  void retire(uint64_t ticket, std::vector<at::Tensor> held) {
    if (ticket) holders[ticket] = std::move(held);
    holders.erase(holders.begin(), holders.upper_bound(ncclMeshRetired()));
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

// The library's workspace allocator: a region lease (region_empty), kept in the MeshStream's scratch (`context`) until
// the group is complete in the stream's order (MeshStream::retire); NULL where the region refuses one, which fails the
// group as a value (nccl-mesh.c: "out of GPU memory") instead of an exception thrown through the library.  It runs on
// torch's queue, where the region allocator (allocator.mm) leases with no dispatch onto it.
static at::Tensor region_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps);
static void *workspace(size_t bytes, void *context) {
  auto *s = static_cast<MeshStream *>(context);
  try {
    s->scratch.push_back(region_empty({(int64_t)bytes}, at::empty({0}, at::TensorOptions().dtype(at::kByte)), true));
  } catch (...) {
    return nullptr;
  }
  return (__bridge void *)at::native::mps::getMTLBufferStorage(s->scratch.back());
}


// A dtype's NCCL type; ncclNumTypes for one NCCL lacks (moved as its bytes, never reduced).
static ncclDataType_t kind(at::ScalarType type) {
  switch (type) {
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
  default: return ncclNumTypes;
  }
}

static void checked(int64_t result) {
  TORCH_CHECK(!result, "libnccl-mesh: ", ncclGetErrorString((ncclResult_t)result), ": ", ncclGetLastError(nullptr));
}

static ncclComm_t communicator(int64_t comm) { return (ncclComm_t)(uintptr_t)comm; }

static at::Tensor region_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps);

// One call of a group: its kind, its send and receive tensors (an all-to-all's none where it moves none that way; an
// all-reduce's, reduce's or broadcast's sends alone where it is in place), its counts (a V call's segment a rank, an
// all-to-all's elements to each rank then from each, in its tensors' elements), its operator (an ncclRedOp_t: a
// built-in or one the library made, as the partials merge) and its root (a send's or receive's peer).
enum Kind : int64_t { ALLREDUCE, REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL };
using Tensors = std::vector<std::optional<at::Tensor>>;
using Call = std::tuple<int64_t, Tensors, Tensors, std::vector<int64_t>, int64_t, int64_t>;

// A tensor as the library reads it, with ProcessGroupNCCL's dtype semantics: a collective that moves data moves any
// dtype (one NCCL lacks as its bytes); a reduction reduces a complex tensor as its real view (sum and average) and a bool
// tensor's sum and product as max and min (its average the library's uint8 average: the sum truncated by the ranks, 1
// only where every rank holds 1).  `at` is an ncclMeshBuffer (its storage's MTLBuffer and byte offset) on the
// Metal path and a host pointer on the host path (NULL: none).
struct Operand {
  const void *at = nullptr;
  size_t count = 0;
  ncclDataType_t type = ncclUint8;
  size_t bytes = 0; /* what a WRITE moves: the tensor's bytes, whatever its values */
};

// A group's prepared fragments. The shared view lowerer resolves logical page
// selections and typed record boundaries before these native scalar bindings.
struct Group {
  bool metal = false;
  std::deque<ncclMeshBuffer> buffers;

  const void *where(const at::Tensor &t) {
    if (!t.is_mps()) return t.data_ptr();
    buffers.push_back({(__bridge void *)at::native::mps::getMTLBufferStorage(t), (size_t)(t.storage_offset() * t.element_size())});
    return &buffers.back();
  }
  const at::Tensor &contiguous(const at::Tensor &t) {
    TORCH_CHECK(t.is_mps() == metal, "the mesh backend's group takes MPS tensors or CPU tensors, not both");
    TORCH_CHECK(t.is_contiguous(), "the mesh backend's operand is a contiguous region tensor (torch_mesh.empty); a non-contiguous "
                "one is not copied into one (zero copy: nccl.h Memory)");
    return t;
  }
  Operand moving(const at::Tensor &tensor) {
    const at::Tensor &t = contiguous(tensor);
    const ncclDataType_t type = kind(t.scalar_type());
    return type == ncclNumTypes ? Operand{where(t), (size_t)t.nbytes(), ncclUint8, (size_t)t.nbytes()}
                                : Operand{where(t), (size_t)t.numel(), type, (size_t)t.nbytes()};
  }
  Operand reducing(const at::Tensor &tensor, int64_t &op) {
    const at::Tensor &t = contiguous(tensor);
    if (t.scalar_type() == at::kBool) op = op == ncclSum ? ncclMax : op == ncclProd ? ncclMin : op;
    if (t.is_complex()) {
      TORCH_CHECK(op == ncclSum || op == ncclAvg, "a complex tensor reduces by sum and average alone");
      return {where(t), (size_t)t.numel() * 2, kind(at::toRealValueType(t.scalar_type()))};
    }
    TORCH_CHECK(kind(t.scalar_type()) != ncclNumTypes, "the mesh backend reduces no ", t.scalar_type());
    return {where(t), (size_t)t.numel(), kind(t.scalar_type())};
  }
};

// A call's prepared operands and counts, made before its group is encoded.
struct Issue {
  int64_t kind, op, root;
  std::vector<Operand> sends, receives;
  std::vector<size_t> counts;
  std::vector<void *> source_views, destination_views;
  ncclDataType_t type = ncclUint8;
};

static Issue prepare(const Call &call, Group &g) {
  const auto &[what, sends, receives, counts, given, root] = call;
  Issue i{what, given, root};
  auto tensor = [](const Tensors &list, size_t k) -> const at::Tensor & {
    TORCH_CHECK(k < list.size() && list[k], "a collective call without its tensor ", k);
    return *list[k];
  };
  switch (what) {
  case ALLREDUCE: case REDUCE:
    i.sends = {g.reducing(tensor(sends, 0), i.op)};
    if (!receives.empty()) i.receives = {g.reducing(tensor(receives, 0), i.op)};
    break;
  case BROADCAST:
    i.sends = {g.moving(tensor(sends, 0))};
    if (!receives.empty()) i.receives = {g.moving(tensor(receives, 0))};
    break;
  case ALLGATHER: case REDUCE_SCATTER: {
    // MPI_Allgatherv and MPI_Reduce_scatter (nccl.h ncclMeshAllGatherV, ncclMeshReduceScatterV): rank r's segment is
    // counts[r] elements, the segments packed in rank order in the whole operand (the receive's, the send's), counted
    // in the operand's NCCL elements; equal counts are the library's equal program
    const bool reduces = what == REDUCE_SCATTER;
    const at::Tensor &whole = reduces ? tensor(sends, 0) : tensor(receives, 0);
    for (size_t rank=0;rank<sends.size();rank++)
      i.sends.push_back(reduces ? g.reducing(tensor(sends,rank),i.op) : g.moving(tensor(sends,rank)));
    if (reduces) i.receives = {g.reducing(tensor(receives, 0), i.op)};
    else for (size_t rank = 0; rank < receives.size(); rank++) i.receives.push_back(g.moving(tensor(receives, rank)));
    const size_t per = reduces && whole.is_complex() ? 2 : kind(whole.scalar_type()) == ncclNumTypes ? whole.element_size() : 1;
    for (auto c : counts) i.counts.push_back((size_t)c * per);
    break;
  }
  case ALLTOALL: {
    size_t per=1;
    for(const Tensors *list : {&sends,&receives}) {
      auto &bindings=list==&sends?i.sends:i.receives;
      for(const auto &tensor:*list) {
        bindings.push_back(tensor?g.moving(*tensor):Operand{});
        if(tensor) {
          i.type=bindings.back().type;
          per=kind(tensor->scalar_type())==ncclNumTypes?tensor->element_size():1;
        }
      }
    }
    for(auto count:counts)i.counts.push_back((size_t)count*per);
    break;
  }
  default: TORCH_CHECK(false, "no collective of kind ", what);
  }
  if ((what == ALLREDUCE || what == REDUCE || what == BROADCAST) && !counts.empty()) {
    const auto &t = tensor(sends,0);
    const size_t per = what != BROADCAST && t.is_complex() ? 2 :
      kind(t.scalar_type()) == ncclNumTypes ? t.element_size() : 1;
    i.sends.front().count = (size_t)counts[0] * per;
  }
  for(const auto &view:i.sends)i.source_views.push_back((void *)view.at);
  for(const auto &view:i.receives)i.destination_views.push_back((void *)view.at);
  return i;
}

// The group's calls on stream s (NULL: the host path), one NCCL group.
static ncclResult_t encoded(const std::vector<Issue> &issues, ncclComm_t comm, ncclMeshStream *s) {
  ncclResult_t result = ncclGroupStart();
  for (const Issue &i : issues) {
    const Operand &a = i.sends.empty() ? Operand{} : i.sends[0];
    void *b = i.receives.empty() ? (void *)a.at : (void *)i.receives[0].at;
    const ncclRedOp_t op = (ncclRedOp_t)i.op;
    if (result) break;
    switch (i.kind) {
    case ALLREDUCE: result = ncclAllReduce(a.at, b, a.count, a.type, op, comm, s); break;
    case REDUCE: result = ncclReduce(a.at, b, a.count, a.type, op, (int)i.root, comm, s); break;
    case BROADCAST: result = ncclBroadcast(a.at, b, a.count, a.type, (int)i.root, comm, s); break;
    case ALLGATHER:
      if (i.receives.size() == 1) result = ncclMeshAllGatherV(a.at, b, i.counts.data(), a.type, comm, s);
      else result = ncclMeshAllGatherBuffers(a.at, i.destination_views.data(), i.counts.data(), a.type, comm, s);
      break;
    case REDUCE_SCATTER:
      result = i.sends.size()==1 ? ncclMeshReduceScatterV(a.at,b,i.counts.data(),a.type,op,comm,s)
        : ncclMeshReduceScatterBuffers(i.source_views.data(),b,i.counts.data(),a.type,op,comm,s);
      break;
    case ALLTOALL:
      result = i.sends.size()==1 && i.receives.size()==1
        ? ncclMeshAlltoAllV(a.at,i.counts.data(),b,i.counts.data()+i.counts.size()/2,i.type,comm,s)
        : ncclMeshAlltoAllBuffers(i.source_views.data(),i.counts.data(),i.destination_views.data(),
                                 i.counts.data()+i.counts.size()/2,i.type,comm,s);
      break;
    }
  }
  const ncclResult_t ended = ncclGroupEnd();
  return result ? result : ended;
}

// design/replay-templates-2026-10-29.md: a group's calls as its crossing's declaration, and the crossing made again
// from one. Each issue's scalars and counts, then its operands, each with its place (an ncclMeshBuffer on the Metal
// path, a host pointer on the host path); the communicator is a place of its own.
struct DeclaredGroup { ncclComm_t comm; uint64_t repeat; uint32_t issues, metal; };
struct DeclaredIssue { int64_t kind, op, root, type; uint64_t sends, receives, counts; };
struct DeclaredOperand { const void *at; size_t offset, count, bytes; int64_t type; };

static void declare_group(const std::vector<Issue> &issues, ncclComm_t comm, size_t repeat, bool metal,
                          std::vector<uint8_t> &bytes, std::vector<MetalPlace> &places) {
  auto put = [&](const void *data, size_t n) { const auto *b = (const uint8_t *)data; bytes.insert(bytes.end(), b, b + n); };
  const DeclaredGroup head{comm, repeat, (uint32_t)issues.size(), metal};
  places.push_back({offsetof(DeclaredGroup, comm), MetalRelocationCommunicator});
  put(&head, sizeof head);
  for (const Issue &i : issues) {
    const DeclaredIssue d{i.kind, i.op, i.root, (int64_t)i.type, i.sends.size(), i.receives.size(), i.counts.size()};
    put(&d, sizeof d);
    put(i.counts.data(), i.counts.size() * sizeof(size_t));
    for (const auto *list : {&i.sends, &i.receives}) for (const Operand &o : *list) {
      DeclaredOperand e{nullptr, 0, o.count, o.bytes, (int64_t)o.type};
      if (o.at) {
        if (metal) { const auto *b = (const ncclMeshBuffer *)o.at; e.at = b->buffer; e.offset = b->offset; }
        else e.at = o.at;
        places.push_back({bytes.size() + offsetof(DeclaredOperand, at), metal ? MetalRelocationBuffer : MetalRelocationAddress});
      }
      put(&e, sizeof e);
    }
  }
}

struct DeclaredCalls { Group group; std::vector<Issue> issues; ncclComm_t comm = nullptr; size_t repeat = 0; };

static std::shared_ptr<DeclaredCalls> declared_calls(NSData *bytes) {
  auto made = std::make_shared<DeclaredCalls>();
  const auto *at = (const uint8_t *)bytes.bytes, *end = at + bytes.length;
  auto take = [&](void *into, size_t n) {
    TORCH_CHECK(at + n <= end, "template: a collective declaration ends early");
    memcpy(into, at, n); at += n;
  };
  DeclaredGroup head; take(&head, sizeof head);
  made->group.metal = head.metal; made->comm = head.comm; made->repeat = head.repeat;
  for (uint32_t k = 0; k < head.issues; k++) {
    DeclaredIssue d; take(&d, sizeof d);
    Issue i{d.kind, d.op, d.root};
    i.type = (ncclDataType_t)d.type;
    i.counts.resize(d.counts); take(i.counts.data(), d.counts * sizeof(size_t));
    for (auto *list : {&i.sends, &i.receives}) for (uint64_t j = 0; j < (list == &i.sends ? d.sends : d.receives); j++) {
      DeclaredOperand e; take(&e, sizeof e);
      Operand o{nullptr, e.count, (ncclDataType_t)e.type, e.bytes};
      if (e.at && head.metal) { made->group.buffers.push_back({(void *)e.at, e.offset}); o.at = &made->group.buffers.back(); }
      else o.at = e.at;
      list->push_back(o);
    }
    for (const auto &view : i.sends) i.source_views.push_back((void *)view.at);
    for (const auto &view : i.receives) i.destination_views.push_back((void *)view.at);
    made->issues.push_back(std::move(i));
  }
  return made;
}

// A native host entry (run, context, retain, release) as the recorder's host operation, the storage it reads held
// with it; `owned`: the caller's reference to its context passes to the operation, else the operation takes its own.
static MetalHostOperation host_operation(const std::array<uintptr_t,4> &entry, bool owned, std::vector<at::Tensor> held) {
  for (auto pointer : entry) TORCH_CHECK(pointer, "host operation requires a complete native binding");
  using Run = const char *(*)(void *);
  using Lifetime = void (*)(void *);
  auto run = reinterpret_cast<Run>(entry[0]);
  auto context = reinterpret_cast<void *>(entry[1]);
  if (!owned) reinterpret_cast<Lifetime>(entry[2])(context);
  auto bound = std::shared_ptr<void>(context, reinterpret_cast<Lifetime>(entry[3]));
  MetalHostOperation operation = ^NSError * {
    // Tensor handles retain the actual storage, including CPU-only weights.
    (void)held;
    const char *error = run(bound.get());
    return error ? [NSError errorWithDomain:@"mesh.host-operation" code:1 userInfo:@{NSLocalizedDescriptionKey:@(error)}] : nil;
  };
  return [[operation copy] autorelease];
}

using Schedule = std::shared_ptr<MetalSchedule>;

static Schedule schedule_after_inputs() {
  at::mps::MPSStream *stream=at::mps::getCurrentMPSStream();
  MeshStream *source=&stream_of(stream);
  Schedule schedule(MetalScheduleCreateOn(source->prepared,MetalFormEncoded),MetalScheduleFree);
  TORCH_CHECK(schedule,"creating the prepared function command stream");
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{ source->signalled(false); });
  MetalScheduleAfter(schedule.get(),source->progress,source->committed);
  return schedule;
}

static pybind11::capsule schedule_completion(const Schedule &schedule) {
  return pybind11::capsule(new Schedule(schedule),"mesh.prepared-function",[](PyObject *token) {
    delete static_cast<Schedule *>(PyCapsule_GetPointer(token,"mesh.prepared-function"));
  });
}

// each mesh process group's communicator, found by its group (attach) with no Python between (fused_prepare)
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
    // design/replay-templates-2026-10-29.md: made from its declaration, which a template names it by
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
          // design/replay-templates-2026-10-29.md: each constant [index, MTLDataType, bytes] in the pipeline's declaration
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
static Dispatch encoded(int64_t handle, const std::vector<std::optional<at::Tensor>> &buffers, const std::vector<int64_t> &constants,
                        int64_t at, int64_t gx, int64_t gy, int64_t threads) {
  Dispatch d{state_of(handle, "encode"), std::vector<id<MTLBuffer>>(buffers.size(), nil), std::vector<NSUInteger>(buffers.size(), 0), std::vector<NSUInteger>(buffers.size(), 0),
             std::vector<uint32_t>(constants.begin(), constants.end()), (NSUInteger)at, MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, 1),
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
  TensorTraversal(const std::vector<TensorLayout> &layouts,const std::vector<std::pair<NSUInteger,NSUInteger>> &domains) {
    NSMutableArray *parts=[NSMutableArray array],*intervals=[NSMutableArray array];
    for(const auto &layout:layouts)[parts addObject:layout.value];
    for(const auto &[first,end]:domains)[intervals addObject:@[@(first),@(end)]];
    @try { value=[[MetalTensorTraversal layouts:parts domains:intervals] retain]; }
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
  explicit TensorOperand(const pybind11::object &object):
    storage((THPVariable_Check(object.ptr())?object:object.attr("storage")).cast<at::Tensor>()),
    pages(tensorPages(storage)),
    layout([pages bind:(THPVariable_Check(object.ptr())?TensorLayout::tensor(storage):object.attr("layout").cast<TensorLayout>()).value
                 elementBytes:storage.element_size()]){}
  ~TensorOperand(){[pages release];}
};

// FX's output is the native recorder's input. This constructor only binds
// storage and constant operands; it neither runs nor records a framework call.
using CommandRecord = std::tuple<int64_t, std::vector<pybind11::object>,
    std::map<uint32_t,pybind11::bytes>, std::array<NSUInteger,3>, std::array<NSUInteger,3>, bool, uint64_t>;
static std::vector<pybind11::object> commandOperands(const std::vector<std::optional<at::Tensor>> &tensors) {
  std::vector<pybind11::object> result;for(const auto &t:tensors)result.push_back(t?pybind11::cast(*t):pybind11::none());return result;
}
struct Commands {
  MetalReplay *replay=nullptr;
  std::vector<at::Tensor> operands;
  explicit Commands(const std::vector<CommandRecord> &records) {
    if(records.empty())return;
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
      for(size_t j=0;j<buffers.size();j++)if(!buffers[j].is_none()) {
        const TensorOperand operand(buffers[j]);const auto &t=operand.storage;const auto &layout=operand.layout;
        id<MTLBuffer> buffer=operand.pages.buffer;
        const uint32_t mode=((access>>j)&1?MetalReachRead:0)|((access>>(MESH_ACCESS_WRITE_SHIFT+j))&1?MetalReachWritten:0);
        bound.push_back({buffer,(NSUInteger)layout.value.offset*t.element_size(),layout.value.count*t.element_size(),(uint32_t)j,mode});
        if(mode) {
          NSData *data=[layout.value reaches:buffer base:0 elementBytes:t.element_size() access:mode binding:(int32_t)j];
          const auto *spans=(const MetalReach *)data.bytes;
          ranges.insert(ranges.end(),spans,spans+data.length/sizeof *spans);
        }
        operands.push_back(t);
      }
      for(const auto &constant:constants[k])
        bound.push_back({constantBuffer,constantOffset+constant[1],constant[2],(uint32_t)constant[0],MetalReachRead});
      descriptions.push_back({state_of(handle,"commands"),bound.data(),bound.size(),nullptr,0,ranges.data(),ranges.size(),
        MTLSizeMake(grid[0],grid[1],grid[2]),MTLSizeMake(group[0],group[1],group[2]),(BOOL)threadGrid});
    }
    replay=MetalReplayCommands(device,descriptions.data(),descriptions.size(),MetalFormEncoded);
    TORCH_CHECK(replay,"native command preparation failed");
  }
  ~Commands(){MetalReplayFree(replay);}
  void append(MetalSchedule *schedule) {
    if(!replay)return;
    NSError *error=MetalScheduleAppend(schedule,replay,NSMakeRange(0,1),@[],nil);
    TORCH_CHECK(!error,error?error.localizedDescription.UTF8String:"");
  }
};
// Fixed operands for a declared block graph. Shared lowering supplies scalar
// fragments and any required numerical conversion edges. They record through
// the same native schedule; invocation creates no layout or tensor bindings.
struct Collectives {
  int64_t comm;
  Group group{true};
  std::vector<Issue> issues;
  std::vector<at::Tensor> operands;
  Commands before, after;
  Collectives(const std::vector<Call> &calls, int64_t comm, std::vector<at::Tensor> operands,
              const std::vector<CommandRecord> &inputs, const std::vector<CommandRecord> &outputs)
      : comm(comm), operands(std::move(operands)), before(inputs), after(outputs) {
    this->operands.insert(this->operands.end(),before.operands.begin(),before.operands.end());
    this->operands.insert(this->operands.end(),after.operands.begin(),after.operands.end());
    for (const Call &call : calls) issues.push_back(prepare(call, group));
  }
  std::vector<Schedule> executions;
  size_t next=0;
  void append(MetalSchedule *schedule,size_t repetitions=1) {
    const auto *list=&issues;const ncclComm_t handle=communicator(comm);
    MetalScheduleCommunicator(schedule,handle);
    const size_t batches=before.replay||after.replay?repetitions:(size_t)(repetitions!=0);
    const size_t repeat=batches==1?repetitions:1;
    std::vector<uint8_t> declared;std::vector<MetalPlace> places;
    declare_group(*list,handle,repeat,group.metal,declared,places);
    const MetalDeclaration declaration{"collectives",declared.data(),declared.size(),places.data(),places.size()};
    for(size_t k=0;k<batches;k++) {
      before.append(schedule);
      checked(MetalScheduleCross(schedule,&declaration,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
        id<MTLResidencySet> residency,NSUInteger,NSUInteger) {
      ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
        .residencySet=(__bridge void *)residency};
      for(size_t r=0;r<repeat;r++) {
        const int result=(int)encoded(*list,handle,&on);
        if(result)return result;
      }
      return 0;
    }));
      after.append(schedule);
    }
  }
  void realize(size_t repetitions=1) {
    auto made=schedule_after_inputs();append(made.get(),repetitions);
    const auto held=operands;
    checked(MetalScheduleFinish(made.get(),^(NSError *){(void)held;}));executions.push_back(std::move(made));
  }
  void enqueue() { MetalScheduleEnqueue(executions[next++].get()); }
  pybind11::capsule completion() { return schedule_completion(executions[next-1]); }

};

static void encode_in(id<MTLComputeCommandEncoder> encoder, const Dispatch &d) {
  [encoder setComputePipelineState:d.state];
  for (size_t i = 0; i < d.bound.size(); i++)
    if (d.bound[i]) [encoder setBuffer:d.bound[i] offset:d.offsets[i] atIndex:i];
  [encoder setBytes:d.words.data() length:d.words.size() * sizeof(uint32_t) atIndex:d.at];
  [encoder dispatchThreadgroups:d.grid threadsPerThreadgroup:d.group];
}
// Configuration binds kernels from the Torch frontend into the native recorder.
// Access modes are explicit operands; the recorder orders the declared byte views.
static void prepare_dispatch(MetalSchedule *schedule,const Dispatch &d,uint64_t access) {
  std::vector<MetalCommandBinding> bindings;
  for(size_t j=0;j<d.bound.size();j++)if(d.bound[j])
    bindings.push_back({d.bound[j],d.offsets[j],d.lengths[j],(uint32_t)j,
      (uint32_t)(((access>>j)&1?MetalReachRead:0)|((access>>(MESH_ACCESS_WRITE_SHIFT+j))&1?MetalReachWritten:0))});
  bindings.push_back(MetalScheduleData(schedule,d.words.data(),d.words.size()*sizeof(uint32_t),(uint32_t)d.at));
  MetalScheduleCompute(schedule,d.state,bindings.data(),bindings.size(),d.grid,d.group);

}

static void encode(int64_t handle, const std::vector<std::optional<at::Tensor>> &buffers, const std::vector<int64_t> &constants,
                   int64_t at, int64_t gx, int64_t gy, int64_t threads) {
  const Dispatch d = encoded(handle, buffers, constants, at, gx, gy, threads);
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    encode_in(encoder, d);
  });
}

// Linear command geometry is compiled from the shared manifest and discovered GPU cores.
static std::string linear_source_;
static int64_t linear_cores_ = 0;
static std::map<std::tuple<int, int, int, int, int>, int64_t> linear_pipelines_;
static constexpr const char *LINEAR_PARAMETERS[]={"device const half* x", "device const half* w", "device half* y", "device const half* b", "device const half* s"};
constexpr size_t LINEAR_BINDINGS = sizeof(LINEAR_PARAMETERS) / sizeof(*LINEAR_PARAMETERS);
constexpr size_t LINEAR_PRODUCTS = (MESH_BUFFER_BINDINGS - 1) / LINEAR_BINDINGS;
struct LinearDimensions { uint32_t rows, inner, columns, bias, weight_stride, first, stride0, stride1, output_row, output_column; };
struct LinearGroupWords {
  LinearDimensions dimensions[LINEAR_PRODUCTS];
  uint32_t first[LINEAR_PRODUCTS], activation[LINEAR_PRODUCTS];
};

static void linear_source(const std::string &source, int64_t cores) {
  linear_source_ = source;
  linear_cores_ = cores;
}

static int64_t linear_fit(int64_t K, int64_t groups) {
  int64_t split = 1;
  while (split * 2 <= MESH_LINEAR_SIMDGROUPS && split * groups < linear_cores_ * MESH_LINEAR_GROUPS_PER_CORE &&
         K / (split * 2) >= MESH_LINEAR_MIN_INNER) split *= 2;
  return split;
}
static int64_t linear_groups(int64_t rows, int64_t columns) {
  return ((rows + MESH_LINEAR_ROWS - 1) / MESH_LINEAR_ROWS) * ((columns + MESH_LINEAR_COLUMNS - 1) / MESH_LINEAR_COLUMNS);
}
static int64_t linear_split(int64_t K, int64_t N, int64_t rows) { return linear_fit(K, linear_groups(rows, N)); }

static std::string linear_group_source(size_t products) {
  const std::string count=std::to_string(products);
  std::string source="\nstruct linear_group_words { linear_dims d["+count+"]; uint first["+count+"], act["+count+"]; };\n"
    "template <uint VW, uint VX, uint COLS, uint NSG, uint U>\n"
    "[[max_total_threads_per_threadgroup(NSG * MESH_SIMD_WIDTH)]] kernel void linear_group(";
  for(size_t p=0;p<products;p++) {
    const std::string n=std::to_string(p);
    for(size_t k=0;k<LINEAR_BINDINGS;k++)source+=std::string(LINEAR_PARAMETERS[k])+n+" [[buffer("+std::to_string(LINEAR_BINDINGS*p+k)+")]],";
  }
  source+="constant linear_group_words& g [[buffer("+std::to_string(products*LINEAR_BINDINGS)+")]],"
    "uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],"
    "uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {"
    "threadgroup float part[NSG*MESH_LINEAR_ROWS*COLS];";
  for(size_t p=products;p-- > 0;) {
    const std::string n=std::to_string(p);
    source+="if(tg>=g.first["+n+"]) { const linear_act_store store{y"+n+",b"+n+",g.d["+n+"].LDY,g.d["+n+"].SDY,g.d["+n+"].B,g.act["+n+"]};"
      "linear_body<VW,VX,COLS,NSG,U>(x"+n+",s"+n+",w"+n+",g.d["+n+"],store,part,sg,lane,tg-g.first["+n+"],tid);return;}";
  }
  return source+"}\n";
}

static int64_t linear_pipeline(const std::tuple<int, int, int, int, int> &key) {
  const auto found = linear_pipelines_.find(key);
  if (found != linear_pipelines_.end()) return found->second;
  const int64_t split = std::get<2>(key), act = std::get<3>(key);
  std::string args = std::to_string(std::get<0>(key)) + ", " + std::to_string(std::get<1>(key)) + ", " +
                     std::to_string(MESH_LINEAR_COLUMNS) + ", " + std::to_string(split) + ", " + std::to_string(MESH_LINEAR_LOADS);
  const std::string kernel = act < 0 ? "linear_group" : "linear_rows";
  if (act >= 0) args += ", " + std::to_string(act);
  std::string name = kernel + "_" + args + "_p" + std::to_string(std::get<4>(key));
  for (auto &ch : name) ch = ch == ',' || ch == ' ' ? '_' : ch;
  return linear_pipelines_.emplace(key, pipeline(linear_source_ + (act < 0 ? linear_group_source(std::get<4>(key)) : "") + "\ntemplate [[host_name(\"" + name + "\")]] [[kernel]] decltype(" +
                                         kernel + "<" + args + ">) " + kernel + "<" + args + ">;\n", name)).first->second;
}

// Independent products share a dispatch up to the manifest binding capacity.
// Row tiles and column tiles both contribute independent groups.
static std::vector<CommandRecord> linear_commands(const std::vector<at::Tensor> &xs, const std::vector<at::Tensor> &ws,
                                       const std::vector<std::optional<at::Tensor>> &biases, const std::vector<int64_t> &acts,
                                       const std::vector<std::optional<at::Tensor>> &x1s, const std::vector<at::Tensor> &ys) {
  const size_t P = xs.size();
  TORCH_CHECK(P >= 1 && P <= LINEAR_PRODUCTS && ws.size() == P && biases.size() == P && acts.size() == P && x1s.size() == P && ys.size() == P,
              "torch_mesh::linears exceeds the manifest binding capacity or lacks product operands");
  std::vector<std::optional<at::Tensor>> buffers(LINEAR_BINDINGS * P);
  LinearGroupWords words{};
  int64_t first = 0, least = INT64_MAX;
  bool wide_w = true, wide_x = true;
  for (size_t p = 0; p < P; p++) {
    const at::Tensor &x = xs[p], &w = ws[p];
    const int64_t K0 = x.size(1), K = K0 + (x1s[p] ? x1s[p]->size(1) : 0);
    TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf && x.dim() == 2 && w.dim() == 2 && w.size(1) == K,
                "torch_mesh::linears: product ", p, " takes fp16 x [R, K0], x1 [R, K - K0] and w [N, K]");
    TORCH_CHECK(x.stride(1)==1 && w.stride(1)==1 && (!x1s[p] || x1s[p]->stride(1)==1) &&
        (!biases[p] || biases[p]->is_contiguous()),
        "linear commands require declared row views; implicit contiguous copies are not supported");
    const at::Tensor &xc=x, &wc=w, &x1c=x1s[p]?*x1s[p]:x;
    const int64_t R = x.size(0), N = w.size(0), ldw = wc.stride(0), ld0 = xc.stride(0), ld1 = x1c.stride(0);
    buffers[LINEAR_BINDINGS * p] = xc, buffers[LINEAR_BINDINGS * p + 1] = wc, buffers[LINEAR_BINDINGS * p + 2] = ys[p];
    buffers[LINEAR_BINDINGS * p + 3] = biases[p] ? biases[p] : std::make_optional(ys[p]), buffers[LINEAR_BINDINGS * p + 4] = x1c;
    words.dimensions[p] = {(uint32_t)R, (uint32_t)K, (uint32_t)N, (uint32_t)bool(biases[p]),
                           (uint32_t)ldw, (uint32_t)K0, (uint32_t)ld0, (uint32_t)ld1,
                           (uint32_t)ys[p].stride(0), (uint32_t)ys[p].stride(1)};
    words.first[p] = first; words.activation[p] = acts[p];
    first += linear_groups(R, N);
    least = std::min(least, K);
    wide_w &= ldw % MESH_LINEAR_VECTOR == 0 && K0 % MESH_LINEAR_VECTOR == 0 && wc.storage_offset() % MESH_LINEAR_VECTOR == 0;
    wide_x &= K0 % MESH_LINEAR_VECTOR == 0 && ld0 % MESH_LINEAR_VECTOR == 0 && ld1 % MESH_LINEAR_VECTOR == 0 &&
              xc.storage_offset() % MESH_LINEAR_VECTOR == 0 && x1c.storage_offset() % MESH_LINEAR_VECTOR == 0;
  }
  const int64_t split = linear_fit(least, first);
  if(P==1) {
    buffers.resize(6);buffers[5]=buffers[4];buffers[4]=std::nullopt;
    return {{linear_pipeline(std::make_tuple(wide_w?MESH_LINEAR_VECTOR:1,wide_x?MESH_LINEAR_VECTOR:1,(int)split,(int)acts[0],1)),commandOperands(buffers),
        {{4,pybind11::bytes((char *)&words.dimensions[0],sizeof(LinearDimensions))}},
        {(NSUInteger)first,1,1},{(NSUInteger)(split*MESH_SIMD_WIDTH),1,1},false,UINT64_C(43)|(UINT64_C(4)<<MESH_ACCESS_WRITE_SHIFT)}};
  }
  const auto key = std::make_tuple(wide_w ? MESH_LINEAR_VECTOR : 1, wide_x ? MESH_LINEAR_VECTOR : 1, (int)split, -1, (int)P);
  std::string packed((char *)words.dimensions,P*sizeof(LinearDimensions));
  packed.append((char *)words.first,P*sizeof(*words.first));
  packed.append((char *)words.activation,P*sizeof(*words.activation));
  uint64_t access=0;
  for(size_t p=0;p<P;p++)access|=(UINT64_C(27)<<(LINEAR_BINDINGS*p))|(UINT64_C(4)<<(MESH_ACCESS_WRITE_SHIFT+LINEAR_BINDINGS*p));
  return {{linear_pipeline(key),commandOperands(buffers),{{LINEAR_BINDINGS*P,pybind11::bytes(packed)}},
      {(NSUInteger)first,1,1},{(NSUInteger)(split*MESH_SIMD_WIDTH),1,1},false,access}};
}

TORCH_LIBRARY_FRAGMENT(torch_mesh, m) {
  m.def("linear(Tensor x, Tensor w, Tensor? bias=None, int act=0, Tensor? x1=None) -> Tensor");
  m.def("linears(Tensor[] x, Tensor[] w, Tensor?[] bias, int[] act, Tensor?[] x1) -> Tensor[]");
  m.def("fused_step(Tensor x, Tensor? w, int dim, int[] sizes, str group_name, bool adds, int repeats) -> Tensor");
  m.def("fused_step_out(Tensor x, Tensor? w, Tensor(a!) out, int dim, int[] sizes, str group_name, bool adds, int repeats) -> ()");
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

// Import frontend attributes once, before recording the function.
// Existing bridge operands retain their pages. Aliases of foreign storage
// share one region allocation and retain their typed views of the same bytes.
static std::vector<at::Tensor> import_attributes(const std::vector<at::Tensor> &values) {
  struct Import {
    std::vector<size_t> members;
    std::vector<std::pair<size_t,size_t>> spans;
    size_t low=std::numeric_limits<size_t>::max(),high=0,alignment=1;
  };
  std::map<const c10::StorageImpl *,Import> groups;
  std::vector<at::Tensor> result=values;
  bool mps=false;
  for(size_t i=0;i<values.size();i++) {
    const auto &value=values[i];
    if(address(value))continue;
    TORCH_CHECK(value.is_cpu()||value.is_mps(),"attribute import requires declared CPU or MPS storage");
    auto &group=groups[value.storage().unsafeGetStorageImpl()];group.members.push_back(i);
    const size_t width=value.element_size();group.alignment=std::lcm(group.alignment,width);
    NSData *data=[TensorLayout::tensor(value).value orderedSpans];
    const auto *spans=(const MetalTensorSpan *)data.bytes;
    for(size_t j=0;j<data.length/sizeof *spans;j++) {
      const size_t first=spans[j].offset*width,last=first+spans[j].count*width;
      group.spans.emplace_back(first,last);group.low=std::min(group.low,first);group.high=std::max(group.high,last);
    }
    mps|=value.is_mps()&&value.numel();
  }
  // A foreign MPS attribute may still be produced by its frontend's setup
  // commands. Complete that producer before the host reads its initial bytes.
  // This import is called only by preparation, never by a recorded invocation.
  if(mps)at::mps::synchronizeAllMPSStreams(at::mps::SyncType::COMMIT_AND_WAIT);
  for(auto &[key,group]:groups) {
    const auto &source=values[group.members.front()];
    const size_t low=group.spans.empty()?0:group.low/group.alignment*group.alignment;
    const size_t bytes=group.spans.empty()?0:(group.high-low+group.alignment-1)/group.alignment*group.alignment;
    at::Tensor storage=region_empty({(int64_t)bytes},at::empty({0},at::TensorOptions().dtype(at::kByte)),source.is_mps());
    // the source storage's bytes as a tensor over it, its spans' merged runs copied by torch on its device (a private
    // MPS buffer has no host view; no address is read)
    const at::Tensor from=at::empty({0},source.options().dtype(at::kByte)).set_(source.storage(),0,{(int64_t)source.storage().nbytes()},{1});
    std::sort(group.spans.begin(),group.spans.end());
    size_t first=0,last=0;
    const auto copy=[&](size_t a,size_t b){if(b>a)storage.narrow(0,(int64_t)(a-low),(int64_t)(b-a)).copy_(from.narrow(0,(int64_t)a,(int64_t)(b-a)));};
    for(const auto &span:group.spans) {
      if(span.first<=last){last=std::max(last,span.second);continue;}
      copy(first,last);
      first=span.first;last=span.second;
    }
    copy(first,last);
    for(size_t i:group.members) {
      const auto &value=values[i];const size_t width=value.element_size();
      const size_t offset=storage.storage_offset()+(value.numel()?value.storage_offset()*width-low:0);
      result[i]=storage.view(value.scalar_type()).as_strided(value.sizes(),value.strides(),offset/width);
    }
  }
  // the copies complete before a prepared function, on its own queue, reads them (preparation only)
  if(mps)at::mps::synchronizeAllMPSStreams(at::mps::SyncType::COMMIT_AND_WAIT);
  return result;
}

// Fused crossings share immutable exact-byte WritePrepare/Execute plans.
// GPU bindings name physical channels, including every parallel cable and
// compiled relay path. Producers publish source ranges; consumers observe
// only their data dependencies. Registration cuts are agreed at realization,
// and no buffer must begin on a transport page. Unsupported inputs fail.
// The library's Metal prelude (ncclMeshMetalPrelude): rdma/mesh.metal as the loaded library was built with it, then its
// steps; the text a kernel that runs them is compiled after.
static std::string metal_prelude() { return ncclMeshMetalPrelude(); }

// Prepared handles retain the same exact-byte contract for generic gathers,
// reductions and the public indexed/affine write API. No collective-specific
// transport planner is selected by a generic data function.
static int physical_channels(ncclComm_t comm) {
  const mesh_fabric *fabric = nullptr;
  int rank = 0, channels = 0;
  checked(ncclCommUserRank(comm, &rank));
  checked(ncclMeshCommFabric(comm, &fabric));
  for (uint32_t k = 0; k < fabric->cables; k++)
    channels += fabric->cable[k].node[0] == (uint32_t)rank || fabric->cable[k].node[1] == (uint32_t)rank;
  return channels;
}

// The most entries one repeat of a group of WRITEs, `root` to each of `destinations` with `count` elements of
// `element_bytes`, lays on any directed cable: every rank's transfers compiled as libnccl-mesh's write_steps compiles
// them (mesh_write_compile over the communicator's fabric, installed paths and page), its SENDs counted a cable (a
// cable's send list at one end and its landing list at the other hold the same entries). A group of R repeats lays R
// times this on that list, and a list holds MESH_STREAM_ENTRIES (write_window).
static int64_t write_entries(const std::string &group, int64_t root, const std::vector<int64_t> &destinations,
                             int64_t count, int64_t element_bytes) {
  const int64_t handle = communicator_of(c10::IValue(group));
  TORCH_CHECK(handle, "torch_mesh.write_entries: ", group, " is no live mesh group of this process");
  ncclComm_t comm = communicator(handle);
  const mesh_fabric *fabric = nullptr;
  size_t page = 0;
  int ranks = 0;
  checked(ncclMeshCommFabric(comm, &fabric));
  checked(ncclMeshCommPage(comm, &page));
  checked(ncclCommCount(comm, &ranks));
  std::map<std::pair<uint32_t, uint32_t>, int64_t> laid;
  const mesh_operand operand{1, (uint32_t)element_bytes, (uint64_t)count};
  for (const int64_t destination : destinations) {
    const mesh_paths *paths = nullptr;
    checked(ncclMeshCommWritePaths(comm, (int)root, (int)destination, &paths));
    for (int r = 0; r < ranks; r++) {
      const int needed = mesh_write_compile(fabric, (uint32_t)r, (uint32_t)root, (uint32_t)destination, paths, operand, page, 0,
                                            nullptr, 0);
      TORCH_CHECK(needed >= 0, "torch_mesh.write_entries: mesh_write_compile ", root, " to ", destination, ": ", strerror(-needed));
      std::vector<mesh_transfer> made((size_t)needed + 1);
      const int got = mesh_write_compile(fabric, (uint32_t)r, (uint32_t)root, (uint32_t)destination, paths, operand, page, 0,
                                         made.data(), (uint32_t)made.size());
      for (int k = 0; k < got; k++)
        if (made[(size_t)k].step.op == MESH_STEP_SEND) laid[{made[(size_t)k].cable, (uint32_t)r}]++;
    }
  }
  int64_t most = 0;
  for (const auto &entry : laid) most = std::max(most, entry.second);
  return most;
}

struct PreparedWrites {
  std::vector<ncclMeshWritePlan_t> handles;
  std::vector<std::array<uint64_t, 6>> publish, land;
  ncclMeshFused records{};
  uint32_t polls = 0;
  ncclComm_t comm = nullptr;
  void prepare(int64_t comm, int64_t waiting, int64_t repeats, int64_t stride) {
    this->comm=communicator(comm);
    const int channels = physical_channels(communicator(comm));
    publish.resize(channels), land.resize(channels);
    records = {reinterpret_cast<uint64_t(*)[6]>(publish.data()), reinterpret_cast<uint64_t(*)[6]>(land.data()),
               (uint32_t)channels, (uint32_t)channels, (uint32_t)repeats, (size_t)stride};
    polls = (uint32_t)waiting;
  }
  ~PreparedWrites() { for (auto plan : handles) ncclMeshWritePlanDestroy(plan); }
};

static void prepare_fused_writes(MetalSchedule *schedule,PreparedWrites &plans) {
  MetalScheduleCommunicator(schedule,plans.comm);
  plans.records.publishes=plans.publish.size();plans.records.lands=plans.land.size();
  if(plans.handles.empty()){plans.records.publishes=plans.records.lands=0;return;}
  PreparedWrites *p=&plans;
  // design/replay-templates-2026-10-29.md: a WRITE plan's crossing has no declaration yet, so its function makes no template
  checked(MetalScheduleCross(schedule,nullptr,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
      id<MTLResidencySet> residency,NSUInteger,NSUInteger) {
    ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
      .residencySet=(__bridge void *)residency,.polls=p->polls,.fused=&p->records};
    ncclResult_t result=ncclGroupStart();
    for(auto handle:p->handles)if(!result)result=ncclMeshWriteExecute(handle,0,&on);
    ncclResult_t ended=ncclGroupEnd();return (int)(result?result:ended);
  }));
}

// Configuration retains the kernels, page generations and channel bindings of a
// crossing. Each compiled occurrence gets its own indexed dispatch and output.
// Execution reads that index and fills the changing operands; it neither looks
// up a form nor allocates a result. Forms live until their communicator changes
// configuration or shuts down.
struct FusedForm {
  int64_t comm = 0, polls = 0, ranks = 0, rank = 0, publish = -1, land = -1, pj = 0, lj = 0, pgx = 0, pthreads = 0, lgx = 0, lthreads = 0,
          slot = 0, ready = -1, channels = 0;
  bool with_w = false;
  std::vector<int64_t> pwords, lwords, starts, counts, shape;
  at::Tensor arrivals;
  struct Ring {
    std::vector<at::Tensor> slots[2];
    std::vector<std::vector<at::Tensor>> segments[2];
    std::vector<at::Tensor> relays[2];
    std::shared_ptr<PreparedWrites> plans[2];
  };
  std::map<int64_t, Ring> rings;
  std::map<int64_t, std::deque<Ring>> prepared_fresh;
  struct Sequence {
    std::vector<Ring *> rings;
    size_t cursor = 0, stride = 0;
  };
  std::map<int64_t, Sequence> sequences;
  Ring *ring = nullptr;
  int64_t turn = 0, j = 0, half = 0;
  std::vector<int64_t> pbind, lbind;
  Schedule transport;
};
static auto *fused_forms_ = new std::map<std::vector<int64_t>, FusedForm>;
static auto *fused_lock_ = new std::mutex;
static pybind11::object *fused_maker_ = nullptr;

static void fused_maker(pybind11::object maker) { fused_maker_ = new pybind11::object(std::move(maker)); }

// The form of a crossing on `group`'s communicator, made during preparation; `held` (fused_lock_) is let go
// while the maker runs (it takes the GIL, which a thread waiting on the lock may hold)
static FusedForm &fused_form_of(const at::Tensor &x, const std::optional<at::Tensor> &w, int64_t dim, const std::vector<int64_t> &sizes,
                                const std::string &group, bool adds, std::unique_lock<std::mutex> &held) {
  const int64_t comm = communicator_of(c10::IValue(group));
  TORCH_CHECK(comm, "torch_mesh::fused_step: ", group, " is no live mesh group of this process");
  std::vector<int64_t> key(x.sizes().begin(), x.sizes().end());
  key.insert(key.end(), {-1, (int64_t)x.scalar_type(), dim, comm, adds ? 1 : 0, -2});
  key.insert(key.end(), sizes.begin(), sizes.end());
  key.push_back(-4);
  key.insert(key.end(), x.strides().begin(), x.strides().end());
  if (w) {
    key.push_back(-3);
    key.insert(key.end(), w->sizes().begin(), w->sizes().end());
    key.insert(key.end(), w->strides().begin(), w->strides().end());
  }
  auto found = fused_forms_->find(key);
  if (found != fused_forms_->end()) return found->second;
  TORCH_CHECK(fused_maker_, "torch_mesh::fused_step: no form maker (partition registers it as it loads)");
  FusedForm f;
  held.unlock();
  {
    pybind11::gil_scoped_acquire gil;
    const int channels = physical_channels(communicator(comm));
    const pybind11::tuple t = (*fused_maker_)(x, w ? pybind11::cast(*w) : pybind11::none(), dim, sizes, group, adds, channels);
    f.comm = t[0].cast<int64_t>(), f.polls = t[1].cast<int64_t>(), f.ranks = t[2].cast<int64_t>(), f.rank = t[3].cast<int64_t>();
    f.publish = t[4].cast<int64_t>(), f.pwords = t[5].cast<std::vector<int64_t>>(), f.pj = t[6].cast<int64_t>();
    f.pgx = t[7].cast<int64_t>(), f.pthreads = t[8].cast<int64_t>();
    f.land = t[9].cast<int64_t>(), f.lwords = t[10].cast<std::vector<int64_t>>(), f.lj = t[11].cast<int64_t>();
    f.lgx = t[12].cast<int64_t>(), f.lthreads = t[13].cast<int64_t>();
    f.starts = t[14].cast<std::vector<int64_t>>(), f.counts = t[15].cast<std::vector<int64_t>>(), f.shape = t[16].cast<std::vector<int64_t>>();
    f.arrivals = t[17].cast<at::Tensor>(), f.with_w = w.has_value(), f.slot = f.starts.back();
    f.ready = t[18].cast<int64_t>(), f.channels = t[19].cast<int64_t>();
  }
  held.lock();
  TORCH_CHECK(f.comm == comm, "torch_mesh::fused_step: ", group, "'s form names another communicator");
  return fused_forms_->try_emplace(key, std::move(f)).first->second;
}

static void release_chains(int64_t comm);

// every form on communicator `comm` released, its leases back to the region once the stream has passed them (the
// group's shutdown: ProcessGroupMesh.shutdown, as destroy_process_group ends it)
static void fused_release(int64_t comm) {
  std::lock_guard<std::mutex> guard(*fused_lock_);
  release_chains(comm);
  for (auto f = fused_forms_->begin(); f != fused_forms_->end();) f = f->second.comm == comm ? fused_forms_->erase(f) : std::next(f);
}

// Route changes happen between complete functions. A partially issued repeat
// train still owns its schedule; discarding its bindings would strand data.
static void fused_reconfigure(int64_t comm) {
  std::lock_guard<std::mutex> guard(*fused_lock_);
  release_chains(comm);
  for (const auto &entry : *fused_forms_)
    TORCH_CHECK(entry.second.comm != comm || entry.second.j == 0,
                "route reconfiguration requires a complete fused repeat train");
  for (auto f = fused_forms_->begin(); f != fused_forms_->end();)
    f = f->second.comm == comm ? fused_forms_->erase(f) : std::next(f);
}

// Allocate and agree registration cuts without presenting a receive or
// executing a producer. A fresh generation owns every source/landing/relay lease.
// A tensor's WRITE from its abstract layout (nccl.h ncclMeshLayout): its storage's page table, its sizes, strides and
// storage offset in elements and its element bits; no address or byte geometry (design/pointer-lowering-2026-10-28.md).
// `frames` frames of `view`, `frame_stride` elements apart, are one train (nccl.h ncclMeshLayout frames).
// An undefined `view` is a rank that neither ends nor forwards the WRITE (ncclMeshCommWriteForwards): no relay, its
// binding an empty page table over `unbound` contiguous elements of `like`'s type, the bytes every rank reports.
static void tensor_write(Group &g, const at::Tensor &view, int source, int destination, ncclComm_t comm, bool metal,
                         ncclMeshWriteLayout layout, ncclMeshWritePlan_t *out, int64_t frames = 1, int64_t frame_stride = 0,
                         int64_t unbound = 0, const at::Tensor &like = at::Tensor()) {
  ncclMeshPageTable_t table = nullptr;
  if (view.defined()) {
    const at::Tensor start = view.as_strided({view.storage().nbytes() ? 1 : 0}, {1}, 0);
    checked(ncclMeshPageTableCreate(g.where(start), comm, metal, &table));
  } else {
    checked(ncclMeshPageTableCreate(nullptr, comm, metal, &table));
  }
  const std::vector<uint64_t> dims = view.defined() ? std::vector<uint64_t>(view.sizes().begin(), view.sizes().end())
                                                    : std::vector<uint64_t>{(uint64_t)unbound};
  const std::vector<int64_t> steps = view.defined() ? std::vector<int64_t>(view.strides().begin(), view.strides().end())
                                                    : std::vector<int64_t>{1};
  const int64_t element = view.defined() ? view.element_size() : like.element_size();
  const ncclMeshLayout local{table, (uint32_t)(8 * element), (uint32_t)dims.size(), dims.data(), steps.data(),
                             (uint64_t)(view.defined() ? view.storage_offset() : 0), nullptr, 0, (uint32_t)frames, frame_stride};
  const ncclResult_t made = ncclMeshWriteLayoutPrepare(&local, source, destination, 0, layout, out);
  ncclMeshPageTableDestroy(table);
  checked(made);
}

static void fused_allocate_ring(FusedForm &f, FusedForm::Ring &ring, const at::Tensor &xc, int64_t repeats) {
  TORCH_CHECK(repeats >= 1 && repeats <= (int64_t)UINT32_MAX && f.slot >= 0 &&
              f.slot <= INT64_MAX / repeats / (int64_t)xc.element_size(),
              "fused gather: repeated byte extent overflows address space");
  if (ring.slots[0].empty()) {
    const at::Tensor leases[2] = {region_empty({repeats * f.slot}, xc, true), region_empty({repeats * f.slot}, xc, true)};
    for (int h = 0; h < 2; h++)
      for (int64_t r = 0; r < repeats; r++) {
        const at::Tensor slot = leases[h].narrow(0, r * f.slot, f.slot);
        ring.slots[h].push_back(slot);
        std::vector<at::Tensor> segments;
        for (int64_t q = 0; q < f.ranks; q++) segments.push_back(slot.narrow(0, f.starts[q], f.counts[q]));
        ring.segments[h].push_back(segments);
      }
  }
  // Prepare each bound generation once. Metadata exchange agrees common
  // registration cuts before any receive is presented. Relay allocations
  // are disjoint per pair, so another destination cannot overwrite a live
  // forwarding range or any local original contribution.
  for (int h = 0; h < 2; h++) if (!ring.plans[h]) {
    auto prepared = std::make_shared<PreparedWrites>();
    Group g{true};
    for (int64_t source = 0; source < f.ranks; source++) if (f.counts[source])
      for (int64_t destination = 0; destination < f.ranks; destination++) if (source != destination) {
        const at::Tensor source_view = ring.segments[h][0][source];
        at::Tensor landing_view = source_view;
        int64_t local_stride = f.slot * (int64_t)xc.element_size();
        int forwards = 1;
        if (f.rank != source && f.rank != destination) {
          // a relay only where an installed path of nonzero weight crosses this rank (ncclMeshCommWriteForwards)
          checked(ncclMeshCommWriteForwards(communicator(f.comm), (int)source, (int)destination, &forwards));
          landing_view = at::Tensor();
          if (forwards) {
            ring.relays[h].push_back(region_empty({repeats * f.counts[source]}, xc, true));
            landing_view = ring.relays[h].back().narrow(0, 0, f.counts[source]);
          }
          local_stride = f.counts[source] * (int64_t)xc.element_size();
        }
        prepared->handles.push_back(nullptr);
        tensor_write(g, f.rank == source ? source_view : landing_view, (int)source, (int)destination, communicator(f.comm), true,
                     ncclMeshWriteLayout{0, 1, 0, 1}, &prepared->handles.back(), repeats, local_stride / (int64_t)xc.element_size(),
                     f.counts[source], xc);
      }
    prepared->prepare(f.comm, f.polls, repeats, f.slot * (int64_t)xc.element_size());
    ring.plans[h] = std::move(prepared);
  }
}

static void fused_prepare(const at::Tensor &x, const std::optional<at::Tensor> &w, int64_t dim,
                          const std::vector<int64_t> &sizes, const std::string &group, bool adds,
                          int64_t repeats, int64_t batches) {
  TORCH_CHECK(batches >= 1, "fused preparation requires a positive batch count");
  std::unique_lock<std::mutex> held(*fused_lock_);
  FusedForm &f = fused_form_of(x, w, dim, sizes, group, adds, held);
  TORCH_CHECK(f.j == 0, "fused preparation requires a complete preceding repeat train");
  const bool fresh = std::any_of(f.counts.begin(), f.counts.end(), [](int64_t n) { return n == 0; });
  auto &sequence = f.sequences[repeats];
  sequence.rings.clear();
  sequence.cursor = 0;
  sequence.stride = fresh;
  if (fresh) {
    auto &queue = f.prepared_fresh[repeats];
    for (int64_t k = (int64_t)queue.size(); k < batches; k++) {
      FusedForm::Ring ring;
      fused_allocate_ring(f, ring, x, repeats);
      queue.push_back(std::move(ring));
    }
    for (int64_t k = 0; k < batches; k++) sequence.rings.push_back(&queue[k]);
  } else {
    fused_allocate_ring(f, f.rings[repeats], x, repeats);
    sequence.rings.push_back(&f.rings[repeats]);
  }
  f.pbind.resize(f.channels * 24), f.lbind.resize(f.channels * 24);
}

// A complete recurrent chain uses the same prepared physical writes as a
// crossing. Its restored processing kernel consumes the preceding landing and
// publishes the next product in one dispatch. All buffers and offsets are
// assigned here before execution.
struct ChainStep {
  std::array<id<MTLBuffer>, 4> buffers;
  std::array<NSUInteger, 4> offsets, lengths;
  uint32_t first, last, previous_j, j, input_j;
  const std::vector<Dispatch *> *dispatches;
};
struct ChainChunk {
  PreparedWrites *plans;
  size_t first, count, input, prefetch_first, prefetch_count;
};
struct PreparedChain {
  int64_t comm;
  std::vector<at::Tensor> held;
  std::vector<ChainStep> steps;
  std::vector<ChainChunk> chunks;
  std::vector<std::unique_ptr<PreparedWrites>> inputs;
  std::vector<uint32_t> input_last;
  Dispatch dispatch, input_publication;
  std::vector<Dispatch> products;
  std::vector<Dispatch *> regular, final;
  Dispatch *publication;
  size_t publication_output, publication_j, publication_words;
  MeshStream *stream;
  uint32_t channels;
  Schedule schedule;
};
static void compile_chain(PreparedChain &chain);
static auto *prepared_chains_ = new std::vector<std::unique_ptr<PreparedChain>>;

static void release_chains(int64_t comm) {
  for (auto &chain : *prepared_chains_) if (chain && chain->comm == comm) chain.reset();
}
static int64_t chain_slot() {
  const int64_t index = prepared_chains_->size();
  prepared_chains_->push_back(nullptr);
  return index;
}
static void prepare_chain(int64_t index, int64_t kernel, int64_t input_kernel, const std::vector<int64_t> &products, bool linear,
                          const std::vector<at::Tensor> &operands,
                          const std::vector<int64_t> &constants, int64_t groups, int64_t threads,
                          const std::string &group, int64_t ranks, int64_t batch, int64_t input_frames) {
  const at::Tensor &input = operands[2], &output = operands[3], &x = operands[4];
  const int64_t steps = input.size(0);
  std::unique_lock<std::mutex> held(*fused_lock_);
  FusedForm &f = fused_form_of(x, std::nullopt, 0, std::vector<int64_t>(ranks, x.size(0)), group, true, held);
  std::vector<std::optional<at::Tensor>> buffers(operands.begin(), operands.end());
  buffers.push_back(x);  // previous landing; replaced by its compiled page below
  std::vector<int64_t> words = constants;
  words.resize(12 + 36 * f.channels);
  auto chain = std::make_unique<PreparedChain>(PreparedChain{f.comm, operands, {}, {}, {}, {},
      encoded(kernel, buffers, words, 15, groups, 1, threads),
      encoded(input_kernel, {}, std::vector<int64_t>(2 + 12 * f.channels), 15, 1, 1, MESH_SIMD_WIDTH),
      {}, {}, {}, nullptr, 6, 9, 12,
      &stream_of(at::mps::getCurrentMPSStream()), (uint32_t)f.channels});
  chain->regular = chain->final = {&chain->dispatch};
  if (linear) {
    const at::Tensor &y = operands[5];
    const int64_t R = x.size(0), K = x.size(1), S = y.size(1);
    chain->products.push_back(encoded(products[0], {x, operands[0], y, y, std::nullopt, x},
        {R, K, S, 0, K, K, K, K}, 4, linear_groups(R, S), 1, products[2] * MESH_SIMD_WIDTH));
    std::vector<int64_t> publication{R, S, K, 0, S, S, S, S, 0, 0};
    publication.resize(10 + 24 * f.channels);
    chain->products.push_back(encoded(products[1], {y, operands[1], x, operands[7], y},
        publication, 15, linear_groups(R, K), 1, products[3] * MESH_SIMD_WIDTH));
    chain->publication_output = 2;
    chain->publication_j = 8;
    chain->publication_words = 10;
  } else {
    for (auto product : products)
      chain->products.push_back(encoded(product, buffers, words, 15, groups, 1, threads));
  }
  for (auto &product : chain->products) chain->regular.push_back(&product);
  chain->publication = &chain->products.back();
  const NSUInteger input_base = input.storage_offset() * input.element_size();
  const NSUInteger output_base = output.storage_offset() * output.element_size();
  const NSUInteger input_pitch = input.stride(0) * input.element_size();
  const NSUInteger output_pitch = output.stride(0) * output.element_size();
  for (int64_t first = 0; first < steps; first += input_frames) {
    const int64_t count = std::min(input_frames, steps - first);
    const at::Tensor source = input.select(0, first);
    auto prepared = std::make_unique<PreparedWrites>();
    Group g{true};
    for (int64_t destination = 1; destination < ranks; destination++) {
      at::Tensor landing = source;
      size_t stride = input_pitch;
      if (f.rank != 0 && f.rank != destination) {
        // a relay only where an installed path of nonzero weight crosses this rank (ncclMeshCommWriteForwards)
        int forwards = 1;
        checked(ncclMeshCommWriteForwards(communicator(f.comm), 0, (int)destination, &forwards));
        landing = at::Tensor();
        if (forwards) {
          chain->held.push_back(region_empty({count * source.numel()}, input, true));
          landing = chain->held.back().narrow(0, 0, source.numel());
        }
        stride = source.nbytes();
      }
      prepared->handles.push_back(nullptr);
      TORCH_CHECK(stride % (size_t)source.element_size() == 0, "a chain's frame pitch is whole elements");
      tensor_write(g, f.rank == 0 ? source : landing, 0, (int)destination, communicator(f.comm), true, ncclMeshWriteLayout{0, 1, 0, 1},
                   &prepared->handles.back(), count, (int64_t)(stride / (size_t)source.element_size()), source.numel(), source);
    }
    prepared->prepare(f.comm, f.polls, count, input_pitch);
    chain->inputs.push_back(std::move(prepared));
    chain->input_last.push_back((uint32_t)(count - 1));
  }
  {
    at::Tensor previous = x;
    uint32_t previous_j = 0;
    auto append = [&](int64_t t, int64_t j, const at::Tensor &send, const at::Tensor &land, bool last) {
      const std::array<at::Tensor, 4> bound{input, output, send, land};
      ChainStep step{{}, {}, {}, (uint32_t)(t == 0), (uint32_t)last, previous_j, (uint32_t)j, (uint32_t)(t % input_frames),
                     last ? &chain->final : &chain->regular};
      for (size_t k = 0; k < bound.size(); k++) {
        step.buffers[k] = at::native::mps::getMTLBufferStorage(bound[k]);
        step.offsets[k] = bound[k].storage_offset() * bound[k].element_size();
        step.lengths[k] = bound[k].nbytes();
      }
      step.lengths[0] = input_pitch; step.lengths[1] = output_pitch;
      step.offsets[0] = input_base + std::min(t, steps - 1) * input_pitch;
      step.offsets[1] = output_base + std::max<int64_t>(0, t - 1) * output_pitch;
      chain->steps.push_back(step);
    };
    for (int64_t first = 0, turn = 0; first < steps; turn++) {
      const int64_t count = std::min({batch, steps - first, input_frames - first % input_frames}), half = turn % 2;
      const size_t input = first / input_frames;
      const size_t prefetch = first % input_frames == 0 && input + 1 < chain->inputs.size();
      FusedForm::Ring &ring = f.rings.at(count);
      chain->chunks.push_back({ring.plans[half].get(), (size_t)first, (size_t)count, input, input + 1, prefetch});
      for (int64_t j = 0; j < count; j++) {
        append(first + j, j, ring.segments[half][j][f.rank], previous, false);
        previous = ring.slots[half][j];
        previous_j = (uint32_t)j;
      }
      first += count;
    }
    append(steps, 0, x, previous, true);
  }
  compile_chain(*chain);
  (*prepared_chains_)[index] = std::move(chain);
}

static void compile_chain(PreparedChain &chain) {
  chain.schedule=schedule_after_inputs();
  MetalSchedule *schedule=chain.schedule.get();
  Dispatch &d = chain.dispatch;
  Dispatch &publication = *chain.publication;
  const auto &steps = chain.steps;
  const auto &chunks = chain.chunks;
  const auto binding = [&](const PreparedWrites *plans, bool publication, size_t part) {
    const uint32_t count = !plans ? 0 : publication ? plans->records.publishes : plans->records.lands;
    const auto *records = !plans ? nullptr : publication ? plans->publish.data() : plans->land.data();
    const size_t offset = (publication ? chain.publication_words : 12) + part * 12 * chain.channels;
    auto &words = publication ? chain.publication->words : d.words;
    for (uint32_t k = 0; k < chain.channels; k++) {
      const auto *record = k < count ? records + k : nullptr;
      for (int word = 0; word < 6; word++) {
        const uint64_t value = record ? (*record)[word] : 0;
        words[offset + 12 * k + 2 * word] = value & 0xffffffffu;
        words[offset + 12 * k + 2 * word + 1] = value >> 32;
      }
    }
  };
  const auto input = [&](size_t index) {
    PreparedWrites &plans = *chain.inputs[index];
    prepare_fused_writes(schedule,plans);
    Dispatch &ready = chain.input_publication;
    ready.words[0] = chain.input_last[index];
    for (uint32_t k = 0; k < chain.channels; k++) {
      const auto *record = k < plans.records.publishes ? &plans.publish[k] : nullptr;
      for (int word = 0; word < 6; word++) {
        const uint64_t value = record ? (*record)[word] : 0;
        ready.words[2 + 12 * k + 2 * word] = value & 0xffffffffu;
        ready.words[3 + 12 * k + 2 * word] = value >> 32;
      }
    }
    prepare_dispatch(schedule,ready,0);

  };
  const auto dispatch = [&](const ChainStep &step) {
    const size_t slots[4] = {2, 3, 6, 8};
    for (size_t k = 0; k < 4; k++) { d.bound[slots[k]] = step.buffers[k]; d.offsets[slots[k]] = step.offsets[k]; d.lengths[slots[k]]=step.lengths[k]; }
    d.words[0] = step.first, d.words[5] = step.last;
    d.words[8] = step.previous_j, d.words[9] = step.j;
    d.words[10] = step.input_j;
    publication.bound[chain.publication_output] = step.buffers[2];
    publication.offsets[chain.publication_output] = step.offsets[2];
    publication.lengths[chain.publication_output] = step.lengths[2];
    publication.words[chain.publication_j] = step.j;
    // Phase 0 reads the incoming pages and writes X. The products consume X
    // then Y. These declarations compile their two data edges; groups within
    // a phase need no residency assumption or cross-group spin barrier.
    for(const Dispatch *item:*step.dispatches) {
      uint64_t access=0;
      if(item==&d)access=UINT64_C(260)|(UINT64_C(24)<<MESH_ACCESS_WRITE_SHIFT);
      else if(chain.publication_words==12)access=item==chain.publication ?
        UINT64_C(162)|(UINT64_C(192)<<MESH_ACCESS_WRITE_SHIFT) : UINT64_C(17)|(UINT64_C(32)<<MESH_ACCESS_WRITE_SHIFT);
      else {
        for(size_t k=0;k<item->bound.size();k++)if(item->bound[k])access|=UINT64_C(1)<<k;
        access|=(item==chain.publication?UINT64_C(12):UINT64_C(4))<<MESH_ACCESS_WRITE_SHIFT;
      }
      prepare_dispatch(schedule,*item,access);
    }

  };
  const PreparedWrites *previous = nullptr;
  input(0);
  for (size_t turn = 0; turn < chunks.size(); turn++) {
    const ChainChunk &chunk = chunks[turn];
    for (size_t j = 0; j < chunk.prefetch_count; j++) input(chunk.prefetch_first + j);
    prepare_fused_writes(schedule,*chunk.plans);
    binding(previous, false, 0);
    binding(chunk.plans, true, 1);
    binding(chain.inputs[chunk.input].get(),false,2);
    const ChainStep *first=steps.data()+chunk.first;
    dispatch(*first);
    binding(chunk.plans,false,0);
    for(size_t j=1;j<chunk.count;j++)dispatch(first[j]);
    previous = chunk.plans;
  }
  dispatch(steps.back());
  checked(MetalScheduleFinish(schedule,nil));
}

static void chain_run(int64_t index) {
  MetalScheduleEnqueue((*prepared_chains_)[index]->schedule.get());
}
static pybind11::capsule chain_completion(int64_t index) {
  return schedule_completion((*prepared_chains_)[index]->schedule);
}

// A declared set of routed writes. Bindings name physical channels, including
// parallel cables; relay readiness stays in the compiled stream entries.

static std::pair<std::vector<std::vector<uint64_t>>, std::vector<std::vector<uint64_t>>> fused_writes_impl(
    MetalSchedule *schedule, const at::Tensor &send, const std::vector<at::Tensor> &landings,
    const std::vector<std::array<int, 2>> &pairs, int64_t comm, int64_t polls, int64_t repeats, int64_t stride,
    const std::optional<at::Tensor> &ready, int64_t ready_value, int64_t tile_bytes) {
  TORCH_CHECK(landings.size() == pairs.size(), "fused writes: one landing lease per declared write");
  TORCH_CHECK(polls >= 0 && polls <= (int64_t)UINT32_MAX, "fused writes: polls is a 32-bit count of loads");
  TORCH_CHECK(repeats > 0 && repeats <= (int64_t)UINT32_MAX && stride >= 0,
              "fused writes: positive 32-bit frame count and nonnegative byte stride required");
  const ncclComm_t c = communicator(comm);
  TORCH_CHECK(tile_bytes >= 0 && (!tile_bytes || (ready && repeats == 1)),
              "tiled writes require a contiguous publication family and its source words");
  const size_t tiles = tile_bytes && send.nbytes() ? (send.nbytes()-1)/(size_t)tile_bytes+1 : 1;
  const auto framed = [repeats, stride](const at::Tensor &tensor) {
    const size_t bytes = tensor.nbytes(), offset = tensor.storage_offset() * tensor.element_size();
    TORCH_CHECK(!stride || (size_t)(repeats - 1) <= (SIZE_MAX - bytes) / (size_t)stride,
                "fused writes: frame extent overflows address space");
    const size_t extent = bytes + (size_t)(repeats - 1) * (size_t)stride;
    TORCH_CHECK(offset <= tensor.storage().nbytes() && extent <= tensor.storage().nbytes() - offset,
                "fused writes: all frames must lie inside the tensor's backing storage");
  };
  // a meta `send` is a rank that sends none of these writes (a relay or a bystander): their shape and type alone
  if (!send.is_meta()) framed(send);
  Group g{true};
  if (!send.is_meta()) g.moving(send);
  std::optional<Operand> produced;
  if (ready) {
    TORCH_CHECK(ready->scalar_type() == at::kLong && (size_t)ready->numel() == tiles && ready->is_contiguous() && ready_value > 0,
                "source readiness requires one contiguous int64 region word per tile and a positive target");
    produced = g.moving(*ready);
  }
  std::vector<Operand> in;
  for (const at::Tensor &landing : landings) {
    if (!landing.defined()) continue;   // a rank that neither ends nor forwards the write: no landing (ScheduleBinding::writes)
    framed(landing);
    in.push_back(g.moving(landing));
    TORCH_CHECK(in.back().bytes == send.nbytes(), "fused writes: a landing holds the source's bytes (a WRITE moves bytes, not typed values)");
  }
  PreparedWrites plans;
  plans.prepare(comm, polls, repeats, stride);
  const size_t binding_words=sizeof(ncclMeshBinding)/sizeof(uint64_t),record_words=1+plans.land.size()*binding_words;
  std::vector<uint64_t> tile_landings(tile_bytes ? tiles*record_words : 0);
  int rank=0;checked(ncclCommUserRank(c,&rank));
  plans.handles.resize(pairs.size(), nullptr);
  for (size_t i = 0; i < pairs.size(); i++) {
    if (tile_bytes) {
      // the rank's own view: the source's values on the source, its landing on every other rank
      tensor_write(g, rank == pairs[i][0] ? send : landings[i], pairs[i][0], pairs[i][1], c, true,
                   ncclMeshWriteLayout{(size_t)tile_bytes, 1, 0, 1}, &plans.handles[i], 1, 0, send.numel(), send);
      if(rank==pairs[i][1])checked(ncclMeshWriteLandings(plans.handles[i],tile_landings.data(),(uint32_t)plans.land.size(),
          ncclMeshTileLayout{.columns=tiles,.tile_stride=record_words*sizeof(uint64_t),.instances=1}));
    } else {
      TORCH_CHECK(stride % send.element_size() == 0, "fused writes: a frame stride is whole elements");
      tensor_write(g, rank == pairs[i][0] ? send : landings[i], pairs[i][0], pairs[i][1], c, true, ncclMeshWriteLayout{0, 1, 0, 1},
                   &plans.handles[i], repeats, stride / send.element_size(), send.numel(), send);
    }
    if (produced) checked(ncclMeshWriteSourceReady(plans.handles[i], produced->at, (uint64_t)ready_value,
        ncclMeshTileLayout{.columns=tiles,.tile_stride=sizeof(uint64_t),.instances=1}));
  }
  prepare_fused_writes(schedule,plans);
  std::vector<std::vector<uint64_t>> published, landed;
  for (uint32_t k = 0; k < plans.records.publishes; k++) published.emplace_back(plans.publish[k].begin(), plans.publish[k].end());
  if(tile_bytes) {
    for(size_t t=0;t<tiles;t++)for(size_t k=0;k<plans.land.size();k++) {
      const uint64_t *record=tile_landings.data()+t*record_words;
      if(k<record[0])landed.emplace_back(record+1+k*binding_words,record+1+(k+1)*binding_words);
      else landed.emplace_back(binding_words,0);
    }
  } else for (uint32_t k = 0; k < plans.records.lands; k++) landed.emplace_back(plans.land[k].begin(), plans.land[k].end());
  return {published, landed};
}

// A Python binding of the shared native schedule. All methods except enqueue
// receive configuration; enqueue is the native header's submission operation.
struct ScheduleBinding {
  Schedule schedule;
  ScheduleBinding():schedule(schedule_after_inputs()){}
  std::vector<at::Tensor> operands;
  // Preparation supplies native run/context/retain/release operands. The
  // completion worker neither owns a Python callback nor acquires the GIL.
  // `declaration` (kind, bytes, places as (offset, relocation)): what the engine makes the operation from again
  // (design/replay-templates-2026-10-29.md); None: the function makes no template.
  using HostDeclaration=std::tuple<std::string,pybind11::bytes,std::vector<std::pair<size_t,int>>>;
  void host(const std::array<uintptr_t,4> &entry,const std::vector<pybind11::object> &inputs,
      const std::vector<pybind11::object> &outputs,const std::optional<HostDeclaration> &declaration) { @autoreleasepool {
    std::vector<MetalReach> views;
    std::vector<at::Tensor> held;
    auto declare=[&](const std::vector<pybind11::object> &tensors,uint32_t access) {
      for(const auto &object:tensors) {
        const TensorOperand operand(object);const auto &t=operand.storage;held.push_back(t);
        NSData *data=[operand.layout.value reaches:operand.pages.buffer base:0 elementBytes:t.element_size() access:access binding:-1];
        const auto *spans=(const MetalReach *)data.bytes;
        views.insert(views.end(),spans,spans+data.length/sizeof *spans);
      }
    };
    declare(inputs,MetalReachRead);declare(outputs,MetalReachWritten);
    std::string kind,bytes;std::vector<MetalPlace> places;
    if(declaration) {
      kind=std::get<0>(*declaration);bytes=std::get<1>(*declaration);
      for(const auto &[at,relocation]:std::get<2>(*declaration))places.push_back({at,(MetalRelocation)relocation});
    }
    const MetalDeclaration declared{kind.c_str(),bytes.data(),bytes.size(),places.data(),places.size()};
    MetalScheduleHost(schedule.get(),views.data(),views.size(),declaration?&declared:nullptr,host_operation(entry,false,held));
  }}
  void compute(int64_t kernel,const Tensors &buffers,const std::vector<int64_t> &words,
      int64_t at,int64_t gx,int64_t gy,int64_t threads,uint64_t access) {
    for(const auto &buffer:buffers)if(buffer)operands.push_back(*buffer);
    if(!gx||!gy)return;
    prepare_dispatch(schedule.get(),encoded(kernel,buffers,words,at,gx,gy,threads),access);
  }
  // A None landing is the library's relay: a rank that ends neither side of its pair, one frame. A relay shaped like
  // `send` (a meta tensor where this rank sends nothing) is leased only where an installed path of nonzero weight
  // crosses this rank (ncclMeshCommWriteForwards), and held with the schedule; elsewhere none (tensor_write's empty table).
  auto writes(const at::Tensor &send,const std::vector<std::optional<at::Tensor>> &given,
      const std::vector<std::array<int,2>> &pairs,int64_t comm,int64_t polls,int64_t repeats,int64_t stride,
      const std::optional<at::Tensor> &ready,int64_t ready_value,int64_t tile_bytes) {
    TORCH_CHECK(given.size()==pairs.size(),"fused writes: one landing lease per declared write");
    int rank=0;checked(ncclCommUserRank(communicator(comm),&rank));
    std::vector<at::Tensor> landings(given.size());
    for(size_t i=0;i<given.size();i++) {
      if(given[i]){landings[i]=*given[i];continue;}
      TORCH_CHECK(repeats==1&&rank!=pairs[i][0]&&rank!=pairs[i][1],"fused writes: only a relay's landing, of one frame, may be the library's");
      int forwards=1;checked(ncclMeshCommWriteForwards(communicator(comm),pairs[i][0],pairs[i][1],&forwards));
      if(forwards)landings[i]=region_empty(send.sizes().vec(),send,true);
    }
    if(!send.is_meta())operands.push_back(send);
    for(const auto &landing:landings)if(landing.defined())operands.push_back(landing);
    if(ready)operands.push_back(*ready);
    return fused_writes_impl(schedule.get(),send,landings,pairs,comm,polls,repeats,stride,ready,ready_value,tile_bytes);
  }

  void finish() { const auto held=operands; checked(MetalScheduleFinish(schedule.get(),^(NSError *){(void)held;})); }
  void enqueue() { MetalScheduleEnqueue(schedule.get()); }
  pybind11::capsule completion() { return schedule_completion(schedule); }

  // design/replay-templates-2026-10-29.md: each bound tensor's storage, as the recorder names it, and its view's byte
  // offset in that storage, so a template binds any view of the same shape
  static NSArray *storages(const std::vector<at::Tensor> &bound,NSArray **offsets) {
    NSMutableArray *buffers=[NSMutableArray array],*at=[NSMutableArray array];
    for(const auto &t:bound) {
      [at addObject:@((NSUInteger)(t.storage_offset()*t.element_size()))];
      if(t.is_mps()){[buffers addObject:at::native::mps::getMTLBufferStorage(t)];continue;}
      MetalTensorPages *pages=tensorPages(t);[buffers addObject:pages.buffer];[pages release];
    }
    *offsets=at;
    return buffers;
  }
  // The function as a template in `directory`: `bound` (its inputs, outputs and the weights a later instance
  // replaces) its operands in order, every other storage it names kept by value.
  void save(const std::string &directory,const std::vector<at::Tensor> &bound) {
    NSError *error=nil;NSArray *offsets=nil;
    NSArray *buffers=storages(bound,&offsets);
    NSDictionary *made=MetalScheduleTemplate(schedule.get(),buffers,offsets,&error);
    TORCH_CHECK(made,"template: ",error.localizedDescription.UTF8String);
    error=MetalTemplateWrite(made,@(directory.c_str()));
    TORCH_CHECK(!error,"template ",directory,": ",error.localizedDescription.UTF8String);
  }
  // The template in `directory` made on a new function after torch's inputs, `bound` its operands in order.
  // `communicators` the instance's, in the order the template's crossings first named theirs; `host(kind, bytes)` the
  // native entry (run, context, retain, release) a host operation's declaration makes, its context the caller's to pass.
  static std::unique_ptr<ScheduleBinding> load(const std::string &directory,const std::vector<at::Tensor> &bound,
      const std::vector<int64_t> &communicators,const std::optional<pybind11::function> &host) {
    auto made=std::make_unique<ScheduleBinding>();
    NSError *error=nil;
    NSDictionary *read=MetalTemplateRead(at::mps::MPSDevice::getInstance()->device(),@(directory.c_str()),&error);
    TORCH_CHECK(read,"template ",directory,": ",error.localizedDescription.UTF8String);
    NSMutableArray *comms=[NSMutableArray array];
    for(auto comm:communicators)[comms addObject:[NSValue valueWithPointer:(void *)communicator(comm)]];
    const std::vector<at::Tensor> held=bound;
    MetalTemplateHost made_host=nil;
    if(host)made_host=^MetalHostOperation(NSString *kind,NSData *bytes,NSError **failed){
      try {
        auto entry=(*host)(std::string(kind.UTF8String),pybind11::bytes((const char *)bytes.bytes,bytes.length)).cast<std::array<uintptr_t,4>>();
        return host_operation(entry,true,held);
      } catch(const std::exception &e) {
        if(failed)*failed=[NSError errorWithDomain:@"mesh.template" code:1 userInfo:@{NSLocalizedDescriptionKey:@(e.what())}];
        return (MetalHostOperation)nil;
      }
    };
    MetalTemplateCross made_cross=^MetalCrossing(NSString *kind,NSData *bytes,MetalSchedule *schedule,NSError **failed){
      if(![kind isEqualToString:@"collectives"]) {
        if(failed)*failed=[NSError errorWithDomain:@"mesh.template" code:1 userInfo:@{NSLocalizedDescriptionKey:
          [NSString stringWithFormat:@"no crossing of kind %@",kind]}];
        return (MetalCrossing)nil;
      }
      std::shared_ptr<DeclaredCalls> calls;
      try { calls=declared_calls(bytes); } catch(const std::exception &e) {
        if(failed)*failed=[NSError errorWithDomain:@"mesh.template" code:1 userInfo:@{NSLocalizedDescriptionKey:@(e.what())}];
        return (MetalCrossing)nil;
      }
      MetalScheduleCommunicator(schedule,calls->comm);
      MetalCrossing crossing=^int(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,id<MTLResidencySet> residency,
          NSUInteger,NSUInteger) {
        ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
          .residencySet=(__bridge void *)residency};
        for(size_t r=0;r<calls->repeat;r++) {
          const int result=(int)encoded(calls->issues,calls->comm,&on);
          if(result)return result;
        }
        return 0;
      };
      return [[crossing copy] autorelease];
    };
    NSArray *offsets=nil;NSArray *buffers=storages(bound,&offsets);
    error=MetalScheduleInstantiate(made->schedule.get(),read,buffers,offsets,comms,made_host,made_cross);
    TORCH_CHECK(!error,"template ",directory,": ",error.localizedDescription.UTF8String);
    // a device of an architecture the template has no pipelines for compiles them from its libraries, and keeps them
    if((error=MetalTemplateArchive(read,@(directory.c_str()))))
      TORCH_WARN("template ",directory,": this architecture's pipelines were not kept: ",error.localizedDescription.UTF8String);
    made->operands=bound;
    return made;
  }
};

// Invocation slots and results are operands. Runtime selects the next prepared
// native function; it does not enter the framework's encoding queue.
struct Executions {
  std::vector<Schedule> schedules;
  pybind11::object output;
  size_t next=0;
  explicit Executions(pybind11::object value):output(std::move(value)){}
  void add(ScheduleBinding &binding){schedules.push_back(binding.schedule);}
  pybind11::object run(){MetalScheduleEnqueue(schedules[next++].get());return output;}
  pybind11::capsule completion(){return schedule_completion(schedules[next-1]);}
};

// Compact row geometry and explicit indexed selections lower to one prepared
// page WRITE. Layout resolution ends before native commands are recorded.
// a view as a WRITE names it (operand.py TensorView.write_layout): its storage, element bits, shape, strides and offset in
// elements, whether it is mapped and its runs; then source and destination ranks
using ViewBinding = std::tuple<at::Tensor, int64_t, std::vector<int64_t>, std::vector<int64_t>, int64_t, bool, std::vector<int64_t>, int, int>;
struct ViewWrites {
  int64_t comm;
  bool metal;
  std::vector<ViewBinding> bindings;
  PreparedWrites plans;
  std::vector<at::Tensor> leases;
  // A meta storage is a relay's: its shape and type alone, a lease taken at preparation only where an installed path of
  // nonzero weight crosses this rank (ncclMeshCommWriteForwards), else an empty page table.
  ViewWrites(const std::vector<ViewBinding> &given, int64_t handle) : comm(handle), metal(true), bindings(given) {
    TORCH_CHECK(!bindings.empty(), "view writes require at least one declared delivery");
    for(const auto &binding:bindings)if(!std::get<0>(binding).is_meta()){metal=std::get<0>(binding).is_mps();break;}
    for(const auto &binding:bindings)if(!std::get<0>(binding).is_meta())leases.push_back(std::get<0>(binding));
  }
  void prepare() {
    if(!plans.handles.empty())return;
    Group g{metal};
    for (const auto &[given_storage, bits, shape, strides, offset, mapped, runs, source, destination] : bindings) {
      at::Tensor storage = given_storage;
      ncclMeshPageTable_t table = nullptr;
      if (storage.is_meta()) {
        int forwards = 1;
        checked(ncclMeshCommWriteForwards(communicator(comm), source, destination, &forwards));
        if (forwards) {
          int64_t extent = offset + 1;
          for (size_t d = 0; d < shape.size(); d++) extent += (shape[d] - 1) * (mapped ? 0 : strides[d]);
          storage = region_empty({std::max<int64_t>(extent, 1)}, given_storage, metal);
          leases.push_back(storage);
        }
      }
      TORCH_CHECK(!storage.defined() || storage.is_meta() || storage.is_mps() == metal, "view writes require one local device");
      if (storage.is_meta()) checked(ncclMeshPageTableCreate(nullptr, communicator(comm), metal, &table));
      else checked(ncclMeshPageTableCreate(g.where(storage), communicator(comm), metal, &table));
      const std::vector<uint64_t> dims(shape.begin(), shape.end()), elements(runs.begin(), runs.end());
      const ncclMeshLayout layout{table, (uint32_t)bits, (uint32_t)dims.size(), dims.data(), strides.data(), (uint64_t)offset,
                                  mapped ? elements.data() : nullptr, elements.size() / 2};
      plans.handles.push_back(nullptr);
      const ncclResult_t made = ncclMeshWriteLayoutPrepare(&layout, source, destination, 0, ncclMeshWriteLayout{0, 1, 0, 1},
                                                           &plans.handles.back());
      ncclMeshPageTableDestroy(table);
      checked(made);
    }
  }
  // Each encoded execution is the plan's next invocation, cycling the plan's instances: NFE k+1 writes other pages
  // than NFE k sends from only where the plan declares more than one (design/pages-and-functions.md, Freeing 7).
  size_t invocations = 0;
  ncclResult_t encode(ncclMeshStream *stream, size_t invocation) {
    ncclResult_t result = ncclGroupStart();
    for (auto plan : plans.handles) if (!result) result = ncclMeshWriteExecute(plan, invocation, stream);
    const ncclResult_t ended = ncclGroupEnd();
    return result ? result : ended;
  }
  std::vector<Schedule> executions;
  size_t next=0;
  void append(MetalSchedule *schedule) {
    TORCH_CHECK(metal,"prepared distributed writes require the Metal backend");
    MetalScheduleCommunicator(schedule,communicator(comm));
    prepare();
    ViewWrites *self=this;
    const size_t invocation=invocations++;
    checked(MetalScheduleCross(schedule,nullptr,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
        id<MTLResidencySet> residency,NSUInteger,NSUInteger) {
      ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
        .residencySet=(__bridge void *)residency};
      return (int)self->encode(&on,invocation);
    }));
  }
  void realize() {
    auto made=schedule_after_inputs();append(made.get());
    const auto held=leases;
    checked(MetalScheduleFinish(made.get(),^(NSError *){(void)held;}));executions.push_back(std::move(made));
  }
  void enqueue() { MetalScheduleEnqueue(executions[next++].get()); }
  pybind11::capsule completion(){return schedule_completion(executions[next-1]);}

};

// design/whole-function.md#observations
static pybind11::capsule completion() {
  auto stream = at::mps::getCurrentMPSStream();
  __block id<MTLCommandBuffer> buffer = nil;
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
    buffer = [stream->commandBuffer().rootCommandBuffer retain];
  });
  return pybind11::capsule((void *)buffer, "mesh.command-buffer", [](PyObject *capsule) {
    [(id<MTLCommandBuffer>)PyCapsule_GetPointer(capsule, "mesh.command-buffer") release];
  });
}

static bool completed(pybind11::object observation) {
  if(pybind11::isinstance<pybind11::tuple>(observation)) {
    bool done=true;
    for(auto item:observation.cast<pybind11::tuple>())done=completed(pybind11::reinterpret_borrow<pybind11::object>(item))&&done;
    return done;
  }
  auto token=observation.cast<pybind11::capsule>();
  if(!strcmp(token.name(),"mesh.prepared-function")) {
    NSError *error=nil;
    int status=MetalScheduleStatus(static_cast<Schedule *>(token.get_pointer())->get(),&error);
    TORCH_CHECK(status>=0,"the prepared function failed: ",error.localizedDescription.UTF8String);return status==1;
  }
  id<MTLCommandBuffer> buffer = (id<MTLCommandBuffer>)token.get_pointer();
  if (buffer.status == MTLCommandBufferStatusError)
    throw std::runtime_error(buffer.error.localizedDescription.UTF8String);
  return buffer.status == MTLCommandBufferStatusCompleted;
}

// The module: prepared functions, their collectives (Collectives, ViewWrites) and fused crossings; no collective is
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
  install_region_allocator();
  module.def("place_layout",[](NSUInteger count,const std::vector<TensorLayout> &uses,
      const std::vector<TensorLayout> &targets,const std::vector<TensorLayout> &values,const std::vector<NSInteger> &widths)->pybind11::object {
    auto layouts=[](const std::vector<TensorLayout> &items) {
      NSMutableArray *result=[NSMutableArray array];for(const auto &item:items)[result addObject:item.value];return result;
    };
    @try {
      MetalTensorLayout *placed=MetalTensorPlace(count,layouts(uses),layouts(targets),layouts(values),tensorNumbers(widths));
      return placed?pybind11::cast(TensorLayout(placed)):pybind11::none();
    } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
  }, pybind11::call_guard<Drained>());
  module.def("import_attributes",&import_attributes, pybind11::call_guard<Drained>());
  module.def("completion", &completion, pybind11::call_guard<Drained>());
  module.def("completed", &completed, pybind11::call_guard<Drained>());
  module.def("pipeline", &pipeline, pybind11::call_guard<Drained>());
  module.def("pipeline_specialized", &pipeline_specialized, pybind11::call_guard<Drained>());
  module.def("index_constant_prelude", [](){return std::string(MetalViewIndexConstantPrelude().UTF8String);}, pybind11::call_guard<Drained>());
  pybind11::class_<TensorTraversal>(module,"TensorTraversal")
    .def(pybind11::init<const std::vector<TensorLayout> &,const std::vector<std::pair<NSUInteger,NSUInteger>> &>(), pybind11::call_guard<Drained>())
    .def_property_readonly("cuts",[](const TensorTraversal &v){return tensorNumbers(v.value.cuts);})
    .def("fragment",[](const TensorTraversal &v,NSUInteger first,NSUInteger end){
      @try { auto f=[v.value fragment:first end:end];return std::make_tuple(f.part,f.offset,f.count); }
      @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
    }, pybind11::call_guard<Drained>());
  pybind11::class_<TensorLayout>(module,"TensorLayout")
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
    .def("equivalent",[](const TensorLayout &v,const TensorLayout &other){return bool([v.value equivalent:other.value]);}, pybind11::call_guard<Drained>())
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
    .def("requirements",[](const TensorLayout &v,const std::vector<std::pair<std::string,std::vector<NSInteger>>> &requirements){
      NSMutableArray *values=[NSMutableArray array];
      for(const auto &[kind,steps]:requirements)[values addObject:@{@"kind":@(kind.c_str()),@"strides":tensorNumbers(steps)}];
      @try { return TensorLayout([v.value requirements:values]); }
      @catch(NSException *error){TORCH_CHECK(false,error.reason.UTF8String);}
    }, pybind11::call_guard<Drained>())
    .def("declaration",[](const TensorLayout &v,NSUInteger binding){return std::string([v.value declaration:binding].UTF8String);}, pybind11::call_guard<Drained>())
    .def("constants",[](const TensorLayout &v,NSUInteger binding)->pybind11::object{
      NSArray<NSArray<NSNumber *> *> *values=[v.value constants:binding];
      if(!values)return pybind11::none();
      pybind11::list out;
      for(NSArray<NSNumber *> *c in values)out.append(pybind11::make_tuple(c[0].longLongValue,c[1].longLongValue,c[2].longLongValue));
      return out;}, pybind11::call_guard<Drained>())
    .def("storage_declaration",[](const TensorLayout &v,NSUInteger binding,const TensorLayout &source){
      return std::string([v.value declaration:binding fromStorage:source.value].UTF8String);}, pybind11::call_guard<Drained>())
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
  module.def("same_operand",[](const pybind11::object &left,const pybind11::object &right){
    const TensorOperand a(left),b(right);
    return a.storage.scalar_type()==b.storage.scalar_type()&&a.pages.buffer==b.pages.buffer&&[a.layout.value isEqual:b.layout.value];
  }, pybind11::call_guard<Drained>());
  module.def("view_prelude", [](){return std::string(MetalViewIndexPrelude().UTF8String);}, pybind11::call_guard<Drained>());
  auto viewSelection=[](NSDictionary *result) {
    auto values=[](NSData *data) {
      std::vector<int32_t> words(data.length/sizeof(int32_t));
      if(data.length)memcpy(words.data(),data.bytes,data.length);
      return words;
    };
    return std::make_pair(values(result[@"blocks"]),values(result[@"order"]));
  };
  module.def("view_block_slice", [viewSelection](const std::vector<int32_t> &order,NSInteger first,NSInteger end,NSInteger step,NSUInteger block) {
    return viewSelection(MetalViewBlockSlice(order.data(),order.size(),first,end,step,block));
  }, pybind11::call_guard<Drained>());
  module.def("view_block_gather", [viewSelection](const std::vector<int32_t> &order,const std::vector<int32_t> &selected,NSUInteger block) {
    return viewSelection(MetalViewBlockGather(order.data(),order.size(),selected.data(),selected.size(),block));
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
  compilation["LINEAR_PRODUCTS"] = LINEAR_PRODUCTS;
  module.attr("compilation") = compilation;
  module.def("linear_source", &linear_source, pybind11::call_guard<Drained>());
  module.def("linear_groups", &linear_groups, pybind11::call_guard<Drained>());
  module.def("linear_commands", &linear_commands, pybind11::call_guard<Drained>());
  pybind11::class_<Commands>(module,"Commands").def(pybind11::init<const std::vector<CommandRecord> &>());
  module.def("linear_split", &linear_split, pybind11::call_guard<Drained>());
  module.def("device_facts", &device_facts, pybind11::call_guard<Drained>());
  module.def("pipeline_facts", &pipeline_facts, pybind11::call_guard<Drained>());
  module.attr("polls") = (int64_t)NCCL_MESH_POLLS;
  module.def("encode", &encode, pybind11::arg("handle"), pybind11::arg("buffers"), pybind11::arg("constants"), pybind11::arg("at"),
             pybind11::arg("gx"), pybind11::arg("gy"), pybind11::arg("threads"), pybind11::call_guard<Drained>());
  module.def("attach", &attach, pybind11::call_guard<Drained>());
  module.def("commit", &commit, pybind11::call_guard<Drained>());
  module.def("publish_boundary", &publish_boundary, pybind11::call_guard<Drained>());
  module.def("region_empty", &region_empty, pybind11::call_guard<Drained>());
  module.def("address", &address, pybind11::call_guard<Drained>());
  module.def("records", &region_allocations, pybind11::call_guard<Drained>());   // the region tensors' leases live now: each its address and bytes
  module.def("metal_prelude", &metal_prelude, pybind11::call_guard<Drained>());
  module.def("fused_maker", &fused_maker, pybind11::call_guard<Drained>());
  module.def("fused_release", &fused_release, pybind11::call_guard<Drained>());
  pybind11::class_<ScheduleBinding>(module,"Schedule")
    .def(pybind11::init<>(), pybind11::call_guard<Drained>())
    .def("save",&ScheduleBinding::save, pybind11::call_guard<Drained>())
    .def_static("load",&ScheduleBinding::load, pybind11::arg("directory"), pybind11::arg("bound"),
                pybind11::arg("communicators")=std::vector<int64_t>{}, pybind11::arg("host")=pybind11::none(),
                pybind11::call_guard<Drained>())
    .def("append",[](ScheduleBinding &s,Commands &commands){commands.append(s.schedule.get());
        s.operands.insert(s.operands.end(),commands.operands.begin(),commands.operands.end());}, pybind11::call_guard<Drained>())
    .def("compute",&ScheduleBinding::compute, pybind11::call_guard<Drained>())
    .def("writes",&ScheduleBinding::writes, pybind11::call_guard<Drained>())
    .def("host",&ScheduleBinding::host, pybind11::arg("entry"), pybind11::arg("inputs"), pybind11::arg("outputs"),
         pybind11::arg("declaration")=pybind11::none(), pybind11::call_guard<Drained>())
    .def("finish",&ScheduleBinding::finish, pybind11::call_guard<Drained>())
    .def("enqueue",&ScheduleBinding::enqueue, pybind11::call_guard<Drained>())
    .def("completion",&ScheduleBinding::completion, pybind11::call_guard<Drained>());
  pybind11::class_<ViewWrites, std::shared_ptr<ViewWrites>>(module, "ViewWrites")
    .def(pybind11::init<const std::vector<ViewBinding> &, int64_t>(), pybind11::call_guard<Drained>())
    .def("realize", &ViewWrites::realize, pybind11::call_guard<Drained>())
    .def("append",[](ViewWrites &w,ScheduleBinding &s){w.append(s.schedule.get());
        s.operands.insert(s.operands.end(),w.leases.begin(),w.leases.end());}, pybind11::call_guard<Drained>())
    .def("completion", &ViewWrites::completion, pybind11::call_guard<Drained>())
    .def("enqueue", &ViewWrites::enqueue, pybind11::call_guard<Drained>());
  pybind11::class_<Collectives, std::shared_ptr<Collectives>>(module, "Collectives")
    .def(pybind11::init<const std::vector<Call> &, int64_t, std::vector<at::Tensor>,
                        const std::vector<CommandRecord> &, const std::vector<CommandRecord> &>(),
         pybind11::arg("calls"),pybind11::arg("comm"),pybind11::arg("operands"),
         pybind11::arg("inputs")=std::vector<CommandRecord>{},pybind11::arg("outputs")=std::vector<CommandRecord>{}, pybind11::call_guard<Drained>())
    .def("realize", &Collectives::realize, pybind11::arg("repetitions")=1, pybind11::call_guard<Drained>())
    .def("append",[](Collectives &c,ScheduleBinding &s){c.append(s.schedule.get());
        s.operands.insert(s.operands.end(),c.operands.begin(),c.operands.end());}, pybind11::call_guard<Drained>())
    .def("completion", &Collectives::completion, pybind11::call_guard<Drained>())
    .def("enqueue", &Collectives::enqueue, pybind11::call_guard<Drained>());
  pybind11::class_<Executions>(module,"Executions")
    .def(pybind11::init<pybind11::object>(), pybind11::call_guard<Drained>())
    .def("add",&Executions::add, pybind11::call_guard<Drained>())
    .def("run",&Executions::run, pybind11::call_guard<Drained>())
    .def("completion",&Executions::completion, pybind11::call_guard<Drained>());
  module.def("fused_reconfigure", &fused_reconfigure, pybind11::call_guard<Drained>());
  module.def("write_entries", &write_entries, pybind11::call_guard<Drained>());
  module.attr("STREAM_ENTRIES") = (int64_t)MESH_STREAM_ENTRIES;
  module.def("chain_slot", &chain_slot, pybind11::call_guard<Drained>());
  module.def("prepare_chain", &prepare_chain, pybind11::call_guard<Drained>());
  module.def("chain_run", &chain_run, pybind11::call_guard<Drained>());
  module.def("chain_completion", &chain_completion, pybind11::call_guard<Drained>());
  module.def("fused_prepare", &fused_prepare, pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("dim"),
             pybind11::arg("sizes"), pybind11::arg("group"), pybind11::arg("adds"),
             pybind11::arg("repeats") = 1, pybind11::arg("batches") = 1, pybind11::call_guard<Drained>());
}
