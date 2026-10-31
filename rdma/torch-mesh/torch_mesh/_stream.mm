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
  // notifying `progress` of the next count, which a schedule after it awaits (schedule_after_inputs).  Notifications
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
      return {where(t), (size_t)t.numel() * 2, kind(at::toRealValueType(t.scalar_type())), (size_t)t.nbytes()};
    }
    TORCH_CHECK(kind(t.scalar_type()) != ncclNumTypes, "the mesh backend reduces no ", t.scalar_type());
    return {where(t), (size_t)t.numel(), kind(t.scalar_type()), (size_t)t.nbytes()};
  }
};

// A call's prepared operands and counts, made before its group is encoded.
struct Issue {
  int64_t kind, op, root;
  std::vector<Operand> sends, receives;
  std::vector<size_t> counts;
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
    Operand &operand=i.sends.front();
    const size_t width=operand.count?operand.bytes/operand.count:0;
    operand.count=(size_t)counts[0]*per;operand.bytes=operand.count*width;
  }
  return i;
}

// Local byte domains of the prepared collective. Role selection is compiler
// input; no runtime call re-derives a rank's reads or writes.
static std::vector<MetalReach> collective_views(const std::vector<Issue> &issues,ncclComm_t comm) {
  int rank=0;checked(ncclCommUserRank(comm,&rank));
  std::vector<MetalReach> views;
  auto add=[&](const Operand &operand,size_t bytes,uint32_t access) {
    if(!operand.at||!bytes)return;
    const auto &binding=*static_cast<const ncclMeshBuffer *>(operand.at);
    views.push_back({(__bridge id<MTLBuffer>)binding.buffer,binding.offset,bytes,access,-1});
  };
  for(const Issue &issue:issues) {
    if(issue.kind==ALLREDUCE||issue.kind==REDUCE||issue.kind==BROADCAST) {
      const Operand &source=issue.sends.front();
      if(issue.kind!=BROADCAST||rank==issue.root)add(source,source.bytes,MetalReachRead);
      if(issue.kind!=REDUCE||rank==issue.root)
        add(issue.receives.empty()?source:issue.receives.front(),source.bytes,MetalReachWritten);
      continue;
    }
    const size_t peers=issue.kind==ALLTOALL?issue.counts.size()/2:issue.counts.size();
    for(size_t side=0;side<2;side++) {
      const auto &operands=side?issue.receives:issue.sends;
      const size_t start=issue.kind==ALLTOALL?side*peers:0;
      for(size_t k=0;k<operands.size();k++) {
        const Operand &operand=operands[k];size_t count=0;
        if((issue.kind==ALLGATHER&&!side)||(issue.kind==REDUCE_SCATTER&&side))count=issue.counts[rank];
        else if(operands.size()==1)count=std::accumulate(issue.counts.begin()+start,issue.counts.begin()+start+peers,size_t(0));
        else count=issue.counts[start+k];
        add(operand,operand.count?count*(operand.bytes/operand.count):0,side?MetalReachWritten:MetalReachRead);
      }
    }
  }
  return views;
}

// Calls, typed views, and progress bindings are serialized once by the owning
// communicator plan. A crossing retains only its position and publication mode.
struct DeclaredGroup {
  ncclComm_t comm;
  uint64_t repeat, invocation, frames, stride;
  uint32_t polls, fused;
};
static void declare_group(const DeclaredGroup &head,std::vector<uint8_t> &bytes,std::vector<MetalPlace> &places) {
  places.push_back({offsetof(DeclaredGroup,comm),MetalRelocationCommunicator});
  bytes.insert(bytes.end(),(const uint8_t *)&head,(const uint8_t *)(&head+1));
}
static DeclaredGroup declared_group(NSData *bytes) {
  TORCH_CHECK(bytes.length==sizeof(DeclaredGroup),"template: invalid compiled group declaration");
  DeclaredGroup result;memcpy(&result,bytes.bytes,sizeof result);return result;
}

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
using CommandReach = std::tuple<uint32_t,uint32_t,pybind11::object>;
using CommandReaches = std::vector<std::optional<std::vector<CommandReach>>>;
static std::vector<pybind11::object> commandOperands(const std::vector<std::optional<at::Tensor>> &tensors) {
  std::vector<pybind11::object> result;for(const auto &t:tensors)result.push_back(t?pybind11::cast(*t):pybind11::none());return result;
}
// A record (words, constants) is refused short of its declared type; a buffer, holding part of one declared element: an
// array of none is a ragged share of nothing (a rank given no columns).
static void fits_binding(id<MTLComputePipelineState> state, NSUInteger index, NSUInteger bound, bool record) {
  const NSUInteger declared = MetalPipelineDeclaredBytes(state, index);
  if (declared <= bound || (!record && !bound)) return;
  NSString *name = MetalPipelineName(state);
  TORCH_CHECK(false, "kernel ", name ? name.UTF8String : "(unnamed)", " declares ", declared, " bytes at buffer ", index,
              ", and its dispatch binds ", bound, " bytes of ", record ? "its record" : "a buffer");
}
struct Commands {
  MetalReplay *replay=nullptr;
  std::vector<at::Tensor> operands;
  explicit Commands(const std::vector<CommandRecord> &records,const std::optional<CommandReaches> &declared=std::nullopt) {
    TORCH_CHECK(!declared||declared->size()==records.size(),"each command has one declared access list");
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
        NSData *data=[operand.layout.value reaches:operand.pages.buffer base:0 elementBytes:t.element_size() access:MetalReachBinding|mode binding:(int32_t)slot];
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
    replay=MetalReplayCommands(device,descriptions.data(),descriptions.size(),MetalFormEncoded);
    TORCH_CHECK(replay,"native command preparation failed");
  }
  ~Commands(){MetalReplayFree(replay);}
  void append(MetalSchedule *schedule) {
    if(!replay)return;
    NSError *error=MetalScheduleAppend(schedule,replay,NSMakeRange(0,1),@[],nil,nil);
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
    const ncclComm_t handle=communicator(comm);
    MetalScheduleCommunicator(schedule,handle);
    const size_t batches=before.replay||after.replay?repetitions:(size_t)(repetitions!=0);
    const size_t repeat=batches==1?repetitions:1;
    std::vector<uint8_t> declared;std::vector<MetalPlace> places;
    declare_group({handle,issues.empty()?0:repeat,0,0,0,0,0},declared,places);
    const MetalDeclaration declaration{"collectives",declared.data(),declared.size(),places.data(),places.size()};
    const auto views=collective_views(issues,handle);
    for(size_t k=0;k<batches;k++) {
      before.append(schedule);
      const ncclMeshGroup publication{handle,0,issues.empty()?0:repeat};
      checked(MetalScheduleCross(schedule,views.data(),views.size(),&publication,&declaration,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
        id<MTLResidencySet> residency,NSUInteger,NSUInteger) {
      ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
        .residencySet=(__bridge void *)residency};
      for(size_t r=0;r<repeat;r++) {
        const int result=(int)ncclMeshCommEncode(handle,&on);
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
static std::pair<std::vector<at::Tensor>,std::vector<std::pair<at::Tensor,at::Tensor>>> import_attributes(const std::vector<at::Tensor> &values) {
  struct Import {
    std::vector<size_t> members;
    std::vector<std::pair<size_t,size_t>> spans;
    size_t low=std::numeric_limits<size_t>::max(),high=0,alignment=1;
  };
  std::map<const c10::StorageImpl *,Import> groups;
  std::vector<at::Tensor> result=values;
  std::vector<std::pair<at::Tensor,at::Tensor>> copies;
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
  }
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
    const auto copy=[&](size_t a,size_t b){if(b>a) {
      auto input=from.narrow(0,(int64_t)a,(int64_t)(b-a));
      auto output=storage.narrow(0,(int64_t)(a-low),(int64_t)(b-a));
      if(source.is_cpu())memcpy(output.data_ptr(),input.data_ptr(),b-a);
      else copies.emplace_back(std::move(input),std::move(output));
    }};
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
  return {std::move(result),std::move(copies)};
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

// The most entries one repeat of a group of WRITEs, `root` to each of `destinations` with `count` elements of
// `element_bytes`, lays on any directed cable (libnccl-mesh's ncclMeshCommWriteEntries, over the compile its WRITE steps
// take). A group of R repeats lays R times this on that list, and a list holds MESH_STREAM_ENTRIES (write_window).
static int64_t write_entries(const std::string &group, int64_t root, const std::vector<int64_t> &destinations,
                             int64_t count, int64_t element_bytes) {
  const int64_t handle = communicator_of(c10::IValue(group));
  TORCH_CHECK(handle, "torch_mesh.write_entries: ", group, " is no live mesh group of this process");
  const std::vector<int> to(destinations.begin(), destinations.end());
  uint64_t most = 0;
  checked(ncclMeshCommWriteEntries(communicator(handle), (int)root, to.data(), (int)to.size(), (size_t)count, (size_t)element_bytes,
                                   &most));
  return (int64_t)most;
}

using ViewBinding = std::tuple<at::Tensor, int64_t, std::vector<int64_t>, std::vector<int64_t>, int64_t, bool, std::vector<int64_t>, int, int>;
struct WriteOperand {
  ViewBinding view;
  size_t bytes=0,frames=1,tile=0;
  int64_t frameStride=0;
  at::Tensor ready,landings;
  uint64_t readyValue=0;
  ncclMeshTileLayout readyLayout{},landLayout{};
  uint32_t landCapacity=0;
};
struct WriteOperands {
  std::vector<WriteOperand> writes;
  std::vector<std::array<uint64_t, 6>> publish, land;
  at::Tensor tileLandings;
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
  pybind11::list declarations() const {
    pybind11::list result;int rank=0;checked(ncclCommUserRank(comm,&rank));
    auto tile=[](const ncclMeshTileLayout &x) {pybind11::dict d;
      d["columns"]=x.columns;d["tile_stride"]=x.tile_stride;d["row_stride"]=x.row_stride;
      d["instances"]=x.instances;d["instance_stride"]=x.instance_stride;return d;};
    for(const WriteOperand &write:writes) {
      const auto &[storage,bits,shape,strides,offset,mapped,runs,source,destination]=write.view;
      pybind11::dict d;d["count"]=write.bytes;d["root"]=source;d["destination"]=destination;d["fused"]=true;
      d["frames"]=write.frames;d["tile_bytes"]=write.tile;
      auto geometry=pybind11::make_tuple(storage,bits,shape,strides,offset,mapped,runs,write.frameStride);
      pybind11::tuple views(2);views[0]=rank==source?pybind11::object(geometry):pybind11::none();
      views[1]=rank==destination?pybind11::object(geometry):pybind11::none();d["views"]=views;
      d["source_ready"]=write.ready.defined()?pybind11::cast(write.ready):pybind11::none();
      d["landings"]=write.landings.defined()?pybind11::cast(write.landings):pybind11::none();
      d["ready_value"]=write.readyValue;d["ready_layout"]=tile(write.readyLayout);
      d["land_layout"]=tile(write.landLayout);d["land_capacity"]=write.landCapacity;
      result.append(d);
    }
    return result;
  }
};

static void prepare_fused_writes(MetalSchedule *schedule,WriteOperands &plans) {
  MetalScheduleCommunicator(schedule,plans.comm);
  plans.records.publishes=plans.publish.size();plans.records.lands=plans.land.size();
  if(plans.writes.empty()){plans.records.publishes=plans.records.lands=0;return;}
  WriteOperands *p=&plans;
  std::vector<uint8_t> declared;std::vector<MetalPlace> places;
  declare_group({plans.comm,1,0,plans.records.repeats,plans.records.stride,plans.polls,1},declared,places);
  const MetalDeclaration declaration{"collectives",declared.data(),declared.size(),places.data(),places.size()};
  // Fused setup publishes descriptors, not payload values. Its page storage
  // stays resident; the declared publishing/landing kernels own data edges.
  std::vector<MetalReach> views;
  for(const auto &write:plans.writes)for(const auto &tensor:{std::get<0>(write.view),write.ready,write.landings})
    if(tensor.defined()&&!tensor.is_meta()) {
      TensorOperand operand(pybind11::cast(tensor));id<MTLBuffer> buffer=operand.pages.buffer;
      views.push_back({buffer,0,buffer.length,MetalReachResident,-1});
    }
  const ncclMeshGroup publication{plans.comm,0,1};
  checked(MetalScheduleCross(schedule,views.data(),views.size(),&publication,&declaration,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
      id<MTLResidencySet> residency,NSUInteger,NSUInteger) {
    ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
      .residencySet=(__bridge void *)residency,.polls=p->polls,.fused=&p->records};
    return (int)ncclMeshCommEncode(p->comm,&on);
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
    std::shared_ptr<WriteOperands> plans[2];
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

// Allocate and agree registration cuts without presenting a receive or
// executing a producer. A fresh generation owns every source/landing/relay lease.
// A tensor's WRITE declaration retains its storage, sizes, strides and
// storage offset in elements and its element bits; no address or byte geometry (design/pointer-lowering-2026-10-28.md).
// `frames` frames of `view`, `frame_stride` elements apart, are one train (nccl.h ncclMeshLayout frames).
// An undefined `view` is a rank that ends neither side of the WRITE: its declared `unbound`
// contiguous elements of `like`'s type; the communicator plan owns relay storage.
static void tensor_write(WriteOperands &declaration,const at::Tensor &view,int source,int destination,
    size_t bytes,int64_t frames=1,int64_t frame_stride=0,int64_t unbound=0,const at::Tensor &like=at::Tensor()) {
  const auto &type=view.defined()?view:like;
  const auto storage=view.defined()?view.as_strided({(int64_t)(view.storage().nbytes()/view.element_size())},{1},0):
    at::empty({unbound},type.options().device(at::kMeta));
  const std::vector<int64_t> shape=view.defined()?view.sizes().vec():std::vector<int64_t>{unbound};
  const std::vector<int64_t> strides=view.defined()?view.strides().vec():std::vector<int64_t>{1};
  WriteOperand operand;operand.view={storage,(int64_t)(8*type.element_size()),shape,strides,
    view.defined()?view.storage_offset():0,false,{},source,destination};
  operand.bytes=bytes;operand.frames=frames;operand.frameStride=frame_stride;
  declaration.writes.push_back(std::move(operand));
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
  // registration cuts before any receive is presented. A rank that ends
  // neither side binds the library's relay, a lease of its own a pair, so
  // another destination cannot overwrite a live forwarding range or any local
  // original contribution.
  for (int h = 0; h < 2; h++) if (!ring.plans[h]) {
    auto prepared = std::make_shared<WriteOperands>();
    for (int64_t source = 0; source < f.ranks; source++) if (f.counts[source])
      for (int64_t destination = 0; destination < f.ranks; destination++) if (source != destination) {
        const bool ends = f.rank == source || f.rank == destination;
        tensor_write(*prepared,ends?ring.segments[h][0][source]:at::Tensor(),(int)source,(int)destination,
                     f.counts[source]*xc.element_size(),repeats,f.slot,f.counts[source],xc);
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
  WriteOperands *plans;
  size_t first, count, input, prefetch_first, prefetch_count;
};
struct PreparedChain {
  int64_t comm;
  std::vector<at::Tensor> held;
  std::vector<ChainStep> steps;
  std::vector<ChainChunk> chunks;
  std::vector<std::unique_ptr<WriteOperands>> inputs;
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
    // linear.metal linear_dims, all ten words: y's row and element strides (LDY, SDY) as linears binds them
    chain->products.push_back(encoded(products[0], {x, operands[0], y, y, std::nullopt, x},
        {R, K, S, 0, K, K, K, K, y.stride(0), y.stride(1)}, 4, linear_groups(R, S), 1, products[2] * MESH_SIMD_WIDTH));
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
    auto prepared = std::make_unique<WriteOperands>();
    // rank 0's input frames to each destination's own; any other rank's binding the library's relay
    for (int64_t destination = 1; destination < ranks; destination++) {
      tensor_write(*prepared,f.rank==0||f.rank==destination?source:at::Tensor(),0,(int)destination,
                   source.nbytes(),count,input.stride(0),source.numel(),source);
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
  (*prepared_chains_)[index] = std::move(chain);
}

static void compile_chain(PreparedChain &chain) {
  chain.schedule=schedule_after_inputs();
  MetalSchedule *schedule=chain.schedule.get();
  Dispatch &d = chain.dispatch;
  Dispatch &publication = *chain.publication;
  const auto &steps = chain.steps;
  const auto &chunks = chain.chunks;
  const auto binding = [&](const WriteOperands *plans, bool publication, size_t part) {
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
    WriteOperands &plans = *chain.inputs[index];
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
  const WriteOperands *previous = nullptr;
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

static pybind11::list chain_declarations(int64_t index) {
  auto &chain=*(*prepared_chains_)[index];pybind11::list result;
  auto append=[&](const WriteOperands &write){result.append(write.declarations());};
  if(!chain.inputs.empty())append(*chain.inputs[0]);
  for(const auto &chunk:chain.chunks) {
    for(size_t j=0;j<chunk.prefetch_count;j++)append(*chain.inputs[chunk.prefetch_first+j]);
    append(*chunk.plans);
  }
  return result;
}
static void chain_record(int64_t index){compile_chain(*(*prepared_chains_)[index]);}

static void chain_run(int64_t index) {
  MetalScheduleEnqueue((*prepared_chains_)[index]->schedule.get());
}
static pybind11::capsule chain_completion(int64_t index) {
  return schedule_completion((*prepared_chains_)[index]->schedule);
}

// A declared set of routed writes. Bindings name physical channels, including
// parallel cables; relay readiness stays in the compiled stream entries.

static std::shared_ptr<WriteOperands> fused_write_operands(const at::Tensor &send,
    const std::vector<std::optional<at::Tensor>> &landings,const std::vector<std::array<int,2>> &pairs,
    int64_t comm,int64_t polls,int64_t repeats,int64_t stride,const std::optional<at::Tensor> &ready,
    int64_t ready_value,int64_t tile_bytes) {
  TORCH_CHECK(landings.size()==pairs.size(),"fused writes: one landing per declared write");
  TORCH_CHECK(polls>=0&&polls<=UINT32_MAX&&repeats>0&&repeats<=UINT32_MAX&&stride>=0,
    "fused writes require finite publication geometry");
  TORCH_CHECK(tile_bytes>=0&&(!tile_bytes||(ready&&repeats==1)),"tiled writes require source readiness words");
  auto made=std::make_shared<WriteOperands>();made->prepare(comm,polls,repeats,stride);
  const size_t tiles=tile_bytes&&send.nbytes()?(send.nbytes()-1)/(size_t)tile_bytes+1:1;
  const size_t bindingWords=sizeof(ncclMeshBinding)/sizeof(uint64_t),recordWords=1+made->land.size()*bindingWords;
  if(tile_bytes) {
    made->tileLandings=region_empty({(int64_t)(tiles*recordWords)},at::empty({0},at::TensorOptions().dtype(at::kLong)),false);
    memset(made->tileLandings.data_ptr(),0,made->tileLandings.nbytes());
  }
  int rank=0;checked(ncclCommUserRank(made->comm,&rank));
  for(size_t i=0;i<pairs.size();i++) {
    const auto view=rank==pairs[i][0]?send:landings[i].value_or(at::Tensor());
    TORCH_CHECK(view.defined()||(rank!=pairs[i][0]&&rank!=pairs[i][1]),"each WRITE endpoint supplies its pages");
    TORCH_CHECK(stride%send.element_size()==0,"frame stride must be whole elements");
    if(view.defined()) {
      const size_t offset=view.storage_offset()*view.element_size(),bytes=view.nbytes();
      TORCH_CHECK(!stride||(size_t)(repeats-1)<=(SIZE_MAX-bytes)/(size_t)stride,"frame extent overflows");
      TORCH_CHECK(offset<=view.storage().nbytes()&&bytes+(repeats-1)*stride<=view.storage().nbytes()-offset,
        "all frames must lie inside the declared backing");
    }
    tensor_write(*made,view,pairs[i][0],pairs[i][1],send.nbytes(),repeats,stride/send.element_size(),send.numel(),send);
    auto &write=made->writes.back();write.tile=tile_bytes;write.readyValue=ready_value;
    if(ready) {
      TORCH_CHECK(ready->scalar_type()==at::kLong&&(size_t)ready->numel()==tiles&&ready->is_contiguous()&&ready_value>0,
        "source readiness requires one int64 word per tile");
      if(rank==pairs[i][0])write.ready=*ready;
      write.readyLayout={.columns=tiles,.tile_stride=sizeof(uint64_t),.instances=1};
    }
    if(tile_bytes&&rank==pairs[i][1]) {
      write.landings=made->tileLandings;write.landCapacity=made->land.size();
      write.landLayout={.columns=tiles,.tile_stride=recordWords*sizeof(uint64_t),.instances=1};
    }
  }
  return made;
}
static std::pair<std::vector<std::vector<uint64_t>>,std::vector<std::vector<uint64_t>>> fused_records(const WriteOperands &made) {
  std::vector<std::vector<uint64_t>> published,landed;
  for(uint32_t k=0;k<made.records.publishes;k++)published.emplace_back(made.publish[k].begin(),made.publish[k].end());
  if(made.tileLandings.defined()) {
    const size_t width=sizeof(ncclMeshBinding)/sizeof(uint64_t),recordWords=1+made.land.size()*width;
    const auto *data=(const uint64_t *)made.tileLandings.data_ptr();
    for(size_t t=0;t<(size_t)made.tileLandings.numel()/recordWords;t++)for(size_t k=0;k<made.land.size();k++) {
      const auto *record=data+t*recordWords;
      if(k<record[0])landed.emplace_back(record+1+k*width,record+1+(k+1)*width);else landed.emplace_back(width,0);
    }
  } else for(uint32_t k=0;k<made.records.lands;k++)landed.emplace_back(made.land[k].begin(),made.land[k].end());
  return {published,landed};
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
  std::vector<std::shared_ptr<WriteOperands>> prepared;
  auto writes(const std::shared_ptr<WriteOperands> &declaration) {
    prepared.push_back(declaration);prepare_fused_writes(schedule.get(),*declaration);
    return fused_records(*declaration);
  }

  void finish() {
    const auto held=operands;const auto plans=prepared;
    checked(MetalScheduleFinish(schedule.get(),^(NSError *){(void)held;(void)plans;}));
  }
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
};

// design/replay-templates-2026-10-29.md: a program's functions as one template in `directory`, `bound` (their inputs,
// outputs and the weights a later instance replaces) their operands in order, every other storage they name kept by
// value; the plan their crossings assume is declared once.
static void save(const std::string &directory,const std::vector<ScheduleBinding *> &functions,const std::vector<at::Tensor> &bound) {
  std::vector<MetalSchedule *> schedules;
  for(auto *function:functions)schedules.push_back(function->schedule.get());
  NSError *error=nil;NSArray *offsets=nil;
  NSArray *buffers=ScheduleBinding::storages(bound,&offsets);
  NSDictionary *made=MetalScheduleTemplate(schedules.data(),schedules.size(),buffers,offsets,&error);
  TORCH_CHECK(made,"template: ",error.localizedDescription.UTF8String);
  error=MetalTemplateWrite(made,@(directory.c_str()));
  TORCH_CHECK(!error,"template ",directory,": ",error.localizedDescription.UTF8String);
}

// The template in `directory` made again: its functions, each a new function after torch's inputs, `bound` their
// operands in order; `communicators` the instance's, in the order its crossings first named theirs; `host(kind, bytes)`
// the native entry (run, context, retain, release) a host operation's declaration makes, its context the caller's to pass.
static pybind11::list load(const std::string &directory,const std::vector<at::Tensor> &bound,
    const std::vector<int64_t> &communicators,const std::optional<pybind11::function> &host) {
  NSError *error=nil;
  NSDictionary *read=MetalTemplateRead(at::mps::MPSDevice::getInstance()->device(),@(directory.c_str()),&error);
  TORCH_CHECK(read,"template ",directory,": ",error.localizedDescription.UTF8String);
  std::vector<std::unique_ptr<ScheduleBinding>> made;std::vector<MetalSchedule *> schedules;
  for(NSUInteger f=0;f<[read[@"functions"] count];f++) {
    made.push_back(std::make_unique<ScheduleBinding>());schedules.push_back(made.back()->schedule.get());
  }
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
    MetalTemplateCross made_cross=^MetalCrossing(NSString *kind,NSData *bytes,MetalSchedule *schedule,ncclMeshGroup *publication,NSError **failed){
      if(![kind isEqualToString:@"collectives"]) {
        if(failed)*failed=[NSError errorWithDomain:@"mesh.template" code:1 userInfo:@{NSLocalizedDescriptionKey:
          [NSString stringWithFormat:@"no crossing of kind %@",kind]}];
        return (MetalCrossing)nil;
      }
      DeclaredGroup calls;
      try { calls=declared_group(bytes); } catch(const std::exception &e) {
        if(failed)*failed=[NSError errorWithDomain:@"mesh.template" code:1 userInfo:@{NSLocalizedDescriptionKey:@(e.what())}];
        return (MetalCrossing)nil;
      }
      MetalScheduleCommunicator(schedule,calls.comm);
      auto fused=std::make_shared<WriteOperands>();
      if(calls.fused)fused->prepare((int64_t)(uintptr_t)calls.comm,calls.polls,calls.frames,calls.stride);
      *publication=ncclMeshGroup{calls.comm,calls.invocation,calls.repeat};
      MetalCrossing crossing=^int(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,id<MTLResidencySet> residency,
          NSUInteger,NSUInteger) {
        ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
          .residencySet=(__bridge void *)residency,.polls=calls.polls,.fused=calls.fused?&fused->records:nullptr,
          .invocation=calls.invocation};
        for(size_t r=0;r<calls.repeat;r++) {
          const int result=(int)ncclMeshCommEncode(calls.comm,&on);
          if(result)return result;
        }
        return 0;
      };
      return [[crossing copy] autorelease];
    };
  NSArray *offsets=nil;NSArray *buffers=ScheduleBinding::storages(bound,&offsets);
  error=MetalScheduleInstantiate(schedules.data(),schedules.size(),read,buffers,offsets,comms,made_host,made_cross);
  TORCH_CHECK(!error,"template ",directory,": ",error.localizedDescription.UTF8String);
  // a device of an architecture the template has no pipelines for compiles them from its libraries, and keeps them
  if((error=MetalTemplateArchive(read,@(directory.c_str()))))
    TORCH_WARN("template ",directory,": this architecture's pipelines were not kept: ",error.localizedDescription.UTF8String);
  pybind11::list functions;
  for(auto &function:made){function->operands=bound;functions.append(pybind11::cast(std::move(function)));}
  return functions;
}

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
struct ViewWrites {
  int64_t comm;
  bool metal;
  std::vector<ViewBinding> bindings;
  std::vector<at::Tensor> leases;
  std::vector<MetalReach> views;
  // A meta storage declares relay shape/type; CommPlan owns its local pages.
  ViewWrites(const std::vector<ViewBinding> &given, int64_t handle) : comm(handle), metal(true), bindings(given) {
    TORCH_CHECK(!bindings.empty(), "view writes require at least one declared delivery");
    for(const auto &binding:bindings)if(!std::get<0>(binding).is_meta()){metal=std::get<0>(binding).is_mps();break;}
    for(const auto &binding:bindings)if(!std::get<0>(binding).is_meta())leases.push_back(std::get<0>(binding));
  }
  void prepare() {
    if(!views.empty())return;
    Group g{metal};
    int rank=0;checked(ncclCommUserRank(communicator(comm),&rank));
    for (const auto &[storage, bits, shape, strides, offset, mapped, runs, source, destination] : bindings) {
      TORCH_CHECK(storage.is_meta() || storage.is_mps() == metal, "view writes require one local device");
      ncclMeshBuffer pageBinding{};
      if(!storage.is_meta())pageBinding=*static_cast<const ncclMeshBuffer *>(g.where(storage));
      if(!storage.is_meta()) {
        MetalTensorLayout *selected;
        if(mapped) {
          NSMutableArray *parts=[NSMutableArray array];
          for(size_t k=0;k<runs.size();k+=2)[parts addObject:@[@(runs[k]),@(runs[k+1]),@0,@1]];
          selected=[MetalTensorLayout runs:parts];
        } else selected=[MetalTensorLayout shape:tensorNumbers(std::vector<NSInteger>(shape.begin(),shape.end()))
            strides:tensorNumbers(std::vector<NSInteger>(strides.begin(),strides.end())) offset:offset maps:@{} indices:nil];
        MetalTensorLayout *bitView=[selected components:(NSUInteger)bits];
        MetalTensorLayout *bytes=[MetalTensorLayout shape:bitView.shape index:@[@"div",@"bit",@8]
            parameters:@{@"bit":bitView}];
        const uint32_t access=rank==source?MetalReachRead:rank==destination?MetalReachWritten:MetalReachRead|MetalReachWritten;
        NSData *reaches=[bytes reaches:(__bridge id<MTLBuffer>)pageBinding.buffer base:pageBinding.offset
            elementBytes:1 access:access binding:-1];
        const auto *spans=(const MetalReach *)reaches.bytes;views.insert(views.end(),spans,spans+reaches.length/sizeof *spans);
      }
    }
  }
  // Each encoded execution is the plan's next invocation, cycling the plan's instances: NFE k+1 writes other pages
  // than NFE k sends from only where the plan declares more than one (design/pages-and-functions.md, Freeing 7).
  size_t invocations = 0;
  std::vector<Schedule> executions;
  size_t next=0;
  void append(MetalSchedule *schedule) {
    TORCH_CHECK(metal,"prepared distributed writes require the Metal backend");
    MetalScheduleCommunicator(schedule,communicator(comm));
    prepare();
    ViewWrites *self=this;
    const size_t invocation=invocations++;
    std::vector<uint8_t> declared;std::vector<MetalPlace> places;
    declare_group({communicator(comm),1,invocation,0,0,0,0},declared,places);
    const MetalDeclaration declaration{"collectives",declared.data(),declared.size(),places.data(),places.size()};
    const ncclMeshGroup publication{communicator(comm),invocation,1};
    checked(MetalScheduleCross(schedule,views.data(),views.size(),&publication,&declaration,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
        id<MTLResidencySet> residency,NSUInteger,NSUInteger) {
      ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
        .residencySet=(__bridge void *)residency};
      on.invocation=invocation;return (int)ncclMeshCommEncode(communicator(self->comm),&on);
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

// a completion's end, once it has ended (MetalScheduleEnd: woken by its last part, its time the GPU's own), the latest
// of a tuple's, in mesh_statistics.now's clock
static void completion_schedules(pybind11::handle observation, std::vector<Schedule> &into) {
  if(pybind11::isinstance<pybind11::tuple>(observation)) {
    for(auto item:observation.cast<pybind11::tuple>())completion_schedules(item,into);
    return;
  }
  auto token=pybind11::reinterpret_borrow<pybind11::capsule>(observation);
  TORCH_CHECK(!strcmp(token.name(),"mesh.prepared-function"),"torch_mesh.completion_end: ",token.name()," is no prepared function's completion");
  into.push_back(*static_cast<Schedule *>(token.get_pointer()));
}
// whether a completion has ended, without waiting (its failure raised)
static bool completion_ended(pybind11::object observation) {
  std::vector<Schedule> schedules;
  completion_schedules(observation,schedules);
  bool ended=true;
  for(const Schedule &schedule:schedules) {
    NSError *error=nil;
    const int status=MetalScheduleStatus(schedule.get(),&error);
    TORCH_CHECK(status>=0,"the prepared function failed: ",error.localizedDescription.UTF8String);
    ended=ended&&status==1;
  }
  return ended;
}
static double completion_end(pybind11::object observation) {
  std::vector<Schedule> schedules;
  completion_schedules(observation,schedules);
  double latest=0;
  for(const Schedule &schedule:schedules) {
    double ended=0;
    NSError *error=nil;
    int status;
    {
      pybind11::gil_scoped_release unlocked;
      status=MetalScheduleEnd(schedule.get(),&ended,&error);
    }
    TORCH_CHECK(status>0,"the prepared function failed: ",error.localizedDescription.UTF8String);
    TORCH_CHECK(ended>0,"torch_mesh.completion_end: a completion none of whose parts ran on the GPU or the host");
    latest=std::max(latest,ended);
  }
  return latest;
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
  module.def("import_attributes",&import_attributes, pybind11::call_guard<Drained>());
  module.def("page_backing",[](const pybind11::object &value) {
    TensorOperand operand(value);
    return std::array<uint64_t,3>{(uint64_t)(uintptr_t)operand.pages.buffer.contents,
      operand.pages.buffer.length,(uint64_t)((operand.layout.value.count?[operand.layout.value index:0]:operand.layout.value.offset)*operand.storage.element_size())};
  }, pybind11::call_guard<Drained>());
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
  module.def("completion_ended", &completion_ended, pybind11::call_guard<Drained>());
  module.def("completion_end", &completion_end, pybind11::call_guard<Drained>());
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
  compilation["LINEAR_PRODUCTS"] = LINEAR_PRODUCTS;
  module.attr("compilation") = compilation;
  module.def("linear_source", &linear_source, pybind11::call_guard<Drained>());
  module.def("linear_groups", &linear_groups, pybind11::call_guard<Drained>());
  module.def("linear_commands", &linear_commands, pybind11::call_guard<Drained>());
  pybind11::class_<Commands>(module,"Commands").def(pybind11::init<const std::vector<CommandRecord> &,const std::optional<CommandReaches> &>(),
    pybind11::arg("records"),pybind11::arg("reaches")=pybind11::none());
  module.def("linear_split", &linear_split, pybind11::call_guard<Drained>());
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
  module.def("fused_maker", &fused_maker, pybind11::call_guard<Drained>());
  module.def("fused_release", &fused_release, pybind11::call_guard<Drained>());
  pybind11::class_<ScheduleBinding>(module,"Schedule")
    .def(pybind11::init<>(), pybind11::call_guard<Drained>())
    .def("append",[](ScheduleBinding &s,Commands &commands){commands.append(s.schedule.get());
        s.operands.insert(s.operands.end(),commands.operands.begin(),commands.operands.end());}, pybind11::call_guard<Drained>())
    .def("copy",&ScheduleBinding::copy,pybind11::call_guard<Drained>())
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
    .def("append",[](Collectives &c,ScheduleBinding &s,size_t repetitions){c.append(s.schedule.get(),repetitions);
        s.operands.insert(s.operands.end(),c.operands.begin(),c.operands.end());}, pybind11::arg("schedule"),
        pybind11::arg("repetitions")=1, pybind11::call_guard<Drained>())
    .def("completion", &Collectives::completion, pybind11::call_guard<Drained>())
    .def("enqueue", &Collectives::enqueue, pybind11::call_guard<Drained>());
  module.def("save",&save,pybind11::arg("directory"),pybind11::arg("functions"),pybind11::arg("bound"),pybind11::call_guard<Drained>());
  module.def("load",&load,pybind11::arg("directory"),pybind11::arg("bound"),
             pybind11::arg("communicators")=std::vector<int64_t>{},pybind11::arg("host")=pybind11::none(),pybind11::call_guard<Drained>());
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
  module.def("write_entries", &write_entries, pybind11::call_guard<Drained>());
  module.attr("STREAM_ENTRIES") = (int64_t)MESH_STREAM_ENTRIES;
  pybind11::class_<WriteOperands,std::shared_ptr<WriteOperands>>(module,"WriteOperands")
    .def("declarations",&WriteOperands::declarations);
  module.def("write_operands",&fused_write_operands,pybind11::call_guard<Drained>());
  module.def("chain_slot", &chain_slot, pybind11::call_guard<Drained>());
  module.def("prepare_chain", &prepare_chain, pybind11::call_guard<Drained>());
  module.def("chain_declarations", &chain_declarations, pybind11::call_guard<Drained>());
  module.def("chain_record", &chain_record, pybind11::call_guard<Drained>());
  module.def("chain_run", &chain_run, pybind11::call_guard<Drained>());
  module.def("chain_completion", &chain_completion, pybind11::call_guard<Drained>());
  module.def("fused_prepare", &fused_prepare, pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("dim"),
             pybind11::arg("sizes"), pybind11::arg("group"), pybind11::arg("adds"),
             pybind11::arg("repeats") = 1, pybind11::arg("batches") = 1, pybind11::call_guard<Drained>());
}
