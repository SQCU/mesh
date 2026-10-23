#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/DLConvertor.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSEvent.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <vector>
#include "nccl.h"

struct MeshStream {
  struct Exchanged { void *bound[3] = {nullptr, nullptr, nullptr}; int64_t last = 0; };
  at::mps::MPSStream *const mps;
  id<MTLSharedEvent> const progress;
  uint64_t committed = 0;
  id<MTLCommandBuffer> open = nil;
  bool streamed = false;
  void *programmed = nullptr;
  uint32_t programs = 0;
  std::unordered_map<int64_t, Exchanged> exchanged;

  explicit MeshStream(at::mps::MPSStream *stream) : mps(stream), progress([stream->device() newSharedEvent]) {}

  bool waiting() {
    if (open && open.status < MTLCommandBufferStatusCommitted) return true;
    [open release];
    open = nil;
    return false;
  }

  void signalled(bool wait) {
    mps->endKernelCoalescing();
    [mps->commandBuffer() encodeSignalEvent:progress value:++committed];
    mps->synchronize(wait ? at::mps::SyncType::COMMIT_AND_WAIT : at::mps::SyncType::COMMIT);
    [open release];
    open = nil;
  }

  void published() {
    if (!waiting()) open = [mps->commandBuffer().rootCommandBuffer retain];
    if (committed - progress.signaledValue <= 1) signalled(false);
  }
};

static std::mutex *streams_lock_ = new std::mutex;
static auto *streams_ = new std::unordered_map<at::mps::MPSStream *, MeshStream *>;

static MeshStream &stream_of(at::mps::MPSStream *mps) {
  thread_local std::pair<at::mps::MPSStream *, MeshStream *> last{nullptr, nullptr};
  if (last.first == mps) return *last.second;
  std::lock_guard<std::mutex> guard(*streams_lock_);
  auto found = streams_->find(mps);
  if (found == streams_->end()) found = streams_->emplace(mps, new MeshStream(mps)).first;
  last = {mps, found->second};
  return *found->second;
}

static std::vector<MeshStream *> streams_made() {
  std::lock_guard<std::mutex> guard(*streams_lock_);
  std::vector<MeshStream *> made;
  for (auto &[mps, s] : *streams_) made.push_back(s);
  return made;
}

static void host_wait(MeshStream &s, bool wait) {
  MeshStream *self = &s;
  at::mps::dispatch_sync_with_rethrow(s.mps->queue(), ^{
    @autoreleasepool {
      if (self->waiting()) self->signalled(wait);
      else if (wait) self->mps->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    }
  });
}

static void commit() {
  for (MeshStream *s : streams_made()) host_wait(*s, false);
}

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

static std::unordered_map<int64_t, std::vector<std::pair<int64_t, int64_t>>> marks_;

static void mark(int64_t comm, int64_t kind) {
  if (marks_.empty()) return;
  auto found = marks_.find(comm);
  if (found == marks_.end()) return;
  auto pool = at::mps::getMPSEventPool();
  const auto id = pool->acquireEvent(true);
  pool->recordEvent(id, false);
  found->second.emplace_back(kind, (int64_t)id);
}

static uint64_t on_stream(const std::function<ncclResult_t(ncclMeshStream *)> &call, bool split, int64_t comm) {
  const auto *encode = &call;
  mark(comm, 0);
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  __block ncclResult_t result = ncclSuccess;
  __block ncclMeshIssue issue = {0, 0};
  dispatch_sync(stream->queue(), ^{
    @autoreleasepool {
      void *root = (__bridge void *)stream->commandBuffer().rootCommandBuffer;
      if (root != s->programmed) { s->programmed = root; s->programs = 0; }
      if (s->programs >= 4) {
        s->signalled(false);
        s->programmed = (__bridge void *)stream->commandBuffer().rootCommandBuffer;
        s->programs = 0;
      }
      s->programs++;
      ncclMeshStream on{(__bridge void *)stream->commandBuffer().rootCommandBuffer, (__bridge void *)stream->commandEncoder(), split ? &issue : nullptr};
      result = (*encode)(&on);
      if (!split || issue.published) s->published();
    }
  });
  s->streamed = true;
  checked(result);
  if (!issue.ticket) mark(comm, 1);
  return issue.ticket;
}

static void complete(uint64_t ticket) {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  if (ncclMeshRetired() >= ticket) return;
  __block ncclResult_t result = ncclSuccess;
  dispatch_sync(stream->queue(), ^{
    @autoreleasepool {
      ncclMeshStream on{(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder()};
      int published = 0;
      result = ncclMeshComplete(ticket, &on, &published);
      if (published) s->published();
    }
  });
  checked(result);
}

class MeshWork : public c10d::Work {
 public:
  MeshWork(uint64_t ticket, std::vector<at::Tensor> outputs, int64_t comm)
      : c10d::Work(-1, c10d::OpType::UNKNOWN), ticket_(ticket), outputs_(std::move(outputs)), comm_(comm) {}
  bool isCompleted() override { finish(); return true; }
  bool isSuccess() const override { return finished_; }
  bool wait(std::chrono::milliseconds) override { finish(); return true; }
  void synchronize() override { finish(); }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    finish();
    auto future = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
    future->markCompleted(c10::IValue(outputs_));
    return future;
  }

 private:
  void finish() {
    if (finished_) return;
    finished_ = true;
    complete(ticket_);
    mark(comm_, 1);
  }
  uint64_t ticket_;
  std::vector<at::Tensor> outputs_;
  int64_t comm_;
  bool finished_ = false;
};

static c10::intrusive_ptr<c10d::Work> work_of(uint64_t ticket, std::vector<at::Tensor> outputs, int64_t comm) {
  if (!ticket) return {};
  return c10::make_intrusive<MeshWork>(ticket, std::move(outputs), comm);
}

static ncclComm_t communicator(int64_t comm) { return (ncclComm_t)(uintptr_t)comm; }

static at::Tensor region_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps);

enum Kind : int64_t { ALLREDUCE, REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL, SEND, RECV, PAIRS };
using Tensors = std::vector<std::optional<at::Tensor>>;
using Call = std::tuple<int64_t, Tensors, Tensors, std::vector<int64_t>, int64_t, int64_t>;

struct Operand {
  const void *at = nullptr;
  size_t count = 0;
  ncclDataType_t type = ncclUint8;
};

struct Group {
  bool metal = false;
  std::deque<ncclMeshBuffer> buffers;

  const void *where(const at::Tensor &t) {
    if (!t.is_mps()) return t.data_ptr();
    buffers.push_back({(__bridge void *)at::native::mps::getMTLBufferStorage(t), (size_t)(t.storage_offset() * t.element_size())});
    return &buffers.back();
  }
  at::Tensor contiguous(const at::Tensor &t) {
    TORCH_CHECK(t.is_mps() == metal, "the mesh backend's group takes MPS tensors or CPU tensors, not both");
    TORCH_CHECK(t.is_contiguous(), "the mesh backend's operand is a contiguous region tensor (torch_mesh.empty); a non-contiguous "
                "one is not copied into one (zero copy: nccl.h Memory)");
    return t;
  }
  Operand moving(const at::Tensor &tensor) {
    const at::Tensor t = contiguous(tensor);
    const ncclDataType_t type = kind(t.scalar_type());
    return type == ncclNumTypes ? Operand{where(t), (size_t)t.nbytes(), ncclUint8} : Operand{where(t), (size_t)t.numel(), type};
  }
  Operand reducing(const at::Tensor &tensor, int64_t &op) {
    const at::Tensor t = contiguous(tensor);
    if (t.scalar_type() == at::kBool) op = op == ncclSum ? ncclMax : op == ncclProd ? ncclMin : op;
    if (t.is_complex()) {
      TORCH_CHECK(op == ncclSum || op == ncclAvg, "a complex tensor reduces by sum and average alone");
      return {where(t), (size_t)t.numel() * 2, kind(at::toRealValueType(t.scalar_type()))};
    }
    TORCH_CHECK(kind(t.scalar_type()) != ncclNumTypes, "the mesh backend reduces no ", t.scalar_type());
    return {where(t), (size_t)t.numel(), kind(t.scalar_type())};
  }
};

struct Issue {
  int64_t kind, op, root;
  std::vector<Operand> sends, receives;
  std::vector<size_t> counts;
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
    const bool reduces = what == REDUCE_SCATTER;
    const at::Tensor &whole = reduces ? tensor(sends, 0) : tensor(receives, 0);
    i.sends = {reduces ? g.reducing(tensor(sends, 0), i.op) : g.moving(tensor(sends, 0))};
    i.receives = {reduces ? g.reducing(tensor(receives, 0), i.op) : g.moving(tensor(receives, 0))};
    const size_t per = whole.numel() ? (reduces ? i.sends : i.receives)[0].count / (size_t)whole.numel() : 1;
    for (auto c : counts) i.counts.push_back((size_t)c * per);
    break;
  }
  case ALLTOALL: {
    TORCH_CHECK(sends.size() == 1 && receives.size() == 1 && (sends[0] || receives[0]), "an all-to-all names its send and receive buffers");
    i.sends = {sends[0] ? g.moving(*sends[0]) : Operand{}};
    i.receives = {receives[0] ? g.moving(*receives[0]) : Operand{}};
    const at::Tensor &whole = sends[0] ? *sends[0] : *receives[0];
    const size_t per = whole.numel() ? (sends[0] ? i.sends : i.receives)[0].count / (size_t)whole.numel() : 1;
    for (auto c : counts) i.counts.push_back((size_t)c * per);
    break;
  }
  case SEND: i.sends = {g.moving(tensor(sends, 0))}; break;
  case RECV: i.receives = {g.moving(tensor(receives, 0))}; break;
  case PAIRS: i.counts.assign(counts.begin(), counts.end()); break;
  default: TORCH_CHECK(false, "no collective of kind ", what);
  }
  return i;
}

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
    case ALLGATHER: result = ncclMeshAllGatherV(a.at, b, i.counts.data(), a.type, comm, s); break;
    case REDUCE_SCATTER: result = ncclMeshReduceScatterV(a.at, b, i.counts.data(), a.type, op, comm, s); break;
    case ALLTOALL:
      result = ncclMeshAlltoAllV(a.at, i.counts.data(), b, a.at ? a.type : i.receives[0].type, comm, s);
      break;
    case SEND: result = ncclSend(a.at, a.count, a.type, (int)i.root, comm, s); break;
    case PAIRS: {
      const size_t count = i.counts.size() / 3;
      std::vector<int> from(count), to(count);
      std::vector<size_t> bytes(count);
      for (size_t k = 0; k < count; k++) { from[k] = (int)i.counts[3 * k]; to[k] = (int)i.counts[3 * k + 1]; bytes[k] = i.counts[3 * k + 2]; }
      result = ncclMeshGroupPairs(from.data(), to.data(), bytes.data(), (int)count, comm, s);
      break;
    }
    case RECV: result = ncclRecv(b, i.receives[0].count, i.receives[0].type, (int)i.root, comm, s); break;
    }
  }
  const ncclResult_t ended = ncclGroupEnd();
  return result ? result : ended;
}

static uint64_t issue(const std::vector<Call> &calls, int64_t comm, bool split, bool unlock) {
  Group g;
  for (const Call &c : calls)
    for (const Tensors *list : {&std::get<1>(c), &std::get<2>(c)})
      for (size_t k = 0; k < list->size(); k++)
        g.metal |= (*list)[k] && (*list)[k]->is_mps();
  std::vector<Issue> issues;
  for (const Call &c : calls) issues.push_back(prepare(c, g));
  uint64_t ticket = 0;
  if (g.metal) {
    ticket = on_stream([&](ncclMeshStream *s) { return encoded(issues, communicator(comm), s); }, split, comm);
  } else {
    bool streamed = false;
    for (MeshStream *s : streams_made()) streamed |= s->streamed;
    if (streamed) {
      complete(UINT64_MAX);
      const MeshStream *current = &stream_of(at::mps::getCurrentMPSStream());
      for (MeshStream *s : streams_made())
        if (s->streamed || s == current) {
          host_wait(*s, true);
          s->streamed = false;
        }
    }
    std::optional<pybind11::gil_scoped_release> unlocked;
    if (unlock) unlocked.emplace();
    checked(encoded(issues, communicator(comm), nullptr));
  }
  return ticket;
}

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

static int64_t reduction(const std::string &op) {
  if (op == "sum") return ncclSum;
  if (op == "avg") return ncclAvg;
  if (op == "product") return ncclProd;
  if (op == "max") return ncclMax;
  if (op == "min") return ncclMin;
  TORCH_CHECK(false, "the mesh backend reduces by sum, avg, product, max and min, not ", op);
}

static int group_argument(const std::string &name) {
  if (name == "all_reduce" || name == "all_reduce_" || name == "broadcast" || name == "broadcast_" || name == "all_gather_into_tensor" ||
      name == "all_gather_into_tensor_out")
    return 2;
  return 3;
}

static void functional(const c10::OperatorHandle &op, torch::jit::Stack *stack) {
  const std::string name = op.schema().name().substr(sizeof "_c10d_functional::" - 1) +
                           (op.schema().overload_name().empty() || op.schema().overload_name() == "default" ? "" : "." + op.schema().overload_name());
  const size_t count = op.schema().arguments().size();
  auto arguments = torch::jit::last(*stack, count);
  const at::Tensor input = arguments[0].toTensor();
  const int64_t comm = communicator_of(arguments[group_argument(name)]);
  if (!comm) {
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
    return;
  }
  int ranks = 0;
  checked(ncclCommCount(communicator(comm), &ranks));
  at::Tensor result;
  Call call;
  if (name.rfind("all_reduce", 0) == 0 || name.rfind("broadcast", 0) == 0) {
    result = name.back() == '_' ? input : region_empty(input.sizes().vec(), input, true).copy_(input);
    call = name[0] == 'a' ? Call{ALLREDUCE, {result}, {}, {}, reduction(arguments[1].toStringRef()), 0}
                          : Call{BROADCAST, {result}, {}, {}, 0, arguments[1].toInt()};
  } else if (name.rfind("all_gather_into_tensor", 0) == 0) {
    std::vector<int64_t> shape = input.sizes().vec();
    TORCH_CHECK(!shape.empty(), "all_gather_into_tensor of a 0-dim tensor");
    shape[0] *= ranks;
    result = name == "all_gather_into_tensor_out" ? arguments[3].toTensor() : region_empty(shape, input, true);
    TORCH_CHECK(result.numel() == input.numel() * ranks, "all_gather_into_tensor_out: out holds ", result.numel(), " elements, not ",
                input.numel() * ranks);
    call = {ALLGATHER, {input}, {result}, std::vector<int64_t>(ranks, input.numel()), 0, 0};
  } else if (name == "reduce_scatter_tensor") {
    std::vector<int64_t> shape = input.sizes().vec();
    TORCH_CHECK(!shape.empty() && shape[0] % ranks == 0, "reduce_scatter_tensor: dim 0 of ", input.sizes(), " is not divisible by ", ranks);
    shape[0] /= ranks;
    result = region_empty(shape, input, true);
    call = {REDUCE_SCATTER, {input}, {result}, std::vector<int64_t>(ranks, result.numel()), reduction(arguments[1].toStringRef()), 0};
  } else {
    std::vector<int64_t> outs, ins, counts;
    for (const auto &v : arguments[1].toListRef()) outs.push_back(v.toInt());
    for (const auto &v : arguments[2].toListRef()) ins.push_back(v.toInt());
    TORCH_CHECK(input.dim(), "all_to_all_single of a 0-dim tensor");
    if (outs.empty()) outs.assign(ranks, input.size(0) / ranks);
    if (ins.empty()) ins.assign(ranks, input.size(0) / ranks);
    std::vector<int64_t> shape = input.sizes().vec();
    int64_t width = 1;
    for (int64_t d = 1; d < input.dim(); d++) width *= input.size(d);
    shape[0] = 0;
    for (auto r : ins) counts.push_back(r * width);
    for (auto r : outs) {
      counts.push_back(r * width);
      shape[0] += r;
    }
    result = region_empty(shape, input, true);
    call = {ALLTOALL, {input}, {result}, counts, 0, 0};
  }
  const uint64_t ticket = issue({call}, comm, true, false);
  if (ticket) c10d::register_work(result, work_of(ticket, {result}, comm));
  torch::jit::drop(*stack, count);
  torch::jit::push(*stack, result);
}

TORCH_LIBRARY_IMPL(_c10d_functional, MPS, m) {
  for (const char *name : {"all_reduce", "all_reduce_", "broadcast", "broadcast_", "all_gather_into_tensor", "all_gather_into_tensor_out",
                           "reduce_scatter_tensor", "all_to_all_single"})
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&functional>());
}

static std::vector<id<MTLComputePipelineState>> pipelines_;

static int64_t pipeline(const std::string &source, const std::string &name) {
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.languageVersion = MTLLanguageVersion4_0;
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:@(source.c_str()) options:options error:&error];
  if (!library) throw std::runtime_error(std::string("kernel source: ") + error.localizedDescription.UTF8String);
  id<MTLFunction> function = [library newFunctionWithName:@(name.c_str())];
  if (!function) throw std::runtime_error("kernel source: no function " + name);
  id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
  if (!state) throw std::runtime_error(std::string("kernel pipeline: ") + error.localizedDescription.UTF8String);
  pipelines_.push_back(state);
  return (int64_t)pipelines_.size() - 1;
}

static std::tuple<int64_t, int64_t, int64_t> device_facts() {
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  int64_t family = 0;
  while ([device supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + family)]) family++;
  return {family, (int64_t)device.maxThreadsPerThreadgroup.width, (int64_t)device.maxThreadgroupMemoryLength};
}

static std::tuple<int64_t, int64_t, int64_t> pipeline_facts(int64_t handle) {
  if (handle < 0 || handle >= (int64_t)pipelines_.size()) throw std::runtime_error("pipeline_facts: no kernel " + std::to_string(handle));
  id<MTLComputePipelineState> state = pipelines_[handle];
  return {(int64_t)state.maxTotalThreadsPerThreadgroup, (int64_t)state.threadExecutionWidth,
          (int64_t)state.staticThreadgroupMemoryLength};
}

static void encode(int64_t handle, const std::vector<std::optional<at::Tensor>> &buffers, const std::vector<int64_t> &constants,
                   int64_t at, int64_t gx, int64_t gy, int64_t threads, int64_t request, int64_t start, bool lands, bool publishes, int64_t crossing) {
  if (handle < 0 || handle >= (int64_t)pipelines_.size()) throw std::runtime_error("encode: no kernel " + std::to_string(handle));
  id<MTLComputePipelineState> state = pipelines_[handle];
  std::vector<id<MTLBuffer>> bound(buffers.size(), nil);
  std::vector<NSUInteger> offsets(buffers.size(), 0);
  for (size_t i = 0; i < buffers.size(); i++)
    if (buffers[i]) {
      const at::Tensor &t = *buffers[i];
      if (!t.is_mps()) throw std::runtime_error("encode: buffer " + std::to_string(i) + " is not an MPS tensor");
      bound[i] = at::native::mps::getMTLBufferStorage(t);
      offsets[i] = (NSUInteger)(t.storage_offset() * t.element_size());
    }
  std::vector<uint32_t> words(constants.begin(), constants.end());
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  __block ncclResult_t result = ncclSuccess;
  dispatch_sync(stream->queue(), ^{
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
      [encoder setComputePipelineState:state];
      for (size_t i = 0; i < bound.size(); i++)
        if (bound[i]) [encoder setBuffer:bound[i] offset:offsets[i] atIndex:i];
      [encoder setBytes:words.data() length:words.size() * sizeof(uint32_t) atIndex:(NSUInteger)at];
      ncclMeshStream on{(__bridge void *)stream->commandBuffer().rootCommandBuffer, (__bridge void *)encoder};
      if (request)
        result = ncclMeshBind((ncclMeshRequest_t)(uintptr_t)request, (uint64_t)start, lands, publishes, &on, (uint32_t)crossing, (uint32_t)(gx * gy));
      if (result) return;
      [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)threads, 1, 1)];
      if (request && (lands || publishes)) {
        MeshStream::Exchanged &x = s->exchanged[request];
        void *root = (__bridge void *)stream->commandBuffer().rootCommandBuffer;
        if (lands) x.bound[(start - 1) % 3] = root;
        if (publishes) x.bound[start % 3] = root;
      }
      if (request && publishes) s->published();
    }
  });
  checked(result);
}

static std::vector<double> timed(int64_t handle, const std::vector<std::optional<at::Tensor>> &buffers,
                                 const std::vector<int64_t> &constants, int64_t at, int64_t gx, int64_t gy, int64_t threads,
                                 int64_t repeats) {
  if (handle < 0 || handle >= (int64_t)pipelines_.size()) throw std::runtime_error("timed: no kernel " + std::to_string(handle));
  id<MTLComputePipelineState> state = pipelines_[handle];
  std::vector<id<MTLBuffer>> bound(buffers.size(), nil);
  std::vector<NSUInteger> offsets(buffers.size(), 0);
  for (size_t i = 0; i < buffers.size(); i++)
    if (buffers[i]) {
      const at::Tensor &t = *buffers[i];
      bound[i] = at::native::mps::getMTLBufferStorage(t);
      offsets[i] = (NSUInteger)(t.storage_offset() * t.element_size());
    }
  std::vector<uint32_t> words(constants.begin(), constants.end());
  id<MTLCommandQueue> queue = at::mps::getCurrentMPSStream()->commandQueue();
  std::vector<double> out;
  for (int64_t r = 0; r < repeats; r++) {
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    for (size_t i = 0; i < bound.size(); i++)
      if (bound[i]) [encoder setBuffer:bound[i] offset:offsets[i] atIndex:i];
    [encoder setBytes:words.data() length:words.size() * sizeof(uint32_t) atIndex:(NSUInteger)at];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)threads, 1, 1)];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
    if (buffer.error) throw std::runtime_error(std::string("timed: ") + buffer.error.localizedDescription.UTF8String);
    out.push_back(buffer.GPUEndTime - buffer.GPUStartTime);
  }
  return out;
}

static double sequence(int64_t handle, const std::vector<std::vector<std::optional<at::Tensor>>> &sets,
                       const std::vector<int64_t> &constants, int64_t at, int64_t gx, int64_t gy, int64_t threads, int64_t count) {
  if (handle < 0 || handle >= (int64_t)pipelines_.size()) throw std::runtime_error("sequence: no kernel " + std::to_string(handle));
  if (sets.empty() || count < 1) throw std::runtime_error("sequence: no buffer sets or dispatches");
  std::vector<std::vector<id<MTLBuffer>>> bound;
  std::vector<std::vector<NSUInteger>> offsets;
  for (const auto &set : sets) {
    bound.emplace_back(set.size(), nil);
    offsets.emplace_back(set.size(), 0);
    for (size_t i = 0; i < set.size(); i++)
      if (set[i]) {
        bound.back()[i] = at::native::mps::getMTLBufferStorage(*set[i]);
        offsets.back()[i] = (NSUInteger)(set[i]->storage_offset() * set[i]->element_size());
      }
  }
  std::vector<uint32_t> words(constants.begin(), constants.end());
  id<MTLCommandQueue> queue = at::mps::getCurrentMPSStream()->commandQueue();
  id<MTLCommandBuffer> buffer = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
  [encoder setComputePipelineState:pipelines_[handle]];
  [encoder setBytes:words.data() length:words.size() * sizeof(uint32_t) atIndex:(NSUInteger)at];
  for (int64_t r = 0; r < count; r++) {
    const size_t s = (size_t)r % bound.size();
    for (size_t i = 0; i < bound[s].size(); i++)
      if (bound[s][i]) [encoder setBuffer:bound[s][i] offset:offsets[s][i] atIndex:i];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)threads, 1, 1)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  [encoder endEncoding];
  [buffer commit];
  [buffer waitUntilCompleted];
  if (buffer.error) throw std::runtime_error(std::string("sequence: ") + buffer.error.localizedDescription.UTF8String);
  return (buffer.GPUEndTime - buffer.GPUStartTime) / (double)count;
}

static std::mutex region_tensors_lock_;
static std::map<uintptr_t, size_t> region_tensors_;

static void forgotten(void *memory) {
  std::lock_guard<std::mutex> guard(region_tensors_lock_);
  region_tensors_.erase((uintptr_t)memory);
}

static void returned(void *memory) {
  ncclMemFree(memory);
  forgotten(memory);
}

struct RegionShape {
  std::vector<int64_t> sizes, strides;
};

static at::Tensor region_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps) {
  const at::ScalarType dtype = like.scalar_type();
  const size_t bytes = std::max<size_t>(1, (size_t)c10::multiply_integers(sizes) * c10::elementSize(dtype));
  const size_t word = (bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t) * sizeof(uint32_t);
  void *memory = nullptr;
  checked(ncclMemAlloc(&memory, mps ? word + sizeof(uint32_t) : bytes));
  if (mps) *(volatile uint32_t *)((char *)memory + word) = 0;
  {
    std::lock_guard<std::mutex> guard(region_tensors_lock_);
    region_tensors_[(uintptr_t)memory] = bytes;
  }
  const auto options = at::TensorOptions().dtype(dtype);
  if (!mps) return at::from_blob(memory, sizes, [memory](void *) { returned(memory); }, options.device(at::kCPU));
  void *buffer = nullptr;
  size_t offset = 0;
  if (const ncclResult_t result = ncclMeshMemBuffer(memory, &buffer, &offset)) {
    returned(memory);
    checked(result);
  }
  auto *shape = new RegionShape{sizes, std::vector<int64_t>(std::max<size_t>(sizes.size(), 1), 1)};
  for (int64_t d = (int64_t)sizes.size() - 2; d >= 0; d--) shape->strides[d] = shape->strides[d + 1] * std::max<int64_t>(sizes[d + 1], 1);
  auto *managed = new DLManagedTensor{};
  managed->dl_tensor.data = buffer;
  managed->dl_tensor.device = {kDLMetal, 0};
  managed->dl_tensor.ndim = (int32_t)sizes.size();
  managed->dl_tensor.dtype = at::getDLDataType(at::empty({0}, options));
  managed->dl_tensor.shape = shape->sizes.data();
  managed->dl_tensor.strides = shape->strides.data();
  managed->dl_tensor.byte_offset = offset;
  managed->manager_ctx = shape;
  managed->deleter = [](DLManagedTensor *self) {
    delete (RegionShape *)self->manager_ctx;
    delete self;
  };
  return at::fromDLPack(managed, [managed, memory, word](void *) {
    managed->deleter(managed);
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
      ncclMeshStream on{(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder()};
      if (ncclMeshMemFreeAfter(memory, word, &on) == ncclSuccess) forgotten(memory);
    });
  });
}

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

static at::Tensor cpu_view(const at::Tensor &t) {
  const uint64_t where = address(t);
  TORCH_CHECK(where, "cpu_view: the tensor's bytes are not in the region (torch_mesh.empty makes a tensor that is)");
  return at::from_blob((void *)(uintptr_t)where, t.sizes(), t.strides(), [held = t](void *) { (void)held; }, t.options().device(at::kCPU));
}

static std::vector<std::pair<uint64_t, uint64_t>> records() {
  std::lock_guard<std::mutex> guard(region_tensors_lock_);
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (auto &w : region_tensors_) out.emplace_back(w.first, w.second);
  return out;
}

static std::string metal_prelude() { return ncclMeshMetalPrelude(); }

static std::pair<int64_t, int64_t> exchange_init(const std::vector<int64_t> &sizes, const at::Tensor &like, int64_t comm) {
  int rank = 0, ranks = 0;
  checked(ncclCommUserRank(communicator(comm), &rank));
  checked(ncclCommCount(communicator(comm), &ranks));
  TORCH_CHECK(sizes.size() == (size_t)ranks, "an exchange names each of the ", ranks, " ranks' parts, not ", sizes.size());
  const ncclDataType_t type = kind(like.scalar_type());
  const size_t scale = type == ncclNumTypes ? (size_t)like.element_size() : 1;
  std::vector<size_t> parts(ranks, 0);
  std::vector<int> lands((size_t)ranks * ranks, 1);
  for (int q = 0; q < ranks; q++) {
    parts[q] = (size_t)sizes[q] * scale;
    lands[(size_t)q * ranks + q] = 0;
  }
  ncclMeshRequest_t made = nullptr;
  checked(ncclMeshExchangeInit(parts.data(), lands.data(), type == ncclNumTypes ? ncclUint8 : type, communicator(comm), &made));
  return {(int64_t)(uintptr_t)made, (int64_t)scale};
}

static int64_t exchange_start(int64_t request, const std::vector<int64_t> &sizes) {
  std::vector<size_t> counts(sizes.begin(), sizes.end());
  const size_t *given = counts.empty() ? nullptr : counts.data();
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  __block ncclResult_t result = ncclSuccess;
  __block uint64_t start = 0;
  dispatch_sync(stream->queue(), ^{
    @autoreleasepool {
      MeshStream::Exchanged &x = s->exchanged[request];
      const int64_t held = x.last - 2;
      if (s->waiting() || (held > 0 && x.bound[held % 3] == (__bridge void *)stream->commandBuffer().rootCommandBuffer)) s->signalled(false);
      ncclMeshStream on{(__bridge void *)stream->commandBuffer().rootCommandBuffer, (__bridge void *)stream->commandEncoder()};
      result = ncclMeshStart((ncclMeshRequest_t)(uintptr_t)request, given, &on, &start);
      if (result == ncclSuccess) x.last = (int64_t)start;
    }
  });
  s->streamed = true;
  checked(result);
  return (int64_t)start;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pipeline", &pipeline);
  module.def("device_facts", &device_facts);
  module.def("pipeline_facts", &pipeline_facts);
  module.def("encode", &encode, pybind11::arg("handle"), pybind11::arg("buffers"), pybind11::arg("constants"), pybind11::arg("at"),
             pybind11::arg("gx"), pybind11::arg("gy"), pybind11::arg("threads"), pybind11::arg("request") = 0, pybind11::arg("start") = 0,
             pybind11::arg("lands") = false, pybind11::arg("publishes") = false, pybind11::arg("crossing") = 14);
  module.def("timed", &timed);
  module.def("sequence", &sequence);
  module.def("watch", [](int64_t comm) { marks_.try_emplace(comm); });
  module.def("mark", &mark);
  module.def("take", [](int64_t comm) {
    auto found = marks_.find(comm);
    std::vector<std::pair<int64_t, int64_t>> out;
    if (found != marks_.end()) out.swap(found->second);
    return out;
  });
  module.def("issue", [](const std::vector<Call> &calls, int64_t comm, bool split, std::vector<at::Tensor> outputs) {
    return work_of(issue(calls, comm, split, true), std::move(outputs), comm);
  });
  module.def("attach", &attach);
  module.def("commit", &commit);
  module.def("region_empty", &region_empty);
  module.def("address", &address);
  module.def("cpu_view", &cpu_view);
  module.def("records", &records);
  module.def("metal_prelude", &metal_prelude);
  module.def("exchange_init", &exchange_init);
  module.def("exchange_start", &exchange_start);
  module.def("exchange_free", [](int64_t request) {
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    MeshStream *s = &stream_of(stream);
    dispatch_sync(stream->queue(), ^{ s->exchanged.erase(request); });
    checked(ncclMeshRequestFree((ncclMeshRequest_t)(uintptr_t)request));
  });
}
