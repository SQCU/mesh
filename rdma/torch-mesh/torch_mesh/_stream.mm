// torch's current MPS stream as libnccl-mesh's ncclMeshStream (nccl.h): begin() gives the stream's command buffer,
// its open compute encoder (the group's kernels join torch's own) and a workspace allocator (MPS tensors from
// torch's caching allocator, held until the group is complete, then reused in the stream's order); commit() commits
// the work without waiting.  A collective issued and not completed (an async op, a functional collective) is a
// MeshWork: its wait encodes the rest (its landings, later rounds and results) on the stream where it is waited.
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <map>
#include <unordered_map>
#include <vector>
#include "nccl.h"

struct Stream {
  void *commandBuffer, *commandEncoder;
  void *(*workspace)(size_t, void *);
  void *context;
  ncclMeshIssue *issue;
};

static Stream stream_;
// Within compiled-graph regions (batch): the command buffers committed, each signalling its count on progress_ as the
// GPU reaches its end (a shared event's value is the GPU's progress at once; a completion handler runs 7-9 us
// later, at the 99th percentile 20-60), and whether a collective was left in torch's open buffer.
static id<MTLSharedEvent> progress_;
static uint64_t committed_ = 0;
static int batching_ = 0;
static bool deferred_ = false;
static std::vector<at::Tensor> held_;
// each issued group's workspace, by ticket, until the library has retired it
static std::map<uint64_t, std::vector<at::Tensor>> holders_;

static void release() {
  const uint64_t retired = ncclMeshRetired();
  holders_.erase(holders_.begin(), holders_.upper_bound(retired));
}

static void *workspace(size_t bytes, void *) {
  held_.push_back(at::empty({(int64_t)bytes}, at::TensorOptions().dtype(at::kByte).device(at::kMPS)));
  return (__bridge void *)at::native::mps::getMTLBufferStorage(held_.back());
}

static int64_t begin() {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  __block void *buffer = nullptr, *encoder = nullptr;
  dispatch_sync(stream->queue(), ^{
    buffer = (__bridge void *)stream->commandBuffer();
    encoder = (__bridge void *)stream->commandEncoder();
  });
  stream_ = Stream{buffer, encoder, workspace, nullptr, nullptr};
  return (int64_t)(uintptr_t)&stream_;
}

// On torch's queue: its open buffer committed (within a compiled graph, counted on progress_).
static void committing(at::mps::MPSStream *stream) {
  if (batching_) {
    if (!progress_) progress_ = [stream->device() newSharedEvent];
    stream->endKernelCoalescing();
    [stream->commandBuffer() encodeSignalEvent:progress_ value:++committed_];
  }
  stream->synchronize(at::mps::SyncType::COMMIT);
  deferred_ = false;
}

// Work the peers wait on (a publication): committed at once outside a compiled graph (the host may next block
// outside torch while a peer waits on it); within one, while two buffers are in flight it joins the open one.
static void publishing(at::mps::MPSStream *stream) {
  if (!batching_ || committed_ - progress_.signaledValue < 2)
    committing(stream);
  else
    deferred_ = true;
}

static void commit() {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{ committing(stream); });
  held_.clear();
  release();
}

// A compiled graph's run (torch_mesh/_mps.py): +1 as it starts, -1 as it ends, when a collective left open is committed.
static void batch(int64_t delta) {
  batching_ += (int)delta;
  if (!batching_ && deferred_) commit();
}

static bool moved(at::ScalarType type) {
  switch (type) {
  case at::kChar: case at::kByte: case at::kBool: case at::kInt: case at::kLong: case at::kHalf: case at::kFloat: case at::kBFloat16:
    return true;
  default: return false;
  }
}

static ncclDataType_t kind(at::ScalarType type) {
  switch (type) {
  case at::kChar: return ncclInt8;
  case at::kByte: case at::kBool: return ncclUint8;
  case at::kInt: return ncclInt32;
  case at::kLong: return ncclInt64;
  case at::kHalf: return ncclFloat16;
  case at::kFloat: return ncclFloat32;
  case at::kBFloat16: return ncclBfloat16;
  default: TORCH_CHECK(false, "the mesh backend's Metal path moves no ", type);
  }
}

static ncclMeshBuffer located(const at::Tensor &tensor) {
  return {(__bridge void *)at::native::mps::getMTLBufferStorage(tensor), (size_t)(tensor.storage_offset() * tensor.element_size())};
}

static void checked(int64_t result) {
  TORCH_CHECK(!result, "libnccl-mesh: ", ncclGetErrorString((ncclResult_t)result), ": ", ncclGetLastError(nullptr));
}

// One collective (or one group of NCCL's lowering) on contiguous MPS tensors in one turn of torch's stream queue: its
// command buffer and open encoder, the call encoded into them, and committed where it published.  Split, it is
// issued and not completed: its ticket (0: complete at once), its workspace held until the library retires it.
static uint64_t on_stream(ncclResult_t (^call)(ncclMeshStream *), bool split) {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  __block ncclResult_t result = ncclSuccess;
  __block ncclMeshIssue issue = {0, 0};
  dispatch_sync(stream->queue(), ^{
    stream_ = Stream{(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder(), workspace, nullptr, split ? &issue : nullptr};
    result = call((ncclMeshStream *)&stream_);
    if (!split || issue.published) publishing(stream);
  });
  if (issue.ticket && !result) holders_[issue.ticket] = std::move(held_);
  held_.clear();
  release();
  checked(result);
  return issue.ticket;
}

// every issued group through `ticket` completed on torch's stream where it is waited
static void complete(uint64_t ticket) {
  if (ncclMeshRetired() >= ticket) return release();
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  __block ncclResult_t result = ncclSuccess;
  dispatch_sync(stream->queue(), ^{
    stream_ = Stream{(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder(), workspace, nullptr, nullptr};
    int published = 0;
    result = ncclMeshComplete(ticket, (ncclMeshStream *)&stream_, &published);
    if (published) publishing(stream);
  });
  release();
  checked(result);
}

static void complete_all() { complete(UINT64_MAX); }

// ProcessGroupGloo's future (torch's MPS has one stream and c10's Future carries only its events, so a consumer's
// stream cannot be made to wait at fut.wait() as ProcessGroupNCCL's CUDA futures do): the group's rest runs on this
// extension's own command queue (ncclMeshCompleteApart), after a shared event torch's stream signals at the hand-over
// (across queues an event, not a spin, makes the stream's writes visible), and the future is marked completed, from
// a dispatch queue of this extension's, as that command buffer completes; work submitted after fut.wait() returns
// reads its results.  The ticket's workspace and the outputs are held until then.
static id<MTLCommandQueue> apart_;
static id<MTLSharedEvent> handover_;
static uint64_t handovers_ = 0;
static dispatch_queue_t completions_;

static void complete_apart(uint64_t ticket, c10::intrusive_ptr<c10::ivalue::Future> future, std::vector<at::Tensor> outputs) {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  if (!apart_) {
    apart_ = [stream->device() newCommandQueue];
    handover_ = [stream->device() newSharedEvent];
    completions_ = dispatch_queue_create("torch_mesh.completions", DISPATCH_QUEUE_SERIAL);
  }
  id<MTLCommandBuffer> buffer = [apart_ commandBuffer];
  const uint64_t handover = ++handovers_;
  [buffer encodeWaitForEvent:handover_ value:handover];
  __block ncclResult_t result = ncclSuccess;
  __block int moved = 0;
  dispatch_sync(stream->queue(), ^{
    stream_ = Stream{(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder(), workspace, nullptr, nullptr};
    result = ncclMeshCompleteApart(ticket, (ncclMeshStream *)&stream_, (__bridge void *)buffer, &moved);
    stream->endKernelCoalescing();
    [stream->commandBuffer() encodeSignalEvent:handover_ value:handover];
    committing(stream);
  });
  auto held = std::make_shared<std::vector<at::Tensor>>(outputs);
  for (auto it = holders_.begin(); it != holders_.end() && it->first <= ticket;) {
    for (auto &t : it->second) held->push_back(std::move(t));
    it = holders_.erase(it);
  }
  release();
  if (result) {
    future->setError(std::make_exception_ptr(std::runtime_error(std::string("libnccl-mesh: ") + ncclGetErrorString(result) + ": " + ncclGetLastError(nullptr))));
    return;
  }
  if (!moved) {
    future->markCompleted(c10::IValue(outputs));
    return;
  }
  [buffer addCompletedHandler:^(id<MTLCommandBuffer> done) {
    const bool failed = done.status == MTLCommandBufferStatusError;
    std::string why = failed ? std::string(done.error.localizedDescription.UTF8String ?: "a command buffer failed") : std::string();
    dispatch_async(completions_, ^{
      if (getenv("MESH_TRACE") && *getenv("MESH_TRACE") == '1') fprintf(stderr, "mesh-trace future %llu %s\n", (unsigned long long)ticket, failed ? "failed" : "completed");
      held->clear();
      if (failed)
        future->setError(std::make_exception_ptr(std::runtime_error("the collective's command buffer apart failed: " + why)));
      else
        future->markCompleted(c10::IValue(outputs));
    });
  }];
  [buffer commit];
}

// c10d's Work for a group issued and not completed: waiting (or synchronizing, or asking whether it completed)
// encodes the rest on torch's current stream, where every later kernel reads its results (ProcessGroupNCCL's
// stream-ordered wait); its future completes it apart (complete_apart), and a wait after that waits for the future.
class MeshWork : public c10d::Work {
 public:
  MeshWork(uint64_t ticket, std::vector<at::Tensor> outputs) : c10d::Work(-1, c10d::OpType::UNKNOWN), ticket_(ticket), outputs_(std::move(outputs)) {}
  bool isCompleted() override {
    if (future_) return future_->completed();
    finish();
    return true;
  }
  bool isSuccess() const override { return future_ ? future_->completed() && !future_->hasError() : finished_; }
  bool wait(std::chrono::milliseconds) override {
    if (future_) {
      future_->wait();
      if (future_->hasError()) std::rethrow_exception(future_->exception_ptr());
      return true;
    }
    finish();
    return true;
  }
  void synchronize() override { wait(kNoTimeout); }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    if (future_) return future_;
    future_ = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
    if (finished_ || ncclMeshRetired() >= ticket_) {
      finished_ = true;
      future_->markCompleted(c10::IValue(outputs_));
    } else {
      finished_ = true;
      complete_apart(ticket_, future_, outputs_);
    }
    return future_;
  }

 private:
  void finish() {
    if (finished_) return;
    finished_ = true;
    complete(ticket_);
  }
  uint64_t ticket_;
  std::vector<at::Tensor> outputs_;
  bool finished_ = false;
  c10::intrusive_ptr<c10::ivalue::Future> future_;
};

static c10::intrusive_ptr<c10d::Work> work_of(uint64_t ticket, std::vector<at::Tensor> outputs) {
  if (!ticket) return {};
  return c10::make_intrusive<MeshWork>(ticket, std::move(outputs));
}

static ncclComm_t communicator(int64_t comm) { return (ncclComm_t)(uintptr_t)comm; }

// A contiguous tensor as an operand, with ProcessGroupNCCL's dtype semantics: a collective that moves data moves any
// dtype (one NCCL lacks as its bytes); a reduction reduces a complex tensor as its real view (sum and average) and a bool
// tensor's sum and product as max and min.
struct Operand {
  ncclMeshBuffer buffer;
  size_t count;
  ncclDataType_t type;
};

static Operand moving(const at::Tensor &tensor) {
  if (moved(tensor.scalar_type())) return {located(tensor), (size_t)tensor.numel(), kind(tensor.scalar_type())};
  return {located(tensor), (size_t)tensor.nbytes(), ncclUint8};
}

static Operand reducing(const at::Tensor &tensor, int64_t &op) {
  if (tensor.scalar_type() == at::kBool) {
    TORCH_CHECK(op != ncclAvg, "a bool tensor has no average");
    op = op == ncclSum ? ncclMax : op == ncclProd ? ncclMin : op;
  }
  if (tensor.is_complex()) {
    TORCH_CHECK(op == ncclSum || op == ncclAvg, "a complex tensor reduces by sum and average alone");
    const at::Tensor real = at::view_as_real(tensor);
    return {located(tensor), (size_t)real.numel(), kind(real.scalar_type())};
  }
  return {located(tensor), (size_t)tensor.numel(), kind(tensor.scalar_type())};
}

static uint64_t allreduce(const at::Tensor &tensor, int64_t op, int64_t comm, bool split) {
  const Operand o = reducing(tensor, op);
  return on_stream(^(ncclMeshStream *s) { return ncclAllReduce(&o.buffer, (void *)&o.buffer, o.count, o.type, (ncclRedOp_t)op, communicator(comm), s); }, split);
}

static uint64_t reduce(const at::Tensor &tensor, int64_t op, int64_t root, int64_t comm, bool split) {
  const Operand o = reducing(tensor, op);
  return on_stream(^(ncclMeshStream *s) {
    return ncclReduce(&o.buffer, (void *)&o.buffer, o.count, o.type, (ncclRedOp_t)op, (int)root, communicator(comm), s);
  }, split);
}

static uint64_t broadcast(const at::Tensor &tensor, int64_t root, int64_t comm, bool split) {
  const Operand o = moving(tensor);
  return on_stream(^(ncclMeshStream *s) { return ncclBroadcast(&o.buffer, (void *)&o.buffer, o.count, o.type, (int)root, communicator(comm), s); }, split);
}

static uint64_t allgather(const at::Tensor &output, const at::Tensor &input, int64_t comm, bool split) {
  const Operand from = moving(input), to = moving(output);
  return on_stream(^(ncclMeshStream *s) { return ncclAllGather(&from.buffer, (void *)&to.buffer, from.count, from.type, communicator(comm), s); }, split);
}

static uint64_t reduce_scatter(const at::Tensor &output, const at::Tensor &input, int64_t op, int64_t comm, bool split) {
  const Operand from = reducing(input, op), to = reducing(output, op);
  return on_stream(^(ncclMeshStream *s) {
    return ncclReduceScatter(&from.buffer, (void *)&to.buffer, to.count, from.type, (ncclRedOp_t)op, communicator(comm), s);
  }, split);
}

// all-to-all with each rank's elements (NCCL's own lowering: each rank's sends and receives in one group)
static uint64_t alltoall(const at::Tensor &output, const at::Tensor &input, std::vector<int64_t> sends, std::vector<int64_t> receives,
                         int64_t comm, bool split) {
  const Operand from = moving(input), to = moving(output);
  const size_t per = input.numel() ? from.count / input.numel() : 1, size = input.element_size() / per;
  return on_stream(^(ncclMeshStream *s) {
    ncclResult_t result = ncclGroupStart();
    size_t in = 0, out = 0;
    for (size_t r = 0; r < sends.size() && !result; r++) {
      ncclMeshBuffer send = {from.buffer.buffer, from.buffer.offset + in * per * size}, receive = {to.buffer.buffer, to.buffer.offset + out * per * size};
      result = ncclSend(&send, (size_t)sends[r] * per, from.type, (int)r, communicator(comm), s);
      if (!result) result = ncclRecv(&receive, (size_t)receives[r] * per, from.type, (int)r, communicator(comm), s);
      in += sends[r]; out += receives[r];
    }
    const ncclResult_t ended = ncclGroupEnd();
    return result ? result : ended;
  }, split);
}

static uint64_t send_to(const at::Tensor &tensor, int64_t peer, int64_t comm, bool split) {
  const Operand o = moving(tensor);
  return on_stream(^(ncclMeshStream *s) { return ncclSend(&o.buffer, o.count, o.type, (int)peer, communicator(comm), s); }, split);
}

static uint64_t receive_from(const at::Tensor &tensor, int64_t peer, int64_t comm, bool split) {
  const Operand o = moving(tensor);
  return on_stream(^(ncclMeshStream *s) { return ncclRecv((void *)&o.buffer, o.count, o.type, (int)peer, communicator(comm), s); }, split);
}

// torch's functional collectives (_c10d_functional, what DTensor and a compiled graph call) on MPS tensors: kernels
// for the MPS dispatch key, as torch/csrc/distributed/c10d/Functional.cpp implements them for every device but
// calling libnccl-mesh directly, the group's communicator found by the group (attach) with no Python between.
// A group the map does not hold (another backend's, or a mesh group since destroyed) takes torch's own kernel.
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

static at::Tensor contiguous_alias(const at::Tensor &tensor) { return tensor.is_contiguous() ? tensor : tensor.contiguous(); }

static void written_back(const at::Tensor &target, const at::Tensor &written) {
  if (!written.is_same(target)) target.copy_(written);
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
  // issued and not completed where the result is the operand itself: torch's wait_tensor (its work registry) waits
  // for it; a non-contiguous result, written back from a contiguous operand, completes at once
  at::Tensor result;
  uint64_t ticket = 0;
  if (name == "all_reduce" || name == "all_reduce_") {
    result = name == "all_reduce_" ? input : input.clone(at::MemoryFormat::Contiguous);
    const at::Tensor operand = contiguous_alias(result);
    ticket = allreduce(operand, reduction(arguments[1].toStringRef()), comm, operand.is_same(result));
    written_back(result, operand);
  } else if (name == "broadcast" || name == "broadcast_") {
    result = name == "broadcast_" ? input : input.clone(at::MemoryFormat::Contiguous);
    const at::Tensor operand = contiguous_alias(result);
    ticket = broadcast(operand, arguments[1].toInt(), comm, operand.is_same(result));
    written_back(result, operand);
  } else if (name == "all_gather_into_tensor" || name == "all_gather_into_tensor_out") {
    const at::Tensor from = contiguous_alias(input);
    std::vector<int64_t> shape = from.sizes().vec();
    TORCH_CHECK(!shape.empty(), "all_gather_into_tensor of a 0-dim tensor");
    shape[0] *= arguments[1].toInt();
    result = name == "all_gather_into_tensor_out" ? arguments[3].toTensor() : at::empty(shape, from.options());
    TORCH_CHECK(result.numel() == from.numel() * arguments[1].toInt(), "all_gather_into_tensor_out: out holds ", result.numel(),
                " elements, not ", from.numel() * arguments[1].toInt());
    const at::Tensor to = contiguous_alias(result);
    ticket = allgather(to, from, comm, to.is_same(result));
    written_back(result, to);
  } else if (name == "reduce_scatter_tensor") {
    const at::Tensor from = contiguous_alias(input);
    const int64_t ranks = arguments[2].toInt();
    std::vector<int64_t> shape = from.sizes().vec();
    TORCH_CHECK(!shape.empty() && shape[0] % ranks == 0, "reduce_scatter_tensor: dim 0 of ", from.sizes(), " is not divisible by ", ranks);
    shape[0] /= ranks;
    result = at::empty(shape, from.options());
    ticket = reduce_scatter(result, from, reduction(arguments[1].toStringRef()), comm, true);
  } else {
    const at::Tensor from = contiguous_alias(input);
    std::vector<int64_t> outs, ins;
    for (const auto &v : arguments[1].toListRef()) outs.push_back(v.toInt());
    for (const auto &v : arguments[2].toListRef()) ins.push_back(v.toInt());
    int ranks = 0;
    checked(ncclCommCount(communicator(comm), &ranks));
    TORCH_CHECK(!from.sizes().empty(), "all_to_all_single of a 0-dim tensor");
    if (outs.empty()) outs.assign(ranks, from.size(0) / ranks);
    if (ins.empty()) ins.assign(ranks, from.size(0) / ranks);
    const int64_t width = from.size(0) ? from.numel() / from.size(0) : 1;
    std::vector<int64_t> shape = from.sizes().vec();
    shape[0] = 0;
    for (auto r : outs) shape[0] += r;
    result = at::empty(shape, from.options());
    std::vector<int64_t> sends, receives;
    for (auto r : ins) sends.push_back(r * width);
    for (auto r : outs) receives.push_back(r * width);
    ticket = alltoall(result, from, sends, receives, comm, true);
  }
  if (ticket) c10d::register_work(result, work_of(ticket, {result}));
  torch::jit::drop(*stack, count);
  torch::jit::push(*stack, result);
}

TORCH_LIBRARY_IMPL(_c10d_functional, MPS, m) {
  for (const char *name : {"all_reduce", "all_reduce_", "broadcast", "broadcast_", "all_gather_into_tensor", "all_gather_into_tensor_out",
                           "reduce_scatter_tensor", "all_to_all_single"})
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&functional>());
}

// The ProcessGroup's calls: split (an async op), a MeshWork to wait on, else None (complete).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("begin", &begin);
  module.def("commit", &commit);
  module.def("complete_all", &complete_all);
  module.def("allreduce", [](const at::Tensor &t, int64_t op, int64_t comm, bool split) { return work_of(allreduce(t, op, comm, split), {t}); });
  module.def("reduce", [](const at::Tensor &t, int64_t op, int64_t root, int64_t comm, bool split) { return work_of(reduce(t, op, root, comm, split), {t}); });
  module.def("broadcast", [](const at::Tensor &t, int64_t root, int64_t comm, bool split) { return work_of(broadcast(t, root, comm, split), {t}); });
  module.def("allgather", [](const at::Tensor &o, const at::Tensor &i, int64_t comm, bool split) { return work_of(allgather(o, i, comm, split), {o}); });
  module.def("reduce_scatter", [](const at::Tensor &o, const at::Tensor &i, int64_t op, int64_t comm, bool split) {
    return work_of(reduce_scatter(o, i, op, comm, split), {o});
  });
  module.def("alltoall", [](const at::Tensor &o, const at::Tensor &i, std::vector<int64_t> sends, std::vector<int64_t> receives, int64_t comm, bool split) {
    return work_of(alltoall(o, i, std::move(sends), std::move(receives), comm, split), {o});
  });
  module.def("send", [](const at::Tensor &t, int64_t peer, int64_t comm, bool split) { return work_of(send_to(t, peer, comm, split), {t}); });
  module.def("recv", [](const at::Tensor &t, int64_t peer, int64_t comm, bool split) { return work_of(receive_from(t, peer, comm, split), {t}); });
  module.def("attach", &attach);
  module.def("batch", &batch);
}
