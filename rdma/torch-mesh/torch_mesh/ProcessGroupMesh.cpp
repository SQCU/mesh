// torch.distributed's "mesh" backend: a c10d::Backend whose collectives are libnccl-mesh's (../../nccl.h),
// as PyTorch's "Customize Process Group Backends Using Cpp Extensions" tutorial prescribes.  Rank 0
// makes the communicator's unique id and hands it out through the group's store.  CPU tensors are the
// buffers themselves; MPS and non-contiguous tensors go through contiguous CPU copies (an MPS tensor's
// copy synchronizes its stream).  Collectives complete before they return; send and recv are enqueued
// on the group's stream (asynchronous, in issue order) so that isend/irecv pairs progress together,
// and their Work waits for it.
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <pybind11/chrono.h>

#include "nccl.h"

namespace c10d {

static void check(ncclResult_t result, ncclComm_t comm, const char *what) {
  TORCH_CHECK(result == ncclSuccess, "mesh: ", what, ": ", ncclGetErrorString(result), ": ", ncclGetLastError(comm));
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

// A tensor's host buffer: the tensor itself when it is a contiguous CPU tensor, else a contiguous CPU
// copy, written back by `back` once the call has written it.
struct Host {
  at::Tensor tensor, host;
  explicit Host(const at::Tensor &t)
      : tensor(t), host(t.device().is_cpu() && t.is_contiguous() ? t : t.to(at::kCPU).contiguous()) {}
  void *data() const { return host.data_ptr(); }
  void back() { if (!host.is_same(tensor)) tensor.copy_(host); }
};

class WorkMesh : public Work {
 public:
  // On a stream: `held` stays alive until the call is done, and is written back to its tensors when
  // `written` (a receive).
  WorkMesh(OpType type, std::vector<at::Tensor> outputs, cudaStream_t stream = nullptr, std::vector<Host> held = {}, bool written = false)
      : Work(-1, type), outputs_(std::move(outputs)), stream_(stream), held_(std::move(held)), written_(written),
        future_(c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()))) {
    if (!stream_) future_->markCompleted(c10::IValue(outputs_));
  }
  bool isCompleted() override { return !stream_; }
  bool isSuccess() const override { return true; }
  bool wait(std::chrono::milliseconds) override {
    if (stream_) {
      check(ncclMeshStreamSynchronize(stream_), nullptr, "the stream");
      if (written_) for (auto &h : held_) h.back();
      held_.clear();
      stream_ = nullptr;
      future_->markCompleted(c10::IValue(outputs_));
    }
    return true;
  }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override { return future_; }

 private:
  std::vector<at::Tensor> outputs_;
  cudaStream_t stream_;
  std::vector<Host> held_;
  bool written_;
  c10::intrusive_ptr<c10::ivalue::Future> future_;
};

class ProcessGroupMesh : public Backend {
 public:
  ProcessGroupMesh(const c10::intrusive_ptr<Store> &store, int rank, int size) : Backend(rank, size) {
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
    check(ncclCommInitRank(&comm_, size, id, rank), nullptr, "ncclCommInitRank");
    check(ncclMeshStreamCreate(&stream_, nullptr), comm_, "ncclMeshStreamCreate");
  }
  ~ProcessGroupMesh() override {
    if (stream_) ncclMeshStreamSynchronize(stream_);
    if (comm_) ncclCommDestroy(comm_);
    if (stream_) ncclMeshStreamDestroy(stream_);
  }
  const std::string getBackendName() const override { return "mesh"; }

  static c10::intrusive_ptr<Backend> create(const c10::intrusive_ptr<Store> &store, int rank, int size,
                                            const std::chrono::duration<float> &) {
    return c10::make_intrusive<ProcessGroupMesh>(store, rank, size);
  }

  c10::intrusive_ptr<Work> broadcast(std::vector<at::Tensor> &tensors, const BroadcastOptions &opts) override {
    std::vector<Host> hosts(tensors.begin(), tensors.end());
    group([&] {
      for (auto &h : hosts) check(ncclBroadcast(h.data(), h.data(), h.host.numel(), datatype(h.host), (int)opts.rootRank, comm_, nullptr), comm_, "ncclBroadcast");
    });
    for (auto &h : hosts) h.back();
    return done(OpType::BROADCAST, tensors);
  }

  c10::intrusive_ptr<Work> allreduce(std::vector<at::Tensor> &tensors, const AllreduceOptions &opts) override {
    std::vector<Host> hosts(tensors.begin(), tensors.end());
    for (auto &h : hosts) reduce_call("ncclAllReduce", opts.reduceOp, h.host, [&](ncclRedOp_t op) {
      return ncclAllReduce(h.data(), h.data(), h.host.numel(), datatype(h.host), op, comm_, nullptr);
    });
    for (auto &h : hosts) h.back();
    return done(OpType::ALLREDUCE, tensors);
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(std::vector<at::Tensor> &tensors, const AllreduceCoalescedOptions &opts) override {
    return allreduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> reduce(std::vector<at::Tensor> &tensors, const ReduceOptions &opts) override {
    std::vector<Host> hosts(tensors.begin(), tensors.end());
    for (auto &h : hosts) reduce_call("ncclReduce", opts.reduceOp, h.host, [&](ncclRedOp_t op) {
      return ncclReduce(h.data(), h.data(), h.host.numel(), datatype(h.host), op, (int)opts.rootRank, comm_, nullptr);
    });
    if (getRank() == opts.rootRank) for (auto &h : hosts) h.back();
    return done(OpType::REDUCE, tensors);
  }

  c10::intrusive_ptr<Work> allgather(std::vector<std::vector<at::Tensor>> &outputs, std::vector<at::Tensor> &inputs,
                                     const AllgatherOptions &) override {
    TORCH_CHECK(inputs.size() == 1 && outputs.size() == 1 && (int)outputs[0].size() == getSize(), "mesh: allgather takes one tensor and a list of world_size");
    Host in(inputs[0]);
    at::Tensor flat = at::empty({getSize() * in.host.numel()}, in.host.options());
    check(ncclAllGather(in.data(), flat.data_ptr(), in.host.numel(), datatype(in.host), comm_, nullptr), comm_, "ncclAllGather");
    for (int r = 0; r < getSize(); r++) outputs[0][r].copy_(flat.narrow(0, r * in.host.numel(), in.host.numel()).view(outputs[0][r].sizes()));
    return done(OpType::ALLGATHER, outputs[0]);
  }

  c10::intrusive_ptr<Work> all_gather_single(at::Tensor &output, at::Tensor &input, const AllgatherOptions &) override {
    Host in(input), out(output);
    TORCH_CHECK(out.host.numel() == getSize() * in.host.numel(), "mesh: all_gather_into_tensor's output is world_size inputs");
    check(ncclAllGather(in.data(), out.data(), in.host.numel(), datatype(in.host), comm_, nullptr), comm_, "ncclAllGather");
    out.back();
    std::vector<at::Tensor> result{output};
    return done(OpType::_ALLGATHER_BASE, result);
  }

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs,
                                                           const AllgatherOptions &opts) override {
    for (size_t i = 0; i < inputs.size(); i++) all_gather_single(outputs[i], inputs[i], opts);
    return done(OpType::COALESCED, outputs);
  }

  c10::intrusive_ptr<Work> gather(std::vector<std::vector<at::Tensor>> &outputs, std::vector<at::Tensor> &inputs,
                                  const GatherOptions &opts) override {
    TORCH_CHECK(inputs.size() == 1, "mesh: gather takes one tensor");
    Host in(inputs[0]);
    const bool root = getRank() == opts.rootRank;
    at::Tensor flat = root ? at::empty({getSize() * in.host.numel()}, in.host.options()) : at::Tensor();
    check(ncclGather(in.data(), root ? flat.data_ptr() : nullptr, in.host.numel(), datatype(in.host), (int)opts.rootRank, comm_, nullptr), comm_, "ncclGather");
    std::vector<at::Tensor> result;
    if (root) {
      TORCH_CHECK(outputs.size() == 1 && (int)outputs[0].size() == getSize(), "mesh: gather's root takes a list of world_size");
      for (int r = 0; r < getSize(); r++) outputs[0][r].copy_(flat.narrow(0, r * in.host.numel(), in.host.numel()).view(outputs[0][r].sizes()));
      result = outputs[0];
    }
    return done(OpType::GATHER, result);
  }

  c10::intrusive_ptr<Work> scatter(std::vector<at::Tensor> &outputs, std::vector<std::vector<at::Tensor>> &inputs,
                                   const ScatterOptions &opts) override {
    TORCH_CHECK(outputs.size() == 1, "mesh: scatter takes one tensor");
    Host out(outputs[0]);
    const bool root = getRank() == opts.rootRank;
    at::Tensor flat;
    if (root) {
      TORCH_CHECK(inputs.size() == 1 && (int)inputs[0].size() == getSize(), "mesh: scatter's root takes a list of world_size");
      std::vector<at::Tensor> parts;
      for (auto &t : inputs[0]) parts.push_back(t.to(at::kCPU).reshape({-1}));
      flat = at::cat(parts);
    }
    check(ncclScatter(root ? flat.data_ptr() : nullptr, out.data(), out.host.numel(), datatype(out.host), (int)opts.rootRank, comm_, nullptr), comm_, "ncclScatter");
    out.back();
    return done(OpType::SCATTER, outputs);
  }

  c10::intrusive_ptr<Work> reduce_scatter(std::vector<at::Tensor> &outputs, std::vector<std::vector<at::Tensor>> &inputs,
                                          const ReduceScatterOptions &opts) override {
    TORCH_CHECK(outputs.size() == 1 && inputs.size() == 1 && (int)inputs[0].size() == getSize(), "mesh: reduce_scatter takes one tensor and a list of world_size");
    Host out(outputs[0]);
    std::vector<at::Tensor> parts;
    for (auto &t : inputs[0]) parts.push_back(t.to(at::kCPU).reshape({-1}));
    at::Tensor flat = at::cat(parts);
    reduce_call("ncclReduceScatter", opts.reduceOp, out.host, [&](ncclRedOp_t op) {
      return ncclReduceScatter(flat.data_ptr(), out.data(), out.host.numel(), datatype(out.host), op, comm_, nullptr);
    });
    out.back();
    return done(OpType::REDUCE_SCATTER, outputs);
  }

  c10::intrusive_ptr<Work> reduce_scatter_single(at::Tensor &output, at::Tensor &input, const ReduceScatterOptions &opts) override {
    Host in(input), out(output);
    TORCH_CHECK(in.host.numel() == getSize() * out.host.numel(), "mesh: reduce_scatter_tensor's input is world_size outputs");
    reduce_call("ncclReduceScatter", opts.reduceOp, out.host, [&](ncclRedOp_t op) {
      return ncclReduceScatter(in.data(), out.data(), out.host.numel(), datatype(out.host), op, comm_, nullptr);
    });
    out.back();
    std::vector<at::Tensor> result{output};
    return done(OpType::_REDUCE_SCATTER_BASE, result);
  }

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs,
                                                           const ReduceScatterOptions &opts) override {
    for (size_t i = 0; i < inputs.size(); i++) reduce_scatter_single(outputs[i], inputs[i], opts);
    return done(OpType::COALESCED, outputs);
  }

  // Equal splits are ncclAlltoAll; others, grouped ncclSend/ncclRecv of each rank's rows.
  c10::intrusive_ptr<Work> all_to_all_single(at::Tensor &output, at::Tensor &input, std::vector<int64_t> &outputSplits,
                                             std::vector<int64_t> &inputSplits, const AllToAllOptions &) override {
    Host in(input), out(output);
    const int n = getSize();
    if (outputSplits.empty() && inputSplits.empty()) {
      TORCH_CHECK(in.host.numel() % n == 0 && in.host.numel() == out.host.numel(), "mesh: all_to_all_single's tensors split evenly");
      check(ncclAlltoAll(in.data(), out.data(), in.host.numel() / n, datatype(in.host), comm_, nullptr), comm_, "ncclAlltoAll");
    } else {
      const int64_t inRow = input.dim() ? in.host.numel() / std::max<int64_t>(input.size(0), 1) : 1;
      const int64_t outRow = output.dim() ? out.host.numel() / std::max<int64_t>(output.size(0), 1) : 1;
      auto splits = [&](std::vector<int64_t> s, const at::Tensor &t) {
        if (s.empty()) s.assign(n, t.size(0) / n);
        return s;
      };
      std::vector<int64_t> ins = splits(inputSplits, input), outs = splits(outputSplits, output);
      const size_t element = in.host.element_size();
      group([&] {
        int64_t at = 0;
        for (int r = 0; r < n; r++) {
          check(ncclSend((char *)in.data() + at * inRow * element, ins[r] * inRow, datatype(in.host), r, comm_, nullptr), comm_, "ncclSend");
          at += ins[r];
        }
        at = 0;
        for (int r = 0; r < n; r++) {
          check(ncclRecv((char *)out.data() + at * outRow * element, outs[r] * outRow, datatype(out.host), r, comm_, nullptr), comm_, "ncclRecv");
          at += outs[r];
        }
      });
    }
    out.back();
    std::vector<at::Tensor> result{output};
    return done(OpType::ALLTOALL_BASE, result);
  }

  c10::intrusive_ptr<Work> alltoall(std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs, const AllToAllOptions &) override {
    TORCH_CHECK((int)outputs.size() == getSize() && (int)inputs.size() == getSize(), "mesh: alltoall takes lists of world_size");
    std::vector<Host> ins(inputs.begin(), inputs.end()), outs(outputs.begin(), outputs.end());
    group([&] {
      for (int r = 0; r < getSize(); r++) {
        check(ncclSend(ins[r].data(), ins[r].host.numel(), datatype(ins[r].host), r, comm_, nullptr), comm_, "ncclSend");
        check(ncclRecv(outs[r].data(), outs[r].host.numel(), datatype(outs[r].host), r, comm_, nullptr), comm_, "ncclRecv");
      }
    });
    for (auto &h : outs) h.back();
    return done(OpType::ALLTOALL, outputs);
  }

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor> &tensors, int dst, int) override {
    std::vector<Host> hosts(tensors.begin(), tensors.end());
    for (auto &h : hosts) check(ncclSend(h.data(), h.host.numel(), datatype(h.host), dst, comm_, stream_), comm_, "ncclSend");
    return c10::make_intrusive<WorkMesh>(OpType::SEND, tensors, stream_, std::move(hosts), false);
  }

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor> &tensors, int src, int) override {
    std::vector<Host> hosts(tensors.begin(), tensors.end());
    for (auto &h : hosts) check(ncclRecv(h.data(), h.host.numel(), datatype(h.host), src, comm_, stream_), comm_, "ncclRecv");
    return c10::make_intrusive<WorkMesh>(OpType::RECV, tensors, stream_, std::move(hosts), true);
  }

  c10::intrusive_ptr<Work> barrier(const BarrierOptions &) override {
    at::Tensor one = at::ones({1}, at::kFloat);
    check(ncclAllReduce(one.data_ptr(), one.data_ptr(), 1, ncclFloat32, ncclSum, comm_, nullptr), comm_, "barrier");
    std::vector<at::Tensor> none;
    return done(OpType::BARRIER, none);
  }

 private:
  ncclComm_t comm_ = nullptr;
  cudaStream_t stream_ = nullptr;

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
  c10::intrusive_ptr<Work> done(OpType type, std::vector<at::Tensor> &outputs) {
    return c10::make_intrusive<WorkMesh>(type, outputs);
  }
};

// MPS tensors: PyTorch registers the c10d operators for CPU (and CUDA) alone, so a collective on an MPS
// tensor reaches no backend.  This kernel of the MPS key runs the operator's CPU kernel (which reaches
// the group's backend for "cpu", this one) on contiguous CPU copies of its MPS tensors, then copies them
// back into the MPS tensors once the call's Work is done: at once for a call already complete, else in
// the Work's wait.
class WorkCopyBack : public Work {
 public:
  WorkCopyBack(c10::intrusive_ptr<Work> inner, std::vector<std::pair<at::Tensor, at::Tensor>> staged)
      : Work(-1, OpType::UNKNOWN), inner_(std::move(inner)), staged_(std::move(staged)) {}
  bool isCompleted() override { return done_; }
  bool isSuccess() const override { return inner_->isSuccess(); }
  bool wait(std::chrono::milliseconds timeout) override {
    bool ok = inner_->wait(timeout);
    back();
    return ok;
  }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override { return inner_->getFuture(); }
  void back() {
    if (done_) return;
    for (auto &p : staged_) p.first.copy_(p.second);
    staged_.clear();
    done_ = true;
  }

 private:
  c10::intrusive_ptr<Work> inner_;
  std::vector<std::pair<at::Tensor, at::Tensor>> staged_;
  bool done_ = false;
};

static void mps_via_cpu(const c10::OperatorHandle &op, torch::jit::Stack *stack) {
  const auto &schema = op.schema();
  std::vector<std::pair<at::Tensor, at::Tensor>> staged;
  auto stage = [&](const at::Tensor &t) {
    if (!t.defined() || !t.is_mps()) return t;
    at::Tensor copy = t.to(at::kCPU).contiguous();
    staged.emplace_back(t, copy);
    return copy;
  };
  auto unstage = [&](const at::Tensor &t) {
    for (auto &p : staged) if (p.second.is_same(t)) return p.first;
    return t;
  };
  auto each = [](c10::IValue &v, const std::function<at::Tensor(const at::Tensor &)> &f) {
    if (v.isTensor()) v = f(v.toTensor());
    else if (v.isTensorList()) {
      auto list = v.toTensorVector();
      for (auto &t : list) t = f(t);
      v = c10::IValue(list);
    } else if (v.isList() && *v.toList().elementType() == *c10::ListType::ofTensors()) {
      c10::impl::GenericList outer(c10::ListType::ofTensors());
      for (const c10::IValue &inner : v.toList()) {
        auto list = inner.toTensorVector();
        for (auto &t : list) t = f(t);
        outer.push_back(c10::IValue(list));
      }
      v = c10::IValue(outer);
    }
  };
  const size_t arguments = schema.arguments().size();
  for (size_t i = stack->size() - arguments; i < stack->size(); i++) each((*stack)[i], stage);
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
  const size_t returns = schema.returns().size();
  c10::intrusive_ptr<Work> work;
  size_t at = 0;
  for (size_t i = stack->size() - returns; i < stack->size(); i++) {
    c10::IValue &v = (*stack)[i];
    if (v.isCustomClass()) { work = v.toCustomClass<Work>(); at = i; }
    else each(v, unstage);
  }
  if (staged.empty()) return;
  auto copy = c10::make_intrusive<WorkCopyBack>(work ? work : c10::intrusive_ptr<Work>(), std::move(staged));
  if (!work || work->isCompleted()) copy->back();
  if (work) (*stack)[at] = c10::IValue(c10::intrusive_ptr<Work>(copy));
}

} // namespace c10d

TORCH_LIBRARY_IMPL(c10d, MPS, m) {
  for (const char *name : {"broadcast_", "allreduce_", "allreduce_coalesced_", "reduce_", "allgather_", "_allgather_base_",
                           "allgather_coalesced_", "allgather_into_tensor_coalesced_", "gather_", "scatter_", "reduce_scatter_",
                           "_reduce_scatter_base_", "reduce_scatter_tensor_coalesced_", "alltoall_", "alltoall_base_",
                           "barrier", "send", "recv_"})
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&c10d::mps_via_cpu>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupMesh", &c10d::ProcessGroupMesh::create);
}
