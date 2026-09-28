// torch.distributed's "mesh" backend: a c10d::Backend whose collectives are libnccl-mesh's (../../nccl.h),
// as PyTorch's "Customize Process Group Backends Using Cpp Extensions" tutorial prescribes.  Rank 0
// makes the communicator's unique id and hands it out through the group's store.  The reductions are
// libnccl-mesh's Metal kernels.
//   CPU tensors are the buffers themselves (libnccl-mesh copies one outside the bridge's registered
// window in and out on the CPU); a collective on them completes before it returns, a send or recv runs
// on a stream of its own (as NCCL's point-to-point calls on separate streams), so isend/irecv pairs
// progress together, and its Work waits for that stream.
//   MPS tensors, through PyTorch 2.14's at::mps API: a tensor whose MTLBuffer (getMTLBufferStorage)
// lies in the window (empty() below) is passed in place; any other is copied by GPU blits on the
// current MPS stream (MPSStream::copy) into a window buffer of the call before it and out after it.
// The call runs on a stream of its own whose shared event orders it with the MPS stream: its input is
// a signal encoded in the MPS command buffer after the blits in, committed without a wait; its Work's
// wait() encodes a wait for the call's completion value and the blits out, so no host synchronization
// sits between the MPS work before and after the call.
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <pybind11/chrono.h>

#include <atomic>
#include <mutex>
#include <unistd.h>
#include <unordered_set>

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

// The backend's own copies (libnccl-mesh counts its own): bytes the CPU copied (a CPU tensor made
// contiguous or its parts laid end to end, and back) and bytes GPU blits copied (an MPS tensor outside
// the window, into and out of the call's window buffer).
static std::atomic<uint64_t> cpu_copied{0}, gpu_copied{0};

// ---- window memory for the GPU ----
// Pages of the bridge's registered window (ncclMemAlloc) as an MTLBuffer, no copy; freed with it.
static id<MTLBuffer> window_buffer(size_t bytes) {
  const size_t page = (size_t)getpagesize(), size = std::max<size_t>(page, (bytes + page - 1) / page * page);
  void *memory = nullptr;
  check(ncclMemAlloc(&memory, size), nullptr, "ncclMemAlloc");
  id<MTLBuffer> buffer = [at::mps::MPSDevice::getInstance()->device() newBufferWithBytesNoCopy:memory length:size
      options:MTLResourceStorageModeShared deallocator:^(void *pointer, NSUInteger) { ncclMemFree(pointer); }];
  if (!buffer) ncclMemFree(memory);
  TORCH_CHECK(buffer, "mesh: window memory as an MTLBuffer (newBufferWithBytesNoCopy)");
  return buffer;
}
// The MTLBuffers over window memory that empty() made tensors of.
static std::mutex windows_lock;
static std::unordered_set<void *> windows;
static bool in_window(id<MTLBuffer> buffer) {
  std::lock_guard<std::mutex> guard(windows_lock);
  return windows.count((void *)buffer) != 0;
}
// An MPS (or CPU) tensor whose bytes are window memory: libnccl-mesh sends and receives it in place.
static at::Tensor empty(std::vector<int64_t> sizes, const at::Tensor &like, const std::string &device) {
  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  id<MTLBuffer> buffer = window_buffer((size_t)numel * like.element_size());
  auto options = at::TensorOptions().dtype(like.scalar_type());
  if (device == "cpu")
    return at::from_blob([buffer contents], sizes, [buffer](void *) { [buffer release]; }, options.device(at::kCPU));
  TORCH_CHECK(device == "mps", "mesh: empty() makes cpu or mps tensors, not ", device);
  {
    std::lock_guard<std::mutex> guard(windows_lock);
    windows.insert((void *)buffer);
  }
  // released when the tensor's storage is; Metal holds it while a committed command buffer uses it
  return at::from_blob((void *)buffer, sizes, [buffer](void *) {
    { std::lock_guard<std::mutex> guard(windows_lock); windows.erase((void *)buffer); }
    [buffer release];
  }, options.device(at::kMPS));
}

// Streams for calls that run asynchronously (a send or recv on CPU tensors, every call on MPS tensors):
// one a call, reused once the Work holding it is gone and its work done.
struct StreamPool {
  std::mutex lock;
  std::vector<cudaStream_t> all, idle;
  cudaStream_t acquire() {
    std::lock_guard<std::mutex> guard(lock);
    for (size_t i = 0; i < idle.size(); i++)
      if (ncclMeshStreamQuery(idle[i]) == ncclSuccess) {
        cudaStream_t s = idle[i];
        idle.erase(idle.begin() + (long)i);
        return s;
      }
    cudaStream_t s;
    check(ncclMeshStreamCreate(&s, nullptr), nullptr, "ncclMeshStreamCreate");
    all.push_back(s);
    return s;
  }
  void release(cudaStream_t s) {
    std::lock_guard<std::mutex> guard(lock);
    idle.push_back(s);
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
// is tensors laid end to end (one tensor, or a list): in place where they already are (CPU: adjacent and
// contiguous; MPS: adjacent in one window MTLBuffer), else a contiguous CPU copy, or a slice of the call's
// window buffer with GPU blits.
class Call {
 public:
  Call(const at::Tensor &like, bool async, std::shared_ptr<StreamPool> pool)
      : mps_(like.is_mps()), async_(async || like.is_mps()), pool_(std::move(pool)) {}
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
  void *ptr(int i) const { return places_[i].pointer; }
  cudaStream_t stream() const { return stream_; }

  // Every place's pointer; on MPS the blits in and the stream's input signal, committed.
  void begin() {
    if (!mps_) {
      for (auto &p : places_) cpu_place(p);
      if (async_) stream_ = pool_->acquire();
      return;
    }
    size_t used = 0;
    for (auto &p : places_)
      if (!window_place(p)) { p.at = used; used += (p.bytes + 255) & ~(size_t)255; }
    if (used) scratch_ = window_buffer(used);
    auto *s = at::mps::getCurrentMPSStream();
    for (auto &p : places_) {
      if (p.at == SIZE_MAX) continue;
      p.pointer = (char *)[scratch_ contents] + p.at;
      if (p.in) blits(s, p, true);
    }
    stream_ = pool_->acquire();
    const uint64_t ready = ++stream_->value;
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)stream_->event;
    dispatch_sync(s->queue(), ^{
      s->endKernelCoalescing();
      [s->commandBuffer() encodeSignalEvent:event value:ready];
      s->synchronize(at::mps::SyncType::COMMIT);
    });
  }
  // Once the call is complete (CPU) or its completion is ordered on the current MPS stream: the places
  // written copied back.
  void after() {
    if (!mps_) {
      for (auto &p : places_)
        if (p.out && p.host.defined()) {
          int64_t at = 0;
          for (auto &t : p.parts) {
            t.copy_(p.host.narrow(0, at, t.numel()).view(t.sizes()));
            at += t.numel();
          }
          cpu_copied += p.bytes;
        }
      return;
    }
    if (!stream_) return;
    auto *s = at::mps::getCurrentMPSStream();
    const uint64_t done = stream_->value;
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)stream_->event;
    dispatch_sync(s->queue(), ^{
      s->endKernelCoalescing();
      [s->commandBuffer() encodeWaitForEvent:event value:done];
    });
    for (auto &p : places_)
      if (p.at != SIZE_MAX && p.out) blits(s, p, false);
    if (scratch_) {
      id<MTLBuffer> held = scratch_;
      scratch_ = nil;
      s->addCompletedHandler(^(id<MTLCommandBuffer>) { [held release]; });
    }
  }
  ~Call() { [scratch_ release]; }

 private:
  struct Place {
    std::vector<at::Tensor> parts;
    bool in = false, out = false;
    size_t bytes = 0, at = SIZE_MAX;
    void *pointer = nullptr;
    at::Tensor host;
  };
  // CPU: the parts themselves where they are contiguous and adjacent, else one contiguous copy.
  void cpu_place(Place &p) {
    char *end = nullptr;
    bool adjacent = true;
    for (auto &t : p.parts) {
      adjacent &= t.is_contiguous() && (!end || (char *)t.data_ptr() == end);
      end = (char *)t.data_ptr() + t.nbytes();
    }
    if (adjacent) { p.pointer = p.parts[0].data_ptr(); return; }
    std::vector<at::Tensor> flat;
    for (auto &t : p.parts) flat.push_back(t.reshape({-1}));
    p.host = p.in ? at::cat(flat) : at::empty({(int64_t)(p.bytes / p.parts[0].element_size())}, p.parts[0].options());
    if (p.in) cpu_copied += p.bytes;
    p.pointer = p.host.data_ptr();
  }
  // MPS: the parts in place where they are contiguous and adjacent in one window MTLBuffer.
  bool window_place(Place &p) {
    id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(p.parts[0]);
    size_t end = SIZE_MAX;
    for (auto &t : p.parts) {
      size_t offset = t.storage_offset() * t.element_size();
      if (!t.is_contiguous() || at::native::mps::getMTLBufferStorage(t) != buffer || (end != SIZE_MAX && offset != end)) return false;
      end = offset + t.nbytes();
    }
    if (!in_window(buffer)) return false;
    p.pointer = (char *)[buffer contents] + p.parts[0].storage_offset() * p.parts[0].element_size();
    return true;
  }
  // Each part's bytes between its MTLBuffer and the call's window buffer, blitted on the MPS stream (a
  // non-contiguous part through a contiguous MPS copy).
  void blits(at::mps::MPSStream *s, Place &p, bool in) {
    size_t at = p.at;
    for (auto &t : p.parts) {
      at::Tensor c = t.is_contiguous() ? t : in ? t.contiguous() : at::empty(t.sizes(), t.options());
      if (in && !c.is_same(t)) gpu_copied += c.nbytes();
      id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(c);
      size_t offset = c.storage_offset() * c.element_size();
      if (in) s->copy(buffer, scratch_, c.nbytes(), offset, at, 0, at::mps::SyncType::NONE);
      else s->copy(scratch_, buffer, c.nbytes(), at, offset, 0, at::mps::SyncType::NONE);
      if (!in && !c.is_same(t)) { t.copy_(c); gpu_copied += c.nbytes(); }
      gpu_copied += c.nbytes();
      at += c.nbytes();
    }
  }
  bool mps_, async_;
  std::shared_ptr<StreamPool> pool_;
  cudaStream_t stream_ = nullptr;
  std::vector<Place> places_;
  id<MTLBuffer> scratch_ = nil;
  friend class WorkMesh;
};

// A call's Work: done at once for a CPU collective; else done once its stream reaches the call's value,
// wait() waiting on the host (CPU) or ordering the current MPS stream after it (MPS), then copying back.
class WorkMesh : public Work {
 public:
  WorkMesh(OpType type, std::vector<at::Tensor> outputs, std::shared_ptr<Call> call)
      : Work(-1, type), outputs_(std::move(outputs)), call_(std::move(call)),
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
  }
  ~ProcessGroupMesh() override {
    streams_->synchronize();
    if (comm_) ncclCommDestroy(comm_);
  }
  const std::string getBackendName() const override { return "mesh"; }

  static c10::intrusive_ptr<Backend> create(const c10::intrusive_ptr<Store> &store, int rank, int size,
                                            const std::chrono::duration<float> &) {
    return c10::make_intrusive<ProcessGroupMesh>(store, rank, size);
  }

  c10::intrusive_ptr<Work> broadcast(std::vector<at::Tensor> &tensors, const BroadcastOptions &opts) override {
    auto call = start(tensors[0]);
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
    auto call = start(tensors[0]);
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
    auto call = start(tensors[0]);
    std::vector<int> at;
    for (auto &t : tensors) at.push_back(call->add(t, true, getRank() == opts.rootRank));
    call->begin();
    for (size_t i = 0; i < tensors.size(); i++) reduce_call("ncclReduce", opts.reduceOp, tensors[i], [&](ncclRedOp_t op) {
      return ncclReduce(call->ptr(at[i]), call->ptr(at[i]), tensors[i].numel(), datatype(tensors[i]), op, (int)opts.rootRank, comm_,
                        call->stream());
    });
    return work(OpType::REDUCE, tensors, call);
  }

  c10::intrusive_ptr<Work> allgather(std::vector<std::vector<at::Tensor>> &outputs, std::vector<at::Tensor> &inputs,
                                     const AllgatherOptions &) override {
    TORCH_CHECK(inputs.size() == 1 && outputs.size() == 1 && (int)outputs[0].size() == getSize(), "mesh: allgather takes one tensor and a list of world_size");
    auto call = start(inputs[0]);
    int in = call->add(inputs[0], true, false), out = call->add(outputs[0], false, true);
    call->begin();
    check(ncclAllGather(call->ptr(in), call->ptr(out), inputs[0].numel(), datatype(inputs[0]), comm_, call->stream()), comm_, "ncclAllGather");
    return work(OpType::ALLGATHER, outputs[0], call);
  }

  c10::intrusive_ptr<Work> all_gather_single(at::Tensor &output, at::Tensor &input, const AllgatherOptions &) override {
    TORCH_CHECK(output.numel() == getSize() * input.numel(), "mesh: all_gather_into_tensor's output is world_size inputs");
    auto call = start(input);
    int in = call->add(input, true, false), out = call->add(output, false, true);
    call->begin();
    check(ncclAllGather(call->ptr(in), call->ptr(out), input.numel(), datatype(input), comm_, call->stream()), comm_, "ncclAllGather");
    std::vector<at::Tensor> result{output};
    return work(OpType::_ALLGATHER_BASE, result, call);
  }

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs,
                                                           const AllgatherOptions &opts) override {
    for (size_t i = 0; i < inputs.size(); i++) all_gather_single(outputs[i], inputs[i], opts)->wait();
    return c10::make_intrusive<WorkMesh>(OpType::COALESCED, outputs, std::make_shared<Call>(inputs[0], false, streams_));
  }

  c10::intrusive_ptr<Work> gather(std::vector<std::vector<at::Tensor>> &outputs, std::vector<at::Tensor> &inputs,
                                  const GatherOptions &opts) override {
    TORCH_CHECK(inputs.size() == 1, "mesh: gather takes one tensor");
    const bool root = getRank() == opts.rootRank;
    TORCH_CHECK(!root || (outputs.size() == 1 && (int)outputs[0].size() == getSize()), "mesh: gather's root takes a list of world_size");
    auto call = start(inputs[0]);
    int in = call->add(inputs[0], true, false), out = root ? call->add(outputs[0], false, true) : -1;
    call->begin();
    check(ncclGather(call->ptr(in), root ? call->ptr(out) : nullptr, inputs[0].numel(), datatype(inputs[0]), (int)opts.rootRank, comm_,
                     call->stream()), comm_, "ncclGather");
    std::vector<at::Tensor> result = root ? outputs[0] : std::vector<at::Tensor>{};
    return work(OpType::GATHER, result, call);
  }

  c10::intrusive_ptr<Work> scatter(std::vector<at::Tensor> &outputs, std::vector<std::vector<at::Tensor>> &inputs,
                                   const ScatterOptions &opts) override {
    TORCH_CHECK(outputs.size() == 1, "mesh: scatter takes one tensor");
    const bool root = getRank() == opts.rootRank;
    TORCH_CHECK(!root || (inputs.size() == 1 && (int)inputs[0].size() == getSize()), "mesh: scatter's root takes a list of world_size");
    auto call = start(outputs[0]);
    int out = call->add(outputs[0], false, true), in = root ? call->add(inputs[0], true, false) : -1;
    call->begin();
    check(ncclScatter(root ? call->ptr(in) : nullptr, call->ptr(out), outputs[0].numel(), datatype(outputs[0]), (int)opts.rootRank, comm_,
                      call->stream()), comm_, "ncclScatter");
    return work(OpType::SCATTER, outputs, call);
  }

  c10::intrusive_ptr<Work> reduce_scatter(std::vector<at::Tensor> &outputs, std::vector<std::vector<at::Tensor>> &inputs,
                                          const ReduceScatterOptions &opts) override {
    TORCH_CHECK(outputs.size() == 1 && inputs.size() == 1 && (int)inputs[0].size() == getSize(), "mesh: reduce_scatter takes one tensor and a list of world_size");
    auto call = start(outputs[0]);
    int out = call->add(outputs[0], false, true), in = call->add(inputs[0], true, false);
    call->begin();
    reduce_call("ncclReduceScatter", opts.reduceOp, outputs[0], [&](ncclRedOp_t op) {
      return ncclReduceScatter(call->ptr(in), call->ptr(out), outputs[0].numel(), datatype(outputs[0]), op, comm_, call->stream());
    });
    return work(OpType::REDUCE_SCATTER, outputs, call);
  }

  c10::intrusive_ptr<Work> reduce_scatter_single(at::Tensor &output, at::Tensor &input, const ReduceScatterOptions &opts) override {
    TORCH_CHECK(input.numel() == getSize() * output.numel(), "mesh: reduce_scatter_tensor's input is world_size outputs");
    auto call = start(output);
    int out = call->add(output, false, true), in = call->add(input, true, false);
    call->begin();
    reduce_call("ncclReduceScatter", opts.reduceOp, output, [&](ncclRedOp_t op) {
      return ncclReduceScatter(call->ptr(in), call->ptr(out), output.numel(), datatype(output), op, comm_, call->stream());
    });
    std::vector<at::Tensor> result{output};
    return work(OpType::_REDUCE_SCATTER_BASE, result, call);
  }

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs,
                                                           const ReduceScatterOptions &opts) override {
    for (size_t i = 0; i < inputs.size(); i++) reduce_scatter_single(outputs[i], inputs[i], opts)->wait();
    return c10::make_intrusive<WorkMesh>(OpType::COALESCED, outputs, std::make_shared<Call>(inputs[0], false, streams_));
  }

  // Equal splits are ncclAlltoAll; others, grouped ncclSend/ncclRecv of each rank's rows.
  c10::intrusive_ptr<Work> all_to_all_single(at::Tensor &output, at::Tensor &input, std::vector<int64_t> &outputSplits,
                                             std::vector<int64_t> &inputSplits, const AllToAllOptions &) override {
    const int n = getSize();
    auto call = start(input);
    int in = call->add(input, true, false), out = call->add(output, false, true);
    call->begin();
    if (outputSplits.empty() && inputSplits.empty()) {
      TORCH_CHECK(input.numel() % n == 0 && input.numel() == output.numel(), "mesh: all_to_all_single's tensors split evenly");
      check(ncclAlltoAll(call->ptr(in), call->ptr(out), input.numel() / n, datatype(input), comm_, call->stream()), comm_, "ncclAlltoAll");
    } else {
      const int64_t inRow = input.dim() ? input.numel() / std::max<int64_t>(input.size(0), 1) : 1;
      const int64_t outRow = output.dim() ? output.numel() / std::max<int64_t>(output.size(0), 1) : 1;
      auto splits = [&](std::vector<int64_t> s, const at::Tensor &t) {
        if (s.empty()) s.assign(n, t.size(0) / n);
        return s;
      };
      std::vector<int64_t> ins = splits(inputSplits, input), outs = splits(outputSplits, output);
      const size_t element = input.element_size();
      group([&] {
        int64_t at = 0;
        for (int r = 0; r < n; r++) {
          check(ncclSend((char *)call->ptr(in) + at * inRow * element, ins[r] * inRow, datatype(input), r, comm_, call->stream()), comm_, "ncclSend");
          at += ins[r];
        }
        at = 0;
        for (int r = 0; r < n; r++) {
          check(ncclRecv((char *)call->ptr(out) + at * outRow * element, outs[r] * outRow, datatype(output), r, comm_, call->stream()), comm_, "ncclRecv");
          at += outs[r];
        }
      });
    }
    std::vector<at::Tensor> result{output};
    return work(OpType::ALLTOALL_BASE, result, call);
  }

  c10::intrusive_ptr<Work> alltoall(std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs, const AllToAllOptions &) override {
    TORCH_CHECK((int)outputs.size() == getSize() && (int)inputs.size() == getSize(), "mesh: alltoall takes lists of world_size");
    auto call = start(inputs[0]);
    std::vector<int> ins, outs;
    for (int r = 0; r < getSize(); r++) {
      ins.push_back(call->add(inputs[r], true, false));
      outs.push_back(call->add(outputs[r], false, true));
    }
    call->begin();
    group([&] {
      for (int r = 0; r < getSize(); r++) {
        check(ncclSend(call->ptr(ins[r]), inputs[r].numel(), datatype(inputs[r]), r, comm_, call->stream()), comm_, "ncclSend");
        check(ncclRecv(call->ptr(outs[r]), outputs[r].numel(), datatype(outputs[r]), r, comm_, call->stream()), comm_, "ncclRecv");
      }
    });
    return work(OpType::ALLTOALL, outputs, call);
  }

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor> &tensors, int dst, int) override {
    auto call = std::make_shared<Call>(tensors[0], true, streams_);
    std::vector<int> at;
    for (auto &t : tensors) at.push_back(call->add(t, true, false));
    call->begin();
    for (size_t i = 0; i < tensors.size(); i++)
      check(ncclSend(call->ptr(at[i]), tensors[i].numel(), datatype(tensors[i]), dst, comm_, call->stream()), comm_, "ncclSend");
    return work(OpType::SEND, tensors, call);
  }

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor> &tensors, int src, int) override {
    auto call = std::make_shared<Call>(tensors[0], true, streams_);
    std::vector<int> at;
    for (auto &t : tensors) at.push_back(call->add(t, false, true));
    call->begin();
    for (size_t i = 0; i < tensors.size(); i++)
      check(ncclRecv(call->ptr(at[i]), tensors[i].numel(), datatype(tensors[i]), src, comm_, call->stream()), comm_, "ncclRecv");
    return work(OpType::RECV, tensors, call);
  }

  c10::intrusive_ptr<Work> barrier(const BarrierOptions &) override {
    at::Tensor one = at::ones({1}, at::kFloat);
    check(ncclAllReduce(one.data_ptr(), one.data_ptr(), 1, ncclFloat32, ncclSum, comm_, nullptr), comm_, "barrier");
    std::vector<at::Tensor> none;
    return c10::make_intrusive<WorkMesh>(OpType::BARRIER, none, std::make_shared<Call>(one, false, streams_));
  }

 private:
  ncclComm_t comm_ = nullptr;
  std::shared_ptr<StreamPool> streams_ = std::make_shared<StreamPool>();

  // A collective's call: synchronous on CPU tensors, on a stream of its own on MPS tensors.
  std::shared_ptr<Call> start(const at::Tensor &like) { return std::make_shared<Call>(like, false, streams_); }
  c10::intrusive_ptr<Work> work(OpType type, std::vector<at::Tensor> &outputs, std::shared_ptr<Call> call) {
    return c10::make_intrusive<WorkMesh>(type, outputs, std::move(call));
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

// What libnccl-mesh and this backend copied and waited for so far (counted, not timed).
static pybind11::dict counts() {
  ncclMeshCounts_t c;
  check(ncclMeshGetCounts(&c), nullptr, "ncclMeshGetCounts");
  pybind11::dict d;
  d["library_cpu_copy_bytes"] = c.cpuCopyBytes;
  d["library_gpu_copy_bytes"] = c.gpuCopyBytes;
  d["library_gpu_kernels"] = c.gpuKernels;
  d["library_host_waits"] = c.hostWaits;
  d["library_input_waits"] = c.inputWaits;
  d["backend_cpu_copy_bytes"] = cpu_copied.load();
  d["backend_gpu_copy_bytes"] = gpu_copied.load();
  return d;
}

} // namespace c10d

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
}
