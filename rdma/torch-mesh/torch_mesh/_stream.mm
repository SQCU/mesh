// torch's current MPS stream as libnccl-mesh's ncclMeshStream (nccl.h): begin() gives the stream's command buffer,
// its open compute encoder (the group's kernels join torch's own) and a workspace allocator (MPS tensors from
// torch's caching allocator, held until commit(), then reused in the stream's order); commit() commits the work
// without waiting.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <vector>
#include "nccl.h"

struct Stream {
  void *commandBuffer, *commandEncoder;
  void *(*workspace)(size_t, void *);
  void *context;
};

static Stream stream_;
static std::vector<at::Tensor> held_;

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
  stream_ = Stream{buffer, encoder, workspace, nullptr};
  return (int64_t)(uintptr_t)&stream_;
}

static void commit() {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{ stream->synchronize(at::mps::SyncType::COMMIT); });
  held_.clear();
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

// One collective (or one group of NCCL's lowering) on contiguous MPS tensors in one turn of torch's stream queue: its command buffer and open encoder,
// the call encoded into them, the work committed.
static int64_t on_stream(ncclResult_t (^call)(ncclMeshStream *)) {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  __block ncclResult_t result = ncclSuccess;
  dispatch_sync(stream->queue(), ^{
    stream_ = Stream{(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder(), workspace, nullptr};
    result = call((ncclMeshStream *)&stream_);
    stream->synchronize(at::mps::SyncType::COMMIT);
  });
  held_.clear();
  return result;
}

static ncclComm_t communicator(int64_t comm) { return (ncclComm_t)(uintptr_t)comm; }

static int64_t allreduce(const at::Tensor &tensor, int64_t op, int64_t comm) {
  const ncclMeshBuffer buffer = located(tensor);
  const size_t count = tensor.numel();
  const ncclDataType_t type = kind(tensor.scalar_type());
  return on_stream(^(ncclMeshStream *s) { return ncclAllReduce(&buffer, (void *)&buffer, count, type, (ncclRedOp_t)op, communicator(comm), s); });
}

static int64_t reduce(const at::Tensor &tensor, int64_t op, int64_t root, int64_t comm) {
  const ncclMeshBuffer buffer = located(tensor);
  const size_t count = tensor.numel();
  const ncclDataType_t type = kind(tensor.scalar_type());
  return on_stream(^(ncclMeshStream *s) { return ncclReduce(&buffer, (void *)&buffer, count, type, (ncclRedOp_t)op, (int)root, communicator(comm), s); });
}

static int64_t broadcast(const at::Tensor &tensor, int64_t root, int64_t comm) {
  const ncclMeshBuffer buffer = located(tensor);
  const size_t count = tensor.numel();
  const ncclDataType_t type = kind(tensor.scalar_type());
  return on_stream(^(ncclMeshStream *s) { return ncclBroadcast(&buffer, (void *)&buffer, count, type, (int)root, communicator(comm), s); });
}

static int64_t allgather(const at::Tensor &output, const at::Tensor &input, int64_t comm) {
  const ncclMeshBuffer from = located(input), to = located(output);
  const size_t count = input.numel();
  const ncclDataType_t type = kind(input.scalar_type());
  return on_stream(^(ncclMeshStream *s) { return ncclAllGather(&from, (void *)&to, count, type, communicator(comm), s); });
}

static int64_t reduce_scatter(const at::Tensor &output, const at::Tensor &input, int64_t op, int64_t comm) {
  const ncclMeshBuffer from = located(input), to = located(output);
  const size_t count = output.numel();
  const ncclDataType_t type = kind(input.scalar_type());
  return on_stream(^(ncclMeshStream *s) { return ncclReduceScatter(&from, (void *)&to, count, type, (ncclRedOp_t)op, communicator(comm), s); });
}

// all-to-all with each rank's elements (NCCL's own lowering: each rank's sends and receives in one group)
static int64_t alltoall(const at::Tensor &output, const at::Tensor &input, std::vector<int64_t> sends, std::vector<int64_t> receives,
                        int64_t comm) {
  const ncclMeshBuffer from = located(input), to = located(output);
  const ncclDataType_t type = kind(input.scalar_type());
  const size_t size = input.element_size();
  return on_stream(^(ncclMeshStream *s) {
    ncclResult_t result = ncclGroupStart();
    size_t in = 0, out = 0;
    for (size_t r = 0; r < sends.size() && !result; r++) {
      ncclMeshBuffer send = {from.buffer, from.offset + in * size}, receive = {to.buffer, to.offset + out * size};
      result = ncclSend(&send, (size_t)sends[r], type, (int)r, communicator(comm), s);
      if (!result) result = ncclRecv(&receive, (size_t)receives[r], type, (int)r, communicator(comm), s);
      in += sends[r]; out += receives[r];
    }
    const ncclResult_t ended = ncclGroupEnd();
    return result ? result : ended;
  });
}

static int64_t send(const at::Tensor &tensor, int64_t peer, int64_t comm) {
  const ncclMeshBuffer buffer = located(tensor);
  const size_t count = tensor.numel();
  const ncclDataType_t type = kind(tensor.scalar_type());
  return on_stream(^(ncclMeshStream *s) { return ncclSend(&buffer, count, type, (int)peer, communicator(comm), s); });
}

static int64_t recv(const at::Tensor &tensor, int64_t peer, int64_t comm) {
  const ncclMeshBuffer buffer = located(tensor);
  const size_t count = tensor.numel();
  const ncclDataType_t type = kind(tensor.scalar_type());
  return on_stream(^(ncclMeshStream *s) { return ncclRecv((void *)&buffer, count, type, (int)peer, communicator(comm), s); });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("begin", &begin);
  module.def("commit", &commit);
  module.def("allreduce", &allreduce);
  module.def("reduce", &reduce);
  module.def("broadcast", &broadcast);
  module.def("allgather", &allgather);
  module.def("reduce_scatter", &reduce_scatter);
  module.def("alltoall", &alltoall);
  module.def("send", &send);
  module.def("recv", &recv);
}
