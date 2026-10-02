// torch's current MPS stream as libnccl-mesh's ncclMeshStream (nccl.h): begin() gives the stream's command buffer,
// its open compute encoder (the group's kernels join torch's own) and a workspace allocator (MPS tensors from
// torch's caching allocator, held until commit(), then reused in the stream's order); commit() commits the work
// without waiting.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <vector>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("begin", &begin);
  module.def("commit", &commit);
}
