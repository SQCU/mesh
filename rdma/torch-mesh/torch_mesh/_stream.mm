// torch's current MPS stream for libnccl-mesh's Metal path (nccl.h ncclMeshStream): begin() closes the stream's
// open encoder and returns its command buffer for a group's work; commit() commits that work without waiting.
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>

static int64_t begin() {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  __block id buffer = nil;
  dispatch_sync(stream->queue(), ^{
    stream->endKernelCoalescing();
    buffer = stream->commandBuffer();
  });
  return (int64_t)(uintptr_t)buffer;
}

static void commit() {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{ stream->synchronize(at::mps::SyncType::COMMIT); });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("begin", &begin);
  module.def("commit", &commit);
}
