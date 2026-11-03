// torch_mesh's binding (rdma/metal.h): region tensors, the stream's Metal 4 queue, compiled pipelines, the run, the run's
// solve, and the communicator a process group names. Everything the GPU does is a node of a run.
#include <torch/extension.h>
#include <torch/library.h>
#include <c10/core/CPUAllocator.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSEvent.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>
#include "metal.h"
#include "mesh-graph.h"
#include "mesh-page-map.h"
#include "nccl.h"

void install_region_allocator();

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


static void checked(int64_t result) {
  TORCH_CHECK(!result, "libnccl-mesh: ", ncclGetErrorString((ncclResult_t)result), ": ", ncclGetLastError(nullptr));
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

// metal.h 2: each compiled pipeline, retained for the process, by its handle
static std::vector<id<MTLComputePipelineState>> *pipelines_ = new std::vector<id<MTLComputePipelineState>>;

// `function` of `source`, specialized by `constants` ((index, MTLDataType, bytes) each); its handle
static int64_t pipeline(const std::string &source, const std::string &function,
                        const std::vector<std::tuple<uint32_t, uint32_t, pybind11::bytes>> &constants) {
  NSMutableArray *values = [NSMutableArray array];
  for (const auto &[index, type, bytes] : constants) {
    const std::string raw = bytes;
    [values addObject:@[@(index), @(type), [NSData dataWithBytes:raw.data() length:raw.size()]]];
  }
  NSError *error = nil;
  id<MTLComputePipelineState> made = MetalPipeline([NSString stringWithUTF8String:source.c_str()],
                                                   [NSString stringWithUTF8String:function.c_str()], values, &error);
  TORCH_CHECK(made, "compiling ", function, ": ", error ? error.localizedDescription.UTF8String : "no pipeline");
  pipelines_->push_back(made);
  return (int64_t)pipelines_->size() - 1;
}

// A run binding: a region tensor (its storage's buffer, from its first element and `offset` more bytes) or a word's
// address (the lease or stream page holding it: nccl.h ncclMeshMemBuffer)
static MetalRunBinding bound(const pybind11::handle &operand, uint32_t index, uint64_t offset,
                             std::vector<at::Tensor> &held) {
  if (THPVariable_Check(operand.ptr())) { // torch's caster is no registered pybind type: isinstance<at::Tensor> is false
    const at::Tensor t = THPVariable_Unpack(operand.ptr());
    TORCH_CHECK(t.is_mps(), "a run binding is an MPS region tensor");
    held.push_back(t);
    return {at::native::mps::getMTLBufferStorage(t), (NSUInteger)(t.storage_offset() * t.element_size() + offset), index};
  }
  void *buffer = nullptr;
  size_t at = 0;
  checked(ncclMeshMemBuffer((void *)(uintptr_t)operand.cast<uint64_t>(), &buffer, &at));
  return {(__bridge id<MTLBuffer>)buffer, (NSUInteger)(at + offset), index};
}

// metal.h 3: a run on torch's stream's Metal 4 queue
struct Run {
  std::shared_ptr<MetalRun> run;
  std::vector<at::Tensor> held;
  Run() : run(MetalRunCreate(stream_of(at::mps::getCurrentMPSStream()).prepared), MetalRunFree) {}
  void event(uint64_t event) { MetalRunEvent(run.get(), event); }
  // operands: (argument index, tensor or address, byte offset) each
  void dispatch(int64_t handle, const std::vector<std::tuple<uint32_t, pybind11::object, uint64_t>> &operands,
                std::array<NSUInteger, 3> grid, std::array<NSUInteger, 3> group, bool threads) {
    TORCH_CHECK(handle >= 0 && (size_t)handle < pipelines_->size(), "no such pipeline");
    std::vector<MetalRunBinding> bindings;
    for (const auto &[index, operand, offset] : operands) bindings.push_back(bound(operand, index, offset, held));
    MetalRunDispatch(run.get(), (*pipelines_)[handle], bindings.data(), bindings.size(),
                     MTLSizeMake(grid[0], grid[1], grid[2]), MTLSizeMake(group[0], group[1], group[2]), threads);
  }
  // a copy of `bytes` from `source` to `destination`, each a region tensor or a word's address (a binding's operand),
  // `from` and `to` bytes past it
  void copy(const pybind11::object &source, uint64_t from, const pybind11::object &destination, uint64_t to, uint64_t bytes) {
    const MetalRunBinding read = bound(source, 0, from, held), written = bound(destination, 0, to, held);
    MetalRunCopy(run.get(), read.buffer, read.offset, written.buffer, written.offset, bytes);
  }
  void landed(uint64_t word, uint64_t threshold) { MetalRunLanded(run.get(), (const void *)(uintptr_t)word, threshold); }
  // torch's work so far, an outside producer of this event's pages
  void inputs() {
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    MeshStream *source = &stream_of(stream);
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{ source->signalled(false); });
    MetalRunInput(run.get(), source->progress, source->committed);
  }
  // torch's work so far, again: the input's value set to its progress now (a repeat's reset after preparation)
  void inputs_now() {
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    MeshStream *source = &stream_of(stream);
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{ source->signalled(false); });
    MetalRunInputValue(run.get(), source->committed);
  }
  void edges(const std::vector<std::pair<uint64_t, uint64_t>> &pairs) {
    std::vector<uint64_t> flat;
    for (const auto &[a, b] : pairs) { flat.push_back(a); flat.push_back(b); }
    MetalRunEdges(run.get(), reinterpret_cast<const uint64_t (*)[2]>(flat.data()), pairs.size());
  }
  void communicator(int64_t comm) { MetalRunCommunicator(run.get(), (struct ncclComm *)(uintptr_t)comm); }
  void finish() {
    const char *defect = MetalRunFinish(run.get());
    TORCH_CHECK(!defect, "preparing the run: ", defect ? defect : "");
  }
  void enqueue() { MetalRunEnqueue(run.get()); }
  // what the run still waits on: (pending parts and host tasks, [(word, value now, threshold)] of each landing not present)
  std::pair<long, std::vector<std::array<uint64_t, 3>>> waiting() {
    long parts = 0;
    const size_t count = MetalRunWaiting(run.get(), nullptr, 0, &parts);
    std::vector<std::array<uint64_t, 3>> landings(count);
    MetalRunWaiting(run.get(), reinterpret_cast<uint64_t (*)[3]>(landings.data()), count, &parts);
    return {parts, landings};
  }
  // nonblocking: None while pending, else the latest end (mesh_statistics_now's clock); raises a failure
  std::optional<double> end() {
    double ended = 0;
    NSError *error = nil;
    const int status = MetalRunEnd(run.get(), &ended, &error);
    TORCH_CHECK(status >= 0, "the run failed: ", error ? error.localizedDescription.UTF8String : "");
    return status ? std::optional<double>(ended) : std::nullopt;
  }
};

// Every call from Python drains the autorelease pool it pushed: Python's thread has no run loop.
extern "C" void *objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void *);
struct Drained {
  void *pool = objc_autoreleasePoolPush();
  Drained() = default;
  Drained(const Drained &) = delete;
  ~Drained() { objc_autoreleasePoolPop(pool); }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::register_exception_translator([](std::exception_ptr raised) {
    @try { if (raised) std::rethrow_exception(raised); }
    @catch (NSException *error) {
      PyErr_SetString(PyExc_ValueError, [NSString stringWithFormat:@"%@: %@", error.name, error.reason].UTF8String);
    }
  });
  install_region_allocator();
  module.attr("RUN_FREE") = pybind11::int_(MESH_RUN_FREE);
  module.attr("RUN_BEFORE") = pybind11::int_((int)MESH_RUN_BEFORE);
  module.attr("RUN_AFTER") = pybind11::int_((int)MESH_RUN_AFTER);
  module.def("run_solve", [](uint64_t page_bytes, uint32_t events,
                             const std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint64_t, uint64_t, uint64_t>> &accesses,
                             const std::vector<std::pair<uint32_t, uint32_t>> &links,
                             const std::vector<std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>> &items,
                             const std::vector<std::pair<uint64_t, uint64_t>> &bins) {
    std::vector<mesh_graph_access> access;
    for (const auto &a : accesses)
      access.push_back({std::get<0>(a), std::get<1>(a), std::get<2>(a), std::get<3>(a), std::get<4>(a), std::get<5>(a)});
    std::vector<uint32_t> link;
    for (const auto &l : links) { link.push_back(l.first); link.push_back(l.second); }
    std::vector<mesh_run_item> item;
    for (const auto &i : items) item.push_back({std::get<0>(i), std::get<1>(i), std::get<2>(i), std::get<3>(i)});
    std::vector<mesh_run_bin> bin;
    for (const auto &b : bins) bin.push_back({b.first, b.second});
    const mesh_page_space space{page_bytes, 0};
    mesh_run_plan plan;
    const int status = mesh_run_solve(&space, events, access.size(), access.data(), links.size(),
                                      reinterpret_cast<const uint32_t (*)[2]>(link.data()), item.size(), item.data(),
                                      bin.size(), bin.data(), nullptr, &plan);
    TORCH_CHECK(!status, "run_solve: ", status == ENOMEM ? "the run's live set exceeds its bins" :
                status == EBUSY ? "two pinned items meet while both are live" :
                "an access outside its item or events, unordered bins, a pin in no bin, or a cycle in the edges",
                " (design/page-maps-2026-11-02.md)");
    pybind11::list edges;
    for (size_t k = 0; k < plan.edges; k++) {
      const mesh_graph_dependence &d = plan.edge[k];
      edges.append(pybind11::make_tuple(d.from, d.to, d.kind, d.item, d.first, d.bytes));
    }
    pybind11::dict out;
    out["edges"] = edges;
    out["order"] = std::vector<uint32_t>(plan.order, plan.order + events);
    pybind11::list runs;
    for (size_t k = 0; k < item.size(); k++) {
      pybind11::list own;
      for (size_t r = plan.run_first[k]; r < plan.run_first[k + 1]; r++) own.append(pybind11::make_tuple(plan.runs[r][0], plan.runs[r][1]));
      runs.append(own);
    }
    out["runs"] = runs;
    out["arena"] = plan.arena;
    mesh_run_plan_free(&plan);
    return out;
  });
  module.def("region_page_bytes", []() {
    uint64_t page_bytes = 0;
    TORCH_CHECK(!ncclMeshRegionPageBytes(&page_bytes), "region_page_bytes: ", ncclGetLastError(nullptr));
    return page_bytes;
  });
  module.def("region_page_bytes", []() {
    uint64_t page_bytes = 0;
    TORCH_CHECK(!ncclMeshRegionPageBytes(&page_bytes), "region_page_bytes: ", ncclGetLastError(nullptr));
    return page_bytes;
  });
  module.def("region_empty", &region_empty, pybind11::call_guard<Drained>());
  module.def("address", &address, pybind11::call_guard<Drained>());
  module.def("attach", &attach, pybind11::call_guard<Drained>());
  module.def("communicator_of", [](const std::string &name) { return communicator_of(c10::IValue(name)); });
  module.def("pipeline", &pipeline, pybind11::arg("source"), pybind11::arg("function"),
             pybind11::arg("constants") = std::vector<std::tuple<uint32_t, uint32_t, pybind11::bytes>>{},
             pybind11::call_guard<Drained>());
  pybind11::class_<Run>(module, "Run")
    .def(pybind11::init<>(), pybind11::call_guard<Drained>())
    .def("event", &Run::event)
    .def("dispatch", &Run::dispatch, pybind11::call_guard<Drained>())
    .def("copy", &Run::copy, pybind11::call_guard<Drained>())
    .def("landed", &Run::landed, pybind11::call_guard<Drained>())
    .def("inputs", &Run::inputs, pybind11::call_guard<Drained>())
    .def("inputs_now", &Run::inputs_now, pybind11::call_guard<Drained>())
    .def("edges", &Run::edges, pybind11::call_guard<Drained>())
    .def("cut", [](Run &r) { MetalRunCut(r.run.get()); }, pybind11::call_guard<Drained>())
    .def("stamp", [](Run &r) { MetalRunStamp(r.run.get()); }, pybind11::call_guard<Drained>())
    .def("trace", [](Run &r) { MetalRunTrace(r.run.get()); }, pybind11::call_guard<Drained>())
    .def("mark", [](Run &r) { MetalRunMark(r.run.get()); }, pybind11::call_guard<Drained>())
    .def("times", [](Run &r) {
      std::vector<double> out(MetalRunTimes(r.run.get(), nullptr, 0));
      MetalRunTimes(r.run.get(), out.data(), out.size());
      return out;
    }, pybind11::call_guard<Drained>())
    .def("spans", [](Run &r) {
      std::vector<double> out(MetalRunSpans(r.run.get(), nullptr, 0));
      MetalRunSpans(r.run.get(), out.data(), out.size());
      return out;
    }, pybind11::call_guard<Drained>())
    .def("communicator", &Run::communicator)
    .def("finish", &Run::finish, pybind11::call_guard<Drained>())
    .def("enqueue", &Run::enqueue, pybind11::call_guard<Drained>())
    .def("end", &Run::end, pybind11::call_guard<Drained>())
    .def("waiting", &Run::waiting, pybind11::call_guard<Drained>())
    .def("landings", [](Run &r) {
      std::vector<std::array<double, 2>> out(MetalRunLandings(r.run.get(), nullptr, 0));
      MetalRunLandings(r.run.get(), reinterpret_cast<double (*)[2]>(out.data()), out.size());
      return out;
    }, pybind11::call_guard<Drained>())
    .def("parts", [](Run &r) {
      std::vector<std::array<double, 3>> times(MetalRunParts(r.run.get(), nullptr, 0));
      MetalRunParts(r.run.get(), reinterpret_cast<double (*)[3]>(times.data()), times.size());
      return times;
    }, pybind11::call_guard<Drained>());
}
