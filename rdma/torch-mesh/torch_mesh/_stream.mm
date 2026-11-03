// torch_mesh's binding (rdma/metal.h): region tensors, each torch stream's Metal 4 queue beside it, compiled pipelines
// (metal.h 1), recorded runs (metal.h 2, over mesh_graph_realize's outputs) and their commits (metal.h 3), the run's
// solve, and the communicator a process group names. The caller drives: it commits a run once, runs its host steps in
// node order, each once the event values its rows name are reached, and reads the event values and words it owns
// (value); nothing here calls back into it.
#include <torch/extension.h>
#include <torch/library.h>
#include <c10/core/CPUAllocator.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <cerrno>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "metal.h"
#include "mesh-graph.h"
#include "mesh-page-map.h"
#include "nccl.h"

void install_region_allocator();

// every shared event of this binding, retained for the process, by its handle: a stream's `progress` (its writer
// torch's own command buffer, inputs) and its `done` (its writer the stream's queue, Run::commit). No host stores one:
// the metal ticket's revision (2026-11-03) deleted `event` and `signal`, by which a caller's loop raised a landing's or
// a host step's event for a part held on it (a CPU-mediated start, 93-178 us on an M5 Ultra: metal-microbench
// docs/facts.md, wake.m); a consumer reads its own guard (metal.h MetalNode). What must stay: GPU writers only.
static std::vector<id<MTLSharedEvent>> *events_ = new std::vector<id<MTLSharedEvent>>;

static int64_t event_of(id<MTLDevice> device) {
  events_->push_back([device newSharedEvent]);
  return (int64_t)events_->size() - 1;
}

// A torch stream's Metal 4 queue and its events: `progress` rises as torch's work commits (inputs), `done` as a run's
// parts complete (Run::commit; `ended`, the value it reaches once every run committed so far has completed).
// `resident`, on the queue for all its command buffers, holds every lease an argument of a run addresses (metal.h 2).
// The metal ticket (2026-11-03) replaced torch's MPSEvent notify block, which raised `progress` from a listener, with
// a signal encoded on torch's own command buffer: no callback, and the signals rise in torch's queue order.
struct MeshStream {
  at::mps::MPSStream *const mps;
  id<MTL4CommandQueue> const queue;
  id<MTLResidencySet> const resident;
  const int64_t progress, done;
  uint64_t committed = 0, ended = 0;
};

static MeshStream *made_stream(at::mps::MPSStream *mps) {
  id<MTLDevice> device = mps->device();
  NSError *failure = nil;
  id<MTLResidencySet> resident = [device newResidencySetWithDescriptor:[[MTLResidencySetDescriptor new] autorelease]
                                                                 error:&failure];
  id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
  TORCH_CHECK(resident && queue, "preparing Metal context: ",
              failure ? failure.localizedDescription.UTF8String : "command queue unavailable");
  [queue addResidencySet:resident];
  return new MeshStream{mps, queue, resident, event_of(device), event_of(device)};
}

static std::mutex *streams_lock_ = new std::mutex;
static auto *streams_ = new std::unordered_map<at::mps::MPSStream *, MeshStream *>;

// torch's stream's MeshStream, made at its first use.
static MeshStream &stream_of(at::mps::MPSStream *mps) {
  thread_local std::pair<at::mps::MPSStream *, MeshStream *> last{nullptr, nullptr};
  if (last.first == mps) return *last.second;
  std::lock_guard<std::mutex> guard(*streams_lock_);
  auto found = streams_->find(mps);
  if (found == streams_->end()) found = streams_->emplace(mps, made_stream(mps)).first;
  last = {mps, found->second};
  return *found->second;
}

// torch's work so far on the current stream, committed with `progress` signalled after it: an outside producer, its
// event's handle and the value its consumers' parts wait for (Run::commit's value)
static std::pair<int64_t, uint64_t> inputs() {
  MeshStream &stream = stream_of(at::mps::getCurrentMPSStream());
  const uint64_t value = ++stream.committed;
  stream.mps->endKernelCoalescing();
  [stream.mps->commandBuffer() encodeSignalEvent:(*events_)[stream.progress] value:value];
  stream.mps->synchronize(at::mps::SyncType::COMMIT);
  return {stream.progress, value};
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

// metal.h 1: each compiled pipeline, retained for the process, by its handle
static std::vector<id<MTLComputePipelineState>> *pipelines_ = new std::vector<id<MTLComputePipelineState>>;

// `function` of `source`, specialized by `constants` ((index, MTLDataType, bytes) each); its handle
static int64_t pipeline(const std::string &source, const std::string &function,
                        const std::vector<std::tuple<uint32_t, uint32_t, pybind11::bytes>> &constants) {
  std::vector<std::string> raw;
  std::vector<MetalConstant> constant;
  for (const auto &[index, type, bytes] : constants) {
    raw.push_back(bytes);
    constant.push_back({index, type, nullptr});
  }
  for (size_t k = 0; k < raw.size(); k++) constant[k].value = raw[k].data();
  char *error = nullptr;
  id<MTLComputePipelineState> made = MetalCompile(source.c_str(), function.c_str(), constant.data(), constant.size(), &error);
  const std::string statement = error ? error : "";
  free(error);
  TORCH_CHECK(made, "compiling ", function, ": ", statement);
  pipelines_->push_back(made);
  return (int64_t)pipelines_->size() - 1;
}

// A dispatch argument: a region tensor (its storage's buffer, from its first element and `offset` more bytes) or a
// word's address (the lease or stream page holding it: nccl.h ncclMeshMemBuffer), as a GPU address, its buffer made
// resident on the stream's queue
static MetalArgument bound(const pybind11::handle &operand, uint32_t index, uint64_t offset, std::vector<at::Tensor> &held,
                           id<MTLResidencySet> resident) {
  id<MTLBuffer> buffer = nil;
  uint64_t at = offset;
  if (THPVariable_Check(operand.ptr())) { // torch's caster is no registered pybind type: isinstance<at::Tensor> is false
    const at::Tensor t = THPVariable_Unpack(operand.ptr());
    TORCH_CHECK(t.is_mps(), "a run argument is an MPS region tensor");
    held.push_back(t);
    buffer = at::native::mps::getMTLBufferStorage(t);
    at += t.storage_offset() * t.element_size();
  } else {
    void *theirs = nullptr;
    size_t within = 0;
    checked(ncclMeshMemBuffer((void *)(uintptr_t)operand.cast<uint64_t>(), &theirs, &within));
    buffer = (__bridge id<MTLBuffer>)theirs;
    at += within;
  }
  [resident addAllocation:buffer];
  return {buffer.gpuAddress + at, index, 0};
}

// metal.h 2 and 3: a run recorded on the current stream's queue, committed once. Its nodes in the solve's order, each
// (pipeline handle or -1, [(argument index, region tensor or word address, byte offset)], groups, threads, event
// handle or -1: a dispatch, its guard's count and threshold words among its arguments; an outside producer, its event
// the stream's `progress` or `done`; or with neither a host step the caller runs); the edges (from, to) between them;
// the nodes ending their parts (mesh_graph_realize `ends`). An edge a guard realizes (a landing's, a SEND
// completion's or a host step's to its GPU consumer) is no edge of the run (metal.h MetalNode). It keeps, of
// realize's waits rows, those a host step waits on (waiter, kind, producer): kind 1 on a part, kind 5 on an outside
// producer; a host step after a host step is in node order, the caller's.
// Changed by the metal ticket's revision (2026-11-03) from exposing realize's part, deferred and waits rows for the
// caller to work out its commit ranges, deferrals and host waits from (readiness logic outside the solve). What must
// stay: the caller walks values, never rows.
using RunNode = std::tuple<int64_t, std::vector<std::tuple<uint32_t, pybind11::object, uint64_t>>,
                           std::array<uint32_t, 3>, std::array<uint32_t, 3>, int64_t>;
struct Run {
  MeshStream *const stream;
  id run = nil;
  std::vector<at::Tensor> held;
  uint32_t parts = 0;
  std::vector<int64_t> event;
  std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> hosts;

  Run(const std::vector<RunNode> &nodes, const std::vector<std::pair<uint32_t, uint32_t>> &edges,
      const std::vector<uint32_t> &ends) : stream(&stream_of(at::mps::getCurrentMPSStream())) {
    const uint32_t n = (uint32_t)nodes.size();
    std::vector<MetalNode> node(n);
    std::vector<std::vector<MetalArgument>> arguments(n);
    std::vector<unsigned char> host(n), end(n), barrier(n), fence(n);
    event.resize(n);
    for (uint32_t v = 0; v < n; v++) {
      const auto &[handle, operands, groups, threads, produced] = nodes[v];
      for (const auto &[index, operand, offset] : operands)
        arguments[v].push_back(bound(operand, index, offset, held, stream->resident));
      node[v] = {handle >= 0 ? (*pipelines_)[handle] : nil, arguments[v].data(), (uint32_t)arguments[v].size(),
                 {groups[0], groups[1], groups[2]}, {threads[0], threads[1], threads[2]},
                 produced >= 0 ? (*events_)[produced] : nil};
      host[v] = handle >= 0 ? 0 : produced >= 0 ? 2 : 1;
      event[v] = produced;
    }
    for (uint32_t v : ends) end[v] = 1;
    [stream->resident commit];
    std::vector<uint32_t> edge;
    for (const auto &[from, to] : edges) { edge.push_back(from); edge.push_back(to); }
    std::vector<mesh_graph_crossing> crossing(edges.size() + 1);
    size_t count = 0;
    std::vector<uint32_t> part(n);
    std::vector<unsigned char> deferred(n); // realize's, unread (metal.h 2)
    const int status = mesh_graph_realize(n, host.data(), edges.size(), reinterpret_cast<const uint32_t (*)[2]>(edge.data()),
                                          part.data(), &parts, barrier.data(), fence.data(), crossing.data(),
                                          crossing.size(), &count, deferred.data(), end.data());
    TORCH_CHECK(!status, "realizing the run: ", status == EINVAL ? "an edge runs backward in the order" : "out of memory");
    char *error = nullptr;
    run = MetalRecord(stream->queue, n, node.data(), part.data(), parts, barrier.data(), fence.data(), crossing.data(),
                      count, &error);
    const std::string statement = error ? error : "";
    free(error);
    TORCH_CHECK(run, "recording the run: ", statement);
    for (size_t j = 0; j < count; j++)
      if (crossing[j].kind == 1 || (crossing[j].kind == 5 && host[crossing[j].producer] == 2))
        hosts.emplace_back(crossing[j].waiter, crossing[j].kind, crossing[j].producer);
  }
  Run(const Run &) = delete;
  ~Run() { MetalRelease(run); }

  // metal.h 3: every part committed once, each after its queue waits for outside producer v's event to reach value[v]
  // (indexed by node). Returns the value the stream's done reaches once the run has completed (the caller keeps the run
  // until then; a later occurrence waits on done at it, an outside producer of its own run), and for each node the
  // (event handle, value) pairs a host step waits for before the caller runs it: done's signal after a part it reads,
  // an outside producer's value.
  std::pair<uint64_t, std::vector<std::vector<std::pair<int64_t, uint64_t>>>> commit(const std::vector<uint64_t> &value) {
    const uint64_t base = stream->ended;
    stream->ended = base + parts;
    MetalCommit(run, value.data(), (*events_)[stream->done], base);
    std::vector<std::vector<std::pair<int64_t, uint64_t>>> waits(event.size());
    for (const auto &[waiter, kind, producer] : hosts)
      waits[waiter].emplace_back(kind == 5 ? event[producer] : stream->done,
                                 kind == 5 ? value[producer] : base + producer + 1);
    return {stream->ended, waits};
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
  module.attr("RUN_COHERENT") = pybind11::int_((int)MESH_RUN_COHERENT);
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
  module.def("region_empty", &region_empty, pybind11::call_guard<Drained>());
  module.def("address", &address, pybind11::call_guard<Drained>());
  module.def("attach", &attach, pybind11::call_guard<Drained>());
  module.def("communicator_of", [](const std::string &name) { return communicator_of(c10::IValue(name)); });
  module.def("pipeline", &pipeline, pybind11::arg("source"), pybind11::arg("function"),
             pybind11::arg("constants") = std::vector<std::tuple<uint32_t, uint32_t, pybind11::bytes>>{},
             pybind11::call_guard<Drained>());
  // shared events by handle: its value now; the current stream's torch inputs and done
  module.def("value", [](int64_t handle) { return (*events_)[handle].signaledValue; });
  module.def("inputs", &inputs, pybind11::call_guard<Drained>());
  module.def("done", []() { return stream_of(at::mps::getCurrentMPSStream()).done; }, pybind11::call_guard<Drained>());
  pybind11::class_<Run>(module, "Run")
    .def(pybind11::init<const std::vector<RunNode> &, const std::vector<std::pair<uint32_t, uint32_t>> &,
                        const std::vector<uint32_t> &>(),
         pybind11::arg("nodes"), pybind11::arg("edges"), pybind11::arg("ends") = std::vector<uint32_t>{},
         pybind11::call_guard<Drained>())
    .def("commit", &Run::commit, pybind11::arg("value"), pybind11::call_guard<Drained>());
}
