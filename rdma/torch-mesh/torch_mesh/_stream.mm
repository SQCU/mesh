// torch's tensors as libnccl-mesh's operands (nccl.h: blocking calls on host memory, each complete when it returns):
// every collective of torch_mesh/backend.py and of torch's functional collectives on MPS is one call of `issue`, a group
// of calls each one kind (an all-reduce, reduce, broadcast, all-gather, reduce-scatter, all-to-all, gather or scatter; an
// all-gather's or reduce-scatter's segment a rank of any size).  A group that names an MPS tensor first drains torch's
// stream (its open command buffer committed and waited for, as torch.mps.synchronize waits), so every kernel that writes
// an operand has run; the group then runs on the host, and torch's later kernels read its results in the stream's
// order.  An operand
// is read and written where it lies where the host reaches its bytes (a CPU tensor; an MPS one of host_empty's, which
// torch_mesh.empty makes); any other MPS operand is staged through a host tensor by torch's copies on its stream.  The
// rest is the library's own kernels on torch's stream (pipeline, encode, linear), their timing (timed, sequence,
// GPUTimings) and the evidence's marks.
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/DLConvertor.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSEvent.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "nccl.h"

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

// The evidence torch_mesh/evidence.py reads (design/heterogeneity.md R2, R3): on a watched communicator, a timing
// event of torch's pool recorded on its stream as each MPS group is issued (0: the GPU has finished the work before it)
// and once it is complete (1), and at each step's boundary (2), in order.  Each mark commits torch's command buffer (the
// pool's synchronizing record), so no stretch holds the GPU idle while work encoded before a mark waits for a later
// commit.
static std::unordered_map<int64_t, std::vector<std::pair<int64_t, int64_t>>> marks_;

static void mark(int64_t comm, int64_t kind) {
  if (marks_.empty()) return;
  auto found = marks_.find(comm);
  if (found == marks_.end()) return;
  auto pool = at::mps::getMPSEventPool();
  const auto id = pool->acquireEvent(true);
  pool->recordEvent(id, true);
  found->second.emplace_back(kind, (int64_t)id);
}

static ncclComm_t communicator(int64_t comm) { return (ncclComm_t)(uintptr_t)comm; }

// Host tensors (torch_mesh.empty, records, address): a tensor whose bytes the host reaches where they lie, so a
// collective reads and writes it in place.  An MPS one's storage is a Metal buffer of shared storage (on unified memory
// the GPU's and the host's alike), made an MPS tensor through torch's documented interop, DLPack (at::fromDLPack of a
// kDLMetal tensor: its data an id<MTLBuffer>, as torch's own toDLPack writes an MPS tensor); its release is the rule
// torch's MPS allocator keeps for a buffer in flight: the buffer let go once the stream's command buffer open at the
// release has completed, so every command bound to it before then has run.  A held one (`held`) is kept then instead,
// by its bytes, and taken again by the next held tensor of its size, so a program's steady steps allocate nothing.  A
// CPU one is torch's own CPU tensor.  No allocator of torch's is replaced: tensors the program makes otherwise stay its
// MPS allocator's, whose buffers are the GPU's private storage, which the host does not reach.
static std::mutex host_tensors_lock_;
// each live MPS host tensor's buffer: its contents' address and its bytes
static std::map<uintptr_t, size_t> host_tensors_;
// the held buffers no tensor holds now (each retained), by their bytes
static std::multimap<size_t, void *> held_;

struct HostShape {
  std::vector<int64_t> sizes, strides;
};

static at::Tensor host_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps, bool held = false) {
  const at::ScalarType dtype = like.scalar_type();
  const auto options = at::TensorOptions().dtype(dtype);
  if (!mps) return at::empty(sizes, options.device(at::kCPU));
  const size_t bytes = std::max<size_t>(1, (size_t)c10::multiply_integers(sizes) * c10::elementSize(dtype));
  id<MTLBuffer> buffer = nil;
  {
    std::lock_guard<std::mutex> guard(host_tensors_lock_);
    auto found = held ? held_.find(bytes) : held_.end();
    if (found != held_.end()) {
      buffer = (id<MTLBuffer>)found->second;
      held_.erase(found);
    }
  }
  if (!buffer) {
    buffer = [at::mps::MPSDevice::getInstance()->device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    TORCH_CHECK(buffer, "torch_mesh.empty: Metal made no shared buffer of ", bytes, " bytes");
  }
  {
    std::lock_guard<std::mutex> guard(host_tensors_lock_);
    host_tensors_[(uintptr_t)[buffer contents]] = bytes;
  }
  auto *shape = new HostShape{sizes, std::vector<int64_t>(std::max<size_t>(sizes.size(), 1), 1)};
  for (int64_t d = (int64_t)sizes.size() - 2; d >= 0; d--) shape->strides[d] = shape->strides[d + 1] * std::max<int64_t>(sizes[d + 1], 1);
  auto *managed = new DLManagedTensor{};
  managed->dl_tensor.data = (void *)buffer;
  managed->dl_tensor.device = {kDLMetal, 0};
  managed->dl_tensor.ndim = (int32_t)sizes.size();
  managed->dl_tensor.dtype = at::getDLDataType(at::empty({0}, options));
  managed->dl_tensor.shape = shape->sizes.data();
  managed->dl_tensor.strides = shape->strides.data();
  managed->dl_tensor.byte_offset = 0;
  managed->manager_ctx = shape;
  managed->deleter = [](DLManagedTensor *self) {
    delete (HostShape *)self->manager_ctx;
    delete self;
  };
  void *kept = (void *)buffer;
  return at::fromDLPack(managed, [managed, kept, bytes, held](void *) {
    managed->deleter(managed);
    at::mps::getCurrentMPSStream()->addCompletedHandler(^(id<MTLCommandBuffer>) {
      id<MTLBuffer> released = (id<MTLBuffer>)kept;
      std::lock_guard<std::mutex> guard(host_tensors_lock_);
      host_tensors_.erase((uintptr_t)[released contents]);
      if (held) held_.emplace(bytes, kept);
      else [released release];
    });
  });
}

// Where the host reaches a tensor's bytes: a CPU tensor's own, an MPS one's where its storage's Metal buffer is of
// shared storage; NULL where the host does not reach them (an MPS tensor of torch's allocator) or the tensor has none.
static char *host_address(const at::Tensor &t) {
  if (!t.is_mps()) return (char *)t.data_ptr();
  id<MTLBuffer> buffer = at::native::mps::getMTLBufferStorage(t);
  char *contents = buffer ? (char *)[buffer contents] : nullptr;
  return contents ? contents + t.storage_offset() * t.element_size() : nullptr;
}

// The host tensors' buffers live now: each its address and bytes.
static std::vector<std::pair<uint64_t, uint64_t>> records() {
  std::lock_guard<std::mutex> guard(host_tensors_lock_);
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (auto &w : host_tensors_) out.emplace_back(w.first, w.second);
  return out;
}

// One call of a group: its kind, its send and receive tensors (an all-reduce's, reduce's or broadcast's sends alone
// where it is in place; a gather's receive and a scatter's send off the root none), its counts (an all-gather's,
// reduce-scatter's or all-to-all's segment a rank, in rank order, in its tensors' elements; an all-to-all's all one),
// its operator (an ncclRedOp_t) and its root.
enum Kind : int64_t { ALLREDUCE, REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL, GATHER, SCATTER };
using Tensors = std::vector<std::optional<at::Tensor>>;
using Call = std::tuple<int64_t, Tensors, Tensors, std::vector<int64_t>, int64_t, int64_t>;

// A tensor as the library reads it, with ProcessGroupNCCL's dtype semantics: a collective that moves data moves any
// dtype (one NCCL lacks as its bytes); a reduction reduces a complex tensor as its real view (sum and average) and a
// bool tensor's sum and product as max and min (its average the library's uint8 average: the sum truncated by the
// ranks, 1 only where every rank holds 1).  `at` is its bytes on the host (NULL: none), `size` the bytes of one of its
// NCCL elements.
struct Operand {
  char *at = nullptr;
  size_t count = 0, size = 1;
  ncclDataType_t type = ncclUint8;
};

// A group's operands, each contiguous (a non-contiguous one fails as a value: nothing is copied to make it one).  An
// operand the host reaches is passed where it lies; any other (an MPS tensor of torch's allocator) is staged through a
// held host tensor by torch's copies on its stream: into it before the group where the call reads it, and back after
// the group where the call writes it (`back`).
struct Group {
  bool mps = false;
  std::vector<at::Tensor> staged;
  std::vector<std::pair<at::Tensor, at::Tensor>> back;

  char *where(const at::Tensor &t, bool read, bool written) {
    TORCH_CHECK(t.is_contiguous(), "the mesh backend's operand is a contiguous tensor; a non-contiguous one is not copied into one");
    mps |= t.is_mps();
    char *reached = host_address(t);
    if (reached || !t.is_mps() || !t.numel()) return reached;
    at::Tensor s = host_empty(t.sizes().vec(), t, true, true);
    if (read) s.copy_(t);
    if (written) back.emplace_back(t, s);
    staged.push_back(s);
    return host_address(s);
  }
  Operand moving(const at::Tensor &t, bool read, bool written) {
    const ncclDataType_t type = kind(t.scalar_type());
    char *bytes = where(t, read, written);
    return type == ncclNumTypes ? Operand{bytes, (size_t)t.nbytes(), 1, ncclUint8} : Operand{bytes, (size_t)t.numel(), (size_t)t.element_size(), type};
  }
  Operand reducing(const at::Tensor &t, int64_t &op, bool read, bool written) {
    if (t.scalar_type() == at::kBool) op = op == ncclSum ? ncclMax : op == ncclProd ? ncclMin : op;
    if (t.is_complex()) {
      TORCH_CHECK(op == ncclSum || op == ncclAvg, "a complex tensor reduces by sum and average alone");
      return {where(t, read, written), (size_t)t.numel() * 2, (size_t)t.element_size() / 2, kind(at::toRealValueType(t.scalar_type()))};
    }
    TORCH_CHECK(kind(t.scalar_type()) != ncclNumTypes, "the mesh backend reduces no ", t.scalar_type());
    return {where(t, read, written), (size_t)t.numel(), (size_t)t.element_size(), kind(t.scalar_type())};
  }
};

// A call's operands and counts (in its operands' NCCL elements), made before its group runs.
struct Issue {
  int64_t kind, op, root;
  Operand send, receive;
  std::vector<size_t> counts;
};

static Issue prepare(const Call &call, Group &g, int rank) {
  const auto &[what, sends, receives, counts, given, root] = call;
  Issue i{what, given, root, {}, {}, {}};
  auto tensor = [](const Tensors &list, size_t k) -> const at::Tensor & {
    TORCH_CHECK(k < list.size() && list[k], "a collective call without its tensor ", k);
    return *list[k];
  };
  switch (what) {
  case ALLREDUCE: case REDUCE: {
    const bool lands = what == ALLREDUCE || rank == root;
    i.send = g.reducing(tensor(sends, 0), i.op, true, receives.empty() && lands);
    i.receive = receives.empty() ? i.send : g.reducing(tensor(receives, 0), i.op, false, lands);
    break;
  }
  case BROADCAST:
    i.send = i.receive = g.moving(tensor(sends, 0), rank == root, rank != root);
    break;
  case ALLGATHER: case REDUCE_SCATTER: case ALLTOALL: {
    // MPI_Allgatherv, MPI_Reduce_scatter and MPI_Alltoall: rank r's segment is counts[r] elements, the segments packed
    // in rank order in the whole operand (the receive's, the send's, either's)
    const bool reduces = what == REDUCE_SCATTER;
    const at::Tensor &whole = reduces ? tensor(sends, 0) : tensor(receives, 0);
    i.send = reduces ? g.reducing(tensor(sends, 0), i.op, true, false) : g.moving(tensor(sends, 0), true, false);
    i.receive = reduces ? g.reducing(tensor(receives, 0), i.op, false, true) : g.moving(tensor(receives, 0), false, true);
    const size_t per = whole.numel() ? (reduces ? i.send : i.receive).count / (size_t)whole.numel() : 1;
    for (auto c : counts) i.counts.push_back((size_t)c * per);
    for (size_t c : i.counts)
      TORCH_CHECK(what != ALLTOALL || c == i.counts[0], "the mesh backend's all-to-all moves segments of one size (nccl.h ncclAlltoAll)");
    break;
  }
  case GATHER:
    i.send = g.moving(tensor(sends, 0), true, false);
    if (rank == root) i.receive = g.moving(tensor(receives, 0), false, true);
    break;
  case SCATTER:
    if (rank == root) i.send = g.moving(tensor(sends, 0), true, false);
    i.receive = g.moving(tensor(receives, 0), false, true);
    break;
  default: TORCH_CHECK(false, "no collective of kind ", what);
  }
  return i;
}

// The group's calls, one NCCL group in their order: an all-gather of equal segments is ncclAllGather, of others a
// broadcast a rank (rank r's segment from rank r into its place); a reduce-scatter of equal segments is
// ncclReduceScatter, of others a reduce a rank (segment r at rank r, in place on the other ranks, which NCCL leaves
// unwritten); an all-to-all, a gather and a scatter are ncclAlltoAll, ncclGather and ncclScatter.
static ncclResult_t called(const std::vector<Issue> &issues, ncclComm_t comm, int rank) {
  ncclResult_t result = ncclGroupStart();
  for (const Issue &i : issues) {
    const ncclRedOp_t op = (ncclRedOp_t)i.op;
    const Operand &a = i.send, &b = i.receive;
    bool equal = true;
    for (size_t c : i.counts) equal &= c == i.counts[0];
    const size_t first = i.counts.empty() ? 0 : i.counts[0];
    size_t before = 0;
    if (result) break;
    switch (i.kind) {
    case ALLREDUCE: result = ncclAllReduce(a.at, b.at, a.count, a.type, op, comm, nullptr); break;
    case REDUCE: result = ncclReduce(a.at, b.at, a.count, a.type, op, (int)i.root, comm, nullptr); break;
    case BROADCAST: result = ncclBroadcast(a.at, b.at, a.count, a.type, (int)i.root, comm, nullptr); break;
    case ALLGATHER:
      if (equal) {
        result = ncclAllGather(a.at, b.at, first, b.type, comm, nullptr);
        break;
      }
      for (size_t r = 0; r < i.counts.size() && !result; before += i.counts[r++])
        if (i.counts[r])
          result = ncclBroadcast((int)r == rank ? a.at : b.at + before * b.size, b.at + before * b.size, i.counts[r], b.type, (int)r, comm,
                                 nullptr);
      break;
    case REDUCE_SCATTER:
      if (equal) {
        result = ncclReduceScatter(a.at, b.at, first, a.type, op, comm, nullptr);
        break;
      }
      for (size_t r = 0; r < i.counts.size() && !result; before += i.counts[r++])
        if (i.counts[r])
          result = ncclReduce(a.at + before * a.size, (int)r == rank ? b.at : a.at + before * a.size, i.counts[r], a.type, op, (int)r, comm,
                              nullptr);
      break;
    case ALLTOALL: result = ncclAlltoAll(a.at, b.at, first, b.type, comm, nullptr); break;
    case GATHER: result = ncclGather(a.at, b.at, a.count, a.type, (int)i.root, comm, nullptr); break;
    case SCATTER: result = ncclScatter(a.at, b.at, b.count, b.type, (int)i.root, comm, nullptr); break;
    }
  }
  const ncclResult_t ended = ncclGroupEnd();
  return result ? result : ended;
}

// torch's stream committed and waited for on its queue (torch's COMMIT_AND_WAIT, which torch.mps.synchronize takes),
// so every kernel encoded before has written what a group reads.
static void drained() {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
    @autoreleasepool {
      stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    }
  });
}

// One group of `calls` on comm, complete when it returns: its operands prepared (staging copies encoded on torch's
// stream), torch's stream drained where it names an MPS tensor, the group run on the host (the GIL released while it
// runs where `unlock`), then the copies back encoded.  An MPS group's marks are taken here (a host group's by
// backend.py).
static void issue(const std::vector<Call> &calls, int64_t comm, bool unlock) {
  int rank = 0;
  checked(ncclCommUserRank(communicator(comm), &rank));
  Group g;
  std::vector<Issue> issues;
  for (const Call &c : calls) issues.push_back(prepare(c, g, rank));
  if (g.mps) mark(comm, 0);
  ncclResult_t result = ncclSuccess;
  {
    std::optional<pybind11::gil_scoped_release> unlocked;
    if (unlock) unlocked.emplace();
    if (g.mps) drained();
    result = called(issues, communicator(comm), rank);
  }
  checked(result);
  for (auto &[t, s] : g.back) t.copy_(s);
  if (g.mps) mark(comm, 1);
}

// torch's functional collectives (_c10d_functional, what DTensor and a compiled graph call) on MPS tensors: kernels
// for the MPS dispatch key, as torch/csrc/distributed/c10d/Functional.cpp implements them for every device but
// issuing their group directly, the group's communicator found by the group (attach) with no Python between.
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

// A functional collective's result is a held host tensor (host_empty), taken again by the next result of its size once
// the stream is done with it, so a program's steady steps allocate nothing; the call is complete when it returns, so no
// work is registered for torch's wait_tensor.
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
    result = name.back() == '_' ? input : host_empty(input.sizes().vec(), input, true, true).copy_(input);
    call = name[0] == 'a' ? Call{ALLREDUCE, {result}, {}, {}, reduction(arguments[1].toStringRef()), 0}
                          : Call{BROADCAST, {result}, {}, {}, 0, arguments[1].toInt()};
  } else if (name.rfind("all_gather_into_tensor", 0) == 0) {
    std::vector<int64_t> shape = input.sizes().vec();
    TORCH_CHECK(!shape.empty(), "all_gather_into_tensor of a 0-dim tensor");
    shape[0] *= ranks;
    result = name == "all_gather_into_tensor_out" ? arguments[3].toTensor() : host_empty(shape, input, true, true);
    TORCH_CHECK(result.numel() == input.numel() * ranks, "all_gather_into_tensor_out: out holds ", result.numel(), " elements, not ",
                input.numel() * ranks);
    call = {ALLGATHER, {input}, {result}, std::vector<int64_t>(ranks, input.numel()), 0, 0};
  } else if (name == "reduce_scatter_tensor") {
    std::vector<int64_t> shape = input.sizes().vec();
    TORCH_CHECK(!shape.empty() && shape[0] % ranks == 0, "reduce_scatter_tensor: dim 0 of ", input.sizes(), " is not divisible by ", ranks);
    shape[0] /= ranks;
    result = host_empty(shape, input, true, true);
    call = {REDUCE_SCATTER, {input}, {result}, std::vector<int64_t>(ranks, result.numel()), reduction(arguments[1].toStringRef()), 0};
  } else {
    // all_to_all_single of equal splits (ncclAlltoAll moves segments of one size)
    TORCH_CHECK(input.dim() && input.size(0) % ranks == 0, "all_to_all_single: dim 0 of ", input.sizes(), " is not divisible by ", ranks);
    for (int k = 1; k <= 2; k++)
      for (const auto &v : arguments[k].toListRef())
        TORCH_CHECK(v.toInt() == input.size(0) / ranks, "all_to_all_single: the mesh backend's all-to-all moves equal splits");
    result = host_empty(input.sizes().vec(), input, true, true);
    call = {ALLTOALL, {input}, {result}, std::vector<int64_t>(ranks, input.numel() / ranks), 0, 0};
  }
  issue({call}, comm, false);
  torch::jit::drop(*stack, count);
  torch::jit::push(*stack, result);
}

TORCH_LIBRARY_IMPL(_c10d_functional, MPS, m) {
  for (const char *name : {"all_reduce", "all_reduce_", "broadcast", "broadcast_", "all_gather_into_tensor", "all_gather_into_tensor_out",
                           "reduce_scatter_tensor", "all_to_all_single"})
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&functional>());
}

// The mesh library's own kernels on torch's stream (torch_mesh/sparse.py): a kernel compiled from its source at Metal
// 4.0 (tensor operations), and a dispatch of it encoded in torch's open compute encoder on its queue, ordered with
// torch's own work and waited on by no host: buffers by index (a tensor's storage at its offset), 32-bit constants at
// one index, the grid in threadgroups.
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
  if (handle < 0 || handle >= (int64_t)pipelines_.size()) throw std::runtime_error("pipeline_facts: no kernel " + std::to_string(handle));
  id<MTLComputePipelineState> state = pipelines_[handle];
  return {(int64_t)state.maxTotalThreadsPerThreadgroup, (int64_t)state.threadExecutionWidth,
          (int64_t)state.staticThreadgroupMemoryLength};
}
// A dispatch made ready on the host (encoded): its pipeline, its buffers bound at their offsets, its constants' words
// set at `at`, its grid; encoded into an open encoder on torch's queue (encode_in)
struct Dispatch {
  id<MTLComputePipelineState> state;
  std::vector<id<MTLBuffer>> bound;
  std::vector<NSUInteger> offsets;
  std::vector<uint32_t> words;
  NSUInteger at;
  MTLSize grid, group;
};
static Dispatch encoded(int64_t handle, const std::vector<std::optional<at::Tensor>> &buffers, const std::vector<int64_t> &constants,
                        int64_t at, int64_t gx, int64_t gy, int64_t threads) {
  if (handle < 0 || handle >= (int64_t)pipelines_.size()) throw std::runtime_error("encode: no kernel " + std::to_string(handle));
  Dispatch d{pipelines_[handle], std::vector<id<MTLBuffer>>(buffers.size(), nil), std::vector<NSUInteger>(buffers.size(), 0),
             std::vector<uint32_t>(constants.begin(), constants.end()), (NSUInteger)at, MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, 1),
             MTLSizeMake((NSUInteger)threads, 1, 1)};
  for (size_t i = 0; i < buffers.size(); i++)
    if (buffers[i]) {
      const at::Tensor &t = *buffers[i];
      if (!t.is_mps()) throw std::runtime_error("encode: buffer " + std::to_string(i) + " is not an MPS tensor");
      d.bound[i] = at::native::mps::getMTLBufferStorage(t);
      d.offsets[i] = (NSUInteger)(t.storage_offset() * t.element_size());
    }
  return d;
}
static void encode_in(id<MTLComputeCommandEncoder> encoder, const Dispatch &d) {
  [encoder setComputePipelineState:d.state];
  for (size_t i = 0; i < d.bound.size(); i++)
    if (d.bound[i]) [encoder setBuffer:d.bound[i] offset:d.offsets[i] atIndex:i];
  [encoder setBytes:d.words.data() length:d.words.size() * sizeof(uint32_t) atIndex:d.at];
  [encoder dispatchThreadgroups:d.grid threadsPerThreadgroup:d.group];
}
static void encode(int64_t handle, const std::vector<std::optional<at::Tensor>> &buffers, const std::vector<int64_t> &constants,
                   int64_t at, int64_t gx, int64_t gy, int64_t threads) {
  const Dispatch d = encoded(handle, buffers, constants, at, gx, gy, threads);
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    encode_in(stream->commandEncoder(), d);
  });
}

// torch_mesh::linear (linear.metal): y[R, N] = act(x[R, K] w[N, K]^T (+ bias[N])) of fp16 MPS tensors (act 0 none, 1
// tanh), x the cat along K of x [R, K0] and x1 [R, K - K0] where x1 is given (never made), rows of each any whole stride
// apart, w's rows any whole
// stride apart (w.stride(0)), one dispatch of linear_rows encoded in torch's open compute encoder (encode), its output a
// new tensor of torch's.  Its K split, the simdgroups a threadgroup, is fitted to K, N and the GPU's cores: about 256
// simdgroups a core over the N / 2 threadgroups, a power of two from 1 to 8 (rounded up), each simdgroup at least 1,024
// elements of K.  Its pipelines (by load width
// and split) compile at their first call from the source backend.py hands over as it loads (linear_source).
static std::string linear_source_;
static int64_t linear_cores_ = 0;
static std::map<std::tuple<int, int, int, int>, int64_t> linear_pipelines_;
constexpr int64_t LINEAR_COLUMNS = 2, LINEAR_LOADS = 2;

static void linear_source(const std::string &source, int64_t cores) {
  linear_source_ = source;
  linear_cores_ = cores;
}

static int64_t linear_fit(int64_t K, int64_t groups) {
  int64_t split = 1;
  while (split < 8 && split * groups < linear_cores_ * 256 && K / (split * 2) >= 1024) split *= 2;
  return split;
}
static int64_t linear_split(int64_t K, int64_t N) { return linear_fit(K, (N + LINEAR_COLUMNS - 1) / LINEAR_COLUMNS); }

static at::Tensor linear(const at::Tensor &x, const at::Tensor &w, const std::optional<at::Tensor> &bias, int64_t act,
                         const std::optional<at::Tensor> &x1) {
  const int64_t K0 = x.size(1), K = K0 + (x1 ? x1->size(1) : 0);
  TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf && x.dim() == 2 && w.dim() == 2 && w.size(1) == K &&
                  (!x1 || (x1->scalar_type() == at::kHalf && x1->dim() == 2 && x1->size(0) == x.size(0))),
              "torch_mesh::linear takes fp16 x [R, K0], x1 [R, K - K0] (or none) and w [N, K], not ", x.sizes(), " ", x.scalar_type(),
              ", ", x1 ? x1->sizes() : at::IntArrayRef{}, " and ", w.sizes(), " ", w.scalar_type());
  const at::Tensor xc = x.stride(1) == 1 ? x : x.contiguous(), wc = w.stride(1) == 1 ? w : w.contiguous();
  const at::Tensor x1c = !x1 ? xc : x1->stride(1) == 1 ? *x1 : x1->contiguous();
  const int64_t R = x.size(0), N = w.size(0), ldw = wc.stride(0), ld0 = xc.stride(0), ld1 = x1c.stride(0);
  const int64_t groups = (N + LINEAR_COLUMNS - 1) / LINEAR_COLUMNS;
  const int split = (int)linear_split(K, N);
  const auto key = std::make_tuple(ldw % 4 || K0 % 4 ? 1 : 4, K0 % 4 || ld0 % 4 || ld1 % 4 ? 1 : 4, split, (int)act);
  auto found = linear_pipelines_.find(key);
  if (found == linear_pipelines_.end()) {
    const std::string args = std::to_string(std::get<0>(key)) + ", " + std::to_string(std::get<1>(key)) + ", " +
                             std::to_string(LINEAR_COLUMNS) + ", " + std::to_string(split) + ", " + std::to_string(LINEAR_LOADS) + ", " +
                             std::to_string(act);
    std::string name = "linear_rows_" + args;
    for (auto &ch : name) ch = ch == ',' || ch == ' ' ? '_' : ch;
    found = linear_pipelines_.emplace(key, pipeline(linear_source_ + "\ntemplate [[host_name(\"" + name + "\")]] [[kernel]] decltype(linear_rows<" +
                                                    args + ">) linear_rows<" + args + ">;\n", name)).first;
  }
  at::Tensor y = at::empty({R, N}, x.options());
  encode(found->second, {xc, wc, y, bias ? std::optional<at::Tensor>(bias->contiguous()) : y, std::nullopt, x1c},
         {R, K, N, bias ? 1 : 0, ldw, K0, ld0, ld1}, 4, groups, 1, split * 32);
  return y;
}

// torch_mesh::linears (linear.metal linear_group): up to four of linear's products, one dispatch: each problem's
// threadgroups after the ones before it, its split fitted as linear's to the threadgroups of them all and the least K
// of any, and the narrowest loads any allows.
static std::vector<at::Tensor> linears(const std::vector<at::Tensor> &xs, const std::vector<at::Tensor> &ws,
                                       const std::vector<std::optional<at::Tensor>> &biases, const std::vector<int64_t> &acts,
                                       const std::vector<std::optional<at::Tensor>> &x1s) {
  const size_t P = xs.size();
  TORCH_CHECK(P >= 1 && P <= 4 && ws.size() == P && biases.size() == P && acts.size() == P && x1s.size() == P,
              "torch_mesh::linears takes one to four products, each its x, w, bias, act and x1");
  std::vector<std::optional<at::Tensor>> buffers(20);
  std::vector<int64_t> words(40, 0);
  std::vector<at::Tensor> ys;
  int64_t first = 0, least = INT64_MAX;
  bool wide_w = true, wide_x = true;
  for (size_t p = 0; p < P; p++) {
    const at::Tensor &x = xs[p], &w = ws[p];
    const int64_t K0 = x.size(1), K = K0 + (x1s[p] ? x1s[p]->size(1) : 0);
    TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf && x.dim() == 2 && w.dim() == 2 && w.size(1) == K,
                "torch_mesh::linears: product ", p, " takes fp16 x [R, K0], x1 [R, K - K0] and w [N, K]");
    const at::Tensor xc = x.stride(1) == 1 ? x : x.contiguous(), wc = w.stride(1) == 1 ? w : w.contiguous();
    const at::Tensor x1c = !x1s[p] ? xc : x1s[p]->stride(1) == 1 ? *x1s[p] : x1s[p]->contiguous();
    const int64_t R = x.size(0), N = w.size(0), ldw = wc.stride(0), ld0 = xc.stride(0), ld1 = x1c.stride(0);
    ys.push_back(at::empty({R, N}, x.options()));
    buffers[5 * p] = xc, buffers[5 * p + 1] = wc, buffers[5 * p + 2] = ys.back();
    buffers[5 * p + 3] = biases[p] ? std::optional<at::Tensor>(biases[p]->contiguous()) : ys.back(), buffers[5 * p + 4] = x1c;
    const int64_t d[8] = {R, K, N, biases[p] ? 1 : 0, ldw, K0, ld0, ld1};
    for (int k = 0; k < 8; k++) words[8 * p + k] = d[k];
    words[32 + p] = first, words[36 + p] = acts[p];
    first += (N + LINEAR_COLUMNS - 1) / LINEAR_COLUMNS;
    least = std::min(least, K);
    wide_w &= ldw % 4 == 0 && K0 % 4 == 0, wide_x &= K0 % 4 == 0 && ld0 % 4 == 0 && ld1 % 4 == 0;
  }
  for (size_t p = P; p < 4; p++) {
    for (int k = 0; k < 5; k++) buffers[5 * p + k] = buffers[k];
    words[32 + p] = first;
  }
  const int64_t split = linear_fit(least, first);
  const auto key = std::make_tuple(wide_w ? 4 : 1, wide_x ? 4 : 1, (int)split, -1);
  auto found = linear_pipelines_.find(key);
  if (found == linear_pipelines_.end()) {
    const std::string args = std::to_string(std::get<0>(key)) + ", " + std::to_string(std::get<1>(key)) + ", " +
                             std::to_string(LINEAR_COLUMNS) + ", " + std::to_string(split) + ", " + std::to_string(LINEAR_LOADS);
    std::string name = "linear_group_" + args;
    for (auto &ch : name) ch = ch == ',' || ch == ' ' ? '_' : ch;
    found = linear_pipelines_.emplace(key, pipeline(linear_source_ + "\ntemplate [[host_name(\"" + name + "\")]] [[kernel]] decltype(linear_group<" +
                                                    args + ">) linear_group<" + args + ">;\n", name)).first;
  }
  encode(found->second, buffers, words, 20, first, 1, split * 32);
  return ys;
}

TORCH_LIBRARY_FRAGMENT(torch_mesh, m) {
  m.def("linear(Tensor x, Tensor w, Tensor? bias=None, int act=0, Tensor? x1=None) -> Tensor");
  m.def("linears(Tensor[] x, Tensor[] w, Tensor?[] bias, int[] act, Tensor?[] x1) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(torch_mesh, MPS, m) {
  m.impl("linear", &linear);
  m.impl("linears", &linears);
}

// The same dispatch in a command buffer of its own on torch's queue, committed and waited `repeats` times: each one's
// GPU time (GPUEndTime - GPUStartTime), the measure the engine's kernel harness reports.  Torch's stream is synchronized
// by the caller first.
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

// `count` dispatches of one kernel in one command buffer on torch's queue, each after a buffer barrier, dispatch i bound to
// sets[i % sets.size()]: the command buffer's GPU time over `count`, the measure tools/native_bench (metal-microbench
// 3a2f72a) takes of the recorded LiteRT kernels, so weights rotated over sets totalling more than the system cache are
// read from DRAM by every dispatch.  Torch's stream is synchronized by the caller first.
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
// Optional whole-function microprofile. Markers commit without waiting; read
// after ready() reports completion, without a synchronization. Use Metal's GPU timestamps,
// not the scheduling time of a CPU event callback. Each buffer is retained
// until the profile is destroyed, and repeated reads do not consume a signal.
struct GPUTimings {
  at::mps::MPSStream *const stream = at::mps::getCurrentMPSStream();
  id<MTLSharedEvent> event = [stream->device() newSharedEvent];
  std::vector<id<MTLCommandBuffer>> buffers;
  ~GPUTimings() {
    for (id<MTLCommandBuffer> buffer : buffers) [buffer release];
    [event release];
  }
  void mark() {
    dispatch_sync(stream->queue(), ^{
      @autoreleasepool {
        stream->endKernelCoalescing();
        id<MTLCommandBuffer> buffer = stream->commandBuffer().rootCommandBuffer;
        [buffer encodeSignalEvent:event value:buffers.size() + 1];
        buffers.push_back([buffer retain]);
        stream->synchronize(at::mps::SyncType::COMMIT);
      }
    });
  }
  bool ready() const {
    for (id<MTLCommandBuffer> buffer : buffers) {
      TORCH_CHECK(buffer.status != MTLCommandBufferStatusError, "GPU timing command buffer failed: ",
                  buffer.error ? buffer.error.localizedDescription.UTF8String : "unknown Metal error");
      if (buffer.status != MTLCommandBufferStatusCompleted) return false;
    }
    return !buffers.empty();
  }
  std::vector<double> read() const {
    std::vector<double> out;
    for (id<MTLCommandBuffer> buffer : buffers) {
      TORCH_CHECK(buffer.status == MTLCommandBufferStatusCompleted, "GPU timings require completed function markers");
      const double end = buffer.GPUEndTime;
      TORCH_CHECK(end > 0 && (out.empty() || end >= out.back()), "GPU timing markers are not monotone");
      out.push_back(end);
    }
    return out;
  }
};

// The module (torch_mesh/backend.py, __init__.py and the kernels' modules): `issue` a group of calls, complete when it
// returns; host_empty, address and records the host tensors; the kernels, their timing and the evidence's marks.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<GPUTimings>(module, "GPUTimings")
    .def(pybind11::init<>()).def("mark", &GPUTimings::mark).def("ready", &GPUTimings::ready).def("read", &GPUTimings::read);
  module.def("pipeline", &pipeline);
  module.def("linear_source", &linear_source);
  module.def("device_facts", &device_facts);
  module.def("pipeline_facts", &pipeline_facts);
  module.def("encode", &encode, pybind11::arg("handle"), pybind11::arg("buffers"), pybind11::arg("constants"), pybind11::arg("at"),
             pybind11::arg("gx"), pybind11::arg("gy"), pybind11::arg("threads"));
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
  module.def("issue", [](const std::vector<Call> &calls, int64_t comm) { issue(calls, comm, true); });
  module.def("attach", &attach);
  module.def("host_empty", [](std::vector<int64_t> sizes, const at::Tensor &like, bool mps, bool held) { return host_empty(sizes, like, mps, held); },
             pybind11::arg("sizes"), pybind11::arg("like"), pybind11::arg("mps"), pybind11::arg("held") = false);
  module.def("address", [](const at::Tensor &t) { return (uint64_t)(uintptr_t)host_address(t); });
  module.def("records", &records);
}
