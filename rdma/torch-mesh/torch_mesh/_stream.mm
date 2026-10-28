#include "mesh-write-compiler.h"
// PyTorch frontend bindings to the common collective compiler and native
// Metal schedule. Prepared calls resolve routes, pages and dispatch operands
// before submission. torch's MPS stream is ended where a prepared function or
// an eager collective reads what it stored, and committed where one publishes.
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
#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <array>
#include <vector>
#include <unistd.h>
#include "metal-recording.h"
#include "nccl.h"
#include "mesh-plan.h"

// Framework stream dependencies and compatibility-call storage. The shared
// native schedule owns prepared execution; this state names MPS data events.
struct MeshStream {
  at::mps::MPSStream *const mps;
  id<MTLSharedEvent> const progress, passed;
  id<MTL4CommandQueue> const prepared;
  // the buffers this stream committed, the k-th signalling k on progress as the GPU reaches its end
  uint64_t committed = 0;
  // the prepared functions enqueued from this stream, the k-th signalling k on passed as the prepared queue passes it,
  // each holding what it runs until then
  uint64_t enqueued = 0;
  std::deque<std::pair<uint64_t, std::shared_ptr<void>>> running;
  // whether a group was issued on this stream since the host path last waited for it
  bool streamed = false;
  // the workspace of the group being encoded (the library's workspace allocator's context is the MeshStream, which
  // outlives every call), then each issued group's, by ticket, until the library has retired it (ncclMeshRetired)
  std::vector<at::Tensor> scratch;
  std::map<uint64_t, std::vector<at::Tensor>> holders;

  explicit MeshStream(at::mps::MPSStream *stream) : mps(stream), progress([stream->device() newSharedEvent]),
      passed([stream->device() newSharedEvent]), prepared([stream->device() newMTL4CommandQueue]) {
    if (!progress || !passed || !prepared) {
      [progress release]; [passed release]; [prepared release];
      TORCH_CHECK(false,"preparing Metal context: shared event or command queue unavailable");
    }
  }

  // On torch's queue: the open buffer committed, its end signalling the next count.
  void signalled() {
    mps->endKernelCoalescing();
    [mps->commandBuffer() encodeSignalEvent:progress value:++committed];
    mps->synchronize(at::mps::SyncType::COMMIT);
  }

  // On torch's queue, before a collective sends what torch stored: the open buffer ended and committed, and the next
  // waiting on the GPU for its end (metal-microbench docs/facts.md, 10-24: neither a later kernel of one buffer nor a
  // commit alone makes a kernel's plain stores visible to the NIC).
  void released() {
    signalled();
    [mps->commandBuffer() encodeWaitForEvent:progress value:committed];
  }

  // On the host: the workspace `held` of the group issued as `ticket` kept until the library retires it (0: a group
  // complete in the stream's order once encoded, whose workspace's lease is held again here, where later work reuses
  // it after the group's), and the workspace of every group retired by now released, with what each prepared function
  // the prepared queue has passed held.
  void retire(uint64_t ticket, std::vector<at::Tensor> held) {
    if (ticket) holders[ticket] = std::move(held);
    holders.erase(holders.begin(), holders.upper_bound(ncclMeshRetired()));
    for (const uint64_t at = passed.signaledValue; !running.empty() && running.front().first <= at;) running.pop_front();
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

// Every MeshStream made so far: those of the torch streams a group was issued on (none before any was, so the host
// path makes no MPS stream).
static std::vector<MeshStream *> streams_made() {
  std::lock_guard<std::mutex> guard(*streams_lock_);
  std::vector<MeshStream *> made;
  for (auto &[mps, s] : *streams_) made.push_back(s);
  return made;
}

// No wait of torch_mesh's outlives its deadline: past it the program ends, naming the wait (the operator, 2026-10-28;
// the library's mesh_expired).  No turn of torch's queue taken here waits, so its dispatch_sync waits only on torch's
// own turns, whose GPU waits on a prepared function `watch` bounds.
[[noreturn]] static void expired(const char *wait, uint64_t ns) {
  fprintf(stderr, "%s: past its deadline of %.3f s; the program ends\n", wait, (double)ns / 1e9);
  _exit(1);
}

// The host until `event` reaches `value`, within `ns` (0: MESH_PLATFORM_NS) of `from` (a CLOCK_MONOTONIC reading; 0: now).
static void bounded(id<MTLSharedEvent> event, uint64_t value, const char *wait, uint64_t ns = 0, uint64_t from = 0) {
  ns = ns ? ns : MESH_PLATFORM_NS;
  const uint64_t until = (from ? from : clock_gettime_nsec_np(CLOCK_MONOTONIC)) + ns;
  for (uint64_t now; event.signaledValue < value;) {
    if ((now = clock_gettime_nsec_np(CLOCK_MONOTONIC)) >= until) expired(wait, ns);
    [event waitUntilSignaledValue:value timeoutMS:(until - now + 999999) / 1000000];
  }
}

// Stream s's work so far complete: its buffer committed, then the host's wait for its end, bounded.
static void settled(MeshStream &s, const char *wait, uint64_t ns = 0) {
  MeshStream *self = &s;
  at::mps::dispatch_sync_with_rethrow(s.mps->queue(), ^{ @autoreleasepool { self->signalled(); } });
  bounded(s.progress, s.committed, wait, ns);
}

// Configuration-time migration may read existing tensors produced by ordinary
// GPU stores. End that producer command buffer to establish NIC visibility;
// enqueue the next buffer on the same serial queue, without a CPU/GPU wait.
static void publish_boundary() {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{ s->signalled(); });
}

// The library's workspace allocator: a region lease (region_empty, held: taken again by the next workspace of its size
// once the stream is done with it), kept in the MeshStream's scratch (`context`) until the group is complete in the
// stream's order (MeshStream::retire); NULL where the region refuses one, which fails the group as a value
// (nccl-mesh.c: "out of GPU memory") instead of an exception thrown through the library.  Not torch's caching allocator:
// on unified memory it serves a tensor of about 10 MiB or more from a 1 GiB heap the GPU wires whole (metal-microbench
// docs/facts.md), and a partials reduce-scatter's workspace wired the Mini's last free GiB under cp.  It runs on torch's
// queue, and takes nothing of torch's allocator there (rdma/torch-mesh/README.md, the stream's commit state).
static at::Tensor region_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps, bool held);
static void *workspace(size_t bytes, void *context) {
  auto *s = static_cast<MeshStream *>(context);
  try {
    s->scratch.push_back(region_empty({(int64_t)bytes}, at::empty({0}, at::TensorOptions().dtype(at::kByte)), true, true));
  } catch (...) {
    return nullptr;
  }
  return (__bridge void *)at::native::mps::getMTLBufferStorage(s->scratch.back());
}


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

// A group encoded in one turn of torch's stream queue: torch's stores released to it, its command buffer and open
// encoder, the group encoded into them, and that buffer committed where it published (a peer waits on it).  Its
// workspace is held: split, it is issued and not completed, its ticket (0: complete at once) keeping them until the
// library retires it.
static uint64_t on_stream(const std::function<ncclResult_t(ncclMeshStream *)> &call, bool split, int64_t comm, std::vector<at::Tensor> held) {
  const auto *encode = &call;
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  s->scratch = std::move(held);
  __block ncclResult_t result = ncclSuccess;
  __block ncclMeshIssue issue = {0, 0};
  {
    dispatch_sync(stream->queue(), ^{
      @autoreleasepool {
        s->released();
        ncclMeshStream on{(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder(), workspace, s, split ? &issue : nullptr};
        result = (*encode)(&on);
        if (!result && (!split || issue.published)) s->mps->synchronize(at::mps::SyncType::COMMIT);
      }
    });
  }
  s->streamed = true;
  s->retire(issue.ticket && !result ? issue.ticket : 0, std::exchange(s->scratch, {}));
  checked(result);
  return issue.ticket;
}

// every issued group through `ticket` completed on torch's stream where it is waited
static void complete(uint64_t ticket) {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  if (ncclMeshRetired() >= ticket) return s->retire(0, {});
  __block ncclResult_t result = ncclSuccess;
  {
    dispatch_sync(stream->queue(), ^{
      @autoreleasepool {
        ncclMeshStream on{(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder(), workspace, s, nullptr};
        int published = 0;
        result = ncclMeshComplete(ticket, &on, &published);
        if (!result && published) s->mps->synchronize(at::mps::SyncType::COMMIT);
      }
    });
  }
  s->retire(0, std::exchange(s->scratch, {}));
  checked(result);
}

// c10d's Work for a group issued and not completed: waiting (or synchronizing, asking whether it completed, or
// asking for its future) encodes the rest on torch's current stream, where every later kernel reads its results; a
// c10d completion follows the submitted stream's dependency.  A wait given a timeout (c10d's kNoTimeout is 0) has the
// host wait for the stream within it too, as NCCL's blocking wait does.
class MeshWork : public c10d::Work {
 public:
  MeshWork(uint64_t ticket, std::vector<at::Tensor> outputs, int64_t comm)
      : c10d::Work(-1, c10d::OpType::UNKNOWN), ticket_(ticket), outputs_(std::move(outputs)), comm_(comm) {}
  bool isCompleted() override { finish(); return true; }
  bool isSuccess() const override { return finished_; }
  bool wait(std::chrono::milliseconds timeout) override {
    finish();
    if (timeout.count() > 0)
      settled(stream_of(at::mps::getCurrentMPSStream()), "MeshWork::wait: its group's completion on torch's MPS stream",
              (uint64_t)timeout.count() * 1000000);
    return true;
  }
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

static at::Tensor region_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps, bool held = false);

// One call of a group: its kind, its send and receive tensors (an all-to-all's none where it moves none that way; an
// all-reduce's, reduce's or broadcast's sends alone where it is in place), its counts (a V call's segment a rank, an
// all-to-all's elements to each rank then from each, in its tensors' elements), its operator (an ncclRedOp_t: a
// built-in or one the library made, as the partials merge) and its root (a send's or receive's peer).
enum Kind : int64_t { ALLREDUCE, REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL };
using Tensors = std::vector<std::optional<at::Tensor>>;
using Call = std::tuple<int64_t, Tensors, Tensors, std::vector<int64_t>, int64_t, int64_t>;

// A tensor as the library reads it, with ProcessGroupNCCL's dtype semantics: a collective that moves data moves any
// dtype (one NCCL lacks as its bytes); a reduction reduces a complex tensor as its real view (sum and average) and a bool
// tensor's sum and product as max and min (its average the library's uint8 average: the sum truncated by the ranks, 1
// only where every rank holds 1).  `at` is an ncclMeshBuffer (its storage's MTLBuffer and byte offset) on the
// Metal path and a host pointer on the host path (NULL: none).
struct Operand {
  const void *at = nullptr;
  size_t count = 0;
  ncclDataType_t type = ncclUint8;
};

// A group's prepared fragments. The shared view lowerer resolves logical page
// selections and typed record boundaries before these native scalar bindings.
struct Group {
  bool metal = false;
  std::deque<ncclMeshBuffer> buffers;

  const void *where(const at::Tensor &t) {
    if (!t.is_mps()) return t.data_ptr();
    buffers.push_back({(__bridge void *)at::native::mps::getMTLBufferStorage(t), (size_t)(t.storage_offset() * t.element_size())});
    return &buffers.back();
  }
  const at::Tensor &contiguous(const at::Tensor &t) {
    TORCH_CHECK(t.is_mps() == metal, "the mesh backend's group takes MPS tensors or CPU tensors, not both");
    TORCH_CHECK(t.is_contiguous(), "the mesh backend's operand is a contiguous region tensor (torch_mesh.empty); a non-contiguous "
                "one is not copied into one (zero copy: nccl.h Memory)");
    return t;
  }
  Operand moving(const at::Tensor &tensor) {
    const at::Tensor &t = contiguous(tensor);
    const ncclDataType_t type = kind(t.scalar_type());
    return type == ncclNumTypes ? Operand{where(t), (size_t)t.nbytes(), ncclUint8} : Operand{where(t), (size_t)t.numel(), type};
  }
  Operand reducing(const at::Tensor &tensor, int64_t &op) {
    const at::Tensor &t = contiguous(tensor);
    if (t.scalar_type() == at::kBool) op = op == ncclSum ? ncclMax : op == ncclProd ? ncclMin : op;
    if (t.is_complex()) {
      TORCH_CHECK(op == ncclSum || op == ncclAvg, "a complex tensor reduces by sum and average alone");
      return {where(t), (size_t)t.numel() * 2, kind(at::toRealValueType(t.scalar_type()))};
    }
    TORCH_CHECK(kind(t.scalar_type()) != ncclNumTypes, "the mesh backend reduces no ", t.scalar_type());
    return {where(t), (size_t)t.numel(), kind(t.scalar_type())};
  }
};

// A call's prepared operands and counts, made before its group is encoded.
struct Issue {
  int64_t kind, op, root;
  std::vector<Operand> sends, receives;
  std::vector<size_t> counts;
  std::vector<void *> source_views, destination_views;
  ncclDataType_t type = ncclUint8;
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
    // MPI_Allgatherv and MPI_Reduce_scatter (nccl.h ncclMeshAllGatherV, ncclMeshReduceScatterV): rank r's segment is
    // counts[r] elements, the segments packed in rank order in the whole operand (the receive's, the send's), counted
    // in the operand's NCCL elements; equal counts are the library's equal program
    const bool reduces = what == REDUCE_SCATTER;
    const at::Tensor &whole = reduces ? tensor(sends, 0) : tensor(receives, 0);
    for (size_t rank=0;rank<sends.size();rank++)
      i.sends.push_back(reduces ? g.reducing(tensor(sends,rank),i.op) : g.moving(tensor(sends,rank)));
    if (reduces) i.receives = {g.reducing(tensor(receives, 0), i.op)};
    else for (size_t rank = 0; rank < receives.size(); rank++) i.receives.push_back(g.moving(tensor(receives, rank)));
    const size_t per = reduces && whole.is_complex() ? 2 : kind(whole.scalar_type()) == ncclNumTypes ? whole.element_size() : 1;
    for (auto c : counts) i.counts.push_back((size_t)c * per);
    break;
  }
  case ALLTOALL: {
    size_t per=1;
    for(const Tensors *list : {&sends,&receives}) {
      auto &bindings=list==&sends?i.sends:i.receives;
      for(const auto &tensor:*list) {
        bindings.push_back(tensor?g.moving(*tensor):Operand{});
        if(tensor) {
          i.type=bindings.back().type;
          per=kind(tensor->scalar_type())==ncclNumTypes?tensor->element_size():1;
        }
      }
    }
    for(auto count:counts)i.counts.push_back((size_t)count*per);
    break;
  }
  default: TORCH_CHECK(false, "no collective of kind ", what);
  }
  if ((what == ALLREDUCE || what == REDUCE || what == BROADCAST) && !counts.empty()) {
    const auto &t = tensor(sends,0);
    const size_t per = what != BROADCAST && t.is_complex() ? 2 :
      kind(t.scalar_type()) == ncclNumTypes ? t.element_size() : 1;
    i.sends.front().count = (size_t)counts[0] * per;
  }
  for(const auto &view:i.sends)i.source_views.push_back((void *)view.at);
  for(const auto &view:i.receives)i.destination_views.push_back((void *)view.at);
  return i;
}

// The group's calls on stream s (NULL: the host path), one NCCL group.
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
    case ALLGATHER:
      if (i.receives.size() == 1) result = ncclMeshAllGatherV(a.at, b, i.counts.data(), a.type, comm, s);
      else result = ncclMeshAllGatherBuffers(a.at, i.destination_views.data(), i.counts.data(), a.type, comm, s);
      break;
    case REDUCE_SCATTER:
      result = i.sends.size()==1 ? ncclMeshReduceScatterV(a.at,b,i.counts.data(),a.type,op,comm,s)
        : ncclMeshReduceScatterBuffers(i.source_views.data(),b,i.counts.data(),a.type,op,comm,s);
      break;
    case ALLTOALL:
      result = i.sends.size()==1 && i.receives.size()==1
        ? ncclMeshAlltoAllV(a.at,i.counts.data(),b,i.counts.data()+i.counts.size()/2,i.type,comm,s)
        : ncclMeshAlltoAllBuffers(i.source_views.data(),i.counts.data(),i.destination_views.data(),
                                 i.counts.data()+i.counts.size()/2,i.type,comm,s);
      break;
    }
  }
  const ncclResult_t ended = ncclGroupEnd();
  return result ? result : ended;
}

// One group of `calls` on comm: on MPS tensors encoded on torch's stream, issued and not completed where `split` (its
// ticket; 0 complete); on CPU tensors the host path, first completing every group
// issued on torch's streams (ncclMeshComplete encodes every issued group's completion on the current one) and waiting
// for each stream a group was issued on (the session's positions run in its order), the GIL released while it runs
// where `unlock`.
static uint64_t submit(const std::function<ncclResult_t(ncclMeshStream *)> &call, bool metal,
                       bool split, int64_t comm, std::vector<at::Tensor> held, bool unlock) {
  if (metal) return on_stream(call, split, comm, std::move(held));
  {
    bool streamed = false;
    for (MeshStream *s : streams_made()) streamed |= s->streamed;
    if (streamed) {
      complete(UINT64_MAX);
      const MeshStream *current = &stream_of(at::mps::getCurrentMPSStream());
      for (MeshStream *s : streams_made())
        if (s->streamed || s == current) {
          settled(*s, "torch_mesh: torch's MPS stream, before a host-path collective (submit)");
          s->retire(0, {});
          s->streamed = false;
        }
    }
    std::optional<pybind11::gil_scoped_release> unlocked;
    if (unlock) unlocked.emplace();
    checked(call(nullptr));
  }
  return 0;
}

static uint64_t issue(const std::vector<Call> &calls, int64_t comm, bool split, bool unlock) {
  Group g;
  for (const Call &c : calls)
    for (const Tensors *list : {&std::get<1>(c), &std::get<2>(c)})
      for (const std::optional<at::Tensor> &t : *list) g.metal |= t && t->is_mps();
  std::vector<Issue> issues;
  for (const Call &c : calls) issues.push_back(prepare(c, g));
  return submit([&](ncclMeshStream *stream) { return encoded(issues, communicator(comm), stream); },
                g.metal, split, comm, {}, unlock);
}

// A prepared function on its stream's prepared queue: its inputs wait for `gate` (its GPU part and the host steps that
// follow no part), which torch's queue signals 1 where it is enqueued; a function with a host step signals 2 as it ends.
struct Schedule {
  std::shared_ptr<MetalSchedule> function;
  id<MTLSharedEvent> gate = nil;  // the function's (MetalScheduleAfter holds it)
  bool host = false;
  MetalSchedule *get() const { return function.get(); }
};

static Schedule schedule_after_inputs() {
  MeshStream *source=&stream_of(at::mps::getCurrentMPSStream());
  id<MTLSharedEvent> gate=[source->mps->device() newSharedEvent];
  Schedule schedule{{MetalScheduleCreateOn(source->prepared,MetalFormEncoded),MetalScheduleFree},gate};
  if(schedule.function&&gate)MetalScheduleAfter(schedule.get(),gate,1);
  [gate release];
  TORCH_CHECK(schedule.function&&gate,"creating the prepared function command stream");
  return schedule;
}

// Each function `ordered` enqueued watched off the host's path (torch's later work waits for it on the GPU): its gate
// within MESH_PLATFORM_NS of the later of its enqueue and the previous one's pass, and its pass within MESH_PLATFORM_NS
// of its gate, or the program ends.
static void watch(id<MTLSharedEvent> gate, id<MTLSharedEvent> passed, uint64_t at) {
  static dispatch_queue_t watching = dispatch_queue_create("torch_mesh.passed", DISPATCH_QUEUE_SERIAL);
  static uint64_t last = 0;  // the previous function's pass, read and written on `watching`
  const uint64_t enqueued = clock_gettime_nsec_np(CLOCK_MONOTONIC);
  dispatch_async(watching, ^{
    bounded(gate, 1, "torch_mesh: torch's MPS work before a prepared function (its gate, ordered)", 0, std::max(last, enqueued));
    bounded(passed, at, "torch_mesh: a prepared function's run on the prepared queue (ordered)");
    last = clock_gettime_nsec_np(CLOCK_MONOTONIC);
  });
}

// A prepared function enqueued in torch's stream order, with no host wait: its gate signalled at the end of torch's
// buffer so far (committed, so what torch stored is visible to it and the NIC: metal-microbench docs/facts.md, 10-24),
// and torch's later work waiting on the GPU for the prepared queue to pass it; it and `keep` held until then.
static void ordered(const Schedule &schedule, std::shared_ptr<void> keep = {}) {
  at::mps::MPSStream *stream=at::mps::getCurrentMPSStream();
  MeshStream *s=&stream_of(stream);
  s->retire(0,{});
  MetalSchedule *function=schedule.get();
  id<MTLSharedEvent> gate=schedule.gate;
  const bool host=schedule.host;
  const uint64_t at=++s->enqueued;
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{ @autoreleasepool {
    s->mps->endKernelCoalescing();
    [s->mps->commandBuffer() encodeSignalEvent:gate value:1];
    s->mps->synchronize(at::mps::SyncType::COMMIT);
    MetalScheduleEnqueue(function);
    if(host)[s->prepared waitForEvent:gate value:2];
    [s->prepared signalEvent:s->passed value:at];
    [s->mps->commandBuffer() encodeWaitForEvent:s->passed value:at];
  }});
  watch(gate,s->passed,at);
  s->running.emplace_back(at,schedule.function);
  if(keep)s->running.emplace_back(at,std::move(keep));
}

// The next of a function's invocations prepared ahead, each enqueued once, and the last enqueued.
static const Schedule &next_of(const std::vector<Schedule> &prepared, size_t &next) {
  TORCH_CHECK(next < prepared.size(), "enqueue past the ", prepared.size(), " prepared invocations: realize more first");
  return prepared[next++];
}
static const Schedule &last_of(const std::vector<Schedule> &prepared, size_t next) {
  TORCH_CHECK(next, "no prepared invocation has been enqueued");
  return prepared[next-1];
}

static pybind11::capsule schedule_completion(const Schedule &schedule) {
  return pybind11::capsule(new Schedule(schedule),"mesh.prepared-function",[](PyObject *token) {
    delete static_cast<Schedule *>(PyCapsule_GetPointer(token,"mesh.prepared-function"));
  });
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

// Issued and not completed (split) where the result is its own operand: torch's wait_tensor (its work registry) waits
// for it.  A result made here is a region tensor (region_empty): the library sends from and lands in leases alone (nccl.h Memory).
// An all-reduce's or a broadcast's is a held lease, taken again by the next result of its size once the stream is done
// with it, so a program's steady steps lease nothing.
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
    result = name.back() == '_' ? input : region_empty(input.sizes().vec(), input, true, true).copy_(input);
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

// The mesh library's own kernels on torch's stream (torch_mesh/sparse.py): a kernel compiled from its source at Metal
// 4.0 (tensor operations), and a dispatch of it encoded in torch's open compute encoder on its queue, ordered with
// torch's own work and waited on by no host: buffers by index (a tensor's storage at its offset), 32-bit constants at
// one index, the grid in threadgroups.
static std::vector<id<MTLComputePipelineState>> pipelines_;

static int64_t pipeline(const std::string &source, const std::string &name) {
  id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
  MTLCompileOptions *options = MetalDeviceCompilerOptions(device);
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:@(source.c_str()) options:options error:&error];
  if (!library) throw std::runtime_error(std::string("kernel source: ") + error.localizedDescription.UTF8String);
  id<MTLFunction> function = [library newFunctionWithName:@(name.c_str())];
  if (!function) throw std::runtime_error("kernel source: no function " + name);
  id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
  if (!state) throw std::runtime_error(std::string("kernel pipeline: ") + error.localizedDescription.UTF8String);
  [function release]; [library release];
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

// A kernel that runs a fused crossing's steps (`fused`) reaches the library's counts, landing buffers and records by
// address, through the residency set, which Metal does not hazard-track: its encoder declares every buffer of the set,
// so its pass is ordered against the library's dispatches that name them (nccl-mesh-metal.h metal_uses).
// A dispatch made ready on the host (encoded): its pipeline, its buffers bound at their offsets, its constants' words
// set at `at`, its grid; encoded into an open encoder on torch's queue (encode_in)
struct Dispatch {
  id<MTLComputePipelineState> state;
  std::vector<id<MTLBuffer>> bound;
  std::vector<NSUInteger> offsets, lengths;
  std::vector<uint32_t> words;
  NSUInteger at;
  MTLSize grid, group;
};
static Dispatch encoded(int64_t handle, const std::vector<std::optional<at::Tensor>> &buffers, const std::vector<int64_t> &constants,
                        int64_t at, int64_t gx, int64_t gy, int64_t threads) {
  if (handle < 0 || handle >= (int64_t)pipelines_.size()) throw std::runtime_error("encode: no kernel " + std::to_string(handle));
  Dispatch d{pipelines_[handle], std::vector<id<MTLBuffer>>(buffers.size(), nil), std::vector<NSUInteger>(buffers.size(), 0), std::vector<NSUInteger>(buffers.size(), 0),
             std::vector<uint32_t>(constants.begin(), constants.end()), (NSUInteger)at, MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, 1),
             MTLSizeMake((NSUInteger)threads, 1, 1)};
  for (size_t i = 0; i < buffers.size(); i++)
    if (buffers[i]) {
      const at::Tensor &t = *buffers[i];
      if (!t.is_mps()) throw std::runtime_error("encode: buffer " + std::to_string(i) + " is not an MPS tensor");
      d.bound[i] = at::native::mps::getMTLBufferStorage(t);
      d.offsets[i] = (NSUInteger)(t.storage_offset() * t.element_size());
      d.lengths[i] = t.nbytes();
    }
  return d;
}

// The shared preparation layout is also the Python frontend's layout type.
// Its object and maps never enter a replay record.
static NSArray<NSNumber *> *tensorNumbers(const std::vector<NSInteger> &values) {
  NSMutableArray *result=[NSMutableArray array];for(auto value:values)[result addObject:@(value)];return result;
}
static std::vector<NSInteger> tensorNumbers(NSArray<NSNumber *> *values) {
  std::vector<NSInteger> result;for(NSNumber *value in values)result.push_back(value.integerValue);return result;
}
static NSData *tensorWords(const std::vector<NSInteger> &values) {
  return [NSData dataWithBytes:values.data() length:values.size()*sizeof(NSInteger)];
}
struct TensorLayout {
  MetalTensorLayout *value;
  explicit TensorLayout(MetalTensorLayout *layout):value([layout retain]){}
  TensorLayout(const TensorLayout &other):value([other.value retain]){}
  TensorLayout(TensorLayout &&other):value(other.value){other.value=nil;}
  ~TensorLayout(){[value release];}
  static TensorLayout make(const std::vector<NSInteger> &shape,const std::vector<NSInteger> &strides,NSInteger offset,
      const std::map<NSUInteger,std::vector<NSInteger>> &axes,const std::optional<std::vector<NSInteger>> &indices) {
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];
    for(const auto &[axis,values]:axes)maps[@(axis)]=tensorWords(values);
    return TensorLayout([MetalTensorLayout shape:tensorNumbers(shape) strides:tensorNumbers(strides) offset:offset
        maps:maps indices:indices?tensorWords(*indices):nil]);
  }
  static TensorLayout tensor(const at::Tensor &t) {
    return make({t.sizes().begin(),t.sizes().end()},{t.strides().begin(),t.strides().end()},t.storage_offset(),{},std::nullopt);
  }
};
// Resolve a frontend storage handle to its bridge pages once, while preparing.
// A tensor containing changing page indices remains an ordinary data operand;
// neither this adapter nor TensorLayout reads those indices from its payload.
static MetalTensorPages *tensorPages(const at::Tensor &t) {
  TORCH_CHECK(t.is_mps()||t.device().is_cpu(),"page operands require CPU or Metal views of bridge storage");
  @try {
    return [(t.is_mps()?[MetalTensorPages storage:at::native::mps::getMTLBufferStorage(t)]
                       :[MetalTensorPages host:t.storage().data_ptr().get()]) retain];
  } @catch(NSException *error) { TORCH_CHECK(false,error.reason.UTF8String); }
}
struct TensorOperand {
  at::Tensor storage;
  MetalTensorPages *pages;
  TensorLayout layout;
  explicit TensorOperand(const pybind11::object &object):
    storage((THPVariable_Check(object.ptr())?object:object.attr("storage")).cast<at::Tensor>()),
    pages(tensorPages(storage)),
    layout([pages bind:(THPVariable_Check(object.ptr())?TensorLayout::tensor(storage):object.attr("layout").cast<TensorLayout>()).value
                 elementBytes:storage.element_size()]){}
  ~TensorOperand(){[pages release];}
};

// FX's output is the native recorder's input. This constructor only binds
// storage and constant operands; it neither runs nor records a framework call.
using CommandRecord = std::tuple<int64_t, std::vector<pybind11::object>,
    std::map<uint32_t,pybind11::bytes>, std::array<NSUInteger,3>, std::array<NSUInteger,3>, bool, uint64_t>;
static std::vector<pybind11::object> commandOperands(const std::vector<std::optional<at::Tensor>> &tensors) {
  std::vector<pybind11::object> result;for(const auto &t:tensors)result.push_back(t?pybind11::cast(*t):pybind11::none());return result;
}
struct Commands {
  MetalReplay *replay=nullptr;
  std::vector<at::Tensor> operands;
  explicit Commands(const std::vector<CommandRecord> &records) {
    if(records.empty())return;
    auto device=at::mps::MPSDevice::getInstance()->device();
    std::vector<std::vector<MetalCommandBinding>> bindings(records.size());
    std::vector<std::vector<MetalReach>> reaches(records.size());
    std::vector<MetalCommandDescription> descriptions;
    std::vector<uint8_t> words;
    std::vector<std::vector<std::array<NSUInteger,3>>> constants(records.size());
    for(size_t k=0;k<records.size();k++)for(const auto &[index,scalar]:std::get<2>(records[k])) {
      const std::string bytes=scalar;
      words.resize((words.size()+MESH_COMMAND_DATA_ALIGNMENT-1)/MESH_COMMAND_DATA_ALIGNMENT*MESH_COMMAND_DATA_ALIGNMENT);
      constants[k].push_back({index,words.size(),bytes.size()});
      words.insert(words.end(),bytes.begin(),bytes.end());
    }
    id<MTLBuffer> constantBuffer=nil;size_t constantOffset=0;
    if(!words.empty()) {
      auto storage=region_empty({(int64_t)words.size()},at::empty({0},at::TensorOptions().dtype(at::kByte)),false);
      memcpy(storage.data_ptr(),words.data(),words.size());
      void *mapped=nullptr;checked(ncclMeshMemBuffer(storage.data_ptr(),&mapped,&constantOffset));
      constantBuffer=(id<MTLBuffer>)mapped;operands.push_back(std::move(storage));
    }
    for(size_t k=0;k<records.size();k++) {
      const auto &[handle,buffers,scalars,grid,group,threadGrid,access]=records[k];
      TORCH_CHECK(handle>=0&&handle<(int64_t)pipelines_.size(),"unprepared Metal kernel");
      auto &bound=bindings[k];
      auto &ranges=reaches[k];
      for(size_t j=0;j<buffers.size();j++)if(!buffers[j].is_none()) {
        const TensorOperand operand(buffers[j]);const auto &t=operand.storage;const auto &layout=operand.layout;
        id<MTLBuffer> buffer=operand.pages.buffer;
        const uint32_t mode=((access>>j)&1?MetalReachRead:0)|((access>>(MESH_ACCESS_WRITE_SHIFT+j))&1?MetalReachWritten:0);
        bound.push_back({buffer,(NSUInteger)layout.value.offset*t.element_size(),layout.value.count*t.element_size(),(uint32_t)j,mode});
        if(mode) {
          NSData *data=[layout.value reaches:buffer base:0 elementBytes:t.element_size() access:mode binding:(int32_t)j];
          const auto *spans=(const MetalReach *)data.bytes;
          ranges.insert(ranges.end(),spans,spans+data.length/sizeof *spans);
        }
        operands.push_back(t);
      }
      for(const auto &constant:constants[k])
        bound.push_back({constantBuffer,constantOffset+constant[1],constant[2],(uint32_t)constant[0],MetalReachRead});
      descriptions.push_back({pipelines_[handle],bound.data(),bound.size(),nullptr,0,ranges.data(),ranges.size(),
        MTLSizeMake(grid[0],grid[1],grid[2]),MTLSizeMake(group[0],group[1],group[2]),(BOOL)threadGrid});
    }
    replay=MetalReplayCommands(device,descriptions.data(),descriptions.size(),MetalFormEncoded);
    TORCH_CHECK(replay,"native command preparation failed");
  }
  ~Commands(){MetalReplayFree(replay);}
  void append(MetalSchedule *schedule) {
    if(!replay)return;
    NSError *error=MetalScheduleAppend(schedule,replay,NSMakeRange(0,1),@[],nil);
    TORCH_CHECK(!error,error?error.localizedDescription.UTF8String:"");
  }
};
// Fixed operands for a declared block graph. Shared lowering supplies scalar
// fragments and any required numerical conversion edges. They record through
// the same native schedule; invocation creates no layout or tensor bindings.
struct Collectives {
  int64_t comm;
  Group group{true};
  std::vector<Issue> issues;
  std::vector<at::Tensor> operands;
  Commands before, after;
  Collectives(const std::vector<Call> &calls, int64_t comm, std::vector<at::Tensor> operands,
              const std::vector<CommandRecord> &inputs, const std::vector<CommandRecord> &outputs)
      : comm(comm), operands(std::move(operands)), before(inputs), after(outputs) {
    this->operands.insert(this->operands.end(),before.operands.begin(),before.operands.end());
    this->operands.insert(this->operands.end(),after.operands.begin(),after.operands.end());
    for (const Call &call : calls) issues.push_back(prepare(call, group));
  }
  std::vector<Schedule> executions;
  size_t next=0;
  void append(MetalSchedule *schedule,size_t repetitions=1) {
    const auto *list=&issues;const ncclComm_t handle=communicator(comm);
    MetalScheduleCommunicator(schedule,handle);
    const size_t batches=before.replay||after.replay?repetitions:(size_t)(repetitions!=0);
    const size_t repeat=batches==1?repetitions:1;
    for(size_t k=0;k<batches;k++) {
      before.append(schedule);
      checked(MetalScheduleCross(schedule,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
        id<MTLResidencySet> residency,NSUInteger,NSUInteger) {
      ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
        .residencySet=(__bridge void *)residency};
      for(size_t r=0;r<repeat;r++) {
        const int result=(int)encoded(*list,handle,&on);
        if(result)return result;
      }
      return 0;
    }));
      after.append(schedule);
    }
  }
  void realize(size_t repetitions=1) {
    auto made=schedule_after_inputs();append(made.get(),repetitions);
    const auto held=operands;
    checked(MetalScheduleFinish(made.get(),^(NSError *){(void)held;}));executions.push_back(std::move(made));
  }
  void enqueue() { ordered(next_of(executions,next)); }
  pybind11::capsule completion() { return schedule_completion(last_of(executions,next)); }

};

static void encode_in(id<MTLComputeCommandEncoder> encoder, const Dispatch &d) {
  [encoder setComputePipelineState:d.state];
  for (size_t i = 0; i < d.bound.size(); i++)
    if (d.bound[i]) [encoder setBuffer:d.bound[i] offset:d.offsets[i] atIndex:i];
  [encoder setBytes:d.words.data() length:d.words.size() * sizeof(uint32_t) atIndex:d.at];
  [encoder dispatchThreadgroups:d.grid threadsPerThreadgroup:d.group];
}
// Configuration binds kernels from the Torch frontend into the native recorder.
// Access modes are explicit operands; the recorder orders the declared byte views.
static void prepare_dispatch(MetalSchedule *schedule,const Dispatch &d,uint64_t access) {
  std::vector<MetalCommandBinding> bindings;
  for(size_t j=0;j<d.bound.size();j++)if(d.bound[j])
    bindings.push_back({d.bound[j],d.offsets[j],d.lengths[j],(uint32_t)j,
      (uint32_t)(((access>>j)&1?MetalReachRead:0)|((access>>(MESH_ACCESS_WRITE_SHIFT+j))&1?MetalReachWritten:0))});
  bindings.push_back(MetalScheduleData(schedule,d.words.data(),d.words.size()*sizeof(uint32_t),(uint32_t)d.at));
  MetalScheduleCompute(schedule,d.state,bindings.data(),bindings.size(),d.grid,d.group);

}

static void encode(int64_t handle, const std::vector<std::optional<at::Tensor>> &buffers, const std::vector<int64_t> &constants,
                   int64_t at, int64_t gx, int64_t gy, int64_t threads) {
  const Dispatch d = encoded(handle, buffers, constants, at, gx, gy, threads);
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    encode_in(encoder, d);
  });
}

// Linear command geometry is compiled from the shared manifest and discovered GPU cores.
static std::string linear_source_;
static int64_t linear_cores_ = 0;
static std::map<std::tuple<int, int, int, int, int>, int64_t> linear_pipelines_;
static constexpr const char *LINEAR_PARAMETERS[]={"device const half* x", "device const half* w", "device half* y", "device const half* b", "device const half* s"};
constexpr size_t LINEAR_BINDINGS = sizeof(LINEAR_PARAMETERS) / sizeof(*LINEAR_PARAMETERS);
constexpr size_t LINEAR_PRODUCTS = (MESH_BUFFER_BINDINGS - 1) / LINEAR_BINDINGS;
struct LinearDimensions { uint32_t rows, inner, columns, bias, weight_stride, first, stride0, stride1, output_row, output_column; };
static_assert(sizeof(LinearDimensions) == 10 * sizeof(uint32_t), "linear.metal's linear_dims, as prepare_chain binds it too");
struct LinearGroupWords {
  LinearDimensions dimensions[LINEAR_PRODUCTS];
  uint32_t first[LINEAR_PRODUCTS], activation[LINEAR_PRODUCTS];
};

static void linear_source(const std::string &source, int64_t cores) {
  linear_source_ = source;
  linear_cores_ = cores;
}

static int64_t linear_fit(int64_t K, int64_t groups) {
  int64_t split = 1;
  while (split * 2 <= MESH_LINEAR_SIMDGROUPS && split * groups < linear_cores_ * MESH_LINEAR_GROUPS_PER_CORE &&
         K / (split * 2) >= MESH_LINEAR_MIN_INNER) split *= 2;
  return split;
}
static int64_t linear_groups(int64_t rows, int64_t columns) {
  return ((rows + MESH_LINEAR_ROWS - 1) / MESH_LINEAR_ROWS) * ((columns + MESH_LINEAR_COLUMNS - 1) / MESH_LINEAR_COLUMNS);
}
static int64_t linear_split(int64_t K, int64_t N, int64_t rows) { return linear_fit(K, linear_groups(rows, N)); }

static std::string linear_group_source(size_t products) {
  const std::string count=std::to_string(products);
  std::string source="\nstruct linear_group_words { linear_dims d["+count+"]; uint first["+count+"], act["+count+"]; };\n"
    "template <uint VW, uint VX, uint COLS, uint NSG, uint U>\n"
    "[[max_total_threads_per_threadgroup(NSG * MESH_SIMD_WIDTH)]] kernel void linear_group(";
  for(size_t p=0;p<products;p++) {
    const std::string n=std::to_string(p);
    for(size_t k=0;k<LINEAR_BINDINGS;k++)source+=std::string(LINEAR_PARAMETERS[k])+n+" [[buffer("+std::to_string(LINEAR_BINDINGS*p+k)+")]],";
  }
  source+="constant linear_group_words& g [[buffer("+std::to_string(products*LINEAR_BINDINGS)+")]],"
    "uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],"
    "uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {"
    "threadgroup float part[NSG*MESH_LINEAR_ROWS*COLS];";
  for(size_t p=products;p-- > 0;) {
    const std::string n=std::to_string(p);
    source+="if(tg>=g.first["+n+"]) { const linear_act_store store{y"+n+",b"+n+",g.d["+n+"].LDY,g.d["+n+"].SDY,g.d["+n+"].B,g.act["+n+"]};"
      "linear_body<VW,VX,COLS,NSG,U>(x"+n+",s"+n+",w"+n+",g.d["+n+"],store,part,sg,lane,tg-g.first["+n+"],tid);return;}";
  }
  return source+"}\n";
}

static int64_t linear_pipeline(const std::tuple<int, int, int, int, int> &key) {
  const auto found = linear_pipelines_.find(key);
  if (found != linear_pipelines_.end()) return found->second;
  const int64_t split = std::get<2>(key), act = std::get<3>(key);
  std::string args = std::to_string(std::get<0>(key)) + ", " + std::to_string(std::get<1>(key)) + ", " +
                     std::to_string(MESH_LINEAR_COLUMNS) + ", " + std::to_string(split) + ", " + std::to_string(MESH_LINEAR_LOADS);
  const std::string kernel = act < 0 ? "linear_group" : "linear_rows";
  if (act >= 0) args += ", " + std::to_string(act);
  std::string name = kernel + "_" + args + "_p" + std::to_string(std::get<4>(key));
  for (auto &ch : name) ch = ch == ',' || ch == ' ' ? '_' : ch;
  return linear_pipelines_.emplace(key, pipeline(linear_source_ + (act < 0 ? linear_group_source(std::get<4>(key)) : "") + "\ntemplate [[host_name(\"" + name + "\")]] [[kernel]] decltype(" +
                                         kernel + "<" + args + ">) " + kernel + "<" + args + ">;\n", name)).first->second;
}

// Independent products share a dispatch up to the manifest binding capacity.
// Row tiles and column tiles both contribute independent groups.
static std::vector<CommandRecord> linear_commands(const std::vector<at::Tensor> &xs, const std::vector<at::Tensor> &ws,
                                       const std::vector<std::optional<at::Tensor>> &biases, const std::vector<int64_t> &acts,
                                       const std::vector<std::optional<at::Tensor>> &x1s, const std::vector<at::Tensor> &ys) {
  const size_t P = xs.size();
  TORCH_CHECK(P >= 1 && P <= LINEAR_PRODUCTS && ws.size() == P && biases.size() == P && acts.size() == P && x1s.size() == P && ys.size() == P,
              "torch_mesh::linears exceeds the manifest binding capacity or lacks product operands");
  std::vector<std::optional<at::Tensor>> buffers(LINEAR_BINDINGS * P);
  LinearGroupWords words{};
  int64_t first = 0, least = INT64_MAX;
  bool wide_w = true, wide_x = true;
  for (size_t p = 0; p < P; p++) {
    const at::Tensor &x = xs[p], &w = ws[p];
    const int64_t K0 = x.size(1), K = K0 + (x1s[p] ? x1s[p]->size(1) : 0);
    TORCH_CHECK(x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf && x.dim() == 2 && w.dim() == 2 && w.size(1) == K,
                "torch_mesh::linears: product ", p, " takes fp16 x [R, K0], x1 [R, K - K0] and w [N, K]");
    TORCH_CHECK(x.stride(1)==1 && w.stride(1)==1 && (!x1s[p] || x1s[p]->stride(1)==1) &&
        (!biases[p] || biases[p]->is_contiguous()),
        "linear commands require declared row views; implicit contiguous copies are not supported");
    const at::Tensor &xc=x, &wc=w, &x1c=x1s[p]?*x1s[p]:x;
    const int64_t R = x.size(0), N = w.size(0), ldw = wc.stride(0), ld0 = xc.stride(0), ld1 = x1c.stride(0);
    buffers[LINEAR_BINDINGS * p] = xc, buffers[LINEAR_BINDINGS * p + 1] = wc, buffers[LINEAR_BINDINGS * p + 2] = ys[p];
    buffers[LINEAR_BINDINGS * p + 3] = biases[p] ? biases[p] : std::make_optional(ys[p]), buffers[LINEAR_BINDINGS * p + 4] = x1c;
    words.dimensions[p] = {(uint32_t)R, (uint32_t)K, (uint32_t)N, (uint32_t)bool(biases[p]),
                           (uint32_t)ldw, (uint32_t)K0, (uint32_t)ld0, (uint32_t)ld1,
                           (uint32_t)ys[p].stride(0), (uint32_t)ys[p].stride(1)};
    words.first[p] = first; words.activation[p] = acts[p];
    first += linear_groups(R, N);
    least = std::min(least, K);
    wide_w &= ldw % MESH_LINEAR_VECTOR == 0 && K0 % MESH_LINEAR_VECTOR == 0 && wc.storage_offset() % MESH_LINEAR_VECTOR == 0;
    wide_x &= K0 % MESH_LINEAR_VECTOR == 0 && ld0 % MESH_LINEAR_VECTOR == 0 && ld1 % MESH_LINEAR_VECTOR == 0 &&
              xc.storage_offset() % MESH_LINEAR_VECTOR == 0 && x1c.storage_offset() % MESH_LINEAR_VECTOR == 0;
  }
  const int64_t split = linear_fit(least, first);
  if(P==1) {
    buffers.resize(6);buffers[5]=buffers[4];buffers[4]=std::nullopt;
    return {{linear_pipeline(std::make_tuple(wide_w?MESH_LINEAR_VECTOR:1,wide_x?MESH_LINEAR_VECTOR:1,(int)split,(int)acts[0],1)),commandOperands(buffers),
        {{4,pybind11::bytes((char *)&words.dimensions[0],sizeof(LinearDimensions))}},
        {(NSUInteger)first,1,1},{(NSUInteger)(split*MESH_SIMD_WIDTH),1,1},false,UINT64_C(43)|(UINT64_C(4)<<MESH_ACCESS_WRITE_SHIFT)}};
  }
  const auto key = std::make_tuple(wide_w ? MESH_LINEAR_VECTOR : 1, wide_x ? MESH_LINEAR_VECTOR : 1, (int)split, -1, (int)P);
  std::string packed((char *)words.dimensions,P*sizeof(LinearDimensions));
  packed.append((char *)words.first,P*sizeof(*words.first));
  packed.append((char *)words.activation,P*sizeof(*words.activation));
  uint64_t access=0;
  for(size_t p=0;p<P;p++)access|=(UINT64_C(27)<<(LINEAR_BINDINGS*p))|(UINT64_C(4)<<(MESH_ACCESS_WRITE_SHIFT+LINEAR_BINDINGS*p));
  return {{linear_pipeline(key),commandOperands(buffers),{{LINEAR_BINDINGS*P,pybind11::bytes(packed)}},
      {(NSUInteger)first,1,1},{(NSUInteger)(split*MESH_SIMD_WIDTH),1,1},false,access}};
}

TORCH_LIBRARY_FRAGMENT(torch_mesh, m) {
  m.def("linear(Tensor x, Tensor w, Tensor? bias=None, int act=0, Tensor? x1=None) -> Tensor");
  m.def("linears(Tensor[] x, Tensor[] w, Tensor?[] bias, int[] act, Tensor?[] x1) -> Tensor[]");
  m.def("fused_step(Tensor x, Tensor? w, int dim, int[] sizes, str group_name, bool adds, int repeats) -> Tensor");
  m.def("fused_step_out(Tensor x, Tensor? w, Tensor(a!) out, int dim, int[] sizes, str group_name, bool adds, int repeats) -> ()");
}

// Region tensors (torch_mesh.empty, records, address): a tensor whose bytes are a lease of this node's bridge's region
// (ncclMemAlloc: pages registered on every device when it came up), so the library reaches it with no staging copy
// where it reads or writes its operands in place.  An MPS one is made through torch's documented interop, DLPack
// (at::fromDLPack of a kDLMetal tensor: its data an id<MTLBuffer>, its byte_offset the offset in it, as torch's own
// toDLPack writes an MPS tensor), over the lease's Metal buffer (ncclMeshMemBuffer); its release is the rule torch's MPS
// allocator keeps for a buffer in flight: the lease returned (ncclMemFree) once the stream's command buffer open at the
// release has completed, so every command bound to it before then has run.  A CPU one is torch::from_blob over the
// lease, returned at its release (a host-path call is complete when it returns).  No allocator of torch's is replaced:
// tensors the program makes otherwise stay its MPS allocator's.
static std::mutex region_tensors_lock_;
static std::map<uintptr_t, size_t> region_tensors_;

static void returned(void *memory) {
  ncclMemFree(memory);
  std::lock_guard<std::mutex> guard(region_tensors_lock_);
  region_tensors_.erase((uintptr_t)memory);
}

struct RegionShape {
  std::vector<int64_t> sizes, strides;
};

// Held MPS leases (region_empty's `held`), by their bytes: each released once the stream's work on it had completed,
// and taken again once the NIC's reads of it have ended too (ncclMeshMemReusable).
static std::multimap<size_t, void *> held_;

static at::Tensor region_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps, bool held) {
  const at::ScalarType dtype = like.scalar_type();
  const size_t bytes = std::max<size_t>(1, (size_t)c10::multiply_integers(sizes) * c10::elementSize(dtype));
  void *memory = nullptr;
  {
    std::lock_guard<std::mutex> guard(region_tensors_lock_);
    for (auto [at, end] = held ? held_.equal_range(bytes) : std::make_pair(held_.end(), held_.end()); at != end; ++at) {
      int reusable = 0;
      checked(ncclMeshMemReusable(at->second, &reusable));
      if (reusable) {
        memory = at->second;
        held_.erase(at);
        break;
      }
    }
  }
  if (!memory) {
    checked(ncclMemAlloc(&memory, bytes));
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
  return at::fromDLPack(managed, [managed, memory, bytes, held](void *) {
    managed->deleter(managed);
    at::mps::getCurrentMPSStream()->addCompletedHandler(^(id<MTLCommandBuffer>) {
      if (!held) {
        returned(memory);
        return;
      }
      std::lock_guard<std::mutex> guard(region_tensors_lock_);
      held_.emplace(bytes, memory);
    });
  });
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

// Import frontend attributes once, before recording the function.
// Existing bridge operands retain their pages. Aliases of foreign storage
// share one region allocation and retain their typed views of the same bytes.
static std::vector<at::Tensor> import_attributes(const std::vector<at::Tensor> &values) {
  struct Import {
    std::vector<size_t> members;
    std::vector<std::pair<size_t,size_t>> spans;
    size_t low=std::numeric_limits<size_t>::max(),high=0,alignment=1;
  };
  std::map<const c10::StorageImpl *,Import> groups;
  std::vector<at::Tensor> result=values;
  bool mps=false;
  for(size_t i=0;i<values.size();i++) {
    const auto &value=values[i];
    if(address(value))continue;
    TORCH_CHECK(value.is_cpu()||value.is_mps(),"attribute import requires declared CPU or MPS storage");
    auto &group=groups[value.storage().unsafeGetStorageImpl()];group.members.push_back(i);
    const size_t width=value.element_size();group.alignment=std::lcm(group.alignment,width);
    NSData *data=[TensorLayout::tensor(value).value orderedSpans];
    const auto *spans=(const MetalTensorSpan *)data.bytes;
    for(size_t j=0;j<data.length/sizeof *spans;j++) {
      const size_t first=spans[j].offset*width,last=first+spans[j].count*width;
      group.spans.emplace_back(first,last);group.low=std::min(group.low,first);group.high=std::max(group.high,last);
    }
    mps|=value.is_mps()&&value.numel();
  }
  // A foreign MPS attribute may still be produced by its frontend's setup
  // commands. Complete that producer before the host reads its initial bytes.
  // This import is called only by preparation, never by a recorded invocation.
  if(mps)settled(stream_of(at::mps::getCurrentMPSStream()),"torch_mesh: a foreign attribute's producer on torch's MPS stream (import_attributes)");
  for(auto &[key,group]:groups) {
    const auto &source=values[group.members.front()];
    const size_t low=group.spans.empty()?0:group.low/group.alignment*group.alignment;
    const size_t bytes=group.spans.empty()?0:(group.high-low+group.alignment-1)/group.alignment*group.alignment;
    at::Tensor storage=region_empty({(int64_t)bytes},at::empty({0},at::TensorOptions().dtype(at::kByte)),source.is_mps());
    const void *from=source.is_mps()?[at::native::mps::getMTLBufferStorage(source) contents]:source.storage().data();
    TORCH_CHECK(from||!bytes,"attribute import requires host-visible source storage");
    auto *to=(char *)(uintptr_t)address(storage);
    std::sort(group.spans.begin(),group.spans.end());
    size_t first=0,last=0;
    for(const auto &span:group.spans) {
      if(span.first<=last){last=std::max(last,span.second);continue;}
      if(last>first)memcpy(to+first-low,(const char *)from+first,last-first);
      first=span.first;last=span.second;
    }
    if(last>first)memcpy(to+first-low,(const char *)from+first,last-first);
    for(size_t i:group.members) {
      const auto &value=values[i];const size_t width=value.element_size();
      const size_t offset=storage.storage_offset()+(value.numel()?value.storage_offset()*width-low:0);
      result[i]=storage.view(value.scalar_type()).as_strided(value.sizes(),value.strides(),offset/width);
    }
  }
  return result;
}

// The region tensors' leases live now: each its address and bytes.
static std::vector<std::pair<uint64_t, uint64_t>> records() {
  std::lock_guard<std::mutex> guard(region_tensors_lock_);
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (auto &w : region_tensors_) out.emplace_back(w.first, w.second);
  return out;
}

// Fused crossings share immutable exact-byte WritePrepare/Execute plans.
// GPU bindings name physical channels, including every parallel cable and
// compiled relay path. Producers publish source ranges; consumers observe
// only their data dependencies. Registration cuts are agreed at realization,
// and no buffer must begin on a transport page. Unsupported inputs fail.
// The library's Metal prelude (ncclMeshMetalPrelude): rdma/mesh.metal as the loaded library was built with it, then its
// steps; the text a kernel that runs them is compiled after.
static std::string metal_prelude() { return ncclMeshMetalPrelude(); }

// Prepared handles retain the same exact-byte contract for generic gathers,
// reductions and the public indexed/affine write API. No collective-specific
// transport planner is selected by a generic data function.
static int physical_channels(ncclComm_t comm) {
  const mesh_fabric *fabric = nullptr;
  int rank = 0, channels = 0;
  checked(ncclCommUserRank(comm, &rank));
  checked(ncclMeshCommFabric(comm, &fabric));
  for (uint32_t k = 0; k < fabric->cables; k++)
    channels += fabric->cable[k].node[0] == (uint32_t)rank || fabric->cable[k].node[1] == (uint32_t)rank;
  return channels;
}

struct PreparedWrites {
  std::vector<ncclMeshWritePlan_t> handles;
  std::vector<std::array<uint64_t, 6>> publish, land;
  ncclMeshFused records{};
  uint32_t polls = 0;
  ncclComm_t comm = nullptr;
  void prepare(int64_t comm, int64_t waiting, int64_t repeats, int64_t stride) {
    this->comm=communicator(comm);
    const int channels = physical_channels(communicator(comm));
    publish.resize(channels), land.resize(channels);
    records = {reinterpret_cast<uint64_t(*)[6]>(publish.data()), reinterpret_cast<uint64_t(*)[6]>(land.data()),
               (uint32_t)channels, (uint32_t)channels, (uint32_t)repeats, (size_t)stride};
    polls = (uint32_t)waiting;
  }
  ~PreparedWrites() { for (auto plan : handles) ncclMeshWritePlanDestroy(plan); }
};

static void prepare_fused_writes(MetalSchedule *schedule,PreparedWrites &plans) {
  MetalScheduleCommunicator(schedule,plans.comm);
  plans.records.publishes=plans.publish.size();plans.records.lands=plans.land.size();
  if(plans.handles.empty()){plans.records.publishes=plans.records.lands=0;return;}
  PreparedWrites *p=&plans;
  checked(MetalScheduleCross(schedule,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
      id<MTLResidencySet> residency,NSUInteger,NSUInteger) {
    ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
      .residencySet=(__bridge void *)residency,.polls=p->polls,.fused=&p->records};
    ncclResult_t result=ncclGroupStart();
    for(auto handle:p->handles)if(!result)result=ncclMeshWriteExecute(handle,0,&on);
    ncclResult_t ended=ncclGroupEnd();return (int)(result?result:ended);
  }));
}

// Configuration retains the kernels, page generations and channel bindings of a
// crossing. Each compiled occurrence gets its own indexed dispatch and output.
// Execution reads that index and fills the changing operands; it neither looks
// up a form nor allocates a result. Forms live until their communicator changes
// configuration or shuts down.
struct FusedForm {
  int64_t comm = 0, polls = 0, ranks = 0, rank = 0, publish = -1, land = -1, pj = 0, lj = 0, pgx = 0, pthreads = 0, lgx = 0, lthreads = 0,
          slot = 0, ready = -1, channels = 0;
  bool with_w = false;
  std::vector<int64_t> pwords, lwords, starts, counts, shape;
  at::Tensor arrivals;
  struct Ring {
    std::vector<at::Tensor> slots[2];
    std::vector<std::vector<at::Tensor>> segments[2];
    std::vector<at::Tensor> relays[2];
    std::shared_ptr<PreparedWrites> plans[2];
  };
  std::map<int64_t, Ring> rings;
  std::map<int64_t, std::deque<Ring>> prepared_fresh;
  struct Sequence {
    std::vector<Ring *> rings;
    size_t cursor = 0, stride = 0;
  };
  std::map<int64_t, Sequence> sequences;
  Ring *ring = nullptr;
  int64_t turn = 0, half = 0;
  std::vector<int64_t> pbind, lbind;
  Schedule transport;
};
static auto *fused_forms_ = new std::map<std::vector<int64_t>, FusedForm>;
static auto *fused_lock_ = new std::mutex;
static pybind11::object *fused_maker_ = nullptr;

static void fused_maker(pybind11::object maker) { fused_maker_ = new pybind11::object(std::move(maker)); }

// The form of a crossing on `group`'s communicator, made during preparation; `held` (fused_lock_) is let go
// while the maker runs (it takes the GIL, which a thread waiting on the lock may hold)
static FusedForm &fused_form_of(const at::Tensor &x, const std::optional<at::Tensor> &w, int64_t dim, const std::vector<int64_t> &sizes,
                                const std::string &group, bool adds, std::unique_lock<std::mutex> &held) {
  const int64_t comm = communicator_of(c10::IValue(group));
  TORCH_CHECK(comm, "torch_mesh::fused_step: ", group, " is no live mesh group of this process");
  std::vector<int64_t> key(x.sizes().begin(), x.sizes().end());
  key.insert(key.end(), {-1, (int64_t)x.scalar_type(), dim, comm, adds ? 1 : 0, -2});
  key.insert(key.end(), sizes.begin(), sizes.end());
  key.push_back(-4);
  key.insert(key.end(), x.strides().begin(), x.strides().end());
  if (w) {
    key.push_back(-3);
    key.insert(key.end(), w->sizes().begin(), w->sizes().end());
    key.insert(key.end(), w->strides().begin(), w->strides().end());
  }
  auto found = fused_forms_->find(key);
  if (found != fused_forms_->end()) return found->second;
  TORCH_CHECK(fused_maker_, "torch_mesh::fused_step: no form maker (partition registers it as it loads)");
  FusedForm f;
  held.unlock();
  {
    pybind11::gil_scoped_acquire gil;
    const int channels = physical_channels(communicator(comm));
    const pybind11::tuple t = (*fused_maker_)(x, w ? pybind11::cast(*w) : pybind11::none(), dim, sizes, group, adds, channels);
    f.comm = t[0].cast<int64_t>(), f.polls = t[1].cast<int64_t>(), f.ranks = t[2].cast<int64_t>(), f.rank = t[3].cast<int64_t>();
    f.publish = t[4].cast<int64_t>(), f.pwords = t[5].cast<std::vector<int64_t>>(), f.pj = t[6].cast<int64_t>();
    f.pgx = t[7].cast<int64_t>(), f.pthreads = t[8].cast<int64_t>();
    f.land = t[9].cast<int64_t>(), f.lwords = t[10].cast<std::vector<int64_t>>(), f.lj = t[11].cast<int64_t>();
    f.lgx = t[12].cast<int64_t>(), f.lthreads = t[13].cast<int64_t>();
    f.starts = t[14].cast<std::vector<int64_t>>(), f.counts = t[15].cast<std::vector<int64_t>>(), f.shape = t[16].cast<std::vector<int64_t>>();
    f.arrivals = t[17].cast<at::Tensor>(), f.with_w = w.has_value(), f.slot = f.starts.back();
    f.ready = t[18].cast<int64_t>(), f.channels = t[19].cast<int64_t>();
  }
  held.lock();
  TORCH_CHECK(f.comm == comm, "torch_mesh::fused_step: ", group, "'s form names another communicator");
  return fused_forms_->try_emplace(key, std::move(f)).first->second;
}

static void release_chains(int64_t comm);

// every form on communicator `comm` released, its leases back to the region once the stream has passed them (the
// group's shutdown: ProcessGroupMesh.shutdown, as destroy_process_group ends it)
static void fused_release(int64_t comm) {
  std::lock_guard<std::mutex> guard(*fused_lock_);
  release_chains(comm);
  for (auto f = fused_forms_->begin(); f != fused_forms_->end();) f = f->second.comm == comm ? fused_forms_->erase(f) : std::next(f);
}

// Route changes happen between complete functions; a chain still running holds the rings it runs (PreparedChain).
static void fused_reconfigure(int64_t comm) {
  std::lock_guard<std::mutex> guard(*fused_lock_);
  release_chains(comm);
  for (auto f = fused_forms_->begin(); f != fused_forms_->end();)
    f = f->second.comm == comm ? fused_forms_->erase(f) : std::next(f);
}

// Allocate and agree registration cuts without presenting a receive or
// executing a producer. A fresh generation owns every source/landing/relay lease.
static void fused_allocate_ring(FusedForm &f, FusedForm::Ring &ring, const at::Tensor &xc, int64_t repeats) {
  TORCH_CHECK(repeats >= 1 && repeats <= (int64_t)UINT32_MAX && f.slot >= 0 &&
              f.slot <= INT64_MAX / repeats / (int64_t)xc.element_size(),
              "fused gather: repeated byte extent overflows address space");
  if (ring.slots[0].empty()) {
    const at::Tensor leases[2] = {region_empty({repeats * f.slot}, xc, true), region_empty({repeats * f.slot}, xc, true)};
    for (int h = 0; h < 2; h++)
      for (int64_t r = 0; r < repeats; r++) {
        const at::Tensor slot = leases[h].narrow(0, r * f.slot, f.slot);
        ring.slots[h].push_back(slot);
        std::vector<at::Tensor> segments;
        for (int64_t q = 0; q < f.ranks; q++) segments.push_back(slot.narrow(0, f.starts[q], f.counts[q]));
        ring.segments[h].push_back(segments);
      }
  }
  // Prepare each bound generation once. Metadata exchange agrees common
  // registration cuts before any receive is presented. Relay allocations
  // are disjoint per pair, so another destination cannot overwrite a live
  // forwarding range or any local original contribution.
  for (int h = 0; h < 2; h++) if (!ring.plans[h]) {
    auto prepared = std::make_shared<PreparedWrites>();
    Group g{true};
    for (int64_t source = 0; source < f.ranks; source++) if (f.counts[source])
      for (int64_t destination = 0; destination < f.ranks; destination++) if (source != destination) {
        const at::Tensor source_view = ring.segments[h][0][source];
        at::Tensor landing_view = source_view;
        int64_t local_stride = f.slot * (int64_t)xc.element_size();
        if (f.rank != source && f.rank != destination) {
          ring.relays[h].push_back(region_empty({repeats * f.counts[source]}, xc, true));
          landing_view = ring.relays[h].back().narrow(0, 0, f.counts[source]);
          local_stride = f.counts[source] * (int64_t)xc.element_size();
        }
        const Operand out = g.moving(source_view), in = g.moving(landing_view);
        prepared->handles.push_back(nullptr);
        checked(mesh_compile_write(out.at, (void *)in.at, out.count, out.type, (int)source, (int)destination,
                                    communicator(f.comm), 1, (uint32_t)repeats, (size_t)local_stride, 0,
                                    &prepared->handles.back()));
      }
    prepared->prepare(f.comm, f.polls, repeats, f.slot * (int64_t)xc.element_size());
    ring.plans[h] = std::move(prepared);
  }
}

static void fused_prepare(const at::Tensor &x, const std::optional<at::Tensor> &w, int64_t dim,
                          const std::vector<int64_t> &sizes, const std::string &group, bool adds,
                          int64_t repeats, int64_t batches) {
  TORCH_CHECK(batches >= 1, "fused preparation requires a positive batch count");
  std::unique_lock<std::mutex> held(*fused_lock_);
  FusedForm &f = fused_form_of(x, w, dim, sizes, group, adds, held);
  const bool fresh = std::any_of(f.counts.begin(), f.counts.end(), [](int64_t n) { return n == 0; });
  auto &sequence = f.sequences[repeats];
  sequence.rings.clear();
  sequence.cursor = 0;
  sequence.stride = fresh;
  if (fresh) {
    auto &queue = f.prepared_fresh[repeats];
    for (int64_t k = (int64_t)queue.size(); k < batches; k++) {
      FusedForm::Ring ring;
      fused_allocate_ring(f, ring, x, repeats);
      queue.push_back(std::move(ring));
    }
    for (int64_t k = 0; k < batches; k++) sequence.rings.push_back(&queue[k]);
  } else {
    fused_allocate_ring(f, f.rings[repeats], x, repeats);
    sequence.rings.push_back(&f.rings[repeats]);
  }
  f.pbind.resize(f.channels * 24), f.lbind.resize(f.channels * 24);
}

// A complete recurrent chain uses the same prepared physical writes as a
// crossing. Its restored processing kernel consumes the preceding landing and
// publishes the next product in one dispatch. All buffers and offsets are
// assigned here before execution.
struct ChainStep {
  std::array<id<MTLBuffer>, 4> buffers;
  std::array<NSUInteger, 4> offsets, lengths;
  uint32_t first, last, previous_j, j, input_j;
  const std::vector<Dispatch *> *dispatches;
};
struct ChainChunk {
  PreparedWrites *plans;
  size_t first, count, input, prefetch_first, prefetch_count;
};
struct PreparedChain {
  int64_t comm;
  std::vector<at::Tensor> held;
  std::vector<ChainStep> steps;
  std::vector<ChainChunk> chunks;
  std::vector<std::unique_ptr<PreparedWrites>> inputs;
  std::vector<uint32_t> input_last;
  Dispatch dispatch, input_publication;
  std::vector<Dispatch> products;
  std::vector<Dispatch *> regular, final;
  Dispatch *publication;
  size_t publication_output, publication_j, publication_words;
  MeshStream *stream;
  uint32_t channels;
  Schedule schedule;
  // the form's rings its steps send from and land in, held with the chain (a reconfiguration lets the form go)
  std::vector<FusedForm::Ring> rings;
};
static void compile_chain(PreparedChain &chain);
// each prepared chain, held too by its run until the prepared queue has passed it (ordered)
static auto *prepared_chains_ = new std::vector<std::shared_ptr<PreparedChain>>;

static void release_chains(int64_t comm) {
  for (auto &chain : *prepared_chains_) if (chain && chain->comm == comm) chain.reset();
}
static int64_t chain_slot() {
  const int64_t index = prepared_chains_->size();
  prepared_chains_->push_back(nullptr);
  return index;
}
static void prepare_chain(int64_t index, int64_t kernel, int64_t input_kernel, const std::vector<int64_t> &products, bool linear,
                          const std::vector<at::Tensor> &operands,
                          const std::vector<int64_t> &constants, int64_t groups, int64_t threads,
                          const std::string &group, int64_t ranks, int64_t batch, int64_t input_frames) {
  const at::Tensor &input = operands[2], &output = operands[3], &x = operands[4];
  const int64_t steps = input.size(0);
  std::unique_lock<std::mutex> held(*fused_lock_);
  FusedForm &f = fused_form_of(x, std::nullopt, 0, std::vector<int64_t>(ranks, x.size(0)), group, true, held);
  std::vector<std::optional<at::Tensor>> buffers(operands.begin(), operands.end());
  buffers.push_back(x);  // previous landing; replaced by its compiled page below
  std::vector<int64_t> words = constants;
  words.resize(12 + 36 * f.channels);
  auto chain = std::make_shared<PreparedChain>(PreparedChain{f.comm, operands, {}, {}, {}, {},
      encoded(kernel, buffers, words, 15, groups, 1, threads),
      encoded(input_kernel, {}, std::vector<int64_t>(2 + 12 * f.channels), 15, 1, 1, MESH_SIMD_WIDTH),
      {}, {}, {}, nullptr, 6, 9, 12,
      &stream_of(at::mps::getCurrentMPSStream()), (uint32_t)f.channels});
  chain->regular = chain->final = {&chain->dispatch};
  if (linear) {
    const at::Tensor &y = operands[5];
    const int64_t R = x.size(0), K = x.size(1), S = y.size(1);
    chain->products.push_back(encoded(products[0], {x, operands[0], y, y, std::nullopt, x},
        {R, K, S, 0, K, K, K, K, y.stride(0), y.stride(1)}, 4, linear_groups(R, S), 1, products[2] * MESH_SIMD_WIDTH));
    std::vector<int64_t> publication{R, S, K, 0, S, S, S, S, 0, 0};
    publication.resize(10 + 24 * f.channels);
    chain->products.push_back(encoded(products[1], {y, operands[1], x, operands[7], y},
        publication, 15, linear_groups(R, K), 1, products[3] * MESH_SIMD_WIDTH));
    chain->publication_output = 2;
    chain->publication_j = 8;
    chain->publication_words = 10;
  } else {
    for (auto product : products)
      chain->products.push_back(encoded(product, buffers, words, 15, groups, 1, threads));
  }
  for (auto &product : chain->products) chain->regular.push_back(&product);
  chain->publication = &chain->products.back();
  const NSUInteger input_base = input.storage_offset() * input.element_size();
  const NSUInteger output_base = output.storage_offset() * output.element_size();
  const NSUInteger input_pitch = input.stride(0) * input.element_size();
  const NSUInteger output_pitch = output.stride(0) * output.element_size();
  for (int64_t first = 0; first < steps; first += input_frames) {
    const int64_t count = std::min(input_frames, steps - first);
    const at::Tensor source = input.select(0, first);
    auto prepared = std::make_unique<PreparedWrites>();
    Group g{true};
    for (int64_t destination = 1; destination < ranks; destination++) {
      at::Tensor landing = source;
      size_t stride = input_pitch;
      if (f.rank != 0 && f.rank != destination) {
        chain->held.push_back(region_empty({count * source.numel()}, input, true));
        landing = chain->held.back().narrow(0, 0, source.numel());
        stride = source.nbytes();
      }
      const Operand out = g.moving(source), in = g.moving(landing);
      prepared->handles.push_back(nullptr);
      checked(mesh_compile_write(out.at, (void *)in.at, out.count, out.type, 0, (int)destination,
                                  communicator(f.comm), 1, (uint32_t)count, stride, 0, &prepared->handles.back()));
    }
    prepared->prepare(f.comm, f.polls, count, input_pitch);
    chain->inputs.push_back(std::move(prepared));
    chain->input_last.push_back((uint32_t)(count - 1));
  }
  {
    at::Tensor previous = x;
    uint32_t previous_j = 0;
    auto append = [&](int64_t t, int64_t j, const at::Tensor &send, const at::Tensor &land, bool last) {
      const std::array<at::Tensor, 4> bound{input, output, send, land};
      ChainStep step{{}, {}, {}, (uint32_t)(t == 0), (uint32_t)last, previous_j, (uint32_t)j, (uint32_t)(t % input_frames),
                     last ? &chain->final : &chain->regular};
      for (size_t k = 0; k < bound.size(); k++) {
        step.buffers[k] = at::native::mps::getMTLBufferStorage(bound[k]);
        step.offsets[k] = bound[k].storage_offset() * bound[k].element_size();
        step.lengths[k] = bound[k].nbytes();
      }
      step.lengths[0] = input_pitch; step.lengths[1] = output_pitch;
      step.offsets[0] = input_base + std::min(t, steps - 1) * input_pitch;
      step.offsets[1] = output_base + std::max<int64_t>(0, t - 1) * output_pitch;
      chain->steps.push_back(step);
    };
    for (int64_t first = 0, turn = 0; first < steps; turn++) {
      const int64_t count = std::min({batch, steps - first, input_frames - first % input_frames}), half = turn % 2;
      const size_t input = first / input_frames;
      const size_t prefetch = first % input_frames == 0 && input + 1 < chain->inputs.size();
      FusedForm::Ring &ring = f.rings.at(count);
      chain->chunks.push_back({ring.plans[half].get(), (size_t)first, (size_t)count, input, input + 1, prefetch});
      for (int64_t j = 0; j < count; j++) {
        append(first + j, j, ring.segments[half][j][f.rank], previous, false);
        previous = ring.slots[half][j];
        previous_j = (uint32_t)j;
      }
      first += count;
    }
    append(steps, 0, x, previous, true);
  }
  for (const auto &[count, ring] : f.rings) chain->rings.push_back(ring);
  compile_chain(*chain);
  (*prepared_chains_)[index] = std::move(chain);
}

static void compile_chain(PreparedChain &chain) {
  chain.schedule=schedule_after_inputs();
  MetalSchedule *schedule=chain.schedule.get();
  Dispatch &d = chain.dispatch;
  Dispatch &publication = *chain.publication;
  const auto &steps = chain.steps;
  const auto &chunks = chain.chunks;
  const auto binding = [&](const PreparedWrites *plans, bool publication, size_t part) {
    const uint32_t count = !plans ? 0 : publication ? plans->records.publishes : plans->records.lands;
    const auto *records = !plans ? nullptr : publication ? plans->publish.data() : plans->land.data();
    const size_t offset = (publication ? chain.publication_words : 12) + part * 12 * chain.channels;
    auto &words = publication ? chain.publication->words : d.words;
    for (uint32_t k = 0; k < chain.channels; k++) {
      const auto *record = k < count ? records + k : nullptr;
      for (int word = 0; word < 6; word++) {
        const uint64_t value = record ? (*record)[word] : 0;
        words[offset + 12 * k + 2 * word] = value & 0xffffffffu;
        words[offset + 12 * k + 2 * word + 1] = value >> 32;
      }
    }
  };
  const auto input = [&](size_t index) {
    PreparedWrites &plans = *chain.inputs[index];
    prepare_fused_writes(schedule,plans);
    Dispatch &ready = chain.input_publication;
    ready.words[0] = chain.input_last[index];
    for (uint32_t k = 0; k < chain.channels; k++) {
      const auto *record = k < plans.records.publishes ? &plans.publish[k] : nullptr;
      for (int word = 0; word < 6; word++) {
        const uint64_t value = record ? (*record)[word] : 0;
        ready.words[2 + 12 * k + 2 * word] = value & 0xffffffffu;
        ready.words[3 + 12 * k + 2 * word] = value >> 32;
      }
    }
    prepare_dispatch(schedule,ready,0);

  };
  const auto dispatch = [&](const ChainStep &step) {
    const size_t slots[4] = {2, 3, 6, 8};
    for (size_t k = 0; k < 4; k++) { d.bound[slots[k]] = step.buffers[k]; d.offsets[slots[k]] = step.offsets[k]; d.lengths[slots[k]]=step.lengths[k]; }
    d.words[0] = step.first, d.words[5] = step.last;
    d.words[8] = step.previous_j, d.words[9] = step.j;
    d.words[10] = step.input_j;
    publication.bound[chain.publication_output] = step.buffers[2];
    publication.offsets[chain.publication_output] = step.offsets[2];
    publication.lengths[chain.publication_output] = step.lengths[2];
    publication.words[chain.publication_j] = step.j;
    // Phase 0 reads the incoming pages and writes X. The products consume X
    // then Y. These declarations compile their two data edges; groups within
    // a phase need no residency assumption or cross-group spin barrier.
    for(const Dispatch *item:*step.dispatches) {
      uint64_t access=0;
      if(item==&d)access=UINT64_C(260)|(UINT64_C(24)<<MESH_ACCESS_WRITE_SHIFT);
      else if(chain.publication_words==12)access=item==chain.publication ?
        UINT64_C(162)|(UINT64_C(192)<<MESH_ACCESS_WRITE_SHIFT) : UINT64_C(17)|(UINT64_C(32)<<MESH_ACCESS_WRITE_SHIFT);
      else {
        for(size_t k=0;k<item->bound.size();k++)if(item->bound[k])access|=UINT64_C(1)<<k;
        access|=(item==chain.publication?UINT64_C(12):UINT64_C(4))<<MESH_ACCESS_WRITE_SHIFT;
      }
      prepare_dispatch(schedule,*item,access);
    }

  };
  const PreparedWrites *previous = nullptr;
  input(0);
  for (size_t turn = 0; turn < chunks.size(); turn++) {
    const ChainChunk &chunk = chunks[turn];
    for (size_t j = 0; j < chunk.prefetch_count; j++) input(chunk.prefetch_first + j);
    prepare_fused_writes(schedule,*chunk.plans);
    binding(previous, false, 0);
    binding(chunk.plans, true, 1);
    binding(chain.inputs[chunk.input].get(),false,2);
    const ChainStep *first=steps.data()+chunk.first;
    dispatch(*first);
    binding(chunk.plans,false,0);
    for(size_t j=1;j<chunk.count;j++)dispatch(first[j]);
    previous = chunk.plans;
  }
  dispatch(steps.back());
  checked(MetalScheduleFinish(schedule,nil));
}

static const std::shared_ptr<PreparedChain> &chain_at(int64_t index) {
  TORCH_CHECK(index >= 0 && index < (int64_t)prepared_chains_->size() && (*prepared_chains_)[index], "no prepared chain ", index);
  return (*prepared_chains_)[index];
}
static void chain_run(int64_t index) {
  const auto &chain = chain_at(index);
  ordered(chain->schedule, chain);
}
static pybind11::capsule chain_completion(int64_t index) {
  return schedule_completion(chain_at(index)->schedule);
}

// A declared set of routed writes. Bindings name physical channels, including
// parallel cables; relay readiness stays in the compiled stream entries.
static std::pair<std::vector<std::vector<uint64_t>>, std::vector<std::vector<uint64_t>>> fused_writes_impl(
    MetalSchedule *schedule, const at::Tensor &send, const std::vector<at::Tensor> &landings,
    const std::vector<std::array<int, 2>> &pairs, int64_t comm, int64_t polls, int64_t repeats, int64_t stride,
    int64_t row_bytes, int64_t rows, const std::optional<at::Tensor> &ready, int64_t ready_value, int64_t tile_bytes) {
  TORCH_CHECK(landings.size() == pairs.size(), "fused writes: one landing lease per declared write");
  TORCH_CHECK(polls >= 0 && polls <= (int64_t)UINT32_MAX, "fused writes: polls is a 32-bit count of loads");
  TORCH_CHECK(repeats > 0 && repeats <= (int64_t)UINT32_MAX && stride >= 0,
              "fused writes: positive 32-bit frame count and nonnegative byte stride required");
  const ncclComm_t c = communicator(comm);
  TORCH_CHECK(tile_bytes >= 0 && (!tile_bytes || (ready && repeats == 1 && rows == 0)),
              "tiled writes require a contiguous publication family and its source words");
  const size_t tiles = tile_bytes && send.nbytes() ? (send.nbytes()-1)/(size_t)tile_bytes+1 : 1;
  TORCH_CHECK(rows >= 0 && row_bytes >= 0 && (!rows || (row_bytes > 0 && stride >= row_bytes)),
              "affine writes need positive row bytes and a local pitch at least that large");
  const auto framed = [repeats, stride](const at::Tensor &tensor) {
    const size_t bytes = tensor.nbytes(), offset = tensor.storage_offset() * tensor.element_size();
    TORCH_CHECK(!stride || (size_t)(repeats - 1) <= (SIZE_MAX - bytes) / (size_t)stride,
                "fused writes: frame extent overflows address space");
    const size_t extent = bytes + (size_t)(repeats - 1) * (size_t)stride;
    TORCH_CHECK(offset <= tensor.storage().nbytes() && extent <= tensor.storage().nbytes() - offset,
                "fused writes: all frames must lie inside the tensor's backing storage");
  };
  if (!rows) framed(send);
  Group g{true};
  const Operand out = g.moving(send);
  std::optional<Operand> produced;
  if (ready) {
    TORCH_CHECK(ready->scalar_type() == at::kLong && (size_t)ready->numel() == tiles && ready->is_contiguous() && ready_value > 0,
                "source readiness requires one contiguous int64 region word per tile and a positive target");
    produced = g.moving(*ready);
  }
  std::vector<Operand> in;
  for (const at::Tensor &landing : landings) {
    if (!rows) framed(landing);
    in.push_back(g.moving(landing));
    TORCH_CHECK(in.back().count == out.count && in.back().type == out.type,
                "fused writes: landing extent and dtype must match the source");
  }
  PreparedWrites plans;
  plans.prepare(comm, polls, repeats, rows ? 0 : stride);
  const size_t binding_words=sizeof(ncclMeshBinding)/sizeof(uint64_t),record_words=1+plans.land.size()*binding_words;
  std::vector<uint64_t> tile_landings(tile_bytes ? tiles*record_words : 0);
  int rank=0;checked(ncclCommUserRank(c,&rank));
  plans.handles.resize(pairs.size(), nullptr);
  for (size_t i = 0; i < pairs.size(); i++) {
    if (tile_bytes) {
      checked(mesh_compile_rows(out.at, (void *)in[i].at, send.nbytes(), 1, send.nbytes(),
          pairs[i][0], pairs[i][1], c, 1, 0, mesh_write_layout{.tile_bytes=(size_t)tile_bytes,.instances=1,.invocations=1}, &plans.handles[i]));
      if(rank==pairs[i][1])checked(ncclMeshWriteLandings(plans.handles[i],tile_landings.data(),(uint32_t)plans.land.size(),
          ncclMeshTileLayout{.columns=tiles,.tile_stride=record_words*sizeof(uint64_t),.instances=1}));
    } else if (rows) {
      TORCH_CHECK(out.type == ncclUint8 && out.count == (size_t)row_bytes,
                  "affine write bindings name the first row as a contiguous byte view");
      checked(mesh_compile_rows(out.at, (void *)in[i].at, (size_t)row_bytes, (size_t)rows, (size_t)stride,
                                      pairs[i][0], pairs[i][1], c, 1, 0, mesh_write_layout{.instances=1,.invocations=1}, &plans.handles[i]));
    } else checked(mesh_compile_write(out.at, (void *)in[i].at, out.count, out.type, pairs[i][0], pairs[i][1], c, 1,
                                      (uint32_t)repeats, (size_t)stride, 0, &plans.handles[i]));
    if (produced) checked(ncclMeshWriteSourceReady(plans.handles[i], produced->at, (uint64_t)ready_value,
        ncclMeshTileLayout{.columns=tiles,.tile_stride=sizeof(uint64_t),.instances=1}));
  }
  prepare_fused_writes(schedule,plans);
  std::vector<std::vector<uint64_t>> published, landed;
  for (uint32_t k = 0; k < plans.records.publishes; k++) published.emplace_back(plans.publish[k].begin(), plans.publish[k].end());
  if(tile_bytes) {
    for(size_t t=0;t<tiles;t++)for(size_t k=0;k<plans.land.size();k++) {
      const uint64_t *record=tile_landings.data()+t*record_words;
      if(k<record[0])landed.emplace_back(record+1+k*binding_words,record+1+(k+1)*binding_words);
      else landed.emplace_back(binding_words,0);
    }
  } else for (uint32_t k = 0; k < plans.records.lands; k++) landed.emplace_back(plans.land[k].begin(), plans.land[k].end());
  return {published, landed};
}

// A Python binding of the shared native schedule. All methods except enqueue
// receive configuration; enqueue is the native header's submission operation.
struct ScheduleBinding {
  Schedule schedule;
  ScheduleBinding():schedule(schedule_after_inputs()){}
  std::vector<at::Tensor> operands;
  std::vector<std::shared_ptr<void>> steps;  // the appended steps' replays and plans
  // Preparation supplies native run/context/retain/release operands. The
  // completion worker neither owns a Python callback nor acquires the GIL.
  void host(const std::array<uintptr_t,4> &entry,const std::vector<pybind11::object> &inputs,
      const std::vector<pybind11::object> &outputs) { @autoreleasepool {
    for(auto pointer:entry)TORCH_CHECK(pointer,"host operation requires a complete native binding");
    schedule.host=true;
    using Run=const char *(*)(void *);
    using Lifetime=void (*)(void *);
    auto run=reinterpret_cast<Run>(entry[0]);
    auto context=reinterpret_cast<void *>(entry[1]);
    reinterpret_cast<Lifetime>(entry[2])(context);
    auto bound=std::shared_ptr<void>(context,reinterpret_cast<Lifetime>(entry[3]));
    std::vector<MetalReach> views;
    std::vector<at::Tensor> held;
    auto declare=[&](const std::vector<pybind11::object> &tensors,uint32_t access) {
      for(const auto &object:tensors) {
        const TensorOperand operand(object);const auto &t=operand.storage;held.push_back(t);
        NSData *data=[operand.layout.value reaches:operand.pages.buffer base:0 elementBytes:t.element_size() access:access binding:-1];
        const auto *spans=(const MetalReach *)data.bytes;
        views.insert(views.end(),spans,spans+data.length/sizeof *spans);
      }
    };
    declare(inputs,MetalReachRead);declare(outputs,MetalReachWritten);
    MetalScheduleHost(schedule.get(),views.data(),views.size(),^NSError * {
      // Tensor handles retain the actual storage, including CPU-only weights.
      (void)held;
      const char *error=run(bound.get());
      return error?[NSError errorWithDomain:@"mesh.host-operation" code:1
        userInfo:@{NSLocalizedDescriptionKey:@(error)}]:nil;
    });
  }}
  void compute(int64_t kernel,const Tensors &buffers,const std::vector<int64_t> &words,
      int64_t at,int64_t gx,int64_t gy,int64_t threads,uint64_t access) {
    for(const auto &buffer:buffers)if(buffer)operands.push_back(*buffer);
    if(!gx||!gy)return;
    prepare_dispatch(schedule.get(),encoded(kernel,buffers,words,at,gx,gy,threads),access);
  }
  auto writes(const at::Tensor &send,const std::vector<at::Tensor> &landings,
      const std::vector<std::array<int,2>> &pairs,int64_t comm,int64_t polls,int64_t repeats,int64_t stride,
      int64_t row_bytes,int64_t rows,const std::optional<at::Tensor> &ready,int64_t ready_value,int64_t tile_bytes) {
    operands.push_back(send);operands.insert(operands.end(),landings.begin(),landings.end());
    if(ready)operands.push_back(*ready);
    return fused_writes_impl(schedule.get(),send,landings,pairs,comm,polls,repeats,stride,row_bytes,rows,ready,ready_value,tile_bytes);
  }

  // its operands and steps held as long as the schedule; a function with a host step ends where all its parts have
  // (its gate 2: `ordered` waits for that)
  void finish() {
    const auto held=operands;const auto kept=steps;id<MTLSharedEvent> ended=schedule.host?schedule.gate:nil;
    checked(MetalScheduleFinish(schedule.get(),^(NSError *){(void)held;(void)kept;ended.signaledValue=2;}));
  }
  void enqueue() { ordered(schedule); }
  pybind11::capsule completion() { return schedule_completion(schedule); }
};

// Invocation slots and results are operands. Runtime selects the next prepared
// native function; it does not enter the framework's encoding queue.
struct Executions {
  std::vector<Schedule> schedules;
  pybind11::object output;
  size_t next=0;
  explicit Executions(pybind11::object value):output(std::move(value)){}
  void add(ScheduleBinding &binding){schedules.push_back(binding.schedule);}
  pybind11::object run(){ordered(next_of(schedules,next));return output;}
  pybind11::capsule completion(){return schedule_completion(last_of(schedules,next));}
};

// Compact row geometry and explicit indexed selections lower to one prepared
// page WRITE. Layout resolution ends before native commands are recorded.
using ViewBinding = std::tuple<at::Tensor, std::array<uint64_t, 3>, std::vector<std::array<uint64_t, 2>>, int, int>;
struct ViewWrites {
  int64_t comm;
  bool metal;
  std::vector<ViewBinding> bindings;
  PreparedWrites plans;
  std::vector<at::Tensor> leases;
  ViewWrites(const std::vector<ViewBinding> &given, int64_t handle) : comm(handle), metal(false), bindings(given) {
    TORCH_CHECK(!bindings.empty(), "view writes require at least one declared delivery");
    metal = std::get<0>(bindings.front()).is_mps();
    for(const auto &binding:bindings)leases.push_back(std::get<0>(binding));
  }
  void prepare() {
    if(!plans.handles.empty())return;
    Group g{metal};
    for (const auto &[base, geometry, indexed, source, destination] : bindings) {
      TORCH_CHECK(base.is_mps() == metal, "view writes require one local device");
      const size_t offset = base.storage_offset() * base.element_size();
      const size_t storage = base.storage().nbytes();
      const void *at = g.where(base);
      plans.handles.push_back(nullptr);
      const mesh_write_layout layout{.instances=1,.invocations=1};
      const auto [width, rows, pitch] = geometry;
      if (rows) {
        TORCH_CHECK(indexed.empty() && width && pitch >= width && offset <= storage && width <= storage - offset &&
                    rows - 1 <= (storage - offset - width) / pitch, "affine write lies outside its tensor storage");
        checked(mesh_compile_rows(at, (void *)at, width, rows, pitch, source, destination,
                                  communicator(comm), metal, 0, layout, &plans.handles.back()));
      } else {
        std::vector<ncclMeshSpan> spans;
        spans.reserve(indexed.size());
        for (const auto &row : indexed) {
          TORCH_CHECK(offset <= storage && row[0] <= storage - offset && row[1] <= storage - offset - row[0],
                      "indexed write lies outside its tensor storage");
          spans.push_back({row[0], row[1]});
        }
        checked(mesh_compile_spans(at, (void *)at, spans.data(), spans.size(), source, destination,
                                   communicator(comm), metal, 0, layout, &plans.handles.back()));
      }
    }
  }
  ncclResult_t encode(ncclMeshStream *stream) {
    ncclResult_t result = ncclGroupStart();
    for (auto plan : plans.handles) if (!result) result = ncclMeshWriteExecute(plan, 0, stream);
    const ncclResult_t ended = ncclGroupEnd();
    return result ? result : ended;
  }
  std::vector<Schedule> executions;
  size_t next=0;
  void append(MetalSchedule *schedule) {
    TORCH_CHECK(metal,"prepared distributed writes require the Metal backend");
    MetalScheduleCommunicator(schedule,communicator(comm));
    prepare();
    ViewWrites *self=this;
    checked(MetalScheduleCross(schedule,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
        id<MTLResidencySet> residency,NSUInteger,NSUInteger) {
      ncclMeshStream on={.commandEncoder=(__bridge void *)encoder,.argumentTable=(__bridge void *)table,
        .residencySet=(__bridge void *)residency};
      return (int)self->encode(&on);
    }));
  }
  void realize() {
    auto made=schedule_after_inputs();append(made.get());
    const auto held=leases;
    checked(MetalScheduleFinish(made.get(),^(NSError *){(void)held;}));executions.push_back(std::move(made));
  }
  // its plans held while it runs (a configuration's writes are let go once enqueued)
  void enqueue(std::shared_ptr<ViewWrites> self) { ordered(next_of(executions,next),std::move(self)); }
  pybind11::capsule completion(){return schedule_completion(last_of(executions,next));}

};

// design/whole-function.md#observations
static pybind11::capsule completion() {
  auto stream = at::mps::getCurrentMPSStream();
  __block id<MTLCommandBuffer> buffer = nil;
  at::mps::dispatch_sync_with_rethrow(stream->queue(), ^{
    buffer = [stream->commandBuffer().rootCommandBuffer retain];
  });
  return pybind11::capsule((void *)buffer, "mesh.command-buffer", [](PyObject *capsule) {
    [(id<MTLCommandBuffer>)PyCapsule_GetPointer(capsule, "mesh.command-buffer") release];
  });
}

static bool completed(pybind11::object observation) {
  if(pybind11::isinstance<pybind11::tuple>(observation)) {
    bool done=true;
    for(auto item:observation.cast<pybind11::tuple>())done=completed(pybind11::reinterpret_borrow<pybind11::object>(item))&&done;
    return done;
  }
  auto token=observation.cast<pybind11::capsule>();
  if(!strcmp(token.name(),"mesh.prepared-function")) {
    NSError *error=nil;
    int status=MetalScheduleStatus(static_cast<Schedule *>(token.get_pointer())->get(),&error);
    TORCH_CHECK(status>=0,"the prepared function failed: ",error.localizedDescription.UTF8String);return status==1;
  }
  id<MTLCommandBuffer> buffer = (id<MTLCommandBuffer>)token.get_pointer();
  if (buffer.status == MTLCommandBufferStatusError)
    throw std::runtime_error(buffer.error.localizedDescription.UTF8String);
  return buffer.status == MTLCommandBufferStatusCompleted;
}

// The group's calls (torch_mesh/backend.py): `issue` a group of calls, issued and not completed where `split` (a
// MeshWork to wait on, its outputs `outputs`), else None (complete).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("place_layout",[](NSUInteger count,const std::vector<TensorLayout> &uses,
      const std::vector<TensorLayout> &targets,const std::vector<TensorLayout> &values,const std::vector<NSInteger> &widths)->pybind11::object {
    auto layouts=[](const std::vector<TensorLayout> &items) {
      NSMutableArray *result=[NSMutableArray array];for(const auto &item:items)[result addObject:item.value];return result;
    };
    @try {
      MetalTensorLayout *placed=MetalTensorPlace(count,layouts(uses),layouts(targets),layouts(values),tensorNumbers(widths));
      return placed?pybind11::cast(TensorLayout(placed)):pybind11::none();
    } @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
  });
  module.def("import_attributes",&import_attributes);
  module.def("completion", &completion);
  module.def("completed", &completed);
  module.def("pipeline", &pipeline);
  pybind11::class_<TensorLayout>(module,"TensorLayout")
    .def(pybind11::init(&TensorLayout::make),pybind11::arg("shape"),pybind11::arg("strides"),pybind11::arg("offset")=0,
         pybind11::arg("maps")=std::map<NSUInteger,std::vector<NSInteger>>{},pybind11::arg("indices")=pybind11::none())
    .def_property_readonly("shape",[](const TensorLayout &v){return tensorNumbers(v.value.shape);})
    .def_property_readonly("strides",[](const TensorLayout &v){return tensorNumbers(v.value.strides);})
    .def_property_readonly("offset",[](const TensorLayout &v){return v.value.offset;})
    .def_property_readonly("count",[](const TensorLayout &v){return v.value.count;})
    .def_property_readonly("affine",[](const TensorLayout &v){return bool(v.value.affine);})
    .def_property_readonly("state",[](const TensorLayout &v){
      std::vector<std::vector<NSInteger>> maps(v.value.shape.count);
      for(NSNumber *axis in v.value.maps){NSData *data=v.value.maps[axis];const auto *p=(const NSInteger *)data.bytes;maps[axis.unsignedIntegerValue]={p,p+data.length/sizeof *p};}
      std::vector<NSInteger> indices;
      if(v.value.indices){const auto *p=(const NSInteger *)v.value.indices.bytes;indices={p,p+v.value.indices.length/sizeof *p};}
      return pybind11::make_tuple(tensorNumbers(v.value.shape),tensorNumbers(v.value.strides),v.value.offset,maps,indices);
    })
    .def("index",[](const TensorLayout &v,NSUInteger i){return [v.value index:i];})
    .def("select",[](const TensorLayout &v,NSUInteger axis,const std::vector<NSInteger> &indices){return TensorLayout([v.value select:axis indices:tensorWords(indices)]);})
    .def("slice",[](const TensorLayout &v,NSUInteger axis,NSInteger first,NSUInteger count,NSInteger step){return TensorLayout([v.value slice:axis first:first count:count step:step]);})
    .def("element",[](const TensorLayout &v,NSUInteger axis,NSUInteger i){return TensorLayout([v.value element:axis at:i]);})
    .def("permute",[](const TensorLayout &v,const std::vector<NSInteger> &axes){return TensorLayout([v.value permute:tensorNumbers(axes)]);})
    .def("reshape",[](const TensorLayout &v,const std::vector<NSInteger> &shape){return TensorLayout([v.value reshape:tensorNumbers(shape)]);})
    .def("components",[](const TensorLayout &v,NSUInteger count){return TensorLayout([v.value components:count]);})
    .def("elements",[](const TensorLayout &v,NSUInteger count){return TensorLayout([v.value elements:count]);})
    .def("gather",[](const TensorLayout &v,const std::vector<NSInteger> &shape,const std::vector<NSInteger> &indices){return TensorLayout([v.value gather:tensorNumbers(shape) indices:tensorWords(indices)]);})
    .def("rebase",[](const TensorLayout &v,const TensorLayout &declared,const TensorLayout &placed){
      @try{return TensorLayout([v.value rebase:declared.value onto:placed.value]);}
      @catch(NSException *error){throw pybind11::value_error(error.reason.UTF8String);}
    })
    .def("matches",[](const TensorLayout &v,const std::vector<NSInteger> &steps){return bool([v.value matches:tensorNumbers(steps)]);})
    .def("required",[](const TensorLayout &v,const std::string &kind,const std::vector<NSInteger> &steps){return TensorLayout([v.value required:@(kind.c_str()) strides:tensorNumbers(steps)]);},pybind11::arg("kind"),pybind11::arg("strides")=std::vector<NSInteger>{})
    .def("requirements",[](const TensorLayout &v,const std::vector<std::pair<std::string,std::vector<NSInteger>>> &requirements){
      NSMutableArray *values=[NSMutableArray array];
      for(const auto &[kind,steps]:requirements)[values addObject:@{@"kind":@(kind.c_str()),@"strides":tensorNumbers(steps)}];
      @try { return TensorLayout([v.value requirements:values]); }
      @catch(NSException *error){TORCH_CHECK(false,error.reason.UTF8String);}
    })
    .def("declaration",[](const TensorLayout &v,NSUInteger binding){return std::string([v.value declaration:binding].UTF8String);})
    .def("storage_declaration",[](const TensorLayout &v,NSUInteger binding,const TensorLayout &source){
      return std::string([v.value declaration:binding fromStorage:source.value].UTF8String);})
    .def("ordered_spans",[](const TensorLayout &v){
      NSData *data=[v.value orderedSpans];const auto *spans=(const MetalTensorSpan *)data.bytes;
      std::vector<std::pair<NSInteger,NSUInteger>> result;
      for(NSUInteger i=0;i<data.length/sizeof *spans;i++)result.emplace_back(spans[i].offset,spans[i].count);
      return result;
    })
    .def("spans",[](const TensorLayout &v,NSUInteger bytes){
      NSData *data=[v.value reaches:nil base:0 elementBytes:bytes access:MetalReachRead binding:0];
      const MetalReach *spans=(const MetalReach *)data.bytes;std::vector<std::pair<NSUInteger,NSUInteger>> result;
      for(NSUInteger i=0;i<data.length/sizeof *spans;i++)result.push_back({spans[i].offset,spans[i].length});return result;
    })
    .def("__eq__",[](const TensorLayout &v,const TensorLayout &w){return bool([v.value isEqual:w.value]);})
    .def("__hash__",[](const TensorLayout &v){return v.value.hash;});
  module.def("operand_layout",[](const pybind11::object &object){return TensorOperand(object).layout;});
  module.def("same_operand",[](const pybind11::object &left,const pybind11::object &right){
    const TensorOperand a(left),b(right);
    return a.storage.scalar_type()==b.storage.scalar_type()&&a.pages.buffer==b.pages.buffer&&[a.layout.value isEqual:b.layout.value];
  });
  module.def("view_prelude", [](){return std::string(MetalViewIndexPrelude().UTF8String);});
  auto viewSelection=[](NSDictionary *result) {
    auto values=[](NSData *data) {
      std::vector<int32_t> words(data.length/sizeof(int32_t));
      if(data.length)memcpy(words.data(),data.bytes,data.length);
      return words;
    };
    return std::make_pair(values(result[@"blocks"]),values(result[@"order"]));
  };
  module.def("view_block_slice", [viewSelection](const std::vector<int32_t> &order,NSInteger first,NSInteger end,NSInteger step,NSUInteger block) {
    return viewSelection(MetalViewBlockSlice(order.data(),order.size(),first,end,step,block));
  });
  module.def("view_block_gather", [viewSelection](const std::vector<int32_t> &order,const std::vector<int32_t> &selected,NSUInteger block) {
    return viewSelection(MetalViewBlockGather(order.data(),order.size(),selected.data(),selected.size(),block));
  });
  module.def("view_tiles", [](const std::vector<NSUInteger> &shape,NSUInteger elements) {
    using Tile=std::pair<std::vector<NSUInteger>,std::vector<NSUInteger>>;
    if(shape.empty())return std::vector<Tile>{{{}, {}}};
    NSData *data=MetalViewTiles(shape.data(),shape.size(),elements);
    const NSUInteger *words=(const NSUInteger *)data.bytes;
    std::vector<Tile> tiles;
    for(size_t i=0;i<data.length/sizeof *words;i+=2*shape.size())
      tiles.push_back({{words+i,words+i+shape.size()},{words+i+shape.size(),words+i+2*shape.size()}});
    return tiles;
  });
  module.def("view_index", [](const std::vector<NSUInteger> &shape,const std::vector<NSInteger> &strides,const std::string &index) {
    TORCH_CHECK(shape.size()==strides.size(),"an index view needs one stride per axis");
    return std::string([MetalViewIndexExpression(shape.data(),strides.data(),shape.size(),@(index.c_str())) UTF8String]);
  });
  pybind11::dict compilation;
#define MESH_METAL_VALUE(name, value) compilation[#name] = value;
#include "metal.def"
#undef MESH_METAL_VALUE
  compilation["LINEAR_PRODUCTS"] = LINEAR_PRODUCTS;
  module.attr("compilation") = compilation;
  module.def("linear_source", &linear_source);
  module.def("linear_groups", &linear_groups);
  module.def("linear_commands", &linear_commands);
  pybind11::class_<Commands,std::shared_ptr<Commands>>(module,"Commands").def(pybind11::init<const std::vector<CommandRecord> &>());
  module.def("linear_split", &linear_split);
  module.def("device_facts", &device_facts);
  module.def("pipeline_facts", &pipeline_facts);
  module.attr("polls") = (int64_t)NCCL_MESH_POLLS;
  module.def("encode", &encode, pybind11::arg("handle"), pybind11::arg("buffers"), pybind11::arg("constants"), pybind11::arg("at"),
             pybind11::arg("gx"), pybind11::arg("gy"), pybind11::arg("threads"));
  module.def("issue", [](const std::vector<Call> &calls, int64_t comm, bool split, std::vector<at::Tensor> outputs) {
    return work_of(issue(calls, comm, split, true), std::move(outputs), comm);
  });
  module.def("attach", &attach);
  module.def("publish_boundary", &publish_boundary);
  module.def("region_empty", [](std::vector<int64_t> sizes, const at::Tensor &like, bool mps, bool held) { return region_empty(sizes, like, mps, held); },
             pybind11::arg("sizes"), pybind11::arg("like"), pybind11::arg("mps"), pybind11::arg("held") = false);
  module.def("address", &address);
  module.def("records", &records);
  module.def("metal_prelude", &metal_prelude);
  module.def("fused_maker", &fused_maker);
  module.def("fused_release", &fused_release);
  pybind11::class_<ScheduleBinding>(module,"Schedule")
    .def(pybind11::init<>())
    .def("append",[](ScheduleBinding &s,const std::shared_ptr<Commands> &commands){commands->append(s.schedule.get());
        s.operands.insert(s.operands.end(),commands->operands.begin(),commands->operands.end());s.steps.push_back(commands);})
    .def("compute",&ScheduleBinding::compute)
    .def("writes",&ScheduleBinding::writes)
    .def("host",&ScheduleBinding::host)
    .def("finish",&ScheduleBinding::finish)
    .def("enqueue",&ScheduleBinding::enqueue)
    .def("completion",&ScheduleBinding::completion);
  pybind11::class_<ViewWrites, std::shared_ptr<ViewWrites>>(module, "ViewWrites")
    .def(pybind11::init<const std::vector<ViewBinding> &, int64_t>())
    .def("realize", &ViewWrites::realize)
    .def("append",[](const std::shared_ptr<ViewWrites> &w,ScheduleBinding &s){w->append(s.schedule.get());
        s.operands.insert(s.operands.end(),w->leases.begin(),w->leases.end());s.steps.push_back(w);})
    .def("completion", &ViewWrites::completion)
    .def("enqueue", [](const std::shared_ptr<ViewWrites> &w){w->enqueue(w);});
  pybind11::class_<Collectives, std::shared_ptr<Collectives>>(module, "Collectives")
    .def(pybind11::init<const std::vector<Call> &, int64_t, std::vector<at::Tensor>,
                        const std::vector<CommandRecord> &, const std::vector<CommandRecord> &>(),
         pybind11::arg("calls"),pybind11::arg("comm"),pybind11::arg("operands"),
         pybind11::arg("inputs")=std::vector<CommandRecord>{},pybind11::arg("outputs")=std::vector<CommandRecord>{})
    .def("realize", &Collectives::realize, pybind11::arg("repetitions")=1)
    .def("append",[](const std::shared_ptr<Collectives> &c,ScheduleBinding &s){c->append(s.schedule.get());
        s.operands.insert(s.operands.end(),c->operands.begin(),c->operands.end());s.steps.push_back(c);})
    .def("completion", &Collectives::completion)
    .def("enqueue", &Collectives::enqueue);
  pybind11::class_<Executions>(module,"Executions")
    .def(pybind11::init<pybind11::object>())
    .def("add",&Executions::add)
    .def("run",&Executions::run)
    .def("completion",&Executions::completion);
  module.def("fused_reconfigure", &fused_reconfigure);
  module.def("chain_slot", &chain_slot);
  module.def("prepare_chain", &prepare_chain);
  module.def("chain_run", &chain_run);
  module.def("chain_completion", &chain_completion);
  module.def("fused_prepare", &fused_prepare, pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("dim"),
             pybind11::arg("sizes"), pybind11::arg("group"), pybind11::arg("adds"),
             pybind11::arg("repeats") = 1, pybind11::arg("batches") = 1);
}
