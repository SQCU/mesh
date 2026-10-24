// torch's MPS stream and the host as libnccl-mesh's streams (nccl.h): every collective of torch_mesh/backend.py and
// torch's functional collectives on MPS is one call of `issue`, a group of calls each one kind (an all-reduce, reduce,
// broadcast, the V all-gather, reduce-scatter and all-to-all, a send, a receive; a reduction by
// any of the library's operators, the partials merge among them).  On MPS tensors the group is encoded in torch's stream's command buffer and open compute encoder (its kernels
// join torch's own) with a workspace allocator (MPS tensors from torch's caching allocator, held until the group is
// complete, then reused in the stream's order), committed by the one commit rule (below) without waiting; on CPU
// tensors it is the host path, complete where it returns.  A group issued and not completed (an async op, a functional
// collective) is a MeshWork: its wait encodes the rest (its landings, later rounds and results) on the stream where it
// is waited.
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
#include <atomic>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <array>
#include <vector>
#include <unistd.h>
#include "nccl.h"

// design/data-model.md section 6: each wait here (a turn of torch's queue, which dispatch_sync waits for, with any
// command buffer waited inside it; a command buffer's end) carries a deadline fixed when it is made, its caller's or
// 5 s (mesh.h MESH_PLATFORM_NS), on its thread's timer, armed as the wait is made and disarmed at its end; the timer's
// expiry ends the process, the wait and its deadline on stderr.
struct Deadline {
  struct Named {
    const char *wait;
    uint64_t ns;
  };
  dispatch_source_t timer;
  explicit Deadline(const char *wait, uint64_t ns = 0) : timer(thread_timer()) {
    Named *named = (Named *)dispatch_get_context(timer);
    *named = {wait, ns ? std::min<uint64_t>(ns, INT64_MAX) : UINT64_C(5000000000)};
    dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, (int64_t)named->ns), DISPATCH_TIME_FOREVER, 0);
  }
  ~Deadline() { dispatch_source_set_timer(timer, DISPATCH_TIME_FOREVER, DISPATCH_TIME_FOREVER, 0); }
  static void expired(void *context) {
    const Named *named = (const Named *)context;
    fprintf(stderr, "torch_mesh: %s: past its deadline of %.3f s; the program ends (design/data-model.md section 6)\n", named->wait,
            (double)named->ns / 1e9);
    _exit(1);
  }
  static dispatch_source_t thread_timer() {
    thread_local dispatch_source_t made = nullptr;
    if (made) return made;
    made = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0));
    dispatch_set_context(made, new Named{"", 0});
    dispatch_source_set_event_handler_f(made, expired);
    dispatch_source_set_timer(made, DISPATCH_TIME_FOREVER, DISPATCH_TIME_FOREVER, 0);
    dispatch_resume(made);
    return made;
  }
};


// The commit rule (README, Performance), one for every caller: eager calls and compiled graphs, the library's own
// callers and a user's program alike, with no region, mode or counter a caller enters.  Its work is throughput: the GPU
// fed with the fewest command buffers while the host keeps issuing.  The GPU starts a command buffer at most every
// 13-19 us whatever it holds (metal-microbench docs/measurement.md, 2026-10-02), so a buffer for each publication (work
// a peer waits on) is a floor of its own.  Each buffer the library commits signals its count on `progress` as the GPU
// reaches its end (a shared event's value is the GPU's progress at once; a completion handler runs 7-9 us later, at the
// 99th percentile 20-60).  A publication is committed at once while at most one of those buffers is still in flight,
// read from the event's value then: double buffering, one committed buffer queued behind the one the GPU runs, so a
// commit overlaps the GPU's work (the 10-02 rule, mesh 3c00d37: 12.7 us a crossing compiled); otherwise it joins
// torch's open command buffer.  The next
// publication applies the rule again, and a host wait commits the open buffer, since nothing else would: the
// library's (commit, host_wait) and torch's own (torch.mps.synchronize, torch.accelerator.synchronize, .item() and
// .cpu() commit torch's open buffer and wait: MPSStream::synchronize COMMIT_AND_WAIT).  A host that blocks outside
// torch while a publication waits stalls its peer, which is the program's to avoid; the peer's wait ends at its
// deadline (design/data-model.md section 6).  In flight counts the buffers this library committed, not torch's own.

// One torch MPS stream's commits and group workspace, held by the stream (made at its first group and never freed:
// torch's streams live as long as the process, and the library's workspace allocator holds it as its context).  Its
// commit state (committed, open) is read and written by the host inside torch's queue (dispatch_sync), where torch
// encodes its own work, so it takes no lock, no caller enters it, and no thread but the host's touches torch's stream.
// The rest (streamed, scratch, holders) is the host's, and workspace is freed outside torch's queue (a free may enter
// that queue: torch's MPSAllocator adds completed handlers through it).
struct MeshStream {
  at::mps::MPSStream *const mps;
  id<MTLSharedEvent> const progress;
  // the buffers this stream committed, the k-th signalling k on progress as the GPU reaches its end
  uint64_t committed = 0;
  // the root command buffer a publication joined (MPSCommandBuffer rootCommandBuffer, which commitAndContinue
  // commits), retained until it is committed by this stream or by torch's own commits, which carry the publication
  // with them; nil while none waits
  id<MTLCommandBuffer> open = nil;
  // whether a group was issued on this stream since the host path last waited for it
  bool streamed = false;
  // the workspace of the group being encoded (the library's workspace allocator's context is the MeshStream, which
  // outlives every call), then each issued group's, by ticket, until the library has retired it (ncclMeshRetired)
  std::vector<at::Tensor> scratch;
  std::map<uint64_t, std::vector<at::Tensor>> holders;

  explicit MeshStream(at::mps::MPSStream *stream) : mps(stream), progress([stream->device() newSharedEvent]) {}

  // On torch's queue: whether a publication still waits in torch's open buffer, the root it joined not yet committed
  // (one torch committed needs no buffer of ours, which would cost the GPU a start of its own).
  bool waiting() {
    if (open && open.status < MTLCommandBufferStatusCommitted) return true;
    [open release];
    open = nil;
    return false;
  }

  // On torch's queue: the open buffer committed (and waited for, where `wait`), its end signalling the next count.
  void signalled(bool wait) {
    mps->endKernelCoalescing();
    [mps->commandBuffer() encodeSignalEvent:progress value:++committed];
    mps->synchronize(wait ? at::mps::SyncType::COMMIT_AND_WAIT : at::mps::SyncType::COMMIT);
    [open release];
    open = nil;
  }

  // On torch's queue, as a group's encoding ends, the rule: a publication encoded in torch's open buffer, committed
  // with every publication waiting there while at most one buffer is in flight (double buffering), else left to join
  // it.
  void published() {
    if (!waiting()) open = [mps->commandBuffer().rootCommandBuffer retain];
    if (committed - progress.signaledValue <= 1) signalled(false);
  }

  // On the host: the workspace `held` of the group issued as `ticket` kept until the library retires it (0: a group
  // complete in the stream's order once encoded, whose workspace returns to torch's allocator here, where later work
  // reuses it after the group's), and the workspace of every group retired by now released.
  void retire(uint64_t ticket, std::vector<at::Tensor> held) {
    if (ticket) holders[ticket] = std::move(held);
    holders.erase(holders.begin(), holders.upper_bound(ncclMeshRetired()));
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

// A host wait of the library's on stream s, on torch's queue: a publication still waiting committed, and with `wait`
// the stream committed and waited for (torch's COMMIT_AND_WAIT, taken here on its queue), within its Deadline.
static void host_wait(MeshStream &s, bool wait) {
  MeshStream *self = &s;
  const Deadline deadline("host_wait: a turn of torch's queue and, waited, its command buffers' end");
  at::mps::dispatch_sync_with_rethrow(s.mps->queue(), ^{
    @autoreleasepool {
      if (self->waiting()) self->signalled(wait);
      else if (wait) self->mps->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
    }
  });
}

// Before a host wait outside torch's streams (a recording's compile meeting): every stream's waiting
// publication committed.
static void commit() {
  for (MeshStream *s : streams_made()) host_wait(*s, false);
}

// The library's workspace allocator: an MPS tensor of torch's caching allocator, kept in the MeshStream's scratch
// (`context`) until the group is complete in the stream's order (MeshStream::retire); NULL where torch's allocator
// throws, which fails the group as a value (nccl-mesh.c: "out of GPU memory") instead of an exception thrown through
// the library.  It runs on torch's queue, where an allocator that frees cached buffers would enter that queue again
// (MPSAllocator release_cached_buffers): rdma/torch-mesh/README.md, the stream's commit state.
static void *workspace(size_t bytes, void *context) {
  auto *s = static_cast<MeshStream *>(context);
  try {
    s->scratch.push_back(at::empty({(int64_t)bytes}, at::TensorOptions().dtype(at::kByte).device(at::kMPS)));
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

// The evidence torch_mesh/evidence.py reads (design/heterogeneity.md R2, R3): on a watched communicator, a timing
// event of torch's pool recorded on its stream as each collective is issued (0: the GPU has finished the work before
// it) and once its wait is encoded (1: the GPU has passed it), and at each step's boundary (2), in order.
static std::unordered_map<int64_t, std::vector<std::pair<int64_t, int64_t>>> marks_;

static void mark(int64_t comm, int64_t kind) {
  if (marks_.empty()) return;
  auto found = marks_.find(comm);
  if (found == marks_.end()) return;
  auto pool = at::mps::getMPSEventPool();
  const auto id = pool->acquireEvent(true);
  pool->recordEvent(id, false);
  found->second.emplace_back(kind, (int64_t)id);
}

// A group encoded in one turn of torch's stream queue: its command buffer and open encoder, the group encoded into them,
// and the commit rule applied where it published.  Its workspace (`held`) is held:
// split, it is issued and not completed, its ticket (0: complete at once) keeping them until the library retires it.
static uint64_t on_stream(const std::function<ncclResult_t(ncclMeshStream *)> &call, bool split, int64_t comm, std::vector<at::Tensor> held) {
  const auto *encode = &call;
  mark(comm, 0);
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  s->scratch = std::move(held);
  __block ncclResult_t result = ncclSuccess;
  __block ncclMeshIssue issue = {0, 0};
  {
    const Deadline deadline("on_stream: a turn of torch's queue, a group encoded");
    dispatch_sync(stream->queue(), ^{
      @autoreleasepool {
        ncclMeshStream on{(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder(), workspace, s, split ? &issue : nullptr};
        result = (*encode)(&on);
        if (!split || issue.published) s->published();
      }
    });
  }
  s->streamed = true;
  s->retire(issue.ticket && !result ? issue.ticket : 0, std::exchange(s->scratch, {}));
  checked(result);
  if (!issue.ticket) mark(comm, 1);
  return issue.ticket;
}

// every issued group through `ticket` completed on torch's stream where it is waited, the turn of torch's queue within
// `ns` (0: the Deadline's 5 s)
static void complete(uint64_t ticket, uint64_t ns = 0) {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  if (ncclMeshRetired() >= ticket) return s->retire(0, {});
  __block ncclResult_t result = ncclSuccess;
  {
    const Deadline deadline("complete: a turn of torch's queue, a group's completion encoded", ns);
    dispatch_sync(stream->queue(), ^{
      @autoreleasepool {
        ncclMeshStream on{(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder(), workspace, s, nullptr};
        int published = 0;
        result = ncclMeshComplete(ticket, &on, &published);
        if (published) s->published();
      }
    });
  }
  s->retire(0, std::exchange(s->scratch, {}));
  checked(result);
}

// c10d's Work for a group issued and not completed: waiting (or synchronizing, asking whether it completed, or
// asking for its future) encodes the rest on torch's current stream, where every later kernel reads its results; a
// wait's timeout is its Deadline (c10d's kNoTimeout, 0: 5 s)
class MeshWork : public c10d::Work {
 public:
  MeshWork(uint64_t ticket, std::vector<at::Tensor> outputs, int64_t comm)
      : c10d::Work(-1, c10d::OpType::UNKNOWN), ticket_(ticket), outputs_(std::move(outputs)), comm_(comm) {}
  bool isCompleted() override { finish(); return true; }
  bool isSuccess() const override { return finished_; }
  bool wait(std::chrono::milliseconds timeout) override { finish((uint64_t)std::max<int64_t>(timeout.count(), 0) * 1000000); return true; }
  void synchronize() override { finish(); }
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    finish();
    auto future = c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
    future->markCompleted(c10::IValue(outputs_));
    return future;
  }

 private:
  void finish(uint64_t ns = 0) {
    if (finished_) return;
    finished_ = true;
    complete(ticket_, ns);
    mark(comm_, 1);
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
enum Kind : int64_t { ALLREDUCE, REDUCE, BROADCAST, ALLGATHER, REDUCE_SCATTER, ALLTOALL, SEND, RECV, WRITE };
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

// A group's operands: one device's tensors (MPS: the Metal path; CPU: the host path), each contiguous; a non-contiguous
// one fails as a value, as nothing is copied to make it one (the operator, 2026-10-21: "zero copy methods must always
// be used over access by copy because memory bandwidth and flops are too scarce on this hardware api"; "code which
// can't perform well can fail instead of being run").
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

// A call's operands and counts, made before its group is encoded (a staging copy is torch's work on its own queue).
struct Issue {
  int64_t kind, op, root;
  std::vector<Operand> sends, receives;
  std::vector<size_t> counts;
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
  case BROADCAST: i.sends = {g.moving(tensor(sends, 0))}; break;
  case WRITE:
    i.sends = {g.moving(tensor(sends, 0))}; i.receives = {g.moving(tensor(receives, 0))};
    TORCH_CHECK(counts.size() == 1, "a one-way write names one destination");
    i.counts = {(size_t)counts[0]};
    break;
  case ALLGATHER: case REDUCE_SCATTER: {
    // MPI_Allgatherv and MPI_Reduce_scatter (nccl.h ncclMeshAllGatherV, ncclMeshReduceScatterV): rank r's segment is
    // counts[r] elements, the segments packed in rank order in the whole operand (the receive's, the send's), counted
    // in the operand's NCCL elements; equal counts are the library's equal program
    const bool reduces = what == REDUCE_SCATTER;
    const at::Tensor &whole = reduces ? tensor(sends, 0) : tensor(receives, 0);
    i.sends = {reduces ? g.reducing(tensor(sends, 0), i.op) : g.moving(tensor(sends, 0))};
    i.receives = {reduces ? g.reducing(tensor(receives, 0), i.op) : g.moving(tensor(receives, 0))};
    const size_t per = whole.numel() ? (reduces ? i.sends : i.receives)[0].count / (size_t)whole.numel() : 1;
    for (auto c : counts) i.counts.push_back((size_t)c * per);
    break;
  }
  case ALLTOALL: {
    // MPI_Alltoallv with its counts on the host (nccl.h ncclMeshAlltoAllV; NCCL's all-to-all, gather and scatter its
    // forms, a gather and a scatter of any sizes the rooted V): segments packed in rank order, a buffer that moves none
    // absent
    TORCH_CHECK(sends.size() == 1 && receives.size() == 1 && (sends[0] || receives[0]), "an all-to-all names its send and receive buffers");
    i.sends = {sends[0] ? g.moving(*sends[0]) : Operand{}};
    i.receives = {receives[0] ? g.moving(*receives[0]) : Operand{}};
    const at::Tensor &whole = sends[0] ? *sends[0] : *receives[0];
    const size_t per = whole.numel() ? (sends[0] ? i.sends : i.receives)[0].count / (size_t)whole.numel() : 1;
    for (auto c : counts) i.counts.push_back((size_t)c * per);
    break;
  }
  case SEND: i.sends = {g.moving(tensor(sends, 0))}; break;
  case RECV: i.receives = {g.moving(tensor(receives, 0))}; break;
  default: TORCH_CHECK(false, "no collective of kind ", what);
  }
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
    case WRITE: result = ncclMeshWrite(a.at, b, a.count, a.type, (int)i.root, (int)i.counts[0], comm, s); break;
    case ALLGATHER: result = ncclMeshAllGatherV(a.at, b, i.counts.data(), a.type, comm, s); break;
    case REDUCE_SCATTER: result = ncclMeshReduceScatterV(a.at, b, i.counts.data(), a.type, op, comm, s); break;
    case ALLTOALL:
      result = ncclMeshAlltoAllV(a.at, i.counts.data(), b, i.counts.data() + i.counts.size() / 2, a.at ? a.type : i.receives[0].type, comm, s);
      break;
    case SEND: result = ncclSend(a.at, a.count, a.type, (int)i.root, comm, s); break;
    case RECV: result = ncclRecv(b, i.receives[0].count, i.receives[0].type, (int)i.root, comm, s); break;
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
static uint64_t issue(const std::vector<Call> &calls, int64_t comm, bool split, bool unlock) {
  Group g;
  for (const Call &c : calls)
    for (const Tensors *list : {&std::get<1>(c), &std::get<2>(c)})
      for (const std::optional<at::Tensor> &t : *list) g.metal |= t && t->is_mps();
  std::vector<Issue> issues;
  for (const Call &c : calls) issues.push_back(prepare(c, g));
  uint64_t ticket = 0;
  if (g.metal) {
    ticket = on_stream([&](ncclMeshStream *s) { return encoded(issues, communicator(comm), s); }, split, comm, {});
  } else {
    bool streamed = false;
    for (MeshStream *s : streams_made()) streamed |= s->streamed;
    if (streamed) {
      complete(UINT64_MAX);
      const MeshStream *current = &stream_of(at::mps::getCurrentMPSStream());
      for (MeshStream *s : streams_made())
        if (s->streamed || s == current) {
          host_wait(*s, true);
          s->retire(0, {});
          s->streamed = false;
        }
    }
    std::optional<pybind11::gil_scoped_release> unlocked;
    if (unlock) unlocked.emplace();
    checked(encoded(issues, communicator(comm), nullptr));
  }
  return ticket;
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

static id<MTLResidencySet> fused_residency_;
// A kernel that runs a fused crossing's steps (`fused`) reaches the library's counts, landing buffers and records by
// address, through the residency set, which Metal does not hazard-track: its encoder declares every buffer of the set,
// so its pass is ordered against the library's dispatches that name them (nccl-mesh-metal.h metal_uses).
static void encode(int64_t handle, const std::vector<std::optional<at::Tensor>> &buffers, const std::vector<int64_t> &constants,
                   int64_t at, int64_t gx, int64_t gy, int64_t threads, bool fused) {
  if (handle < 0 || handle >= (int64_t)pipelines_.size()) throw std::runtime_error("encode: no kernel " + std::to_string(handle));
  id<MTLComputePipelineState> state = pipelines_[handle];
  std::vector<id<MTLBuffer>> bound(buffers.size(), nil);
  std::vector<NSUInteger> offsets(buffers.size(), 0);
  for (size_t i = 0; i < buffers.size(); i++)
    if (buffers[i]) {
      const at::Tensor &t = *buffers[i];
      if (!t.is_mps()) throw std::runtime_error("encode: buffer " + std::to_string(i) + " is not an MPS tensor");
      bound[i] = at::native::mps::getMTLBufferStorage(t);
      offsets[i] = (NSUInteger)(t.storage_offset() * t.element_size());
    }
  std::vector<uint32_t> words(constants.begin(), constants.end());
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  const Deadline deadline("encode: a turn of torch's queue, a kernel encoded");
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:state];
    for (size_t i = 0; i < bound.size(); i++)
      if (bound[i]) [encoder setBuffer:bound[i] offset:offsets[i] atIndex:i];
    [encoder setBytes:words.data() length:words.size() * sizeof(uint32_t) atIndex:(NSUInteger)at];
    if (fused && fused_residency_)
      for (id<MTLAllocation> held in fused_residency_.allAllocations)
        [encoder useResource:(id<MTLResource>)held usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, 1) threadsPerThreadgroup:MTLSizeMake((NSUInteger)threads, 1, 1)];
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
         {R, K, N, bias ? 1 : 0, ldw, K0, ld0, ld1}, 4, groups, 1, split * 32, false);
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
  encode(found->second, buffers, words, 20, first, 1, split * 32, false);
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

// Held MPS leases (region_empty's `held`), by their bytes: each released once the stream's work on it had completed.
static std::multimap<size_t, void *> held_;

static at::Tensor region_empty(std::vector<int64_t> sizes, const at::Tensor &like, bool mps, bool held) {
  const at::ScalarType dtype = like.scalar_type();
  const size_t bytes = std::max<size_t>(1, (size_t)c10::multiply_integers(sizes) * c10::elementSize(dtype));
  void *memory = nullptr;
  {
    std::lock_guard<std::mutex> guard(region_tensors_lock_);
    const auto found = held ? held_.find(bytes) : held_.end();
    if (found != held_.end()) {
      memory = found->second;
      held_.erase(found);
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

// The region tensors' leases live now: each its address and bytes.
static std::vector<std::pair<uint64_t, uint64_t>> records() {
  std::lock_guard<std::mutex> guard(region_tensors_lock_);
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (auto &w : region_tensors_) out.emplace_back(w.first, w.second);
  return out;
}

// Fused crossings (torch_mesh/fused.py; nccl.h ncclMeshFused): the one-shot exchange, ncclSend
// of a rank's part to every other rank and ncclRecv of each one's part into a landing tensor of its own, in one group,
// whose steps the caller's kernels run on torch's stream: the kernel that makes the rank's part writes it into its send
// tensor and publishes it to every peer (one store of each holding's `ready`), and the kernel after the call awaits
// every landing and reads each peer's part from its landing tensor, adding them to its own (an all-reduce's sum, which
// the caller makes); the call dispatches no step (a fused group's channels complete in the stream's order with no
// dispatch: nccl-mesh.c complete, complete_track).  The send tensor and every landing tensor lie in region tensors
// (torch_mesh.empty: a lease), each starting on a page, which the bridge sends from and lands in, zero copy.  The call
// is the library's classic form on torch's command buffer and open encoder with a residency set: the library writes
// each publication's and each landing's argument record into the lists this call gives it room in (n - 1 of each:
// an all-reduce's one-shot counts) and returns their bindings, GPU addresses the caller's kernels reach by
// address, and adds every buffer they name to the set, which is resident for every command buffer of each queue it
// was added to (the device's one set, added to a stream's queue at its first fused call); nothing removes a buffer from
// it.  A call the library does not fuse (a tensor that is no region tensor or does not start on a page) fails as a
// value, the library's refusal; nothing falls back to a slower form.  The peers are every other rank, in rank order.
static std::vector<void *> fused_queues_;

static id<MTLResidencySet> fused_residency(at::mps::MPSStream *stream) {
  if (!fused_residency_) {
    NSError *error = nil;
    fused_residency_ = [stream->device() newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
    TORCH_CHECK(fused_residency_, "fused crossing: torch's device made no residency set: ",
                error ? error.localizedDescription.UTF8String : "no reason given");
  }
  void *queue = (__bridge void *)stream->commandQueue();
  bool added = false;
  for (void *q : fused_queues_) added |= q == queue;
  if (!added) {
    id<MTLResidencySet> set = fused_residency_;
    const Deadline deadline("fused_residency: a turn of torch's queue");
    dispatch_sync(stream->queue(), ^{ [stream->commandQueue() addResidencySet:set]; });
    fused_queues_.push_back(queue);
  }
  return fused_residency_;
}

// The library's Metal prelude (ncclMeshMetalPrelude): rdma/mesh.metal as the loaded library was built with it, then its
// steps; the text a kernel that runs them is compiled after.
static std::string metal_prelude() { return ncclMeshMetalPrelude(); }

// The fused exchange of `send` (this rank's part) and `landings` (where each other rank's part lands, in rank order),
// contiguous MPS region tensors of one dtype and size, on communicator `comm`, each landing's wait bounded by `polls`
// (nccl.h ncclMeshStream polls; _stream.polls the library's most): one group of ncclSend(send) and ncclRecv(landing)
// toward each other rank, on torch's stream, the stream's `fused` set; each publication's bindings and each landing's,
// in rank order (ncclMeshFused, six GPU addresses each), `repeats` crossings at once, each `stride` bytes past the last
// (ncclMeshFused repeats).  A dtype NCCL lacks moves as its bytes.  Nothing is committed here: the publication is the
// caller's kernel, encoded after this returns (fused_published).
static std::pair<std::vector<std::vector<uint64_t>>, std::vector<std::vector<uint64_t>>> fused_exchange(
    const at::Tensor &send, const std::vector<at::Tensor> &landings, int64_t comm, int64_t polls, int64_t repeats, int64_t stride) {
  int rank = 0, ranks = 0;
  checked(ncclCommUserRank(communicator(comm), &rank));
  checked(ncclCommCount(communicator(comm), &ranks));
  TORCH_CHECK(landings.size() == (size_t)(ranks - 1), "fused exchange: a landing tensor for each of the ", ranks - 1,
              " other ranks, not ", landings.size());
  TORCH_CHECK(polls >= 0 && polls <= (int64_t)UINT32_MAX, "fused exchange: polls ", polls, " is not a 32-bit count of loads");
  Group g{true};
  const Operand out = g.moving(send);
  std::vector<Operand> in;
  for (const at::Tensor &landing : landings) in.push_back(g.moving(landing));
  std::vector<std::array<uint64_t, 6>> publish(landings.size()), land(landings.size());
  ncclMeshFused steps = {reinterpret_cast<uint64_t(*)[6]>(publish.data()), reinterpret_cast<uint64_t(*)[6]>(land.data()),
                         (uint32_t)landings.size(), (uint32_t)landings.size(), (uint32_t)repeats, (size_t)stride};
  ncclMeshFused *const records = &steps;
  const Operand *const ins = in.data();
  __block ncclResult_t result = ncclSuccess;
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  id<MTLResidencySet> residency = fused_residency(stream);
  const ncclComm_t c = communicator(comm);
  {
    const Deadline deadline("fused_exchange: a turn of torch's queue, a group encoded");
    dispatch_sync(stream->queue(), ^{
      @autoreleasepool {
        ncclMeshStream on = {(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder(), workspace, s, nullptr,
                             nullptr, (__bridge void *)residency, (uint32_t)polls, records};
        ncclResult_t made = ncclGroupStart();
        for (int q = 0, i = 0; q < ranks && !made; q++) {
          if (q == rank) continue;
          made = ncclSend(out.at, out.count, out.type, q, c, &on);
          if (!made) made = ncclRecv((void *)ins[i].at, ins[i].count, ins[i].type, q, c, &on);
          i++;
        }
        const ncclResult_t ended = ncclGroupEnd();
        result = made ? made : ended;
      }
    });
  }
  s->streamed = true;
  s->retire(0, std::exchange(s->scratch, {}));
  checked(result);
  std::vector<std::vector<uint64_t>> published, landed;
  for (uint32_t k = 0; k < steps.publishes; k++) published.emplace_back(publish[k].begin(), publish[k].end());
  for (uint32_t k = 0; k < steps.lands; k++) landed.emplace_back(land[k].begin(), land[k].end());
  return {published, landed};
}

// A declared set of routed writes. Bindings name physical channels, including
// parallel cables; relay readiness stays in the compiled stream entries.
static std::pair<std::vector<std::vector<uint64_t>>, std::vector<std::vector<uint64_t>>> fused_writes(
    const at::Tensor &send, const std::vector<at::Tensor> &landings,
    const std::vector<std::array<int, 2>> &pairs, int64_t comm, int64_t polls) {
  TORCH_CHECK(landings.size() == pairs.size(), "fused writes: one landing lease per declared write");
  TORCH_CHECK(polls >= 0 && polls <= (int64_t)UINT32_MAX, "fused writes: polls is a 32-bit count of loads");
  const ncclComm_t c = communicator(comm);
  int channels = 0;
  checked(ncclMeshCommSample(c, nullptr, 0, &channels));
  Group g{true};
  const Operand out = g.moving(send);
  std::vector<Operand> in;
  for (const at::Tensor &landing : landings) {
    in.push_back(g.moving(landing));
    TORCH_CHECK(in.back().count == out.count && in.back().type == out.type,
                "fused writes: landing extent and dtype must match the source");
  }
  std::vector<std::array<uint64_t, 6>> publish(channels), land(channels);
  ncclMeshFused steps = {reinterpret_cast<uint64_t(*)[6]>(publish.data()), reinterpret_cast<uint64_t(*)[6]>(land.data()),
                         (uint32_t)channels, (uint32_t)channels};
  ncclMeshFused *const records = &steps;
  const Operand *const ins = in.data();
  const auto *const endpoints = pairs.data();
  const size_t count = pairs.size();
  __block ncclResult_t result = ncclSuccess;
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  MeshStream *s = &stream_of(stream);
  id<MTLResidencySet> residency = fused_residency(stream);
  {
    const Deadline deadline("fused_writes: a turn of torch's queue, a group encoded");
    dispatch_sync(stream->queue(), ^{
      @autoreleasepool {
        ncclMeshStream on = {(__bridge void *)stream->commandBuffer(), (__bridge void *)stream->commandEncoder(), workspace, s, nullptr,
                             nullptr, (__bridge void *)residency, (uint32_t)polls, records};
        ncclResult_t made = ncclGroupStart();
        for (size_t i = 0; i < count && !made; i++)
          made = ncclMeshWrite(out.at, (void *)ins[i].at, out.count, out.type, endpoints[i][0], endpoints[i][1], c, &on);
        const ncclResult_t ended = ncclGroupEnd();
        result = made ? made : ended;
      }
    });
  }
  s->streamed = true;
  s->retire(0, std::exchange(s->scratch, {}));
  checked(result);
  std::vector<std::vector<uint64_t>> published, landed;
  for (uint32_t k = 0; k < steps.publishes; k++) published.emplace_back(publish[k].begin(), publish[k].end());
  for (uint32_t k = 0; k < steps.lands; k++) landed.emplace_back(land[k].begin(), land[k].end());
  return {published, landed};
}

// A fused crossing's publishing kernel just encoded on torch's stream (fused.encode): a publication under the stream's
// one commit rule (MeshStream::published), as a group that published is.
static void fused_published() {
  MeshStream *s = &stream_of(at::mps::getCurrentMPSStream());
  const Deadline deadline("fused_published: a turn of torch's queue");
  dispatch_sync(s->mps->queue(), ^{
    @autoreleasepool {
      s->published();
    }
  });
}


// The group's calls (torch_mesh/backend.py): `issue` a group of calls, issued and not completed where `split` (a
// MeshWork to wait on, its outputs `outputs`), else None (complete).
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pipeline", &pipeline);
  module.def("linear_source", &linear_source);
  module.def("linear_split", &linear_split);
  module.def("device_facts", &device_facts);
  module.def("pipeline_facts", &pipeline_facts);
  module.attr("polls") = (int64_t)NCCL_MESH_POLLS;
  module.def("encode", &encode, pybind11::arg("handle"), pybind11::arg("buffers"), pybind11::arg("constants"), pybind11::arg("at"),
             pybind11::arg("gx"), pybind11::arg("gy"), pybind11::arg("threads"), pybind11::arg("fused") = false);
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
  module.def("issue", [](const std::vector<Call> &calls, int64_t comm, bool split, std::vector<at::Tensor> outputs) {
    return work_of(issue(calls, comm, split, true), std::move(outputs), comm);
  });
  module.def("attach", &attach);
  module.def("commit", &commit);
  module.def("region_empty", [](std::vector<int64_t> sizes, const at::Tensor &like, bool mps) { return region_empty(sizes, like, mps); });
  module.def("address", &address);
  module.def("records", &records);
  module.def("metal_prelude", &metal_prelude);
  module.def("fused_exchange", &fused_exchange);
  module.def("fused_writes", &fused_writes);
  module.def("fused_published", &fused_published);
}
