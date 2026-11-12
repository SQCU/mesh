// A torch program's publish of its sent ranges on torch's MPS stream (design/aot-compilation-change-request.md, appendix
// item 7, "A publish waits for nothing": "a torch MPS event is recorded after the producer's out-variant kernel, the
// publish is done in that event's notification, and torch's command buffer is committed without waiting"; the event
// form torch_mesh's allocator.mm used, mesh fba41159, "an action after torch's work so far, done in a torch MPSEvent's
// notification", the item as mesh 0ffd880e states it, which cites that form alone; design/compilation-coordination.md,
// "The revisions of e4a7398e", TP).  Each call is torch 2.14.0's own,
// as its headers state it (torch/include/ATen/mps/MPSEvent.h, MPSStream.h): MPSEventPool::acquireEvent, MPSEvent::record
// ("records an event on the stream"), MPSEvent::notify ("schedules a notifyListener callback for the event") and
// MPSStream::synchronize(SyncType::COMMIT) ("commit and flush the command buffer").  No classic Metal is called here.
// Built by rdma/mesh-torch.mk against the node's installed torch, as torch_mesh's _stream.mm was (mesh 116b27dc).
// 2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", B3: "The record's
// release must follow torch's work in an MPSEvent notification, as torch_mesh's allocator.mm did its release (fba41159's
// event form, item 7).  It is then bounded by the notification wait already there"; design/aot-compilation-change-
// request.md, appendix item 7, "A publish waits for nothing"; "The revisions of 55b9cbad"): the event form is one
// function (after_torch), which the publish, the program's retirement (mesh_torch_retire) and an execution's end
// (mesh_torch_end) each give their action, where the publish held it alone.  Why: the requirement line "Refactor by
// pattern analysis: merge duplicated patterns found by analysis".  What must stay: every action after torch's work so
// far, in an MPSEvent's notification, the command buffer committed with no wait
#include <ATen/mps/MPSEvent.h>
#include <ATen/mps/MPSStream.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <memory>
#include <vector>

// rdma/mesh-collective.h mesh_host_publish, whose address the caller passes (rdma/mesh_compile.py), with the context it
// takes, so this library binds no mesh symbol of its own
// 2026-11-06 (design/compilation-coordination.md, "The root cause of the copies, and the revised work list", R-1; "The
// revisions of 14b9a757"; rdma/mesh-collective.h mesh_host_publish at mesh 31ee5d4a: "each by its declared transfer,
// `plan` its function's state section ... and `transfer` its index among the function's declared transfers"), reversing
// a range named by its first row's section: the publish takes (context, plan, transfer, execution), and
// rdma/mesh-section.h, included for struct mesh_section alone, goes with it.  What must stay: no mesh symbol bound here
typedef void (*mesh_publish)(void *, uint64_t, uint32_t, uint64_t);
// rdma/mesh-dataflow.h mesh_retire(struct hdr *, uint64_t client), whose address the caller passes as it passes
// mesh_host_publish's (2026-11-05, "The revisions of 55b9cbad", B3).  What must stay: no mesh symbol bound here
typedef void (*mesh_retirement)(void *, uint64_t);

// the notifications scheduled and not yet run, which a plan's retirement waits out before its claim is closed
// (rdma/mesh_compile.py _retire: deferred reclamation, design/disjoint-pages-change-request.md C3.4).
// 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
// revisions of 4e78f6f4"): the program's retirement waits them out before its root is closed (mesh-dataflow.h
// mesh_retire) and its mapping released, where a plan's claim was closed.  What must stay: the count's release order
// 2026-11-05 (B3; "The revisions of 55b9cbad"), reversing "before its root is closed": the retirement is itself one of
// them (mesh_torch_retire), and rdma/mesh_compile.py _retire waits them all out, bounded by MESH_PLATFORM_NS, before
// it releases the program's mapping.  What must stay: the count's release order
static std::atomic<uint64_t> notifying{0};
// the executions whose end torch's MPS stream has signalled, as a count (execution t's end raises it to t + 1), never
// lowered (2026-11-05, design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", V5;
// design/disjoint-pages-change-request.md D.3.3: "The queue signals each step's end with `signalEvent:value:` after its
// last commit", which the host reads before it writes a step's inputs; "The revisions of 55b9cbad").  What must stay:
// raised by the notifications alone, read with acquire
static std::atomic<uint64_t> ended{0};

// `action` done after torch's work so far on the thread's current MPS stream, in the notification of a torch MPSEvent
// recorded there, then that stream's command buffer committed, with no wait; counted in `notifying` until it has run.
// Its caller is Python, never on the stream's serial queue, so every call takes torch's lock (fba41159's `off`).  true;
// false where torch scheduled no notification (`action` not done); torch's exceptions propagate
static bool after_torch(void (^action)(void)) {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  std::shared_ptr<at::mps::MPSEvent> done = at::mps::getMPSEventPool()->acquireEvent(false, stream);
  done->record(true);
  notifying.fetch_add(1, std::memory_order_relaxed);
  if (!done->notify(true, ^(id<MTLSharedEvent>, uint64_t) {
        action();
        notifying.fetch_sub(1, std::memory_order_release);
        (void)done;
      })) {
    notifying.fetch_sub(1, std::memory_order_relaxed);
    return false;
  }
  stream->synchronize(at::mps::SyncType::COMMIT);
  return true;
}

// execution `execution` of each of `count` sections published by `publish(context, section, execution)` after torch's
// work so far (after_torch).  0; EIO where torch threw or scheduled no notification (nothing published)
// 2026-11-06 (R-1; "The revisions of 14b9a757"), reversing `sections`: execution `execution` of each of `count`
// declared transfers of plan `plan`, `transfers` their indices, published by `publish(context, plan, transfer,
// execution)`.  What must stay: every publish after torch's work so far, in one notification
extern "C" int mesh_torch_publish(mesh_publish publish, void *context, uint64_t plan, const uint32_t *transfers,
                                  uint32_t count, uint64_t execution) {
  try {
    const std::vector<uint32_t> released(transfers, transfers + count);
    return after_torch(^{
             for (const uint32_t transfer : released) publish(context, plan, transfer, execution);
           })
               ? 0
               : EIO;
  } catch (...) {
    return EIO;
  }
}

// the program's retirement, `retire(region, client)` (mesh-dataflow.h mesh_retire), after torch's work so far on the
// program's values (after_torch), where rdma/mesh_compile.py _retire synchronized the device first (B3: "`_retire`'s
// device synchronization ... is an unbounded device wait").  0; EIO where torch threw or scheduled no notification
// (nothing retired)
extern "C" int mesh_torch_retire(mesh_retirement retire, void *region, uint64_t client) {
  try {
    return after_torch(^{
             retire(region, client);
           })
               ? 0
               : EIO;
  } catch (...) {
    return EIO;
  }
}

// execution `execution`'s end after torch's work so far (after_torch): `ended` raised to execution + 1 (release),
// never lowered, so ends signalled in any order leave it at the greatest, as mesh_host_publish raises a supplied word
// (mesh-collective.h).  0; EIO where torch threw or scheduled no notification
extern "C" int mesh_torch_end(uint64_t execution) {
  try {
    return after_torch(^{
             uint64_t seen = ended.load(std::memory_order_relaxed);
             while (seen < execution + 1 &&
                    !ended.compare_exchange_weak(seen, execution + 1, std::memory_order_release,
                                                 std::memory_order_relaxed)) {
             }
           })
               ? 0
               : EIO;
  } catch (...) {
    return EIO;
  }
}

extern "C" uint64_t mesh_torch_notifying(void) { return notifying.load(std::memory_order_acquire); }
extern "C" uint64_t mesh_torch_ended(void) { return ended.load(std::memory_order_acquire); }
