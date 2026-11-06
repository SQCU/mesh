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
#include <ATen/mps/MPSEvent.h>
#include <ATen/mps/MPSStream.h>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// rdma/mesh-call.h struct mesh_section, field for field, as rdma/mesh.py Section binds it: mesh's C headers are C, and
// C++ cannot include them (mesh.h's stdatomic.h and infiniband/verbs.h)
struct mesh_section { uint32_t first, pages; size_t bytes; uint32_t count, stride; };
static_assert(sizeof(mesh_section) == 24 && offsetof(mesh_section, bytes) == 8 && offsetof(mesh_section, stride) == 20,
              "mesh_section: rdma/mesh-call.h struct mesh_section");
// rdma/mesh-collective.h mesh_host_publish, whose address the caller passes (rdma/mesh_compile.py), with the context it
// takes, so this library binds no mesh symbol of its own
typedef void (*mesh_publish)(void *, struct mesh_section, uint64_t);

// the notifications scheduled and not yet run, which a plan's retirement waits out before its claim is closed
// (rdma/mesh_compile.py _retire: deferred reclamation, design/disjoint-pages-change-request.md C3.4)
static std::atomic<uint64_t> notifying{0};

// execution `execution` of each of `count` sections published by `publish(context, section, execution)` in the
// notification of a torch MPSEvent recorded on the thread's current MPS stream after torch's work so far, then that
// stream's command buffer committed, with no wait.  Its caller is Python, never on the stream's serial queue, so every
// call takes torch's lock (fba41159's `off`).  0; EIO where torch threw or scheduled no notification (nothing published)
extern "C" int mesh_torch_publish(mesh_publish publish, void *context, const struct mesh_section *sections,
                                  uint32_t count, uint64_t execution) {
  try {
    const std::vector<mesh_section> released(sections, sections + count);
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    std::shared_ptr<at::mps::MPSEvent> ended = at::mps::getMPSEventPool()->acquireEvent(false, stream);
    ended->record(true);
    notifying.fetch_add(1, std::memory_order_relaxed);
    if (!ended->notify(true, ^(id<MTLSharedEvent>, uint64_t) {
          for (const mesh_section &section : released) publish(context, section, execution);
          notifying.fetch_sub(1, std::memory_order_release);
          (void)ended;
        })) {
      notifying.fetch_sub(1, std::memory_order_relaxed);
      return EIO;
    }
    stream->synchronize(at::mps::SyncType::COMMIT);
    return 0;
  } catch (...) {
    return EIO;
  }
}

extern "C" uint64_t mesh_torch_notifying(void) { return notifying.load(std::memory_order_acquire); }
