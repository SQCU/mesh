/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* libnccl-mesh: NCCL's API over the mesh's Thunderbolt 5 RDMA bridges.

   This is NVIDIA/nccl master 12df1a1 src/nccl.h.in (NCCL 2.32.3) with its CUDA includes removed, its version filled in,
   and only the functions libnccl-mesh.dylib exports declared (each also as its pnccl profiling name).  Every type,
   enum, config struct and initializer is NCCL's, member for member, so code written against nccl.h compiles against
   this one.  Installing, linking and running it: rdma/NCCL.md.  The mesh's own calls (ncclMesh*) are last.

   A rank is a process on a node, a client of its node's resident bridge (mesh-flow): the rank asks, the bridge does.
   It passes no region, map file, silent bound or queue-pair count, and keeps no step executor of its own: every
   crossing a program makes between nodes is one of these calls, and this library (nccl-mesh.c's walk of a program's
   steps, nccl-mesh-kernels.h's kernels) is the only code that publishes a piece, lands or combines it, and waits on it.

   Communicators.  ncclCommInitRank is NCCL's bootstrap: the unique id is the root's host:port (NCCL_COMM_ID, else the
   listener ncclGetUniqueId opens, on the node's fabric address where it holds one; the bindings take
   MASTER_ADDR:MASTER_PORT and RANK of WORLD_SIZE, as torchrun sets them), every rank's node (mesh_observe) meets there,
   the root listening on that host alone where it is an IPv6 address; the id is its name.  ncclCommInitRankConfig with
   config->blocking 0 is NCCL's non-blocking init (ncclMeshCommInitRank, below).  Its first gather carries each rank's
   node and every live physical endpoint, including availability (mesh-call.h mesh_endpoints);
   each rank then holds, on every physical cable joining it to a member, the stream of the least identity free at both
   ends, as it holds rows (mesh_stream_take: design/transport.md section 14), one communicator a stream (a holding's
   lists and counts are its holder's alone; a link has several, so communicators sharing a node pair hold their own;
   EBUSY where a physical cable has none free at both ends); the second gather carries each rank's holding's
   status.  Each phase of the bootstrap (the connect or accepts, each gather) and each configuration exchange has a
   deadline, MESH_PREPARATION_NS from its start (it covers the ranks' preparation skew), past which the program ends,
   its GPU work stopped first, naming the phase and the rank awaited.  The data program names nodes and its own pages; the compiled fabric retains every physical cable.
   The bridge owns ports, lanes, frames, queue pairs and SEND sizing.  An init fails as a value where two ranks are on one node (its
   holdings are kept a node), a stream is held by another process,
   or its bridge is still ending its last holding, or gone; what it held is then let go and the next init reads the streams as they stand.  A
   connection to the root whose report fails or names no rank still to meet is closed alone and the root meets the
   next (one that sends nothing holds the root's read until the phase ends).  A communicator's map is its members joined where their bridges reach each other's nodes, made alike by
   every member from the same reciprocal reports and kept for its life: its nodes are its members,
   every physical cable is a separate channel, including all parallel cables (ncclMeshCommFabric),
   and a route crosses members alone, so a node a route is to cross takes part as a member.  The communicator's own
   table makes every member a contributor, so a member on a route combines its own operand into what it sends on; only
   a program passed to ncclMeshCommInitRank with mesh_compile's contributors, or a prepared WRITE path,
   has a member that sends on what lands alone (the identity, data-model.md section 9's relay).  Two members no path between members
   joins fail a collective, and a call between two members not joined fails, as values (ncclInvalidUsage).  ncclMeshCommInitRank (below) is the same init for a program
   that passes its own programs or paths.  ncclCommSplit and ncclCommShrink make communicators of the ranks they take;
   Split holds its children's streams over comm (NCCL's bootstrapSplit), and Shrink makes no exchange, so a shrink to
   two ranks or more holds none: it is made with no channel, and its first call that crosses a link fails as a value
   (a rank among those excluded gets NULL, as NCCL_SPLIT_NOCOLOR's).  Inside a group, Split and Grow run their own
   collectives at once, the group's calls set aside, and Shrink makes none.  ncclCommGrow meets at the bootstrap of the
   id that ncclCommGetUniqueId gives comm's root, which the root broadcasts on comm.  ncclCommRevoke is NCCL's: comm's
   operations in flight are revoked and its resources kept (its streams stay held, and its bridge sends what its
   groups published and nothing else), an init still
   running (a non-blocking one's) ends as a value, its host-path groups return at their next look and its GPU waits end
   at their landings or their bounds (no kernel stops at the revoke), every later group on comm is refused (ncclInvalidUsage), ncclCommFinalize after
   it returns ncclSuccess (it flushes nothing), and ncclCommGetAsyncError reads ncclSuccess once it is done; its
   revokeFlags is not read; from any thread, under comm's own
   lock, never one a group holds while it waits.  ncclCommDestroy waits for the GPU work the library committed for comm
   until its command buffer's status is Completed or Error (none not committed), within MESH_PLATFORM_NS, past which the
   program ends (design/eager-execution.md, assumption A4), then gives its streams back
   and frees it (a holding given with units outstanding takes its
   stream out of its link's pairing: design/transport.md section 14); a lease a group of a revoked or failed comm named
   is the caller's to keep until then, since its bridge may still post receives into it for entries presented; work in a caller's encoder is the caller's to
   finish first (a kernel of it still queued would write a stream given back: rdma/NCCL.md "Failures").  Making a communicator (a lone rank's too) attaches the process's client of its node's bridge where it has
   none, so it needs the node's bridge running and fails as ncclMemAlloc does (Memory, below): every client slot held,
   no region of a bridge that runs, or a lease of a bridge that stopped still held.

   Programs.  A communicator's programs are a table of compiled collectives, operands made ahead of time (mesh-plan.h
   struct mesh_program): the caller's own (ncclMeshCommInitRank), or else the structural one, compiled on its map at its
   first call that needs it, along the caller's paths (ncclMeshCommInitRank) or else the map's shortest-path trees (each
   member's in-tree of the routes between its members): the single-phase all-reduce (each member's sum up its own tree; the
   direct exchange where every two members are linked), the reduce-scatter and the all-gather (segment s up and down
   s's trees) and each root's reduce and broadcast, every one uncut (a step's piece one entry, any size, which the
   bridge alone cuts into SENDs).  The communicator's configuration names other forms of a collective ahead of these
   (ncclMeshCommForms, below: the packed trees, the two-phase all-reduce, chunked pipelining, each below a size bound),
   compiled first in the same table.  A call runs its table's first program of its collective (and root) whose size
   bound its operand is under; nothing about it is decided where it runs, no price chooses it, and
   ncclCollConfig_t.algSelection is not read.  Every member compiles the same programs from the same map and the same
   operands (every rank passes the same paths, programs and forms), so nothing is compared at run time.  A group
   (ncclGroupStart/End; a call outside one is a group of one) walks each call's program (mesh_program_steps_cut: the
   program's typed cuts and declared WRITE endpoints), lowered to explicit physical cable/range
   records on its communicator's channels: one holding per cable, each transfer one entry of its send
   list or landing list of immutable exact byte spans. Each direction has one
   matching DMA-unit sequence at both ends, in compiled dependency order,
   presented to the bridge as the group is issued (mesh-call.h
   mesh_stream_present).  A member that fails, is revoked or ends mid-group sends nothing more: its peers' host-path
   waits end at a revoke, the link's loss or their deadlines, and their GPU waits at their bounds (a deadline past
   which the program ends: design/eager-execution.md, assumption A4), and its holding's end with units outstanding takes its stream out of its link's pairing
   (design/transport.md section 14).  A publication
   raises the holding's `ready` past the entry; a landing is awaited on its `landed`.
   The data program owns source/relay lifetimes; there are no `sent` waits (design/pages-and-functions.md, Freeing 6-7).  Nothing credits a publication: the hardware holds a SEND until its receive is posted.  No
   payload is checked, so calls that do not pair (unequal counts, different collectives or orders) are the caller's
   error and are not detected.  Uneven segments are their own calls
   (ncclMeshAllGatherV, ncclMeshReduceScatterV, below), whose count vector every rank passes alike.

   Host path: a NULL stream.  Buffers are host pointers into this node's region (a lease's: Memory, below), sent from
   and landed in in place; a group runs as the Metal path's group in a command buffer of its communicator's own queue,
   its buffers as their leases' Metal buffers, committed and awaited before its call returns (a float64 reduction fails
   as it does there); every call returns once its publications
   and required landings have run; outgoing buffer lifetimes remain the caller's.  Metal path: a stream that
   is an ncclMeshStream.  The group is encoded into its commandBuffer (the caller commits it), every buffer argument an
   ncclMeshBuffer (an id<MTLBuffer> and a byte offset into it: an MPS tensor's storage and offset) of any type, a
   crossing's over a lease (Memory, below); float64 moves as its bytes, and a float64 reduction there fails as a value
   (Apple GPUs have no float64 arithmetic).  With commandEncoder, the kernels go into that open compute encoder, left
   open with nothing signalled at the group's end; workspace is not read (a call that crosses no link makes its result
   in recvbuff, and what a crossing sends from or lands in that is not the caller's is the communicator's: Memory),
   and the caller orders any later host-path group after the stream's work.  The calls of one group name one
   command buffer or encoder (the group's dispatches are one program); their bound is the greatest of their polls.  A
   group on several communicators is one group of each communicator's calls, in the order the group first names them,
   progressing together: on the Metal path each is issued and, where the stream asked no issue, all are completed
   through the greatest ticket on the last one's stream, their channels interleaved as tracks; on the host path each
   runs on a thread of its own; every one is launched whatever another's result, and the first failure is returned.  With issue, the group is issued and not completed (NCCL's work on its own stream, waited for later):
   its operands made in place and every publication that needs no wait encoded, with no wait for a peer; issue->ticket names it (0: it completed at once),
   issue->published says whether anything was published (the caller then commits, so the peers' waits progress), and the
   issue pointer stays valid until ncclGroupEnd returns.  ncclMeshComplete completes it, with every group issued before
   it, on a later stream of the same queue; its operands are not touched until then.  A group without issue completes at
   once, with every group before it; a host-path group fails while its own communicator's groups are issued and not
   complete, and waits only for its own communicator's GPU work, until its command buffer completes (an error status,
   which the platform's watchdog gives a command buffer that overruns, fails the group, and one not yet committed fails
   it at once, ncclInvalidUsage), a revoke or a holding's end.  A group waits for room in its bridge's lists until the
   bridge takes its entries, a revoke or its holding's end.  A host-path group spins on
   its landings, never parks, and holds no lock another communicator's groups take; its waits end at their counts, a
   count's ~0, its holding's end or a revoke.  Each of these waits has a deadline, MESH_PLATFORM_NS from its start,
   past which the program ends (design/eager-execution.md, assumption A4).  A group whose entries lie on one channel completes as
   plain dispatches in the stream's order; a group on two or more, with each channel progressing on its own
   (rdma/NCCL.md "One track", "Per-channel progress").  A group's landings in a stream wait to be posted until the stream
   reaches the group (its first publication on the channel, or a gate store of its first dispatch), so no receive lands
   in pages the stream's earlier work still reads. Its end does not await outgoing SEND completion.  A SEND reads the
   caller's bytes as given (a send buffer read in place, an operand given in place, an all-to-all's segments) as the
   NIC sees them, and the library stores none of them again: their producer stores them system-coherent, or its
   command buffer has ended before the group publishes, since a plain store reaches the NIC only once its command
   buffer ends (metal-microbench docs/facts.md, 10-24); a WRITE's or a fused crossing's producer stores so itself
   (mesh_step_write).
   A wait is a kernel polling its holding's required `landed` count, and nothing else (rdma/mesh.metal mesh_wait_bounded), which the bridge sets ~0 where the link is
   lost; every wait is bounded: the stream's `polls` is the most loads of a count a wait makes,
   the caller's bound for the group's crossings, which it derives from its plan (the lag it expects at them, over the
   period of one load on the node's GPU: rdma/NCCL.md "Bounded waits"); 0, or more than NCCL_MESH_POLLS, is
   NCCL_MESH_POLLS (design/transport.md); the waits a group encodes at its call take its calls' greatest, and those
   ncclMeshComplete encodes its stream's.  A wait that reaches its bound or reads a lost link records its
   cause, entry and channel in its communicator's failure words, on which no kernel halts (every later dispatch runs as
   encoded, a later wait on those words ending at once, one under way within 2^10 loads: nccl-mesh-kernels.h
   mesh_step_failed), read by the host after its command buffer (ncclCommGetAsyncError, or the communicator's next
   group): the bound is the wait's deadline, and a wait past it ends the program there (design/data-model.md section
   6); a lost link, or a bound reached after a revoke, fails the group as a value.
   The Metal 4 form: a stream whose argumentTable (an id<MTL4ArgumentTable>) is set, commandEncoder then an
   id<MTL4ComputeCommandEncoder> and residencySet an id<MTLResidencySet> (commandBuffer not read), its polls as the
   classic form's.  The group is encoded as compute dispatches between the encoder's commands before and after it, bound
   through the caller's argument table (at least 6 buffer bindings; no setBytes, each dispatch's arguments a record of
   the library's ring), on pipelines of the residency set's device, every buffer of the library's it touches added to
   the caller's residency set (committed where one was added); nothing is committed or signalled.  The classic form
   reads residencySet only for a fused crossing (below), whose steps the caller's kernels run by their bindings' GPU
   addresses:
   an id<MTLResidencySet> of the command buffer's device, to which the library adds every buffer those bindings name
   (committed where one was added), and which the caller makes resident for the command buffers its kernels are encoded
   into (its queue's addResidencySet:, or each command buffer's useResidencySet:), since a classic encoder makes resident
   only the buffers bound to it.

   Memory.  The process is one client of its node's bridge: one attach of the node's region, made at its first
   ncclMemAlloc or communicator, whose slot owns the process's leases and holds the streams its communicators use; a
   communicator leases a page for its gate and one for its failure words, and, for each group definition (a declared
   group, or a prepared WRITE's), two slots of what its groups send from and land in that are not the caller's (a
   reduce-scatter's partials of the others' segments, a non-receiving rank's operand, a tree's scratch, a relay's
   generations, a combine's landing area): a declared group's made at its declaration (ncclMeshCommPlan, which
   fails as a value where the region cannot hold them), a WRITE's at its first executions, its groups taking turns
   at them.  A group that no declaration covers ends the program with a warning: nothing compiles one at its issue
   (design/eager-execution.md),
   ncclMemAlloc called again only where one must grow (to one of eight sizes an octave), a dropped definition's slots
   kept for the next, one too small returned once its last group is done (the host path's, or a Metal path's that the
   stream has run past: it stores that group's ticket where its generation let the slot go), all returned at
   ncclCommDestroy.  What the GPU reads of a group once it is encoded (its entries for the bridge's lists, its tracks'
   records, a Metal 4 dispatch's argument record) is cut from the process's pieces of the region, each a lease with its
   Metal buffer made once, taken again once the GPU has done with all cut from it (the entries presented, the tracks
   ended, the residency set that holds argument records ended): no Metal buffer is made at a call.  A group's host
   bookkeeping (its schedules, its steps' state, its record in flight) and its entries' span tables (a lease of them)
   are its definition's too, a generation's laid and placed at its first group for that group's buffers and taken
   again by each later group of the generation with the same buffers, which binds only its entries on the channels; a
   group with other buffers lays and places its generation again (its old tables let go once the bridge's reads of
   them end).  A group on several communicators keeps its parts in its thread's lists.  A result is made in recvbuff (a reduce-scatter's own segment too: no compiled piece spans two
   segments), and nothing is copied into a slot from a caller's buffer but the operand of a premultiplied,
   partials-merging or fused call where it is not the result, and an in-place single-phase all-reduce's original.  A
   group writes a slot only once the slot's last group has every landing awaited: the stream's order, and a landed
   wait for those its stream did not await.  A SEND's source is reused as design/pages-and-functions.md (Freeing) states:
   after a natural downstream landing or the bridge's own DMA-read completion; nothing is sent back.  ncclMemAlloc is the lease,
   the call the operator's division of the work makes of the bridge ("the bridge is there to behave like a system
   call"): whole pages of the region covering size, which the bridge registered on every device when the device came up,
   before any queue pair existed; a lease registers nothing.  The client sizes its leases ("ontologically it must be the
   responsibility of client programs to know how much allocation they need"): a lease the region cannot meet is
   ncclSystemError naming the bytes asked and the pages free, one while every client slot is held ncclSystemError naming
   mesh.h MESH_CLIENTS, and one where the process attaches no region of a bridge that runs names the requirement, a
   bridge whose configuration's region holds the client's footprint (etc/bridge/README.md; ncclRemoteError where the
   bridge stopped, else ncclSystemError); nothing falls back to other memory.  ncclMemFree logically releases the lease
   without waiting. Its pages remain retained for all registered source DMA uses, including future SENDs whose
   descriptors the bridge has not posted yet. Once a holding can post no further reads, outstanding posted reads may
   instead retain the pages through bridge-owned pins; this handoff never permits the caller to overwrite them.
   The caller must finish its GPU and destination uses before logical free and may schedule no new use afterward.
   The leases of a process gone are returned by its bridge, as every retired client's rows are.
   ncclCommRegister takes a buffer in
   the region, its handle the buffer itself, and refuses any other (ncclInvalidArgument: lease it with ncclMemAlloc);
   ncclCommDeregister releases nothing.  A lease is an operand of the region it came from, which outlives a bridge's
   restart (its next bridge adopts it: the lease stands, nothing is reloaded).  A region a stop released, or a bridge
   of another layout replaced, is expired: while the process holds a lease or communicator of it, its next ncclMemAlloc
   and communicator fail with ncclRemoteError naming the expiry (and a communicator's next group that crosses a link
   fails), until the process has freed every one (ncclMemFree, ncclCommDestroy) or ended; the library then lets the
   expired region go and leases from the bridge's next one, so a process never holds two regions (design/transport.md
   section 2).  A process that held a communicator, or a lease's Metal buffer (ncclMeshMemBuffer), ends to leave it:
   the library made Metal buffers over the region for them, which its GPU work and residency set may hold after they
   are released, so the library never lets that region go.  A caller that makes its own Metal buffers over a lease
   releases them, and their residency, before ncclMemFree, or ends.  Weights, KV and
   every buffer a peer reads are leases: "why would weights or kv be outside of the memory pinned pool? being outside of
   it means that they cannot be zero copy passed to peers."  A lease's Metal buffer is ncclMeshMemBuffer's (below), the
   buffer an MPS tensor or a kernel takes over the lease.  The operator's 2026-10-21 requirement was "zero copy
   methods must always be used over access by copy because memory bandwidth and flops are too scarce on this hardware
   api". The 2026-10-24 amendment explicitly permits compiled masked streaming data functions: pack sub-page or strided
   values into separately owned padded ingress frames, then consume their valid intervals at precompiled destinations.
   Invalid source loads and destination stores are masked; padding never overwrites neighboring final-output values.
   This is explicit arithmetic in the data graph, not a native transport repair or a receiver's address-selection
   protocol. Its copies, padding and dispatch costs count in whole-function latency and useful bandwidth.
   The native transport writes exact byte spans from their declared sources to their declared destinations.
   A COPY lands in its destination; a reduction lands in a separate area before its data function combines it.
   Allocation pages, DMA units, and hardware credit frames are distinct. Each request has one SGE and must stay
   inside its registered-memory extent. Prepared WRITE bindings jointly compile packet, registration, indexed-view
   and affine-row cuts during configuration; execution transmits only their declared payload, without a metadata
   exchange or a staging repair. Every crossing source and destination remains in this node's region.
   Generic collective bindings must preserve those same physical constraints. A lease that fits one registration
   extent lies within one (mesh-dataflow.c mesh_arena_alloc), so no span of one, a slot of the library's or a caller's
   buffer, crosses an extent. Over a lease larger than an extent, their relative packet grid does not yet compile
   registration cuts for every ragged placement: place_group reports an invalid span before presentation if one
   crosses. That integration gap is not a page-alignment requirement on the data model.
   A collective whose
   send buffer lies apart from its result reads it in place where its program allows (a SEND of a range no step of the
   rank wrote reads it, the first combine of such a range writes send op piece, the ranges of the result no step writes
   are copied from it after the publications, and no scratch buffer no step writes is made); else its operand is made in
   place in the region (a premultiply, or a copy within the region). The NIC reads a SEND's pages until the SEND is
   done, and a group completes without waiting for that. Published source bytes are immutable until every registered
   physical DMA reader has finished, including readers on parallel cables, relay paths and not-yet-posted entries.
   Mutating them earlier is invalid. The compiled page assignment includes local GPU reads,
   incoming writes and destination consumers before it assigns an overwrite. A natural landing
   is an alternative witness only if its actual causal ancestry proves every relevant reader finished; merely being
   later or coming from each peer is insufficient. Logical free defers physical reuse under these source obligations
   without adding a receipt, acknowledgment or data-program wait.
   ncclCommRegister refuses a buffer outside the region.  Design: design/transport.md.

   Failures are ncclResult_t values, ncclGetLastError naming what was lost; nothing resumes a failed group, and a
   communicator whose group failed fails its later groups, made again on the map current then (ncclCommDestroy, then an
   init).  No timer of the library's declares a peer lost: a landing that reads ~0 (the bridge's, at its link's loss or
   its stop), a GPU wait's bound, its holding's end or a revoke ends the group, and a group on a lost link or whose
   peer's entry never landed fails with ncclRemoteError; a failure the Metal path finds after its group returned is
   the communicator's asynchronous error (ncclCommGetAsyncError) and the next group's result.  A
   Metal-path group that returns a failure signals nothing the library waits for: the commands it left in the stream's
   command buffer are not committed after the communicator's ncclCommDestroy or ncclCommAbort, which wait for none of
   them.

   Not implemented (not exported): ncclCommInitRankScalable,
   ncclCommSuspend, ncclCommResume, ncclCommMemStats, ncclCommWindowRegister, ncclCommWindowDeregister,
   ncclWinGetUserPtr, ncclPutSignal, ncclSignal, ncclWaitSignal (TB5 RDMA has no one-sided write), ncclGroupSimulateEnd,
   ncclSetEncryption, the ncclParam* functions.  ncclCommInitAll takes one device and ncclCommCuDevice returns 0 (a node
   has one Metal device); ncclRedOpCreatePreMulSum's scalar is read when the operator is made
   (ncclScalarHostImmediate), else where it lies at each host-path call that reduces by it (ncclScalarDevice: a
   Metal-path call by it fails as a value, its kernels taking the scalar as an argument written when the group is
   encoded); an operator reduces the datatype it was made for alone (NCCL's hostToDevRedOp alike); a communicator
   makes as many operators as it holds handles for. */

#ifndef NCCL_H_
#define NCCL_H_

#define NCCL_MAJOR 2
#define NCCL_MINOR 32
#define NCCL_PATCH 3
#define NCCL_SUFFIX ""

#define NCCL_VERSION_CODE 23203
#define NCCL_VERSION(X,Y,Z) (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>
#include <stdint.h>
#include <stddef.h>
struct mesh_program;
struct mesh_trees;

/* design/transport.md: the most loads any GPU wait makes, and the bound of a stream whose polls is 0 or
   above it: MESH_PLATFORM_NS over the slowest load period read on a node's GPU (the M4 Pro's 1.1 us while its clock
   rises, rdma/NCCL.md "Bounded waits"); each wait's own, and once one wait has recorded a failure every other wait on
   the same failure words ends at once or within 2^10 loads (nccl-mesh-kernels.h mesh_step_failed), so waits that never
   land hold the GPU about one bound in all */
#include "metal-capabilities.h"
#include "mesh-fabric.h"
#define NCCL_MESH_POLLS MESH_POLL_LIMIT
/* The stream: NULL (the host path) or an ncclMeshStream (the Metal path, above). */
typedef void *cudaStream_t;
typedef struct { void *buffer; size_t offset; } ncclMeshBuffer;
typedef struct { uint64_t ticket; int published; } ncclMeshIssue;
/* Fused publication and landing bindings, in physical channel order. Each
   record contains six GPU addresses: d, s, argument record, a, v, progress
   (ncclMeshMetalPrelude's mesh_step_binding). The caller supplies room using
   its compiled group's counts; the result gives actual active-channel counts.
   Parallel channels remain separate and may contain multiple physical fragments.
   Zero counts mean no corresponding local publication/consumption, not permission
   to silently execute an unfused source read before its producer.

   Producers release all ordinary source publications. Relay fragments are
   supplied by their incoming landed prefixes. Consumers await the final own
   landing on each relevant channel, excluding any relay-only trailing entries.
   Source and relay bytes remain immutable through all declared reads; receiving
   ranges have a first-write ownership proof and no reciprocal producer gate.

   repeats (0 as 1) declares consecutive executions of the same compiled form.
   stride is their local byte storage displacement. Only the first frame's
   binding records are returned; their prefix increments describe subsequent
   frames, and kernels use mesh_step_release/mesh_step_await with that frame's
   index. A zero stride does not establish permission to overwrite live pages. */
typedef uint64_t ncclMeshBinding[6];
typedef struct { ncclMeshBinding *publish, *land; uint32_t publishes, lands, repeats; size_t stride; } ncclMeshFused;
typedef struct {
  void *commandBuffer;
  void *commandEncoder;
  void *(*workspace)(size_t bytes, void *context);
  void *context;
  ncclMeshIssue *issue;
  void *argumentTable;
  void *residencySet;
  uint32_t polls;
  ncclMeshFused *fused;
} ncclMeshStream;
/* ncclCollConfig_t.launchCompletionEvent: not used; must be NULL. */
typedef void *cudaEvent_t;

/* Opaque handle to communicator */
typedef struct ncclComm* ncclComm_t;
#define NCCL_COMM_NULL NULL

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

/* Error type */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclRemoteError             =  6,
               ncclInProgress              =  7,
               ncclTimeout                 =  8,
               ncclNumResults              =  9 } ncclResult_t;

#define NCCL_CONFIG_UNDEF_INT INT_MIN
#define NCCL_CONFIG_UNDEF_PTR NULL
#define NCCL_SPLIT_NOCOLOR -1
#define NCCL_UNDEF_FLOAT -1.0f

/* Internal use only */
#define NCCL_API_MAGIC 0xcafebeef

/* NCCL performance policy */
#define NCCL_CTA_POLICY_DEFAULT 0x00
#define NCCL_CTA_POLICY_EFFICIENCY 0x01
#define NCCL_CTA_POLICY_ZERO 0x02

/* ncclCommShrink flags*/
#define NCCL_SHRINK_DEFAULT 0x00 /* shrink the parent communicator */
#define NCCL_SHRINK_ABORT 0x01   /* First, terminate ongoing parent operations, and then shrink the parent communicator */

/* ncclCommRevoke flags */
#define NCCL_REVOKE_DEFAULT 0x00 /* reserved for future use; must be 0 */

typedef enum {
  ncclHostCftDefault = NCCL_CONFIG_UNDEF_INT,
  ncclHostCftEnable = 1,
  ncclHostCftDisable = 2,
  ncclHostCftFallback = 3
} ncclHostCftMode_t;

/* Host NVLS component-disable flags. Individual component flags may be combined. */
typedef enum {
  ncclNvlsHostModeDefault = NCCL_CONFIG_UNDEF_INT, /* library default */
  ncclNvlsHostModeEnable = 0,                      /* enable all host NVLS components */
  ncclNvlsHostModeDisableTransport = 1 << 0,       /* disable NVLS transport and registered-buffer optimization */
  ncclNvlsHostModeDisableSymmetricMultimem = 1 << 1, /* disable multimem in internal symmetric kernels and CE */
  ncclNvlsHostModeDisable = INT_MAX,               /* disable all present and future host NVLS components */
} ncclNvlsHostMode_t;

/* Communicator configuration. Users can assign value to attributes to specify the
 * behavior of a communicator. */
typedef struct ncclConfig_v23100 {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;
  /* attributes that users are able to customize. */
  int blocking;
  int cgaClusterSize;
  int minCTAs;
  int maxCTAs;
  const char *netName;
  int splitShare;
  int trafficClass;
  const char *commName;
  int collnetEnable;
  int CTAPolicy;
  int shrinkShare;
  int nvlsCTAs;
  int nChannelsPerNetPeer;
  int nvlinkCentricSched;
  int graphUsageMode;
  int numRmaCtx;
  int maxP2pPeers;
  int graphStreamOrdering;
  int launchOrderImplicit;
  int numRmaSig;
  int rmaEagerInit;
  int hostCftMode;
  int nvlsHostMode;
} ncclConfig_t;

/* Config initializer must be assigned to initialize config structure when it is created.
 * Not initialized config will result in NCCL error. */
#define NCCL_CONFIG_INITIALIZER {                                       \
  sizeof(ncclConfig_t),                     /* size */                  \
  NCCL_API_MAGIC,                           /* magic */                 \
  NCCL_VERSION_CODE,                        /* version */               \
  NCCL_CONFIG_UNDEF_INT,                    /* blocking */              \
  NCCL_CONFIG_UNDEF_INT,                    /* cgaClusterSize */        \
  NCCL_CONFIG_UNDEF_INT,                    /* minCTAs */               \
  NCCL_CONFIG_UNDEF_INT,                    /* maxCTAs */               \
  NCCL_CONFIG_UNDEF_PTR,                    /* netName */               \
  NCCL_CONFIG_UNDEF_INT,                    /* splitShare */            \
  NCCL_CONFIG_UNDEF_INT,                    /* trafficClass */          \
  NCCL_CONFIG_UNDEF_PTR,                    /* commName */              \
  NCCL_CONFIG_UNDEF_INT,                    /* collnetEnable */         \
  NCCL_CONFIG_UNDEF_INT,                    /* CTAPolicy */             \
  NCCL_CONFIG_UNDEF_INT,                    /* shrinkShare */           \
  NCCL_CONFIG_UNDEF_INT,                    /* nvlsCTAs */              \
  NCCL_CONFIG_UNDEF_INT,                    /* nChannelsPerNetPeer */   \
  NCCL_CONFIG_UNDEF_INT,                    /* nvlinkCentricSched */    \
  NCCL_CONFIG_UNDEF_INT,                    /* graphUsageMode */        \
  NCCL_CONFIG_UNDEF_INT,                    /* numRmaCtx */             \
  NCCL_CONFIG_UNDEF_INT,                    /* maxP2pPeers */           \
  NCCL_CONFIG_UNDEF_INT,                    /* graphStreamOrdering */   \
  NCCL_CONFIG_UNDEF_INT,                    /* launchOrderImplicit */   \
  NCCL_CONFIG_UNDEF_INT,                    /* numRmaSig */             \
  NCCL_CONFIG_UNDEF_INT,                    /* rmaEagerInit */          \
  NCCL_CONFIG_UNDEF_INT,                    /* hostCftMode */           \
  NCCL_CONFIG_UNDEF_INT,                    /* nvlsHostMode */          \
}

/* A single vendor-specific option expressed as a key-value pair. Multiple options can be
 * chained into a non-circular linked list through the next pointer and attached to a ncclCollConfig_t
 * via its ext member. The official NCCL library ignores every extension; a vendor library
 * should assign a unique value to key.vendorId to avoid collision among vendor libraries.
 * It can usually be done by picking a random number.
 * Vendors should choose a non-zero vendorId (less than 2^24) that is unlikely to collide
 * with other vendor libraries. The list is unordered and must not contain duplicate keys. */
typedef struct ncclConfigExt {
  struct ncclConfigExt* next;       /* next option in the list, or NULL to terminate */
  struct {
    int vendorId;                   /* vendor choose their own unique value as vendorId */
    int optionId;                   /* vendor-defined id distinguishing options within a vendor */
  } key;
  union {
    int i;                          /* integer value */
    const char* s;                  /* C-style string value */
    void* raw;                      /* opaque pointer-sized value */
  } val;
} ncclConfigExt_t;

/* Per-collective configuration. Users assign values to attributes to customize an
 * individual collective call issued through one of the nccl*Config() entry points.
 * Like ncclConfig_t, this is a versioned, append-only struct: in the future, any new
 * members will be added at the end, and existing members will never be removed or
 * reordered. The user owns the storage and must keep the struct (and any linked ncclConfigExt_t)
 * valid for the duration of every call that uses it. The same configuration must be
 * set on every rank; NCCL validates it only locally. */
typedef struct ncclCollConfig_v23200 {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;
  /* non-circular linked list of vendor-specific extension options, or NULL. */
  ncclConfigExt_t* ext;
  /* resource tuning (NCCL_CONFIG_UNDEF_INT == unset; NCCL picks). */
  int minCTAs;          /* lower bound on channels/CTAs */
  int maxCTAs;          /* upper bound on channels/CTAs, must be <= comm->maxCTAs */
  int nvlsCTAs;         /* NVLS-pool-specific channel cap */
  int cgaClusterSize;   /* CUDA thread-block-cluster size (Hopper+), should be the same for all
                           collectives in the same Group. Inconsistent cgaClusterSize in a Group
                           is undefined behavior. */
  /* algorithm selection: filter which algorithm(s) this call may use. */
  const char* algSelection;   /* selection string; NULL/UNDEF/"" == automatic */
  int forceAlgSelection;      /* default 1 (true): unsatisfiable selection is an error */
  int CTAPolicy;              /* NCCL_CTA_POLICY_* bitmask; UNDEF == inherit comm-level CTAPolicy */
  /* user-provided profiler tag: an opaque value delivered verbatim to profiler plugins with this
   * call's profiler events. Default to 0 as untagged. It does not affect execution. NCCL reserves
   * profiler-tag values with the most-significant bit set (>= 0x8000000000000000) for possible future
   * predefined tags; applications should use values with the MSB clear. */
  uint64_t userProfilerTag;
  /* Caller-owned, rank-local CUDA event recorded at collective kernel launch completion.
   * Create it with cudaEventDisableTiming; interprocess and interop events are unsupported.
   * Every rank must provide a non-NULL event or every rank must provide NULL. Keep the event
   * alive through all queued waits and captured-graph executions. At most one non-NULL event
   * per communicator is supported in a group. CUDA < 12.3 records it before launch. */
  cudaEvent_t launchCompletionEvent;
} ncclCollConfig_t;

/* NCCL_COLLCONFIG_INITIALIZER must be used to initialize a ncclCollConfig_t when it is
 * created. An uninitialized config will result in a NCCL error. */
#define NCCL_COLLCONFIG_INITIALIZER {                                 \
  sizeof(ncclCollConfig_t),                 /* size */                \
  NCCL_API_MAGIC,                           /* magic */               \
  NCCL_VERSION_CODE,                        /* version */             \
  NCCL_CONFIG_UNDEF_PTR,                    /* ext */                 \
  NCCL_CONFIG_UNDEF_INT,                    /* minCTAs */             \
  NCCL_CONFIG_UNDEF_INT,                    /* maxCTAs */             \
  NCCL_CONFIG_UNDEF_INT,                    /* nvlsCTAs */            \
  NCCL_CONFIG_UNDEF_INT,                    /* cgaClusterSize */      \
  NCCL_CONFIG_UNDEF_PTR,                    /* algSelection */        \
  1,                                        /* forceAlgSelection */   \
  NCCL_CONFIG_UNDEF_INT,                    /* CTAPolicy */           \
  0,                                        /* userProfilerTag */     \
  NCCL_CONFIG_UNDEF_PTR,                    /* launchCompletionEvent */ \
}

/* NCCL malloc and free function for all types of NCCL optimizations
 * (e.g. user buffer registration). The actual allocated size might
 * be larger than requested due to granularity requirement. */
ncclResult_t  ncclMemAlloc(void** ptr, size_t size);
ncclResult_t pncclMemAlloc(void** ptr, size_t size);

ncclResult_t  ncclMemFree(void *ptr);
ncclResult_t pncclMemFree(void *ptr);

/* Return the NCCL_VERSION_CODE of the NCCL library in the supplied integer.
 * This integer is coded with the MAJOR, MINOR and PATCH level of the
 * NCCL library
 */
ncclResult_t  ncclGetVersion(int *version);
ncclResult_t pncclGetVersion(int *version);

/* Generates an Id to be used in ncclCommInitRank. ncclGetUniqueId should be
 * called once and the Id should be distributed to all ranks in the
 * communicator before calling ncclCommInitRank. */
ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t pncclGetUniqueId(ncclUniqueId* uniqueId);

/* Create a new communicator (multi thread/process version) with a configuration
 * set by users. */
ncclResult_t  ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);
ncclResult_t pncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);

/* Creates a new communicator (multi thread/process version).
 * rank must be between 0 and nranks-1 and unique within a communicator clique.
 * Each rank is associated to a CUDA device, which has to be set before calling
 * ncclCommInitRank.
 * ncclCommInitRank implicitly syncronizes with other ranks, so it must be
 * called by different threads/processes or use ncclGroupStart/ncclGroupEnd. */
ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t pncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);

/* Creates a clique of communicators (single process version).
 * This is a convenience function to create a single-process communicator clique.
 * Returns an array of ndev newly initialized communicators in comm.
 * comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
 * If devlist is NULL, the first ndev CUDA devices are used.
 * Order of devlist defines user-order of processors within the communicator. */
ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
ncclResult_t pncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);

/* Finalize a communicator. ncclCommFinalize flushes all issued communications,
 * and marks communicator state as ncclInProgress. The state will change to ncclSuccess
 * when the communicator is globally quiescent and related resources are freed; then,
 * calling ncclCommDestroy can locally free the rest of the resources (e.g. communicator
 * itself) without blocking. */
ncclResult_t  ncclCommFinalize(ncclComm_t comm);
ncclResult_t pncclCommFinalize(ncclComm_t comm);

/* Frees local resources associated with communicator object. */
ncclResult_t  ncclCommDestroy(ncclComm_t comm);
ncclResult_t pncclCommDestroy(ncclComm_t comm);

/* Frees resources associated with communicator object and aborts any operations
 * that might still be running on the device. */
ncclResult_t  ncclCommAbort(ncclComm_t comm);
ncclResult_t pncclCommAbort(ncclComm_t comm);

/* Creates one or more communicators from an existing one.
 * Ranks with the same color will end up in the same communicator.
 * Within the new communicator, key will be used to order ranks.
 * NCCL_SPLIT_NOCOLOR as color will indicate the rank will not be part of any group
 * and will therefore return a NULL communicator.
 * If config is NULL, the new communicator will inherit the original communicator's
 * configuration*/
ncclResult_t  ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config);
ncclResult_t pncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config);

/* Shrink existing communicator.
 * Ranks in excludeRanksList will be removed form the existing communicator.
 * Within the new communicator, ranks will be re-ordered to fill the gap of removed ones.
 * If config is NULL, the new communicator will inherit the original communicator's configuration
 * The flag enables NCCL to adapt to various states of the parent communicator, see NCCL_SHRINK flags.*/
ncclResult_t  ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
ncclResult_t pncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);

/* Revoke: comm's operations in flight end with an error and none is issued after it (NCCL 2.29). */
ncclResult_t  ncclCommRevoke(ncclComm_t comm, int revokeFlags);
ncclResult_t pncclCommRevoke(ncclComm_t comm, int revokeFlags);

/* Grow: the unique id comm's root gives the ranks joining (ncclCommGetUniqueId), and the communicator of comm's ranks
   and nRanks - comm's count more, made at that id's bootstrap (NCCL 2.29). */
ncclResult_t  ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId);
ncclResult_t pncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId);
ncclResult_t  ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config);
ncclResult_t pncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config);

/* Returns a string for each error code. */
const char*  ncclGetErrorString(ncclResult_t result);
const char* pncclGetErrorString(ncclResult_t result);

/* Returns a human-readable message of the last error that occurred. */
const char*  ncclGetLastError(ncclComm_t comm);
const char* pncclGetLastError(ncclComm_t comm);

#define ncclResetDebugInit()
#define pncclResetDebugInit()

/* Checks whether the comm has encountered any asynchronous errors */
ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t pncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);

/* Gets the number of ranks in the communicator clique. */
ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t pncclCommCount(const ncclComm_t comm, int* count);

/* Returns the cuda device number associated with the communicator. */
ncclResult_t  ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t pncclCommCuDevice(const ncclComm_t comm, int* device);

/* Returns the user-ordered "rank" associated with the communicator. */
ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank);
ncclResult_t pncclCommUserRank(const ncclComm_t comm, int* rank);

/* Register CUDA buffer for zero-copy operation */
ncclResult_t  ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t pncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);

/* Deregister CUDA buffer */
ncclResult_t  ncclCommDeregister(const ncclComm_t comm, void* handle);
ncclResult_t pncclCommDeregister(const ncclComm_t comm, void* handle);

/* Reduction operation selector */
typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;
typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               ncclAvg        = 4,
               /* ncclNumOps: The number of built-in ncclRedOp_t values. Also
                * serves as the least possible value for dynamic ncclRedOp_t's
                * as constructed by ncclRedOpCreate*** functions. */
               ncclNumOps     = 5,
               /* ncclMaxRedOp: The largest valid value for ncclRedOp_t.
                * It is defined to be the largest signed value (since compilers
                * are permitted to use signed enums) that won't grow
                * sizeof(ncclRedOp_t) when compared to previous NCCL versions to
                * maintain ABI compatibility. */
               ncclMaxRedOp   = 0x7fffffff>>(32-8*sizeof(ncclRedOp_dummy_t))
             } ncclRedOp_t;

/* Data types */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
               ncclBfloat16   = 9,
               ncclFloat8e4m3 = 10,
               ncclFloat8e5m2 = 11,
               ncclNumTypes   = 12
} ncclDataType_t;

/* ncclScalarResidence_t: Location and dereferencing logic for scalar arguments. */
typedef enum {
  /* ncclScalarDevice: The scalar is in device-visible memory and will be
   * dereferenced while the collective is running. */
  ncclScalarDevice = 0,

  /* ncclScalarHostImmediate: The scalar is in host-visible memory and will be
   * dereferenced before the ncclRedOpCreate***() function returns. */
  ncclScalarHostImmediate = 1
} ncclScalarResidence_t;

/*
 * ncclRedOpCreatePreMulSum
 *
 * Creates a new reduction operator which pre-multiplies input values by a given
 * scalar locally before reducing them with peer values via summation. For use
 * only with collectives launched against *comm* and *datatype*. The
 * *residence* argument indicates how/when the memory pointed to by *scalar*
 * will be dereferenced. Upon return, the newly created operator's handle
 * is stored in *op*.
 */
ncclResult_t  ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
ncclResult_t pncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);

/*
 * ncclRedOpDestroy
 *
 * Destroys the reduction operator *op*. The operator must have been created by
 * ncclRedOpCreatePreMul with the matching communicator *comm*. An operator may be
 * destroyed as soon as the last NCCL function which is given that operator returns.
 */
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);
ncclResult_t pncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);

/*
 * Collective communication operations
 *
 * Collective communication operations must be called separately for each
 * communicator in a communicator clique.
 *
 * They return when operations have been enqueued on the CUDA stream.
 *
 * Since they may perform inter-CPU synchronization, each call has to be done
 * from a different thread or process, or need to use Group Semantics (see
 * below).
 */

/*
 * Reduce
 *
 * Reduces data arrays of length count in sendbuff into recvbuff using op
 * operation.
 * recvbuff may be NULL on all calls except for root device.
 * root is the rank (not the CUDA device) where data will reside after the
 * operation is complete.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

/*
 * (deprecated) Broadcast (in-place)
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * This operation is implicitely in place.
 */
ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

/*
 * Broadcast
 *
 * Copies count values from root to all other devices.
 * root is the rank (not the CUDA device) where data resides before the
 * operation is started.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

/*
 * All-Reduce
 *
 * Reduces data arrays of length count in sendbuff using op operation, and
 * leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

/*
 * Reduce-Scatter
 *
 * Reduces data in sendbuff using op operation and leaves reduced result
 * scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the result.
 * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
 */
ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t pncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);

/*
 * All-Gather
 *
 * Each device gathers sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

/*
 * All-to-All
 *
 * Each device sends count values to all other devices and receives count values
 * from all other devices. Data to send to destination rank j is taken from
 * sendbuff+j*count and data received from source rank i is placed at
 * recvbuff+i*count.
 */
ncclResult_t  ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

/*
 * Gather
 *
 * Each rank sends count elements from sendbuff to the root rank.
 * On the root rank, data from rank i is placed at recvbuff + i*count.
 * On non-root ranks, recvbuff is not used.
 * root is the rank where data will be gathered.
 *
 * In-place operations will happen if sendbuff == recvbuff + root * count.
 */
ncclResult_t  ncclGather(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclGather(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

/*
 * Scatter
 *
 * On the root rank, count elements from sendbuff+i*count are sent to rank i.
 * On non-root ranks, sendbuff is not used.
 * Each rank receives count elements into recvbuff.
 * root is the rank that will distribute the data.
 *
 * In-place operations will happen if recvbuff == sendbuff + root * count.
 */
ncclResult_t  ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

/* nccl*Config: per-collective-config variant of every collective. It takes a trailing
 * ncclCollConfig_t describing per-call customization. Initialize the config with
 * NCCL_COLLCONFIG_INITIALIZER and set it identically on every rank. config == NULL is
 * equivalent to the plain API. */
ncclResult_t  ncclAllReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t pncclAllReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);

ncclResult_t  ncclBroadcastConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t pncclBroadcastConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);

ncclResult_t  ncclReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t pncclReduceConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);

ncclResult_t  ncclAllGatherConfig(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t pncclAllGatherConfig(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);

ncclResult_t  ncclReduceScatterConfig(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t pncclReduceScatterConfig(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype,
    ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);

ncclResult_t  ncclAlltoAllConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t pncclAlltoAllConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);

ncclResult_t  ncclGatherConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t pncclGatherConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);

ncclResult_t  ncclScatterConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t pncclScatterConfig(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    int root, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);

/*
 * Group semantics
 *
 * When managing multiple GPUs from a single thread, and since NCCL collective
 * calls may perform inter-CPU synchronization, we need to "group" calls for
 * different ranks/devices into a single call.
 *
 * Grouping NCCL calls as being part of the same collective operation is done
 * using ncclGroupStart and ncclGroupEnd. ncclGroupStart will enqueue all
 * collective calls until the ncclGroupEnd call, which will wait for all calls
 * to be complete. Note that for collective communication, ncclGroupEnd only
 * guarantees that the operations are enqueued on the streams, not that
 * the operation is effectively done.
 *
 * Both collective communication and ncclCommInitRank can be used in conjunction
 * of ncclGroupStart/ncclGroupEnd, but not together.
 *
 * Group semantics also allow to fuse multiple operations on the same device
 * to improve performance (for aggregated collective calls), or to permit
 * concurrent progress of the declared collective/WRITE fragments.
 */

/*
 * Group Start
 *
 * Start a group call. All calls to NCCL until ncclGroupEnd will be fused into
 * a single NCCL operation. Nothing will be started on the CUDA stream until
 * ncclGroupEnd.
 */
ncclResult_t  ncclGroupStart(void);
ncclResult_t pncclGroupStart(void);

/*
 * Group End
 *
 * End a group call. Start a fused NCCL operation consisting of all calls since
 * ncclGroupStart. Operations on the CUDA stream depending on the NCCL operations
 * need to be called after ncclGroupEnd.
 */
ncclResult_t  ncclGroupEnd(void);
ncclResult_t pncclGroupEnd(void);

/* A communicator over a program's own operands, made as ncclCommInitRank makes one (Communicators, above): NCCL's
   bootstrap at `commId`, the root's host:port (ncclGetUniqueId, or NCCL_COMM_ID's form; the root listens on that host
   alone where it is an IPv6 address), which the caller hands every rank, its two gathers holding the communicator's
   streams, rank `rank` of `nranks` on the bridge of its own node (each rank's node its bridge's, gathered there).
   `programs` its compiled collectives (mesh-plan.h struct mesh_program, copied, kept for its life; a call runs the
   first whose collective, and a rooted one's root, is the call's and whose `below` its bytes are under, 0 any size), or
   with none (count 0) the structural table (above), every collective along `paths` (its routing tables: each member's
   trees; NULL: the map's shortest-path trees), checked against its map; `small` not read; `name` the communicator's
   (NULL: commId's).  Every rank passes the same commId,
   nranks, paths, programs and name.  `config` NULL, or its `blocking` not 0, is a blocking init.  With `blocking` 0 it
   is NCCL's non-blocking init: *comm is set at once and the init runs on a thread of its own, the call returning
   ncclInProgress (ncclSystemError where no thread could run it); ncclCommGetAsyncError gives ncclInProgress until the
   init ends, then its result, its failure named (ncclGetLastError).  No call takes comm before then but
   ncclCommGetAsyncError, ncclCommRevoke, ncclCommAbort and ncclCommDestroy: a revoke shuts its bootstrap's sockets
   down, and its connect and its root's accept read the revoke at each 100 ms look, so the init ends as a value with
   what it held let go; ncclCommAbort or ncclCommDestroy then joins its thread and frees comm, failed or not.
   ncclCommInitRank is this call with no paths, no programs, small 0, no name and no config, and ncclCommInitRankConfig
   the same with the caller's config (of which only `blocking` is read). */
typedef struct { const struct mesh_program* program; uint64_t below; } ncclMeshProgram;
ncclResult_t ncclMeshCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, const struct mesh_trees* paths,
                                  const ncclMeshProgram* programs, int count, uint64_t small, const char* name, ncclConfig_t* config);
/* A collective's form, an entry of a communicator's configuration (ncclMeshCommForms): `collective` (mesh-plan.h
   MESH_ALLREDUCE, MESH_BROADCAST, MESH_REDUCE, MESH_REDUCE_SCATTER, MESH_ALLGATHER) at `root` (a reduce's or a
   broadcast's; -1 every root; not read for the others), for calls of fewer bytes than `below` (0: any size), compiled
   (mesh-plan.h mesh_compile) along `trees`: ncclMeshTreesOwn the communicator's own (its paths, else the map's packed
   trees, the default's), ncclMeshTreesPacked the map's packed trees (mesh_trees_pack at `eps` and `most` trees a root,
   both the caller's: Blink's packing [Wang et al. 2020], every link a share), a segment cut among its root's trees by
   `shares` (root r's k-th tree in the packing's order, heaviest first, or the communicator's trees' order, shares[r x
   most + k], each 0 or more, a tree of 0 or past `most` carrying none, for the communicator's ranks; NULL: the
   packing's flows); an all-reduce `whole` (1: single-phase, each member's sum up its own tree, its heaviest share's; 0: the
   reduce-scatter and then the all-gather [Patarasuk & Yuan 2009; Rabenseifner 2004]); each tree's part one entry a
   hop, a relay sending each page on as it lands (mesh-plan.h mesh_compile). */
typedef enum { ncclMeshTreesOwn = 0, ncclMeshTreesPacked = 1 } ncclMeshTrees;
typedef struct {
  int collective;
  int root;
  uint64_t below;
  ncclMeshTrees trees;
  int whole;
  uint32_t most;
  double eps;
  const double* shares;
} ncclMeshForm;
/* A communicator's configuration operand: its forms, in order, compiled ahead of its structural table at each map read
   (Programs, above), so a call runs the first form of its collective (and root) whose bound its bytes are under, else
   the caller's programs (a communicator over its own, ncclMeshCommInitRank's) or the default; no entry, no form (the
   default for every collective and size).  Taken at the communicator's next group (or ncclMeshCommPlan), copied
   with their shares; count 0 clears them.  Every rank passes the same forms at the same point between its groups (its
   members compile its table alike from them); its splits and shrinks keep them (a form at a root past a split's ranks
   makes no entry there).  A share is 0 or more (a tree of share 0 carries none of its root's segment); a form whose
   root's trees all take share 0 fails as a value where it is compiled.  The forms of mesh cc8a6f8e^'s default table (its tree sets, single-phase or two-phase, 1 to 64 chunks,
   by size class), named by the caller's configuration where a price chose them: the caller's configuration document
   is the one place a form is decided (metal-microbench docs/principles.md section 6). */
ncclResult_t ncclMeshCommForms(ncclComm_t comm, const ncclMeshForm* forms, int count);
/* design/whole-function.md#interface: the communicator's fabric and a pair's write paths (mesh-fabric.h) */
ncclResult_t ncclMeshCommFabric(ncclComm_t comm, const struct mesh_fabric** fabric);
ncclResult_t ncclMeshCommWriteRoutes(ncclComm_t comm, int source, int destination, const struct mesh_paths* paths);
/* Configuration-time inspection of the installed paths, including exact integer
   weights. The borrowed value remains valid until that pair is reconfigured or
   the communicator is destroyed. Execution does not query routing policy. */
ncclResult_t ncclMeshCommWritePaths(ncclComm_t comm, int source, int destination, const struct mesh_paths** paths);
/* Realization metadata only, before function execution: each rank's fixed
   bytes gathered into rank-major reports. Uses the retained configuration
   channel, with the communicator's deadline; never a payload receipt. */
ncclResult_t ncclMeshCommConfigGather(ncclComm_t comm, const void* local, void* reports, size_t bytes);
/* Finish realization after encoding the complete function. Peers exchange the
   compiled send/landing list ends on every cable and match both directions.
   This is a compiler operation, before submission; no runtime invocation calls it. */
ncclResult_t ncclMeshCommRealized(ncclComm_t comm);
/* Page tables bind local storage during preparation. A message names pages
   and masks confined to those pages. It never supplies a remote address or an
   extent that crosses a page. Each peer uses its own page size and page list;
   their selected bytes form the same logical message in list order.
   Allocation stays alive through execution. Table metadata may be destroyed
   once the WRITE has been prepared; its immutable descriptors own the binding. */
typedef struct ncclMeshWritePlan *ncclMeshWritePlan_t;
typedef struct ncclMeshPageTable *ncclMeshPageTable_t;
typedef struct { uint64_t page, first, count; } ncclMeshPageMask;
ncclResult_t ncclMeshPageTableCreate(const void* local, ncclComm_t comm, int metal, ncclMeshPageTable_t* out);
ncclResult_t ncclMeshPageTableShape(ncclMeshPageTable_t table, size_t* pages, size_t* page_bytes);
/* Local compute preparation may pass a NULL communicator to Create. Binding
   returns the backend view over those same bridge pages, never new storage.
   These objects describe storage during preparation. Recorded bindings may
   name pages containing indices produced during execution; a fixed binding
   and fixed index-list width do not make those index values constant. */
ncclResult_t ncclMeshPageTableBinding(ncclMeshPageTable_t table, ncclMeshBuffer* binding);
ncclResult_t ncclMeshPageTableDestroy(ncclMeshPageTable_t table);
/* A repeated local binding. Instance selection happens while compiling commands;
   execution reads the resulting page lists. Each rank supplies its own period
   and page stride in its local table. tile_bytes is common logical publication geometry, 0 for a
   single publication. invocations is the common function extent; each local
   binding period can differ. A relay can retain a distinct slot per invocation. */
typedef struct { size_t tile_bytes, instances, instance_pages, invocations; } ncclMeshWriteLayout;
/* Affine metadata table: tile t, invocation i names
   (i % instances)*instance_stride + (t/columns)*row_stride + (t%columns)*tile_stride.
   All strides are bytes. These indices are resolved during preparation. */
typedef struct { size_t columns, tile_stride, row_stride, instances, instance_stride; } ncclMeshTileLayout;
ncclResult_t ncclMeshWritePagesPrepare(ncclMeshPageTable_t table, const ncclMeshPageMask* masks, size_t count,
    int source, int destination, size_t max_packet, ncclMeshWriteLayout layout, ncclMeshWritePlan_t* out);
/* A view of a page table's storage, in elements: each `element_bits` bits (sub-byte ones too: the innermost axis then
   packs whole bytes), `rank` axes of `shape` elements `strides` elements apart, its first element `offset` elements past
   the table's first byte; or, where `runs` (not NULL), a mapped view's `nruns` runs in logical order, each its first
   element's offset past the table's first byte and its element count (runs[2k], runs[2k + 1]).  Abstract layout, never an
   address: preparation lowers it to the table's pages, affine rows where it is affine, else its spans in logical order
   (mesh-write-compiler.h mesh_view_layout). */
typedef struct { ncclMeshPageTable_t storage; uint32_t element_bits, rank; const uint64_t* shape; const int64_t* strides;
                 uint64_t offset; const uint64_t* runs; size_t nruns; } ncclMeshLayout;
/* The WRITE whose local view is `local`: the source's values on `source`, the landing on `destination`, a relay's on any
   other rank; every rank's view holds the same logical bytes, in its own layout. */
ncclResult_t ncclMeshWriteLayoutPrepare(const ncclMeshLayout* local, int source, int destination, size_t max_packet,
    ncclMeshWriteLayout layout, ncclMeshWritePlan_t* out);
/* Bind source words once (ncclMeshBuffer on Metal, region address on host).
   Only the source supplies the words and their local layout. All ranks supply
   the same value. value covers one logical tile (the entire write when
   tile_bytes is 0). Each tile/occurrence has its own word or nonoverlapping value
   interval. Initial words are below value - largest_tile_bytes + 1 (using
   the whole byte count for an untiled write); no reset may race an earlier reader.
   A relay still reads its incoming landed word. */
ncclResult_t ncclMeshWriteSourceReady(ncclMeshWritePlan_t plan, const void* words, uint64_t value, ncclMeshTileLayout layout);
/* Destination records are { uint64_t count; ncclMeshBinding bindings[capacity]; }
   in the supplied affine table. Compilation fills each tile's last incoming
   prefix per physical channel, without a whole-write completion command.
   Plans in one group may share records for tiles consumed together; compilation
   then takes the last incoming prefix across those contributors on each cable. */
ncclResult_t ncclMeshWriteLandings(ncclMeshWritePlan_t plan, void* records, uint32_t capacity, ncclMeshTileLayout layout);
/* Execution binds what preparation made: a WRITE alone in its group (not on a fused stream) the definition its plan
   compiled as it was prepared (an untiled plan's; a tiled one's is its declaration's), its issues laid and placed once
   a generation; WRITEs grouped together their group's declared or first-compiled definition. */
ncclResult_t ncclMeshWriteExecute(ncclMeshWritePlan_t plan, size_t invocation, cudaStream_t stream);
ncclResult_t ncclMeshWritePlanDestroy(ncclMeshWritePlan_t plan);
/* A call a program declares ahead (ncclMeshCommPlan): `what` (ncclMeshCallAllReduce, ncclMeshCallBroadcast,
   ncclMeshCallReduce, ncclMeshCallReduceScatter, ncclMeshCallAllGather, ncclMeshCallAlltoAllV or ncclMeshCallWrite:
   the first five the planner's mesh-plan.h MESH_*), its `peer` (a
   send's or a receive's, a reduce's or a broadcast's root), `count` as its call takes it (a reduce-scatter's
   recvcount, an all-gather's sendcount), `counts` (NULL: none; read for these alone) a V form's (ncclMeshAllGatherV's
   and ncclMeshReduceScatterV's, a rank's count each; ncclMeshAlltoAllV's sendcounts then recvcounts), `type` (a WRITE's
   `count` of it is the WRITE's bytes, all a WRITE moves: its values' type is its producer's and consumer's), `op` (a
   reduction's), and `group`: consecutive calls of one number are one group (ncclGroupStart to ncclGroupEnd; a call
   outside one its own group). ncclMeshPlanRun supplies their invocation order. `fused` (not 0): the call is
   issued on a fused stream (ncclMeshStream fused). It uses the same physical striping and routed-write
   representation as other calls; fusion changes publication/consumption encoding, not available cables. */
enum { ncclMeshCallAllReduce = 0, ncclMeshCallBroadcast = 1, ncclMeshCallReduce = 2, ncclMeshCallReduceScatter = 3,
       ncclMeshCallAllGather = 4, ncclMeshCallAlltoAllV = 18, ncclMeshCallWrite = 19 };
/* One record per logical publication tile, in payload order. seconds is a
   prediction relative to this group's producers; order is the producer's
   launch order (UINT64_MAX is reserved), used when predictions tie. Zero seconds explicitly expresses
   no timing distinction, not a measured completion time. These operands are
   copied at declaration and never read by the transport at execution.
   Every rank supplies the same source records. A tiled WRITE requires them;
   an untiled WRITE may supply one record or declare an already available input. */
typedef struct { double seconds; uint64_t order; } ncclMeshProduction;
/* Preparation only. Map an explicit producer traversal onto the fitted duration
   of this group. Predictions order physical descriptors; they add no wait. */
ncclResult_t ncclMeshProductionTimes(ncclMeshProduction* records, size_t count, double seconds);
typedef struct {
  int what, peer;
  size_t count;
  const size_t* counts;
  ncclDataType_t type;
  ncclRedOp_t op;
  int group;
  int destination;
  int fused;
  size_t tile_bytes; /* WRITE publication geometry, 0: whole operand */
  const ncclMeshProduction* production;
  /* Optional complete source-major count matrix for AlltoAllV. Otherwise
     CommPlan exchanges local count metadata once during configuration. */
  const size_t* matrix;
  /* Full-fabric rank bitmaps, ceil(nranks/64) words; NULL means every rank.
     Contributions and result recipients are independent. Rooted Reduce still
     writes to its root; ReduceScatter owners have nonzero counts. */
  const uint64_t* contributors;
  const uint64_t* recipients;
  /* Optional actual bindings, an extension (NULL: none, as before): [0] the send buffer and [1] the receive buffer
     the call will be issued with, as it passes them (on a stream the ncclMeshBuffer itself, on the host {NULL,
     address}; {NULL, 0} where this rank has none). Given on every rank or on none for a call, CommPlan exchanges each
     rank's bindings' offsets within their registration extents once and cuts the call's pieces wherever some rank's
     binding crosses an extent, so no DMA span crosses one for a lease of any size where the call is issued with
     bindings at those offsets (others fail as values at placement). Collectives and all-to-alls; a WRITE has its
     own cuts. A group whose every call declares its bindings is laid and placed for them at CommPlan, in each of its
     generations, so its issues with those buffers bind only their entries. */
  const ncclMeshBuffer* bindings;
} ncclMeshCall;
/* A communicator's plan (ncclMeshCommPlan): its `groups` declared groups, the groups one pass of its order issues
   (`issues`, the sum of each block's count times repeat), the number of the communicator's group the order starts at (`first`: its groups
   numbered from 1 in issue order, the first after the call), and of declared group g this rank's entries on its send
   lists and on its landing lists (entries[g][0] and [1]: a fused crossing's publications and landings, ncclMeshFused). */
typedef struct {
  uint32_t groups;
  uint64_t issues, first;
  const uint64_t (*entries)[2];
} ncclMeshPlan;
/* A compact sequence of contiguous declared groups, repeated as a block.
   first is a group index, not the caller's arbitrary group label. */
typedef struct { uint32_t first, count; uint64_t repeat; } ncclMeshPlanRun;
/* Compile each declared group once. order describes the complete function's
   sequence without duplicating calls for each invocation. Issued groups bind
   their prepared operands to these definitions; execution does not replan.
   count=0 and runs=0 drop the declaration. Every rank supplies the same order. */
ncclResult_t ncclMeshCommPlan(ncclComm_t comm, const ncclMeshCall* calls, int count,
    const ncclMeshPlanRun* order, size_t runs, const ncclMeshPlan** plan);
/* Compiler-derived useful directed bytes for one entire declared function.
   demand has nranks*nranks entries (source-major); group repeats are included.
   Relay amplification is excluded. Query after configuration, before execution;
   the same declarations and route vectors are then realized for the next call. */
ncclResult_t ncclMeshCommPlanDemands(ncclComm_t comm, uint64_t* demand, size_t entries);
/* Finest observable route allocation in the declared graph: maximum independent
   units in any single logical transaction for each ordered pair. Repeats do not
   increase this number. A weight change smaller than 1/units may compile to the
   identical schedule; this is probe resolution, never a minimum path share. */
ncclResult_t ncclMeshCommPlanRouteUnits(ncclComm_t comm, uint64_t* units, size_t entries);
/* Every issued group through `ticket` that is not complete, completed on `stream` (its commandBuffer and
   commandEncoder, or the Metal 4 form's, and its polls; issue not read); *published (may be NULL)
   whether a publication was encoded.  A ticket already complete encodes nothing; one whose communicator failed returns
   the failure. */
ncclResult_t ncclMeshComplete(uint64_t ticket, const ncclMeshStream* stream, int* published);
/* The highest ticket through which every issued group is complete (or lost with a failed communicator). */
uint64_t ncclMeshRetired(void);
/* Fused execution uses the same physical fragment compiler and channel bindings
   as ordinary compiled writes. A peer may have any number of parallel channels,
   and a logical transfer may contain multiple fragments and relay hops. There is
   no peer-sized restriction and no silent fallback that reads an operand before
   the caller's publishing kernel has produced it. Unsupported arithmetic or
   ownership forms must be resolved at realization, not changed to a different
   publication contract while executing.

   The caller supplies room for the declared physical publication and landing
   bindings (ncclMeshFused). The producer writes source pages with the required
   coherence and releases each active channel's cumulative publication. Relay
   entries instead depend on their actual incoming landed prefix and forward from
   those same pages. A source publication never declares an unlanded relay ready.
   The consumer observes only the final incoming prefixes needed by its local
   data uses. It need not await later relay-only entries or local SEND completion.
   Consolidate repeated observation/fence work where the execution graph permits;
   observing availability is not a separate network message.

   Receives have AOT addresses and orders. Fresh disjoint landing ranges can be
   preposted immediately, independent of this node's outgoing publication. Reuse
   needs a proof before the next first write. The recurrent all-gather proof in
   design/whole-function.md permits two source/landing slots when every rank's
   next publication follows consumption of all previous pieces. It does not
   apply to arbitrary asymmetric sends or independent feed-forward calls. Their
   immutable source leases survive until their final local DMA reads end, with
   asynchronous bridge-owned retirement; destination consumers have separate
   lifetimes. No receive gate, per-tensor receipt all-gather or remote ACK is a
   substitute for those proofs.

   Fused kernels are ordered by the caller's serial compute stream. An encoding
   group publishes binding records, not the source values themselves. Repeated
   groups return the first frame's physical records and their per-repeat prefix
   increments; stride addresses distinct page-aligned storage for later repeats.
   These contracts describe expressibility and ownership, not a proof of optimal
   routing, collective decomposition or attained bandwidth. */
/* rdma/mesh.metal's text as this library was built with it, then the Metal text of the steps the library's kernels and
   a fused crossing's kernels run: the whole of what a caller's kernel is compiled after, from the library the process
   loaded, so it never reads a file that may be another build's (a checkout's mesh.metal newer or older than the
   installed library), and needs no path to one.  The steps' text is the argument record (struct mesh_args), a fused
   step's six bindings as one record (struct mesh_step_binding: ncclMeshFused's order, a classic kernel's form of
   them) and mesh_step_enter, mesh_step_wait, mesh_step_claim, mesh_step_failed,
   mesh_step_check, mesh_step_counts, mesh_step_await, mesh_step_write, mesh_step_release and mesh_step_read (any element type: a landed
   element, loaded system-coherent from the caller's own landing buffer), on a holding's counts (`ready` to store,
   `landed` to await); the library's own, so the caller compiles them with its kernels and keeps no copy.
   Every thread of the threadgroup calls mesh_step_await and mesh_step_release (each holds a barrier);
   mesh_step_enter's and mesh_step_await's results are true (no kernel halts on a failure: rdma/NCCL.md "Bounded
   waits"), kept so that a kernel testing them compiles as it is. */
const char* ncclMeshMetalPrelude(void);
/* The Metal buffer (an id<MTLBuffer>) over the whole lease (ncclMemAlloc) holding `ptr`, any byte of it, and ptr's offset
   in it: the library's, made at the lease's first such call and released during deferred lease retirement after
   ncclMemFree, the buffer a tensor allocator
   or the caller's kernels bind over the lease (mesh 3981ca80^'s).  A Metal-path collective's buffers are leases, which
   its bridge sends from and lands in and its kernels read in place (Memory, above); once its group is complete in the
   stream's order, its encoded GPU uses have ended, but registered source SENDs can still be unposted or reading it.
   ncclMemFree is a logical asynchronous release that retains those obligations, not permission to mutate the source.
   Compiled bindings account for every GPU use and destination consumer.
   Explicit masked data functions may produce padded ingress frames and read
   only their valid intervals; this API and the native transport introduce no hidden copy to repair an operand.
   ncclInvalidArgument where ptr is in no lease of this process. */
ncclResult_t ncclMeshMemBuffer(const void* ptr, void** buffer, size_t* offset);
/* The bytes of a page of comm's region (its block of pages, the unit of a lease and of an entry: design/
   pages-and-functions.md), on which each segment of a V collective and each entry starts (Memory, above).
   ncclInvalidArgument where comm or bytes is NULL. */
ncclResult_t ncclMeshCommPage(ncclComm_t comm, size_t* bytes);
/* Whether this process holds leases or communicators of an expired region (*expired 1; Memory, above): the value a
   client reads in its own error path, after a lease or communicator failed, to free what it holds of that region, or
   end, before it leases again.  ncclInvalidArgument where expired is NULL. */
ncclResult_t ncclMeshRegionExpired(int* expired);
/* MPI_Allgatherv and MPI_Reduce_scatter: rank r's segment is counts[r] elements (counts: nranks entries, the same on
   every rank, any of them 0), the segments packed in rank order, so rank r's lies after the segments of the ranks
   before it.  ncclMeshAllGatherV sends counts[rank] elements of sendbuff and leaves every segment in recvbuff (in place
   where sendbuff is rank's segment of recvbuff); ncclMeshReduceScatterV reduces sendbuff's segments (the sum of counts
   elements) by op and leaves rank's in recvbuff, counts[rank] elements.  Equal counts are ncclAllGather and
   ncclReduceScatter; others are one program, the communicator's all-gather or reduce-scatter (its table's, by the
   operand's bytes), its operand cut into those segments (mesh-plan.h mesh_program_steps_cut): one tree program over
   unequal segments, not a rooted call a rank.  Both paths, every type and operator of the equal calls.  Every rank
   passes the same counts: the program's steps on every rank are cut by them, and a rank whose counts differ from its
   peers' lays other entries (the calls do not pair: ncclInvalidUsage, as unequal counts to the equal calls are).  A
   segment that does not start on a page in its buffer fails the call as a value (Memory, above). */
ncclResult_t ncclMeshAllGatherV(const void* sendbuff, void* recvbuff, const size_t* counts, ncclDataType_t datatype, ncclComm_t comm,
                                cudaStream_t stream);
/* Gather bindings in logical rank order. Each receive names its final contiguous
   segment. The same collective graph routes directly into these bindings; no
   packed receive buffer or copy-back operation is constructed. */
ncclResult_t ncclMeshAllGatherBuffers(const void* sendbuff, void* const* recvbuffs, const size_t* counts,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclMeshReduceScatterBuffers(void* const* sendbuffs, void* recvbuff, const size_t* counts,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclMeshAlltoAllBuffers(void* const* sendbuffs, const size_t* sendcounts, void* const* recvbuffs,
    const size_t* recvcounts, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclMeshReduceScatterV(const void* sendbuff, void* recvbuff, const size_t* counts, ncclDataType_t datatype, ncclRedOp_t op,
                                    ncclComm_t comm, cudaStream_t stream);
/* MPI_Alltoallv with counts on the host: rank r sends sendcounts[q] elements to rank q, its segments packed in rank order
   in sendbuff, and receives recvcounts[q] elements from rank q into recvbuff, packed in rank order (rank r's
   sendcounts[q] is rank q's recvcounts[r], else the calls do not pair: ncclInvalidUsage; its own segment copied, the
   lesser of sendcounts[rank] and recvcounts[rank] elements).  Its counts known to both ends, no count crosses, and each pair's
   elements move as one entry of their size, in one group, from and into its segment's place: each segment starts on a
   page of its buffer, else the call fails as a value (Memory, above).  ncclAlltoAll is
   it with every count `count`, and ncclGather and ncclScatter its rooted forms (every count 0 but toward or from the
   root), so recvbuff off a gather's root and sendbuff off a scatter's are not read and may be NULL.  Both paths. */
ncclResult_t ncclMeshAlltoAllV(const void* sendbuff, const size_t* sendcounts, void* recvbuff, const size_t* recvcounts, ncclDataType_t datatype,
                               ncclComm_t comm, cudaStream_t stream);
/* The partials merge, a reduction operator made as ncclRedOpCreatePreMulSum makes one (an operator of comm's, freed by
   ncclRedOpDestroy): split-KV attention partials merged by their log-sum-exp [Dao et al. 2023, Flash-Decoding;
   FlashInfer's state merge].  Its operand is rows of 2 + `dim` elements of `datatype` (ncclFloat32), a row [m, l, O:
   dim] (a row's maximum score, its sum of exponentials and its unnormalized output); two rows merge to the greater m,
   with each one's l and O rescaled by e^(its m - that) and summed, a row whose l is not above 0 counting for none (two
   such merge to [the least float, 0, 0]), so the result's O / l is the attention over every member's positions and the
   result is itself a partial a later merge takes.  Every reducing collective takes it, on either path, inside a group
   or not, each count (a reduce-scatter's, each of ncclMeshReduceScatterV's) whole rows (ncclInvalidArgument else): it
   runs the communicator's program of that collective (its table's, by the operand's bytes) with a row for an element,
   one group, each row merged up its tree where a sum adds an element, every step's piece whole rows, of any width
   (the program's own cuts on whole rows and pages: mesh-plan.h mesh_program_steps_cut). */
ncclResult_t ncclMeshRedOpCreatePartials(ncclRedOp_t* op, size_t dim, ncclDataType_t datatype, ncclComm_t comm);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
