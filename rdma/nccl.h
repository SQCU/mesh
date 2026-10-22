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
   node, the nodes its bridge reaches and the streams free to take (mesh-call.h mesh_nodes_reached, mesh_streams_free);
   each rank then holds, toward each other member's node it is joined to, the stream of the least identity free at both
   ends, as it holds rows (mesh_stream_take: design/transport.md section 14), one communicator a stream (a holding's
   lists and counts are its holder's alone; a link has several, so communicators sharing a node pair hold their own;
   EBUSY where none toward a member's node is free at both ends); the second gather, bounded by mesh.h MESH_PLATFORM_NS
   a send or receive, carries each rank's holding's status and its holdings' start words (the bridge's, opaque), which
   each rank tells its own holding (mesh_stream_tell).  A communicator names nodes and its own pages, never a link,
   port, lane, frame, queue pair or SEND size.  An init fails as a value where two ranks are on one node (its holdings
   and the start words they exchange are kept a node), a stream is held by another process or gone, or a peer is
   silent past that bound; what it held is then let go and the next init reads the streams as they stand.  A
   connection to the root whose report fails or names no rank still to meet is closed alone and the root meets the
   next (one that sends nothing holds the root's read until the phase ends).  A communicator's map is its members joined where their bridges reach each other's nodes, made alike by
   every member from the same reports and kept for its life: no node outside the communicator is on it, since the
   bridges forward nothing, so two members no path between members joins fail a collective, and a call between two
   members not joined fails, as values (ncclInvalidUsage).  ncclMeshCommInitRank (below) is the same init for a program
   that passes its own programs or paths.  ncclCommSplit and ncclCommShrink make communicators of the ranks they take;
   Split holds its children's streams over comm (NCCL's bootstrapSplit), and Shrink makes no exchange, so a shrink to
   two ranks or more holds none: it is made with no channel, and its first call that crosses a link fails as a value
   (a rank among those excluded gets NULL, as NCCL_SPLIT_NOCOLOR's).  Inside a group, Split and Grow run their own
   collectives at once, the group's calls set aside, and Shrink makes none.  ncclCommGrow meets at the bootstrap of the
   id that ncclCommGetUniqueId gives comm's root, which the root broadcasts on comm.  ncclCommRevoke is NCCL's: comm's
   operations in flight are revoked and its resources kept (its streams stay held; the bytes its groups presented and
   published nothing for go at once as duds, so its peers' groups fail rather than wait), an init still
   running (a non-blocking one's) ends as a value, its host-path groups return at their next look and its GPU waits end
   at their landings or their bounds (no kernel stops at the revoke), every later group on comm is refused (ncclInvalidUsage), ncclCommFinalize after
   it returns ncclSuccess (it flushes nothing), and ncclCommGetAsyncError reads ncclSuccess once it is done; its
   revokeFlags is not read; from any thread, under comm's own
   lock, never one a group holds while it waits.  ncclCommDestroy waits for the GPU work the library committed for comm
   at most MESH_PLATFORM_NS, then gives its streams back and frees it; work in a caller's encoder is the caller's to
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
   program's own cuts on pages; a point-to-point call one step) on its communicator's channels: toward each member it
   is joined to, the holding its init took of the stream toward the member's node, each step one entry of its send
   list (the pages its bytes lie in) or its landing list (the pages they land in), each direction one list both ends
   lay alike (mesh-plan.h mesh_group_lay: a rank's k-th send to a peer is the peer's k-th receive from it, the
   point-to-point calls in call order, as NCCL pairs them), presented to the bridge as the group is issued (mesh-call.h
   mesh_stream_present; the holding's deadline none while its member lives: no dud reaches the peer of a producer still
   at work), so a member that fails, is revoked or ends mid-group is replaced at its peers by duds, at once, all ones:
   what is placed where a send must happen and its producer is gone, which its peers land as bytes like any other and
   never read as a signal (design/transport.md section 10, design/rubrics.md M4: nothing else changes).  A publication
   raises the holding's `ready` past the entry; a landing is awaited on its `landed`; a write over bytes a SEND reads
   waits for its `sent`.  Nothing credits a publication: the hardware holds a SEND until its receive is posted.  No
   payload is checked, so calls that do not pair (unequal counts, different collectives or orders) are the caller's
   error and are not detected.  Uneven segments are their own calls
   (ncclMeshAllGatherV, ncclMeshReduceScatterV, below), whose count vector every rank passes alike.

   Host path: a NULL stream.  Buffers are host pointers into this node's region (a lease's: Memory, below), sent from
   and landed in in place and combined on the host; every call returns once its group has run and its SENDs are done.  Metal path: a stream that
   is an ncclMeshStream.  The group is encoded into its commandBuffer (the caller commits it), every buffer argument an
   ncclMeshBuffer (an id<MTLBuffer> and a byte offset into it: an MPS tensor's storage and offset) of any type, a
   crossing's over a lease (Memory, below); float64 moves as its bytes, and a float64 reduction there fails as a value
   (Apple GPUs have no float64 arithmetic).  With commandEncoder, the kernels go into that open compute encoder, left
   open with nothing signalled at the group's end; workspace, where given, gives the scratch of a call that crosses no
   link (an id<MTLBuffer> of at least `bytes`, valid until the group is complete; without it the communicator's own,
   reused in the encoder's order: the scratch a crossing sends from or lands in is the communicator's, leases of its
   pool), and the caller orders any later host-path group after the stream's work.  The calls of one group name one
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
   complete, and waits only for its own communicator's GPU work, at most MESH_PLATFORM_NS.  A host-path group spins on
   its landings, never parks, and holds no lock another communicator's groups take; it fails as a value where nothing
   lands and nothing can be published for MESH_PLATFORM_NS.  A group whose entries lie on one channel completes as
   plain dispatches in the stream's order; a group on two or more, with each channel progressing on its own
   (rdma/NCCL.md "One track", "Per-channel progress").  A group's landings in a stream wait to be posted until the stream
   reaches the group (its first publication on the channel, or a gate store of its first dispatch), so no receive lands
   in pages the stream's earlier work still reads; its end waits until its SENDs are done (`sent`), so the caller may
   write its send buffer again.  A wait is a kernel polling its holding's `landed` count, and `sent` where a write meets
   bytes a SEND reads, and nothing else (rdma/mesh.metal mesh_wait_bounded), which the bridge sets ~0 where the link is
   lost; every wait is bounded: the stream's `polls` is the most loads of a count a wait makes,
   the caller's bound for the group's crossings, which it derives from its plan (the lag it expects at them, over the
   period of one load on the node's GPU: rdma/NCCL.md "Bounded waits"); 0, or more than NCCL_MESH_POLLS, is
   NCCL_MESH_POLLS (design/transport.md); the waits a group encodes at its call take its calls' greatest, and those
   ncclMeshComplete encodes its stream's.  A wait that reaches its bound or reads a lost link records its
   cause, entry and channel in its communicator's failure words, on which no kernel halts (every later dispatch runs as
   encoded, a later wait ending at its landing or its own bound), and the group fails as a value after its command
   buffer (ncclCommGetAsyncError, or the communicator's next group).
   The Metal 4 form: a stream whose argumentTable (an id<MTL4ArgumentTable>) is set, commandEncoder then an
   id<MTL4ComputeCommandEncoder> and residencySet an id<MTLResidencySet> (commandBuffer not read), its polls as the
   classic form's.  The group is encoded as compute dispatches between the encoder's commands before and after it, bound
   through the caller's argument table (at least 6 buffer bindings; no setBytes, each dispatch's arguments a record of
   the library's ring), on pipelines of the residency set's device, every buffer of the library's it touches added to
   the caller's residency set (committed where one was added); nothing is committed or signalled, and with no workspace
   its scratch is the communicator's, reused in the encoder's order.  The classic form reads residencySet only for a
   fused crossing (ncclMeshFusableRooted, below), whose steps the caller's kernels run by their bindings' GPU addresses:
   an id<MTLResidencySet> of the command buffer's device, to which the library adds every buffer those bindings name
   (committed where one was added), and which the caller makes resident for the command buffers its kernels are encoded
   into (its queue's addResidencySet:, or each command buffer's useResidencySet:), since a classic encoder makes resident
   only the buffers bound to it.

   Memory.  The process is one client of its node's bridge: one attach of the node's region, made at its first
   ncclMemAlloc or communicator, whose slot owns the process's leases and holds the streams its communicators use; a
   communicator leases a page for its gate and a pool of leases for the pages its groups send from and land in that are
   not the caller's (a reduce-scatter's operand, a tree's scratch, a combine's landing area), held by a group until it
   completes and returned at ncclCommDestroy.  ncclMemAlloc is the lease,
   the call the operator's division of the work makes of the bridge ("the bridge is there to behave like a system
   call"): whole pages of the region covering size, which the bridge registered on every device when the device came up,
   before any queue pair existed; a lease registers nothing.  The client sizes its leases ("ontologically it must be the
   responsibility of client programs to know how much allocation they need"): a lease the region cannot meet is
   ncclSystemError naming the bytes asked and the pages free, one while every client slot is held ncclSystemError naming
   mesh.h MESH_CLIENTS, and one where the process attaches no region of a bridge that runs names the requirement, a
   bridge whose configuration's region holds the client's footprint (etc/bridge/README.md; ncclRemoteError where the
   bridge stopped, else ncclSystemError); nothing falls back to other memory.  ncclMemFree returns the pages; the leases
   of a process gone are returned by its bridge, as every retired client's rows are.  ncclCommRegister takes a buffer in
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
   buffer an MPS tensor or a kernel takes over the lease.  Zero copy, always (the operator, 2026-10-21: "zero copy
   methods must always be used over access by copy because memory bandwidth and flops are too scarce on this hardware
   api"): the bridge sends a crossing's bytes from the pages they lie in and lands them in the pages they belong in (a
   COPY in its destination, a combine in a landing area of the communicator's, combined from there), each step one entry
   that starts on a page (design/transport.md section 14: max_sge 1, and registrations in extents no page crosses).  So
   every crossing call's send buffer, and every buffer it sends from or lands in, is in this node's region, and every
   piece starts on a page: the program's own cuts lie on pages (mesh-plan.h mesh_program_steps_cut), and a cut NCCL's
   call fixes (a reduce-scatter's and an all-gather's segment r at r x count, an all-to-all's segments, a send's or a
   receive's buffer) that is not on one, or an operand outside the region, fails the call as a value
   (ncclInvalidUsage, naming the piece) before anything is presented; nothing is copied to make it fit ("good
   performance cannot be opt-in ... code which can't perform well can fail instead of being run").  A collective whose
   send buffer lies apart from its result reads it in place where its program allows (a SEND of a range no step of the
   rank wrote reads it, the first combine of such a range writes send op piece, the ranges of the result no step writes
   are copied from it after the publications, and no scratch buffer no step writes is made); else its operand is made in
   place in the region (a premultiply, or a copy within the region).  The NIC reads a SEND's pages until its `sent`
   passes them, so a group completes only once its SENDs are done, and the caller then frees or rewrites its buffers.
   ncclCommRegister refuses a buffer outside the region.  Design: design/transport.md.

   Failures are ncclResult_t values, ncclGetLastError naming what was lost; nothing resumes a failed group, and a
   communicator whose group failed fails its later groups, made again on the map current then (ncclCommDestroy, then an
   init).  No timer of the library's declares a peer lost: a landing that reads ~0 (the bridge's, at its link's loss or
   its stop), a GPU wait's bound or a host wait's MESH_PLATFORM_NS ends the group, and a group on a lost link or whose
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
   rises, rdma/NCCL.md "Bounded waits"); each wait's own, so k waits that never land hold the GPU up to k times it */
#define NCCL_MESH_POLLS 4500000u
/* The stream: NULL (the host path) or an ncclMeshStream (the Metal path, above). */
typedef void *cudaStream_t;
typedef struct { void *buffer; size_t offset; } ncclMeshBuffer;
typedef struct { uint64_t ticket; int published; } ncclMeshIssue;
/* A fused crossing's steps, run by the caller's kernels (ncclMeshFusableRooted, below): a publication toward each peer
   it sends to and a landing from each peer it receives from, each one's six bindings' GPU addresses, in the library's
   binding order (0 d, 1 s, 2 its argument record, 3 a, 4 v, 5 progress); a Metal 4 kernel takes them through its
   argument table, a classic one as one record of its bytes (ncclMeshMetalSteps' mesh_step_binding, the six in this
   order).  The caller gives `publish` and `land` room for `publishes` and `lands` records (ncclMeshFusableRooted's
   counts); the group writes its records in its peers' order (its channels') and sets the two counts to those written:
   0 and 0 where the group is not a fused crossing, which then runs as an ordinary group (ncclMeshFusableRooted). */
typedef struct { uint64_t (*publish)[6], (*land)[6]; uint32_t publishes, lands; } ncclMeshFused;
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
typedef struct ncclWindow_vidmem* ncclWindow_t;
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

/* Window Registration flags */
#define NCCL_WIN_DEFAULT 0x00
#define NCCL_WIN_COLL_SYMMETRIC 0x01
#define NCCL_WIN_STRICT_ORDERING 0x02
/* Restrict window registration to GIN only. By default, register for all supported capabilities. */
#define NCCL_WIN_GIN_ONLY 0x04
/* Restrict window registration to CFT COUNTED only. By default, register for non-COUNTED LEs. */
#define NCCL_WIN_CFT_COUNTED 0x08

#define NCCL_WIN_REQUIRED_ALIGNMENT 4096

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

/* This struct will be used by ncclGroupSimulateEnd() API to query information about simulation. */
typedef struct ncclSimInfo_v22200 {
    size_t size;
    unsigned int magic;
    unsigned int version;
    float estimatedTime;
} ncclSimInfo_t;

/* NCCL_SIM_INFO_INITIALIZER must be assigned to initialize simInfo structure when it is created.
 * Not initialized simInfo will result in NCCL error. */
#define NCCL_SIM_INFO_INITIALIZER {                                  \
  sizeof(ncclSimInfo_t),                     /* size */              \
  0x74685283,                                /* magic */             \
  NCCL_VERSION_CODE,                         /* version */           \
  NCCL_UNDEF_FLOAT                           /* estimated time */    \
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
 * Send
 *
 * Send data from sendbuff to rank peer.
 *
 * Rank peer needs to call ncclRecv with the same datatype and the same count from this
 * rank.
 *
 * This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
 * need to progress concurrently to complete, they must be fused within a ncclGroupStart/
 * ncclGroupEnd section.
 */
ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

/*
 * Receive
 *
 * Receive data from rank peer into recvbuff.
 *
 * Rank peer needs to call ncclSend with the same datatype and the same count to this
 * rank.
 *
 * This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
 * need to progress concurrently to complete, they must be fused within a ncclGroupStart/
 * ncclGroupEnd section.
 */
ncclResult_t pncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

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
 * concurrent progress of multiple send/receive operations.
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
   (mesh-plan.h mesh_compile) along `trees`: ncclMeshTreesShortest the communicator's own (its paths, else the map's
   shortest-path trees), ncclMeshTreesPacked the map's packed trees (mesh_trees_pack at `eps` and `most` trees a root,
   both the caller's: Blink's packing [Wang et al. 2020], every link a share), a segment cut among its root's trees by
   `shares` (root r's k-th tree in the packing's order, heaviest first, or the shortest trees' order, shares[r x most +
   k], each 0 or more, a tree of 0 or past `most` carrying none, for the communicator's ranks; NULL: equally), the
   configuration's allocation, not the flows the packing infers from the map's links; an all-reduce `whole` (1: single-phase, each member's sum up its own tree, its heaviest share's; 0: the
   reduce-scatter and then the all-gather [Patarasuk & Yuan 2009; Rabenseifner 2004]); each tree's part cut into
   `chunks` (1: uncut), pipelined down and up the trees [Blink; NCCL's chunked trees]. */
typedef enum { ncclMeshTreesShortest = 0, ncclMeshTreesPacked = 1 } ncclMeshTrees;
typedef struct {
  int collective;
  int root;
  uint64_t below;
  ncclMeshTrees trees;
  int whole;
  uint32_t chunks;
  uint32_t most;
  double eps;
  const double* shares;
} ncclMeshForm;
/* A communicator's configuration operand: its forms, in order, compiled ahead of its structural table at each map read
   (Programs, above), so a call runs the first form of its collective (and root) whose bound its bytes are under, else
   the caller's programs (a communicator over its own, ncclMeshCommInitRank's) or the default; no entry, no form (the
   default for every collective and size).  Taken at the communicator's next group (or ncclMeshFusableRooted), copied
   with their shares; count 0 clears them.  Every rank passes the same forms at the same point between its groups (its
   members compile its table alike from them); its splits and shrinks keep them (a form at a root past a split's ranks
   makes no entry there).  A share is 0 or more (a tree of share 0 carries none of its root's segment); a form whose
   root's trees all take share 0 fails as a value where it is compiled.  The forms of mesh cc8a6f8e^'s default table (its tree sets, single-phase or two-phase, 1 to 64 chunks,
   by size class), named by the caller's configuration where a price chose them: the caller's configuration document
   is the one place a form is decided (metal-microbench docs/principles.md section 6). */
ncclResult_t ncclMeshCommForms(ncclComm_t comm, const ncclMeshForm* forms, int count);
/* Every issued group through `ticket` that is not complete, completed on `stream` (its commandBuffer and
   commandEncoder, or the Metal 4 form's, and its polls; issue and workspace not read); *published (may be NULL)
   whether a publication was encoded.  A ticket already complete encodes nothing; one whose communicator failed returns
   the failure. */
ncclResult_t ncclMeshComplete(uint64_t ticket, const ncclMeshStream* stream, int* published);
/* The highest ticket through which every issued group is complete (or lost with a failed communicator): its
   workspace may be reused in the stream's order. */
uint64_t ncclMeshRetired(void);
/* A crossing whose steps the caller's own kernels run (either Metal form: the Metal 4 form, or the classic form with its
   residencySet, Metal path above; metal-microbench docs/kernels.md#collectives, Q5P's in-stream form and its one-sided
   fused crossings, and torch-mesh's torch_mesh/fused.py on torch's stream, the steps still this library's:
   docs/acceptance-20261014.md A39, A40).  A fused group, on a stream whose `fused` is set, is completed at once (issue
   or not: issue->ticket 0), the groups in flight before it completed first, so it completes alone, each of its channels
   in the stream's order, and its steps that cross are at most one SEND and one landing in place toward each peer
   (ncclMeshFused's records), each of a whole buffer, no combine: ncclSend(send, n, type, q) and ncclRecv(landing_q, n,
   type, q) toward each peer q in one group (an all-reduce's or a reduce's one-shot form: each rank sends its part to
   every peer and receives every peer's; the library cannot see when the caller's kernels read a landing area of its
   own, so the caller sums its own part with what lands in its landing buffers), any of them alone, an in-place
   ncclBroadcast, or an in-place ncclAllGather (sendbuff recvbuff + rank x sendcount, NCCL's), any type and any size,
   each buffer a lease's and starting on a page (Memory, above).  The call presents its entries, encodes every other step of the
   group as ever (its gate store, where its landing has no publication of its own to wait on) and the fused steps in no
   dispatch: their records are written and their bindings returned in *fused (each six GPU addresses, ncclMeshFused's
   order; none where the group has no such step, a group that crosses no link among them).  A publication: the caller's
   kernel that writes the operand (d + record.dst: its send buffer) calls mesh_step_enter on each publication's
   bindings, writes it, and from its last threadgroup to arrive (the one that knows every other threadgroup's stores are
   done: the caller's own count of arrivals) calls mesh_step_release on each publication's a, one system fence and one
   store of that holding's `ready`; it is encoded after the call returns.  The landing: the
   kernel after it calls mesh_step_enter and mesh_step_await on a (every threadgroup may: each its own bounded wait for
   the holding's `landed` past the landing, and its `sent` past the group's send, so that once it returns the send
   buffer may be written again; each ends at once after any wait on its failure words has failed), then reads the landed bytes from d + record.dst (its landing buffer) with
   mesh_step_read and combines them with its own part where it reduces.  A kernel that writes the send buffer again
   with no landing after the publication calls mesh_step_sent on the publication's bindings first.  The call encodes no
   dispatch of either fused step, so the caller's kernels are ordered by its stream alone: on the classic form, its
   queue's order (a serial compute encoder's dispatches, its command buffers in commit order).  A group on such a stream
   that is not a fused crossing (one that crosses no link; more than one SEND or landing toward a peer, a combine or a
   scratch; a SEND of bytes its call also receives, a forwarding rank's) is not refused: it runs as an ordinary group
   on that stream, its steps encoded by the library as on any stream (issued where the stream asks issue, else
   completed), writes no record and sets both counts to 0, so the caller's kernels publish and await nothing; as an
   ordinary call it reads its send buffer at its place in the stream, so the kernel that writes it is encoded before
   the call.  A group on a classic stream without its residencySet fails (ncclInvalidArgument).  The landing's receive
   is posted only once its gate opens.  On a channel where the group also publishes, the gate is that publication's
   `ready`, which the caller's publishing kernel stores from its last threadgroup, so the landing buffer may be the one
   the previous crossing's landing kernel read where that kernel is the publishing kernel itself (fused.py's step
   kernel, which lands crossing t - 1 and publishes crossing t) or one encoded before it.  On a channel where the group only lands (a reduce's
   root, a broadcast's receiver: no publication of its own to wait on) the gate is the communicator's gate word, which
   the call stores in the stream: its receive is posted once the stream reaches the call, so where its landing buffer
   is the one the previous crossing's landing kernel read, that kernel is encoded before the call (as for any buffer
   an ordinary call writes); the library sees no kernel of the caller's, so it can order nothing after one.
   *publishes and *lands are this rank's fused steps where a fused group of `kind` rooted at `root` (an all-reduce's and
   an all-gather's root 0) of `count` elements of `type` (an all-gather's a rank's piece, its sendcount) runs on comm
   of n ranks (one rank, or no elements, crosses nothing and has no fused step): an all-reduce's one-shot form n - 1 of
   each (the caller's send to and receive from every peer), a reduce's root n - 1 landings and every other rank one
   publication (the same calls, toward the root), a broadcast's and an all-gather's the SENDs and landings of this
   rank's moves in the communicator's program, where it moves each whole operand in one entry each way and sends no
   part it also receives (a configured form that cuts it, and a forwarding rank, a chain's middle or a broadcast tree's
   inner rank, do not fuse: both 0).  Every count is from comm's map: a fused step is a SEND or a landing on this
   rank's channel toward its peer, so where the form uses a peer this rank has no channel to (its bridge reaches no
   node of that rank's: an all-reduce's or a reduce root's any other rank, a reduce's other ranks' root, a program's
   move toward it) the form has no fused step here and both are 0.  Both 0 is a form choice, not a refusal: the caller
   issues the unfused call, which completes.  `op` is not read: no fused step combines. */
typedef enum { ncclMeshFuseAllReduce = 0, ncclMeshFuseBroadcast = 1, ncclMeshFuseReduce = 2, ncclMeshFuseAllGather = 3 } ncclMeshFuseKind;
ncclResult_t ncclMeshFusableRooted(ncclComm_t comm, ncclMeshFuseKind kind, int root, size_t count, ncclDataType_t type, ncclRedOp_t op,
                                   int* publishes, int* lands);
/* The Metal text of the steps the library's kernels and a fused crossing's kernels run, after rdma/mesh.metal's: the
   argument record (struct mesh_args), a fused step's six bindings as one record (struct mesh_step_binding: ncclMeshFused's
   order, a classic kernel's form of them) and mesh_step_enter, mesh_step_wait, mesh_step_claim, mesh_step_failed,
   mesh_step_check, mesh_step_counts, mesh_step_await, mesh_step_sent, mesh_step_release and mesh_step_read (any element type: a landed
   element, loaded system-coherent from the caller's own landing buffer), on a holding's counts (`ready` to store,
   `landed` and `sent` to await); the library's own, so the caller compiles them with its kernels and keeps no copy.
   Every thread of the threadgroup calls mesh_step_await, mesh_step_sent and mesh_step_release (each holds a barrier);
   mesh_step_enter's and mesh_step_await's results are true (no kernel halts on a failure: rdma/NCCL.md "Bounded
   waits"), kept so that a kernel testing them compiles as it is. */
const char* ncclMeshMetalSteps(void);
/* rdma/mesh.metal's text as this library was built with it, then ncclMeshMetalSteps' text: the whole of what a caller's
   kernel is compiled after, from the library the process loaded, so it never reads a file that may be another
   build's (a checkout's mesh.metal newer or older than the installed library), and needs no path to one. */
const char* ncclMeshMetalPrelude(void);
/* The argument records this process has encoded on the default device (the Metal 4 form's dispatches' and either form's
   fused steps'; `issued`, counting from its first program) and how many its ring holds unread by the GPU at most
   (`capacity`): an encode past that refuses with
   EOVERFLOW mid-group (a failed group), so a caller that encodes several submissions ahead bounds each submission's records by it (the
   records a step encodes, read across one step's encode). */
void ncclMeshMetal4Records(uint64_t* issued, uint64_t* capacity);
/* The number comm's next group takes (*group): every group the communicator issues takes the next number, from 0 in
   the order they are issued, whatever its elements and whether it crosses a link (each ncclGroupEnd, and each call
   outside ncclGroupStart and ncclGroupEnd, a group of its own); a group refused as a value before it is issued (an
   invalid group or argument, a revoked communicator, a lease's GPU work past the deadline) takes none.
   ncclInvalidArgument where comm or group is NULL. */
ncclResult_t ncclMeshPosition(ncclComm_t comm, uint64_t* group);
/* For each of comm's groups in [first, first + count) (numbered as ncclMeshPosition gives them), issued[i] and
   ended[i]: `issued`, the earliest time this node's bridge posted the first SEND of the group's first send entry on any
   of its channels (where it sends none, its `ended`); `ended`, the later of `issued` and the latest time its last
   landing entry on any channel wholly landed here; times are ns of this node's monotonic clock (CLOCK_UPTIME_RAW's),
   converted here from the ticks its bridge stored in the holdings' time rings (design/transport.md section 14).  0 in
   both where any of those entries is untimed (its bridge's time ring of 16,384 entries a list lapped it, or has not
   stored it yet) or the group has left the record (the last 65,536 groups of the communicator; one not yet issued has
   not entered it); a group that crosses no link (no elements, a member the tree does not join, a communicator of one
   rank, a communicator without streams: a shrink's) records no entries and is 0 too.  Read as the counts are: no bridge
   call, no request, no file.  ncclSuccess (ncclInvalidArgument where comm is NULL, or issued or ended is NULL and count
   is not 0). */
ncclResult_t ncclMeshTimes(ncclComm_t comm, uint64_t first, uint64_t count, uint64_t* issued, uint64_t* ended);
/* The Metal buffer (an id<MTLBuffer>) over the whole lease (ncclMemAlloc) holding `ptr`, any byte of it, and ptr's offset
   in it: the library's, made at the lease's first such call and released by ncclMemFree, the buffer a tensor allocator
   or the caller's kernels bind over the lease (mesh 3981ca80^'s).  A Metal-path collective's buffers are leases, which
   its bridge sends from and lands in and its kernels read in place (Memory, above); once its group is complete in the
   stream's order (its landings landed, its SENDs done) nothing reads or writes the lease for it, and ncclMemFree may
   return its pages.  ncclInvalidArgument where ptr is in no lease of this process. */
ncclResult_t ncclMeshMemBuffer(const void* ptr, void** buffer, size_t* offset);
/* Whether this process holds leases or communicators of an expired region (*expired 1; Memory, above): the value a
   client reads in its own error path, after a lease or communicator failed, to free what it holds of that region, or
   end, before it leases again.  ncclInvalidArgument where expired is NULL. */
ncclResult_t ncclMeshRegionExpired(int* expired);
/* The caller's GPU work on the lease holding `ptr` (writing it where `write`, else reading it) is done once `event` (an
   id<MTLSharedEvent>, retained until then) reaches `value` (mesh 3981ca80^'s, for leases): a host-path group that reads
   the lease waits for its writers' points, one that writes it for every point, on the host, each until the caller's
   GPU work reaches it, before the group takes any lock; a point reached is dropped, and ncclMemFree drops the rest.
   `bytes` is not read (a point is the whole lease's).  The Metal path orders its work by its stream and reads no point.
   ncclInvalidArgument where ptr is in no lease of this process. */
ncclResult_t ncclMeshMemUse(const void* ptr, size_t bytes, void* event, uint64_t value, int write);
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
/* MPI_Alltoallv with counts the GPU wrote (mesh 3981ca80^'s signature; expert-parallel dispatch and combine with no
   host count): refused as a value (ncclInvalidUsage, *arrived set to 2).  Its rows' places in sendbuff and recvbuff
   are computed on the GPU from its counts, so none of them can be an entry the host presents (a page's first byte and
   its bytes: Memory, above); its rows could cross only through a copy, which this library never makes.  Its rows move
   with ncclMeshAlltoAllV, from counts the host holds. */
ncclResult_t ncclMeshAlltoAllCounted(const void* sendbuff, size_t sendrows, const int64_t* counts, const size_t* sendsegs, void* recvbuff,
                                     size_t capacity, int64_t* received, int64_t* landed, const size_t* recvsegs, uint64_t* arrived, size_t row,
                                     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
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
