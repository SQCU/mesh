/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* libnccl-mesh: NCCL's API over the mesh's Thunderbolt 5 RDMA bridges.

   Communicator, datatype and reduction declarations derive from NVIDIA/nccl master
   12df1a1 src/nccl.h.in (NCCL 2.32.3). The NCCL-style communicator functions retain
   their profiling aliases. Tensor work is declared through ncclMeshCommPlan and
   lowered with the complete function; eager collective/group entry points are absent.  Installing, linking and running it: rdma/NCCL.md.  The mesh's own calls (ncclMesh*) are last.

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
   bound its operand is under; these choices are resolved during declaration preparation.  Every member compiles the same programs from the same map and the same
   operands (every rank passes the same paths, programs and forms), so nothing is compared at run time.  A declared group
   walks each call's program (mesh_program_steps_cut: the
   program's typed cuts and declared WRITE endpoints), lowered to explicit physical cable/range
   records on its communicator's channels: one holding per cable, each transfer one entry of its send
   list or landing list of immutable exact byte spans. Each direction has one
   matching DMA-unit sequence at both ends, in compiled dependency order,
   its tables linked to the bridge at the group's place in its program (mesh.h struct mesh_zwoggle,
   design/zorvek.md zinth).  A member that fails, is revoked or ends mid-group sends nothing more: its peers' host-path
   waits end at a revoke, the link's loss or their deadlines, and their GPU waits at their bounds (a deadline past
   which the program ends: design/eager-execution.md, assumption A4), and its holding's end with units outstanding takes its stream out of its link's pairing
   (design/transport.md section 14).  A publication
   raises the holding's `ready` past the entry; a landing is awaited on its `landed`.
   The data program owns source/relay lifetimes; there are no `sent` waits (design/pages-and-functions.md, Freeing 6-7).  Nothing credits a publication: the hardware holds a SEND until its receive is posted.  No
   payload is checked, so calls that do not pair (unequal counts, different collectives or orders) are the caller's
   error and are not detected. Uneven segments are declared by the same collective kinds
   with an explicit count vector shared by the ranks.

   Recording. ncclMeshCommPlan declares a complete function's collective calls and bindings.
   ncclMeshCommLower selects an occurrence's pages and transfers. ncclMeshCommSources
   exposes its numerical source and exact page relations to the enclosing function's
   common dependency solve. ncclMeshCommSpecialize retains those emitted sources;
   ncclMeshCommEncode records only that prepared result into the function's recorder.
   CPU and Metal numerical leaves use the same source/effect contract. No collective
   call takes new payload bindings or compiles a different execution path at recording.

   Memory.  The process is one client of its node's bridge: one attach of the node's region, made at its first
   ncclMemAlloc or communicator, whose slot owns the process's leases and holds the streams its communicators use; a
   communicator leases a page for its gate and one for its failure words, and, for each group definition (a declared
   group, or a prepared WRITE's), two slots of what its groups send from and land in that are not the caller's (a
   reduce-scatter's partials of the others' segments, a non-receiving rank's operand, a tree's scratch, a relay's
   generations, a combine's landing area): a declared group's made at its declaration (ncclMeshCommPlan, which
   fails as a value where the region cannot hold them), a WRITE's as its plan is prepared, its groups taking turns
   at them.  A group that no declaration covers, or a group of collectives whose declaration binds no buffers, ends
   the program with a warning: nothing compiles or binds one at its issue (design/eager-execution.md),
   ncclMemAlloc called again only where one must grow (to one of eight sizes an octave), a dropped definition's slots
   kept for the next, one too small returned once its last group is done (the host path's, or a Metal path's that the
   stream has run past: it stores that group's ticket where its generation let the slot go), all returned at
   ncclCommDestroy.  What the GPU reads of a group once it is encoded (its entries for the bridge's lists, its tracks'
   records, a Metal 4 dispatch's argument record) is cut from the process's pieces of the region, each a lease with its
   Metal buffer made once, taken again once the GPU has done with all cut from it (the entries presented, the tracks
   ended, the residency set that holds argument records ended): no Metal buffer is made at a call.  A group's host
   bookkeeping (its schedules, its steps' state, its record in flight) and its entries' span tables (a lease of them)
   are its definition's too.  A group of collectives' are laid and placed at its declaration for its bindings, one for
   each generation and each slot of them (ncclMeshBufferSlots), and a group binds only its entries on the channels:
   the buffers its calls name are not read.  A prepared WRITE's alone are laid as its plan is prepared, an occurrence
   each; a declared group of WRITEs' at each of its groups, for the plans its calls name (the old tables let go once
   the bridge's reads of them end).  A result is made in recvbuff (a reduce-scatter's own segment too: no compiled piece spans two
   segments), and nothing is copied into a slot from a caller's buffer but the operand of a premultiplied,
   partials-merging or fused call where it is not the result, and an in-place single-phase all-reduce's original.  A
   group writes a slot only once the slot's last group has every landing awaited: the stream's order, and a landed
   wait for those its stream did not await.  A SEND's source is reused as design/pages-and-functions.md (Freeing) states:
   after a natural downstream landing or the bridge's own DMA-read completion; nothing is sent back.  ncclMemAlloc is the lease,
   the call the operator's division of the work makes of the bridge ("the bridge is there to behave like a system
   call"): whole pages of the region covering size, which the bridge registered on every device when the device came up,
   before any queue pair existed; a lease registers nothing. Preparation maps its selected bridge
   pages into a fixed local virtual interval. Those pages may be scattered in the arena; CPU, Metal and CoreML views
   share their contents without a copy. Execution does not change these mappings. Transport preparation resolves
   their bridge offsets from page-table rows, never from subtraction of a local virtual address. A descriptor table
   declares linear backing because the bridge reads it as one interval; tensor storage has no such requirement.
   The client sizes its leases ("ontologically it must be the
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
   Generic collective bindings preserve those same physical constraints. Preparation joins actual backing breaks,
   registration cuts and fragment endpoints across the declared slots and relays in common byte coordinates.
   Each emitted packet stays inside its backing run and registration extent, including a break within a numerical
   row. Numerical readiness names the packet containing the row's final byte.
   A collective whose
   send buffer lies apart from its result reads it in place where its program allows (a SEND of a range no step of the
   rank wrote reads it, the first combine of such a range writes send op piece, the ranges of the result no step writes
   are copied from it after the publications, and no scratch buffer no step writes is made); else its operand is made in
   place in the region (a premultiply, or a copy within the region). The NIC reads a SEND's pages until the SEND is
   done, and a group completes without waiting for that. Published source bytes are immutable until every registered
   physical DMA reader has finished, including readers on parallel cables, relay paths and not-yet-posted entries.
   Mutating them earlier is invalid. The compiled page assignment includes local GPU reads,
   incoming writes and destination consumers before it assigns an overwrite. A landing into bytes a SEND reads (an
   in-place collective's later step, a later call over the same pages) is posted by the bridge only once that SEND
   completed, its channel's `sent` at its end (mesh.h struct mesh_zwoggle_entry `after`), a wait the compiler states
   and orders (an exchange in place, each side's landing on its own SEND, is refused as a value). Logical free defers
   physical reuse under these source obligations without adding a receipt, acknowledgment or data-program wait.
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
   land hold the GPU about one bound in all. This is a load-count budget, not a
   stored absolute timestamp; actual elapsed time depends on the device. */
#include "metal-capabilities.h"
#include "mesh-fabric.h"
#define NCCL_MESH_POLLS MESH_POLL_LIMIT
/* Recording context: an ncclMeshStream supplied by the common function recorder. */
typedef void *cudaStream_t;
typedef struct { void *buffer; size_t offset; } ncclMeshBuffer;
/* Coverage is ordinary page data over the declared cut domain. active[k] is
   the payload mask; expected[k] is the number of numerical contributions.
   Both are uint32 values. Their producers/arrivals are declared dependencies,
   so zero-initialized storage never stands in for a completed masked cut. */
typedef struct { ncclMeshBuffer active, expected; size_t cuts; } ncclMeshPageCoverage;
/* Typed preparation output, before backend argument records contain addresses.
   operand names the declaration's source or destination binding. first/bytes
   are frame-local logical bytes in that binding: a gathered source and a
   scattered result start at zero on their owning rank. WRITE masks retain
   their selected-byte coordinates. The compiler consumes
   these operands before it binds numerical commands. Publications on one word
   form an ordered prefix: its value advances only through completed source
   cuts, using the common page-effect completion relation. */
typedef struct {
  size_t call,first,bytes;
  uint32_t kind,frame,channel,polls,operand;
  ncclMeshBuffer word,violation;
  uint64_t threshold,entry;
  /* 1 + the retained native transport ordinal, or 0 for a purely local
     effect. transport_first is the selected byte's offset in that transport. */
  size_t transport,transport_first;
} ncclMeshPageEffect;
enum { ncclMeshPagePublish=1, ncclMeshPageObserve=2 };
enum { ncclMeshPageSource=0, ncclMeshPageDestination=1, ncclMeshPageInternal=2 };
/* Portions of a declared source supplied by the enclosing function, rather than
   one of its numerical commands. Prior remote arrivals remain typed word
   operands on the actual SEND, not separate observation kernels. */
typedef struct {
  size_t call,first,bytes;
  uint32_t frame;
  const ncclMeshPageEffect* observations;
  size_t count;
} ncclMeshSourceInput;
typedef struct {
  void *commandEncoder;
  void *argumentTable;
  void *residencySet;
  uint32_t polls;
  /* Each declared binding uses (invocation + displacement) % instances. */
  size_t invocation;
} ncclMeshStream;

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
 * called by different threads/processes. */
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
/* Source-level fold laws. The graph compiler applies these before capturing
   numerical functions and their page transfers. */
enum { ncclMeshReductionReassociate = 1, ncclMeshReductionReorder = 2 };

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
/* The channels this rank's lists lay, a cable each that ends here: the count every kernel's per-channel records hold. */
ncclResult_t ncclMeshCommChannels(ncclComm_t comm, int* channels);
/* The most entries one repeat of a group of WRITEs, `root` to each of the n `destinations` with `count` elements of
   `element_bytes`, lays on any directed cable: every rank's transfers as the library compiles a WRITE's steps (its
   fabric, installed paths and page), each SEND an entry on its cable's send list at one end and its landing list at the
   other. A group of R repeats lays R times this on that list (configuration time only). */
ncclResult_t ncclMeshCommWriteEntries(ncclComm_t comm, int root, const int* destinations, int n, size_t count,
                                      size_t element_bytes, uint64_t* most);
/* Realization metadata only, before function execution: each rank's fixed
   bytes gathered into rank-major reports. Uses the retained configuration
   channel, with the communicator's deadline; never a payload receipt. */
ncclResult_t ncclMeshCommConfigGather(ncclComm_t comm, const void* local, void* reports, size_t bytes);
/* Realization metadata only, before function execution: the root's fixed
   bytes at `data` on every rank, no other rank sending any. Uses the retained
   configuration channel, with the communicator's deadline; every rank passes
   the same root and bytes. */
ncclResult_t ncclMeshCommConfigBroadcast(ncclComm_t comm, int root, void* data, size_t bytes);
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
typedef struct ncclMeshPageTable *ncclMeshPageTable_t;
typedef struct { uint64_t page, first, count; } ncclMeshPageMask;
ncclResult_t ncclMeshPageTableCreate(const void* local, ncclComm_t comm, int metal, ncclMeshPageTable_t* out);
/* the region bytes [local, local + bytes) of a live local lease lie in: each physically contiguous run, {its first
   byte's region offset, its bytes}, in order, into `runs` (capacity *count in, the runs out: those past the capacity
   not written); those of the range alone, never the lease's whole table.  ncclInvalidArgument where `local` lies in
   no live lease (ordinary storage) */
typedef struct { uint64_t offset, bytes; } ncclMeshRegionRun;
ncclResult_t ncclMeshRegionRuns(const void* local, size_t bytes, ncclMeshRegionRun* runs, size_t* count);
ncclResult_t ncclMeshPageTableShape(ncclMeshPageTable_t table, size_t* pages, size_t* page_bytes);
/* Local compute preparation may pass a NULL communicator to Create. Binding
   returns the backend view over those same bridge pages, never new storage.
   These objects describe storage during preparation. Recorded bindings may
   name pages containing indices produced during execution; a fixed binding
   and fixed index-list width do not make those index values constant. */
ncclResult_t ncclMeshPageTableBinding(ncclMeshPageTable_t table, ncclMeshBuffer* binding);
/* Preparation-only page composition. The result maps the selected existing
   pages into one local view and retains their owners; no payload is copied.
   alignment is an explicit multiple of the source page size. Destroying the
   result releases the view and its retained owners through ordinary retirement. */
typedef struct { ncclMeshPageTable_t storage; uint64_t first, count; } ncclMeshPageSelection;
ncclResult_t ncclMeshPageTableCompose(const ncclMeshPageSelection* selections, size_t count, size_t alignment,
    ncclComm_t comm, int metal, ncclMeshPageTable_t* out);
/* Physical page identities, coalesced into runs. With runs NULL, return the
   required count; otherwise *count supplies capacity and receives the count.
   `page` is the first page in this process's one registered region mapping. */
typedef struct { uint64_t first, count, page; } ncclMeshPageBacking;
ncclResult_t ncclMeshPageTableBacking(ncclMeshPageTable_t table, ncclMeshPageBacking* runs, size_t* count);
/* Local base of the prepared page view; creates no backend resource. */
ncclResult_t ncclMeshPageTableAddress(ncclMeshPageTable_t table, const void** address);
ncclResult_t ncclMeshPageTableDestroy(ncclMeshPageTable_t table);
/* Declaration metadata retained by the common command recorder. Accesses
   belong to individual lowered commands, not an enclosing collective group. */
typedef struct ncclMeshGroup {
  ncclComm_t comm;
  size_t invocation;
  uint64_t groups;
} ncclMeshGroup;
/* Observe qualifies a monotone progress read. Initialization stays ordered;
   exact payload-page accesses carry producer dependencies. Atomic contributors
   commute, so neither peers nor prefix observers acquire a whole-output join. */
enum { ncclMeshAccessRead = 1, ncclMeshAccessWritten = 2, ncclMeshAccessObserve = 4, ncclMeshAccessNext = 64, ncclMeshAccessAtomic = 128 };
typedef struct { ncclMeshBuffer storage; size_t bytes; uint32_t access; } ncclMeshAccess;
/* Affine progress table. Tile t of invocation i names
   (i % instances)*instance_stride + (t/columns)*row_stride + (t%columns)*tile_stride.
   All strides are bytes; preparation resolves these indices. */
typedef struct { size_t columns, tile_stride, row_stride, instances, instance_stride; } ncclMeshTileLayout;
/* Local geometry over a call's page binding. Elements have element_bits bits.
   shape and strides describe an affine view. runs, when present, instead names
   pairs of element offsets and counts in logical order. offset and frame_stride
   use elements. This declaration contains no storage or separate lifetime. */
typedef struct { uint32_t element_bits, rank; const uint64_t* shape; const int64_t* strides;
                 uint64_t offset; const uint64_t* runs; size_t nruns; int64_t frame_stride; } ncclMeshLayout;
/* A call a program declares ahead (ncclMeshCommPlan): `what` (ncclMeshCallAllReduce, ncclMeshCallBroadcast,
   ncclMeshCallReduce, ncclMeshCallReduceScatter, ncclMeshCallAllGather, ncclMeshCallAlltoAllV or ncclMeshCallWrite:
   the first five the planner's mesh-plan.h MESH_*), its `peer` (a
   send's or a receive's, a reduce's or a broadcast's root), `count` as its call takes it (a reduce-scatter's
   recvcount, an all-gather's sendcount), `counts` (NULL: none; read for these alone) uneven segments (all-gather
   and reduce-scatter: a rank's count each; all-to-all: sendcounts then recvcounts), `type` (a WRITE's
   `count` of it is the WRITE's bytes, all a WRITE moves: its values' type is its producer's and consumer's), `op` (a
   reduction's), and `group`: consecutive declarations of one number form one prepared group.
   ncclMeshPlanRun supplies their invocation order. Numerical collectives and
   WRITE calls enter the common source graph. Calls in one group share a frame count. It uses the same physical striping and routed-write
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
/* Preparation arithmetic shared by numerical and collective bindings. A
   positive instance count names cyclic storage; zero names linear storage.
   Displacement belongs to a use of the pages, independently of its byte origin. */
static inline size_t ncclMeshInvocationIndex(uint64_t invocation, int64_t displacement, size_t instances) {
  __int128 index = (__int128)invocation + displacement;
  if (instances) {
    index %= instances;
    if (index < 0) index += instances;
  }
  return (size_t)index;
}
/* A prepared operand's page-backed local view and its storage slots. Zero
   stride names one view. instances=0 uses the finite invocation domain; a
   positive instances count declares cyclic storage. An
   invocation i (ncclMeshStream invocation) binds the indexed slot * stride
   bytes past buffer.offset; linear slots use i + invocation directly. The signed
   displacement is independent of buffer.offset. A group is laid and placed for
   every slot as it is declared, and its issue names none. */
typedef struct { ncclMeshBuffer buffer; size_t stride, instances; int64_t invocation; } ncclMeshBufferSlots;
/* A semantic interval of one original logical SEND, before physical routing
   and packetization. Transactions follow source-rank order, then logical SEND
   order; a WRITE has transaction 0. first/bytes are relative to that SEND.
   Every routed fragment retains this identity, including relay fragments.
   active/expected are local page operands with ordinary occurrence slots;
   cut selects their shared fixed-width domain. before names an original call
   boundary within this declared group, not a physical command ordinal.
   Each rank supplies the same mask meaning for matched fragments, using its
   declared local pages. A rank outside the transaction's physical routes may
   leave those local pages absent. A consuming rank supplies them in its
   declaration. Semantic interval ends join physical backing cuts
   before either endpoint's DMA spans are constructed. */
typedef struct {
  uint32_t transaction, frame;
  size_t first, bytes, cut, cuts, before;
  ncclMeshBufferSlots active, expected;
} ncclMeshCoverage;
/* Exact source declarations and their selected occurrence/work relations.
   Strings are canonical serialized frontend values, not hashes or generated
   physical call names. They are retained only during preparation/reporting. */
typedef struct { const char *source, *occurrences, *work; } ncclMeshSource;
/* Native routing choices, retained as explicit preparation operands. A demand's
   call is its ordinal within the group; directions is bit 0 up, bit 1 down.
   Extents, grain (bytes per allocation quantum), units and realized bytes are
   readback, recomputed from the current calls. They are not family identity. */
/* form names the retained configured program entry, or UINT32_MAX for the
   solved tree basis. Size bounds choose an initial entry only; explicit reuse
   validates the program's collective/root/member semantics at current extents. */
typedef struct { uint32_t call, what, type, root, element_bytes, directions, laws, form;
                 uint64_t up, down, grain, units; } ncclMeshRouteDemand;
typedef struct { uint32_t demand, adjustable; double initial, share;
                 uint64_t units, bytes; } ncclMeshRouteTree;
/* Source-level association operands use the same tree columns and page-range
   allocation as native reductions. Each segment has its own share simplex.
   A zero-tree declaration seeds that simplex; preparation retains its columns
   and adds legal association neighbors. Leaves use rank IDs; binary nodes use
   nodes + local node index. No execution chooses or traverses this graph. */
typedef struct {
  uint32_t nodes,laws,segments,trees;
  const uint64_t* contributors;
  const uint32_t* segment;
  const uint32_t* parents;
  const double* share;
} ncclMeshFoldBasis;
typedef struct { uint32_t left,right,owner,round; } ncclMeshFoldNode;
typedef struct {
  ncclMeshFoldBasis basis;
  const uint64_t* first;
  const uint64_t* cells;
  const uint32_t* node_first;
  const uint32_t* result;
  const ncclMeshFoldNode* node;
} ncclMeshFoldTopology;
ncclResult_t ncclMeshFoldPrepare(const ncclMeshFoldBasis* basis,uint64_t cells,
  uint32_t element_bytes,uint64_t page,const uint64_t* counts,ncclMeshFoldTopology** out);
void ncclMeshFoldDrop(ncclMeshFoldTopology* topology);
/* kind 0 routes a WRITE/all-to-all transaction; kind 1 realizes a collective
   tree edge on physical cables. Zero weight disables a choice, not its path.
   units/bytes sum actual allocations over this group's logical transactions.
   resolution is their maximum transaction allocation-unit count. */
typedef struct { uint32_t kind, source, destination, scope; uint64_t initial, weight, units, bytes, resolution; } ncclMeshRoutePath;
typedef struct {
  uint32_t nodes, cables, calls, demands, trees, paths, hops, sources, scopes;
  const struct mesh_cable* cable;
  const ncclMeshRouteDemand* demand;
  const uint64_t* leaves;       /* demands * ceil(nodes/64) */
  const uint64_t* contributors; /* calls * ceil(nodes/64) */
  const uint64_t* recipients;   /* calls * ceil(nodes/64) */
  const ncclMeshRouteTree* tree;
  const uint32_t* parents;     /* trees * nodes; UINT32_MAX outside a tree */
  const ncclMeshRoutePath* path;
  const uint32_t* first;       /* paths + 1 offsets into links */
  const uint32_t* links;       /* hops physical cable indices */
  const ncclMeshSource* source; /* sources complete source/occurrence records */
  const uint32_t* source_first; /* calls + 1 offsets into source */
  const char* const* scope;     /* scopes exact declared choice identities */
  const uint32_t* call_scope;   /* calls indices into scope */
} ncclMeshRouting;
typedef struct {
  int what, peer;
  size_t count;
  const size_t* counts;
  ncclDataType_t type;
  ncclRedOp_t op;
  int group;
  int destination;
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
  /* Actual source and destination page bindings, including their slot layouts.
     Supply on every rank to prepare execution; {NULL,0} is a rank's absent
     operand. A declaration without bindings describes geometry for planning
     queries only. Its transfers cannot be recorded until it is bound.
     Preparation joins endpoint and relay backing cuts in common byte domains,
     across all declared slots, without splitting numerical rows. */
  const ncclMeshBufferSlots* bindings;
  /* Source and result geometry, in bindings order. NULL means contiguous.
     frames (0 or 1: one) repeats the same logical transfer; each view supplies
     its frame stride. max_packet is the declared physical packet bound. */
  const ncclMeshLayout* views;
  uint32_t frames;
  size_t max_packet;
  const ncclMeshCoverage* coverage;
  size_t coverage_count;
  /* Explicit external source publication pages. The common source pass consumes
     these as typed observations; destination effects come from transfer lowering. */
  ncclMeshBuffer source_ready;
  uint64_t ready_value;
  ncclMeshTileLayout ready_layout;
  /* A physical call can cover multiple original equations/occurrences. An
     explicit equal route_scope permits their routing choices to be shared,
     independently of physical group boundaries. NULL declares a fresh local
     scope which cannot match a prior catalog. */
  const ncclMeshSource* sources;
  uint32_t source_count;
  const char* route_scope;
  /* First function call only: the full prior basis catalog. Native preparation
     reuses compatible explicitly scoped choices before binding transfers;
     newly declared legal domains generate their initial basis once. */
  const ncclMeshRouting* routing;
  uint32_t routing_count;
} ncclMeshCall;
/* A communicator's plan (ncclMeshCommPlan): its `groups` declared groups, the groups one pass of its order issues
   (`issues`, the sum of each block's count times repeat), the number of the communicator's group the order starts at (`first`: its groups
   numbered from 1 in issue order, the first after the call), and of declared group g this rank's entries on its send
   lists and on its landing lists (entries[g][0] and [1]: the physical send and landing counts);
   and of declared group g its compiled finish (seconds[g]: from its first unit to its schedule's last unit's end, every
   page moved as soon as it is ready at its cables' rates and nothing else waited for, mesh-graph.h mesh_graph_schedule),
   and of declared call k (ncclMeshCommPlan's calls[k]) its complete source-major count matrix, nranks x nranks
   (matrices[k]): an AlltoAllV's as its declaration gave it or as the plan exchanged its ranks' counts, NULL for any
   other call. */
typedef struct {
  uint32_t groups;
  uint64_t issues, first;
  const uint64_t (*entries)[2];
  const double* seconds;
  const uint64_t* const* matrices;
  /* groups immutable routing records, valid until this plan is replaced.
     Defaults and selected weights accompany exact realized allocations. */
  const ncclMeshRouting* routing;
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
/* Local source compiler operands, installed before preparing numerical leaves.
   compiler is the CPU implementation's explicit build JSON; resources is its
   declared resource array (including [] when empty). Execution never reads
   these strings. A CPU leaf has no inferred compiler or provider defaults. */
ncclResult_t ncclMeshCommCompiler(ncclComm_t comm,const char* compiler,const char* resources);
/* Lower the next declared group occurrence before binding numerical programs.
   CommPlan owns the selected issue, page cuts, word operands and channel
   predecessor snapshots. The callback consumes typed source/landing effects;
   no backend encoder, argument table or executable is created here. */
ncclResult_t ncclMeshCommLower(ncclComm_t comm, uint64_t invocation, uint32_t polls,
    int (*effect)(void* context,const ncclMeshPageEffect* effect), void* context,
    uint64_t* occurrence);
struct isl_ctx;
/* Borrowed preparation arrays of MetalPageCommand and MetalPageTransfer.
   They use the enclosing compiler's relation context and retain local source
   order. The enclosing compiler places that order in its whole function,
   solves all page effects together, and returns one MetalPageSource per
   command in this array. The occurrence owns these objects until plan release. */
/* A transport declaration retains the packet domain produced by the common
   routing compiler. first/bytes name that domain, independently of local
   source/result offsets. spans is the fixed-width resolved span operand;
   units is its length. Direction 0 is a SEND, 1 a landing. These records and
   their ordinal identities exist only during preparation. */
typedef struct {
  size_t call,first,bytes,units,value_first;
  uint32_t frame,domain,channel,direction,transaction;
  uint64_t entry;
  ncclMeshBuffer spans;
} ncclMeshTransport;
typedef struct {
  void* commands;
  void* transfers;
  /* Exact preparation authority for solved relation reuse. definition is
     assigned by CommPlan, not derived from source text or physical addresses.
     binding contains every declared payload/coverage occurrence phase. */
  uint64_t definition;
  const uint64_t* binding;
  size_t bindings;
  const ncclMeshTransport* transport;
  size_t transports;
  /* Original command boundaries for each declared call, before coverage
     insertion. The enclosing compiler maps retained source identities here;
     descriptor-publication commands precede call_first[0]. */
  const size_t* call_first;
  size_t calls;
} ncclMeshSources;
ncclResult_t ncclMeshCommSources(ncclComm_t comm,uint64_t occurrence,
    struct isl_ctx* context,const ncclMeshSources** sources);
ncclResult_t ncclMeshCommSpecialize(ncclComm_t comm,uint64_t occurrence,void* sources);
/* Complete the source dependency solve for a lowered occurrence. Input spans
   use the same logical coordinates as PageEffect. This runs before Encode and
   extends the occurrence's immutable descriptor prerequisites. */
ncclResult_t ncclMeshCommSourceInputs(ncclComm_t comm, uint64_t occurrence,
    const ncclMeshSourceInput* inputs, size_t count);
/* Bind the explicitly selected retained command ordinals into the enclosing
   recorder. The compiler interleaves these leaves with external numerical
   source before dependency compilation. Whole occurrences supply every ordinal;
   there is no implicit whole-occurrence or alternate execution mode. */
ncclResult_t ncclMeshCommEncode(ncclComm_t comm, uint64_t occurrence,
    const size_t* commands, size_t count, cudaStream_t stream);
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
/* The shared Metal payload and page-effect primitives as built into the loaded
   library. Frontends retain their numerical source and exact page relations;
   the common source pass supplies observations and publications before binding.
   No caller receives a second executable protocol or physical binding records. */
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
/* The partials merge, a reduction operator made as ncclRedOpCreatePreMulSum makes one (an operator of comm's, freed by
   ncclRedOpDestroy): split-KV attention partials merged by their log-sum-exp [Dao et al. 2023, Flash-Decoding;
   FlashInfer's state merge].  Its operand is rows of 2 + `dim` elements of `datatype` (ncclFloat32), a row [m, l, O:
   dim] (a row's maximum score, its sum of exponentials and its unnormalized output); two rows merge to the greater m,
   with each one's l and O rescaled by e^(its m - that) and summed, a row whose l is not above 0 counting for none (two
   such merge to [the least float, 0, 0]), so the result's O / l is the attention over every member's positions and the
   result is itself a partial a later merge takes.  Every declared reducing collective takes it, with each segment count containing whole rows (ncclInvalidArgument else): it
   runs the communicator's program of that collective (its table's, by the operand's bytes) with a row for an element,
   one group, each row merged up its tree where a sum adds an element, every step's piece whole rows, of any width
   (the program's own cuts on whole rows and pages: mesh-plan.h mesh_program_steps_cut). */
ncclResult_t ncclMeshRedOpCreatePartials(ncclRedOp_t* op, size_t dim, ncclDataType_t datatype, ncclComm_t comm);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
