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
   steps, nccl-mesh-metal.m's kernels) is the only code that publishes a piece, lands or combines it, and waits on it.

   Communicators.  ncclCommInitRank is NCCL's bootstrap: the unique id is the root's host:port (NCCL_COMM_ID, else the
   listener ncclGetUniqueId opens; the bindings take MASTER_ADDR:MASTER_PORT and RANK of WORLD_SIZE, as torchrun sets
   them), every rank's node (mesh_observe) meets there; the id is its name.  Every communicator reads its bridge's map
   over its members (mesh-plan.h mesh_map: the other nodes its relays, with the routing table the bridges forward
   along) each time its session opens.  ncclMeshCommInitRank (below) is for a program that passes its own programs or
   paths.  ncclCommSplit and ncclCommShrink make communicators of the ranks they take, on the bridges' map like any
   other; Shrink makes no exchange.  ncclCommGrow meets at the bootstrap of the id that ncclCommGetUniqueId gives comm's
   root, which the root broadcasts on comm.  ncclCommRevoke closes comm's session, its groups in flight lost, and ends a
   session opening (its wait for its peers' programs, or for their agreement) from any thread: it takes no lock such a
   wait holds.  Making a communicator (a lone rank's too) attaches the process's client of its node's bridge where it
   has none, so it needs the node's bridge running and fails as ncclMemAlloc does (Memory, below): every client slot
   held, no region of a bridge that runs, or a lease of a bridge that stopped still held.

   Programs.  A communicator's programs are made from the map its session read (mesh-plan.h): a program's own table
   (ncclMeshCommInitRank), or the default, chosen for each collective, root and size class (the least 2^k of at least
   the call's bytes, mesh-plan.h MESH_CLASS_LEAST to MESH_CLASS_MOST) at its first call by mesh_choose: the fastest in
   the alpha-beta model over the map's links (relayed hops sharing the links they cross) of the collective along the
   shortest-path trees (or the caller's paths) and along the packed trees, an all-reduce also the single-phase one,
   each cut into 1, 2, 4, ... chunks while that is faster (the trees pipelined).  A call runs the first of its
   collective whose size bound its operand is under; ncclCollConfig_t.algSelection is not read.  The members agree
   before any group runs: each publishes its digest of what its programs are made from (its members' map, the
   planner's version, the caller's paths and programs) on every channel at the session's opening and compares its
   peers', so members that read the map on either side of a change fail the session as a value naming the
   disagreement (ncclRemoteError), never run programs of two maps, and the next group opens on the map then.  A group
   (ncclGroupStart/End; a call outside one is a group of one) walks each call's program (mesh_program_steps; a
   point-to-point call one step) on its communicator's session: its program's own slot on the process's client of its
   node's bridge (Memory, below), taken under its name at its first group that crosses a link and again after a
   failure.  A session is never closed by a change of the map: it runs on its agreed map until a link or node its
   channels' routes cross is lost (the bridges fail its flows), so a change elsewhere costs it nothing, and its members
   each open the next after their own failure.  A message to a rank the map does not link takes the first link of its
   route, and the bridges forward it along the rest.  Every position a rank receives on is checked against its role,
   so calls that do not pair (a separate send and receive in the same order on two ranks, unequal counts, different
   collectives) are ncclInvalidUsage.

   Host path: a NULL stream.  Buffers are host pointers (unified memory, e.g. an MTLBuffer's contents), copied through
   the session's slots and combined on the host; every call returns once its group has run.  Metal path: a stream that
   is an ncclMeshStream.  The group is encoded into its commandBuffer (the caller commits it), every buffer argument an
   ncclMeshBuffer (an id<MTLBuffer> and a byte offset into it: an MPS tensor's storage and offset) of any type but
   float64.  With commandEncoder, the kernels go into that open compute encoder, left open with nothing signalled at the
   group's end; workspace then gives the group its scratch (an id<MTLBuffer> of at least `bytes`, valid until the group
   is complete), and the caller orders any later host-path group after the stream's work.  With issue, the group is
   issued and not completed (NCCL's work on its own stream, waited for later): its operands staged and every publication
   its flow control allows encoded, with no wait for a peer; issue->ticket names it (0: it completed at once),
   issue->published says whether anything was published (the caller then commits, so the peers' waits progress), and the
   issue pointer stays valid until ncclGroupEnd returns.  ncclMeshComplete completes it, with every group issued before
   it, on a later stream of the same queue; its operands are not touched until then.  A group without issue completes at
   once, with every group before it; a host-path group fails while groups are issued and not complete.  A wait is a
   kernel polling its position's completion word (rdma/mesh.metal mesh_remote_wait).  The Metal 4 form: a stream whose
   argumentTable (an id<MTL4ArgumentTable>) is set, commandEncoder then an id<MTL4ComputeCommandEncoder> and residencySet
   an id<MTLResidencySet> (commandBuffer not read): ncclMeshEncodeMetal4, below.

   Memory.  The process is one client of its node's bridge: one attach of the node's region, made at its first
   ncclMemAlloc or communicator, whose slot owns the process's leases and on which each communicator's session takes
   its program's own slot under the communicator's name (a slot is one execution's program).  ncclMemAlloc is the lease,
   the call the operator's division of the work makes of the bridge ("the bridge is there to behave like a system
   call"): whole pages of the window covering size, which the bridge registered on every device when the device came up,
   before any queue pair existed; a lease registers nothing.  The client sizes its leases ("ontologically it must be the
   responsibility of client programs to know how much allocation they need"): a lease the window cannot meet is
   ncclSystemError naming the bytes asked and the pages free, one while every client slot is held ncclSystemError naming
   mesh.h MESH_CLIENTS, and one where the process attaches no region of a bridge that runs names the requirement, a
   bridge whose configuration's window holds the client's footprint (etc/bridge/README.md; ncclRemoteError where the
   bridge stopped, else ncclSystemError); nothing falls back to other memory.  ncclMemFree returns the pages; the leases
   of a process gone are returned by its bridge, as every retired client's rows are.  ncclCommRegister takes a buffer in
   the window, its handle the buffer itself, and refuses any other (ncclInvalidArgument: lease it with ncclMemAlloc);
   ncclCommDeregister releases nothing.  A lease is an operand of the window it came from: once its bridge stops its
   pages are no longer registered, and while the process holds such a lease its next ncclMemAlloc, communicator,
   session (a communicator's next group that crosses a link) and ncclCommRegister fail with ncclRemoteError naming it,
   as ncclMeshMemLive (below) does, until the process has freed every one (ncclMemFree); it then leases from the bridge
   that runs, so a process never holds leases of two regions (design/recovery.md mandates 12 and 13).  Weights, KV and
   every buffer a peer reads are leases: "why would weights or kv be outside of the memory pinned pool? being outside of
   it means that they cannot be zero copy passed to peers."  Both paths copy every operand, a lease or not, through the
   session's rings: no libnccl-mesh collective crosses a caller's buffer without a copy yet.  The window is where a
   program that binds its own section (mesh-call.h mesh_transfer_bind) crosses without one, and ncclCommRegister refuses
   a buffer outside it.  Design: design/prepared-machine.md#M09.

   Failures are ncclResult_t values, ncclGetLastError naming what was lost; nothing resumes a failed group, and the
   group after a failure is returned opens a session again, on the map current then.  The session's progress thread is the
   one silent-link watcher: a link with positions pending that delivers none for mesh.h MESH_SILENT_NS is cancelled,
   ending every wait on it.  A group on a cancelled, silent or lost link fails with ncclRemoteError; a failure the Metal
   path finds after its group returned (calls that do not pair, too) is the communicator's asynchronous error
   (ncclCommGetAsyncError) and the next group's result.

   Not implemented (not exported): ncclCommInitRankScalable,
   ncclCommSuspend, ncclCommResume, ncclCommMemStats, ncclCommWindowRegister, ncclCommWindowDeregister,
   ncclWinGetUserPtr, ncclPutSignal, ncclSignal, ncclWaitSignal (TB5 RDMA has no one-sided write), ncclGroupSimulateEnd,
   ncclSetEncryption, the ncclParam* functions.  ncclCommInitAll takes one device and ncclCommCuDevice returns 0 (a node
   has one Metal device); ncclRedOpCreatePreMulSum takes ncclScalarHostImmediate. */

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
#include "mesh-plan.h"

/* The stream: NULL (the host path) or an ncclMeshStream (the Metal path, above). */
typedef void *cudaStream_t;
typedef struct { void *buffer; size_t offset; } ncclMeshBuffer;
typedef struct { uint64_t ticket; int published; } ncclMeshIssue;
typedef struct {
  void *commandBuffer;
  void *commandEncoder;
  void *(*workspace)(size_t bytes, void *context);
  void *context;
  ncclMeshIssue *issue;
  void *argumentTable;
  void *residencySet;
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

/* A communicator over a program's own operands: `programs` its compiled collectives (mesh-plan.h struct mesh_program,
   copied, kept for its life; a call runs the first whose collective, and a rooted one's root, is the call's and whose
   `below` its bytes are under, 0 any size), or with none (count 0) the default (above), whose single-phase all-reduce
   runs along `paths` (NULL: the map's shortest-path trees), checked against the map at each session; rank r of
   `nranks` on the bridge of node node[r]; `name` the communicator's (NULL: "nccl").  Like every communicator, it reads
   the bridges' map at each opening of its session.  Every rank passes the same nranks, paths, programs, nodes and
   name. */
typedef struct { const struct mesh_program* program; uint64_t below; } ncclMeshProgram;
ncclResult_t ncclMeshCommInitRank(ncclComm_t* comm, int nranks, int rank, const struct mesh_trees* paths,
                                  const ncclMeshProgram* programs, int count, const int* node, const char* name);
/* Every issued group through `ticket` that is not complete, completed on `stream` (its commandBuffer and
   commandEncoder; issue and workspace not read); *published (may be NULL) whether a publication was encoded.  A
   ticket already complete encodes nothing; one whose session failed returns the failure. */
ncclResult_t ncclMeshComplete(uint64_t ticket, const ncclMeshStream* stream, int* published);
/* The highest ticket through which every issued group is complete (or lost with a failed session): its
   workspace may be reused in the stream's order. */
uint64_t ncclMeshRetired(void);
/* The Metal 4 form of a crossing (a stream whose argumentTable is set; Metal path, above): an all-reduce of `count`
   elements of `type` by `op` from sendbuff to recvbuff (each an ncclMeshBuffer) on comm, encoded into the caller's
   id<MTL4ComputeCommandEncoder> `encoder` as compute dispatches between the commands before and after it, bound
   through the caller's id<MTL4ArgumentTable> `table` (at least 6 buffer bindings; no setBytes, the arguments records
   of the library's ring), on pipelines of the caller's device (the residency set's), every buffer of the library's it
   touches added to the caller's id<MTLResidencySet> `residency` (committed where one was added).  Nothing is committed
   or signalled: the caller commits its command buffer.  Its waits are bounded GPU polls (nccl-mesh-metal.h METAL_POLLS)
   on the words the bridge writes, never a spin on another queue's work; one that reaches its bound leaves its position
   in the session's words, every later dispatch of the session does nothing, and ncclCommGetAsyncError (or the
   communicator's next group) returns the failure as a value after the window.  The group completes in the encoder:
   with no workspace its scratch is the communicator's, reused in the encoder's order.  Total over any member set: comm
   is any communicator (ncclMeshCommInitRank's: the engine's MeshRankGroup), its program the one its map chooses.  The
   same as ncclAllReduce on an ncclMeshStream of {.commandEncoder = encoder, .argumentTable = table, .residencySet =
   residency}, which every other collective and group also takes. */
ncclResult_t ncclMeshEncodeMetal4(ncclComm_t comm, ncclRedOp_t op, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type,
                                  void* encoder, void* table, void* residency);
/* Every lease this process holds (ncclMemAlloc) is of its node's bridge that runs: ncclSuccess, or ncclRemoteError
   where one is of a bridge that stopped, its pages no longer registered (Memory, above).  A process that holds leases
   and makes no other call (a server between calls) reads it to know its leases are to be freed and taken again. */
ncclResult_t ncclMeshMemLive(void);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
