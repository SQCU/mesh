/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* libnccl-mesh: NCCL's API over the mesh's Thunderbolt 5 RDMA bridges.

   This is NVIDIA/nccl master 12df1a1 src/nccl.h.in (NCCL 2.32.3) with its CUDA includes removed,
   its version filled in, and only the functions libnccl-mesh.dylib exports declared (each also as
   its pnccl profiling name).  Every type, enum, config struct and initializer is NCCL's, member for
   member, so code written against nccl.h compiles against this one.

   Build and link (the mesh repo's rdma/):
     make -C rdma libmesh.dylib libnccl-mesh.dylib
     cc app.c -I rdma -L rdma -lnccl-mesh -Wl,-rpath,<rdma>
   A rank is a process on a node whose bridge (mesh-flow) is up: MESH_REGION names the bridge's
   region (default /mesh0).  The collectives are planned by mesh-collective.h's one planner
   (mesh_collective_choose / _plan) over the link map, an operand whose shape is fixed and whose
   contents vary: the bridge's link table (mesh-dataflow.h, ncclMeshLinksAttach) of stated nodes and
   per directed link a stated alpha-beta cost (its beta, with the bridges' estimator on, mesh-flow -E, as
   the node it leads into estimates it) and an up as the link's node observes it (this node's
   bridge its own links, every other node's bridge its links in the reports the bridges pass on; a link
   no report names is down, whatever is stated), which the communicator takes through
   ncclMeshConfig_t (ncclCommInitRankConfig; no table, no communicator) and a split inherits.  Each call
   plans on one snapshot of it, read when its group ends, and records its epoch; plans are kept by
   (epoch, call); a call keeps its plan whatever the table does while it runs (the next call plans on the new
   contents), on the links up, or where they carry none of its algorithms on the links stated (a lost link's
   session resumes, and the call waits for it).  Nothing fails a call in time: its results are a function of
   its inputs, and how late the network or a peer is changes only when it ends.  A call fails where its
   bridge observed a failure (a peer's bridge or process exited), and a failed call revokes the communicator
   (ULFM's MPI_Comm_revoke): the calls in flight fail, its connections are closed, so each peer's calls
   with this rank fail too, and it makes no call until every rank has called ncclMeshCommAgree, which
   agrees on the first failed call and makes the connections again.  Calls run as their plan's SEND /
   REDUCE / COPY steps on the bridge's communicator sessions (mesh-net.h, NCCL's ncclNet_v12 model).
   ncclCommInitRankConfig connects this rank to every rank the table links it to, on the bridge link
   that reaches it.
   ncclCollConfig_t.algSelection names the planner's algorithms: "direct", "ring", "tree",
   "binomial" (comma-separated).  MESH_NCCL_TIMEOUT bounds making a communicator's connections and an
   agreement, in seconds (default 300), never a call.  A GPU program's waits on the network are leaky spins
   (MESH_NCCL_SPIN polls, a performance setting) followed by gates: the work after a gate is in a command buffer
   committed only once an event the library signals says the words it needs are set (the library's dispatcher
   for its own programs, the caller's ncclMeshGate_t for a kept one), as Metal ends a command buffer that waits
   past its watchdog, running kernels or waiting on an event at its start.

   Buffers are host pointers: unified memory.  The bridge's registered window is handed out as
   allocations (ncclMemAlloc), each a record of the library: its pages, the one Metal buffer over them
   (ncclMeshMemBuffer: bind it in the caller's kernels too), and the points after which its last
   writer and its readers since are done, each an event and a value.  A buffer in an allocation is
   sent and received in place; any other buffer (a numpy array, a CPU tensor, other MTLBuffer contents)
   is copied between it and window memory by the library's copy kernel, through a Metal buffer over the
   host pages that hold it (no CPU copy; the window stays the network's target, as the provider lands a
   receive only in memory registered before its queue pair was set up).  An out-of-place collective
   whose send buffer is in an allocation reads it in place, with no copy into its operand: a SEND of a
   range no earlier step of this rank's plan has written reads the send buffer, the first combine of a
   range writes the result from the send buffer and the received piece, and the ranges of the result no
   step writes (an all-gather's own segment, a broadcast root's) are copied from it after the transfers
   start; a reduce-scatter whose plan writes only this rank's segment combines straight into recvbuff.  The reductions (sum, prod,
   max, min, avg, PreMulSum, every datatype) are Metal kernels: a group's calls are one command buffer
   that waits for each received piece on the completion word the bridge stores when it lands (a
   kernel polling mapped memory, no host thread between), combines it where it landed in the window,
   and stores system-coherent; float64, and float32 or bfloat16 below 2^-101, are rounded exactly in
   software (the GPU has no float64 and flushes float32 subnormals).

   Ordering is per allocation, from its record: a group waits for the recorded points of the
   allocations it touches (of a buffer it reads, the writer; of one it writes, the writer and the
   readers) and records its completion there, nothing else.  The caller's own GPU work on an
   allocation declares its point (ncclMeshMemUse: a value its event reaches once that work is done),
   and waits for the library's through ncclMeshMemWaits.  Memory outside the window that GPU work
   writes (another MTLBuffer's contents) is ordered the same way by records of its ranges, which
   hold points but no lifetime; host memory the CPU wrote before the call needs none.  Lifetime is the same data: ncclMemFree, or
   ncclMeshMemRelease with the point of the caller's last use (a tensor's release closure), marks the
   record freed, and its pages are handed out again only once every recorded point is reached.

   The stream: cudaStream_t is a struct ncclMeshStream *.  NULL makes a call (or a group holding a
   call with a NULL stream) synchronous: its program runs on the communicator's own queue and it
   returns once it and its transfers are complete.  On a stream a call returns once enqueued; its
   program is committed to the stream's own queue, never a caller's: the work after a gate is committed
   once the gate is reached, after the call has returned, and a Metal queue holds every command buffer
   behind one that waits on an event at its start, so a caller's command buffer waiting there for the
   call would hold the call's own work until the watchdog ended it (metal-microbench
   output_data/maybe-20260930/probe/queueorder).  Metal orders the work of a queue's command buffers
   only where they bind the same buffer objects, so the records, not a queue, order a call after the
   work it reads and the caller's work after it; its
   completion is the stream's next value.  `value` is the timeline's last reserved value; a call reserves the next
   one(s).  Calls on one stream run in order; calls on different streams progress together (a send
   completes only once its peer has posted the receive, so a send and a receive that wait on each
   other's peers belong in one group or on different streams).  ncclMeshStreamQuery is
   cudaStreamQuery.  A failed call still signals its completion value (the GPU is never left waiting);
   its error is the communicator's ncclCommGetAsyncError until an agreement.  ncclMeshStreamSynchronize waits on the host
   for `value`.  A deferred stream (ncclMeshStreamDefer) keeps a call's GPU work for a command buffer
   the caller hands over (ncclMeshStreamEncodeWait) instead of committing it to the stream's queue.

   Not implemented (not exported): ncclCommRevoke, ncclCommShrink, ncclCommGetUniqueId,
   ncclCommGrow, ncclCommInitRankScalable, ncclCommSuspend, ncclCommResume, ncclCommMemStats,
   ncclCommWindowRegister, ncclCommWindowDeregister, ncclWinGetUserPtr, ncclPutSignal, ncclSignal,
   ncclWaitSignal (TB5 RDMA has no one-sided write), ncclGroupSimulateEnd, ncclSetEncryption, the
   ncclParam* functions.  ncclCommInitAll takes one device (a node has one Metal device);
   ncclRedOpCreatePreMulSum takes ncclScalarHostImmediate; ncclCommRegister of memory outside the
   window records it and it stays copied in and out by the GPU. */

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

/* The stream (above).  queue: id<MTLCommandQueue>, the stream's own; event: id<MTLSharedEvent>; value: the event's last
   reserved value; made: 1 (the queue is the stream's own); deferred: ncclMeshStreamDefer's; pending: the
   programs kept for ncclMeshStreamEncodeWait, oldest first; committed: the last value a program
   committed to the queue signals. */
struct ncclMeshStream { void *queue; void *event; uint64_t value; int made, deferred; void *pending; uint64_t committed; };
typedef struct ncclMeshStream *cudaStream_t;
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

/* Returns the cuda device number associated with the communicator.  (Here: the node's one Metal
 * device, 0.) */
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

/* The mesh's own additions (not NCCL's). */
/* The link map (above): the table of the bridge of `region` (NULL: MESH_REGION, else /mesh0), mapped
   (ncclMeshLinksAttach) until ncclMeshLinksDetach, after every communicator that references it.
   ncclMeshLinksState states a link-map file (mesh-collective.h) into it, a write between calls: its
   nodes, its links' alpha and beta; the up of the bridge's own links stays as observed.
   ncclMeshLinksRead copies one consistent snapshot: *nodes the table's nodes N (the bridge's
   configuration, mesh-flow -N), links[a * N + b] of a to b, present[v] of node v, reported[v] the
   sequence of node v's last report the table holds (0: none), the bridge's node and the epoch (each
   where not NULL; call it once with only `nodes` to size the others). */
typedef struct { float alpha, beta; uint32_t stated, up; } ncclMeshLink_t;
ncclResult_t ncclMeshLinksAttach(const char* region, void** links);
ncclResult_t ncclMeshLinksDetach(void* links);
ncclResult_t ncclMeshLinksState(void* links, const char* path);
ncclResult_t ncclMeshLinksRead(void* links, uint32_t* nodes, ncclMeshLink_t* snapshot, uint32_t* present, uint64_t* reported,
    uint32_t* node, uint64_t* epoch);
/* ncclCommInitRankConfig's and ncclCommSplit's config, recognized by base.size: `links` the table,
   nodes[r] rank r's node in it (every rank's node present and distinct, below the table's nodes), each
   read at the call; the communicator's arrays are sized then, by its ranks.  A split without it inherits
   its parent's, restricted to its ranks. */
typedef struct { ncclConfig_t base; void* links; const int* nodes; } ncclMeshConfig_t;
#define NCCL_MESH_CONFIG_INITIALIZER {                                  \
  { sizeof(ncclMeshConfig_t), NCCL_API_MAGIC, NCCL_VERSION_CODE,        \
    NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, \
    NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_PTR, NCCL_CONFIG_UNDEF_INT, \
    NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_PTR, NCCL_CONFIG_UNDEF_INT, \
    NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, \
    NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, \
    NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, \
    NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, \
    NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT }, NULL, NULL }
/* ULFM's MPI_Comm_agree after a failure: every rank calls it (a collective of its own).  Once this
   rank's calls are done, each rank's vote (its first failed call, whether its connections are whole,
   its link table's map of the communicator) is flooded over the stated links [FloodSet, Lynch 1996
   §6.2.1], again until every rank's map is the same.  *failed: the index of the first call that
   failed on any rank since the last agreement (the calls issued on the communicator counted from 1;
   0: none), *epoch: this rank's link-table epoch whose map every rank holds.  After a failure, or
   where a rank's connections are not whole, every rank closes its connections and takes the
   agreement's; the memory a failed call received into is held until the bridge has vacated its old
   connections; the async error is cleared and the count starts again.  Every rank must answer by
   MESH_NCCL_TIMEOUT (a rank that does not fails the agreement: there is no shrink). */
ncclResult_t ncclMeshCommAgree(ncclComm_t comm, uint64_t* failed, uint64_t* epoch);
/* MPI_Allgatherv and MPI_Reduce_scatter: rank r's segment is counts[r] elements (counts: nranks
   entries, the same on every rank, each at least 1), the segments packed in rank order, so rank r's
   lies after the segments of the ranks before it.  ncclMeshAllGatherV sends counts[rank] elements of
   sendbuff and leaves every segment in recvbuff (in place where sendbuff is rank's segment of
   recvbuff); ncclMeshReduceScatterV reduces sendbuff's segments (the sum of counts elements) and leaves
   rank's in recvbuff, counts[rank] elements (in place where recvbuff is rank's segment of sendbuff).
   Equal counts are ncclAllGather and ncclReduceScatter; others are the planner's cut of the operand
   into those segments (mesh_collective.segments).  The *Config forms take ncclCollConfig_t as NCCL's. */
ncclResult_t ncclMeshAllGatherV(const void* sendbuff, void* recvbuff, const size_t* counts,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclMeshAllGatherVConfig(const void* sendbuff, void* recvbuff, const size_t* counts,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
ncclResult_t ncclMeshReduceScatterV(const void* sendbuff, void* recvbuff, const size_t* counts,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclMeshReduceScatterVConfig(const void* sendbuff, void* recvbuff, const size_t* counts,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, const ncclCollConfig_t* config);
/* MPI_Alltoallv with counts the GPU wrote: rank r sends each rank q the rows of sendbuff (grouped by
   destination, in rank order: sendrows rows of `row` elements) that segment q of `counts` sums to, and
   receives each rank's rows into recvbuff packed in rank order, at most `capacity` rows.  `counts` (int64,
   sendsegs[q] entries for rank q, in a window allocation or host memory) is read once the call's
   recorded points are reached; each rank's segment for this rank (recvsegs[q] entries, recvsegs[rank] =
   sendsegs[rank]) moves with its rows in one exchange (the segment, then as many rows as it says, sent
   together), is written to `received` (host memory, in rank order), and `*arrived` set to 1 once it all
   has landed (2: the call failed first), so the caller can read the counts on the host with no wait on
   the GPU while the rows may still be on the wire; the GPU copies this rank's own rows and, where
   `landed` is not NULL, writes the same counts there (int64, for the caller's GPU work after the call).  The caller keeps `received` and `arrived` until `*arrived` is set, or has them in
   window allocations (their records then hold them to the call's end). */
ncclResult_t ncclMeshAlltoAllCounted(const void* sendbuff, size_t sendrows, const int64_t* counts, const size_t* sendsegs, void* recvbuff,
    size_t capacity, int64_t* received, int64_t* landed, const size_t* recvsegs, uint64_t* arrived, size_t row, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream);
/* A stream, its calls committed to a queue of its own (above); `queue` (an id<MTLCommandQueue>, or
   NULL) is not used; its event value 0. */
ncclResult_t ncclMeshStreamCreate(cudaStream_t* stream, void* queue);
ncclResult_t ncclMeshStreamDestroy(cudaStream_t stream);
/* Waits on the host until the stream's event reaches its value (cudaStreamSynchronize). */
ncclResult_t ncclMeshStreamSynchronize(cudaStream_t stream);
/* ncclSuccess once the stream's event has reached its value, else ncclInProgress (cudaStreamQuery). */
ncclResult_t ncclMeshStreamQuery(cudaStream_t stream);
/* A deferred stream (defer 1) keeps the program of a group whose GPU work all follows its transfers'
   start (no copy in or premultiplication before them, no combine a later send of the call reads, one
   communicator) instead of committing it: the workers start on the recorded points themselves (no gate
   program between the caller's GPU work and the network), and ncclMeshStreamEncodeWait encodes the kept
   programs, in order, into a command buffer of the caller's: its leaky spins on the completion words (the
   bridge sets them) and first combines, its gate, then the combines its spins gave up on, post-division,
   copies out and the stream's completion value, so the caller's later work follows them on its own queue
   with no other queue between.  A group that cannot defer first takes the stream's kept programs into its
   own; ncclMeshStreamSynchronize commits them to the stream's queue.  Every kept program must be encoded
   (EncodeWait or Synchronize) for the stream to complete. */
ncclResult_t ncclMeshStreamDefer(cudaStream_t stream, int defer);
/* A gate of a kept program (above), the caller's: `commandBuffer` holds the work before it, and the work after
   it goes into the command buffer returned, which must not run before `event` (an id<MTLSharedEvent>) reaches
   `value` (the library signals it once the words that work needs are set): committed only once it has (Metal
   ends a command buffer that waits past its watchdog, running kernels or waiting on an event at its start), or,
   for a recording, the same one with the recording's gate (event NULL, value its place among the recording's
   gates: a persistent call's, ncclMeshPersistentGate), at which its replay commits the work after it only once
   the gate's event says so.
   NULL: the wait inside `commandBuffer`, the caller's to keep short. */
typedef void* (*ncclMeshGate_t)(void* argument, void* commandBuffer, void* event, uint64_t value);
/* Into `commandBuffer` (an id<MTLCommandBuffer> not yet committed): the stream's kept programs, each gate by
   `gate(argument, ...)` (a gate of the stream's committed programs first where they have not run); the caller
   commits the command buffer its gate gave last. */
ncclResult_t ncclMeshStreamEncodeWait(cudaStream_t stream, void* commandBuffer, ncclMeshGate_t gate, void* argument);
/* Into `commandBuffer` (an id<MTLCommandBuffer> not yet committed, no encoder open): n copies into the
   window, bytes[i] to dst[i] (in an allocation) from the id<MTLBuffer> src[i] at offset[i], stored
   system-coherent, then `word` (in an allocation) set to `value`, system-coherent, once they are done.
   The GPU's plain stores reach the NIC only when their command buffer completes; these reach it as soon
   as the word is seen (nccl-mesh-metal.m). */
ncclResult_t ncclMeshEncodeCopies(void* commandBuffer, int n, void* const* dst, void* const* src, const size_t* offset,
    const size_t* bytes, uint64_t* word, uint64_t value);
/* Persistent calls [MPI-4's persistent collectives, MPI_Allreduce_init and MPI_Start: MPI Forum, MPI 4.0,
   2021, §6.12]: between ncclMeshPersistentBegin and ncclMeshPersistentEnd the process's calls are planned,
   placed and their GPU work kept for ncclMeshStreamEncodeWait as ever (so a recording of the caller's
   command buffers takes it: metal-microbench metal_recording.h), but no worker starts them; each must defer
   (ncclMeshStreamDefer) and be a collective of one communicator.  ncclMeshPersistentNext, before a call,
   declares its receive buffer free for the network once the recording's cut `fresh` - 1 of each iteration is passed
   (1: its leading cut; 0: not before the call) and, where
   `ready` is not NULL, its send buffer's `bytes` published by the caller's GPU work in ranges of `range` bytes:
   range r's word ready[r] (8 bytes in the window, stored system-coherent) counts up by one each iteration once
   the range is written.  ncclMeshPersistentCut then says whether the last group needs a cut of its own (1: its
   send buffer is not published in ranges, or a receive of its plan may land only once its part starts), and how
   many persistent groups were made since the last ncclMeshPersistentCut.  End
   gives the calls as one handle, `calls` of them in issue order.  ncclMeshPersistentStart runs them `count`
   times: iteration i once `event` (an id<MTLSharedEvent>) reaches value + i * stride + 1 (the caller's leading
   cut: the iteration before it done), each call's requests then posted ahead where its receives may land
   (a REDUCE step's piece, or a free or own operand's range no send of it reads, a free one once its cut is
   passed) and its isends held
   (mesh-net.h mesh_net_isend_held: announced and granted before their bytes are ready, so their first byte
   waits for no request and credit exchange); call k with a cut starts once `event` reaches value + i * stride
   + c + 1 (c: its cut's place among the cuts, from 1: the command buffer that ends with its inputs complete,
   as plain stores reach the NIC then), releasing its isends; without one, once the call before it is done,
   each isend released as the ranges it reads are published.  Its GPU work zeroes its completion words once it
   has waited for them all.  ncclMeshPersistentWait waits for the last, and gives the first failure (a failed
   call revokes the communicator, and every later one fails at its start, setting its words, so the GPU work
   waiting on them goes on); the communicator makes no other call meanwhile.  ncclMeshPersistentCounts gives
   each call's counts of its last iteration (ncclMeshGroupCounts' terms), ncclMeshPersistentFree releases
   their allocations. */
ncclResult_t ncclMeshPersistentBegin(void);
ncclResult_t ncclMeshPersistentNext(const uint64_t* ready, uint64_t range, uint64_t bytes, int fresh);
ncclResult_t ncclMeshPersistentCut(int* cut, int* groups);
ncclResult_t ncclMeshPersistentEnd(void** handle, int* calls);
ncclResult_t ncclMeshPersistentStart(void* handle, void* event, uint64_t value, uint64_t stride, uint64_t count);
/* The persistent calls' gates: `gates` of them in the recording (numbered as it played them: ncclMeshGate_t),
   gate g of iteration i opened (the work after it runs) once `event` (an id<MTLSharedEvent>) reaches
   value + i gates + g + 1, which the library signals once every completion word that work needs is set;
   ncclMeshPersistentGate before ncclMeshPersistentStart. */
ncclResult_t ncclMeshPersistentGate(void* handle, void* event, uint64_t value);
ncclResult_t ncclMeshPersistentGates(void* handle, int* gates);
ncclResult_t ncclMeshPersistentWait(void* handle);
ncclResult_t ncclMeshPersistentFree(void* handle);
/* The algorithms the planner took for this thread's last ended group, a call each in issue
   order (at most `capacity`): 0 direct, 1 ring, 2 tree, 3 binomial (mesh-collective.h MESH_*),
   -1 a point-to-point call or a one-rank communicator's local copy; `roots` the tree's root. */
ncclResult_t ncclMeshGroupPlans(int* algorithms, int* roots, int capacity, int* count);
/* The link map's epoch each of those calls planned on (0 for a point-to-point call). */
ncclResult_t ncclMeshGroupEpochs(uint64_t* epochs, int capacity, int* count);
/* What the library copied, sent and waited for, counted: bytes a library thread copied on the CPU
   (none: every copy is the GPU's), bytes its GPU programs copied (into and out of the window, within
   it), the combine, premultiply and post-divide kernels they ran, the waits of a library thread for a
   shared event (the NULL stream's end, room in the window), a part's waits for recorded points that are
   events, the bytes this rank sent and received on the network; per call, the monotonic times (ns) its
   part started (after the gate) and ended (its transfers complete) and the worker saw its last piece
   land (0: none); and the handoffs between the GPU and the host: the GPU's waits on shared events its
   program encodes, the completion words its kernels wait on (the bridge's, or another program's), a
   library thread's waits on words a program publishes (a gate, a combine before its SEND, a recorded
   word), and the command buffers the library commits; the times a communicator's worker woke from its
   condition variable to start a part (it spins for WORKER_SPIN_NS after its last part before it
   sleeps), and the Metal buffers the library made (an allocation's, where no retired one of its size
   was kept); the isends a collective posted or released once their bytes were ready, and of them those
   whose receiver had granted no chunk by then (the first byte waits for a request and credit exchange);
   the time (ns) the worker saw a persistent call's send buffer ready (its cut, or its first range). */
typedef struct {
  uint64_t cpuCopyBytes, gpuCopyBytes, gpuKernels, hostWaits, inputWaits, sentBytes, receivedBytes, startNs, endNs, arrivedNs;
  uint64_t gpuEventWaits, gpuWordWaits, hostWordWaits, commits, wakeups, buffers, sends, grantWaits, readyNs;
} ncclMeshCounts_t;
/* The process's counts as they stood K evaluations ago or earlier (ncclMeshStats: its last entry among those a
   reader now may read; the times 0; all 0 where there is none). */
ncclResult_t ncclMeshGetCounts(ncclMeshCounts_t* counts);
/* The transport's statistics, lagged (mesh.h): the bridge's ring holds, for each function evaluation (a part of a
   group, as it ends), every link's counts as they stood then (ncclMeshStatsLink_t: flow control, bytes, the
   communicator session's pairings, its heartbeats sent and heard, the longest silence it heard from its peer, its
   resumptions and the chunks sent and posted again) and the ending process's (its pid, and client: ncclMeshCounts_t's
   counts in their order, then the gates opened and the spins that gave up).  The one read: the reader's evaluation `evaluation` (0: the
   evaluations ended so far) and the entries of evaluations first..evaluation - lag still in the ring, at most
   `capacity` into `out`, `count` of them; the evaluations ended so far and the lag (the bridge's -K), where not
   NULL.  Nothing reads a fresher one: a statistic reaches a reader K evaluations after the one it describes. */
typedef struct { uint64_t sendStalls, receiveStalls, creditWaits, sends, sendBytes, receives, receiveBytes, netSends, netSendBytes, netReceives,
  netReceiveBytes, sessions, heartbeatsSent, heartbeatsHeard, silenceNs, resumes, resends, reposts; } ncclMeshStatsLink_t;
typedef struct { uint64_t evaluation, ns; uint32_t links, pid; uint64_t client[24]; ncclMeshStatsLink_t link[8]; } ncclMeshStats_t;
ncclResult_t ncclMeshStats(uint64_t evaluation, uint64_t first, ncclMeshStats_t* out, int capacity, int* count, uint64_t* evaluations, uint32_t* lag);
/* The counts of each call of this thread's last ended group, in issue order (at most `capacity`),
   as ncclMeshGroupPlans; complete once the group has (a part's wait for its gate counts on its first
   call, the NULL stream's end on the group's last). */
ncclResult_t ncclMeshGroupCounts(ncclMeshCounts_t* counts, int capacity, int* count);
/* The same counts held past the thread's next group: ncclMeshGroupTally takes a reference to this
   thread's last ended group's tallies (NULL: none), ncclMeshTallyCounts reads them as
   ncclMeshGroupCounts does, ncclMeshTallyRelease drops the reference. */
ncclResult_t ncclMeshGroupTally(void** tally);
ncclResult_t ncclMeshTallyCounts(void* tally, ncclMeshCounts_t* counts, int capacity, int* count);
ncclResult_t ncclMeshTallyRelease(void* tally);
/* A persistent handle's calls' counts of their last iteration (ncclMeshPersistentBegin), in issue order. */
ncclResult_t ncclMeshPersistentCounts(void* handle, ncclMeshCounts_t* counts, int capacity, int* count);

/* The window's allocations (above).  The Metal buffer (id<MTLBuffer>) over the allocation holding
   `ptr`, and ptr's offset in it: the library's for as long as the allocation lives. */
ncclResult_t ncclMeshMemBuffer(const void* ptr, void** buffer, size_t* offset);
/* The caller's GPU work on the allocation holding `ptr` (writing it, or reading it) is done once
   `event` (an id<MTLSharedEvent>) reaches `value`.  Outside the window, a record of the `bytes` at
   `ptr` (other MTLBuffer contents the caller's kernels write) holds the point, without a lifetime: it
   is dropped once every point on it is reached. */
ncclResult_t ncclMeshMemUse(const void* ptr, size_t bytes, void* event, uint64_t value, int write);
/* A window allocation of `size` bytes whose Metal buffer, made with `options` (MTLResourceOptions), the
   caller owns (*buffer, an id<MTLBuffer> of one reference; the library keeps none): the allocation is
   freed (ncclMeshMemRelease's rule) once that buffer is deallocated, its last reference dropped by the
   caller, a command buffer or a kept program.  It never waits for room: where the window has none, or
   the caller's buffers would hold more than three quarters of it, it fails (ncclSystemError) and the
   caller makes its memory elsewhere.  A tensor allocator's buffers, sent and received in place. */
ncclResult_t ncclMeshMemAllocBuffer(void** ptr, size_t size, uint64_t options, void** buffer);
/* The recorded points not yet reached that a use of the allocation holding `ptr` (outside the window,
   of the records over its `bytes`) waits for: the writer's, and for a writer (`write`) the readers'
   too; at most `capacity`, `count` all of them. */
ncclResult_t ncclMeshMemWaits(const void* ptr, size_t bytes, int write, void** events, uint64_t* values, int capacity, int* count);
/* The allocation at `ptr` freed (the allocation's closure) once `event` (if not NULL) reaches `value`
   and every recorded point is reached; ncclMemFree(ptr) is ncclMeshMemRelease(ptr, NULL, 0). */
ncclResult_t ncclMeshMemRelease(void* ptr, void* event, uint64_t value);
/* The allocator's records: each allocation's address (as this process maps the window) and bytes,
   whether it is freed (its pages not yet handed out again), and its recorded points: each value (the
   writer's first) and its event's value now. */
typedef struct {
  uint64_t address, bytes; int32_t freed, points; uint64_t value[9], reached[9];
} ncclMeshMemRecord_t;
ncclResult_t ncclMeshMemRecords(ncclMeshMemRecord_t* records, int capacity, int* count);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
