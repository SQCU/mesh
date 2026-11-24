/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* libnccl-mesh: NCCL's API over the mesh's page tables (design/PAGE-TABLE-SPEC-20261122.md section 6).

   This is NVIDIA/nccl master 12df1a1 src/nccl.h.in (NCCL 2.32.3) with its CUDA includes removed, its version filled in,
   and only the functions libnccl-mesh.dylib exports in it (each also as its pnccl profiling name).  Every type, enum,
   config struct and initializer it has is NCCL's, member for member, so code written against nccl.h for these calls
   builds against this one.  Installing and linking it: rdma/NCCL.md.

   A rank is a process on a node, a client of its node's resident bridge (mesh-flow).  Its calls move bytes between
   ranks as pages of the node's pool, named by indices of each node's page table (mesh-call.h): a rank copies what it
   sends into source pages of its own and writes each to the indices its peers took for it; the bridges carry the page
   to every destination node and make each destination index name it; the receiving rank copies (or combines) what
   landed at its indices into recvbuff, as it landed, and invalidates them.  The bridge owns cables, lanes, frames and
   the path of every page; this library owns nothing but its indices, its pages and the copies, and checks nothing
   that lands.

   Calls.  Every call is blocking, on host memory: it returns once this rank's part is done (its pages written, every
   page due to it landed and taken into recvbuff), and its stream argument is NULL (any other is refused,
   ncclInvalidArgument).  sendbuff and recvbuff are ordinary memory; ncclMemAlloc allocates it page-aligned and
   ncclMemFree frees it.  Every rank makes the same calls on a communicator, in the same sequence, with the same
   count, type, operator and root, as NCCL requires; nothing is compared at call time, so calls that do not pair are
   the caller's error (each rank takes the bytes due to it as they landed).  The calls move pages so (B a rank's count
   of bytes):
     ncclAllGather      each rank's B to every other rank, one write a page to them all, landing at recvbuff + rank B
     ncclBroadcast      the root's B to every other rank
     ncclReduce         every other rank's B to the root, which reduces them with its own
     ncclReduceScatter  segment d of every rank's sendbuff to rank d, which reduces them with its own
     ncclAllReduce      where B is at most the single-phase threshold (36864 bytes), each rank's B to every other rank, each
                        reducing them all (one round of pages); else a reduce-scatter of the count split as evenly as the
                        ranks allow, then an all-gather of the reduced segments
     ncclAlltoAll       segment d of each rank's sendbuff to rank d, landing at recvbuff + rank B
     ncclGather         every other rank's B to the root, landing at its recvbuff + rank B
     ncclScatter        the root's segment d to rank d
   Every message is cut into 4096-byte raw flat pages (mesh-call.h MESH_PAGE_BYTES), every rank cutting it
   alike.  A reduction combines rank by rank (rank 0's, then rank 1's, ...), so every rank computes the same bits, on
   the host, for all twelve types and ncclSum, ncclProd, ncclMax, ncclMin and ncclAvg (a floating average premultiplies
   every contribution by 1 / nranks, an integer one divides the sum by nranks); the in-place forms are NCCL's.  A group
   (ncclGroupStart/End; a call outside one is a group of one) makes its calls one after another, as they were made,
   each to its end, on whatever communicators they name; every one is made whatever another's result, and the first
   failure is returned.  A communicator's calls are made by one thread at a time.  Calls on different communicators
   may come from different threads, and are moved one after the other: the process moves one call at a time, on its
   one pool of threads, and a call that waits for the one before it to end ends at its own communicator's revoke as
   any wait of a call does.  So, as NCCL asks of communicators used at once, the ranks that several communicators share
   make their calls on them in the same sequence; where one process calls on two communicators from two threads, their
   sequence is whichever call takes the pool first, so such an application sequences those calls itself, or makes them
   from one thread.  A process is one rank of a communicator (two ranks of one communicator in one process would each
   wait for the other's pages).

   Channels and threads.  Every rank cuts a call's pages into CHANNELS (8) channels alike: page j of a message, and a
   reduction's position j, are channel j % CHANNELS's.  A rank moves its channels on its process's pool of T threads,
   channel c on thread c % T, the first thread the calling thread: at its first communicator the process takes a
   client slot of its node's bridge for each thread (NCCL_MESH_THREADS where set, 1 to CHANNELS, else 4, at most half
   the node's top-level cores; fewer where no more slots are vacant) and starts a worker for each slot past the
   first.  The workers spin between calls and never sleep, from then until the process lets its attach go (once its
   bridge's region has expired and it has destroyed every communicator of it) or exits.  A call with pages in one
   channel only (a message of one page) moves on the calling thread alone.  Each thread copies its channels' pages in
   and out, and makes its channels' part of the call's own copy and of a reduction, so the copies of a large call are
   spread over the threads.

   Indices.  Each channel of a communicator has, at each rank's node, a receive ring of `depth` indices for every rank
   and SOURCES (128) source pages of each class, of its thread's slot.  Every rank computes the same depth from the
   first round of the init's reports: at most RECEIVES (512), and the most for which, at every node v of the
   communicator, its c_v ranks' source pages and a page at each index of their channels' rings, c_v CHANNELS (SOURCES +
   (nranks - 1) depth) pages, fit in half the fewer pages of the node's two pool classes (the other half its bridge's
   lanes' receives, its relays' pages and its other clients'); a depth of 0 fails the init.  An index names one page
   at a time, so that is the most of either class this communicator ever has at v, whatever its peers write ahead;
   communicators and clients that share a node share its pool.  The k-th page of channel c that rank s sends rank d on
   the communicator is written to d's index base[d][c] + s depth + k mod depth (mesh-call.h mesh_write), so it is that
   index's (k / depth + 1)-th landing: both ends count each pair's pages of each channel alike, so no exchange beyond
   the init names any index.  A page to several ranks is one write to each MESH_DESTS of them by rank, each
   destination the index its own pair's count names and each write from a source page of its own, and a source page
   is copied into again only once its uses are back to 1 (no write, SEND or landing of it under way).  A receiving
   rank reads the indices of each rank's next depth pages due and takes page k once its index's landed count reaches
   k / depth + 1: where the index names a page, the bytes due are copied from it to their place, whatever its bytes
   (a reduction page that lands before its rank's turn into the call's scratch, and combined from there rank by rank),
   and the index is invalidated, so the rank has no page of the phase it is in while it waits for another; where it
   names none (its page was taken under an earlier count), nothing is copied.  A page written depth pages or more
   ahead of the page its receiver takes there replaces that page at the index, and is taken as it.

   Communicators.  ncclCommInitRank is NCCL's bootstrap: the unique id is the root's host:port (NCCL_COMM_ID, else the
   listener ncclGetUniqueId opens, on the node's fabric address where it has one), the root listening on that host
   alone where it is an IPv6 address; the id is the communicator's name.  Making a communicator attaches the process's
   client of its node's bridge where it has none (every client slot taken, or no region of a bridge that is up, fails
   the init), takes its source pages, and gathers each rank's status, node, pool and standing listener's address over
   the bootstrap; every rank then computes the depth and takes its receive rings, and a second round gathers each
   rank's status and the first receive index of each of its channels: the only exchanges its calls need.  A rank that
   cannot go on fails every rank's init alike.  The bootstrap's waits are plain blocking calls on its sockets: an
   init's waits end when its peers answer (a rank whose root does not listen yet connects again every 100 ms), and a
   split's or a grow's exchange over comm also at comm's revoke, which shuts comm's sockets down; nothing else bounds
   them.  ncclCommSplit gathers every rank's (color, key) over comm's bootstrap and makes each color's communicator,
   sorted by key then rank, at its first rank's standing listener; ncclCommShrink makes the
   remaining ranks' communicator at the first remaining rank's standing listener, exchanging nothing over comm (with
   NCCL_SHRINK_ABORT, comm is revoked first); a rank of no color, or excluded, gets NULL.  ncclCommGrow meets at the id
   that ncclCommGetUniqueId gives comm's root, which the root gives comm's ranks over comm's bootstrap; a joining
   process passes NULL and the root's id.  Two ranks of a communicator may share a node.  ncclCommRevoke is NCCL's:
   comm's calls under way end as a value at their next look, every later call on it is refused (ncclInvalidUsage),
   and ncclCommGetAsyncError reads ncclSuccess after it; its revokeFlags is not read.  ncclCommDestroy (and
   ncclCommAbort) revokes comm and returns at once, waiting for nothing: comm is freed, its indices with it (their
   pages drop as their last uses end), there where nothing else is under way on it, else by the last of its calls
   (queued or under way), splits, shrinks and grows to end, each at its next look; comm is not named again after it.
   A region its bridge's stop released, or a bridge of another layout replaced, is expired: while the process has a
   communicator of it, a new one fails (ncclRemoteError), and once it has destroyed them all its next one attaches the
   bridge's next region.

   Failures are ncclResult_t values, ncclGetLastError naming what failed.  A call records three failures on its
   communicator: a write its node's table refuses, an index it cannot read, and no memory for the pages that land
   before their turn; ncclCommGetAsyncError reads the first one recorded, and a communicator that recorded one fails
   its later calls alike, until it is destroyed and made again.  Every other failure (an invalid argument or usage,
   other memory, the bootstrap) is the result of its call alone, and a revoked communicator's calls return
   ncclInvalidUsage while ncclCommGetAsyncError reads ncclSuccess.  A call never fails on what lands: the library
   checks nothing that lands, and each page's bytes are taken as they landed (a write may land wrong, or not at all:
   mesh-call.h).  A call's wait ends once its pages have landed, or at ncclCommRevoke or ncclCommAbort: no timer
   judges a peer lost, and a page that never lands leaves its call waiting until the application revokes or aborts
   that communicator, as with NCCL's own.

   Not implemented (not exported): ncclCommInitRankConfig and the non-blocking init, ncclCommInitAll,
   ncclCommInitRankScalable, ncclCommFinalize, ncclCommSuspend, ncclCommResume, ncclCommMemStats, ncclCommCuDevice,
   ncclCommRegister, ncclCommDeregister, ncclCommWindowRegister, ncclCommWindowDeregister, ncclWinGetUserPtr,
   ncclRedOpCreatePreMulSum, ncclRedOpDestroy, ncclBcast, ncclSend, ncclRecv, the nccl*Config collectives,
   ncclPutSignal, ncclSignal, ncclWaitSignal, ncclGroupSimulateEnd, ncclSetEncryption and the ncclParam* functions. */

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

/* The stream: NULL (every call is blocking, on host memory). */
typedef void *cudaStream_t;

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

/* NCCL's malloc and free: here ordinary host memory, aligned to the system's page (sendbuff and recvbuff may be any
   memory). */
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

/* Creates a new communicator (multi thread/process version).
 * rank must be between 0 and nranks-1 and unique within a communicator clique.
 * ncclCommInitRank implicitly syncronizes with other ranks, so it must be
 * called by different threads/processes. */
ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t pncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);

/* Frees local resources associated with communicator object. */
ncclResult_t  ncclCommDestroy(ncclComm_t comm);
ncclResult_t pncclCommDestroy(ncclComm_t comm);

/* Frees resources associated with communicator object and aborts any operations
 * that might still be running. */
ncclResult_t  ncclCommAbort(ncclComm_t comm);
ncclResult_t pncclCommAbort(ncclComm_t comm);

/* Creates one or more communicators from an existing one.
 * Ranks with the same color will end up in the same communicator.
 * Within the new communicator, key will be used to order ranks.
 * NCCL_SPLIT_NOCOLOR as color will indicate the rank will not be part of any group
 * and will therefore return a NULL communicator.
 * config is not read. */
ncclResult_t  ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config);
ncclResult_t pncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config);

/* Shrink existing communicator.
 * Ranks in excludeRanksList will be removed form the existing communicator.
 * Within the new communicator, ranks will be re-ordered to fill the gap of removed ones.
 * config is not read.
 * The flag enables NCCL to adapt to various states of the parent communicator, see NCCL_SHRINK flags.*/
ncclResult_t  ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
ncclResult_t pncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);

/* Revoke: comm's operations in flight end with an error and none starts after it (NCCL 2.29). */
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

/* Returns the user-ordered "rank" associated with the communicator. */
ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank);
ncclResult_t pncclCommUserRank(const ncclComm_t comm, int* rank);

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

/*
 * Collective communication operations
 *
 * Collective communication operations must be called separately for each
 * communicator in a communicator clique.
 *
 * Here each returns once this rank's part of it is done (stream NULL).
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

/*
 * Group semantics
 *
 * ncclGroupStart and ncclGroupEnd bracket calls that are made as one group: here
 * ncclGroupEnd makes them one after another, as they were made, each to its end.
 */

/*
 * Group Start
 *
 * Start a group call. All calls to NCCL until ncclGroupEnd will be fused into
 * a single NCCL operation.
 */
ncclResult_t  ncclGroupStart(void);
ncclResult_t pncclGroupStart(void);

/*
 * Group End
 *
 * End a group call: make the calls queued since ncclGroupStart.
 */
ncclResult_t  ncclGroupEnd(void);
ncclResult_t pncclGroupEnd(void);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
