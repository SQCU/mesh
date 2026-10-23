/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

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
struct mesh_link_map;
struct mesh_demand;

/* rdma/NCCL.md#two-paths */
typedef void *cudaStream_t;
typedef struct { void *buffer; size_t offset; } ncclMeshBuffer;
typedef struct { uint64_t ticket; int published; } ncclMeshIssue;
typedef struct {
  void *commandBuffer;
  void *commandEncoder;
  ncclMeshIssue *issue;
  void *argumentTable;
  void *residencySet;
} ncclMeshStream;

typedef void *cudaEvent_t;

typedef struct ncclComm* ncclComm_t;
typedef struct ncclWindow_vidmem* ncclWindow_t;
#define NCCL_COMM_NULL NULL

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

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

#define NCCL_API_MAGIC 0xcafebeef

#define NCCL_WIN_DEFAULT 0x00
#define NCCL_WIN_COLL_SYMMETRIC 0x01
#define NCCL_WIN_STRICT_ORDERING 0x02

#define NCCL_WIN_GIN_ONLY 0x04

#define NCCL_WIN_CFT_COUNTED 0x08

#define NCCL_WIN_REQUIRED_ALIGNMENT 4096

#define NCCL_CTA_POLICY_DEFAULT 0x00
#define NCCL_CTA_POLICY_EFFICIENCY 0x01
#define NCCL_CTA_POLICY_ZERO 0x02

#define NCCL_SHRINK_DEFAULT 0x00
#define NCCL_SHRINK_ABORT 0x01

#define NCCL_REVOKE_DEFAULT 0x00

typedef enum {
  ncclHostCftDefault = NCCL_CONFIG_UNDEF_INT,
  ncclHostCftEnable = 1,
  ncclHostCftDisable = 2,
  ncclHostCftFallback = 3
} ncclHostCftMode_t;

typedef enum {
  ncclNvlsHostModeDefault = NCCL_CONFIG_UNDEF_INT,
  ncclNvlsHostModeEnable = 0,
  ncclNvlsHostModeDisableTransport = 1 << 0,
  ncclNvlsHostModeDisableSymmetricMultimem = 1 << 1,
  ncclNvlsHostModeDisable = INT_MAX,
} ncclNvlsHostMode_t;

typedef struct ncclConfig_v23100 {

  size_t size;
  unsigned int magic;
  unsigned int version;

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

#define NCCL_CONFIG_INITIALIZER {                                       \
  sizeof(ncclConfig_t),                                       \
  NCCL_API_MAGIC,                                            \
  NCCL_VERSION_CODE,                                       \
  NCCL_CONFIG_UNDEF_INT,                                  \
  NCCL_CONFIG_UNDEF_INT,                            \
  NCCL_CONFIG_UNDEF_INT,                                   \
  NCCL_CONFIG_UNDEF_INT,                                   \
  NCCL_CONFIG_UNDEF_PTR,                                   \
  NCCL_CONFIG_UNDEF_INT,                                \
  NCCL_CONFIG_UNDEF_INT,                              \
  NCCL_CONFIG_UNDEF_PTR,                                  \
  NCCL_CONFIG_UNDEF_INT,                             \
  NCCL_CONFIG_UNDEF_INT,                                 \
  NCCL_CONFIG_UNDEF_INT,                               \
  NCCL_CONFIG_UNDEF_INT,                                  \
  NCCL_CONFIG_UNDEF_INT,                       \
  NCCL_CONFIG_UNDEF_INT,                        \
  NCCL_CONFIG_UNDEF_INT,                            \
  NCCL_CONFIG_UNDEF_INT,                                 \
  NCCL_CONFIG_UNDEF_INT,                               \
  NCCL_CONFIG_UNDEF_INT,                       \
  NCCL_CONFIG_UNDEF_INT,                       \
  NCCL_CONFIG_UNDEF_INT,                                 \
  NCCL_CONFIG_UNDEF_INT,                              \
  NCCL_CONFIG_UNDEF_INT,                               \
  NCCL_CONFIG_UNDEF_INT,                              \
}

typedef struct ncclConfigExt {
  struct ncclConfigExt* next;
  struct {
    int vendorId;
    int optionId;
  } key;
  union {
    int i;
    const char* s;
    void* raw;
  } val;
} ncclConfigExt_t;

typedef struct ncclCollConfig_v23200 {

  size_t size;
  unsigned int magic;
  unsigned int version;

  ncclConfigExt_t* ext;

  int minCTAs;
  int maxCTAs;
  int nvlsCTAs;
  int cgaClusterSize;

  const char* algSelection;
  int forceAlgSelection;
  int CTAPolicy;

  uint64_t userProfilerTag;

  cudaEvent_t launchCompletionEvent;
} ncclCollConfig_t;

#define NCCL_COLLCONFIG_INITIALIZER {                                 \
  sizeof(ncclCollConfig_t),                                 \
  NCCL_API_MAGIC,                                          \
  NCCL_VERSION_CODE,                                     \
  NCCL_CONFIG_UNDEF_PTR,                                     \
  NCCL_CONFIG_UNDEF_INT,                                 \
  NCCL_CONFIG_UNDEF_INT,                                 \
  NCCL_CONFIG_UNDEF_INT,                                \
  NCCL_CONFIG_UNDEF_INT,                          \
  NCCL_CONFIG_UNDEF_PTR,                            \
  1,                                           \
  NCCL_CONFIG_UNDEF_INT,                               \
  0,                                             \
  NCCL_CONFIG_UNDEF_PTR,                     \
}

typedef struct ncclSimInfo_v22200 {
    size_t size;
    unsigned int magic;
    unsigned int version;
    float estimatedTime;
} ncclSimInfo_t;

#define NCCL_SIM_INFO_INITIALIZER {                                  \
  sizeof(ncclSimInfo_t),                                   \
  0x74685283,                                             \
  NCCL_VERSION_CODE,                                    \
  NCCL_UNDEF_FLOAT                               \
}

ncclResult_t  ncclMemAlloc(void** ptr, size_t size);
ncclResult_t pncclMemAlloc(void** ptr, size_t size);

ncclResult_t  ncclMemFree(void *ptr);
ncclResult_t pncclMemFree(void *ptr);

ncclResult_t  ncclGetVersion(int *version);
ncclResult_t pncclGetVersion(int *version);

ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t pncclGetUniqueId(ncclUniqueId* uniqueId);

ncclResult_t  ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);
ncclResult_t pncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);

ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t pncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);

ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
ncclResult_t pncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);

ncclResult_t  ncclCommFinalize(ncclComm_t comm);
ncclResult_t pncclCommFinalize(ncclComm_t comm);

ncclResult_t  ncclCommDestroy(ncclComm_t comm);
ncclResult_t pncclCommDestroy(ncclComm_t comm);

ncclResult_t  ncclCommAbort(ncclComm_t comm);
ncclResult_t pncclCommAbort(ncclComm_t comm);

ncclResult_t  ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config);
ncclResult_t pncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config);

ncclResult_t  ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
ncclResult_t pncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);

ncclResult_t  ncclCommRevoke(ncclComm_t comm, int revokeFlags);
ncclResult_t pncclCommRevoke(ncclComm_t comm, int revokeFlags);

ncclResult_t  ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId);
ncclResult_t pncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId);
ncclResult_t  ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config);
ncclResult_t pncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config);

const char*  ncclGetErrorString(ncclResult_t result);
const char* pncclGetErrorString(ncclResult_t result);

const char*  ncclGetLastError(ncclComm_t comm);
const char* pncclGetLastError(ncclComm_t comm);

#define ncclResetDebugInit()
#define pncclResetDebugInit()

ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t pncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);

ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t pncclCommCount(const ncclComm_t comm, int* count);

ncclResult_t  ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t pncclCommCuDevice(const ncclComm_t comm, int* device);

ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank);
ncclResult_t pncclCommUserRank(const ncclComm_t comm, int* rank);

ncclResult_t  ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t pncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);

ncclResult_t  ncclCommDeregister(const ncclComm_t comm, void* handle);
ncclResult_t pncclCommDeregister(const ncclComm_t comm, void* handle);

typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;
typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               ncclAvg        = 4,

               ncclNumOps     = 5,

               ncclMaxRedOp   = 0x7fffffff>>(32-8*sizeof(ncclRedOp_dummy_t))
             } ncclRedOp_t;

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

typedef enum {

  ncclScalarDevice = 0,

  ncclScalarHostImmediate = 1
} ncclScalarResidence_t;

ncclResult_t  ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
ncclResult_t pncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);
ncclResult_t pncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);

ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);
ncclResult_t pncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    cudaStream_t stream);

ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t  ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t  ncclGather(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclGather(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

ncclResult_t  ncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclScatter(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

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

ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t pncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t pncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t  ncclGroupStart(void);
ncclResult_t pncclGroupStart(void);

ncclResult_t  ncclGroupEnd(void);
ncclResult_t pncclGroupEnd(void);

typedef struct { const struct mesh_program* program; uint64_t below; } ncclMeshProgram;
ncclResult_t ncclMeshCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, const ncclMeshProgram* programs, int count,
                                  uint64_t small, const char* name, ncclConfig_t* config);

typedef struct {
  int collective;
  int root;
  uint64_t below;
  int whole;
  uint32_t chunks;
} ncclMeshForm;

ncclResult_t ncclMeshCommForms(ncclComm_t comm, const ncclMeshForm* forms, int count);

typedef uint32_t (*ncclMeshSolver)(const struct mesh_link_map* map, const struct mesh_demand* demands, uint32_t count, uint32_t room,
                                   uint32_t* demand, uint32_t* parent, double* share, void* context);
ncclResult_t ncclMeshCommSetSolver(ncclComm_t comm, ncclMeshSolver solver, void* context);

ncclResult_t ncclMeshComplete(uint64_t ticket, const ncclMeshStream* stream, int* published);

uint64_t ncclMeshRetired(void);

/* rdma/NCCL.md#persistent-exchanges: MPI-4.0 MPI_Neighbor_allgatherv_init, MPI_Start, MPI_Request_free */
typedef struct ncclMeshRequest* ncclMeshRequest_t;
ncclResult_t ncclMeshExchangeInit(const size_t* parts, const int* lands, ncclDataType_t datatype, ncclComm_t comm, ncclMeshRequest_t* request);
ncclResult_t ncclMeshStart(ncclMeshRequest_t request, const size_t* counts, cudaStream_t stream, uint64_t* start);
ncclResult_t ncclMeshBind(ncclMeshRequest_t request, uint64_t start, int lands, int publishes, cudaStream_t stream, uint32_t index,
                          uint32_t groups);
ncclResult_t ncclMeshRequestFree(ncclMeshRequest_t request);

const char* ncclMeshMetalPrelude(void);

ncclResult_t ncclMeshPosition(ncclComm_t comm, uint64_t* group);

ncclResult_t ncclMeshTimes(ncclComm_t comm, uint64_t first, uint64_t count, uint64_t* issued, uint64_t* ended);

ncclResult_t ncclMeshMemBuffer(const void* ptr, void** buffer, size_t* offset);

ncclResult_t ncclMeshRegionExpired(int* expired);

ncclResult_t ncclMeshAllGatherV(const void* sendbuff, void* recvbuff, const size_t* counts, ncclDataType_t datatype, ncclComm_t comm,
                                cudaStream_t stream);
ncclResult_t ncclMeshReduceScatterV(const void* sendbuff, void* recvbuff, const size_t* counts, ncclDataType_t datatype, ncclRedOp_t op,
                                    ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclMeshAlltoAllV(const void* sendbuff, const size_t* counts, void* recvbuff, ncclDataType_t datatype, ncclComm_t comm,
                               cudaStream_t stream);

ncclResult_t ncclMeshAlltoAllCounted(const void* sendbuff, size_t sendrows, const int64_t* counts, const size_t* sendsegs, void* recvbuff,
                                     size_t capacity, int64_t* received, int64_t* landed, const size_t* recvsegs, uint64_t* arrived, size_t row,
                                     ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclMeshRedOpCreatePartials(ncclRedOp_t* op, size_t dim, ncclDataType_t datatype, ncclComm_t comm);

#ifdef __cplusplus
}
#endif

#endif
