#ifndef MESH_NET_H
#define MESH_NET_H
#include <stddef.h>
#include <stdint.h>

/* NCCL's network plugin, ncclNet_v12_t (NVIDIA/nccl master, plugins/net/example/nccl/net_v12.h), served
   by the bridge's communicator sessions (mesh-flow.c): every call of the plugin with its arguments in
   its order, returning ncclResult_t's values, and `mesh_net_plugin` the table in the plugin's member
   order.  A device is one of the bridge's links (one peer); a comm is one direction of one connection;
   memory is registered as it is, in place: the region's registered window, which mesh_net_mem_alloc
   hands out, is named by the device's registered regions (a receive lands only in memory registered
   before its queue pair was set up, so nothing is registered later).  Host pointers only; one
   receive per irecv (maxRecvs 1). */
enum { MESH_NET_SUCCESS=0, MESH_NET_SYSTEM_ERROR=2, MESH_NET_INTERNAL_ERROR=3, MESH_NET_INVALID_ARGUMENT=4,
       MESH_NET_INVALID_USAGE=5, MESH_NET_REMOTE_ERROR=6, MESH_NET_IN_PROGRESS=7 };
#define MESH_NET_HANDLE_BYTES 128
#define MESH_NET_PTR_HOST 0x1
#define MESH_NET_MAX_DEVS_PER_NIC 8
#define MESH_NET_MAX_SIZE_BYTES ((size_t)1<<40)
struct mesh_net_comm_config { int trafficClass; };
struct mesh_net_vdevice { int ndevs; int devs[MESH_NET_MAX_DEVS_PER_NIC]; };
struct mesh_net_properties {
  char *name,*pciPath;
  uint64_t guid;
  int ptrSupport,regIsGlobal,forceFlush,speed,port;
  float latency;
  int maxComms,maxRecvs,netDeviceType,netDeviceVersion;
  struct mesh_net_vdevice vProps;
  size_t maxP2pBytes,maxCollBytes;
  int maxMultiRequestSize;
  int16_t railId,planeId;
};
struct mesh_net_comm_attr { int32_t maxConcurrentPeers,minConcurrentPeers,maxFlowsPerPeer,minFlowsPerPeer; };
struct mesh_net_attr { struct mesh_net_comm_attr sendCommAttr,recvCommAttr; uint32_t op,algo,proto; };
typedef void (*mesh_net_logger)(int level,unsigned long flags,const char *file,int line,const char *format,...);
typedef int (*mesh_net_profiler)(void **eHandle,int type,void *phandle,int64_t pluginId,void *extData);

/* init attaches this process to its bridge's region (MESH_REGION, else /mesh0) as a client. */
int mesh_net_init(void **ctx,uint64_t commId,struct mesh_net_comm_config *config,mesh_net_logger logFunction,mesh_net_profiler profFunction);
int mesh_net_devices(int *ndev);
int mesh_net_get_properties(int dev,struct mesh_net_properties *props);
/* A handle whose bytes on entry already hold a mesh handle's magic and a key names the listen by that
   key: peers that derive each other's handles from one commId need no bootstrap exchange. */
int mesh_net_listen(void *ctx,int dev,void *handle,void **listenComm);
int mesh_net_connect(void *ctx,int dev,void *handle,void **sendComm,void **sendDevComm);
int mesh_net_accept(void *listenComm,void **recvComm,void **recvDevComm);
int mesh_net_reg_mr(void *comm,void *data,size_t size,int type,void **mhandle);
int mesh_net_reg_mr_dma_buf(void *comm,void *data,size_t size,int type,uint64_t offset,int fd,void **mhandle);
int mesh_net_dereg_mr(void *comm,void *mhandle);
int mesh_net_isend(void *sendComm,void *data,size_t size,int tag,void *mhandle,void *phandle,void **request);
int mesh_net_irecv(void *recvComm,int n,void **data,size_t *sizes,int *tags,void **mhandles,void **phandles,void **request);
int mesh_net_iflush(void *recvComm,int n,void **data,int *sizes,void **mhandles,void **request);
int mesh_net_test(void *request,int *done,int *sizes);
int mesh_net_close_send(void *sendComm);
int mesh_net_close_recv(void *recvComm);
int mesh_net_close_listen(void *listenComm);
int mesh_net_get_device_mr(void *comm,void *mhandle,void **dptr_mhandle);
int mesh_net_irecv_consumed(void *recvComm,int n,void *request);
int mesh_net_make_vdevice(int *d,struct mesh_net_vdevice *props);
int mesh_net_finalize(void *ctx);
int mesh_net_set_net_attr(void *ctx,struct mesh_net_attr *netAttr);

struct mesh_net_v12 {
  const char *name;
  int (*init)(void **,uint64_t,struct mesh_net_comm_config *,mesh_net_logger,mesh_net_profiler);
  int (*devices)(int *);
  int (*getProperties)(int,struct mesh_net_properties *);
  int (*listen)(void *,int,void *,void **);
  int (*connect)(void *,int,void *,void **,void **);
  int (*accept)(void *,void **,void **);
  int (*regMr)(void *,void *,size_t,int,void **);
  int (*regMrDmaBuf)(void *,void *,size_t,int,uint64_t,int,void **);
  int (*deregMr)(void *,void *);
  int (*isend)(void *,void *,size_t,int,void *,void *,void **);
  int (*irecv)(void *,int,void **,size_t *,int *,void **,void **,void **);
  int (*iflush)(void *,int,void **,int *,void **,void **);
  int (*test)(void *,int *,int *);
  int (*closeSend)(void *);
  int (*closeRecv)(void *);
  int (*closeListen)(void *);
  int (*getDeviceMr)(void *,void *,void **);
  int (*irecvConsumed)(void *,int,void *);
  int (*makeVDevice)(int *,struct mesh_net_vdevice *);
  int (*finalize)(void *);
  int (*setNetAttr)(void *,struct mesh_net_attr *);
};
extern const struct mesh_net_v12 mesh_net_plugin;

/* Memory a registration takes as it is (ncclMemAlloc's place): whole blocks of the region's registered
   window, shared with the bridge, mapped in this process by init. */
int mesh_net_mem_alloc(void **pointer,size_t size);
int mesh_net_mem_free(void *pointer);
/* The errno behind this thread's last result other than success, and a comm's counts: bytes moved,
   requests completed, sends that waited for credit, requests posted. */
int mesh_net_error(void);
void mesh_net_comm_counts(void *comm,uint64_t counts[4]);
#endif
