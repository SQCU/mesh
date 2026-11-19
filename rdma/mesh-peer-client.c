#include "mesh-peer-private.h"
#include <string.h>
#include <limits.h>
#include <sys/param.h>

static int peer_request(struct mesh_ctx *ctx,struct mesh_peer_request request,int *socket_out){
  if(!ctx || !ctx->M || !ctx->client)return EINVAL;
  int fd=socket(AF_UNIX,SOCK_STREAM,0);
  if(fd<0)return errno;
  uint64_t pid=atomic_load(&ctx->M->bridge_pid);
  struct sockaddr_un address=mesh_peer_address(pid);
  int error=connect(fd,(struct sockaddr *)&address,sizeof address)?errno:0;
  pid_t remote=0;socklen_t size=sizeof remote;
  if(!error && (getsockopt(fd,SOL_LOCAL,LOCAL_PEERPID,&remote,&size) || (uint64_t)remote!=pid))error=EPROTO;
  int one=1;setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof one);
  request.owner=ctx->client;
  const char *data=(const char *)&request;
  for(size_t off=0;!error && off<sizeof request;){
    ssize_t n=send(fd,data+off,sizeof request-off,0);
    if(n>0)off+=(size_t)n;else if(n==0 || errno!=EINTR)error=n==0?EPIPE:errno;
  }
  if(error){close(fd);return error;}
  *socket_out=fd;return 0;
}
int mesh_peer_install(struct mesh_ctx *ctx,uint32_t peer,uint64_t group,uint64_t records,uint32_t count,uint64_t *installation){
  if(!count || !installation || !peer || !ctx || !ctx->M || records<ctx->M->data_off || records%8 ||
     records>ctx->M->length || count>(ctx->M->length-records)/sizeof(struct mesh_peer_transfer))return EINVAL;
  int fd=0,error=peer_request(ctx,(struct mesh_peer_request){.op=MESH_PEER_INSTALL,.group=group,.records=records,
    .peer=peer,.count=count},&fd);
  if(!error)*installation=(uint64_t)fd+1;
  return error;
}
int mesh_peer_wait(struct mesh_ctx *ctx,uint64_t installation,uint64_t timeout_ns){
  if(!ctx || !ctx->M || !installation || installation>(uint64_t)INT_MAX+1)return EINVAL;
  int fd=(int)(installation-1),status=0;
  uint64_t limit=mesh_clock_ns()+(timeout_ns?timeout_ns:MESH_PLATFORM_NS);
  for(;;){
    ssize_t size=recv(fd,&status,sizeof status,MSG_PEEK|MSG_DONTWAIT);
    if(size==sizeof status)return status;
    if(!size)return EPIPE;
    if(size<0 && errno!=EAGAIN && errno!=EINTR)return errno;
    uint64_t now=mesh_clock_ns();
    if(now>=limit)return ETIMEDOUT;
    struct pollfd ready={.fd=fd,.events=POLLIN};
    if(poll(&ready,1,(int)MIN((limit-now)/1000000+1,INT_MAX))<0 && errno!=EINTR)return errno;
  }
}
int mesh_peer_capable(struct mesh_ctx *ctx,uint32_t peer){
  int fd=0,status=peer_request(ctx,(struct mesh_peer_request){.op=MESH_PEER_QUERY,.peer=peer},&fd);
  if(status)return status;
  status=mesh_peer_wait(ctx,(uint64_t)fd+1,MESH_PLATFORM_NS);close(fd);return status;
}
int mesh_peer_release(struct mesh_ctx *ctx,uint64_t installation){
  if(!ctx || !ctx->M || !installation || installation>(uint64_t)INT_MAX+1)return EINVAL;
  int fd=(int)(installation-1);
  shutdown(fd,SHUT_WR);
  char tail[32];
  for(;;){
    ssize_t n=recv(fd,tail,sizeof tail,0);
    if(!n)break;
    if(n<0 && errno!=EINTR){int error=errno;close(fd);return error;}
  }
  return close(fd)?errno:0;
}
