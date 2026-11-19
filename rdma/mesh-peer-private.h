#ifndef MESH_PEER_PRIVATE_H
#define MESH_PEER_PRIVATE_H
#include "mesh-peer.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>

enum { MESH_PEER_QUERY=1,MESH_PEER_INSTALL };
struct mesh_peer_request {uint64_t owner,group,records;uint32_t op,peer,count,padding;};
static inline int mesh_peer_ready(const struct hdr *m,uint64_t off,uint64_t threshold,uint32_t count){
  if(!off)return 1;
  const char *base=(const char *)m;
  if(!count){
    uint64_t value=atomic_load_explicit((const _Atomic uint64_t *)(base+off),memory_order_acquire);
    return value==UINT64_MAX?-EIO:value>=threshold;
  }
  const struct mesh_peer_gate *g=(const void *)(base+off);
  int ready=1;
  for(uint32_t i=0;i<count;i++){
    uint64_t value=atomic_load_explicit((const _Atomic uint64_t *)(base+g[i].word),memory_order_acquire);
    if(value==UINT64_MAX)return -EIO;
    ready&=value>=g[i].threshold;
  }
  return ready;
}
static inline void mesh_peer_complete(struct hdr *m,uint64_t off,uint64_t value){
  _Atomic uint64_t *word=(void *)((char *)m+off);
  uint64_t old=atomic_load_explicit(word,memory_order_relaxed);
  while(old<value && !atomic_compare_exchange_weak_explicit(word,&old,value,memory_order_release,memory_order_relaxed)){}
}
static inline struct sockaddr_un mesh_peer_address(uint64_t pid){
  struct sockaddr_un name={.sun_family=AF_UNIX,.sun_len=sizeof(struct sockaddr_un)};
  snprintf(name.sun_path,sizeof name.sun_path,"/tmp/mesh-peer-%llu.sock",(unsigned long long)pid);
  return name;
}
#endif
