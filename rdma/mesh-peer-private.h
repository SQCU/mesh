#ifndef MESH_PEER_PRIVATE_H
#define MESH_PEER_PRIVATE_H
#include "mesh-peer.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>

enum { MESH_PEER_QUERY=1,MESH_PEER_INSTALL };
struct mesh_peer_request {uint64_t owner,group,records;uint32_t op,peer,count,padding;};
static inline struct sockaddr_un mesh_peer_address(uint64_t pid){
  struct sockaddr_un name={.sun_family=AF_UNIX,.sun_len=sizeof(struct sockaddr_un)};
  snprintf(name.sun_path,sizeof name.sun_path,"/tmp/mesh-peer-%llu.sock",(unsigned long long)pid);
  return name;
}
#endif
