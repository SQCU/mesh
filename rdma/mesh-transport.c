#include "mesh-transport.h"

// ../design/algorithm-sources.md#literal-transport-primitives
void mesh_transport_send_span(struct ibv_send_wr *request,
  struct ibv_sge *span, uint64_t index, void *address,
  uint32_t bytes, uint32_t key, struct ibv_send_wr *next){
  *span=(struct ibv_sge){.addr=(uintptr_t)address,.length=bytes,.lkey=key};
  *request=(struct ibv_send_wr){.wr_id=index,.next=next,.sg_list=span,
    .num_sge=1,.opcode=IBV_WR_SEND,.send_flags=IBV_SEND_SIGNALED};
}

// ../design/algorithm-sources.md#literal-transport-primitives
void mesh_transport_receive_span(struct ibv_recv_wr *request,
  struct ibv_sge *span, uint64_t index, void *address,
  uint32_t bytes, uint32_t key, struct ibv_recv_wr *next){
  *span=(struct ibv_sge){.addr=(uintptr_t)address,.length=bytes,.lkey=key};
  *request=(struct ibv_recv_wr){.wr_id=index,.next=next,.sg_list=span,.num_sge=1};
}

// ../design/algorithm-sources.md#literal-transport-primitives
int mesh_transport_send(struct ibv_qp *pair, struct ibv_send_wr *requests,
  struct ibv_send_wr **unposted){
  return ibv_post_send(pair,requests,unposted);
}

// ../design/algorithm-sources.md#literal-transport-primitives
int mesh_transport_receive(struct ibv_qp *pair, struct ibv_recv_wr *requests,
  struct ibv_recv_wr **unposted){
  return ibv_post_recv(pair,requests,unposted);
}

// ../design/algorithm-sources.md#literal-transport-primitives
int mesh_transport_complete(struct ibv_cq *queue, int capacity,
  struct ibv_wc *completions){
  return ibv_poll_cq(queue,capacity,completions);
}
