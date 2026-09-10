#ifndef MESH_TRANSPORT_H
#define MESH_TRANSPORT_H
#include <infiniband/verbs.h>
#include <stdint.h>

void mesh_transport_send_span(struct ibv_send_wr *request,
  struct ibv_sge *span, uint64_t index, void *address,
  uint32_t bytes, uint32_t key, struct ibv_send_wr *next);
void mesh_transport_receive_span(struct ibv_recv_wr *request,
  struct ibv_sge *span, uint64_t index, void *address,
  uint32_t bytes, uint32_t key, struct ibv_recv_wr *next);
int mesh_transport_send(struct ibv_qp *pair, struct ibv_send_wr *requests,
  struct ibv_send_wr **unposted);
int mesh_transport_receive(struct ibv_qp *pair, struct ibv_recv_wr *requests,
  struct ibv_recv_wr **unposted);
int mesh_transport_complete(struct ibv_cq *queue, int capacity,
  struct ibv_wc *completions);
#endif
