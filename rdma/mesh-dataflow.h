#ifndef MESH_DATAFLOW_H
#define MESH_DATAFLOW_H
#include "mesh.h"
#include "mesh-plan.h"
#include <errno.h>
/* design/pages-and-functions.md#what-the-page-table-is */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client,send_off,send_bytes,inputs_off,inputs_bytes; uint32_t row_cursor,arena_cursor; int fd; };
/* design/bridge-and-ipc.md#clients: a client slot of this node's one region, its execution `name` (the same on every
   member: its programs' queue pairs pair with the members' of the same name), taken vacant by one compare-exchange
   (mesh_slot_take, which the bridge's own leases take too).  A client takes a vacant slot only: it never retires, waits
   for or names another's, and a slot whose process is gone is the bridge's to retire.  0, EBUSY as a value where none
   is vacant, ESRCH where the node's bridge is gone */
int mesh_attach(struct mesh_ctx *,const char *name);
int mesh_slot_take(struct mesh_ctx *,struct hdr *,uint64_t key);
int mesh_detach(struct mesh_ctx *);
/* design/bridge-and-ipc.md#clients: the destinations a slot bound, in order (they are bound in order) */
static inline uint32_t mesh_destinations(const struct mesh_client *client){
  uint32_t destinations=0;
  while(destinations<MESH_DESTINATIONS && client->tx[destinations].node)destinations++;
  return destinations;
}
/* design/algorithm-sources.md#programtensor */
void mesh_retire(struct hdr *,uint64_t client);
/* design/bridge-and-ipc.md#ports: the one join of two nodes' records, linked where each one's record holds a port
   paired with the other (*beta: 8e9 over the widest such port's bandwidth, MESH_BETA_NS where none is known); and the
   route between nodes `from` and `to` on the map of region `m`, mesh-plan.h mesh_map's (the lower's path up the higher's
   shortest-path tree), from `from` to `to` in a path the caller frees: its node count, 0 where none */
int mesh_joined(struct mesh_record *,struct mesh_record *,double *beta);
uint32_t mesh_route(struct hdr *m,uint32_t from,uint32_t to,uint32_t **path);

uint32_t mesh_rows_alloc(struct mesh_ctx *,uint32_t count);
/* design/prepared-machine.md#M09 */
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
void mesh_backing_bind(struct mesh_ctx *,uint32_t first,uint32_t pages,uint32_t page,uint32_t index);
/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *,uint32_t row,uint64_t address);
void mesh_rows_release(struct mesh_ctx *,uint32_t first,uint32_t count);
/* design/algorithm-sources.md#index-hand-off */
struct mesh_target *mesh_publish_bind(struct mesh_ctx *,uint32_t row,uint32_t destination);
/* design/algorithm-sources.md#programkernel_call */
uint32_t mesh_publication_prepare(struct hdr *,uint32_t row,struct prepared_publication *);
void mesh_retired_release(struct hdr *);
#endif
