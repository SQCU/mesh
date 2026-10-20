#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
/* design/algorithm-sources.md#programtensor: a lease of `count` slots of `bytes`, its rows and arena pages claimed by
   compare-exchange (mesh-dataflow.c mesh_claim); 0, or the refusal as a value (ENOMEM, ENOSPC) */
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/transport.md section 14: a rank's report, gathered at a communicator's init through NCCL's bootstrap, of its
   bridge's node and of each port whose link has its MESH_LINK_LANES lanes made and a peer (`peer`, `port`), the lane
   pairs free there (`free`, a bit a pair: pair p is lanes MESH_LANE_FIRST + 2 p and + 2 p + 1), as its bridge publishes
   them (struct hdr `free`: mesh_lanes_free; 0, or EINVAL).  The rule every rank applies alike to the gathered reports
   (mesh_lanes_choose, a pure function), in rank order: for rank r's links toward a node another rank occupies, the
   lowest pair free in every report of either end for every cable between the two nodes (the reports do not match the
   cables port to port) that no earlier rank of r's node chose on one of those ports, into chosen[r * MESH_PORTS + k]
   for each such link k (UINT32_MAX for every other k); 0, EINVAL where a report holds more than MESH_PORTS links or a
   port past them, or ENOSPC where such a link has none (its init fails as a value; the next init reads the pairs as
   they then stand).  A rank's binding of its chosen pairs (its row of `chosen`) asked of its own bridge through its
   slot, each pair's answer awaited at most MESH_PLATFORM_NS: 0 with each lane's first unit in start[k][j] (k the link,
   j 0 the large lane, 1 the small), EINVAL, EBUSY where the bridge refused a pair (another client binds it, or its last
   binding's units are still going), ETIMEDOUT where no answer came; any refusal unbinds what was bound.  Its unbinding
   of them: each pair's want 0, no wait (the bridge sends what the binding declared and frees the pair) */
struct mesh_lane_link { uint32_t peer,port,free,padding; };
struct mesh_lane_report { uint32_t node,count; struct mesh_lane_link link[MESH_PORTS]; };
int mesh_lanes_free(struct hdr *,struct mesh_lane_report *);
int mesh_lanes_choose(const struct mesh_lane_report *reports,uint32_t ranks,uint32_t *chosen);
int mesh_lanes_bind(struct mesh_ctx *,const struct mesh_lane_report *mine,const uint32_t *chosen,uint64_t (*start)[2]);
void mesh_lanes_unbind(struct mesh_ctx *,const struct mesh_lane_report *mine,const uint32_t *chosen);
/* design/transport.md section 14: port `port`'s lane `lane` of a pair bound to this client, as the client maps it
   (mesh.h struct mesh_lane): its in and out rings, a ready word and a landed word a unit, its unit's bytes and its
   units; 0, EINVAL, EPERM where the pair is not bound to this client, or ENOLCK where the port has no rings.  A client
   supplies unit u (its sequence on the lane, from the binding's start) once its slot u mod units of the out ring holds
   the unit's bytes: ready[u mod units] := 1 + u / units, after a release fence; unit u from the peer has landed in slot
   u mod units of the in ring once landed[u mod units] reads at least 1 + u / units (~0: the lane's link lost); the
   peer's units begin at the peer's start, which the communicator's init exchanges.  A client declares the units of a
   lane it will supply, up to `declared` (absolute) by `deadline` (uptime ns, at most MESH_PLATFORM_NS away):
   mesh_lane_declare; a declared unit not supplied by then, or by the binding's end, goes as a dud (all ones).
   mesh_lane_wake rings the port's bell: its progress threads, parked after a while of nothing, look at once */
struct mesh_lane_view { unsigned char *in,*out; _Atomic uint64_t *ready,*landed; uint32_t unit,units,port,lane; };
int mesh_lane_view(struct mesh_ctx *,uint32_t port,uint32_t lane,struct mesh_lane_view *);
int mesh_lane_declare(struct mesh_ctx *,uint32_t port,uint32_t lane,uint64_t declared,uint64_t deadline);
void mesh_lane_wake(struct hdr *,uint32_t port);
#endif
