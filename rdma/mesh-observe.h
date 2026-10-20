/* design/transport.md: what the bridge publishes for its own observation tools (mesh-stat, the node services) in its
   own pages of the region (struct hdr `bridge_off`): a port of the node by the index the bridge gave it, its peer node
   while paired, its phase (MESH_UNKNOWN where the bridge has no such port), the code it last stopped with, its
   bandwidth and the pairings it made.  Mesh code alone includes this: no client header does, and no client reads these
   pages (design/rubrics.md, Three kinds of source) */
#ifndef MESH_OBSERVE_H
#define MESH_OBSERVE_H
#include "mesh.h"
/* a link's streams, and so the most ports: link p's streams are the region's p MESH_LINK_STREAMS to p MESH_LINK_STREAMS
   + MESH_LINK_STREAMS - 1 */
#define MESH_LINK_STREAMS 4u
#define MESH_PORTS (MESH_STREAMS/MESH_LINK_STREAMS)
_Static_assert(MESH_PORTS*MESH_LINK_STREAMS==MESH_STREAMS,"a port's streams");
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
struct mesh_link_info { _Atomic uint32_t peer,phase; _Atomic int64_t code; _Atomic uint64_t bandwidth,pairings; };
_Static_assert(sizeof(struct mesh_link_info)==32 && MESH_PORTS*sizeof(struct mesh_link_info)<=MESH_BRIDGE_BYTES,"mesh_link_info");
static inline struct mesh_link_info *mesh_links(struct hdr *m){ return (struct mesh_link_info *)((char *)m+m->bridge_off); }
#endif
