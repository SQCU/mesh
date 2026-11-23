#ifndef MESH_VIEW_H
#define MESH_VIEW_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* The fabric as this node's bridge has it (libmesh: mesh-dataflow.c, mesh-collective.c), read-only: plain C
   declarations and integer constants, which C, Objective-C and C++ include and Python binds through mesh_c_build.py.
   Each reader maps the region read-only and attaches no client.  A node is all a reader names of the mesh: never a
   link, port or queue pair. */

/* This node's bridge's node and the nodes it reaches, each with the bandwidth of its own cables toward it in bits a
   second (0: reached through relays alone); their count (at most `capacity` written; 0: the node alone), or -errno
   where the bridge is unreachable. */
struct mesh_reach { uint32_t node,padding; uint64_t bandwidth; };
int mesh_reached(struct mesh_reach *out,uint32_t capacity,uint32_t *node);
/* The bridge's region: its arena's bytes and the arena bytes no lease of the bridge has; 0, or -errno where the
   bridge is unreachable. */
int mesh_region(uint64_t *bytes,uint64_t *unleased);
/* What the bridge's lanes have queued (struct hdr `busy`: 0 while they have nothing) and the wall clock in ns at which
   they last had nothing (0 while they have any; a read that meets the bridge as it becomes busy is told busy, 1); 0,
   or -errno where the bridge is unreachable. */
int mesh_busy(uint64_t *busy,uint64_t *idle_since);

/* A link map: `nodes` nodes and `links` links of a kind.
     MESH_LINKS_MESH   every pair is linked (`link` lists only the pairs given bandwidths, if any)
     MESH_LINKS_RING   link[i][1] follows link[i][0] around one cycle through every node
     MESH_LINKS_TREE   link[i][0] is link[i][1]'s parent; the root is nobody's child
     MESH_LINKS_GRAPH  the links as given, either way
   bandwidth (NULL: none given) is link i's bandwidth either way in bits a second (mesh_map's: the least either end's
   record reports of the cable; 0 where neither reports one).  Several links listed between one pair are parallel
   cables.  `link` and `bandwidth` are the caller's storage. */
enum { MESH_LINKS_MESH, MESH_LINKS_RING, MESH_LINKS_TREE, MESH_LINKS_GRAPH };
struct mesh_link_map { uint32_t kind,nodes,links; uint32_t (*link)[2]; uint64_t *bandwidth; };
/* 0 where a map is one, else EINVAL: every link's nodes among its nodes, and none a loop; a ring one cycle through
   every node with one link into each; a tree nodes-1 links, one into each node but the root, every node reaching the
   root.  A graph need not join its nodes. */
int mesh_link_map_check(const struct mesh_link_map *);
/* The bridge's map: its nodes the records' bridge nodes ascending (node[i] map node i's bridge node; node NULL: none),
   a link a cable where two nodes' records each name the other (the fewer of the two ends' counts), each at the least
   bandwidth either end reports, the kind MESH_LINKS_MESH where a link joins every pair, else MESH_LINKS_GRAPH.  On
   entry map->nodes and map->links are the room of node and of link and bandwidth; returns the map's nodes, or -errno
   (ENOSPC: the room is short, map->nodes and map->links then the room needed; ENOMEM).  It never waits on the
   bridge. */
int mesh_map(uint32_t *node,struct mesh_link_map *map);
/* The bridge's map serial (never 0; it changes where the mesh does, not at a health sample): waits on it until it
   differs from *serial and writes it there (*serial 0: at once); 0, or ESRCH where the region expires while it is
   waited on (no bridge serves it again), else the errno of an unreachable bridge.  The wait is the caller's own and has
   no clock: a bridge that ends without a successor or a release leaves it waiting until the next bridge's first map. */
int mesh_map_wait(uint64_t *serial);

/* What a node's record carries of its health: `booted`, the wall clock of the node's boot (ns since the epoch), and its
   `available` memory as its bridge last sampled it (the bytes the kernel can give without pressure). */
struct mesh_health { uint64_t booted,available; };
/* A node's record as its bridge last sent it: its bridge's node, its bridge's region (bytes its clients may fill), the
   node's memory (hw.memsize) and model (hw.model, NUL-terminated, MESH_MODEL_BYTES with its NUL), and its health. */
#define MESH_MODEL_BYTES 32
struct mesh_node_health { uint32_t node; uint64_t region,memory; char model[MESH_MODEL_BYTES];
                          struct mesh_health health; };
/* Every node's record the bridge has (mesh_map's nodes, this node's own among them), never waiting on the bridge:
   their count (at most `capacity` written), or -errno where the bridge is unreachable. */
int mesh_nodes_health(struct mesh_node_health *out,uint32_t capacity);

#ifdef __cplusplus
}
#endif
#endif
