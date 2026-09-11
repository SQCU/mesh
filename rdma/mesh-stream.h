#ifndef MESH_STREAM_H
#define MESH_STREAM_H
#include <stdint.h>
/* design/pages-and-functions.md#streaming-tiles */
#define MESH_STREAM_CALLS 1024
#define MESH_STREAM_TILES 2048
/* Ragged issue: when a node's compute is free it launches on every tile of its input that is present and unread, in
   production order; a launch of k tiles costs overhead_ns + k*produce_ns. launch_tiles_min: the smallest launch the
   backend can execute (a compiled artifact's shape; 1 for a kernel). reduce_ns: the owner's reduce, per tile. */
struct mesh_stream_node { uint32_t launch_tiles_min; double produce_ns, overhead_ns, reduce_ns; };
struct mesh_stream {
  uint32_t tiles;           /* tiles of the value per invocation */
  uint32_t owner_tiles[2];  /* tiles whose reduction each node owns; node 0 owns the first owner_tiles[0] */
  uint32_t window;          /* blocks one direction holds posted: mesh_window_blocks() */
  uint32_t gathers_held;    /* 1: a returned tile stays landed until the invocation ends (the caller consumes returns in one hold); 0: freed on landing */
  double transfer_ns;       /* one block on the wire */
  double input_ns;          /* interval at which the node's input tiles become present, in its production order (the upstream stream: a layer's input arrives at the link rate); 0: all present at the start */
  struct mesh_stream_node node[2];
};
struct mesh_stream_plan {
  uint32_t launches[2];                     /* launches the simulation made per node: the ragged sizes are emergent */
  uint32_t order[2][MESH_STREAM_TILES];     /* production order of tiles per node: a priority, not a size */
  uint32_t peak_landed[2];                  /* landed blocks not yet consumed, maximum, per node */
  double makespan_ns;                       /* last reduced tile landed */
  double idle[2];                           /* fraction of makespan each node produces nothing */
  double idle_parts[2][4];                  /* that idle, ns, by cause: fill (before the first launch), window (a block for this node waits for credit), peer (waiting on the peer's production or the wire), drain (after this node's last launch) */
};
/* A partial tensor is a permutation of spans of geometrically increasing extent: span 1 is one tile; span i+1 is
   min(remaining, tile*floor(rho*s_i/tile)), rho = link rows/s over the slowest producer's rows/s, so span i+1 is on the
   wire while span i computes, in expectation. Fills spans[] (rows each) for a region of `rows`; returns the count, 0 if
   capacity is short. Both nodes compute the same partition from the same constants. */
uint32_t mesh_stream_spans(uint32_t rows,uint32_t tile,double rho,uint32_t *spans,uint32_t capacity);
/* The peer's streaming tile for a value of `columns` elements of `element_bytes`, consumed `granularity` rows at a time. 0 if a block holds no granule. */
uint32_t mesh_stream_tile(uint32_t block_pages,uint32_t page_bytes,uint32_t columns,uint32_t element_bytes,uint32_t granularity);
/* The production order per node of least makespan under ragged issue whose landed peak fits the window. EINVAL: geometry; ENOSPC: no candidate fits.
   A node's GPU is one timeline: a ready reduce runs before the next launch (the caller queues reduces on their own queue). */
int mesh_stream_plan(const struct mesh_stream *s,struct mesh_stream_plan *plan);
/* One invocation under a given tile order; each maximal run of the order inside one owner region is launched as the spans of
   mesh_stream_spans over that run. Fills launches, peak, makespan, idle. EDEADLK when the window blocks a needed landing. */
int mesh_stream_simulate(const struct mesh_stream *s,const uint32_t order[2][MESH_STREAM_TILES],struct mesh_stream_plan *plan);
/* Tiles in launch i of node n in the last simulation on this thread; 0 past the end. For profiles and reports. */
uint32_t mesh_stream_launch_tiles(int node,uint32_t i);
#endif
