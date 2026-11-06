/* The bridge-side region: the map's records and their cables, the ports' link table and the region's layout.  Only
   bridge-side code includes this header: the bridge's files (through mesh-bridge.h), mesh-stat.c, and mesh-dataflow.c,
   the library's own internals that reduce the map to node pairs (mesh_nodes_read) and wait on the ports' data paths
   (mesh_transfers_start).  No client header includes it (mesh.h, mesh-dataflow.h, mesh-call.h, mesh-collective.h,
   mesh-metal.h), nor any client: the operator (design/page-queue-change-request.md, "Queue pairs and links belong to the
   bridge"), no "global object representing the literal connection between two physical ... computers" is "ever to be
   exposed to collective or data processing code"; the collective "sees no cable, queue pair or link identity" ("What the
   collective decides").  Each block below was moved here whole from the header its note names (reviews R1, C1 and R2,
   B2).  It also holds the one copy of the bridge's log line (mesh_say) and monotonic clock (clock_ns), which the bridge
   (mesh-bridge.h) and its heartbeat (mesh-heartbeat.c, which holds no device) share: this header includes no verbs
   header of the mesh's (mesh-verbs.h). */
#ifndef MESH_MAP_H
#define MESH_MAP_H
#include "mesh.h"
/* mesh_say's and clock_ns's system headers (mesh-bridge.h had them through mesh-verbs.h) */
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* moved from mesh-bridge.h (review R2, B2): the one copy of mesh_say, its bounds and the clock, which mesh-bridge.h and
   mesh-heartbeat.c include from here (the heartbeat's own copies removed) */
/* tag etc/bridge/layout.h: mesh_say's line, and its log's bound */
#define MESH_DIAGNOSTIC_BYTES 640
#define MESH_LOG_BYTES 4194304
/* tag rdma/mesh-observe.h mesh_say, verbatim */
/* design/transport.md section 14: a line of the node's bridge log, the launchd job's stderr (bin/mesh-user.sh
   StandardErrorPath): the UTC time, `who` and the message, in one write; the losses and failures of the bridge and its
   heartbeat, nothing on the data path's success.  A log that has reached MESH_LOG_BYTES is emptied first (the processes
   that share it append: O_APPEND), so it holds at most that and a line */
__attribute__((format(printf,2,3))) static inline void mesh_say(const char *who,const char *format,...){
  char line[MESH_DIAGNOSTIC_BYTES];
  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);
  struct tm utc;gmtime_r(&t.tv_sec,&utc);
  int n=snprintf(line,sizeof line,"%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ %s: ",utc.tm_year+1900,utc.tm_mon+1,utc.tm_mday,
                 utc.tm_hour,utc.tm_min,utc.tm_sec,t.tv_nsec/1000000,who);
  va_list arguments;va_start(arguments,format);
  if(n>0 && (size_t)n<sizeof line)n+=vsnprintf(line+n,sizeof line-(size_t)n,format,arguments);
  va_end(arguments);
  if(n<=0)return;
  if((size_t)n>=sizeof line-1)n=(int)sizeof line-2;
  line[n++]='\n';
  struct stat log;
  const int flags=fcntl(2,F_GETFL);
  if(!fstat(2,&log) && S_ISREG(log.st_mode) && flags>=0){
    if(!(flags&O_APPEND))fcntl(2,F_SETFL,flags|O_APPEND);
    if(log.st_size>=MESH_LOG_BYTES)(void)!ftruncate(2,0);
  }
  (void)!write(2,line,(size_t)n);
}
/* tag rdma/mesh-link.c:48 clock_ns, verbatim but `static inline`, as a header's: the monotonic clock, ns (the tag's
   mesh.h mesh_clock_ns, c119d3e2, the same clock, which the heartbeat called by that name) */
static inline uint64_t clock_ns(void){ return clock_gettime_nsec_np(CLOCK_MONOTONIC); }

/* moved from mesh.h (review R1, C1): a port's link names its peer and device, bridge-side only (the note above) */
/* a port's link, at `link_off`: the resident bridge's MESH_PORTS of them (MESH_UNKNOWN where it has no such port) */
struct mesh_link_info { uint32_t peer; char device[32]; uint64_t bandwidth; struct mesh_port_info port; };
/* moved from mesh.h (review R1, C1): the link table's reader, bridge-side only (the note above) */
/* design/algorithm-sources.md#programcopy */
static inline struct mesh_link_info *mesh_links(struct hdr *m){return (struct mesh_link_info *)((char *)m+m->link_off);}

/* moved from mesh.h (review R1, C1): the map's records and their cables, bridge-side only (the note above); the one
   MESH_MODEL_BYTES (review R2, B2: mesh-bridge.h's copy removed) */
/* tag pre-return-20261103 etc/bridge/layout.h MESH_MODEL_BYTES */
#define MESH_MODEL_BYTES 32
/* tag pre-return-20261103 rdma/mesh-plan.h struct mesh_health, verbatim */
/* what a node's record carries of its health to every bridge: `booted`, the wall clock of the node's boot (ns since
   the epoch), and its `available` memory as its bridge last sampled it (the bytes the kernel can give without
   pressure, the measure jetsam keeps: kern.memorystatus_level percent of hw.memsize). */
struct mesh_health { uint64_t booted,available; };
/* tag pre-return-20261103 rdma/mesh-plan.h struct mesh_reach, adapted: an entry a cable, not a stream */
/* a cable of the record's node, as its pairing made it: `node` the peer node at its other end, `port` the record's
   node's port index and `peer_port` the peer's (the index each bridge gave its port), `qps` the queue pairs the
   pairing negotiated, the same at both ends, and `bandwidth` its bandwidth in bits a second.  Each of the two records
   lists the cable from its own side; a cable's id, the same at both ends, is its port at the end of the lower node
   id.  The bridges' rule reads cables (mesh-flow.c link_prepare); the collective reads only node pairs and each
   pair's capacity, the sum of its cables' bandwidth, never a port or queue pair. */
struct mesh_reach { uint32_t node,port,peer_port,qps; uint64_t bandwidth; };
_Static_assert(sizeof(struct mesh_reach)==24,"mesh_reach");
/* tag pre-return-20261103 rdma/mesh.h struct mesh_record, verbatim but `reach`: MESH_PORTS cables, not MESH_STREAMS
   streams */
/* a node's record, the newest of which every bridge floods to the mesh: its node and boot, the record's sequence in
   that boot, the bytes of its bridge's region its clients may hold (the region's arena whole: from 108 the bridge
   holds no arena page, its map the layout's area, hdr `map_off`), the node's memory in bytes and hardware model
   (hw.model, MESH_MODEL_BYTES with its NUL), and the `reached` cables of its node (struct mesh_reach).  No host name: a name a person reads is a label the readers take from the
   name table by node number (etc/mesh-nodes.json, rdma/peers.py label; AGENTS.md "A node is its hardware; names are
   virtual") */
/* 2026-11-05 (115; design/aot-compilation-change-request.md A7, "The view identifier is Transis's": "a positive
   integer.  This integer is computed based on the values of local counters, maintained by all processes.  This local
   counter is increased by a process upon each installation" (G. V. Chockler, I. Keidar and R. Vitenberg, ACM Comput.
   Surv. 33(4), 2001, §3.1.1); "A bridge's counter is the identifier of the last view it installed, 0 before its first.
   Each bridge's announcement of itself to the sequencer carries its counter"; design/compilation-coordination.md, "The
   revisions of 3f106747", VW): `view` appended, the record's bridge's counter, 0 from its start until it installs a
   view (mesh-records.c view_install).  Why: the record is the bridge's announcement of itself, flooded to every node
   and so to the sequencer (RFC 2328 s13's database, mesh-records.c record_heard).  What must stay: the record only
   appended to */
/* 2026-11-05 (116; design/aot-compilation-change-request.md A7, "The membership service is the sequencer": "To tell such
   a member from one that installed the sequencer's own view, each announcement names the process that created its
   bridge's installed view as well as its counter"; design/compilation-coordination.md, "The revisions of b5a3c6e6",
   RV): `view_boot` and `view_sequencer` appended, that process, the creating sequencer's start (hdr `boot`) and node
   id as mesh-records.c struct installed names it (A7: "A process is one start of a bridge"), 0 both until the
   record's bridge installs a view; `padding` 0, so the record holds no byte its writer leaves unset.  Why: the
   sequencer creates a view where a member's announcement names a view other than its installed one with a counter
   not less than that view's identifier (mesh-records.c view_announced).  What must stay: the record only appended
   to */
struct mesh_record { uint32_t node,reached; uint64_t boot,sequence,region,memory;
  char model[MESH_MODEL_BYTES]; struct mesh_reach reach[MESH_PORTS]; struct mesh_health health; uint64_t view;
  uint64_t view_boot; uint32_t view_sequencer,padding; };
/* the map's room: the records its area of the layout holds (hdr `map_room`, mesh_layout) */
#define MESH_MAP_ROOM 256u
/* moved from mesh.h (review R1, C1): the map's records' reader, bridge-side only (the note above); mesh_map_whole, the
   seqlock's, stays in mesh.h */
/* tag pre-return-20261103 rdma/mesh.h mesh_records, verbatim (the map: mesh.h, above mesh_map_whole) */
static inline struct mesh_record *mesh_records(struct hdr *m){ return (struct mesh_record *)((char *)m+atomic_load_explicit(&m->map_off,memory_order_relaxed)); }

/* moved from mesh-dataflow.h (review R1, C1): the records' joins, bridge-side only (the note above) */
/* tag pre-return-20261103 rdma/mesh-dataflow.h mesh_joined, verbatim (the bridge's map reads it: mesh-records.c; the
   collective reads mesh_nodes_read, which sums a pair's capacity through the same join) */
/* the one join of two nodes' records: the cables each one's bridge reaches the other over (both records list them,
   mesh_cables), 0 where none; *beta 8e9 over the least bandwidth either reports of them (0 where neither reports one) */
int mesh_joined(struct mesh_record *,struct mesh_record *,double *beta);
/* The bridge's per-cable join (mesh-flow.c link_prepare, "The rule both ends compute"), the one matching of two
   records' cables that mesh_joined and mesh_nodes_read read too: the cables a and b both list for each other, ports
   matching both ways, each `key` (the lower node id's port index, the higher's), sorted by key, `bandwidth` and `qps`
   each the lesser of the two records'; `out` holds MESH_PORTS; their count.  No client API gives a cable: hidden, so
   libmesh.dylib does not export it, and the bridge's image, built from mesh-dataflow.c, calls it */
struct mesh_cable { uint32_t key[2],qps; uint64_t bandwidth; };
__attribute__((visibility("hidden")))
uint32_t mesh_cables(const struct mesh_record *a,const struct mesh_record *b,struct mesh_cable *out /* room MESH_PORTS */);
/* 2026-11-05 (design/compilation-coordination.md, "The revisions of f3f7cf46", SW): mesh_ports_named (the start's port
   set: the ports whose peer a hop of the plan names, which mesh_transfers_start's wait read) is removed with that wait,
   its last caller.  Why: design/aot-compilation-change-request.md, appendix item 2, "The code must: ... remove the
   start's wait for every port to load the plan, and the late-loading port's refusal of it (`mesh_ports_held`).
   Nothing requires them."  What must stay: nothing of it */

/* moved from mesh.h (review R1, C1): the region's layout, which sizes the link table (struct mesh_link_info); a client
   never lays out a region, the bridge does (mesh-link.c) */
/* design/algorithm-sources.md#programkernel_call */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M08 */
/* Three independent dimensions, none of them derived from another: arena pages, page-table rows
   and the registered window.  rows was welded to pages because the
   arena bitmap lived in a row plane and the order table was rows-strided; both are now their own
   table, so a 512 GiB arena costs a 4 MiB bitmap and nothing else.  `links`: the link table's ports (MESH_PORTS,
   mesh-link.c).
   The areas, in order, each at a page boundary and owning the rest of its last page: the row planes (MESH_PLANES,
   MESH_ROW_OWN alone), the arena bitmap, the page table, the buffers, the link table, the publications, the bridge's
   pages (struct mesh_bridge_pages), the map's records (`map_room` = MESH_MAP_ROOM of struct mesh_record at
   `map_off`), the deliveries' channel (`deliveries_room` = MESH_DELIVERY_ROOM of struct mesh_delivery at
   `deliveries_off`; 113, 2026-11-05, design/compilation-coordination.md, "The revisions of 46ae8fdf", CH and M3: the
   bridge's own pages laid out once with the region, as the map's are), then the arena at a block boundary.  A plan's state (struct mesh_state: its phases, counts, supplied
   words, status and entries) is no area of the layout: mesh_plan places it in the arena with the plan's other values,
   in the plan's claim (its claim head's publication `state`). */
/* 2026-11-05 (114; design/compilation-coordination.md, "The revisions of abecd389", PQ): the deliveries' channel's area
   is gone (hdr `deliveries_off` and `deliveries_room` no longer set, kept by name), each plan's delivered calls being a
   queue of its own in its state section (mesh.h struct mesh_delivery).  Why: design/aot-compilation-change-request.md,
   appendix item 2: "there is one queue for each plan at each node of it"; "This replaces ... the one channel for every
   plan (the implementation's CH2, with M3's room)".  What must stay: no area of the layout holds a plan's words. */
/* 2026-11-04, owner L (design/compilation-coordination.md, "The region"; design/plan-to-bridge-change-request.md R8):
   the page lists' area (mesh_geometry `hops`, its room, at hdr `hop_off` with `hop_capacity`, `rows` entries where
   absent) is gone.  Why: each plan's entries are in its own state section, in its claim, and any number of plans
   stand at once, so no area of the layout holds a plan's entries.  What must stay: no area of the layout holds a
   plan's words. */
/* 2026-11-03, sub-assignment 1 (design/disjoint-pages-change-request.md, Appendix A, A3): the map's records are laid
   out once with the region (`map_off`, `map_room`), where the bridge leased arena pages for them and leased again
   larger as the mesh grew; no page of the map is taken from the arena.  What must stay: the map's room fixed with the
   region. */
/* 2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, Appendix C, C3.2, C3.4, C3.7 and C4;
   design/disjoint-pages-revision-coordination.md, "The layout"): reversing 108's areas of the supplied words
   (`word_capacity`, 8 bytes a hop) and of the landed list (`landed_capacity`, 40 bytes a hop) at hdr `word_off` and
   `landed_off`, which every plan reused in place and so zeroed behind a wait for the ports to drop the previous plan.
   Why: C4 removes those fixed areas; the allocator places every value, the landed list and the supplied words among
   them (C3.2), and a new plan's state is in new pages, never the previous plan's, so nothing erases a plan or waits
   for one to be erased (C3.4) and nothing is zeroed or reset (C3.7).  What must stay: the plan's state in its own
   section, apart from the data pages it tells of (mesh.h struct mesh_hop's note, "The state section"), placed by
   mesh_plan alone; no area of the layout holds a plan's words. */
struct mesh_geometry { uint32_t pgsz,block,pages,rows,links,wire_pages; };
static inline uint64_t mesh_layout(struct hdr *h,struct mesh_geometry g){
  uint32_t blocks=g.pages/g.block,arena=blocks*g.block;
  if(!g.rows)g.rows=arena;
  if(!g.wire_pages || g.wire_pages>arena)g.wire_pages=arena;
  g.wire_pages=(g.wire_pages+g.block-1)/g.block*g.block;
  if(g.wire_pages>arena)g.wire_pages=arena;
  uint32_t pgsz=g.pgsz;
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)g.rows+63)/64;
  h->pgsz=pgsz; h->block=g.block; h->rows=g.rows; h->wire_pages=g.wire_pages;
  h->links=g.links;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->arena_off=at; at+=(((uint64_t)arena+63)/64)*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)g.rows*sizeof(struct mesh_page_entry); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)g.rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->link_off=at; at+=(uint64_t)(g.links>MESH_PORTS?g.links:MESH_PORTS)*sizeof(struct mesh_link_info); at=(at+pgsz-1)/pgsz*pgsz;
  h->target_stride=(offsetof(struct mesh_publication,targets)+(uint64_t)g.links*sizeof(struct mesh_target)+63)&~UINT64_C(63);
  h->target_off=at; at+=(uint64_t)g.rows*h->target_stride; at=(at+pgsz-1)/pgsz*pgsz;
  h->bridge_off=at; at+=sizeof(struct mesh_bridge_pages); at=(at+pgsz-1)/pgsz*pgsz;
  /* 2026-11-05 (117; design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5: "the
     serialized operand's place is a constant the region's layout names once, as the bridge's own pages are "laid out
     once with the region" (disjoint-pages A3)"; "The revisions of 4e78f6f4"): the program's serialized operand's root
     (mesh.h struct mesh_program), after the bridge's pages.  What must stay: every area before it where it was */
  h->program_off=at; at+=sizeof(struct mesh_program); at=(at+pgsz-1)/pgsz*pgsz;
  atomic_store_explicit(&h->map_off,at,memory_order_relaxed); h->map_room=MESH_MAP_ROOM;
  at+=(uint64_t)MESH_MAP_ROOM*sizeof(struct mesh_record); at=(at+pgsz-1)/pgsz*pgsz;
  uint64_t bytes=(uint64_t)g.block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)g.pages*pgsz;
  h->length=at; return at;
}
/* moved from mesh-link.c (be0e1e48), where it was static, to beside mesh_layout (review R2, B2) */
/* tag pre-return-20261103 rdma/mesh.h:290-302 mesh_layout_within, verbatim but main's mesh_layout (378dbd5f): the
   configuration's geometry `g` (its rows and window as configured, 0 still mesh_layout's default)
   laid with `pages` the most whole blocks the object's bytes hold beside the metadata, where the tag's took pgsz and
   block alone */
/* the layout that fits a region object of `bytes` (its size, fixed when it was made): its arena the most pages its
   metadata leaves, so a layout whose metadata grew or shrank takes the same object's pages (mesh-link.c region_open);
   0 where none fits */
static inline uint64_t mesh_layout_within(struct hdr *h,struct mesh_geometry g,uint64_t bytes){
  /* blocks: lo the most known to fit, hi the most that might (a layout's length grows with its arena) */
  uint64_t lo=0,hi=bytes/g.pgsz/g.block;
  if(hi>UINT32_MAX/g.block)hi=UINT32_MAX/g.block;
  while(lo<hi){
    const uint64_t mid=lo+(hi-lo+1)/2;
    g.pages=(uint32_t)(mid*g.block);
    if(mesh_layout(h,g)<=bytes)lo=mid;else hi=mid-1;
  }
  g.pages=(uint32_t)(lo*g.block);
  return lo && mesh_layout(h,g)<=bytes?h->length:0;
}
#endif
