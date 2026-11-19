#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <os/os_sync_wait_on_address.h>
#include "mesh-plan.h"
/* design/transport.md */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
/* the region's name: MESH_NAME, or $MESH_REGION where set (the two-node simulator's regions, tests/sim-wire.c) */
static inline const char *mesh_name(void){ const char *name=getenv("MESH_REGION"); return name && *name?name:MESH_NAME; }
#define MESH_PORT "18519"
/* the mesh's unique local prefix, each node's fabric identity under it (bin/mesh-router-init.sh) */
#define MESH_ULA_PREFIX "fd6d:6573:68:"
#define MESH_MODE 0666
/* The wire's contract between bridges (the versions before it, this file's history; the operator, 2026-10-21: "commits
   which do not change an interface 'contract' in any way are not interface-version changes regardless of their
   hashes").  136 (design/node-addressed-transfers.md): no client names a cable, a stream or a lane; every client's
   transfers toward a node are one shared order per direction, each piece of it on the cable and queue pair a pure
   function of that order gives (mesh-order.h), and the bridges exchange ORDER only as the cables between them change.
   A bridge of another version is not paired with (port_heard) */
#define MESH_VERSION 136u
/* the region's layout (struct hdr `version`), apart from the wire's MESH_VERSION: a bridge adopts the region a bridge
   before it left where its layout is this one (design/transport.md section 2, the region outlives its bridge).
   155 replaces the clients' streams with node orders (struct mesh_node) */
#define MESH_LAYOUT 155u
#define MESH_ABSENT UINT32_MAX
/* the region's lane records (struct mesh_lanes: the bridge's, MESH_LINK_STREAMS a port), the most ports, the nodes it
   holds an order toward (struct mesh_node) and its client slots; and the bytes of the bridge's own pages, which it
   publishes for its own observation tools and no client reads (`bridge_off`, mesh-observe.h) */
#define MESH_STREAMS 32
#define MESH_LINK_STREAMS 4u
#define MESH_PORTS (MESH_STREAMS/MESH_LINK_STREAMS)
#define MESH_NODES 8u
#define MESH_CLIENTS 64
#define MESH_BRIDGE_BYTES (MESH_STREAMS*32u)
#define MESH_RETIRING (UINT64_C(1)<<63)
_Static_assert(38+25<=63,"an owner word");
/* design/transport.md: the pause between a port's hellos and between the bridge's liveness turns */
#define MESH_PAUSE_NS UINT64_C(3000000000)
/* design/transport.md, the operator's deadline (2026-10-18): no DMA or network send/receive channel of the
   mesh is used longer than this without something happening on it or its scope concluding */
#define MESH_PLATFORM_NS UINT64_C(5000000000)
/* design/data-model.md section 6: a wait's deadline, fixed when the wait is made, `ns` its caller's (0: none given,
   MESH_PLATFORM_NS) and `at` the monotonic clock's reading it ends at; `wait` names the wait */
struct mesh_deadline { const char *wait; uint64_t ns,at; };
static inline uint64_t mesh_clock_ns(void){ return clock_gettime_nsec_np(CLOCK_MONOTONIC); }
static inline struct mesh_deadline mesh_deadline(const char *wait,uint64_t ns){
  const uint64_t now=mesh_clock_ns(),length=ns?ns:MESH_PLATFORM_NS;
  return (struct mesh_deadline){wait,length,length>UINT64_MAX-now?UINT64_MAX:now+length};
}
/* the wait `wait` past its deadline of `ns`: the program ends at once, the wait and its deadline on stderr */
__attribute__((noreturn)) static inline void mesh_expired(const char *wait,uint64_t ns){
  char text[640];
  const int n=snprintf(text,sizeof text,"%s: past its deadline of %.3f s; the program ends (design/data-model.md section 6)\n",
                       wait,(double)ns/1e9);
  if(n>0)(void)!write(2,text,(size_t)n<sizeof text?(size_t)n:sizeof text-1);
  _exit(1);
}
/* the nanoseconds left before d's deadline; past it, mesh_expired */
static inline uint64_t mesh_deadline_left(const struct mesh_deadline *d){
  const uint64_t now=mesh_clock_ns();
  if(now>=d->at)mesh_expired(d->wait,d->ns);
  return d->at-now;
}
/* design/algorithm-sources.md#programtensor */
enum { MESH_ROW_OWN, MESH_ROW_HOT, MESH_FREE, MESH_PLANES };
/* design/algorithm-sources.md#programtensor; design/transport.md, #M02: `pages`, on a lease's first row, its arena
   pages (not the transport's pages, which are rows) */
struct mesh_buffer {
  _Alignas(32) _Atomic uint64_t owner;
  _Atomic uint32_t closed;
  uint32_t pages;
};
_Static_assert(sizeof(struct mesh_buffer)==32 && _Alignof(struct mesh_buffer)==32,"mesh_buffer");
/* design/algorithm-sources.md#programtensor; design/transport.md */
struct mesh_page_entry {
  _Alignas(16) _Atomic uint64_t mapping;
  _Atomic uintptr_t address;
};
_Static_assert(sizeof(struct mesh_page_entry)==16 && _Alignof(struct mesh_page_entry)==16 &&
  offsetof(struct mesh_page_entry,address)==8,"mesh_page_entry");
/* An exact byte extent of the region (offset from its base, bytes); one never crosses a registration extent */
#define MESH_DMA_EXTENT (UINT64_C(1)<<30)
struct mesh_dma_span { uint64_t offset,bytes; };
_Static_assert(sizeof(struct mesh_dma_span)==16,"mesh_dma_span ABI");
struct mesh_units { uint64_t n; };
static inline struct mesh_units mesh_units_add(struct mesh_units a,struct mesh_units b){ return (struct mesh_units){a.n+b.n}; }
static inline struct mesh_units mesh_units_sub(struct mesh_units a,struct mesh_units b){ return (struct mesh_units){a.n-b.n}; }
static inline int mesh_units_lt(struct mesh_units a,struct mesh_units b){ return a.n<b.n; }
static inline struct mesh_units mesh_units_load(const _Atomic uint64_t *word){ return (struct mesh_units){atomic_load_explicit(word,memory_order_acquire)}; }
static inline void mesh_units_store(_Atomic uint64_t *word,struct mesh_units p){ atomic_store_explicit(word,p.n,memory_order_release); }
#define MESH_UNITS_LOST ((struct mesh_units){UINT64_MAX})
/* Allocation-page counts and page-table row IDs, separate from transport units */
struct mesh_pages { uint64_t n; };
struct mesh_row { uint64_t r; };
static inline struct mesh_pages mesh_pages_add(struct mesh_pages a,struct mesh_pages b){ return (struct mesh_pages){a.n+b.n}; }
static inline struct mesh_pages mesh_pages_sub(struct mesh_pages a,struct mesh_pages b){ return (struct mesh_pages){a.n-b.n}; }
static inline struct mesh_pages mesh_pages_min(struct mesh_pages a,struct mesh_pages b){ return a.n<b.n?a:b; }
static inline int mesh_pages_lt(struct mesh_pages a,struct mesh_pages b){ return a.n<b.n; }
static inline struct mesh_row mesh_row_after(struct mesh_row row,struct mesh_pages i){ return (struct mesh_row){row.r+i.n}; }
static inline struct mesh_pages mesh_pages_load(const _Atomic uint64_t *word){ return (struct mesh_pages){atomic_load_explicit(word,memory_order_acquire)}; }
static inline void mesh_pages_store(_Atomic uint64_t *word,struct mesh_pages p){ atomic_store_explicit(word,p.n,memory_order_release); }
#define MESH_PAGES_LOST ((struct mesh_pages){UINT64_MAX})
static inline int mesh_span_overlaps(struct mesh_dma_span a,struct mesh_dma_span b){
  return a.bytes && b.bytes && (a.offset<=b.offset?b.offset-a.offset<a.bytes:a.offset-b.offset<b.bytes);
}

/* ---- Node-addressed transfers (design/node-addressed-transfers.md) ----------------------------------------------
   A client names a node and bytes.  It appends a transfer to the region's order toward that node, one ring a
   direction (0: what this node sends, 1: what it receives), shared by every client.  The order of each ring is the
   order its transfers were appended in; the transfers of the other node's ring of the other direction are, by the
   clients' shared program, the same transfers in the same order with spans of the same lengths.  The bridge cuts
   every span into pieces (mesh-order.h), and places piece n of a direction's order, of the k-th transfer, on the cable
   and queue pair the order's epoch maps (n, k) to; both ends compute it alike, so no transfer is ever announced,
   granted or claimed.

   A transfer: `units` spans at region offset `spans` (mesh_dma_span, each in the region's arena, none crossing a 1 GiB
   extent).  Its gate: a SEND of span s goes once the word at `gate` reaches base+s+1, a receive is posted once it
   reaches `base` (gate 0: at once); `gates` > 0 names instead `gates` mesh_gate {word, threshold} at `gate`, each of
   which must hold for any of it.  Once every byte of it has crossed, the word at `done` (0: none) is raised to `value`;
   transfers naming the same done word raise it in their order, so a count kept over several stays a prefix.  A lost
   transfer (a cable toward the node lost, the order restarted) raises it to UINT64_MAX.  `generation` is the order's
   (struct mesh_node) the client opened it in: a transfer of another is not in the order, and is lost at once */
struct mesh_gate { uint64_t word,threshold; };
struct mesh_crossing {
  uint64_t spans,units,gate,base,done,value;
  uint32_t gates,generation;
  uint64_t reserved;
};
_Static_assert(sizeof(struct mesh_crossing)==64,"mesh_crossing ABI");
/* A ring slot: `sequence` 2i while it is free for the order's transfer i, 2i+1 once that transfer is in it; `owner` the
   client that took it (by compare-and-swap from 0); the bridge's `pieces` (its piece count + 1, 0 until counted),
   `completed` (pieces done) and `state` (0 until its done word was raised).  A slot is free again once the bridge
   retires its transfer (struct mesh_node `retired`) */
#define MESH_RING 4096u
_Static_assert(!(MESH_RING&(MESH_RING-1)),"a ring a power of two");
struct mesh_slot {
  _Alignas(64) _Atomic uint64_t sequence;
  _Atomic uint64_t owner,pieces,completed,state;
  uint64_t padding[3];
  struct mesh_crossing t;
};
_Static_assert(sizeof(struct mesh_slot)==128,"mesh_slot ABI");
/* An epoch of a direction's order: from piece first[way] on, piece n of the k-th transfer is on cable
   (n - first) mod cables, queue pair k mod lanes of that cable's (mesh-order.h order_place).  Cables are named by the
   bridge's port and sorted by their identity, alike at both ends */
struct mesh_epoch { uint64_t first[2]; uint32_t cables,padding; struct { uint32_t port,lanes; uint64_t wire; } cable[MESH_PORTS]; };
#define MESH_EPOCHS 16u
/* The order toward node `node` (0: the slot unused): `live` while it has an epoch (clients open it only then),
   `generation` (a restart of the order: every transfer of an earlier one lost), the bandwidth of its cables, its
   rings (`ring`: a region offset, MESH_RING slots each way), each way's slots taken (`tail`, the clients') and retired
   (`retired`, the parent's, with the anchor: the transfers and pieces of the order before `retired`, under
   `anchored` odd while written, by whichever of the bridge's processes holds `retiring`), the parent's `freeze` (odd while it agrees the next epoch with the peer: no piece
   is posted meanwhile) and the epochs of the generation */
struct mesh_node {
  _Alignas(64) _Atomic uint32_t node;
  _Atomic uint32_t live;
  _Atomic uint64_t generation,bandwidth;
  uint64_t ring;
  _Atomic uint64_t freeze;
  _Atomic uint32_t epochs,retiring;
  _Alignas(64) _Atomic uint64_t tail[2];
  _Alignas(64) _Atomic uint64_t retired[2];
  _Atomic uint64_t anchored,anchor[2][2];
  struct mesh_epoch epoch[MESH_EPOCHS];
};
static inline struct mesh_slot *mesh_ring(void *m,const struct mesh_node *x,int way,uint64_t i){
  return (struct mesh_slot *)((char *)m+x->ring)+(uint64_t)way*MESH_RING+i%MESH_RING;
}

/* The bridge's lanes (never a client's): a port's lane pair s, its peer and identity while live, each lane's units
   posted and completed each way and their interval pins (`rows`, a region offset), and the port child that posts them
   (`child`, `child_start`): a unit's buffer is the hardware's until it completes or that process is gone
   (mesh-dataflow.c mesh_interval_pinned).  On a port's first record, `ack` the order's freeze its child has stopped
   for and `frontier` the pieces it had posted each way */
struct mesh_posts { _Alignas(64) _Atomic uint64_t posted[2]; _Atomic uint64_t done[2],failed_span[2]; };
_Static_assert(sizeof(struct mesh_posts)==64 && offsetof(struct mesh_posts,done)==16,"mesh_posts ABI");
#define MESH_LANE_UNITS 4096u
_Static_assert(!(MESH_LANE_UNITS&(MESH_LANE_UNITS-1)),"a ring a power of two");
struct mesh_dma_pin { _Atomic uint64_t sequence,offset,bytes; };
static inline struct mesh_dma_pin *mesh_unit_pin(void *m,uint64_t pins,int way,int lane,uint64_t u){
  return (struct mesh_dma_pin *)((char *)m+pins)+((uint64_t)(2*way+lane)*MESH_LANE_UNITS+u%MESH_LANE_UNITS);
}
static inline void mesh_pin_store(struct mesh_dma_pin *pin,uint64_t unit,struct mesh_dma_span span){
  atomic_store_explicit(&pin->sequence,2*unit+1,memory_order_seq_cst);
  atomic_store_explicit(&pin->offset,span.offset,memory_order_seq_cst);
  atomic_store_explicit(&pin->bytes,span.bytes,memory_order_seq_cst);
  atomic_store_explicit(&pin->sequence,2*unit+2,memory_order_seq_cst);
}
static inline int mesh_pin_load(const struct mesh_dma_pin *pin,uint64_t unit,struct mesh_dma_span *span){
  if(unit>=(UINT64_MAX-1)/2)return 0;
  const uint64_t sequence=atomic_load_explicit(&pin->sequence,memory_order_seq_cst);
  if(sequence!=2*unit+2)return 0;
  *span=(struct mesh_dma_span){atomic_load_explicit(&pin->offset,memory_order_seq_cst),atomic_load_explicit(&pin->bytes,memory_order_seq_cst)};
  return atomic_load_explicit(&pin->sequence,memory_order_seq_cst)==sequence;
}
struct mesh_lanes {
  _Alignas(64) _Atomic uint32_t peer;
  _Atomic uint32_t live;
  uint64_t rows;
  _Atomic uint64_t bandwidth,wire;
  _Atomic uint64_t ack,frontier[2];
  _Atomic uint32_t child,padding;
  uint64_t child_start;
  struct mesh_posts posts[2];
};
_Static_assert(sizeof(struct mesh_lanes)==256,"mesh_lanes");
/* design/transport.md: a node's record, the newest of which every bridge floods to the mesh: its node and boot, the
   record's sequence in that boot, the bytes of its bridge's region its clients may hold (the region's arena less what the
   bridge holds itself: the map, the streams' pages and rings, the duds' block), the node's memory in bytes and hardware
   model (hw.model, mesh-plan.h MESH_MODEL_BYTES with its NUL), and the `reached` nodes its bridge reaches (mesh-plan.h
   struct mesh_reach).  No host name: a name a person reads is a label the readers take from the name table by node
   number (etc/mesh-nodes.json, rdma/peers.py label; AGENTS.md "A node is its hardware; names are virtual") */
struct mesh_record { uint32_t node,reached; uint64_t boot,sequence,region,memory;
  char model[MESH_MODEL_BYTES]; struct mesh_reach reach[MESH_STREAMS]; struct mesh_health health; };
/* design/transport.md: a client slot, a process's attach (`owner`, 0 vacant; its low 32 bits the process's pid) and
   that process's start (`start`, mesh_started: with the pid, its identity, which a reused pid does not share).  The
   slot's fields, as the header's, are only ever appended to, so any later bridge or heartbeat reads a region's holders
   at these offsets (design/transport.md section 2) */
struct mesh_client {
  _Alignas(64) _Atomic uint64_t owner;
  uint64_t start,padding[6];
};
/* design/transport.md; `busy`: the transfers outstanding toward any node (0 between calls) and since when none has
   been (`idle_since`, the wall clock in ns; 0 while any is); `node_off`, the orders toward nodes (struct mesh_node) */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,streams,clients;
  uint64_t planes_off,arena_off,page_off,buffer_off,bridge_off,stream_off,client_off,data_off,length,boot;
  _Atomic uint64_t bridge_pid,serial,control,map,map_off,map_nodes,health,busy,idle_since,stopping;
  uint64_t node_off;
};
/* the header within its own first page at any page size (mesh_layout) */
_Static_assert(sizeof(struct hdr)<=4096,"struct hdr");
/* design/transport.md */
_Static_assert(sizeof(((struct hdr *)0)->control)==8 && offsetof(struct hdr,control)%8==0,"M26");
/* design/transport.md */
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");
/* design/algorithm-sources.md#meshresult; design/transport.md: a shared word (`control`, `map`) bumped and
   its waiters woken, the writes after it ordered behind it */
static inline void mesh_notify(_Atomic uint64_t *word){
  atomic_fetch_add_explicit(word,1,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* design/transport.md: the map, `map_nodes` records at `map_off` in arena pages the bridge leases for
   itself, and leases again larger as the mesh grows; the bridge notifies `map`, its serial, before and after each
   change (odd while it writes; its own record is in before the region opens, so a client never reads 0), and a
   reader's copy is whole where `map` read (acquired) before it was even and is unchanged after it */
static inline struct mesh_record *mesh_records(struct hdr *m){ return (struct mesh_record *)((char *)m+atomic_load_explicit(&m->map_off,memory_order_relaxed)); }
static inline int mesh_map_whole(struct hdr *m,uint64_t before){
  atomic_thread_fence(memory_order_acquire);
  return !(before&1) && atomic_load_explicit(&m->map,memory_order_relaxed)==before;
}
/* design/transport.md: an owner word is its generation, slot and process: (generation << 38) | (slot << 32) | pid */
static inline struct mesh_client *mesh_client_at(struct hdr *m,uint64_t owner){return (struct mesh_client *)((char *)m+m->client_off)+((owner>>32)&(MESH_CLIENTS-1));}
/* the bridge's lane records of port p's lane pair s: record p MESH_LINK_STREAMS + s */
static inline struct mesh_lanes *mesh_lanes_at(struct hdr *m,uint32_t p){return (struct mesh_lanes *)((char *)m+m->stream_off)+p;}
/* the order toward node slot n (struct mesh_node) */
static inline struct mesh_node *mesh_node_at(struct hdr *m,uint32_t n){return (struct mesh_node *)((char *)m+m->node_off)+n;}
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
/* whether `pages` pages from `row` lie in the page table: as many rows from it */
static inline int mesh_rows_hold(const struct hdr *m,struct mesh_row row,struct mesh_pages pages){
  return row.r<mesh_rows(m) && pages.n<=mesh_rows(m)-row.r;
}
/* design/transport.md; design/transport.md */
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz))*m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
/* design/transport.md */
/* the arena occupancy bitmap is indexed by arena page and sized by arena pages: it is not a row plane */
static inline _Atomic uint64_t *mesh_arena_bits(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->arena_off); }
/* whether a span lies in the region's arena, of at most UINT32_MAX bytes, crossing no registration extent */
static inline int mesh_dma_span_valid(const struct hdr *m,struct mesh_dma_span span){
  const uint64_t arena=(uint64_t)mesh_arena_pages(m)*m->pgsz;
  if(!span.bytes || span.bytes>UINT32_MAX || span.offset<m->data_off)return 0;
  const uint64_t at=span.offset-m->data_off;
  return at<arena && span.bytes<=arena-at && span.bytes<=MESH_DMA_EXTENT-at%MESH_DMA_EXTENT;
}
/* whether the region holds `bytes` at offset `at`, aligned to `align` */
static inline int mesh_region_holds(const struct hdr *m,uint64_t at,uint64_t bytes,uint64_t align){
  return at && !(at%align) && at<=m->length && bytes<=m->length-at;
}
/* whether a transfer's words and table lie in the region, and each of its spans is valid */
static inline int mesh_crossing_valid(const struct hdr *m,const struct mesh_crossing *t){
  if(!t->units || t->units>(m->length-m->data_off)/sizeof(struct mesh_dma_span) ||
     !mesh_region_holds(m,t->spans,t->units*sizeof(struct mesh_dma_span),_Alignof(struct mesh_dma_span)) || t->spans<m->data_off ||
     (t->gate && !mesh_region_holds(m,t->gate,t->gates?(uint64_t)t->gates*sizeof(struct mesh_gate):8,8)) ||
     (t->done && !mesh_region_holds(m,t->done,8,8)))return 0;
  const struct mesh_dma_span *spans=(const struct mesh_dma_span *)((const char *)m+t->spans);
  for(uint64_t i=0;i<t->units;i++)if(!mesh_dma_span_valid(m,spans[i]))return 0;
  return 1;
}
static inline struct mesh_page_entry *mesh_page(struct hdr *m){ return (struct mesh_page_entry*)((unsigned char*)m+m->page_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }

static inline uint64_t mesh_word_mask(uint32_t first,uint32_t count,uint32_t word){
  uint32_t lo=word*64,hi=lo+64,a=first>lo?first:lo,b=first+count<hi?first+count:hi;
  if(b<=a) return 0;
  uint64_t bits=b-a==64?~UINT64_C(0):((UINT64_C(1)<<(b-a))-1);
  return bits<<(a-lo);
}
static inline void mesh_bits_set(_Atomic uint64_t *p,uint32_t first,uint32_t count){
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++) atomic_fetch_or_explicit(&p[w],mesh_word_mask(first,count,w),memory_order_acq_rel);
}
static inline void mesh_bits_clear(_Atomic uint64_t *p,uint32_t first,uint32_t count){
  for(uint32_t w=first/64;count && w<=(first+count-1)/64;w++) atomic_fetch_and_explicit(&p[w],~mesh_word_mask(first,count,w),memory_order_acq_rel);
}

/* design/algorithm-sources.md#programkernel_call; design/transport.md */
/* design/transport.md; design/transport.md: the node's region: `pages` arena pages of `block`
   pages a block (the bridge's registered memory, its node's configured share of its memory), a page-table row a block (its page
   entry and buffer, and a bit in each plane), the bridge's own pages, its lane records, the orders toward nodes and
   the client slots; the map's records, the lanes' pins and the orders' rings are leases the bridge takes from the arena (mesh_records, mesh_lanes_at, mesh_node_at), which the
   bridge registers whole once */
static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t pages){
  uint32_t arena=pages/block*block,rows=arena/block;
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)rows+63)/64;
  h->pgsz=pgsz; h->block=block; h->rows=rows; h->streams=MESH_STREAMS; h->clients=MESH_CLIENTS;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->arena_off=at; at+=(((uint64_t)arena+63)/64)*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)rows*sizeof(struct mesh_page_entry); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->bridge_off=at; at+=MESH_BRIDGE_BYTES; at=(at+pgsz-1)/pgsz*pgsz;
  h->stream_off=at; at+=(uint64_t)MESH_STREAMS*sizeof(struct mesh_lanes); at=(at+pgsz-1)/pgsz*pgsz;
  h->node_off=at; at+=(uint64_t)MESH_NODES*sizeof(struct mesh_node); at=(at+pgsz-1)/pgsz*pgsz;
  h->client_off=at; at+=(uint64_t)MESH_CLIENTS*sizeof(struct mesh_client);
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)arena*pgsz;
  h->length=at; return at;
}
#endif
