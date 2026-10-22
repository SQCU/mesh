#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <os/os_sync_wait_on_address.h>
#include "mesh-plan.h"
/* design/transport.md */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
/* the mesh's unique local prefix, each node's fabric identity under it (bin/mesh-router-init.sh) */
#define MESH_ULA_PREFIX "fd6d:6573:68:"
#define MESH_MODE 0666
/* 129: a link's streams toward its peer node (struct mesh_stream), each held by one client at a time by its owner word
   (as rows are), so as many communicators as a link has streams use a node pair at once, the client naming the node
   alone (design/transport.md section 14; the versions before it, this file's history).  The wire's contract between
   bridges: HELLO, FAREWELL, RECORD and LINK, and each lane's units, a SEND filling a receive of a whole unit.  It
   changes only where that contract does (the operator, 2026-10-21: "commits which do not change an interface 'contract'
   in any way are not interface-version changes regardless of their hashes").  132: every unit on a stream's lanes is
   a whole page, landing in the page the receiver's plan put it in; a holding's unit u goes on its stream's lane
   u mod 2 */
#define MESH_VERSION 132u
/* the region's layout (struct hdr `version`), apart from the wire's MESH_VERSION: a bridge adopts the region a bridge
   before it left where its layout is this one (design/transport.md section 2, the region outlives its bridge) */
#define MESH_LAYOUT 134u
#define MESH_ABSENT UINT32_MAX
/* design/transport.md section 14: the region's streams (several toward each node a bridge reaches), and its client
   slots; and the bytes of the bridge's own pages, which it publishes for its own observation tools and no client reads
   (`bridge_off`, mesh-observe.h) */
#define MESH_STREAMS 32
#define MESH_CLIENTS 64
#define MESH_BRIDGE_BYTES (MESH_STREAMS*32u)
#define MESH_RETIRING (UINT64_C(1)<<63)
/* design/transport.md section 14: the entries each of a stream's two lists holds, the words of a holding's start
   (struct mesh_stream `start`), and the entries of each of its two time rings; an owner word leaves bit 63 clear (its
   generation is 25 bits, mesh-dataflow.c mesh_slot_take) */
#define MESH_STREAM_ENTRIES 4096u
#define MESH_STREAM_START 8u
#define MESH_STREAM_TIMES 16384u
_Static_assert(38+25<=63,"an owner word");
_Static_assert(!(MESH_STREAM_ENTRIES&(MESH_STREAM_ENTRIES-1)) && !(MESH_STREAM_TIMES&(MESH_STREAM_TIMES-1)),"a ring a power of two");
/* design/transport.md: the pause between a port's hellos and between the bridge's liveness turns */
#define MESH_PAUSE_NS UINT64_C(3000000000)
/* design/transport.md, the operator's deadline (2026-10-18): no DMA or network send/receive channel of the
   mesh is used longer than this without something happening on it or its scope concluding */
#define MESH_PLATFORM_NS UINT64_C(5000000000)
/* design/algorithm-sources.md#programtensor */
enum { MESH_ROW_OWN, MESH_ROW_HOT, MESH_FREE, MESH_PLANES };
/* design/algorithm-sources.md#programtensor; design/transport.md, #M02 */
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
/* design/transport.md section 14: an entry of a holding's send list or landing list: a run of the holder's own pages,
   `bytes` of them from region offset `at` (a page's first byte: a lease's), and the count that governs them: a send
   entry's byte k is ready once the word at region offset `word` (0: the holding's `ready`) reads at least `base` + k + 1;
   a landing entry is received into once the word at `word` (0: none) reads at least `base` (its pages released by
   their reader: design/pages-and-functions.md, a page is written again only after release).  `at` is the literal page
   the holder's plan put the run in, compiled once from its page table before the holding's first call, so the bridge
   reads no table on the data path (the operator, 2026-10-21: "all placements can be demanded as ahead of time known
   compiled plans which do not change dynamically during a chain of function calls"); scattered pages are several
   entries */
struct mesh_entry { uint64_t at,bytes,word,base; };
/* design/transport.md section 14: the stream toward node `peer` (0 none), `live` while its link is up, its share of the
   link's bandwidth and `wire`, its identity, the same at both its ends and no other stream's between the two nodes (0
   until it is live): the bridge's.  A holder takes `owner` from 0 to its own while `free`; the bridge then makes the
   holding's words new (every count 0) and freezes `start`, its own ends' places on the link (per lane: its SENDs and
   its receives posted so far), which it stores before `begun` (the owner it serves); the holder hands `start` to its
   peer through its communicator's bootstrap and stores the peer's in `peer_start`, then `told` (the owner), and the
   bridge serves the holding.  The holder presents its
   entries in order: entry e of a list in slot e mod MESH_STREAM_ENTRIES of `lists` (region offsets: the send list,
   then the landing list), then `sends` or `lands` (entries presented) past it, a slot reusable once `taken` (the
   bridge's, per list) is past it.  Three byte counts over the holding, each in its list's order and all of it before:
   `ready`, the holder's, of the send list produced; `sent`, the bridge's, of the send list whose SENDs have completed;
   `landed`, the bridge's, of the landing list landed; `sent` and `landed` ~0 once the link is lost.  `deadline`
   (uptime ns), the holder's: UINT64_MAX while it supplies, now where it can supply no more, after which every presented
   byte not ready goes as a dud (all ones), as at the holding's end (its owner word let go to 0).  The two time rings
   (`posted_at`, `landed_at`, region offsets) of MESH_STREAM_TIMES entries of two words: entry e's, slot e mod
   MESH_STREAM_TIMES, 1 + e and the ticks (mach_absolute_time) at which its first SEND was posted, or its last byte
   landed; its first word 0 while its second is stored.  `touching`, the bridge's: the owner word of a holding into whose
   rows the bridge may still map a landed page (0 none), whose rows are reclaimed only once it no longer reads it.
   Nothing in it names a lane, a unit, a frame or a queue pair */
struct mesh_stream {
  _Alignas(64) _Atomic uint32_t peer;
  _Atomic uint32_t live;
  uint64_t lists,posted_at,landed_at;
  _Atomic uint64_t bandwidth,wire,free,begun;
  _Alignas(64) _Atomic uint64_t owner;
  _Atomic uint64_t deadline,sends,lands,told;
  _Alignas(64) _Atomic uint64_t ready;
  _Alignas(64) _Atomic uint64_t sent;
  _Atomic uint64_t landed,taken[2],touching;
  _Alignas(64) _Atomic uint64_t start[MESH_STREAM_START];
  _Alignas(64) _Atomic uint64_t peer_start[MESH_STREAM_START];
};
_Static_assert(sizeof(struct mesh_stream)==384,"mesh_stream");
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
/* design/transport.md; `busy`: the streams held (0 between calls) and since when none has been (`idle_since`, the
   wall clock in ns; 0 while any is) */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,streams,clients;
  uint64_t planes_off,arena_off,page_off,buffer_off,bridge_off,stream_off,client_off,data_off,length,boot;
  _Atomic uint64_t bridge_pid,serial,control,map,map_off,map_nodes,health,busy,idle_since,stopping;
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
/* design/transport.md section 14: the stream of the node's link `p` */
static inline struct mesh_stream *mesh_stream_at(struct hdr *m,uint32_t p){return (struct mesh_stream *)((char *)m+m->stream_off)+p;}
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
/* design/transport.md; design/transport.md */
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz))*m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
/* design/transport.md */
/* the arena occupancy bitmap is indexed by arena page and sized by arena pages: it is not a row plane */
static inline _Atomic uint64_t *mesh_arena_bits(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->arena_off); }
static inline struct mesh_page_entry *mesh_page(struct hdr *m){ return (struct mesh_page_entry*)((unsigned char*)m+m->page_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
/* design/algorithm-sources.md#programcopy; design/transport.md */
static inline uint32_t mesh_row_chunks(struct hdr *m,uint32_t row,uint64_t bytes){
  uint64_t block=(uint64_t)m->block*m->pgsz;
  uint64_t offset=atomic_load_explicit(&mesh_page(m)[row].address,memory_order_relaxed)-m->data_off;
  return (uint32_t)((offset%block+bytes+block-1)/block);
}

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
   entry and buffer, and a bit in each plane), the bridge's own pages, the streams and the client slots; the map's
   records and the streams' rings are leases the bridge takes from the arena (mesh_records, mesh_stream_at), which the
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
  h->stream_off=at; at+=(uint64_t)MESH_STREAMS*sizeof(struct mesh_stream); at=(at+pgsz-1)/pgsz*pgsz;
  h->client_off=at; at+=(uint64_t)MESH_CLIENTS*sizeof(struct mesh_client);
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)arena*pgsz;
  h->length=at; return at;
}
#endif
