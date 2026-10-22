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
   alone (design/transport.md section 14; the versions before it, this file's history).  The wire is HELLO, FAREWELL,
   RECORD and LINK */
#define MESH_VERSION 131u
/* the region's layout (struct hdr `version`), apart from the wire's MESH_VERSION: a bridge adopts the region a bridge
   before it left where its layout is this one (design/transport.md section 2, the region outlives its bridge) */
#define MESH_LAYOUT 133u
#define MESH_ABSENT UINT32_MAX
/* design/transport.md section 14: the region's streams (several toward each node a bridge reaches), and its client
   slots; and the bytes of the bridge's own pages, which it publishes for its own observation tools and no client reads
   (`bridge_off`, mesh-observe.h) */
#define MESH_STREAMS 32
#define MESH_CLIENTS 64
#define MESH_BRIDGE_BYTES (MESH_STREAMS*32u)
#define MESH_RETIRING (UINT64_C(1)<<63)
/* design/transport.md section 14: a stream's two classes, its large units (the region's block) and its small units
   (one frame), each a ring of MESH_STREAM_LARGE_UNITS or MESH_STREAM_SMALL_UNITS units; an owner word leaves bit 63 clear
   (its generation is 25 bits, mesh-dataflow.c mesh_slot_take) */
#define MESH_FRAME 4096u
#define MESH_STREAM_LARGE_UNITS 512u
#define MESH_STREAM_SMALL_UNITS 256u
/* design/transport.md section 10: a unit's dud bit, the top bit of its words (struct mesh_stream), above any cycle's 1 +
   cycle.  In a landed word, the bridge's: the unit came as a dud (1 + its cycle with this bit), or its link was lost
   (~0, which has it too); a consumer compares the word's other bits with 1 + the cycle it awaits, and a landing with the
   bit is its group's failure.  In a ready word, its holder's: the unit is supplied as a dud (its producer's group has
   failed), which its bridge sends as one */
#define MESH_DUD (UINT64_C(1)<<63)
/* design/transport.md section 14: the entries of each of a stream class's two time rings (struct mesh_stream) */
#define MESH_STREAM_TIMES 16384u
_Static_assert(38+25<=63,"an owner word");
_Static_assert(!(MESH_STREAM_LARGE_UNITS&(MESH_STREAM_LARGE_UNITS-1)) && !(MESH_STREAM_SMALL_UNITS&(MESH_STREAM_SMALL_UNITS-1)) &&
  !(MESH_STREAM_TIMES&(MESH_STREAM_TIMES-1)),"a ring's units a power of two");
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
/* design/transport.md section 14: the stream toward node `peer` (0 none), `live` while its link is up, the bridge's to
   write; per class c (0 large, 1 small) its rings, the in ring the bridge's receives land in and the out ring its holder
   supplies (`in`, `out`, region offsets), of `units[c]` units of `unit[c]` bytes, a word a unit its holder stores to
   supply it (`ready`: 1 plus the unit's cycle, its sequence over the ring, with MESH_DUD where it is supplied as a dud)
   and a word a unit the bridge stores as it lands (`landed`: 1 plus its cycle, with MESH_DUD where it came as a dud; ~0
   once the link is lost: a landing only raises a word's other bits, so ~0 stands); its share of the link's bandwidth
   (the link's over its live streams, so its live streams' shares together are the link's: mesh-call.h
   mesh_nodes_reached); `wire`, the stream's identity, the same at both its ends and no other stream's between the two
   nodes, by which the two ends of a communicator hold the two ends of one stream (0 until it is live); `free` while no
   client holds it and every unit its last holder declared has gone, the first unit a holder then supplies of each class
   (`next`); and its holder's: its owner word (`owner`, taken 0 to its own while `free`, as rows are, and let go to 0),
   and the units of each class it will supply, up to `declared` (an absolute sequence), by the uptime `deadline` (ns): a
   declared unit not supplied by then, or by the holding's end, goes as a dud.  Per class, two time rings the bridge alone
   writes (`posted_at`, `landed_at`, region offsets, each on lines of its own) of MESH_STREAM_TIMES entries of two words:
   unit u's, entry u mod MESH_STREAM_TIMES, 1 + u and the ticks (mach_absolute_time, unconverted) read once at the end of
   the send thread's pass that posted u, or of the receive thread's pass that released it, after every post or landed
   word of the pass; its first word 0 while its second is stored, so an entry whose first word does not read 1 + u
   around its second has no time for u (lapped, or not yet stored) */
struct mesh_stream {
  _Atomic uint32_t peer,live;
  uint32_t unit[2],units[2];
  uint64_t in[2],out[2],ready[2],landed[2];
  _Atomic uint64_t bandwidth,wire,free,next[2];
  _Atomic uint64_t owner,declared[2],deadline;
  uint64_t posted_at[2],landed_at[2];
};
_Static_assert(sizeof(struct mesh_stream)==192,"mesh_stream");
/* design/transport.md: a node's record, the newest of which every bridge floods to the mesh: its node, name and boot,
   the record's sequence in that boot, the bytes of its bridge's region its clients may hold (the region's arena less what the
   bridge holds itself: the map, the streams' pages and rings, the duds' block), the node's memory in bytes and hardware
   model (hw.model, mesh-plan.h MESH_MODEL_BYTES with its NUL), its bridge's `reached` links, an entry a link: the node
   it reaches, its streams and its bandwidth (mesh-plan.h struct mesh_reach; mesh-call.h mesh_nodes_reached), and its
   host UUID (`uuid`: gethostuuid, the identity its node number is drawn from, AGENTS.md "A node is its hardware"; a
   loop bridge's, its host's mixed with its region's name), by which two hosts that share a number are told apart and
   logged (mesh-records.c number_shared) */
struct mesh_record { uint32_t node,reached; uint64_t boot,sequence; char name[MESH_NAME_BYTES]; uint64_t region,memory;
  char model[MESH_MODEL_BYTES]; struct mesh_reach reach[MESH_STREAMS]; struct mesh_health health; uint8_t uuid[16]; };
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
