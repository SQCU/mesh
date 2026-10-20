#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <infiniband/verbs.h>
#include <os/os_sync_wait_on_address.h>
#include "mesh-plan.h"
/* design/transport.md */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
/* the mesh's unique local prefix, each node's fabric identity under it (bin/mesh-router-init.sh) */
#define MESH_ULA_PREFIX "fd6d:6573:68:"
#define MESH_MODE 0666
/* 124: the link lanes' words declared; 125: communicators' lane pairs leased at init through NCCL's bootstrap, the
   channels, requests, exec, services, holds, stripe rings and ledger deleted (metal-microbench
   output_data/structural-20261018/FINAL-BRIDGE.md); 126: a link lane's rings are the lane's (struct mesh_lane: the
   bridge's arena, its receives posted continuously over the pairing), a lease is the lane's word alone, and a
   position is a run of the lane's units both ends lay alike (design/transport.md); programs, flows,
   transfer lists, deliveries, cancellation records, relays and a lease's DIRTY deleted; 127: a lane pair bound by the
   bridge at a client's asking through the client's own slot (struct mesh_bind), the lease words deleted, the free pairs
   published by the bridge (struct hdr `free`), a binding's declared units sent whole, the unsupplied ones as duds
   (design/transport.md section 14).  The wire is HELLO, FAREWELL, RECORD and LINK */
#define MESH_VERSION 127u
#define MESH_ABSENT UINT32_MAX
/* design/transport.md: the node's ports and its client slots */
#define MESH_PORTS 8
#define MESH_CLIENTS 64
#define MESH_RETIRING (UINT64_C(1)<<63)
/* design/transport.md: a link's queue pairs, made once at its pairing, are its MESH_LINK_LANES lanes: lane
   MESH_LANE_KEEPALIVE the keepalive's, lane 1 reserved and unused, and from MESH_LANE_FIRST pairs, a pair's large lane
   (MESH_LANE_FIRST + 2 p: its unit the window's block) and its small lane (+ 2 p + 1: its unit one frame).  A pair is
   bound to one client at a time by the bridge (struct mesh_bind); an owner word leaves bit 63 clear (its generation is 25
   bits, mesh-dataflow.c mesh_slot_take) */
#define MESH_LINK_LANES 10u
#define MESH_LANE_KEEPALIVE 0u
#define MESH_LANE_FIRST 2u
#define MESH_LANE_PAIRS ((MESH_LINK_LANES-MESH_LANE_FIRST)/2)
#define MESH_FRAME 4096u
#define MESH_LANE_LARGE_UNITS 512u
#define MESH_LANE_SMALL_UNITS 256u
_Static_assert(38+25<=63 && MESH_LANE_FIRST+2*MESH_LANE_PAIRS==MESH_LINK_LANES,"an owner word; the lanes' pairs");
_Static_assert(!(MESH_LANE_LARGE_UNITS&(MESH_LANE_LARGE_UNITS-1)) && !(MESH_LANE_SMALL_UNITS&(MESH_LANE_SMALL_UNITS-1)),"a ring's units a power of two");
static inline int mesh_lane_large(uint32_t lane){ return lane>=MESH_LANE_FIRST && !((lane-MESH_LANE_FIRST)&1); }
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
/* design/transport.md section 14: a link lane's rings, the bridge's for its port's life (taken from the arena as
   the port first pairs): its in and out rings (`in`, `out`, region offsets) of `units` units of `unit` bytes, a word a
   unit its pair's bound client stores to supply it (`ready`: 1 plus the unit's cycle, its sequence over `units`), and a
   word a unit the bridge stores as it lands (`landed`: 1 plus its cycle; ~0 once the link the lane rode was lost: a
   landing only raises a word, so ~0 stands); `live` while its rings run under the link's pairing: its receives posted
   and its SENDs served, from its reset (every word 0, its sequences 0: at the pairing's connect where no binding holds
   it, else as a binding held across a loss ends) to the link's next loss */
struct mesh_lane { uint64_t in,out,ready,landed; uint32_t unit,units; _Atomic uint32_t live,padding; };
_Static_assert(sizeof(struct mesh_lane)==48,"mesh_lane");
/* design/transport.md: a port of the node, by the index the bridge gave it (its peer node while paired, its
   phase, MESH_UNKNOWN where the bridge has no such port, the code it last stopped with, its bandwidth, the pairings it
   made, and the queue pairs its device holds against those it can use, `pairs` of `max_pairs`, its max_qp less one, and
   the link lanes made at its pairing, `lanes`, 0 none); and a node's record, the newest of which every bridge floods to
   the mesh (its node, name and boot, the record's sequence in that boot, the bytes of its bridge's window its clients may
   hold (the window less what the bridge holds itself: the map, the links' pages and rings, the duds' block) and the
   node's memory in bytes and hardware model (hw.model, mesh-plan.h MESH_MODEL_BYTES with its NUL), and its `ports` ports by index) */
struct mesh_link_info { _Atomic uint32_t peer,phase; _Atomic int64_t code; _Atomic uint64_t bandwidth,pairings; _Atomic uint32_t pairs,max_pairs;
  _Atomic uint32_t lanes,padding; };
_Static_assert(sizeof(struct mesh_link_info)==48 && offsetof(struct mesh_link_info,lanes)==40,"mesh_link_info");
struct mesh_record { uint32_t node,ports; uint64_t boot,sequence; char name[MESH_NAME_BYTES]; uint64_t window,memory;
  char model[MESH_MODEL_BYTES]; struct mesh_link_info port[MESH_PORTS]; struct mesh_health health; };
/* design/transport.md section 14: a client's binding of port p's lane pair q, in its own slot (bind[p][q]): the
   client asks with `want` (a serial of its own; 0, none) and declares, per lane j (0 the pair's large lane, 1 its small),
   the units it will supply up to (`declared[j]`, an absolute sequence) and the uptime by which it supplies them
   (`deadline`, ns); the bridge answers `answer`: the serial once the pair is the client's, MESH_BIND_REFUSED with the
   reason, or 0 once a binding has ended, and `start[j]`, the first unit lane j of the binding supplies (the bridge's
   count of units sent on it).  A declared unit not supplied by the deadline, or by the binding's end, goes as a dud */
struct mesh_bind { _Atomic uint64_t want,answer,declared[2],deadline,start[2],padding; };
_Static_assert(sizeof(struct mesh_bind)==64,"mesh_bind");
#define MESH_BIND_REFUSED (UINT64_C(1)<<63)
/* a refusal of request `serial` for `error`, and the request an answer answers */
#define MESH_BIND_REFUSAL(serial,error) (MESH_BIND_REFUSED|((uint64_t)(serial)<<16)|(uint64_t)(uint16_t)(error))
static inline uint64_t mesh_bind_serial(uint64_t answer){ return answer&MESH_BIND_REFUSED?(answer&~MESH_BIND_REFUSED)>>16:answer; }
/* design/transport.md: a client slot, a process's attach (`owner`, 0 vacant) under its execution's key (`key`), and
   its bindings of lane pairs */
struct mesh_client {
  _Alignas(64) _Atomic uint64_t owner;
  uint64_t key,padding[6];
  struct mesh_bind bind[MESH_PORTS][MESH_LANE_PAIRS];
};
/* design/transport.md; `busy`: the lanes bound (0 between calls) and since when none has been (`idle_since`, the wall
   clock in ns; 0 while any is); `bell`: a port's word its parked progress threads wait on, which a client rings
   (mesh_notify) as it supplies a lane's units or declares them; `free`: each port's lane pairs a client may ask to bind,
   a bit a pair, the bridge's to write (design/transport.md section 14) */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,links,clients;
  uint64_t planes_off,arena_off,page_off,buffer_off,link_off,lane_off,client_off,data_off,length,boot;
  _Atomic uint64_t bridge_pid,serial,control,map,map_off,map_nodes,health,busy,idle_since,stopping;
  _Atomic uint64_t bell[MESH_PORTS];
  _Atomic uint64_t free[MESH_PORTS];
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
/* design/algorithm-sources.md#programcopy */
static inline struct mesh_link_info *mesh_links(struct hdr *m){return (struct mesh_link_info *)((char *)m+m->link_off);}
/* design/transport.md: an owner word is its generation, slot and process: (generation << 38) | (slot << 32) | pid */
static inline struct mesh_client *mesh_client_at(struct hdr *m,uint64_t owner){return (struct mesh_client *)((char *)m+m->client_off)+((owner>>32)&(MESH_CLIENTS-1));}
/* design/transport.md: port p's lane k */
static inline struct mesh_lane *mesh_lane_at(struct hdr *m,uint32_t p,uint32_t k){return (struct mesh_lane *)((char *)m+m->lane_off)+(size_t)p*MESH_LINK_LANES+k;}
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
   pages a block (the bridge's window, its node's configured share of its memory), a page-table row a block (its page
   entry and buffer, and a bit in each plane), the node's ports, their lanes and the client slots; the map's records and
   the lanes' rings are leases the bridge takes from the arena (mesh_records, mesh_lane_at); the arena is registered
   whole on every port's device before any queue pair exists (mesh-verbs.h device_up) */
static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t pages){
  uint32_t arena=pages/block*block,rows=arena/block;
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)rows+63)/64;
  h->pgsz=pgsz; h->block=block; h->rows=rows; h->links=MESH_PORTS; h->clients=MESH_CLIENTS;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->arena_off=at; at+=(((uint64_t)arena+63)/64)*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)rows*sizeof(struct mesh_page_entry); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->link_off=at; at+=(uint64_t)MESH_PORTS*sizeof(struct mesh_link_info); at=(at+pgsz-1)/pgsz*pgsz;
  h->lane_off=at; at+=(uint64_t)MESH_PORTS*MESH_LINK_LANES*sizeof(struct mesh_lane); at=(at+pgsz-1)/pgsz*pgsz;
  h->client_off=at; at+=(uint64_t)MESH_CLIENTS*sizeof(struct mesh_client);
  uint64_t bytes=(uint64_t)block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)arena*pgsz;
  h->length=at; return at;
}
#endif
