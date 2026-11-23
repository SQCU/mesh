#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <libproc.h>
#include <os/os_sync_wait_on_address.h>
#include "mesh-view.h"
#include "mesh-call.h"
/* design/PAGE-TABLE-SPEC-20261122.md: the node's region, which the bridge makes and every process of the node maps */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
/* the region's name: MESH_NAME, or $MESH_REGION where set (another region on the same machine) */
static inline const char *mesh_name(void){ const char *name=getenv("MESH_REGION"); return name && *name?name:MESH_NAME; }
#define MESH_PORT "18519"
/* the mesh's unique local prefix, each node's fabric identity under it (bin/mesh-router-init.sh) */
#define MESH_ULA_PREFIX "fd6d:6573:68:"
#define MESH_MODE 0666
/* The wire's contract between bridges: a bridge of another version is not paired with (mesh-link.c port_heard).  164:
   every SEND is one page of the node's pool, MESH_HEAD bytes of header and then its payload, the header naming each
   destination as a node and an index of that node's page table (mesh-switched.h), a page sent on the lanes of its
   payload's class alone, and a link's lane j its queue pair j (mesh-bridge.h link_plan).  Against 163: a line
   (mesh-switched.h struct mesh_switch_line, the same 32 bytes) carries for each page class its link's measured time now
   and its idle time (163's one wait, spread, rate and latency, the last two nominal), stamped with its sender's boot
   plus uptime (163's count of its sender's refreshes, which began again at 0 as the sender restarted) */
#define MESH_VERSION 164u
/* The region's layout (struct hdr `version`), apart from the wire's MESH_VERSION: a bridge adopts the region a bridge
   before it left where its layout is this one.  187: the page table and its owner bytes (`table_off`), the clients'
   write rings (`ring_off`), the pool (`pool_off`), the node table (`node_off`), and a hand for each port child; against
   186, the switch keeps the remote links' lines apart from the view, under their ends' node table entries
   (mesh-switched.h struct mesh_switch `heard`, the view's `slot`), and none of its own links' (the view's links have
   no line or stamp), each cable has its measured rate and a page's time alone for each class (struct
   mesh_switch_cable), and the stamps' clock offset (`stamp_offset`) is where the refresh's count (`stamp`) was */
#define MESH_LAYOUT 187u
#define MESH_ABSENT UINT32_MAX
/* the region's lane records (struct mesh_lanes: the bridge's, MESH_LINK_STREAMS a port), the most ports, the nodes the
   node table has (struct mesh_node) and the client slots; and the bytes of the bridge's own pages, which it publishes
   for its own observation tools and no client reads (`bridge_off`, mesh-observe.h) */
#define MESH_STREAMS 32
#define MESH_LINK_STREAMS 4u
#define MESH_PORTS (MESH_STREAMS/MESH_LINK_STREAMS)
#define MESH_NODES 8u
#define MESH_CLIENTS 64
#define MESH_BRIDGE_BYTES (MESH_STREAMS*32u)
/* the switch's bytes (`switch_off`, mesh-switched.h struct mesh_switch: the node's lanes and their outboxes, the view,
   the weights and the trainer's log rings) */
#define MESH_SWITCH_BYTES (UINT64_C(128)<<20)
#define MESH_LEAVING (UINT64_C(1)<<63)
_Static_assert(38+25<=63,"an owner word");
/* the pause between a port's hellos and between the bridge's liveness turns */
#define MESH_PAUSE_NS UINT64_C(3000000000)
/* the operator's bound (2026-10-18): no DMA or network send/receive channel of the mesh is used longer than this
   without something happening on it or its scope concluding */
#define MESH_PLATFORM_NS UINT64_C(5000000000)
/* The pool: the pages of each class the bridge carves at its start (fewer, in proportion, where the arena has
   fewer), MESH_CLASSES classes (mesh-call.h MESH_CLASS_SHORT, MESH_CLASS_LONG); and a client slot's write ring's
   cells */
#define MESH_POOL_SHORT 16384u
#define MESH_POOL_LONG 16384u
#define MESH_POOL_PAGES (MESH_POOL_SHORT+MESH_POOL_LONG)
#define MESH_CLASSES 2u
#define MESH_RING 1024u
_Static_assert(!(MESH_RING&(MESH_RING-1)),"a ring a power of two");
_Static_assert(MESH_TABLE<=UINT32_MAX/2 && MESH_POOL_PAGES<UINT32_MAX,"an index and a page id in 32 bits");
/* the payload bytes of a page of `frames` frames */
#define MESH_PAYLOAD(frames) ((frames)*MESH_FRAME-MESH_HEAD)

/* An exact byte extent of the region (offset from its base, bytes); one never crosses a registration extent */
#define MESH_DMA_EXTENT (UINT64_C(1)<<30)
struct mesh_dma_span { uint64_t offset,bytes; };
_Static_assert(sizeof(struct mesh_dma_span)==16,"mesh_dma_span ABI");

/* A page of the pool (`pool_off`), its id its descriptor's index: its region offset (`offset`, where its header is;
   its payload MESH_HEAD after it), its class (`cls`, its frames: MESH_CLASS_SHORT or MESH_CLASS_LONG), its payload's
   bytes, its uses (a table word naming it, a SEND pushed or posted of it, a write of it pending in a ring, a landing
   being processed: at 0 it goes back to its class's free stack), its state, the lane a receive of it is posted on, and
   `next`, its link on the free stack (the next page's id + 1, 0 none).  The bridge carves the pages from the arena
   once, at its start, each run of them within one MESH_DMA_EXTENT; a SEND of a page covers its payload's class's
   frames (mesh-switched.h mesh_switch_forward), never more than its own */
enum { MESH_PD_FREE, MESH_PD_RECEIVE, MESH_PD_LIVE };
struct mesh_pd { _Atomic uint64_t next; uint64_t offset; _Atomic uint32_t uses,bytes; uint16_t cls; _Atomic uint16_t state;
  _Atomic uint32_t lane; };
_Static_assert(sizeof(struct mesh_pd)==32 && offsetof(struct mesh_pd,uses)==16 && offsetof(struct mesh_pd,lane)==28,"mesh_pd");
/* The pool's header: a class's free stack (`head`: the top page's id + 1 in its low 32 bits, 0 empty, and a tag in its
   high 32 that every change moves), its free pages (never fewer than the stack has), its frames and its pages; and
   the pages the bridge carved (`pages`, the descriptors that name pages).  The descriptors follow it */
struct mesh_pool_class { _Alignas(64) _Atomic uint64_t head; _Atomic uint32_t free; uint32_t frames,pages; };
struct mesh_pool { struct mesh_pool_class cls[MESH_CLASSES]; _Alignas(64) uint32_t pages; };

/* A write as a client's ring cell has it: the page the index named when it was written (`pd`: the write's use is on
   it until the bridge takes it; the take reads the page, never the index), its payload bytes and its destinations */
struct mesh_write { uint32_t pd,bytes,dests,unused; struct mesh_dest dest[MESH_DESTS]; };
_Static_assert(sizeof(struct mesh_write)==16+8*MESH_DESTS,"mesh_write");
/* A client slot's write ring (`ring_off`, one a slot): one producer, the slot's client, and one taker, the port child
   whose share it is (ring s the (s mod n)-th of the n serving ports, by the port lines: mesh-switched.h struct
   mesh_switch `port`; as the serving ports change, two may take a moment, each position one compare-exchange's).
   Cell c has the write of position p (c = p mod MESH_RING) while its sequence is 2 (p / MESH_RING) + 1 and is free for
   it while its sequence is 2 (p / MESH_RING), so a zeroed ring is empty; `put` is the next position written and `take`
   the next taken */
struct mesh_cell { _Atomic uint64_t sequence; struct mesh_write w; };
struct mesh_ring { _Alignas(64) _Atomic uint64_t put; _Alignas(64) _Atomic uint64_t take; _Alignas(64) struct mesh_cell cell[MESH_RING]; };

/* A node this node's bridge reaches (`node_off`, MESH_NODES of them; node 0: the entry unused), written by the bridge:
   `live` while the map reaches it, `generation` the entry's, and the bandwidth of this node's own cables toward it (0:
   reached through relays) */
struct mesh_node { _Alignas(64) _Atomic uint32_t node; _Atomic uint32_t live; _Atomic uint64_t generation,bandwidth; };
_Static_assert(sizeof(struct mesh_node)==64,"mesh_node");

/* The bridge's lane records (never a client's): a port's lane pair s, its peer and identity as its lanes last opened
   (whether they serve is its port's line: mesh-switched.h struct mesh_switch `port`), each lane's requests posted and
   completed each way, its passes and SEND times, and the port child that posts them (`child`, `child_start`); `unused`
   and `spare` are unused (every field keeps its offset: the heartbeat and observers read them) */
struct mesh_posts { _Alignas(64) _Atomic uint64_t posted[2]; _Atomic uint64_t done[2],unused[2]; };
_Static_assert(sizeof(struct mesh_posts)==64 && offsetof(struct mesh_posts,done)==16,"mesh_posts ABI");
struct mesh_lanes {
  _Alignas(64) _Atomic uint32_t peer;
  uint32_t unused;
  uint64_t spare;
  _Atomic uint64_t bandwidth,wire;
  /* the passes its port's progress thread made (mesh-link.c port_pass) */
  _Atomic uint64_t passes;
  /* its SENDs' post -> completion (the NIC's, the wire's and the peer NIC's acknowledgement, no pass of either bridge):
     the nanoseconds summed and the SENDs timed, both lanes of the record, by its port's progress thread */
  _Atomic uint64_t send_ns,sends;
  _Atomic uint32_t child,padding;
  uint64_t child_start;
  struct mesh_posts posts[2];
};
_Static_assert(sizeof(struct mesh_lanes)==256,"mesh_lanes");
/* design/transport.md: a node's record, the newest of which every bridge floods to the mesh: its node and boot, the
   record's sequence in that boot, the bytes its clients' pages may fill (`region`: the payload bytes of its bridge's
   pool, every class's pages), the node's memory in bytes and hardware model (hw.model, mesh-view.h MESH_MODEL_BYTES with its
   NUL), and the `reached` nodes its bridge reaches (mesh-view.h struct mesh_reach).  No host name: a name a person reads
   is a label the readers take from the name table by node number (etc/mesh-nodes.json, rdma/peers.py label) */
struct mesh_record { uint32_t node,reached; uint64_t boot,sequence,region,memory;
  char model[MESH_MODEL_BYTES]; struct mesh_reach reach[MESH_STREAMS]; struct mesh_health health; };
/* design/transport.md: a client slot, a process's attach (`owner`, 0 vacant; its low 32 bits the process's pid) and
   that process's start (`start`, mesh_started: with the pid, its identity, which a reused pid does not share).  The
   slot's fields, as the header's, are only ever appended to, so any later bridge or heartbeat reads a region's
   processes at these offsets */
struct mesh_client {
  _Alignas(64) _Atomic uint64_t owner;
  uint64_t start,padding[6];
};
/* The region's header.  Every field keeps its offset in every layout (the install observer and an earlier build's
   heartbeat read them); a field whose role ends is renamed in place.  `entries` the page table's indices; `table_off`
   the page table's words and owner bytes, `arena_off` the arena's bitmap (a bit an arena page of `pgsz` bytes, `block`
   pages a block), `ring_off` the clients' write rings, `pool_off` the pool, `bridge_off` the bridge's own pages,
   `stream_off` the lane records, `client_off` the client slots, `data_off` the arena, `node_off` the node table and
   `switch_off` the switch; `attached` the client slots whose rings the bridge's port children look at, a bit a slot
   (mesh_ring_flag); `busy` the frames the bridge's lanes have queued (0 while none) and `idle_since` the wall clock in
   ns at which they last had none (0 while they have any); `spare` is unused */
struct hdr {
  uint32_t magic,version,pgsz,block,entries,node,streams,clients;
  uint64_t table_off,arena_off,ring_off,pool_off,bridge_off,stream_off,client_off,data_off,length,boot;
  _Atomic uint64_t bridge_pid,serial,attached,map,map_off,map_nodes,health,busy,idle_since;
  uint64_t spare;
  uint64_t node_off,switch_off;
};
_Static_assert(offsetof(struct hdr,version)==4 && offsetof(struct hdr,pgsz)==8 && offsetof(struct hdr,block)==12 &&
               offsetof(struct hdr,entries)==16 && offsetof(struct hdr,node)==20 && offsetof(struct hdr,clients)==28 &&
               offsetof(struct hdr,table_off)==32 && offsetof(struct hdr,arena_off)==40 && offsetof(struct hdr,ring_off)==48 &&
               offsetof(struct hdr,pool_off)==56 && offsetof(struct hdr,bridge_off)==64 && offsetof(struct hdr,stream_off)==72 &&
               offsetof(struct hdr,client_off)==80 && offsetof(struct hdr,data_off)==88 && offsetof(struct hdr,length)==96 &&
               offsetof(struct hdr,bridge_pid)==112 && offsetof(struct hdr,attached)==128 && offsetof(struct hdr,spare)==184 &&
               offsetof(struct hdr,node_off)==192 && offsetof(struct hdr,switch_off)==200 && sizeof(struct hdr)==208,
               "struct hdr: every field at its offset");
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");
/* a shared word (`map`, `health`) bumped and its waiters woken, the writes after it ordered behind it */
static inline void mesh_notify(_Atomic uint64_t *word){
  atomic_fetch_add_explicit(word,1,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* design/transport.md: the map, `map_nodes` records at `map_off` in arena pages the bridge leases for itself, and
   leases again larger as the mesh grows; the bridge notifies `map`, its serial, before and after each change (odd
   while it writes; its own record is in before the region opens, so a reader never reads 0), and a reader's copy is
   whole where `map` read (acquired) before it was even and is unchanged after it */
static inline struct mesh_record *mesh_records(struct hdr *m){ return (struct mesh_record *)((char *)m+atomic_load_explicit(&m->map_off,memory_order_relaxed)); }
static inline int mesh_map_whole(struct hdr *m,uint64_t before){
  atomic_thread_fence(memory_order_acquire);
  return !(before&1) && atomic_load_explicit(&m->map,memory_order_relaxed)==before;
}
/* an owner word is its generation, slot and process: (generation << 38) | (slot << 32) | pid */
static inline struct mesh_client *mesh_client_at(struct hdr *m,uint64_t owner){return (struct mesh_client *)((char *)m+m->client_off)+((owner>>32)&(MESH_CLIENTS-1));}
/* the bridge's lane records of port p's lane pair s: record p MESH_LINK_STREAMS + s */
static inline struct mesh_lanes *mesh_lanes_at(struct hdr *m,uint32_t p){return (struct mesh_lanes *)((char *)m+m->stream_off)+p;}
/* the node table's entry n */
static inline struct mesh_node *mesh_node_at(struct hdr *m,uint32_t n){return (struct mesh_node *)((char *)m+m->node_off)+n;}
/* the arena's pages of pgsz bytes, and its bitmap (a bit a page, set while a lease of the bridge has it) */
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz))*m->block; }
static inline _Atomic uint64_t *mesh_arena_bits(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->arena_off); }
/* arena page `page`'s address */
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }

/* ---- The page table (`table_off`) ------------------------------------------------------------------------------
   A word an index, (landed << 32) | (pd + 1), where pd + 1 == 0 names no page and `landed` counts the landings at the
   index; a word is only ever replaced whole, by compare-exchange (release), and read by an acquire load.  After the
   words, an owner byte an index: its client's slot + 1 (mesh_owner_byte), 0 while the index is free */
static inline _Atomic uint64_t *mesh_table(struct hdr *m){ return (_Atomic uint64_t *)((char *)m+m->table_off); }
static inline _Atomic uint8_t *mesh_owners(struct hdr *m){ return (_Atomic uint8_t *)((char *)m+m->table_off+(uint64_t)m->entries*sizeof(uint64_t)); }
static inline uint32_t mesh_word_pd(uint64_t word){ return (uint32_t)word-1u; }
static inline uint32_t mesh_word_landed(uint64_t word){ return (uint32_t)(word>>32); }
static inline uint64_t mesh_entry_load(struct hdr *m,uint32_t index){ return atomic_load_explicit(&mesh_table(m)[index],memory_order_acquire); }
/* index's word set to name page pd (MESH_ABSENT: none), its landed raised by `landing` (1: a landing; 0: an
   invalidation): the page the word named before (MESH_ABSENT: none), whose use the caller drops (mesh_pd_drop); pd's
   use for the word the caller has counted (mesh_pd_use) */
static inline uint32_t mesh_entry_put(struct hdr *m,uint32_t index,uint32_t pd,uint32_t landing){
  _Atomic uint64_t *word=&mesh_table(m)[index];
  uint64_t old=atomic_load_explicit(word,memory_order_acquire);
  while(!atomic_compare_exchange_weak_explicit(word,&old,((uint64_t)(mesh_word_landed(old)+landing)<<32)|(uint32_t)(pd+1u),
                                                memory_order_acq_rel,memory_order_acquire));
  return mesh_word_pd(old);
}
/* index's word made anew, as its allocation makes it: naming pd (MESH_ABSENT: none), landed 0; the page it named
   before, as mesh_entry_put's */
static inline uint32_t mesh_entry_reset(struct hdr *m,uint32_t index,uint32_t pd){
  return mesh_word_pd(atomic_exchange_explicit(&mesh_table(m)[index],(uint32_t)(pd+1u),memory_order_acq_rel));
}

/* ---- The pool (`pool_off`) ---------------------------------------------------------------------------------------- */
static inline struct mesh_pool *mesh_pool(struct hdr *m){ return (struct mesh_pool *)((char *)m+m->pool_off); }
static inline struct mesh_pd *mesh_pd_at(struct hdr *m,uint32_t pd){ return (struct mesh_pd *)((char *)m+m->pool_off+sizeof(struct mesh_pool))+pd; }
/* whether pd names a page the bridge carved (MESH_ABSENT names none) */
static inline int mesh_pd_valid(struct hdr *m,uint32_t pd){ return pd<mesh_pool(m)->pages; }
/* the class of pages of `frames` frames; NULL where no class has them */
static inline struct mesh_pool_class *mesh_pool_class(struct hdr *m,uint32_t frames){
  return frames==MESH_CLASS_SHORT?&mesh_pool(m)->cls[0]:frames==MESH_CLASS_LONG?&mesh_pool(m)->cls[1]:NULL;
}
/* the frames of the least class whose payload fits `bytes`; 0 where none does */
static inline uint32_t mesh_class_frames(uint32_t bytes){ return bytes<=MESH_SHORT_PAYLOAD?MESH_CLASS_SHORT:bytes<=MESH_LONG_PAYLOAD?MESH_CLASS_LONG:0; }
/* page pd's header (its first byte, where a SEND of it starts) and its payload */
static inline unsigned char *mesh_pd_head(struct hdr *m,uint32_t pd){ return (unsigned char *)m+mesh_pd_at(m,pd)->offset; }
static inline unsigned char *mesh_pd_data(struct hdr *m,uint32_t pd){ return mesh_pd_head(m,pd)+MESH_HEAD; }
/* page pd, its uses 0, made FREE and pushed onto its class's free stack */
static inline void mesh_pool_push(struct hdr *m,uint32_t pd){
  struct mesh_pd *p=mesh_pd_at(m,pd);
  struct mesh_pool_class *c=mesh_pool_class(m,p->cls);
  if(!c)return;
  atomic_store_explicit(&p->state,MESH_PD_FREE,memory_order_relaxed);
  atomic_store_explicit(&p->bytes,0,memory_order_relaxed);
  atomic_fetch_add_explicit(&c->free,1,memory_order_relaxed);
  uint64_t head=atomic_load_explicit(&c->head,memory_order_relaxed);
  do atomic_store_explicit(&p->next,(uint32_t)head,memory_order_relaxed);
  while(!atomic_compare_exchange_weak_explicit(&c->head,&head,(((head>>32)+1)<<32)|(uint64_t)(pd+1u),memory_order_release,
                                               memory_order_relaxed));
}
/* a page popped from the free stack of the class of `frames` frames: its id, or MESH_ABSENT where the stack is empty or
   no class has such pages; its state, uses and lane are the caller's to set */
static inline uint32_t mesh_pool_pop(struct hdr *m,uint32_t frames){
  struct mesh_pool_class *c=mesh_pool_class(m,frames);
  if(!c)return MESH_ABSENT;
  uint64_t head=atomic_load_explicit(&c->head,memory_order_acquire);
  for(;;){
    const uint32_t top=(uint32_t)head;
    if(!top)return MESH_ABSENT;
    const uint64_t next=atomic_load_explicit(&mesh_pd_at(m,top-1)->next,memory_order_relaxed);
    if(atomic_compare_exchange_weak_explicit(&c->head,&head,(((head>>32)+1)<<32)|(uint32_t)next,memory_order_acquire,
                                             memory_order_acquire)){
      atomic_fetch_sub_explicit(&c->free,1,memory_order_relaxed);
      return top-1;
    }
  }
}
/* a use of page pd counted */
static inline void mesh_pd_use(struct hdr *m,uint32_t pd){ atomic_fetch_add_explicit(&mesh_pd_at(m,pd)->uses,1,memory_order_relaxed); }
/* a use of page pd dropped (MESH_ABSENT, a page not carved, or one of no uses: nothing); the last use's drop puts it
   back on its class's free stack */
static inline void mesh_pd_drop(struct hdr *m,uint32_t pd){
  if(!mesh_pd_valid(m,pd))return;
  _Atomic uint32_t *uses=&mesh_pd_at(m,pd)->uses;
  uint32_t count=atomic_load_explicit(uses,memory_order_relaxed);
  while(count && !atomic_compare_exchange_weak_explicit(uses,&count,count-1,memory_order_acq_rel,memory_order_relaxed));
  if(count==1)mesh_pool_push(m,pd);
}

/* ---- The write rings (`ring_off`) --------------------------------------------------------------------------------- */
static inline struct mesh_ring *mesh_ring_at(struct hdr *m,uint32_t slot){ return (struct mesh_ring *)((char *)m+m->ring_off)+slot; }
/* write w put on the ring by its one producer: 1, or 0 where the ring is full.  The put's store is sequentially
   consistent, ordered before the writer's read of its flag (mesh_ring_flag) */
static inline int mesh_ring_push(struct mesh_ring *r,const struct mesh_write *w){
  const uint64_t at=atomic_load_explicit(&r->put,memory_order_relaxed),lap=at/MESH_RING;
  struct mesh_cell *cell=&r->cell[at%MESH_RING];
  if(atomic_load_explicit(&cell->sequence,memory_order_acquire)!=2*lap)return 0;
  cell->w=*w;
  atomic_store_explicit(&cell->sequence,2*lap+1,memory_order_release);
  atomic_store(&r->put,at+1);
  return 1;
}
/* client slot `slot`'s ring flagged for the bridge's port children (struct hdr `attached`, a bit a slot), which look
   at the flagged rings alone: by its writer after each write, where the flag is not set yet (one load a write, and an
   or at its first), and by a slot's take that finds a write left at put (mesh-dataflow.c mesh_slot_take).  The port
   child that takes the ring clears the flag where it finds the slot vacant and the ring empty, then reads the owner
   and the put again and flags the ring again where either moved (mesh-link.c switch_takes).  Each of the four is
   sequentially consistent, so a write is never left on a ring not flagged: a writer that reads its flag set before the
   clear stored its put before the clear, and the child's read after the clear sees that put */
static inline void mesh_ring_flag(struct hdr *m,uint32_t slot){
  const uint64_t bit=UINT64_C(1)<<(slot&(MESH_CLIENTS-1));
  if(!(atomic_load(&m->attached)&bit))atomic_fetch_or(&m->attached,bit);
}
_Static_assert(MESH_CLIENTS==64,"a client slot a bit of `attached`");
/* the writes on the ring as its counters read now (0: none to take) */
static inline uint64_t mesh_ring_pending(struct mesh_ring *r){
  const uint64_t put=atomic_load_explicit(&r->put,memory_order_acquire),take=atomic_load_explicit(&r->take,memory_order_relaxed);
  return put>take?put-take:0;
}

/* ---- Index allocation (the owner bytes) ------------------------------------------------------------------------ */
/* client `client`'s owner byte: its slot + 1 */
static inline uint8_t mesh_owner_byte(uint64_t client){ return (uint8_t)(((client>>32)&(MESH_CLIENTS-1))+1); }
/* the first `count` consecutive free indices in [begin, end) taken for `owner`, first fit: each byte taken by a
   compare-exchange that expects it free, and a stretch that meets one another took first given back and the scan
   continued past it; the first index, or MESH_ABSENT */
static inline uint32_t mesh_index_scan(_Atomic uint8_t *owners,uint8_t owner,uint32_t count,uint32_t begin,uint32_t end){
  for(uint32_t first=begin;first<end && count<=end-first;){
    uint32_t i=0;
    while(i<count && !atomic_load_explicit(&owners[first+i],memory_order_relaxed))i++;
    if(i==count){
      for(i=0;i<count;i++){
        uint8_t free=0;
        if(!atomic_compare_exchange_strong_explicit(&owners[first+i],&free,owner,memory_order_acq_rel,memory_order_relaxed))break;
      }
      if(i==count)return first;
      for(uint32_t k=0;k<i;k++)atomic_store_explicit(&owners[first+k],0,memory_order_release);
    }
    first+=i+1;
  }
  return MESH_ABSENT;
}
/* `count` consecutive free indices taken for `owner`, the lowest such (first fit from index 0), so indices just freed
   are the next ones given and their words are made new (mesh-call.c mesh_pages drops a page a landing left there): the
   first, or MESH_ABSENT */
static inline uint32_t mesh_index_take(struct hdr *m,uint8_t owner,uint32_t count){
  if(!count || count>m->entries)return MESH_ABSENT;
  return mesh_index_scan(mesh_owners(m),owner,count,0,m->entries);
}
/* indices [first, first + count) freed */
static inline void mesh_index_release(struct hdr *m,uint32_t first,uint32_t count){
  for(uint32_t i=0;i<count;i++)atomic_store_explicit(&mesh_owners(m)[first+i],0,memory_order_release);
}

/* ---- Processes of the region -------------------------------------------------------------------------------------- */
/* a process exited (its slot's or the bridge's pid; 0: none) */
static inline int mesh_exited(pid_t process){ return !process || (kill(process,0) && errno==ESRCH); }
/* a process's start in microseconds of the wall clock (libproc PROC_PIDTBSDINFO), 0 where it has exited or is in its
   exit (XNU takes a process from every lookup by pid as its exit begins, before the kernel releases its device
   objects); with its pid, the identity of a process of the region, which a later process given the same pid does not
   share */
static inline uint64_t mesh_started(pid_t process){
  struct proc_bsdinfo info;
  return process && proc_pidinfo(process,PROC_PIDTBSDINFO,0,&info,(int)sizeof info)==(int)sizeof info?
         (uint64_t)info.pbi_start_tvsec*UINT64_C(1000000)+info.pbi_start_tvusec:0;
}
/* the process `pid` of start `start` alive (a start of 0, which a region an earlier bridge or library wrote has: the
   pid alone) */
static inline int mesh_alive(pid_t pid,uint64_t start){ return start?mesh_started(pid)==start:!mesh_exited(pid); }

/* ---- libmesh (mesh-dataflow.c, mesh-call.c) ----------------------------------------------------------------------- */
/* a vacant client slot of region m taken for this process by one compare-exchange (its generation from the header's
   `serial`), stamped with its start; the bridge watches its exit from its event loop's next turn.  0, or EBUSY where
   none is vacant */
int mesh_slot_take(struct mesh_ctx *,struct hdr *);
/* client `client`'s slot left, by its detach or by the bridge where its process has exited: the owner word marked
   MESH_LEAVING (an owner word already marked: its leaving finished), each index it owns left naming no page and
   freed, then the slot vacant; the writes on its ring stay there, each on its own use, for the port children to carry
   (mesh-switched.h mesh_switch_ring_dequeue), its flag kept until they have (mesh_ring_flag) */
void mesh_leave(struct hdr *,uint64_t client);
/* the one join of two nodes' records: the cables each one's bridge reaches the other over, the fewer of the two ends'
   counts (0: none); *beta 8e9 over the least bandwidth either reports of them (0 where neither reports one) */
int mesh_joined(struct mesh_record *,struct mesh_record *,double *beta);
/* `pages` arena pages leased (the bridge's own: map records, the pool's carve), the first a multiple of `align`, all
   within one MESH_DMA_EXTENT, taken first fit by compare-exchange from the context's rover (`arena_rover`): the
   first, or MESH_ABSENT with errno EINVAL or ENOMEM */
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
/* `pages` arena pages from `page` given back */
void mesh_arena_give(struct hdr *,uint32_t page,uint32_t pages);
/* the arena pages no lease has */
uint32_t mesh_arena_free(struct hdr *);

/* The node's region as the bridge makes it: `pages` arena pages of `pgsz` bytes, `block` pages a block (the bridge's
   registered memory), after the header and, each at a multiple of pgsz, the page table's words and owner bytes, the
   arena's bitmap, the clients' write rings, the pool's header and descriptors, the bridge's own pages, the lane
   records, the node table, the switch and the client slots; the arena at a multiple of a block.  Its length */
static inline uint64_t mesh_layout(struct hdr *h,uint32_t pgsz,uint32_t block,uint32_t pages){
  const uint32_t arena=pages/block*block;
  const uint64_t page=pgsz,bytes=(uint64_t)block*pgsz;
  uint64_t at=(sizeof *h+page-1)/page*page;
  h->pgsz=pgsz; h->block=block; h->entries=MESH_TABLE; h->streams=MESH_STREAMS; h->clients=MESH_CLIENTS;
  h->table_off=at; at+=(uint64_t)MESH_TABLE*(sizeof(uint64_t)+sizeof(uint8_t)); at=(at+page-1)/page*page;
  h->arena_off=at; at+=(((uint64_t)arena+63)/64)*sizeof(uint64_t); at=(at+page-1)/page*page;
  h->ring_off=at; at+=(uint64_t)MESH_CLIENTS*sizeof(struct mesh_ring); at=(at+page-1)/page*page;
  h->pool_off=at; at+=sizeof(struct mesh_pool)+(uint64_t)MESH_POOL_PAGES*sizeof(struct mesh_pd); at=(at+page-1)/page*page;
  h->bridge_off=at; at+=MESH_BRIDGE_BYTES; at=(at+page-1)/page*page;
  h->stream_off=at; at+=(uint64_t)MESH_STREAMS*sizeof(struct mesh_lanes); at=(at+page-1)/page*page;
  h->node_off=at; at+=(uint64_t)MESH_NODES*sizeof(struct mesh_node); at=(at+page-1)/page*page;
  h->switch_off=at; at+=MESH_SWITCH_BYTES; at=(at+page-1)/page*page;
  h->client_off=at; at+=(uint64_t)MESH_CLIENTS*sizeof(struct mesh_client);
  at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)arena*pgsz;
  h->length=at; return at;
}
#endif
