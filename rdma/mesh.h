#ifndef MESH_H
#define MESH_H
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
#include <infiniband/verbs.h>
#include <os/os_sync_wait_on_address.h>
/* design/pages-and-functions.md#block-addressing */
#define MESH_MAGIC 0x4d455348u
#define MESH_NAME "/mesh0"
#define MESH_PORT "18519"
#define MESH_MODE 0666
/* 105: the page lists (struct mesh_hop) and the bridge's pages (struct mesh_bridge_pages) in the layout, the map's
   fields (struct mesh_record) in the header, `executions` in place of its padding
   (design/page-queue-change-request.md).  106: the tag's header fields `boot`, `health`, `busy`, `idle_since`,
   `stopping` appended */
#define MESH_VERSION 106u
#define MESH_ABSENT UINT32_MAX
/* tag pre-return-20261103 etc/bridge/layout.h MESH_PLATFORM_NS, verbatim: a wait's deadline where its caller gives
   none; the operator's deadline (2026-10-18): no DMA or network send/receive channel of the mesh is used longer than this
   without something happening on it or its scope concluding */
#ifndef MESH_PLATFORM_NS
#define MESH_PLATFORM_NS UINT64_C(5000000000)
#endif
/* design/collective-dependency-ledger.md#d6-paired-send-and-receive-frame-counts-match */
#define MESH_QPS 8
/* replaced by struct mesh_hop (the page lists); removed at sub-assignment 16 */
struct mesh_transfer { uint32_t local_row,binding,count,stride,invocation_pages,first; uint64_t bytes; };
_Static_assert(sizeof(struct mesh_transfer)==32,"M08 prepared transfer");
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
/* design/algorithm-sources.md#programtensor */
enum { MESH_ROW_OWN, MESH_ROW_HOT, MESH_FREE, MESH_PLANES };
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
struct mesh_buffer {
  _Alignas(32) _Atomic uint64_t owner;
  _Atomic uint32_t closed;
  uint32_t pages,constant;
};
_Static_assert(sizeof(struct mesh_buffer)==32 && _Alignof(struct mesh_buffer)==32,"mesh_buffer");
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M10 */
struct mesh_page_entry {
  _Alignas(16) _Atomic uint64_t mapping;
  _Atomic uintptr_t address;
  _Atomic uint64_t device;
};
_Static_assert(sizeof(struct mesh_page_entry)==32 && _Alignof(struct mesh_page_entry)==16 &&
  offsetof(struct mesh_page_entry,address)==8 && offsetof(struct mesh_page_entry,device)==16,"mesh_page_entry");
/* design/collective-dependency-ledger.md#d5-receive-consumption-has-per-queue-fifo-order */
enum { MESH_SEND, MESH_RECEIVE };
#define MESH_NOTICE_BANKS 2
/* design/prepared-machine.md#M04 */
struct mesh_send {
  _Alignas(128) _Atomic uint64_t ready;
  uintptr_t pair;
  _Alignas(32) struct ibv_sge span;
  struct ibv_send_wr request;
};
struct mesh_tx { uint32_t count,slots,once,invocations; uint64_t cancel,cells; };
_Static_assert(sizeof(struct mesh_send)==256 && _Alignof(struct mesh_send)==128 &&
  offsetof(struct mesh_send,span)==32 && offsetof(struct mesh_send,request)==48 &&
  offsetof(struct mesh_send,request.send_flags)+sizeof(unsigned int)<=128,"M04/M29");
_Static_assert(offsetof(struct mesh_tx,cancel)==16 && offsetof(struct mesh_tx,cells)==24 && sizeof(struct mesh_tx)==32,"M04/M12");
struct mesh_target { uint64_t stream; uint32_t count,stride; };
_Static_assert(sizeof(struct mesh_target)==16,"mesh_target");
/* design/algorithm-sources.md#index-hand-off */
/* design/prepared-machine.md#M10 */
struct mesh_publication { _Alignas(64) uint32_t sends; _Atomic uint64_t argument; uint64_t device_input,device_stride; struct mesh_target targets[]; };
_Static_assert(sizeof(struct mesh_publication)==64 && _Alignof(struct mesh_publication)==64 && offsetof(struct mesh_publication,argument)==8 && offsetof(struct mesh_publication,device_input)==16 && offsetof(struct mesh_publication,device_stride)==24 && offsetof(struct mesh_publication,targets)==32,"mesh_publication");
/* design/prepared-machine.md#M13 */
struct prepared_publication { _Alignas(16) uint64_t destination; uint64_t argument,padding[2]; };
_Static_assert(sizeof(struct prepared_publication)==32 && _Alignof(struct prepared_publication)==16 && offsetof(struct prepared_publication,argument)==8,"M13");
struct mesh_port_info { _Atomic uint32_t phase,domain; _Atomic int64_t code; _Atomic uint64_t prepared; };
/* a port's link, at `link_off`: the resident bridge's MESH_PORTS of them (MESH_UNKNOWN where it has no such port), the
   --link form's `links` */
struct mesh_link_info { uint32_t peer; char device[32]; uint64_t bandwidth; struct mesh_port_info port; };

/* tag pre-return-20261103 rdma/mesh-observe.h MESH_PORTS: the most ports (there MESH_STREAMS / MESH_LINK_STREAMS) */
#define MESH_PORTS 8u
/* tag pre-return-20261103 rdma/mesh-observe.h struct mesh_holder, verbatim */
/* a holder of the region (a port's child): its pid, its device (`unit`: its interface's unit + 1; 0, any, as builds
   before 8ac5c9fb's form wrote) and its start (mesh-holders.h mesh_started) */
struct mesh_holder { uint32_t pid,unit; uint64_t start; };
/* tag pre-return-20261103 rdma/mesh-observe.h struct mesh_bridge_pages, verbatim, at the layout's bridge area
   (`bridge_off`) rather than after the tag's MESH_PORTS link infos */
/* the bridge's own pages of the region, recorded in the bridge area so the next bridge adopts them with the region
   instead of leasing them again: each port's link pages and its streams' lists, the duds' block and the map's pages
   (MESH_ABSENT where none); and the processes of the bridge that hold the region, each by its pid and start
   (mesh_started): the bridge's own start (beside the header's `bridge_pid`) and each child, which registers the region
   and holds it until its exit, in an entry no live child holds (twice the ports: an earlier launch's child that has not
   ended keeps its entry), so the heartbeat unlinks an expired region only once none of them lives.  Fields are only
   ever appended */
struct mesh_bridge_pages { uint32_t link[MESH_PORTS],ones,map,map_pages,padding; uint64_t bridge_start;
  struct mesh_holder child[2*MESH_PORTS]; };
_Static_assert(sizeof(struct mesh_holder)==16 && offsetof(struct mesh_bridge_pages,bridge_start)==(MESH_PORTS+4)*sizeof(uint32_t),
               "mesh_bridge_pages");

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
   id.  The bridges' rule reads cables (struct mesh_hop, "Both ends"); the collective reads only node pairs and each
   pair's capacity, the sum of its cables' bandwidth, never a port or queue pair. */
struct mesh_reach { uint32_t node,port,peer_port,qps; uint64_t bandwidth; };
_Static_assert(sizeof(struct mesh_reach)==24,"mesh_reach");
/* tag pre-return-20261103 rdma/mesh.h struct mesh_record, verbatim but `reach`: MESH_PORTS cables, not MESH_STREAMS
   streams */
/* a node's record, the newest of which every bridge floods to the mesh: its node and boot, the record's sequence in
   that boot, the bytes of its bridge's region its clients may hold (the region's arena less what the bridge holds
   itself: struct mesh_bridge_pages), the node's memory in bytes and hardware model (hw.model, MESH_MODEL_BYTES with its NUL), and the `reached`
   cables of its node (struct mesh_reach).  No host name: a name a person reads is a label the readers take from the
   name table by node number (etc/mesh-nodes.json, rdma/peers.py label; AGENTS.md "A node is its hardware; names are
   virtual") */
struct mesh_record { uint32_t node,reached; uint64_t boot,sequence,region,memory;
  char model[MESH_MODEL_BYTES]; struct mesh_reach reach[MESH_PORTS]; struct mesh_health health; };

/* The page lists (design/page-queue-change-request.md, "The page lists (the queue)"): this node's hops, which the plan
   call writes once a plan and the bridge alone reads, in place of the per-link, per-queue transfer lists (struct
   mesh_transfer, mesh_order_length).  An entry is one hop of one page:
   - `page`: an arena page of this region (mesh_at), a multiple of `block`, its block within the registered window
     (mesh_wired): the page sent, the page a receive lands in, or a relay's page.  A hop moves that whole page, `block`
     arena pages (block x pgsz bytes), from it or into it; nothing is copied.
   - `node`: the adjacent node by its index in the map (hdr `map_serial`): a send's next hop, a receive's previous hop.
   - `direction`: MESH_SEND or MESH_RECEIVE.
   - `forwards`, the plan call's, so the bridge matches no entries: on a receive, its page's forwards, the send hops of
     the slot that name its landed word; on a send, 1 where it is a forwarding SEND (its word a landed word of a
     receive at this node), else 0.
   - `word`: the region offset of the hop's word of execution 0, an _Atomic uint64_t (8-aligned, anywhere in the
     region: no SGE names it); `position`: the value that releases a send or that a receive's landing stores,
     0 < position < UINT64_MAX.
   - `serial`: the map serial the plan was made against, every entry's the same.
   Executions (main's invocations and ring depth: mesh-call.c mesh_transfers_prepare, mesh-flow.c link_configure,
   mesh-metal.m).  A plan runs hdr `executions` times, t = 0 .. executions - 1, its pages ringing through hdr `depth`
   slots (depth == executions: no ring).  These and hdr `hop_stride` are the plan's values, beside the list.
   - The list is `hops` = depth x n entries, slot s's n at [s n, s n + n).  Every slot holds the same n hops in the same
     order, with the same node, direction, forwards, word, position and serial; only the pages differ.
   - Which page execution t uses: slot (t mod depth)'s page of the hop.  A hop's receive of execution t is posted only
     once its receives of executions 0 .. t - (depth - 1) have landed (main's receive window, a slot short of the ring)
     and, from t = depth, once its cell of execution t - depth reads at offset 8 (acquire) its `forwards` (below), so no
     two posted receives name one page and no landing overwrites a page a forwarding SEND still reads.
   - Execution t's word of an entry is offset 0 of its cell of execution t: the 256-byte cell at region offset `word`
     + t x `hop_stride`, `hop_stride` 256 for every word of the plan, no two words' cells overlapping.  This is the
     GPU's word protocol as it is: its stores and waits are at a binding's offset + t x stride, and here the
     publication's stride (mesh_metal_publication `stride`, which its completion statuses share for the GPU's direct
     publication) and the receive's (mesh_metal_input `stride`) are `hop_stride`, main's 256 (struct mesh_send, struct
     mesh_output_status).  Every cell of every execution reads 0 when the plan is loaded; a word is written once and
     never cleared.
   - A landed word's cell of execution t holds at offset 8 (an _Atomic uint64_t) its page's sent count of t: the
     forwarding SENDs of t from that page whose completion was taken, each port's child adding 1 (release) as it takes
     one of a send whose `forwards` is 1.  A forwarding SEND is a send hop whose `word` is a receive's landed word, as
     a relay's are; the receive's `forwards` counts them.
   - What releases a send in execution t: its word of execution t reaching `position` (acquire; position <= value <
     UINT64_MAX).  That word is a supplied word, stored with release by its producer after its last write to the page:
     the GPU's publication store (prepared_publication `argument` = `position`) or mesh_host_publish; or, at a relay,
     the landed word of the hop relayed: a relay's send names its receive's word and position.  Several sends may name
     one word (a range's pages released by one store, a page sent to several nodes).
   - What a receive's landing stores in execution t: `position`, with release, into its word of execution t, once the
     page's bytes are in its page.  A landed word (offset 0) has that one writer: no other receive of a slot and no
     producer names it, so it tells of one page.  A reader of several pages waits on each one's word (a GPU range a
     page): a transfer's pages spread over every cable and queue pair and land in any order, unlike main's chunks on
     one queue pair, whose last word stood for all (mesh_host_arrived, mesh_metal_receive_prepare).
   - UINT64_MAX in a word is cancellation (main's mesh_cancel; the GPU's waits end on ~0): it releases nothing.  The
     bridge stores it into this node's landed words not landed when it ends the plan's executions.
   Both ends: for adjacent nodes a and b, a slot's entries at a naming b with MESH_SEND and those at b naming a with
   MESH_RECEIVE are the same hops in the same order.  Both bridges assign the pair's hops of every execution in turn
   (execution 0's in list order, then execution 1's, ...) to the pair's cables and queue pairs by one rule from the map
   (struct mesh_reach; design/page-queue-change-request.md, "The rule both ends compute"), and each queue pair carries
   its hops in that order: the k-th SEND on a queue pair fills the k-th receive posted on it.
   Writing a plan: the plan call stores `hops` 0 (release), writes the entries, `executions`, `depth`, `hop_stride` and
   `plan`, the plan's generation (the last plan's + 1, never 0), then stores `hops` (release), at most `hop_capacity`;
   the bridge reads none of it while `hops` (acquire) is 0, and stores the generation it loaded in each port's
   mesh_links(m)[p].port.prepared.  A plan whose entries' `serial` is not `map_serial` when it is loaded is refused:
   the call fails by error and its caller plans again on the map as it is. */
struct mesh_hop { uint32_t page,node,direction,forwards; uint64_t word,position,serial; };
_Static_assert(sizeof(struct mesh_hop)==40 && offsetof(struct mesh_hop,word)==16,"mesh_hop");

/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M09 */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,qps,links;
  uint32_t orders,wire_pages;
  /* the ring depth of the transfer lists and of the page lists (struct mesh_hop): execution t uses slot t mod depth */
  _Atomic uint32_t depth;
  /* the page lists: the executions of the plan written (struct mesh_hop), main's invocations */
  uint32_t executions;
  _Atomic uint64_t configured;
  /* length_off, order_off: the per-link, per-queue transfer lists, replaced by hop_off; removed at sub-assignment 16 */
  uint64_t planes_off,arena_off,page_off,buffer_off,link_off,length_off,target_off,order_off,notice_off,data_off,length;
  uint64_t notice_bytes,target_stride;
  _Atomic uint64_t client,bridge_pid,device_client,serial,control;
  struct mesh_port_info port;
  /* the page lists (struct mesh_hop, mesh_hops): room for `hop_capacity` entries at `hop_off` (mesh_geometry `hops`),
     the plan's word stride `hop_stride`, `hops` the entries of the plan written (0 none) and `plan` its generation (the
     last plan's + 1, never 0; each port's mesh_links(m)[p].port.prepared the generation the bridge loaded) */
  uint64_t hop_off,hop_capacity,hop_stride;
  _Atomic uint64_t hops,plan;
  /* tag pre-return-20261103 rdma/mesh.h struct hdr `map`, `map_off`, `map_nodes`, verbatim: the map's seqlock
     (mesh_notify, mesh_map_whole) and its `map_nodes` records at `map_off` (mesh_records) */
  _Atomic uint64_t map,map_off,map_nodes;
  /* the map's serial: moved by one (mesh-records.c, while `map` is odd) at every change of the map's nodes or cables
     (a record added or dropped, any field of a reach entry); a plan is made against it (struct mesh_hop `serial`).
     A node's index is its place in ascending node-id order among the map's records. */
  _Atomic uint64_t map_serial;
  /* the bridge's pages (struct mesh_bridge_pages) at `bridge_off`; the attached client's start (mesh-holders.h
     mesh_started), with `client`'s low 32 bits its identity as a holder of the region */
  uint64_t bridge_off;
  _Atomic uint64_t client_start;
  /* tag pre-return-20261103 rdma/mesh.h struct hdr `boot`, `health`, `busy`, `idle_since`, `stopping`, verbatim (their
     tag types; read by the tag's mesh-records.c and mesh-link.c region_open and main): `boot` the bridge's start on the
     wall clock (ns, CLOCK_REALTIME), its records' boot; `health` the serial a health-only change of a record moves,
     never the map's (mesh-records.c health_put); `busy`: the streams held (0 between calls) and since when none has
     been (`idle_since`, the wall clock in ns; 0 while any is); `stopping` stored 0 as the region opens */
  uint64_t boot;
  _Atomic uint64_t health,busy,idle_since,stopping;
};
/* design/prepared-machine.md#M26 */
_Static_assert(sizeof(((struct hdr *)0)->control)==8 && offsetof(struct hdr,control)%8==0,"M26");
/* design/prepared-machine.md#M07 */
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");
/* design/prepared-machine.md#M12 */
struct mesh_cancel_range { _Alignas(32) uint64_t offset; uint64_t count,padding[2]; };
struct mesh_cancellation { _Alignas(32) _Atomic uint32_t requested; uint32_t padding[7]; struct mesh_cancel_range ranges[]; };
_Static_assert(sizeof(struct mesh_cancel_range)==32 && sizeof(struct mesh_cancellation)==32 && offsetof(struct mesh_cancellation,ranges)==32,"M12");
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M12 */
static inline void mesh_cancel(struct hdr *m,struct mesh_cancellation *cancel,uint32_t link){
  _Atomic uint64_t *words=(void *)((char *)m+cancel->ranges[link].offset);
  for(uint64_t j=0;j<cancel->ranges[link].count;j++)
    if(!atomic_load_explicit(words+j,memory_order_relaxed)){
      atomic_store_explicit(&cancel->requested,1,memory_order_release);
      atomic_store_explicit(words+j,UINT64_MAX,memory_order_release);
    }
}
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M26 */
static inline void mesh_control_notify(struct hdr *m){
  atomic_fetch_add_explicit(&m->control,1,memory_order_release);
  os_sync_wake_by_address_all(&m->control,sizeof m->control,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* tag pre-return-20261103 rdma/mesh.h mesh_notify, verbatim */
/* design/algorithm-sources.md#meshresult: a shared word (`control`, `map`) bumped and its waiters woken, the writes
   after it ordered behind it */
static inline void mesh_notify(_Atomic uint64_t *word){
  atomic_fetch_add_explicit(word,1,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* design/algorithm-sources.md#programtensor */
static inline uint32_t mesh_notice_queue(struct hdr *m,uint64_t owner,uint32_t queue){return (uint32_t)(owner>>63)*m->links+queue;}
/* design/algorithm-sources.md#programcopy */
static inline struct mesh_link_info *mesh_links(struct hdr *m){return (struct mesh_link_info *)((char *)m+m->link_off);}
/* design/algorithm-sources.md#index-hand-off */
static inline struct mesh_publication *mesh_publication_at(struct hdr *m,uint32_t row){return (struct mesh_publication *)((char *)m+m->target_off+(size_t)row*m->target_stride);}
static inline uint32_t mesh_rows(const struct hdr *m){ return m->rows; }
static inline uint32_t mesh_words(const struct hdr *m){ return (mesh_rows(m)+63)/64; }
/* design/prepared-machine.md#M09 */
static inline uint32_t mesh_blocks(const struct hdr *m){ return (uint32_t)((m->length-m->data_off)/((uint64_t)m->block*m->pgsz)); }
/* design/pages-and-functions.md#what-the-page-table-is */
static inline uint32_t mesh_arena_pages(const struct hdr *m){ return mesh_blocks(m)*m->block; }
static inline _Atomic uint64_t *mesh_plane(struct hdr *m,int plane){ return (_Atomic uint64_t*)((unsigned char*)m+m->planes_off)+(size_t)plane*mesh_words(m); }
/* design/prepared-machine.md#M01 */
/* the arena occupancy bitmap is indexed by arena page and sized by arena pages: it is not a row plane */
static inline _Atomic uint64_t *mesh_arena_bits(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->arena_off); }
/* design/prepared-machine.md#M09 */
/* the registered window is the first wire_pages of the arena; every other page is addressable and unregistered */
static inline uint64_t mesh_wire_bytes(const struct hdr *m){ return (uint64_t)m->wire_pages*m->pgsz; }
static inline int mesh_wired(const struct hdr *m,uint64_t offset,uint64_t bytes){ return offset<=mesh_wire_bytes(m) && bytes<=mesh_wire_bytes(m)-offset; }
static inline struct mesh_page_entry *mesh_page(struct hdr *m){ return (struct mesh_page_entry*)((unsigned char*)m+m->page_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
/* ledger D5 */
/* one index for the transfer list and for its length: both are (bank,queue,direction), both stride by the
   runtime queue count, so a client built with another MESH_QPS cannot disagree with the bridge about either */
/* replaced by mesh_hops (the page lists); removed at sub-assignment 16 */
static inline size_t mesh_order_index(const struct hdr *m,uint64_t owner,uint32_t queue,int direction){
  return (((size_t)(owner>>63)*m->links*m->qps)+queue)*2+(uint32_t)direction;
}
/* replaced by mesh_hops (the page lists); removed at sub-assignment 16 */
static inline struct mesh_transfer *mesh_transfers(struct hdr *m,uint64_t owner,uint32_t queue,int direction){ return (struct mesh_transfer*)((unsigned char*)m+m->order_off)+mesh_order_index(m,owner,queue,direction)*m->orders; }
/* replaced by hdr `hops` (the entries of the plan written); removed at sub-assignment 16 */
static inline _Atomic uint32_t *mesh_order_length(struct hdr *m,uint64_t owner,uint32_t queue,int direction){ return (_Atomic uint32_t*)((unsigned char*)m+m->length_off)+mesh_order_index(m,owner,queue,direction); }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
/* the page lists: the plan's entries (struct mesh_hop), hdr `hops` of them at `hop_off` */
static inline struct mesh_hop *mesh_hops(struct hdr *m){ return (struct mesh_hop *)((char *)m+m->hop_off); }
/* tag pre-return-20261103 rdma/mesh.h mesh_records and mesh_map_whole, verbatim */
/* the map, `map_nodes` records at `map_off` in arena pages the bridge leases for itself, and leases again larger as
   the mesh grows; the bridge notifies `map`, its serial, before and after each change (odd while it writes; its own
   record is in before the region opens, so a client never reads 0), and a reader's copy is whole where `map` read
   (acquired) before it was even and is unchanged after it */
static inline struct mesh_record *mesh_records(struct hdr *m){ return (struct mesh_record *)((char *)m+atomic_load_explicit(&m->map_off,memory_order_relaxed)); }
static inline int mesh_map_whole(struct hdr *m,uint64_t before){
  atomic_thread_fence(memory_order_acquire);
  return !(before&1) && atomic_load_explicit(&m->map,memory_order_relaxed)==before;
}
/* tag pre-return-20261103 rdma/mesh-observe.h mesh_bridge_pages, at the layout's bridge area (`bridge_off`), no link
   infos before it */
static inline struct mesh_bridge_pages *mesh_bridge_pages(struct hdr *m){
  return (struct mesh_bridge_pages *)((char *)m+m->bridge_off);
}
/* design/algorithm-sources.md#programcopy */
/* design/prepared-machine.md#M08 */
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

/* design/algorithm-sources.md#index-hand-off */
static inline void *mesh_events(struct hdr *m,uint32_t queue){
  return (char *)m+m->notice_off+(uint64_t)queue*m->notice_bytes;
}

/* design/algorithm-sources.md#programkernel_call */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M08 */
/* Four independent dimensions, none of them derived from another: arena pages, page-table rows,
   transfer-list entries (orders) and the registered window.  rows was welded to pages because the
   arena bitmap lived in a row plane and the order table was rows-strided; both are now their own
   table, so a 512 GiB arena costs a 4 MiB bitmap and nothing else.  `hops`: the page lists' room
   (struct mesh_hop); absent (0), `rows`, as `orders`. */
/* links x qps: the count of the per-link, per-queue transfer lists, replaced by hops; removed at sub-assignment 16 */
struct mesh_geometry { uint32_t pgsz,block,pages,rows,orders,links,qps,wire_pages,hops; };
static inline uint64_t mesh_layout(struct hdr *h,struct mesh_geometry g){
  uint32_t blocks=g.pages/g.block,arena=blocks*g.block;
  if(!g.rows)g.rows=arena;
  if(!g.orders)g.orders=g.rows;
  if(!g.hops)g.hops=g.rows;
  if(!g.wire_pages || g.wire_pages>arena)g.wire_pages=arena;
  g.wire_pages=(g.wire_pages+g.block-1)/g.block*g.block;
  if(g.wire_pages>arena)g.wire_pages=arena;
  uint32_t pgsz=g.pgsz;
  uint64_t at=(sizeof *h+pgsz-1)/pgsz*pgsz,words=((uint64_t)g.rows+63)/64;
  h->pgsz=pgsz; h->block=g.block; h->rows=g.rows; h->orders=g.orders; h->wire_pages=g.wire_pages;
  h->links=g.links; h->qps=g.qps;
  h->planes_off=at; at+=(uint64_t)MESH_PLANES*words*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->arena_off=at; at+=(((uint64_t)arena+63)/64)*sizeof(uint64_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->page_off=at; at+=(uint64_t)g.rows*sizeof(struct mesh_page_entry); at=(at+pgsz-1)/pgsz*pgsz;
  h->buffer_off=at; at+=(uint64_t)g.rows*sizeof(struct mesh_buffer); at=(at+pgsz-1)/pgsz*pgsz;
  h->link_off=at; at+=(uint64_t)(g.links>MESH_PORTS?g.links:MESH_PORTS)*sizeof(struct mesh_link_info); at=(at+pgsz-1)/pgsz*pgsz;
  /* replaced by hop_off (the page lists); removed at sub-assignment 16 */
  h->length_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*2*g.links*g.qps*sizeof(uint32_t); at=(at+pgsz-1)/pgsz*pgsz;
  h->target_stride=(offsetof(struct mesh_publication,targets)+(uint64_t)g.links*sizeof(struct mesh_target)+63)&~UINT64_C(63);
  h->target_off=at; at+=(uint64_t)g.rows*h->target_stride; at=(at+pgsz-1)/pgsz*pgsz;
  /* replaced by hop_off (the page lists); removed at sub-assignment 16 */
  h->order_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*2*g.links*g.qps*g.orders*sizeof(struct mesh_transfer); at=(at+pgsz-1)/pgsz*pgsz;
  h->hop_off=at; h->hop_capacity=g.hops; at+=(uint64_t)g.hops*sizeof(struct mesh_hop); at=(at+pgsz-1)/pgsz*pgsz;
  h->bridge_off=at; at+=sizeof(struct mesh_bridge_pages); at=(at+pgsz-1)/pgsz*pgsz;
  h->notice_bytes=sizeof(struct mesh_tx);
  uint64_t bytes=(uint64_t)g.block*pgsz; at=(at+bytes-1)/bytes*bytes;
  h->notice_off=at; at+=(uint64_t)MESH_NOTICE_BANKS*g.links*h->notice_bytes; at=(at+bytes-1)/bytes*bytes;
  h->data_off=at; at+=(uint64_t)g.pages*pgsz;
  h->length=at; return at;
}
#endif
