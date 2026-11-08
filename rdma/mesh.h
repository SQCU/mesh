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
#define MESH_VERSION 120u
#define MESH_ABSENT UINT32_MAX
#ifndef MESH_PLATFORM_NS
#define MESH_PLATFORM_NS UINT64_C(5000000000)
#endif
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
/* design/algorithm-sources.md#programtensor */
enum { MESH_ROW_OWN, MESH_PLANES };
/* design/algorithm-sources.md#programtensor */
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M02 */
struct mesh_buffer {
  _Alignas(32) _Atomic uint64_t owner;
  _Atomic uint32_t closed;
  uint32_t pages,constant;
  uint64_t start;
};
_Static_assert(sizeof(struct mesh_buffer)==32 && _Alignof(struct mesh_buffer)==32 && offsetof(struct mesh_buffer,closed)==8 &&
  offsetof(struct mesh_buffer,pages)==12 && offsetof(struct mesh_buffer,constant)==16 &&
  offsetof(struct mesh_buffer,start)==24,"mesh_buffer");
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
struct mesh_target { uint64_t word; uint32_t count,stride; };
_Static_assert(sizeof(struct mesh_target)==16,"mesh_target");
/* design/algorithm-sources.md#index-hand-off */
/* design/prepared-machine.md#M10 */
/* rdma-core libibverbs/man/ibv_post_send.3, NOTES; libfabric man/fi_cntr.3.md, DESCRIPTION */
struct mesh_publication { _Alignas(64) uint32_t sends; uint32_t pages; uint64_t phases,count,state;
  struct mesh_target targets[1]; uint64_t completion,padding; };
_Static_assert(sizeof(struct mesh_publication)==64 && _Alignof(struct mesh_publication)==64 &&
  offsetof(struct mesh_publication,sends)==0 && offsetof(struct mesh_publication,pages)==4 &&
  offsetof(struct mesh_publication,phases)==8 && offsetof(struct mesh_publication,count)==16 &&
  offsetof(struct mesh_publication,state)==24 && offsetof(struct mesh_publication,targets)==32 &&
  offsetof(struct mesh_publication,completion)==48,"mesh_publication");
/* M. M. Michael, IEEE TPDS 15(6), 2004 */
struct mesh_port_info { _Atomic uint32_t phase,domain; _Atomic int64_t code; _Atomic uint64_t prepared; };
/* design/page-queue-change-request.md, "Queue pairs and links belong to the bridge" */

#define MESH_PORTS 8u
struct mesh_holder { uint32_t pid,unit; uint64_t start; };
struct mesh_bridge_pages { uint32_t link[MESH_PORTS],ones,map,map_pages,padding; uint64_t bridge_start;
  struct mesh_holder child[2*MESH_PORTS]; };
_Static_assert(sizeof(struct mesh_holder)==16 && offsetof(struct mesh_bridge_pages,bridge_start)==(MESH_PORTS+4)*sizeof(uint32_t),
               "mesh_bridge_pages");

/* design/page-queue-change-request.md, "The page lists (the queue)";
   design/disjoint-pages-change-request.md, Appendix C, C3.1 to C3.4 and C3.7, and Appendix D, D.3.6;
   design/plan-to-bridge-change-request.md R2, R4, R5, R7 and R8;
   design/aot-compilation-change-request.md A7, A9, A10 and appendix items 1, 2, 5 and 6;
   M. Lam, Software pipelining, PLDI 1988; B. R. Rau, Iterative modulo scheduling, MICRO 1994;
   M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2; C.-Y. Chen, arXiv 2205.02474, 2022, LP (14);
   L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997;
   F. B. Schneider, ACM Comput. Surv., 1990; X. Défago, A. Schiper and P. Urbán, ACM Comput. Surv. 36(4), 2004, §7.1;
   L. Lamport, CACM 21(7), 1978, IR1; J. M. Mellor-Crummey and M. L. Scott, ACM TOCS 9(1), 1991, §3.1;
   M. M. Michael, IEEE TPDS 15(6), 2004; P. E. McKenney and J. D. Slingwine, Read-copy update, PDCS 1998;
   K. Fraser, Practical lock-freedom, 2004; rdma-core libibverbs/man/ibv_post_send.3, NOTES;
   libfabric man/fi_cntr.3.md, fi_cntr_readerr, and man/fi_endpoint.3.md, fi_cancel; Portals 4, counting events;
   PTX ISA, mbarrier */
struct mesh_hop { uint32_t page,node,direction,field; uint64_t word,count; uint32_t pages,bytes; uint64_t serial;
  uint32_t set,offset; };
_Static_assert(sizeof(struct mesh_hop)==56 && offsetof(struct mesh_hop,word)==16 && offsetof(struct mesh_hop,count)==24 &&
  offsetof(struct mesh_hop,pages)==32 && offsetof(struct mesh_hop,bytes)==36 && offsetof(struct mesh_hop,serial)==40 &&
  offsetof(struct mesh_hop,set)==48 && offsetof(struct mesh_hop,offset)==52,"mesh_hop");
#define MESH_LANE_BYTES 16u
static inline uint32_t mesh_phase(uint64_t k){ k&=3; return (uint32_t)(k^(k>>1)); }
static inline uint64_t mesh_group_bytes(uint32_t pages){ return ((uint64_t)pages+63)/64*MESH_LANE_BYTES; }
struct mesh_status { _Atomic uint32_t behind,ahead,first,padding; };
_Static_assert(sizeof(struct mesh_status)==16 && offsetof(struct mesh_status,ahead)==4 &&
  offsetof(struct mesh_status,first)==8,"mesh_status");
/* JGroups (https://raw.githubusercontent.com/belaban/JGroups/master/src/org/jgroups/ViewId.java);
   src/org/jgroups/util/UUID.java */
struct mesh_view_id { uint64_t id; uint32_t sequencer,padding; uint64_t boot; };
_Static_assert(sizeof(struct mesh_view_id)==24 && offsetof(struct mesh_view_id,sequencer)==8 &&
  offsetof(struct mesh_view_id,boot)==16,"mesh_view_id");
/* JGroups ViewId.java compareTo and equals, UUID.java compareTo */
static inline int mesh_view_compare(struct mesh_view_id a,struct mesh_view_id b){
  return a.id>b.id?1:a.id<b.id?-1:a.sequencer>b.sequencer?1:a.sequencer<b.sequencer?-1:
    a.boot>b.boot?1:a.boot<b.boot?-1:0;
}
/* design/aot-compilation-change-request.md A7, A9, A10 and appendix item 2; design/plan-to-bridge-change-request.md R2;
   S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, Theorem 2 and §9, Fig. 6;
   JGroups ViewId.java; https://raw.githubusercontent.com/torvalds/linux/master/Documentation/RCU/whatisRCU.rst;
   libfabric man/fi_cntr.3.md; L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997, §2
   (`times`) */
struct mesh_state { _Alignas(64) uint32_t copies; uint32_t receives,ranges,releases;
  uint64_t phases,counts,supplied,status; _Atomic uint64_t executions; uint64_t completions;
  uint64_t group,epoch,serial,hops,entries; _Atomic uint64_t held; _Atomic int32_t refused[MESH_PORTS];
  uint64_t nodes; uint32_t sender,outstanding; uint64_t submitted; _Atomic uint64_t delivered;
  _Atomic uint64_t submissions; uint64_t broadcast,arrived,room,deliveries,submits,announced;
  uint32_t waits,tells; uint64_t syncs; uint32_t calls; _Atomic uint32_t closed; uint64_t rows,submit_room,transfers;
  uint32_t sequencer; _Atomic uint32_t removed; uint64_t boot,times,padding3[2]; };
_Static_assert(sizeof(struct mesh_state)==320 && _Alignof(struct mesh_state)==64 &&
  offsetof(struct mesh_state,receives)==4 && offsetof(struct mesh_state,ranges)==8 && offsetof(struct mesh_state,releases)==12 &&
  offsetof(struct mesh_state,phases)==16 && offsetof(struct mesh_state,counts)==24 &&
  offsetof(struct mesh_state,supplied)==32 && offsetof(struct mesh_state,status)==40 &&
  offsetof(struct mesh_state,executions)==48 && offsetof(struct mesh_state,completions)==56 &&
  offsetof(struct mesh_state,group)==64 && offsetof(struct mesh_state,epoch)==72 && offsetof(struct mesh_state,serial)==80 &&
  offsetof(struct mesh_state,hops)==88 && offsetof(struct mesh_state,entries)==96 && offsetof(struct mesh_state,held)==104 &&
  offsetof(struct mesh_state,refused)==112 && offsetof(struct mesh_state,nodes)==144 && offsetof(struct mesh_state,sender)==152 &&
  offsetof(struct mesh_state,outstanding)==156 && offsetof(struct mesh_state,submitted)==160 &&
  offsetof(struct mesh_state,delivered)==168 && offsetof(struct mesh_state,submissions)==176 &&
  offsetof(struct mesh_state,broadcast)==184 && offsetof(struct mesh_state,arrived)==192 &&
  offsetof(struct mesh_state,room)==200 && offsetof(struct mesh_state,deliveries)==208 &&
  offsetof(struct mesh_state,submits)==216 && offsetof(struct mesh_state,announced)==224 &&
  offsetof(struct mesh_state,waits)==232 && offsetof(struct mesh_state,tells)==236 &&
  offsetof(struct mesh_state,syncs)==240 && offsetof(struct mesh_state,calls)==248 &&
  offsetof(struct mesh_state,closed)==252 && offsetof(struct mesh_state,rows)==256 &&
  offsetof(struct mesh_state,submit_room)==264 && offsetof(struct mesh_state,transfers)==272 &&
  offsetof(struct mesh_state,sequencer)==280 && offsetof(struct mesh_state,removed)==284 &&
  offsetof(struct mesh_state,boot)==288 && offsetof(struct mesh_state,times)==296,"mesh_state");
/* JGroups ViewId.java */
static inline struct mesh_view_id mesh_plan_view(const struct mesh_state *s){
  return (struct mesh_view_id){.id=s->serial,.sequencer=s->sequencer,.boot=s->boot};
}
/* design/aot-compilation-change-request.md A10 and appendix item 2; S. S. Bhattacharyya, S. Sriram and E. A. Lee,
   IEEE Trans. Signal Processing 45(6), 1997, §9, Figs. 6 and 8, eq. (7) */
struct mesh_sync { uint32_t node,function,entry,every; int64_t base; _Atomic uint64_t count; };
_Static_assert(sizeof(struct mesh_sync)==32 && offsetof(struct mesh_sync,entry)==8 && offsetof(struct mesh_sync,every)==12 &&
  offsetof(struct mesh_sync,base)==16 && offsetof(struct mesh_sync,count)==24,"mesh_sync");
/* design/aot-compilation-change-request.md A7 and A9; design/disjoint-pages-change-request.md, Appendix C, C3;
   M. Cowan et al., MSCCLang, ASPLOS 2023, §5;
   L. Lamport, On interprocess communication, Distributed Computing 1(2), 1986;
   P. E. McKenney and J. D. Slingwine, Read-copy update, PDCS 1998 */
struct mesh_algo { _Alignas(64) _Atomic uint64_t published; _Atomic uint64_t retired,released; uint32_t plans,nodes;
  uint64_t group,table,adjacent,padding1; };
_Static_assert(sizeof(struct mesh_algo)==64 && offsetof(struct mesh_algo,retired)==8 &&
  offsetof(struct mesh_algo,released)==16 && offsetof(struct mesh_algo,plans)==24 && offsetof(struct mesh_algo,nodes)==28 &&
  offsetof(struct mesh_algo,group)==32 && offsetof(struct mesh_algo,table)==40 &&
  offsetof(struct mesh_algo,adjacent)==48,"mesh_algo");

/* design/aot-compilation-change-request.md, appendix item 2;
   E. A. Lee and D. G. Messerschmitt, IEEE Trans. Computers C-36(1), 1987 */
struct mesh_delivery { uint64_t sequence,first,end,padding; };
_Static_assert(sizeof(struct mesh_delivery)==32,"mesh_delivery");
struct mesh_submission { uint64_t end,clock; };
_Static_assert(sizeof(struct mesh_submission)==16,"mesh_submission");
/* design/prepared-machine.md#M01 */
/* design/prepared-machine.md#M09 */
struct hdr {
  uint32_t magic,version,pgsz,block,rows,node,vacant0,links;
  uint32_t vacant1,wire_pages;
  uint32_t vacant7;
  uint32_t vacant14;
  uint64_t vacant2;
  uint64_t planes_off,arena_off,page_off,buffer_off,link_off,vacant3,target_off,vacant4,vacant5,data_off,length;
  uint64_t vacant6,target_stride;
  /* design/plan-to-bridge-change-request.md R8 */
  _Atomic uint64_t client,bridge_pid,device_client,serial,control;
  struct mesh_port_info port;
  /* design/aot-compilation-change-request.md A7 */
  _Atomic uint64_t view_map,view;
  uint64_t vacant8;
  uint64_t view_off;
  _Atomic uint64_t view_nodes;
  /* design/disjoint-pages-change-request.md, Appendix A, A3 */
  _Atomic uint64_t map,map_off,map_nodes;
  _Atomic uint64_t map_serial;
  uint64_t bridge_off;
  _Atomic uint64_t client_start;
  uint64_t boot;
  _Atomic uint64_t health,busy,idle_since,stopping;
  uint64_t vacant9,vacant10,vacant11,vacant12,vacant13,map_room;
  uint64_t algo_off;
  _Atomic uint64_t plans;
  /* design/aot-compilation-change-request.md, appendix item 2;
     X. Défago, A. Schiper and P. Urbán, ACM Comput. Surv. 36(4), 2004, §5.2.4 and Fig. 5 */
  uint64_t deliveries_off,deliveries_room;
  _Atomic uint64_t delivered;
  _Atomic uint64_t delivery_read[MESH_PORTS];
  /* JGroups ViewId.java */
  _Atomic uint32_t view_sequencer;
  uint32_t vacant22;
  _Atomic uint64_t view_boot;
};
_Static_assert(offsetof(struct hdr,links)==28 && offsetof(struct hdr,wire_pages)==36 && offsetof(struct hdr,planes_off)==56 &&
  offsetof(struct hdr,target_off)==104 && offsetof(struct hdr,data_off)==128 && offsetof(struct hdr,target_stride)==152 &&
  offsetof(struct hdr,client)==160 && offsetof(struct hdr,view_map)==224 && offsetof(struct hdr,view)==232 &&
  offsetof(struct hdr,map_serial)==288 &&
  offsetof(struct hdr,bridge_off)==296 && offsetof(struct hdr,client_start)==304 && offsetof(struct hdr,stopping)==344 &&
  offsetof(struct hdr,vacant7)==40 && offsetof(struct hdr,vacant14)==44 && offsetof(struct hdr,vacant8)==240 &&
  offsetof(struct hdr,view_off)==248 && offsetof(struct hdr,view_nodes)==256 && offsetof(struct hdr,map_off)==272 &&
  offsetof(struct hdr,vacant9)==352 && offsetof(struct hdr,vacant13)==384 && offsetof(struct hdr,map_room)==392 &&
  offsetof(struct hdr,algo_off)==400 && offsetof(struct hdr,plans)==408 && offsetof(struct hdr,deliveries_off)==416 &&
  offsetof(struct hdr,delivered)==432 && offsetof(struct hdr,delivery_read)==440 &&
  offsetof(struct hdr,view_sequencer)==504 && offsetof(struct hdr,view_boot)==512 && sizeof(struct hdr)==520,
  "struct hdr: no field moves");
/* JGroups ViewId.java */
static inline struct mesh_view_id mesh_view_installed(struct hdr *m,memory_order order){
  return (struct mesh_view_id){.id=atomic_load_explicit(&m->view,order),
                               .sequencer=atomic_load_explicit(&m->view_sequencer,order),
                               .boot=atomic_load_explicit(&m->view_boot,order)};
}
/* design/prepared-machine.md#M26 */
_Static_assert(sizeof(((struct hdr *)0)->control)==8 && offsetof(struct hdr,control)%8==0,"M26");
/* design/prepared-machine.md#M07 */
_Static_assert(sizeof(_Atomic uint64_t)==8 && _Alignof(_Atomic uint64_t)==8,"M07");
/* design/algorithm-sources.md#meshresult */
/* design/prepared-machine.md#M26 */
static inline void mesh_control_notify(struct hdr *m){
  atomic_fetch_add_explicit(&m->control,1,memory_order_release);
  os_sync_wake_by_address_all(&m->control,sizeof m->control,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
/* design/algorithm-sources.md#meshresult */
static inline void mesh_notify(_Atomic uint64_t *word){
  atomic_fetch_add_explicit(word,1,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(word,sizeof *word,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}
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
static inline _Atomic uint64_t *mesh_arena_bits(struct hdr *m){ return (_Atomic uint64_t*)((unsigned char*)m+m->arena_off); }
/* design/prepared-machine.md#M09 */
static inline uint64_t mesh_wire_bytes(const struct hdr *m){ return (uint64_t)m->wire_pages*m->pgsz; }
static inline int mesh_wired(const struct hdr *m,uint64_t offset,uint64_t bytes){ return offset<=mesh_wire_bytes(m) && bytes<=mesh_wire_bytes(m)-offset; }
static inline struct mesh_page_entry *mesh_page(struct hdr *m){ return (struct mesh_page_entry*)((unsigned char*)m+m->page_off); }
/* design/algorithm-sources.md#programtensor */
static inline struct mesh_buffer *mesh_buffers(struct hdr *m){ return (struct mesh_buffer *)((char *)m+m->buffer_off); }
static inline unsigned char *mesh_at(struct hdr *m,uint32_t page){ return (unsigned char*)m+m->data_off+(size_t)page*m->pgsz; }
static inline struct mesh_hop *mesh_hops(struct hdr *m,const struct mesh_state *state){
  return (struct mesh_hop *)((char *)m+state->hops);
}
static inline struct mesh_state *mesh_state_at(struct hdr *m,uint64_t at){
  return at?(struct mesh_state *)((char *)m+at):NULL;
}
static inline struct mesh_algo *mesh_algo(struct hdr *m){ return (struct mesh_algo *)((char *)m+m->algo_off); }
static inline struct mesh_state *mesh_algo_state(struct hdr *m,const struct mesh_algo *algo,uint32_t k){
  return mesh_state_at(m,((const uint64_t *)(const void *)((const char *)m+algo->table))[k]);
}
/* design/aot-compilation-change-request.md A9; P. E. McKenney and J. D. Slingwine, Read-copy update, PDCS 1998 */
static inline uint64_t mesh_algo_unreleased(struct hdr *m){
  const struct mesh_algo *algo=mesh_algo(m);
  const uint64_t published=atomic_load_explicit(&algo->published,memory_order_acquire);
  return published && atomic_load_explicit(&algo->released,memory_order_acquire)!=published?published:0;
}
static inline uint64_t mesh_algo_open(struct hdr *m){
  const uint64_t published=mesh_algo_unreleased(m);
  return published && atomic_load_explicit(&mesh_algo(m)->retired,memory_order_acquire)!=published?published:0;
}
static inline struct mesh_state *mesh_algo_standing(struct hdr *m,uint32_t k){
  struct mesh_state *state=mesh_algo_state(m,mesh_algo(m),k);
  return !atomic_load_explicit(&state->closed,memory_order_acquire)?state:NULL;
}
static inline struct mesh_state *mesh_algo_plan(struct hdr *m,uint32_t k){
  return mesh_algo_open(m)?mesh_algo_standing(m,k):NULL;
}
static inline const uint32_t *mesh_algo_adjacent(struct hdr *m,uint32_t i,uint32_t *count){
  const struct mesh_algo *algo=mesh_algo(m);
  const uint32_t *first=(const uint32_t *)(const void *)((const char *)m+algo->adjacent);
  *count=first[i+1]-first[i];
  return first+algo->nodes+1+first[i];
}
static inline const struct mesh_publication *mesh_transfer_at(struct hdr *m,const struct mesh_state *state,
                                                              uint32_t transfer){
  return (const struct mesh_publication *)(const void *)((const char *)m+state->transfers+(size_t)transfer*64);
}
static inline struct mesh_sync *mesh_syncs(struct hdr *m,const struct mesh_state *state){
  return (struct mesh_sync *)(void *)((char *)m+state->syncs);
}
static inline int mesh_seqlock_whole(const _Atomic uint64_t *lock,uint64_t before){
  atomic_thread_fence(memory_order_acquire);
  return !(before&1) && atomic_load_explicit(lock,memory_order_relaxed)==before;
}
static inline int mesh_map_whole(struct hdr *m,uint64_t before){ return mesh_seqlock_whole(&m->map,before); }
static inline struct mesh_bridge_pages *mesh_bridge_pages(struct hdr *m){
  return (struct mesh_bridge_pages *)((char *)m+m->bridge_off);
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
#endif
