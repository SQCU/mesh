#ifndef MESH_BRIDGE_H
#define MESH_BRIDGE_H
#include "mesh-call.h"
#include "mesh-observe.h"
#include <infiniband/verbs.h>
#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <arpa/inet.h>
#include <crt_externs.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <mach-o/getsect.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <net/if.h>
#include <net/if_var.h>
#include <net/net_kev.h>
#include <os/os_sync_wait_on_address.h>
#include <poll.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/event.h>
#include <sys/ioctl.h>
#include <sys/kern_event.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <sysexits.h>
#include <unistd.h>
/* design/transport.md section 14, "The wire"; design/HETEROGENEOUS-LINKS.md section 5 */
#define MESH_LINK_LANES 10u
_Static_assert(MESH_LINK_LANES<32,"a bit a lane");
static inline uint32_t link_plan(uint32_t budget){ return (UINT32_C(1)<<(budget<MESH_LINK_LANES?budget:MESH_LINK_LANES))-1; }
static inline uint32_t plan_lanes(uint32_t plan){ return (uint32_t)__builtin_popcount(plan); }
#define MESH_FRAME 4096u
struct frames { uint64_t n; };
static inline struct frames frames_of(uint64_t bytes){ return (struct frames){(bytes+MESH_FRAME-1)/MESH_FRAME}; }
static inline struct frames frames_min(struct frames a,struct frames b){ return a.n<b.n?a:b; }
#define MESH_RUN 32u
struct key_unit { uint64_t key[5]; uint32_t offset,count,total,padding; uint64_t spare; };
_Static_assert(sizeof(struct key_unit)==64 && sizeof(struct key_unit)<=MESH_FRAME,"a key unit one frame");
#define MESH_KEY_SLOTS 128u
/* design/transport.md section 14, "Relays" */
#define MESH_RELAYS 64u
#define MESH_POOL_BLOCKS 256u
enum { RELAY_FREE, RELAY_LANDING, RELAY_READY, RELAY_SENDING };
static inline uint32_t relay_sending(uint32_t port){ return RELAY_SENDING|(port+1)<<8; }
struct relay { _Atomic uint32_t state; uint32_t lane,order,total; uint64_t key[5]; uint32_t offset,count,page,node; _Atomic uint32_t landed,sent; };
_Static_assert(sizeof(struct relay)==80,"a relay entry");
/* design/transport.md section 14, "Plug to link, at the hardware's pace" */
union mesh_network_event {
  struct kern_event_msg header;
  unsigned char bytes[KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data)];
};
_Static_assert(sizeof(union mesh_network_event)==48,"M24");
/* design/transport.md section 14, "Rows"; section 15, "The SEND gate" */
#define MESH_RING 256u
enum { UNIT_KEY=1, UNIT_PAGE, UNIT_RELAY, UNIT_SINK };
struct unit { uint64_t table,entry,relay,k; uint32_t kind,slot; };
struct run_record { struct unit unit; uint64_t start; uint32_t count,padding; };
struct lane_record { _Atomic uint64_t posted[2],done[2]; struct run_record run[2]; struct unit ring[2][MESH_RING]; };
static inline uint64_t link_keys_at(uint64_t page){ return page; }
static inline uint64_t link_heard_at(uint64_t page){ return page+(uint64_t)MESH_LINK_LANES*MESH_KEY_SLOTS*sizeof(struct key_unit); }
static inline uint64_t link_relays_at(uint64_t page){ return link_heard_at(page)+(uint64_t)MESH_LINK_LANES*sizeof(struct key_unit); }
static inline uint64_t link_lanes_at(uint64_t page){ return link_relays_at(page)+(uint64_t)MESH_RELAYS*sizeof(struct relay); }
static inline uint32_t link_blocks(uint64_t page){ return (uint32_t)((link_lanes_at(page)+MESH_LINK_LANES*sizeof(struct lane_record)+page-1)/page); }
enum { RUN_NONE, RUN_KEY, RUN_HELD, RUN_PAGES };
struct run { struct mesh_plan *table; struct mesh_entry *entry; struct relay *relay; struct key_unit unit; uint64_t k,first; uint32_t state,kind,posted; };
struct lane_state {
  _Alignas(MESH_LINE) uint64_t sent;
  uint64_t done,keys,keys_done;
  uint32_t sends;
  struct run sending;
  _Alignas(MESH_LINE) uint64_t posted;
  uint64_t got;
  uint32_t window;
  struct run landing;
  struct lane_record *record;
  struct arrival *arrival;
  uint32_t arrivals,arrival_room;
};
_Static_assert(offsetof(struct lane_state,posted)%MESH_LINE==0,"a thread's words a line of their own");
static inline uint64_t *lane_posts(struct lane_state *s,int way){ return way?&s->posted:&s->sent; }
static inline uint64_t *lane_done(struct lane_state *s,int way){ return way?&s->got:&s->done; }
/* design/transport.md section 3; section 14, "A process a port" */
enum { HELLO=1, RECORD=3, LINK };
struct message { uint32_t magic,kind,node,count; uint64_t boot; int64_t code; uint32_t block,port,bytes;
  uint16_t lid,version; uint8_t gid[16]; uint32_t pair[MESH_LINK_LANES][2]; };
#define XMAGIC (0x4d595048u+131u+(1u<<12))
struct wire_mapping { char *data; size_t length; };
#define QD 4095
#define MESH_BANK ((size_t)1<<32)
struct mesh_device {
  char name[32];
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_mr **extent_mrs;
  uint32_t frame_capacity,pairs,max_pairs;
  struct ibv_port_attr port;
  union ibv_gid gid;
  uint8_t gid_index,gid_matched,extent_shift;
};
struct mesh_queue {
  _Alignas(64) struct ibv_qp *pair;
  int (*send)(struct ibv_qp *,struct ibv_send_wr *,struct ibv_send_wr **);
  int (*receive)(struct ibv_qp *,struct ibv_recv_wr *,struct ibv_recv_wr **);
  uint32_t receive_capacity,send_capacity;
};
_Static_assert(sizeof(struct mesh_queue)==64 && _Alignof(struct mesh_queue)==64,"mesh_queue native dispatch");
struct mesh_verbs { struct mesh_device *device; struct mesh_queue *queues; struct ibv_cq *completion,*sent; uint32_t psn;
  int (*poll)(struct ibv_cq *,int,struct ibv_wc *); };
enum { LINK_DOWN, LINK_MADE, LINK_CONNECTING, LINK_UP };
struct link {
  struct mesh_verbs provider;
  int state,heard,lanes;
  _Atomic int lost;
  _Atomic uint32_t served,broken,pairing;
  uint32_t plan,peer[MESH_LINK_LANES][2],qpn[MESH_LINK_LANES],page,pool,pages,psn,cable;
  struct frames share;
  struct lane_state lane[MESH_LINK_LANES];
  uint64_t pool_free[MESH_POOL_BLOCKS/64],order[MESH_LINK_LANES];
};
/* design/transport.md section 14, "A process a port"; section 15, "The children"; section 16 */
#define MESH_CHILD_VERSION 10u
enum { CHILD_OPEN=1, CHILD_MAKE, CHILD_CONNECT, CHILD_HALT, CHILD_END,
       CHILD_OPENED, CHILD_MADE, CHILD_CONNECTED, CHILD_BROKEN, CHILD_FAULTED, CHILD_NAMED };
struct child_message {
  uint32_t kind,pairing;
  int32_t result;
  uint32_t qpn[MESH_LINK_LANES],peer[MESH_LINK_LANES][2],psn,share,broken,node,page,pool,lanes;
  uint16_t lid,padding;
  uint8_t gid[16];
  uint64_t bandwidth;
};
#define TAG_CHILD (UINT64_C(1)<<58)
struct port;
/* design/transport.md section 14, "Sending", "The index" and "Relays" */
struct named { uint64_t name[2]; uint32_t node,padding; };
struct joined { uint32_t count,cable[MESH_REACH]; uint64_t weight[MESH_REACH],cycle; };
struct arrival { struct mesh_entry *entry; uint64_t k; uint32_t runs,padding; };
struct indexed { uint64_t key[5]; struct mesh_plan *table; struct mesh_entry *entry; uint64_t k; };
struct walker {
  uint64_t serial,told;
  struct named *name;
  uint32_t names,nodes,*node,*hop,*relaying,*path,*paths;
  struct joined *joined;
  struct indexed *index;
  uint32_t indexed,index_room,blocked[MESH_CLIENTS],blocks;
  struct mesh_plan **table;
  uint32_t tables,table_room;
  int fresh;
};
struct progress { _Alignas(MESH_LINE) _Atomic uint64_t passes; struct port *port; int way; struct walker walker; };
void mesh_tuples_lost(struct hdr *,uint64_t client);
int mesh_map_copy(struct hdr *,struct mesh_record **,uint64_t (**names)[2]);

struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,greeted,connecting,writing,open,opening,stuck,budget,family,unit,job_pairing;
  char name[IFNAMSIZ];
  int control,job,result,threads;
  _Atomic int done,closing;
  uint64_t next,bandwidth,job_at;
  _Atomic uint64_t quiet_at;
  struct message heard;
  uint16_t lid;
  uint8_t gid[16];
  unsigned char *in,*out;
  size_t held,in_room,out_length,out_sent,out_room;
  dispatch_queue_t work;
  struct link link;
  struct progress progress[2];
  pid_t child,elsewhere;
  uint32_t holder;
  int child_fd,recycle,ending;
  uint32_t child_held;
  int deaf,hung,stage;
  uint64_t child_at,late_from,late_at,elsewhere_start;
  unsigned char child_in[sizeof(struct child_message)];
};
#endif
