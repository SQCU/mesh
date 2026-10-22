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
#include <poll.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
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
/* design/transport.md section 14, "The stream"; design/HETEROGENEOUS-LINKS.md sections 5 and 10 */
#define MESH_LINK_LANES 10u
#define MESH_LANE_FIRST 2u
#define MESH_STREAM_PAIRS ((MESH_LINK_LANES-MESH_LANE_FIRST)/2/MESH_LINK_STREAMS)
_Static_assert(MESH_STREAM_PAIRS==1 && MESH_LANE_FIRST+2*MESH_LINK_STREAMS==MESH_LINK_LANES,"a stream one lane pair");
_Static_assert(MESH_LINK_LANES<32,"a bit a lane");
static inline struct mesh_pages stream_pages(uint64_t a,uint64_t b){ return (struct mesh_pages){a+b}; }
static inline struct mesh_pages stream_through(uint64_t a,uint64_t b){ return (struct mesh_pages){2*a<2*b+1?2*a:2*b+1}; }
static inline uint32_t page_lane(struct mesh_pages p){ return (uint32_t)(p.n&1); }
#define MESH_FRAME 4096u
struct frames { uint64_t n; };
static inline struct frames frames_of(uint64_t bytes){ return (struct frames){(bytes+MESH_FRAME-1)/MESH_FRAME}; }
static inline struct frames frames_min(struct frames a,struct frames b){ return a.n<b.n?a:b; }
static inline uint32_t link_plan(uint32_t budget){
  if(budget>=MESH_LINK_LANES)return (UINT32_C(1)<<MESH_LINK_LANES)-1;
  const uint32_t first=budget>=3,pairs=(budget-first)/(2*MESH_STREAM_PAIRS);
  const uint32_t streams=pairs<MESH_LINK_STREAMS?pairs:MESH_LINK_STREAMS;
  return first|(((UINT32_C(1)<<(2*MESH_STREAM_PAIRS*streams))-1)<<MESH_LANE_FIRST);
}
static inline uint32_t plan_lanes(uint32_t plan){ return (uint32_t)__builtin_popcount(plan); }
static inline uint32_t plan_streams(uint32_t plan){ return (uint32_t)__builtin_popcount(plan>>MESH_LANE_FIRST)/(2*MESH_STREAM_PAIRS); }
/* design/transport.md section 15, "Completions" */
_Static_assert(sizeof(struct ibv_wc)==48 && offsetof(struct ibv_wc,status)==8 && offsetof(struct ibv_wc,opcode)==12 &&
  sizeof(((struct ibv_wc *)0)->status)==4 && sizeof(((struct ibv_wc *)0)->opcode)==4,"M11");
union mesh_network_event {
  struct kern_event_msg header;
  unsigned char bytes[KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data)];
};
_Static_assert(sizeof(union mesh_network_event)==48,"M24");
/* design/transport.md section 15, "The SEND gate" */
#define MESH_GATE_MASK ((UINT64_C(1)<<48)-1)
struct send_gate {
  _Alignas(64) struct frames posted;
  struct frames retired,capacity;
};
_Static_assert(sizeof(struct send_gate)==64,"a gate half a line, the send thread's");
static inline int gate_holds(const struct send_gate *gate,struct frames f){
  return ((gate->posted.n-gate->retired.n)&MESH_GATE_MASK)+f.n<=gate->capacity.n;
}
static inline uint64_t gate_post(struct send_gate *gate,uint32_t k,struct frames f){
  gate->posted.n+=f.n;
  return ((uint64_t)k<<48)|(gate->posted.n&MESH_GATE_MASK);
}
static inline void gate_retire(struct send_gate *gate,uint64_t id){ gate->retired.n=id&MESH_GATE_MASK; }
struct lane_state {
  _Alignas(MESH_LINE) uint64_t sent;
  uint64_t done;
  _Alignas(MESH_LINE) uint64_t posted;
  uint64_t got;
  uint32_t window;
};
_Static_assert(offsetof(struct lane_state,posted)==MESH_LINE && sizeof(struct lane_state)==2*MESH_LINE,"a thread's words a line of their own");
static inline uint64_t *lane_posts(struct lane_state *s,int way){ return way?&s->posted:&s->sent; }
static inline uint64_t *lane_done(struct lane_state *s,int way){ return way?&s->got:&s->done; }
struct taken { struct mesh_row row; struct mesh_pages pages,base,page; const _Atomic uint64_t *word; uint32_t whole; };
struct walk { struct taken *ring; uint64_t taken,done,timed,at; struct mesh_pages pages,base,count; };
enum { HOLD_NONE, HOLD_SERVE, HOLD_END, HOLD_GONE };
/* design/transport.md section 3; section 14, "A process a port" */
enum { HELLO=1, FAREWELL, RECORD, LINK };
struct message { uint32_t magic,kind,node,count; uint64_t boot; int64_t code; uint32_t block,port,bytes;
  uint16_t lid,version; uint8_t gid[16]; uint32_t pair[MESH_LINK_LANES][2]; };
static inline uint32_t hello_budget(uint32_t count){ return count?count-1:MESH_LINK_LANES; }
#define XMAGIC (0x4d595048u+131u+(1u<<12))
struct wire_mapping { char *data; size_t length; };
#define QD 4095
#define MESH_BANK ((size_t)1<<32)
#define MESH_EXTENT ((size_t)1<<30)
struct mesh_device {
  char name[32];
  struct ibv_context *context; struct ibv_pd *domain; struct ibv_mr **extent_mrs;
  uint32_t frame_capacity,pairs,max_pairs;
  struct ibv_port_attr port;
  union ibv_gid gid;
  uint8_t gid_index,gid_matched;
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
enum { LINK_JOB_MAKE=1, LINK_JOB_CONNECT };
struct link {
  struct mesh_verbs provider;
  int state,job,result,heard,lanes;
  _Atomic int done,lost;
  _Atomic uint32_t served,broken,pairing;
  uint32_t plan,peer[MESH_LINK_LANES][2],qpn[MESH_LINK_LANES],page,pages,lists,job_pairing,psn;
  struct frames share;
  struct send_gate gate[MESH_LINK_LANES];
  struct lane_state lane[MESH_LINK_LANES];
  uint64_t job_at;
  uint64_t held[MESH_LINK_STREAMS],number[MESH_LINK_STREAMS];
  uint32_t hold[MESH_LINK_STREAMS];
  struct walk sending[MESH_LINK_STREAMS];
  _Atomic uint64_t *posted_at[MESH_LINK_STREAMS];
  _Atomic uint64_t begun[MESH_LINK_STREAMS],ended[MESH_LINK_STREAMS];
  uint64_t wire[MESH_LINK_STREAMS];
  _Alignas(MESH_LINE) _Atomic uint64_t stopped[MESH_LINK_STREAMS];
  uint64_t rnumber[MESH_LINK_STREAMS],rheld[MESH_LINK_STREAMS];
  struct walk landing[MESH_LINK_STREAMS];
  _Atomic uint64_t *landed_at[MESH_LINK_STREAMS];
};
_Static_assert(offsetof(struct link,stopped)%MESH_LINE==0,"the receive thread's words a line of their own");
/* design/transport.md section 14, "A process a port"; section 15, "The children" */
#define MESH_CHILD_VERSION 3u
#define LINK_STREAMS_ENDED ECONNABORTED
enum { CHILD_OPEN=1, CHILD_MAKE, CHILD_CONNECT, CHILD_HALT, CHILD_END,
       CHILD_OPENED, CHILD_MADE, CHILD_CONNECTED, CHILD_BROKEN, CHILD_FAULTED, CHILD_TERMINATED };
struct child_message {
  uint32_t kind,pairing;
  int32_t result,refused;
  uint32_t qpn[MESH_LINK_LANES],peer[MESH_LINK_LANES][2],psn,share,broken,node,page,lanes;
  uint16_t lid,padding;
  uint8_t gid[16];
  uint64_t bandwidth;
};
#define MESH_CHILD_SETTLE_NS (UINT64_C(60)*1000000000)
#define TAG_CHILD (UINT64_C(1)<<58)
struct port;
struct progress { struct port *port; int way; };
struct port {
  struct mesh_device device;
  uint32_t index,interface,peer,peer_port,greeted,parted,connecting,writing,open,opening,stuck,budget,refusals,family,unit;
  int64_t farewell;
  char name[IFNAMSIZ];
  int control,job,result,refused,threads;
  _Atomic int done,closing;
  uint64_t next,bandwidth,job_at,retry_at;
  _Atomic uint64_t quiet_at;
  struct message heard;
  uint16_t lid;
  uint8_t gid[16];
  unsigned char *in,*out;
  size_t held,in_room,out_length,out_sent,out_room;
  dispatch_queue_t work;
  struct link link;
  struct progress progress[2];
  _Atomic(pthread_t) thread[3];
  pid_t child,elsewhere;
  uint32_t holder;
  int child_fd,recycle,ending;
  uint32_t aborts,child_held;
  uint64_t child_at,hold_to;
  unsigned char child_in[sizeof(struct child_message)];
};
#define MESH_HEALTH_NS UINT64_C(10000000000)
struct known { uint32_t node; uint64_t heard,gone,boot,sequence; };
#define MESH_KNOWN 256
#endif
