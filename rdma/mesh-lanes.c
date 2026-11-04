/* The port child's verbs jobs, on its port's serial queue (`port->work`): its device's open (device_open) and its link's
   make and connect (link_job: link_make, link_connect, lane_reset).  From tag pre-return-20261103 rdma/mesh-link.c,
   verbatim but for interfaces (design/page-queue-change-request.md, "The functions", the bridge).  A port is one cable,
   and its link's lanes are that cable's queue pairs (mesh-bridge.h link_plan), made once and connected once in the
   child's life and never destroyed by a call: the kernel releases them at the child's exit (mesh-verbs.h, the note where
   down_pair was).  Every call here returns.  A failure is an error value in port->result or l->result, which child_jobs
   (mesh-child.c) tells the parent, and nothing here retries it. */
#include "mesh-bridge.h"

/* tag pre-return-20261103 rdma/mesh-link.c:554-571, the link jobs' description, struct link_numbers and link_lapsed,
   verbatim (3f72cbde).  Differs here, as mesh-bridge.h link_plan says: no lane is set apart to post nothing (the tag's
   lanes 0 and 1), so the plan's lanes alone divide the completion queues, every one of them (link_make), and every lane
   of the plan has its progress made new at the connect (link_connect) */
/* design/transport.md section 14: a link job on the port's queue, guarded (no queue pair of the device is posted on
   while one is made or connected: guard_raise), its verbs calls made on hearing the peer, from the peer's numbers as the
   job was started (`numbers`, its LID and GID among them: the event loop takes the peer's LINK and hello meanwhile), and
   none once its pairing is lost (B10: read as the guard stands and between its steps; a guard not held within
   MESH_PLATFORM_NS halts the port, its pairing with it, port_wait), each in the port's child and once in its life
   (design/transport.md section 14, "A process a port"): the lanes made (LINK_JOB_MAKE: the port's own address read
   again, then a UC queue pair in INIT for each lane of the link's plan, d6dceac's verbs_up, each lane's share of the
   completion queues the plan's streams' lanes alone dividing them, lanes 0 and 1 posting nothing:
   design/HETEROGENEOUS-LINKS.md; EALREADY for a make asked
   again); or connected to the peer's (LINK_JOB_CONNECT, the plan's lanes alone: RTR, then RTS, each lane's number and
   gate set, every lane's progress made new, and the port's progress thread started).  No queue pair is moved to
   another state, destroyed or forgotten after it: the in-place RESET of 127 panicked the M5 in AppleThunderboltRDMA on its
   first run (2026-10-19, design/RDMA-KERNEL-RECOVERY.md), and the kernel releases every object at the child's exit, its
   one end */
struct link_numbers { uint32_t peer[MESH_LINK_LANES][2],pairing; uint16_t lid; uint8_t gid[16]; };
static int link_lapsed(const struct link *l,const struct link_numbers *numbers){
  return atomic_load_explicit(&l->pairing,memory_order_relaxed)!=numbers->pairing?ECANCELED:0;
}
/* tag pre-return-20261103 rdma/mesh-link.c:572-586 link_make, verbatim (3f72cbde) but its share: the tag divided the
   completion queues among the streams' lanes, plan_lanes(l->plan>>MESH_LANE_FIRST), lanes 0 and 1 posting nothing; here
   every lane of the plan carries pages (mesh-bridge.h link_plan, which has no MESH_LANE_FIRST), so plan_lanes(l->plan) */
static int link_make(struct port *port,const struct link_numbers *numbers){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  struct ibv_port_attr attributes;
  int error=0;
  if(p->queues)return EALREADY;
  /* design/transport.md: the port's own address (its attributes and GID) read again at the make, as its opening read
     it: a peer's restart can move the port's GID table, and LINKs naming the address of the port's opening failed every
     lane's connection at the peer with EIO (2026-10-19, after the Mini's power loss) */
  if(ibv_query_port(port->device.context,1,&attributes))return errno?errno:EIO;
  port->device.port=attributes;
  if((error=device_gid(&port->device,port->name)) || (error=link_lapsed(l,numbers)) || (error=pair_up(p,l->plan)))return error;
  l->share=(struct frames){(uint64_t)MIN(p->completion->cqe,p->sent->cqe)/MAX(plan_lanes(l->plan),1u)};
  return 0;
}
/* tag pre-return-20261103 rdma/mesh-link.c:587, verbatim */
static void lane_reset(struct port *port,uint32_t k);
/* tag pre-return-20261103 rdma/mesh-link.c:601-617 link_connect, verbatim (4bb01da0) but two lines: each lane of the plan
   has its progress made new (lane_reset from lane 0, where the tag's began at MESH_LANE_FIRST), and the tag's last loop,
   each stream's identity (l->wire[s]=stream_wire(...), tag :588-600), is gone with the streams (mesh-bridge.h struct
   link) */
static int link_connect(struct port *port,const struct link_numbers *numbers){
  struct link *l=&port->link;
  struct mesh_verbs *p=&l->provider;
  const uint32_t plan=l->plan;
  int error=0;
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)error=queue_receive(p,(int)k,numbers->peer[k],numbers->lid,numbers->gid);
  if(!error)error=link_lapsed(l,numbers);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)error=queue_send(p,(int)k);
  const struct frames share={MAX(l->share.n,1)};
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++){
    l->qpn[k]=plan>>k&1?p->queues[k].pair->qp_num:UINT32_MAX;
    l->gate[k]=(struct send_gate){.capacity=plan>>k&1?frames_min((struct frames){p->queues[k].send_capacity},share):(struct frames){0}};
  }
  if(!error)error=link_lapsed(l,numbers);
  for(uint32_t k=0;k<MESH_LINK_LANES && !error;k++)if(plan>>k&1)lane_reset(port,k);
  return error;
}
/* tag pre-return-20261103 rdma/mesh-link.c:655-670 link_job, verbatim (60b00774), but not static: mesh-child.c's
   child_command starts it (mesh-bridge.h) */
void link_job(struct port *port,int job){
  struct link *l=&port->link;
  struct link_numbers numbers={.pairing=atomic_load_explicit(&l->pairing,memory_order_relaxed),.lid=port->heard.lid};
  memcpy(numbers.peer,l->peer,sizeof numbers.peer);memcpy(numbers.gid,port->heard.gid,sizeof numbers.gid);
  l->job=job;l->job_pairing=numbers.pairing;l->job_at=now;atomic_store_explicit(&l->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    atomic_store_explicit(&port->thread[1],pthread_self(),memory_order_release);
    const int raised=guard_raise(port);
    int result=link_lapsed(l,&numbers);
    if(!result)result=job==LINK_JOB_MAKE?link_make(port,&numbers):link_connect(port,&numbers);
    if(!result && job==LINK_JOB_CONNECT)result=port_threads(port);
    guard_lower(port,raised);
    l->result=result;
    atomic_store_explicit(&port->thread[1],NULL,memory_order_release);
    atomic_store_explicit(&l->done,1,memory_order_release);
    wake();
  });
}
/* tag pre-return-20261103 rdma/mesh-link.c:691-697 lane_reset, verbatim (4bb01da0) */
/* Reset separate request and frame accounting at connect. Each receive uses
   its actual byte extent's frame charge; a one-byte request consumes one slot. */
static void lane_reset(struct port *port,uint32_t k){
  struct link *l=&port->link;
  const struct frames room=frames_min(l->share,(struct frames){l->provider.queues[k].receive_capacity});
  l->lane[k]=(struct lane_state){.frame_capacity=room.n};
  atomic_thread_fence(memory_order_release);
}
/* tag pre-return-20261103 rdma/mesh-link.c:922-934 device_open, verbatim (abeffc1b), but not static (mesh-child.c's
   child_command starts it, mesh-bridge.h), and its device_up main's (mesh-verbs.h): the region `M` names the window's
   blocks, and its -1 and errno read as the tag's error code, EIO where the call set none */
/* the child's open of its port's device, on its port's queue, guarded; its end told the child's loop (child_main) */
void device_open(struct port *port){
  port->job=1;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  dispatch_async(port->work,^{
    const int raised=guard_raise(port);
    int refused=0,result=device_up(&port->device,&wire,M,&refused)?(errno?errno:EIO):0;
    if(!result)result=device_gid(&port->device,port->name);
    guard_lower(port,raised);
    port->result=result;port->refused=refused;
    atomic_store_explicit(&port->done,1,memory_order_release);
    wake();
  });
}
