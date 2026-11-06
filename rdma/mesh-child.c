/* The bridge's per-port child, `mesh-flow --child VERSION DEVICE INDEX ONES REGION`: verbatim from tag pre-return-20261103
   rdma/mesh-link.c lines 1536-1740 (the comment above `child_asked` through `child_main`), as the operator requires
   (design/page-queue-change-request.md, "The bridge's process lifecycle": "you have to verbatim reuse that process
   lifecycle code"), adapted only at interfaces: main's region (mesh.h struct hdr, MESH_VERSION, mapped by the name the
   command line's sixth word gives), main's device's registrations (mesh-verbs.h struct mesh_device `regions`,
   `region_count`) and mesh-bridge.h's prototypes for the other files' functions.  Each line that differs from the tag
   says so at the spot.  A port is one cable; the tag's lanes are that cable's queue pairs (`l->provider.queues[k].pair`,
   `l->plan`). */
#include "mesh-bridge.h"


/* tag pre-return-20261103 rdma/mesh-link.c:1536-1554, the child's statics and child_tell (c18f7a0e, 437e2c38, 3f72cbde,
   abeffc1b), verbatim; the command line gains REGION, the region's name, as its sixth word (child_main) */
/* design/transport.md section 14, "A process a port": a port's child, `mesh-flow --child VERSION DEVICE INDEX ONES` as
   its parent spawned it (child_spawn; a parent of an earlier build adds its keepalive pace, which is not read: no
   child sends a keepalive), VERSION the contract's (mesh-bridge.h MESH_CHILD_VERSION): the parent's region mapped (its magic, version and
   bridge the parent's), its port the parent's port INDEX on DEVICE (its streams' words the region's), the duds' block
   ONES the parent's, the provider's faults absorbed (fault_heard) and its peer interfaces'
   terminations heard (B9b: its lanes served no more at once, port_halt).  It takes the parent's messages on descriptor 3
   in order: an open, a make or a connect of its link as a job on its port's queue (link_job, device_open; one asked
   while a job is out, or while a halt of its own is not yet answered by the parent's, is refused), a halt (no lane
   served, a job out lapsed), an end, and a message of any other kind let be; and tells each job's end, naming the pairing it was asked for.  A connect that
   ends while its pairing stands opens the link's streams (streams_open) and then serves every lane, its pairing read
   again (port_halt's order).  The child ends at its parent's word, once its own windows have ended (at most
   MESH_PLATFORM_NS on), or at its parent's loss (the socket's end), its lanes served no more, MESH_PLATFORM_NS and its
   windows on (at most twice that): its exit, the kernel's release of its device's objects, no verbs call made */
static uint32_t child_asked;
static uint64_t child_bandwidth;
static int child_halted;
static unsigned char child_in[sizeof(struct child_message)];
static uint32_t child_got;
static void child_tell(struct child_message said){ message_send(3,&said); }
/* tag pre-return-20261103 rdma/mesh-link.c:1555-1592 lanes_flush (07949b28, 60be5eb7, 34a0d80c), verbatim: the lanes
   are this cable's queue pairs (mesh-verbs.h pair_up: queue pair k lane k, its receives on `completion`, its SENDs on
   `sent`) */
/* design/transport.md section 14, "A process a port": the queue pairs a child leaves holding work released as the
   literature releases one (rdmamojo "Flushing Work Requests"; Linux ib_drain_qp before ib_destroy_qp; the operator's
   permission of 2026-10-22, "you can do this and nothing else with error states"): each lane with a SEND not completed
   or a receive not landed moved to the error state, so the device completes each (IBV_WC_WR_FLUSH_ERR), and those
   completions taken, at most until `until`; the kernel destroys the rest at the exit.  An exit with work posted can
   wait in the kernel's release with no deadline (IOThunderboltPath::deactivateSynchronous, metal-microbench
   docs/facts.md), holding its device and its pages; on this provider the flush took 0.41 and 0.46 s for 687 requests.
   Under the port's guard (child_leave): no progress thread posts or polls meanwhile */
static void lanes_flush(struct port *port,uint64_t until){
  struct link *l=&port->link;
  uint64_t left[2]={0,0};
  uint32_t moved=0;
  for(uint32_t k=0;k<MESH_LINK_LANES;k++){
    const struct lane_state *s=l->lane+k;
    struct ibv_qp_attr error={.qp_state=IBV_QPS_ERR};
    if((s->sent!=s->done || s->posted!=s->got) && l->plan>>k&1 && !ibv_modify_qp(l->provider.queues[k].pair,&error,IBV_QP_STATE)){
      moved|=UINT32_C(1)<<k;left[0]+=s->sent-s->done;left[1]+=s->posted-s->got;
    }
  }
  const uint64_t held[2]={left[0],left[1]};
  if(held[0]||held[1])mesh_say("bridge","link %u (%s): its child leaves with %llu SENDs and %llu receives outstanding on lanes %#x: "
                               "moved to the error state",port->index,port->name,(unsigned long long)held[0],(unsigned long long)held[1],moved);
  struct ibv_wc done[MESH_COMPLETION_BATCH];
  /* c 0 the receive queue's completions, 1 the send queue's; taken until all are or none has come for MESH_FLUSH_QUIET_NS
     (the provider flushed 687 requests in 0.41 s, and gives none for a SEND its peer's credit held: 1024 of 2048 at
     each exit, 2026-10-31), at most until `until` */
  uint64_t heard=clock_ns();
  for(int n=0,c=0;(left[0]||left[1]) && n>=0;c^=1){
    const uint64_t at=clock_ns();
    if(at>=until || at-heard>=MESH_FLUSH_QUIET_NS)break;
    n=l->provider.poll(c?l->provider.sent:l->provider.completion,MESH_COMPLETION_BATCH,done);
    if(n>0)heard=at;
    /* rdma-core libibverbs/enum_strs.c: [IBV_WC_WR_FLUSH_ERR] = "Work Request Flushed Error"; rdma-core
       libibverbs/man/ibv_poll_cq.3: "If the completion status is other than IBV_WC_SUCCESS, only the following
       attributes are valid: wr_id, status, qp_num, and vendor_err."; libfabric man/fi_cntr.3.md: "fi_cntr_readerr:
       Reads the number of operations which have completed in error." */
    for(int i=0;i<n;i++)if(left[c?0:1] && moved>>link_lane(l,done[i].qp_num)&1){
      left[c?0:1]--;
      if(c)atomic_fetch_add_explicit(((const struct prepared_send *)(uintptr_t)done[i].wr_id)->count+1,1,memory_order_release);
    }
  }
  if(held[0]||held[1])mesh_say("bridge","link %u (%s): its child's flush took %llu of %llu SENDs and %llu of %llu receives",port->index,
                               port->name,(unsigned long long)(held[0]-left[0]),(unsigned long long)held[0],(unsigned long long)(held[1]-left[1]),
                               (unsigned long long)held[1]);
}
/* tag pre-return-20261103 rdma/mesh-link.c:1593-1618 child_leave (07949b28, 437e2c38, bc17811b), verbatim but its
   registrations: main's device.regions and region_count for the tag's extent_mrs and extents */
/* the child's end, asked (its parent's word, or a signal that ends a process: SIGTERM, SIGINT, SIGHUP) or not (its
   parent's loss): its progress threads held at the port's guard (none where a job of the port's is out), its lanes
   served no more, its own windows waited out (B9, B9b: no release inside a termination), what its lanes hold flushed
   (lanes_flush), and its exit, within (asked 1, else 2) MESH_PLATFORM_NS and the flush's MESH_PLATFORM_NS */
static __attribute__((noreturn)) void child_leave(struct port *port,int asked){
  const uint64_t start=clock_ns(),bound=start+(asked?1:2)*MESH_PLATFORM_NS;
  const int held=port->link.job?0:guard_raise(port);
  atomic_fetch_add(&port->link.pairing,1);
  atomic_store(&port->link.served,0);
  atomic_store_explicit(&port->closing,1,memory_order_release);
  guard_wake(port);
  for(;;){
    const uint64_t at=clock_ns(),windows=MAX(atomic_load(&exit_at),atomic_load(&port->quiet_at));
    const uint64_t until=MIN(bound,MAX(asked?start:start+MESH_PLATFORM_NS,windows));
    if(at>=until)break;
    struct timespec rest={(time_t)((until-at)/1000000000u),(long)((until-at)%1000000000u)};
    nanosleep(&rest,NULL);
  }
  if(held>0)lanes_flush(port,clock_ns()+MESH_PLATFORM_NS);
  /* its registrations of the region released before its exit, then the exit itself: the kernel's own release of
     them at an exit took 11.4-13 s with nothing posted, ibv_dereg_mr of the same 82 extents 1.0-1.3 s and the exit
     after 0.85 s (m5ult1, 2026-11-01).  No queue pair is moved or destroyed here (rubrics M3): those the exit releases */
  /* main's registrations are the window's regions (mesh-verbs.h device_up: `regions`, `region_count` of them), the
     tag's were the arena's extents (`extent_mrs`, `extents`); the loop is the tag's over main's names */
  for(uint32_t e=0;port->device.regions && e<port->device.region_count;e++)
    if(port->device.regions[e]){ibv_dereg_mr(port->device.regions[e]);port->device.regions[e]=NULL;}
  _exit(0);
}
/* tag pre-return-20261103 rdma/mesh-link.c:1619-1642 child_command (11807bec, 306b22be, 76be3217), verbatim */
static void child_command(struct port *port,const struct child_message *heard){
  struct link *l=&port->link;
  if(heard->kind==CHILD_OPEN){if(!port->job)device_open(port);return;}
  if(heard->kind==CHILD_HALT){
    atomic_fetch_add(&l->pairing,1);atomic_store(&l->served,0);guard_wake(port);
    child_halted=0;
    return;
  }
  if(heard->kind==CHILD_END)child_leave(port,1);
  if(heard->kind!=CHILD_MAKE && heard->kind!=CHILD_CONNECT)return;
  const int job=heard->kind==CHILD_MAKE?LINK_JOB_MAKE:LINK_JOB_CONNECT;
  int refused=l->job?EBUSY:child_halted || atomic_load(&faulted) || atomic_load(&terminated)?ECANCELED:0;
  if(!refused && job==LINK_JOB_CONNECT){
    memcpy(l->peer,heard->peer,sizeof l->peer);port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
    port->peer=heard->node;child_bandwidth=heard->bandwidth;
  }
  if(refused){
    child_tell((struct child_message){.kind=job==LINK_JOB_MAKE?CHILD_MADE:CHILD_CONNECTED,.pairing=heard->pairing,.result=refused});
    return;
  }
  child_asked=heard->pairing;
  if(job==LINK_JOB_MAKE){l->page=heard->page;l->plan=heard->lanes;}
  link_job(port,job);
}
/* tag pre-return-20261103 rdma/mesh-link.c:1643-1673 child_jobs (3f72cbde, 0905738c, 76be3217), verbatim */
static void child_jobs(struct port *port){
  struct link *l=&port->link;
  if(port->job && atomic_load_explicit(&port->done,memory_order_acquire)){
    port->job=0;
    struct child_message said={.kind=CHILD_OPENED,.result=port->result,.refused=port->refused};
    if(!port->result){
      port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
      said.lid=port->lid;memcpy(said.gid,port->gid,16);said.bandwidth=port_bandwidth(&port->device.port);
      said.lanes=port->device.max_pairs;
    }
    child_tell(said);
  }
  if(!l->job || !atomic_load_explicit(&l->done,memory_order_acquire))return;
  const int job=l->job;
  l->job=0;
  struct child_message said={.kind=job==LINK_JOB_MAKE?CHILD_MADE:CHILD_CONNECTED,.pairing=child_asked,.result=l->result};
  if(!said.result && l->job_pairing!=atomic_load(&l->pairing))said.result=ECANCELED;
  if(!said.result && job==LINK_JOB_MAKE){
    for(uint32_t k=0;k<MESH_LINK_LANES;k++)said.qpn[k]=l->plan>>k&1?l->provider.queues[k].pair->qp_num:UINT32_MAX;
    said.psn=l->provider.psn;said.share=(uint32_t)l->share.n;
    port->lid=port->device.port.lid;memcpy(port->gid,&port->device.gid,16);
    said.lid=port->lid;memcpy(said.gid,port->gid,16);
  }
  else if(!said.result){
    streams_open(port,port->peer,child_bandwidth);
    atomic_store(&l->served,l->plan);
    if(atomic_load(&l->pairing)!=l->job_pairing){atomic_store(&l->served,0);said.result=ECANCELED;}
    guard_wake(port);
  }
  child_tell(said);
}
/* tag pre-return-20261103 rdma/mesh-link.c:1674-1680 child_reports (306b22be, 3f72cbde, ad55cbf4), verbatim */
static void child_reports(struct port *port){
  const uint32_t broken=atomic_exchange_explicit(&port->link.broken,0,memory_order_acq_rel);
  if(broken)child_tell((struct child_message){.kind=CHILD_BROKEN,.pairing=child_asked,.broken=broken,
                                              .result=atomic_exchange_explicit(&port->link.lost,0,memory_order_relaxed)});
  if(atomic_exchange(&faulted,0)){child_halted=1;child_tell((struct child_message){.kind=CHILD_FAULTED});}
  if(atomic_exchange(&terminated,0)){child_halted=1;child_tell((struct child_message){.kind=CHILD_TERMINATED});}
}
/* tag pre-return-20261103 rdma/mesh-link.c:1681-1692 child_read (3f72cbde, ad55cbf4, 33ca0dab), verbatim */
static void child_read(struct port *port){
  for(;;){
    const ssize_t got=recv(3,child_in+child_got,sizeof child_in-child_got,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0)child_leave(port,0);
    if((child_got+=(uint32_t)got)<sizeof child_in)continue;
    child_got=0;
    struct child_message heard;
    memcpy(&heard,child_in,sizeof heard);
    child_command(port,&heard);
  }
}
/* tag pre-return-20261103 rdma/mesh-link.c:1693-1740 child_main (07949b28, 378dbd5f, 437e2c38), verbatim but its
   interfaces: not static (mesh-link.c's main calls it, mesh-bridge.h), the region by its name, and main's MESH_VERSION */
int child_main(int argc,char **argv){
  signal(SIGPIPE,SIG_IGN);
  /* argc>=7 for the tag's argc>=6: the command line's sixth word, REGION, is read below */
  const unsigned long index=argc>=7?strtoul(argv[4],NULL,10):MESH_PORTS;
  if(index>=MESH_PORTS){return 1;}
  faults_install();
  /* the region by its name, REGION, the command line's sixth word: MESH_NAME, the node's one region, which the parent
     passes there (mesh-children.c child_spawn), as the tag's child opened MESH_NAME itself */
  const int fd=shm_open(argv[6],O_RDWR);
  struct stat shape;
  if(fd<0 || fstat(fd,&shape))die("child: its bridge's region");
  M=mmap(NULL,(size_t)shape.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  /* main's layout version, MESH_VERSION, for the tag's MESH_LAYOUT */
  if(M==MAP_FAILED || M->magic!=MESH_MAGIC || M->version!=MESH_VERSION || atomic_load(&M->bridge_pid)!=(uint64_t)getppid())
    die("child: its bridge's region is not its parent's");
  if(wire_map(&wire,M,fd))die("child: transport page aliases");
  close(fd);
  struct port *port=ports;
  *port=(struct port){.index=(uint32_t)index,.control=-1,.child_fd=-1,.work=dispatch_queue_create("mesh.port",DISPATCH_QUEUE_SERIAL)};
  port->link.provider.device=&port->device;port->link.page=MESH_ABSENT;
  snprintf(port->device.name,sizeof port->device.name,"%s",argv[3]);snprintf(port->name,sizeof port->name,"%s",argv[3]+5);
  ones=(uint32_t)strtoul(argv[5],NULL,10);
  __atomic_store_n(&port_count,1,__ATOMIC_SEQ_CST);
  struct kevent64_s event;
  if((events=kqueue())<0)die("child events");
  EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
  EV_SET64(&event,3,EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
  /* a signal that ends a process ends the child as its parent's word does (child_leave: its lanes flushed first), within
     its bound; SIGKILL ends it at once */
  static const int ending[]={SIGTERM,SIGINT,SIGHUP};
  for(size_t i=0;i<sizeof ending/sizeof *ending;i++){
    EV_SET64(&event,(uint64_t)ending[i],EVFILT_SIGNAL,EV_ADD,0,0,UINT64_MAX,0,0);
    if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("child events");
    signal(ending[i],SIG_IGN);
  }
  terminations_watch(0);
  for(;;){
    now=clock_ns();
    child_jobs(port);
    child_reports(port);
    struct kevent64_s ready[4];
    struct timespec tick={(time_t)(MESH_PAUSE_NS/1000000000u),(long)(MESH_PAUSE_NS%1000000000u)};
    const int count=kevent64(events,NULL,0,ready,4,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++){
      if(ready[i].filter==EVFILT_SIGNAL)child_leave(port,1);
      if(ready[i].filter==EVFILT_READ && ready[i].ident==3)child_read(port);
    }
  }
}
