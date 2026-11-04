/* The bridge's parent side of its per-port children: their spawn, messages, end and reaping, and the holder table's
   lines that go with them.  The failport's process lifecycle, verbatim (design/page-queue-change-request.md, "The
   bridge's process lifecycle"): tag pre-return-20261103 rdma/mesh-link.c 1288-1496 (message_send to child_reaped) and
   2079-2121 (main after the windows' hold, here children_end).  Only interfaces are adapted: the shared state and the
   prototypes are mesh-bridge.h's, the holder probes mesh-holders.h's, the holder table mesh.h's; each line that is not
   the tag's has a note at its spot. */
#include "mesh-bridge.h"
#include "mesh-holders.h"
/* tag rdma/mesh-link.c:2-7: those of its headers these functions use (libproc.h is mesh-holders.h's) */
#include <spawn.h>
#include <sys/wait.h>
#include <crt_externs.h>
#include <sysexits.h>

/* design/transport.md section 14, "A process a port": the parent's side of a port's child.  The bridge is a parent
   (the region, the map and records, the clients, discovery, hellos and the control sockets) and a child a port (its
   device, its link's queue pairs and completion queues, its progress thread): mesh-flow spawned again, the same image
   (`self_path`), told the contract's version it is spawned for (mesh-bridge.h MESH_CHILD_VERSION), in a process group of its own (a job's group signalled at its end reaches no verbs owner), with
   the parent's socket at descriptor 3, the parent's stderr, and nothing else.  The child's exit is the one release of
   its port's verbs objects (the kernel's), and a link once connected and then lost pairs again with the port's next
   child */
/* tag rdma/mesh-link.c:1295 `static char self_path[PATH_MAX];` is not here: mesh-bridge.h's extern, defined in
   mesh-link.c, whose main sets it as the tag's did */
/* tag pre-return-20261103 rdma/mesh-link.c:1296-1314 message_send, verbatim but extern (111d42f7 90179f5e d556370e) */
/* a message whole on the socket between a port's child and its parent, within MESH_PLATFORM_NS of the call (the operator,
   2026-10-23: no wait runs longer than a deadline presented when it is made, 5 s by default): a peer gone drops it, and
   one that takes no more by the deadline ends this process at once */
void message_send(int fd,const struct child_message *said){
  const char *at=(const char *)said;
  size_t left=sizeof *said;
  const uint64_t until=clock_ns()+MESH_PLATFORM_NS;
  while(left){
    const ssize_t sent=send(fd,at,left,MSG_DONTWAIT);
    if(sent>0){at+=sent;left-=(size_t)sent;continue;}
    if(sent<0 && errno==EINTR)continue;
    if(sent==0 || errno!=EAGAIN)return;
    const uint64_t t=clock_ns();
    if(t>=until){mesh_say("bridge","a message to a port's child or parent not taken in %.0f s: this process exits",
                          (double)MESH_PLATFORM_NS/1e9);exit(EX_SOFTWARE);}
    struct pollfd room={.fd=fd,.events=POLLOUT};
    poll(&room,1,(int)((until-t+999999)/1000000));
  }
}
/* tag pre-return-20261103 rdma/mesh-link.c:1315-1319 child_send, verbatim but extern (1f9be1d5 111d42f7) */
/* a message to the port's child, whole (message_send); one not taken (the child gone) is dropped, its end reaped
   (child_reaped) */
void child_send(struct port *port,struct child_message said){
  if(port->child_fd>=0)message_send(port->child_fd,&said);
}
/* tag rdma/mesh-link.c:1320, child_reaped's forward declaration, is mesh-bridge.h's prototype */
/* tag pre-return-20261103 rdma/mesh-link.c:1321-1335 child_elsewhere, verbatim (b346ef2b 8ac5c9fb c18f7a0e 07949b28) */
/* a child of the port's device that is not one of this bridge's: one of an earlier launch, alive or in its exit (the
   region's holder table, mesh_holds: a process in its exit holds its device's queue pairs until the kernel has released
   them, which no scan of the processes' arguments sees), its entry naming this device or any (`unit` 0); its pid, or 0.
   A device's queue pairs are one child's at a time: a link takes as many as its device holds for it, up to
   MESH_LINK_LANES (ten of a macOS 26 device's ten, design/HETEROGENEOUS-LINKS.md) */
static pid_t child_elsewhere(const struct port *port){
  const struct mesh_holder *holders=mesh_bridge_pages(M)->child;
  for(uint32_t entry=0;entry<2*MESH_PORTS;entry++){
    const pid_t pid=(pid_t)holders[entry].pid;
    int own=!pid;
    for(uint32_t p=0;p<port_count;p++)own|=pid==ports[p].child;
    if(!own && (!holders[entry].unit || holders[entry].unit==port->unit+1) && mesh_holds(pid,holders[entry].start))return pid;
  }
  return 0;
}
/* tag pre-return-20261103 rdma/mesh-link.c:1336-1397 child_spawn, extern, + REGION (1f9be1d5 b346ef2b 8ac5c9fb c18f7a0e 07949b28) */
/* the port's child spawned (its socket's messages and its exit watched, tagged by the port); 0, or the error, the
   port's next spawn held MESH_PAUSE_NS.  A child of its device from an earlier launch (child_elsewhere) holds this
   port's spawn alone, the other ports spawning, until its exit, which the event loop hears (EVFILT_PROC, tagged by the
   port: `elsewhere` cleared) and spawns at once; one already in its exit takes no such filter (ESRCH), and is read
   again MESH_PAUSE_NS on, never at every turn */
int child_spawn(struct port *port){
  int pair[2],on=1;
  if(port->elsewhere)return EBUSY;
  const pid_t elsewhere=child_elsewhere(port);
  if(elsewhere){
    struct kevent64_s watch;
    EV_SET64(&watch,(uint64_t)elsewhere,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,TAG_CHILD|port->index,0,0);
    if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){port->hold_to=now+MESH_PAUSE_NS;return EBUSY;}
    port->elsewhere=elsewhere;
    return EBUSY;
  }
  /* design/transport.md section 2: the child's entry among the region's holders, one no live child holds; where every
     one is held, the child waits as for an earlier launch's (a holder of the region is never left unrecorded) */
  struct mesh_holder *holders=mesh_bridge_pages(M)->child;
  uint32_t entry=0;
  while(entry<2*MESH_PORTS && mesh_holds((pid_t)holders[entry].pid,holders[entry].start))entry++;
  if(entry==2*MESH_PORTS){port->hold_to=now+MESH_PAUSE_NS;return EBUSY;}
  if(socketpair(AF_UNIX,SOCK_STREAM,0,pair)){
    const int error=errno?errno:EIO;
    port->hold_to=now+MESH_PAUSE_NS;
    return error;
  }
  setsockopt(pair[0],SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on);
  char index[16],page[16];
  snprintf(index,sizeof index,"%u",port->index);snprintf(page,sizeof page,"%u",ones);
  char version[16];
  snprintf(version,sizeof version,"%u",MESH_CHILD_VERSION);
  /* differs from the tag: the region's name, MESH_NAME, after ONES (`mesh-flow --child VERSION DEVICE INDEX ONES
     REGION`, the page-queue change's command line, which mesh-child.c child_main reads); the tag's child took the
     region as MESH_NAME without a word for it (8e5df675) */
  char *arguments[]={self_path,"--child",version,port->device.name,index,page,MESH_NAME,NULL};
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions,0,"/dev/null",O_RDONLY,0);
  posix_spawn_file_actions_addopen(&actions,1,"/dev/null",O_WRONLY,0);
  posix_spawn_file_actions_addinherit_np(&actions,2);
  posix_spawn_file_actions_adddup2(&actions,pair[1],3);
  posix_spawnattr_init(&attributes);
  posix_spawnattr_setflags(&attributes,POSIX_SPAWN_CLOEXEC_DEFAULT|POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&attributes,0);
  pid_t pid=0;
  const int error=posix_spawn(&pid,self_path,&actions,&attributes,arguments,*_NSGetEnviron());
  posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attributes);
  close(pair[1]);
  if(error){
    close(pair[0]);
    port->hold_to=now+MESH_PAUSE_NS;
    return error;
  }
  port->child=pid;port->child_fd=pair[0];port->child_held=0;port->ending=0;
  holders[entry]=(struct mesh_holder){(uint32_t)pid,port->unit+1,mesh_started(pid)};
  port->holder=entry;
  struct kevent64_s watch;
  EV_SET64(&watch,(uint64_t)pair[0],EVFILT_READ,EV_ADD,0,0,TAG_CHILD|port->index,0,0);
  if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){close(port->child_fd);port->child_fd=-1;}
  EV_SET64(&watch,(uint64_t)pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,TAG_CHILD|port->index,0,0);
  if(kevent64(events,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){child_reaped(port);return ECHILD;}
  return 0;
}
/* tag pre-return-20261103 rdma/mesh-link.c:1398-1411 child_job, extern, no link page (1f9be1d5 76be3217 128d0e84) */
/* a link job asked of the port's child: its lanes made (the link's pages named) or connected to the peer's
   (the peer's lanes as its LINK told them, its address as its hello did, its node and the link's bandwidth), under
   the link's pairing as it is asked; its end the child's answer (child_heard), taken by link_turn */
void child_job(struct port *port,int job){
  struct link *l=&port->link;
  l->job=job;l->job_pairing=atomic_load_explicit(&l->pairing,memory_order_relaxed);l->job_at=now;
  atomic_store_explicit(&l->done,0,memory_order_relaxed);
  /* differs from the tag (`.page=l->page`): the make names no link page, MESH_ABSENT.  This design leases no link pages
     and posts nothing on lanes that carry nothing (the page-queue contract, "Link pages"), so `l->page` and mesh.h's
     mesh_bridge_pages(M)->link[] stay MESH_ABSENT; the field stays, as the tag's struct child_message */
  struct child_message said={.kind=job==LINK_JOB_MAKE?CHILD_MAKE:CHILD_CONNECT,.pairing=l->job_pairing,.page=MESH_ABSENT,.lanes=l->plan};
  if(job==LINK_JOB_CONNECT){
    memcpy(said.peer,l->peer,sizeof said.peer);said.lid=port->heard.lid;memcpy(said.gid,port->heard.gid,16);
    said.node=port->peer;said.bandwidth=port->bandwidth;
  }
  child_send(port,said);
}
/* tag pre-return-20261103 rdma/mesh-link.c:1412-1451 child_heard, verbatim but extern (1f9be1d5 76be3217 b53804bf 306b22be) */
/* the child's messages, each whole one taken: an open's, a make's or a connect's end for the job out (a connect of a
   pairing since lost has its streams cancelled again: the child may have opened them after the loss's cancel); the
   lanes a progress thread found failed, for the link's pairing while it is up; a fault absorbed, a guard not held, a
   termination heard, each its port's bit for the event loop's conclusion (terminations_turn, ports_stalled).  The
   socket's end closes it; the child's exit is reaped (child_reaped) */
void child_heard(struct port *port){
  struct link *l=&port->link;
  const uint32_t bit=UINT32_C(1)<<port->index;
  for(;;){
    const ssize_t got=recv(port->child_fd,port->child_in+port->child_held,sizeof port->child_in-port->child_held,MSG_DONTWAIT);
    if(got<0 && (errno==EAGAIN || errno==EINTR))return;
    if(got<=0){close(port->child_fd);port->child_fd=-1;return;}
    if((port->child_held+=(uint32_t)got)<sizeof port->child_in)continue;
    port->child_held=0;
    struct child_message heard;
    memcpy(&heard,port->child_in,sizeof heard);
    const int current=heard.pairing==atomic_load_explicit(&l->pairing,memory_order_relaxed);
    if(heard.kind==CHILD_OPENED && port->job){
      port->result=heard.result;port->refused=heard.refused;
      if(!heard.result){port->lid=heard.lid;memcpy(port->gid,heard.gid,16);port->bandwidth=heard.bandwidth;port->budget=heard.lanes;}
      atomic_store_explicit(&port->done,1,memory_order_release);
    }
    else if(heard.kind==CHILD_MADE || heard.kind==CHILD_CONNECTED){
      if(heard.kind==CHILD_CONNECTED && !current)streams_cancel(port);
      if(l->job!=(heard.kind==CHILD_MADE?LINK_JOB_MAKE:LINK_JOB_CONNECT) || heard.pairing!=l->job_pairing)continue;
      l->result=heard.result;
      if(heard.kind==CHILD_MADE && !heard.result){
        memcpy(l->qpn,heard.qpn,sizeof l->qpn);l->psn=heard.psn;l->share=(struct frames){heard.share};
        port->lid=heard.lid;memcpy(port->gid,heard.gid,16);
      }
      atomic_store_explicit(&l->done,1,memory_order_release);
    }
    else if(heard.kind==CHILD_BROKEN && current && l->state==LINK_UP){
      if(heard.result)atomic_store_explicit(&l->lost,heard.result,memory_order_relaxed);
      atomic_fetch_or(&l->broken,heard.broken);
    }
    else if(heard.kind==CHILD_FAULTED)atomic_fetch_or(&faulted,bit);
    else if(heard.kind==CHILD_TERMINATED)atomic_fetch_or(&terminated,bit);
  }
}
/* tag pre-return-20261103 rdma/mesh-link.c:1452-1460 child_turn, verbatim but extern (1f9be1d5 b78c7712 90179f5e d556370e) */
/* a port whose link was lost once its lanes were asked for (`recycle`) asks its child to end once its quiet window and
   the exit's have ended (B9, B9b: no release inside a termination) and no job of its own is out (B12) */
void child_turn(struct port *port){
  if(!port->recycle || !port->child || port->ending || port->job || port->link.job || now<atomic_load(&port->quiet_at) ||
     now<atomic_load(&exit_at))return;
  child_send(port,(struct child_message){.kind=CHILD_END});
  mesh_say("bridge","link %u (%s): child %d asked to end",port->index,port->name,(int)port->child);
  port->ending=1;
}
/* tag pre-return-20261103 rdma/mesh-link.c:1461-1496 child_reaped, extern, no stream stores (b346ef2b 0d14fb73 90179f5e 2548cbce) */
/* the port's child ended: its last messages taken, its exit reaped, a job of it out ended (ECHILD); one that ended
   unbidden loses the link (port_lost, ECHILD) and is spawned again once the port's quiet window has passed, at the
   events that open any port, however often its children ended (the operator, 2026-10-23 01:03Z: "this is not a real feature.
   delete it and anything like it").  Its streams cancelled again,
   the region's closed leases looked at again (`unreleased`: the rows its units named are held no more, its queue
   pairs gone with it, mesh-dataflow.c mesh_holding_retained), its link down with no lanes, and the port to be opened by its next
   child */
void child_reaped(struct port *port){
  struct link *l=&port->link;
  const pid_t pid=port->child;
  int status=0;
  if(port->child_fd>=0)child_heard(port);
  if(port->child_fd>=0){close(port->child_fd);port->child_fd=-1;}
  while(waitpid(pid,&status,0)<0 && errno==EINTR){}
  mesh_say("bridge","link %u (%s): child %d ended%s",port->index,port->name,(int)pid,port->ending?"":" unasked");
  port->child=0;
  /* differs from the tag: its stores of 0 in each of the port's streams' `child` (MESH_LINK_STREAMS of
     stream_of(port,t)->child, 2548cbce: the reaped child's proof for a reclaim, design/reclamation-2026-10-30.md) are
     not here, there being no streams under the page lists (mesh-bridge.h); what a stream's end is now is
     streams_cancel's, below (mesh-flow.c) */
  if(port->holder<2*MESH_PORTS && mesh_bridge_pages(M)->child[port->holder].pid==(uint32_t)pid)
    mesh_bridge_pages(M)->child[port->holder]=(struct mesh_holder){0};
  const int asked=port->ending;
  port->ending=port->recycle=0;
  if(!asked){
    port->open=0;
    port_lost(port,ECHILD);
  }
  if(port->job){port->result=ECHILD;port->refused=0;atomic_store_explicit(&port->done,1,memory_order_release);}
  if(l->job){l->result=ECHILD;atomic_store_explicit(&l->done,1,memory_order_release);}
  streams_cancel(port);
  unreleased=1;
  l->state=LINK_DOWN;l->lanes=l->heard=0;l->share=(struct frames){0};l->plan=0;
  memset(l->qpn,0,sizeof l->qpn);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
  port->opening=1;
}

/* tag pre-return-20261103 rdma/mesh-link.c:2079-2121, main after the windows' hold, verbatim as a function (3f72cbde 90179f5e 07949b28) */
void children_end(void){
  /* This bridge ends after its children, never before.  Each is asked to end (CHILD_END: it closes, its lanes
     flushed, child_leave), awaited MESH_CHILD_END_NS and reaped as it ends (waitpid: its exit whole, the kernel's
     release of its device's objects and its pages done); one not reaped by then is ended (SIGKILL) and awaited
     MESH_CHILD_END_NS more.  One the kernel holds in its exit past that stays named in the region's holder table (struct
     mesh_holder: its pid, device and start), so no later launch's child opens its device while it lives
     (child_elsewhere), and it is named here.  Each exit heard on a queue of the children's exits alone (EVFILT_PROC) */
  for(uint32_t p=0;p<port_count;p++)child_send(ports+p,(struct child_message){.kind=CHILD_END});
  const int exits=kqueue();
  for(uint32_t p=0;exits>=0 && p<port_count;p++)if(ports[p].child){
    struct kevent64_s watch;EV_SET64(&watch,(uint64_t)ports[p].child,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,p,0,0);
    kevent64(exits,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
  }
  const uint64_t asked=clock_ns()+MESH_CHILD_END_NS,killed=asked+MESH_CHILD_END_NS;
  for(int kill_sent=0;;){
    uint32_t left=0;
    for(uint32_t p=0;p<port_count;p++)if(ports[p].child){
      if(!waitpid(ports[p].child,NULL,WNOHANG))left++;
      else ports[p].child=0;
    }
    const uint64_t at=clock_ns();
    if(!left)break;
    if(at>=asked && !kill_sent){
      kill_sent=1;
      for(uint32_t p=0;p<port_count;p++)if(ports[p].child){
        mesh_say("bridge","link %u (%s): child %d has not ended %.1f s after it was asked: ended (SIGKILL)",ports[p].index,
                 ports[p].name,(int)ports[p].child,(double)MESH_CHILD_END_NS/1e9);
        kill(ports[p].child,SIGKILL);
      }
    }
    if(at>=killed){
      for(uint32_t p=0;p<port_count;p++)if(ports[p].child)
        mesh_say("bridge","link %u (%s): child %d not reaped %.1f s after its SIGKILL: its exit waits in the kernel, named "
                 "in the region's holders, so no later child opens its device while it lives",ports[p].index,ports[p].name,
                 (int)ports[p].child,(double)MESH_CHILD_END_NS/1e9);
      break;
    }
    const uint64_t until=kill_sent?killed:asked;
    struct kevent64_s ended;
    struct timespec rest={(time_t)((until-at)/1000000000u),(long)((until-at)%1000000000u)};
    if(exits<0 || (kevent64(exits,NULL,0,&ended,1,0,&rest)<0 && errno!=EINTR))poll(NULL,0,(int)((until-at+999999)/1000000));
  }
  /* differs from the tag: its `return 0;` here is main's own (mesh-link.c), after it calls this */
}
