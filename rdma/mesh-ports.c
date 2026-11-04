/* The bridge's ports and their self-pairing, in the parent: the devices found (ports_find, ports_found), each port's
   device opened by its child (port_open, port_opened), its hellos (port_hello, hello_heard), its control socket's
   pairing exchange (port_greet, control_heard, port_heard, message_heard, link_heard, port_say and its queue), its link's
   turn and loss (link_turn, ports_stalled, port_lost, link_lost, link_tell, link_set) and the map's cables (record_reach).
   From tag pre-return-20261103 rdma/mesh-link.c, verbatim but for interfaces (design/page-queue-change-request.md, "Queue
   pairs and links belong to the bridge", "Relays", "The rule both ends compute", "Replacing the --link form"): the
   bridge pairs its own ports, each port one cable, with no --link, -I, oob, dial, listener_up or per-session exchange;
   the queue pairs a cable's pairing makes are the failport's budget, both ends' smaller (port_heard, 76be3217; the rule
   of bd85613e).  A node reached is one a Thunderbolt 5 cable of this node pairs with: no other node takes part.  Where
   the coordinator's pairing messages (CONTRACT.md "Pairing messages") differ from the tag, a note says so at the spot.
   Every failure is port_lost with a code; nothing here waits. */
#include "mesh-bridge.h"


/* The coordinator's pairing messages (CONTRACT.md "Pairing messages"), which differ from the tag, where a port's
   control socket stayed open for its link's life: the socket carries one exchange (HELLO, the records the map holds
   then, LINK) and ends once the link is up; RECORD and FAREWELL then go as datagrams on the hello socket (port_say).
   So the tag's `port->control>=0`, where it meant "this port has a peer or a pairing under way", is this: its control
   socket open (dialed, accepted, exchanging), or its peer's HELLO heard (`greeted`) and the link not lost since, its
   socket closed once the exchange ended.  A socket closes without a loss only once greeted (link_turn, port_flush,
   control_heard), and port_lost clears both together.  What must stay: one test, read wherever the tag read
   `control>=0` for a peer (port_open, port_opened, port_hello; mesh-link.c's accept, turn_due; mesh-events.c
   arrivals_turn) */
/* port_paired is mesh-bridge.h's, where every file's test reads it */
/* the coordinator's pairing messages: each port's peer at its link-local address and the mesh's port (MESH_PORT, the
   hello socket's, on the port's interface), read from the control socket as it is greeted (port_greet): where
   port_say's datagrams go.  A port's, here until mesh-bridge.h's struct port carries it */
static struct sockaddr_in6 peer_at[MESH_PORTS];

/* tag pre-return-20261103 rdma/mesh-link.c:16 `finding`, extern as mesh-bridge.h declares it (3f72cbde 0905738c 1835bef4 5dcb4f69; first 6b33cc33) */
int finding;
/* tag pre-return-20261103 rdma/mesh-link.c:19 `finder`, extern (static there) (6b33cc33) */
dispatch_queue_t finder;
/* tag pre-return-20261103 rdma/mesh-link.c:20-22 `find_wanted`, extern (static there) (3f72cbde 0905738c cf483b3e 7c500c88; first 5b49fbb9) */
/* design/transport.md: the devices to be read again (an interface that no port is attached or its link came on), read
   on the event loop's turn once no window stands and no call is stuck (ports_find) */
int find_wanted;
/* tag pre-return-20261103 rdma/mesh-link.c:23-25 `stalled`, extern (static there) (306b22be 3a82ede9 b78c7712 3f72cbde; first 5b49fbb9) */
/* design/transport.md B12: a port's job out past MESH_PLATFORM_NS, during which the devices are not read
   (ports_stalled's, each event-loop turn) */
_Atomic int stalled;
/* tag pre-return-20261103 rdma/mesh-link.c:43-46 `found`, `devices`, `device_count`, `everyone`, verbatim but `everyone` extern (3f72cbde) */
_Atomic int found;
static char devices[MESH_PORTS][32];
static uint32_t device_count;
const struct in6_addr everyone={{{0xff,0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}};

/* tag pre-return-20261103 rdma/mesh-link.c:475-480 port_watch, verbatim (3f72cbde 33ca0dab 128d0e84 a8952de9; first 6b33cc33) */
/* design/transport.md: a message to the port's peer, queued behind what the socket has not taken yet (a node's record
   in place of its older one: port_coalesce); a send that fails, or a queue the process cannot grow, is the link's loss */
static void port_watch(struct port *port,int filter){
  struct kevent64_s event;EV_SET64(&event,(uint64_t)port->control,filter,EV_ADD|(filter==EVFILT_WRITE?EV_ONESHOT:0),0,0,port->index,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))port_lost(port,errno);
}
/* tag pre-return-20261103 rdma/mesh-link.c:481-492 port_flush, extern, + its write side's end once the link is up (ceff26ec; first 6b33cc33) */
void port_flush(struct port *port){
  while(port->control>=0 && port->out_sent<port->out_length){
    ssize_t sent=send(port->control,port->out+port->out_sent,port->out_length-port->out_sent,0);
    if(sent<0 && (errno==EAGAIN || errno==EINTR)){
      if(!port->writing){port->writing=1;port_watch(port,EVFILT_WRITE);}
      return;
    }
    if(sent<=0){port_lost(port,sent?errno:EIO);return;}
    port->out_sent+=(size_t)sent;
  }
  /* Differs from the tag (CONTRACT.md "Pairing messages"): once the link is up, the queue sent whole is the socket's
     last, and this end's write side is shut (its end read at the peer, control_heard; the socket closed here at the
     peer's end).  A shut the socket refuses is the link's loss, as a send it refuses is.  At the tag the socket stayed
     for the link's life */
  const int ended=port->control>=0 && port->out_length && port->link.state==LINK_UP;
  port->out_sent=port->out_length=0;
  if(ended && shutdown(port->control,SHUT_WR))port_lost(port,errno);
}
/* tag pre-return-20261103 rdma/mesh-link.c:493-514 port_coalesce, verbatim (3f72cbde ad55cbf4 33ca0dab 128d0e84; first 6b33cc33) */
/* a RECORD whose node has a RECORD waiting wholly unsent in the port's queue takes that one's place where it is no older
   (RFC 2328 s13: a node's newest record is the one that counts), so a peer that reads nothing holds a record a node in
   the queue; 1 where it did */
static int port_coalesce(struct port *port,const struct message *said,const void *payload,uint32_t bytes){
  struct mesh_record fresh,queued;
  if(said->kind!=RECORD || bytes!=sizeof fresh)return 0;
  memcpy(&fresh,payload,sizeof fresh);
  for(size_t at=0;at+sizeof(struct message)<=port->out_length;){
    struct message header;
    memcpy(&header,port->out+at,sizeof header);
    if(at>=port->out_sent && header.kind==RECORD && header.bytes==sizeof queued){
      memcpy(&queued,port->out+at+sizeof header,sizeof queued);
      if(queued.node==fresh.node){
        if(fresh.boot>queued.boot || (fresh.boot==queued.boot && fresh.sequence>=queued.sequence))
          memcpy(port->out+at+sizeof header,&fresh,sizeof fresh);
        return 1;
      }
    }
    at+=sizeof header+header.bytes;
  }
  return 0;
}
/* tag pre-return-20261103 rdma/mesh-link.c:515-529 port_say, extern, XMAGIC as MESH_WIRE_MAGIC, + its datagrams (3f72cbde ad55cbf4 33ca0dab 8fc0d3d7; first 6b33cc33) */
/* Differs from the tag (CONTRACT.md "Pairing messages", the coordinator's instruction): the control socket carries the
   pairing's exchange alone (HELLO, the records the map holds then, LINK) and ends once the link is up (link_turn,
   port_flush: this end's write side shut; control_heard: the socket closed at the peer's end).  A RECORD after that is a
   datagram on the hello socket to the peer's link-local address (peer_at), and a FAREWELL is one always: so port_lost
   tells the peer of a loss without the socket's end, which no longer tells one, and without a failed send losing the
   port again inside its loss.  A datagram the kernel has no room for (EAGAIN, ENOBUFS) is dropped, as the tag's queue
   took the socket's EAGAIN, and the record goes again (this node's each MESH_PAUSE_NS, record_retell; any node's at its
   next sequence, its health sample at the latest); any other refusal of a RECORD is the link's loss, as a failed send
   was at the tag, and of a FAREWELL nothing (its port is lost, or being lost).  What must stay: HELLO and LINK only ever
   on the socket, and a RECORD on it, in order behind what it queued, while the socket is this end's to write */
void port_say(struct port *port,struct message said,const void *payload,uint32_t bytes){
  const int datagram=port->greeted && (said.kind==FAREWELL || port->control<0 || (port->link.state==LINK_UP && !port->out_length));
  if((port->control<0 && !datagram) || (said.kind!=HELLO && !port->greeted) || (!datagram && port_coalesce(port,&said,payload,bytes)))return;
  said.magic=MESH_WIRE_MAGIC;said.version=MESH_VERSION;said.node=M->node;said.boot=M->boot;said.block=(uint32_t)page_bytes();said.port=port->index;said.bytes=bytes;
  said.lid=port->lid;memcpy(said.gid,port->gid,16);
  if(datagram){
    unsigned char out[sizeof said+sizeof(struct mesh_record)];
    if((said.kind!=RECORD && said.kind!=FAREWELL) || bytes>sizeof out-sizeof said)return;
    memcpy(out,&said,sizeof said);
    if(bytes)memcpy(out+sizeof said,payload,bytes);
    if(sendto(hello,out,sizeof said+bytes,0,(const struct sockaddr *)&peer_at[port->index],sizeof peer_at[port->index])<0 &&
       said.kind==RECORD && errno!=EAGAIN && errno!=EINTR && errno!=ENOBUFS)port_lost(port,errno);
    return;
  }
  size_t length=port->out_length+sizeof said+bytes;
  if(length>port->out_room){
    unsigned char *grown=realloc(port->out,length*2);
    if(!grown){port_lost(port,ENOMEM);return;}
    port->out=grown;port->out_room=length*2;
  }
  memcpy(port->out+port->out_length,&said,sizeof said);
  if(bytes)memcpy(port->out+port->out_length+sizeof said,payload,bytes);
  port->out_length=length;
  port_flush(port);
}
/* tag pre-return-20261103 rdma/mesh-link.c:534-544 record_reach, extern, a cable's ports and queue pairs (601a219e f04b13b3 1692c638 3f72cbde; first 6b33cc33) */
/* design/transport.md section 14: this node's reach as its record tells it (mesh-records.c record_changed): an entry
   a cable whose link is up with a stream in its plan, its peer node and bandwidth, at most `room`; their count.  The
   cables, not the streams live this instant, so two nodes several cables join are as many parallel links of the map
   (mesh-dataflow.c mesh_join), which a cable's link coming up or lost moves (link_turn, link_set) */
/* Differs from the tag (mesh.h struct mesh_reach, 17612857): the streams went, so a cable reaches where its link is up
   with a queue pair in its plan (plan_lanes, the tag's plan_streams), and its entry names both ends' ports and the
   queue pairs its pairing negotiated, the plan's (link_plan of both ends' smaller budget, port_heard), the same at
   both ends: the rule's inputs (the change request, "The rule both ends compute") */
uint32_t record_reach(struct mesh_reach *reach,uint32_t room){
  uint32_t count=0;
  for(uint32_t p=0;p<port_count && count<room;p++)
    if(ports[p].link.state==LINK_UP && plan_lanes(ports[p].link.plan))
      reach[count++]=(struct mesh_reach){.node=ports[p].peer,.port=ports[p].index,.peer_port=ports[p].peer_port,
                                         .qps=plan_lanes(ports[p].link.plan),.bandwidth=ports[p].bandwidth};
  return count;
}
/* tag pre-return-20261103 rdma/mesh-link.c:545-552 link_set, main's struct mesh_link_info (3f72cbde; first 6b33cc33) */
/* a port's link as the port table tells it; a change is a new record */
/* Differs from the tag: main's struct mesh_link_info (mesh.h) keeps `peer` as a plain word, stored and loaded as one
   (mesh-flow.c's --link form: __atomic_*), its phase and code in `port`, and the port's `device`, which the tag's had
   not: named here, the same name every time, as the --link form names it once */
static void link_set(struct port *port,uint32_t peer,uint32_t phase,int64_t code){
  struct mesh_link_info *link=&mesh_links(M)[port->index];
  int changed=__atomic_load_n(&link->peer,__ATOMIC_RELAXED)!=peer || atomic_load(&link->port.phase)!=phase;
  snprintf(link->device,sizeof link->device,"%s",port->device.name);
  __atomic_store_n(&link->peer,peer,__ATOMIC_RELAXED);atomic_store_explicit(&link->port.code,code,memory_order_relaxed);
  atomic_store_explicit(&link->port.phase,phase,memory_order_release);
  if(changed)record_changed();
}

/* tag pre-return-20261103 rdma/mesh-link.c:554-571, the link jobs' description, struct link_numbers and link_lapsed, are
   mesh-lanes.c's (ae3c36f9), beside their only callers, the child's link jobs: none of the functions here reads them */

/* tag pre-return-20261103 rdma/mesh-link.c:673-680 link_tell, verbatim (3f72cbde d9e6d3de 33ca0dab db2748a9; first 6b33cc33) */
/* a LINK telling the peer every lane of this end's pairing, the lanes of its plan (`count` theirs): each one's queue pair
   number and first sequence number, as its child told them (CHILD_MADE), 0 for a lane the plan leaves out */
static void link_tell(struct port *port){
  const struct link *l=&port->link;
  struct message said={.kind=LINK,.count=plan_lanes(l->plan)};
  for(uint32_t k=0;k<MESH_LINK_LANES;k++)if(l->plan>>k&1){said.pair[k][0]=l->qpn[k];said.pair[k][1]=(l->psn+k)&0xffffff;}
  port_say(port,said,NULL,0);
}
/* tag pre-return-20261103 rdma/mesh-link.c:681-690 link_lost, verbatim (3f72cbde d9e6d3de 33ca0dab db2748a9; first 6b33cc33) */
/* the link of a port lost (its peer's loss, a failure, the bridge stopping): its lanes served no more and left as they
   are, no verbs call made, and, once connected, never paired again in this bridge's life (design/transport.md section
   14); a job of this pairing still on the port's queue makes no call and is dropped as it ends (B10) */
static void link_lost(struct port *port){
  struct link *l=&port->link;
  atomic_fetch_add_explicit(&l->pairing,1,memory_order_relaxed);l->state=LINK_DOWN;l->heard=0;
  atomic_store_explicit(&l->served,0,memory_order_release);
  atomic_store_explicit(&l->broken,0,memory_order_relaxed);
  atomic_store_explicit(&l->lost,0,memory_order_relaxed);
}
/* tag pre-return-20261103 rdma/mesh-link.c:723-766 link_turn, extern, no link page, the socket's end at the link's coming up (11807bec abeffc1b b78c7712 3f72cbde; first 6b33cc33) */
/* design/transport.md section 14: the link's turn, on the event loop: a job's end taken (made: LINK told the peer every
   lane; connected: up, every lane served, the stream live); a job that failed loses the link (port_lost); while it is
   up, a lane a progress thread found failed (`broken`) loses it; then, while no port's job is out past MESH_PLATFORM_NS (B12) and once the port's last
   quiet window has ended (B9), the lanes made once the port is paired (`lanes`: its child asked for them), and connected
   once both ends' are made, each by the port's child and once in its life: a link lost once its lanes were asked for
   pairs again with the port's next child (design/transport.md section 14, "A process a port": port_lost, child_turn,
   child_reaped) */
void link_turn(struct port *port){
  struct link *l=&port->link;
  if(l->job){
    if(!atomic_load_explicit(&l->done,memory_order_acquire))return;
    const int job=l->job,result=l->result;
    l->job=0;
    if(l->job_pairing!=atomic_load_explicit(&l->pairing,memory_order_relaxed)){}
    else if(result){
      port_lost(port,result);
      return;
    }
    else if(job==LINK_JOB_MAKE){l->state=LINK_MADE;link_tell(port);}
    else{
      l->state=LINK_UP;
      /* Differs from the tag (CONTRACT.md "Pairing messages"): the link up, the control socket's exchange is over, so
         this end's write side is shut once its queue is sent (here where it is, else as port_flush sends the last of
         it), its records going as datagrams from then (port_say); the socket is read to its end at the peer's shut
         (control_heard) and closed there.  A shut the socket refuses is the link's loss, as a send it refuses is */
      if(port->control>=0 && !port->out_length && shutdown(port->control,SHUT_WR)){port_lost(port,errno);return;}
      record_changed();
    }
  }
  const int held=now<port->quiet_at;
  if(!port->greeted)return;
  if(l->state==LINK_UP){
    const uint32_t broken=atomic_exchange_explicit(&l->broken,0,memory_order_acq_rel);
    const int lost=atomic_exchange_explicit(&l->lost,0,memory_order_relaxed);
    if(broken){port_lost(port,lost?lost:EIO);return;}
  }
  if(l->job || held)return;
  if(l->state==LINK_DOWN){
    /* Differs from the tag (CONTRACT.md "Link pages"): no link block is leased (the tag's lease of `l->pages` at
       `l->page` from the arena, ENOMEM where none, as the link first paired): this design posts nothing on a lane that
       carries nothing, so `l->page` stays MESH_ABSENT */
    l->lanes=1;
    child_job(port,LINK_JOB_MAKE);
  }
  else if(l->state==LINK_MADE && l->heard){l->state=LINK_CONNECTING;child_job(port,LINK_JOB_CONNECT);}
}
/* tag pre-return-20261103 rdma/mesh-link.c:767-802 ports_stalled, extern (static there) (d556370e abeffc1b 306b22be 3a82ede9; first 6b33cc33) */
/* design/transport.md B12 (22891630^ mesh-flight.h's pending-operation age, per port: its dispatch's
   `job_at` in place of flight_begin's stamp, its `done` in place of flight_end's): a port whose open's or link's job
   is still out MESH_PLATFORM_NS after it began is concluded around
   it, once: open 0 and port_lost (ETIMEDOUT: its socket closed, the map's phase STOPPED, its stream's words ~0), no
   verbs call.  The call itself cannot
   be ended.  While any port's job is so out the devices are not read (`stalled`: ports_find, the parent's own driver
   entry), and the port itself takes no job until the call returns (`stuck`, its job's word); each port's calls are its
   own child's, a process of its own (section 14), so one port's stuck call holds no other port's (a hold of every
   port's was 10-18's, when one process made every port's calls).  Each conclusion moves the port's epoch and wakes its threads; the port stays
   `stuck` until the call returns and each progress thread has told the conclusion's epoch (a thread tells it on its
   first pass after its call returns, port_pass, or as it wakes where it is in none), then opens again (`opening`,
   port_open past its quiet window, B9; an open's own return is port_opened's) and so takes jobs again.  Read on the
   event loop's turn */
void ports_stalled(void){
  int any=0;
  for(uint32_t p=0;p<port_count;p++){
    struct port *port=ports+p;
    const struct link *l=&port->link;
    const int opening=port->job && !atomic_load_explicit(&port->done,memory_order_acquire) && now-port->job_at>=MESH_PLATFORM_NS;
    const int linking=l->job && !atomic_load_explicit(&l->done,memory_order_acquire) && now-l->job_at>=MESH_PLATFORM_NS;
    const int out=opening||linking;
    any|=out;
    if(port->stuck){
      const int returned=!out && (!port->threads || atomic_load_explicit(&port->seen,memory_order_acquire)>=atomic_load(&port->epoch));
      if(returned){port->stuck=0;if(!port->job)port->opening=1;}
      continue;
    }
    if(!out)continue;
    port->stuck=1;atomic_fetch_add(&port->epoch,1);guard_wake(port);
    mesh_say("bridge","link %u (%s): its %s job out %.1f s: the port concluded around it",port->index,port->name,
             opening?"open":"lane",(double)(now-(opening?port->job_at:l->job_at))/1e9);
    port->open=0;
    port_lost(port,ETIMEDOUT);
  }
  atomic_store_explicit(&stalled,any,memory_order_relaxed);
}
/* tag pre-return-20261103 rdma/mesh-link.c:803-821 link_heard, verbatim (abeffc1b 3f72cbde 0905738c cf483b3e; 76be3217; first 6b33cc33) */
/* a peer's LINK: every lane of its pairing and the address they were made on (its port's LID and GID as its make read
   them, which its hello's may predate: the connect's address, and an end of each stream's identity alike at both ends,
   stream_wire), taken once, before this end's connect (a LINK heard again is let be: the first stands), its lanes this
   end's plan's (both ends plan from the same two budgets: design/HETEROGENEOUS-LINKS.md); one naming other lanes loses
   the link, since a lane can be connected only to a queue pair its peer made (a port whose child was asked for its
   lanes greets no peer once its link is lost until its next child opens it: design/transport.md section 14, "A
   process a port") */
static void link_heard(struct port *port,const struct message *heard){
  struct link *l=&port->link;
  if(l->heard || (l->state!=LINK_DOWN && l->state!=LINK_MADE))return;
  if(heard->count!=plan_lanes(l->plan)){
    port_lost(port,EPROTO);
    return;
  }
  memcpy(l->peer,heard->pair,sizeof l->peer);
  port->heard.lid=heard->lid;memcpy(port->heard.gid,heard->gid,16);
  l->heard=1;
  link_turn(port);
}


/* tag pre-return-20261103 rdma/mesh-link.c:824-831 message_heard, verbatim (3f72cbde 0905738c cf483b3e 1835bef4; 76be3217; first 6b33cc33) */
/* design/transport.md: the bridge's wire, a message a kind: a record, the peer's farewell (its "release
   everything": the link lost at once, so nothing is posted toward it again, design/transport.md) or its
   LINK; a message of any other kind is let be (AGENTS.md, "Fail toward access": unknown input, keep going) */
static void message_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(heard->kind==RECORD)record_heard(port,payload,heard->bytes);
  else if(heard->kind==FAREWELL){port->parted=1;port->farewell=heard->code;port_lost(port,ECONNRESET);}
  else if(heard->kind==LINK)link_heard(port,heard);
}
/* tag pre-return-20261103 rdma/mesh-link.c:832-856 port_heard, the budget both ends' smaller (MIN, 76be3217; bd85613e), main's struct mesh_link_info (abeffc1b 4bb01da0 3f72cbde 0905738c; first 6b33cc33) */
/* a message heard on a port's control socket: before its peer's hello, that hello alone (any other let be); the hello
   names the peer (the node it states, whatever node's hello this end dialed), and pairs the link, its plan the fewer
   queue pairs of the two ends', design/HETEROGENEOUS-LINKS.md, its lanes made (link_turn). The allocation-block equality
   remains required by legacy generic collectives' shared packet grid; exact byte-span DMA itself has no such
   restriction. Remove it only after every front end negotiates matching packet cuts. Then each message is taken
   (message_heard) */
static void port_heard(struct port *port,const struct message *heard,const unsigned char *payload){
  if(!port->greeted){
    if(heard->kind!=HELLO)return;
    if(heard->block!=page_bytes()){port_lost(port,EPROTO);return;}
    port->peer=heard->node;port->peer_port=heard->port;port->heard=*heard;port->greeted=1;
    const uint32_t theirs=hello_budget(heard->count);
    port->link.plan=link_plan(MIN(port->budget,theirs));
    struct mesh_link_info *link=&mesh_links(M)[port->index];
    /* Differs from the tag: main's struct mesh_link_info keeps `bandwidth` as a plain word, stored as one (mesh-flow.c's
       --link form: __atomic_store_n), and has no `pairings`, the tag's count of a port's pairings, which nothing of
       main reads: none is counted */
    __atomic_store_n(&link->bandwidth,port->bandwidth,__ATOMIC_RELAXED);
    link_set(port,port->peer,MESH_PAIRED,0);
    /* a port newly paired is sent every record the map holds, this node's first */
    for(uint64_t i=0;port->greeted && i<atomic_load(&M->map_nodes);i++)if(mesh_records(M)[i].node!=M->node)
      port_say(port,(struct message){.kind=RECORD},mesh_records(M)+i,sizeof(struct mesh_record));
    if(port->greeted)link_turn(port);
    return;
  }
  message_heard(port,heard,payload);
}

/* tag pre-return-20261103 rdma/mesh-link.c:858-882 port_lost, extern, + its FAREWELL to the peer (d556370e abeffc1b 3f72cbde 0905738c; first 6b33cc33) */
/* design/transport.md section 14, #loss: a link lost has its stream's landed words ~0 (streams_cancel: every client
   waiting on one reads the loss); its lanes are served no more before its control socket closes (its child told,
   CHILD_HALT: no post on them once the peer can read the loss); no verbs call is made (design/RDMA-KERNEL-RECOVERY.md
   "October 18").  A link whose child was asked for its lanes (`lanes`) waits for that child's end (`recycle`: child_turn
   asks it once the port's quiet window has ended), whose exit the kernel's release of the lanes is, the port closed and
   greeting no peer meanwhile, and pairs again with the port's next child ("A process a port"); a port lost before its
   lanes were asked for and still open says hello at once (port_hello: the records go on).  Its quiet window begins (B9:
   no port job on it, and no exit, for MESH_PLATFORM_NS, the operator's bound on the termination a loss may follow) */
void port_lost(struct port *port,int64_t code){
  if(port->greeted||port->connecting)
    mesh_say("bridge","link %u (%s): toward node %u lost (code %lld)%s",port->index,port->name,port->peer,(long long)code,
             stop?": this bridge is stopping":port->parted?": its peer's bridge said farewell":"");
  /* Differs from the tag (CONTRACT.md "Pairing messages"): the tag's peer read this end's loss as its control socket's
     end; with the socket ended once the exchange is over (and its end, once the peer's LINK is heard, the peer's link
     coming up: control_heard), a paired port's loss is told its peer in a FAREWELL, a datagram (port_say), the loss's
     code its `code`, unless the peer's own farewell lost it.  The peer loses the link on hearing it, as at the tag's
     FAREWELL (message_heard).  A FAREWELL that does not arrive is the peer's hello heard on the paired port's cable
     (hello_heard) */
  if(port->greeted && !port->parted)port_say(port,(struct message){.kind=FAREWELL,.code=code},NULL,0);
  quiet_until(&port->quiet_at,now+MESH_PLATFORM_NS);
  link_lost(port);
  if(port->child){
    child_send(port,(struct child_message){.kind=CHILD_HALT,.pairing=atomic_load(&port->link.pairing)});
    if(port->link.lanes && !stop){port->recycle=1;port->open=0;}
  }
  if(port->control>=0){close(port->control);port->control=-1;}
  if(port->greeted)port->next=now;
  streams_cancel(port);
  port->greeted=port->parted=port->connecting=port->writing=0;port->farewell=0;port->peer=0;port->held=0;port->out_length=port->out_sent=0;
  link_set(port,0,port->open?MESH_PAIRING:MESH_STOPPED,code);
}
/* tag pre-return-20261103 rdma/mesh-link.c:883-893 port_greet, extern, + the peer's address (3f72cbde 76be3217 ceff26ec; first 6b33cc33) */
/* the control socket: non-blocking, its sends queued (port_say), its HELLO stating the queue pairs the port's device
   holds for a link (hello_budget); its peer's loss is the kernel's to tell (its end closed, a reset, the link's going
   off) */
/* Differs from the tag (CONTRACT.md "Pairing messages"): the peer's address read from the socket as it is greeted, its
   port the hello socket's (MESH_PORT) and its scope this port's interface, so port_say's datagrams reach the peer on
   this cable once the socket has ended; a socket whose peer cannot be read is the link's loss, as one that cannot be
   made non-blocking is */
void port_greet(struct port *port){
  int on=1,flags=fcntl(port->control,F_GETFL);
  socklen_t length=sizeof peer_at[port->index];
  if(flags<0 || fcntl(port->control,F_SETFL,flags|O_NONBLOCK) || setsockopt(port->control,SOL_SOCKET,SO_NOSIGPIPE,&on,sizeof on) ||
     getpeername(port->control,(struct sockaddr *)&peer_at[port->index],&length)){
    port_lost(port,errno?errno:EIO);return;
  }
  peer_at[port->index].sin6_port=htons((uint16_t)atoi(MESH_PORT));peer_at[port->index].sin6_scope_id=port->interface;
  port_watch(port,EVFILT_READ);
  port_say(port,(struct message){.kind=HELLO,.count=1+port->budget},NULL,0);
}
/* tag pre-return-20261103 rdma/mesh-link.c:894-899 port_at, extern (3f72cbde ceff26ec; first 6b33cc33) */
/* design/transport.md: the port a link-local peer reached, by its interface; none for a scope of 0 (no
   interface: a LAN address, or the index of an interface that is gone) */
struct port *port_at(uint32_t interface){
  for(uint32_t p=0;interface && p<port_count;p++)if(ports[p].interface==interface)return ports+p;
  return NULL;
}
/* tag pre-return-20261103 rdma/mesh-link.c:900-921 port_open, extern, paired by port_paired (86457f4d b78c7712 3f72cbde 4dc9e98a; first 6b33cc33) */
/* design/transport.md: a port's device opened, on its own queue (guarded), where it is to be (at the start,
   at its link's coming on or its interface's attaching, at a hello heard while it is not open, while it has no peer,
   and as the call its port was concluded around returns, ports_stalled), once its quiet window has ended and no job of
   its own is out (design/transport.md B9, B12): the hellos' group joined on its
   interface, so the port hears them open or not (hello_heard, B16); its port read (none where the guard was not held
   within MESH_PLATFORM_NS: ETIMEDOUT, port_wait); a context that fails stays as it is, and the port with it, until its
   child ends (design/transport.md section 14: nothing made is forgotten or made again in a process).  The device is
   its child's: spawned here where the port has none (child_spawn: not while its last child's end is awaited, `recycle`,
   nor before the hold a spawn that failed set, `hold_to`), and asked to open it (CHILD_OPEN), its answer the
   job's end (child_heard) */
/* Differs from the tag: a port with a peer is one port_paired names (the tag's `port->control>=0`: the coordinator's
   pairing ends the socket while the port stays paired) */
void port_open(struct port *port){
  if(port->job || port->stuck || !port->opening || port_paired(port) || now<port->quiet_at || port->recycle || now<port->hold_to ||
     now<port->retry_at)return;
  if(!port->child && child_spawn(port))return;
  port->opening=0;
  port->interface=if_nametoindex(port->name);
  struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=port->interface};
  if(port->interface)setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
  port->job=1;port->job_at=now;atomic_store_explicit(&port->done,0,memory_order_relaxed);
  child_send(port,(struct child_message){.kind=CHILD_OPEN});
}
/* tag pre-return-20261103 rdma/mesh-link.c:922-934 device_open is mesh-lanes.c's (ae3c36f9): the child's, which the parent asks by CHILD_OPEN */
/* tag pre-return-20261103 rdma/mesh-link.c:935-957 port_opened, extern, paired by port_paired (ea04d19f abeffc1b 3f72cbde 02002590; first 6b33cc33) */
/* design/transport.md: a port's opening taken (its address and bandwidth as its child told them, CHILD_OPENED): a
   device that failed to open opens again at the next hello heard on it (hello_heard) or its link's next coming on; one
   whose registration of the region it refused (it keeps the extents it took: device_up logged the extent and why) does
   so too, the k-th refusal in a row no sooner than 60 2^(k-1) s on, at most an hour, unless its link comes on again
   (network_heard), since each try wires more of the node's memory (design/rubrics.md, Failure across interfaces 4 and
   6: the Mini's watchdog halted it once its memory was wired whole, 10-20); one opened says hello at once (port_hello) */
/* Differs from the tag: a port with a peer is one port_paired names (the tag's `port->control>=0`) */
void port_opened(struct port *port){
  const int result=port->result;
  port->job=0;
  if(result){
    if(port->refused){
      port->refusals++;
      const uint64_t hold=MIN((uint64_t)MESH_RETRY_FIRST_SECONDS<<MIN(port->refusals-1,MESH_RETRY_MAX_SHIFT),MESH_RETRY_MAX_SECONDS);
      port->retry_at=now+hold*UINT64_C(1000000000);
    }
    port->open=0;if(!port_paired(port))link_set(port,0,MESH_STOPPED,result);
    return;
  }
  port->open=1;port->refusals=0;
  if(port_paired(port))return;
  link_set(port,0,MESH_PAIRING,0);
  port->next=now;
}
/* tag pre-return-20261103 rdma/mesh-link.c:958-971 port_hello, extern, XMAGIC as MESH_WIRE_MAGIC, paired by port_paired (5c662972 3f72cbde 33ca0dab 8fc0d3d7; first 6b33cc33) */
/* design/transport.md: an open port with no peer says hello to every node on its link, every MESH_PAUSE_NS (a
   paced emission, deciding nothing: a peer that starts later is heard as it says its own) */
/* Differs from the tag: a port with a peer is one port_paired names (the tag's `port->control>=0`), so a paired port
   whose socket has ended says no hello */
void port_hello(struct port *port){
  if(!port->open || port_paired(port) || port->job || now<port->next)return;
  port->next=now+MESH_PAUSE_NS;
  setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_IF,&port->interface,sizeof port->interface);
  struct sockaddr_in6 to={.sin6_len=sizeof to,.sin6_family=AF_INET6,.sin6_port=htons((uint16_t)atoi(MESH_PORT)),.sin6_addr=everyone,.sin6_scope_id=port->interface};
  struct message said={.magic=MESH_WIRE_MAGIC,.version=MESH_VERSION,.kind=HELLO,.node=M->node,.boot=M->boot,.block=(uint32_t)page_bytes(),.port=port->index};
  /* a hello the system refuses (EHOSTUNREACH under Local Network privacy) leaves the port unheard: logged once a change */
  const int unheard=sendto(hello,&said,sizeof said,0,(struct sockaddr *)&to,sizeof to)<0?errno:0;
  if(unheard!=port->unheard && unheard)
    mesh_say("bridge","link %u (%s): hello not sent (%s): no peer on this link can hear it",port->index,port->name,strerror(unheard));
  port->unheard=unheard;
}
/* tag pre-return-20261103 rdma/mesh-link.c:972-994 hello_heard, extern, XMAGIC as MESH_WIRE_MAGIC, + the datagrams after the exchange (3f72cbde 33ca0dab 8fc0d3d7 7d236bbb; first 6b33cc33) */
/* design/transport.md: a node heard on a port: the lower node dials the higher's listener on the address
   it heard from, the higher answers its hello so the lower hears it at once; a port not open is opened (its open's job
   out aside), so a port whose link job or open failed pairs again on hearing its peer, never on a clock */
/* Differs from the tag (CONTRACT.md "Pairing messages"): the hello socket also carries a paired port's RECORDs and
   FAREWELLs once its control socket's exchange is over (port_say), so a datagram is read whole, a message and what its
   `bytes` say follows it, at most a record: one of those from the node a paired port's peer is is taken as the socket's
   message is (message_heard), any other let be.  And with the socket ended, a peer's loss is no longer the socket's end
   here: a port paired whose socket has ended hears its peer's hello only where that peer has no peer on this cable (an
   open port with no peer alone says hello, port_hello): its link lost at its end with a FAREWELL that did not arrive,
   or its bridge started again (a crash says none).  The link is lost here too (ECONNRESET, as at the peer's farewell),
   and the two ends pair again.  While a socket is open, a hello on its port is let be, as at the tag (one said before
   the dial can arrive after it) */
void hello_heard(void){
  struct message heard;struct sockaddr_in6 from;socklen_t length=sizeof from;
  unsigned char got[sizeof heard+sizeof(struct mesh_record)];
  for(ssize_t n;(n=recvfrom(hello,got,sizeof got,0,(struct sockaddr *)&from,&length))>=(ssize_t)sizeof heard;){
    memcpy(&heard,got,sizeof heard);
    struct port *port=port_at(from.sin6_scope_id);
    length=sizeof from;
    if(!port || heard.magic!=MESH_WIRE_MAGIC || heard.node==M->node)continue;
    if(heard.kind!=HELLO){
      if(port->greeted && heard.node==port->peer && (heard.kind==RECORD || heard.kind==FAREWELL) &&
         (size_t)n==sizeof heard+heard.bytes)message_heard(port,&heard,got+sizeof heard);
      continue;
    }
    if((size_t)n!=sizeof heard || port->control>=0)continue;
    if(port->greeted){if(heard.node==port->peer)port_lost(port,ECONNRESET);continue;}
    if(!port->open){if(!port->job)port->opening=1;continue;}
    if(M->node>heard.node){
      struct message said={.magic=MESH_WIRE_MAGIC,.version=MESH_VERSION,.kind=HELLO,.node=M->node,.boot=M->boot,.block=(uint32_t)page_bytes(),.port=port->index};
      sendto(hello,&said,sizeof said,0,(struct sockaddr *)&from,sizeof from);
      continue;
    }
    from.sin6_port=htons((uint16_t)atoi(MESH_PORT));
    port->control=socket(AF_INET6,SOCK_STREAM,0);
    if(port->control<0 || fcntl(port->control,F_SETFL,O_NONBLOCK) ||
       (connect(port->control,(struct sockaddr *)&from,sizeof from) && errno!=EINPROGRESS)){port_lost(port,errno);continue;}
    port->peer=heard.node;port->connecting=1;
    port_watch(port,EVFILT_WRITE);
  }
}
/* the coordinator's pairing messages (CONTRACT.md): a control socket whose exchange is over closed, its queues emptied
   as port_lost empties them (tag rdma/mesh-link.c:877, :880), the port still paired (`greeted`, port_paired) */
static void control_end(struct port *port){
  close(port->control);port->control=-1;port->held=0;port->out_length=port->out_sent=0;port->writing=0;
}
/* tag pre-return-20261103 rdma/mesh-link.c:995-1021 control_heard, extern, XMAGIC as MESH_WIRE_MAGIC, the socket's end once the exchange is over (ea04d19f 3f72cbde 7c500c88 0e34c13e; first 6b33cc33) */
/* a control socket's messages, each a header and its `bytes`, each whole one taken */
void control_heard(struct port *port){
  for(;;){
    size_t need=sizeof(struct message);
    if(port->held>=need){
      const struct message *header=(const void *)port->in;
      if(header->magic!=MESH_WIRE_MAGIC || header->bytes>MESH_TOPOLOGY_MESSAGE_BYTES){port_lost(port,EPROTO);return;}
      need+=header->bytes;
    }
    if(need>port->in_room){
      unsigned char *grown=realloc(port->in,need);
      if(!grown){port_lost(port,ENOMEM);return;}
      port->in=grown;port->in_room=need;
    }
    if(port->held<need){
      ssize_t got=recv(port->control,port->in+port->held,need-port->held,0);
      if(got<0 && (errno==EAGAIN || errno==EINTR))return;
      /* Differs from the tag (CONTRACT.md "Pairing messages"): the socket's end once the peer's LINK is heard and this
         end's told (its state past LINK_DOWN) is the peer's shut at its link's coming up (link_turn, port_flush), or its
         loss, which its FAREWELL tells (port_lost), not the end: the socket closed here, the port paired, its records
         going as datagrams (port_say).  Its end before then, or a failure, is the link's loss, as at the tag */
      if(!got && port->link.heard && port->link.state!=LINK_DOWN){control_end(port);return;}
      if(got<=0){port_lost(port,got?errno:ENETRESET);return;}
      port->held+=(size_t)got;
      if(port->held<need || need==sizeof(struct message))continue;
    }
    struct message heard;memcpy(&heard,port->in,sizeof heard);
    port->held=0;
    port_heard(port,&heard,port->in+sizeof heard);
    if(port->control<0)return;
  }
}
/* tag pre-return-20261103 rdma/mesh-link.c:1022-1041 link_page, link_reserve and their comments are left out (CONTRACT.md
   "Link pages": no link block is leased; `ones` and `held_pages` are mesh-bridge.h's externs, defined in mesh-link.c) */
/* tag pre-return-20261103 rdma/mesh-link.c:1042-1059 ports_find, extern (3f72cbde 0905738c 1835bef4 d9e6d3de; first 6b33cc33) */
/* design/transport.md: every RDMA device of the node a port, in the order found, read on the finder's queue
   (at the start, behind IOKit's quiet registry, and where an interface attached or its link came on that no port has,
   on the event loop's turn once the windows have ended and no call is stuck: find_wanted, design/transport.md
   B9, B9b, B12) and added on the event loop */
void ports_find(void){
  if(finding)return;
  finding=1;
  dispatch_async(finder,^{
    struct ibv_device **list=ibv_get_device_list(NULL);
    uint32_t n=0;
    for(int i=0;list && list[i] && n<MESH_PORTS;i++)if(!strncmp(ibv_get_device_name(list[i]),"rdma_",5))
      snprintf(devices[n++],sizeof devices[0],"%s",ibv_get_device_name(list[i]));
    if(list)ibv_free_device_list(list);
    device_count=n;
    atomic_store_explicit(&found,1,memory_order_release);
    wake();
  });
}
/* tag pre-return-20261103 rdma/mesh-link.c:1060-1081 ports_found, extern, no link page (11807bec 3f72cbde 0905738c 1835bef4; first 6b33cc33) */
void ports_found(void){
  finding=0;atomic_store_explicit(&found,0,memory_order_relaxed);
  for(uint32_t i=0;i<device_count && port_count<MESH_PORTS;i++){
    uint32_t known=0;
    for(uint32_t p=0;p<port_count;p++)known|=!strcmp(ports[p].device.name,devices[i]);
    if(known)continue;
    struct port *port=ports+port_count;
    *port=(struct port){.index=port_count,.control=-1,.child_fd=-1,.opening=1};
    /* Differs from the tag (CONTRACT.md "Link pages"): no link block is leased for the port's index (the tag's
       `link.pages` link_pages() and `link.page` link_page[port_count], link_reserve's), so `link.page` is MESH_ABSENT
       and `link.pages` 0 */
    port->link.page=MESH_ABSENT;
    port->link.provider.device=&port->device;
    snprintf(port->device.name,sizeof port->device.name,"%s",devices[i]);snprintf(port->name,sizeof port->name,"%s",devices[i]+5);
    /* the interface as the kernel's events name it, its name read once here into typed values (addresses_heard) */
    if(!strncmp(port->name,"en",2)){port->family=APPLE_IF_FAM_ETHERNET;port->unit=(uint32_t)strtoul(port->name+2,NULL,10);}
    /* design/transport.md B9b: counted for a termination's queue (quiet_all, peer_port) once whole, then
       held to the window a termination heard meanwhile set (`exit_at`) */
    __atomic_store_n(&port_count,port_count+1,__ATOMIC_SEQ_CST);
    quiet_until(&port->quiet_at,atomic_load(&exit_at));
    link_set(port,0,MESH_STOPPED,0);
  }
}
