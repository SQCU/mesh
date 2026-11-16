/* The bridge's events: the kernel's link and address events on a port's interface, the Thunderbolt RDMA peer
   interfaces' terminations and arrivals and the windows they hold, and the provider's doorbell fault absorbed in a
   port's child (design/page-queue-change-request.md: "Relays" and "Queue pairs and links belong to the bridge", the
   bridge alone hears its links come and go; "The bridge's process lifecycle", the failport's windows verbatim; "Failing
   by error, not by hardware fault", a provider's fault absorbed and named, never left to crash the node).  From tag
   pre-return-20261103 (90677543) rdma/mesh-link.c, verbatim but for the interfaces (mesh-bridge.h, mesh.h) and the notes
   at the spots that differ.  The tag's text of these is abd476a5's, restored whole by 3f72cbde, with 60b00774,
   abeffc1b and ea04d19f after it; each piece's first line names the commits that shaped it.  Nothing here is named to
   a client or the collective */
#include "mesh-bridge.h"


/* `faulted` and `exit_at` (tag rdma/mesh-link.c:26-30, static there: f090d50d 8bbc01a5 d62c879b a8952de9 0e34c13e
   ba674fbe) and `terminated` (1128-1132) are mesh-bridge.h's externs, defined in mesh-link.c */

/* tag rdma/mesh-link.c:37-42 windows_end (8bbc01a5 d62c879b; 3f72cbde), verbatim but extern (mesh-link.c reads it) */
/* the latest of the exit's window and every port's (B9, B9b): the stop's hold, and the devices' read (ports_find) */
uint64_t windows_end(void){
  uint64_t latest=atomic_load(&exit_at);
  for(uint32_t p=0;p<port_count;p++){const uint64_t quiet=atomic_load(&ports[p].quiet_at);latest=MAX(latest,quiet);}
  return latest;
}
/* tag rdma/mesh-link.c:1082-1107 network_heard (6b33cc33 ceff26ec 5b49fbb9 8bbc01a5 d62c879b 0e34c13e 128d0e84; 3f72cbde), verbatim but extern */
/* design/transport.md: a port's link off or its interface detached (the link lost, its lanes left as they
   are, design/transport.md B6 (b), and the port down until its link comes on: its device opened then,
   port_open); an interface of any name that attached or whose link came on and is no port yet: the node's devices to
   be read again (a link's coming on stays a reading, for a device registered after its interface attached), the read
   put off to the windows' end (find_wanted, kept while a read is out).  At the stop (`stopping`), a port's link going off only extends
   its quiet window, which the exit waits out (B9) */
void network_heard(int stopping){
  union mesh_network_event event;
  int unknown=0;
  while(recv(network,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    const uint32_t code=event.header.event_code;
    char name[IFNAMSIZ+16];
    snprintf(name,sizeof name,"%.*s%u",(int)sizeof interface->if_name,interface->if_name,interface->if_unit);
    int known=0;
    for(uint32_t p=0;p<port_count;p++)if(!strcmp(name,ports[p].name)){
      known=1;
      if(code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED){ports[p].opening=1;ports[p].retry_at=0;}
      else if(code!=KEV_DL_LINK_OFF && code!=KEV_DL_IF_DETACHED){}
      else if(stopping)quiet_until(&ports[p].quiet_at,now+MESH_PLATFORM_NS);
      else{ports[p].open=0;port_lost(ports+p,ENETDOWN);}
    }
    unknown|=!known && (code==KEV_DL_LINK_ON || code==KEV_DL_IF_ATTACHED);
  }
  if(unknown && !stopping)find_wanted=1;
}
/* tag rdma/mesh-link.c:1108-1126 addresses_heard (ba674fbe b505bae4; 3f72cbde; ea04d19f), verbatim but extern */
/* design/transport.md: an IPv6 address of a port's interface added, changed or deleted (the kernel's KEV_INET6_SUBCLASS
   events on `addresses`: a link-local address bin/mesh-fabric-init.sh puts in place, or its duplicate address detection
   done): the hellos' group joined on the interface again (a join while it held no IPv6 address took nothing) and the
   port's hello said at this turn, so a hello that left before the address could carry it is said again as the address
   comes, not a pace (MESH_PAUSE_NS) later.  The port is the event's interface family and unit, the port's own read once
   (struct port `family`, `unit`): no name is built or compared per event (design/rubrics.md, Strings are messages).
   Another Ethernet-family interface of the same unit (bridge0, awdl0, anpi<N>) is taken as the port's, and its event
   says the port's hello early, as the pace would: a hello on an open port with no peer is never wrong */
void addresses_heard(void){
  union { struct kern_event_msg header; unsigned char bytes[MESH_KERNEL_EVENT_BYTES]; } event;
  while(recv(addresses,event.bytes,sizeof event,0)>=(ssize_t)(KEV_MSG_HEADER_SIZE+sizeof(struct net_event_data))){
    const struct net_event_data *interface=(const void *)(event.bytes+KEV_MSG_HEADER_SIZE);
    for(uint32_t p=0;p<port_count;p++)if(interface->if_family==ports[p].family && interface->if_unit==ports[p].unit){
      struct ipv6_mreq group={.ipv6mr_multiaddr=everyone,.ipv6mr_interface=ports[p].interface};
      if(ports[p].interface)setsockopt(hello,IPPROTO_IPV6,IPV6_JOIN_GROUP,&group,sizeof group);
      ports[p].next=now;
    }
  }
}

/* tag rdma/mesh-link.c:1128-1138, the terminations' and arrivals' bits (8bbc01a5 d71970f8 a8952de9 ba674fbe; 3f72cbde; abeffc1b) */
/* design/transport.md B9b (EXIT-RECLAIM.md P2, and P3 as a window in place of IOServiceWaitQuiet): a
   Thunderbolt RDMA peer interface's termination, which IOKit tells in its first phase (kIOTerminatedNotification), on
   a queue of its own: the ports whose peer interface is terminating
   (`terminated`, a bit a port index), which the event loop concludes (terminations_turn) */
/* `terminated`: mesh-bridge.h's extern, defined in mesh-link.c (tag rdma/mesh-link.c:1132, static there) */
/* design/transport.md: a Thunderbolt RDMA peer interface's arrival (kIOFirstMatchNotification: a cable's far end come
   to RDMA), on the terminations' queue, so a port's termination and a later arrival are taken in their order: the ports
   whose peer interface arrived (`arrived`, a bit a port index) and whether one arrived whose interface is no port
   (`arrived_elsewhere`), which the event loop takes (arrivals_turn) */
static _Atomic uint32_t arrived;
static _Atomic int arrived_elsewhere;
/* tag rdma/mesh-link.c:1139-1145 quiet_all (d62c879b; 3f72cbde), verbatim */
/* the exit's window and every found port's held to `until` at least (a port found meanwhile takes `exit_at`'s as it is
   counted: ports_found) */
static void quiet_all(uint64_t until){
  quiet_until(&exit_at,until);
  const uint32_t count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);
  for(uint32_t p=0;p<count;p++)quiet_until(&ports[p].quiet_at,until);
}
/* tag rdma/mesh-link.c:1146-1167 peer_port (8bbc01a5; 3f72cbde), verbatim */
/* the port whose interface is the terminating peer interface's, by mesh-fabric-init.sh's rule (an
   AppleThunderboltRDMAInterface rdma_<port> and its peer interface carry the same "RID"); NULL where none is found
   (the property gone with the terminating service, or no port of that interface yet) */
static struct port *peer_port(io_service_t peer){
  CFTypeRef rid=IORegistryEntryCreateCFProperty(peer,CFSTR("RID"),kCFAllocatorDefault,0);
  io_iterator_t interfaces=IO_OBJECT_NULL;
  struct port *found=NULL;
  if(rid && IOServiceGetMatchingServices(kIOMainPortDefault,IOServiceMatching("AppleThunderboltRDMAInterface"),&interfaces)==KERN_SUCCESS){
    for(io_service_t interface;(interface=IOIteratorNext(interfaces));IOObjectRelease(interface)){
      CFTypeRef its=found?NULL:IORegistryEntryCreateCFProperty(interface,CFSTR("RID"),kCFAllocatorDefault,0);
      io_name_t name;
      if(its && CFEqual(its,rid) && IORegistryEntryGetName(interface,name)==KERN_SUCCESS){
        const uint32_t count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);
        for(uint32_t p=0;p<count;p++)if(!strcmp(ports[p].device.name,name))found=ports+p;
      }
      if(its)CFRelease(its);
    }
    IOObjectRelease(interfaces);
  }
  if(rid)CFRelease(rid);
  return found;
}
/* tag rdma/mesh-link.c:1168-1190 peer_terminated (8bbc01a5 d62c879b d71970f8; 3f72cbde; abeffc1b), verbatim */
/* the iterator drained first, which arms the notification again: for each terminated service the exit's window and
   every port's held to MESH_PLATFORM_NS from now (no port job and no exit inside the termination, whichever port's it
   is) and the event loop woken, before any property is read; then the port each maps to halted (port_halt: its
   pairing lost, served no more, marked for the event loop's conclusion); a service that maps to no port holds the
   windows alone; each released */
static void peer_terminated(void *unused,io_iterator_t iterator){
  (void)unused;
  io_service_t services[MESH_PORTS];
  uint32_t count=0;
  for(io_service_t service;(service=IOIteratorNext(iterator));){
    quiet_all(clock_ns()+MESH_PLATFORM_NS);
    wake();
    if(count<MESH_PORTS)services[count++]=service;
    else IOObjectRelease(service);
  }
  for(uint32_t k=0;k<count;k++){
    struct port *port=peer_port(services[k]);
    IOObjectRelease(services[k]);
    if(!port)continue;
    port_halt(port,&terminated);
    wake();
  }
}
/* tag rdma/mesh-link.c:1191-1201 peer_arrived (ba674fbe; 3f72cbde), verbatim */
/* the arrivals drained (which arms the notification again), each service released: the port its interface is marked
   to be opened, one whose interface is no port of this process marked for the devices' read */
static void peer_arrived(void *unused,io_iterator_t iterator){
  (void)unused;
  for(io_service_t service;(service=IOIteratorNext(iterator));IOObjectRelease(service)){
    struct port *port=peer_port(service);
    if(port)atomic_fetch_or(&arrived,UINT32_C(1)<<port->index);
    else atomic_store(&arrived_elsewhere,1);
    wake();
  }
}
/* tag rdma/mesh-link.c:1202-1221 terminations_watch (8bbc01a5 ba674fbe; 3f72cbde; abeffc1b), verbatim but extern, its two failures logged */
/* armed at the start, before IOKit's registry is waited on and the devices are read: the notification on a serial
   queue of its own, its iterator drained once (which arms it); without it a link's going off is the first sign.  The
   parent (`arrivals`) hears the peer interfaces' arrivals on the same port and queue, its iterator drained once too:
   without them a port whose open met its device's peer not yet present (ENETDOWN at both ends, no hello sent) opened
   only at its link's next coming on or the next hello heard on it, which might never come */
void terminations_watch(int arrivals){
  IONotificationPortRef notifications=IONotificationPortCreate(kIOMainPortDefault);
  io_iterator_t iterator=IO_OBJECT_NULL,came=IO_OBJECT_NULL;
  kern_return_t error=KERN_FAILURE,heard=KERN_FAILURE;
  if(notifications){
    IONotificationPortSetDispatchQueue(notifications,dispatch_queue_create("mesh.terminations",DISPATCH_QUEUE_SERIAL));
    error=IOServiceAddMatchingNotification(notifications,kIOTerminatedNotification,IOServiceMatching("AppleThunderboltRDMAPeerInterface"),
                                           peer_terminated,NULL,&iterator);
    if(arrivals)heard=IOServiceAddMatchingNotification(notifications,kIOFirstMatchNotification,
                                                      IOServiceMatching("AppleThunderboltRDMAPeerInterface"),peer_arrived,NULL,&came);
  }
  /* differs from the tag, which returned here without a word (abeffc1b deleted every log line of the bridge, and
     d556370e brought back the losses and failures it chose, not these): a watch not armed is logged, abd476a5's lines
     (8bbc01a5, ba674fbe) through mesh_say, as the change request asks of every failure.  What must stay: the bridge
     runs on without the watch, a link's going off its first sign */
  if(arrivals && heard)mesh_say("bridge","Thunderbolt RDMA peer interfaces' arrivals not heard (%#x): a port whose device was not yet "
                                "present at its link's coming on opens at the next hello heard on it",(unsigned)heard);
  if(arrivals && !heard)peer_arrived(NULL,came);
  if(error){mesh_say("bridge","Thunderbolt RDMA peer interfaces' terminations not heard (%#x): a link's going off is its first sign",
                     (unsigned)error);return;}
  peer_terminated(NULL,iterator);
}
/* tag rdma/mesh-link.c:1222-1250 fault_heard, `provider`, `retried` (8bbc01a5 2f4c511f b11621e6 ba674fbe; 3f72cbde; 60b00774), verbatim */
/* design/transport.md (EXIT-RECLAIM.md P4): the provider's doorbell fault absorbed.  A SIGSEGV or SIGBUS
   whose pc lies in the provider's text (`provider`, read at the start) on a thread that posts on a port's queue
   pairs (struct port `thread`) has one anonymous page mapped at the faulting page by hint, never MAP_FIXED, so nothing
   mapped is replaced: where it got that page (no mapping was there: the doorbell the driver unmapped in a termination),
   or the port's bit in `faulted` is set (its other thread got the page first; else the store faults again until the
   event loop takes the bit), the store lands in that page, no doorbell rings, and the port is halted for the event loop
   to lose (port_halt, terminations_turn: ENETDOWN, no verbs call).  Where the page could not be mapped and the bit is
   clear, the store is made once more (`retried`, the port's last such page: its other thread may have mapped it, its
   bit not yet set or already taken by the event loop), and only a second fault at that page is any other fault.  The
   bridge does not exit.  Any other fault: the default action restored, the store made again, the process
   ended.  On the thread's alternate stack where it has one (port_progress); mmap, munmap and sigaction alone */
static uintptr_t provider[2];
static _Atomic uintptr_t retried[MESH_PORTS];
static void fault_heard(int number,siginfo_t *info,void *context){
  const uintptr_t pc=__darwin_arm_thread_state64_get_pc(((ucontext_t *)context)->uc_mcontext->__ss);
  const uintptr_t page=trunc_page((uintptr_t)info->si_addr);
  const pthread_t self=pthread_self();
  struct port *port=NULL;
  for(uint32_t p=0,count=__atomic_load_n(&port_count,__ATOMIC_SEQ_CST);p<count;p++)
    for(int t=0;t<2;t++)if(atomic_load_explicit(&ports[p].thread[t],memory_order_acquire)==self)port=ports+p;
  if(port && pc>=provider[0] && pc<provider[1]){
    void *got=mmap((void *)page,vm_page_size,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
    if(got!=MAP_FAILED && got!=(void *)page)munmap(got,vm_page_size);
    if(got==(void *)page || atomic_load(&faulted)>>port->index&1){port_halt(port,&faulted);return;}
    if(atomic_exchange(&retried[port->index],page)!=page)return;
  }
  struct sigaction fallback={.sa_handler=SIG_DFL};
  sigaction(number,&fallback,NULL);
}
/* tag rdma/mesh-link.c:1251-1263 faults_install (2f4c511f 1f9be1d5; 3f72cbde), verbatim but extern, its failure logged */
/* the provider's text read and the faults' handler installed, by a child before it opens its device */
void faults_install(void){
  for(uint32_t i=0;i<_dyld_image_count();i++){
    const char *image=_dyld_get_image_name(i);
    unsigned long size=0;
    const uint8_t *text=image && strstr(image,"/libthunderboltrdma.")?
      getsegmentdata((const void *)_dyld_get_image_header(i),"__TEXT",&size):NULL;
    if(text){provider[0]=(uintptr_t)text;provider[1]=(uintptr_t)text+size;}
  }
  struct sigaction faults={.sa_sigaction=fault_heard,.sa_flags=SA_SIGINFO|SA_ONSTACK};
  sigemptyset(&faults.sa_mask);
  /* differs from the tag (die("bridge faults"), an unstamped line): the failure is the bridge log's line (mesh_say,
     naming errno), and the child ends by error as die ended it, exit(1), before it opens its device, so no queue pair is
     made where a doorbell fault could not be absorbed (the change request, "Failing by error, not by hardware fault").
     mesh-bridge.h's prototype returns nothing, so the child's end is the error here */
  if(sigaction(SIGSEGV,&faults,NULL) || sigaction(SIGBUS,&faults,NULL)){
    mesh_say("bridge","bridge faults: the provider's fault handler not installed (%s): this child exits before it opens its device",
             strerror(errno));
    exit(1);
  }
}
/* tag rdma/mesh-link.c:1264-1276 terminations_turn (8bbc01a5 2f4c511f a8952de9 0e34c13e; 3f72cbde; abeffc1b), verbatim but extern */
/* the terminations and the absorbed faults taken on the event loop: each port whose peer
   interface is terminating lost (port_lost, ETIMEDOUT: its socket closed, its stream's words ~0), its lanes left as they
   are, no verbs call; each port a fault was absorbed on lost as at its link's going off (open 0, ENETDOWN:
   network_heard) */
void terminations_turn(void){
  const uint32_t lost=atomic_exchange(&terminated,0),faults=atomic_exchange(&faulted,0);
  for(uint32_t p=0;p<port_count;p++)if(lost>>p&1){
    port_lost(ports+p,ETIMEDOUT);
  }
  for(uint32_t p=0;p<port_count;p++)if(faults>>p&1){
    ports[p].open=0;port_lost(ports+p,ENETDOWN);
  }
}
/* tag rdma/mesh-link.c:1277-1286 arrivals_turn (ba674fbe; 3f72cbde), verbatim but extern */
/* the arrivals taken on the event loop, after the terminations (a peer interface gone and then come is lost, then
   opened): each port whose peer interface arrived and that has no peer is to be opened as at its link's coming on
   (network_heard), so its device is opened at the hardware's word that its far end is present and not only at the en
   interface's link, which can come on before it (bin/mesh-fabric-init.sh: link active, no RDMA peer); an arrival at an
   interface that is no port has the devices read again (find_wanted) */
void arrivals_turn(void){
  const uint32_t came=atomic_exchange(&arrived,0);
  /* the coordinator, at integration: the tag's `ports[p].control<0` ("has no peer") is !port_paired, since the control
     socket now closes after the pairing's exchange (mesh-bridge.h port_paired); what must stay: one "has a peer" test */
  for(uint32_t p=0;p<port_count;p++)if(came>>p&1 && !port_paired(ports+p)){ports[p].opening=1;ports[p].retry_at=0;}
  if(atomic_exchange(&arrived_elsewhere,0))find_wanted=1;
}
