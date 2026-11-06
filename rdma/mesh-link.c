/* The node's resident bridge, its `main` (`mesh-flow CONFIGURATION`; a port's child, `mesh-flow --child ...`, is the
   same image, mesh-child.c): the shared state of the bridge's translation units (mesh-bridge.h), the configuration,
   the node's region opened or adopted, the event loop, and the stop.  From tag pre-return-20261103 rdma/mesh-link.c,
   verbatim but its interfaces, which are main's (design/page-queue-change-request.md, "The bridge's process
   lifecycle": the failport's lifecycle reused verbatim).  No -I, no --link: one configuration runs in a runtime, the
   node's id and its ports found by the bridge itself ("Replacing the --link form").  Every wait is bounded as at the
   tag (control_watch's park on `control`, as there, is no wait of the stop's: the stop's mesh_notify wakes it, and
   nothing joins it), and every failure is an exit with an error, never a fault ("Failing by error, not by hardware
   fault").  Left
   out with what they served: the streams, the client slots and the link pages and duds' block (link_reserve; the
   contract's "Link pages"); the children's end after the windows' hold is mesh-children.c children_end. */
/* tag pre-return-20261103 rdma/mesh-link.c:1-7, its includes: those of this file's pieces (spawn.h, sys/wait.h,
   crt_externs.h and mach/mach_time.h went with the children's code, mesh-children.c), with main's headers of the
   holder probes (mesh-holders.h: mesh_started, mesh_holds; libproc.h with them) and of the client library (mesh-dataflow.h:
   mesh_retire, mesh_retired_release, which take the region; struct mesh_ctx no longer, the bridge's own `bridge` gone:
   the shared state's note), which mesh-bridge.h does not include */
#include "mesh-bridge.h"
#include "mesh-holders.h"
#include "mesh-dataflow.h"
#include <sysexits.h>


/* tag pre-return-20261103 rdma/mesh-link.c:9-10 stop, verbatim (3f72cbde 0905738c 1835bef4 5dcb4f69 b71dcb36 7c500c88 f9f2aff0 ad55cbf4 33ca0dab ba674fbe e68bba3e 8e5df675 8df706e2 b346ef2b 7e0a797e 1f9be1d5 a8952de9 c24265e0 6b33cc33) */
/* mesh-verbs.h: the stop flag every loop reads */
_Atomic sig_atomic_t stop;
/* tag pre-return-20261103 rdma/mesh-link.c:12-18 the shared state, verbatim but `wire` main's type, `finding`, `found`
   and `finder` mesh-ports.c's (mesh-bridge.h), and `bridge` not here (3f72cbde 0905738c 1835bef4 5dcb4f69 b71dcb36 7c500c88 f9f2aff0 ad55cbf4 33ca0dab ba674fbe e68bba3e 8e5df675 8df706e2 b346ef2b 7e0a797e 1f9be1d5 a8952de9 c24265e0 6b33cc33) */
/* 2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 ("Sharing the region between programs") and A4
   (mesh-records.c's call of mesh_arena_alloc goes): the tag's `struct mesh_ctx bridge={.fd=-1};`, the bridge's own
   client context (main's mesh-dataflow.h struct mesh_ctx), is removed with its one use, mesh-records.c map_grow's
   lease (and main's `bridge.M=M;` below): the bridge allocates nothing from the arena, and mesh_retire and
   mesh_retired_release take the region.  What must stay: no context of the bridge's takes rows or pages */
struct hdr *M;
struct mesh_wire wire;
struct port ports[MESH_PORTS];
uint32_t port_count;
int events=-1,listener=-1,hello=-1,network=-1,addresses=-1,unreleased;
uint64_t now,sequence;
/* tag pre-return-20261103 rdma/mesh-link.c:26-27 faulted, verbatim but not static (mesh-bridge.h) (3f72cbde 0905738c cf483b3e 7c500c88 f9f2aff0 b53804bf ad55cbf4 33ca0dab ba674fbe 128d0e84 0e34c13e a8952de9 d71970f8 d62c879b 8bbc01a5 f090d50d 5b49fbb9) */
/* the ports a provider's doorbell fault was absorbed on (fault_heard), a bit a port */
_Atomic uint32_t faulted;
/* tag pre-return-20261103 rdma/mesh-link.c:28-30 exit_at, verbatim but not static (mesh-bridge.h) (8bbc01a5 f090d50d) */
/* design/transport.md B9b: the window of a peer interface's termination (`exit_at`), as every port's quiet
   window holds, before which the bridge does not exit nor read the devices */
_Atomic uint64_t exit_at;
/* tag pre-return-20261103 rdma/mesh-link.c:49-51 memory, stop_reason, known, verbatim but `stop_reason` not static (mesh-bridge.h) (3f72cbde) */
uint64_t memory;
int64_t stop_reason;
struct known known[MESH_KNOWN];
/* tag pre-return-20261103 rdma/mesh-link.c:530-533 owners and model; `owners` one entry, main's one client word (hdr
   `client`), where the tag had MESH_CLIENTS slots, which main has not (3f72cbde 0905738c cf483b3e b71dcb36 33ca0dab 77349292 6f90dc97 a8952de9 6b33cc33) */
/* 2026-11-04, owner B2 (design/plan-to-bridge-change-request.md R8; design/compilation-coordination.md, L3 and Q5):
   `uint64_t owners[1];`, the client word as clients_turn last read it, is removed: no client word is written from 110,
   and clients_turn walks the claims' heads, keeping its last read of hdr `plans` itself (mesh-records.c `plans_seen`).
   What must stay: `model` */
/* design/transport.md: this node's record, its region, memory and model with its ports and health */
char model[MESH_MODEL_BYTES];
/* tag pre-return-20261103 rdma/mesh-link.c:1027-1030 ones, verbatim, and `held_pages` not here.  No duds' block is taken
   (link_reserve is left out: the contract's "Link pages"), so `ones` stays MESH_ABSENT, as each child is told (3f72cbde 33ca0dab 2fb2b270 6f90dc97 923d73fd a8952de9 ceff26ec 6b33cc33) */
/* design/transport.md section 10: the all-ones block every dud is sent from (stream_send), taken and filled at the
   bridge's start, never written again */
uint32_t ones=MESH_ABSENT;
/* 2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 ("Sharing the region between programs": "The
   bridge's own pages (its records) are laid out once with the region"): the tag's `uint64_t held_pages;`, the arena
   pages the bridge held itself, is removed.  Its last holding was the map's lease (mesh-records.c map_grow's note),
   and with the map the layout's area, no link pages and no duds' block, the bridge holds no arena page.  What must
   stay: a record's `region` the arena's bytes whole (mesh-records.c record_changed) */
/* tag pre-return-20261103 rdma/mesh-link.c:1128-1132 terminated, verbatim but not static (mesh-bridge.h) (abeffc1b 3f72cbde b53804bf ad55cbf4 33ca0dab ba674fbe 736ce24d a8952de9 d71970f8 d62c879b 8bbc01a5 6b33cc33) */
/* design/transport.md B9b (EXIT-RECLAIM.md P2, and P3 as a window in place of IOServiceWaitQuiet): a
   Thunderbolt RDMA peer interface's termination, which IOKit tells in its first phase (kIOTerminatedNotification), on
   a queue of its own: the ports whose peer interface is terminating
   (`terminated`, a bit a port index), which the event loop concludes (terminations_turn) */
_Atomic uint32_t terminated;
/* tag pre-return-20261103 rdma/mesh-link.c:1295 self_path, verbatim but not static (mesh-bridge.h); its comment, on
   the child's spawn, is mesh-children.c's (3f72cbde 33ca0dab ce897b81 8e5df675 1f9be1d5) */
char self_path[PATH_MAX];

/* tag pre-return-20261103 rdma/mesh-link.c:52-55 wake, verbatim (3f72cbde) */
void wake(void){
  struct kevent64_s event;EV_SET64(&event,0,EVFILT_USER,0,NOTE_TRIGGER,0,0,0,0);
  kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL);
}

/* tag pre-return-20261103 rdma/mesh-link.c:1498-1507 control_watch, verbatim (90179f5e 111d42f7 3f72cbde 0905738c cf483b3e 1835bef4 7c500c88 ad55cbf4 33ca0dab 1f9be1d5 a8952de9 6b33cc33) */
/* design/transport.md: the clients' control word, watched for the event loop */
/* 2026-11-04, owner B2 (design/compilation-coordination.md, "The region": hdr `plans`, "a count that only rises: every
   publication of a plan, every close of a claim and every start adds 1, release, then wakes it"; Q2 and Q4: the
   parent's turn submits a plan's executions once its start raises them, and delivers an ORDER once the ports have
   served the plan released before), reversing the tag's watch of hdr `control`: the watcher parks on `plans`, each
   move of which, and each wake of it, wakes the event loop for a turn, so a start's raise, a publication and a close
   (mesh_retire's among them) are taken at once, not at the loop's next due time.  The stop's notification of `control`
   (main) no longer wakes it; nothing joins it, as before, and the process's exit ends it.  What must stay: one
   watcher, a wait that a move of the word or a wake ends.
   2026-11-04, owner B2, the coordinator's ruling on reviews V1's and V3's finding 11 ("every wait ends in bounded
   time"): the wait takes the timed form, bounded by MESH_PAUSE_NS, as each port's progress thread waits on `plans`
   (mesh-flow.c), so the watcher reads the stop within MESH_PAUSE_NS of it.  What must stay: the word read before the
   loop's wake is the one waited on.
   2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL2 and DL3;
   design/aot-compilation-change-request.md, appendix item 2, "Delivery at a node"), reversing the first note's "delivers
   an ORDER once the ports have served the plan released before" (Q4's hold, gone): the turn publishes each of the
   head's executions as L. Lamport's Message Buffer sender may (Concurrent reading and writing, CACM 20(11), 1977:
   "wait until ms < mr + P"; mesh-records.c orders_deliver), and a port wakes `plans` (no raise) where its `taken[p]`,
   the sender's mr, rose (mesh-flow.c taken_tell), so this watcher wakes the loop for the turn that publishes on.  No
   behavior changed.  What must stay: as above.
   2026-11-05, owner B4 (the coordinator's ruling DL9): the port raises `plans` there (mesh.h mesh_notify), as the
   parent does after a turn that published, where each woke it without a raise, so a hand-off that lands between this
   watcher's read and its wait ends the wait at once.  No behavior changed here.  What must stay: as above */
static void *control_watch(void *unused){
  (void)unused;
  while(!stop){
    uint64_t seen=atomic_load_explicit(&M->plans,memory_order_acquire);
    wake();
    os_sync_wait_on_address_with_timeout(&M->plans,seen,sizeof M->plans,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                         OS_CLOCK_MACH_ABSOLUTE_TIME,MESH_PAUSE_NS);
  }
  return NULL;
}

/* tag pre-return-20261103 rdma/mesh-link.c:1509-1534 configuration_read, verbatim but main's geometry read beside the tag's keys (ea04d19f 3f72cbde 1ffbd907 128d0e84 736ce24d 6f90dc97 a8952de9 5a5dec1b 6b33cc33) */
/* design/transport.md: the bridge's configuration, the canonical property list bin/mesh-user.sh writes from
   the consumer's document (rdma/bridge_configuration.py): its name and its region in bytes (`region_bytes`); NULL, or
   why it is refused */
/* Differs from the tag: the writer is bin/mesh-bridge.sh (sub-assignment 1), and the list carries main's geometry
   besides the tag's two keys (the coordinator's contract, "The configuration"), each an integer: `block_pages`
   (struct mesh_geometry `block`), `rows`, `window_pages` (`wire_pages`) and `hops` (the page lists' room), into
   *geometry.  An absent one is 0, mesh_layout's default (main's mesh.h: rows the arena's, the window the whole arena,
   no hops; `block`, which mesh_layout divides by, is given the tag's 1 by main below).  A present one is a positive
   number of at most INT32_MAX, main's bound on its page counts (main's mesh-flow.c "configured page count"), else the
   list is refused, as the tag refuses a region_bytes that is not a positive number.  No node index and no link is
   configured: the bridge finds its ports and its node id itself. */
static const char *configuration_read(const char *path,char *name,size_t room,uint64_t *region_bytes,struct mesh_geometry *geometry){
  FILE *file=fopen(path,"rb");
  if(!file)return strerror(errno);
  unsigned char bytes[MESH_CONFIGURATION_BYTES];
  const size_t length=fread(bytes,1,sizeof bytes,file);
  const int whole=feof(file);
  fclose(file);
  if(!whole)return "not a configuration property list (too long)";
  CFDataRef data=CFDataCreate(NULL,bytes,(CFIndex)length);
  CFPropertyListRef list=data?CFPropertyListCreateWithData(NULL,data,kCFPropertyListImmutable,NULL,NULL):NULL;
  if(data)CFRelease(data);
  const char *refused=NULL;
  CFTypeRef named=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("name")):NULL;
  CFTypeRef sized=list && CFGetTypeID(list)==CFDictionaryGetTypeID()?CFDictionaryGetValue(list,CFSTR("region_bytes")):NULL;
  int64_t value=0;
  if(!list || CFGetTypeID(list)!=CFDictionaryGetTypeID())refused="not a property list of a configuration (bin/mesh-bridge.sh writes it)";
  else if(!named || CFGetTypeID(named)!=CFStringGetTypeID() || !CFStringGetCString(named,name,(CFIndex)room,kCFStringEncodingUTF8))refused="its name is missing";
  else if(!sized || CFGetTypeID(sized)!=CFNumberGetTypeID() || !CFNumberGetValue(sized,kCFNumberSInt64Type,&value) || value<=0)
    refused="its region_bytes, a positive number of bytes, is missing";
  /* 2026-11-04, owner B2 (mesh-map.h mesh_geometry, 110: the page lists' area and its room `hops` gone, each plan's
     entries in its own state section; design/compilation-coordination.md, "The region"): the key `hops` is read no
     more, and a list that holds it is taken as one that does not, as any other key this list does not name.  What must
     stay: the three keys of the layout, each a positive number of at most INT32_MAX */
  const CFStringRef keys[]={CFSTR("block_pages"),CFSTR("rows"),CFSTR("window_pages")};
  uint32_t *const shape[]={&geometry->block,&geometry->rows,&geometry->wire_pages};
  static const char *const wrong[]={"its block_pages is not a positive number of pages of at most INT32_MAX",
    "its rows is not a positive number of rows of at most INT32_MAX","its window_pages is not a positive number of pages of at most INT32_MAX"};
  for(size_t k=0;!refused && k<sizeof keys/sizeof *keys;k++){
    CFTypeRef given=CFDictionaryGetValue(list,keys[k]);
    int64_t count=0;
    if(given && (CFGetTypeID(given)!=CFNumberGetTypeID() || !CFNumberGetValue(given,kCFNumberSInt64Type,&count) || count<=0 || count>INT32_MAX))
      refused=wrong[k];
    *shape[k]=(uint32_t)count;
  }
  if(list)CFRelease(list);
  *region_bytes=(uint64_t)value;
  return refused;
}

/* tag pre-return-20261103 rdma/mesh-link.c:1742-1763 region_same (its comment region_open's), verbatim but main's layout fields (07949b28 378dbd5f 3f72cbde 33ca0dab ce897b81 8e5df675 736ce24d 6f90dc97 b346ef2b 923d73fd) */
/* design/transport.md section 2, the region and the heartbeat: the node's region (MESH_NAME) opened.  The region a
   bridge before this one kept (a restart, SIGHUP; a crash) is adopted where its magic, layout version (MESH_LAYOUT) and
   layout (`geometry`, mesh_layout of this configuration's region) are this bridge's: the same memory, its clients'
   slots, rows and leases standing and the GPU buffers over them valid, its pages registered again by the port children
   (the same pages: nothing is wired twice, nothing reloaded).  A region a live bridge serves is never read: a second
   bridge of the node ends at its port's bind, before region_open.  The node has one region object, made once, when none
   is named, at the configured size; every later layout is fitted to that object's size (mesh.h mesh_layout_within: the
   arena what its metadata leaves) and takes its pages.  A region of another layout that is not yet expired is made
   expired (its layout version 0: no client's mesh_open takes it, and each client's next lease or communicator returns
   the expiry, nccl-mesh.c attach_expired), and this bridge ends with EX_TEMPFAIL: its heartbeat opens a bridge again once
   none of the region's holders it can see lives.  An expired region (or an object that never became one) is made
   again in place: its metadata zeroed and laid as this layout's, over the same pages.  So a restart never makes a
   second region beside the first, whatever holds the first (a process in its exit keeps the same pages wired, never
   others besides).  Only a release unlinks the object (mesh-heartbeat.c, SIGUSR1 or
   SIGTERM).  A region that cannot be read is left as it is, and this bridge ends.  1
   adopted, 0 made; *out its descriptor; *geometry fitted to the object's size */
/* Differs from the tag, its interfaces: the layout version is main's MESH_VERSION (main has one version, its layout's,
   which every client's mesh_attach compares), where the tag had MESH_LAYOUT; the layout is main's mesh_layout of the
   configuration's geometry (`layout`, region_open's operand), fitted to an existing object as at the tag by
   mesh_layout_within below.  Two layouts are the same where every field main's mesh_layout sets is (the tag's rule:
   its fields were those its mesh_layout set) */
/* 2026-11-03, design/disjoint-pages-coordination.md, "The layout": mesh_layout sets the supplied words' area
   (`word_off`, `word_capacity`), the landed list's (`landed_off`, `landed_capacity`) and the map's records' (`map_off`,
   now the layout's, and `map_room`), so each is compared, by the rule above; `landed_stride` is the plan's, not the
   layout's, and is not.  What must stay: every field mesh_layout sets compared, so an adopted region's areas lie where
   this bridge's layout puts them (mesh-records.c map_adopt takes the map's area as it lies) */
/* 2026-11-04, sub-assignment 7 (design/disjoint-pages-change-request.md, C3.2 and C4: "the supplied words' and landed
   list's fixed areas of the layout (hdr `word_off`, `landed_off`)" go; design/disjoint-pages-revision-coordination.md,
   "The region" and "The layout"), reversing the 2026-11-03 note above: from layout 109 mesh_layout sets no supplied
   words' or landed list's area, and hdr `word_off`, `word_capacity`, `landed_off`, `landed_capacity` and
   `landed_stride` are `vacant9` .. `vacant13`, never read or written, so none of them is compared.  The plan's state
   section (hdr `state`, struct mesh_state: its phases, counts and supplied words) is placed by mesh_plan in the arena
   with the plan's other values, not by the layout, and is not compared either.  What must stay: exactly the fields
   mesh_layout sets compared (the map's records' `map_off` and `map_room` among them), so an adopted region's areas lie
   where this bridge's layout puts them */
/* 2026-11-04, review R2 (L1), reversing this file's first form (d77eec2a), which refused an object smaller than the
   configuration's length: a configuration that grew the region then left the bridge exiting 1 and its heartbeat
   opening it again, forever.  What must stay: the layout fitted to the object, a bridge ending only where none fits */
/* mesh_layout_within (be0e1e48) is mesh-map.h's, beside mesh_layout (review R2, B2) */
/* 2026-11-04, owner B2 (mesh-map.h mesh_layout, 110: the page lists' area gone; mesh.h: hdr `hop_off` and
   `hop_capacity` vacant, `vacant16` and `vacant17`, never read or written): those two are not compared, since
   mesh_layout sets them no more.  What must stay: exactly the fields mesh_layout sets compared (the 2026-11-04 note
   above).
   2026-11-05 (113; design/compilation-coordination.md, "The revisions of 46ae8fdf", CH): `deliveries_off` and
   `deliveries_room` compared, the deliveries' channel's area mesh_layout now sets.  What must stay: as above */
static int region_same(const struct hdr *a,const struct hdr *b){
  return a->pgsz==b->pgsz && a->block==b->block && a->rows==b->rows && a->wire_pages==b->wire_pages && a->links==b->links &&
         a->planes_off==b->planes_off && a->arena_off==b->arena_off && a->page_off==b->page_off && a->buffer_off==b->buffer_off &&
         a->link_off==b->link_off && a->target_stride==b->target_stride && a->target_off==b->target_off &&
         a->bridge_off==b->bridge_off &&
         atomic_load_explicit(&a->map_off,memory_order_relaxed)==atomic_load_explicit(&b->map_off,memory_order_relaxed) &&
         a->map_room==b->map_room && a->deliveries_off==b->deliveries_off && a->deliveries_room==b->deliveries_room &&
         a->data_off==b->data_off && a->length==b->length;
}
/* tag pre-return-20261103 rdma/mesh-link.c:1764 region_children, verbatim (07949b28) */
static struct mesh_holder region_children[2*MESH_PORTS];
/* tag pre-return-20261103 rdma/mesh-link.c:1765-1771 region_running, verbatim (07949b28) */
/* whether process `pid` of start `start` still runs: its exit not begun (argument 0 of PROC_PIDTBSDINFO reads no process
   in its exit, mesh-dataflow.h mesh_started) */
static int region_running(pid_t pid,uint64_t start){
  struct proc_bsdinfo info;
  return pid && proc_pidinfo(pid,PROC_PIDTBSDINFO,0,&info,(int)sizeof info)==(int)sizeof info &&
         (!start || (uint64_t)info.pbi_start_tvsec*UINT64_C(1000000)+info.pbi_start_tvusec==start);
}
/* tag pre-return-20261103 rdma/mesh-link.c:1772-1795 region_orphans_end, verbatim (07949b28) */
/* an earlier launch's children (region_children) that still run, before a layout is laid over their region: each
   told to end as a signal ends a process (SIGTERM: it closes, its lanes flushed, child_leave) and awaited its bounds
   (2 MESH_PLATFORM_NS and MESH_PAUSE_NS), then ended (SIGKILL: a process it is pending for runs none of its own code
   again, even one in a kernel call now), so none writes its words into the new layout; one in its exit runs nothing of
   its own, and its device is the next child's to wait for (child_elsewhere).  A bridge that ended as asked ended after
   its children; these are a bridge's that did not */
static void region_orphans_end(void){
  uint32_t running=0;
  for(uint32_t e=0;e<2*MESH_PORTS;e++)
    if(region_running((pid_t)region_children[e].pid,region_children[e].start)){running++;kill((pid_t)region_children[e].pid,SIGTERM);}
  if(running)mesh_say("bridge","%u children of an earlier launch still run: told to end (SIGTERM) before a layout is laid over "
                      "their region",running);
  for(const uint64_t asked=clock_ns()+2*MESH_PLATFORM_NS+MESH_PAUSE_NS;running;poll(NULL,0,100)){
    running=0;
    for(uint32_t e=0;e<2*MESH_PORTS;e++)running+=region_running((pid_t)region_children[e].pid,region_children[e].start);
    if(running && clock_ns()>=asked){
      for(uint32_t e=0;e<2*MESH_PORTS;e++)
        if(region_running((pid_t)region_children[e].pid,region_children[e].start))kill((pid_t)region_children[e].pid,SIGKILL);
      mesh_say("bridge","%u children of an earlier launch ran on %.0f s after SIGTERM: ended (SIGKILL)",running,
               (double)(2*MESH_PLATFORM_NS+MESH_PAUSE_NS)/1e9);
      break;
    }
  }
}
/* tag pre-return-20261103 rdma/mesh-link.c:1796-1833 region_open, verbatim but main's layout (region_same's note: the configuration's geometry `layout` an operand, for the fit) and the bridge area's bounds (07949b28 378dbd5f 11807bec abeffc1b 306b22be 3f72cbde 0905738c 1835bef4 1ffbd907 ad55cbf4 33ca0dab d70e843a f06759d4 8e5df675 6f90dc97 b346ef2b 923d73fd) */
static int region_open(struct hdr *geometry,struct mesh_geometry layout,int *out){
  int fd=shm_open(MESH_NAME,O_RDWR,0);
  if(fd<0 && errno!=ENOENT){exit(1);}
  struct stat shape={0};
  if(fd>=0 && fstat(fd,&shape)){exit(1);}
  if(fd>=0 && !shape.st_size){close(fd);fd=-1;shm_unlink(MESH_NAME);}
  if(fd>=0){
    const size_t size=(size_t)shape.st_size;
    if(!mesh_layout_within(geometry,layout,size))die("the region object holds no layout");
    struct hdr *old=mmap(NULL,size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(old==MAP_FAILED){exit(1);}
    if(old->magic==MESH_MAGIC){
      const uint32_t version=__atomic_load_n(&old->version,__ATOMIC_ACQUIRE);
      if(version==MESH_VERSION && region_same(old,geometry)){M=old;*out=fd;return 1;}
      if(version){__atomic_store_n(&old->version,0,__ATOMIC_SEQ_CST);exit(EX_TEMPFAIL);}
    }
    /* the children an earlier launch recorded kept across the new layout, those that still hold (child_elsewhere) */
    /* Differs from the tag: the bridge area is sizeof(struct mesh_bridge_pages), main's (the tag's MESH_BRIDGE_BYTES),
       and it is read only where `bridge_off` lies past the header.  At the tag every layout had a bridge area at a
       header field of the same place; main's layouts before 105 have none, and an object one of them left (main's
       --link bridge's, ended without its unlink) reads 0 there, which would take the header's own words (a client's
       pid among them) for children and signal them.  Such an object has no holders of this form to end */
    if(old->magic==MESH_MAGIC && old->bridge_off>=sizeof *old && old->bridge_off+sizeof(struct mesh_bridge_pages)<=size)
      memcpy(region_children,mesh_bridge_pages(old)->child,sizeof region_children);
    region_orphans_end();
    /* 2026-11-04, sub-assignment 7: this is C3.7's one zeroing left (design/disjoint-pages-change-request.md, C3.7:
       "The only zeroing left is mesh-link.c's zeroing of the region's header, once, when the bridge creates the
       region"): the metadata before `data_off` (the header and the layout's areas) zeroed once, as the region is made
       again in place; a new object (below) is zero as ftruncate makes it.  Nothing from `data_off` on (the arena: data
       pages and plans' state sections) is zeroed or reset by the bridge: a data page is written only by its landing, a
       state word only as C3.7 gives it (one writer; counts take arrivals), never reset while its plan stands.  What
       must stay: this zeroing once a making, and none past `data_off` */
    M=old;memset((void *)M,0,(size_t)geometry->data_off);
  }else{
    if((fd=shm_open(MESH_NAME,O_CREAT|O_EXCL|O_RDWR,MESH_MODE))<0)die("shm");
    if(ftruncate(fd,(off_t)geometry->length))die("ftruncate");
    fchmod(fd,MESH_MODE);
    M=mmap(NULL,(size_t)geometry->length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(M==MAP_FAILED)die("mmap");
  }
  *M=*geometry;M->version=MESH_VERSION;
  for(uint32_t r=0;r<mesh_rows(M);r++)atomic_store_explicit(&mesh_page(M)[r].mapping,MESH_ABSENT,memory_order_relaxed);
  struct mesh_bridge_pages *own=mesh_bridge_pages(M);
  for(uint32_t p=0;p<MESH_PORTS;p++)own->link[p]=MESH_ABSENT;
  /* 2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 ("Sharing the region between programs"): the
     tag's line, kept, but `map` and `map_pages` are now only recorded vacant (MESH_ABSENT, 0) and never written again:
     no map lease exists, the map's records being the layout's area (hdr `map_off`, `map_room`; mesh-records.c
     map_grow's note), and the fields stay since struct mesh_bridge_pages is only appended to (mesh.h).  What must
     stay: no reader takes the map's place from them */
  own->ones=own->map=MESH_ABSENT;own->map_pages=0;
  for(uint32_t e=0;e<2*MESH_PORTS;e++)
    if(region_children[e].pid && mesh_holds((pid_t)region_children[e].pid,region_children[e].start))own->child[e]=region_children[e];
  *out=fd;
  return 0;
}

/* the adopted region's bridge state made this bridge's, before any client reads it as live (`bridge_pid` set last): every
   stream not live and held by none, its landed words ~0 (a lost pairing's holding ended with its bridge, and a
   wait on it reads the loss; the next pairing opens each new, streams_open), the links' table cleared, a client slot
   whose retirement a process that died left half done finished (mesh_retire's walk, which any party can repeat to the
   same end), the closed rows released, a map or health serial a bridge that died mid-write left odd made even (its
   readers wait while it is odd), and the map's records this bridge's to write again (map_adopt) */
/* tag pre-return-20261103 rdma/mesh-link.c:1835-1868 region_adopt, verbatim but main's region (4bb01da0 f440f786 a1c70649 7876150c 11c52f5e 3f72cbde 0905738c 1835bef4 d9e6d3de 5dcb4f69 ed429e7c 76b85f5e f9f2aff0 ad55cbf4 33ca0dab 292d4516 88fa2452 db2748a9 2fb2b270 8fc0d3d7 f06759d4 6f90dc97 923d73fd) */
/* Differs from the tag, its interfaces: main's region has no streams; in place of the tag's loop storing each one's
   landed words lost, this node's receives of the plan written that have not landed are cancelled (mesh-flow.c
   streams_cancel(NULL), the stop's own cancellation), so a wait on a dead bridge's plan reads the loss, as at the
   tag, rather than its deadline; and no client slots, but one client word (hdr `client`, its process the low 32 bits, its start hdr `client_start`: mesh-holders.h), which
   is retired where its process does not hold the region (mesh_holds) by main's own retirement (mesh-dataflow.c
   mesh_retire, as mesh_attach retires a vacant word whose process is gone), where the tag finished a slot's
   retirement half done; and main's link table (struct mesh_link_info) has `peer` and `bandwidth` plain (stored
   atomically as the tag's, by __atomic_store_n), `phase` and `code` in `port`, and `port.prepared` (the plan a port's
   data path loaded, which no new child has) cleared beside them */
/* 2026-11-04, sub-assignment 7 (design/disjoint-pages-change-request.md, C3.1: "Landings and cancellations both arrive
   on the count", and C3.6; design/disjoint-pages-revision-coordination.md, "The bridge"), reversing the 2026-11-03
   form of the note above, in which a cancellation stored UINT64_MAX into each not-landed receive's execution's cancel
   word in the landed list (mesh.h's landed records, gone at 109 with the cancel word and its other writers): a
   cancellation now arrives on the receive's count word, raised (never lowered) to its final value, and no phase is
   written, so a reader of an execution it cancelled finds the receive's field one phase behind with its count complete:
   cancelled, an error value, not a wait to its deadline.  What must stay: the cancellation before `bridge_pid` is set,
   and before the release of retired claims below, on this one thread (so its stores land in pages still held) */
static void region_adopt(void){
  /* 2026-11-04, review R2 (L2), reversing this file's first form (d77eec2a), which left the plan's words as a dead
     bridge left them: a client waiting on one reached its deadline instead of the loss.  No port is found yet
     (port_count 0), so this is the plan's receives alone.  What must stay: the adopted region's receives not landed
     read cancelled (their count words at their final values, their phases unwritten) before `bridge_pid` is set */
  streams_cancel(NULL);
  struct mesh_link_info *links=mesh_links(M);
  for(uint32_t p=0;p<MESH_PORTS;p++){
    __atomic_store_n(&links[p].peer,0,__ATOMIC_SEQ_CST);atomic_store(&links[p].port.phase,MESH_UNKNOWN);atomic_store(&links[p].port.code,0);
    __atomic_store_n(&links[p].bandwidth,0,__ATOMIC_SEQ_CST);atomic_store(&links[p].port.prepared,0);
  }
  /* 2026-11-04, owner B2 (design/plan-to-bridge-change-request.md R8; e6bd7db2's section 6, "Holders and clients", as
     design/compilation-coordination.md ruling 1 and Q5 take it: "region_open retires each gone owner"), reversing the
     one-client form above (hdr `client` and `client_start`, which nothing writes from 110, L3): each open claim (its
     head's owner not 0, `closed` 0) whose owner's process does not hold the region (mesh-holders.h mesh_holds, by its
     pid and the head's `start`) has that owner retired (mesh-dataflow.h mesh_retire: its claims closed).  Here, where
     the adopted region's state is made this bridge's, after region_open and before the release, as the one client's
     retirement was.  What must stay: every gone owner retired before the release below.
     2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5, and "The
     revisions of 4e78f6f4"), reversing the walk of the claims' heads: the program's root (mesh.h struct mesh_program,
     at the layout's one place), where it is open (`owner` not 0, `closed` 0, each acquire) and its owner's process
     does not hold the region (mesh_holds, by its pid and the root's `start`), has that owner retired (mesh-dataflow.h
     mesh_retire: the root closed); the release below is the root's.  Why: point 5, "These go: `mesh_claim_take`'s
     first fit and the claim's head record"; "Release at retirement stays deferred reclamation (RCU)".  What must
     stay: the gone owner retired before the release below */
  struct mesh_program *program=mesh_program(M);
  const uint64_t owner=atomic_load_explicit(&program->owner,memory_order_acquire);
  if(owner && !atomic_load_explicit(&program->closed,memory_order_acquire) && !mesh_holds((pid_t)(uint32_t)owner,program->start))
    mesh_retire(M,owner);
  /* 2026-11-04, sub-assignment 7 (design/disjoint-pages-change-request.md, C3.4; the contract, "The program's claim and
     its sections": mesh_retired_release releases each closed claim whose generation no port's `prepared` holds, after
     a seq_cst fence): every port's `prepared` is 0 above, so every closed claim is released here, after
     streams_cancel(NULL)'s stores, on the same thread (main, before the event loop: region_open's caller).
     2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 5): the
     program's root released here where it is closed, every port's `prepared` 0 above and every port's `held` bit of
     its plans cleared by streams_cancel(NULL) (no port is found yet: mesh-flow.c held_clear), where every closed claim
     was.  What must stay: the release after streams_cancel(NULL)'s stores, on this thread */
  mesh_retired_release(M);
  if(atomic_load(&M->map)&1)atomic_fetch_add(&M->map,1);
  if(atomic_load(&M->health)&1)atomic_fetch_add(&M->health,1);
  map_adopt();
  atomic_store(&M->busy,0);atomic_store(&M->stopping,0);
}

/* tag pre-return-20261103 rdma/mesh-link.c:1870-1892 due_at and turn_due, verbatim (90179f5e 306b22be 111d42f7 3f72cbde 7c500c88 4dc9e98a 02002590 ad55cbf4 33ca0dab ba674fbe 923d73fd) */
/* design/transport.md: when the event loop's next turn is due if no event comes first: the earliest of the times its turn
   waits for that are still ahead (the next liveness turn, `liveness`; the next health sample;
   where the devices are to be read, the windows' end; the exit's window; each port's quiet window, spawn hold, open's
   retry, next hello, and the bound of its open's or link's job out), and no later than MESH_PAUSE_NS on (design/rubrics.md M3: the wait ends in finite
   time).  Every other change a turn takes wakes the loop where it happens (wake, `control`, a socket, a signal and an
   exit on the queue), so a turn comes at the change or at the time it waits for, where a fixed tick (250 ms) took each
   such time up to a tick late */
static inline void due_at(uint64_t *due,uint64_t at){ if(at>now && at<*due)*due=at; }
static uint64_t turn_due(uint64_t liveness){
  uint64_t due=MIN(liveness,now+MESH_PAUSE_NS);
  due_at(&due,health_due());
  /* 2026-11-04, owner B3 (design/compilation-coordination.md, "The appendix's revisions", O1 and H1), reversing owner
     B2's line here (the coordinator's ruling Q4'), `due_at(&due,orders_due());`, the pending ORDER's bound, at which a
     head whose plan did not stand here was delivered with nothing released: the bound is gone (mesh-records.c
     orders_deliver: a head stays at the head, Defago et al., Fig. 5, no timeout and no skip;
     design/aot-compilation-change-request.md, appendix items 2 and 3).  What must stay: the tag's times, verbatim.
     2026-11-04, owner B4 (design/compilation-coordination.md, "The documents' revisions of edd4b583", DL2): a head
     also stays where Lamport's sender waits for its buffer (some port's `taken[p]` below t - u + 1), and that wait has
     no due time either: the port's wake of `plans` where its `taken[p]` rose wakes the loop (control_watch), and the
     turn's MESH_PAUSE_NS bound above stands.  No behavior changed.  What must stay: as above */
  if(find_wanted)due_at(&due,windows_end());
  due_at(&due,atomic_load(&exit_at));
  for(uint32_t p=0;p<port_count;p++){
    const struct port *port=ports+p;
    due_at(&due,atomic_load(&port->quiet_at));due_at(&due,port->hold_to);
    if(port->opening)due_at(&due,port->retry_at);
    /* the coordinator, at integration: the tag's `port->control<0` is !port_paired (mesh-bridge.h; the socket closes
       after the pairing's exchange) */
    if(port->open && !port_paired(port))due_at(&due,port->next);
    if(port->job)due_at(&due,port->job_at+MESH_PLATFORM_NS);
    if(port->link.job)due_at(&due,port->link.job_at+MESH_PLATFORM_NS);
  }
  return due;
}

/* tag pre-return-20261103 rdma/mesh-link.c:1894-2078 main to the windows' hold, verbatim but its interfaces (noted at each); the tag's tail after the hold (2079-2121) is mesh-children.c children_end (d556370e ea04d19f 5eabe91e abeffc1b 90179f5e 306b22be 111d42f7 f440f786 a1c70649 7876150c 11c52f5e 3f72cbde 0905738c 1835bef4 5dcb4f69 7c500c88 4dc9e98a f9f2aff0 ad55cbf4 33ca0dab 2fb2b270 ba674fbe e68bba3e 128d0e84 ce897b81 8e5df675 736ce24d 6f90dc97 b346ef2b 923d73fd 1f9be1d5 a9d0f556 0e34c13e a8952de9 388ecd87 5a5dec1b 2f4c511f d62c879b 8bbc01a5 f090d50d 5b49fbb9 dd87ea01 5bccb9b2 c24265e0 ceff26ec 6b33cc33) */
/* design/transport.md: the node's resident bridge; its one operand the path of its configuration (no default); a port's
   child as its parent spawns it (child_main) */
int main(int argc,char **argv){
  if(argc>=3 && !strcmp(argv[1],"--child"))return child_main(argc,argv);
  /* differs from the tag: the configuration's writer named, bin/mesh-bridge.sh (sub-assignment 1), where the tag named
     bin/mesh-user.sh and its etc/bridge/README.md, which main has not */
  if(argc!=2)die("usage: mesh-flow CONFIGURATION: the canonical property list bin/mesh-bridge.sh writes for the node's bridge; the bridge has no default and does not start without one");
  char name[MESH_REGION_NAME_BYTES];uint64_t region_bytes;
  /* differs from the tag: main's geometry read with the tag's keys (configuration_read) */
  struct mesh_geometry shape={0};
  const char *refused=configuration_read(argv[1],name,sizeof name,&region_bytes,&shape);
  if(refused)die(refused);
  uint32_t room=sizeof self_path;
  char resolved[PATH_MAX];
  if(_NSGetExecutablePath(self_path,&room) || !realpath(self_path,resolved))die("the bridge's own image");
  snprintf(self_path,sizeof self_path,"%s",resolved);
  /* design/transport.md: every catchable signal whose default ends the process takes the stop, its filter
     added before its default is ignored, so none is lost (one arriving before takes its default action, before anything
     shared is made) */
  struct kevent64_s event;
  static const int stopping[]={SIGINT,SIGTERM,SIGHUP,SIGQUIT,SIGUSR1,SIGUSR2,SIGALRM,SIGVTALRM,SIGPROF,SIGXCPU,SIGXFSZ};
  if((events=kqueue())<0)die("bridge events");
  for(size_t i=0;i<sizeof stopping/sizeof *stopping;i++){
    EV_SET64(&event,(uint64_t)stopping[i],EVFILT_SIGNAL,EV_ADD,0,0,UINT64_MAX,0,0);
    if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
    signal(stopping[i],SIG_IGN);
  }
  signal(SIGPIPE,SIG_IGN);
  uuid_t uuid;struct timespec bound={5,0};
  if(gethostuuid(uuid,&bound)){arc4random_buf(uuid,sizeof uuid);}
  uint32_t node=0;
  for(int i=0;i<16;i++)node=(node*0x01000193u)^uuid[i];
  node=(node&0x7fffffffu)?node&0x7fffffffu:1;
  /* design/transport.md: the node's hello port, the mesh's */
  const uint16_t service=(uint16_t)atoi(MESH_PORT);
  const uint32_t pg=(uint32_t)getpagesize();
  /* Differs from the tag, its interface, main's mesh_layout (mesh.h struct mesh_geometry): the tag laid blocks of one
     page, `region_bytes` the arena's bytes (`pages` region_bytes / pg).  Here a block is the configuration's
     `block_pages` (the tag's 1 where it gives none: mesh_layout divides by it), so the request bound and the region's
     least size are a block's (main's mesh-flow.c checks, block x pg and arena < block), and the rows, the window and
     the page lists' room are the configuration's.  `links` is MESH_PORTS: the link table holds the resident bridge's
     MESH_PORTS ports (mesh.h struct mesh_link_info), hdr `links` its count (mesh-stat reads it), and each
     publication has room for a target a port (`target_stride`) */
  if(!shape.block)shape.block=1;
  /* design/transport.md */
  if((uint64_t)shape.block*pg+MESH_FRAME>MESH_REQUEST_BYTES)die("transport chunk exceeds native request capacity");
  if(region_bytes/pg<shape.block || region_bytes/pg>UINT32_MAX){exit(1);}
  struct hdr geometry={0};
  /* review R2 (L1): `pages` whole blocks, region_bytes / pg cut to the block, as the tag's mesh_layout laid its arena
     (pages / block x block) where main's lays `pages` whole: the layout of an object this bridge makes is then the one
     mesh_layout_within fits to it at the next start (whole blocks), so a restart adopts it.  What must stay: a made
     region's layout its own fit */
  shape.pgsz=pg;shape.pages=(uint32_t)(region_bytes/pg/shape.block*shape.block);shape.links=MESH_PORTS;
  mesh_layout(&geometry,shape);
  /* design/transport.md section 2: the bridge's port bound before its region is read, so a second bridge of the node
     (one started beside the heartbeat's, or while the last one ends) ends at its bind, never touching the region */
  int on=1;
  struct sockaddr_in6 any={.sin6_len=sizeof any,.sin6_family=AF_INET6,.sin6_port=htons(service),.sin6_addr=IN6ADDR_ANY_INIT};
  listener=socket(AF_INET6,SOCK_STREAM,0);hello=socket(AF_INET6,SOCK_DGRAM,0);network=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT);
  struct kev_request filter={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_DL_SUBCLASS};
  unsigned loop=0;
  if(listener<0 || hello<0 || network<0 || setsockopt(listener,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) ||
     setsockopt(hello,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on) || setsockopt(hello,IPPROTO_IPV6,IPV6_MULTICAST_LOOP,&loop,sizeof loop) ||
     bind(listener,(struct sockaddr *)&any,sizeof any) || listen(listener,SOMAXCONN) || bind(hello,(struct sockaddr *)&any,sizeof any) ||
     fcntl(listener,F_SETFL,O_NONBLOCK) || fcntl(hello,F_SETFL,O_NONBLOCK) || fcntl(network,F_SETFL,O_NONBLOCK) || ioctl(network,SIOCSKEVFILT,&filter))
    die("bridge sockets");
  /* design/transport.md: the kernel's IPv6 address events (addresses_heard); a bridge that cannot hear them says a
     port's hello at its pace alone */
  struct kev_request inet6={KEV_VENDOR_APPLE,KEV_NETWORK_CLASS,KEV_INET6_SUBCLASS};
  if((addresses=socket(PF_SYSTEM,SOCK_RAW,SYSPROTO_EVENT))<0 || fcntl(addresses,F_SETFL,O_NONBLOCK) || ioctl(addresses,SIOCSKEVFILT,&inet6)){
    if(addresses>=0)close(addresses);
    addresses=-1;
  }
  int fd=-1;
  const int adopted=region_open(&geometry,shape,&fd);
  if(adopted)region_adopt();
  /* a boot is the bridge's start on the wall clock, so a node's later boot is a newer record */
  M->node=node;M->boot=clock_gettime_nsec_np(CLOCK_REALTIME);M->idle_since=M->boot;
  if(wire_map(&wire,M,fd))die("transport page aliases");
  close(fd);
  /* the tag's `bridge.M=M;` is not here: the bridge's own context is gone (the shared state's note, 2026-11-03) */
  finder=dispatch_queue_create("mesh.ports",DISPATCH_QUEUE_SERIAL);
  /* design/transport.md: the node's memory and hardware model, which its record carries with its region */
  size_t size=sizeof memory;
  if(sysctlbyname("hw.memsize",&memory,&size,NULL,0))memory=0;
  sysctl_text("hw.model",model,sizeof model);
  health_init();
  record_changed();
  /* differs from the tag (2026-11-03, mesh-records.c map_grow's note): the tag's `die("the map's lease")`, its message
     now what an empty map after this node's own record means, there being no lease to fail: map_put could not hold
     the record (its copies' allocation, which leaves the map as it was).  What must stay: no region opens without this
     node's record in its map */
  if(!atomic_load(&M->map_nodes))die("this node's record is not in the map");
  /* left out: the tag's link_reserve() here, its link pages and the duds' block (the coordinator's contract, "Link
     pages": this design leases no link pages and posts nothing on lanes that carry nothing), so each port's `link[p]`
     and `ones` stay MESH_ABSENT as region_open laid them; the bridge holds no arena page (`held_pages` gone,
     2026-11-03) */
  int watched[4]={listener,hello,network,addresses};
  for(int i=0;i<4 && watched[i]>=0;i++){EV_SET64(&event,(uint64_t)watched[i],EVFILT_READ,EV_ADD,0,0,UINT64_MAX,0,0);if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");}
  EV_SET64(&event,0,EVFILT_USER,EV_ADD|EV_CLEAR,0,0,UINT64_MAX,0,0);
  if(kevent64(events,&event,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL))die("bridge events");
  mesh_bridge_pages(M)->bridge_start=mesh_started(getpid());
  atomic_store(&M->bridge_pid,(uint64_t)getpid());
  __sync_synchronize();M->magic=MESH_MAGIC;
  pthread_t watcher;
  if(pthread_create(&watcher,NULL,control_watch,NULL))die("control watcher");
  /* design/transport.md B9b, B9: the peer interfaces' terminations heard from here on; the devices first read
     once IOKit's registry is quiet (no termination in its course: XNU's waitForState, its deadline MESH_PLATFORM_NS),
     else as they stand at the bound */
  terminations_watch(1);
  mach_timespec_t quiet={(unsigned)(MESH_PLATFORM_NS/1000000000u),(clock_res_t)(MESH_PLATFORM_NS%1000000000u)};
  IOKitWaitQuiet(kIOMainPortDefault,&quiet);
  ports_find();
  uint64_t liveness=0;
  while(!stop){
    now=clock_ns();
    if(atomic_load_explicit(&found,memory_order_acquire))ports_found();
    terminations_turn();
    arrivals_turn();
    ports_stalled();
    if(find_wanted && !finding && !atomic_load_explicit(&stalled,memory_order_relaxed) && now>=windows_end()){find_wanted=0;ports_find();}
    for(uint32_t p=0;p<port_count;p++){
      struct port *port=ports+p;
      if(port->job && atomic_load_explicit(&port->done,memory_order_acquire))port_opened(port);
      port_open(port);
      port_hello(port);
      link_turn(port);
      child_turn(port);
    }
    health_turn();
    int alive=now>=liveness;
    if(alive){
      liveness=now+MESH_PAUSE_NS;
      record_retell();
    }
    clients_turn(alive);
    struct kevent64_s ready[16];
    const uint64_t due=turn_due(liveness),at=clock_ns(),rest=due>at?due-at:0;
    struct timespec tick={(time_t)(rest/1000000000u),(long)(rest%1000000000u)};
    int count=kevent64(events,NULL,0,ready,16,0,&tick);
    now=clock_ns();
    for(int i=0;i<count;i++){
      struct kevent64_s *e=ready+i;
      if(e->filter==EVFILT_SIGNAL){stop=1;stop_reason=(int64_t)e->ident;mesh_say("bridge","signal %d: stopping",(int)e->ident);}
      /* design/transport.md section 14, "A process a port": a port's child's message or exit */
      else if(e->udata!=UINT64_MAX && (e->udata&TAG_CHILD)){
        struct port *port=(e->udata&0xff)<port_count?ports+(e->udata&0xff):NULL;
        if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->child)child_reaped(port);
        else if(port && e->filter==EVFILT_PROC && (pid_t)e->ident==port->elsewhere)port->elsewhere=0;
        else if(port && e->filter==EVFILT_READ && (int)e->ident==port->child_fd)child_heard(port);
      }
      /* differs from the tag: main's one client word (hdr `client`) where the tag walked its MESH_CLIENTS slots */
      /* 2026-11-04, owner B2 (e6bd7db2's section 6, "Holders and clients", as design/compilation-coordination.md
         ruling 1 and Q5 take it: "The event loop's EVFILT_PROC retires every owner of the event's pid"), reversing the
         one-client form (hdr `client`, never written from 110, L3): every open claim's owner of that pid retired
         (mesh-records.c clients_gone).  What must stay: a holder's death is this event, not a poll */
      else if(e->filter==EVFILT_PROC)clients_gone((pid_t)e->ident);
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)listener){
        struct sockaddr_in6 from;socklen_t size=sizeof from;int accepted;
        while((accepted=accept(listener,(struct sockaddr *)&from,&size))>=0){
          struct port *port=port_at(from.sin6_scope_id);size=sizeof from;
          /* the coordinator, at integration: the tag's `port->control>=0` is port_paired (mesh-bridge.h) */
          if(!port || port_paired(port) || !port->open){close(accepted);continue;}
          port->control=accepted;
          port_greet(port);
        }
      }
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)hello)hello_heard();
      else if(e->filter==EVFILT_READ && e->ident==(uint64_t)network)network_heard(0);
      else if(e->filter==EVFILT_READ && addresses>=0 && e->ident==(uint64_t)addresses)addresses_heard();
      else if(e->udata<port_count && (int)e->ident==ports[e->udata].control){
        struct port *port=ports+e->udata;
        if(e->filter==EVFILT_WRITE && port->connecting){
          int error=0;socklen_t size=sizeof error;
          if(getsockopt(port->control,SOL_SOCKET,SO_ERROR,&error,&size))error=errno;
          if(error)port_lost(port,error);
          else {port->connecting=0;port_greet(port);}
        }
        else if(e->filter==EVFILT_WRITE){port->writing=0;port_flush(port);}
        else control_heard(port);
      }
    }
  }
  /* design/transport.md: FAREWELL on every control socket (queued as the socket takes it; the kernel sends
     what it holds after the close), every program's words ~0 and its status the stop, stored from this thread, then the
     bridge gone; no queue, thread or verbs call waited on, and every object the kernel's to release at the exit
     (design/RDMA-KERNEL-RECOVERY.md "October 18"), the exit no sooner than every port's quiet window ends
     (design/transport.md B9: its reclaim never inside the termination a conclusion may follow) and a peer
     interface's termination's (`exit_at`, B9b), the kernel's link events read meanwhile, so a link going off then
     extends its port's (network_heard), as a termination then extends them all (peer_terminated), each no later than
     MESH_PLATFORM_NS past the stop's start (`held_to`: every window set before the stop ends by then, so the hold is
     the one 5 s budget, well inside its heartbeat's bound of 3 MESH_PLATFORM_NS) */
  for(uint32_t p=0;p<port_count;p++){
    child_send(ports+p,(struct child_message){.kind=CHILD_HALT});
    port_say(ports+p,(struct message){.kind=FAREWELL,.code=stop_reason},NULL,0);
    if(ports[p].control>=0){close(ports[p].control);ports[p].control=-1;}
  }
  /* 2026-11-04, sub-assignment 7 (design/disjoint-pages-change-request.md, C3.1, C3.6): "every program's words ~0" above
     is the tag's streams'; here the stop's cancellation raises each not-landed receive's count word of the plan written
     to its final value, no phase written (mesh-flow.c streams_cancel, plan_cancel), on this thread, after which no
     release of a claim runs in this bridge.  What must stay: the cancellation before `bridge_pid` is 0 */
  streams_cancel(NULL);
  /* design/transport.md section 2: every stop keeps the region for the next bridge to adopt; a release is the
     heartbeat's, which expires the region once this bridge has ended (mesh-heartbeat.c region_expire) */
  atomic_store(&M->bridge_pid,0);
  mesh_notify(&M->control);
  const uint64_t held_to=clock_ns()+MESH_PLATFORM_NS;
  for(;;){
    now=clock_ns();
    const uint64_t windows=windows_end(),latest=MIN(windows,held_to);
    if(latest<=now)break;
    struct pollfd heard={.fd=network,.events=POLLIN};
    if(poll(&heard,1,(int)((latest-now+999999)/1000000))>0){now=clock_ns();network_heard(1);}
  }
  /* the tag's main from here (rdma/mesh-link.c:2079-2121: CHILD_END to every child, their exits awaited on a queue of
     their own, SIGKILL past the bound, one the kernel still holds named), mesh-children.c's */
  children_end();
  return 0;
}
