/* design/transport.md: what the bridge publishes for its own observation tools (mesh-stat, the node services) in its
   own pages of the region (struct hdr `bridge_off`): a port of the node by the index the bridge gave it, its peer node
   while paired, its phase (MESH_UNKNOWN where the bridge has no such port), the code it last stopped with, its
   bandwidth and the pairings it made; and the one watch on the exits of the region's processes (mesh_exits_watch).
   Mesh code alone includes this: no client header does, and no client reads these pages
   (design/rubrics.md, Three kinds of source) */
#ifndef MESH_OBSERVE_H
#define MESH_OBSERVE_H
#include "mesh.h"
#include <time.h>
#include <sys/event.h>
#include <sys/wait.h>
#include <sysexits.h>
/* a port's lane records: port p's are the region's p MESH_LINK_STREAMS to p MESH_LINK_STREAMS + MESH_LINK_STREAMS - 1
   (mesh.h) */
_Static_assert(MESH_PORTS*MESH_LINK_STREAMS==MESH_STREAMS,"a port's lane records");
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
struct mesh_link_info { _Atomic uint32_t peer,phase; _Atomic int64_t code; _Atomic uint64_t bandwidth,pairings; };
_Static_assert(sizeof(struct mesh_link_info)==32 && MESH_PORTS*sizeof(struct mesh_link_info)<=MESH_BRIDGE_BYTES,"mesh_link_info");
static inline struct mesh_link_info *mesh_links(struct hdr *m){ return (struct mesh_link_info *)((char *)m+m->bridge_off); }
/* design/transport.md section 2: the bridge's own pages of the region, recorded beside the links so the next bridge
   adopts them with the region instead of leasing them again: the map's pages (MESH_ABSENT where none); and the
   processes of the bridge that map the region, each by its pid and start (mesh_started): the bridge's own start
   (beside the header's `bridge_pid`) and each child, which registers the region and maps it until its exit, in an
   entry no live child has (twice the ports: an earlier launch's child that has not ended keeps its entry), so the
   heartbeat unlinks an expired region only once none of them lives.  Fields are only ever appended, so a heartbeat of
   any build finds `bridge_start` and `child` at their offsets; `unused` is written by no build */
struct mesh_process { uint32_t pid,padding; uint64_t start; };
struct mesh_bridge_pages { uint32_t unused[2*MESH_PORTS+1],map,map_pages,padding; uint64_t bridge_start;
  struct mesh_process child[2*MESH_PORTS]; };
_Static_assert(MESH_PORTS*sizeof(struct mesh_link_info)+sizeof(struct mesh_bridge_pages)<=MESH_BRIDGE_BYTES,"mesh_bridge_pages");
static inline struct mesh_bridge_pages *mesh_bridge_pages(struct hdr *m){
  return (struct mesh_bridge_pages *)((char *)m+m->bridge_off+MESH_PORTS*sizeof(struct mesh_link_info));
}

/* design/transport.md section 2: the watch on the exits of the region's processes (the bridge, its children, the
   clients), written once for every caller: the bridge for the children of the bridge before (mesh-link.c
   adopting_read, NODE_ADOPTING), and the heartbeat for an expired region's processes (mesh-heartbeat.c
   region_replace).
   Whether the process a struct mesh_process names is alive: a child of the caller until its exit is complete (waitid
   with WNOWAIT, which leaves it to be reaped; until it is reaped its pid names it alone), any other process until its
   exit begins (mesh_alive); pid 0 names none */
static inline int mesh_lives(struct mesh_process process){
  if(!process.pid)return 0;
  siginfo_t info={0};
  if(!waitid(P_PID,(id_t)process.pid,&info,WEXITED|WNOHANG|WNOWAIT))return !info.si_pid;
  return mesh_alive((pid_t)process.pid,process.start);
}
/* the exits of processes[0 .. count) watched on kqueue `queue`, never waited for here: each live process's exit is
   watched there (EVFILT_PROC NOTE_EXIT, one-shot, udata `tag`), and one whose watch finds it exited (ESRCH) is read
   again, so each exit is either seen here or told on the queue, which the caller's own loop reads.  The live processes
   are moved to the front of `processes`, as they were given, and their count returned.  A live process whose exit
   cannot be watched ends the caller (EX_OSERR: an exit, never an abort, whose crash report would write to storage), so
   no caller waits on an exit that no event will tell */
static inline uint32_t mesh_exits_watch(int queue,struct mesh_process *processes,uint32_t count,uint64_t tag){
  uint32_t alive=0;
  for(uint32_t k=0;k<count;k++){
    if(!mesh_lives(processes[k]))continue;
    struct kevent64_s watch;
    EV_SET64(&watch,processes[k].pid,EVFILT_PROC,EV_ADD|EV_ONESHOT,NOTE_EXIT,0,tag,0,0);
    if(kevent64(queue,&watch,1,NULL,0,KEVENT_FLAG_IMMEDIATE,NULL)){
      if(errno!=ESRCH)exit(EX_OSERR);
      if(!mesh_lives(processes[k]))continue;
    }
    processes[alive++]=processes[k];
  }
  return alive;
}
#endif
