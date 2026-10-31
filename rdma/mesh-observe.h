/* design/transport.md: what the bridge publishes for its own observation tools (mesh-stat, the node services) in its
   own pages of the region (struct hdr `bridge_off`): a port of the node by the index the bridge gave it, its peer node
   while paired, its phase (MESH_UNKNOWN where the bridge has no such port), the code it last stopped with, its
   bandwidth and the pairings it made.  Mesh code alone includes this: no client header does, and no client reads these
   pages (design/rubrics.md, Three kinds of source) */
#ifndef MESH_OBSERVE_H
#define MESH_OBSERVE_H
#include "mesh.h"
#include <fcntl.h>
#include <stdarg.h>
#include <sys/stat.h>
/* design/transport.md section 14: a line of the node's bridge log, the launchd job's stderr (bin/mesh-user.sh
   StandardErrorPath): the UTC time, `who` and the message, in one write; the losses and failures of the bridge and its
   heartbeat, nothing on the data path's success.  A log that has reached MESH_LOG_BYTES is emptied first (the processes
   that share it append: O_APPEND), so it holds at most that and a line */
__attribute__((format(printf,2,3))) static inline void mesh_say(const char *who,const char *format,...){
  char line[MESH_DIAGNOSTIC_BYTES];
  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);
  struct tm utc;gmtime_r(&t.tv_sec,&utc);
  int n=snprintf(line,sizeof line,"%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ %s: ",utc.tm_year+1900,utc.tm_mon+1,utc.tm_mday,
                 utc.tm_hour,utc.tm_min,utc.tm_sec,t.tv_nsec/1000000,who);
  va_list arguments;va_start(arguments,format);
  if(n>0 && (size_t)n<sizeof line)n+=vsnprintf(line+n,sizeof line-(size_t)n,format,arguments);
  va_end(arguments);
  if(n<=0)return;
  if((size_t)n>=sizeof line-1)n=(int)sizeof line-2;
  line[n++]='\n';
  struct stat log;
  const int flags=fcntl(2,F_GETFL);
  if(!fstat(2,&log) && S_ISREG(log.st_mode) && flags>=0){
    if(!(flags&O_APPEND))fcntl(2,F_SETFL,flags|O_APPEND);
    if(log.st_size>=MESH_LOG_BYTES)(void)!ftruncate(2,0);
  }
  (void)!write(2,line,(size_t)n);
}
/* a link's streams, and so the most ports: link p's streams are the region's p MESH_LINK_STREAMS to p MESH_LINK_STREAMS
   + MESH_LINK_STREAMS - 1 */
#define MESH_PORTS (MESH_STREAMS/MESH_LINK_STREAMS)
_Static_assert(MESH_PORTS*MESH_LINK_STREAMS==MESH_STREAMS,"a port's streams");
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
struct mesh_link_info { _Atomic uint32_t peer,phase; _Atomic int64_t code; _Atomic uint64_t bandwidth,pairings; };
_Static_assert(sizeof(struct mesh_link_info)==32 && MESH_PORTS*sizeof(struct mesh_link_info)<=MESH_BRIDGE_BYTES,"mesh_link_info");
static inline struct mesh_link_info *mesh_links(struct hdr *m){ return (struct mesh_link_info *)((char *)m+m->bridge_off); }
/* design/transport.md section 2: the bridge's own pages of the region, recorded beside the links so the next bridge
   adopts them with the region instead of leasing them again: each port's link pages and its streams' lists, the duds' block and the
   map's pages (MESH_ABSENT where none); and the processes of the bridge that hold the region, each by its pid and start
   (mesh_started): the bridge's own start (beside the header's `bridge_pid`) and each child, which registers the region
   and holds it until its exit, in an entry no live child holds (twice the ports: an earlier launch's child that has not
   ended keeps its entry), so the heartbeat unlinks an expired region only once none of them lives.  Fields are only
   ever appended */
struct mesh_holder { uint32_t pid,padding; uint64_t start; };
struct mesh_bridge_pages { uint32_t link[MESH_PORTS],ones,map,map_pages,padding; uint64_t bridge_start;
  struct mesh_holder child[2*MESH_PORTS]; };
_Static_assert(MESH_PORTS*sizeof(struct mesh_link_info)+sizeof(struct mesh_bridge_pages)<=MESH_BRIDGE_BYTES,"mesh_bridge_pages");
static inline struct mesh_bridge_pages *mesh_bridge_pages(struct hdr *m){
  return (struct mesh_bridge_pages *)((char *)m+m->bridge_off+MESH_PORTS*sizeof(struct mesh_link_info));
}
/* each port's bytes landed on its link, appended after the bridge's pages: the payload bytes of every receive its
   progress thread took (link_receive, each completion's byte_len), since the region was made, with the clock
   (CLOCK_MONOTONIC, ns) of the pass that took the last of them.  The progress thread writes the pair to slot
   (sequence + 1) & 1, then stores sequence + 1 (release), and never waits; a reader (mesh-stat) loads sequence
   (acquire), the pair at slot sequence & 1, and keeps it where sequence has since moved by less than 2.  Nothing on the
   data path reads it: a rate is two pairs' bytes over their clocks, as exact as the bridge's clock, however late the
   reader looks */
struct mesh_link_landed { _Atomic uint64_t sequence; _Atomic uint64_t at[2][2]; };
_Static_assert(MESH_PORTS*(sizeof(struct mesh_link_info)+sizeof(struct mesh_link_landed))+sizeof(struct mesh_bridge_pages)<=MESH_BRIDGE_BYTES,
               "mesh_link_landed");
static inline struct mesh_link_landed *mesh_links_landed(struct hdr *m){
  return (struct mesh_link_landed *)((char *)mesh_bridge_pages(m)+sizeof(struct mesh_bridge_pages));
}
#endif
