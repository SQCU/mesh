#ifndef MESH_OBSERVE_H
#define MESH_OBSERVE_H
#include "mesh.h"
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
/* design/transport.md section 2; section 15, "The bridge's own pages"; section 16 */
#define MESH_LINK_STREAMS 4u
#define MESH_PORTS (MESH_STREAMS/MESH_LINK_STREAMS)
_Static_assert(MESH_PORTS*MESH_LINK_STREAMS==MESH_STREAMS,"a port's streams");
enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_STOPPED };
struct mesh_link_info { _Atomic uint32_t peer,phase; _Atomic int64_t code; _Atomic uint64_t bandwidth,pairings; };
_Static_assert(sizeof(struct mesh_link_info)==32 && MESH_PORTS*sizeof(struct mesh_link_info)<=MESH_BRIDGE_BYTES,"mesh_link_info");
static inline struct mesh_link_info *mesh_links(struct hdr *m){ return (struct mesh_link_info *)((char *)m+m->bridge_off); }
struct mesh_holder { uint32_t pid,device; uint64_t start; };
struct mesh_bridge_pages { uint32_t link[MESH_PORTS],lists[MESH_PORTS],ones,map,map_pages,padding; uint64_t bridge_start;
  struct mesh_holder child[2*MESH_PORTS];
  _Atomic uint64_t turned,hung_from,progress[MESH_PORTS],active,hung,hangs; };
_Static_assert(MESH_PORTS*sizeof(struct mesh_link_info)+sizeof(struct mesh_bridge_pages)<=MESH_BRIDGE_BYTES,"mesh_bridge_pages");
static inline void mesh_say(const char *prefix,const char *format,va_list arguments){
  struct timespec t;clock_gettime(CLOCK_REALTIME,&t);
  struct tm utc;gmtime_r(&t.tv_sec,&utc);
  char line[4096];
  size_t n=strftime(line,sizeof line,"%Y-%m-%dT%H:%M:%S",&utc);
  n+=(size_t)snprintf(line+n,sizeof line-n,".%03ldZ %.3f %s",t.tv_nsec/1000000,(double)clock_gettime_nsec_np(CLOCK_MONOTONIC)/1e9,prefix);
  vsnprintf(line+n,sizeof line-n,format,arguments);
  fputs(line,stderr);
}
static inline struct mesh_bridge_pages *mesh_bridge_pages(struct hdr *m){
  return (struct mesh_bridge_pages *)((char *)m+m->bridge_off+MESH_PORTS*sizeof(struct mesh_link_info));
}
#endif
