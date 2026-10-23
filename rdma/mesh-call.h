#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh.h"
#include <errno.h>
#include <libproc.h>
#include <signal.h>
/* design/transport.md section 15, "The library" */
struct mesh_ctx { struct hdr *M; size_t len; uint64_t client; int fd; };
int mesh_attach(struct mesh_ctx *);
int mesh_detach(struct mesh_ctx *);
static inline int mesh_gone(pid_t process){ return !process || (kill(process,0) && errno==ESRCH); }
static inline uint64_t mesh_started(pid_t process){
  struct proc_bsdinfo info;
  return process && proc_pidinfo(process,PROC_PIDTBSDINFO,0,&info,(int)sizeof info)==(int)sizeof info?
         (uint64_t)info.pbi_start_tvsec*UINT64_C(1000000)+info.pbi_start_tvusec:0;
}
static inline int mesh_holds(pid_t pid,uint64_t start){ return start?mesh_started(pid)==start:!mesh_gone(pid); }
void mesh_retire(struct hdr *,uint64_t client);
uint32_t mesh_joined(const struct mesh_record *,const struct mesh_record *,uint32_t *cable,uint64_t *bandwidth,uint32_t room);
uint32_t mesh_arena_alloc(struct mesh_ctx *,uint32_t pages,uint32_t align);
uint32_t mesh_arena_free(struct hdr *);
int mesh_lease_return(struct mesh_ctx *,uint32_t row);
uint32_t mesh_retired_release(struct hdr *,uint64_t client);
struct mesh_section { uint32_t first,pages; size_t bytes; uint32_t count,stride; };
int mesh_section_create(struct mesh_ctx *,size_t bytes,uint32_t count,struct mesh_section *);
void *mesh_section_address(struct mesh_ctx *,struct mesh_section,uint32_t index);
/* design/transport.md section 14, "The queue" */
static inline struct mesh_client *mesh_slot(struct hdr *m,uint32_t s){ return (struct mesh_client *)((char *)m+m->client_off)+s; }
void mesh_queue_link(struct mesh_ctx *,struct mesh_plan *);
int mesh_queue_unlink(struct mesh_ctx *,struct mesh_plan *);
struct mesh_walk { uint64_t word[MESH_PORTS+1][2]; };
void mesh_walks(struct hdr *,struct mesh_walk *);
int mesh_walked(struct hdr *,const struct mesh_walk *);
int mesh_plan_balanced(const struct mesh_plan *);
uint32_t mesh_names_find(struct hdr *,const uint64_t (*name)[2],uint32_t count,uint32_t *node);
#endif
