/* The probes of a process that holds this node's region, each holder recorded in the region by its pid and start
   (mesh.h): the bridge (hdr `bridge_pid`, struct mesh_bridge_pages `bridge_start`), each port's child (a struct
   mesh_holder of mesh_bridge_pages(m)->child: pid, device `unit`, start) and each publication's publisher (its record's
   whole identity, struct mesh_identity: its client's low 32 bits the pid, and its start).  The holder table's operations are lines of its readers and writers at tag
   pre-return-20261103 (mesh-link.c child_elsewhere, child_spawn, child_reaped, region_open and main; mesh-heartbeat.c
   bridge_of and replace_turn), brought over verbatim by their owners and calling these; none is a function here. */
#ifndef MESH_HOLDERS_H
#define MESH_HOLDERS_H
#include "mesh.h"
/* tag pre-return-20261103 rdma/mesh-dataflow.h: the headers mesh_gone, mesh_started and mesh_holds need (07949b28) */
#include <errno.h>
#include <libproc.h>
#include <signal.h>
#include <sys/proc.h>
/* tag pre-return-20261103 rdma/mesh-dataflow.h mesh_gone, verbatim (written 2310753d) */
/* a process gone (the client's or the bridge's pid; 0: none) */
static inline int mesh_gone(pid_t process){ return !process || (kill(process,0) && errno==ESRCH); }
/* tag pre-return-20261103 rdma/mesh-dataflow.h mesh_started, verbatim (b346ef2b; in its exit a holder 8ac5c9fb, reverted c18f7a0e, restored 07949b28) */
/* a process's start in microseconds of the wall clock (libproc PROC_PIDTBSDINFO), 0 where it is gone; with its pid,
   the identity of a holder of the region, which a later process given the same pid does not share.  A process in its
   exit is not gone: it holds its registrations and devices until the kernel has released them, and XNU takes it from
   every lookup by pid as its exit begins (kern_exit.c proc_prepareexit), so it is read as a zombie is (argument 1:
   proc_info.c proc_find_zombref) and gone only once its status is SZOMB, its exit whole */
static inline uint64_t mesh_started(pid_t process){
  struct proc_bsdinfo info;
  return process && proc_pidinfo(process,PROC_PIDTBSDINFO,1,&info,(int)sizeof info)==(int)sizeof info &&
         info.pbi_status!=SZOMB?(uint64_t)info.pbi_start_tvsec*UINT64_C(1000000)+info.pbi_start_tvusec:0;
}
/* tag pre-return-20261103 rdma/mesh-dataflow.h mesh_holds, verbatim (written b346ef2b) */
/* the process `pid` of start `start` alive (a start of 0, which a region an earlier bridge or library wrote holds:
   the pid alone) */
static inline int mesh_holds(pid_t pid,uint64_t start){ return start?mesh_started(pid)==start:!mesh_gone(pid); }
/* a whole holder identity alive (mesh_holds, the one failure detector), and its 16-byte atomics: one compare-and-
   swap of the whole identity (arm64 caspal), one store */
static inline int mesh_holder_live(struct mesh_identity holder){
  return mesh_holds((pid_t)(uint32_t)holder.client,holder.start);
}
static inline int mesh_holder_swap(struct mesh_identity *at,struct mesh_identity *expected,struct mesh_identity desired){
  return __atomic_compare_exchange((unsigned __int128 *)(void *)at,(unsigned __int128 *)(void *)expected,
                                   (unsigned __int128 *)(void *)&desired,0,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE);
}
static inline void mesh_holder_store(struct mesh_identity *at,struct mesh_identity value){
  __atomic_store((unsigned __int128 *)(void *)at,(unsigned __int128 *)(void *)&value,__ATOMIC_RELEASE);
}
/* A diagnostic's copy of row `row`'s open publication's holder, from a mapping of `length` bytes that may be read-only
   (mesh-stat.c, mesh-heartbeat.c), with no hazard, lock or registration: the root row's slot G read (acquire); its
   mapping (the record's page) and the record's two holder scalars each read once (relaxed atomic loads), at addresses
   bounded by the mapping; an acquire fence; the same slot read again.  1 and *holder where both reads are the same open
   G, else 0 (the diagnostic's next turn reads again).  Why it is the holder: a slot goes 0, then G (mesh-collective.c
   plan_publish: a unique even generation, `generation` raised by 2), then G | 1 (mesh-dataflow.c mesh_retire,
   mesh_retired_release's close), then 0 (the release), and a record's page is reused only by a preparation after its
   slot is 0 (plan_reserve pins every slot not 0), so the same G read after the copies means no release, and no reuse,
   came between.  No pointer is kept past the second read.  The copy is an observation: whether its process lives is
   mesh_holds' (mesh_holder_live) */
static inline int mesh_holder_snapshot(struct hdr *m,size_t length,uint32_t row,struct mesh_identity *holder){
  if(m->page_off>length || (length-m->page_off)/sizeof(struct mesh_page_entry)<=row)return 0;
  const uint64_t slot=mesh_slot(m,row);
  if(!mesh_slot_open(slot))return 0;
  const uint64_t page=(uint32_t)atomic_load_explicit(&mesh_page(m)[row].mapping,memory_order_relaxed);
  if(m->data_off>length || !m->pgsz || page>(length-m->data_off)/m->pgsz ||
     length-m->data_off-page*m->pgsz<sizeof(struct mesh_algo))return 0;
  const struct mesh_algo *algo=(const struct mesh_algo *)(const void *)((const char *)m+m->data_off+page*m->pgsz);
  const struct mesh_identity copied={__atomic_load_n(&algo->holder.client,__ATOMIC_RELAXED),
                                     __atomic_load_n(&algo->holder.start,__ATOMIC_RELAXED)};
  atomic_thread_fence(memory_order_acquire);
  if(mesh_slot(m,row)!=slot)return 0;
  *holder=copied;
  return 1;
}
#endif
