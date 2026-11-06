/* The probes of a process that holds this node's region, each holder recorded in the region by its pid and start
   (mesh.h): the bridge (hdr `bridge_pid`, struct mesh_bridge_pages `bridge_start`), each port's child (a struct
   mesh_holder of mesh_bridge_pages(m)->child: pid, device `unit`, start) and the attached client (hdr `client`'s low 32
   bits, hdr `client_start`).  The holder table's operations are lines of its readers and writers at tag
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
/* The open operand's holder (2026-11-06, design/compilation-coordination.md, "The root cause of the copies, and the
   revised work list", R-8: "a gone holder is read from the heartbeat's words", and R-10: "the holder's read and
   gone-holder test", W13: "(mesh-link.c 377–381; mesh-records.c 1425–1430, 1491–1493; mesh-stat.c 118–120)", each
   merged; "The revisions of 14b9a757"): hdr `client`'s low 32 bits its pid and hdr `client_start` its start, the words
   mesh-heartbeat.c replace_turn reads as its one client holder, read after the publication's acquire (mesh.h
   mesh_algo_open: mesh_plan stores them before `published`); {0} where the operand is not open.  The one read of the
   holder: the bridge parent's turn and its exit event, the region's adoption, mesh-stat, the release and a port's test
   of a plan's standing read it here, and test it gone by mesh_holds(pid, start) alone; none writes `retired` for it
   (R-8).  What must stay: the holder read after the publication */
struct mesh_algo_holder { pid_t pid; uint32_t padding; uint64_t start; };
static inline struct mesh_algo_holder mesh_algo_holder(struct hdr *m){
  if(!mesh_algo_open(m))return (struct mesh_algo_holder){0};
  return (struct mesh_algo_holder){.pid=(pid_t)(uint32_t)atomic_load_explicit(&m->client,memory_order_acquire),
                                   .start=atomic_load_explicit(&m->client_start,memory_order_acquire)};
}
#endif
