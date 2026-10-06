/* mesh-recovery.h: a link's recovery from loss, as one module.  Which losses a link survives and for how long, how a
   loss is told (detection), how the link is suspended, paired again and resumed at what each side landed, how an
   orderly end is told from a loss, and when the link gives up to its call (metal-microbench docs/elastic.md, whose
   driver and ranks take over past it).

   STATUS: CRITICAL PRIORITY.  Unfinished, unsettled and not properly reviewed: design/recovery.md holds the
   requirements and mandates it answers to, which of them it meets, which it does not, and which no live test can
   reach yet.  It runs as built in the interim (its first form inside mesh-flow.c: mesh 828e41c, 33a9d35, 3c6fc6a).

   The module holds its own state (struct mesh_recovery), its policy (the window), its detection (the control
   socket's keepalive, the farewell) and its protocol (the landed counts' exchange, the barrier), and reaches the
   transport only through struct mesh_recovery_transport: a link's threads halted and its landed receives taken, its
   sockets and queue pairs released, a pairing again on new queue pairs, whether the call abandoned the link, and its
   threads restarted; and tells the first pairing when a refusal may be tried again.  It includes mesh-verbs.h for the
   pairing layer's exchange. */
#ifndef MESH_RECOVERY_H
#define MESH_RECOVERY_H
#include "mesh-verbs.h"
#include <poll.h>
#include <signal.h>

/* design/recovery.md#window */
/* How long a lost link may take to pair again before it gives up to its call: MESH_RESUME_SECONDS (default 10, the
   ranks' silent bound, past which they cancel it and the GPU's waits on it end); 0 gives up at once. */
static uint64_t mesh_recovery_window_ns(void){
  const char *given=getenv("MESH_RESUME_SECONDS");
  const double seconds=given?atof(given):10.0;
  return seconds>0?(uint64_t)(seconds*1e9):0;
}

/* design/recovery.md#pause */
/* The pause before every pairing again (and between first pairings a down port refuses): the pause a lost session
   took before pairing again (mesh be9731b), since failed completions, queue-pair teardown and immediate re-pairing
   preceded the 2026-09-05 kernel panic.  MESH_RECOVERY_PAUSE_MS, default 3000. */
static int mesh_recovery_pause_ms(void){
  const char *given=getenv("MESH_RECOVERY_PAUSE_MS");
  return given?atoi(given):3000;
}

/* design/recovery.md#first */
/* Whether a first pairing that failed may try again within its pairing window: its port was not active (ENETDOWN,
   device_up, before any queue pair was made), as a link down at a call's start is (Q4). */
static int mesh_recovery_first_again(int error){ return error==ENETDOWN; }

/* design/recovery.md#detection */
/* The control socket's keepalive: a side whose own link stays up (its peer's interface went down, its peer's host
   died) is told by the socket's end within idle + interval x count seconds; the queue pairs (UC) report nothing.
   0, or setsockopt's error. */
static int mesh_recovery_keepalive(int control){
  int on=1,idle=1,interval=1,count=2;
  if(setsockopt(control,SOL_SOCKET,SO_KEEPALIVE,&on,sizeof on) || setsockopt(control,IPPROTO_TCP,TCP_KEEPALIVE,&idle,sizeof idle) ||
     setsockopt(control,IPPROTO_TCP,TCP_KEEPINTVL,&interval,sizeof interval) || setsockopt(control,IPPROTO_TCP,TCP_KEEPCNT,&count,sizeof count))
    return errno?errno:EIO;
  return 0;
}

/* design/recovery.md#loss */
/* A link's loss: the first loss's code and domain (a later one while it recovers is the same loss), and the link's
   resumptions. */
struct mesh_recovery { _Atomic int64_t lost; uint32_t domain; uint64_t resumes; };

/* A loss the link may survive (its cable's link-off, its control socket's end without a farewell, a failed
   completion or posting): 1 where the link is to suspend and pair again, the loss recorded; 0 where recovery is off
   (the window 0), and the link stops. */
static int mesh_recovery_lose(struct mesh_recovery *recovery,int64_t code,uint32_t domain){
  if(!mesh_recovery_window_ns())return 0;
  int64_t none=0;
  if(atomic_compare_exchange_strong(&recovery->lost,&none,code?code:EIO))recovery->domain=domain;
  return 1;
}
static int mesh_recovery_pending(struct mesh_recovery *recovery){ return atomic_load_explicit(&recovery->lost,memory_order_acquire)!=0; }
/* The loss given up to the call: its code, the link no longer lost (it stops once). */
static int64_t mesh_recovery_taken(struct mesh_recovery *recovery){ return atomic_exchange(&recovery->lost,0); }

/* design/recovery.md#farewell */
/* An orderly end tells the peer so, which a lost link cannot: the peer then stops rather than resumes. */
#define MESH_FAREWELL 0x46
static void mesh_recovery_farewell(int control){ const char farewell=MESH_FAREWELL;send(control,&farewell,1,MSG_DONTWAIT); }
/* Whether the readable control socket holds the peer's farewell (an orderly end), not its silence or reset (a loss). */
static int mesh_recovery_parted(int control){ char said=0;return recv(control,&said,1,MSG_DONTWAIT)==1 && said==MESH_FAREWELL; }

/* design/recovery.md#resumption */
/* Each side tells the other its queue pairs' landed counts (its rings' records landed, in the order posted) on the
   new control socket: `peer` receives the peer's.  0, or -1 with errno (EPROTO: the peer is no resumption of the
   same link, a fresh pairing). */
#define MESH_RESUME 0x4d524553554d45ull
static int mesh_recovery_exchange(int control,int queues,const uint64_t *mine,uint64_t *peer,struct hdr *m,uint64_t client,uint64_t deadline){
  uint64_t *sent=calloc((size_t)queues+1,sizeof *sent),*heard=calloc((size_t)queues+1,sizeof *heard);
  if(!sent||!heard){free(sent);free(heard);errno=ENOMEM;return -1;}
  sent[0]=MESH_RESUME;
  memcpy(sent+1,mine,(size_t)queues*sizeof *mine);
  if(exchange(control,sent,heard,((size_t)queues+1)*sizeof *sent,((size_t)queues+1)*sizeof *heard,m,client,deadline)){
    int error=errno;free(sent);free(heard);errno=error;return -1;}
  if(heard[0]!=MESH_RESUME){free(sent);free(heard);errno=EPROTO;return -1;}
  memcpy(peer,heard+1,(size_t)queues*sizeof *peer);
  free(sent);free(heard);
  return 0;
}
/* The barrier after each side posted its receives again: no SEND before the peer's receive is posted.  0, or -1. */
static int mesh_recovery_barrier(int control,struct hdr *m,uint64_t client,uint64_t deadline){
  uint32_t posted=1,peer_posted;
  return exchange(control,&posted,&peer_posted,sizeof posted,sizeof peer_posted,m,client,deadline);
}

/* design/recovery.md#suspension */
/* The transport's side of a resumption, each of the link `link`. */
struct mesh_recovery_transport {
  void *link;
  /* its progress threads stopped and joined, the receives that landed before the loss taken */
  void (*halt)(void *link);
  /* its control socket, listener and queue pairs closed: 0, or the error */
  int (*release)(void *link,int *control);
  /* a pairing again on new queue pairs within `window` ns (verbs_up with the transport's resumption, which exchanges
     the landed counts: mesh_recovery_exchange, mesh_recovery_barrier): the control socket, or -1 */
  int (*pair)(void *link,uint64_t window);
  /* whether the call abandoned the link: the bridge stopped, its client exited, another client took the region, a
     rank cancelled the link (its silent bound) */
  int (*abandoned)(void *link);
  /* the new control socket watched (the outage's link events dropped): 0, or the error */
  int (*watch)(void *link,int control);
  /* its progress threads started again: 0, or the error */
  int (*restart)(void *link);
};

/* A lost link suspended (MESH_SUSPENDED: its phase), its transport halted and released, then paired again until it
   pairs or the window passes or the call abandons it; resumed, its phase MESH_PAIRED.  The ranks' reads wait
   meanwhile.  0, resumed; else the reason it gave up (the loss stays recorded for the caller to stop the link). */
static int mesh_recovery_resume(struct mesh_recovery *recovery,const struct mesh_recovery_transport *transport,
                                _Atomic uint32_t *phase,uint32_t index,int *control){
  const uint64_t began=clock_gettime_nsec_np(CLOCK_MONOTONIC),deadline=began+mesh_recovery_window_ns();
  atomic_store_explicit(phase,MESH_SUSPENDED,memory_order_release);
  fprintf(stderr,"link %u suspended: code %lld domain %u\n",index,(long long)atomic_load(&recovery->lost),recovery->domain);
  transport->halt(transport->link);
  int error=transport->release(transport->link,control);
  if(error)return error;
  for(;;){
    if(transport->abandoned(transport->link))return ECANCELED;
    if(clock_gettime_nsec_np(CLOCK_MONOTONIC)+(uint64_t)mesh_recovery_pause_ms()*1000000ull>=deadline)return ETIMEDOUT;
    poll(NULL,0,mesh_recovery_pause_ms());
    if(transport->abandoned(transport->link))return ECANCELED;
    const uint64_t now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    int f=transport->pair(transport->link,deadline-now);
    if(f>=0){
      error=mesh_recovery_keepalive(f);
      if(!error){*control=f;break;}
      close(f);
    }
    error=transport->release(transport->link,control);
    if(error)return error;
  }
  error=transport->watch(transport->link,*control);
  if(error)return error;
  atomic_store_explicit(&recovery->lost,0,memory_order_release);
  error=transport->restart(transport->link);
  if(error)return error;
  recovery->resumes++;
  atomic_store_explicit(phase,MESH_PAIRED,memory_order_release);
  fprintf(stderr,"link %u resumed after %.3f s (resumption %llu)\n",index,
    (double)(clock_gettime_nsec_np(CLOCK_MONOTONIC)-began)/1e9,(unsigned long long)recovery->resumes);
  return 0;
}
#endif
