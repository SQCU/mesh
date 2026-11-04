#include <signal.h>
#include "mesh-dataflow.h"
/* the library's own internals that reduce the map to node pairs (mesh_cables, mesh_joined, mesh_nodes_read) and wait on
   the ports' data paths (mesh_links, mesh_transfers_start): the bridge-side region types, which no client header
   includes (review R1, C1) */
#include "mesh-map.h"
/* mesh_transfers_start's declaration: defined here, beside the map reader it reads */
#include "mesh-call.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
/* design/algorithm-sources.md#programtensor */
/* A plan's claim (design/disjoint-pages-change-request.md, Appendix C, C3.4; design/disjoint-pages-revision-
   coordination.md, "The program's claim and its sections"; mesh-dataflow.h mesh_claim_take): the only place rows and
   pages are taken, mesh_plan's one call, for exactly what that plan places.  Any number of claims hold disjoint runs at
   once, a retired plan's beside the next plan's, until the bridge releases it (mesh_retired_release), which nobody
   waits for.  Nothing else takes rows or pages from the region: the bridge's own areas are laid out with it
   (mesh-map.h mesh_layout).
   The claim's record, in its head row (the run's first row), which no value's rows include, so nothing but the claim,
   its close and its release writes it:
   - the head's struct mesh_buffer: `owner` the client (stored last, release; 0 once released), `closed` 0 (1 once
     closed: mesh_claim_close, mesh_retire), `pages` the claim's rows, the head among them (rows.count), `constant` 0;
   - the head's struct mesh_page_entry: `mapping` the window run, window.count << 32 | window.first; `address` the
     local run, local.count << 32 | local.first (an empty run {0, 0}: 0); `device` the claim's generation (the plan's,
     hdr `plan` once that plan is written; never 0).
   No row but a head holds an owner word.  A claim's rows are held by their bits of MESH_ROW_OWN and its pages by their
   bits of the arena's, so a walk of the heads reads the set bits of MESH_ROW_OWN, from each head past its rows.
   2026-11-04, sub-assignment 5: reversing sub-assignment 7's 2026-11-03 form (mesh_claim, at attach): one claim a
   client, taken at attach as the widest run of free rows and of free pages in the window and outside it (one run
   halved where the window was the whole arena), recorded in two head rows, the window run in the first and the local
   run in the second, by mesh_bit_widest's scan.  Why: C3.4 ("Nothing is claimed at attach"; a program holds exactly
   the pages its plan uses) and C4 (mesh_claim's claim at attach of everything free; the claim's local run, where
   values were laid out when created).  What must stay: a claim's rows let go after its pages, and only by the winner
   of its head's owner exchange; no row but a head holds an owner word. */

/* the first index in [at, end) whose bit of `bits` is `set`, else `end` */
static uint32_t mesh_bit_next(_Atomic uint64_t *bits,uint32_t at,uint32_t end,int set){
  while(at<end){
    uint64_t word=atomic_load_explicit(&bits[at/64],memory_order_acquire);
    if(!set)word=~word;
    word&=~UINT64_C(0)<<(at%64);
    if(word){
      const uint32_t found=at/64*64+(uint32_t)__builtin_ctzll(word);
      return found<end?found:end;
    }
    at=at/64*64+64;
  }
  return end;
}

/* the row a walk of the claims' heads reads after `row`, whose owner word read `owner` (acquire): past the claim's rows
   where `row` is a head (its `pages`, stored before its owner word), else the next row (a claim being taken, its owner
   word not yet stored, or being released); there or after it, the first row MESH_ROW_OWN holds, else mesh_rows */
static uint32_t mesh_head_after(struct hdr *m,uint32_t row,uint64_t owner){
  const uint32_t rows=mesh_rows(m),claimed=owner?mesh_buffers(m)[row].pages:0;
  const uint64_t next=(uint64_t)row+(claimed?claimed:1);
  return next>=rows?rows:mesh_bit_next(mesh_plane(m,MESH_ROW_OWN),(uint32_t)next,rows,1);
}

/* design/algorithm-sources.md#programtensor */
/* 2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 "Sharing the region between programs"
   (sub-assignment 7): the client's rows are no longer let go here (main's form cleared each owned row's bit of
   MESH_ROW_OWN, the pages kept by MESH_ROW_HOT until a reclaim); nothing of a claim is let go until the bridge's
   release (mesh_retired_release).
   2026-11-04, sub-assignment 5: every claim the client owns is closed, where sub-assignment 7's 2026-11-03 form closed
   the first head it found, the client's one claim, taken at attach.  Why: a client holds a claim a plan, and a retired
   plan's claim stands beside the next plan's until the bridge releases it (design/disjoint-pages-change-request.md,
   Appendix C, C3.4).  What must stay: a retired claim's rows and pages held until that release.
   2026-11-04, sub-assignment 5, the coordinator's ruling on owner 7's finding (mesh 9d72a118's report): the retired
   client's plan is unpublished before its claims are closed, RCU's order (unpublish, then reclaim once no reader holds
   it; C3.4's deferred reclamation): once the client word is this retirer's, hdr `hops` 0 (release), a release fence
   and hdr `plan` woken, then every claim of the client closed, the client word cleared and woken as before.  Why: as
   the contract stood, the retired plan stayed published (hdr `hops`, `plan`, `state`) after its claims were closed, so
   the release could free pages the bridge parent's plan_cancel still raised counts in, and a port's link_run (`hops`
   and `client`) reloaded the stale plan for the next client, posting receives no peer would fill.  With `hops` 0
   nothing reaches the retired plan: plan_cancel returns at its first read, the ports end it (link_close) and clear
   `prepared`, and the release frees the claim on a later turn.  Nothing waits.
   2026-11-04, sub-assignment 5, the coordinator's ruling on review 15's finding 2: hdr `state` 0 beside `hops` 0, in
   the order `hops` 0, `state` 0 (each release), a release fence, `plan` woken, as mesh_plan's ESTALE path leaves them.
   Why: with `state` left naming the retired plan's state section, whose claim the bridge then releases, a start before
   the next plan passed mesh_transfers_start's `state` check and raised `executions` in a released page, and
   mesh_metal_transport_create would wrap released pages.  What must stay: `hops` 0 and `state` 0 before any claim is
   closed */
void mesh_retire(struct hdr *m,uint64_t client){
  uint64_t generation=(atomic_fetch_add_explicit(&m->serial,1,memory_order_relaxed)+1)&UINT64_C(0x7fffffff);
  uint64_t retiring=(client&(UINT64_C(1)<<63))|(generation<<32)|(uint32_t)getpid();
  if(!atomic_compare_exchange_strong_explicit(&m->client,&client,retiring,memory_order_seq_cst,memory_order_acquire))return;
  atomic_store_explicit(&m->hops,0,memory_order_release);
  atomic_store_explicit(&m->state,0,memory_order_release);
  atomic_thread_fence(memory_order_release);
  os_sync_wake_by_address_all(&m->plan,sizeof m->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  const uint32_t rows=mesh_rows(m);
  for(uint32_t row=mesh_bit_next(mesh_plane(m,MESH_ROW_OWN),0,rows,1);row<rows;){
    struct mesh_buffer *head=&mesh_buffers(m)[row];
    const uint64_t owner=atomic_load_explicit(&head->owner,memory_order_acquire);
    row=mesh_head_after(m,row,owner);
    if(owner==client)atomic_store_explicit(&head->closed,1,memory_order_release);
  }
  atomic_store_explicit(&m->client,0,memory_order_release);
  /* design/prepared-machine.md#M26 */
  mesh_control_notify(m);
  /* the ports' progress threads wait on hdr `plan` while idle (mesh-flow.c): woken, they read the retirement at once,
     not MESH_PAUSE_NS later */
  os_sync_wake_by_address_all(&m->plan,sizeof m->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}

/* bits [first, first + count) of `bits` set (count >= 1), word by word (atomic_fetch_or): 0 where every one was clear;
   1 where one was already set (another claim's, taken since the scan read it clear), the bits this call set cleared
   again and *busy that bit */
static int mesh_bits_take(_Atomic uint64_t *bits,uint32_t first,uint32_t count,uint32_t *busy){
  for(uint32_t w=first/64;w<=(first+count-1)/64;w++){
    const uint64_t mask=mesh_word_mask(first,count,w),was=atomic_fetch_or_explicit(&bits[w],mask,memory_order_acq_rel);
    if(!(was&mask))continue;
    atomic_fetch_and_explicit(&bits[w],~(mask&~was),memory_order_acq_rel);
    if(w>first/64)mesh_bits_clear(bits,first,w*64-first);
    *busy=w*64+(uint32_t)__builtin_ctzll(was&mask);
    return 1;
  }
  return 0;
}

/* First fit (D. E. Knuth, The Art of Computer Programming, vol. 1, 2.5): the lowest run of `count` (>= 1) clear bits of
   `bits` in [begin, end) whose first is a multiple of `align`, taken by setting its bits; its first, else `end` with
   nothing left set.  The scan only moves forward, past a set bit it reads and past one its taking finds set
   (mesh_bits_take), so it ends within one pass over [begin, end) */
static uint32_t mesh_run_take(_Atomic uint64_t *bits,uint32_t begin,uint32_t end,uint32_t count,uint32_t align){
  uint64_t at=begin;
  for(;;){
    at=(at+align-1)/align*align;
    if(at+count>end)return end;
    const uint32_t set=mesh_bit_next(bits,(uint32_t)at,(uint32_t)(at+count),1);
    uint32_t busy=set;
    if(set==at+count && !mesh_bits_take(bits,(uint32_t)at,count,&busy))return (uint32_t)at;
    at=mesh_bit_next(bits,busy+1,end,0);
  }
}

/* mesh-dataflow.h mesh_claim_take: the rows first, then the window run (mesh_arena_range(m, 1)), then the local run
   (mesh_arena_range(m, 0)), each first fit, a later one's failure clearing the earlier ones' bits again.  Where the
   window is the whole arena, mesh_arena_range's one range over it, the local run is first fit in that same range once
   the window run is taken, so the window run's set bits keep the two apart.  EINVAL where `rows` is 0 (no head) */
int mesh_claim_take(struct mesh_ctx *c,uint64_t generation,uint32_t rows,uint32_t window,uint32_t local,
                    struct mesh_claim *out){
  struct hdr *m=c->M;
  if(!rows)return EINVAL;
  _Atomic uint64_t *own=mesh_plane(m,MESH_ROW_OWN),*bits=mesh_arena_bits(m);
  const struct mesh_range wired=mesh_arena_range(m,1),outside=mesh_arena_range(m,0);
  const uint32_t wired_end=wired.first+wired.count,outside_end=outside.first+outside.count;
  struct mesh_claim claim={{0,rows},{0,window},{0,local}};
  if((claim.rows.first=mesh_run_take(own,0,mesh_rows(m),rows,1))==mesh_rows(m))return ENOMEM;
  if(window && (claim.window.first=mesh_run_take(bits,wired.first,wired_end,window,m->block))==wired_end){
    mesh_bits_clear(own,claim.rows.first,rows);
    return ENOMEM;
  }
  if(local && (claim.local.first=mesh_run_take(bits,outside.first,outside_end,local,m->block))==outside_end){
    if(window)mesh_bits_clear(bits,claim.window.first,window);
    mesh_bits_clear(own,claim.rows.first,rows);
    return ENOMEM;
  }
  struct mesh_buffer *head=&mesh_buffers(m)[claim.rows.first];
  struct mesh_page_entry *entry=&mesh_page(m)[claim.rows.first];
  atomic_store_explicit(&entry->mapping,(uint64_t)claim.window.count<<32|claim.window.first,memory_order_relaxed);
  atomic_store_explicit(&entry->address,(uintptr_t)((uint64_t)claim.local.count<<32|claim.local.first),memory_order_relaxed);
  atomic_store_explicit(&entry->device,generation,memory_order_relaxed);
  head->pages=rows;head->constant=0;
  atomic_store_explicit(&head->closed,0,memory_order_relaxed);
  atomic_store_explicit(&head->owner,c->client,memory_order_release);
  *out=claim;
  return 0;
}

/* mesh-dataflow.h mesh_claim_close: each head this client owns whose generation (its entry's `device`) is `generation`
   (others 0) or is not (others 1) closed (release) */
void mesh_claim_close(struct mesh_ctx *c,uint64_t generation,int others){
  struct hdr *m=c->M;
  const uint32_t rows=mesh_rows(m);
  for(uint32_t row=mesh_bit_next(mesh_plane(m,MESH_ROW_OWN),0,rows,1);row<rows;){
    struct mesh_buffer *head=&mesh_buffers(m)[row];
    const uint64_t owner=atomic_load_explicit(&head->owner,memory_order_acquire);
    const int same=atomic_load_explicit(&mesh_page(m)[row].device,memory_order_relaxed)==generation;
    row=mesh_head_after(m,row,owner);
    if(owner==c->client && (others?!same:same))atomic_store_explicit(&head->closed,1,memory_order_release);
  }
  os_sync_wake_by_address_all(&m->plan,sizeof m->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}

/* 2026-11-04, sub-assignment 5: the claim at attach (mesh_claim) and the release before it, waited for while a port's
   `prepared` held a plan, bounded by MESH_PLATFORM_NS (sub-assignment 7's 2026-11-03 form; review 12, finding 3), are
   gone: attach claims nothing, and a retired client's claim is released on the bridge's own turn, which nobody waits
   for.  Why: design/disjoint-pages-change-request.md, Appendix C, C3.4 and C3.6 ("A port that keeps a retired client's
   plan past MESH_PLATFORM_NS: gone with the claim at attach"), and C4.  What must stay: the client word's
   compare-and-swap, a gone client retired, the version read again after the swap */
int mesh_attach(struct mesh_ctx *c,const char *name){
  if(c->M) return 0;
  if(!name) name=getenv("MESH_REGION");
  if(!name) name=MESH_NAME;
  int file=shm_open(name,O_RDWR,MESH_MODE);
  if(file<0) return errno;
  struct stat info;
  if(fstat(file,&info)){ int error=errno; close(file); return error; }
  struct hdr *memory=mmap(NULL,(size_t)info.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,file,0);
  int error=errno;
  if(memory==MAP_FAILED){ close(file); return error; }
  if((size_t)info.st_size<sizeof *memory || memory->magic!=MESH_MAGIC || memory->version!=MESH_VERSION || memory->length>(uint64_t)info.st_size){
    munmap(memory,(size_t)info.st_size); close(file); return EINVAL;
  }
  uint64_t client=(((atomic_fetch_add_explicit(&memory->serial,1,memory_order_relaxed)+1)&UINT64_C(0x7fffffff))<<32)|(uint32_t)getpid(),vacant=0;
  while(!atomic_compare_exchange_strong_explicit(&memory->client,&vacant,client,memory_order_seq_cst,memory_order_acquire)){
    /* an earlier client whose process is gone (mesh-holders.h mesh_gone, main's kill/ESRCH test) is retired */
    if(!vacant || !mesh_gone((pid_t)(uint32_t)vacant)){ munmap(memory,(size_t)info.st_size); close(file); return EADDRINUSE; }
    mesh_retire(memory,vacant);
    vacant=0;
  }
  /* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_slot_take:284, the slot's start as hdr `client_start`: with
     `client`'s pid, this client's identity as a holder of the region (mesh-holders.h mesh_holds) */
  memory->client_start=mesh_started(getpid());
  uint64_t device=atomic_load_explicit(&memory->device_client,memory_order_seq_cst);
  client|=(~device)&(UINT64_C(1)<<63);
  atomic_store_explicit(&memory->client,client,memory_order_release);
  *c=(struct mesh_ctx){.M=memory,.len=(size_t)info.st_size,.client=client,.fd=file};
  /* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_attach: the expiry (a store, seq_cst) read again after the
     client's compare-exchange (seq_cst): a client the heartbeat's read of the holders missed reads it expired here,
     and is let go */
  if(__atomic_load_n(&memory->version,__ATOMIC_SEQ_CST)!=MESH_VERSION){mesh_detach(c);return EINVAL;}
  return 0;
}

int mesh_detach(struct mesh_ctx *c){
  if(!c->M) return 0;
  mesh_retire(c->M,c->client);
  int status=munmap(c->M,c->len);
  int error=status?errno:0;
  if(close(c->fd) && !error) error=errno;
  *c=(struct mesh_ctx){0};
  return error;
}

/* design/algorithm-sources.md#programtensor */
/* the plan's binding of a copy's rows to its pages (mesh-collective.h mesh_plan): no page is taken here, only named */
void mesh_backing_bind(struct mesh_ctx *c,uint32_t first,uint32_t pages,uint32_t page,uint32_t index){
  for(uint32_t offset=0;offset<pages;offset+=c->M->block){
    struct mesh_page_entry *entry=&mesh_page(c->M)[first+offset/c->M->block];
    atomic_store_explicit(&entry->mapping,((uint64_t)index<<32)|(page+offset),memory_order_relaxed);
    atomic_store_explicit(&entry->address,(uintptr_t)mesh_at(c->M,page+offset)-(uintptr_t)c->M,memory_order_relaxed);
  }
}

/* design/algorithm-sources.md#device-operands */
void mesh_device_bind(struct mesh_ctx *c,uint32_t row,uint64_t address){
  if(row!=MESH_ABSENT)atomic_store_explicit(&mesh_page(c->M)[row].device,address,memory_order_relaxed);
}

/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_holding_retained, adapted to the page lists: whether a closed
   claim's pages may still be named by a request a port's child posted (a receive not landed, a SEND not done).  At the
   tag, a stream's holding while its port child lived; here, a port whose `prepared` (struct mesh_port_info) holds the
   claim's generation: mesh-flow.c stores the plan's generation there once the port's data path loaded it and 0 once
   it stops serving it cleanly, and a port lost with requests outstanding keeps it until its child is reaped, where the
   parent clears it and sets `unreleased` (mesh-children.c child_reaped), so the bridge releases again.  Every port of
   the link table is read, not only those a plan's hops name: a port the map no longer lists may still hold it.
   A port in phase MESH_STOPPED retains too, though the review (R1, round 2) asked to skip it: a refusal (mesh-flow.c
   link_run) holds no request, but a port lost while it serves is MESH_STOPPED as well (mesh-ports.c port_lost: its
   child recycled, `open` 0), and its `prepared` stays the plan's while that child lives (mesh-flow.c streams_cancel),
   its receives posted until the reap; the region tells the two apart by nothing.  A refusal's `prepared` is cleared
   by link_close on its port's next pass once the plan or the client moves, which mesh_plan and mesh_retire wake (hdr
   `plan`), so a refused port keeps a closed claim only until then.
   2026-11-04, sub-assignment 5: the rule as given, a port's `prepared` equal to the claim's generation, where
   sub-assignment 7's 2026-11-03 form retained every closed claim while any port held any plan (main's header kept no
   generation of a retired client's plan).  Why: a claim's head records its generation (mesh_claim_take), and a
   retired plan's claim is released while the next plan is served (design/disjoint-pages-change-request.md, Appendix
   C, C3.4).  What must stay: no STOPPED port skipped while a loss sets it */
static int mesh_holding_retained(struct hdr *m,uint64_t generation){
  const uint32_t links=m->links>MESH_PORTS?m->links:MESH_PORTS;
  for(uint32_t p=0;p<links;p++)
    if(atomic_load_explicit(&mesh_links(m)[p].port.prepared,memory_order_acquire)==generation)return 1;
  return 0;
}

/* design/algorithm-sources.md#programtensor */
/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_retired_release, adapted: closed claims released, but none while
   a port's requests may name their pages (mesh_holding_retained); the count kept, which the tag's clients_turn keeps
   `unreleased` by.  Main's form released every closed row at once.  This is deferred reclamation (RCU's call_rcu: P. E.
   McKenney and J. D. Slingwine, Read-copy update, 1998; K. Fraser, Practical lock-freedom, 2004, an epoch announced and
   read again; design/disjoint-pages-revision-coordination.md, D10), on the bridge parent's one event-loop thread
   (mesh-records.c clients_turn, mesh-link.c region_adopt), and nobody waits for it.
   - A claim is released whole, from its head row (mesh_claim_take's record): its window and local runs' bits of the
     arena, then its rows' bits of MESH_ROW_OWN, last, so no later claim takes a row whose head fields are still being
     read.  The head's fields are read before its owner word is taken back: they are stored before the owner word and
     changed only by the release that takes it, so a release that loses the exchange reads nothing another claim wrote.
     The winner returns the head's entry to a free row's form (mesh-link.c: `mapping` MESH_ABSENT, `address` and
     `device` 0) and its `pages` and `closed` to 0 before the rows' bits are cleared.
   - The releaser's half of the ports' announcement (D10): a port stores its `prepared`, fences (seq_cst), and reads
     hdr `hops` and `plan` again, withdrawing where they moved; the releaser reads a claim closed (acquire: what its
     closer stored before, the plan's move or mesh_retire's `hops` 0, happens before), fences (seq_cst), and then reads
     the ports' `prepared`.  Of the two fences one precedes the other, so either the port reads the move and withdraws
     or the releaser reads its `prepared` and keeps the claim (C++20 [atomics.order] p4.4: a fence that happens before
     a load coherence-ordered before a store that happens before another fence precedes that fence).  One fence for
     each closed claim, after its acquire and before its ports are read, not one before the walk: a fence before that
     acquire orders nothing its closer stored.
   2026-11-04, sub-assignment 5: reversing sub-assignment 7's 2026-11-03 form, which released a retired client's one
   claim from two head rows (`pages` and `mapping` of each, the rows' count in the first's `mapping` high half), every
   claim retained while any port held any plan, and which a client's mesh_attach also called.  Why: claims per plan
   with their generation (design/disjoint-pages-change-request.md, Appendix C, C3.4), and the release the bridge
   parent's alone (D10; the coordinator's ruling on owner 7's finding).  What must stay: mesh_holding_retained's rule;
   the rows let go after the pages, and only by the exchange's winner */
uint32_t mesh_retired_release(struct hdr *m){
  _Atomic uint64_t *own=mesh_plane(m,MESH_ROW_OWN),*bits=mesh_arena_bits(m);
  const uint32_t rows=mesh_rows(m);
  uint32_t kept=0;
  for(uint32_t row=mesh_bit_next(own,0,rows,1);row<rows;){
    const uint32_t at=row;
    struct mesh_buffer *head=&mesh_buffers(m)[at];
    struct mesh_page_entry *entry=&mesh_page(m)[at];
    uint64_t owner=atomic_load_explicit(&head->owner,memory_order_acquire);
    row=mesh_head_after(m,at,owner);
    if(!owner || !atomic_load_explicit(&head->closed,memory_order_acquire))continue;
    atomic_thread_fence(memory_order_seq_cst);
    const uint64_t window=atomic_load_explicit(&entry->mapping,memory_order_relaxed),
                   local=(uint64_t)atomic_load_explicit(&entry->address,memory_order_relaxed),
                   generation=atomic_load_explicit(&entry->device,memory_order_relaxed);
    const uint32_t claimed=head->pages;
    if(mesh_holding_retained(m,generation)){kept++;continue;}
    if(!atomic_compare_exchange_strong_explicit(&head->owner,&owner,0,memory_order_acq_rel,memory_order_relaxed))continue;
    if(window>>32)mesh_bits_clear(bits,(uint32_t)window,(uint32_t)(window>>32));
    if(local>>32)mesh_bits_clear(bits,(uint32_t)local,(uint32_t)(local>>32));
    atomic_store_explicit(&entry->mapping,MESH_ABSENT,memory_order_relaxed);
    atomic_store_explicit(&entry->address,0,memory_order_relaxed);
    atomic_store_explicit(&entry->device,0,memory_order_relaxed);
    head->pages=0;
    atomic_store_explicit(&head->closed,0,memory_order_release);
    mesh_bits_clear(own,at,claimed);
  }
  return kept;
}

/* 2026-11-04, sub-assignment 5: mesh_publication_prepare is removed, which read a row's supplied words into the GPU's
   publication records (struct prepared_publication, gone with layout 109's MESH_SUPPLIED and MESH_WORD_STRIDE: mesh.h).
   Its one caller, mesh-metal.m, reads a range's first row's publication itself (its `targets[0]` the supplied word,
   mesh.h struct mesh_publication).  What must stay: one store releases a range */

/* The map reader and the client's start (mesh-dataflow.h mesh_nodes_read, mesh-call.h mesh_transfers_start).  The
   map's records, a cable's both ends and its ports stay in this file: the collective is given node pairs and summed
   capacity (struct mesh_nodes), and the start a status.  No graph algorithm is here (design/rubrics.md C9; the C9
   review of 7da3c125 asked these into mesh-graph.h): they read the map's record format into the node-pair weights
   mesh-graph.h's functions take as operands, and mesh-graph.h knows no cable.  The matching of two records' cables is
   one copy, mesh_cables, which the join (mesh_join: mesh_joined, the bridge's map in mesh-records.c), the reader's
   capacity and the bridges' rule (mesh-flow.c link_prepare) share. */

/* the cables both records list for each other (mesh.h struct mesh_reach: each record lists a cable from its own side,
   so b's entry names a's node with the two ports crossed), in `out` (MESH_PORTS of room: a record lists at most
   MESH_PORTS) sorted by key, the lower node id's port index, then the higher's; each one's bandwidth and queue pairs
   the lesser of the two records' (the pairing's queue pairs, the same at both ends); their count.  A cable one record
   lists alone (the other's record not yet heard) is none: both ends' rule reads a cable from both records */
uint32_t mesh_cables(const struct mesh_record *a,const struct mesh_record *b,struct mesh_cable *out){
  const int lower=a->node<b->node;
  uint32_t n=0;
  for(uint32_t i=0;i<a->reached && i<MESH_PORTS;i++){
    const struct mesh_reach *e=a->reach+i;
    if(e->node!=b->node)continue;
    for(uint32_t k=0;k<b->reached && k<MESH_PORTS;k++){
      const struct mesh_reach *f=b->reach+k;
      if(f->node!=a->node || f->port!=e->peer_port || f->peer_port!=e->port)continue;
      const struct mesh_cable cable={.key={lower?e->port:e->peer_port,lower?e->peer_port:e->port},
        .qps=e->qps<f->qps?e->qps:f->qps,.bandwidth=e->bandwidth<f->bandwidth?e->bandwidth:f->bandwidth};
      uint32_t at=n++;
      for(;at && (out[at-1].key[0]>cable.key[0] || (out[at-1].key[0]==cable.key[0] && out[at-1].key[1]>cable.key[1]));at--)
        out[at]=out[at-1];
      out[at]=cable;
      break;
    }
  }
  return n;
}

/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_join and mesh_joined, mesh_joined verbatim; mesh_join over
   mesh_cables, so the matching is one copy: its count the cables both records list for each other (the tag's, the
   fewer of the two ends' counts of entries naming the other, mesh_paired, which is gone), *bandwidth the least of
   theirs that either reports (0: none reports one) */
/* the one join of two nodes' records: the cables each one's bridge reaches the other over, 0 where none */
static uint32_t mesh_join(struct mesh_record *a,struct mesh_record *b,uint64_t *bandwidth){
  struct mesh_cable cables[MESH_PORTS];
  const uint32_t joined=mesh_cables(a,b,cables);
  uint64_t slowest=0;
  for(uint32_t c=0;c<joined;c++)if(cables[c].bandwidth && (!slowest || cables[c].bandwidth<slowest))slowest=cables[c].bandwidth;
  if(joined)*bandwidth=slowest;
  return joined;
}
int mesh_joined(struct mesh_record *a,struct mesh_record *b,double *beta){
  uint64_t bandwidth=0;
  const uint32_t joined=mesh_join(a,b,&bandwidth);
  if(joined)*beta=bandwidth?8e9/(double)bandwidth:0;
  return (int)joined;
}

/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_records_read, adapted: the map's records as one serial left them,
   the seqlock's reader (mesh.h mesh_map_whole), copied into r (MESH_NODES_MOST of room) with their count and the map's
   serial (hdr `map_serial`, moved only while `map` is odd); the read ended by `deadline` (the monotonic clock's ns),
   which its caller sets MESH_PLATFORM_NS past the start of its wait.  It differs from the tag's: no copy is allocated
   (the room is the reader's MESH_NODES_MOST, E2BIG past it); main has no expiry to register (mesh_expiry_set), so
   past the deadline it returns ETIMEDOUT, or ESRCH where the bridge's process is gone (its write left open); and an
   open write is waited on until the deadline, where the tag looked again each MESH_PAUSE_NS */
static int mesh_map_copy(struct mesh_ctx *c,uint64_t deadline,struct mesh_record *r,uint32_t *count,uint64_t *serial){
  struct hdr *m=c->M;
  for(;;){
    const uint64_t before=atomic_load_explicit(&m->map,memory_order_acquire),now=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    if(now>=deadline)return mesh_gone((pid_t)atomic_load_explicit(&m->bridge_pid,memory_order_acquire))?ESRCH:ETIMEDOUT;
    if(before&1){
      os_sync_wait_on_address_with_timeout(&m->map,before,sizeof m->map,OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                                           OS_CLOCK_MACH_ABSOLUTE_TIME,deadline-now);
      continue;
    }
    const uint64_t nodes=atomic_load_explicit(&m->map_nodes,memory_order_relaxed),at=atomic_load_explicit(&m->map_off,memory_order_relaxed);
    *serial=atomic_load_explicit(&m->map_serial,memory_order_relaxed);
    const int fits=at>=sizeof *m && at<=c->len && nodes<=(c->len-at)/sizeof *r;
    if(fits && nodes<=MESH_NODES_MOST)memcpy(r,(char *)m+at,nodes*sizeof *r);
    if(!mesh_map_whole(m,before))continue;
    if(!fits)return EINVAL;
    if(nodes>MESH_NODES_MOST)return E2BIG;
    *count=(uint32_t)nodes;return 0;
  }
}

/* the records' indices in ascending node-id order: order[i] the record of map index i */
static void mesh_map_order(const struct mesh_record *r,uint32_t count,uint32_t *order){
  for(uint32_t i=0;i<count;i++){
    uint32_t k=i;
    for(;k && r[order[k-1]].node>r[i].node;k--)order[k]=order[k-1];
    order[k]=i;
  }
}

/* a pair's capacity, the sum of its cables' bandwidths (mesh_cables), the rates the bridges' rule spreads its pages
   over; 0 where none joins them */
static uint64_t mesh_capacity(const struct mesh_record *a,const struct mesh_record *b){
  struct mesh_cable cables[MESH_PORTS];
  uint64_t sum=0;
  for(uint32_t c=mesh_cables(a,b,cables);c;c--)sum+=cables[c-1].bandwidth;
  return sum;
}

int mesh_nodes_read(struct mesh_ctx *c,struct mesh_nodes *out){
  struct mesh_record r[MESH_NODES_MOST];
  uint32_t count,order[MESH_NODES_MOST];
  uint64_t serial;
  int error=mesh_map_copy(c,clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS,r,&count,&serial);
  if(error)return error;
  mesh_map_order(r,count,order);
  *out=(struct mesh_nodes){.serial=serial,.count=count,.self=MESH_ABSENT};
  for(uint32_t i=0;i<count;i++){
    out->node[i]=r[order[i]].node;
    if(out->node[i]==c->M->node)out->self=i;
    for(uint32_t j=0;j<i;j++)out->capacity[i][j]=out->capacity[j][i]=mesh_capacity(r+order[i],r+order[j]);
  }
  return out->self==MESH_ABSENT?ENOENT:0;
}

/* design/prepared-machine.md#M06 */
/* design/algorithm-sources.md#programcopy */
/* The configured latch (hdr `configured`'s compare-exchange) and the unbounded spin on each link's `prepared` are gone:
   the plan is the bridge's to find (hdr `plan` its generation, `hops`), and each port whose cable carries it stores
   that generation in its `prepared`, last (release), once its data path has loaded it (code 0, phase MESH_PAIRED
   before it) or refused it (its `code`, phase MESH_STOPPED before it: mesh-flow.c link_run); a plan whose entries'
   serial (struct mesh_hop `serial`) is not the map's is stale (ESTALE).  The ports waited
   on are every port whose peer a hop names: the cables this node's own record lists (mesh-ports.c record_reach: a
   port whose link is up), in the map at the plan's serial, to the nodes the hops name (map indices, struct mesh_hop
   `node`).  One deadline, MESH_PLATFORM_NS from the start, bounds the map's read
   and the ports' wait together.  A spin, bounded: `prepared` is stored with no wake.  A port is done once its
   `prepared` is this generation (acquire), and only then are its phase and code read, which are this plan's (review
   R1, round 2: a phase read before it was an earlier refusal's, which a start after the next mesh_plan returned).  A
   refusal returns its code, ENOTCONN where that is 0, so that a stop never reads as a start; a port lost before it
   loads the plan never tells this generation, and the start ends at the bound with ETIMEDOUT.
   2026-11-04, sub-assignment 5: the start says how many executions the bridge serves (design/disjoint-pages-revision-
   coordination.md, D2: `executions` is no input of mesh_plan, design/disjoint-pages-change-request.md, Appendix C,
   C3.3): once the wait (mesh_ports_loaded) has found every named port's data path holding the plan, the plan's struct
   mesh_state `executions` is raised to it, never lowered (its one writer this call; C3.7, no state word reset while
   its plan stands), and hdr `plan` woken so the ports serve the rise.  EINVAL where no plan is written, or where hdr
   `state` names no struct mesh_state inside the region.
   2026-11-04, sub-assignment 5, the coordinator's ruling on owner 6's finding (mesh-flow.c 04b5e561): the raise after
   the wait, where this note's first form raised `executions` before it.  Why: a port child that pairs again after a
   loss loads the same plan and would serve executions from 0 again over phases and counts that already hold
   arrivals; so every port loads a plan before any of its executions is served, and a port that carries units of the
   plan refuses to load it where its `executions` is already above 0 (mesh-flow.c; a port with none posts nothing and
   loads it, since a refusal would read MESH_STOPPED to the next start: the coordinator's ruling on review 15's finding
   5), which holds only where the start raises it after every named port has loaded.  A second start on the same plan (its ports loaded and serving) waits as the first did and raises
   it.  What must stay: the wait bounded by one deadline; the raise only after every named port holds the plan */
static int mesh_ports_loaded(struct mesh_ctx *context){
  struct hdr *m=context->M;
  const uint64_t deadline=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;
  const uint64_t hops=atomic_load_explicit(&m->hops,memory_order_acquire),plan=atomic_load_explicit(&m->plan,memory_order_relaxed);
  if(!hops)return 0;
  if(hops>m->hop_capacity)return EINVAL;
  struct mesh_record r[MESH_NODES_MOST];
  uint32_t count,order[MESH_NODES_MOST];
  uint64_t serial;
  int error=mesh_map_copy(context,deadline,r,&count,&serial);
  if(error)return error;
  if(mesh_hops(m)[0].serial!=serial)return ESTALE;
  mesh_map_order(r,count,order);
  uint32_t self=count;
  for(uint32_t i=0;i<count;i++)if(r[i].node==m->node)self=i;
  if(self==count)return ENOENT;
  uint64_t named=0;
  for(uint64_t k=0;k<hops;k++){
    const uint32_t node=mesh_hops(m)[k].node;
    if(node>=count)return EINVAL;
    named|=UINT64_C(1)<<node;
  }
  uint32_t port[MESH_PORTS],ports=0;
  for(uint32_t i=0;i<r[self].reached && i<MESH_PORTS;i++)
    for(uint32_t j=0;j<count;j++)
      if(named>>j&1 && r[order[j]].node==r[self].reach[i].node){port[ports++]=r[self].reach[i].port;break;}
  const uint32_t links=m->links>MESH_PORTS?m->links:MESH_PORTS;
  for(uint32_t i=0;i<ports;i++){
    if(port[i]>=links)return EINVAL;
    struct mesh_port_info *info=&mesh_links(m)[port[i]].port;
    while(atomic_load_explicit(&info->prepared,memory_order_acquire)!=plan)
      if(clock_gettime_nsec_np(CLOCK_MONOTONIC)>=deadline)return ETIMEDOUT;
    if(atomic_load_explicit(&info->phase,memory_order_relaxed)==MESH_STOPPED){
      const int64_t code=atomic_load_explicit(&info->code,memory_order_relaxed);
      return code?(int)code:ENOTCONN;
    }
  }
  return 0;
}
/* 2026-11-04, sub-assignment 1 (design/disjoint-pages-change-request.md, D.3.6; design/disjoint-pages-rulings-
   coordination.md, R3): after the raise of the plan's `executions` and the wake, a seq_cst fence, then hdr
   `map_serial` read; where it is not the plan's entries' serial (struct mesh_hop `serial`, the map the plan was made
   from; read only where hdr `hops` is above 0), ESTALE, the raise standing.  Why: a plan stands only while its map
   does.  Where the map moves, the bridge moves the serial, fences (seq_cst) and reads the plan's `executions`, then
   cancels every receive of the plan not landed for the executions below it, each an arrival on its count with no phase
   written (mesh-flow.c plan_stale); a raise after that read would leave the raised executions uncancelled, and their
   readers would end at their deadline (ETIMEDOUT), not at once.  The start raises, fences and reads the serial: of the
   two fences one comes first, and a load after the later one sees the store before the earlier one, so at least one
   side sees the other.  Either the cancellation covers the raised executions, or the start reads the moved serial and
   returns ESTALE, and the program runs none of them and plans again on the map as it is.  The raise left standing
   reads no execution as landed that has not: `executions` only rises (C3.7, no state word set back while its plan
   stands), the program reads and releases nothing of an execution it does not run, and the plan's end (the next
   plan, or the program's retirement) cancels every receive of it not landed.  That is its safety, not that nothing
   follows: the bridge has already set the raised executions' receives, which end with the plan as any cancelled
   run's do, by the bridge's own means (review 8's finding 1; review 10's finding 5).  The pairing is D10's
   announcement, fence and re-read
   (design/disjoint-pages-revision-coordination.md; K. Fraser, Practical lock-freedom, 2004), and the rule is the one
   mesh_ports_loaded applies before its wait (its ESTALE), applied after the raise; nobody waits.  The fence after the
   wake, so every raise is woken as before.  What must stay: the raise before the fence, the serial's read after it */
int mesh_transfers_start(struct mesh_ctx *context,uint64_t executions){
  struct hdr *m=context->M;
  const uint64_t at=atomic_load_explicit(&m->state,memory_order_acquire);
  if(!at || at>context->len || context->len-at<sizeof(struct mesh_state))return EINVAL;
  const int error=mesh_ports_loaded(context);
  if(error)return error;
  struct mesh_state *state=mesh_plan_state(m);
  uint64_t served=atomic_load_explicit(&state->executions,memory_order_relaxed);
  while(served<executions && !atomic_compare_exchange_strong_explicit(&state->executions,&served,executions,
                                                                      memory_order_release,memory_order_relaxed));
  os_sync_wake_by_address_all(&m->plan,sizeof m->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
  atomic_thread_fence(memory_order_seq_cst);
  const uint64_t serial=atomic_load_explicit(&m->map_serial,memory_order_relaxed);
  const uint64_t hops=atomic_load_explicit(&m->hops,memory_order_acquire);
  return hops && mesh_hops(m)[0].serial!=serial?ESTALE:0;
}

/* 2026-11-04, sub-assignment 5: mesh_plan_erased is removed (sub-assignment 7's 2026-11-03 wait, bounded by
   MESH_PLATFORM_NS, until no port held the previous plan whose supplied words and landed records the next plan zeroed
   and rewrote in place, and its seq_cst fence, Dekker's pair with the ports' data path).  Why: a new plan's state is
   in new pages, never the previous plan's, so nothing erases a plan or waits for one to be erased
   (design/disjoint-pages-change-request.md, Appendix C, C3.4, C3.6 and C4).  What must stay: a port's announcement of
   the generation it loads, which the release reads (mesh_retired_release; design/disjoint-pages-revision-
   coordination.md, D10) */
