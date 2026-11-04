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
/* 2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 "Sharing the region between programs"
   (sub-assignment 7): the client's rows are no longer let go here (main's form cleared each owned row's bit of
   MESH_ROW_OWN, the pages kept by MESH_ROW_HOT until a reclaim).  The client's claim is closed whole: its head row
   (mesh_claim), the one row its owner word names, and nothing of it is let go until the bridge's release
   (mesh_retired_release).  What must stay: a retired claim's rows and pages held until that release */
void mesh_retire(struct hdr *m,uint64_t client){
  uint64_t generation=(atomic_fetch_add_explicit(&m->serial,1,memory_order_relaxed)+1)&UINT64_C(0x7fffffff);
  uint64_t retiring=(client&(UINT64_C(1)<<63))|(generation<<32)|(uint32_t)getpid();
  if(!atomic_compare_exchange_strong_explicit(&m->client,&client,retiring,memory_order_seq_cst,memory_order_acquire))return;
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *buffer=&mesh_buffers(m)[row];
    if(atomic_load_explicit(&buffer->owner,memory_order_acquire)==client){
      atomic_store_explicit(&buffer->closed,1,memory_order_release);
      break;
    }
  }
  atomic_store_explicit(&m->client,0,memory_order_release);
  /* design/prepared-machine.md#M26 */
  mesh_control_notify(m);
  /* the ports' progress threads wait on hdr `plan` while idle (mesh-flow.c): woken, they read the retirement at once,
     not MESH_PAUSE_NS later */
  os_sync_wake_by_address_all(&m->plan,sizeof m->plan,OS_SYNC_WAKE_BY_ADDRESS_SHARED);
}

/* The program's claim (design/disjoint-pages-change-request.md, Appendix A, A3 "Sharing the region between programs";
   A4, "One claim of a program's section at attach remains"; mesh-dataflow.h struct mesh_ctx), in place of the
   call-site allocators (A2: first fit at the call that needs memory), which went with mesh_rows_alloc and
   mesh_arena_alloc: one client holds the region at a time (hdr `client`), so its claim takes every row and page free,
   the widest run of each, and excludes what a retired client's claim still holds (its bits set until
   mesh_retired_release).  Nothing else takes rows or pages from the region: the bridge's own areas are laid out with it
   (mesh-map.h mesh_layout). */

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

/* the widest run of clear bits in [begin, end), its first and its count multiples of `align`; {0, 0} where none */
static struct mesh_range mesh_bit_widest(_Atomic uint64_t *bits,uint32_t begin,uint32_t end,uint32_t align){
  struct mesh_range widest={0,0};
  for(uint32_t at=begin;at<end;){
    const uint32_t clear=mesh_bit_next(bits,at,end,0),set=mesh_bit_next(bits,clear,end,1);
    const uint64_t first=((uint64_t)clear+align-1)/align*align,last=(uint64_t)set/align*align;
    if(last>first && last-first>widest.count)widest=(struct mesh_range){(uint32_t)first,(uint32_t)(last-first)};
    at=set;
  }
  return widest;
}

/* The claim, recorded in its first two rows, its heads, so the bridge's release finds it after a later client has
   attached (mesh_retired_release):
   - row `rows.first`: its buffer's `owner` the client (`closed` once it retires) and `pages` the window run's pages; its
     entry's `mapping` the window run's first page (low half; MESH_ABSENT where the run is empty) and the claim's rows
     (high half);
   - row `rows.first` + 1: its buffer's `pages` the local run's pages, its entry's `mapping` the local run's first page
     (MESH_ABSENT where empty); no owner.
   The claim's rows are held by their bits of MESH_ROW_OWN and its pages by their bits of the arena's; no other row
   holds an owner word or `pages`.  The window's run is the widest free block-aligned run inside the registered window
   (mesh_arena_range(m, 1)), the local run the widest outside it (mesh_arena_range(m, 0)).  Where the window is the
   whole arena, mesh_arena_range's one range over it, the claim's one run is halved, `local` its upper half: the change
   request says nothing of that case, and this is its one run split so neither kind of section is refused outright.
   0; ENOMEM where fewer than two rows are free (nothing is taken) */
static int mesh_claim(struct mesh_ctx *c){
  struct hdr *m=c->M;
  const uint32_t block=m->block;
  _Atomic uint64_t *own=mesh_plane(m,MESH_ROW_OWN),*bits=mesh_arena_bits(m);
  const struct mesh_range rows=mesh_bit_widest(own,0,mesh_rows(m),1);
  if(rows.count<2)return ENOMEM;
  mesh_bits_set(own,rows.first,rows.count);
  const struct mesh_range wired=mesh_arena_range(m,1),outside=mesh_arena_range(m,0);
  struct mesh_range window=mesh_bit_widest(bits,wired.first,wired.first+wired.count,block),local;
  if(!outside.first){
    const uint32_t half=window.count/block/2*block;
    window.count-=half;
    local=half?(struct mesh_range){window.first+window.count,half}:(struct mesh_range){0,0};
  }else local=mesh_bit_widest(bits,outside.first,outside.first+outside.count,block);
  if(window.count)mesh_bits_set(bits,window.first,window.count);
  if(local.count)mesh_bits_set(bits,local.first,local.count);
  struct mesh_buffer *head=&mesh_buffers(m)[rows.first],*second=head+1;
  atomic_store_explicit(&mesh_page(m)[rows.first+1].mapping,local.count?local.first:MESH_ABSENT,memory_order_relaxed);
  second->pages=local.count;second->constant=0;
  atomic_store_explicit(&second->closed,0,memory_order_relaxed);
  atomic_store_explicit(&mesh_page(m)[rows.first].mapping,
    (uint64_t)rows.count<<32|(window.count?window.first:MESH_ABSENT),memory_order_relaxed);
  head->pages=window.count;head->constant=0;
  atomic_store_explicit(&head->closed,0,memory_order_relaxed);
  atomic_store_explicit(&head->owner,c->client,memory_order_release);
  c->rows=rows;c->window=window;c->local=local;
  c->row_cursor=rows.first+2;c->local_cursor=local.first;
  return 0;
}

static int mesh_holding_retained(struct hdr *);

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
  /* A retired claim is released before this one is taken, as the bridge's turn would release it (mesh-records.c
     clients_turn), by the same function and its rule (mesh_holding_retained): a claim takes every free row and page, so
     a client attached just after another let go (back-to-back runs, the hourly) would otherwise find that one's claim
     still held, fewer than two rows free, and ENOMEM.  While a port's `prepared` still holds a plan (this client has
     made none yet, so it is the retired client's, or a lost port's until its child is reaped) the claim is kept; the
     ports let it go on their next pass once the client word changes (mesh_retire wakes them on hdr `plan`).  So a wait,
     bounded by MESH_PLATFORM_NS from its start (review 12, finding 3): a spin, as mesh_transfers_start's (`prepared` is
     stored with no wake), on the ports alone, the release called again once none holds a plan.  Past the bound the
     claim takes what is free, ENOMEM where that is fewer than two rows: the retired claim stays held while a port's
     requests may name its pages */
  const uint64_t deadline=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;
  while(mesh_retired_release(memory) && clock_gettime_nsec_np(CLOCK_MONOTONIC)<deadline)
    while(mesh_holding_retained(memory) && clock_gettime_nsec_np(CLOCK_MONOTONIC)<deadline);
  const int claimed=mesh_claim(c);
  if(claimed){mesh_detach(c);return claimed;}
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
/* the plan's binding of an instance's rows to its pages (mesh-collective.h mesh_plan), and mesh_section_create's of a
   local section's: no page is taken here, only named */
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

/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_holding_retained, adapted to the page lists: whether a retired
   client's pages may still be named by a request a port's child posted (a receive not landed, a SEND not done).  At
   the tag, a stream's holding while its port child lived; here, a port whose `prepared` holds a plan (struct
   mesh_port_info): mesh-flow.c stores the plan's generation there once the port's data path loaded it and 0 once it
   stops serving it cleanly, and a port lost with requests outstanding keeps it until its child is reaped, where the
   parent clears it and sets `unreleased` (mesh-children.c child_reaped), so the bridge releases again.  The rule as
   given is a port's `prepared` equal to the retired client's plan (hdr `plan` as it retired); main's header keeps
   no generation of a retired client's plan, and hdr `plan` is the next client's once it plans, by when a port lost
   with requests outstanding may still hold the earlier plan's receives.  So any port holding a plan retains: the
   rule's every case, and also a later client's plan served while an earlier one's rows are closed.
   A port in phase MESH_STOPPED retains too, though the review (R1, round 2) asked to skip it: a refusal (mesh-flow.c
   link_run) holds no request, but a port lost while it serves is MESH_STOPPED as well (mesh-ports.c port_lost: its
   child recycled, `open` 0), and its `prepared` stays the plan's while that child lives (mesh-flow.c streams_cancel),
   its receives posted until the reap; the region tells the two apart by nothing.  A refusal's `prepared` is cleared
   by link_close on its port's next pass once the client changes, which mesh_retire wakes (hdr `plan`), so a refused
   port keeps a retired client's rows only until then; what must stay: no STOPPED port skipped while a loss sets it */
static int mesh_holding_retained(struct hdr *m){
  const uint32_t links=m->links>MESH_PORTS?m->links:MESH_PORTS;
  for(uint32_t p=0;p<links;p++)if(atomic_load_explicit(&mesh_links(m)[p].port.prepared,memory_order_acquire))return 1;
  return 0;
}

/* design/algorithm-sources.md#programtensor */
/* tag pre-return-20261103 rdma/mesh-dataflow.c mesh_retired_release, adapted: a retired client's closed rows
   released, but none while a port's requests may name their pages (mesh_holding_retained); the count kept, which the
   tag's clients_turn keeps `unreleased` by.  Main's form released every closed row at once.
   2026-11-03, design/disjoint-pages-change-request.md, Appendix A, A3 "Sharing the region between programs"
   (sub-assignment 7): what is released is a retired client's claim (mesh_claim), whole, from its head row: its two
   page runs' bits of the arena, then its rows' bits of MESH_ROW_OWN, last, so no later claim takes a row whose head
   fields are still being read.  Main's form marked each closed row MESH_FREE for the next allocating client's reclaim
   (mesh_reclaim, gone with the call-site allocators).  The head's fields are read before its owner word is taken back:
   they are stored before the owner (mesh_claim) and changed only by the release that takes it, so a release that
   loses the exchange (the bridge's and an attaching client's may run at once) reads nothing another claim wrote.
   A head is never the table's last row, and its rows lie in the table: mesh_claim takes two or more of them.  The count
   returned is of the claims kept.  What must stay: mesh_holding_retained's rule; the rows let go after the pages, and
   only by the exchange's winner */
uint32_t mesh_retired_release(struct hdr *m){
  const int retained=mesh_holding_retained(m);
  uint32_t kept=0;
  for(uint32_t row=0;row<mesh_rows(m);row++){
    struct mesh_buffer *head=&mesh_buffers(m)[row];
    uint64_t owner=atomic_load_explicit(&head->owner,memory_order_acquire);
    if(!owner || !atomic_load_explicit(&head->closed,memory_order_acquire))continue;
    if(retained){kept++;continue;}
    struct mesh_buffer *second=head+1;
    struct mesh_page_entry *entry=&mesh_page(m)[row];
    const uint64_t window=atomic_load_explicit(&entry->mapping,memory_order_relaxed),
                   local=atomic_load_explicit(&entry[1].mapping,memory_order_relaxed);
    const uint32_t window_pages=head->pages,local_pages=second->pages,rows=(uint32_t)(window>>32);
    if(!atomic_compare_exchange_strong_explicit(&head->owner,&owner,0,memory_order_acq_rel,memory_order_relaxed))continue;
    if(window_pages)mesh_bits_clear(mesh_arena_bits(m),(uint32_t)window,window_pages);
    if(local_pages)mesh_bits_clear(mesh_arena_bits(m),(uint32_t)local,local_pages);
    head->pages=second->pages=0;
    atomic_store_explicit(&entry->mapping,MESH_ABSENT,memory_order_relaxed);
    atomic_store_explicit(&entry[1].mapping,MESH_ABSENT,memory_order_relaxed);
    atomic_store_explicit(&head->closed,0,memory_order_release);
    mesh_bits_clear(mesh_plane(m,MESH_ROW_OWN),row,rows);
  }
  return kept;
}

/* design/prepared-machine.md#M13 */
/* design/prepared-machine.md#M04 */
/* design/algorithm-sources.md#programkernel_call */
/* A row's supplied words as mesh_plan records them (design/disjoint-pages-coordination.md, "Rows' publications"):
   `targets[i].stream` the region offset of a range's supplied word of execution 0 (mesh.h mesh_supplied; count 1,
   stride 1) and the publication's `argument` MESH_SUPPLIED, stored there at execution t's word, `stream` + t x
   MESH_WORD_STRIDE, by the GPU or mesh_host_publish.  A target's word k is `stream` + MESH_WORD_STRIDE x `stride` x k.
   2026-11-03, design/disjoint-pages-change-request.md, "Claude's proposal: the canonical path", item 3
   (sub-assignment 7): no 256-byte cell (sizeof(struct mesh_send), hdr `hop_stride`) a word; a supplied word is one
   word, MESH_WORD_STRIDE apart.  What must stay: one store releases a range */
uint32_t mesh_publication_prepare(struct hdr *m,uint32_t row,struct prepared_publication *records){
  struct mesh_publication *publication=mesh_publication_at(m,row);
  uint32_t count=0;
  for(uint32_t i=0;i<publication->sends;i++){
    struct mesh_target target=publication->targets[i];
    for(uint32_t k=0;k<target.count;k++){
      if(records){
        char *word=(char *)m+target.stream+(uint64_t)MESH_WORD_STRIDE*target.stride*k;
        records[count]=(struct prepared_publication){.destination=(uintptr_t)word,
          .argument=atomic_load_explicit(&publication->argument,memory_order_relaxed)};
      }
      count++;
    }
  }
  return count;
}

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
   loads the plan never tells this generation, and the start ends at the bound with ETIMEDOUT */
int mesh_transfers_start(struct mesh_ctx *context){
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

/* mesh-call.h mesh_plan_erased (design/disjoint-pages-change-request.md, Appendix A, A3; design/
   disjoint-pages-coordination.md, "Writing a plan"): mesh_plan, having stored hdr `hops` 0 and woken `plan`, waits here
   until no port's data path holds the previous plan, whose supplied words and landed records it is about to zero and
   rewrite: each port's `prepared` leaves `generation` once its data path has let the plan go (mesh-flow.c link_close),
   and never comes back to it, `hops` being 0.  It replaces the retention of the plan's words section (the context's
   `send_off`, `send_bytes`, gone), the words being fixed areas of the layout.  Every port of the link table is read
   (mesh_holding_retained's bound), not only those a hop names: a port the map no longer lists may still hold it.  A
   spin, bounded by one deadline, MESH_PLATFORM_NS from the start, as mesh_transfers_start's: `prepared` is stored with
   no wake */
int mesh_plan_erased(struct mesh_ctx *context,uint64_t generation){
  if(!generation)return 0;
  /* Dekker's pair with a port's data path (review 12, finding 1): the caller has stored hdr `hops` 0, and a port stores
     its `prepared` before its last read of `hops`, a seq_cst fence between (mesh-flow.c).  With this fence between the
     caller's store and the reads below, either the port reads `hops` 0 and withdraws, or this wait reads its
     `prepared`; without it a port could begin serving the erased plan after this returned */
  atomic_thread_fence(memory_order_seq_cst);
  struct hdr *m=context->M;
  const uint64_t deadline=clock_gettime_nsec_np(CLOCK_MONOTONIC)+MESH_PLATFORM_NS;
  const uint32_t links=m->links>MESH_PORTS?m->links:MESH_PORTS;
  for(uint32_t p=0;p<links;p++)
    while(atomic_load_explicit(&mesh_links(m)[p].port.prepared,memory_order_acquire)==generation)
      if(clock_gettime_nsec_np(CLOCK_MONOTONIC)>=deadline)return ETIMEDOUT;
  return 0;
}
