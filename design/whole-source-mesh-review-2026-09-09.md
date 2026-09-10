# Whole-source review of mesh algorithms, September 9, 2026

The operator requested the complete algorithm source inline with dispositions,
before any further incremental implementation. This document reproduces the
reviewed files without omissions, with original line numbers and annotations.
It is a review exhibit, not executable replacement source or a source backup.
No algorithm implementation was changed for this review.

## Authority and scope

The operator's instructions and `pages-and-functions.md` govern. Conflicting
implementation descriptions in `distributed-reduce.md` and
`algorithm-sources.md` do not authorize exceptions. In particular,
`distributed-reduce.md` still authorizes progress/producible counters, received
publication bitmaps, digest-based storage dependencies, task/stage terminology,
and materialized FP16 reduction. Those passages conflict with the user's
mandates and the plain specification; this review rejects those mechanisms.

The corpus includes the complete active page runtime and its header, the entire
NFE caller (including configuration, shaders, and reporting), Metal page bindings,
wire adapter, complete shared client/header, and the alternative function, reduction, Metal executor and tensor
algorithm interfaces/implementations still built by rdma/Makefile. These latter
implementations are not linked into metal-microbench's measured mesh-forward.
Bridge/verbs implementation and generic local numerical-kernel internals are
outside this exhibit; their inclusion in the program is not a claim that they
have passed this review. This is a complete listing of the explicitly enumerated
algorithm corpus, not a claim to reproduce every dependency in either repository.

`R` means the operation is required, not that its exact spelling is mandatory.
`A` means allowed configuration, arithmetic or literal-page access.
`X` means forbidden algorithm/control/storage representation.
`D` means redundant representation to delete with its users.
`M` marks a block containing different dispositions, identified in its annotation.
`B` means backend/transport mechanics whose hardware obligations must be preserved,
without elevating their events into a second numerical readiness system.
`O` means observation or measurement outside the algorithm.
Syntax, includes and blank lines inherit their containing block's disposition.
A mixed block is not approval of all its lines.

## Data and execution flows

Actual active data flow:

```
foreign weight/token/constant buffers + hidden pages
    -> configured local numerical function
    -> partial pages
    -> CPU FP32 register sums -> FP16 reduced pages -> send
    -> separate normalization/residual function -> hidden pages
```

Actual active execution flow:

```
embedding stamp selects current generation
    -> producible limit + operand stamps + destination stamps + owner array
    -> command completion -> publication bitmap
    -> progress thread drains bitmap -> actual output stamp + pending bitmap
    -> transport submission
    -> fill/highest/complete counters -> next producible limit
endpoint verdict completion -> permission to submit next embedding
```

The required data flow is partial pages -> accumulator pages and index pages ->
normalized output pages -> peer/local next function. The required execution flow
is a static configured function list scanning literal operand/destination rows;
completion directly publishes output stamps after writes are visible. Release
follows the actual output rows proving every dependent read, with use counts in
the page table. No second readiness representation participates.

## Allowed vocabulary and obligations

The only mutable algorithm storage is the canonical page table (physical page,
stamp, use count), its actual RDMA-sendable pages (including accumulator, index,
digest, input and result values), and the specified link status word. Immutable
configuration names numerical functions, their input/output row maps, participant
ownership and numerical layout. Loop indices and arithmetic registers are not
persistent algorithm state. Backend descriptors cannot become operand stores or
an independently indexed completion system. A multi-page API mapping is not a
new physical allocation, but its use as an independent buffer abstraction is not
automatically authorized by being a no-copy alias.

The existing acquire/release atomic stamp loads/stores implement the required
visibility boundary. An issued encoding in that same destination stamp is
consistent with the documented destination-claim obligation, provided it is
never read as a completed generation. It requires an explicit single-writer
proof for overlapping outputs. It does not authorize a separate lock, token,
owner array or issued bitmap. Completion must not stamp a failed write as valid.

The algorithm can be stated without introducing a replacement helper API:

```
For each configured function and NFE k represented by its configured rows:
  Select independent outputs whose actual input rows all have stamp k,
  and whose actual destination pages have been released and are available.
  Claim only those destination rows, using their own table state.
  Execute the configured function for the selected rows together.
  On completed visible writes, put k in those output rows' stamps.

For each matching pair of partial pages p at k:
  Add their elements in high precision into the configured accumulator pages.
  Record p in the configured index page after that addition is complete.
  When the index page proves a complete numerical row:
    Normalize that row into the configured output pages; stamp them k.
  Send required output pages directly through the existing RDMA binding.

For each input page:
  Observe the actual downstream output rows that prove each required read.
  Release each completed use exactly once; decrement the row's use count.
  At zero, zero asynchronously before making the physical page reusable.
  Return received pages to the bridge.

After consumption:
  Hash the required still-valid page contents into digest pages.
  Compare when both digest pages are present.
  A mismatch is an NFE failure value; the caller repeats that evaluation.
  Neither digest arrival nor comparison authorizes numerical execution/reuse.

On negative link status:
  Conclude every in-flight NFE with failure, recover the program, and repeat.
```

This is specification notation, not pretend compilable code. In particular it
does not hide implementations behind invented `ready`, `schedule`, `retire` or
`recover_all` helpers. The replacement must provide exact-once claims, exact-once
use decrements, GPU/CPU/NIC visibility, hashing-before-zeroing, and all dependent
read proofs within that representation. A repeated scan cannot decrement the
same use repeatedly. Deleting the old tracking arrays without realizing these
obligations would be another incorrect implementation.

## Authors and what they establish

- Gregory M. Papadopoulos and David E. Culler, *Monsoon: An Explicit Token-Store
  Architecture*, ISCA 1990, sections 2–3: storage-associated operand matching.
  Its own frames and token queues are not permission to add them here.
- Rolf Rabenseifner, *Optimization of Collective Reduction Operations*, ICCS 2004;
  Pitch Patarasuk and Xin Yuan, *Bandwidth Optimal All-reduce Algorithms for
  Clusters of Workstations*, JPDC 2009: collective reduction and communication
  bounds. They do not justify moving normalization after the specified send.
- Jerome H. Saltzer, David P. Reed and David D. Clark, *End-to-End Arguments in
  System Design*, TOCS 1984: endpoint checking. The specific requirement that a
  digest never gate numerical progress comes from this repository's mandate.
- PyTorch authors, *Introducing Async Tensor Parallelism in PyTorch*, 2024:
  computation/communication overlap; no universal Metal speedup guarantee.

Primary-source links and scoped discussion are in [algorithm-sources.md](algorithm-sources.md).
None of these citations turns an unimplemented obligation into a completed one.
The latest full-graph measurement remains around 1.6 seconds per NFE; that is an
end-to-end result, not measured RDMA latency. Numerical agreement with an earlier
implementation does not establish conformance to the required reduction order.

## Deletion boundaries

Delete the alternative call/argument state machine, token reduction state
machine, event-based selection and phase/arena-copy tensor algorithm as algorithm
representations, including their interfaces and build/caller dependencies.
Preserve supported numerical behavior by expressing it over the allowed rows;
renaming their structs or moving them into canonical mesh is not a replacement.
In the active runtime delete duplicate publication/progress/reuse/digest state
and replace materialized reduction with the specified page arithmetic as one
coherent change. The caller must cease deriving invocation readiness from
endpoint records, verdicts or a single current-generation cursor. Page-backed
weights/constants/inputs and all lifetime proofs are part of that same change.

No deletion is claimed by this document. It identifies the complete replacement
boundary; it is not another measurement-driven series of local optimizations.

## Annotated source

Reviewed mesh commit: `26b78394b1bc5dc2f01b6179fde58ca2d6eced95`.

Reviewed metal-microbench commit: `91346a54139e3297adc711ed98b97f65e86533bd`.

Corpus: **16 complete files, 3,970 source lines**. Every line belongs to an annotated range.

### rdma/mesh-pages.h — 59 lines

**A, lines 1–6.** Header and local-peer constant.

```text
   1 | #ifndef MESH_PAGES_H
   2 | #define MESH_PAGES_H
   3 | #include "mesh.h"
   4 | 
   5 | #define MESH_PAGES_LOCAL UINT16_MAX
   6 | #define MESH_PAGES_DEPENDENCIES 8
```

**M, lines 7–16.** Static row maps/function descriptions are allowed. Slot pagewise/lag/storage machinery must not manufacture readiness; reduce kind/group encode the divergent materialization. Fault policy is observation input. Hook is a redundant progress notification.

```text
   7 | struct mesh_pages_slot { uint32_t sid, pages; uint16_t peer; uint8_t receive, depends, pagewise; uint32_t dependency[MESH_PAGES_DEPENDENCIES], lag[MESH_PAGES_DEPENDENCIES]; uint32_t storage; };
   8 | struct mesh_pages_policy { uint64_t fault_seed; uint32_t fault_period; };
   9 | #define MESH_REDUCE_MATERIALIZED 0
  10 | #define MESH_REDUCE_PARTIAL 1
  11 | struct mesh_pages_reduce { uint32_t output; uint32_t input[MESH_PAGES_DEPENDENCIES]; uint8_t inputs, kind; uint32_t group, offset, bytes; };
  12 | typedef struct mesh_pages mesh_pages;
  13 | typedef struct mesh_pages_function mesh_pages_function;
  14 | struct mesh_pages_map { uint32_t slot, first, count, stride; uint64_t lag; };
  15 | struct mesh_pages_function_spec { const struct mesh_pages_map *input, *output; uint32_t inputs, outputs, rows; };
  16 | typedef void (*mesh_pages_hook)(void *capture, uint32_t slot, uint64_t generation);
```

**M, lines 17–59.** Retain configuration, literal row/page access, direct completion, release, status/recovery concepts. Remove filled/highest/producible/hook and heap-digest APIs with their state. Claim/cancel must operate solely on real row state. Diagnostics are O, not dependency APIs.

```text
  17 | 
  18 | mesh_pages *mesh_pages_compile(struct mesh_ctx *context, struct mesh_epoch epoch, const unsigned char plan[32],
  19 |   const struct mesh_pages_slot *slots, size_t count, uint32_t versions, struct mesh_pages_policy policy);
  20 | int mesh_pages_reduces(mesh_pages *p, const struct mesh_pages_reduce *reduces, size_t count);
  21 | int mesh_pages_start(mesh_pages *p);
  22 | void mesh_pages_stop(mesh_pages *p);
  23 | int mesh_pages_progress(mesh_pages *p);
  24 | int mesh_pages_free(mesh_pages *p);
  25 | 
  26 | size_t mesh_pages_header(const mesh_pages *p);
  27 | size_t mesh_pages_payload(const mesh_pages *p);
  28 | const uint32_t *mesh_pages_entries(const mesh_pages *p, uint32_t slot);
  29 | const uint32_t *mesh_pages_table(const mesh_pages *p, size_t *bytes);
  30 | const uint64_t *mesh_pages_stamps(const mesh_pages *p, uint32_t slot);
  31 | static inline uint64_t mesh_pages_stamp(const uint64_t *stamps, uint32_t page){ return __atomic_load_n(stamps+page,__ATOMIC_ACQUIRE); }
  32 | static inline uint32_t mesh_pages_entry(const uint32_t *entries, uint32_t page){ return __atomic_load_n(entries+page,__ATOMIC_ACQUIRE); }
  33 | static inline void mesh_pages_store(uint64_t *word, uint64_t value){ __atomic_store_n(word,value,__ATOMIC_RELEASE); }
  34 | static inline uint64_t mesh_pages_load(const uint64_t *word){ return __atomic_load_n(word,__ATOMIC_ACQUIRE); }
  35 | void *mesh_pages_data(const mesh_pages *p, uint32_t slot, uint32_t page);
  36 | mesh_pages_function *mesh_pages_bind(mesh_pages *p, struct mesh_pages_function_spec spec);
  37 | size_t mesh_pages_scan(mesh_pages_function *f, uint64_t generation, const uint32_t **indices);
  38 | int mesh_pages_complete(mesh_pages_function *f, uint32_t row, uint64_t generation);
  39 | int mesh_pages_claim(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation);
  40 | void mesh_pages_cancel(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation);
  41 | size_t mesh_pages_writing(const mesh_pages *p);
  42 | uint32_t mesh_pages_filled(const mesh_pages *p, uint32_t slot, uint64_t generation);
  43 | uint64_t mesh_pages_highest(const mesh_pages *p, uint32_t slot);
  44 | 
  45 | void mesh_pages_produce_hook(mesh_pages *p, mesh_pages_hook hook, void *capture);
  46 | uint64_t mesh_pages_producible(const mesh_pages *p, uint32_t slot);
  47 | int mesh_pages_publish(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation);
  48 | int mesh_pages_consume(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation);
  49 | int mesh_pages_faulted(const mesh_pages *p, uint32_t slot, uint64_t generation, uint32_t *first, uint32_t *second);
  50 | uint64_t mesh_pages_hash(const void *data, size_t bytes, uint64_t seed);
  51 | uint64_t mesh_pages_foreign_epoch(const mesh_pages *p);
  52 | int mesh_pages_digest(const mesh_pages *p, uint32_t slot, uint64_t generation, uint64_t *hash);
  53 | int mesh_pages_status(const mesh_pages *p);
  54 | int mesh_pages_settled(const mesh_pages *p);
  55 | int mesh_pages_recover(mesh_pages *p);
  56 | uint64_t mesh_pages_incarnation(const mesh_pages *p);
  57 | void mesh_pages_fail(mesh_pages *p, int error);
  58 | size_t mesh_pages_describe(const mesh_pages *p, char *out, size_t bytes);
  59 | #endif
```

### rdma/mesh-pages.c — 805 lines

**M, lines 1–23.** Includes/types are A. ABSENT and an issued stamp encoding are table representation; parity/window/digest-ring constants do not authorize independent algorithm state.

```text
   1 | #include "mesh-pages.h"
   2 | #include "mesh-wire.h"
   3 | #include <errno.h>
   4 | #include <pthread.h>
   5 | #include <sched.h>
   6 | #include <signal.h>
   7 | #include <stdio.h>
   8 | #include <stdlib.h>
   9 | #include <string.h>
  10 | #include <sys/mman.h>
  11 | #include <time.h>
  12 | #include <unistd.h>
  13 | 
  14 | #define UNUSED (UINT32_MAX-1)
  15 | #define ABSENT UINT32_MAX
  16 | #define WINDOW (MESH_RING/2)
  17 | #define MIX 0xff51afd7ed558ccdULL
  18 | #define MIX2 0xc4ceb9fe1a85ec53ULL
  19 | #define DIGESTS 8
  20 | #define WRITING (UINT64_C(1)<<63)
  21 | typedef _Float16 half8 __attribute__((ext_vector_type(8)));
  22 | typedef float float8 __attribute__((ext_vector_type(8)));
  23 | 
```

**M, lines 24–63.** R: table, stamp, uses. A: immutable spec/maps/function list. X/D: publish/publishing/pending, fill_generation/fill_count/highest/complete/producible/released, heap digest ring, owner array and algorithm work queue. Transport handles and hardware completion ownership are B; counters may not authorize numerical work. Fault/diagnostic fields are O.

```text
  24 | struct slot {
  25 |   struct mesh_pages_slot spec;
  26 |   uint32_t *table; uint64_t *stamp; unsigned char *inflight;
  27 |   _Atomic uint64_t *publish[2]; _Atomic uint64_t publishing[2]; uint64_t *pending; size_t words;
  28 |   size_t base;
  29 |   uint32_t flying;
  30 |   _Atomic uint64_t fill_generation[2], fill_count[2], highest, complete, producible;
  31 |   uint64_t released;
  32 |   uint32_t dependent[MESH_PAGES_DEPENDENCIES], dependent_lag[MESH_PAGES_DEPENDENCIES], dependents;
  33 |   uint64_t fault_generation; uint32_t fault_first, fault_second, fault_have;
  34 |   int transported;
  35 |   unsigned char *uses;
  36 |   struct digest { uint64_t pending; _Atomic uint64_t hash, generation; uint32_t count; } digest[DIGESTS];
  37 | };
  38 | struct work { uint32_t page, slot, index; uint64_t generation; unsigned char kind; };
  39 | enum { W_ZERO, W_RELEASE, W_SENT, W_RECYCLE };
  40 | struct mesh_pages_function {
  41 |   struct mesh_pages *owner;
  42 |   struct mesh_pages_map *maps;
  43 |   uint32_t inputs, outputs, rows, *indices;
  44 |   struct mesh_pages_function *next;
  45 | };
  46 | struct reduce { struct mesh_pages_reduce spec; mesh_pages_function *function; };
  47 | struct mesh_pages {
  48 |   struct mesh_ctx *context; struct hdr *M;
  49 |   struct mesh_epoch epoch; unsigned char plan[32];
  50 |   struct slot *slots; size_t count;
  51 |   uint32_t versions; struct mesh_pages_policy policy;
  52 |   uint32_t *table; size_t table_bytes; uint64_t *stamps;
  53 |   uint32_t *owner;
  54 |   uint32_t *later; size_t later_count;
  55 |   struct work *work; _Atomic uint64_t work_head, work_tail;
  56 |   size_t flying;
  57 |   pthread_t thread, helper, reducer; _Atomic int running, status;
  58 |   uint64_t incarnation, foreign_epoch;
  59 |   mesh_pages_hook hook; void *capture;
  60 |   struct reduce *reduces; size_t reduce_count;
  61 |   mesh_pages_function *functions;
  62 |   uint64_t now, refreshed, integrity, stale, duplicates, overwrites, faults;
  63 | };
```

**M, lines 64–93.** Page address calculation and hash arithmetic are A. parity is D with rotating publication/digest storage. Clock/fault injection is O, not readiness.

```text
  64 | 
  65 | static uint64_t clock_ns(void){
  66 |   struct timespec clock; clock_gettime(CLOCK_MONOTONIC,&clock);
  67 |   return (uint64_t)clock.tv_sec*1000000000u+clock.tv_nsec; }
  68 | static size_t header_bytes(void){ return sizeof(struct wire)+sizeof(struct mesh_frame); }
  69 | static void *zeroed(size_t n){ void *p=calloc(1,n?n:1); if(!p) errno=ENOMEM; return p; }
  70 | static uint64_t mix(uint64_t x){ x^=x>>33; x*=MIX; x^=x>>33; x*=MIX2; x^=x>>33; return x; }
  71 | static unsigned char *payload_at(const mesh_pages *p, uint32_t page){ return mesh_at(p->M,page)+header_bytes(); }
  72 | static uint32_t parity(const mesh_pages *p, uint64_t generation){ return (uint32_t)(((generation-1)/p->versions)&1); }
  73 | static int faulted(const mesh_pages *p, const struct slot *s, uint64_t generation, uint32_t *first, uint32_t *second){
  74 |   if(!p->policy.fault_period || s->spec.pages<2) return 0;
  75 |   uint64_t r=mix(p->policy.fault_seed^((uint64_t)s->spec.sid<<32)^generation*0x9e3779b97f4a7c15ULL);
  76 |   if(r%p->policy.fault_period) return 0;
  77 |   uint64_t a=mix(r+1)%s->spec.pages, b=mix(r+2)%(s->spec.pages-1);
  78 |   if(b>=a) b++;
  79 |   *first=(uint32_t)a; *second=(uint32_t)b; return 1;
  80 | }
  81 | 
  82 | __attribute__((target("crc")))
  83 | uint64_t mesh_pages_hash(const void *data, size_t bytes, uint64_t seed){
  84 |   const unsigned char *d=data; size_t n=bytes/32;
  85 |   uint32_t a=(uint32_t)seed, b=(uint32_t)(seed>>32), c=~a, e=~b;
  86 |   for(size_t i=0;i<n;i++){
  87 |     uint64_t w0,w1,w2,w3; memcpy(&w0,d+i*32,8); memcpy(&w1,d+i*32+8,8); memcpy(&w2,d+i*32+16,8); memcpy(&w3,d+i*32+24,8);
  88 |     a=__builtin_arm_crc32cd(a,w0); b=__builtin_arm_crc32cd(b,w1); c=__builtin_arm_crc32cd(c,w2); e=__builtin_arm_crc32cd(e,w3);
  89 |   }
  90 |   for(size_t i=n*32;i<bytes;i++) a=__builtin_arm_crc32cb(a,d[i]);
  91 |   return mix(((uint64_t)a<<32|b)^mix((uint64_t)c<<32|e));
  92 | }
  93 | 
```

**M, lines 94–158.** A: pre-invocation allocation, range/layout checks, static maps and reverse edges. X/D: allocations for separate ownership/publication/pending arrays. Only storage-root spans initialize use counts on claim and recycle; other local allocations do not implement the general freeing contract. The table may use arrays for its three fields, but must remain one authority.

```text
  94 | mesh_pages *mesh_pages_compile(struct mesh_ctx *context, struct mesh_epoch epoch, const unsigned char plan[32],
  95 |   const struct mesh_pages_slot *slots, size_t count, uint32_t versions, struct mesh_pages_policy policy){
  96 |   if(!context || !context->M || !mesh_epoch_set(epoch) || !count || versions<2){ errno=EINVAL; return NULL; }
  97 |   struct hdr *M=context->M;
  98 |   mesh_pages *p=zeroed(sizeof *p); if(!p) return NULL;
  99 |   p->context=context; p->M=M; p->epoch=epoch; memcpy(p->plan,plan,32);
 100 |   p->versions=versions; p->policy=policy;
 101 |   p->count=count;
 102 |   p->slots=zeroed(count*sizeof *p->slots);
 103 |   p->owner=zeroed((size_t)M->arena*sizeof *p->owner);
 104 |   if(p->owner) for(uint32_t j=0;j<M->arena;j++) p->owner[j]=UNUSED;
 105 |   p->later=zeroed(MESH_RING*sizeof *p->later);
 106 |   p->work=zeroed(MESH_RING*sizeof *p->work);
 107 |   if(!p->slots || !p->owner || !p->later || !p->work){ mesh_pages_free(p); return NULL; }
 108 |   size_t entries=0, arena=0;
 109 |   for(size_t i=0;i<count;i++){
 110 |     const struct mesh_pages_slot *x=&slots[i];
 111 |     if(!x->pages || x->pages>M->pool || x->depends>MESH_PAGES_DEPENDENCIES){ mesh_pages_free(p); errno=EINVAL; return NULL; }
 112 |     for(uint8_t d=0;d<x->depends;d++) if(x->dependency[d]>=count){ mesh_pages_free(p); errno=EINVAL; return NULL; }
 113 |     if(x->storage && (x->receive || x->storage>count || slots[x->storage-1].receive || slots[x->storage-1].storage!=x->storage || x->pages>slots[x->storage-1].pages)){ mesh_pages_free(p); errno=EINVAL; return NULL; }
 114 |     entries+=x->pages;
 115 |     if(!x->receive && (!x->storage || x->storage==i+1)) arena+=x->pages;
 116 |   }
 117 |   if(arena>M->arena){ mesh_pages_free(p); errno=ENOMEM; return NULL; }
 118 |   size_t alignment=(size_t)getpagesize();
 119 |   p->table_bytes=(entries*sizeof(uint32_t)+alignment-1)/alignment*alignment;
 120 |   p->table=mmap(NULL,p->table_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
 121 |   if(p->table==MAP_FAILED){ p->table=NULL; mesh_pages_free(p); return NULL; }
 122 |   memset(p->table,0xff,p->table_bytes);
 123 |   p->stamps=zeroed(entries*sizeof *p->stamps);
 124 |   if(!p->stamps){ mesh_pages_free(p); return NULL; }
 125 |   arena=0; entries=0;
 126 |   for(size_t i=0;i<count;i++){
 127 |     struct slot *s=&p->slots[i]; s->spec=slots[i];
 128 |     s->table=p->table+entries; s->stamp=p->stamps+entries; entries+=s->spec.pages;
 129 |     s->words=mesh_page_words(s->spec.pages);
 130 |     s->inflight=zeroed(s->spec.pages); s->uses=zeroed(s->spec.pages);
 131 |     s->publish[0]=zeroed(s->words*sizeof *s->publish[0]); s->publish[1]=zeroed(s->words*sizeof *s->publish[1]);
 132 |     s->pending=zeroed(s->words*sizeof *s->pending);
 133 |     if(!s->inflight || !s->uses || !s->publish[0] || !s->publish[1] || !s->pending){ mesh_pages_free(p); return NULL; }
 134 |     s->transported=!s->spec.receive && s->spec.peer!=MESH_PAGES_LOCAL;
 135 |     if(!s->spec.receive && (!s->spec.storage || s->spec.storage==i+1)){
 136 |       s->base=arena;
 137 |       for(uint32_t j=0;j<s->spec.pages;j++){ s->table[j]=M->pool+(uint32_t)(arena+j); p->owner[arena+j]=s->spec.storage?UNUSED:(uint32_t)i; }
 138 |       arena+=s->spec.pages;
 139 |     }
 140 |     for(uint8_t d=0;d<s->spec.depends;d++){
 141 |       struct slot *a=&p->slots[s->spec.dependency[d]];
 142 |       if(a->dependents==MESH_PAGES_DEPENDENCIES){ mesh_pages_free(p); errno=E2BIG; return NULL; }
 143 |       a->dependent_lag[a->dependents]=s->spec.lag[d];
 144 |       a->dependent[a->dependents++]=(uint32_t)i;
 145 |     }
 146 |   }
 147 |   for(size_t i=0;i<count;i++){
 148 |     struct slot *s=&p->slots[i];
 149 |     if(!s->spec.storage) continue;
 150 |     s->base=p->slots[s->spec.storage-1].base;
 151 |     for(uint32_t j=0;j<s->spec.pages;j++){
 152 |       s->table[j]=M->pool+(uint32_t)(s->base+j);
 153 |       if(s->spec.storage==i+1) memset(payload_at(p,s->table[j]),0,mesh_pages_payload(p));
 154 |     }
 155 |   }
 156 |   return p;
 157 | }
 158 | 
```

**M, lines 159–184.** Resource disposal and actual page/table access are A/B. filled/highest queries are D. Disposing forbidden state is deleted with that state, not retained as an API.

```text
 159 | int mesh_pages_free(mesh_pages *p){
 160 |   if(!p) return 0;
 161 |   mesh_pages_stop(p);
 162 |   for(size_t i=0;i<p->count;i++){ struct slot *s=&p->slots[i]; free(s->inflight); free(s->uses); free((void*)s->publish[0]); free((void*)s->publish[1]); free(s->pending); }
 163 |   while(p->functions){ mesh_pages_function *f=p->functions; p->functions=f->next; free(f->maps); free(f->indices); free(f); }
 164 |   free(p->reduces);
 165 |   if(p->table) munmap(p->table,p->table_bytes);
 166 |   free(p->stamps); free(p->slots); free(p->owner); free(p->later); free(p->work); free(p);
 167 |   return 0;
 168 | }
 169 | 
 170 | size_t mesh_pages_header(const mesh_pages *p){ (void)p; return header_bytes(); }
 171 | size_t mesh_pages_payload(const mesh_pages *p){ return p->M->pgsz-header_bytes(); }
 172 | const uint32_t *mesh_pages_entries(const mesh_pages *p, uint32_t slot){ return slot<p->count?p->slots[slot].table:NULL; }
 173 | const uint32_t *mesh_pages_table(const mesh_pages *p, size_t *bytes){ *bytes=p->table_bytes; return p->table; }
 174 | const uint64_t *mesh_pages_stamps(const mesh_pages *p, uint32_t slot){ return slot<p->count?p->slots[slot].stamp:NULL; }
 175 | void *mesh_pages_data(const mesh_pages *p,uint32_t slot,uint32_t page){
 176 |   return payload_at(p,__atomic_load_n(&p->slots[slot].table[page],__ATOMIC_ACQUIRE)); }
 177 | uint32_t mesh_pages_filled(const mesh_pages *p, uint32_t slot, uint64_t generation){
 178 |   if(slot>=p->count || !generation) return 0;
 179 |   const struct slot *s=&p->slots[slot]; uint32_t k=parity(p,generation);
 180 |   if(atomic_load_explicit(&s->fill_generation[k],memory_order_acquire)!=generation) return 0;
 181 |   return (uint32_t)atomic_load_explicit(&s->fill_count[k],memory_order_acquire);
 182 | }
 183 | uint64_t mesh_pages_highest(const mesh_pages *p, uint32_t slot){ return slot<p->count?atomic_load_explicit(&p->slots[slot].highest,memory_order_acquire):0; }
 184 | 
```

**A, lines 185–228.** Configuration-time input/output maps and overlap checks implement known functions. The indices array is scan scratch, not a persistent consumed mask. Logical overlap checks alone do not prove physical-alias single-writer safety.

```text
 185 | static int maps_overlap(const struct mesh_pages_map *a,uint32_t rows_a,const struct mesh_pages_map *b,uint32_t rows_b){
 186 |   if(a->slot!=b->slot) return 0;
 187 |   uint32_t r=0,q=0;
 188 |   while(r<rows_a && q<rows_b){
 189 |     uint64_t x=(uint64_t)a->first+(uint64_t)r*a->stride, y=(uint64_t)b->first+(uint64_t)q*b->stride;
 190 |     if(x<y+b->count && y<x+a->count) return 1;
 191 |     if(x<y) r++; else q++;
 192 |   }
 193 |   return 0;
 194 | }
 195 | static mesh_pages_function *bind_function(mesh_pages *p, struct mesh_pages_function_spec spec, int replacing_reduces){
 196 |   if(atomic_load_explicit(&p->running,memory_order_acquire) || !spec.rows || !spec.inputs || !spec.outputs || !spec.input || !spec.output){ errno=EINVAL; return NULL; }
 197 |   for(uint32_t side=0;side<2;side++){
 198 |     const struct mesh_pages_map *maps=side?spec.output:spec.input;
 199 |     uint32_t count=side?spec.outputs:spec.inputs;
 200 |     for(uint32_t i=0;i<count;i++){
 201 |       const struct mesh_pages_map *m=&maps[i];
 202 |       if(m->slot>=p->count || !m->count || m->lag>=WRITING){ errno=EINVAL; return NULL; }
 203 |       uint64_t end=(uint64_t)m->first+(uint64_t)(spec.rows-1)*m->stride+m->count;
 204 |       if(end>p->slots[m->slot].spec.pages || (side && (p->slots[m->slot].spec.receive || m->lag || (spec.rows>1 && m->stride<m->count)))){ errno=EINVAL; return NULL; }
 205 |       if(side){
 206 |         for(uint32_t j=0;j<i;j++) if(maps_overlap(m,spec.rows,&maps[j],spec.rows)){ errno=EINVAL; return NULL; }
 207 |         for(mesh_pages_function *f=p->functions;f;f=f->next){
 208 |           int replaced=0;
 209 |           if(replacing_reduces) for(size_t r=0;r<p->reduce_count;r++) replaced|=f==p->reduces[r].function;
 210 |           if(replaced) continue;
 211 |           for(uint32_t j=0;j<f->outputs;j++) if(maps_overlap(m,spec.rows,&f->maps[f->inputs+j],f->rows)){ errno=EINVAL; return NULL; }
 212 |         }
 213 |       }
 214 |     }
 215 |   }
 216 |   mesh_pages_function *f=zeroed(sizeof *f); if(!f) return NULL;
 217 |   size_t count=(size_t)spec.inputs+spec.outputs;
 218 |   f->maps=zeroed(count*sizeof *f->maps); f->indices=zeroed((size_t)spec.rows*sizeof *f->indices);
 219 |   if(!f->maps || !f->indices){ free(f->maps); free(f->indices); free(f); return NULL; }
 220 |   memcpy(f->maps,spec.input,(size_t)spec.inputs*sizeof *f->maps);
 221 |   memcpy(f->maps+spec.inputs,spec.output,(size_t)spec.outputs*sizeof *f->maps);
 222 |   f->owner=p; f->inputs=spec.inputs; f->outputs=spec.outputs; f->rows=spec.rows;
 223 |   f->next=p->functions; p->functions=f;
 224 |   return f;
 225 | }
 226 | 
 227 | mesh_pages_function *mesh_pages_bind(mesh_pages *p, struct mesh_pages_function_spec spec){ return bind_function(p,spec,0); }
 228 | 
```

**M, lines 229–247.** X: storage_ready consults independent owner array. R/A: claim actual destination stamp and row page/use count. Delete separate physical owner mutation. WRITING is allowed only as an uncompleted destination-stamp encoding, with single-writer/visibility proof.

```text
 229 | // ../design/algorithm-sources.md#operand-matching-and-storage
 230 | static int storage_ready(const mesh_pages *p, const struct slot *s, uint32_t first, uint32_t count){
 231 |   if(!s->spec.storage) return 1;
 232 |   for(uint32_t j=first;j<first+count;j++) if(__atomic_load_n(p->owner+s->base+j,__ATOMIC_ACQUIRE)!=UNUSED) return 0;
 233 |   return 1;
 234 | }
 235 | 
 236 | // ../design/algorithm-sources.md#operand-matching-and-storage
 237 | static void claim_rows(mesh_pages *p, struct slot *s, uint32_t first, uint32_t count, uint64_t generation){
 238 |   for(uint32_t j=first;j<first+count;j++){
 239 |     __atomic_store_n(s->stamp+j,WRITING|generation,__ATOMIC_RELEASE);
 240 |     if(s->spec.storage){
 241 |       s->uses[j]=(unsigned char)s->dependents;
 242 |       __atomic_store_n(s->table+j,p->M->pool+(uint32_t)(s->base+j),__ATOMIC_RELEASE);
 243 |       __atomic_store_n(p->owner+s->base+j,(uint32_t)(s-p->slots),__ATOMIC_RELEASE);
 244 |     }
 245 |   }
 246 | }
 247 | 
```

**M, lines 248–284.** R: input stamp/presence conjunction and actual destination claims. D: producible precondition. X: owner-array storage predicate. Short-circuit traversal is A, but optimizing this mixed predicate did not make it conformant.

```text
 248 | // ../design/algorithm-sources.md#operand-matching-and-storage
 249 | size_t mesh_pages_scan(mesh_pages_function *f, uint64_t generation, const uint32_t **indices){
 250 |   mesh_pages *p=f->owner; *indices=f->indices;
 251 |   if(!generation || generation>=WRITING || mesh_pages_status(p)<0) return 0;
 252 |   for(uint32_t i=0;i<f->outputs;i++){
 253 |     const struct mesh_pages_map *m=&f->maps[f->inputs+i];
 254 |     if(mesh_pages_producible(p,m->slot)<generation) return 0;
 255 |   }
 256 |   size_t selected=0;
 257 |   for(uint32_t row=0;row<f->rows;row++){
 258 |     const struct mesh_pages_map *first=&f->maps[f->inputs];
 259 |     if(__atomic_load_n(p->slots[first->slot].stamp+first->first+row*first->stride,__ATOMIC_ACQUIRE)>=generation) continue;
 260 |     unsigned ready=1;
 261 |     for(uint32_t i=0;i<f->inputs && ready;i++){
 262 |       const struct mesh_pages_map *m=&f->maps[i]; const struct slot *s=&p->slots[m->slot];
 263 |       if(generation<=m->lag){ ready=0; break; }
 264 |       for(uint32_t j=0;j<m->count && ready;j++){
 265 |         uint32_t at=m->first+row*m->stride+j;
 266 |         ready=__atomic_load_n(s->stamp+at,__ATOMIC_ACQUIRE)==generation-m->lag
 267 |           && __atomic_load_n(s->table+at,__ATOMIC_ACQUIRE)!=ABSENT;
 268 |       }
 269 |     }
 270 |     if(!ready) continue;
 271 |     for(uint32_t i=0;i<f->outputs && ready;i++){
 272 |       const struct mesh_pages_map *m=&f->maps[f->inputs+i]; const struct slot *s=&p->slots[m->slot];
 273 |       for(uint32_t j=0;j<m->count && ready;j++) ready=__atomic_load_n(s->stamp+m->first+row*m->stride+j,__ATOMIC_ACQUIRE)<generation;
 274 |       if(ready) ready=storage_ready(p,s,m->first+row*m->stride,m->count);
 275 |     }
 276 |     if(!ready) continue;
 277 |     for(uint32_t i=0;i<f->outputs;i++){
 278 |       const struct mesh_pages_map *m=&f->maps[f->inputs+i]; struct slot *s=&p->slots[m->slot];
 279 |       claim_rows(p,s,m->first+row*m->stride,m->count,generation);
 280 |     }
 281 |     f->indices[selected++]=row;
 282 |   }
 283 |   return selected;
 284 | }
```

**M, lines 285–321.** Claim/cancel/complete have allowed row-transition purposes. D: producible condition and indirect publication path. Cancel resets the stamp but does not restore aliased physical ownership; this cannot be treated as complete recovery.

```text
 285 | 
 286 | int mesh_pages_claim(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation){
 287 |   if(slot>=p->count || !generation || generation>=WRITING || !count || mesh_pages_status(p)<0) return 0;
 288 |   struct slot *s=&p->slots[slot];
 289 |   if(s->spec.receive || first>s->spec.pages || count>s->spec.pages-first || mesh_pages_producible(p,slot)<generation) return 0;
 290 |   for(uint32_t j=first;j<first+count;j++) if(__atomic_load_n(s->stamp+j,__ATOMIC_ACQUIRE)>=generation) return 0;
 291 |   if(!storage_ready(p,s,first,count)) return 0;
 292 |   claim_rows(p,s,first,count,generation);
 293 |   return 1;
 294 | }
 295 | void mesh_pages_cancel(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation){
 296 |   if(slot>=p->count || first>p->slots[slot].spec.pages || count>p->slots[slot].spec.pages-first) return;
 297 |   for(uint32_t j=first;j<first+count;j++){
 298 |     uint64_t expected=WRITING|generation;
 299 |     __atomic_compare_exchange_n(p->slots[slot].stamp+j,&expected,0,0,__ATOMIC_RELEASE,__ATOMIC_RELAXED);
 300 |   }
 301 | }
 302 | size_t mesh_pages_writing(const mesh_pages *p){
 303 |   size_t count=0;
 304 |   for(size_t i=0;i<p->count;i++) for(uint32_t j=0;j<p->slots[i].spec.pages;j++)
 305 |     count+=(__atomic_load_n(p->slots[i].stamp+j,__ATOMIC_ACQUIRE)&WRITING)!=0;
 306 |   return count;
 307 | }
 308 | int mesh_pages_complete(mesh_pages_function *f, uint32_t row, uint64_t generation){
 309 |   if(row>=f->rows || !generation || generation>=WRITING) return -EINVAL;
 310 |   for(uint32_t i=0;i<f->outputs;i++){
 311 |     const struct mesh_pages_map *m=&f->maps[f->inputs+i];
 312 |     const struct slot *s=&f->owner->slots[m->slot];
 313 |     for(uint32_t j=0;j<m->count;j++) if(__atomic_load_n(s->stamp+m->first+row*m->stride+j,__ATOMIC_ACQUIRE)!=(WRITING|generation)) return -EINVAL;
 314 |   }
 315 |   for(uint32_t i=0;i<f->outputs;i++){
 316 |     const struct mesh_pages_map *m=&f->maps[f->inputs+i];
 317 |     int result=mesh_pages_publish(f->owner,m->slot,m->first+row*m->stride,m->count,generation);
 318 |     if(result) return result;
 319 |   }
 320 |   return 0;
 321 | }
```

**D, lines 322–338.** Hook, producible and heap digest-ring query restate algorithm state. Fault and foreign-epoch observation may remain outside numerical readiness.

```text
 322 | void mesh_pages_produce_hook(mesh_pages *p, mesh_pages_hook hook, void *capture){ p->hook=hook; p->capture=capture; }
 323 | uint64_t mesh_pages_producible(const mesh_pages *p, uint32_t slot){
 324 |   return slot<p->count?atomic_load_explicit(&p->slots[slot].producible,memory_order_acquire):0; }
 325 | int mesh_pages_faulted(const mesh_pages *p, uint32_t slot, uint64_t generation, uint32_t *first, uint32_t *second){
 326 |   return slot<p->count && p->slots[slot].spec.receive && faulted(p,&p->slots[slot],generation,first,second); }
 327 | uint64_t mesh_pages_foreign_epoch(const mesh_pages *p){ return __atomic_load_n(&p->foreign_epoch,__ATOMIC_ACQUIRE); }
 328 | int mesh_pages_digest(const mesh_pages *p, uint32_t slot, uint64_t generation, uint64_t *hash){
 329 |   if(slot>=p->count) return 0;
 330 |   for(size_t i=0;i<DIGESTS;i++){
 331 |     const struct digest *d=&p->slots[slot].digest[i];
 332 |     if(atomic_load_explicit(&d->generation,memory_order_acquire)!=generation) continue;
 333 |     *hash=atomic_load_explicit(&d->hash,memory_order_relaxed);
 334 |     atomic_thread_fence(memory_order_acquire);
 335 |     return atomic_load_explicit(&d->generation,memory_order_relaxed)==generation;
 336 |   }
 337 |   return 0;
 338 | }
```

**M, lines 339–376.** Configuration-time binding/freeing is A. X: materialized-vs-partial reduce representation and group matching replacing accumulator/index-page composition. A numerical row span is valid geometry, but cannot force partial addition to wait for every page in that span.

```text
 339 | static void function_free(mesh_pages_function *f){
 340 |   if(!f) return;
 341 |   mesh_pages_function **at=&f->owner->functions;
 342 |   while(*at && *at!=f) at=&(*at)->next;
 343 |   if(*at) *at=f->next;
 344 |   free(f->maps); free(f->indices); free(f);
 345 | }
 346 | // ../design/algorithm-sources.md#collective-arithmetic-and-asynchronous-reduction
 347 | int mesh_pages_reduces(mesh_pages *p, const struct mesh_pages_reduce *reduces, size_t count){
 348 |   if(atomic_load_explicit(&p->running,memory_order_acquire)) return EBUSY;
 349 |   struct reduce *list=zeroed(count*sizeof *list); if(!list) return ENOMEM;
 350 |   int error=EINVAL;
 351 |   for(size_t r=0;r<count;r++){
 352 |     const struct mesh_pages_reduce *x=&reduces[r];
 353 |     if(x->output>=p->count || !x->inputs || x->inputs>MESH_PAGES_DEPENDENCIES || !x->group || x->kind>MESH_REDUCE_PARTIAL) goto invalid;
 354 |     if(x->input[0]>=p->count) goto invalid;
 355 |     const struct slot *out=&p->slots[x->output];
 356 |     uint32_t pages=p->slots[x->input[0]].spec.pages, factor=x->kind==MESH_REDUCE_PARTIAL?2:1;
 357 |     if(out->spec.receive || pages%x->group || (uint64_t)out->spec.pages!=(uint64_t)pages*factor || !out->spec.pagewise) goto invalid;
 358 |     if(x->bytes%16 || !x->bytes || (size_t)x->offset+x->bytes>mesh_pages_payload(p) || (factor==2 && x->bytes%32)) goto invalid;
 359 |     struct mesh_pages_map inputs[MESH_PAGES_DEPENDENCIES], output={x->output,0,x->group*factor,x->group*factor,0};
 360 |     for(uint8_t i=0;i<x->inputs;i++){
 361 |       if(x->input[i]>=p->count || p->slots[x->input[i]].spec.pages!=pages) goto invalid;
 362 |       int listed=0; for(uint8_t d=0;d<out->spec.depends;d++) listed|=out->spec.dependency[d]==x->input[i];
 363 |       if(!listed) goto invalid;
 364 |       inputs[i]=(struct mesh_pages_map){x->input[i],0,x->group,x->group,0};
 365 |     }
 366 |     list[r].spec=*x;
 367 |     list[r].function=bind_function(p,(struct mesh_pages_function_spec){inputs,&output,x->inputs,1,pages/x->group},1);
 368 |     if(!list[r].function){ error=errno; goto invalid; }
 369 |   }
 370 |   for(size_t r=0;r<p->reduce_count;r++) function_free(p->reduces[r].function);
 371 |   free(p->reduces); p->reduces=list; p->reduce_count=count;
 372 |   return 0;
 373 | invalid:
 374 |   for(size_t q=0;q<count;q++) function_free(list[q].function);
 375 |   free(list); return error;
 376 | }
```

**M, lines 377–382.** R: link status word and negative failure publication. Incarnation is connection metadata, not numerical readiness. GPU computation failures must conclude failed evaluations and never publish valid result stamps.

```text
 377 | int mesh_pages_status(const mesh_pages *p){ return atomic_load_explicit(&p->status,memory_order_acquire); }
 378 | uint64_t mesh_pages_incarnation(const mesh_pages *p){ return p->incarnation; }
 379 | void mesh_pages_fail(mesh_pages *p, int error){
 380 |   int expected=0; atomic_compare_exchange_strong(&p->status,&expected,-(error>0?error:ECANCELED));
 381 | }
 382 | 
```

**X, lines 383–405.** mark/publish/consume queue bitmap completion instead of directly publishing/releasing rows. publishing[parity] is a second generation authority. Delete this indirection and its APIs implementation, retaining only the required literal row operations.

```text
 383 | static int mark(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation, int receive){
 384 |   if(slot>=p->count || (p->slots[slot].spec.receive!=0)!=receive || !generation || generation>=WRITING) return -EINVAL;
 385 |   struct slot *s=&p->slots[slot];
 386 |   if(first>s->spec.pages || count>s->spec.pages-first) return -EINVAL;
 387 |   if(!receive && mesh_pages_status(p)<0){ mesh_pages_cancel(p,slot,first,count,generation); return mesh_pages_status(p); }
 388 |   if(!receive && atomic_load_explicit(&s->producible,memory_order_acquire)<generation) return -EBUSY;
 389 |   uint32_t k=parity(p,generation);
 390 |   atomic_store_explicit(&s->publishing[k],generation,memory_order_release);
 391 |   for(size_t at=first,end=first+count;at<end;){
 392 |     size_t bit=at%64, bits=end-at; if(bits>64-bit) bits=64-bit;
 393 |     atomic_fetch_or_explicit(&s->publish[k][at/64],(UINT64_MAX>>(64-bits))<<bit,memory_order_release);
 394 |     at+=bits;
 395 |   }
 396 |   return 0;
 397 | }
 398 | 
 399 | int mesh_pages_publish(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation){
 400 |   return mark(p,slot,first,count,generation,0);
 401 | }
 402 | int mesh_pages_consume(mesh_pages *p, uint32_t slot, uint32_t first, uint32_t count, uint64_t generation){
 403 |   return mark(p,slot,first,count,generation,1);
 404 | }
 405 | 
```

**M, lines 406–439.** R/B: submit actual page descriptors, preserve NIC access lifetime, asynchronously zero and return pages. X/D: algorithm inflight/window state, W_SENT/W_RELEASE queue and later list duplicating the binding. Do not remove actual transport completion safety; it belongs to the substrate/page lifetime.

```text
 406 | static void digest_add(mesh_pages *p, struct slot *s, uint64_t generation, uint32_t page, const void *payload);
 407 | static void enqueue(mesh_pages *p, struct work w);
 408 | static int push_page(mesh_pages *p, struct slot *s, uint32_t page){
 409 |   if(s->inflight[page] || p->flying>=WINDOW) return 0;
 410 |   unsigned char *q=mesh_at(p->M,s->table[page])+sizeof(struct wire);
 411 |   struct mesh_frame frame={.h={page,s->spec.sid,K_DATA},.epoch={p->epoch.high,s->stamp[page]},.source=(uint16_t)p->M->node,.target=s->spec.peer};
 412 |   size_t n=mesh_frame_encode(q,&frame,mesh_pages_payload(p));
 413 |   struct desc d={.page=s->table[page],.bytes=(uint32_t)n,.node=s->spec.peer};
 414 |   if(push(p->M,SUB,&d)) return 0;
 415 |   s->inflight[page]=1; s->flying++; p->flying++;
 416 |   enqueue(p,(struct work){.page=s->table[page],.slot=(uint32_t)(s-p->slots),.index=page,.generation=s->stamp[page],.kind=W_SENT});
 417 |   return 1;
 418 | }
 419 | static void enqueue(mesh_pages *p, struct work w){
 420 |   for(;;){
 421 |     uint64_t head=atomic_load_explicit(&p->work_head,memory_order_relaxed);
 422 |     if(head-atomic_load_explicit(&p->work_tail,memory_order_acquire)<MESH_RING){
 423 |       p->work[head%MESH_RING]=w; atomic_store_explicit(&p->work_head,head+1,memory_order_release); return;
 424 |     }
 425 |     sched_yield();
 426 |   }
 427 | }
 428 | static void flush_later(mesh_pages *p){
 429 |   while(p->later_count){
 430 |     struct desc d={.page=p->later[0]};
 431 |     if(push(p->M,REL,&d)) break;
 432 |     memmove(p->later,p->later+1,(--p->later_count)*sizeof *p->later);
 433 |   }
 434 | }
 435 | static void release_page(mesh_pages *p, uint32_t page){
 436 |   memset(payload_at(p,page),0,mesh_pages_payload(p));
 437 |   struct desc d={.page=page};
 438 |   if(push(p->M,REL,&d) && p->later_count<MESH_RING) p->later[p->later_count++]=page;
 439 | }
```

**D, lines 440–468.** filled and digest ring duplicate information outside the table/pages. Hash arithmetic is allowed, but persistent hash/index values must be in configured digest/index pages.

```text
 440 | static void filled(mesh_pages *p, struct slot *s, uint64_t generation){
 441 |   uint32_t k=parity(p,generation);
 442 |   if(atomic_load_explicit(&s->fill_generation[k],memory_order_relaxed)!=generation){
 443 |     atomic_store_explicit(&s->fill_count[k],0,memory_order_relaxed);
 444 |     atomic_store_explicit(&s->fill_generation[k],generation,memory_order_release);
 445 |   }
 446 |   uint64_t count=atomic_fetch_add_explicit(&s->fill_count[k],1,memory_order_release)+1;
 447 |   if(generation>atomic_load_explicit(&s->highest,memory_order_relaxed)) atomic_store_explicit(&s->highest,generation,memory_order_release);
 448 |   if(count==s->spec.pages && generation>atomic_load_explicit(&s->complete,memory_order_relaxed)) atomic_store_explicit(&s->complete,generation,memory_order_release);
 449 | }
 450 | static struct digest *digest_entry(struct slot *s, uint64_t generation){
 451 |   struct digest *oldest=s->digest;
 452 |   for(size_t i=0;i<DIGESTS;i++){
 453 |     struct digest *d=&s->digest[i];
 454 |     if(d->pending==generation) return d;
 455 |     if(d->pending<oldest->pending) oldest=d;
 456 |   }
 457 |   atomic_store_explicit(&oldest->generation,0,memory_order_relaxed);
 458 |   atomic_thread_fence(memory_order_release);
 459 |   oldest->pending=generation; oldest->count=0;
 460 |   atomic_store_explicit(&oldest->hash,0,memory_order_relaxed);
 461 |   return oldest;
 462 | }
 463 | static void digest_add(mesh_pages *p, struct slot *s, uint64_t generation, uint32_t page, const void *payload){
 464 |   struct digest *d=digest_entry(s,generation);
 465 |   uint64_t hash=atomic_load_explicit(&d->hash,memory_order_relaxed)+mesh_pages_hash(payload,mesh_pages_payload(p),(uint64_t)page+1);
 466 |   atomic_store_explicit(&d->hash,hash,memory_order_relaxed);
 467 |   if(++d->count==s->spec.pages) atomic_store_explicit(&d->generation,generation,memory_order_release);
 468 | }
```

**M, lines 469–493.** R: decrement uses and zero/return released pages. X: per-work queue authority and separate owner recycling. Hashing must precede zeroing and release must occur once per proven read. uses > 1 branch alone does not prove exact-once release.

```text
 469 | static void release_entry(mesh_pages *p, struct slot *a, uint32_t j, uint64_t generation){
 470 |   if(a->stamp[j]!=generation || a->table[j]==ABSENT) return;
 471 |   if(a->uses[j]>1){ a->uses[j]--; return; }
 472 |   enqueue(p,(struct work){.page=a->table[j],.slot=(uint32_t)(a-p->slots),.index=j,.generation=generation,.kind=W_RELEASE});
 473 |   __atomic_store_n(&a->table[j],ABSENT,__ATOMIC_RELEASE);
 474 | }
 475 | static void *helper_run(void *argument){
 476 |   mesh_pages *p=argument;
 477 |   pthread_setname_np("mesh-release");
 478 |   unsigned idle=0;
 479 |   while(atomic_load_explicit(&p->running,memory_order_relaxed) || atomic_load_explicit(&p->work_head,memory_order_acquire)!=atomic_load_explicit(&p->work_tail,memory_order_relaxed)){
 480 |     flush_later(p);
 481 |     uint64_t tail=atomic_load_explicit(&p->work_tail,memory_order_relaxed);
 482 |     if(tail==atomic_load_explicit(&p->work_head,memory_order_acquire)){ if(++idle>4096){ sched_yield(); idle=4096; } continue; }
 483 |     idle=0;
 484 |     struct work w=p->work[tail%MESH_RING];
 485 |     if(w.kind==W_RELEASE || w.kind==W_SENT) digest_add(p,&p->slots[w.slot],w.generation,w.index,payload_at(p,w.page));
 486 |     if(w.kind==W_RECYCLE){
 487 |       memset(payload_at(p,w.page),0,mesh_pages_payload(p));
 488 |       __atomic_store_n(p->owner+w.page-p->M->pool,UNUSED,__ATOMIC_RELEASE);
 489 |     } else if(w.kind!=W_SENT) release_page(p,w.page);
 490 |     atomic_store_explicit(&p->work_tail,tail+1,memory_order_release);
 491 |   }
 492 |   return NULL;
 493 | }
```

**X, lines 494–509.** pagewise matching assumes identical input/output page indices. Whole-input path releases every received page after first output publication using released cursor. Neither replaces exact configured read-map proofs; local pages are excluded here.

```text
 494 | static void release_dependencies(mesh_pages *p, struct slot *s, uint32_t page, uint64_t generation){
 495 |   if(s->spec.pagewise){
 496 |     for(uint8_t d=0;d<s->spec.depends;d++){
 497 |       struct slot *a=&p->slots[s->spec.dependency[d]];
 498 |       if(a->spec.receive && page<a->spec.pages && generation>s->spec.lag[d]) release_entry(p,a,page,generation-s->spec.lag[d]);
 499 |     }
 500 |     return;
 501 |   }
 502 |   if(generation<=s->released) return;
 503 |   s->released=generation;
 504 |   for(uint8_t d=0;d<s->spec.depends;d++){
 505 |     struct slot *a=&p->slots[s->spec.dependency[d]];
 506 |     if(!a->spec.receive || generation<=s->spec.lag[d]) continue;
 507 |     for(uint32_t j=0;j<a->spec.pages;j++) release_entry(p,a,j,generation-s->spec.lag[d]);
 508 |   }
 509 | }
```

**M, lines 510–534.** R: install physical page and use count, then publish stamp. O: deliberate fault injection. D: filled counter publication. Arrival must not overwrite still-live prior data.

```text
 510 | static void arrive(mesh_pages *p, struct slot *s, uint32_t page, uint32_t physical, uint64_t generation){
 511 |   s->uses[page]=s->dependents?(unsigned char)s->dependents:1;
 512 |   __atomic_store_n(&s->table[page],physical,__ATOMIC_RELEASE);
 513 |   if(s->fault_generation!=generation){
 514 |     s->fault_generation=generation; s->fault_have=0;
 515 |     if(!faulted(p,s,generation,&s->fault_first,&s->fault_second)) s->fault_have=4;
 516 |   }
 517 |   if(s->fault_have<4 && (page==s->fault_first || page==s->fault_second)){
 518 |     s->fault_have|=page==s->fault_first?1:2;
 519 |     if(s->fault_have==3){
 520 |       uint32_t first=s->table[s->fault_first], second=s->table[s->fault_second];
 521 |       size_t bytes=mesh_pages_payload(p);
 522 |       if(first!=ABSENT && second!=ABSENT){
 523 |         unsigned char *a=payload_at(p,first), *b=payload_at(p,second);
 524 |         for(size_t i=0;i<bytes;i++){ unsigned char x=a[i]^b[i]; a[i]=x; b[i]=x; }
 525 |       } else {
 526 |         unsigned char *a=payload_at(p,first!=ABSENT?first:second);
 527 |         for(size_t i=0;i<bytes;i++) a[i]^=0xa5;
 528 |       }
 529 |       p->faults++; s->fault_have=4;
 530 |     }
 531 |   }
 532 |   __atomic_store_n(&s->stamp[page],generation,__ATOMIC_RELEASE);
 533 |   filled(p,s,generation);
 534 | }
```

**X, lines 535–552.** offer derives progress/highest/complete minimum plus versions, publishes producible and invokes hook. It is an extra reuse/admission algorithm, not literal release proof.

```text
 535 | static void offer(mesh_pages *p, size_t i){
 536 |   struct slot *s=&p->slots[i];
 537 |   if(s->spec.receive) return;
 538 |   uint64_t limit;
 539 |   if(s->dependents){
 540 |     limit=UINT64_MAX;
 541 |     for(uint32_t d=0;d<s->dependents;d++){
 542 |       const struct slot *b=&p->slots[s->dependent[d]];
 543 |       uint64_t shown=b->spec.pagewise?atomic_load_explicit(&b->complete,memory_order_acquire):atomic_load_explicit(&b->highest,memory_order_acquire);
 544 |       shown=shown>s->dependent_lag[d]?shown-s->dependent_lag[d]:0;
 545 |       if(shown<limit) limit=shown;
 546 |     }
 547 |     limit+=p->versions;
 548 |   } else limit=(s->transported && s->flying?0:atomic_load_explicit(&s->complete,memory_order_acquire))+p->versions;
 549 |   if(limit<=atomic_load_explicit(&s->producible,memory_order_relaxed)) return;
 550 |   atomic_store_explicit(&s->producible,limit,memory_order_release);
 551 |   if(p->hook) p->hook(p->capture,(uint32_t)i,limit);
 552 | }
```

**X, lines 553–575.** retire_storage depends on complete counters and heap digest readiness, rewrites whole-slot use counts and frees through owner/work state. Required release/zeroing purpose survives; this representation does not.

```text
 553 | 
 554 | // ../design/algorithm-sources.md#operand-matching-and-storage
 555 | static void retire_storage(mesh_pages *p, struct slot *s){
 556 |   if(!s->spec.storage || __atomic_load_n(s->table,__ATOMIC_ACQUIRE)==ABSENT) return;
 557 |   uint64_t g=__atomic_load_n(s->stamp,__ATOMIC_ACQUIRE);
 558 |   if(!g || g>=WRITING || atomic_load_explicit(&s->complete,memory_order_acquire)!=g) return;
 559 |   unsigned uses=0;
 560 |   for(uint32_t d=0;d<s->dependents;d++){
 561 |     uint64_t complete=atomic_load_explicit(&p->slots[s->dependent[d]].complete,memory_order_acquire);
 562 |     uses+=complete<g+s->dependent_lag[d];
 563 |   }
 564 |   for(uint32_t j=0;j<s->spec.pages;j++) s->uses[j]=(unsigned char)uses;
 565 |   if(uses) return;
 566 |   if(s->transported){
 567 |     uint64_t hash;
 568 |     if(s->flying || !mesh_pages_digest(p,(uint32_t)(s-p->slots),g,&hash)) return;
 569 |     for(size_t w=0;w<s->words;w++) if(s->pending[w]) return;
 570 |   }
 571 |   for(uint32_t j=0;j<s->spec.pages;j++){
 572 |     uint32_t page=__atomic_exchange_n(s->table+j,ABSENT,__ATOMIC_ACQ_REL);
 573 |     if(page!=ABSENT) enqueue(p,(struct work){.page=page,.kind=W_RECYCLE});
 574 |   }
 575 | }
```

**M, lines 576–605.** A/B: configured destination lookup and actual receive delivery. X: application frame-kind dispatcher and overwriting a present previous page without proof of all reads. Connection identity/bounds validation is not a second numerical completion protocol.

```text
 576 | 
 577 | static struct slot *lookup(mesh_pages *p, uint32_t sid, int receive, int from){
 578 |   for(size_t i=0;i<p->count;i++){
 579 |     struct slot *s=&p->slots[i];
 580 |     if(s->spec.sid==sid && (s->spec.receive!=0)==(receive!=0) && s->spec.peer==from) return s;
 581 |   }
 582 |   return NULL;
 583 | }
 584 | static void receive(mesh_pages *p, uint32_t page, size_t bytes, int from){
 585 |   int keep=0;
 586 |   unsigned char *q=mesh_at(p->M,page)+sizeof(struct wire);
 587 |   struct mesh_frame frame; size_t header=mesh_frame_decode(q,bytes,&frame);
 588 |   if(header!=sizeof frame || frame.source!=from || frame.target!=p->M->node){ p->integrity++; goto done; }
 589 |   if(frame.epoch.high!=p->epoch.high){ __atomic_store_n(&p->foreign_epoch,frame.epoch.high,__ATOMIC_RELEASE); p->stale++; goto done; }
 590 |   struct slot *s=lookup(p,frame.h.sid,mesh_frame_receive(frame.h.k),from);
 591 |   if(!s){ p->stale++; goto done; }
 592 |   uint64_t g=frame.epoch.low;
 593 |   switch(frame.h.k){
 594 |   case K_DATA: {
 595 |     uint32_t at=(uint32_t)frame.h.off;
 596 |     if(at>=s->spec.pages || bytes!=header+mesh_pages_payload(p)){ p->integrity++; break; }
 597 |     if(g<=s->stamp[at]){ if(g==s->stamp[at]) p->duplicates++; else p->stale++; break; }
 598 |     if(s->table[at]!=ABSENT){ p->overwrites++; enqueue(p,(struct work){.page=s->table[at],.kind=W_ZERO}); }
 599 |     keep=1; arrive(p,s,at,page,g);
 600 |     break; }
 601 |   default: p->stale++; break;
 602 |   }
 603 | done:
 604 |   if(!keep) enqueue(p,(struct work){.page=page,.kind=W_ZERO});
 605 | }
```

**M, lines 606–645.** B: transport completion needed before hardware may lose ownership. X/D: owner-to-slot lookup and algorithm inflight state; publication bitmap drain and pending bitmap. Direct output stamping is R but currently delayed behind this drain.

```text
 606 | 
 607 | static void acknowledge(mesh_pages *p){
 608 |   struct desc d;
 609 |   while(!pop(p->M,ACK,&d)){
 610 |     if(d.page<p->M->pool || d.page>=p->M->pool+p->M->arena) continue;
 611 |     uint32_t index=d.page-p->M->pool, owner=__atomic_load_n(p->owner+index,__ATOMIC_ACQUIRE);
 612 |     if(owner>=p->count) continue;
 613 |     if(p->flying) p->flying--;
 614 |     struct slot *s=&p->slots[owner]; size_t page=index-s->base;
 615 |     if(s->inflight[page]){ s->inflight[page]=0; s->flying--; }
 616 |   }
 617 | }
 618 | static void transmit(mesh_pages *p, size_t i){
 619 |   struct slot *s=&p->slots[i];
 620 |   for(uint32_t k=0;k<2;k++){
 621 |     for(size_t w=0;w<s->words;w++){
 622 |       uint64_t bits=atomic_exchange_explicit(&s->publish[k][w],0,memory_order_acquire);
 623 |       if(!bits) continue;
 624 |       uint64_t generation=atomic_load_explicit(&s->publishing[k],memory_order_acquire);
 625 |       for(;bits;bits&=bits-1){
 626 |         uint32_t page=(uint32_t)(w*64+(size_t)__builtin_ctzll(bits));
 627 |         if(s->spec.receive){ release_entry(p,s,page,generation); continue; }
 628 |         uint64_t stamp=__atomic_load_n(s->stamp+page,__ATOMIC_ACQUIRE);
 629 |         if(stamp!=(WRITING|generation) && stamp>=generation) continue;
 630 |         __atomic_store_n(&s->stamp[page],generation,__ATOMIC_RELEASE);
 631 |         release_dependencies(p,s,page,generation);
 632 |         if(s->transported) s->pending[w]|=UINT64_C(1)<<(page%64);
 633 |         filled(p,s,generation);
 634 |       }
 635 |     }
 636 |   }
 637 |   if(!s->transported || mesh_pages_status(p)<0) return;
 638 |   for(size_t w=0;w<s->words && p->flying<WINDOW;w++){
 639 |     uint64_t bits=s->pending[w];
 640 |     for(;bits && p->flying<WINDOW;bits&=bits-1){
 641 |       uint32_t page=(uint32_t)(w*64+(size_t)__builtin_ctzll(bits));
 642 |       if(push_page(p,s,page)) s->pending[w]&=~(UINT64_C(1)<<(page%64));
 643 |     }
 644 |   }
 645 | }
```

**M, lines 646–676.** B: observe bridge/link status and deliver incoming pages. R: fail evaluations on lost link. X: running transmit/retire/offer state machines above stamps. A timer used for diagnosis must not become numerical readiness.

```text
 646 | static int linked(const mesh_pages *p){
 647 |   uint64_t ports=atomic_load_explicit(&p->M->port_count,memory_order_acquire);
 648 |   for(uint64_t i=0;i<ports;i++) if(atomic_load_explicit(&mesh_ports(p->M)[i].phase,memory_order_acquire)==MESH_PAIRED) return 1;
 649 |   return ports==0;
 650 | }
 651 | static void refresh(mesh_pages *p){
 652 |   if(atomic_load(&p->M->phase)>=MESH_STOPPING){ mesh_pages_fail(p,ESTALE); return; }
 653 |   if(p->now-p->refreshed<250000000ull) return;
 654 |   p->refreshed=p->now;
 655 |   uint64_t pid=atomic_load(&p->M->bridge_pid);
 656 |   if(pid && kill((pid_t)pid,0)<0 && errno==ESRCH){ mesh_pages_fail(p,ESTALE); return; }
 657 |   if(!linked(p)) mesh_pages_fail(p,ENOTCONN);
 658 | }
 659 | 
 660 | int mesh_pages_progress(mesh_pages *p){
 661 |   p->now=clock_ns();
 662 |   int status=atomic_load_explicit(&p->status,memory_order_acquire);
 663 |   refresh(p);
 664 |   acknowledge(p);
 665 |   for(int turn=0;turn<256;turn++){
 666 |     struct desc d; if(pop(p->M,CMP,&d)) break;
 667 |     if(d.page>=p->M->pool){ p->integrity++; continue; }
 668 |     receive(p,d.page,d.bytes,d.node);
 669 |   }
 670 |   for(size_t i=0;i<p->count;i++) transmit(p,i);
 671 |   if(status<0) return status;
 672 |   for(size_t i=0;i<p->count;i++) retire_storage(p,&p->slots[i]);
 673 |   for(size_t i=0;i<p->count;i++) offer(p,i);
 674 |   return atomic_load_explicit(&p->status,memory_order_acquire);
 675 | }
 676 | // ../design/algorithm-sources.md#collective-arithmetic-and-asynchronous-reduction
```

**X, lines 677–704.** reduce_step chooses highest generation, waits for group, accumulates only in registers, rounds materialized sums to FP16 before normalization and send. FP32 add instruction itself is A. Missing configured persistent accumulator/index pages and normalized publication.

```text
 677 | static size_t reduce_step(mesh_pages *p, struct reduce *r){
 678 |   const struct mesh_pages_reduce *x=&r->spec;
 679 |   uint64_t g=mesh_pages_highest(p,x->input[0]);
 680 |   const uint32_t *indices;
 681 |   size_t selected=mesh_pages_scan(r->function,g,&indices);
 682 |   size_t vectors=x->bytes/16, factor=x->kind==MESH_REDUCE_PARTIAL?2:1;
 683 |   for(size_t n=0;n<selected;n++){
 684 |     uint32_t row=indices[n];
 685 |     for(uint32_t k=0;k<x->group;k++){
 686 |       uint32_t page=row*x->group+k;
 687 |       const half8 *inputs[MESH_PAGES_DEPENDENCIES];
 688 |       for(uint8_t i=0;i<x->inputs;i++) inputs[i]=(const half8*)(mesh_pages_data(p,x->input[i],page)+x->offset);
 689 |       unsigned char *out=payload_at(p,p->slots[x->output].table[page*factor])+x->offset;
 690 |       for(size_t v=0;v<vectors;v++){
 691 |         float8 sum=0;
 692 |         for(uint8_t i=0;i<x->inputs;i++) sum+=__builtin_convertvector(inputs[i][v],float8);
 693 |         if(x->kind==MESH_REDUCE_MATERIALIZED) ((half8*)out)[v]=__builtin_convertvector(sum,half8);
 694 |         else {
 695 |           size_t half=vectors/2;
 696 |           float8 *destination=v<half?(float8*)out:(float8*)(payload_at(p,p->slots[x->output].table[page*factor+1])+x->offset);
 697 |           destination[v%half]=sum;
 698 |         }
 699 |       }
 700 |     }
 701 |     if(mesh_pages_complete(r->function,row,g)) mesh_pages_fail(p,EINVAL);
 702 |   }
 703 |   return selected;
 704 | }
```

**M, lines 705–749.** B: worker lifecycle can implement scanning, delivery and asynchronous zeroing. D: highest sums/idle progress proxies. A distinct worker is not by itself a forbidden algorithm object; its state machines are. Runtime may perform canonical reduction arithmetic, not interpret/schedule application kernels.

```text
 705 | // ../design/algorithm-sources.md#collective-arithmetic-and-asynchronous-reduction
 706 | static void *reduce_run(void *argument){
 707 |   mesh_pages *p=argument;
 708 |   pthread_setname_np("mesh-reduce");
 709 |   unsigned idle=0;
 710 |   while(atomic_load_explicit(&p->running,memory_order_relaxed)){
 711 |     size_t selected=0;
 712 |     for(size_t r=0;r<p->reduce_count;r++) selected+=reduce_step(p,&p->reduces[r]);
 713 |     idle=selected?0:idle+1;
 714 |     if(idle>4096){ sched_yield(); idle=4096; }
 715 |   }
 716 |   return NULL;
 717 | }
 718 | static void *run(void *argument){
 719 |   mesh_pages *p=argument;
 720 |   pthread_setname_np("mesh-pages");
 721 |   unsigned idle=0;
 722 |   while(atomic_load_explicit(&p->running,memory_order_relaxed)){
 723 |     size_t before=p->flying; uint64_t highest=0;
 724 |     for(size_t i=0;i<p->count;i++) highest+=atomic_load_explicit(&p->slots[i].highest,memory_order_relaxed);
 725 |     mesh_pages_progress(p);
 726 |     uint64_t after=0; for(size_t i=0;i<p->count;i++) after+=atomic_load_explicit(&p->slots[i].highest,memory_order_relaxed);
 727 |     idle=before==p->flying && highest==after?idle+1:0;
 728 |     if(idle>4096){ sched_yield(); idle=4096; }
 729 |   }
 730 |   return NULL;
 731 | }
 732 | // ../design/algorithm-sources.md#collective-arithmetic-and-asynchronous-reduction
 733 | int mesh_pages_start(mesh_pages *p){
 734 |   if(atomic_exchange(&p->running,1)) return 0;
 735 |   int error=pthread_create(&p->helper,NULL,helper_run,p);
 736 |   if(error){ atomic_store(&p->running,0); return error; }
 737 |   error=pthread_create(&p->thread,NULL,run,p);
 738 |   if(error){ atomic_store(&p->running,0); pthread_join(p->helper,NULL); return error; }
 739 |   error=pthread_create(&p->reducer,NULL,reduce_run,p);
 740 |   if(error){ atomic_store(&p->running,0); pthread_join(p->thread,NULL); pthread_join(p->helper,NULL); return error; }
 741 |   return 0;
 742 | }
 743 | // ../design/algorithm-sources.md#collective-arithmetic-and-asynchronous-reduction
 744 | void mesh_pages_stop(mesh_pages *p){
 745 |   if(!atomic_exchange(&p->running,0)) return;
 746 |   pthread_join(p->thread,NULL);
 747 |   pthread_join(p->reducer,NULL);
 748 |   pthread_join(p->helper,NULL);
 749 | }
```

**M, lines 750–779.** R: recover after link failure. X: restoration of duplicated algorithm state. The two-second ACK wait can expire while flying remains, after which storage/state are cleared; safety requires proven cessation of device access, not elapsed time. This caller never invokes recovery on its failed-NFE path.

```text
 750 | 
 751 | int mesh_pages_recover(mesh_pages *p){
 752 |   if(atomic_load_explicit(&p->running,memory_order_acquire)) return EBUSY;
 753 |   if(atomic_load_explicit(&p->status,memory_order_acquire)==-ESTALE) return ESTALE;
 754 |   uint64_t began=clock_ns();
 755 |   while(p->flying && clock_ns()-began<2000000000ull){ p->now=clock_ns(); acknowledge(p); sched_yield(); }
 756 |   for(size_t i=0;i<p->count;i++){
 757 |     struct slot *s=&p->slots[i];
 758 |     if(s->spec.receive) for(uint32_t j=0;j<s->spec.pages;j++) if(s->table[j]!=ABSENT){ release_page(p,s->table[j]); s->table[j]=ABSENT; }
 759 |     if(s->spec.storage) for(uint32_t j=0;j<s->spec.pages;j++){
 760 |       s->table[j]=p->M->pool+(uint32_t)(s->base+j);
 761 |       if(s->spec.storage==i+1){ memset(payload_at(p,s->table[j]),0,mesh_pages_payload(p)); p->owner[s->base+j]=UNUSED; }
 762 |     }
 763 |     memset(s->stamp,0,s->spec.pages*sizeof *s->stamp);
 764 |     memset(s->inflight,0,s->spec.pages);
 765 |     for(size_t w=0;w<s->words;w++){ atomic_store(&s->publish[0][w],0); atomic_store(&s->publish[1][w],0); s->pending[w]=0; }
 766 |     atomic_store(&s->publishing[0],0); atomic_store(&s->publishing[1],0);
 767 |     for(int k=0;k<2;k++){ atomic_store(&s->fill_generation[k],0); atomic_store(&s->fill_count[k],0); }
 768 |     atomic_store(&s->highest,0); atomic_store(&s->complete,0); atomic_store(&s->producible,0);
 769 |     s->flying=0; s->released=0; s->fault_generation=0; s->fault_have=0;
 770 |     memset(s->uses,0,s->spec.pages);
 771 |     for(size_t i=0;i<DIGESTS;i++){ struct digest *d=&s->digest[i]; d->pending=0; d->count=0; atomic_store(&d->hash,0); atomic_store(&d->generation,0); }
 772 |   }
 773 |   flush_later(p);
 774 |   atomic_store(&p->work_tail,atomic_load(&p->work_head));
 775 |   p->flying=0; p->incarnation++;
 776 |   atomic_store_explicit(&p->status,0,memory_order_release);
 777 |   return 0;
 778 | }
 779 | 
```

**O, lines 780–805.** Shutdown/description observations are not computation dependencies. Remove diagnostics for deleted state with that state. A settled predicate must not restore a second readiness authority.

```text
 780 | int mesh_pages_settled(const mesh_pages *p){
 781 |   if(p->flying || p->later_count || atomic_load_explicit(&p->work_head,memory_order_acquire)!=atomic_load_explicit(&p->work_tail,memory_order_acquire)) return 0;
 782 |   for(size_t i=0;i<p->count;i++){
 783 |     const struct slot *s=&p->slots[i];
 784 |     for(uint32_t j=0;j<s->spec.pages;j++) if(__atomic_load_n(s->stamp+j,__ATOMIC_ACQUIRE)&WRITING) return 0;
 785 |     for(size_t w=0;w<s->words;w++)
 786 |       if(s->pending[w] || atomic_load_explicit(&s->publish[0][w],memory_order_acquire) || atomic_load_explicit(&s->publish[1][w],memory_order_acquire)) return 0;
 787 |   }
 788 |   return 1;
 789 | }
 790 | size_t mesh_pages_describe(const mesh_pages *p, char *out, size_t bytes){
 791 |   size_t n=(size_t)snprintf(out,bytes,"status=%d incarnation=%llu flying=%zu integrity=%llu stale=%llu duplicates=%llu overwrites=%llu faults=%llu foreign_epoch=%llu\n",
 792 |     atomic_load(&p->status),(unsigned long long)p->incarnation,p->flying,(unsigned long long)p->integrity,(unsigned long long)p->stale,
 793 |     (unsigned long long)p->duplicates,(unsigned long long)p->overwrites,(unsigned long long)p->faults,(unsigned long long)p->foreign_epoch);
 794 |   for(size_t i=0;i<p->count && n<bytes;i++){
 795 |     const struct slot *s=&p->slots[i];
 796 |     uint32_t present=0, writing=0;
 797 |     for(uint32_t j=0;j<s->spec.pages;j++){ present+=s->table[j]!=ABSENT; writing+=(__atomic_load_n(s->stamp+j,__ATOMIC_ACQUIRE)&WRITING)!=0; }
 798 |     n+=(size_t)snprintf(out+n,bytes-n,"slot %zu sid=%u %s peer=%u pages=%u producible=%llu highest=%llu complete=%llu present=%u writing=%u flying=%u released=%llu fills=[%llu:%llu %llu:%llu]\n",
 799 |       i,s->spec.sid,s->spec.receive?"rx":s->transported?"tx":"local",s->spec.peer,s->spec.pages,
 800 |       (unsigned long long)atomic_load(&s->producible),(unsigned long long)atomic_load(&s->highest),(unsigned long long)atomic_load(&s->complete),present,writing,s->flying,
 801 |       (unsigned long long)s->released,(unsigned long long)atomic_load(&s->fill_generation[0]),(unsigned long long)atomic_load(&s->fill_count[0]),
 802 |       (unsigned long long)atomic_load(&s->fill_generation[1]),(unsigned long long)atomic_load(&s->fill_count[1]));
 803 |   }
 804 |   return n;
 805 | }
```

### metal-microbench/reduce_scatter.swift — 751 lines

**A, lines 1–28.** Imports and configuration-time slot/pipeline realization. Slot metadata inherits canonical-header restrictions; no model invocation may build these.

```text
   1 | // docs/amdahl_superiority.md
   2 | 
   3 | import Foundation
   4 | import CoreML
   5 | import Metal
   6 | import MetalPerformanceShaders
   7 | 
   8 | #if MMB_MESH
   9 | func slotSpec(_ sid: UInt32, _ pages: Int, _ peer: UInt16, _ receive: Bool, _ depends: [Int], pagewise: Bool = false, lags: [UInt32] = []) -> mesh_pages_slot {
  10 |     var spec = mesh_pages_slot()
  11 |     spec.sid = sid; spec.pages = UInt32(pages); spec.peer = peer; spec.receive = receive ? 1 : 0; spec.depends = UInt8(depends.count); spec.pagewise = pagewise ? 1 : 0
  12 |     withUnsafeMutableBytes(of: &spec.dependency) { bytes in
  13 |         let list = bytes.bindMemory(to: UInt32.self)
  14 |         for (i, d) in depends.enumerated() { list[i] = UInt32(d) }
  15 |     }
  16 |     withUnsafeMutableBytes(of: &spec.lag) { bytes in
  17 |         let list = bytes.bindMemory(to: UInt32.self)
  18 |         for (i, lag) in lags.enumerated() { list[i] = lag }
  19 |     }
  20 |     return spec
  21 | }
  22 | 
  23 | func pagedPipeline(_ source: String, _ name: String, pages: Bool = false) -> MTLComputePipelineState {
  24 |     let constants = MTLFunctionConstantValues()
  25 |     var value = pages
  26 |     constants.setConstantValue(&value, type: .bool, index: 0)
  27 |     let library = try! device.makeLibrary(source: source, options: nil)
  28 |     return try! device.makeComputePipelineState(function: library.makeFunction(name: name, constantValues: constants))
```

**M, lines 29–91.** A: page-indexed arithmetic and on-chip reductions. X under the operand mandate: gamma/scalar/token/embedding buffers not guaranteed actual table pages. Post-reduce normalization currently follows already-sent FP16 output; required normalization belongs before reduced-row publication/send.

```text
  29 | }
  30 | let pagedNormSource = """
  31 | #include <metal_stdlib>
  32 | using namespace metal;
  33 | constant bool normPages [[function_constant(0)]];
  34 | kernel void rms_norm_add_scale_paged(device const uint *table [[buffer(0)]], device const uchar *receive [[buffer(1)]], device uchar *transmit [[buffer(2)]],
  35 |                                      device const half *gamma [[buffer(3)]], device const float *scalar [[buffer(4)]],
  36 |                                      constant ulong *geometry [[buffer(5)]], constant uint &firstRow [[buffer(6)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
  37 |                                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  38 |     threadgroup float partial[8];
  39 |     ulong halves = geometry[0], extent = geometry[1], stride = geometry[2], off = geometry[3], columns = geometry[8];
  40 |     row += firstRow;
  41 |     ulong input = geometry[row / geometry[10] == geometry[11] ? 12 : 13] + (row % geometry[10]) * halves;
  42 |     float ss = 0;
  43 |     for (uint h = 0; h < halves; h++) {
  44 |         ulong p = table[input + h];
  45 |         device const half *x = (device const half *)(p < geometry[6] ? receive + p * stride + off - geometry[4] : transmit + p * stride + off - geometry[5]);
  46 |         for (uint i = t; i < extent; i += 256) { float v = x[i]; ss += v * v; }
  47 |     }
  48 |     ss = simd_sum(ss); if (lane == 0) partial[sg] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);
  49 |     float total = 0; for (uint i = 0; i < 8; i++) total += partial[i];
  50 |     float scale = rsqrt(total / float(columns) + 1e-6f), ls = scalar[0];
  51 |     for (uint h = 0; h < halves; h++) {
  52 |         ulong p = table[input + h];
  53 |         device const half *x = (device const half *)(p < geometry[6] ? receive + p * stride + off - geometry[4] : transmit + p * stride + off - geometry[5]);
  54 |         device half *o = (device half *)(transmit + table[geometry[7] + row * halves + h] * stride + off - geometry[5]);
  55 |         device const half *r = (device const half *)(transmit + table[geometry[9] + row * halves + h] * stride + off - geometry[5]);
  56 |         for (uint i = t; i < extent; i += 256) { uint c = h * extent + i; o[i] = half((float(x[i]) * scale * float(gamma[c]) + float(r[i])) * ls); }
  57 |     }
  58 | }
  59 | kernel void embed_paged(device const half *table [[buffer(0)]], device const uint *tokens [[buffer(1)]], device uchar *transmit [[buffer(2)]],
  60 |                         constant ulong *geometry [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  61 |     ulong halves = geometry[0], extent = geometry[1], stride = geometry[2], off = geometry[3], origin = geometry[4], first = geometry[5], hidden = geometry[6];
  62 |     float scale = as_type<float>(uint(geometry[7]));
  63 |     device const half *source = table + tokens[row] * hidden;
  64 |     for (uint h = 0; h < halves; h++) {
  65 |         device half *o = (device half *)(transmit + (first + row * halves + h) * stride + off - origin);
  66 |         for (uint i = t; i < extent; i += 256) o[i] = half(float(source[h * extent + i]) * scale);
  67 |     }
  68 | }
  69 | kernel void rms_norm_paged(device const uchar *transmit [[buffer(0)]], device const half *gamma [[buffer(1)]], device half *y [[buffer(2)]],
  70 |                            constant ulong *geometry [[buffer(3)]], uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
  71 |                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  72 |     threadgroup float partial[8];
  73 |     ulong halves = geometry[0], extent = geometry[1], stride = geometry[2], off = geometry[3], origin = geometry[4], first = geometry[5];
  74 |     ulong columns = geometry[6], rowStride = geometry[7], columnStride = geometry[8], base = geometry[9];
  75 |     float ss = 0;
  76 |     for (uint h = 0; h < halves; h++) {
  77 |         device const half *x = (device const half *)(transmit + (first + (base + row) * halves + h) * stride + off - origin);
  78 |         for (uint i = t; i < extent; i += 256) { float v = x[i]; ss += v * v; }
  79 |     }
  80 |     ss = simd_sum(ss); if (lane == 0) partial[sg] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);
  81 |     float total = 0; for (uint i = 0; i < 8; i++) total += partial[i];
  82 |     float scale = rsqrt(total / float(columns) + 1e-6f);
  83 |     for (uint h = 0; h < halves; h++) {
  84 |         device const half *x = (device const half *)(transmit + (first + (base + row) * halves + h) * stride + off - origin);
  85 |         for (uint i = t; i < extent; i += 256) { uint c = h * extent + i; ulong at = normPages ? (c / geometry[10]) * geometry[11] + row * rowStride + (c % geometry[10]) * columnStride : row * rowStride + c * columnStride; y[at] = half(float(x[i]) * scale * float(gamma[c])); }
  86 |     }
  87 | }
  88 | """
  89 | 
  90 | struct ConfiguredNumericalFunction {
  91 |     let configuration: ParameterConfiguration
```

**M, lines 92–134.** A: pre-invocation model ownership/layout realization. X: separately allocated unit, weights, norm/gamma/scale/embedding operands; loading before invocation does not grant storage exemption. Environment/timing setup is O.

```text
  92 |     let norm, gamma, scale: MTLBuffer
  93 | }
  94 | 
  95 | // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
  96 | func runReduceScatter(deployment: NumericalConfiguration, file: ModelFile) async {
  97 |     let env = ProcessInfo.processInfo.environment
  98 |     let collective = deployment.reduction!
  99 |     let functions = deployment.functions.filter { [.attention, .ffn].contains($0.kind) }
 100 |     let head = deployment.functions.last.flatMap { $0.kind == .vocabulary ? $0 : nil }
 101 |     precondition(!functions.isEmpty && functions.count + (head == nil ? 0 : 1) == deployment.functions.count)
 102 |     collective.validate(backend: functions[0].backend)
 103 |     precondition(collective.layout == .rows && collective.partialElement == .float16 && collective.accumulatorElement == .float32)
 104 |     let rows = functions[0].rows, hidden = HIDDEN, columns = HIDDEN, vocabulary = VOCAB
 105 |     for function in functions {
 106 |         precondition(function.rows == rows && rows % 2 == 0)
 107 |         function.validate(hidden: hidden, intermediate: SHARED_INT, vocabulary: vocabulary, heads: SLIDE_H)
 108 |     }
 109 |     if let head { head.validate(hidden: hidden, intermediate: SHARED_INT, vocabulary: vocabulary, heads: SLIDE_H); precondition(head.rows == 1 && head.start == 0 && head.width == vocabulary) }
 110 |     let inflight = collective.versions, configuredCount = functions.count * collective.versions
 111 |     let local = collective.participants.firstIndex(of: deployment.participant)!, remote = 1 - local
 112 |     let blockRows = rows / 2
 113 |     let measured = Int(env["LM_REFERENCE_STEPS"] ?? env["LM_BENCH_STEPS"] ?? "20")!, warmup = Int(env["LM_REFERENCE_WARMUP"] ?? "4")!
 114 |     let extentColumns = collective.extentColumns ?? columns / 2
 115 |     precondition(extentColumns > 0 && columns % extentColumns == 0 && extentColumns % 8 == 0)
 116 |     let halves = columns / extentColumns, blockPages = blockRows * halves
 117 |     let unit = emptyFloat(1); unit.contents().assumingMemoryBound(to: Float.self)[0] = 1
 118 |     let model = functions.map { configuration -> ConfiguredNumericalFunction in
 119 |         let prefix = "blk.\(configuration.layer).", attention = configuration.kind == .attention
 120 |         return ConfiguredNumericalFunction(configuration: configuration,
 121 |             norm: try! file.loadHalf(prefix + (attention ? "attn_norm.weight" : "ffn_norm.weight"), device: device),
 122 |             gamma: try! file.loadHalf(prefix + (attention ? "post_attention_norm.weight" : "post_ffw_norm.weight"), device: device),
 123 |             scale: attention ? unit : try! file.makeMetalBuffer(prefix + "layer_output_scale.weight", device: device))
 124 |     }
 125 |     let embedding = try! file.makeMatrixBuffer("token_embd.weight", device: device)
 126 |     let outputNorm = head == nil ? nil : try! file.loadHalf("output_norm.weight", device: device)
 127 |     let tokens = (0..<rows).map { UInt32((2 + $0 * 7919) % vocabulary) }
 128 | 
 129 |     let mesh = mesh_context()!
 130 |     if mesh_attach(mesh, env["MESH_NAME"] ?? "/mesh0") != 0 { fail("mesh attach: \(String(cString: strerror(errno))) (\(errno))") }
 131 |     var scope = mesh_scope()
 132 |     scope.epoch = mesh_epoch(high: collective.epoch, low: 1)
 133 |     withUnsafeMutableBytes(of: &scope.plan) { $0.storeBytes(of: UInt64(rows * 1000 + functions.count), as: UInt64.self) }
 134 |     var geometry = mesh_metal_rows()
```

**M, lines 135–169.** A: static slot indexing and numerical dimensions. D: rotating digestPage convention. Kind is immutable layout metadata, not automatically a stage; however its represented values omit accumulator/index pages.

```text
 135 |     precondition(mesh_metal_row_layout(mesh, scope, 1, extentColumns * 2, 64, &geometry) == 0 && geometry.rows_per_page == 1)
 136 |     let payloadOffset = Int(geometry.offset), rowBytes = extentColumns * 2
 137 |     enum Kind: Int, CaseIterable { case partialOut = 0, partialLocal, partialIn, reducedOut, reducedIn, digestOut, digestIn, activation, up, query, key, value, attended }
 138 |     let kinds = Kind.allCases.count, perFamily = functions.count * (kinds + 2) + 3
 139 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 140 |     func hiddenSlot(_ f: Int, _ function: Int) -> Int { f * perFamily + function * (kinds + 2) }
 141 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 142 |     func slot(_ f: Int, _ function: Int, _ kind: Kind) -> Int { hiddenSlot(f, function) + 1 + kind.rawValue }
 143 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 144 |     func normalizedSlot(_ f: Int, _ function: Int) -> Int { hiddenSlot(f, function) + 1 + kinds }
 145 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 146 |     func embeddingSlot(_ f: Int) -> Int { f * perFamily + functions.count * (kinds + 2) }
 147 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 148 |     func resultSlot(_ f: Int) -> Int { embeddingSlot(f) + 1 }
 149 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 150 |     func headInputSlot(_ f: Int) -> Int { resultSlot(f) + 1 }
 151 |     let resultPages = (vocabulary + extentColumns - 1) / extentColumns
 152 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 153 |     func pageChannels(_ configuration: ParameterConfiguration) -> Int { try! ParameterArtifact.read(configuration.artifact!).inputPageChannels ?? 1 }
 154 |     // ../dox/mesh/design/algorithm-sources.md#endpoint-checking
 155 |     func digestPage(_ g: UInt64) -> Int { Int((g - 1) / UInt64(inflight)) & 1 }
 156 |     let peer = UInt16(collective.participants[remote]), here = UInt16(MESH_PAGES_LOCAL), generationStride = UInt32(max(2, inflight))
 157 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 158 |     let attentionChannels = functions.map { configuration -> (query: Int, key: Int) in
 159 |         guard configuration.kind == .attention && [.mps, .tensor].contains(configuration.backend) else { return (1, 1) }
 160 |         let prefix = "blk.\(configuration.layer)."
 161 |         let dimension = try! file.tensor(prefix + "attn_q_norm.weight").shape[0]
 162 |         let totalKV = try! file.tensor(prefix + "attn_k.weight").shape[1] / dimension
 163 |         let group = SLIDE_H / totalKV, heads = configuration.coordinates
 164 |         return (heads.count * dimension, ((heads.upperBound - 1) / group + 1 - heads.lowerBound / group) * dimension)
 165 |     }
 166 |     var slots: [mesh_pages_slot] = []
 167 |     for f in 0..<inflight {
 168 |         let sid = UInt32(32000 + f * (functions.count * 12 + 3))
 169 |         for (index, configuration) in functions.enumerated() {
```

**M, lines 170–195.** A: static declared functions and row dependencies. X: fabricated previous-reduce lag cycle and digest storage dependencies. Hidden slot omits received-reduce input later consumed manually. Logical dependency declarations and actual bound maps must be the same read relation.

```text
 170 |             let base = sid + UInt32(index * 12)
 171 |             let input = index == 0 ? embeddingSlot(f) : hiddenSlot(f, index - 1)
 172 |             let precedingReduce = slot(f, index == 0 ? functions.count - 1 : index - 1, .reducedOut)
 173 |             let native = [.ane, .cpu, .gpu, .all].contains(configuration.backend)
 174 |             let pagedFFN = configuration.kind == .ffn && configuration.backend == .mps
 175 |             let pagedInput = native || configuration.backend == .mps || configuration.kind == .attention && configuration.backend == .tensor
 176 |             let pagedAttention = configuration.kind == .attention && [.mps, .tensor].contains(configuration.backend)
 177 |             let operand = pagedFFN ? slot(f, index, .activation) : pagedAttention ? slot(f, index, .attended) : pagedInput ? normalizedSlot(f, index) : input
 178 |             slots.append(slotSpec(base, 2 * blockPages, here, false, [slot(f, index, .reducedOut), input], pagewise: true))
 179 |             slots.append(slotSpec(base + 1, blockPages, peer, false, [operand]))
 180 |             slots.append(slotSpec(base + 2, blockPages, here, false, [operand]))
 181 |             slots.append(slotSpec(base + 1, blockPages, peer, true, [precedingReduce], pagewise: true, lags: [index == 0 ? generationStride : 0]))
 182 |             slots.append(slotSpec(base + 3, blockPages, peer, false, [slot(f, index, .partialLocal), slot(f, index, .partialIn)], pagewise: true))
 183 |             slots.append(slotSpec(base + 3, blockPages, peer, true, [slot(f, index, .partialOut)], pagewise: true))
 184 |             slots.append(slotSpec(base + 4, 2, peer, false, [slot(f, index, .digestIn)], lags: [generationStride]))
 185 |             slots.append(slotSpec(base + 4, 2, peer, true, [slot(f, index, .digestOut)]))
 186 |             let activationPages = pagedFFN ? rows / configuration.nativeRows * configuration.width : 1
 187 |             slots.append(slotSpec(base + 6, activationPages, here, false, pagedFFN ? [normalizedSlot(f, index), slot(f, index, .up)] : []))
 188 |             slots.append(slotSpec(base + 7, activationPages, here, false, pagedFFN ? [normalizedSlot(f, index)] : []))
 189 |             let channels = attentionChannels[index]
 190 |             slots.append(slotSpec(base + 8, channels.query, here, false, pagedAttention ? [normalizedSlot(f, index)] : []))
 191 |             slots.append(slotSpec(base + 9, channels.key, here, false, pagedAttention ? [normalizedSlot(f, index)] : []))
 192 |             slots.append(slotSpec(base + 10, channels.key, here, false, pagedAttention ? [normalizedSlot(f, index)] : []))
 193 |             slots.append(slotSpec(base + 11, channels.query, here, false, pagedAttention ? [slot(f, index, .query), slot(f, index, .key), slot(f, index, .value)] : []))
 194 |             let pages = native ? rows / configuration.nativeRows * (columns / pageChannels(configuration))
 195 |                 : pagedInput ? (configuration.kind == .attention ? columns : rows / configuration.nativeRows * columns) : 1
```

**M, lines 196–214.** A: configured page allocation and physical reuse intentions. X: alias-root storage supported by a separate owner array. Complete lifetime proof for all intermediates remains absent.

```text
 196 |             slots.append(slotSpec(base + 5, pages, here, false, pagedInput ? [input] : []))
 197 |         }
 198 |         slots.append(slotSpec(sid + UInt32(functions.count * 12), 2 * blockPages, here, false, []))
 199 |         slots.append(slotSpec(sid + UInt32(functions.count * 12 + 1), head == nil ? 1 : resultPages, here, false, head == nil ? [] : [headInputSlot(f)]))
 200 |         slots.append(slotSpec(sid + UInt32(functions.count * 12 + 2), head == nil ? 1 : halves, here, false, head == nil ? [] : [hiddenSlot(f, functions.count - 1)]))
 201 |     }
 202 |     for f in 0..<inflight {
 203 |         let inputs = functions.indices.map { normalizedSlot(f, $0) }.filter { slots[$0].pages > 1 }
 204 |         if inputs.count > 1, let root = inputs.max(by: { slots[$0].pages < slots[$1].pages }) {
 205 |             for input in inputs { slots[input].storage = UInt32(root + 1) }
 206 |         }
 207 |     }
 208 |     let policy = mesh_pages_policy(fault_seed: collective.faultSeed ?? 0, fault_period: UInt32(collective.faultPeriod ?? 0))
 209 |     let program = slots.withUnsafeBufferPointer { buffer in
 210 |         withUnsafeBytes(of: &scope.plan) { plan in
 211 |             mesh_pages_compile(mesh, scope.epoch, plan.baseAddress!.assumingMemoryBound(to: UInt8.self), buffer.baseAddress, buffer.count, generationStride, policy)
 212 |         }
 213 |     }
 214 |     guard let program else { fail("mesh pages compile: \(String(cString: strerror(errno)))") }
```

**M, lines 215–228.** X: MESH_REDUCE_MATERIALIZED and group=halves instead of page-progress accumulation/index and normalization before send. B: actual RDMA pool mapping; no-copy does not independently authorize an aggregate buffer abstraction.

```text
 215 |     let pageStride = Int(mesh_pages_header(program)) + Int(mesh_pages_payload(program)), padding = payloadOffset - Int(mesh_pages_header(program))
 216 |     var reduces = (0..<configuredCount).map { L -> mesh_pages_reduce in
 217 |         let f = L / functions.count, index = L % functions.count
 218 |         var r = mesh_pages_reduce()
 219 |         r.output = UInt32(slot(f, index, .reducedOut)); r.inputs = 2; r.kind = UInt8(MESH_REDUCE_MATERIALIZED)
 220 |         withUnsafeMutableBytes(of: &r.input) { let list = $0.bindMemory(to: UInt32.self); list[0] = UInt32(slot(f, index, .partialLocal)); list[1] = UInt32(slot(f, index, .partialIn)) }
 221 |         r.group = UInt32(halves); r.offset = UInt32(padding); r.bytes = UInt32(rowBytes)
 222 |         return r
 223 |     }
 224 |     precondition(reduces.withUnsafeBufferPointer { mesh_pages_reduces(program, $0.baseAddress, $0.count) } == 0)
 225 |     var transmitLayout = mesh_metal_layout(), receiveLayout = mesh_metal_layout()
 226 |     let transmit = mesh_metal_transmit_pool(device, mesh, &transmitLayout)!
 227 |     let receive = mesh_metal_receive_pool(device, mesh, &receiveLayout)!
 228 |     let poolPages = Int(transmitLayout.origin) / pageStride
```

**M, lines 229–253.** R/A: direct row stamps/entries and page address calculations. D: producible/hash wrappers reading extra runtime state. Precomputed contiguous views require actual stable configured physical mapping; arbitrary received pages require literal lookup.

```text
 229 |     func entries(_ slot: Int) -> UnsafePointer<UInt32> { mesh_pages_entries(program, UInt32(slot))! }
 230 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 231 |     func stamped(_ slot: Int, _ first: Int, _ count: Int, _ g: UInt64) -> Bool {
 232 |         let stamps = mesh_pages_stamps(program, UInt32(slot))!
 233 |         let table = entries(slot)
 234 |         for page in first..<(first + count) {
 235 |             if mesh_pages_stamp(stamps, UInt32(page)) != g || mesh_pages_entry(table, UInt32(page)) == UInt32.max { return false }
 236 |         }
 237 |         return true
 238 |     }
 239 |     func producible(_ slot: Int, _ g: UInt64) -> Bool { mesh_pages_producible(program, UInt32(slot)) >= g }
 240 |     func words(_ slot: Int, _ page: Int) -> UnsafeMutablePointer<UInt64> {
 241 |         mesh_pages_data(program, UInt32(slot), UInt32(page))!.advanced(by: padding).assumingMemoryBound(to: UInt64.self)
 242 |     }
 243 |     func hash(_ slot: Int, _ g: UInt64) -> UInt64? { var h: UInt64 = 0; return mesh_pages_digest(program, UInt32(slot), g, &h) == 1 ? h : nil }
 244 |     func pageWords(_ slot: Int, _ page: Int) -> UnsafeMutablePointer<Float16> {
 245 |         transmit.contents().advanced(by: Int(entries(slot)[page]) * pageStride + payloadOffset - Int(transmitLayout.origin)).assumingMemoryBound(to: Float16.self)
 246 |     }
 247 |     func producerView(_ f: Int, _ index: Int, block: Int, half: Int) -> MatrixView {
 248 |         let first = Int(entries(slot(f, index, block == local ? .partialLocal : .partialOut))[0])
 249 |         return MatrixView(transmit, rows: blockRows, columns: extentColumns, rowStride: halves * pageStride / 2, columnStride: 1,
 250 |             offset: (first + half) * pageStride + payloadOffset - Int(transmitLayout.origin), type: .float16)
 251 |     }
 252 |     let prenormPipeline = pagedPipeline(pagedNormSource, "rms_norm_paged")
 253 |     let prenormPagesPipeline = pagedPipeline(pagedNormSource, "rms_norm_paged", pages: true)
```

**M, lines 254–288.** A: configure numerical launch binding before invocation. X: separate shared geometry/token buffers under literal page-storage mandate; values must inhabit configured sendable pages. Direct output-page writes and completed GPU callbacks are allowed mechanisms.

```text
 254 |     func prenorm(_ source: Int, rows region: Range<Int>, gamma: MTLBuffer, output: MatrixView, pageColumns: Int = 0) -> MatrixStep {
 255 |         let pipeline = pageColumns == 0 ? prenormPipeline : prenormPagesPipeline
 256 |         let buffer = device.makeBuffer(length: 12 * 8, options: .storageModeShared)!
 257 |         let g = buffer.contents().assumingMemoryBound(to: UInt64.self)
 258 |         g[0] = UInt64(halves); g[1] = UInt64(extentColumns); g[2] = UInt64(pageStride); g[3] = UInt64(payloadOffset); g[4] = transmitLayout.origin
 259 |         g[5] = UInt64(entries(source)[0]); g[6] = UInt64(columns); g[7] = UInt64(output.rowStride); g[8] = UInt64(output.columnStride); g[9] = UInt64(region.lowerBound)
 260 |         g[10] = UInt64(pageColumns); g[11] = UInt64(pageStride / 2)
 261 |         return { cb in
 262 |             let encoder = cb.makeComputeCommandEncoder()!
 263 |             encoder.setComputePipelineState(pipeline)
 264 |             encoder.setBuffer(transmit, offset: 0, index: 0); encoder.setBuffer(gamma, offset: 0, index: 1)
 265 |             encoder.setBuffer(output.buffer, offset: output.offset, index: 2); encoder.setBuffer(buffer, offset: 0, index: 3)
 266 |             encoder.dispatchThreadgroups(MTLSize(width: region.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
 267 |             encoder.endEncoding()
 268 |         }
 269 |     }
 270 |     let embedPipeline = pagedPipeline(pagedNormSource, "embed_paged")
 271 |     let tokenBuffer = device.makeBuffer(length: rows * 4, options: .storageModeShared)!
 272 |     tokenBuffer.contents().assumingMemoryBound(to: UInt32.self).update(from: tokens, count: rows)
 273 |     let embedGeometries = (0..<inflight).map { f -> MTLBuffer in
 274 |         let buffer = device.makeBuffer(length: 8 * 8, options: .storageModeShared)!
 275 |         let g = buffer.contents().assumingMemoryBound(to: UInt64.self)
 276 |         g[0] = UInt64(halves); g[1] = UInt64(extentColumns); g[2] = UInt64(pageStride); g[3] = UInt64(payloadOffset); g[4] = transmitLayout.origin
 277 |         g[5] = UInt64(entries(embeddingSlot(f))[0]); g[6] = UInt64(hidden); g[7] = UInt64(Float(hidden).squareRoot().bitPattern)
 278 |         return buffer
 279 |     }
 280 |     func embed(_ f: Int, _ cb: MTLCommandBuffer) {
 281 |         let encoder = cb.makeComputeCommandEncoder()!
 282 |         encoder.setComputePipelineState(embedPipeline)
 283 |         encoder.setBuffer(embedding, offset: 0, index: 0); encoder.setBuffer(tokenBuffer, offset: 0, index: 1)
 284 |         encoder.setBuffer(transmit, offset: 0, index: 2); encoder.setBuffer(embedGeometries[f], offset: 0, index: 3)
 285 |         encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
 286 |         encoder.endEncoding()
 287 |     }
 288 | 
```

**M, lines 289–295.** X: dense normalized fallback. A: Publication is a static output row range, Call/Prediction/BoundFunction are immutable configured numerical functions. Their names alone do not prove per-job state, but duplicated maps and publication lists must describe one canonical relation.

```text
 289 |     let normalized = (0..<inflight).map { _ -> MTLBuffer? in functions.allSatisfy {
 290 |         [.ane, .cpu, .gpu, .all].contains($0.backend) || $0.backend == .mps || $0.kind == .attention && $0.backend == .tensor
 291 |     } ? nil : emptyHalf(rows * hidden) }
 292 |     struct Publication { let slot: Int; let first, count: UInt32 }
 293 |     struct Call { let rows: Range<Int>; let step: MatrixStep; let publications: [Publication] }
 294 |     struct Prediction { let input: Publication; let outputs: [Publication]; let function: CoreMLFunction }
 295 |     struct BoundFunction { let calls: [Call]; let predictions: [Prediction] }
```

**M, lines 296–382.** A: configure existing numerical functions, layout and direct page outputs. X: dense fallback paths. Combined attention/FFN calls publish intermediates only at enclosing command completion; they do not expose each configured intermediate as independently ready. Fixed nativeRows may be required backend geometry, not permission for an extra streaming row-group scheduler.

```text
 296 |     let loadStart = Date()
 297 |     func partialOutputs(_ f: Int, _ index: Int, rows region: Range<Int>) -> ([MatrixOutput], [Publication]) {
 298 |         var outputs: [MatrixOutput] = [], ranges: [Publication] = []
 299 |         for block in 0..<2 {
 300 |             let first = max(region.lowerBound, block * blockRows), end = min(region.upperBound, (block + 1) * blockRows)
 301 |             if first >= end { continue }
 302 |             for half in 0..<halves {
 303 |                 outputs.append(MatrixOutput(producerView(f, index, block: block, half: half).slice(rows: (first - block * blockRows)..<(end - block * blockRows), columns: 0..<extentColumns),
 304 |                     row: first - region.lowerBound, column: half * extentColumns))
 305 |             }
 306 |             ranges.append(Publication(slot: slot(f, index, block == local ? .partialLocal : .partialOut), first: UInt32((first - block * blockRows) * halves), count: UInt32((end - first) * halves)))
 307 |         }
 308 |         return (outputs, ranges)
 309 |     }
 310 |     let bound: [[BoundFunction]] = model.enumerated().map { (s, function) in
 311 |         let configuration = function.configuration, index = s, source = s - 1
 312 |         let native = [.ane, .cpu, .gpu, .all].contains(configuration.backend)
 313 |         let coreML = native ? try! CoreMLParameter(configuration) : nil
 314 |         let metal = native || configuration.kind == .attention ? nil : try! realizeMetalParameter(configuration, file: file)
 315 |         let attention = configuration.kind == .attention ? try! realizeMetalAttentionRows(configuration, file: file) : nil
 316 |         return (0..<inflight).map { f in
 317 |             let hiddenIn = s == 0 ? embeddingSlot(f) : hiddenSlot(f, source)
 318 |             if let attention {
 319 |                 let paged = configuration.backend == .mps || configuration.backend == .tensor
 320 |                 precondition(!paged || rows * 2 <= pageStride - payloadOffset)
 321 |                 let input = paged ? MatrixView(transmit, rows: rows, columns: hidden, rowStride: 1, columnStride: pageStride / 2,
 322 |                     offset: Int(entries(normalizedSlot(f, s))[0]) * pageStride + payloadOffset - Int(transmitLayout.origin))
 323 |                     : MatrixView(normalized[f]!, rows: rows, columns: hidden)
 324 |                 let (outputs, ranges) = partialOutputs(f, index, rows: 0..<rows)
 325 |                 let channels = attentionChannels[s]
 326 |                 let operands: [(Kind, Int)] = [(.query, channels.query), (.key, channels.key), (.value, channels.key), (.attended, channels.query)]
 327 |                 let views = paged ? operands.map { kind, width in
 328 |                     MatrixView(transmit, rows: rows, columns: width, rowStride: 1, columnStride: pageStride / 2,
 329 |                         offset: Int(entries(slot(f, s, kind))[0]) * pageStride + payloadOffset - Int(transmitLayout.origin))
 330 |                 } : []
 331 |                 let storage = paged ? AttentionOperands(query: views[0], key: views[1], value: views[2], attended: views[3]) : nil
 332 |                 let realized = attention(input, outputs, [0..<rows], storage)
 333 |                 let norm = prenorm(hiddenIn, rows: 0..<rows, gamma: function.norm, output: input, pageColumns: paged ? 1 : 0)
 334 |                 let publications = ranges + (paged ? [Publication(slot: normalizedSlot(f, s), first: 0, count: UInt32(columns))]
 335 |                     + operands.map { Publication(slot: slot(f, s, $0.0), first: 0, count: UInt32($0.1)) } : [])
 336 |                 let call = Call(rows: 0..<rows, step: { cb in norm(cb); realized.projections[0](cb); realized.attend(cb); for step in realized.outputs { step(cb) } }, publications: publications)
 337 |                 return BoundFunction(calls: [call], predictions: [])
 338 |             }
 339 |             let callRows = configuration.nativeRows
 340 |             let pagedFFN = configuration.backend == .mps
 341 |             precondition(!pagedFFN || callRows * 2 <= pageStride - payloadOffset)
 342 |             var predictions: [Prediction] = []
 343 |             let calls = stride(from: 0, to: rows, by: callRows).map { firstRow -> Call in
 344 |                 let region = firstRow..<(firstRow + callRows), channels = configuration.layout == .channels
 345 |                 let pageColumns = coreML?.inputShape[2] ?? 1
 346 |                 let pages = columns / pageColumns, firstPage = firstRow / callRows * pages
 347 |                 let input = native
 348 |                     ? MatrixView(transmit, rows: callRows, columns: pageColumns, rowStride: 1, columnStride: callRows,
 349 |                         offset: Int(entries(normalizedSlot(f, s))[firstPage]) * pageStride + payloadOffset - Int(transmitLayout.origin))
 350 |                     : pagedFFN ? MatrixView(transmit, rows: callRows, columns: hidden, rowStride: 1, columnStride: pageStride / 2,
 351 |                         offset: Int(entries(normalizedSlot(f, s))[firstPage]) * pageStride + payloadOffset - Int(transmitLayout.origin))
 352 |                     : MatrixView(normalized[f]!, rows: callRows, columns: hidden, rowStride: channels ? 1 : hidden, columnStride: channels ? callRows : 1, offset: firstRow * hidden * 2)
 353 |                 precondition(!native || (columns % pageColumns == 0 && callRows * pageColumns * 2 <= pageStride - payloadOffset))
 354 |                 let norm = prenorm(hiddenIn, rows: region, gamma: function.norm, output: input, pageColumns: native ? pageColumns : pagedFFN ? 1 : 0)
 355 |                 let (outputs, ranges) = partialOutputs(f, index, rows: region)
 356 |                 if let coreML {
 357 |                     let destination = Publication(slot: normalizedSlot(f, s), first: UInt32(firstPage), count: UInt32(pages))
 358 |                     let tensor = try! MLMultiArray(dataPointer: transmit.contents().advanced(by: input.offset),
 359 |                         shape: coreML.inputShape.map { NSNumber(value: $0) }, dataType: .float16,
 360 |                         strides: [pages * pageStride / 2, pageStride / 2, callRows, 1].map { NSNumber(value: $0) }, deallocator: nil)
 361 |                     predictions.append(Prediction(input: destination, outputs: ranges,
 362 |                         function: try! coreML.bind(input: tensor, buffers: [transmit], outputs: outputs, rows: callRows)))
 363 |                     return Call(rows: region, step: norm, publications: [destination])
 364 |                 }
 365 |                 var activation: MatrixView?, up: MatrixView?, publications = ranges
 366 |                 if pagedFFN {
 367 |                     let first = firstRow / callRows * configuration.width
 368 |                     activation = MatrixView(transmit, rows: callRows, columns: configuration.width, rowStride: 1, columnStride: pageStride / 2,
 369 |                         offset: Int(entries(slot(f, s, .activation))[first]) * pageStride + payloadOffset - Int(transmitLayout.origin))
 370 |                     up = MatrixView(transmit, rows: callRows, columns: configuration.width, rowStride: 1, columnStride: pageStride / 2,
 371 |                         offset: Int(entries(slot(f, s, .up))[first]) * pageStride + payloadOffset - Int(transmitLayout.origin))
 372 |                     publications += [Publication(slot: normalizedSlot(f, s), first: UInt32(firstPage), count: UInt32(pages)),
 373 |                         Publication(slot: slot(f, s, .activation), first: UInt32(first), count: UInt32(configuration.width)),
 374 |                         Publication(slot: slot(f, s, .up), first: UInt32(first), count: UInt32(configuration.width))]
 375 |                 }
 376 |                 let operation = metal!(input, outputs, activation, up)
 377 |                 return Call(rows: region, step: { cb in norm(cb); operation.prefix?(cb); for step in operation.outputs { step(cb) } }, publications: publications)
 378 |             }
 379 |             return BoundFunction(calls: calls, predictions: predictions)
 380 |         }
 381 |     }
 382 |     let lastFunction = functions.count - 1
```

**M, lines 383–432.** A: configured vocabulary and normalization functions with actual result pages. X: shared geometry buffers outside pages. Backend command descriptors are B; they must not store numerical readiness.

```text
 383 |     let headFunctions: [MatrixStep] = head.map { head in
 384 |         let bind = try! realizeMetalParameter(head, file: file)
 385 |         return (0..<inflight).map { f in
 386 |             let input = MatrixView(transmit, rows: 1, columns: hidden,
 387 |                 offset: Int(entries(headInputSlot(f))[0]) * pageStride + payloadOffset - Int(transmitLayout.origin),
 388 |                 pageColumns: extentColumns, pageStride: pageStride / 2)
 389 |             let normalize = prenorm(hiddenSlot(f, lastFunction), rows: (rows - 1)..<rows, gamma: outputNorm!, output: input, pageColumns: extentColumns)
 390 |             let outputs = (0..<resultPages).map { page -> MatrixOutput in
 391 |                 let first = page * extentColumns, count = min(extentColumns, vocabulary - first)
 392 |                 let view = MatrixView(transmit, rows: 1, columns: count,
 393 |                     offset: Int(entries(resultSlot(f))[page]) * pageStride + payloadOffset - Int(transmitLayout.origin))
 394 |                 return MatrixOutput(view, row: 0, column: first)
 395 |             }
 396 |             let project = bind(input, outputs, nil, nil)
 397 |             let softcaps: [MatrixStep] = (0..<resultPages).map { page in
 398 |                 let count = min(extentColumns, vocabulary - page * extentColumns)
 399 |                 let offset = Int(entries(resultSlot(f))[page]) * pageStride + payloadOffset - Int(transmitLayout.origin)
 400 |                 return { cb in
 401 |                     let encoder = cb.makeComputeCommandEncoder()!
 402 |                     encoder.setComputePipelineState(softcapPSO)
 403 |                     encoder.setBuffer(transmit, offset: offset, index: 0)
 404 |                     var n = UInt32(count), cap: Float = 30
 405 |                     encoder.setBytes(&n, length: 4, index: 1); encoder.setBytes(&cap, length: 4, index: 2)
 406 |                     dispatchPointwise(encoder, columns: count, rows: 1)
 407 |                     encoder.endEncoding()
 408 |                 }
 409 |             }
 410 |             return { cb in normalize(cb); project.prefix?(cb); for step in project.outputs { step(cb) }; for step in softcaps { step(cb) } }
 411 |         }
 412 |     } ?? []
 413 |     print("configured mesh participant \(deployment.participant): \(functions.count) functions, \(rows) rows, \(inflight) in flight, page-stamp normalization, bound in \(Int(Date().timeIntervalSince(loadStart))) s")
 414 | 
 415 |     let normPipeline = pagedPipeline(pagedNormSource, "rms_norm_add_scale_paged")
 416 |     let pageTable = mesh_metal_page_table(device, program)!
 417 |     var pageTableBytes = 0
 418 |     let tableBase = mesh_pages_table(program, &pageTableBytes)!
 419 |     func tableIndex(_ slot: Int) -> UInt64 { UInt64(tableBase.distance(to: entries(slot))) }
 420 |     let normGeometries = model.indices.map { s in
 421 |         let index = s
 422 |         return (0..<inflight).map { f -> MTLBuffer in
 423 |             let buffer = device.makeBuffer(length: 14 * 8, options: .storageModeShared)!
 424 |             let geo = buffer.contents().assumingMemoryBound(to: UInt64.self)
 425 |             geo[0] = UInt64(halves); geo[1] = UInt64(extentColumns); geo[2] = UInt64(pageStride); geo[3] = UInt64(payloadOffset)
 426 |             geo[4] = receiveLayout.origin; geo[5] = transmitLayout.origin; geo[6] = UInt64(poolPages); geo[7] = tableIndex(hiddenSlot(f, index)); geo[8] = UInt64(columns)
 427 |             geo[9] = tableIndex(s == 0 ? embeddingSlot(f) : hiddenSlot(f, s - 1)); geo[10] = UInt64(blockRows); geo[11] = UInt64(local)
 428 |             geo[12] = tableIndex(slot(f, index, .reducedOut)); geo[13] = tableIndex(slot(f, index, .reducedIn))
 429 |             return buffer
 430 |         }
 431 |     }
 432 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
```

**M, lines 433–476.** A: static function input/output maps. D: duplicate publication maps beside slot dependency graph. Head requires every hidden row although numerical head reads only last row; that extra dependency is not justified by head arithmetic and must not cover a missing lifetime proof.

```text
 433 |     func bindMaps(_ inputs: [mesh_pages_map], _ outputs: [mesh_pages_map], rows: Int = 1) -> OpaquePointer {
 434 |         inputs.withUnsafeBufferPointer { inputs in
 435 |             outputs.withUnsafeBufferPointer { outputs in
 436 |                 mesh_pages_bind(program, mesh_pages_function_spec(input: inputs.baseAddress, output: outputs.baseAddress,
 437 |                     inputs: UInt32(inputs.count), outputs: UInt32(outputs.count), rows: UInt32(rows)))!
 438 |             }
 439 |         }
 440 |     }
 441 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 442 |     func bindPublication(_ input: Publication, _ outputs: [Publication]) -> OpaquePointer {
 443 |         bindMaps([mesh_pages_map(slot: UInt32(input.slot), first: input.first, count: input.count, stride: 0, lag: 0)],
 444 |             outputs.map { mesh_pages_map(slot: UInt32($0.slot), first: $0.first, count: $0.count, stride: 0, lag: 0) })
 445 |     }
 446 |     let callMaps = bound.enumerated().map { s, families in
 447 |         families.enumerated().map { f, function in
 448 |             function.calls.map { call in
 449 |                 bindPublication(Publication(slot: s == 0 ? embeddingSlot(f) : hiddenSlot(f, s - 1),
 450 |                     first: UInt32(call.rows.lowerBound * halves), count: UInt32(call.rows.count * halves)), call.publications)
 451 |             }
 452 |         }
 453 |     }
 454 |     let predictionMaps = bound.map { families in
 455 |         families.map { function in function.predictions.map { bindPublication($0.input, $0.outputs) } }
 456 |     }
 457 |     let headMaps = headFunctions.indices.map { f in
 458 |         bindPublication(Publication(slot: hiddenSlot(f, lastFunction), first: 0, count: UInt32(rows * halves)),
 459 |             [Publication(slot: headInputSlot(f), first: 0, count: UInt32(halves)),
 460 |              Publication(slot: resultSlot(f), first: 0, count: UInt32(resultPages))])
 461 |     }
 462 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 463 |     let normalizationMaps: [[[OpaquePointer]]] = functions.indices.map { s in
 464 |         (0..<inflight).map { f in
 465 |             (0..<2).map { owner in
 466 |                 let first = UInt32(owner * blockPages), width = UInt32(halves)
 467 |                 let inputs = [
 468 |                     mesh_pages_map(slot: UInt32(slot(f, s, owner == local ? .reducedOut : .reducedIn)), first: 0, count: width, stride: width, lag: 0),
 469 |                     mesh_pages_map(slot: UInt32(s == 0 ? embeddingSlot(f) : hiddenSlot(f, s - 1)), first: first, count: width, stride: width, lag: 0)
 470 |                 ]
 471 |                 let output = mesh_pages_map(slot: UInt32(hiddenSlot(f, s)), first: first, count: width, stride: width, lag: 0)
 472 |                 return bindMaps(inputs, [output], rows: blockRows)
 473 |             }
 474 |         }
 475 |     }
 476 |     let commandQueue = device.makeCommandQueue(maxCommandBufferCount: max(64, inflight * 16))!
```

**M, lines 477–498.** A/B: one command buffer for selected configured rows and completion after device writes. D: publish wrapper reaches bitmap rather than literal stamps. O: spans timing; it must not govern readiness. Per-invocation flatMap allocation is avoidable scan bookkeeping.

```text
 477 |     let spans = UnsafeMutablePointer<UInt64>.allocate(capacity: inflight * 2); spans.initialize(repeating: 0, count: inflight * 2)
 478 |     defer { spans.deallocate() }
 479 |     precondition(mesh_pages_start(program) == 0)
 480 | 
 481 |     func publish(_ ranges: [Publication], _ g: UInt64) { for r in ranges { _ = mesh_pages_publish(program, UInt32(r.slot), r.first, r.count, g) } }
 482 |     func issue(_ s: Int, _ f: Int, _ indices: [Int], _ g: UInt64) {
 483 |         let function = bound[s][f]
 484 |         let command = commandQueue.makeCommandBuffer()!
 485 |         let publications = indices.flatMap { function.calls[$0].publications }
 486 |         for i in indices { function.calls[i].step(command) }
 487 |         command.addCompletedHandler { command in
 488 |             if command.error != nil { mesh_pages_fail(program, EIO) }
 489 |             if mesh_pages_status(program) < 0 {
 490 |                 for r in publications { mesh_pages_cancel(program, UInt32(r.slot), r.first, r.count, g) }
 491 |                 return
 492 |             }
 493 |             publish(publications, g)
 494 |             mesh_pages_store(spans + f * 2, UInt64(command.gpuStartTime * 1e9)); mesh_pages_store(spans + f * 2 + 1, UInt64(command.gpuEndTime * 1e9))
 495 |         }
 496 |         command.commit()
 497 |     }
 498 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
```

**M, lines 499–514.** A/B: backend completion callback after ready input scan; no await-based scheduler here. D: indirect publish path. Native internal allocation/copy behavior is not proven conformant by these bindings.

```text
 499 |     func predict(_ s: Int, _ f: Int, _ g: UInt64) {
 500 |         for (i, prediction) in bound[s][f].predictions.enumerated() {
 501 |             var indices: UnsafePointer<UInt32>?
 502 |             guard mesh_pages_scan(predictionMaps[s][f][i], g, &indices) == 1 else { continue }
 503 |             let native = prediction.function, outputs = prediction.outputs
 504 |             native.call(g) { error in
 505 |                 if error != nil { mesh_pages_fail(program, EIO) }
 506 |                 if mesh_pages_status(program) < 0 {
 507 |                     for r in outputs { mesh_pages_cancel(program, UInt32(r.slot), r.first, r.count, g) }
 508 |                 } else {
 509 |                     publish(outputs, g)
 510 |                     mesh_pages_store(spans + f * 2, native.started); mesh_pages_store(spans + f * 2 + 1, native.finished)
 511 |                 }
 512 |             }
 513 |         }
 514 |     }
```

**M, lines 515–544.** A: coalesce actually ready rows and execute normalization in one buffer. X: separate post-send normalization architecture. D: explicit remote consume duplicates/mends incomplete static read relation; lifetime belongs in the canonical rows, not manual caller policy.

```text
 515 |     func normalize(_ s: Int, _ f: Int, _ ready: [Int], _ g: UInt64) {
 516 |         let index = s, hiddenOut = hiddenSlot(f, index), reducedIn = slot(f, index, .reducedIn)
 517 |         let command = commandQueue.makeCommandBuffer()!
 518 |         let encoder = command.makeComputeCommandEncoder()!
 519 |         encoder.setComputePipelineState(normPipeline)
 520 |         encoder.setBuffer(pageTable, offset: 0, index: 0); encoder.setBuffer(receive, offset: 0, index: 1); encoder.setBuffer(transmit, offset: 0, index: 2)
 521 |         encoder.setBuffer(model[s].gamma, offset: 0, index: 3); encoder.setBuffer(model[s].scale, offset: 0, index: 4); encoder.setBuffer(normGeometries[s][f], offset: 0, index: 5)
 522 |         var at = 0
 523 |         while at < ready.count {
 524 |             let first = ready[at]
 525 |             var end = at + 1
 526 |             while end < ready.count && ready[end] == ready[end - 1] + 1 { end += 1 }
 527 |             var firstRow = UInt32(first)
 528 |             encoder.setBytes(&firstRow, length: MemoryLayout<UInt32>.size, index: 6)
 529 |             encoder.dispatchThreadgroups(MTLSize(width: end - at, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
 530 |             at = end
 531 |         }
 532 |         encoder.endEncoding()
 533 |         command.addCompletedHandler { command in
 534 |             if command.error != nil { mesh_pages_fail(program, EIO) }
 535 |             for row in ready {
 536 |                 if mesh_pages_status(program) < 0 {
 537 |                     mesh_pages_cancel(program, UInt32(hiddenOut), UInt32(row * halves), UInt32(halves), g)
 538 |                 } else {
 539 |                     _ = mesh_pages_publish(program, UInt32(hiddenOut), UInt32(row * halves), UInt32(halves), g)
 540 |                     if row / blockRows == remote { _ = mesh_pages_consume(program, UInt32(reducedIn), UInt32((row % blockRows) * halves), UInt32(halves), g) }
 541 |                 }
 542 |             }
 543 |         }
 544 |         command.commit()
```

**M, lines 545–566.** R: compare literal digest pages as an endpoint value. X/D: heap-hash readiness, producible digest gate and rotating digest selection; page layout itself is allowed.

```text
 545 |     }
 546 |     // ../dox/mesh/design/algorithm-sources.md#endpoint-checking
 547 |     func compare(_ f: Int, _ index: Int, _ g: UInt64) -> Bool? {
 548 |         let page = digestPage(g)
 549 |         if !stamped(slot(f, index, .digestIn), page, 1, g) || !stamped(slot(f, index, .digestOut), page, 1, g) { return nil }
 550 |         let mine = words(slot(f, index, .digestOut), page), theirs = words(slot(f, index, .digestIn), page)
 551 |         return theirs[0] == g && mine[1] == theirs[3] && mine[2] == theirs[4] && mine[3] == theirs[1] && mine[4] == theirs[2] && mine[5] == theirs[5]
 552 |     }
 553 |     // ../dox/mesh/design/algorithm-sources.md#endpoint-checking
 554 |     func digest(_ f: Int, _ index: Int, _ g: UInt64) -> Bool {
 555 |         let page = digestPage(g)
 556 |         guard producible(slot(f, index, .digestOut), g),
 557 |               let sentPartial = hash(slot(f, index, .partialOut), g), let sentReduced = hash(slot(f, index, .reducedOut), g),
 558 |               let receivedPartial = hash(slot(f, index, .partialIn), g), let receivedReduced = hash(slot(f, index, .reducedIn), g),
 559 |               mesh_pages_claim(program, UInt32(slot(f, index, .digestOut)), UInt32(page), 1, g) == 1 else { return false }
 560 |         let mine = words(slot(f, index, .digestOut), page)
 561 |         mine[0] = g; mine[1] = sentPartial; mine[2] = sentReduced; mine[3] = receivedPartial; mine[4] = receivedReduced
 562 |         mine[5] = UInt64(rows) << 32 | UInt64(functions.count) << 8 | UInt64(generationStride)
 563 |         precondition(mesh_pages_publish(program, UInt32(slot(f, index, .digestOut)), UInt32(page), 1, g) == 0)
 564 |         return true
 565 |     }
 566 | 
```

**M, lines 567–594.** O: records/timing outside numerical flow. A: scanning configured functions and coalescing selected indices. D: duplicate ready arrays are temporary overhead, not persistent authority; do not turn them into queues/masks. Records become forbidden when used for admission below.

```text
 567 |     struct Record { let index, family: Int; let generation: UInt64; let valid: Bool; let latency: Double; let started, concluded, functionEnd: UInt64 }
 568 |     var records: [Record] = [], retries: [[String: Any]] = [], agreements = 0, disagreements = 0, lastFamily = 0
 569 |     var started = [UInt64](repeating: 0, count: inflight)
 570 |     var progressed = DispatchTime.now().uptimeNanoseconds, contacted = false
 571 |     // ../dox/mesh/design/algorithm-sources.md#overlap-and-performance-evidence
 572 |     func describe() -> String {
 573 |         var text = [CChar](repeating: 0, count: 65536)
 574 |         _ = mesh_pages_describe(program, &text, text.count)
 575 |         return String(cString: text)
 576 |     }
 577 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 578 |     func scanFunction(_ s: Int, _ f: Int, _ g: UInt64) {
 579 |         let ready = callMaps[s][f].indices.filter { i in
 580 |             var indices: UnsafePointer<UInt32>?
 581 |             return mesh_pages_scan(callMaps[s][f][i], g, &indices) == 1
 582 |         }
 583 |         guard !ready.isEmpty else { return }
 584 |         issue(s, f, ready, g)
 585 |     }
 586 |     // ../dox/mesh/design/algorithm-sources.md#operand-matching-and-storage
 587 |     func scanNorm(_ s: Int, _ f: Int, _ g: UInt64) {
 588 |         let ready = normalizationMaps[s][f].enumerated().flatMap { owner, function in
 589 |             var indices: UnsafePointer<UInt32>?
 590 |             let count = mesh_pages_scan(function, g, &indices)
 591 |             return UnsafeBufferPointer(start: indices, count: count).map { owner * blockRows + Int($0) }
 592 |         }
 593 |         if !ready.isEmpty { normalize(s, f, ready, g) }
 594 |     }
```

**M, lines 595–630.** A: scan configured functions using actual rows and dispatch only ready work. X: restrict all scanning to embedding first-row generation; this is coupled to endpoint-gated reuse. O: watchdog. R: negative link status, but break without recovery/repetition is incomplete.

```text
 595 |     while records.filter({ $0.valid }).count < measured + warmup {
 596 |         let now = DispatchTime.now().uptimeNanoseconds
 597 |         if mesh_pages_status(program) < 0 { break }
 598 |         if !contacted { contacted = (0..<inflight).contains { f in functions.indices.contains { mesh_pages_highest(program, UInt32(slot(f, $0, .partialIn))) > 0 } }; if contacted { progressed = now } }
 599 |         if now - progressed > (contacted ? collective.stallNanoseconds : collective.attachNanoseconds) { mesh_pages_fail(program, ETIMEDOUT); break }
 600 |         for f in 0..<inflight {
 601 |             let g = mesh_pages_stamp(mesh_pages_stamps(program, UInt32(embeddingSlot(f)))!, 0)
 602 |             if g & (UInt64(1) << 63) != 0 { continue }
 603 |             var finished = false, verdicts: [Bool] = []
 604 |             if g != 0 {
 605 |                 for s in functions.indices {
 606 |                     scanFunction(s, f, g)
 607 |                     predict(s, f, g)
 608 |                     scanNorm(s, f, g)
 609 |                     _ = digest(f, s, g)
 610 |                     if let verdict = compare(f, s, g) { verdicts.append(verdict) }
 611 |                 }
 612 |                 var headIndices: UnsafePointer<UInt32>?
 613 |                 if !headFunctions.isEmpty && mesh_pages_scan(headMaps[f], g, &headIndices) == 1 {
 614 |                     let output = UInt32(resultSlot(f)), input = UInt32(headInputSlot(f))
 615 |                     let command = commandQueue.makeCommandBuffer()!
 616 |                     headFunctions[f](command)
 617 |                     command.addCompletedHandler { command in
 618 |                         if command.error != nil { mesh_pages_fail(program, EIO) }
 619 |                         if mesh_pages_status(program) < 0 {
 620 |                             mesh_pages_cancel(program, input, 0, UInt32(halves), g)
 621 |                             mesh_pages_cancel(program, output, 0, UInt32(resultPages), g)
 622 |                         } else {
 623 |                             _ = mesh_pages_publish(program, input, 0, UInt32(halves), g)
 624 |                             _ = mesh_pages_publish(program, output, 0, UInt32(resultPages), g)
 625 |                         }
 626 |                     }
 627 |                     command.commit()
 628 |                 }
 629 |                 finished = headFunctions.isEmpty
 630 |                     ? stamped(hiddenSlot(f, lastFunction), 0, rows * halves, g)
```

**M, lines 631–642.** A/O: endpoint acceptance and measurement records. They must not become numerical storage release/admission proof. Numerical completion and checked endpoint conclusion are different observables.

```text
 631 |                     : stamped(resultSlot(f), 0, resultPages, g)
 632 |                 if finished && verdicts.count == functions.count && !records.contains(where: { $0.family == f && $0.generation == g }) {
 633 |                     let concluded = DispatchTime.now().uptimeNanoseconds
 634 |                     let index = records.filter { $0.family == f && $0.valid }.count * inflight + f + 1
 635 |                     let valid = verdicts.allSatisfy { $0 }
 636 |                     agreements += verdicts.filter { $0 }.count; disagreements += verdicts.filter { !$0 }.count
 637 |                     if !valid { retries.append(["step": index, "generation": Int(g), "reason": "digest"]) }
 638 |                     records.append(Record(index: index, family: f, generation: g, valid: valid,
 639 |                         latency: Double(concluded - started[f]) / 1e6, started: started[f], concluded: concluded,
 640 |                         functionEnd: mesh_pages_load(spans + f * 2 + 1)))
 641 |                     if valid { lastFamily = f }
 642 |                     progressed = concluded
```

**X, lines 643–661.** The next embedding is blocked on finished and every digest verdict. Records and verdicts are admission authority. Replace this dependency, not its spelling; input/output rows and released physical pages authorize execution.

```text
 643 |                 }
 644 |             }
 645 |             let count = records.filter { $0.family == f && $0.valid }.count
 646 |             if count * inflight + f + 1 > measured + warmup { continue }
 647 |             if g != 0 && (!finished || verdicts.count != functions.count) { continue }
 648 |             let next = g == 0 ? UInt64(f + 1) : g + UInt64(generationStride), output = UInt32(embeddingSlot(f))
 649 |             if mesh_pages_claim(program, output, 0, UInt32(rows * halves), next) != 1 { continue }
 650 |             started[f] = DispatchTime.now().uptimeNanoseconds
 651 |             let command = commandQueue.makeCommandBuffer()!
 652 |             embed(f, command)
 653 |             command.addCompletedHandler { command in
 654 |                 if command.error != nil { mesh_pages_fail(program, EIO) }
 655 |                 if mesh_pages_status(program) < 0 { mesh_pages_cancel(program, output, 0, UInt32(rows * halves), next) }
 656 |                 else { _ = mesh_pages_publish(program, output, 0, UInt32(rows * halves), next) }
 657 |             }
 658 |             command.commit()
 659 |             progressed = now
 660 |         }
 661 |     }
```

**M, lines 662–698.** B/O: teardown and reporting. X as algorithm storage: dense MTLBuffer output/logit copies even after timing; page values can be inspected directly. R missing: failed in-flight evaluations returned and recovered/repeated rather than eventual precondition failure.

```text
 662 |     while mesh_pages_writing(program) != 0 { sched_yield() }
 663 |     var interrupted = mesh_pages_status(program) < 0
 664 |     let settling = DispatchTime.now().uptimeNanoseconds
 665 |     while !interrupted && mesh_pages_settled(program) == 0 {
 666 |         if mesh_pages_status(program) < 0 { interrupted = true; break }
 667 |         if DispatchTime.now().uptimeNanoseconds - settling > collective.stallNanoseconds { mesh_pages_fail(program, ETIMEDOUT); interrupted = true; break }
 668 |         sched_yield()
 669 |     }
 670 |     let logits = (0..<inflight).map { _ in emptyHalf(head == nil ? 1 : vocabulary) }
 671 |     if head != nil {
 672 |         for page in 0..<resultPages {
 673 |             let first = page * extentColumns, count = min(extentColumns, vocabulary - first)
 674 |             memcpy(logits[lastFamily].contents().advanced(by: first * 2), pageWords(resultSlot(lastFamily), page), count * 2)
 675 |         }
 676 |     }
 677 |     let output = emptyHalf(rows * columns)
 678 |     for r in 0..<rows { for h in 0..<halves { memcpy(output.contents().advanced(by: (r * columns + h * extentColumns) * 2), pageWords(hiddenSlot(lastFamily, lastFunction), r * halves + h), rowBytes) } }
 679 |     var faultsScheduled = 0, generationsIssued = 0
 680 |     for L in 0..<configuredCount {
 681 |         let highest = mesh_pages_highest(program, UInt32(slot(L / functions.count, L % functions.count, .partialLocal)))
 682 |         let count = highest == 0 ? 0 : (highest - UInt64(L / functions.count) - 1) / UInt64(generationStride) + 1
 683 |         generationsIssued += Int(count)
 684 |         for e in 0..<count {
 685 |             for kind in [Kind.partialIn, .reducedIn] {
 686 |                 var a: UInt32 = 0, b: UInt32 = 0
 687 |                 if mesh_pages_faulted(program, UInt32(slot(L / functions.count, L % functions.count, kind)), e * UInt64(generationStride) + UInt64(L / functions.count) + 1, &a, &b) == 1 { faultsScheduled += 1 }
 688 |             }
 689 |         }
 690 |     }
 691 |     let finalStatus = mesh_pages_status(program), settled = mesh_pages_settled(program), foreign = mesh_pages_foreign_epoch(program)
 692 |     mesh_pages_stop(program)
 693 |     let summary = describe()
 694 |     if interrupted { FileHandle.standardError.write(("mesh reduce ended: " + String(cString: strerror(-finalStatus)) + "\n" + summary).data(using: .utf8)!) }
 695 |     precondition(mesh_pages_free(program) == 0)
 696 |     var detached = mesh_detach(mesh)
 697 |     while detached == EBUSY { sched_yield(); detached = mesh_detach(mesh) }
 698 |     precondition(detached == 0)
```

**O, lines 699–724.** Existing local reference arithmetic is outside the distributed numerical call graph. It does not prove the changed reduction/normalization algebra matches the specification. No new evaluator is required.

```text
 699 | 
 700 |     let scan = scanHalfBuf(head == nil ? output : logits[lastFamily], count: head == nil ? rows * columns : vocabulary)
 701 |     var numericalError: [String: Double] = [:]
 702 |     if env["LM_BENCH_REFERENCE"] == "1" && functions.count == 1 && functions[0].kind == .ffn {
 703 |         let residual = emptyHalf(rows * hidden), input = emptyHalf(rows * hidden), expected = emptyHalf(rows * columns)
 704 |         do {
 705 |             let table = embedding.contents().assumingMemoryBound(to: Float16.self), target = residual.contents().assumingMemoryBound(to: Float16.self)
 706 |             let scale = Float(hidden).squareRoot()
 707 |             for r in 0..<rows { let row = Int(tokens[r]) * hidden; for c in 0..<hidden { target[r * hidden + c] = Float16(Float(table[row + c]) * scale) } }
 708 |         }
 709 |         let reference = ParameterConfiguration(kind: .ffn, layer: functions[0].layer, rows: rows, start: 0, width: SHARED_INT, tileRows: rows, backend: .mps, layout: .rows, artifact: nil)
 710 |         let function = try! realizeMetalParameter(reference, file: file)(MatrixView(input, rows: rows, columns: hidden), [MatrixOutput(MatrixView(expected, rows: rows, columns: columns))], nil, nil)
 711 |         let command = queue.makeCommandBuffer()!
 712 |         encRMSNormG(command, x: residual, gammaBuf: model[0].norm, out: input, D: hidden, numVecs: rows)
 713 |         function.prefix?(command); for step in function.outputs { step(command) }
 714 |         encRmsNormAddScale(command, x: expected, gammaBuf: model[0].gamma, residual: residual, scalar: model[0].scale, out: expected, N: columns, numVecs: rows)
 715 |         command.commit(); command.waitUntilCompleted()
 716 |         if let error = command.error { fail("FFN reference: \(error)") }
 717 |         let actual = output.contents().assumingMemoryBound(to: Float16.self), target = expected.contents().assumingMemoryBound(to: Float16.self)
 718 |         var squaredError = 0.0, squaredReference = 0.0, maximumError = 0.0
 719 |         for j in 0..<(rows * columns) {
 720 |             let value = Double(target[j]), delta = Double(actual[j]) - value
 721 |             squaredError += delta * delta; squaredReference += value * value; maximumError = max(maximumError, abs(delta))
 722 |         }
 723 |         numericalError = ["relative_rms": sqrt(squaredError / squaredReference), "maximum_absolute": maximumError]
 724 |     }
```

**O, lines 725–751.** Statistics/reporting are outside algorithm readiness. Reported algorithm string itself admits digest-gated reuse. Final precondition is not required link recovery/repetition.

```text
 725 |     if head != nil { dumpLogits(logits[lastFamily], rows: 1, prefix: env["LM_BENCH_DUMP"]) }
 726 |     func median(_ values: [Double]) -> Double { values.isEmpty ? 0 : values.sorted()[values.count / 2] }
 727 |     func mean(_ values: [Double]) -> Double { values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count) }
 728 |     func variance(_ values: [Double]) -> Double { let m = mean(values); return values.isEmpty ? 0 : values.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(values.count) }
 729 |     let measuredRecords = records.filter { $0.valid && $0.index > warmup }.sorted { $0.concluded < $1.concluded }
 730 |     let latencies = measuredRecords.map { $0.latency }
 731 |     let periods = zip(measuredRecords.dropFirst(), measuredRecords).map { Double($0.concluded - $1.concluded) / 1e6 }
 732 |     let tails = measuredRecords.map { Double($0.concluded - $0.functionEnd) / 1e6 }
 733 |     let wall = measuredRecords.isEmpty ? 0 : Double(measuredRecords.last!.concluded - measuredRecords.map { $0.started }.min()!) / 1e6 / Double(measuredRecords.count)
 734 |     let record: [String: Any] = ["participant": deployment.participant, "rows": rows, "functions": functions.count, "layers": functions.count / 2, "in_flight": inflight, "configuredCount": configuredCount,
 735 |         "algorithm": "static configured functions scan actual input and destination stamps for one NFE identity; direct partial/reduced/result pages; endpoint digest acceptance still precedes benchmark input reuse", "latency_scope": "input submission through endpoint digest acceptance",
 736 |         "measured": measured, "warmup": warmup, "completed": measuredRecords.map { $0.index }, "completed_steps": records.filter { $0.valid }.count,
 737 |         "invocation_ms": latencies, "invocation_median_ms": median(latencies), "invocation_mean_ms": mean(latencies), "invocation_variance_ms2": variance(latencies),
 738 |         "nfe_latency_ms": latencies, "nfe_latency_median_ms": median(latencies),
 739 |         "period_ms": periods, "median_ms": median(periods), "period_mean_ms": mean(periods), "period_variance_ms2": variance(periods), "throughput_ms_per_nfe": wall,
 740 |         "after_local_ffn_ms": tails,
 741 |         "agreements": agreements, "disagreements": disagreements, "retries": retries, "retry_count": retries.count, "faults_scheduled": faultsScheduled,
 742 |         "reduction": ["extent_columns": extentColumns, "versions": inflight, "fault_period": collective.faultPeriod ?? 0, "fault_seed": Int(collective.faultSeed ?? 0)] as [String: Any],
 743 |         "generations_issued": generationsIssued, "nonfinite": scan.nan + scan.inf, "minimum": scan.mn, "maximum": scan.mx, "numerical_error": numericalError,
 744 |         "final_status": Int(finalStatus), "final_status_text": finalStatus < 0 ? String(cString: strerror(-finalStatus)) : "settled", "settled": Int(settled), "foreign_epoch": foreign,
 745 |         "pages": summary, "model": ModelDimensions.current.name, "device": device.name,
 746 |         "configuration": try! JSONSerialization.jsonObject(with: JSONEncoder().encode(deployment))]
 747 |     print(String(data: try! JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]), encoding: .utf8)!)
 748 |     fflush(stdout)
 749 |     precondition(finalStatus == 0 && settled != 0 && measuredRecords.count == measured)
 750 | }
 751 | #endif
```

### rdma/mesh-metal.h — 13 lines

**B, lines 1–13.** Backend aliases/layout descriptors. They are not independent operand allocation or numerical readiness; actual page storage remains mandatory.

```text
   1 | #ifndef MESH_METAL_H
   2 | #define MESH_METAL_H
   3 | #import <Metal/Metal.h>
   4 | #include "mesh.h"
   5 | #include "mesh-pages.h"
   6 | struct mesh_metal_layout { uint64_t origin; uint32_t stride, payload, capacity; };
   7 | struct mesh_metal_rows { uint64_t offset; uint32_t stride, payload, padding, rows_per_page; };
   8 | int mesh_metal_row_layout(struct mesh_ctx *context, struct mesh_scope scope, size_t rows,
   9 |   size_t row_bytes, size_t alignment, struct mesh_metal_rows *result);
  10 | id<MTLBuffer> mesh_metal_receive_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout);
  11 | id<MTLBuffer> mesh_metal_transmit_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout);
  12 | id<MTLBuffer> mesh_metal_page_table(id<MTLDevice> device, const mesh_pages *pages);
  13 | #endif
```

### rdma/mesh-metal.m — 51 lines

**B, lines 1–40.** mach_vm_remap(FALSE) and bytes-no-copy map existing physical storage. No payload copy here. Large pool aliases are still aggregate API buffers; do not claim they literally satisfy a one-page buffer mandate. The permitted use is an address binding to canonical pages, never alternate storage/ownership.

```text
   1 | #import "mesh-metal.h"
   2 | #include "mesh-wire.h"
   3 | #include <errno.h>
   4 | #include <mach/mach_vm.h>
   5 | #include <time.h>
   6 | #include <unistd.h>
   7 | 
   8 | static id<MTLBuffer> mesh_metal_memory(id<MTLDevice> device,const void *source,size_t bytes){
   9 |   mach_vm_address_t address=0;
  10 |   vm_prot_t current,maximum;
  11 |   kern_return_t status=mach_vm_remap(mach_task_self(),&address,bytes,0,VM_FLAGS_ANYWHERE,
  12 |     mach_task_self(),(mach_vm_address_t)source,FALSE,&current,&maximum,VM_INHERIT_NONE);
  13 |   if(status!=KERN_SUCCESS){ errno=ENOMEM; return nil; }
  14 |   id<MTLBuffer> buffer=[device newBufferWithBytesNoCopy:(void*)address length:bytes options:MTLResourceStorageModeShared
  15 |     deallocator:^(void *pointer,NSUInteger length){ mach_vm_deallocate(mach_task_self(),(mach_vm_address_t)pointer,length); }];
  16 |   if(!buffer){ mach_vm_deallocate(mach_task_self(),address,bytes); errno=ENOMEM; }
  17 |   return buffer;
  18 | }
  19 | static id<MTLBuffer> mesh_metal_pool(id<MTLDevice> device, struct mesh_ctx *context, uint32_t first, uint32_t count, struct mesh_metal_layout *layout){
  20 |   size_t alignment=(size_t)getpagesize(), stride=context->M->pgsz;
  21 |   unsigned char *base=mesh_at(context->M,0);
  22 |   uintptr_t begin=((uintptr_t)base+(size_t)first*stride)/alignment*alignment;
  23 |   uintptr_t end=((uintptr_t)base+((size_t)first+count)*stride+alignment-1)/alignment*alignment;
  24 |   if(end-begin>device.maxBufferLength){ errno=EOVERFLOW; return nil; }
  25 |   if(context->mapping_pinned>=0) context->mapping_pinned=1;
  26 |   *layout=(struct mesh_metal_layout){begin-(uintptr_t)base,(uint32_t)stride,0,(uint32_t)stride};
  27 |   return mesh_metal_memory(device,(void*)begin,end-begin);
  28 | }
  29 | id<MTLBuffer> mesh_metal_page_table(id<MTLDevice> device, const mesh_pages *pages){
  30 |   size_t bytes;
  31 |   const uint32_t *table=mesh_pages_table(pages,&bytes);
  32 |   if(bytes>device.maxBufferLength){ errno=EOVERFLOW; return nil; }
  33 |   return mesh_metal_memory(device,table,bytes);
  34 | }
  35 | id<MTLBuffer> mesh_metal_receive_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout){
  36 |   return mesh_metal_pool(device,context,0,context->M->pool,layout);
  37 | }
  38 | id<MTLBuffer> mesh_metal_transmit_pool(id<MTLDevice> device, struct mesh_ctx *context, struct mesh_metal_layout *layout){
  39 |   return mesh_metal_pool(device,context,context->M->pool,context->M->arena,layout);
  40 | }
```

**M, lines 41–51.** A: page payload/alignment arithmetic and validation. A smaller virtual row stride must never become an independently allocated transport page or a row-group scheduling unit.

```text
  41 | int mesh_metal_row_layout(struct mesh_ctx *context,struct mesh_scope scope,size_t rows,
  42 |   size_t row_bytes,size_t alignment,struct mesh_metal_rows *result){
  43 |   size_t header=sizeof(struct wire)+(mesh_epoch_set(scope.epoch)?sizeof(struct mesh_frame):MESH_OFF), stride=context->M->pgsz;
  44 |   if(!rows || !alignment || stride%alignment || !row_bytes) return EINVAL;
  45 |   size_t padding=(alignment-header%alignment)%alignment;
  46 |   if(header+padding>=stride || row_bytes>stride-header-padding) return EOVERFLOW;
  47 |   size_t packed=1, first=header+padding;
  48 |   for(;stride%2==0 && (stride/2)%alignment==0 && rows%(packed*2)==0 && first+row_bytes<=stride/2;packed*=2) stride/=2;
  49 |   *result=(struct mesh_metal_rows){first,(uint32_t)stride,(uint32_t)(padding+(packed-1)*stride+row_bytes),(uint32_t)padding,(uint32_t)packed};
  50 |   return 0;
  51 | }
```

### rdma/mesh-wire.h — 43 lines

**M, lines 1–43.** A/B: actual page addressing, identity and header encoding needed by the binding. X: legacy OPEN/FIN/CLOSE/ABORT/reply application protocol types and resident reply machinery as numerical algorithm. Active page path sends K_DATA only; legacy branches remain available to other linked clients.

```text
   1 | #ifndef MESH_WIRE_H
   2 | #define MESH_WIRE_H
   3 | #include "mesh.h"
   4 | #include <string.h>
   5 | 
   6 | #define MESH_SCOPED UINT32_C(0x4d590000)
   7 | struct mesh_frame { struct shdr h; struct mesh_epoch epoch; uint16_t source, target; };
   8 | static inline int mesh_epoch_equal(struct mesh_epoch a,struct mesh_epoch b){ return a.high==b.high && a.low==b.low; }
   9 | static inline int mesh_epoch_set(struct mesh_epoch e){ return e.high || e.low; }
  10 | static inline size_t mesh_stream_header(const struct mstream *s){ return mesh_epoch_set(s->scope.epoch)?sizeof(struct mesh_frame):MESH_OFF; }
  11 | static inline size_t mesh_stream_payload(struct hdr *m,const struct mstream *s){ return s->chunk?s->chunk:mesh_pay(m)-mesh_stream_header(s); }
  12 | static inline size_t mesh_frame_decode(const void *data,size_t bytes,struct mesh_frame *frame){
  13 |   if(bytes<MESH_OFF) return 0;
  14 |   *frame=(struct mesh_frame){0}; memcpy(&frame->h,data,MESH_OFF);
  15 |   if(frame->h.k<=K_ABORTED_TX) return MESH_OFF;
  16 |   if((frame->h.k&UINT32_C(0xffff0000))!=MESH_SCOPED || bytes<sizeof *frame) return 0;
  17 |   memcpy(frame,data,sizeof *frame);
  18 |   if(!mesh_epoch_set(frame->epoch)) return 0;
  19 |   frame->h.k&=UINT32_C(0xffff); return sizeof *frame;
  20 | }
  21 | static inline size_t mesh_frame_encode(void *data,const struct mesh_frame *frame,size_t payload){
  22 |   if(!mesh_epoch_set(frame->epoch)){ memcpy(data,&frame->h,MESH_OFF); return MESH_OFF+payload; }
  23 |   struct mesh_frame out=*frame; out.h.k|=MESH_SCOPED;
  24 |   memcpy(data,&out,sizeof out); return sizeof out+payload;
  25 | }
  26 | static inline int mesh_frame_receive(uint32_t kind){
  27 |   return kind==K_DATA || kind==K_FIN || kind==K_CLOSE || kind==K_OPEN || kind==K_ABORT_RX || kind==K_ABORTED_RX;
  28 | }
  29 | static inline int mesh_frame_reply(uint32_t kind){
  30 |   return kind==K_CLOSE?K_CLOSED:kind==K_ABORT_RX?K_ABORTED_TX:kind==K_ABORT_TX?K_ABORTED_RX:-1;
  31 | }
  32 | static inline size_t mesh_resident_reply(void *data,size_t bytes,uint16_t node){
  33 |   if(bytes<sizeof(struct wire)+sizeof(struct mesh_frame)) return 0;
  34 |   struct wire *wire=data; struct mesh_frame frame;
  35 |   if(wire->dst!=node || mesh_frame_decode((char*)data+sizeof *wire,bytes-sizeof *wire,&frame)!=sizeof frame ||
  36 |      frame.source!=wire->src || frame.target!=node) return 0;
  37 |   int kind=mesh_frame_reply(frame.h.k); if(frame.h.k!=K_CLOSE) return 0;
  38 |   uint16_t peer=wire->src;
  39 |   *wire=(struct wire){node,peer,0}; frame.h.k=(uint32_t)kind; frame.h.off=0;
  40 |   frame.source=node; frame.target=peer;
  41 |   return sizeof *wire+mesh_frame_encode((char*)data+sizeof *wire,&frame,0);
  42 | }
  43 | #endif
```

### rdma/mesh-functions.h — 57 lines

**X, lines 1–57.** Alternative call/executor API. Static row descriptions/page addressing have reusable purposes, but mesh_selection, mesh_join_state, request/acquire/rearm/complete state and open_retry policy are not allowed algorithm abstractions. Delete the interface with implementation/users rather than rebranding it.

```text
   1 | #ifndef MESH_FUNCTIONS_H
   2 | #define MESH_FUNCTIONS_H
   3 | #include "mesh.h"
   4 | 
   5 | struct mesh_extent { size_t offset, bytes; uint32_t channel; int peer, receive; uint32_t chunk; };
   6 | struct mesh_view {
   7 |   unsigned char *base;
   8 |   const uint32_t *pages;
   9 |   const struct mesh_epoch *epoch;
  10 |   size_t offset, bytes;
  11 |   uint32_t stride, payload, capacity;
  12 | };
  13 | typedef int (*mesh_index_fn)(void *, size_t, struct mesh_extent *);
  14 | typedef struct mesh_function mesh_function;
  15 | typedef struct mesh_executor mesh_executor;
  16 | typedef struct mesh_call mesh_call;
  17 | 
  18 | mesh_function *mesh_compile(size_t count, mesh_index_fn index, void *capture);
  19 | const struct mesh_extent *mesh_function_extent(const mesh_function *f, size_t index);
  20 | size_t mesh_function_count(const mesh_function *f);
  21 | void mesh_function_free(mesh_function *f);
  22 | struct mesh_executor_policy { uint64_t open_retry_ns; };
  23 | mesh_executor *mesh_executor_create(struct mesh_ctx *context, size_t window_pages);
  24 | mesh_executor *mesh_executor_create_with(struct mesh_ctx *context, size_t window_pages, struct mesh_executor_policy policy);
  25 | size_t mesh_call_unsent(const mesh_call *call);
  26 | mesh_call *mesh_bind(mesh_executor *e, mesh_function *f, uint32_t channel_base);
  27 | mesh_call *mesh_bind_scoped(mesh_executor *e, mesh_function *f, uint32_t channel_base, struct mesh_scope scope);
  28 | int mesh_call_rearm(mesh_call *call, struct mesh_epoch epoch);
  29 | void mesh_call_retain_transmit(mesh_call *call);
  30 | const struct mesh_view *mesh_call_view(const mesh_call *call, size_t index);
  31 | const uint32_t *mesh_call_indices(const mesh_call *call, size_t *bytes);
  32 | int mesh_progress(mesh_executor *e);
  33 | void mesh_request(mesh_call *call, size_t index);
  34 | int mesh_acquire(mesh_call *call, size_t index, struct mesh_view *view);
  35 | int mesh_acquire_page(mesh_call *call, size_t index, size_t page);
  36 | ptrdiff_t mesh_ready_pages(const mesh_call *call, size_t index);
  37 | const uint32_t *mesh_arrived_pages(const mesh_call *call, size_t index, size_t *count);
  38 | struct mesh_selection { size_t consumed, arrivals, published; };
  39 | struct mesh_join_state { uint64_t generation, coverage; };
  40 | size_t mesh_join_indices(struct mesh_join_state *state, uint64_t generation, uint64_t required,
  41 |   uint64_t contribution, uint32_t offset, const uint32_t *indices, size_t count, uint32_t *ready);
  42 | ptrdiff_t mesh_select_arrivals(mesh_call *call, size_t received, size_t produced, struct mesh_selection *selection,
  43 |   const uint32_t **selected);
  44 | int mesh_call_cycle(mesh_call *call, uint64_t period, uint64_t last);
  45 | int mesh_publish_page(mesh_call *call, size_t index, size_t page);
  46 | int mesh_page_published(const mesh_call *call, size_t index, size_t page);
  47 | ptrdiff_t mesh_flush(mesh_call *call, size_t index);
  48 | int mesh_publish(mesh_call *call, size_t index);
  49 | int mesh_complete(mesh_call *call, size_t index, int error);
  50 | int mesh_call_status(const mesh_call *call);
  51 | void mesh_call_cancel(mesh_call *call, int error);
  52 | int mesh_call_retire(mesh_call *call);
  53 | int mesh_call_abandon(mesh_call *call);
  54 | int mesh_executor_free(mesh_executor *e);
  55 | void mesh_view_read(const struct mesh_view *v, size_t offset, void *out, size_t bytes);
  56 | void mesh_view_write(const struct mesh_view *v, size_t offset, const void *in, size_t bytes);
  57 | #endif
```

### rdma/mesh-functions.c — 587 lines

**X, lines 1–63.** Argument state enum, requested/leased/active state, call runnable/work masks and executor mutex replace literal row readiness.

```text
   1 | #include "mesh-functions.h"
   2 | #include "mesh-wire.h"
   3 | #include <errno.h>
   4 | #include <limits.h>
   5 | #include <pthread.h>
   6 | #include <stdlib.h>
   7 | #include <string.h>
   8 | #include <sys/mman.h>
   9 | #include <unistd.h>
  10 | 
  11 | enum { WAITING, AVAILABLE, BORROWED, COMPLETING, COMMITTED, SETTLED, INVALID };
  12 | struct mesh_function { _Atomic size_t refs; size_t count; uint64_t *channels; struct mesh_extent extents[]; };
  13 | struct mesh_argument {
  14 |   struct mstream stream;
  15 |   struct mesh_view view;
  16 |   uint32_t *pages;
  17 |   size_t page_count;
  18 |   int active, released, leased;
  19 |   _Atomic int state, error, requested;
  20 | };
  21 | struct mesh_call {
  22 |   struct mesh_executor *executor;
  23 |   struct mesh_function *function;
  24 |   uint32_t channel_base;
  25 |   int retain_transmit;
  26 |   uint64_t period, last;
  27 |   struct mesh_scope scope;
  28 |   uint32_t *indices; size_t indices_bytes;
  29 |   uint64_t *work, *runnable;
  30 |   size_t settled;
  31 |   _Atomic int status;
  32 |   struct mesh_argument arguments[];
  33 | };
  34 | struct mesh_executor {
  35 |   struct mesh_ctx *context;
  36 |   pthread_mutex_t lock;
  37 |   struct mesh_call **calls;
  38 |   size_t count;
  39 |   struct mstream **streams;
  40 |   size_t capacity, window, received;
  41 |   struct mesh_executor_policy policy;
  42 | };
  43 | 
  44 | static void *allocation(size_t bytes){
  45 |   void *p=calloc(1,bytes); if(!p) errno=ENOMEM; return p; }
  46 | static int channel_order(const void *a,const void *b){
  47 |   uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b; return (x>y)-(x<y); }
  48 | static void change_argument(mesh_call *c,size_t i){
  49 |   mesh_stream_changed(&c->arguments[i].stream);
  50 | }
  51 | static void change_call(mesh_call *c){
  52 |   for(size_t word=0;word<mesh_page_words(c->function->count);word++){
  53 |     size_t bits=c->function->count-word*64; if(bits>64) bits=64;
  54 |     __atomic_fetch_or(&c->work[word],UINT64_MAX>>(64-bits),__ATOMIC_RELEASE);
  55 |     __atomic_fetch_or(&c->runnable[word],UINT64_MAX>>(64-bits),__ATOMIC_RELEASE);
  56 |   }
  57 | }
  58 | static void bind_argument_work(mesh_call *c,size_t i){
  59 |   c->arguments[i].stream.changed=c->work+i/64;
  60 |   c->arguments[i].stream.runnable=c->runnable+i/64;
  61 |   c->arguments[i].stream.changed_bit=UINT64_C(1)<<(i%64);
  62 | }
  63 | 
```

**M, lines 64–201.** Configuration shape validation is A in purpose, but this implementation constructs the forbidden call/stream state machine, duplicate arrivals table and synchronization owner. Remove as an algorithm implementation with its callers.

```text
  64 | mesh_function *mesh_compile(size_t count, mesh_index_fn index, void *capture){
  65 |   if(count>INT_MAX || count>(SIZE_MAX-sizeof(mesh_function))/sizeof(struct mesh_extent) || (count&&!index)){
  66 |     errno=EINVAL; return NULL; }
  67 |   mesh_function *f=allocation(sizeof *f+count*sizeof(struct mesh_extent)); if(!f) return NULL;
  68 |   atomic_init(&f->refs,1); f->count=count;
  69 |   for(size_t i=0;i<count;i++){
  70 |     int status=index(capture,i,&f->extents[i]);
  71 |     struct mesh_extent *x=&f->extents[i];
  72 |     if(status || !x->bytes || x->offset>SIZE_MAX-x->bytes || x->peer<0 || x->peer>=UINT16_MAX || (x->receive!=0&&x->receive!=1)){
  73 |       errno=status>0?status:EINVAL; free(f); return NULL; }
  74 |   }
  75 |   uint64_t *channels=allocation((count?count:1)*sizeof *channels);
  76 |   if(!channels){ free(f); return NULL; }
  77 |   for(size_t i=0;i<count;i++) channels[i]=((uint64_t)f->extents[i].channel<<17)|((uint64_t)f->extents[i].peer<<1)|(unsigned)f->extents[i].receive;
  78 |   qsort(channels,count,sizeof *channels,channel_order);
  79 |   int duplicate=0; for(size_t i=1;i<count;i++) duplicate|=channels[i]==channels[i-1];
  80 |   if(duplicate){ errno=EADDRINUSE; free(channels); free(f); return NULL; }
  81 |   f->channels=channels;
  82 |   return f;
  83 | }
  84 | const struct mesh_extent *mesh_function_extent(const mesh_function *f, size_t i){ return i<f->count?&f->extents[i]:NULL; }
  85 | size_t mesh_function_count(const mesh_function *f){ return f->count; }
  86 | void mesh_function_free(mesh_function *f){ if(f&&atomic_fetch_sub(&f->refs,1)==1){ free(f->channels); free(f); } }
  87 | 
  88 | static int channel_overlap(const mesh_function *a,uint32_t abase,const mesh_function *b,uint32_t bbase){
  89 |   if(!a->count || !b->count) return 0;
  90 |   uint64_t x=(uint64_t)abase<<17, y=(uint64_t)bbase<<17;
  91 |   if(a->channels[a->count-1]+x<b->channels[0]+y || b->channels[b->count-1]+y<a->channels[0]+x) return 0;
  92 |   size_t i=0,j=0;
  93 |   while(i<a->count && j<b->count){
  94 |     uint64_t left=a->channels[i]+x,right=b->channels[j]+y;
  95 |     if(left==right) return 1;
  96 |     if(left<right) i++; else j++;
  97 |   }
  98 |   return 0;
  99 | }
 100 | 
 101 | mesh_executor *mesh_executor_create(struct mesh_ctx *context, size_t window_pages){
 102 |   return mesh_executor_create_with(context,window_pages,(struct mesh_executor_policy){0});
 103 | }
 104 | mesh_executor *mesh_executor_create_with(struct mesh_ctx *context, size_t window_pages, struct mesh_executor_policy policy){
 105 |   if(!context || !context->M || context->mapping_pinned<0 || !window_pages || window_pages>=context->M->pool || window_pages>=context->M->arena){
 106 |     errno=EINVAL; return NULL; }
 107 |   mesh_executor *e=allocation(sizeof *e); if(!e) return NULL;
 108 |   int status=pthread_mutex_init(&e->lock,NULL);
 109 |   if(status){ free(e); errno=status; return NULL; }
 110 |   int vacant=0;
 111 |   if(!atomic_compare_exchange_strong(&context->executor_attached,&vacant,1)){
 112 |     pthread_mutex_destroy(&e->lock); free(e); errno=EBUSY; return NULL;
 113 |   }
 114 |   e->context=context; e->window=window_pages; e->policy=policy; context->mapping_pinned=1;
 115 |   return e;
 116 | }
 117 | 
 118 | static mesh_call *bind_function(mesh_executor *e, mesh_function *f, uint32_t base, struct mesh_scope scope){
 119 |   mesh_call *c=allocation(sizeof *c+f->count*sizeof(struct mesh_argument)+2*mesh_page_words(f->count)*sizeof(uint64_t)); if(!c) return NULL;
 120 |   c->executor=e; c->function=f; c->channel_base=base; c->scope=scope;
 121 |   c->work=(uint64_t*)(c->arguments+f->count);
 122 |   c->runnable=c->work+mesh_page_words(f->count);
 123 |   if(!f->count){ atomic_fetch_add(&f->refs,1); atomic_store(&c->status,1); c->executor=NULL; return c; }
 124 |   if(!e){ free(c); errno=EINVAL; return NULL; }
 125 |   size_t header=mesh_epoch_set(scope.epoch)?sizeof(struct mesh_frame):MESH_OFF;
 126 |   size_t payload=mesh_pay(e->context->M)-header;
 127 |   size_t pages=0, alignment=(size_t)getpagesize();
 128 |   for(size_t i=0;i<f->count;i++){
 129 |     size_t chunk=f->extents[i].chunk?f->extents[i].chunk:payload;
 130 |     if(chunk>payload){ free(c); errno=EINVAL; return NULL; }
 131 |     size_t count=f->extents[i].bytes/chunk+(f->extents[i].bytes%chunk!=0);
 132 |     size_t copies=f->extents[i].receive?2:1;
 133 |     if(count>((SIZE_MAX-alignment+1)/sizeof(uint32_t)-pages)/copies){ free(c); errno=EOVERFLOW; return NULL; }
 134 |     pages+=count*copies;
 135 |   }
 136 |   c->indices_bytes=(pages*sizeof(uint32_t)+alignment-1)/alignment*alignment;
 137 |   c->indices=mmap(NULL,c->indices_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
 138 |   if(c->indices==MAP_FAILED){ free(c); return NULL; }
 139 |   pages=0;
 140 |   int error=0; size_t receives=0;
 141 |   pthread_mutex_lock(&e->lock);
 142 |   if(e->context->mapping_pinned<0) error=ESTALE;
 143 |   if(f->count>INT_MAX-e->capacity) error=EOVERFLOW;
 144 |   for(size_t i=0;i<f->count&&!error;i++){
 145 |     const struct mesh_extent *x=&f->extents[i];
 146 |     struct mesh_argument *a=&c->arguments[i];
 147 |     size_t chunk=x->chunk?x->chunk:payload;
 148 |     a->page_count=x->bytes/chunk+(x->bytes%chunk!=0);
 149 |     if(x->channel>UINT32_MAX-base || a->page_count>e->window){ error=EOVERFLOW; break; }
 150 |     if(x->receive){
 151 |       size_t capacity=e->context->M->pool-e->window;
 152 |       if(a->page_count>capacity-e->received-receives){ error=EAGAIN; break; }
 153 |       receives+=a->page_count;
 154 |     }
 155 |     a->pages=c->indices+pages; pages+=a->page_count*(x->receive?2:1);
 156 |     a->view=(struct mesh_view){.base=mesh_at(e->context->M,0),.pages=a->pages,.epoch=&a->stream.scope.epoch,
 157 |       .offset=x->offset,.bytes=x->bytes,.stride=e->context->M->pgsz,.payload=(uint32_t)(sizeof(struct wire)+header),.capacity=(uint32_t)chunk};
 158 |     a->stream=(struct mstream){.n=x->bytes,.sid=base+x->channel,.node=x->peer,.rx=x->receive,
 159 |         .scope=scope,.logical_offset=x->offset,.chunk=x->chunk,.closing=1,.retain_seen=1,.open_retry_ns=e->policy.open_retry_ns,
 160 |         .arrivals=x->receive?a->pages+a->page_count:NULL,.parked=1};
 161 |     if(!error && (mesh_epoch_set(scope.epoch) || x->chunk)){
 162 |       if(x->receive && mesh_stream_receive(e->context,&a->stream,a->pages)) error=ENOMEM;
 163 |       a->active=1; a->stream.parked=0;
 164 |     }
 165 |     bind_argument_work(c,i);
 166 |   }
 167 |   for(size_t j=0;j<e->count&&!error;j++){
 168 |     mesh_call *other=e->calls[j];
 169 |     if((mesh_epoch_equal(scope.epoch,other->scope.epoch) || (other->period && scope.epoch.high==other->scope.epoch.high)) &&
 170 |        channel_overlap(f,base,other->function,other->channel_base)) error=EADDRINUSE;
 171 |   }
 172 |   size_t capacity=e->capacity+f->count;
 173 |   if(!error) error=mesh_stream_reserve(e->context,capacity);
 174 |   if(!error){
 175 |     mesh_call **calls=realloc(e->calls,(e->count+1)*sizeof *calls);
 176 |     if(calls) e->calls=calls; else error=ENOMEM;
 177 |   }
 178 |   if(!error){
 179 |     struct mstream **streams=realloc(e->streams,capacity*sizeof *streams);
 180 |     if(streams) e->streams=streams; else error=ENOMEM;
 181 |   }
 182 |   if(!error){
 183 |     for(size_t i=0;i<f->count;i++) e->streams[e->capacity+i]=&c->arguments[i].stream;
 184 |     e->capacity=capacity; e->received+=receives;
 185 |     atomic_fetch_add(&f->refs,1);
 186 |     e->calls[e->count++]=c;
 187 |     change_call(c);
 188 |     mesh_stream_register(e->context,e->streams,e->capacity);
 189 |   }
 190 |   pthread_mutex_unlock(&e->lock);
 191 |   if(error){ for(size_t i=0;i<f->count;i++) free(c->arguments[i].stream.seen); munmap(c->indices,c->indices_bytes); free(c); errno=error; return NULL; }
 192 |   return c;
 193 | }
 194 | mesh_call *mesh_bind(mesh_executor *e, mesh_function *f, uint32_t base){
 195 |   return bind_function(e,f,base,(struct mesh_scope){0});
 196 | }
 197 | mesh_call *mesh_bind_scoped(mesh_executor *e, mesh_function *f, uint32_t base, struct mesh_scope scope){
 198 |   if(!mesh_epoch_set(scope.epoch)){ errno=EINVAL; return NULL; }
 199 |   return bind_function(e,f,base,scope);
 200 | }
 201 | 
```

**X, lines 202–385.** Cycle/rearm/reset, argument state transitions, stream scheduling and mutex-protected progress are the alternative execution flow. These are not the static function scan required by the spec.

```text
 202 | void mesh_call_retain_transmit(mesh_call *c){ c->retain_transmit=1; }
 203 | int mesh_call_cycle(mesh_call *c,uint64_t period,uint64_t last){
 204 |   if(period<2 || last<c->scope.epoch.low || !mesh_epoch_set(c->scope.epoch)) return EINVAL;
 205 |   c->period=period; c->last=last; c->retain_transmit=1;
 206 |   for(size_t i=0;i<c->function->count;i++){
 207 |     struct mstream *s=&c->arguments[i].stream;
 208 |     s->resident=1; s->closing=0;
 209 |     if(c->function->extents[i].receive) continue;
 210 |     if(s->paged) continue;
 211 |     s->work=calloc(mesh_page_words(c->arguments[i].page_count),sizeof *s->work);
 212 |     if(!s->work) return ENOMEM;
 213 |     s->paged=1; s->retain_seen=1;
 214 |   }
 215 |   return 0;
 216 | }
 217 | 
 218 | static void reset_argument(mesh_call *c,size_t i,struct mesh_epoch epoch,int borrowed){
 219 |   struct mesh_argument *a=&c->arguments[i]; const struct mesh_extent *x=&c->function->extents[i];
 220 |   struct mstream *s=&a->stream;
 221 |   int retained=c->retain_transmit && !x->receive && s->stride;
 222 |   s->off=0; s->done=0; s->retry_ns=0; s->fin_after_ns=0; s->scan_word=0;
 223 |   s->st=MS_RUN; s->agreed=borrowed?s->agreed:0; s->error=0; s->abort_ack=0; s->closed=0; s->parked=0;
 224 |   s->scope.epoch=epoch; s->nb=a->page_count; s->pages=x->receive?a->pages:NULL;
 225 |   atomic_store(&s->available,0);
 226 |   if(s->seen) memset(s->seen,0,s->paged?mesh_page_words(a->page_count)*sizeof *s->work:(a->page_count+7)/8);
 227 |   a->active=1; a->released=0; a->leased=retained;
 228 |   atomic_store(&a->error,0); atomic_store(&a->requested,borrowed);
 229 |   atomic_store_explicit(&a->state,retained?(borrowed?BORROWED:AVAILABLE):WAITING,memory_order_release);
 230 |   change_argument(c,i);
 231 | }
 232 | 
 233 | int mesh_call_rearm(mesh_call *c, struct mesh_epoch epoch){
 234 |   if(!mesh_epoch_set(c->scope.epoch) || epoch.high!=c->scope.epoch.high || epoch.low<=c->scope.epoch.low) return EINVAL;
 235 |   mesh_executor *e=c->executor;
 236 |   if(!e){ c->scope.epoch=epoch; return 0; }
 237 |   pthread_mutex_lock(&e->lock);
 238 |   int error=e->context->mapping_pinned<0?ESTALE:0;
 239 |   size_t receives=0;
 240 |   for(size_t i=0;i<c->function->count&&!error;i++){
 241 |     if(atomic_load(&c->arguments[i].state)!=SETTLED) error=EBUSY;
 242 |     if(epoch.low<=c->arguments[i].stream.scope.epoch.low) error=EINVAL;
 243 |     if(mesh_stream_conflict(e->context,&c->arguments[i].stream,epoch)) error=EADDRINUSE;
 244 |     if(c->function->extents[i].receive) receives+=c->arguments[i].page_count;
 245 |   }
 246 |   if(!error && receives>e->context->M->pool-e->window-e->received) error=EAGAIN;
 247 |   if(!error){
 248 |     c->scope.epoch=epoch; e->received+=receives;
 249 |     for(size_t i=0;i<c->function->count;i++) reset_argument(c,i,epoch,0);
 250 |     c->settled=0;
 251 |     atomic_store_explicit(&c->status,0,memory_order_release);
 252 |   }
 253 |   pthread_mutex_unlock(&e->lock); return error;
 254 | }
 255 | 
 256 | const struct mesh_view *mesh_call_view(const mesh_call *c, size_t i){ return i<c->function->count?&c->arguments[i].view:NULL; }
 257 | const uint32_t *mesh_call_indices(const mesh_call *c, size_t *bytes){ *bytes=c->indices_bytes; return c->indices; }
 258 | 
 259 | static void argument_fail(mesh_call *c, struct mesh_argument *a, int error){
 260 |   int status=atomic_load(&c->status);
 261 |   while(status>=0&&!atomic_compare_exchange_weak(&c->status,&status,-error)){}
 262 |   if(status>=0) change_call(c);
 263 |   atomic_store(&a->error,-atomic_load(&c->status));
 264 |   if(a->active){
 265 |     if(!a->stream.error){ a->stream.error=error; a->stream.fin_after_ns=0; a->stream.retry_ns=0; }
 266 |     a->stream.st=MS_FAIL;
 267 |   }
 268 |   int state=atomic_load(&a->state);
 269 |   while(state!=BORROWED && state!=COMPLETING && state!=INVALID && state!=SETTLED &&
 270 |         !atomic_compare_exchange_weak(&a->state,&state,INVALID)){}
 271 | }
 272 | 
 273 | static int advance_argument(mesh_call *c,size_t i){
 274 |   mesh_executor *e=c->executor; struct mesh_ctx *ctx=e->context;
 275 |   struct mesh_argument *a=&c->arguments[i]; const struct mesh_extent *x=&c->function->extents[i];
 276 |   int state=atomic_load_explicit(&a->state,memory_order_acquire), previous=state;
 277 |   if(ctx->mapping_pinned<0) argument_fail(c,a,ESTALE);
 278 |   if(a->active && a->stream.st==MS_FAIL) argument_fail(c,a,a->stream.error?a->stream.error:EIO);
 279 |   if(atomic_load(&c->status)<0) argument_fail(c,a,-atomic_load(&c->status));
 280 |   if(a->active && x->receive && a->stream.done==x->bytes){
 281 |     int waiting=WAITING;
 282 |     atomic_compare_exchange_strong_explicit(&a->state,&waiting,AVAILABLE,memory_order_release,memory_order_relaxed);
 283 |   }
 284 |   state=atomic_load_explicit(&a->state,memory_order_acquire);
 285 |   if(state==COMMITTED && atomic_load(&a->error)){ argument_fail(c,a,atomic_load(&a->error)); state=atomic_load(&a->state); }
 286 |   if(!a->released && (state==INVALID || (state==COMMITTED && (x->receive?a->stream.done==x->bytes:a->stream.st==MS_DONE)))){
 287 |     if(!a->active || (c->retain_transmit && !x->receive && state==COMMITTED) || !mesh_release_view(ctx,&a->stream)){
 288 |       a->released=1;
 289 |       if(x->receive) e->received-=a->page_count;
 290 |     } else change_argument(c,i);
 291 |   }
 292 |   if(a->released && (!a->active || (a->stream.st==MS_DONE && a->stream.closed &&
 293 |      (!a->stream.stride || mesh_stream_idle(ctx,&a->stream))) ||
 294 |      (atomic_load(&c->status)<0 && (!mesh_epoch_set(c->scope.epoch) || a->stream.abort_ack || ctx->mapping_pinned<0)))){
 295 |     a->stream.parked=1;
 296 |     atomic_store_explicit(&a->state,SETTLED,memory_order_release); state=SETTLED;
 297 |   } else if(a->released && a->stream.stride && a->stream.st==MS_DONE && a->stream.closed) change_argument(c,i);
 298 |   if(state==SETTLED && c->period && atomic_load(&c->status)>=0 &&
 299 |      c->period<=c->last && a->stream.scope.epoch.low<=c->last-c->period){
 300 |     struct mesh_epoch epoch=a->stream.scope.epoch; epoch.low+=c->period;
 301 |     reset_argument(c,i,epoch,1);
 302 |     if(x->receive) e->received+=a->page_count;
 303 |     state=atomic_load_explicit(&a->state,memory_order_acquire);
 304 |   }
 305 |   c->settled+=(state==SETTLED)-(previous==SETTLED);
 306 |   if(c->settled==c->function->count){
 307 |     int running=0; atomic_compare_exchange_strong_explicit(&c->status,&running,1,memory_order_release,memory_order_relaxed);
 308 |   }
 309 |   return state;
 310 | }
 311 | 
 312 | static void progress_arguments(mesh_executor *e){
 313 |   struct mesh_ctx *ctx=e->context;
 314 |   for(size_t j=0;j<e->count;j++){
 315 |     mesh_call *c=e->calls[j];
 316 |     for(size_t word=0;word<mesh_page_words(c->function->count);word++){
 317 |       if(!__atomic_load_n(&c->work[word],__ATOMIC_RELAXED)) continue;
 318 |       uint64_t work=__atomic_exchange_n(&c->work[word],0,__ATOMIC_ACQUIRE);
 319 |       for(;work;work&=work-1){
 320 |         size_t i=word*64+(size_t)__builtin_ctzll(work);
 321 |         struct mesh_argument *a=&c->arguments[i]; const struct mesh_extent *x=&c->function->extents[i];
 322 |         int state=advance_argument(c,i);
 323 |         if(state==SETTLED) continue;
 324 |         if(state==WAITING && atomic_load(&a->requested) && atomic_load(&c->status)>=0 &&
 325 |            (x->receive?!a->active:!a->leased)){
 326 |           if(x->receive){
 327 |             int error=mesh_lissen_view(ctx,&a->stream,a->pages,x->bytes,c->channel_base+x->channel);
 328 |             bind_argument_work(c,i);
 329 |             if(error){
 330 |               argument_fail(c,a,ENOMEM); continue; }
 331 |             a->stream.node=x->peer;
 332 |             a->stream.arrivals=a->pages+a->page_count;
 333 |           } else {
 334 |             void *p=a->active?mesh_stream_lease(ctx,&a->stream):mesh_yell_view(ctx,&a->stream,x->bytes,x->peer,c->channel_base+x->channel);
 335 |             bind_argument_work(c,i);
 336 |             if(!p){ change_argument(c,i); continue; }
 337 |             size_t first=((unsigned char*)p-mesh_at(ctx->M,0))/ctx->M->pgsz;
 338 |             for(size_t j=0;j<a->page_count;j++) a->pages[j]=(uint32_t)(first+j);
 339 |             a->leased=1;
 340 |             atomic_store_explicit(&a->state,AVAILABLE,memory_order_release);
 341 |           }
 342 |           a->stream.closing=!a->stream.resident; a->stream.parked=0; a->active=1;
 343 |           mesh_stream_schedule(&a->stream);
 344 |         }
 345 |       }
 346 |     }
 347 |   }
 348 | }
 349 | 
 350 | static void progress_streams(mesh_executor *e,uint64_t now_ns){
 351 |   size_t turn=e->context->stream_cursor++;
 352 |   for(size_t j=0;j<e->count;j++){
 353 |     mesh_call *c=e->calls[(turn%e->count+j)%e->count];
 354 |     size_t words=mesh_page_words(c->function->count), step=turn/e->count;
 355 |     for(size_t w=0;w<words;w++){
 356 |       size_t word=(step/64%words+w)%words;
 357 |       if(!__atomic_load_n(&c->runnable[word],__ATOMIC_RELAXED)) continue;
 358 |       uint64_t work=__atomic_exchange_n(&c->runnable[word],0,__ATOMIC_ACQUIRE);
 359 |       size_t shift=step%64;
 360 |       work=(work>>shift)|(work<<((64-shift)%64));
 361 |       for(;work;work&=work-1){
 362 |         size_t i=word*64+((size_t)__builtin_ctzll(work)+shift)%64;
 363 |         struct mstream *s=&c->arguments[i].stream;
 364 |         if(mesh_progress_stream(e->context,s,e->window,now_ns)) mesh_stream_schedule(s);
 365 |       }
 366 |     }
 367 |   }
 368 | }
 369 | 
 370 | int mesh_progress(mesh_executor *e){
 371 |   if(pthread_mutex_trylock(&e->lock)) return 0;
 372 |   struct mesh_ctx *ctx=e->context;
 373 |   progress_arguments(e);
 374 |   uint64_t now_ns;
 375 |   if(mesh_poll_streams(ctx,e->streams,(int)e->capacity,&now_ns)) progress_streams(e,now_ns);
 376 |   progress_arguments(e);
 377 |   int completed=0;
 378 |   for(size_t j=0;j<e->count;j++){
 379 |     mesh_call *c=e->calls[j];
 380 |     completed+=atomic_load(&c->status)==1;
 381 |   }
 382 |   pthread_mutex_unlock(&e->lock);
 383 |   return completed;
 384 | }
 385 | 
```

**X, lines 386–519.** Request/acquire states, heap join coverage, publication masks, arrival rearrangement/consumption cursors and completing/committed transitions substitute for actual stamped rows. Plain page address reads are reusable arithmetic, not a reason to retain this API.

```text
 386 | void mesh_request(mesh_call *c, size_t i){
 387 |   if(i<c->function->count && !atomic_exchange_explicit(&c->arguments[i].requested,1,memory_order_acq_rel)) change_argument(c,i);
 388 | }
 389 | int mesh_acquire(mesh_call *c, size_t i, struct mesh_view *view){
 390 |   if(i>=c->function->count) return -EINVAL;
 391 |   int error=atomic_load(&c->status); if(error<0) return error;
 392 |   struct mesh_argument *a=&c->arguments[i]; int available=AVAILABLE;
 393 |   mesh_request(c,i);
 394 |   if(!atomic_compare_exchange_strong_explicit(&a->state,&available,BORROWED,memory_order_acquire,memory_order_relaxed)) return 0;
 395 |   *view=a->view; return 1;
 396 | }
 397 | size_t mesh_join_indices(struct mesh_join_state *state,uint64_t generation,uint64_t required,
 398 |   uint64_t contribution,uint32_t offset,const uint32_t *indices,size_t count,uint32_t *ready){
 399 |   size_t n=0;
 400 |   for(size_t i=0;i<count;i++){
 401 |     uint32_t row=offset+indices[i]; struct mesh_join_state *s=&state[row];
 402 |     uint64_t previous=s->coverage*(s->generation==generation), coverage=previous|contribution;
 403 |     *s=(struct mesh_join_state){generation,coverage};
 404 |     ready[n]=row; n+=(coverage==required && previous!=required);
 405 |   }
 406 |   return n;
 407 | }
 408 | 
 409 | static void publish_word(struct mesh_argument *a,size_t index,uint64_t mask){
 410 |   struct mesh_page_bits *word=&a->stream.work[index];
 411 |   uint64_t added=mask&~__atomic_fetch_or(&word->published,mask,__ATOMIC_RELEASE);
 412 |   if(!added) return;
 413 |   size_t tail=a->page_count-1, capacity=a->view.capacity;
 414 |   size_t omitted=(tail/64==index && ((added>>(tail%64))&1))*(capacity-(a->view.bytes-tail*capacity));
 415 |   atomic_fetch_add_explicit(&a->stream.available,(size_t)__builtin_popcountll(added)*capacity-omitted,memory_order_release);
 416 |   __atomic_fetch_or(&word->pending,added,__ATOMIC_RELEASE);
 417 |   mesh_stream_schedule(&a->stream);
 418 | }
 419 | int mesh_publish(mesh_call *c, size_t i){
 420 |   if(i>=c->function->count || c->function->extents[i].receive) return -EINVAL;
 421 |   struct mesh_argument *a=&c->arguments[i];
 422 |   if(atomic_load_explicit(&a->state,memory_order_acquire)!=BORROWED) return -EINVAL;
 423 |   if(a->stream.paged) for(size_t word=0;word<mesh_page_words(a->page_count);word++){
 424 |     size_t bits=a->page_count-word*64; if(bits>64) bits=64;
 425 |     publish_word(a,word,UINT64_MAX>>(64-bits));
 426 |   }
 427 |   if(!a->stream.paged){
 428 |     atomic_store_explicit(&a->stream.available,a->view.bytes,memory_order_release);
 429 |     mesh_stream_schedule(&a->stream);
 430 |   }
 431 |   return 0;
 432 | }
 433 | int mesh_acquire_page(mesh_call *c, size_t i, size_t page){
 434 |   if(i>=c->function->count) return -EINVAL;
 435 |   int error=atomic_load(&c->status); if(error<0) return error;
 436 |   struct mesh_argument *a=&c->arguments[i];
 437 |   if(page>=a->page_count) return -EINVAL;
 438 |   if(!c->function->extents[i].receive) return mesh_page_published(c,i,page);
 439 |   if(a->pages[page]==UINT32_MAX) return 0;
 440 |   int state=atomic_load_explicit(&a->state,memory_order_acquire);
 441 |   if(state==WAITING || state==AVAILABLE)
 442 |     atomic_compare_exchange_strong_explicit(&a->state,&state,BORROWED,memory_order_acquire,memory_order_relaxed);
 443 |   return state==WAITING || state==AVAILABLE || state==BORROWED?1:-EINVAL;
 444 | }
 445 | ptrdiff_t mesh_ready_pages(const mesh_call *c,size_t i){
 446 |   if(i>=c->function->count) return -EINVAL;
 447 |   int error=atomic_load(&c->status); if(error<0) return error;
 448 |   const struct mesh_argument *a=&c->arguments[i];
 449 |   size_t bytes=c->function->extents[i].receive?a->stream.done:atomic_load_explicit(&a->stream.available,memory_order_acquire);
 450 |   return (ptrdiff_t)(bytes/a->view.capacity+(bytes%a->view.capacity!=0));
 451 | }
 452 | const uint32_t *mesh_arrived_pages(const mesh_call *c,size_t i,size_t *count){
 453 |   *count=0;
 454 |   if(i>=c->function->count || !c->function->extents[i].receive) return NULL;
 455 |   const struct mesh_argument *a=&c->arguments[i];
 456 |   *count=a->stream.done/a->view.capacity+(a->stream.done%a->view.capacity!=0);
 457 |   return a->stream.arrivals;
 458 | }
 459 | ptrdiff_t mesh_select_arrivals(mesh_call *c,size_t received,size_t produced,struct mesh_selection *selection,const uint32_t **selected){
 460 |   if(received>=c->function->count || produced>=c->function->count ||
 461 |      !c->function->extents[received].receive || c->function->extents[produced].receive) return -EINVAL;
 462 |   struct mesh_argument *rx=&c->arguments[received],*tx=&c->arguments[produced];
 463 |   if(!tx->stream.paged || tx->page_count<rx->page_count) return -EINVAL;
 464 |   if(!mesh_epoch_equal(rx->stream.scope.epoch,tx->stream.scope.epoch)){ *selected=NULL; return 0; }
 465 |   size_t count=0; mesh_arrived_pages(c,received,&count);
 466 |   size_t consumed=selection->consumed;
 467 |   if(consumed>count || selection->arrivals>count) return -EINVAL;
 468 |   size_t published=atomic_load_explicit(&tx->stream.available,memory_order_acquire);
 469 |   uint32_t *indices=rx->stream.arrivals;
 470 |   size_t begin=published?(published==selection->published?selection->arrivals:consumed):count;
 471 |   size_t end=consumed;
 472 |   if(published==tx->stream.n){ end=count; begin=count; }
 473 |   for(size_t i=begin;i<count;i++){
 474 |     uint32_t enabled=(uint32_t)mesh_stream_published(&tx->stream,indices[i]);
 475 |     uint32_t swap=(indices[i]^indices[end])*(uint32_t)enabled;
 476 |     indices[i]^=swap; indices[end]^=swap; end+=enabled;
 477 |   }
 478 |   if(end>consumed && !consumed){
 479 |     int status=mesh_acquire_page(c,received,indices[0]);
 480 |     if(status!=1) return status?status:-EAGAIN;
 481 |   }
 482 |   selection->arrivals=count; selection->published=published;
 483 |   *selected=indices?indices+consumed:NULL;
 484 |   return (ptrdiff_t)(end-consumed);
 485 | }
 486 | int mesh_publish_page(mesh_call *c, size_t i, size_t page){
 487 |   if(i>=c->function->count || c->function->extents[i].receive) return -EINVAL;
 488 |   struct mesh_argument *a=&c->arguments[i]; struct mstream *s=&a->stream;
 489 |   if(!s->paged || page>=a->page_count || atomic_load_explicit(&a->state,memory_order_acquire)!=BORROWED) return -EINVAL;
 490 |   publish_word(a,page/64,UINT64_C(1)<<(page%64));
 491 |   return 0;
 492 | }
 493 | int mesh_page_published(const mesh_call *c,size_t i,size_t page){
 494 |   if(i>=c->function->count || c->function->extents[i].receive) return -EINVAL;
 495 |   const struct mesh_argument *a=&c->arguments[i];
 496 |   if(!a->stream.paged || page>=a->page_count) return -EINVAL;
 497 |   if(!a->stream.work) return -EINVAL;
 498 |   int published=mesh_stream_published(&a->stream,page);
 499 |   if(!published && atomic_load_explicit(&a->state,memory_order_acquire)!=BORROWED) return -EINVAL;
 500 |   return published;
 501 | }
 502 | ptrdiff_t mesh_flush(mesh_call *c,size_t i){
 503 |   if(i>=c->function->count || c->function->extents[i].receive) return -EINVAL;
 504 |   struct mesh_argument *a=&c->arguments[i];
 505 |   if(!a->stream.paged) return -EINVAL;
 506 |   return mesh_stream_push_pages(c->executor->context,&a->stream,c->executor->window);
 507 | }
 508 | int mesh_complete(mesh_call *c, size_t i, int error){
 509 |   if(i>=c->function->count || error<0) return -EINVAL;
 510 |   struct mesh_argument *a=&c->arguments[i];
 511 |   int borrowed=BORROWED;
 512 |   if(!atomic_compare_exchange_strong(&a->state,&borrowed,COMPLETING)) return -EINVAL;
 513 |   atomic_store(&a->error,error);
 514 |   if(!error && !c->function->extents[i].receive && !a->stream.paged)
 515 |     atomic_store_explicit(&a->stream.available,a->view.bytes,memory_order_release);
 516 |   atomic_store_explicit(&a->state,COMMITTED,memory_order_release);
 517 |   change_argument(c,i);
 518 |   return 0;
 519 | }
```

**X, lines 520–576.** Unsent/status/cancel/retire machinery exists to maintain the alternative call state machine. Preserve necessary transport teardown in the substrate, remove algorithm owner/state.

```text
 520 | size_t mesh_call_unsent(const mesh_call *c){
 521 |   size_t pending=0;
 522 |   for(size_t i=0;i<c->function->count;i++){
 523 |     const struct mesh_argument *a=&c->arguments[i]; const struct mstream *s=&a->stream;
 524 |     if(c->function->extents[i].receive || !a->active || s->st!=MS_RUN) continue;
 525 |     if(s->paged){
 526 |       size_t bytes=atomic_load_explicit(&s->available,memory_order_acquire)-s->off;
 527 |       pending+=bytes/a->view.capacity+(bytes%a->view.capacity!=0);
 528 |     }
 529 |     else pending+=s->off<atomic_load_explicit(&s->available,memory_order_acquire);
 530 |   }
 531 |   return pending;
 532 | }
 533 | int mesh_call_status(const mesh_call *c){ return atomic_load_explicit(&c->status,memory_order_acquire); }
 534 | void mesh_call_cancel(mesh_call *c, int error){
 535 |   int status=atomic_load(&c->status);
 536 |   while(status>=0&&!atomic_compare_exchange_weak(&c->status,&status,-(error>0?error:ECANCELED))){}
 537 |   if(status>=0) change_call(c);
 538 | }
 539 | 
 540 | static int retire_call(mesh_call *c, int abandon){
 541 |   if(abandon && (!mesh_epoch_set(c->scope.epoch) || mesh_call_status(c)>=0)) return EINVAL;
 542 |   mesh_executor *e=c->executor;
 543 |   if(!e){ mesh_function_free(c->function); free(c); return 0; }
 544 |   pthread_mutex_lock(&e->lock);
 545 |   int busy=0;
 546 |   for(size_t i=0;i<c->function->count;i++){
 547 |     struct mesh_argument *a=&c->arguments[i];
 548 |     int state=atomic_load(&a->state);
 549 |     busy|=abandon?(!a->released || state==BORROWED || state==COMPLETING):state!=SETTLED;
 550 |   }
 551 |   if(busy){ pthread_mutex_unlock(&e->lock); return EBUSY; }
 552 |   size_t at=0,first=0;
 553 |   for(;at<e->count && e->calls[at]!=c;at++) first+=e->calls[at]->function->count;
 554 |   memmove(e->calls+at,e->calls+at+1,(e->count-at-1)*sizeof *e->calls); e->count--;
 555 |   e->capacity-=c->function->count;
 556 |   memmove(e->streams+first,e->streams+first+c->function->count,(e->capacity-first)*sizeof *e->streams);
 557 |   mesh_stream_register(e->context,e->streams,e->capacity);
 558 |   for(size_t i=0;i<c->function->count;i++)
 559 |     if(c->arguments[i].stream.stride) mesh_release_view(e->context,&c->arguments[i].stream);
 560 |   pthread_mutex_unlock(&e->lock);
 561 |   for(size_t i=0;i<c->function->count;i++) free(c->arguments[i].stream.seen);
 562 |   munmap(c->indices,c->indices_bytes);
 563 |   mesh_function_free(c->function); free(c); return 0;
 564 | }
 565 | int mesh_call_retire(mesh_call *c){ return retire_call(c,0); }
 566 | int mesh_call_abandon(mesh_call *c){ return retire_call(c,1); }
 567 | int mesh_executor_free(mesh_executor *e){
 568 |   pthread_mutex_lock(&e->lock);
 569 |   int busy=e->count!=0 || (e->context->mapping_pinned>=0 && e->context->inflight!=0);
 570 |   pthread_mutex_unlock(&e->lock);
 571 |   if(busy) return EBUSY;
 572 |   atomic_store(&e->context->executor_attached,0);
 573 |   mesh_stream_register(e->context,NULL,0);
 574 |   pthread_mutex_destroy(&e->lock); free(e->calls); free(e->streams); free(e); return 0;
 575 | }
 576 | 
```

**M, lines 577–587.** A: literal page/offset calculation. X for numerical flow: copy to/from a foreign data buffer. Direct gather/scatter arithmetic must address the pages where they are.

```text
 577 | static void view_copy(const struct mesh_view *v, size_t offset, void *data, size_t bytes, int write){
 578 |   size_t payload=v->capacity?v->capacity:v->stride-v->payload;
 579 |   while(bytes){
 580 |     size_t page=offset/payload, position=offset%payload, n=payload-position; if(n>bytes) n=bytes;
 581 |     unsigned char *p=v->base+(size_t)v->pages[page]*v->stride+v->payload+position;
 582 |     if(write) memcpy(p,data,n); else memcpy(data,p,n);
 583 |     offset+=n; data=(unsigned char*)data+n; bytes-=n;
 584 |   }
 585 | }
 586 | void mesh_view_read(const struct mesh_view *v, size_t offset, void *out, size_t bytes){ view_copy(v,offset,out,bytes,0); }
 587 | void mesh_view_write(const struct mesh_view *v, size_t offset, const void *in, size_t bytes){ view_copy(v,offset,(void*)in,bytes,1); }
```

### rdma/mesh-reduce.h — 25 lines

**X, lines 1–25.** Independent reduction tokens, readiness, claim/commit/status/cancel API. All algorithm abstractions are replaced by configured page functions.

```text
   1 | #ifndef MESH_REDUCE_H
   2 | #define MESH_REDUCE_H
   3 | #include <stddef.h>
   4 | #include <stdint.h>
   5 | 
   6 | struct mesh_reduce_node { uint64_t owner, contributor; size_t input[2]; };
   7 | struct mesh_reduce_token { uint64_t program, invocation, node; };
   8 | typedef int (*mesh_reduce_index_fn)(void *, size_t, struct mesh_reduce_node *);
   9 | typedef struct mesh_reduce_function mesh_reduce_function;
  10 | typedef struct mesh_reduction mesh_reduction;
  11 | 
  12 | mesh_reduce_function *mesh_reduce_compile(size_t contributors, mesh_reduce_index_fn index, void *capture);
  13 | size_t mesh_reduce_count(const mesh_reduce_function *f);
  14 | const struct mesh_reduce_node *mesh_reduce_node(const mesh_reduce_function *f, size_t index);
  15 | void mesh_reduce_free(mesh_reduce_function *f);
  16 | mesh_reduction *mesh_reduce_bind(mesh_reduce_function *f, uint64_t program, uint64_t invocation, uint64_t owner);
  17 | struct mesh_reduce_token mesh_reduce_token(const mesh_reduction *r, size_t index);
  18 | int mesh_reduce_needed(const mesh_reduction *r, size_t index);
  19 | int mesh_reduce_ready(const mesh_reduction *r, size_t index);
  20 | int mesh_reduce_claim(mesh_reduction *r, struct mesh_reduce_token token);
  21 | int mesh_reduce_commit(mesh_reduction *r, struct mesh_reduce_token token, int error);
  22 | int mesh_reduce_status(const mesh_reduction *r);
  23 | void mesh_reduce_cancel(mesh_reduction *r, int error);
  24 | int mesh_reduce_retire(mesh_reduction *r);
  25 | #endif
```

### rdma/mesh-reduce.c — 124 lines

**X, lines 1–124.** R_WAITING/R_BORROWED/R_READY/R_COMPLETING and states[] form a second reduction machine, with program/invocation/node tokens. Tree algebra/configuration is representable by static functions over pages; this token machine has no permitted role.

```text
   1 | #include "mesh-reduce.h"
   2 | #include <errno.h>
   3 | #include <limits.h>
   4 | #include <stdatomic.h>
   5 | #include <stdlib.h>
   6 | 
   7 | enum { R_WAITING, R_BORROWED, R_READY, R_COMPLETING };
   8 | struct mesh_reduce_function {
   9 |   _Atomic size_t refs;
  10 |   size_t count;
  11 |   size_t *parent;
  12 |   struct mesh_reduce_node nodes[];
  13 | };
  14 | struct mesh_reduction {
  15 |   mesh_reduce_function *function;
  16 |   uint64_t program, invocation, owner;
  17 |   _Atomic int error;
  18 |   _Atomic int states[];
  19 | };
  20 | static int contributor_order(const void *a,const void *b){
  21 |   uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b; return (x>y)-(x<y); }
  22 | 
  23 | mesh_reduce_function *mesh_reduce_compile(size_t contributors, mesh_reduce_index_fn index, void *capture){
  24 |   if(!contributors || contributors>(size_t)INT_MAX/2 || !index){ errno=EINVAL; return NULL; }
  25 |   size_t count=contributors*2-1;
  26 |   mesh_reduce_function *f=calloc(1,sizeof *f+count*sizeof *f->nodes);
  27 |   size_t *parent=malloc(count*sizeof *parent);
  28 |   uint64_t *ids=malloc(contributors*sizeof *ids);
  29 |   int error=0; size_t leaves=0;
  30 |   if(!f||!parent||!ids){ error=ENOMEM; goto finish; }
  31 |   for(size_t i=0;i<count;i++) parent[i]=SIZE_MAX;
  32 |   for(size_t i=0;i<count&&!error;i++){
  33 |     int status=index(capture,i,&f->nodes[i]);
  34 |     if(status){ error=status>0?status:EINVAL; break; }
  35 |     const struct mesh_reduce_node *n=&f->nodes[i];
  36 |     if(n->input[0]==SIZE_MAX && n->input[1]==SIZE_MAX){
  37 |       if(leaves==contributors || n->contributor>=contributors){ error=EINVAL; break; }
  38 |       ids[leaves++]=n->contributor;
  39 |     } else for(size_t j=0;j<2;j++){
  40 |       size_t child=n->input[j];
  41 |       if(child>=i || parent[child]!=SIZE_MAX){ error=EINVAL; break; }
  42 |       parent[child]=i;
  43 |     }
  44 |   }
  45 |   if(!error && leaves!=contributors) error=EINVAL;
  46 |   if(!error) for(size_t i=0;i+1<count;i++) if(parent[i]==SIZE_MAX){ error=EINVAL; break; }
  47 |   if(!error){
  48 |     qsort(ids,contributors,sizeof *ids,contributor_order);
  49 |     for(size_t i=1;i<contributors;i++) if(ids[i]==ids[i-1]){ error=EEXIST; break; }
  50 |   }
  51 | finish:
  52 |   free(ids);
  53 |   if(error){ free(parent); free(f); errno=error; return NULL; }
  54 |   atomic_init(&f->refs,1); f->count=count; f->parent=parent; return f;
  55 | }
  56 | size_t mesh_reduce_count(const mesh_reduce_function *f){ return f->count; }
  57 | const struct mesh_reduce_node *mesh_reduce_node(const mesh_reduce_function *f, size_t i){ return i<f->count?&f->nodes[i]:NULL; }
  58 | void mesh_reduce_free(mesh_reduce_function *f){
  59 |   if(f && atomic_fetch_sub(&f->refs,1)==1){ free(f->parent); free(f); } }
  60 | mesh_reduction *mesh_reduce_bind(mesh_reduce_function *f, uint64_t program, uint64_t invocation, uint64_t owner){
  61 |   mesh_reduction *r=calloc(1,sizeof *r+f->count*sizeof *r->states);
  62 |   if(!r){ errno=ENOMEM; return NULL; }
  63 |   atomic_fetch_add(&f->refs,1);
  64 |   r->function=f; r->program=program; r->invocation=invocation; r->owner=owner;
  65 |   return r;
  66 | }
  67 | struct mesh_reduce_token mesh_reduce_token(const mesh_reduction *r, size_t i){
  68 |   return (struct mesh_reduce_token){r->program,r->invocation,i}; }
  69 | int mesh_reduce_needed(const mesh_reduction *r, size_t i){
  70 |   if(i>=r->function->count) return 0;
  71 |   size_t parent=r->function->parent[i];
  72 |   return r->function->nodes[i].owner==r->owner || parent==SIZE_MAX || r->function->nodes[parent].owner==r->owner;
  73 | }
  74 | int mesh_reduce_ready(const mesh_reduction *r, size_t i){
  75 |   if(!mesh_reduce_needed(r,i)) return -EINVAL;
  76 |   int error=atomic_load_explicit(&r->error,memory_order_acquire);
  77 |   if(error) return -error;
  78 |   int state=atomic_load_explicit(&r->states[i],memory_order_acquire);
  79 |   return state<0?state:state==R_READY;
  80 | }
  81 | static int reduction_token(const mesh_reduction *r, struct mesh_reduce_token token){
  82 |   if(token.program!=r->program || token.invocation!=r->invocation) return -ESTALE;
  83 |   if(token.node>=r->function->count) return -EINVAL;
  84 |   return mesh_reduce_needed(r,(size_t)token.node)?0:-EXDEV;
  85 | }
  86 | int mesh_reduce_claim(mesh_reduction *r, struct mesh_reduce_token token){
  87 |   int error=reduction_token(r,token); if(error) return error;
  88 |   error=atomic_load_explicit(&r->error,memory_order_acquire); if(error) return -error;
  89 |   size_t i=(size_t)token.node;
  90 |   const struct mesh_reduce_node *n=&r->function->nodes[i];
  91 |   if(n->owner==r->owner && n->input[0]!=SIZE_MAX){
  92 |     for(size_t j=0;j<2;j++){
  93 |       int state=mesh_reduce_ready(r,n->input[j]);
  94 |       if(state!=1) return state;
  95 |     }
  96 |   }
  97 |   int waiting=R_WAITING;
  98 |   if(atomic_compare_exchange_strong_explicit(&r->states[i],&waiting,R_BORROWED,memory_order_acquire,memory_order_relaxed)) return 1;
  99 |   return waiting<0?waiting:0;
 100 | }
 101 | int mesh_reduce_commit(mesh_reduction *r, struct mesh_reduce_token token, int error){
 102 |   int status=reduction_token(r,token); if(status) return status;
 103 |   if(error<0) return -EINVAL;
 104 |   int borrowed=R_BORROWED;
 105 |   if(!atomic_compare_exchange_strong(&r->states[token.node],&borrowed,R_COMPLETING)) return -EINVAL;
 106 |   atomic_store_explicit(&r->states[token.node],error?-error:R_READY,memory_order_release);
 107 |   return 0;
 108 | }
 109 | int mesh_reduce_status(const mesh_reduction *r){
 110 |   int complete=1;
 111 |   for(size_t i=0;i<r->function->count;i++) if(mesh_reduce_needed(r,i)){
 112 |     int state=mesh_reduce_ready(r,i); if(state<0) return state; complete&=state;
 113 |   }
 114 |   return complete;
 115 | }
 116 | void mesh_reduce_cancel(mesh_reduction *r, int error){
 117 |   int clear=0; atomic_compare_exchange_strong(&r->error,&clear,error>0?error:ECANCELED); }
 118 | int mesh_reduce_retire(mesh_reduction *r){
 119 |   for(size_t i=0;i<r->function->count;i++){
 120 |     int state=atomic_load_explicit(&r->states[i],memory_order_acquire);
 121 |     if(state==R_BORROWED || state==R_COMPLETING) return EBUSY;
 122 |   }
 123 |   mesh_reduce_free(r->function); free(r); return 0;
 124 | }
```

### rdma/mesh-metal-executor.h — 33 lines

**X, lines 1–33.** Shared-event dependencies, mesh_call owner, mesh_selection cursor and span work API comprise alternative numerical readiness. Layout/addressing declarations may be realized using canonical rows, without retaining this executor.

```text
   1 | #ifndef MESH_METAL_EXECUTOR_H
   2 | #define MESH_METAL_EXECUTOR_H
   3 | #include "mesh-metal.h"
   4 | #include "mesh-functions.h"
   5 | struct mesh_metal_dependency { uint64_t value; size_t argument; int publish; };
   6 | size_t mesh_metal_row_chunk(struct mesh_ctx *context, struct mesh_scope scope, size_t row_bytes, size_t alignment);
   7 | int mesh_metal_bind_rows(const struct mesh_view *view, struct mesh_metal_layout layout,
   8 |   size_t rows, size_t row_bytes, size_t alignment, struct mesh_metal_rows *result);
   9 | id<MTLBuffer> mesh_metal_alias(id<MTLDevice> device, const struct mesh_view *view, struct mesh_metal_layout *layout);
  10 | id<MTLBuffer> mesh_metal_indices(id<MTLDevice> device, const mesh_call *call);
  11 | NSString *mesh_metal_source(void);
  12 | void mesh_metal_publish_cpu(uint32_t *generation, uint32_t value);
  13 | void mesh_metal_completion(id<MTLCommandBuffer> command, void (^complete)(int));
  14 | ptrdiff_t mesh_metal_advance(id<MTLSharedEvent> event, mesh_call *call, uint64_t base,
  15 |   const struct mesh_metal_dependency *dependencies, size_t count);
  16 | int mesh_metal_acquire(id<MTLSharedEvent> event, mesh_call *call, size_t argument, uint64_t value);
  17 | 
  18 | struct mesh_metal_reduction {
  19 |   mesh_call *call;
  20 |   size_t argument, source, elements, padding;
  21 |   struct mesh_selection selection;
  22 |   __unsafe_unretained id<MTLSharedEvent> event;
  23 |   uint64_t value;
  24 | };
  25 | struct mesh_metal_span { size_t edge; const uint32_t *indices; size_t count; };
  26 | struct mesh_metal_reduce_policy { uint32_t element_bytes, accumulator_bytes; int cancel_on_error; };
  27 | ptrdiff_t mesh_metal_select(struct mesh_metal_reduction *edges, size_t count,
  28 |   struct mesh_metal_reduce_policy policy, struct mesh_metal_span *work, size_t *work_count);
  29 | ptrdiff_t mesh_metal_reduce(struct mesh_metal_reduction *edges, size_t count,
  30 |   struct mesh_metal_reduce_policy policy, struct mesh_metal_span *work, size_t *work_count);
  31 | int mesh_metal_gather(mesh_executor *executor, struct mesh_metal_reduction *edges, size_t count,
  32 |   struct mesh_metal_reduce_policy policy, struct mesh_metal_span *work);
  33 | #endif
```

### rdma/mesh-metal-executor.m — 152 lines

**M, lines 1–48.** A/B: aliasing actual pages, offset validation and device completion callback. D: separate mesh_call indices and generation publication word. Geometry must not force foreign storage.

```text
   1 | #import "mesh-metal-executor.h"
   2 | #include "mesh-wire.h"
   3 | #include <errno.h>
   4 | #include <unistd.h>
   5 | 
   6 | id<MTLBuffer> mesh_metal_indices(id<MTLDevice> device, const mesh_call *call){
   7 |   size_t bytes; const uint32_t *indices=mesh_call_indices(call,&bytes);
   8 |   if(!bytes || bytes>device.maxBufferLength){ errno=EINVAL; return nil; }
   9 |   id<MTLBuffer> buffer=[device newBufferWithBytesNoCopy:(void*)indices length:bytes options:MTLResourceStorageModeShared deallocator:nil];
  10 |   if(!buffer) errno=ENOMEM;
  11 |   return buffer;
  12 | }
  13 | void mesh_metal_publish_cpu(uint32_t *generation,uint32_t value){ __atomic_store_n(generation,value,__ATOMIC_RELEASE); }
  14 | size_t mesh_metal_row_chunk(struct mesh_ctx *context,struct mesh_scope scope,size_t row_bytes,size_t alignment){
  15 |   struct mstream stream={.scope=scope};
  16 |   size_t header=sizeof(struct wire)+mesh_stream_header(&stream), stride=context->M->pgsz;
  17 |   if(!alignment || stride%alignment || !row_bytes){ errno=EINVAL; return 0; }
  18 |   size_t padding=(alignment-header%alignment)%alignment;
  19 |   if(header+padding>=stride || row_bytes>stride-header-padding){ errno=EOVERFLOW; return 0; }
  20 |   return row_bytes+padding;
  21 | }
  22 | int mesh_metal_bind_rows(const struct mesh_view *view,struct mesh_metal_layout layout,
  23 |   size_t rows,size_t row_bytes,size_t alignment,struct mesh_metal_rows *result){
  24 |   if(!rows || !alignment || view->stride%alignment || view->payload>=view->stride || !row_bytes) return EINVAL;
  25 |   size_t payload=view->capacity?view->capacity:view->stride-view->payload;
  26 |   size_t padding=(alignment-view->payload%alignment)%alignment;
  27 |   if(padding>=payload || row_bytes>payload-padding || rows>view->bytes/payload) return EINVAL;
  28 |   for(size_t i=0;i<rows;i++) if(view->pages[i]==UINT32_MAX || view->pages[i]!=view->pages[0]+i) return ENOTSUP;
  29 |   *result=(struct mesh_metal_rows){(uint64_t)view->pages[0]*view->stride+view->payload+padding-layout.origin,
  30 |     view->stride,(uint32_t)payload,(uint32_t)padding,1};
  31 |   return 0;
  32 | }
  33 | id<MTLBuffer> mesh_metal_alias(id<MTLDevice> device, const struct mesh_view *view, struct mesh_metal_layout *layout){
  34 |   size_t payload=view->capacity?view->capacity:view->stride-view->payload, count=view->bytes/payload+(view->bytes%payload!=0);
  35 |   if(!count){ errno=EINVAL; return nil; }
  36 |   uint32_t low=UINT32_MAX,high=0;
  37 |   for(size_t i=0;i<count;i++){ if(view->pages[i]<low) low=view->pages[i]; if(view->pages[i]>high) high=view->pages[i]; }
  38 |   size_t alignment=(size_t)getpagesize();
  39 |   uintptr_t begin=((uintptr_t)view->base+(size_t)low*view->stride)/alignment*alignment;
  40 |   uintptr_t end=((uintptr_t)view->base+((size_t)high+1)*view->stride+alignment-1)/alignment*alignment;
  41 |   if(end-begin>device.maxBufferLength){ errno=EOVERFLOW; return nil; }
  42 |   *layout=(struct mesh_metal_layout){begin-(uintptr_t)view->base,view->stride,view->payload,(uint32_t)payload};
  43 |   id<MTLBuffer> memory=[device newBufferWithBytesNoCopy:(void*)begin length:end-begin options:MTLResourceStorageModeShared deallocator:nil];
  44 |   if(!memory) errno=ENOMEM;
  45 |   return memory;
  46 | }
  47 | void mesh_metal_completion(id<MTLCommandBuffer> command, void (^complete)(int)){
  48 |   [command addCompletedHandler:^(id<MTLCommandBuffer> finished){ complete(finished.status==MTLCommandBufferStatusCompleted?0:EIO); }];
```

**X, lines 49–62.** Shared-event advance/acquire uses event values as another readiness authority.

```text
  49 | }
  50 | ptrdiff_t mesh_metal_advance(id<MTLSharedEvent> event, mesh_call *call, uint64_t base,
  51 |   const struct mesh_metal_dependency *dependencies, size_t count){
  52 |   uint64_t value=event.signaledValue;
  53 |   size_t i=0;
  54 |   for(;i<count && base+dependencies[i].value<=value;i++){
  55 |     int status=dependencies[i].publish?mesh_publish(call,dependencies[i].argument):mesh_complete(call,dependencies[i].argument,0);
  56 |     if(status) return status;
  57 |   }
  58 |   return (ptrdiff_t)i;
  59 | }
  60 | int mesh_metal_acquire(id<MTLSharedEvent> event, mesh_call *call, size_t argument, uint64_t value){
  61 |   struct mesh_view view;
  62 |   int status=mesh_acquire(call,argument,&view);
```

**M, lines 63–78.** A: generated literal-page gather/scatter address arithmetic; contiguous MeshMetalSpan requires proven physical contiguity. This arithmetic does not authorize its surrounding executor.

```text
  63 |   if(status==1) event.signaledValue=value;
  64 |   return status;
  65 | }
  66 | 
  67 | NSString *mesh_metal_source(void){
  68 |   return @"#include <metal_stdlib>\nusing namespace metal;\n"
  69 |     "struct MeshMetalLayout { ulong origin; uint stride,payload,capacity; };\n"
  70 |     "constant uint2 MeshMetalShape [[function_constant(23)]];\n"
  71 |     "constant uint MeshMetalCapacity [[function_constant(29)]];\n"
  72 |     "template<typename T> device const T* mesh_metal_page(device const uchar* memory,device const uint* pages,constant MeshMetalLayout& l,uint page){ uint stride=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.x:l.stride,payload=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.y:l.payload; return (device const T*)(memory+ulong(pages[page])*stride+payload-l.origin); }\n"
  73 |     "ulong mesh_metal_address(constant MeshMetalLayout& l,device const uint* pages,ulong i){ uint stride=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.x:l.stride,payload=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.y:l.payload; ulong n=is_function_constant_defined(MeshMetalCapacity)?MeshMetalCapacity:(l.capacity?l.capacity:stride-payload); return ulong(pages[i/n])*stride+payload+i%n-l.origin; }\n"
  74 |     "struct MeshMetalSpan { device uchar* memory; ulong first; uint stride,capacity;\n"
  75 |     "MeshMetalSpan(device uchar* m,device const uint* pages,constant MeshMetalLayout& l):memory(m){ stride=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.x:l.stride; uint payload=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.y:l.payload; capacity=(is_function_constant_defined(MeshMetalCapacity)?MeshMetalCapacity:(l.capacity?l.capacity:stride-payload))/2; first=ulong(pages[0])*stride+payload-l.origin; }\n"
  76 |     "void store_half(ulong i,half value) const thread { *(device half*)(memory+first+(i/capacity)*stride+(i%capacity)*2)=value; }\n"
  77 |     "template<typename T> void store(ulong i,T value) const thread { ulong n=(ulong(capacity)*2)/sizeof(T); *(device T*)(memory+first+(i/n)*stride+(i%n)*sizeof(T))=value; } };\n"
  78 |     "struct MeshMetalGather { device const uchar* memory; ulong first,origin; uint page,stride,payload,capacity,begin;\n"
```

**X, lines 79–152.** Independent select/consumed/published state, shared-event readiness, blocking gather loop and in-place rounding reduction replace page/index/accumulator algorithm. Keep numerical FP32 addition purpose only in actual accumulator pages.

```text
  79 |     "MeshMetalGather(device const uchar* m,device const uint* pages,constant MeshMetalLayout& l,ulong start,uint count,uint lane):memory(m),first(start),origin(l.origin){ stride=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.x:l.stride; payload=is_function_constant_defined(MeshMetalShape)?MeshMetalShape.y:l.payload; capacity=(is_function_constant_defined(MeshMetalCapacity)?MeshMetalCapacity:(l.capacity?l.capacity:stride-payload))/2; begin=uint(start/capacity); uint end=uint((start+count+capacity-1)/capacity); page=begin+lane<end?pages[begin+lane]:0; }\n"
  80 |     "half load_half(uint i) const thread { ulong at=first+i; ulong base=ulong(simd_shuffle(page,ushort(uint(at/capacity)-begin)))*stride+payload-origin; return *(device const half*)(memory+base+(at%capacity)*2); } };\n"
  81 |     "template<typename T> T mesh_metal_load(device const uchar* memory,device const uint* pages,constant MeshMetalLayout& layout,ulong i){ if constexpr(sizeof(T)==2) return *(device const T*)(memory+mesh_metal_address(layout,pages,i*2)); T value; thread uchar* out=(thread uchar*)&value; for(uint b=0;b<sizeof(T);b++) out[b]=memory[mesh_metal_address(layout,pages,i*sizeof(T)+b)]; return value; }\n"
  82 |     "template<typename T> void mesh_metal_store(device uchar* memory,device const uint* pages,constant MeshMetalLayout& layout,ulong i,T value){ if constexpr(sizeof(T)==2){ *(device T*)(memory+mesh_metal_address(layout,pages,i*2))=value; return; } thread uchar* in=(thread uchar*)&value; for(uint b=0;b<sizeof(T);b++) memory[mesh_metal_address(layout,pages,i*sizeof(T)+b)]=in[b]; }\n";
  83 | }
  84 | 
  85 | #define MESH_ADD_PAGES(T,A) \
  86 |   for(size_t k=0;k<count;k++){ \
  87 |     size_t page=indices[k]; \
  88 |     T *out=(T*)(destination->base+(size_t)destination->pages[page]*destination->stride+destination->payload+padding); \
  89 |     const T *in=(const T*)(source->base+(size_t)source->pages[page]*source->stride+source->payload+padding); \
  90 |     for(size_t i=0;i<elements;i++) out[i]=(T)((A)out[i]+(A)in[i]); \
  91 |   }
  92 | static void mesh_add_pages(struct mesh_metal_reduce_policy policy,const struct mesh_view *destination,
  93 |   const struct mesh_view *source,size_t elements,size_t padding,const uint32_t *indices,size_t count){
  94 |   if(policy.element_bytes==4){ MESH_ADD_PAGES(float,float) }
  95 |   else if(policy.accumulator_bytes==2){ MESH_ADD_PAGES(_Float16,_Float16) }
  96 |   else { MESH_ADD_PAGES(_Float16,float) }
  97 | }
  98 | #undef MESH_ADD_PAGES
  99 | static int mesh_metal_fail(const struct mesh_metal_reduction *edges,size_t count,struct mesh_metal_reduce_policy policy,int error){
 100 |   if(policy.cancel_on_error) for(size_t i=0;i<count;i++) mesh_call_cancel(edges[i].call,error);
 101 |   return -error;
 102 | }
 103 | ptrdiff_t mesh_metal_select(struct mesh_metal_reduction *edges,size_t count,struct mesh_metal_reduce_policy policy,
 104 |   struct mesh_metal_span *work,size_t *work_count){
 105 |   size_t done=0,n=0; *work_count=0;
 106 |   for(size_t i=0;i<count;i++){
 107 |     struct mesh_metal_reduction *edge=&edges[i];
 108 |     int status=mesh_call_status(edge->call); if(status<0) return mesh_metal_fail(edges,count,policy,-status);
 109 |     const struct mesh_view *local=mesh_call_view(edge->call,edge->argument);
 110 |     const struct mesh_view *remote=mesh_call_view(edge->call,edge->source);
 111 |     uint64_t generation=local->epoch->low;
 112 |     if(edge->value!=generation){ edge->value=generation; edge->selection=(struct mesh_selection){0}; }
 113 |     size_t pages=local->bytes/local->capacity+(local->bytes%local->capacity!=0);
 114 |     size_t publish=pages*(edge->event && edge->selection.published<local->bytes &&
 115 |       mesh_ready_pages(edge->call,edge->argument)<(ptrdiff_t)pages && edge->event.signaledValue>=edge->value);
 116 |     if(publish){
 117 |       status=mesh_publish(edge->call,edge->argument);
 118 |       if(status) return mesh_metal_fail(edges,count,policy,-status);
 119 |       ptrdiff_t flushed=mesh_flush(edge->call,edge->argument);
 120 |       if(flushed<0) return mesh_metal_fail(edges,count,policy,(int)-flushed);
 121 |     }
 122 |     const uint32_t *indices=NULL;
 123 |     ptrdiff_t selected=mesh_select_arrivals(edge->call,edge->source,edge->argument,&edge->selection,&indices);
 124 |     if(selected<0) return mesh_metal_fail(edges,count,policy,(int)-selected);
 125 |     edge->selection.consumed+=(size_t)selected;
 126 |     work[n]=(struct mesh_metal_span){i,indices,(size_t)selected}; n+=selected>0;
 127 |     done+=(edge->selection.consumed==pages) && mesh_epoch_equal(*local->epoch,*remote->epoch);
 128 |   }
 129 |   *work_count=n; return (ptrdiff_t)done;
 130 | }
 131 | ptrdiff_t mesh_metal_reduce(struct mesh_metal_reduction *edges,size_t count,struct mesh_metal_reduce_policy policy,
 132 |   struct mesh_metal_span *work,size_t *work_count){
 133 |   ptrdiff_t done=mesh_metal_select(edges,count,policy,work,work_count);
 134 |   if(done<0) return done;
 135 |   for(size_t i=0;i<*work_count;i++){
 136 |     struct mesh_metal_reduction *edge=&edges[work[i].edge];
 137 |     mesh_add_pages(policy,mesh_call_view(edge->call,edge->source),mesh_call_view(edge->call,edge->argument),
 138 |       edge->elements,edge->padding,work[i].indices,work[i].count);
 139 |   }
 140 |   return done;
 141 | }
 142 | int mesh_metal_gather(mesh_executor *executor,struct mesh_metal_reduction *edges,size_t count,struct mesh_metal_reduce_policy policy,struct mesh_metal_span *work){
 143 |   if((policy.element_bytes!=2 && policy.element_bytes!=4) || (policy.accumulator_bytes!=2 && policy.accumulator_bytes!=4) ||
 144 |      policy.accumulator_bytes<policy.element_bytes) return -EINVAL;
 145 |   ptrdiff_t done;
 146 |   do {
 147 |     mesh_progress(executor);
 148 |     size_t selected;
 149 |     done=mesh_metal_reduce(edges,count,policy,work,&selected);
 150 |   } while(done>=0 && (size_t)done<count);
 151 |   return done<0?(int)done:0;
 152 | }
```

### rdma/mesh-tensor.h — 44 lines

**X, lines 1–44.** Phase submission, tensor arena, transfer/channel plan and state API are not the prescribed mesh algorithm. Static numerical shape/index information remains configuration, without this execution interface.

```text
   1 | #ifndef MESH_TENSOR_H
   2 | #define MESH_TENSOR_H
   3 | #include <stddef.h>
   4 | #include <stdint.h>
   5 | struct mesh_tensor_view {
   6 |   uint64_t offset, size, shape[8], stride[8];
   7 |   uint32_t dtype, rank, pool, reserved;
   8 |   uint64_t page_start, origin;
   9 |   uint32_t page_stride, payload;
  10 | };
  11 | struct mesh_tensor_command {
  12 |   uint32_t kernel, argument_offset;
  13 |   uint32_t grid[3], group[3];
  14 | };
  15 | struct mesh_tensor_span { uint64_t stream, offset, bytes, target; };
  16 | struct mesh_tensor_binding { uint64_t stream, offset; uint32_t tensor, padding; };
  17 | void *mesh_tensor_create(const char *source);
  18 | const char *mesh_tensor_error(void *program);
  19 | int mesh_tensor_kernel(void *program, const char *name);
  20 | int mesh_tensor_reserve(void *program, size_t bytes, const struct mesh_tensor_view *views,
  21 |                         size_t count, const uint64_t *dimensions, size_t rank,
  22 |                         const uint32_t *arguments, size_t argument_count);
  23 | void *mesh_tensor_data(void *program);
  24 | void *mesh_tensor_memory(void *program);
  25 | void mesh_tensor_memory_free(void *memory);
  26 | int mesh_tensor_phase(void *program, uint32_t phase, const struct mesh_tensor_command *commands, size_t count);
  27 | int mesh_tensor_submit(void *program, uint32_t phase);
  28 | int mesh_tensor_status(void *program);
  29 | void mesh_tensor_free(void *program);
  30 | void *mesh_tensor_transport(void *context);
  31 | size_t mesh_tensor_transport_bytes(void *transport);
  32 | int mesh_tensor_transport_progress(void *transport);
  33 | int mesh_tensor_transport_free(void *transport);
  34 | void *mesh_tensor_channel(void *transport, void *program, int peer, int receive,
  35 |                           uint32_t channel, uint64_t epoch, const unsigned char plan[32], size_t bytes);
  36 | int mesh_tensor_transfer(void *channel, uint64_t epoch, uint64_t offset, uint64_t bytes);
  37 | int mesh_tensor_transfer_plan(void *channel, const struct mesh_tensor_span *spans, size_t count,
  38 |                               const struct mesh_tensor_binding *bindings, size_t binding_count);
  39 | int mesh_tensor_transfer_release(void *channel);
  40 | int mesh_tensor_transfer_commit(void *channel);
  41 | int mesh_tensor_transfer_status(void *channel);
  42 | void mesh_tensor_transfer_cancel(void *channel, int error);
  43 | int mesh_tensor_channel_free(void *channel);
  44 | #endif
```

### rdma/mesh-tensor.m — 452 lines

**M, lines 1–64.** A/B: numerical kernel compilation before invocation. X: phase arrays and independent arena/pending algorithm state in the program object.

```text
   1 | #import <Foundation/Foundation.h>
   2 | #import <Metal/Metal.h>
   3 | #include "mesh-tensor.h"
   4 | #include <errno.h>
   5 | 
   6 | @interface MeshTensorProgram : NSObject
   7 | @property id<MTLDevice> device;
   8 | @property id<MTLCommandQueue> queue;
   9 | @property id<MTLLibrary> library;
  10 | @property id<MTLBuffer> arena;
  11 | @property id<MTLBuffer> views;
  12 | @property id<MTLBuffer> dimensions;
  13 | @property id<MTLBuffer> arguments;
  14 | @property id<MTLBuffer> receive;
  15 | @property id<MTLBuffer> transmit;
  16 | @property id<MTLBuffer> pages;
  17 | @property NSMutableArray<id<MTLComputePipelineState>> *kernels;
  18 | @property NSMutableArray *phases;
  19 | @property NSMutableArray<NSNumber*> *counts;
  20 | @property NSMutableArray<NSData*> *specifications;
  21 | @property id<MTLCommandBuffer> pending;
  22 | @property NSString *error;
  23 | @end
  24 | @implementation MeshTensorProgram
  25 | @end
  26 | 
  27 | static NSString *mesh_tensor_creation_error;
  28 | 
  29 | void *mesh_tensor_create(const char *source){
  30 |   @autoreleasepool {
  31 |     MeshTensorProgram *program=[MeshTensorProgram new];
  32 |     program.device=MTLCreateSystemDefaultDevice();
  33 |     program.queue=[program.device newCommandQueue];
  34 |     program.kernels=[NSMutableArray new];
  35 |     program.phases=[NSMutableArray new];
  36 |     program.counts=[NSMutableArray new];
  37 |     program.specifications=[NSMutableArray new];
  38 |     MTLCompileOptions *options=[MTLCompileOptions new];
  39 |     options.mathMode=MTLMathModeSafe;
  40 |     NSError *error=nil;
  41 |     program.library=[program.device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
  42 |     if(!program.library){ mesh_tensor_creation_error=error.localizedDescription; return NULL; }
  43 |     return (__bridge_retained void*)program;
  44 |   }
  45 | }
  46 | 
  47 | const char *mesh_tensor_error(void *handle){
  48 |   MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  49 |   return (program?program.error:mesh_tensor_creation_error).UTF8String;
  50 | }
  51 | 
  52 | int mesh_tensor_kernel(void *handle,const char *name){
  53 |   MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  54 |   MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
  55 |   descriptor.computeFunction=[program.library newFunctionWithName:[NSString stringWithUTF8String:name]];
  56 |   descriptor.supportIndirectCommandBuffers=YES;
  57 |   NSError *error=nil;
  58 |   id<MTLComputePipelineState> kernel=[program.device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
  59 |   if(!kernel){ program.error=error.localizedDescription; return -1; }
  60 |   int index=(int)program.kernels.count;
  61 |   [program.kernels addObject:kernel];
  62 |   return index;
  63 | }
  64 | 
```

**X, lines 65–91.** Foreign tensor arena growth/copy and separate metadata buffers, including mutable allocation, violate actual page storage.

```text
  65 | static id<MTLBuffer> mesh_tensor_buffer(MeshTensorProgram *program,id<MTLBuffer> previous,size_t bytes){
  66 |   if(previous.length>=bytes) return previous;
  67 |   id<MTLBuffer> buffer=[program.device newBufferWithLength:MAX(bytes,256) options:MTLResourceStorageModeShared];
  68 |   if(buffer && previous) memcpy(buffer.contents,previous.contents,previous.length);
  69 |   return buffer;
  70 | }
  71 | 
  72 | int mesh_tensor_reserve(void *handle,size_t bytes,const struct mesh_tensor_view *views,size_t count,
  73 |                         const uint64_t *dimensions,size_t rank,const uint32_t *arguments,size_t argument_count){
  74 |   MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  75 |   if(program.pending && program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
  76 |   id<MTLBuffer> arena=mesh_tensor_buffer(program,program.arena,bytes);
  77 |   id<MTLBuffer> metadata=mesh_tensor_buffer(program,program.views,count*sizeof(*views));
  78 |   id<MTLBuffer> dims=mesh_tensor_buffer(program,program.dimensions,rank*sizeof(*dimensions));
  79 |   id<MTLBuffer> args=mesh_tensor_buffer(program,program.arguments,argument_count*sizeof(*arguments));
  80 |   if(!arena || !metadata || !dims || !args){ program.error=@"persistent tensor allocation failed"; return ENOMEM; }
  81 |   program.arena=arena; program.views=metadata; program.dimensions=dims; program.arguments=args;
  82 |   memcpy(program.views.contents,views,count*sizeof(*views));
  83 |   memcpy(program.dimensions.contents,dimensions,rank*sizeof(*dimensions));
  84 |   memcpy(program.arguments.contents,arguments,argument_count*sizeof(*arguments));
  85 |   return 0;
  86 | }
  87 | 
  88 | void *mesh_tensor_data(void *handle){ return ((__bridge MeshTensorProgram*)handle).arena.contents; }
  89 | void *mesh_tensor_memory(void *handle){ return (__bridge_retained void*)((__bridge MeshTensorProgram*)handle).arena; }
  90 | void mesh_tensor_memory_free(void *handle){ (void)CFBridgingRelease(handle); }
  91 | 
```

**X, lines 92–160.** Phase recording/submission, whole-program pending command authority and status form an alternative execution flow.

```text
  92 | int mesh_tensor_phase(void *handle,uint32_t phase,const struct mesh_tensor_command *commands,size_t count){
  93 |   MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
  94 |   if(program.pending && program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
  95 |   MTLIndirectCommandBufferDescriptor *descriptor=[MTLIndirectCommandBufferDescriptor new];
  96 |   descriptor.commandTypes=MTLIndirectCommandTypeConcurrentDispatch;
  97 |   descriptor.inheritBuffers=NO;
  98 |   descriptor.inheritPipelineState=NO;
  99 |   descriptor.maxKernelBufferBindCount=7;
 100 |   id<MTLIndirectCommandBuffer> buffer=phase<program.phases.count && program.counts[phase].unsignedIntegerValue>=count?program.phases[phase]:nil;
 101 |   if(buffer) [buffer resetWithRange:NSMakeRange(0,program.counts[phase].unsignedIntegerValue)];
 102 |   else buffer=[program.device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:MAX(count,1) options:MTLResourceStorageModePrivate];
 103 |   if(!buffer){ program.error=@"persistent tensor command allocation failed"; return ENOMEM; }
 104 |   for(size_t i=0;i<count;i++){
 105 |     id<MTLIndirectComputeCommand> command=[buffer indirectComputeCommandAtIndex:i];
 106 |     [command setComputePipelineState:program.kernels[commands[i].kernel]];
 107 |     [command setKernelBuffer:program.arena offset:0 atIndex:0];
 108 |     [command setKernelBuffer:program.views offset:0 atIndex:1];
 109 |     [command setKernelBuffer:program.dimensions offset:0 atIndex:2];
 110 |     [command setKernelBuffer:program.receive?:program.arena offset:0 atIndex:3];
 111 |     [command setKernelBuffer:program.transmit?:program.arena offset:0 atIndex:4];
 112 |     [command setKernelBuffer:program.pages?:program.arena offset:0 atIndex:5];
 113 |     [command setKernelBuffer:program.arguments offset:(size_t)commands[i].argument_offset*sizeof(uint32_t) atIndex:6];
 114 |     [command concurrentDispatchThreadgroups:MTLSizeMake(commands[i].grid[0],commands[i].grid[1],commands[i].grid[2])
 115 |                      threadsPerThreadgroup:MTLSizeMake(commands[i].group[0],commands[i].group[1],commands[i].group[2])];
 116 |     [command setBarrier];
 117 |   }
 118 |   while(program.phases.count<=phase){ [program.phases addObject:[NSNull null]]; [program.counts addObject:@0]; [program.specifications addObject:[NSData data]]; }
 119 |   program.phases[phase]=buffer;
 120 |   program.counts[phase]=@(count);
 121 |   program.specifications[phase]=[NSData dataWithBytes:commands length:count*sizeof(*commands)];
 122 |   return 0;
 123 | }
 124 | 
 125 | int mesh_tensor_submit(void *handle,uint32_t phase){
 126 |   @autoreleasepool {
 127 |     MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
 128 |     if(program.pending && program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
 129 |     id<MTLCommandBuffer> command=[program.queue commandBuffer];
 130 |     id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
 131 |     [encoder useResource:program.arena usage:MTLResourceUsageRead|MTLResourceUsageWrite];
 132 |     [encoder useResource:program.views usage:MTLResourceUsageRead];
 133 |     [encoder useResource:program.dimensions usage:MTLResourceUsageRead];
 134 |     [encoder useResource:program.arguments usage:MTLResourceUsageRead];
 135 |     if(program.receive) [encoder useResource:program.receive usage:MTLResourceUsageRead];
 136 |     if(program.transmit) [encoder useResource:program.transmit usage:MTLResourceUsageRead|MTLResourceUsageWrite];
 137 |     if(program.pages) [encoder useResource:program.pages usage:MTLResourceUsageRead];
 138 |     [encoder executeCommandsInBuffer:program.phases[phase] withRange:NSMakeRange(0,program.counts[phase].unsignedIntegerValue)];
 139 |     [encoder endEncoding];
 140 |     [command commit];
 141 |     program.pending=command;
 142 |     return 0;
 143 |   }
 144 | }
 145 | 
 146 | int mesh_tensor_status(void *handle){
 147 |   MeshTensorProgram *program=(__bridge MeshTensorProgram*)handle;
 148 |   if(!program.pending || program.pending.status==MTLCommandBufferStatusCompleted) return 1;
 149 |   if(program.pending.status==MTLCommandBufferStatusError){ program.error=program.pending.error.localizedDescription; return -EIO; }
 150 |   return 0;
 151 | }
 152 | 
 153 | void mesh_tensor_free(void *handle){
 154 |   (void)CFBridgingRelease(handle);
 155 | }
 156 | 
 157 | #import "mesh-metal-executor.h"
 158 | #include "mesh-wire.h"
 159 | 
 160 | @class MeshTensorChannel;
```

**X, lines 161–249.** Transfer/commit kernels copy between arena and pool, plus separate channel state and arbitrary transport extent. These are the prohibited copies and wrapping synchronization abstractions.

```text
 161 | @interface MeshTensorTransport : NSObject
 162 | @property struct mesh_ctx *context;
 163 | @property mesh_executor *executor;
 164 | @property id<MTLDevice> device;
 165 | @property id<MTLBuffer> receive;
 166 | @property id<MTLBuffer> transmit;
 167 | @property struct mesh_metal_layout receiveLayout;
 168 | @property struct mesh_metal_layout transmitLayout;
 169 | @property id<MTLComputePipelineState> transferKernel;
 170 | @property id<MTLComputePipelineState> commitKernel;
 171 | @property NSMutableArray<MeshTensorChannel*> *channels;
 172 | @property size_t bytes;
 173 | @end
 174 | @implementation MeshTensorTransport
 175 | @end
 176 | 
 177 | struct mesh_tensor_transfer_parameters { uint64_t offset, bytes, extent, origin; uint32_t stride, payload, receive, spans; };
 178 | @interface MeshTensorChannel : NSObject
 179 | @property MeshTensorTransport *transport;
 180 | @property MeshTensorProgram *program;
 181 | @property mesh_call *call;
 182 | @property id<MTLBuffer> indices;
 183 | @property id<MTLBuffer> parameters;
 184 | @property id<MTLBuffer> spans;
 185 | @property id<MTLBuffer> bindings;
 186 | @property id<MTLBuffer> arena;
 187 | @property id<MTLIndirectCommandBuffer> command;
 188 | @property id<MTLIndirectCommandBuffer> commit;
 189 | @property id<MTLCommandBuffer> pending;
 190 | @property uint64_t epoch;
 191 | @property int state;
 192 | @property int error;
 193 | @property BOOL receiving;
 194 | @property size_t bytes;
 195 | @property size_t streamBytes;
 196 | @property size_t spanCount;
 197 | @property size_t bindingCount;
 198 | @end
 199 | @implementation MeshTensorChannel
 200 | @end
 201 | 
 202 | void *mesh_tensor_transport(void *context){
 203 |   @autoreleasepool {
 204 |     MeshTensorTransport *transport=[MeshTensorTransport new];
 205 |     transport.context=context;
 206 |     struct mesh_ctx *ctx=context;
 207 |     size_t window=MIN(ctx->M->pool,ctx->M->arena)/4;
 208 |     transport.executor=mesh_executor_create(ctx,window);
 209 |     if(!transport.executor) return NULL;
 210 |     transport.bytes=MIN((size_t)1048576,window*(mesh_pay(ctx->M)-sizeof(struct mesh_frame)));
 211 |     transport.device=MTLCreateSystemDefaultDevice();
 212 |     struct mesh_metal_layout receive,transmit;
 213 |     transport.receive=mesh_metal_receive_pool(transport.device,ctx,&receive);
 214 |     transport.transmit=mesh_metal_transmit_pool(transport.device,ctx,&transmit);
 215 |     transport.receiveLayout=receive; transport.transmitLayout=transmit;
 216 |     transport.channels=[NSMutableArray new];
 217 |     NSString *source=@"#include <metal_stdlib>\nusing namespace metal;\n"
 218 |       "struct Transfer { ulong offset,bytes,extent,origin; uint stride,payload,receive,spans; };\n"
 219 |       "struct Span { ulong stream,offset,bytes,target; };\n"
 220 |       "kernel void mesh_tensor_copy(device uchar* arena [[buffer(0)]],device uchar* pool [[buffer(1)]],device const uint* pages [[buffer(2)]],constant Transfer& p [[buffer(3)]],device const Span* spans [[buffer(4)]],uint t [[thread_position_in_grid]]){\n"
 221 |       "if(t>=p.extent) return; ulong n=p.stride-p.payload; ulong at=ulong(pages[t/n])*p.stride+p.payload+t%n-p.origin;\n"
 222 |       "if(t>=p.bytes){ if(!p.receive) pool[at]=0; return; } ulong logical=p.offset+t; uint low=0,high=p.spans;\n"
 223 |       "while(low+1<high){ uint middle=(low+high)/2; if(spans[middle].stream<=logical) low=middle; else high=middle; }\n"
 224 |       "ulong address=spans[low].offset+logical-spans[low].stream; if(p.receive) arena[address]=pool[at]; else pool[at]=arena[address]; }\n"
 225 |       "kernel void mesh_tensor_commit(device uchar* arena [[buffer(0)]],constant Transfer& p [[buffer(1)]],device const Span* spans [[buffer(2)]],uint tid [[thread_position_in_grid]]){\n"
 226 |       "for(ulong t=tid;t<p.bytes;t+=65536){ uint low=0,high=p.spans; while(low+1<high){ uint middle=(low+high)/2; if(spans[middle].stream<=t) low=middle; else high=middle; }\n"
 227 |       "ulong at=t-spans[low].stream; arena[spans[low].target+at]=arena[spans[low].offset+at]; }}\n";
 228 |     NSError *error=nil;
 229 |     id<MTLLibrary> library=[transport.device newLibraryWithSource:source options:nil error:&error];
 230 |     MTLComputePipelineDescriptor *descriptor=[MTLComputePipelineDescriptor new];
 231 |     descriptor.computeFunction=[library newFunctionWithName:@"mesh_tensor_copy"];
 232 |     descriptor.supportIndirectCommandBuffers=YES;
 233 |     transport.transferKernel=[transport.device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
 234 |     descriptor.computeFunction=[library newFunctionWithName:@"mesh_tensor_commit"];
 235 |     transport.commitKernel=[transport.device newComputePipelineStateWithDescriptor:descriptor options:MTLPipelineOptionNone reflection:nil error:&error];
 236 |     if(!transport.receive || !transport.transmit || !transport.transferKernel || !transport.commitKernel){
 237 |       mesh_tensor_creation_error=error.localizedDescription?:@"mesh page alias allocation failed";
 238 |       mesh_executor_free(transport.executor); return NULL;
 239 |     }
 240 |     return (__bridge_retained void*)transport;
 241 |   }
 242 | }
 243 | 
 244 | size_t mesh_tensor_transport_bytes(void *handle){ return ((__bridge MeshTensorTransport*)handle).bytes; }
 245 | 
 246 | static int mesh_tensor_extent(void *capture,size_t index,struct mesh_extent *extent){
 247 |   *extent=*(struct mesh_extent*)capture; return 0;
 248 | }
 249 | 
```

**X, lines 250–369.** Channel binding, transfer plan, binding mutation, release/commit phases and dynamic transfer command creation maintain the alternative storage/execution flow.

```text
 250 | void *mesh_tensor_channel(void *handle,void *program,int peer,int receive,uint32_t channel,uint64_t epoch,const unsigned char plan[32],size_t bytes){
 251 |   MeshTensorTransport *transport=(__bridge MeshTensorTransport*)handle;
 252 |   struct mesh_extent extent={.bytes=bytes,.peer=peer,.receive=receive};
 253 |   mesh_function *function=mesh_compile(1,mesh_tensor_extent,&extent);
 254 |   if(!function) return NULL;
 255 |   struct mesh_scope scope={.epoch={epoch,1}};
 256 |   memcpy(scope.plan,plan,32);
 257 |   mesh_call *call=mesh_bind_scoped(transport.executor,function,channel,scope);
 258 |   mesh_function_free(function);
 259 |   if(!call) return NULL;
 260 |   mesh_call_retain_transmit(call);
 261 |   MeshTensorChannel *binding=[MeshTensorChannel new];
 262 |   binding.transport=transport; binding.program=(__bridge MeshTensorProgram*)program;
 263 |   binding.call=call; binding.receiving=receive;
 264 |   binding.epoch=epoch;
 265 |   binding.bytes=bytes;
 266 |   binding.indices=mesh_metal_indices(transport.device,call);
 267 |   binding.parameters=[transport.device newBufferWithLength:256 options:MTLResourceStorageModeShared];
 268 |   size_t count=binding.program.views.length/sizeof(struct mesh_tensor_view);
 269 |   binding.spans=[transport.device newBufferWithLength:MAX(256,count*sizeof(struct mesh_tensor_span)) options:MTLResourceStorageModeShared];
 270 |   binding.bindings=[transport.device newBufferWithLength:MAX(256,count*sizeof(struct mesh_tensor_binding)) options:MTLResourceStorageModeShared];
 271 |   if(!binding.indices || !binding.parameters || !binding.spans || !binding.bindings) binding.error=ENOMEM;
 272 |   MeshTensorProgram *tensor=binding.program;
 273 |   BOOL changed=tensor.receive!=transport.receive || !tensor.pages || tensor.pages.length<binding.indices.length;
 274 |   tensor.receive=transport.receive; tensor.transmit=transport.transmit;
 275 |   tensor.pages=mesh_tensor_buffer(tensor,tensor.pages,binding.indices.length);
 276 |   if(!tensor.pages) binding.error=ENOMEM;
 277 |   if(changed) for(NSUInteger i=0;i<tensor.specifications.count;i++){
 278 |     NSData *commands=tensor.specifications[i];
 279 |     int status=mesh_tensor_phase((__bridge void*)tensor,(uint32_t)i,commands.bytes,commands.length/sizeof(struct mesh_tensor_command));
 280 |     if(status) binding.error=status;
 281 |   }
 282 |   [transport.channels addObject:binding];
 283 |   return (__bridge_retained void*)binding;
 284 | }
 285 | 
 286 | int mesh_tensor_transfer_plan(void *handle,const struct mesh_tensor_span *spans,size_t count,const struct mesh_tensor_binding *bindings,size_t binding_count){
 287 |   MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
 288 |   if(channel.error) return channel.error;
 289 |   if(channel.state && mesh_call_status(channel.call)!=1) return EBUSY;
 290 |   if(count*sizeof(*spans)>channel.spans.length || binding_count*sizeof(*bindings)>channel.bindings.length) return EOVERFLOW;
 291 |   memcpy(channel.spans.contents,spans,count*sizeof(*spans));
 292 |   memcpy(channel.bindings.contents,bindings,binding_count*sizeof(*bindings));
 293 |   channel.spanCount=count; channel.bindingCount=binding_count;
 294 |   channel.streamBytes=count?spans[count-1].stream+spans[count-1].bytes:0;
 295 |   return 0;
 296 | }
 297 | 
 298 | int mesh_tensor_transfer_release(void *handle){
 299 |   MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
 300 |   if(channel.state!=4) return 0;
 301 |   if(channel.program.pending && channel.program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
 302 |   struct mesh_tensor_view *views=channel.program.views.contents;
 303 |   const struct mesh_tensor_binding *bindings=channel.bindings.contents;
 304 |   for(size_t i=0;i<channel.bindingCount;i++){
 305 |     views[bindings[i].tensor].pool=0; views[bindings[i].tensor].offset=bindings[i].offset;
 306 |   }
 307 |   int status=mesh_complete(channel.call,0,channel.error);
 308 |   channel.state=3;
 309 |   return status;
 310 | }
 311 | 
 312 | static void mesh_tensor_transfer_record(MeshTensorChannel *channel){
 313 |   MTLIndirectCommandBufferDescriptor *descriptor=[MTLIndirectCommandBufferDescriptor new];
 314 |   descriptor.commandTypes=MTLIndirectCommandTypeConcurrentDispatch;
 315 |   descriptor.inheritBuffers=NO; descriptor.inheritPipelineState=NO;
 316 |   descriptor.maxKernelBufferBindCount=5;
 317 |   channel.command=[channel.transport.device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:1 options:MTLResourceStorageModePrivate];
 318 |   channel.arena=channel.program.arena;
 319 |   id<MTLIndirectComputeCommand> command=[channel.command indirectComputeCommandAtIndex:0];
 320 |   [command setComputePipelineState:channel.transport.transferKernel];
 321 |   [command setKernelBuffer:channel.arena offset:0 atIndex:0];
 322 |   [command setKernelBuffer:channel.receiving?channel.transport.receive:channel.transport.transmit offset:0 atIndex:1];
 323 |   [command setKernelBuffer:channel.indices offset:0 atIndex:2];
 324 |   [command setKernelBuffer:channel.parameters offset:0 atIndex:3];
 325 |   [command setKernelBuffer:channel.spans offset:0 atIndex:4];
 326 |   [command concurrentDispatchThreadgroups:MTLSizeMake((channel.bytes+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 327 |   channel.commit=[channel.transport.device newIndirectCommandBufferWithDescriptor:descriptor maxCommandCount:1 options:MTLResourceStorageModePrivate];
 328 |   command=[channel.commit indirectComputeCommandAtIndex:0];
 329 |   [command setComputePipelineState:channel.transport.commitKernel];
 330 |   [command setKernelBuffer:channel.arena offset:0 atIndex:0];
 331 |   [command setKernelBuffer:channel.parameters offset:0 atIndex:1];
 332 |   [command setKernelBuffer:channel.spans offset:0 atIndex:2];
 333 |   [command concurrentDispatchThreadgroups:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
 334 | }
 335 | 
 336 | int mesh_tensor_transfer_commit(void *handle){
 337 |   MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
 338 |   if(mesh_call_status(channel.call)!=1 || channel.program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
 339 |   struct mesh_tensor_transfer_parameters *parameters=channel.parameters.contents;
 340 |   parameters->bytes=channel.streamBytes;
 341 |   id<MTLCommandBuffer> command=[channel.program.queue commandBuffer];
 342 |   id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
 343 |   [encoder useResource:channel.arena usage:MTLResourceUsageRead|MTLResourceUsageWrite];
 344 |   [encoder useResource:channel.parameters usage:MTLResourceUsageRead];
 345 |   [encoder useResource:channel.spans usage:MTLResourceUsageRead];
 346 |   [encoder executeCommandsInBuffer:channel.commit withRange:NSMakeRange(0,1)];
 347 |   [encoder endEncoding]; [command commit]; channel.program.pending=command;
 348 |   return 0;
 349 | }
 350 | 
 351 | int mesh_tensor_transfer(void *handle,uint64_t epoch,uint64_t offset,uint64_t bytes){
 352 |   MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
 353 |   if(channel.state && mesh_call_status(channel.call)!=1) return EBUSY;
 354 |   if(bytes>channel.bytes || offset>channel.streamBytes || bytes>channel.streamBytes-offset) return EOVERFLOW;
 355 |   if(channel.state){
 356 |     int status=mesh_call_rearm(channel.call,(struct mesh_epoch){channel.epoch,epoch});
 357 |     if(status) return status;
 358 |   }
 359 |   else if(epoch!=1) return EINVAL;
 360 |   if(channel.arena!=channel.program.arena) mesh_tensor_transfer_record(channel);
 361 |   const struct mesh_view *view=mesh_call_view(channel.call,0);
 362 |   struct mesh_metal_layout layout=channel.receiving?channel.transport.receiveLayout:channel.transport.transmitLayout;
 363 |   struct mesh_tensor_transfer_parameters parameters={offset,bytes,channel.bytes,layout.origin,view->stride,view->payload,channel.receiving,(uint32_t)channel.spanCount};
 364 |   memcpy(channel.parameters.contents,&parameters,sizeof parameters);
 365 |   channel.state=1; channel.error=0;
 366 |   mesh_request(channel.call,0);
 367 |   return 0;
 368 | }
 369 | 
```

**X, lines 370–452.** Channel states 1/2/3/4 drive transfer and completion; page-table copying and runtime view rebinding duplicate storage authority. Teardown users disappear with this algorithm; preserve substrate lifetime obligations.

```text
 370 | static void mesh_tensor_channel_progress(MeshTensorChannel *channel){
 371 |   if(channel.state==1){
 372 |     struct mesh_view view;
 373 |     int status=mesh_acquire(channel.call,0,&view);
 374 |     if(status==1){
 375 |       id<MTLCommandBuffer> command=[channel.program.queue commandBuffer];
 376 |       id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
 377 |       [encoder useResource:channel.arena usage:MTLResourceUsageRead|MTLResourceUsageWrite];
 378 |       [encoder useResource:channel.receiving?channel.transport.receive:channel.transport.transmit usage:MTLResourceUsageRead|MTLResourceUsageWrite];
 379 |       [encoder useResource:channel.indices usage:MTLResourceUsageRead];
 380 |       [encoder useResource:channel.parameters usage:MTLResourceUsageRead];
 381 |       [encoder useResource:channel.spans usage:MTLResourceUsageRead];
 382 |       [encoder executeCommandsInBuffer:channel.command withRange:NSMakeRange(0,1)];
 383 |       [encoder endEncoding]; [command commit];
 384 |       channel.pending=command; channel.program.pending=command; channel.state=2;
 385 |     } else if(status<0) channel.error=-status;
 386 |   }
 387 |   if(channel.state==2 && channel.pending.status>=MTLCommandBufferStatusCompleted){
 388 |     int error=channel.error?channel.error:(channel.pending.status==MTLCommandBufferStatusError?EIO:0);
 389 |     channel.error=error;
 390 |     if(!error && channel.receiving && channel.bindingCount && channel.streamBytes<=channel.bytes){
 391 |       memcpy(channel.program.pages.contents,channel.indices.contents,channel.indices.length);
 392 |       const struct mesh_view *view=mesh_call_view(channel.call,0);
 393 |       struct mesh_tensor_view *views=channel.program.views.contents;
 394 |       const struct mesh_tensor_binding *bindings=channel.bindings.contents;
 395 |       for(size_t i=0;i<channel.bindingCount;i++){
 396 |         struct mesh_tensor_view *tensor=&views[bindings[i].tensor];
 397 |         tensor->offset=bindings[i].stream; tensor->pool=1; tensor->page_start=0;
 398 |         tensor->origin=channel.transport.receiveLayout.origin; tensor->page_stride=view->stride; tensor->payload=view->payload;
 399 |       }
 400 |       channel.state=4;
 401 |     } else { mesh_complete(channel.call,0,error); channel.state=3; }
 402 |   }
 403 |   if(channel.state==4 && channel.error) mesh_tensor_transfer_release((__bridge void*)channel);
 404 | }
 405 | 
 406 | int mesh_tensor_transport_progress(void *handle){
 407 |   @autoreleasepool {
 408 |     MeshTensorTransport *transport=(__bridge MeshTensorTransport*)handle;
 409 |     int activity=mesh_progress(transport.executor);
 410 |     for(MeshTensorChannel *channel in transport.channels) mesh_tensor_channel_progress(channel);
 411 |     return activity;
 412 |   }
 413 | }
 414 | 
 415 | int mesh_tensor_transfer_status(void *handle){
 416 |   MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
 417 |   int status=mesh_call_status(channel.call);
 418 |   return channel.error?-channel.error:channel.state==4?2:status;
 419 | }
 420 | 
 421 | void mesh_tensor_transfer_cancel(void *handle,int error){
 422 |   MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
 423 |   channel.error=error; mesh_call_cancel(channel.call,error);
 424 | }
 425 | 
 426 | int mesh_tensor_channel_free(void *handle){
 427 |   MeshTensorChannel *channel=(__bridge MeshTensorChannel*)handle;
 428 |   if(channel.program.pending && channel.program.pending.status<MTLCommandBufferStatusCompleted) return EBUSY;
 429 |   int status=mesh_call_status(channel.call)<0?mesh_call_abandon(channel.call):mesh_call_retire(channel.call);
 430 |   if(status) return status;
 431 |   [channel.transport.channels removeObjectIdenticalTo:channel];
 432 |   BOOL shared=NO;
 433 |   for(MeshTensorChannel *other in channel.transport.channels) shared|=other.program==channel.program;
 434 |   if(!shared){
 435 |     MeshTensorProgram *tensor=channel.program;
 436 |     tensor.receive=nil; tensor.transmit=nil;
 437 |     for(NSUInteger i=0;i<tensor.specifications.count;i++){
 438 |       NSData *commands=tensor.specifications[i];
 439 |       mesh_tensor_phase((__bridge void*)tensor,(uint32_t)i,commands.bytes,commands.length/sizeof(struct mesh_tensor_command));
 440 |     }
 441 |   }
 442 |   channel.transport=nil; channel.program=nil;
 443 |   (void)CFBridgingRelease(handle); return 0;
 444 | }
 445 | 
 446 | int mesh_tensor_transport_free(void *handle){
 447 |   MeshTensorTransport *transport=(__bridge MeshTensorTransport*)handle;
 448 |   if(transport.channels.count) return EBUSY;
 449 |   int status=mesh_executor_free(transport.executor);
 450 |   if(status) return status;
 451 |   (void)CFBridgingRelease(handle); return 0;
 452 | }
```

### rdma/mesh.h — 147 lines

**B, lines 1–71.** Actual RDMA shared region, page addressing, descriptor rings and bridge lifecycle. These are substrate mechanics, not a permitted second numerical readiness system. Ring head/tail atomics preserve the real SPSC binding.

```text
   1 | 
   2 | #ifndef MESH_H
   3 | #define MESH_H
   4 | #include <stdint.h>
   5 | #include <stddef.h>
   6 | #include <stdatomic.h>
   7 | #include <string.h>
   8 | #define MESH_MAGIC   0x4d455348u
   9 | #define MESH_NAME    "/mesh0"
  10 | #define MESH_PORT    "18519"
  11 | #define MESH_MODE    0666
  12 | #define MESH_VERSION 5u
  13 | #define MESH_CLIENT_DRAIN UINT64_MAX
  14 | #define MESH_CLIENT_DETACH (UINT64_MAX-1)
  15 | #define MESH_CL      128
  16 | #define MESH_RING    65536
  17 | #define MESH_OFF     16
  18 | #define RINGS        ((sizeof(struct hdr)+MESH_CL-1)/MESH_CL*MESH_CL)
  19 | enum { FREE, RECV, SEND, APP, NOWN };
  20 | enum { SUB, CMP, REL, ACK, NRING };
  21 | enum { MESH_UNKNOWN, MESH_PAIRING, MESH_PAIRED, MESH_RETIRING, MESH_STOPPING, MESH_STOPPED };
  22 | struct wire { uint16_t src, dst, hops; };
  23 | struct desc { uint32_t page, bytes; uint16_t node; };
  24 | struct ring { _Alignas(MESH_CL) _Atomic uint64_t head, tail; };
  25 | static inline int ring_select(const struct ring *q, uint64_t *cursor, uint64_t *index){
  26 |   uint64_t tail=atomic_load_explicit(&q->tail,memory_order_relaxed);
  27 |   uint64_t head=atomic_load_explicit(&q->head,memory_order_acquire);
  28 |   if(head==tail) return 0;
  29 |   if(*cursor-tail>=head-tail) *cursor=tail;
  30 |   *index=(*cursor)++; return 1;
  31 | }
  32 | static inline void ring_erase(struct ring *q, void *entries, size_t stride, size_t capacity, uint64_t index){
  33 |   uint64_t tail=atomic_load_explicit(&q->tail,memory_order_relaxed);
  34 |   if(index!=tail) memcpy((char*)entries+(index%capacity)*stride,(char*)entries+(tail%capacity)*stride,stride);
  35 |   atomic_store_explicit(&q->tail,tail+1,memory_order_release);
  36 | }
  37 | struct hdr {
  38 |   uint32_t magic, version, pgsz, pool, arena, node;
  39 |   uint64_t data_off;
  40 |   struct ring r[NRING];
  41 |   _Alignas(MESH_CL) _Atomic uint64_t client, sent, recvd, bad, up_ms;
  42 |   _Atomic uint64_t bridge_pid, heartbeat_ms, phase, operation, operation_ms;
  43 |   _Atomic uint64_t port_count;
  44 |   _Alignas(MESH_CL) _Atomic uint64_t mean[NOWN], sd[NOWN];
  45 | };
  46 | struct mesh_port_info {
  47 |   char device[32];
  48 |   uint16_t peer;
  49 |   _Atomic uint64_t phase, heartbeat_us, operation, generation, reset_request;
  50 | };
  51 | _Static_assert(offsetof(struct hdr,mean)==offsetof(struct hdr,up_ms)+MESH_CL,"diagnostics must fit the existing header padding");
  52 | static inline unsigned char *mesh_at(struct hdr *m, uint32_t i){
  53 |   return (unsigned char*)m + m->data_off + (size_t)i * m->pgsz; }
  54 | static inline unsigned char *mesh_data(struct hdr *m, uint32_t i){
  55 |   return mesh_at(m,i) + sizeof(struct wire); }
  56 | static inline uint32_t mesh_pay(struct hdr *m){
  57 |   return m->pgsz - (uint32_t)sizeof(struct wire); }
  58 | static inline struct desc *slot(struct hdr *m, int k, uint64_t i){
  59 |   return &((struct desc*)((unsigned char*)m + RINGS))[k*MESH_RING + i%MESH_RING]; }
  60 | static inline struct mesh_port_info *mesh_ports(struct hdr *m){
  61 |   return (struct mesh_port_info*)((unsigned char*)m+RINGS+NRING*MESH_RING*sizeof(struct desc)); }
  62 | static inline int push(struct hdr *m, int k, const struct desc *d){
  63 |   struct ring *q=&m->r[k];
  64 |   uint64_t h=atomic_load_explicit(&q->head,memory_order_relaxed);
  65 |   if(h - atomic_load_explicit(&q->tail,memory_order_acquire) >= MESH_RING) return -1;
  66 |   *slot(m,k,h)=*d; atomic_store_explicit(&q->head,h+1,memory_order_release); return 0; }
  67 | static inline int pop(struct hdr *m, int k, struct desc *d){
  68 |   struct ring *q=&m->r[k];
  69 |   uint64_t t=atomic_load_explicit(&q->tail,memory_order_relaxed);
  70 |   if(t == atomic_load_explicit(&q->head,memory_order_acquire)) return -1;
  71 |   *d=*slot(m,k,t); atomic_store_explicit(&q->tail,t+1,memory_order_release); return 0; }
```

**X, lines 72–97.** Application frame kinds, mstream state, publication/pending bits and runnable/changed scheduling duplicate the specified dataflow. Retain only page address/identity information required by the canonical binding.

```text
  72 | struct shdr { uint64_t off; uint32_t sid, k; };
  73 | enum { K_DATA, K_FIN, K_REQ, K_OK, K_CLOSE, K_CLOSED, K_OPEN, K_READY,
  74 |        K_ABORT_RX, K_ABORT_TX, K_ABORTED_RX, K_ABORTED_TX };
  75 | struct mesh_epoch { uint64_t high, low; };
  76 | struct mesh_scope { struct mesh_epoch epoch; unsigned char plan[32]; };
  77 | enum { MS_RUN, MS_DONE, MS_FAIL };
  78 | struct mesh_page_bits { uint64_t published, pending; };
  79 | static inline size_t mesh_page_words(size_t pages){ return pages/64+(pages%64!=0); }
  80 | struct mstream { char *buf; const char *src; size_t n, off, done, nb;
  81 |                  union { size_t hole; size_t retry_ns; };
  82 |                  union { unsigned char *seen; struct mesh_page_bits *work; }; uint32_t sid; int node, rx, st; uint64_t fin_after_ns;
  83 |                  size_t stride, chunk; uint32_t *pages; _Atomic size_t available; int closing, closed;
  84 |                  struct mesh_scope scope; size_t logical_offset; int agreed, error, abort_ack, retain_seen;
  85 |                  size_t scan_word; int paged, resident, parked; uint64_t open_retry_ns; uint32_t *arrivals;
  86 |                  uint64_t *changed, *runnable, changed_bit; };
  87 | static inline void mesh_stream_schedule(const struct mstream *s){
  88 |   if(s->runnable) __atomic_fetch_or(s->runnable,s->changed_bit,__ATOMIC_RELEASE);
  89 | }
  90 | static inline void mesh_stream_changed(const struct mstream *s){
  91 |   if(s->changed) __atomic_fetch_or(s->changed,s->changed_bit,__ATOMIC_RELEASE);
  92 |   mesh_stream_schedule(s);
  93 | }
  94 | static inline int mesh_stream_published(const struct mstream *s,size_t page){
  95 |   return (__atomic_load_n(&s->work[page/64].published,__ATOMIC_ACQUIRE)>>(page%64))&1;
  96 | }
  97 | typedef int (*mesh_receive_fn)(void *, const void *, size_t, int);
```

**M, lines 98–147.** B: attach/detach/reset and physical-page access. X: copying pending store, stream registry/state machine and copy/scatter/gather APIs that materialize foreign buffers. An API name containing scatter or gather does not establish the required direct-page algorithm.

```text
  98 | struct mesh_ctx { struct hdr *M; unsigned char *arena; size_t len;
  99 |                   unsigned char *busy; size_t cursor;
 100 |                   unsigned char *pending; int *pending_nodes; uint32_t *pending_bytes;
 101 |                   size_t pending_head, pending_count, pending_capacity;
 102 |                   size_t inflight;
 103 |                   uint64_t sub, ack, ino, idle; int last; char *name; int mapping_pinned;
 104 |                   uint64_t integrity_failures, stale_frames; int name_owned, detaching;
 105 |                   _Atomic int executor_attached;
 106 |                   uint32_t *stream_index; size_t stream_index_capacity;
 107 |                   struct mstream **streams; size_t stream_count, stream_cursor;
 108 |                   mesh_receive_fn receiver; void *receiver_capture; };
 109 | struct mesh_ctx *mesh_context(void);
 110 | void mesh_receiver(struct mesh_ctx *context, mesh_receive_fn receiver, void *capture);
 111 | int    mesh_try_attach(struct mesh_ctx *c, const char *name);
 112 | int    mesh_attach(struct mesh_ctx *c, const char *name);
 113 | int    mesh_detach(struct mesh_ctx *c);
 114 | int    mesh_link_reset(struct mesh_ctx *c, size_t port);
 115 | int    mesh_turn(struct mesh_ctx *c, struct mstream **v, int k);
 116 | int    mesh_turn_window(struct mesh_ctx *c, struct mstream **v, int k, size_t window);
 117 | int    mesh_poll_streams(struct mesh_ctx *c, struct mstream **v, int k, uint64_t *now_ns);
 118 | int    mesh_progress_stream(struct mesh_ctx *c, struct mstream *s, size_t window, uint64_t now_ns);
 119 | void  *mesh_yell_view(struct mesh_ctx *c, struct mstream *s, size_t n, int node, uint32_t sid);
 120 | void  *mesh_stream_lease(struct mesh_ctx *c, struct mstream *s);
 121 | int    mesh_stream_idle(struct mesh_ctx *c, const struct mstream *s);
 122 | int    mesh_stream_push_page(struct mesh_ctx *c, struct mstream *s, size_t page, size_t window);
 123 | ptrdiff_t mesh_stream_push_pages(struct mesh_ctx *c, struct mstream *s, size_t window);
 124 | int    mesh_stream_receive(struct mesh_ctx *c, struct mstream *s, uint32_t *pages);
 125 | int    mesh_stream_reserve(struct mesh_ctx *c, size_t count);
 126 | int    mesh_stream_register(struct mesh_ctx *c, struct mstream **streams, size_t count);
 127 | int    mesh_stream_conflict(const struct mesh_ctx *c, const struct mstream *stream, struct mesh_epoch epoch);
 128 | int    mesh_lissen_view(struct mesh_ctx *c, struct mstream *s, uint32_t *pages, size_t n, uint32_t sid);
 129 | size_t mesh_release_view(struct mesh_ctx *c, struct mstream *s);
 130 | void   mesh_yell_start(struct mesh_ctx *c, struct mstream *s, const void *p, size_t n, int node, uint32_t sid);
 131 | int    mesh_lissen_start(struct mesh_ctx *c, struct mstream *s, void *p, size_t n, uint32_t sid);
 132 | int    mesh_scatter(struct mesh_ctx *c, struct mstream *ss, const void *p, size_t n, const int *nodes, int k, uint32_t sid0);
 133 | int    mesh_gather(struct mesh_ctx *c, struct mstream *ss, void *p, size_t n, int k, uint32_t sid0);
 134 | void  *mesh_open(size_t *nslots, size_t *stride, size_t *usable);
 135 | void  *mesh_try_open(size_t *nslots, size_t *stride, size_t *usable);
 136 | int    mesh_close(void);
 137 | size_t mesh_write(const void *p, size_t nbytes, int node);
 138 | size_t mesh_write_copy(const void *p, size_t stride, size_t bytes, size_t nslots, int node);
 139 | size_t mesh_queue_copy(const void *p, size_t stride, size_t bytes, size_t nslots, int node);
 140 | size_t mesh_pump(void);
 141 | size_t mesh_queued(void);
 142 | size_t mesh_inflight(void);
 143 | size_t mesh_read(void **p, int *from);
 144 | size_t mesh_readv(void *p, size_t stride, uint32_t *sizes, int *from, size_t count);
 145 | size_t mesh_yell(const void *p, size_t n, int node);
 146 | size_t mesh_lissen(void *p, size_t n);
 147 | #endif
```

### rdma/mesh-client.c — 627 lines

**B, lines 1–80.** Shared region attachment and explicit link reset. No per-NFE rendezvous is required here. The busy array is another physical ownership representation: canonical ownership must not diverge from literal rows. Startup retry is distinct from a reduction handshake.

```text
   1 | 
   2 | #include "mesh.h"
   3 | #include "mesh-wire.h"
   4 | #include <errno.h>
   5 | #include <sys/mman.h>
   6 | #include <sys/stat.h>
   7 | #include <fcntl.h>
   8 | #include <unistd.h>
   9 | #include <string.h>
  10 | #include <stdlib.h>
  11 | #include <stdio.h>
  12 | #include <time.h>
  13 | #include <signal.h>
  14 | #ifdef __APPLE__
  15 | #include <mach/mach.h>
  16 | #include <mach/mach_vm.h>
  17 | #endif
  18 | 
  19 | #ifndef MESH_ATTACH_ATTEMPTS
  20 | #define MESH_ATTACH_ATTEMPTS 3000
  21 | #endif
  22 | 
  23 | static struct mesh_ctx CTX0={.last=-1};
  24 | struct mesh_ctx *mesh_context(void){ return &CTX0; }
  25 | void mesh_receiver(struct mesh_ctx *context,mesh_receive_fn receiver,void *capture){
  26 |   context->receiver=receiver; context->receiver_capture=capture;
  27 | }
  28 | 
  29 | static const char *rname(const char *name){
  30 |   if(!name) name=getenv("MESH_REGION");
  31 |   return name?name:MESH_NAME; }
  32 | 
  33 | static uint64_t identity(void *p, struct stat *s){
  34 | #ifdef __APPLE__
  35 |   (void)s;
  36 |   mach_vm_address_t address=(mach_vm_address_t)p; mach_vm_size_t size=0;
  37 |   natural_t depth=0; vm_region_submap_info_data_64_t info={0};
  38 |   mach_msg_type_number_t count=VM_REGION_SUBMAP_INFO_COUNT_64;
  39 |   kern_return_t rc=mach_vm_region_recurse(mach_task_self(),&address,&size,&depth,
  40 |                                          (vm_region_recurse_info_t)&info,&count);
  41 |   if(rc) fprintf(stderr,"mesh region identity: %s\n",mach_error_string(rc));
  42 |   return rc?0:info.object_id_full;
  43 | #else
  44 |   (void)p; return (uint64_t)s->st_ino;
  45 | #endif
  46 | }
  47 | 
  48 | static int mesh_attach_attempts(struct mesh_ctx *c, const char *name, int attempts){
  49 |   if(c->M){ if(c->mapping_pinned<0 || c->detaching){ errno=ESTALE; return -1; } return 0; }
  50 |   name=name?name:(c->name?c->name:rname(0));
  51 |   for(int t=0;t<attempts && !c->M;t++){
  52 |     int f=shm_open(name,O_RDWR,MESH_MODE);
  53 |     if(f>=0){
  54 |       struct stat s;
  55 |       if(!fstat(f,&s) && (size_t)s.st_size>=sizeof(struct hdr)){
  56 |         struct hdr *b=mmap(NULL,(size_t)s.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,f,0);
  57 |         if(b!=MAP_FAILED){
  58 |           if(b->magic==MESH_MAGIC && b->version==MESH_VERSION && atomic_load(&b->phase)<MESH_STOPPING){
  59 |             uint64_t vacant=0;
  60 |             if(!atomic_compare_exchange_strong_explicit(&b->client,&vacant,(uint64_t)getpid(),memory_order_acq_rel,memory_order_acquire)){
  61 |               munmap(b,(size_t)s.st_size); errno=EBUSY;
  62 |             } else {
  63 |               c->M=b; c->len=(size_t)s.st_size; c->ino=identity(b,&s);
  64 |               if(!c->name){ c->name=strdup(name); c->name_owned=1; } } }
  65 |           else munmap(b,(size_t)s.st_size); } }
  66 |       close(f); }
  67 |     if(!c->M && t+1<attempts) usleep(10000); }
  68 |   if(!c->M) return -1;
  69 |   c->arena=mesh_data(c->M,c->M->pool); c->busy=calloc(c->M->arena,1);
  70 |   if(!c->busy){ atomic_store(&c->M->client,0); munmap(c->M,c->len); c->M=0; c->arena=0; c->len=0; return -1; }
  71 |   c->cursor=0; c->inflight=0; c->last=-1; c->sub=c->ack=0; c->idle=0;
  72 |   return 0; }
  73 | 
  74 | int mesh_attach(struct mesh_ctx *c, const char *name){ return mesh_attach_attempts(c,name,MESH_ATTACH_ATTEMPTS); }
  75 | int mesh_try_attach(struct mesh_ctx *c, const char *name){ return mesh_attach_attempts(c,name,1); }
  76 | int mesh_link_reset(struct mesh_ctx *c,size_t port){
  77 |   if(!c->M || port>=atomic_load_explicit(&c->M->port_count,memory_order_acquire)) return EINVAL;
  78 |   atomic_store_explicit(&mesh_ports(c->M)[port].reset_request,1,memory_order_release); return 0;
  79 | }
  80 | 
```

**M, lines 81–123.** B: stale connection detection and remapping/lifecycle. D: propagating stream states and changed masks belongs to the alternative algorithm, not the required negative link status/value recovery contract.

```text
  81 | static int stale(struct mesh_ctx *c){
  82 |   uint64_t pid=atomic_load(&c->M->bridge_pid);
  83 |   if(atomic_load(&c->M->phase)>=MESH_STOPPING || (pid && kill((pid_t)pid,0)<0 && errno==ESRCH)) return 1;
  84 |   if(c->mapping_pinned) return 0;
  85 |   struct stat s; int f=shm_open(c->name?c->name:rname(0),O_RDWR,MESH_MODE);
  86 |   if(f<0) return errno==ENOENT;
  87 |   int gone=0;
  88 |   if(!fstat(f,&s) && (size_t)s.st_size>=sizeof(struct hdr)){
  89 |     void *p=mmap(NULL,sizeof(struct hdr),PROT_READ,MAP_SHARED,f,0);
  90 |     if(p!=MAP_FAILED){
  91 |       uint64_t id=identity(p,&s);
  92 |       uint64_t current=c->ino?c->ino:identity(c->M,&s);
  93 |       gone=(id && current && id!=current) || atomic_load(&((struct hdr*)p)->bridge_pid)!=pid;
  94 |       if(!gone && current) c->ino=current;
  95 |       munmap(p,sizeof(struct hdr)); } }
  96 |   close(f); return gone; }
  97 | 
  98 | static void mesh_retire(struct mesh_ctx *c, struct mstream **v, int k){
  99 |   for(int i=0;i<k;i++){ struct mstream *s=v[i];
 100 |     if(s->st==MS_RUN) s->st=MS_FAIL;
 101 |     if(!s->retain_seen){ free(s->seen); s->seen=0; } }
 102 |   free(c->busy); c->busy=0; munmap(c->M,c->len); c->M=0; c->arena=0; }
 103 | 
 104 | static void reattach(struct mesh_ctx *c, struct mstream **v, int k){
 105 |   mesh_retire(c,v,k);
 106 |   if(!mesh_try_attach(c,0)) fprintf(stderr,"mesh reattached %s object %llu\n",c->name,(unsigned long long)c->ino); }
 107 | 
 108 | static int refresh(struct mesh_ctx *c, struct mstream **v, int k){
 109 |   if(c->mapping_pinned && atomic_load(&c->M->bridge_pid) && atomic_load(&c->M->phase)>=MESH_STOPPING) c->mapping_pinned=-1;
 110 |   if(c->mapping_pinned<0){
 111 |     for(int i=0;i<k;i++){ v[i]->st=MS_FAIL; mesh_stream_changed(v[i]); }
 112 |     errno=ESTALE;
 113 |     return 1; }
 114 |   if(++c->idle<600) return 0;
 115 |   c->idle=0;
 116 |   if(!stale(c)) return 0;
 117 |   if(c->mapping_pinned){
 118 |     c->mapping_pinned=-1;
 119 |     for(int i=0;i<k;i++){ v[i]->st=MS_FAIL; mesh_stream_changed(v[i]); }
 120 |     errno=ESTALE;
 121 |     return 1; }
 122 |   reattach(c,v,k); return 1; }
 123 | 
```

**B, lines 124–161.** Submit/receive actual pages and preserve NIC ownership through the binding. Receive auto-release on next read is not valid for asynchronous numerical readers; use canonical table lifetimes instead. The active page runtime consumes descriptor rings directly.

```text
 124 | static size_t cwrite(struct mesh_ctx *c, const void *p, size_t nbytes, int node){
 125 |   struct hdr *M=c->M;
 126 |   uint32_t s=(uint32_t)(((const unsigned char*)p-c->arena)/M->pgsz); size_t done=0;
 127 |   while(done<nbytes && s<M->arena){
 128 |     uint32_t u=mesh_pay(M);
 129 |     if(c->busy && (c->busy[s]&1)) break;
 130 |     struct desc d={.page=M->pool+s,.bytes=nbytes-done<u?(uint32_t)(nbytes-done):u,.node=(uint16_t)node};
 131 |     if(push(M,SUB,&d)) break;
 132 |     if(c->busy){ c->busy[s]|=1; c->inflight++; }
 133 |     done+=d.bytes; s++; }
 134 |   return done; }
 135 | 
 136 | static size_t cread(struct mesh_ctx *c, void **p, int *from){
 137 |   struct hdr *M=c->M;
 138 |   if(c->last>=0){ struct desc r={.page=(uint32_t)c->last};
 139 |     if(push(M,REL,&r)) return 0;
 140 |     c->last=-1; }
 141 |   struct desc d; if(pop(M,CMP,&d)) return 0;
 142 |   c->last=(int)d.page;
 143 |   if(p) *p=mesh_data(M,d.page);
 144 |   if(from) *from=d.node;
 145 |   return d.bytes; }
 146 | 
 147 | static void reclaim(struct mesh_ctx *c){
 148 |   struct desc d;
 149 |   while(!pop(c->M,ACK,&d)){
 150 |     if(c->busy && d.page>=c->M->pool && d.page<c->M->pool+c->M->arena && (c->busy[d.page-c->M->pool]&1)){
 151 |       c->busy[d.page-c->M->pool]&=2; c->inflight--; }
 152 |     c->ack++; } }
 153 | 
 154 | static unsigned char *credit(struct mesh_ctx *c){
 155 |   if(c->inflight>=c->M->arena) return 0;
 156 |   for(size_t i=0;i<c->M->arena;i++){
 157 |     size_t s=(c->cursor+i)%c->M->arena;
 158 |     if(!c->busy || !c->busy[s]){ c->cursor=(s+1)%c->M->arena;
 159 |       return c->arena+s*c->M->pgsz; } }
 160 |   return 0; }
 161 | 
```

**X, lines 162–206.** pending_grow/pending_push/pending_flush allocate a foreign payload store and copy into RDMA pages. Delete as numerical transport representation.

```text
 162 | static int pending_grow(struct mesh_ctx *c, size_t want){
 163 |   size_t cap=c->pending_capacity?c->pending_capacity:1;
 164 |   while(cap<want){
 165 |     if(cap>SIZE_MAX/2){ cap=want; break; }
 166 |     cap*=2; }
 167 |   size_t u=mesh_pay(c->M);
 168 |   if(!u || cap>SIZE_MAX/u || cap>SIZE_MAX/sizeof *c->pending_nodes || cap>SIZE_MAX/sizeof *c->pending_bytes) return -1;
 169 |   unsigned char *p=realloc(c->pending,cap*u);
 170 |   if(!p) return -1;
 171 |   c->pending=p;
 172 |   int *nodes=realloc(c->pending_nodes,cap*sizeof *nodes);
 173 |   if(!nodes) return -1;
 174 |   c->pending_nodes=nodes;
 175 |   uint32_t *bytes=realloc(c->pending_bytes,cap*sizeof *bytes);
 176 |   if(!bytes) return -1;
 177 |   c->pending_bytes=bytes; c->pending_capacity=cap; return 0; }
 178 | 
 179 | static size_t pending_flush(struct mesh_ctx *c){
 180 |   reclaim(c); size_t before=c->pending_count;
 181 |   while(c->pending_count){
 182 |     unsigned char *q=credit(c); if(!q) break;
 183 |     size_t at=c->pending_head, u=mesh_pay(c->M); uint32_t bytes=c->pending_bytes[at];
 184 |     memcpy(q,c->pending+at*u,bytes);
 185 |     if(cwrite(c,q,bytes,c->pending_nodes[at])!=bytes) break;
 186 |     c->sub++; c->pending_head++; c->pending_count--; }
 187 |   if(!c->pending_count) c->pending_head=0;
 188 |   return before-c->pending_count; }
 189 | 
 190 | static size_t pending_push(struct mesh_ctx *c, const void *p, size_t stride,
 191 |                            size_t bytes, size_t nslots, int node){
 192 |   size_t u=mesh_pay(c->M);
 193 |   if(nslots>SIZE_MAX-c->pending_head-c->pending_count) return 0;
 194 |   if(c->pending_head && c->pending_head+c->pending_count+nslots>c->pending_capacity){
 195 |     memmove(c->pending,c->pending+c->pending_head*u,c->pending_count*u);
 196 |     memmove(c->pending_nodes,c->pending_nodes+c->pending_head,c->pending_count*sizeof *c->pending_nodes);
 197 |     memmove(c->pending_bytes,c->pending_bytes+c->pending_head,c->pending_count*sizeof *c->pending_bytes);
 198 |     c->pending_head=0; }
 199 |   size_t want=c->pending_head+c->pending_count+nslots;
 200 |   if(want>c->pending_capacity && pending_grow(c,want)) return 0;
 201 |   for(size_t i=0;i<nslots;i++){
 202 |     size_t at=c->pending_head+c->pending_count+i;
 203 |     memset(c->pending+at*u,0,u); memcpy(c->pending+at*u,(const char*)p+i*stride,bytes);
 204 |     c->pending_nodes[at]=node; c->pending_bytes[at]=(uint32_t)bytes; }
 205 |   c->pending_count+=nslots; pending_flush(c); return nslots; }
 206 | 
```

**X, lines 207–252.** Application control frames, exponential retry, stream agreement predicate, published/pending bitmap and transmission cursor are the forbidden protocol above page delivery.

```text
 207 | static int stream_ctl(struct mesh_ctx *c, const struct mstream *s, uint32_t kind, uint64_t offset, const void *body, size_t bytes){
 208 |   unsigned char *q=credit(c); if(!q) return -1;
 209 |   size_t header=mesh_stream_header(s);
 210 |   if(bytes) memcpy(q+header,body,bytes);
 211 |   struct mesh_frame frame={.h={offset,s->sid,kind},.epoch=s->scope.epoch,.source=(uint16_t)c->M->node,.target=(uint16_t)s->node};
 212 |   size_t n=mesh_frame_encode(q,&frame,bytes);
 213 |   if(cwrite(c,q,n,s->node)!=n) return -1;
 214 |   c->sub++; return 0;
 215 | }
 216 | static void stream_retry(struct mstream *s,uint64_t now_ns,uint64_t floor_ns){
 217 |   s->retry_ns=s->retry_ns?s->retry_ns<500000000?s->retry_ns*2:1000000000:floor_ns?floor_ns:1000000;
 218 |   s->fin_after_ns=now_ns+s->retry_ns;
 219 | }
 220 | static size_t stream_page(struct mesh_ctx *c,struct mstream *s,size_t offset){
 221 |   size_t header=mesh_stream_header(s), u=mesh_stream_payload(c->M,s);
 222 |   uint32_t len=s->n-offset<u?(uint32_t)(s->n-offset):(uint32_t)u;
 223 |   unsigned char *q=s->stride?(unsigned char*)s->src+(offset/u)*s->stride-header:credit(c);
 224 |   if(!q || (s->stride && (c->busy[(q-c->arena)/c->M->pgsz]&1))) return 0;
 225 |   if(!s->stride) memcpy(q+header,s->src+offset,len);
 226 |   struct mesh_frame frame={.h={offset,s->sid,K_DATA},.epoch=s->scope.epoch,.source=(uint16_t)c->M->node,.target=(uint16_t)s->node};
 227 |   size_t bytes=mesh_frame_encode(q,&frame,len);
 228 |   if(cwrite(c,q,bytes,s->node)!=bytes) return 0;
 229 |   c->sub++; return len;
 230 | }
 231 | int mesh_stream_push_page(struct mesh_ctx *c,struct mstream *s,size_t page,size_t window){
 232 |   struct mesh_page_bits *word=&s->work[page/64]; uint64_t bit=UINT64_C(1)<<(page%64);
 233 |   if(s->st!=MS_RUN || (mesh_epoch_set(s->scope.epoch) && !s->agreed) || c->inflight>=window || !(__atomic_load_n(&word->pending,__ATOMIC_ACQUIRE)&bit)) return 0;
 234 |   size_t bytes=stream_page(c,s,page*mesh_stream_payload(c->M,s));
 235 |   if(!bytes) return 0;
 236 |   __atomic_fetch_and(&word->pending,~bit,__ATOMIC_RELEASE);
 237 |   s->off+=bytes; s->fin_after_ns=0; s->retry_ns=0; return 1;
 238 | }
 239 | ptrdiff_t mesh_stream_push_pages(struct mesh_ctx *c,struct mstream *s,size_t window){
 240 |   size_t before=s->off;
 241 |   size_t words=mesh_page_words(s->nb);
 242 |   for(size_t examined=0;s->off<atomic_load_explicit(&s->available,memory_order_acquire) && c->inflight<window && examined<words;examined++){
 243 |     size_t word=s->scan_word;
 244 |     uint64_t pending=__atomic_load_n(&s->work[word].pending,__ATOMIC_ACQUIRE);
 245 |     for(;pending && c->inflight<window;pending&=pending-1)
 246 |       mesh_stream_push_page(c,s,word*64+(size_t)__builtin_ctzll(pending),window);
 247 |     s->scan_word=word+1==words?0:word+1;
 248 |   }
 249 |   size_t bytes=s->off-before, payload=mesh_stream_payload(c->M,s);
 250 |   return (ptrdiff_t)(bytes/payload+(bytes%payload!=0));
 251 | }
 252 | 
```

**M, lines 253–326.** A/B: actual page addresses and local bridge release. X: stream lease/receive wrappers with non-table ownership, seen masks and dense-copy fallback. Release must follow canonical read proofs and zeroing, not stream settlement.

```text
 253 | void mesh_yell_start(struct mesh_ctx *c, struct mstream *s,
 254 |                      const void *p, size_t n, int node, uint32_t sid){
 255 |   *s=(struct mstream){.src=p,.n=n,.node=node,.sid=sid,.st=n?MS_RUN:MS_DONE,.available=n}; (void)c; }
 256 | 
 257 | int mesh_stream_receive(struct mesh_ctx *c, struct mstream *s, uint32_t *pages){
 258 |   if(mesh_attach(c,0)) return -1;
 259 |   size_t u=mesh_stream_payload(c->M,s);
 260 |   s->nb=s->n/u+(s->n%u!=0); s->pages=pages;
 261 |   s->seen=pages?NULL:calloc((s->nb+7)/8,1);
 262 |   if(pages) for(size_t i=0;i<s->nb;i++) pages[i]=UINT32_MAX;
 263 |   return pages||s->seen?0:-1;
 264 | }
 265 | static int stream_missing(const struct mstream *s,size_t page){
 266 |   return s->pages?s->pages[page]==UINT32_MAX:s->seen && !(s->seen[page>>3]>>(page&7)&1);
 267 | }
 268 | int mesh_lissen_start(struct mesh_ctx *c, struct mstream *s, void *p, size_t n, uint32_t sid){
 269 |   *s=(struct mstream){.buf=p,.n=n,.sid=sid,.rx=1,.st=n?MS_RUN:MS_DONE,.node=-1};
 270 |   return mesh_stream_receive(c,s,NULL);
 271 | }
 272 | void *mesh_stream_lease(struct mesh_ctx *c, struct mstream *s){
 273 |   if(mesh_attach(c,0)) return NULL;
 274 |   reclaim(c);
 275 |   size_t header=mesh_stream_header(s), u=mesh_stream_payload(c->M,s), pages=s->n/u+(s->n%u!=0), run=0;
 276 |   for(size_t i=0;pages && i<c->M->arena;i++){
 277 |     run=c->busy[i]?0:run+1;
 278 |     if(run<pages) continue;
 279 |     size_t first=i+1-pages;
 280 |     memset(c->busy+first,2,pages);
 281 |     void *base=c->arena+first*c->M->pgsz-sizeof(struct wire);
 282 |     s->src=(char*)base+sizeof(struct wire)+header;
 283 |     s->stride=c->M->pgsz; s->nb=pages; atomic_store(&s->available,0);
 284 |     return base;
 285 |   }
 286 |   return NULL;
 287 | }
 288 | void *mesh_yell_view(struct mesh_ctx *c, struct mstream *s, size_t n, int node, uint32_t sid){
 289 |   mesh_yell_start(c,s,NULL,n,node,sid); return mesh_stream_lease(c,s);
 290 | }
 291 | 
 292 | int mesh_stream_idle(struct mesh_ctx *c, const struct mstream *s){
 293 |   reclaim(c);
 294 |   if(!c->inflight || !s->stride) return 1;
 295 |   size_t first=((const unsigned char*)s->src-mesh_stream_header(s)-c->arena)/c->M->pgsz;
 296 |   for(size_t i=0;i<s->nb;i++) if(c->busy[first+i]&1) return 0;
 297 |   return 1;
 298 | }
 299 | int mesh_lissen_view(struct mesh_ctx *c, struct mstream *s, uint32_t *pages, size_t n, uint32_t sid){
 300 |   *s=(struct mstream){.n=n,.sid=sid,.rx=1,.st=n?MS_RUN:MS_DONE,.node=-1};
 301 |   return mesh_stream_receive(c,s,pages);
 302 | }
 303 | 
 304 | size_t mesh_release_view(struct mesh_ctx *c, struct mstream *s){
 305 |   size_t pending=0;
 306 |   if(s->stride){
 307 |     size_t first=((const unsigned char*)s->src-mesh_stream_header(s)-c->arena)/c->M->pgsz;
 308 |     for(size_t i=0;i<s->nb;i++) c->busy[first+i]&=1;
 309 |     s->stride=0; }
 310 |   if(s->pages){
 311 |     struct ring *ring=&c->M->r[REL];
 312 |     uint64_t head=atomic_load_explicit(&ring->head,memory_order_relaxed);
 313 |     uint64_t tail=atomic_load_explicit(&ring->tail,memory_order_acquire);
 314 |     size_t count=s->arrivals?s->done/mesh_stream_payload(c->M,s)+(s->done%mesh_stream_payload(c->M,s)!=0):s->nb;
 315 |     for(size_t i=0;i<count;i++){
 316 |       size_t page=s->arrivals?s->arrivals[i]:i;
 317 |       if(s->pages[page]==UINT32_MAX) continue;
 318 |       if(c->mapping_pinned>=0 && head-tail>=MESH_RING){ pending++; continue; }
 319 |       if(c->mapping_pinned>=0) *slot(c->M,REL,head++)=(struct desc){.page=s->pages[page]};
 320 |       s->pages[page]=UINT32_MAX;
 321 |     }
 322 |     atomic_store_explicit(&ring->head,head,memory_order_release);
 323 |   }
 324 |   if(!pending) s->pages=NULL;
 325 |   return pending; }
 326 | 
```

**X, lines 327–369.** Separate stream registry and conflict/epoch machinery support the alternate execution protocol. Configured row routing may be immutable configuration without a mutable stream scheduler.

```text
 327 | static size_t stream_bucket(uint32_t sid,int receive,size_t mask){
 328 |   uint64_t key=((uint64_t)sid<<1)^(unsigned)receive;
 329 |   key^=key>>33; key*=0xff51afd7ed558ccdULL; key^=key>>33;
 330 |   return (size_t)key&mask;
 331 | }
 332 | static void stream_index(struct mesh_ctx *c,struct mstream **v,size_t count){
 333 |   memset(c->stream_index,0,c->stream_index_capacity*sizeof *c->stream_index);
 334 |   for(size_t i=0;i<count;i++){
 335 |     size_t bucket=stream_bucket(v[i]->sid,v[i]->rx,c->stream_index_capacity-1);
 336 |     while(c->stream_index[bucket]) bucket=(bucket+1)&(c->stream_index_capacity-1);
 337 |     c->stream_index[bucket]=(uint32_t)i+1;
 338 |   }
 339 | }
 340 | 
 341 | int mesh_stream_reserve(struct mesh_ctx *c, size_t count){
 342 |   if(count>SIZE_MAX/4/sizeof(uint32_t)) return EOVERFLOW;
 343 |   size_t capacity=1;
 344 |   while(capacity<count*2) capacity*=2;
 345 |   if(capacity>c->stream_index_capacity){
 346 |     uint32_t *index=realloc(c->stream_index,capacity*sizeof *index);
 347 |     if(!index) return ENOMEM;
 348 |     c->stream_index=index; c->stream_index_capacity=capacity;
 349 |     stream_index(c,c->streams,c->stream_count);
 350 |   }
 351 |   return 0;
 352 | }
 353 | int mesh_stream_register(struct mesh_ctx *c,struct mstream **v,size_t count){
 354 |   int error=mesh_stream_reserve(c,count); if(error) return error;
 355 |   stream_index(c,v,count);
 356 |   c->streams=v; c->stream_count=count; c->stream_cursor=0;
 357 |   return 0;
 358 | }
 359 | int mesh_stream_conflict(const struct mesh_ctx *c,const struct mstream *stream,struct mesh_epoch epoch){
 360 |   size_t mask=c->stream_index_capacity-1;
 361 |   size_t bucket=stream_bucket(stream->sid,stream->rx,mask);
 362 |   for(size_t probe=0;probe<c->stream_index_capacity;probe++){
 363 |     uint32_t index=c->stream_index[(bucket+probe)&mask]; if(!index) break;
 364 |     const struct mstream *other=c->streams[index-1];
 365 |     if(other!=stream && other->sid==stream->sid && other->rx==stream->rx && other->node==stream->node &&
 366 |        mesh_epoch_equal(other->scope.epoch,epoch)) return 1;
 367 |   }
 368 |   return 0;
 369 | }
```

**X, lines 370–501.** OPEN/READY, FIN/REQ/OK, CLOSE/CLOSED, ABORT/ABORTED and retry timing are literal active implementations of the forbidden protocol. Receive may copy into dense buffers and drives done/agreed/state instead of row stamps. These functions are not invoked by mesh-pages_progress in the measured NFE.

```text
 370 | int mesh_poll_streams(struct mesh_ctx *c, struct mstream **v, int k, uint64_t *now_ns){
 371 |   if(mesh_try_attach(c,0)){
 372 |     if(errno==ESTALE) for(int i=0;i<k;i++){ v[i]->st=MS_FAIL; v[i]->error=ESTALE; mesh_stream_changed(v[i]); }
 373 |     return 0; }
 374 |   reclaim(c);
 375 |   if(refresh(c,v,k)) return 0;
 376 |   size_t capacity=c->stream_index_capacity;
 377 |   uint32_t *index=v==c->streams && (size_t)k==c->stream_count?c->stream_index:NULL;
 378 |   struct timespec clock; clock_gettime(CLOCK_MONOTONIC,&clock);
 379 |   *now_ns=(uint64_t)clock.tv_sec*1000000000u+clock.tv_nsec;
 380 |   for(int turn=0;turn<256;turn++){
 381 |     void *q; int from; size_t b=cread(c,&q,&from);
 382 |     if(!b) break;
 383 |     if(c->receiver && c->receiver(c->receiver_capture,q,b,from)) continue;
 384 |     if(b<MESH_OFF){ c->integrity_failures++; continue; }
 385 |     struct mesh_frame frame; size_t header=mesh_frame_decode(q,b,&frame);
 386 |     if(!header){ c->integrity_failures++; continue; }
 387 |     if(header>MESH_OFF && (frame.source!=from || frame.target!=c->M->node)){ c->integrity_failures++; continue; }
 388 |     struct shdr sh=frame.h; struct mstream *s=NULL;
 389 |     int receive=mesh_frame_receive(sh.k);
 390 |     if(header>MESH_OFF && sh.k!=K_DATA) for(int i=0;i<k;i++)
 391 |       if(!v[i]->rx && !v[i]->parked && !v[i]->agreed && v[i]->st==MS_RUN && v[i]->node==from){ v[i]->fin_after_ns=0; v[i]->retry_ns=0; }
 392 |     size_t bucket=stream_bucket(sh.sid,receive,capacity-1);
 393 |     for(size_t probe=0;probe<(index?capacity:(size_t)k);probe++){
 394 |       int i=index?(int)index[(bucket+probe)&(capacity-1)]-1:(int)probe;
 395 |       if(i<0) break;
 396 |       if(!v[i]->parked && v[i]->sid==sh.sid && v[i]->rx==receive &&
 397 |          (v[i]->node<0 || v[i]->node==from) && mesh_epoch_equal(v[i]->scope.epoch,frame.epoch)) s=v[i];
 398 |     }
 399 |     if(!s){
 400 |       int kind=sh.k==K_FIN&&header>MESH_OFF?K_OK:mesh_frame_reply(sh.k);
 401 |       if(kind==K_CLOSED || kind==K_OK){
 402 |         struct mstream reply={.sid=sh.sid,.node=from,.scope.epoch=frame.epoch};
 403 |         stream_ctl(c,&reply,(uint32_t)kind,0,NULL,0);
 404 |       } else if(header>MESH_OFF) c->stale_frames++;
 405 |       continue;
 406 |     }
 407 |     if(sh.k==K_ABORT_RX || sh.k==K_ABORT_TX){
 408 |       if(!s->error) s->error=sh.off>0&&sh.off<=INT32_MAX?(int)sh.off:ECANCELED;
 409 |       s->st=MS_FAIL; s->abort_ack=1;
 410 |       stream_ctl(c,s,(uint32_t)mesh_frame_reply(sh.k),0,NULL,0); mesh_stream_changed(s); continue;
 411 |     }
 412 |     if(sh.k==K_ABORTED_RX || sh.k==K_ABORTED_TX){ if(s->error){ s->abort_ack=1; mesh_stream_changed(s); } continue; }
 413 |     if(s->st==MS_FAIL) continue;
 414 |     size_t u=mesh_stream_payload(c->M,s);
 415 |     if(sh.k==K_OPEN && header>MESH_OFF){
 416 |       uint64_t logical_offset=0,chunk=0;
 417 |       size_t agreement=sizeof s->scope.plan+sizeof logical_offset;
 418 |       if(b==header+agreement || b==header+agreement+sizeof chunk){
 419 |         memcpy(&logical_offset,(char*)q+header+sizeof s->scope.plan,sizeof logical_offset);
 420 |         if(b==header+agreement+sizeof chunk) memcpy(&chunk,(char*)q+header+agreement,sizeof chunk);
 421 |       }
 422 |       uint64_t expected_chunk=s->chunk|((uint64_t)s->resident<<63);
 423 |       if(b!=header+agreement+(expected_chunk?sizeof chunk:0) || chunk!=expected_chunk || sh.off!=s->n || logical_offset!=s->logical_offset ||
 424 |          memcmp((char*)q+header,s->scope.plan,sizeof s->scope.plan)){
 425 |         s->error=EPROTO; s->st=MS_FAIL; s->fin_after_ns=0; s->retry_ns=0;
 426 |       } else { s->agreed=1; stream_ctl(c,s,K_READY,0,NULL,0); }
 427 |     }
 428 |     else if(sh.k==K_READY && header>MESH_OFF){ s->agreed=1; s->fin_after_ns=0; s->retry_ns=0; }
 429 |     else if(header>MESH_OFF && !s->agreed) continue;
 430 |     else if(sh.k==K_DATA){
 431 |       size_t len=b-header, ci=sh.off/u;
 432 |       if(s->st==MS_RUN && sh.off<s->n && sh.off%u==0 && len==(s->n-sh.off<u?s->n-sh.off:u) &&
 433 |          ci<s->nb && stream_missing(s,ci)){
 434 |         if(s->pages){ __atomic_store_n(&s->pages[ci],(uint32_t)c->last,__ATOMIC_RELEASE); c->last=-1; }
 435 |         else { s->seen[ci>>3]|=(unsigned char)(1u<<(ci&7)); memcpy(s->buf+sh.off,(char*)q+header,len); }
 436 |         if(s->arrivals) s->arrivals[s->done/u+(s->done%u!=0)]=(uint32_t)ci;
 437 |         __atomic_store_n(&s->done,s->done+len,__ATOMIC_RELEASE);
 438 |         if(s->resident && s->done==s->n){ s->st=MS_DONE; s->closed=1; }
 439 |       }
 440 |     }
 441 |     else if(sh.k==K_FIN && sh.off==s->n){
 442 |       s->node=from;
 443 |       if(s->done>=s->n){ stream_ctl(c,s,K_OK,0,NULL,0); s->st=MS_DONE; }
 444 |       else { while(s->hole<s->nb && !stream_missing(s,s->hole)) s->hole++;
 445 |              stream_ctl(c,s,K_REQ,s->hole*(uint64_t)u,NULL,0); }
 446 |     }
 447 |     else if(sh.k==K_REQ){ if(s->st==MS_RUN && sh.off<s->n && sh.off%u==0){
 448 |       if(s->paged){
 449 |         size_t page=sh.off/u; uint64_t bit=UINT64_C(1)<<(page%64);
 450 |         if(mesh_stream_published(s,page) && !(__atomic_fetch_or(&s->work[page/64].pending,bit,__ATOMIC_RELEASE)&bit))
 451 |           s->off-=s->n-sh.off<u?s->n-sh.off:u;
 452 |       } else s->off=sh.off;
 453 |       s->fin_after_ns=0; s->retry_ns=0;
 454 |     } }
 455 |     else if(sh.k==K_OK && s->st==MS_RUN && s->off==s->n){
 456 |       __atomic_store_n(&s->done,s->n,__ATOMIC_RELEASE); s->st=MS_DONE; s->fin_after_ns=0; s->retry_ns=0;
 457 |       if(s->resident) s->closed=1;
 458 |     }
 459 |     else if(sh.k==K_CLOSE && s->st==MS_DONE){ stream_ctl(c,s,K_CLOSED,0,NULL,0); s->closed=1; }
 460 |     else if(sh.k==K_CLOSED && s->st==MS_DONE) s->closed=1;
 461 |     if(sh.k!=K_DATA || s->done==s->n || s->st!=MS_RUN) mesh_stream_changed(s);
 462 |     if(s->st!=MS_RUN && s->seen && !s->retain_seen){ free(s->seen); s->seen=NULL; }
 463 |   }
 464 |   return 1;
 465 | }
 466 | 
 467 | int mesh_progress_stream(struct mesh_ctx *c, struct mstream *s, size_t window, uint64_t now_ns){
 468 |   if(s->parked) return 0;
 469 |   size_t header=mesh_stream_header(s), u=mesh_stream_payload(c->M,s);
 470 |   if(s->error && header>MESH_OFF && !s->abort_ack){
 471 |     if(now_ns>=s->fin_after_ns && !stream_ctl(c,s,s->rx?K_ABORT_TX:K_ABORT_RX,(uint64_t)s->error,NULL,0)) stream_retry(s,now_ns,0);
 472 |     return 1;
 473 |   }
 474 |   if(!s->rx && s->st==MS_DONE && s->closing && !s->closed){
 475 |     if(now_ns>=s->fin_after_ns && !stream_ctl(c,s,K_CLOSE,0,NULL,0)) stream_retry(s,now_ns,0);
 476 |     return 1;
 477 |   }
 478 |   if(s->rx || s->st!=MS_RUN) return 0;
 479 |   if(header>MESH_OFF && !s->agreed){
 480 |     if(now_ns>=s->fin_after_ns){
 481 |       unsigned char body[48]; uint64_t logical_offset=s->logical_offset,chunk=s->chunk|((uint64_t)s->resident<<63);
 482 |       memcpy(body,s->scope.plan,32); memcpy(body+32,&logical_offset,8); memcpy(body+40,&chunk,8);
 483 |       if(!stream_ctl(c,s,K_OPEN,s->n,body,chunk?48:40)) stream_retry(s,now_ns,s->open_retry_ns);
 484 |     }
 485 |     return 1;
 486 |   }
 487 |   size_t was=s->off;
 488 |   if(s->paged) mesh_stream_push_pages(c,s,window);
 489 |   else while(s->off<s->n && c->inflight<window){
 490 |     uint32_t len=s->n-s->off<u?(uint32_t)(s->n-s->off):(uint32_t)u;
 491 |     if(s->off+len>atomic_load_explicit(&s->available,memory_order_acquire)) break;
 492 |     if(!stream_page(c,s,s->off)) break;
 493 |     s->off+=len;
 494 |   }
 495 |   if(s->off>was){ s->fin_after_ns=0; s->retry_ns=0; }
 496 |   else if(s->off>=s->n && now_ns>=s->fin_after_ns){
 497 |     if(!stream_ctl(c,s,K_FIN,s->n,NULL,0)) stream_retry(s,now_ns,0);
 498 |   }
 499 |   return s->off==s->n || s->off<atomic_load_explicit(&s->available,memory_order_acquire);
 500 | }
 501 | 
```

**X, lines 502–518.** Round-robin stream progress and whole-stream done count are an alternate scheduler.

```text
 502 | int mesh_turn_window(struct mesh_ctx *c, struct mstream **v, int k, size_t window){
 503 |   uint64_t now_ns;
 504 |   if(!mesh_poll_streams(c,v,k,&now_ns)) return 0;
 505 |   int ndone=0;
 506 |   size_t start=k?c->stream_cursor%(size_t)k:0;
 507 |   for(int i=0;i<k;i++){
 508 |     struct mstream *s=v[(start+(size_t)i)%(size_t)k];
 509 |     ndone+=!s->parked && s->st==MS_DONE;
 510 |     mesh_progress_stream(c,s,window,now_ns);
 511 |   }
 512 |   c->stream_cursor=start+1;
 513 |   return ndone;
 514 | }
 515 | 
 516 | int mesh_turn(struct mesh_ctx *c, struct mstream **v, int k){
 517 |   return mesh_turn_window(c,v,k,SIZE_MAX); }
 518 | 
```

**B, lines 519–543.** Actual client detach and resource lifetime must remain correct. Remove stream/copy-store cleanup only with removed owners; do not erase bridge/verbs lifetime requirements.

```text
 519 | int mesh_detach(struct mesh_ctx *c){
 520 |   if(!c->M) return 0;
 521 |   if(atomic_load(&c->executor_attached)) return EBUSY;
 522 |   if(c->mapping_pinned<0 || stale(c)) goto detached;
 523 |   if(c->detaching){
 524 |     if(atomic_load_explicit(&c->M->client,memory_order_acquire)==MESH_CLIENT_DETACH) return EBUSY;
 525 |     goto detached;
 526 |   }
 527 |   reclaim(c);
 528 |   if(c->inflight || c->pending_count) return EBUSY;
 529 |   for(size_t i=0;i<c->M->arena;i++) if(c->busy[i]) return EBUSY;
 530 |   for(int i=0;i<MESH_RING;i++){
 531 |     void *payload; int from;
 532 |     if(!cread(c,&payload,&from)) break;
 533 |   }
 534 |   if(c->last>=0 || atomic_load(&c->M->r[CMP].head)!=atomic_load(&c->M->r[CMP].tail)) return EBUSY;
 535 |   uint64_t owner=(uint64_t)getpid();
 536 |   if(!atomic_compare_exchange_strong_explicit(&c->M->client,&owner,MESH_CLIENT_DETACH,memory_order_acq_rel,memory_order_acquire)) return EOWNERDEAD;
 537 |   c->detaching=1; return EBUSY;
 538 | detached:
 539 |   munmap(c->M,c->len); free(c->busy); free(c->pending); free(c->pending_nodes); free(c->pending_bytes); free(c->stream_index);
 540 |   if(c->name_owned) free(c->name);
 541 |   *c=(struct mesh_ctx){.last=-1}; return 0;
 542 | }
 543 | 
```

**X, lines 544–557.** mesh_scatter/mesh_gather split a dense buffer among stream objects; this is not indexed gather/scatter directly over the canonical pages.

```text
 544 | int mesh_scatter(struct mesh_ctx *c, struct mstream *ss, const void *p, size_t n,
 545 |                  const int *nodes, int k, uint32_t sid0){
 546 |   size_t sh=(n+k-1)/k;
 547 |   for(int i=0;i<k;i++){ size_t o=(size_t)i*sh, l=o<n?(n-o<sh?n-o:sh):0;
 548 |     mesh_yell_start(c,&ss[i],(const char*)p+o,l,nodes[i],sid0+i); }
 549 |   return k; }
 550 | 
 551 | int mesh_gather(struct mesh_ctx *c, struct mstream *ss, void *p, size_t n,
 552 |                 int k, uint32_t sid0){
 553 |   size_t sh=(n+k-1)/k;
 554 |   for(int i=0;i<k;i++){ size_t o=(size_t)i*sh, l=o<n?(n-o<sh?n-o:sh):0;
 555 |     if(mesh_lissen_start(c,&ss[i],(char*)p+o,l,sid0+i)) return -1; }
 556 |   return k; }
 557 | 
```

**B, lines 558–574.** Attach and direct page-write access can be retained in the substrate, with canonical ownership and no separate numerical completion channel.

```text
 558 | void *mesh_open(size_t *ns, size_t *sp, size_t *up){
 559 |   if(mesh_attach(&CTX0,0)) return NULL;
 560 |   return mesh_try_open(ns,sp,up); }
 561 | 
 562 | void *mesh_try_open(size_t *ns, size_t *sp, size_t *up){
 563 |   if(mesh_try_attach(&CTX0,0)) return NULL;
 564 |   if(ns) *ns=CTX0.M->arena;
 565 |   if(sp) *sp=CTX0.M->pgsz; if(up) *up=mesh_pay(CTX0.M);
 566 |   return CTX0.arena; }
 567 | 
 568 | int mesh_close(void){ return mesh_detach(&CTX0); }
 569 | 
 570 | size_t mesh_write(const void *p, size_t nbytes, int node){
 571 |   if(mesh_try_attach(&CTX0,0)) return 0;
 572 |   reclaim(&CTX0);
 573 |   return cwrite(&CTX0,p,nbytes,node); }
 574 | 
```

**X, lines 575–595.** Copy and pending-pump API maintains foreign payload copies and counters.

```text
 575 | size_t mesh_write_copy(const void *p, size_t stride, size_t bytes, size_t nslots, int node){
 576 |   if(mesh_try_attach(&CTX0,0) || !bytes || bytes>mesh_pay(CTX0.M)) return 0;
 577 |   reclaim(&CTX0); size_t done=0;
 578 |   while(done<nslots){
 579 |     unsigned char *q=credit(&CTX0); if(!q) break;
 580 |     memcpy(q,(const unsigned char*)p+done*stride,bytes);
 581 |     if(cwrite(&CTX0,q,bytes,node)!=bytes) break;
 582 |     CTX0.sub++; done++; }
 583 |   return done; }
 584 | 
 585 | size_t mesh_queue_copy(const void *p, size_t stride, size_t bytes, size_t nslots, int node){
 586 |   if(mesh_try_attach(&CTX0,0) || !bytes || bytes>mesh_pay(CTX0.M)) return 0;
 587 |   return pending_push(&CTX0,p,stride,bytes,nslots,node); }
 588 | 
 589 | size_t mesh_pump(void){
 590 |   if(mesh_try_attach(&CTX0,0)) return 0;
 591 |   if(refresh(&CTX0,0,0)) return CTX0.pending_count;
 592 |   pending_flush(&CTX0); return CTX0.pending_count+CTX0.inflight; }
 593 | 
 594 | size_t mesh_queued(void){ return CTX0.pending_count; }
 595 | size_t mesh_inflight(void){ return CTX0.inflight; }
```

**M, lines 596–617.** B: direct receive-page access. X: readv copies into foreign contiguous buffers and automatic next-read release does not prove asynchronous numerical consumption.

```text
 596 | 
 597 | size_t mesh_read(void **p, int *from){
 598 |   if(mesh_try_attach(&CTX0,0)) return 0;
 599 |   if(refresh(&CTX0,0,0)) return 0;
 600 |   pending_flush(&CTX0);
 601 |   return cread(&CTX0,p,from); }
 602 | 
 603 | size_t mesh_readv(void *p, size_t stride, uint32_t *sizes, int *from, size_t count){
 604 |   if(mesh_try_attach(&CTX0,0)) return 0;
 605 |   if(refresh(&CTX0,0,0)) return 0;
 606 |   pending_flush(&CTX0);
 607 |   size_t got=0;
 608 |   while(got<count){
 609 |     struct ring *ring=&CTX0.M->r[CMP];
 610 |     uint64_t tail=atomic_load_explicit(&ring->tail,memory_order_relaxed);
 611 |     if(tail!=atomic_load_explicit(&ring->head,memory_order_acquire) && slot(CTX0.M,CMP,tail)->bytes>stride){ errno=EMSGSIZE; break; }
 612 |     void *q=0; int src=0; size_t b=cread(&CTX0,&q,&src);
 613 |     if(!b) break;
 614 |     memcpy((char*)p+got*stride,q,b);
 615 |     sizes[got]=(uint32_t)b; from[got]=src; got++; }
 616 |   return got; }
 617 | 
```

**X, lines 618–627.** Blocking yell/lissen loops wait for stream state and retain seen masks; replace callers with the canonical configured page functions.

```text
 618 | size_t mesh_yell(const void *p, size_t n, int node){
 619 |   struct mstream s, *v=&s; mesh_yell_start(&CTX0,&s,p,n,node,0);
 620 |   while(s.st==MS_RUN) mesh_turn(&CTX0,&v,1);
 621 |   return s.st==MS_DONE?n:0; }
 622 | 
 623 | size_t mesh_lissen(void *p, size_t n){
 624 |   struct mstream s, *v=&s;
 625 |   if(mesh_lissen_start(&CTX0,&s,p,n,0)) return 0;
 626 |   while(s.st==MS_RUN) mesh_turn(&CTX0,&v,1);
 627 |   size_t g=s.done; free(s.seen); return g; }
```

