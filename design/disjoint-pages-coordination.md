# Disjoint-pages change: the coordinator's contract and its routing record

The notes the coordinating agent kept while carrying out design/disjoint-pages-change-request.md (mesh main
cb433318, metal-microbench f154814d): the contract every sub-agent works to, then the record of every sub-assignment
and review finding and where each went. The requirements are in the change request, not here: its operator's turns are
the requirements; Claude's turns, "Claude's proposal: the canonical path", Appendix A and Appendix B are the design;
"State of the code" and "Scope" bound what is touched. design/page-queue-change-request.md's "Superseded in part"
applies. Below are only the file split, the names files share, and the rules.

## Rules (every sub-agent)

- Read the change request in full first, then this contract, then your brief. Work in /Users/mdot/dox/mesh and
  /Users/mdot/metal-microbench, on main. No worktrees, no /tmp checkouts, no stash (never `git stash`, never read
  stashes).
- Edit only the files your brief says you own. If you need something in another owner's file, do not edit it: say
  what you need in your final report and the coordinator routes it. You may read any file.
- Run nothing: no bridge, no program, no ssh, no tests, on any machine. Compiling is allowed: your own file alone
  (`cc ... -fsyntax-only`, or `-c` into the scratchpad), and the document's builds. Other owners edit their files at the
  same time, so a whole build may fail on their half-done work; judge your file by compiling it alone against the
  headers. Every compile finishes within ~8 s.
- Do not open docs/facts.md at any revision. Do not use design/transport.md as a source (old comments cite it; it is not
  a source for this change). Do not touch rdma/mesh_hourly.py, rdma/mesh_preflight.py or rdma/mesh_graph.py. Never edit
  xonotic/.
- Every wait that remains is bounded (a deadline or a count set when it begins). Every failure is an error value, never
  a hardware fault and never an unbounded retry. Main is what the nodes run every hour: nothing may be able to crash or
  hang a machine (no unbounded GPU loop, no post on a queue pair across a teardown, no write outside a named page).
- No invented mechanisms: nothing the change request or this contract does not name (no hashing, receipts, credits,
  gates, filters, locks, leases, slots, rings). One configuration runs in a runtime; no call site handles both the old
  and new forms; the old form is removed once the new one covers it.
- Comments: follow the file's existing form. Cite the change request by section ("design/disjoint-pages-change-request.md,
  Appendix A, A3"). Where you reverse or change another session's behavior, leave a dated note at the spot (2026-11-03):
  what, why, what must stay. Where your code has to differ from the change request, say so at the spot and in your
  report. metal-microbench files follow docs/source_standard.md (citations and formal algebra; the dated notes the file
  already uses are its form for changes).
- Headers first: if you own a header, land its new interface (as written below) in your first commit, within your
  first minutes, so other owners compile against it.
- Commits: only your own files, by explicit pathspec (`git -C <repo> commit -m "..." -- <files>`); never `git add -A`
  or `git add .`; a new file is added by pathspec first. Commit as soon as your file compiles alone. The message says
  which of the document's builds it passes (`make -C rdma`, `make -C rdma native-audit`, `make .build/e2b_replay`), or
  "builds: none yet (compiles alone)", and ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  If `.git/index.lock` exists, wait 1 s and retry, at most 10 times. **Do not push**: the coordinator pushes both repos
  together once the document's builds pass on the committed tree (a half-integrated main would fail the nodes' hourly
  build).
- Report (your final message, short): files and functions written or removed; every difference from the change request
  and why; what you need from other owners; commit shas; compile result.

## Where things live (owner = sub-assignment)

| file | owner |
|---|---|
| mesh rdma/mesh.h, rdma/mesh-map.h (mesh_layout, mesh_layout_within) | 1 |
| mesh rdma/mesh-graph.h, rdma/mesh-graph.c | 2 |
| mesh rdma/mesh-collective.h, rdma/mesh-collective.c | 3, then 4, then 5, in turn |
| mesh rdma/mesh-flow.c | 6 |
| mesh rdma/mesh-call.h, rdma/mesh-call.c, rdma/mesh-dataflow.h, rdma/mesh-dataflow.c, rdma/mesh.py | 7 |
| mesh rdma/mesh-records.c, rdma/mesh-link.c, rdma/mesh-bridge.h | 8 |
| mesh rdma/mesh-metal.h, rdma/mesh-metal.m | 9 |
| metal-microbench metal_recording.h, metal_recording.m | 10 |
| metal-microbench tools/e2b_replay.m | 11 |
| reviews (written findings only) | 12, 13 |
| rdma/Makefile, this file | coordinator |

## Names files share (fixed now; owners define them exactly so)

### The region (mesh.h, owner 1)

- `MESH_VERSION` 108.
- `struct hdr`: `depth` (offset 40) becomes `uint32_t vacant7` and `hop_stride` (offset 240) becomes `uint64_t vacant8`,
  never read or written (the file's rule: no field moves). Appended after `stopping`, in this order:
  `uint64_t word_off,word_capacity,landed_off,landed_capacity,landed_stride,map_room;` (sizeof 400). `word_off`,
  `word_capacity`, `landed_off`, `landed_capacity`, `map_room` are the layout's; `landed_stride` is the plan's (written
  by mesh_plan before `hops`, like `executions`). `map_off` stays where it is and is now set by the layout.
- `#define MESH_SUPPLIED UINT64_C(1)`: the value a producer stores into a supplied word.
- `#define MESH_WORD_STRIDE 8u`: bytes between one supplied word's consecutive executions.
- `struct mesh_hop { uint32_t page,node,direction,padding; uint64_t word,mask,serial; }` (40 bytes, `word` at 16):
  `forwards` gone, `position` renamed `mask`. One entry is one hop of one page in one execution.
- `struct mesh_publication { _Alignas(64) uint32_t sends,landed; _Atomic uint64_t argument; uint64_t device_input,device_stride; struct mesh_target targets[]; }`
  (offsets unchanged; `landed` takes the padding at 4).
- `struct mesh_cancel_range { _Alignas(32) uint64_t offset,count,stride,padding; }` and `mesh_cancel(m, cancel, link)`:
  for j < count, the word at region offset `offset + j * stride` is stored UINT64_MAX (release) where it reads 0, and
  `requested` set, as now.
- Helpers: `mesh_supplied(m, release, t)` = `(_Atomic uint64_t *)((char *)m + m->word_off + (release * m->executions + t) * MESH_WORD_STRIDE)`;
  `mesh_landed_record(m, t)` = `(_Atomic uint64_t *)((char *)m + m->landed_off + t * m->landed_stride)`.
- `struct mesh_send` stays (other code names it); nothing of the plan uses its 256-byte size any more.

### The layout (mesh-map.h mesh_layout, owner 1)

Laid out after the page lists (`hop_off`), in this order, each page-aligned like the rest: the supplied words
(`word_off`, `word_capacity` = 8 x `hop_capacity` bytes), the landed list (`landed_off`, `landed_capacity` = 40 x
`hop_capacity` bytes), then the bridge's pages (`bridge_off`, as now), then the map's records (`map_off`, `map_room` =
`MESH_MAP_ROOM` records of `struct mesh_record`, `#define MESH_MAP_ROOM 256u` in mesh-map.h), then the arena
(`data_off`). The map is laid out once with the region (Appendix A, A3, "Sharing the region between programs"); no
bridge page is taken from the arena.

### The plan's words: supplied words and the landed list (written by mesh_plan; read by the bridge, the GPU, the host)

- **Supplied word** of release r (this node's r-th range it is the source of, in the plan's demand order) in execution
  t: `mesh_supplied(m, r, t)`. mesh_plan zeroes the ones it uses. The producer stores `MESH_SUPPLIED` (release) after
  its last write to the range in execution t (the GPU's publication, or mesh_host_publish). Nothing cancels a supplied
  word: 0 releases nothing.
- **Landed record** of execution t: `mesh_landed_record(m, t)`, `landed_stride` bytes, `landed_stride = 32 x ceil((1 +
  ceil(R / 64)) / 4)` where R is this node's receive hops in one execution (the same in every execution; R = 0 gives
  32). Word 0 is the step's **cancel word**: 0, or UINT64_MAX once cancelled. Bit b (b < R, the b-th receive hop of the
  execution in list order) is word `1 + b / 64`, bit `b % 64`. mesh_plan zeroes records 0 .. executions - 1. A
  receive's landing ORs its bit in (`atomic_fetch_or`, release). Nothing clears a bit while the plan stands.
- A destination's pages of one range are consecutive receive hops at that node, so a range's bits are one run
  `[first, first + P)` (P its pages).
- **Hop entries.** `hops` = executions x n entries; execution t's n entries at `[t n, t n + n)`; every execution's
  entries in the same order with the same `node`, `direction`, `serial`; `page`, `word`, `mask` are execution t's own.
  - A receive: `page` its page in execution t; `word` the region offset of the landed-record word holding its bit;
    `mask` = `1 << (b % 64)`.
  - A send from a range's source: `word` = the region offset of `mesh_supplied(m, r, t)`, `mask` = `MESH_SUPPLIED`.
  - A forwarding send (a relay's, or a destination's that forwards): `word` and `mask` are those of its receive of the
    same page in the same execution.
  - A send is released when `(atomic_load(word, acquire) & mask) == mask`. A landing does
    `atomic_fetch_or(word, mask, release)`.
  - Both ends: for adjacent a, b, execution t's entries at a naming b with MESH_SEND and at b naming a with
    MESH_RECEIVE are the same hops in the same order; the bridges assign the pair's hops in list order (execution 0's,
    then 1's, ...) by the rule both ends compute, as now.
- **Cancellation.** The bridge cancels a receive not landed (link_close, plan_cancel) by storing UINT64_MAX into its
  execution's cancel word; that record's start is
  `landed_off + ((word - landed_off) / landed_stride) * landed_stride`. The program's cancellation (mesh-metal.m's
  transport `cancel`) is the range `{offset = landed_off, count = executions, stride = landed_stride}`. The GPU lane
  that reaches its bound stores UINT64_MAX into its step's cancel word and every later step's (review 13, rounds 1 and
  2), so the program reads every step's cancel word after the run and a cancelled step is its error value.
- **Rows' publications** (struct mesh_publication), on instance 0's rows of a declared section (rows `first + k`, k <
  P): a destination page: `landed` = its bit b, `device_input` = `landed_off` (execution 0's record), `device_stride` =
  `landed_stride`; a source range: its first row's `targets[i] = {stream = region offset of mesh_supplied(m, r, 0),
  count 1, stride 1}`, `sends`, `argument = MESH_SUPPLIED`; execution t's word is `stream + t x MESH_WORD_STRIDE`.

### Writing a plan (mesh_plan, owners 3-5)

Compute everything first (demands, routing, values, dependence graph, allocation); a failure there leaves the previous
plan standing. Then: store `hops` 0 (release) and a release fence; wake `plan`; `mesh_plan_erased(context, previous
generation)` (below); write `executions` and `landed_stride` (so mesh_supplied and mesh_landed_record name this plan's
words wherever it zeroes and writes); bind the declared sections' rows to their pages; zero every page it binds to a declared section's instances (a peer's SEND held in credit lands as soon as this node's bridge
loads the plan, so the program cannot zero a routed operand after the plan), and the supplied words and landed records
it uses; write the entries and the new generation `plan`; read the map's serial again
(ESTALE where it moved); store `hops` (release); wake `plan`. ENOSPC past `hop_capacity`, `word_capacity` or
`landed_capacity`; ENOMEM where the allocator finds no room in the program's window.

### The program's section (mesh-dataflow.h / mesh-call.h, owner 7)

- `struct mesh_ctx` gains `struct mesh_range window, local;`: the program's claim, arena pages `[first, first +
  count)`, block-aligned, `window` inside the registered window and `local` outside it; and loses `send_off`,
  `send_bytes`. Owner 7 decides the rest of its fields (rows of the claim, cursors) and mirrors the struct in mesh.py.
- `mesh_attach` makes the program's one claim (Appendix A, A3 "Sharing the region between programs", A4 "One claim of
  a program's section at attach remains"): its rows and its two page runs, held until the client retires and released
  by the bridge's retired release (`mesh_retired_release`), never while a port's `prepared` holds a plan. Nothing else
  takes rows or pages from the region: `mesh_rows_alloc` and `mesh_arena_alloc` are no call site's allocator.
- `int mesh_section_create(struct mesh_ctx *, size_t bytes, uint32_t count, int wire, struct mesh_section *)` (same
  signature): names `count x stride` rows of the claim, `stride` = span / block rows an instance, instance s at rows
  `first + s x stride`. `wire` 1: no page is bound (mapping MESH_ABSENT, address 0): the plan's allocator assigns its
  pages; `wire` 0: a value no plan names, laid out in the claim's `local` run in creation order and touched, as now.
  ENOMEM past the claim.
- `void *mesh_section_address(struct mesh_ctx *, struct mesh_section, uint32_t index)`: NULL where the row has no page.
- `mesh_section_slice` is removed (its one caller's page cut is gone).
- `int mesh_plan_erased(struct mesh_ctx *, uint64_t generation)` (mesh-call.h, defined in mesh-dataflow.c beside
  mesh_transfers_start): 0 once no port's `prepared` holds `generation` (0 at once where generation is 0), ETIMEDOUT
  past MESH_PLATFORM_NS from its start. It replaces the retention of the plan's words section (the landed list and
  supplied words are fixed areas the next plan reuses). It begins with a seq_cst fence; a port stores `prepared` before
  its final re-read of `hops`/`plan`, with a seq_cst fence between, and stores 0 again where they moved (Dekker's
  pair: either the port sees `hops` 0 and withdraws, or the wait sees its `prepared`).
- `mesh_backing_bind(context, first_row, pages, page, index)` stays: the plan binds an instance's rows with it.
- `mesh_publication_prepare(m, row, records)`: a target's word k is `stream + MESH_WORD_STRIDE x stride x k`.

### The plan call (mesh-collective.h, owner 3)

```c
struct mesh_declared { uint32_t source,destination,identity,order; struct mesh_section section; };
/* a program's data edge: the destination of declared[consumed] has read it in execution t before the source of
   declared[produced] first writes it in execution t + lag (that node is both); EINVAL where it is not */
struct mesh_dependence { uint32_t consumed,produced,lag; };
int mesh_plan(struct mesh_ctx *,const struct mesh_declared *,uint32_t count,uint32_t executions,
              const struct mesh_dependence *,uint32_t dependences);
```
A declared section at its source or a destination is a `mesh_section_create(bytes, executions, 1, ...)` section:
`count` == executions, P = ceil(bytes / (block x pgsz)) pages an instance, instance t at rows `first + t x stride`. Its
rows are unbound, or bound by an earlier plan inside `window`; a row bound outside `window` is refused (EINVAL). The
plan assigns every instance's pages and binds its rows; a program reads its sections' addresses after its plan. A
range is one declared transfer per destination (no page cut).

### The allocator (mesh-collective.c, owners 3 and 4; the graph algorithms owner 2's)

- Values at this node: each demand whose pages this node holds: its declared section where this node is the source or a
  destination, else (a relay) its relay pages, `d->pages` blocks, with no rows. Instance (v, t) for t < executions.
- The dependence graph is periodic: a template, one copy per execution, edges carrying an execution distance `lag`
  (lag 0 within an execution). Template vertices per demand d: P(d) its production at the source; for each node y on
  d's trees other than the source, W(d, y) its first landing and L(d, y) its last; for each tree edge y -> c (distinct
  pairs over d's trees), F(d, y, c) y's last forward to c; for each destination z, C(d, z) its consumption. Edges (lag
  0): P -> W(y) for every y; P -> F(source, c); W(y) -> L(y); W(y) -> F(y, c); F(y, c) -> L(c); L(z) -> C(z); and each
  `mesh_dependence` e: C(demand of consumed, x) -> P(demand of produced) with lag e.lag. Every edge is sound: its tail
  happens before its head whatever the cables do.
- An instance's write is P(d) at the source, W(d, y) elsewhere; its reads are the F(d, y, c) of its node and C(d, y)
  where its node is a destination. Two instances A = (v, t), B = (w, t') may share a page only if every read of A
  happens before B's write (a path of one or more edges from each read at t to the write at t') or every read of B
  happens before A's write (Appendix A, A3, "Interference, by happens-before").
- mesh-graph.h (owner 2):

```c
/* Happens-before over a dependence graph repeated `iterations` times, its edges carrying iteration distances (M. Lam,
   PLDI 1988; B. R. Rau, MICRO 1994): template vertex v of iteration i is (v, i); template edge k, in the row
   [first[v], first[v + 1]) of the vertex v it leaves, joins (v, i) to (head(context, k), i + lag(context, k)) for every
   i, lag >= 0.  For source[s] (s < sources) and target[x] (x < targets), bit L % 64 of
   reach[(s * targets + x) * words + L / 64] (words = (iterations + 63) / 64) is set where a path of one or more edges
   joins (source[s], i) to (target[x], i + L) for i + L < iterations (the same for every such i).  0; EDEADLK where the
   lag-0 edges hold a cycle; ENOMEM. */
int mesh_graph_happens_before(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                              uint32_t (*lag)(const void *context, size_t k), const void *context, uint32_t iterations,
                              uint32_t sources, const uint32_t *source, uint32_t targets, const uint32_t *target,
                              uint64_t *reach);
/* The left-edge colouring of items that each need a run of colours (T. Hashimoto and J. Stevens, DAC 1971; M. C.
   Golumbic 1980; with sizes, greedy by size: Y. Pisarchyk and J. Lee, arXiv 2001.03288, 2020): the items in the
   caller's order order[0 .. items), item i given colours [colour[i], colour[i] + size[i]) at the lowest colour where
   its run overlaps no item placed before it that it interferes with (interferes(context, a, b) != 0, symmetric);
   *used the most colour[i] + size[i].  0; ENOSPC where a run would pass `capacity`; ENOMEM. */
int mesh_graph_left_edge(uint32_t items, const uint64_t *size, const uint32_t *order,
                         int (*interferes)(const void *context, uint32_t a, uint32_t b), const void *context,
                         uint64_t capacity, uint64_t *colour, uint64_t *used);
```

- The allocator (owner 4) is one function called once per plan inside mesh_plan before anything is written: the items
  are every instance at this node, sizes its blocks, ordered by size (largest first), then by value in the plan's demand
  order, then by execution; interference as above from `mesh_graph_happens_before`'s table; capacity the claim's
  `window` blocks. The colours are blocks of `window`: instance (v, t)'s page k is arena page `window.first + (colour +
  k) x block`. ENOSPC from the colouring is mesh_plan's ENOMEM. Where no read at this node happens before any write at
  this node at any distance (no program edge reaches back), every pair interferes and the colouring is each instance
  after the last in that order, without the pairwise test. With that order and no reuse, a value's executions lie in
  one run, instance t at `run + t x blocks`, which the GPU binds at one stride.

### The bridge (mesh-flow.c, owner 6)

The ring goes: `plan_shape`, `struct flow_hop`, `landed_through`, the receive window (`need`), the forwards gate
(`sent`, `forwarding`, the offset-8 count). Units are the pair's hops in list order. A send's record keeps `word` and
`mask`, released as above; a receive's record keeps its landed word, its `mask`, and its execution's cancel word. A
receive is posted as its lane's room allows (MESH_LANE_UNITS, frames). link_close and plan_cancel cancel by the cancel
word as above.

### The GPU (mesh-metal.h/.m owner 9; metal_recording owner 10)

```c
struct mesh_metal_transport { void *publication,*inputs; struct mesh_cancellation *cancel; };  /* the supplied words'
   area and the landed list's area (hdr word_off/word_capacity, landed_off/landed_capacity), wrapped once */
struct mesh_metal_input { void *landed; uint64_t offset,stride; uint32_t first,count; };      /* 32 bytes */
struct mesh_output_status { uint32_t remaining,padding; };                                     /* MESH_WORD_STRIDE bytes */
int mesh_metal_receive_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,struct mesh_metal_input *);
```
- `mesh_metal_receive_prepare(section)`: the declared destination section; `landed` the landed buffer (retained),
  `offset` the byte offset in it of execution 0's record, `stride` = `landed_stride`, `first`/`count` the range's bit run
  (count = P); EINVAL where a page has no receive or the bits are not one run.
- `mesh_metal_publication_prepare`: its `stride` is MESH_WORD_STRIDE; records relative to the supplied words' buffer;
  statuses `struct mesh_output_status`, one a range and invocation, MESH_WORD_STRIDE apart (the direct publication binds
  statuses and supplied words at one per-execution stride).
- metal_recording.h:
  `- (void)operand:(id<MTLBuffer>)operand stride:(NSUInteger)stride outputOffset:(NSUInteger)offset;` (the instance
  count and its modulus go: execution t binds `offset + t x stride`), and
  `- (void)receive:(id<MTLBuffer>)operand landed:(id<MTLBuffer>)landed offset:(NSUInteger)offset stride:(NSUInteger)stride first:(NSUInteger)first count:(NSUInteger)count;`
- A remote read (mesh_read_N): the kernel's input binding is execution t's landed record (bound at `offset + t x
  stride`); a lane reads the step's mask with vector loads (the cancel word and the words holding bits [first, first +
  count)) once per operand, not one word per page; a lane whose operand's pages have not all landed is predicated off
  its loads and takes them on a later pass, its next read of the step's mask; the passes are bounded at 2^24 (about 8 s
  at the recorded 0.58 us fenced read); one acquire fence once the mask is whole; a cancelled step (cancel word
  UINT64_MAX) or the bound gives T{}, as the bounded wait did. Nothing spins on a page. Local inputs (a direct
  publication's countdown on this GPU) are no landing and stay as they are.

### The program (e2b_replay.m, owner 11)

One declared range per route and destination, the operand's whole section; no page cut, no MESH_RING_DEPTH; a routed
operand is `mesh_section_create(bytes, steps, 1)`; `mesh_plan(&context, declared, at, steps, NULL, 0)`; addresses read
after the plan, each routed operand's executions checked to lie at one stride (refused otherwise, at preparation); one
`mesh_metal_receive_prepare` per received operand; the recorder's new `receive:` and `operand:` forms.

## Record

(Each sub-assignment, its commits, and each review finding with where it was routed, appended as the work proceeds.)

### Sub-assignments

1. mesh.h, mesh-map.h: proposal items 2 and 3, item 1's `depth` and slot layout, A4's hdr fields, A3's bridge pages laid
   out with the region (the map's area).
2. mesh-graph.{h,c}: A3's happens-before (periodic) and the left-edge colouring with sizes, consumers as operands (C9).
3. mesh-collective.{h,c}, first: item 1's unrolling into value instances, A3's inputs (the values and the periodic
   dependence graph with the program's and the transport's edges), Appendix B; the call's new signature.
4. mesh-collective.c, second: A3's allocator (interference by happens-before, pages by the colouring, runs greedy by
   size, ENOMEM before anything runs).
5. mesh-collective.{h,c}, third: item 1's slots go (`plan_pages`, `plan_row`, the relay section's slots), hops per
   execution with the landed list and supplied words, the host functions, the write order.
6. mesh-flow.c: item 1's ring on the bridge (plan_shape, the window, the forwards gate), item 2's landings into the
   landed list, item 3.
7. mesh-call.{h,c}, mesh-dataflow.{h,c}, mesh.py: A3 and A4's call-site allocators; the one claim at attach.
8. mesh-records.c, mesh-link.c, mesh-bridge.h: A3's bridge records laid out with the region; A4's mesh-records.c call.
9. mesh-metal.{h,m}: item 4's struct mesh_metal_input; item 3.
10. metal_recording.{h,m}: items 2 and 4 on the GPU (mesh_read_N by the step's mask); item 1's GPU modulus.
11. tools/e2b_replay.m: its page-sized ranges, now one range an operand and one mask per step; item 1's depth.
12, 13. Reviews of finished pieces against the change request, written findings routed to the owners.

### Commits and routing

- 1: 67631dbe (interface), e3c2442c (comments; the map's lease comments, from 8's report), 4bc6bc29 (MESH_ROW_HOT and
  MESH_FREE, read by no code once 7's claim landed, removed).
- 2: 9617a36b (declarations), b399746f, 84f0e1c8.
- 3: d1b1f88f (signature), 86e14049 (sections checked, values, periodic template). Its C9 hook remark on
  plan_template: ruled it stays in mesh-collective.c, since it builds the graph module's operands from the
  collective's own events and every graph computation in it goes through mesh-graph.
- 4: 38162b8d (plan_allocate, p->colour). Its C9 hook remarks: plan_before and plan_interferes stay (the colouring's
  interference operand, as plan_leaf is the flow solve's); the inline placement where no bit is set was the colouring's
  own result outside the graph module, so 2 gives mesh_graph_left_edge `interferes == NULL` (every pair interferes,
  O(items)) and 5 swaps the call. Left open: with program edges reaching back across executions, the colouring's walk
  is O(items^2) interference tests, past ~8 s at e2b's largest sizes; no program declares such edges today.
- 2 again: cc08dc6a (`interferes == NULL`).
- 5: fd49e3b5 (the per-execution list, landed bits, supplied words, publications, rows bound and their pages zeroed
  before `hops`, the write order, the host functions; plan_pages, plan_row, plan_word, plan_hop, plan_release,
  plan_held, MESH_PLAN_POSITION and the relay and words sections removed). Its difference: plan_disjoint refuses two
  demands whose sections at one node share rows (a value is one demand's); tools/prepare_e2b.py publishes each name
  from the one rank that computes it, so no e2b operand is both received and sent at one rank. `executions` and
  `landed_stride` are written before the zeroing that reads them (review 12's 7). On ETIMEDOUT or ESTALE after the
  erase, `hops` stays 0.
- 6 again: 3c685a57 (review 12's 1, the port's side of the pair). Its remaining edge: a start reading `prepared` in
  the instant before an EAGAIN clears it, which needs a client starting one plan while writing the next.
- 7 again: 8c7c0ea9 (review 12's 1, the plan's side of the pair; 3, the bounded wait at attach).
- 10 again: e9c16dd2 (review 13's 1-6, 13). Left open: a local (countdown) wait past its bound ends with zeros and no
  mark, as before (it has no cancel word).
- 6: 5d919bd7; aab6b100 (6's own finding: plan_cancel in the bridge parent could read entries mid-rewrite and pair a
  receive with a supplied word; fixed by the plan's write order, the seqlock re-read link_prepare makes, not a bounds
  check). Left open: a cancel that passes the check can still land after a re-plan zeroes the records and cancel a step
  of the new plan (a step cancelled, never a write outside the landed list).
- 7: add0ef2c (headers, mesh.py), ae0bc2f1. Left open: a retired claim a port still holds leaves the next client no
  pages until the bridge releases it.
- 8: b4e1915b; 9b9ded33 (mesh-ports.c:447's held_pages comment, a file with no owner, given to 8 for that hunk).
- 9: 8e2e2f7e (interface), 90e28099; 6a134767 (review 13's 10-12: a row's record and stride compared with hdr
  `landed_off`/`landed_stride` exactly; invocations past `executions` refused; a plan written is hdr `plan` != 0, not
  `hops` != 0). The multi-range publication form stays: it predates the page cut (operands.json's ranges) and is
  outside the change request's Scope.
- 10: 728e3b39 (interface), 07c808fa.
- 11: ddf5ce0b. Its finding that the program zeroes a received operand after mesh_plan, when a peer's SEND held in
  credit may already have landed: ruled the plan's job (5 zeroes the pages it binds before `hops`; 11 stops zeroing
  routed operands). tools/mesh/run_pair.py's MESH_RING_DEPTH forwarding and ring_depth parser, dead with the ring,
  given to 11 for that hunk. 01499f5f (both). 96a22b62 (review 13's 6, 9, 13: after the run each step's cancel word is
  read and a cancelled step is the run's error value, ECANCELED for the program's own stop, else ETIMEDOUT; the
  placement stride's cases stated; comments to the source standard). 11's own finding: a stop signal between the GPU's
  end and that check now fails a finished run (a finished step's cancel word stays 0, so mesh_cancel marks it); the
  note at the stop handler says so.

- Second review round (12: 38162b8d cc08dc6a fd49e3b5 3c685a57 8c7c0ea9 430c3cda 4bc6bc29 9b9ded33; 13: 6a134767
  e9c16dd2 96a22b62 and the plan's side of fd49e3b5): 13's 1 (a lane past its bound cancelled only its own step, so
  with the bridge gone every later step spent its own bound, about 21 min at 256 steps) to 10: da3c9718 (the lane
  cancels records t .. T-1 through one stride-0 binding of the landed list a dispatch); 13's 3 (the landed binding's
  write bit excluded by a name that map inlining renames) to 10 (excluded by the buffer); 13's 4 (an uncited claim
  about SIMD-group broadcast) to 10; 13's 5 to 11 (24ec8cda: steps != executions refused); 13's 6 to 9 (e2db9343);
  13's 2 and 12's 2 (the cancel word's writers; where `executions`/`landed_stride` are written) to 1 (36c72208,
  707faf3a, c4a3fcc0), 5 (b77c6b30, d8f36c83) and this contract; 12's 1 to 5 (b77c6b30); 12's 3 to 6 (73d6853c, `hops`
  loaded with acquire in both re-checks); 12's 4 to 1 (mesh_row_chunks, without a caller, removed). Left: 13's 7 (a
  link loss and a GPU bound both read as ETIMEDOUT: the cancel word has two values). The C9 hook's remarks on
  da3c9718 name MetalPromoteOperand's and MetalInlineMaps' forward walks, from 864a0ae4 (2026-10-01), outside this
  change's Scope; not routed.

### Integration

The document's builds on the committed trees, every in-scope output rebuilt: `make -C rdma` 0, `make -C rdma
native-audit` 0, `make .build/e2b_replay` 0. Pushed: mesh cb433318..fd49e3b5, metal-microbench f154814d..e9c16dd2.
After the second round, the same three builds 0 again; pushed: mesh fd49e3b5..d8f36c83, metal-microbench
e9c16dd2..da3c9718. Review 13's last pass (da3c9718 24ec8cda e2db9343): no defect; its one cleanup (the recorder's
repetitions must not exceed the plan's executions, which only the caller enforces) stated at both spots, cfada099
(10) and 472390a8 (11).

### Open at the end

- A cancel from the bridge parent's plan_cancel that passes its re-check can still land after a re-plan zeroes the
  landed records and cancel a step of the new plan (a step cancelled, never a write outside the landed list).
- A port that keeps a retired client's plan past MESH_PLATFORM_NS (lost with requests outstanding until its child is
  reaped) leaves the next client's claim whatever is free after a 5 s wait at attach.
- With declared program edges that reach back across executions, the colouring's walk is O(items^2) interference
  tests, past ~8 s at e2b's largest sizes; no program declares edges today, so nothing is reused across executions.
- A local (countdown) wait past its bound ends with zeros and no mark; a link loss and a GPU bound both read as
  ETIMEDOUT; the host's final completion wait in e2b_replay.m is unbounded (pre-existing).
- The "later pass" is a bounded wait on the step's mask (2^23 fenced reads, about 4.9 s), not a later dispatch: the
  replay is one command buffer of every step, and a step's peer wait is about 1.9 ms.

### Reviews and where their findings went

- 13 (GPU and program; 8e2e2f7e 90e28099 728e3b39 07c808fa ddf5ce0b 01499f5f): 1 (a lane past its bound restarted the
  bound on every later read: GPU work could outlive its process) and 2 (2^24 passes is at least 9.7 s) to 10, with 2^23
  and a failed memo; 3, 4, 5 (the "later pass" is a bounded wait on the step's mask; the memo is per lane; a fence a
  pass) to 10, stated at the spot with the measured peer wait as the reason; 6 (a bound or a bridge cancel finished
  with zeros as a success) to 10 (the lane past its bound cancels its step through the cancel word) and 11 (the run
  reads every step's cancel word); 7 (the plan's zeroing missing from the contract) taken into "Writing a plan" and to
  5; 8 (the multi-range publication forms) ruled out of Scope; 9, 13 to 11; 10, 11, 12 to 9. Left open: the host's
  final completion wait in e2b_replay.m is unbounded (pre-existing, outside Scope).
- 12 (mesh side; 67631dbe e3c2442c 9617a36b b399746f 84f0e1c8 5d919bd7 aab6b100 add0ef2c ae0bc2f1 b4e1915b d1b1f88f
  86e14049): 1 (a port could store `prepared` after mesh_plan_erased read it and serve an erased plan over the reused
  areas) to 6 (store `prepared` before the final re-read, seq_cst fence between, 0 again on EAGAIN) and 7 (a seq_cst
  fence first in mesh_plan_erased), stated in mesh.h by 1 (430c3cda); 2 recorded open (above, 6); 3 (a retained claim
  of every row leaves the next client ENOMEM at attach) to 7 (mesh_attach waits, bounded, for the release; 8c7c0ea9);
  4, 6, 7, 8, 9, 10, 11, 12 to 5; 5 to 1 (430c3cda).
