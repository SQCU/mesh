# Pages and functions

The [complete replacement requirements](completion-requirements.md) track the
whole implementation and its acceptance evidence. On the actual Thunderbolt
substrate, literal pages use SEND/RECV, as documented by Apple TN3205; software
publication of a received page does not require hardware remote-write support.
Statements below about instant execution and topology-independent utilization
express the desired absence of added control dependencies, not a zero-latency
theorem or a measured performance guarantee.

Plain statements of the algorithm. Only the words used in the operator's own
turns: mesh, peers, pages, page table, rows, stamps, use count, buffers as
function returns, functions, release, reduce, accumulator page, index page,
NFE, status word, digest, metadata. Every other word that has appeared in code
("stage", "chunk", "version", "gate", "token", "group", "consumer",
"scheduler") named a control-flow object standing in for something the page
table already holds, and is not part of the algorithm.

## One NFE on two peers

### Setup

One page table holds every page the runtime will use. Each function output
is a range of rows in that table. A row holds three things: which page, its
stamp, and its use count. The use count is the use-checker. The page table is
the only state.

Every function has a known number of output pages. Every function input is
the output pages of some function, on this node or on a peer. All of this is
known before the NFE starts.

All page-table allocations needed by the complete compute graph are determined
before launch, including parameters, intermediates, accumulators, indices and
outputs. Numerical execution does not discover allocations or introduce
data-dependent control flow to repair missing storage.

The calling process may invalidate the page table whenever it chooses, including
when a kernel is hung. The mesh API must expose that operation directly. Pages
may be consumed, freed and invalidated; these operations do not start recovery,
retry, parity, or waiting protocols inside computation, transport or reduction.
The caller decides whether to rerun the complete feed-forward NFE.

Every node has NVMe storage or is transitively RDMA-connected to a node that
does. This is a deployment invariant, not a capability to infer during an NFE.
Deleted or invalidated DNN parameter pages can be loaded again from local NVMe
or retransmitted from such a peer. Parameter residency is not a reason to retain
an invalidated graph or add a recovery protocol to its numerical functions.

### Asynchronous error metadata

The calling interface is `out, meta = meshfunction(x)`. `out` contains numerical
values; `meta` contains literal error codes and their callgraph location and
occurrence (where and when). Error status returns are permitted only through
this asynchronous metadata channel. Returning the pair does not synchronously
wait for execution or for a final error verdict.

Composition carries metadata monadically alongside values. A function in the
callgraph does not consume error metadata to branch, short-circuit, suppress
arithmetic, synchronize, wait, retry or recover. Only the calling context
interprets and handles it, including deciding whether to repeat the complete
feed-forward NFE. Here monadic propagation does not mean exception-style
short-circuiting on an error.

Metadata storage is also configured before launch and occupies literal
page-table pages. It is a distinct return channel, not a private allocation,
completion token or readiness gate. Error codes never replace numerical output
values. A missing numerical output remains missing; publishing metadata neither
fabricates an output nor certifies that outstanding work has completed.

### The NFE, number k

1. A function runs on the GPU. It writes its output straight into pages. When
   a page is written, its row gets stamp k. Nothing else is said.
2. Pages the peer needs go over the mesh. On the peer they land in free pages.
   The peer's rows for that output get the page and stamp k.
3. A reduce is a function. Its inputs are two buffers: my partial and the
   peer's partial. It looks at the rows of both. When row p carries stamp k in
   both, it reads page p from each, adds them in high precision into an
   accumulator page, and marks p in its index page. When every page of a row
   of the result is in, it writes the normalized row into its output pages and
   stamps them k. It does this page by page as pages arrive. Its output pages
   go to the peer.
4. The next function has two inputs: my reduce output and the peer's reduce
   output. When all their rows carry stamp k, it runs. It reads those pages
   directly. There is no copy.
5. The GPU is given a function only after all its input rows carry stamp k.
   So the GPU never waits.

### Freeing

6. When function g's output rows carry stamp k on every node that needed
   function f's output, f's output pages of NFE k are done. The runtime calls
   release on those rows. The use count goes down. At zero the page goes back
   to the free list and its contents are zeroed, asynchronously. A page that
   came from the peer goes back to the bridge. No node tells another node
   anything.
7. A page is written again only after release. Two NFEs can be in flight
   because rows for NFE k+1 point at other pages until the rows for NFE k are
   released.

### Checking

8. After a reduce has read its inputs, the node hashes the pages it sent and
   the pages it received. It writes the numbers into a page. That page goes to
   the peer. When both digest pages are present, the node compares them.
9. A difference is reported in asynchronous error metadata. Only the calling
   context interprets it and decides whether to repeat the complete NFE.

### Link error

10. A negative link status is reported with its location and occurrence in
    asynchronous metadata for the affected invocation. The callgraph does not
    consume that status. The calling context may reject the result and rerun
    the complete feed-forward NFE; mesh does not perform recovery or replay.

## The same NFE on infinitely many Mac Minis

(After `metal-microbench/docs/mesh_distributed_reduction_analysis.md`:
install the program once; keep weights where they live; keep each
intermediate at its next consumer; a finished piece enables its local
continuation with no host decision; every Mini has at most three links; a
result at depth D comes back in D hops; infinite workers do not remove the
per-link ceiling; consume a partition as soon as it has arrived.)

### The mesh

There are infinitely many Mac Minis. Each one has at most three links. Each
link joins two adjacent Minis. We look at one link and the two Minis on its
ends. Everything below is what one Mini does. Every Mini does the same thing.
Nothing a Mini does depends on how many Minis there are.

### What a Mini holds

A Mini holds its share of the program's weights. They never move. It holds
one page table. The table has a row for every page the Mini will ever use. A
row holds a page, a stamp, and a use count. That table is the Mini's whole
state. Every function output is a range of rows in the table. Every function
input is some function's output rows, on this Mini or on an adjacent Mini.

### One NFE, number k

1. A function's input rows all carry stamp k. Now the function runs. It is
   given to the GPU only now. The GPU never waits.
2. The function writes its output into pages. When a page is written, its row
   gets stamp k. Nothing else is said.
3. An output page that an adjacent Mini needs goes over the link to that Mini.
   It travels one hop. It lands in a free page there. The row for it on that
   Mini gets the page and stamp k. A result at depth D from here arrives after
   D hops. That is the whole latency of transport.
4. A reduce is a function. Its inputs are my partial and the partial that
   arrived from the adjacent Mini. When row p of both carries stamp k, the
   reduce adds page p of each in high precision into an accumulator page, and
   marks p in its index page. It does this page by page as pages land. It does
   not wait for the rest. When a row of the result is complete, it writes that
   row into its output pages and stamps them k. Those pages go one hop to the
   adjacent Mini.
5. The next function reads my reduce output pages and the pages that came
   back from the adjacent Mini. When all those rows carry stamp k, it runs. It
   reads the pages where they are. There is no copy.
6. Partials are combined as they pass. A Mini that has two partials for the
   same rows adds them before the pages go on. So the pages crossing any link
   are bounded no matter how many Minis lie beyond it. Infinite Minis do not
   raise the load on this link.

### Freeing

7. When the rows of function g's output carry stamp k on every Mini that
   needed function f's output, f's output pages of NFE k are done. The runtime
   calls release on those rows. The use count goes down. At zero the page
   returns to the free list, and it is zeroed asynchronously. A page that came
   over the link goes back to the bridge. No Mini tells any Mini anything. The
   proof that the page may be freed is the stamp on the dependent's rows, and
   that stamp already had to arrive for the next function to run.
8. A page is written again only after release. Two NFEs can be in flight
   because the rows for NFE k+1 point at other pages until the rows for NFE k
   are released.

### Checking

9. After a reduce has read its inputs, the Mini hashes the pages it sent over
   the link and the pages it received over it. It writes both numbers into a
   page. That page goes one hop. When the adjacent Mini's digest page has
   arrived and mine is written, the Mini compares them.
10. A difference is reported in asynchronous error metadata. Only the calling
    context interprets it and decides whether to repeat the complete NFE.

### Link error

11. A negative link status is reported with its location and occurrence in
    asynchronous metadata for the affected invocation. The callgraph does not
    consume that status. The calling context may reject the result and rerun
    the complete feed-forward NFE; mesh does not perform recovery or replay.

### Why two Minis are the same as infinitely many

Each Mini waits only on its own input rows. Each link carries only the pages
its two ends exchange, already combined. So the latency of an NFE is the
number of hops on its longest dependency chain times the hop time, plus the
service time of the functions on that chain. The FLOPs utilization of a Mini
is the fraction of its roofline its functions reach while their inputs are
present. Neither number changes if you draw the mesh as two Minis or as an
infinite tree of rings. A presentation that changes them — a stage index, a
chunk, a gate, a group, a copy into a hidden buffer, a wait on anything but
stamped rows — is not describing this algorithm.

## Addendum: waiting for messages about data, on infinitely many Minis

Suppose a Mini does not wait on its input rows. Suppose it waits for a
message that says the data is coming, or is done, or may be used: a token, a
gate value, a "ready" flag, a scheduler's decision, a completion callback that
stands in for the pages. Then it acts on the message. This is what goes wrong.

1. **The message is a second thing on the same link.** The pages take one
   hop. The message about them takes a hop too. Every dependency on the chain
   now costs a hop for the pages and a hop for the message. On a chain of
   depth D the NFE waits for D messages it did not need. On an infinite mesh
   D is not bounded, so the extra wait is not bounded either. Waiting on the
   rows costs nothing extra: the row is stamped when the page lands.
2. **The message can be true and the data still absent.** A message says
   something about the sender's state. The pages may still be in flight, or
   lost, or land out of order, or be corrupt. Acting on the message is acting
   on a claim. To make the claim safe, the sender and receiver need
   acknowledgements, retries, and ordering. That is a protocol on top of page
   delivery. The digest, computed after the pages were used, already finds a
   wrong page without any of it.
3. **The message can be false and the data present.** The pages landed and
   the rows are stamped, but the message is late, or the sender concluded
   early, or the link that carried the message broke while the link that
   carried the pages did not. The Mini sits with everything it needs and does
   nothing. Waiting on the rows would have run the function at once.
4. **The wait binds this Mini to another Mini's control flow.** A message
   comes from a sender's scheduler, not from data. If that scheduler is slow,
   this Mini is slow, and every Mini downstream is slow, even though their
   pages have all arrived. On an infinite mesh there is always a slowest
   scheduler somewhere, and the chain of "waiting for whose message" has no
   end. Waiting on rows binds a Mini only to the pages in front of it.
5. **Work handed to the GPU while it waits for a message is killed or idles.**
   We measured this: a command buffer that waits on a signal for seconds is
   ended by the watchdog, and every buffer behind it dies with it. A GPU given
   only functions whose rows are stamped runs at its roofline fraction. Its
   utilization then does not depend on the mesh at all.
6. **Messages need their own storage and their own freeing.** Every message in
   flight is state: a queue entry, a counter, an event value, a callback. On an
   infinite mesh the number of messages in flight is not bounded, so this
   state is not bounded, and it is freed by other messages. Pages are freed by
   release when the dependent's rows are stamped. That state is bounded by the
   pages, and the pages were needed anyway.
7. **A speculative message lets a function run before its data.** "The pages
   will arrive" is not "the pages have arrived". A function that runs on the
   promise reads stale pages and writes a wrong output with a good stamp. The
   digest will disagree later, and the NFE will be repeated, but the work was
   wasted and the wrong output already travelled a hop.
8. **Fan-in multiplies all of the above.** A function whose inputs come from
   many Minis would need a message from each, each a hop, each a race with its
   pages. The rows give one check for all of them: are they all stamped k.

The rule that follows is the one already in the spec. A Mini acts on data,
and only on data, and acts the moment the data is present. The row's stamp is
the only signal. The GPU is handed only functions whose rows are stamped. The
calling context receives errors through asynchronous metadata, never through
numerical values or a status gate inside the callgraph. With that
rule a Mini's latency and utilization are the same whether it has one
neighbour or an infinite mesh behind it. Without it they are not.

## What this is called elsewhere

The firing principle is tagged-token dataflow; operand slots and presence
bits have hardware prior art in Monsoon. In-data flags, dependency-driven
execution and pipelined reductions also have published implementations.
The [citation audit](dataflow-implementation-audit.md) distinguishes those
mechanisms from the exact caller required here, which remains an implementation
obligation. It also records the divergent code and corrects the unmeasured
claim of strict performance superiority. The binding reduce description and
historical incident record are in [distributed-reduce.md](distributed-reduce.md).
