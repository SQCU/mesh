# Pages and functions

Plain statements of the algorithm, in the operator's words: mesh, peers, pages,
page table, rows, stamps, use count, buffers as function returns, functions,
release, reduce, index page, NFE, metadata. Every other word that has appeared
in code ("stage", "chunk", "version", "gate", "token", "group", "consumer",
"scheduler", "header", "record", "claim") named a control-flow object standing
in for something the page table already holds, and is not part of the algorithm.

On the Thunderbolt substrate, pages travel by SEND/RECV as documented by Apple
TN3205: a receive lands in whichever posted buffer is next, the hardware holds a
send until the peer has posted a receive, and completion of a send means the
NIC has finished reading the source. Nothing below requires more than that.

## Backing memory and logical values

Operator clarification, September 10, 2026: a page is a memory backing and
indirection unit, not a semantic object for numerical functions. Mapping makes
literal byte extents available to views; it does not give those extents tensor
shape, validity or dataflow identity. A metadata struct can occupy any configured
extent within registered backing, just as operand data can. Headers are not
mandatory prefixes on every backing page. Numerical functions operate on values,
indices and strides; memory binding owns physical translation. Dataflow owns
logical presence and use counts. Transport completes registered memory accesses;
it neither interprets numerical stamps nor changes them on storage retirement.

## What the page table is

A row is memory. Nothing describing a page is stored in the page or beside it;
anything per page is a bit in a page-indexed bitmap inside the page table
struct. The table's operations are OR a bit, AND-compare a range against a
static mask, push and pop the free index. No other operation exists, no
per-page struct exists, and no state is created after configuration.

Concretely, one region of registered memory per node holds, in this order:

- the free index and the submission index: two fixed rings of page numbers;
- bit planes over logical rows: PRESENT, CONSTANT, FREED, and one READ plane
  per reader ordinal. A plane is one bit per row, sixty-four rows per word;
- `page[row]`: which backing page a logical row currently names;
- `mask[row]`: which READ planes must be set before the row's value is done;
- `base[binding]`: the first logical row of each receive binding;
- the pages.

The table gives every actor — bridge, client threads, GPU, peer — enough
information to know what it may touch. A page in the free index may be written
by the NIC. A row with PRESENT set and a READ bit clear holds a value some
reader still needs. A row whose READ bits cover its mask is finished. Nothing
in the table prevents a write; an actor that acts on stale information is a
configuration error, found by the numerical comparison, not a runtime
condition to be handled.

Every transition is an atomic OR, so it is idempotent: a duplicate sets a bit
that is already set and nothing happens. There is nothing to claim, count,
lock, retry, sweep, or acknowledge. A function is issued when its inputs' rows
are PRESENT (or CONSTANT) and unread by it, and its outputs' rows are either
never produced or read by every reader in their mask; issue clears the outputs'
bits. Completion ORs PRESENT on the outputs and the reader's bit on the inputs.
A landing block whose rows are all read is pushed on the free index by whichever
reader observed that first — decided by one OR whose old value it inspects.

## Blocks

Producers on this substrate write contiguous multi-page matrices, and a SEND
lands anonymously, so the identity of a transfer cannot ride in a per-page
prefix. The transfer unit is therefore a block: the pages of one packing group
of a transmitted value, followed by one page holding a tag (binding, index).
One block is one work request and one completion. The tag pages are written
once, at configuration; they are never read by a numerical function. On
arrival the bridge reads the tag, stores `page[]` for the block's rows and ORs
PRESENT. On send completion it ORs the NIC's READ bit. That is the whole
transport-to-table interface.

## Streaming tiles

A value exchanged with a peer is streamed in tiles. The **peer's streaming
tile** is the number of rows of that value one transfer block carries:
`tile = floor((block − 1)·pgsz / (columns·elembytes))`, rounded down to a
multiple of the granularity at which the peer's consuming function reads. It
is a property of the link geometry, the value's shape, and the peer's consumer
— never of the host's kernel call. A host call produces whole tiles: a call is
`call_tiles` consecutive tiles, and an ownership boundary is a call boundary.

A tile is contiguous memory: a block's pages are consecutive in the arena and
in a landing block. A function therefore addresses a tile by one base and the
rows inside it by offset. No per-element lookup exists.

**Function boundaries.** Configuration chooses the numerical backend's call
shapes and the row extents consumed and produced by each function. Those
boundaries describe actual arithmetic dependencies. A projection reads its
activation rows; it does not depend on other projections having completed.
Reduction reads the matching local and remote partials; it does not depend on
completion of every partial in the tensor. A completed output extent becomes
available to transport and arithmetic independently of unrelated output extents.

A command containing several dependent numerical operations exposes only its
final completion. When that hides an intermediate needed by independently
executable functions, the configured graph must expose that intermediate as an
output and give its consumers their own functions. This does not require changing
the selected numerical kernels or replacing their measured call shapes.

Timing models and utilization measurements belong to the calling context, not
to this specification. They do not authorize additional data dependencies.

## One NFE on two peers

1. A function runs on the GPU. It writes its output straight into pages. When
   it completes, its output rows get PRESENT. Nothing else is said.
2. Rows the peer needs go over the mesh as blocks. On the peer they land in
   free pages; the peer's rows for that output get `page[]` and PRESENT.
3. A reduce is a function. Its inputs are my partial rows and the peer's
   landed partial rows. When both are PRESENT and unread by it, it runs: it
   adds in high precision, normalizes, and writes the result into its output
   rows, which are then PRESENT. Its output rows go to the peer.
4. The next function reads my reduce output and the peer's. When all its
   input rows are PRESENT, it runs, reading the pages where they are. There
   is no copy.
5. The GPU is given a function only after its inputs are present, so it never
   waits.
6. When every reader in a row's mask has ORed its bit, the row is done. An
   arena row is rewritten by its producer's next issue. A landing block goes
   back to the free index, and the bridge posts it as the next receive.
7. Two NFEs can be in flight because their rows are different rows.
8. A link or device error is written to the port record or the function's
   metadata record. No function reads it. The calling context decides whether
   to rerun the whole NFE.

## The same NFE on infinitely many Mac Minis

Each Mini holds its share of the weights, one page table, and links to at most
three neighbours. A function's input rows all PRESENT means it runs. Output
rows a neighbour needs go one hop as blocks and land in that neighbour's rows.
Partials are combined as they pass, so the load on any link is bounded no
matter how many Minis lie beyond it. Latency is hops times hop time plus the
service time on the longest dependency chain; utilization is the roofline
fraction reached while inputs are present. Neither number changes with the
number of Minis. A presentation that changes them — a stage, a chunk, a gate, a
group, a copy into a hidden buffer, a wait on anything but bits — is not this
algorithm.

## Waiting for messages about data

Suppose a Mini waits for a message that says the data is coming, or is done,
or may be used — a token, a gate, a ready flag, a scheduler's decision, a
completion callback standing in for the pages — and acts on the message.

1. The message is a second thing on the link; every dependency now costs a hop
   for the pages and a hop for the message, and on a deep chain the extra wait
   is unbounded.
2. The message can be true and the data absent; making that safe needs
   acknowledgements, retries, and ordering — a protocol on top of delivery.
3. The message can be false and the data present; the Mini sits idle with
   everything it needs.
4. The wait binds this Mini to another's control flow; there is always a
   slowest scheduler somewhere.
5. Work handed to the GPU while it waits for a message is killed by the
   watchdog, and every buffer behind it dies with it (measured 2026-09-08).
6. Messages need their own storage and their own freeing, unbounded on an
   unbounded mesh; bits are bounded by the rows, which were needed anyway.
7. A speculative message lets a function run before its data and write a wrong
   output with a good bit.
8. Fan-in multiplies all of the above; the bits give one check for all inputs.

A Mini acts on data, and only on data, the moment it is present. The bit is the
only signal.

## Where this comes from

The firing principle is tagged-token dataflow; operand slots and presence bits
have hardware prior art in Papadopoulos and Culler's Monsoon (ISCA 1990), where
presence bits are a small structure beside the data words, not a header on
them. Reduce-scatter followed by all-gather is Rabenseifner (ICCS 2004) and
Patarasuk–Yuan (JPDC 2009). Numerical acceptance at the endpoints is Saltzer,
Reed and Clark (TOCS 1984). None of these certifies this implementation's
latency; that is what `transport-one-gib-2026-09-10.md` and the numerical
comparison in `metal-microbench` are for.
