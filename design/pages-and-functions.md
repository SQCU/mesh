# Pages and functions

The [user's requirements](collective-goals.md) define scope. This note describes
operand and lifetime relationships; it prescribes no executor or scheduling method.
The caller supplies the mesh and tensor placement. Mesh realizes their storage,
indexed dependencies and numerical function bindings before invocation.

Pages back values; they do not define tensor dimensions or numerical call extents.
Configured views name the actual registered storage. Presence denotes available
values, and reader ownership retains their storage through actual use. Transport
completion and numerical availability are different facts.

Functions consume available indexed input regions and publish their output regions
after the corresponding writes are visible. A publication does not finish or pause
the enclosing operation. Other producer work and consumers with available inputs
continue independently. Missing inputs constrain only the work that reads them.

Distinct simultaneously live values have distinct storage. Source storage remains
valid until numerical readers and the transport device finish accessing it. That
lifetime does not impose a dependency on independent production.

The substrate is Thunderbolt SEND/RECV, as specified by Apple TN3205. Mesh binds
send and receive endpoints to registered storage and uses the existing completion
mechanisms. Numerical callers do not implement transport or page-table operations.

Papadopoulos and Culler, *Monsoon: an Explicit Token-Store Architecture* (1990),
supplies operand-associated presence prior art, not a requirement for a function
scan. The JAX authors' Pallas collective matmul supplies indexed forwarding and
distinct live receive buffers. Sources are [references](algorithm-sources.md),
not additional requirements.

## What the page table is

A logical tensor section is an index range with a configured layout. Its page-table
entries name actual registered backing. A view alone does not establish contiguity:
setup decomposes the operand into the contiguous sections required by the chosen
numerical calls, preserving contraction contributions and output coordinates.

For section s, outstanding ownership is R(s) = P(s) + C(s) + T(s) + E(s): unfinished
production, configured numerical uses, transport uses, and external observations.
Setup derives known uses from the feed-forward graph. Existing completions and
automatic object lifetimes discharge them. Early publication does not release an
unfinished producer. A background collector returns zero-reference backing to the
writable pool without clearing payload bytes. Independent work already has its
configured sections and never awaits that collection.

The implementation trusts caller configuration and backend completion contracts.
Its ownership records describe actual accesses; they do not police arbitrary
external code. No caller free/done call or consumer-stamp protocol is required.

## Block addressing

Each contiguous value has one logical block head s and a block of B physical
pages beginning at page p(s). Byte offset d in that value has address
`arena + page_size * p(s) + d`. Shape and stride indexing stay in the numerical
operand binding. No page-table entry for an interior page is needed: the relative
offset determines its address. Value i uses the head `first + i * stride`;
a shared constant has stride zero. Ownership and presence also belong to that head.

Before publication, receive blocks form a bijection between unfilled logical
destinations and reserved physical blocks. If the next completion fills physical
block p for destination s, let d be p's current logical owner and q = p(s).
Assignment exchanges `(s,q), (d,p)` for `(s,p), (d,q)`. Two forward-table stores
and two inverse-table stores maintain that bijection. The identity case d = s
uses the same writes. Every physical receive slot appears once in the finite
posting list, so a previously published block cannot be displaced by a later
completion. No payload is copied and no published operand changes address.

Only after assignment and outgoing-tag preparation does `mesh_publish` publish
the head's presence bit with release ordering and enqueue its configured uses.
A numerical consumer resolves that head to the arrived physical block. Prepared
Metal/MPS and Core ML bindings select that same block. There is no loop over the
block's interior pages on the receive or publication path.

Each value has exactly one producer: setup for a constant, one numerical call,
or one receive. Its initial reference represents that write. `mesh_publish`
releases this reference once after publishing the declared uses. There is no
producer flag or duplicate-publication check. Numerical and transport references
were retained during realization and end through their own native completions;
external handles have ordinary automatic lifetimes. This preserves
`R = P + C + T + E` while removing a redundant atomic producer-flag operation.
The writable pool is still returned by the existing background collector.
