# The ABI is the boundary, and streams are the unit

Written after two design errors were caught in review. Both are recorded because both were
mine and both were defended before they were fixed.

## Error 1: the workload is statically linked into the transport

```make
mesh-flow: mesh-flow.c $(F) mesh-f.h
	cc -O2 -o $@ mesh-flow.c $(F) -lrdma
```

`make F=f-yourthing.c` does not produce a workload. It produces **a different transport**.
Every workload is therefore its own binary, which means its own TCP out-of-band connect, its
own `ibv_open_device`, its own protection domain, and its own queue pair. On a device with
`max_qp: 11` that is a hard ceiling of about ten concurrent workloads per node, and each one
is a distinct application identity to the host firewall.

That last consequence is not hypothetical. The OOB handshake failure debugged on this link
was the macOS application firewall filtering per-binary: it completes the TCP handshake for
an unapproved binary and drops it, so the connector sees success and the listener never sees
an accept. **That was this antipattern announcing itself, and it was diagnosed as a macOS
quirk instead of a design error.** An architecture where arbitrary applications open TCP to
arbitrary targets makes every new workload a new firewall negotiation. The correct
architecture has exactly one binary that ever touches TCP, once, at bootstrap.

## Error 2: spans were treated as ontologically real

```c
enum { SPAN_MAX = 4096 };
...
if(nspan>=SPAN_MAX){ ...; span_abort++; ... }
```

A span longer than 4096 pages is not rejected with an error. It is **silently discarded** and
counted in `span_abort`. A transport-layer buffer bound was promoted into a semantic limit on
what a datum is permitted to be. A stream that takes 23000 spans to transmit is not an edge
case to be supported later; it is the normal case, and the span is not a thing the workload
should ever have been shown.

Two adjacent bugs come from the same mistake. Line 329 aborts an entire span if a page arrives
out of order, and line 325 discards an open span when a new `F_FIRST` appears. Both are
reassembly logic that should not exist.

## The rule

**The number of copies per datum must not scale with N.** For a datum of content size N the
budget is one: the NIC DMA that puts the bytes in memory. Everything after that is a view.
Reassembling a large stream by concatenating spans violates this by construction, because the
concatenation cost is O(N) and buys nothing — the bytes were already in memory.

## The shape

```
app --DLPack view--> ABI region <--DMA--> wire <--DMA--> ABI region --DLPack view--> app
```

One region, mapped by both the daemon and the application, registered as the MR. The
application never names a peer, never opens a verbs device, never opens a socket.

Today `mesh-flow` registers its own private allocation:

```c
mrs[nmr]=ibv_reg_mr(g_pd,(char*)mem+off,len,IBV_ACCESS_LOCAL_WRITE);
```

`mem` is the daemon's, not the application's. That is the copy boundary that must not exist.
`ibv_reg_mr` must instead cover the `mesh_abi_create` mapping, so that a page arriving off the
wire lands in memory the application already has mapped.

## Streams, not spans

A `F_META` page announces `{stream_id, nbytes, npages}` before its payload. On receipt the
consumer reserves a contiguous extent of the region for that stream. Every subsequent page of
the stream is placed at `extent_base + seq * payload` by the page allocator.

Three properties follow, none of which the current code has:

- **Arrival order stops mattering.** A page is written where its sequence number says, so
  reorder is not a reassembly problem and needs no abort path.
- **Length stops mattering.** 23000 pages and 3 pages take the same path. There is no
  `SPAN_MAX`, because nothing accumulates a list of pages.
- **Completion is a count, not a concatenation.** The stream is ready when its page count is
  met. The datum was assembled by the NIC, in place, for free.

Extents will exceed one memory region. `mesh-mem` handles this: it chunks at a fixed 1 GiB
and maps an address to its region by ordered lookup, so an extent spans regions and the page
allocator respects region boundaries when placing. The 16.4 MB figure this section originally
cited is the advertised `max_mr_size`, which turned out to be neither enforced nor usable;
the real limit is 2^32.

## What `mesh_f` becomes

Per-page and per-span delivery both go away. The workload sees a stream open, and a stream
complete carrying one zero-copy tensor view:

```c
void mesh_stream_open(uint32_t stream, uint64_t nbytes, uint64_t npages, int node_idx);
void mesh_stream_ready(uint32_t stream, struct mtensor *t, int node_idx);
```

`struct mtensor` is DLPack-shaped: data pointer, ndim, shape, strides, dtype, device. The data
pointer is the extent base. No copy occurs between the wire and that pointer, and the
workload is a separate process that obtained it by attaching to the region, not by being
linked into the transport.
